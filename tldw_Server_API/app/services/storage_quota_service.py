# storage_quota_service.py
# Description: User storage quota management service with async operations
#
# Imports
import os
import asyncio
from pathlib import Path
from typing import Dict, Optional, List, Any, Tuple
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta
#
# 3rd-party imports
from cachetools import TTLCache
from loguru import logger
#
# Local imports
from tldw_Server_API.app.core.AuthNZ.settings import Settings, get_settings
from tldw_Server_API.app.core.AuthNZ.database import DatabasePool, get_db_pool
from tldw_Server_API.app.core.AuthNZ.exceptions import (
    StorageError,
    QuotaExceededError,
    UserNotFoundError
)

#######################################################################################################################
#
# Storage Quota Service Class

class StorageQuotaService:
    """Service for managing user storage quotas and tracking usage"""
    
    def __init__(
        self,
        db_pool: Optional[DatabasePool] = None,
        settings: Optional[Settings] = None
    ):
        """Initialize storage quota service"""
        self.settings = settings or get_settings()
        self.db_pool = db_pool
        
        # Thread pool for file system operations
        self.executor = ThreadPoolExecutor(max_workers=4)
        
        # TTL cache for quota checks (5 minutes)
        self.quota_cache = TTLCache(maxsize=1000, ttl=300)
        
        # TTL cache for storage calculations (10 minutes)
        self.storage_cache = TTLCache(maxsize=1000, ttl=600)
        
        self._initialized = False
        
    async def initialize(self):
        """Initialize the storage quota service"""
        if self._initialized:
            return
        
        if not self.db_pool:
            self.db_pool = await get_db_pool()
        
        self._initialized = True
        logger.info("StorageQuotaService initialized")
    
    async def check_quota(
        self,
        user_id: int,
        new_bytes: int,
        raise_on_exceed: bool = False
    ) -> Tuple[bool, Dict[str, Any]]:
        """
        Check if user has quota for new content
        
        Args:
            user_id: User's database ID
            new_bytes: Size of new content in bytes
            raise_on_exceed: Raise exception if quota exceeded
            
        Returns:
            Tuple of (has_quota, quota_info)
            
        Raises:
            QuotaExceededError: If quota exceeded and raise_on_exceed is True
        """
        if not self._initialized:
            await self.initialize()
        
        # Check cache first
        cache_key = f"quota:{user_id}"
        if cache_key in self.quota_cache:
            current_mb, quota_mb = self.quota_cache[cache_key]
        else:
            # Fetch from database
            user_info = await self._get_user_storage_info(user_id)
            if not user_info:
                raise UserNotFoundError(f"User {user_id}")
            
            current_mb = float(user_info['storage_used_mb'])
            quota_mb = user_info['storage_quota_mb']
            
            # Update cache
            self.quota_cache[cache_key] = (current_mb, quota_mb)
        
        # Calculate new usage
        new_mb = new_bytes / (1024 * 1024)
        projected_mb = current_mb + new_mb
        has_quota = projected_mb <= quota_mb
        
        quota_info = {
            "user_id": user_id,
            "current_usage_mb": round(current_mb, 2),
            "quota_mb": quota_mb,
            "new_size_mb": round(new_mb, 2),
            "projected_usage_mb": round(projected_mb, 2),
            "available_mb": round(max(0, quota_mb - current_mb), 2),
            "usage_percentage": round((current_mb / quota_mb * 100) if quota_mb > 0 else 0, 1),
            "has_quota": has_quota
        }
        
        if not has_quota and raise_on_exceed:
            raise QuotaExceededError(projected_mb, quota_mb)
        
        return has_quota, quota_info
    
    async def update_usage(
        self,
        user_id: int,
        bytes_delta: int,
        operation: str = "add"
    ) -> Dict[str, Any]:
        """
        Update storage usage for a user
        
        Args:
            user_id: User's database ID
            bytes_delta: Bytes to add or remove
            operation: 'add' or 'remove'
            
        Returns:
            Updated storage information
        """
        if not self._initialized:
            await self.initialize()
        
        mb_delta = bytes_delta / (1024 * 1024)
        if operation == "remove":
            mb_delta = -mb_delta
        
        try:
            async with self.db_pool.transaction() as conn:
                if hasattr(conn, 'fetchrow'):
                    # PostgreSQL
                    result = await conn.fetchrow(
                        """
                        UPDATE users 
                        SET storage_used_mb = GREATEST(0, storage_used_mb + $1)
                        WHERE id = $2
                        RETURNING storage_used_mb, storage_quota_mb
                        """,
                        mb_delta, user_id
                    )
                    
                    if not result:
                        raise UserNotFoundError(f"User {user_id}")
                    
                    new_usage = float(result['storage_used_mb'])
                    quota = result['storage_quota_mb']
                    
                else:
                    # SQLite
                    await conn.execute(
                        """
                        UPDATE users 
                        SET storage_used_mb = MAX(0, storage_used_mb + ?)
                        WHERE id = ?
                        """,
                        (mb_delta, user_id)
                    )
                    
                    cursor = await conn.execute(
                        "SELECT storage_used_mb, storage_quota_mb FROM users WHERE id = ?",
                        (user_id,)
                    )
                    result = await cursor.fetchone()
                    
                    if not result:
                        raise UserNotFoundError(f"User {user_id}")
                    
                    new_usage = float(result[0])
                    quota = result[1]
                    
                    await conn.commit()
                
                # Check if over quota
                if new_usage > quota:
                    logger.warning(
                        f"User {user_id} exceeded quota: {new_usage:.2f}MB / {quota}MB"
                    )
                
                # Invalidate cache
                cache_key = f"quota:{user_id}"
                self.quota_cache.pop(cache_key, None)
                
                # Log significant changes
                if abs(mb_delta) > 10:  # Log changes over 10MB
                    logger.info(
                        f"Updated storage for user {user_id}: "
                        f"{operation} {abs(mb_delta):.2f}MB "
                        f"(new total: {new_usage:.2f}MB / {quota}MB)"
                    )
                
                return {
                    "user_id": user_id,
                    "storage_used_mb": round(new_usage, 2),
                    "storage_quota_mb": quota,
                    "available_mb": round(max(0, quota - new_usage), 2),
                    "usage_percentage": round((new_usage / quota * 100) if quota > 0 else 0, 1)
                }
                
        except Exception as e:
            logger.error(f"Failed to update storage usage: {e}")
            raise StorageError(f"Failed to update storage usage: {e}")
    
    async def calculate_user_storage(
        self,
        user_id: int,
        update_database: bool = True
    ) -> Dict[str, Any]:
        """
        Calculate actual storage usage for a user
        
        Args:
            user_id: User's database ID
            update_database: Update database with calculated value
            
        Returns:
            Storage calculation results
        """
        if not self._initialized:
            await self.initialize()
        
        # Check cache first
        cache_key = f"storage_calc:{user_id}"
        if cache_key in self.storage_cache and not update_database:
            return self.storage_cache[cache_key]
        
        # Calculate storage in thread pool
        user_dir = Path(self.settings.USER_DATA_BASE_PATH) / str(user_id)
        
        loop = asyncio.get_event_loop()
        size_bytes = await loop.run_in_executor(
            self.executor,
            self._calculate_directory_size,
            str(user_dir)
        )
        
        # Also calculate ChromaDB if configured
        chroma_bytes = 0
        if self.settings.CHROMADB_BASE_PATH:
            chroma_dir = Path(self.settings.CHROMADB_BASE_PATH) / str(user_id)
            chroma_bytes = await loop.run_in_executor(
                self.executor,
                self._calculate_directory_size,
                str(chroma_dir)
            )
        
        total_bytes = size_bytes + chroma_bytes
        total_mb = total_bytes / (1024 * 1024)
        
        # Get quota from database
        user_info = await self._get_user_storage_info(user_id)
        if not user_info:
            raise UserNotFoundError(f"User {user_id}")
        
        quota_mb = user_info['storage_quota_mb']
        
        # Update database if requested
        if update_database:
            async with self.db_pool.transaction() as conn:
                if hasattr(conn, 'execute'):
                    # PostgreSQL
                    await conn.execute(
                        "UPDATE users SET storage_used_mb = $1 WHERE id = $2",
                        total_mb, user_id
                    )
                else:
                    # SQLite
                    await conn.execute(
                        "UPDATE users SET storage_used_mb = ? WHERE id = ?",
                        (total_mb, user_id)
                    )
                    await conn.commit()
            
            # Invalidate quota cache
            self.quota_cache.pop(f"quota:{user_id}", None)
            
            logger.info(
                f"Recalculated storage for user {user_id}: "
                f"{total_mb:.2f}MB / {quota_mb}MB"
            )
        
        result = {
            "user_id": user_id,
            "user_data_bytes": size_bytes,
            "user_data_mb": round(size_bytes / (1024 * 1024), 2),
            "chromadb_bytes": chroma_bytes,
            "chromadb_mb": round(chroma_bytes / (1024 * 1024), 2),
            "total_bytes": total_bytes,
            "total_mb": round(total_mb, 2),
            "quota_mb": quota_mb,
            "available_mb": round(max(0, quota_mb - total_mb), 2),
            "usage_percentage": round((total_mb / quota_mb * 100) if quota_mb > 0 else 0, 1),
            "calculated_at": datetime.utcnow().isoformat()
        }
        
        # Update cache
        self.storage_cache[cache_key] = result
        
        return result
    
    def _calculate_directory_size(self, path: str) -> int:
        """
        Calculate directory size (runs in thread pool)
        
        Args:
            path: Directory path
            
        Returns:
            Size in bytes
        """
        total = 0
        path_obj = Path(path)
        
        if not path_obj.exists():
            return 0
        
        try:
            for entry in path_obj.rglob('*'):
                if entry.is_file():
                    try:
                        total += entry.stat().st_size
                    except (OSError, PermissionError):
                        continue
        except (OSError, PermissionError) as e:
            logger.error(f"Error calculating size for {path}: {e}")
        
        return total
    
    async def get_storage_breakdown(self, user_id: int) -> Dict[str, Any]:
        """
        Get detailed storage breakdown by file type
        
        Args:
            user_id: User's database ID
            
        Returns:
            Detailed storage breakdown
        """
        if not self._initialized:
            await self.initialize()
        
        user_dir = Path(self.settings.USER_DATA_BASE_PATH) / str(user_id)
        
        # Calculate breakdown in thread pool
        loop = asyncio.get_event_loop()
        breakdown = await loop.run_in_executor(
            self.executor,
            self._calculate_storage_breakdown,
            str(user_dir)
        )
        
        # Get quota info
        user_info = await self._get_user_storage_info(user_id)
        if not user_info:
            raise UserNotFoundError(f"User {user_id}")
        
        breakdown.update({
            "user_id": user_id,
            "quota_mb": user_info['storage_quota_mb'],
            "current_usage_mb": float(user_info['storage_used_mb'])
        })
        
        return breakdown
    
    def _calculate_storage_breakdown(self, path: str) -> Dict[str, Any]:
        """
        Calculate storage breakdown by file type
        
        Args:
            path: Directory path
            
        Returns:
            Storage breakdown
        """
        breakdown = {
            "media": {"count": 0, "bytes": 0},
            "notes": {"count": 0, "bytes": 0},
            "embeddings": {"count": 0, "bytes": 0},
            "exports": {"count": 0, "bytes": 0},
            "temp": {"count": 0, "bytes": 0},
            "other": {"count": 0, "bytes": 0}
        }
        
        path_obj = Path(path)
        if not path_obj.exists():
            return breakdown
        
        try:
            for entry in path_obj.rglob('*'):
                if entry.is_file():
                    try:
                        size = entry.stat().st_size
                        
                        # Determine category based on path
                        relative_path = entry.relative_to(path_obj)
                        parts = relative_path.parts
                        
                        if parts and parts[0] in breakdown:
                            category = parts[0]
                        else:
                            category = "other"
                        
                        breakdown[category]["count"] += 1
                        breakdown[category]["bytes"] += size
                        
                    except (OSError, PermissionError):
                        continue
                        
        except (OSError, PermissionError) as e:
            logger.error(f"Error calculating breakdown for {path}: {e}")
        
        # Convert bytes to MB
        for category in breakdown:
            breakdown[category]["mb"] = round(
                breakdown[category]["bytes"] / (1024 * 1024), 2
            )
        
        return breakdown
    
    async def set_user_quota(
        self,
        user_id: int,
        quota_mb: int
    ) -> Dict[str, Any]:
        """
        Set storage quota for a user
        
        Args:
            user_id: User's database ID
            quota_mb: New quota in megabytes
            
        Returns:
            Updated quota information
        """
        if not self._initialized:
            await self.initialize()
        
        if quota_mb < 100:  # Minimum 100MB
            quota_mb = 100
        
        try:
            async with self.db_pool.transaction() as conn:
                if hasattr(conn, 'fetchrow'):
                    # PostgreSQL
                    result = await conn.fetchrow(
                        """
                        UPDATE users 
                        SET storage_quota_mb = $1
                        WHERE id = $2
                        RETURNING storage_used_mb, storage_quota_mb
                        """,
                        quota_mb, user_id
                    )
                    
                    if not result:
                        raise UserNotFoundError(f"User {user_id}")
                    
                    current_usage = float(result['storage_used_mb'])
                    
                else:
                    # SQLite
                    await conn.execute(
                        "UPDATE users SET storage_quota_mb = ? WHERE id = ?",
                        (quota_mb, user_id)
                    )
                    
                    cursor = await conn.execute(
                        "SELECT storage_used_mb FROM users WHERE id = ?",
                        (user_id,)
                    )
                    result = await cursor.fetchone()
                    
                    if not result:
                        raise UserNotFoundError(f"User {user_id}")
                    
                    current_usage = float(result[0])
                    await conn.commit()
                
                # Invalidate cache
                self.quota_cache.pop(f"quota:{user_id}", None)
                
                logger.info(f"Set quota for user {user_id}: {quota_mb}MB")
                
                return {
                    "user_id": user_id,
                    "storage_quota_mb": quota_mb,
                    "storage_used_mb": round(current_usage, 2),
                    "available_mb": round(max(0, quota_mb - current_usage), 2),
                    "usage_percentage": round((current_usage / quota_mb * 100) if quota_mb > 0 else 0, 1)
                }
                
        except Exception as e:
            logger.error(f"Failed to set user quota: {e}")
            raise StorageError(f"Failed to set quota: {e}")
    
    async def get_all_users_storage(self) -> List[Dict[str, Any]]:
        """
        Get storage information for all users
        
        Returns:
            List of user storage information
        """
        if not self._initialized:
            await self.initialize()
        
        users = await self.db_pool.fetchall(
            """
            SELECT id, username, storage_used_mb, storage_quota_mb
            FROM users
            WHERE is_active = ?
            ORDER BY storage_used_mb DESC
            """,
            True
        )
        
        result = []
        for user in users:
            used = float(user['storage_used_mb'])
            quota = user['storage_quota_mb']
            
            result.append({
                "user_id": user['id'],
                "username": user['username'],
                "storage_used_mb": round(used, 2),
                "storage_quota_mb": quota,
                "available_mb": round(max(0, quota - used), 2),
                "usage_percentage": round((used / quota * 100) if quota > 0 else 0, 1)
            })
        
        return result
    
    async def cleanup_temp_files(
        self,
        user_id: Optional[int] = None,
        older_than_hours: int = 24
    ) -> Dict[str, Any]:
        """
        Clean up temporary files older than specified time
        
        Args:
            user_id: Specific user or all if None
            older_than_hours: Age threshold in hours
            
        Returns:
            Cleanup statistics
        """
        if not self._initialized:
            await self.initialize()
        
        cutoff_time = datetime.now() - timedelta(hours=older_than_hours)
        
        # Determine directories to clean
        if user_id:
            temp_dirs = [Path(self.settings.USER_DATA_BASE_PATH) / str(user_id) / "temp"]
        else:
            base_path = Path(self.settings.USER_DATA_BASE_PATH)
            temp_dirs = list(base_path.glob("*/temp"))
        
        # Clean in thread pool
        loop = asyncio.get_event_loop()
        stats = await loop.run_in_executor(
            self.executor,
            self._cleanup_temp_directories,
            temp_dirs,
            cutoff_time
        )
        
        if stats['files_deleted'] > 0:
            logger.info(
                f"Cleaned up {stats['files_deleted']} temp files "
                f"({stats['bytes_freed'] / (1024*1024):.2f}MB)"
            )
        
        return stats
    
    def _cleanup_temp_directories(
        self,
        temp_dirs: List[Path],
        cutoff_time: datetime
    ) -> Dict[str, Any]:
        """
        Clean up temporary directories
        
        Args:
            temp_dirs: List of temp directories
            cutoff_time: Delete files older than this
            
        Returns:
            Cleanup statistics
        """
        stats = {
            "files_deleted": 0,
            "bytes_freed": 0,
            "errors": 0
        }
        
        for temp_dir in temp_dirs:
            if not temp_dir.exists():
                continue
            
            try:
                for file_path in temp_dir.rglob('*'):
                    if file_path.is_file():
                        try:
                            # Check file age
                            mtime = datetime.fromtimestamp(file_path.stat().st_mtime)
                            if mtime < cutoff_time:
                                size = file_path.stat().st_size
                                file_path.unlink()
                                stats['files_deleted'] += 1
                                stats['bytes_freed'] += size
                        except (OSError, PermissionError):
                            stats['errors'] += 1
                            continue
                            
            except (OSError, PermissionError):
                stats['errors'] += 1
                continue
        
        return stats
    
    async def _get_user_storage_info(self, user_id: int) -> Optional[Dict[str, Any]]:
        """Get user storage information from database"""
        return await self.db_pool.fetchone(
            "SELECT storage_used_mb, storage_quota_mb FROM users WHERE id = ?",
            user_id
        )
    
    async def shutdown(self):
        """Shutdown the storage quota service"""
        self.executor.shutdown(wait=False)
        logger.info("StorageQuotaService shutdown complete")


#######################################################################################################################
#
# Module Functions

# Global instance
_storage_service: Optional[StorageQuotaService] = None


async def get_storage_service() -> StorageQuotaService:
    """Get storage service singleton"""
    global _storage_service
    if not _storage_service:
        _storage_service = StorageQuotaService()
        await _storage_service.initialize()
    return _storage_service


#
# End of storage_quota_service.py
#######################################################################################################################