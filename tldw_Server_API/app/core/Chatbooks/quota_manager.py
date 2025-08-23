# quota_manager.py
# Description: User quota management for chatbook operations
#
"""
Quota Manager for Chatbook Operations
--------------------------------------

Manages user quotas for storage, export/import operations, and rate limits.
"""

import asyncio
from datetime import datetime, timedelta
from typing import Dict, Optional, Tuple
from pathlib import Path
from loguru import logger


class QuotaManager:
    """Manages user quotas and usage limits."""
    
    # Default quotas (can be overridden per user tier)
    DEFAULT_QUOTAS = {
        'max_storage_mb': 1000,  # 1GB default storage
        'max_exports_per_day': 10,
        'max_imports_per_day': 10,
        'max_file_size_mb': 100,
        'max_concurrent_jobs': 2,
        'max_chatbooks': 50
    }
    
    # Premium user quotas
    PREMIUM_QUOTAS = {
        'max_storage_mb': 5000,  # 5GB for premium users
        'max_exports_per_day': 50,
        'max_imports_per_day': 50,
        'max_file_size_mb': 500,
        'max_concurrent_jobs': 5,
        'max_chatbooks': 200
    }
    
    def __init__(self, user_id: str, user_tier: str = 'free'):
        """
        Initialize quota manager for a user.
        
        Args:
            user_id: User identifier
            user_tier: User tier (free, premium, enterprise)
        """
        self.user_id = user_id
        self.user_tier = user_tier
        self.quotas = self._get_quotas_for_tier(user_tier)
        
        # Usage tracking (in production, use database)
        self.usage_cache: Dict[str, any] = {}
    
    def _get_quotas_for_tier(self, tier: str) -> Dict[str, int]:
        """Get quota limits based on user tier."""
        if tier == 'premium':
            return self.PREMIUM_QUOTAS.copy()
        elif tier == 'enterprise':
            # Enterprise users get unlimited quotas
            return {
                'max_storage_mb': float('inf'),
                'max_exports_per_day': float('inf'),
                'max_imports_per_day': float('inf'),
                'max_file_size_mb': 1000,
                'max_concurrent_jobs': 10,
                'max_chatbooks': float('inf')
            }
        else:
            return self.DEFAULT_QUOTAS.copy()
    
    async def check_storage_quota(self, additional_bytes: int = 0) -> Tuple[bool, str]:
        """
        Check if user has enough storage quota.
        
        Args:
            additional_bytes: Additional bytes to be added
            
        Returns:
            Tuple of (allowed, message)
        """
        current_usage = await self._get_current_storage_usage()
        max_bytes = self.quotas['max_storage_mb'] * 1024 * 1024
        
        if current_usage + additional_bytes > max_bytes:
            remaining_mb = (max_bytes - current_usage) / (1024 * 1024)
            return False, f"Storage quota exceeded. You have {remaining_mb:.1f}MB remaining."
        
        return True, "Storage quota OK"
    
    async def check_export_quota(self) -> Tuple[bool, str]:
        """
        Check if user can perform another export today.
        
        Returns:
            Tuple of (allowed, message)
        """
        exports_today = await self._get_operations_count_today('export')
        
        if exports_today >= self.quotas['max_exports_per_day']:
            return False, f"Daily export limit ({self.quotas['max_exports_per_day']}) reached. Try again tomorrow."
        
        return True, "Export quota OK"
    
    async def check_import_quota(self) -> Tuple[bool, str]:
        """
        Check if user can perform another import today.
        
        Returns:
            Tuple of (allowed, message)
        """
        imports_today = await self._get_operations_count_today('import')
        
        if imports_today >= self.quotas['max_imports_per_day']:
            return False, f"Daily import limit ({self.quotas['max_imports_per_day']}) reached. Try again tomorrow."
        
        return True, "Import quota OK"
    
    async def check_file_size(self, file_size_bytes: int) -> Tuple[bool, str]:
        """
        Check if file size is within limits.
        
        Args:
            file_size_bytes: File size in bytes
            
        Returns:
            Tuple of (allowed, message)
        """
        max_bytes = self.quotas['max_file_size_mb'] * 1024 * 1024
        
        if file_size_bytes > max_bytes:
            return False, f"File too large. Maximum size is {self.quotas['max_file_size_mb']}MB"
        
        return True, "File size OK"
    
    async def check_concurrent_jobs(self) -> Tuple[bool, str]:
        """
        Check if user can start another concurrent job.
        
        Returns:
            Tuple of (allowed, message)
        """
        active_jobs = await self._get_active_jobs_count()
        
        if active_jobs >= self.quotas['max_concurrent_jobs']:
            return False, f"Maximum concurrent jobs ({self.quotas['max_concurrent_jobs']}) reached. Wait for current jobs to complete."
        
        return True, "Concurrent jobs OK"
    
    async def record_operation(self, operation_type: str, size_bytes: int = 0):
        """
        Record an operation for quota tracking.
        
        Args:
            operation_type: Type of operation (export, import)
            size_bytes: Size of the operation in bytes
        """
        # In production, this would update database
        logger.info(f"Recording {operation_type} operation for user {self.user_id} ({size_bytes} bytes)")
    
    async def get_usage_summary(self) -> Dict[str, any]:
        """
        Get current usage summary for the user.
        
        Returns:
            Dictionary with usage statistics
        """
        storage_used = await self._get_current_storage_usage()
        exports_today = await self._get_operations_count_today('export')
        imports_today = await self._get_operations_count_today('import')
        active_jobs = await self._get_active_jobs_count()
        
        return {
            'storage': {
                'used_mb': storage_used / (1024 * 1024),
                'limit_mb': self.quotas['max_storage_mb'],
                'percentage': (storage_used / (self.quotas['max_storage_mb'] * 1024 * 1024)) * 100
            },
            'exports': {
                'today': exports_today,
                'limit': self.quotas['max_exports_per_day']
            },
            'imports': {
                'today': imports_today,
                'limit': self.quotas['max_imports_per_day']
            },
            'jobs': {
                'active': active_jobs,
                'limit': self.quotas['max_concurrent_jobs']
            },
            'tier': self.user_tier
        }
    
    async def _get_current_storage_usage(self) -> int:
        """Get current storage usage in bytes."""
        # In production, query database for actual usage
        # For now, calculate from user's data directory
        try:
            import os
            import tempfile
            
            # Use environment variable, or temp dir for testing, or system default
            if os.environ.get('TLDW_USER_DATA_PATH'):
                base_dir = Path(os.environ.get('TLDW_USER_DATA_PATH'))
            elif os.environ.get('PYTEST_CURRENT_TEST') or os.environ.get('CI'):
                base_dir = Path(tempfile.gettempdir()) / 'tldw_test_data'
            else:
                base_dir = Path('/var/lib/tldw/user_data')
            
            user_dir = base_dir / 'users' / str(self.user_id)
            
            if user_dir.exists():
                total_size = 0
                for path in user_dir.rglob('*'):
                    if path.is_file():
                        total_size += path.stat().st_size
                return total_size
            return 0
        except Exception as e:
            logger.error(f"Error calculating storage usage: {e}")
            return 0
    
    async def _get_operations_count_today(self, operation_type: str) -> int:
        """Get count of operations performed today."""
        # In production, query database
        # For now, return cached value or 0
        cache_key = f"{operation_type}_count_{datetime.now().date()}"
        return self.usage_cache.get(cache_key, 0)
    
    async def _get_active_jobs_count(self) -> int:
        """Get count of currently active jobs."""
        # In production, query database for jobs with status IN ('pending', 'in_progress')
        return self.usage_cache.get('active_jobs', 0)
    
    def cleanup_expired_files(self, days_old: int = 7) -> int:
        """
        Clean up old export files to free up space.
        
        Args:
            days_old: Delete files older than this many days
            
        Returns:
            Number of files deleted
        """
        try:
            import os
            import tempfile
            
            # Use environment variable, or temp dir for testing, or system default
            if os.environ.get('TLDW_USER_DATA_PATH'):
                base_dir = Path(os.environ.get('TLDW_USER_DATA_PATH'))
            elif os.environ.get('PYTEST_CURRENT_TEST') or os.environ.get('CI'):
                base_dir = Path(tempfile.gettempdir()) / 'tldw_test_data'
            else:
                base_dir = Path('/var/lib/tldw/user_data')
            
            user_export_dir = base_dir / 'users' / str(self.user_id) / 'chatbooks' / 'exports'
            
            if not user_export_dir.exists():
                return 0
            
            cutoff_date = datetime.now() - timedelta(days=days_old)
            deleted_count = 0
            
            for file_path in user_export_dir.glob('*.zip'):
                if file_path.stat().st_mtime < cutoff_date.timestamp():
                    try:
                        file_path.unlink()
                        deleted_count += 1
                        logger.info(f"Deleted old export: {file_path}")
                    except Exception as e:
                        logger.error(f"Failed to delete {file_path}: {e}")
            
            return deleted_count
            
        except Exception as e:
            logger.error(f"Error cleaning up old files: {e}")
            return 0


# Dependency injection helper
async def get_quota_manager(user_id: str, user_tier: str = 'free') -> QuotaManager:
    """Get quota manager instance for a user."""
    return QuotaManager(user_id, user_tier)