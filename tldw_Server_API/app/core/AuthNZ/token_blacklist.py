# token_blacklist.py
# Description: Token blacklist service for JWT revocation and invalidation
#
# Imports
from datetime import datetime, timedelta
from typing import Optional, Dict, Any, List, Set
import json
import asyncio
#
# 3rd-party imports
import redis
from redis.exceptions import RedisError
from loguru import logger
#
# Local imports
from tldw_Server_API.app.core.AuthNZ.settings import Settings, get_settings
from tldw_Server_API.app.core.AuthNZ.database import DatabasePool, get_db_pool
from tldw_Server_API.app.core.AuthNZ.exceptions import DatabaseError, InvalidTokenError

#######################################################################################################################
#
# Token Blacklist Service
#

class TokenBlacklist:
    """
    Service for managing revoked/blacklisted JWT tokens.
    
    Supports both Redis (for performance) and database (for persistence) storage.
    Automatically cleans up expired tokens to prevent unbounded growth.
    """
    
    def __init__(
        self,
        db_pool: Optional[DatabasePool] = None,
        settings: Optional[Settings] = None
    ):
        """Initialize token blacklist service"""
        self.settings = settings or get_settings()
        self.db_pool = db_pool
        self.redis_client: Optional[redis.Redis] = None
        self._initialized = False
        
        # Cache for recently checked tokens (in-memory)
        self._local_cache: Set[str] = set()
        self._cache_size_limit = 1000
        
    async def initialize(self):
        """Initialize blacklist service and create tables if needed"""
        if self._initialized:
            return
        
        # Get database pool
        if not self.db_pool:
            self.db_pool = await get_db_pool()
        
        # Create blacklist table if it doesn't exist
        await self._create_tables()
        
        # Initialize Redis if configured
        if self.settings.REDIS_URL:
            try:
                self.redis_client = redis.from_url(
                    self.settings.REDIS_URL,
                    decode_responses=True,
                    socket_connect_timeout=1
                )
                self.redis_client.ping()
                logger.debug("Redis connected for token blacklist")
            except (RedisError, Exception) as e:
                logger.warning(f"Redis unavailable for token blacklist: {e}")
                self.redis_client = None
        
        self._initialized = True
        logger.info("TokenBlacklist service initialized")
    
    async def _create_tables(self):
        """Create token blacklist table if it doesn't exist"""
        try:
            async with self.db_pool.transaction() as conn:
                if hasattr(conn, 'execute'):
                    # PostgreSQL
                    await conn.execute("""
                        CREATE TABLE IF NOT EXISTS token_blacklist (
                            id SERIAL PRIMARY KEY,
                            jti VARCHAR(255) UNIQUE NOT NULL,
                            user_id INTEGER,
                            token_type VARCHAR(50),
                            revoked_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                            expires_at TIMESTAMP NOT NULL,
                            reason VARCHAR(255),
                            revoked_by INTEGER,
                            ip_address VARCHAR(45),
                            FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
                        )
                    """)
                    
                    # Create indexes
                    await conn.execute(
                        "CREATE INDEX IF NOT EXISTS idx_blacklist_jti ON token_blacklist(jti)"
                    )
                    await conn.execute(
                        "CREATE INDEX IF NOT EXISTS idx_blacklist_expires ON token_blacklist(expires_at)"
                    )
                    await conn.execute(
                        "CREATE INDEX IF NOT EXISTS idx_blacklist_user ON token_blacklist(user_id)"
                    )
                    
                else:
                    # SQLite
                    await conn.execute("""
                        CREATE TABLE IF NOT EXISTS token_blacklist (
                            id INTEGER PRIMARY KEY AUTOINCREMENT,
                            jti TEXT UNIQUE NOT NULL,
                            user_id INTEGER,
                            token_type TEXT,
                            revoked_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                            expires_at TIMESTAMP NOT NULL,
                            reason TEXT,
                            revoked_by INTEGER,
                            ip_address TEXT,
                            FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
                        )
                    """)
                    
                    # Create indexes
                    await conn.execute(
                        "CREATE INDEX IF NOT EXISTS idx_blacklist_jti ON token_blacklist(jti)"
                    )
                    await conn.execute(
                        "CREATE INDEX IF NOT EXISTS idx_blacklist_expires ON token_blacklist(expires_at)"
                    )
                    await conn.execute(
                        "CREATE INDEX IF NOT EXISTS idx_blacklist_user ON token_blacklist(user_id)"
                    )
                    
                    await conn.commit()
                    
        except Exception as e:
            logger.error(f"Failed to create token blacklist table: {e}")
            raise DatabaseError(f"Failed to create blacklist table: {e}")
    
    async def revoke_token(
        self,
        jti: str,
        expires_at: datetime,
        user_id: Optional[int] = None,
        token_type: str = "access",
        reason: Optional[str] = None,
        revoked_by: Optional[int] = None,
        ip_address: Optional[str] = None
    ) -> bool:
        """
        Add a token to the blacklist
        
        Args:
            jti: JWT ID (unique identifier for the token)
            expires_at: Token expiration time
            user_id: User who owns the token
            token_type: Type of token (access, refresh, etc.)
            reason: Reason for revocation
            revoked_by: User who revoked the token (for admin actions)
            ip_address: IP address of revocation request
            
        Returns:
            True if successfully blacklisted
        """
        if not self._initialized:
            await self.initialize()
        
        # Add to local cache
        self._local_cache.add(jti)
        if len(self._local_cache) > self._cache_size_limit:
            # Remove oldest entries if cache is too large
            self._local_cache = set(list(self._local_cache)[-self._cache_size_limit:])
        
        # Add to Redis if available
        if self.redis_client:
            try:
                key = f"blacklist:{jti}"
                ttl = int((expires_at - datetime.utcnow()).total_seconds())
                
                if ttl > 0:
                    self.redis_client.setex(
                        key,
                        ttl,
                        json.dumps({
                            "user_id": user_id,
                            "token_type": token_type,
                            "reason": reason,
                            "revoked_at": datetime.utcnow().isoformat()
                        })
                    )
                    logger.debug(f"Token {jti} added to Redis blacklist")
                    
            except (RedisError, Exception) as e:
                logger.warning(f"Failed to add token to Redis blacklist: {e}")
        
        # Add to database for persistence
        try:
            async with self.db_pool.transaction() as conn:
                if hasattr(conn, 'execute'):
                    # PostgreSQL
                    await conn.execute("""
                        INSERT INTO token_blacklist 
                        (jti, user_id, token_type, expires_at, reason, revoked_by, ip_address)
                        VALUES ($1, $2, $3, $4, $5, $6, $7)
                        ON CONFLICT (jti) DO NOTHING
                    """, jti, user_id, token_type, expires_at, reason, revoked_by, ip_address)
                else:
                    # SQLite
                    await conn.execute("""
                        INSERT OR IGNORE INTO token_blacklist 
                        (jti, user_id, token_type, expires_at, reason, revoked_by, ip_address)
                        VALUES (?, ?, ?, ?, ?, ?, ?)
                    """, (jti, user_id, token_type, expires_at.isoformat(), reason, revoked_by, ip_address))
                    await conn.commit()
            
            logger.info(f"Token {jti} blacklisted for user {user_id} - Reason: {reason}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to blacklist token: {e}")
            return False
    
    async def is_blacklisted(self, jti: str) -> bool:
        """
        Check if a token is blacklisted
        
        Args:
            jti: JWT ID to check
            
        Returns:
            True if token is blacklisted
        """
        if not jti:
            return False
        
        # Check local cache first (fastest)
        if jti in self._local_cache:
            return True
        
        if not self._initialized:
            await self.initialize()
        
        # Check Redis if available
        if self.redis_client:
            try:
                key = f"blacklist:{jti}"
                if self.redis_client.exists(key):
                    # Add to local cache for next time
                    self._local_cache.add(jti)
                    return True
            except (RedisError, Exception) as e:
                logger.warning(f"Redis error checking blacklist: {e}")
        
        # Check database
        try:
            async with self.db_pool.acquire() as conn:
                if hasattr(conn, 'fetchval'):
                    # PostgreSQL
                    exists = await conn.fetchval(
                        "SELECT EXISTS(SELECT 1 FROM token_blacklist WHERE jti = $1 AND expires_at > $2)",
                        jti, datetime.utcnow()
                    )
                else:
                    # SQLite
                    cursor = await conn.execute(
                        "SELECT 1 FROM token_blacklist WHERE jti = ? AND expires_at > ? LIMIT 1",
                        (jti, datetime.utcnow().isoformat())
                    )
                    result = await cursor.fetchone()
                    exists = result is not None
                
                if exists:
                    # Add to local cache
                    self._local_cache.add(jti)
                    return True
                    
        except Exception as e:
            logger.error(f"Database error checking blacklist: {e}")
            # Fail closed - treat as blacklisted on error
            return True
        
        return False
    
    async def revoke_all_user_tokens(
        self,
        user_id: int,
        reason: str = "User requested logout from all devices",
        revoked_by: Optional[int] = None,
        ip_address: Optional[str] = None
    ) -> int:
        """
        Revoke all tokens for a specific user
        
        Args:
            user_id: User whose tokens to revoke
            reason: Reason for revocation
            revoked_by: User who initiated revocation
            ip_address: IP address of request
            
        Returns:
            Number of tokens revoked
        """
        if not self._initialized:
            await self.initialize()
        
        try:
            # Get all active sessions for user
            async with self.db_pool.acquire() as conn:
                if hasattr(conn, 'fetch'):
                    # PostgreSQL
                    sessions = await conn.fetch(
                        "SELECT id, token_hash FROM sessions WHERE user_id = $1 AND is_revoked = 0",
                        user_id
                    )
                else:
                    # SQLite
                    cursor = await conn.execute(
                        "SELECT id, token_hash FROM sessions WHERE user_id = ? AND is_revoked = 0",
                        (user_id,)
                    )
                    sessions = await cursor.fetchall()
                
                # Mark sessions as revoked
                if hasattr(conn, 'execute'):
                    # PostgreSQL
                    await conn.execute(
                        "UPDATE sessions SET is_revoked = 1 WHERE user_id = $1",
                        user_id
                    )
                else:
                    # SQLite
                    await conn.execute(
                        "UPDATE sessions SET is_revoked = 1 WHERE user_id = ?",
                        (user_id,)
                    )
                    await conn.commit()
            
            logger.info(f"Revoked {len(sessions)} tokens for user {user_id}")
            return len(sessions)
            
        except Exception as e:
            logger.error(f"Failed to revoke user tokens: {e}")
            return 0
    
    async def cleanup_expired(self) -> int:
        """
        Remove expired tokens from the blacklist
        
        Returns:
            Number of tokens removed
        """
        if not self._initialized:
            await self.initialize()
        
        try:
            async with self.db_pool.transaction() as conn:
                if hasattr(conn, 'execute'):
                    # PostgreSQL
                    result = await conn.execute(
                        "DELETE FROM token_blacklist WHERE expires_at < $1",
                        datetime.utcnow()
                    )
                    # PostgreSQL returns number of affected rows
                    count = int(result.split()[-1]) if isinstance(result, str) else 0
                else:
                    # SQLite
                    cursor = await conn.execute(
                        "DELETE FROM token_blacklist WHERE expires_at < ?",
                        (datetime.utcnow().isoformat(),)
                    )
                    await conn.commit()
                    count = cursor.rowcount
            
            if count > 0:
                logger.info(f"Cleaned up {count} expired tokens from blacklist")
                
            # Clear local cache periodically
            if len(self._local_cache) > self._cache_size_limit / 2:
                self._local_cache.clear()
                
            return count
            
        except Exception as e:
            logger.error(f"Failed to cleanup expired tokens: {e}")
            return 0
    
    async def get_blacklist_stats(self, user_id: Optional[int] = None) -> Dict[str, Any]:
        """
        Get statistics about blacklisted tokens
        
        Args:
            user_id: Optional user ID to filter by
            
        Returns:
            Dictionary with blacklist statistics
        """
        if not self._initialized:
            await self.initialize()
        
        try:
            async with self.db_pool.acquire() as conn:
                if user_id:
                    if hasattr(conn, 'fetchrow'):
                        # PostgreSQL
                        stats = await conn.fetchrow("""
                            SELECT 
                                COUNT(*) as total,
                                COUNT(CASE WHEN token_type = 'access' THEN 1 END) as access_tokens,
                                COUNT(CASE WHEN token_type = 'refresh' THEN 1 END) as refresh_tokens,
                                MIN(revoked_at) as earliest_revocation,
                                MAX(revoked_at) as latest_revocation
                            FROM token_blacklist
                            WHERE user_id = $1 AND expires_at > $2
                        """, user_id, datetime.utcnow())
                    else:
                        # SQLite
                        cursor = await conn.execute("""
                            SELECT 
                                COUNT(*) as total,
                                SUM(CASE WHEN token_type = 'access' THEN 1 ELSE 0 END) as access_tokens,
                                SUM(CASE WHEN token_type = 'refresh' THEN 1 ELSE 0 END) as refresh_tokens,
                                MIN(revoked_at) as earliest_revocation,
                                MAX(revoked_at) as latest_revocation
                            FROM token_blacklist
                            WHERE user_id = ? AND expires_at > ?
                        """, (user_id, datetime.utcnow().isoformat()))
                        stats = await cursor.fetchone()
                else:
                    if hasattr(conn, 'fetchrow'):
                        # PostgreSQL - global stats
                        stats = await conn.fetchrow("""
                            SELECT 
                                COUNT(*) as total,
                                COUNT(DISTINCT user_id) as unique_users,
                                COUNT(CASE WHEN token_type = 'access' THEN 1 END) as access_tokens,
                                COUNT(CASE WHEN token_type = 'refresh' THEN 1 END) as refresh_tokens
                            FROM token_blacklist
                            WHERE expires_at > $1
                        """, datetime.utcnow())
                    else:
                        # SQLite - global stats
                        cursor = await conn.execute("""
                            SELECT 
                                COUNT(*) as total,
                                COUNT(DISTINCT user_id) as unique_users,
                                SUM(CASE WHEN token_type = 'access' THEN 1 ELSE 0 END) as access_tokens,
                                SUM(CASE WHEN token_type = 'refresh' THEN 1 ELSE 0 END) as refresh_tokens
                            FROM token_blacklist
                            WHERE expires_at > ?
                        """, (datetime.utcnow().isoformat(),))
                        stats = await cursor.fetchone()
                
                # Convert to dictionary
                if stats:
                    return dict(stats) if hasattr(stats, 'keys') else {
                        "total": stats[0],
                        "unique_users": stats[1] if not user_id else 1,
                        "access_tokens": stats[2] if not user_id else stats[1],
                        "refresh_tokens": stats[3] if not user_id else stats[2]
                    }
                    
        except Exception as e:
            logger.error(f"Failed to get blacklist stats: {e}")
        
        return {
            "total": 0,
            "unique_users": 0,
            "access_tokens": 0,
            "refresh_tokens": 0
        }


#######################################################################################################################
#
# Module Functions for convenience
#

# Global instance
_token_blacklist: Optional[TokenBlacklist] = None


def get_token_blacklist() -> TokenBlacklist:
    """Get token blacklist singleton instance"""
    global _token_blacklist
    if not _token_blacklist:
        _token_blacklist = TokenBlacklist()
    return _token_blacklist


async def revoke_token(
    jti: str,
    expires_at: datetime,
    user_id: Optional[int] = None,
    reason: Optional[str] = None
) -> bool:
    """Convenience function to revoke a token"""
    blacklist = get_token_blacklist()
    return await blacklist.revoke_token(jti, expires_at, user_id, reason=reason)


async def is_token_blacklisted(jti: str) -> bool:
    """Convenience function to check if token is blacklisted"""
    blacklist = get_token_blacklist()
    return await blacklist.is_blacklisted(jti)


async def revoke_all_user_tokens(user_id: int, reason: str = "User logout") -> int:
    """Convenience function to revoke all user tokens"""
    blacklist = get_token_blacklist()
    return await blacklist.revoke_all_user_tokens(user_id, reason)


#
# End of token_blacklist.py
#######################################################################################################################