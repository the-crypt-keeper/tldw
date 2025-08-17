# rate_limiter.py
# Description: Database-backed rate limiting with token bucket algorithm
#
# Imports
from datetime import datetime, timedelta
from typing import Optional, Dict, Any, Tuple
import hashlib
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
from tldw_Server_API.app.core.AuthNZ.exceptions import RateLimitError

#######################################################################################################################
#
# Rate Limiter Class

class RateLimiter:
    """
    Token bucket rate limiter with database backend and optional Redis caching
    
    Implements a sliding window rate limiter that:
    - Tracks requests per identifier (IP, user_id, API key)
    - Supports burst traffic handling
    - Uses database as source of truth
    - Optionally uses Redis for performance
    """
    
    def __init__(
        self,
        db_pool: Optional[DatabasePool] = None,
        settings: Optional[Settings] = None
    ):
        """Initialize rate limiter"""
        self.settings = settings or get_settings()
        self.db_pool = db_pool
        self.redis_client: Optional[redis.Redis] = None
        self.enabled = self.settings.RATE_LIMIT_ENABLED
        self._initialized = False
        
        # Default limits
        self.default_limit = self.settings.RATE_LIMIT_PER_MINUTE
        self.default_burst = self.settings.RATE_LIMIT_BURST
        
        # Service account limits
        self.service_limit = self.settings.SERVICE_ACCOUNT_RATE_LIMIT
        
    async def initialize(self):
        """Initialize rate limiter"""
        if self._initialized:
            return
        
        # Get database pool
        if not self.db_pool:
            self.db_pool = await get_db_pool()
        
        # Initialize Redis if configured
        if self.settings.REDIS_URL:
            try:
                self.redis_client = redis.from_url(
                    self.settings.REDIS_URL,
                    decode_responses=True,
                    socket_connect_timeout=1
                )
                self.redis_client.ping()
                logger.debug("Redis connected for rate limiting")
            except (RedisError, Exception) as e:
                logger.warning(f"Redis unavailable for rate limiting: {e}")
                self.redis_client = None
        
        self._initialized = True
        logger.info(f"RateLimiter initialized (enabled={self.enabled})")
    
    async def check_rate_limit(
        self,
        identifier: str,
        endpoint: str,
        limit: Optional[int] = None,
        burst: Optional[int] = None,
        window_minutes: int = 1
    ) -> Tuple[bool, Dict[str, Any]]:
        """
        Check if request is within rate limit
        
        Args:
            identifier: Unique identifier (IP, user_id, etc.)
            endpoint: API endpoint being accessed
            limit: Requests allowed per window (default from settings)
            burst: Burst requests allowed
            window_minutes: Time window in minutes
            
        Returns:
            Tuple of (is_allowed, metadata)
            - is_allowed: True if request is allowed
            - metadata: Rate limit information (remaining, reset_time, etc.)
        """
        if not self.enabled:
            return True, {"rate_limit_enabled": False}
        
        if not self._initialized:
            await self.initialize()
        
        # Use provided limits or defaults
        limit = limit or self.default_limit
        burst = burst or self.default_burst
        
        # Create unique key for rate limiting
        key = self._create_key(identifier, endpoint)
        
        # Try Redis first if available
        if self.redis_client:
            result = await self._check_redis_rate_limit(
                key, limit, burst, window_minutes
            )
            if result is not None:
                return result
        
        # Fallback to database
        return await self._check_database_rate_limit(
            identifier, endpoint, limit, burst, window_minutes
        )
    
    async def _check_redis_rate_limit(
        self,
        key: str,
        limit: int,
        burst: int,
        window_minutes: int
    ) -> Optional[Tuple[bool, Dict[str, Any]]]:
        """Check rate limit using Redis"""
        if not self.redis_client:
            return None
        
        try:
            # Use Redis pipeline for atomic operations
            pipe = self.redis_client.pipeline()
            
            now = datetime.utcnow()
            window_start = now - timedelta(minutes=window_minutes)
            window_key = f"rate:{key}:{window_start.strftime('%Y%m%d%H%M')}"
            
            # Increment counter
            pipe.incr(window_key)
            pipe.expire(window_key, window_minutes * 60 + 10)  # Extra 10 seconds buffer
            
            results = pipe.execute()
            current_count = results[0]
            
            # Check burst by looking at previous window
            if current_count > limit - burst:
                prev_window_start = window_start - timedelta(minutes=window_minutes)
                prev_window_key = f"rate:{key}:{prev_window_start.strftime('%Y%m%d%H%M')}"
                prev_count = self.redis_client.get(prev_window_key)
                prev_count = int(prev_count) if prev_count else 0
                
                if prev_count + current_count > limit + burst:
                    # Rate limit exceeded
                    reset_time = window_start + timedelta(minutes=window_minutes * 2)
                    return False, {
                        "limit": limit,
                        "remaining": max(0, limit - current_count),
                        "reset_time": reset_time.isoformat(),
                        "retry_after": int((reset_time - now).total_seconds())
                    }
            
            # Request allowed
            return True, {
                "limit": limit,
                "remaining": max(0, limit - current_count),
                "reset_time": (window_start + timedelta(minutes=window_minutes)).isoformat()
            }
            
        except RedisError as e:
            logger.warning(f"Redis rate limit check failed: {e}")
            return None
    
    async def _check_database_rate_limit(
        self,
        identifier: str,
        endpoint: str,
        limit: int,
        burst: int,
        window_minutes: int
    ) -> Tuple[bool, Dict[str, Any]]:
        """Check rate limit using database"""
        now = datetime.utcnow()
        window_start = now.replace(second=0, microsecond=0)
        
        try:
            async with self.db_pool.transaction() as conn:
                if hasattr(conn, 'fetchval'):
                    # PostgreSQL - atomic upsert
                    result = await conn.fetchval(
                        """
                        INSERT INTO rate_limits (identifier, endpoint, request_count, window_start)
                        VALUES ($1, $2, 1, $3)
                        ON CONFLICT (identifier, endpoint, window_start) 
                        DO UPDATE SET request_count = rate_limits.request_count + 1
                        RETURNING request_count
                        """,
                        identifier, endpoint, window_start
                    )
                    current_count = result
                    
                    # Check burst with previous window
                    if current_count > limit - burst:
                        prev_window = window_start - timedelta(minutes=window_minutes)
                        prev_count = await conn.fetchval(
                            """
                            SELECT request_count FROM rate_limits
                            WHERE identifier = $1 AND endpoint = $2 AND window_start = $3
                            """,
                            identifier, endpoint, prev_window
                        )
                        prev_count = prev_count or 0
                        
                        if prev_count + current_count > limit + burst:
                            # Rate limit exceeded
                            reset_time = window_start + timedelta(minutes=window_minutes)
                            return False, {
                                "limit": limit,
                                "remaining": 0,
                                "reset_time": reset_time.isoformat(),
                                "retry_after": int((reset_time - now).total_seconds())
                            }
                    
                else:
                    # SQLite - manual upsert
                    cursor = await conn.execute(
                        """
                        SELECT request_count FROM rate_limits
                        WHERE identifier = ? AND endpoint = ? AND window_start = ?
                        """,
                        (identifier, endpoint, window_start.isoformat())
                    )
                    row = await cursor.fetchone()
                    
                    if row:
                        current_count = row[0] + 1
                        await conn.execute(
                            """
                            UPDATE rate_limits
                            SET request_count = ?
                            WHERE identifier = ? AND endpoint = ? AND window_start = ?
                            """,
                            (current_count, identifier, endpoint, window_start.isoformat())
                        )
                    else:
                        current_count = 1
                        await conn.execute(
                            """
                            INSERT INTO rate_limits (identifier, endpoint, request_count, window_start)
                            VALUES (?, ?, ?, ?)
                            """,
                            (identifier, endpoint, 1, window_start.isoformat())
                        )
                    
                    # Check burst
                    if current_count > limit - burst:
                        prev_window = window_start - timedelta(minutes=window_minutes)
                        cursor = await conn.execute(
                            """
                            SELECT request_count FROM rate_limits
                            WHERE identifier = ? AND endpoint = ? AND window_start = ?
                            """,
                            (identifier, endpoint, prev_window.isoformat())
                        )
                        row = await cursor.fetchone()
                        prev_count = row[0] if row else 0
                        
                        if prev_count + current_count > limit + burst:
                            # Rate limit exceeded
                            reset_time = window_start + timedelta(minutes=window_minutes)
                            await conn.commit()
                            return False, {
                                "limit": limit,
                                "remaining": 0,
                                "reset_time": reset_time.isoformat(),
                                "retry_after": int((reset_time - now).total_seconds())
                            }
                    
                    await conn.commit()
                
                # Request allowed
                return True, {
                    "limit": limit,
                    "remaining": max(0, limit - current_count),
                    "reset_time": (window_start + timedelta(minutes=window_minutes)).isoformat()
                }
                
        except Exception as e:
            logger.error(f"Database rate limit check failed: {e}")
            # On error, deny the request (fail closed) for security
            return False, {
                "error": "Rate limit check failed",
                "limit": limit,
                "remaining": 0,
                "retry_after": 60  # Conservative retry time
            }
    
    async def check_user_rate_limit(
        self,
        user_id: int,
        endpoint: str,
        role: str = "user"
    ) -> Tuple[bool, Dict[str, Any]]:
        """
        Check rate limit for authenticated user
        
        Args:
            user_id: User's database ID
            endpoint: API endpoint
            role: User's role (affects limits)
            
        Returns:
            Rate limit check result
        """
        # Determine limits based on role
        if role == "service":
            limit = self.service_limit
            burst = self.service_limit // 10
        elif role in ["admin", "root"]:
            limit = self.default_limit * 10
            burst = self.default_burst * 10
        elif role == "moderator":
            limit = self.default_limit * 2
            burst = self.default_burst * 2
        else:
            limit = self.default_limit
            burst = self.default_burst
        
        identifier = f"user:{user_id}"
        return await self.check_rate_limit(
            identifier, endpoint, limit, burst
        )
    
    async def check_ip_rate_limit(
        self,
        ip_address: str,
        endpoint: str
    ) -> Tuple[bool, Dict[str, Any]]:
        """
        Check rate limit for IP address
        
        Args:
            ip_address: Client IP address
            endpoint: API endpoint
            
        Returns:
            Rate limit check result
        """
        identifier = f"ip:{ip_address}"
        return await self.check_rate_limit(identifier, endpoint)
    
    async def check_api_key_rate_limit(
        self,
        api_key_hash: str,
        endpoint: str,
        is_service_account: bool = False
    ) -> Tuple[bool, Dict[str, Any]]:
        """
        Check rate limit for API key
        
        Args:
            api_key_hash: Hashed API key
            endpoint: API endpoint
            is_service_account: Whether this is a service account
            
        Returns:
            Rate limit check result
        """
        identifier = f"api:{api_key_hash[:16]}"  # Use first 16 chars of hash
        
        if is_service_account:
            limit = self.service_limit
            burst = self.service_limit // 10
        else:
            limit = self.default_limit
            burst = self.default_burst
        
        return await self.check_rate_limit(
            identifier, endpoint, limit, burst
        )
    
    async def reset_rate_limit(
        self,
        identifier: str,
        endpoint: Optional[str] = None
    ):
        """
        Reset rate limit for an identifier
        
        Args:
            identifier: Identifier to reset
            endpoint: Specific endpoint or all if None
        """
        if not self._initialized:
            await self.initialize()
        
        try:
            async with self.db_pool.transaction() as conn:
                if hasattr(conn, 'execute'):
                    # PostgreSQL
                    if endpoint:
                        await conn.execute(
                            "DELETE FROM rate_limits WHERE identifier = $1 AND endpoint = $2",
                            identifier, endpoint
                        )
                    else:
                        await conn.execute(
                            "DELETE FROM rate_limits WHERE identifier = $1",
                            identifier
                        )
                else:
                    # SQLite
                    if endpoint:
                        await conn.execute(
                            "DELETE FROM rate_limits WHERE identifier = ? AND endpoint = ?",
                            (identifier, endpoint)
                        )
                    else:
                        await conn.execute(
                            "DELETE FROM rate_limits WHERE identifier = ?",
                            (identifier,)
                        )
                    await conn.commit()
                
                # Clear Redis cache if available
                if self.redis_client:
                    pattern = f"rate:{self._create_key(identifier, endpoint or '*')}:*"
                    for key in self.redis_client.scan_iter(pattern):
                        self.redis_client.delete(key)
                
                logger.info(f"Reset rate limit for {identifier}")
                
        except Exception as e:
            logger.error(f"Failed to reset rate limit: {e}")
    
    async def cleanup_old_entries(self, hours: int = 1):
        """
        Clean up old rate limit entries
        
        Args:
            hours: Remove entries older than this many hours
        """
        if not self._initialized:
            await self.initialize()
        
        try:
            cutoff = datetime.utcnow() - timedelta(hours=hours)
            
            async with self.db_pool.transaction() as conn:
                if hasattr(conn, 'fetchval'):
                    # PostgreSQL
                    deleted = await conn.fetchval(
                        """
                        DELETE FROM rate_limits
                        WHERE window_start < $1
                        RETURNING COUNT(*)
                        """,
                        cutoff
                    )
                else:
                    # SQLite
                    cursor = await conn.execute(
                        """
                        DELETE FROM rate_limits
                        WHERE datetime(window_start) < datetime(?)
                        """,
                        (cutoff.isoformat(),)
                    )
                    deleted = cursor.rowcount
                    await conn.commit()
                
                if deleted:
                    logger.info(f"Cleaned up {deleted} old rate limit entries")
                    
        except Exception as e:
            logger.error(f"Rate limit cleanup failed: {e}")
    
    def _create_key(self, identifier: str, endpoint: str) -> str:
        """Create a unique key for rate limiting"""
        combined = f"{identifier}:{endpoint}"
        return hashlib.md5(combined.encode()).hexdigest()
    
    async def get_current_usage(
        self,
        identifier: str,
        endpoint: str
    ) -> Dict[str, Any]:
        """
        Get current rate limit usage for an identifier
        
        Args:
            identifier: Identifier to check
            endpoint: API endpoint
            
        Returns:
            Current usage statistics
        """
        if not self._initialized:
            await self.initialize()
        
        now = datetime.utcnow()
        window_start = now.replace(second=0, microsecond=0)
        
        try:
            result = await self.db_pool.fetchone(
                """
                SELECT request_count FROM rate_limits
                WHERE identifier = ? AND endpoint = ? AND window_start = ?
                """,
                identifier, endpoint, window_start
            )
            
            current_count = result['request_count'] if result else 0
            
            return {
                "identifier": identifier,
                "endpoint": endpoint,
                "current_count": current_count,
                "limit": self.default_limit,
                "remaining": max(0, self.default_limit - current_count),
                "window_start": window_start.isoformat(),
                "window_end": (window_start + timedelta(minutes=1)).isoformat()
            }
            
        except Exception as e:
            logger.error(f"Failed to get rate limit usage: {e}")
            return {"error": "Failed to retrieve usage"}


#######################################################################################################################
#
# Module Functions

# Global instance
_rate_limiter: Optional[RateLimiter] = None


async def get_rate_limiter() -> RateLimiter:
    """Get rate limiter singleton instance"""
    global _rate_limiter
    if not _rate_limiter:
        _rate_limiter = RateLimiter()
        await _rate_limiter.initialize()
    return _rate_limiter


async def check_rate_limit(
    identifier: str,
    endpoint: str,
    limit: Optional[int] = None
) -> Tuple[bool, Dict[str, Any]]:
    """Convenience function to check rate limit"""
    limiter = await get_rate_limiter()
    return await limiter.check_rate_limit(identifier, endpoint, limit)


#
# End of rate_limiter.py
#######################################################################################################################