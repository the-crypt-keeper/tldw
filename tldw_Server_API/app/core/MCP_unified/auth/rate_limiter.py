"""
Rate limiting for unified MCP module

Supports both in-memory and distributed (Redis) rate limiting.
"""

import time
import asyncio
from typing import Optional, Dict, Any
from collections import defaultdict, deque
from datetime import datetime, timedelta
from abc import ABC, abstractmethod
from loguru import logger

from ..config import get_config


class RateLimitExceeded(Exception):
    """Exception raised when rate limit is exceeded"""
    def __init__(self, retry_after: int):
        self.retry_after = retry_after
        super().__init__(f"Rate limit exceeded. Retry after {retry_after} seconds")


class BaseRateLimiter(ABC):
    """Abstract base class for rate limiters"""
    
    @abstractmethod
    async def is_allowed(self, key: str) -> tuple[bool, int]:
        """
        Check if request is allowed.
        
        Returns:
            Tuple of (allowed, retry_after_seconds)
        """
        pass
    
    @abstractmethod
    async def reset(self, key: str):
        """Reset rate limit for a key"""
        pass
    
    @abstractmethod
    async def get_usage(self, key: str) -> Dict[str, Any]:
        """Get current usage statistics for a key"""
        pass


class TokenBucketRateLimiter(BaseRateLimiter):
    """
    Token bucket rate limiter for smooth rate limiting.
    
    Good for allowing burst traffic while maintaining average rate.
    """
    
    def __init__(self, rate: int, per: int, burst: int = None):
        """
        Initialize token bucket rate limiter.
        
        Args:
            rate: Number of requests allowed
            per: Time period in seconds
            burst: Maximum burst size (defaults to rate)
        """
        self.rate = rate
        self.per = per
        self.burst = burst or rate
        self.allowance = {}
        self.last_check = {}
        self._lock = asyncio.Lock()
    
    async def is_allowed(self, key: str) -> tuple[bool, int]:
        """Check if request is allowed"""
        async with self._lock:
            current = time.time()
            
            # Initialize for new key
            if key not in self.allowance:
                self.allowance[key] = self.burst
                self.last_check[key] = current
                return (True, 0)
            
            # Calculate time passed
            time_passed = current - self.last_check[key]
            self.last_check[key] = current
            
            # Add tokens based on time passed
            self.allowance[key] += time_passed * (self.rate / self.per)
            
            # Cap at burst limit
            if self.allowance[key] > self.burst:
                self.allowance[key] = self.burst
            
            # Check if we have tokens
            if self.allowance[key] < 1.0:
                # Calculate retry after
                tokens_needed = 1.0 - self.allowance[key]
                retry_after = int(tokens_needed * (self.per / self.rate)) + 1
                return (False, retry_after)
            
            # Consume a token
            self.allowance[key] -= 1.0
            return (True, 0)
    
    async def reset(self, key: str):
        """Reset rate limit for a key"""
        async with self._lock:
            if key in self.allowance:
                del self.allowance[key]
                del self.last_check[key]
    
    async def get_usage(self, key: str) -> Dict[str, Any]:
        """Get current usage statistics"""
        async with self._lock:
            return {
                "tokens_remaining": int(self.allowance.get(key, self.burst)),
                "burst_limit": self.burst,
                "rate": f"{self.rate}/{self.per}s"
            }
    
    async def cleanup_old_entries(self, max_age: int = 3600):
        """Clean up old entries to prevent memory leak"""
        async with self._lock:
            current = time.time()
            keys_to_delete = []
            
            for key, last_check in self.last_check.items():
                if current - last_check > max_age:
                    keys_to_delete.append(key)
            
            for key in keys_to_delete:
                del self.allowance[key]
                del self.last_check[key]
            
            if keys_to_delete:
                logger.debug(f"Cleaned up {len(keys_to_delete)} old rate limit entries")


class SlidingWindowRateLimiter(BaseRateLimiter):
    """
    Sliding window rate limiter for precise rate limiting.
    
    More accurate than token bucket but uses more memory.
    """
    
    def __init__(self, rate: int, window: int):
        """
        Initialize sliding window rate limiter.
        
        Args:
            rate: Number of requests allowed
            window: Time window in seconds
        """
        self.rate = rate
        self.window = window
        self.requests = defaultdict(deque)
        self._lock = asyncio.Lock()
    
    async def is_allowed(self, key: str) -> tuple[bool, int]:
        """Check if request is allowed"""
        async with self._lock:
            current = time.time()
            window_start = current - self.window
            
            # Get request timestamps for this key
            timestamps = self.requests[key]
            
            # Remove old timestamps outside window
            while timestamps and timestamps[0] < window_start:
                timestamps.popleft()
            
            # Check if under rate limit
            if len(timestamps) >= self.rate:
                # Calculate retry after (when oldest request exits window)
                retry_after = int(timestamps[0] + self.window - current) + 1
                return (False, retry_after)
            
            # Add current timestamp
            timestamps.append(current)
            return (True, 0)
    
    async def reset(self, key: str):
        """Reset rate limit for a key"""
        async with self._lock:
            if key in self.requests:
                del self.requests[key]
    
    async def get_usage(self, key: str) -> Dict[str, Any]:
        """Get current usage statistics"""
        async with self._lock:
            current = time.time()
            window_start = current - self.window
            
            # Count requests in current window
            timestamps = self.requests[key]
            count = sum(1 for ts in timestamps if ts >= window_start)
            
            return {
                "requests_in_window": count,
                "limit": self.rate,
                "window": f"{self.window}s",
                "remaining": max(0, self.rate - count)
            }
    
    async def cleanup_old_entries(self, max_age: int = 3600):
        """Clean up old entries to prevent memory leak"""
        async with self._lock:
            current = time.time()
            empty_keys = []
            
            for key, timestamps in self.requests.items():
                # Remove old timestamps
                cutoff = current - max_age
                while timestamps and timestamps[0] < cutoff:
                    timestamps.popleft()
                
                # Mark empty keys for deletion
                if not timestamps:
                    empty_keys.append(key)
            
            # Delete empty keys
            for key in empty_keys:
                del self.requests[key]
            
            if empty_keys:
                logger.debug(f"Cleaned up {len(empty_keys)} empty rate limit entries")


class DistributedRateLimiter(BaseRateLimiter):
    """
    Redis-backed distributed rate limiter for multi-instance deployments.
    
    Uses Redis Lua scripts for atomic operations.
    """
    
    def __init__(self, rate: int, window: int, redis_client=None):
        """
        Initialize distributed rate limiter.
        
        Args:
            rate: Number of requests allowed
            window: Time window in seconds
            redis_client: Redis client instance
        """
        self.rate = rate
        self.window = window
        self.redis = redis_client
        
        # Lua script for atomic rate limit check
        self.lua_script = """
        local key = KEYS[1]
        local limit = tonumber(ARGV[1])
        local window = tonumber(ARGV[2])
        local current_time = tonumber(ARGV[3])
        
        -- Remove old entries
        redis.call('ZREMRANGEBYSCORE', key, 0, current_time - window)
        
        -- Count current entries
        local current_count = redis.call('ZCARD', key)
        
        if current_count < limit then
            -- Add new entry
            redis.call('ZADD', key, current_time, current_time)
            redis.call('EXPIRE', key, window)
            return {1, 0}  -- allowed, no retry
        else
            -- Get oldest entry
            local oldest = redis.call('ZRANGE', key, 0, 0, 'WITHSCORES')
            if oldest[2] then
                local retry_after = math.ceil(oldest[2] + window - current_time)
                return {0, retry_after}  -- not allowed, retry after
            else
                return {0, window}  -- not allowed, retry after window
            end
        end
        """
        
        if self.redis:
            self.script_sha = self.redis.script_load(self.lua_script)
    
    async def is_allowed(self, key: str) -> tuple[bool, int]:
        """Check if request is allowed using Redis"""
        if not self.redis:
            # Fallback to in-memory if Redis not available
            logger.warning("Redis not available, using in-memory rate limiting")
            return (True, 0)
        
        try:
            # Execute Lua script
            result = await self.redis.evalsha(
                self.script_sha,
                1,  # number of keys
                f"rate_limit:{key}",  # key
                self.rate,  # limit
                self.window,  # window
                time.time()  # current time
            )
            
            return (bool(result[0]), int(result[1]))
            
        except Exception as e:
            logger.error(f"Redis rate limit error: {e}")
            # Allow request on error (fail open)
            return (True, 0)
    
    async def reset(self, key: str):
        """Reset rate limit for a key in Redis"""
        if self.redis:
            try:
                await self.redis.delete(f"rate_limit:{key}")
            except Exception as e:
                logger.error(f"Redis reset error: {e}")
    
    async def get_usage(self, key: str) -> Dict[str, Any]:
        """Get current usage statistics from Redis"""
        if not self.redis:
            return {"error": "Redis not available"}
        
        try:
            redis_key = f"rate_limit:{key}"
            current_time = time.time()
            
            # Remove old entries and count current
            await self.redis.zremrangebyscore(redis_key, 0, current_time - self.window)
            count = await self.redis.zcard(redis_key)
            
            return {
                "requests_in_window": count,
                "limit": self.rate,
                "window": f"{self.window}s",
                "remaining": max(0, self.rate - count)
            }
            
        except Exception as e:
            logger.error(f"Redis usage error: {e}")
            return {"error": str(e)}


class RateLimiter:
    """
    Main rate limiter that automatically selects appropriate backend.
    
    Uses Redis if available, otherwise falls back to in-memory.
    """
    
    def __init__(self):
        self.config = get_config()
        self.limiters = {}
        self._init_limiters()
        
        # Start cleanup task for in-memory limiters
        if not self.config.rate_limit_use_redis:
            asyncio.create_task(self._cleanup_task())
    
    def _init_limiters(self):
        """Initialize rate limiters based on configuration"""
        if self.config.rate_limit_use_redis and self.config.redis_url:
            # Initialize Redis client
            try:
                import redis.asyncio as redis
                redis_params = self.config.get_redis_connection_params()
                redis_client = redis.from_url(**redis_params)
                
                # Create distributed limiter
                self.default_limiter = DistributedRateLimiter(
                    rate=self.config.rate_limit_requests_per_minute,
                    window=60,
                    redis_client=redis_client
                )
                logger.info("Using Redis-backed distributed rate limiting")
            except ImportError:
                logger.warning("Redis not installed, falling back to in-memory rate limiting")
                self._init_memory_limiter()
            except Exception as e:
                logger.error(f"Failed to initialize Redis: {e}")
                self._init_memory_limiter()
        else:
            self._init_memory_limiter()
    
    def _init_memory_limiter(self):
        """Initialize in-memory rate limiter"""
        self.default_limiter = TokenBucketRateLimiter(
            rate=self.config.rate_limit_requests_per_minute,
            per=60,
            burst=self.config.rate_limit_burst_size
        )
        logger.info("Using in-memory token bucket rate limiting")
    
    async def check_rate_limit(
        self,
        key: str,
        limiter: Optional[BaseRateLimiter] = None
    ) -> None:
        """
        Check rate limit and raise exception if exceeded.
        
        Args:
            key: Unique identifier for rate limiting
            limiter: Optional custom limiter
        
        Raises:
            RateLimitExceeded: If rate limit is exceeded
        """
        if not self.config.rate_limit_enabled:
            return
        
        limiter = limiter or self.default_limiter
        allowed, retry_after = await limiter.is_allowed(key)
        
        if not allowed:
            logger.warning(f"Rate limit exceeded for key: {key}")
            raise RateLimitExceeded(retry_after)
    
    async def get_usage(self, key: str) -> Dict[str, Any]:
        """Get rate limit usage for a key"""
        return await self.default_limiter.get_usage(key)
    
    async def reset(self, key: str):
        """Reset rate limit for a key"""
        await self.default_limiter.reset(key)
        logger.info(f"Rate limit reset for key: {key}", extra={"audit": True})
    
    async def _cleanup_task(self):
        """Background task to clean up old entries"""
        while True:
            await asyncio.sleep(3600)  # Run every hour
            
            try:
                if hasattr(self.default_limiter, 'cleanup_old_entries'):
                    await self.default_limiter.cleanup_old_entries()
                
                for limiter in self.limiters.values():
                    if hasattr(limiter, 'cleanup_old_entries'):
                        await limiter.cleanup_old_entries()
                        
            except Exception as e:
                logger.error(f"Error in rate limit cleanup: {e}")


# Singleton instance
_rate_limiter = None


def get_rate_limiter() -> RateLimiter:
    """Get or create rate limiter singleton"""
    global _rate_limiter
    if _rate_limiter is None:
        _rate_limiter = RateLimiter()
    return _rate_limiter