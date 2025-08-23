# character_rate_limiter.py
"""
Rate limiting for character operations to prevent abuse.
Supports both Redis (for distributed deployments) and in-memory (for single-instance).
"""

import time
from typing import Dict, List, Optional, Tuple
from collections import defaultdict
import redis
from loguru import logger
from fastapi import HTTPException, status


class CharacterRateLimiter:
    """
    Rate limiter for character operations with Redis and in-memory fallback.
    
    Limits are per-user and apply to character creation, updates, and imports.
    """
    
    def __init__(
        self,
        redis_client: Optional[redis.Redis] = None,
        max_operations: int = 100,
        window_seconds: int = 3600,
        max_characters: int = 1000,  # Max total characters per user
        max_import_size_mb: int = 10  # Max import file size
    ):
        """
        Initialize the rate limiter.
        
        Args:
            redis_client: Optional Redis client for distributed rate limiting
            max_operations: Maximum operations per window
            window_seconds: Time window in seconds
            max_characters: Maximum total characters per user
            max_import_size_mb: Maximum import file size in MB
        """
        self.redis = redis_client
        self.max_operations = max_operations
        self.window_seconds = window_seconds
        self.max_characters = max_characters
        self.max_import_size_mb = max_import_size_mb
        
        # In-memory fallback storage
        self.memory_store: Dict[int, List[float]] = defaultdict(list) if not redis_client else {}
        
        logger.info(
            f"CharacterRateLimiter initialized: "
            f"max_ops={max_operations}/{window_seconds}s, "
            f"max_chars={max_characters}, "
            f"redis={'enabled' if redis_client else 'disabled (using memory)'}"
        )
    
    async def check_rate_limit(self, user_id: int, operation: str = "character_op") -> Tuple[bool, int]:
        """
        Check if user has exceeded rate limit.
        
        Args:
            user_id: User ID to check
            operation: Type of operation (for logging)
            
        Returns:
            Tuple of (allowed, remaining_operations)
            
        Raises:
            HTTPException: If rate limit exceeded
        """
        key = f"rate_limit:character:{user_id}"
        
        if self.redis:
            try:
                # Use Redis for distributed rate limiting
                pipe = self.redis.pipeline()
                now = time.time()
                window_start = now - self.window_seconds
                
                # Remove old entries and count current
                pipe.zremrangebyscore(key, 0, window_start)
                pipe.zcard(key)
                pipe.zadd(key, {str(now): now})
                pipe.expire(key, self.window_seconds)
                
                results = pipe.execute()
                current_count = results[1]
                
                if current_count >= self.max_operations:
                    logger.warning(
                        f"Rate limit exceeded for user {user_id}: "
                        f"{current_count}/{self.max_operations} operations"
                    )
                    raise HTTPException(
                        status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                        detail=f"Rate limit exceeded. Max {self.max_operations} character operations per hour."
                    )
                
                remaining = self.max_operations - current_count
                logger.debug(f"Rate limit check for user {user_id}: {remaining} operations remaining")
                return True, remaining
                
            except redis.RedisError as e:
                logger.error(f"Redis error in rate limiter: {e}. Falling back to memory.")
                # Fall through to in-memory implementation
        
        # In-memory fallback
        now = time.time()
        window_start = now - self.window_seconds
        
        # Clean old entries
        self.memory_store[user_id] = [
            t for t in self.memory_store[user_id]
            if t > window_start
        ]
        
        current_count = len(self.memory_store[user_id])
        
        if current_count >= self.max_operations:
            logger.warning(
                f"Rate limit exceeded for user {user_id} (memory): "
                f"{current_count}/{self.max_operations} operations"
            )
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail=f"Rate limit exceeded. Max {self.max_operations} character operations per hour."
            )
        
        self.memory_store[user_id].append(now)
        remaining = self.max_operations - current_count - 1
        logger.debug(f"Rate limit check for user {user_id} (memory): {remaining} operations remaining")
        return True, remaining
    
    async def check_character_limit(self, user_id: int, current_count: int) -> bool:
        """
        Check if user has exceeded maximum character limit.
        
        Args:
            user_id: User ID to check
            current_count: Current number of characters
            
        Returns:
            True if under limit, raises exception if over
            
        Raises:
            HTTPException: If character limit exceeded
        """
        if current_count >= self.max_characters:
            logger.warning(
                f"Character limit exceeded for user {user_id}: "
                f"{current_count}/{self.max_characters} characters"
            )
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Character limit exceeded. Maximum {self.max_characters} characters allowed."
            )
        return True
    
    def check_import_size(self, file_size_bytes: int) -> bool:
        """
        Check if import file size is within limits.
        
        Args:
            file_size_bytes: File size in bytes
            
        Returns:
            True if under limit, raises exception if over
            
        Raises:
            HTTPException: If file size exceeds limit
        """
        max_bytes = self.max_import_size_mb * 1024 * 1024
        if file_size_bytes > max_bytes:
            logger.warning(
                f"Import file too large: {file_size_bytes} bytes "
                f"(max: {max_bytes} bytes)"
            )
            raise HTTPException(
                status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                detail=f"File too large. Maximum size is {self.max_import_size_mb}MB."
            )
        return True
    
    async def get_usage_stats(self, user_id: int) -> Dict[str, any]:
        """
        Get current usage statistics for a user.
        
        Args:
            user_id: User ID to check
            
        Returns:
            Dictionary with usage statistics
        """
        key = f"rate_limit:character:{user_id}"
        now = time.time()
        window_start = now - self.window_seconds
        
        if self.redis:
            try:
                # Count operations in current window
                count = self.redis.zcount(key, window_start, now)
                return {
                    "operations_used": count,
                    "operations_limit": self.max_operations,
                    "operations_remaining": max(0, self.max_operations - count),
                    "window_seconds": self.window_seconds,
                    "reset_time": now + self.window_seconds
                }
            except redis.RedisError:
                pass
        
        # In-memory fallback
        self.memory_store[user_id] = [
            t for t in self.memory_store[user_id]
            if t > window_start
        ]
        count = len(self.memory_store[user_id])
        
        return {
            "operations_used": count,
            "operations_limit": self.max_operations,
            "operations_remaining": max(0, self.max_operations - count),
            "window_seconds": self.window_seconds,
            "reset_time": now + self.window_seconds
        }


# Global instance (initialized in dependencies)
_rate_limiter: Optional[CharacterRateLimiter] = None


def get_character_rate_limiter() -> CharacterRateLimiter:
    """Get the global rate limiter instance."""
    global _rate_limiter
    
    if _rate_limiter is None:
        # Initialize with Redis if available
        from tldw_Server_API.app.core.config import settings
        
        redis_client = None
        if settings.get("REDIS_ENABLED", False):
            try:
                import redis
                redis_client = redis.from_url(
                    settings.get("REDIS_URL", "redis://localhost:6379/0"),
                    decode_responses=False
                )
                # Test connection
                redis_client.ping()
                logger.info("Redis connected for character rate limiting")
            except Exception as e:
                logger.warning(f"Redis not available for rate limiting: {e}. Using in-memory fallback.")
                redis_client = None
        
        _rate_limiter = CharacterRateLimiter(
            redis_client=redis_client,
            max_operations=settings.get("CHARACTER_RATE_LIMIT_OPS", 100),
            window_seconds=settings.get("CHARACTER_RATE_LIMIT_WINDOW", 3600),
            max_characters=settings.get("MAX_CHARACTERS_PER_USER", 1000),
            max_import_size_mb=settings.get("MAX_CHARACTER_IMPORT_SIZE_MB", 10)
        )
    
    return _rate_limiter