"""
Rate limiting for MCP v2
"""

from typing import Optional, Dict, Any, Callable
from datetime import datetime, timedelta
import time
import asyncio
from collections import defaultdict, deque
from fastapi import HTTPException, Request, status
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware
from loguru import logger

from ..schemas import UserRole


class RateLimiter:
    """Token bucket rate limiter"""
    
    def __init__(self, rate: int, per: int, burst: int = None):
        """
        Initialize rate limiter
        
        Args:
            rate: Number of requests allowed
            per: Time period in seconds
            burst: Optional burst allowance
        """
        self.rate = rate
        self.per = per
        self.burst = burst or rate
        self.allowance = {}
        self.last_check = {}
    
    def is_allowed(self, key: str) -> bool:
        """Check if request is allowed"""
        current = time.time()
        
        # Initialize for new key
        if key not in self.allowance:
            self.allowance[key] = self.burst
            self.last_check[key] = current
            return True
        
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
            return False
        
        # Consume a token
        self.allowance[key] -= 1.0
        return True
    
    def reset(self, key: str):
        """Reset rate limit for a key"""
        if key in self.allowance:
            del self.allowance[key]
            del self.last_check[key]


class SlidingWindowRateLimiter:
    """Sliding window rate limiter for more accurate rate limiting"""
    
    def __init__(self, rate: int, window: int):
        """
        Initialize sliding window rate limiter
        
        Args:
            rate: Number of requests allowed
            window: Time window in seconds
        """
        self.rate = rate
        self.window = window
        self.requests = defaultdict(deque)
    
    def is_allowed(self, key: str) -> bool:
        """Check if request is allowed"""
        current = time.time()
        window_start = current - self.window
        
        # Get request timestamps for this key
        timestamps = self.requests[key]
        
        # Remove old timestamps outside window
        while timestamps and timestamps[0] < window_start:
            timestamps.popleft()
        
        # Check if under rate limit
        if len(timestamps) >= self.rate:
            return False
        
        # Add current timestamp
        timestamps.append(current)
        return True
    
    def reset(self, key: str):
        """Reset rate limit for a key"""
        if key in self.requests:
            del self.requests[key]


class RateLimitManager:
    """Manages different rate limits for different user roles and endpoints"""
    
    def __init__(self):
        # Define rate limits by role
        self.role_limits = {
            UserRole.ADMIN: RateLimiter(rate=1000, per=60, burst=100),  # 1000/min
            UserRole.USER: RateLimiter(rate=100, per=60, burst=20),     # 100/min
            UserRole.API_CLIENT: RateLimiter(rate=500, per=60, burst=50), # 500/min
            UserRole.GUEST: RateLimiter(rate=10, per=60, burst=5),       # 10/min
        }
        
        # Define rate limits by endpoint pattern
        self.endpoint_limits = {
            "/chat/completions": SlidingWindowRateLimiter(rate=10, window=60),  # 10/min for chat
            "/transcribe": SlidingWindowRateLimiter(rate=5, window=300),        # 5/5min for transcription
            "/tools/call": RateLimiter(rate=50, per=60, burst=10),              # 50/min for tool calls
        }
        
        # Global rate limiter (fallback)
        self.global_limiter = RateLimiter(rate=200, per=60, burst=30)
        
        # Track requests for analytics
        self.request_counts = defaultdict(int)
        self.last_reset = time.time()
    
    def check_rate_limit(
        self,
        key: str,
        role: Optional[UserRole] = None,
        endpoint: Optional[str] = None
    ) -> bool:
        """
        Check if request is allowed under rate limits
        
        Args:
            key: Unique identifier (user_id, IP, etc.)
            role: User role for role-based limits
            endpoint: Endpoint path for endpoint-specific limits
        
        Returns:
            True if allowed, False if rate limited
        """
        # Check role-based limit
        if role and role in self.role_limits:
            if not self.role_limits[role].is_allowed(f"{key}:{role.value}"):
                logger.warning(f"Rate limit exceeded for role {role.value}: {key}")
                return False
        
        # Check endpoint-specific limit
        if endpoint:
            for pattern, limiter in self.endpoint_limits.items():
                if pattern in endpoint:
                    if not limiter.is_allowed(f"{key}:{endpoint}"):
                        logger.warning(f"Rate limit exceeded for endpoint {endpoint}: {key}")
                        return False
        
        # Check global limit
        if not self.global_limiter.is_allowed(key):
            logger.warning(f"Global rate limit exceeded: {key}")
            return False
        
        # Track request
        self.request_counts[key] += 1
        
        return True
    
    def get_limit_info(
        self,
        key: str,
        role: Optional[UserRole] = None
    ) -> Dict[str, Any]:
        """Get rate limit information for a key"""
        info = {
            "key": key,
            "role": role.value if role else None,
            "limits": {}
        }
        
        if role and role in self.role_limits:
            limiter = self.role_limits[role]
            info["limits"]["role"] = {
                "rate": limiter.rate,
                "per": limiter.per,
                "remaining": int(limiter.allowance.get(f"{key}:{role.value}", limiter.burst))
            }
        
        # Global limit info
        info["limits"]["global"] = {
            "rate": self.global_limiter.rate,
            "per": self.global_limiter.per,
            "remaining": int(self.global_limiter.allowance.get(key, self.global_limiter.burst))
        }
        
        info["request_count"] = self.request_counts.get(key, 0)
        
        return info
    
    def reset_limits(self, key: str):
        """Reset all limits for a key"""
        # Reset role limits
        for limiter in self.role_limits.values():
            for k in list(limiter.allowance.keys()):
                if k.startswith(f"{key}:"):
                    limiter.reset(k)
        
        # Reset endpoint limits
        for limiter in self.endpoint_limits.values():
            for k in list(limiter.requests.keys()) if hasattr(limiter, 'requests') else list(limiter.allowance.keys()):
                if k.startswith(f"{key}:"):
                    limiter.reset(k)
        
        # Reset global limit
        self.global_limiter.reset(key)
        
        # Reset request count
        if key in self.request_counts:
            del self.request_counts[key]
    
    def cleanup_old_entries(self, max_age: int = 3600):
        """Clean up old rate limit entries"""
        current = time.time()
        
        # Clean up token bucket limiters
        for limiter in list(self.role_limits.values()) + [self.global_limiter]:
            for key in list(limiter.last_check.keys()):
                if current - limiter.last_check[key] > max_age:
                    limiter.reset(key)
        
        # Reset request counts periodically
        if current - self.last_reset > 3600:  # Reset hourly
            self.request_counts.clear()
            self.last_reset = current


# Global rate limit manager
rate_limit_manager = RateLimitManager()


class RateLimitMiddleware(BaseHTTPMiddleware):
    """FastAPI middleware for rate limiting"""
    
    def __init__(self, app, manager: RateLimitManager = None):
        super().__init__(app)
        self.manager = manager or rate_limit_manager
    
    async def dispatch(self, request: Request, call_next):
        # Extract key (use IP or user ID if authenticated)
        client_ip = request.client.host if request.client else "unknown"
        key = client_ip
        
        # Try to get user from request state (if authenticated)
        user = getattr(request.state, "user", None)
        role = None
        
        if user:
            key = f"user:{user.id}"
            role = user.roles[0] if user.roles else UserRole.USER
        
        # Get endpoint path
        endpoint = request.url.path
        
        # Check rate limit
        if not self.manager.check_rate_limit(key, role, endpoint):
            # Get limit info for response headers
            limit_info = self.manager.get_limit_info(key, role)
            
            return JSONResponse(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                content={
                    "error": "Rate limit exceeded",
                    "message": "Too many requests. Please slow down.",
                    "limits": limit_info["limits"]
                },
                headers={
                    "X-RateLimit-Limit": str(limit_info["limits"].get("role", limit_info["limits"]["global"])["rate"]),
                    "X-RateLimit-Remaining": str(limit_info["limits"].get("role", limit_info["limits"]["global"])["remaining"]),
                    "X-RateLimit-Reset": str(int(time.time()) + 60),  # Reset in 60 seconds
                    "Retry-After": "60"
                }
            )
        
        # Process request
        response = await call_next(request)
        
        # Add rate limit headers to response
        limit_info = self.manager.get_limit_info(key, role)
        response.headers["X-RateLimit-Limit"] = str(limit_info["limits"].get("role", limit_info["limits"]["global"])["rate"])
        response.headers["X-RateLimit-Remaining"] = str(limit_info["limits"].get("role", limit_info["limits"]["global"])["remaining"])
        
        return response


# Dependency for rate limiting specific endpoints
def rate_limit(
    rate: int = 10,
    per: int = 60,
    key_func: Optional[Callable] = None
):
    """
    Dependency to rate limit specific endpoints
    
    Args:
        rate: Number of requests allowed
        per: Time period in seconds
        key_func: Optional function to extract key from request
    """
    limiter = RateLimiter(rate=rate, per=per)
    
    async def check_limit(request: Request):
        # Extract key
        if key_func:
            key = key_func(request)
        else:
            # Default to IP address
            key = request.client.host if request.client else "unknown"
        
        # Check rate limit
        if not limiter.is_allowed(key):
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail="Rate limit exceeded",
                headers={
                    "Retry-After": str(per)
                }
            )
    
    return check_limit


# Background task to clean up old entries
async def cleanup_rate_limits():
    """Background task to clean up old rate limit entries"""
    while True:
        await asyncio.sleep(3600)  # Run every hour
        rate_limit_manager.cleanup_old_entries()
        logger.info("Cleaned up old rate limit entries")