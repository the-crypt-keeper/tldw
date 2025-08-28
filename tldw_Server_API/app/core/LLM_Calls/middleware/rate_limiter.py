"""
Rate limiting middleware for LLM API calls.

Implements token bucket algorithm for rate limiting API requests to prevent
overuse and comply with provider limits.
"""

import time
import threading
from typing import Dict, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from collections import defaultdict
from loguru import logger


@dataclass
class RateLimitConfig:
    """Configuration for rate limiting."""
    requests_per_minute: int = 60
    requests_per_hour: Optional[int] = None
    requests_per_day: Optional[int] = None
    tokens_per_minute: Optional[int] = None
    tokens_per_hour: Optional[int] = None
    burst_size: int = 10  # Allow burst of requests
    

@dataclass 
class TokenBucket:
    """Token bucket for rate limiting."""
    capacity: int
    refill_rate: float  # Tokens per second
    tokens: float = field(init=False)
    last_refill: float = field(init=False)
    lock: threading.Lock = field(default_factory=threading.Lock, init=False)
    
    def __post_init__(self):
        self.tokens = float(self.capacity)
        self.last_refill = time.time()
    
    def consume(self, tokens: int = 1) -> bool:
        """
        Try to consume tokens from the bucket.
        
        Args:
            tokens: Number of tokens to consume
            
        Returns:
            True if tokens were consumed, False if not enough tokens
        """
        with self.lock:
            self._refill()
            
            if self.tokens >= tokens:
                self.tokens -= tokens
                return True
            return False
    
    def wait_time(self, tokens: int = 1) -> float:
        """
        Calculate wait time until tokens are available.
        
        Args:
            tokens: Number of tokens needed
            
        Returns:
            Seconds to wait (0 if tokens available now)
        """
        with self.lock:
            self._refill()
            
            if self.tokens >= tokens:
                return 0.0
            
            deficit = tokens - self.tokens
            return deficit / self.refill_rate
    
    def _refill(self):
        """Refill tokens based on elapsed time."""
        now = time.time()
        elapsed = now - self.last_refill
        
        # Add tokens based on refill rate
        tokens_to_add = elapsed * self.refill_rate
        self.tokens = min(self.capacity, self.tokens + tokens_to_add)
        self.last_refill = now


class RateLimiter:
    """
    Rate limiter for LLM API calls.
    
    Supports per-provider and per-user rate limiting with multiple time windows.
    """
    
    # Default rate limits for providers (can be overridden)
    PROVIDER_DEFAULTS = {
        'openai': RateLimitConfig(
            requests_per_minute=3000,
            tokens_per_minute=90000,
            burst_size=20
        ),
        'anthropic': RateLimitConfig(
            requests_per_minute=1000,
            tokens_per_minute=100000,
            burst_size=15
        ),
        'cohere': RateLimitConfig(
            requests_per_minute=100,
            burst_size=10
        ),
        'groq': RateLimitConfig(
            requests_per_minute=30,
            burst_size=5
        ),
        'mistral': RateLimitConfig(
            requests_per_minute=120,
            burst_size=10
        ),
        'deepseek': RateLimitConfig(
            requests_per_minute=60,
            burst_size=10
        ),
        'google': RateLimitConfig(
            requests_per_minute=60,
            burst_size=10
        ),
        # Local providers typically don't need rate limiting
        'ollama': RateLimitConfig(
            requests_per_minute=10000,  # Effectively unlimited
            burst_size=100
        ),
    }
    
    def __init__(self):
        """Initialize the rate limiter."""
        self.provider_configs: Dict[str, RateLimitConfig] = {}
        self.provider_buckets: Dict[str, Dict[str, TokenBucket]] = defaultdict(dict)
        self.user_buckets: Dict[str, Dict[str, TokenBucket]] = defaultdict(dict)
        self.stats: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
        self._lock = threading.Lock()
        
        # Initialize default provider configs
        for provider, config in self.PROVIDER_DEFAULTS.items():
            self.set_provider_config(provider, config)
    
    def set_provider_config(self, provider: str, config: RateLimitConfig):
        """
        Set rate limit configuration for a provider.
        
        Args:
            provider: Provider name
            config: Rate limit configuration
        """
        with self._lock:
            self.provider_configs[provider] = config
            
            # Create token buckets for different time windows
            buckets = {}
            
            # Requests per minute bucket
            if config.requests_per_minute:
                buckets['rpm'] = TokenBucket(
                    capacity=config.burst_size,
                    refill_rate=config.requests_per_minute / 60.0
                )
            
            # Requests per hour bucket
            if config.requests_per_hour:
                buckets['rph'] = TokenBucket(
                    capacity=config.burst_size * 2,
                    refill_rate=config.requests_per_hour / 3600.0
                )
            
            # Requests per day bucket
            if config.requests_per_day:
                buckets['rpd'] = TokenBucket(
                    capacity=config.burst_size * 3,
                    refill_rate=config.requests_per_day / 86400.0
                )
            
            # Tokens per minute bucket
            if config.tokens_per_minute:
                buckets['tpm'] = TokenBucket(
                    capacity=config.tokens_per_minute,
                    refill_rate=config.tokens_per_minute / 60.0
                )
            
            # Tokens per hour bucket
            if config.tokens_per_hour:
                buckets['tph'] = TokenBucket(
                    capacity=config.tokens_per_hour,
                    refill_rate=config.tokens_per_hour / 3600.0
                )
            
            self.provider_buckets[provider] = buckets
            logger.info(f"Rate limiter configured for {provider}")
    
    def check_rate_limit(
        self, 
        provider: str, 
        user_id: Optional[str] = None,
        token_count: Optional[int] = None
    ) -> Tuple[bool, Optional[float]]:
        """
        Check if a request is allowed under rate limits.
        
        Args:
            provider: Provider name
            user_id: Optional user identifier
            token_count: Optional token count for the request
            
        Returns:
            Tuple of (allowed, wait_time_seconds)
        """
        # Get provider config
        config = self.provider_configs.get(provider)
        if not config:
            # No rate limit configured, allow
            return True, None
        
        # Get provider buckets
        buckets = self.provider_buckets.get(provider, {})
        
        # Check all buckets
        max_wait = 0.0
        
        # Check request buckets
        for bucket_name in ['rpm', 'rph', 'rpd']:
            bucket = buckets.get(bucket_name)
            if bucket:
                wait_time = bucket.wait_time(1)
                if wait_time > 0:
                    max_wait = max(max_wait, wait_time)
        
        # Check token buckets if token count provided
        if token_count:
            for bucket_name in ['tpm', 'tph']:
                bucket = buckets.get(bucket_name)
                if bucket:
                    wait_time = bucket.wait_time(token_count)
                    if wait_time > 0:
                        max_wait = max(max_wait, wait_time)
        
        # If we need to wait, return the wait time
        if max_wait > 0:
            logger.debug(f"Rate limit hit for {provider}: wait {max_wait:.2f}s")
            self.stats[provider]['rate_limited'] += 1
            return False, max_wait
        
        # Consume tokens from all buckets
        for bucket_name in ['rpm', 'rph', 'rpd']:
            bucket = buckets.get(bucket_name)
            if bucket:
                bucket.consume(1)
        
        if token_count:
            for bucket_name in ['tpm', 'tph']:
                bucket = buckets.get(bucket_name)
                if bucket:
                    bucket.consume(token_count)
        
        # Update stats
        self.stats[provider]['requests'] += 1
        if token_count:
            self.stats[provider]['tokens'] += token_count
        
        return True, None
    
    def wait_if_needed(
        self,
        provider: str,
        user_id: Optional[str] = None,
        token_count: Optional[int] = None,
        max_wait: float = 60.0
    ) -> bool:
        """
        Wait if rate limited, up to max_wait seconds.
        
        Args:
            provider: Provider name
            user_id: Optional user identifier
            token_count: Optional token count
            max_wait: Maximum seconds to wait
            
        Returns:
            True if request can proceed, False if wait exceeded max
        """
        allowed, wait_time = self.check_rate_limit(provider, user_id, token_count)
        
        if allowed:
            return True
        
        if wait_time and wait_time <= max_wait:
            logger.info(f"Rate limited, waiting {wait_time:.2f}s for {provider}")
            time.sleep(wait_time)
            # Try again after waiting
            allowed, _ = self.check_rate_limit(provider, user_id, token_count)
            return allowed
        
        logger.warning(f"Rate limit wait time {wait_time}s exceeds max {max_wait}s for {provider}")
        return False
    
    def get_stats(self, provider: Optional[str] = None) -> Dict[str, Any]:
        """
        Get rate limiting statistics.
        
        Args:
            provider: Optional provider to get stats for
            
        Returns:
            Dictionary of statistics
        """
        if provider:
            return dict(self.stats.get(provider, {}))
        
        return {
            provider: dict(stats)
            for provider, stats in self.stats.items()
        }
    
    def reset_stats(self, provider: Optional[str] = None):
        """
        Reset statistics.
        
        Args:
            provider: Optional provider to reset stats for
        """
        if provider:
            self.stats[provider].clear()
            logger.info(f"Reset rate limiter stats for {provider}")
        else:
            self.stats.clear()
            logger.info("Reset all rate limiter stats")
    
    def get_limits(self, provider: str) -> Optional[Dict[str, Any]]:
        """
        Get current rate limits for a provider.
        
        Args:
            provider: Provider name
            
        Returns:
            Dictionary of limits or None
        """
        config = self.provider_configs.get(provider)
        if not config:
            return None
        
        return {
            'requests_per_minute': config.requests_per_minute,
            'requests_per_hour': config.requests_per_hour,
            'requests_per_day': config.requests_per_day,
            'tokens_per_minute': config.tokens_per_minute,
            'tokens_per_hour': config.tokens_per_hour,
            'burst_size': config.burst_size,
        }
    
    def get_remaining(self, provider: str) -> Dict[str, float]:
        """
        Get remaining capacity for a provider.
        
        Args:
            provider: Provider name
            
        Returns:
            Dictionary of remaining tokens in each bucket
        """
        buckets = self.provider_buckets.get(provider, {})
        remaining = {}
        
        for name, bucket in buckets.items():
            with bucket.lock:
                bucket._refill()
                remaining[name] = bucket.tokens
        
        return remaining


# Global rate limiter instance
_rate_limiter = None


def get_rate_limiter() -> RateLimiter:
    """Get the global rate limiter instance."""
    global _rate_limiter
    if _rate_limiter is None:
        _rate_limiter = RateLimiter()
    return _rate_limiter