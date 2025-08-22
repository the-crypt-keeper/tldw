"""
Circuit breaker implementation for resilient service calls.

Prevents cascading failures by temporarily disabling calls to failing services.
"""

import time
from loguru import logger
from typing import Callable, Any, Optional, TypeVar, Generic
from dataclasses import dataclass, field
from enum import Enum
import asyncio
from functools import wraps


T = TypeVar('T')


class CircuitState(Enum):
    """Circuit breaker states."""
    CLOSED = "closed"  # Normal operation
    OPEN = "open"      # Failure threshold exceeded, blocking calls
    HALF_OPEN = "half_open"  # Testing if service recovered


@dataclass
class CircuitBreakerConfig:
    """Configuration for circuit breaker."""
    failure_threshold: int = 5        # Failures before opening
    recovery_timeout: float = 60.0    # Seconds before trying half-open
    expected_exception: type = Exception  # Exception types to catch
    success_threshold: int = 2        # Successes needed to close from half-open
    # Exponential backoff settings
    enable_exponential_backoff: bool = True  # Enable exponential backoff
    backoff_multiplier: float = 2.0   # Multiplier for each retry
    max_recovery_timeout: float = 300.0  # Maximum recovery timeout (5 minutes)
    min_recovery_timeout: float = 30.0   # Minimum recovery timeout


@dataclass 
class CircuitBreakerState:
    """Tracks the current state of the circuit breaker."""
    state: CircuitState = CircuitState.CLOSED
    failure_count: int = 0
    success_count: int = 0
    last_failure_time: Optional[float] = None
    last_state_change: float = field(default_factory=time.time)
    consecutive_failures: int = 0  # Track consecutive failures for backoff
    current_recovery_timeout: Optional[float] = None  # Current backoff timeout


class CircuitBreaker(Generic[T]):
    """
    Circuit breaker pattern implementation.
    
    Prevents cascading failures by temporarily blocking calls to failing services.
    """
    
    def __init__(self, name: str, config: Optional[CircuitBreakerConfig] = None):
        """
        Initialize circuit breaker.
        
        Args:
            name: Name for logging and metrics
            config: Circuit breaker configuration
        """
        self.name = name
        self.config = config or CircuitBreakerConfig()
        self._state = CircuitBreakerState()
        self._state.current_recovery_timeout = self.config.recovery_timeout
        self._lock = asyncio.Lock()
        
        logger.info(f"Circuit breaker '{name}' initialized with threshold={self.config.failure_threshold}, "
                   f"exponential_backoff={self.config.enable_exponential_backoff}")
    
    @property
    def state(self) -> CircuitState:
        """Get current circuit state."""
        return self._state.state
    
    def _calculate_recovery_timeout(self) -> float:
        """Calculate recovery timeout with exponential backoff."""
        if not self.config.enable_exponential_backoff:
            return self.config.recovery_timeout
        
        # Calculate exponential backoff based on consecutive failures
        backoff_timeout = self.config.recovery_timeout * (
            self.config.backoff_multiplier ** self._state.consecutive_failures
        )
        
        # Apply min/max bounds
        backoff_timeout = max(self.config.min_recovery_timeout, backoff_timeout)
        backoff_timeout = min(self.config.max_recovery_timeout, backoff_timeout)
        
        return backoff_timeout
    
    def _should_attempt_reset(self) -> bool:
        """Check if we should try to reset from open state."""
        if self._state.state != CircuitState.OPEN:
            return False
            
        if self._state.last_failure_time is None:
            return False
        
        # Use the current recovery timeout (which may be backed off)
        recovery_timeout = self._state.current_recovery_timeout or self.config.recovery_timeout
        return time.time() - self._state.last_failure_time >= recovery_timeout
    
    async def _record_success(self):
        """Record a successful call."""
        async with self._lock:
            self._state.failure_count = 0
            
            if self._state.state == CircuitState.HALF_OPEN:
                self._state.success_count += 1
                
                if self._state.success_count >= self.config.success_threshold:
                    self._state.state = CircuitState.CLOSED
                    self._state.success_count = 0
                    self._state.consecutive_failures = 0  # Reset consecutive failures
                    self._state.current_recovery_timeout = self.config.recovery_timeout  # Reset timeout
                    self._state.last_state_change = time.time()
                    logger.info(f"Circuit breaker '{self.name}' closed after successful recovery")
    
    async def _record_failure(self):
        """Record a failed call."""
        async with self._lock:
            self._state.failure_count += 1
            self._state.last_failure_time = time.time()
            self._state.success_count = 0
            
            if self._state.state == CircuitState.HALF_OPEN:
                self._state.state = CircuitState.OPEN
                self._state.consecutive_failures += 1  # Increment consecutive failures
                self._state.current_recovery_timeout = self._calculate_recovery_timeout()
                self._state.last_state_change = time.time()
                logger.warning(f"Circuit breaker '{self.name}' reopened after failure in half-open state. "
                             f"Next retry in {self._state.current_recovery_timeout:.1f}s")
                
            elif (self._state.state == CircuitState.CLOSED and 
                  self._state.failure_count >= self.config.failure_threshold):
                self._state.state = CircuitState.OPEN
                self._state.consecutive_failures = 1  # First consecutive failure
                self._state.current_recovery_timeout = self._calculate_recovery_timeout()
                self._state.last_state_change = time.time()
                logger.error(f"Circuit breaker '{self.name}' opened after {self._state.failure_count} failures. "
                           f"Will retry in {self._state.current_recovery_timeout:.1f}s")
    
    async def _transition_to_half_open(self):
        """Transition to half-open state for testing."""
        async with self._lock:
            if self._state.state == CircuitState.OPEN:
                self._state.state = CircuitState.HALF_OPEN
                self._state.last_state_change = time.time()
                self._state.failure_count = 0
                self._state.success_count = 0
                logger.info(f"Circuit breaker '{self.name}' half-open, testing recovery")
    
    async def call_async(self, func: Callable[..., T], *args, **kwargs) -> T:
        """
        Execute an async function with circuit breaker protection.
        
        Args:
            func: Async function to call
            *args, **kwargs: Arguments for the function
            
        Returns:
            Function result
            
        Raises:
            CircuitBreakerOpenError: If circuit is open
            Original exception: If function fails
        """
        # Check if we should attempt reset
        if self._should_attempt_reset():
            await self._transition_to_half_open()
        
        # Check circuit state
        if self._state.state == CircuitState.OPEN:
            raise CircuitBreakerOpenError(
                f"Circuit breaker '{self.name}' is open. "
                f"Service unavailable for {time.time() - self._state.last_failure_time:.1f}s"
            )
        
        try:
            # Make the call
            result = await func(*args, **kwargs)
            await self._record_success()
            return result
            
        except self.config.expected_exception as e:
            await self._record_failure()
            raise
    
    def call_sync(self, func: Callable[..., T], *args, **kwargs) -> T:
        """
        Execute a sync function with circuit breaker protection.
        
        Args:
            func: Sync function to call
            *args, **kwargs: Arguments for the function
            
        Returns:
            Function result
            
        Raises:
            CircuitBreakerOpenError: If circuit is open
            Original exception: If function fails
        """
        # Run async logic in sync context
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            # Wrap sync function as async
            async def async_wrapper():
                return func(*args, **kwargs)
            
            return loop.run_until_complete(self.call_async(async_wrapper))
        finally:
            loop.close()
    
    def get_stats(self) -> dict:
        """Get circuit breaker statistics."""
        stats = {
            "name": self.name,
            "state": self._state.state.value,
            "failure_count": self._state.failure_count,
            "success_count": self._state.success_count,
            "consecutive_failures": self._state.consecutive_failures,
            "last_failure_time": self._state.last_failure_time,
            "last_state_change": self._state.last_state_change,
            "current_recovery_timeout": self._state.current_recovery_timeout,
            "config": {
                "failure_threshold": self.config.failure_threshold,
                "recovery_timeout": self.config.recovery_timeout,
                "success_threshold": self.config.success_threshold,
                "enable_exponential_backoff": self.config.enable_exponential_backoff,
                "backoff_multiplier": self.config.backoff_multiplier,
                "max_recovery_timeout": self.config.max_recovery_timeout,
                "min_recovery_timeout": self.config.min_recovery_timeout
            }
        }
        
        if self._state.state == CircuitState.OPEN and self._state.last_failure_time:
            current_timeout = self._state.current_recovery_timeout or self.config.recovery_timeout
            stats["time_until_recovery"] = max(
                0, 
                current_timeout - (time.time() - self._state.last_failure_time)
            )
        
        return stats


class CircuitBreakerOpenError(Exception):
    """Raised when circuit breaker is open."""
    pass


def circuit_breaker(name: str, **config_kwargs):
    """
    Decorator to add circuit breaker protection to a function.
    
    Args:
        name: Circuit breaker name
        **config_kwargs: Configuration parameters
        
    Example:
        @circuit_breaker("external_api", failure_threshold=3, recovery_timeout=30)
        async def call_external_api():
            ...
    """
    config = CircuitBreakerConfig(**config_kwargs)
    breaker = CircuitBreaker(name, config)
    
    def decorator(func):
        if asyncio.iscoroutinefunction(func):
            @wraps(func)
            async def async_wrapper(*args, **kwargs):
                return await breaker.call_async(func, *args, **kwargs)
            return async_wrapper
        else:
            @wraps(func)
            def sync_wrapper(*args, **kwargs):
                return breaker.call_sync(func, *args, **kwargs)
            return sync_wrapper
    
    return decorator


# Global circuit breaker registry
_circuit_breakers: dict[str, CircuitBreaker] = {}


def get_circuit_breaker(name: str, config: Optional[CircuitBreakerConfig] = None) -> CircuitBreaker:
    """
    Get or create a named circuit breaker.
    
    Args:
        name: Circuit breaker name
        config: Configuration (used only for new breakers)
        
    Returns:
        Circuit breaker instance
    """
    if name not in _circuit_breakers:
        _circuit_breakers[name] = CircuitBreaker(name, config)
    return _circuit_breakers[name]


def get_all_circuit_breaker_stats() -> dict:
    """Get statistics for all circuit breakers."""
    return {
        name: breaker.get_stats() 
        for name, breaker in _circuit_breakers.items()
    }