# circuit_breaker.py
# Circuit breaker pattern implementation for embeddings service
# Provides fault tolerance and prevents cascading failures

import asyncio
import time
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Callable, Dict, Optional, TypeVar, Union
from functools import wraps
import threading

from loguru import logger
from prometheus_client import Counter, Gauge

# Type variables for generic typing
T = TypeVar('T')

# Prometheus metrics
CIRCUIT_BREAKER_STATE = Gauge(
    'circuit_breaker_state',
    'Current state of circuit breaker (0=closed, 1=open, 2=half_open)',
    ['service', 'operation']
)

CIRCUIT_BREAKER_FAILURES = Counter(
    'circuit_breaker_failures_total',
    'Total number of failures tracked by circuit breaker',
    ['service', 'operation']
)

CIRCUIT_BREAKER_SUCCESSES = Counter(
    'circuit_breaker_successes_total',
    'Total number of successes tracked by circuit breaker',
    ['service', 'operation']
)

CIRCUIT_BREAKER_TRIPS = Counter(
    'circuit_breaker_trips_total',
    'Total number of times circuit breaker has tripped',
    ['service', 'operation']
)


class CircuitState(Enum):
    """Circuit breaker states"""
    CLOSED = 0      # Normal operation, requests pass through
    OPEN = 1        # Circuit broken, requests fail immediately
    HALF_OPEN = 2   # Testing if service recovered


class CircuitBreakerError(Exception):
    """Raised when circuit breaker is open"""
    pass


class CircuitBreaker:
    """
    Circuit breaker implementation for fault tolerance.
    
    The circuit breaker pattern prevents cascading failures by failing fast
    when a service is unavailable, giving it time to recover.
    
    States:
    - CLOSED: Normal operation, requests pass through
    - OPEN: Service is failing, requests fail immediately
    - HALF_OPEN: Testing if service has recovered
    """
    
    def __init__(
        self,
        name: str,
        failure_threshold: int = 5,
        recovery_timeout: float = 60.0,
        expected_exception: type = Exception,
        success_threshold: int = 2,
        half_open_max_calls: int = 3
    ):
        """
        Initialize circuit breaker.
        
        Args:
            name: Identifier for this circuit breaker
            failure_threshold: Number of failures before opening circuit
            recovery_timeout: Seconds to wait before trying half-open
            expected_exception: Exception type to catch (others pass through)
            success_threshold: Successes needed in half-open to close circuit
            half_open_max_calls: Max concurrent calls in half-open state
        """
        self.name = name
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.expected_exception = expected_exception
        self.success_threshold = success_threshold
        self.half_open_max_calls = half_open_max_calls
        
        # State management
        self._state = CircuitState.CLOSED
        self._failure_count = 0
        self._success_count = 0
        self._last_failure_time: Optional[float] = None
        self._half_open_calls = 0
        
        # Thread safety
        self._lock = threading.RLock()
        
        logger.info(
            f"Circuit breaker '{name}' initialized: "
            f"failure_threshold={failure_threshold}, "
            f"recovery_timeout={recovery_timeout}s"
        )
    
    @property
    def state(self) -> CircuitState:
        """Get current circuit state, updating if necessary"""
        with self._lock:
            self._update_state()
            return self._state
    
    @property
    def is_closed(self) -> bool:
        """Check if circuit is closed (normal operation)"""
        return self.state == CircuitState.CLOSED
    
    @property
    def is_open(self) -> bool:
        """Check if circuit is open (failing)"""
        return self.state == CircuitState.OPEN
    
    @property
    def is_half_open(self) -> bool:
        """Check if circuit is half-open (testing)"""
        return self.state == CircuitState.HALF_OPEN
    
    def _update_state(self):
        """Update circuit state based on current conditions"""
        if self._state == CircuitState.OPEN:
            if self._last_failure_time and \
               time.time() - self._last_failure_time >= self.recovery_timeout:
                # Enough time has passed, try half-open
                self._transition_to_half_open()
    
    def _transition_to_closed(self):
        """Transition to closed state"""
        self._state = CircuitState.CLOSED
        self._failure_count = 0
        self._success_count = 0
        self._half_open_calls = 0
        
        CIRCUIT_BREAKER_STATE.labels(
            service=self.name,
            operation="state_change"
        ).set(CircuitState.CLOSED.value)
        
        logger.info(f"Circuit breaker '{self.name}' CLOSED")
    
    def _transition_to_open(self):
        """Transition to open state"""
        self._state = CircuitState.OPEN
        self._last_failure_time = time.time()
        self._failure_count = 0
        self._success_count = 0
        self._half_open_calls = 0
        
        CIRCUIT_BREAKER_STATE.labels(
            service=self.name,
            operation="state_change"
        ).set(CircuitState.OPEN.value)
        
        CIRCUIT_BREAKER_TRIPS.labels(
            service=self.name,
            operation="trip"
        ).inc()
        
        logger.warning(
            f"Circuit breaker '{self.name}' OPEN - "
            f"will retry in {self.recovery_timeout}s"
        )
    
    def _transition_to_half_open(self):
        """Transition to half-open state"""
        self._state = CircuitState.HALF_OPEN
        self._success_count = 0
        self._failure_count = 0
        self._half_open_calls = 0
        
        CIRCUIT_BREAKER_STATE.labels(
            service=self.name,
            operation="state_change"
        ).set(CircuitState.HALF_OPEN.value)
        
        logger.info(f"Circuit breaker '{self.name}' HALF-OPEN (testing recovery)")
    
    def call(self, func: Callable[..., T], *args, **kwargs) -> T:
        """
        Execute function through circuit breaker.
        
        Args:
            func: Function to execute
            *args: Function arguments
            **kwargs: Function keyword arguments
            
        Returns:
            Function result
            
        Raises:
            CircuitBreakerError: If circuit is open
            Exception: If function fails
        """
        with self._lock:
            self._update_state()
            
            if self._state == CircuitState.OPEN:
                raise CircuitBreakerError(
                    f"Circuit breaker '{self.name}' is OPEN"
                )
            
            if self._state == CircuitState.HALF_OPEN:
                if self._half_open_calls >= self.half_open_max_calls:
                    raise CircuitBreakerError(
                        f"Circuit breaker '{self.name}' half-open call limit reached"
                    )
                self._half_open_calls += 1
        
        try:
            result = func(*args, **kwargs)
            self._on_success()
            return result
        except self.expected_exception as e:
            self._on_failure()
            raise
    
    async def call_async(self, func: Callable[..., T], *args, **kwargs) -> T:
        """
        Execute async function through circuit breaker.
        
        Args:
            func: Async function to execute
            *args: Function arguments
            **kwargs: Function keyword arguments
            
        Returns:
            Function result
            
        Raises:
            CircuitBreakerError: If circuit is open
            Exception: If function fails
        """
        with self._lock:
            self._update_state()
            
            if self._state == CircuitState.OPEN:
                raise CircuitBreakerError(
                    f"Circuit breaker '{self.name}' is OPEN"
                )
            
            if self._state == CircuitState.HALF_OPEN:
                if self._half_open_calls >= self.half_open_max_calls:
                    raise CircuitBreakerError(
                        f"Circuit breaker '{self.name}' half-open call limit reached"
                    )
                self._half_open_calls += 1
        
        try:
            result = await func(*args, **kwargs)
            self._on_success()
            return result
        except self.expected_exception as e:
            self._on_failure()
            raise
    
    def _on_success(self):
        """Handle successful call"""
        with self._lock:
            CIRCUIT_BREAKER_SUCCESSES.labels(
                service=self.name,
                operation="call"
            ).inc()
            
            if self._state == CircuitState.HALF_OPEN:
                self._success_count += 1
                if self._success_count >= self.success_threshold:
                    self._transition_to_closed()
            elif self._state == CircuitState.CLOSED:
                # Reset failure count on success in closed state
                self._failure_count = 0
    
    def _on_failure(self):
        """Handle failed call"""
        with self._lock:
            CIRCUIT_BREAKER_FAILURES.labels(
                service=self.name,
                operation="call"
            ).inc()
            
            self._failure_count += 1
            self._last_failure_time = time.time()
            
            if self._state == CircuitState.HALF_OPEN:
                # Single failure in half-open reopens circuit
                self._transition_to_open()
            elif self._state == CircuitState.CLOSED:
                if self._failure_count >= self.failure_threshold:
                    self._transition_to_open()
    
    def reset(self):
        """Manually reset circuit breaker to closed state"""
        with self._lock:
            self._transition_to_closed()
            logger.info(f"Circuit breaker '{self.name}' manually reset")
    
    def get_status(self) -> Dict[str, Any]:
        """Get circuit breaker status"""
        with self._lock:
            return {
                "name": self.name,
                "state": self._state.name,
                "failure_count": self._failure_count,
                "success_count": self._success_count,
                "last_failure_time": self._last_failure_time,
                "half_open_calls": self._half_open_calls,
                "settings": {
                    "failure_threshold": self.failure_threshold,
                    "recovery_timeout": self.recovery_timeout,
                    "success_threshold": self.success_threshold,
                    "half_open_max_calls": self.half_open_max_calls
                }
            }


def circuit_breaker(
    name: Optional[str] = None,
    failure_threshold: int = 5,
    recovery_timeout: float = 60.0,
    expected_exception: type = Exception,
    success_threshold: int = 2
):
    """
    Decorator to apply circuit breaker pattern to a function.
    
    Args:
        name: Circuit breaker name (defaults to function name)
        failure_threshold: Failures before opening circuit
        recovery_timeout: Seconds before trying half-open
        expected_exception: Exception type to catch
        success_threshold: Successes needed to close from half-open
    
    Example:
        @circuit_breaker(name="openai_api", failure_threshold=3)
        async def call_openai_api():
            # API call that might fail
            pass
    """
    def decorator(func):
        breaker_name = name or f"{func.__module__}.{func.__name__}"
        breaker = CircuitBreaker(
            name=breaker_name,
            failure_threshold=failure_threshold,
            recovery_timeout=recovery_timeout,
            expected_exception=expected_exception,
            success_threshold=success_threshold
        )
        
        if asyncio.iscoroutinefunction(func):
            @wraps(func)
            async def async_wrapper(*args, **kwargs):
                return await breaker.call_async(func, *args, **kwargs)
            return async_wrapper
        else:
            @wraps(func)
            def sync_wrapper(*args, **kwargs):
                return breaker.call(func, *args, **kwargs)
            return sync_wrapper
    
    return decorator


# Global circuit breaker registry for management
class CircuitBreakerRegistry:
    """Registry for managing multiple circuit breakers"""
    
    def __init__(self):
        self._breakers: Dict[str, CircuitBreaker] = {}
        self._lock = threading.Lock()
    
    def register(self, breaker: CircuitBreaker):
        """Register a circuit breaker"""
        with self._lock:
            self._breakers[breaker.name] = breaker
    
    def get(self, name: str) -> Optional[CircuitBreaker]:
        """Get circuit breaker by name"""
        return self._breakers.get(name)
    
    def get_all_status(self) -> Dict[str, Dict[str, Any]]:
        """Get status of all circuit breakers"""
        with self._lock:
            return {
                name: breaker.get_status()
                for name, breaker in self._breakers.items()
            }
    
    def reset_all(self):
        """Reset all circuit breakers"""
        with self._lock:
            for breaker in self._breakers.values():
                breaker.reset()


# Global registry instance
registry = CircuitBreakerRegistry()