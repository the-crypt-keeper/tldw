"""
Circuit breaker middleware for LLM API calls.

Implements circuit breaker pattern to prevent cascading failures and
protect against repeatedly calling failing services.
"""

import time
import threading
from enum import Enum
from typing import Callable, Any, Optional, Dict, List
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from collections import deque
from loguru import logger


class CircuitState(Enum):
    """States of the circuit breaker."""
    CLOSED = "closed"  # Normal operation
    OPEN = "open"  # Failures exceeded threshold, blocking calls
    HALF_OPEN = "half_open"  # Testing if service recovered


@dataclass
class CircuitBreakerConfig:
    """Configuration for circuit breaker."""
    failure_threshold: int = 5  # Failures before opening
    success_threshold: int = 2  # Successes in half-open before closing
    timeout: float = 60.0  # Seconds before trying half-open
    window_size: int = 10  # Size of sliding window for failure rate
    failure_rate_threshold: float = 0.5  # Failure rate to open circuit
    

@dataclass
class CircuitStats:
    """Statistics for circuit breaker."""
    total_calls: int = 0
    successful_calls: int = 0
    failed_calls: int = 0
    blocked_calls: int = 0
    state_changes: int = 0
    last_failure_time: Optional[datetime] = None
    last_success_time: Optional[datetime] = None
    current_state: CircuitState = CircuitState.CLOSED


class CircuitBreaker:
    """
    Circuit breaker for a single service/provider.
    
    Monitors failures and temporarily blocks calls to failing services.
    """
    
    def __init__(self, name: str, config: CircuitBreakerConfig):
        """
        Initialize circuit breaker.
        
        Args:
            name: Name of the service/provider
            config: Circuit breaker configuration
        """
        self.name = name
        self.config = config
        self.state = CircuitState.CLOSED
        self.failure_count = 0
        self.success_count = 0
        self.last_failure_time = None
        self.last_state_change = time.time()
        self.call_history = deque(maxlen=config.window_size)
        self.stats = CircuitStats()
        self._lock = threading.Lock()
    
    def call(self, func: Callable, *args, **kwargs) -> Any:
        """
        Execute a function through the circuit breaker.
        
        Args:
            func: Function to call
            *args: Positional arguments for func
            **kwargs: Keyword arguments for func
            
        Returns:
            Result of func
            
        Raises:
            CircuitOpenError: If circuit is open
            Original exception: If func fails
        """
        with self._lock:
            self.stats.total_calls += 1
            
            # Check circuit state
            if self.state == CircuitState.OPEN:
                if self._should_attempt_reset():
                    self._transition_to_half_open()
                else:
                    self.stats.blocked_calls += 1
                    raise CircuitOpenError(
                        f"Circuit breaker is OPEN for {self.name}"
                    )
            
        # Execute the function
        try:
            result = func(*args, **kwargs)
            self._on_success()
            return result
        except Exception as e:
            self._on_failure()
            raise
    
    async def async_call(self, func: Callable, *args, **kwargs) -> Any:
        """
        Execute an async function through the circuit breaker.
        
        Args:
            func: Async function to call
            *args: Positional arguments for func
            **kwargs: Keyword arguments for func
            
        Returns:
            Result of func
            
        Raises:
            CircuitOpenError: If circuit is open
            Original exception: If func fails
        """
        with self._lock:
            self.stats.total_calls += 1
            
            # Check circuit state
            if self.state == CircuitState.OPEN:
                if self._should_attempt_reset():
                    self._transition_to_half_open()
                else:
                    self.stats.blocked_calls += 1
                    raise CircuitOpenError(
                        f"Circuit breaker is OPEN for {self.name}"
                    )
        
        # Execute the async function
        try:
            result = await func(*args, **kwargs)
            self._on_success()
            return result
        except Exception as e:
            self._on_failure()
            raise
    
    def _on_success(self):
        """Handle successful call."""
        with self._lock:
            self.stats.successful_calls += 1
            self.stats.last_success_time = datetime.now()
            self.call_history.append(True)
            
            if self.state == CircuitState.HALF_OPEN:
                self.success_count += 1
                if self.success_count >= self.config.success_threshold:
                    self._transition_to_closed()
            elif self.state == CircuitState.CLOSED:
                self.failure_count = 0  # Reset consecutive failures
    
    def _on_failure(self):
        """Handle failed call."""
        with self._lock:
            self.stats.failed_calls += 1
            self.stats.last_failure_time = datetime.now()
            self.last_failure_time = time.time()
            self.call_history.append(False)
            
            if self.state == CircuitState.HALF_OPEN:
                self._transition_to_open()
            elif self.state == CircuitState.CLOSED:
                self.failure_count += 1
                
                # Check if we should open the circuit
                if self._should_open_circuit():
                    self._transition_to_open()
    
    def _should_open_circuit(self) -> bool:
        """
        Check if circuit should be opened based on failures.
        
        Returns:
            True if circuit should open
        """
        # Check consecutive failure threshold
        if self.failure_count >= self.config.failure_threshold:
            return True
        
        # Check failure rate in sliding window
        if len(self.call_history) >= self.config.window_size:
            failure_rate = self.call_history.count(False) / len(self.call_history)
            if failure_rate >= self.config.failure_rate_threshold:
                return True
        
        return False
    
    def _should_attempt_reset(self) -> bool:
        """
        Check if enough time has passed to attempt reset.
        
        Returns:
            True if should attempt reset
        """
        if self.last_failure_time is None:
            return True
        
        time_since_failure = time.time() - self.last_failure_time
        return time_since_failure >= self.config.timeout
    
    def _transition_to_open(self):
        """Transition to OPEN state."""
        logger.warning(f"Circuit breaker for {self.name} is now OPEN")
        self.state = CircuitState.OPEN
        self.stats.current_state = CircuitState.OPEN
        self.stats.state_changes += 1
        self.last_state_change = time.time()
        self.failure_count = 0
        self.success_count = 0
    
    def _transition_to_closed(self):
        """Transition to CLOSED state."""
        logger.info(f"Circuit breaker for {self.name} is now CLOSED")
        self.state = CircuitState.CLOSED
        self.stats.current_state = CircuitState.CLOSED
        self.stats.state_changes += 1
        self.last_state_change = time.time()
        self.failure_count = 0
        self.success_count = 0
    
    def _transition_to_half_open(self):
        """Transition to HALF_OPEN state."""
        logger.info(f"Circuit breaker for {self.name} is now HALF_OPEN")
        self.state = CircuitState.HALF_OPEN
        self.stats.current_state = CircuitState.HALF_OPEN
        self.stats.state_changes += 1
        self.last_state_change = time.time()
        self.success_count = 0
    
    def reset(self):
        """Manually reset the circuit breaker to CLOSED state."""
        with self._lock:
            logger.info(f"Manually resetting circuit breaker for {self.name}")
            self._transition_to_closed()
            self.call_history.clear()
    
    def get_state(self) -> CircuitState:
        """Get current circuit state."""
        return self.state
    
    def get_stats(self) -> Dict[str, Any]:
        """Get circuit breaker statistics."""
        with self._lock:
            return {
                'name': self.name,
                'state': self.state.value,
                'total_calls': self.stats.total_calls,
                'successful_calls': self.stats.successful_calls,
                'failed_calls': self.stats.failed_calls,
                'blocked_calls': self.stats.blocked_calls,
                'state_changes': self.stats.state_changes,
                'failure_count': self.failure_count,
                'success_count': self.success_count,
                'last_failure': self.stats.last_failure_time.isoformat() if self.stats.last_failure_time else None,
                'last_success': self.stats.last_success_time.isoformat() if self.stats.last_success_time else None,
                'failure_rate': self.call_history.count(False) / len(self.call_history) if self.call_history else 0,
            }


class CircuitOpenError(Exception):
    """Exception raised when circuit breaker is open."""
    pass


class CircuitBreakerManager:
    """
    Manager for multiple circuit breakers.
    
    Manages circuit breakers for different providers/services.
    """
    
    # Default configurations for providers
    PROVIDER_DEFAULTS = {
        'openai': CircuitBreakerConfig(
            failure_threshold=5,
            success_threshold=3,
            timeout=60.0,
            window_size=20,
            failure_rate_threshold=0.5
        ),
        'anthropic': CircuitBreakerConfig(
            failure_threshold=5,
            success_threshold=3,
            timeout=60.0,
            window_size=20,
            failure_rate_threshold=0.5
        ),
        'cohere': CircuitBreakerConfig(
            failure_threshold=3,
            success_threshold=2,
            timeout=30.0,
            window_size=10,
            failure_rate_threshold=0.6
        ),
        # More tolerant for less reliable providers
        'groq': CircuitBreakerConfig(
            failure_threshold=10,
            success_threshold=5,
            timeout=120.0,
            window_size=30,
            failure_rate_threshold=0.7
        ),
        # Local providers should rarely circuit break
        'ollama': CircuitBreakerConfig(
            failure_threshold=20,
            success_threshold=2,
            timeout=30.0,
            window_size=50,
            failure_rate_threshold=0.9
        ),
    }
    
    def __init__(self):
        """Initialize the circuit breaker manager."""
        self.breakers: Dict[str, CircuitBreaker] = {}
        self._lock = threading.Lock()
        
        # Initialize default circuit breakers
        for provider, config in self.PROVIDER_DEFAULTS.items():
            self.add_breaker(provider, config)
    
    def add_breaker(self, name: str, config: CircuitBreakerConfig) -> CircuitBreaker:
        """
        Add a new circuit breaker.
        
        Args:
            name: Name of the service/provider
            config: Circuit breaker configuration
            
        Returns:
            The created circuit breaker
        """
        with self._lock:
            breaker = CircuitBreaker(name, config)
            self.breakers[name] = breaker
            logger.info(f"Added circuit breaker for {name}")
            return breaker
    
    def get_breaker(self, name: str) -> Optional[CircuitBreaker]:
        """
        Get a circuit breaker by name.
        
        Args:
            name: Name of the service/provider
            
        Returns:
            Circuit breaker or None if not found
        """
        return self.breakers.get(name)
    
    def call(self, name: str, func: Callable, *args, **kwargs) -> Any:
        """
        Execute a function through a circuit breaker.
        
        Args:
            name: Name of the service/provider
            func: Function to call
            *args: Positional arguments for func
            **kwargs: Keyword arguments for func
            
        Returns:
            Result of func
            
        Raises:
            CircuitOpenError: If circuit is open
            ValueError: If breaker not found
        """
        breaker = self.get_breaker(name)
        if not breaker:
            # No circuit breaker, just call the function
            logger.debug(f"No circuit breaker for {name}, calling directly")
            return func(*args, **kwargs)
        
        return breaker.call(func, *args, **kwargs)
    
    async def async_call(self, name: str, func: Callable, *args, **kwargs) -> Any:
        """
        Execute an async function through a circuit breaker.
        
        Args:
            name: Name of the service/provider
            func: Async function to call
            *args: Positional arguments for func
            **kwargs: Keyword arguments for func
            
        Returns:
            Result of func
            
        Raises:
            CircuitOpenError: If circuit is open
            ValueError: If breaker not found
        """
        breaker = self.get_breaker(name)
        if not breaker:
            # No circuit breaker, just call the function
            logger.debug(f"No circuit breaker for {name}, calling directly")
            return await func(*args, **kwargs)
        
        return await breaker.async_call(func, *args, **kwargs)
    
    def reset(self, name: Optional[str] = None):
        """
        Reset circuit breaker(s).
        
        Args:
            name: Optional name of specific breaker to reset
        """
        if name:
            breaker = self.get_breaker(name)
            if breaker:
                breaker.reset()
        else:
            for breaker in self.breakers.values():
                breaker.reset()
    
    def get_all_stats(self) -> Dict[str, Dict[str, Any]]:
        """Get statistics for all circuit breakers."""
        return {
            name: breaker.get_stats()
            for name, breaker in self.breakers.items()
        }
    
    def get_open_circuits(self) -> List[str]:
        """Get list of currently open circuits."""
        return [
            name for name, breaker in self.breakers.items()
            if breaker.get_state() == CircuitState.OPEN
        ]


# Global circuit breaker manager
_manager = None


def get_circuit_breaker_manager() -> CircuitBreakerManager:
    """Get the global circuit breaker manager."""
    global _manager
    if _manager is None:
        _manager = CircuitBreakerManager()
    return _manager