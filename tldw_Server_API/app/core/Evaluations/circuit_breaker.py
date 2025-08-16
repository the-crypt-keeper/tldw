"""
Circuit breaker implementation for LLM calls.

Provides fault tolerance and prevents cascading failures when external services
(LLMs, embeddings) are unavailable or slow.
"""

import asyncio
import time
from typing import Optional, Callable, Any, Dict
from enum import Enum
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from loguru import logger


class CircuitState(Enum):
    """Circuit breaker states."""
    CLOSED = "closed"  # Normal operation
    OPEN = "open"      # Circuit tripped, rejecting calls
    HALF_OPEN = "half_open"  # Testing if service recovered


@dataclass
class CircuitBreakerConfig:
    """Configuration for circuit breaker."""
    failure_threshold: int = 5  # Failures before opening circuit
    success_threshold: int = 2  # Successes in half-open before closing
    timeout: float = 30.0  # Timeout for each call (seconds)
    recovery_timeout: float = 60.0  # Time before trying half-open (seconds)
    expected_exception: type = Exception  # Exceptions that trigger the breaker


@dataclass
class CircuitBreakerStats:
    """Statistics for circuit breaker monitoring."""
    total_calls: int = 0
    successful_calls: int = 0
    failed_calls: int = 0
    rejected_calls: int = 0
    timeouts: int = 0
    last_failure_time: Optional[datetime] = None
    last_success_time: Optional[datetime] = None
    consecutive_failures: int = 0
    consecutive_successes: int = 0


class CircuitBreaker:
    """
    Circuit breaker for protecting against cascading failures.
    
    Implements the circuit breaker pattern with three states:
    - CLOSED: Normal operation, calls pass through
    - OPEN: Too many failures, calls are rejected immediately
    - HALF_OPEN: Testing recovery, limited calls allowed
    """
    
    def __init__(self, name: str, config: Optional[CircuitBreakerConfig] = None):
        """
        Initialize circuit breaker.
        
        Args:
            name: Name for logging and identification
            config: Circuit breaker configuration
        """
        self.name = name
        self.config = config or CircuitBreakerConfig()
        self.state = CircuitState.CLOSED
        self.stats = CircuitBreakerStats()
        self._state_changed_at = time.time()
        self._lock = asyncio.Lock()
    
    async def call(self, func: Callable, *args, **kwargs) -> Any:
        """
        Execute a function through the circuit breaker.
        
        Args:
            func: Function to call (can be async or sync)
            *args: Positional arguments for func
            **kwargs: Keyword arguments for func
            
        Returns:
            Result from func
            
        Raises:
            CircuitOpenError: If circuit is open
            TimeoutError: If call times out
        """
        async with self._lock:
            # Check circuit state
            if self.state == CircuitState.OPEN:
                if self._should_attempt_reset():
                    self._transition_to_half_open()
                else:
                    self.stats.rejected_calls += 1
                    raise CircuitOpenError(f"Circuit breaker {self.name} is OPEN")
        
        # Attempt the call
        self.stats.total_calls += 1
        
        try:
            # Handle both async and sync functions
            if asyncio.iscoroutinefunction(func):
                result = await asyncio.wait_for(
                    func(*args, **kwargs),
                    timeout=self.config.timeout
                )
            else:
                # Run sync function in executor with timeout
                loop = asyncio.get_event_loop()
                result = await asyncio.wait_for(
                    loop.run_in_executor(None, func, *args),
                    timeout=self.config.timeout
                )
            
            # Call succeeded
            await self._on_success()
            return result
            
        except asyncio.TimeoutError:
            self.stats.timeouts += 1
            await self._on_failure()
            raise TimeoutError(f"Call through circuit breaker {self.name} timed out after {self.config.timeout}s")
            
        except self.config.expected_exception as e:
            await self._on_failure()
            raise
            
        except Exception as e:
            # Unexpected exception, don't count as circuit failure
            logger.warning(f"Unexpected exception in circuit breaker {self.name}: {e}")
            raise
    
    async def _on_success(self):
        """Handle successful call."""
        async with self._lock:
            self.stats.successful_calls += 1
            self.stats.last_success_time = datetime.now()
            self.stats.consecutive_successes += 1
            self.stats.consecutive_failures = 0
            
            if self.state == CircuitState.HALF_OPEN:
                if self.stats.consecutive_successes >= self.config.success_threshold:
                    self._transition_to_closed()
    
    async def _on_failure(self):
        """Handle failed call."""
        async with self._lock:
            self.stats.failed_calls += 1
            self.stats.last_failure_time = datetime.now()
            self.stats.consecutive_failures += 1
            self.stats.consecutive_successes = 0
            
            if self.state == CircuitState.CLOSED:
                if self.stats.consecutive_failures >= self.config.failure_threshold:
                    self._transition_to_open()
            elif self.state == CircuitState.HALF_OPEN:
                self._transition_to_open()
    
    def _should_attempt_reset(self) -> bool:
        """Check if enough time has passed to attempt reset."""
        return time.time() - self._state_changed_at >= self.config.recovery_timeout
    
    def _transition_to_open(self):
        """Transition to OPEN state."""
        logger.warning(f"Circuit breaker {self.name} transitioning to OPEN")
        self.state = CircuitState.OPEN
        self._state_changed_at = time.time()
        self.stats.consecutive_failures = 0
    
    def _transition_to_closed(self):
        """Transition to CLOSED state."""
        logger.info(f"Circuit breaker {self.name} transitioning to CLOSED")
        self.state = CircuitState.CLOSED
        self._state_changed_at = time.time()
        self.stats.consecutive_successes = 0
    
    def _transition_to_half_open(self):
        """Transition to HALF_OPEN state."""
        logger.info(f"Circuit breaker {self.name} transitioning to HALF_OPEN")
        self.state = CircuitState.HALF_OPEN
        self._state_changed_at = time.time()
        self.stats.consecutive_successes = 0
        self.stats.consecutive_failures = 0
    
    def get_state(self) -> Dict[str, Any]:
        """Get current state and statistics."""
        return {
            "name": self.name,
            "state": self.state.value,
            "stats": {
                "total_calls": self.stats.total_calls,
                "successful_calls": self.stats.successful_calls,
                "failed_calls": self.stats.failed_calls,
                "rejected_calls": self.stats.rejected_calls,
                "timeouts": self.stats.timeouts,
                "success_rate": (
                    self.stats.successful_calls / self.stats.total_calls
                    if self.stats.total_calls > 0 else 0
                ),
                "last_failure": (
                    self.stats.last_failure_time.isoformat()
                    if self.stats.last_failure_time else None
                ),
                "last_success": (
                    self.stats.last_success_time.isoformat()
                    if self.stats.last_success_time else None
                )
            },
            "config": {
                "failure_threshold": self.config.failure_threshold,
                "success_threshold": self.config.success_threshold,
                "timeout": self.config.timeout,
                "recovery_timeout": self.config.recovery_timeout
            }
        }
    
    def reset(self):
        """Reset circuit breaker to initial state."""
        self.state = CircuitState.CLOSED
        self.stats = CircuitBreakerStats()
        self._state_changed_at = time.time()
        logger.info(f"Circuit breaker {self.name} reset")


class CircuitOpenError(Exception):
    """Raised when circuit breaker is open."""
    pass


class LLMCircuitBreaker:
    """
    Specialized circuit breaker for LLM calls with provider-specific configs.
    """
    
    def __init__(self):
        """Initialize LLM circuit breakers for different providers."""
        self.breakers = {}
        
        # Default configurations for different providers
        self.provider_configs = {
            "openai": CircuitBreakerConfig(
                failure_threshold=3,
                timeout=30.0,
                recovery_timeout=60.0
            ),
            "anthropic": CircuitBreakerConfig(
                failure_threshold=3,
                timeout=45.0,
                recovery_timeout=60.0
            ),
            "local": CircuitBreakerConfig(
                failure_threshold=5,
                timeout=60.0,
                recovery_timeout=30.0
            ),
            "default": CircuitBreakerConfig(
                failure_threshold=5,
                timeout=30.0,
                recovery_timeout=60.0
            )
        }
    
    def get_breaker(self, provider: str) -> CircuitBreaker:
        """Get or create circuit breaker for a provider."""
        if provider not in self.breakers:
            config = self.provider_configs.get(
                provider,
                self.provider_configs["default"]
            )
            self.breakers[provider] = CircuitBreaker(
                name=f"llm_{provider}",
                config=config
            )
        return self.breakers[provider]
    
    async def call_with_breaker(
        self,
        provider: str,
        func: Callable,
        *args,
        **kwargs
    ) -> Any:
        """
        Call an LLM function with circuit breaker protection.
        
        Args:
            provider: LLM provider name
            func: Function to call
            *args: Positional arguments
            **kwargs: Keyword arguments
            
        Returns:
            Result from function
        """
        breaker = self.get_breaker(provider)
        return await breaker.call(func, *args, **kwargs)
    
    def get_all_states(self) -> Dict[str, Dict[str, Any]]:
        """Get states of all circuit breakers."""
        return {
            provider: breaker.get_state()
            for provider, breaker in self.breakers.items()
        }
    
    def reset_all(self):
        """Reset all circuit breakers."""
        for breaker in self.breakers.values():
            breaker.reset()


# Global instance for use across the application
llm_circuit_breaker = LLMCircuitBreaker()


# Decorator for adding circuit breaker to functions
def with_circuit_breaker(provider: str = "default"):
    """
    Decorator to add circuit breaker protection to a function.
    
    Args:
        provider: Provider name for circuit breaker configuration
    """
    def decorator(func):
        async def wrapper(*args, **kwargs):
            return await llm_circuit_breaker.call_with_breaker(
                provider, func, *args, **kwargs
            )
        return wrapper
    return decorator