# circuit_breaker.py
# Description: Circuit breaker pattern implementation for TTS providers
#
# Imports
import asyncio
import time
from typing import Dict, Optional, Any, Callable
from enum import Enum
from dataclasses import dataclass, field
from datetime import datetime, timedelta
#
# Third-party Imports
from loguru import logger
#
#######################################################################################################################
#
# Circuit Breaker Implementation

class CircuitState(Enum):
    """Circuit breaker states"""
    CLOSED = "closed"  # Normal operation
    OPEN = "open"      # Failing, reject requests
    HALF_OPEN = "half_open"  # Testing if service recovered

@dataclass
class CircuitStats:
    """Statistics for circuit breaker"""
    failure_count: int = 0
    success_count: int = 0
    last_failure_time: Optional[float] = None
    last_success_time: Optional[float] = None
    consecutive_failures: int = 0
    consecutive_successes: int = 0
    total_requests: int = 0
    
    def record_success(self):
        """Record a successful request"""
        self.success_count += 1
        self.total_requests += 1
        self.consecutive_successes += 1
        self.consecutive_failures = 0
        self.last_success_time = time.time()
    
    def record_failure(self):
        """Record a failed request"""
        self.failure_count += 1
        self.total_requests += 1
        self.consecutive_failures += 1
        self.consecutive_successes = 0
        self.last_failure_time = time.time()
    
    def reset(self):
        """Reset statistics"""
        self.failure_count = 0
        self.success_count = 0
        self.consecutive_failures = 0
        self.consecutive_successes = 0
    
    @property
    def failure_rate(self) -> float:
        """Calculate failure rate"""
        if self.total_requests == 0:
            return 0.0
        return self.failure_count / self.total_requests


class CircuitBreaker:
    """
    Circuit breaker for TTS providers.
    
    Prevents cascading failures by temporarily blocking requests to failing providers.
    """
    
    def __init__(
        self,
        provider_name: str,
        failure_threshold: int = 5,
        recovery_timeout: float = 60.0,
        half_open_max_calls: int = 3,
        success_threshold: int = 2
    ):
        """
        Initialize circuit breaker.
        
        Args:
            provider_name: Name of the provider
            failure_threshold: Number of failures before opening circuit
            recovery_timeout: Seconds to wait before trying half-open
            half_open_max_calls: Max calls allowed in half-open state
            success_threshold: Successes needed to close circuit
        """
        self.provider_name = provider_name
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.half_open_max_calls = half_open_max_calls
        self.success_threshold = success_threshold
        
        self._state = CircuitState.CLOSED
        self._stats = CircuitStats()
        self._state_changed_at = time.time()
        self._half_open_calls = 0
        self._lock = asyncio.Lock()
    
    @property
    def state(self) -> CircuitState:
        """Get current circuit state"""
        return self._state
    
    @property
    def is_available(self) -> bool:
        """Check if circuit allows requests"""
        if self._state == CircuitState.CLOSED:
            return True
        
        if self._state == CircuitState.OPEN:
            # Check if recovery timeout has passed
            if time.time() - self._state_changed_at >= self.recovery_timeout:
                self._transition_to_half_open()
                return True
            return False
        
        if self._state == CircuitState.HALF_OPEN:
            return self._half_open_calls < self.half_open_max_calls
        
        return False
    
    async def call(self, func: Callable, *args, **kwargs) -> Any:
        """
        Execute function through circuit breaker.
        
        Args:
            func: Async function to call
            *args: Positional arguments
            **kwargs: Keyword arguments
            
        Returns:
            Function result
            
        Raises:
            CircuitOpenError: If circuit is open
        """
        async with self._lock:
            if not self.is_available:
                raise CircuitOpenError(
                    f"Circuit breaker for {self.provider_name} is open. "
                    f"Failed {self._stats.consecutive_failures} times."
                )
            
            if self._state == CircuitState.HALF_OPEN:
                self._half_open_calls += 1
        
        try:
            # Execute the function
            result = await func(*args, **kwargs)
            
            # Record success
            await self._record_success()
            return result
            
        except Exception as e:
            # Record failure
            await self._record_failure()
            raise
    
    async def _record_success(self):
        """Record successful call"""
        async with self._lock:
            self._stats.record_success()
            
            if self._state == CircuitState.HALF_OPEN:
                if self._stats.consecutive_successes >= self.success_threshold:
                    self._transition_to_closed()
                    logger.info(
                        f"Circuit breaker for {self.provider_name} closed after "
                        f"{self._stats.consecutive_successes} successful calls"
                    )
    
    async def _record_failure(self):
        """Record failed call"""
        async with self._lock:
            self._stats.record_failure()
            
            if self._state == CircuitState.HALF_OPEN:
                self._transition_to_open()
                logger.warning(
                    f"Circuit breaker for {self.provider_name} reopened after "
                    f"failure in half-open state"
                )
            
            elif self._state == CircuitState.CLOSED:
                if self._stats.consecutive_failures >= self.failure_threshold:
                    self._transition_to_open()
                    logger.warning(
                        f"Circuit breaker for {self.provider_name} opened after "
                        f"{self._stats.consecutive_failures} consecutive failures"
                    )
    
    def _transition_to_open(self):
        """Transition to open state"""
        self._state = CircuitState.OPEN
        self._state_changed_at = time.time()
        self._half_open_calls = 0
        logger.info(f"Circuit breaker for {self.provider_name}: CLOSED -> OPEN")
    
    def _transition_to_closed(self):
        """Transition to closed state"""
        self._state = CircuitState.CLOSED
        self._state_changed_at = time.time()
        self._stats.reset()
        self._half_open_calls = 0
        logger.info(f"Circuit breaker for {self.provider_name}: {self._state} -> CLOSED")
    
    def _transition_to_half_open(self):
        """Transition to half-open state"""
        self._state = CircuitState.HALF_OPEN
        self._state_changed_at = time.time()
        self._half_open_calls = 0
        self._stats.consecutive_failures = 0
        self._stats.consecutive_successes = 0
        logger.info(f"Circuit breaker for {self.provider_name}: OPEN -> HALF_OPEN")
    
    def get_status(self) -> Dict[str, Any]:
        """Get circuit breaker status"""
        return {
            "provider": self.provider_name,
            "state": self._state.value,
            "stats": {
                "failure_count": self._stats.failure_count,
                "success_count": self._stats.success_count,
                "failure_rate": round(self._stats.failure_rate, 3),
                "consecutive_failures": self._stats.consecutive_failures,
                "consecutive_successes": self._stats.consecutive_successes,
                "total_requests": self._stats.total_requests
            },
            "config": {
                "failure_threshold": self.failure_threshold,
                "recovery_timeout": self.recovery_timeout,
                "half_open_max_calls": self.half_open_max_calls,
                "success_threshold": self.success_threshold
            }
        }
    
    def reset(self):
        """Manually reset circuit breaker"""
        self._state = CircuitState.CLOSED
        self._stats.reset()
        self._state_changed_at = time.time()
        self._half_open_calls = 0
        logger.info(f"Circuit breaker for {self.provider_name} manually reset")


class CircuitOpenError(Exception):
    """Exception raised when circuit is open"""
    pass


class CircuitBreakerManager:
    """
    Manages circuit breakers for all TTS providers.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize circuit breaker manager.
        
        Args:
            config: Configuration for circuit breakers
        """
        self.config = config or {}
        self._breakers: Dict[str, CircuitBreaker] = {}
        self._lock = asyncio.Lock()
        
        # Default configuration
        self.default_config = {
            "failure_threshold": self.config.get("circuit_failure_threshold", 5),
            "recovery_timeout": self.config.get("circuit_recovery_timeout", 60.0),
            "half_open_max_calls": self.config.get("circuit_half_open_calls", 3),
            "success_threshold": self.config.get("circuit_success_threshold", 2)
        }
    
    async def get_breaker(self, provider_name: str) -> CircuitBreaker:
        """
        Get or create circuit breaker for provider.
        
        Args:
            provider_name: Name of the provider
            
        Returns:
            Circuit breaker instance
        """
        async with self._lock:
            if provider_name not in self._breakers:
                # Get provider-specific config or use defaults
                provider_config = self.config.get(f"{provider_name}_circuit", {})
                config = {**self.default_config, **provider_config}
                
                self._breakers[provider_name] = CircuitBreaker(
                    provider_name=provider_name,
                    **config
                )
                logger.debug(f"Created circuit breaker for {provider_name}")
            
            return self._breakers[provider_name]
    
    def get_all_status(self) -> Dict[str, Any]:
        """Get status of all circuit breakers"""
        return {
            name: breaker.get_status()
            for name, breaker in self._breakers.items()
        }
    
    def reset_all(self):
        """Reset all circuit breakers"""
        for breaker in self._breakers.values():
            breaker.reset()
        logger.info("All circuit breakers reset")
    
    def reset_provider(self, provider_name: str):
        """Reset specific provider's circuit breaker"""
        if provider_name in self._breakers:
            self._breakers[provider_name].reset()
            logger.info(f"Circuit breaker for {provider_name} reset")


# Singleton instance
_circuit_manager: Optional[CircuitBreakerManager] = None
_manager_lock = asyncio.Lock()


async def get_circuit_manager(config: Optional[Dict[str, Any]] = None) -> CircuitBreakerManager:
    """
    Get or create circuit breaker manager singleton.
    
    Args:
        config: Configuration for circuit breakers
        
    Returns:
        CircuitBreakerManager instance
    """
    global _circuit_manager
    
    if _circuit_manager is None:
        async with _manager_lock:
            if _circuit_manager is None:
                _circuit_manager = CircuitBreakerManager(config)
                logger.info("Circuit breaker manager initialized")
    
    return _circuit_manager

#
# End of circuit_breaker.py
#######################################################################################################################