# circuit_breaker.py
# Description: Circuit breaker pattern implementation for TTS providers
#
# Imports
import asyncio
import time
import random
from typing import Dict, Optional, Any, Callable, List, Tuple
from enum import Enum
from dataclasses import dataclass, field
from datetime import datetime, timedelta
#
# Third-party Imports
from loguru import logger
#
# Local Imports
from .tts_exceptions import (
    TTSCircuitOpenError,
    TTSProviderError,
    TTSProviderUnavailableError,
    TTSNetworkError,
    TTSTimeoutError,
    categorize_error
)
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
    
    def get_backoff_delay(self, base_delay: float = 1.0, max_delay: float = 60.0) -> float:
        """Calculate exponential backoff delay based on consecutive failures"""
        if self.consecutive_failures == 0:
            return 0.0
        
        # Exponential backoff with jitter
        delay = base_delay * (2 ** min(self.consecutive_failures - 1, 10))  # Cap at 2^10
        delay = min(delay, max_delay)  # Cap at max_delay
        
        # Add jitter (±25% of the delay)
        jitter = delay * 0.25 * (2 * random.random() - 1)
        return max(0.1, delay + jitter)  # Minimum 0.1 seconds


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
        success_threshold: int = 2,
        backoff_base_delay: float = 1.0,
        backoff_max_delay: float = 60.0,
        health_check_interval: float = 300.0  # 5 minutes
    ):
        """
        Initialize circuit breaker.
        
        Args:
            provider_name: Name of the provider
            failure_threshold: Number of failures before opening circuit
            recovery_timeout: Seconds to wait before trying half-open
            half_open_max_calls: Max calls allowed in half-open state
            success_threshold: Successes needed to close circuit
            backoff_base_delay: Base delay for exponential backoff
            backoff_max_delay: Maximum delay for exponential backoff
            health_check_interval: Interval for health checks in seconds
        """
        self.provider_name = provider_name
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.half_open_max_calls = half_open_max_calls
        self.success_threshold = success_threshold
        self.backoff_base_delay = backoff_base_delay
        self.backoff_max_delay = backoff_max_delay
        self.health_check_interval = health_check_interval
        
        self._state = CircuitState.CLOSED
        self._stats = CircuitStats()
        self._state_changed_at = time.time()
        self._half_open_calls = 0
        self._lock = asyncio.Lock()
        self._last_health_check = time.time()
        self._health_check_task: Optional[asyncio.Task] = None
        self._error_categories: Dict[str, int] = {}  # Track error types
    
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
                backoff_delay = self._stats.get_backoff_delay(
                    self.backoff_base_delay,
                    self.backoff_max_delay
                )
                raise TTSCircuitOpenError(
                    f"Circuit breaker for {self.provider_name} is open. "
                    f"Failed {self._stats.consecutive_failures} times. "
                    f"Retry after {backoff_delay:.1f}s.",
                    provider=self.provider_name,
                    details={
                        "consecutive_failures": self._stats.consecutive_failures,
                        "backoff_delay": backoff_delay,
                        "failure_rate": self._stats.failure_rate,
                        "error_categories": self._error_categories.copy()
                    }
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
            # Record failure with error context
            await self._record_failure(e)
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
    
    async def _record_failure(self, error: Optional[Exception] = None):
        """Record failed call with error categorization"""
        async with self._lock:
            self._stats.record_failure()
            
            # Categorize the error for better insights
            if error:
                error_category = categorize_error(error)
                self._error_categories[error_category] = self._error_categories.get(error_category, 0) + 1
            
            if self._state == CircuitState.HALF_OPEN:
                self._transition_to_open()
                logger.warning(
                    f"Circuit breaker for {self.provider_name} reopened after "
                    f"failure in half-open state. Error: {error if error else 'Unknown'}"
                )
            
            elif self._state == CircuitState.CLOSED:
                if self._stats.consecutive_failures >= self.failure_threshold:
                    self._transition_to_open()
                    logger.warning(
                        f"Circuit breaker for {self.provider_name} opened after "
                        f"{self._stats.consecutive_failures} consecutive failures. "
                        f"Error categories: {self._error_categories}"
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
        """Get basic circuit breaker status"""
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
        self._error_categories.clear()
        self._last_health_check = time.time()
        logger.info(f"Circuit breaker for {self.provider_name} manually reset")
    
    async def start_health_monitoring(self, health_check_func: Optional[Callable] = None):
        """Start health monitoring for the provider"""
        if self._health_check_task and not self._health_check_task.done():
            return
        
        self._health_check_task = asyncio.create_task(
            self._health_monitor_loop(health_check_func)
        )
        logger.info(f"Started health monitoring for {self.provider_name}")
    
    async def stop_health_monitoring(self):
        """Stop health monitoring"""
        if self._health_check_task and not self._health_check_task.done():
            self._health_check_task.cancel()
            try:
                await self._health_check_task
            except asyncio.CancelledError:
                pass
        logger.info(f"Stopped health monitoring for {self.provider_name}")
    
    async def _health_monitor_loop(self, health_check_func: Optional[Callable]):
        """Health monitoring loop"""
        while True:
            try:
                await asyncio.sleep(self.health_check_interval)
                
                # Only perform health checks when circuit is open or half-open
                if self._state in [CircuitState.OPEN, CircuitState.HALF_OPEN]:
                    await self._perform_health_check(health_check_func)
                    
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in health monitoring for {self.provider_name}: {e}")
    
    async def _perform_health_check(self, health_check_func: Optional[Callable]):
        """Perform health check on the provider"""
        try:
            self._last_health_check = time.time()
            
            if health_check_func:
                # Use custom health check function
                if asyncio.iscoroutinefunction(health_check_func):
                    is_healthy = await health_check_func()
                else:
                    is_healthy = health_check_func()
                
                if is_healthy and self._state == CircuitState.OPEN:
                    # If provider is healthy and circuit is open, try half-open
                    if time.time() - self._state_changed_at >= self.recovery_timeout:
                        self._transition_to_half_open()
                        logger.info(
                            f"Health check passed for {self.provider_name}, "
                            "transitioning to half-open state"
                        )
            else:
                # Default health check based on time and error patterns
                if self._should_attempt_recovery():
                    self._transition_to_half_open()
                    logger.info(
                        f"Automatic recovery attempt for {self.provider_name}, "
                        "transitioning to half-open state"
                    )
                    
        except Exception as e:
            logger.error(f"Health check failed for {self.provider_name}: {e}")
    
    def _should_attempt_recovery(self) -> bool:
        """Determine if automatic recovery should be attempted"""
        if self._state != CircuitState.OPEN:
            return False
        
        # Check if enough time has passed
        if time.time() - self._state_changed_at < self.recovery_timeout:
            return False
        
        # Analyze error patterns to decide if recovery is likely
        # If errors are primarily network/timeout related, recovery is more likely
        network_errors = self._error_categories.get('network', 0)
        total_errors = sum(self._error_categories.values()) or 1
        network_error_ratio = network_errors / total_errors
        
        # More aggressive recovery for network errors
        if network_error_ratio > 0.7:
            return True
        
        # Conservative recovery for other types of errors
        return time.time() - self._state_changed_at >= self.recovery_timeout * 2
    
    def get_detailed_status(self) -> Dict[str, Any]:
        """Get detailed circuit breaker status including error analysis"""
        return {
            "provider": self.provider_name,
            "state": self._state.value,
            "stats": {
                "failure_count": self._stats.failure_count,
                "success_count": self._stats.success_count,
                "failure_rate": round(self._stats.failure_rate, 3),
                "consecutive_failures": self._stats.consecutive_failures,
                "consecutive_successes": self._stats.consecutive_successes,
                "total_requests": self._stats.total_requests,
                "last_failure_time": self._stats.last_failure_time,
                "last_success_time": self._stats.last_success_time
            },
            "config": {
                "failure_threshold": self.failure_threshold,
                "recovery_timeout": self.recovery_timeout,
                "half_open_max_calls": self.half_open_max_calls,
                "success_threshold": self.success_threshold,
                "backoff_base_delay": self.backoff_base_delay,
                "backoff_max_delay": self.backoff_max_delay
            },
            "error_analysis": {
                "error_categories": self._error_categories.copy(),
                "backoff_delay": self._stats.get_backoff_delay(
                    self.backoff_base_delay, self.backoff_max_delay
                )
            },
            "health_monitoring": {
                "enabled": self._health_check_task is not None and not self._health_check_task.done(),
                "last_health_check": self._last_health_check,
                "health_check_interval": self.health_check_interval
            },
            "state_info": {
                "state_changed_at": self._state_changed_at,
                "time_in_state": time.time() - self._state_changed_at,
                "half_open_calls": self._half_open_calls if self._state == CircuitState.HALF_OPEN else None
            }
        }


# CircuitOpenError is now defined in tts_exceptions.py as TTSCircuitOpenError
# Maintain backward compatibility
CircuitOpenError = TTSCircuitOpenError


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
    
    def get_all_status(self, detailed: bool = False) -> Dict[str, Any]:
        """Get status of all circuit breakers"""
        if detailed:
            return {
                name: breaker.get_detailed_status()
                for name, breaker in self._breakers.items()
            }
        else:
            return {
                name: breaker.get_status()
                for name, breaker in self._breakers.items()
            }
    
    async def enable_health_monitoring(self, provider_name: Optional[str] = None, health_check_functions: Optional[Dict[str, Callable]] = None):
        """Enable health monitoring for providers"""
        health_check_functions = health_check_functions or {}
        
        if provider_name:
            # Enable for specific provider
            if provider_name in self._breakers:
                health_check_func = health_check_functions.get(provider_name)
                await self._breakers[provider_name].start_health_monitoring(health_check_func)
        else:
            # Enable for all providers
            for name, breaker in self._breakers.items():
                health_check_func = health_check_functions.get(name)
                await breaker.start_health_monitoring(health_check_func)
    
    async def disable_health_monitoring(self, provider_name: Optional[str] = None):
        """Disable health monitoring for providers"""
        if provider_name:
            # Disable for specific provider
            if provider_name in self._breakers:
                await self._breakers[provider_name].stop_health_monitoring()
        else:
            # Disable for all providers
            for breaker in self._breakers.values():
                await breaker.stop_health_monitoring()
    
    def get_error_analysis(self) -> Dict[str, Any]:
        """Get error analysis across all circuit breakers"""
        analysis = {
            "total_providers": len(self._breakers),
            "providers_by_state": {"closed": 0, "open": 0, "half_open": 0},
            "error_categories": {},
            "failure_patterns": [],
            "recovery_candidates": []
        }
        
        for name, breaker in self._breakers.items():
            status = breaker.get_detailed_status()
            state = status["state"]
            analysis["providers_by_state"][state] += 1
            
            # Aggregate error categories
            for category, count in status["error_analysis"]["error_categories"].items():
                analysis["error_categories"][category] = analysis["error_categories"].get(category, 0) + count
            
            # Identify failure patterns
            if status["stats"]["consecutive_failures"] >= 3:
                analysis["failure_patterns"].append({
                    "provider": name,
                    "consecutive_failures": status["stats"]["consecutive_failures"],
                    "failure_rate": status["stats"]["failure_rate"],
                    "dominant_error": max(status["error_analysis"]["error_categories"].items(), key=lambda x: x[1], default=("unknown", 0))[0]
                })
            
            # Identify recovery candidates
            if state == "open" and breaker._should_attempt_recovery():
                analysis["recovery_candidates"].append({
                    "provider": name,
                    "time_in_open_state": status["state_info"]["time_in_state"],
                    "recovery_timeout": status["config"]["recovery_timeout"]
                })
        
        return analysis
    
    def reset_all(self):
        """Reset all circuit breakers"""
        for breaker in self._breakers.values():
            breaker.reset()
        logger.info("All circuit breakers reset")
    
    async def reset_provider(self, provider_name: str, restart_health_monitoring: bool = True):
        """Reset specific provider's circuit breaker"""
        if provider_name in self._breakers:
            breaker = self._breakers[provider_name]
            
            # Stop health monitoring
            await breaker.stop_health_monitoring()
            
            # Reset the breaker
            breaker.reset()
            
            # Restart health monitoring if requested
            if restart_health_monitoring:
                await breaker.start_health_monitoring()
            
            logger.info(f"Circuit breaker for {provider_name} reset")
    
    async def shutdown(self):
        """Shutdown circuit breaker manager and stop all health monitoring"""
        logger.info("Shutting down circuit breaker manager")
        
        # Stop all health monitoring
        tasks = []
        for breaker in self._breakers.values():
            tasks.append(breaker.stop_health_monitoring())
        
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
        
        logger.info("Circuit breaker manager shutdown complete")


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