# resilience.py
"""
Error recovery and resilience mechanisms for the RAG service.

This module provides circuit breakers, retry policies, fallback strategies,
and health monitoring to ensure robust operation of the RAG pipeline.
"""

import asyncio
import time
import random
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Union, TypeVar
from collections import deque, defaultdict
import traceback

from loguru import logger


T = TypeVar('T')


class CircuitState(Enum):
    """Circuit breaker states."""
    CLOSED = "closed"      # Normal operation
    OPEN = "open"          # Failing, reject requests
    HALF_OPEN = "half_open"  # Testing recovery


class HealthStatus(Enum):
    """Component health status."""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    UNKNOWN = "unknown"


@dataclass
class CircuitBreakerConfig:
    """Configuration for circuit breaker."""
    failure_threshold: int = 5          # Failures before opening
    success_threshold: int = 2           # Successes in half-open before closing
    timeout: float = 60.0                # Seconds before trying half-open
    window_size: int = 10                # Rolling window for tracking
    failure_rate_threshold: float = 0.5  # Failure rate to open circuit


@dataclass
class RetryConfig:
    """Configuration for retry policy."""
    max_attempts: int = 3
    initial_delay: float = 1.0
    max_delay: float = 60.0
    exponential_base: float = 2.0
    jitter: bool = True
    retry_on: List[type] = field(default_factory=lambda: [Exception])
    dont_retry_on: List[type] = field(default_factory=list)


@dataclass
class ErrorContext:
    """Context information for errors."""
    error: Exception
    timestamp: float
    component: str
    operation: str
    attempt: int
    metadata: Dict[str, Any] = field(default_factory=dict)
    traceback: Optional[str] = None
    
    def __post_init__(self):
        """Capture traceback if not provided."""
        if self.traceback is None and self.error:
            self.traceback = traceback.format_exc()


class CircuitBreaker:
    """Circuit breaker implementation."""
    
    def __init__(
        self,
        name: str,
        config: Optional[CircuitBreakerConfig] = None
    ):
        """
        Initialize circuit breaker.
        
        Args:
            name: Circuit breaker name
            config: Circuit breaker configuration
        """
        self.name = name
        self.config = config or CircuitBreakerConfig()
        self.state = CircuitState.CLOSED
        self.failure_count = 0
        self.success_count = 0
        self.last_failure_time = None
        self.last_state_change = time.time()
        
        # Rolling window for tracking
        self.call_results = deque(maxlen=self.config.window_size)
        
        # Callbacks
        self.on_open_callbacks = []
        self.on_close_callbacks = []
        self.on_half_open_callbacks = []
    
    async def call(self, func: Callable, *args, **kwargs) -> Any:
        """
        Call function through circuit breaker.
        
        Args:
            func: Function to call
            *args: Function arguments
            **kwargs: Function keyword arguments
            
        Returns:
            Function result
            
        Raises:
            Exception: If circuit is open or function fails
        """
        # Check if circuit should transition
        self._check_state()
        
        if self.state == CircuitState.OPEN:
            raise CircuitOpenError(f"Circuit breaker '{self.name}' is open")
        
        try:
            # Execute function
            if asyncio.iscoroutinefunction(func):
                result = await func(*args, **kwargs)
            else:
                result = func(*args, **kwargs)
            
            # Record success
            self._record_success()
            return result
            
        except Exception as e:
            # Record failure
            self._record_failure()
            raise
    
    def _check_state(self):
        """Check and update circuit state."""
        current_time = time.time()
        
        if self.state == CircuitState.OPEN:
            # Check if timeout has passed
            if self.last_failure_time and \
               current_time - self.last_failure_time >= self.config.timeout:
                self._transition_to_half_open()
        
        elif self.state == CircuitState.HALF_OPEN:
            # Check success threshold
            if self.success_count >= self.config.success_threshold:
                self._transition_to_closed()
            elif self.failure_count >= 1:  # Single failure in half-open
                self._transition_to_open()
        
        elif self.state == CircuitState.CLOSED:
            # Check failure threshold
            if len(self.call_results) >= self.config.window_size:
                failure_rate = sum(1 for r in self.call_results if not r) / len(self.call_results)
                if failure_rate >= self.config.failure_rate_threshold:
                    self._transition_to_open()
    
    def _record_success(self):
        """Record successful call."""
        self.call_results.append(True)
        
        if self.state == CircuitState.HALF_OPEN:
            self.success_count += 1
        elif self.state == CircuitState.CLOSED:
            self.failure_count = max(0, self.failure_count - 1)
    
    def _record_failure(self):
        """Record failed call."""
        self.call_results.append(False)
        self.failure_count += 1
        self.last_failure_time = time.time()
        
        if self.state == CircuitState.HALF_OPEN:
            self._transition_to_open()
        elif self.state == CircuitState.CLOSED:
            if self.failure_count >= self.config.failure_threshold:
                self._transition_to_open()
    
    def _transition_to_open(self):
        """Transition to open state."""
        if self.state != CircuitState.OPEN:
            self.state = CircuitState.OPEN
            self.last_state_change = time.time()
            logger.warning(f"Circuit breaker '{self.name}' opened")
            
            for callback in self.on_open_callbacks:
                try:
                    callback(self)
                except Exception as e:
                    logger.error(f"Error in open callback: {e}")
    
    def _transition_to_closed(self):
        """Transition to closed state."""
        if self.state != CircuitState.CLOSED:
            self.state = CircuitState.CLOSED
            self.failure_count = 0
            self.success_count = 0
            self.last_state_change = time.time()
            logger.info(f"Circuit breaker '{self.name}' closed")
            
            for callback in self.on_close_callbacks:
                try:
                    callback(self)
                except Exception as e:
                    logger.error(f"Error in close callback: {e}")
    
    def _transition_to_half_open(self):
        """Transition to half-open state."""
        if self.state != CircuitState.HALF_OPEN:
            self.state = CircuitState.HALF_OPEN
            self.success_count = 0
            self.failure_count = 0
            self.last_state_change = time.time()
            logger.info(f"Circuit breaker '{self.name}' half-open")
            
            for callback in self.on_half_open_callbacks:
                try:
                    callback(self)
                except Exception as e:
                    logger.error(f"Error in half-open callback: {e}")
    
    def reset(self):
        """Reset circuit breaker to closed state."""
        self.state = CircuitState.CLOSED
        self.failure_count = 0
        self.success_count = 0
        self.last_failure_time = None
        self.call_results.clear()
        logger.info(f"Circuit breaker '{self.name}' reset")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get circuit breaker statistics."""
        return {
            "name": self.name,
            "state": self.state.value,
            "failure_count": self.failure_count,
            "success_count": self.success_count,
            "call_history": list(self.call_results),
            "failure_rate": sum(1 for r in self.call_results if not r) / len(self.call_results) 
                           if self.call_results else 0,
            "last_failure_time": self.last_failure_time,
            "last_state_change": self.last_state_change
        }


class RetryPolicy:
    """Retry policy with exponential backoff."""
    
    def __init__(self, config: Optional[RetryConfig] = None):
        """
        Initialize retry policy.
        
        Args:
            config: Retry configuration
        """
        self.config = config or RetryConfig()
    
    async def execute(self, func: Callable[..., T], *args, **kwargs) -> T:
        """
        Execute function with retry policy.
        
        Args:
            func: Function to execute
            *args: Function arguments
            **kwargs: Function keyword arguments
            
        Returns:
            Function result
            
        Raises:
            Exception: If all retries fail
        """
        last_exception = None
        
        for attempt in range(1, self.config.max_attempts + 1):
            try:
                # Execute function
                if asyncio.iscoroutinefunction(func):
                    result = await func(*args, **kwargs)
                else:
                    result = await asyncio.to_thread(func, *args, **kwargs)
                
                if attempt > 1:
                    logger.info(f"Retry succeeded on attempt {attempt}")
                
                return result
                
            except Exception as e:
                last_exception = e
                
                # Check if we should retry this exception
                should_retry = self._should_retry(e)
                
                if not should_retry or attempt == self.config.max_attempts:
                    logger.error(f"Retry failed after {attempt} attempts: {e}")
                    raise
                
                # Calculate delay
                delay = self._calculate_delay(attempt)
                
                logger.warning(
                    f"Attempt {attempt} failed: {e}. "
                    f"Retrying in {delay:.2f}s..."
                )
                
                await asyncio.sleep(delay)
        
        raise last_exception
    
    def _should_retry(self, exception: Exception) -> bool:
        """Check if exception should be retried."""
        # Check dont_retry_on first
        for exc_type in self.config.dont_retry_on:
            if isinstance(exception, exc_type):
                return False
        
        # Check retry_on
        for exc_type in self.config.retry_on:
            if isinstance(exception, exc_type):
                return True
        
        return False
    
    def _calculate_delay(self, attempt: int) -> float:
        """Calculate retry delay with exponential backoff."""
        delay = self.config.initial_delay * (self.config.exponential_base ** (attempt - 1))
        delay = min(delay, self.config.max_delay)
        
        if self.config.jitter:
            # Add random jitter (±25%)
            jitter = delay * 0.25 * (2 * random.random() - 1)
            delay += jitter
        
        return max(0, delay)


class FallbackChain:
    """Chain of fallback strategies."""
    
    def __init__(self):
        """Initialize fallback chain."""
        self.strategies = []
    
    def add_strategy(
        self,
        func: Callable,
        condition: Optional[Callable[[Exception], bool]] = None
    ):
        """
        Add fallback strategy.
        
        Args:
            func: Fallback function
            condition: Condition to use this fallback
        """
        self.strategies.append((func, condition))
    
    async def execute(self, primary_func: Callable, *args, **kwargs) -> Any:
        """
        Execute with fallback chain.
        
        Args:
            primary_func: Primary function to try
            *args: Function arguments
            **kwargs: Function keyword arguments
            
        Returns:
            Result from successful function
        """
        # Try primary function
        try:
            if asyncio.iscoroutinefunction(primary_func):
                return await primary_func(*args, **kwargs)
            else:
                return await asyncio.to_thread(primary_func, *args, **kwargs)
        except Exception as primary_error:
            logger.warning(f"Primary function failed: {primary_error}")
            
            # Try fallback strategies
            for fallback_func, condition in self.strategies:
                if condition is None or condition(primary_error):
                    try:
                        logger.info(f"Trying fallback: {fallback_func.__name__}")
                        
                        if asyncio.iscoroutinefunction(fallback_func):
                            return await fallback_func(*args, **kwargs)
                        else:
                            return await asyncio.to_thread(fallback_func, *args, **kwargs)
                            
                    except Exception as fallback_error:
                        logger.warning(f"Fallback failed: {fallback_error}")
                        continue
            
            # All fallbacks failed
            logger.error("All fallback strategies failed")
            raise primary_error


class HealthMonitor:
    """Monitor component health."""
    
    def __init__(self):
        """Initialize health monitor."""
        self.components: Dict[str, ComponentHealth] = {}
        self.check_interval = 30  # seconds
        self.monitoring_task = None
    
    def register_component(
        self,
        name: str,
        health_check: Callable[[], bool],
        critical: bool = False
    ):
        """
        Register component for monitoring.
        
        Args:
            name: Component name
            health_check: Function to check health
            critical: Whether component is critical
        """
        self.components[name] = ComponentHealth(
            name=name,
            health_check=health_check,
            critical=critical
        )
    
    async def start_monitoring(self):
        """Start health monitoring."""
        if not self.monitoring_task:
            self.monitoring_task = asyncio.create_task(self._monitoring_loop())
    
    async def stop_monitoring(self):
        """Stop health monitoring."""
        if self.monitoring_task:
            self.monitoring_task.cancel()
            try:
                await self.monitoring_task
            except asyncio.CancelledError:
                pass
            self.monitoring_task = None
    
    async def _monitoring_loop(self):
        """Background health monitoring loop."""
        while True:
            try:
                await self.check_all_health()
                await asyncio.sleep(self.check_interval)
            except Exception as e:
                logger.error(f"Error in health monitoring: {e}")
                await asyncio.sleep(self.check_interval)
    
    async def check_all_health(self) -> Dict[str, HealthStatus]:
        """Check health of all components."""
        results = {}
        
        for name, component in self.components.items():
            try:
                if asyncio.iscoroutinefunction(component.health_check):
                    is_healthy = await component.health_check()
                else:
                    is_healthy = await asyncio.to_thread(component.health_check)
                
                component.update_health(is_healthy)
                results[name] = component.status
                
                if not is_healthy and component.critical:
                    logger.error(f"Critical component '{name}' is unhealthy")
                
            except Exception as e:
                logger.error(f"Health check failed for '{name}': {e}")
                component.update_health(False)
                results[name] = HealthStatus.UNKNOWN
        
        return results
    
    def get_overall_health(self) -> HealthStatus:
        """Get overall system health."""
        if not self.components:
            return HealthStatus.UNKNOWN
        
        # Check critical components first
        critical_unhealthy = any(
            c.critical and c.status != HealthStatus.HEALTHY
            for c in self.components.values()
        )
        
        if critical_unhealthy:
            return HealthStatus.UNHEALTHY
        
        # Check overall health
        statuses = [c.status for c in self.components.values()]
        
        if all(s == HealthStatus.HEALTHY for s in statuses):
            return HealthStatus.HEALTHY
        elif any(s == HealthStatus.UNHEALTHY for s in statuses):
            return HealthStatus.DEGRADED
        else:
            return HealthStatus.DEGRADED


@dataclass
class ComponentHealth:
    """Health information for a component."""
    name: str
    health_check: Callable[[], bool]
    critical: bool = False
    status: HealthStatus = HealthStatus.UNKNOWN
    last_check: Optional[float] = None
    consecutive_failures: int = 0
    
    def update_health(self, is_healthy: bool):
        """Update health status."""
        self.last_check = time.time()
        
        if is_healthy:
            self.status = HealthStatus.HEALTHY
            self.consecutive_failures = 0
        else:
            self.consecutive_failures += 1
            
            if self.consecutive_failures >= 3:
                self.status = HealthStatus.UNHEALTHY
            else:
                self.status = HealthStatus.DEGRADED


class ErrorRecoveryCoordinator:
    """Coordinates error recovery across components."""
    
    def __init__(self):
        """Initialize error recovery coordinator."""
        self.circuit_breakers: Dict[str, CircuitBreaker] = {}
        self.retry_policies: Dict[str, RetryPolicy] = {}
        self.fallback_chains: Dict[str, FallbackChain] = {}
        self.health_monitor = HealthMonitor()
        self.error_history: deque = deque(maxlen=100)
    
    def register_circuit_breaker(
        self,
        name: str,
        config: Optional[CircuitBreakerConfig] = None
    ) -> CircuitBreaker:
        """Register a circuit breaker."""
        breaker = CircuitBreaker(name, config)
        self.circuit_breakers[name] = breaker
        return breaker
    
    def register_retry_policy(
        self,
        name: str,
        config: Optional[RetryConfig] = None
    ) -> RetryPolicy:
        """Register a retry policy."""
        policy = RetryPolicy(config)
        self.retry_policies[name] = policy
        return policy
    
    def register_fallback_chain(
        self,
        name: str
    ) -> FallbackChain:
        """Register a fallback chain."""
        chain = FallbackChain()
        self.fallback_chains[name] = chain
        return chain
    
    def record_error(self, error_context: ErrorContext):
        """Record an error for analysis."""
        self.error_history.append(error_context)
        
        # Log based on error frequency
        recent_errors = [
            e for e in self.error_history
            if e.component == error_context.component and
            time.time() - e.timestamp < 60
        ]
        
        if len(recent_errors) > 5:
            logger.error(
                f"High error rate for component '{error_context.component}': "
                f"{len(recent_errors)} errors in last minute"
            )
    
    def get_recovery_stats(self) -> Dict[str, Any]:
        """Get recovery system statistics."""
        return {
            "circuit_breakers": {
                name: breaker.get_stats()
                for name, breaker in self.circuit_breakers.items()
            },
            "health_status": self.health_monitor.get_overall_health().value,
            "error_count": len(self.error_history),
            "recent_errors": [
                {
                    "component": e.component,
                    "operation": e.operation,
                    "timestamp": e.timestamp,
                    "error": str(e.error)
                }
                for e in list(self.error_history)[-10:]
            ]
        }


# Custom exceptions
class CircuitOpenError(Exception):
    """Raised when circuit breaker is open."""
    pass


# Global coordinator instance
_coordinator: Optional[ErrorRecoveryCoordinator] = None


def get_coordinator() -> ErrorRecoveryCoordinator:
    """Get or create global error recovery coordinator."""
    global _coordinator
    if _coordinator is None:
        _coordinator = ErrorRecoveryCoordinator()
    return _coordinator


# Pipeline integration functions

async def with_circuit_breaker(
    context: Any,
    component: str = "default",
    **kwargs
) -> Any:
    """Wrap pipeline function with circuit breaker."""
    coordinator = get_coordinator()
    
    # Get or create circuit breaker
    if component not in coordinator.circuit_breakers:
        config = CircuitBreakerConfig(
            failure_threshold=kwargs.get("failure_threshold", 5),
            timeout=kwargs.get("timeout", 60.0)
        )
        coordinator.register_circuit_breaker(component, config)
    
    breaker = coordinator.circuit_breakers[component]
    
    # Store breaker state in context
    context.metadata[f"circuit_breaker_{component}"] = breaker.state.value
    
    return context


async def with_retry(
    context: Any,
    component: str = "default",
    **kwargs
) -> Any:
    """Wrap pipeline function with retry policy."""
    coordinator = get_coordinator()
    
    # Get or create retry policy
    if component not in coordinator.retry_policies:
        config = RetryConfig(
            max_attempts=kwargs.get("max_attempts", 3),
            initial_delay=kwargs.get("initial_delay", 1.0)
        )
        coordinator.register_retry_policy(component, config)
    
    policy = coordinator.retry_policies[component]
    
    # Store retry info in context
    context.metadata[f"retry_policy_{component}"] = {
        "max_attempts": policy.config.max_attempts,
        "initial_delay": policy.config.initial_delay
    }
    
    return context


async def with_fallback(
    context: Any,
    component: str = "default",
    fallback_func: Optional[Callable] = None,
    **kwargs
) -> Any:
    """Add fallback strategy for pipeline function."""
    coordinator = get_coordinator()
    
    # Get or create fallback chain
    if component not in coordinator.fallback_chains:
        coordinator.register_fallback_chain(component)
    
    chain = coordinator.fallback_chains[component]
    
    # Add fallback if provided
    if fallback_func:
        chain.add_strategy(fallback_func)
    
    # Store fallback info in context
    context.metadata[f"fallback_{component}"] = {
        "strategies_count": len(chain.strategies)
    }
    
    return context


async def check_component_health(
    context: Any,
    component: str,
    health_check: Callable[[], bool],
    **kwargs
) -> Any:
    """Check component health before proceeding."""
    coordinator = get_coordinator()
    
    # Register component if not already
    if component not in coordinator.health_monitor.components:
        coordinator.health_monitor.register_component(
            component,
            health_check,
            critical=kwargs.get("critical", False)
        )
    
    # Check health
    try:
        if asyncio.iscoroutinefunction(health_check):
            is_healthy = await health_check()
        else:
            is_healthy = health_check()
        
        context.metadata[f"health_{component}"] = "healthy" if is_healthy else "unhealthy"
        
        if not is_healthy and kwargs.get("critical", False):
            raise Exception(f"Critical component '{component}' is unhealthy")
            
    except Exception as e:
        logger.error(f"Health check failed for '{component}': {e}")
        context.metadata[f"health_{component}"] = "unknown"
        
        if kwargs.get("critical", False):
            raise
    
    return context