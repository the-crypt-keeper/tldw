# metrics_integration.py
# Integration with the existing metrics module for comprehensive monitoring

from typing import Dict, Any, Optional
import time
from functools import wraps
import asyncio

from loguru import logger
from tldw_Server_API.app.core.Metrics.metrics_logger import (
    log_counter, 
    log_histogram,
    log_resource_usage
)


class EmbeddingMetrics:
    """
    Comprehensive metrics collection for the embeddings module.
    Integrates with the existing metrics infrastructure.
    """
    
    def __init__(self):
        self.metrics_prefix = "embeddings"
        
    # Request metrics
    def log_request(self, provider: str, model: str, status: str = "success"):
        """Log an embedding request"""
        log_counter(
            f"{self.metrics_prefix}_requests_total",
            labels={"provider": provider, "model": model, "status": status}
        )
    
    def log_request_latency(self, provider: str, model: str, latency_seconds: float):
        """Log request latency"""
        log_histogram(
            f"{self.metrics_prefix}_request_latency_seconds",
            latency_seconds,
            labels={"provider": provider, "model": model}
        )
    
    def log_batch_size(self, provider: str, size: int):
        """Log batch processing size"""
        log_histogram(
            f"{self.metrics_prefix}_batch_size",
            size,
            labels={"provider": provider}
        )
    
    # Cache metrics
    def log_cache_hit(self, model: str):
        """Log a cache hit"""
        log_counter(
            f"{self.metrics_prefix}_cache_hits_total",
            labels={"model": model}
        )
    
    def log_cache_miss(self, model: str):
        """Log a cache miss"""
        log_counter(
            f"{self.metrics_prefix}_cache_misses_total",
            labels={"model": model}
        )
    
    def update_cache_size(self, size: int):
        """Update current cache size"""
        log_histogram(
            f"{self.metrics_prefix}_cache_size",
            size
        )
    
    def update_cache_memory(self, memory_bytes: int):
        """Update cache memory usage"""
        log_histogram(
            f"{self.metrics_prefix}_cache_memory_bytes",
            memory_bytes
        )
    
    # Model metrics
    def log_model_load(self, model: str, load_time_seconds: float):
        """Log model loading time"""
        log_histogram(
            f"{self.metrics_prefix}_model_load_time_seconds",
            load_time_seconds,
            labels={"model": model}
        )
        log_counter(
            f"{self.metrics_prefix}_model_loads_total",
            labels={"model": model}
        )
    
    def log_model_eviction(self, model: str, reason: str):
        """Log model eviction"""
        log_counter(
            f"{self.metrics_prefix}_model_evictions_total",
            labels={"model": model, "reason": reason}
        )
    
    def update_models_in_memory(self, count: int):
        """Update number of models in memory"""
        log_histogram(
            f"{self.metrics_prefix}_models_in_memory",
            count
        )
    
    def update_model_memory_usage(self, model: str, memory_gb: float):
        """Update memory usage for a model"""
        log_histogram(
            f"{self.metrics_prefix}_model_memory_gb",
            memory_gb,
            labels={"model": model}
        )
    
    # Error metrics
    def log_error(self, provider: str, error_type: str):
        """Log an error"""
        log_counter(
            f"{self.metrics_prefix}_errors_total",
            labels={"provider": provider, "error_type": error_type}
        )
    
    def log_retry(self, provider: str, attempt: int):
        """Log a retry attempt"""
        log_counter(
            f"{self.metrics_prefix}_retries_total",
            labels={"provider": provider, "attempt": str(attempt)}
        )
    
    # Rate limiting metrics
    def log_rate_limit_hit(self, user_id: str, tier: str):
        """Log rate limit hit"""
        log_counter(
            f"{self.metrics_prefix}_rate_limits_hit_total",
            labels={"tier": tier}
        )
    
    def update_rate_limit_usage(self, user_id: str, usage_percent: float):
        """Update rate limit usage percentage"""
        log_histogram(
            f"{self.metrics_prefix}_rate_limit_usage_percent",
            usage_percent,
            labels={"user_id": user_id[:8]}  # Use prefix for privacy
        )
    
    # Connection pool metrics
    def update_pool_connections(self, provider: str, active: int, idle: int):
        """Update connection pool stats"""
        log_histogram(
            f"{self.metrics_prefix}_pool_active_connections",
            active,
            labels={"provider": provider}
        )
        log_histogram(
            f"{self.metrics_prefix}_pool_idle_connections",
            idle,
            labels={"provider": provider}
        )
    
    # DLQ metrics
    def update_dlq_size(self, size: int):
        """Update dead letter queue size"""
        log_histogram(
            f"{self.metrics_prefix}_dlq_size",
            size
        )
    
    def log_dlq_addition(self, reason: str):
        """Log addition to DLQ"""
        log_counter(
            f"{self.metrics_prefix}_dlq_additions_total",
            labels={"reason": reason}
        )
    
    def log_dlq_recovery(self):
        """Log successful recovery from DLQ"""
        log_counter(f"{self.metrics_prefix}_dlq_recoveries_total")
    
    # Circuit breaker metrics
    def update_circuit_breaker_state(self, provider: str, state: str):
        """Update circuit breaker state"""
        state_value = {"closed": 0, "open": 1, "half_open": 2}.get(state.lower(), -1)
        log_histogram(
            f"{self.metrics_prefix}_circuit_breaker_state",
            state_value,
            labels={"provider": provider}
        )
    
    def log_circuit_breaker_trip(self, provider: str):
        """Log circuit breaker trip"""
        log_counter(
            f"{self.metrics_prefix}_circuit_breaker_trips_total",
            labels={"provider": provider}
        )
    
    # Performance tracking decorator
    def track_performance(self, operation: str):
        """Decorator to track function performance"""
        def decorator(func):
            if asyncio.iscoroutinefunction(func):
                @wraps(func)
                async def async_wrapper(*args, **kwargs):
                    start_time = time.time()
                    try:
                        result = await func(*args, **kwargs)
                        elapsed = time.time() - start_time
                        log_histogram(
                            f"{self.metrics_prefix}_{operation}_duration_seconds",
                            elapsed
                        )
                        return result
                    except Exception as e:
                        self.log_error("unknown", f"{operation}_error")
                        raise
                return async_wrapper
            else:
                @wraps(func)
                def sync_wrapper(*args, **kwargs):
                    start_time = time.time()
                    try:
                        result = func(*args, **kwargs)
                        elapsed = time.time() - start_time
                        log_histogram(
                            f"{self.metrics_prefix}_{operation}_duration_seconds",
                            elapsed
                        )
                        return result
                    except Exception as e:
                        self.log_error("unknown", f"{operation}_error")
                        raise
                return sync_wrapper
        return decorator
    
    # Resource tracking
    def log_resource_usage_stats(self):
        """Log current resource usage"""
        log_resource_usage()
    
    # Aggregated metrics
    def get_summary_metrics(self) -> Dict[str, Any]:
        """Get summary of key metrics"""
        # This would integrate with the metrics backend to pull current values
        # For now, return a structure that would be populated
        return {
            "total_requests": 0,  # Would be fetched from metrics store
            "cache_hit_rate": 0.0,
            "average_latency": 0.0,
            "error_rate": 0.0,
            "models_in_memory": 0,
            "dlq_size": 0,
            "active_connections": 0
        }


# Global metrics instance
_metrics: Optional[EmbeddingMetrics] = None


def get_metrics() -> EmbeddingMetrics:
    """Get or create the global metrics instance."""
    global _metrics
    if _metrics is None:
        _metrics = EmbeddingMetrics()
    return _metrics


# Convenience functions for common metrics
def track_embedding_request(provider: str, model: str):
    """Track an embedding request"""
    metrics = get_metrics()
    
    def decorator(func):
        if asyncio.iscoroutinefunction(func):
            @wraps(func)
            async def async_wrapper(*args, **kwargs):
                start_time = time.time()
                try:
                    result = await func(*args, **kwargs)
                    elapsed = time.time() - start_time
                    
                    metrics.log_request(provider, model, "success")
                    metrics.log_request_latency(provider, model, elapsed)
                    
                    return result
                except Exception as e:
                    metrics.log_request(provider, model, "failure")
                    metrics.log_error(provider, type(e).__name__)
                    raise
            return async_wrapper
        else:
            @wraps(func)
            def sync_wrapper(*args, **kwargs):
                start_time = time.time()
                try:
                    result = func(*args, **kwargs)
                    elapsed = time.time() - start_time
                    
                    metrics.log_request(provider, model, "success")
                    metrics.log_request_latency(provider, model, elapsed)
                    
                    return result
                except Exception as e:
                    metrics.log_request(provider, model, "failure")
                    metrics.log_error(provider, type(e).__name__)
                    raise
            return sync_wrapper
    return decorator


def track_cache_access(model: str):
    """Track cache access"""
    metrics = get_metrics()
    
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            result = func(*args, **kwargs)
            
            if result is not None:
                metrics.log_cache_hit(model)
            else:
                metrics.log_cache_miss(model)
            
            return result
        return wrapper
    return decorator


# Periodic metrics collection
async def collect_periodic_metrics(interval_seconds: int = 60):
    """
    Collect metrics periodically.
    
    Args:
        interval_seconds: Collection interval
    """
    metrics = get_metrics()
    
    while True:
        try:
            # Collect resource usage
            metrics.log_resource_usage_stats()
            
            # You can add more periodic collections here
            # e.g., checking connection pool status, cache size, etc.
            
            await asyncio.sleep(interval_seconds)
            
        except Exception as e:
            logger.error(f"Error collecting periodic metrics: {e}")
            await asyncio.sleep(interval_seconds)