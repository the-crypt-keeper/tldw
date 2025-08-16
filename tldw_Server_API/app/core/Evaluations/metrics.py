# metrics.py - Metrics collection for evaluation service
"""
Metrics collection for the evaluation service using Prometheus client.

Provides metrics for:
- Request counts and latencies
- Circuit breaker state changes
- Evaluation success/failure rates
- Resource utilization
"""

import time
from typing import Dict, Any, Optional, Callable
from functools import wraps
from contextlib import contextmanager
from datetime import datetime

try:
    from prometheus_client import Counter, Histogram, Gauge, Info, generate_latest, REGISTRY
    PROMETHEUS_AVAILABLE = True
except ImportError:
    PROMETHEUS_AVAILABLE = False

from loguru import logger


class EvaluationMetrics:
    """Metrics collector for evaluation service"""
    
    def __init__(self):
        """Initialize metrics collectors"""
        self.enabled = PROMETHEUS_AVAILABLE
        
        if not self.enabled:
            logger.warning("Prometheus client not installed. Metrics collection disabled.")
            logger.info("Install with: pip install prometheus-client")
            return
            
        # Request metrics
        self.request_counter = Counter(
            'evaluation_requests_total',
            'Total number of evaluation requests',
            ['endpoint', 'method', 'status']
        )
        
        self.request_duration = Histogram(
            'evaluation_request_duration_seconds',
            'Request duration in seconds',
            ['endpoint', 'method'],
            buckets=(0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0, 60.0)
        )
        
        # Evaluation-specific metrics
        self.evaluation_counter = Counter(
            'evaluations_total',
            'Total number of evaluations',
            ['type', 'provider', 'status']
        )
        
        self.evaluation_duration = Histogram(
            'evaluation_duration_seconds',
            'Evaluation processing time',
            ['type', 'provider'],
            buckets=(0.5, 1.0, 2.5, 5.0, 10.0, 30.0, 60.0, 120.0)
        )
        
        self.evaluation_score = Histogram(
            'evaluation_scores',
            'Distribution of evaluation scores',
            ['type', 'metric'],
            buckets=(0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0)
        )
        
        # Circuit breaker metrics
        self.circuit_breaker_state = Gauge(
            'circuit_breaker_state',
            'Circuit breaker state (0=closed, 1=open, 2=half-open)',
            ['provider']
        )
        
        self.circuit_breaker_failures = Counter(
            'circuit_breaker_failures_total',
            'Total circuit breaker failures',
            ['provider', 'error_type']
        )
        
        # Resource metrics
        self.active_evaluations = Gauge(
            'active_evaluations',
            'Number of currently active evaluations',
            ['type']
        )
        
        self.database_connections = Gauge(
            'evaluation_database_connections',
            'Number of active database connections'
        )
        
        self.embedding_cache_hits = Counter(
            'embedding_cache_hits_total',
            'Total embedding cache hits',
            ['provider']
        )
        
        self.embedding_cache_misses = Counter(
            'embedding_cache_misses_total',
            'Total embedding cache misses',
            ['provider']
        )
        
        # Rate limiting metrics
        self.rate_limit_hits = Counter(
            'rate_limit_hits_total',
            'Total rate limit hits',
            ['endpoint', 'client_type']
        )
        
        # Error metrics
        self.error_counter = Counter(
            'evaluation_errors_total',
            'Total evaluation errors',
            ['type', 'error_category']
        )
        
        # Service info
        self.service_info = Info(
            'evaluation_service',
            'Evaluation service information'
        )
        
        self.service_info.info({
            'version': '1.0.0',
            'start_time': datetime.utcnow().isoformat()
        })
        
        logger.info("Evaluation metrics initialized")
    
    def record_request(self, endpoint: str, method: str, status: int, duration: float):
        """Record HTTP request metrics"""
        if not self.enabled:
            return
            
        self.request_counter.labels(
            endpoint=endpoint,
            method=method,
            status=str(status)
        ).inc()
        
        self.request_duration.labels(
            endpoint=endpoint,
            method=method
        ).observe(duration)
    
    def record_evaluation(
        self,
        eval_type: str,
        provider: str,
        status: str,
        duration: float,
        scores: Optional[Dict[str, float]] = None
    ):
        """Record evaluation metrics"""
        if not self.enabled:
            return
            
        self.evaluation_counter.labels(
            type=eval_type,
            provider=provider,
            status=status
        ).inc()
        
        self.evaluation_duration.labels(
            type=eval_type,
            provider=provider
        ).observe(duration)
        
        if scores:
            for metric_name, score in scores.items():
                self.evaluation_score.labels(
                    type=eval_type,
                    metric=metric_name
                ).observe(score)
    
    def update_circuit_breaker(self, provider: str, state: int, error_type: Optional[str] = None):
        """Update circuit breaker metrics"""
        if not self.enabled:
            return
            
        self.circuit_breaker_state.labels(provider=provider).set(state)
        
        if error_type:
            self.circuit_breaker_failures.labels(
                provider=provider,
                error_type=error_type
            ).inc()
    
    def record_cache_access(self, provider: str, hit: bool):
        """Record embedding cache access"""
        if not self.enabled:
            return
            
        if hit:
            self.embedding_cache_hits.labels(provider=provider).inc()
        else:
            self.embedding_cache_misses.labels(provider=provider).inc()
    
    def record_rate_limit(self, endpoint: str, client_type: str = "unknown"):
        """Record rate limit hit"""
        if not self.enabled:
            return
            
        self.rate_limit_hits.labels(
            endpoint=endpoint,
            client_type=client_type
        ).inc()
    
    def record_error(self, eval_type: str, error_category: str):
        """Record evaluation error"""
        if not self.enabled:
            return
            
        self.error_counter.labels(
            type=eval_type,
            error_category=error_category
        ).inc()
    
    @contextmanager
    def track_active_evaluation(self, eval_type: str):
        """Context manager to track active evaluations"""
        if not self.enabled:
            yield
            return
            
        self.active_evaluations.labels(type=eval_type).inc()
        try:
            yield
        finally:
            self.active_evaluations.labels(type=eval_type).dec()
    
    def set_database_connections(self, count: int):
        """Update database connection count"""
        if not self.enabled:
            return
            
        self.database_connections.set(count)
    
    def get_metrics(self) -> bytes:
        """Generate Prometheus metrics output"""
        if not self.enabled:
            return b"# Metrics collection disabled"
            
        return generate_latest(REGISTRY)


# Global metrics instance
_metrics_instance: Optional[EvaluationMetrics] = None


def get_metrics() -> EvaluationMetrics:
    """Get or create global metrics instance"""
    global _metrics_instance
    if _metrics_instance is None:
        _metrics_instance = EvaluationMetrics()
    return _metrics_instance


def track_request_metrics(endpoint: str):
    """Decorator to track request metrics"""
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            start_time = time.time()
            status = 200
            
            try:
                result = await func(*args, **kwargs)
                return result
            except Exception as e:
                status = getattr(e, 'status_code', 500)
                raise
            finally:
                duration = time.time() - start_time
                metrics = get_metrics()
                metrics.record_request(
                    endpoint=endpoint,
                    method='POST',
                    status=status,
                    duration=duration
                )
        
        @wraps(func)
        def sync_wrapper(*args, **kwargs):
            start_time = time.time()
            status = 200
            
            try:
                result = func(*args, **kwargs)
                return result
            except Exception as e:
                status = getattr(e, 'status_code', 500)
                raise
            finally:
                duration = time.time() - start_time
                metrics = get_metrics()
                metrics.record_request(
                    endpoint=endpoint,
                    method='POST',
                    status=status,
                    duration=duration
                )
        
        # Return appropriate wrapper based on function type
        import asyncio
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        else:
            return sync_wrapper
    
    return decorator


def track_evaluation_metrics(eval_type: str):
    """Decorator to track evaluation metrics"""
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            metrics = get_metrics()
            start_time = time.time()
            
            with metrics.track_active_evaluation(eval_type):
                try:
                    result = await func(*args, **kwargs)
                    
                    # Extract provider and scores from result if available
                    provider = kwargs.get('api_name', 'unknown')
                    scores = None
                    
                    if hasattr(result, 'metrics'):
                        scores = {
                            name: metric.score 
                            for name, metric in result.metrics.items()
                        }
                    
                    metrics.record_evaluation(
                        eval_type=eval_type,
                        provider=provider,
                        status='success',
                        duration=time.time() - start_time,
                        scores=scores
                    )
                    
                    return result
                    
                except Exception as e:
                    metrics.record_evaluation(
                        eval_type=eval_type,
                        provider=kwargs.get('api_name', 'unknown'),
                        status='failure',
                        duration=time.time() - start_time
                    )
                    
                    # Categorize error
                    if 'rate limit' in str(e).lower():
                        error_category = 'rate_limit'
                    elif 'timeout' in str(e).lower():
                        error_category = 'timeout'
                    elif 'api' in str(e).lower():
                        error_category = 'api_error'
                    else:
                        error_category = 'unknown'
                    
                    metrics.record_error(eval_type, error_category)
                    raise
        
        @wraps(func)
        def sync_wrapper(*args, **kwargs):
            metrics = get_metrics()
            start_time = time.time()
            
            with metrics.track_active_evaluation(eval_type):
                try:
                    result = func(*args, **kwargs)
                    
                    provider = kwargs.get('api_name', 'unknown')
                    scores = None
                    
                    if hasattr(result, 'metrics'):
                        scores = {
                            name: metric.score 
                            for name, metric in result.metrics.items()
                        }
                    
                    metrics.record_evaluation(
                        eval_type=eval_type,
                        provider=provider,
                        status='success',
                        duration=time.time() - start_time,
                        scores=scores
                    )
                    
                    return result
                    
                except Exception as e:
                    metrics.record_evaluation(
                        eval_type=eval_type,
                        provider=kwargs.get('api_name', 'unknown'),
                        status='failure',
                        duration=time.time() - start_time
                    )
                    
                    error_category = 'unknown'
                    if 'rate limit' in str(e).lower():
                        error_category = 'rate_limit'
                    elif 'timeout' in str(e).lower():
                        error_category = 'timeout'
                    elif 'api' in str(e).lower():
                        error_category = 'api_error'
                    
                    metrics.record_error(eval_type, error_category)
                    raise
        
        import asyncio
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        else:
            return sync_wrapper
    
    return decorator