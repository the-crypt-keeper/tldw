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
        
        # Rate limiting metrics
        self.rate_limit_violations = Counter(
            'rate_limit_violations_total',
            'Total rate limit violations',
            ['user_tier', 'limit_type', 'endpoint']
        )
        
        self.rate_limit_usage = Histogram(
            'rate_limit_usage_ratio',
            'Rate limit usage as percentage of limit',
            ['user_tier', 'limit_type'],
            buckets=(0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 1.0)
        )
        
        # Security metrics
        self.authentication_attempts = Counter(
            'authentication_attempts_total',
            'Total authentication attempts',
            ['outcome', 'method']
        )
        
        self.suspicious_activities = Counter(
            'suspicious_activities_total',
            'Total suspicious activities detected',
            ['activity_type', 'severity']
        )
        
        # Webhook metrics
        self.webhook_deliveries = Counter(
            'webhook_deliveries_total',
            'Total webhook delivery attempts',
            ['event_type', 'outcome']
        )
        
        self.webhook_response_time = Histogram(
            'webhook_response_time_seconds',
            'Webhook delivery response time',
            ['event_type'],
            buckets=(0.1, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0)
        )
        
        # Cache and database metrics
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
        
        # Additional database metrics
        self.database_query_duration = Histogram(
            'database_query_duration_seconds',
            'Database query execution time',
            ['operation', 'table'],
            buckets=(0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0)
        )
        
        self.database_errors = Counter(
            'database_errors_total',
            'Total database errors',
            ['operation', 'error_type']
        )
        
        # User tier metrics
        self.user_tier_distribution = Gauge(
            'user_tier_distribution',
            'Number of users by tier',
            ['tier']
        )
        
        self.evaluation_cost = Counter(
            'evaluation_cost_total',
            'Total evaluation costs',
            ['user_tier', 'provider', 'evaluation_type']
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
    
    def record_rate_limit_violation(self, user_tier: str, limit_type: str, endpoint: str):
        """Record a rate limit violation"""
        if not self.enabled:
            return
            
        self.rate_limit_violations.labels(
            user_tier=user_tier,
            limit_type=limit_type,
            endpoint=endpoint
        ).inc()
    
    def record_rate_limit_usage(self, user_tier: str, limit_type: str, usage_ratio: float):
        """Record rate limit usage ratio"""
        if not self.enabled:
            return
            
        self.rate_limit_usage.labels(
            user_tier=user_tier,
            limit_type=limit_type
        ).observe(usage_ratio)
    
    def record_authentication(self, outcome: str, method: str):
        """Record authentication attempt"""
        if not self.enabled:
            return
            
        self.authentication_attempts.labels(
            outcome=outcome,
            method=method
        ).inc()
    
    def record_suspicious_activity(self, activity_type: str, severity: str):
        """Record suspicious activity"""
        if not self.enabled:
            return
            
        self.suspicious_activities.labels(
            activity_type=activity_type,
            severity=severity
        ).inc()
    
    def record_webhook_delivery(self, event_type: str, outcome: str, response_time: float):
        """Record webhook delivery metrics"""
        if not self.enabled:
            return
            
        self.webhook_deliveries.labels(
            event_type=event_type,
            outcome=outcome
        ).inc()
        
        if response_time > 0:
            self.webhook_response_time.labels(
                event_type=event_type
            ).observe(response_time)
    
    def record_database_query(self, operation: str, table: str, duration: float, success: bool = True, error_type: str = None):
        """Record database query metrics"""
        if not self.enabled:
            return
            
        self.database_query_duration.labels(
            operation=operation,
            table=table
        ).observe(duration)
        
        if not success and error_type:
            self.database_errors.labels(
                operation=operation,
                error_type=error_type
            ).inc()
    
    def update_user_tier_distribution(self, tier_counts: Dict[str, int]):
        """Update user tier distribution"""
        if not self.enabled:
            return
            
        for tier, count in tier_counts.items():
            self.user_tier_distribution.labels(tier=tier).set(count)
    
    def record_evaluation_cost(self, user_tier: str, provider: str, evaluation_type: str, cost: float):
        """Record evaluation cost"""
        if not self.enabled:
            return
            
        self.evaluation_cost.labels(
            user_tier=user_tier,
            provider=provider,
            evaluation_type=evaluation_type
        ).inc(cost)
    
    def get_health_metrics(self) -> Dict[str, Any]:
        """Get system health metrics for monitoring"""
        if not self.enabled:
            return {"metrics_enabled": False}
        
        # This would typically query the metrics registry for current values
        # For simplicity, we'll return basic health info
        return {
            "metrics_enabled": True,
            "prometheus_registry_metrics": len(list(REGISTRY._collector_to_names.keys())),
            "last_updated": datetime.utcnow().isoformat()
        }
    
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