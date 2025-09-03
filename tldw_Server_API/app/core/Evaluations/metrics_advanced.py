"""
Advanced metrics collection for Evaluations module.

Extends basic Prometheus metrics with business metrics, SLI/SLO tracking,
and custom exporters for comprehensive observability.
"""

import time
from typing import Dict, Any, Optional, Callable, List
from functools import wraps
from contextlib import contextmanager
from datetime import datetime, timedelta
from collections import defaultdict
from loguru import logger

try:
    from prometheus_client import (
        Counter, Histogram, Gauge, Info, Summary,
        generate_latest, REGISTRY, CollectorRegistry,
        push_to_gateway, Enum
    )
    PROMETHEUS_AVAILABLE = True
except ImportError:
    PROMETHEUS_AVAILABLE = False
    logger.warning("Prometheus client not installed. Advanced metrics disabled.")


class AdvancedEvaluationMetrics:
    """Advanced metrics collector for evaluation service."""
    
    def __init__(self, registry: Optional[CollectorRegistry] = None, use_separate_registry: bool = False):
        """Initialize advanced metrics collectors."""
        self.enabled = PROMETHEUS_AVAILABLE
        
        if use_separate_registry:
            # For testing, use a new registry to avoid conflicts
            self.registry = CollectorRegistry() if PROMETHEUS_AVAILABLE else None
        else:
            self.registry = registry or (REGISTRY if PROMETHEUS_AVAILABLE else None)
        
        if not self.enabled:
            return
        
        # === Business Metrics ===
        
        # Cost tracking
        self.evaluation_cost = Counter(
            'evaluation_cost_total_dollars',
            'Total cost of evaluations in dollars',
            ['user_tier', 'provider', 'model', 'evaluation_type'],
            registry=self.registry
        )
        
        self.user_spend_gauge = Gauge(
            'user_spend_dollars',
            'Current user spend',
            ['user_id', 'period'],  # period: daily, monthly
            registry=self.registry
        )
        
        # Accuracy and quality metrics
        self.evaluation_accuracy = Histogram(
            'evaluation_accuracy_score',
            'Evaluation accuracy scores',
            ['evaluation_type', 'model'],
            buckets=(0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99, 1.0),
            registry=self.registry
        )
        
        self.evaluation_confidence = Histogram(
            'evaluation_confidence_score',
            'Evaluation confidence scores',
            ['evaluation_type'],
            buckets=(0.1, 0.3, 0.5, 0.7, 0.9, 1.0),
            registry=self.registry
        )
        
        # User engagement
        self.active_users = Gauge(
            'active_users_total',
            'Number of active users',
            ['tier', 'time_window'],  # time_window: 1h, 24h, 7d, 30d
            registry=self.registry
        )
        
        self.user_retention = Gauge(
            'user_retention_rate',
            'User retention rate',
            ['cohort', 'days_since_signup'],
            registry=self.registry
        )
        
        # === SLI/SLO Metrics ===
        
        # Availability SLI
        self.service_up = Gauge(
            'evaluation_service_up',
            'Whether the evaluation service is up (1) or down (0)',
            registry=self.registry
        )
        
        # Latency SLI
        self.request_latency_sli = Summary(
            'evaluation_request_latency_seconds_sli',
            'Request latency for SLI calculation',
            ['endpoint', 'percentile'],
            registry=self.registry
        )
        
        # Error rate SLI
        self.error_rate_sli = Gauge(
            'evaluation_error_rate_sli',
            'Error rate for SLI calculation',
            ['time_window'],
            registry=self.registry
        )
        
        # SLO compliance
        self.slo_compliance = Gauge(
            'evaluation_slo_compliance',
            'SLO compliance percentage',
            ['slo_type', 'target'],  # slo_type: availability, latency, error_rate
            registry=self.registry
        )
        
        # Error budget
        self.error_budget_remaining = Gauge(
            'evaluation_error_budget_remaining_percentage',
            'Remaining error budget as percentage',
            ['slo_type'],
            registry=self.registry
        )
        
        # === Rate Limiting Metrics ===
        
        self.rate_limit_hits = Counter(
            'rate_limit_hits_total',
            'Total number of rate limit hits',
            ['user_tier', 'limit_type'],  # limit_type: minute, daily, token, cost
            registry=self.registry
        )
        
        self.rate_limit_utilization = Gauge(
            'rate_limit_utilization_percentage',
            'Rate limit utilization percentage',
            ['user_id', 'limit_type'],
            registry=self.registry
        )
        
        # === Webhook Metrics ===
        
        self.webhook_deliveries = Counter(
            'webhook_deliveries_total',
            'Total webhook deliveries',
            ['event_type', 'status'],  # status: success, failure
            registry=self.registry
        )
        
        self.webhook_latency = Histogram(
            'webhook_delivery_latency_seconds',
            'Webhook delivery latency',
            ['event_type'],
            buckets=(0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 30.0),
            registry=self.registry
        )
        
        self.webhook_retry_count = Counter(
            'webhook_retries_total',
            'Total webhook retry attempts',
            ['event_type'],
            registry=self.registry
        )
        
        # === Model Performance Metrics ===
        
        self.model_performance = Histogram(
            'model_evaluation_performance',
            'Model performance scores',
            ['model', 'evaluation_type', 'metric_name'],
            buckets=(0.1, 0.3, 0.5, 0.7, 0.9, 1.0),
            registry=self.registry
        )
        
        self.model_comparison = Gauge(
            'model_comparison_delta',
            'Performance delta between models',
            ['model_a', 'model_b', 'metric'],
            registry=self.registry
        )
        
        # === Resource Utilization ===
        
        self.token_utilization = Histogram(
            'token_utilization_per_request',
            'Tokens used per request',
            ['provider', 'model'],
            buckets=(10, 50, 100, 500, 1000, 5000, 10000, 50000),
            registry=self.registry
        )
        
        self.database_connections = Gauge(
            'database_connections_active',
            'Active database connections',
            ['database_type'],
            registry=self.registry
        )
        
        self.queue_depth = Gauge(
            'evaluation_queue_depth',
            'Number of evaluations in queue',
            ['priority'],
            registry=self.registry
        )
        
        # === Custom Business Metrics ===
        
        self.evaluation_value = Counter(
            'evaluation_business_value_total',
            'Total business value generated by evaluations',
            ['customer_segment', 'use_case'],
            registry=self.registry
        )
        
        self.feature_adoption = Gauge(
            'feature_adoption_rate',
            'Feature adoption rate',
            ['feature_name', 'user_tier'],
            registry=self.registry
        )
        
        # Initialize SLO targets
        self.slo_targets = {
            'availability': 0.999,  # 99.9%
            'latency_p95': 2.0,     # 95th percentile under 2s
            'latency_p99': 5.0,     # 99th percentile under 5s
            'error_rate': 0.001     # 0.1% error rate
        }
        
        # Tracking for SLI calculations
        self._request_count = 0
        self._error_count = 0
        self._latency_buffer = []
        self._last_sli_calculation = time.time()
    
    def track_evaluation_cost(
        self,
        user_tier: str,
        provider: str,
        model: str,
        evaluation_type: str,
        cost: float
    ):
        """Track evaluation cost."""
        if not self.enabled:
            return
        
        self.evaluation_cost.labels(
            user_tier=user_tier,
            provider=provider,
            model=model,
            evaluation_type=evaluation_type
        ).inc(cost)
    
    def track_user_spend(self, user_id: str, daily_spend: float, monthly_spend: float):
        """Track user spending."""
        if not self.enabled:
            return
        
        self.user_spend_gauge.labels(user_id=user_id, period='daily').set(daily_spend)
        self.user_spend_gauge.labels(user_id=user_id, period='monthly').set(monthly_spend)
    
    def track_evaluation_quality(
        self,
        evaluation_type: str,
        model: str,
        accuracy: float,
        confidence: float
    ):
        """Track evaluation quality metrics."""
        if not self.enabled:
            return
        
        self.evaluation_accuracy.labels(
            evaluation_type=evaluation_type,
            model=model
        ).observe(accuracy)
        
        self.evaluation_confidence.labels(
            evaluation_type=evaluation_type
        ).observe(confidence)
    
    def track_rate_limit_hit(self, user_tier: str, limit_type: str):
        """Track rate limit hit."""
        if not self.enabled:
            return
        
        self.rate_limit_hits.labels(
            user_tier=user_tier,
            limit_type=limit_type
        ).inc()
    
    def track_rate_limit_utilization(self, user_id: str, limit_type: str, utilization: float):
        """Track rate limit utilization."""
        if not self.enabled:
            return
        
        self.rate_limit_utilization.labels(
            user_id=user_id,
            limit_type=limit_type
        ).set(utilization * 100)  # Convert to percentage
    
    def track_webhook_delivery(
        self,
        event_type: str,
        success: bool,
        latency: Optional[float] = None,
        retry_count: int = 0
    ):
        """Track webhook delivery metrics."""
        if not self.enabled:
            return
        
        status = 'success' if success else 'failure'
        self.webhook_deliveries.labels(
            event_type=event_type,
            status=status
        ).inc()
        
        if latency is not None:
            self.webhook_latency.labels(event_type=event_type).observe(latency)
        
        if retry_count > 0:
            self.webhook_retry_count.labels(event_type=event_type).inc(retry_count)
    
    def track_model_performance(
        self,
        model: str,
        evaluation_type: str,
        metrics: Dict[str, float]
    ):
        """Track model performance metrics."""
        if not self.enabled:
            return
        
        for metric_name, value in metrics.items():
            self.model_performance.labels(
                model=model,
                evaluation_type=evaluation_type,
                metric_name=metric_name
            ).observe(value)
    
    def compare_models(self, model_a: str, model_b: str, metric: str, delta: float):
        """Track model comparison metrics."""
        if not self.enabled:
            return
        
        self.model_comparison.labels(
            model_a=model_a,
            model_b=model_b,
            metric=metric
        ).set(delta)
    
    @contextmanager
    def track_sli_request(self, endpoint: str):
        """Context manager to track SLI metrics for a request."""
        if not self.enabled:
            yield
            return
        
        start_time = time.time()
        error_occurred = False
        
        try:
            yield
        except Exception as e:
            error_occurred = True
            self._error_count += 1
            raise
        finally:
            latency = time.time() - start_time
            self._request_count += 1
            self._latency_buffer.append(latency)
            
            # Keep buffer size manageable
            if len(self._latency_buffer) > 1000:
                self._latency_buffer = self._latency_buffer[-1000:]
            
            # Update SLI metrics
            self.request_latency_sli.labels(
                endpoint=endpoint,
                percentile='p50'
            ).observe(latency)
            
            # Calculate SLOs periodically
            if time.time() - self._last_sli_calculation > 60:  # Every minute
                self._calculate_slos()
    
    def _calculate_slos(self):
        """Calculate and update SLO compliance metrics."""
        if not self.enabled or self._request_count == 0:
            return
        
        # Calculate availability SLO
        availability = 1.0 - (self._error_count / self._request_count)
        availability_compliance = (availability / self.slo_targets['availability']) * 100
        self.slo_compliance.labels(
            slo_type='availability',
            target=str(self.slo_targets['availability'])
        ).set(min(availability_compliance, 100))
        
        # Calculate error rate SLO
        error_rate = self._error_count / self._request_count
        self.error_rate_sli.labels(time_window='1m').set(error_rate)
        
        error_rate_compliance = ((self.slo_targets['error_rate'] - error_rate) / 
                                self.slo_targets['error_rate']) * 100
        self.slo_compliance.labels(
            slo_type='error_rate',
            target=str(self.slo_targets['error_rate'])
        ).set(max(error_rate_compliance, 0))
        
        # Calculate latency SLOs if we have data
        if self._latency_buffer:
            sorted_latencies = sorted(self._latency_buffer)
            p95_index = int(len(sorted_latencies) * 0.95)
            p99_index = int(len(sorted_latencies) * 0.99)
            
            p95_latency = sorted_latencies[p95_index] if p95_index < len(sorted_latencies) else sorted_latencies[-1]
            p99_latency = sorted_latencies[p99_index] if p99_index < len(sorted_latencies) else sorted_latencies[-1]
            
            # P95 latency compliance
            p95_compliance = (self.slo_targets['latency_p95'] / max(p95_latency, 0.001)) * 100
            self.slo_compliance.labels(
                slo_type='latency_p95',
                target=str(self.slo_targets['latency_p95'])
            ).set(min(p95_compliance, 100))
            
            # P99 latency compliance
            p99_compliance = (self.slo_targets['latency_p99'] / max(p99_latency, 0.001)) * 100
            self.slo_compliance.labels(
                slo_type='latency_p99',
                target=str(self.slo_targets['latency_p99'])
            ).set(min(p99_compliance, 100))
        
        # Calculate error budgets
        monthly_requests = self._request_count * 43200  # Extrapolate to month (30 days)
        availability_budget = (1 - self.slo_targets['availability']) * monthly_requests
        errors_used = self._error_count * 43200 / max(self._request_count, 1)
        
        budget_remaining = max(0, (availability_budget - errors_used) / availability_budget) * 100
        self.error_budget_remaining.labels(slo_type='availability').set(budget_remaining)
        
        # Reset counters periodically
        self._last_sli_calculation = time.time()
        if self._request_count > 10000:
            self._request_count = 0
            self._error_count = 0
    
    def update_active_users(self, tier_counts: Dict[str, Dict[str, int]]):
        """Update active user counts."""
        if not self.enabled:
            return
        
        for tier, windows in tier_counts.items():
            for window, count in windows.items():
                self.active_users.labels(tier=tier, time_window=window).set(count)
    
    def track_feature_adoption(self, feature_name: str, tier: str, adoption_rate: float):
        """Track feature adoption rates."""
        if not self.enabled:
            return
        
        self.feature_adoption.labels(
            feature_name=feature_name,
            user_tier=tier
        ).set(adoption_rate * 100)
    
    def track_queue_depth(self, priority: str, depth: int):
        """Track evaluation queue depth."""
        if not self.enabled:
            return
        
        self.queue_depth.labels(priority=priority).set(depth)
    
    def track_database_connections(self, db_type: str, active_connections: int):
        """Track database connection pool status."""
        if not self.enabled:
            return
        
        self.database_connections.labels(database_type=db_type).set(active_connections)
    
    def get_metrics(self) -> str:
        """Get metrics in Prometheus format."""
        if not self.enabled:
            return ""
        
        return generate_latest(self.registry).decode('utf-8')
    
    def push_to_gateway(self, gateway_url: str, job_name: str = 'evaluation_service'):
        """Push metrics to Prometheus Pushgateway."""
        if not self.enabled:
            return
        
        try:
            push_to_gateway(gateway_url, job=job_name, registry=self.registry)
            logger.info(f"Pushed metrics to gateway: {gateway_url}")
        except Exception as e:
            logger.error(f"Failed to push metrics to gateway: {e}")


# Global instance - use singleton pattern to avoid duplicate registration
_advanced_metrics_instance = None

def get_advanced_metrics(use_separate_registry: bool = False):
    """Get or create the global advanced metrics instance."""
    global _advanced_metrics_instance
    if _advanced_metrics_instance is None or use_separate_registry:
        try:
            _advanced_metrics_instance = AdvancedEvaluationMetrics(use_separate_registry=use_separate_registry)
        except ValueError as e:
            # If metrics are already registered (e.g., during reload), create without registry
            if "Duplicated timeseries" in str(e):
                logger.warning("Metrics already registered, creating instance with separate registry")
                _advanced_metrics_instance = AdvancedEvaluationMetrics(use_separate_registry=True)
            else:
                raise
    return _advanced_metrics_instance

def reset_advanced_metrics():
    """Reset the global advanced metrics instance (for testing)."""
    global _advanced_metrics_instance
    _advanced_metrics_instance = None

# Create the global instance
advanced_metrics = get_advanced_metrics()