"""
Centralized metrics management for the tldw_server application.

This module provides a unified interface for all metric operations,
supporting both OpenTelemetry and fallback implementations.
"""

import time
import asyncio
from typing import Dict, Any, Optional, List, Callable, Union
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from collections import defaultdict, deque
from contextlib import contextmanager
import statistics

from loguru import logger

from .telemetry import get_telemetry_manager, OTEL_AVAILABLE

if OTEL_AVAILABLE:
    from opentelemetry.metrics import CallbackOptions, Observation


class MetricType(Enum):
    """Types of metrics supported."""
    COUNTER = "counter"
    GAUGE = "gauge"
    HISTOGRAM = "histogram"
    UP_DOWN_COUNTER = "up_down_counter"


@dataclass
class MetricDefinition:
    """Definition of a metric."""
    name: str
    type: MetricType
    description: str
    unit: str = ""
    labels: List[str] = field(default_factory=list)
    buckets: Optional[List[float]] = None  # For histograms


@dataclass
class MetricValue:
    """A metric value with metadata."""
    value: float
    timestamp: float = field(default_factory=time.time)
    labels: Dict[str, str] = field(default_factory=dict)


class MetricsRegistry:
    """Registry for all application metrics."""
    
    def __init__(self):
        """Initialize the metrics registry."""
        self.metrics: Dict[str, MetricDefinition] = {}
        self.instruments: Dict[str, Any] = {}
        self.values: Dict[str, deque] = defaultdict(lambda: deque(maxlen=1000))
        self.callbacks: Dict[str, List[Callable]] = defaultdict(list)
        
        # Initialize with telemetry manager
        self.telemetry = get_telemetry_manager()
        self.meter = self.telemetry.get_meter("tldw_server.metrics")
        
        # Register standard metrics
        self._register_standard_metrics()
    
    def _register_standard_metrics(self):
        """Register standard application metrics."""
        # HTTP metrics
        self.register_metric(
            MetricDefinition(
                name="http_requests_total",
                type=MetricType.COUNTER,
                description="Total number of HTTP requests",
                labels=["method", "endpoint", "status"]
            )
        )
        
        self.register_metric(
            MetricDefinition(
                name="http_request_duration_seconds",
                type=MetricType.HISTOGRAM,
                description="HTTP request duration in seconds",
                unit="s",
                labels=["method", "endpoint"],
                buckets=[0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1, 2.5, 5, 10]
            )
        )
        
        # Database metrics
        self.register_metric(
            MetricDefinition(
                name="db_connections_active",
                type=MetricType.GAUGE,
                description="Number of active database connections",
                labels=["database"]
            )
        )
        
        self.register_metric(
            MetricDefinition(
                name="db_queries_total",
                type=MetricType.COUNTER,
                description="Total number of database queries",
                labels=["database", "operation"]
            )
        )
        
        self.register_metric(
            MetricDefinition(
                name="db_query_duration_seconds",
                type=MetricType.HISTOGRAM,
                description="Database query duration in seconds",
                unit="s",
                labels=["database", "operation"],
                buckets=[0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1, 5]
            )
        )
        
        # LLM metrics
        self.register_metric(
            MetricDefinition(
                name="llm_requests_total",
                type=MetricType.COUNTER,
                description="Total number of LLM API requests",
                labels=["provider", "model", "status"]
            )
        )
        
        self.register_metric(
            MetricDefinition(
                name="llm_tokens_used_total",
                type=MetricType.COUNTER,
                description="Total number of tokens used",
                labels=["provider", "model", "type"]  # type: prompt/completion
            )
        )
        
        self.register_metric(
            MetricDefinition(
                name="llm_request_duration_seconds",
                type=MetricType.HISTOGRAM,
                description="LLM request duration in seconds",
                unit="s",
                labels=["provider", "model"],
                buckets=[0.1, 0.5, 1, 2.5, 5, 10, 30, 60]
            )
        )
        
        self.register_metric(
            MetricDefinition(
                name="llm_cost_dollars",
                type=MetricType.COUNTER,
                description="Cumulative LLM API cost in dollars",
                unit="$",
                labels=["provider", "model"]
            )
        )
        
        # RAG metrics
        self.register_metric(
            MetricDefinition(
                name="rag_queries_total",
                type=MetricType.COUNTER,
                description="Total number of RAG queries",
                labels=["pipeline", "status"]
            )
        )
        
        self.register_metric(
            MetricDefinition(
                name="rag_retrieval_latency_seconds",
                type=MetricType.HISTOGRAM,
                description="RAG retrieval latency in seconds",
                unit="s",
                labels=["source", "pipeline"],
                buckets=[0.01, 0.05, 0.1, 0.25, 0.5, 1, 2.5, 5]
            )
        )
        
        self.register_metric(
            MetricDefinition(
                name="rag_documents_retrieved",
                type=MetricType.HISTOGRAM,
                description="Number of documents retrieved",
                labels=["source", "pipeline"],
                buckets=[1, 5, 10, 25, 50, 100, 250, 500]
            )
        )
        
        self.register_metric(
            MetricDefinition(
                name="rag_cache_hits_total",
                type=MetricType.COUNTER,
                description="Total RAG cache hits",
                labels=["cache_type"]
            )
        )
        
        self.register_metric(
            MetricDefinition(
                name="rag_cache_misses_total",
                type=MetricType.COUNTER,
                description="Total RAG cache misses",
                labels=["cache_type"]
            )
        )
        
        # Embedding metrics
        self.register_metric(
            MetricDefinition(
                name="embeddings_generated_total",
                type=MetricType.COUNTER,
                description="Total number of embeddings generated",
                labels=["provider", "model"]
            )
        )
        
        self.register_metric(
            MetricDefinition(
                name="embedding_generation_duration_seconds",
                type=MetricType.HISTOGRAM,
                description="Embedding generation duration in seconds",
                unit="s",
                labels=["provider", "model"],
                buckets=[0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1]
            )
        )
        
        # System metrics
        self.register_metric(
            MetricDefinition(
                name="system_cpu_usage_percent",
                type=MetricType.GAUGE,
                description="System CPU usage percentage",
                unit="%"
            )
        )
        
        self.register_metric(
            MetricDefinition(
                name="system_memory_usage_bytes",
                type=MetricType.GAUGE,
                description="System memory usage in bytes",
                unit="bytes"
            )
        )
        
        self.register_metric(
            MetricDefinition(
                name="system_disk_usage_bytes",
                type=MetricType.GAUGE,
                description="System disk usage in bytes",
                unit="bytes",
                labels=["mount_point"]
            )
        )
        
        # Error metrics
        self.register_metric(
            MetricDefinition(
                name="errors_total",
                type=MetricType.COUNTER,
                description="Total number of errors",
                labels=["component", "error_type"]
            )
        )
        
        # Circuit breaker metrics
        self.register_metric(
            MetricDefinition(
                name="circuit_breaker_state",
                type=MetricType.GAUGE,
                description="Circuit breaker state (0=closed, 1=open, 2=half-open)",
                labels=["service"]
            )
        )
        
        self.register_metric(
            MetricDefinition(
                name="circuit_breaker_trips_total",
                type=MetricType.COUNTER,
                description="Total circuit breaker trips",
                labels=["service", "reason"]
            )
        )
    
    def register_metric(self, definition: MetricDefinition) -> bool:
        """
        Register a new metric definition.
        
        Args:
            definition: MetricDefinition object
            
        Returns:
            True if registered successfully
        """
        if definition.name in self.metrics:
            logger.warning(f"Metric {definition.name} already registered")
            return False
        
        self.metrics[definition.name] = definition
        
        # Create OpenTelemetry instrument
        if OTEL_AVAILABLE and self.meter:
            instrument = self._create_instrument(definition)
            if instrument:
                self.instruments[definition.name] = instrument
        
        logger.debug(f"Registered metric: {definition.name}")
        return True
    
    def _create_instrument(self, definition: MetricDefinition):
        """Create an OpenTelemetry instrument for a metric definition."""
        try:
            if definition.type == MetricType.COUNTER:
                return self.meter.create_counter(
                    name=definition.name,
                    description=definition.description,
                    unit=definition.unit
                )
            
            elif definition.type == MetricType.GAUGE:
                # Gauges are handled via callbacks in OpenTelemetry
                return self.meter.create_observable_gauge(
                    name=definition.name,
                    description=definition.description,
                    unit=definition.unit,
                    callbacks=[lambda options: self._gauge_callback(definition.name, options)]
                )
            
            elif definition.type == MetricType.HISTOGRAM:
                return self.meter.create_histogram(
                    name=definition.name,
                    description=definition.description,
                    unit=definition.unit
                )
            
            elif definition.type == MetricType.UP_DOWN_COUNTER:
                return self.meter.create_up_down_counter(
                    name=definition.name,
                    description=definition.description,
                    unit=definition.unit
                )
            
        except Exception as e:
            logger.error(f"Failed to create instrument for {definition.name}: {e}")
            return None
    
    def _gauge_callback(self, metric_name: str, options: 'CallbackOptions'):
        """Callback for observable gauge metrics."""
        observations = []
        
        # Get latest values for this metric
        if metric_name in self.values:
            values = list(self.values[metric_name])
            if values:
                latest = values[-1]
                observations.append(
                    Observation(
                        value=latest.value,
                        attributes=latest.labels
                    )
                )
        
        return observations
    
    def record(self, metric_name: str, value: float, labels: Optional[Dict[str, str]] = None):
        """
        Record a metric value.
        
        Args:
            metric_name: Name of the metric
            value: Value to record
            labels: Optional labels/dimensions
        """
        if metric_name not in self.metrics:
            logger.warning(f"Metric {metric_name} not registered")
            return
        
        definition = self.metrics[metric_name]
        labels = labels or {}
        
        # Store value for aggregation
        metric_value = MetricValue(value=value, labels=labels)
        self.values[metric_name].append(metric_value)
        
        # Record in OpenTelemetry
        if metric_name in self.instruments:
            instrument = self.instruments[metric_name]
            
            try:
                if definition.type == MetricType.COUNTER:
                    instrument.add(value, attributes=labels)
                
                elif definition.type == MetricType.HISTOGRAM:
                    instrument.record(value, attributes=labels)
                
                elif definition.type == MetricType.UP_DOWN_COUNTER:
                    instrument.add(value, attributes=labels)
                
                # Gauges are handled via callbacks
                
            except Exception as e:
                logger.error(f"Failed to record metric {metric_name}: {e}")
        
        # Execute callbacks
        for callback in self.callbacks[metric_name]:
            try:
                callback(metric_name, value, labels)
            except Exception as e:
                logger.error(f"Metric callback error: {e}")
    
    def increment(self, metric_name: str, value: float = 1, labels: Optional[Dict[str, str]] = None):
        """
        Increment a counter metric.
        
        Args:
            metric_name: Name of the counter metric
            value: Amount to increment (default 1)
            labels: Optional labels
        """
        self.record(metric_name, value, labels)
    
    def set_gauge(self, metric_name: str, value: float, labels: Optional[Dict[str, str]] = None):
        """
        Set a gauge metric value.
        
        Args:
            metric_name: Name of the gauge metric
            value: Value to set
            labels: Optional labels
        """
        self.record(metric_name, value, labels)
    
    def observe(self, metric_name: str, value: float, labels: Optional[Dict[str, str]] = None):
        """
        Observe a value for histogram metric.
        
        Args:
            metric_name: Name of the histogram metric
            value: Value to observe
            labels: Optional labels
        """
        self.record(metric_name, value, labels)
    
    @contextmanager
    def timer(self, metric_name: str, labels: Optional[Dict[str, str]] = None):
        """
        Context manager to time an operation.
        
        Args:
            metric_name: Name of the histogram metric for timing
            labels: Optional labels
            
        Yields:
            Timer context
        """
        start_time = time.time()
        try:
            yield
        finally:
            duration = time.time() - start_time
            self.observe(metric_name, duration, labels)
    
    def add_callback(self, metric_name: str, callback: Callable):
        """
        Add a callback for metric events.
        
        Args:
            metric_name: Name of the metric
            callback: Callable(metric_name, value, labels)
        """
        self.callbacks[metric_name].append(callback)
    
    def get_metric_stats(self, metric_name: str, labels: Optional[Dict[str, str]] = None) -> Dict[str, Any]:
        """
        Get statistics for a metric.
        
        Args:
            metric_name: Name of the metric
            labels: Optional label filter
            
        Returns:
            Dictionary with metric statistics
        """
        if metric_name not in self.values:
            return {}
        
        # Filter values by labels if provided
        values = list(self.values[metric_name])
        if labels:
            values = [v for v in values if all(
                v.labels.get(k) == v for k, v in labels.items()
            )]
        
        if not values:
            return {}
        
        numeric_values = [v.value for v in values]
        
        return {
            "count": len(numeric_values),
            "sum": sum(numeric_values),
            "mean": statistics.mean(numeric_values),
            "median": statistics.median(numeric_values),
            "min": min(numeric_values),
            "max": max(numeric_values),
            "stddev": statistics.stdev(numeric_values) if len(numeric_values) > 1 else 0,
            "latest": numeric_values[-1],
            "latest_timestamp": values[-1].timestamp
        }
    
    def get_all_metrics(self) -> Dict[str, Dict[str, Any]]:
        """
        Get all current metric values and statistics.
        
        Returns:
            Dictionary of metric names to their statistics
        """
        result = {}
        
        for metric_name, definition in self.metrics.items():
            stats = self.get_metric_stats(metric_name)
            if stats:
                result[metric_name] = {
                    "type": definition.type.value,
                    "description": definition.description,
                    "unit": definition.unit,
                    "stats": stats
                }
        
        return result
    
    def export_prometheus_format(self) -> str:
        """
        Export metrics in Prometheus text format.
        
        Returns:
            Prometheus-formatted metrics string
        """
        lines = []
        
        for metric_name, definition in self.metrics.items():
            if metric_name not in self.values:
                continue
            
            # Add HELP and TYPE lines
            lines.append(f"# HELP {metric_name} {definition.description}")
            lines.append(f"# TYPE {metric_name} {definition.type.value}")
            
            # Group values by labels
            label_groups = defaultdict(list)
            for value in self.values[metric_name]:
                label_key = ",".join(f'{k}="{v}"' for k, v in sorted(value.labels.items()))
                label_groups[label_key].append(value)
            
            # Export values
            for label_str, values in label_groups.items():
                if definition.type == MetricType.GAUGE:
                    # For gauges, use the latest value
                    latest = values[-1]
                    if label_str:
                        lines.append(f"{metric_name}{{{label_str}}} {latest.value}")
                    else:
                        lines.append(f"{metric_name} {latest.value}")
                
                elif definition.type in [MetricType.COUNTER, MetricType.UP_DOWN_COUNTER]:
                    # For counters, use the sum
                    total = sum(v.value for v in values)
                    if label_str:
                        lines.append(f"{metric_name}{{{label_str}}} {total}")
                    else:
                        lines.append(f"{metric_name} {total}")
                
                elif definition.type == MetricType.HISTOGRAM:
                    # For histograms, calculate buckets
                    numeric_values = [v.value for v in values]
                    if definition.buckets:
                        for bucket in definition.buckets:
                            count = sum(1 for v in numeric_values if v <= bucket)
                            bucket_label = f'le="{bucket}"'
                            if label_str:
                                lines.append(f"{metric_name}_bucket{{{label_str},{bucket_label}}} {count}")
                            else:
                                lines.append(f"{metric_name}_bucket{{{bucket_label}}} {count}")
                    
                    # Add +Inf bucket, sum, and count
                    if label_str:
                        lines.append(f"{metric_name}_bucket{{{label_str},le=\"+Inf\"}} {len(numeric_values)}")
                        lines.append(f"{metric_name}_sum{{{label_str}}} {sum(numeric_values)}")
                        lines.append(f"{metric_name}_count{{{label_str}}} {len(numeric_values)}")
                    else:
                        lines.append(f"{metric_name}_bucket{{le=\"+Inf\"}} {len(numeric_values)}")
                        lines.append(f"{metric_name}_sum {sum(numeric_values)}")
                        lines.append(f"{metric_name}_count {len(numeric_values)}")
        
        return "\n".join(lines) + "\n"


# Global metrics registry instance
_metrics_registry: Optional[MetricsRegistry] = None


def get_metrics_registry() -> MetricsRegistry:
    """
    Get or create the global metrics registry.
    
    Returns:
        MetricsRegistry instance
    """
    global _metrics_registry
    if _metrics_registry is None:
        _metrics_registry = MetricsRegistry()
    return _metrics_registry


# Convenience functions for common operations
def record_metric(metric_name: str, value: float, labels: Optional[Dict[str, str]] = None):
    """Record a metric value."""
    get_metrics_registry().record(metric_name, value, labels)


def increment_counter(metric_name: str, value: float = 1, labels: Optional[Dict[str, str]] = None):
    """Increment a counter metric."""
    get_metrics_registry().increment(metric_name, value, labels)


def set_gauge(metric_name: str, value: float, labels: Optional[Dict[str, str]] = None):
    """Set a gauge metric value."""
    get_metrics_registry().set_gauge(metric_name, value, labels)


def observe_histogram(metric_name: str, value: float, labels: Optional[Dict[str, str]] = None):
    """Observe a value for histogram metric."""
    get_metrics_registry().observe(metric_name, value, labels)


@contextmanager
def time_operation(metric_name: str, labels: Optional[Dict[str, str]] = None):
    """Time an operation and record to histogram metric."""
    with get_metrics_registry().timer(metric_name, labels):
        yield