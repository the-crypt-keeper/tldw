"""
Unified metrics and telemetry module for tldw_server.

This module provides comprehensive observability through:
- OpenTelemetry-based metrics and tracing
- Prometheus-compatible metrics export
- Distributed tracing support
- Easy-to-use decorators for automatic instrumentation
"""

# Telemetry initialization and management
from .telemetry import (
    TelemetryConfig,
    TelemetryManager,
    get_telemetry_manager,
    initialize_telemetry,
    shutdown_telemetry,
    OTEL_AVAILABLE
)

# Metrics management
from .metrics_manager import (
    MetricType,
    MetricDefinition,
    MetricValue,
    MetricsRegistry,
    get_metrics_registry,
    record_metric,
    increment_counter,
    set_gauge,
    observe_histogram,
    time_operation
)

# Distributed tracing
from .traces import (
    TraceContext,
    TracingManager,
    get_tracing_manager,
    trace_operation,
    trace_method,
    start_span,
    start_async_span,
    add_span_event,
    set_span_attribute,
    set_span_attributes,
    record_span_exception,
    set_span_status
)

# Decorators for easy instrumentation
from .decorators import (
    track_metrics,
    measure_latency,
    count_calls,
    track_errors,
    monitor_resource,
    track_llm_usage,
    cache_metrics
)

# Legacy compatibility (will be deprecated)
from .metrics_logger import (
    log_counter,
    log_histogram,
    timeit,
    log_resource_usage
)

__all__ = [
    # Telemetry
    "TelemetryConfig",
    "TelemetryManager",
    "get_telemetry_manager",
    "initialize_telemetry",
    "shutdown_telemetry",
    "OTEL_AVAILABLE",
    
    # Metrics
    "MetricType",
    "MetricDefinition",
    "MetricValue",
    "MetricsRegistry",
    "get_metrics_registry",
    "record_metric",
    "increment_counter",
    "set_gauge",
    "observe_histogram",
    "time_operation",
    
    # Tracing
    "TraceContext",
    "TracingManager",
    "get_tracing_manager",
    "trace_operation",
    "trace_method",
    "start_span",
    "start_async_span",
    "add_span_event",
    "set_span_attribute",
    "set_span_attributes",
    "record_span_exception",
    "set_span_status",
    
    # Decorators
    "track_metrics",
    "measure_latency",
    "count_calls",
    "track_errors",
    "monitor_resource",
    "track_llm_usage",
    "cache_metrics",
    
    # Legacy
    "log_counter",
    "log_histogram",
    "timeit",
    "log_resource_usage"
]

# Version information
__version__ = "2.0.0"