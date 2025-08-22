"""
OpenTelemetry configuration and initialization for unified metrics.

This module provides centralized telemetry configuration supporting:
- Metrics collection and export
- Distributed tracing
- Log correlation
- Multiple export backends (OTLP, Prometheus, Jaeger)
"""

import os
import socket
from typing import Dict, Any, Optional, List
from contextlib import contextmanager
import logging

from loguru import logger

# Try to import OpenTelemetry components
try:
    from opentelemetry import trace, metrics, baggage
    from opentelemetry.trace import Status, StatusCode, Tracer
    from opentelemetry.metrics import Meter
    
    # SDK components
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.metrics import MeterProvider
    from opentelemetry.sdk.resources import Resource, SERVICE_NAME, SERVICE_VERSION
    from opentelemetry.sdk.trace.export import BatchSpanProcessor, ConsoleSpanExporter
    from opentelemetry.sdk.metrics.export import (
        PeriodicExportingMetricReader,
        ConsoleMetricExporter
    )
    
    # Exporters
    from opentelemetry.exporter.prometheus import PrometheusMetricReader
    from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
    from opentelemetry.exporter.otlp.proto.grpc.metric_exporter import OTLPMetricExporter
    
    # Instrumentation
    from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
    from opentelemetry.instrumentation.httpx import HTTPXClientInstrumentor
    from opentelemetry.instrumentation.sqlalchemy import SQLAlchemyInstrumentor
    from opentelemetry.instrumentation.psycopg2 import Psycopg2Instrumentor
    from opentelemetry.instrumentation.aiohttp_client import AioHttpClientInstrumentor
    
    # Context propagation
    from opentelemetry.propagate import set_global_textmap
    from opentelemetry.trace.propagation.tracecontext import TraceContextTextMapPropagator
    
    OTEL_AVAILABLE = True
except ImportError as e:
    logger.warning(f"OpenTelemetry not fully available: {e}")
    logger.info("Install with: pip install opentelemetry-distro opentelemetry-exporter-otlp opentelemetry-instrumentation-fastapi")
    OTEL_AVAILABLE = False
    
    # Provide dummy classes for fallback
    class DummyTracer:
        def start_span(self, name, **kwargs):
            return DummySpan()
    
    class DummySpan:
        def __enter__(self):
            return self
        def __exit__(self, *args):
            pass
        def set_attribute(self, key, value):
            pass
        def set_status(self, status):
            pass
        def add_event(self, name, attributes=None):
            pass
        def record_exception(self, exception):
            pass
    
    class DummyMeter:
        def create_counter(self, name, **kwargs):
            return DummyInstrument()
        def create_histogram(self, name, **kwargs):
            return DummyInstrument()
        def create_gauge(self, name, **kwargs):
            return DummyInstrument()
        def create_up_down_counter(self, name, **kwargs):
            return DummyInstrument()
    
    class DummyInstrument:
        def add(self, amount, attributes=None):
            pass
        def record(self, amount, attributes=None):
            pass
        def set(self, amount, attributes=None):
            pass


class TelemetryConfig:
    """Configuration for OpenTelemetry setup."""
    
    def __init__(self):
        """Initialize telemetry configuration from environment variables."""
        # Service identification
        self.service_name = os.getenv("OTEL_SERVICE_NAME", "tldw_server")
        self.service_version = os.getenv("OTEL_SERVICE_VERSION", "1.0.0")
        self.service_namespace = os.getenv("OTEL_SERVICE_NAMESPACE", "production")
        self.deployment_environment = os.getenv("DEPLOYMENT_ENV", "development")
        
        # OTLP Configuration
        self.otlp_endpoint = os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT", "")
        self.otlp_protocol = os.getenv("OTEL_EXPORTER_OTLP_PROTOCOL", "grpc")
        self.otlp_headers = os.getenv("OTEL_EXPORTER_OTLP_HEADERS", "")
        self.otlp_insecure = os.getenv("OTEL_EXPORTER_OTLP_INSECURE", "true").lower() == "true"
        
        # Exporter selection
        self.metrics_exporters = os.getenv("OTEL_METRICS_EXPORTER", "prometheus,console").split(",")
        self.traces_exporters = os.getenv("OTEL_TRACES_EXPORTER", "console").split(",")
        
        # Prometheus Configuration
        self.prometheus_port = int(os.getenv("PROMETHEUS_PORT", "9090"))
        self.prometheus_host = os.getenv("PROMETHEUS_HOST", "0.0.0.0")
        
        # Feature flags
        self.enable_metrics = os.getenv("ENABLE_METRICS", "true").lower() == "true"
        self.enable_tracing = os.getenv("ENABLE_TRACING", "true").lower() == "true"
        self.enable_logging = os.getenv("ENABLE_OTEL_LOGGING", "false").lower() == "true"
        self.enable_profiling = os.getenv("ENABLE_PROFILING", "false").lower() == "true"
        
        # Performance settings
        self.metrics_export_interval = int(os.getenv("METRICS_EXPORT_INTERVAL_MS", "60000"))
        self.traces_export_batch_size = int(os.getenv("TRACES_EXPORT_BATCH_SIZE", "512"))
        self.traces_export_timeout = int(os.getenv("TRACES_EXPORT_TIMEOUT_MS", "30000"))
        self.sample_rate = float(os.getenv("METRICS_SAMPLE_RATE", "1.0"))
        
        # Additional metadata
        self.hostname = socket.gethostname()
        self.pod_name = os.getenv("POD_NAME", self.hostname)
        self.pod_namespace = os.getenv("POD_NAMESPACE", "default")
        
    def get_resource_attributes(self) -> Dict[str, Any]:
        """Get resource attributes for telemetry."""
        return {
            SERVICE_NAME: self.service_name,
            SERVICE_VERSION: self.service_version,
            "service.namespace": self.service_namespace,
            "deployment.environment": self.deployment_environment,
            "host.name": self.hostname,
            "k8s.pod.name": self.pod_name,
            "k8s.namespace.name": self.pod_namespace,
        }


class TelemetryManager:
    """Manages OpenTelemetry initialization and provides access to telemetry components."""
    
    def __init__(self, config: Optional[TelemetryConfig] = None):
        """
        Initialize the telemetry manager.
        
        Args:
            config: TelemetryConfig instance or None to use defaults
        """
        self.config = config or TelemetryConfig()
        self.tracer_provider: Optional[TracerProvider] = None
        self.meter_provider: Optional[MeterProvider] = None
        self.tracer: Optional[Tracer] = None
        self.meter: Optional[Meter] = None
        self.initialized = False
        
        if not OTEL_AVAILABLE:
            logger.warning("OpenTelemetry not available, using fallback implementations")
            self.tracer = DummyTracer()
            self.meter = DummyMeter()
            return
        
        self._initialize()
    
    def _initialize(self):
        """Initialize OpenTelemetry providers and exporters."""
        if self.initialized:
            return
        
        try:
            # Create resource
            resource = Resource.create(self.config.get_resource_attributes())
            
            # Initialize tracing
            if self.config.enable_tracing:
                self._initialize_tracing(resource)
            
            # Initialize metrics
            if self.config.enable_metrics:
                self._initialize_metrics(resource)
            
            # Set up context propagation
            set_global_textmap(TraceContextTextMapPropagator())
            
            # Auto-instrumentation
            self._setup_auto_instrumentation()
            
            self.initialized = True
            logger.info(f"Telemetry initialized for service: {self.config.service_name}")
            
        except Exception as e:
            logger.error(f"Failed to initialize telemetry: {e}")
            # Fall back to dummy implementations
            self.tracer = DummyTracer()
            self.meter = DummyMeter()
    
    def _initialize_tracing(self, resource: Resource):
        """Initialize tracing with configured exporters."""
        self.tracer_provider = TracerProvider(resource=resource)
        
        # Add exporters based on configuration
        for exporter_name in self.config.traces_exporters:
            exporter = self._create_trace_exporter(exporter_name)
            if exporter:
                processor = BatchSpanProcessor(
                    exporter,
                    max_queue_size=2048,
                    max_export_batch_size=self.config.traces_export_batch_size,
                    max_export_timeout_millis=self.config.traces_export_timeout
                )
                self.tracer_provider.add_span_processor(processor)
        
        # Set as global tracer provider
        trace.set_tracer_provider(self.tracer_provider)
        self.tracer = trace.get_tracer(
            self.config.service_name,
            self.config.service_version
        )
    
    def _initialize_metrics(self, resource: Resource):
        """Initialize metrics with configured exporters."""
        readers = []
        
        # Add metric readers based on configuration
        for exporter_name in self.config.metrics_exporters:
            reader = self._create_metric_reader(exporter_name)
            if reader:
                readers.append(reader)
        
        if readers:
            self.meter_provider = MeterProvider(
                resource=resource,
                metric_readers=readers
            )
            
            # Set as global meter provider
            metrics.set_meter_provider(self.meter_provider)
            self.meter = metrics.get_meter(
                self.config.service_name,
                self.config.service_version
            )
    
    def _create_trace_exporter(self, exporter_name: str):
        """Create a trace exporter based on name."""
        exporter_name = exporter_name.strip().lower()
        
        if exporter_name == "console":
            return ConsoleSpanExporter()
        
        elif exporter_name == "otlp" and self.config.otlp_endpoint:
            headers = {}
            if self.config.otlp_headers:
                for header in self.config.otlp_headers.split(","):
                    key, value = header.split("=")
                    headers[key] = value
            
            return OTLPSpanExporter(
                endpoint=self.config.otlp_endpoint,
                insecure=self.config.otlp_insecure,
                headers=headers
            )
        
        elif exporter_name == "jaeger":
            # Jaeger support via OTLP
            jaeger_endpoint = os.getenv("JAEGER_ENDPOINT", "http://localhost:4317")
            return OTLPSpanExporter(
                endpoint=jaeger_endpoint,
                insecure=True
            )
        
        else:
            logger.warning(f"Unknown trace exporter: {exporter_name}")
            return None
    
    def _create_metric_reader(self, exporter_name: str):
        """Create a metric reader based on name."""
        exporter_name = exporter_name.strip().lower()
        
        if exporter_name == "console":
            exporter = ConsoleMetricExporter()
            return PeriodicExportingMetricReader(
                exporter,
                export_interval_millis=self.config.metrics_export_interval
            )
        
        elif exporter_name == "prometheus":
            # Prometheus pull-based metrics
            return PrometheusMetricReader(
                host=self.config.prometheus_host,
                port=self.config.prometheus_port
            )
        
        elif exporter_name == "otlp" and self.config.otlp_endpoint:
            headers = {}
            if self.config.otlp_headers:
                for header in self.config.otlp_headers.split(","):
                    key, value = header.split("=")
                    headers[key] = value
            
            exporter = OTLPMetricExporter(
                endpoint=self.config.otlp_endpoint,
                insecure=self.config.otlp_insecure,
                headers=headers
            )
            return PeriodicExportingMetricReader(
                exporter,
                export_interval_millis=self.config.metrics_export_interval
            )
        
        else:
            logger.warning(f"Unknown metric exporter: {exporter_name}")
            return None
    
    def _setup_auto_instrumentation(self):
        """Set up automatic instrumentation for common libraries."""
        if not OTEL_AVAILABLE:
            return
        
        # FastAPI instrumentation
        try:
            FastAPIInstrumentor.instrument(
                tracer_provider=self.tracer_provider,
                meter_provider=self.meter_provider
            )
        except Exception as e:
            logger.debug(f"Could not instrument FastAPI: {e}")
        
        # HTTP client instrumentation
        try:
            HTTPXClientInstrumentor().instrument(
                tracer_provider=self.tracer_provider
            )
        except Exception as e:
            logger.debug(f"Could not instrument HTTPX: {e}")
        
        try:
            AioHttpClientInstrumentor().instrument(
                tracer_provider=self.tracer_provider
            )
        except Exception as e:
            logger.debug(f"Could not instrument aiohttp: {e}")
        
        # Database instrumentation
        try:
            SQLAlchemyInstrumentor().instrument(
                tracer_provider=self.tracer_provider
            )
        except Exception as e:
            logger.debug(f"Could not instrument SQLAlchemy: {e}")
        
        try:
            Psycopg2Instrumentor().instrument(
                tracer_provider=self.tracer_provider
            )
        except Exception as e:
            logger.debug(f"Could not instrument psycopg2: {e}")
    
    def get_tracer(self, name: Optional[str] = None) -> Tracer:
        """
        Get a tracer instance.
        
        Args:
            name: Optional name for the tracer
            
        Returns:
            Tracer instance
        """
        if not self.tracer:
            return DummyTracer()
        
        if name:
            return trace.get_tracer(name, self.config.service_version)
        return self.tracer
    
    def get_meter(self, name: Optional[str] = None) -> Meter:
        """
        Get a meter instance.
        
        Args:
            name: Optional name for the meter
            
        Returns:
            Meter instance
        """
        if not self.meter:
            return DummyMeter()
        
        if name:
            return metrics.get_meter(name, self.config.service_version)
        return self.meter
    
    @contextmanager
    def trace_context(self, operation_name: str, attributes: Optional[Dict[str, Any]] = None):
        """
        Context manager for creating a traced operation.
        
        Args:
            operation_name: Name of the operation
            attributes: Optional attributes to add to the span
            
        Yields:
            The span object
        """
        tracer = self.get_tracer()
        with tracer.start_as_current_span(operation_name) as span:
            if attributes:
                for key, value in attributes.items():
                    span.set_attribute(key, value)
            
            try:
                yield span
            except Exception as e:
                span.record_exception(e)
                span.set_status(Status(StatusCode.ERROR, str(e)))
                raise
    
    def shutdown(self):
        """Shutdown telemetry providers and flush data."""
        if not OTEL_AVAILABLE or not self.initialized:
            return
        
        try:
            if self.tracer_provider:
                self.tracer_provider.shutdown()
            
            if self.meter_provider:
                self.meter_provider.shutdown()
            
            logger.info("Telemetry providers shut down successfully")
        except Exception as e:
            logger.error(f"Error shutting down telemetry: {e}")


# Global telemetry manager instance
_telemetry_manager: Optional[TelemetryManager] = None


def get_telemetry_manager() -> TelemetryManager:
    """
    Get or create the global telemetry manager.
    
    Returns:
        TelemetryManager instance
    """
    global _telemetry_manager
    if _telemetry_manager is None:
        _telemetry_manager = TelemetryManager()
    return _telemetry_manager


def initialize_telemetry(config: Optional[TelemetryConfig] = None) -> TelemetryManager:
    """
    Initialize telemetry with optional configuration.
    
    Args:
        config: Optional TelemetryConfig instance
        
    Returns:
        TelemetryManager instance
    """
    global _telemetry_manager
    _telemetry_manager = TelemetryManager(config)
    return _telemetry_manager


def shutdown_telemetry():
    """Shutdown the global telemetry manager."""
    if _telemetry_manager:
        _telemetry_manager.shutdown()