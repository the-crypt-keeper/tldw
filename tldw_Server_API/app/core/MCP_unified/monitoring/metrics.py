"""
Metrics collection and monitoring for unified MCP

Provides Prometheus-compatible metrics and health monitoring.
"""

import time
import asyncio
from typing import Dict, Any, Optional, List
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from collections import defaultdict, deque
from enum import Enum
from loguru import logger

try:
    from prometheus_client import (
        Counter,
        Histogram,
        Gauge,
        Summary,
        Info,
        generate_latest,
        CONTENT_TYPE_LATEST,
        CollectorRegistry
    )
    PROMETHEUS_AVAILABLE = True
except ImportError:
    PROMETHEUS_AVAILABLE = False
    logger.warning("Prometheus client not installed. Metrics will be limited.")


class MetricType(str, Enum):
    """Types of metrics"""
    COUNTER = "counter"
    GAUGE = "gauge"
    HISTOGRAM = "histogram"
    SUMMARY = "summary"


@dataclass
class MetricData:
    """Container for metric data"""
    name: str
    type: MetricType
    value: float
    labels: Dict[str, str] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.utcnow)
    description: str = ""


class MetricsCollector:
    """
    Centralized metrics collection for MCP.
    
    Supports both Prometheus and internal metrics collection.
    """
    
    def __init__(self, enable_prometheus: bool = True):
        self.enable_prometheus = enable_prometheus and PROMETHEUS_AVAILABLE
        
        # Internal metrics storage
        self._metrics: Dict[str, deque] = defaultdict(lambda: deque(maxlen=1000))
        self._metric_definitions: Dict[str, MetricData] = {}
        
        # Prometheus metrics (if available)
        if self.enable_prometheus:
            self.registry = CollectorRegistry()
            self._init_prometheus_metrics()
        else:
            self.registry = None
        
        # Start collection tasks
        self._collection_task = None
        self._aggregation_interval = 60  # seconds
        
        logger.info(f"Metrics collector initialized (Prometheus: {self.enable_prometheus})")
    
    def _init_prometheus_metrics(self):
        """Initialize Prometheus metrics"""
        # Request metrics
        self.request_counter = Counter(
            'mcp_requests_total',
            'Total number of MCP requests',
            ['method', 'status'],
            registry=self.registry
        )
        
        self.request_duration = Histogram(
            'mcp_request_duration_seconds',
            'Request duration in seconds',
            ['method'],
            buckets=(0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0),
            registry=self.registry
        )
        
        # Module metrics
        self.module_health = Gauge(
            'mcp_module_health',
            'Module health status (1=healthy, 0=unhealthy)',
            ['module'],
            registry=self.registry
        )
        
        self.module_operations = Counter(
            'mcp_module_operations_total',
            'Total module operations',
            ['module', 'operation', 'status'],
            registry=self.registry
        )
        
        # Connection metrics
        self.active_connections = Gauge(
            'mcp_active_connections',
            'Number of active connections',
            ['type'],
            registry=self.registry
        )
        
        self.connection_errors = Counter(
            'mcp_connection_errors_total',
            'Total connection errors',
            ['type', 'error'],
            registry=self.registry
        )
        
        # Rate limiting metrics
        self.rate_limit_hits = Counter(
            'mcp_rate_limit_hits_total',
            'Total rate limit hits',
            ['key_type'],
            registry=self.registry
        )
        
        # Cache metrics
        self.cache_hits = Counter(
            'mcp_cache_hits_total',
            'Cache hit count',
            ['cache_name'],
            registry=self.registry
        )
        
        self.cache_misses = Counter(
            'mcp_cache_misses_total',
            'Cache miss count',
            ['cache_name'],
            registry=self.registry
        )
        
        # System metrics
        self.memory_usage = Gauge(
            'mcp_memory_usage_bytes',
            'Memory usage in bytes',
            registry=self.registry
        )
        
        self.cpu_usage = Gauge(
            'mcp_cpu_usage_percent',
            'CPU usage percentage',
            registry=self.registry
        )
        
        # Server info
        self.server_info = Info(
            'mcp_server',
            'MCP server information',
            registry=self.registry
        )
        
        self.server_info.info({
            'version': '3.0.0',
            'protocol_version': '2024-11-05'
        })
    
    def record_request(
        self,
        method: str,
        duration: float,
        status: str = "success",
        labels: Optional[Dict[str, str]] = None
    ):
        """Record an API request"""
        # Internal metrics
        metric = MetricData(
            name=f"request_{method}",
            type=MetricType.HISTOGRAM,
            value=duration,
            labels=labels or {},
            description=f"Request to {method}"
        )
        self._metrics[f"request_{method}"].append(metric)
        
        # Prometheus metrics
        if self.enable_prometheus:
            self.request_counter.labels(method=method, status=status).inc()
            self.request_duration.labels(method=method).observe(duration)
    
    def record_module_operation(
        self,
        module: str,
        operation: str,
        duration: float,
        success: bool = True
    ):
        """Record a module operation"""
        status = "success" if success else "failure"
        
        # Internal metrics
        metric = MetricData(
            name=f"module_{module}_{operation}",
            type=MetricType.HISTOGRAM,
            value=duration,
            labels={"module": module, "operation": operation, "status": status}
        )
        self._metrics[f"module_{module}_{operation}"].append(metric)
        
        # Prometheus metrics
        if self.enable_prometheus:
            self.module_operations.labels(
                module=module,
                operation=operation,
                status=status
            ).inc()
    
    def set_module_health(self, module: str, is_healthy: bool):
        """Set module health status"""
        health_value = 1.0 if is_healthy else 0.0
        
        # Internal metrics
        metric = MetricData(
            name=f"module_health_{module}",
            type=MetricType.GAUGE,
            value=health_value,
            labels={"module": module}
        )
        self._metrics[f"module_health_{module}"].append(metric)
        
        # Prometheus metrics
        if self.enable_prometheus:
            self.module_health.labels(module=module).set(health_value)
    
    def update_connection_count(self, connection_type: str, count: int):
        """Update active connection count"""
        # Internal metrics
        metric = MetricData(
            name=f"connections_{connection_type}",
            type=MetricType.GAUGE,
            value=float(count),
            labels={"type": connection_type}
        )
        self._metrics[f"connections_{connection_type}"].append(metric)
        
        # Prometheus metrics
        if self.enable_prometheus:
            self.active_connections.labels(type=connection_type).set(count)
    
    def record_connection_error(self, connection_type: str, error: str):
        """Record a connection error"""
        # Internal metrics
        metric = MetricData(
            name=f"connection_error_{connection_type}",
            type=MetricType.COUNTER,
            value=1,
            labels={"type": connection_type, "error": error}
        )
        self._metrics[f"connection_error_{connection_type}"].append(metric)
        
        # Prometheus metrics
        if self.enable_prometheus:
            self.connection_errors.labels(
                type=connection_type,
                error=error
            ).inc()
    
    def record_rate_limit_hit(self, key_type: str = "user"):
        """Record a rate limit hit"""
        # Internal metrics
        metric = MetricData(
            name=f"rate_limit_{key_type}",
            type=MetricType.COUNTER,
            value=1,
            labels={"key_type": key_type}
        )
        self._metrics[f"rate_limit_{key_type}"].append(metric)
        
        # Prometheus metrics
        if self.enable_prometheus:
            self.rate_limit_hits.labels(key_type=key_type).inc()
    
    def record_cache_access(self, cache_name: str, hit: bool):
        """Record cache access"""
        metric_name = f"cache_{'hit' if hit else 'miss'}_{cache_name}"
        
        # Internal metrics
        metric = MetricData(
            name=metric_name,
            type=MetricType.COUNTER,
            value=1,
            labels={"cache": cache_name, "result": "hit" if hit else "miss"}
        )
        self._metrics[metric_name].append(metric)
        
        # Prometheus metrics
        if self.enable_prometheus:
            if hit:
                self.cache_hits.labels(cache_name=cache_name).inc()
            else:
                self.cache_misses.labels(cache_name=cache_name).inc()
    
    def update_system_metrics(self, memory_bytes: int, cpu_percent: float):
        """Update system resource metrics"""
        # Internal metrics
        self._metrics["memory_usage"].append(
            MetricData(
                name="memory_usage",
                type=MetricType.GAUGE,
                value=float(memory_bytes)
            )
        )
        self._metrics["cpu_usage"].append(
            MetricData(
                name="cpu_usage",
                type=MetricType.GAUGE,
                value=cpu_percent
            )
        )
        
        # Prometheus metrics
        if self.enable_prometheus:
            self.memory_usage.set(memory_bytes)
            self.cpu_usage.set(cpu_percent)
    
    def get_prometheus_metrics(self) -> bytes:
        """Get metrics in Prometheus format"""
        if not self.enable_prometheus:
            return b"# Prometheus metrics not enabled\n"
        
        return generate_latest(self.registry)
    
    def get_internal_metrics(self, period_seconds: int = 300) -> Dict[str, Any]:
        """
        Get internal metrics for the specified period.
        
        Args:
            period_seconds: Time period to aggregate metrics
        
        Returns:
            Aggregated metrics dictionary
        """
        cutoff_time = datetime.utcnow() - timedelta(seconds=period_seconds)
        aggregated = {}
        
        for metric_name, metric_deque in self._metrics.items():
            # Filter metrics by time
            recent_metrics = [
                m for m in metric_deque
                if m.timestamp >= cutoff_time
            ]
            
            if not recent_metrics:
                continue
            
            # Aggregate based on metric type
            first_metric = recent_metrics[0]
            
            if first_metric.type == MetricType.COUNTER:
                # Sum for counters
                aggregated[metric_name] = {
                    "type": "counter",
                    "value": sum(m.value for m in recent_metrics),
                    "count": len(recent_metrics)
                }
            
            elif first_metric.type == MetricType.GAUGE:
                # Latest value for gauges
                aggregated[metric_name] = {
                    "type": "gauge",
                    "value": recent_metrics[-1].value,
                    "timestamp": recent_metrics[-1].timestamp.isoformat()
                }
            
            elif first_metric.type in [MetricType.HISTOGRAM, MetricType.SUMMARY]:
                # Statistics for histograms/summaries
                values = [m.value for m in recent_metrics]
                aggregated[metric_name] = {
                    "type": first_metric.type.value,
                    "count": len(values),
                    "sum": sum(values),
                    "avg": sum(values) / len(values) if values else 0,
                    "min": min(values) if values else 0,
                    "max": max(values) if values else 0,
                    "p50": self._percentile(values, 50),
                    "p95": self._percentile(values, 95),
                    "p99": self._percentile(values, 99)
                }
        
        return aggregated
    
    def _percentile(self, values: List[float], percentile: int) -> float:
        """Calculate percentile of values"""
        if not values:
            return 0
        
        sorted_values = sorted(values)
        index = int(len(sorted_values) * percentile / 100)
        
        if index >= len(sorted_values):
            return sorted_values[-1]
        
        return sorted_values[index]
    
    async def start_collection(self):
        """Start background metrics collection"""
        if self._collection_task is None:
            self._collection_task = asyncio.create_task(self._collection_loop())
            logger.info("Metrics collection started")
    
    async def stop_collection(self):
        """Stop metrics collection"""
        if self._collection_task:
            self._collection_task.cancel()
            try:
                await self._collection_task
            except asyncio.CancelledError:
                pass
            self._collection_task = None
            logger.info("Metrics collection stopped")
    
    async def _collection_loop(self):
        """Background task for system metrics collection"""
        while True:
            try:
                # Collect system metrics
                import psutil
                
                # Memory usage
                memory = psutil.virtual_memory()
                self.update_system_metrics(
                    memory_bytes=memory.used,
                    cpu_percent=psutil.cpu_percent(interval=1)
                )
                
                # Clean old metrics
                self._clean_old_metrics()
                
                await asyncio.sleep(self._aggregation_interval)
                
            except Exception as e:
                logger.error(f"Error in metrics collection: {e}")
                await asyncio.sleep(10)
    
    def _clean_old_metrics(self, max_age_hours: int = 24):
        """Clean metrics older than max_age"""
        cutoff_time = datetime.utcnow() - timedelta(hours=max_age_hours)
        
        for metric_deque in self._metrics.values():
            # deque automatically maintains max size, but we can clean by time
            while metric_deque and metric_deque[0].timestamp < cutoff_time:
                metric_deque.popleft()
    
    def get_summary(self) -> Dict[str, Any]:
        """Get a summary of current metrics"""
        return {
            "total_metrics": sum(len(d) for d in self._metrics.values()),
            "metric_types": list(set(
                m.type.value
                for d in self._metrics.values()
                for m in d
            )),
            "prometheus_enabled": self.enable_prometheus,
            "collection_interval": self._aggregation_interval,
            "recent_metrics": self.get_internal_metrics(60)  # Last minute
        }


# Global metrics collector
_metrics_collector: Optional[MetricsCollector] = None


def get_metrics_collector() -> MetricsCollector:
    """Get or create metrics collector singleton"""
    global _metrics_collector
    if _metrics_collector is None:
        _metrics_collector = MetricsCollector()
    return _metrics_collector


# Convenience decorators for metric collection

def track_request_time(method: str):
    """Decorator to track request execution time"""
    def decorator(func):
        async def wrapper(*args, **kwargs):
            start_time = time.time()
            collector = get_metrics_collector()
            
            try:
                result = await func(*args, **kwargs)
                duration = time.time() - start_time
                collector.record_request(method, duration, "success")
                return result
            except Exception as e:
                duration = time.time() - start_time
                collector.record_request(method, duration, "failure")
                raise
        
        return wrapper
    return decorator


def track_module_operation(module: str, operation: str):
    """Decorator to track module operation metrics"""
    def decorator(func):
        async def wrapper(*args, **kwargs):
            start_time = time.time()
            collector = get_metrics_collector()
            
            try:
                result = await func(*args, **kwargs)
                duration = time.time() - start_time
                collector.record_module_operation(module, operation, duration, True)
                return result
            except Exception as e:
                duration = time.time() - start_time
                collector.record_module_operation(module, operation, duration, False)
                raise
        
        return wrapper
    return decorator