# monitoring.py
# Monitoring and metrics integration for Prompt Studio

from typing import Dict, Any, Optional, List
from contextlib import contextmanager
import time
from datetime import datetime
from loguru import logger

from tldw_Server_API.app.core.Metrics.metrics_manager import (
    MetricsRegistry, MetricDefinition, MetricType
)
from tldw_Server_API.app.core.Metrics.decorators import track_metrics, monitor_resource
from tldw_Server_API.app.core.Metrics.traces import trace_operation

########################################################################################################################
# Prompt Studio Metrics

class PromptStudioMetrics:
    """Metrics collection for Prompt Studio operations."""
    
    def __init__(self):
        """Initialize Prompt Studio metrics."""
        self.metrics_manager = MetricsRegistry()
        self._register_metrics()
    
    def _register_metrics(self):
        """Register Prompt Studio specific metrics."""
        
        # Prompt execution metrics
        self.metrics_manager.register_metric(
            MetricDefinition(
                name="prompt_studio.executions.total",
                type=MetricType.COUNTER,
                description="Total number of prompt executions",
                labels=["provider", "model", "status"]
            )
        )
        
        self.metrics_manager.register_metric(
            MetricDefinition(
                name="prompt_studio.executions.duration_seconds",
                type=MetricType.HISTOGRAM,
                description="Prompt execution duration",
                unit="s",
                labels=["provider", "model"],
                buckets=[0.1, 0.5, 1, 2, 5, 10, 30]
            )
        )
        
        self.metrics_manager.register_metric(
            MetricDefinition(
                name="prompt_studio.tokens.used",
                type=MetricType.COUNTER,
                description="Total tokens used",
                labels=["provider", "model", "type"]
            )
        )
        
        self.metrics_manager.register_metric(
            MetricDefinition(
                name="prompt_studio.cost.total",
                type=MetricType.COUNTER,
                description="Total cost in USD",
                unit="usd",
                labels=["provider", "model"]
            )
        )
        
        # Test and evaluation metrics
        self.metrics_manager.register_metric(
            MetricDefinition(
                name="prompt_studio.tests.total",
                type=MetricType.COUNTER,
                description="Total test cases executed",
                labels=["project", "status"]
            )
        )
        
        self.metrics_manager.register_metric(
            MetricDefinition(
                name="prompt_studio.evaluations.score",
                type=MetricType.HISTOGRAM,
                description="Evaluation scores",
                labels=["project", "metric_type"],
                buckets=[0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
            )
        )
        
        self.metrics_manager.register_metric(
            MetricDefinition(
                name="prompt_studio.evaluations.duration_seconds",
                type=MetricType.HISTOGRAM,
                description="Evaluation duration",
                unit="s",
                labels=["project"],
                buckets=[1, 5, 10, 30, 60, 300, 600]
            )
        )
        
        # Optimization metrics
        self.metrics_manager.register_metric(
            MetricDefinition(
                name="prompt_studio.optimizations.total",
                type=MetricType.COUNTER,
                description="Total optimizations run",
                labels=["strategy", "status"]
            )
        )
        
        self.metrics_manager.register_metric(
            MetricDefinition(
                name="prompt_studio.optimizations.improvement",
                type=MetricType.HISTOGRAM,
                description="Optimization improvement percentage",
                labels=["strategy"],
                buckets=[-50, -25, -10, 0, 10, 25, 50, 100, 200]
            )
        )
        
        self.metrics_manager.register_metric(
            MetricDefinition(
                name="prompt_studio.optimizations.iterations",
                type=MetricType.HISTOGRAM,
                description="Number of optimization iterations",
                labels=["strategy"],
                buckets=[1, 5, 10, 20, 50, 100]
            )
        )
        
        # Job queue metrics
        self.metrics_manager.register_metric(
            MetricDefinition(
                name="prompt_studio.jobs.queued",
                type=MetricType.GAUGE,
                description="Number of queued jobs",
                labels=["job_type"]
            )
        )
        
        self.metrics_manager.register_metric(
            MetricDefinition(
                name="prompt_studio.jobs.processing",
                type=MetricType.GAUGE,
                description="Number of processing jobs",
                labels=["job_type"]
            )
        )
        
        self.metrics_manager.register_metric(
            MetricDefinition(
                name="prompt_studio.jobs.completed",
                type=MetricType.COUNTER,
                description="Total completed jobs",
                labels=["job_type", "status"]
            )
        )
        
        self.metrics_manager.register_metric(
            MetricDefinition(
                name="prompt_studio.jobs.duration_seconds",
                type=MetricType.HISTOGRAM,
                description="Job processing duration",
                unit="s",
                labels=["job_type"],
                buckets=[1, 5, 10, 30, 60, 300, 600, 1800]
            )
        )
        
        # WebSocket metrics
        self.metrics_manager.register_metric(
            MetricDefinition(
                name="prompt_studio.websocket.connections",
                type=MetricType.GAUGE,
                description="Active WebSocket connections"
            )
        )
        
        self.metrics_manager.register_metric(
            MetricDefinition(
                name="prompt_studio.websocket.messages",
                type=MetricType.COUNTER,
                description="WebSocket messages sent",
                labels=["event_type"]
            )
        )
        
        # Database metrics
        self.metrics_manager.register_metric(
            MetricDefinition(
                name="prompt_studio.database.operations",
                type=MetricType.COUNTER,
                description="Database operations",
                labels=["operation", "table"]
            )
        )
        
        self.metrics_manager.register_metric(
            MetricDefinition(
                name="prompt_studio.database.latency_ms",
                type=MetricType.HISTOGRAM,
                description="Database operation latency",
                unit="ms",
                labels=["operation"],
                buckets=[1, 5, 10, 25, 50, 100, 250, 500, 1000]
            )
        )
    
    ####################################################################################################################
    # Context Managers for Tracking
    
    @contextmanager
    def track_prompt_execution(self, provider: str, model: str):
        """
        Track prompt execution metrics.
        
        Args:
            provider: LLM provider
            model: Model name
        """
        start_time = time.time()
        labels = {"provider": provider, "model": model}
        
        try:
            yield
            # Success
            self.metrics_manager.increment(
                "prompt_studio.executions.total",
                labels={**labels, "status": "success"}
            )
        except Exception as e:
            # Failure
            self.metrics_manager.increment(
                "prompt_studio.executions.total",
                labels={**labels, "status": "error"}
            )
            raise
        finally:
            # Record duration
            duration = time.time() - start_time
            self.metrics_manager.observe(
                "prompt_studio.executions.duration_seconds",
                duration,
                labels=labels
            )
    
    @contextmanager
    def track_evaluation(self, project_id: str):
        """
        Track evaluation metrics.
        
        Args:
            project_id: Project identifier
        """
        start_time = time.time()
        
        try:
            yield
        finally:
            duration = time.time() - start_time
            self.metrics_manager.observe(
                "prompt_studio.evaluations.duration_seconds",
                duration,
                labels={"project": str(project_id)}
            )
    
    @contextmanager
    def track_optimization(self, strategy: str):
        """
        Track optimization metrics.
        
        Args:
            strategy: Optimization strategy
        """
        start_time = time.time()
        
        try:
            yield
            self.metrics_manager.increment(
                "prompt_studio.optimizations.total",
                labels={"strategy": strategy, "status": "success"}
            )
        except Exception:
            self.metrics_manager.increment(
                "prompt_studio.optimizations.total",
                labels={"strategy": strategy, "status": "failed"}
            )
            raise
    
    @contextmanager
    def track_job_processing(self, job_type: str):
        """
        Track job processing metrics.
        
        Args:
            job_type: Type of job
        """
        start_time = time.time()
        
        # Increment processing gauge
        self.metrics_manager.increment(
            "prompt_studio.jobs.processing",
            labels={"job_type": job_type}
        )
        
        try:
            yield
            # Success
            self.metrics_manager.increment(
                "prompt_studio.jobs.completed",
                labels={"job_type": job_type, "status": "success"}
            )
        except Exception:
            # Failure
            self.metrics_manager.increment(
                "prompt_studio.jobs.completed",
                labels={"job_type": job_type, "status": "failed"}
            )
            raise
        finally:
            # Decrement processing gauge
            self.metrics_manager.decrement(
                "prompt_studio.jobs.processing",
                labels={"job_type": job_type}
            )
            
            # Record duration
            duration = time.time() - start_time
            self.metrics_manager.observe(
                "prompt_studio.jobs.duration_seconds",
                duration,
                labels={"job_type": job_type}
            )
    
    ####################################################################################################################
    # Metric Recording Methods
    
    def record_token_usage(self, provider: str, model: str, 
                          input_tokens: int, output_tokens: int):
        """Record token usage."""
        self.metrics_manager.increment(
            "prompt_studio.tokens.used",
            value=input_tokens,
            labels={"provider": provider, "model": model, "type": "input"}
        )
        
        self.metrics_manager.increment(
            "prompt_studio.tokens.used",
            value=output_tokens,
            labels={"provider": provider, "model": model, "type": "output"}
        )
    
    def record_cost(self, provider: str, model: str, cost: float):
        """Record cost in USD."""
        self.metrics_manager.increment(
            "prompt_studio.cost.total",
            value=cost,
            labels={"provider": provider, "model": model}
        )
    
    def record_test_execution(self, project_id: str, success: bool):
        """Record test case execution."""
        status = "success" if success else "failed"
        self.metrics_manager.increment(
            "prompt_studio.tests.total",
            labels={"project": str(project_id), "status": status}
        )
    
    def record_evaluation_score(self, project_id: str, metric_type: str, score: float):
        """Record evaluation score."""
        self.metrics_manager.observe(
            "prompt_studio.evaluations.score",
            score,
            labels={"project": str(project_id), "metric_type": metric_type}
        )
    
    def record_optimization_improvement(self, strategy: str, improvement: float,
                                       iterations: int):
        """Record optimization improvement."""
        self.metrics_manager.observe(
            "prompt_studio.optimizations.improvement",
            improvement * 100,  # Convert to percentage
            labels={"strategy": strategy}
        )
        
        self.metrics_manager.observe(
            "prompt_studio.optimizations.iterations",
            iterations,
            labels={"strategy": strategy}
        )
    
    def update_job_queue_size(self, job_type: str, queued_count: int):
        """Update job queue size."""
        self.metrics_manager.set_gauge(
            "prompt_studio.jobs.queued",
            queued_count,
            labels={"job_type": job_type}
        )
    
    def update_websocket_connections(self, count: int):
        """Update WebSocket connection count."""
        self.metrics_manager.set_gauge(
            "prompt_studio.websocket.connections",
            count
        )
    
    def record_websocket_message(self, event_type: str):
        """Record WebSocket message sent."""
        self.metrics_manager.increment(
            "prompt_studio.websocket.messages",
            labels={"event_type": event_type}
        )
    
    def record_database_operation(self, operation: str, table: str, latency_ms: float):
        """Record database operation."""
        self.metrics_manager.increment(
            "prompt_studio.database.operations",
            labels={"operation": operation, "table": table}
        )
        
        self.metrics_manager.observe(
            "prompt_studio.database.latency_ms",
            latency_ms,
            labels={"operation": operation}
        )

########################################################################################################################
# Monitoring Decorators

def monitor_prompt_execution(provider: str, model: str):
    """
    Decorator to monitor prompt execution.
    
    Args:
        provider: LLM provider
        model: Model name
    """
    def decorator(func):
        async def async_wrapper(*args, **kwargs):
            metrics = PromptStudioMetrics()
            with metrics.track_prompt_execution(provider, model):
                with trace_operation(f"prompt_execution_{provider}_{model}"):
                    result = await func(*args, **kwargs)
                    
                    # Extract metrics from result if available
                    if isinstance(result, dict):
                        if "tokens_used" in result:
                            metrics.record_token_usage(
                                provider, model,
                                result.get("input_tokens", 0),
                                result.get("output_tokens", result["tokens_used"])
                            )
                        if "cost_estimate" in result:
                            metrics.record_cost(provider, model, result["cost_estimate"])
                    
                    return result
        
        def sync_wrapper(*args, **kwargs):
            metrics = PromptStudioMetrics()
            with metrics.track_prompt_execution(provider, model):
                with trace_operation(f"prompt_execution_{provider}_{model}"):
                    result = func(*args, **kwargs)
                    
                    # Extract metrics from result if available
                    if isinstance(result, dict):
                        if "tokens_used" in result:
                            metrics.record_token_usage(
                                provider, model,
                                result.get("input_tokens", 0),
                                result.get("output_tokens", result["tokens_used"])
                            )
                        if "cost_estimate" in result:
                            metrics.record_cost(provider, model, result["cost_estimate"])
                    
                    return result
        
        # Return appropriate wrapper based on function type
        import asyncio
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        else:
            return sync_wrapper
    
    return decorator

def monitor_optimization(strategy: str):
    """
    Decorator to monitor optimization operations.
    
    Args:
        strategy: Optimization strategy name
    """
    def decorator(func):
        async def async_wrapper(*args, **kwargs):
            metrics = PromptStudioMetrics()
            with metrics.track_optimization(strategy):
                with trace_operation(f"optimization_{strategy}"):
                    result = await func(*args, **kwargs)
                    
                    # Record improvement if available
                    if isinstance(result, dict):
                        if "improvement" in result and "iterations" in result:
                            metrics.record_optimization_improvement(
                                strategy,
                                result["improvement"],
                                result["iterations"]
                            )
                    
                    return result
        
        def sync_wrapper(*args, **kwargs):
            metrics = PromptStudioMetrics()
            with metrics.track_optimization(strategy):
                with trace_operation(f"optimization_{strategy}"):
                    result = func(*args, **kwargs)
                    
                    # Record improvement if available
                    if isinstance(result, dict):
                        if "improvement" in result and "iterations" in result:
                            metrics.record_optimization_improvement(
                                strategy,
                                result["improvement"],
                                result["iterations"]
                            )
                    
                    return result
        
        # Return appropriate wrapper based on function type
        import asyncio
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        else:
            return sync_wrapper
    
    return decorator

########################################################################################################################
# Health Check and Status

class PromptStudioHealthCheck:
    """Health check for Prompt Studio components."""
    
    def __init__(self, db, job_manager=None, websocket_manager=None):
        """
        Initialize health check.
        
        Args:
            db: Database instance
            job_manager: Optional job manager
            websocket_manager: Optional WebSocket manager
        """
        self.db = db
        self.job_manager = job_manager
        self.websocket_manager = websocket_manager
        self.metrics = PromptStudioMetrics()
    
    def check_health(self) -> Dict[str, Any]:
        """
        Check health of Prompt Studio components.
        
        Returns:
            Health status dictionary
        """
        health = {
            "status": "healthy",
            "timestamp": datetime.utcnow().isoformat(),
            "components": {}
        }
        
        # Check database
        try:
            conn = self.db.get_connection()
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM prompt_studio_projects")
            project_count = cursor.fetchone()[0]
            
            health["components"]["database"] = {
                "status": "healthy",
                "projects": project_count
            }
        except Exception as e:
            health["components"]["database"] = {
                "status": "unhealthy",
                "error": str(e)
            }
            health["status"] = "degraded"
        
        # Check job queue
        if self.job_manager:
            try:
                stats = self.job_manager.get_job_stats()
                health["components"]["job_queue"] = {
                    "status": "healthy",
                    "queued": stats.get("queue_depth", 0),
                    "processing": stats.get("processing", 0)
                }
                
                # Update metrics
                for job_type, count in stats.get("by_type", {}).items():
                    self.metrics.update_job_queue_size(job_type, count)
                    
            except Exception as e:
                health["components"]["job_queue"] = {
                    "status": "unhealthy",
                    "error": str(e)
                }
                health["status"] = "degraded"
        
        # Check WebSocket connections
        if self.websocket_manager:
            try:
                connection_count = self.websocket_manager.get_connection_count()
                health["components"]["websocket"] = {
                    "status": "healthy",
                    "connections": connection_count
                }
                
                # Update metrics
                self.metrics.update_websocket_connections(connection_count)
                
            except Exception as e:
                health["components"]["websocket"] = {
                    "status": "unhealthy",
                    "error": str(e)
                }
        
        return health
    
    def get_metrics_summary(self) -> Dict[str, Any]:
        """
        Get summary of current metrics.
        
        Returns:
            Metrics summary
        """
        return self.metrics.metrics_manager.get_summary([
            "prompt_studio.executions.total",
            "prompt_studio.tokens.used",
            "prompt_studio.cost.total",
            "prompt_studio.tests.total",
            "prompt_studio.evaluations.score",
            "prompt_studio.optimizations.total",
            "prompt_studio.jobs.queued",
            "prompt_studio.jobs.processing",
            "prompt_studio.websocket.connections"
        ])

# Global metrics instance
prompt_studio_metrics = PromptStudioMetrics()