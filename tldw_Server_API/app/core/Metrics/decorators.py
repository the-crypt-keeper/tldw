"""
Decorators for easy metric collection and tracing.

Provides a set of decorators to automatically track metrics,
traces, and performance for functions and methods.
"""

import asyncio
import functools
import time
import traceback
from typing import Any, Callable, Dict, Optional, TypeVar, Union, List
from dataclasses import dataclass
import inspect
import json

from loguru import logger

from .metrics_manager import (
    get_metrics_registry,
    increment_counter,
    observe_histogram,
    set_gauge
)
from .traces import get_tracing_manager

# Import OpenTelemetry types conditionally
try:
    from opentelemetry.trace import SpanKind, StatusCode
except ImportError:
    SpanKind = None
    StatusCode = None

# Type variable for decorators
F = TypeVar('F', bound=Callable[..., Any])


@dataclass
class MetricConfig:
    """Configuration for metric decorators."""
    track_duration: bool = True
    track_calls: bool = True
    track_errors: bool = True
    track_success_rate: bool = False
    duration_metric: Optional[str] = None
    call_metric: Optional[str] = None
    error_metric: Optional[str] = None
    success_metric: Optional[str] = None
    labels: Optional[Dict[str, str]] = None
    label_extractor: Optional[Callable] = None
    include_args: bool = False
    include_result: bool = False


def track_metrics(
    name: Optional[str] = None,
    labels: Optional[Dict[str, str]] = None,
    track_duration: bool = True,
    track_calls: bool = True,
    track_errors: bool = True,
    duration_metric: Optional[str] = None,
    call_metric: Optional[str] = None,
    error_metric: Optional[str] = None,
    label_extractor: Optional[Callable] = None
) -> Callable[[F], F]:
    """
    Decorator to automatically track metrics for a function.
    
    Args:
        name: Base name for metrics (defaults to function name)
        labels: Static labels to add to all metrics
        track_duration: Whether to track execution duration
        track_calls: Whether to track call count
        track_errors: Whether to track errors
        duration_metric: Custom name for duration metric
        call_metric: Custom name for call counter metric
        error_metric: Custom name for error counter metric
        label_extractor: Function to extract labels from args/kwargs
        
    Returns:
        Decorated function
    """
    def decorator(func: F) -> F:
        # Determine metric names
        base_name = name or f"{func.__module__}.{func.__name__}"
        duration_metric_name = duration_metric or f"{base_name}_duration_seconds"
        call_metric_name = call_metric or f"{base_name}_calls_total"
        error_metric_name = error_metric or f"{base_name}_errors_total"
        
        if asyncio.iscoroutinefunction(func):
            @functools.wraps(func)
            async def async_wrapper(*args, **kwargs):
                # Extract dynamic labels
                metric_labels = dict(labels or {})
                if label_extractor:
                    try:
                        extracted = label_extractor(*args, **kwargs)
                        if isinstance(extracted, dict):
                            metric_labels.update(extracted)
                    except Exception as e:
                        logger.debug(f"Label extraction failed: {e}")
                
                # Track call count
                if track_calls:
                    increment_counter(call_metric_name, labels=metric_labels)
                
                start_time = time.time()
                try:
                    result = await func(*args, **kwargs)
                    
                    # Track duration
                    if track_duration:
                        duration = time.time() - start_time
                        observe_histogram(duration_metric_name, duration, labels=metric_labels)
                    
                    return result
                    
                except Exception as e:
                    # Track errors
                    if track_errors:
                        error_labels = dict(metric_labels)
                        error_labels["error_type"] = type(e).__name__
                        increment_counter(error_metric_name, labels=error_labels)
                    
                    # Still track duration for failed calls
                    if track_duration:
                        duration = time.time() - start_time
                        failed_labels = dict(metric_labels)
                        failed_labels["status"] = "error"
                        observe_histogram(duration_metric_name, duration, labels=failed_labels)
                    
                    raise
            
            return async_wrapper
        
        else:
            @functools.wraps(func)
            def sync_wrapper(*args, **kwargs):
                # Extract dynamic labels
                metric_labels = dict(labels or {})
                if label_extractor:
                    try:
                        extracted = label_extractor(*args, **kwargs)
                        if isinstance(extracted, dict):
                            metric_labels.update(extracted)
                    except Exception as e:
                        logger.debug(f"Label extraction failed: {e}")
                
                # Track call count
                if track_calls:
                    increment_counter(call_metric_name, labels=metric_labels)
                
                start_time = time.time()
                try:
                    result = func(*args, **kwargs)
                    
                    # Track duration
                    if track_duration:
                        duration = time.time() - start_time
                        observe_histogram(duration_metric_name, duration, labels=metric_labels)
                    
                    return result
                    
                except Exception as e:
                    # Track errors
                    if track_errors:
                        error_labels = dict(metric_labels)
                        error_labels["error_type"] = type(e).__name__
                        increment_counter(error_metric_name, labels=error_labels)
                    
                    # Still track duration for failed calls
                    if track_duration:
                        duration = time.time() - start_time
                        failed_labels = dict(metric_labels)
                        failed_labels["status"] = "error"
                        observe_histogram(duration_metric_name, duration, labels=failed_labels)
                    
                    raise
            
            return sync_wrapper
    
    return decorator


def measure_latency(
    metric_name: Optional[str] = None,
    labels: Optional[Dict[str, str]] = None,
    buckets: Optional[List[float]] = None
) -> Callable[[F], F]:
    """
    Decorator to measure function latency.
    
    Args:
        metric_name: Name of the histogram metric
        labels: Static labels
        buckets: Custom histogram buckets
        
    Returns:
        Decorated function
    """
    def decorator(func: F) -> F:
        histogram_name = metric_name or f"{func.__module__}.{func.__name__}_latency_seconds"
        
        if asyncio.iscoroutinefunction(func):
            @functools.wraps(func)
            async def async_wrapper(*args, **kwargs):
                start_time = time.time()
                try:
                    return await func(*args, **kwargs)
                finally:
                    latency = time.time() - start_time
                    observe_histogram(histogram_name, latency, labels=labels)
            
            return async_wrapper
        else:
            @functools.wraps(func)
            def sync_wrapper(*args, **kwargs):
                start_time = time.time()
                try:
                    return func(*args, **kwargs)
                finally:
                    latency = time.time() - start_time
                    observe_histogram(histogram_name, latency, labels=labels)
            
            return sync_wrapper
    
    return decorator


def count_calls(
    metric_name: Optional[str] = None,
    labels: Optional[Dict[str, str]] = None,
    label_extractor: Optional[Callable] = None
) -> Callable[[F], F]:
    """
    Decorator to count function calls.
    
    Args:
        metric_name: Name of the counter metric
        labels: Static labels
        label_extractor: Function to extract labels from args
        
    Returns:
        Decorated function
    """
    def decorator(func: F) -> F:
        counter_name = metric_name or f"{func.__module__}.{func.__name__}_calls_total"
        
        if asyncio.iscoroutinefunction(func):
            @functools.wraps(func)
            async def async_wrapper(*args, **kwargs):
                metric_labels = dict(labels or {})
                if label_extractor:
                    try:
                        extracted = label_extractor(*args, **kwargs)
                        if isinstance(extracted, dict):
                            metric_labels.update(extracted)
                    except:
                        pass
                
                increment_counter(counter_name, labels=metric_labels)
                return await func(*args, **kwargs)
            
            return async_wrapper
        else:
            @functools.wraps(func)
            def sync_wrapper(*args, **kwargs):
                metric_labels = dict(labels or {})
                if label_extractor:
                    try:
                        extracted = label_extractor(*args, **kwargs)
                        if isinstance(extracted, dict):
                            metric_labels.update(extracted)
                    except:
                        pass
                
                increment_counter(counter_name, labels=metric_labels)
                return func(*args, **kwargs)
            
            return sync_wrapper
    
    return decorator


def track_errors(
    metric_name: Optional[str] = None,
    labels: Optional[Dict[str, str]] = None,
    include_traceback: bool = False
) -> Callable[[F], F]:
    """
    Decorator to track function errors.
    
    Args:
        metric_name: Name of the error counter metric
        labels: Static labels
        include_traceback: Whether to log full traceback
        
    Returns:
        Decorated function
    """
    def decorator(func: F) -> F:
        error_metric = metric_name or f"{func.__module__}.{func.__name__}_errors_total"
        
        if asyncio.iscoroutinefunction(func):
            @functools.wraps(func)
            async def async_wrapper(*args, **kwargs):
                try:
                    return await func(*args, **kwargs)
                except Exception as e:
                    error_labels = dict(labels or {})
                    error_labels["error_type"] = type(e).__name__
                    error_labels["function"] = func.__name__
                    
                    increment_counter(error_metric, labels=error_labels)
                    
                    if include_traceback:
                        logger.error(f"Error in {func.__name__}: {e}\n{traceback.format_exc()}")
                    
                    raise
            
            return async_wrapper
        else:
            @functools.wraps(func)
            def sync_wrapper(*args, **kwargs):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    error_labels = dict(labels or {})
                    error_labels["error_type"] = type(e).__name__
                    error_labels["function"] = func.__name__
                    
                    increment_counter(error_metric, labels=error_labels)
                    
                    if include_traceback:
                        logger.error(f"Error in {func.__name__}: {e}\n{traceback.format_exc()}")
                    
                    raise
            
            return sync_wrapper
    
    return decorator


def monitor_resource(
    resource_name: str,
    metric_name: Optional[str] = None,
    track_count: bool = True,
    track_usage: bool = False
) -> Callable[[F], F]:
    """
    Decorator to monitor resource usage (e.g., database connections).
    
    Args:
        resource_name: Name of the resource
        metric_name: Base name for metrics
        track_count: Track active resource count
        track_usage: Track resource usage duration
        
    Returns:
        Decorated function
    """
    def decorator(func: F) -> F:
        base_metric = metric_name or f"resource_{resource_name}"
        count_metric = f"{base_metric}_active"
        usage_metric = f"{base_metric}_usage_seconds"
        
        if asyncio.iscoroutinefunction(func):
            @functools.wraps(func)
            async def async_wrapper(*args, **kwargs):
                if track_count:
                    increment_counter(count_metric, 1, labels={"resource": resource_name})
                
                start_time = time.time() if track_usage else None
                
                try:
                    return await func(*args, **kwargs)
                finally:
                    if track_count:
                        increment_counter(count_metric, -1, labels={"resource": resource_name})
                    
                    if track_usage and start_time:
                        usage = time.time() - start_time
                        observe_histogram(usage_metric, usage, labels={"resource": resource_name})
            
            return async_wrapper
        else:
            @functools.wraps(func)
            def sync_wrapper(*args, **kwargs):
                if track_count:
                    increment_counter(count_metric, 1, labels={"resource": resource_name})
                
                start_time = time.time() if track_usage else None
                
                try:
                    return func(*args, **kwargs)
                finally:
                    if track_count:
                        increment_counter(count_metric, -1, labels={"resource": resource_name})
                    
                    if track_usage and start_time:
                        usage = time.time() - start_time
                        observe_histogram(usage_metric, usage, labels={"resource": resource_name})
            
            return sync_wrapper
    
    return decorator


def track_llm_usage(
    provider: Optional[str] = None,
    model: Optional[str] = None,
    track_tokens: bool = True,
    track_cost: bool = True,
    cost_per_1k_prompt: float = 0.0,
    cost_per_1k_completion: float = 0.0
) -> Callable[[F], F]:
    """
    Decorator to track LLM API usage and costs.
    
    Args:
        provider: LLM provider name
        model: Model name
        track_tokens: Track token usage
        track_cost: Track API costs
        cost_per_1k_prompt: Cost per 1000 prompt tokens
        cost_per_1k_completion: Cost per 1000 completion tokens
        
    Returns:
        Decorated function
    """
    def decorator(func: F) -> F:
        if asyncio.iscoroutinefunction(func):
            @functools.wraps(func)
            async def async_wrapper(*args, **kwargs):
                labels = {
                    "provider": provider or "unknown",
                    "model": model or "unknown"
                }
                
                # Track request
                increment_counter("llm_requests_total", labels={**labels, "status": "started"})
                
                start_time = time.time()
                try:
                    result = await func(*args, **kwargs)
                    
                    # Track success
                    increment_counter("llm_requests_total", labels={**labels, "status": "success"})
                    
                    # Track duration
                    duration = time.time() - start_time
                    observe_histogram("llm_request_duration_seconds", duration, labels=labels)
                    
                    # Extract token counts if available
                    if track_tokens and isinstance(result, dict):
                        prompt_tokens = result.get("usage", {}).get("prompt_tokens", 0)
                        completion_tokens = result.get("usage", {}).get("completion_tokens", 0)
                        
                        if prompt_tokens:
                            increment_counter(
                                "llm_tokens_used_total",
                                prompt_tokens,
                                labels={**labels, "type": "prompt"}
                            )
                        
                        if completion_tokens:
                            increment_counter(
                                "llm_tokens_used_total",
                                completion_tokens,
                                labels={**labels, "type": "completion"}
                            )
                        
                        # Calculate and track cost
                        if track_cost and (cost_per_1k_prompt or cost_per_1k_completion):
                            cost = (
                                (prompt_tokens / 1000) * cost_per_1k_prompt +
                                (completion_tokens / 1000) * cost_per_1k_completion
                            )
                            increment_counter("llm_cost_dollars", cost, labels=labels)
                    
                    return result
                    
                except Exception as e:
                    increment_counter("llm_requests_total", labels={**labels, "status": "error"})
                    raise
            
            return async_wrapper
        else:
            @functools.wraps(func)
            def sync_wrapper(*args, **kwargs):
                labels = {
                    "provider": provider or "unknown",
                    "model": model or "unknown"
                }
                
                # Track request
                increment_counter("llm_requests_total", labels={**labels, "status": "started"})
                
                start_time = time.time()
                try:
                    result = func(*args, **kwargs)
                    
                    # Track success
                    increment_counter("llm_requests_total", labels={**labels, "status": "success"})
                    
                    # Track duration
                    duration = time.time() - start_time
                    observe_histogram("llm_request_duration_seconds", duration, labels=labels)
                    
                    # Extract token counts if available
                    if track_tokens and isinstance(result, dict):
                        prompt_tokens = result.get("usage", {}).get("prompt_tokens", 0)
                        completion_tokens = result.get("usage", {}).get("completion_tokens", 0)
                        
                        if prompt_tokens:
                            increment_counter(
                                "llm_tokens_used_total",
                                prompt_tokens,
                                labels={**labels, "type": "prompt"}
                            )
                        
                        if completion_tokens:
                            increment_counter(
                                "llm_tokens_used_total",
                                completion_tokens,
                                labels={**labels, "type": "completion"}
                            )
                        
                        # Calculate and track cost
                        if track_cost and (cost_per_1k_prompt or cost_per_1k_completion):
                            cost = (
                                (prompt_tokens / 1000) * cost_per_1k_prompt +
                                (completion_tokens / 1000) * cost_per_1k_completion
                            )
                            increment_counter("llm_cost_dollars", cost, labels=labels)
                    
                    return result
                    
                except Exception as e:
                    increment_counter("llm_requests_total", labels={**labels, "status": "error"})
                    raise
            
            return sync_wrapper
    
    return decorator


def cache_metrics(
    cache_name: str,
    track_ratio: bool = True
) -> Callable[[F], F]:
    """
    Decorator to track cache hit/miss metrics.
    
    Args:
        cache_name: Name of the cache
        track_ratio: Track hit ratio as gauge
        
    Returns:
        Decorated function
    """
    def decorator(func: F) -> F:
        # Assume function returns tuple (result, cache_hit: bool)
        # or has a way to determine cache hit
        
        if asyncio.iscoroutinefunction(func):
            @functools.wraps(func)
            async def async_wrapper(*args, **kwargs):
                result = await func(*args, **kwargs)
                
                # Check if result indicates cache hit
                cache_hit = False
                actual_result = result
                
                if isinstance(result, tuple) and len(result) == 2:
                    actual_result, cache_hit = result
                elif hasattr(result, "from_cache"):
                    cache_hit = result.from_cache
                
                # Track metrics
                if cache_hit:
                    increment_counter("cache_hits_total", labels={"cache": cache_name})
                else:
                    increment_counter("cache_misses_total", labels={"cache": cache_name})
                
                # Track hit ratio if enabled
                if track_ratio:
                    registry = get_metrics_registry()
                    hits = registry.get_metric_stats("cache_hits_total", {"cache": cache_name}).get("sum", 0)
                    misses = registry.get_metric_stats("cache_misses_total", {"cache": cache_name}).get("sum", 0)
                    total = hits + misses
                    ratio = hits / total if total > 0 else 0
                    set_gauge("cache_hit_ratio", ratio, labels={"cache": cache_name})
                
                return actual_result
            
            return async_wrapper
        else:
            @functools.wraps(func)
            def sync_wrapper(*args, **kwargs):
                result = func(*args, **kwargs)
                
                # Check if result indicates cache hit
                cache_hit = False
                actual_result = result
                
                if isinstance(result, tuple) and len(result) == 2:
                    actual_result, cache_hit = result
                elif hasattr(result, "from_cache"):
                    cache_hit = result.from_cache
                
                # Track metrics
                if cache_hit:
                    increment_counter("cache_hits_total", labels={"cache": cache_name})
                else:
                    increment_counter("cache_misses_total", labels={"cache": cache_name})
                
                # Track hit ratio if enabled
                if track_ratio:
                    registry = get_metrics_registry()
                    hits = registry.get_metric_stats("cache_hits_total", {"cache": cache_name}).get("sum", 0)
                    misses = registry.get_metric_stats("cache_misses_total", {"cache": cache_name}).get("sum", 0)
                    total = hits + misses
                    ratio = hits / total if total > 0 else 0
                    set_gauge("cache_hit_ratio", ratio, labels={"cache": cache_name})
                
                return actual_result
            
            return sync_wrapper
    
    return decorator