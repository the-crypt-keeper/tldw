# utils/metrics.py
"""
Metrics collection for the chunking system.
Uses Prometheus client for metrics export.
"""

from typing import Optional, Dict, Any
from functools import wraps
import time
from loguru import logger

# Try to import prometheus_client
try:
    from prometheus_client import Counter, Histogram, Gauge, Summary
    PROMETHEUS_AVAILABLE = True
except ImportError:
    PROMETHEUS_AVAILABLE = False
    logger.warning(
        "prometheus_client not available. Metrics collection disabled. "
        "Install with: pip install prometheus-client"
    )
    
    # Create dummy classes for when Prometheus is not available
    class DummyMetric:
        def labels(self, **kwargs):
            return self
        def inc(self, amount=1):
            pass
        def dec(self, amount=1):
            pass
        def set(self, value):
            pass
        def observe(self, value):
            pass
        def time(self):
            return DummyTimer()
    
    class DummyTimer:
        def __enter__(self):
            return self
        def __exit__(self, *args):
            pass
    
    Counter = Histogram = Gauge = Summary = lambda *args, **kwargs: DummyMetric()


class ChunkingMetrics:
    """
    Metrics collector for the chunking system.
    Tracks performance, usage, and error metrics.
    """
    
    def __init__(self):
        """Initialize metrics collectors."""
        
        # Request metrics
        self.chunking_requests = Counter(
            'chunking_requests_total',
            'Total number of chunking requests',
            ['method', 'status']
        )
        
        # Performance metrics
        self.chunking_duration = Histogram(
            'chunking_duration_seconds',
            'Time spent chunking text',
            ['method'],
            buckets=(0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0, 5.0, 10.0)
        )
        
        self.chunk_size_histogram = Histogram(
            'chunk_size_characters',
            'Size of generated chunks in characters',
            ['method'],
            buckets=(10, 50, 100, 500, 1000, 5000, 10000, 50000)
        )
        
        self.chunks_per_request = Histogram(
            'chunks_per_request',
            'Number of chunks generated per request',
            ['method'],
            buckets=(1, 5, 10, 25, 50, 100, 250, 500, 1000)
        )
        
        # Input metrics
        self.input_size = Histogram(
            'chunking_input_size_bytes',
            'Size of input text in bytes',
            ['method'],
            buckets=(100, 1000, 10000, 100000, 1000000, 10000000)
        )
        
        # Cache metrics
        self.cache_hits = Counter(
            'chunking_cache_hits_total',
            'Total number of cache hits',
            ['method']
        )
        
        self.cache_misses = Counter(
            'chunking_cache_misses_total',
            'Total number of cache misses',
            ['method']
        )
        
        self.cache_size = Gauge(
            'chunking_cache_size',
            'Current size of the chunking cache'
        )
        
        # Error metrics
        self.chunking_errors = Counter(
            'chunking_errors_total',
            'Total number of chunking errors',
            ['method', 'error_type']
        )
        
        # Strategy-specific metrics
        self.tokenizer_calls = Counter(
            'tokenizer_calls_total',
            'Total number of tokenizer calls',
            ['tokenizer_type']
        )
        
        self.semantic_similarity_computations = Counter(
            'semantic_similarity_computations_total',
            'Total number of semantic similarity computations'
        )
        
        # Memory metrics
        self.memory_usage = Gauge(
            'chunking_memory_usage_bytes',
            'Current memory usage of chunking system'
        )
        
        logger.info(f"ChunkingMetrics initialized (Prometheus available: {PROMETHEUS_AVAILABLE})")
    
    def record_request(self, method: str, status: str = 'success'):
        """
        Record a chunking request.
        
        Args:
            method: Chunking method used
            status: Request status ('success', 'error', 'cached')
        """
        self.chunking_requests.labels(method=method, status=status).inc()
    
    def record_duration(self, method: str, duration: float):
        """
        Record chunking duration.
        
        Args:
            method: Chunking method used
            duration: Time taken in seconds
        """
        self.chunking_duration.labels(method=method).observe(duration)
    
    def record_chunks(self, method: str, chunks: list):
        """
        Record chunk statistics.
        
        Args:
            method: Chunking method used
            chunks: List of generated chunks
        """
        # Record number of chunks
        self.chunks_per_request.labels(method=method).observe(len(chunks))
        
        # Record size of each chunk
        for chunk in chunks:
            if isinstance(chunk, str):
                self.chunk_size_histogram.labels(method=method).observe(len(chunk))
            elif isinstance(chunk, dict) and 'text' in chunk:
                self.chunk_size_histogram.labels(method=method).observe(len(chunk['text']))
    
    def record_input_size(self, method: str, text: str):
        """
        Record input text size.
        
        Args:
            method: Chunking method used
            text: Input text
        """
        self.input_size.labels(method=method).observe(len(text.encode('utf-8')))
    
    def record_cache_hit(self, method: str):
        """Record a cache hit."""
        self.cache_hits.labels(method=method).inc()
    
    def record_cache_miss(self, method: str):
        """Record a cache miss."""
        self.cache_misses.labels(method=method).inc()
    
    def update_cache_size(self, size: int):
        """Update cache size metric."""
        self.cache_size.set(size)
    
    def record_error(self, method: str, error_type: str):
        """
        Record a chunking error.
        
        Args:
            method: Chunking method that failed
            error_type: Type of error (e.g., 'InvalidInput', 'TokenizerError')
        """
        self.chunking_errors.labels(method=method, error_type=error_type).inc()
    
    def record_tokenizer_call(self, tokenizer_type: str):
        """Record a tokenizer call."""
        self.tokenizer_calls.labels(tokenizer_type=tokenizer_type).inc()
    
    def record_similarity_computation(self):
        """Record a semantic similarity computation."""
        self.semantic_similarity_computations.inc()
    
    def update_memory_usage(self, bytes_used: int):
        """Update memory usage metric."""
        self.memory_usage.set(bytes_used)
    
    def time_chunking(self, method: str):
        """
        Context manager for timing chunking operations.
        
        Args:
            method: Chunking method being timed
            
        Returns:
            Timer context manager
        """
        return self.chunking_duration.labels(method=method).time()


# Global metrics instance
_metrics_instance: Optional[ChunkingMetrics] = None


def get_metrics() -> ChunkingMetrics:
    """
    Get the global metrics instance.
    
    Returns:
        ChunkingMetrics instance
    """
    global _metrics_instance
    if _metrics_instance is None:
        _metrics_instance = ChunkingMetrics()
    return _metrics_instance


def metrics_decorator(method_name: Optional[str] = None):
    """
    Decorator for automatically collecting metrics.
    
    Args:
        method_name: Override method name for metrics
        
    Returns:
        Decorated function
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Extract method from kwargs or use override
            method = method_name or kwargs.get('method', 'unknown')
            metrics = get_metrics()
            
            # Record input size if text is provided
            if args and isinstance(args[0], str):
                metrics.record_input_size(method, args[0])
            elif 'text' in kwargs:
                metrics.record_input_size(method, kwargs['text'])
            
            # Time the operation
            start_time = time.time()
            
            try:
                result = func(*args, **kwargs)
                
                # Record success
                metrics.record_request(method, 'success')
                
                # Record chunks if result is a list
                if isinstance(result, list):
                    metrics.record_chunks(method, result)
                
                return result
                
            except Exception as e:
                # Record error
                error_type = type(e).__name__
                metrics.record_request(method, 'error')
                metrics.record_error(method, error_type)
                raise
                
            finally:
                # Record duration
                duration = time.time() - start_time
                metrics.record_duration(method, duration)
        
        return wrapper
    return decorator


class MetricsContext:
    """
    Context manager for collecting metrics during chunking operations.
    """
    
    def __init__(self, method: str, metrics: Optional[ChunkingMetrics] = None):
        """
        Initialize metrics context.
        
        Args:
            method: Chunking method being used
            metrics: Metrics instance (uses global if not provided)
        """
        self.method = method
        self.metrics = metrics or get_metrics()
        self.start_time = None
        self.success = False
        self.error_type = None
    
    def __enter__(self):
        """Enter context."""
        self.start_time = time.time()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Exit context and record metrics."""
        duration = time.time() - self.start_time
        
        if exc_type is None:
            self.metrics.record_request(self.method, 'success')
        else:
            self.metrics.record_request(self.method, 'error')
            self.metrics.record_error(self.method, exc_type.__name__)
        
        self.metrics.record_duration(self.method, duration)
        
        # Don't suppress exceptions
        return False
    
    def record_chunks(self, chunks: list):
        """Record chunk statistics."""
        self.metrics.record_chunks(self.method, chunks)
    
    def record_cache_hit(self):
        """Record a cache hit."""
        self.metrics.record_cache_hit(self.method)
    
    def record_cache_miss(self):
        """Record a cache miss."""
        self.metrics.record_cache_miss(self.method)