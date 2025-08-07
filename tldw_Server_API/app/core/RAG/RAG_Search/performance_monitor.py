# performance_monitor.py - RAG Performance Monitoring
"""
Performance monitoring integration for RAG components.

Integrates with the existing metrics system to track:
- Query expansion performance
- Reranking latency
- Cache hit rates
- Chunking efficiency
- Vector search performance
- End-to-end latency
"""

import time
import asyncio
from typing import Dict, Any, List, Optional, Callable, TypeVar, Coroutine
from functools import wraps
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import json
import numpy as np
from loguru import logger

# Import existing metrics system
from tldw_Server_API.app.core.Metrics.metrics_logger import (
    log_counter, log_histogram, log_gauge, log_summary
)

T = TypeVar('T')


@dataclass
class PerformanceMetrics:
    """Container for performance metrics"""
    component: str
    operation: str
    start_time: float
    end_time: Optional[float] = None
    duration_ms: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def complete(self):
        """Mark operation as complete"""
        self.end_time = time.time()
        self.duration_ms = (self.end_time - self.start_time) * 1000


class RAGPerformanceMonitor:
    """Performance monitoring for RAG components"""
    
    def __init__(self, enable_detailed_tracking: bool = True):
        self.enable_detailed_tracking = enable_detailed_tracking
        self._metrics_buffer: List[PerformanceMetrics] = []
        self._component_stats: Dict[str, Dict[str, Any]] = {}
        
        # Start background metrics aggregator
        if enable_detailed_tracking:
            asyncio.create_task(self._metrics_aggregator())
    
    def track_operation(self, component: str, operation: str) -> PerformanceMetrics:
        """Start tracking an operation"""
        metric = PerformanceMetrics(
            component=component,
            operation=operation,
            start_time=time.time()
        )
        self._metrics_buffer.append(metric)
        return metric
    
    @staticmethod
    def monitor_async(component: str, operation: str):
        """Decorator for monitoring async functions"""
        def decorator(func: Callable[..., Coroutine[Any, Any, T]]) -> Callable[..., Coroutine[Any, Any, T]]:
            @wraps(func)
            async def wrapper(*args, **kwargs) -> T:
                start_time = time.time()
                labels = {"component": component, "operation": operation}
                
                try:
                    result = await func(*args, **kwargs)
                    
                    # Log success metrics
                    duration_ms = (time.time() - start_time) * 1000
                    log_histogram(
                        f"rag_{component}_{operation}_duration_ms",
                        duration_ms,
                        labels=labels
                    )
                    log_counter(
                        f"rag_{component}_{operation}_success",
                        labels=labels
                    )
                    
                    return result
                    
                except Exception as e:
                    # Log failure metrics
                    log_counter(
                        f"rag_{component}_{operation}_failure",
                        labels={**labels, "error_type": type(e).__name__}
                    )
                    raise
            
            return wrapper
        return decorator
    
    @staticmethod
    def monitor_sync(component: str, operation: str):
        """Decorator for monitoring sync functions"""
        def decorator(func: Callable[..., T]) -> Callable[..., T]:
            @wraps(func)
            def wrapper(*args, **kwargs) -> T:
                start_time = time.time()
                labels = {"component": component, "operation": operation}
                
                try:
                    result = func(*args, **kwargs)
                    
                    # Log success metrics
                    duration_ms = (time.time() - start_time) * 1000
                    log_histogram(
                        f"rag_{component}_{operation}_duration_ms",
                        duration_ms,
                        labels=labels
                    )
                    log_counter(
                        f"rag_{component}_{operation}_success",
                        labels=labels
                    )
                    
                    return result
                    
                except Exception as e:
                    # Log failure metrics
                    log_counter(
                        f"rag_{component}_{operation}_failure",
                        labels={**labels, "error_type": type(e).__name__}
                    )
                    raise
            
            return wrapper
        return decorator
    
    # Specific monitoring methods for RAG components
    
    @monitor_async("query_expansion", "expand")
    async def monitor_query_expansion(self, expander, query: str) -> List[str]:
        """Monitor query expansion performance"""
        start_time = time.time()
        
        # Track input characteristics
        log_gauge("rag_query_expansion_input_length", len(query))
        
        # Perform expansion
        expansions = await expander.expand_query(query)
        
        # Track output characteristics
        log_gauge("rag_query_expansion_output_count", len(expansions))
        log_gauge(
            "rag_query_expansion_avg_length",
            sum(len(e) for e in expansions) / len(expansions) if expansions else 0
        )
        
        # Track expansion ratio
        if query:
            total_expansion_length = sum(len(e) for e in expansions)
            expansion_ratio = total_expansion_length / len(query)
            log_gauge("rag_query_expansion_ratio", expansion_ratio)
        
        return expansions
    
    @monitor_async("reranking", "rerank")
    async def monitor_reranking(self, reranker, query: str, documents: List[Dict], 
                               strategy: str = "unknown") -> List[Any]:
        """Monitor reranking performance"""
        start_time = time.time()
        
        # Track input size
        log_gauge(
            "rag_reranking_input_docs",
            len(documents),
            labels={"strategy": strategy}
        )
        
        # Perform reranking
        reranked = await reranker.rerank(query, documents)
        
        # Track output characteristics
        log_gauge(
            "rag_reranking_output_docs",
            len(reranked),
            labels={"strategy": strategy}
        )
        
        # Track score distribution if available
        if reranked and hasattr(reranked[0], 'rerank_score'):
            scores = [doc.rerank_score for doc in reranked]
            log_summary(
                "rag_reranking_score_distribution",
                scores,
                labels={"strategy": strategy}
            )
        
        return reranked
    
    def monitor_cache_operation(self, cache_strategy: str, operation: str, 
                               hit: bool, latency_ms: float):
        """Monitor cache performance"""
        labels = {
            "strategy": cache_strategy,
            "operation": operation,
            "hit": str(hit).lower()
        }
        
        # Log cache metrics
        log_histogram("rag_cache_latency_ms", latency_ms, labels=labels)
        log_counter(f"rag_cache_{operation}", labels=labels)
        
        if operation == "get":
            log_counter(
                "rag_cache_hits" if hit else "rag_cache_misses",
                labels={"strategy": cache_strategy}
            )
    
    @monitor_sync("chunking", "chunk")
    def monitor_chunking(self, chunker, text: str, strategy: str = "unknown") -> List[Any]:
        """Monitor chunking performance"""
        start_time = time.time()
        
        # Track input size
        log_gauge(
            "rag_chunking_input_chars",
            len(text),
            labels={"strategy": strategy}
        )
        log_gauge(
            "rag_chunking_input_words",
            len(text.split()),
            labels={"strategy": strategy}
        )
        
        # Perform chunking
        chunks = chunker.chunk(text)
        
        # Track output characteristics
        log_gauge(
            "rag_chunking_output_chunks",
            len(chunks),
            labels={"strategy": strategy}
        )
        
        if chunks:
            # Track chunk size distribution
            chunk_sizes = [chunk.metadata.char_count for chunk in chunks]
            log_summary(
                "rag_chunking_size_distribution",
                chunk_sizes,
                labels={"strategy": strategy}
            )
            
            # Track chunking efficiency
            total_chunk_chars = sum(chunk_sizes)
            overlap_ratio = (total_chunk_chars - len(text)) / len(text) if text else 0
            log_gauge(
                "rag_chunking_overlap_ratio",
                overlap_ratio,
                labels={"strategy": strategy}
            )
        
        return chunks
    
    @monitor_async("vector_search", "search")
    async def monitor_vector_search(self, vector_store, query_embedding: Any, 
                                   top_k: int = 10) -> List[Any]:
        """Monitor vector search performance"""
        start_time = time.time()
        
        # Track search parameters
        log_gauge("rag_vector_search_top_k", top_k)
        
        if hasattr(query_embedding, 'shape'):
            log_gauge("rag_vector_search_embedding_dim", query_embedding.shape[-1])
        
        # Perform search
        results = await vector_store.search(query_embedding, top_k)
        
        # Track results
        log_gauge("rag_vector_search_results_returned", len(results))
        
        if results:
            # Track score distribution
            scores = [r.score for r in results if hasattr(r, 'score')]
            if scores:
                log_summary("rag_vector_search_score_distribution", scores)
                
                # Track score gap (difference between top scores)
                if len(scores) > 1:
                    score_gap = scores[0] - scores[1]
                    log_gauge("rag_vector_search_top_score_gap", score_gap)
        
        return results
    
    def monitor_end_to_end(self, query: str, start_time: float, 
                          stages: Dict[str, float], result_count: int):
        """Monitor end-to-end RAG performance"""
        total_duration = time.time() - start_time
        
        # Log total duration
        log_histogram("rag_e2e_total_duration_ms", total_duration * 1000)
        
        # Log stage durations
        for stage, duration in stages.items():
            log_histogram(
                f"rag_e2e_stage_duration_ms",
                duration * 1000,
                labels={"stage": stage}
            )
            
            # Calculate stage percentage
            stage_pct = (duration / total_duration * 100) if total_duration > 0 else 0
            log_gauge(
                "rag_e2e_stage_percentage",
                stage_pct,
                labels={"stage": stage}
            )
        
        # Log query characteristics
        log_gauge("rag_e2e_query_length", len(query))
        log_gauge("rag_e2e_results_count", result_count)
        
        # Log throughput
        if total_duration > 0:
            qps = 1 / total_duration
            log_gauge("rag_e2e_queries_per_second", qps)
    
    async def _metrics_aggregator(self):
        """Background task to aggregate and report metrics"""
        while True:
            try:
                await asyncio.sleep(60)  # Aggregate every minute
                
                if self._metrics_buffer:
                    # Process buffered metrics
                    completed_metrics = [
                        m for m in self._metrics_buffer 
                        if m.end_time is not None
                    ]
                    
                    # Group by component and operation
                    for metric in completed_metrics:
                        key = f"{metric.component}:{metric.operation}"
                        if key not in self._component_stats:
                            self._component_stats[key] = {
                                "count": 0,
                                "total_ms": 0,
                                "min_ms": float('inf'),
                                "max_ms": 0
                            }
                        
                        stats = self._component_stats[key]
                        stats["count"] += 1
                        stats["total_ms"] += metric.duration_ms
                        stats["min_ms"] = min(stats["min_ms"], metric.duration_ms)
                        stats["max_ms"] = max(stats["max_ms"], metric.duration_ms)
                    
                    # Log aggregated stats
                    for key, stats in self._component_stats.items():
                        component, operation = key.split(":")
                        avg_ms = stats["total_ms"] / stats["count"] if stats["count"] > 0 else 0
                        
                        labels = {"component": component, "operation": operation}
                        log_gauge("rag_aggregated_avg_duration_ms", avg_ms, labels=labels)
                        log_gauge("rag_aggregated_min_duration_ms", stats["min_ms"], labels=labels)
                        log_gauge("rag_aggregated_max_duration_ms", stats["max_ms"], labels=labels)
                        log_counter("rag_aggregated_operation_count", value=stats["count"], labels=labels)
                    
                    # Clear processed metrics
                    self._metrics_buffer = [
                        m for m in self._metrics_buffer 
                        if m.end_time is None
                    ]
                    
            except Exception as e:
                logger.error(f"Error in metrics aggregator: {e}")
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get performance summary"""
        summary = {
            "timestamp": datetime.now().isoformat(),
            "components": {}
        }
        
        for key, stats in self._component_stats.items():
            component, operation = key.split(":")
            if component not in summary["components"]:
                summary["components"][component] = {}
            
            avg_ms = stats["total_ms"] / stats["count"] if stats["count"] > 0 else 0
            summary["components"][component][operation] = {
                "count": stats["count"],
                "avg_ms": round(avg_ms, 2),
                "min_ms": round(stats["min_ms"], 2),
                "max_ms": round(stats["max_ms"], 2)
            }
        
        return summary


# Global monitor instance
_monitor = None

def get_performance_monitor() -> RAGPerformanceMonitor:
    """Get or create global performance monitor"""
    global _monitor
    if _monitor is None:
        _monitor = RAGPerformanceMonitor()
    return _monitor


# Convenience decorators
def monitor_query_expansion(func):
    """Decorator to monitor query expansion functions"""
    return RAGPerformanceMonitor.monitor_async("query_expansion", func.__name__)(func)

def monitor_reranking(func):
    """Decorator to monitor reranking functions"""
    return RAGPerformanceMonitor.monitor_async("reranking", func.__name__)(func)

def monitor_chunking(func):
    """Decorator to monitor chunking functions"""
    return RAGPerformanceMonitor.monitor_sync("chunking", func.__name__)(func)

def monitor_vector_search(func):
    """Decorator to monitor vector search functions"""
    return RAGPerformanceMonitor.monitor_async("vector_search", func.__name__)(func)


# Example usage with existing components
class MonitoredRAGPipeline:
    """Example of RAG pipeline with performance monitoring"""
    
    def __init__(self, query_expander, reranker, chunker, vector_store):
        self.query_expander = query_expander
        self.reranker = reranker
        self.chunker = chunker
        self.vector_store = vector_store
        self.monitor = get_performance_monitor()
    
    @monitor_query_expansion
    async def expand_query(self, query: str) -> List[str]:
        """Expand query with monitoring"""
        return await self.query_expander.expand_query(query)
    
    @monitor_reranking
    async def rerank_results(self, query: str, documents: List[Dict]) -> List[Any]:
        """Rerank results with monitoring"""
        return await self.reranker.rerank(query, documents)
    
    @monitor_chunking
    def chunk_document(self, text: str) -> List[Any]:
        """Chunk document with monitoring"""
        return self.chunker.chunk(text)
    
    @monitor_vector_search
    async def search(self, query_embedding: Any, top_k: int = 10) -> List[Any]:
        """Search with monitoring"""
        return await self.vector_store.search(query_embedding, top_k)
    
    async def process_query(self, query: str) -> Dict[str, Any]:
        """Complete RAG pipeline with monitoring"""
        start_time = time.time()
        stages = {}
        
        # Stage 1: Query expansion
        stage_start = time.time()
        expansions = await self.expand_query(query)
        stages["query_expansion"] = time.time() - stage_start
        
        # Stage 2: Vector search (mock)
        stage_start = time.time()
        # In real implementation, generate embedding first
        mock_embedding = [0.1] * 384  # Mock embedding
        search_results = await self.search(mock_embedding)
        stages["vector_search"] = time.time() - stage_start
        
        # Stage 3: Reranking
        stage_start = time.time()
        if search_results:
            # Convert to format expected by reranker
            docs_for_rerank = [
                {"content": r.document if hasattr(r, 'document') else str(r)}
                for r in search_results
            ]
            reranked = await self.rerank_results(query, docs_for_rerank)
        else:
            reranked = []
        stages["reranking"] = time.time() - stage_start
        
        # Monitor end-to-end
        self.monitor.monitor_end_to_end(
            query, start_time, stages, len(reranked)
        )
        
        return {
            "query": query,
            "expansions": expansions,
            "results": reranked,
            "metrics": {
                "total_duration_ms": (time.time() - start_time) * 1000,
                "stages_ms": {k: v * 1000 for k, v in stages.items()}
            }
        }


# Testing
if __name__ == "__main__":
    async def test_monitoring():
        """Test performance monitoring"""
        from .advanced_query_expansion import AdvancedQueryExpander, ExpansionConfig
        from .advanced_reranker import create_reranker, RerankingStrategy
        from .advanced_chunking import create_chunker, ChunkingStrategy
        
        # Mock vector store
        class MockVectorStore:
            async def search(self, embedding, top_k):
                # Simulate search delay
                await asyncio.sleep(0.1)
                return [
                    type('Result', (), {
                        'document': f'Document {i}',
                        'score': 0.9 - i * 0.1
                    })()
                    for i in range(min(5, top_k))
                ]
        
        # Create components
        expander = AdvancedQueryExpander(ExpansionConfig())
        reranker = create_reranker(RerankingStrategy.HYBRID)
        chunker = create_chunker(ChunkingStrategy.ADAPTIVE)
        vector_store = MockVectorStore()
        
        # Create monitored pipeline
        pipeline = MonitoredRAGPipeline(expander, reranker, chunker, vector_store)
        
        # Test queries
        test_queries = [
            "What is machine learning?",
            "How does RAG work?",
            "Explain NLP applications"
        ]
        
        print("Testing RAG Performance Monitoring")
        print("="*50)
        
        for query in test_queries:
            result = await pipeline.process_query(query)
            print(f"\nQuery: {query}")
            print(f"Expansions: {len(result['expansions'])}")
            print(f"Results: {len(result['results'])}")
            print(f"Total time: {result['metrics']['total_duration_ms']:.2f}ms")
            print("Stage breakdown:")
            for stage, duration in result['metrics']['stages_ms'].items():
                print(f"  - {stage}: {duration:.2f}ms")
        
        # Get performance summary
        summary = pipeline.monitor.get_performance_summary()
        print("\nPerformance Summary:")
        print(json.dumps(summary, indent=2))
    
    asyncio.run(test_monitoring())