"""
Functional Pipeline for RAG Service

A modular, plug-n-play functional pipeline where each component is a function
that can be composed and chained together based on runtime configuration.

Key Features:
- Pure function composition
- Easy to extend with new functions
- Runtime configurable pipeline
- No complex class hierarchies
"""

import asyncio
from typing import Dict, List, Any, Optional, Callable, Tuple
from dataclasses import dataclass, field
from functools import wraps
import time
from loguru import logger

from .types import Document, SearchResult, DataSource
from .config import RAGConfig

# Pipeline context that flows through all functions
@dataclass
class RAGPipelineContext:
    """Context that flows through the pipeline functions."""
    query: str
    original_query: str
    documents: List[Document] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    config: Dict[str, Any] = field(default_factory=dict)
    cache_hit: bool = False
    timings: Dict[str, float] = field(default_factory=dict)
    errors: List[Dict[str, Any]] = field(default_factory=list)


def timer(func_name: Optional[str] = None):
    """Decorator to time pipeline functions."""
    def decorator(func):
        @wraps(func)
        async def wrapper(context: RAGPipelineContext, *args, **kwargs):
            name = func_name or func.__name__
            start = time.time()
            try:
                result = await func(context, *args, **kwargs)
                context.timings[name] = time.time() - start
                return result
            except Exception as e:
                context.timings[name] = time.time() - start
                context.errors.append({"function": name, "error": str(e)})
                logger.error(f"Error in {name}: {e}")
                raise
        return wrapper
    return decorator


# Query Expansion Functions

@timer("query_expansion")
async def expand_query(context: RAGPipelineContext, 
                       strategies: List[str] = None) -> RAGPipelineContext:
    """Expand query using multiple strategies."""
    from .query_expansion import (
        HybridQueryExpansion, AcronymExpansion, SemanticExpansion,
        DomainExpansion, EntityExpansion
    )
    
    strategies = strategies or context.config.get("expansion_strategies", ["acronym", "semantic"])
    
    expander = HybridQueryExpansion()
    
    # Add requested strategies
    if "acronym" in strategies:
        expander.add_strategy(AcronymExpansion())
    if "semantic" in strategies:
        expander.add_strategy(SemanticExpansion())
    if "domain" in strategies:
        expander.add_strategy(DomainExpansion())
    if "entity" in strategies:
        expander.add_strategy(EntityExpansion())
    
    expanded = await expander.expand(context.query)
    
    context.metadata["expanded_queries"] = expanded
    context.metadata["expansion_strategies"] = strategies
    logger.debug(f"Expanded query to {len(expanded)} variants")
    
    return context


# Cache Functions

@timer("cache_lookup")
async def check_cache(context: RAGPipelineContext,
                      threshold: float = None) -> RAGPipelineContext:
    """Check semantic cache for similar queries."""
    if not context.config.get("enable_cache", True):
        return context
    
    from .semantic_cache import SemanticCache, AdaptiveCache
    
    threshold = threshold or context.config.get("cache_threshold", 0.85)
    use_adaptive = context.config.get("use_adaptive_cache", True)
    
    cache = AdaptiveCache(initial_threshold=threshold) if use_adaptive else SemanticCache(similarity_threshold=threshold)
    
    cached_result = await cache.find_similar(context.query)
    
    if cached_result:
        cached_query, similarity = cached_result
        cached_docs = await cache.get(cached_query)
        
        if cached_docs:
            context.cache_hit = True
            context.documents = cached_docs
            context.metadata["cache_similarity"] = similarity
            context.metadata["cached_query"] = cached_query
            logger.info(f"Cache hit with similarity {similarity:.3f}")
    
    return context


@timer("cache_store")
async def store_in_cache(context: RAGPipelineContext) -> RAGPipelineContext:
    """Store results in cache for future queries."""
    if context.cache_hit or not context.documents:
        return context
    
    if not context.config.get("enable_cache", True):
        return context
    
    from .semantic_cache import SemanticCache
    
    cache = SemanticCache()
    await cache.add(context.query, context.documents)
    
    context.metadata["cached_for_future"] = True
    logger.debug(f"Cached {len(context.documents)} documents")
    
    return context


# Retrieval Functions

@timer("retrieval")
async def retrieve_documents(context: RAGPipelineContext,
                            sources: List[DataSource] = None) -> RAGPipelineContext:
    """Retrieve documents from configured sources."""
    if context.cache_hit:
        logger.debug("Skipping retrieval due to cache hit")
        return context
    
    sources = sources or context.config.get("sources", [DataSource.MEDIA_DB])
    
    # This would integrate with actual retrieval
    # For now, placeholder
    context.metadata["sources_searched"] = [s.value for s in sources]
    
    return context


# ChromaDB Optimization Functions

@timer("chromadb_optimization")
async def optimize_chromadb_search(context: RAGPipelineContext,
                                  enable_hybrid: bool = None,
                                  alpha: float = None) -> RAGPipelineContext:
    """Apply ChromaDB optimizations for hybrid search."""
    if context.cache_hit:
        return context
    
    from .chromadb_optimizer import ChromaDBOptimizer, ChromaDBOptimizationConfig
    
    enable_hybrid = enable_hybrid if enable_hybrid is not None else context.config.get("enable_hybrid_search", True)
    alpha = alpha or context.config.get("hybrid_alpha", 0.7)
    
    if enable_hybrid:
        config = ChromaDBOptimizationConfig(
            enable_hybrid_search=True,
            hybrid_alpha=alpha,
            cache_size=5000,
            batch_size=500,
            max_collection_size=100_000
        )
        
        optimizer = ChromaDBOptimizer(config)
        
        # This would integrate with actual ChromaDB operations
        context.metadata["chromadb_optimized"] = True
        context.metadata["hybrid_search_config"] = {
            "enabled": True,
            "alpha": alpha
        }
        
        logger.debug(f"ChromaDB optimization applied with alpha={alpha}")
    
    return context


# Processing Functions

@timer("table_serialization")
async def process_tables(context: RAGPipelineContext,
                        method: str = None) -> RAGPipelineContext:
    """Process and serialize tables in documents."""
    if not context.documents:
        return context
    
    from .table_serialization import TableProcessor
    
    method = method or context.config.get("table_serialize_method", "hybrid")
    processor = TableProcessor()
    
    total_tables = 0
    
    for doc in context.documents:
        if any(indicator in doc.content.lower() 
               for indicator in ["table", "|", "csv", "tsv", "<table"]):
            
            processed_content, tables = processor.process_document_tables(
                doc.content,
                serialize_method=method
            )
            
            if tables:
                doc.content = processed_content
                doc.metadata["tables_processed"] = len(tables)
                total_tables += len(tables)
    
    if total_tables > 0:
        context.metadata["total_tables_processed"] = total_tables
        logger.debug(f"Processed {total_tables} tables using {method} serialization")
    
    return context


# Reranking Functions

@timer("reranking")
async def rerank_documents(context: RAGPipelineContext,
                          strategy: str = None,
                          top_k: int = None) -> RAGPipelineContext:
    """Rerank documents using advanced strategies."""
    if not context.documents:
        return context
    
    from .advanced_reranking import create_reranker, RerankingStrategy, RerankingConfig
    
    strategy = strategy or context.config.get("reranking_strategy", "hybrid")
    top_k = top_k or context.config.get("top_k", 10)
    
    config = RerankingConfig(
        strategy=RerankingStrategy[strategy.upper()],
        top_k=top_k,
        diversity_weight=context.config.get("diversity_weight", 0.3),
        relevance_weight=context.config.get("relevance_weight", 0.7)
    )
    
    reranker = create_reranker(RerankingStrategy[strategy.upper()], config)
    
    original_scores = [doc.score for doc in context.documents]
    scored_docs = await reranker.rerank(
        context.query,
        context.documents,
        original_scores
    )
    
    context.documents = [sd.document for sd in scored_docs[:top_k]]
    context.metadata["reranking_applied"] = True
    context.metadata["reranking_strategy"] = strategy
    context.metadata["documents_reranked"] = f"{len(original_scores)} -> {len(context.documents)}"
    
    logger.debug(f"Reranked to {len(context.documents)} documents using {strategy}")
    
    return context


# Performance Monitoring Functions

@timer("performance_analysis")
async def analyze_performance(context: RAGPipelineContext) -> RAGPipelineContext:
    """Analyze and log performance metrics."""
    from .performance_monitor import PerformanceMonitor
    
    if context.config.get("enable_monitoring", True):
        monitor = PerformanceMonitor()
        
        total_time = sum(context.timings.values())
        
        monitor.record_query(
            query=context.query,
            total_duration=total_time,
            component_timings=context.timings,
            cache_hit=context.cache_hit
        )
        
        # Identify bottlenecks
        if total_time > 1.0:
            bottlenecks = [(k, v) for k, v in context.timings.items() if v > 0.3]
            if bottlenecks:
                logger.warning(f"Performance bottlenecks: {bottlenecks}")
        
        context.metadata["total_time"] = total_time
        context.metadata["performance_analyzed"] = True
    
    return context


# Main Pipeline Functions (Composable)

async def minimal_pipeline(query: str, config: Dict[str, Any] = None) -> RAGPipelineContext:
    """
    Minimal pipeline: just retrieval and basic reranking.
    Fast but less accurate.
    """
    config = config or {}
    context = RAGPipelineContext(query=query, original_query=query, config=config)
    
    # Simple chain
    context = await retrieve_documents(context)
    context = await rerank_documents(context, strategy="flashrank")
    
    return context


async def standard_pipeline(query: str, config: Dict[str, Any] = None) -> RAGPipelineContext:
    """
    Standard pipeline with caching, expansion, and reranking.
    Balanced between speed and quality.
    """
    config = config or {}
    context = RAGPipelineContext(query=query, original_query=query, config=config)
    
    # Standard chain
    context = await expand_query(context, strategies=["acronym", "semantic"])
    context = await check_cache(context)
    
    if not context.cache_hit:
        context = await retrieve_documents(context)
        context = await rerank_documents(context, strategy="flashrank")
        context = await store_in_cache(context)
    
    context = await analyze_performance(context)
    
    return context


async def quality_pipeline(query: str, config: Dict[str, Any] = None) -> RAGPipelineContext:
    """
    Quality pipeline with all enhancements.
    Best results but slower.
    """
    config = config or {}
    context = RAGPipelineContext(query=query, original_query=query, config=config)
    
    # Full chain with all modules
    context = await expand_query(context, strategies=["acronym", "semantic", "domain", "entity"])
    context = await check_cache(context, threshold=0.9)
    
    if not context.cache_hit:
        context = await optimize_chromadb_search(context, enable_hybrid=True, alpha=0.7)
        context = await retrieve_documents(context, sources=[DataSource.MEDIA_DB, DataSource.NOTES])
        context = await process_tables(context, method="hybrid")
        context = await rerank_documents(context, strategy="hybrid", top_k=20)
        context = await store_in_cache(context)
    
    context = await analyze_performance(context)
    
    return context


async def custom_pipeline(query: str, 
                         functions: List[Callable],
                         config: Dict[str, Any] = None) -> RAGPipelineContext:
    """
    Custom pipeline with user-defined function chain.
    Maximum flexibility.
    
    Args:
        query: Search query
        functions: List of async functions to execute in order
        config: Configuration dictionary
    
    Returns:
        Pipeline context with results
    """
    config = config or {}
    context = RAGPipelineContext(query=query, original_query=query, config=config)
    
    # Execute custom function chain
    for func in functions:
        try:
            context = await func(context)
        except Exception as e:
            logger.error(f"Error in {func.__name__}: {e}")
            if config.get("stop_on_error", False):
                raise
    
    return context


# Dynamic pipeline builder
def build_pipeline(*functions) -> Callable:
    """
    Build a custom pipeline from a sequence of functions.
    
    Example:
        my_pipeline = build_pipeline(
            expand_query,
            check_cache,
            retrieve_documents,
            rerank_documents
        )
        
        result = await my_pipeline(query, config)
    """
    async def pipeline(query: str, config: Dict[str, Any] = None) -> RAGPipelineContext:
        context = RAGPipelineContext(query=query, original_query=query, config=config or {})
        
        for func in functions:
            context = await func(context)
        
        return context
    
    # Set a meaningful name
    pipeline.__name__ = f"custom_pipeline_{'_'.join(f.__name__ for f in functions[:3])}"
    
    return pipeline


# Conditional pipeline components
async def conditional(condition: Callable,
                     if_true: Callable,
                     if_false: Optional[Callable] = None):
    """
    Conditional execution of pipeline functions.
    
    Example:
        await conditional(
            lambda ctx: not ctx.cache_hit,
            retrieve_documents,
            lambda ctx: ctx  # no-op if cache hit
        )
    """
    async def wrapper(context: RAGPipelineContext) -> RAGPipelineContext:
        if condition(context):
            return await if_true(context)
        elif if_false:
            return await if_false(context)
        return context
    
    return wrapper


# Parallel execution for independent operations
async def parallel(*functions):
    """
    Execute multiple functions in parallel and merge results.
    
    Example:
        await parallel(
            check_cache,
            expand_query
        )
    """
    async def wrapper(context: RAGPipelineContext) -> RAGPipelineContext:
        # Create copies of context for parallel execution
        contexts = [RAGPipelineContext(
            query=context.query,
            original_query=context.original_query,
            documents=context.documents.copy(),
            metadata=context.metadata.copy(),
            config=context.config.copy()
        ) for _ in functions]
        
        # Execute in parallel
        results = await asyncio.gather(*[
            func(ctx) for func, ctx in zip(functions, contexts)
        ])
        
        # Merge results back into original context
        for result in results:
            context.metadata.update(result.metadata)
            context.timings.update(result.timings)
            if result.cache_hit:
                context.cache_hit = True
                context.documents = result.documents
        
        return context
    
    return wrapper


# Pipeline registry for easy access
PIPELINES = {
    "minimal": minimal_pipeline,
    "standard": standard_pipeline,
    "quality": quality_pipeline,
}


def get_pipeline(name: str) -> Callable:
    """Get a predefined pipeline by name."""
    if name not in PIPELINES:
        raise ValueError(f"Unknown pipeline: {name}. Available: {list(PIPELINES.keys())}")
    return PIPELINES[name]


def register_pipeline(name: str, pipeline: Callable):
    """Register a custom pipeline for reuse."""
    PIPELINES[name] = pipeline
    logger.info(f"Registered pipeline: {name}")


# Example usage
async def example_usage():
    """Example of using the functional pipeline."""
    
    # Use predefined pipeline
    result = await standard_pipeline("What is RAG?", config={
        "enable_cache": True,
        "expansion_strategies": ["acronym", "semantic"],
        "reranking_strategy": "flashrank",
        "top_k": 10
    })
    
    print(f"Found {len(result.documents)} documents")
    print(f"Cache hit: {result.cache_hit}")
    print(f"Timings: {result.timings}")
    
    # Build custom pipeline
    my_pipeline = build_pipeline(
        expand_query,
        parallel(check_cache, optimize_chromadb_search),
        conditional(
            lambda ctx: not ctx.cache_hit,
            retrieve_documents
        ),
        process_tables,
        rerank_documents,
        analyze_performance
    )
    
    result = await my_pipeline("machine learning", config={
        "expansion_strategies": ["acronym"],
        "enable_hybrid_search": True
    })
    
    # Use custom function chain
    result = await custom_pipeline(
        "deep learning",
        functions=[
            expand_query,
            check_cache,
            retrieve_documents,
            process_tables,
            rerank_documents
        ],
        config={"top_k": 5}
    )
    
    return result


if __name__ == "__main__":
    asyncio.run(example_usage())