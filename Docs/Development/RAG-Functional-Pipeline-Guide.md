# RAG Functional Pipeline Guide

## Overview

The RAG module uses a **functional pipeline architecture** where pure, composable functions are chained together to create flexible retrieval and processing workflows. This approach replaces the previous object-oriented design with a simpler, more maintainable pattern.

## Core Concepts

### What is a Functional Pipeline?

A functional pipeline is a series of pure functions where:
- Each function accepts a context object as input
- Each function returns a modified context object as output
- Functions can be composed in any order
- Side effects are minimized and well-contained
- Functions are independently testable

### RAGPipelineContext

The central data structure that flows through all pipeline functions:

```python
@dataclass
class RAGPipelineContext:
    query: str                    # Current (possibly expanded) query
    original_query: str            # Original user query
    documents: List[Document] = field(default_factory=list)
    expanded_queries: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    config: Dict[str, Any] = field(default_factory=dict)
    timings: Dict[str, float] = field(default_factory=dict)
    cache_key: Optional[str] = None
    error: Optional[str] = None
```

## Pre-built Pipelines

### 1. Minimal Pipeline

**Use Case**: Fast, simple lookups where speed is critical

```python
from tldw_Server_API.app.core.RAG import minimal_pipeline

result = await minimal_pipeline("What is Python?")
```

**Flow**: `Retrieve → Rerank`

**Performance**: ~50-100ms

### 2. Standard Pipeline

**Use Case**: Balanced performance and accuracy for most queries

```python
from tldw_Server_API.app.core.RAG import standard_pipeline

result = await standard_pipeline(
    "What is ML?",
    config={
        "enable_cache": True,
        "expansion_strategies": ["acronym", "synonym"],
        "top_k": 10
    }
)
```

**Flow**: `Expand → Cache Check → Retrieve → Rerank → Cache Store`

**Performance**: ~200-300ms (cache miss), ~20-30ms (cache hit)

### 3. Quality Pipeline

**Use Case**: Complex queries requiring maximum accuracy

```python
from tldw_Server_API.app.core.RAG import quality_pipeline

result = await quality_pipeline(
    "machine learning vs deep learning",
    config={
        "enable_chromadb": True,
        "process_tables": True,
        "reranking_strategy": "cross_encoder"
    }
)
```

**Flow**: `Expand → Cache → Retrieve → ChromaDB → Tables → Rerank → Performance → Store`

**Performance**: ~500-800ms

### 4. Enhanced Pipeline

**Use Case**: Advanced document processing with chunking and parent retrieval

```python
from tldw_Server_API.app.core.RAG import enhanced_pipeline

result = await enhanced_pipeline(
    "explain transformers architecture",
    config={
        "enable_parent_retrieval": True,
        "chunk_type_filter": ["paragraph", "section"],
        "enhanced_chunking": True
    }
)
```

**Flow**: All quality features + enhanced chunking + parent document retrieval

**Performance**: ~800-1200ms

## Building Custom Pipelines

### Basic Custom Pipeline

```python
from tldw_Server_API.app.core.RAG import (
    build_pipeline,
    expand_query,
    retrieve_documents,
    rerank_documents
)

# Create a simple custom pipeline
my_pipeline = build_pipeline(
    expand_query,
    retrieve_documents,
    rerank_documents
)

# Execute
context = RAGPipelineContext(
    query="your query",
    original_query="your query",
    config={"top_k": 5}
)
result = await my_pipeline(context)
```

### Advanced Custom Pipeline

```python
from tldw_Server_API.app.core.RAG import (
    build_pipeline,
    expand_query,
    check_cache,
    retrieve_documents,
    process_tables,
    rerank_documents,
    store_in_cache,
    analyze_performance
)

# Build a comprehensive pipeline
advanced_pipeline = build_pipeline(
    expand_query,           # Enhance the query
    check_cache,           # Check if we've seen this before
    retrieve_documents,    # Fetch from databases
    process_tables,        # Extract tabular data
    rerank_documents,      # Reorder by relevance
    store_in_cache,        # Save for future
    analyze_performance    # Collect metrics
)
```

## Pipeline Functions Reference

### Query Processing Functions

#### expand_query
Expands the query with synonyms, acronyms, and related terms.

```python
async def expand_query(
    context: RAGPipelineContext,
    strategies: List[str] = None
) -> RAGPipelineContext
```

**Strategies**:
- `"acronym"` - Expand acronyms (ML → Machine Learning)
- `"synonym"` - Add synonyms and related terms
- `"domain"` - Add domain-specific terms
- `"entity"` - Identify and expand entities

#### check_cache
Checks semantic cache for similar queries.

```python
async def check_cache(
    context: RAGPipelineContext
) -> RAGPipelineContext
```

Returns cached results if similarity > threshold (default 0.85).

### Retrieval Functions

#### retrieve_documents
Fetches documents from configured databases.

```python
async def retrieve_documents(
    context: RAGPipelineContext
) -> RAGPipelineContext
```

**Supported Sources**:
- `media_db` - Media transcripts and content
- `notes` - User notes
- `prompts` - Prompt library
- `character_cards` - Character definitions and chats

#### optimize_chromadb_search
Optimizes vector search for large collections.

```python
async def optimize_chromadb_search(
    context: RAGPipelineContext
) -> RAGPipelineContext
```

Uses hierarchical search and metadata filtering for 100k+ documents.

### Processing Functions

#### process_tables
Extracts and processes tabular data from documents.

```python
async def process_tables(
    context: RAGPipelineContext
) -> RAGPipelineContext
```

Serializes tables to markdown for better comprehension.

#### rerank_documents
Reorders documents by relevance using various strategies.

```python
async def rerank_documents(
    context: RAGPipelineContext
) -> RAGPipelineContext
```

**Strategies**:
- `"flashrank"` - Fast neural reranking
- `"cross_encoder"` - High-quality cross-encoder model
- `"hybrid"` - Weighted combination of strategies

### Utility Functions

#### store_in_cache
Saves results to semantic cache.

```python
async def store_in_cache(
    context: RAGPipelineContext
) -> RAGPipelineContext
```

#### analyze_performance
Collects performance metrics and timing data.

```python
async def analyze_performance(
    context: RAGPipelineContext
) -> RAGPipelineContext
```

## Configuration

### Basic Configuration

```python
config = {
    # Pipeline selection
    "pipeline": "standard",  # minimal, standard, quality, enhanced, custom
    
    # Data sources
    "sources": ["media_db", "notes"],
    "databases": {
        "media_db_path": "/path/to/media.db",
        "notes_db_path": "/path/to/notes.db"
    },
    
    # Retrieval
    "top_k": 10,
    "min_score": 0.0,
    "use_fts": True,
    "use_vector": False,
    
    # Query expansion
    "expansion_strategies": ["acronym", "synonym"],
    
    # Caching
    "enable_cache": True,
    "cache_threshold": 0.85,
    
    # Reranking
    "reranking_strategy": "hybrid"
}
```

### Production Configuration with Resilience

```python
config = {
    "pipeline": "quality",
    "enable_resilience": True,  # Enable fault tolerance
    
    "resilience": {
        "retry": {
            "enabled": True,
            "max_attempts": 3,
            "initial_delay": 0.5,
            "exponential_base": 2.0
        },
        "circuit_breaker": {
            "enabled": True,
            "failure_threshold": 5,
            "timeout": 60
        }
    },
    
    # Performance monitoring
    "enable_monitoring": True,
    "log_performance": True,
    
    # Advanced features
    "enable_chromadb": True,
    "process_tables": True,
    "enhanced_chunking": True
}
```

## Creating Custom Pipeline Functions

### Basic Template

```python
from tldw_Server_API.app.core.RAG.rag_service.functional_pipeline import (
    RAGPipelineContext,
    timer
)

@timer("my_custom_function")
async def my_custom_function(
    context: RAGPipelineContext
) -> RAGPipelineContext:
    """
    Custom processing function.
    
    Args:
        context: Pipeline context
        
    Returns:
        Modified context
    """
    # Your logic here
    for doc in context.documents:
        # Process each document
        doc.metadata["custom_field"] = compute_value(doc)
    
    return context
```

### With Resilience

```python
from tldw_Server_API.app.core.RAG.rag_service.functional_pipeline import (
    RAGPipelineContext,
    timer,
    with_resilience
)

# Define fallback
async def my_function_fallback(
    context: RAGPipelineContext
) -> RAGPipelineContext:
    logger.warning("Custom function failed, using fallback")
    return context

# Main function with resilience
@timer("my_resilient_function")
@with_resilience("my_resilient_function", my_function_fallback)
async def my_resilient_function(
    context: RAGPipelineContext
) -> RAGPipelineContext:
    """
    Resilient custom function with automatic retry and fallback.
    """
    # Your logic that might fail
    result = await external_service_call()
    context.metadata["external_data"] = result
    return context
```

## Conditional and Parallel Execution

### Conditional Pipeline

```python
async def conditional_pipeline(query: str, config: dict):
    context = RAGPipelineContext(
        query=query,
        original_query=query,
        config=config
    )
    
    # Always expand query
    context = await expand_query(context)
    
    # Conditionally use cache
    if config.get("enable_cache", True):
        context = await check_cache(context)
        if context.documents:  # Cache hit
            return context
    
    # Retrieve documents
    context = await retrieve_documents(context)
    
    # Conditionally process tables
    if config.get("process_tables", False):
        context = await process_tables(context)
    
    # Always rerank
    context = await rerank_documents(context)
    
    # Store in cache if enabled
    if config.get("enable_cache", True):
        context = await store_in_cache(context)
    
    return context
```

### Parallel Execution

```python
import asyncio

async def parallel_retrieval_pipeline(query: str, config: dict):
    context = RAGPipelineContext(
        query=query,
        original_query=query,
        config=config
    )
    
    # Expand query first
    context = await expand_query(context)
    
    # Parallel retrieval from multiple sources
    async def retrieve_from_source(source: str):
        source_context = RAGPipelineContext(
            query=context.query,
            original_query=context.original_query,
            config={**config, "sources": [source]}
        )
        return await retrieve_documents(source_context)
    
    # Execute retrievals in parallel
    results = await asyncio.gather(
        retrieve_from_source("media_db"),
        retrieve_from_source("notes"),
        retrieve_from_source("prompts")
    )
    
    # Merge results
    all_documents = []
    for result in results:
        all_documents.extend(result.documents)
    
    context.documents = all_documents
    
    # Continue with pipeline
    context = await rerank_documents(context)
    return context
```

## Performance Optimization

### 1. Pipeline Selection

Choose the appropriate pipeline for your use case:

```python
# For simple, fast lookups
if query_is_simple(query):
    result = await minimal_pipeline(query)

# For standard queries with caching
elif enable_caching:
    result = await standard_pipeline(query, config)

# For complex queries needing accuracy
else:
    result = await quality_pipeline(query, config)
```

### 2. Caching Strategy

```python
config = {
    "enable_cache": True,
    "cache_threshold": 0.85,  # Similarity threshold
    "use_adaptive_cache": True,  # Adjust threshold based on hit rate
    "cache_ttl": 3600  # Cache for 1 hour
}
```

### 3. Batch Processing

```python
async def batch_search(queries: List[str], config: dict):
    """Process multiple queries efficiently."""
    # Process in parallel with concurrency limit
    semaphore = asyncio.Semaphore(5)
    
    async def search_with_limit(query):
        async with semaphore:
            return await standard_pipeline(query, config)
    
    results = await asyncio.gather(
        *[search_with_limit(q) for q in queries]
    )
    return results
```

## Error Handling

### Basic Error Handling

```python
try:
    result = await standard_pipeline(query, config)
except Exception as e:
    logger.error(f"Pipeline failed: {e}")
    # Fallback to simpler pipeline
    result = await minimal_pipeline(query)
```

### With Resilience Features

```python
config = {
    "enable_resilience": True,
    "resilience": {
        "retry": {"enabled": True, "max_attempts": 3},
        "circuit_breaker": {"enabled": True, "failure_threshold": 5}
    }
}

# Functions will automatically retry and circuit break
result = await standard_pipeline(query, config)
```

## Testing Custom Pipelines

```python
import pytest
from tldw_Server_API.app.core.RAG import (
    RAGPipelineContext,
    build_pipeline
)

@pytest.mark.asyncio
async def test_custom_pipeline():
    # Build custom pipeline
    pipeline = build_pipeline(
        expand_query,
        retrieve_documents,
        custom_processing,
        rerank_documents
    )
    
    # Create test context
    context = RAGPipelineContext(
        query="test query",
        original_query="test query",
        config={
            "sources": ["media_db"],
            "top_k": 5
        }
    )
    
    # Execute pipeline
    result = await pipeline(context)
    
    # Assertions
    assert len(result.documents) <= 5
    assert result.expanded_queries
    assert "custom_processing" in result.timings
```

## Best Practices

### 1. Keep Functions Pure

```python
# Good: Pure function
async def add_scores(context: RAGPipelineContext) -> RAGPipelineContext:
    for doc in context.documents:
        doc.score = calculate_score(doc, context.query)
    return context

# Bad: Side effects
async def add_scores_bad(context: RAGPipelineContext) -> RAGPipelineContext:
    global score_cache  # Don't use global state
    for doc in context.documents:
        doc.score = calculate_score(doc, context.query)
        score_cache[doc.id] = doc.score  # Side effect
    return context
```

### 2. Use Decorators Consistently

```python
@timer("custom_function")  # Always add timing
@with_resilience("custom_function", fallback)  # Add resilience if needed
async def custom_function(context: RAGPipelineContext) -> RAGPipelineContext:
    # Implementation
    pass
```

### 3. Document Pipeline Behavior

```python
async def my_pipeline(query: str, config: dict):
    """
    Custom pipeline for domain-specific search.
    
    Pipeline flow:
    1. Expand query with domain terms
    2. Retrieve from specialized database
    3. Apply custom scoring
    4. Rerank by relevance
    
    Args:
        query: User search query
        config: Pipeline configuration
        
    Returns:
        RAGPipelineContext with results
    """
    # Implementation
    pass
```

### 4. Handle Context Properly

```python
# Always preserve original query
context.original_query = original_query

# Add metadata, don't replace
context.metadata.update({"new_field": value})

# Check for existing documents
if not context.documents:
    logger.warning("No documents found")
    return context
```

## Migration from Object-Oriented Design

### Old Way (Deprecated)

```python
# OLD: Object-oriented approach
from app.core.RAG.rag_service.integration import RAGService

service = RAGService(config)
await service.initialize()
result = await service.search(query)
await service.cleanup()
```

### New Way (Functional)

```python
# NEW: Functional pipeline
from app.core.RAG import standard_pipeline

result = await standard_pipeline(query, config)
# No initialization or cleanup needed
```

## Troubleshooting

### Pipeline Not Working

1. Check imports are from correct module:
```python
from tldw_Server_API.app.core.RAG import standard_pipeline  # Correct
# NOT from old paths
```

2. Verify configuration:
```python
# Print config to debug
logger.debug(f"Pipeline config: {config}")
result = await standard_pipeline(query, config)
```

3. Check function composition:
```python
# Ensure functions are in correct order
pipeline = build_pipeline(
    expand_query,      # Expand first
    retrieve_documents, # Then retrieve
    rerank_documents   # Finally rerank
)
```

### Performance Issues

1. Enable monitoring:
```python
config["enable_monitoring"] = True
result = await standard_pipeline(query, config)
print(f"Timings: {result.timings}")
```

2. Use appropriate pipeline:
```python
# Don't use quality pipeline for simple lookups
if is_simple_query(query):
    result = await minimal_pipeline(query)
```

3. Check cache configuration:
```python
config = {
    "enable_cache": True,
    "cache_threshold": 0.85  # Lower for more cache hits
}
```

## Summary

The functional pipeline architecture provides:
- **Flexibility**: Compose functions in any order
- **Simplicity**: Pure functions with clear inputs/outputs
- **Testability**: Each function independently testable
- **Performance**: Optional caching and parallel execution
- **Resilience**: Optional fault tolerance features
- **Maintainability**: Easy to understand and extend

Start with pre-built pipelines and create custom ones as needed. The functional approach makes it easy to experiment with different configurations and optimize for your specific use case.