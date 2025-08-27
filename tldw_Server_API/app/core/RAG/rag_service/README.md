# RAG Service - Functional Pipeline Implementation

A modern, efficient RAG (Retrieval-Augmented Generation) implementation using a functional pipeline architecture for the tldw_server application.

## Overview

This RAG service provides a **functional pipeline architecture** where pure functions are composed into custom retrieval and processing workflows. It replaces the previous object-oriented approach with a simpler, more flexible design that allows callers to build pipelines at call time.

## Features

- **Functional Architecture**: Composable pure functions instead of class hierarchies
- **Multiple Data Sources**: Search across media files, notes, prompts, and character cards
- **Hybrid Search**: Combines keyword (FTS5) and vector search for best results
- **Smart Processing**: Document deduplication, reranking, and context optimization
- **Security & Privacy**: PII detection, content filtering, and access control
- **Batch Processing**: Handle multiple queries concurrently with priority scheduling
- **User Feedback**: Collect relevance feedback to improve search quality
- **Citation Management**: Automatic citation extraction and formatting
- **Flexible Pipelines**: Pre-built pipelines or build your own
- **Production Ready**: Optional resilience features (circuit breakers, retries)
- **Performance**: Built-in caching, metrics, and efficient resource usage

## Architecture

```
rag_service/
├── functional_pipeline.py      # Core pipeline functions and presets
├── database_retrievers.py      # Database retrieval strategies
├── query_expansion.py          # Query enhancement strategies
├── semantic_cache.py           # Semantic caching implementation
├── advanced_cache.py          # Advanced caching strategies
├── advanced_reranking.py       # Document reranking strategies
├── security_filters.py        # PII detection and content filtering
├── batch_processing.py        # Batch query processing
├── feedback_system.py         # User feedback collection
├── citations.py              # Citation generation
├── parent_retrieval.py       # Parent document retrieval
├── generation.py             # Answer generation
├── table_serialization.py      # Table processing functionality
├── performance_monitor.py      # Performance monitoring
├── metrics_collector.py      # Comprehensive metrics
├── resilience.py              # Fault tolerance (circuit breakers, retries)
├── observability.py           # Logging and tracing
├── health_check.py           # Health monitoring
├── config.py                  # Configuration management
├── types.py                   # Type definitions
└── utils.py                   # Helper utilities
```

## Quick Start

### Basic Usage

```python
from tldw_Server_API.app.core.RAG import minimal_pipeline, standard_pipeline, quality_pipeline

# Simple search with minimal pipeline (fastest)
result = await minimal_pipeline("What is machine learning?")

# Standard search with caching and expansion
result = await standard_pipeline(
    "What is ML?",
    config={
        "enable_cache": True,
        "expansion_strategies": ["acronym", "synonym"],
        "top_k": 10
    }
)

# High-quality search with all features
result = await quality_pipeline(
    "machine learning applications",
    config={
        "enable_chromadb": True,
        "process_tables": True,
        "reranking_strategy": "cross_encoder"
    }
)

# Access results
print(f"Found {len(result.documents)} documents")
print(f"Search took {sum(result.timings.values()):.2f} seconds")
```

### Building Custom Pipelines

```python
from tldw_Server_API.app.core.RAG import (
    build_pipeline,
    expand_query,
    check_cache,
    retrieve_documents,
    rerank_documents,
    RAGPipelineContext
)

# Build your custom pipeline
my_pipeline = build_pipeline(
    expand_query,      # Enhance the query
    check_cache,       # Check if cached
    retrieve_documents, # Fetch documents
    rerank_documents   # Reorder by relevance
)

# Create context with configuration
context = RAGPipelineContext(
    query="your search query",
    original_query="your search query",
    config={
        "enable_cache": True,
        "sources": ["media_db", "notes"],
        "top_k": 20
    }
)

# Execute pipeline
result = await my_pipeline(context)
```

## Pipeline Functions

### Core Functions

#### Query Processing
- `expand_query()` - Expand query with synonyms, acronyms, entities
- `check_cache()` - Check semantic cache for similar queries
- `store_in_cache()` - Store results for future queries

#### Retrieval
- `retrieve_documents()` - Fetch from configured databases
- `filter_by_keywords()` - Keyword-based filtering
- `optimize_chromadb_search()` - Optimize vector search

#### Processing
- `process_tables()` - Extract and process tabular data
- `rerank_documents()` - Reorder documents by relevance
- `analyze_performance()` - Collect performance metrics

### Pre-built Pipelines

1. **minimal_pipeline**
   ```python
   # Just retrieval and basic reranking
   # ~50-100ms execution time
   result = await minimal_pipeline(query)
   ```

2. **standard_pipeline**
   ```python
   # Query expansion, caching, retrieval, reranking
   # ~200-300ms (cache miss), ~20-30ms (cache hit)
   result = await standard_pipeline(query, config)
   ```

3. **quality_pipeline**
   ```python
   # All features for maximum accuracy
   # ~500-800ms execution time
   result = await quality_pipeline(query, config)
   ```

4. **enhanced_pipeline**
   ```python
   # Advanced chunking and parent document retrieval
   # ~800-1200ms execution time
   result = await enhanced_pipeline(query, config)
   ```

## Configuration

### Basic Configuration

```python
config = {
    # Data sources
    "databases": {
        "media_db_path": "/path/to/media.db",
        "notes_db_path": "/path/to/notes.db",
    },
    "sources": ["media_db", "notes"],  # Which to search
    
    # Retrieval settings
    "top_k": 10,                       # Max results
    "min_score": 0.0,                  # Min relevance score
    "use_fts": True,                   # Full-text search
    "use_vector": False,                # Vector search
    
    # Query expansion
    "expansion_strategies": ["acronym", "synonym"],
    
    # Caching
    "enable_cache": True,
    "cache_threshold": 0.85,           # Similarity threshold
    
    # Reranking
    "reranking_strategy": "hybrid",    # flashrank, cross_encoder, hybrid
    
    # Performance
    "enable_monitoring": True,
}
```

### Production Configuration with Resilience

```python
config = {
    # Enable resilience features
    "enable_resilience": True,
    
    # Resilience configuration
    "resilience": {
        "retry": {
            "enabled": True,
            "max_attempts": 3,
            "initial_delay": 0.5
        },
        "circuit_breaker": {
            "enabled": True,
            "failure_threshold": 5,
            "timeout": 60
        }
    },
    
    # Other settings...
}
```

## Database Retrievers

The service supports multiple database types through specialized retrievers:

### MediaDBRetriever
Searches media transcripts using SQLite FTS5:
```python
retriever = MediaDBRetriever("/path/to/media.db")
documents = await retriever.retrieve(query, media_type="video")
```

### NotesDBRetriever
Searches user notes:
```python
retriever = NotesDBRetriever("/path/to/notes.db")
documents = await retriever.retrieve(query, notebook_id=123)
```

### CharacterCardsRetriever
Searches character cards and chat history:
```python
retriever = CharacterCardsRetriever("/path/to/characters.db")
documents = await retriever.retrieve(query, include_chats=True)
```

### MultiDatabaseRetriever
Searches across multiple databases:
```python
retriever = MultiDatabaseRetriever({
    "media_db": "/path/to/media.db",
    "notes_db": "/path/to/notes.db"
})
documents = await retriever.retrieve(
    query,
    sources=[DataSource.MEDIA_DB, DataSource.NOTES],
    config=RetrievalConfig(max_results=20)
)
```

## Query Expansion

Multiple strategies for query enhancement:

- **AcronymExpansion**: Expands acronyms (ML → Machine Learning)
- **SynonymExpansion**: Adds synonyms and related terms
- **DomainExpansion**: Adds domain-specific terms
- **EntityExpansion**: Identifies and expands entities

```python
from rag_service.query_expansion import HybridQueryExpansion, AcronymExpansion

expander = HybridQueryExpansion([AcronymExpansion()])
expanded = await expander.expand("What is ML?")
# Returns: ExpandedQuery with variations like "machine learning"
```

## Caching

Semantic caching with adaptive thresholds:

```python
from rag_service.semantic_cache import SemanticCache, AdaptiveCache

# Basic cache
cache = SemanticCache(similarity_threshold=0.85)

# Adaptive cache that adjusts thresholds
cache = AdaptiveCache(initial_threshold=0.85)

# Check for similar queries
cached = await cache.find_similar("machine learning")
if cached:
    documents = await cache.get(cached[0])
```

## Reranking

Multiple reranking strategies for relevance optimization:

```python
from rag_service.advanced_reranking import HybridReranker

reranker = HybridReranker(weights={"flashrank": 0.6, "cross_encoder": 0.4})
reranked = await reranker.rerank(documents, query)
```

## Performance Monitoring

Built-in performance tracking:

```python
from rag_service.performance_monitor import PerformanceMonitor

monitor = PerformanceMonitor()
monitor.record_query(
    query="test query",
    total_duration=0.234,
    component_timings={"retrieval": 0.1, "reranking": 0.05},
    cache_hit=False
)

stats = monitor.get_statistics()
print(f"Average latency: {stats['avg_latency_ms']}ms")
print(f"Cache hit rate: {stats['cache_hit_rate']:.2%}")
```

## Resilience Features

Optional fault tolerance for production environments:

### Circuit Breakers
Prevent cascading failures:
```python
from rag_service.resilience import CircuitBreaker, CircuitBreakerConfig

config = CircuitBreakerConfig(
    failure_threshold=5,
    timeout=60.0
)
breaker = CircuitBreaker("retrieval", config)
```

### Retry Logic
Automatic retries with exponential backoff:
```python
from rag_service.resilience import RetryPolicy, RetryConfig

config = RetryConfig(
    max_attempts=3,
    initial_delay=0.5,
    exponential_base=2.0
)
policy = RetryPolicy(config)
```

## Testing

```bash
# Run all tests
python -m pytest tests/RAG/ -v

# Run specific component tests
python -m pytest tests/RAG/test_functional_pipeline.py -v
python -m pytest tests/RAG/test_database_retrievers.py -v
python -m pytest tests/RAG/test_query_expansion.py -v

# Run with coverage
python -m pytest tests/RAG/ --cov=app.core.RAG.rag_service
```

## Migration from Old Architecture

### From RAGService Class (Old)

```python
# Old way (deprecated)
from rag_service.integration import RAGService

service = RAGService(
    media_db_path=Path("/path/to/media.db"),
    config=config
)
await service.initialize()
result = await service.search(query)
```

### To Functional Pipeline (New)

```python
# New way (functional)
from app.core.RAG import standard_pipeline

result = await standard_pipeline(
    query,
    config={
        "databases": {"media_db_path": "/path/to/media.db"},
        **config
    }
)
```

## Advanced Features

### Security Filtering

Filter out sensitive content and PII:

```python
from rag_service.security_filters import SecurityFilter, SensitivityLevel

filter = SecurityFilter()
# Detect and redact PII
filtered = await filter.filter_documents(
    documents,
    detect_pii=True,
    redact_sensitive=True,
    min_sensitivity=SensitivityLevel.INTERNAL
)
```

### Batch Processing

Process multiple queries efficiently:

```python
from rag_service.batch_processing import BatchProcessor, PriorityLevel

processor = BatchProcessor(
    max_concurrent=10,
    timeout_per_query=5.0
)
results = await processor.process_batch(
    queries=["query1", "query2", "query3"],
    pipeline=standard_pipeline,
    priority=PriorityLevel.HIGH
)
```

### User Feedback Integration

Collect and apply user feedback:

```python
from rag_service.feedback_system import FeedbackCollector, RelevanceScore

collector = FeedbackCollector()
# Record feedback
await collector.record_feedback(
    query_id="abc123",
    document_id="doc456",
    relevance=RelevanceScore.GOOD,
    helpful=True
)

# Use feedback to boost relevance
boosted = await collector.apply_feedback_boost(documents, query)
```

### Citation Generation

Generate properly formatted citations:

```python
from rag_service.citations import CitationGenerator

generator = CitationGenerator()
citations = await generator.generate_citations(
    documents,
    style="apa",  # or "mla", "chicago"
    include_metadata=True,
    include_page_numbers=True
)
```

### Health Monitoring

Monitor RAG service health:

```python
from rag_service.health_check import HealthChecker

checker = HealthChecker()
health = await checker.check_health()
print(f"Status: {health.status}")
print(f"Latency: {health.avg_latency_ms}ms")
print(f"Cache hit rate: {health.cache_hit_rate:.2%}")
```

## Extending the Service

To add new functionality:

1. Create a new async function that accepts and returns `RAGPipelineContext`
2. Add timing with `@timer` decorator
3. Add resilience with `@with_resilience` decorator (optional)
4. Compose into pipelines using `build_pipeline()`

Example:
```python
@timer("custom_processing")
@with_resilience("custom_processing", fallback_func)
async def custom_processing(context: RAGPipelineContext) -> RAGPipelineContext:
    # Your custom logic here
    context.documents = process_documents(context.documents)
    return context

# Use in pipeline
pipeline = build_pipeline(
    retrieve_documents,
    custom_processing,
    rerank_documents
)
```

## Troubleshooting

### Common Issues

1. **Import errors**
   - Ensure you're importing from `app.core.RAG` not old paths
   - Check that all dependencies are installed

2. **Poor search results**
   - Try different expansion strategies
   - Adjust reranking strategy
   - Increase `top_k` value

3. **Slow performance**
   - Enable caching
   - Use minimal pipeline for simple queries
   - Check database indexes

4. **Resilience not working**
   - Ensure `enable_resilience: True` in config
   - Check resilience configuration parameters
   - Review logs for circuit breaker status

## License

Same as tldw_server (AGPLv3)