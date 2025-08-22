# RAG Module - Functional Pipeline Architecture

## Overview

The RAG (Retrieval-Augmented Generation) module provides intelligent search and question-answering capabilities for the tldw_server application. It uses a **functional pipeline architecture** where composable functions are chained together to create custom retrieval and processing workflows.

## Key Features

- **Functional Pipeline**: Pure functions that compose into pipelines
- **Multi-Database Search**: Query across media, notes, prompts, and character cards
- **Query Expansion**: Automatic query enhancement with synonyms, acronyms, and domain terms
- **Hybrid Search**: Combines keyword (FTS5) and vector similarity search
- **Smart Caching**: Semantic cache with adaptive thresholds
- **Document Reranking**: Multiple strategies for relevance optimization
- **Production Ready**: Optional resilience features (circuit breakers, retries, fallbacks)
- **Performance Monitoring**: Built-in metrics and timing analysis

## Quick Start

```python
from tldw_Server_API.app.core.RAG import standard_pipeline

# Simple usage with standard pipeline
result = await standard_pipeline(
    query="What is machine learning?",
    config={
        "enable_cache": True,
        "expansion_strategies": ["acronym", "synonym"],
        "top_k": 10
    }
)

# Access results
documents = result.documents
timings = result.timings
metadata = result.metadata
```

## Directory Structure

```
RAG/
├── README.md                    # This file
├── __init__.py                 # Module exports
├── exceptions.py               # Custom exceptions
├── rag_audit_logger.py        # Audit logging
├── rag_custom_metrics.py      # Metrics collection
├── rag_service/               # Core implementation
│   ├── functional_pipeline.py # Main pipeline functions
│   ├── database_retrievers.py # Database retrieval
│   ├── query_expansion.py     # Query enhancement
│   ├── semantic_cache.py      # Caching layer
│   ├── advanced_reranking.py  # Document reranking
│   ├── resilience.py          # Fault tolerance
│   ├── performance_monitor.py # Performance tracking
│   ├── config.py              # Configuration
│   ├── types.py               # Type definitions
│   └── ... (additional modules)
├── ARCHIVE/                    # Deprecated implementations
└── DEPRECATION_NOTICE.md      # Migration notes
```

## Available Pipelines

### Pre-built Pipelines

1. **minimal_pipeline** - Fast, basic search
   - Direct retrieval
   - Basic reranking
   - No caching or expansion

2. **standard_pipeline** - Balanced performance
   - Query expansion
   - Caching enabled
   - Hybrid reranking
   - Performance monitoring

3. **quality_pipeline** - Maximum accuracy
   - All expansion strategies
   - ChromaDB optimization
   - Table processing
   - Advanced reranking

4. **enhanced_pipeline** - Advanced features
   - Enhanced chunking
   - Parent document retrieval
   - Chunk type filtering
   - Structure-aware processing

### Custom Pipelines

Build your own pipeline by composing functions:

```python
from tldw_Server_API.app.core.RAG import (
    build_pipeline,
    expand_query,
    check_cache,
    retrieve_documents,
    rerank_documents,
    store_in_cache
)

# Create custom pipeline
my_pipeline = build_pipeline(
    expand_query,
    check_cache,
    retrieve_documents,
    rerank_documents,
    store_in_cache
)

# Use it
context = RAGPipelineContext(
    query="your query",
    original_query="your query",
    config={"enable_cache": True}
)
result = await my_pipeline(context)
```

## Configuration Options

```python
config = {
    # Core settings
    "enable_cache": True,           # Enable semantic caching
    "enable_monitoring": True,       # Performance monitoring
    "enable_resilience": False,      # Fault tolerance (opt-in)
    
    # Retrieval
    "sources": ["media_db", "notes"],  # Data sources to search
    "top_k": 10,                       # Max results
    "use_fts": True,                   # Full-text search
    "use_vector": False,               # Vector search
    
    # Query expansion
    "expansion_strategies": ["acronym", "synonym", "domain", "entity"],
    
    # Reranking
    "reranking_strategy": "hybrid",    # flashrank, cross_encoder, hybrid
    
    # Caching
    "cache_threshold": 0.85,           # Similarity threshold
    "use_adaptive_cache": True,        # Adaptive thresholds
    
    # Resilience (when enabled)
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
    }
}
```

## Pipeline Functions

### Core Functions

- `expand_query()` - Enhance query with variations
- `check_cache()` - Check semantic cache
- `retrieve_documents()` - Fetch from databases
- `optimize_chromadb_search()` - Vector search optimization
- `process_tables()` - Extract and process tables
- `rerank_documents()` - Reorder by relevance
- `store_in_cache()` - Cache results
- `analyze_performance()` - Collect metrics

### Resilience

All functions support optional resilience through configuration:

```python
config = {
    "enable_resilience": True,
    "resilience": {
        "retry": {"enabled": True, "max_attempts": 3},
        "circuit_breaker": {"enabled": True, "failure_threshold": 5}
    }
}
```

When enabled, functions automatically:
- Retry on transient failures
- Circuit break on repeated failures
- Fall back to safe defaults
- Log errors with context

## API Endpoints

The RAG module is exposed through FastAPI endpoints:

- `POST /api/v1/rag/search/simple` - Simple search interface
- `POST /api/v1/rag/search/complex` - Advanced search with full options
- `GET /api/v1/rag/health` - Health check endpoint

See [RAG API Documentation](/Docs/API-related/RAG_API_Documentation.md) for details.

## Testing

```bash
# Run all RAG tests
python -m pytest tests/RAG/ -v

# Run specific test suites
python -m pytest tests/RAG/test_functional_pipeline.py -v
python -m pytest tests/RAG/test_rag_refactored.py -v

# Run with coverage
python -m pytest tests/RAG/ --cov=app.core.RAG --cov-report=html
```

## Performance

Typical pipeline execution times (on standard hardware):
- Minimal pipeline: ~50-100ms
- Standard pipeline: ~200-300ms (with cache miss)
- Standard pipeline: ~20-30ms (with cache hit)
- Quality pipeline: ~500-800ms

## Migration from Old Architecture

If you were using the old object-oriented approach:

```python
# Old way (deprecated)
from app.core.RAG.rag_service.integration import RAGService
service = RAGService(config)
result = await service.search(query)

# New way (functional)
from app.core.RAG import standard_pipeline
result = await standard_pipeline(query, config)
```

## Contributing

When extending the RAG module:

1. Add new functions to `functional_pipeline.py` or create new modules
2. Follow the pattern: async functions that accept and return `RAGPipelineContext`
3. Use the `@timer` decorator for performance tracking
4. Add `@with_resilience` decorator for optional fault tolerance
5. Write tests in `tests/RAG/`
6. Update this README

## Related Documentation

- [RAG Service Implementation](rag_service/README.md)
- [API Documentation](/Docs/API-related/RAG_API_Documentation.md)
- [Developer Guide](/Docs/Development/RAG-Developer-Guide.md)
- [Functional Pipeline Guide](/Docs/Development/RAG-Functional-Pipeline-Guide.md)

## License

Same as tldw_server (AGPLv3)