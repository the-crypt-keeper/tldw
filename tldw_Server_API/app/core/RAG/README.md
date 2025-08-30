# RAG Module - Functional Pipeline Architecture

## Overview

The RAG (Retrieval-Augmented Generation) module provides intelligent search and question-answering capabilities for the tldw_server application. It uses a **functional pipeline architecture** where composable functions are chained together to create custom retrieval and processing workflows.

⚠️ **Important**: This documentation reflects the intended architecture. For actual implementation status, see [IMPLEMENTATION_STATUS.md](IMPLEMENTATION_STATUS.md). Approximately 40% of documented features are implemented but not yet connected to API endpoints.

## Currently Available Features

### ✅ Fully Functional
- **Functional Pipeline**: Pure functions that compose into pipelines
- **Multi-Database Search**: Query across media, notes, ~~prompts~~, and character cards
- **Query Expansion**: Automatic query enhancement with synonyms, acronyms, and domain terms
- **Hybrid Search**: Combines keyword (FTS5) and vector similarity search
- **Smart Caching**: Semantic cache with adaptive thresholds
- **Document Reranking**: Multiple strategies for relevance optimization
- **Production Ready**: Optional resilience features (circuit breakers, retries, fallbacks)
- **Performance Monitoring**: Built-in metrics and timing analysis

### ❌ Implemented but Not Connected (Code exists but not accessible via API)
- **Security Features**: PII detection, content filtering, and access control
- **Batch Processing**: Efficient handling of multiple queries simultaneously
- **User Feedback**: Collect and analyze user feedback to improve search quality
- **Citation Generation**: Automatic citation extraction from search results
- **Answer Generation**: Generate answers from retrieved context
- **Observability**: System tracing and detailed metrics

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
├── IMPLEMENTATION_STATUS.md     # Actual feature availability
├── DEPRECATION_NOTICE.md       # Migration information
├── __init__.py                 # Module exports
├── exceptions.py               # Custom exceptions
├── rag_audit_logger.py        # Audit logging (not used)
├── rag_custom_metrics.py      # Metrics collection
├── rag_service/               # Core implementation
│   ├── functional_pipeline.py # Main pipeline functions
│   ├── database_retrievers.py # Database retrieval
│   ├── query_expansion.py     # Query enhancement
│   ├── semantic_cache.py      # Caching layer
│   ├── advanced_cache.py      # Advanced caching strategies
│   ├── advanced_reranking.py  # Document reranking
│   ├── resilience.py          # Fault tolerance
│   ├── performance_monitor.py # Performance tracking
│   ├── metrics_collector.py   # Comprehensive metrics
│   ├── security_filters.py   # PII detection & content filtering
│   ├── batch_processing.py   # Batch query handling
│   ├── feedback_system.py    # User feedback collection
│   ├── citations.py          # Citation generation
│   ├── parent_retrieval.py   # Parent document retrieval
│   ├── generation.py         # Answer generation
│   ├── health_check.py      # Health monitoring
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
- `filter_by_keywords()` - Filter documents by keywords
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

### Currently Active
- `POST /api/v1/rag/search/simple` - Simple search interface with basic parameters
- `POST /api/v1/rag/search/complex` - Advanced search with configuration options
- `GET /api/v1/rag/health` - Basic health check endpoint
- `GET /api/v1/rag/pipelines` - List available pipeline presets (minimal, standard, quality, enhanced)
- `GET /api/v1/rag/capabilities` - Get service capabilities (static response)

### Not Yet Implemented
- Citation generation endpoint
- User feedback endpoint
- Security filter configuration
- Batch processing endpoint
- Observability metrics endpoint

⚠️ Note: Documentation may reference endpoints that don't exist. Check IMPLEMENTATION_STATUS.md for current state.

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

## Advanced Features

### ❌ Security & Privacy (NOT YET CONNECTED)

**Status**: Code exists but is not integrated into the pipeline or API.

The RAG module includes security features code that is not yet accessible:

```python
# This code exists but CANNOT be used via API currently
from tldw_Server_API.app.core.RAG.rag_service.security_filters import SecurityFilter

# The following would work if integrated:
security_filter = SecurityFilter()
filtered_docs = await security_filter.filter_documents(
    documents,
    detect_pii=True,
    redact_sensitive=True
)
```

Planned Features:
- PII detection (emails, SSNs, credit cards, etc.)
- Content filtering based on sensitivity levels
- Access control enforcement
- Audit logging for compliance

### ❌ Batch Processing (NOT YET CONNECTED)

**Status**: Code exists in batch_processing.py but is not accessible via API.

```python
# This code exists but CANNOT be used currently
from tldw_Server_API.app.core.RAG.rag_service.batch_processing import BatchProcessor

# Would work if integrated:
processor = BatchProcessor(max_concurrent=5)
batch_results = await processor.process_batch(
    queries=["query1", "query2", "query3"],
    pipeline=standard_pipeline,
    priority=PriorityLevel.HIGH
)
```

Planned Features:
- Concurrent query processing
- Priority-based scheduling
- Resource management
- Partial failure handling

### ❌ Feedback System (NOT YET CONNECTED)

**Status**: Code exists in feedback_system.py but no API endpoint or pipeline integration.

```python
# This code exists but CANNOT be used currently
from tldw_Server_API.app.core.RAG.rag_service.feedback_system import FeedbackCollector

# Would work if integrated:
collector = FeedbackCollector()
await collector.record_feedback(
    query_id="...",
    relevance_score=4,
    helpful=True,
    user_id="..."
)
```

### ❌ Citation Generation (NOT YET CONNECTED)

**Status**: Code exists in citations.py but is not integrated into any pipeline.

```python
# This code exists but CANNOT be used currently
from tldw_Server_API.app.core.RAG.rag_service.citations import CitationGenerator

# Would work if integrated:
generator = CitationGenerator()
citations = await generator.generate_citations(
    documents,
    style="apa",  # or "mla", "chicago"
    include_metadata=True
)
```

## Performance

Typical pipeline execution times (on standard hardware):
- Minimal pipeline: ~50-100ms
- Standard pipeline: ~200-300ms (with cache miss)
- Standard pipeline: ~20-30ms (with cache hit)
- Quality pipeline: ~500-800ms
- ~~Batch processing: ~100ms per query (concurrent)~~ (Not yet connected)

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