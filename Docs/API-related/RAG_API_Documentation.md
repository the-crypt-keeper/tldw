# RAG API Documentation

**Version**: 3.0  
**Last Updated**: 2025-08-19  
**Status**: Production Ready

## Table of Contents

1. [Overview](#overview)
2. [Architecture](#architecture)
3. [Authentication](#authentication)
4. [Endpoints](#endpoints)
   - [Simple Search](#simple-search)
   - [Complex Search](#complex-search)
   - [Health Check](#health-check)
5. [Data Models](#data-models)
6. [Configuration](#configuration)
7. [Error Handling](#error-handling)
8. [Performance](#performance)
9. [Examples](#examples)
10. [Migration Guide](#migration-guide)

## Overview

The RAG (Retrieval-Augmented Generation) API provides powerful search and AI-powered question-answering capabilities across your indexed content. It uses a **functional pipeline architecture** where composable functions are chained together to create custom retrieval workflows.

### Key Features

- **Functional Pipeline**: Composable pure functions for flexible workflows
- **Hybrid Search**: Combines keyword (FTS5) and semantic search for optimal results
- **Multi-Database Support**: Search across media, notes, prompts, and character cards
- **Query Expansion**: Automatic enhancement with synonyms, acronyms, and domain terms
- **Smart Caching**: Semantic cache with adaptive thresholds
- **Document Reranking**: Multiple strategies for relevance optimization
- **Resilience**: Optional circuit breakers and retry logic for production
- **Performance Monitoring**: Built-in metrics and timing analysis

### Base URL

```
http://localhost:8000/api/v1/rag
```

## Architecture

The RAG API uses a functional pipeline architecture:

```
Query → [Expand] → [Cache Check] → [Retrieve] → [Rerank] → [Cache Store] → Response
           ↓           ↓              ↓            ↓            ↓
      (optional)  (optional)    (required)   (optional)   (optional)
```

Pre-built pipelines:
- **minimal**: Retrieve → Rerank (fastest)
- **standard**: Expand → Cache → Retrieve → Rerank → Store (balanced)
- **quality**: All features enabled (most accurate)

## Authentication

In single-user mode (default), use the API key from config:

```http
X-API-Key: default-secret-key-for-single-user
```

In multi-user mode, use JWT Bearer tokens:

```http
Authorization: Bearer <your-jwt-token>
```

## Endpoints

### Simple Search

**POST** `/api/v1/rag/search/simple`

Simplified search interface with essential parameters.

#### Request Body

```json
{
  "query": "What is machine learning?",
  "databases": ["media", "notes"],
  "max_context_size": 4000,
  "top_k": 10,
  "enable_reranking": true,
  "enable_citations": false,
  "keywords": ["python", "tensorflow"]
}
```

#### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| query | string | required | Search query (1-1000 chars) |
| databases | array | ["media"] | Databases to search: media, notes, characters, chats |
| max_context_size | integer | 4000 | Maximum total size of returned content |
| top_k | integer | 10 | Number of top results (1-100) |
| enable_reranking | boolean | true | Enable document reranking |
| enable_citations | boolean | false | Include source citations |
| keywords | array | null | Filter results by keywords |

#### Response

```json
{
  "results": [
    {
      "content": "Machine learning is a subset of artificial intelligence...",
      "source": "media",
      "metadata": {
        "title": "Introduction to ML",
        "media_id": 123,
        "timestamp": "2024-01-15T10:30:00Z"
      },
      "score": 0.95,
      "citation": "Introduction to ML, timestamp: 10:30"
    }
  ],
  "total_results": 42,
  "processing_time": 0.234,
  "cache_hit": false,
  "query_expanded": ["machine learning", "ML", "artificial intelligence"]
}
```

### Complex Search

**POST** `/api/v1/rag/search/complex`

Advanced search with full configuration options.

#### Request Body

```json
{
  "query": "machine learning applications",
  "config": {
    "pipeline": "quality",
    "databases": {
      "media_db_path": "/path/to/media.db",
      "notes_db_path": "/path/to/notes.db"
    },
    "sources": ["media_db", "notes"],
    "retrieval": {
      "top_k": 20,
      "min_score": 0.5,
      "use_fts": true,
      "use_vector": true,
      "hybrid_alpha": 0.7
    },
    "expansion": {
      "enabled": true,
      "strategies": ["acronym", "synonym", "domain", "entity"],
      "max_expansions": 5
    },
    "caching": {
      "enabled": true,
      "threshold": 0.85,
      "adaptive": true,
      "ttl": 3600
    },
    "reranking": {
      "enabled": true,
      "strategy": "hybrid",
      "top_k": 10,
      "diversity": 0.3
    },
    "processing": {
      "chunk_size": 512,
      "overlap": 50,
      "process_tables": true,
      "extract_metadata": true
    },
    "resilience": {
      "enable_resilience": true,
      "retry": {
        "enabled": true,
        "max_attempts": 3,
        "initial_delay": 0.5
      },
      "circuit_breaker": {
        "enabled": true,
        "failure_threshold": 5,
        "timeout": 60
      }
    },
    "monitoring": {
      "enable_monitoring": true,
      "enable_tracing": false,
      "log_level": "INFO"
    }
  }
}
```

#### Configuration Options

##### Pipeline Selection
- `minimal` - Basic retrieval and reranking
- `standard` - Includes caching and query expansion
- `quality` - All features enabled
- `enhanced` - Advanced chunking and parent retrieval
- `custom` - Build your own (specify functions)

##### Custom Pipeline
```json
{
  "pipeline": "custom",
  "custom_functions": [
    "expand_query",
    "check_cache",
    "retrieve_documents",
    "process_tables",
    "rerank_documents",
    "store_in_cache"
  ]
}
```

#### Response

```json
{
  "results": [
    {
      "id": "doc_123",
      "content": "Full document content...",
      "source": "media_db",
      "metadata": {
        "title": "ML Applications",
        "author": "John Doe",
        "date": "2024-01-15",
        "media_type": "video",
        "duration": 3600,
        "tags": ["ml", "ai", "tutorial"]
      },
      "score": 0.92,
      "relevance_scores": {
        "bm25": 0.88,
        "vector": 0.94,
        "rerank": 0.95
      },
      "chunk_info": {
        "chunk_id": 5,
        "total_chunks": 10,
        "chunk_type": "paragraph"
      }
    }
  ],
  "total_results": 15,
  "metadata": {
    "pipeline_used": "quality",
    "cache_hit": false,
    "query_expanded": true,
    "expansion_count": 3,
    "databases_searched": ["media_db", "notes"],
    "reranking_applied": true,
    "processing_time": 0.456,
    "component_timings": {
      "query_expansion": 0.023,
      "cache_lookup": 0.012,
      "retrieval": 0.234,
      "reranking": 0.089,
      "total": 0.456
    }
  },
  "debug_info": {
    "original_query": "machine learning applications",
    "expanded_queries": [
      "machine learning applications",
      "ML apps",
      "artificial intelligence applications"
    ],
    "search_stats": {
      "documents_retrieved": 50,
      "documents_after_filtering": 30,
      "documents_after_reranking": 15
    }
  }
}
```

### Health Check

**GET** `/api/v1/rag/health`

Check the health status of the RAG service.

#### Response

```json
{
  "status": "healthy",
  "components": {
    "database": "healthy",
    "cache": "healthy",
    "vector_store": "healthy",
    "llm_connection": "healthy"
  },
  "metrics": {
    "queries_processed": 1234,
    "average_latency_ms": 234,
    "cache_hit_rate": 0.45,
    "error_rate": 0.002
  },
  "version": "3.0.0",
  "uptime_seconds": 3600
}
```

## Data Models

### SimpleSearchRequest
```python
{
  "query": str,                    # Required, 1-1000 chars
  "databases": List[str],          # Default: ["media"]
  "max_context_size": int,         # Default: 4000
  "top_k": int,                    # Default: 10, range: 1-100
  "enable_reranking": bool,        # Default: true
  "enable_citations": bool,        # Default: false
  "keywords": Optional[List[str]]  # Optional keyword filters
}
```

### ComplexSearchRequest
```python
{
  "query": str,                    # Required
  "config": Dict[str, Any]         # Full configuration object
}
```

### SearchResult
```python
{
  "id": str,                       # Document ID
  "content": str,                  # Document content
  "source": str,                   # Source database
  "metadata": Dict[str, Any],      # Document metadata
  "score": float,                  # Relevance score (0-1)
  "citation": Optional[str]        # Citation if enabled
}
```

## Configuration

### Environment Variables

```bash
# API Configuration
TLDW_API_HOST=0.0.0.0
TLDW_API_PORT=8000

# Database Paths
MEDIA_DB_PATH=/path/to/media.db
NOTES_DB_PATH=/path/to/notes.db

# Cache Settings
ENABLE_CACHE=true
CACHE_TTL=3600
CACHE_SIZE=1000

# Performance
MAX_WORKERS=4
BATCH_SIZE=32

# Resilience
ENABLE_RESILIENCE=true
CIRCUIT_BREAKER_THRESHOLD=5
RETRY_MAX_ATTEMPTS=3
```

### Configuration File (config.toml)

```toml
[rag]
default_pipeline = "standard"
enable_monitoring = true

[rag.retrieval]
default_top_k = 10
min_score = 0.0
use_fts = true
use_vector = false

[rag.expansion]
enabled = true
strategies = ["acronym", "synonym"]
max_expansions = 5

[rag.cache]
enabled = true
threshold = 0.85
adaptive = true
ttl = 3600

[rag.reranking]
enabled = true
strategy = "hybrid"
diversity = 0.3

[rag.resilience]
enabled = false
retry_attempts = 3
circuit_breaker_threshold = 5
```

## Error Handling

### HTTP Status Codes

| Code | Description |
|------|-------------|
| 200 | Success |
| 400 | Bad Request - Invalid parameters |
| 401 | Unauthorized - Invalid authentication |
| 404 | Not Found - Resource not found |
| 429 | Too Many Requests - Rate limit exceeded |
| 500 | Internal Server Error |
| 503 | Service Unavailable - Circuit breaker open |

### Error Response Format

```json
{
  "error": {
    "code": "INVALID_QUERY",
    "message": "Query must be between 1 and 1000 characters",
    "details": {
      "field": "query",
      "provided_length": 1500,
      "max_length": 1000
    }
  },
  "request_id": "req_123456",
  "timestamp": "2024-01-15T10:30:00Z"
}
```

### Common Error Codes

- `INVALID_QUERY` - Query validation failed
- `DATABASE_ERROR` - Database connection or query failed
- `CACHE_ERROR` - Cache operation failed
- `RERANKING_ERROR` - Reranking failed
- `CIRCUIT_BREAKER_OPEN` - Service temporarily unavailable
- `RATE_LIMIT_EXCEEDED` - Too many requests

## Performance

### Typical Response Times

| Pipeline | Cache Hit | Cache Miss |
|----------|-----------|------------|
| minimal | 20-30ms | 50-100ms |
| standard | 20-30ms | 200-300ms |
| quality | 30-40ms | 500-800ms |
| enhanced | 40-50ms | 800-1200ms |

### Optimization Tips

1. **Enable Caching**: Dramatically improves response time for repeated queries
2. **Use Minimal Pipeline**: For simple lookups where speed is critical
3. **Batch Requests**: Use batch endpoints for multiple queries
4. **Configure Limits**: Set appropriate `top_k` values
5. **Enable Monitoring**: Track performance metrics

## Examples

### Python Client

```python
import httpx
import asyncio

async def search(query: str):
    async with httpx.AsyncClient() as client:
        response = await client.post(
            "http://localhost:8000/api/v1/rag/search/simple",
            json={"query": query},
            headers={"X-API-Key": "your-api-key"}
        )
        return response.json()

# Simple search
results = asyncio.run(search("What is RAG?"))

# Complex search with custom config
async def complex_search(query: str):
    config = {
        "pipeline": "quality",
        "expansion": {"strategies": ["acronym", "synonym"]},
        "reranking": {"strategy": "cross_encoder"}
    }
    
    async with httpx.AsyncClient() as client:
        response = await client.post(
            "http://localhost:8000/api/v1/rag/search/complex",
            json={"query": query, "config": config},
            headers={"X-API-Key": "your-api-key"}
        )
        return response.json()
```

### cURL Examples

```bash
# Simple search
curl -X POST http://localhost:8000/api/v1/rag/search/simple \
  -H "Content-Type: application/json" \
  -H "X-API-Key: your-api-key" \
  -d '{"query": "machine learning"}'

# Complex search with quality pipeline
curl -X POST http://localhost:8000/api/v1/rag/search/complex \
  -H "Content-Type: application/json" \
  -H "X-API-Key: your-api-key" \
  -d '{
    "query": "deep learning vs machine learning",
    "config": {
      "pipeline": "quality",
      "retrieval": {"top_k": 20},
      "reranking": {"strategy": "cross_encoder", "top_k": 5}
    }
  }'

# Health check
curl http://localhost:8000/api/v1/rag/health \
  -H "X-API-Key: your-api-key"
```

### JavaScript/TypeScript

```typescript
interface SearchRequest {
  query: string;
  databases?: string[];
  top_k?: number;
}

async function search(request: SearchRequest): Promise<any> {
  const response = await fetch('http://localhost:8000/api/v1/rag/search/simple', {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
      'X-API-Key': 'your-api-key'
    },
    body: JSON.stringify(request)
  });
  
  return response.json();
}

// Usage
const results = await search({
  query: 'machine learning',
  databases: ['media', 'notes'],
  top_k: 5
});
```

## Migration Guide

### From v2 to v3

The v3 API uses a functional pipeline architecture. Key changes:

1. **Endpoint Consolidation**
   - Old: `/api/v1/rag/v2/search`, `/api/v1/rag/v3/search`
   - New: `/api/v1/rag/search/simple`, `/api/v1/rag/search/complex`

2. **Configuration Structure**
   ```python
   # Old (v2)
   {
     "query": "test",
     "search_type": "hybrid",
     "kwargs": {...}
   }
   
   # New (v3)
   {
     "query": "test",
     "config": {
       "pipeline": "standard",
       "retrieval": {...}
     }
   }
   ```

3. **Pipeline Selection**
   - v2: Fixed search types (simple, hybrid, semantic)
   - v3: Flexible pipelines (minimal, standard, quality, custom)

4. **Response Format**
   - More detailed metadata
   - Component timings included
   - Debug information available

### Backward Compatibility

For backward compatibility during migration:

1. Use the simple search endpoint for basic queries
2. Map old search types to new pipelines:
   - `simple` → `minimal` pipeline
   - `hybrid` → `standard` pipeline
   - `semantic` → `quality` pipeline with vector search

## Monitoring and Metrics

### Available Metrics

- `rag_queries_total` - Total number of queries processed
- `rag_query_duration_seconds` - Query processing time
- `rag_cache_hits_total` - Cache hit count
- `rag_cache_misses_total` - Cache miss count
- `rag_errors_total` - Error count by type
- `rag_documents_retrieved_total` - Documents retrieved
- `rag_reranking_duration_seconds` - Reranking time

### Logging

Structured logging with levels:
- `DEBUG` - Detailed execution flow
- `INFO` - Normal operations
- `WARNING` - Performance issues, fallbacks
- `ERROR` - Failures requiring attention

### Performance Dashboard

Access metrics at: `http://localhost:8000/metrics`

## Support

For issues or questions:
- GitHub Issues: [tldw_server/issues](https://github.com/your-org/tldw_server/issues)
- Documentation: [Full Documentation](https://docs.tldw-server.com)
- API Status: [status.tldw-server.com](https://status.tldw-server.com)