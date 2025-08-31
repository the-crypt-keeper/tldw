# RAG API Documentation - Unified Pipeline

## Overview

The RAG (Retrieval-Augmented Generation) API provides a unified pipeline where ALL features are accessible through direct parameters. No configuration files, no presets - just explicit parameter control for maximum flexibility.

**Base URL**: `/api/v1/rag/`

## Authentication

All endpoints require authentication (when auth is enabled):
```bash
Authorization: Bearer <your-jwt-token>
```

## Primary Endpoints

### POST `/search` - Unified RAG Search

The main RAG endpoint with complete feature access.

#### Request Schema

```json
{
  // ========== REQUIRED ==========
  "query": "string (1-2000 chars)",
  
  // ========== DATA SOURCES ==========
  "sources": ["media_db", "notes", "characters", "chats"],  // Default: ["media_db"]
  "media_db_path": "string (optional)",
  "notes_db_path": "string (optional)",
  "character_db_path": "string (optional)",
  
  // ========== SEARCH CONFIGURATION ==========
  "search_mode": "hybrid",  // "fts" | "vector" | "hybrid"
  "top_k": 10,              // Max results (1-100)
  "similarity_threshold": 0.3,  // Vector similarity threshold (0.0-1.0)
  "fts_weight": 0.5,        // FTS weight in hybrid search (0.0-1.0)
  "vector_weight": 0.5,     // Vector weight in hybrid search (0.0-1.0)
  
  // ========== QUERY ENHANCEMENT ==========
  "enable_query_expansion": false,
  "expansion_strategies": ["acronym", "synonym", "domain", "entity"],
  "enable_spell_check": false,
  "spell_check_language": "en",
  
  // ========== FILTERING ==========
  "keyword_filter": ["term1", "term2"],  // Must contain these keywords
  "exclude_keywords": ["term1"],         // Must not contain these
  "date_range": {
    "start": "2024-01-01",
    "end": "2024-12-31"
  },
  "source_filter": ["youtube", "pdf"],   // Filter by content source type
  
  // ========== CACHING ==========
  "enable_cache": true,
  "cache_threshold": 0.85,    // Semantic similarity threshold (0.0-1.0)
  "cache_ttl": 3600,          // Cache TTL in seconds
  
  // ========== DOCUMENT PROCESSING ==========
  "enable_reranking": true,
  "reranking_strategy": "hybrid",  // "flashrank" | "cross_encoder" | "hybrid"
  "reranking_top_k": 20,          // Candidates before reranking
  "enable_table_processing": false,
  "enable_parent_retrieval": false,
  "parent_context_size": 2,        // Number of surrounding chunks
  
  // ========== CITATIONS ==========
  "enable_citations": false,
  "citation_style": "apa",  // "mla" | "apa" | "chicago" | "harvard" | "ieee"
  "enable_chunk_citations": true,
  "citation_threshold": 0.7,  // Minimum confidence for citations
  
  // ========== ANSWER GENERATION ==========
  "enable_answer_generation": false,
  "llm_provider": "openai",     // "openai" | "anthropic" | "cohere" | etc.
  "model": "gpt-4o",           // Model name
  "generation_prompt": "string (optional custom prompt)",
  "max_tokens": 1000,          // Max tokens for generated answer
  "temperature": 0.1,          // Generation temperature (0.0-2.0)
  
  // ========== SECURITY & PRIVACY ==========
  "enable_pii_detection": false,
  "pii_types": ["email", "ssn", "credit_card", "phone"],
  "enable_content_filtering": false,
  "content_filter_level": "medium",  // "none" | "low" | "medium" | "high"
  "redact_pii": false,              // Whether to redact detected PII
  
  // ========== ANALYTICS & FEEDBACK ==========
  "enable_analytics": true,
  "enable_feedback_collection": true,
  "user_id": "string (optional)",
  "session_id": "string (optional)",
  
  // ========== PERFORMANCE ==========
  "enable_connection_pooling": true,
  "enable_embedding_cache": true,
  "enable_monitoring": false,
  "enable_debug_mode": false,
  
  // ========== RESILIENCE ==========
  "enable_resilience": false,
  "resilience": {
    "retry": {
      "enabled": true,
      "max_attempts": 3,
      "initial_delay": 0.5,
      "backoff_multiplier": 2.0
    },
    "circuit_breaker": {
      "enabled": true,
      "failure_threshold": 5,
      "timeout": 60
    }
  },
  
  // ========== OUTPUT CONFIGURATION ==========
  "include_metadata": true,
  "include_embeddings": false,
  "include_scores": true,
  "enable_result_highlighting": false
}
```

#### Response Schema

```json
{
  "documents": [
    {
      "id": "string",
      "content": "string",
      "metadata": {
        "title": "string",
        "author": "string",
        "date": "string",
        "url": "string",
        // ... additional metadata
      },
      "source": "media_db",  // DataSource enum
      "score": 0.95,         // Relevance score
      "source_document_id": "string",
      "chunk_index": 0,
      "total_chunks": 5,
      "page_number": 42,
      "section_title": "Chapter 3"
    }
  ],
  "query": "original query",
  "expanded_queries": ["expanded", "queries"],
  "metadata": {
    "total_results": 10,
    "search_mode": "hybrid",
    "sources_searched": ["media_db", "notes"],
    "cache_hit": false,
    "reranked": true
  },
  "timings": {
    "total_time": 0.245,
    "retrieval_time": 0.120,
    "reranking_time": 0.085,
    "citation_time": 0.040
  },
  "citations": [
    "Smith, J. (2024). Machine Learning Fundamentals. Tech Publications."
  ],
  "chunk_citations": [
    {
      "chunk_id": "doc1",
      "source_document_id": "source1",
      "source_document_title": "ML Introduction",
      "location": "Page 42, Section: Chapter 3",
      "text_snippet": "Machine learning is...",
      "confidence": 0.95,
      "usage_context": "Direct answer to query"
    }
  ],
  "feedback_id": "fb_12345",  // For user feedback collection
  "generated_answer": "Machine learning is a subset of artificial intelligence...",
  "cache_hit": false,
  "errors": [],
  "security_report": {
    "pii_detected": ["email"],
    "content_filtered": false,
    "risk_level": "low"
  },
  "total_time": 0.245
}
```

#### HTTP Status Codes

- `200` - Success
- `400` - Bad Request (invalid parameters)
- `401` - Unauthorized
- `422` - Validation Error
- `429` - Rate Limited
- `500` - Internal Server Error

#### Example Requests

**Basic Search**:
```bash
curl -X POST "http://localhost:8000/api/v1/rag/search" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "What is machine learning?",
    "sources": ["media_db"],
    "top_k": 5
  }'
```

**Advanced Search with Citations**:
```bash
curl -X POST "http://localhost:8000/api/v1/rag/search" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "Explain neural networks in detail",
    "sources": ["media_db", "notes"],
    "search_mode": "hybrid",
    "enable_query_expansion": true,
    "expansion_strategies": ["acronym", "synonym", "domain"],
    "enable_reranking": true,
    "reranking_strategy": "hybrid",
    "enable_citations": true,
    "citation_style": "apa",
    "enable_chunk_citations": true,
    "enable_answer_generation": true,
    "llm_provider": "openai",
    "model": "gpt-4o",
    "top_k": 15
  }'
```

### POST `/batch` - Batch RAG Processing

Process multiple queries concurrently.

#### Request Schema

```json
{
  "queries": ["query1", "query2", "query3"],  // Required: list of queries
  "max_concurrent": 3,                        // Max concurrent processing
  
  // All unified pipeline parameters supported
  "sources": ["media_db"],
  "search_mode": "hybrid",
  "enable_citations": true,
  "citation_style": "apa",
  "top_k": 10
  // ... any other unified pipeline parameters
}
```

#### Response Schema

```json
{
  "results": [
    {
      // Same structure as single search response
      "documents": [...],
      "query": "query1",
      "metadata": {...},
      // ... full unified search result
    },
    // ... results for each query
  ],
  "metadata": {
    "total_queries": 3,
    "successful_queries": 3,
    "failed_queries": 0,
    "total_time": 0.456,
    "concurrent_processing": true
  },
  "errors": [
    {
      "query_index": 1,
      "query": "problematic query",
      "error": "Error message"
    }
  ]
}
```

### GET `/simple` - Simplified Search Interface

Quick search with common parameters only.

#### Query Parameters

- `q` (required): Search query
- `sources`: Comma-separated sources (default: "media_db")
- `limit`: Max results (default: 10)
- `mode`: Search mode ("fts", "vector", "hybrid", default: "hybrid")

#### Example

```bash
curl "http://localhost:8000/api/v1/rag/simple?q=machine%20learning&sources=media_db,notes&limit=5&mode=hybrid"
```

### GET `/advanced` - Pre-configured Advanced Search

Advanced search with commonly used features enabled.

#### Query Parameters

Same as simple, plus:
- `expand`: Enable query expansion (true/false)
- `rerank`: Enable reranking (true/false) 
- `citations`: Enable citations (true/false)
- `style`: Citation style (mla/apa/chicago/harvard/ieee)

### GET `/features` - Available Features

Get list of all available features and parameters.

#### Response

```json
{
  "features": {
    "search_modes": ["fts", "vector", "hybrid"],
    "sources": ["media_db", "notes", "characters", "chats"],
    "expansion_strategies": ["acronym", "synonym", "domain", "entity"],
    "reranking_strategies": ["flashrank", "cross_encoder", "hybrid"],
    "citation_styles": ["mla", "apa", "chicago", "harvard", "ieee"],
    "llm_providers": ["openai", "anthropic", "cohere", "groq"],
    "content_filter_levels": ["none", "low", "medium", "high"],
    "pii_types": ["email", "ssn", "credit_card", "phone", "address"]
  },
  "parameters": {
    "query": {
      "type": "string",
      "required": true,
      "min_length": 1,
      "max_length": 2000
    },
    "top_k": {
      "type": "integer",
      "default": 10,
      "min": 1,
      "max": 100
    }
    // ... all parameter specifications
  }
}
```

### GET `/health` - Health Check

Check health status of all RAG components.

#### Response

```json
{
  "status": "healthy",
  "components": {
    "databases": {
      "media_db": "healthy",
      "notes_db": "healthy",
      "characters_db": "healthy"
    },
    "embedding_service": "healthy",
    "cache": "healthy",
    "llm_providers": {
      "openai": "healthy",
      "anthropic": "degraded"
    }
  },
  "performance": {
    "avg_response_time": 0.156,
    "cache_hit_rate": 0.78,
    "error_rate": 0.02
  },
  "version": "4.0",
  "timestamp": "2024-08-30T10:00:00Z"
}
```

## Parameter Reference

### Core Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `query` | string | - | Search query (required) |
| `sources` | array | ["media_db"] | Databases to search |
| `search_mode` | string | "hybrid" | fts, vector, or hybrid |
| `top_k` | integer | 10 | Maximum results (1-100) |

### Query Enhancement

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `enable_query_expansion` | boolean | false | Enable query expansion |
| `expansion_strategies` | array | ["acronym"] | Expansion strategies |
| `enable_spell_check` | boolean | false | Correct query spelling |
| `spell_check_language` | string | "en" | Language for spell check |

### Filtering

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `keyword_filter` | array | [] | Required keywords |
| `exclude_keywords` | array | [] | Excluded keywords |
| `date_range` | object | null | Date range filter |
| `source_filter` | array | [] | Content source types |
| `similarity_threshold` | float | 0.3 | Vector similarity threshold |

### Caching

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `enable_cache` | boolean | true | Enable semantic caching |
| `cache_threshold` | float | 0.85 | Similarity threshold for cache |
| `cache_ttl` | integer | 3600 | Cache TTL in seconds |

### Document Processing

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `enable_reranking` | boolean | true | Enable document reranking |
| `reranking_strategy` | string | "hybrid" | Reranking algorithm |
| `reranking_top_k` | integer | 20 | Candidates before reranking |
| `enable_table_processing` | boolean | false | Process table content |
| `enable_parent_retrieval` | boolean | false | Include parent context |

### Citations

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `enable_citations` | boolean | false | Generate academic citations |
| `citation_style` | string | "apa" | Citation format |
| `enable_chunk_citations` | boolean | true | Include chunk citations |
| `citation_threshold` | float | 0.7 | Min confidence for citations |

### Answer Generation

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `enable_answer_generation` | boolean | false | Generate LLM responses |
| `llm_provider` | string | "openai" | LLM provider |
| `model` | string | "gpt-4o" | Model name |
| `max_tokens` | integer | 1000 | Max response tokens |
| `temperature` | float | 0.1 | Generation temperature |

### Security & Privacy

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `enable_pii_detection` | boolean | false | Detect PII in results |
| `pii_types` | array | ["email", "ssn"] | PII types to detect |
| `enable_content_filtering` | boolean | false | Filter inappropriate content |
| `content_filter_level` | string | "medium" | Filter sensitivity |
| `redact_pii` | boolean | false | Redact detected PII |

### Performance & Monitoring

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `enable_connection_pooling` | boolean | true | Use connection pooling |
| `enable_embedding_cache` | boolean | true | Cache embeddings |
| `enable_monitoring` | boolean | false | Collect performance metrics |
| `enable_debug_mode` | boolean | false | Include debug information |

## Error Handling

### Common Error Responses

**Validation Error (422)**:
```json
{
  "detail": [
    {
      "loc": ["body", "top_k"],
      "msg": "ensure this value is less than or equal to 100",
      "type": "value_error.number.not_le",
      "ctx": {"limit_value": 100}
    }
  ]
}
```

**Rate Limited (429)**:
```json
{
  "detail": "Rate limit exceeded. Try again in 60 seconds.",
  "retry_after": 60
}
```

**Internal Error (500)**:
```json
{
  "detail": "Internal server error occurred",
  "error_id": "err_12345",
  "message": "Database connection failed"
}
```

## Rate Limits

- **Search endpoint**: 60 requests per minute per user
- **Batch endpoint**: 10 requests per minute per user
- **Health endpoint**: 120 requests per minute per user

## Best Practices

### Performance Optimization

1. **Use Caching**: Enable `enable_cache=true` for repeated queries
2. **Connection Pooling**: Keep `enable_connection_pooling=true`
3. **Embedding Cache**: Keep `enable_embedding_cache=true`
4. **Limit Results**: Use appropriate `top_k` values (don't over-fetch)
5. **Batch Processing**: Use `/batch` endpoint for multiple queries

### Security Considerations

1. **PII Detection**: Enable for sensitive data sources
2. **Content Filtering**: Use appropriate filter levels
3. **Authentication**: Always authenticate requests
4. **Input Validation**: Validate query inputs client-side

### Citation Best Practices

1. **Academic Work**: Use `enable_citations=true` with appropriate style
2. **Verification**: Always enable `enable_chunk_citations=true`
3. **Confidence Thresholds**: Adjust `citation_threshold` based on needs
4. **Multiple Styles**: Different styles for different audiences

### Analytics Privacy

1. **User Consent**: Only enable analytics with user consent
2. **Data Minimization**: RAG automatically hashes sensitive data
3. **Retention**: Analytics data follows configured retention policies
4. **Transparency**: Users can see what analytics are collected

## Integration Examples

### Python SDK Example

```python
import httpx
import asyncio

async def search_rag(query: str, **kwargs):
    async with httpx.AsyncClient() as client:
        response = await client.post(
            "http://localhost:8000/api/v1/rag/search",
            json={"query": query, **kwargs}
        )
        return response.json()

# Usage
result = await search_rag(
    "What is machine learning?",
    sources=["media_db", "notes"],
    enable_citations=True,
    citation_style="apa",
    enable_answer_generation=True,
    top_k=10
)
```

### JavaScript Example

```javascript
async function searchRAG(query, options = {}) {
  const response = await fetch('/api/v1/rag/search', {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
      'Authorization': `Bearer ${token}`
    },
    body: JSON.stringify({
      query,
      ...options
    })
  });
  
  return response.json();
}

// Usage
const result = await searchRAG("Explain neural networks", {
  sources: ["media_db"],
  enable_citations: true,
  citation_style: "mla",
  top_k: 15
});
```

## Changelog

### v4.0 (Current)
- Unified pipeline architecture
- All features accessible via parameters
- Dual citation system
- Analytics integration
- Performance optimizations
- Batch processing
- Enhanced security features

### v3.0 (Deprecated)
- Functional pipeline with presets
- Limited feature accessibility
- Configuration-based approach

### v2.0 (Legacy)
- Object-oriented architecture
- Complex configuration classes
- Limited API coverage

---

For more information, see:
- [Implementation Status](IMPLEMENTATION_STATUS.md)
- [Migration Guide](MIGRATION_GUIDE.md) 
- [Main README](README.md)