# RAG API Documentation

**Version**: 2.0  
**Last Updated**: 2025-08-18  
**Status**: Production Ready

## Table of Contents

1. [Overview](#overview)
2. [Authentication](#authentication)
3. [Rate Limiting](#rate-limiting)
4. [Endpoints](#endpoints)
   - [Search Endpoints](#search-endpoints)
   - [Agent Endpoints](#agent-endpoints)
5. [Data Models](#data-models)
6. [Error Handling](#error-handling)
7. [Performance](#performance)
8. [Examples](#examples)
9. [SDKs and Client Libraries](#sdks-and-client-libraries)
10. [Monitoring and Metrics](#monitoring-and-metrics)

## Overview

The RAG (Retrieval-Augmented Generation) API provides powerful search and AI-powered question-answering capabilities across your indexed content. It combines traditional search with semantic understanding and LLM-based generation to deliver accurate, contextual responses.

### Key Features

- **Hybrid Search**: Combines keyword and semantic search for optimal results
- **Multi-Database Support**: Search across media, notes, characters, and chat history
- **Intelligent Agents**: AI-powered assistants that retrieve context and generate responses
- **Streaming Support**: Real-time streaming for long-form responses
- **Audit Logging**: Complete audit trail of all operations
- **Rate Limiting**: Tiered rate limits based on user subscription
- **Custom Metrics**: Advanced evaluation metrics for quality monitoring

### Base URL

```
https://api.yourdomain.com/api/v1/rag
```

## Authentication

All endpoints require authentication using Bearer tokens in the Authorization header.

```http
Authorization: Bearer <your-jwt-token>
```

### Obtaining a Token

```http
POST /api/v1/auth/login
Content-Type: application/json

{
  "username": "your_username",
  "password": "your_password"
}
```

Response:
```json
{
  "access_token": "eyJ0eXAiOiJKV1QiLCJhbGc...",
  "token_type": "bearer",
  "expires_in": 86400
}
```

## Rate Limiting

Rate limits are enforced per user based on subscription tier:

| Tier | Search/min | Agent/min | Tokens/day | Cost/day |
|------|------------|-----------|------------|----------|
| Free | 60 | 30 | 100,000 | $1.00 |
| Basic | 180 | 90 | 1,000,000 | $10.00 |
| Premium | 600 | 300 | 10,000,000 | $100.00 |
| Enterprise | Custom | Custom | Custom | Custom |

Rate limit headers are included in all responses:
```http
X-RateLimit-Limit: 60
X-RateLimit-Remaining: 45
X-RateLimit-Reset: 1629823200
```

## Endpoints

### Search Endpoints

#### Simple Search

**POST** `/search`

Performs a search across specified databases with minimal configuration.

**Request Body:**
```json
{
  "query": "machine learning basics",
  "search_type": "hybrid",
  "limit": 10,
  "databases": ["media_db", "notes"],
  "keywords": ["AI", "ML"]
}
```

**Parameters:**
- `query` (string, required): Search query text (1-1000 chars)
- `search_type` (enum): `hybrid`, `semantic`, or `fulltext` (default: `hybrid`)
- `limit` (integer): Max results to return, 1-100 (default: 10)
- `databases` (array): Databases to search (default: `["media_db"]`)
- `keywords` (array, optional): Keywords to filter results

**Response:**
```json
{
  "results": [
    {
      "id": "doc_123",
      "title": "Introduction to Machine Learning",
      "content": "Machine learning is a subset of artificial intelligence...",
      "score": 0.95,
      "source": "media_db",
      "metadata": {
        "author": "John Doe",
        "date": "2024-01-15",
        "tags": ["ML", "AI", "tutorial"]
      }
    }
  ],
  "total_results": 42,
  "query_id": "550e8400-e29b-41d4-a716-446655440000",
  "search_type_used": "hybrid"
}
```

**Status Codes:**
- `200 OK`: Successful search
- `400 Bad Request`: Invalid parameters
- `401 Unauthorized`: Invalid or missing token
- `429 Too Many Requests`: Rate limit exceeded
- `500 Internal Server Error`: Server error

**Performance:**
- Average latency: < 500ms
- P95 latency: < 2000ms
- Max results: 100

---

#### Advanced Search

**POST** `/search/advanced`

Performs an advanced search with full configuration options.

**Request Body:**
```json
{
  "query": "transformer architecture in deep learning",
  "search_config": {
    "search_type": "hybrid",
    "limit": 20,
    "offset": 0,
    "databases": ["media_db", "notes"],
    "keywords": ["transformer", "attention"],
    "date_range": {
      "start": "2024-01-01",
      "end": "2024-12-31"
    },
    "metadata_filters": {
      "author": "Vaswani",
      "type": "paper"
    },
    "include_scores": true,
    "include_full_content": false
  },
  "hybrid_config": {
    "semantic_weight": 0.7,
    "fulltext_weight": 0.3,
    "rrf_k": 60
  },
  "semantic_config": {
    "similarity_threshold": 0.7,
    "rerank": true,
    "embedding_model": "text-embedding-3-small"
  },
  "strategy": "query_fusion"
}
```

**Parameters:**

*search_config:*
- `search_type`: Type of search (`hybrid`, `semantic`, `fulltext`)
- `limit`: Results per page (1-1000)
- `offset`: Pagination offset
- `databases`: List of databases to search
- `keywords`: Keyword filters
- `date_range`: Date filtering with ISO format dates
- `metadata_filters`: Complex metadata filtering
- `include_scores`: Include relevance scores
- `include_full_content`: Return full document content

*hybrid_config:*
- `semantic_weight`: Weight for semantic search (0-1)
- `fulltext_weight`: Weight for keyword search (0-1)
- `rrf_k`: Reciprocal rank fusion parameter

*semantic_config:*
- `similarity_threshold`: Minimum similarity score (0-1)
- `rerank`: Enable result reranking
- `embedding_model`: Specific embedding model to use

*strategy:*
- `vanilla`: Standard search
- `query_fusion`: Multiple query generation and fusion
- `hyde`: Hypothetical document embeddings

**Response:**
```json
{
  "results": [...],
  "total_results": 156,
  "query_id": "550e8400-e29b-41d4-a716-446655440000",
  "search_type_used": "hybrid",
  "strategy_used": "query_fusion",
  "search_config": {...},
  "debug_info": {
    "query_expansion": ["transformer model", "attention mechanism"],
    "databases_searched": 2,
    "time_ms": 342
  }
}
```

### Agent Endpoints

#### Simple Agent

**POST** `/agent`

AI agent for question answering with automatic context retrieval.

**Request Body:**
```json
{
  "message": "What are the key concepts in deep learning?",
  "conversation_id": "550e8400-e29b-41d4-a716-446655440000",
  "search_databases": ["media_db", "notes"],
  "model": "gpt-4"
}
```

**Parameters:**
- `message` (string, required): User's question or message (1-4000 chars)
- `conversation_id` (string, optional): ID to maintain conversation context
- `search_databases` (array): Databases to search for context (default: `["media_db"]`)
- `model` (string, optional): LLM model override

**Response:**
```json
{
  "response": "Deep learning involves several key concepts:\n\n1. **Neural Networks**: ...",
  "conversation_id": "550e8400-e29b-41d4-a716-446655440000",
  "sources": [
    {
      "title": "Deep Learning Fundamentals",
      "snippet": "Neural networks are the foundation...",
      "relevance": 0.92,
      "metadata": {
        "page": 42,
        "author": "Ian Goodfellow"
      }
    }
  ],
  "tokens_used": 850,
  "model_used": "gpt-4"
}
```

**Streaming Response:**

Add `Accept: text/event-stream` header for streaming:

```
data: {"chunk": "Deep learning involves", "index": 0}
data: {"chunk": " several key concepts:", "index": 1}
data: {"done": true, "sources": [...]}
```

---

#### Advanced Agent

**POST** `/agent/advanced`

Advanced agent with research capabilities and tool usage.

**Request Body:**
```json
{
  "message": "Research the latest developments in quantum computing",
  "conversation_id": "550e8400-e29b-41d4-a716-446655440000",
  "mode": "research",
  "generation_config": {
    "model": "gpt-4",
    "temperature": 0.7,
    "max_tokens": 2048,
    "top_p": 0.9,
    "frequency_penalty": 0.0,
    "presence_penalty": 0.0,
    "stop_sequences": ["\\n\\n"]
  },
  "search_config": {
    "search_type": "hybrid",
    "databases": ["media_db", "notes"],
    "keywords": ["quantum", "computing"],
    "limit": 20
  },
  "tools": ["web_search", "reasoning"],
  "system_prompt": "You are an expert researcher in quantum computing."
}
```

**Parameters:**

*Core:*
- `message`: User's message (1-4000 chars)
- `conversation_id`: Conversation tracking
- `mode`: `rag` (Q&A) or `research` (multi-step)

*generation_config:*
- `model`: LLM model to use
- `temperature`: Randomness (0-2)
- `max_tokens`: Max response length
- `top_p`: Nucleus sampling
- `frequency_penalty`: Reduce repetition
- `presence_penalty`: Encourage new topics
- `stop_sequences`: Stop generation triggers

*search_config:*
- Similar to simple search parameters

*tools:*
- `web_search`: Search the web
- `web_scrape`: Extract web content
- `reasoning`: Multi-step reasoning
- `python_executor`: Execute Python code

**Response:**
```json
{
  "response": "Based on my research...",
  "conversation_id": "550e8400-e29b-41d4-a716-446655440000",
  "sources": [...],
  "mode_used": "research",
  "tools_used": ["web_search", "reasoning"],
  "search_stats": {
    "documents_retrieved": 20,
    "documents_used": 5,
    "search_time_ms": 450
  },
  "generation_stats": {
    "tokens_prompt": 1200,
    "tokens_completion": 850,
    "time_ms": 3400,
    "model": "gpt-4"
  }
}
```

## Data Models

### SearchResult
```typescript
interface SearchResult {
  id: string;
  title: string;
  content: string;
  score: number;
  source: string;
  metadata: Record<string, any>;
}
```

### Source
```typescript
interface Source {
  title: string;
  snippet: string;
  relevance: number;
  metadata?: Record<string, any>;
}
```

### ErrorResponse
```typescript
interface ErrorResponse {
  error: string;
  detail?: string;
  code?: string;
}
```

## Error Handling

All errors follow a consistent format:

```json
{
  "error": "Invalid search type",
  "detail": "Search type 'invalid' is not supported. Use: hybrid, semantic, or fulltext",
  "code": "INVALID_SEARCH_TYPE"
}
```

### Common Error Codes

| Code | Description | Resolution |
|------|-------------|------------|
| `INVALID_SEARCH_TYPE` | Invalid search type specified | Use valid type: hybrid, semantic, fulltext |
| `RATE_LIMIT_EXCEEDED` | Too many requests | Wait for rate limit reset |
| `INVALID_DATABASE` | Unknown database specified | Check available databases |
| `QUERY_TOO_LONG` | Query exceeds max length | Limit query to 1000 characters |
| `INSUFFICIENT_CREDITS` | Out of tokens/credits | Upgrade subscription or wait for reset |
| `MODEL_UNAVAILABLE` | Requested model offline | Use different model or retry |
| `CONTEXT_TOO_LARGE` | Retrieved context too large | Reduce search limit or refine query |

## Performance

### Latency Targets

| Endpoint | P50 | P95 | P99 |
|----------|-----|-----|-----|
| Simple Search | 200ms | 500ms | 1s |
| Advanced Search | 300ms | 800ms | 2s |
| Simple Agent | 2s | 5s | 10s |
| Advanced Agent | 3s | 8s | 15s |

### Throughput

- Search endpoints: 1000+ req/s
- Agent endpoints: 100+ req/s
- Concurrent users: 1000+

### Optimization Tips

1. **Use appropriate search types**: Fulltext for keywords, semantic for concepts
2. **Limit result counts**: Only request what you need
3. **Cache frequently used queries**: Implement client-side caching
4. **Use streaming for long responses**: Reduces perceived latency
5. **Batch related queries**: Combine multiple searches when possible

## Examples

### Python

```python
import httpx
import asyncio

class RAGClient:
    def __init__(self, api_key: str, base_url: str = "https://api.yourdomain.com"):
        self.api_key = api_key
        self.base_url = base_url
        self.headers = {"Authorization": f"Bearer {api_key}"}
    
    async def search(self, query: str, **kwargs):
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{self.base_url}/api/v1/rag/search",
                json={"query": query, **kwargs},
                headers=self.headers
            )
            response.raise_for_status()
            return response.json()
    
    async def ask_agent(self, message: str, conversation_id: str = None):
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{self.base_url}/api/v1/rag/agent",
                json={
                    "message": message,
                    "conversation_id": conversation_id
                },
                headers=self.headers
            )
            response.raise_for_status()
            return response.json()

# Usage
async def main():
    client = RAGClient("your-api-key")
    
    # Search
    results = await client.search(
        "machine learning",
        search_type="hybrid",
        limit=5
    )
    
    # Agent
    response = await client.ask_agent(
        "Explain machine learning in simple terms"
    )
    print(response["response"])

asyncio.run(main())
```

### JavaScript/TypeScript

```typescript
class RAGClient {
  private apiKey: string;
  private baseUrl: string;

  constructor(apiKey: string, baseUrl = "https://api.yourdomain.com") {
    this.apiKey = apiKey;
    this.baseUrl = baseUrl;
  }

  async search(query: string, options = {}) {
    const response = await fetch(`${this.baseUrl}/api/v1/rag/search`, {
      method: 'POST',
      headers: {
        'Authorization': `Bearer ${this.apiKey}`,
        'Content-Type': 'application/json'
      },
      body: JSON.stringify({ query, ...options })
    });

    if (!response.ok) {
      throw new Error(`Search failed: ${response.statusText}`);
    }

    return response.json();
  }

  async askAgent(message: string, conversationId?: string) {
    const response = await fetch(`${this.baseUrl}/api/v1/rag/agent`, {
      method: 'POST',
      headers: {
        'Authorization': `Bearer ${this.apiKey}`,
        'Content-Type': 'application/json'
      },
      body: JSON.stringify({
        message,
        conversation_id: conversationId
      })
    });

    if (!response.ok) {
      throw new Error(`Agent request failed: ${response.statusText}`);
    }

    return response.json();
  }

  // Streaming support
  async *streamAgent(message: string) {
    const response = await fetch(`${this.baseUrl}/api/v1/rag/agent`, {
      method: 'POST',
      headers: {
        'Authorization': `Bearer ${this.apiKey}`,
        'Content-Type': 'application/json',
        'Accept': 'text/event-stream'
      },
      body: JSON.stringify({ message })
    });

    const reader = response.body?.getReader();
    const decoder = new TextDecoder();

    while (true) {
      const { done, value } = await reader!.read();
      if (done) break;
      
      const chunk = decoder.decode(value);
      const lines = chunk.split('\n');
      
      for (const line of lines) {
        if (line.startsWith('data: ')) {
          const data = JSON.parse(line.slice(6));
          yield data;
        }
      }
    }
  }
}

// Usage
const client = new RAGClient('your-api-key');

// Search
const results = await client.search('machine learning', {
  searchType: 'hybrid',
  limit: 5
});

// Streaming agent
for await (const chunk of client.streamAgent('Explain quantum computing')) {
  if (chunk.chunk) {
    process.stdout.write(chunk.chunk);
  }
}
```

### cURL

```bash
# Simple search
curl -X POST https://api.yourdomain.com/api/v1/rag/search \
  -H "Authorization: Bearer your-api-key" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "machine learning",
    "search_type": "hybrid",
    "limit": 10
  }'

# Agent with streaming
curl -X POST https://api.yourdomain.com/api/v1/rag/agent \
  -H "Authorization: Bearer your-api-key" \
  -H "Content-Type: application/json" \
  -H "Accept: text/event-stream" \
  -d '{
    "message": "What is artificial intelligence?"
  }'
```

## SDKs and Client Libraries

### Official SDKs

- **Python**: `pip install tldw-rag-client`
- **JavaScript/TypeScript**: `npm install @tldw/rag-client`
- **Go**: `go get github.com/tldw/rag-client-go`
- **Ruby**: `gem install tldw-rag-client`

### Community SDKs

- **Rust**: `cargo add tldw-rag`
- **Java**: Maven package `com.tldw:rag-client`
- **PHP**: Composer package `tldw/rag-client`

## Monitoring and Metrics

### Available Metrics

Metrics are exposed at `/api/v1/rag/metrics` in Prometheus format:

```
# Request latency
rag_request_duration_seconds{endpoint="search",quantile="0.5"} 0.2
rag_request_duration_seconds{endpoint="search",quantile="0.95"} 0.5
rag_request_duration_seconds{endpoint="search",quantile="0.99"} 1.0

# Request count
rag_requests_total{endpoint="search",status="success"} 10543
rag_requests_total{endpoint="search",status="error"} 23

# Token usage
rag_tokens_used_total{user="user123",model="gpt-4"} 450230

# Cost tracking
rag_estimated_cost_dollars{user="user123",endpoint="agent"} 12.45
```

### Audit Logs

All operations are logged for compliance and debugging:

```json
{
  "timestamp": "2024-08-18T10:30:45Z",
  "event_type": "search_request",
  "user_id": "user123",
  "request_id": "550e8400-e29b-41d4-a716-446655440000",
  "endpoint": "/api/v1/rag/search",
  "query": "machine learning",
  "databases_searched": ["media_db"],
  "result_count": 10,
  "latency_ms": 234,
  "tokens_used": 0,
  "status": "success"
}
```

### Health Checks

```http
GET /api/v1/rag/health
```

Response:
```json
{
  "status": "healthy",
  "version": "2.0.0",
  "uptime_seconds": 86400,
  "database_status": "connected",
  "vector_store_status": "connected",
  "llm_status": "available"
}
```

## Support

### Resources

- **API Status**: https://status.yourdomain.com
- **Support Portal**: https://support.yourdomain.com
- **Community Forum**: https://forum.yourdomain.com
- **GitHub Issues**: https://github.com/yourdomain/tldw-server/issues

### Contact

- **Email**: api-support@yourdomain.com
- **Discord**: https://discord.gg/yourdomain
- **Enterprise Support**: enterprise@yourdomain.com

---

*This documentation is version-controlled and updated with each API release. For the latest version, visit https://docs.yourdomain.com/api/rag*