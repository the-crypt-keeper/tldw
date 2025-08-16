# RAG API Consumer Guide

## Table of Contents
1. [Quick Start](#quick-start)
2. [Authentication](#authentication)
3. [Endpoint Reference](#endpoint-reference)
4. [Request & Response Schemas](#request--response-schemas)
5. [Search Types & Strategies](#search-types--strategies)
6. [Agent Modes & Tools](#agent-modes--tools)
7. [Code Examples](#code-examples)
8. [Best Practices](#best-practices)
9. [Error Handling](#error-handling)
10. [Rate Limiting & Performance](#rate-limiting--performance)

## Quick Start

The RAG API provides powerful search and question-answering capabilities across your content. Here's how to get started:

### Base URL
```
http://localhost:8000/api/v1/rag
```

### Available Endpoints
- `POST /search` - Simple search across your content
- `POST /search/advanced` - Advanced search with full control
- `POST /agent` - Simple Q&A agent
- `POST /agent/advanced` - Research agent with tools
- `GET /health` - Service health check

### Quick Example

```bash
# Simple search
curl -X POST http://localhost:8000/api/v1/rag/search \
  -H "X-API-KEY: your-api-key" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "machine learning concepts",
    "limit": 5,
    "databases": ["media_db"]
  }'
```

## Authentication

All RAG endpoints require authentication via API key.

### Single-User Mode
Use the default API key in the header:
```http
X-API-KEY: default-secret-key-for-single-user
```

### Multi-User Mode
Use your personal API key:
```http
X-API-KEY: your-personal-api-key
```

### Example Headers
```javascript
const headers = {
  'X-API-KEY': 'your-api-key',
  'Content-Type': 'application/json'
};
```

## Endpoint Reference

### 1. Simple Search - `POST /search`

Perform a straightforward search across your content with minimal configuration.

#### Request
```typescript
interface SimpleSearchRequest {
  query: string;                    // Search query (1-1000 chars)
  search_type?: "hybrid" | "semantic" | "fulltext";  // Default: "hybrid"
  limit?: number;                    // Max results (1-100, default: 10)
  databases?: string[];              // Default: ["media_db"]
  keywords?: string[];               // Optional keyword filters
}
```

#### Response
```typescript
interface SimpleSearchResponse {
  results: SearchResult[];
  total_results: number;
  search_type: string;
  databases_searched: string[];
  keywords_used?: string[];
}

interface SearchResult {
  id: string;
  title: string;
  content: string;          // Snippet (max 500 chars)
  score: number;
  source: string;
  metadata?: object;
}
```

#### Example
```javascript
const response = await fetch('http://localhost:8000/api/v1/rag/search', {
  method: 'POST',
  headers: {
    'X-API-KEY': 'your-api-key',
    'Content-Type': 'application/json'
  },
  body: JSON.stringify({
    query: "deep learning tutorials",
    search_type: "hybrid",
    limit: 10,
    databases: ["media_db", "notes"],
    keywords: ["python", "tensorflow"]
  })
});

const data = await response.json();
```

### 2. Advanced Search - `POST /search/advanced`

Full control over search with advanced filtering and strategies.

#### Request
```typescript
interface AdvancedSearchRequest {
  query: string;
  strategy?: "vanilla" | "query_fusion" | "hyde";  // Default: "vanilla"
  search_config?: {
    search_type?: "hybrid" | "semantic" | "fulltext";
    limit?: number;
    offset?: number;
    databases?: string[];
    keywords?: string[];
    metadata_filters?: object;
    date_range?: {
      start?: string;  // ISO date
      end?: string;
    };
    include_full_content?: boolean;
    include_scores?: boolean;
  };
  hybrid_config?: {
    semantic_weight?: number;  // 0.0-1.0
    fulltext_weight?: number;  // 0.0-1.0
    rrf_k?: number;           // Default: 60
  };
  semantic_config?: {
    similarity_threshold?: number;  // 0.0-1.0
    rerank?: boolean;
  };
}
```

#### Response
```typescript
interface AdvancedSearchResponse {
  results: SearchResult[];
  total_results: number;
  search_metadata: {
    strategy_used: string;
    databases_searched: string[];
    processing_time_ms: number;
    config_applied: object;
  };
  debug_info?: object;  // If debug enabled
}
```

#### Example with Query Fusion
```javascript
const response = await fetch('http://localhost:8000/api/v1/rag/search/advanced', {
  method: 'POST',
  headers: {
    'X-API-KEY': 'your-api-key',
    'Content-Type': 'application/json'
  },
  body: JSON.stringify({
    query: "quantum computing breakthroughs",
    strategy: "query_fusion",
    search_config: {
      search_type: "hybrid",
      limit: 20,
      databases: ["media_db", "notes"],
      metadata_filters: {
        "language": "en",
        "quality": "high"
      },
      date_range: {
        start: "2023-01-01",
        end: "2024-12-31"
      },
      include_full_content: true
    },
    hybrid_config: {
      semantic_weight: 0.7,
      fulltext_weight: 0.3,
      rrf_k: 60
    },
    semantic_config: {
      similarity_threshold: 0.5,
      rerank: true
    }
  })
});
```

### 3. Simple Agent - `POST /agent`

Conversational Q&A with automatic context retrieval.

#### Request
```typescript
interface SimpleAgentRequest {
  message: string;                  // User question/message
  conversation_id?: string;          // Continue conversation
  search_databases?: string[];      // Default: ["media_db"]
  model?: string;                    // LLM model override
}
```

#### Response
```typescript
interface SimpleAgentResponse {
  response: string;                  // Generated answer
  conversation_id: string;           // For follow-ups
  sources: Source[];                 // Top 3 sources
}

interface Source {
  title: string;
  snippet: string;
  relevance_score: number;
  source_type: string;
}
```

#### Example Conversation
```javascript
// Initial question
let response = await fetch('http://localhost:8000/api/v1/rag/agent', {
  method: 'POST',
  headers: {
    'X-API-KEY': 'your-api-key',
    'Content-Type': 'application/json'
  },
  body: JSON.stringify({
    message: "What are neural networks?",
    search_databases: ["media_db", "notes"]
  })
});

let data = await response.json();
const conversationId = data.conversation_id;

// Follow-up question
response = await fetch('http://localhost:8000/api/v1/rag/agent', {
  method: 'POST',
  headers: {
    'X-API-KEY': 'your-api-key',
    'Content-Type': 'application/json'
  },
  body: JSON.stringify({
    message: "How do they learn?",
    conversation_id: conversationId
  })
});
```

### 4. Advanced Agent - `POST /agent/advanced`

Research agent with tools and streaming support.

#### Request
```typescript
interface AdvancedAgentRequest {
  message: string;
  mode?: "rag" | "research";         // Default: "rag"
  tools?: string[];                  // For research mode
  conversation_id?: string;
  system_prompt?: string;            // Custom personality
  search_config?: object;            // Same as advanced search
  generation_config?: {
    model?: string;
    temperature?: number;            // 0.0-2.0
    max_tokens?: number;
    stream?: boolean;                // Enable SSE streaming
    top_p?: number;
    frequency_penalty?: number;
    presence_penalty?: number;
  };
}
```

#### Response (Non-Streaming)
```typescript
interface AdvancedAgentResponse {
  response: string;
  conversation_id: string;
  sources: Source[];
  tool_usage?: ToolUsage[];
  statistics?: {
    search_time_ms: number;
    generation_time_ms: number;
    total_tokens: number;
    sources_consulted: number;
  };
}

interface ToolUsage {
  tool: string;
  input: string;
  output: string;
  success: boolean;
}
```

#### Streaming Example (Server-Sent Events)
```javascript
const response = await fetch('http://localhost:8000/api/v1/rag/agent/advanced', {
  method: 'POST',
  headers: {
    'X-API-KEY': 'your-api-key',
    'Content-Type': 'application/json'
  },
  body: JSON.stringify({
    message: "Compare quantum computing approaches from IBM and Google",
    mode: "research",
    tools: ["web_search", "reasoning"],
    generation_config: {
      model: "gpt-4",
      temperature: 0.7,
      max_tokens: 2048,
      stream: true  // Enable streaming
    }
  })
});

// Handle Server-Sent Events
const reader = response.body.getReader();
const decoder = new TextDecoder();

while (true) {
  const { done, value } = await reader.read();
  if (done) break;
  
  const chunk = decoder.decode(value);
  const lines = chunk.split('\n');
  
  for (const line of lines) {
    if (line.startsWith('data: ')) {
      const data = JSON.parse(line.slice(6));
      console.log('Received:', data.content);
    }
  }
}
```

### 5. Health Check - `GET /health`

Check service availability.

#### Response
```typescript
interface HealthResponse {
  status: "healthy" | "unhealthy";
  service: "rag_v2";
  timestamp: number;
}
```

#### Example
```javascript
const response = await fetch('http://localhost:8000/api/v1/rag/health');
const health = await response.json();

if (health.status === 'healthy') {
  console.log('RAG service is operational');
}
```

## Request & Response Schemas

### Available Databases

| Database | Aliases | Description |
|----------|---------|-------------|
| `media_db` | `media` | Ingested videos, audio, documents |
| `notes` | - | Personal notes and annotations |
| `characters` | - | Character cards and personas |
| `chat_history` | `chats` | Conversation history |

### Search Types

| Type | Description | Use Case |
|------|-------------|----------|
| `hybrid` | Combines BM25 and vector search | Best overall results (default) |
| `semantic` | Vector similarity only | Conceptual searches |
| `fulltext` | BM25 full-text only | Exact phrase matching |

### Search Strategies

| Strategy | Description | Use Case |
|----------|-------------|----------|
| `vanilla` | Standard single query | Simple searches (default) |
| `query_fusion` | Multiple query variants | Improve coverage |
| `hyde` | Hypothetical Document Embeddings | Better semantic matching |

## Search Types & Strategies

### Hybrid Search Configuration

Hybrid search combines full-text and semantic search. Configure the balance:

```javascript
{
  "hybrid_config": {
    "semantic_weight": 0.7,    // 70% semantic
    "fulltext_weight": 0.3,    // 30% full-text
    "rrf_k": 60                // Reciprocal Rank Fusion parameter
  }
}
```

### Query Fusion Strategy

Generates multiple query variants for better coverage:

```javascript
// Original query: "machine learning algorithms"
// Query fusion generates:
// - "ML algorithms and techniques"
// - "artificial intelligence algorithms"
// - "deep learning methods"
// Then merges results from all variants
```

### HyDE (Hypothetical Document Embeddings)

Generates a hypothetical answer to improve semantic search:

```javascript
// Query: "How does gradient descent work?"
// HyDE generates a hypothetical document:
// "Gradient descent is an optimization algorithm that..."
// Then searches for similar documents
```

## Agent Modes & Tools

### RAG Mode (Default)

Standard retrieval-augmented generation:
1. Search databases for relevant context
2. Generate response using retrieved information
3. Include source citations

### Research Mode

Multi-step reasoning with tool usage:

#### Available Tools

| Tool | Description | Example Use |
|------|-------------|-------------|
| `web_search` | Search current web information | Latest news, current events |
| `reasoning` | Multi-step logical reasoning | Complex problem solving |
| `calculator` | Mathematical calculations | Numerical analysis |
| `code_execution` | Execute code (sandboxed) | Data processing, algorithms |

#### Research Mode Example
```javascript
{
  "message": "Analyze the environmental impact of cryptocurrency mining in 2024",
  "mode": "research",
  "tools": ["web_search", "reasoning", "calculator"],
  "system_prompt": "You are an environmental analyst. Provide data-driven insights."
}
```

## Code Examples

### JavaScript/TypeScript

```typescript
class RAGClient {
  private baseUrl: string;
  private apiKey: string;

  constructor(baseUrl: string, apiKey: string) {
    this.baseUrl = baseUrl;
    this.apiKey = apiKey;
  }

  async search(query: string, options: SearchOptions = {}): Promise<SearchResponse> {
    const response = await fetch(`${this.baseUrl}/search`, {
      method: 'POST',
      headers: {
        'X-API-KEY': this.apiKey,
        'Content-Type': 'application/json'
      },
      body: JSON.stringify({
        query,
        search_type: options.searchType || 'hybrid',
        limit: options.limit || 10,
        databases: options.databases || ['media_db'],
        keywords: options.keywords
      })
    });

    if (!response.ok) {
      throw new Error(`Search failed: ${response.statusText}`);
    }

    return response.json();
  }

  async ask(message: string, conversationId?: string): Promise<AgentResponse> {
    const response = await fetch(`${this.baseUrl}/agent`, {
      method: 'POST',
      headers: {
        'X-API-KEY': this.apiKey,
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

  async *streamAgent(message: string, config: AgentConfig = {}): AsyncGenerator<string> {
    const response = await fetch(`${this.baseUrl}/agent/advanced`, {
      method: 'POST',
      headers: {
        'X-API-KEY': this.apiKey,
        'Content-Type': 'application/json'
      },
      body: JSON.stringify({
        message,
        mode: config.mode || 'rag',
        generation_config: {
          ...config.generationConfig,
          stream: true
        }
      })
    });

    const reader = response.body!.getReader();
    const decoder = new TextDecoder();

    while (true) {
      const { done, value } = await reader.read();
      if (done) break;

      const chunk = decoder.decode(value);
      const lines = chunk.split('\n');

      for (const line of lines) {
        if (line.startsWith('data: ')) {
          const data = JSON.parse(line.slice(6));
          yield data.content;
        }
      }
    }
  }
}

// Usage
const rag = new RAGClient('http://localhost:8000/api/v1/rag', 'your-api-key');

// Search
const results = await rag.search('machine learning', {
  searchType: 'hybrid',
  limit: 5,
  databases: ['media_db', 'notes']
});

// Q&A
const answer = await rag.ask('What is deep learning?');

// Streaming
for await (const chunk of rag.streamAgent('Explain quantum computing')) {
  process.stdout.write(chunk);
}
```

### Python

```python
import requests
import json
from typing import List, Dict, Optional, Generator
import sseclient

class RAGClient:
    def __init__(self, base_url: str, api_key: str):
        self.base_url = base_url
        self.api_key = api_key
        self.headers = {
            'X-API-KEY': api_key,
            'Content-Type': 'application/json'
        }
    
    def search(
        self,
        query: str,
        search_type: str = 'hybrid',
        limit: int = 10,
        databases: List[str] = None,
        keywords: List[str] = None
    ) -> Dict:
        """Perform a search query."""
        data = {
            'query': query,
            'search_type': search_type,
            'limit': limit,
            'databases': databases or ['media_db'],
            'keywords': keywords
        }
        
        response = requests.post(
            f'{self.base_url}/search',
            headers=self.headers,
            json=data
        )
        response.raise_for_status()
        return response.json()
    
    def ask(
        self,
        message: str,
        conversation_id: Optional[str] = None,
        databases: List[str] = None
    ) -> Dict:
        """Ask a question to the agent."""
        data = {
            'message': message,
            'conversation_id': conversation_id,
            'search_databases': databases or ['media_db']
        }
        
        response = requests.post(
            f'{self.base_url}/agent',
            headers=self.headers,
            json=data
        )
        response.raise_for_status()
        return response.json()
    
    def research(
        self,
        message: str,
        tools: List[str] = None,
        stream: bool = False
    ) -> Generator[str, None, None]:
        """Use the research agent with tools."""
        data = {
            'message': message,
            'mode': 'research',
            'tools': tools or ['web_search', 'reasoning'],
            'generation_config': {
                'stream': stream
            }
        }
        
        response = requests.post(
            f'{self.base_url}/agent/advanced',
            headers=self.headers,
            json=data,
            stream=stream
        )
        response.raise_for_status()
        
        if stream:
            client = sseclient.SSEClient(response)
            for event in client.events():
                if event.data:
                    data = json.loads(event.data)
                    yield data.get('content', '')
        else:
            yield response.json()['response']
    
    def advanced_search(
        self,
        query: str,
        strategy: str = 'vanilla',
        filters: Dict = None,
        weights: Dict = None
    ) -> Dict:
        """Perform an advanced search with full control."""
        data = {
            'query': query,
            'strategy': strategy,
            'search_config': {
                'search_type': 'hybrid',
                'limit': 20,
                'metadata_filters': filters or {},
                'include_full_content': True,
                'include_scores': True
            }
        }
        
        if weights:
            data['hybrid_config'] = weights
        
        response = requests.post(
            f'{self.base_url}/search/advanced',
            headers=self.headers,
            json=data
        )
        response.raise_for_status()
        return response.json()


# Usage example
if __name__ == '__main__':
    rag = RAGClient('http://localhost:8000/api/v1/rag', 'your-api-key')
    
    # Simple search
    results = rag.search('python tutorials', limit=5)
    for result in results['results']:
        print(f"- {result['title']}: {result['score']:.2f}")
    
    # Q&A conversation
    response = rag.ask("What is machine learning?")
    print(f"Answer: {response['response']}")
    
    # Continue conversation
    follow_up = rag.ask(
        "How does it differ from deep learning?",
        conversation_id=response['conversation_id']
    )
    print(f"Follow-up: {follow_up['response']}")
    
    # Research with streaming
    print("Research output:")
    for chunk in rag.research(
        "Latest advances in quantum computing",
        tools=['web_search', 'reasoning'],
        stream=True
    ):
        print(chunk, end='', flush=True)
```

### cURL Examples

```bash
# Simple search
curl -X POST http://localhost:8000/api/v1/rag/search \
  -H "X-API-KEY: your-api-key" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "machine learning",
    "limit": 5
  }'

# Advanced search with filters
curl -X POST http://localhost:8000/api/v1/rag/search/advanced \
  -H "X-API-KEY: your-api-key" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "neural networks",
    "strategy": "query_fusion",
    "search_config": {
      "search_type": "hybrid",
      "limit": 10,
      "metadata_filters": {
        "language": "en"
      }
    },
    "hybrid_config": {
      "semantic_weight": 0.8,
      "fulltext_weight": 0.2
    }
  }'

# Agent Q&A
curl -X POST http://localhost:8000/api/v1/rag/agent \
  -H "X-API-KEY: your-api-key" \
  -H "Content-Type: application/json" \
  -d '{
    "message": "Explain gradient descent"
  }'

# Research agent with tools
curl -X POST http://localhost:8000/api/v1/rag/agent/advanced \
  -H "X-API-KEY: your-api-key" \
  -H "Content-Type: application/json" \
  -d '{
    "message": "Compare BERT and GPT architectures",
    "mode": "research",
    "tools": ["web_search", "reasoning"]
  }'

# Health check
curl -X GET http://localhost:8000/api/v1/rag/health \
  -H "X-API-KEY: your-api-key"
```

## Best Practices

### 1. Query Optimization

**Do:**
- Keep queries concise and specific
- Use keywords for filtering when possible
- Choose appropriate search type for your use case

**Don't:**
- Send very long queries (>1000 chars)
- Use special characters unnecessarily
- Rely solely on semantic search for exact matches

### 2. Database Selection

Choose databases based on content type:

```javascript
// For technical documentation
databases: ["media_db"]

// For personal insights
databases: ["notes", "chat_history"]

// For comprehensive search
databases: ["media_db", "notes", "characters", "chat_history"]
```

### 3. Result Limits

Balance between coverage and performance:

```javascript
// Quick preview
limit: 5

// Standard search
limit: 10-20

// Comprehensive research
limit: 50-100
```

### 4. Conversation Management

Maintain conversation context:

```javascript
class ConversationManager {
  private conversations = new Map();
  
  async ask(userId: string, message: string) {
    let conversationId = this.conversations.get(userId);
    
    const response = await ragClient.ask(message, conversationId);
    
    // Store conversation ID for follow-ups
    this.conversations.set(userId, response.conversation_id);
    
    return response;
  }
  
  clearConversation(userId: string) {
    this.conversations.delete(userId);
  }
}
```

### 5. Streaming for Long Responses

Use streaming for better UX with long responses:

```javascript
async function streamResponse(message: string, onChunk: (chunk: string) => void) {
  const response = await fetch('/api/v1/rag/agent/advanced', {
    method: 'POST',
    headers: {
      'X-API-KEY': apiKey,
      'Content-Type': 'application/json'
    },
    body: JSON.stringify({
      message,
      generation_config: { stream: true }
    })
  });
  
  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  
  while (true) {
    const { done, value } = await reader.read();
    if (done) break;
    
    const chunk = decoder.decode(value);
    // Parse SSE format
    if (chunk.startsWith('data: ')) {
      const data = JSON.parse(chunk.slice(6));
      onChunk(data.content);
    }
  }
}

// Usage
streamResponse('Explain quantum computing', (chunk) => {
  document.getElementById('output').innerHTML += chunk;
});
```

## Error Handling

### HTTP Status Codes

| Code | Meaning | Action |
|------|---------|--------|
| 200 | Success | Process response |
| 400 | Bad Request | Check request format |
| 401 | Unauthorized | Verify API key |
| 404 | Not Found | Check endpoint URL |
| 422 | Validation Error | Review request parameters |
| 429 | Rate Limited | Implement backoff |
| 500 | Server Error | Retry with backoff |
| 503 | Service Unavailable | Check health endpoint |

### Error Response Format

```typescript
interface ErrorResponse {
  error: string;        // Error type
  message: string;      // Human-readable message
  details?: object;     // Additional error details
  request_id?: string;  // For support reference
}
```

### Error Handling Examples

```javascript
async function safeSearch(query: string): Promise<SearchResponse | null> {
  try {
    const response = await fetch('/api/v1/rag/search', {
      method: 'POST',
      headers: {
        'X-API-KEY': apiKey,
        'Content-Type': 'application/json'
      },
      body: JSON.stringify({ query })
    });
    
    if (!response.ok) {
      const error = await response.json();
      
      switch (response.status) {
        case 400:
          console.error('Invalid request:', error.message);
          return null;
        
        case 401:
          console.error('Authentication failed. Check API key.');
          throw new Error('Authentication failed');
        
        case 429:
          console.warn('Rate limited. Retrying in 5 seconds...');
          await new Promise(resolve => setTimeout(resolve, 5000));
          return safeSearch(query); // Retry
        
        case 500:
        case 503:
          console.error('Server error. Retrying with backoff...');
          await exponentialBackoff();
          return safeSearch(query); // Retry
        
        default:
          console.error('Unexpected error:', error);
          return null;
      }
    }
    
    return response.json();
    
  } catch (error) {
    console.error('Network error:', error);
    return null;
  }
}

function exponentialBackoff(attempt: number = 0): Promise<void> {
  const delay = Math.min(1000 * Math.pow(2, attempt), 30000);
  return new Promise(resolve => setTimeout(resolve, delay));
}
```

### Validation Errors

Handle validation errors gracefully:

```javascript
{
  "error": "validation_error",
  "message": "Request validation failed",
  "details": {
    "query": "Query must be between 1 and 1000 characters",
    "limit": "Limit must be between 1 and 100"
  }
}
```

## Rate Limiting & Performance

### Rate Limits

Default rate limits (configurable):
- **Search endpoints**: 100 requests/minute
- **Agent endpoints**: 30 requests/minute
- **Health check**: Unlimited

### Rate Limit Headers

```http
X-RateLimit-Limit: 100
X-RateLimit-Remaining: 95
X-RateLimit-Reset: 1704067200
```

### Implementing Rate Limit Handling

```javascript
class RateLimitedClient {
  private requestQueue: Array<() => Promise<any>> = [];
  private processing = false;
  private requestCount = 0;
  private resetTime = Date.now() + 60000;
  
  async request(fn: () => Promise<any>): Promise<any> {
    return new Promise((resolve, reject) => {
      this.requestQueue.push(async () => {
        try {
          // Check rate limit
          if (Date.now() > this.resetTime) {
            this.requestCount = 0;
            this.resetTime = Date.now() + 60000;
          }
          
          if (this.requestCount >= 100) {
            // Wait until reset
            const waitTime = this.resetTime - Date.now();
            await new Promise(r => setTimeout(r, waitTime));
            this.requestCount = 0;
          }
          
          this.requestCount++;
          const result = await fn();
          resolve(result);
        } catch (error) {
          reject(error);
        }
      });
      
      this.processQueue();
    });
  }
  
  private async processQueue() {
    if (this.processing) return;
    this.processing = true;
    
    while (this.requestQueue.length > 0) {
      const fn = this.requestQueue.shift();
      await fn();
      // Small delay between requests
      await new Promise(r => setTimeout(r, 100));
    }
    
    this.processing = false;
  }
}
```

### Performance Tips

1. **Batch Requests**: Combine multiple queries when possible
2. **Cache Results**: Implement client-side caching for repeated queries
3. **Use Appropriate Limits**: Don't request more results than needed
4. **Stream Large Responses**: Use SSE for long-form content
5. **Implement Pagination**: For large result sets

```javascript
// Pagination example
async function* paginatedSearch(query: string, totalLimit: number = 100) {
  const pageSize = 20;
  let offset = 0;
  
  while (offset < totalLimit) {
    const response = await fetch('/api/v1/rag/search/advanced', {
      method: 'POST',
      headers: {
        'X-API-KEY': apiKey,
        'Content-Type': 'application/json'
      },
      body: JSON.stringify({
        query,
        search_config: {
          limit: pageSize,
          offset: offset
        }
      })
    });
    
    const data = await response.json();
    yield data.results;
    
    if (data.results.length < pageSize) break;
    offset += pageSize;
  }
}

// Usage
for await (const batch of paginatedSearch('machine learning', 100)) {
  processBatch(batch);
}
```

### Caching Strategy

Implement intelligent caching:

```javascript
class CachedRAGClient {
  private cache = new Map();
  private cacheExpiry = 5 * 60 * 1000; // 5 minutes
  
  private getCacheKey(method: string, params: object): string {
    return `${method}:${JSON.stringify(params)}`;
  }
  
  async search(query: string, options: object = {}): Promise<any> {
    const cacheKey = this.getCacheKey('search', { query, ...options });
    
    // Check cache
    const cached = this.cache.get(cacheKey);
    if (cached && Date.now() - cached.timestamp < this.cacheExpiry) {
      return cached.data;
    }
    
    // Make request
    const data = await this.ragClient.search(query, options);
    
    // Cache result
    this.cache.set(cacheKey, {
      data,
      timestamp: Date.now()
    });
    
    return data;
  }
  
  clearCache() {
    this.cache.clear();
  }
}
```

## Common Use Cases

### 1. Document Q&A System

```javascript
async function documentQA(question: string) {
  // Search for relevant documents
  const searchResults = await ragClient.search(question, {
    searchType: 'hybrid',
    databases: ['media_db'],
    limit: 5
  });
  
  // Generate answer using agent
  const answer = await ragClient.ask(question);
  
  return {
    answer: answer.response,
    sources: answer.sources,
    relatedDocuments: searchResults.results
  };
}
```

### 2. Research Assistant

```javascript
async function researchTopic(topic: string) {
  const response = await fetch('/api/v1/rag/agent/advanced', {
    method: 'POST',
    headers: {
      'X-API-KEY': apiKey,
      'Content-Type': 'application/json'
    },
    body: JSON.stringify({
      message: `Research the topic: ${topic}. Provide a comprehensive overview.`,
      mode: 'research',
      tools: ['web_search', 'reasoning'],
      search_config: {
        limit: 20,
        databases: ['media_db', 'notes']
      },
      generation_config: {
        max_tokens: 2048,
        temperature: 0.7
      }
    })
  });
  
  return response.json();
}
```

### 3. Semantic Search Interface

```javascript
async function semanticSearch(query: string, filters: object = {}) {
  return await fetch('/api/v1/rag/search/advanced', {
    method: 'POST',
    headers: {
      'X-API-KEY': apiKey,
      'Content-Type': 'application/json'
    },
    body: JSON.stringify({
      query,
      strategy: 'hyde',  // Better semantic matching
      search_config: {
        search_type: 'semantic',
        limit: 15,
        metadata_filters: filters,
        include_scores: true
      },
      semantic_config: {
        similarity_threshold: 0.7,
        rerank: true
      }
    })
  }).then(r => r.json());
}
```

### 4. Conversational Learning Assistant

```javascript
class LearningAssistant {
  private conversationId: string | null = null;
  
  async startLesson(topic: string) {
    const response = await this.ask(
      `Let's learn about ${topic}. Start with the basics.`,
      "You are a patient teacher. Break down complex topics into simple concepts."
    );
    return response;
  }
  
  async askQuestion(question: string) {
    return await this.ask(question);
  }
  
  async getExample() {
    return await this.ask("Can you give me a practical example?");
  }
  
  async checkUnderstanding(concept: string) {
    return await this.ask(
      `Test my understanding of ${concept} with a question.`
    );
  }
  
  private async ask(message: string, systemPrompt?: string) {
    const data: any = {
      message,
      conversation_id: this.conversationId,
      search_databases: ['media_db', 'notes']
    };
    
    if (systemPrompt) {
      data.system_prompt = systemPrompt;
    }
    
    const response = await fetch('/api/v1/rag/agent', {
      method: 'POST',
      headers: {
        'X-API-KEY': apiKey,
        'Content-Type': 'application/json'
      },
      body: JSON.stringify(data)
    });
    
    const result = await response.json();
    this.conversationId = result.conversation_id;
    return result;
  }
  
  resetConversation() {
    this.conversationId = null;
  }
}
```

## Conclusion

The RAG API provides powerful search and question-answering capabilities with a clean, intuitive interface. Whether you need simple search, conversational Q&A, or advanced research capabilities, the API offers the flexibility and performance for production applications.

For implementation details and extending the RAG module, see the [RAG Developer Guide](../Development/RAG-Developer-Guide.md).