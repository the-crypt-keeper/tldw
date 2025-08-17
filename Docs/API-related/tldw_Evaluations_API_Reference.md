# tldw_server Evaluations API Reference

## Overview

The tldw_server Evaluations API provides comprehensive evaluation capabilities for assessing the quality of AI-generated content, including summaries, RAG responses, and general text generation. The API supports multiple evaluation frameworks and metrics, with built-in rate limiting, health monitoring, and metrics collection.

**Base URL**: `/api/v1/evaluations`  
**Authentication**: API key (single-user mode uses default key)

## Production Features

- **Rate Limiting**: 10 req/min (standard), 5 req/min (batch)
- **Input Validation**: Comprehensive sanitization and size limits
- **Health Monitoring**: Dedicated health endpoints
- **Metrics Collection**: Prometheus-compatible metrics
- **Circuit Breakers**: Automatic failure recovery
- **Load Tested**: Supports 100+ concurrent users

## Rate Limiting

All evaluation endpoints are rate-limited:
- **Standard endpoints**: 10 requests/minute/IP
- **Batch endpoints**: 5 requests/minute/IP
- **429 response** includes `Retry-After` header

## Core Endpoints

### 1. G-Eval Summarization

**Endpoint**: `POST /evaluations/geval`

Evaluates summary quality using G-Eval metrics (fluency, consistency, relevance, coherence).

**Request**:
```json
{
  "source_text": "string",  // 10-100,000 chars
  "summary": "string",       // 10-50,000 chars
  "metrics": ["fluency", "consistency", "relevance", "coherence"],
  "api_name": "openai",
  "save_results": false
}
```

**Response** (200 OK):
```json
{
  "metrics": {
    "fluency": {
      "name": "fluency",
      "score": 0.85,
      "raw_score": 2.55,
      "explanation": "Grammar and readability assessment"
    }
  },
  "average_score": 0.82,
  "summary_assessment": "High-quality summary",
  "evaluation_time": 2.34
}
```

### 2. RAG Evaluation

**Endpoint**: `POST /evaluations/rag`

Evaluates retrieval-augmented generation quality.

**Request**:
```json
{
  "query": "string",
  "retrieved_contexts": ["string"],  // 1-20 items
  "generated_response": "string",
  "ground_truth": "string",  // optional
  "metrics": ["relevance", "faithfulness", "answer_similarity"]
}
```

### 3. Response Quality

**Endpoint**: `POST /evaluations/response-quality`

Evaluates general response quality and format compliance.

**Request**:
```json
{
  "prompt": "string",
  "response": "string",
  "expected_format": "string",
  "evaluation_criteria": {
    "accuracy": "Factually correct"
  }
}
```

### 4. Batch Evaluation

**Endpoint**: `POST /evaluations/batch`

Evaluates multiple items in parallel (max 100 items, 10MB total).

**Request**:
```json
{
  "evaluation_type": "geval",
  "items": [{...}],
  "parallel_workers": 4
}
```

### 5. Evaluation History

**Endpoint**: `POST /evaluations/history`

Retrieves past evaluations with filtering.

### 6. Custom Metrics

**Endpoint**: `POST /evaluations/custom-metric`

Evaluates using custom-defined metrics.

### 7. Compare Evaluations

**Endpoint**: `POST /evaluations/compare`

Compares multiple evaluation results.

## Health & Monitoring

### Health Check

**Endpoint**: `GET /health/evaluations`

Returns comprehensive health status including:
- Database connectivity
- Circuit breaker states
- Embeddings availability
- Recent metrics

### Metrics

**Endpoint**: `GET /evaluations/metrics`

Prometheus-compatible metrics including:
- Request counts and latencies
- Evaluation success/failure rates
- Circuit breaker states
- Resource utilization

## Input Validation

All inputs are sanitized:
- HTML/script tags removed
- HTML entities escaped
- Control characters filtered
- Size limits enforced

**Size Limits**:
- Source text: 100KB
- Summary/Response: 50KB
- Context chunks: 20KB each
- Batch: 10MB total

## Supported Providers

- `openai` - OpenAI GPT models
- `anthropic` - Claude models
- `google` - Google AI
- `cohere` - Cohere
- `mistral` - Mistral AI
- `groq` - Groq inference
- `openrouter` - OpenRouter
- `deepseek` - DeepSeek
- `local-llm` - Local models

## Error Handling

- **400**: Invalid input
- **429**: Rate limit exceeded
- **500**: Internal error
- **503**: Service unavailable

## Performance Targets

**Response Times (p99)**:
- G-Eval: <2 seconds
- RAG: <3 seconds
- Batch (10): <15 seconds

**Throughput**:
- 100 concurrent users
- 1000 requests/minute
- <2s p99 latency

## Examples

### cURL
```bash
curl -X POST http://localhost:8000/api/v1/evaluations/geval \
  -H "Content-Type: application/json" \
  -d '{"source_text": "...", "summary": "..."}'
```

### Python
```python
import requests

response = requests.post(
    "http://localhost:8000/api/v1/evaluations/rag",
    json={
        "query": "What is ML?",
        "retrieved_contexts": ["..."],
        "generated_response": "..."
    }
)
```

### Load Testing
```bash
# Using the provided Locust script
locust -f load_test_evaluations.py \
  --host=http://localhost:8000 \
  --users 100 --spawn-rate 10
```

## Migration from v0.x

- Rate limiting now enforced
- Stricter input validation
- New metrics endpoint
- Health endpoint relocated
- Batch size limits

## Best Practices

1. Implement client-side rate limiting
2. Use batch endpoint for multiple items
3. Cache evaluation results
4. Handle 429 responses with backoff
5. Monitor health endpoints
6. Keep inputs within size limits

## Support

- Check `/health/evaluations` for status
- Review `/evaluations/metrics` for performance
- Include `evaluation_id` in support requests