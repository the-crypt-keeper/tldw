# tldw_server Unified Evaluations API Reference

## Overview

The tldw_server Evaluations API provides a comprehensive, OpenAI-compatible evaluation framework for assessing AI-generated content quality. This unified API combines industry-standard evaluation patterns with tldw-specific features for advanced content assessment.

**Version**: 1.0.0 (Unified)  
**Base URL**: `/api/v1/evaluations`  
**Standards**: OpenAI Evals compatible with extensions

## Architecture

### Core Components

```
┌─────────────────────────────────────────────────────────────┐
│                        API Layer                             │
│  /api/v1/evaluations/* (Unified OpenAI-compatible + tldw)    │
└─────────────────────────────────────────────────────────────┘
                               │
┌─────────────────────────────────────────────────────────────┐
│                    Service Layer                             │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐      │
│  │ Evaluation   │  │   Dataset    │  │     Run      │      │
│  │  Manager     │  │   Manager    │  │   Manager    │      │
│  └──────────────┘  └──────────────┘  └──────────────┘      │
│                                                              │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐      │
│  │   Metrics    │  │   Webhooks   │  │Rate Limiting │      │
│  │   Service    │  │   Manager    │  │   Service    │      │
│  └──────────────┘  └──────────────┘  └──────────────┘      │
└─────────────────────────────────────────────────────────────┘
                               │
┌─────────────────────────────────────────────────────────────┐
│                    Evaluation Engines                        │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐      │
│  │   G-Eval     │  │     RAG      │  │   Response   │      │
│  │   Engine     │  │  Evaluator   │  │   Quality    │      │
│  └──────────────┘  └──────────────┘  └──────────────┘      │
│                                                              │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐      │
│  │Model-Graded  │  │Exact Match   │  │   Custom     │      │
│  │  Evaluator   │  │  Evaluator   │  │  Evaluator   │      │
│  └──────────────┘  └──────────────┘  └──────────────┘      │
└─────────────────────────────────────────────────────────────┘
                               │
┌─────────────────────────────────────────────────────────────┐
│                      Data Layer                              │
│  ┌────────────────────────────────────────────────────┐     │
│  │         Unified Evaluations Database               │     │
│  │  - evaluations table (definitions)                 │     │
│  │  - runs table (execution records)                  │     │
│  │  - datasets table (test data)                      │     │
│  │  - results table (evaluation outputs)              │     │
│  │  - metrics table (performance data)                │     │
│  └────────────────────────────────────────────────────┘     │
└─────────────────────────────────────────────────────────────┘
```

## Authentication

The API supports multiple authentication modes:

### Single-User Mode (Default)
```bash
# Using API Key header
curl -H "X-API-KEY: your-api-key" https://api.example.com/api/v1/evaluations

# Using Bearer token
curl -H "Authorization: Bearer your-api-key" https://api.example.com/api/v1/evaluations
```

### Multi-User Mode (JWT)
```bash
# Using JWT token
curl -H "Authorization: Bearer eyJhbGc..." https://api.example.com/api/v1/evaluations
```

### OpenAI Compatibility
```bash
# Using OpenAI-style key
curl -H "Authorization: Bearer sk-..." https://api.example.com/api/v1/evaluations
```

## Rate Limiting

### Global Limits
- **Standard Operations**: 60 requests/minute
- **Evaluation Runs**: 10 requests/minute
- **Batch Operations**: 5 requests/minute
- **Burst Protection**: 10 requests/second max

### User-Based Limits (Multi-User Mode)
| Tier       | Requests/Min | Tokens/Min | Batch Size | Cost/Month |
|------------|-------------|------------|------------|------------|
| Free       | 10          | 10,000     | 10         | $0         |
| Basic      | 30          | 50,000     | 50         | $10        |
| Premium    | 100         | 200,000    | 200        | $50        |
| Enterprise | Unlimited   | Unlimited  | 1000       | Custom     |

## Core API Endpoints

### Evaluation Management

#### Create Evaluation
`POST /api/v1/evaluations`

Creates a new evaluation definition that can be run multiple times.

**Request:**
```json
{
  "name": "summary_quality_eval",
  "description": "Evaluate summary quality using multiple metrics",
  "eval_type": "model_graded",
  "eval_spec": {
    "metrics": ["fluency", "consistency", "relevance", "coherence"],
    "thresholds": {
      "pass": 0.7,
      "excellent": 0.9
    },
    "model": "gpt-4",
    "temperature": 0.3
  },
  "dataset_id": "dataset_123",
  "metadata": {
    "project": "content_quality",
    "version": "1.0"
  }
}
```

**Response (201 Created):**
```json
{
  "id": "eval_abc123",
  "name": "summary_quality_eval",
  "description": "Evaluate summary quality using multiple metrics",
  "eval_type": "model_graded",
  "eval_spec": {...},
  "dataset_id": "dataset_123",
  "created_at": 1234567890,
  "created_by": "user_123",
  "metadata": {...}
}
```

#### List Evaluations
`GET /api/v1/evaluations`

**Query Parameters:**
- `limit`: 1-100 (default: 20)
- `after`: Cursor for pagination
- `eval_type`: Filter by type
- `project`: Filter by project metadata

**Response:**
```json
{
  "object": "list",
  "data": [...],
  "has_more": true,
  "first_id": "eval_abc123",
  "last_id": "eval_xyz789"
}
```

#### Get Evaluation
`GET /api/v1/evaluations/{eval_id}`

#### Update Evaluation
`PATCH /api/v1/evaluations/{eval_id}`

#### Delete Evaluation
`DELETE /api/v1/evaluations/{eval_id}`

### Evaluation Runs

#### Create Run
`POST /api/v1/evaluations/{eval_id}/runs`

Starts an asynchronous evaluation run.

**Request:**
```json
{
  "target_model": "gpt-3.5-turbo",
  "dataset_override": null,
  "config": {
    "temperature": 0.7,
    "max_workers": 4,
    "timeout_seconds": 300
  },
  "webhook_url": "https://example.com/webhook"
}
```

**Response (202 Accepted):**
```json
{
  "id": "run_def456",
  "eval_id": "eval_abc123",
  "status": "pending",
  "target_model": "gpt-3.5-turbo",
  "created_at": 1234567890,
  "progress": {
    "completed_samples": 0,
    "total_samples": 100
  }
}
```

#### Get Run Status
`GET /api/v1/runs/{run_id}`

#### Stream Run Progress (SSE)
`GET /api/v1/runs/{run_id}/stream`

Server-sent events for real-time progress updates.

**Event Types:**
- `progress`: Sample completion updates
- `completed`: Run finished successfully
- `failed`: Run encountered error
- `cancelled`: Run was cancelled

#### Get Run Results
`GET /api/v1/runs/{run_id}/results`

#### Cancel Run
`POST /api/v1/runs/{run_id}/cancel`

### Dataset Management

#### Create Dataset
`POST /api/v1/datasets`

**Request:**
```json
{
  "name": "qa_test_set",
  "description": "Question-answer pairs for testing",
  "samples": [
    {
      "input": "What is the capital of France?",
      "expected": "Paris",
      "metadata": {"difficulty": "easy"}
    }
  ]
}
```

#### List Datasets
`GET /api/v1/datasets`

#### Get Dataset
`GET /api/v1/datasets/{dataset_id}`

#### Delete Dataset
`DELETE /api/v1/datasets/{dataset_id}`

## Specialized Evaluation Endpoints

### G-Eval Summarization
`POST /api/v1/evaluations/geval`

Evaluates summary quality using Google's G-Eval framework.

**Request:**
```json
{
  "source_text": "Original long document...",
  "summary": "Concise summary...",
  "metrics": ["fluency", "consistency", "relevance", "coherence"],
  "api_name": "openai",
  "save_results": true
}
```

**Response:**
```json
{
  "metrics": {
    "fluency": {
      "score": 0.85,
      "raw_score": 2.55,
      "explanation": "Well-structured with minor grammatical issues"
    },
    "consistency": {
      "score": 0.92,
      "raw_score": 4.6,
      "explanation": "Highly consistent with source material"
    }
  },
  "average_score": 0.88,
  "summary_assessment": "High-quality summary with excellent factual accuracy",
  "evaluation_time": 2.34,
  "metadata": {
    "evaluation_id": "eval_result_789"
  }
}
```

### RAG Evaluation
`POST /api/v1/evaluations/rag`

Evaluates retrieval-augmented generation quality.

**Request:**
```json
{
  "query": "What are the benefits of exercise?",
  "retrieved_contexts": [
    "Exercise improves cardiovascular health...",
    "Regular physical activity boosts mood..."
  ],
  "generated_response": "Exercise provides numerous benefits including...",
  "ground_truth": "Expected answer for comparison",
  "metrics": ["relevance", "faithfulness", "answer_similarity", "context_precision"]
}
```

**Response:**
```json
{
  "metrics": {
    "relevance": {"score": 0.89, "explanation": "Highly relevant to query"},
    "faithfulness": {"score": 0.95, "explanation": "Well-grounded in contexts"},
    "answer_similarity": {"score": 0.82, "explanation": "Close to ground truth"},
    "context_precision": {"score": 0.78, "explanation": "Good context selection"}
  },
  "overall_score": 0.86,
  "retrieval_quality": 0.78,
  "generation_quality": 0.89,
  "suggestions": [
    "Consider adding more diverse contexts",
    "Response could be more concise"
  ]
}
```

### Response Quality
`POST /api/v1/evaluations/response-quality`

Evaluates general response quality and format compliance.

**Request:**
```json
{
  "prompt": "Write a professional email...",
  "response": "Dear colleague...",
  "expected_format": "email",
  "evaluation_criteria": {
    "professionalism": "Appropriate tone and language",
    "completeness": "All required elements present",
    "clarity": "Clear and unambiguous"
  }
}
```

### Batch Evaluation
`POST /api/v1/evaluations/batch`

Process multiple evaluations in parallel.

**Request:**
```json
{
  "evaluation_type": "geval",
  "items": [...],
  "parallel_workers": 4,
  "continue_on_error": true
}
```

### Custom Metrics
`POST /api/v1/evaluations/custom-metric`

Define and run custom evaluation metrics.

**Request:**
```json
{
  "name": "technical_accuracy",
  "description": "Evaluate technical content accuracy",
  "evaluation_prompt": "Rate the technical accuracy of this response...",
  "input_data": {...},
  "scoring_criteria": {...}
}
```

## Webhook Management

### Register Webhook
`POST /api/v1/evaluations/webhooks`

```json
{
  "url": "https://example.com/webhook",
  "events": ["evaluation.completed", "evaluation.failed"],
  "secret": "webhook_secret_key"
}
```

### Webhook Events

Events are sent as POST requests with HMAC-SHA256 signatures.

**Headers:**
- `X-Webhook-Signature`: HMAC-SHA256 signature
- `X-Webhook-Timestamp`: Unix timestamp
- `X-Webhook-Event`: Event type

**Payload Example:**
```json
{
  "event": "evaluation.completed",
  "timestamp": 1234567890,
  "data": {
    "evaluation_id": "eval_123",
    "run_id": "run_456",
    "results": {...}
  }
}
```

## Metrics & Monitoring

### Health Check
`GET /api/v1/evaluations/health`

```json
{
  "status": "healthy",
  "version": "1.0.0",
  "uptime": 3600,
  "database": "connected",
  "rate_limit": {
    "requests_remaining": 50,
    "reset_at": 1234567890
  }
}
```

### Prometheus Metrics
`GET /api/v1/evaluations/metrics`

Exports metrics in Prometheus format:
- `evaluation_requests_total`
- `evaluation_duration_seconds`
- `evaluation_errors_total`
- `evaluation_queue_depth`
- `evaluation_cost_dollars`

## Error Handling

All errors follow a consistent format:

```json
{
  "error": {
    "message": "Detailed error description",
    "type": "error_category",
    "param": "field_name",
    "code": "ERROR_CODE"
  }
}
```

### Error Types
- `authentication_error`: Auth failures
- `invalid_request_error`: Validation errors
- `rate_limit_error`: Rate limit exceeded
- `not_found_error`: Resource not found
- `server_error`: Internal errors

### HTTP Status Codes
- `200`: Success
- `201`: Created
- `202`: Accepted (async operation)
- `400`: Bad Request
- `401`: Unauthorized
- `403`: Forbidden
- `404`: Not Found
- `429`: Too Many Requests
- `500`: Internal Server Error

## Migration Guide

### From Legacy Endpoints

#### Old tldw endpoints → New unified endpoints
- `/api/v1/evaluations/geval` → `/api/v1/evaluations/geval` (unchanged)
- `/api/v1/evaluations/rag` → `/api/v1/evaluations/rag` (unchanged)
- `/api/v1/evaluations/batch` → `/api/v1/evaluations/batch` (unchanged)

#### Old OpenAI endpoints → New unified endpoints
- `/api/v1/evals` → `/api/v1/evaluations`
- `/api/v1/evals/{id}/runs` → `/api/v1/evaluations/{id}/runs`
- `/api/v1/runs/{id}` → `/api/v1/runs/{id}` (unchanged)

### Breaking Changes
1. Webhook event names standardized (see Webhook Events section)
2. Rate limit headers now use `X-RateLimit-*` prefix
3. Dataset samples format standardized to OpenAI format

### Deprecation Timeline
- **v0.x**: Both old and new endpoints available
- **v1.0**: Old endpoints deprecated with warnings
- **v2.0**: Old endpoints removed

## SDK Examples

### Python
```python
from tldw_server import EvaluationClient

client = EvaluationClient(api_key="your-key")

# Create evaluation
eval = client.evaluations.create(
    name="my_eval",
    eval_type="model_graded",
    eval_spec={...}
)

# Run evaluation
run = client.evaluations.run(
    eval_id=eval.id,
    target_model="gpt-3.5-turbo"
)

# Stream progress
for event in client.runs.stream(run.id):
    print(f"Progress: {event.data}")
```

### JavaScript/TypeScript
```typescript
import { EvaluationClient } from '@tldw/evaluations';

const client = new EvaluationClient({ apiKey: 'your-key' });

// Create and run evaluation
const eval = await client.evaluations.create({
  name: 'my_eval',
  evalType: 'model_graded',
  evalSpec: {...}
});

const run = await client.evaluations.run(eval.id, {
  targetModel: 'gpt-3.5-turbo'
});

// Subscribe to progress
const stream = client.runs.stream(run.id);
stream.on('progress', (data) => console.log(data));
```

## Best Practices

### Evaluation Design
1. **Choose appropriate metrics** for your use case
2. **Use consistent datasets** for comparable results
3. **Set reasonable thresholds** based on baseline testing
4. **Version your evaluations** for reproducibility

### Performance Optimization
1. **Batch similar evaluations** to reduce overhead
2. **Use parallel workers** for large datasets
3. **Set appropriate timeouts** to prevent hanging
4. **Cache evaluation results** when possible

### Cost Management
1. **Monitor token usage** via metrics endpoint
2. **Use smaller models** for initial testing
3. **Implement sampling** for large datasets
4. **Set spending limits** via configuration

### Security
1. **Rotate API keys** regularly
2. **Use webhook secrets** for verification
3. **Implement IP allowlisting** for production
4. **Audit evaluation access** via logs

## Support

### Resources
- GitHub Issues: https://github.com/tldw/tldw_server/issues
- Documentation: https://docs.tldw.ai/evaluations
- Discord Community: https://discord.gg/tldw

### Feature Requests
Submit feature requests via GitHub issues with the `enhancement` label.

### Contributing
See CONTRIBUTING.md for guidelines on contributing to the evaluation module.

---

*Last Updated: 2024*  
*Version: 1.0.0*  
*Status: Unified Implementation*