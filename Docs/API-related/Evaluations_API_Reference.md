# Evaluations API Reference

## Overview

The Evaluations API provides OpenAI-compatible endpoints for creating, managing, and running evaluations on AI model outputs. All endpoints require authentication via Bearer token.

**Base URL**: `http://localhost:8000`  
**API Version**: `v1`  
**Authentication**: Bearer token in Authorization header

## Authentication

All requests must include an Authorization header:
```http
Authorization: Bearer YOUR_API_KEY
```

In development mode, use: `default-secret-key-for-single-user`

## Endpoints

### Evaluations

#### Create Evaluation
`POST /v1/evals`

Creates a new evaluation definition.

**Request Body:**
```json
{
  "name": "string",           // Required, unique name
  "description": "string",    // Optional description
  "eval_type": "string",      // Required: model_graded|exact_match|fuzzy_match|includes
  "eval_spec": {              // Required, evaluation configuration
    "sub_type": "string",     // For model_graded: summarization|rag|response_quality
    "evaluator_model": "string", // LLM to use for evaluation
    "metrics": ["string"],    // Metrics to evaluate
    "threshold": 0.7,         // Pass/fail threshold (0-1)
    "custom_criteria": {}     // Optional custom evaluation criteria
  },
  "dataset": [                // Optional inline dataset
    {
      "input": {},            // Input data for evaluation
      "expected": {}          // Expected output
    }
  ],
  "dataset_id": "string",     // Optional reference to existing dataset
  "metadata": {}              // Optional metadata
}
```

**Response (201 Created):**
```json
{
  "id": "eval_xxxxxxxxxxxx",
  "object": "evaluation",
  "name": "string",
  "description": "string",
  "eval_type": "string",
  "eval_spec": {},
  "dataset_id": "string",
  "created_at": 1234567890,
  "updated_at": 1234567890,
  "created_by": "string",
  "metadata": {}
}
```

**Error Responses:**
- `400 Bad Request` - Invalid request body
- `401 Unauthorized` - Missing or invalid API key
- `409 Conflict` - Evaluation name already exists

---

#### List Evaluations
`GET /v1/evals`

Lists evaluations with pagination.

**Query Parameters:**
- `limit` (integer, 1-100, default: 20) - Number of items to return
- `after` (string) - Cursor for pagination
- `eval_type` (string) - Filter by evaluation type

**Response (200 OK):**
```json
{
  "object": "list",
  "data": [
    {
      "id": "eval_xxxxxxxxxxxx",
      "object": "evaluation",
      "name": "string",
      "eval_type": "string",
      "created_at": 1234567890
    }
  ],
  "has_more": false,
  "first_id": "eval_xxxxxxxxxxxx",
  "last_id": "eval_yyyyyyyyyyyy"
}
```

---

#### Get Evaluation
`GET /v1/evals/{eval_id}`

Retrieves a specific evaluation.

**Path Parameters:**
- `eval_id` (string) - Evaluation ID

**Response (200 OK):**
```json
{
  "id": "eval_xxxxxxxxxxxx",
  "object": "evaluation",
  "name": "string",
  "description": "string",
  "eval_type": "string",
  "eval_spec": {},
  "dataset_id": "string",
  "created_at": 1234567890,
  "updated_at": 1234567890,
  "created_by": "string",
  "metadata": {}
}
```

**Error Responses:**
- `404 Not Found` - Evaluation not found

---

#### Update Evaluation
`PATCH /v1/evals/{eval_id}`

Updates an evaluation definition.

**Path Parameters:**
- `eval_id` (string) - Evaluation ID

**Request Body:**
```json
{
  "description": "string",    // Optional
  "eval_spec": {},           // Optional
  "metadata": {}             // Optional
}
```

**Response (200 OK):** Updated evaluation object

**Error Responses:**
- `404 Not Found` - Evaluation not found
- `400 Bad Request` - Invalid update data

---

#### Delete Evaluation
`DELETE /v1/evals/{eval_id}`

Soft deletes an evaluation.

**Path Parameters:**
- `eval_id` (string) - Evaluation ID

**Response (204 No Content):** Empty response

**Error Responses:**
- `404 Not Found` - Evaluation not found

---

### Evaluation Runs

#### Create Run
`POST /v1/evals/{eval_id}/runs`

Starts an evaluation run.

**Path Parameters:**
- `eval_id` (string) - Evaluation ID

**Request Body:**
```json
{
  "target_model": "string",    // Model being evaluated
  "dataset_override": {        // Optional dataset override
    "samples": []
  },
  "config": {                  // Run configuration
    "temperature": 0.0,        // LLM temperature
    "max_workers": 4,          // Parallel workers
    "timeout_seconds": 300,    // Timeout per sample
    "batch_size": 10          // Batch size
  },
  "webhook_url": "string"      // Optional webhook for completion
}
```

**Response (202 Accepted):**
```json
{
  "id": "run_xxxxxxxxxxxx",
  "object": "evaluation.run",
  "eval_id": "eval_xxxxxxxxxxxx",
  "status": "pending",         // pending|running|completed|failed|cancelled
  "target_model": "string",
  "config": {},
  "created_at": 1234567890,
  "started_at": null,
  "completed_at": null,
  "progress": {
    "total_samples": 0,
    "completed_samples": 0,
    "failed_samples": 0
  }
}
```

**Error Responses:**
- `404 Not Found` - Evaluation not found
- `429 Too Many Requests` - Rate limit exceeded

---

#### List Runs
`GET /v1/evals/{eval_id}/runs`

Lists runs for an evaluation.

**Path Parameters:**
- `eval_id` (string) - Evaluation ID

**Query Parameters:**
- `limit` (integer, 1-100, default: 20)
- `after` (string) - Cursor for pagination
- `status` (string) - Filter by status

**Response (200 OK):** List of run objects

---

#### Get Run Status
`GET /v1/runs/{run_id}`

Gets run status and progress.

**Path Parameters:**
- `run_id` (string) - Run ID

**Response (200 OK):**
```json
{
  "id": "run_xxxxxxxxxxxx",
  "object": "evaluation.run",
  "eval_id": "eval_xxxxxxxxxxxx",
  "status": "running",
  "target_model": "string",
  "config": {},
  "progress": {
    "total_samples": 100,
    "completed_samples": 45,
    "failed_samples": 2,
    "current_batch": 5
  },
  "created_at": 1234567890,
  "started_at": 1234567891,
  "completed_at": null,
  "estimated_completion": 1234567950,
  "error_message": null
}
```

---

#### Get Run Results
`GET /v1/runs/{run_id}/results`

Gets results for a completed run.

**Path Parameters:**
- `run_id` (string) - Run ID

**Response (200 OK):**
```json
{
  "id": "run_xxxxxxxxxxxx",
  "eval_id": "eval_xxxxxxxxxxxx",
  "status": "completed",
  "started_at": 1234567890,
  "completed_at": 1234567920,
  "results": {
    "aggregate": {
      "mean_score": 0.85,
      "std_dev": 0.12,
      "min_score": 0.60,
      "max_score": 0.98,
      "pass_rate": 0.75,
      "total_samples": 100,
      "failed_samples": 5
    },
    "by_metric": {
      "fluency": {
        "mean": 0.88,
        "std": 0.10,
        "min": 0.65,
        "max": 0.98
      },
      "relevance": {
        "mean": 0.82,
        "std": 0.15,
        "min": 0.55,
        "max": 0.96
      }
    },
    "sample_results": [
      {
        "sample_id": "sample_0001",
        "scores": {
          "fluency": 0.85,
          "relevance": 0.90
        },
        "passed": true,
        "avg_score": 0.875
      }
    ],
    "failed_samples": []
  },
  "usage": {
    "total_tokens": 15000,
    "prompt_tokens": 10000,
    "completion_tokens": 5000
  },
  "duration_seconds": 30
}
```

**Error Responses:**
- `404 Not Found` - Run not found
- `400 Bad Request` - Run not completed

---

#### Cancel Run
`POST /v1/runs/{run_id}/cancel`

Cancels a running evaluation.

**Path Parameters:**
- `run_id` (string) - Run ID

**Response (200 OK):**
```json
{
  "id": "run_xxxxxxxxxxxx",
  "status": "cancelled"
}
```

---

#### Stream Run Progress
`GET /v1/runs/{run_id}/stream`

Streams run progress via Server-Sent Events.

**Path Parameters:**
- `run_id` (string) - Run ID

**Response:** Server-Sent Events stream

**Event Types:**
```
event: progress
data: {"total_samples": 100, "completed_samples": 50}

event: completed
data: {"results": {...}}

event: failed
data: {"error": "Error message"}

event: cancelled
data: {"message": "Run was cancelled"}

event: heartbeat
data: {"timestamp": "2024-01-01T00:00:00Z"}
```

---

### Datasets

#### Create Dataset
`POST /v1/datasets`

Creates a new dataset.

**Request Body:**
```json
{
  "name": "string",
  "description": "string",
  "samples": [
    {
      "input": {},
      "expected": {}
    }
  ],
  "metadata": {}
}
```

**Response (201 Created):**
```json
{
  "id": "dataset_xxxxxxxxxxxx",
  "object": "dataset",
  "name": "string",
  "description": "string",
  "sample_count": 100,
  "created_at": 1234567890,
  "created_by": "string",
  "metadata": {}
}
```

---

#### List Datasets
`GET /v1/datasets`

Lists datasets with pagination.

**Query Parameters:**
- `limit` (integer, 1-100, default: 20)
- `after` (string) - Cursor for pagination

**Response (200 OK):** List of dataset objects

---

#### Get Dataset
`GET /v1/datasets/{dataset_id}`

Gets a specific dataset.

**Path Parameters:**
- `dataset_id` (string) - Dataset ID

**Response (200 OK):**
```json
{
  "id": "dataset_xxxxxxxxxxxx",
  "object": "dataset",
  "name": "string",
  "description": "string",
  "samples": [
    {
      "input": {},
      "expected": {}
    }
  ],
  "sample_count": 100,
  "created_at": 1234567890,
  "created_by": "string",
  "metadata": {}
}
```

---

#### Delete Dataset
`DELETE /v1/datasets/{dataset_id}`

Deletes a dataset.

**Path Parameters:**
- `dataset_id` (string) - Dataset ID

**Response (204 No Content):** Empty response

---

## Evaluation Types

### model_graded
Uses an LLM to evaluate outputs. Subtypes:

#### summarization
Evaluates text summaries using G-Eval metrics.

**Required in eval_spec:**
- `sub_type`: "summarization"
- `evaluator_model`: LLM model name
- `metrics`: Array of ["fluency", "consistency", "relevance", "coherence"]

**Sample format:**
```json
{
  "input": {
    "source_text": "Original document",
    "summary": "Summary to evaluate"
  }
}
```

#### rag
Evaluates RAG system outputs.

**Required in eval_spec:**
- `sub_type`: "rag"
- `evaluator_model`: LLM model name
- `metrics`: Array of ["relevance", "faithfulness", "answer_similarity", "context_precision", "context_recall"]

**Sample format:**
```json
{
  "input": {
    "query": "User question",
    "contexts": ["Retrieved context 1", "Retrieved context 2"],
    "response": "Generated response"
  },
  "expected": {
    "answer": "Ground truth answer"
  }
}
```

#### response_quality
Evaluates general response quality.

**Required in eval_spec:**
- `sub_type`: "response_quality"
- `evaluator_model`: LLM model name
- `custom_criteria`: Dictionary of criteria

**Sample format:**
```json
{
  "input": {
    "prompt": "User prompt",
    "response": "Generated response",
    "expected_format": "Optional format specification"
  }
}
```

### exact_match
Checks exact string match (case-insensitive).

**Sample format:**
```json
{
  "input": {"output": "Generated text"},
  "expected": {"output": "Expected text"}
}
```

### fuzzy_match
Uses string similarity scoring.

**Required in eval_spec:**
- `threshold`: Similarity threshold (0-1)

**Sample format:**
```json
{
  "input": {"output": "Generated text"},
  "expected": {"output": "Similar expected text"}
}
```

### includes
Checks if output contains expected items.

**Sample format:**
```json
{
  "input": {"output": "Generated text"},
  "expected": {"includes": ["keyword1", "keyword2"]}
}
```

## Error Responses

All error responses follow this format:

```json
{
  "error": {
    "message": "Human-readable error message",
    "type": "error_type",
    "param": "parameter_name",
    "code": "error_code"
  }
}
```

**Error Types:**
- `authentication_error` - Invalid or missing API key
- `invalid_request_error` - Invalid request parameters
- `not_found_error` - Resource not found
- `server_error` - Internal server error
- `rate_limit_error` - Rate limit exceeded

**HTTP Status Codes:**
- `200 OK` - Successful GET request
- `201 Created` - Successful POST creating resource
- `202 Accepted` - Request accepted for async processing
- `204 No Content` - Successful DELETE
- `400 Bad Request` - Invalid request
- `401 Unauthorized` - Authentication required
- `404 Not Found` - Resource not found
- `429 Too Many Requests` - Rate limit exceeded
- `500 Internal Server Error` - Server error

## Rate Limits

Default rate limits:
- Create operations: 100/minute
- Read operations: 100/minute
- Run operations: 50/minute

Rate limit headers:
```http
X-RateLimit-Limit: 100
X-RateLimit-Remaining: 95
X-RateLimit-Reset: 1234567890
```

## Webhooks

When a `webhook_url` is provided for a run, the following payload is sent on completion:

```json
{
  "event": "run.completed",  // or run.failed
  "run_id": "run_xxxxxxxxxxxx",
  "eval_id": "eval_xxxxxxxxxxxx",
  "status": "completed",
  "completed_at": 1234567890,
  "results_url": "/v1/runs/run_xxxxxxxxxxxx/results",
  "summary": {
    "mean_score": 0.85,
    "pass_rate": 0.75
  }
}
```

## Code Examples

### Python
```python
import requests

API_KEY = "your-api-key"
BASE_URL = "http://localhost:8000"

# Create evaluation
response = requests.post(
    f"{BASE_URL}/v1/evals",
    json={
        "name": "my_eval",
        "eval_type": "exact_match",
        "eval_spec": {"threshold": 1.0},
        "dataset": [
            {"input": {"output": "test"}, "expected": {"output": "test"}}
        ]
    },
    headers={"Authorization": f"Bearer {API_KEY}"}
)
eval_id = response.json()["id"]

# Run evaluation
response = requests.post(
    f"{BASE_URL}/v1/evals/{eval_id}/runs",
    json={"target_model": "test", "config": {"temperature": 0}},
    headers={"Authorization": f"Bearer {API_KEY}"}
)
run_id = response.json()["id"]

# Get results
response = requests.get(
    f"{BASE_URL}/v1/runs/{run_id}/results",
    headers={"Authorization": f"Bearer {API_KEY}"}
)
print(response.json())
```

### cURL
```bash
# Create evaluation
curl -X POST http://localhost:8000/v1/evals \
  -H "Authorization: Bearer $API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"name": "test", "eval_type": "exact_match", "eval_spec": {"threshold": 1.0}}'

# Run evaluation
curl -X POST http://localhost:8000/v1/evals/eval_xxx/runs \
  -H "Authorization: Bearer $API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"target_model": "test", "config": {}}'

# Get results
curl http://localhost:8000/v1/runs/run_xxx/results \
  -H "Authorization: Bearer $API_KEY"
```

### JavaScript
```javascript
const API_KEY = 'your-api-key';
const BASE_URL = 'http://localhost:8000';

// Create evaluation
const evalResponse = await fetch(`${BASE_URL}/v1/evals`, {
  method: 'POST',
  headers: {
    'Authorization': `Bearer ${API_KEY}`,
    'Content-Type': 'application/json'
  },
  body: JSON.stringify({
    name: 'my_eval',
    eval_type: 'exact_match',
    eval_spec: { threshold: 1.0 },
    dataset: [
      { input: { output: 'test' }, expected: { output: 'test' } }
    ]
  })
});
const { id: evalId } = await evalResponse.json();

// Run evaluation
const runResponse = await fetch(`${BASE_URL}/v1/evals/${evalId}/runs`, {
  method: 'POST',
  headers: {
    'Authorization': `Bearer ${API_KEY}`,
    'Content-Type': 'application/json'
  },
  body: JSON.stringify({
    target_model: 'test',
    config: { temperature: 0 }
  })
});
const { id: runId } = await runResponse.json();
```

## OpenAPI Specification

The complete OpenAPI specification is available at:
- JSON: `http://localhost:8000/openapi.json`
- Interactive docs: `http://localhost:8000/docs`
- ReDoc: `http://localhost:8000/redoc`