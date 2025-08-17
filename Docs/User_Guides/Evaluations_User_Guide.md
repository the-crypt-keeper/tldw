# Evaluations Module User Guide

## Table of Contents
- [Overview](#overview)
- [Getting Started](#getting-started)
- [Built-in Evaluation Types](#built-in-evaluation-types)
- [Creating Custom Evaluations](#creating-custom-evaluations)
- [Uploading Vendor Evaluations](#uploading-vendor-evaluations)
- [Running Evaluations](#running-evaluations)
- [Interpreting Results](#interpreting-results)
- [Best Practices](#best-practices)
- [Examples](#examples)

## Overview

The Evaluations module in tldw_server provides a comprehensive framework for evaluating AI model outputs, including summarizations, RAG systems, and custom metrics. It follows the OpenAI Evals API specification for compatibility with existing tools and workflows.

### Key Features
- **Multiple evaluation types**: G-Eval, exact match, fuzzy match, includes, and model-graded evaluations
- **Async processing**: Run evaluations in the background with progress tracking
- **Dataset management**: Store and reuse evaluation datasets
- **Flexible metrics**: Define custom evaluation criteria
- **OpenAI-compatible API**: Works with existing evaluation tools

## Getting Started

### Prerequisites
- tldw_server running (default: http://localhost:8000)
- API key for authentication (default in dev mode: `default-secret-key-for-single-user`)
- (Optional) LLM API key for model-graded evaluations (OpenAI, Anthropic, etc.)

### Quick Setup

1. **Verify the server is running:**
```bash
curl http://localhost:8000/health
```

2. **Set your authentication:**
```bash
export TLDW_API_KEY="default-secret-key-for-single-user"  # or your API key
```

3. **Test with a simple evaluation:**
```bash
curl -X POST http://localhost:8000/v1/evals \
  -H "Authorization: Bearer $TLDW_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "name": "my_first_eval",
    "eval_type": "exact_match",
    "eval_spec": {"threshold": 1.0},
    "dataset": [
      {"input": {"output": "hello"}, "expected": {"output": "hello"}}
    ]
  }'
```

## Built-in Evaluation Types

### 1. G-Eval (Summarization)
Evaluates text summarization quality using four metrics:
- **Fluency**: Grammar, spelling, punctuation (1-3 scale)
- **Consistency**: Factual alignment with source (1-5 scale)
- **Relevance**: Selection of important content (1-5 scale)
- **Coherence**: Structure and organization (1-5 scale)

**Example:**
```json
{
  "name": "summarization_eval",
  "eval_type": "model_graded",
  "eval_spec": {
    "sub_type": "summarization",
    "evaluator_model": "openai",
    "metrics": ["fluency", "consistency", "relevance", "coherence"],
    "threshold": 0.7
  },
  "dataset": [
    {
      "input": {
        "source_text": "The original long document text...",
        "summary": "A concise summary of the document."
      }
    }
  ]
}
```

### 2. Exact Match
Checks if outputs exactly match expected values (case-insensitive).

**Example:**
```json
{
  "name": "exact_match_eval",
  "eval_type": "exact_match",
  "eval_spec": {"threshold": 1.0},
  "dataset": [
    {"input": {"output": "Paris"}, "expected": {"output": "paris"}}
  ]
}
```

### 3. Fuzzy Match
Uses string similarity for approximate matching.

**Example:**
```json
{
  "name": "fuzzy_eval",
  "eval_type": "fuzzy_match",
  "eval_spec": {"threshold": 0.8},
  "dataset": [
    {"input": {"output": "The cat sat on the mat"}, 
     "expected": {"output": "The cat was sitting on the mat"}}
  ]
}
```

### 4. Includes
Checks if output contains expected keywords or phrases.

**Example:**
```json
{
  "name": "includes_eval",
  "eval_type": "includes",
  "eval_spec": {"threshold": 0.7},
  "dataset": [
    {
      "input": {"output": "Paris is the capital of France"},
      "expected": {"includes": ["Paris", "capital", "France"]}
    }
  ]
}
```

### 5. RAG Evaluation
Evaluates Retrieval-Augmented Generation systems:
- **Relevance**: Response relevance to query
- **Faithfulness**: Grounding in retrieved contexts
- **Answer Similarity**: Similarity to ground truth
- **Context Precision**: Quality of retrieved contexts
- **Context Recall**: Coverage of necessary information

**Example:**
```json
{
  "name": "rag_eval",
  "eval_type": "model_graded",
  "eval_spec": {
    "sub_type": "rag",
    "evaluator_model": "openai",
    "metrics": ["relevance", "faithfulness", "context_precision"],
    "threshold": 0.7
  },
  "dataset": [
    {
      "input": {
        "query": "What is the capital of France?",
        "contexts": ["Paris is the capital city of France.", "France is in Europe."],
        "response": "The capital of France is Paris."
      },
      "expected": {"answer": "Paris"}
    }
  ]
}
```

### 6. Response Quality
Evaluates general response quality against custom criteria.

**Example:**
```json
{
  "name": "quality_eval",
  "eval_type": "model_graded",
  "eval_spec": {
    "sub_type": "response_quality",
    "evaluator_model": "openai",
    "custom_criteria": {
      "technical_accuracy": "Response should be technically correct",
      "clarity": "Response should be clear and easy to understand"
    }
  }
}
```

## Creating Custom Evaluations

### Step 1: Define Your Evaluation
```python
import requests

evaluation = {
    "name": "custom_code_eval",
    "description": "Evaluates code generation quality",
    "eval_type": "model_graded",
    "eval_spec": {
        "evaluator_model": "gpt-4",
        "metrics": ["correctness", "efficiency", "readability"],
        "threshold": 0.8,
        "custom_prompt": """
        Evaluate the generated code on:
        1. Correctness: Does it solve the problem?
        2. Efficiency: Is it optimized?
        3. Readability: Is it clean and well-structured?
        
        Score each from 0-1.
        """
    }
}

response = requests.post(
    "http://localhost:8000/v1/evals",
    json=evaluation,
    headers={"Authorization": f"Bearer {API_KEY}"}
)
eval_id = response.json()["id"]
```

### Step 2: Create Your Dataset
```python
dataset = {
    "name": "code_problems",
    "description": "Programming challenges",
    "samples": [
        {
            "input": {
                "problem": "Write a function to reverse a string",
                "solution": "def reverse(s): return s[::-1]"
            },
            "expected": {
                "passes_tests": True,
                "is_efficient": True
            }
        }
    ]
}

response = requests.post(
    "http://localhost:8000/v1/datasets",
    json=dataset,
    headers={"Authorization": f"Bearer {API_KEY}"}
)
dataset_id = response.json()["id"]
```

## Uploading Vendor Evaluations

### From OpenAI Evals
```python
# Import OpenAI eval format
openai_eval = {
    "name": "imported_openai_eval",
    "eval_type": "model_graded",
    "eval_spec": {
        # Copy eval spec from OpenAI format
        "evaluator_model": "gpt-4",
        "metrics": ["accuracy"],
        "threshold": 0.8
    },
    "dataset": [
        # Convert OpenAI dataset format
        {"input": sample["input"], "expected": sample["ideal"]}
        for sample in openai_samples
    ]
}
```

### From Custom Formats
```python
def convert_vendor_eval(vendor_data):
    """Convert vendor-specific format to tldw format"""
    return {
        "name": vendor_data["test_name"],
        "eval_type": determine_eval_type(vendor_data),
        "eval_spec": {
            "evaluator_model": vendor_data.get("model", "gpt-3.5-turbo"),
            "metrics": vendor_data.get("metrics", ["accuracy"]),
            "threshold": vendor_data.get("pass_threshold", 0.7)
        },
        "dataset": [
            transform_sample(s) for s in vendor_data["samples"]
        ]
    }
```

## Running Evaluations

### Basic Run
```python
# Start evaluation run
run_request = {
    "target_model": "gpt-3.5-turbo",
    "config": {
        "temperature": 0.0,
        "max_workers": 4,
        "timeout_seconds": 300
    }
}

response = requests.post(
    f"http://localhost:8000/v1/evals/{eval_id}/runs",
    json=run_request,
    headers={"Authorization": f"Bearer {API_KEY}"}
)
run_id = response.json()["id"]
```

### Monitor Progress
```python
import time

while True:
    response = requests.get(
        f"http://localhost:8000/v1/runs/{run_id}",
        headers={"Authorization": f"Bearer {API_KEY}"}
    )
    status = response.json()["status"]
    
    if status == "completed":
        break
    elif status == "failed":
        print("Evaluation failed:", response.json().get("error_message"))
        break
    
    print(f"Progress: {response.json().get('progress', {})}")
    time.sleep(2)
```

### Stream Progress (Server-Sent Events)
```python
import sseclient

response = requests.get(
    f"http://localhost:8000/v1/runs/{run_id}/stream",
    headers={"Authorization": f"Bearer {API_KEY}"},
    stream=True
)

client = sseclient.SSEClient(response)
for event in client.events():
    if event.event == "progress":
        print(f"Progress: {event.data}")
    elif event.event == "completed":
        print(f"Completed: {event.data}")
        break
```

## Interpreting Results

### Understanding Metrics
```python
# Get results
response = requests.get(
    f"http://localhost:8000/v1/runs/{run_id}/results",
    headers={"Authorization": f"Bearer {API_KEY}"}
)
results = response.json()

# Aggregate metrics
print(f"Mean Score: {results['results']['aggregate']['mean_score']:.2f}")
print(f"Pass Rate: {results['results']['aggregate']['pass_rate']:.1%}")
print(f"Std Dev: {results['results']['aggregate']['std_dev']:.3f}")

# Per-metric breakdown
for metric, stats in results['results']['by_metric'].items():
    print(f"{metric}: {stats['mean']:.2f} (±{stats['std']:.3f})")

# Sample-level results
for sample in results['results']['sample_results'][:5]:
    print(f"Sample {sample['sample_id']}: {'✓' if sample['passed'] else '✗'}")
    for metric, score in sample['scores'].items():
        print(f"  {metric}: {score:.2f}")
```

### Result Structure
```json
{
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
      "fluency": {"mean": 0.88, "std": 0.10},
      "relevance": {"mean": 0.82, "std": 0.15}
    },
    "sample_results": [...]
  }
}
```

## Best Practices

### 1. Dataset Design
- **Representative samples**: Include edge cases and typical inputs
- **Balanced distribution**: Mix easy and hard cases
- **Clear expectations**: Define unambiguous expected outputs
- **Sufficient size**: At least 20-50 samples for statistical significance

### 2. Evaluation Configuration
- **Appropriate metrics**: Choose metrics that align with your goals
- **Reasonable thresholds**: Set pass/fail criteria based on requirements
- **Model selection**: Use capable evaluator models (GPT-4 for complex evals)
- **Temperature settings**: Use 0.0 for consistency, 0.1-0.3 for variation

### 3. Performance Optimization
- **Batch processing**: Use `max_workers` for parallel evaluation
- **Timeout settings**: Set appropriate timeouts for your use case
- **Resource management**: Monitor API usage and costs
- **Caching**: Reuse datasets across evaluations

### 4. Result Analysis
- **Statistical significance**: Consider standard deviation and sample size
- **Error analysis**: Review failed samples for patterns
- **Metric correlation**: Check if metrics align with human judgment
- **Iterative improvement**: Use results to refine your evaluation

## Examples

### Complete Summarization Evaluation
```python
import requests
import json

# Configuration
API_KEY = "your-api-key"
BASE_URL = "http://localhost:8000"
headers = {"Authorization": f"Bearer {API_KEY}"}

# 1. Create evaluation
eval_config = {
    "name": "news_summarization_quality",
    "description": "Evaluate news article summarization",
    "eval_type": "model_graded",
    "eval_spec": {
        "sub_type": "summarization",
        "evaluator_model": "gpt-4",
        "metrics": ["fluency", "consistency", "relevance", "coherence"],
        "threshold": 0.75
    },
    "dataset": [
        {
            "input": {
                "source_text": """
                Scientists at MIT have developed a new type of battery that could 
                revolutionize energy storage. The aluminum-sulfur battery is made 
                from abundant materials and costs a fraction of lithium-ion batteries. 
                Initial tests show it can charge in under a minute and maintains 
                capacity after thousands of cycles. The breakthrough could accelerate 
                adoption of renewable energy by solving intermittent storage challenges.
                """,
                "summary": "MIT scientists created an affordable aluminum-sulfur battery that charges quickly and could advance renewable energy storage."
            }
        }
        # Add more samples...
    ]
}

response = requests.post(f"{BASE_URL}/v1/evals", json=eval_config, headers=headers)
eval_id = response.json()["id"]
print(f"Created evaluation: {eval_id}")

# 2. Run evaluation
run_config = {
    "target_model": "gpt-3.5-turbo",  # Model being evaluated
    "config": {
        "temperature": 0.0,
        "max_workers": 2,
        "timeout_seconds": 120
    }
}

response = requests.post(
    f"{BASE_URL}/v1/evals/{eval_id}/runs", 
    json=run_config, 
    headers=headers
)
run_id = response.json()["id"]
print(f"Started run: {run_id}")

# 3. Wait for completion
import time
while True:
    response = requests.get(f"{BASE_URL}/v1/runs/{run_id}", headers=headers)
    status = response.json()["status"]
    if status in ["completed", "failed"]:
        break
    time.sleep(2)

# 4. Get results
if status == "completed":
    response = requests.get(f"{BASE_URL}/v1/runs/{run_id}/results", headers=headers)
    results = response.json()
    
    print("\n=== Evaluation Results ===")
    print(f"Overall Score: {results['results']['aggregate']['mean_score']:.2%}")
    print(f"Pass Rate: {results['results']['aggregate']['pass_rate']:.1%}")
    
    print("\nMetric Breakdown:")
    for metric, stats in results['results']['by_metric'].items():
        print(f"  {metric}: {stats['mean']:.2f}")
```

### Batch Evaluation with Multiple Models
```python
models_to_test = ["gpt-3.5-turbo", "gpt-4", "claude-2"]
results_comparison = {}

for model in models_to_test:
    # Run evaluation for each model
    run_config = {"target_model": model, "config": {"temperature": 0.0}}
    response = requests.post(
        f"{BASE_URL}/v1/evals/{eval_id}/runs",
        json=run_config,
        headers=headers
    )
    run_id = response.json()["id"]
    
    # Wait and collect results
    # ... (monitoring code)
    
    results_comparison[model] = results

# Compare models
for model, results in results_comparison.items():
    score = results['results']['aggregate']['mean_score']
    print(f"{model}: {score:.2%}")
```

## Troubleshooting

### Common Issues

1. **"API key required" error**
   - Ensure you're setting the Authorization header correctly
   - Check if the API requires a specific key for model-graded evaluations

2. **"Evaluation failed" with no results**
   - Check API keys for the evaluator model (e.g., OpenAI API key)
   - Verify the evaluator model name is correct
   - Check logs for detailed error messages

3. **Slow evaluation performance**
   - Increase `max_workers` for parallel processing
   - Use a faster evaluator model for development
   - Reduce dataset size for testing

4. **Inconsistent results**
   - Use temperature=0.0 for deterministic evaluation
   - Ensure your evaluation criteria are well-defined
   - Consider using a more capable evaluator model

### Getting Help
- Check the API documentation at `/docs` endpoint
- Review example notebooks in the repository
- Submit issues on GitHub with evaluation configuration and error logs

## Next Steps
- Explore the [Developer Guide](../Code_Documentation/Evaluations_Developer_Guide.md) for extending the module
- Review the [API Reference](../API-related/Evaluations_API_Reference.md) for detailed endpoint documentation
- Try the example evaluations in the `examples/` directory