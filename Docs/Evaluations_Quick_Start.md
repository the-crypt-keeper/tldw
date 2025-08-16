# Evaluations Module - Quick Start Guide

Get started with evaluations in 5 minutes! This guide covers the essentials.

## 🚀 Prerequisites

1. **tldw_server is running**
   ```bash
   python -m uvicorn tldw_Server_API.app.main:app --reload
   ```

2. **API is accessible**
   ```bash
   curl http://localhost:8000/health
   ```

## 🔑 Authentication

Set your API key (use default for development):
```bash
export API_KEY="default-secret-key-for-single-user"
```

## 📝 Your First Evaluation (No LLM Required)

### 1. Create and Run an Exact Match Evaluation

```python
import requests
import json

API_KEY = "default-secret-key-for-single-user"
BASE_URL = "http://localhost:8000"

# Create a simple exact match evaluation
eval_data = {
    "name": "capital_cities_test",
    "eval_type": "exact_match",
    "eval_spec": {"threshold": 1.0},
    "dataset": [
        {"input": {"output": "Paris"}, "expected": {"output": "Paris"}},
        {"input": {"output": "London"}, "expected": {"output": "London"}},
        {"input": {"output": "Berlin"}, "expected": {"output": "Berlin"}},
        {"input": {"output": "Rome"}, "expected": {"output": "Madrid"}}  # This will fail
    ]
}

# Create the evaluation
response = requests.post(
    f"{BASE_URL}/v1/evals",
    json=eval_data,
    headers={"Authorization": f"Bearer {API_KEY}"}
)
eval_id = response.json()["id"]
print(f"✅ Created evaluation: {eval_id}")

# Run the evaluation
run_response = requests.post(
    f"{BASE_URL}/v1/evals/{eval_id}/runs",
    json={"target_model": "none", "config": {"temperature": 0}},
    headers={"Authorization": f"Bearer {API_KEY}"}
)
run_id = run_response.json()["id"]
print(f"🏃 Started run: {run_id}")

# Wait a moment and get results
import time
time.sleep(2)

results = requests.get(
    f"{BASE_URL}/v1/runs/{run_id}/results",
    headers={"Authorization": f"Bearer {API_KEY}"}
)

if results.status_code == 200:
    data = results.json()
    print(f"\n📊 Results:")
    print(f"Pass Rate: {data['results']['aggregate']['pass_rate']:.0%}")
    print(f"Mean Score: {data['results']['aggregate']['mean_score']:.2f}")
else:
    print("⏳ Still processing...")
```

**Expected Output:**
```
✅ Created evaluation: eval_xxxxxxxxxxxx
🏃 Started run: run_xxxxxxxxxxxx

📊 Results:
Pass Rate: 75%
Mean Score: 0.75
```

## 🤖 LLM-Based Evaluation (Requires API Key)

### 2. Summarization Quality Evaluation

```python
# Evaluate a summary using G-Eval
summary_eval = {
    "name": "news_summary_quality",
    "eval_type": "model_graded",
    "eval_spec": {
        "sub_type": "summarization",
        "evaluator_model": "openai",  # or "anthropic", "groq", etc.
        "metrics": ["fluency", "relevance"],
        "threshold": 0.7
    },
    "dataset": [{
        "input": {
            "source_text": """
            The James Webb Space Telescope has captured unprecedented images 
            of distant galaxies, revealing details about the early universe 
            that were previously impossible to observe. The telescope's infrared 
            capabilities allow it to see through cosmic dust and detect light 
            from the first stars and galaxies that formed after the Big Bang.
            """,
            "summary": "Webb telescope shows early universe details using infrared to see through dust."
        }
    }]
}

# Note: Requires OPENAI_API_KEY in your config.txt
response = requests.post(f"{BASE_URL}/v1/evals", json=summary_eval, headers={"Authorization": f"Bearer {API_KEY}"})
```

## 📚 Common Use Cases

### Check if Text Contains Keywords
```python
keyword_eval = {
    "name": "keyword_check",
    "eval_type": "includes",
    "eval_spec": {"threshold": 0.8},  # Must include 80% of keywords
    "dataset": [{
        "input": {"output": "The quick brown fox jumps over the lazy dog"},
        "expected": {"includes": ["quick", "fox", "dog"]}
    }]
}
```

### Fuzzy String Matching
```python
fuzzy_eval = {
    "name": "fuzzy_match_test",
    "eval_type": "fuzzy_match",
    "eval_spec": {"threshold": 0.85},  # 85% similarity required
    "dataset": [{
        "input": {"output": "The capital of France is Paris"},
        "expected": {"output": "Paris is the capital city of France"}
    }]
}
```

### RAG System Evaluation
```python
rag_eval = {
    "name": "rag_quality",
    "eval_type": "model_graded",
    "eval_spec": {
        "sub_type": "rag",
        "evaluator_model": "openai",
        "metrics": ["relevance", "faithfulness"],
        "threshold": 0.75
    },
    "dataset": [{
        "input": {
            "query": "What is Python?",
            "contexts": [
                "Python is a high-level programming language.",
                "It was created by Guido van Rossum."
            ],
            "response": "Python is a high-level programming language created by Guido van Rossum."
        }
    }]
}
```

## 📊 Understanding Results

Results include:
- **pass_rate**: Percentage of samples that passed
- **mean_score**: Average score across all samples
- **std_dev**: Standard deviation (consistency measure)
- **by_metric**: Breakdown by individual metrics

```python
# Pretty print results
def print_results(run_id):
    response = requests.get(
        f"{BASE_URL}/v1/runs/{run_id}/results",
        headers={"Authorization": f"Bearer {API_KEY}"}
    )
    
    if response.status_code == 200:
        results = response.json()["results"]["aggregate"]
        print(f"""
        ✅ Pass Rate: {results['pass_rate']:.1%}
        📊 Mean Score: {results['mean_score']:.2f}
        📈 Std Dev: {results['std_dev']:.3f}
        📋 Total Samples: {results['total_samples']}
        ❌ Failed: {results['failed_samples']}
        """)
```

## 🔧 Troubleshooting

### Issue: "API key required for openai"
**Solution**: Add your OpenAI API key to `Config_Files/config.txt`:
```ini
[API]
openai_api_key = sk-your-api-key-here
```

### Issue: "Evaluation not found"
**Solution**: Check the evaluation ID is correct:
```python
# List all evaluations
response = requests.get(f"{BASE_URL}/v1/evals", headers={"Authorization": f"Bearer {API_KEY}"})
for eval in response.json()["data"]:
    print(f"{eval['id']}: {eval['name']}")
```

### Issue: Results show status "pending" or "running"
**Solution**: Wait a bit longer or check run status:
```python
response = requests.get(f"{BASE_URL}/v1/runs/{run_id}", headers={"Authorization": f"Bearer {API_KEY}"})
print(f"Status: {response.json()['status']}")
print(f"Progress: {response.json().get('progress', {})}")
```

## 🎯 Best Practices

1. **Start Simple**: Use exact_match or includes for initial testing
2. **Small Datasets**: Test with 3-5 samples before scaling up
3. **Clear Metrics**: Choose metrics that align with your goals
4. **Temperature Zero**: Use temperature=0 for consistent results
5. **Save Evaluations**: Reuse evaluation definitions across runs

## 📖 Next Steps

- **[User Guide](Docs/User_Guides/Evaluations_User_Guide.md)** - Detailed usage and examples
- **[Developer Guide](Docs/Code_Documentation/Evaluations_Developer_Guide.md)** - Extend the module
- **[API Reference](Docs/API-related/Evaluations_API_Reference.md)** - Complete endpoint documentation

## 💡 Quick Tips

### Monitor Progress in Real-Time
```python
import sseclient

# Stream progress updates
response = requests.get(
    f"{BASE_URL}/v1/runs/{run_id}/stream",
    headers={"Authorization": f"Bearer {API_KEY}"},
    stream=True
)

for event in sseclient.SSEClient(response).events():
    print(f"{event.event}: {event.data}")
    if event.event in ["completed", "failed"]:
        break
```

### Batch Multiple Evaluations
```python
# Run multiple evaluations in parallel
eval_ids = ["eval_1", "eval_2", "eval_3"]
run_ids = []

for eval_id in eval_ids:
    response = requests.post(
        f"{BASE_URL}/v1/evals/{eval_id}/runs",
        json={"target_model": "test", "config": {}},
        headers={"Authorization": f"Bearer {API_KEY}"}
    )
    run_ids.append(response.json()["id"])

# Check all results
for run_id in run_ids:
    # ... get and process results
```

## 🚀 Ready to evaluate!

You now have everything you need to start evaluating AI outputs. Remember:
- No LLM needed for exact_match, includes, fuzzy_match
- Add API keys in config.txt for model-graded evaluations
- Start with simple evaluations and build complexity

Happy evaluating! 🎉