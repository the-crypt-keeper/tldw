# Benchmark API Integration Guide

## Current Compatibility

The new Evaluations module with benchmark support is **mostly compatible** with the existing API. Here's the status:

### ✅ Fully Compatible Components

1. **Custom Metric Endpoint** (`/evaluations/custom-metric`)
   - Works with all new benchmark types
   - The `format_for_custom_metric()` method outputs compatible data
   - No changes needed

2. **Batch Evaluation** (`/evaluations/batch`)
   - Can process benchmark items in batches
   - Supports parallel processing
   - Compatible with existing schemas

3. **Evaluation Manager**
   - Storage, retrieval, and comparison work unchanged
   - History tracking is fully compatible

### 🔧 Integration Options

You have three options for integrating the new benchmark system:

## Option 1: Use Existing Endpoints (No API Changes)

The benchmarks can work through existing endpoints:

```python
# Using /evaluations/custom-metric
from benchmark_registry import get_registry

registry = get_registry()
evaluator = registry.create_evaluator("simpleqa")
formatted = evaluator.format_for_custom_metric(question_data)

# POST to /evaluations/custom-metric
response = await client.post("/evaluations/custom-metric", json=formatted)
```

## Option 2: Add Benchmark Router (Recommended)

Add the new `benchmark_api.py` router to your FastAPI app:

```python
# In app/main.py or your router configuration
from app.api.v1.endpoints import benchmark_api

# Add to your FastAPI app
app.include_router(benchmark_api.router, prefix="/api/v1")
```

This adds these new endpoints:
- `GET /api/v1/benchmarks/list` - List available benchmarks
- `GET /api/v1/benchmarks/{name}/info` - Get benchmark details
- `GET /api/v1/benchmarks/{name}/samples` - Preview benchmark questions
- `POST /api/v1/benchmarks/{name}/run` - Run a benchmark
- `POST /api/v1/benchmarks/simpleqa/evaluate` - SimpleQA-specific evaluation

## Option 3: Minimal Enhancement (Add Helper Endpoint)

Add just one endpoint to make benchmark running easier:

```python
@router.post("/evaluations/run-benchmark/{benchmark_name}")
async def run_benchmark(
    benchmark_name: str,
    limit: Optional[int] = None,
    api_name: str = "openai"
):
    """Simple benchmark runner using existing infrastructure."""
    registry = get_registry()
    evaluator = registry.create_evaluator(benchmark_name)
    dataset = load_benchmark_dataset(benchmark_name, limit=limit)
    
    # Use existing batch endpoint
    items = [evaluator.format_for_custom_metric(item) for item in dataset]
    
    # Process through existing batch endpoint
    return await evaluate_batch(
        BatchEvaluationRequest(
            evaluation_type="custom",
            items=items,
            api_name=api_name
        )
    )
```

## SimpleQA Special Considerations

SimpleQA's three-grade system (correct/incorrect/not_attempted) works with the existing system but could benefit from dedicated handling:

### Current: Through Custom Metric
```python
# SimpleQA returns score + grade in metadata
result = {
    "score": 1.0,  # or 0.0
    "explanation": "Correct answer",
    "metadata": {
        "grade": "correct"  # or "incorrect", "not_attempted"
    }
}
```

### Enhanced: With Dedicated Support
The `benchmark_api.py` includes a SimpleQA-specific endpoint that properly handles the three-grade system.

## Migration Path

1. **Phase 1**: Use existing endpoints (no changes needed)
2. **Phase 2**: Add benchmark router for better UX
3. **Phase 3**: Deprecate direct custom-metric calls for benchmarks

## Testing Integration

```bash
# Test with existing endpoint
curl -X POST "/api/v1/evaluations/custom-metric" \
  -H "Content-Type: application/json" \
  -d '{
    "name": "simpleqa_factuality",
    "description": "SimpleQA test",
    "evaluation_prompt": "...",
    "input_data": {...},
    "scoring_criteria": {...}
  }'

# Test with new benchmark endpoint (if added)
curl -X POST "/api/v1/benchmarks/simpleqa/run" \
  -H "Content-Type: application/json" \
  -d '{
    "limit": 10,
    "api_name": "openai"
  }'
```

## Summary

- **No breaking changes** - Existing API continues to work
- **Optional enhancements** - Add benchmark-specific endpoints for better UX
- **Full compatibility** - All benchmark types work through existing infrastructure
- **Recommended approach** - Add the benchmark router for cleaner API