# Evaluations Module Developer Guide

## Table of Contents
- [Architecture Overview](#architecture-overview)
- [Module Structure](#module-structure)
- [Database Schema](#database-schema)
- [Internal API Usage](#internal-api-usage)
- [Adding New Evaluation Types](#adding-new-evaluation-types)
- [Extending the Evaluation Runner](#extending-the-evaluation-runner)
- [Integration Points](#integration-points)
- [Testing](#testing)
- [Performance Considerations](#performance-considerations)

## Architecture Overview

The Evaluations module follows a layered architecture pattern with clear separation of concerns:

```
┌─────────────────────────────────────┐
│         API Layer (FastAPI)         │
│   /app/api/v1/endpoints/            │
├─────────────────────────────────────┤
│        Business Logic Layer         │
│   /app/core/Evaluations/            │
├─────────────────────────────────────┤
│        Database Layer               │
│   /app/core/DB_Management/          │
├─────────────────────────────────────┤
│        Storage Layer                │
│   SQLite (evaluations.db)           │
└─────────────────────────────────────┘
```

### Key Components

1. **API Endpoints** (`evals_openai.py`)
   - OpenAI-compatible REST API
   - Request validation with Pydantic
   - Async request handling

2. **Evaluation Runner** (`eval_runner.py`)
   - Async task orchestration
   - Progress tracking
   - Result aggregation

3. **Evaluators** (Various `*_evaluator.py` files)
   - Specific evaluation implementations
   - Metric calculations
   - LLM integration

4. **Database Manager** (`Evaluations_DB.py`)
   - CRUD operations
   - Transaction management
   - Query optimization

## Module Structure

```
app/core/Evaluations/
├── eval_runner.py              # Main evaluation orchestrator
├── ms_g_eval.py                # G-Eval implementation
├── rag_evaluator.py            # RAG system evaluator
├── response_quality_evaluator.py # Response quality evaluator
└── evaluation_manager.py       # Legacy evaluation manager

app/api/v1/
├── endpoints/
│   ├── evals_openai.py        # OpenAI-compatible API
│   └── evals.py               # Legacy API (disabled)
└── schemas/
    ├── openai_eval_schemas.py # Request/response models
    └── evaluation_schema.py   # Legacy schemas

app/core/DB_Management/
└── Evaluations_DB.py          # Database operations
```

## Database Schema

### Tables

#### evaluations
```sql
CREATE TABLE evaluations (
    id TEXT PRIMARY KEY,           -- eval_xxxxxxxxxxxx
    name TEXT NOT NULL,
    description TEXT,
    eval_type TEXT NOT NULL,        -- model_graded, exact_match, etc.
    eval_spec TEXT NOT NULL,        -- JSON configuration
    dataset_id TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    created_by TEXT,
    metadata TEXT,                  -- JSON metadata
    deleted_at TIMESTAMP NULL       -- Soft delete
);
```

#### evaluation_runs
```sql
CREATE TABLE evaluation_runs (
    id TEXT PRIMARY KEY,           -- run_xxxxxxxxxxxx
    eval_id TEXT NOT NULL,
    status TEXT NOT NULL,          -- pending, running, completed, failed
    target_model TEXT,
    config TEXT,                   -- JSON run configuration
    progress TEXT,                 -- JSON progress info
    results TEXT,                  -- JSON results
    error_message TEXT,
    started_at TIMESTAMP,
    completed_at TIMESTAMP,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    webhook_url TEXT,
    usage TEXT,                    -- JSON token usage
    FOREIGN KEY (eval_id) REFERENCES evaluations(id)
);
```

#### datasets
```sql
CREATE TABLE datasets (
    id TEXT PRIMARY KEY,           -- dataset_xxxxxxxxxxxx
    name TEXT NOT NULL,
    description TEXT,
    samples TEXT NOT NULL,         -- JSON array of samples
    sample_count INTEGER,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    created_by TEXT,
    metadata TEXT                  -- JSON metadata
);
```

## Internal API Usage

### Using Evaluations Programmatically

```python
from tldw_Server_API.app.core.DB_Management.Evaluations_DB import EvaluationsDatabase
from tldw_Server_API.app.core.Evaluations.eval_runner import EvaluationRunner

# Initialize components
db = EvaluationsDatabase("path/to/evaluations.db")
runner = EvaluationRunner("path/to/evaluations.db")

# Create an evaluation
eval_id = db.create_evaluation(
    name="my_evaluation",
    eval_type="model_graded",
    eval_spec={
        "sub_type": "summarization",
        "evaluator_model": "gpt-4",
        "metrics": ["fluency", "relevance"],
        "threshold": 0.7
    },
    dataset_id="dataset_123"
)

# Create a run
run_id = db.create_run(
    eval_id=eval_id,
    target_model="gpt-3.5-turbo",
    config={"temperature": 0.0}
)

# Execute evaluation
await runner.run_evaluation(
    run_id=run_id,
    eval_id=eval_id,
    eval_config={
        "eval_type": "model_graded",
        "eval_spec": eval_spec,
        "dataset_id": dataset_id,
        "config": {"temperature": 0.0}
    }
)

# Get results
results = db.get_run_results(run_id)
```

### Direct Evaluator Usage

```python
from tldw_Server_API.app.core.Evaluations.rag_evaluator import RAGEvaluator

# For RAG evaluation
rag_eval = RAGEvaluator()
results = await rag_eval.evaluate(
    query="What is the capital of France?",
    contexts=["Paris is the capital of France.", "France is in Europe."],
    response="The capital of France is Paris.",
    ground_truth="Paris",
    metrics=["relevance", "faithfulness"],
    api_name="openai"
)

# For G-Eval summarization
from tldw_Server_API.app.core.Evaluations.ms_g_eval import run_geval

result = run_geval(
    transcript="Long document text...",
    summary="Summary text...",
    api_key="your-api-key",
    api_name="openai",
    save=False
)
```

## Adding New Evaluation Types

### Step 1: Create Evaluator Class

Create a new file in `/app/core/Evaluations/`:

```python
# my_custom_evaluator.py
from typing import Dict, List, Any, Optional
from loguru import logger

class MyCustomEvaluator:
    """Custom evaluation implementation"""
    
    async def evaluate(
        self,
        input_data: Dict[str, Any],
        expected: Dict[str, Any],
        config: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Evaluate input against expected output.
        
        Returns:
            Dict with scores and metadata
        """
        # Your evaluation logic here
        score = self._calculate_score(input_data, expected)
        
        return {
            "scores": {"custom_metric": score},
            "passed": score >= config.get("threshold", 0.7),
            "avg_score": score,
            "metadata": {"evaluator": "custom"}
        }
    
    def _calculate_score(self, input_data, expected):
        # Implement scoring logic
        return 0.85
```

### Step 2: Register in Evaluation Runner

Update `/app/core/Evaluations/eval_runner.py`:

```python
# In _get_evaluation_function method
def _get_evaluation_function(self, eval_type: str, eval_spec: Dict[str, Any]) -> Callable:
    if eval_type == "model_graded":
        sub_type = eval_spec.get("sub_type")
        # ... existing code ...
    elif eval_type == "custom_type":  # Add your type
        return self._eval_custom
    # ... rest of the code ...

# Add evaluation method
async def _eval_custom(
    self,
    sample: Dict[str, Any],
    eval_spec: Dict[str, Any],
    config: Dict[str, Any],
    sample_id: str
) -> Dict[str, Any]:
    """Custom evaluation implementation"""
    try:
        from .my_custom_evaluator import MyCustomEvaluator
        evaluator = MyCustomEvaluator()
        
        result = await evaluator.evaluate(
            input_data=sample["input"],
            expected=sample.get("expected", {}),
            config=eval_spec
        )
        
        return {
            "sample_id": sample_id,
            **result
        }
    except Exception as e:
        logger.error(f"Custom eval failed for {sample_id}: {e}")
        return {"sample_id": sample_id, "error": str(e)}
```

### Step 3: Update API Schema (Optional)

If your evaluation needs special parameters, update schemas:

```python
# In openai_eval_schemas.py
class CustomEvalSpec(BaseModel):
    """Custom evaluation specification"""
    custom_param: str
    another_param: Optional[int] = 10
    
# Update EvaluationSpec union type
EvaluationSpec = Union[ModelGradedSpec, ExactMatchSpec, CustomEvalSpec]
```

## Extending the Evaluation Runner

### Adding Progress Callbacks

```python
class ExtendedEvaluationRunner(EvaluationRunner):
    def __init__(self, db_path: str, progress_callback=None):
        super().__init__(db_path)
        self.progress_callback = progress_callback
    
    async def _process_batch(self, batch, eval_fn, eval_spec, eval_config, max_workers):
        results = await super()._process_batch(
            batch, eval_fn, eval_spec, eval_config, max_workers
        )
        
        # Call progress callback
        if self.progress_callback:
            await self.progress_callback(len(results))
        
        return results
```

### Custom Result Aggregation

```python
def _calculate_custom_aggregate(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Custom aggregation logic"""
    # Calculate percentiles
    import numpy as np
    scores = [r["avg_score"] for r in results if "avg_score" in r]
    
    return {
        "mean_score": np.mean(scores),
        "median_score": np.median(scores),
        "p25": np.percentile(scores, 25),
        "p75": np.percentile(scores, 75),
        "p95": np.percentile(scores, 95)
    }
```

## Integration Points

### With LLM Module

```python
from tldw_Server_API.app.core.LLM_Calls.Summarization_General_Lib import analyze
from tldw_Server_API.app.core.Chat.Chat_Functions import chat_api_call

# Use for model-graded evaluations
response = await analyze(
    input_data=text,
    custom_prompt=evaluation_prompt,
    api_name="openai",
    api_key=api_key,
    temp=0.0,
    system_message="You are an evaluation expert."
)
```

### With Embeddings Module (Future)

```python
# When embeddings module is ready
from tldw_Server_API.app.core.Embeddings import EmbeddingsServiceWrapper

class EmbeddingEvaluator:
    def __init__(self):
        self.embeddings = EmbeddingsServiceWrapper()
    
    async def calculate_similarity(self, text1: str, text2: str) -> float:
        emb1 = await self.embeddings.get_embedding(text1)
        emb2 = await self.embeddings.get_embedding(text2)
        return cosine_similarity(emb1, emb2)
```

### With Media Processing

```python
from tldw_Server_API.app.core.Ingestion_Media_Processing import process_media

# Evaluate transcription quality
async def evaluate_transcription(media_file: str):
    # Process media
    transcription = await process_media(media_file)
    
    # Evaluate against ground truth
    eval_result = await evaluator.evaluate(
        transcription=transcription,
        ground_truth=reference_text
    )
    return eval_result
```

## Testing

### Unit Testing Evaluators

```python
import pytest
from unittest.mock import Mock, patch

@pytest.mark.asyncio
async def test_custom_evaluator():
    from tldw_Server_API.app.core.Evaluations.my_custom_evaluator import MyCustomEvaluator
    
    evaluator = MyCustomEvaluator()
    result = await evaluator.evaluate(
        input_data={"text": "test"},
        expected={"output": "expected"},
        config={"threshold": 0.5}
    )
    
    assert "scores" in result
    assert result["passed"] == True
    assert result["avg_score"] >= 0.5
```

### Integration Testing

```python
@pytest.mark.asyncio
async def test_evaluation_workflow():
    """Test complete evaluation workflow"""
    from fastapi.testclient import TestClient
    from tldw_Server_API.app.main import app
    
    client = TestClient(app)
    
    # Create evaluation
    response = client.post("/v1/evals", json={
        "name": "test_eval",
        "eval_type": "exact_match",
        "eval_spec": {"threshold": 1.0},
        "dataset": [{"input": {"output": "test"}, "expected": {"output": "test"}}]
    })
    assert response.status_code == 201
    eval_id = response.json()["id"]
    
    # Run evaluation
    response = client.post(f"/v1/evals/{eval_id}/runs", json={
        "target_model": "test",
        "config": {"temperature": 0}
    })
    assert response.status_code == 202
    run_id = response.json()["id"]
    
    # Check results
    import asyncio
    await asyncio.sleep(1)  # Wait for processing
    
    response = client.get(f"/v1/runs/{run_id}/results")
    assert response.status_code == 200
    assert response.json()["results"]["aggregate"]["pass_rate"] == 1.0
```

### Mocking External Services

```python
@patch('tldw_Server_API.app.core.Chat.Chat_Functions.chat_api_call')
async def test_geval_with_mock(mock_chat):
    """Test G-Eval with mocked LLM"""
    mock_chat.return_value = "4.5"  # Mock score response
    
    from tldw_Server_API.app.core.Evaluations.ms_g_eval import run_geval
    
    result = run_geval(
        transcript="Test document",
        summary="Test summary",
        api_key="mock-key",
        api_name="openai",
        save=False
    )
    
    assert "Coherence:" in result
    mock_chat.assert_called()
```

## Performance Considerations

### Database Optimization

```python
# Use transactions for bulk operations
with db.get_connection() as conn:
    cursor = conn.cursor()
    cursor.executemany(
        "INSERT INTO evaluation_runs ...",
        batch_data
    )
    conn.commit()

# Add indexes for common queries
CREATE INDEX idx_runs_eval_status ON evaluation_runs(eval_id, status);
CREATE INDEX idx_runs_created ON evaluation_runs(created_at DESC);
```

### Async Processing

```python
# Process evaluations concurrently
async def process_evaluations_batch(evaluations: List[Dict]):
    tasks = [
        evaluate_single(eval_data) 
        for eval_data in evaluations
    ]
    
    # Limit concurrency
    semaphore = asyncio.Semaphore(10)
    async def bounded_evaluate(eval_data):
        async with semaphore:
            return await evaluate_single(eval_data)
    
    results = await asyncio.gather(
        *[bounded_evaluate(e) for e in evaluations]
    )
    return results
```

### Caching Strategy

```python
from functools import lru_cache
from typing import Tuple

class CachedEvaluator:
    @lru_cache(maxsize=1000)
    def _get_cached_score(self, input_hash: str) -> float:
        """Cache evaluation results for identical inputs"""
        return self._calculate_score(input_hash)
    
    async def evaluate(self, input_data: Dict) -> Dict:
        # Create deterministic hash
        input_hash = self._hash_input(input_data)
        
        # Check cache
        if cached_score := self._get_cached_score(input_hash):
            return {"score": cached_score, "cached": True}
        
        # Calculate and cache
        score = await self._calculate_score_async(input_data)
        return {"score": score, "cached": False}
```

### Memory Management

```python
# Stream large datasets
async def stream_dataset(dataset_id: str):
    """Stream dataset samples instead of loading all at once"""
    offset = 0
    batch_size = 100
    
    while True:
        samples = db.get_dataset_samples(
            dataset_id, 
            offset=offset, 
            limit=batch_size
        )
        if not samples:
            break
            
        for sample in samples:
            yield sample
        
        offset += batch_size

# Process with streaming
async for sample in stream_dataset(dataset_id):
    result = await evaluate_sample(sample)
    await store_result(result)
```

## Configuration Best Practices

### Environment-Specific Settings

```python
# config.py
from pydantic import BaseSettings

class EvaluationSettings(BaseSettings):
    max_workers: int = 4
    default_timeout: int = 300
    batch_size: int = 10
    cache_ttl: int = 3600
    max_retries: int = 3
    
    class Config:
        env_prefix = "EVAL_"

settings = EvaluationSettings()
```

### Feature Flags

```python
# Enable/disable features dynamically
FEATURE_FLAGS = {
    "enable_embeddings": False,  # Until embeddings module ready
    "enable_caching": True,
    "enable_webhooks": True,
    "enable_streaming": True
}

def is_feature_enabled(feature: str) -> bool:
    return FEATURE_FLAGS.get(feature, False)

# Usage
if is_feature_enabled("enable_embeddings"):
    similarity = await calculate_embedding_similarity(text1, text2)
else:
    similarity = calculate_text_similarity(text1, text2)
```

## Debugging and Monitoring

### Logging Best Practices

```python
from loguru import logger

# Structured logging
logger.info(
    "Evaluation started",
    eval_id=eval_id,
    run_id=run_id,
    samples=len(dataset),
    config=config
)

# Performance logging
with logger.contextualize(run_id=run_id):
    start = time.time()
    result = await evaluate()
    logger.info(f"Evaluation completed in {time.time()-start:.2f}s")
```

### Error Tracking

```python
class EvaluationError(Exception):
    """Base exception for evaluation errors"""
    pass

class EvaluatorNotFoundError(EvaluationError):
    """Raised when evaluator type is not found"""
    pass

class EvaluationTimeoutError(EvaluationError):
    """Raised when evaluation times out"""
    pass

# Usage with context
try:
    result = await evaluator.evaluate(sample)
except EvaluationTimeoutError as e:
    logger.error(f"Evaluation timeout: {e}", sample_id=sample_id)
    # Store partial results
    db.update_run_status(run_id, "partial", error=str(e))
```

## Migration Guide

### From Legacy to OpenAI-Compatible API

```python
# Old API call
from tldw_Server_API.app.core.Evaluations.evaluation_manager import EvaluationManager

manager = EvaluationManager()
eval_id = await manager.store_evaluation(
    evaluation_type="geval",
    input_data={"source": text, "summary": summary},
    results=results
)

# New API call
from tldw_Server_API.app.core.DB_Management.Evaluations_DB import EvaluationsDatabase

db = EvaluationsDatabase()
eval_id = db.create_evaluation(
    name="geval_summary",
    eval_type="model_graded",
    eval_spec={
        "sub_type": "summarization",
        "evaluator_model": "gpt-4"
    }
)
```

## Future Enhancements

### Planned Features
1. **Embeddings Integration**: Full vector similarity support
2. **Custom Metrics UI**: Web interface for creating evaluations
3. **Evaluation Templates**: Pre-built evaluation configurations
4. **Comparison Tools**: A/B testing framework
5. **Export Formats**: JSON, CSV, Markdown reports
6. **Scheduling**: Automated periodic evaluations
7. **Notifications**: Email/Slack integration for results

### Extension Points
- Custom storage backends (PostgreSQL, MongoDB)
- Alternative evaluation frameworks (HELM, BigBench)
- Multi-language support for evaluations
- Distributed evaluation processing
- Real-time collaboration features

## Resources
- [API Reference](../API-related/Evaluations_API_Reference.md)
- [User Guide](../User_Guides/Evaluations_User_Guide.md)
- [OpenAI Evals](https://github.com/openai/evals) - Compatible format
- [Test Suite](/tests/Evaluations/) - Example implementations