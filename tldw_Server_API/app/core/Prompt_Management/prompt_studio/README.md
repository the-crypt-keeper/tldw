# Prompt Studio Documentation

## Overview

Prompt Studio is a comprehensive prompt engineering platform integrated into tldw_server. It combines DSPy's programmatic optimization capabilities with Anthropic Console's testing features to provide a powerful environment for developing, testing, and optimizing LLM prompts.

## Table of Contents

1. [Features](#features)
2. [Architecture](#architecture)
3. [Installation](#installation)
4. [Quick Start](#quick-start)
5. [Core Components](#core-components)
6. [API Reference](#api-reference)
7. [Usage Examples](#usage-examples)
8. [Best Practices](#best-practices)
9. [Troubleshooting](#troubleshooting)

## Features

### Core Capabilities
- **Multi-Provider Support**: Integration with 15+ LLM providers (OpenAI, Anthropic, Groq, Mistral, etc.)
- **Test Management**: Create, import/export, and generate test cases
- **Evaluation System**: Comprehensive metrics for prompt performance assessment
- **Optimization Engine**: 5 different optimization strategies (MIPRO, Bootstrap, Genetic, etc.)
- **Real-time Updates**: WebSocket support for live progress tracking
- **Job Queue**: Async processing with retry logic and prioritization
- **Version Control**: Full prompt versioning with history tracking

### Key Components
1. **Projects & Signatures**: Organize prompts with DSPy-style signatures
2. **Test Cases**: Comprehensive test case management with golden sets
3. **Evaluations**: Run prompts against test cases with multiple metrics
4. **Optimizations**: Automated prompt improvement strategies
5. **Monitoring**: Built-in metrics and health checks

## Architecture

```
prompt_studio/
├── Database Layer
│   ├── PromptStudioDatabase.py     # Extended database operations
│   └── Schemas (8 tables)           # Projects, prompts, tests, etc.
├── Core Components
│   ├── prompt_executor.py          # LLM execution engine
│   ├── evaluation_metrics.py       # Metric calculations
│   ├── test_runner.py              # Test execution
│   └── optimization_engine.py      # Optimization strategies
├── Management Systems
│   ├── test_case_manager.py        # Test CRUD operations
│   ├── job_manager.py              # Job queue management
│   └── event_broadcaster.py        # Real-time events
├── API Layer
│   ├── endpoints/                  # FastAPI endpoints
│   └── schemas/                     # Pydantic models
└── Supporting Systems
    ├── auth_permissions.py         # Authentication & authorization
    ├── monitoring.py                # Metrics & health checks
    └── evaluation_reports.py        # Report generation
```

## Installation

### Prerequisites
```bash
# Required
pip install fastapi sqlalchemy pydantic loguru numpy

# Optional but recommended
pip install rapidfuzz pandas matplotlib scipy jwt
```

### Database Setup
The database tables are automatically created on first use. To manually initialize:

```python
from tldw_Server_API.app.core.DB_Management.PromptStudioDatabase import PromptStudioDatabase

db = PromptStudioDatabase("path/to/prompts.db", "client_id")
# Tables are created automatically
```

## Quick Start

### 1. Create a Project
```python
from tldw_Server_API.app.core.DB_Management.PromptStudioDatabase import PromptStudioDatabase

db = PromptStudioDatabase("prompts.db", "my_client")

project_id = db.create_project(
    name="My First Project",
    description="Testing prompt optimization"
)
```

### 2. Define a Signature
```python
signature = {
    "name": "Summarization",
    "instruction": "Summarize the given text",
    "input_schema": [
        {"name": "text", "type": "string", "required": True},
        {"name": "max_words", "type": "integer", "required": False}
    ],
    "output_schema": [
        {"name": "summary", "type": "string"},
        {"name": "key_points", "type": "array"}
    ]
}

signature_id = db.create_signature(project_id, signature)
```

### 3. Create a Prompt
```python
prompt = {
    "name": "Basic Summarizer",
    "content": "Summarize this text in {max_words} words:\n\n{text}",
    "system_prompt": "You are a helpful assistant that creates concise summaries."
}

prompt_id = db.create_prompt(project_id, signature_id, prompt)
```

### 4. Create Test Cases
```python
from prompt_studio.test_case_manager import TestCaseManager

manager = TestCaseManager(db)

test_case = manager.create_test_case(
    project_id=project_id,
    name="News Article Test",
    inputs={
        "text": "Long news article text...",
        "max_words": 100
    },
    expected_outputs={
        "summary": "Expected summary...",
        "key_points": ["point1", "point2"]
    },
    is_golden=True
)
```

### 5. Run Evaluation
```python
from prompt_studio.test_runner import TestRunner

runner = TestRunner(db)

result = await runner.run_single_test(
    prompt_id=prompt_id,
    test_case_id=test_case["id"],
    model_config={
        "provider": "openai",
        "model": "gpt-3.5-turbo",
        "parameters": {"temperature": 0.7}
    }
)

print(f"Score: {result['scores']['aggregate_score']}")
```

### 6. Optimize Prompt
```python
from prompt_studio.optimization_engine import OptimizationEngine

engine = OptimizationEngine(db)

optimization_result = await engine.optimize(
    optimization_id=optimization_id  # Created via API
)

print(f"Improvement: {optimization_result['improvement']*100:.1f}%")
```

## Core Components

### Projects and Signatures

**Projects** organize related prompts and tests:
```python
project = {
    "name": "Customer Support Bot",
    "description": "Prompts for customer service",
    "tags": ["production", "support"],
    "config": {
        "default_model": "gpt-4",
        "max_test_cases": 100
    }
}
```

**Signatures** define input/output schemas (DSPy-style):
```python
signature = {
    "name": "QuestionAnswer",
    "instruction": "Answer the question based on context",
    "input_schema": [
        {"name": "question", "type": "string"},
        {"name": "context", "type": "string"}
    ],
    "output_schema": [
        {"name": "answer", "type": "string"},
        {"name": "confidence", "type": "number"}
    ]
}
```

### Test Case Management

```python
# Import from CSV
from prompt_studio.test_case_io import TestCaseIO

io = TestCaseIO(manager)
imported, errors = io.import_from_csv(
    project_id=project_id,
    csv_data=csv_content,
    auto_generate_names=True
)

# Generate test cases
from prompt_studio.test_case_generator import TestCaseGenerator

generator = TestCaseGenerator(manager)
cases = generator.generate_diverse_cases(
    project_id=project_id,
    signature_id=signature_id,
    num_cases=10
)

# Export to JSON
json_data = io.export_to_json(
    project_id=project_id,
    include_golden_only=True
)
```

### Evaluation Metrics

Available metrics:
- **Text Matching**: exact_match, fuzzy_match, levenshtein, token_overlap
- **Structured Data**: json_match, json_schema_valid
- **Classification**: accuracy, precision, recall, f1_score
- **Numerical**: mae, mse, rmse
- **Custom**: regex_match, contains, length_match

```python
from prompt_studio.evaluation_metrics import EvaluationMetrics, MetricType

metrics = EvaluationMetrics()
scores = metrics.evaluate(
    output="Generated text",
    expected="Expected text",
    metrics=[MetricType.FUZZY_MATCH, MetricType.TOKEN_OVERLAP]
)
```

### Optimization Strategies

#### MIPRO (Multi-Instruction Prompt Optimization)
```python
config = {
    "strategy": "mipro",
    "target_metric": "accuracy",
    "min_improvement": 0.01,
    "max_iterations": 20
}
```

#### Bootstrap Few-Shot
```python
config = {
    "strategy": "bootstrap",
    "num_examples": 3,
    "selection_strategy": "diverse"  # or "best", "random"
}
```

#### Hyperparameter Tuning
```python
config = {
    "strategy": "hyperparameter",
    "params_to_optimize": ["temperature", "max_tokens", "top_p"],
    "search_method": "bayesian"
}
```

#### Iterative Refinement
```python
config = {
    "strategy": "iterative",
    "max_iterations": 10
}
```

#### Genetic Algorithm
```python
config = {
    "strategy": "genetic",
    "population_size": 10,
    "generations": 20,
    "mutation_rate": 0.1
}
```

## API Reference

### Base URL
```
http://localhost:8000/api/v1/prompt_studio
```

### Endpoints

#### Projects
- `POST /projects/create` - Create new project
- `GET /projects/list` - List projects
- `GET /projects/{id}` - Get project details
- `PUT /projects/{id}` - Update project
- `DELETE /projects/{id}` - Delete project

#### Prompts
- `POST /prompts/create` - Create prompt
- `GET /prompts/{id}` - Get prompt
- `GET /prompts/{id}/versions` - Get version history
- `POST /prompts/{id}/revert` - Revert to version

#### Test Cases
- `POST /test_cases/create` - Create test case
- `POST /test_cases/bulk` - Bulk create
- `POST /test_cases/import` - Import from CSV/JSON
- `POST /test_cases/export/{project_id}` - Export test cases
- `POST /test_cases/generate` - Auto-generate tests

#### Evaluations
- `POST /evaluations/create` - Create evaluation
- `GET /evaluations/{id}` - Get results
- `GET /evaluations/{id}/report` - Get report
- `POST /evaluations/compare` - Compare prompts

#### Optimizations
- `POST /optimizations/create` - Start optimization
- `GET /optimizations/{id}` - Get status
- `POST /optimizations/cancel/{id}` - Cancel
- `GET /optimizations/strategies` - List strategies

### Authentication

Include API key in headers:
```http
X-API-Key: pstudio_your_api_key_here
```

Or use JWT token:
```http
Authorization: Bearer your_jwt_token_here
```

## Usage Examples

### Complete Workflow Example

```python
import asyncio
from prompt_studio import PromptStudio

async def optimize_customer_support_prompt():
    # Initialize
    studio = PromptStudio(api_key="your_key")
    
    # Create project
    project = await studio.create_project(
        name="Customer Support Optimization",
        description="Improve response quality"
    )
    
    # Define signature
    signature = await studio.create_signature(
        project_id=project.id,
        name="SupportResponse",
        inputs=["customer_query", "product_info"],
        outputs=["response", "satisfaction_prediction"]
    )
    
    # Create initial prompt
    prompt = await studio.create_prompt(
        project_id=project.id,
        signature_id=signature.id,
        content="""
        Customer Query: {customer_query}
        Product Info: {product_info}
        
        Please provide a helpful response.
        """,
        system_prompt="You are a friendly customer support agent."
    )
    
    # Import test cases from CSV
    test_cases = await studio.import_test_cases(
        project_id=project.id,
        file_path="support_test_cases.csv"
    )
    
    # Run baseline evaluation
    baseline = await studio.evaluate(
        prompt_id=prompt.id,
        test_case_ids=[tc.id for tc in test_cases],
        model="gpt-3.5-turbo"
    )
    
    print(f"Baseline score: {baseline.overall_score:.2f}")
    
    # Optimize using MIPRO
    optimization = await studio.optimize(
        prompt_id=prompt.id,
        strategy="mipro",
        test_cases=test_cases,
        max_iterations=20
    )
    
    # Get optimized prompt
    optimized = await studio.get_prompt(optimization.optimized_prompt_id)
    
    # Run final evaluation
    final = await studio.evaluate(
        prompt_id=optimized.id,
        test_case_ids=[tc.id for tc in test_cases],
        model="gpt-3.5-turbo"
    )
    
    print(f"Final score: {final.overall_score:.2f}")
    print(f"Improvement: {(final.overall_score - baseline.overall_score)*100:.1f}%")
    
    # Generate report
    report = await studio.generate_report(
        evaluation_id=final.id,
        format="pdf"
    )
    
    print(f"Report saved to: {report.path}")

# Run
asyncio.run(optimize_customer_support_prompt())
```

### WebSocket Real-time Updates

```javascript
// JavaScript client example
const ws = new WebSocket('ws://localhost:8000/ws/prompt_studio?client_id=my_client');

ws.onmessage = (event) => {
    const data = JSON.parse(event.data);
    
    switch(data.type) {
        case 'evaluation_progress':
            console.log(`Progress: ${data.data.progress}%`);
            break;
        case 'optimization_iteration':
            console.log(`Iteration ${data.data.iteration}: Score ${data.data.score}`);
            break;
        case 'job_completed':
            console.log('Job completed!', data.data);
            break;
    }
};

// Subscribe to specific optimization
ws.send(JSON.stringify({
    type: 'subscribe',
    entity_type: 'optimization',
    entity_id: 123
}));
```

## Best Practices

### 1. Test Case Design
- **Golden Sets**: Mark high-quality test cases as golden for regression testing
- **Diversity**: Include edge cases, typical cases, and challenging examples
- **Balanced**: Ensure coverage of all input variations
- **Realistic**: Use real-world data when possible

### 2. Prompt Optimization
- **Start Simple**: Begin with basic prompts and iterate
- **Measure Everything**: Track metrics throughout optimization
- **Version Control**: Keep all prompt versions for rollback
- **A/B Testing**: Compare multiple variants simultaneously

### 3. Performance
- **Batch Processing**: Use bulk operations for efficiency
- **Caching**: Results are cached automatically
- **Rate Limiting**: Respect provider rate limits
- **Async Operations**: Use async/await for concurrent execution

### 4. Security
- **API Keys**: Store securely, never in code
- **Permissions**: Use role-based access control
- **Audit Logging**: All actions are logged
- **Data Privacy**: Sensitive data should be anonymized

## Troubleshooting

### Common Issues

#### Database Lock Errors
```python
# Use connection pooling
db = PromptStudioDatabase(
    db_path="prompts.db",
    client_id="client",
    pool_size=5
)
```

#### Rate Limiting
```python
# Configure retry logic
config = {
    "max_retries": 3,
    "retry_delay": 1.0,
    "backoff_factor": 2.0
}
```

#### Memory Issues with Large Evaluations
```python
# Use streaming and pagination
results = runner.run_evaluation_streaming(
    evaluation_id=eval_id,
    batch_size=10,
    max_concurrent=3
)
```

### Debugging

Enable debug logging:
```python
from loguru import logger
logger.add("prompt_studio_debug.log", level="DEBUG")
```

Check metrics:
```python
from prompt_studio.monitoring import PromptStudioHealthCheck

health = PromptStudioHealthCheck(db)
status = health.check_health()
print(json.dumps(status, indent=2))
```

## Configuration

### Environment Variables
```bash
# Database
PROMPT_STUDIO_DB_PATH=/path/to/prompts.db

# API Keys (for LLM providers)
OPENAI_API_KEY=sk-...
ANTHROPIC_API_KEY=sk-ant-...

# Performance
PROMPT_STUDIO_MAX_WORKERS=4
PROMPT_STUDIO_CACHE_SIZE=1000

# Monitoring
OTEL_EXPORTER_OTLP_ENDPOINT=http://localhost:4317
```

### Config File (prompt_studio_config.yaml)
```yaml
database:
  path: ./prompts.db
  pool_size: 5
  timeout: 30

execution:
  max_concurrent_tests: 10
  default_timeout: 60
  retry_attempts: 3

optimization:
  max_iterations: 50
  early_stopping_patience: 5
  min_improvement: 0.001

monitoring:
  metrics_enabled: true
  export_interval: 60
  health_check_interval: 30
```

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for development setup and guidelines.

## License

Part of tldw_server project - see main LICENSE file.