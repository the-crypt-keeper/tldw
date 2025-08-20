# Prompt Studio Design Document

## Overview
A comprehensive prompt engineering platform combining DSPy's programmatic optimization with Anthropic Console's intuitive testing and evaluation features.

## Core Concepts from Research

### From DSPy:
1. **Signature-Based Programming**: Define inputs/outputs declaratively
2. **Automatic Optimization**: Use metrics to automatically improve prompts
3. **Modular Architecture**: Composable prompt modules (CoT, ReAct, etc.)
4. **Bootstrapping**: Generate few-shot examples automatically
5. **MIPRO Optimization**: Multi-stage optimization with surrogate models

### From Anthropic Console:
1. **Prompt Generator**: Auto-generate production-ready templates
2. **Test Case Management**: Import/export/auto-generate test cases
3. **Side-by-Side Comparison**: Compare multiple prompt versions
4. **Prompt Improver**: Automatically refine existing prompts
5. **Role Setting & CoT**: Built-in prompt engineering techniques

## Architecture Design

### 1. Prompt Signatures (DSPy-inspired)
```python
class PromptSignature:
    """Define the contract for a prompt"""
    input_fields: List[Field]  # Expected inputs
    output_fields: List[Field]  # Expected outputs
    constraints: List[Constraint]  # Validation rules
    examples: List[Example]  # Few-shot examples
```

### 2. Prompt Modules
```python
class PromptModule:
    """Composable prompt building blocks"""
    - ChainOfThought: Add reasoning steps
    - ReAct: Reasoning + Acting pattern
    - ProgramOfThought: Step-by-step execution
    - MultiChainComparison: Compare multiple reasoning paths
    - RolePlay: Expert role assignment
```

### 3. Optimization Pipeline
```python
class OptimizationPipeline:
    """MIPRO-style optimization"""
    stages:
    1. Bootstrap: Collect input/output traces
    2. Propose: Generate instruction candidates
    3. Search: Find optimal combinations
    4. Evaluate: Score against metrics
    5. Refine: Iterative improvement
```

### 4. Test Framework
```python
class TestFramework:
    """Anthropic Console-style testing"""
    - Auto-generate test cases from descriptions
    - Import/export CSV test suites
    - Batch execution with parallel processing
    - 5-point quality grading system
    - Regression testing
```

## API Endpoints

### Core Studio Endpoints
```
POST   /api/v1/prompt-studio/projects
GET    /api/v1/prompt-studio/projects/{id}
DELETE /api/v1/prompt-studio/projects/{id}

POST   /api/v1/prompt-studio/projects/{id}/signatures
GET    /api/v1/prompt-studio/projects/{id}/signatures
PUT    /api/v1/prompt-studio/projects/{id}/signatures/{sig_id}

POST   /api/v1/prompt-studio/projects/{id}/generate
POST   /api/v1/prompt-studio/projects/{id}/improve
POST   /api/v1/prompt-studio/projects/{id}/optimize
```

### Testing Endpoints
```
POST   /api/v1/prompt-studio/projects/{id}/test-cases
POST   /api/v1/prompt-studio/projects/{id}/test-cases/generate
POST   /api/v1/prompt-studio/projects/{id}/test-cases/import
GET    /api/v1/prompt-studio/projects/{id}/test-cases/export

POST   /api/v1/prompt-studio/projects/{id}/evaluate
GET    /api/v1/prompt-studio/projects/{id}/evaluations/{eval_id}
POST   /api/v1/prompt-studio/projects/{id}/compare
```

### Module Management
```
GET    /api/v1/prompt-studio/modules
POST   /api/v1/prompt-studio/projects/{id}/modules/{module_type}
GET    /api/v1/prompt-studio/projects/{id}/modules
```

## Database Schema

### New Tables
```sql
-- Projects table
CREATE TABLE prompt_studio_projects (
    id INTEGER PRIMARY KEY,
    uuid TEXT UNIQUE NOT NULL,
    name TEXT NOT NULL,
    description TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    user_id TEXT,
    status TEXT DEFAULT 'draft',
    metadata JSON
);

-- Signatures table
CREATE TABLE prompt_signatures (
    id INTEGER PRIMARY KEY,
    project_id INTEGER REFERENCES prompt_studio_projects(id),
    name TEXT NOT NULL,
    input_schema JSON NOT NULL,
    output_schema JSON NOT NULL,
    constraints JSON,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Test cases table
CREATE TABLE prompt_test_cases (
    id INTEGER PRIMARY KEY,
    project_id INTEGER REFERENCES prompt_studio_projects(id),
    name TEXT,
    inputs JSON NOT NULL,
    expected_outputs JSON,
    tags TEXT,
    is_golden BOOLEAN DEFAULT FALSE,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Evaluation runs table
CREATE TABLE prompt_evaluations (
    id INTEGER PRIMARY KEY,
    project_id INTEGER REFERENCES prompt_studio_projects(id),
    prompt_version_id INTEGER,
    test_case_ids JSON,
    results JSON,
    metrics JSON,
    model_used TEXT,
    total_tokens INTEGER,
    cost_estimate REAL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Optimization runs table
CREATE TABLE prompt_optimizations (
    id INTEGER PRIMARY KEY,
    project_id INTEGER REFERENCES prompt_studio_projects(id),
    optimizer_type TEXT,
    initial_prompt_id INTEGER,
    optimized_prompt_id INTEGER,
    metrics_before JSON,
    metrics_after JSON,
    optimization_config JSON,
    iterations INTEGER,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Prompt versions table
CREATE TABLE prompt_versions (
    id INTEGER PRIMARY KEY,
    project_id INTEGER REFERENCES prompt_studio_projects(id),
    version_number INTEGER,
    system_prompt TEXT,
    user_prompt TEXT,
    few_shot_examples JSON,
    modules_config JSON,
    parent_version_id INTEGER,
    change_description TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
```

## Key Features Implementation

### 1. Prompt Generator (Anthropic-style)
- User describes task in natural language
- System generates production-ready template
- Includes role setting, CoT, examples
- Customizable based on task type

### 2. Automatic Optimization (DSPy-style)
- Define metric functions (accuracy, fluency, etc.)
- Bootstrap few-shot examples from data
- MIPRO-style multi-stage optimization
- Track optimization history

### 3. Test Case Management
- Auto-generate from task description
- Import from CSV/JSON
- Tag-based organization
- Golden test sets for regression

### 4. Evaluation & Comparison
- Side-by-side prompt comparison
- Multi-model testing
- Cost/performance analysis
- 5-point quality grading

### 5. Module Composition
- Pre-built reasoning modules
- Custom module creation
- Module chaining/composition
- Module performance tracking

## Implementation Phases

### Phase 1: Foundation (Week 1)
- [ ] Database schema creation
- [ ] Basic project management API
- [ ] Signature definition system
- [ ] Simple test case CRUD

### Phase 2: Generation & Testing (Week 2)
- [ ] Prompt generator using meta-prompting
- [ ] Test case auto-generation
- [ ] Basic evaluation framework
- [ ] Integration with existing LLM calls

### Phase 3: Optimization (Week 3)
- [ ] Bootstrap few-shot examples
- [ ] MIPRO-style optimizer
- [ ] Prompt improver feature
- [ ] Version tracking system

### Phase 4: Advanced Features (Week 4)
- [ ] Module system implementation
- [ ] Multi-model comparison
- [ ] Cost optimization analysis
- [ ] Export to production prompts

## Success Metrics
1. **Accuracy Improvement**: 30%+ improvement in prompt performance
2. **Time Savings**: 80% reduction in prompt development time
3. **Test Coverage**: 100% of prompts have test suites
4. **Cost Optimization**: 20% reduction in token usage
5. **User Adoption**: Active use by all prompt engineers

## Integration Points
- Existing PromptsDatabase for storage
- Evaluation framework for metrics
- LLM API calls for execution
- RAG system for context-aware prompts
- Export to main prompt library

## Security & Best Practices
- Input sanitization for all prompts
- Rate limiting on optimization runs
- Cost caps for expensive operations
- Audit logging for all changes
- Role-based access control