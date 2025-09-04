# Evaluation Module Test Strategy

## Overview

This document describes the comprehensive testing strategy for the Evaluations module, following best practices with clear separation between unit, integration, and property-based tests.

## Test Philosophy

### Core Principles

1. **No Excessive Mocking**: 
   - Unit tests: Mock only external services (LLMs, embeddings APIs)
   - Integration tests: Zero mocking - use real components
   - Property tests: Zero mocking - use generated data

2. **No Skipped Tests**: Every test must be functional and provide value

3. **Clear Separation**: Tests are organized by type with distinct purposes

4. **Best Practices**: Fixtures, proper assertions, descriptive names, comprehensive docstrings

## Directory Structure

```
tests/Evaluations/
├── unit/                      # Isolated component testing
│   ├── test_evaluation_manager.py
│   ├── test_rag_evaluator.py
│   ├── test_response_quality_evaluator.py
│   ├── test_circuit_breaker.py
│   ├── test_connection_pool.py
│   ├── test_webhook_manager.py
│   ├── test_rate_limiter.py
│   └── test_validators.py
│
├── integration/               # Real component interaction testing
│   ├── test_api_endpoints.py
│   ├── test_database_operations.py
│   ├── test_webhook_delivery.py
│   ├── test_llm_integration.py
│   └── test_batch_processing.py
│
├── property/                  # Invariant and property testing
│   ├── test_evaluation_invariants.py
│   ├── test_metric_properties.py
│   ├── test_rate_limit_properties.py
│   └── test_concurrent_operations.py
│
└── fixtures/                  # Shared test resources
    ├── database.py           # Database helpers
    ├── sample_data.py        # Data generators
    └── llm_responses.py      # Cached LLM responses
```

## Test Types

### Unit Tests

**Purpose**: Test individual components in isolation

**Characteristics**:
- Fast execution (<100ms per test)
- Minimal external dependencies
- Mock only external services (LLMs, APIs)
- Focus on business logic validation

**Example**:
```python
@pytest.mark.unit
def test_normalize_score(self):
    """Test score normalization from 1-5 to 0-1 scale."""
    evaluator = RAGEvaluator()
    assert evaluator._normalize_score(1) == 0
    assert evaluator._normalize_score(3) == 0.5
    assert evaluator._normalize_score(5) == 1.0
```

### Integration Tests

**Purpose**: Test real interactions between components

**Characteristics**:
- Use actual database (SQLite in-memory or temp file)
- Real API calls with test endpoints
- No mocking of internal components
- May use cached/deterministic external responses

**Example**:
```python
@pytest.mark.integration
async def test_complete_evaluation_flow(self, async_api_client):
    """Test complete evaluation workflow with real components."""
    response = await async_api_client.post(
        "/api/v1/evaluations/geval",
        json={"source_text": "...", "summary": "...", "criteria": "coherence"}
    )
    assert response.status_code == 200
    # Verify database persistence
    # Verify webhook triggers
    # Verify metrics recording
```

### Property-Based Tests

**Purpose**: Verify invariants hold for all possible inputs

**Characteristics**:
- Use Hypothesis for data generation
- Test mathematical properties
- Verify system invariants
- No mocking whatsoever

**Example**:
```python
@given(scores=st.lists(st.floats(0, 1), min_size=1))
def test_average_score_bounds(scores):
    """Average score must be between min and max."""
    avg = calculate_average(scores)
    assert min(scores) <= avg <= max(scores)
```

## Test Data Strategy

### Unit Tests
- Minimal, focused test data
- Deterministic values
- Edge cases explicitly tested

### Integration Tests
- Realistic data volumes
- Database seeding helpers
- Cached external responses for reproducibility

### Property Tests
- Hypothesis strategies for generation
- Wide range including edge cases
- Minimum 100 examples per property

## Fixtures

### Database Fixtures
```python
@pytest.fixture
def temp_db_path() -> Path:
    """Temporary database with full schema."""
    
@pytest.fixture
def in_memory_db() -> sqlite3.Connection:
    """In-memory database for fast tests."""
```

### Component Fixtures
```python
@pytest.fixture
def evaluation_manager(temp_db_path):
    """Configured EvaluationManager instance."""
    
@pytest.fixture
def rag_evaluator():
    """RAGEvaluator without embeddings (unit tests)."""
    
@pytest.fixture
def rag_evaluator_with_embeddings():
    """RAGEvaluator with embeddings (integration tests)."""
```

### Data Fixtures
```python
@pytest.fixture
def sample_evaluation_data():
    """Standard evaluation test data."""
    
@pytest.fixture
def evaluation_data_generator():
    """Random data generator for property tests."""
```

## Coverage Requirements

### Line Coverage
- Target: >95%
- Current: Measured by `pytest --cov`
- Exclusions: Deprecated code, debug statements

### Branch Coverage
- Target: >90%
- All conditional paths tested
- Edge cases explicitly covered

### Integration Coverage
- All API endpoints tested
- All database operations verified
- All external service interactions validated

## Performance Benchmarks

### Unit Test Performance
- Individual test: <100ms
- Test class: <1 second
- Full suite: <10 seconds

### Integration Test Performance
- Individual test: <1 second
- Test class: <10 seconds
- Full suite: <30 seconds

### Property Test Performance
- Per property: 100+ examples
- Timeout: 60 seconds per property
- Parallelization enabled

## Error Scenarios

### Tested Error Conditions

1. **Invalid Input**
   - Missing required fields
   - Invalid data types
   - Out-of-range values

2. **External Service Failures**
   - LLM API errors
   - Embedding service unavailable
   - Database connection issues

3. **Concurrency Issues**
   - Race conditions
   - Deadlocks
   - Connection pool exhaustion

4. **Rate Limiting**
   - Exceeded limits
   - Tier enforcement
   - Recovery behavior

## Continuous Integration

### Test Execution Order
1. Unit tests (fail fast)
2. Integration tests
3. Property tests
4. Performance benchmarks

### Test Markers
```bash
# Run only unit tests
pytest -m unit

# Run only integration tests
pytest -m integration

# Run only property tests
pytest -m property

# Run tests requiring LLM
pytest -m requires_llm
```

## Maintenance Guidelines

### Adding New Tests

1. **Determine test type** (unit/integration/property)
2. **Place in correct directory**
3. **Use appropriate fixtures**
4. **Follow naming convention**: `test_<component>_<scenario>`
5. **Add docstring** explaining what and why
6. **Mark with appropriate marker**

### Updating Existing Tests

1. **Maintain backward compatibility**
2. **Update fixtures if needed**
3. **Ensure no test skipping**
4. **Verify coverage maintained**

### Test Review Checklist

- [ ] Test has clear purpose
- [ ] Test name describes scenario
- [ ] Appropriate test type used
- [ ] No excessive mocking
- [ ] Assertions are meaningful
- [ ] Edge cases covered
- [ ] Error cases tested
- [ ] Performance acceptable

## Common Patterns

### Testing Async Operations
```python
@pytest.mark.asyncio
async def test_async_operation():
    result = await async_function()
    assert result is not None
```

### Testing Database Operations
```python
def test_database_operation(temp_db_path):
    with sqlite3.connect(temp_db_path) as conn:
        # Perform operations
        # Verify results
```

### Testing Concurrent Operations
```python
def test_concurrency():
    with ThreadPoolExecutor(max_workers=10) as executor:
        futures = [executor.submit(operation, i) for i in range(10)]
        results = [f.result() for f in futures]
        assert_no_race_conditions(results)
```

## Anti-Patterns to Avoid

### ❌ Excessive Mocking
```python
# BAD: Mocking internal components
@patch('app.core.evaluation_manager')
@patch('app.core.rag_evaluator')
@patch('app.core.database')
def test_something(mock_db, mock_rag, mock_manager):
    # This tests mocks, not actual behavior
```

### ❌ Skipped Tests
```python
# BAD: Skipping tests
@pytest.mark.skip("TODO: Fix this test")
def test_important_feature():
    pass
```

### ❌ Unclear Assertions
```python
# BAD: Vague assertions
assert result  # What does this mean?
assert len(data) > 0  # Why must it be non-empty?
```

### ✅ Good Patterns
```python
# GOOD: Clear, specific assertions
assert response.status_code == 200, "API should return success"
assert "evaluation_id" in result, "Response must include evaluation ID"
assert 0 <= score <= 1, f"Score {score} outside valid range [0,1]"
```

## Troubleshooting

### Common Issues

1. **Database Lock Errors**
   - Use separate database files for parallel tests
   - Properly close connections in fixtures

2. **Flaky Tests**
   - Remove time dependencies
   - Use deterministic data
   - Mock system time if needed

3. **Slow Tests**
   - Profile with `pytest --durations=10`
   - Use in-memory databases
   - Cache expensive operations

## Metrics and Reporting

### Test Metrics to Track
- Total test count
- Execution time
- Coverage percentage
- Failure rate
- Flakiness index

### Reporting Commands
```bash
# Coverage report
pytest --cov=tldw_Server_API.app.core.Evaluations --cov-report=html

# Performance profiling
pytest --profile

# Parallel execution
pytest -n auto

# Verbose failure output
pytest -vvs --tb=short
```

## Conclusion

This test strategy ensures comprehensive, maintainable, and reliable testing of the Evaluations module. By following these guidelines, we maintain high code quality while avoiding common testing anti-patterns.

Key takeaways:
- **Clear separation** between test types
- **No excessive mocking** - test real behavior
- **No skipped tests** - all tests provide value
- **Property-based testing** - verify invariants
- **Comprehensive coverage** - all paths tested