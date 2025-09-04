# ChromaDB Module Test Strategy

## Overview

This document outlines the comprehensive testing strategy for the ChromaDB module, which provides vector database functionality for the tldw_server application. The strategy follows a three-tier testing pyramid with unit, integration, and property-based tests.

## Module Architecture

The ChromaDB module consists of several key components:

1. **ChromaDBManager**: Core interface for ChromaDB operations
2. **Embedding Workers**: Pipeline for chunking, embedding generation, and storage
3. **Job Management**: Queue-based job processing system
4. **Error Recovery**: Circuit breakers and retry mechanisms
5. **Security & Audit**: Input validation and operation logging

## Testing Philosophy

### Core Principles

1. **No Mocking in Integration/Property Tests**: These tests use real components to verify actual behavior
2. **Minimal Mocking in Unit Tests**: Only mock external services (LLMs, APIs) and I/O operations
3. **Comprehensive Coverage**: Every public method and error path must be tested
4. **Property-Based Testing**: Verify invariants across all valid inputs
5. **Deterministic Tests**: All tests must be reproducible and not flaky

### Test Categories

#### 1. Unit Tests (60% of tests)
- **Location**: `tests/ChromaDB/unit/`
- **Purpose**: Test individual components in isolation
- **Mocking**: External services only (ChromaDB client, LLM APIs, file I/O)
- **Focus**: Business logic, error handling, edge cases

#### 2. Integration Tests (25% of tests)
- **Location**: `tests/ChromaDB/integration/`
- **Purpose**: Test component interactions with real ChromaDB
- **Mocking**: None (use real ChromaDB, real embeddings)
- **Focus**: End-to-end workflows, data persistence, concurrent operations

#### 3. Property Tests (15% of tests)
- **Location**: `tests/ChromaDB/property/`
- **Purpose**: Verify invariants hold across all inputs
- **Mocking**: None
- **Focus**: Mathematical properties, consistency guarantees, security invariants

## Test Coverage Areas

### 1. ChromaDBManager Tests

#### Unit Tests
- Initialization with valid/invalid user IDs
- Collection management (create, delete, reset)
- Storage operations with various input combinations
- Search operations with different parameters
- Security validation (path traversal, injection)
- Error handling and recovery
- Resource limit enforcement

#### Integration Tests
- Real ChromaDB client creation and persistence
- End-to-end storage and retrieval
- Vector search with actual embeddings
- Metadata filtering
- Large batch processing
- Concurrent operations
- Database integration

#### Property Tests
- User ID validation invariants
- Storage/retrieval consistency
- Count accuracy
- Metadata preservation
- Embedding dimension consistency
- ID uniqueness guarantees

### 2. Worker System Tests

#### Unit Tests
- Worker initialization and lifecycle
- Task processing logic
- Queue management
- Error handling and retries
- Metrics tracking
- Batch processing

#### Integration Tests
- Full pipeline coordination
- Multi-worker interaction
- Queue backpressure handling
- Error propagation
- Transaction rollbacks

#### Property Tests
- Task ordering preservation
- No data loss under load
- Consistent state after failures

### 3. Embedding Generation Tests

#### Unit Tests
- Model loading/unloading
- Embedding dimension validation
- Batch processing logic
- Cache functionality
- Provider switching

#### Integration Tests
- Real model loading (HuggingFace)
- Actual embedding generation
- Auto-unload mechanisms
- Multi-provider support

#### Property Tests
- Embedding normalization
- Dimension consistency
- Deterministic generation
- Batch consistency

### 4. Security Tests

#### Unit Tests
- Input sanitization
- Path traversal prevention
- SQL injection prevention
- Resource limit checks

#### Integration Tests
- Multi-user isolation
- Audit logging verification
- Rate limiting

#### Property Tests
- All invalid inputs rejected
- No data leakage between users
- Consistent security policies

## Test Data Strategy

### Fixtures

1. **Database Fixtures**
   - Use real MediaDatabase with proper schema
   - Temporary databases for each test
   - Proper cleanup after tests

2. **ChromaDB Fixtures**
   - Temporary ChromaDB instances
   - Mock clients for unit tests
   - Real clients for integration tests

3. **Test Data**
   - Sample texts of various lengths
   - Pre-computed embeddings for deterministic tests
   - Edge case documents (empty, very long, special characters)
   - Metadata with various types

### Data Generation

- **Hypothesis Strategies**: Generate valid/invalid inputs systematically
- **Deterministic Seeds**: Ensure reproducible random data
- **Edge Cases**: Empty inputs, maximum sizes, boundary values

## Performance Testing

While not the primary focus, tests should monitor:
- Operation latency
- Memory usage
- Concurrent operation throughput
- Large dataset handling

## Error Scenarios

### Must Test
1. ChromaDB connection failures
2. Embedding API failures
3. Database lock conflicts
4. Memory exhaustion
5. Disk space issues
6. Malformed input data
7. Concurrent modification conflicts
8. Network timeouts

### Recovery Testing
- Circuit breaker activation and recovery
- Retry mechanism effectiveness
- Graceful degradation
- Transaction rollback completeness

## CI/CD Integration

### Test Execution
```bash
# Run all ChromaDB tests
pytest tests/ChromaDB/ -v

# Run by category
pytest tests/ChromaDB/unit/ -m unit
pytest tests/ChromaDB/integration/ -m integration
pytest tests/ChromaDB/property/ -m property

# Run with coverage
pytest tests/ChromaDB/ --cov=tldw_Server_API.app.core.Embeddings --cov-report=html
```

### Markers
- `@pytest.mark.unit`: Unit tests
- `@pytest.mark.integration`: Integration tests
- `@pytest.mark.property`: Property-based tests
- `@pytest.mark.slow`: Tests taking >1 second
- `@pytest.mark.requires_model`: Tests needing ML models
- `@pytest.mark.concurrent`: Concurrent operation tests

## Success Metrics

### Coverage Goals
- Line coverage: >90%
- Branch coverage: >85%
- All public APIs tested
- All error paths tested

### Quality Indicators
- Zero skipped tests
- No test interdependencies
- All tests pass consistently
- Tests complete in <60 seconds (excluding slow marked tests)

## Maintenance Guidelines

### Adding New Tests
1. Determine test category (unit/integration/property)
2. Use existing fixtures when possible
3. Follow naming convention: `test_{component}_{scenario}`
4. Include docstring explaining test purpose
5. Mark with appropriate pytest markers

### Test Review Checklist
- [ ] Tests are deterministic
- [ ] No hard-coded paths or credentials
- [ ] Proper cleanup in fixtures
- [ ] Clear assertion messages
- [ ] Appropriate use of mocking
- [ ] Tests isolated from each other

## Known Limitations

1. **Model Dependencies**: Some tests require downloading ML models
2. **Performance Tests**: Limited performance testing in unit tests
3. **Platform Specific**: Some tests may behave differently on different OS
4. **External Services**: LLM API tests require mock responses

## Future Improvements

1. Add performance benchmarks
2. Implement chaos testing for resilience
3. Add cross-platform testing
4. Create test data generators for large-scale testing
5. Implement continuous fuzzing

## Conclusion

This comprehensive testing strategy ensures the ChromaDB module is robust, secure, and reliable. By following the three-tier testing approach with strict mocking rules, we achieve high confidence in the module's behavior while maintaining test maintainability and execution speed.