# RAG Module Test Strategy

## Overview

This document outlines the comprehensive testing strategy for the RAG (Retrieval-Augmented Generation) module, which is the core of the tldw_server's information retrieval and generation capabilities. The strategy follows a three-tier testing pyramid with unit, integration, and property-based tests.

## Module Architecture

The RAG module is complex with multiple components:

1. **Functional Pipeline**: Composable pipeline functions for flexible RAG workflows
2. **Unified Pipeline**: Single entry point with all features accessible via parameters
3. **Database Retrievers**: Multi-source retrieval from media DB, vector stores, etc.
4. **Query Expansion**: Synonym, acronym, and entity-based query enhancement
5. **Reranking**: Cross-encoder and semantic reranking strategies
6. **Caching**: Semantic cache for improved performance
7. **Generation**: Answer generation with LLM integration
8. **Citations**: Academic and inline citation generation

## Testing Philosophy

### Core Principles

1. **No Mocking in Integration/Property Tests**: These tests use real components
2. **Minimal Mocking in Unit Tests**: Only mock external services (LLMs, APIs)
3. **Comprehensive Coverage**: Every pipeline stage and configuration must be tested
4. **Property-Based Testing**: Verify invariants across all valid inputs
5. **Performance Testing**: Ensure scalability and efficiency

### Test Categories

#### 1. Unit Tests (50% of tests)
- **Location**: `tests/RAG_NEW/unit/`
- **Files Created**:
  - `test_functional_pipeline.py` - 50+ tests for pipeline functions
  - `test_unified_pipeline.py` - 40+ tests for unified pipeline
  - `test_retrieval.py` - 45+ tests for retrieval components
- **Purpose**: Test individual components in isolation
- **Mocking**: External services only (LLMs, vector stores)
- **Focus**: Business logic, error handling, configuration

#### 2. Integration Tests (30% of tests)
- **Location**: `tests/RAG_NEW/integration/`
- **Files Created**:
  - `test_rag_integration.py` - 35+ tests for end-to-end workflows
- **Purpose**: Test component interactions with real databases
- **Mocking**: None (use real MediaDatabase, real cache)
- **Focus**: End-to-end pipelines, performance, concurrency

#### 3. Property Tests (20% of tests)
- **Location**: `tests/RAG_NEW/property/`
- **Files Created**:
  - `test_rag_properties.py` - 30+ property tests with Hypothesis
- **Purpose**: Verify invariants hold across all inputs
- **Mocking**: None
- **Focus**: Mathematical properties, consistency, boundaries

## Test Coverage Areas

### 1. Pipeline Components

#### Functional Pipeline Tests
- Pipeline context initialization and modification
- Timer decorator functionality
- Query expansion (disabled/enabled/error cases)
- Cache operations (hit/miss/store)
- Document retrieval with filters
- Reranking with different strategies
- Answer generation with/without context
- Pipeline building and composition
- Performance analysis

#### Unified Pipeline Tests
- Minimal parameter execution
- All features enabled simultaneously
- Caching behavior
- Query expansion strategies
- Reranking algorithms
- Security filtering
- Citation generation
- Streaming support
- Error handling and fallbacks
- Metadata preservation

### 2. Retrieval System

#### Database Retriever Tests
- MediaDatabase retrieval
- Vector store retrieval
- Hybrid retrieval (BM25 + vectors)
- Multi-database coordination
- Parent document retrieval
- Score filtering
- Metadata filtering
- Batch retrieval
- Parallel retrieval
- Error recovery

### 3. Query Processing

#### Expansion Tests
- Acronym expansion
- Synonym expansion
- Entity recognition
- Multi-strategy expansion
- Expansion preservation
- Error handling

#### Reranking Tests
- Semantic reranking
- Cross-encoder reranking
- Score fusion
- Top-k selection
- Document preservation
- Performance impact

### 4. Caching System

#### Cache Tests
- Semantic similarity matching
- TTL expiration
- Cache invalidation
- Size limits
- LRU eviction
- Performance benefits
- Deterministic key generation

### 5. Error Handling

#### Resilience Tests
- Database failures
- LLM API failures
- Partial retrieval failures
- Retry mechanisms
- Circuit breakers
- Fallback strategies
- Graceful degradation

## Property Test Coverage

### Invariants Verified

1. **Pipeline Context**:
   - Original query preservation
   - Document consistency
   - Error tracking completeness

2. **Retrieval**:
   - Result count ≤ min(available, top_k)
   - Score ordering (descending)
   - Score filtering correctness
   - Document uniqueness

3. **Query Expansion**:
   - Original query preserved
   - Additive only (no removal)
   - Format consistency

4. **Reranking**:
   - Document preservation (no modification)
   - Count invariants
   - Relevance improvement

5. **Caching**:
   - Deterministic key generation
   - TTL behavior
   - Size limit enforcement

6. **Error Handling**:
   - Invalid query handling
   - Invalid configuration handling
   - No crashes on edge cases

## Test Data Strategy

### Fixtures

1. **Database Fixtures**:
   - Real MediaDatabase with schema
   - Populated test database
   - Empty database
   - Failing database (for error testing)

2. **Document Fixtures**:
   - Sample documents with metadata
   - Search results with scores
   - Malicious/sensitive documents
   - Large documents for performance

3. **Configuration Fixtures**:
   - Basic configurations
   - Advanced configurations
   - Invalid configurations
   - Performance test configs

### Data Generation

- **Hypothesis Strategies**: 
  - Valid queries
  - Document generation
  - Configuration generation
  - Score generation
- **Deterministic Seeds**: For reproducible tests
- **Edge Cases**: Empty inputs, maximum sizes, special characters

## Performance Testing

### Metrics Monitored
- Pipeline execution time
- Retrieval latency
- Cache hit rates
- Memory usage
- Concurrent request handling
- Large document processing

### Performance Goals
- Minimal pipeline: < 100ms for simple queries
- Standard pipeline: < 500ms with cache hit
- Quality pipeline: < 2s with all features
- Concurrent handling: 20+ simultaneous requests
- Memory stability: < 100MB growth over 50 requests

## Integration Points

### Components Tested Together
1. MediaDatabase + Retriever
2. Cache + Pipeline
3. Query Expansion + Retrieval
4. Retrieval + Reranking
5. Multiple Retrievers + Score Fusion
6. Pipeline + Error Recovery

### Real Component Testing
- Actual database operations
- Real embedding generation (when available)
- Actual cache persistence
- Real file I/O operations

## CI/CD Integration

### Test Execution
```bash
# Run all RAG tests
pytest tests/RAG_NEW/ -v

# Run by category
pytest tests/RAG_NEW/unit/ -m unit
pytest tests/RAG_NEW/integration/ -m integration
pytest tests/RAG_NEW/property/ -m property

# Run with coverage
pytest tests/RAG_NEW/ --cov=tldw_Server_API.app.core.RAG --cov-report=html
```

### Markers
- `@pytest.mark.unit`: Unit tests
- `@pytest.mark.integration`: Integration tests
- `@pytest.mark.property`: Property tests
- `@pytest.mark.slow`: Tests > 1 second
- `@pytest.mark.requires_llm`: Needs LLM API
- `@pytest.mark.pipeline`: Pipeline-specific

## Success Metrics

### Coverage Goals
- Line coverage: > 85%
- Branch coverage: > 80%
- All public APIs tested
- All error paths tested
- All configuration combinations tested

### Quality Indicators
- Zero skipped tests
- No test interdependencies
- All tests deterministic
- Tests complete in < 60 seconds (excluding slow marked)
- Clear test names and documentation

## Maintenance Guidelines

### Adding New Tests
1. Determine category (unit/integration/property)
2. Use existing fixtures
3. Follow naming: `test_{component}_{scenario}`
4. Include docstring
5. Add appropriate markers
6. Verify no mocking in integration/property tests

### Test Review Checklist
- [ ] Tests are deterministic
- [ ] No hard-coded paths
- [ ] Proper cleanup in fixtures
- [ ] Clear assertions
- [ ] Appropriate mocking (unit only)
- [ ] Tests isolated
- [ ] Performance acceptable

## Known Limitations

1. **LLM Dependencies**: Some features require LLM API mocking
2. **Vector Store**: Full vector search requires ChromaDB setup
3. **Large Models**: Embedding models may be mocked for speed
4. **External APIs**: Third-party services are mocked

## Migration from Old Tests

### Old Test Issues
- 25 files with mixed quality
- Many skipped tests
- Excessive mocking
- Poor separation of concerns
- Deprecated implementations tested

### New Test Improvements
- Clear separation (unit/integration/property)
- Zero skipped tests
- Minimal mocking
- Comprehensive fixtures
- Current implementation focus
- Property-based testing added

## Future Improvements

1. **Benchmark Suite**: Add performance regression tests
2. **Load Testing**: Add stress tests for production scenarios
3. **Security Testing**: Add injection and vulnerability tests
4. **Model Testing**: Add tests for different LLM providers
5. **Visualization**: Add test result dashboards

## Conclusion

This comprehensive testing strategy ensures the RAG module is robust, performant, and reliable. The three-tier approach with strict mocking rules provides high confidence while maintaining fast test execution. The property-based tests verify mathematical correctness across all inputs, while integration tests ensure real-world functionality.