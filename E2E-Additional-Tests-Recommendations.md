# Additional E2E Tests Recommendations - Revised

## Overview
This document identifies gaps in the current E2E test suite and recommends additional tests specifically relevant to the tldw_server REST API architecture.

## Architecture Context
- **Type**: REST API server (FastAPI)
- **Database**: SQLite with FTS5, parameterized queries
- **Vector Store**: ChromaDB for embeddings
- **Response Format**: JSON (no HTML rendering)
- **Security**: JWT tokens, rate limiting, input validation
- **Processing**: Background jobs for transcription, embeddings

## Current Coverage Analysis

### Well-Covered Areas ✅
- Basic CRUD operations for media, notes, prompts, characters
- User authentication flow (JWT-based)
- Simple chat completions
- Character-based chat with personality
- Basic RAG search functionality
- File upload and processing
- Soft delete operations
- Character editing with optimistic locking

### Gaps Identified 🔍
1. Database transaction handling and rollback scenarios
2. Concurrent operation conflicts (optimistic locking)
3. Background job processing and failure handling
4. External service resilience (LLM providers, embeddings)
5. Media processing edge cases (FFmpeg, Whisper)
6. FTS5 and vector search advanced features
7. Performance under sustained load
8. Complete workflow integration tests

## Recommended Additional Tests

### Priority 1: Database & Concurrency 🔴

#### 1.1 Transaction Handling
```python
def test_database_transaction_atomicity():
    """Test database transaction rollback on failure."""
    # Start multi-step operation (create media + notes + tags)
    # Force failure mid-transaction
    # Verify complete rollback (no partial data)
    # Check database consistency
    # Verify version numbers unchanged

def test_optimistic_locking_conflicts():
    """Test concurrent updates with version control."""
    # Simultaneous updates to same resource
    # Verify version conflict detection
    # Test retry mechanisms
    # Validate final state consistency
    # Check all entities using versions (Media, Notes, Characters)

def test_soft_delete_recovery():
    """Test soft delete and recovery mechanisms."""
    # Soft delete items (media, notes, characters)
    # Verify excluded from normal queries
    # Test recovery/undelete
    # Check version increments on recovery
    # Test permanent deletion after soft delete
    # Verify cascade behavior for related entities
```

#### 1.2 UUID and Sync Conflicts
```python
def test_uuid_collision_handling():
    """Test UUID uniqueness and collision handling."""
    # Attempt to create items with duplicate UUIDs
    # Test UUID generation uniqueness
    # Verify sync conflict resolution
    # Test client_id tracking
    # Check merge strategies for conflicts

def test_multi_client_sync():
    """Test multi-client synchronization."""
    # Simulate multiple clients (different client_ids)
    # Concurrent modifications
    # Test last-write-wins strategy
    # Verify sync_log entries
    # Test conflict resolution
```

### Priority 2: Media Processing & Background Jobs 🟡

#### 2.1 Media Processing Edge Cases
```python
def test_ffmpeg_failure_handling():
    """Test audio/video processing failures."""
    # Upload video with unsupported codec
    # Audio file with corruption
    # Test conversion timeout handling
    # Verify cleanup after failure
    # Check error reporting to user
    # Test retry mechanisms

def test_whisper_transcription_edge_cases():
    """Test transcription service edge cases."""
    # Very long audio (>2 hours)
    # Multiple languages in same file
    # Silent audio files
    # Very noisy audio
    # Test transcription queue handling
    # Verify chunking for long files

def test_document_parsing_failures():
    """Test document processing edge cases."""
    # Encrypted PDFs
    # Corrupted EPUB files
    # DOCX with complex formatting
    # Very large documents (>1000 pages)
    # Test memory management during parsing
    # Verify partial extraction handling
```

#### 2.2 Background Job Processing
```python
def test_embedding_generation_queue():
    """Test embedding generation job queue."""
    # Queue 100+ embedding jobs
    # Test worker pool handling
    # Verify job failure recovery
    # Test job prioritization
    # Monitor memory usage
    # Test queue persistence across restarts

def test_background_job_cancellation():
    """Test canceling long-running jobs."""
    # Start long transcription
    # Cancel mid-process
    # Verify cleanup
    # Test partial result handling
    # Check resource release
```

### Priority 3: External Service Resilience 🟢

#### 3.1 LLM Provider Failures
```python
def test_llm_provider_failover():
    """Test handling of LLM provider failures."""
    # Simulate OpenAI API down
    # Test fallback to alternative provider
    # Verify graceful degradation
    # Test retry with exponential backoff
    # Check error message clarity
    # Test provider-specific error handling

def test_llm_rate_limit_handling():
    """Test LLM API rate limit handling."""
    # Trigger rate limits
    # Verify backoff behavior
    # Test request queuing
    # Check user notification
    # Test different rate limit types (TPM, RPM)
```

#### 3.2 Embedding Service Failures
```python
def test_embedding_service_resilience():
    """Test embedding service failure handling."""
    # ChromaDB connection failure
    # Embedding model loading failure
    # Out of memory during embedding
    # Test fallback to CPU
    # Verify partial batch handling
```

### Priority 4: Search & Retrieval Features 🔵

#### 4.1 FTS5 Advanced Search
```python
def test_fts5_search_capabilities():
    """Test SQLite FTS5 search features."""
    # Phrase searches with quotes
    # Boolean operators (AND, OR, NOT)
    # Prefix searches (term*)
    # NEAR operator for proximity
    # Column-specific searches
    # Test search result ranking
    # Verify snippet generation
    # Test highlight functionality

def test_fts5_index_optimization():
    """Test FTS5 index maintenance."""
    # Large bulk inserts
    # Test index rebuild
    # Verify search performance
    # Test incremental indexing
    # Check index size management
```

#### 4.2 Vector Search Operations
```python
def test_chromadb_vector_operations():
    """Test ChromaDB vector storage and retrieval."""
    # Store embeddings with metadata
    # Test similarity search accuracy
    # Verify metadata filtering
    # Test collection management
    # Check persistence across restarts
    # Test memory usage with large collections

def test_hybrid_search_ranking():
    """Test hybrid BM25 + vector search."""
    # Compare BM25 only vs vector only vs hybrid
    # Test weight adjustments
    # Verify result merging
    # Test re-ranking effectiveness
    # Check performance impact
```

### Priority 5: API-Specific Features 🟣

#### 5.1 Rate Limiting & Throttling
```python
def test_api_rate_limiting():
    """Test rate limiting enforcement."""
    # Rapid sequential requests
    # Verify 429 responses
    # Test per-endpoint limits
    # Check rate limit headers
    # Test authenticated vs anonymous limits
    # Verify rate limit reset

def test_request_throttling():
    """Test request throttling under load."""
    # Concurrent requests from multiple IPs
    # Test queue management
    # Verify fairness
    # Check timeout handling
    # Test circuit breaker activation
```

#### 5.2 Input Validation
```python
def test_file_upload_validation():
    """Test file upload security and validation."""
    # File size limits (>100MB)
    # Dangerous file types (.exe, .sh)
    # Path traversal in filenames
    # Unicode in filenames
    # Test virus scanning if configured
    # Verify file type detection

def test_input_sanitization():
    """Test input data sanitization."""
    # Very long strings (>1MB)
    # Special characters in text fields
    # Invalid UTF-8 sequences
    # Control characters
    # Test truncation behavior
    # Verify data integrity after sanitization
```

### Priority 6: Performance & Load Testing 🟤

#### 6.1 Sustained Load Testing
```python
def test_sustained_api_load():
    """Test API under sustained load."""
    # 100 requests/second for 30 minutes
    # Monitor response times (p50, p95, p99)
    # Check memory usage trend
    # Verify no memory leaks
    # Test connection pool exhaustion
    # Monitor database lock contention

def test_bulk_operations_performance():
    """Test performance of bulk operations."""
    # Upload 100 files concurrently
    # Create 1000 notes rapidly
    # Process 50 documents simultaneously
    # Measure throughput
    # Check for deadlocks
    # Monitor resource usage
```

#### 6.2 Database Performance
```python
def test_database_query_performance():
    """Test database query optimization."""
    # Complex JOIN queries
    # Full table scans
    # Test with 100K+ records
    # Verify index usage
    # Check query plan efficiency
    # Test vacuum performance

def test_database_connection_pooling():
    """Test connection pool management."""
    # Max connections stress test
    # Connection timeout handling
    # Pool exhaustion recovery
    # Test with long-running transactions
    # Verify proper connection release
```

### Priority 7: Complete Workflow Integration 🟠

#### 7.1 Research Assistant Workflow
```python
def test_research_workflow_integration():
    """Complete research assistant workflow."""
    # Upload multiple research papers (PDF, EPUB)
    # Wait for transcription/extraction
    # Generate embeddings
    # Create summary notes from each
    # Use RAG to find cross-paper connections
    # Generate comprehensive literature review
    # Export results with citations
    # Verify data consistency throughout

def test_knowledge_base_building():
    """Test building a searchable knowledge base."""
    # Bulk import documentation (100+ files)
    # Auto-categorization with tags
    # Generate embeddings for all content
    # Test incremental updates
    # Verify search accuracy
    # Test performance with large dataset
    # Monitor index growth
```

#### 7.2 Content Processing Pipeline
```python
def test_media_pipeline_end_to_end():
    """Test complete media processing pipeline."""
    # Upload video with multiple speakers
    # Verify transcription with diarization
    # Test chapter detection
    # Generate summary
    # Create searchable chunks
    # Test playback position sync
    # Export transcript with timestamps
```

### Priority 8: Monitoring & Observability 🔷

#### 8.1 Metrics and Monitoring
```python
def test_metrics_endpoint():
    """Test metrics collection and reporting."""
    # Verify Prometheus metrics format
    # Check metric accuracy
    # Test custom metrics
    # Verify no sensitive data in metrics
    # Test metric aggregation
    # Check cardinality limits

def test_health_check_accuracy():
    """Test health check endpoint accuracy."""
    # Database connection failure
    # ChromaDB unavailable
    # Disk space issues
    # Memory pressure
    # Verify accurate status reporting
    # Test dependent service checks
```

## Implementation Strategy

### Test Organization
```
tests/e2e/
├── test_full_user_workflow.py          # Current comprehensive test
├── test_database_operations.py         # Database transactions, concurrency
├── test_media_processing.py            # FFmpeg, Whisper, document parsing
├── test_background_jobs.py             # Queue processing, job management
├── test_external_services.py           # LLM, embedding service resilience
├── test_search_features.py             # FTS5, ChromaDB, hybrid search
├── test_api_features.py                # Rate limiting, validation
├── test_performance.py                 # Load testing, benchmarks
└── test_integration_workflows.py       # Complete user journeys
```

### Execution Strategy

1. **Test Categories**
   - **Smoke** (5 min): Critical path tests
   - **Integration** (30 min): API functionality
   - **Full** (2 hours): Complete test suite
   - **Performance** (4 hours): Load and stress tests

2. **Test Isolation**
   - Separate SQLite database per test class
   - Isolated ChromaDB collections
   - Mock external services where appropriate
   - Clean background job queues

3. **Performance Baselines**
   - API response time: p95 < 500ms
   - Transcription: 1 hour audio < 5 minutes
   - Embedding generation: 100 docs/minute
   - Search latency: < 100ms for 100K documents

## Success Metrics

### Coverage Goals
- API endpoint coverage: 100%
- Database transaction paths: > 90%
- Error handling paths: > 85%
- Background job scenarios: > 80%

### Reliability Metrics
- Test flakiness: < 2%
- False positives: < 1%
- Test execution time: < 30 min for PR suite
- Performance regression detection: 100%

## Next Steps

1. **Immediate (Week 1)**
   - Implement database transaction tests
   - Add concurrent operation tests
   - Test soft delete/recovery

2. **Short Term (Week 2-3)**
   - Media processing edge cases
   - Background job handling
   - External service resilience

3. **Medium Term (Month 1-2)**
   - Performance benchmarks
   - Load testing suite
   - Complete workflow tests

4. **Ongoing**
   - Monitor test metrics
   - Update based on production issues
   - Expand coverage for new features

## Conclusion

These recommendations focus on the specific architecture and requirements of the tldw_server REST API. The tests emphasize:

1. **Database integrity** through transaction and concurrency testing
2. **Service resilience** with proper failure handling
3. **Processing reliability** for media and background jobs
4. **Search capabilities** leveraging FTS5 and vector stores
5. **API robustness** through validation and rate limiting
6. **Performance assurance** under realistic loads

This targeted approach will provide confidence in the API's reliability and performance while avoiding irrelevant tests for features not present in a REST API architecture.