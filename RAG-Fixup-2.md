# RAG Module Complete Refactoring Plan

## Current Feature Inventory

### Core Pipeline Features (functional_pipeline.py)
- ✅ Query expansion (multiple strategies)
- ✅ Caching (check & store)
- ✅ Document retrieval
- ✅ ChromaDB optimization
- ✅ Table processing
- ✅ Document reranking
- ✅ Performance analysis
- ✅ Multiple pipeline presets (minimal, standard, quality, enhanced)
- ✅ Custom pipeline building
- ✅ Parallel & conditional execution

### Advanced Features (separate modules)
1. **Data Sources** (database_retrievers.py)
   - Media DB retrieval
   - Notes DB retrieval
   - Prompts DB retrieval
   - Character Cards retrieval
   - Multi-database search

2. **Processing Features**
   - Enhanced chunking (enhanced_chunking_integration.py)
   - Document processing (document_processing_integration.py)
   - Table serialization (table_serialization.py)
   - Parent document retrieval (parent_retrieval.py)
   - Citations generation (citations.py)

3. **Search & Ranking**
   - Query expansion strategies (query_expansion.py)
   - Advanced reranking (advanced_reranking.py)
   - Query features extraction (query_features.py)
   - Security filters (security_filters.py)

4. **Performance & Optimization**
   - Advanced caching (advanced_cache.py, semantic_cache.py)
   - ChromaDB optimization for 100k+ docs (chromadb_optimizer.py)
   - Batch processing (batch_processing.py)
   - Performance monitoring (performance_monitor.py)
   - Metrics collection (metrics_collector.py)

5. **Production Features**
   - Resilience (circuit breakers, retries) (resilience.py)
   - Observability & logging (observability.py)
   - Feedback system (feedback_system.py)
   - Advanced configuration (advanced_config.py)
   - Generation with LLMs (generation.py)
   - Prompt templates (prompt_templates.py)

6. **Quick Wins** (quick_wins.py)
   - Spell checking
   - Result highlighting
   - Cost tracking
   - Query templates
   - Domain-specific dictionaries

## Refactoring Plan with All Features

### Phase 1: Fix Core Architecture (Day 1)
1. **Fix imports and initialization**
   - Update `__init__.py` files to export functional components
   - Remove broken imports of RAGService class
   - Ensure functional_pipeline.py can import all modules

2. **Consolidate endpoints**
   - Keep only `rag_api.py` as main endpoint
   - Ensure it exposes all pipeline options
   - Remove redundant endpoints

3. **Fix database retrievers integration**
   - Ensure all retrievers work with functional pipeline
   - Test connection to each database type

### Phase 2: Integrate All Features (Day 1-2)
1. **Wire up all processing features**
   - Enhanced chunking in pipeline
   - Parent document retrieval
   - Citations generation
   - Table processing

2. **Complete search features**
   - All query expansion strategies
   - Advanced reranking options
   - Security filters for sensitive data

3. **Enable caching layers**
   - Semantic cache for similar queries
   - Advanced cache with TTL
   - Result caching

### Phase 3: Production Features (Day 2)
1. **Add resilience**
   - Circuit breakers for external services
   - Retry logic with exponential backoff
   - Graceful degradation

2. **Enable monitoring**
   - Performance metrics collection
   - Request/response logging
   - Cost tracking for LLM calls

3. **Quick wins integration**
   - Spell checking on queries
   - Result highlighting
   - Query templates

### Phase 4: Testing & Documentation (Day 2-3)
1. **Comprehensive testing**
   - Unit tests for each module
   - Integration tests for pipelines
   - Performance benchmarks
   - Load testing

2. **Documentation**
   - API documentation with examples
   - Configuration guide
   - Feature matrix
   - Performance tuning guide

## Implementation Checklist

### Immediate Actions
- [ ] Fix `__init__.py` files
- [ ] Remove old endpoint files or update them
- [ ] Update main.py routing
- [ ] Create basic integration test

### Core Features
- [ ] Database retrievers working
- [ ] Query expansion functional
- [ ] Caching operational
- [ ] Reranking enabled
- [ ] ChromaDB search optimized

### Advanced Features
- [ ] Enhanced chunking integrated
- [ ] Parent retrieval working
- [ ] Citations generation
- [ ] Table processing
- [ ] Security filters active

### Production Ready
- [ ] Circuit breakers configured
- [ ] Metrics collection enabled
- [ ] Performance monitoring active
- [ ] Feedback system operational
- [ ] Cost tracking implemented

### Quality Assurance
- [ ] All tests passing
- [ ] API documentation complete
- [ ] Configuration validated
- [ ] Performance benchmarked

## Configuration Requirements

The system should support configuration for:
- Pipeline selection (minimal/standard/quality/custom)
- Cache settings (TTL, size limits)
- Database connections
- Reranking strategies
- Query expansion options
- Security filters
- Performance thresholds
- Cost limits

## Expected Outcomes

After refactoring:
1. **Single, cohesive API** with all features accessible
2. **Functional pipeline** as the core architecture
3. **All features integrated** and working
4. **Production-ready** with monitoring, resilience, and performance optimization
5. **Well-tested** with comprehensive test coverage
6. **Documented** with clear usage examples

## Progress Tracking

### Current Status: Phase 4 - Documentation

#### TODO List:
1. [COMPLETED] Fix __init__.py files to export functional components
2. [COMPLETED] Remove/update broken RAG endpoints
3. [COMPLETED] Update main.py routing to use single RAG endpoint
4. [COMPLETED] Verify database retrievers work with functional pipeline
5. [COMPLETED] Create basic integration test
6. [COMPLETED] Wire up all processing features
7. [COMPLETED] Enable caching and monitoring
8. [COMPLETED] Add production features (resilience, observability)
9. [COMPLETED] Create comprehensive tests
10. [IN PROGRESS] Document the refactored system

### Completed Work Summary:

#### Phase 1: Core Architecture Fix
- ✅ Fixed `__init__.py` files to export functional pipeline components
- ✅ Removed deprecated rag_v2.py and rag_v3_functional.py endpoints
- ✅ Updated main.py to use single RAG endpoint (rag_api.py)
- ✅ Fixed conftest.py to remove old imports
- ✅ Created basic functional pipeline tests
- ✅ Verified server starts and endpoints are registered correctly

#### Phase 2: Feature Integration
- ✅ Fixed query expansion to use correct class methods
- ✅ Integrated database retrievers with proper RetrievalConfig
- ✅ Connected all processing features (chunking, tables, reranking)
- ✅ Enabled semantic caching with adaptive thresholds
- ✅ Added performance monitoring and analysis

#### Phase 3: Production Features
- ✅ Added resilience decorator (`@with_resilience`) for optional fault tolerance
- ✅ Integrated circuit breakers and retry logic as configuration options
- ✅ Added fallback functions for critical operations
- ✅ Resilience is opt-in via config - keeps pipelines simple by default

#### Phase 4: Testing & Documentation
- ✅ Created comprehensive test suite (test_rag_refactored.py)
- ✅ Tested all pipeline variants (minimal, standard, quality)
- ✅ Verified resilience features work when enabled
- ✅ Documented architecture and usage patterns