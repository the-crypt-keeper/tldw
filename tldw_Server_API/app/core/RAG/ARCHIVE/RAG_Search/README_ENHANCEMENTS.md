# RAG Search Enhancements Documentation

This document provides comprehensive documentation for the enhanced RAG (Retrieval-Augmented Generation) search components added to the tldw_server project.

## Overview

The RAG enhancements introduce advanced capabilities for improving search quality, performance, and flexibility. These components work together to provide a more intelligent and efficient search experience.

## Components

### 1. Advanced Query Expansion (`advanced_query_expansion.py`)

Expands user queries using multiple strategies to improve recall and find relevant documents that might use different terminology.

#### Features
- **Multiple Expansion Strategies**:
  - `SEMANTIC`: Find semantically similar terms using word embeddings
  - `LINGUISTIC`: Apply linguistic rules (synonyms, stemming, lemmatization)
  - `ENTITY`: Extract and expand named entities
  - `ACRONYM`: Expand/contract acronyms (e.g., "ML" ↔ "Machine Learning")
  - `DOMAIN`: Add domain-specific related terms

#### Usage
```python
from tldw_Server_API.app.core.RAG.RAG_Search.advanced_query_expansion import (
    AdvancedQueryExpander, ExpansionConfig, ExpansionStrategy
)

# Configure expander
config = ExpansionConfig(
    strategies=[ExpansionStrategy.LINGUISTIC, ExpansionStrategy.ACRONYM],
    max_expansions_per_strategy=3,
    total_max_expansions=10
)

expander = AdvancedQueryExpander(config)
expansions = await expander.expand_query("What is ML?")
# Returns: ["What is ML?", "What is Machine Learning?", "What is machine learning?", ...]
```

#### Configuration
- `strategies`: List of strategies to use
- `max_expansions_per_strategy`: Max expansions per strategy (default: 5)
- `total_max_expansions`: Total max expansions across all strategies (default: 20)
- `min_similarity_score`: Minimum similarity for semantic expansions (default: 0.7)
- `enable_caching`: Cache expansion results (default: True)

### 2. Advanced Reranking (`advanced_reranker.py`)

Reranks search results using sophisticated algorithms to improve precision and relevance.

#### Reranking Strategies

1. **Cross-Encoder**: Uses a BERT-based model to score query-document pairs
2. **LLM Scoring**: Uses LLM to score relevance (requires API key)
3. **Diversity**: Promotes diverse results using MMR algorithm
4. **Multi-Criteria**: Combines multiple scoring factors
5. **Hybrid**: Combines multiple strategies with weighted voting

#### Usage
```python
from tldw_Server_API.app.core.RAG.RAG_Search.advanced_reranker import (
    create_reranker, RerankingStrategy
)

# Create reranker
reranker = create_reranker(
    RerankingStrategy.HYBRID,
    top_k=10,
    config={"diversity_weight": 0.3}
)

# Rerank results
reranked = await reranker.rerank(
    query="machine learning applications",
    documents=[{"content": "...", "metadata": {...}}, ...]
)
```

#### Key Features
- Async/await support for all strategies
- Configurable scoring weights
- Metadata-aware reranking
- Diversity promotion to avoid redundant results

### 3. Enhanced Caching (`enhanced_cache.py`)

Multi-level caching system for improved performance.

#### Cache Strategies

1. **LRU Cache**: Fast in-memory cache with TTL support
2. **Semantic Cache**: Finds cached results for semantically similar queries
3. **Tiered Cache**: Two-tier cache with memory and disk levels
4. **Adaptive Cache**: Adjusts caching based on access patterns

#### Usage
```python
from tldw_Server_API.app.core.RAG.RAG_Search.enhanced_cache import (
    CacheManager, LRUCacheStrategy, SemanticCacheStrategy
)

# Create cache manager with multiple strategies
cache_manager = CacheManager([
    LRUCacheStrategy(max_size=1000, ttl=3600),
    SemanticCacheStrategy(similarity_threshold=0.9)
])

# Cache operations
await cache_manager.set("query_key", result_data, query="user query")
cached = await cache_manager.get("query_key")

# Get cache stats
stats = cache_manager.get_stats()
```

#### Features
- Multiple cache strategies can work together
- Query-aware caching (semantic similarity)
- Configurable eviction policies
- Performance statistics tracking

### 4. Advanced Chunking (`advanced_chunking.py`)

Intelligent document chunking strategies for better context preservation.

#### Chunking Strategies

1. **Semantic**: Creates chunks based on semantic coherence
2. **Structural**: Preserves document structure (headings, paragraphs)
3. **Adaptive**: Adjusts chunk size based on content density
4. **Sliding Window**: Overlapping chunks for context continuity
5. **Hybrid**: Combines multiple strategies

#### Usage
```python
from tldw_Server_API.app.core.RAG.RAG_Search.advanced_chunking import (
    create_chunker, ChunkingStrategy
)

# Create chunker
chunker = create_chunker(
    ChunkingStrategy.ADAPTIVE,
    chunk_size=512,
    overlap=50
)

# Chunk document
chunks = chunker.chunk(document_text)
for chunk in chunks:
    print(f"Chunk {chunk.metadata.chunk_id}: {chunk.text[:50]}...")
    print(f"  Type: {chunk.metadata.chunk_type}")
    print(f"  Size: {chunk.metadata.char_count} chars")
```

#### Features
- Metadata-rich chunks (position, type, hierarchy)
- Configurable overlap for context preservation
- Structure-aware chunking (preserves headings, code blocks)
- Sentence boundary detection

### 5. Performance Monitoring (`performance_monitor.py`)

Comprehensive performance monitoring for all RAG components.

#### Features
- Decorator-based monitoring for easy integration
- Integrates with existing metrics system
- Component-specific metrics
- End-to-end pipeline monitoring

#### Usage
```python
from tldw_Server_API.app.core.RAG.RAG_Search.performance_monitor import (
    RAGPerformanceMonitor, monitor_query_expansion, monitor_reranking
)

# Use decorators
@monitor_query_expansion
async def expand_query(query: str):
    # Your expansion logic
    pass

@monitor_reranking  
async def rerank_results(query: str, docs: List[Dict]):
    # Your reranking logic
    pass

# Get performance stats
monitor = get_performance_monitor()
summary = monitor.get_performance_summary()
```

#### Metrics Tracked
- Query expansion: input length, output count, expansion ratio
- Reranking: input/output document counts, score distributions
- Caching: hit rates, latency by strategy
- Chunking: chunk counts, size distributions, overlap ratios
- Vector search: result counts, score distributions
- End-to-end: total latency, stage breakdowns

### 6. ChromaDB Optimization (`chromadb_optimizer.py`)

ChromaDB-specific optimizations that complement its built-in features.

#### Features
- **Query Result Caching**: Caches ChromaDB query results
- **Hybrid Search Optimization**: Intelligently combines vector and FTS results
- **Batch Operations**: Optimized batch document addition
- **Connection Pooling**: Manages ChromaDB client connections

#### Usage
```python
from tldw_Server_API.app.core.RAG.RAG_Search.chromadb_optimizer import (
    OptimizedChromaStore, ChromaDBOptimizationConfig
)

# Create optimized store
config = ChromaDBOptimizationConfig(
    enable_result_cache=True,
    hybrid_alpha=0.7,
    batch_size=100
)

store = OptimizedChromaStore(
    path="/path/to/chromadb",
    collection_name="documents",
    optimization_config=config
)

# Use like regular ChromaDB but with optimizations
results = await store.search(
    query_text="machine learning",
    n_results=10
)

# Hybrid search
hybrid_results = await store.hybrid_search(
    query_text="machine learning",
    query_embeddings=embeddings,
    fts_results=fts_results,
    n_results=10
)
```

### 7. Integration Testing (`integration_test.py`)

Comprehensive test suite for validating all components work together.

#### Test Coverage
- Individual component testing
- End-to-end pipeline testing
- Performance benchmarking
- Error handling validation

#### Running Tests
```bash
# Run all integration tests
python -m pytest app/core/RAG/RAG_Search/integration_test.py -v

# Run specific test
python -m pytest app/core/RAG/RAG_Search/integration_test.py::test_query_expansion_integration -v

# Run from module directly
python app/core/RAG/RAG_Search/integration_test.py
```

## Integration with Existing System

### 1. With RAG Service
The enhanced components integrate seamlessly with the existing RAG service:

```python
from tldw_Server_API.app.core.RAG.rag_service.integration import RAGService
from tldw_Server_API.app.core.RAG.RAG_Search.advanced_query_expansion import AdvancedQueryExpander
from tldw_Server_API.app.core.RAG.RAG_Search.advanced_reranker import create_reranker

# Enhance RAG service
rag_service = RAGService(config=config)
rag_service.query_expander = AdvancedQueryExpander()
rag_service.reranker = create_reranker(RerankingStrategy.HYBRID)
```

### 2. With API Endpoints
The components are used in the `/retrieval` endpoints:

- `/retrieval/search`: Uses query expansion and reranking
- `/retrieval/agent`: Full pipeline with all enhancements

### 3. With Existing Databases
Works with existing SQLite and ChromaDB databases without modification.

## Configuration Best Practices

### For Development
```python
# Fast iteration, lower quality
config = {
    "query_expansion": {
        "strategies": ["LINGUISTIC"],  # Fast strategies only
        "max_expansions": 5
    },
    "reranking": {
        "strategy": "DIVERSITY",  # No external API calls
        "top_k": 10
    },
    "caching": {
        "strategies": ["LRU"],  # Simple caching
        "size": 100
    }
}
```

### For Production
```python
# High quality, with caching for performance
config = {
    "query_expansion": {
        "strategies": ["SEMANTIC", "LINGUISTIC", "ACRONYM", "DOMAIN"],
        "max_expansions": 20,
        "enable_caching": True
    },
    "reranking": {
        "strategy": "HYBRID",  # Best quality
        "top_k": 20,
        "diversity_weight": 0.3
    },
    "caching": {
        "strategies": ["LRU", "SEMANTIC", "TIERED"],
        "size": 10000,
        "ttl": 3600
    },
    "chunking": {
        "strategy": "ADAPTIVE",
        "chunk_size": 512,
        "overlap": 50
    }
}
```

## Performance Considerations

### Query Expansion
- Semantic expansion requires embeddings (adds ~50-100ms)
- Linguistic expansion is fast (<10ms)
- Cache hit rate typically 40-60% for repeated queries

### Reranking
- Cross-encoder: ~100-200ms for 20 documents
- LLM scoring: ~500-2000ms depending on provider
- Diversity/Multi-criteria: <50ms

### Caching
- LRU cache: <1ms lookup
- Semantic cache: ~50ms (embedding computation)
- Tiered cache: Memory <1ms, Disk ~10-50ms

### Chunking
- Throughput: ~1-5 MB/s depending on strategy
- Adaptive chunking: ~2x slower than fixed-size
- Structural chunking: Fastest for structured documents

## Troubleshooting

### Common Issues

1. **Import Errors**
   ```python
   # Ensure PYTHONPATH includes project root
   export PYTHONPATH=/path/to/tldw_server:$PYTHONPATH
   ```

2. **Semantic Features Not Working**
   - Check if sentence-transformers is installed
   - Verify model downloads completed
   - Check available disk space for models

3. **Performance Issues**
   - Monitor with performance_monitor
   - Check cache hit rates
   - Reduce expansion strategies
   - Use simpler reranking strategies

4. **Memory Usage**
   - Reduce cache sizes
   - Use disk-based caching
   - Limit batch sizes

## Future Enhancements

1. **Query Expansion**
   - User-specific expansion learning
   - Multi-language support
   - Query intent detection

2. **Reranking**
   - Fine-tuned ranking models
   - User preference learning
   - Real-time feedback incorporation

3. **Caching**
   - Distributed caching support
   - Predictive cache warming
   - Cache compression

4. **Performance**
   - GPU acceleration for embeddings
   - Async batch processing
   - Query planning optimization

## API Reference

See individual module docstrings for detailed API documentation:
- `advanced_query_expansion.py`
- `advanced_reranker.py`
- `enhanced_cache.py`
- `advanced_chunking.py`
- `performance_monitor.py`
- `chromadb_optimizer.py`

## Contributing

When adding new enhancements:
1. Follow existing patterns and interfaces
2. Add comprehensive tests
3. Update this documentation
4. Add performance monitoring
5. Consider backward compatibility