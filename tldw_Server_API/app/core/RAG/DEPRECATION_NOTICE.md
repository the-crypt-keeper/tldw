# RAG Module Deprecation Notice

## Current Architecture (v3.0 - Functional Pipeline)

The RAG system has been refactored to use a **functional pipeline architecture** as of 2024-08-19.

### Active Modules (Keep These)

Located in `/app/core/RAG/rag_service/`:

1. **functional_pipeline.py** - Main functional pipeline implementation
2. **table_serialization.py** - Table processing functionality
3. **query_expansion.py** - Query expansion strategies
4. **semantic_cache.py** - Semantic caching implementation
5. **advanced_reranking.py** - Document reranking strategies
6. **performance_monitor.py** - Performance monitoring
7. **chromadb_optimizer.py** - ChromaDB optimizations for 100k+ docs
8. **types.py** - Core type definitions
9. **config.py** - Configuration management

### Deprecated Modules (To Be Archived)

These modules are replaced by the functional pipeline and should be archived:

#### Object-Oriented Pipeline (Replaced by functional_pipeline.py)
- **app.py** - Old RAGApplication class
- **integration.py** - Old RAGService class
- **pipeline_orchestrator.py** - Object-oriented pipeline (replaced by functional approach)

#### Old Retrieval/Processing (Now integrated in functional_pipeline.py)
- **retrieval.py** - Separate retriever classes
- **processing.py** - Separate processor classes
- **generation.py** - Separate generator classes

#### Duplicates/Old Implementations
- **cache.py** - Replaced by semantic_cache.py
- **metrics.py** - Replaced by performance_monitor.py
- **enhanced_chunking.py** - Functionality in table_serialization.py
- **connection_pool.py** - Not needed with functional approach
- **citation_retriever.py** - Not implemented in current pipeline
- **parent_retriever.py** - Not implemented in current pipeline

#### Example/Test Files
- **example_usage.py** - Old examples
- **tui_example.py** - Old TUI examples

### API Endpoints

#### Active
- **/api/v1/rag/v3/** - New functional pipeline API (rag_v3_functional.py)

#### To Be Deprecated
- **/api/v1/rag/** - Old v1 endpoints
- **/api/v1/rag/v2/** - v2 endpoints (can remain for backward compatibility)

### Migration Path

1. **For API Users:**
   - Switch from `/api/v1/rag/v2/search` to `/api/v1/rag/v3/search`
   - Update to use pipeline selection ("minimal", "standard", "quality", "custom")
   - Configuration now passed as a dictionary

2. **For Direct Module Users:**
   ```python
   # Old way (deprecated)
   from app.core.RAG.rag_service.app import RAGApplication
   app = RAGApplication(config)
   result = await app.search(query)
   
   # New way (functional)
   from app.core.RAG.rag_service.functional_pipeline import standard_pipeline
   result = await standard_pipeline(query, config)
   ```

3. **Custom Pipeline Example:**
   ```python
   from app.core.RAG.rag_service.functional_pipeline import (
       build_pipeline,
       expand_query,
       check_cache,
       retrieve_documents,
       rerank_documents
   )
   
   # Build custom pipeline
   my_pipeline = build_pipeline(
       expand_query,
       check_cache,
       retrieve_documents,
       rerank_documents
   )
   
   result = await my_pipeline(query, config)
   ```

### Archive Structure

Move deprecated files to:
```
/app/core/RAG/ARCHIVE/deprecated_v2/
├── app.py
├── integration.py
├── retrieval.py
├── processing.py
├── generation.py
├── pipeline_orchestrator.py
├── cache.py
├── metrics.py
├── enhanced_chunking.py
├── connection_pool.py
├── citation_retriever.py
├── parent_retriever.py
├── example_usage.py
└── tui_example.py
```

### Timeline

- **2024-08-19**: Functional pipeline implemented
- **2024-09-01**: Mark old modules as deprecated
- **2024-10-01**: Move deprecated modules to archive
- **2024-11-01**: Remove deprecated API endpoints

### Benefits of New Architecture

1. **Simplicity**: Pure functions instead of complex class hierarchies
2. **Flexibility**: Easy to compose custom pipelines
3. **Performance**: Less overhead, better caching
4. **Testability**: Functions are easier to test in isolation
5. **Modularity**: Each function is independent and reusable

### Contact

For questions about the migration, please contact the development team.