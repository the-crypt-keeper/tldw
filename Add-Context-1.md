# Contextual Retrieval Implementation Plan

## Objective
Expose contextual retrieval functionality to users in both Chunking and RAG pipelines as an optional toggle while adhering to baseline configuration settings.

## Implementation Status

### Phase 1: Configuration File Updates ✅
**Status: COMPLETE**

#### Updated config.txt with:
1. **[Chunking] Section**:
   - `enable_contextual_retrieval = false` (baseline default)
   - `context_window_size = 500` (characters around chunks)
   - `include_parent_context = false` (parent document retrieval)

2. **[Embeddings] Section**:
   - `enable_contextual_chunking = false` (baseline default)
   - `contextual_llm_model = gpt-3.5-turbo` (LLM for contextualization)
   - `contextual_chunk_method = situate_context` (context generation method)

3. **[RAG] Section**: ✅ COMPLETE
   - `enable_parent_expansion = false`
   - `parent_expansion_size = 500`
   - `include_sibling_chunks = false`
   - `semantic_cache_enabled = true`
   - `cache_similarity_threshold = 0.85`
   - `enable_reranking = true`
   - `rerank_top_k = 10`

### Phase 2: API Schema Updates ✅
**Status: COMPLETE**

#### Files Updated:
1. ✅ `/api/v1/schemas/media_request_models.py` - Added contextual options to AddMediaForm
2. ✅ `/api/v1/endpoints/rag_api.py` - Added contextual retrieval to both Simple and Complex APIs
3. ✅ `/core/Embeddings/queue_schemas.py` - Added contextual fields to ChunkingConfig

### Phase 3: Core Implementation Updates ✅
**Status: COMPLETE**

#### Files Updated:
1. ✅ `/core/Embeddings/ChromaDB_Library.py` - Reads config defaults and accepts API overrides
2. ✅ `/api/v1/endpoints/media.py` - Passes contextual options through chunking pipeline
3. ✅ `/api/v1/endpoints/rag_api.py` - Handles contextual retrieval in both Simple and Complex APIs

### Phase 4: Integration Points ✅
**Status: COMPLETE**

#### Files Updated:
1. ✅ `/core/Embeddings/queue_schemas.py` - Added contextualize, contextual_llm_model, context_window_size fields
2. ⚠️ `/core/Embeddings/workers/` - Workers should already respect contextualize flag (verify in testing)

### Phase 5: Testing & Documentation ⏳
**Status: PENDING**

#### Tasks:
1. ⏳ Write unit tests for contextual chunking
2. ⏳ Write integration tests for RAG with contextual retrieval
3. ⏳ Update API documentation
4. ⏳ Add usage examples

## Technical Details

### How Contextual Retrieval Works
1. **During Ingestion/Chunking**:
   - When `enable_contextual_chunking=true`, each chunk is processed with an LLM
   - The LLM generates a contextual summary using the full document and chunk
   - The chunk + context is embedded together for better semantic search

2. **During RAG Retrieval**:
   - When `enable_parent_expansion=true`, retrieved chunks are expanded
   - Parent document context is fetched (configurable size)
   - Sibling chunks can be included for continuity

### Configuration Hierarchy
1. **Baseline**: config.txt settings (defaults)
2. **API Override**: Request parameters override baseline
3. **Fallback**: If not specified in request, use baseline

### Performance Considerations
- **Cost**: Contextual chunking uses LLM calls (increases cost)
- **Speed**: Additional processing time for contextualization
- **Accuracy**: Generally improves retrieval accuracy
- **Storage**: Slightly increased metadata storage

## Current Issues/Risks
1. **LLM Dependency**: Requires working LLM API for contextualization
2. **Cost Impact**: Each chunk requires an LLM call when contextualizing
3. **Backward Compatibility**: Must ensure existing functionality unaffected

## Next Steps
1. ✅ Complete config.txt updates
2. ⏳ Add [RAG] section to config.txt
3. ⏳ Update API schemas for user-facing options
4. ⏳ Implement config reading in ChromaDB_Library
5. ⏳ Wire up media endpoint
6. ⏳ Update RAG pipeline
7. ⏳ Write tests
8. ⏳ Update documentation

## Files Modified
- ✅ `/Config_Files/config.txt` - Added [Chunking], [Embeddings], and [RAG] contextual settings
- ✅ `/api/v1/schemas/media_request_models.py` - Added contextual chunking fields to ChunkingOptions
- ✅ `/api/v1/endpoints/rag_api.py` - Added contextual retrieval to SimpleSearchRequest and ComplexSearchRequest
- ✅ `/core/Embeddings/ChromaDB_Library.py` - Modified to read config defaults and accept API overrides
- ✅ `/api/v1/endpoints/media.py` - Updated _prepare_chunking_options_dict to include contextual options
- ✅ `/core/Embeddings/queue_schemas.py` - Enhanced ChunkingConfig with contextual fields

## API Usage Examples

### Media Ingestion with Contextual Chunking
```python
POST /api/v1/media/add
{
  "media_type": "document",
  "urls": ["https://example.com/document.pdf"],
  "perform_chunking": true,
  "chunk_size": 500,
  "chunk_overlap": 100,
  "enable_contextual_chunking": true,  # New option
  "contextual_llm_model": "gpt-3.5-turbo",  # Optional, uses config default if not specified
  "context_window_size": 500  # Optional
}
```

### RAG Search with Contextual Retrieval
```python
POST /api/v1/rag/search/simple
{
  "query": "What is machine learning?",
  "databases": ["media"],
  "enable_contextual_retrieval": true,  # New option
  "parent_expansion_size": 500,  # Optional
  "include_sibling_chunks": true,  # Optional
  "enable_reranking": true,
  "top_k": 10
}
```

## Testing Checklist
- [ ] Config defaults work correctly
- [ ] API parameters override config
- [ ] Contextual chunking can be toggled on/off
- [ ] RAG parent expansion works
- [ ] Performance impact is acceptable
- [ ] Backward compatibility maintained
- [ ] LLM calls are made when contextualization is enabled
- [ ] Parent/sibling chunks are properly expanded in RAG