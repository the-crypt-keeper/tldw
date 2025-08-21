# Chunking Module Refactoring Plan - v33

## Executive Summary
Comprehensive refactoring plan to modernize the chunking module, split the monolithic 2000+ line file, implement advanced features from the reference implementation, and improve performance, maintainability, and scalability.

## Current State Analysis

### Issues Identified
1. **Monolithic Structure**: Single 2000+ line file (Chunk_Lib.py)
2. **Mixed Responsibilities**: Chunking, validation, preprocessing all in one class
3. **Missing Features**: No template system, limited language support, no metrics
4. **Performance Issues**: No async support, limited caching, memory inefficient for large files
5. **Security Concerns**: Already addressed (DoS prevention, input validation)

### Reference Implementation Features Missing
1. **Template System**
   - JSON-based templates with inheritance
   - Multi-stage pipelines (preprocess → chunk → postprocess)
   - Pre-built templates for different document types
   - User-defined template storage

2. **Language-Specific Support**
   - Chinese tokenization (jieba)
   - Japanese tokenization (fugashi)
   - Proper non-Latin script handling

3. **Advanced Operations**
   - Section detection
   - Code block detection
   - Metadata extraction
   - Context addition to chunks
   - Smart chunk merging

4. **Database Integration**
   - Template storage
   - Per-document configurations
   - Caching and versioning

## Proposed Architecture

### Module Structure
```
tldw_Server_API/app/core/Chunking/
├── __init__.py                 # Public API exports
├── base.py                     # Base classes and protocols ✅
├── chunker.py                  # Main Chunker class (simplified)
├── exceptions.py               # Custom exceptions
├── strategies/
│   ├── __init__.py
│   ├── words.py               # Word-based chunking
│   ├── sentences.py           # Sentence-based chunking
│   ├── paragraphs.py          # Paragraph-based chunking
│   ├── tokens.py              # Token-based chunking
│   ├── semantic.py            # Semantic chunking
│   ├── json_chunker.py        # JSON chunking
│   ├── xml_chunker.py         # XML chunking
│   ├── ebook.py               # Ebook chapter chunking
│   └── rolling_summarize.py   # Rolling summarization
├── language/
│   ├── __init__.py
│   ├── chinese.py             # Chinese-specific (jieba)
│   ├── japanese.py            # Japanese-specific (fugashi)
│   ├── multilingual.py        # Multi-language support
│   └── default.py             # Default language handler
├── templates/
│   ├── __init__.py
│   ├── manager.py             # Template management
│   ├── pipeline.py            # Pipeline execution
│   ├── operations.py          # Template operations
│   └── builtin/               # Pre-built templates
│       ├── academic.json
│       ├── legal.json
│       ├── code.json
│       └── conversation.json
├── utils/
│   ├── __init__.py
│   ├── validation.py          # Input validation
│   ├── cache.py               # Caching utilities
│   ├── metrics.py             # Performance metrics
│   └── text_processing.py     # Text utilities
└── async_chunker.py           # Async wrapper

```

## Implementation Phases

### Phase 1: Core Refactoring (Week 1)
**Goal**: Split monolithic file into modular structure

1. **Extract Base Components** ✅
   - [x] Create base.py with protocols and base classes
   - [ ] Move exceptions to exceptions.py
   - [ ] Create chunker.py with simplified main class

2. **Extract Strategies**
   - [ ] Create strategies/ directory
   - [ ] Extract word chunking → strategies/words.py
   - [ ] Extract sentence chunking → strategies/sentences.py
   - [ ] Extract token chunking → strategies/tokens.py
   - [ ] Extract semantic chunking → strategies/semantic.py
   - [ ] Extract JSON/XML chunking → respective files

3. **Implement Strategy Pattern**
   ```python
   class Chunker:
       def __init__(self, config: ChunkerConfig):
           self.strategies = {
               'words': WordChunkingStrategy(),
               'sentences': SentenceChunkingStrategy(),
               # ... etc
           }
       
       def chunk_text(self, text: str, method: str, **options):
           strategy = self.strategies[method]
           return strategy.chunk(text, **options)
   ```

### Phase 2: Template System (Week 2)
**Goal**: Implement template-based chunking pipelines

1. **Template Manager**
   ```python
   class ChunkingTemplateManager:
       def load_template(self, name: str) -> ChunkingTemplate
       def save_template(self, template: ChunkingTemplate)
       def apply_template(self, text: str, template: str)
   ```

2. **Pipeline Executor**
   ```python
   class ChunkingPipeline:
       def execute(self, text: str, template: ChunkingTemplate)
       def preprocess(self, text: str, operations: List)
       def chunk(self, text: str, method: str, options: Dict)
       def postprocess(self, chunks: List, operations: List)
   ```

3. **Pre-built Templates**
   - Academic papers
   - Legal documents
   - Code documentation
   - Conversations/dialogues
   - General text

### Phase 3: Advanced Features (Week 3)
**Goal**: Add language support, metrics, and async processing

1. **Language-Specific Chunkers**
   ```python
   class ChineseChunker:
       def __init__(self):
           self.jieba = safe_import('jieba')
       
       def tokenize(self, text: str) -> List[str]:
           if self.jieba:
               return list(self.jieba.cut(text))
           return fallback_tokenize(text)
   ```

2. **Metrics Collection**
   ```python
   from prometheus_client import Counter, Histogram
   
   chunks_created = Counter('chunking_chunks_created_total')
   chunking_duration = Histogram('chunking_duration_seconds')
   cache_hits = Counter('chunking_cache_hits_total')
   ```

3. **Async Processing**
   ```python
   class AsyncChunker:
       async def chunk_text_async(self, text: str, method: str)
       async def chunk_multiple_async(self, texts: List[str])
       async def chunk_stream_async(self, stream: AsyncIterator[str])
   ```

### Phase 4: Integration & Optimization (Week 4)
**Goal**: Integrate with existing system, optimize performance

1. **RAG Integration**
   - Update RAG module to use new chunking API
   - Migrate existing chunking calls
   - Ensure backward compatibility

2. **Database Integration**
   - Store templates in SQLite
   - Cache chunking results
   - Per-document configurations

3. **Rate Limiting**
   ```python
   @router.post("/chunk")
   @limiter.limit("100/minute")
   async def chunk_text(request: ChunkRequest):
       # Rate-limited endpoint
   ```

4. **Performance Optimization**
   - Implement chunk streaming
   - Add result caching
   - Optimize memory usage
   - Parallel processing for multiple documents

## Migration Strategy

### Backward Compatibility
1. Keep old Chunk_Lib.py temporarily as deprecated
2. Create compatibility wrapper:
   ```python
   # Chunk_Lib.py (deprecated)
   from .chunker import Chunker as NewChunker
   
   class Chunker:
       """Deprecated: Use chunker.Chunker instead"""
       def __init__(self, *args, **kwargs):
           logger.warning("Using deprecated Chunker")
           self._chunker = NewChunker(*args, **kwargs)
   ```

### Migration Steps
1. **Phase 1**: Create new structure alongside old
2. **Phase 2**: Update internal uses to new API
3. **Phase 3**: Update external API endpoints
4. **Phase 4**: Deprecate old module
5. **Phase 5**: Remove old module (after 2 releases)

## Testing Strategy

### Unit Tests
- Test each strategy independently
- Test template loading/saving
- Test language-specific tokenizers
- Test cache behavior
- Test metrics collection

### Integration Tests
- Test full pipeline execution
- Test RAG integration
- Test API endpoints
- Test database operations
- Test async operations

### Performance Tests
- Benchmark chunking speed
- Memory usage profiling
- Cache hit rates
- Concurrent request handling

## Risk Assessment

### Technical Risks
1. **Breaking Changes**: Mitigated by compatibility wrapper
2. **Performance Regression**: Mitigated by benchmarking
3. **Data Loss**: Templates backed up before migration
4. **Integration Issues**: Phased rollout with testing

### Mitigation Strategies
1. Feature flags for gradual rollout
2. Comprehensive test coverage
3. Performance monitoring
4. Rollback plan for each phase

## Success Metrics

### Performance
- [ ] 50% reduction in memory usage for large files
- [ ] 30% improvement in chunking speed
- [ ] <100ms latency for template application
- [ ] Support for 100+ concurrent requests

### Quality
- [ ] 95% test coverage
- [ ] Zero critical bugs in production
- [ ] All existing features maintained
- [ ] New features fully documented

### Adoption
- [ ] All internal modules migrated
- [ ] 10+ pre-built templates available
- [ ] API documentation complete
- [ ] Migration guide published

## Dependencies

### Required Libraries
- transformers (optional, for token chunking)
- jieba (optional, for Chinese)
- fugashi (optional, for Japanese)
- prometheus-client (for metrics)
- slowapi (for rate limiting)

### Internal Dependencies
- MediaDatabase (for template storage)
- RAG module (needs updating)
- API endpoints (needs updating)

## Timeline

### Week 1 (Days 1-7)
- Day 1-2: Create module structure, extract base components
- Day 3-4: Extract chunking strategies
- Day 5-6: Implement strategy pattern
- Day 7: Testing and documentation

### Week 2 (Days 8-14)
- Day 8-9: Implement template manager
- Day 10-11: Create pipeline executor
- Day 12-13: Build pre-built templates
- Day 14: Testing and integration

### Week 3 (Days 15-21)
- Day 15-16: Add language-specific chunkers
- Day 17-18: Implement metrics collection
- Day 19-20: Add async processing
- Day 21: Performance testing

### Week 4 (Days 22-28)
- Day 22-23: RAG integration
- Day 24-25: Database integration
- Day 26: Rate limiting
- Day 27-28: Final testing and documentation

## RAG Module Integration

### Current RAG Features to Preserve
After reviewing the RAG module, critical features to integrate:

1. **Table Processing** (`table_serializer.py`)
   - Format detection (Markdown, CSV, TSV, JSON, HTML)
   - Multiple serialization strategies
   - Structure preservation

2. **Document Structure** 
   - Markdown headers preservation
   - Code block detection and preservation
   - List and nested structure handling
   - Metadata tracking for structure

3. **Integration Points**
   - `improved_chunking_process()` function
   - `EnhancedChunk` and `ChunkType` classes
   - `ChunkingError` and `InvalidInputError` exceptions

### Migration Strategy (No Backward Compatibility)
- Direct update of all RAG imports
- Integration of table/structure features into new chunking
- Complete removal of old Chunk_Lib.py

## Multi-lingual Chunking Insights

### From Chunklet Project Analysis
**Three-Tier Language Support:**
1. **Primary** (pysbd): Arabic, Chinese, Japanese, French, etc.
2. **Secondary** (custom): Portuguese, Norwegian, Czech, etc.  
3. **Fallback** (regex): Universal support

**Key Features:**
- Automatic language detection
- Clause-level overlap preservation
- Context-aware chunking per language

## Dependencies

### Required Libraries
- **Core**: transformers, prometheus-client, slowapi
- **Language**: pysbd, jieba, fugashi, langdetect
- **Table Processing**: pandas (optional), beautifulsoup4 (optional)

### Internal Dependencies
- MediaDatabase (template storage)
- RAG module (update to new API)
- API endpoints (update imports)
- Embeddings module (potential updates)

## Risk Analysis & Mitigation

### Identified Risks
1. **RAG Integration Complexity**
   - Risk: Breaking existing RAG functionality
   - Mitigation: Comprehensive integration tests

2. **Performance Regression**
   - Risk: New modular structure slower
   - Mitigation: Benchmark before/after

3. **Table Processing Loss**
   - Risk: Losing sophisticated table handling
   - Mitigation: Port complete table_serializer

4. **Multi-lingual Support**
   - Risk: Incorrect language detection
   - Mitigation: Fallback handlers, extensive testing

## Next Steps

1. **Immediate Actions**
   - [x] Create module structure
   - [x] Write base.py
   - [x] Review RAG module for integration points
   - [x] Analyze table/structure preservation needs
   - [x] Review multi-lingual chunking approaches
   - [ ] Begin extracting strategies with table support

2. **Implementation Priority**
   - [ ] Extract and enhance word/sentence strategies
   - [ ] Port table_serializer from RAG
   - [ ] Implement multi-lingual support
   - [ ] Update RAG to use new API

3. **Documentation**
   - [ ] Migration guide for RAG module
   - [ ] Table processing documentation
   - [ ] Multi-lingual support guide

## Appendix

### A. Code Examples
[Included in plan above]

### B. API Changes
- Old: `chunker.chunk_text(text, method='words')`
- New: `chunker.chunk_text(text, method='words', template='academic', preserve_tables=True)`

### C. Configuration Changes
- New config file: `chunking_config.yaml`
- Template directory: `templates/`
- Cache directory: `cache/chunking/`
- Language models: `language_models/`

---
*Document Version: 33*
*Last Updated: 2025-08-20*
*Status: In Progress - RAG Review Complete*