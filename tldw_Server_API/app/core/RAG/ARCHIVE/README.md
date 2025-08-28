# Archived RAG Implementations

This folder contains deprecated RAG implementations that have been consolidated into the main `/app/core/RAG/rag_service/` implementation.

## Archived Contents

### RAG_Search/
The original multi-implementation RAG module containing:
- **simplified/** - Three different RAG service implementations (RAGService, EnhancedRAGService, EnhancedRAGServiceV2)
- **pipeline_*.py** - Functional programming pipeline approach
- Various chunking, reranking, and optimization modules

### Documentation Files
- **RAG-REPORT2.md** - Original re-architecture report for single-user TUI
- **RAG_EMBEDDINGS_INTEGRATION_GUIDE.md** - Old embeddings integration guide

### Utility Files
- **import_fixer.py** - Script used during migration from tldw_chatbook to tldw_Server_API
- **rag_embeddings_integration.py** - Old embeddings integration (needs updating if restored)

## Status

**⚠️ DEPRECATED - DO NOT USE**

All functionality from these implementations has been consolidated into the main RAG service at `/app/core/RAG/rag_service/` with the following improvements:

1. **Unified Architecture** - Single implementation instead of 4
2. **Feature Complete** - All features available through configuration
3. **Better Testing** - Comprehensive test coverage
4. **Improved Performance** - Connection pooling and optimizations

## Migration

If you need to reference old code:
1. Look here for implementation details
2. Use the new consolidated service for all new work
3. Features are now available via configuration flags

## Files Still Referencing Archived Code

The following files may need updating as they reference the archived implementations:
- `/app/core/Evaluations/rag_evaluator.py` 
- `/tests/RAG/test_rag_embeddings_integration.py`

These should be updated to use the new consolidated implementation at `/app/core/RAG/rag_service/`

---
*Archived: 2025-08-18*