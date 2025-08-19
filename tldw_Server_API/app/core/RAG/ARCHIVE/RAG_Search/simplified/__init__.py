"""
Simplified RAG services with citations support.

This module provides a cleaner, more maintainable implementation of RAG services
while reusing existing robust components like Embeddings_Lib.
"""

from .citations import (
    Citation,
    CitationType,
    SearchResultWithCitations,
    merge_citations,
    group_citations_by_document,
    filter_overlapping_citations
)

from .embeddings_wrapper import (
    EmbeddingsServiceWrapper,
    EmbeddingsServiceWrapper as EmbeddingsService,  # Alias for backward compatibility
    create_embeddings_service
)

from .vector_store import (
    SearchResult,
    VectorStore,
    ChromaVectorStore,
    InMemoryVectorStore,
    create_vector_store
)

from .rag_service import (
    RAGService,
    RAGConfig,
    create_and_index,
    create_rag_service
)

from .data_models import (
    IndexingResult
)

from .simple_cache import (
    SimpleRAGCache,
    get_rag_cache
)

from .config import (
    EmbeddingConfig,
    VectorStoreConfig,
    ChunkingConfig,
    SearchConfig,
    create_config_for_collection,
    create_config_for_testing
)

__all__ = [
    # Citations
    "Citation",
    "CitationType", 
    "SearchResultWithCitations",
    "merge_citations",
    "group_citations_by_document",
    "filter_overlapping_citations",
    
    # Embeddings
    "EmbeddingsServiceWrapper",
    "EmbeddingsService",
    "create_embeddings_service",
    
    # Vector stores
    "SearchResult",
    "VectorStore",
    "ChromaVectorStore", 
    "InMemoryVectorStore",
    "create_vector_store",
    
    # Main RAG service
    "RAGService",
    "RAGConfig",
    "IndexingResult",
    "create_and_index",
    "create_rag_service",
    
    # Cache
    "SimpleRAGCache",
    "get_rag_cache",
    
    # Configuration
    "EmbeddingConfig",
    "VectorStoreConfig",
    "ChunkingConfig",
    "SearchConfig",
    "create_config_for_collection",
    "create_config_for_testing",
]

__version__ = "0.1.0"