# __init__.py
# Description: RAG Search package initialization
#
# The new simplified RAG implementation is in the 'simplified' subdirectory
# For backward compatibility, we re-export the main components here

try:
    # Try to import from the simplified implementation
    from .simplified import (
        EmbeddingsService,
        RAGService,
        RAGConfig,
        SearchResult,
        SearchResultWithCitations,
        create_rag_service,
        create_config_for_collection,
        create_config_for_testing
    )
    
    # Import the chunking service wrapper
    from .chunking_service import ChunkingService
    
    # For backward compatibility, create IndexingService as an alias
    IndexingService = RAGService
    
    __all__ = [
        'EmbeddingsService',
        'ChunkingService', 
        'IndexingService',
        'RAGService',
        'RAGConfig',
        'SearchResult',
        'SearchResultWithCitations',
        'create_rag_service',
        'create_config_for_collection',
        'create_config_for_testing'
    ]
    
except ImportError as e:
    # If simplified imports fail, provide stub implementations
    import logging
    logger = logging.getLogger(__name__)
    logger.error(f"Failed to import simplified RAG services: {e}")
    
    # Provide minimal stubs to prevent import errors
    class EmbeddingsService:
        def __init__(self, *args, **kwargs):
            raise ImportError("RAG services not available. Please check dependencies.")
    
    class ChunkingService:
        def __init__(self, *args, **kwargs):
            raise ImportError("RAG services not available. Please check dependencies.")
    
    class IndexingService:
        def __init__(self, *args, **kwargs):
            raise ImportError("RAG services not available. Please check dependencies.")
    
    RAGService = IndexingService
    
    __all__ = [
        'EmbeddingsService',
        'ChunkingService',
        'IndexingService'
    ]