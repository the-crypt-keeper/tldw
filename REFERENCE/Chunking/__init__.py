# __init__.py
"""
Chunking module for flexible text chunking with template support.
"""

from .Chunk_Lib import (
    Chunker,
    ChunkingError,
    InvalidChunkingMethodError,
    InvalidInputError,
    LanguageDetectionError,
    improved_chunking_process,
    chunk_for_embedding,
    process_document_with_metadata,
    DEFAULT_CHUNK_OPTIONS
)

from .chunking_templates import (
    ChunkingTemplate,
    ChunkingTemplateManager,
    ChunkingPipeline,
    ChunkingStage,
    ChunkingOperation
)

from .language_chunkers import (
    LanguageChunkerFactory,
    ChineseChunker,
    JapaneseChunker,
    DefaultChunker
)

from .token_chunker import (
    TokenBasedChunker,
    create_token_chunker
)

__all__ = [
    # Main chunking classes
    "Chunker",
    "improved_chunking_process",
    "chunk_for_embedding",
    "process_document_with_metadata",
    "DEFAULT_CHUNK_OPTIONS",
    
    # Template system
    "ChunkingTemplate",
    "ChunkingTemplateManager", 
    "ChunkingPipeline",
    "ChunkingStage",
    "ChunkingOperation",
    
    # Language support
    "LanguageChunkerFactory",
    "ChineseChunker",
    "JapaneseChunker",
    "DefaultChunker",
    
    # Token support
    "TokenBasedChunker",
    "create_token_chunker",
    
    # Exceptions
    "ChunkingError",
    "InvalidChunkingMethodError",
    "InvalidInputError",
    "LanguageDetectionError"
]