# __init__.py
"""
Chunking module for text processing and segmentation.
Provides various strategies for splitting text into manageable chunks.
"""

from .base import (
    ChunkingMethod,
    ChunkMetadata,
    ChunkResult,
    ChunkerConfig,
    BaseChunkingStrategy,
)

from .exceptions import (
    ChunkingError,
    InvalidInputError,
    InvalidChunkingMethodError,
    TokenizerError,
    TemplateError,
    LanguageNotSupportedError,
    ChunkSizeError,
    ProcessingError,
    ConfigurationError,
    CacheError,
)

from .chunker import (
    Chunker,
    create_chunker,
)

# Default chunking options for backward compatibility
DEFAULT_CHUNK_OPTIONS = {
    'method': 'words',
    'max_size': 400,
    'overlap': 200,
    'language': 'en',
    'adaptive': False,
    'multi_level': False,
    'semantic_similarity_threshold': 0.7,
    'semantic_overlap_sentences': 2,
    'json_chunkable_data_key': 'data',
    'summarization_detail': 0.5,
    'tokenizer_name_or_path': 'gpt2'
}

# For backward compatibility with existing code
# These will be implemented as we port more functionality
def improved_chunking_process(text: str, 
                             chunk_options: dict = None,
                             tokenizer_name_or_path: str = None,
                             llm_call_func = None,
                             llm_api_config: dict = None) -> list:
    """
    Backward compatibility function for improved chunking process.
    
    Args:
        text: Text to chunk
        chunk_options: Dictionary of chunking options
        tokenizer_name_or_path: Optional tokenizer (not used in new API)
        llm_call_func: Optional LLM function for methods like rolling_summarize
        llm_api_config: Optional LLM configuration
        
    Returns:
        List of chunk dictionaries with text and metadata
    """
    options = chunk_options or {}
    
    # Extract options
    method = options.get('method', 'words')
    max_size = options.get('max_size', 400)
    overlap = options.get('overlap', 200)
    language = options.get('language', 'en')
    
    # Create chunker with LLM support if provided
    chunker = Chunker(llm_call_func=llm_call_func, llm_config=llm_api_config)
    # Remove duplicates from options
    filtered_options = {k: v for k, v in options.items() 
                       if k not in ['method', 'max_size', 'overlap', 'language']}
    chunks = chunker.chunk_text_with_metadata(
        text=text,
        method=method,
        max_size=max_size,
        overlap=overlap,
        language=language,
        **filtered_options
    )
    
    # Convert to expected format
    result = []
    for chunk in chunks:
        result.append({
            'text': chunk.text,
            'metadata': {
                'index': chunk.metadata.index,
                'start_index': chunk.metadata.start_char,
                'end_index': chunk.metadata.end_char,
                'word_count': chunk.metadata.word_count,
                'language': chunk.metadata.language,
            }
        })
    
    return result


def chunk_for_embedding(text: str, file_name: str, **kwargs) -> list:
    """
    Backward compatibility function for chunking for embeddings.
    
    Args:
        text: Text to chunk
        file_name: Name of the file being processed
        **kwargs: Additional options
        
    Returns:
        List of chunk dictionaries suitable for embedding
    """
    # Create chunker with embedding-optimized settings
    chunker = Chunker()
    
    # Use semantic chunking if available, otherwise sentences
    method = kwargs.get('method', 'sentences')
    max_size = kwargs.get('max_size', 512)  # Good size for embeddings
    overlap = kwargs.get('overlap', 50)
    
    # Remove duplicates from kwargs
    filtered_kwargs = {k: v for k, v in kwargs.items() 
                      if k not in ['method', 'max_size', 'overlap']}
    
    chunks = chunker.chunk_text_with_metadata(
        text=text,
        method=method,
        max_size=max_size,
        overlap=overlap,
        **filtered_kwargs
    )
    
    # Format for embedding
    result = []
    for chunk in chunks:
        result.append({
            'text': chunk.text,
            'text_for_embedding': f"File: {file_name}\n{chunk.text}",
            'metadata': {
                'file_name': file_name,
                'chunk_index': chunk.metadata.index,
                'start_char': chunk.metadata.start_char,
                'end_char': chunk.metadata.end_char,
            }
        })
    
    return result


# Enhanced chunk support for RAG integration
from dataclasses import dataclass, field
from typing import Dict, Any, Optional
from enum import Enum


class ChunkType(Enum):
    """Enumeration of chunk types for structure-aware chunking."""
    TEXT = "text"
    PARAGRAPH = "paragraph"
    CODE = "code"
    TABLE = "table"
    HEADER = "header"
    LIST = "list"
    QUOTE = "quote"
    METADATA = "metadata"


@dataclass
class EnhancedChunk:
    """Enhanced chunk with type and position tracking for RAG."""
    id: str
    content: str
    chunk_type: ChunkType
    start_char: int  # Position in original document
    end_char: int    # Position in original document
    chunk_index: int
    metadata: Dict[str, Any] = field(default_factory=dict)
    parent_id: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "id": self.id,
            "content": self.content,
            "chunk_type": self.chunk_type.value if isinstance(self.chunk_type, Enum) else self.chunk_type,
            "start_char": self.start_char,
            "end_char": self.end_char,
            "chunk_index": self.chunk_index,
            "metadata": self.metadata,
            "parent_id": self.parent_id
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'EnhancedChunk':
        """Create from dictionary."""
        chunk_type = data.get("chunk_type", "text")
        if isinstance(chunk_type, str):
            try:
                chunk_type = ChunkType(chunk_type)
            except ValueError:
                chunk_type = ChunkType.TEXT
        
        return cls(
            id=data["id"],
            content=data["content"],
            chunk_type=chunk_type,
            start_char=data.get("start_char", 0),
            end_char=data.get("end_char", len(data["content"])),
            chunk_index=data.get("chunk_index", 0),
            metadata=data.get("metadata", {}),
            parent_id=data.get("parent_id")
        )


__all__ = [
    # Main classes
    'Chunker',
    'create_chunker',
    
    # Configuration
    'ChunkerConfig',
    'ChunkingMethod',
    
    # Results
    'ChunkResult',
    'ChunkMetadata',
    
    # Exceptions
    'ChunkingError',
    'InvalidInputError',
    'InvalidChunkingMethodError',
    'TokenizerError',
    'TemplateError',
    'LanguageNotSupportedError',
    'ChunkSizeError',
    'ProcessingError',
    'ConfigurationError',
    'CacheError',
    
    # Backward compatibility
    'improved_chunking_process',
    'chunk_for_embedding',
    'EnhancedChunk',
    'ChunkType',
    
    # Base classes for extensions
    'BaseChunkingStrategy',
]

__version__ = '2.0.0'