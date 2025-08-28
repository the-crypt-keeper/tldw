# base.py
"""
Base classes and protocols for the chunking system.
Provides abstract interfaces and common functionality for all chunking strategies.
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Protocol, Union, Generator
from dataclasses import dataclass, field
from enum import Enum
from loguru import logger


class ChunkingMethod(Enum):
    """Enumeration of available chunking methods."""
    WORDS = "words"
    SENTENCES = "sentences"
    PARAGRAPHS = "paragraphs"
    TOKENS = "tokens"
    SEMANTIC = "semantic"
    JSON = "json"
    XML = "xml"
    EBOOK_CHAPTERS = "ebook_chapters"
    ROLLING_SUMMARIZE = "rolling_summarize"
    FIXED_SIZE = "fixed_size"


@dataclass
class ChunkMetadata:
    """Metadata for a text chunk."""
    index: int
    start_char: int
    end_char: int
    word_count: int
    token_count: Optional[int] = None
    section: Optional[str] = None
    language: Optional[str] = None
    overlap_with_previous: int = 0
    overlap_with_next: int = 0
    char_count: Optional[int] = None  # Add for compatibility
    sentence_count: Optional[int] = None  # Add for compatibility
    method: Optional[str] = None  # Add chunking method used
    options: Optional[Dict[str, Any]] = field(default=None)  # Add options used
    
    def __post_init__(self):
        """Calculate derived fields if not provided."""
        if self.char_count is None and self.end_char is not None and self.start_char is not None:
            self.char_count = self.end_char - self.start_char
    

@dataclass
class ChunkResult:
    """Result of a chunking operation."""
    text: str
    metadata: ChunkMetadata
    

class ChunkingStrategy(Protocol):
    """Protocol for chunking strategies."""
    
    def chunk(self, 
              text: str, 
              max_size: int,
              overlap: int = 0,
              **options) -> List[str]:
        """
        Chunk text according to the strategy.
        
        Args:
            text: Text to chunk
            max_size: Maximum size of each chunk
            overlap: Overlap between chunks
            **options: Strategy-specific options
            
        Returns:
            List of text chunks
        """
        ...
    
    def chunk_with_metadata(self,
                           text: str,
                           max_size: int,
                           overlap: int = 0,
                           **options) -> List[ChunkResult]:
        """
        Chunk text and return with metadata.
        
        Args:
            text: Text to chunk
            max_size: Maximum size of each chunk
            overlap: Overlap between chunks
            **options: Strategy-specific options
            
        Returns:
            List of ChunkResult objects
        """
        ...


class BaseChunkingStrategy(ABC):
    """Base class for chunking strategies."""
    
    def __init__(self, language: str = 'en'):
        """
        Initialize the chunking strategy.
        
        Args:
            language: Language code for text processing
        """
        self.language = language
        self._cache = {}
        logger.debug(f"Initialized {self.__class__.__name__} for language: {language}")
    
    @abstractmethod
    def chunk(self, 
              text: str, 
              max_size: int,
              overlap: int = 0,
              **options) -> List[str]:
        """
        Chunk text according to the strategy.
        
        Args:
            text: Text to chunk
            max_size: Maximum size of each chunk
            overlap: Overlap between chunks
            **options: Strategy-specific options
            
        Returns:
            List of text chunks
        """
        pass
    
    def chunk_with_metadata(self,
                           text: str,
                           max_size: int,
                           overlap: int = 0,
                           **options) -> List[ChunkResult]:
        """
        Chunk text and return with metadata.
        
        Args:
            text: Text to chunk
            max_size: Maximum size of each chunk  
            overlap: Overlap between chunks
            **options: Strategy-specific options
            
        Returns:
            List of ChunkResult objects
        """
        chunks = self.chunk(text, max_size, overlap, **options)
        results = []
        current_pos = 0
        
        for i, chunk in enumerate(chunks):
            # Find chunk position in original text
            chunk_start = text.find(chunk, current_pos)
            if chunk_start == -1:
                chunk_start = current_pos
            chunk_end = chunk_start + len(chunk)
            
            metadata = ChunkMetadata(
                index=i,
                start_char=chunk_start,
                end_char=chunk_end,
                word_count=len(chunk.split()),
                language=self.language,
                overlap_with_previous=overlap if i > 0 else 0,
                overlap_with_next=overlap if i < len(chunks) - 1 else 0
            )
            
            results.append(ChunkResult(text=chunk, metadata=metadata))
            current_pos = chunk_end - overlap if overlap > 0 else chunk_end
        
        return results
    
    def validate_parameters(self, text: str, max_size: int, overlap: int):
        """
        Validate chunking parameters.
        
        Args:
            text: Text to chunk
            max_size: Maximum size of each chunk
            overlap: Overlap between chunks
            
        Raises:
            ValueError: If parameters are invalid
        """
        if not isinstance(text, str):
            raise ValueError(f"Text must be a string, got {type(text).__name__}")
        
        if not text.strip():
            logger.debug("Empty text provided for chunking")
            return False
            
        if max_size <= 0:
            raise ValueError(f"max_size must be positive, got {max_size}")
            
        if overlap < 0:
            raise ValueError(f"overlap cannot be negative, got {overlap}")
            
        if overlap >= max_size:
            logger.warning(f"Overlap ({overlap}) >= max_size ({max_size}), adjusting to max_size - 1")
            overlap = max_size - 1
            
        return True
    
    def chunk_generator(self,
                       text: str,
                       max_size: int,
                       overlap: int = 0,
                       **options) -> Generator[str, None, None]:
        """
        Generator version of chunk for memory efficiency.
        
        Args:
            text: Text to chunk
            max_size: Maximum size of each chunk
            overlap: Overlap between chunks
            **options: Strategy-specific options
            
        Yields:
            Individual text chunks
        """
        chunks = self.chunk(text, max_size, overlap, **options)
        for chunk in chunks:
            yield chunk


class ChunkerConfig:
    """Configuration for the chunking system."""
    
    def __init__(self,
                 default_method: ChunkingMethod = ChunkingMethod.WORDS,
                 default_max_size: int = 400,
                 default_overlap: int = 200,
                 language: str = 'en',
                 enable_cache: bool = True,
                 cache_size: int = 100,
                 max_text_size: int = 100_000_000,  # 100MB
                 enable_metrics: bool = True):
        """
        Initialize chunker configuration.
        
        Args:
            default_method: Default chunking method
            default_max_size: Default maximum chunk size
            default_overlap: Default overlap between chunks
            language: Default language for text processing
            enable_cache: Whether to enable caching
            cache_size: Maximum cache size
            max_text_size: Maximum text size to process
            enable_metrics: Whether to collect metrics
        """
        # Convert string to enum if needed
        if isinstance(default_method, str):
            try:
                default_method = ChunkingMethod(default_method)
            except ValueError:
                logger.warning(f"Unknown chunking method '{default_method}', using default")
                default_method = ChunkingMethod.WORDS
        
        self.default_method = default_method
        self.default_max_size = default_max_size
        self.default_overlap = default_overlap
        self.language = language
        self.enable_cache = enable_cache
        self.cache_size = cache_size
        self.max_text_size = max_text_size
        self.enable_metrics = enable_metrics
        
        logger.info(f"ChunkerConfig initialized with method={self.default_method.value if hasattr(self.default_method, 'value') else self.default_method}, "
                   f"max_size={default_max_size}, overlap={default_overlap}")


# Custom exceptions
class ChunkingError(Exception):
    """Base exception for chunking errors."""
    pass


class InvalidInputError(ChunkingError):
    """Exception raised for invalid input."""
    pass


class InvalidChunkingMethodError(ChunkingError):
    """Exception raised for invalid chunking method."""
    pass


class TokenizerError(ChunkingError):
    """Exception raised for tokenizer-related errors."""
    pass