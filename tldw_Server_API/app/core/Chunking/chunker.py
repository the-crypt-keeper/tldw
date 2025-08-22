# chunker.py
"""
Main Chunker class that provides a unified interface for all chunking strategies.
This is the primary entry point for the chunking module.
"""

from typing import List, Dict, Any, Optional, Union, Generator
from pathlib import Path
from loguru import logger

from .base import ChunkerConfig, ChunkingMethod, ChunkResult
from .exceptions import (
    InvalidChunkingMethodError,
    InvalidInputError,
    ChunkingError,
    ConfigurationError
)
from .strategies.words import WordChunkingStrategy
from .strategies.sentences import SentenceChunkingStrategy
from .strategies.tokens import TokenChunkingStrategy
from .strategies.structure_aware import StructureAwareChunkingStrategy
from .strategies.rolling_summarize import RollingSummarizeStrategy


class Chunker:
    """
    Main chunker class that manages different chunking strategies.
    Provides a unified interface for text chunking with support for
    multiple methods, languages, and configurations.
    """
    
    def __init__(self, config: Optional[ChunkerConfig] = None, 
                 llm_call_func: Optional[Any] = None,
                 llm_config: Optional[Dict[str, Any]] = None):
        """
        Initialize the chunker with configuration.
        
        Args:
            config: Chunker configuration (uses defaults if not provided)
            llm_call_func: Optional LLM function for strategies that need it
            llm_config: Optional LLM configuration
        """
        self.config = config or ChunkerConfig()
        self.llm_call_func = llm_call_func
        self.llm_config = llm_config or {}
        
        # Initialize strategy instances
        self._strategies = {}
        self._initialize_strategies()
        
        # Cache for processed results
        self._cache = {} if self.config.enable_cache else None
        
        logger.info(f"Chunker initialized with default method: {self.config.default_method.value}")
    
    def _initialize_strategies(self):
        """Initialize available chunking strategies."""
        # Import strategies here to avoid circular imports
        from .strategies.semantic import SemanticChunkingStrategy
        from .strategies.json_xml import JSONChunkingStrategy, XMLChunkingStrategy
        from .strategies.paragraphs import ParagraphChunkingStrategy
        from .strategies.ebook_chapters import EbookChapterChunkingStrategy
        
        # Core strategies
        self._strategies[ChunkingMethod.WORDS.value] = WordChunkingStrategy(
            language=self.config.language
        )
        self._strategies[ChunkingMethod.SENTENCES.value] = SentenceChunkingStrategy(
            language=self.config.language
        )
        self._strategies[ChunkingMethod.PARAGRAPHS.value] = ParagraphChunkingStrategy(
            language=self.config.language
        )
        self._strategies[ChunkingMethod.TOKENS.value] = TokenChunkingStrategy(
            language=self.config.language
        )
        self._strategies['structure_aware'] = StructureAwareChunkingStrategy(
            language=self.config.language
        )
        self._strategies[ChunkingMethod.SEMANTIC.value] = SemanticChunkingStrategy(
            language=self.config.language
        )
        self._strategies[ChunkingMethod.JSON.value] = JSONChunkingStrategy(
            language=self.config.language
        )
        self._strategies[ChunkingMethod.XML.value] = XMLChunkingStrategy(
            language=self.config.language
        )
        self._strategies[ChunkingMethod.EBOOK_CHAPTERS.value] = EbookChapterChunkingStrategy(
            language=self.config.language
        )
        self._strategies[ChunkingMethod.ROLLING_SUMMARIZE.value] = RollingSummarizeStrategy(
            language=self.config.language,
            llm_call_func=self.llm_call_func,
            llm_config=self.llm_config
        )
        
        logger.debug(f"Initialized {len(self._strategies)} chunking strategies")
    
    def chunk_text(self,
                   text: str,
                   method: Optional[str] = None,
                   max_size: Optional[int] = None,
                   overlap: Optional[int] = None,
                   language: Optional[str] = None,
                   **options) -> List[str]:
        """
        Chunk text using the specified method.
        
        Args:
            text: Text to chunk
            method: Chunking method (uses default if not specified)
            max_size: Maximum chunk size (uses default if not specified)
            overlap: Overlap between chunks (uses default if not specified)
            language: Language of the text (uses default if not specified)
            **options: Additional method-specific options
            
        Returns:
            List of text chunks
            
        Raises:
            InvalidInputError: If input validation fails
            InvalidChunkingMethodError: If method is not supported
            ChunkingError: For other chunking errors
        """
        # Validate input
        if not isinstance(text, str):
            raise InvalidInputError(f"Expected string input, got {type(text).__name__}")
        
        if not text.strip():
            logger.debug("Empty text provided, returning empty list")
            return []
        
        if len(text) > self.config.max_text_size:
            raise InvalidInputError(
                f"Text size ({len(text)} bytes) exceeds maximum allowed size "
                f"({self.config.max_text_size} bytes)"
            )
        
        # Use defaults if not specified
        method = method or self.config.default_method.value
        max_size = max_size if max_size is not None else self.config.default_max_size
        overlap = overlap if overlap is not None else self.config.default_overlap
        language = language or self.config.language
        
        # Check cache if enabled
        if self._cache is not None:
            cache_key = self._get_cache_key(text, method, max_size, overlap, language, options)
            if cache_key in self._cache:
                logger.debug("Returning cached result")
                return self._cache[cache_key]
        
        # Get strategy
        if method not in self._strategies:
            available = ', '.join(self._strategies.keys())
            raise InvalidChunkingMethodError(
                f"Unknown chunking method: {method}. Available methods: {available}"
            )
        
        strategy = self._strategies[method]
        
        # Update strategy language if different
        if language != strategy.language:
            strategy.language = language
        
        try:
            # Perform chunking
            logger.debug(f"Chunking with method={method}, max_size={max_size}, "
                        f"overlap={overlap}, language={language}")
            
            chunks = strategy.chunk(text, max_size, overlap, **options)
            
            # Cache result if enabled
            if self._cache is not None and len(chunks) > 0:
                self._cache[cache_key] = chunks
                # Limit cache size
                if len(self._cache) > self.config.cache_size:
                    # Remove oldest entry (simple FIFO)
                    self._cache.pop(next(iter(self._cache)))
            
            logger.info(f"Created {len(chunks)} chunks using {method} method")
            return chunks
            
        except Exception as e:
            logger.error(f"Chunking failed: {e}")
            if isinstance(e, ChunkingError):
                raise
            raise ChunkingError(f"Chunking failed: {str(e)}")
    
    def chunk_text_with_metadata(self,
                                 text: str,
                                 method: Optional[str] = None,
                                 max_size: Optional[int] = None,
                                 overlap: Optional[int] = None,
                                 language: Optional[str] = None,
                                 **options) -> List[ChunkResult]:
        """
        Chunk text and return results with metadata.
        
        Args:
            text: Text to chunk
            method: Chunking method (uses default if not specified)
            max_size: Maximum chunk size (uses default if not specified)
            overlap: Overlap between chunks (uses default if not specified)
            language: Language of the text (uses default if not specified)
            **options: Additional method-specific options
            
        Returns:
            List of ChunkResult objects with text and metadata
        """
        # Use defaults if not specified
        method = method or self.config.default_method.value
        max_size = max_size if max_size is not None else self.config.default_max_size
        overlap = overlap if overlap is not None else self.config.default_overlap
        language = language or self.config.language
        
        # Get strategy
        if method not in self._strategies:
            raise InvalidChunkingMethodError(f"Unknown chunking method: {method}")
        
        strategy = self._strategies[method]
        
        # Update strategy language if different
        if language != strategy.language:
            strategy.language = language
        
        try:
            # Get chunks with metadata
            return strategy.chunk_with_metadata(text, max_size, overlap, **options)
        except Exception as e:
            logger.error(f"Chunking with metadata failed: {e}")
            if isinstance(e, ChunkingError):
                raise
            raise ChunkingError(f"Chunking failed: {str(e)}")
    
    def chunk_text_generator(self,
                           text: str,
                           method: Optional[str] = None,
                           max_size: Optional[int] = None,
                           overlap: Optional[int] = None,
                           language: Optional[str] = None,
                           **options) -> Generator[str, None, None]:
        """
        Memory-efficient generator for chunking large texts.
        
        Args:
            text: Text to chunk
            method: Chunking method (uses default if not specified)
            max_size: Maximum chunk size (uses default if not specified)
            overlap: Overlap between chunks (uses default if not specified)
            language: Language of the text (uses default if not specified)
            **options: Additional method-specific options
            
        Yields:
            Individual text chunks
        """
        # Use defaults if not specified
        method = method or self.config.default_method.value
        max_size = max_size if max_size is not None else self.config.default_max_size
        overlap = overlap if overlap is not None else self.config.default_overlap
        language = language or self.config.language
        
        # Get strategy
        if method not in self._strategies:
            raise InvalidChunkingMethodError(f"Unknown chunking method: {method}")
        
        strategy = self._strategies[method]
        
        # Update strategy language if different
        if language != strategy.language:
            strategy.language = language
        
        # Use generator method of strategy
        for chunk in strategy.chunk_generator(text, max_size, overlap, **options):
            yield chunk
    
    def get_available_methods(self) -> List[str]:
        """
        Get list of available chunking methods.
        
        Returns:
            List of method names
        """
        return list(self._strategies.keys())
    
    def get_strategy(self, method: str):
        """
        Get a specific chunking strategy instance.
        
        Args:
            method: Method name
            
        Returns:
            Strategy instance
            
        Raises:
            InvalidChunkingMethodError: If method not found
        """
        if method not in self._strategies:
            raise InvalidChunkingMethodError(f"Unknown chunking method: {method}")
        return self._strategies[method]
    
    def _get_cache_key(self, text: str, method: str, max_size: int,
                      overlap: int, language: str, options: Dict) -> str:
        """
        Generate cache key for chunking parameters.
        
        Args:
            text: Input text
            method: Chunking method
            max_size: Maximum chunk size
            overlap: Overlap size
            language: Language code
            options: Additional options
            
        Returns:
            Cache key string
        """
        # Use hash of text to avoid storing full text in key
        text_hash = hash(text)
        options_str = str(sorted(options.items()))
        return f"{text_hash}:{method}:{max_size}:{overlap}:{language}:{options_str}"
    
    def clear_cache(self):
        """Clear the chunk cache."""
        if self._cache is not None:
            self._cache.clear()
            logger.debug("Chunk cache cleared")


# Convenience function for backward compatibility
def create_chunker(config: Optional[Dict[str, Any]] = None) -> Chunker:
    """
    Create a chunker instance with optional configuration.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        Configured Chunker instance
    """
    if config:
        chunker_config = ChunkerConfig(**config)
    else:
        chunker_config = ChunkerConfig()
    
    return Chunker(chunker_config)