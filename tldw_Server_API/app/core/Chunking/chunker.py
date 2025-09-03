# chunker.py
"""
Main Chunker class that provides a unified interface for all chunking strategies.
This is the primary entry point for the chunking module.
"""

from typing import List, Dict, Any, Optional, Union, Generator
from pathlib import Path
import unicodedata
import ast
import threading
from functools import lru_cache
from collections import OrderedDict
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
from .security_logger import get_security_logger, SecurityEventType


class LRUCache:
    """
    Thread-safe LRU (Least Recently Used) cache implementation.
    """
    
    def __init__(self, max_size: int = 100):
        """
        Initialize the LRU cache.
        
        Args:
            max_size: Maximum number of items to store in cache
        """
        self.max_size = max_size
        self.cache = OrderedDict()
        self._lock = threading.Lock()
        self.hits = 0
        self.misses = 0
        
    def get(self, key: str) -> Optional[Any]:
        """
        Get an item from cache.
        
        Args:
            key: Cache key
            
        Returns:
            Cached value or None if not found
        """
        with self._lock:
            if key in self.cache:
                # Move to end (most recently used)
                self.cache.move_to_end(key)
                self.hits += 1
                return self.cache[key]
            self.misses += 1
            return None
    
    def put(self, key: str, value: Any) -> None:
        """
        Put an item into cache.
        
        Args:
            key: Cache key
            value: Value to cache
        """
        with self._lock:
            if key in self.cache:
                # Update existing and move to end
                self.cache.move_to_end(key)
                self.cache[key] = value
            else:
                # Add new item
                self.cache[key] = value
                # Remove oldest if over capacity
                if len(self.cache) > self.max_size:
                    self.cache.popitem(last=False)
    
    def clear(self) -> None:
        """Clear the cache."""
        with self._lock:
            self.cache.clear()
            self.hits = 0
            self.misses = 0
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get cache statistics.
        
        Returns:
            Dictionary with cache statistics
        """
        with self._lock:
            total = self.hits + self.misses
            hit_rate = self.hits / total if total > 0 else 0
            return {
                'size': len(self.cache),
                'max_size': self.max_size,
                'hits': self.hits,
                'misses': self.misses,
                'hit_rate': hit_rate
            }


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
        
        # Cache for processed results - using LRU cache
        self._cache = LRUCache(max_size=self.config.cache_size) if self.config.enable_cache else None
        
        # Security logger
        self._security_logger = get_security_logger()
        
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
    
    def _sanitize_input(self, text: str) -> str:
        """
        Sanitize input text for security.
        
        Args:
            text: Raw input text
            
        Returns:
            Sanitized text
            
        Raises:
            InvalidInputError: If text contains dangerous content
        """
        if not isinstance(text, str):
            raise InvalidInputError(f"Expected string input, got {type(text).__name__}")
        
        # Remove null bytes which could cause issues
        if '\x00' in text:
            logger.warning("Null bytes detected in input, removing them")
            null_byte_count = text.count('\x00')
            self._security_logger.log_suspicious_content(
                "null_bytes", f"Found {null_byte_count} null bytes in input", source="sanitize_input"
            )
            text = text.replace('\x00', '')
        
        # Normalize unicode to prevent various unicode-based attacks
        # Using NFC (Canonical Decomposition, followed by Canonical Composition)
        text = unicodedata.normalize('NFC', text)
        
        # Check for control characters (except common ones like \n, \t, \r)
        allowed_control_chars = {'\n', '\t', '\r', '\f'}
        control_chars = []
        for char in text:
            if unicodedata.category(char) == 'Cc' and char not in allowed_control_chars:
                control_chars.append(repr(char))
        
        if control_chars:
            logger.warning(f"Suspicious control characters found: {control_chars[:10]}")
            self._security_logger.log_suspicious_content(
                "control_characters", f"Found {len(control_chars)} suspicious control characters", source="sanitize_input"
            )
            # Remove suspicious control characters
            for char in set(control_chars):
                text = text.replace(ast.literal_eval(char), '')
        
        # Check for bidirectional text override characters (could be used for spoofing)
        bidi_chars = ['\u202a', '\u202b', '\u202c', '\u202d', '\u202e', '\u2066', '\u2067', '\u2068', '\u2069']
        for bidi_char in bidi_chars:
            if bidi_char in text:
                logger.warning("Bidirectional override characters detected and removed")
                text = text.replace(bidi_char, '')
        
        return text
    
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
        # Sanitize and validate input
        text = self._sanitize_input(text)
        
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
            cached_result = self._cache.get(cache_key)
            if cached_result is not None:
                logger.debug("Returning cached result")
                return cached_result
        
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
                self._cache.put(cache_key, chunks)
            
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
        # Sanitize input
        text = self._sanitize_input(text)
        
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
        # Sanitize input
        text = self._sanitize_input(text)
        
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
    
    def get_cache_stats(self) -> Optional[Dict[str, Any]]:
        """
        Get cache statistics.
        
        Returns:
            Dictionary with cache statistics or None if cache is disabled
        """
        if self._cache is not None:
            return self._cache.get_stats()
        return None
    
    def chunk_text_generator(self,
                            text: str,
                            method: Optional[str] = None,
                            max_size: Optional[int] = None,
                            overlap: Optional[int] = None,
                            language: Optional[str] = None,
                            **options) -> Generator[str, None, None]:
        """
        Memory-efficient generator for chunking large texts.
        Yields chunks one at a time instead of loading all into memory.
        
        Args:
            text: Text to chunk
            method: Chunking method to use
            max_size: Maximum size per chunk
            overlap: Number of units to overlap between chunks
            language: Language code for text processing
            **options: Additional method-specific options
            
        Yields:
            Text chunks one at a time
        """
        # Sanitize input
        text = self._sanitize_input(text)
        if not text:
            return
        
        # Use defaults where not specified
        method = method or self.config.default_method.value
        max_size = max_size if max_size is not None else self.config.default_max_size
        overlap = overlap if overlap is not None else self.config.default_overlap
        language = language or self.config.language
        
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
            logger.debug(f"Starting generator chunking with method={method}, max_size={max_size}")
            
            # Check if strategy supports generator mode
            if hasattr(strategy, 'chunk_generator'):
                # Use generator method if available
                for chunk in strategy.chunk_generator(text, max_size, overlap, **options):
                    yield chunk
            else:
                # Fallback to regular chunking but yield one at a time
                chunks = strategy.chunk(text, max_size, overlap, **options)
                for chunk in chunks:
                    yield chunk
                    
        except Exception as e:
            logger.error(f"Generator chunking failed: {e}")
            if isinstance(e, ChunkingError):
                raise
            raise ChunkingError(f"Generator chunking failed: {str(e)}")
    
    def chunk_file_stream(self,
                         file_path: Union[str, Path],
                         method: Optional[str] = None,
                         max_size: Optional[int] = None,
                         overlap: Optional[int] = None,
                         language: Optional[str] = None,
                         buffer_size: int = 8192,
                         **options) -> Generator[str, None, None]:
        """
        Stream-process a file for memory-efficient chunking of very large files.
        
        Args:
            file_path: Path to the file to chunk
            method: Chunking method to use
            max_size: Maximum size per chunk
            overlap: Number of units to overlap between chunks
            language: Language code for text processing
            buffer_size: Size of read buffer in bytes
            **options: Additional method-specific options
            
        Yields:
            Text chunks one at a time
        """
        file_path = Path(file_path)
        
        if not file_path.exists():
            raise InvalidInputError(f"File not found: {file_path}")
        
        # Check file size
        file_size = file_path.stat().st_size
        if file_size > self.config.max_text_size:
            logger.warning(f"File size ({file_size} bytes) exceeds max size "
                         f"({self.config.max_text_size} bytes), will process in streaming mode")
        
        logger.info(f"Stream processing file: {file_path} ({file_size} bytes)")
        
        # Read file in chunks and accumulate until we have enough for chunking
        buffer = ""
        overlap_buffer = ""
        
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                while True:
                    # Read next buffer
                    chunk = f.read(buffer_size)
                    if not chunk:
                        # Process remaining buffer
                        if buffer:
                            for text_chunk in self.chunk_text_generator(
                                buffer, method, max_size, overlap, language, **options
                            ):
                                yield text_chunk
                        break
                    
                    buffer += chunk
                    
                    # Process buffer when it's large enough
                    if len(buffer) >= max_size * 2:  # Keep some extra for overlap
                        # Find a good split point (end of sentence/paragraph)
                        split_point = self._find_split_point(buffer, max_size)
                        
                        # Process the first part
                        to_process = buffer[:split_point]
                        for text_chunk in self.chunk_text_generator(
                            overlap_buffer + to_process, 
                            method, max_size, overlap, language, **options
                        ):
                            yield text_chunk
                        
                        # Keep overlap for next iteration
                        if overlap > 0:
                            overlap_buffer = to_process[-overlap:]
                        
                        # Keep the rest in buffer
                        buffer = buffer[split_point:]
                        
        except Exception as e:
            logger.error(f"File stream processing failed: {e}")
            raise ChunkingError(f"Failed to process file stream: {str(e)}")
    
    def _find_split_point(self, text: str, target: int) -> int:
        """
        Find a good split point in text near the target position.
        Prefers to split at paragraph or sentence boundaries.
        
        Args:
            text: Text to find split point in
            target: Target position for split
            
        Returns:
            Best split position
        """
        if len(text) <= target:
            return len(text)
        
        # Look for paragraph break near target
        for i in range(target, min(target + 500, len(text))):
            if text[i:i+2] == '\n\n':
                return i + 2
        
        # Look for sentence end near target
        for i in range(target, min(target + 200, len(text))):
            if text[i] in '.!?' and i + 1 < len(text) and text[i + 1].isspace():
                return i + 1
        
        # Look for any newline
        for i in range(target, min(target + 100, len(text))):
            if text[i] == '\n':
                return i + 1
        
        # Default to target position
        return target


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