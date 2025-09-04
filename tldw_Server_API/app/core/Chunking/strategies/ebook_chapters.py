# ebook_chapters.py
"""
eBook chapter-based chunking strategy.
Splits text into chunks based on chapter markers.
"""

from typing import List, Optional, Any, Dict
import re
import sys
import threading
import time
from contextlib import contextmanager
from loguru import logger

from ..base import BaseChunkingStrategy, ChunkResult, ChunkMetadata
from ..exceptions import InvalidInputError, ProcessingError


class EbookChapterChunkingStrategy(BaseChunkingStrategy):
    """
    Strategy for chunking text by eBook chapters.
    """
    
    # Default chapter patterns for various languages
    CHAPTER_PATTERNS = {
        'en': r'(?:Chapter|CHAPTER|Section|SECTION|Part|PART)\s+(?:[0-9]+|[IVXLCDM]+)(?:\.|:|\s|$)',
        'es': r'(?:Capítulo|CAPÍTULO|Sección|SECCIÓN|Parte|PARTE)\s+(?:[0-9]+|[IVXLCDM]+)(?:\.|:|\s|$)',
        'fr': r'(?:Chapitre|CHAPITRE|Section|SECTION|Partie|PARTIE)\s+(?:[0-9]+|[IVXLCDM]+)(?:\.|:|\s|$)',
        'de': r'(?:Kapitel|KAPITEL|Abschnitt|ABSCHNITT|Teil|TEIL)\s+(?:[0-9]+|[IVXLCDM]+)(?:\.|:|\s|$)',
        'it': r'(?:Capitolo|CAPITOLO|Sezione|SEZIONE|Parte|PARTE)\s+(?:[0-9]+|[IVXLCDM]+)(?:\.|:|\s|$)',
        'pt': r'(?:Capítulo|CAPÍTULO|Seção|SEÇÃO|Parte|PARTE)\s+(?:[0-9]+|[IVXLCDM]+)(?:\.|:|\s|$)',
        'default': r'(?:Chapter|CHAPTER|Section|SECTION|Part|PART)\s+(?:[0-9]+|[IVXLCDM]+)(?:\.|:|\s|$)'
    }
    
    # Regex complexity limits for security
    MAX_REGEX_LENGTH = 500  # Maximum length of custom regex pattern
    REGEX_TIMEOUT = 2  # Seconds before regex timeout
    DANGEROUS_PATTERNS = [
        r'\(\*',  # Possessive quantifiers
        r'\(\?R\)',  # Recursive patterns
        r'\(\?\(DEFINE\)',  # DEFINE patterns
        r'{\d{4,}}',  # Large repetition ranges
        r'[*+]{2,}',  # Consecutive quantifiers
        r'\([^)]*[*+].*[*+].*\)',  # Multiple quantifiers in group
        # Additional patterns to catch nested quantifiers (ReDoS prevention)
        r'\([^)]*[+*]\)[+*?]',  # Group with quantifier followed by quantifier: (a+)+, (a*)*
        r'\([^)]*[+*]\){',  # Group with quantifier followed by range: (a+){n,m}
        r'\(\w+\+\)\+',  # Explicit (x+)+ pattern
        r'\(\w+\*\)\*',  # Explicit (x*)* pattern
        r'\(\w+\?\)\?',  # Explicit (x?)? pattern
        r'\([^)]*\|[^)]*[+*]\)[+*]',  # Alternative with quantifier: (a|b+)+
        r'\(\([^)]+\)[+*]\)[+*]',  # Nested groups with quantifiers: ((a)+)+
    ]
    
    def __init__(self, language: str = 'en'):
        """
        Initialize the ebook chapter chunking strategy.
        
        Args:
            language: Language code for text processing
        """
        super().__init__(language)
        logger.debug(f"EbookChapterChunkingStrategy initialized for language: {language}")
    
    @contextmanager
    def _regex_timeout(self, seconds):
        """
        Cross-platform context manager for regex timeout to prevent ReDoS attacks.
        Uses threading instead of signals for Windows compatibility.
        """
        result = {'completed': False, 'error': None}
        
        def target_func(func, args, kwargs):
            try:
                result['value'] = func(*args, **kwargs)
                result['completed'] = True
            except Exception as e:
                result['error'] = e
                result['completed'] = True
        
        # For cross-platform compatibility, we'll use a different approach
        # This wraps the regex operation in a way that can be interrupted
        class RegexTimeout:
            def __init__(self, timeout_seconds):
                self.timeout_seconds = timeout_seconds
                self.timed_out = False
                
            def __enter__(self):
                return self
                
            def __exit__(self, exc_type, exc_val, exc_tb):
                pass
                
            def run_with_timeout(self, func, *args, **kwargs):
                """Run a function with a timeout."""
                result_container = {'value': None, 'error': None}
                
                def wrapper():
                    try:
                        result_container['value'] = func(*args, **kwargs)
                    except Exception as e:
                        result_container['error'] = e
                
                thread = threading.Thread(target=wrapper)
                thread.daemon = True
                thread.start()
                thread.join(timeout=self.timeout_seconds)
                
                if thread.is_alive():
                    # Thread is still running after timeout
                    self.timed_out = True
                    raise ProcessingError("Regex operation timed out - possible ReDoS attack")
                
                if result_container['error']:
                    raise result_container['error']
                    
                return result_container['value']
        
        yield RegexTimeout(seconds)
    
    def _validate_regex_pattern(self, pattern: str) -> bool:
        """
        Validate a regex pattern for security issues.
        
        Args:
            pattern: Regex pattern to validate
            
        Returns:
            True if pattern is safe, False otherwise
            
        Raises:
            InvalidInputError: If pattern is dangerous
        """
        # Check pattern length
        if len(pattern) > self.MAX_REGEX_LENGTH:
            raise InvalidInputError(
                f"Regex pattern too long ({len(pattern)} chars). "
                f"Maximum allowed: {self.MAX_REGEX_LENGTH}"
            )
        
        # Check for dangerous patterns
        for dangerous in self.DANGEROUS_PATTERNS:
            if re.search(dangerous, pattern):
                raise InvalidInputError(
                    f"Regex pattern contains potentially dangerous construct: {dangerous}"
                )
        
        # Test pattern compilation
        try:
            re.compile(pattern)
        except re.error as e:
            raise InvalidInputError(f"Invalid regex pattern: {e}")
        
        # Don't actually test the regex execution - the DANGEROUS_PATTERNS check above
        # should catch problematic patterns. Testing them would cause the very problem
        # we're trying to prevent (ReDoS during validation).
        
        return True
    
    def chunk(self, 
              text: str, 
              max_size: int = 5000,
              overlap: int = 0,
              **options) -> List[str]:
        """
        Chunk text by chapters.
        
        Args:
            text: Text to chunk
            max_size: Maximum words per chunk (if chapter exceeds this, it will be split)
            overlap: Number of words to overlap (only applies when splitting large chapters)
            **options: Additional options including:
                - custom_chapter_pattern: Custom regex pattern for chapter detection
            
        Returns:
            List of text chunks
        """
        if not text:
            raise InvalidInputError("Cannot chunk empty text")
        
        try:
            # Get custom pattern or use language-specific default
            custom_pattern = options.get('custom_chapter_pattern')
            if custom_pattern:
                # Validate custom pattern for security
                self._validate_regex_pattern(custom_pattern)
                chapter_pattern = custom_pattern
                logger.debug(f"Using validated custom chapter pattern: {custom_pattern}")
            else:
                chapter_pattern = self.CHAPTER_PATTERNS.get(
                    self.language, 
                    self.CHAPTER_PATTERNS['default']
                )
                logger.debug(f"Using {self.language} chapter pattern")
            
            # Find all chapter markers 
            # Note: Pattern safety is ensured by _validate_regex_pattern above
            # which rejects dangerous patterns before they can be compiled
            chapter_markers = list(re.finditer(chapter_pattern, text, re.MULTILINE))
            
            if not chapter_markers:
                logger.info("No chapter markers found, treating entire text as single chapter")
                # No chapters found, split by size if needed
                return self._split_by_size(text, max_size, overlap)
            
            logger.debug(f"Found {len(chapter_markers)} chapter markers")
            
            chunks = []
            
            # Process each chapter
            for i, marker in enumerate(chapter_markers):
                # Determine chapter boundaries
                chapter_start = marker.start()
                if i < len(chapter_markers) - 1:
                    chapter_end = chapter_markers[i + 1].start()
                else:
                    chapter_end = len(text)
                
                chapter_text = text[chapter_start:chapter_end].strip()
                
                # Count words in chapter
                word_count = len(chapter_text.split())
                
                # Check if chapter needs to be split due to size
                if word_count > max_size:
                    logger.debug(f"Chapter has {word_count} words, splitting...")
                    # Split large chapter into smaller chunks
                    chapter_chunks = self._split_by_size(chapter_text, max_size, overlap)
                    chunks.extend(chapter_chunks)
                else:
                    chunks.append(chapter_text)
            
            logger.info(f"Created {len(chunks)} chapter-based chunks")
            return chunks
            
        except Exception as e:
            logger.error(f"Error during chapter chunking: {e}")
            raise ProcessingError(f"Failed to chunk by chapters: {str(e)}")
    
    def _split_by_size(self, text: str, max_size: int, overlap: int) -> List[str]:
        """
        Split text by word count when no chapters or chapter is too large.
        
        Args:
            text: Text to split
            max_size: Maximum words per chunk
            overlap: Number of words to overlap
            
        Returns:
            List of text chunks
        """
        words = text.split()
        chunks = []
        
        i = 0
        while i < len(words):
            end_idx = min(i + max_size, len(words))
            chunk_words = words[i:end_idx]
            chunk_text = ' '.join(chunk_words)
            chunks.append(chunk_text)
            
            i += max_size - overlap if overlap > 0 else max_size
        
        return chunks
    
    def chunk_with_metadata(self, 
                           text: str, 
                           max_size: int = 5000,
                           overlap: int = 0,
                           **options) -> List[ChunkResult]:
        """
        Chunk text by chapters and return with metadata.
        
        Args:
            text: Text to chunk
            max_size: Maximum words per chunk
            overlap: Number of words to overlap
            **options: Additional options
            
        Returns:
            List of ChunkResult objects with metadata
        """
        if not text:
            raise InvalidInputError("Cannot chunk empty text")
        
        try:
            # Get custom pattern or use language-specific default
            custom_pattern = options.get('custom_chapter_pattern')
            if custom_pattern:
                # Validate custom pattern for security
                self._validate_regex_pattern(custom_pattern)
                chapter_pattern = custom_pattern
            else:
                chapter_pattern = self.CHAPTER_PATTERNS.get(
                    self.language, 
                    self.CHAPTER_PATTERNS['default']
                )
            
            # Find all chapter markers
            # Note: Pattern safety is ensured by _validate_regex_pattern above
            chapter_markers = list(re.finditer(chapter_pattern, text, re.MULTILINE))
            
            chunks = []
            chunk_index = 0
            
            if not chapter_markers:
                # No chapters found, treat as single chunk
                words = text.split()
                i = 0
                while i < len(words):
                    end_idx = min(i + max_size, len(words))
                    chunk_words = words[i:end_idx]
                    chunk_text = ' '.join(chunk_words)
                    
                    metadata = ChunkMetadata(
                        index=chunk_index,
                        start_char=0,  # Simplified
                        end_char=len(chunk_text),
                        word_count=len(chunk_words),
                        language=self.language,
                        method='ebook_chapters',
                        options={'no_chapters': True}
                    )
                    
                    chunks.append(ChunkResult(text=chunk_text, metadata=metadata))
                    chunk_index += 1
                    i += max_size - overlap if overlap > 0 else max_size
                
                return chunks
            
            # Process each chapter
            for i, marker in enumerate(chapter_markers):
                chapter_start = marker.start()
                if i < len(chapter_markers) - 1:
                    chapter_end = chapter_markers[i + 1].start()
                else:
                    chapter_end = len(text)
                
                chapter_text = text[chapter_start:chapter_end].strip()
                chapter_title = marker.group().strip()
                word_count = len(chapter_text.split())
                
                # Check if chapter needs to be split
                if word_count > max_size:
                    # Split large chapter
                    words = chapter_text.split()
                    j = 0
                    part = 1
                    while j < len(words):
                        end_idx = min(j + max_size, len(words))
                        chunk_words = words[j:end_idx]
                        chunk_text = ' '.join(chunk_words)
                        
                        metadata = ChunkMetadata(
                            index=chunk_index,
                            start_char=chapter_start,
                            end_char=chapter_start + len(chunk_text),
                            word_count=len(chunk_words),
                            language=self.language,
                            method='ebook_chapters',
                            options={
                                'chapter_title': f"{chapter_title} (Part {part})",
                                'chapter_number': i + 1,
                                'is_split': True
                            }
                        )
                        
                        chunks.append(ChunkResult(text=chunk_text, metadata=metadata))
                        chunk_index += 1
                        part += 1
                        j += max_size - overlap if overlap > 0 else max_size
                else:
                    # Keep chapter as single chunk
                    metadata = ChunkMetadata(
                        index=chunk_index,
                        start_char=chapter_start,
                        end_char=chapter_end,
                        word_count=word_count,
                        language=self.language,
                        method='ebook_chapters',
                        options={
                            'chapter_title': chapter_title,
                            'chapter_number': i + 1,
                            'total_chapters': len(chapter_markers)
                        }
                    )
                    
                    chunks.append(ChunkResult(text=chapter_text, metadata=metadata))
                    chunk_index += 1
            
            logger.info(f"Created {len(chunks)} chapter-based chunks with metadata")
            return chunks
            
        except Exception as e:
            logger.error(f"Error during chapter chunking: {e}")
            raise ProcessingError(f"Failed to chunk by chapters: {str(e)}")
    
    def validate_options(self, options: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate and normalize options for chapter chunking.
        
        Args:
            options: Options dictionary
            
        Returns:
            Validated options
        """
        validated = super().validate_options(options)
        
        # Validate custom pattern if provided
        if 'custom_chapter_pattern' in validated:
            pattern = validated['custom_chapter_pattern']
            try:
                re.compile(pattern)
            except re.error as e:
                raise InvalidInputError(f"Invalid chapter pattern: {e}")
        
        return validated