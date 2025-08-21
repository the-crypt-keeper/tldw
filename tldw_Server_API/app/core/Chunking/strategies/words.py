# strategies/words.py
"""
Word-based chunking strategy.
Splits text into chunks based on word count with optional overlap.
"""

import re
from typing import List, Optional, Dict, Any, Generator
from loguru import logger

from ..base import BaseChunkingStrategy, ChunkResult, ChunkMetadata


class WordChunkingStrategy(BaseChunkingStrategy):
    """
    Chunks text by word count.
    Supports multiple languages and various word tokenization methods.
    """
    
    def __init__(self, language: str = 'en'):
        """
        Initialize word chunking strategy.
        
        Args:
            language: Language code for text processing
        """
        super().__init__(language)
        
        # Language-specific word tokenizers
        self._word_tokenizers = {
            'zh': self._tokenize_chinese,
            'zh-cn': self._tokenize_chinese,
            'zh-tw': self._tokenize_chinese,
            'ja': self._tokenize_japanese,
            'ko': self._tokenize_korean,
            'th': self._tokenize_thai,
            'default': self._tokenize_default
        }
        
        logger.debug(f"WordChunkingStrategy initialized for language: {language}")
    
    def chunk(self,
              text: str,
              max_size: int,
              overlap: int = 0,
              **options) -> List[str]:
        """
        Chunk text by word count.
        
        Args:
            text: Text to chunk
            max_size: Maximum words per chunk
            overlap: Number of words to overlap between chunks
            **options: Additional options:
                - preserve_sentences: Try to break at sentence boundaries
                - min_chunk_size: Minimum words per chunk
                
        Returns:
            List of text chunks
        """
        if not self.validate_parameters(text, max_size, overlap):
            return []
        
        # Adjust overlap if needed
        if overlap >= max_size:
            logger.warning(f"Overlap ({overlap}) >= max_size ({max_size}), setting to max_size - 1")
            overlap = max_size - 1
        
        # Tokenize text into words
        words = self._tokenize_text(text)
        
        if not words:
            return []
        
        logger.debug(f"Chunking {len(words)} words with max_size={max_size}, overlap={overlap}")
        
        # Create chunks
        chunks = []
        step = max(1, max_size - overlap)
        
        for i in range(0, len(words), step):
            chunk_words = words[i:i + max_size]
            
            # Try to preserve sentence boundaries if requested
            if options.get('preserve_sentences', False) and i + max_size < len(words):
                chunk_text = ' '.join(chunk_words)
                # Look for sentence boundary near the end
                last_period = chunk_text.rfind('. ')
                last_question = chunk_text.rfind('? ')
                last_exclaim = chunk_text.rfind('! ')
                
                boundary = max(last_period, last_question, last_exclaim)
                if boundary > len(chunk_text) * 0.8:  # If boundary is in last 20% of chunk
                    # Find word index at boundary
                    partial_text = chunk_text[:boundary + 2]
                    boundary_words = partial_text.split()
                    chunk_words = chunk_words[:len(boundary_words)]
            
            chunk = self._join_words(chunk_words)
            
            # Apply minimum chunk size if specified
            min_size = options.get('min_chunk_size', 0)
            if len(chunk_words) >= min_size or not chunks:
                chunks.append(chunk)
            elif chunks:
                # Merge with previous chunk if too small
                chunks[-1] = chunks[-1] + ' ' + chunk
        
        logger.info(f"Created {len(chunks)} chunks from {len(words)} words")
        return chunks
    
    def _tokenize_text(self, text: str) -> List[str]:
        """
        Tokenize text into words based on language.
        
        Args:
            text: Text to tokenize
            
        Returns:
            List of words
        """
        # Get appropriate tokenizer for language
        tokenizer = self._word_tokenizers.get(
            self.language, 
            self._word_tokenizers['default']
        )
        
        return tokenizer(text)
    
    def _tokenize_default(self, text: str) -> List[str]:
        """
        Default word tokenization using simple splitting.
        
        Args:
            text: Text to tokenize
            
        Returns:
            List of words
        """
        # Simple word splitting on whitespace and punctuation
        words = text.split()
        return words
    
    def _tokenize_chinese(self, text: str) -> List[str]:
        """
        Chinese word tokenization.
        
        Args:
            text: Chinese text to tokenize
            
        Returns:
            List of words
        """
        try:
            import jieba
            words = list(jieba.cut(text))
            logger.debug(f"Tokenized Chinese text into {len(words)} words using jieba")
            return words
        except ImportError:
            logger.warning("jieba not available, falling back to character splitting for Chinese")
            # For Chinese without jieba, split on characters
            # Remove spaces and newlines first
            text = text.replace(' ', '').replace('\n', ' ')
            # Split into characters but keep some punctuation as boundaries
            import re
            pattern = r'[。！？，；：、]'
            segments = re.split(pattern, text)
            words = []
            for segment in segments:
                if segment:
                    # Add characters as individual "words"
                    words.extend(list(segment))
            return words
    
    def _tokenize_japanese(self, text: str) -> List[str]:
        """
        Japanese word tokenization.
        
        Args:
            text: Japanese text to tokenize
            
        Returns:
            List of words
        """
        try:
            import fugashi
            tagger = fugashi.Tagger('-Owakati')
            words = tagger.parse(text).split()
            logger.debug(f"Tokenized Japanese text into {len(words)} words using fugashi")
            return words
        except ImportError:
            logger.warning("fugashi not available, falling back to character splitting for Japanese")
            # Similar to Chinese fallback
            text = text.replace(' ', '').replace('\n', ' ')
            import re
            pattern = r'[。！？、]'
            segments = re.split(pattern, text)
            words = []
            for segment in segments:
                if segment:
                    words.extend(list(segment))
            return words
    
    def _tokenize_korean(self, text: str) -> List[str]:
        """
        Korean word tokenization.
        
        Args:
            text: Korean text to tokenize
            
        Returns:
            List of words
        """
        try:
            from konlpy.tag import Okt
            okt = Okt()
            words = okt.morphs(text)
            logger.debug(f"Tokenized Korean text into {len(words)} words using KoNLPy")
            return words
        except ImportError:
            logger.warning("KoNLPy not available, using space splitting for Korean")
            return text.split()
    
    def _tokenize_thai(self, text: str) -> List[str]:
        """
        Thai word tokenization.
        
        Args:
            text: Thai text to tokenize
            
        Returns:
            List of words
        """
        try:
            from pythainlp import word_tokenize
            words = word_tokenize(text, engine='newmm')
            logger.debug(f"Tokenized Thai text into {len(words)} words using PyThaiNLP")
            return words
        except ImportError:
            logger.warning("PyThaiNLP not available, using character splitting for Thai")
            # Thai doesn't use spaces between words
            return list(text.replace(' ', '').replace('\n', ' '))
    
    def _join_words(self, words: List[str]) -> str:
        """
        Join words back into text based on language.
        
        Args:
            words: List of words to join
            
        Returns:
            Joined text
        """
        if self.language in ['zh', 'zh-cn', 'zh-tw', 'ja', 'th']:
            # For languages without spaces between words
            return ''.join(words)
        else:
            # For languages with spaces
            return ' '.join(words)
    
    def chunk_generator(self,
                       text: str,
                       max_size: int,
                       overlap: int = 0,
                       **options) -> Generator[str, None, None]:
        """
        Memory-efficient generator version of chunk.
        
        Args:
            text: Text to chunk
            max_size: Maximum words per chunk
            overlap: Number of words to overlap between chunks
            **options: Additional options
            
        Yields:
            Individual text chunks
        """
        if not self.validate_parameters(text, max_size, overlap):
            return
        
        # Adjust overlap if needed
        if overlap >= max_size:
            overlap = max_size - 1
        
        # Process text in blocks for memory efficiency
        BLOCK_SIZE = 10000  # Process 10k words at a time
        
        words = self._tokenize_text(text)
        step = max(1, max_size - overlap)
        
        for i in range(0, len(words), step):
            chunk_words = words[i:i + max_size]
            chunk = self._join_words(chunk_words)
            
            # Apply minimum size if specified
            min_size = options.get('min_chunk_size', 0)
            if len(chunk_words) >= min_size:
                yield chunk