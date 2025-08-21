# strategies/sentences.py
"""
Sentence-based chunking strategy.
Splits text into chunks based on sentence count with optional overlap.
"""

import re
from typing import List, Optional, Dict, Any, Generator
from loguru import logger

from ..base import BaseChunkingStrategy


class SentenceChunkingStrategy(BaseChunkingStrategy):
    """
    Chunks text by sentence count.
    Uses language-specific sentence boundary detection.
    """
    
    def __init__(self, language: str = 'en'):
        """
        Initialize sentence chunking strategy.
        
        Args:
            language: Language code for text processing
        """
        super().__init__(language)
        
        # Try to import pysbd for better sentence splitting
        self.pysbd_available = False
        try:
            import pysbd
            self.pysbd = pysbd
            self.pysbd_available = True
            logger.debug("pysbd available for sentence segmentation")
        except ImportError:
            logger.debug("pysbd not available, using fallback sentence splitting")
        
        # Language-specific sentence delimiters
        self.sentence_delimiters = {
            'zh': ['。', '！', '？', '；'],
            'zh-cn': ['。', '！', '？', '；'],
            'zh-tw': ['。', '！', '？', '；'],
            'ja': ['。', '！', '？'],
            'ko': ['.', '!', '?', '。', '！', '？'],
            'ar': ['.', '!', '?', '؟', '۔'],
            'hi': ['।', '|', '.', '!', '?'],
            'th': [' ', '!', '?'],
            'default': ['.', '!', '?']
        }
        
        logger.debug(f"SentenceChunkingStrategy initialized for language: {language}")
    
    def chunk(self,
              text: str,
              max_size: int,
              overlap: int = 0,
              **options) -> List[str]:
        """
        Chunk text by sentence count.
        
        Args:
            text: Text to chunk
            max_size: Maximum sentences per chunk
            overlap: Number of sentences to overlap between chunks
            **options: Additional options:
                - combine_short: Combine short sentences
                - min_sentence_length: Minimum characters per sentence
                
        Returns:
            List of text chunks
        """
        if not self.validate_parameters(text, max_size, overlap):
            return []
        
        # Adjust overlap if needed
        if overlap >= max_size:
            logger.warning(f"Overlap ({overlap}) >= max_size ({max_size}), setting to max_size - 1")
            overlap = max_size - 1
        
        # Split text into sentences
        sentences = self._split_sentences(text)
        
        if not sentences:
            return []
        
        logger.debug(f"Split text into {len(sentences)} sentences")
        
        # Optionally combine short sentences
        if options.get('combine_short', False):
            min_length = options.get('min_sentence_length', 10)
            sentences = self._combine_short_sentences(sentences, min_length)
        
        # Create chunks
        chunks = []
        step = max(1, max_size - overlap)
        
        for i in range(0, len(sentences), step):
            chunk_sentences = sentences[i:i + max_size]
            
            # Join sentences with appropriate delimiter
            if self.language in ['zh', 'zh-cn', 'zh-tw', 'ja']:
                chunk = ''.join(chunk_sentences)
            else:
                chunk = ' '.join(chunk_sentences)
            
            chunks.append(chunk.strip())
        
        logger.info(f"Created {len(chunks)} chunks from {len(sentences)} sentences")
        return chunks
    
    def _split_sentences(self, text: str) -> List[str]:
        """
        Split text into sentences based on language.
        
        Args:
            text: Text to split
            
        Returns:
            List of sentences
        """
        # Try pysbd first if available
        if self.pysbd_available:
            sentences = self._split_with_pysbd(text)
            if sentences:
                return sentences
        
        # Fallback to language-specific splitting
        return self._split_with_regex(text)
    
    def _split_with_pysbd(self, text: str) -> List[str]:
        """
        Split sentences using pysbd library.
        
        Args:
            text: Text to split
            
        Returns:
            List of sentences
        """
        try:
            # Map our language codes to pysbd language codes
            pysbd_lang_map = {
                'en': 'en',
                'de': 'de', 
                'fr': 'fr',
                'it': 'it',
                'es': 'es',
                'pt': 'pt',
                'nl': 'nl',
                'pl': 'pl',
                'zh': 'zh',
                'zh-cn': 'zh',
                'zh-tw': 'zh',
                'ja': 'ja',
                'ar': 'ar',
                'hi': 'hi',
                'ru': 'ru',
                'da': 'da',
                'sv': 'sv',
                'no': 'no'
            }
            
            lang_code = pysbd_lang_map.get(self.language, 'en')
            segmenter = self.pysbd.Segmenter(language=lang_code, clean=False)
            sentences = segmenter.segment(text)
            
            logger.debug(f"pysbd split text into {len(sentences)} sentences")
            return sentences
            
        except Exception as e:
            logger.warning(f"pysbd sentence splitting failed: {e}, falling back to regex")
            return []
    
    def _split_with_regex(self, text: str) -> List[str]:
        """
        Split sentences using regex patterns.
        
        Args:
            text: Text to split
            
        Returns:
            List of sentences
        """
        # Get delimiters for language
        delimiters = self.sentence_delimiters.get(
            self.language,
            self.sentence_delimiters['default']
        )
        
        # Build regex pattern
        delimiter_pattern = '|'.join(re.escape(d) for d in delimiters)
        pattern = f'([{delimiter_pattern}])'
        
        # Split on delimiters but keep them
        parts = re.split(pattern, text)
        
        # Reconstruct sentences with their delimiters
        sentences = []
        current_sentence = ""
        
        for i, part in enumerate(parts):
            if part in delimiters:
                if current_sentence:
                    current_sentence += part
                    sentences.append(current_sentence.strip())
                    current_sentence = ""
            else:
                current_sentence += part
        
        # Add any remaining text
        if current_sentence.strip():
            sentences.append(current_sentence.strip())
        
        # Filter out empty sentences
        sentences = [s for s in sentences if s.strip()]
        
        logger.debug(f"Regex split text into {len(sentences)} sentences")
        return sentences
    
    def _combine_short_sentences(self, sentences: List[str], min_length: int) -> List[str]:
        """
        Combine short sentences to meet minimum length.
        
        Args:
            sentences: List of sentences
            min_length: Minimum characters per sentence
            
        Returns:
            List of combined sentences
        """
        combined = []
        current = ""
        
        for sentence in sentences:
            if len(current) + len(sentence) < min_length:
                if self.language in ['zh', 'zh-cn', 'zh-tw', 'ja']:
                    current += sentence
                else:
                    current = (current + " " + sentence).strip()
            else:
                if current:
                    combined.append(current)
                current = sentence
        
        if current:
            combined.append(current)
        
        logger.debug(f"Combined {len(sentences)} sentences into {len(combined)} sentences")
        return combined
    
    def chunk_generator(self,
                       text: str,
                       max_size: int,
                       overlap: int = 0,
                       **options) -> Generator[str, None, None]:
        """
        Memory-efficient generator version of chunk.
        
        Args:
            text: Text to chunk
            max_size: Maximum sentences per chunk
            overlap: Number of sentences to overlap between chunks
            **options: Additional options
            
        Yields:
            Individual text chunks
        """
        chunks = self.chunk(text, max_size, overlap, **options)
        for chunk in chunks:
            yield chunk