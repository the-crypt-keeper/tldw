# token_chunker.py
"""
Token-based chunking implementation using transformers.
This module provides token-based chunking with graceful fallbacks
when transformers library is not available.
"""

from typing import List, Optional, Protocol
from loguru import logger
from ..Utils.optional_deps import get_safe_import, require_dependency


class TokenizerProtocol(Protocol):
    """Protocol for tokenizer interface."""
    
    def encode(self, text: str) -> List[int]:
        """Encode text to token IDs."""
        ...
    
    def decode(self, token_ids: List[int], skip_special_tokens: bool = True) -> str:
        """Decode token IDs back to text."""
        ...


class TransformersTokenizer:
    """Tokenizer wrapper using transformers library."""
    
    def __init__(self, tokenizer_name_or_path: str = "gpt2"):
        self.tokenizer_name_or_path = tokenizer_name_or_path
        self._tokenizer = None
        self._transformers = get_safe_import('transformers')
        self.available = self._transformers is not None
        
        if not self.available:
            logger.debug(f"transformers not available, token-based chunking will be disabled")
    
    @property
    def tokenizer(self):
        """Lazy-load the tokenizer when first accessed."""
        if self._tokenizer is None:
            if not self.available:
                raise ImportError(
                    "transformers library not found. Please install it to use token-based chunking. "
                    "Install with: pip install tldw_chatbook[chunker]"
                )
            
            try:
                AutoTokenizer = self._transformers.AutoTokenizer
                logger.info(f"Loading tokenizer: {self.tokenizer_name_or_path}")
                self._tokenizer = AutoTokenizer.from_pretrained(self.tokenizer_name_or_path)
                logger.debug(f"Tokenizer {self.tokenizer_name_or_path} loaded successfully")
            except Exception as e:
                logger.error(f"Failed to load tokenizer '{self.tokenizer_name_or_path}': {e}")
                raise ImportError(f"Failed to load tokenizer: {e}") from e
        
        return self._tokenizer
    
    def encode(self, text: str) -> List[int]:
        """Encode text to token IDs."""
        return self.tokenizer.encode(text)
    
    def decode(self, token_ids: List[int], skip_special_tokens: bool = True) -> str:
        """Decode token IDs back to text."""
        return self.tokenizer.decode(token_ids, skip_special_tokens=skip_special_tokens)
    
    def count_tokens(self, text: str) -> int:
        """Count tokens in text."""
        return len(self.encode(text))


class FallbackTokenizer:
    """Fallback tokenizer using simple word splitting when transformers unavailable."""
    
    def __init__(self, tokenizer_name_or_path: str = "fallback"):
        self.tokenizer_name_or_path = tokenizer_name_or_path
        self.available = True
        logger.warning(
            "Using fallback tokenizer (word-based). Token counts will be approximate. "
            "Install transformers for accurate token-based chunking."
        )
    
    def encode(self, text: str) -> List[int]:
        """Simulate encoding by splitting into words and converting to fake token IDs."""
        words = text.split()
        # Simple hash-based fake token IDs for consistency
        return [hash(word) % 50000 for word in words]
    
    def decode(self, token_ids: List[int], skip_special_tokens: bool = True) -> str:
        """Cannot accurately decode from fake token IDs."""
        raise NotImplementedError(
            "Fallback tokenizer cannot decode token IDs back to text. "
            "This should not be called in normal chunking operations."
        )
    
    def count_tokens(self, text: str) -> int:
        """Approximate token count using word count."""
        return len(text.split())


class TokenBasedChunker:
    """Token-based text chunker with optional transformers support."""
    
    def __init__(self, tokenizer_name_or_path: str = "gpt2"):
        self.tokenizer_name_or_path = tokenizer_name_or_path
        self._tokenizer = None
    
    @property
    def tokenizer(self) -> TokenizerProtocol:
        """Get tokenizer, creating it if needed."""
        if self._tokenizer is None:
            try:
                self._tokenizer = TransformersTokenizer(self.tokenizer_name_or_path)
                # Test if tokenizer actually works
                _ = self._tokenizer.encode("test")
            except (ImportError, Exception) as e:
                logger.warning(f"Could not initialize transformers tokenizer: {e}")
                logger.info("Falling back to word-based approximation")
                self._tokenizer = FallbackTokenizer(self.tokenizer_name_or_path)
        
        return self._tokenizer
    
    def chunk_by_tokens(self, text: str, max_tokens: int, overlap_tokens: int = 0) -> List[str]:
        """
        Chunk text by token count.
        
        Args:
            text: Text to chunk
            max_tokens: Maximum tokens per chunk
            overlap_tokens: Number of tokens to overlap between chunks
            
        Returns:
            List of text chunks
        """
        # Import the constant from Chunk_Lib to avoid circular import
        MAX_CHUNK_SIZE_TOKENS = 10000  # Maximum tokens per chunk
        
        if max_tokens <= 0:
            raise ValueError(f"max_tokens must be positive, got {max_tokens}")
        
        if max_tokens > MAX_CHUNK_SIZE_TOKENS:
            raise ValueError(f"max_tokens {max_tokens} exceeds maximum allowed {MAX_CHUNK_SIZE_TOKENS}")
        
        if overlap_tokens < 0:
            raise ValueError(f"overlap_tokens must be non-negative, got {overlap_tokens}")
        
        if overlap_tokens >= max_tokens:
            raise ValueError(f"Token overlap {overlap_tokens} must be less than max_tokens {max_tokens}")
        
        # For fallback tokenizer, we need to use a different approach
        if isinstance(self.tokenizer, FallbackTokenizer):
            return self._chunk_by_word_approximation(text, max_tokens, overlap_tokens)
        
        # Use actual tokenizer
        try:
            tokens = self.tokenizer.encode(text)
            logger.debug(f"Text tokenized into {len(tokens)} tokens")
            
            if len(tokens) <= max_tokens:
                return [text]
            
            chunks = []
            step = max_tokens - overlap_tokens
            if step <= 0:
                step = max_tokens
            
            for i in range(0, len(tokens), step):
                chunk_tokens = tokens[i:i + max_tokens]
                chunk_text = self.tokenizer.decode(chunk_tokens, skip_special_tokens=True)
                chunks.append(chunk_text)
                logger.debug(f"Created token chunk {len(chunks)} with {len(chunk_tokens)} tokens")
            
            return [chunk.strip() for chunk in chunks if chunk.strip()]
            
        except Exception as e:
            logger.error(f"Token-based chunking failed: {e}")
            logger.info("Falling back to word-based approximation")
            return self._chunk_by_word_approximation(text, max_tokens, overlap_tokens)
    
    def _chunk_by_word_approximation(self, text: str, max_tokens: int, overlap_tokens: int) -> List[str]:
        """Approximate token-based chunking using words."""
        words = text.split()
        
        # Rough approximation: 1 token ≈ 0.75 words for English
        # This varies significantly by language and tokenizer
        approx_max_words = int(max_tokens * 0.75)
        approx_overlap_words = int(overlap_tokens * 0.75)
        
        if len(words) <= approx_max_words:
            return [text]
        
        chunks = []
        step = approx_max_words - approx_overlap_words
        if step <= 0:
            step = approx_max_words
        
        for i in range(0, len(words), step):
            chunk_words = words[i:i + approx_max_words]
            chunks.append(' '.join(chunk_words))
        
        logger.debug(f"Created {len(chunks)} chunks using word approximation")
        return [chunk.strip() for chunk in chunks if chunk.strip()]
    
    def count_tokens(self, text: str) -> int:
        """Count tokens in text."""
        return self.tokenizer.count_tokens(text)
    
    def is_transformers_available(self) -> bool:
        """Check if transformers tokenizer is available."""
        return isinstance(self.tokenizer, TransformersTokenizer) and self.tokenizer.available


def create_token_chunker(tokenizer_name_or_path: str = "gpt2") -> TokenBasedChunker:
    """Factory function to create a token-based chunker."""
    return TokenBasedChunker(tokenizer_name_or_path)