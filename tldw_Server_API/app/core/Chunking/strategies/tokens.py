# strategies/tokens.py
"""
Token-based chunking strategy.
Splits text into chunks based on token count using transformers or fallback methods.
"""

from typing import List, Optional, Dict, Any, Generator, Protocol
from loguru import logger

from ..base import BaseChunkingStrategy
from ..exceptions import TokenizerError


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
    
    def __init__(self, model_name: str = "gpt2"):
        """
        Initialize transformers tokenizer.
        
        Args:
            model_name: Name of the tokenizer model
        """
        self.model_name = model_name
        self._tokenizer = None
        self._transformers = None
        
        # Try to import transformers
        try:
            import transformers
            self._transformers = transformers
            self.available = True
            logger.debug(f"transformers library available for tokenization")
        except ImportError:
            self.available = False
            logger.debug("transformers not available, will use fallback tokenization")
    
    @property
    def tokenizer(self):
        """Lazy-load the tokenizer when first accessed."""
        if self._tokenizer is None:
            if not self.available:
                raise ImportError(
                    "transformers library not found. Please install it for token-based chunking: "
                    "pip install transformers"
                )
            
            try:
                AutoTokenizer = self._transformers.AutoTokenizer
                logger.info(f"Loading tokenizer: {self.model_name}")
                self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
                logger.debug(f"Tokenizer {self.model_name} loaded successfully")
            except Exception as e:
                logger.error(f"Failed to load tokenizer '{self.model_name}': {e}")
                raise TokenizerError(f"Failed to load tokenizer: {e}")
        
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
    """Fallback tokenizer using simple word/character splitting."""
    
    def __init__(self, model_name: str = "fallback"):
        """
        Initialize fallback tokenizer.
        
        Args:
            model_name: Name identifier (for compatibility)
        """
        self.model_name = model_name
        self.available = True
        # Average tokens per word for different models
        self.tokens_per_word = {
            'gpt2': 1.3,
            'gpt-3.5-turbo': 1.3,
            'gpt-4': 1.3,
            'claude': 1.2,
            'llama': 1.5,
            'default': 1.3
        }
        
        logger.warning(
            "Using fallback tokenizer (word-based approximation). "
            "Install transformers for accurate token-based chunking: pip install transformers"
        )
    
    def encode(self, text: str) -> List[int]:
        """
        Simulate encoding by splitting into words and creating fake token IDs.
        
        Args:
            text: Text to encode
            
        Returns:
            List of fake token IDs
        """
        # Split into words and subwords
        import re
        
        # Split on whitespace and punctuation
        tokens = re.findall(r'\w+|[^\w\s]', text)
        
        # Create consistent fake token IDs
        token_ids = []
        for token in tokens:
            # Use hash for consistency
            token_id = abs(hash(token)) % 50000
            token_ids.append(token_id)
            
            # Simulate subword tokenization for longer words
            if len(token) > 7:
                # Add extra tokens for long words
                extra_tokens = (len(token) - 7) // 3
                for i in range(extra_tokens):
                    token_ids.append(abs(hash(f"{token}_{i}")) % 50000)
        
        return token_ids
    
    def decode(self, token_ids: List[int], skip_special_tokens: bool = True) -> str:
        """
        Cannot accurately decode from fake token IDs.
        
        Args:
            token_ids: List of token IDs
            skip_special_tokens: Whether to skip special tokens
            
        Returns:
            Placeholder text
            
        Raises:
            NotImplementedError: Always (cannot reverse fake tokenization)
        """
        raise NotImplementedError(
            "Fallback tokenizer cannot decode token IDs back to text. "
            "This should not be called in normal chunking operations."
        )
    
    def count_tokens(self, text: str) -> int:
        """
        Approximate token count using word count.
        
        Args:
            text: Text to count tokens for
            
        Returns:
            Approximate token count
        """
        word_count = len(text.split())
        # Use model-specific ratio if available
        ratio = self.tokens_per_word.get(self.model_name, self.tokens_per_word['default'])
        return int(word_count * ratio)


class TokenChunkingStrategy(BaseChunkingStrategy):
    """
    Chunks text by token count.
    Uses transformers library when available, falls back to word-based approximation.
    """
    
    def __init__(self, 
                 language: str = 'en',
                 tokenizer_name: str = 'gpt2'):
        """
        Initialize token chunking strategy.
        
        Args:
            language: Language code for text processing
            tokenizer_name: Name of the tokenizer to use
        """
        super().__init__(language)
        self.tokenizer_name = tokenizer_name
        self._tokenizer = None
        
        logger.debug(f"TokenChunkingStrategy initialized with tokenizer: {tokenizer_name}")
    
    @property
    def tokenizer(self) -> TokenizerProtocol:
        """Get or create tokenizer instance."""
        if self._tokenizer is None:
            # Try transformers first
            try:
                self._tokenizer = TransformersTokenizer(self.tokenizer_name)
                # Test if it actually works
                _ = self._tokenizer.encode("test")
                logger.info(f"Using transformers tokenizer: {self.tokenizer_name}")
            except (ImportError, Exception) as e:
                logger.warning(f"Could not initialize transformers tokenizer: {e}")
                logger.info("Falling back to word-based token approximation")
                self._tokenizer = FallbackTokenizer(self.tokenizer_name)
        
        return self._tokenizer
    
    def chunk(self,
              text: str,
              max_size: int,
              overlap: int = 0,
              **options) -> List[str]:
        """
        Chunk text by token count.
        
        Args:
            text: Text to chunk
            max_size: Maximum tokens per chunk
            overlap: Number of tokens to overlap between chunks
            **options: Additional options:
                - preserve_words: Try not to break words (when possible)
                - add_special_tokens: Include special tokens in encoding
                
        Returns:
            List of text chunks
        """
        if not self.validate_parameters(text, max_size, overlap):
            return []
        
        # Validate token-specific parameters
        MAX_CHUNK_SIZE_TOKENS = 10000  # Safety limit
        if max_size > MAX_CHUNK_SIZE_TOKENS:
            raise ValueError(
                f"max_size {max_size} exceeds maximum allowed {MAX_CHUNK_SIZE_TOKENS} tokens"
            )
        
        # Adjust overlap if needed
        if overlap >= max_size:
            logger.warning(f"Overlap ({overlap}) >= max_size ({max_size}), setting to max_size - 1")
            overlap = max_size - 1
        
        # For fallback tokenizer, use different approach
        if isinstance(self.tokenizer, FallbackTokenizer):
            return self._chunk_with_fallback(text, max_size, overlap, **options)
        
        # Encode text to tokens
        try:
            add_special = options.get('add_special_tokens', False)
            if hasattr(self.tokenizer, 'tokenizer'):
                # For TransformersTokenizer
                tokens = self.tokenizer.tokenizer.encode(
                    text, 
                    add_special_tokens=add_special
                )
            else:
                tokens = self.tokenizer.encode(text)
            
        except Exception as e:
            logger.error(f"Tokenization failed: {e}")
            raise TokenizerError(f"Failed to tokenize text: {e}")
        
        if not tokens:
            return []
        
        logger.debug(f"Tokenized text into {len(tokens)} tokens")
        
        # Create chunks
        chunks = []
        step = max(1, max_size - overlap)
        
        for i in range(0, len(tokens), step):
            chunk_tokens = tokens[i:i + max_size]
            
            # Decode tokens back to text
            try:
                chunk_text = self.tokenizer.decode(chunk_tokens)
                
                # Clean up chunk
                chunk_text = chunk_text.strip()
                
                if chunk_text:
                    chunks.append(chunk_text)
                    
            except Exception as e:
                logger.warning(f"Failed to decode chunk at position {i}: {e}")
                continue
        
        logger.info(f"Created {len(chunks)} token-based chunks")
        return chunks
    
    def _chunk_with_fallback(self, 
                            text: str,
                            max_size: int,
                            overlap: int,
                            **options) -> List[str]:
        """
        Chunk using fallback tokenizer (word-based approximation).
        
        Args:
            text: Text to chunk
            max_size: Maximum tokens per chunk
            overlap: Token overlap between chunks
            **options: Additional options
            
        Returns:
            List of text chunks
        """
        # Convert token counts to approximate word counts
        if hasattr(self.tokenizer, 'tokens_per_word'):
            ratio = self.tokenizer.tokens_per_word.get(
                self.tokenizer.model_name, 
                1.3
            )
        else:
            ratio = 1.3
        
        # Calculate word counts
        max_words = int(max_size / ratio)
        overlap_words = int(overlap / ratio)
        
        logger.debug(f"Using fallback: {max_size} tokens ≈ {max_words} words")
        
        # Split into words
        words = text.split()
        
        if not words:
            return []
        
        # Create chunks
        chunks = []
        step = max(1, max_words - overlap_words)
        
        for i in range(0, len(words), step):
            chunk_words = words[i:i + max_words]
            chunk_text = ' '.join(chunk_words)
            
            if chunk_text:
                chunks.append(chunk_text)
        
        return chunks
    
    def count_tokens(self, text: str) -> int:
        """
        Count tokens in text.
        
        Args:
            text: Text to count tokens for
            
        Returns:
            Token count
        """
        if hasattr(self.tokenizer, 'count_tokens'):
            return self.tokenizer.count_tokens(text)
        else:
            return len(self.tokenizer.encode(text))
    
    def chunk_generator(self,
                       text: str,
                       max_size: int,
                       overlap: int = 0,
                       **options) -> Generator[str, None, None]:
        """
        Memory-efficient generator version of chunk.
        
        Args:
            text: Text to chunk
            max_size: Maximum tokens per chunk
            overlap: Number of tokens to overlap between chunks
            **options: Additional options
            
        Yields:
            Individual text chunks
        """
        # For token-based chunking, we need to encode the full text
        # So we just use the regular chunk method
        chunks = self.chunk(text, max_size, overlap, **options)
        for chunk in chunks:
            yield chunk