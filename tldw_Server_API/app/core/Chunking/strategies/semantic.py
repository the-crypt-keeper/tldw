# strategies/semantic.py
"""
Semantic chunking strategy.
Groups text based on semantic similarity using TF-IDF vectors and cosine similarity.
"""

from typing import List, Optional, Generator, Dict, Any
from loguru import logger
import warnings

from ..base import BaseChunkingStrategy, ChunkResult, ChunkMetadata
from ..exceptions import ChunkingError


class SemanticChunkingStrategy(BaseChunkingStrategy):
    """
    Chunks text based on semantic similarity between sentences.
    Uses TF-IDF vectorization and cosine similarity to identify
    natural semantic boundaries in the text.
    """
    
    def __init__(self, 
                 language: str = 'en',
                 similarity_threshold: float = 0.3):
        """
        Initialize semantic chunking strategy.
        
        Args:
            language: Language code for text processing
            similarity_threshold: Threshold for semantic similarity (0-1)
        """
        super().__init__(language)
        self.similarity_threshold = similarity_threshold
        self._vectorizer = None
        self._sklearn_available = None
        self._nltk_available = None
        
        # Check dependencies
        self._check_dependencies()
        
        logger.debug(f"SemanticChunkingStrategy initialized with threshold: {similarity_threshold}")
    
    def _check_dependencies(self):
        """Check if required dependencies are available."""
        # Check scikit-learn
        try:
            from sklearn.feature_extraction.text import TfidfVectorizer
            from sklearn.metrics.pairwise import cosine_similarity
            self._sklearn_available = True
        except ImportError:
            self._sklearn_available = False
            logger.warning(
                "scikit-learn not available. Semantic chunking will not be available. "
                "Install with: pip install scikit-learn"
            )
        
        # Check NLTK
        try:
            import nltk
            self._nltk_available = True
            # Try to ensure punkt is downloaded
            try:
                nltk.data.find('tokenizers/punkt')
            except LookupError:
                logger.info("Downloading NLTK punkt tokenizer...")
                nltk.download('punkt', quiet=True)
        except ImportError:
            self._nltk_available = False
            logger.warning(
                "NLTK not available. Will use simple sentence splitting. "
                "Install with: pip install nltk"
            )
    
    def chunk(self,
              text: str,
              max_size: int,
              overlap: int = 0,
              **options) -> List[str]:
        """
        Chunk text based on semantic similarity.
        
        Args:
            text: Text to chunk
            max_size: Maximum size per chunk (in units specified by 'unit' option)
            overlap: Number of sentences to overlap between chunks
            **options: Additional options:
                - unit: Unit for max_size ('words', 'tokens', 'characters')
                - similarity_threshold: Override default similarity threshold
                - min_chunk_size: Minimum chunk size before breaking on similarity
                
        Returns:
            List of text chunks grouped by semantic similarity
        """
        if not self.validate_parameters(text, max_size, overlap):
            return []
        
        # Check dependencies
        if not self._sklearn_available:
            raise ChunkingError(
                "scikit-learn not installed. Cannot use semantic chunking. "
                "Install with: pip install scikit-learn"
            )
        
        # Import here to avoid import errors if not available
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.metrics.pairwise import cosine_similarity
        
        # Get options
        unit = options.get('unit', 'words')
        similarity_threshold = options.get('similarity_threshold', self.similarity_threshold)
        min_chunk_size = options.get('min_chunk_size', max_size // 2)
        
        # Split into sentences
        sentences = self._split_sentences(text)
        
        if not sentences:
            return []
        
        # Single sentence edge case
        if len(sentences) == 1:
            return [text.strip()] if text.strip() else []
        
        # Vectorize sentences
        try:
            vectorizer = TfidfVectorizer()
            # Filter out empty sentences
            valid_sentences = [s for s in sentences if s.strip()]
            if not valid_sentences:
                return []
            
            sentence_vectors = vectorizer.fit_transform(valid_sentences)
            
        except ValueError as e:
            # Can happen if all words are stop words or text is too short
            logger.warning(
                f"TF-IDF vectorization failed (possibly all stop words): {e}. "
                "Returning single chunk."
            )
            return [text.strip()] if text.strip() else []
        
        # Build chunks based on semantic similarity
        chunks = []
        current_chunk_sentences = []
        current_size = 0
        
        for i, sentence in enumerate(valid_sentences):
            sentence_size = self._count_units(sentence, unit)
            
            # Check if adding sentence exceeds max size
            if current_size + sentence_size > max_size and current_chunk_sentences:
                # Save current chunk
                chunks.append(' '.join(current_chunk_sentences))
                
                # Handle overlap
                if overlap > 0 and len(current_chunk_sentences) > overlap:
                    current_chunk_sentences = current_chunk_sentences[-overlap:]
                    current_size = sum(self._count_units(s, unit) for s in current_chunk_sentences)
                else:
                    current_chunk_sentences = []
                    current_size = 0
            
            # Add sentence to current chunk
            current_chunk_sentences.append(sentence)
            current_size += sentence_size
            
            # Check semantic similarity with next sentence
            if i + 1 < len(valid_sentences):
                # Calculate cosine similarity
                current_vector = sentence_vectors[i:i+1]
                next_vector = sentence_vectors[i+1:i+2]
                
                try:
                    similarity = cosine_similarity(current_vector, next_vector)[0, 0]
                except (IndexError, ValueError) as e:
                    logger.warning(f"Could not compute similarity at index {i}: {e}")
                    similarity = 1.0  # Assume high similarity on error
                
                # Break if similarity is low and chunk is large enough
                if (similarity < similarity_threshold and 
                    current_size >= min_chunk_size and 
                    current_chunk_sentences):
                    
                    # Save current chunk
                    chunks.append(' '.join(current_chunk_sentences))
                    
                    # Handle overlap
                    if overlap > 0 and len(current_chunk_sentences) > overlap:
                        current_chunk_sentences = current_chunk_sentences[-overlap:]
                        current_size = sum(self._count_units(s, unit) for s in current_chunk_sentences)
                    else:
                        current_chunk_sentences = []
                        current_size = 0
        
        # Add remaining sentences
        if current_chunk_sentences:
            chunks.append(' '.join(current_chunk_sentences))
        
        return chunks
    
    def _split_sentences(self, text: str) -> List[str]:
        """
        Split text into sentences.
        
        Args:
            text: Text to split
            
        Returns:
            List of sentences
        """
        if self._nltk_available:
            import nltk
            
            # Language mapping for NLTK
            nltk_lang_map = {
                'en': 'english',
                'es': 'spanish', 
                'fr': 'french',
                'de': 'german',
                'it': 'italian',
                'pt': 'portuguese',
                'nl': 'dutch',
                'pl': 'polish',
                'ru': 'russian',
                'tr': 'turkish'
            }
            
            nltk_language = nltk_lang_map.get(self.language, 'english')
            
            try:
                sentences = nltk.sent_tokenize(text, language=nltk_language)
                return sentences
            except LookupError:
                logger.warning(
                    f"NLTK punkt tokenizer for '{nltk_language}' not found. "
                    "Using fallback sentence splitting."
                )
            except Exception as e:
                logger.error(f"NLTK sentence tokenization failed: {e}")
        
        # Fallback: simple regex-based sentence splitting
        import re
        
        # Split on sentence-ending punctuation followed by space and capital letter
        sentence_pattern = r'(?<=[.!?])\s+(?=[A-Z])'
        sentences = re.split(sentence_pattern, text)
        
        # Filter out empty sentences
        sentences = [s.strip() for s in sentences if s.strip()]
        
        # If no sentences found, split on newlines
        if not sentences:
            sentences = [s.strip() for s in text.split('\n') if s.strip()]
        
        return sentences
    
    def _count_units(self, text: str, unit: str) -> int:
        """
        Count units in text.
        
        Args:
            text: Text to count units in
            unit: Unit type ('words', 'tokens', 'characters')
            
        Returns:
            Unit count
        """
        if unit == 'words':
            return len(text.split())
        elif unit == 'characters':
            return len(text)
        elif unit == 'tokens':
            # Try to use a tokenizer if available
            try:
                from transformers import AutoTokenizer
                tokenizer = AutoTokenizer.from_pretrained('gpt2')
                return len(tokenizer.encode(text))
            except ImportError:
                # Fallback to word count * 1.3 (approximate)
                return int(len(text.split()) * 1.3)
        else:
            logger.warning(f"Unknown unit type '{unit}'. Using word count.")
            return len(text.split())
    
    def chunk_generator(self,
                       text: str,
                       max_size: int,
                       overlap: int = 0,
                       **options) -> Generator[str, None, None]:
        """
        Generator version of chunk method.
        
        Note: Semantic chunking requires analyzing the entire text
        to compute similarities, so this just yields pre-computed chunks.
        
        Args:
            text: Text to chunk
            max_size: Maximum size per chunk
            overlap: Number of sentences to overlap
            **options: Additional options
            
        Yields:
            Individual text chunks
        """
        # Semantic chunking needs full text analysis, so compute all chunks first
        chunks = self.chunk(text, max_size, overlap, **options)
        for chunk in chunks:
            yield chunk
    
    def chunk_with_metadata(self,
                           text: str,
                           max_size: int,
                           overlap: int = 0,
                           **options) -> List[ChunkResult]:
        """
        Chunk text and return results with metadata.
        
        Args:
            text: Text to chunk
            max_size: Maximum size per chunk
            overlap: Number of sentences to overlap
            **options: Additional options
            
        Returns:
            List of ChunkResult objects with metadata
        """
        chunks = self.chunk(text, max_size, overlap, **options)
        results = []
        
        current_pos = 0
        for i, chunk in enumerate(chunks):
            # Find chunk position in original text
            start_pos = text.find(chunk, current_pos)
            if start_pos == -1:
                # Chunk might have been modified, try to find approximate position
                start_pos = current_pos
            
            end_pos = start_pos + len(chunk)
            current_pos = end_pos
            
            metadata = ChunkMetadata(
                index=i,
                start_char=start_pos,
                end_char=end_pos,
                char_count=len(chunk),
                word_count=len(chunk.split()),
                sentence_count=len(self._split_sentences(chunk)),
                language=self.language,
                method='semantic',
                options={
                    'similarity_threshold': options.get('similarity_threshold', self.similarity_threshold),
                    'unit': options.get('unit', 'words'),
                    'overlap': overlap
                }
            )
            
            results.append(ChunkResult(
                text=chunk,
                metadata=metadata
            ))
        
        return results