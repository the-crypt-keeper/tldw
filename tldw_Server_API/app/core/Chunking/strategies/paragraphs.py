# paragraphs.py
"""
Paragraph-based chunking strategy.
Splits text into chunks based on paragraph boundaries.
"""

from typing import List, Optional, Any, Dict
import re
from loguru import logger

from ..base import BaseChunkingStrategy, ChunkResult, ChunkMetadata
from ..exceptions import InvalidInputError, ProcessingError


class ParagraphChunkingStrategy(BaseChunkingStrategy):
    """
    Strategy for chunking text by paragraphs.
    """
    
    def __init__(self, language: str = 'en'):
        """
        Initialize the paragraph chunking strategy.
        
        Args:
            language: Language code for text processing
        """
        super().__init__(language)
        logger.debug(f"ParagraphChunkingStrategy initialized for language: {language}")
    
    def chunk(self, 
              text: str, 
              max_size: int = 2,
              overlap: int = 0,
              **options) -> List[str]:
        """
        Chunk text by paragraphs.
        
        Args:
            text: Text to chunk
            max_size: Maximum number of paragraphs per chunk
            overlap: Number of paragraphs to overlap between chunks
            **options: Additional options
            
        Returns:
            List of text chunks
        """
        if not text:
            raise InvalidInputError("Cannot chunk empty text")
        
        if max_size < 1:
            raise InvalidInputError(f"max_size must be at least 1, got {max_size}")
        
        if overlap < 0:
            raise InvalidInputError(f"overlap must be non-negative, got {overlap}")
        
        if overlap >= max_size:
            raise InvalidInputError(f"overlap ({overlap}) must be less than max_size ({max_size})")
        
        try:
            # Split text into paragraphs (handling various paragraph separators)
            # Match two or more newlines, optionally with whitespace
            paragraphs = re.split(r'\n\s*\n+', text.strip())
            
            # Filter out empty paragraphs
            paragraphs = [p.strip() for p in paragraphs if p.strip()]
            
            if not paragraphs:
                # If no paragraphs found, treat entire text as one paragraph
                paragraphs = [text.strip()]
            
            logger.debug(f"Split text into {len(paragraphs)} paragraphs")
            
            chunks = []
            i = 0
            chunk_index = 0
            
            while i < len(paragraphs):
                # Determine the end index for this chunk
                end_idx = min(i + max_size, len(paragraphs))
                
                # Extract paragraphs for this chunk
                chunk_paragraphs = paragraphs[i:end_idx]
                
                # Join paragraphs with double newline
                chunk_text = '\n\n'.join(chunk_paragraphs)
                chunks.append(chunk_text)
                
                chunk_index += 1
                
                # Move to next chunk with overlap
                i += max_size - overlap if overlap > 0 else max_size
            
            logger.info(f"Created {len(chunks)} paragraph-based chunks")
            return chunks
            
        except Exception as e:
            logger.error(f"Error during paragraph chunking: {e}")
            raise ProcessingError(f"Failed to chunk by paragraphs: {str(e)}")
    
    def chunk_with_metadata(self, 
                           text: str, 
                           max_size: int = 2,
                           overlap: int = 0,
                           **options) -> List[ChunkResult]:
        """
        Chunk text by paragraphs and return with metadata.
        
        Args:
            text: Text to chunk
            max_size: Maximum number of paragraphs per chunk
            overlap: Number of paragraphs to overlap between chunks
            **options: Additional options
            
        Returns:
            List of ChunkResult objects with metadata
        """
        if not text:
            raise InvalidInputError("Cannot chunk empty text")
        
        if max_size < 1:
            raise InvalidInputError(f"max_size must be at least 1, got {max_size}")
        
        if overlap < 0:
            raise InvalidInputError(f"overlap must be non-negative, got {overlap}")
        
        if overlap >= max_size:
            raise InvalidInputError(f"overlap ({overlap}) must be less than max_size ({max_size})")
        
        try:
            # Split text into paragraphs
            paragraphs = re.split(r'\n\s*\n+', text.strip())
            paragraphs = [p.strip() for p in paragraphs if p.strip()]
            
            if not paragraphs:
                paragraphs = [text.strip()]
            
            logger.debug(f"Split text into {len(paragraphs)} paragraphs")
            
            chunks = []
            i = 0
            chunk_index = 0
            
            while i < len(paragraphs):
                end_idx = min(i + max_size, len(paragraphs))
                chunk_paragraphs = paragraphs[i:end_idx]
                chunk_text = '\n\n'.join(chunk_paragraphs)
                
                # Calculate character positions
                if i == 0:
                    start_char = 0
                else:
                    start_char = sum(len(p) + 2 for p in paragraphs[:i])  # +2 for \n\n
                
                end_char = start_char + len(chunk_text)
                word_count = len(chunk_text.split())
                
                # Create metadata
                metadata = ChunkMetadata(
                    index=chunk_index,
                    start_char=start_char,
                    end_char=end_char,
                    word_count=word_count,
                    language=self.language,
                    method='paragraphs',
                    options={
                        'max_paragraphs': max_size,
                        'overlap': overlap,
                        'paragraph_count': len(chunk_paragraphs)
                    }
                )
                
                chunks.append(ChunkResult(
                    text=chunk_text,
                    metadata=metadata
                ))
                
                chunk_index += 1
                i += max_size - overlap if overlap > 0 else max_size
            
            logger.info(f"Created {len(chunks)} paragraph-based chunks with metadata")
            return chunks
            
        except Exception as e:
            logger.error(f"Error during paragraph chunking: {e}")
            raise ProcessingError(f"Failed to chunk by paragraphs: {str(e)}")
    
    def validate_options(self, options: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate and normalize options for paragraph chunking.
        
        Args:
            options: Options dictionary
            
        Returns:
            Validated options
        """
        validated = super().validate_options(options)
        
        # Ensure max_size is reasonable for paragraphs
        if 'max_size' in validated:
            if validated['max_size'] > 100:
                logger.warning(f"Very large max_size for paragraphs: {validated['max_size']}")
        
        return validated