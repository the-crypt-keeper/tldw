"""
Simple chunking service wrapper for the simplified RAG implementation.

This provides a minimal interface to the existing chunking functionality
to satisfy the import requirements of the simplified RAG service.
"""

from typing import List, Dict, Any, Optional
import logging
from tldw_chatbook.Chunking.Chunk_Lib import Chunker

logger = logging.getLogger(__name__)


class ChunkingError(Exception):
    """Base exception for chunking-related errors."""
    pass


class InvalidChunkingMethodError(ChunkingError):
    """Exception raised when an invalid chunking method is specified."""
    pass


class ChunkingService:
    """
    Minimal chunking service that wraps the existing Chunk_Lib functionality.
    
    This is a temporary compatibility layer for the simplified RAG implementation.
    """
    
    def __init__(self):
        """Initialize the chunking service."""
        # Initialize with default options
        self.default_options = {
            'method': 'words',
            'max_size': 400,
            'overlap': 200
        }
        logger.info("Initialized ChunkingService wrapper")
    
    def chunk_text(self, 
                   content: str, 
                   chunk_size: int = 400, 
                   chunk_overlap: int = 100,
                   method: str = "words") -> List[Dict[str, Any]]:
        """
        Chunk text using the specified method.
        
        Args:
            content: Text to chunk
            chunk_size: Target size of chunks
            chunk_overlap: Overlap between chunks
            method: Chunking method ("words", "sentences", "paragraphs", "tokens")
            
        Returns:
            List of chunk dictionaries with text and metadata
        """
        try:
            # Create options for the chunker
            options = {
                'method': method,
                'max_size': chunk_size,
                'overlap': chunk_overlap
            }
            
            # Create a new Chunker instance with the specific options
            chunker = Chunker(options)
            
            # Use the main chunk_text method
            chunks = chunker.chunk_text(content, method=method)
            
            # Convert to expected format
            result_chunks = []
            
            # Calculate approximate character positions based on chunking method
            if method == "words":
                # For word-based chunking, calculate positions more accurately
                words = content.split()
                word_positions = []
                current_pos = 0
                
                # Build word position map efficiently
                # Instead of O(n²) find operations, use a single pass
                for i, word in enumerate(words):
                    # Skip ahead to the next word occurrence
                    while current_pos < len(content) and content[current_pos:current_pos+1].isspace():
                        current_pos += 1
                    
                    if current_pos < len(content):
                        # Found start of word
                        word_start = current_pos
                        word_end = word_start + len(word)
                        
                        # Verify it matches (handle edge cases)
                        if current_pos + len(word) <= len(content) and content[word_start:word_end] == word:
                            word_positions.append((word_start, word_end))
                            current_pos = word_end
                        else:
                            # Fallback for mismatched words (rare edge case)
                            word_start = content.find(word, current_pos)
                            if word_start != -1:
                                word_positions.append((word_start, word_start + len(word)))
                                current_pos = word_start + len(word)
                
                # Map chunks to positions
                # Calculate word indices for each chunk accounting for overlap
                current_word_idx = 0
                for i, chunk_text in enumerate(chunks):
                    if not chunk_text:
                        continue
                    
                    chunk_words = chunk_text.split()
                    
                    # For chunks after the first, adjust for overlap
                    if i > 0 and chunk_overlap > 0:
                        # Move back by overlap amount
                        current_word_idx = max(0, current_word_idx - chunk_overlap)
                    
                    # Calculate start position
                    if word_positions and current_word_idx < len(word_positions):
                        start_char = word_positions[current_word_idx][0]
                        
                        # Calculate end position
                        end_word_idx = min(current_word_idx + len(chunk_words) - 1, len(word_positions) - 1)
                        if end_word_idx >= 0 and end_word_idx < len(word_positions):
                            end_char = word_positions[end_word_idx][1]
                        else:
                            # If we can't find the exact end, estimate it
                            end_char = min(start_char + len(chunk_text), len(content))
                    else:
                        # Fallback for edge cases - shouldn't normally happen
                        logger.warning(f"Word position calculation fallback for chunk {i}")
                        if i == 0:
                            start_char = 0
                            end_char = min(len(chunk_text), len(content))
                        else:
                            # Use the previous chunk's end as a starting point
                            if result_chunks:
                                start_char = result_chunks[-1]['end_char']
                                end_char = min(start_char + len(chunk_text), len(content))
                            else:
                                start_char = 0
                                end_char = min(len(chunk_text), len(content))
                    
                    result_chunks.append({
                        'text': chunk_text,
                        'start_char': start_char,
                        'end_char': end_char,
                        'word_count': len(chunk_words),
                        'chunk_index': i
                    })
                    
                    # Update current position for next chunk
                    current_word_idx += len(chunk_words)
            else:
                # For other methods, use approximate positions
                total_length = len(content)
                num_chunks = len([c for c in chunks if c])
                
                for i, chunk_text in enumerate(chunks):
                    if not chunk_text:
                        continue
                    
                    # Approximate position based on chunk index
                    if num_chunks > 1:
                        start_ratio = i / num_chunks
                        start_char = int(total_length * start_ratio)
                        end_char = min(start_char + len(chunk_text), total_length)
                    else:
                        start_char = 0
                        end_char = len(chunk_text)
                    
                    result_chunks.append({
                        'text': chunk_text,
                        'start_char': start_char,
                        'end_char': end_char,
                        'word_count': len(chunk_text.split()),
                        'chunk_index': i
                    })
            
            logger.debug(f"Chunked text into {len(result_chunks)} chunks using method '{method}'")
            return result_chunks
            
        except Exception as e:
            logger.error(f"Error chunking text with method '{method}': {e}", exc_info=True)
            # Re-raise with more context instead of hiding the error
            raise ChunkingError(f"Failed to chunk text using method '{method}': {str(e)}") from e


def improved_chunking_process(text: str, options: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Wrapper function to provide compatibility with the server's chunking interface.
    
    Args:
        text: The text to chunk
        options: Dictionary containing chunking options:
            - method: The chunking method to use
            - max_size: Maximum size of each chunk
            - overlap: Overlap between chunks
            
    Returns:
        List of chunk dictionaries
        
    Raises:
        InvalidChunkingMethodError: If the chunking method is not supported
        ChunkingError: For other chunking-related errors
    """
    service = ChunkingService()
    
    # Validate method
    valid_methods = ['words', 'sentences', 'paragraphs', 'tokens', 'semantic']
    method = options.get('method', 'words')
    if method not in valid_methods:
        raise InvalidChunkingMethodError(f"Invalid chunking method: {method}. Valid methods are: {valid_methods}")
    
    try:
        return service.chunk_text(
            text, 
            chunk_size=options.get('max_size', 400),
            chunk_overlap=options.get('overlap', 100),
            method=options.get('method', 'words')
        )
    except Exception as e:
        raise ChunkingError(f"Error during chunking: {str(e)}") from e