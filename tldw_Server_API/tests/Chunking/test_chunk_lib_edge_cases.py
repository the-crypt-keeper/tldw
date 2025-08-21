# test_chunk_lib_edge_cases.py
# Tests for edge cases and security fixes in the Chunking module

import pytest
import json
from unittest.mock import patch, MagicMock

from tldw_Server_API.app.core.Chunking.Chunk_Lib import (
    Chunker,
    InvalidInputError,
    ChunkingError,
    InvalidChunkingMethodError
)

class TestChunkerEdgeCases:
    """Test edge cases and security improvements in the Chunker class."""
    
    def test_none_input_handling(self):
        """Test that None input is handled gracefully."""
        with patch('tldw_Server_API.app.core.Chunking.Chunk_Lib.AutoTokenizer.from_pretrained', MagicMock()):
            chunker = Chunker()
            result = chunker.chunk_text(None)
            assert result == []
    
    def test_non_string_input_raises_error(self):
        """Test that non-string input raises appropriate error."""
        with patch('tldw_Server_API.app.core.Chunking.Chunk_Lib.AutoTokenizer.from_pretrained', MagicMock()):
            chunker = Chunker()
            with pytest.raises(InvalidInputError, match="Expected string input"):
                chunker.chunk_text(123)
            with pytest.raises(InvalidInputError, match="Expected string input"):
                chunker.chunk_text(['list', 'of', 'strings'])
    
    def test_empty_string_handling(self):
        """Test that empty strings are handled properly."""
        with patch('tldw_Server_API.app.core.Chunking.Chunk_Lib.AutoTokenizer.from_pretrained', MagicMock()):
            chunker = Chunker()
            assert chunker.chunk_text("") == []
            assert chunker.chunk_text("   ") == []
            assert chunker.chunk_text("\n\n\n") == []
    
    def test_text_size_limit_enforcement(self):
        """Test that oversized text is rejected."""
        with patch('tldw_Server_API.app.core.Chunking.Chunk_Lib.AutoTokenizer.from_pretrained', MagicMock()):
            chunker = Chunker()
            # Create text larger than 100MB limit
            huge_text = "a" * (100_000_001)
            with pytest.raises(InvalidInputError, match="exceeds maximum allowed size"):
                chunker.chunk_text(huge_text)
    
    def test_json_size_limit_enforcement(self):
        """Test that oversized JSON is rejected."""
        with patch('tldw_Server_API.app.core.Chunking.Chunk_Lib.AutoTokenizer.from_pretrained', MagicMock()):
            chunker = Chunker()
            # Create JSON text larger than 50MB limit
            huge_json = json.dumps({"data": "x" * 50_000_001})
            with pytest.raises(InvalidInputError, match="JSON text size.*exceeds maximum"):
                chunker._chunk_text_by_json(huge_json, max_size=10, overlap=0)
    
    def test_negative_max_size_validation(self):
        """Test that negative or zero max_size is properly validated."""
        with patch('tldw_Server_API.app.core.Chunking.Chunk_Lib.AutoTokenizer.from_pretrained', MagicMock()):
            chunker = Chunker(options={"max_size": -1})
            with pytest.raises(ValueError, match="max_size must be positive"):
                chunker.chunk_text("Some text")
            
            chunker = Chunker(options={"max_size": 0})
            with pytest.raises(ValueError, match="max_size must be positive"):
                chunker.chunk_text("Some text")
    
    def test_negative_overlap_validation(self):
        """Test that negative overlap is properly validated."""
        with patch('tldw_Server_API.app.core.Chunking.Chunk_Lib.AutoTokenizer.from_pretrained', MagicMock()):
            chunker = Chunker(options={"max_size": 10, "overlap": -5})
            with pytest.raises(ValueError, match="overlap cannot be negative"):
                chunker.chunk_text("Some text to chunk")
    
    def test_overlap_exceeds_max_size_handling(self):
        """Test that overlap >= max_size is handled properly."""
        with patch('tldw_Server_API.app.core.Chunking.Chunk_Lib.AutoTokenizer.from_pretrained', MagicMock()):
            chunker = Chunker(options={"method": "words", "max_size": 10, "overlap": 15})
            # Should warn and adjust overlap, not crash
            result = chunker.chunk_text("This is a test text with many words for chunking properly")
            assert len(result) > 0  # Should produce chunks without infinite loop
            
            chunker = Chunker(options={"method": "words", "max_size": 10, "overlap": 10})
            # Equal overlap should also be handled
            result = chunker.chunk_text("This is a test text with many words for chunking properly")
            assert len(result) > 0
    
    def test_step_calculation_never_zero(self):
        """Test that step calculation never becomes zero or negative."""
        with patch('tldw_Server_API.app.core.Chunking.Chunk_Lib.AutoTokenizer.from_pretrained', MagicMock()):
            test_text = " ".join(["word"] * 100)
            
            # Test with max overlap
            chunker = Chunker(options={"method": "words", "max_size": 5, "overlap": 5})
            result = chunker.chunk_text(test_text)
            assert len(result) > 0
            
            # Test with overlap = max_size - 1
            chunker = Chunker(options={"method": "words", "max_size": 5, "overlap": 4})
            result = chunker.chunk_text(test_text)
            assert len(result) > 0
    
    def test_language_detection_caching(self):
        """Test that language detection is cached properly."""
        with patch('tldw_Server_API.app.core.Chunking.Chunk_Lib.AutoTokenizer.from_pretrained', MagicMock()):
            with patch('tldw_Server_API.app.core.Chunking.Chunk_Lib.detect') as mock_detect:
                mock_detect.return_value = 'en'
                
                chunker = Chunker()
                text = "This is test text for language detection caching."
                
                # First call should detect language
                lang1 = chunker.detect_language(text)
                assert mock_detect.call_count == 1
                assert lang1 == 'en'
                
                # Second call with same text should use cache
                lang2 = chunker.detect_language(text)
                assert mock_detect.call_count == 1  # Should not increase
                assert lang2 == 'en'
                
                # Different text should trigger new detection
                lang3 = chunker.detect_language("Different text entirely")
                assert mock_detect.call_count == 2
    
    def test_cache_size_limit(self):
        """Test that language cache respects size limits."""
        with patch('tldw_Server_API.app.core.Chunking.Chunk_Lib.AutoTokenizer.from_pretrained', MagicMock()):
            with patch('tldw_Server_API.app.core.Chunking.Chunk_Lib.detect') as mock_detect:
                mock_detect.return_value = 'en'
                
                chunker = Chunker()
                chunker._cache_size_limit = 3  # Set small limit for testing
                
                # Fill cache beyond limit
                for i in range(5):
                    chunker.detect_language(f"Text {i}")
                
                # Cache should not exceed limit
                assert len(chunker._language_cache) <= 3
    
    def test_generator_method_for_large_texts(self):
        """Test that the generator method works for memory-efficient processing."""
        with patch('tldw_Server_API.app.core.Chunking.Chunk_Lib.AutoTokenizer.from_pretrained', MagicMock()):
            chunker = Chunker(options={"method": "words", "max_size": 10, "overlap": 2})
            
            text = " ".join(["word"] * 1000)
            chunks = list(chunker.chunk_text_generator(text))
            
            assert len(chunks) > 0
            # Each chunk should have roughly max_size words
            for chunk in chunks[:-1]:  # All but last chunk
                word_count = len(chunk.split())
                assert word_count <= 10
    
    def test_invalid_json_handling(self):
        """Test proper error handling for invalid JSON."""
        with patch('tldw_Server_API.app.core.Chunking.Chunk_Lib.AutoTokenizer.from_pretrained', MagicMock()):
            chunker = Chunker()
            
            with pytest.raises(InvalidInputError, match="Invalid JSON data"):
                chunker._chunk_text_by_json("{invalid json}", max_size=10, overlap=0)
            
            with pytest.raises(InvalidInputError, match="Invalid JSON data"):
                chunker._chunk_text_by_json('{"unclosed": ', max_size=10, overlap=0)
    
    def test_tokenizer_failure_handling(self):
        """Test that tokenizer loading failure is handled gracefully."""
        with patch('tldw_Server_API.app.core.Chunking.Chunk_Lib.AutoTokenizer.from_pretrained', 
                  side_effect=Exception("Failed to load")):
            chunker = Chunker()
            assert chunker.tokenizer is None
            
            # Token-based methods should raise appropriate error
            with pytest.raises(ChunkingError, match="Tokenizer not loaded"):
                chunker.chunk_text("Some text", method="tokens")
    
    def test_concurrent_chunk_processing(self):
        """Test that chunking can handle concurrent requests safely."""
        import threading
        
        with patch('tldw_Server_API.app.core.Chunking.Chunk_Lib.AutoTokenizer.from_pretrained', MagicMock()):
            chunker = Chunker(options={"method": "words", "max_size": 10})
            
            results = []
            errors = []
            
            def chunk_text_thread(text, index):
                try:
                    result = chunker.chunk_text(text)
                    results.append((index, len(result)))
                except Exception as e:
                    errors.append((index, str(e)))
            
            # Create multiple threads
            threads = []
            for i in range(10):
                text = f"Thread {i} " * 50  # Different text for each thread
                t = threading.Thread(target=chunk_text_thread, args=(text, i))
                threads.append(t)
                t.start()
            
            # Wait for all threads to complete
            for t in threads:
                t.join()
            
            # All threads should complete without errors
            assert len(errors) == 0
            assert len(results) == 10

# Additional integration tests

def test_improved_chunking_process_with_edge_cases():
    """Test improved_chunking_process with various edge cases."""
    from tldw_Server_API.app.core.Chunking.Chunk_Lib import improved_chunking_process
    
    with patch('tldw_Server_API.app.core.Chunking.Chunk_Lib.AutoTokenizer.from_pretrained', MagicMock()):
        # Test with empty text
        result = improved_chunking_process("")
        assert result == []
        
        # Test with None options
        result = improved_chunking_process("Test text", chunk_options_dict=None)
        assert len(result) > 0
        
        # Test with invalid options that should be corrected
        options = {"max_size": "10", "overlap": "2"}  # String values should be converted
        result = improved_chunking_process("Test text for chunking", chunk_options_dict=options)
        assert len(result) > 0

def test_chunk_for_embedding_edge_cases():
    """Test chunk_for_embedding with edge cases."""
    from tldw_Server_API.app.core.Chunking.Chunk_Lib import chunk_for_embedding
    
    with patch('tldw_Server_API.app.core.Chunking.Chunk_Lib.AutoTokenizer.from_pretrained', MagicMock()):
        # Test with empty text
        result = chunk_for_embedding("", "test.txt")
        assert result == []
        
        # Test with very long filename
        long_filename = "a" * 1000 + ".txt"
        result = chunk_for_embedding("Test content", long_filename)
        assert len(result) > 0
        assert long_filename in result[0]['text_for_embedding']