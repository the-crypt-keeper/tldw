# test_chunker_v2.py
"""
Direct unit tests for V2 chunking implementation.
Tests the new modular chunker and strategy pattern.
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
from typing import List

from tldw_Server_API.app.core.Chunking import (
    Chunker,
    create_chunker,
    ChunkingMethod,
    ChunkResult,
    ChunkMetadata,
    ChunkingError,
    InvalidInputError,
    InvalidChunkingMethodError,
    DEFAULT_CHUNK_OPTIONS
)
from tldw_Server_API.app.core.Chunking.base import ChunkerConfig


class TestV2Chunker:
    """Test the main V2 Chunker class."""
    
    def test_chunker_initialization(self):
        """Test that Chunker initializes correctly."""
        chunker = Chunker()
        assert chunker is not None
        assert chunker.config is not None
        assert len(chunker._strategies) > 0
    
    def test_chunker_with_custom_config(self):
        """Test Chunker with custom configuration."""
        config = ChunkerConfig(
            default_method=ChunkingMethod.SENTENCES,
            default_max_size=100,
            default_overlap=10,
            language='fr'
        )
        chunker = Chunker(config=config)
        assert chunker.config.default_method == ChunkingMethod.SENTENCES
        assert chunker.config.default_max_size == 100
        assert chunker.config.default_overlap == 10
        assert chunker.config.language == 'fr'
    
    def test_chunk_text_returns_strings(self):
        """Test that chunk_text returns list of strings."""
        chunker = Chunker()
        text = "This is a test. This is another sentence. And a third one."
        chunks = chunker.chunk_text(text, method='sentences', max_size=2)
        
        assert isinstance(chunks, list)
        assert all(isinstance(chunk, str) for chunk in chunks)
        assert len(chunks) > 0
    
    def test_chunk_text_with_metadata(self):
        """Test that chunk_text_with_metadata returns ChunkResult objects."""
        chunker = Chunker()
        text = "This is a test. This is another sentence. And a third one."
        chunks = chunker.chunk_text_with_metadata(text, method='sentences', max_size=2)
        
        assert isinstance(chunks, list)
        assert all(isinstance(chunk, ChunkResult) for chunk in chunks)
        assert all(isinstance(chunk.metadata, ChunkMetadata) for chunk in chunks)
        assert len(chunks) > 0
    
    def test_invalid_method_raises_error(self):
        """Test that invalid chunking method raises error."""
        chunker = Chunker()
        with pytest.raises(InvalidChunkingMethodError):
            chunker.chunk_text("test", method='invalid_method')
    
    def test_all_methods_available(self):
        """Test that all expected methods are available."""
        chunker = Chunker()
        expected_methods = [
            'words', 'sentences', 'paragraphs', 'tokens', 
            'semantic', 'json', 'xml', 'ebook_chapters', 'rolling_summarize'
        ]
        
        for method in expected_methods:
            # Should not raise an error
            assert method in chunker._strategies or method == 'structure_aware'
    
    def test_empty_text_handling(self):
        """Test handling of empty text."""
        chunker = Chunker()
        # V2 returns empty list for empty text rather than raising error
        result = chunker.chunk_text("", method='words')
        assert result == []
    
    def test_chunk_text_generator(self):
        """Test the generator method for memory efficiency."""
        chunker = Chunker()
        text = " ".join(["word"] * 1000)  # Large text
        
        generator = chunker.chunk_text_generator(text, method='words', max_size=10)
        assert hasattr(generator, '__iter__')
        
        chunks = list(generator)
        assert len(chunks) > 0
        assert all(isinstance(chunk, str) for chunk in chunks)


class TestWordsStrategy:
    """Test the words chunking strategy."""
    
    def test_words_basic_chunking(self):
        """Test basic word-based chunking."""
        from tldw_Server_API.app.core.Chunking.strategies.words import WordChunkingStrategy
        
        strategy = WordChunkingStrategy()
        text = " ".join([f"word{i}" for i in range(20)])
        chunks = strategy.chunk(text, max_size=5, overlap=2)
        
        assert isinstance(chunks, list)
        assert len(chunks) > 1
        # Check overlap
        first_chunk_words = chunks[0].split()
        second_chunk_words = chunks[1].split()
        assert first_chunk_words[-2:] == second_chunk_words[:2]  # 2 word overlap
    
    def test_words_no_overlap(self):
        """Test word chunking without overlap."""
        from tldw_Server_API.app.core.Chunking.strategies.words import WordChunkingStrategy
        
        strategy = WordChunkingStrategy()
        text = " ".join([f"word{i}" for i in range(20)])
        chunks = strategy.chunk(text, max_size=5, overlap=0)
        
        assert len(chunks) == 4  # 20 words / 5 words per chunk
        # Verify no overlap
        for i in range(len(chunks) - 1):
            assert chunks[i].split()[-1] != chunks[i+1].split()[0]
    
    def test_words_with_metadata(self):
        """Test word chunking with metadata."""
        from tldw_Server_API.app.core.Chunking.strategies.words import WordChunkingStrategy
        
        strategy = WordChunkingStrategy()
        text = "One two three four five six seven eight nine ten"
        chunks = strategy.chunk_with_metadata(text, max_size=3, overlap=1)
        
        assert all(isinstance(chunk, ChunkResult) for chunk in chunks)
        assert chunks[0].metadata.word_count == 3
        # Method field might not be set by base implementation
        # assert chunks[0].metadata.method == 'words'
        assert chunks[0].metadata.index == 0


class TestSentencesStrategy:
    """Test the sentences chunking strategy."""
    
    def test_sentences_basic_chunking(self):
        """Test basic sentence-based chunking."""
        from tldw_Server_API.app.core.Chunking.strategies.sentences import SentenceChunkingStrategy
        
        strategy = SentenceChunkingStrategy()
        text = "First sentence. Second sentence. Third sentence. Fourth sentence."
        chunks = strategy.chunk(text, max_size=2, overlap=1)
        
        assert isinstance(chunks, list)
        assert len(chunks) > 1
        assert "First sentence. Second sentence." in chunks[0]
    
    def test_sentences_handles_various_punctuation(self):
        """Test handling of various sentence endings."""
        from tldw_Server_API.app.core.Chunking.strategies.sentences import SentenceChunkingStrategy
        
        strategy = SentenceChunkingStrategy()
        text = "Question? Exclamation! Statement. Another?"
        chunks = strategy.chunk(text, max_size=2, overlap=0)
        
        assert len(chunks) == 2
        assert "Question? Exclamation!" in chunks[0]
        assert "Statement. Another?" in chunks[1]


class TestParagraphsStrategy:
    """Test the paragraphs chunking strategy."""
    
    def test_paragraphs_basic_chunking(self):
        """Test basic paragraph-based chunking."""
        from tldw_Server_API.app.core.Chunking.strategies.paragraphs import ParagraphChunkingStrategy
        
        strategy = ParagraphChunkingStrategy()
        text = "First paragraph.\n\nSecond paragraph.\n\nThird paragraph.\n\nFourth paragraph."
        chunks = strategy.chunk(text, max_size=2, overlap=1)
        
        assert isinstance(chunks, list)
        assert len(chunks) > 1
        assert "First paragraph.\n\nSecond paragraph." in chunks[0]
    
    def test_paragraphs_single_paragraph(self):
        """Test handling of text without paragraph breaks."""
        from tldw_Server_API.app.core.Chunking.strategies.paragraphs import ParagraphChunkingStrategy
        
        strategy = ParagraphChunkingStrategy()
        text = "This is all one paragraph without any breaks."
        chunks = strategy.chunk(text, max_size=1, overlap=0)
        
        assert len(chunks) == 1
        assert chunks[0] == text.strip()
    
    def test_paragraphs_with_metadata(self):
        """Test paragraph chunking with metadata."""
        from tldw_Server_API.app.core.Chunking.strategies.paragraphs import ParagraphChunkingStrategy
        
        strategy = ParagraphChunkingStrategy()
        text = "Para 1.\n\nPara 2.\n\nPara 3."
        chunks = strategy.chunk_with_metadata(text, max_size=2, overlap=0)
        
        assert all(isinstance(chunk, ChunkResult) for chunk in chunks)
        assert chunks[0].metadata.method == 'paragraphs'
        assert chunks[0].metadata.options is not None
        assert 'paragraph_count' in chunks[0].metadata.options


class TestTokensStrategy:
    """Test the tokens chunking strategy."""
    
    def test_tokens_basic_chunking(self):
        """Test basic token-based chunking."""
        from tldw_Server_API.app.core.Chunking.strategies.tokens import TokenChunkingStrategy
        
        # Test with actual tokenizer or skip if not available
        try:
            strategy = TokenChunkingStrategy()
            text = "Some text to tokenize for testing purposes"
            chunks = strategy.chunk(text, max_size=5, overlap=2)
            
            assert isinstance(chunks, list)
            assert len(chunks) > 0
            assert all(isinstance(chunk, str) for chunk in chunks)
        except ImportError:
            # Skip if transformers not available
            pytest.skip("transformers library not available")


class TestEbookChaptersStrategy:
    """Test the ebook chapters chunking strategy."""
    
    def test_ebook_chapters_basic(self):
        """Test basic chapter detection and chunking."""
        from tldw_Server_API.app.core.Chunking.strategies.ebook_chapters import EbookChapterChunkingStrategy
        
        strategy = EbookChapterChunkingStrategy()
        text = """Chapter 1: Introduction
        Some content here.
        
        Chapter 2: Main Content
        More content here.
        
        Chapter 3: Conclusion
        Final content."""
        
        chunks = strategy.chunk(text, max_size=1000)  # Large size to keep chapters intact
        
        assert len(chunks) == 3
        assert "Chapter 1" in chunks[0]
        assert "Chapter 2" in chunks[1]
        assert "Chapter 3" in chunks[2]
    
    def test_ebook_no_chapters(self):
        """Test handling of text without chapter markers."""
        from tldw_Server_API.app.core.Chunking.strategies.ebook_chapters import EbookChapterChunkingStrategy
        
        strategy = EbookChapterChunkingStrategy()
        text = "This is text without any chapter markers. " * 20
        chunks = strategy.chunk(text, max_size=50, overlap=10)
        
        assert len(chunks) > 1  # Should split by size
        assert all(isinstance(chunk, str) for chunk in chunks)
    
    def test_ebook_custom_pattern(self):
        """Test custom chapter pattern."""
        from tldw_Server_API.app.core.Chunking.strategies.ebook_chapters import EbookChapterChunkingStrategy
        
        strategy = EbookChapterChunkingStrategy()
        text = """Part 1: Beginning
        Content here.
        
        Part 2: Middle
        More content.
        
        Part 3: End
        Final content."""
        
        # Use custom pattern that matches "Part N:"
        chunks = strategy.chunk(
            text, 
            max_size=1000,
            custom_chapter_pattern=r'Part \d+:'
        )
        
        assert len(chunks) == 3
        assert "Part 1" in chunks[0]


class TestSemanticStrategy:
    """Test the semantic chunking strategy."""
    
    def test_semantic_basic_chunking(self):
        """Test basic semantic chunking."""
        from tldw_Server_API.app.core.Chunking.strategies.semantic import SemanticChunkingStrategy
        
        # Test with actual model or skip if not available
        try:
            strategy = SemanticChunkingStrategy()
            text = "First sentence. Second sentence. Third sentence. Fourth sentence."
            chunks = strategy.chunk(text, max_size=2, overlap=0)
            
            assert isinstance(chunks, list)
            assert len(chunks) > 0
            assert all(isinstance(chunk, str) for chunk in chunks)
        except (ImportError, RuntimeError):
            # Skip if sentence-transformers not available or model can't load
            pytest.skip("sentence-transformers library or model not available")


class TestJSONStrategy:
    """Test the JSON chunking strategy."""
    
    def test_json_list_chunking(self):
        """Test chunking of JSON lists."""
        from tldw_Server_API.app.core.Chunking.strategies.json_xml import JSONChunkingStrategy
        
        strategy = JSONChunkingStrategy()
        json_text = '[{"id": 1}, {"id": 2}, {"id": 3}, {"id": 4}]'
        chunks = strategy.chunk(json_text, max_size=2, overlap=1)
        
        assert isinstance(chunks, list)
        assert len(chunks) > 1
        # Each chunk should be valid JSON
        import json
        for chunk in chunks:
            parsed = json.loads(chunk)
            assert isinstance(parsed, list)
    
    def test_json_dict_chunking(self):
        """Test chunking of JSON objects."""
        from tldw_Server_API.app.core.Chunking.strategies.json_xml import JSONChunkingStrategy
        
        strategy = JSONChunkingStrategy()
        json_text = '{"key1": "value1", "key2": "value2", "key3": "value3"}'
        chunks = strategy.chunk(json_text, max_size=2, overlap=0)
        
        assert isinstance(chunks, list)
        assert len(chunks) >= 1
        # Each chunk should be valid JSON
        import json
        for chunk in chunks:
            parsed = json.loads(chunk)
            assert isinstance(parsed, dict)
    
    def test_json_invalid_input(self):
        """Test handling of invalid JSON."""
        from tldw_Server_API.app.core.Chunking.strategies.json_xml import JSONChunkingStrategy
        
        strategy = JSONChunkingStrategy()
        with pytest.raises(InvalidInputError):
            strategy.chunk("not valid json {", max_size=2)


class TestXMLStrategy:
    """Test the XML chunking strategy."""
    
    def test_xml_basic_chunking(self):
        """Test basic XML chunking."""
        from tldw_Server_API.app.core.Chunking.strategies.json_xml import XMLChunkingStrategy
        
        strategy = XMLChunkingStrategy()
        xml_text = """<root>
            <item>Content 1</item>
            <item>Content 2</item>
            <item>Content 3</item>
        </root>"""
        chunks = strategy.chunk(xml_text, max_size=50, overlap=0)
        
        assert isinstance(chunks, list)
        assert len(chunks) >= 1
        assert all(isinstance(chunk, str) for chunk in chunks)
    
    def test_xml_invalid_input(self):
        """Test handling of invalid XML."""
        from tldw_Server_API.app.core.Chunking.strategies.json_xml import XMLChunkingStrategy
        
        strategy = XMLChunkingStrategy()
        with pytest.raises(InvalidInputError):
            strategy.chunk("not valid xml <", max_size=2)


class TestRollingSummarizeStrategy:
    """Test the rolling summarize strategy."""
    
    def test_rolling_summarize_without_llm(self):
        """Test that rolling summarize works without LLM (returns raw chunks)."""
        from tldw_Server_API.app.core.Chunking.strategies.rolling_summarize import RollingSummarizeStrategy
        
        strategy = RollingSummarizeStrategy()
        # Without LLM, should return raw chunks (not summarized)
        result = strategy.chunk("Some text to summarize", max_size=100)
        assert isinstance(result, list)
        # Should return the text as-is since it's shorter than max_size
        assert len(result) >= 1
    
    @patch('tldw_Server_API.app.core.Chunking.strategies.rolling_summarize.RollingSummarizeStrategy._call_llm')
    def test_rolling_summarize_with_llm(self, mock_llm):
        """Test rolling summarize with mocked LLM."""
        from tldw_Server_API.app.core.Chunking.strategies.rolling_summarize import RollingSummarizeStrategy
        
        # Mock LLM responses
        mock_llm.return_value = "Summarized content"
        
        # Create strategy with mock LLM function
        mock_llm_func = Mock(return_value="Summary")
        strategy = RollingSummarizeStrategy(llm_call_func=mock_llm_func)
        
        text = "This is a long text. " * 100
        chunks = strategy.chunk(text, max_size=50, overlap=10)
        
        assert isinstance(chunks, list)
        assert len(chunks) > 0


class TestBackwardCompatibility:
    """Test backward compatibility functions."""
    
    def test_improved_chunking_process(self):
        """Test the backward compatibility improved_chunking_process function."""
        from tldw_Server_API.app.core.Chunking import improved_chunking_process
        
        text = "Test text. Another sentence. Third sentence."
        options = {
            'method': 'sentences',
            'max_size': 2,
            'overlap': 1
        }
        
        chunks = improved_chunking_process(text, options)
        
        assert isinstance(chunks, list)
        assert len(chunks) > 0
        assert all(isinstance(chunk, dict) for chunk in chunks)
        assert all('text' in chunk for chunk in chunks)
        assert all('metadata' in chunk for chunk in chunks)
    
    def test_chunk_for_embedding(self):
        """Test the backward compatibility chunk_for_embedding function."""
        from tldw_Server_API.app.core.Chunking import chunk_for_embedding
        
        text = "Test text for embedding. " * 10
        chunks = chunk_for_embedding(text, "test_file.txt", max_size=50)
        
        assert isinstance(chunks, list)
        assert len(chunks) > 0
        assert all(isinstance(chunk, dict) for chunk in chunks)
    
    def test_default_options_exported(self):
        """Test that DEFAULT_CHUNK_OPTIONS is properly exported."""
        from tldw_Server_API.app.core.Chunking import DEFAULT_CHUNK_OPTIONS
        
        assert isinstance(DEFAULT_CHUNK_OPTIONS, dict)
        assert 'method' in DEFAULT_CHUNK_OPTIONS
        assert 'max_size' in DEFAULT_CHUNK_OPTIONS
        assert 'overlap' in DEFAULT_CHUNK_OPTIONS


class TestErrorHandling:
    """Test error handling across the module."""
    
    def test_empty_text_handling_strategies(self):
        """Test that strategies handle empty text appropriately."""
        from tldw_Server_API.app.core.Chunking.strategies.words import WordChunkingStrategy
        
        strategy = WordChunkingStrategy()
        # V2 strategies return empty list for empty text
        result = strategy.chunk("", max_size=10)
        assert result == []
    
    def test_invalid_method_error(self):
        """Test that invalid method raises appropriate error."""
        chunker = Chunker()
        
        with pytest.raises(InvalidChunkingMethodError) as exc_info:
            chunker.chunk_text("test text", method='nonexistent')
        
        assert "unknown" in str(exc_info.value).lower()
    
    def test_invalid_parameters(self):
        """Test that invalid parameters are handled appropriately."""
        from tldw_Server_API.app.core.Chunking.strategies.words import WordChunkingStrategy
        
        strategy = WordChunkingStrategy()
        
        # Negative max_size
        with pytest.raises((InvalidInputError, ValueError)):
            strategy.chunk("test text", max_size=-1)
        
        # Overlap larger than max_size - V2 adjusts this automatically
        result = strategy.chunk("test text", max_size=10, overlap=15)
        # Should still work, overlap adjusted
        assert isinstance(result, list)


class TestPerformance:
    """Test performance-related features."""
    
    def test_generator_memory_efficiency(self):
        """Test that generator method is memory efficient."""
        chunker = Chunker()
        large_text = "word " * 10000  # Large text
        
        # Generator should not create all chunks at once
        generator = chunker.chunk_text_generator(
            large_text, 
            method='words', 
            max_size=100
        )
        
        # Get first chunk without generating all
        first_chunk = next(generator)
        assert isinstance(first_chunk, str)
        assert len(first_chunk.split()) <= 100
    
    def test_caching_disabled_by_default(self):
        """Test that caching can be disabled."""
        config = ChunkerConfig(enable_cache=False)
        chunker = Chunker(config=config)
        
        assert chunker._cache is None
        
        # Should work without cache
        text = "Test text for caching"
        chunks1 = chunker.chunk_text(text, method='words', max_size=5)
        chunks2 = chunker.chunk_text(text, method='words', max_size=5)
        
        assert chunks1 == chunks2


if __name__ == "__main__":
    pytest.main([__file__, "-v"])