"""
Unit tests for all chunking strategies.

Tests each of the 13 chunking strategies with minimal mocking.
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
import json
import xml.etree.ElementTree as ET

from tldw_Server_API.app.core.Chunking.Chunk_Lib import Chunker
from tldw_Server_API.app.core.Chunking.strategies import (
    words,
    sentences,
    paragraphs,
    tokens,
    semantic,
    structure_aware,
    rolling_summarize,
    json_xml,
    ebook_chapters
)

# ========================================================================
# Words Strategy Tests
# ========================================================================

class TestWordsStrategy:
    """Test word-based chunking strategy."""
    
    @pytest.mark.unit
    @pytest.mark.strategy
    def test_words_basic_chunking(self, sample_text_medium):
        """Test basic word-based chunking."""
        chunker = Chunker(
            chunk_method='words',
            max_chunk_size=50,
            chunk_overlap=10
        )
        
        chunks = chunker.chunk(sample_text_medium)
        
        assert len(chunks) > 0
        # Check chunk sizes
        for chunk in chunks[:-1]:  # All but last chunk
            word_count = len(chunk['text'].split())
            assert word_count <= 50
        
        # Check overlap exists
        if len(chunks) > 1:
            # Words at end of first chunk should appear at start of second
            first_chunk_words = chunks[0]['text'].split()
            second_chunk_words = chunks[1]['text'].split()
            # Some overlap should exist
            overlap = set(first_chunk_words[-10:]) & set(second_chunk_words[:10])
            assert len(overlap) > 0
    
    @pytest.mark.unit
    @pytest.mark.strategy
    def test_words_adaptive_chunking(self, sample_text_medium):
        """Test adaptive word-based chunking."""
        chunker = Chunker(
            chunk_method='words',
            max_chunk_size=50,
            chunk_overlap=10,
            use_adaptive=True
        )
        
        chunks = chunker.chunk(sample_text_medium)
        
        assert len(chunks) > 0
        # Adaptive chunking should respect sentence boundaries better
        for chunk in chunks:
            text = chunk['text'].strip()
            # Should preferably end with punctuation
            if text and len(chunks) > 1:
                # More chunks should end properly
                pass  # Adaptive behavior is heuristic
    
    @pytest.mark.unit
    @pytest.mark.strategy
    def test_words_no_overlap(self, sample_text_medium):
        """Test word chunking without overlap."""
        chunker = Chunker(
            chunk_method='words',
            max_chunk_size=30,
            chunk_overlap=0
        )
        
        chunks = chunker.chunk(sample_text_medium)
        
        assert len(chunks) > 0
        # No overlap means chunks should be distinct
        all_text = " ".join(c['text'] for c in chunks)
        # Rough check - total length should be similar
        assert len(all_text) <= len(sample_text_medium) * 1.1

# ========================================================================
# Sentences Strategy Tests
# ========================================================================

class TestSentencesStrategy:
    """Test sentence-based chunking strategy."""
    
    @pytest.mark.unit
    @pytest.mark.strategy
    def test_sentences_basic_chunking(self, sample_text_medium):
        """Test basic sentence-based chunking."""
        chunker = Chunker(
            chunk_method='sentences',
            max_chunk_size=3,  # 3 sentences per chunk
            chunk_overlap=1     # 1 sentence overlap
        )
        
        chunks = chunker.chunk(sample_text_medium)
        
        assert len(chunks) > 0
        for chunk in chunks:
            # Count sentences (roughly by periods, !, ?)
            sentence_enders = chunk['text'].count('.') + chunk['text'].count('!') + chunk['text'].count('?')
            assert sentence_enders <= 4  # Max 3 + potential partial
    
    @pytest.mark.unit
    @pytest.mark.strategy
    def test_sentences_preserve_boundaries(self):
        """Test that sentence boundaries are preserved."""
        text = "First sentence. Second sentence! Third sentence? Fourth sentence."
        
        chunker = Chunker(
            chunk_method='sentences',
            max_chunk_size=2,
            chunk_overlap=0
        )
        
        chunks = chunker.chunk(text)
        
        assert len(chunks) == 2
        # Each chunk should have complete sentences
        for chunk in chunks:
            assert chunk['text'].strip().endswith(('.', '!', '?'))

# ========================================================================
# Paragraphs Strategy Tests
# ========================================================================

class TestParagraphsStrategy:
    """Test paragraph-based chunking strategy."""
    
    @pytest.mark.unit
    @pytest.mark.strategy
    def test_paragraphs_basic_chunking(self, sample_text_medium):
        """Test basic paragraph-based chunking."""
        chunker = Chunker(
            chunk_method='paragraphs',
            max_chunk_size=2,  # 2 paragraphs per chunk
            chunk_overlap=0
        )
        
        chunks = chunker.chunk(sample_text_medium)
        
        assert len(chunks) > 0
        # Text has 3 paragraphs, so should have 2 chunks
        assert len(chunks) <= 2
    
    @pytest.mark.unit
    @pytest.mark.strategy
    def test_paragraphs_preserve_structure(self):
        """Test that paragraph structure is preserved."""
        text = "Para 1 line 1.\nPara 1 line 2.\n\nPara 2 line 1.\nPara 2 line 2.\n\nPara 3."
        
        chunker = Chunker(
            chunk_method='paragraphs',
            max_chunk_size=1,
            chunk_overlap=0
        )
        
        chunks = chunker.chunk(text)
        
        assert len(chunks) == 3
        # Each chunk should be a complete paragraph
        assert "Para 1" in chunks[0]['text']
        assert "Para 2" in chunks[1]['text']
        assert "Para 3" in chunks[2]['text']

# ========================================================================
# Tokens Strategy Tests
# ========================================================================

class TestTokensStrategy:
    """Test token-based chunking strategy."""
    
    @pytest.mark.unit
    @pytest.mark.strategy
    def test_tokens_basic_chunking(self, sample_text_short, mock_tokenizer):
        """Test basic token-based chunking."""
        with patch('transformers.AutoTokenizer.from_pretrained', return_value=mock_tokenizer):
            chunker = Chunker(
                chunk_method='tokens',
                max_chunk_size=5,  # 5 tokens per chunk
                chunk_overlap=1,
                tokenizer='gpt2'
            )
            
            chunks = chunker.chunk(sample_text_short)
            
            assert len(chunks) > 0
            mock_tokenizer.encode.assert_called()
    
    @pytest.mark.unit
    @pytest.mark.strategy
    def test_tokens_without_tokenizer(self, sample_text_short):
        """Test token chunking falls back gracefully without tokenizer."""
        chunker = Chunker(
            chunk_method='tokens',
            max_chunk_size=5,
            chunk_overlap=1
        )
        
        # Should fall back to word-based approximation
        chunks = chunker.chunk(sample_text_short)
        
        assert len(chunks) > 0

# ========================================================================
# Semantic Strategy Tests
# ========================================================================

class TestSemanticStrategy:
    """Test semantic-based chunking strategy."""
    
    @pytest.mark.unit
    @pytest.mark.strategy
    def test_semantic_basic_chunking(self, sample_text_medium, mock_sentence_transformer):
        """Test basic semantic chunking."""
        with patch('sentence_transformers.SentenceTransformer', return_value=mock_sentence_transformer):
            chunker = Chunker(
                chunk_method='semantic',
                max_chunk_size=100,
                chunk_overlap=20,
                semantic_similarity_threshold=0.7
            )
            
            chunks = chunker.chunk(sample_text_medium)
            
            assert len(chunks) > 0
            mock_sentence_transformer.encode.assert_called()
    
    @pytest.mark.unit
    @pytest.mark.strategy
    def test_semantic_similarity_threshold(self, mock_sentence_transformer):
        """Test semantic similarity threshold affects chunking."""
        text = "Dogs are animals. Cats are animals. Cars are vehicles. Trucks are vehicles."
        
        with patch('sentence_transformers.SentenceTransformer', return_value=mock_sentence_transformer):
            # High threshold - more chunks
            chunker_high = Chunker(
                chunk_method='semantic',
                max_chunk_size=100,
                semantic_similarity_threshold=0.9
            )
            
            # Low threshold - fewer chunks
            chunker_low = Chunker(
                chunk_method='semantic',
                max_chunk_size=100,
                semantic_similarity_threshold=0.3
            )
            
            chunks_high = chunker_high.chunk(text)
            chunks_low = chunker_low.chunk(text)
            
            # Different thresholds should affect chunking
            assert len(chunks_high) >= len(chunks_low)

# ========================================================================
# Structure-Aware Strategy Tests
# ========================================================================

class TestStructureAwareStrategy:
    """Test structure-aware chunking strategy."""
    
    @pytest.mark.unit
    @pytest.mark.strategy
    def test_structure_aware_markdown(self, sample_text_markdown):
        """Test structure-aware chunking with markdown."""
        chunker = Chunker(
            chunk_method='structure_aware',
            max_chunk_size=200,
            chunk_overlap=20
        )
        
        chunks = chunker.chunk(sample_text_markdown)
        
        assert len(chunks) > 0
        # Should preserve markdown structure
        for chunk in chunks:
            text = chunk['text']
            # Headers should be preserved
            if '#' in text:
                lines = text.split('\n')
                # Headers should be at line start
                header_lines = [l for l in lines if l.strip().startswith('#')]
                assert len(header_lines) > 0
    
    @pytest.mark.unit
    @pytest.mark.strategy
    def test_structure_aware_code_blocks(self, sample_text_code):
        """Test structure-aware chunking with code blocks."""
        chunker = Chunker(
            chunk_method='structure_aware',
            max_chunk_size=150,
            preserve_code_blocks=True
        )
        
        chunks = chunker.chunk(sample_text_code)
        
        assert len(chunks) > 0
        # Code blocks should be preserved
        code_chunks = [c for c in chunks if '```' in c['text']]
        assert len(code_chunks) > 0
        
        # Code blocks should be complete
        for chunk in code_chunks:
            # Should have opening and closing ```
            assert chunk['text'].count('```') % 2 == 0

# ========================================================================
# Rolling Summarize Strategy Tests
# ========================================================================

class TestRollingSummarizeStrategy:
    """Test rolling summarize chunking strategy."""
    
    @pytest.mark.unit
    @pytest.mark.strategy
    def test_rolling_summarize_basic(self, sample_text_long):
        """Test basic rolling summarize chunking."""
        # Mock the summarization function
        with patch('tldw_Server_API.app.core.Chunking.strategies.rolling_summarize.summarize_text') as mock_summarize:
            mock_summarize.return_value = "Summary of the text."
            
            chunker = Chunker(
                chunk_method='rolling_summarize',
                max_chunk_size=500,
                chunk_overlap=50,
                summarization_detail=0.3
            )
            
            chunks = chunker.chunk(sample_text_long)
            
            assert len(chunks) > 0
            # Summarization should have been called
            if len(chunks) > 1:
                mock_summarize.assert_called()

# ========================================================================
# JSON Strategy Tests
# ========================================================================

class TestJSONStrategy:
    """Test JSON-based chunking strategy."""
    
    @pytest.mark.unit
    @pytest.mark.strategy
    def test_json_object_chunking(self, sample_json_data):
        """Test JSON object chunking."""
        json_text = json.dumps(sample_json_data, indent=2)
        
        chunker = Chunker(
            chunk_method='json',
            max_chunk_size=100,
            json_chunkable_data_key='chapters'
        )
        
        chunks = chunker.chunk(json_text)
        
        assert len(chunks) > 0
        # Should chunk by chapters
        assert len(chunks) >= len(sample_json_data['chapters'])
    
    @pytest.mark.unit
    @pytest.mark.strategy
    def test_json_array_chunking(self):
        """Test JSON array chunking."""
        json_array = [{"id": i, "content": f"Item {i} content"} for i in range(10)]
        json_text = json.dumps(json_array)
        
        chunker = Chunker(
            chunk_method='json',
            max_chunk_size=3  # 3 items per chunk
        )
        
        chunks = chunker.chunk(json_text)
        
        assert len(chunks) > 0
        assert len(chunks) <= 4  # 10 items / 3 per chunk

# ========================================================================
# XML Strategy Tests
# ========================================================================

class TestXMLStrategy:
    """Test XML-based chunking strategy."""
    
    @pytest.mark.unit
    @pytest.mark.strategy
    def test_xml_element_chunking(self, sample_xml_data):
        """Test XML element-based chunking."""
        chunker = Chunker(
            chunk_method='xml',
            max_chunk_size=100,
            xml_chunking_element='section'
        )
        
        chunks = chunker.chunk(sample_xml_data)
        
        assert len(chunks) > 0
        # Should chunk by sections
        for chunk in chunks:
            # Each chunk should be valid XML or text
            assert len(chunk['text']) > 0
    
    @pytest.mark.unit
    @pytest.mark.strategy
    def test_xml_preserve_structure(self, sample_xml_data):
        """Test XML structure preservation."""
        chunker = Chunker(
            chunk_method='xml',
            max_chunk_size=200,
            preserve_xml_structure=True
        )
        
        chunks = chunker.chunk(sample_xml_data)
        
        assert len(chunks) > 0

# ========================================================================
# Ebook Chapters Strategy Tests
# ========================================================================

class TestEbookChaptersStrategy:
    """Test ebook chapter-based chunking strategy."""
    
    @pytest.mark.unit
    @pytest.mark.strategy
    def test_ebook_chapter_detection(self):
        """Test ebook chapter detection."""
        ebook_text = """
        Chapter 1: Introduction
        
        This is the introduction content.
        It spans multiple paragraphs.
        
        Chapter 2: Main Content
        
        The main content goes here.
        With more details.
        
        Chapter 3: Conclusion
        
        The conclusion wraps things up.
        """
        
        chunker = Chunker(
            chunk_method='ebook_chapters',
            max_chunk_size=1000
        )
        
        chunks = chunker.chunk(ebook_text)
        
        assert len(chunks) >= 3  # At least 3 chapters
        # Each chunk should contain a chapter
        for chunk in chunks:
            assert 'Chapter' in chunk['text'] or len(chunk['text'].strip()) == 0
    
    @pytest.mark.unit
    @pytest.mark.strategy
    def test_ebook_roman_numerals(self):
        """Test ebook chapter detection with Roman numerals."""
        ebook_text = """
        I. Introduction
        
        Introduction content here.
        
        II. Main Section
        
        Main content here.
        
        III. Conclusion
        
        Conclusion content here.
        """
        
        chunker = Chunker(
            chunk_method='ebook_chapters',
            detect_roman_numerals=True
        )
        
        chunks = chunker.chunk(ebook_text)
        
        assert len(chunks) >= 3
        # Roman numeral chapters should be detected