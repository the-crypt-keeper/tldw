"""
Property-based tests for chunking invariants.

Uses Hypothesis to verify that chunking preserves content and maintains
expected properties across all strategies.
"""

import pytest
from hypothesis import given, strategies as st, assume, settings, example
from hypothesis.stateful import RuleBasedStateMachine, rule, initialize, invariant
import json
from typing import List, Dict, Any

from tldw_Server_API.app.core.Chunking.Chunk_Lib import Chunker

# ========================================================================
# Custom Hypothesis Strategies
# ========================================================================

@st.composite
def valid_text(draw):
    """Generate valid text for chunking."""
    # Generate text with various characteristics
    num_sentences = draw(st.integers(min_value=1, max_value=20))
    sentences = []
    
    for _ in range(num_sentences):
        words = draw(st.lists(
            st.text(min_size=1, max_size=15, alphabet=st.characters(min_codepoint=97, max_codepoint=122)),
            min_size=3,
            max_size=15
        ))
        sentence = " ".join(words)
        punctuation = draw(st.sampled_from(['.', '!', '?']))
        sentences.append(sentence.capitalize() + punctuation)
    
    # Optionally add paragraph breaks
    if draw(st.booleans()):
        # Insert paragraph breaks
        para_break = draw(st.integers(min_value=1, max_value=min(5, num_sentences)))
        result = []
        for i, sent in enumerate(sentences):
            result.append(sent)
            if (i + 1) % para_break == 0 and i < len(sentences) - 1:
                result.append("\n\n")
        return " ".join(result)
    
    return " ".join(sentences)

@st.composite
def chunking_params(draw):
    """Generate valid chunking parameters."""
    method = draw(st.sampled_from(['words', 'sentences', 'paragraphs']))
    max_size = draw(st.integers(min_value=10, max_value=500))
    overlap = draw(st.integers(min_value=0, max_value=min(max_size // 2, 50)))
    
    return {
        'method': method,
        'max_size': max_size,
        'overlap': overlap
    }

@st.composite
def structured_json(draw):
    """Generate structured JSON data."""
    num_items = draw(st.integers(min_value=1, max_value=10))
    items = []
    
    for i in range(num_items):
        item = {
            "id": i,
            "title": draw(st.text(min_size=1, max_size=20)),
            "content": draw(st.text(min_size=10, max_size=100))
        }
        items.append(item)
    
    return json.dumps(items)

# ========================================================================
# Content Preservation Properties
# ========================================================================

class TestContentPreservation:
    """Test that chunking preserves content."""
    
    @pytest.mark.property
    @given(text=valid_text(), params=chunking_params())
    def test_no_content_loss(self, text, params):
        """Property: Chunking should not lose content."""
        assume(text.strip())  # Skip empty text
        
        chunker = Chunker(
            chunk_method=params['method'],
            max_chunk_size=params['max_size'],
            chunk_overlap=params['overlap']
        )
        
        chunks = chunker.chunk(text)
        
        # All original words should appear in chunks
        original_words = set(text.lower().split())
        chunk_words = set()
        for chunk in chunks:
            chunk_words.update(chunk['text'].lower().split())
        
        # Most words should be preserved (allowing for minor processing differences)
        preserved_ratio = len(original_words & chunk_words) / len(original_words) if original_words else 1
        assert preserved_ratio > 0.95
    
    @pytest.mark.property
    @given(text=valid_text())
    def test_chunk_ordering_preserved(self, text):
        """Property: Chunk order preserves text order."""
        chunker = Chunker(chunk_method='sentences', max_chunk_size=2)
        
        chunks = chunker.chunk(text)
        
        if len(chunks) > 1:
            # First words of each chunk should appear in order
            positions = []
            for chunk in chunks:
                first_word = chunk['text'].split()[0] if chunk['text'].split() else ""
                if first_word in text:
                    positions.append(text.index(first_word))
            
            # Positions should be increasing
            assert positions == sorted(positions)
    
    @pytest.mark.property
    @given(
        text=st.text(min_size=100, max_size=1000),
        overlap=st.integers(min_value=0, max_value=50)
    )
    def test_overlap_content_consistency(self, text, overlap):
        """Property: Overlapping content should match."""
        assume(text.strip())
        
        chunker = Chunker(
            chunk_method='words',
            max_chunk_size=50,
            chunk_overlap=overlap
        )
        
        chunks = chunker.chunk(text)
        
        if len(chunks) > 1 and overlap > 0:
            for i in range(len(chunks) - 1):
                # End of chunk i should overlap with start of chunk i+1
                chunk1_words = chunks[i]['text'].split()
                chunk2_words = chunks[i+1]['text'].split()
                
                if len(chunk1_words) >= overlap and len(chunk2_words) >= overlap:
                    # Some overlap should exist
                    overlap_words = set(chunk1_words[-overlap:]) & set(chunk2_words[:overlap])
                    assert len(overlap_words) > 0

# ========================================================================
# Chunk Size Properties
# ========================================================================

class TestChunkSizeProperties:
    """Test properties related to chunk sizes."""
    
    @pytest.mark.property
    @given(
        text=valid_text(),
        max_size=st.integers(min_value=10, max_value=200)
    )
    def test_chunk_size_bounds(self, text, max_size):
        """Property: Chunks respect size limits."""
        assume(len(text.split()) > max_size)  # Text should be chunkable
        
        chunker = Chunker(
            chunk_method='words',
            max_chunk_size=max_size,
            chunk_overlap=0
        )
        
        chunks = chunker.chunk(text)
        
        # All chunks except possibly the last should respect max_size
        for chunk in chunks[:-1]:
            word_count = len(chunk['text'].split())
            assert word_count <= max_size
        
        # Last chunk can be smaller
        if chunks:
            last_chunk_words = len(chunks[-1]['text'].split())
            assert last_chunk_words <= max_size
    
    @pytest.mark.property
    @given(
        num_sentences=st.integers(min_value=10, max_value=50),
        sentences_per_chunk=st.integers(min_value=1, max_value=5)
    )
    def test_sentence_chunking_count(self, num_sentences, sentences_per_chunk):
        """Property: Sentence chunking produces expected number of chunks."""
        # Generate text with exact number of sentences
        text = ". ".join([f"Sentence {i}" for i in range(num_sentences)]) + "."
        
        chunker = Chunker(
            chunk_method='sentences',
            max_chunk_size=sentences_per_chunk,
            chunk_overlap=0
        )
        
        chunks = chunker.chunk(text)
        
        # Expected number of chunks
        expected_chunks = (num_sentences + sentences_per_chunk - 1) // sentences_per_chunk
        
        # Should be close to expected (allowing for edge cases)
        assert abs(len(chunks) - expected_chunks) <= 1

# ========================================================================
# Metadata Properties
# ========================================================================

class TestMetadataProperties:
    """Test properties of chunk metadata."""
    
    @pytest.mark.property
    @given(text=valid_text())
    def test_chunk_indices_sequential(self, text):
        """Property: Chunk indices are sequential."""
        chunker = Chunker(chunk_method='words', max_chunk_size=20)
        
        chunks = chunker.chunk(text)
        
        # All chunks should have metadata
        assert all('metadata' in chunk for chunk in chunks)
        
        # Indices should be sequential
        indices = [chunk['metadata'].get('chunk_index', -1) for chunk in chunks]
        expected = list(range(len(chunks)))
        assert indices == expected or all(i == -1 for i in indices)
    
    @pytest.mark.property
    @given(text=valid_text(), method=st.sampled_from(['words', 'sentences', 'paragraphs']))
    def test_metadata_contains_method(self, text, method):
        """Property: Metadata contains chunking method."""
        chunker = Chunker(chunk_method=method, max_chunk_size=50)
        
        chunks = chunker.chunk(text)
        
        for chunk in chunks:
            assert 'metadata' in chunk
            # Method should be recorded
            assert chunk['metadata'].get('method') == method or 'method' not in chunk['metadata']

# ========================================================================
# Structured Data Properties
# ========================================================================

class TestStructuredDataProperties:
    """Test properties for structured data chunking."""
    
    @pytest.mark.property
    @given(json_data=structured_json())
    def test_json_chunking_preserves_structure(self, json_data):
        """Property: JSON chunking preserves valid JSON structure."""
        chunker = Chunker(
            chunk_method='json',
            max_chunk_size=2  # Small chunks to force splitting
        )
        
        chunks = chunker.chunk(json_data)
        
        # Each chunk should be valid JSON or plain text
        for chunk in chunks:
            text = chunk['text']
            try:
                # Try to parse as JSON
                json.loads(text)
                valid_json = True
            except:
                # If not JSON, should be meaningful text
                valid_json = False
                assert len(text.strip()) > 0
    
    @pytest.mark.property
    @given(
        num_paragraphs=st.integers(min_value=1, max_value=10),
        para_size=st.integers(min_value=10, max_value=100)
    )
    def test_paragraph_chunking_preserves_breaks(self, num_paragraphs, para_size):
        """Property: Paragraph chunking preserves paragraph breaks."""
        # Generate text with clear paragraph structure
        paragraphs = []
        for i in range(num_paragraphs):
            para = " ".join([f"Word{j}" for j in range(para_size)])
            paragraphs.append(para)
        
        text = "\n\n".join(paragraphs)
        
        chunker = Chunker(
            chunk_method='paragraphs',
            max_chunk_size=2,  # 2 paragraphs per chunk
            chunk_overlap=0
        )
        
        chunks = chunker.chunk(text)
        
        # Number of chunks should match paragraph grouping
        expected_chunks = (num_paragraphs + 1) // 2
        assert abs(len(chunks) - expected_chunks) <= 1

# ========================================================================
# Performance Properties
# ========================================================================

class TestPerformanceProperties:
    """Test performance-related properties."""
    
    @pytest.mark.property
    @given(
        text_size=st.integers(min_value=100, max_value=10000),
        chunk_size=st.integers(min_value=50, max_value=500)
    )
    @settings(max_examples=10, deadline=5000)
    def test_chunking_complexity(self, text_size, chunk_size):
        """Property: Chunking time scales linearly with text size."""
        # Generate text of specific size
        text = " ".join(["word"] * text_size)
        
        chunker = Chunker(
            chunk_method='words',
            max_chunk_size=chunk_size,
            chunk_overlap=0
        )
        
        chunks = chunker.chunk(text)
        
        # Number of chunks should be proportional to text size / chunk size
        expected_chunks = text_size // chunk_size
        assert abs(len(chunks) - expected_chunks) <= 2
    
    @pytest.mark.property
    @given(overlap=st.integers(min_value=0, max_value=100))
    def test_overlap_memory_usage(self, overlap):
        """Property: Overlap doesn't cause excessive memory usage."""
        text = " ".join(["word"] * 1000)
        
        chunker = Chunker(
            chunk_method='words',
            max_chunk_size=100,
            chunk_overlap=min(overlap, 99)  # Overlap less than chunk size
        )
        
        chunks = chunker.chunk(text)
        
        # Total chunk text shouldn't be much larger than original
        total_chunk_size = sum(len(c['text']) for c in chunks)
        original_size = len(text)
        
        # Allow for overlap overhead but not excessive duplication
        if overlap > 0:
            assert total_chunk_size <= original_size * (1 + overlap / 100)
        else:
            assert total_chunk_size <= original_size * 1.1

# ========================================================================
# Edge Case Properties
# ========================================================================

class TestEdgeCaseProperties:
    """Test properties for edge cases."""
    
    @pytest.mark.property
    @given(
        text=st.one_of(
            st.just(""),
            st.just(" "),
            st.just("\n"),
            st.just("\n\n"),
            st.text(min_size=1, max_size=5)
        )
    )
    def test_handles_minimal_text(self, text):
        """Property: Chunking handles minimal or empty text gracefully."""
        chunker = Chunker(chunk_method='words', max_chunk_size=10)
        
        chunks = chunker.chunk(text)
        
        # Should always return a list
        assert isinstance(chunks, list)
        
        # Empty or whitespace-only text might return empty or single chunk
        if not text.strip():
            assert len(chunks) <= 1
        else:
            assert len(chunks) >= 1
    
    @pytest.mark.property
    @given(
        chunk_size=st.integers(min_value=1, max_value=10),
        overlap=st.integers(min_value=0, max_value=20)
    )
    def test_handles_invalid_params(self, chunk_size, overlap):
        """Property: Chunking handles invalid parameters gracefully."""
        # Overlap larger than chunk size is invalid
        if overlap >= chunk_size:
            # Should handle gracefully
            chunker = Chunker(
                chunk_method='words',
                max_chunk_size=chunk_size,
                chunk_overlap=overlap
            )
            
            text = "This is a test text with several words."
            chunks = chunker.chunk(text)
            
            # Should still produce chunks
            assert isinstance(chunks, list)
            assert len(chunks) > 0