"""
Unit tests for the dual citation system.

Tests both academic citation formatting and chunk-level citations
for answer verification.
"""

import pytest
from datetime import datetime
from typing import List, Dict, Any

from tldw_Server_API.app.core.RAG.rag_service.citations import (
    CitationGenerator,
    ChunkCitation,
    DualCitationResult,
    CitationStyle
)
from tldw_Server_API.app.core.RAG.rag_service.types import Document, DataSource


class TestCitationGenerator:
    """Test the CitationGenerator class."""
    
    @pytest.fixture
    def generator(self):
        """Create a citation generator instance."""
        return CitationGenerator()
    
    @pytest.fixture
    def sample_documents(self) -> List[Document]:
        """Create sample documents for testing."""
        return [
            Document(
                id="doc1",
                content="Machine learning is a subset of artificial intelligence that enables systems to learn from data.",
                metadata={
                    "title": "Introduction to Machine Learning",
                    "author": "John Smith",
                    "date": "2024-01-15",
                    "url": "https://example.com/ml-intro",
                    "publisher": "Tech Publications"
                },
                source=DataSource.MEDIA_DB,
                score=0.95,
                source_document_id="source1",
                chunk_index=0,
                total_chunks=3,
                page_number=42,
                section_title="Chapter 3: Fundamentals"
            ),
            Document(
                id="doc2",
                content="Neural networks are computing systems inspired by biological neural networks.",
                metadata={
                    "title": "Deep Learning Fundamentals",
                    "author": ["Jane Doe", "Bob Wilson"],
                    "date": "2024-02-20",
                    "journal": "AI Research Journal",
                    "volume": "15",
                    "issue": "3",
                    "pages": "123-145",
                    "doi": "10.1234/airj.2024.15.3.123"
                },
                source=DataSource.MEDIA_DB,
                score=0.88,
                source_document_id="source2",
                chunk_index=2,
                total_chunks=5,
                page_number=128
            ),
            Document(
                id="doc3",
                content="Transformers have revolutionized natural language processing.",
                metadata={
                    "title": "Attention Is All You Need",
                    "author": ["Vaswani, Ashish", "Shazeer, Noam", "Parmar, Niki"],
                    "date": "2017",
                    "conference": "NeurIPS",
                    "url": "https://arxiv.org/abs/1706.03762"
                },
                source=DataSource.MEDIA_DB,
                score=0.82,
                source_document_id="source3"
            )
        ]
    
    @pytest.mark.asyncio
    async def test_mla_formatting(self, generator, sample_documents):
        """Test MLA citation formatting."""
        result = await generator.generate_citations(
            documents=sample_documents,
            style=CitationStyle.MLA,
            include_chunks=False
        )
        
        assert len(result.academic_citations) == 3
        
        # Check first citation (single author)
        mla1 = result.academic_citations[0]
        assert "Smith, John" in mla1
        assert '"Introduction to Machine Learning."' in mla1
        assert "Tech Publications" in mla1
        assert "2024" in mla1
        
        # Check second citation (multiple authors)
        mla2 = result.academic_citations[1]
        assert "Doe, Jane" in mla2
        assert "et al." in mla2 or "and Bob Wilson" in mla2
        assert '"Deep Learning Fundamentals."' in mla2
        assert "AI Research Journal" in mla2
    
    @pytest.mark.asyncio
    async def test_apa_formatting(self, generator, sample_documents):
        """Test APA citation formatting."""
        result = await generator.generate_citations(
            documents=sample_documents,
            style=CitationStyle.APA,
            include_chunks=False
        )
        
        assert len(result.academic_citations) == 3
        
        # Check first citation
        apa1 = result.academic_citations[0]
        assert "Smith, J." in apa1
        assert "(2024)" in apa1
        assert "Introduction to Machine Learning" in apa1
        
        # Check DOI formatting
        apa2 = result.academic_citations[1]
        if "doi" in sample_documents[1].metadata:
            assert "https://doi.org/10.1234/airj.2024.15.3.123" in apa2
    
    @pytest.mark.asyncio
    async def test_chicago_formatting(self, generator, sample_documents):
        """Test Chicago citation formatting."""
        result = await generator.generate_citations(
            documents=sample_documents,
            style=CitationStyle.CHICAGO,
            include_chunks=False
        )
        
        assert len(result.academic_citations) == 3
        
        chicago1 = result.academic_citations[0]
        assert "John Smith" in chicago1
        assert "Introduction to Machine Learning" in chicago1
        assert "Tech Publications" in chicago1
    
    @pytest.mark.asyncio
    async def test_harvard_formatting(self, generator, sample_documents):
        """Test Harvard citation formatting."""
        result = await generator.generate_citations(
            documents=sample_documents,
            style=CitationStyle.HARVARD,
            include_chunks=False
        )
        
        assert len(result.academic_citations) == 3
        
        harvard1 = result.academic_citations[0]
        assert "Smith, J" in harvard1
        assert "2024" in harvard1
        assert "'Introduction to Machine Learning'" in harvard1
    
    @pytest.mark.asyncio
    async def test_ieee_formatting(self, generator, sample_documents):
        """Test IEEE citation formatting."""
        result = await generator.generate_citations(
            documents=sample_documents,
            style=CitationStyle.IEEE,
            include_chunks=False
        )
        
        assert len(result.academic_citations) == 3
        
        # IEEE uses numbered references
        ieee1 = result.academic_citations[0]
        assert ieee1.startswith("[1]")
        assert "J. Smith" in ieee1
        assert '"Introduction to Machine Learning,"' in ieee1
    
    @pytest.mark.asyncio
    async def test_chunk_citations(self, generator, sample_documents):
        """Test chunk-level citation generation."""
        result = await generator.generate_citations(
            documents=sample_documents,
            include_chunks=True
        )
        
        assert len(result.chunk_citations) == 3
        
        # Check first chunk citation
        chunk1 = result.chunk_citations[0]
        assert chunk1.document_id == "doc1"
        assert chunk1.chunk_id == "doc1"
        assert chunk1.text == sample_documents[0].content
        assert chunk1.confidence == 0.95
        assert chunk1.chunk_index == 0
        assert chunk1.total_chunks == 3
        assert chunk1.page_number == 42
        assert chunk1.section == "Chapter 3: Fundamentals"
    
    @pytest.mark.asyncio
    async def test_dual_citations(self, generator, sample_documents):
        """Test generating both academic and chunk citations."""
        result = await generator.generate_citations(
            documents=sample_documents,
            style=CitationStyle.APA,
            include_chunks=True
        )
        
        # Should have both types
        assert len(result.academic_citations) == 3
        assert len(result.chunk_citations) == 3
        
        # Check citation map
        assert "source1" in result.citation_map
        assert "doc1" in result.citation_map["source1"]
    
    @pytest.mark.asyncio
    async def test_inline_markers(self, generator, sample_documents):
        """Test inline citation marker generation."""
        answer = "Machine learning is a subset of AI. Neural networks are inspired by biology."
        
        result = await generator.generate_citations(
            documents=sample_documents[:2],
            style=CitationStyle.APA,
            include_chunks=True
        )
        
        # Apply inline markers
        marked_answer = generator.add_inline_markers(answer, result)
        
        # Should have markers added
        assert "[" in marked_answer
        assert "]" in marked_answer
    
    @pytest.mark.asyncio
    async def test_empty_documents(self, generator):
        """Test handling of empty document list."""
        result = await generator.generate_citations(
            documents=[],
            style=CitationStyle.MLA
        )
        
        assert len(result.academic_citations) == 0
        assert len(result.chunk_citations) == 0
        assert len(result.citation_map) == 0
    
    @pytest.mark.asyncio
    async def test_missing_metadata(self, generator):
        """Test handling of documents with missing metadata."""
        docs = [
            Document(
                id="doc1",
                content="Test content",
                metadata={},  # Empty metadata
                source=DataSource.MEDIA_DB
            )
        ]
        
        result = await generator.generate_citations(
            documents=docs,
            style=CitationStyle.APA
        )
        
        # Should handle gracefully with "Unknown" values
        citation = result.academic_citations[0]
        assert "Unknown" in citation
    
    @pytest.mark.asyncio
    async def test_confidence_threshold(self, generator, sample_documents):
        """Test filtering by confidence threshold."""
        result = await generator.generate_citations(
            documents=sample_documents,
            include_chunks=True
        )
        
        # Only doc1 has confidence >= 0.9
        assert len(result.chunk_citations) == 1
        assert result.chunk_citations[0].document_id == "doc1"
    
    @pytest.mark.asyncio
    async def test_deduplication(self, generator):
        """Test deduplication of citations from same source."""
        # Create documents from same source
        docs = [
            Document(
                id="chunk1",
                content="First chunk",
                metadata={"title": "Same Source", "author": "Author"},
                source=DataSource.MEDIA_DB,
                source_document_id="source1"
            ),
            Document(
                id="chunk2",
                content="Second chunk",
                metadata={"title": "Same Source", "author": "Author"},
                source=DataSource.MEDIA_DB,
                source_document_id="source1"
            )
        ]
        
        result = await generator.generate_citations(
            documents=docs,
            style=CitationStyle.APA
        )
        
        # Should have only one academic citation for the source
        assert len(result.academic_citations) == 1
        # But two chunk citations
        assert len(result.chunk_citations) == 2


class TestChunkCitation:
    """Test the ChunkCitation dataclass."""
    
    def test_chunk_citation_creation(self):
        """Test creating a chunk citation."""
        citation = ChunkCitation(
            document_id="doc1",
            chunk_id="chunk1",
            text="Sample text",
            start_char=0,
            end_char=11,
            confidence=0.95,
            relevance_score=0.88,
            chunk_index=2,
            total_chunks=5,
            page_number=10,
            section="Introduction"
        )
        
        assert citation.document_id == "doc1"
        assert citation.confidence == 0.95
        assert citation.page_number == 10
    
    def test_chunk_citation_to_dict(self):
        """Test converting chunk citation to dictionary."""
        citation = ChunkCitation(
            document_id="doc1",
            chunk_id="chunk1",
            text="Sample text",
            confidence=0.95
        )
        
        citation_dict = citation.to_dict()
        
        assert citation_dict["document_id"] == "doc1"
        assert citation_dict["confidence"] == 0.95
        assert "text" in citation_dict


class TestDualCitationResult:
    """Test the DualCitationResult dataclass."""
    
    def test_result_creation(self):
        """Test creating a dual citation result."""
        result = DualCitationResult(
            academic_citations=["Citation 1", "Citation 2"],
            chunk_citations=[
                ChunkCitation("doc1", "chunk1", "text", confidence=0.9)
            ],
            inline_markers={"[1]": "chunk1"},
            citation_map={"source1": ["chunk1", "chunk2"]}
        )
        
        assert len(result.academic_citations) == 2
        assert len(result.chunk_citations) == 1
        assert "[1]" in result.inline_markers
    
    def test_result_to_dict(self):
        """Test converting result to dictionary."""
        result = DualCitationResult(
            academic_citations=["Citation 1"],
            chunk_citations=[
                ChunkCitation("doc1", "chunk1", "text", confidence=0.9)
            ]
        )
        
        result_dict = result.to_dict()
        
        assert "academic_citations" in result_dict
        assert "chunk_citations" in result_dict
        assert len(result_dict["academic_citations"]) == 1
        assert len(result_dict["chunk_citations"]) == 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])