"""
Unit tests for RAG contextual retrieval functionality.

Tests parent document expansion and sibling chunk inclusion.
"""

import pytest
from unittest.mock import Mock, MagicMock, patch, AsyncMock
from typing import List, Dict, Any
import asyncio

from tldw_Server_API.app.core.RAG.rag_service.enhanced_chunking_integration import (
    expand_with_parent_context,
    filter_chunks_by_type,
    prioritize_by_chunk_type
)
from tldw_Server_API.app.core.RAG.rag_service.types import Document, DataSource
from tldw_Server_API.app.core.RAG.rag_service.functional_pipeline import RAGPipelineContext


class TestParentContextExpansion:
    """Test suite for parent context expansion in RAG."""
    
    @pytest.fixture
    def mock_context(self):
        """Create a mock RAG pipeline context with documents."""
        context = RAGPipelineContext(
            query="test query",
            original_query="test query",
            config={
                "parent_expansion_size": 500,
                "include_siblings": True
            }
        )
        
        # Add test documents with parent-child relationships
        context.documents = [
            Document(
                id="doc1_chunk_0",
                content="First chunk of document",
                metadata={
                    "parent_id": "doc1",
                    "chunk_index": 0,
                    "chunk_type": "text"
                },
                source=DataSource.MEDIA_DB,
                score=0.9
            ),
            Document(
                id="doc1_chunk_1",
                content="Second chunk of document",
                metadata={
                    "parent_id": "doc1",
                    "chunk_index": 1,
                    "chunk_type": "text"
                },
                source=DataSource.MEDIA_DB,
                score=0.85
            ),
            Document(
                id="doc2_chunk_0",
                content="Different document chunk",
                metadata={
                    "parent_id": "doc2",
                    "chunk_index": 0,
                    "chunk_type": "code"
                },
                source=DataSource.MEDIA_DB,
                score=0.8
            )
        ]
        
        return context
    
    @pytest.mark.asyncio
    async def test_expand_with_parent_context_basic(self, mock_context):
        """Test basic parent context expansion."""
        result = await expand_with_parent_context(
            mock_context,
            expansion_size=100,
            include_siblings=False
        )
        
        # Check that context was returned
        assert result is not None
        assert isinstance(result.documents, list)
        
        # Check metadata was updated
        assert result.metadata.get("parent_expansion_applied") == True
    
    @pytest.mark.asyncio
    async def test_expand_with_siblings(self, mock_context):
        """Test expansion with sibling chunks included."""
        mock_context.config["include_siblings"] = True
        
        result = await expand_with_parent_context(
            mock_context,
            include_siblings=True
        )
        
        # Check that documents were expanded
        assert result.metadata.get("parent_expansion_applied") == True
        
        # Check metadata indicates siblings were considered
        for doc in result.documents:
            # For doc2_chunk_0, has_siblings should be False since it's the only chunk
            if doc.id == "doc2_chunk_0":
                assert doc.metadata.get("has_siblings") == False
            # For doc1 chunks, should have siblings
            elif doc.metadata.get("parent_id") == "doc1":
                # doc1 has multiple chunks (0 and 1)
                assert doc.metadata.get("expanded") == True
    
    @pytest.mark.asyncio
    async def test_expand_without_siblings(self, mock_context):
        """Test expansion without sibling chunks."""
        result = await expand_with_parent_context(
            mock_context,
            include_siblings=False
        )
        
        # Check that documents were expanded
        assert result.metadata.get("parent_expansion_applied") == True
        
        # When siblings are disabled, documents should still be expanded
        # but without sibling context
        for doc in result.documents:
            assert doc.metadata.get("expanded") == True
    
    @pytest.mark.asyncio
    async def test_expansion_size_configuration(self, mock_context):
        """Test that expansion size is configurable."""
        # Test with small expansion
        result_small = await expand_with_parent_context(
            mock_context,
            expansion_size=50
        )
        
        # Test with large expansion
        result_large = await expand_with_parent_context(
            mock_context,
            expansion_size=1000
        )
        
        # Both should work without errors
        assert result_small.metadata.get("parent_expansion_applied") == True
        assert result_large.metadata.get("parent_expansion_applied") == True
    
    @pytest.mark.asyncio
    async def test_empty_documents_handling(self):
        """Test handling of empty document list."""
        context = RAGPipelineContext(
            query="test",
            original_query="test",
            config={}
        )
        context.documents = []
        
        result = await expand_with_parent_context(context)
        
        # When empty, documents stay empty and context is returned as-is
        assert result.documents == []
        # The function returns early for empty documents, so metadata isn't set
        assert result is context  # Same context object returned
    
    @pytest.mark.asyncio 
    async def test_missing_parent_id_handling(self):
        """Test handling of documents without parent IDs."""
        context = RAGPipelineContext(
            query="test",
            original_query="test",
            config={}
        )
        context.documents = [
            Document(
                id="orphan_doc",
                content="Document without parent",
                metadata={},  # No parent_id
                source=DataSource.MEDIA_DB,
                score=0.9
            )
        ]
        
        result = await expand_with_parent_context(context)
        
        # Documents without parent_id are not included in parent groups
        # So they won't appear in the expanded results
        assert len(result.documents) == 0
        assert result.metadata.get("parent_expansion_applied") == True


class TestChunkTypeFiltering:
    """Test suite for chunk type filtering in RAG."""
    
    @pytest.fixture
    def mixed_type_context(self):
        """Create context with mixed chunk types."""
        context = RAGPipelineContext(
            query="test",
            original_query="test",
            config={}
        )
        
        context.documents = [
            Document(
                id="text_chunk",
                content="Text content",
                metadata={"chunk_type": "text"},
                source=DataSource.MEDIA_DB,
                score=0.9
            ),
            Document(
                id="code_chunk",
                content="def function():",
                metadata={"chunk_type": "code"},
                source=DataSource.MEDIA_DB,
                score=0.85
            ),
            Document(
                id="table_chunk",
                content="|Col1|Col2|",
                metadata={"chunk_type": "table"},
                source=DataSource.MEDIA_DB,
                score=0.8
            ),
            Document(
                id="header_chunk",
                content="# Section Header",
                metadata={"chunk_type": "header"},
                source=DataSource.MEDIA_DB,
                score=0.75
            )
        ]
        
        return context
    
    @pytest.mark.asyncio
    async def test_filter_include_types(self, mixed_type_context):
        """Test filtering to include only specific chunk types."""
        result = await filter_chunks_by_type(
            mixed_type_context,
            include_types=["code", "text"]
        )
        
        # Should only have code and text chunks
        assert len(result.documents) == 2
        chunk_types = {doc.metadata.get("chunk_type") for doc in result.documents}
        assert chunk_types == {"code", "text"}
    
    @pytest.mark.asyncio
    async def test_filter_exclude_types(self, mixed_type_context):
        """Test filtering to exclude specific chunk types."""
        result = await filter_chunks_by_type(
            mixed_type_context,
            exclude_types=["table", "header"]
        )
        
        # Should not have table or header chunks
        assert len(result.documents) == 2
        chunk_types = {doc.metadata.get("chunk_type") for doc in result.documents}
        assert "table" not in chunk_types
        assert "header" not in chunk_types
    
    @pytest.mark.asyncio
    async def test_filter_both_include_exclude(self, mixed_type_context):
        """Test filtering with both include and exclude lists."""
        result = await filter_chunks_by_type(
            mixed_type_context,
            include_types=["code", "text", "table"],
            exclude_types=["table"]  # Exclude takes precedence
        )
        
        # Should have code and text only (table excluded)
        assert len(result.documents) == 2
        chunk_types = {doc.metadata.get("chunk_type") for doc in result.documents}
        assert chunk_types == {"code", "text"}
    
    @pytest.mark.asyncio
    async def test_no_filtering_when_none_specified(self, mixed_type_context):
        """Test that no filtering occurs when no types specified."""
        result = await filter_chunks_by_type(
            mixed_type_context,
            include_types=None,
            exclude_types=None
        )
        
        # Should have all documents
        assert len(result.documents) == 4
    
    @pytest.mark.asyncio
    async def test_filter_metadata_updated(self, mixed_type_context):
        """Test that filtering updates context metadata."""
        result = await filter_chunks_by_type(
            mixed_type_context,
            include_types=["code"]
        )
        
        assert result.metadata.get("chunk_filtering_applied") == True
        assert result.metadata.get("chunks_before_filter") == 4
        assert result.metadata.get("chunks_after_filter") == 1


class TestChunkTypePrioritization:
    """Test suite for chunk type prioritization in RAG."""
    
    @pytest.fixture
    def prioritization_context(self):
        """Create context for prioritization testing."""
        context = RAGPipelineContext(
            query="test",
            original_query="test",
            config={}
        )
        
        context.documents = [
            Document(
                id="code_chunk",
                content="code",
                metadata={"chunk_type": "code"},
                source=DataSource.MEDIA_DB,
                score=0.5  # Low initial score
            ),
            Document(
                id="text_chunk",
                content="text",
                metadata={"chunk_type": "text"},
                source=DataSource.MEDIA_DB,
                score=0.8  # High initial score
            )
        ]
        
        return context
    
    @pytest.mark.asyncio
    async def test_prioritize_by_type(self, prioritization_context):
        """Test that chunk types can be prioritized with multipliers."""
        result = await prioritize_by_chunk_type(
            prioritization_context,
            type_priorities={
                "code": 2.0,  # Double code score
                "text": 0.5   # Halve text score
            }
        )
        
        # Code chunk should now have higher score
        code_doc = next(d for d in result.documents if d.id == "code_chunk")
        text_doc = next(d for d in result.documents if d.id == "text_chunk")
        
        assert code_doc.score == 1.0  # 0.5 * 2.0
        assert text_doc.score == 0.4  # 0.8 * 0.5
        
        # Documents should be re-sorted by score
        assert result.documents[0].id == "code_chunk"  # Highest score first
    
    @pytest.mark.asyncio
    async def test_prioritization_metadata(self, prioritization_context):
        """Test that prioritization adds metadata."""
        result = await prioritize_by_chunk_type(
            prioritization_context,
            type_priorities={"code": 1.5}
        )
        
        code_doc = next(d for d in result.documents if d.id == "code_chunk")
        
        assert code_doc.metadata.get("score_adjusted") == True
        assert code_doc.metadata.get("score_multiplier") == 1.5
    
    @pytest.mark.asyncio
    async def test_default_priorities(self, prioritization_context):
        """Test that default priorities are applied when not specified."""
        result = await prioritize_by_chunk_type(
            prioritization_context,
            type_priorities=None  # Use defaults
        )
        
        # Should apply default priorities (all 1.0)
        for doc in result.documents:
            # Scores should be unchanged with default 1.0 multipliers
            if doc.id == "code_chunk":
                assert doc.score == 0.5
            elif doc.id == "text_chunk":
                assert doc.score == 0.8
    
    @pytest.mark.asyncio
    async def test_prioritization_context_metadata(self, prioritization_context):
        """Test that context metadata is updated."""
        priorities = {"code": 1.5, "text": 0.8}
        
        result = await prioritize_by_chunk_type(
            prioritization_context,
            type_priorities=priorities
        )
        
        assert result.metadata.get("chunk_type_prioritization_applied") == True
        assert result.metadata.get("type_priorities") == priorities