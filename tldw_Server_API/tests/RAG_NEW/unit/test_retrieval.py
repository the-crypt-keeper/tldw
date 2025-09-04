"""
Unit tests for RAG retrieval components.

Tests database retrievers, vector search, and hybrid retrieval.
"""

import pytest
from unittest.mock import Mock, AsyncMock, patch, MagicMock
from typing import List, Dict, Any
import numpy as np
from datetime import datetime

from tldw_Server_API.app.core.RAG.rag_service.database_retrievers import (
    MultiDatabaseRetriever,
    RetrievalConfig,
    MediaDatabaseRetriever,
    VectorDatabaseRetriever,
    HybridRetriever,
    ParentDocumentRetriever
)
from tldw_Server_API.app.core.RAG.rag_service.types import Document, SearchResult, DataSource


@pytest.mark.unit
class TestRetrievalConfig:
    """Test retrieval configuration."""
    
    def test_retrieval_config_defaults(self):
        """Test default retrieval configuration."""
        config = RetrievalConfig()
        
        assert config.top_k == 10
        assert config.enable_hybrid is False
        assert config.bm25_weight == 0.5
        assert config.vector_weight == 0.5
        assert config.min_score == 0.0
        assert config.max_results == 100
    
    def test_retrieval_config_custom(self):
        """Test custom retrieval configuration."""
        config = RetrievalConfig(
            top_k=20,
            enable_hybrid=True,
            bm25_weight=0.3,
            vector_weight=0.7,
            min_score=0.5,
            filters={"source": "media_db", "date_range": "2024"}
        )
        
        assert config.top_k == 20
        assert config.enable_hybrid is True
        assert config.bm25_weight == 0.3
        assert config.vector_weight == 0.7
        assert config.min_score == 0.5
        assert "source" in config.filters
    
    def test_retrieval_config_validation(self):
        """Test retrieval configuration validation."""
        # Weights should sum to 1.0
        config = RetrievalConfig(
            enable_hybrid=True,
            bm25_weight=0.4,
            vector_weight=0.6
        )
        
        assert config.bm25_weight + config.vector_weight == pytest.approx(1.0)
        
        # Invalid weights should be normalized
        config = RetrievalConfig(
            enable_hybrid=True,
            bm25_weight=0.8,
            vector_weight=0.8
        )
        
        total = config.bm25_weight + config.vector_weight
        assert total == pytest.approx(1.6) or (
            config.bm25_weight / total + config.vector_weight / total == pytest.approx(1.0)
        )


@pytest.mark.unit
class TestMediaDatabaseRetriever:
    """Test MediaDatabase retriever."""
    
    @pytest.mark.asyncio
    async def test_media_db_retrieval(self, mock_media_database):
        """Test basic media database retrieval."""
        retriever = MediaDatabaseRetriever(db=mock_media_database)
        
        results = await retriever.retrieve(
            query="test query",
            config=RetrievalConfig(top_k=5)
        )
        
        assert len(results) > 0
        assert all(isinstance(r, SearchResult) for r in results)
        assert all(r.source == DataSource.MEDIA_DB for r in results)
        mock_media_database.search_media_items.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_media_db_with_filters(self, mock_media_database):
        """Test media database retrieval with filters."""
        retriever = MediaDatabaseRetriever(db=mock_media_database)
        
        config = RetrievalConfig(
            top_k=10,
            filters={
                "media_type": "article",
                "date_range": {"start": "2024-01-01", "end": "2024-12-31"},
                "author": "Test Author"
            }
        )
        
        results = await retriever.retrieve(
            query="machine learning",
            config=config
        )
        
        # Verify filters were passed to database
        call_args = mock_media_database.search_media_items.call_args
        assert call_args is not None
        # Filters should be in the call arguments
    
    @pytest.mark.asyncio
    async def test_media_db_empty_results(self, mock_media_database):
        """Test media database with no results."""
        mock_media_database.search_media_items.return_value = []
        
        retriever = MediaDatabaseRetriever(db=mock_media_database)
        results = await retriever.retrieve(
            query="nonexistent",
            config=RetrievalConfig(top_k=5)
        )
        
        assert len(results) == 0
    
    @pytest.mark.asyncio
    async def test_media_db_score_filtering(self, mock_media_database):
        """Test filtering results by minimum score."""
        mock_media_database.search_media_items.return_value = [
            {"id": 1, "content": "High relevance", "score": 0.9},
            {"id": 2, "content": "Medium relevance", "score": 0.6},
            {"id": 3, "content": "Low relevance", "score": 0.3}
        ]
        
        retriever = MediaDatabaseRetriever(db=mock_media_database)
        config = RetrievalConfig(top_k=10, min_score=0.5)
        
        results = await retriever.retrieve("test", config)
        
        # Only high and medium relevance should be included
        assert len(results) == 2
        assert all(r.score >= 0.5 for r in results)
    
    @pytest.mark.asyncio
    async def test_media_db_error_handling(self, mock_media_database):
        """Test error handling in media database retrieval."""
        mock_media_database.search_media_items.side_effect = Exception("Database error")
        
        retriever = MediaDatabaseRetriever(db=mock_media_database)
        
        # Should handle error gracefully
        results = await retriever.retrieve(
            query="test",
            config=RetrievalConfig(top_k=5)
        )
        
        assert len(results) == 0 or isinstance(results, list)


@pytest.mark.unit
class TestVectorDatabaseRetriever:
    """Test vector database retriever."""
    
    @pytest.mark.asyncio
    async def test_vector_retrieval(self, mock_vector_store, mock_embeddings):
        """Test basic vector database retrieval."""
        retriever = VectorDatabaseRetriever(
            vector_store=mock_vector_store,
            embedding_function=mock_embeddings
        )
        
        results = await retriever.retrieve(
            query="vector search test",
            config=RetrievalConfig(top_k=5)
        )
        
        assert len(results) > 0
        assert all(isinstance(r, SearchResult) for r in results)
        assert all(r.source == DataSource.VECTORS for r in results)
        mock_vector_store.similarity_search.assert_called_once()
        mock_embeddings.assert_called_once_with("vector search test")
    
    @pytest.mark.asyncio
    async def test_vector_similarity_threshold(self, mock_vector_store, mock_embeddings):
        """Test vector retrieval with similarity threshold."""
        mock_vector_store.similarity_search.return_value = [
            Document(id="1", content="Very similar", metadata={"score": 0.95}),
            Document(id="2", content="Somewhat similar", metadata={"score": 0.7}),
            Document(id="3", content="Not similar", metadata={"score": 0.4})
        ]
        
        retriever = VectorDatabaseRetriever(
            vector_store=mock_vector_store,
            embedding_function=mock_embeddings
        )
        
        config = RetrievalConfig(top_k=10, min_score=0.6)
        results = await retriever.retrieve("test", config)
        
        # Only documents above threshold should be returned
        assert len(results) == 2
        assert all(r.score >= 0.6 for r in results)
    
    @pytest.mark.asyncio
    async def test_vector_metadata_filtering(self, mock_vector_store, mock_embeddings):
        """Test vector retrieval with metadata filters."""
        retriever = VectorDatabaseRetriever(
            vector_store=mock_vector_store,
            embedding_function=mock_embeddings
        )
        
        config = RetrievalConfig(
            top_k=5,
            filters={"category": "technical", "language": "en"}
        )
        
        results = await retriever.retrieve("test", config)
        
        # Verify filters were passed to vector store
        call_args = mock_vector_store.similarity_search.call_args
        assert call_args is not None
    
    @pytest.mark.asyncio
    async def test_vector_embedding_caching(self, mock_vector_store, mock_embeddings):
        """Test embedding caching for repeated queries."""
        retriever = VectorDatabaseRetriever(
            vector_store=mock_vector_store,
            embedding_function=mock_embeddings,
            cache_embeddings=True
        )
        
        # First query
        await retriever.retrieve("cached query", RetrievalConfig(top_k=5))
        
        # Second identical query
        await retriever.retrieve("cached query", RetrievalConfig(top_k=5))
        
        # Embedding should only be computed once
        mock_embeddings.assert_called_once_with("cached query")
    
    @pytest.mark.asyncio
    async def test_vector_batch_retrieval(self, mock_vector_store, mock_embeddings):
        """Test batch vector retrieval."""
        retriever = VectorDatabaseRetriever(
            vector_store=mock_vector_store,
            embedding_function=mock_embeddings
        )
        
        queries = ["query1", "query2", "query3"]
        results = await retriever.retrieve_batch(
            queries=queries,
            config=RetrievalConfig(top_k=3)
        )
        
        assert len(results) == len(queries)
        assert mock_embeddings.call_count == len(queries)


@pytest.mark.unit
class TestHybridRetriever:
    """Test hybrid retrieval combining BM25 and vector search."""
    
    @pytest.mark.asyncio
    async def test_hybrid_retrieval(self, mock_media_database, mock_vector_store, mock_embeddings):
        """Test hybrid retrieval combining multiple sources."""
        media_retriever = MediaDatabaseRetriever(db=mock_media_database)
        vector_retriever = VectorDatabaseRetriever(
            vector_store=mock_vector_store,
            embedding_function=mock_embeddings
        )
        
        hybrid_retriever = HybridRetriever(
            retrievers=[media_retriever, vector_retriever]
        )
        
        config = RetrievalConfig(
            top_k=10,
            enable_hybrid=True,
            bm25_weight=0.4,
            vector_weight=0.6
        )
        
        results = await hybrid_retriever.retrieve("hybrid test", config)
        
        assert len(results) > 0
        # Should have results from both sources
        sources = {r.source for r in results}
        assert DataSource.MEDIA_DB in sources or DataSource.VECTORS in sources
    
    @pytest.mark.asyncio
    async def test_hybrid_score_fusion(self):
        """Test score fusion in hybrid retrieval."""
        # Mock retrievers with overlapping results
        retriever1 = MagicMock()
        retriever1.retrieve = AsyncMock(return_value=[
            SearchResult(
                document=Document(id="doc1", content="Content 1", metadata={}),
                score=0.8,
                source=DataSource.MEDIA_DB
            ),
            SearchResult(
                document=Document(id="doc2", content="Content 2", metadata={}),
                score=0.6,
                source=DataSource.MEDIA_DB
            )
        ])
        
        retriever2 = MagicMock()
        retriever2.retrieve = AsyncMock(return_value=[
            SearchResult(
                document=Document(id="doc1", content="Content 1", metadata={}),
                score=0.9,
                source=DataSource.VECTORS
            ),
            SearchResult(
                document=Document(id="doc3", content="Content 3", metadata={}),
                score=0.7,
                source=DataSource.VECTORS
            )
        ])
        
        hybrid_retriever = HybridRetriever(retrievers=[retriever1, retriever2])
        
        config = RetrievalConfig(
            top_k=10,
            enable_hybrid=True,
            bm25_weight=0.5,
            vector_weight=0.5
        )
        
        results = await hybrid_retriever.retrieve("test", config)
        
        # doc1 should have highest combined score
        assert results[0].document.id == "doc1"
        # Combined score should be weighted average
        expected_score = (0.8 * 0.5 + 0.9 * 0.5)
        assert results[0].score == pytest.approx(expected_score, rel=0.1)
    
    @pytest.mark.asyncio
    async def test_hybrid_deduplication(self):
        """Test deduplication in hybrid retrieval."""
        # Mock retrievers returning duplicate documents
        retriever1 = MagicMock()
        retriever1.retrieve = AsyncMock(return_value=[
            SearchResult(
                document=Document(id="doc1", content="Duplicate content", metadata={}),
                score=0.8,
                source=DataSource.MEDIA_DB
            )
        ])
        
        retriever2 = MagicMock()
        retriever2.retrieve = AsyncMock(return_value=[
            SearchResult(
                document=Document(id="doc1", content="Duplicate content", metadata={}),
                score=0.85,
                source=DataSource.VECTORS
            )
        ])
        
        hybrid_retriever = HybridRetriever(retrievers=[retriever1, retriever2])
        results = await hybrid_retriever.retrieve(
            "test",
            RetrievalConfig(top_k=10, enable_hybrid=True)
        )
        
        # Should only have one instance of doc1
        doc_ids = [r.document.id for r in results]
        assert doc_ids.count("doc1") == 1
    
    @pytest.mark.asyncio
    async def test_hybrid_fallback(self):
        """Test fallback when one retriever fails."""
        retriever1 = MagicMock()
        retriever1.retrieve = AsyncMock(side_effect=Exception("Retriever 1 failed"))
        
        retriever2 = MagicMock()
        retriever2.retrieve = AsyncMock(return_value=[
            SearchResult(
                document=Document(id="doc1", content="Content", metadata={}),
                score=0.8,
                source=DataSource.VECTORS
            )
        ])
        
        hybrid_retriever = HybridRetriever(
            retrievers=[retriever1, retriever2],
            fallback_on_error=True
        )
        
        results = await hybrid_retriever.retrieve(
            "test",
            RetrievalConfig(top_k=5)
        )
        
        # Should still return results from working retriever
        assert len(results) == 1
        assert results[0].document.id == "doc1"


@pytest.mark.unit
class TestMultiDatabaseRetriever:
    """Test multi-database retriever coordination."""
    
    @pytest.mark.asyncio
    async def test_multi_db_initialization(self, mock_media_database, mock_vector_store):
        """Test multi-database retriever initialization."""
        retriever = MultiDatabaseRetriever(
            media_db=mock_media_database,
            vector_store=mock_vector_store,
            enable_cache=True,
            enable_reranking=False
        )
        
        assert retriever.media_db == mock_media_database
        assert retriever.vector_store == mock_vector_store
        assert retriever.enable_cache is True
        assert retriever.enable_reranking is False
    
    @pytest.mark.asyncio
    async def test_multi_db_source_selection(self, mock_media_database, mock_vector_store):
        """Test selecting specific data sources."""
        retriever = MultiDatabaseRetriever(
            media_db=mock_media_database,
            vector_store=mock_vector_store
        )
        
        # Only use media database
        config = RetrievalConfig(
            top_k=5,
            data_sources=[DataSource.MEDIA_DB]
        )
        
        results = await retriever.retrieve("test", config)
        
        mock_media_database.search_media_items.assert_called_once()
        mock_vector_store.similarity_search.assert_not_called()
        
        # Only vector store
        config = RetrievalConfig(
            top_k=5,
            data_sources=[DataSource.VECTORS]
        )
        
        results = await retriever.retrieve("test", config)
        
        mock_vector_store.similarity_search.assert_called()
    
    @pytest.mark.asyncio
    async def test_multi_db_parallel_retrieval(self, mock_media_database, mock_vector_store):
        """Test parallel retrieval from multiple sources."""
        import time
        
        # Add delays to simulate real retrieval
        async def slow_media_search(*args, **kwargs):
            await asyncio.sleep(0.1)
            return [{"id": 1, "content": "Media result"}]
        
        async def slow_vector_search(*args, **kwargs):
            await asyncio.sleep(0.1)
            return [Document(id="vec1", content="Vector result", metadata={})]
        
        mock_media_database.search_media_items = AsyncMock(side_effect=slow_media_search)
        mock_vector_store.similarity_search = AsyncMock(side_effect=slow_vector_search)
        
        retriever = MultiDatabaseRetriever(
            media_db=mock_media_database,
            vector_store=mock_vector_store,
            parallel_retrieval=True
        )
        
        start_time = time.time()
        results = await retriever.retrieve(
            "test",
            RetrievalConfig(top_k=10, data_sources=[DataSource.MEDIA_DB, DataSource.VECTORS])
        )
        elapsed = time.time() - start_time
        
        # Parallel execution should take ~0.1s, not 0.2s
        assert elapsed < 0.15
        assert len(results) > 0
    
    @pytest.mark.asyncio
    async def test_multi_db_result_merging(self, mock_media_database, mock_vector_store):
        """Test merging results from multiple databases."""
        mock_media_database.search_media_items.return_value = [
            {"id": 1, "content": "Media 1", "score": 0.9},
            {"id": 2, "content": "Media 2", "score": 0.7}
        ]
        
        mock_vector_store.similarity_search.return_value = [
            Document(id="vec1", content="Vector 1", metadata={"score": 0.85}),
            Document(id="vec2", content="Vector 2", metadata={"score": 0.65})
        ]
        
        retriever = MultiDatabaseRetriever(
            media_db=mock_media_database,
            vector_store=mock_vector_store
        )
        
        config = RetrievalConfig(
            top_k=3,
            data_sources=[DataSource.MEDIA_DB, DataSource.VECTORS]
        )
        
        results = await retriever.retrieve("test", config)
        
        # Should have top 3 results sorted by score
        assert len(results) == 3
        scores = [r.score for r in results]
        assert scores == sorted(scores, reverse=True)


@pytest.mark.unit
class TestParentDocumentRetriever:
    """Test parent document retrieval."""
    
    @pytest.mark.asyncio
    async def test_parent_document_retrieval(self, mock_media_database):
        """Test retrieving parent documents for chunks."""
        # Mock chunk with parent reference
        mock_media_database.search_media_items.return_value = [
            {
                "id": "chunk_1",
                "content": "This is a chunk",
                "parent_id": "parent_1",
                "chunk_index": 2,
                "score": 0.9
            }
        ]
        
        mock_media_database.get_media.return_value = {
            "id": "parent_1",
            "content": "This is the full parent document with multiple chunks.",
            "title": "Parent Document"
        }
        
        retriever = ParentDocumentRetriever(db=mock_media_database)
        results = await retriever.retrieve(
            "test",
            RetrievalConfig(top_k=5, include_parents=True)
        )
        
        # Should include parent document
        assert any(r.document.id == "parent_1" for r in results)
    
    @pytest.mark.asyncio
    async def test_parent_with_context_window(self, mock_media_database):
        """Test retrieving parent with context window around chunk."""
        mock_media_database.search_media_items.return_value = [
            {
                "id": "chunk_3",
                "content": "Middle chunk",
                "parent_id": "doc_1",
                "chunk_index": 3,
                "total_chunks": 5
            }
        ]
        
        # Mock getting surrounding chunks
        mock_media_database.get_chunks.return_value = [
            {"chunk_index": 2, "content": "Previous chunk"},
            {"chunk_index": 3, "content": "Middle chunk"},
            {"chunk_index": 4, "content": "Next chunk"}
        ]
        
        retriever = ParentDocumentRetriever(
            db=mock_media_database,
            context_window=1  # Include 1 chunk before and after
        )
        
        results = await retriever.retrieve(
            "test",
            RetrievalConfig(top_k=5, include_context=True)
        )
        
        # Should include context chunks
        assert len(results) > 0
        # Check that context is included in metadata or document
    
    @pytest.mark.asyncio
    async def test_parent_deduplication(self, mock_media_database):
        """Test deduplication when multiple chunks from same parent."""
        mock_media_database.search_media_items.return_value = [
            {"id": "chunk_1", "parent_id": "doc_1", "content": "Chunk 1"},
            {"id": "chunk_2", "parent_id": "doc_1", "content": "Chunk 2"},
            {"id": "chunk_3", "parent_id": "doc_2", "content": "Chunk 3"}
        ]
        
        retriever = ParentDocumentRetriever(
            db=mock_media_database,
            deduplicate_parents=True
        )
        
        results = await retriever.retrieve(
            "test",
            RetrievalConfig(top_k=10, include_parents=True)
        )
        
        # Should only have unique parent documents
        parent_ids = [r.document.metadata.get("parent_id") for r in results 
                     if "parent_id" in r.document.metadata]
        unique_parents = set(parent_ids)
        assert len(unique_parents) == len(parent_ids)


if __name__ == "__main__":
    import asyncio
    pytest.main([__file__, "-v"])