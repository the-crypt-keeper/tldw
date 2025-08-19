"""
Tests for ChromaDB optimization functionality.
"""

import pytest
import asyncio
import time
import numpy as np
from typing import List, Dict, Any
from unittest.mock import Mock, MagicMock, AsyncMock, patch

from tldw_Server_API.app.core.RAG.rag_service.chromadb_optimizer import (
    ChromaDBOptimizationConfig, QueryResultCache, ChromaDBOptimizer,
    OptimizedChromaStore, CHROMADB_AVAILABLE
)


class TestQueryResultCache:
    """Test query result caching."""
    
    @pytest.mark.asyncio
    async def test_cache_hit_miss(self):
        """Test cache hits and misses."""
        cache = QueryResultCache(max_size=10, ttl=3600)
        
        # First query - miss
        result = await cache.get("test query", "collection1")
        assert result is None
        assert cache.misses == 1
        assert cache.hits == 0
        
        # Store result
        test_result = {"ids": [["doc1"]], "distances": [[0.1]]}
        await cache.set("test query", "collection1", test_result)
        
        # Second query - hit
        result = await cache.get("test query", "collection1")
        assert result == test_result
        assert cache.hits == 1
        assert cache.misses == 1
    
    @pytest.mark.asyncio
    async def test_cache_ttl(self):
        """Test cache TTL expiration."""
        cache = QueryResultCache(max_size=10, ttl=0.1)  # 0.1 second TTL
        
        # Store result
        test_result = {"data": "test"}
        await cache.set("query", "collection", test_result)
        
        # Immediate get - should hit
        result = await cache.get("query", "collection")
        assert result == test_result
        
        # Wait for TTL
        await asyncio.sleep(0.2)
        
        # Should miss due to expiration
        result = await cache.get("query", "collection")
        assert result is None
    
    @pytest.mark.asyncio
    async def test_cache_eviction(self):
        """Test LRU eviction."""
        cache = QueryResultCache(max_size=3, ttl=3600)
        
        # Fill cache
        for i in range(4):
            await cache.set(f"query{i}", "collection", {f"result": i})
        
        # First query should be evicted
        result = await cache.get("query0", "collection")
        assert result is None
        
        # Others should still be there
        for i in range(1, 4):
            result = await cache.get(f"query{i}", "collection")
            assert result == {"result": i}
    
    def test_hit_rate_calculation(self):
        """Test cache hit rate calculation."""
        cache = QueryResultCache()
        
        # Initial hit rate
        assert cache.hit_rate == 0.0
        
        # After some hits and misses
        cache.hits = 7
        cache.misses = 3
        assert cache.hit_rate == 0.7
    
    @pytest.mark.asyncio
    async def test_cache_key_uniqueness(self):
        """Test that cache keys are unique for different parameters."""
        cache = QueryResultCache()
        
        # Store with different parameters
        await cache.set("query", "collection1", {"result": 1}, n_results=10)
        await cache.set("query", "collection1", {"result": 2}, n_results=20)
        await cache.set("query", "collection2", {"result": 3}, n_results=10)
        
        # Each should be cached separately
        result1 = await cache.get("query", "collection1", n_results=10)
        result2 = await cache.get("query", "collection1", n_results=20)
        result3 = await cache.get("query", "collection2", n_results=10)
        
        assert result1 == {"result": 1}
        assert result2 == {"result": 2}
        assert result3 == {"result": 3}


class TestChromaDBOptimizer:
    """Test ChromaDB optimizer functionality."""
    
    def test_hybrid_search_optimization(self):
        """Test hybrid search combining vector and FTS results."""
        config = ChromaDBOptimizationConfig(
            hybrid_alpha=0.7,
            enable_hybrid_search=True
        )
        optimizer = ChromaDBOptimizer(config)
        
        # Mock vector results
        vector_results = {
            "ids": [["doc1", "doc2", "doc3"]],
            "distances": [[0.1, 0.2, 0.3]],
            "documents": [["Content 1", "Content 2", "Content 3"]],
            "metadatas": [[{"type": "vector"}, {"type": "vector"}, {"type": "vector"}]]
        }
        
        # Mock FTS results
        fts_results = [
            {"id": "doc2", "content": "FTS Content 2", "rank": -2.0},
            {"id": "doc4", "content": "FTS Content 4", "rank": -1.5},
            {"id": "doc5", "content": "FTS Content 5", "rank": -3.0}
        ]
        
        # Run optimization
        combined = optimizer.optimize_hybrid_search(
            vector_results, fts_results, top_k=5
        )
        
        # Check results
        assert len(combined) <= 5
        assert all("score" in r for r in combined)
        assert all("vector_score" in r for r in combined)
        assert all("fts_score" in r for r in combined)
        
        # Check ordering (highest score first)
        scores = [r["score"] for r in combined]
        assert scores == sorted(scores, reverse=True)
    
    def test_hybrid_search_with_rrf(self):
        """Test hybrid search with reciprocal rank fusion."""
        config = ChromaDBOptimizationConfig(
            hybrid_alpha=0.5,  # Equal weight
            hybrid_rerank=False  # No diversity reranking
        )
        optimizer = ChromaDBOptimizer(config)
        
        # Results with overlap
        vector_results = {
            "ids": [["doc1", "doc2", "doc3"]],
            "distances": [[0.1, 0.2, 0.3]],
            "documents": [["V1", "V2", "V3"]],
            "metadatas": [[{}, {}, {}]]
        }
        
        fts_results = [
            {"id": "doc2", "content": "F2", "rank": -1},  # Best FTS
            {"id": "doc3", "content": "F3", "rank": -2},
            {"id": "doc4", "content": "F4", "rank": -3}
        ]
        
        combined = optimizer.optimize_hybrid_search(vector_results, fts_results)
        
        # doc2 should rank high (appears in both)
        doc_ids = [r["id"] for r in combined]
        assert "doc2" in doc_ids[:2]  # Should be in top 2
        
        # Check RRF scores
        doc2_result = next(r for r in combined if r["id"] == "doc2")
        assert doc2_result["rrf_score"] > 0  # Has RRF score
    
    def test_diversity_reranking(self):
        """Test diversity reranking to reduce redundancy."""
        config = ChromaDBOptimizationConfig(hybrid_rerank=True)
        optimizer = ChromaDBOptimizer(config)
        
        # Create redundant results
        results = [
            {"id": "1", "score": 0.9, "document": "machine learning algorithms"},
            {"id": "2", "score": 0.85, "document": "machine learning algorithms"},  # Duplicate
            {"id": "3", "score": 0.8, "document": "deep learning networks"},
            {"id": "4", "score": 0.75, "document": "database systems"},
        ]
        
        reranked = optimizer._diversity_rerank(results, top_k=3)
        
        assert len(reranked) == 3
        # Should include diverse content
        docs = [r["document"] for r in reranked]
        # Check that not all documents are identical
        assert len(set(docs)) > 1
    
    def test_text_similarity(self):
        """Test text similarity calculation."""
        config = ChromaDBOptimizationConfig()
        optimizer = ChromaDBOptimizer(config)
        
        # Identical texts
        sim1 = optimizer._text_similarity("hello world", "hello world")
        assert sim1 == 1.0
        
        # Completely different
        sim2 = optimizer._text_similarity("alpha beta", "gamma delta")
        assert sim2 == 0.0
        
        # Partial overlap
        sim3 = optimizer._text_similarity("hello world test", "hello world example")
        assert 0 < sim3 < 1
        
        # Empty texts
        sim4 = optimizer._text_similarity("", "test")
        assert sim4 == 0.0
    
    @pytest.mark.asyncio
    async def test_batch_add_optimization(self):
        """Test batch addition optimization."""
        config = ChromaDBOptimizationConfig(
            batch_size=2,
            parallel_batch_workers=2
        )
        optimizer = ChromaDBOptimizer(config)
        
        # Mock collection
        mock_collection = Mock()
        mock_collection.add = Mock()
        
        # Test data
        documents = ["doc1", "doc2", "doc3", "doc4", "doc5"]
        embeddings = [[0.1] * 10] * 5
        metadatas = [{"id": i} for i in range(5)]
        ids = [f"id_{i}" for i in range(5)]
        
        # Run batch addition
        await optimizer.batch_add_optimized(
            mock_collection, documents, embeddings, metadatas, ids
        )
        
        # Should be called in batches
        assert mock_collection.add.call_count >= 2  # At least 2 batches
    
    def test_collection_partitioning_strategy(self):
        """Test collection partitioning for large datasets."""
        config = ChromaDBOptimizationConfig(
            max_collection_size=100_000,
            partition_by_date=True,
            partition_by_source=True
        )
        optimizer = ChromaDBOptimizer(config)
        
        # Test date-based partitioning
        strategy1 = optimizer.get_collection_strategy(
            150_000, 
            metadata={"date": "2024-01-15"}
        )
        assert strategy1 == "collection_2024-01"
        
        # Test source-based partitioning
        strategy2 = optimizer.get_collection_strategy(
            150_000,
            metadata={"source": "Media DB"}
        )
        assert strategy2 == "collection_media_db"
        
        # Test count-based partitioning
        strategy3 = optimizer.get_collection_strategy(250_000, metadata={})
        assert strategy3 == "collection_part_002"
        
        # Test no partitioning needed
        strategy4 = optimizer.get_collection_strategy(50_000)
        assert strategy4 == "main_collection"
    
    def test_performance_stats(self):
        """Test performance statistics gathering."""
        config = ChromaDBOptimizationConfig(
            enable_result_cache=True,
            hybrid_alpha=0.6,
            batch_size=100
        )
        optimizer = ChromaDBOptimizer(config)
        
        stats = optimizer.get_stats()
        
        assert "cache" in stats
        assert stats["cache"]["enabled"] is True
        assert "config" in stats
        assert stats["config"]["hybrid_alpha"] == 0.6
        assert stats["config"]["batch_size"] == 100


@pytest.mark.skipif(not CHROMADB_AVAILABLE, reason="ChromaDB not installed")
class TestOptimizedChromaStore:
    """Test integrated ChromaDB store with optimizations."""
    
    @pytest.mark.asyncio
    async def test_store_initialization(self, tmp_path):
        """Test store initialization with optimizations."""
        config = ChromaDBOptimizationConfig(
            enable_result_cache=True,
            enable_hybrid_search=True
        )
        
        store = OptimizedChromaStore(
            path=str(tmp_path),
            collection_name="test_collection",
            optimization_config=config
        )
        
        assert store.config.enable_result_cache is True
        assert store.config.enable_hybrid_search is True
        assert store.optimizer is not None
    
    @pytest.mark.asyncio
    async def test_hybrid_search_integration(self, tmp_path):
        """Test integrated hybrid search functionality."""
        store = OptimizedChromaStore(
            path=str(tmp_path),
            collection_name="test_collection"
        )
        
        # Mock vector results
        with patch.object(store.optimizer, 'search_with_cache') as mock_search:
            mock_search.return_value = {
                "ids": [["doc1", "doc2"]],
                "distances": [[0.1, 0.2]],
                "documents": [["Content 1", "Content 2"]],
                "metadatas": [[{}, {}]]
            }
            
            # Mock FTS results
            fts_results = [
                {"id": "doc3", "content": "FTS Content", "rank": -1.0}
            ]
            
            # Run hybrid search
            results = await store.hybrid_search(
                query_text="test query",
                query_embeddings=[0.1] * 10,
                fts_results=fts_results,
                n_results=5
            )
            
            assert isinstance(results, list)
            assert len(results) <= 5
    
    @pytest.mark.asyncio
    async def test_large_document_handling(self, tmp_path):
        """Test handling of large document collections."""
        config = ChromaDBOptimizationConfig(
            batch_size=500,
            parallel_batch_workers=4,
            max_collection_size=100_000
        )
        
        store = OptimizedChromaStore(
            path=str(tmp_path),
            collection_name="large_collection",
            optimization_config=config
        )
        
        # Mock large document set
        num_docs = 1000
        documents = [f"Document {i}" for i in range(num_docs)]
        embeddings = [[0.1] * 10] * num_docs
        metadatas = [{"index": i} for i in range(num_docs)]
        ids = [f"doc_{i}" for i in range(num_docs)]
        
        # Mock the collection
        if store.collection:
            with patch.object(store.collection, 'add') as mock_add:
                mock_add.return_value = None
                
                result = await store.add_documents(
                    documents, embeddings, metadatas, ids
                )
                
                assert result is True
                # Should batch the additions
                assert mock_add.call_count >= 2
    
    def test_performance_monitoring(self, tmp_path):
        """Test performance statistics collection."""
        store = OptimizedChromaStore(
            path=str(tmp_path),
            collection_name="test_collection"
        )
        
        stats = store.get_performance_stats()
        
        assert "cache" in stats
        assert "config" in stats
        # Collection size would be included if collection exists
        if store.collection:
            assert "collection_size" in stats


class TestLargeScaleOptimizations:
    """Test optimizations specifically for 100k+ document collections."""
    
    def test_large_collection_config(self):
        """Test configuration for large collections."""
        config = ChromaDBOptimizationConfig()
        
        # Check defaults are suitable for large collections
        assert config.cache_size == 5000  # Large cache
        assert config.batch_size == 500  # Large batches
        assert config.max_collection_size == 100_000  # Partition threshold
        assert config.parallel_batch_workers == 4  # Parallel processing
        assert config.max_connections == 20  # Many connections
    
    @pytest.mark.asyncio
    async def test_parallel_batch_processing(self):
        """Test parallel batch processing for large datasets."""
        config = ChromaDBOptimizationConfig(
            batch_size=100,
            parallel_batch_workers=3
        )
        optimizer = ChromaDBOptimizer(config)
        
        # Mock collection
        mock_collection = Mock()
        mock_collection.add = Mock()
        
        # Large dataset (triggers parallel processing)
        num_docs = 15000
        documents = [f"doc_{i}" for i in range(num_docs)]
        embeddings = [[0.1] * 10] * num_docs
        metadatas = [{"i": i} for i in range(num_docs)]
        ids = [f"id_{i}" for i in range(num_docs)]
        
        await optimizer.batch_add_optimized(
            mock_collection, documents, embeddings, metadatas, ids
        )
        
        # Should use parallel processing
        assert mock_collection.add.call_count >= num_docs // config.batch_size
    
    def test_hybrid_search_performance(self):
        """Test hybrid search performance with large result sets."""
        config = ChromaDBOptimizationConfig()
        optimizer = ChromaDBOptimizer(config)
        
        # Large result sets
        num_vector_results = 1000
        num_fts_results = 500
        
        vector_results = {
            "ids": [[f"v{i}" for i in range(num_vector_results)]],
            "distances": [[0.1 + i * 0.001 for i in range(num_vector_results)]],
            "documents": [[f"Vector doc {i}" for i in range(num_vector_results)]],
            "metadatas": [[{"source": "vector"}] * num_vector_results]
        }
        
        fts_results = [
            {"id": f"f{i}", "content": f"FTS doc {i}", "rank": -i}
            for i in range(num_fts_results)
        ]
        
        # Time the optimization
        start = time.time()
        combined = optimizer.optimize_hybrid_search(
            vector_results, fts_results, top_k=100
        )
        duration = time.time() - start
        
        assert len(combined) == 100
        # Should complete quickly even with large inputs
        assert duration < 1.0  # Under 1 second
    
    @pytest.mark.asyncio
    async def test_metadata_indexing_optimization(self):
        """Test metadata indexing for large collections."""
        config = ChromaDBOptimizationConfig(enable_metadata_indexing=True)
        optimizer = ChromaDBOptimizer(config)
        
        # Mock collection with sample metadata
        mock_collection = Mock()
        mock_collection.get = Mock(return_value={
            "metadatas": [
                {"date": "2024-01-01", "source": "web", "type": "article"},
                {"date": "2024-01-02", "source": "pdf", "type": "paper"},
                {"date": "2024-01-03", "source": "web", "type": "article"},
            ] * 30  # 90 documents
        })
        
        # Run optimization
        await optimizer.optimize_metadata_indexing(mock_collection)
        
        # Should analyze metadata
        mock_collection.get.assert_called_once()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])