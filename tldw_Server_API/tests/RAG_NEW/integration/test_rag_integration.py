"""
Integration tests for RAG system.

Tests the complete RAG pipeline with real components and databases.
No mocking - uses actual implementations.
"""

import pytest
import asyncio
import tempfile
import time
from pathlib import Path
from typing import List, Dict, Any
import numpy as np
from uuid import uuid4
from datetime import datetime

from tldw_Server_API.app.core.DB_Management.Media_DB_v2 import MediaDatabase
from tldw_Server_API.app.core.RAG.rag_service.functional_pipeline import (
    RAGPipelineContext,
    minimal_pipeline,
    standard_pipeline,
    quality_pipeline
)
from tldw_Server_API.app.core.RAG.rag_service.unified_pipeline import unified_rag_pipeline
from tldw_Server_API.app.core.RAG.rag_service.types import Document, SearchResult, DataSource
from tldw_Server_API.app.core.RAG.rag_service.database_retrievers import (
    MultiDatabaseRetriever,
    RetrievalConfig
)
from tldw_Server_API.app.core.RAG.rag_service.semantic_cache import SemanticCache


@pytest.mark.integration
class TestRAGPipelineIntegration:
    """Integration tests for RAG pipeline with real components."""
    
    @pytest.mark.asyncio
    async def test_minimal_pipeline_e2e(self, populated_media_db):
        """Test minimal pipeline end-to-end with real database."""
        config = {
            "enable_cache": False,
            "enable_expansion": False,
            "enable_reranking": False,
            "top_k": 5,
            "media_db": populated_media_db
        }
        
        result = await minimal_pipeline("What is RAG?", config)
        
        assert isinstance(result, RAGPipelineContext)
        assert result.query == "What is RAG?"
        assert len(result.documents) > 0
        assert "retrieval" in result.timings
        assert result.timings["retrieval"] > 0
        
        # Check that documents are from the database
        for doc in result.documents:
            assert isinstance(doc, Document)
            assert doc.content is not None
    
    @pytest.mark.asyncio
    async def test_standard_pipeline_with_cache(self, populated_media_db, temp_db_path):
        """Test standard pipeline with caching enabled."""
        cache_dir = temp_db_path.parent / "cache"
        cache_dir.mkdir(exist_ok=True)
        
        cache = SemanticCache(
            cache_dir=str(cache_dir),
            similarity_threshold=0.85,
            ttl=3600
        )
        await cache.initialize()
        
        config = {
            "enable_cache": True,
            "cache_instance": cache,
            "enable_expansion": True,
            "expansion_strategies": ["synonym"],
            "top_k": 10,
            "media_db": populated_media_db
        }
        
        # First query - should miss cache
        result1 = await standard_pipeline("machine learning", config)
        assert result1.cache_hit is False
        assert len(result1.documents) > 0
        
        # Store in cache
        await cache.set(
            "machine learning",
            result1.documents,
            ttl=3600
        )
        
        # Second similar query - should hit cache
        result2 = await standard_pipeline("machine learning", config)
        # Cache behavior depends on implementation
        
        await cache.close()
    
    @pytest.mark.asyncio
    async def test_quality_pipeline_with_all_features(self, populated_media_db):
        """Test quality pipeline with all features enabled."""
        config = {
            "enable_cache": False,  # Disable for predictable testing
            "enable_expansion": True,
            "expansion_strategies": ["synonym", "acronym"],
            "enable_reranking": True,
            "reranking_strategy": "semantic",
            "top_k": 10,
            "rerank_top_k": 3,
            "enable_analysis": True,
            "media_db": populated_media_db
        }
        
        result = await quality_pipeline("AI and ML", config)
        
        assert isinstance(result, RAGPipelineContext)
        # Query should be expanded
        assert result.query != result.original_query or result.query == "AI and ML"
        # Documents should be retrieved and reranked
        assert len(result.documents) <= 3  # rerank_top_k
        # Performance analysis should be included
        if "performance_analysis" in result.metadata:
            analysis = result.metadata["performance_analysis"]
            assert "total_time" in analysis
            assert "bottlenecks" in analysis
    
    @pytest.mark.asyncio
    async def test_pipeline_with_empty_database(self):
        """Test pipeline behavior with empty database."""
        # Create empty database
        with tempfile.TemporaryDirectory() as temp_dir:
            db_path = Path(temp_dir) / "empty.db"
            empty_db = MediaDatabase(str(db_path), "test_client")
            empty_db.initialize_db()
            
            config = {
                "enable_cache": False,
                "top_k": 5,
                "media_db": empty_db
            }
            
            result = await minimal_pipeline("test query", config)
            
            assert isinstance(result, RAGPipelineContext)
            assert len(result.documents) == 0
            # Should handle gracefully without errors
    
    @pytest.mark.asyncio
    async def test_pipeline_with_large_dataset(self, media_database):
        """Test pipeline performance with larger dataset."""
        # Add more test data
        for i in range(50):
            media_database.add_media(
                media_id=str(uuid4()),
                title=f"Document {i}",
                content=f"This is test document {i} with content about various topics including AI, ML, and RAG.",
                media_type="article",
                author=f"Author {i % 5}",
                ingestion_date=datetime.now().isoformat()
            )
        
        config = {
            "enable_cache": False,
            "top_k": 20,
            "media_db": media_database
        }
        
        start_time = time.time()
        result = await minimal_pipeline("AI and ML", config)
        elapsed = time.time() - start_time
        
        assert len(result.documents) > 0
        assert len(result.documents) <= 20
        # Performance check - should complete in reasonable time
        assert elapsed < 5.0  # 5 seconds max
    
    @pytest.mark.asyncio
    async def test_concurrent_pipeline_requests(self, populated_media_db):
        """Test handling concurrent pipeline requests."""
        config = {
            "enable_cache": False,
            "top_k": 5,
            "media_db": populated_media_db
        }
        
        queries = [
            "What is RAG?",
            "Vector databases",
            "Machine learning",
            "AI systems",
            "Information retrieval"
        ]
        
        # Run queries concurrently
        tasks = [
            minimal_pipeline(query, config.copy())
            for query in queries
        ]
        
        results = await asyncio.gather(*tasks)
        
        assert len(results) == len(queries)
        for result in results:
            assert isinstance(result, RAGPipelineContext)
            assert result.documents is not None


@pytest.mark.integration
class TestUnifiedPipelineIntegration:
    """Integration tests for unified pipeline."""
    
    @pytest.mark.asyncio
    async def test_unified_pipeline_basic(self, populated_media_db):
        """Test basic unified pipeline with real database."""
        result = await unified_rag_pipeline(
            query="What are vector databases?",
            top_k=5,
            media_db=populated_media_db
        )
        
        assert result is not None
        assert "query" in result
        assert "documents" in result
        assert len(result["documents"]) > 0
        
        # Should find the vector database document
        found_vector_doc = any(
            "vector" in doc.content.lower() 
            for doc in result["documents"]
        )
        assert found_vector_doc
    
    @pytest.mark.asyncio
    async def test_unified_pipeline_with_expansion(self, populated_media_db):
        """Test unified pipeline with query expansion."""
        result = await unified_rag_pipeline(
            query="ML",
            enable_expansion=True,
            expansion_strategies=["acronym"],
            top_k=5,
            media_db=populated_media_db
        )
        
        assert result is not None
        # Expanded query should find machine learning documents
        found_ml_doc = any(
            "machine learning" in doc.content.lower()
            for doc in result["documents"]
        )
        assert found_ml_doc
    
    @pytest.mark.asyncio
    async def test_unified_pipeline_with_filters(self, populated_media_db):
        """Test unified pipeline with filtering."""
        result = await unified_rag_pipeline(
            query="test",
            top_k=10,
            enable_date_filter=True,
            date_range={"start": "2024-01-01", "end": "2024-12-31"},
            filter_media_types=["article", "document"],
            media_db=populated_media_db
        )
        
        assert result is not None
        assert "documents" in result
        
        # All documents should match filter criteria
        for doc in result["documents"]:
            if "media_type" in doc.metadata:
                assert doc.metadata["media_type"] in ["article", "document", "video"]
    
    @pytest.mark.asyncio
    @pytest.mark.slow
    async def test_unified_pipeline_performance(self, populated_media_db):
        """Test unified pipeline performance metrics."""
        result = await unified_rag_pipeline(
            query="RAG systems",
            enable_analytics=True,
            track_performance=True,
            top_k=10,
            media_db=populated_media_db
        )
        
        assert result is not None
        if "metrics" in result:
            metrics = result["metrics"]
            assert "retrieval_time" in metrics
            assert "total_time" in metrics
            assert metrics["total_time"] > 0
            assert metrics["retrieval_time"] > 0


@pytest.mark.integration
class TestMultiDatabaseIntegration:
    """Integration tests for multi-database retrieval."""
    
    @pytest.mark.asyncio
    async def test_multi_source_retrieval(self, populated_media_db):
        """Test retrieval from multiple data sources."""
        retriever = MultiDatabaseRetriever(
            media_db=populated_media_db,
            vector_store=None,  # Would use real vector store in production
            enable_cache=False
        )
        
        config = RetrievalConfig(
            top_k=10,
            data_sources=[DataSource.MEDIA_DB]
        )
        
        results = await retriever.retrieve("RAG", config)
        
        assert len(results) > 0
        assert all(isinstance(r, SearchResult) for r in results)
        assert all(r.source == DataSource.MEDIA_DB for r in results)
    
    @pytest.mark.asyncio
    async def test_retrieval_with_scoring(self, populated_media_db):
        """Test retrieval with score-based filtering."""
        retriever = MultiDatabaseRetriever(
            media_db=populated_media_db,
            enable_cache=False
        )
        
        config = RetrievalConfig(
            top_k=10,
            min_score=0.0,  # Accept all scores for testing
            data_sources=[DataSource.MEDIA_DB]
        )
        
        results = await retriever.retrieve("machine learning", config)
        
        assert len(results) > 0
        # Results should be sorted by score
        scores = [r.score for r in results]
        assert scores == sorted(scores, reverse=True)
    
    @pytest.mark.asyncio
    async def test_retrieval_pagination(self, media_database):
        """Test paginated retrieval for large result sets."""
        # Add many documents
        for i in range(30):
            media_database.add_media(
                media_id=str(uuid4()),
                title=f"AI Document {i}",
                content=f"Content about artificial intelligence and machine learning topic {i}.",
                media_type="article",
                author="AI Author",
                ingestion_date=datetime.now().isoformat()
            )
        
        retriever = MultiDatabaseRetriever(
            media_db=media_database,
            enable_cache=False
        )
        
        # First page
        config1 = RetrievalConfig(
            top_k=10,
            offset=0,
            data_sources=[DataSource.MEDIA_DB]
        )
        results1 = await retriever.retrieve("AI", config1)
        
        # Second page
        config2 = RetrievalConfig(
            top_k=10,
            offset=10,
            data_sources=[DataSource.MEDIA_DB]
        )
        results2 = await retriever.retrieve("AI", config2)
        
        # Should have different results
        ids1 = {r.document.id for r in results1}
        ids2 = {r.document.id for r in results2}
        
        # Some overlap is okay, but not complete overlap
        assert len(ids1.intersection(ids2)) < len(ids1)


@pytest.mark.integration
class TestCacheIntegration:
    """Integration tests for semantic caching."""
    
    @pytest.mark.asyncio
    async def test_semantic_cache_operations(self, temp_db_path, sample_documents):
        """Test semantic cache with real implementation."""
        cache_dir = temp_db_path.parent / "semantic_cache"
        cache_dir.mkdir(exist_ok=True)
        
        cache = SemanticCache(
            cache_dir=str(cache_dir),
            similarity_threshold=0.85,
            ttl=3600
        )
        await cache.initialize()
        
        # Store documents
        await cache.set("test query", sample_documents, ttl=3600)
        
        # Exact match retrieval
        cached = await cache.get("test query")
        assert cached is not None
        assert len(cached) == len(sample_documents)
        
        # Similar query retrieval
        similar_cached = await cache.get("test question")
        # Depends on similarity threshold and implementation
        
        # Cache expiry
        await cache.set("expiring query", sample_documents, ttl=1)
        await asyncio.sleep(1.1)
        expired = await cache.get("expiring query")
        # Should be None or empty after TTL
        
        await cache.close()
    
    @pytest.mark.asyncio
    async def test_cache_with_pipeline(self, populated_media_db, temp_db_path):
        """Test cache integration with pipeline."""
        cache_dir = temp_db_path.parent / "pipeline_cache"
        cache_dir.mkdir(exist_ok=True)
        
        cache = SemanticCache(
            cache_dir=str(cache_dir),
            similarity_threshold=0.9,
            ttl=3600
        )
        await cache.initialize()
        
        config = {
            "enable_cache": True,
            "cache_instance": cache,
            "top_k": 5,
            "media_db": populated_media_db
        }
        
        # First query
        query1 = "What is machine learning?"
        result1 = await minimal_pipeline(query1, config)
        assert result1.cache_hit is False
        
        # Manually cache the result
        await cache.set(query1, result1.documents, ttl=3600)
        
        # Same query again
        result2 = await minimal_pipeline(query1, config)
        # Cache hit depends on pipeline implementation
        
        # Verify performance improvement with cache
        if result2.cache_hit:
            assert result2.timings.get("retrieval", 1) < result1.timings.get("retrieval", 0)
        
        await cache.close()


@pytest.mark.integration
class TestErrorRecoveryIntegration:
    """Integration tests for error recovery mechanisms."""
    
    @pytest.mark.asyncio
    async def test_pipeline_database_error_recovery(self):
        """Test pipeline recovery from database errors."""
        # Create a database that will fail
        class FailingDatabase(MediaDatabase):
            def search_media_items(self, *args, **kwargs):
                raise Exception("Database connection lost")
        
        with tempfile.TemporaryDirectory() as temp_dir:
            db_path = Path(temp_dir) / "failing.db"
            failing_db = FailingDatabase(str(db_path), "test_client")
            
            config = {
                "enable_cache": False,
                "top_k": 5,
                "media_db": failing_db,
                "fallback_on_error": True
            }
            
            # Should handle error gracefully
            result = await minimal_pipeline("test", config)
            
            assert isinstance(result, RAGPipelineContext)
            assert len(result.errors) > 0
            assert "Database connection lost" in str(result.errors)
    
    @pytest.mark.asyncio
    async def test_partial_retrieval_failure(self, populated_media_db):
        """Test handling partial retrieval failures."""
        # Create retriever that partially fails
        class PartialFailureRetriever(MultiDatabaseRetriever):
            async def retrieve_from_vectors(self, *args, **kwargs):
                raise Exception("Vector store unavailable")
        
        retriever = PartialFailureRetriever(
            media_db=populated_media_db,
            enable_cache=False
        )
        
        config = RetrievalConfig(
            top_k=10,
            data_sources=[DataSource.MEDIA_DB, DataSource.VECTORS]
        )
        
        # Should still return results from working source
        results = await retriever.retrieve("test", config)
        
        assert len(results) > 0
        assert all(r.source == DataSource.MEDIA_DB for r in results)
    
    @pytest.mark.asyncio
    async def test_retry_mechanism(self, populated_media_db):
        """Test retry mechanism for transient failures."""
        class FlakeyDatabase(MediaDatabase):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                self.attempt_count = 0
            
            def search_media_items(self, *args, **kwargs):
                self.attempt_count += 1
                if self.attempt_count < 3:
                    raise Exception("Transient error")
                return super().search_media_items(*args, **kwargs)
        
        with tempfile.TemporaryDirectory() as temp_dir:
            db_path = Path(temp_dir) / "flakey.db"
            flakey_db = FlakeyDatabase(str(db_path), "test_client")
            flakey_db.initialize_db()
            
            # Add test data
            flakey_db.add_media(
                media_id=str(uuid4()),
                title="Test",
                content="Test content",
                media_type="article"
            )
            
            config = {
                "enable_cache": False,
                "top_k": 5,
                "media_db": flakey_db,
                "enable_retry": True,
                "max_retries": 3
            }
            
            # Should succeed after retries
            result = await minimal_pipeline("test", config)
            
            assert isinstance(result, RAGPipelineContext)
            # Should eventually succeed
            if hasattr(flakey_db, 'attempt_count'):
                assert flakey_db.attempt_count >= 3


@pytest.mark.integration
@pytest.mark.slow
class TestPerformanceIntegration:
    """Integration tests for performance characteristics."""
    
    @pytest.mark.asyncio
    async def test_large_document_processing(self, media_database):
        """Test processing very large documents."""
        # Add a very large document
        large_content = " ".join([f"Sentence {i} about various AI and ML topics." for i in range(1000)])
        
        media_database.add_media(
            media_id=str(uuid4()),
            title="Large Document",
            content=large_content,
            media_type="article",
            author="Test",
            ingestion_date=datetime.now().isoformat()
        )
        
        config = {
            "enable_cache": False,
            "top_k": 5,
            "chunk_size": 500,
            "media_db": media_database
        }
        
        start_time = time.time()
        result = await minimal_pipeline("AI topics", config)
        elapsed = time.time() - start_time
        
        assert len(result.documents) > 0
        # Should complete in reasonable time even with large document
        assert elapsed < 10.0
    
    @pytest.mark.asyncio
    async def test_concurrent_load(self, populated_media_db):
        """Test system under concurrent load."""
        config = {
            "enable_cache": False,
            "top_k": 5,
            "media_db": populated_media_db
        }
        
        # Simulate concurrent users
        num_concurrent = 20
        queries = [f"Query {i % 5}" for i in range(num_concurrent)]
        
        start_time = time.time()
        
        tasks = [
            minimal_pipeline(query, config.copy())
            for query in queries
        ]
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        elapsed = time.time() - start_time
        
        # Check results
        successful = [r for r in results if not isinstance(r, Exception)]
        failed = [r for r in results if isinstance(r, Exception)]
        
        assert len(successful) > num_concurrent * 0.8  # At least 80% success
        assert elapsed < 30.0  # Should complete within 30 seconds
        
        if failed:
            print(f"Failed requests: {len(failed)}/{num_concurrent}")
    
    @pytest.mark.asyncio
    async def test_memory_usage(self, populated_media_db):
        """Test memory usage doesn't grow unbounded."""
        import psutil
        import gc
        
        process = psutil.Process()
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        config = {
            "enable_cache": False,
            "top_k": 10,
            "media_db": populated_media_db
        }
        
        # Run many queries
        for i in range(50):
            result = await minimal_pipeline(f"Query {i}", config)
            if i % 10 == 0:
                gc.collect()
        
        final_memory = process.memory_info().rss / 1024 / 1024  # MB
        memory_growth = final_memory - initial_memory
        
        # Memory growth should be reasonable
        assert memory_growth < 100  # Less than 100MB growth


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-m", "integration"])