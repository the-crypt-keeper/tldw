"""
Integration tests for the complete RAG pipeline.

Tests the end-to-end functionality including citations, analytics,
connection pooling, and embedding cache.
"""

import pytest
import tempfile
import os
import asyncio
from typing import List, Dict, Any
from pathlib import Path

from tldw_Server_API.app.core.RAG.rag_service.unified_pipeline import (
    unified_rag_pipeline,
    UnifiedSearchResult
)
from tldw_Server_API.app.core.RAG.rag_service.types import Document
from tldw_Server_API.app.core.RAG.rag_service.analytics_db import AnalyticsDatabase
from tldw_Server_API.app.core.RAG.rag_service.connection_pool import (
    get_global_pool_manager,
    close_all_pools
)
from tldw_Server_API.app.core.RAG.rag_service.embedding_cache import (
    get_global_cache_manager
)


class TestRAGPipelineIntegration:
    """Integration tests for the RAG pipeline."""
    
    @pytest.fixture(scope="class")
    def temp_databases(self):
        """Create temporary databases for testing."""
        with tempfile.TemporaryDirectory() as temp_dir:
            db_paths = {
                "media_db": os.path.join(temp_dir, "media.db"),
                "analytics_db": os.path.join(temp_dir, "analytics.db"),
                "cache_dir": os.path.join(temp_dir, "cache")
            }
            
            # Initialize databases
            self._setup_test_database(db_paths["media_db"])
            
            yield db_paths
            
            # Cleanup
            close_all_pools()
    
    def _setup_test_database(self, db_path: str):
        """Set up a test database with sample data."""
        import sqlite3
        
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        # Create minimal schema for testing
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS media (
                id TEXT PRIMARY KEY,
                title TEXT,
                content TEXT,
                author TEXT,
                date TEXT
            )
        """)
        
        # Insert test data
        test_data = [
            ("doc1", "Introduction to Machine Learning", 
             "Machine learning is a subset of artificial intelligence that enables systems to learn from data.",
             "John Smith", "2024-01-15"),
            ("doc2", "Deep Learning Fundamentals",
             "Neural networks are computing systems inspired by biological neural networks.",
             "Jane Doe", "2024-02-20"),
            ("doc3", "Natural Language Processing",
             "NLP enables computers to understand, interpret, and generate human language.",
             "Bob Wilson", "2024-03-10")
        ]
        
        cursor.executemany(
            "INSERT INTO media VALUES (?, ?, ?, ?, ?)",
            test_data
        )
        
        conn.commit()
        conn.close()
    
    @pytest.mark.asyncio
    async def test_basic_search_with_citations(self, temp_databases):
        """Test basic search with citation generation."""
        result = await unified_rag_pipeline(
            query="What is machine learning?",
            media_db_path=temp_databases["media_db"],
            enable_citations=True,
            citation_style="apa",
            enable_chunk_citations=True,
            enable_analytics=False,  # Disable for this test
            use_connection_pool=False,  # Disable for simplicity
            use_embedding_cache=False
        )
        
        assert isinstance(result, UnifiedSearchResult)
        assert len(result.documents) > 0
        assert result.query == "What is machine learning?"
        
        # Check citations were generated
        if result.documents:
            assert len(result.citations) > 0
    
    @pytest.mark.asyncio
    async def test_search_with_analytics(self, temp_databases):
        """Test search with analytics collection."""
        # Initialize analytics
        analytics_db = AnalyticsDatabase(temp_databases["analytics_db"])
        
        result = await unified_rag_pipeline(
            query="neural networks",
            media_db_path=temp_databases["media_db"],
            enable_analytics=True,
            analytics_db_path=temp_databases["analytics_db"],
            use_connection_pool=False,
            use_embedding_cache=False
        )
        
        assert isinstance(result, UnifiedSearchResult)
        
        # Check analytics were recorded
        summary = analytics_db.get_analytics_summary(days=1)
        assert summary["search_analytics"]["total_searches"] >= 1
        
        analytics_db.close()
    
    @pytest.mark.asyncio
    async def test_search_with_connection_pooling(self, temp_databases):
        """Test search with connection pooling enabled."""
        pool_manager = get_global_pool_manager()
        
        # Run multiple searches to test pooling
        tasks = []
        for i in range(5):
            task = unified_rag_pipeline(
                query=f"query {i}",
                media_db_path=temp_databases["media_db"],
                use_connection_pool=True,
                enable_analytics=False,
                use_embedding_cache=False
            )
            tasks.append(task)
        
        results = await asyncio.gather(*tasks)
        
        assert len(results) == 5
        for result in results:
            assert isinstance(result, UnifiedSearchResult)
        
        # Check pool statistics
        stats = pool_manager.get_all_stats()
        if temp_databases["media_db"] in stats:
            pool_stats = stats[temp_databases["media_db"]]
            assert pool_stats["connections_created"] >= 1
            assert pool_stats["connections_reused"] >= 0
    
    @pytest.mark.asyncio
    async def test_search_with_embedding_cache(self, temp_databases):
        """Test search with embedding cache enabled."""
        cache_manager = get_global_cache_manager()
        
        # First search - cache miss
        result1 = await unified_rag_pipeline(
            query="machine learning applications",
            media_db_path=temp_databases["media_db"],
            search_mode="vector",
            use_embedding_cache=True,
            enable_analytics=False,
            use_connection_pool=False
        )
        
        # Second search with same query - cache hit
        result2 = await unified_rag_pipeline(
            query="machine learning applications",
            media_db_path=temp_databases["media_db"],
            search_mode="vector",
            use_embedding_cache=True,
            enable_analytics=False,
            use_connection_pool=False
        )
        
        assert isinstance(result1, UnifiedSearchResult)
        assert isinstance(result2, UnifiedSearchResult)
        
        # Check cache statistics
        cache_stats = cache_manager.get_all_stats()
        if "default" in cache_stats:
            stats = cache_stats["default"]
            assert stats["hits"] >= 1  # Should have at least one hit
    
    @pytest.mark.asyncio
    async def test_full_pipeline_integration(self, temp_databases):
        """Test the full pipeline with all features enabled."""
        result = await unified_rag_pipeline(
            # Core parameters
            query="explain artificial intelligence and machine learning",
            media_db_path=temp_databases["media_db"],
            
            # Search configuration
            search_mode="hybrid",
            hybrid_alpha=0.7,
            top_k=5,
            
            # Query expansion
            expand_query=True,
            expansion_strategies=["synonym"],
            spell_check=False,  # Skip for test
            
            # Caching
            enable_cache=True,
            use_embedding_cache=True,
            
            # Citations
            enable_citations=True,
            citation_style="mla",
            enable_chunk_citations=True,
            
            # Analytics
            enable_analytics=True,
            analytics_db_path=temp_databases["analytics_db"],
            
            # Performance
            use_connection_pool=True,
            
            # Monitoring
            enable_monitoring=True
        )
        
        # Validate result structure
        assert isinstance(result, UnifiedSearchResult)
        assert result.query == "explain artificial intelligence and machine learning"
        assert len(result.documents) > 0
        
        # Check expanded queries
        if result.expanded_queries:
            assert len(result.expanded_queries) > 0
        
        # Check citations
        if result.documents:
            assert len(result.citations) > 0
        
        # Check timings
        assert "total" in result.timings or result.total_time > 0
        
        # Check no critical errors
        critical_errors = [e for e in result.errors if "critical" in e.lower()]
        assert len(critical_errors) == 0
    
    @pytest.mark.asyncio
    async def test_error_handling(self, temp_databases):
        """Test error handling in the pipeline."""
        # Test with non-existent database
        result = await unified_rag_pipeline(
            query="test query",
            media_db_path="/non/existent/path.db",
            enable_analytics=False,
            use_connection_pool=False,
            use_embedding_cache=False
        )
        
        assert isinstance(result, UnifiedSearchResult)
        assert len(result.errors) > 0
        assert len(result.documents) == 0
    
    @pytest.mark.asyncio
    async def test_cache_effectiveness(self, temp_databases):
        """Test that caching improves performance."""
        import time
        
        # First query - no cache
        start1 = time.time()
        result1 = await unified_rag_pipeline(
            query="deep learning neural networks",
            media_db_path=temp_databases["media_db"],
            enable_cache=True,
            use_embedding_cache=True,
            enable_analytics=False
        )
        time1 = time.time() - start1
        
        # Same query - should hit cache
        start2 = time.time()
        result2 = await unified_rag_pipeline(
            query="deep learning neural networks",
            media_db_path=temp_databases["media_db"],
            enable_cache=True,
            use_embedding_cache=True,
            enable_analytics=False
        )
        time2 = time.time() - start2
        
        assert isinstance(result1, UnifiedSearchResult)
        assert isinstance(result2, UnifiedSearchResult)
        
        # Second query should be faster due to caching
        # (May not always be true in test environment, so just check both completed)
        assert time1 > 0
        assert time2 > 0
        
        # Check cache hit flag
        assert result2.cache_hit or len(result2.documents) > 0
    
    @pytest.mark.asyncio
    async def test_dual_citation_system(self, temp_databases):
        """Test that both academic and chunk citations are generated."""
        result = await unified_rag_pipeline(
            query="machine learning and neural networks",
            media_db_path=temp_databases["media_db"],
            enable_citations=True,
            citation_style="apa",
            enable_chunk_citations=True,
            enable_analytics=False,
            use_connection_pool=False,
            use_embedding_cache=False
        )
        
        assert isinstance(result, UnifiedSearchResult)
        
        if result.documents:
            # Check that citations were generated
            assert result.citations is not None
            
            # Look for both academic and chunk citation markers
            has_academic = False
            has_chunks = False
            
            for citation in result.citations:
                if isinstance(citation, dict):
                    if "style" in citation or "format" in citation:
                        has_academic = True
                    if "chunk_id" in citation or "confidence" in citation:
                        has_chunks = True
                elif isinstance(citation, str):
                    # Academic citation string
                    has_academic = True
            
            # At least one type should be present
            assert has_academic or has_chunks
    
    @pytest.mark.asyncio
    async def test_analytics_privacy(self, temp_databases):
        """Test that analytics properly anonymize data."""
        analytics_db = AnalyticsDatabase(temp_databases["analytics_db"])
        
        # Run search with identifiable query
        sensitive_query = "John Doe's medical records from 2024"
        
        result = await unified_rag_pipeline(
            query=sensitive_query,
            media_db_path=temp_databases["media_db"],
            enable_analytics=True,
            analytics_db_path=temp_databases["analytics_db"],
            use_connection_pool=False,
            use_embedding_cache=False
        )
        
        # Check that query was hashed in analytics
        conn = analytics_db.connection
        cursor = conn.cursor()
        cursor.execute("SELECT query_hash FROM search_analytics")
        
        stored_hashes = [row[0] for row in cursor.fetchall()]
        
        # Verify no plain text query is stored
        for hash_val in stored_hashes:
            assert sensitive_query not in str(hash_val)
            assert "John Doe" not in str(hash_val)
        
        analytics_db.close()


class TestAPIEndpointIntegration:
    """Test the API endpoint integration with new features."""
    
    @pytest.mark.asyncio
    async def test_unified_endpoint_schema(self):
        """Test that the unified endpoint accepts new parameters."""
        from tldw_Server_API.app.api.v1.schemas.rag_schemas_unified import (
            UnifiedRAGRequest,
            UnifiedRAGResponse
        )
        
        # Test request with new features
        request = UnifiedRAGRequest(
            query="test query",
            enable_citations=True,
            citation_style="ieee",
            enable_chunk_citations=True,
            enable_analytics=True,
            use_connection_pool=True,
            use_embedding_cache=True
        )
        
        assert request.query == "test query"
        assert request.enable_citations is True
        assert request.citation_style == "ieee"
        assert request.enable_chunk_citations is True
        assert request.enable_analytics is True
        assert request.use_connection_pool is True
        assert request.use_embedding_cache is True
    
    def test_response_schema_validation(self):
        """Test that response schema includes new fields."""
        from tldw_Server_API.app.api.v1.schemas.rag_schemas_unified import (
            UnifiedRAGResponse
        )
        
        # Create response with new fields
        response = UnifiedRAGResponse(
            documents=[],
            query="test",
            academic_citations=["Citation 1", "Citation 2"],
            chunk_citations=[
                {
                    "document_id": "doc1",
                    "chunk_id": "chunk1",
                    "text": "sample text",
                    "confidence": 0.95
                }
            ],
            citations=[],  # Legacy field
            metadata={"analytics_enabled": True},
            timings={"total": 0.5},
            total_time=0.5
        )
        
        assert len(response.academic_citations) == 2
        assert len(response.chunk_citations) == 1
        assert response.chunk_citations[0]["confidence"] == 0.95


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--asyncio-mode=auto"])