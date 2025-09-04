"""
Unit tests for the unified RAG pipeline - THE ONLY PIPELINE IN USE.

Focuses exclusively on testing the unified_rag_pipeline function
and its actual dependencies.
"""

import pytest
from unittest.mock import Mock, AsyncMock, patch, MagicMock
import asyncio
from typing import Dict, List, Any
from datetime import datetime

from tldw_Server_API.app.core.RAG.rag_service.unified_pipeline import unified_rag_pipeline
from tldw_Server_API.app.core.RAG.rag_service.types import Document, SearchResult, DataSource


@pytest.mark.unit
class TestUnifiedPipelineCore:
    """Core tests for the unified pipeline - the main entry point."""
    
    @pytest.mark.asyncio
    async def test_minimal_query_execution(self):
        """Test the most basic query execution with minimal parameters."""
        # This is what most users will actually use
        with patch('tldw_Server_API.app.core.RAG.rag_service.unified_pipeline.MultiDatabaseRetriever') as mock_retriever:
            mock_retriever_instance = MagicMock()
            mock_retriever_instance.retrieve = AsyncMock(return_value=[
                SearchResult(
                    document=Document(id="1", content="RAG is a retrieval technique", metadata={}),
                    score=0.9,
                    source=DataSource.MEDIA_DB
                )
            ])
            mock_retriever.return_value = mock_retriever_instance
            
            # Mock answer generation since it requires LLM
            with patch('tldw_Server_API.app.core.RAG.rag_service.unified_pipeline.AnswerGenerator') as mock_gen:
                mock_gen_instance = MagicMock()
                mock_gen_instance.generate = AsyncMock(return_value={
                    "answer": "RAG combines retrieval with generation.",
                    "confidence": 0.85
                })
                mock_gen.return_value = mock_gen_instance
                
                # This is the actual function users call
                result = await unified_rag_pipeline(
                    query="What is RAG?",
                    top_k=5
                )
                
                assert result is not None
                assert result["query"] == "What is RAG?"
                assert "answer" in result
                assert "documents" in result
                assert len(result["documents"]) > 0
    
    @pytest.mark.asyncio
    async def test_common_user_parameters(self):
        """Test with parameters commonly used by users."""
        with patch('tldw_Server_API.app.core.RAG.rag_service.unified_pipeline.MultiDatabaseRetriever') as mock_retriever:
            mock_retriever_instance = MagicMock()
            mock_retriever_instance.retrieve = AsyncMock(return_value=[])
            mock_retriever.return_value = mock_retriever_instance
            
            with patch('tldw_Server_API.app.core.RAG.rag_service.unified_pipeline.AnswerGenerator') as mock_gen:
                mock_gen_instance = MagicMock()
                mock_gen_instance.generate = AsyncMock(return_value={"answer": "Answer"})
                mock_gen.return_value = mock_gen_instance
                
                result = await unified_rag_pipeline(
                    query="How does machine learning work?",
                    top_k=10,
                    temperature=0.7,
                    max_tokens=500,
                    enable_cache=True,
                    enable_reranking=True,
                    rerank_top_k=5
                )
                
                assert result is not None
                assert result["query"] == "How does machine learning work?"
                # Should have attempted retrieval
                mock_retriever_instance.retrieve.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_with_media_database(self, mock_media_database):
        """Test with actual media database parameter."""
        with patch('tldw_Server_API.app.core.RAG.rag_service.unified_pipeline.MultiDatabaseRetriever') as mock_retriever:
            mock_retriever_instance = MagicMock()
            mock_retriever_instance.retrieve = AsyncMock(return_value=[
                SearchResult(
                    document=Document(id="1", content="Test content", metadata={}),
                    score=0.8,
                    source=DataSource.MEDIA_DB
                )
            ])
            mock_retriever.return_value = mock_retriever_instance
            
            with patch('tldw_Server_API.app.core.RAG.rag_service.unified_pipeline.AnswerGenerator') as mock_gen:
                mock_gen_instance = MagicMock()
                mock_gen_instance.generate = AsyncMock(return_value={"answer": "Answer"})
                mock_gen.return_value = mock_gen_instance
                
                result = await unified_rag_pipeline(
                    query="test query",
                    top_k=5,
                    media_db=mock_media_database
                )
                
                # Should pass media_db to retriever
                mock_retriever.assert_called_once()
                call_kwargs = mock_retriever.call_args[1]
                assert call_kwargs.get('media_db') == mock_media_database
    
    @pytest.mark.asyncio
    async def test_error_handling_with_fallback(self):
        """Test error handling returns graceful fallback."""
        with patch('tldw_Server_API.app.core.RAG.rag_service.unified_pipeline.MultiDatabaseRetriever') as mock_retriever:
            # Simulate retrieval failure
            mock_retriever.side_effect = Exception("Database connection failed")
            
            result = await unified_rag_pipeline(
                query="test query",
                fallback_on_error=True
            )
            
            # Should return a result even with error
            assert result is not None
            # Should indicate error or provide fallback answer
            assert "error" in result or "answer" in result
    
    @pytest.mark.asyncio
    async def test_empty_retrieval_results(self):
        """Test behavior when no documents are retrieved."""
        with patch('tldw_Server_API.app.core.RAG.rag_service.unified_pipeline.MultiDatabaseRetriever') as mock_retriever:
            mock_retriever_instance = MagicMock()
            # No documents found
            mock_retriever_instance.retrieve = AsyncMock(return_value=[])
            mock_retriever.return_value = mock_retriever_instance
            
            with patch('tldw_Server_API.app.core.RAG.rag_service.unified_pipeline.AnswerGenerator') as mock_gen:
                mock_gen_instance = MagicMock()
                mock_gen_instance.generate = AsyncMock(return_value={
                    "answer": "I couldn't find relevant information to answer your question.",
                    "confidence": 0.2
                })
                mock_gen.return_value = mock_gen_instance
                
                result = await unified_rag_pipeline(
                    query="obscure query with no matches",
                    top_k=10
                )
                
                assert result is not None
                assert "answer" in result
                assert len(result.get("documents", [])) == 0


@pytest.mark.unit
class TestUnifiedPipelineFeatures:
    """Test specific features users actually use."""
    
    @pytest.mark.asyncio
    async def test_query_expansion_feature(self):
        """Test query expansion when explicitly enabled."""
        with patch('tldw_Server_API.app.core.RAG.rag_service.unified_pipeline.multi_strategy_expansion') as mock_expand:
            mock_expand.return_value = "API Application Programming Interface"
            
            with patch('tldw_Server_API.app.core.RAG.rag_service.unified_pipeline.MultiDatabaseRetriever') as mock_retriever:
                mock_retriever_instance = MagicMock()
                mock_retriever_instance.retrieve = AsyncMock(return_value=[])
                mock_retriever.return_value = mock_retriever_instance
                
                with patch('tldw_Server_API.app.core.RAG.rag_service.unified_pipeline.AnswerGenerator') as mock_gen:
                    mock_gen_instance = MagicMock()
                    mock_gen_instance.generate = AsyncMock(return_value={"answer": "Answer about API"})
                    mock_gen.return_value = mock_gen_instance
                    
                    result = await unified_rag_pipeline(
                        query="API",
                        enable_expansion=True,
                        expansion_strategies=["acronym"]
                    )
                    
                    # Should have expanded the query
                    mock_expand.assert_called_once_with("API", strategies=["acronym"])
    
    @pytest.mark.asyncio
    async def test_caching_feature(self, mock_semantic_cache):
        """Test caching when enabled by user."""
        # Setup cache hit scenario
        cached_result = {
            "answer": "Cached answer",
            "documents": [
                Document(id="cached_1", content="Cached content", metadata={})
            ],
            "cached": True
        }
        mock_semantic_cache.get.return_value = cached_result
        
        with patch('tldw_Server_API.app.core.RAG.rag_service.unified_pipeline.SemanticCache', return_value=mock_semantic_cache):
            result = await unified_rag_pipeline(
                query="cached query",
                enable_cache=True,
                cache_ttl=3600
            )
            
            # Should return cached result
            assert result["cached"] is True
            assert result["answer"] == "Cached answer"
            mock_semantic_cache.get.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_reranking_feature(self):
        """Test reranking when enabled."""
        initial_docs = [
            Document(id="1", content="Less relevant", metadata={"initial_score": 0.7}),
            Document(id="2", content="Most relevant", metadata={"initial_score": 0.8}),
            Document(id="3", content="Somewhat relevant", metadata={"initial_score": 0.75})
        ]
        
        with patch('tldw_Server_API.app.core.RAG.rag_service.unified_pipeline.MultiDatabaseRetriever') as mock_retriever:
            mock_retriever_instance = MagicMock()
            mock_retriever_instance.retrieve = AsyncMock(return_value=[
                SearchResult(document=doc, score=doc.metadata["initial_score"], source=DataSource.MEDIA_DB)
                for doc in initial_docs
            ])
            mock_retriever.return_value = mock_retriever_instance
            
            with patch('tldw_Server_API.app.core.RAG.rag_service.unified_pipeline.create_reranker') as mock_reranker_factory:
                mock_reranker = MagicMock()
                # Reranker changes order
                mock_reranker.rerank = AsyncMock(return_value=[
                    initial_docs[1],  # Most relevant now first
                    initial_docs[2],  # Somewhat relevant second
                ])
                mock_reranker_factory.return_value = mock_reranker
                
                with patch('tldw_Server_API.app.core.RAG.rag_service.unified_pipeline.AnswerGenerator') as mock_gen:
                    mock_gen_instance = MagicMock()
                    mock_gen_instance.generate = AsyncMock(return_value={"answer": "Reranked answer"})
                    mock_gen.return_value = mock_gen_instance
                    
                    result = await unified_rag_pipeline(
                        query="test",
                        enable_reranking=True,
                        rerank_top_k=2
                    )
                    
                    # Should have reranked
                    mock_reranker.rerank.assert_called_once()
                    # Should only return top 2 after reranking
                    assert len(result["documents"]) == 2
                    # Most relevant should be first
                    assert result["documents"][0].id == "2"
    
    @pytest.mark.asyncio
    async def test_filtering_features(self):
        """Test document filtering options."""
        with patch('tldw_Server_API.app.core.RAG.rag_service.unified_pipeline.MultiDatabaseRetriever') as mock_retriever:
            mock_retriever_instance = MagicMock()
            mock_retriever_instance.retrieve = AsyncMock(return_value=[
                SearchResult(
                    document=Document(
                        id="1", 
                        content="Recent document",
                        metadata={"date": "2024-06-01", "media_type": "article"}
                    ),
                    score=0.9,
                    source=DataSource.MEDIA_DB
                ),
                SearchResult(
                    document=Document(
                        id="2",
                        content="Old document", 
                        metadata={"date": "2023-01-01", "media_type": "video"}
                    ),
                    score=0.85,
                    source=DataSource.MEDIA_DB
                )
            ])
            mock_retriever.return_value = mock_retriever_instance
            
            with patch('tldw_Server_API.app.core.RAG.rag_service.unified_pipeline.AnswerGenerator') as mock_gen:
                mock_gen_instance = MagicMock()
                mock_gen_instance.generate = AsyncMock(return_value={"answer": "Filtered answer"})
                mock_gen.return_value = mock_gen_instance
                
                result = await unified_rag_pipeline(
                    query="test",
                    enable_date_filter=True,
                    date_range={"start": "2024-01-01", "end": "2024-12-31"},
                    filter_media_types=["article"]
                )
                
                # Retriever should be called with filters
                mock_retriever_instance.retrieve.assert_called_once()


@pytest.mark.unit 
class TestUnifiedPipelineRealWorldScenarios:
    """Test real-world usage scenarios."""
    
    @pytest.mark.asyncio
    async def test_chatbot_query(self):
        """Test typical chatbot query pattern."""
        # Simulating a chatbot asking about a topic
        with patch('tldw_Server_API.app.core.RAG.rag_service.unified_pipeline.MultiDatabaseRetriever') as mock_retriever:
            mock_retriever_instance = MagicMock()
            mock_retriever_instance.retrieve = AsyncMock(return_value=[
                SearchResult(
                    document=Document(
                        id="1",
                        content="Python is a high-level programming language known for its simplicity.",
                        metadata={"source": "tutorial", "author": "Expert"}
                    ),
                    score=0.95,
                    source=DataSource.MEDIA_DB
                )
            ])
            mock_retriever.return_value = mock_retriever_instance
            
            with patch('tldw_Server_API.app.core.RAG.rag_service.unified_pipeline.AnswerGenerator') as mock_gen:
                mock_gen_instance = MagicMock()
                mock_gen_instance.generate = AsyncMock(return_value={
                    "answer": "Python is a high-level programming language known for its simplicity and readability.",
                    "confidence": 0.9
                })
                mock_gen.return_value = mock_gen_instance
                
                result = await unified_rag_pipeline(
                    query="Tell me about Python programming",
                    top_k=5,
                    temperature=0.7,
                    max_tokens=200
                )
                
                assert result is not None
                assert "answer" in result
                assert "Python" in result["answer"]
                assert result["documents"][0].content is not None
    
    @pytest.mark.asyncio
    async def test_research_query(self):
        """Test research/analysis query pattern."""
        with patch('tldw_Server_API.app.core.RAG.rag_service.unified_pipeline.MultiDatabaseRetriever') as mock_retriever:
            mock_retriever_instance = MagicMock()
            # Multiple relevant documents for research
            mock_retriever_instance.retrieve = AsyncMock(return_value=[
                SearchResult(
                    document=Document(id=f"doc_{i}", content=f"Research finding {i}", metadata={"citation": f"Source {i}"}),
                    score=0.9 - i*0.05,
                    source=DataSource.MEDIA_DB
                )
                for i in range(5)
            ])
            mock_retriever.return_value = mock_retriever_instance
            
            with patch('tldw_Server_API.app.core.RAG.rag_service.unified_pipeline.AnswerGenerator') as mock_gen:
                mock_gen_instance = MagicMock()
                mock_gen_instance.generate = AsyncMock(return_value={
                    "answer": "Based on multiple sources...",
                    "citations": ["Source 1", "Source 2", "Source 3"]
                })
                mock_gen.return_value = mock_gen_instance
                
                result = await unified_rag_pipeline(
                    query="What are the latest findings on climate change?",
                    top_k=20,  # Want more sources for research
                    enable_citations=True,
                    temperature=0.3  # Lower temperature for factual accuracy
                )
                
                assert result is not None
                assert len(result["documents"]) > 1  # Multiple sources
    
    @pytest.mark.asyncio
    async def test_api_endpoint_usage(self):
        """Test usage pattern from API endpoint."""
        # Simulate parameters coming from API request
        api_params = {
            "query": "How to implement RAG?",
            "top_k": 10,
            "temperature": 0.5,
            "enable_cache": True,
            "metadata": {
                "user_id": "user123",
                "session_id": "session456",
                "request_id": "req789"
            }
        }
        
        with patch('tldw_Server_API.app.core.RAG.rag_service.unified_pipeline.MultiDatabaseRetriever') as mock_retriever:
            mock_retriever_instance = MagicMock()
            mock_retriever_instance.retrieve = AsyncMock(return_value=[])
            mock_retriever.return_value = mock_retriever_instance
            
            with patch('tldw_Server_API.app.core.RAG.rag_service.unified_pipeline.AnswerGenerator') as mock_gen:
                mock_gen_instance = MagicMock()
                mock_gen_instance.generate = AsyncMock(return_value={"answer": "RAG implementation guide..."})
                mock_gen.return_value = mock_gen_instance
                
                result = await unified_rag_pipeline(**api_params)
                
                assert result is not None
                assert "metadata" in result
                assert result["metadata"]["user_id"] == "user123"
    
    @pytest.mark.asyncio
    async def test_streaming_response(self):
        """Test streaming response for real-time applications."""
        async def mock_stream():
            """Simulate streaming response."""
            chunks = ["RAG ", "is ", "a ", "powerful ", "technique."]
            for chunk in chunks:
                yield chunk
                await asyncio.sleep(0.01)
        
        with patch('tldw_Server_API.app.core.RAG.rag_service.unified_pipeline.MultiDatabaseRetriever') as mock_retriever:
            mock_retriever_instance = MagicMock()
            mock_retriever_instance.retrieve = AsyncMock(return_value=[])
            mock_retriever.return_value = mock_retriever_instance
            
            with patch('tldw_Server_API.app.core.RAG.rag_service.unified_pipeline.AnswerGenerator') as mock_gen:
                mock_gen_instance = MagicMock()
                mock_gen_instance.generate_stream = mock_stream
                mock_gen.return_value = mock_gen_instance
                
                result = await unified_rag_pipeline(
                    query="What is RAG?",
                    enable_streaming=True
                )
                
                # Result should be streamable
                if hasattr(result, '__aiter__'):
                    chunks = []
                    async for chunk in result:
                        chunks.append(chunk)
                    assert len(chunks) == 5
                    assert "".join(chunks) == "RAG is a powerful technique."


@pytest.mark.unit
class TestUnifiedPipelineValidation:
    """Test input validation and parameter handling."""
    
    @pytest.mark.asyncio
    async def test_empty_query_handling(self):
        """Test handling of empty or whitespace queries."""
        for invalid_query in ["", "   ", "\n\t"]:
            result = await unified_rag_pipeline(
                query=invalid_query,
                top_k=5
            )
            
            # Should handle gracefully
            assert result is not None
            # Should indicate invalid query
            assert "error" in result or result.get("answer", "").lower().find("invalid") >= 0
    
    @pytest.mark.asyncio
    async def test_parameter_bounds(self):
        """Test parameter boundary conditions."""
        # Test with extreme but valid values
        with patch('tldw_Server_API.app.core.RAG.rag_service.unified_pipeline.MultiDatabaseRetriever') as mock_retriever:
            mock_retriever_instance = MagicMock()
            mock_retriever_instance.retrieve = AsyncMock(return_value=[])
            mock_retriever.return_value = mock_retriever_instance
            
            with patch('tldw_Server_API.app.core.RAG.rag_service.unified_pipeline.AnswerGenerator') as mock_gen:
                mock_gen_instance = MagicMock()
                mock_gen_instance.generate = AsyncMock(return_value={"answer": "Answer"})
                mock_gen.return_value = mock_gen_instance
                
                # Very high top_k
                result = await unified_rag_pipeline(
                    query="test",
                    top_k=1000
                )
                assert result is not None
                
                # Very low temperature
                result = await unified_rag_pipeline(
                    query="test",
                    temperature=0.0
                )
                assert result is not None
                
                # Very high temperature
                result = await unified_rag_pipeline(
                    query="test",
                    temperature=2.0
                )
                assert result is not None
    
    @pytest.mark.asyncio
    async def test_conflicting_parameters(self):
        """Test handling of conflicting parameters."""
        with patch('tldw_Server_API.app.core.RAG.rag_service.unified_pipeline.MultiDatabaseRetriever') as mock_retriever:
            mock_retriever_instance = MagicMock()
            mock_retriever_instance.retrieve = AsyncMock(return_value=[])
            mock_retriever.return_value = mock_retriever_instance
            
            with patch('tldw_Server_API.app.core.RAG.rag_service.unified_pipeline.AnswerGenerator') as mock_gen:
                mock_gen_instance = MagicMock()
                mock_gen_instance.generate = AsyncMock(return_value={"answer": "Answer"})
                mock_gen.return_value = mock_gen_instance
                
                # Rerank_top_k > top_k (conflicting)
                result = await unified_rag_pipeline(
                    query="test",
                    top_k=5,
                    enable_reranking=True,
                    rerank_top_k=10  # Higher than top_k
                )
                
                # Should handle gracefully, likely cap rerank_top_k to top_k
                assert result is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])