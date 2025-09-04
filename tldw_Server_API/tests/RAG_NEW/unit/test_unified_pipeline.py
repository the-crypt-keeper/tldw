"""
Unit tests for the unified RAG pipeline.

Tests the single unified pipeline function with all feature combinations.
"""

import pytest
from unittest.mock import Mock, AsyncMock, patch, MagicMock, call
import asyncio
from typing import Dict, List, Any, Optional
from datetime import datetime

from tldw_Server_API.app.core.RAG.rag_service.unified_pipeline import (
    unified_rag_pipeline,
    UnifiedRAGResult,
    create_unified_context,
    validate_parameters,
    select_features,
    execute_pipeline_stages
)
from tldw_Server_API.app.core.RAG.rag_service.types import Document, SearchResult, DataSource


@pytest.mark.unit
class TestUnifiedPipeline:
    """Test the unified RAG pipeline function."""
    
    @pytest.mark.asyncio
    async def test_unified_pipeline_minimal(self):
        """Test unified pipeline with minimal parameters."""
        with patch('tldw_Server_API.app.core.RAG.rag_service.unified_pipeline.MultiDatabaseRetriever') as mock_retriever:
            mock_retriever_instance = MagicMock()
            mock_retriever_instance.retrieve = AsyncMock(return_value=[
                SearchResult(
                    document=Document(id="1", content="Test content", metadata={}),
                    score=0.9,
                    source=DataSource.MEDIA_DB
                )
            ])
            mock_retriever.return_value = mock_retriever_instance
            
            with patch('tldw_Server_API.app.core.RAG.rag_service.unified_pipeline.AnswerGenerator') as mock_generator:
                mock_generator_instance = MagicMock()
                mock_generator_instance.generate = AsyncMock(return_value={
                    "answer": "Generated answer",
                    "confidence": 0.85
                })
                mock_generator.return_value = mock_generator_instance
                
                result = await unified_rag_pipeline(
                    query="What is RAG?",
                    top_k=5
                )
                
                assert result is not None
                assert "answer" in result
                assert "documents" in result
                assert result["answer"] == "Generated answer"
                assert len(result["documents"]) > 0
    
    @pytest.mark.asyncio
    async def test_unified_pipeline_with_cache(self, mock_semantic_cache):
        """Test unified pipeline with caching enabled."""
        # Test cache hit
        cached_result = {
            "answer": "Cached answer",
            "documents": [
                Document(id="cached_1", content="Cached content", metadata={})
            ]
        }
        mock_semantic_cache.get.return_value = cached_result
        
        with patch('tldw_Server_API.app.core.RAG.rag_service.unified_pipeline.SemanticCache', return_value=mock_semantic_cache):
            result = await unified_rag_pipeline(
                query="cached query",
                enable_cache=True,
                cache_ttl=3600
            )
            
            assert result["answer"] == "Cached answer"
            assert result["documents"][0].id == "cached_1"
            mock_semantic_cache.get.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_unified_pipeline_with_expansion(self):
        """Test unified pipeline with query expansion."""
        with patch('tldw_Server_API.app.core.RAG.rag_service.unified_pipeline.multi_strategy_expansion') as mock_expand:
            mock_expand.return_value = "API Application Programming Interface"
            
            with patch('tldw_Server_API.app.core.RAG.rag_service.unified_pipeline.MultiDatabaseRetriever') as mock_retriever:
                mock_retriever_instance = MagicMock()
                mock_retriever_instance.retrieve = AsyncMock(return_value=[])
                mock_retriever.return_value = mock_retriever_instance
                
                with patch('tldw_Server_API.app.core.RAG.rag_service.unified_pipeline.AnswerGenerator') as mock_generator:
                    mock_generator_instance = MagicMock()
                    mock_generator_instance.generate = AsyncMock(return_value={"answer": "Answer"})
                    mock_generator.return_value = mock_generator_instance
                    
                    result = await unified_rag_pipeline(
                        query="API",
                        enable_expansion=True,
                        expansion_strategies=["acronym", "synonym"]
                    )
                    
                    mock_expand.assert_called_once_with(
                        "API",
                        strategies=["acronym", "synonym"]
                    )
                    # Expanded query should be used for retrieval
                    call_args = mock_retriever_instance.retrieve.call_args
                    assert "API Application Programming Interface" in str(call_args)
    
    @pytest.mark.asyncio
    async def test_unified_pipeline_with_reranking(self, sample_documents):
        """Test unified pipeline with reranking enabled."""
        with patch('tldw_Server_API.app.core.RAG.rag_service.unified_pipeline.MultiDatabaseRetriever') as mock_retriever:
            # Return documents in one order
            mock_retriever_instance = MagicMock()
            mock_retriever_instance.retrieve = AsyncMock(return_value=[
                SearchResult(document=doc, score=0.8, source=DataSource.MEDIA_DB)
                for doc in sample_documents
            ])
            mock_retriever.return_value = mock_retriever_instance
            
            with patch('tldw_Server_API.app.core.RAG.rag_service.unified_pipeline.create_reranker') as mock_reranker_factory:
                mock_reranker = MagicMock()
                # Return documents in different order
                mock_reranker.rerank = AsyncMock(return_value=[
                    sample_documents[2],
                    sample_documents[0],
                    sample_documents[1]
                ])
                mock_reranker_factory.return_value = mock_reranker
                
                with patch('tldw_Server_API.app.core.RAG.rag_service.unified_pipeline.AnswerGenerator') as mock_generator:
                    mock_generator_instance = MagicMock()
                    mock_generator_instance.generate = AsyncMock(return_value={"answer": "Answer"})
                    mock_generator.return_value = mock_generator_instance
                    
                    result = await unified_rag_pipeline(
                        query="test",
                        enable_reranking=True,
                        reranking_strategy="cross_encoder",
                        rerank_top_k=3
                    )
                    
                    mock_reranker.rerank.assert_called_once()
                    # Documents should be reordered
                    assert len(result["documents"]) <= 3
    
    @pytest.mark.asyncio
    async def test_unified_pipeline_with_filters(self):
        """Test unified pipeline with various filters."""
        with patch('tldw_Server_API.app.core.RAG.rag_service.unified_pipeline.SecurityFilter') as mock_security:
            mock_filter = MagicMock()
            mock_filter.filter_documents = lambda docs, level: [
                d for d in docs if d.metadata.get("sensitive") != True
            ]
            mock_security.return_value = mock_filter
            
            with patch('tldw_Server_API.app.core.RAG.rag_service.unified_pipeline.MultiDatabaseRetriever') as mock_retriever:
                mock_retriever_instance = MagicMock()
                mock_retriever_instance.retrieve = AsyncMock(return_value=[
                    SearchResult(
                        document=Document(id="1", content="Public", metadata={"sensitive": False}),
                        score=0.9,
                        source=DataSource.MEDIA_DB
                    ),
                    SearchResult(
                        document=Document(id="2", content="Secret", metadata={"sensitive": True}),
                        score=0.85,
                        source=DataSource.MEDIA_DB
                    )
                ])
                mock_retriever.return_value = mock_retriever_instance
                
                with patch('tldw_Server_API.app.core.RAG.rag_service.unified_pipeline.AnswerGenerator') as mock_generator:
                    mock_generator_instance = MagicMock()
                    mock_generator_instance.generate = AsyncMock(return_value={"answer": "Answer"})
                    mock_generator.return_value = mock_generator_instance
                    
                    result = await unified_rag_pipeline(
                        query="test",
                        enable_security_filter=True,
                        user_clearance_level="public"
                    )
                    
                    # Only non-sensitive document should be in results
                    assert len(result["documents"]) == 1
                    assert result["documents"][0].id == "1"
    
    @pytest.mark.asyncio
    async def test_unified_pipeline_with_citations(self, sample_documents):
        """Test unified pipeline with citation generation."""
        with patch('tldw_Server_API.app.core.RAG.rag_service.unified_pipeline.MultiDatabaseRetriever') as mock_retriever:
            mock_retriever_instance = MagicMock()
            mock_retriever_instance.retrieve = AsyncMock(return_value=[
                SearchResult(document=doc, score=0.9, source=DataSource.MEDIA_DB)
                for doc in sample_documents
            ])
            mock_retriever.return_value = mock_retriever_instance
            
            with patch('tldw_Server_API.app.core.RAG.rag_service.unified_pipeline.DualCitationGenerator') as mock_citation:
                mock_citation_instance = MagicMock()
                mock_citation_instance.generate_citations = AsyncMock(return_value={
                    "inline_citations": "[1] Reference to document 1",
                    "bibliography": ["[1] Document 1 - Author (2024)"]
                })
                mock_citation.return_value = mock_citation_instance
                
                with patch('tldw_Server_API.app.core.RAG.rag_service.unified_pipeline.AnswerGenerator') as mock_generator:
                    mock_generator_instance = MagicMock()
                    mock_generator_instance.generate = AsyncMock(return_value={"answer": "Answer with citations"})
                    mock_generator.return_value = mock_generator_instance
                    
                    result = await unified_rag_pipeline(
                        query="test",
                        enable_citations=True,
                        citation_style="academic"
                    )
                    
                    assert "citations" in result
                    assert "inline_citations" in result["citations"]
                    assert "bibliography" in result["citations"]
                    mock_citation_instance.generate_citations.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_unified_pipeline_all_features(self):
        """Test unified pipeline with all features enabled."""
        result = await unified_rag_pipeline(
            query="What is RAG?",
            # Retrieval settings
            top_k=20,
            enable_hybrid_search=True,
            bm25_weight=0.3,
            vector_weight=0.7,
            
            # Expansion settings
            enable_expansion=True,
            expansion_strategies=["synonym", "acronym", "entity"],
            
            # Cache settings
            enable_cache=True,
            cache_ttl=7200,
            
            # Reranking settings
            enable_reranking=True,
            reranking_strategy="cross_encoder",
            rerank_top_k=5,
            
            # Filter settings
            enable_security_filter=True,
            user_clearance_level="confidential",
            enable_date_filter=True,
            date_range={"start": "2024-01-01", "end": "2024-12-31"},
            
            # Generation settings
            temperature=0.5,
            max_tokens=1000,
            enable_streaming=False,
            
            # Citation settings
            enable_citations=True,
            citation_style="numeric",
            
            # Analytics
            enable_analytics=True,
            track_performance=True,
            
            # Advanced features
            enable_spell_check=True,
            enable_result_highlighting=True,
            enable_cost_tracking=True,
            enable_feedback=True
        )
        
        # Basic assertions - with all mocking this should at least not error
        assert result is not None
        assert "query" in result
        assert result["query"] == "What is RAG?"
    
    @pytest.mark.asyncio
    async def test_unified_pipeline_error_handling(self):
        """Test unified pipeline error handling."""
        with patch('tldw_Server_API.app.core.RAG.rag_service.unified_pipeline.MultiDatabaseRetriever') as mock_retriever:
            mock_retriever.side_effect = Exception("Retrieval failed")
            
            result = await unified_rag_pipeline(
                query="test",
                fallback_on_error=True
            )
            
            # Should return a fallback result instead of raising
            assert result is not None
            assert "error" in result or "answer" in result
            if "answer" in result:
                assert "error" in result["answer"].lower() or "unable" in result["answer"].lower()
    
    @pytest.mark.asyncio
    async def test_unified_pipeline_with_metadata(self):
        """Test unified pipeline with custom metadata."""
        custom_metadata = {
            "user_id": "user123",
            "session_id": "session456",
            "request_id": "req789",
            "custom_field": "custom_value"
        }
        
        with patch('tldw_Server_API.app.core.RAG.rag_service.unified_pipeline.MultiDatabaseRetriever') as mock_retriever:
            mock_retriever_instance = MagicMock()
            mock_retriever_instance.retrieve = AsyncMock(return_value=[])
            mock_retriever.return_value = mock_retriever_instance
            
            with patch('tldw_Server_API.app.core.RAG.rag_service.unified_pipeline.AnswerGenerator') as mock_generator:
                mock_generator_instance = MagicMock()
                mock_generator_instance.generate = AsyncMock(return_value={"answer": "Answer"})
                mock_generator.return_value = mock_generator_instance
                
                result = await unified_rag_pipeline(
                    query="test",
                    metadata=custom_metadata
                )
                
                assert "metadata" in result
                for key, value in custom_metadata.items():
                    assert result["metadata"].get(key) == value


@pytest.mark.unit
class TestPipelineValidation:
    """Test parameter validation and feature selection."""
    
    def test_validate_parameters_valid(self):
        """Test validation with valid parameters."""
        params = {
            "query": "test query",
            "top_k": 10,
            "temperature": 0.7,
            "enable_cache": True
        }
        
        errors = validate_parameters(params)
        assert len(errors) == 0
    
    def test_validate_parameters_invalid(self):
        """Test validation with invalid parameters."""
        params = {
            "query": "",  # Empty query
            "top_k": -1,  # Negative
            "temperature": 2.5,  # Out of range
            "rerank_top_k": 1000  # Too large
        }
        
        errors = validate_parameters(params)
        assert len(errors) > 0
        assert any("query" in str(e).lower() for e in errors)
        assert any("top_k" in str(e).lower() for e in errors)
        assert any("temperature" in str(e).lower() for e in errors)
    
    def test_validate_parameters_missing_required(self):
        """Test validation with missing required parameters."""
        params = {
            "top_k": 10
            # Missing query
        }
        
        errors = validate_parameters(params)
        assert len(errors) > 0
        assert any("query" in str(e).lower() for e in errors)
    
    def test_select_features_basic(self):
        """Test feature selection with basic parameters."""
        params = {
            "enable_cache": False,
            "enable_expansion": False,
            "enable_reranking": False
        }
        
        features = select_features(params)
        
        assert features["use_cache"] is False
        assert features["use_expansion"] is False
        assert features["use_reranking"] is False
        assert features["pipeline_type"] == "minimal"
    
    def test_select_features_advanced(self):
        """Test feature selection with advanced parameters."""
        params = {
            "enable_cache": True,
            "enable_expansion": True,
            "expansion_strategies": ["synonym", "acronym"],
            "enable_reranking": True,
            "reranking_strategy": "cross_encoder",
            "enable_citations": True,
            "enable_analytics": True
        }
        
        features = select_features(params)
        
        assert features["use_cache"] is True
        assert features["use_expansion"] is True
        assert features["use_reranking"] is True
        assert features["use_citations"] is True
        assert features["use_analytics"] is True
        assert features["pipeline_type"] == "quality"
    
    def test_select_features_auto_detection(self):
        """Test automatic feature detection based on parameters."""
        params = {
            "expansion_strategies": ["synonym"],  # Implies expansion
            "reranking_strategy": "bm25",  # Implies reranking
            "cache_ttl": 3600  # Implies cache
        }
        
        features = select_features(params)
        
        assert features["use_expansion"] is True
        assert features["use_reranking"] is True
        assert features["use_cache"] is True


@pytest.mark.unit
class TestPipelineStages:
    """Test individual pipeline stages."""
    
    @pytest.mark.asyncio
    async def test_execute_expansion_stage(self):
        """Test expansion stage execution."""
        context = {
            "query": "ML",
            "original_query": "ML",
            "features": {"use_expansion": True},
            "params": {"expansion_strategies": ["acronym"]}
        }
        
        with patch('tldw_Server_API.app.core.RAG.rag_service.unified_pipeline.expand_acronyms') as mock_expand:
            mock_expand.return_value = "ML Machine Learning"
            
            result = await execute_pipeline_stages(context, ["expansion"])
            
            assert result["query"] == "ML Machine Learning"
            mock_expand.assert_called_once_with("ML")
    
    @pytest.mark.asyncio
    async def test_execute_retrieval_stage(self):
        """Test retrieval stage execution."""
        context = {
            "query": "test query",
            "features": {},
            "params": {"top_k": 5}
        }
        
        with patch('tldw_Server_API.app.core.RAG.rag_service.unified_pipeline.MultiDatabaseRetriever') as mock_retriever:
            mock_retriever_instance = MagicMock()
            mock_retriever_instance.retrieve = AsyncMock(return_value=[
                SearchResult(
                    document=Document(id="1", content="Result", metadata={}),
                    score=0.9,
                    source=DataSource.MEDIA_DB
                )
            ])
            mock_retriever.return_value = mock_retriever_instance
            
            result = await execute_pipeline_stages(context, ["retrieval"])
            
            assert "documents" in result
            assert len(result["documents"]) == 1
            mock_retriever_instance.retrieve.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_execute_reranking_stage(self, sample_documents):
        """Test reranking stage execution."""
        context = {
            "query": "test",
            "documents": sample_documents,
            "features": {"use_reranking": True},
            "params": {
                "reranking_strategy": "semantic",
                "rerank_top_k": 2
            }
        }
        
        with patch('tldw_Server_API.app.core.RAG.rag_service.unified_pipeline.create_reranker') as mock_reranker_factory:
            mock_reranker = MagicMock()
            mock_reranker.rerank = AsyncMock(return_value=sample_documents[:2])
            mock_reranker_factory.return_value = mock_reranker
            
            result = await execute_pipeline_stages(context, ["reranking"])
            
            assert len(result["documents"]) == 2
            mock_reranker.rerank.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_execute_generation_stage(self, sample_documents):
        """Test generation stage execution."""
        context = {
            "query": "What is RAG?",
            "documents": sample_documents,
            "features": {},
            "params": {
                "temperature": 0.7,
                "max_tokens": 500
            }
        }
        
        with patch('tldw_Server_API.app.core.RAG.rag_service.unified_pipeline.AnswerGenerator') as mock_generator:
            mock_generator_instance = MagicMock()
            mock_generator_instance.generate = AsyncMock(return_value={
                "answer": "RAG is Retrieval-Augmented Generation.",
                "confidence": 0.9,
                "tokens_used": 50
            })
            mock_generator.return_value = mock_generator_instance
            
            result = await execute_pipeline_stages(context, ["generation"])
            
            assert "answer" in result
            assert result["answer"] == "RAG is Retrieval-Augmented Generation."
            assert "confidence" in result
            mock_generator_instance.generate.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_execute_multiple_stages(self):
        """Test executing multiple pipeline stages."""
        context = {
            "query": "API",
            "original_query": "API",
            "features": {
                "use_expansion": True,
                "use_reranking": False
            },
            "params": {
                "expansion_strategies": ["acronym"],
                "top_k": 5,
                "temperature": 0.7
            }
        }
        
        with patch('tldw_Server_API.app.core.RAG.rag_service.unified_pipeline.expand_acronyms') as mock_expand:
            mock_expand.return_value = "API Application Programming Interface"
            
            with patch('tldw_Server_API.app.core.RAG.rag_service.unified_pipeline.MultiDatabaseRetriever') as mock_retriever:
                mock_retriever_instance = MagicMock()
                mock_retriever_instance.retrieve = AsyncMock(return_value=[
                    SearchResult(
                        document=Document(id="1", content="API documentation", metadata={}),
                        score=0.95,
                        source=DataSource.MEDIA_DB
                    )
                ])
                mock_retriever.return_value = mock_retriever_instance
                
                with patch('tldw_Server_API.app.core.RAG.rag_service.unified_pipeline.AnswerGenerator') as mock_generator:
                    mock_generator_instance = MagicMock()
                    mock_generator_instance.generate = AsyncMock(return_value={
                        "answer": "An API is an Application Programming Interface."
                    })
                    mock_generator.return_value = mock_generator_instance
                    
                    result = await execute_pipeline_stages(
                        context,
                        ["expansion", "retrieval", "generation"]
                    )
                    
                    assert result["query"] == "API Application Programming Interface"
                    assert len(result["documents"]) == 1
                    assert "answer" in result
                    
                    # All stages should be executed
                    mock_expand.assert_called_once()
                    mock_retriever_instance.retrieve.assert_called_once()
                    mock_generator_instance.generate.assert_called_once()


@pytest.mark.unit
class TestStreamingSupport:
    """Test streaming support in unified pipeline."""
    
    @pytest.mark.asyncio
    async def test_streaming_disabled(self):
        """Test pipeline with streaming disabled."""
        with patch('tldw_Server_API.app.core.RAG.rag_service.unified_pipeline.AnswerGenerator') as mock_generator:
            mock_generator_instance = MagicMock()
            mock_generator_instance.generate = AsyncMock(return_value={
                "answer": "Complete answer",
                "streaming": False
            })
            mock_generator.return_value = mock_generator_instance
            
            with patch('tldw_Server_API.app.core.RAG.rag_service.unified_pipeline.MultiDatabaseRetriever') as mock_retriever:
                mock_retriever_instance = MagicMock()
                mock_retriever_instance.retrieve = AsyncMock(return_value=[])
                mock_retriever.return_value = mock_retriever_instance
                
                result = await unified_rag_pipeline(
                    query="test",
                    enable_streaming=False
                )
                
                assert isinstance(result, dict)
                assert "answer" in result
                assert result["answer"] == "Complete answer"
    
    @pytest.mark.asyncio
    async def test_streaming_enabled(self):
        """Test pipeline with streaming enabled."""
        async def mock_stream_generator():
            """Mock streaming generator."""
            chunks = ["This ", "is ", "a ", "streamed ", "response."]
            for chunk in chunks:
                yield chunk
                await asyncio.sleep(0.01)
        
        with patch('tldw_Server_API.app.core.RAG.rag_service.unified_pipeline.AnswerGenerator') as mock_generator:
            mock_generator_instance = MagicMock()
            mock_generator_instance.generate_stream = mock_stream_generator
            mock_generator.return_value = mock_generator_instance
            
            with patch('tldw_Server_API.app.core.RAG.rag_service.unified_pipeline.MultiDatabaseRetriever') as mock_retriever:
                mock_retriever_instance = MagicMock()
                mock_retriever_instance.retrieve = AsyncMock(return_value=[])
                mock_retriever.return_value = mock_retriever_instance
                
                result = await unified_rag_pipeline(
                    query="test",
                    enable_streaming=True
                )
                
                # With streaming, result might be an async generator
                if hasattr(result, '__aiter__'):
                    chunks = []
                    async for chunk in result:
                        chunks.append(chunk)
                    
                    assert len(chunks) > 0
                    full_response = "".join(chunks)
                    assert full_response == "This is a streamed response."


if __name__ == "__main__":
    pytest.main([__file__, "-v"])