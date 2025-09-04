"""
Unit tests for the RAG functional pipeline.

Tests individual pipeline functions with mocked dependencies.
"""

import pytest
from unittest.mock import Mock, AsyncMock, patch, MagicMock
import time
from typing import List, Dict, Any

from tldw_Server_API.app.core.RAG.rag_service.functional_pipeline import (
    RAGPipelineContext,
    timer,
    # Pipeline building functions
    minimal_pipeline,
    standard_pipeline,
    quality_pipeline,
    build_pipeline,
    compose_pipeline,
    # Core pipeline functions
    expand_query,
    check_cache,
    retrieve_documents,
    rerank_documents,
    generate_answer,
    store_in_cache,
    # Analysis functions
    analyze_performance,
    collect_metrics
)
from tldw_Server_API.app.core.RAG.rag_service.types import Document, SearchResult, DataSource


@pytest.mark.unit
class TestPipelineContext:
    """Test RAGPipelineContext functionality."""
    
    def test_context_initialization(self):
        """Test creating a new pipeline context."""
        context = RAGPipelineContext(
            query="test query",
            original_query="test query"
        )
        
        assert context.query == "test query"
        assert context.original_query == "test query"
        assert context.documents == []
        assert context.metadata == {}
        assert context.config == {}
        assert context.cache_hit is False
        assert context.timings == {}
        assert context.errors == []
    
    def test_context_with_config(self):
        """Test context with configuration."""
        config = {
            "enable_cache": True,
            "top_k": 10,
            "temperature": 0.7
        }
        
        context = RAGPipelineContext(
            query="test",
            original_query="test",
            config=config
        )
        
        assert context.config == config
        assert context.config["enable_cache"] is True
        assert context.config["top_k"] == 10
    
    def test_context_modification(self):
        """Test modifying context during pipeline."""
        context = RAGPipelineContext(
            query="original",
            original_query="original"
        )
        
        # Modify query (expansion)
        context.query = "original expanded"
        assert context.query != context.original_query
        
        # Add documents
        doc = Document(id="1", content="test", metadata={})
        context.documents.append(doc)
        assert len(context.documents) == 1
        
        # Add timings
        context.timings["retrieval"] = 0.05
        assert "retrieval" in context.timings
        
        # Add errors
        context.errors.append({"function": "test", "error": "test error"})
        assert len(context.errors) == 1


@pytest.mark.unit
class TestTimerDecorator:
    """Test the timer decorator functionality."""
    
    @pytest.mark.asyncio
    async def test_timer_decorator(self):
        """Test timer decorator records timing."""
        context = RAGPipelineContext(
            query="test",
            original_query="test"
        )
        
        @timer("test_function")
        async def test_func(ctx: RAGPipelineContext):
            await asyncio.sleep(0.01)
            return "result"
        
        import asyncio
        result = await test_func(context)
        
        assert "test_function" in context.timings
        assert context.timings["test_function"] >= 0.01
        assert result == "result"
    
    @pytest.mark.asyncio
    async def test_timer_with_error(self):
        """Test timer decorator handles errors."""
        context = RAGPipelineContext(
            query="test",
            original_query="test"
        )
        
        @timer("error_function")
        async def error_func(ctx: RAGPipelineContext):
            raise ValueError("Test error")
        
        with pytest.raises(ValueError):
            await error_func(context)
        
        assert "error_function" in context.timings
        assert len(context.errors) == 1
        assert context.errors[0]["function"] == "error_function"
        assert "Test error" in context.errors[0]["error"]
    
    @pytest.mark.asyncio
    async def test_timer_with_metrics(self, query_metrics):
        """Test timer updates metrics object."""
        context = RAGPipelineContext(
            query="test",
            original_query="test",
            metrics=query_metrics
        )
        
        @timer("retrieval")
        async def retrieval_func(ctx: RAGPipelineContext):
            await asyncio.sleep(0.01)
            return []
        
        import asyncio
        await retrieval_func(context)
        
        assert context.metrics.retrieval_time >= 0.01


@pytest.mark.unit
class TestQueryExpansion:
    """Test query expansion functionality."""
    
    @pytest.mark.asyncio
    async def test_expand_query_disabled(self):
        """Test query expansion when disabled."""
        context = RAGPipelineContext(
            query="API test",
            original_query="API test",
            config={"enable_expansion": False}
        )
        
        with patch('tldw_Server_API.app.core.RAG.rag_service.functional_pipeline.multi_strategy_expansion') as mock_expand:
            result = await expand_query(context)
            
            assert result == context
            assert context.query == "API test"
            mock_expand.assert_not_called()
    
    @pytest.mark.asyncio
    async def test_expand_query_enabled(self):
        """Test query expansion when enabled."""
        context = RAGPipelineContext(
            query="API",
            original_query="API",
            config={
                "enable_expansion": True,
                "expansion_strategies": ["acronym"]
            }
        )
        
        with patch('tldw_Server_API.app.core.RAG.rag_service.functional_pipeline.multi_strategy_expansion') as mock_expand:
            mock_expand.return_value = "API Application Programming Interface"
            
            result = await expand_query(context)
            
            assert result == context
            assert context.query == "API Application Programming Interface"
            mock_expand.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_expand_query_error_handling(self):
        """Test query expansion error handling."""
        context = RAGPipelineContext(
            query="test",
            original_query="test",
            config={"enable_expansion": True}
        )
        
        with patch('tldw_Server_API.app.core.RAG.rag_service.functional_pipeline.multi_strategy_expansion') as mock_expand:
            mock_expand.side_effect = Exception("Expansion failed")
            
            # Should not raise, but log error
            result = await expand_query(context)
            
            assert result == context
            assert context.query == "test"  # Unchanged
            assert len(context.errors) > 0


@pytest.mark.unit
class TestCacheFunctions:
    """Test cache-related functions."""
    
    @pytest.mark.asyncio
    async def test_check_cache_disabled(self):
        """Test cache check when disabled."""
        context = RAGPipelineContext(
            query="test",
            original_query="test",
            config={"enable_cache": False}
        )
        
        result = await check_cache(context)
        
        assert result == context
        assert context.cache_hit is False
        assert len(context.documents) == 0
    
    @pytest.mark.asyncio
    async def test_check_cache_miss(self, mock_semantic_cache):
        """Test cache miss."""
        context = RAGPipelineContext(
            query="test",
            original_query="test",
            config={"enable_cache": True}
        )
        
        mock_semantic_cache.get.return_value = None
        
        with patch('tldw_Server_API.app.core.RAG.rag_service.functional_pipeline.get_cache', return_value=mock_semantic_cache):
            result = await check_cache(context)
            
            assert result == context
            assert context.cache_hit is False
            assert len(context.documents) == 0
            mock_semantic_cache.get.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_check_cache_hit(self, mock_semantic_cache, sample_documents):
        """Test cache hit."""
        context = RAGPipelineContext(
            query="test",
            original_query="test",
            config={"enable_cache": True}
        )
        
        mock_semantic_cache.get.return_value = sample_documents
        
        with patch('tldw_Server_API.app.core.RAG.rag_service.functional_pipeline.get_cache', return_value=mock_semantic_cache):
            result = await check_cache(context)
            
            assert result == context
            assert context.cache_hit is True
            assert len(context.documents) == len(sample_documents)
            assert context.documents == sample_documents
    
    @pytest.mark.asyncio
    async def test_store_in_cache(self, mock_semantic_cache, sample_documents):
        """Test storing results in cache."""
        context = RAGPipelineContext(
            query="test",
            original_query="test",
            config={"enable_cache": True},
            documents=sample_documents
        )
        
        with patch('tldw_Server_API.app.core.RAG.rag_service.functional_pipeline.get_cache', return_value=mock_semantic_cache):
            result = await store_in_cache(context)
            
            assert result == context
            mock_semantic_cache.set.assert_called_once_with(
                "test",
                sample_documents,
                ttl=3600  # Default TTL
            )


@pytest.mark.unit
class TestDocumentRetrieval:
    """Test document retrieval functionality."""
    
    @pytest.mark.asyncio
    async def test_retrieve_documents_basic(self, mock_multi_db_retriever):
        """Test basic document retrieval."""
        context = RAGPipelineContext(
            query="RAG systems",
            original_query="RAG systems",
            config={"top_k": 5}
        )
        
        with patch('tldw_Server_API.app.core.RAG.rag_service.functional_pipeline.get_retriever', return_value=mock_multi_db_retriever):
            result = await retrieve_documents(context)
            
            assert result == context
            assert len(context.documents) > 0
            mock_multi_db_retriever.retrieve.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_retrieve_documents_with_filters(self, mock_multi_db_retriever):
        """Test document retrieval with filters."""
        context = RAGPipelineContext(
            query="test",
            original_query="test",
            config={
                "top_k": 10,
                "filter_source": "media_db",
                "date_range": {"start": "2024-01-01", "end": "2024-12-31"}
            }
        )
        
        with patch('tldw_Server_API.app.core.RAG.rag_service.functional_pipeline.get_retriever', return_value=mock_multi_db_retriever):
            result = await retrieve_documents(context)
            
            assert result == context
            # Verify filters were passed
            call_args = mock_multi_db_retriever.retrieve.call_args
            assert "filter_source" in call_args[1] or "filter_source" in context.config
    
    @pytest.mark.asyncio
    async def test_retrieve_documents_error_handling(self):
        """Test retrieval error handling."""
        context = RAGPipelineContext(
            query="test",
            original_query="test",
            config={"top_k": 5}
        )
        
        mock_retriever = MagicMock()
        mock_retriever.retrieve = AsyncMock(side_effect=Exception("Retrieval failed"))
        
        with patch('tldw_Server_API.app.core.RAG.rag_service.functional_pipeline.get_retriever', return_value=mock_retriever):
            # Should handle error gracefully
            result = await retrieve_documents(context)
            
            assert result == context
            assert len(context.documents) == 0
            assert len(context.errors) > 0


@pytest.mark.unit
class TestReranking:
    """Test document reranking functionality."""
    
    @pytest.mark.asyncio
    async def test_rerank_documents_disabled(self, sample_documents):
        """Test reranking when disabled."""
        context = RAGPipelineContext(
            query="test",
            original_query="test",
            config={"enable_reranking": False},
            documents=sample_documents
        )
        
        original_order = [doc.id for doc in sample_documents]
        
        result = await rerank_documents(context)
        
        assert result == context
        assert [doc.id for doc in context.documents] == original_order
    
    @pytest.mark.asyncio
    async def test_rerank_documents_enabled(self, sample_documents):
        """Test reranking when enabled."""
        context = RAGPipelineContext(
            query="vector databases",
            original_query="vector databases",
            config={
                "enable_reranking": True,
                "reranking_strategy": "cross_encoder",
                "rerank_top_k": 2
            },
            documents=sample_documents
        )
        
        mock_reranker = MagicMock()
        mock_reranker.rerank = AsyncMock(return_value=[
            sample_documents[1],  # Vector database doc should rank higher
            sample_documents[0]
        ])
        
        with patch('tldw_Server_API.app.core.RAG.rag_service.functional_pipeline.get_reranker', return_value=mock_reranker):
            result = await rerank_documents(context)
            
            assert result == context
            assert len(context.documents) == 2  # Top k=2
            assert context.documents[0].id == "2"  # Vector database doc
            mock_reranker.rerank.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_rerank_empty_documents(self):
        """Test reranking with no documents."""
        context = RAGPipelineContext(
            query="test",
            original_query="test",
            config={"enable_reranking": True},
            documents=[]
        )
        
        result = await rerank_documents(context)
        
        assert result == context
        assert len(context.documents) == 0


@pytest.mark.unit
class TestAnswerGeneration:
    """Test answer generation functionality."""
    
    @pytest.mark.asyncio
    async def test_generate_answer_basic(self, sample_documents, mock_llm):
        """Test basic answer generation."""
        context = RAGPipelineContext(
            query="What is RAG?",
            original_query="What is RAG?",
            documents=sample_documents,
            config={"temperature": 0.7, "max_tokens": 500}
        )
        
        with patch('tldw_Server_API.app.core.RAG.rag_service.functional_pipeline.get_llm', return_value=mock_llm):
            result = await generate_answer(context)
            
            assert result == context
            assert "answer" in context.metadata
            assert context.metadata["answer"] == mock_llm.generate.return_value
            mock_llm.generate.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_generate_answer_no_documents(self, mock_llm):
        """Test answer generation with no documents."""
        context = RAGPipelineContext(
            query="test",
            original_query="test",
            documents=[],
            config={}
        )
        
        mock_llm.generate.return_value = "I don't have enough information to answer."
        
        with patch('tldw_Server_API.app.core.RAG.rag_service.functional_pipeline.get_llm', return_value=mock_llm):
            result = await generate_answer(context)
            
            assert result == context
            assert "answer" in context.metadata
            # Should still attempt generation, but with no context
            mock_llm.generate.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_generate_answer_with_citations(self, sample_documents, mock_llm):
        """Test answer generation with citations."""
        context = RAGPipelineContext(
            query="test",
            original_query="test",
            documents=sample_documents,
            config={"enable_citations": True}
        )
        
        mock_llm.generate.return_value = "Answer with citation [1]."
        
        with patch('tldw_Server_API.app.core.RAG.rag_service.functional_pipeline.get_llm', return_value=mock_llm):
            with patch('tldw_Server_API.app.core.RAG.rag_service.functional_pipeline.add_citations') as mock_citations:
                mock_citations.return_value = "Answer with citation [1] (Source: article)."
                
                result = await generate_answer(context)
                
                assert result == context
                assert "answer" in context.metadata
                mock_citations.assert_called_once()


@pytest.mark.unit
class TestPipelineBuilding:
    """Test pipeline building and composition."""
    
    @pytest.mark.asyncio
    async def test_minimal_pipeline(self):
        """Test minimal pipeline execution."""
        config = {
            "enable_cache": False,
            "enable_expansion": False,
            "enable_reranking": False,
            "top_k": 5
        }
        
        with patch('tldw_Server_API.app.core.RAG.rag_service.functional_pipeline.retrieve_documents') as mock_retrieve:
            with patch('tldw_Server_API.app.core.RAG.rag_service.functional_pipeline.generate_answer') as mock_generate:
                mock_retrieve.side_effect = lambda ctx: ctx
                mock_generate.side_effect = lambda ctx: ctx
                
                result = await minimal_pipeline("test query", config)
                
                assert isinstance(result, RAGPipelineContext)
                assert result.query == "test query"
                mock_retrieve.assert_called_once()
                mock_generate.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_standard_pipeline(self):
        """Test standard pipeline with cache and expansion."""
        config = {
            "enable_cache": True,
            "enable_expansion": True,
            "enable_reranking": False,
            "top_k": 10
        }
        
        with patch('tldw_Server_API.app.core.RAG.rag_service.functional_pipeline.check_cache') as mock_cache:
            with patch('tldw_Server_API.app.core.RAG.rag_service.functional_pipeline.expand_query') as mock_expand:
                with patch('tldw_Server_API.app.core.RAG.rag_service.functional_pipeline.retrieve_documents') as mock_retrieve:
                    with patch('tldw_Server_API.app.core.RAG.rag_service.functional_pipeline.generate_answer') as mock_generate:
                        # Setup mocks
                        mock_cache.side_effect = lambda ctx: ctx
                        mock_expand.side_effect = lambda ctx: ctx
                        mock_retrieve.side_effect = lambda ctx: ctx
                        mock_generate.side_effect = lambda ctx: ctx
                        
                        result = await standard_pipeline("test query", config)
                        
                        assert isinstance(result, RAGPipelineContext)
                        mock_cache.assert_called_once()
                        mock_expand.assert_called_once()
                        mock_retrieve.assert_called_once()
                        mock_generate.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_quality_pipeline(self):
        """Test quality pipeline with all features."""
        config = {
            "enable_cache": True,
            "enable_expansion": True,
            "enable_reranking": True,
            "enable_analysis": True,
            "top_k": 20,
            "rerank_top_k": 5
        }
        
        # Mock all pipeline functions
        with patch('tldw_Server_API.app.core.RAG.rag_service.functional_pipeline.check_cache') as mock_cache:
            with patch('tldw_Server_API.app.core.RAG.rag_service.functional_pipeline.expand_query') as mock_expand:
                with patch('tldw_Server_API.app.core.RAG.rag_service.functional_pipeline.retrieve_documents') as mock_retrieve:
                    with patch('tldw_Server_API.app.core.RAG.rag_service.functional_pipeline.rerank_documents') as mock_rerank:
                        with patch('tldw_Server_API.app.core.RAG.rag_service.functional_pipeline.generate_answer') as mock_generate:
                            with patch('tldw_Server_API.app.core.RAG.rag_service.functional_pipeline.store_in_cache') as mock_store:
                                with patch('tldw_Server_API.app.core.RAG.rag_service.functional_pipeline.analyze_performance') as mock_analyze:
                                    # Setup mocks
                                    for mock in [mock_cache, mock_expand, mock_retrieve, mock_rerank, 
                                               mock_generate, mock_store, mock_analyze]:
                                        mock.side_effect = lambda ctx: ctx
                                    
                                    result = await quality_pipeline("test query", config)
                                    
                                    assert isinstance(result, RAGPipelineContext)
                                    mock_cache.assert_called_once()
                                    mock_expand.assert_called_once()
                                    mock_retrieve.assert_called_once()
                                    mock_rerank.assert_called_once()
                                    mock_generate.assert_called_once()
                                    mock_analyze.assert_called_once()
    
    def test_build_pipeline_custom(self):
        """Test building custom pipeline."""
        config = {
            "pipeline_functions": ["expand_query", "retrieve_documents", "generate_answer"]
        }
        
        pipeline = build_pipeline(config)
        
        assert callable(pipeline)
        assert len(pipeline.__wrapped__) == 3  # Should compose 3 functions
    
    def test_compose_pipeline(self):
        """Test pipeline composition."""
        async def func1(ctx):
            ctx.metadata["func1"] = True
            return ctx
        
        async def func2(ctx):
            ctx.metadata["func2"] = True
            return ctx
        
        async def func3(ctx):
            ctx.metadata["func3"] = True
            return ctx
        
        composed = compose_pipeline([func1, func2, func3])
        
        assert callable(composed)
        
        # Test execution
        import asyncio
        context = RAGPipelineContext("test", "test")
        result = asyncio.run(composed(context))
        
        assert result.metadata.get("func1") is True
        assert result.metadata.get("func2") is True
        assert result.metadata.get("func3") is True


@pytest.mark.unit
class TestPerformanceAnalysis:
    """Test performance analysis functions."""
    
    @pytest.mark.asyncio
    async def test_analyze_performance(self):
        """Test performance analysis."""
        context = RAGPipelineContext(
            query="test",
            original_query="test",
            documents=[Document(id="1", content="test", metadata={})],
            timings={
                "expansion": 0.01,
                "retrieval": 0.05,
                "reranking": 0.02,
                "generation": 0.1
            }
        )
        
        result = await analyze_performance(context)
        
        assert result == context
        assert "performance_analysis" in context.metadata
        
        analysis = context.metadata["performance_analysis"]
        assert "total_time" in analysis
        assert "bottlenecks" in analysis
        assert analysis["total_time"] == pytest.approx(0.18, rel=0.01)
    
    @pytest.mark.asyncio
    async def test_collect_metrics(self, metrics_collector):
        """Test metrics collection."""
        context = RAGPipelineContext(
            query="test",
            original_query="test",
            documents=[Document(id="1", content="test", metadata={})],
            cache_hit=False,
            timings={
                "retrieval": 0.05,
                "generation": 0.1
            },
            metrics_collector=metrics_collector
        )
        
        result = await collect_metrics(context)
        
        assert result == context
        assert context.metrics is not None
        assert context.metrics.query == "test"
        assert context.metrics.retrieval_time == 0.05
        assert context.metrics.generation_time == 0.1
        assert context.metrics.num_documents_retrieved == 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])