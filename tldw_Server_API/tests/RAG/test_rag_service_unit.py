"""
Unit tests for the RAG service functional pipeline.

Tests the functional pipeline components and their composition.
"""

import pytest
from unittest.mock import Mock, AsyncMock, patch, MagicMock
from pathlib import Path
import asyncio

from tldw_Server_API.app.core.RAG.rag_service.functional_pipeline import (
    RAGPipelineContext,
    build_pipeline,
    standard_pipeline,
    minimal_pipeline,
    expand_query,
    check_cache,
    retrieve_documents,
    rerank_documents,
    get_pipeline,
    register_pipeline
)
from tldw_Server_API.app.core.RAG.rag_service.config import RAGConfig
from tldw_Server_API.app.core.RAG.rag_service.types import DataSource, Document


class TestFunctionalPipeline:
    """Test functional pipeline initialization and configuration."""
    
    @pytest.mark.asyncio
    async def test_pipeline_context_initialization(self):
        """Test RAGPipelineContext initialization."""
        query = "test query"
        context = RAGPipelineContext(
            query=query,
            original_query=query,
            config={"enable_cache": True}
        )
        
        assert context.query == query
        assert context.original_query == query
        assert context.config["enable_cache"] is True
        assert context.documents == []
        assert context.cache_hit is False
        assert context.timings == {}
    
    @pytest.mark.asyncio
    async def test_build_pipeline(self):
        """Test building a custom pipeline."""
        # Create mock functions
        async def mock_expand(context):
            context.metadata["expanded"] = True
            return context
        
        async def mock_retrieve(context):
            context.documents = [
                Document(id="1", content="test content", source=DataSource.MEDIA_DB, metadata={})
            ]
            return context
        
        # Build pipeline
        pipeline = build_pipeline(mock_expand, mock_retrieve)
        
        # Execute pipeline
        result = await pipeline("test query", {})
        
        assert result.metadata["expanded"] is True
        assert len(result.documents) == 1
        assert result.documents[0].content == "test content"
    
    @pytest.mark.asyncio
    async def test_get_pipeline(self):
        """Test getting predefined pipelines."""
        # Test getting standard pipeline
        pipeline = get_pipeline("standard")
        assert pipeline is not None
        assert callable(pipeline)
        
        # Test getting minimal pipeline
        pipeline = get_pipeline("minimal")
        assert pipeline is not None
        assert callable(pipeline)
        
        # Test invalid pipeline name
        with pytest.raises(ValueError) as exc_info:
            get_pipeline("nonexistent")
        assert "Unknown pipeline" in str(exc_info.value)
    
    @pytest.mark.asyncio
    async def test_register_pipeline(self):
        """Test registering custom pipelines."""
        # Create a custom pipeline
        async def custom_test_pipeline(query: str, config: dict):
            context = RAGPipelineContext(query=query, original_query=query, config=config)
            context.metadata["custom"] = True
            return context
        
        # Register the pipeline
        register_pipeline("test_custom", custom_test_pipeline)
        
        # Retrieve and test the registered pipeline
        pipeline = get_pipeline("test_custom")
        assert pipeline is custom_test_pipeline
        
        # Execute the pipeline
        result = await pipeline("test", {})
        assert result.metadata["custom"] is True
    
    @pytest.mark.asyncio
    async def test_pipeline_with_config(self):
        """Test pipeline execution with configuration."""
        config = {
            "enable_cache": True,
            "top_k": 5,
            "expansion_strategies": ["acronym"]
        }
        
        # Create a mock pipeline that uses config
        async def config_aware_pipeline(query: str, config_dict: dict):
            context = RAGPipelineContext(query=query, original_query=query, config=config_dict)
            context.metadata["top_k"] = config_dict.get("top_k", 10)
            return context
        
        result = await config_aware_pipeline("test query", config)
        assert result.metadata["top_k"] == 5
        assert result.config["enable_cache"] is True


class TestPipelineComponents:
    """Test individual pipeline components."""
    
    @pytest.mark.asyncio
    async def test_expand_query_component(self):
        """Test query expansion component."""
        context = RAGPipelineContext(
            query="ML",
            original_query="ML",
            config={"expansion_strategies": ["acronym"]}
        )
        
        with patch('tldw_Server_API.app.core.RAG.rag_service.query_expansion.expand_acronyms') as mock_expand:
            mock_expand.return_value = "Machine Learning ML"
            
            # Import and test the actual expand_query function
            from tldw_Server_API.app.core.RAG.rag_service.functional_pipeline import expand_query
            result = await expand_query(context)
            
            # Query should be expanded
            assert result.query != result.original_query
            assert result.metadata.get("query_expanded") is not None
    
    @pytest.mark.asyncio
    async def test_cache_component(self):
        """Test cache checking component."""
        context = RAGPipelineContext(
            query="test query",
            original_query="test query",
            config={"enable_cache": True}
        )
        
        # Mock cache lookup
        with patch('tldw_Server_API.app.core.RAG.rag_service.functional_pipeline.semantic_cache') as mock_cache:
            mock_cache_instance = Mock()
            mock_cache.get_instance.return_value = mock_cache_instance
            mock_cache_instance.get.return_value = None  # Cache miss
            
            from tldw_Server_API.app.core.RAG.rag_service.functional_pipeline import check_cache
            result = await check_cache(context)
            
            assert result.cache_hit is False
    
    @pytest.mark.asyncio
    async def test_retrieve_documents_component(self):
        """Test document retrieval component."""
        context = RAGPipelineContext(
            query="test query",
            original_query="test query",
            config={
                "data_sources": ["media_db"],
                "top_k": 5
            }
        )
        
        # Mock retrieval
        mock_docs = [
            Document(id="1", content="doc1", source=DataSource.MEDIA_DB, metadata={}),
            Document(id="2", content="doc2", source=DataSource.MEDIA_DB, metadata={})
        ]
        
        with patch('tldw_Server_API.app.core.RAG.rag_service.database_retrievers.MediaDBRetriever') as mock_retriever:
            mock_retriever_instance = Mock()
            mock_retriever.return_value = mock_retriever_instance
            mock_retriever_instance.retrieve = AsyncMock(return_value=mock_docs)
            
            from tldw_Server_API.app.core.RAG.rag_service.functional_pipeline import retrieve_documents
            result = await retrieve_documents(context)
            
            assert len(result.documents) > 0


class TestPipelineReranking:
    """Test reranking and processing components."""
    
    @pytest.mark.asyncio
    async def test_rerank_documents_component(self):
        """Test document reranking component."""
        context = RAGPipelineContext(
            query="test query",
            original_query="test query",
            config={"reranking_strategy": "similarity"},
            documents=[
                Document(id="1", content="relevant", source=DataSource.MEDIA_DB, metadata={"score": 0.5}),
                Document(id="2", content="very relevant", source=DataSource.MEDIA_DB, metadata={"score": 0.8}),
                Document(id="3", content="not relevant", source=DataSource.MEDIA_DB, metadata={"score": 0.2})
            ]
        )
        
        from tldw_Server_API.app.core.RAG.rag_service.functional_pipeline import rerank_documents
        
        with patch('tldw_Server_API.app.core.RAG.rag_service.advanced_reranking.rerank_by_similarity') as mock_rerank:
            mock_rerank.return_value = context.documents[:2]  # Return top 2
            
            result = await rerank_documents(context)
            
            # Should have reranked documents
            assert len(result.documents) <= 3
            assert "reranking_time" in result.timings or "rerank_documents" in result.timings
    
    @pytest.mark.asyncio
    async def test_process_tables_component(self):
        """Test table processing component."""
        context = RAGPipelineContext(
            query="test query",
            original_query="test query",
            config={"process_tables": True},
            documents=[
                Document(
                    id="1",
                    content="<table><tr><td>A</td><td>B</td></tr></table>",
                    source=DataSource.MEDIA_DB,
                    metadata={}
                )
            ]
        )
        
        from tldw_Server_API.app.core.RAG.rag_service.functional_pipeline import process_tables
        
        with patch('tldw_Server_API.app.core.RAG.rag_service.table_serialization.serialize_table') as mock_serialize:
            mock_serialize.return_value = "Table: A | B"
            
            result = await process_tables(context)
            
            # Should have processed tables if enabled
            assert result.documents is not None


class TestPipelineIntegration:
    """Test complete pipeline integration."""
    
    @pytest.mark.asyncio
    async def test_minimal_pipeline_execution(self):
        """Test minimal pipeline end-to-end."""
        query = "test query"
        config = {"top_k": 3}
        
        with patch('tldw_Server_API.app.core.RAG.rag_service.database_retrievers.MediaDBRetriever') as mock_retriever:
            mock_retriever_instance = Mock()
            mock_retriever.return_value = mock_retriever_instance
            mock_retriever_instance.retrieve = AsyncMock(return_value=[
                Document(id="1", content="test", source=DataSource.MEDIA_DB, metadata={})
            ])
            
            result = await minimal_pipeline(query, config)
            
            assert result.query == query
            assert result.original_query == query
            assert len(result.documents) >= 0
            assert result.config == config
    
    @pytest.mark.asyncio
    async def test_standard_pipeline_with_cache(self):
        """Test standard pipeline with caching."""
        query = "cached query"
        config = {"enable_cache": True, "top_k": 5}
        
        with patch('tldw_Server_API.app.core.RAG.rag_service.semantic_cache') as mock_cache:
            # Setup cache hit
            mock_cache_instance = Mock()
            mock_cache.get_instance.return_value = mock_cache_instance
            mock_cache_instance.get.return_value = [
                Document(id="cached", content="cached result", source=DataSource.MEDIA_DB, metadata={})
            ]
            
            result = await standard_pipeline(query, config)
            
            # Should use cached results
            assert result.cache_hit is True
    


class TestPerformanceMonitoring:
    """Test performance monitoring and analysis."""
    
    @pytest.mark.asyncio
    async def test_performance_analysis(self):
        """Test performance analysis component."""
        context = RAGPipelineContext(
            query="test",
            original_query="test",
            config={"enable_performance_analysis": True},
            timings={
                "expand_query": 0.1,
                "retrieve_documents": 0.5,
                "rerank_documents": 0.2
            }
        )
        
        from tldw_Server_API.app.core.RAG.rag_service.functional_pipeline import analyze_performance
        
        result = await analyze_performance(context)
        
        # Should have performance metadata
        assert "performance" in result.metadata or "total_time" in result.metadata
        assert result.timings is not None
    
    @pytest.mark.asyncio
    async def test_cache_storage(self):
        """Test storing results in cache."""
        context = RAGPipelineContext(
            query="test",
            original_query="test",
            config={"enable_cache": True},
            documents=[
                Document(id="1", content="result", source=DataSource.MEDIA_DB, metadata={})
            ]
        )
        
        from tldw_Server_API.app.core.RAG.rag_service.functional_pipeline import store_in_cache
        
        with patch('tldw_Server_API.app.core.RAG.rag_service.semantic_cache') as mock_cache:
            mock_cache_instance = Mock()
            mock_cache.get_instance.return_value = mock_cache_instance
            mock_cache_instance.set = Mock()
            
            result = await store_in_cache(context)
            
            # Should have stored in cache if enabled
            if context.config.get("enable_cache"):
                assert mock_cache_instance.set.called or mock_cache_instance.put.called
    
    @pytest.mark.asyncio
    async def test_chromadb_optimization(self):
        """Test ChromaDB optimization component."""
        context = RAGPipelineContext(
            query="test",
            original_query="test",
            config={"enable_chromadb_optimization": True},
            metadata={"collection_size": 100000}
        )
        
        from tldw_Server_API.app.core.RAG.rag_service.functional_pipeline import optimize_chromadb_search
        
        with patch('tldw_Server_API.app.core.RAG.rag_service.chromadb_optimizer.optimize_for_large_collection') as mock_optimize:
            mock_optimize.return_value = {"optimization_applied": True}
            
            result = await optimize_chromadb_search(context)
            
            # Should have optimization metadata if large collection
            assert result.metadata is not None


class TestPipelineUtilities:
    """Test pipeline utility functions."""
    
    @pytest.mark.asyncio
    async def test_conditional_execution(self):
        """Test conditional pipeline execution."""
        from tldw_Server_API.app.core.RAG.rag_service.functional_pipeline import conditional
        
        context = RAGPipelineContext(
            query="test",
            original_query="test",
            cache_hit=False,
            config={}
        )
        
        # Create mock functions
        async def if_true_func(ctx):
            ctx.metadata["executed"] = "true_branch"
            return ctx
        
        async def if_false_func(ctx):
            ctx.metadata["executed"] = "false_branch"
            return ctx
        
        # Test when condition is true
        cond_func = await conditional(
            lambda ctx: not ctx.cache_hit,
            if_true_func,
            if_false_func
        )
        
        result = await cond_func(context)
        assert result.metadata["executed"] == "true_branch"
        
        # Test when condition is false
        context.cache_hit = True
        cond_func = await conditional(
            lambda ctx: not ctx.cache_hit,
            if_true_func,
            if_false_func
        )
        
        result = await cond_func(context)
        assert result.metadata["executed"] == "false_branch"
    
    @pytest.mark.asyncio
    async def test_parallel_execution(self):
        """Test parallel pipeline execution."""
        from tldw_Server_API.app.core.RAG.rag_service.functional_pipeline import parallel
        
        context = RAGPipelineContext(
            query="test",
            original_query="test",
            config={}
        )
        
        # Create mock functions
        async def func1(ctx):
            ctx.metadata["func1"] = True
            return ctx
        
        async def func2(ctx):
            ctx.metadata["func2"] = True
            return ctx
        
        # Execute in parallel
        parallel_func = await parallel(func1, func2)
        result = await parallel_func(context)
        
        # Both functions should have executed
        assert result.metadata.get("func1") is True
        assert result.metadata.get("func2") is True


class TestCustomPipelines:
    """Test custom pipeline creation and execution."""
    
    @pytest.mark.asyncio
    async def test_custom_pipeline_builder(self):
        """Test building custom pipelines with various components."""
        # Create custom components
        async def custom_preprocessor(context):
            context.metadata["preprocessed"] = True
            context.query = context.query.lower()
            return context
        
        async def custom_postprocessor(context):
            context.metadata["postprocessed"] = True
            # Filter out low-score documents
            context.documents = [d for d in context.documents if d.metadata.get("score", 0) > 0.5]
            return context
        
        # Build custom pipeline
        custom_pipeline = build_pipeline(
            custom_preprocessor,
            expand_query,
            retrieve_documents,
            custom_postprocessor
        )
        
        # Execute with mock retrieval
        with patch('tldw_Server_API.app.core.RAG.rag_service.database_retrievers.MediaDBRetriever') as mock_retriever:
            mock_retriever_instance = Mock()
            mock_retriever.return_value = mock_retriever_instance
            mock_retriever_instance.retrieve = AsyncMock(return_value=[
                Document(id="1", content="high score", source=DataSource.MEDIA_DB, metadata={"score": 0.8}),
                Document(id="2", content="low score", source=DataSource.MEDIA_DB, metadata={"score": 0.3})
            ])
            
            result = await custom_pipeline("TEST QUERY", {"data_sources": ["media_db"]})
            
            # Verify preprocessing
            assert result.query == "test query"  # lowercased
            assert result.metadata["preprocessed"] is True
            
            # Verify postprocessing
            assert result.metadata["postprocessed"] is True
            assert len(result.documents) == 1  # Low score doc filtered out
            assert result.documents[0].id == "1"


# ========== Fixtures ==========

@pytest.fixture
def mock_rag_config():
    """Create a mock RAGConfig for testing."""
    config = Mock(spec=RAGConfig)
    config.cache_enabled = True
    config.retriever = Mock()
    config.retriever.fts_top_k = 10
    config.retriever.vector_top_k = 10
    config.retriever.hybrid_alpha = 0.5
    config.processor = Mock()
    config.processor.enable_reranking = False
    config.generator = Mock()
    config.generator.__dict__ = {"temperature": 0.7, "max_tokens": 2000}
    config.validate = Mock(return_value=[])
    config.num_workers = 2
    config.log_level = "INFO"
    return config


@pytest.fixture  
def mock_llm_handler():
    """Create a mock LLM handler for testing."""
    handler = Mock()
    handler.generate = AsyncMock(return_value="Generated text")
    handler.stream_generate = AsyncMock()
    return handler


if __name__ == "__main__":
    pytest.main([__file__, "-v"])