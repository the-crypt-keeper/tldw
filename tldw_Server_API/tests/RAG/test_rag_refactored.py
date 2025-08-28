"""
Comprehensive tests for the refactored RAG module with functional pipeline.
"""

import pytest
import asyncio
from typing import List, Dict, Any
from unittest.mock import Mock, AsyncMock, patch

from tldw_Server_API.app.core.RAG import (
    # Pipelines
    minimal_pipeline,
    standard_pipeline,
    quality_pipeline,
    build_pipeline,
    # Context
    RAGPipelineContext,
    # Functions
    expand_query,
    check_cache,
    retrieve_documents,
    rerank_documents,
    store_in_cache,
    analyze_performance,
    # Types
    DataSource,
    Document,
    RAGConfig
)


class TestPipelineContext:
    """Test the RAGPipelineContext."""
    
    def test_context_creation(self):
        """Test creating a pipeline context."""
        context = RAGPipelineContext(
            query="test query",
            original_query="test query"
        )
        
        assert context.query == "test query"
        assert context.original_query == "test query"
        assert context.documents == []
        assert context.metadata == {}
        assert context.config == {}
        assert context.cache_hit == False
        assert context.timings == {}
        assert context.errors == []
        assert context.resilience_enabled == False
    
    def test_context_with_config(self):
        """Test context with configuration."""
        config = {
            "enable_cache": True,
            "enable_resilience": True,
            "top_k": 10
        }
        
        context = RAGPipelineContext(
            query="test",
            original_query="test",
            config=config
        )
        
        assert context.config == config
        assert context.config["enable_cache"] == True
        assert context.config["enable_resilience"] == True


class TestPipelineFunctions:
    """Test individual pipeline functions."""
    
    @pytest.mark.asyncio
    async def test_minimal_pipeline(self):
        """Test minimal pipeline execution."""
        config = {
            "enable_cache": False,
            "databases": {}  # No databases configured
        }
        
        result = await minimal_pipeline("test query", config)
        
        assert isinstance(result, RAGPipelineContext)
        assert result.query == "test query"
        assert result.original_query == "test query"
        assert "retrieval" in result.timings
        assert "reranking" in result.timings
    
    @pytest.mark.asyncio
    async def test_standard_pipeline(self):
        """Test standard pipeline execution."""
        config = {
            "enable_cache": False,
            "expansion_strategies": ["acronym"],
            "databases": {}
        }
        
        result = await standard_pipeline("API test", config)
        
        assert isinstance(result, RAGPipelineContext)
        assert result.query == "API test"
        assert "query_expansion" in result.timings
        assert "cache_lookup" in result.timings
        assert "retrieval" in result.timings
        assert "reranking" in result.timings
    
    @pytest.mark.asyncio
    async def test_custom_pipeline_building(self):
        """Test building a custom pipeline."""
        # Define a simple test function
        async def test_function(context: RAGPipelineContext) -> RAGPipelineContext:
            context.metadata["test_ran"] = True
            context.metadata["test_value"] = 42
            return context
        
        # Build pipeline with test function
        pipeline = build_pipeline(test_function)
        
        # Execute pipeline
        context = RAGPipelineContext(
            query="test",
            original_query="test"
        )
        
        result = await pipeline(context)
        
        assert result.metadata["test_ran"] == True
        assert result.metadata["test_value"] == 42


class TestResilience:
    """Test resilience features."""
    
    @pytest.mark.asyncio
    async def test_resilience_disabled(self):
        """Test that resilience is disabled by default."""
        config = {
            "enable_resilience": False,
            "databases": {}
        }
        
        result = await minimal_pipeline("test", config)
        
        # Should complete without resilience features
        assert isinstance(result, RAGPipelineContext)
        assert result.resilience_enabled == False
    
    @pytest.mark.asyncio
    async def test_resilience_enabled(self):
        """Test enabling resilience features."""
        config = {
            "enable_resilience": True,
            "resilience": {
                "retry": {
                    "enabled": True,
                    "max_attempts": 2
                },
                "circuit_breaker": {
                    "enabled": True,
                    "failure_threshold": 3
                }
            },
            "databases": {}
        }
        
        # The pipeline should still work with resilience enabled
        result = await minimal_pipeline("test", config)
        
        assert isinstance(result, RAGPipelineContext)
    
    @pytest.mark.asyncio
    async def test_fallback_functions(self):
        """Test that fallback functions are defined."""
        context = RAGPipelineContext(
            query="test",
            original_query="test"
        )
        
        # Import fallback functions from the module
        from tldw_Server_API.app.core.RAG.rag_service.functional_pipeline import (
            expand_query_fallback,
            retrieve_documents_fallback
        )
        
        # Test expand query fallback
        result = await expand_query_fallback(context)
        assert result.metadata["expansion_failed"] == True
        assert result.metadata["expanded_queries"] == []
        
        # Test retrieve documents fallback
        result = await retrieve_documents_fallback(context)
        assert result.documents == []
        assert result.metadata["retrieval_failed"] == True


class TestCaching:
    """Test caching functionality."""
    
    @pytest.mark.asyncio
    async def test_cache_disabled(self):
        """Test with cache disabled."""
        config = {
            "enable_cache": False,
            "databases": {}
        }
        
        result = await standard_pipeline("test", config)
        
        assert result.cache_hit == False
        assert "cache_lookup" in result.timings
    
    @pytest.mark.asyncio
    async def test_cache_check_function(self):
        """Test cache check function."""
        context = RAGPipelineContext(
            query="test",
            original_query="test",
            config={"enable_cache": False}
        )
        
        result = await check_cache(context)
        
        # With cache disabled, should just return context
        assert result.cache_hit == False


class TestPerformanceMonitoring:
    """Test performance monitoring features."""
    
    @pytest.mark.asyncio
    async def test_performance_analysis(self):
        """Test performance analysis function."""
        context = RAGPipelineContext(
            query="test",
            original_query="test",
            config={"enable_monitoring": True}
        )
        
        # Add some fake timings
        context.timings = {
            "retrieval": 0.5,
            "reranking": 0.2,
            "cache_lookup": 0.1
        }
        
        result = await analyze_performance(context)
        
        assert result.metadata["total_time"] > 0
        assert result.metadata["performance_analyzed"] == True
    
    @pytest.mark.asyncio
    async def test_timing_decorator(self):
        """Test that timing decorator works."""
        from tldw_Server_API.app.core.RAG.rag_service.functional_pipeline import timer
        
        @timer("test_function")
        async def timed_function(context: RAGPipelineContext) -> RAGPipelineContext:
            await asyncio.sleep(0.01)  # Small delay
            context.metadata["executed"] = True
            return context
        
        context = RAGPipelineContext(
            query="test",
            original_query="test"
        )
        
        result = await timed_function(context)
        
        assert "test_function" in result.timings
        assert result.timings["test_function"] > 0
        assert result.metadata["executed"] == True


class TestIntegration:
    """Integration tests for the complete RAG system."""
    
    @pytest.mark.asyncio
    async def test_pipeline_composition(self):
        """Test composing multiple functions into a pipeline."""
        # Create test functions
        async def add_metadata(context: RAGPipelineContext) -> RAGPipelineContext:
            context.metadata["step1"] = "completed"
            return context
        
        async def modify_query(context: RAGPipelineContext) -> RAGPipelineContext:
            context.query = context.query.upper()
            context.metadata["step2"] = "completed"
            return context
        
        # Build and execute pipeline
        pipeline = build_pipeline(add_metadata, modify_query)
        
        context = RAGPipelineContext(
            query="test",
            original_query="test"
        )
        
        result = await pipeline(context)
        
        assert result.query == "TEST"
        assert result.original_query == "test"
        assert result.metadata["step1"] == "completed"
        assert result.metadata["step2"] == "completed"
    
    @pytest.mark.asyncio
    async def test_error_handling(self):
        """Test error handling in pipelines."""
        async def failing_function(context: RAGPipelineContext) -> RAGPipelineContext:
            raise ValueError("Test error")
        
        pipeline = build_pipeline(failing_function)
        
        context = RAGPipelineContext(
            query="test",
            original_query="test"
        )
        
        with pytest.raises(ValueError):
            await pipeline(context)
    
    @pytest.mark.asyncio
    async def test_document_object(self):
        """Test Document object creation."""
        doc = Document(
            id="test_1",
            content="Test content",
            source=DataSource.MEDIA_DB,
            metadata={"key": "value"},
            score=0.95
        )
        
        assert doc.id == "test_1"
        assert doc.content == "Test content"
        assert doc.source == DataSource.MEDIA_DB
        assert doc.metadata["key"] == "value"
        assert doc.score == 0.95


if __name__ == "__main__":
    # Run basic tests
    pytest.main([__file__, "-v"])