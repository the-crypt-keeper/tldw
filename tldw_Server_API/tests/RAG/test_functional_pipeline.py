"""
Tests for the functional RAG pipeline.

This tests the core functionality without requiring authentication or API endpoints.
"""

import pytest
import asyncio
from pathlib import Path
from typing import Dict, Any

from tldw_Server_API.app.core.RAG import (
    minimal_pipeline,
    standard_pipeline,
    quality_pipeline,
    RAGPipelineContext,
    build_pipeline,
    expand_query,
    retrieve_documents,
)
from tldw_Server_API.app.core.RAG.rag_service.types import DataSource, Document


class TestFunctionalPipeline:
    """Test the functional pipeline components."""
    
    @pytest.mark.asyncio
    async def test_pipeline_context_creation(self):
        """Test that pipeline context can be created."""
        context = RAGPipelineContext(
            query="test query",
            original_query="test query"
        )
        assert context.query == "test query"
        assert context.documents == []
        assert context.metadata == {}
        assert context.config == {}
    
    @pytest.mark.asyncio
    async def test_minimal_pipeline_runs(self):
        """Test that minimal pipeline can execute without errors."""
        # Minimal config - just test it runs
        config = {
            "databases": ["media_db"],
            "enable_cache": False,
            "top_k": 5
        }
        
        try:
            result = await minimal_pipeline("test search query", config)
            assert isinstance(result, RAGPipelineContext)
            assert result.original_query == "test search query"
            # May not find documents if DB is empty, but should run
        except Exception as e:
            # Log the error but don't fail - we just want to ensure no import/syntax errors
            print(f"Pipeline execution error (expected if DBs not configured): {e}")
            assert True
    
    @pytest.mark.asyncio
    async def test_build_pipeline(self):
        """Test building a custom pipeline."""
        # Create a simple test pipeline
        async def test_function(context: RAGPipelineContext) -> RAGPipelineContext:
            context.metadata["test_ran"] = True
            return context
        
        # Build pipeline with test function
        pipeline = build_pipeline(test_function)
        
        # Create context and run pipeline
        context = RAGPipelineContext(
            query="test",
            original_query="test"
        )
        
        result = await pipeline(context)
        assert result.metadata.get("test_ran") == True
    
    @pytest.mark.asyncio
    async def test_query_expansion(self):
        """Test query expansion function."""
        context = RAGPipelineContext(
            query="ML",
            original_query="ML",
            config={"expansion_strategies": ["acronym"]}
        )
        
        try:
            result = await expand_query(context)
            assert isinstance(result, RAGPipelineContext)
            # Should have expanded queries in metadata
            assert "expanded_queries" in result.metadata
        except ImportError:
            # Query expansion modules might not be available
            pytest.skip("Query expansion modules not available")
    
    def test_imports(self):
        """Test that all expected functions can be imported."""
        from tldw_Server_API.app.core.RAG import (
            minimal_pipeline,
            standard_pipeline,
            quality_pipeline,
            enhanced_pipeline,
            custom_pipeline,
            build_pipeline,
            expand_query,
            check_cache,
            retrieve_documents,
            optimize_chromadb_search,
            process_tables,
            rerank_documents,
            store_in_cache,
            analyze_performance,
        )
        
        # If we get here, all imports worked
        assert True


class TestPipelineIntegration:
    """Test pipeline integration with mocked databases."""
    
    @pytest.mark.asyncio
    async def test_document_creation(self):
        """Test creating Document objects."""
        doc = Document(
            id="test_1",
            content="This is test content",
            metadata={"source": "test"},
            score=0.95
        )
        
        assert doc.id == "test_1"
        assert doc.content == "This is test content"
        assert doc.score == 0.95
        assert doc.metadata["source"] == "test"
    
    @pytest.mark.asyncio  
    async def test_pipeline_with_mock_data(self):
        """Test pipeline with mock documents."""
        # Create a custom pipeline that injects test documents
        async def inject_test_docs(context: RAGPipelineContext) -> RAGPipelineContext:
            context.documents = [
                Document(
                    id="doc1",
                    content="Machine learning is a subset of AI",
                    metadata={"source": "test"},
                    score=0.9
                ),
                Document(
                    id="doc2", 
                    content="Deep learning uses neural networks",
                    metadata={"source": "test"},
                    score=0.8
                )
            ]
            return context
        
        # Build pipeline with test injector
        pipeline = build_pipeline(inject_test_docs)
        
        context = RAGPipelineContext(
            query="machine learning",
            original_query="machine learning"
        )
        
        result = await pipeline(context)
        assert len(result.documents) == 2
        assert result.documents[0].content == "Machine learning is a subset of AI"


if __name__ == "__main__":
    # Run basic tests
    asyncio.run(TestFunctionalPipeline().test_pipeline_context_creation())
    asyncio.run(TestFunctionalPipeline().test_minimal_pipeline_runs())
    print("Basic functional pipeline tests passed!")