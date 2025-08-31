"""
Integration tests for RAG search API with contextual retrieval.

Tests the full RAG pipeline with parent expansion and sibling chunks.
"""

import pytest
from unittest.mock import Mock, MagicMock, patch, AsyncMock
from fastapi.testclient import TestClient
from fastapi import FastAPI
import json
import asyncio
from pathlib import Path

from tldw_Server_API.app.api.v1.endpoints.rag_api import router as rag_router
from tldw_Server_API.app.core.RAG.rag_service.types import Document, DataSource


class TestRAGContextualSearchIntegration:
    """Integration tests for RAG search with contextual retrieval."""
    
    @pytest.fixture
    def test_app(self):
        """Create a test FastAPI app with RAG router."""
        app = FastAPI()
        app.include_router(rag_router)
        return app
    
    @pytest.fixture
    def test_client(self, test_app):
        """Create a test client."""
        return TestClient(test_app)
    
    @pytest.fixture
    def mock_dependencies(self):
        """Mock all required dependencies."""
        with patch('tldw_Server_API.app.api.v1.endpoints.rag_api.get_request_user') as mock_user:
            with patch('tldw_Server_API.app.api.v1.endpoints.rag_api.get_media_db_for_user') as mock_media_db:
                with patch('tldw_Server_API.app.api.v1.endpoints.rag_api.get_chacha_db_for_user') as mock_chacha_db:
                    mock_user.return_value = Mock(id="test_user")
                    mock_media_db.return_value = Mock(db_path="/test/media.db")
                    mock_chacha_db.return_value = Mock(db_path="/test/chacha.db")
                    yield {
                        "user": mock_user,
                        "media_db": mock_media_db,
                        "chacha_db": mock_chacha_db
                    }
    
    @pytest.fixture
    def mock_pipeline_results(self):
        """Create mock pipeline results with parent-child relationships."""
        return [
            Document(
                id="doc1_chunk_0",
                content="Machine learning is a subset of artificial intelligence.",
                metadata={
                    "parent_id": "doc1",
                    "chunk_index": 0,
                    "chunk_type": "text",
                    "title": "Introduction to ML"
                },
                source=DataSource.MEDIA_DB,
                score=0.95
            ),
            Document(
                id="doc1_chunk_1",
                content="Neural networks are inspired by biological neurons.",
                metadata={
                    "parent_id": "doc1",
                    "chunk_index": 1,
                    "chunk_type": "text",
                    "title": "Introduction to ML"
                },
                source=DataSource.MEDIA_DB,
                score=0.90
            ),
            Document(
                id="doc2_chunk_0",
                content="Deep learning uses multiple layers of neural networks.",
                metadata={
                    "parent_id": "doc2",
                    "chunk_index": 0,
                    "chunk_type": "text",
                    "title": "Deep Learning Guide"
                },
                source=DataSource.MEDIA_DB,
                score=0.85
            )
        ]
    
    @pytest.mark.asyncio
    async def test_simple_search_with_contextual_retrieval(self, test_client, mock_dependencies, mock_pipeline_results):
        """Test simple search API with contextual retrieval enabled."""
        request_data = {
            "query": "What is machine learning?",
            "databases": ["media"],
            "enable_contextual_retrieval": True,
            "parent_expansion_size": 500,
            "include_sibling_chunks": True,
            "enable_reranking": True,
            "top_k": 10
        }
        
        with patch('tldw_Server_API.app.api.v1.endpoints.rag_api.build_pipeline') as mock_build:
            # Create a mock pipeline that returns our test results
            mock_pipeline = AsyncMock()
            mock_context = Mock()
            mock_context.documents = mock_pipeline_results
            mock_context.metadata = {
                "parent_expansion_applied": True,
                "query_expanded": True,
                "reranking_applied": True
            }
            mock_context.cache_hit = False
            mock_pipeline.return_value = mock_context
            mock_build.return_value = mock_pipeline
            
            response = test_client.post(
                "/api/v1/rag/simple",
                json=request_data
            )
            
            assert response.status_code == 200
            result = response.json()
            
            # Verify response structure
            assert "query" in result
            assert "results" in result
            assert len(result["results"]) > 0
            
            # Verify contextual retrieval was configured
            mock_build.assert_called_once()
            call_args = mock_build.call_args[0]
            
            # Check that expand_with_parent_context is in the pipeline
            from tldw_Server_API.app.core.RAG.rag_service.enhanced_chunking_integration import expand_with_parent_context
            assert expand_with_parent_context in call_args
    
    @pytest.mark.asyncio
    async def test_simple_search_without_contextual_retrieval(self, test_client, mock_dependencies, mock_pipeline_results):
        """Test simple search API with contextual retrieval disabled."""
        request_data = {
            "query": "What is machine learning?",
            "databases": ["media"],
            "enable_contextual_retrieval": False,  # Disabled
            "enable_reranking": True,
            "top_k": 10
        }
        
        with patch('tldw_Server_API.app.api.v1.endpoints.rag_api.build_pipeline') as mock_build:
            mock_pipeline = AsyncMock()
            mock_context = Mock()
            mock_context.documents = mock_pipeline_results
            mock_context.metadata = {"reranking_applied": True}
            mock_context.cache_hit = False
            mock_pipeline.return_value = mock_context
            mock_build.return_value = mock_pipeline
            
            response = test_client.post(
                "/api/v1/rag/simple",
                json=request_data
            )
            
            assert response.status_code == 200
            
            # Verify expand_with_parent_context is NOT in the pipeline
            mock_build.assert_called_once()
            call_args = mock_build.call_args[0]
            
            from tldw_Server_API.app.core.RAG.rag_service.enhanced_chunking_integration import expand_with_parent_context
            assert expand_with_parent_context not in call_args
    
    @pytest.mark.asyncio
    async def test_complex_search_with_contextual_retrieval(self, test_client, mock_dependencies, mock_pipeline_results):
        """Test complex search API with contextual retrieval configuration."""
        request_data = {
            "query": "Explain neural networks",
            "databases": {
                "media": {"enabled": True, "weight": 1.0},
                "notes": {"enabled": False}
            },
            "processing": {
                "max_context_size": 10000,
                "contextual_retrieval": {
                    "enabled": True,
                    "parent_expansion_size": 750,
                    "include_siblings": True,
                    "context_window_size": 600
                }
            },
            "reranking": {
                "enabled": True,
                "strategy": "hybrid",
                "top_k": 5
            }
        }
        
        with patch('tldw_Server_API.app.api.v1.endpoints.rag_api.build_pipeline') as mock_build:
            mock_pipeline = AsyncMock()
            mock_context = Mock()
            mock_context.documents = mock_pipeline_results[:2]  # Return fewer docs
            mock_context.metadata = {
                "parent_expansion_applied": True,
                "performance_metrics": {"total_time": 0.5}
            }
            mock_context.cache_hit = False
            mock_context.query_expansion = ["neural", "networks", "brain"]
            mock_pipeline.return_value = mock_context
            mock_build.return_value = mock_pipeline
            
            response = test_client.post(
                "/api/v1/rag/complex",
                json=request_data
            )
            
            assert response.status_code == 200
            result = response.json()
            
            # Verify complex response structure
            assert "request_id" in result
            assert "results" in result
            assert "metadata" in result
            
            # Verify contextual retrieval configuration was passed
            pipeline_call = mock_pipeline.call_args
            if pipeline_call:
                config = pipeline_call[0][1]  # Second argument is config
                assert config.get("expand_parent_context") == True
                assert config.get("parent_expansion_size") == 750
                assert config.get("include_siblings") == True
    
    @pytest.mark.asyncio
    async def test_contextual_retrieval_with_citations(self, test_client, mock_dependencies, mock_pipeline_results):
        """Test that citations work with contextual retrieval."""
        request_data = {
            "query": "What is deep learning?",
            "databases": ["media"],
            "enable_contextual_retrieval": True,
            "include_sibling_chunks": True,
            "enable_citations": True
        }
        
        with patch('tldw_Server_API.app.api.v1.endpoints.rag_api.build_pipeline') as mock_build:
            mock_pipeline = AsyncMock()
            mock_context = Mock()
            mock_context.documents = mock_pipeline_results
            mock_context.metadata = {"parent_expansion_applied": True}
            mock_context.cache_hit = False
            mock_pipeline.return_value = mock_context
            mock_build.return_value = mock_pipeline
            
            response = test_client.post(
                "/api/v1/rag/simple",
                json=request_data
            )
            
            assert response.status_code == 200
            result = response.json()
            
            # Check that citations are included
            assert all("citation" in r for r in result["results"])
            
            # Verify citation structure
            first_citation = result["results"][0]["citation"]
            assert "source_id" in first_citation
            assert "source_type" in first_citation
            assert "title" in first_citation
    
    @pytest.mark.asyncio
    async def test_contextual_retrieval_performance_tracking(self, test_client, mock_dependencies, mock_pipeline_results):
        """Test that performance metrics track contextual retrieval."""
        request_data = {
            "query": "Machine learning algorithms",
            "databases": ["media"],
            "enable_contextual_retrieval": True,
            "parent_expansion_size": 1000
        }
        
        with patch('tldw_Server_API.app.api.v1.endpoints.rag_api.build_pipeline') as mock_build:
            mock_pipeline = AsyncMock()
            mock_context = Mock()
            mock_context.documents = mock_pipeline_results
            mock_context.metadata = {
                "parent_expansion_applied": True,
                "expansion_time_ms": 50,
                "total_expanded_chars": 2500
            }
            mock_context.cache_hit = False
            mock_pipeline.return_value = mock_context
            mock_build.return_value = mock_pipeline
            
            response = test_client.post(
                "/api/v1/rag/simple",
                json=request_data
            )
            
            assert response.status_code == 200
            result = response.json()
            
            # Performance time should be tracked
            assert "processing_time" in result
            assert result["processing_time"] > 0
    
    @pytest.mark.parametrize("expansion_size", [100, 500, 1000, 2000])
    async def test_different_expansion_sizes(self, test_client, mock_dependencies, mock_pipeline_results, expansion_size):
        """Test different parent expansion sizes."""
        request_data = {
            "query": "Neural networks",
            "databases": ["media"],
            "enable_contextual_retrieval": True,
            "parent_expansion_size": expansion_size
        }
        
        with patch('tldw_Server_API.app.api.v1.endpoints.rag_api.build_pipeline') as mock_build:
            mock_pipeline = AsyncMock()
            mock_context = Mock()
            
            # Simulate different amounts of content based on expansion size
            expanded_content = f"Expanded content " * (expansion_size // 20)
            mock_context.documents = [
                Document(
                    id=doc.id,
                    content=doc.content + expanded_content[:expansion_size],
                    metadata=doc.metadata,
                    source=doc.source,
                    score=doc.score
                ) for doc in mock_pipeline_results
            ]
            mock_context.metadata = {"parent_expansion_applied": True}
            mock_context.cache_hit = False
            mock_pipeline.return_value = mock_context
            mock_build.return_value = mock_pipeline
            
            response = test_client.post(
                "/api/v1/rag/simple",
                json=request_data
            )
            
            assert response.status_code == 200
            result = response.json()
            
            # Verify expansion was applied
            total_content_size = sum(len(r["content"]) for r in result["results"])
            assert total_content_size > len("".join(doc.content for doc in mock_pipeline_results))
    
    @pytest.mark.asyncio
    async def test_contextual_retrieval_with_multiple_databases(self, test_client, mock_dependencies):
        """Test contextual retrieval across multiple databases."""
        request_data = {
            "query": "AI and machine learning",
            "databases": ["media", "notes", "characters"],
            "enable_contextual_retrieval": True,
            "include_sibling_chunks": True
        }
        
        # Create documents from different sources
        mixed_results = [
            Document(
                id="media_doc",
                content="Media content about ML",
                metadata={"parent_id": "media_1", "chunk_index": 0},
                source=DataSource.MEDIA_DB,
                score=0.9
            ),
            Document(
                id="notes_doc",
                content="Notes about AI",
                metadata={"parent_id": "note_1", "chunk_index": 0},
                source=DataSource.NOTES,
                score=0.85
            ),
            Document(
                id="char_doc",
                content="Character discussion on ML",
                metadata={"parent_id": "char_1", "chunk_index": 0},
                source=DataSource.CHARACTER_CARDS,
                score=0.8
            )
        ]
        
        with patch('tldw_Server_API.app.api.v1.endpoints.rag_api.build_pipeline') as mock_build:
            mock_pipeline = AsyncMock()
            mock_context = Mock()
            mock_context.documents = mixed_results
            mock_context.metadata = {"parent_expansion_applied": True}
            mock_context.cache_hit = False
            mock_pipeline.return_value = mock_context
            mock_build.return_value = mock_pipeline
            
            response = test_client.post(
                "/api/v1/rag/simple",
                json=request_data
            )
            
            assert response.status_code == 200
            result = response.json()
            
            # Should have results from multiple sources
            sources = {r["source"] for r in result["results"]}
            assert len(sources) > 1
    
    @pytest.mark.asyncio
    async def test_contextual_retrieval_error_handling(self, test_client, mock_dependencies):
        """Test error handling in contextual retrieval."""
        request_data = {
            "query": "Test query",
            "databases": ["media"],
            "enable_contextual_retrieval": True
        }
        
        with patch('tldw_Server_API.app.api.v1.endpoints.rag_api.build_pipeline') as mock_build:
            # Simulate pipeline error
            mock_pipeline = AsyncMock()
            mock_pipeline.side_effect = Exception("Pipeline error during expansion")
            mock_build.return_value = mock_pipeline
            
            response = test_client.post(
                "/api/v1/rag/simple",
                json=request_data
            )
            
            # Should handle error gracefully
            assert response.status_code == 500
            error_detail = response.json()["detail"]
            assert "Search failed" in error_detail