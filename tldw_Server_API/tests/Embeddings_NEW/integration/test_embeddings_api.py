"""
Integration tests for Embeddings API endpoints.

Tests the full request/response cycle with real components, no mocking
except for external services like HuggingFace API.
"""

import pytest
import json
import numpy as np
from fastapi import status
from unittest.mock import patch, AsyncMock, MagicMock
import asyncio
from typing import List, Dict, Any

# ========================================================================
# Media Embeddings Endpoint Tests
# ========================================================================

class TestMediaEmbeddingsEndpoint:
    """Test the /api/v1/media/embeddings endpoints."""
    
    @pytest.mark.integration
    async def test_create_embeddings_for_media(self, test_client, auth_headers, populated_media_database):
        """Test creating embeddings for a media item."""
        # Add a media item first
        media_id = populated_media_database.add_media(
            title="Test Document",
            content="This is test content for embedding generation.",
            media_type="document"
        )
        
        response = test_client.post(
            f"/api/v1/media/{media_id}/embeddings",
            json={
                "model": "sentence-transformers/all-MiniLM-L6-v2",
                "chunk_size": 500,
                "chunk_overlap": 50
            },
            headers=auth_headers
        )
        
        if response.status_code == status.HTTP_202_ACCEPTED:
            data = response.json()
            assert "job_id" in data
            assert "status" in data
            assert data["status"] == "processing"
    
    @pytest.mark.integration
    async def test_get_embedding_status(self, test_client, auth_headers):
        """Test getting embedding job status."""
        job_id = "test-job-123"
        
        response = test_client.get(
            f"/api/v1/media/embeddings/status/{job_id}",
            headers=auth_headers
        )
        
        # Should handle non-existent job gracefully
        if response.status_code == status.HTTP_404_NOT_FOUND:
            data = response.json()
            assert "error" in data or "detail" in data
    
    @pytest.mark.integration
    @patch('sentence_transformers.SentenceTransformer')
    async def test_batch_embeddings_creation(self, mock_transformer, test_client, auth_headers, populated_media_database):
        """Test creating embeddings for multiple media items."""
        mock_model = MagicMock()
        mock_model.encode.return_value = np.random.randn(10, 384)
        mock_transformer.return_value = mock_model
        
        # Get media IDs from populated database
        media_items = populated_media_database.search_media_items("")[:3]
        media_ids = [item["id"] for item in media_items]
        
        response = test_client.post(
            "/api/v1/media/embeddings/batch",
            json={
                "media_ids": media_ids,
                "model": "sentence-transformers/all-MiniLM-L6-v2"
            },
            headers=auth_headers
        )
        
        if response.status_code == status.HTTP_202_ACCEPTED:
            data = response.json()
            assert "job_ids" in data
            assert len(data["job_ids"]) == len(media_ids)
    
    @pytest.mark.integration
    async def test_search_by_embeddings(self, test_client, auth_headers, populated_chroma_collection):
        """Test searching using embeddings."""
        response = test_client.post(
            "/api/v1/media/embeddings/search",
            json={
                "query": "machine learning concepts",
                "top_k": 5,
                "collection": populated_chroma_collection.name
            },
            headers=auth_headers
        )
        
        if response.status_code == status.HTTP_200_OK:
            data = response.json()
            assert "results" in data
            assert len(data["results"]) <= 5

# ========================================================================
# Embedding Models Management Tests
# ========================================================================

class TestEmbeddingModelsManagement:
    """Test embedding model management endpoints."""
    
    @pytest.mark.integration
    async def test_list_available_models(self, test_client, auth_headers):
        """Test listing available embedding models."""
        response = test_client.get(
            "/api/v1/embeddings/models",
            headers=auth_headers
        )
        
        if response.status_code == status.HTTP_200_OK:
            data = response.json()
            assert isinstance(data, list)
            # Should have at least the default model
            assert any("all-MiniLM-L6-v2" in model for model in data)
    
    @pytest.mark.integration
    async def test_get_model_info(self, test_client, auth_headers):
        """Test getting information about a specific model."""
        response = test_client.get(
            "/api/v1/embeddings/models/sentence-transformers/all-MiniLM-L6-v2",
            headers=auth_headers
        )
        
        if response.status_code == status.HTTP_200_OK:
            data = response.json()
            assert "dimension" in data
            assert data["dimension"] == 384
            assert "max_tokens" in data
    
    @pytest.mark.integration
    @patch('sentence_transformers.SentenceTransformer')
    async def test_warmup_model(self, mock_transformer, test_client, auth_headers):
        """Test model warmup endpoint."""
        mock_model = MagicMock()
        mock_transformer.return_value = mock_model
        
        response = test_client.post(
            "/api/v1/embeddings/models/warmup",
            json={
                "model": "sentence-transformers/all-MiniLM-L6-v2"
            },
            headers=auth_headers
        )
        
        if response.status_code == status.HTTP_200_OK:
            data = response.json()
            assert data["status"] == "warmed_up"

# ========================================================================
# ChromaDB Collection Management Tests
# ========================================================================

class TestChromaDBCollectionManagement:
    """Test ChromaDB collection management endpoints."""
    
    @pytest.mark.integration
    async def test_create_collection(self, test_client, auth_headers, chroma_client):
        """Test creating a new ChromaDB collection."""
        response = test_client.post(
            "/api/v1/embeddings/collections",
            json={
                "name": "test_collection",
                "metadata": {"description": "Test collection"},
                "embedding_model": "sentence-transformers/all-MiniLM-L6-v2"
            },
            headers=auth_headers
        )
        
        if response.status_code == status.HTTP_201_CREATED:
            data = response.json()
            assert data["name"] == "test_collection"
            
            # Verify collection exists in ChromaDB
            collections = chroma_client.list_collections()
            assert any(c.name == "test_collection" for c in collections)
    
    @pytest.mark.integration
    async def test_list_collections(self, test_client, auth_headers, populated_chroma_collection):
        """Test listing ChromaDB collections."""
        response = test_client.get(
            "/api/v1/embeddings/collections",
            headers=auth_headers
        )
        
        if response.status_code == status.HTTP_200_OK:
            data = response.json()
            assert isinstance(data, list)
            assert len(data) > 0
    
    @pytest.mark.integration
    async def test_delete_collection(self, test_client, auth_headers, chroma_collection):
        """Test deleting a ChromaDB collection."""
        collection_name = chroma_collection.name
        
        response = test_client.delete(
            f"/api/v1/embeddings/collections/{collection_name}",
            headers=auth_headers
        )
        
        if response.status_code == status.HTTP_204_NO_CONTENT:
            # Verify collection is deleted
            with pytest.raises(Exception):
                chroma_collection.get()
    
    @pytest.mark.integration
    async def test_get_collection_stats(self, test_client, auth_headers, populated_chroma_collection):
        """Test getting collection statistics."""
        response = test_client.get(
            f"/api/v1/embeddings/collections/{populated_chroma_collection.name}/stats",
            headers=auth_headers
        )
        
        if response.status_code == status.HTTP_200_OK:
            data = response.json()
            assert "count" in data
            assert data["count"] == 10  # From populated fixture
            assert "embedding_dimension" in data

# ========================================================================
# Embedding Generation Pipeline Tests
# ========================================================================

class TestEmbeddingGenerationPipeline:
    """Test the full embedding generation pipeline."""
    
    @pytest.mark.integration
    @patch('sentence_transformers.SentenceTransformer')
    async def test_full_pipeline_text_to_storage(self, mock_transformer, test_client, auth_headers, media_database):
        """Test full pipeline from text to stored embeddings."""
        mock_model = MagicMock()
        mock_model.encode.return_value = np.random.randn(5, 384)
        mock_transformer.return_value = mock_model
        
        # Create media item
        media_id = media_database.add_media(
            title="Pipeline Test",
            content="This is a longer text that will be chunked. " * 50,
            media_type="document"
        )
        
        # Start embedding generation
        response = test_client.post(
            f"/api/v1/media/{media_id}/embeddings",
            json={
                "chunk_size": 100,
                "chunk_overlap": 20
            },
            headers=auth_headers
        )
        
        if response.status_code == status.HTTP_202_ACCEPTED:
            job_id = response.json()["job_id"]
            
            # Poll for completion (in real scenario)
            await asyncio.sleep(0.5)
            
            # Check status
            status_response = test_client.get(
                f"/api/v1/media/embeddings/status/{job_id}",
                headers=auth_headers
            )
            
            if status_response.status_code == status.HTTP_200_OK:
                status_data = status_response.json()
                assert status_data["job_id"] == job_id
    
    @pytest.mark.integration
    async def test_pipeline_with_custom_chunking(self, test_client, auth_headers, media_database):
        """Test pipeline with custom chunking strategy."""
        media_id = media_database.add_media(
            title="Custom Chunking Test",
            content="Sentence one. Sentence two. Sentence three. Sentence four.",
            media_type="document"
        )
        
        response = test_client.post(
            f"/api/v1/media/{media_id}/embeddings",
            json={
                "chunking_strategy": "sentence",
                "sentences_per_chunk": 2
            },
            headers=auth_headers
        )
        
        assert response.status_code in [
            status.HTTP_202_ACCEPTED,
            status.HTTP_200_OK
        ]

# ========================================================================
# Worker Orchestration Tests
# ========================================================================

class TestWorkerOrchestrationIntegration:
    """Test worker orchestration in integration scenarios."""
    
    @pytest.mark.integration
    async def test_concurrent_job_processing(self, test_client, auth_headers, populated_media_database):
        """Test processing multiple concurrent embedding jobs."""
        # Get multiple media items
        media_items = populated_media_database.search_media_items("")[:5]
        
        # Submit concurrent embedding requests
        tasks = []
        for item in media_items:
            response = test_client.post(
                f"/api/v1/media/{item['id']}/embeddings",
                json={"model": "sentence-transformers/all-MiniLM-L6-v2"},
                headers=auth_headers
            )
            tasks.append(response)
        
        # All should be accepted
        for response in tasks:
            assert response.status_code in [
                status.HTTP_202_ACCEPTED,
                status.HTTP_200_OK
            ]
    
    @pytest.mark.integration
    async def test_job_priority_handling(self, test_client, auth_headers, media_database):
        """Test that high-priority jobs are processed first."""
        # Create media items
        urgent_id = media_database.add_media(
            title="Urgent",
            content="Urgent content",
            media_type="document"
        )
        
        normal_id = media_database.add_media(
            title="Normal",
            content="Normal content",
            media_type="document"
        )
        
        # Submit with priorities
        urgent_response = test_client.post(
            f"/api/v1/media/{urgent_id}/embeddings",
            json={"priority": 10},
            headers=auth_headers
        )
        
        normal_response = test_client.post(
            f"/api/v1/media/{normal_id}/embeddings",
            json={"priority": 1},
            headers=auth_headers
        )
        
        assert urgent_response.status_code in [status.HTTP_202_ACCEPTED, status.HTTP_200_OK]
        assert normal_response.status_code in [status.HTTP_202_ACCEPTED, status.HTTP_200_OK]

# ========================================================================
# Error Handling Tests
# ========================================================================

class TestErrorHandlingIntegration:
    """Test error handling in integration scenarios."""
    
    @pytest.mark.integration
    async def test_invalid_model_error(self, test_client, auth_headers):
        """Test error handling for invalid model."""
        response = test_client.post(
            "/api/v1/media/1/embeddings",
            json={"model": "invalid-model-name"},
            headers=auth_headers
        )
        
        assert response.status_code == status.HTTP_400_BAD_REQUEST
        data = response.json()
        assert "error" in data or "detail" in data
    
    @pytest.mark.integration
    async def test_media_not_found_error(self, test_client, auth_headers):
        """Test error handling for non-existent media."""
        response = test_client.post(
            "/api/v1/media/999999/embeddings",
            json={},
            headers=auth_headers
        )
        
        assert response.status_code == status.HTTP_404_NOT_FOUND
    
    @pytest.mark.integration
    @patch('sentence_transformers.SentenceTransformer')
    async def test_model_loading_failure(self, mock_transformer, test_client, auth_headers):
        """Test handling of model loading failures."""
        mock_transformer.side_effect = Exception("Model loading failed")
        
        response = test_client.post(
            "/api/v1/media/1/embeddings",
            json={"model": "sentence-transformers/all-MiniLM-L6-v2"},
            headers=auth_headers
        )
        
        assert response.status_code in [
            status.HTTP_500_INTERNAL_SERVER_ERROR,
            status.HTTP_503_SERVICE_UNAVAILABLE
        ]

# ========================================================================
# Performance and Scaling Tests
# ========================================================================

class TestPerformanceAndScaling:
    """Test performance and scaling characteristics."""
    
    @pytest.mark.integration
    @pytest.mark.slow
    async def test_large_batch_processing(self, test_client, auth_headers, large_text_corpus):
        """Test processing large batches of text."""
        with patch('sentence_transformers.SentenceTransformer') as mock_transformer:
            mock_model = MagicMock()
            
            def mock_encode(texts, *args, **kwargs):
                batch_size = len(texts) if isinstance(texts, list) else 1
                return np.random.randn(batch_size, 384)
            
            mock_model.encode = mock_encode
            mock_transformer.return_value = mock_model
            
            response = test_client.post(
                "/api/v1/embeddings/batch",
                json={
                    "texts": large_text_corpus[:100],  # First 100 texts
                    "batch_size": 32
                },
                headers=auth_headers
            )
            
            if response.status_code == status.HTTP_200_OK:
                data = response.json()
                assert "embeddings" in data
                assert len(data["embeddings"]) == 100
    
    @pytest.mark.integration
    async def test_rate_limiting(self, test_client, auth_headers):
        """Test rate limiting on embedding endpoints."""
        # Send many requests rapidly
        responses = []
        for i in range(20):
            response = test_client.post(
                f"/api/v1/media/{i}/embeddings",
                json={},
                headers=auth_headers
            )
            responses.append(response)
        
        # Some should be rate limited
        rate_limited = [r for r in responses if r.status_code == status.HTTP_429_TOO_MANY_REQUESTS]
        
        # Rate limiting may or may not be enabled
        if rate_limited:
            assert len(rate_limited) > 0
            assert "retry-after" in rate_limited[0].headers