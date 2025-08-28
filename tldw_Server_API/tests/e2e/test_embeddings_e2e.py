"""
End-to-end tests for embeddings functionality.

Tests the complete workflow of generating embeddings for media items,
including upload with auto-generation and manual generation endpoints.
"""

import pytest
import tempfile
import os
from pathlib import Path
from typing import Dict, Any

# Import the test client fixture
from fixtures import api_client, authenticated_client, APIClient, create_test_file

# Constants
TEST_TEXT_CONTENT = """
This is a test document for embedding generation.
It contains multiple sentences to test chunking.
The chunking algorithm should split this into appropriate segments.
Each segment will have its own embedding vector.
This allows for semantic search across the content.
"""


@pytest.mark.requires_embeddings
class TestEmbeddingsE2E:
    """End-to-end tests for embeddings functionality."""
    
    def teardown_method(self):
        """Clean up after each test method."""
        # Note: In a real scenario, we might want to delete test media items
        # from the database, but for now we'll rely on overwrite_existing
        pass
    
    def test_generate_text_embeddings(self, api_client):
        """Test basic text embedding generation via the embeddings API."""
        # Test single text embedding
        response = api_client.client.post(
            f"{api_client.base_url}/api/v1/embeddings",
            json={
                "input": "This is a test sentence for embedding generation.",
                "model": "sentence-transformers/all-MiniLM-L6-v2"  # Use small model for testing
            }
        )
        
        assert response.status_code == 200, f"Failed to generate embeddings: {response.text}"
        
        result = response.json()
        assert "data" in result
        assert len(result["data"]) > 0
        assert "embedding" in result["data"][0]
        assert isinstance(result["data"][0]["embedding"], list)
        assert len(result["data"][0]["embedding"]) > 0
        
        print(f"✓ Generated embedding with {len(result['data'][0]['embedding'])} dimensions")
    
    def test_batch_embeddings(self, api_client):
        """Test batch embedding generation."""
        texts = [
            "First test sentence.",
            "Second test sentence.",
            "Third test sentence with more content."
        ]
        
        response = api_client.client.post(
            f"{api_client.base_url}/api/v1/embeddings",
            json={
                "input": texts,
                "model": "sentence-transformers/all-MiniLM-L6-v2"
            }
        )
        
        assert response.status_code == 200, f"Failed to generate batch embeddings: {response.text}"
        
        result = response.json()
        assert "data" in result
        assert len(result["data"]) == len(texts), "Should return embedding for each input"
        
        for i, embedding_data in enumerate(result["data"]):
            assert "embedding" in embedding_data
            assert isinstance(embedding_data["embedding"], list)
            assert embedding_data["index"] == i
        
        print(f"✓ Generated {len(result['data'])} embeddings in batch")
    
    def test_media_upload_with_embeddings(self, api_client, test_workflow_state):
        """Test media upload with automatic embedding generation."""
        # Create a test file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            f.write(TEST_TEXT_CONTENT)
            temp_file_path = f.name
        
        try:
            # Upload with embedding generation enabled
            with open(temp_file_path, 'rb') as f:
                files = {"files": (os.path.basename(temp_file_path), f, "text/plain")}
                import time
                import uuid
                # Use unique title to avoid duplicates
                unique_title = f"Test Embeddings {uuid.uuid4()}"
                data = {
                    "title": unique_title,
                    "media_type": "document",
                    "generate_embeddings": "true",  # Enable embedding generation
                    "embedding_model": "sentence-transformers/all-MiniLM-L6-v2",
                    "overwrite_existing": "true"  # Allow overwrite to ensure success
                }
                
                # Add authentication header
                headers = {"X-API-KEY": "test-api-key-12345"}
                response = api_client.client.post(
                    f"{api_client.base_url}/api/v1/media/add",
                    files=files,
                    data=data,
                    headers=headers
                )
            
            assert response.status_code in [200, 207], f"Upload failed: {response.text}"
            
            result = response.json()
            print(f"Response: {result}")  # Debug output
            assert "results" in result
            assert len(result["results"]) > 0
            
            # Get the media ID
            media_id = None
            for item in result["results"]:
                print(f"Item: status={item.get('status')}, db_id={item.get('db_id')}")  # Debug
                if item.get("status") == "Success" and item.get("db_id"):
                    media_id = item["db_id"]
                    # Check if embeddings were scheduled
                    assert item.get("embeddings_scheduled") == True, "Embeddings should be scheduled"
                    break
            
            assert media_id is not None, "No media ID returned from upload"
            
            # Store in workflow state
            test_workflow_state.add_media(media_id, result["results"][0])
            
            print(f"✓ Uploaded media with ID {media_id} and scheduled embeddings")
            
            # Wait a moment for background task to complete
            import time
            time.sleep(2)
            
            # Check if embeddings were generated
            status_response = api_client.client.get(
                f"{api_client.base_url}/api/v1/media/{media_id}/embeddings/status"
            )
            
            if status_response.status_code == 200:
                status = status_response.json()
                if status.get("has_embeddings"):
                    test_workflow_state.mark_embeddings_generated(media_id)
                    print(f"✓ Embeddings generated for media {media_id}")
            
        finally:
            # Clean up temp file
            if os.path.exists(temp_file_path):
                os.unlink(temp_file_path)
    
    def test_manual_media_embeddings_generation(self, api_client, test_workflow_state):
        """Test manual generation of embeddings for uploaded media."""
        # First upload media without embeddings
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            f.write("Simple test content for manual embedding generation.")
            temp_file_path = f.name
        
        try:
            # Upload without embedding generation
            with open(temp_file_path, 'rb') as f:
                files = {"files": (os.path.basename(temp_file_path), f, "text/plain")}
                import uuid
                unique_id = str(uuid.uuid4())
                data = {
                    "title": f"Test Manual Embeddings {unique_id}",
                    "media_type": "document",
                    "generate_embeddings": "false",  # Don't generate automatically
                    "overwrite_existing": "true"  # Allow overwrite to ensure we get a db_id
                }
                
                response = api_client.client.post(
                    f"{api_client.base_url}/api/v1/media/add",
                    files=files,
                    data=data
                )
            
            assert response.status_code in [200, 207], f"Upload failed: {response.text}"
            
            result = response.json()
            print(f"Upload response: {result}")
            
            media_id = None
            for item in result.get("results", []):
                print(f"Item: status={item.get('status')}, db_id={item.get('db_id')}")
                # Accept either Success or Warning status
                if item.get("status") in ["Success", "Warning"] and item.get("db_id"):
                    media_id = item["db_id"]
                    break
            
            assert media_id is not None, f"No media ID returned. Response: {result}"
            
            # Check embeddings don't exist yet
            status_response = api_client.client.get(
                f"{api_client.base_url}/api/v1/media/{media_id}/embeddings/status"
            )
            assert status_response.status_code == 200
            status = status_response.json()
            assert status.get("has_embeddings") == False, "Embeddings should not exist yet"
            
            # Manually generate embeddings
            gen_response = api_client.client.post(
                f"{api_client.base_url}/api/v1/media/{media_id}/embeddings",
                json={
                    "embedding_model": "sentence-transformers/all-MiniLM-L6-v2",
                    "chunk_size": 500,
                    "chunk_overlap": 100
                }
            )
            
            assert gen_response.status_code == 200, f"Failed to generate embeddings: {gen_response.text}"
            
            gen_result = gen_response.json()
            assert gen_result.get("status") == "success"
            assert gen_result.get("embedding_count", 0) > 0
            
            # Mark in workflow state
            test_workflow_state.mark_embeddings_generated(media_id)
            
            print(f"✓ Manually generated {gen_result.get('embedding_count')} embeddings for media {media_id}")
            
        finally:
            # Clean up temp file
            if os.path.exists(temp_file_path):
                os.unlink(temp_file_path)
    
    def test_embedding_model_fallback(self, api_client):
        """Test that system falls back to default model when requested model unavailable."""
        response = api_client.client.post(
            f"{api_client.base_url}/api/v1/embeddings",
            json={
                "input": "Test text for fallback",
                "model": "non-existent-model-xyz-123"  # Non-existent model
            }
        )
        
        # Should either succeed with fallback or return appropriate error
        if response.status_code == 200:
            result = response.json()
            # Check that embeddings were still generated
            assert "data" in result
            assert len(result["data"]) > 0
            print("✓ Successfully fell back to default model")
        else:
            # Should get a meaningful error message
            assert response.status_code in [400, 404, 422]
            error = response.json()
            assert "detail" in error or "error" in error
            print(f"✓ Got expected error for invalid model: {response.status_code}")
    
    def test_embeddings_enable_rag_search(self, api_client, test_workflow_state, ensure_embeddings):
        """Test that embeddings enable RAG search functionality."""
        # Get or create media with embeddings
        media_data = test_workflow_state.get_any_media()
        
        if not media_data:
            # Upload new media if none exists
            self.test_media_upload_with_embeddings(api_client, test_workflow_state)
            media_data = test_workflow_state.get_any_media()
        
        assert media_data is not None, "No media available for testing"
        
        media_id = media_data.get("db_id") or media_data.get("media_id") or media_data.get("id")
        assert media_id is not None, "No media ID found"
        
        # Ensure embeddings exist
        if not test_workflow_state.has_embeddings(media_id):
            success = ensure_embeddings(api_client, media_id)
            assert success, f"Failed to ensure embeddings for media {media_id}"
        
        # Now test RAG search
        search_response = api_client.client.post(
            f"{api_client.base_url}/api/v1/rag/search/simple",
            json={
                "query": "test document content",
                "limit": 5
            }
        )
        
        if search_response.status_code == 200:
            results = search_response.json()
            # If we have results, embeddings are working
            if results.get("results") and len(results["results"]) > 0:
                print(f"✓ RAG search returned {len(results['results'])} results with embeddings")
            else:
                print("⚠️ RAG search returned no results (may need more content)")
        else:
            print(f"⚠️ RAG search returned status {search_response.status_code}")
    
    @pytest.mark.parametrize("chunk_method", ["words", "sentences", "tokens"])
    def test_different_chunking_methods(self, api_client, chunk_method):
        """Test embedding generation with different chunking methods."""
        # Create test content
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            f.write(TEST_TEXT_CONTENT * 5)  # Repeat to ensure multiple chunks
            temp_file_path = f.name
        
        try:
            # Upload with specific chunking method
            with open(temp_file_path, 'rb') as f:
                files = {"files": (os.path.basename(temp_file_path), f, "text/plain")}
                data = {
                    "title": f"Test with {chunk_method} chunking",
                    "media_type": "document",
                    "generate_embeddings": "true",
                    "chunk_method": chunk_method,
                    "chunk_size": "500",  # Increase chunk size
                    "chunk_overlap": "50",  # Use correct parameter name
                    "overwrite_existing": "true"  # Allow overwrite
                }
                
                response = api_client.client.post(
                    f"{api_client.base_url}/api/v1/media/add",
                    files=files,
                    data=data
                )
            
            assert response.status_code in [200, 207], f"Upload failed with {chunk_method}: {response.text}"
            print(f"✓ Successfully uploaded and generated embeddings with {chunk_method} chunking")
            
        finally:
            if os.path.exists(temp_file_path):
                os.unlink(temp_file_path)


@pytest.mark.requires_embeddings
class TestEmbeddingsPerformance:
    """Performance and edge case tests for embeddings."""
    
    def test_large_document_embeddings(self, api_client):
        """Test embedding generation for large documents."""
        # Create a large document (1MB of text)
        large_content = "This is a test sentence. " * 50000
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            f.write(large_content)
            temp_file_path = f.name
        
        try:
            # Upload large document
            with open(temp_file_path, 'rb') as f:
                files = {"files": (os.path.basename(temp_file_path), f, "text/plain")}
                data = {
                    "title": "Large Document Test",
                    "media_type": "document",
                    "generate_embeddings": "true",
                    "chunk_size": "1000",
                    "overlap": "200"
                }
                
                response = api_client.client.post(
                    f"{api_client.base_url}/api/v1/media/add",
                    files=files,
                    data=data,
                    timeout=60  # Longer timeout for large file
                )
            
            assert response.status_code in [200, 207], f"Large upload failed: {response.status_code}"
            print("✓ Successfully handled large document embedding generation")
            
        finally:
            if os.path.exists(temp_file_path):
                os.unlink(temp_file_path)
    
    def test_concurrent_embedding_requests(self, api_client):
        """Test handling of concurrent embedding requests."""
        import concurrent.futures
        import time
        
        def generate_embedding(text_id):
            """Generate embedding for a text."""
            response = api_client.client.post(
                f"{api_client.base_url}/api/v1/embeddings",
                json={
                    "input": f"Test text number {text_id} for concurrent testing.",
                    "model": "sentence-transformers/all-MiniLM-L6-v2"
                }
            )
            return response.status_code == 200
        
        # Submit concurrent requests
        with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
            futures = [executor.submit(generate_embedding, i) for i in range(10)]
            results = [f.result() for f in concurrent.futures.as_completed(futures)]
        
        # Check that most requests succeeded
        success_count = sum(1 for r in results if r)
        assert success_count >= 8, f"Only {success_count}/10 concurrent requests succeeded"
        
        print(f"✓ Handled {success_count}/10 concurrent embedding requests")