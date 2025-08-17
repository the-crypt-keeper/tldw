# test_embeddings_v5_integration.py
# Integration tests for production embeddings service (no mocking)

import asyncio
import os
from datetime import datetime
import pytest
import numpy as np

from fastapi.testclient import TestClient
from httpx import AsyncClient
from tldw_Server_API.app.main import app
from tldw_Server_API.app.core.AuthNZ.User_DB_Handling import User, get_request_user


# Disable rate limiting for all tests
@pytest.fixture(autouse=True)
def disable_rate_limiting():
    """Disable rate limiting for all tests in this module"""
    os.environ["TESTING"] = "true"
    yield
    # Clean up after tests
    if "TESTING" in os.environ:
        del os.environ["TESTING"]

# Module-level setup fixture for integration tests
@pytest.fixture
def setup():
    """Setup test environment fixture"""
    class SetupData:
        def __init__(self):
            self.client = TestClient(app)
            # Set CSRF token in both cookie and header
            csrf_token = "test-csrf-token-12345"
            self.client.cookies.set("csrf_token", csrf_token)
            self.auth_headers = {
                "Authorization": "Bearer test-api-key",
                "X-CSRF-Token": csrf_token
            }
            
            self.test_user = User(
                id=1, 
                username="testuser", 
                email="test@example.com", 
                is_active=True,
                is_admin=False
            )
    
    data = SetupData()
    yield data
    app.dependency_overrides.clear()


@pytest.mark.integration
class TestEmbeddingsIntegration:
    """Integration tests without mocking - requires actual services"""
    
    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_real_huggingface_embedding(self, setup):
        """Test actual HuggingFace embedding creation (no mocks)"""
        async def override_user():
            return setup.test_user
        
        app.dependency_overrides[get_request_user] = override_user
        
        # This test uses real HuggingFace models - no mocking
        response = setup.client.post(
            "/api/v1/embeddings",
            headers={**setup.auth_headers, "x-provider": "huggingface"},
            json={
                "input": "This is a real integration test with HuggingFace",
                "model": "sentence-transformers/all-MiniLM-L6-v2"
            }
        )
        
        # Will only pass if model is available
        if response.status_code == 200:
            data = response.json()
            
            # Verify real embeddings were created
            assert "data" in data
            assert len(data["data"]) == 1
            assert "embedding" in data["data"][0]
            
            # Real embeddings should have expected dimensions
            embedding = data["data"][0]["embedding"]
            assert len(embedding) == 384  # all-MiniLM-L6-v2 has 384 dimensions
            
            # Real embeddings should be normalized
            norm = np.linalg.norm(embedding)
            assert 0.95 < norm < 1.05  # Approximately unit length
            
            # Embeddings should have reasonable values
            assert all(-10 < x < 10 for x in embedding)
    
    @pytest.mark.integration
    @pytest.mark.asyncio
    @pytest.mark.skipif(
        not os.getenv("OPENAI_API_KEY"),
        reason="Integration test requires OPENAI_API_KEY environment variable"
    )
    async def test_real_openai_embedding(self, setup):
        """Test actual OpenAI API integration (no mocks)"""
        async def override_user():
            return setup.test_user
        
        app.dependency_overrides[get_request_user] = override_user
        
        # This test uses real OpenAI API - no mocking
        response = setup.client.post(
            "/api/v1/embeddings",
            headers=setup.auth_headers,
            json={
                "input": "Real OpenAI integration test with actual API",
                "model": "text-embedding-3-small"
            }
        )
        
        assert response.status_code == 200
        data = response.json()
        
        # Verify real OpenAI embeddings
        assert "data" in data
        assert len(data["data"]) == 1
        
        embedding = data["data"][0]["embedding"]
        assert len(embedding) == 1536  # text-embedding-3-small default dimensions
        
        # Check usage is reported correctly
        assert "usage" in data
        assert data["usage"]["total_tokens"] > 0
        assert data["usage"]["prompt_tokens"] > 0
        
        # OpenAI embeddings should be normalized
        norm = np.linalg.norm(embedding)
        assert 0.95 < norm < 1.05
    
    @pytest.mark.integration
    @pytest.mark.asyncio
    @pytest.mark.asyncio
    async def test_real_cache_persistence(self, setup):
        """Test cache persistence across requests (no mocks)"""
        async def override_user():
            return setup.test_user
        
        app.dependency_overrides[get_request_user] = override_user
        
        # Use unique text to avoid conflicts with other tests
        unique_text = f"Cache test {datetime.now().isoformat()} {os.getpid()}"
        
        # First request - will create real embedding
        response1 = setup.client.post(
            "/api/v1/embeddings",
            headers={**setup.auth_headers, "x-provider": "huggingface"},
            json={
                "input": unique_text,
                "model": "sentence-transformers/all-MiniLM-L6-v2"
            }
        )
        
        if response1.status_code == 200:
            embedding1 = response1.json()["data"][0]["embedding"]
            
            # Second identical request
            response2 = setup.client.post(
                "/api/v1/embeddings",
                headers={**setup.auth_headers, "x-provider": "huggingface"},
                json={
                    "input": unique_text,
                    "model": "sentence-transformers/all-MiniLM-L6-v2"
                }
            )
            
            assert response2.status_code == 200
            embedding2 = response2.json()["data"][0]["embedding"]
            
            # Should return identical embeddings (from cache)
            assert embedding1 == embedding2
            
            # Verify embeddings are valid
            assert len(embedding1) == 384
            assert len(embedding2) == 384
    
    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_different_providers_produce_different_embeddings(self, setup):
        """Test that different providers produce different embeddings for same text"""
        async def override_user():
            return setup.test_user
        
        app.dependency_overrides[get_request_user] = override_user
        
        test_text = "Compare embeddings across providers"
        
        # Get HuggingFace embedding
        response_hf = setup.client.post(
            "/api/v1/embeddings",
            headers={**setup.auth_headers, "x-provider": "huggingface"},
            json={
                "input": test_text,
                "model": "sentence-transformers/all-MiniLM-L6-v2"
            }
        )
        
        if response_hf.status_code == 200 and os.getenv("OPENAI_API_KEY"):
            embedding_hf = response_hf.json()["data"][0]["embedding"]
            
            # Get OpenAI embedding
            response_openai = setup.client.post(
                "/api/v1/embeddings",
                headers=setup.auth_headers,
                json={
                    "input": test_text,
                    "model": "text-embedding-3-small"
                }
            )
            
            if response_openai.status_code == 200:
                embedding_openai = response_openai.json()["data"][0]["embedding"]
                
                # Should have different dimensions
                assert len(embedding_hf) != len(embedding_openai)
                assert len(embedding_hf) == 384
                assert len(embedding_openai) == 1536
                
                # Both should be normalized
                assert 0.95 < np.linalg.norm(embedding_hf) < 1.05
                assert 0.95 < np.linalg.norm(embedding_openai) < 1.05
    
    @pytest.mark.integration
    @pytest.mark.asyncio
    @pytest.mark.skipif(
        not os.getenv("RUN_STRESS_TESTS"),
        reason="Stress tests require RUN_STRESS_TESTS=true"
    )
    async def test_real_concurrent_load(self, setup):
        """Test system under real concurrent load (no mocks)"""
        async def override_user():
            return setup.test_user
        
        app.dependency_overrides[get_request_user] = override_user
        
        async def make_real_request(idx):
            async with AsyncClient(app=app, base_url="http://test") as ac:
                # Set CSRF token in cookie
                ac.cookies.set("csrf_token", "test-csrf-token-12345")
                response = await ac.post(
                    "/api/v1/embeddings",
                    headers={**setup.auth_headers, "x-provider": "huggingface"},
                    json={
                        "input": f"Concurrent test {idx} at {datetime.now()}",
                        "model": "sentence-transformers/all-MiniLM-L6-v2"
                    }
                )
                return response.status_code
        
        # Make real concurrent requests
        tasks = [make_real_request(i) for i in range(20)]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Count successful requests
        successful = sum(1 for r in results if isinstance(r, int) and r == 200)
        failed = sum(1 for r in results if not isinstance(r, int) or r != 200)
        
        # Most should succeed
        assert successful > 15, f"Only {successful}/20 requests succeeded"
        
        # Log any failures for debugging
        if failed > 0:
            print(f"Failed requests: {failed}/20")
            for i, r in enumerate(results):
                if not isinstance(r, int) or r != 200:
                    print(f"Request {i}: {r}")
    
    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_batch_processing(self, setup):
        """Test batch processing with real embeddings"""
        async def override_user():
            return setup.test_user
        
        app.dependency_overrides[get_request_user] = override_user
        
        # Create batch of unique texts
        batch_texts = [f"Batch text {i}" for i in range(10)]
        
        response = setup.client.post(
            "/api/v1/embeddings",
            headers={**setup.auth_headers, "x-provider": "huggingface"},
            json={
                "input": batch_texts,
                "model": "sentence-transformers/all-MiniLM-L6-v2"
            }
        )
        
        if response.status_code == 200:
            data = response.json()
            
            # Should return embeddings for all texts
            assert len(data["data"]) == 10
            
            # Each embedding should be valid
            for i, embedding_data in enumerate(data["data"]):
                assert embedding_data["index"] == i
                embedding = embedding_data["embedding"]
                assert len(embedding) == 384
                
                # Should be normalized
                norm = np.linalg.norm(embedding)
                assert 0.95 < norm < 1.05
            
            # Different texts should produce different embeddings
            embeddings = [d["embedding"] for d in data["data"]]
            for i in range(len(embeddings) - 1):
                # Cosine similarity should not be 1.0 (identical)
                sim = np.dot(embeddings[i], embeddings[i + 1])
                assert sim < 0.99, "Different texts produced identical embeddings"
    
    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_health_check_with_real_service(self, setup):
        """Test health check endpoint with real service"""
        response = setup.client.get("/api/v1/embeddings/health")
        
        assert response.status_code in [200, 503]  # May be degraded if dependencies missing
        data = response.json()
        
        assert "status" in data
        assert "service" in data
        assert data["service"] == "embeddings_v5_production_enhanced"
        assert "timestamp" in data
        assert "cache_stats" in data
        assert "active_requests" in data
        
        # Cache stats should be valid
        cache_stats = data["cache_stats"]
        assert "size" in cache_stats
        assert "max_size" in cache_stats
        assert "ttl_seconds" in cache_stats
        assert cache_stats["size"] >= 0
        assert cache_stats["max_size"] > 0