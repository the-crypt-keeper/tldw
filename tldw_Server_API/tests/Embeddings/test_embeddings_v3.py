# test_embeddings_v3.py
# Tests for enhanced embeddings API with batch processing, dimensions, and token support
import json
import base64
import numpy as np
import pytest
import asyncio
from fastapi.testclient import TestClient
from httpx import AsyncClient, ASGITransport
from unittest.mock import Mock, patch, MagicMock

from tldw_Server_API.app.main import app
from tldw_Server_API.app.core.AuthNZ.User_DB_Handling import User, get_request_user


class TestEmbeddingsV3Enhanced:
    """Test suite for the enhanced embeddings API v3"""
    
    @pytest.fixture(autouse=True)
    def setup(self):
        """Setup test client and authentication"""
        self.client = TestClient(app)
        self.DEFAULT_API_KEY = "default-secret-key-for-single-user"
        self.auth_headers = {"Authorization": f"Bearer {self.DEFAULT_API_KEY}"}
        
        # Mock user for authentication
        self.test_user = User(id=1, username="testuser", email="test@example.com", is_active=True)
        
        # Override authentication for tests
        async def override_get_user():
            return self.test_user
        
        app.dependency_overrides[get_request_user] = override_get_user
        
        yield
        
        # Cleanup
        app.dependency_overrides.clear()
    
    # ===== DIMENSIONS PARAMETER TESTS =====
    
    def test_dimensions_parameter_with_v3_model(self):
        """Test dimensions parameter with text-embedding-3-small"""
        response = self.client.post(
            "/api/v1/embeddings",
            headers=self.auth_headers,
            json={
                "input": "Test with custom dimensions",
                "model": "text-embedding-3-small",
                "dimensions": 512
            }
        )
        
        assert response.status_code == 200
        data = response.json()
        
        # Check embedding dimensions match requested
        embedding = data["data"][0]["embedding"]
        assert len(embedding) == 512
    
    def test_dimensions_parameter_with_v3_large(self):
        """Test dimensions parameter with text-embedding-3-large"""
        response = self.client.post(
            "/api/v1/embeddings",
            headers=self.auth_headers,
            json={
                "input": "Test with large model dimensions",
                "model": "text-embedding-3-large",
                "dimensions": 1024
            }
        )
        
        assert response.status_code == 200
        data = response.json()
        
        # Check embedding dimensions
        embedding = data["data"][0]["embedding"]
        assert len(embedding) == 1024
    
    def test_dimensions_not_supported_for_ada(self):
        """Test that dimensions parameter is rejected for ada-002"""
        response = self.client.post(
            "/api/v1/embeddings",
            headers=self.auth_headers,
            json={
                "input": "Test dimensions with ada",
                "model": "text-embedding-ada-002",
                "dimensions": 512
            }
        )
        
        assert response.status_code == 400
        data = response.json()
        assert "only supported for text-embedding-3" in data["detail"].lower()
    
    def test_dimensions_exceeds_model_limit(self):
        """Test error when dimensions exceed model's native dimensions"""
        response = self.client.post(
            "/api/v1/embeddings",
            headers=self.auth_headers,
            json={
                "input": "Test exceeding dimensions",
                "model": "text-embedding-3-small",
                "dimensions": 2048  # Exceeds 1536 limit
            }
        )
        
        assert response.status_code == 400
        data = response.json()
        assert "exceeds maximum" in data["detail"].lower()
    
    def test_dimensions_minimum_value(self):
        """Test minimum dimensions value"""
        response = self.client.post(
            "/api/v1/embeddings",
            headers=self.auth_headers,
            json={
                "input": "Test minimum dimensions",
                "model": "text-embedding-3-small",
                "dimensions": 1
            }
        )
        
        assert response.status_code == 200
        data = response.json()
        embedding = data["data"][0]["embedding"]
        assert len(embedding) == 1
    
    def test_dimensions_zero_rejected(self):
        """Test that dimensions=0 is rejected"""
        response = self.client.post(
            "/api/v1/embeddings",
            headers=self.auth_headers,
            json={
                "input": "Test zero dimensions",
                "model": "text-embedding-3-small",
                "dimensions": 0
            }
        )
        
        assert response.status_code == 400
        data = response.json()
        assert "at least 1" in data["detail"].lower()
    
    # ===== TOKEN ARRAY INPUT TESTS =====
    
    def test_single_token_array_input(self):
        """Test single token array as input"""
        # Example token IDs (would be actual tokens in production)
        token_ids = [15339, 11, 1917, 0, 1115, 374, 264, 1296, 315, 279, 40188, 5446, 13]
        
        response = self.client.post(
            "/api/v1/embeddings",
            headers=self.auth_headers,
            json={
                "input": token_ids,
                "model": "text-embedding-3-small"
            }
        )
        
        assert response.status_code == 200
        data = response.json()
        
        # Should get one embedding
        assert len(data["data"]) == 1
        assert "embedding" in data["data"][0]
        
        # Usage should reflect token count
        assert data["usage"]["prompt_tokens"] == len(token_ids)
    
    def test_batch_token_arrays_input(self):
        """Test batch of token arrays as input"""
        token_arrays = [
            [15339, 11, 1917, 0],
            [1115, 374, 264, 1296],
            [315, 279, 40188, 5446]
        ]
        
        response = self.client.post(
            "/api/v1/embeddings",
            headers=self.auth_headers,
            json={
                "input": token_arrays,
                "model": "text-embedding-3-small"
            }
        )
        
        assert response.status_code == 200
        data = response.json()
        
        # Should get three embeddings
        assert len(data["data"]) == 3
        
        # Usage should reflect total token count
        total_tokens = sum(len(arr) for arr in token_arrays)
        assert data["usage"]["prompt_tokens"] == total_tokens
    
    def test_mixed_input_types_rejected(self):
        """Test that mixed input types are rejected"""
        response = self.client.post(
            "/api/v1/embeddings",
            headers=self.auth_headers,
            json={
                "input": ["text string", [123, 456], 789],
                "model": "text-embedding-3-small"
            }
        )
        
        assert response.status_code == 400
        data = response.json()
        assert "invalid input format" in data["detail"].lower()
    
    def test_token_arrays_with_dimensions(self):
        """Test token arrays with dimensions parameter"""
        token_arrays = [
            [15339, 11, 1917],
            [1115, 374, 264]
        ]
        
        response = self.client.post(
            "/api/v1/embeddings",
            headers=self.auth_headers,
            json={
                "input": token_arrays,
                "model": "text-embedding-3-small",
                "dimensions": 256
            }
        )
        
        assert response.status_code == 200
        data = response.json()
        
        # Check dimensions applied
        for item in data["data"]:
            assert len(item["embedding"]) == 256
    
    # ===== BATCH PROCESSING OPTIMIZATION TESTS =====
    
    def test_large_batch_processing(self):
        """Test that large batches are processed efficiently"""
        # Create 250 strings (should be split into 3 batches of 100, 100, 50)
        large_batch = [f"Test string number {i}" for i in range(250)]
        
        response = self.client.post(
            "/api/v1/embeddings",
            headers=self.auth_headers,
            json={
                "input": large_batch,
                "model": "text-embedding-3-small"
            }
        )
        
        assert response.status_code == 200
        data = response.json()
        
        # Should get embeddings for all inputs
        assert len(data["data"]) == 250
        
        # Check indices are correct
        for i in range(250):
            assert data["data"][i]["index"] == i
    
    @pytest.mark.asyncio
    async def test_batch_endpoint(self):
        """Test the /embeddings/batch endpoint"""
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            # Override auth for async client
            async def override_get_user():
                return self.test_user
            app.dependency_overrides[get_request_user] = override_get_user
            
            requests = [
                {
                    "input": "First batch request",
                    "model": "text-embedding-3-small"
                },
                {
                    "input": ["Second batch", "with multiple strings"],
                    "model": "text-embedding-3-small"
                },
                {
                    "input": "Third batch request",
                    "model": "text-embedding-3-small",
                    "dimensions": 512
                }
            ]
            
            response = await client.post(
                "/api/v1/embeddings/batch",
                headers=self.auth_headers,
                json=requests
            )
            
            assert response.status_code == 200
            data = response.json()
            
            # Should get 3 responses
            assert len(data) == 3
            
            # First response: single embedding
            assert len(data[0]["data"]) == 1
            
            # Second response: two embeddings
            assert len(data[1]["data"]) == 2
            
            # Third response: check dimensions
            if "error" not in data[2]:
                embedding = data[2]["data"][0]["embedding"]
                assert len(embedding) == 512
    
    def test_batch_endpoint_exceeds_limit(self):
        """Test batch endpoint with too many requests"""
        # Create 11 requests (exceeds limit of 10)
        requests = [
            {"input": f"Request {i}", "model": "text-embedding-3-small"}
            for i in range(11)
        ]
        
        response = self.client.post(
            "/api/v1/embeddings/batch",
            headers=self.auth_headers,
            json=requests
        )
        
        assert response.status_code == 400
        data = response.json()
        assert "maximum 10 requests" in data["detail"].lower()
    
    # ===== CACHING TESTS =====
    
    @pytest.mark.asyncio
    async def test_caching_same_input(self):
        """Test that same input returns cached result"""
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            # Override auth
            async def override_get_user():
                return self.test_user
            app.dependency_overrides[get_request_user] = override_get_user
            
            # First request
            response1 = await client.post(
                "/api/v1/embeddings",
                headers=self.auth_headers,
                json={
                    "input": "This text should be cached",
                    "model": "text-embedding-3-small"
                }
            )
            
            assert response1.status_code == 200
            data1 = response1.json()
            
            # Second identical request (should hit cache)
            response2 = await client.post(
                "/api/v1/embeddings",
                headers=self.auth_headers,
                json={
                    "input": "This text should be cached",
                    "model": "text-embedding-3-small"
                }
            )
            
            assert response2.status_code == 200
            data2 = response2.json()
            
            # Embeddings should be identical
            assert data1["data"][0]["embedding"] == data2["data"][0]["embedding"]
    
    def test_cache_stats_endpoint(self):
        """Test cache statistics endpoint"""
        # Generate some embeddings first
        for i in range(5):
            self.client.post(
                "/api/v1/embeddings",
                headers=self.auth_headers,
                json={
                    "input": f"Cache test string {i}",
                    "model": "text-embedding-3-small"
                }
            )
        
        # Get cache stats
        response = self.client.get(
            "/api/v1/embeddings/cache/stats",
            headers=self.auth_headers
        )
        
        assert response.status_code == 200
        data = response.json()
        
        assert "cache_size" in data
        assert "max_cache_size" in data
        assert "cache_ttl" in data
        assert "average_age_seconds" in data
        assert "oldest_entry_age_seconds" in data
    
    def test_clear_cache_endpoint(self):
        """Test cache clearing endpoint"""
        # Generate some embeddings
        for i in range(3):
            self.client.post(
                "/api/v1/embeddings",
                headers=self.auth_headers,
                json={
                    "input": f"Cache clear test {i}",
                    "model": "text-embedding-3-small"
                }
            )
        
        # Clear cache
        response = self.client.delete(
            "/api/v1/embeddings/cache",
            headers=self.auth_headers
        )
        
        assert response.status_code == 200
        data = response.json()
        
        assert "entries_removed" in data
        assert data["message"] == "Cache cleared"
    
    # ===== MODEL LISTING TESTS =====
    
    def test_list_models_with_enhanced_info(self):
        """Test model listing with enhanced information"""
        response = self.client.get(
            "/api/v1/embeddings/models",
            headers=self.auth_headers
        )
        
        assert response.status_code == 200
        data = response.json()
        
        assert "models" in data
        assert "features" in data
        
        # Check features
        features = data["features"]
        assert features["batch_processing"] == True
        assert features["dimensions_reduction"] == True
        assert features["token_input"] == True
        assert features["caching"] == True
        assert "max_batch_size" in features
        
        # Check model info
        if data["models"]:
            model = data["models"][0]
            assert "supports_dimensions" in model
            if model["supports_dimensions"]:
                assert "min_dimensions" in model
    
    # ===== TEST ENDPOINT VARIATIONS =====
    
    def test_test_endpoint_text_type(self):
        """Test the test endpoint with text type"""
        response = self.client.post(
            "/api/v1/embeddings/test",
            headers=self.auth_headers,
            params={"test_type": "text", "dimensions": 256}
        )
        
        assert response.status_code == 200
        data = response.json()
        embedding = data["data"][0]["embedding"]
        assert len(embedding) == 256
    
    def test_test_endpoint_batch_type(self):
        """Test the test endpoint with batch type"""
        response = self.client.post(
            "/api/v1/embeddings/test",
            headers=self.auth_headers,
            params={"test_type": "batch"}
        )
        
        assert response.status_code == 200
        data = response.json()
        assert len(data["data"]) == 3
    
    def test_test_endpoint_tokens_type(self):
        """Test the test endpoint with tokens type"""
        response = self.client.post(
            "/api/v1/embeddings/test",
            headers=self.auth_headers,
            params={"test_type": "tokens"}
        )
        
        assert response.status_code == 200
        data = response.json()
        assert len(data["data"]) == 1
    
    def test_test_endpoint_batch_tokens_type(self):
        """Test the test endpoint with batch_tokens type"""
        response = self.client.post(
            "/api/v1/embeddings/test",
            headers=self.auth_headers,
            params={"test_type": "batch_tokens"}
        )
        
        assert response.status_code == 200
        data = response.json()
        assert len(data["data"]) == 3
    
    def test_test_endpoint_invalid_type(self):
        """Test the test endpoint with invalid type"""
        response = self.client.post(
            "/api/v1/embeddings/test",
            headers=self.auth_headers,
            params={"test_type": "invalid"}
        )
        
        assert response.status_code == 400
        data = response.json()
        assert "invalid test_type" in data["detail"].lower()
    
    # ===== HEALTH CHECK =====
    
    def test_health_check_v3(self):
        """Test health check shows v3 features"""
        response = self.client.get("/api/v1/embeddings/health")
        
        assert response.status_code == 200
        data = response.json()
        
        assert data["status"] == "healthy"
        assert data["service"] == "embeddings_v3"
        assert "features" in data
        
        features = data["features"]
        assert features["batch_processing"] == True
        assert features["dimensions_support"] == True
        assert features["token_input_support"] == True
        assert features["caching_enabled"] == True
    
    # ===== BASE64 ENCODING WITH DIMENSIONS =====
    
    def test_base64_encoding_with_dimensions(self):
        """Test base64 encoding with custom dimensions"""
        response = self.client.post(
            "/api/v1/embeddings",
            headers=self.auth_headers,
            json={
                "input": "Test base64 with dimensions",
                "model": "text-embedding-3-small",
                "encoding_format": "base64",
                "dimensions": 256
            }
        )
        
        assert response.status_code == 200
        data = response.json()
        
        # Decode and check dimensions
        embedding_b64 = data["data"][0]["embedding"]
        decoded = base64.b64decode(embedding_b64)
        array = np.frombuffer(decoded, dtype=np.float32)
        assert len(array) == 256
    
    # ===== PERFORMANCE TEST =====
    
    @pytest.mark.asyncio
    async def test_concurrent_requests(self):
        """Test handling of concurrent embedding requests"""
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            # Override auth
            async def override_get_user():
                return self.test_user
            app.dependency_overrides[get_request_user] = override_get_user
            
            # Create 10 concurrent requests
            tasks = []
            for i in range(10):
                task = client.post(
                    "/api/v1/embeddings",
                    headers=self.auth_headers,
                    json={
                        "input": f"Concurrent request {i}",
                        "model": "text-embedding-3-small",
                        "dimensions": 512 if i % 2 == 0 else None
                    }
                )
                tasks.append(task)
            
            # Execute concurrently
            responses = await asyncio.gather(*tasks)
            
            # All should succeed
            for response in responses:
                assert response.status_code == 200
                data = response.json()
                assert len(data["data"]) == 1