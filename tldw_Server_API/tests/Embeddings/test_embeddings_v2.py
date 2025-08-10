# test_embeddings_v2.py
# Tests for the fixed embeddings API
import json
import base64
import numpy as np
import pytest
from fastapi.testclient import TestClient
from unittest.mock import Mock, patch, MagicMock

from tldw_Server_API.app.main import app
from tldw_Server_API.app.core.AuthNZ.User_DB_Handling import User, get_request_user


class TestEmbeddingsV2:
    """Test suite for the fixed embeddings API"""
    
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
    
    def test_single_string_embedding(self):
        """Test embedding a single string"""
        response = self.client.post(
            "/api/v1/embeddings",
            headers=self.auth_headers,
            json={
                "input": "This is a test string",
                "model": "text-embedding-3-small"
            }
        )
        
        assert response.status_code == 200
        data = response.json()
        
        # Check response structure
        assert data["object"] == "list"
        assert "data" in data
        assert len(data["data"]) == 1
        assert "embedding" in data["data"][0]
        assert data["data"][0]["index"] == 0
        assert data["data"][0]["object"] == "embedding"
        assert "usage" in data
        assert data["model"] == "text-embedding-3-small"
    
    def test_batch_string_embedding(self):
        """Test embedding multiple strings"""
        test_strings = [
            "First test string",
            "Second test string",
            "Third test string"
        ]
        
        response = self.client.post(
            "/api/v1/embeddings",
            headers=self.auth_headers,
            json={
                "input": test_strings,
                "model": "text-embedding-3-small"
            }
        )
        
        assert response.status_code == 200
        data = response.json()
        
        # Check we got embeddings for all inputs
        assert len(data["data"]) == 3
        for i, embedding_data in enumerate(data["data"]):
            assert embedding_data["index"] == i
            assert isinstance(embedding_data["embedding"], list)
    
    def test_base64_encoding_format(self):
        """Test base64 output format"""
        response = self.client.post(
            "/api/v1/embeddings",
            headers=self.auth_headers,
            json={
                "input": "Test for base64 encoding",
                "model": "text-embedding-3-small",
                "encoding_format": "base64"
            }
        )
        
        assert response.status_code == 200
        data = response.json()
        
        # Check embedding is base64 string
        embedding_b64 = data["data"][0]["embedding"]
        assert isinstance(embedding_b64, str)
        
        # Verify it's valid base64
        try:
            decoded = base64.b64decode(embedding_b64)
            # Should decode to float32 array
            array = np.frombuffer(decoded, dtype=np.float32)
            assert len(array) > 0
        except Exception as e:
            pytest.fail(f"Failed to decode base64 embedding: {e}")
    
    def test_empty_string_error(self):
        """Test error handling for empty string"""
        response = self.client.post(
            "/api/v1/embeddings",
            headers=self.auth_headers,
            json={
                "input": "",
                "model": "text-embedding-3-small"
            }
        )
        
        assert response.status_code == 400
        data = response.json()
        assert "empty" in data["detail"].lower()
    
    def test_empty_list_error(self):
        """Test error handling for empty list"""
        response = self.client.post(
            "/api/v1/embeddings",
            headers=self.auth_headers,
            json={
                "input": [],
                "model": "text-embedding-3-small"
            }
        )
        
        assert response.status_code == 400
        data = response.json()
        assert "empty" in data["detail"].lower()
    
    def test_list_with_empty_strings_error(self):
        """Test error handling for list with empty strings"""
        response = self.client.post(
            "/api/v1/embeddings",
            headers=self.auth_headers,
            json={
                "input": ["valid", "", "another valid"],
                "model": "text-embedding-3-small"
            }
        )
        
        assert response.status_code == 400
        data = response.json()
        assert "empty" in data["detail"].lower()
    
    def test_exceed_batch_limit(self):
        """Test error when exceeding batch size limit"""
        # Create list with 2049 items (exceeds 2048 limit)
        huge_list = [f"string_{i}" for i in range(2049)]
        
        response = self.client.post(
            "/api/v1/embeddings",
            headers=self.auth_headers,
            json={
                "input": huge_list,
                "model": "text-embedding-3-small"
            }
        )
        
        assert response.status_code == 400
        data = response.json()
        assert "2048" in data["detail"]
    
    def test_token_array_not_supported(self):
        """Test that token arrays are not yet supported"""
        response = self.client.post(
            "/api/v1/embeddings",
            headers=self.auth_headers,
            json={
                "input": [1234, 5678, 91011],  # Token IDs
                "model": "text-embedding-3-small"
            }
        )
        
        assert response.status_code == 501  # Not Implemented
        data = response.json()
        assert "not yet supported" in data["detail"].lower()
    
    def test_authentication_required(self):
        """Test that authentication is required"""
        # Remove auth override temporarily
        app.dependency_overrides.clear()
        
        response = self.client.post(
            "/api/v1/embeddings",
            json={
                "input": "test",
                "model": "text-embedding-3-small"
            }
        )
        
        assert response.status_code == 401
        
        # Restore override
        async def override_get_user():
            return self.test_user
        app.dependency_overrides[get_request_user] = override_get_user
    
    def test_invalid_auth_token(self):
        """Test with invalid authentication token"""
        app.dependency_overrides.clear()
        
        response = self.client.post(
            "/api/v1/embeddings",
            headers={"Authorization": "Bearer invalid-token"},
            json={
                "input": "test",
                "model": "text-embedding-3-small"
            }
        )
        
        assert response.status_code == 401
        
        # Restore override
        async def override_get_user():
            return self.test_user
        app.dependency_overrides[get_request_user] = override_get_user
    
    def test_list_models_endpoint(self):
        """Test the list models endpoint"""
        response = self.client.get(
            "/api/v1/embeddings/models",
            headers=self.auth_headers
        )
        
        assert response.status_code == 200
        data = response.json()
        
        assert "models" in data
        assert "default_model" in data
        assert isinstance(data["models"], list)
        
        # Check model structure
        if data["models"]:
            model = data["models"][0]
            assert "id" in model
            assert "provider" in model
            assert "dimensions" in model
            assert "max_tokens" in model
    
    def test_test_endpoint(self):
        """Test the test endpoint"""
        response = self.client.post(
            "/api/v1/embeddings/test",
            headers=self.auth_headers
        )
        
        assert response.status_code == 200
        data = response.json()
        
        # Should return a valid embedding response
        assert data["object"] == "list"
        assert len(data["data"]) == 1
        assert "embedding" in data["data"][0]
    
    def test_health_check(self):
        """Test the health check endpoint"""
        response = self.client.get(
            "/api/v1/embeddings/health"
        )
        
        assert response.status_code == 200
        data = response.json()
        
        assert data["status"] == "healthy"
        assert data["service"] == "embeddings_v2"
        assert "implementation_available" in data
    
    def test_usage_tracking(self):
        """Test that usage information is tracked"""
        response = self.client.post(
            "/api/v1/embeddings",
            headers=self.auth_headers,
            json={
                "input": "This is a test for token counting",
                "model": "text-embedding-3-small"
            }
        )
        
        assert response.status_code == 200
        data = response.json()
        
        assert "usage" in data
        assert "prompt_tokens" in data["usage"]
        assert "total_tokens" in data["usage"]
        assert data["usage"]["prompt_tokens"] > 0
        assert data["usage"]["total_tokens"] == data["usage"]["prompt_tokens"]
    
    def test_different_models(self):
        """Test with different model names"""
        models = [
            "text-embedding-ada-002",
            "text-embedding-3-small",
            "text-embedding-3-large"
        ]
        
        for model in models:
            response = self.client.post(
                "/api/v1/embeddings",
                headers=self.auth_headers,
                json={
                    "input": f"Testing model {model}",
                    "model": model
                }
            )
            
            # May succeed or fail based on configuration
            # but should not crash with 500
            assert response.status_code in [200, 400, 404]
            
            if response.status_code == 200:
                data = response.json()
                assert data["model"] == model
    
    @pytest.mark.parametrize("encoding_format", ["float", "base64"])
    def test_encoding_formats(self, encoding_format):
        """Test both encoding formats"""
        response = self.client.post(
            "/api/v1/embeddings",
            headers=self.auth_headers,
            json={
                "input": "Test encoding formats",
                "model": "text-embedding-3-small",
                "encoding_format": encoding_format
            }
        )
        
        assert response.status_code == 200
        data = response.json()
        
        embedding = data["data"][0]["embedding"]
        if encoding_format == "float":
            assert isinstance(embedding, list)
            assert all(isinstance(x, (int, float)) for x in embedding)
        else:  # base64
            assert isinstance(embedding, str)
            # Should be valid base64
            base64.b64decode(embedding)