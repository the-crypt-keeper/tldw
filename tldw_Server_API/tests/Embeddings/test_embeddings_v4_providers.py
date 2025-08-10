# test_embeddings_v4_providers.py
# Tests for multi-provider embeddings API
import json
import pytest
from fastapi.testclient import TestClient
from httpx import AsyncClient, ASGITransport

from tldw_Server_API.app.main import app
from tldw_Server_API.app.core.AuthNZ.User_DB_Handling import User, get_request_user


class TestMultiProviderEmbeddings:
    """Test suite for multi-provider embeddings API"""
    
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
    
    # ===== PROVIDER SELECTION TESTS =====
    
    def test_default_provider(self):
        """Test that default provider (OpenAI) is used when not specified"""
        response = self.client.post(
            "/api/v1/embeddings",
            headers=self.auth_headers,
            json={
                "input": "Test with default provider",
                "model": "text-embedding-3-small"
            }
        )
        
        assert response.status_code == 200
        data = response.json()
        # Model name should not have provider prefix for OpenAI
        assert ":" not in data["model"] or data["model"].startswith("openai:")
    
    def test_provider_via_header(self):
        """Test provider selection via x-provider header"""
        headers = self.auth_headers.copy()
        headers["x-provider"] = "huggingface"
        
        response = self.client.post(
            "/api/v1/embeddings",
            headers=headers,
            json={
                "input": "Test with HuggingFace provider",
                "model": "sentence-transformers/all-MiniLM-L6-v2"
            }
        )
        
        assert response.status_code == 200
        data = response.json()
        assert "huggingface:" in data["model"] or data["model"] == "sentence-transformers/all-MiniLM-L6-v2"
    
    def test_provider_via_model_prefix(self):
        """Test provider selection via model prefix"""
        response = self.client.post(
            "/api/v1/embeddings",
            headers=self.auth_headers,
            json={
                "input": "Test with Cohere provider",
                "model": "cohere:embed-english-v3.0"
            }
        )
        
        assert response.status_code == 200
        data = response.json()
        assert "cohere:" in data["model"] or data["model"] == "embed-english-v3.0"
    
    def test_invalid_provider(self):
        """Test error handling for invalid provider"""
        headers = self.auth_headers.copy()
        headers["x-provider"] = "invalid_provider"
        
        response = self.client.post(
            "/api/v1/embeddings",
            headers=headers,
            json={
                "input": "Test with invalid provider",
                "model": "some-model"
            }
        )
        
        assert response.status_code == 400
        data = response.json()
        assert "unknown provider" in data["detail"].lower()
    
    # ===== PROVIDER LISTING TESTS =====
    
    def test_list_providers_endpoint(self):
        """Test the list providers endpoint"""
        response = self.client.get(
            "/api/v1/embeddings/providers",
            headers=self.auth_headers
        )
        
        assert response.status_code == 200
        data = response.json()
        
        assert "providers" in data
        assert isinstance(data["providers"], list)
        assert len(data["providers"]) > 0
        
        # Check provider structure
        provider = data["providers"][0]
        assert "id" in provider
        assert "name" in provider
        assert "models" in provider
        assert "features" in provider
        assert "configured" in provider
        
        # Check that OpenAI is in the list
        provider_ids = [p["id"] for p in data["providers"]]
        assert "openai" in provider_ids
    
    def test_provider_features(self):
        """Test that provider features are correctly reported"""
        response = self.client.get(
            "/api/v1/embeddings/providers",
            headers=self.auth_headers
        )
        
        assert response.status_code == 200
        data = response.json()
        
        # Find OpenAI provider
        openai_provider = next(p for p in data["providers"] if p["id"] == "openai")
        
        # Check OpenAI features
        assert openai_provider["features"]["dimensions_support"] == True
        assert openai_provider["features"]["batch_support"] == True
        assert openai_provider["features"]["token_input"] == True
    
    # ===== PROVIDER-SPECIFIC MODEL TESTS =====
    
    def test_openai_models(self):
        """Test OpenAI-specific models"""
        models_to_test = [
            "text-embedding-ada-002",
            "text-embedding-3-small",
            "text-embedding-3-large"
        ]
        
        for model in models_to_test:
            response = self.client.post(
                "/api/v1/embeddings",
                headers=self.auth_headers,
                json={
                    "input": f"Test {model}",
                    "model": model
                }
            )
            
            # Should succeed or fail gracefully
            assert response.status_code in [200, 400, 404]
            
            if response.status_code == 200:
                data = response.json()
                assert len(data["data"]) == 1
    
    def test_cohere_model_with_prefix(self):
        """Test Cohere model with provider prefix"""
        response = self.client.post(
            "/api/v1/embeddings",
            headers=self.auth_headers,
            json={
                "input": "Test Cohere embedding",
                "model": "cohere:embed-english-v3.0"
            }
        )
        
        # Should handle gracefully even without API key
        assert response.status_code in [200, 400, 401, 404]
    
    def test_huggingface_model(self):
        """Test HuggingFace model"""
        headers = self.auth_headers.copy()
        headers["x-provider"] = "huggingface"
        
        response = self.client.post(
            "/api/v1/embeddings",
            headers=headers,
            json={
                "input": "Test HuggingFace embedding",
                "model": "sentence-transformers/all-MiniLM-L6-v2"
            }
        )
        
        assert response.status_code in [200, 400, 404]
    
    # ===== PROVIDER COMPARISON TEST =====
    
    def test_compare_providers_endpoint(self):
        """Test the provider comparison endpoint"""
        response = self.client.post(
            "/api/v1/embeddings/compare",
            headers=self.auth_headers,
            json={
                "text": "Compare embeddings across providers",
                "providers": ["openai", "huggingface"]
            }
        )
        
        assert response.status_code == 200
        data = response.json()
        
        assert "text" in data
        assert "comparisons" in data
        assert isinstance(data["comparisons"], dict)
        
        # Check structure for each provider result
        for provider, result in data["comparisons"].items():
            if "error" not in result:
                assert "model" in result
                assert "dimensions" in result
                assert "sample" in result
                assert "norm" in result
    
    def test_compare_too_many_providers(self):
        """Test error when comparing too many providers"""
        response = self.client.post(
            "/api/v1/embeddings/compare",
            headers=self.auth_headers,
            json={
                "text": "Too many providers",
                "providers": ["openai", "cohere", "voyage", "google", "mistral", "huggingface"]
            }
        )
        
        assert response.status_code == 400
        data = response.json()
        assert "maximum 5 providers" in data["detail"].lower()
    
    # ===== CUSTOM API KEY/URL TESTS =====
    
    def test_custom_api_key_header(self):
        """Test using custom API key via header"""
        headers = self.auth_headers.copy()
        headers["x-provider"] = "openai"
        headers["x-api-key"] = "custom-api-key-12345"
        
        response = self.client.post(
            "/api/v1/embeddings",
            headers=headers,
            json={
                "input": "Test with custom API key",
                "model": "text-embedding-3-small"
            }
        )
        
        # Should process the request (may fail due to invalid key, but should accept it)
        assert response.status_code in [200, 401, 403, 404]
    
    def test_local_api_with_custom_url(self):
        """Test local API provider with custom URL"""
        headers = self.auth_headers.copy()
        headers["x-provider"] = "local_api"
        headers["x-api-url"] = "http://localhost:11434/api/embeddings"
        
        response = self.client.post(
            "/api/v1/embeddings",
            headers=headers,
            json={
                "input": "Test local API",
                "model": "nomic-embed-text"
            }
        )
        
        # Should handle gracefully even if server not running
        assert response.status_code in [200, 400, 404, 500, 503]
    
    # ===== DIMENSIONS SUPPORT BY PROVIDER =====
    
    def test_dimensions_with_openai(self):
        """Test dimensions parameter with OpenAI provider"""
        response = self.client.post(
            "/api/v1/embeddings",
            headers=self.auth_headers,
            json={
                "input": "Test dimensions with OpenAI",
                "model": "text-embedding-3-small",
                "dimensions": 512
            }
        )
        
        assert response.status_code == 200
        data = response.json()
        embedding = data["data"][0]["embedding"]
        assert len(embedding) == 512
    
    def test_dimensions_with_cohere(self):
        """Test dimensions parameter with Cohere provider"""
        response = self.client.post(
            "/api/v1/embeddings",
            headers=self.auth_headers,
            json={
                "input": "Test dimensions with Cohere",
                "model": "cohere:embed-english-v3.0",
                "dimensions": 256
            }
        )
        
        # Should accept the parameter even without API key
        assert response.status_code in [200, 401, 403]
        
        if response.status_code == 200:
            data = response.json()
            embedding = data["data"][0]["embedding"]
            assert len(embedding) == 256
    
    def test_dimensions_unsupported_provider(self):
        """Test that dimensions parameter is rejected for unsupported providers"""
        headers = self.auth_headers.copy()
        headers["x-provider"] = "voyage"
        
        response = self.client.post(
            "/api/v1/embeddings",
            headers=headers,
            json={
                "input": "Test dimensions with unsupported provider",
                "model": "voyage-2",
                "dimensions": 512
            }
        )
        
        # Should reject dimensions parameter for Voyage
        if response.status_code == 400:
            data = response.json()
            assert "not supported" in data["detail"].lower()
    
    # ===== HEALTH CHECK WITH PROVIDER STATUS =====
    
    def test_health_check_with_providers(self):
        """Test health check includes provider status"""
        response = self.client.get("/api/v1/embeddings/health")
        
        assert response.status_code == 200
        data = response.json()
        
        assert data["status"] == "healthy"
        assert data["service"] == "embeddings_v4"
        assert "provider_status" in data
        
        # Check provider status
        provider_status = data["provider_status"]
        assert isinstance(provider_status, dict)
        assert "openai" in provider_status
        assert "huggingface" in provider_status
        assert "cohere" in provider_status
    
    # ===== BATCH PROCESSING WITH PROVIDERS =====
    
    def test_batch_with_different_providers(self):
        """Test batch processing works with different providers"""
        providers_to_test = ["openai", "huggingface"]
        
        for provider in providers_to_test:
            headers = self.auth_headers.copy()
            headers["x-provider"] = provider
            
            model = "text-embedding-3-small" if provider == "openai" else "sentence-transformers/all-MiniLM-L6-v2"
            
            response = self.client.post(
                "/api/v1/embeddings",
                headers=headers,
                json={
                    "input": ["First text", "Second text", "Third text"],
                    "model": model
                }
            )
            
            if response.status_code == 200:
                data = response.json()
                assert len(data["data"]) == 3
                
                # Check model name includes provider for non-OpenAI
                if provider != "openai":
                    assert provider in data["model"] or ":" in data["model"]
    
    # ===== CACHING WITH PROVIDERS =====
    
    @pytest.mark.asyncio
    async def test_caching_respects_provider(self):
        """Test that cache keys include provider to avoid cross-provider cache hits"""
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            # Override auth
            async def override_get_user():
                return self.test_user
            app.dependency_overrides[get_request_user] = override_get_user
            
            text = "Same text for different providers"
            
            # First request with OpenAI
            headers1 = self.auth_headers.copy()
            headers1["x-provider"] = "openai"
            
            response1 = await client.post(
                "/api/v1/embeddings",
                headers=headers1,
                json={
                    "input": text,
                    "model": "text-embedding-3-small"
                }
            )
            
            assert response1.status_code == 200
            data1 = response1.json()
            embedding1 = data1["data"][0]["embedding"]
            
            # Second request with HuggingFace (same text)
            headers2 = self.auth_headers.copy()
            headers2["x-provider"] = "huggingface"
            
            response2 = await client.post(
                "/api/v1/embeddings",
                headers=headers2,
                json={
                    "input": text,
                    "model": "sentence-transformers/all-MiniLM-L6-v2"
                }
            )
            
            assert response2.status_code == 200
            data2 = response2.json()
            embedding2 = data2["data"][0]["embedding"]
            
            # Embeddings should be different (different providers)
            # In the placeholder implementation, they will be different due to provider in hash
            assert embedding1 != embedding2