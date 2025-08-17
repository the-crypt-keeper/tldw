# test_embeddings_v5_unit.py
# Unit tests for production embeddings service (with mocking)

import asyncio
import time
from unittest.mock import AsyncMock, MagicMock, patch, Mock
import pytest
import numpy as np

from fastapi.testclient import TestClient
from tldw_Server_API.app.main import app
from tldw_Server_API.app.core.AuthNZ.User_DB_Handling import User, get_request_user


@pytest.mark.unit
class TestEmbeddingsUnit:
    """Unit tests with mocked dependencies"""
    
    @pytest.fixture(autouse=True)
    async def setup(self):
        """Setup test environment"""
        self.client = TestClient(app)
        self.auth_headers = {"Authorization": "Bearer test-api-key"}
        
        self.regular_user = User(
            id=1, 
            username="testuser", 
            email="test@example.com", 
            is_active=True,
            is_admin=False
        )
        
        self.admin_user = User(
            id=2,
            username="admin",
            email="admin@example.com", 
            is_active=True,
            is_admin=True
        )
        
        yield
        app.dependency_overrides.clear()


class TestCriticalSecurity(TestEmbeddingsUnit):
    """Test critical security fixes"""
    
    @pytest.mark.unit
    def test_no_placeholder_embeddings(self):
        """Verify system fails properly when dependencies missing"""
        with patch('tldw_Server_API.app.api.v1.endpoints.embeddings_v5_production.EMBEDDINGS_AVAILABLE', False):
            with pytest.raises(RuntimeError, match="Embeddings service dependencies not available"):
                from tldw_Server_API.app.api.v1.endpoints.embeddings_v5_production import create_embeddings_batch
    
    @pytest.mark.unit
    async def test_admin_authorization_required(self):
        """Test admin endpoints require proper authorization"""
        async def override_regular_user():
            return self.regular_user
        
        app.dependency_overrides[get_request_user] = override_regular_user
        
        response = self.client.delete(
            "/api/v1/embeddings/cache",
            headers=self.auth_headers
        )
        
        assert response.status_code == 403
        assert "Admin privileges required" in response.json()["detail"]


class TestTTLCache:
    """Test TTL cache implementation"""
    
    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_cache_ttl_expiration(self):
        """Test that cache entries expire after TTL"""
        from tldw_Server_API.app.api.v1.endpoints.embeddings_v5_production import TTLCache
        
        cache = TTLCache(max_size=10, ttl_seconds=1)
        
        await cache.set("test_key", [1.0, 2.0, 3.0])
        value = await cache.get("test_key")
        assert value == [1.0, 2.0, 3.0]
        
        await asyncio.sleep(1.5)
        
        value = await cache.get("test_key")
        assert value is None
    
    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_cache_lru_eviction(self):
        """Test LRU eviction when cache is full"""
        from tldw_Server_API.app.api.v1.endpoints.embeddings_v5_production import TTLCache
        
        cache = TTLCache(max_size=3, ttl_seconds=3600)
        
        await cache.set("key1", [1.0])
        await cache.set("key2", [2.0])
        await cache.set("key3", [3.0])
        
        await cache.get("key1")  # Make key1 recently used
        
        await cache.set("key4", [4.0])  # Should evict key2
        
        assert await cache.get("key1") == [1.0]
        assert await cache.get("key3") == [3.0]
        assert await cache.get("key4") == [4.0]
        assert await cache.get("key2") is None
    
    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_cache_thread_safety(self):
        """Test cache is thread-safe under concurrent access"""
        from tldw_Server_API.app.api.v1.endpoints.embeddings_v5_production import TTLCache
        
        cache = TTLCache(max_size=100, ttl_seconds=3600)
        
        async def add_items(start_idx):
            for i in range(10):
                key = f"key_{start_idx}_{i}"
                await cache.set(key, [float(start_idx + i)])
        
        async def get_items(start_idx):
            for i in range(10):
                key = f"key_{start_idx}_{i}"
                await cache.get(key)
        
        tasks = []
        for i in range(10):
            tasks.append(add_items(i * 10))
            tasks.append(get_items(i * 10))
        
        await asyncio.gather(*tasks)
        
        stats = cache.stats()
        assert stats['size'] <= 100


class TestConnectionPooling:
    """Test connection pool management"""
    
    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_connection_pool_creation(self):
        """Test connection pools are created per provider"""
        from tldw_Server_API.app.api.v1.endpoints.embeddings_v5_production import ConnectionPoolManager
        
        manager = ConnectionPoolManager()
        
        try:
            session1 = await manager.get_session("openai")
            session2 = await manager.get_session("cohere")
            session3 = await manager.get_session("openai")
            
            assert len(manager.pools) == 2
            assert session1 is session3
            assert session1 is not session2
            
        finally:
            await manager.close_all()


class TestRetryLogic:
    """Test retry logic and error handling"""
    
    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_retry_on_connection_error(self):
        """Test that connection errors trigger retries"""
        from tldw_Server_API.app.api.v1.endpoints.embeddings_v5_production import create_embeddings_with_retry
        
        attempt_count = 0
        
        async def mock_embeddings(texts, config, model_id):
            nonlocal attempt_count
            attempt_count += 1
            
            if attempt_count < 3:
                raise ConnectionError("Connection failed")
            
            return [[1.0, 2.0, 3.0]] * len(texts)
        
        with patch('tldw_Server_API.app.api.v1.endpoints.embeddings_v5_production.create_embeddings_batch', mock_embeddings):
            result = await create_embeddings_with_retry(
                ["test text"],
                {},
                "test-model"
            )
            
            assert attempt_count == 3
            assert result == [[1.0, 2.0, 3.0]]
    
    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_no_retry_on_value_error(self):
        """Test that value errors don't trigger retries"""
        from tldw_Server_API.app.api.v1.endpoints.embeddings_v5_production import create_embeddings_with_retry
        
        attempt_count = 0
        
        async def mock_embeddings(texts, config, model_id):
            nonlocal attempt_count
            attempt_count += 1
            raise ValueError("Invalid input")
        
        with patch('tldw_Server_API.app.api.v1.endpoints.embeddings_v5_production.create_embeddings_batch', mock_embeddings):
            with pytest.raises(ValueError):
                await create_embeddings_with_retry(
                    ["test text"],
                    {},
                    "test-model"
                )
            
            assert attempt_count == 1


class TestErrorHandling:
    """Test error handling with mocked dependencies"""
    
    @pytest.mark.unit
    def test_empty_input_error(self, setup):
        """Test error on empty input"""
        async def override_user():
            return setup.regular_user
        
        app.dependency_overrides[get_request_user] = override_user
        
        response = setup.client.post(
            "/api/v1/embeddings",
            headers=setup.auth_headers,
            json={
                "input": "",
                "model": "text-embedding-3-small"
            }
        )
        
        assert response.status_code == 400
        assert "Input cannot be empty" in response.json()["detail"]
    
    @pytest.mark.unit
    def test_invalid_provider_error(self, setup):
        """Test error on invalid provider"""
        async def override_user():
            return setup.regular_user
        
        app.dependency_overrides[get_request_user] = override_user
        
        response = setup.client.post(
            "/api/v1/embeddings",
            headers={**setup.auth_headers, "x-provider": "invalid_provider"},
            json={
                "input": "test text",
                "model": "some-model"
            }
        )
        
        assert response.status_code == 400
        assert "Unknown provider" in response.json()["detail"]


class TestMockedFlow:
    """Test complete flow with mocked embeddings"""
    
    @pytest.mark.unit
    def test_end_to_end_flow_mocked(self, setup):
        """Test complete flow with mocked embeddings"""
        async def override_user():
            return setup.regular_user
        
        app.dependency_overrides[get_request_user] = override_user
        
        async def mock_embeddings(texts, provider, model_id, dimensions, api_key, api_url):
            return [[float(i), float(i+1), float(i+2)] for i, _ in enumerate(texts)]
        
        with patch('tldw_Server_API.app.api.v1.endpoints.embeddings_v5_production.create_embeddings_batch_async', mock_embeddings):
            
            response = setup.client.post(
                "/api/v1/embeddings",
                headers=setup.auth_headers,
                json={
                    "input": ["text1", "text2", "text3"],
                    "model": "text-embedding-3-small"
                }
            )
            
            assert response.status_code == 200
            data = response.json()
            
            assert "data" in data
            assert "model" in data
            assert "usage" in data
            assert len(data["data"]) == 3
    
    @pytest.mark.unit
    def test_caching_behavior_mocked(self, setup):
        """Test caching behavior with mocked API calls"""
        async def override_user():
            return setup.regular_user
        
        app.dependency_overrides[get_request_user] = override_user
        
        call_count = 0
        
        async def mock_embeddings(texts, provider, model_id, dimensions, api_key, api_url):
            nonlocal call_count
            call_count += 1
            return [[1.0, 2.0, 3.0]] * len(texts)
        
        with patch('tldw_Server_API.app.api.v1.endpoints.embeddings_v5_production.create_embeddings_with_retry', mock_embeddings):
            
            response1 = setup.client.post(
                "/api/v1/embeddings",
                headers=setup.auth_headers,
                json={
                    "input": "cached text",
                    "model": "text-embedding-3-small"
                }
            )
            
            assert response1.status_code == 200
            assert call_count == 1
            
            response2 = setup.client.post(
                "/api/v1/embeddings",
                headers=setup.auth_headers,
                json={
                    "input": "cached text",
                    "model": "text-embedding-3-small"
                }
            )
            
            assert response2.status_code == 200
            assert call_count == 1  # Should use cache
            
            response3 = setup.client.post(
                "/api/v1/embeddings",
                headers=setup.auth_headers,
                json={
                    "input": "different text",
                    "model": "text-embedding-3-small"
                }
            )
            
            assert response3.status_code == 200
            assert call_count == 2