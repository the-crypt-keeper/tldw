# conftest.py - Shared test fixtures for embeddings tests
"""
Shared fixtures and configuration for embeddings tests.
Provides proper test environment setup including:
- CSRF token handling
- Prometheus metrics mocking
- Database setup
- Authentication mocking
"""

import pytest
from unittest.mock import MagicMock, AsyncMock, patch
from fastapi.testclient import TestClient
from tldw_Server_API.app.main import app
from tldw_Server_API.app.core.AuthNZ.User_DB_Handling import User


@pytest.fixture(autouse=True)
def reset_app_state():
    """Reset app state before each test"""
    app.dependency_overrides.clear()
    yield
    app.dependency_overrides.clear()


@pytest.fixture
def mock_prometheus_metrics():
    """Mock Prometheus metrics to avoid registration conflicts"""
    mock_counter = MagicMock()
    mock_counter.labels.return_value.inc = MagicMock()
    
    mock_histogram = MagicMock()
    mock_histogram.labels.return_value.observe = MagicMock()
    
    mock_gauge = MagicMock()
    mock_gauge.inc = MagicMock()
    mock_gauge.dec = MagicMock()
    mock_gauge.set = MagicMock()
    
    # Patch all metrics in the embeddings module
    with patch.multiple(
        'tldw_Server_API.app.api.v1.endpoints.embeddings_v5_production_enhanced',
        embedding_requests_total=mock_counter,
        embedding_request_duration=mock_histogram,
        embedding_cache_hits=mock_counter,
        embedding_cache_size=mock_gauge,
        active_embedding_requests=mock_gauge
    ):
        yield {
            'counter': mock_counter,
            'histogram': mock_histogram,
            'gauge': mock_gauge
        }


@pytest.fixture
def test_client(mock_prometheus_metrics):
    """Create test client with proper CSRF setup"""
    client = TestClient(app)
    
    # Set CSRF token in cookie and get it for headers
    csrf_token = "test-csrf-token-12345"
    client.cookies.set("csrf_token", csrf_token)
    
    # Add CSRF token to default headers
    client.headers["X-CSRF-Token"] = csrf_token
    client.headers["Authorization"] = "Bearer test-api-key"
    
    return client


@pytest.fixture
def regular_user():
    """Create a regular test user"""
    return User(
        id=1,
        username="testuser",
        email="test@example.com",
        is_active=True,
        is_admin=False
    )


@pytest.fixture
def admin_user():
    """Create an admin test user"""
    return User(
        id=2,
        username="admin",
        email="admin@example.com",
        is_active=True,
        is_admin=True
    )


@pytest.fixture
def auth_headers():
    """Get authentication headers with CSRF token"""
    return {
        "Authorization": "Bearer test-api-key",
        "X-CSRF-Token": "test-csrf-token-12345"
    }


@pytest.fixture
def mock_redis():
    """Mock Redis client for worker tests"""
    mock_client = AsyncMock()
    mock_client.xadd = AsyncMock()
    mock_client.xreadgroup = AsyncMock(return_value=[])
    mock_client.xack = AsyncMock()
    mock_client.hset = AsyncMock()
    mock_client.expire = AsyncMock()
    mock_client.setex = AsyncMock()
    mock_client.xlen = AsyncMock(return_value=0)
    mock_client.close = AsyncMock()
    
    return mock_client


@pytest.fixture
def mock_chromadb():
    """Mock ChromaDB manager"""
    mock_manager = MagicMock()
    mock_manager.add_embeddings = MagicMock(return_value=True)
    mock_manager.search = MagicMock(return_value=[])
    mock_manager.delete = MagicMock(return_value=True)
    
    return mock_manager