# conftest.py
# Pytest configuration for Prompt Studio tests

import pytest
import os
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch

# Set test environment variables
os.environ["TEST_MODE"] = "true"
os.environ["AUTH_MODE"] = "single_user"
os.environ["CSRF_ENABLED"] = "false"

@pytest.fixture(autouse=True)
def enable_test_mode(monkeypatch):
    """Enable test mode for all tests"""
    monkeypatch.setenv("TEST_MODE", "true")
    monkeypatch.setenv("AUTH_MODE", "single_user")
    monkeypatch.setenv("CSRF_ENABLED", "false")

@pytest.fixture
def mock_current_user():
    """Mock user for authentication"""
    return {
        "id": "test-user-123",
        "username": "testuser",
        "email": "test@example.com",
        "is_active": True,
        "is_authenticated": True
    }

@pytest.fixture(autouse=True)
def mock_auth_dependency(mock_current_user):
    """Automatically mock authentication for all tests"""
    with patch('tldw_Server_API.app.api.v1.API_Deps.auth_deps.get_current_user') as mock_get_user:
        mock_get_user.return_value = mock_current_user
        with patch('tldw_Server_API.app.api.v1.API_Deps.auth_deps.get_current_active_user') as mock_active:
            mock_active.return_value = mock_current_user
            yield

@pytest.fixture(scope="session")
def test_db_path():
    """Session-scoped test database path"""
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir) / "test_prompt_studio.db"
        yield str(db_path)

@pytest.fixture
def isolated_db(test_db_path):
    """Isolated database for each test"""
    from tldw_Server_API.app.core.DB_Management.PromptStudioDatabase import PromptStudioDatabase
    
    # Create unique database for this test
    import uuid
    unique_path = f"{test_db_path}_{uuid.uuid4()}.db"
    db = PromptStudioDatabase(unique_path, "test-client")
    
    yield db
    
    # Cleanup
    if os.path.exists(unique_path):
        os.unlink(unique_path)

@pytest.fixture
def auth_headers():
    """Authentication headers for API tests"""
    return {
        "Authorization": "Bearer test-token",
        "Content-Type": "application/json"
    }
