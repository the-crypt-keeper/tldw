"""
Simplified test configuration that works with the existing system.
"""

import pytest
import tempfile
import os
from unittest.mock import MagicMock, AsyncMock
from fastapi.testclient import TestClient
import datetime

from tldw_Server_API.app.main import app
from tldw_Server_API.app.core.AuthNZ.settings import get_settings
from tldw_Server_API.app.core.AuthNZ.jwt_service import get_jwt_service
from tldw_Server_API.app.core.AuthNZ.User_DB_Handling import User, get_request_user
from tldw_Server_API.app.core.DB_Management.ChaChaNotes_DB import CharactersRAGDB
from tldw_Server_API.app.api.v1.API_Deps.ChaCha_Notes_DB_Deps import get_chacha_db_for_user
from tldw_Server_API.app.api.v1.API_Deps.DB_Deps import get_media_db_for_user


@pytest.fixture
def test_user():
    """Create a test user object."""
    return User(
        id=1,
        username="test_user",
        email="test@example.com",
        is_active=True
    )


@pytest.fixture
def auth_token(test_user):
    """Generate a valid JWT token for the test user."""
    settings = get_settings()
    
    if settings.AUTH_MODE == "multi_user":
        jwt_service = get_jwt_service()
        # Use the correct method signature
        access_token = jwt_service.create_access_token(
            user_id=test_user.id,
            username=test_user.username,
            role="user"
        )
        return f"Bearer {access_token}"
    else:
        # For single-user mode - return the actual API key from settings
        # This should match what's in the environment
        api_key = settings.SINGLE_USER_API_KEY
        if not api_key:
            # Fallback to the test key if not set
            api_key = "test-api-key-12345"
        return api_key


@pytest.fixture
def mock_user_db(test_user):
    """Mock the user database to return our test user."""
    mock_db = MagicMock()
    
    # Mock get_user_by_id to return test user data
    async def mock_get_user_by_id(user_id):
        if user_id == test_user.id:
            return {
                "id": test_user.id,
                "username": test_user.username,
                "email": test_user.email,
                "is_active": test_user.is_active
            }
        return None
    
    # Patch the actual function
    import tldw_Server_API.app.core.DB_Management.Users_DB as users_db_module
    if hasattr(users_db_module, 'get_user_by_id'):
        users_db_module.get_user_by_id = mock_get_user_by_id
    
    return mock_db


@pytest.fixture
def mock_chacha_db(test_user):
    """Create a mock ChaChaNotes database."""
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as tmp:
        db_path = tmp.name
    
    db = CharactersRAGDB(db_path, f"user_{test_user.id}")
    
    # Add default character with the expected name
    char_id = db.add_character_card({
        "name": "Default Character",  # This is the name expected by the system
        "description": "A helpful AI assistant",
        "personality": "Helpful",
        "scenario": "General",
        "system_prompt": "You are a helpful assistant"
    })
    print(f"Created default character with ID: {char_id}")
    
    yield db
    
    # Cleanup
    try:
        os.unlink(db_path)
    except:
        pass


@pytest.fixture
def mock_media_db(test_user):
    """Create a mock media database."""
    mock_db = MagicMock()
    mock_db.client_id = f"user_{test_user.id}"
    return mock_db


@pytest.fixture(autouse=True)
def setup_dependencies(test_user, mock_user_db, mock_chacha_db, mock_media_db):
    """Override all dependencies for testing."""
    settings = get_settings()
    
    # Override authentication
    if settings.AUTH_MODE == "multi_user":
        # For multi-user mode, override to return test user
        async def mock_get_request_user(api_key=None, token=None):
            return test_user
        app.dependency_overrides[get_request_user] = mock_get_request_user
    
    # Override databases
    app.dependency_overrides[get_chacha_db_for_user] = lambda: mock_chacha_db
    app.dependency_overrides[get_media_db_for_user] = lambda: mock_media_db
    
    yield
    
    # Cleanup
    app.dependency_overrides.clear()


@pytest.fixture
def client():
    """Create test client with CSRF handling."""
    # Make sure we're using the same app instance
    from tldw_Server_API.app.main import app as main_app
    with TestClient(main_app) as test_client:
        # Get CSRF token
        response = test_client.get("/api/v1/health")
        csrf_token = response.cookies.get("csrf_token", "")
        test_client.csrf_token = csrf_token
        test_client.cookies = {"csrf_token": csrf_token}
        
        # Add helper method
        def post_with_auth(url, auth_token, **kwargs):
            headers = kwargs.pop("headers", {})
            headers["X-CSRF-Token"] = csrf_token
            
            settings = get_settings()
            if settings.AUTH_MODE == "multi_user":
                headers["Authorization"] = auth_token
            else:
                # Use X-API-KEY (with hyphen, all caps) as expected by the dependency
                headers["X-API-KEY"] = auth_token
            
            return test_client.post(url, headers=headers, **kwargs)
        
        test_client.post_with_auth = post_with_auth
        
        yield test_client