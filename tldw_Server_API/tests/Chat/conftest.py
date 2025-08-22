"""
Simplified test configuration that works with the existing system.
"""

import pytest
import tempfile
import os
# Set environment variables BEFORE any tldw imports
import os
os.environ["OPENAI_API_KEY"] = "sk-mock-key-12345"
os.environ["OPENAI_API_BASE"] = "http://localhost:8080/v1"

import subprocess
import time
import requests
import atexit
import signal
import sys
from pathlib import Path
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


# Global variable to track mock server process
_mock_server_process = None


def cleanup_mock_server():
    """Cleanup function to ensure mock server is stopped."""
    global _mock_server_process
    if _mock_server_process:
        try:
            _mock_server_process.terminate()
            _mock_server_process.wait(timeout=5)
        except:
            try:
                _mock_server_process.kill()
            except:
                pass
        _mock_server_process = None


# Register cleanup function
atexit.register(cleanup_mock_server)


@pytest.fixture(scope="session")
def mock_openai_server():
    """Start the mock OpenAI server for testing."""
    global _mock_server_process
    
    # Check if server is already running
    try:
        response = requests.get("http://localhost:8080/v1/models", timeout=1)
        if response.status_code == 200:
            print("Mock OpenAI server already running")
            yield "http://localhost:8080"
            return
    except:
        pass
    
    # FIXME: Once mock_openai_server is published as a PyPI package,
    # replace this relative path approach with:
    # from mock_openai_server import start_server
    # or use: python -m mock_openai_server
    
    # Get the path to mock_openai_server relative to this test file
    test_dir = Path(__file__).parent
    project_root = test_dir.parent.parent.parent  # Go up to tldw_server root
    mock_server_dir = project_root / "mock_openai_server"
    
    if not mock_server_dir.exists():
        pytest.skip("mock_openai_server not found - skipping integration tests")
    
    # Start the mock server
    print(f"Starting mock OpenAI server from {mock_server_dir}...")
    _mock_server_process = subprocess.Popen(
        [sys.executable, "-m", "mock_openai.server"],
        cwd=str(mock_server_dir),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,  # Combine stderr with stdout
        text=True  # Text mode for easier debugging
    )
    
    # Wait for server to start
    max_retries = 60  # Increase retries
    for i in range(max_retries):
        # Check if process is still running
        if _mock_server_process.poll() is not None:
            # Process has died, get output for debugging
            output = _mock_server_process.stdout.read()
            cleanup_mock_server()
            pytest.fail(f"Mock server process died. Output:\n{output}")
        
        try:
            # Test with the auth header that the mock server expects
            response = requests.get(
                "http://localhost:8080/v1/models", 
                headers={"Authorization": "Bearer sk-mock-key-12345"},
                timeout=1
            )
            if response.status_code == 200:
                print("Mock OpenAI server started successfully")
                break
        except requests.exceptions.ConnectionError:
            # Server not ready yet
            time.sleep(0.5)
        except Exception as e:
            print(f"Error checking server: {e}")
            time.sleep(0.5)
    else:
        # Get any output for debugging
        output = ""
        if _mock_server_process and _mock_server_process.stdout:
            try:
                output = _mock_server_process.stdout.read()
            except:
                pass
        cleanup_mock_server()
        pytest.fail(f"Failed to start mock OpenAI server after {max_retries} retries. Output:\n{output}")
    
    yield "http://localhost:8080"
    
    # Cleanup
    cleanup_mock_server()


@pytest.fixture(autouse=True)
def configure_for_mock_server(mock_openai_server, monkeypatch):
    """Configure the application to use the mock OpenAI server."""
    # Ensure environment variables are set
    monkeypatch.setenv("OPENAI_API_KEY", "sk-mock-key-12345")
    monkeypatch.setenv("OPENAI_API_BASE", f"{mock_openai_server}/v1")
    
    # Also set custom endpoint variables
    monkeypatch.setenv("CUSTOM_OPENAI_API_IP", f"{mock_openai_server}/v1/chat/completions")
    monkeypatch.setenv("CUSTOM_OPENAI_API_KEY", "sk-mock-key-12345")
    
    # Reload the schemas module to pick up the new environment variables
    import importlib
    import tldw_Server_API.app.api.v1.schemas.chat_request_schemas as chat_schemas
    importlib.reload(chat_schemas)
    
    # Update API_KEYS directly
    chat_schemas.API_KEYS['openai'] = 'sk-mock-key-12345'
    
    # Also patch the chat endpoint's imported API_KEYS
    from tldw_Server_API.app.api.v1.endpoints import chat as chat_endpoint
    if hasattr(chat_endpoint, 'API_KEYS'):
        chat_endpoint.API_KEYS = chat_schemas.API_KEYS
    
    # Patch the OpenAI API URL in the config
    from tldw_Server_API.app.core.config import load_and_log_configs
    config = load_and_log_configs()
    if 'openai_api' not in config:
        config['openai_api'] = {}
    config['openai_api']['api_key'] = 'sk-mock-key-12345'
    config['openai_api']['api_base_url'] = f'{mock_openai_server}/v1'
    
    # Patch the load_and_log_configs function to return our patched config
    def mock_load_and_log_configs():
        return config
    
    monkeypatch.setattr('tldw_Server_API.app.core.config.load_and_log_configs', mock_load_and_log_configs)
    monkeypatch.setattr('tldw_Server_API.app.core.LLM_Calls.LLM_API_Calls.load_and_log_configs', mock_load_and_log_configs)
    
    yield


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


@pytest.fixture
def authenticated_client(client, auth_token):
    """Create an authenticated test client."""
    settings = get_settings()
    
    # Add authentication to all requests
    original_post = client.post
    
    def authenticated_post(url, **kwargs):
        headers = kwargs.pop("headers", {})
        
        if settings.AUTH_MODE == "multi_user":
            headers["Authorization"] = auth_token
        else:
            headers["X-API-KEY"] = auth_token
        
        return original_post(url, headers=headers, **kwargs)
    
    client.post = authenticated_post
    return client


def get_auth_headers(auth_token, csrf_token=""):
    """Helper function to get authentication headers."""
    settings = get_settings()
    headers = {"X-CSRF-Token": csrf_token}
    
    if settings.AUTH_MODE == "multi_user":
        headers["Authorization"] = auth_token
    else:
        # Use X-API-KEY (with hyphen, all caps) as expected by the dependency
        headers["X-API-KEY"] = auth_token
    
    return headers