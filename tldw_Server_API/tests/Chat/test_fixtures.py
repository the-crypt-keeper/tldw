"""
Shared test fixtures using real database instances for integration tests.
"""
import pytest
import tempfile
import os
import shutil
from pathlib import Path

from tldw_Server_API.app.core.DB_Management.ChaChaNotes_DB import CharactersRAGDB
from tldw_Server_API.app.core.DB_Management.Media_DB_v2 import MediaDatabase
from tldw_Server_API.app.core.AuthNZ.User_DB_Handling import User
from tldw_Server_API.app.core.AuthNZ.settings import get_settings
from tldw_Server_API.app.main import app


@pytest.fixture(scope="function")
def real_test_db():
    """Create a real temporary ChaChaNotes database for testing."""
    # Create a temporary directory for the test database
    temp_dir = tempfile.mkdtemp(prefix="test_chacha_")
    db_path = os.path.join(temp_dir, "test_chacha.db")
    
    # Initialize real database
    db = CharactersRAGDB(db_path, client_id="test_client")
    
    # Add default character
    char_id = db.add_character_card({
        "name": "Default Character",
        "description": "A helpful AI assistant for testing",
        "personality": "Helpful and friendly",
        "scenario": "Testing environment",
        "system_prompt": "You are a helpful test assistant",
        "first_message": "Hello! I'm here to help with testing.",
        "creator_notes": "Created for integration testing"
    })
    
    # Add a test character
    test_char_id = db.add_character_card({
        "name": "TestCharacter",
        "description": "A specific test character",
        "personality": "Test personality",
        "scenario": "Test scenario",
        "system_prompt": "Test system prompt",
        "first_message": "Test first message",
        "creator_notes": "Test notes"
    })
    
    # Create a test conversation with messages
    conv_id = db.create_conversation(
        character_id=char_id,
        conversation_name="Test Conversation",
        client_id="test_client"
    )
    
    # Add some test messages to the conversation
    db.add_message(
        conversation_id=conv_id,
        sender="user",
        content="Hello, how are you?",
        client_id="test_client"
    )
    
    db.add_message(
        conversation_id=conv_id,
        sender="assistant", 
        content="I'm doing well, thank you! How can I help you today?",
        client_id="test_client"
    )
    
    db.add_message(
        conversation_id=conv_id,
        sender="user",
        content="Can you explain quantum computing?",
        client_id="test_client"
    )
    
    db.add_message(
        conversation_id=conv_id,
        sender="assistant",
        content="Quantum computing uses quantum bits (qubits) that can exist in superposition...",
        client_id="test_client"
    )
    
    yield db
    
    # Cleanup
    try:
        # Close any open connections
        if hasattr(db, 'close_connection'):
            db.close_connection()
        # Remove the temporary directory
        shutil.rmtree(temp_dir, ignore_errors=True)
    except Exception as e:
        print(f"Cleanup error: {e}")


@pytest.fixture(scope="function")
def real_media_db():
    """Create a real temporary Media database for testing."""
    # Create a temporary directory for the test database
    temp_dir = tempfile.mkdtemp(prefix="test_media_")
    db_path = os.path.join(temp_dir, "test_media.db")
    
    # Initialize real database
    db = MediaDatabase(db_path, client_id="test_client")
    
    yield db
    
    # Cleanup
    try:
        # Close any open connections
        if hasattr(db, 'close_connection'):
            db.close_connection()
        # Remove the temporary directory
        shutil.rmtree(temp_dir, ignore_errors=True)
    except Exception as e:
        print(f"Cleanup error: {e}")


@pytest.fixture(scope="function")
def test_user():
    """Create a test user object."""
    return User(
        id=1,
        username="test_user",
        email="test@example.com",
        is_active=True
    )


@pytest.fixture(scope="function")
def auth_headers(test_user):
    """Generate proper authentication headers."""
    settings = get_settings()
    
    if settings.AUTH_MODE == "multi_user":
        from tldw_Server_API.app.core.AuthNZ.jwt_service import get_jwt_service
        jwt_service = get_jwt_service()
        access_token = jwt_service.create_access_token(
            user_id=test_user.id,
            username=test_user.username,
            role="user"
        )
        return {
            "Authorization": f"Bearer {access_token}",
            "X-CSRF-Token": ""
        }
    else:
        api_key = settings.SINGLE_USER_API_KEY or "test-api-key-12345"
        return {
            "X-API-KEY": api_key,
            "X-CSRF-Token": ""
        }


@pytest.fixture(scope="function", autouse=True)
def setup_test_auth(test_user, real_test_db, real_media_db):
    """Set up authentication and database dependencies for tests."""
    from tldw_Server_API.app.api.v1.API_Deps.ChaCha_Notes_DB_Deps import get_chacha_db_for_user
    from tldw_Server_API.app.api.v1.API_Deps.DB_Deps import get_media_db_for_user
    from tldw_Server_API.app.core.AuthNZ.User_DB_Handling import get_request_user
    
    # Override authentication to return test user
    async def mock_get_request_user(api_key=None, token=None):
        return test_user
    
    # Override database dependencies to return real test databases
    app.dependency_overrides[get_request_user] = mock_get_request_user
    app.dependency_overrides[get_chacha_db_for_user] = lambda: real_test_db
    app.dependency_overrides[get_media_db_for_user] = lambda: real_media_db
    
    yield
    
    # Cleanup - don't clear, let the autouse fixture handle it