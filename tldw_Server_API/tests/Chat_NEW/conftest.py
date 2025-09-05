"""
Chat Module Test Configuration and Fixtures

Provides fixtures for testing the chat functionality with proper separation
of unit, integration, and property tests. Focuses on the OpenAI-compatible
/chat/completions endpoint which is the primary production interface.
"""

import os
# Set test environment variables before any imports
os.environ["TEST_MODE"] = "true"
os.environ["DEFAULT_LLM_PROVIDER"] = "openai"
os.environ["API_BEARER"] = "default-secret-key-for-single-user"
os.environ["SINGLE_USER_API_KEY"] = "default-secret-key-for-single-user"

# Load config to get API keys
from tldw_Server_API.app.core.config import load_and_log_configs
_test_config = load_and_log_configs()
if _test_config and 'openai_api' in _test_config:
    _openai_key = _test_config['openai_api'].get('api_key')
    if _openai_key:
        os.environ["OPENAI_API_KEY"] = _openai_key

# Add dummy API keys for other providers used in tests
os.environ["ANTHROPIC_API_KEY"] = "test-anthropic-key-for-testing"
os.environ["GROQ_API_KEY"] = "test-groq-key-for-testing"
os.environ["MISTRAL_API_KEY"] = "test-mistral-key-for-testing"

import tempfile
from pathlib import Path
from typing import Dict, Any, List, Generator, Optional
from unittest.mock import MagicMock, AsyncMock, Mock
from datetime import datetime
import uuid

import pytest
from fastapi.testclient import TestClient
from httpx import AsyncClient

# Import actual components for integration tests
from tldw_Server_API.app.core.DB_Management.ChaChaNotes_DB import CharactersRAGDB
from tldw_Server_API.app.api.v1.schemas.chat_request_schemas import (
    ChatCompletionRequest,
    ChatCompletionUserMessageParam,
    ChatCompletionSystemMessageParam,
    ChatCompletionAssistantMessageParam,
)

# =====================================================================
# Test Markers
# =====================================================================

def pytest_configure(config):
    """Register custom markers for test categorization."""
    config.addinivalue_line("markers", "unit: Unit tests with minimal mocking")
    config.addinivalue_line("markers", "integration: Integration tests with real components")
    config.addinivalue_line("markers", "property: Property-based tests")
    config.addinivalue_line("markers", "slow: Tests that take > 1 second")
    config.addinivalue_line("markers", "requires_llm: Tests requiring LLM API")
    config.addinivalue_line("markers", "streaming: Tests for streaming responses")

# =====================================================================
# Environment Configuration
# =====================================================================

@pytest.fixture
def test_env_vars():
    """Placeholder for test environment variables - already set at module level."""
    yield

# =====================================================================
# Database Fixtures
# =====================================================================

@pytest.fixture
def temp_db_path() -> Generator[Path, None, None]:
    """Create a temporary database path."""
    with tempfile.TemporaryDirectory() as temp_dir:
        db_path = Path(temp_dir) / "test_chacha.db"
        yield db_path

@pytest.fixture
def chacha_db(temp_db_path) -> CharactersRAGDB:
    """Create a real CharactersRAGDB instance for testing."""
    db = CharactersRAGDB(
        db_path=str(temp_db_path),
        client_id="test_user"
    )
    # Database is initialized in __init__, no need to call initialize_db
    return db

@pytest.fixture
def populated_chacha_db(chacha_db) -> CharactersRAGDB:
    """Create a CharactersRAGDB with test data."""
    # First, add a character card
    # Create Default Character that the system expects
    character_data = {
        'name': 'Default Character',
        'description': 'A helpful assistant',
        'personality': 'Helpful and friendly',
        'system_prompt': 'You are a helpful assistant.',
        'client_id': 'test_user'
    }
    character_id = chacha_db.add_character_card(character_data)
    
    # Add test conversations
    conversation_data = {
        'title': "Test Conversation",
        'character_id': character_id
    }
    conversation_id = chacha_db.add_conversation(conversation_data)
    
    # Add test messages
    chacha_db.add_message({
        'conversation_id': conversation_id,
        'sender': "user",
        'content': "Hello, how are you?"
    })
    
    chacha_db.add_message({
        'conversation_id': conversation_id,
        'sender': "assistant",
        'content': "I'm doing well, thank you! How can I help you today?"
    })
    
    return chacha_db

# =====================================================================
# Mock Fixtures for Unit Tests
# =====================================================================

@pytest.fixture
def mock_chacha_db():
    """Mock CharactersRAGDB for unit tests."""
    mock_db = MagicMock(spec=CharactersRAGDB)
    
    # Setup default return values
    mock_db.add_conversation.return_value = "test-conversation-id"
    mock_db.add_message.return_value = "test-message-id"
    mock_db.get_conversation.return_value = {
        "id": "test-conversation-id",
        "title": "Test Conversation",
        "character_id": 1
    }
    mock_db.get_messages.return_value = []
    mock_db.add_character_card.return_value = 1
    
    return mock_db

@pytest.fixture
def mock_llm_response():
    """Mock LLM response for unit tests."""
    return {
        "id": "chatcmpl-test123",
        "object": "chat.completion",
        "created": 1234567890,
        "model": "gpt-3.5-turbo",
        "choices": [{
            "index": 0,
            "message": {
                "role": "assistant",
                "content": "This is a test response from the LLM."
            },
            "finish_reason": "stop"
        }],
        "usage": {
            "prompt_tokens": 10,
            "completion_tokens": 20,
            "total_tokens": 30
        }
    }

@pytest.fixture
def mock_streaming_response():
    """Mock streaming LLM response for unit tests."""
    async def stream_generator():
        chunks = [
            'data: {"choices":[{"delta":{"content":"Hello"}}]}\n\n',
            'data: {"choices":[{"delta":{"content":" world"}}]}\n\n',
            'data: {"choices":[{"delta":{"content":"!"}}]}\n\n',
            'data: [DONE]\n\n'
        ]
        for chunk in chunks:
            yield chunk
    
    return stream_generator()

# =====================================================================
# Request Fixtures
# =====================================================================

@pytest.fixture
def basic_chat_request() -> Dict[str, Any]:
    """Basic chat completion request."""
    return {
        "model": "gpt-3.5-turbo",
        "messages": [
            {"role": "user", "content": "Hello, how are you?"}
        ]
    }

@pytest.fixture
def multi_turn_chat_request() -> Dict[str, Any]:
    """Multi-turn conversation request."""
    return {
        "model": "gpt-3.5-turbo",
        "messages": [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "What is the capital of France?"},
            {"role": "assistant", "content": "The capital of France is Paris."},
            {"role": "user", "content": "What is its population?"}
        ],
        "temperature": 0.7,
        "max_tokens": 150
    }

@pytest.fixture
def streaming_chat_request() -> Dict[str, Any]:
    """Streaming chat completion request."""
    return {
        "model": "gpt-3.5-turbo",
        "messages": [
            {"role": "user", "content": "Tell me a short story."}
        ],
        "stream": True
    }

@pytest.fixture
def provider_specific_request() -> Dict[str, Any]:
    """Request with specific provider."""
    return {
        "api_provider": "anthropic",
        "model": "claude-3-sonnet",
        "messages": [
            {"role": "user", "content": "Explain quantum computing."}
        ],
        "max_tokens": 200
    }

# =====================================================================
# Message Fixtures
# =====================================================================

@pytest.fixture
def valid_messages() -> List[Dict[str, str]]:
    """Collection of valid message formats."""
    return [
        [{"role": "user", "content": "Simple message"}],
        [{"role": "system", "content": "You are helpful."}, 
         {"role": "user", "content": "Hi"}],
        [{"role": "user", "content": "Question?"},
         {"role": "assistant", "content": "Answer."},
         {"role": "user", "content": "Follow-up?"}]
    ]

@pytest.fixture
def invalid_messages() -> List[Any]:
    """Collection of invalid message formats."""
    return [
        [],  # Empty messages
        [{"role": "invalid", "content": "test"}],  # Invalid role
        [{"content": "missing role"}],  # Missing role
        [{"role": "user"}],  # Missing content
        "not a list",  # Wrong type
        [{"role": "user", "content": ""}]  # Empty content
    ]

# =====================================================================
# API Client Fixtures
# =====================================================================

@pytest.fixture
def test_client(test_env_vars):
    """Create a test client for the FastAPI app."""
    from tldw_Server_API.app.main import app
    return TestClient(app)

@pytest.fixture
async def async_client(test_env_vars):
    """Create an async test client for streaming tests."""
    from tldw_Server_API.app.main import app
    # Use httpx.AsyncClient with transport instead of app parameter
    from httpx import ASGITransport
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        yield client

@pytest.fixture
def auth_headers():
    """Authentication headers for API requests."""
    return {
        "Token": "Bearer default-secret-key-for-single-user",
        "X-API-KEY": "default-secret-key-for-single-user",
        "Content-Type": "application/json"
    }

# =====================================================================
# Provider Configuration Fixtures
# =====================================================================

@pytest.fixture
def provider_configs():
    """Configuration for different LLM providers."""
    return {
        "openai": {
            "api_key": "test-openai-key",
            "base_url": "https://api.openai.com/v1",
            "models": ["gpt-3.5-turbo", "gpt-4"]
        },
        "anthropic": {
            "api_key": "test-anthropic-key",
            "base_url": "https://api.anthropic.com",
            "models": ["claude-3-sonnet", "claude-3-opus"]
        },
        "local": {
            "base_url": "http://localhost:8080",
            "models": ["llama-2-7b"]
        }
    }

# =====================================================================
# Error Response Fixtures
# =====================================================================

@pytest.fixture
def rate_limit_error():
    """Rate limit error response."""
    return {
        "error": {
            "message": "Rate limit exceeded",
            "type": "rate_limit_error",
            "code": 429
        }
    }

@pytest.fixture
def auth_error():
    """Authentication error response."""
    return {
        "error": {
            "message": "Invalid API key",
            "type": "authentication_error",
            "code": 401
        }
    }

@pytest.fixture
def validation_error():
    """Validation error response."""
    return {
        "error": {
            "message": "Invalid request format",
            "type": "validation_error",
            "code": 400
        }
    }

# =====================================================================
# Test Data Generators
# =====================================================================

@pytest.fixture
def message_generator():
    """Factory for generating test messages."""
    def _generate(count: int = 5) -> List[Dict[str, str]]:
        messages = []
        roles = ["user", "assistant"]
        for i in range(count):
            role = roles[i % 2]
            messages.append({
                "role": role,
                "content": f"Test message {i} from {role}"
            })
        return messages
    return _generate

@pytest.fixture
def conversation_generator():
    """Factory for generating test conversations."""
    def _generate(num_turns: int = 3) -> Dict[str, Any]:
        messages = []
        for i in range(num_turns):
            messages.append({
                "role": "user",
                "content": f"User message {i}"
            })
            if i < num_turns - 1:  # Don't add assistant message for last turn
                messages.append({
                    "role": "assistant", 
                    "content": f"Assistant response {i}"
                })
        
        return {
            "model": "gpt-3.5-turbo",
            "messages": messages,
            "temperature": 0.7
        }
    return _generate

# =====================================================================
# Cleanup Fixtures
# =====================================================================

@pytest.fixture(autouse=True)
def cleanup_after_test():
    """Cleanup after each test."""
    yield
    # Any cleanup code here
    import gc
    gc.collect()