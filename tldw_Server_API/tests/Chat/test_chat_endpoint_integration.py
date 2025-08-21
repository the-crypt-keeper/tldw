# test_chat_endpoint_integration.py
# Integration tests for the refactored chat endpoint with security modules
# These tests use the actual FastAPI application with minimal mocking

import pytest
import asyncio
import json
import tempfile
import os
from unittest.mock import patch, AsyncMock
from fastapi.testclient import TestClient
from httpx import AsyncClient

from tldw_Server_API.app.main import app
from tldw_Server_API.app.core.DB_Management.ChaChaNotes_DB import CharactersRAGDB
from tldw_Server_API.app.api.v1.API_Deps.ChaCha_Notes_DB_Deps import get_chacha_db_for_user


@pytest.fixture
def test_db():
    """Create a temporary test database."""
    with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as tmp:
        db_path = tmp.name
    
    # Initialize database
    db = CharactersRAGDB(db_path, "test_client")
    
    # Add test character
    db.add_character_card({
        "name": "TestCharacter",
        "description": "A test character",
        "personality": "Helpful and friendly",
        "scenario": "Testing",
        "system_prompt": "You are a helpful test assistant",
        "first_message": "Hello! I'm here to help test.",
        "creator_notes": "Created for testing"
    })
    
    yield db
    
    # Cleanup
    try:
        os.unlink(db_path)
    except:
        pass


@pytest.fixture
def test_client():
    """Create test client for the actual FastAPI app."""
    return TestClient(app)


@pytest.fixture
async def async_test_client():
    """Create async test client for the actual FastAPI app."""
    async with AsyncClient(app=app, base_url="http://test") as client:
        yield client


@pytest.fixture
def mock_llm():
    """Mock only the LLM calls to avoid actual API calls."""
    with patch('tldw_Server_API.app.core.Chat.Chat_Functions.chat_api_call') as mock:
        mock.return_value = "This is a test response from the mocked LLM."
        yield mock


@pytest.fixture
def mock_llm_streaming():
    """Mock streaming LLM calls."""
    async def stream_generator():
        chunks = ["This ", "is ", "a ", "streaming ", "test ", "response."]
        for chunk in chunks:
            yield chunk
            await asyncio.sleep(0.01)
    
    with patch('tldw_Server_API.app.core.Chat.Chat_Functions.chat_api_call') as mock:
        mock.return_value = stream_generator()
        yield mock


@pytest.fixture
def auth_headers():
    """Provide test authentication headers."""
    return {"X-API-Key": "test-api-key"}


class TestChatEndpointIntegration:
    """Integration tests for the chat endpoint using actual components."""
    
    def test_chat_completion_basic(self, test_client, test_db, mock_llm, auth_headers):
        """Test basic chat completion through the actual endpoint."""
        with patch('tldw_Server_API.app.api.v1.API_Deps.ChaCha_Notes_DB_Deps.get_chacha_db_for_user', return_value=test_db):
            with patch('tldw_Server_API.app.api.v1.endpoints.chat.verify_api_key', return_value={"client_id": "test_client"}):
                response = test_client.post(
                    "/api/v1/chat/completions",
                    json={
                        "messages": [
                            {"role": "user", "content": "Hello, how are you?"}
                        ],
                        "model": "gpt-4",
                        "provider": "openai"
                    },
                    headers=auth_headers
                )
        
        assert response.status_code == 200
        data = response.json()
        assert "choices" in data
        assert len(data["choices"]) > 0
        assert data["choices"][0]["message"]["content"] == "This is a test response from the mocked LLM."
    
    def test_chat_with_character(self, test_client, test_db, mock_llm, auth_headers):
        """Test chat with character context."""
        with patch('tldw_Server_API.app.api.v1.API_Deps.ChaCha_Notes_DB_Deps.get_chacha_db_for_user', return_value=test_db):
            with patch('tldw_Server_API.app.api.v1.endpoints.chat.verify_api_key', return_value={"client_id": "test_client"}):
                response = test_client.post(
                    "/api/v1/chat/completions",
                    json={
                        "messages": [
                            {"role": "user", "content": "Tell me about yourself"}
                        ],
                        "model": "gpt-4",
                        "provider": "openai",
                        "character_id": "TestCharacter"
                    },
                    headers=auth_headers
                )
        
        assert response.status_code == 200
        
        # Verify character was loaded by checking database
        characters = test_db.get_character_cards()
        assert any(c["name"] == "TestCharacter" for c in characters)
        
        # Verify the LLM was called with character context
        mock_llm.assert_called()
        call_args = mock_llm.call_args
        # System prompt should include character info
        messages = call_args[0][0] if call_args[0] else call_args[1].get("messages", [])
        assert any("test assistant" in str(msg).lower() for msg in messages)
    
    def test_conversation_persistence(self, test_client, test_db, mock_llm, auth_headers):
        """Test that conversations are persisted in the database."""
        with patch('tldw_Server_API.app.api.v1.API_Deps.ChaCha_Notes_DB_Deps.get_chacha_db_for_user', return_value=test_db):
            with patch('tldw_Server_API.app.api.v1.endpoints.chat.verify_api_key', return_value={"client_id": "test_client"}):
                # First message
                response1 = test_client.post(
                    "/api/v1/chat/completions",
                    json={
                        "messages": [
                            {"role": "user", "content": "My name is Alice"}
                        ],
                        "model": "gpt-4",
                        "provider": "openai"
                    },
                    headers=auth_headers
                )
                
                assert response1.status_code == 200
                data1 = response1.json()
                
                # Extract conversation ID from response metadata (if available)
                # or query the database
                conversations = test_db.get_all_conversations(client_id="test_client")
                assert len(conversations) > 0
                conv_id = conversations[0]["conversation_id"]
                
                # Second message in same conversation
                response2 = test_client.post(
                    "/api/v1/chat/completions",
                    json={
                        "messages": [
                            {"role": "user", "content": "What's my name?"}
                        ],
                        "model": "gpt-4",
                        "provider": "openai",
                        "conversation_id": conv_id
                    },
                    headers=auth_headers
                )
                
                assert response2.status_code == 200
                
                # Verify messages are in database
                messages = test_db.get_messages_for_conversation(conv_id)
                assert len(messages) >= 2  # At least user and assistant messages
                assert any("Alice" in msg["content"] for msg in messages)
    
    def test_streaming_response(self, test_client, test_db, mock_llm_streaming, auth_headers):
        """Test streaming chat completion."""
        with patch('tldw_Server_API.app.api.v1.API_Deps.ChaCha_Notes_DB_Deps.get_chacha_db_for_user', return_value=test_db):
            with patch('tldw_Server_API.app.api.v1.endpoints.chat.verify_api_key', return_value={"client_id": "test_client"}):
                with test_client as client:
                    response = client.post(
                        "/api/v1/chat/completions",
                        json={
                            "messages": [
                                {"role": "user", "content": "Stream a response"}
                            ],
                            "model": "gpt-4",
                            "provider": "openai",
                            "stream": True
                        },
                        headers=auth_headers,
                        stream=True
                    )
                    
                    assert response.status_code == 200
                    
                    # Collect streamed chunks
                    chunks = []
                    for line in response.iter_lines():
                        if line:
                            chunks.append(line)
                    
                    assert len(chunks) > 0
                    # Should have SSE format
                    assert any(b"data:" in chunk for chunk in chunks)
                    assert any(b"[DONE]" in chunk for chunk in chunks)
    
    def test_message_validation(self, test_client, test_db, auth_headers):
        """Test message validation and error handling."""
        with patch('tldw_Server_API.app.api.v1.API_Deps.ChaCha_Notes_DB_Deps.get_chacha_db_for_user', return_value=test_db):
            with patch('tldw_Server_API.app.api.v1.endpoints.chat.verify_api_key', return_value={"client_id": "test_client"}):
                # Test empty messages
                response = test_client.post(
                    "/api/v1/chat/completions",
                    json={
                        "messages": [],
                        "model": "gpt-4",
                        "provider": "openai"
                    },
                    headers=auth_headers
                )
                
                assert response.status_code == 400
                assert "empty" in response.json()["detail"].lower()
                
                # Test invalid temperature
                response = test_client.post(
                    "/api/v1/chat/completions",
                    json={
                        "messages": [{"role": "user", "content": "Test"}],
                        "model": "gpt-4",
                        "provider": "openai",
                        "temperature": 3.0  # Too high
                    },
                    headers=auth_headers
                )
                
                assert response.status_code == 400
                assert "temperature" in response.json()["detail"].lower()
    
    def test_image_handling(self, test_client, test_db, mock_llm, auth_headers):
        """Test handling of image inputs."""
        with patch('tldw_Server_API.app.api.v1.API_Deps.ChaCha_Notes_DB_Deps.get_chacha_db_for_user', return_value=test_db):
            with patch('tldw_Server_API.app.api.v1.endpoints.chat.verify_api_key', return_value={"client_id": "test_client"}):
                # Small valid PNG image (1x1 pixel red dot)
                image_data = "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8DwHwAFBQIAX8jx0gAAAABJRU5ErkJggg=="
                
                response = test_client.post(
                    "/api/v1/chat/completions",
                    json={
                        "messages": [
                            {
                                "role": "user",
                                "content": [
                                    {"type": "text", "text": "What's in this image?"},
                                    {"type": "image_url", "image_url": {"url": image_data}}
                                ]
                            }
                        ],
                        "model": "gpt-4-vision",
                        "provider": "openai"
                    },
                    headers=auth_headers
                )
                
                assert response.status_code == 200
                
                # Verify image was stored in database
                conversations = test_db.get_all_conversations(client_id="test_client")
                if conversations:
                    messages = test_db.get_messages_for_conversation(conversations[0]["conversation_id"])
                    assert any(msg.get("image_data") or msg.get("has_image") for msg in messages)
    
    def test_tool_usage(self, test_client, test_db, mock_llm, auth_headers):
        """Test chat with tool definitions."""
        with patch('tldw_Server_API.app.api.v1.API_Deps.ChaCha_Notes_DB_Deps.get_chacha_db_for_user', return_value=test_db):
            with patch('tldw_Server_API.app.api.v1.endpoints.chat.verify_api_key', return_value={"client_id": "test_client"}):
                response = test_client.post(
                    "/api/v1/chat/completions",
                    json={
                        "messages": [
                            {"role": "user", "content": "What's the weather?"}
                        ],
                        "model": "gpt-4",
                        "provider": "openai",
                        "tools": [
                            {
                                "type": "function",
                                "function": {
                                    "name": "get_weather",
                                    "description": "Get current weather",
                                    "parameters": {
                                        "type": "object",
                                        "properties": {
                                            "location": {"type": "string"}
                                        }
                                    }
                                }
                            }
                        ]
                    },
                    headers=auth_headers
                )
                
                assert response.status_code == 200
                
                # Verify tools were passed to LLM
                mock_llm.assert_called()
                call_kwargs = mock_llm.call_args[1]
                assert "tools" in call_kwargs or "functions" in call_kwargs
    
    def test_transaction_handling(self, test_client, test_db, mock_llm, auth_headers):
        """Test database transaction handling."""
        with patch('tldw_Server_API.app.api.v1.API_Deps.ChaCha_Notes_DB_Deps.get_chacha_db_for_user', return_value=test_db):
            with patch('tldw_Server_API.app.api.v1.endpoints.chat.verify_api_key', return_value={"client_id": "test_client"}):
                # Create multiple requests with transaction flag
                response = test_client.post(
                    "/api/v1/chat/completions",
                    json={
                        "messages": [
                            {"role": "user", "content": "Test transaction"}
                        ],
                        "model": "gpt-4",
                        "provider": "openai",
                        "use_transaction": True
                    },
                    headers=auth_headers
                )
                
                assert response.status_code == 200
                
                # Verify conversation and messages were created atomically
                conversations = test_db.get_all_conversations(client_id="test_client")
                assert len(conversations) > 0
                
                messages = test_db.get_messages_for_conversation(conversations[0]["conversation_id"])
                assert len(messages) >= 2  # User and assistant messages
    
    def test_concurrent_requests(self, test_client, test_db, mock_llm, auth_headers):
        """Test handling of concurrent requests."""
        import threading
        import time
        
        results = []
        errors = []
        
        def make_request(index):
            try:
                with patch('tldw_Server_API.app.api.v1.API_Deps.ChaCha_Notes_DB_Deps.get_chacha_db_for_user', return_value=test_db):
                    with patch('tldw_Server_API.app.api.v1.endpoints.chat.verify_api_key', return_value={"client_id": f"client_{index}"}):
                        response = test_client.post(
                            "/api/v1/chat/completions",
                            json={
                                "messages": [
                                    {"role": "user", "content": f"Request {index}"}
                                ],
                                "model": "gpt-4",
                                "provider": "openai"
                            },
                            headers=auth_headers
                        )
                        results.append(response.status_code)
            except Exception as e:
                errors.append(str(e))
        
        # Create threads for concurrent requests
        threads = []
        for i in range(5):
            thread = threading.Thread(target=make_request, args=(i,))
            threads.append(thread)
            thread.start()
        
        # Wait for all threads to complete
        for thread in threads:
            thread.join(timeout=10)
        
        # All requests should succeed
        assert len(results) == 5
        assert all(status == 200 for status in results)
        assert len(errors) == 0
    
    def test_rate_limiting(self, test_client, test_db, mock_llm, auth_headers):
        """Test rate limiting functionality."""
        with patch('tldw_Server_API.app.api.v1.API_Deps.ChaCha_Notes_DB_Deps.get_chacha_db_for_user', return_value=test_db):
            with patch('tldw_Server_API.app.api.v1.endpoints.chat.verify_api_key', return_value={"client_id": "test_client"}):
                # Make multiple rapid requests
                responses = []
                for i in range(10):
                    response = test_client.post(
                        "/api/v1/chat/completions",
                        json={
                            "messages": [
                                {"role": "user", "content": f"Request {i}"}
                            ],
                            "model": "gpt-4",
                            "provider": "openai"
                        },
                        headers=auth_headers
                    )
                    responses.append(response.status_code)
                
                # Should handle all requests (rate limiting may return 429 for some)
                assert all(status in [200, 429] for status in responses)


class TestChatEndpointSecurity:
    """Security-focused integration tests."""
    
    def test_sql_injection_prevention(self, test_client, test_db, mock_llm, auth_headers):
        """Test SQL injection prevention."""
        with patch('tldw_Server_API.app.api.v1.API_Deps.ChaCha_Notes_DB_Deps.get_chacha_db_for_user', return_value=test_db):
            with patch('tldw_Server_API.app.api.v1.endpoints.chat.verify_api_key', return_value={"client_id": "test_client"}):
                response = test_client.post(
                    "/api/v1/chat/completions",
                    json={
                        "messages": [
                            {"role": "user", "content": "Test"}
                        ],
                        "model": "gpt-4",
                        "provider": "openai",
                        "conversation_id": "'; DROP TABLE conversations; --"
                    },
                    headers=auth_headers
                )
                
                # Should handle safely without SQL execution
                assert response.status_code in [200, 400, 404]
                
                # Verify tables still exist
                assert test_db.get_all_conversations(client_id="test_client") is not None
    
    def test_xss_prevention(self, test_client, test_db, mock_llm, auth_headers):
        """Test XSS prevention."""
        with patch('tldw_Server_API.app.api.v1.API_Deps.ChaCha_Notes_DB_Deps.get_chacha_db_for_user', return_value=test_db):
            with patch('tldw_Server_API.app.api.v1.endpoints.chat.verify_api_key', return_value={"client_id": "test_client"}):
                response = test_client.post(
                    "/api/v1/chat/completions",
                    json={
                        "messages": [
                            {"role": "user", "content": "<script>alert('XSS')</script>"}
                        ],
                        "model": "gpt-4",
                        "provider": "openai"
                    },
                    headers=auth_headers
                )
                
                assert response.status_code == 200
                
                # Verify content is stored safely
                conversations = test_db.get_all_conversations(client_id="test_client")
                if conversations:
                    messages = test_db.get_messages_for_conversation(conversations[0]["conversation_id"])
                    # Script tags should be stored as text, not executed
                    assert any("<script>" in msg["content"] for msg in messages)
    
    def test_large_request_dos_prevention(self, test_client, test_db, auth_headers):
        """Test DoS prevention for large requests."""
        with patch('tldw_Server_API.app.api.v1.API_Deps.ChaCha_Notes_DB_Deps.get_chacha_db_for_user', return_value=test_db):
            with patch('tldw_Server_API.app.api.v1.endpoints.chat.verify_api_key', return_value={"client_id": "test_client"}):
                # Create a very large message
                large_content = "x" * 500000  # 500KB of text
                
                response = test_client.post(
                    "/api/v1/chat/completions",
                    json={
                        "messages": [
                            {"role": "user", "content": large_content}
                        ],
                        "model": "gpt-4",
                        "provider": "openai"
                    },
                    headers=auth_headers
                )
                
                # Should reject overly large requests
                assert response.status_code == 400
                assert "too long" in response.json()["detail"].lower() or "too large" in response.json()["detail"].lower()
    
    def test_authentication_required(self, test_client, test_db):
        """Test that authentication is required."""
        response = test_client.post(
            "/api/v1/chat/completions",
            json={
                "messages": [
                    {"role": "user", "content": "Test without auth"}
                ],
                "model": "gpt-4",
                "provider": "openai"
            }
            # No auth headers
        )
        
        # Should require authentication
        assert response.status_code in [401, 403]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])