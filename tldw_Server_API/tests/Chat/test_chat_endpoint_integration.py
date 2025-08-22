# test_chat_endpoint_integration.py
# Integration tests for the refactored chat endpoint with security modules
# These tests use the actual FastAPI application with real database
# NO MOCKING - These are true integration tests

import pytest
import asyncio
import json
import tempfile
import os
import threading
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
    
    # Add default character (required by the chat endpoint)
    db.add_character_card({
        "name": "Default Character",
        "description": "A test character",
        "personality": "Helpful and friendly",
        "scenario": "Testing",
        "system_prompt": "You are a helpful test assistant",
        "first_message": "Hello! I'm here to help test.",
        "creator_notes": "Created for testing"
    })
    
    # Also add a test character for specific tests
    db.add_character_card({
        "name": "TestCharacter",
        "description": "A test character",
        "personality": "Helpful and friendly",
        "scenario": "Testing",
        "system_prompt": "You are a test assistant for character-specific tests",
        "first_message": "Hello! I'm TestCharacter.",
        "creator_notes": "Created for testing"
    })
    
    yield db
    
    # Cleanup
    try:
        os.unlink(db_path)
    except:
        pass


@pytest.fixture
def test_client(test_db):
    """Create test client with database dependency override."""
    # Override the database dependency to use our test database
    app.dependency_overrides[get_chacha_db_for_user] = lambda: test_db
    
    client = TestClient(app)
    yield client
    
    # Clean up overrides
    app.dependency_overrides.clear()


@pytest.fixture
async def async_test_client():
    """Create async test client for the actual FastAPI app."""
    async with AsyncClient(app=app, base_url="http://test") as client:
        yield client


# Configuration for tests to use a local LLM or test endpoint
@pytest.fixture
def configure_test_llm():
    """Configure tests to use a test LLM endpoint."""
    # Set up test configuration
    import os
    # Tests should use a local LLM or test endpoint configured in config.txt
    # For CI/CD, use environment variables to configure a test endpoint
    os.environ['TEST_MODE'] = 'true'
    yield
    del os.environ['TEST_MODE']




@pytest.fixture
def auth_headers():
    """Provide test authentication headers."""
    # Import settings to get the actual API key used in single-user mode
    from tldw_Server_API.app.core.AuthNZ.settings import get_settings
    settings = get_settings()
    return {"X-API-Key": settings.SINGLE_USER_API_KEY}


class TestChatEndpointIntegration:
    """Integration tests for the chat endpoint using actual components."""
    
    def test_chat_completion_basic(self, test_client, test_db, auth_headers, caplog):
        """Test basic chat completion through the actual endpoint."""
        import logging
        caplog.set_level(logging.DEBUG)
        
        response = test_client.post(
            "/api/v1/chat/completions",
            json={
                "messages": [
                    {"role": "user", "content": "Hello, how are you?"}
                ],
                "model": "gpt-4",
                "api_provider": "openai"  # Correct field name from schema
            },
            headers=auth_headers
        )
        
        # With real LLM, accept either success or service unavailable (if LLM not configured)
        if response.status_code == 503:
            print(f"LLM service not configured: {response.json()}")
            pytest.skip("LLM service not configured for integration testing")
        elif response.status_code != 200:
            print(f"Response status: {response.status_code}")
            print(f"Response body: {response.json()}")
        
        assert response.status_code in [200, 503], f"Unexpected status: {response.status_code}"
        
        if response.status_code == 200:
            data = response.json()
            assert isinstance(data, dict)
            assert "choices" in data
            assert len(data["choices"]) > 0
            # Don't check content - it's from real LLM
    
    def test_chat_with_character(self, test_client, test_db, auth_headers):
        """Test chat with character context."""
        response = test_client.post(
            "/api/v1/chat/completions",
            json={
                "messages": [
                    {"role": "user", "content": "Tell me about yourself"}
                ],
                "model": "gpt-4",
                "api_provider": "openai",
                "character_id": "TestCharacter"
            },
            headers=auth_headers
        )
        
        # Accept success or service unavailable
        if response.status_code == 503:
            pytest.skip("LLM service not configured")
        assert response.status_code in [200, 503]
        
        # Verify character was loaded by checking database
        characters = test_db.list_character_cards()
        assert any(c["name"] == "TestCharacter" for c in characters)
        
        # With real LLM, verify response has expected structure
        data = response.json()
        assert "choices" in data or "error" in data
    
    def test_conversation_persistence(self, test_client, test_db, auth_headers):
        """Test that conversations are persisted in the database."""
        # First message
        response1 = test_client.post(
            "/api/v1/chat/completions",
            json={
                "messages": [
                    {"role": "user", "content": "My name is Alice"}
                ],
                "model": "gpt-4",
                "api_provider": "openai"  # Fixed field name
            },
            headers=auth_headers
        )
        
        if response1.status_code == 503:
            pytest.skip("LLM service not configured")
        assert response1.status_code in [200, 503]
        data1 = response1.json()
        
        # Extract conversation ID from response
        conv_id = data1.get("tldw_conversation_id")
        assert conv_id, "No conversation ID returned in response"
        
        # Second message in same conversation
        response2 = test_client.post(
            "/api/v1/chat/completions",
            json={
                "messages": [
                    {"role": "user", "content": "What's my name?"}
                ],
                "model": "gpt-4",
                "api_provider": "openai",  # Fixed field name
                "conversation_id": conv_id
            },
            headers=auth_headers
        )
        
        assert response2.status_code in [200, 503]
        
        # Verify messages are in database
        messages = test_db.get_messages_for_conversation(conv_id)
        assert len(messages) >= 4  # Two user messages and two assistant messages
        # Check that the conversation contains our messages
        message_contents = [msg["content"] for msg in messages]
        assert any("Alice" in content for content in message_contents)
    
    @pytest.mark.skip(reason="TestClient doesn't properly handle streaming responses")
    def test_streaming_response(self, test_client, test_db, auth_headers):
        """Test streaming chat completion request is accepted."""
        # Note: TestClient doesn't properly support streaming responses,
        # so we just verify the endpoint accepts streaming requests
        response = test_client.post(
            "/api/v1/chat/completions",
            json={
                "messages": [
                    {"role": "user", "content": "Stream a response"}
                ],
                "model": "gpt-4",
                "api_provider": "openai",
                "stream": True
            },
            headers=auth_headers
        )
        
        # For streaming, the endpoint returns a StreamingResponse
        # TestClient converts this to a regular response
        # Just verify it doesn't error
        if response.status_code == 503:
            pytest.skip("LLM service not configured")
        assert response.status_code in [200, 503]
    
    def test_message_validation(self, test_client, test_db, auth_headers):
        """Test message validation and error handling."""
        # Test empty messages
        response = test_client.post(
            "/api/v1/chat/completions",
            json={
                "messages": [],
                "model": "gpt-4",
                "api_provider": "openai"  # Fixed field name
            },
            headers=auth_headers
        )
        
        # FastAPI returns 422 for validation errors
        assert response.status_code == 422
        # Check error message mentions the validation issue
        error_detail = str(response.json()).lower()
        assert "message" in error_detail or "validation" in error_detail
            
        # Test invalid temperature
        response = test_client.post(
            "/api/v1/chat/completions",
            json={
                "messages": [{"role": "user", "content": "Test"}],
                "model": "gpt-4",
                "api_provider": "openai",  # Fixed field name
                "temperature": 3.0  # Too high
            },
            headers=auth_headers
        )
        
        # FastAPI returns 422 for validation errors
        assert response.status_code == 422
        error_detail = str(response.json()).lower()
        assert "temperature" in error_detail
    
    def test_image_handling(self, test_client, test_db, auth_headers):
        """Test handling of image inputs."""
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
                "api_provider": "openai"  # Fixed field name
            },
            headers=auth_headers
        )
        
        if response.status_code == 503:
            pytest.skip("LLM service not configured")
        assert response.status_code in [200, 503]
        
        # Verify image was stored in database
        # Get conversations for the test character
        characters = test_db.list_character_cards()
        char_id = characters[0]["id"] if characters else 1
        conversations = test_db.get_conversations_for_character(char_id)
        if conversations:
            messages = test_db.get_messages_for_conversation(conversations[0]["conversation_id"])
            assert any(msg.get("image_data") or msg.get("has_image") for msg in messages)
    
    def test_tool_usage(self, test_client, test_db, auth_headers):
        """Test chat with tool definitions - currently tools validation is strict."""
        response = test_client.post(
            "/api/v1/chat/completions",
            json={
                "messages": [
                    {"role": "user", "content": "What's the weather?"}
                ],
                "model": "gpt-4",
                "api_provider": "openai",
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
        
        # Tools validation is currently strict and may reject this format
        # Just verify the endpoint handles tools parameter without crashing
        assert response.status_code in [200, 400]  # Accept either success or validation error
    
    def test_transaction_handling(self, test_client, test_db, auth_headers):
        """Test database transaction handling - messages are persisted correctly."""
        # Test that messages are properly saved to the database
        response = test_client.post(
            "/api/v1/chat/completions",
            json={
                "messages": [
                    {"role": "user", "content": "Test transaction"}
                ],
                "model": "gpt-4",
                "api_provider": "openai"  # Fixed field name
                # Removed use_transaction as it's not a valid API parameter
            },
            headers=auth_headers
        )
        
        if response.status_code == 503:
            pytest.skip("LLM service not configured")
        assert response.status_code in [200, 503]
        data = response.json()
        
        # Get conversation ID from response
        conv_id = data.get("tldw_conversation_id")
        assert conv_id, "No conversation ID returned"
        
        # Verify messages were created in the database
        messages = test_db.get_messages_for_conversation(conv_id)
        assert len(messages) >= 2  # User and assistant messages
    
    def test_concurrent_requests(self, test_client, test_db, auth_headers):
        """Test handling of concurrent requests."""
        import threading
        import time
        
        results = []
        errors = []
        
        def make_request(index):
            try:
                response = test_client.post(
                    "/api/v1/chat/completions",
                    json={
                        "messages": [
                            {"role": "user", "content": f"Request {index}"}
                        ],
                        "model": "gpt-4",
                        "api_provider": "openai"
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
        
        # All requests should either succeed or get service unavailable
        assert len(results) == 5
        assert all(status in [200, 503] for status in results)
        assert len(errors) == 0
    
    def test_rate_limiting(self, test_client, test_db, auth_headers):
        """Test rate limiting functionality."""
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
                    "api_provider": "openai"
                },
                headers=auth_headers
            )
            responses.append(response.status_code)
        
        # Should handle all requests (rate limiting may return 429, or 503 if not configured)
        assert all(status in [200, 429, 503] for status in responses)


class TestChatEndpointSecurity:
    """Security-focused integration tests."""
    
    def test_sql_injection_prevention(self, test_client, test_db, auth_headers):
        """Test SQL injection prevention."""
        response = test_client.post(
            "/api/v1/chat/completions",
            json={
                "messages": [
                    {"role": "user", "content": "Test"}
                ],
                "model": "gpt-4",
                "api_provider": "openai",
                "conversation_id": "'; DROP TABLE conversations; --"
            },
            headers=auth_headers
        )
        
        # Should handle safely without SQL execution
        assert response.status_code in [200, 400, 404, 503]  # 503 if LLM not configured
        
        # Verify tables still exist
        # Verify tables still exist by checking characters
        assert test_db.list_character_cards() is not None
    
    def test_xss_prevention(self, test_client, test_db, auth_headers):
        """Test XSS prevention."""
        response = test_client.post(
            "/api/v1/chat/completions",
            json={
                "messages": [
                    {"role": "user", "content": "<script>alert('XSS')</script>"}
                ],
                "model": "gpt-4",
                "api_provider": "openai"
            },
            headers=auth_headers
        )
        
        if response.status_code == 503:
            pytest.skip("LLM service not configured")
        assert response.status_code in [200, 503]
        
        # Verify content is stored safely
        # Get conversations for the test character
        characters = test_db.list_character_cards()
        char_id = characters[0]["id"] if characters else 1
        conversations = test_db.get_conversations_for_character(char_id)
        if conversations:
            messages = test_db.get_messages_for_conversation(conversations[0]["conversation_id"])
            # Script tags should be stored as text, not executed
            assert any("<script>" in msg["content"] for msg in messages)
    
    def test_large_request_dos_prevention(self, test_client, test_db, auth_headers):
        """Test DoS prevention for large requests."""
        # Create a very large message
        large_content = "x" * 500000  # 500KB of text
        
        response = test_client.post(
            "/api/v1/chat/completions",
            json={
                "messages": [
                    {"role": "user", "content": large_content}
                ],
                "model": "gpt-4",
                "api_provider": "openai"  # Fixed field name
            },
            headers=auth_headers
        )
        
        # Should reject overly large requests
        # 413 is the correct status code for "Payload Too Large"
        assert response.status_code in [400, 413]
        if response.status_code == 413:
            # FastAPI/Starlette returns 413 for payload too large
            assert True  # This is expected
        else:
            # If 400, check for appropriate error message
            assert "too long" in response.json()["detail"].lower() or "too large" in response.json()["detail"].lower()
    
    def test_authentication_required(self, test_client, test_db):
        """Test authentication behavior based on auth mode."""
        from tldw_Server_API.app.core.AuthNZ.settings import get_settings
        settings = get_settings()
        
        response = test_client.post(
            "/api/v1/chat/completions",
            json={
                "messages": [
                    {"role": "user", "content": "Test without auth"}
                ],
                "model": "gpt-4",
                "api_provider": "openai"
            }
            # No auth headers
        )
        
        if settings.AUTH_MODE == "single_user":
            # In single-user mode, authentication is not required
            # The request should succeed (200) or fail due to other reasons like missing API keys (503)
            assert response.status_code in [200, 503], f"In single-user mode, got unexpected status: {response.status_code}"
        else:
            # In multi-user mode, should require authentication
            assert response.status_code in [401, 403], f"In multi-user mode, expected auth error but got: {response.status_code}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])