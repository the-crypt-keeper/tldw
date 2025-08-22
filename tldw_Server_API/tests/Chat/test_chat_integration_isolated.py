"""
Integration tests for chat functionality using isolated fixtures.
These tests require the mock OpenAI server to be running.
"""
import pytest
from fastapi import status

# Import isolated fixtures
from .conftest_isolated import (
    isolated_db,
    integration_test_client,
    mock_server_url,
    sample_chat_request
)


@pytest.mark.integration
class TestChatIntegrationIsolated:
    """Integration tests that use the mock OpenAI server."""
    
    def test_chat_completion_with_mock_server(self, integration_test_client):
        """Test actual chat completion using mock server."""
        request_data = {
            "model": "gpt-4",
            "api_provider": "openai",
            "messages": [{"role": "user", "content": "Hello, how are you?"}]
        }
        
        response = integration_test_client.post_with_auth(
            "/api/v1/chat/completions",
            json_data=request_data
        )
        
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert "choices" in data
        assert len(data["choices"]) > 0
        # Mock server returns specific response
        assert "mock" in data["choices"][0]["message"]["content"].lower()
    
    def test_streaming_with_mock_server(self, integration_test_client):
        """Test streaming response with mock server."""
        request_data = {
            "model": "gpt-4",
            "api_provider": "openai",
            "messages": [{"role": "user", "content": "Tell me a story"}],
            "stream": True
        }
        
        response = integration_test_client.post_with_auth(
            "/api/v1/chat/completions",
            json_data=request_data
        )
        
        # TestClient doesn't handle streaming properly, but we can verify the endpoint accepts it
        assert response.status_code == status.HTTP_200_OK
        # Response should have SSE content type
        assert 'text/event-stream' in response.headers.get('content-type', '').lower()
    
    def test_character_with_mock_server(self, integration_test_client, isolated_db):
        """Test character-based chat with mock server."""
        # Add a character
        char_id = isolated_db.add_character_card({
            "name": "Pirate",
            "description": "A pirate character",
            "personality": "Adventurous",
            "scenario": "High seas",
            "system_prompt": "You are a pirate. Speak like a pirate!",
            "first_message": "Ahoy matey!",
            "creator_notes": "Test pirate"
        })
        
        request_data = {
            "model": "gpt-4",
            "api_provider": "openai",
            "messages": [{"role": "user", "content": "Hello"}],
            "character_id": char_id
        }
        
        response = integration_test_client.post_with_auth(
            "/api/v1/chat/completions",
            json_data=request_data
        )
        
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert "choices" in data
    
    def test_conversation_with_mock_server(self, integration_test_client, isolated_db):
        """Test conversation persistence with mock server."""
        # Create conversation
        conv_id = isolated_db.create_conversation(
            user_id="test_user",
            character_id=1,
            title="Test Conv"
        )
        
        # First message
        request_data = {
            "model": "gpt-4",
            "api_provider": "openai",
            "messages": [{"role": "user", "content": "Hello"}],
            "conversation_id": conv_id
        }
        
        response = integration_test_client.post_with_auth(
            "/api/v1/chat/completions",
            json_data=request_data
        )
        
        assert response.status_code == status.HTTP_200_OK
        
        # Second message in same conversation
        request_data["messages"] = [{"role": "user", "content": "How are you?"}]
        
        response = integration_test_client.post_with_auth(
            "/api/v1/chat/completions",
            json_data=request_data
        )
        
        assert response.status_code == status.HTTP_200_OK
        
        # Check messages were saved
        messages = isolated_db.get_messages(conv_id)
        assert len(messages) >= 2
    
    def test_system_message_with_mock_server(self, integration_test_client):
        """Test system message handling with mock server."""
        request_data = {
            "model": "gpt-4",
            "api_provider": "openai",
            "messages": [
                {"role": "system", "content": "You are a helpful assistant"},
                {"role": "user", "content": "What is 2+2?"}
            ]
        }
        
        response = integration_test_client.post_with_auth(
            "/api/v1/chat/completions",
            json_data=request_data
        )
        
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert "choices" in data
    
    def test_multimodal_with_mock_server(self, integration_test_client):
        """Test multimodal content with mock server."""
        request_data = {
            "model": "gpt-4-vision",
            "api_provider": "openai",
            "messages": [{
                "role": "user",
                "content": [
                    {"type": "text", "text": "What's in this image?"},
                    {"type": "image_url", "image_url": {"url": "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNkYPhfDwAChwGA60e6kgAAAABJRU5ErkJggg=="}}
                ]
            }]
        }
        
        response = integration_test_client.post_with_auth(
            "/api/v1/chat/completions",
            json_data=request_data
        )
        
        # Mock server should handle multimodal requests
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert "choices" in data
    
    def test_temperature_with_mock_server(self, integration_test_client):
        """Test temperature parameter with mock server."""
        request_data = {
            "model": "gpt-4",
            "api_provider": "openai",
            "messages": [{"role": "user", "content": "Be creative"}],
            "temperature": 0.9
        }
        
        response = integration_test_client.post_with_auth(
            "/api/v1/chat/completions",
            json_data=request_data
        )
        
        assert response.status_code == status.HTTP_200_OK
    
    def test_max_tokens_with_mock_server(self, integration_test_client):
        """Test max_tokens parameter with mock server."""
        request_data = {
            "model": "gpt-4",
            "api_provider": "openai",
            "messages": [{"role": "user", "content": "Tell me a long story"}],
            "max_tokens": 50
        }
        
        response = integration_test_client.post_with_auth(
            "/api/v1/chat/completions",
            json_data=request_data
        )
        
        assert response.status_code == status.HTTP_200_OK
    
    def test_tools_with_mock_server(self, integration_test_client):
        """Test function calling/tools with mock server."""
        request_data = {
            "model": "gpt-4",
            "api_provider": "openai",
            "messages": [{"role": "user", "content": "What's the weather?"}],
            "tools": [{
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "description": "Get the weather",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "location": {"type": "string"}
                        }
                    }
                }
            }]
        }
        
        response = integration_test_client.post_with_auth(
            "/api/v1/chat/completions",
            json_data=request_data
        )
        
        assert response.status_code == status.HTTP_200_OK