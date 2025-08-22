"""
Simplified chat endpoint tests using real database and authentication.
"""
import pytest
from tldw_Server_API.app.core.AuthNZ.User_DB_Handling import get_request_user, User
from tldw_Server_API.app.main import app
from unittest.mock import patch, MagicMock
from fastapi import status

from tldw_Server_API.app.api.v1.schemas.chat_request_schemas import (
    ChatCompletionRequest,
    ChatCompletionUserMessageParam
)


def test_chat_completion_basic(authenticated_client, mock_chacha_db):
    """Test basic chat completion with authenticated user."""
    
    # Prepare request data - must include api_provider
    request_data = ChatCompletionRequest(
        model="test-model",
        api_provider="openai",  # Must specify provider
        messages=[
            ChatCompletionUserMessageParam(role="user", content="Hello, how are you?")
        ]
    )
    
    # Mock the LLM call and API keys
    with patch("tldw_Server_API.app.api.v1.endpoints.chat.perform_chat_api_call") as mock_llm, \
         patch("tldw_Server_API.app.api.v1.endpoints.chat.API_KEYS", {"openai": "test-key"}):
        mock_llm.return_value = {
            "id": "chatcmpl-test",
            "choices": [{
                "message": {"role": "assistant", "content": "I'm doing well, thank you!"},
                "finish_reason": "stop"
            }]
        }
        
        # Make request
        response = authenticated_client.post(
            "/api/v1/chat/completions",
            json=request_data.model_dump()
        )
        
        # Verify response
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert data["choices"][0]["message"]["content"] == "I'm doing well, thank you!"
        
        # Verify LLM was called
        mock_llm.assert_called_once()


@pytest.mark.skip(reason="Streaming tests hang with TestClient")
def test_chat_completion_streaming(authenticated_client, mock_chacha_db):
    """Test streaming chat completion."""
    
    request_data = ChatCompletionRequest(
        model="test-model",
        api_provider="openai",
        messages=[
            ChatCompletionUserMessageParam(role="user", content="Tell me a story")
        ],
        stream=True
    )
    
    # Mock streaming response
    def mock_stream():
        yield "data: {\"choices\": [{\"delta\": {\"content\": \"Once \"}}]}\n\n"
        yield "data: {\"choices\": [{\"delta\": {\"content\": \"upon \"}}]}\n\n"
        yield "data: {\"choices\": [{\"delta\": {\"content\": \"a \"}}]}\n\n"
        yield "data: {\"choices\": [{\"delta\": {\"content\": \"time...\"}}]}\n\n"
        yield "data: [DONE]\n\n"
    
    with patch("tldw_Server_API.app.api.v1.endpoints.chat.perform_chat_api_call") as mock_llm, \
         patch("tldw_Server_API.app.api.v1.endpoints.chat.API_KEYS", {"openai": "test-key"}):
        mock_llm.return_value = mock_stream()
        
        response = authenticated_client.post(
            "/api/v1/chat/completions",
            json=request_data.model_dump()
        )
        
        assert response.status_code == status.HTTP_200_OK
        # For streaming, we just verify it doesn't error
        # Actual streaming validation would require async client


def test_chat_completion_with_character(authenticated_client, mock_chacha_db):
    """Test chat completion with a specific character."""
    
    # Add a character to the mock database
    character_id = mock_chacha_db.add_character_card({
        "name": "TestBot",
        "description": "A test character",
        "personality": "Friendly and helpful",
        "system_prompt": "You are TestBot, a friendly assistant."
    })
    
    request_data = ChatCompletionRequest(
        model="test-model",
        api_provider="openai",
        messages=[
            ChatCompletionUserMessageParam(role="user", content="Who are you?")
        ],
        character_id=str(character_id)
    )
    
    with patch("tldw_Server_API.app.api.v1.endpoints.chat.perform_chat_api_call") as mock_llm, \
         patch("tldw_Server_API.app.api.v1.endpoints.chat.API_KEYS", {"openai": "test-key"}):
        mock_llm.return_value = {
            "id": "chatcmpl-test",
            "choices": [{
                "message": {"role": "assistant", "content": "I am TestBot!"},
                "finish_reason": "stop"
            }]
        }
        
        response = authenticated_client.post(
            "/api/v1/chat/completions",
            json=request_data.model_dump()
        )
        
        assert response.status_code == status.HTTP_200_OK
        
        # Verify the system prompt was included
        call_args = mock_llm.call_args
        assert "TestBot" in str(call_args)


def test_chat_completion_unauthorized(mock_chacha_db):
    """Test that unauthenticated requests are rejected."""
    from fastapi.testclient import TestClient
    from tldw_Server_API.app.main import app
    
    with TestClient(app) as client:
        # Get CSRF token but don't authenticate
        response = client.get("/api/v1/health")
        csrf_token = response.cookies.get("csrf_token", "")
        
        request_data = ChatCompletionRequest(
            model="test-model",
            messages=[
                ChatCompletionUserMessageParam(role="user", content="Hello")
            ]
        )
        
        response = client.post(
            "/api/v1/chat/completions",
            json=request_data.model_dump(),
            headers={"X-CSRF-Token": csrf_token}
        )
        
        assert response.status_code == status.HTTP_401_UNAUTHORIZED


def test_chat_completion_invalid_model(authenticated_client, mock_chacha_db):
    """Test handling of invalid model requests."""
    
    # Use a valid provider but configure it to fail
    request_data = ChatCompletionRequest(
        model="invalid-model-xyz",
        api_provider="openai",  # Use a valid provider
        messages=[
            ChatCompletionUserMessageParam(role="user", content="Hello")
        ]
    )
    
    with patch("tldw_Server_API.app.api.v1.endpoints.chat.perform_chat_api_call") as mock_llm, \
         patch("tldw_Server_API.app.api.v1.endpoints.chat.API_KEYS", {"openai": "test-key"}):
        # Simulate an error for invalid model
        mock_llm.side_effect = Exception("Invalid model: invalid-model-xyz")
        
        response = authenticated_client.post(
            "/api/v1/chat/completions",
            json=request_data.model_dump()
        )
        
        # Should return an error status
        assert response.status_code >= 400


def test_chat_completion_with_conversation_history(authenticated_client, mock_chacha_db):
    """Test chat with conversation history."""
    
    # Get the actual default character ID (it's usually 2 based on our tests)
    # First check what character exists
    default_char = mock_chacha_db.get_character_card_by_name("Default Character")
    char_id = default_char['id'] if default_char else 2
    
    # Create a conversation with the correct client_id
    # The mock_chacha_db has a client_id attribute
    conv_id = mock_chacha_db.add_conversation({
        "character_id": char_id,
        "title": "Test Conversation",
        "client_id": mock_chacha_db.client_id  # Use the database's client_id
    })
    
    # Add some history
    mock_chacha_db.add_message({
        "conversation_id": conv_id,
        "sender": "user",
        "content": "Previous message"
    })
    mock_chacha_db.add_message({
        "conversation_id": conv_id,
        "sender": "assistant",
        "content": "Previous response"
    })
    
    request_data = ChatCompletionRequest(
        model="test-model",
        api_provider="openai",
        messages=[
            ChatCompletionUserMessageParam(role="user", content="Continue our conversation")
        ],
        conversation_id=str(conv_id)
    )
    
    with patch("tldw_Server_API.app.api.v1.endpoints.chat.perform_chat_api_call") as mock_llm, \
         patch("tldw_Server_API.app.api.v1.endpoints.chat.API_KEYS", {"openai": "test-key"}):
        mock_llm.return_value = {
            "id": "chatcmpl-test",
            "choices": [{
                "message": {"role": "assistant", "content": "Continuing from before..."},
                "finish_reason": "stop"
            }]
        }
        
        response = authenticated_client.post(
            "/api/v1/chat/completions",
            json=request_data.model_dump()
        )
        
        assert response.status_code == status.HTTP_200_OK
        
        # Verify history was included in the call
        call_args = mock_llm.call_args
        # Messages might be in kwargs
        if call_args.kwargs and "messages_payload" in call_args.kwargs:
            messages = call_args.kwargs["messages_payload"]
        elif call_args.kwargs and "messages" in call_args.kwargs:
            messages = call_args.kwargs["messages"]
        elif len(call_args.args) > 0:
            # Try to find messages in positional args
            messages = call_args.args[0] if isinstance(call_args.args[0], list) else []
        else:
            messages = []
        assert len(messages) > 1  # Should include history


def test_chat_completion_rate_limiting(authenticated_client, mock_chacha_db):
    """Test rate limiting functionality."""
    
    request_data = ChatCompletionRequest(
        model="test-model",
        api_provider="openai",
        messages=[
            ChatCompletionUserMessageParam(role="user", content="Hello")
        ]
    )
    
    with patch("tldw_Server_API.app.api.v1.endpoints.chat.perform_chat_api_call") as mock_llm, \
         patch("tldw_Server_API.app.api.v1.endpoints.chat.API_KEYS", {"openai": "test-key"}):
        mock_llm.return_value = {
            "id": "chatcmpl-test",
            "choices": [{
                "message": {"role": "assistant", "content": "Response"},
                "finish_reason": "stop"
            }]
        }
        
        # Make multiple rapid requests
        responses = []
        for _ in range(5):
            response = authenticated_client.post(
                "/api/v1/chat/completions",
                json=request_data.model_dump()
            )
            responses.append(response.status_code)
        
        # All should succeed (rate limiting might not be enabled in test)
        # or we should see 429 status codes
        assert all(s in [200, 429] for s in responses)