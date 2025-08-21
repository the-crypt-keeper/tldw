"""
Simple chat endpoint test using the simplified fixtures.
"""
import pytest
from unittest.mock import patch
from fastapi import status

# Fixtures are imported automatically from conftest.py

from tldw_Server_API.app.api.v1.schemas.chat_request_schemas import (
    ChatCompletionRequest,
    ChatCompletionUserMessageParam
)


def test_chat_completion_works(client, auth_token, mock_chacha_db):
    """Test that chat completion works with proper auth."""
    
    request_data = ChatCompletionRequest(
        model="test-model",
        messages=[
            ChatCompletionUserMessageParam(role="user", content="Hello, how are you?")
        ]
    )
    
    # Mock the LLM call
    with patch("tldw_Server_API.app.api.v1.endpoints.chat.perform_chat_api_call") as mock_llm:
        mock_llm.return_value = {
            "id": "chatcmpl-test",
            "choices": [{
                "message": {"role": "assistant", "content": "I'm doing well!"},
                "finish_reason": "stop"
            }]
        }
        
        # Make request with authentication
        from tldw_Server_API.app.core.AuthNZ.settings import get_settings
        settings = get_settings()
        print(f"AUTH_MODE: {settings.AUTH_MODE}")
        print(f"Expected API key: {settings.SINGLE_USER_API_KEY}")
        print(f"Auth token from fixture: {auth_token[:50]}...")
        
        # Debug: make the request manually to see what headers are sent
        headers = {"X-CSRF-Token": client.csrf_token}
        if settings.AUTH_MODE == "multi_user":
            headers["Authorization"] = auth_token
            print(f"Using Authorization header")
        else:
            # Use X-API-KEY (with hyphen, all caps) as expected by the dependency
            headers["X-API-KEY"] = auth_token
            print(f"Using X-API-KEY header")
        
        print(f"Headers being sent: {headers}")
        
        response = client.post(
            "/api/v1/chat/completions",
            json=request_data.model_dump(),
            headers=headers
        )
        
        # Check response
        print(f"Status: {response.status_code}")
        if response.status_code != 200:
            print(f"Response: {response.text}")
            from tldw_Server_API.app.main import app as main_app
            print(f"Main app overrides: {list(main_app.dependency_overrides.keys())}")
        
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert data["choices"][0]["message"]["content"] == "I'm doing well!"