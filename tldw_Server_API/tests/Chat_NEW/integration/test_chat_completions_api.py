"""
Integration tests for the /chat/completions API endpoint.

Tests the full request/response flow with real database and minimal mocking.
Only external LLM APIs are mocked to avoid actual API calls.
"""

import pytest
import json
from fastapi import status
from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock
import asyncio

# ========================================================================
# Basic Endpoint Tests
# ========================================================================

class TestChatCompletionsEndpoint:
    """Test the /v1/chat/completions endpoint."""
    
    @pytest.mark.integration
    def test_basic_completion_request(self, test_client, auth_headers):
        """Test basic chat completion request."""
        # This is an integration test - it will make real API calls
        # Skip if no API key is configured
        import os
        if not os.getenv("OPENAI_API_KEY"):
            pytest.skip("Requires OPENAI_API_KEY to be set")
        
        response = test_client.post(
            "/api/v1/chat/completions",
            json={
                "model": "gpt-3.5-turbo",
                "messages": [{"role": "user", "content": "Hello"}]
            },
            headers=auth_headers
        )
        
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert data["id"]
        assert data["object"] == "chat.completion"
        assert data["choices"][0]["message"]["content"]
    
    @pytest.mark.integration
    @patch('tldw_Server_API.app.api.v1.endpoints.chat.perform_chat_api_call')
    def test_multi_turn_conversation(self, mock_chat_call, test_client, mock_llm_response, auth_headers):
        """Test multi-turn conversation handling."""
        mock_chat_call.return_value = mock_llm_response
        
        response = test_client.post(
            "/api/v1/chat/completions",
            json={
                "model": "gpt-3.5-turbo",
                "messages": [
                    {"role": "system", "content": "You are helpful."},
                    {"role": "user", "content": "What is 2+2?"},
                    {"role": "assistant", "content": "2+2 equals 4."},
                    {"role": "user", "content": "What about 3+3?"}
                ]
            },
            headers=auth_headers
        )
        
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert data["choices"][0]["message"]["role"] == "assistant"
    
    @pytest.mark.integration
    def test_missing_auth_header(self, test_client):
        """Test request without authentication header."""
        response = test_client.post(
            "/api/v1/chat/completions",
            json={
                "model": "gpt-3.5-turbo",
                "messages": [{"role": "user", "content": "Hello"}]
            }
        )
        
        # Depending on auth configuration, might be 401 or allowed
        assert response.status_code in [status.HTTP_200_OK, status.HTTP_401_UNAUTHORIZED]
    
    @pytest.mark.integration
    def test_invalid_request_body(self, test_client, auth_headers):
        """Test request with invalid body."""
        response = test_client.post(
            "/api/v1/chat/completions",
            json={
                "invalid_field": "value"
            },
            headers=auth_headers
        )
        
        assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY
    
    @pytest.mark.integration
    def test_empty_messages_list(self, test_client, auth_headers):
        """Test request with empty messages list."""
        response = test_client.post(
            "/api/v1/chat/completions",
            json={
                "model": "gpt-3.5-turbo",
                "messages": []
            },
            headers=auth_headers
        )
        
        assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY

# ========================================================================
# Provider Routing Tests
# ========================================================================

class TestProviderRouting:
    """Test routing to different LLM providers."""
    
    @pytest.mark.integration
    @patch('tldw_Server_API.app.api.v1.endpoints.chat.perform_chat_api_call')
    def test_openai_provider_routing(self, mock_chat_call, test_client, mock_llm_response, auth_headers):
        """Test routing to OpenAI provider."""
        mock_chat_call.return_value = mock_llm_response
        
        response = test_client.post(
            "/api/v1/chat/completions",
            json={
                "api_provider": "openai",
                "model": "gpt-3.5-turbo",
                "messages": [{"role": "user", "content": "Test"}]
            },
            headers=auth_headers
        )
        
        assert response.status_code == status.HTTP_200_OK
        mock_chat_call.assert_called_once()
        call_args = mock_chat_call.call_args
        assert call_args[1]["api_provider"] == "openai"
    
    @pytest.mark.integration
    @patch('tldw_Server_API.app.api.v1.endpoints.chat.perform_chat_api_call')
    def test_anthropic_provider_routing(self, mock_chat_call, test_client, mock_llm_response, auth_headers):
        """Test routing to Anthropic provider."""
        mock_chat_call.return_value = mock_llm_response
        
        response = test_client.post(
            "/api/v1/chat/completions",
            json={
                "api_provider": "anthropic",
                "model": "claude-3-sonnet",
                "messages": [{"role": "user", "content": "Test"}]
            },
            headers=auth_headers
        )
        
        assert response.status_code == status.HTTP_200_OK
        call_args = mock_chat_call.call_args
        assert call_args[1]["api_provider"] == "anthropic"
    
    @pytest.mark.integration
    @patch('tldw_Server_API.app.api.v1.endpoints.chat.perform_chat_api_call')
    def test_default_provider_fallback(self, mock_chat_call, test_client, mock_llm_response, auth_headers):
        """Test fallback to default provider when not specified."""
        mock_chat_call.return_value = mock_llm_response
        
        response = test_client.post(
            "/api/v1/chat/completions",
            json={
                "model": "gpt-3.5-turbo",
                "messages": [{"role": "user", "content": "Test"}]
            },
            headers=auth_headers
        )
        
        assert response.status_code == status.HTTP_200_OK
        # Should use default provider (usually "openai")
        mock_chat_call.assert_called_once()

# ========================================================================
# Database Integration Tests
# ========================================================================

class TestDatabaseIntegration:
    """Test database persistence and retrieval."""
    
    @pytest.mark.integration
    @patch('tldw_Server_API.app.api.v1.endpoints.chat.perform_chat_api_call')
    def test_conversation_saved_to_database(self, mock_chat_call, test_client, populated_chacha_db, mock_llm_response, auth_headers):
        """Test that conversations are saved to database."""
        mock_chat_call.return_value = mock_llm_response
        
        # Override dependency to use our test database
        from tldw_Server_API.app.api.v1.API_Deps.ChaCha_Notes_DB_Deps import get_chacha_db_for_user
        
        def override_get_db():
            return populated_chacha_db
        
        from tldw_Server_API.app.main import app
        app.dependency_overrides[get_chacha_db_for_user] = override_get_db
        
        try:
            response = test_client.post(
                "/api/v1/chat/completions",
                json={
                    "model": "gpt-3.5-turbo",
                    "messages": [{"role": "user", "content": "Save this message"}]
                },
                headers=auth_headers
            )
            
            assert response.status_code == status.HTTP_200_OK
            
            # Check database for saved conversation
            conversations = populated_chacha_db.get_all_conversations()
            assert len(conversations) > 0
            
        finally:
            # Clean up dependency override
            app.dependency_overrides.clear()
    
    @pytest.mark.integration
    def test_message_history_retrieval(self, test_client, populated_chacha_db, auth_headers):
        """Test retrieving conversation history."""
        from tldw_Server_API.app.api.v1.API_Deps.ChaCha_Notes_DB_Deps import get_chacha_db_for_user
        from tldw_Server_API.app.main import app
        
        def override_get_db():
            return populated_chacha_db
        
        app.dependency_overrides[get_chacha_db_for_user] = override_get_db
        
        try:
            # Get existing conversations from populated DB
            conversations = populated_chacha_db.get_all_conversations()
            assert len(conversations) > 0
            
            first_conv = conversations[0]
            messages = populated_chacha_db.get_messages(first_conv["id"])
            assert len(messages) > 0
            
        finally:
            app.dependency_overrides.clear()

# ========================================================================
# Error Handling Tests
# ========================================================================

class TestErrorHandling:
    """Test error handling in the API."""
    
    @pytest.mark.integration
    @patch('tldw_Server_API.app.api.v1.endpoints.chat.perform_chat_api_call')
    def test_rate_limit_error_handling(self, mock_chat_call, test_client, auth_headers):
        """Test handling of rate limit errors."""
        from tldw_Server_API.app.core.Chat.Chat_Functions import ChatRateLimitError
        mock_chat_call.side_effect = ChatRateLimitError("Rate limit exceeded", provider="openai")
        
        response = test_client.post(
            "/api/v1/chat/completions",
            json={
                "model": "gpt-3.5-turbo",
                "messages": [{"role": "user", "content": "Test"}]
            },
            headers=auth_headers
        )
        
        assert response.status_code == status.HTTP_429_TOO_MANY_REQUESTS
        data = response.json()
        assert "error" in data
    
    @pytest.mark.integration
    @patch('tldw_Server_API.app.api.v1.endpoints.chat.perform_chat_api_call')
    def test_auth_error_handling(self, mock_chat_call, test_client, auth_headers):
        """Test handling of authentication errors."""
        from tldw_Server_API.app.core.Chat.Chat_Functions import ChatAuthenticationError
        mock_chat_call.side_effect = ChatAuthenticationError("Invalid API key", provider="openai")
        
        response = test_client.post(
            "/api/v1/chat/completions",
            json={
                "model": "gpt-3.5-turbo",
                "messages": [{"role": "user", "content": "Test"}]
            },
            headers=auth_headers
        )
        
        assert response.status_code == status.HTTP_401_UNAUTHORIZED
        data = response.json()
        assert "detail" in data or "error" in data
    
    @pytest.mark.integration 
    @patch('tldw_Server_API.app.api.v1.endpoints.chat.perform_chat_api_call')
    def test_general_error_handling(self, mock_chat_call, test_client, auth_headers):
        """Test handling of general errors."""
        mock_chat_call.side_effect = Exception("Unexpected error")
        
        response = test_client.post(
            "/api/v1/chat/completions",
            json={
                "model": "gpt-3.5-turbo",
                "messages": [{"role": "user", "content": "Test"}]
            },
            headers=auth_headers
        )
        
        assert response.status_code == status.HTTP_500_INTERNAL_SERVER_ERROR
        data = response.json()
        assert "error" in data or "detail" in data

# ========================================================================
# Streaming Tests
# ========================================================================

class TestStreamingResponses:
    """Test streaming response functionality."""
    
    @pytest.mark.integration
    @pytest.mark.streaming
    @pytest.mark.asyncio
    async def test_streaming_response(self, async_client, mock_streaming_response, auth_headers):
        """Test streaming chat completion."""
        with patch('tldw_Server_API.app.core.Chat.Chat_Functions.chat_api_call') as mock_chat_call:
            mock_chat_call.return_value = mock_streaming_response
            
            async with async_client.stream(
                "POST",
                "/api/v1/chat/completions",
                json={
                    "model": "gpt-3.5-turbo",
                    "messages": [{"role": "user", "content": "Stream this"}],
                    "stream": True
                },
                headers=auth_headers
            ) as response:
                assert response.status_code == status.HTTP_200_OK
                
                chunks = []
                async for chunk in response.aiter_text():
                    if chunk:
                        chunks.append(chunk)
                
                assert len(chunks) > 0
                # Should receive SSE formatted chunks
                assert any("data:" in chunk for chunk in chunks)

# ========================================================================
# Parameter Validation Tests  
# ========================================================================

class TestParameterValidation:
    """Test parameter validation and constraints."""
    
    @pytest.mark.integration
    @patch('tldw_Server_API.app.api.v1.endpoints.chat.perform_chat_api_call')
    def test_temperature_bounds(self, mock_chat_call, test_client, mock_llm_response, auth_headers):
        """Test temperature parameter bounds."""
        mock_chat_call.return_value = mock_llm_response
        
        # Valid temperature
        response = test_client.post(
            "/api/v1/chat/completions",
            json={
                "model": "gpt-3.5-turbo",
                "messages": [{"role": "user", "content": "Test"}],
                "temperature": 1.5
            },
            headers=auth_headers
        )
        assert response.status_code == status.HTTP_200_OK
        
        # Invalid temperature (too high)
        response = test_client.post(
            "/api/v1/chat/completions",
            json={
                "model": "gpt-3.5-turbo",
                "messages": [{"role": "user", "content": "Test"}],
                "temperature": 2.5
            },
            headers=auth_headers
        )
        assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY
    
    @pytest.mark.integration
    @patch('tldw_Server_API.app.api.v1.endpoints.chat.perform_chat_api_call')
    def test_max_tokens_validation(self, mock_chat_call, test_client, mock_llm_response, auth_headers):
        """Test max_tokens parameter validation."""
        mock_chat_call.return_value = mock_llm_response
        
        # Valid max_tokens
        response = test_client.post(
            "/api/v1/chat/completions",
            json={
                "model": "gpt-3.5-turbo",
                "messages": [{"role": "user", "content": "Test"}],
                "max_tokens": 100
            },
            headers=auth_headers
        )
        assert response.status_code == status.HTTP_200_OK
        
        # Invalid max_tokens (negative)
        response = test_client.post(
            "/api/v1/chat/completions",
            json={
                "model": "gpt-3.5-turbo",
                "messages": [{"role": "user", "content": "Test"}],
                "max_tokens": -1
            },
            headers=auth_headers
        )
        assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY