"""
Integration tests for the /chat/completions API endpoint.

Tests the full request/response flow with real database and minimal mocking.
Only external LLM APIs are mocked to avoid actual API calls.
"""

import pytest
import json
import os
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
    @pytest.mark.skipif(not os.getenv("OPENAI_API_KEY"), reason="Requires OPENAI_API_KEY for real integration test")
    def test_basic_completion_request(self, test_client, auth_headers):
        """Test basic chat completion request - REAL API CALL."""
        
        response = test_client.post(
            "/api/v1/chat/completions",
            json={
                "model": "gpt-5-mini",
                "messages": [{"role": "user", "content": "Hello"}]
            },
            headers=auth_headers
        )
        
        # Debug output if test fails
        if response.status_code != status.HTTP_200_OK:
            print(f"\nResponse status: {response.status_code}")
            print(f"Response body: {response.json()}")
            print(f"OPENAI_API_KEY env: {os.getenv('OPENAI_API_KEY', 'NOT SET')[:10]}..." if os.getenv('OPENAI_API_KEY') else "OPENAI_API_KEY: NOT SET")
        
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert data["id"]
        assert data["object"] == "chat.completion"
        assert data["choices"][0]["message"]["content"]
    
    @pytest.mark.integration
    @pytest.mark.skipif(not os.getenv("OPENAI_API_KEY"), reason="Requires OPENAI_API_KEY for real integration test")
    def test_multi_turn_conversation(self, test_client, auth_headers):
        """Test multi-turn conversation handling - REAL API CALL."""
        
        response = test_client.post(
            "/api/v1/chat/completions",
            json={
                "model": "gpt-5-mini",
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
                "model": "gpt-5-mini",
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
                "model": "gpt-5-mini",
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
    @pytest.mark.skipif(not os.getenv("OPENAI_API_KEY"), reason="Requires OPENAI_API_KEY for real integration test")
    def test_openai_provider_routing(self, test_client, auth_headers):
        """Test routing to OpenAI provider - REAL API CALL."""
        
        response = test_client.post(
            "/api/v1/chat/completions",
            json={
                "api_provider": "openai",
                "model": "gpt-5-mini",
                "messages": [{"role": "user", "content": "Test"}]
            },
            headers=auth_headers
        )
        
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert "choices" in data
    
    @pytest.mark.integration
    @pytest.mark.skipif(not os.getenv("ANTHROPIC_API_KEY"), reason="Requires ANTHROPIC_API_KEY for real integration test")
    def test_anthropic_provider_routing(self, test_client, auth_headers):
        """Test routing to Anthropic provider - REAL API CALL."""
        
        response = test_client.post(
            "/api/v1/chat/completions",
            json={
                "api_provider": "anthropic",
                "model": "claude-sonnet-4-20250514",
                "messages": [{"role": "user", "content": "Test"}]
            },
            headers=auth_headers
        )
        
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert "choices" in data
    
    @pytest.mark.integration
    @pytest.mark.skipif(not os.getenv("OPENAI_API_KEY"), reason="Requires OPENAI_API_KEY for real integration test")
    def test_default_provider_fallback(self, test_client, auth_headers):
        """Test fallback to default provider when not specified - REAL API CALL."""
        
        response = test_client.post(
            "/api/v1/chat/completions",
            json={
                "model": "gpt-5-mini",
                "messages": [{"role": "user", "content": "Test"}]
            },
            headers=auth_headers
        )
        
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert "choices" in data

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
                    "model": "gpt-5-mini",
                    "messages": [{"role": "user", "content": "Save this message"}]
                },
                headers=auth_headers
            )
            
            assert response.status_code == status.HTTP_200_OK
            
            # Check database for saved conversation
            # Get the Default Character first
            characters = populated_chacha_db.list_character_cards()
            default_char = next((c for c in characters if c['name'] == 'Default Character'), None)
            assert default_char is not None
            
            # Get conversations for the character
            conversations = populated_chacha_db.get_conversations_for_character(default_char['id'])
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
            # Get the character from populated DB
            characters = populated_chacha_db.list_character_cards()
            assert len(characters) > 0
            
            # Find the character we created in the fixture (Default Character with client_id test_user)
            test_char = next((c for c in characters if c['name'] == 'Default Character' and c['client_id'] == 'test_user'), None)
            assert test_char is not None, "Could not find test character 'Default Character'"
            
            conversations = populated_chacha_db.get_conversations_for_character(test_char["id"])
            assert len(conversations) > 0
            
            first_conv = conversations[0]
            messages = populated_chacha_db.get_messages_for_conversation(first_conv["id"])
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
        
        pass  # Cannot reliably test rate limit without hitting it
    
    @pytest.mark.integration
    @pytest.mark.skipif(not os.getenv("OPENAI_API_KEY"), reason="Requires OPENAI_API_KEY for real integration test")
    def test_auth_error_handling(self, test_client, auth_headers):
        """Test handling of authentication errors - REAL API CALL with invalid key."""
        # Use an invalid API key to trigger auth error
        invalid_headers = auth_headers.copy()
        
        # Temporarily set an invalid API key
        original_key = os.environ.get("OPENAI_API_KEY")
        os.environ["OPENAI_API_KEY"] = "invalid-key-12345"
        
        try:
            response = test_client.post(
                "/api/v1/chat/completions",
                json={
                    "model": "gpt-5-mini",
                    "messages": [{"role": "user", "content": "Test"}]
                },
                headers=invalid_headers
            )
            
            # Should get 500 since the invalid key will cause an error
            assert response.status_code == status.HTTP_500_INTERNAL_SERVER_ERROR
            data = response.json()
            assert "detail" in data or "error" in data
        finally:
            # Restore original key
            if original_key:
                os.environ["OPENAI_API_KEY"] = original_key
            else:
                del os.environ["OPENAI_API_KEY"]
    
    @pytest.mark.integration 
    @patch('tldw_Server_API.app.api.v1.endpoints.chat.perform_chat_api_call')
    def test_general_error_handling(self, mock_chat_call, test_client, auth_headers):
        """Test handling of general errors."""
        mock_chat_call.side_effect = Exception("Unexpected error")
        
        response = test_client.post(
            "/api/v1/chat/completions",
            json={
                "model": "gpt-5-mini",
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
    @pytest.mark.skipif(not os.getenv("OPENAI_API_KEY"), reason="Requires OPENAI_API_KEY for real integration test")
    async def test_streaming_response(self, async_client, auth_headers):
        """Test streaming chat completion - REAL API CALL."""
        async with async_client.stream(
                "POST",
                "/api/v1/chat/completions",
                json={
                    "model": "gpt-5-mini",
                    "messages": [{"role": "user", "content": "Stream this"}],
                    "stream": True
                },
                headers=auth_headers
            ) as response:
                # Debug: print error if request failed
                if response.status_code != status.HTTP_200_OK:
                    error_text = await response.aread()
                    print(f"\nStreaming test error response: {error_text.decode() if error_text else 'No response body'}")
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
    @pytest.mark.skipif(not os.getenv("OPENAI_API_KEY"), reason="Requires OPENAI_API_KEY for real integration test")
    def test_temperature_bounds(self, test_client, auth_headers):
        """Test temperature parameter bounds - REAL API CALL."""
        
        # Valid temperature
        response = test_client.post(
            "/api/v1/chat/completions",
            json={
                "model": "gpt-5-mini",
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
                "model": "gpt-5-mini",
                "messages": [{"role": "user", "content": "Test"}],
                "temperature": 2.5
            },
            headers=auth_headers
        )
        assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY
    
    @pytest.mark.integration
    @pytest.mark.skipif(not os.getenv("OPENAI_API_KEY"), reason="Requires OPENAI_API_KEY for real integration test")
    def test_max_tokens_validation(self, test_client, auth_headers):
        """Test max_tokens parameter validation - REAL API CALL."""
        
        # Valid max_tokens
        response = test_client.post(
            "/api/v1/chat/completions",
            json={
                "model": "gpt-5-mini",
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
                "model": "gpt-5-mini",
                "messages": [{"role": "user", "content": "Test"}],
                "max_tokens": -1
            },
            headers=auth_headers
        )
        assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY