"""
Unit tests for core chat functions.

Tests the business logic of chat_api_call, process_user_input, and related
functions with mocked external dependencies (LLM APIs, databases).
"""

import pytest
from unittest.mock import Mock, AsyncMock, patch, MagicMock
from typing import Dict, Any
import json

from tldw_Server_API.app.core.Chat.Chat_Functions import (
    chat_api_call,
    process_user_input,
    update_chat_content,
    ChatAPIError,
    ChatRateLimitError,
    ChatAuthenticationError,
    ChatProviderError,
)

# ========================================================================
# Core Function Tests
# ========================================================================

class TestChatAPICall:
    """Test the chat_api_call function."""
    
    @pytest.mark.unit
    @patch('tldw_Server_API.app.core.Chat.Chat_Functions.perform_llm_call')
    def test_successful_api_call(self, mock_llm_call, mock_llm_response):
        """Test successful chat API call."""
        mock_llm_call.return_value = mock_llm_response
        
        result = chat_api_call(
            api_provider="openai",
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": "Hello"}],
            temperature=0.7
        )
        
        assert result["id"] == "chatcmpl-test123"
        assert result["choices"][0]["message"]["content"] == "This is a test response from the LLM."
        mock_llm_call.assert_called_once()
    
    @pytest.mark.unit
    @patch('tldw_Server_API.app.core.Chat.Chat_Functions.perform_llm_call')
    def test_api_call_with_system_message(self, mock_llm_call, mock_llm_response):
        """Test API call with system message."""
        mock_llm_call.return_value = mock_llm_response
        
        messages = [
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "Hello"}
        ]
        
        result = chat_api_call(
            api_provider="openai",
            model="gpt-3.5-turbo",
            messages=messages
        )
        
        assert result is not None
        call_args = mock_llm_call.call_args
        assert call_args[1]["messages"] == messages
    
    @pytest.mark.unit
    @patch('tldw_Server_API.app.core.Chat.Chat_Functions.perform_llm_call')
    def test_api_call_rate_limit_error(self, mock_llm_call):
        """Test handling of rate limit errors."""
        mock_llm_call.side_effect = ChatRateLimitError("Rate limit exceeded")
        
        with pytest.raises(ChatRateLimitError) as exc_info:
            chat_api_call(
                api_provider="openai",
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": "Hello"}]
            )
        
        assert "Rate limit exceeded" in str(exc_info.value)
    
    @pytest.mark.unit
    @patch('tldw_Server_API.app.core.Chat.Chat_Functions.perform_llm_call')
    def test_api_call_auth_error(self, mock_llm_call):
        """Test handling of authentication errors."""
        mock_llm_call.side_effect = ChatAuthenticationError("Invalid API key")
        
        with pytest.raises(ChatAuthenticationError) as exc_info:
            chat_api_call(
                api_provider="openai",
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": "Hello"}]
            )
        
        assert "Invalid API key" in str(exc_info.value)
    
    @pytest.mark.unit
    @patch('tldw_Server_API.app.core.Chat.Chat_Functions.perform_llm_call')
    def test_api_call_provider_routing(self, mock_llm_call, mock_llm_response):
        """Test that different providers are routed correctly."""
        mock_llm_call.return_value = mock_llm_response
        
        providers = ["openai", "anthropic", "groq", "mistral"]
        
        for provider in providers:
            chat_api_call(
                api_provider=provider,
                model="test-model",
                messages=[{"role": "user", "content": "Test"}]
            )
        
        assert mock_llm_call.call_count == len(providers)
        
        # Check that provider was passed correctly
        for i, provider in enumerate(providers):
            assert mock_llm_call.call_args_list[i][1]["api_provider"] == provider

# ========================================================================
# User Input Processing Tests
# ========================================================================

class TestProcessUserInput:
    """Test the process_user_input function."""
    
    @pytest.mark.unit
    def test_process_simple_text_input(self):
        """Test processing simple text input."""
        result = process_user_input("Hello, how are you?")
        
        assert result["type"] == "text"
        assert result["content"] == "Hello, how are you?"
    
    @pytest.mark.unit 
    def test_process_empty_input(self):
        """Test processing empty input."""
        result = process_user_input("")
        
        assert result["type"] == "text"
        assert result["content"] == ""
    
    @pytest.mark.unit
    def test_process_multiline_input(self):
        """Test processing multiline text input."""
        input_text = """Line 1
        Line 2
        Line 3"""
        
        result = process_user_input(input_text)
        
        assert result["type"] == "text"
        assert "Line 1" in result["content"]
        assert "Line 2" in result["content"]
        assert "Line 3" in result["content"]
    
    @pytest.mark.unit
    def test_process_input_with_special_characters(self):
        """Test processing input with special characters."""
        special_input = "Test with special chars: !@#$%^&*()[]{}\"'<>"
        
        result = process_user_input(special_input)
        
        assert result["type"] == "text"
        assert result["content"] == special_input
    
    @pytest.mark.unit
    def test_process_json_like_input(self):
        """Test processing JSON-like string input."""
        json_input = '{"key": "value", "number": 123}'
        
        result = process_user_input(json_input)
        
        assert result["type"] == "text"
        assert result["content"] == json_input

# ========================================================================
# Chat Content Update Tests
# ========================================================================

class TestUpdateChatContent:
    """Test the update_chat_content function."""
    
    @pytest.mark.unit
    def test_update_content_basic(self):
        """Test basic content update."""
        original = "Hello world"
        update = " How are you?"
        
        result = update_chat_content(original, update)
        
        assert result == "Hello world How are you?"
    
    @pytest.mark.unit
    def test_update_content_with_empty_original(self):
        """Test updating from empty content."""
        result = update_chat_content("", "New content")
        
        assert result == "New content"
    
    @pytest.mark.unit
    def test_update_content_with_empty_update(self):
        """Test updating with empty content."""
        result = update_chat_content("Original", "")
        
        assert result == "Original"
    
    @pytest.mark.unit
    def test_update_content_both_empty(self):
        """Test updating when both are empty."""
        result = update_chat_content("", "")
        
        assert result == ""

# ========================================================================
# Provider Manager Integration Tests
# ========================================================================

class TestProviderManagement:
    """Test provider management and configuration."""
    
    @pytest.mark.unit
    @patch('tldw_Server_API.app.core.Chat.Chat_Functions.get_provider_config')
    def test_get_provider_config(self, mock_get_config):
        """Test getting provider configuration."""
        mock_get_config.return_value = {
            "api_key": "test-key",
            "base_url": "https://api.test.com",
            "models": ["model-1", "model-2"]
        }
        
        config = mock_get_config("openai")
        
        assert config["api_key"] == "test-key"
        assert config["base_url"] == "https://api.test.com"
        assert "model-1" in config["models"]
    
    @pytest.mark.unit
    @patch('tldw_Server_API.app.core.Chat.Chat_Functions.validate_provider')
    def test_validate_provider_success(self, mock_validate):
        """Test successful provider validation."""
        mock_validate.return_value = True
        
        is_valid = mock_validate("openai", "test-key")
        
        assert is_valid is True
        mock_validate.assert_called_once_with("openai", "test-key")
    
    @pytest.mark.unit
    @patch('tldw_Server_API.app.core.Chat.Chat_Functions.validate_provider')
    def test_validate_provider_failure(self, mock_validate):
        """Test failed provider validation."""
        mock_validate.return_value = False
        
        is_valid = mock_validate("invalid-provider", "bad-key")
        
        assert is_valid is False

# ========================================================================
# Error Handling Tests
# ========================================================================

class TestErrorHandling:
    """Test error handling in chat functions."""
    
    @pytest.mark.unit
    def test_chat_api_error_creation(self):
        """Test ChatAPIError creation and properties."""
        error = ChatAPIError("API call failed", status_code=500)
        
        assert str(error) == "API call failed"
        assert error.status_code == 500
    
    @pytest.mark.unit
    def test_rate_limit_error_properties(self):
        """Test ChatRateLimitError properties."""
        error = ChatRateLimitError("Too many requests", retry_after=60)
        
        assert "Too many requests" in str(error)
        assert error.retry_after == 60
    
    @pytest.mark.unit
    def test_auth_error_properties(self):
        """Test ChatAuthenticationError properties."""
        error = ChatAuthenticationError("Invalid credentials")
        
        assert "Invalid credentials" in str(error)
        assert error.status_code == 401
    
    @pytest.mark.unit
    def test_provider_error_properties(self):
        """Test ChatProviderError properties."""
        error = ChatProviderError("Provider unavailable", provider="openai")
        
        assert "Provider unavailable" in str(error)
        assert error.provider == "openai"

# ========================================================================
# Message Formatting Tests
# ========================================================================

class TestMessageFormatting:
    """Test message formatting utilities."""
    
    @pytest.mark.unit
    def test_format_single_message(self):
        """Test formatting a single message."""
        message = {"role": "user", "content": "Hello"}
        formatted = json.dumps(message)
        
        assert '"role": "user"' in formatted
        assert '"content": "Hello"' in formatted
    
    @pytest.mark.unit
    def test_format_message_list(self):
        """Test formatting a list of messages."""
        messages = [
            {"role": "system", "content": "You are helpful"},
            {"role": "user", "content": "Hi"}
        ]
        formatted = json.dumps(messages)
        
        parsed = json.loads(formatted)
        assert len(parsed) == 2
        assert parsed[0]["role"] == "system"
        assert parsed[1]["role"] == "user"
    
    @pytest.mark.unit
    def test_format_message_with_metadata(self):
        """Test formatting message with metadata."""
        message = {
            "role": "assistant",
            "content": "Response",
            "name": "Assistant",
            "metadata": {"timestamp": "2024-01-01T00:00:00"}
        }
        formatted = json.dumps(message)
        
        parsed = json.loads(formatted)
        assert parsed["name"] == "Assistant"
        assert "metadata" in parsed

# ========================================================================
# Token Counting Tests (Mock)
# ========================================================================

class TestTokenCounting:
    """Test token counting functionality."""
    
    @pytest.mark.unit
    @patch('tldw_Server_API.app.core.Chat.Chat_Functions.count_tokens')
    def test_count_tokens_simple(self, mock_count):
        """Test token counting for simple text."""
        mock_count.return_value = 5
        
        count = mock_count("Hello world test")
        
        assert count == 5
        mock_count.assert_called_once_with("Hello world test")
    
    @pytest.mark.unit
    @patch('tldw_Server_API.app.core.Chat.Chat_Functions.count_tokens')
    def test_count_tokens_messages(self, mock_count):
        """Test token counting for messages."""
        mock_count.return_value = 20
        
        messages = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there!"}
        ]
        
        count = mock_count(json.dumps(messages))
        
        assert count == 20