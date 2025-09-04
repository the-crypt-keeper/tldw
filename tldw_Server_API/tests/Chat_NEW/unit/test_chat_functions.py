"""
Unit tests for core chat functions.

Tests the business logic of chat_api_call, process_user_input, and related
functions with mocked external dependencies (LLM APIs, databases).
"""

import pytest
from unittest.mock import Mock, AsyncMock, patch, MagicMock
from typing import Dict, Any
import json

from tldw_Server_API.app.core.Chat import Chat_Functions
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
    @patch.dict('tldw_Server_API.app.core.Chat.Chat_Functions.API_CALL_HANDLERS')
    def test_successful_api_call(self, mock_llm_response):
        """Test successful chat API call."""
        mock_handler = MagicMock(return_value=mock_llm_response)
        mock_handler.__name__ = 'mock_chat_with_openai'  # Add __name__ attribute
        Chat_Functions.API_CALL_HANDLERS['openai'] = mock_handler
        
        result = chat_api_call(
            api_endpoint="openai",
            messages_payload=[{"role": "user", "content": "Hello"}],
            model="gpt-3.5-turbo",
            temp=0.7
        )
        
        assert result["id"] == "chatcmpl-test123"
        assert result["choices"][0]["message"]["content"] == "This is a test response from the LLM."
        mock_handler.assert_called_once()
    
    @pytest.mark.unit
    @patch.dict('tldw_Server_API.app.core.Chat.Chat_Functions.API_CALL_HANDLERS')
    def test_api_call_with_system_message(self, mock_llm_response):
        """Test API call with system message."""
        mock_handler = MagicMock(return_value=mock_llm_response)
        mock_handler.__name__ = 'mock_chat_with_openai'
        Chat_Functions.API_CALL_HANDLERS['openai'] = mock_handler
        
        messages = [
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "Hello"}
        ]
        
        result = chat_api_call(
            api_endpoint="openai",
            messages_payload=messages,
            model="gpt-3.5-turbo"
        )
        
        assert result is not None
        mock_handler.assert_called_once()
    
    @pytest.mark.unit
    @patch.dict('tldw_Server_API.app.core.Chat.Chat_Functions.API_CALL_HANDLERS')
    def test_api_call_rate_limit_error(self):
        """Test handling of rate limit errors."""
        mock_handler = MagicMock(side_effect=ChatRateLimitError("Rate limit exceeded", provider="openai"))
        mock_handler.__name__ = 'mock_chat_with_openai'
        Chat_Functions.API_CALL_HANDLERS['openai'] = mock_handler
        
        with pytest.raises(ChatRateLimitError) as exc_info:
            chat_api_call(
                api_endpoint="openai",
                messages_payload=[{"role": "user", "content": "Hello"}],
                model="gpt-3.5-turbo"
            )
        
        assert "Rate limit exceeded" in str(exc_info.value)
    
    @pytest.mark.unit
    @patch.dict('tldw_Server_API.app.core.Chat.Chat_Functions.API_CALL_HANDLERS')
    def test_api_call_auth_error(self):
        """Test handling of authentication errors."""
        mock_handler = MagicMock(side_effect=ChatAuthenticationError("Invalid API key", provider="openai"))
        mock_handler.__name__ = 'mock_chat_with_openai'
        Chat_Functions.API_CALL_HANDLERS['openai'] = mock_handler
        
        with pytest.raises(ChatAuthenticationError) as exc_info:
            chat_api_call(
                api_endpoint="openai",
                messages_payload=[{"role": "user", "content": "Hello"}],
                model="gpt-3.5-turbo"
            )
        
        assert "Invalid API key" in str(exc_info.value)
    
    @pytest.mark.unit
    @patch.dict('tldw_Server_API.app.core.Chat.Chat_Functions.API_CALL_HANDLERS')
    def test_api_call_provider_routing(self, mock_llm_response):
        """Test that different providers are routed correctly."""
        providers = ["openai", "anthropic", "groq", "mistral"]
        
        # Create mocks for each provider
        mocks = {}
        for provider in providers:
            mock_handler = MagicMock(return_value=mock_llm_response)
            mock_handler.__name__ = f'mock_chat_with_{provider}'
            Chat_Functions.API_CALL_HANDLERS[provider] = mock_handler
            mocks[provider] = mock_handler
        
        # Call each provider
        for provider in providers:
            chat_api_call(
                api_endpoint=provider,
                messages_payload=[{"role": "user", "content": "Test"}],
                model="test-model"
            )
        
        # Verify each provider was called once
        for provider, mock in mocks.items():
            assert mock.call_count == 1

# ========================================================================
# User Input Processing Tests
# ========================================================================

class TestProcessUserInput:
    """Test the process_user_input function."""
    
    @pytest.mark.unit
    def test_process_simple_text_input(self):
        """Test processing simple text input."""
        result = process_user_input("Hello, how are you?", entries=[])
        
        assert isinstance(result, str)
        assert result == "Hello, how are you?"
    
    @pytest.mark.unit 
    def test_process_empty_input(self):
        """Test processing empty input."""
        result = process_user_input("", entries=[])
        
        assert isinstance(result, str)
        assert result == ""
    
    @pytest.mark.unit
    def test_process_multiline_input(self):
        """Test processing multiline text input."""
        input_text = """Line 1
        Line 2
        Line 3"""
        
        result = process_user_input(input_text, entries=[])
        
        assert isinstance(result, str)
        assert "Line 1" in result
        assert "Line 2" in result
        assert "Line 3" in result
    
    @pytest.mark.unit
    def test_process_input_with_special_characters(self):
        """Test processing input with special characters."""
        special_input = "Test with special chars: !@#$%^&*()[]{}\"'<>"
        
        result = process_user_input(special_input, entries=[])
        
        assert isinstance(result, str)
        assert result == special_input
    
    @pytest.mark.unit
    def test_process_json_like_input(self):
        """Test processing JSON-like string input."""
        json_input = '{"key": "value", "number": 123}'
        
        result = process_user_input(json_input, entries=[])
        
        assert isinstance(result, str)
        assert result == json_input

# ========================================================================
# Chat Content Update Tests
# ========================================================================

class TestUpdateChatContent:
    """Test the update_chat_content function."""
    
    @pytest.mark.unit
    def test_update_content_basic(self):
        """Test basic content update."""
        # Mock database
        mock_db = MagicMock()
        mock_db.get_note_by_id.return_value = {
            'content': '{"content": "Note content", "summary": "Note summary", "prompt": "Note prompt"}',
        }
        
        result, tags = update_chat_content(
            selected_item="Test Item",
            use_content=True,
            use_summary=False,
            use_prompt=False,
            item_mapping={"Test Item": "1"},
            db_instance=mock_db
        )
        
        assert isinstance(result, dict)
        assert isinstance(tags, list)
        assert 'content' in result
        mock_db.get_note_by_id.assert_called_once_with("1")
    
    @pytest.mark.unit
    def test_update_content_with_summary(self):
        """Test updating with summary."""
        mock_db = MagicMock()
        mock_db.get_note_by_id.return_value = {
            'content': '{"content": "Note content", "summary": "Note summary", "prompt": "Note prompt"}',
        }
        
        result, tags = update_chat_content(
            selected_item="Test Item",
            use_content=False,
            use_summary=True,
            use_prompt=False,
            item_mapping={"Test Item": "2"},
            db_instance=mock_db
        )
        
        assert 'summary' in result
        assert result['summary'] == 'Note summary'
    
    @pytest.mark.unit
    def test_update_content_no_selection(self):
        """Test updating with no item selected."""
        mock_db = MagicMock()
        
        result, tags = update_chat_content(
            selected_item=None,
            use_content=True,
            use_summary=False,
            use_prompt=False,
            item_mapping={},
            db_instance=mock_db
        )
        
        assert result == {}
        assert tags == []
        mock_db.get_note_by_id.assert_not_called()
    
    @pytest.mark.unit
    def test_update_content_all_options(self):
        """Test updating with all content options."""
        mock_db = MagicMock()
        mock_db.get_note_by_id.return_value = {
            'content': '{"content": "Note content", "summary": "Note summary", "prompt": "Note prompt"}',
            'keywords': 'tag1, tag2'
        }
        
        result, tags = update_chat_content(
            selected_item="Test Item",
            use_content=True,
            use_summary=True,
            use_prompt=True,
            item_mapping={"Test Item": "3"},
            db_instance=mock_db
        )
        
        assert 'content' in result
        assert 'summary' in result
        assert 'prompt' in result
        assert len(tags) > 0

# ========================================================================
# Provider Manager Integration Tests
# ========================================================================

# TestProviderManagement class removed - these functions don't exist:
# - get_provider_config
# - validate_provider

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
        error = ChatRateLimitError("Too many requests", provider="openai")
        
        assert "Too many requests" in str(error)
        assert error.provider == "openai"
    
    @pytest.mark.unit
    def test_auth_error_properties(self):
        """Test ChatAuthenticationError properties."""
        error = ChatAuthenticationError("Invalid credentials", provider="openai")
        
        assert "Invalid credentials" in str(error)
        assert error.provider == "openai"
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
# Token Counting Tests (Removed - function doesn't exist)
# ========================================================================
# The count_tokens function doesn't exist in the actual implementation.
# There is an approximate_token_count function that could be tested instead.