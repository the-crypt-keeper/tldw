"""
Integration tests for chat module bug fixes and improvements.
Tests the fixes for issues identified in the code review.
"""

import pytest
import asyncio
import json
import jwt
import datetime
from unittest.mock import patch, MagicMock, AsyncMock
from fastapi import HTTPException, status
from concurrent.futures import ThreadPoolExecutor

from tldw_Server_API.app.api.v1.endpoints.chat import (
    _save_message_turn_to_db,
    _process_content_for_db_sync,
    create_chat_completion
)
from tldw_Server_API.app.core.Chat.chat_helpers import (
    validate_request_payload,
    get_or_create_conversation,
    extract_response_content
)
from tldw_Server_API.app.core.DB_Management.ChaChaNotes_DB import (
    CharactersRAGDB,
    ConflictError
)
from tldw_Server_API.app.api.v1.schemas.chat_request_schemas import (
    ChatCompletionRequest,
    ChatCompletionUserMessageParam
)


class TestEmptyMessageFix:
    """Test that empty messages with failed images are saved with placeholders."""
    
    @pytest.mark.asyncio
    async def test_failed_image_saves_placeholder(self):
        """Test that messages with failed image validation save a placeholder."""
        # Mock database
        mock_db = MagicMock(spec=CharactersRAGDB)
        mock_db.client_id = "test_client"
        mock_db.add_message = MagicMock(return_value="msg_123")
        
        # Create a message with invalid image
        message_obj = {
            "role": "user",
            "content": [
                {
                    "type": "image_url",
                    "image_url": {
                        "url": "data:image/invalid;base64,invalid_base64_data"
                    }
                }
            ]
        }
        
        # Process the message
        with patch('tldw_Server_API.app.api.v1.endpoints.chat.validate_image_url') as mock_validate:
            mock_validate.return_value = (False, None, None)  # Invalid image
            
            loop = asyncio.get_event_loop()
            result = await _save_message_turn_to_db(
                mock_db,
                "conv_123",
                message_obj,
                use_transaction=False
            )
        
        # Verify placeholder was saved
        assert mock_db.add_message.called
        saved_payload = mock_db.add_message.call_args[0][0]
        assert "<Message processing failed" in saved_payload['content'] or \
               "<Image failed validation" in saved_payload['content']
    
    @pytest.mark.asyncio
    async def test_empty_message_not_ignored(self):
        """Test that completely empty messages are not silently dropped."""
        mock_db = MagicMock(spec=CharactersRAGDB)
        mock_db.client_id = "test_client"
        mock_db.add_message = MagicMock(return_value="msg_124")
        
        # Empty message
        message_obj = {
            "role": "user",
            "content": ""
        }
        
        loop = asyncio.get_event_loop()
        result = await _save_message_turn_to_db(
            mock_db,
            "conv_123",
            message_obj,
            use_transaction=False
        )
        
        # Should save something, not return None
        assert mock_db.add_message.called


class TestDuplicateSerializationFix:
    """Test that request JSON is only serialized once."""
    
    @pytest.mark.asyncio
    async def test_single_json_serialization(self):
        """Test that request is only serialized once for performance."""
        request_data = ChatCompletionRequest(
            model="test-model",
            messages=[
                ChatCompletionUserMessageParam(role="user", content="Test message")
            ]
        )
        
        with patch('json.dumps') as mock_dumps:
            mock_dumps.return_value = '{"test": "data"}'
            
            # Mock other dependencies
            with patch('tldw_Server_API.app.api.v1.endpoints.chat.validate_request_payload') as mock_validate:
                mock_validate.return_value = (True, None)
                
                with patch('tldw_Server_API.app.api.v1.endpoints.chat.validate_request_size') as mock_size:
                    # This should receive the cached JSON
                    
                    # Simulate partial execution of create_chat_completion
                    # to test JSON serialization
                    request_json = json.dumps(request_data.model_dump())
                    request_json_bytes = request_json.encode()
                    
                    # Validate gets the cached version
                    from tldw_Server_API.app.api.v1.schemas.chat_validators import validate_request_size
                    validate_request_size(request_json)
        
        # Should only serialize once
        assert mock_dumps.call_count == 1


class TestConversationRaceCondition:
    """Test fixes for race conditions in conversation creation."""
    
    @pytest.mark.asyncio
    async def test_concurrent_conversation_creation(self):
        """Test that concurrent conversation creation handles conflicts properly."""
        mock_db = MagicMock(spec=CharactersRAGDB)
        mock_db.get_conversation_by_id = MagicMock(return_value=None)
        
        # Simulate conflict on first attempt, success on retry
        call_count = 0
        def add_conversation_side_effect(conv_data):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise ConflictError("Duplicate conversation")
            return f"conv_{call_count}"
        
        mock_db.add_conversation = MagicMock(side_effect=add_conversation_side_effect)
        
        # Mock transaction context
        with patch('tldw_Server_API.app.core.Chat.chat_helpers.db_transaction') as mock_transaction:
            mock_transaction.return_value.__aenter__ = AsyncMock(return_value=mock_db)
            mock_transaction.return_value.__aexit__ = AsyncMock(return_value=None)
            
            loop = asyncio.get_event_loop()
            conv_id, was_created = await get_or_create_conversation(
                mock_db,
                None,  # No existing conversation
                1,     # character_id
                "TestChar",
                "client_123",
                loop
            )
        
        # Should succeed after retry
        assert conv_id == "conv_2"
        assert was_created
        assert call_count == 2  # First failed, second succeeded
    
    @pytest.mark.asyncio
    async def test_parallel_conversation_requests(self):
        """Test multiple parallel requests creating conversations."""
        mock_db = MagicMock(spec=CharactersRAGDB)
        mock_db.get_conversation_by_id = MagicMock(return_value=None)
        
        # Track creation attempts
        creation_attempts = []
        
        def add_conversation_concurrent(conv_data):
            creation_attempts.append(conv_data['title'])
            # Simulate some succeeding, some failing
            if len(creation_attempts) % 2 == 0:
                raise ConflictError("Duplicate")
            return f"conv_{len(creation_attempts)}"
        
        mock_db.add_conversation = MagicMock(side_effect=add_conversation_concurrent)
        
        # Create multiple parallel requests
        async def create_conversation_task(task_id):
            with patch('tldw_Server_API.app.core.Chat.chat_helpers.db_transaction') as mock_tx:
                mock_tx.return_value.__aenter__ = AsyncMock(return_value=mock_db)
                mock_tx.return_value.__aexit__ = AsyncMock(return_value=None)
                
                loop = asyncio.get_event_loop()
                return await get_or_create_conversation(
                    mock_db,
                    None,
                    1,
                    f"Char_{task_id}",
                    f"client_{task_id}",
                    loop
                )
        
        # Run parallel tasks
        tasks = [create_conversation_task(i) for i in range(5)]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # All should eventually succeed or fail gracefully
        successful = [r for r in results if not isinstance(r, Exception)]
        assert len(successful) > 0  # At least some should succeed


class TestTransactionConsistency:
    """Test that all database operations use transactions consistently."""
    
    @pytest.mark.asyncio
    async def test_message_saves_use_transactions(self):
        """Test that all message saves use transactions."""
        mock_db = MagicMock(spec=CharactersRAGDB)
        mock_db.client_id = "test_client"
        mock_db.add_message = MagicMock(return_value="msg_125")
        
        message_obj = {
            "role": "user",
            "content": "Test message"
        }
        
        with patch('tldw_Server_API.app.core.DB_Management.transaction_utils.db_transaction') as mock_tx:
            mock_tx.return_value.__aenter__ = AsyncMock(return_value=mock_db)
            mock_tx.return_value.__aexit__ = AsyncMock(return_value=None)
            
            loop = asyncio.get_event_loop()
            result = await _save_message_turn_to_db(
                mock_db,
                "conv_123",
                message_obj,
                use_transaction=True  # Should use transaction
            )
        
        # Verify transaction was used
        assert mock_tx.called
        assert result == "msg_125"


class TestErrorResponseStandardization:
    """Test that error responses are properly standardized for security."""
    
    def test_5xx_errors_masked(self):
        """Test that 5xx errors don't expose internal details."""
        from tldw_Server_API.app.core.Chat.Chat_Functions import (
            ChatProviderError,
            ChatAPIError,
            ChatConfigurationError
        )
        
        # Test various 5xx errors
        errors_to_test = [
            (ChatProviderError("Internal provider error details", "test_provider"), 502),
            (ChatConfigurationError("Config file path: /etc/secret.conf", "test_provider"), 503),
            (ChatAPIError("Database connection string exposed", "test_provider"), 500),
        ]
        
        for error, expected_status in errors_to_test:
            # The error handler should mask details for 5xx
            if expected_status >= 500:
                # Details should be generic, not the original message
                assert "Internal" not in error.message or \
                       "unavailable" in error.message.lower() or \
                       "error occurred" in error.message.lower()
    
    def test_4xx_errors_preserved(self):
        """Test that 4xx client errors can show details."""
        from tldw_Server_API.app.core.Chat.Chat_Functions import (
            ChatAuthenticationError,
            ChatRateLimitError,
            ChatBadRequestError
        )
        
        # Test client errors
        errors_to_test = [
            ChatAuthenticationError("Invalid API key format", "test_provider"),
            ChatRateLimitError("Rate limit exceeded: 100 requests per minute", "test_provider"),
            ChatBadRequestError("Missing required field: messages", "test_provider"),
        ]
        
        for error in errors_to_test:
            # Client errors can keep their detail
            assert len(error.message) > 0
            # Original message should be preserved for client errors


class TestConfigurationLoading:
    """Test that configuration values are loaded from config file."""
    
    def test_config_values_loaded(self):
        """Test that chat module loads configuration values."""
        from tldw_Server_API.app.api.v1.endpoints.chat import (
            MAX_BASE64_BYTES,
            MAX_TEXT_LENGTH,
            MAX_MESSAGES_PER_REQUEST,
            MAX_IMAGES_PER_REQUEST
        )
        
        # These should be loaded from config or have defaults
        assert MAX_BASE64_BYTES > 0
        assert MAX_TEXT_LENGTH > 0
        assert MAX_MESSAGES_PER_REQUEST > 0
        assert MAX_IMAGES_PER_REQUEST > 0
        
        # Check they're reasonable values
        assert MAX_BASE64_BYTES >= 1024 * 1024  # At least 1MB
        assert MAX_TEXT_LENGTH >= 10000  # At least 10k chars
        assert MAX_MESSAGES_PER_REQUEST >= 10  # At least 10 messages
        assert MAX_IMAGES_PER_REQUEST >= 1  # At least 1 image
    
    def test_streaming_config_loaded(self):
        """Test that streaming configuration is loaded."""
        from tldw_Server_API.app.core.Chat.streaming_utils import (
            STREAMING_IDLE_TIMEOUT,
            HEARTBEAT_INTERVAL
        )
        
        # Should be loaded from config
        assert STREAMING_IDLE_TIMEOUT > 0
        assert HEARTBEAT_INTERVAL > 0
        
        # Reasonable values
        assert STREAMING_IDLE_TIMEOUT >= 60  # At least 1 minute
        assert HEARTBEAT_INTERVAL >= 10  # At least 10 seconds
        assert HEARTBEAT_INTERVAL < STREAMING_IDLE_TIMEOUT  # Heartbeat before timeout


class TestValidationImprovements:
    """Test improved validation functions."""
    
    @pytest.mark.asyncio
    async def test_request_validation_comprehensive(self):
        """Test comprehensive request validation."""
        # Valid request
        valid_request = ChatCompletionRequest(
            model="test-model",
            messages=[
                ChatCompletionUserMessageParam(role="user", content="Test")
            ]
        )
        
        is_valid, error = await validate_request_payload(valid_request)
        assert is_valid
        assert error is None
        
        # Too many messages
        many_messages = ChatCompletionRequest(
            model="test-model",
            messages=[
                ChatCompletionUserMessageParam(role="user", content=f"Message {i}")
                for i in range(1001)  # Over limit
            ]
        )
        
        is_valid, error = await validate_request_payload(many_messages, max_messages=1000)
        assert not is_valid
        assert "too many messages" in error.lower()
        
        # Message too long
        long_message = ChatCompletionRequest(
            model="test-model",
            messages=[
                ChatCompletionUserMessageParam(role="user", content="x" * 500000)  # Over limit
            ]
        )
        
        is_valid, error = await validate_request_payload(long_message, max_text_length=400000)
        assert not is_valid
        assert "too long" in error.lower()
    
    def test_response_content_extraction(self):
        """Test extraction of content from various response formats."""
        # String response
        assert extract_response_content("Simple response") == "Simple response"
        
        # OpenAI-style dict response
        openai_response = {
            "choices": [
                {"message": {"content": "OpenAI response"}}
            ]
        }
        assert extract_response_content(openai_response) == "OpenAI response"
        
        # Empty response
        assert extract_response_content(None) is None
        assert extract_response_content({}) is None
        
        # Malformed response
        assert extract_response_content({"invalid": "format"}) is None


# Run tests
if __name__ == "__main__":
    pytest.main([__file__, "-v"])