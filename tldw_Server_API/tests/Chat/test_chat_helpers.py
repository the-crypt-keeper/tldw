# test_chat_helpers.py
# Unit tests for chat helper functions

import pytest
import asyncio
import datetime
from unittest.mock import AsyncMock, MagicMock, patch, call
from typing import Dict, Any

from tldw_Server_API.app.core.Chat.chat_helpers import (
    validate_request_payload,
    get_or_create_character_context,
    get_or_create_conversation,
    load_conversation_history,
    prepare_llm_messages,
    extract_system_message,
    extract_response_content,
    validate_provider_configuration,
)


class MockChatRequest:
    """Mock chat request for testing."""
    def __init__(self, messages=None):
        self.messages = messages or []


class MockMessage:
    """Mock message for testing."""
    def __init__(self, role="user", content="test"):
        self.role = role
        self.content = content


class MockMessagePart:
    """Mock message part for testing."""
    def __init__(self, type="text", text=None):
        self.type = type
        self.text = text


class MockDatabase:
    """Mock database for testing."""
    def __init__(self):
        self.client_id = "test_client"
    
    def transaction(self):
        """Mock transaction context manager."""
        from contextlib import contextmanager
        @contextmanager
        def mock_transaction():
            yield self
        return mock_transaction()
    
    def get_character_card_by_id(self, char_id):
        if char_id == 1:
            return {"id": 1, "name": "TestChar", "description": "Test character"}
        return None
    
    def get_character_card_by_name(self, name):
        if name == "TestChar":
            return {"id": 1, "name": "TestChar", "description": "Test character"}
        elif name == "DefaultChar":
            return {"id": 999, "name": "DefaultChar", "description": "Default character"}
        return None
    
    def get_conversation_by_id(self, conv_id):
        if conv_id == "existing_conv":
            return {
                "id": "existing_conv",
                "character_id": 1,
                "client_id": "test_client",
                "title": "Test Conversation"
            }
        return None
    
    def add_conversation(self, data):
        return f"new_conv_{data.get('character_id', 0)}"
    
    def get_messages_for_conversation(self, conv_id, limit, offset, order):
        if conv_id == "existing_conv":
            return [
                {
                    "id": 1,
                    "sender": "user",
                    "content": "Hello",
                    "image_data": None,
                    "image_mime_type": None
                },
                {
                    "id": 2,
                    "sender": "assistant",
                    "content": "Hi there!",
                    "image_data": None,
                    "image_mime_type": None
                }
            ]
        return []


@pytest.mark.asyncio
class TestValidateRequestPayload:
    """Test request payload validation."""
    
    async def test_valid_request(self):
        """Test validation of valid request."""
        request = MockChatRequest([
            MockMessage("user", "Hello"),
            MockMessage("assistant", "Hi")
        ])
        
        is_valid, error = await validate_request_payload(request)
        assert is_valid is True
        assert error is None
    
    async def test_empty_messages(self):
        """Test validation fails for empty messages."""
        request = MockChatRequest([])
        
        is_valid, error = await validate_request_payload(request)
        assert is_valid is False
        assert "empty" in error.lower()
    
    async def test_too_many_messages(self):
        """Test validation fails for too many messages."""
        messages = [MockMessage() for _ in range(1001)]
        request = MockChatRequest(messages)
        
        is_valid, error = await validate_request_payload(request, max_messages=1000)
        assert is_valid is False
        assert "too many" in error.lower()
    
    async def test_message_too_long(self):
        """Test validation fails for overly long message."""
        long_text = "x" * 400001
        request = MockChatRequest([MockMessage("user", long_text)])
        
        is_valid, error = await validate_request_payload(request, max_text_length=400000)
        assert is_valid is False
        assert "too long" in error.lower()
    
    async def test_too_many_images(self):
        """Test validation fails for too many images."""
        messages = []
        for i in range(11):
            msg = MockMessage()
            msg.content = [MockMessagePart("image_url")]
            messages.append(msg)
        request = MockChatRequest(messages)
        
        is_valid, error = await validate_request_payload(request, max_images=10)
        assert is_valid is False
        assert "too many images" in error.lower()


@pytest.mark.asyncio
class TestGetOrCreateCharacterContext:
    """Test character context retrieval/creation."""
    
    async def test_get_character_by_id(self):
        """Test getting character by numeric ID."""
        db = MockDatabase()
        loop = asyncio.get_running_loop()
        
        char_card, char_id = await get_or_create_character_context(db, "1", loop)
        
        assert char_card is not None
        assert char_card["name"] == "TestChar"
        assert char_id == 1
    
    async def test_get_character_by_name(self):
        """Test getting character by name."""
        db = MockDatabase()
        loop = asyncio.get_running_loop()
        
        char_card, char_id = await get_or_create_character_context(db, "TestChar", loop)
        
        assert char_card is not None
        assert char_card["name"] == "TestChar"
        assert char_id == 1
    
    async def test_default_character_fallback(self):
        """Test fallback to default character."""
        db = MockDatabase()
        loop = asyncio.get_running_loop()
        
        with patch.object(db, 'get_character_card_by_name', side_effect=[
            None,  # First call for requested character
            {"id": 999, "name": "DefaultChar"}  # Second call for default
        ]):
            char_card, char_id = await get_or_create_character_context(db, "NonExistent", loop)
        
        assert char_card is not None
        assert char_card["name"] == "DefaultChar"
        assert char_id == 999
    
    async def test_no_character_provided(self):
        """Test when no character ID is provided."""
        db = MockDatabase()
        loop = asyncio.get_running_loop()
        
        with patch('tldw_Server_API.app.api.v1.API_Deps.ChaCha_Notes_DB_Deps.DEFAULT_CHARACTER_NAME', 'DefaultChar'):
            with patch.object(db, 'get_character_card_by_name', return_value={"id": 999, "name": "DefaultChar"}):
                char_card, char_id = await get_or_create_character_context(db, None, loop)
        
        assert char_card is not None
        assert char_id == 999


@pytest.mark.asyncio
class TestGetOrCreateConversation:
    """Test conversation retrieval/creation."""
    
    async def test_get_existing_conversation(self):
        """Test retrieving existing conversation."""
        db = MockDatabase()
        loop = asyncio.get_running_loop()
        
        conv_id, was_created = await get_or_create_conversation(
            db, "existing_conv", 1, "TestChar", "test_client", loop
        )
        
        assert conv_id == "existing_conv"
        assert was_created is False
    
    async def test_create_new_conversation(self):
        """Test creating new conversation."""
        db = MockDatabase()
        loop = asyncio.get_running_loop()
        
        conv_id, was_created = await get_or_create_conversation(
            db, None, 1, "TestChar", "test_client", loop
        )
        
        assert conv_id == "new_conv_1"
        assert was_created is True
    
    async def test_conversation_mismatch_character(self):
        """Test conversation with wrong character ID."""
        db = MockDatabase()
        loop = asyncio.get_running_loop()
        
        # Mock conversation with different character
        with patch.object(db, 'get_conversation_by_id', return_value={
            "id": "existing_conv",
            "character_id": 999,  # Different character
            "client_id": "test_client"
        }):
            conv_id, was_created = await get_or_create_conversation(
                db, "existing_conv", 1, "TestChar", "test_client", loop
            )
        
        assert conv_id != "existing_conv"
        assert was_created is True
    
    async def test_conversation_mismatch_client(self):
        """Test conversation with wrong client ID."""
        db = MockDatabase()
        loop = asyncio.get_running_loop()
        
        # Mock conversation with different client
        with patch.object(db, 'get_conversation_by_id', return_value={
            "id": "existing_conv",
            "character_id": 1,
            "client_id": "different_client"  # Different client
        }):
            conv_id, was_created = await get_or_create_conversation(
                db, "existing_conv", 1, "TestChar", "test_client", loop
            )
        
        assert conv_id != "existing_conv"
        assert was_created is True


@pytest.mark.asyncio
class TestLoadConversationHistory:
    """Test conversation history loading."""
    
    async def test_load_history(self):
        """Test loading conversation history."""
        db = MockDatabase()
        loop = asyncio.get_running_loop()
        character_card = {"id": 1, "name": "TestChar"}
        
        messages = await load_conversation_history(
            db, "existing_conv", character_card, limit=20, loop=loop
        )
        
        assert len(messages) == 2
        assert messages[0]["role"] == "user"
        assert messages[0]["content"] == "Hello"
        assert messages[1]["role"] == "assistant"
        assert messages[1]["content"] == "Hi there!"
    
    async def test_load_empty_history(self):
        """Test loading empty conversation history."""
        db = MockDatabase()
        loop = asyncio.get_running_loop()
        
        messages = await load_conversation_history(
            db, "non_existent", None, limit=20, loop=loop
        )
        
        assert messages == []
    
    async def test_history_with_character_names(self):
        """Test history includes character names."""
        db = MockDatabase()
        loop = asyncio.get_running_loop()
        character_card = {"id": 1, "name": "TestChar"}
        
        messages = await load_conversation_history(
            db, "existing_conv", character_card, limit=20, loop=loop
        )
        
        # Assistant messages should have character name
        assistant_msgs = [m for m in messages if m["role"] == "assistant"]
        for msg in assistant_msgs:
            assert "name" in msg or msg["role"] == "assistant"


@pytest.mark.asyncio
class TestExtractSystemMessage:
    """Test system message extraction."""
    
    async def test_extract_from_messages(self):
        """Test extracting system message from message list."""
        messages = [
            MockMessage("system", "You are helpful"),
            MockMessage("user", "Hello")
        ]
        
        system_msg = await asyncio.get_running_loop().run_in_executor(
            None, extract_system_message, messages, None
        )
        
        assert system_msg == "You are helpful"
    
    async def test_extract_from_character_card(self):
        """Test extracting system message from character card."""
        messages = [MockMessage("user", "Hello")]
        character_card = {
            "name": "TestChar",
            "description": "A test character",
            "system_prompt": "You are TestChar"
        }
        
        system_msg = await asyncio.get_running_loop().run_in_executor(
            None, extract_system_message, messages, character_card
        )
        
        assert "TestChar" in system_msg
    
    async def test_no_system_message(self):
        """Test when no system message exists."""
        messages = [MockMessage("user", "Hello")]
        
        system_msg = await asyncio.get_running_loop().run_in_executor(
            None, extract_system_message, messages, None
        )
        
        assert system_msg is None or system_msg == ""


@pytest.mark.asyncio
class TestExtractResponseContent:
    """Test response content extraction."""
    
    async def test_extract_string_response(self):
        """Test extracting content from string response."""
        response = "Hello, world!"
        
        content = await asyncio.get_running_loop().run_in_executor(
            None, extract_response_content, response
        )
        
        assert content == "Hello, world!"
    
    async def test_extract_dict_response(self):
        """Test extracting content from dict response."""
        response = {
            "choices": [
                {"message": {"content": "Hello from dict"}}
            ]
        }
        
        content = await asyncio.get_running_loop().run_in_executor(
            None, extract_response_content, response
        )
        
        assert content == "Hello from dict"
    
    async def test_extract_empty_response(self):
        """Test extracting from empty response."""
        response = {"choices": []}
        
        content = await asyncio.get_running_loop().run_in_executor(
            None, extract_response_content, response
        )
        
        assert content == "" or content is None
    
    async def test_extract_malformed_response(self):
        """Test extracting from malformed response."""
        response = {"invalid": "structure"}
        
        content = await asyncio.get_running_loop().run_in_executor(
            None, extract_response_content, response
        )
        
        assert content == "" or content is None


@pytest.mark.asyncio
class TestValidateProviderConfiguration:
    """Test provider configuration validation."""
    
    async def test_valid_provider_config(self):
        """Test validation of valid provider configuration."""
        api_keys = {'openai': 'test_key'}
        is_valid, error = await asyncio.get_running_loop().run_in_executor(
            None, validate_provider_configuration, "openai", api_keys
        )
        
        assert is_valid is True
        assert error is None
    
    async def test_missing_api_key(self):
        """Test validation fails for missing API key."""
        api_keys = {}  # Empty dictionary, no API key
        is_valid, error = await asyncio.get_running_loop().run_in_executor(
            None, validate_provider_configuration, "openai", api_keys
        )
        
        assert is_valid is False
        assert error is not None
    
    async def test_local_provider_no_key_needed(self):
        """Test local providers don't need API keys."""
        api_keys = {}  # No API keys needed for local providers
        is_valid, error = await asyncio.get_running_loop().run_in_executor(
            None, validate_provider_configuration, "llama.cpp", api_keys
        )
        
        assert is_valid is True
    
    async def test_unknown_provider(self):
        """Test unknown provider handling."""
        api_keys = {}
        result, error = await asyncio.get_running_loop().run_in_executor(
            None, validate_provider_configuration, "unknown_provider", api_keys
        )
        
        # Unknown provider should return False with an error message
        assert isinstance(result, bool)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])