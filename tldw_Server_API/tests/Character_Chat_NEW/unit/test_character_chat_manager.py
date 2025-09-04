"""
Unit tests for CharacterChatManager.

Tests the core character chat functionality with minimal mocking -
only the database layer is mocked.
"""

import pytest
from unittest.mock import Mock, MagicMock, patch, call
from datetime import datetime
import json

from tldw_Server_API.tests.Character_Chat_NEW.test_utils import CharacterChatManager

# ========================================================================
# Character Card Management Tests
# ========================================================================

class TestCharacterCardManagement:
    """Test character card CRUD operations."""
    
    @pytest.mark.unit
    def test_create_character_card(self, mock_chat_manager, sample_character_card):
        """Test creating a character card."""
        manager = mock_chat_manager
        
        card_id = manager.create_character_card(**sample_character_card)
        
        assert card_id == 1
        manager.db.add_character_card.assert_called_once()
        # Note: add_character_card takes a prepared character dict, not kwargs
        call_args = manager.db.add_character_card.call_args[0]
        assert len(call_args) > 0  # Should have the character data dict
    
    @pytest.mark.unit
    def test_get_character_card(self, mock_chat_manager):
        """Test getting a character card."""
        manager = mock_chat_manager
        
        card = manager.get_character_card(card_id=1)
        
        assert card is not None
        assert card['id'] == 1
        assert card['name'] == 'Test Character'
        manager.db.get_character_card_by_id.assert_called_once_with(1)
    
    @pytest.mark.unit
    def test_list_character_cards(self, mock_chat_manager, sample_character_cards):
        """Test listing character cards."""
        manager = mock_chat_manager
        # The wrapper calls get_character_list_for_ui which we need to mock
        manager.db.get_character_list_for_ui = Mock(return_value=sample_character_cards)
        
        cards = manager.list_character_cards()
        
        assert len(cards) == len(sample_character_cards)
        assert cards[0]['name'] == 'Science Teacher'
        manager.db.get_character_list_for_ui.assert_called_once()
    
    @pytest.mark.unit
    def test_update_character_card(self, mock_chat_manager):
        """Test updating a character card."""
        manager = mock_chat_manager
        
        result = manager.update_character_card(
            card_id=1,
            name="Updated Character",
            description="Updated description"
        )
        
        assert result['success'] is True
        manager.db.update_character_card.assert_called_once()
    
    @pytest.mark.unit
    def test_delete_character_card(self, mock_chat_manager):
        """Test deleting a character card."""
        manager = mock_chat_manager
        
        result = manager.delete_character_card(card_id=1)
        
        assert result['success'] is True
        manager.db.delete_character_card.assert_called_once_with(1)
    
    @pytest.mark.unit
    def test_search_character_cards(self, mock_chat_manager):
        """Test searching character cards."""
        manager = mock_chat_manager
        manager.db.search_character_cards = Mock(return_value=[
            {'id': 1, 'name': 'Found Character'}
        ])
        
        results = manager.search_character_cards(query="test")
        
        assert len(results) == 1
        assert results[0]['name'] == 'Found Character'
    
    @pytest.mark.unit
    def test_filter_by_tags(self, mock_chat_manager):
        """Test filtering characters by tags."""
        manager = mock_chat_manager
        manager.db.filter_by_tags = Mock(return_value=[
            {'id': 1, 'name': 'Fantasy Character', 'tags': ['fantasy']}
        ])
        
        results = manager.filter_by_tags(['fantasy'])
        
        assert len(results) == 1
        assert 'fantasy' in results[0]['tags']

# ========================================================================
# Chat Session Management Tests
# ========================================================================

class TestChatSessionManagement:
    """Test chat session operations."""
    
    @pytest.mark.unit
    def test_create_chat_session(self, mock_chat_manager):
        """Test creating a new chat session."""
        manager = mock_chat_manager
        
        chat_id = manager.create_chat_session(
            character_id=1,
            user_id="test_user",
            title="New Chat"
        )
        
        assert chat_id == 1
        manager.db.create_chat.assert_called_once_with(
            character_id=1,
            user_id="test_user",
            title="New Chat"
        )
    
    @pytest.mark.unit
    def test_get_chat_session(self, mock_chat_manager):
        """Test getting a chat session."""
        manager = mock_chat_manager
        
        chat = manager.get_chat_session(chat_id=1)
        
        assert chat is not None
        assert chat['id'] == 1
        assert chat['title'] == 'Test Chat'
        manager.db.get_chat.assert_called_once_with(1)
    
    @pytest.mark.unit
    def test_list_user_chats(self, mock_chat_manager):
        """Test listing user's chat sessions."""
        manager = mock_chat_manager
        manager.db.list_user_chats = Mock(return_value=[
            {'id': 1, 'title': 'Chat 1'},
            {'id': 2, 'title': 'Chat 2'}
        ])
        
        chats = manager.list_user_chats(user_id="test_user")
        
        assert len(chats) == 2
        assert chats[0]['title'] == 'Chat 1'
    
    @pytest.mark.unit
    def test_delete_chat_session(self, mock_chat_manager):
        """Test deleting a chat session."""
        manager = mock_chat_manager
        manager.db.delete_chat = Mock(return_value={'success': True})
        
        result = manager.delete_chat_session(chat_id=1)
        
        assert result['success'] is True
        manager.db.delete_chat.assert_called_once_with(1)
    
    @pytest.mark.unit
    def test_clear_chat_history(self, mock_chat_manager):
        """Test clearing chat history."""
        manager = mock_chat_manager
        manager.db.clear_chat_messages = Mock(return_value={'success': True})
        
        result = manager.clear_chat_history(chat_id=1)
        
        assert result['success'] is True
        manager.db.clear_chat_messages.assert_called_once_with(1)

# ========================================================================
# Message Handling Tests
# ========================================================================

class TestMessageHandling:
    """Test message processing and storage."""
    
    @pytest.mark.unit
    def test_add_message(self, mock_chat_manager):
        """Test adding a message to chat."""
        manager = mock_chat_manager
        
        msg_id = manager.add_message(
            chat_id=1,
            role="user",
            content="Hello!"
        )
        
        assert msg_id == 1
        manager.db.add_message.assert_called_once_with(
            1, "user", "Hello!"
        )
    
    @pytest.mark.unit
    def test_get_messages(self, mock_chat_manager, sample_messages):
        """Test getting chat messages."""
        manager = mock_chat_manager
        manager.db.get_messages.return_value = sample_messages
        
        messages = manager.get_messages(chat_id=1)
        
        assert len(messages) == len(sample_messages)
        assert messages[0]['content'] == 'Hello!'
        manager.db.get_messages.assert_called_once_with(1)
    
    @pytest.mark.unit
    def test_get_recent_messages(self, mock_chat_manager, long_chat_history):
        """Test getting recent messages with limit."""
        manager = mock_chat_manager
        manager.db.get_messages.return_value = long_chat_history[-20:]
        
        messages = manager.get_recent_messages(chat_id=1, limit=10)
        
        assert len(messages) <= 10
        manager.db.get_messages.assert_called()
    
    @pytest.mark.unit
    def test_edit_message(self, mock_chat_manager):
        """Test editing a message."""
        manager = mock_chat_manager
        manager.db.update_message = Mock(return_value={'success': True})
        
        result = manager.edit_message(
            message_id=1,
            new_content="Updated content"
        )
        
        assert result['success'] is True
        manager.db.update_message.assert_called_once()
    
    @pytest.mark.unit
    def test_delete_message(self, mock_chat_manager):
        """Test deleting a message."""
        manager = mock_chat_manager
        manager.db.delete_message = Mock(return_value={'success': True})
        
        result = manager.delete_message(message_id=1)
        
        assert result['success'] is True
        manager.db.delete_message.assert_called_once_with(1)

# ========================================================================
# Context Management Tests
# ========================================================================

class TestContextManagement:
    """Test context building and management."""
    
    @pytest.mark.unit
    def test_build_context_basic(self, mock_chat_manager):
        """Test basic context building."""
        manager = mock_chat_manager
        
        # Mock character and messages
        manager.db.get_character_card.return_value = {
            'id': 1,
            'name': 'Test Character',
            'personality': 'Friendly',
            'scenario': 'Modern chat'
        }
        manager.db.get_messages.return_value = [
            {'role': 'user', 'content': 'Hello'},
            {'role': 'assistant', 'content': 'Hi there!'}
        ]
        
        context = manager.build_context(chat_id=1)
        
        assert 'character' in context
        assert 'messages' in context
        assert context['character']['name'] == 'Test Character'
        assert len(context['messages']) == 2
    
    @pytest.mark.unit
    def test_build_context_with_system_prompt(self, mock_chat_manager):
        """Test context building with system prompt."""
        manager = mock_chat_manager
        
        manager.db.get_character_card.return_value = {
            'id': 1,
            'name': 'Assistant',
            'personality': 'Helpful',
            'system_prompt': 'You are a helpful assistant.'
        }
        manager.db.get_messages.return_value = []
        
        context = manager.build_context(chat_id=1)
        
        assert 'system_prompt' in context
        assert context['system_prompt'] == 'You are a helpful assistant.'
    
    @pytest.mark.unit
    def test_truncate_context_by_tokens(self, mock_chat_manager, long_chat_history):
        """Test context truncation by token limit."""
        manager = mock_chat_manager
        manager.db.get_messages.return_value = long_chat_history
        
        # Mock token counting
        with patch.object(manager, 'count_tokens', side_effect=lambda x: len(x.split())):
            truncated = manager.truncate_context(
                messages=long_chat_history,
                max_tokens=100
            )
        
        assert len(truncated) < len(long_chat_history)
    
    @pytest.mark.unit
    def test_inject_world_book_entries(self, mock_chat_manager):
        """Test injecting world book entries into context."""
        manager = mock_chat_manager
        
        context = {
            'messages': [
                {'role': 'user', 'content': 'Tell me about dragons'}
            ]
        }
        
        world_entries = [
            'Dragons are ancient magical creatures.',
            'They can breathe fire.'
        ]
        
        updated_context = manager.inject_world_entries(context, world_entries)
        
        assert 'world_info' in updated_context
        assert len(updated_context['world_info']) == 2

# ========================================================================
# Import/Export Tests
# ========================================================================

class TestImportExport:
    """Test character import/export functionality."""
    
    @pytest.mark.unit
    def test_export_character_card(self, mock_chat_manager):
        """Test exporting a character card."""
        manager = mock_chat_manager
        
        manager.db.get_character_card.return_value = {
            'id': 1,
            'name': 'Test Character',
            'description': 'Test description',
            'personality': 'Test personality',
            'first_message': 'Hello!'
        }
        
        exported = manager.export_character_card(card_id=1)
        
        assert 'name' in exported
        assert exported['name'] == 'Test Character'
        assert 'spec' in exported or 'format' in exported
    
    @pytest.mark.unit
    def test_import_character_v3(self, mock_chat_manager, character_card_v3_format):
        """Test importing a V3 format character card."""
        manager = mock_chat_manager
        
        card_id = manager.import_character_card(character_card_v3_format)
        
        assert card_id == 1
        manager.db.create_character_card.assert_called_once()
        
        # Check mapping of V3 fields
        call_args = manager.db.create_character_card.call_args[1]
        assert call_args['name'] == 'Imported Character'
        assert call_args['first_message'] == 'Greetings from V3!'
    
    @pytest.mark.unit
    def test_export_chat_history(self, mock_chat_manager, sample_messages):
        """Test exporting chat history."""
        manager = mock_chat_manager
        manager.db.get_messages.return_value = sample_messages
        manager.db.get_chat.return_value = {
            'id': 1,
            'title': 'Exported Chat',
            'character_id': 1
        }
        
        exported = manager.export_chat_history(chat_id=1)
        
        assert 'messages' in exported
        assert len(exported['messages']) == len(sample_messages)
        assert 'metadata' in exported
    
    @pytest.mark.unit
    def test_import_chat_history(self, mock_chat_manager):
        """Test importing chat history."""
        manager = mock_chat_manager
        
        import_data = {
            'messages': [
                {'role': 'user', 'content': 'Imported message 1'},
                {'role': 'assistant', 'content': 'Imported response 1'}
            ],
            'metadata': {
                'character_id': 1,
                'title': 'Imported Chat'
            }
        }
        
        chat_id = manager.import_chat_history(import_data, user_id='test_user')
        
        assert chat_id == 1
        manager.db.create_chat.assert_called_once()
        assert manager.db.add_message.call_count == 2

# ========================================================================
# Character Validation Tests
# ========================================================================

class TestCharacterValidation:
    """Test character data validation."""
    
    @pytest.mark.unit
    def test_validate_character_name(self, mock_chat_manager):
        """Test character name validation."""
        manager = mock_chat_manager
        
        # Valid name
        assert manager.validate_character_name("Valid Name") is True
        
        # Invalid names
        assert manager.validate_character_name("") is False
        assert manager.validate_character_name("A" * 201) is False  # Too long
        assert manager.validate_character_name("   ") is False  # Only whitespace
    
    @pytest.mark.unit
    def test_validate_character_description(self, mock_chat_manager):
        """Test character description validation."""
        manager = mock_chat_manager
        
        # Valid description
        assert manager.validate_description("Valid description") is True
        
        # Edge cases
        assert manager.validate_description("") is True  # Empty allowed
        assert manager.validate_description("A" * 5000) is True  # Long but valid
        assert manager.validate_description("A" * 10001) is False  # Too long
    
    @pytest.mark.unit
    def test_validate_tags(self, mock_chat_manager):
        """Test tag validation."""
        manager = mock_chat_manager
        
        # Valid tags
        assert manager.validate_tags(['fantasy', 'roleplay']) is True
        assert manager.validate_tags([]) is True  # Empty is valid
        
        # Invalid tags
        assert manager.validate_tags(['']) is False  # Empty string tag
        assert manager.validate_tags(['a' * 51]) is False  # Tag too long
        assert manager.validate_tags(['valid'] * 21) is False  # Too many tags

# ========================================================================
# Statistics and Analytics Tests
# ========================================================================

class TestStatistics:
    """Test character statistics and analytics."""
    
    @pytest.mark.unit
    def test_get_character_statistics(self, mock_chat_manager):
        """Test getting character statistics."""
        manager = mock_chat_manager
        manager.db.get_character_statistics = Mock(return_value={
            'total_chats': 10,
            'total_messages': 100,
            'avg_chat_length': 10,
            'most_active_users': ['user1', 'user2']
        })
        
        stats = manager.get_character_statistics(character_id=1)
        
        assert stats['total_chats'] == 10
        assert stats['total_messages'] == 100
        assert 'user1' in stats['most_active_users']
    
    @pytest.mark.unit
    def test_get_user_statistics(self, mock_chat_manager):
        """Test getting user statistics."""
        manager = mock_chat_manager
        manager.db.get_user_statistics = Mock(return_value={
            'total_characters_used': 5,
            'total_messages': 50,
            'favorite_character': 'Test Character'
        })
        
        stats = manager.get_user_statistics(user_id='test_user')
        
        assert stats['total_characters_used'] == 5
        assert stats['favorite_character'] == 'Test Character'

# ========================================================================
# Error Handling Tests
# ========================================================================

class TestErrorHandling:
    """Test error handling in character chat manager."""
    
    @pytest.mark.unit
    def test_handle_database_error(self, mock_chat_manager):
        """Test handling of database errors."""
        manager = mock_chat_manager
        manager.db.create_character_card.side_effect = Exception("Database error")
        
        with pytest.raises(Exception) as exc_info:
            manager.create_character_card(
                name="Test",
                description="Test"
            )
        
        assert "Database error" in str(exc_info.value)
    
    @pytest.mark.unit
    def test_handle_not_found(self, mock_chat_manager):
        """Test handling of not found errors."""
        manager = mock_chat_manager
        manager.db.get_character_card.return_value = None
        
        result = manager.get_character_card(card_id=999)
        
        assert result is None
    
    @pytest.mark.unit
    def test_handle_invalid_import_format(self, mock_chat_manager):
        """Test handling invalid import format."""
        manager = mock_chat_manager
        
        invalid_data = {"invalid": "format"}
        
        with pytest.raises(ValueError) as exc_info:
            manager.import_character_card(invalid_data)
        
        assert "format" in str(exc_info.value).lower()

# ========================================================================
# Performance Tests
# ========================================================================

class TestPerformance:
    """Test performance-related functionality."""
    
    @pytest.mark.unit
    def test_batch_create_characters(self, mock_chat_manager, large_character_collection):
        """Test batch creation of characters."""
        manager = mock_chat_manager
        manager.db.create_character_card.side_effect = range(1, 101)
        
        created_ids = []
        for char_data in large_character_collection[:10]:  # Test with subset
            card_id = manager.create_character_card(**char_data)
            created_ids.append(card_id)
        
        assert len(created_ids) == 10
        assert manager.db.create_character_card.call_count == 10
    
    @pytest.mark.unit
    def test_cache_character_data(self, mock_chat_manager):
        """Test character data caching."""
        manager = mock_chat_manager
        
        # First call
        char1 = manager.get_character_card(card_id=1)
        
        # Second call should use cache (if implemented)
        char2 = manager.get_character_card(card_id=1)
        
        assert char1 == char2
        # In real implementation, would check cache hit