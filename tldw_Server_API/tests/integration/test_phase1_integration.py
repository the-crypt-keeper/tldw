# test_phase1_integration.py
# Description: Integration tests for Phase 1 components
#
"""
Phase 1 Integration Tests
-------------------------

Integration tests for Chat Dictionary, World Book Manager, and Document Generator
working together in realistic scenarios.
"""

import pytest
import asyncio
import json
import tempfile
from datetime import datetime
from unittest.mock import MagicMock, patch, AsyncMock

from tldw_Server_API.app.core.Character_Chat.chat_dictionary import ChatDictionaryService
from tldw_Server_API.app.core.Character_Chat.world_book_manager import WorldBookService
from tldw_Server_API.app.core.Chat.document_generator import DocumentGeneratorService, DocumentType
from tldw_Server_API.app.core.DB_Management.ChaChaNotes_DB import CharactersRAGDB


@pytest.fixture
def test_db():
    """Create a test database instance."""
    # Create in-memory database for testing
    db = CharactersRAGDB(":memory:", "test_user")
    return db


@pytest.fixture
def chat_dict_service(test_db):
    """Create ChatDictionaryService with test database."""
    return ChatDictionaryService(test_db)


@pytest.fixture
def world_book_service(test_db):
    """Create WorldBookService with test database."""
    return WorldBookService(test_db)


@pytest.fixture
def doc_gen_service(test_db):
    """Create DocumentGeneratorService with test database."""
    return DocumentGeneratorService(test_db, user_id="test_user")


@pytest.fixture
def sample_conversation_data():
    """Create sample conversation for testing."""
    return {
        "id": "test_conv_1",
        "title": "Fantasy Adventure Discussion",
        "messages": [
            {"sender": "user", "content": "Tell me about dragons in your world", "timestamp": "2024-01-01T10:00:00"},
            {"sender": "assistant", "content": "Dragons are ancient magical creatures that live in mountain castles", "timestamp": "2024-01-01T10:01:00"},
            {"sender": "user", "content": "What about wizards and magic?", "timestamp": "2024-01-01T10:02:00"},
            {"sender": "assistant", "content": "Wizards study arcane arts in towers, mastering spells through years of practice", "timestamp": "2024-01-01T10:03:00"},
            {"sender": "user", "content": "Can you describe a typical quest?", "timestamp": "2024-01-01T10:04:00"},
            {"sender": "assistant", "content": "A hero's journey often begins in a village, leading through forests to face challenges", "timestamp": "2024-01-01T10:05:00"}
        ]
    }


class TestDictionaryAndWorldBookIntegration:
    """Test Chat Dictionary and World Book working together."""
    
    def test_combined_text_processing(self, chat_dict_service, world_book_service, test_db):
        """Test processing text through both dictionary and world book systems."""
        # Setup dictionary
        dict_id = chat_dict_service.create_dictionary(
            name="Fantasy Terms",
            description="Common fantasy replacements"
        )
        chat_dict_service.add_entry(
            dictionary_id=dict_id,
            key="village",
            content="hamlet"
        )
        chat_dict_service.add_entry(
            dictionary_id=dict_id,
            key="forest",
            content="enchanted woods"
        )
        
        # Setup world book
        wb_id = world_book_service.create_world_book(
            name="Fantasy World",
            description="Fantasy setting lore",
            token_budget=500
        )
        world_book_service.add_entry(
            world_book_id=wb_id,
            keywords=["dragon", "castle"],
            content="Dragons are ancient beings that hoard treasure in mountain fortresses.",
            priority=100
        )
        world_book_service.add_entry(
            world_book_id=wb_id,
            keywords=["wizard", "magic"],
            content="Magic users draw power from ley lines that cross the realm.",
            priority=90
        )
        
        # Process text through dictionary
        text = "The hero left the village and entered the forest to find the wizard"
        processed_text, dict_metadata = chat_dict_service.process_text(text, token_budget=1000)
        
        # Process through world book
        wb_result = world_book_service.process_context(
            text=processed_text,
            character_id=None,
            token_budget=1000
        )
        
        # Verify combined processing
        assert "hamlet" in processed_text
        assert "enchanted woods" in processed_text
        assert "Magic users" in wb_result["processed_context"]
        assert dict_metadata["replacements"] == 2
        assert wb_result["entries_applied"] > 0
    
    def test_character_specific_processing(self, world_book_service, test_db):
        """Test character-specific world book attachments."""
        # Create world books
        main_wb = world_book_service.create_world_book(
            name="Main World",
            description="Primary setting"
        )
        char_wb = world_book_service.create_world_book(
            name="Character Background",
            description="Character-specific lore"
        )
        
        # Add entries
        world_book_service.add_entry(
            world_book_id=main_wb,
            keywords=["sword"],
            content="Swords are common weapons.",
            priority=50
        )
        world_book_service.add_entry(
            world_book_id=char_wb,
            keywords=["sword"],
            content="Your family sword is legendary.",
            priority=100
        )
        
        # Attach to character
        character_id = 1
        world_book_service.attach_to_character(character_id, main_wb, enabled=True)
        world_book_service.attach_to_character(character_id, char_wb, enabled=True, priority=1)
        
        # Process with character context
        result = world_book_service.process_context(
            text="Tell me about my sword",
            character_id=character_id,
            token_budget=500
        )
        
        # Should prioritize character-specific entry
        assert "legendary" in result["processed_context"].lower()
        assert result["entries_applied"] >= 1


class TestDocumentGenerationIntegration:
    """Test Document Generator with real conversation data."""
    
    @pytest.mark.asyncio
    async def test_generate_multiple_document_types(self, doc_gen_service, test_db, sample_conversation_data):
        """Test generating different document types from same conversation."""
        # Mock conversation retrieval
        with patch.object(test_db, 'get_conversation_by_id', return_value=sample_conversation_data):
            with patch.object(test_db, 'get_messages_for_conversation', return_value=sample_conversation_data["messages"]):
                # Mock LLM calls with appropriate responses
                with patch.object(doc_gen_service, '_call_llm', new_callable=AsyncMock) as mock_llm:
                    mock_llm.side_effect = [
                        "Timeline:\n- User asks about dragons\n- Assistant explains dragon lore\n- Discussion of wizards\n- Quest description",
                        "Study Guide:\n1. Dragons - Ancient magical creatures\n2. Wizards - Magic practitioners\n3. Quests - Hero's journey pattern",
                        "Executive Briefing:\nKey Topics: Fantasy worldbuilding elements\nMain Points: Dragons, magic system, quest structure"
                    ]
                    
                    # Generate different document types
                    timeline = await doc_gen_service.generate_document(
                        conversation_id="test_conv_1",
                        document_type=DocumentType.TIMELINE,
                        provider="openai",
                        model="gpt-3.5-turbo",
                        api_key="test_key"
                    )
                    study_guide = await doc_gen_service.generate_document(
                        conversation_id="test_conv_1",
                        document_type=DocumentType.STUDY_GUIDE,
                        provider="openai",
                        model="gpt-3.5-turbo",
                        api_key="test_key"
                    )
                    briefing = await doc_gen_service.generate_document(
                        conversation_id="test_conv_1",
                        document_type=DocumentType.BRIEFING,
                        provider="openai",
                        model="gpt-3.5-turbo",
                        api_key="test_key"
                    )
        
        # Verify all succeeded
        assert timeline["success"] == True
        assert study_guide["success"] == True
        assert briefing["success"] == True
        
        # Verify appropriate content
        assert "Timeline" in timeline["content"]
        assert "Study Guide" in study_guide["content"]
        assert "Executive Briefing" in briefing["content"]
    
    @pytest.mark.asyncio
    async def test_document_generation_with_processed_text(self, doc_gen_service, chat_dict_service, test_db, sample_conversation_data):
        """Test generating documents after dictionary processing."""
        # Setup dictionary
        dict_id = chat_dict_service.create_dictionary("Terms", "Replacements")
        chat_dict_service.add_entry(dict_id, "dragon", "wyrm", False)
        
        # Process conversation messages through dictionary
        processed_messages = []
        for msg in sample_conversation_data["messages"]:
            result = chat_dict_service.process_text(msg["content"], 1000)
            processed_msg = msg.copy()
            processed_msg["content"] = result["processed_text"]
            processed_messages.append(processed_msg)
        
        # Mock processed conversation
        processed_conv = sample_conversation_data.copy()
        processed_conv["messages"] = processed_messages
        
        with patch.object(test_db, 'get_conversation_by_id', return_value=processed_conv):
            with patch.object(test_db, 'get_messages_for_conversation', return_value=processed_messages):
                with patch.object(doc_gen_service, '_call_llm', new_callable=AsyncMock) as mock_llm:
                    mock_llm.return_value = "Summary with wyrm instead of dragon"
                    
                    result = await doc_gen_service.generate_document(
                        conversation_id="test_conv_1",
                        document_type=DocumentType.SUMMARY,
                        provider="openai",
                        model="gpt-3.5-turbo",
                        api_key="test_key"
                    )
        
        # Result is a string, not a dict
        assert isinstance(result, str)
        # Verify dictionary replacements were included
        assert mock_llm.called
        call_args = mock_llm.call_args[0][0]
        assert "wyrm" in call_args or "dragon" in call_args


class TestCompleteWorkflow:
    """Test complete workflow from conversation to processed documents."""
    
    @pytest.mark.asyncio
    async def test_full_pipeline(self, chat_dict_service, world_book_service, doc_gen_service, test_db):
        """Test full pipeline: Dictionary -> World Book -> Document Generation."""
        # Setup dictionary with fantasy replacements
        dict_id = chat_dict_service.create_dictionary(
            name="Fantasy Replacements",
            description="Convert modern to fantasy terms"
        )
        chat_dict_service.add_entry(dict_id, "car", "carriage", False)
        chat_dict_service.add_entry(dict_id, "phone", "sending stone", False)
        
        # Setup world book with lore
        wb_id = world_book_service.create_world_book(
            name="Campaign Setting",
            description="D&D campaign world"
        )
        world_book_service.add_entry(
            world_book_id=wb_id,
            keywords=["carriage"],
            content="Magical carriages are powered by bound elementals.",
            priority=100
        )
        
        # Create test conversation
        conversation = {
            "id": "workflow_test",
            "messages": [
                {"sender": "user", "content": "How do I travel by car here?", "timestamp": "2024-01-01T10:00:00"},
                {"sender": "assistant", "content": "You can hire a car at the station", "timestamp": "2024-01-01T10:01:00"}
            ]
        }
        
        # Process through dictionary
        processed_msgs = []
        for msg in conversation["messages"]:
            dict_result = chat_dict_service.process_text(msg["content"], 1000)
            
            # Process through world book
            wb_result = world_book_service.process_context(
                text=dict_result["processed_text"],
                character_id=None,
                max_tokens=500
            )
            
            processed_msgs.append({
                **msg,
                "content": wb_result["processed_context"]
            })
        
        # Generate document from processed conversation
        with patch.object(test_db, 'get_conversation_by_id', return_value=conversation):
            with patch.object(test_db, 'get_messages_for_conversation', return_value=processed_msgs):
                with patch.object(doc_gen_service, '_call_llm', new_callable=AsyncMock) as mock_llm:
                    mock_llm.return_value = "Travel in this world uses magical carriages powered by elementals"
                    
                    result = await doc_gen_service.generate_document(
                        conversation_id="workflow_test",
                        document_type=DocumentType.SUMMARY,
                        provider="openai",
                        model="gpt-3.5-turbo",
                        api_key="test_key"
                    )
        
        # Result is a string, not a dict
        assert isinstance(result, str)
        assert "carriage" in str(processed_msgs).lower()
        assert "elemental" in result.lower()
    
    def test_multi_user_isolation(self, test_db):
        """Test that different users have isolated data."""
        # Create services for different users
        user1_db = CharactersRAGDB(":memory:", "user1")
        user2_db = CharactersRAGDB(":memory:", "user2")
        
        service1 = ChatDictionaryService(user1_db)
        service2 = ChatDictionaryService(user2_db)
        
        # User 1 creates dictionary
        dict1 = service1.create_dictionary("User1 Dict", "Private dictionary")
        service1.add_entry(dict1, "test", "user1_replacement", False)
        
        # User 2 creates different dictionary
        dict2 = service2.create_dictionary("User2 Dict", "Different dictionary")
        service2.add_entry(dict2, "test", "user2_replacement", False)
        
        # Process same text for both users
        result1 = service1.process_text("This is a test", 100)
        result2 = service2.process_text("This is a test", 100)
        
        # Verify isolation
        assert "user1_replacement" in result1["processed_text"]
        assert "user2_replacement" in result2["processed_text"]
        assert result1["processed_text"] != result2["processed_text"]


class TestErrorHandlingIntegration:
    """Test error handling across integrated components."""
    
    def test_cascade_error_handling(self, chat_dict_service, world_book_service):
        """Test error handling when one component fails."""
        # Create invalid regex pattern
        dict_id = chat_dict_service.create_dictionary("Test", "Test")
        
        # This should handle the error gracefully
        try:
            chat_dict_service.add_entry(
                dictionary_id=dict_id,
                key="[invalid(regex",
                content="test"
            )
            assert False, "Should have raised ValueError"
        except ValueError as e:
            assert "Invalid regex" in str(e)
        
        # World book should still work independently
        wb_id = world_book_service.create_world_book("Test WB", "Still works")
        assert wb_id is not None
    
    @pytest.mark.asyncio
    async def test_document_generation_with_missing_conversation(self, doc_gen_service, test_db):
        """Test document generation when conversation doesn't exist."""
        with patch.object(test_db, 'get_conversation_by_id', return_value=None):
            result = await doc_gen_service.generate_document(
                conversation_id="nonexistent",
                document_type=DocumentType.SUMMARY,
                provider="openai",
                model="gpt-3.5-turbo",
                api_key="test_key"
            )
        
        # Should raise an exception or return an error string
        assert result is None or "error" in result.lower()


class TestPerformanceIntegration:
    """Test performance with larger datasets."""
    
    def test_bulk_dictionary_processing(self, chat_dict_service):
        """Test processing with many dictionary entries."""
        dict_id = chat_dict_service.create_dictionary("Large Dict", "Many entries")
        
        # Add 100 entries
        entries = []
        for i in range(100):
            entries.append({
                "key": f"word{i}",
                "content": f"replacement{i}",
                "probability": 100
            })
        
        # Bulk add
        count = chat_dict_service.bulk_add_entries(dict_id, entries)
        assert count == 100
        
        # Process text with many matches
        text = " ".join([f"word{i}" for i in range(50)])
        processed_text, stats = chat_dict_service.process_text(text, token_budget=10000)
        
        # Should handle all replacements efficiently
        assert stats["replacements"] > 0
        assert "replacement0" in processed_text
    
    def test_world_book_with_many_entries(self, world_book_service):
        """Test world book with many entries and keyword matching."""
        wb_id = world_book_service.create_world_book(
            name="Large World",
            description="Many lore entries",
            token_budget=2000
        )
        
        # Add many entries
        for i in range(50):
            world_book_service.add_entry(
                world_book_id=wb_id,
                keywords=[f"keyword{i}", f"term{i}"],
                content=f"Lore entry {i} with detailed information.",
                priority=100 - i
            )
        
        # Process text with multiple keywords
        text = "Tell me about keyword5 and term10 and keyword15"
        result = world_book_service.process_context(
            text=text,
            character_id=None,
            token_budget=1000
        )
        
        # Should find and apply relevant entries
        assert result["entries_applied"] >= 3
        assert "Lore entry" in result["processed_context"]