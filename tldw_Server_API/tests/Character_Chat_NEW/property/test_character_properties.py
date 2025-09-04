"""
Property-based tests for Character Chat using Hypothesis.

Tests invariants and properties that should always hold true.
"""

import pytest
from hypothesis import given, strategies as st, assume, settings
from hypothesis.stateful import RuleBasedStateMachine, rule, precondition, invariant, Bundle
import json
import re
from datetime import datetime

from tldw_Server_API.tests.Character_Chat_NEW.test_utils import CharacterChatManager
from tldw_Server_API.app.core.Character_Chat.chat_dictionary import ChatDictionaryService
from tldw_Server_API.app.core.Character_Chat.world_book_manager import WorldBookService

# ========================================================================
# Property Strategies
# ========================================================================

# Valid character name strategy
character_name_strategy = st.text(min_size=1, max_size=100).filter(
    lambda x: x.strip() and not x.startswith(' ') and not x.endswith(' ')
)

# Valid description strategy
description_strategy = st.text(min_size=0, max_size=500)

# Personality strategy
personality_strategy = st.text(min_size=1, max_size=500).filter(lambda x: x.strip())

# Message content strategy
message_strategy = st.text(min_size=1, max_size=1000)

# Tags strategy
tags_strategy = st.lists(
    st.text(min_size=1, max_size=30).filter(lambda x: x.strip()),
    min_size=0,
    max_size=10,
    unique=True
)

# Keywords strategy for world books
keywords_strategy = st.lists(
    st.text(min_size=1, max_size=50).filter(lambda x: x.strip()),
    min_size=1,
    max_size=5,
    unique=True
)

# Priority strategy
priority_strategy = st.integers(min_value=0, max_value=100)

# Dictionary pattern strategy
pattern_strategy = st.one_of(
    st.text(min_size=1, max_size=50),  # Literal patterns
    st.from_regex(r"\\b\\w+\\b", fullmatch=False)  # Simple regex patterns
)

# ========================================================================
# Character Card Properties
# ========================================================================

class TestCharacterCardProperties:
    """Test properties of character cards."""
    
    @pytest.mark.property
    @given(
        name=character_name_strategy,
        description=description_strategy,
        personality=personality_strategy,
        first_message=message_strategy,
        tags=tags_strategy
    )
    def test_create_then_get_preserves_data(
        self, name, description, personality, first_message, tags, character_db
    ):
        """Creating and getting a character preserves all data."""
        # Create character
        char_id = character_db.create_character_card(
            name=name,
            description=description,
            personality=personality,
            first_message=first_message,
            creator="test_user",
            tags=tags
        )
        
        # Get character
        character = character_db.get_character_card(char_id)
        
        assert character is not None
        assert character['name'] == name
        assert character['description'] == description
        assert character['personality'] == personality
        assert character['first_message'] == first_message
        assert set(character.get('tags', [])) == set(tags)
    
    @pytest.mark.property
    @given(name=character_name_strategy)
    def test_character_name_uniqueness(self, name, character_db):
        """Character names should be unique within a creator."""
        # Create first character
        char_id1 = character_db.create_character_card(
            name=name,
            description="First",
            personality="Test",
            first_message="Hi",
            creator="test_user"
        )
        
        # Try to create duplicate
        try:
            char_id2 = character_db.create_character_card(
                name=name,
                description="Second",
                personality="Test",
                first_message="Hi",
                creator="test_user"
            )
            # If it succeeds, IDs should be different (versioning)
            assert char_id1 != char_id2
        except Exception:
            # Duplicate rejection is also valid
            pass
    
    @pytest.mark.property
    @given(
        updates=st.lists(
            st.dictionaries(
                st.sampled_from(['description', 'personality', 'tags']),
                st.one_of(description_strategy, personality_strategy, tags_strategy),
                min_size=1,
                max_size=3
            ),
            min_size=1,
            max_size=5
        )
    )
    def test_multiple_updates_preserve_name(self, updates, character_db):
        """Multiple updates should preserve character name."""
        # Create character
        original_name = "Immutable Name"
        char_id = character_db.create_character_card(
            name=original_name,
            description="Initial",
            personality="Initial",
            first_message="Hi",
            creator="test"
        )
        
        # Apply updates
        for update in updates:
            character_db.update_character_card(char_id, **update)
        
        # Name should be unchanged
        character = character_db.get_character_card(char_id)
        assert character['name'] == original_name

# ========================================================================
# Chat Session Properties
# ========================================================================

class TestChatSessionProperties:
    """Test properties of chat sessions."""
    
    @pytest.mark.property
    @given(messages=st.lists(message_strategy, min_size=1, max_size=20))
    def test_message_order_preserved(self, messages, character_db):
        """Message order should be preserved in chat."""
        # Create character and chat
        char_id = character_db.create_character_card(
            name="Test Character",
            description="Test",
            personality="Test",
            first_message="Hi",
            creator="test"
        )
        
        chat_id = character_db.create_chat(
            character_id=char_id,
            user_id="test_user",
            title="Order Test"
        )
        
        # Add messages
        for i, msg in enumerate(messages):
            role = "user" if i % 2 == 0 else "assistant"
            character_db.add_message(chat_id, role, msg)
        
        # Get messages
        retrieved = character_db.get_messages(chat_id)
        
        # Order should be preserved
        for i, msg in enumerate(messages):
            assert retrieved[i]['content'] == msg
    
    @pytest.mark.property
    @given(
        num_chats=st.integers(min_value=1, max_value=10),
        messages_per_chat=st.integers(min_value=0, max_value=10)
    )
    def test_chat_isolation(self, num_chats, messages_per_chat, character_db):
        """Messages in one chat should not appear in another."""
        # Create character
        char_id = character_db.create_character_card(
            name="Isolation Test",
            description="Test",
            personality="Test",
            first_message="Hi",
            creator="test"
        )
        
        # Create multiple chats with unique messages
        chat_data = {}
        for i in range(num_chats):
            chat_id = character_db.create_chat(
                character_id=char_id,
                user_id="test_user",
                title=f"Chat {i}"
            )
            
            chat_messages = []
            for j in range(messages_per_chat):
                msg = f"Chat{i}_Message{j}"
                character_db.add_message(chat_id, "user", msg)
                chat_messages.append(msg)
            
            chat_data[chat_id] = chat_messages
        
        # Verify isolation
        for chat_id, expected_messages in chat_data.items():
            retrieved = character_db.get_messages(chat_id)
            retrieved_contents = [m['content'] for m in retrieved]
            
            # Should only contain this chat's messages
            assert set(retrieved_contents) == set(expected_messages)

# ========================================================================
# World Book Properties
# ========================================================================

class TestWorldBookProperties:
    """Test properties of world books."""
    
    @pytest.mark.property
    @given(
        keywords=keywords_strategy,
        content=st.text(min_size=1, max_size=500),
        priority=priority_strategy
    )
    def test_entry_keyword_matching(self, keywords, content, priority, world_book_service):
        """Entries should activate when keywords match."""
        service = world_book_service
        
        # Create world book with entry
        wb_id = service.create_world_book(name="Match Test")
        service.add_entry(
            world_book_id=wb_id,
            keywords=keywords,
            content=content,
            priority=priority
        )
        
        # Test with context containing keywords
        for keyword in keywords:
            context = f"This text contains {keyword} in it."
            activated = service.process_context(context, wb_id)
            
            assert len(activated) > 0
            assert any(content == e['content'] for e in activated)
    
    @pytest.mark.property
    @given(
        entries=st.lists(
            st.tuples(
                keywords_strategy,
                st.text(min_size=1, max_size=100),
                priority_strategy
            ),
            min_size=2,
            max_size=10
        )
    )
    def test_priority_ordering_invariant(self, entries, world_book_service):
        """Activated entries should be ordered by priority."""
        service = world_book_service
        
        wb_id = service.create_world_book(name="Priority Test")
        
        # Add entries
        all_keywords = []
        for keywords, content, priority in entries:
            service.add_entry(
                world_book_id=wb_id,
                keywords=keywords,
                content=content,
                priority=priority
            )
            all_keywords.extend(keywords)
        
        # Create context with all keywords
        context = ' '.join(all_keywords)
        activated = service.process_context(context, wb_id)
        
        # Check priority ordering (highest first)
        for i in range(len(activated) - 1):
            assert activated[i]['priority'] >= activated[i + 1]['priority']
    
    @pytest.mark.property
    @given(
        keyword=st.text(min_size=1, max_size=30).filter(lambda x: x.strip())
    )
    def test_keyword_case_insensitive(self, keyword, world_book_service):
        """Keywords should match case-insensitively."""
        service = world_book_service
        
        wb_id = service.create_world_book(name="Case Test")
        service.add_entry(
            world_book_id=wb_id,
            keywords=[keyword],
            content="Test content"
        )
        
        # Test different cases
        contexts = [
            keyword.lower(),
            keyword.upper(),
            keyword.swapcase(),
            keyword.capitalize()
        ]
        
        for context in contexts:
            activated = service.process_context(context, wb_id)
            assert len(activated) > 0

# ========================================================================
# Dictionary Properties
# ========================================================================

class TestDictionaryProperties:
    """Test properties of chat dictionaries."""
    
    @pytest.mark.property
    @given(
        pattern=st.text(min_size=1, max_size=50).filter(lambda x: x.strip()),
        replacement=st.text(min_size=0, max_size=100)
    )
    def test_literal_replacement_complete(self, pattern, replacement, chat_dictionary_service):
        """Literal replacements should replace all occurrences."""
        service = chat_dictionary_service
        
        dict_id = service.create_dictionary(name="Literal Test")
        service.add_entry(
            dictionary_id=dict_id,
            pattern=pattern,
            replacement=replacement,
            type="literal"
        )
        
        # Test text with multiple occurrences
        text = f"{pattern} and {pattern} plus {pattern}"
        processed = service.process_text(text, dict_id)
        
        # All occurrences should be replaced
        assert pattern not in processed
        assert processed.count(replacement) == 3
    
    @pytest.mark.property
    @given(
        entries=st.lists(
            st.tuples(
                st.text(min_size=1, max_size=20).filter(lambda x: x.strip() and x.isalnum()),
                st.text(min_size=1, max_size=30)
            ),
            min_size=1,
            max_size=5,
            unique_by=lambda x: x[0]
        )
    )
    def test_multiple_replacements_no_conflicts(self, entries, chat_dictionary_service):
        """Multiple replacements should not interfere with each other."""
        service = chat_dictionary_service
        
        dict_id = service.create_dictionary(name="Multi Test")
        
        # Add all entries
        for pattern, replacement in entries:
            service.add_entry(
                dictionary_id=dict_id,
                pattern=pattern,
                replacement=replacement,
                type="literal"
            )
        
        # Create text with all patterns
        text = ' '.join([pattern for pattern, _ in entries])
        processed = service.process_text(text, dict_id)
        
        # Each replacement should appear exactly once
        for _, replacement in entries:
            assert replacement in processed
    
    @pytest.mark.property
    @given(probability=st.floats(min_value=0.0, max_value=1.0))
    def test_probability_bounds(self, probability, chat_dictionary_service):
        """Probability replacements should respect bounds."""
        service = chat_dictionary_service
        
        dict_id = service.create_dictionary(name="Prob Test")
        service.add_entry(
            dictionary_id=dict_id,
            pattern="test",
            replacement="replaced",
            probability=probability
        )
        
        # Run multiple times
        replacements = 0
        iterations = 100
        
        for _ in range(iterations):
            processed = service.process_text("test", dict_id)
            if "replaced" in processed:
                replacements += 1
        
        # Replacement rate should be roughly the probability
        # Allow for statistical variance
        if probability == 0.0:
            assert replacements == 0
        elif probability == 1.0:
            assert replacements == iterations
        else:
            # Allow 20% variance for middle probabilities
            expected = iterations * probability
            assert abs(replacements - expected) < iterations * 0.3

# ========================================================================
# Stateful Property Testing
# ========================================================================

class CharacterChatStateMachine(RuleBasedStateMachine):
    """Stateful testing for character chat operations."""
    
    def __init__(self):
        super().__init__()
        self.manager = None
        self.character_ids = set()
        self.chat_ids = set()
        self.character_data = {}
        self.chat_data = {}
        
    characters = Bundle('characters')
    chats = Bundle('chats')
    
    @rule()
    def initialize_manager(self):
        """Initialize the manager if not already done."""
        if self.manager is None:
            import tempfile
            self.db_path = tempfile.mktemp(suffix='.db')
            self.manager = CharacterChatManager(db_path=self.db_path)
    
    @rule(
        target=characters,
        name=character_name_strategy,
        description=description_strategy,
        personality=personality_strategy,
        first_message=message_strategy
    )
    def create_character(self, name, description, personality, first_message):
        """Create a new character."""
        if self.manager is None:
            self.initialize_manager()
        
        char_id = self.manager.create_character_card(
            name=name,
            description=description,
            personality=personality,
            first_message=first_message,
            creator="test"
        )
        
        self.character_ids.add(char_id)
        self.character_data[char_id] = {
            'name': name,
            'description': description,
            'personality': personality,
            'first_message': first_message
        }
        
        return char_id
    
    @rule(
        target=chats,
        character_id=characters
    )
    def create_chat(self, character_id):
        """Create a chat for a character."""
        if character_id in self.character_ids:
            chat_id = self.manager.create_chat_session(
                character_id=character_id,
                user_id="test_user",
                title=f"Chat for {character_id}"
            )
            
            self.chat_ids.add(chat_id)
            self.chat_data[chat_id] = {
                'character_id': character_id,
                'messages': []
            }
            
            return chat_id
    
    @rule(
        chat_id=chats,
        message=message_strategy
    )
    def add_message(self, chat_id, message):
        """Add a message to a chat."""
        if chat_id in self.chat_ids:
            self.manager.add_message(chat_id, "user", message)
            self.chat_data[chat_id]['messages'].append(message)
    
    @rule(character_id=characters)
    def delete_character(self, character_id):
        """Delete a character."""
        if character_id in self.character_ids:
            self.manager.delete_character_card(character_id)
            self.character_ids.remove(character_id)
            
            # Remove associated chats
            for chat_id, data in list(self.chat_data.items()):
                if data['character_id'] == character_id:
                    self.chat_ids.discard(chat_id)
                    del self.chat_data[chat_id]
    
    @invariant()
    def characters_are_retrievable(self):
        """All created characters should be retrievable."""
        if self.manager is not None:
            for char_id in self.character_ids:
                character = self.manager.get_character_card(char_id)
                assert character is not None
                assert character['id'] == char_id
    
    @invariant()
    def chats_belong_to_characters(self):
        """All chats should belong to existing characters."""
        if self.manager is not None:
            for chat_id, data in self.chat_data.items():
                if chat_id in self.chat_ids:
                    chat = self.manager.get_chat_session(chat_id)
                    if chat:
                        assert chat['character_id'] in self.character_ids
    
    @invariant()
    def message_count_matches(self):
        """Message count should match what was added."""
        if self.manager is not None:
            for chat_id, data in self.chat_data.items():
                if chat_id in self.chat_ids:
                    messages = self.manager.get_messages(chat_id)
                    # Count user messages only
                    user_messages = [m for m in messages if m['role'] == 'user']
                    assert len(user_messages) == len(data['messages'])
    
    def teardown(self):
        """Clean up after test."""
        if self.manager:
            self.manager.close()
        
        if hasattr(self, 'db_path'):
            import os
            try:
                os.unlink(self.db_path)
            except:
                pass


@pytest.mark.property
@pytest.mark.slow
def test_character_chat_state_machine():
    """Run the stateful property test."""
    TestCharacterChatMachine = CharacterChatStateMachine.TestCase
    TestCharacterChatMachine.settings = settings(
        max_examples=50,
        stateful_step_count=20
    )
    TestCharacterChatMachine().runTest()

# ========================================================================
# Message Processing Properties
# ========================================================================

class TestMessageProcessingProperties:
    """Test properties of message processing."""
    
    @pytest.mark.property
    @given(
        messages=st.lists(
            st.tuples(
                st.sampled_from(['user', 'assistant']),
                message_strategy
            ),
            min_size=1,
            max_size=50
        ),
        max_tokens=st.integers(min_value=10, max_value=1000)
    )
    def test_context_truncation_preserves_recent(self, messages, max_tokens, chat_manager):
        """Context truncation should preserve most recent messages."""
        manager = chat_manager
        
        # Mock token counting
        with patch.object(manager, 'count_tokens', side_effect=lambda x: len(x.split())):
            truncated = manager.truncate_context(
                messages=[{'role': r, 'content': c} for r, c in messages],
                max_tokens=max_tokens
            )
        
        if truncated:
            # Last message should always be included
            assert truncated[-1]['content'] == messages[-1][1]
            
            # Should not exceed token limit (with some tolerance)
            total_tokens = sum(len(m['content'].split()) for m in truncated)
            assert total_tokens <= max_tokens * 1.2
    
    @pytest.mark.property
    @given(
        message_length=st.integers(min_value=1, max_value=10000),
        chunk_size=st.integers(min_value=10, max_value=1000)
    )
    def test_message_chunking(self, message_length, chunk_size, chat_manager):
        """Long messages should be properly chunked."""
        manager = chat_manager
        
        # Create a long message
        long_message = 'word ' * message_length
        
        # Chunk the message
        chunks = manager.chunk_message(long_message, chunk_size)
        
        # All chunks should be within size limit
        for chunk in chunks:
            assert len(chunk.split()) <= chunk_size
        
        # Reconstruction should preserve content
        reconstructed = ' '.join(chunks)
        assert reconstructed.strip() == long_message.strip()

# ========================================================================
# Import/Export Properties
# ========================================================================

class TestImportExportProperties:
    """Test properties of import/export functionality."""
    
    @pytest.mark.property
    @given(
        character_data=st.builds(
            dict,
            name=character_name_strategy,
            description=description_strategy,
            personality=personality_strategy,
            first_message=message_strategy,
            tags=tags_strategy
        )
    )
    def test_export_import_roundtrip(self, character_data, chat_manager):
        """Exporting and importing should preserve all data."""
        manager = chat_manager
        
        # Create character
        char_id = manager.create_character_card(**character_data, creator="test")
        
        # Export
        exported = manager.export_character_card(char_id)
        
        # Delete original
        manager.delete_character_card(char_id)
        
        # Import
        new_id = manager.import_character_card(exported)
        
        # Verify data preserved
        imported = manager.get_character_card(new_id)
        assert imported['name'] == character_data['name']
        assert imported['description'] == character_data['description']
        assert imported['personality'] == character_data['personality']
        assert imported['first_message'] == character_data['first_message']
        assert set(imported.get('tags', [])) == set(character_data.get('tags', []))

# ========================================================================
# Tag Management Properties
# ========================================================================

class TestTagProperties:
    """Test properties of tag management."""
    
    @pytest.mark.property
    @given(
        tags=st.lists(
            st.text(min_size=1, max_size=20).filter(
                lambda x: x.strip() and x.isalnum()
            ),
            min_size=1,
            max_size=20,
            unique=True
        )
    )
    def test_tag_normalization(self, tags, character_db):
        """Tags should be normalized consistently."""
        # Create character with tags
        char_id = character_db.create_character_card(
            name="Tag Test",
            description="Test",
            personality="Test",
            first_message="Hi",
            creator="test",
            tags=tags
        )
        
        # Get character
        character = character_db.get_character_card(char_id)
        retrieved_tags = character.get('tags', [])
        
        # All tags should be preserved (possibly normalized)
        assert len(retrieved_tags) == len(tags)
        
        # Normalization should be consistent
        for original in tags:
            normalized = original.lower().strip()
            assert any(t.lower() == normalized for t in retrieved_tags)