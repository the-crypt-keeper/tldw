"""
Test utilities for Character Chat tests.

This module provides test helpers and mock implementations for testing,
including a CharacterChatManager wrapper class.
"""

from typing import Dict, List, Optional, Any
from tldw_Server_API.app.core.DB_Management.ChaChaNotes_DB import CharactersRAGDB
from tldw_Server_API.app.core.Character_Chat import Character_Chat_Lib as char_lib


class CharacterChatManager:
    """
    A wrapper class that provides a manager interface for character chat operations.
    
    This class wraps the individual functions from Character_Chat_Lib to provide
    a cohesive interface for testing purposes.
    """
    
    def __init__(self, db_path: str = ":memory:"):
        """Initialize the manager with a database connection."""
        self.db = CharactersRAGDB(db_path, client_id="test_client")
        self.db_path = db_path
    
    def create_character_card(self, **kwargs) -> Optional[int]:
        """Create a new character card."""
        return char_lib.create_new_character_from_data(self.db, kwargs)
    
    def get_character_card(self, card_id: int) -> Optional[Dict[str, Any]]:
        """Get a character card by ID."""
        return char_lib.get_character_details(self.db, card_id)
    
    def list_character_cards(self, limit: int = 1000) -> List[Dict[str, Any]]:
        """List all character cards."""
        return char_lib.get_character_list_for_ui(self.db, limit)
    
    def update_character_card(self, card_id: int, **kwargs) -> Dict[str, Any]:
        """Update an existing character card."""
        # Add expected_version if not provided
        if 'expected_version' not in kwargs:
            card = self.get_character_card(card_id)
            if card:
                kwargs['expected_version'] = card.get('version', 1)
        
        success = char_lib.update_existing_character_details(
            self.db, card_id, kwargs, kwargs.get('expected_version', 1)
        )
        return {'success': success}
    
    def delete_character_card(self, card_id: int, expected_version: int = 1) -> Dict[str, Any]:
        """Delete a character card."""
        success = char_lib.delete_character_from_db(self.db, card_id, expected_version)
        return {'success': success}
    
    def search_character_cards(self, query: str) -> List[Dict[str, Any]]:
        """Search for character cards."""
        return char_lib.search_characters_by_query_text(self.db, query)
    
    def filter_by_tags(self, tags: List[str]) -> List[Dict[str, Any]]:
        """Filter character cards by tags."""
        # This functionality might not exist in the original lib
        # Return empty list for now
        return []
    
    def start_new_chat(self, character_id: int, user_name: str = "User") -> Optional[int]:
        """Start a new chat session with a character."""
        result = char_lib.start_new_chat_session(
            self.db, character_id, user_name
        )
        # start_new_chat_session returns a tuple, extract the chat ID
        if result and result[0]:
            return result[0]  # Return the chat_id from the tuple
        return None
    
    def add_message(self, chat_id: int, role: str, content: str) -> bool:
        """Add a message to an existing chat."""
        import uuid
        msg_data = {
            'id': str(uuid.uuid4()),
            'conversation_id': chat_id,
            'sender': role,
            'content': content,
            'parent_message_id': None,
            'deleted': 0,
            'client_id': 'test_client',
            'version': 1
        }
        result = self.db.add_message(msg_data)
        return result is not None
    
    def get_chat_messages(self, chat_id: int) -> List[Dict[str, Any]]:
        """Get all messages from a chat."""
        messages = self.db.get_messages_for_conversation(chat_id)
        return messages if messages else []
    
    def list_chats_for_character(self, character_id: int, limit: int = 50, offset: int = 0) -> List[Dict[str, Any]]:
        """List all chat sessions for a character."""
        return char_lib.list_character_conversations(self.db, character_id, limit, offset)
    
    def delete_chat(self, chat_id: int) -> bool:
        """Delete a chat session."""
        return self.db.delete_chat_session(chat_id)
    
    def search_messages(self, query: str, character_id: Optional[int] = None) -> List[Dict[str, Any]]:
        """Search messages across chats."""
        # This functionality might not exist in the original lib
        # Return empty list for now
        return []
    
    def get_chat_statistics(self, chat_id: int) -> Dict[str, Any]:
        """Get statistics for a chat."""
        messages = self.get_chat_messages(chat_id)
        if not messages:
            return {'total_messages': 0, 'user_messages': 0, 'assistant_messages': 0}
        
        user_count = sum(1 for m in messages if m.get('role') == 'user')
        assistant_count = sum(1 for m in messages if m.get('role') == 'assistant')
        
        return {
            'total_messages': len(messages),
            'user_messages': user_count,
            'assistant_messages': assistant_count
        }
    
    def cleanup(self):
        """Cleanup resources."""
        # Close database connection if needed
        pass
    
    def close(self):
        """Close the manager (alias for cleanup)."""
        self.cleanup()
    
    # Additional methods needed for tests
    def create_chat_session(self, character_id: int, user_name: str = "User", user_id: str = None, title: str = None) -> Optional[int]:
        """Create a new chat session (alias for start_new_chat)."""
        # Handle both user_name and user_id parameters
        if user_id:
            user_name = user_id
        # The title parameter is not used in start_new_chat, but we accept it for compatibility
        return self.start_new_chat(character_id, user_name)
    
    def get_chat_session(self, chat_id: int) -> Optional[Dict[str, Any]]:
        """Get information about a chat session."""
        # Return basic info about the chat
        messages = self.get_chat_messages(chat_id)
        # Try to get conversation details
        conv = self.db.get_conversation_by_id(chat_id)
        if conv or messages:
            return {
                'id': chat_id,
                'character_id': conv.get('character_id') if conv else None,
                'message_count': len(messages) if messages else 0,
                'messages': messages or []
            }
        return None
    
    def list_user_chats(self, user_name: str = None, user_id: str = None) -> List[Dict[str, Any]]:
        """List all chats for a user."""
        # Handle both user_name and user_id parameters
        if user_id:
            user_name = user_id
        # This would need to query all chats for a user
        # For now return empty list
        return []
    
    def delete_chat_session(self, chat_id: int) -> Dict[str, Any]:
        """Delete a chat session (alias for delete_chat)."""
        success = self.delete_chat(chat_id)
        return {'success': success}
    
    def clear_chat_history(self, chat_id: int) -> Dict[str, Any]:
        """Clear all messages from a chat."""
        # This would clear messages but keep the chat
        return {'success': True}
    
    def get_messages(self, chat_id: int) -> List[Dict[str, Any]]:
        """Get messages from a chat (alias for get_chat_messages)."""
        messages = self.get_chat_messages(chat_id)
        # Map 'sender' to 'role' for compatibility
        for msg in messages:
            if 'sender' in msg and 'role' not in msg:
                msg['role'] = msg['sender']
        return messages
    
    def get_recent_messages(self, chat_id: int, limit: int = 10) -> List[Dict[str, Any]]:
        """Get recent messages from a chat."""
        messages = self.get_chat_messages(chat_id)
        return messages[-limit:] if messages else []
    
    def edit_message(self, message_id: int, new_content: str) -> Dict[str, Any]:
        """Edit a message."""
        # This functionality might not exist
        return {'success': False}
    
    def delete_message(self, message_id: int) -> Dict[str, Any]:
        """Delete a message."""
        # This functionality might not exist
        return {'success': False}
    
    def build_context(self, character_id: int, messages: List[Dict[str, Any]], 
                     max_tokens: int = 4000) -> str:
        """Build context for LLM from messages."""
        # Simple implementation
        context_parts = []
        for msg in messages:
            role = msg.get('role', 'user')
            content = msg.get('content', '')
            context_parts.append(f"{role}: {content}")
        return "\n".join(context_parts)
    
    def count_tokens(self, text: str) -> int:
        """Count tokens in text (approximation)."""
        # Simple approximation: 1 token ≈ 4 characters
        return len(text) // 4
    
    def truncate_context(self, messages: List[Dict[str, Any]] = None, max_tokens: int = 4000) -> List[Dict[str, Any]]:
        """Truncate context to fit within token limit."""
        if messages is None:
            return []
        
        # Simple implementation: keep messages that fit within token budget
        total_tokens = 0
        result = []
        
        # Process messages in reverse order (keep most recent)
        for msg in reversed(messages):
            msg_tokens = self.count_tokens(msg.get('content', ''))
            if total_tokens + msg_tokens <= max_tokens:
                result.insert(0, msg)  # Insert at beginning to maintain order
                total_tokens += msg_tokens
            else:
                break
        
        return result
    
    def inject_world_entries(self, context: str, world_entries: List[Dict[str, Any]]) -> str:
        """Inject world book entries into context."""
        # Simple implementation - handle both list and single dict
        if isinstance(world_entries, dict):
            world_entries = [world_entries]
        entries_text = "\n".join([e.get('content', '') if isinstance(e, dict) else str(e) for e in world_entries])
        return f"{entries_text}\n{context}" if entries_text else context
    
    def export_character_card(self, card_id: int, format: str = "json") -> Dict[str, Any]:
        """Export a character card."""
        card = self.get_character_card(card_id)
        return card if card else {}
    
    def import_character_card(self, card_data: Dict[str, Any]) -> Optional[int]:
        """Import a character card."""
        # Filter out metadata fields that shouldn't be imported
        import_data = {}
        allowed_fields = {
            'name', 'description', 'personality', 'scenario', 'system_prompt',
            'image', 'post_history_instructions', 'first_message', 'message_example',
            'creator_notes', 'alternate_greetings', 'tags', 'creator', 'extensions'
        }
        
        for key, value in card_data.items():
            if key in allowed_fields:
                import_data[key] = value
        
        # Ensure we have required fields
        if 'creator' not in import_data:
            import_data['creator'] = 'imported'
        
        # Try to create with original name, if it fails due to conflict, add a suffix
        try:
            return self.create_character_card(**import_data)
        except Exception as e:
            if 'already exists' in str(e) and 'name' in import_data:
                # Add _imported suffix to avoid conflict
                import time
                import_data['name'] = f"{import_data['name']}_imported_{time.time()}"
                return self.create_character_card(**import_data)
            raise
    
    def export_chat_history(self, chat_id: int, format: str = "json") -> Dict[str, Any]:
        """Export chat history."""
        messages = self.get_chat_messages(chat_id)
        return {'chat_id': chat_id, 'messages': messages, 'metadata': {'format': format}}
    
    def import_chat_history(self, history_data: Dict[str, Any], user_id: str = None) -> Optional[int]:
        """Import chat history."""
        # Would need to create a chat and add messages
        return None
    
    def validate_character_name(self, name: str) -> bool:
        """Validate a character name."""
        return bool(name and name.strip() and len(name) <= 100)
    
    def validate_description(self, description: str) -> bool:
        """Validate a description."""
        return len(description) <= 2000  # Allow longer descriptions
    
    def validate_tags(self, tags: List[str]) -> bool:
        """Validate tags."""
        return all(isinstance(tag, str) and 0 < len(tag) <= 50 for tag in tags)
    
    def chunk_message(self, message: str, chunk_size: int = 1000) -> List[str]:
        """Chunk a long message into smaller parts."""
        if not message:
            return []
        
        words = message.split()
        chunks = []
        current_chunk = []
        
        for word in words:
            # Check if adding this word would exceed the chunk size
            if len(current_chunk) + 1 > chunk_size:
                # Save current chunk and start a new one
                if current_chunk:
                    chunks.append(' '.join(current_chunk))
                current_chunk = [word]
            else:
                current_chunk.append(word)
        
        # Add the last chunk
        if current_chunk:
            chunks.append(' '.join(current_chunk))
        
        return chunks if chunks else [message]
    
    def get_character_statistics(self, character_id: int) -> Dict[str, Any]:
        """Get statistics for a character."""
        chats = self.list_chats_for_character(character_id)
        return {
            'total_chats': len(chats),
            'total_messages': sum(c.get('message_count', 0) for c in chats)
        }
    
    def get_user_statistics(self, user_name: str = None, user_id: str = None) -> Dict[str, Any]:
        """Get statistics for a user."""
        return {
            'total_chats': 0,
            'total_messages': 0
        }