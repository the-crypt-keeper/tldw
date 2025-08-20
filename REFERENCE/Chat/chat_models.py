# chat_models.py
# Description: Data models for chat functionality including session management
#
# Imports
from dataclasses import dataclass, field
from typing import Optional, Dict, Any
from textual.worker import Worker
#
#######################################################################################################################
#
# Classes:

@dataclass
class ChatSessionData:
    """
    Holds the state for a single chat session/tab.
    
    This class encapsulates all the state needed to manage an independent
    chat conversation within a tab, including its database ID, streaming
    state, and UI references.
    """
    # Unique identifier for this tab (not the conversation ID)
    tab_id: str
    
    # Display title for the tab
    title: str = "New Chat"
    
    # Database conversation ID (None for ephemeral chats)
    conversation_id: Optional[str] = None
    
    # Whether this is an ephemeral (unsaved) chat
    is_ephemeral: bool = True
    
    # Character assignment for this chat
    character_id: Optional[int] = None
    character_name: Optional[str] = None
    
    # Streaming/worker state
    is_streaming: bool = False
    current_worker: Optional[Worker] = None
    current_ai_message_widget: Optional[Any] = None  # ChatMessage widget reference
    
    # Additional session state
    notes_content: str = ""
    message_count: int = 0
    has_unsaved_changes: bool = False
    
    # Settings that might vary per session
    system_prompt_override: Optional[str] = None
    temperature_override: Optional[float] = None
    max_tokens_override: Optional[int] = None
    
    # UI state
    is_active: bool = False
    
    # Metadata
    created_at: Optional[str] = None
    last_activity: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert session data to dictionary for serialization."""
        return {
            'tab_id': self.tab_id,
            'title': self.title,
            'conversation_id': self.conversation_id,
            'is_ephemeral': self.is_ephemeral,
            'character_id': self.character_id,
            'character_name': self.character_name,
            'notes_content': self.notes_content,
            'message_count': self.message_count,
            'system_prompt_override': self.system_prompt_override,
            'temperature_override': self.temperature_override,
            'max_tokens_override': self.max_tokens_override,
            'created_at': self.created_at,
            'last_activity': self.last_activity,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ChatSessionData':
        """Create session data from dictionary."""
        return cls(
            tab_id=data['tab_id'],
            title=data.get('title', 'New Chat'),
            conversation_id=data.get('conversation_id'),
            is_ephemeral=data.get('is_ephemeral', True),
            character_id=data.get('character_id'),
            character_name=data.get('character_name'),
            notes_content=data.get('notes_content', ''),
            message_count=data.get('message_count', 0),
            system_prompt_override=data.get('system_prompt_override'),
            temperature_override=data.get('temperature_override'),
            max_tokens_override=data.get('max_tokens_override'),
            created_at=data.get('created_at'),
            last_activity=data.get('last_activity'),
        )

#
# End of chat_models.py
#######################################################################################################################