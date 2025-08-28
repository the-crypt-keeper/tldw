"""
Character chat operations module.

This module contains functions for managing chat sessions and messages.
"""

import json
import time
from typing import Dict, List, Optional, Tuple, Any

from loguru import logger

from tldw_Server_API.app.core.DB_Management.ChaChaNotes_DB import CharactersRAGDB, CharactersRAGDBError, ConflictError, InputError


def process_db_messages_to_ui_history(
    messages: List[Dict[str, Any]]
) -> List[Tuple[Optional[str], Optional[str]]]:
    """Convert database messages to UI history format.
    
    Args:
        messages: List of message dictionaries from the database
        
    Returns:
        List of tuples (user_message, bot_message) for UI display
    """
    history = []
    current_user_msg = None
    
    for msg in messages:
        if msg.get('role') == 'user':
            current_user_msg = msg.get('content')
        elif msg.get('role') in ['assistant', 'character']:
            bot_msg = msg.get('content')
            history.append((current_user_msg, bot_msg))
            current_user_msg = None
    
    # Handle trailing user message without response
    if current_user_msg:
        history.append((current_user_msg, None))
    
    return history


def load_chat_and_character(
    db: CharactersRAGDB,
    conversation_id: str
) -> Tuple[Optional[Dict[str, Any]], Optional[Dict[str, Any]], Optional[List[Dict[str, Any]]]]:
    """Load a chat conversation along with its associated character.
    
    Args:
        db: Database instance
        conversation_id: The conversation ID to load
        
    Returns:
        Tuple of (character_data, conversation_metadata, messages)
    """
    try:
        # Get conversation metadata
        conversation = db.get_conversation_by_id(conversation_id)
        if not conversation:
            logger.warning(f"Conversation {conversation_id} not found")
            return None, None, None
        
        # Get character data
        character_id = conversation.get('character_id')
        if character_id:
            character = db.get_character_card_by_id(character_id)
        else:
            character = None
            logger.warning(f"No character_id in conversation {conversation_id}")
        
        # Get messages
        messages = db.get_messages_for_conversation(conversation_id)
        
        return character, conversation, messages
    except CharactersRAGDBError as e:
        logger.error(f"Database error loading chat {conversation_id}: {e}")
        return None, None, None
    except Exception as e:
        logger.error(f"Unexpected error loading chat {conversation_id}: {e}", exc_info=True)
        return None, None, None


def start_new_chat_session(
    db: CharactersRAGDB,
    character_id: int,
    title: Optional[str] = None,
    metadata: Optional[Dict[str, Any]] = None
) -> Optional[str]:
    """Start a new chat session with a character.
    
    Args:
        db: Database instance
        character_id: The character to chat with
        title: Optional title for the conversation
        metadata: Optional metadata for the conversation
        
    Returns:
        The new conversation ID, or None on error
    """
    try:
        if not title:
            title = f"Chat - {time.strftime('%Y-%m-%d %H:%M:%S')}"
        
        conversation_id = db.create_conversation(
            character_id=character_id,
            title=title,
            metadata=metadata or {}
        )
        
        logger.info(f"Started new chat session {conversation_id} with character {character_id}")
        return conversation_id
    except CharactersRAGDBError as e:
        logger.error(f"Database error starting chat session: {e}")
        return None
    except Exception as e:
        logger.error(f"Unexpected error starting chat session: {e}", exc_info=True)
        return None


def list_character_conversations(
    db: CharactersRAGDB,
    character_id: int,
    limit: int = 50,
    offset: int = 0
) -> List[Dict[str, Any]]:
    """List conversations for a character.
    
    Args:
        db: Database instance
        character_id: The character ID
        limit: Maximum number of results
        offset: Offset for pagination
        
    Returns:
        List of conversation metadata dictionaries
    """
    try:
        conversations = db.get_conversations_for_character(character_id, limit, offset)
        return conversations
    except CharactersRAGDBError as e:
        logger.error(f"Database error listing conversations for character {character_id}: {e}")
        return []
    except Exception as e:
        logger.error(f"Unexpected error listing conversations: {e}", exc_info=True)
        return []


def get_conversation_metadata(
    db: CharactersRAGDB,
    conversation_id: str
) -> Optional[Dict[str, Any]]:
    """Get metadata for a conversation.
    
    Args:
        db: Database instance
        conversation_id: The conversation ID
        
    Returns:
        Conversation metadata dictionary, or None if not found
    """
    try:
        conversation = db.get_conversation_by_id(conversation_id)
        return conversation
    except CharactersRAGDBError as e:
        logger.error(f"Database error getting conversation {conversation_id}: {e}")
        return None
    except Exception as e:
        logger.error(f"Unexpected error getting conversation: {e}", exc_info=True)
        return None


def update_conversation_metadata(
    db: CharactersRAGDB,
    conversation_id: str,
    update_data: Dict[str, Any],
    expected_version: int
) -> bool:
    """Update conversation metadata with version checking.
    
    Args:
        db: Database instance
        conversation_id: The conversation ID
        update_data: Data to update
        expected_version: Expected version for optimistic locking
        
    Returns:
        True if successful, False otherwise
    """
    try:
        success = db.update_conversation(conversation_id, update_data, expected_version)
        if success:
            logger.info(f"Updated conversation {conversation_id}")
        else:
            logger.warning(f"Failed to update conversation {conversation_id} - version mismatch")
        return success
    except CharactersRAGDBError as e:
        logger.error(f"Database error updating conversation {conversation_id}: {e}")
        return False
    except Exception as e:
        logger.error(f"Unexpected error updating conversation: {e}", exc_info=True)
        return False


def delete_conversation_by_id(
    db: CharactersRAGDB,
    conversation_id: str,
    expected_version: int
) -> bool:
    """Delete a conversation.
    
    Args:
        db: Database instance
        conversation_id: The conversation ID
        expected_version: Expected version for optimistic locking
        
    Returns:
        True if successful, False otherwise
    """
    try:
        success = db.delete_conversation(conversation_id, expected_version)
        if success:
            logger.info(f"Deleted conversation {conversation_id}")
        else:
            logger.warning(f"Failed to delete conversation {conversation_id} - version mismatch")
        return success
    except CharactersRAGDBError as e:
        logger.error(f"Database error deleting conversation {conversation_id}: {e}")
        return False
    except Exception as e:
        logger.error(f"Unexpected error deleting conversation: {e}", exc_info=True)
        return False


def search_conversations_by_title_query(
    db: CharactersRAGDB,
    title_query: str,
    character_id: Optional[int] = None,
    limit: int = 10
) -> List[Dict[str, Any]]:
    """Search conversations by title.
    
    Args:
        db: Database instance
        title_query: Search query for title
        character_id: Optional character ID to filter by
        limit: Maximum number of results
        
    Returns:
        List of matching conversation metadata dictionaries
    """
    try:
        conversations = db.search_conversations(title_query, character_id, limit)
        return conversations
    except CharactersRAGDBError as e:
        logger.error(f"Database error searching conversations: {e}")
        return []
    except Exception as e:
        logger.error(f"Unexpected error searching conversations: {e}", exc_info=True)
        return []


def post_message_to_conversation(
    db: CharactersRAGDB,
    conversation_id: str,
    role: str,
    content: str,
    metadata: Optional[Dict[str, Any]] = None
) -> Optional[str]:
    """Post a new message to a conversation.
    
    Args:
        db: Database instance
        conversation_id: The conversation ID
        role: Message role ('user', 'assistant', 'character')
        content: Message content
        metadata: Optional message metadata
        
    Returns:
        The new message ID, or None on error
    """
    try:
        message_id = db.add_message_to_conversation(
            conversation_id=conversation_id,
            role=role,
            content=content,
            metadata=metadata or {}
        )
        logger.info(f"Added message {message_id} to conversation {conversation_id}")
        return message_id
    except CharactersRAGDBError as e:
        logger.error(f"Database error posting message: {e}")
        return None
    except Exception as e:
        logger.error(f"Unexpected error posting message: {e}", exc_info=True)
        return None


def retrieve_message_details(
    db: CharactersRAGDB,
    message_id: str
) -> Optional[Dict[str, Any]]:
    """Retrieve details for a specific message.
    
    Args:
        db: Database instance
        message_id: The message ID
        
    Returns:
        Message data dictionary, or None if not found
    """
    try:
        message = db.get_message_by_id(message_id)
        return message
    except CharactersRAGDBError as e:
        logger.error(f"Database error retrieving message {message_id}: {e}")
        return None
    except Exception as e:
        logger.error(f"Unexpected error retrieving message: {e}", exc_info=True)
        return None


def retrieve_conversation_messages_for_ui(
    db: CharactersRAGDB,
    conversation_id: str,
    limit: int = 100,
    offset: int = 0
) -> List[Dict[str, Any]]:
    """Retrieve messages for a conversation formatted for UI.
    
    Args:
        db: Database instance
        conversation_id: The conversation ID
        limit: Maximum number of messages
        offset: Offset for pagination
        
    Returns:
        List of message dictionaries
    """
    try:
        messages = db.get_messages_for_conversation(conversation_id, limit, offset)
        return messages
    except CharactersRAGDBError as e:
        logger.error(f"Database error retrieving messages for conversation {conversation_id}: {e}")
        return []
    except Exception as e:
        logger.error(f"Unexpected error retrieving messages: {e}", exc_info=True)
        return []


def edit_message_content(
    db: CharactersRAGDB,
    message_id: str,
    new_content: str,
    expected_version: int
) -> bool:
    """Edit the content of an existing message.
    
    Args:
        db: Database instance
        message_id: The message ID
        new_content: New message content
        expected_version: Expected version for optimistic locking
        
    Returns:
        True if successful, False otherwise
    """
    try:
        success = db.update_message(message_id, {'content': new_content}, expected_version)
        if success:
            logger.info(f"Edited message {message_id}")
        else:
            logger.warning(f"Failed to edit message {message_id} - version mismatch")
        return success
    except CharactersRAGDBError as e:
        logger.error(f"Database error editing message {message_id}: {e}")
        return False
    except Exception as e:
        logger.error(f"Unexpected error editing message: {e}", exc_info=True)
        return False


def set_message_ranking(
    db: CharactersRAGDB,
    message_id: str,
    ranking: int
) -> bool:
    """Set the ranking for a message.
    
    Args:
        db: Database instance
        message_id: The message ID
        ranking: The ranking value
        
    Returns:
        True if successful, False otherwise
    """
    try:
        success = db.update_message(message_id, {'ranking': ranking})
        if success:
            logger.info(f"Set ranking {ranking} for message {message_id}")
        else:
            logger.warning(f"Failed to set ranking for message {message_id}")
        return success
    except CharactersRAGDBError as e:
        logger.error(f"Database error setting message ranking: {e}")
        return False
    except Exception as e:
        logger.error(f"Unexpected error setting message ranking: {e}", exc_info=True)
        return False


def remove_message_from_conversation(
    db: CharactersRAGDB,
    message_id: str,
    expected_version: int
) -> bool:
    """Remove a message from a conversation.
    
    Args:
        db: Database instance
        message_id: The message ID
        expected_version: Expected version for optimistic locking
        
    Returns:
        True if successful, False otherwise
    """
    try:
        success = db.delete_message(message_id, expected_version)
        if success:
            logger.info(f"Removed message {message_id}")
        else:
            logger.warning(f"Failed to remove message {message_id} - version mismatch")
        return success
    except CharactersRAGDBError as e:
        logger.error(f"Database error removing message: {e}")
        return False
    except Exception as e:
        logger.error(f"Unexpected error removing message: {e}", exc_info=True)
        return False


def find_messages_in_conversation(
    db: CharactersRAGDB,
    conversation_id: str,
    search_query: str,
    limit: int = 10
) -> List[Dict[str, Any]]:
    """Search for messages within a conversation.
    
    Args:
        db: Database instance
        conversation_id: The conversation ID
        search_query: Search query for message content
        limit: Maximum number of results
        
    Returns:
        List of matching message dictionaries
    """
    try:
        messages = db.search_messages_in_conversation(conversation_id, search_query, limit)
        return messages
    except CharactersRAGDBError as e:
        logger.error(f"Database error searching messages: {e}")
        return []
    except Exception as e:
        logger.error(f"Unexpected error searching messages: {e}", exc_info=True)
        return []