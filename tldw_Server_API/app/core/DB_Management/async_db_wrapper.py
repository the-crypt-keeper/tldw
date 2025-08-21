# async_db_wrapper.py
# Description: Async wrapper for database operations to prevent blocking the event loop
#
# Imports
import asyncio
import functools
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Callable, Dict, List, Optional, TypeVar
from loguru import logger

from tldw_Server_API.app.core.DB_Management.ChaChaNotes_DB import CharactersRAGDB

#######################################################################################################################
#
# Type definitions:

T = TypeVar('T')

#######################################################################################################################
#
# Constants:

# Thread pool for database operations
DB_THREAD_POOL = ThreadPoolExecutor(max_workers=10, thread_name_prefix="db_worker")

#######################################################################################################################
#
# Functions:

class AsyncDatabaseWrapper:
    """
    Async wrapper for CharactersRAGDB to prevent blocking the event loop.
    All database operations are executed in a thread pool.
    """
    
    def __init__(self, db: CharactersRAGDB):
        """
        Initialize the async database wrapper.
        
        Args:
            db: The synchronous database instance to wrap
        """
        self.db = db
        self._executor = DB_THREAD_POOL
        
    async def add_conversation(self, conversation_data: Dict[str, Any]) -> Optional[str]:
        """
        Async version of add_conversation.
        
        Args:
            conversation_data: Conversation data dictionary
            
        Returns:
            Conversation ID if successful, None otherwise
        """
        return await asyncio.get_event_loop().run_in_executor(
            self._executor,
            self.db.add_conversation,
            conversation_data
        )
    
    async def get_conversation_by_id(self, conversation_id: str) -> Optional[Dict[str, Any]]:
        """
        Async version of get_conversation_by_id.
        
        Args:
            conversation_id: The conversation ID to retrieve
            
        Returns:
            Conversation data if found, None otherwise
        """
        return await asyncio.get_event_loop().run_in_executor(
            self._executor,
            self.db.get_conversation_by_id,
            conversation_id
        )
    
    async def add_message(self, message_data: Dict[str, Any]) -> Optional[str]:
        """
        Async version of add_message.
        
        Args:
            message_data: Message data dictionary
            
        Returns:
            Message ID if successful, None otherwise
        """
        return await asyncio.get_event_loop().run_in_executor(
            self._executor,
            self.db.add_message,
            message_data
        )
    
    async def get_messages_for_conversation(
        self, 
        conversation_id: str,
        limit: Optional[int] = None,
        offset: int = 0
    ) -> List[Dict[str, Any]]:
        """
        Async version of get_messages_for_conversation with pagination support.
        
        Args:
            conversation_id: The conversation ID
            limit: Maximum number of messages to retrieve
            offset: Number of messages to skip
            
        Returns:
            List of message dictionaries
        """
        # Use a wrapper function to handle pagination
        def get_paginated_messages():
            all_messages = self.db.get_messages_for_conversation(conversation_id)
            if limit:
                return all_messages[offset:offset + limit]
            return all_messages[offset:]
        
        return await asyncio.get_event_loop().run_in_executor(
            self._executor,
            get_paginated_messages
        )
    
    async def get_character_card_by_id(self, character_id: Any) -> Optional[Dict[str, Any]]:
        """
        Async version of get_character_card_by_id.
        
        Args:
            character_id: The character ID to retrieve
            
        Returns:
            Character card data if found, None otherwise
        """
        return await asyncio.get_event_loop().run_in_executor(
            self._executor,
            self.db.get_character_card_by_id,
            character_id
        )
    
    async def get_character_card_by_name(self, name: str) -> Optional[Dict[str, Any]]:
        """
        Async version of get_character_card_by_name.
        
        Args:
            name: The character name to search for
            
        Returns:
            Character card data if found, None otherwise
        """
        return await asyncio.get_event_loop().run_in_executor(
            self._executor,
            self.db.get_character_card_by_name,
            name
        )
    
    async def batch_add_messages(self, messages: List[Dict[str, Any]]) -> List[Optional[str]]:
        """
        Add multiple messages in a single batch operation.
        
        Args:
            messages: List of message data dictionaries
            
        Returns:
            List of message IDs (None for failed inserts)
        """
        def batch_insert():
            results = []
            with self.db.transaction():
                for msg in messages:
                    try:
                        msg_id = self.db.add_message(msg)
                        results.append(msg_id)
                    except Exception as e:
                        logger.error(f"Failed to insert message in batch: {e}")
                        results.append(None)
            return results
        
        return await asyncio.get_event_loop().run_in_executor(
            self._executor,
            batch_insert
        )
    
    def __getattr__(self, name: str) -> Any:
        """
        Fallback for any methods not explicitly wrapped.
        Automatically wraps them in async execution.
        
        Args:
            name: Method name to access
            
        Returns:
            Async wrapper for the method
        """
        attr = getattr(self.db, name)
        if callable(attr):
            @functools.wraps(attr)
            async def async_wrapper(*args, **kwargs):
                return await asyncio.get_event_loop().run_in_executor(
                    self._executor,
                    attr,
                    *args,
                    **kwargs
                )
            return async_wrapper
        return attr


def create_async_db(db: CharactersRAGDB) -> AsyncDatabaseWrapper:
    """
    Factory function to create an async database wrapper.
    
    Args:
        db: Synchronous database instance
        
    Returns:
        Async database wrapper
    """
    return AsyncDatabaseWrapper(db)


async def paginated_history_retrieval(
    db: AsyncDatabaseWrapper,
    conversation_id: str,
    page_size: int = 50,
    max_messages: Optional[int] = None
) -> List[Dict[str, Any]]:
    """
    Retrieve conversation history in pages to manage memory.
    
    Args:
        db: Async database wrapper
        conversation_id: Conversation to retrieve
        page_size: Number of messages per page
        max_messages: Maximum total messages to retrieve
        
    Returns:
        List of messages (most recent first)
    """
    messages = []
    offset = 0
    
    while True:
        # Get a page of messages
        page = await db.get_messages_for_conversation(
            conversation_id,
            limit=page_size,
            offset=offset
        )
        
        if not page:
            break
            
        messages.extend(page)
        offset += page_size
        
        # Check if we've reached the maximum
        if max_messages and len(messages) >= max_messages:
            messages = messages[:max_messages]
            break
        
        # Small delay to prevent overwhelming the database
        await asyncio.sleep(0.01)
    
    return messages


async def cleanup_old_conversations(
    db: AsyncDatabaseWrapper,
    days_to_keep: int = 30,
    batch_size: int = 100
) -> int:
    """
    Clean up old conversations for data retention compliance.
    
    Args:
        db: Async database wrapper
        days_to_keep: Number of days to retain conversations
        batch_size: Number of conversations to delete per batch
        
    Returns:
        Number of conversations deleted
    """
    import datetime
    
    cutoff_date = datetime.datetime.now(datetime.timezone.utc) - datetime.timedelta(days=days_to_keep)
    deleted_count = 0
    
    # This would need to be implemented in the base DB class
    # For now, we'll return a placeholder
    logger.info(f"Would clean up conversations older than {cutoff_date}")
    
    return deleted_count