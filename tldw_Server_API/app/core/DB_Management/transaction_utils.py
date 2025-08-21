# transaction_utils.py
# Description: Database transaction utilities for ensuring ACID properties
#
# Imports
import asyncio
import functools
from contextlib import asynccontextmanager
from typing import Any, Callable, Optional, TypeVar
from loguru import logger

from tldw_Server_API.app.core.DB_Management.ChaChaNotes_DB import (
    CharactersRAGDB,
    CharactersRAGDBError,
    ConflictError,
    InputError
)

#######################################################################################################################
#
# Type definitions:

T = TypeVar('T')

#######################################################################################################################
#
# Functions:

@asynccontextmanager
async def db_transaction(db: CharactersRAGDB, max_retries: int = 3):
    """
    Async context manager for database transactions with automatic retry logic.
    
    Args:
        db: Database instance
        max_retries: Maximum number of retries for transient failures
        
    Yields:
        Transaction context
        
    Raises:
        CharactersRAGDBError: On database errors after retries exhausted
    """
    retry_count = 0
    last_error = None
    
    while retry_count < max_retries:
        try:
            # Start transaction
            with db.transaction():
                yield db
                # If we get here, transaction was successful
                return
                
        except ConflictError as e:
            # Conflict due to concurrent modification - retry
            retry_count += 1
            last_error = e
            if retry_count < max_retries:
                wait_time = 0.1 * (2 ** retry_count)  # Exponential backoff
                logger.warning(f"Transaction conflict, retrying in {wait_time}s (attempt {retry_count}/{max_retries})")
                await asyncio.sleep(wait_time)
            else:
                logger.error(f"Transaction failed after {max_retries} retries due to conflicts")
                raise
                
        except InputError as e:
            # Input validation error - don't retry
            logger.error(f"Transaction failed due to input error: {e}")
            raise
            
        except CharactersRAGDBError as e:
            # Other database errors - don't retry
            logger.error(f"Transaction failed due to database error: {e}")
            raise
            
        except Exception as e:
            # Unexpected error - log and re-raise
            logger.error(f"Unexpected error in transaction: {e}", exc_info=True)
            raise CharactersRAGDBError(f"Transaction failed: {str(e)}")
    
    # If we get here, all retries exhausted
    if last_error:
        raise last_error
    else:
        raise CharactersRAGDBError(f"Transaction failed after {max_retries} retries")


def transactional(max_retries: int = 3):
    """
    Decorator to make a function transactional with automatic retry logic.
    
    Args:
        max_retries: Maximum number of retries for transient failures
        
    Returns:
        Decorated function
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            # Try to find the db parameter
            db = None
            for arg in args:
                if isinstance(arg, CharactersRAGDB):
                    db = arg
                    break
            if not db:
                db = kwargs.get('db')
            
            if not db:
                # No database parameter found, run without transaction
                return await func(*args, **kwargs)
            
            async with db_transaction(db, max_retries):
                return await func(*args, **kwargs)
        
        return wrapper
    return decorator


async def save_conversation_with_messages(
    db: CharactersRAGDB,
    conversation_data: dict,
    messages: list[dict],
    max_retries: int = 3
) -> tuple[str, list[str]]:
    """
    Save a conversation and its messages in a single transaction.
    
    Args:
        db: Database instance
        conversation_data: Conversation data dictionary
        messages: List of message dictionaries
        max_retries: Maximum retry attempts
        
    Returns:
        Tuple of (conversation_id, list of message_ids)
        
    Raises:
        CharactersRAGDBError: On database errors
    """
    async with db_transaction(db, max_retries):
        # Create conversation
        conversation_id = db.add_conversation(conversation_data)
        if not conversation_id:
            raise CharactersRAGDBError("Failed to create conversation")
        
        # Add messages
        message_ids = []
        for message in messages:
            message['conversation_id'] = conversation_id
            message_id = db.add_message(message)
            if not message_id:
                raise CharactersRAGDBError(f"Failed to add message to conversation {conversation_id}")
            message_ids.append(message_id)
        
        return conversation_id, message_ids


async def update_conversation_with_rollback(
    db: CharactersRAGDB,
    conversation_id: str,
    updates: dict,
    new_messages: Optional[list[dict]] = None
) -> bool:
    """
    Update a conversation and optionally add new messages, with rollback on failure.
    
    Args:
        db: Database instance
        conversation_id: ID of the conversation to update
        updates: Dictionary of updates to apply
        new_messages: Optional list of new messages to add
        
    Returns:
        True if successful, False otherwise
    """
    try:
        async with db_transaction(db):
            # Update conversation
            if updates:
                success = db.update_conversation(conversation_id, updates)
                if not success:
                    raise CharactersRAGDBError(f"Failed to update conversation {conversation_id}")
            
            # Add new messages if provided
            if new_messages:
                for message in new_messages:
                    message['conversation_id'] = conversation_id
                    message_id = db.add_message(message)
                    if not message_id:
                        raise CharactersRAGDBError(f"Failed to add message to conversation {conversation_id}")
            
            return True
            
    except Exception as e:
        logger.error(f"Failed to update conversation {conversation_id}: {e}")
        return False


#
# End of transaction_utils.py
#######################################################################################################################