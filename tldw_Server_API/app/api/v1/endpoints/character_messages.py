# character_messages.py
"""
API endpoints for message management within character chat sessions.
Provides CRUD operations for messages in conversations.
"""

import uuid
from datetime import datetime, timezone
from typing import List, Optional, Dict, Any
from fastapi import (
    APIRouter, 
    Depends, 
    HTTPException, 
    Query, 
    Path,
    BackgroundTasks,
    status
)
from loguru import logger

# Database and authentication dependencies
from tldw_Server_API.app.api.v1.API_Deps.ChaCha_Notes_DB_Deps import get_chacha_db_for_user
from tldw_Server_API.app.core.AuthNZ.User_DB_Handling import get_request_user, User
from tldw_Server_API.app.core.DB_Management.ChaChaNotes_DB import (
    CharactersRAGDB,
    CharactersRAGDBError
)

# Schemas
from tldw_Server_API.app.api.v1.schemas.chat_session_schemas import (
    MessageCreate,
    MessageResponse,
    MessageUpdate,
    MessageListResponse
)

# Character chat helpers  
from tldw_Server_API.app.core.Character_Chat.Character_Chat_Lib import (
    post_message_to_conversation,
    retrieve_message_details,
    retrieve_conversation_messages_for_ui,
    edit_message_content,
    remove_message_from_conversation,
    find_messages_in_conversation
)

# Rate limiting
from tldw_Server_API.app.core.Character_Chat.character_rate_limiter import (
    get_character_rate_limiter
)

router = APIRouter()

# ========================================================================
# Helper Functions
# ========================================================================

def _convert_db_message_to_response(msg_data: Dict[str, Any]) -> MessageResponse:
    """Convert database message to response model."""
    return MessageResponse(
        id=msg_data.get('id', ''),
        conversation_id=msg_data.get('conversation_id', ''),
        parent_message_id=msg_data.get('parent_message_id'),
        sender=msg_data.get('sender', ''),
        content=msg_data.get('content', ''),
        timestamp=msg_data.get('timestamp', datetime.now(timezone.utc)),
        ranking=msg_data.get('ranking'),
        has_image=bool(msg_data.get('image_data')),
        version=msg_data.get('version', 1)
    )

def _verify_conversation_access(
    db: CharactersRAGDB,
    conversation_id: str,
    user_id: int
) -> Dict[str, Any]:
    """
    Verify user has access to a conversation.
    
    Args:
        db: Database instance
        conversation_id: Conversation ID to check
        user_id: User ID to verify
        
    Returns:
        Conversation data if access allowed
        
    Raises:
        HTTPException: 404 if not found, 403 if unauthorized
    """
    conversation = db.get_conversation_by_id(conversation_id)
    
    if not conversation:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Chat session {conversation_id} not found"
        )
    
    if conversation.get('client_id') != str(user_id):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="You don't have access to this chat session"
        )
    
    return conversation

def _verify_message_access(
    db: CharactersRAGDB,
    message_id: str,
    user_id: int
) -> Dict[str, Any]:
    """
    Verify user has access to a message.
    
    Args:
        db: Database instance
        message_id: Message ID to check
        user_id: User ID to verify
        
    Returns:
        Message data if access allowed
        
    Raises:
        HTTPException: 404 if not found, 403 if unauthorized
    """
    # Get message from database
    conn = db.get_connection()
    cursor = conn.cursor()
    
    cursor.execute("""
        SELECT m.*, c.client_id 
        FROM messages m
        JOIN conversations c ON m.conversation_id = c.id
        WHERE m.id = ? AND m.deleted = 0
    """, (message_id,))
    
    result = cursor.fetchone()
    
    if not result:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Message {message_id} not found"
        )
    
    # Convert to dict
    columns = [description[0] for description in cursor.description]
    message = dict(zip(columns, result))
    
    if message.get('client_id') != str(user_id):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="You don't have access to this message"
        )
    
    return message

# ========================================================================
# Message Endpoints
# ========================================================================

@router.post("/chats/{chat_id}/messages", response_model=MessageResponse,
             status_code=status.HTTP_201_CREATED,
             summary="Send a message in a chat", tags=["Messages"])
async def send_message(
    message_data: MessageCreate,
    chat_id: str = Path(..., description="Chat session ID"),
    db: CharactersRAGDB = Depends(get_chacha_db_for_user),
    current_user: User = Depends(get_request_user)
):
    """
    Add a new message to a chat session.
    
    Args:
        chat_id: Chat session ID
        message_data: Message content and metadata
        db: Database instance
        current_user: Authenticated user
        
    Returns:
        Created message details
        
    Raises:
        HTTPException: 404 if chat not found, 403 if unauthorized, 429 if rate limited
    """
    try:
        # Check rate limits
        rate_limiter = get_character_rate_limiter()
        await rate_limiter.check_rate_limit(current_user.id, "message_send")
        
        # Verify conversation access
        conversation = _verify_conversation_access(db, chat_id, current_user.id)
        
        # Validate parent message if provided
        if message_data.parent_message_id:
            parent_msg = _verify_message_access(db, message_data.parent_message_id, current_user.id)
            if parent_msg.get('conversation_id') != chat_id:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="Parent message must be from the same conversation"
                )
        
        # Create message
        message_id = str(uuid.uuid4())
        
        # Map role to sender format used in database
        sender_map = {
            "user": "user",
            "assistant": "assistant",
            "system": "system"
        }
        sender = sender_map.get(message_data.role, "user")
        
        msg_data = {
            'id': message_id,
            'conversation_id': chat_id,
            'parent_message_id': message_data.parent_message_id,
            'sender': sender,
            'content': message_data.content,
            'client_id': str(current_user.id),
            'version': 1
        }
        
        # Handle image if provided
        if message_data.image_base64:
            try:
                import base64
                img_data = base64.b64decode(message_data.image_base64)
                msg_data['image_data'] = img_data
                msg_data['image_mime_type'] = 'image/png'  # Default, could detect
            except Exception as e:
                logger.warning(f"Failed to decode image data: {e}")
        
        # Add to database
        created_id = db.add_message(msg_data)
        
        if not created_id:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to create message"
            )
        
        # Update conversation last_modified
        conn = db.get_connection()
        cursor = conn.cursor()
        cursor.execute("""
            UPDATE conversations 
            SET last_modified = CURRENT_TIMESTAMP
            WHERE id = ?
        """, (chat_id,))
        conn.commit()
        
        # Retrieve created message
        created_msg = retrieve_message_details(db, created_id)
        
        logger.info(f"Created message {created_id} in chat {chat_id} by user {current_user.id}")
        
        return _convert_db_message_to_response(created_msg)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error sending message to chat {chat_id}: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An unexpected error occurred while sending message"
        )


@router.get("/chats/{chat_id}/messages", response_model=MessageListResponse,
            summary="Get messages in a chat", tags=["Messages"])
async def get_chat_messages(
    chat_id: str = Path(..., description="Chat session ID"),
    limit: int = Query(50, ge=1, le=200, description="Number of messages to return"),
    offset: int = Query(0, ge=0, description="Number of messages to skip"),
    include_deleted: bool = Query(False, description="Include deleted messages"),
    db: CharactersRAGDB = Depends(get_chacha_db_for_user),
    current_user: User = Depends(get_request_user)
):
    """
    Get messages from a chat session.
    
    Args:
        chat_id: Chat session ID
        limit: Maximum number of messages to return
        offset: Number of messages to skip
        include_deleted: Whether to include soft-deleted messages
        db: Database instance
        current_user: Authenticated user
        
    Returns:
        List of messages with pagination info
        
    Raises:
        HTTPException: 404 if chat not found, 403 if unauthorized
    """
    try:
        # Verify conversation access
        conversation = _verify_conversation_access(db, chat_id, current_user.id)
        
        # Get messages
        messages = db.get_messages_for_conversation(chat_id, limit=limit+offset)
        
        if not messages:
            messages = []
        
        # Filter deleted if needed
        if not include_deleted:
            messages = [msg for msg in messages if not msg.get('deleted')]
        
        # Apply offset
        paginated = messages[offset:offset+limit]
        
        return MessageListResponse(
            messages=[_convert_db_message_to_response(msg) for msg in paginated],
            total=len(messages),
            limit=limit,
            offset=offset
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting messages for chat {chat_id}: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An unexpected error occurred while retrieving messages"
        )


@router.get("/messages/{message_id}", response_model=MessageResponse,
            summary="Get a specific message", tags=["Messages"])
async def get_message(
    message_id: str = Path(..., description="Message ID"),
    db: CharactersRAGDB = Depends(get_chacha_db_for_user),
    current_user: User = Depends(get_request_user)
):
    """
    Get details of a specific message.
    
    Args:
        message_id: Message ID
        db: Database instance
        current_user: Authenticated user
        
    Returns:
        Message details
        
    Raises:
        HTTPException: 404 if not found, 403 if unauthorized
    """
    try:
        message = _verify_message_access(db, message_id, current_user.id)
        return _convert_db_message_to_response(message)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting message {message_id}: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An unexpected error occurred while retrieving message"
        )


@router.put("/messages/{message_id}", response_model=MessageResponse,
            summary="Edit a message", tags=["Messages"])
async def edit_message(
    update_data: MessageUpdate,
    message_id: str = Path(..., description="Message ID"),
    expected_version: int = Query(..., description="Expected version for optimistic locking"),
    db: CharactersRAGDB = Depends(get_chacha_db_for_user),
    current_user: User = Depends(get_request_user)
):
    """
    Edit the content of a message.
    
    Args:
        message_id: Message ID to edit
        update_data: New message content
        expected_version: Expected version for optimistic locking
        db: Database instance
        current_user: Authenticated user
        
    Returns:
        Updated message details
        
    Raises:
        HTTPException: 404 if not found, 403 if unauthorized, 409 if version conflict
    """
    try:
        # Verify message access
        message = _verify_message_access(db, message_id, current_user.id)
        
        # Check version
        if message.get('version', 1) != expected_version:
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail=f"Version mismatch. Expected {expected_version}, found {message.get('version', 1)}"
            )
        
        # Update message content
        success = edit_message_content(db, message_id, update_data.content, expected_version)
        
        if not success:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to update message"
            )
        
        # Update conversation last_modified
        conn = db.get_connection()
        cursor = conn.cursor()
        cursor.execute("""
            UPDATE conversations 
            SET last_modified = CURRENT_TIMESTAMP
            WHERE id = ?
        """, (message['conversation_id'],))
        conn.commit()
        
        # Retrieve updated message
        updated_msg = retrieve_message_details(db, message_id)
        
        logger.info(f"Updated message {message_id} by user {current_user.id}")
        
        return _convert_db_message_to_response(updated_msg)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error editing message {message_id}: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An unexpected error occurred while editing message"
        )


@router.delete("/messages/{message_id}", status_code=status.HTTP_204_NO_CONTENT,
               summary="Delete a message", tags=["Messages"])
async def delete_message(
    message_id: str = Path(..., description="Message ID"),
    expected_version: int = Query(..., description="Expected version for optimistic locking"),
    db: CharactersRAGDB = Depends(get_chacha_db_for_user),
    current_user: User = Depends(get_request_user)
):
    """
    Soft delete a message from a conversation.
    
    Args:
        message_id: Message ID to delete
        expected_version: Expected version for optimistic locking
        db: Database instance
        current_user: Authenticated user
        
    Raises:
        HTTPException: 404 if not found, 403 if unauthorized, 409 if version conflict
    """
    try:
        # Verify message access
        message = _verify_message_access(db, message_id, current_user.id)
        
        # Check version
        if message.get('version', 1) != expected_version:
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail=f"Version mismatch. Expected {expected_version}, found {message.get('version', 1)}"
            )
        
        # Soft delete the message
        success = remove_message_from_conversation(db, message_id, expected_version)
        
        if not success:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to delete message"
            )
        
        # Update conversation last_modified
        conn = db.get_connection()
        cursor = conn.cursor()
        cursor.execute("""
            UPDATE conversations 
            SET last_modified = CURRENT_TIMESTAMP
            WHERE id = ?
        """, (message['conversation_id'],))
        conn.commit()
        
        logger.info(f"Soft deleted message {message_id} by user {current_user.id}")
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting message {message_id}: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An unexpected error occurred while deleting message"
        )


@router.get("/chats/{chat_id}/messages/search", response_model=MessageListResponse,
            summary="Search messages in a chat", tags=["Messages"])
async def search_messages(
    chat_id: str = Path(..., description="Chat session ID"),
    query: str = Query(..., description="Search query", min_length=1),
    limit: int = Query(50, ge=1, le=200, description="Maximum results"),
    db: CharactersRAGDB = Depends(get_chacha_db_for_user),
    current_user: User = Depends(get_request_user)
):
    """
    Search for messages in a chat session.
    
    Args:
        chat_id: Chat session ID
        query: Search query string
        limit: Maximum number of results
        db: Database instance
        current_user: Authenticated user
        
    Returns:
        List of matching messages
        
    Raises:
        HTTPException: 404 if chat not found, 403 if unauthorized
    """
    try:
        # Verify conversation access
        conversation = _verify_conversation_access(db, chat_id, current_user.id)
        
        # Search messages
        results = find_messages_in_conversation(db, chat_id, query, limit=limit)
        
        if not results:
            results = []
        
        return MessageListResponse(
            messages=[_convert_db_message_to_response(msg) for msg in results],
            total=len(results),
            limit=limit,
            offset=0
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error searching messages in chat {chat_id}: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An unexpected error occurred while searching messages"
        )


#
# End of character_messages.py
######################################################################################################################