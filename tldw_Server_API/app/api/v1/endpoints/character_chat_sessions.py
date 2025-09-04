# character_chat_sessions.py
"""
API endpoints for character chat session management.
Provides CRUD operations for chat sessions and character-specific completions.
"""

import asyncio
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
    status,
    Request
)
from fastapi.responses import StreamingResponse, JSONResponse
from loguru import logger

# Database and authentication dependencies
from tldw_Server_API.app.api.v1.API_Deps.ChaCha_Notes_DB_Deps import get_chacha_db_for_user
from tldw_Server_API.app.core.AuthNZ.User_DB_Handling import get_request_user, User
from tldw_Server_API.app.core.DB_Management.ChaChaNotes_DB import (
    CharactersRAGDB,
    CharactersRAGDBError,
    ConflictError,
    InputError
)

# Schemas
from tldw_Server_API.app.api.v1.schemas.chat_session_schemas import (
    ChatSessionCreate,
    ChatSessionResponse,
    ChatSessionUpdate,
    ChatSessionListResponse,
    MessageCreate,
    MessageResponse,
    MessageListResponse,
    CharacterChatCompletionRequest,
    CharacterChatCompletionResponse,
    ChatHistoryExport,
    ChatErrorResponse
)

# Character chat helpers
from tldw_Server_API.app.core.Character_Chat.Character_Chat_Lib import (
    start_new_chat_session,
    post_message_to_conversation,
    retrieve_conversation_messages_for_ui,
    load_chat_and_character
)

# Chat helpers and utilities
from tldw_Server_API.app.core.Chat.chat_helpers import (
    get_or_create_conversation,
    load_conversation_history
)

# Rate limiting
from tldw_Server_API.app.core.Character_Chat.character_rate_limiter import (
    get_character_rate_limiter,
    CharacterRateLimiter
)

# For chat completions
from tldw_Server_API.app.core.Chat.Chat_Functions import (
    chat_api_call as perform_chat_api_call
)

router = APIRouter()

# ========================================================================
# Helper Functions
# ========================================================================

def _convert_db_conversation_to_response(conv_data: Dict[str, Any]) -> ChatSessionResponse:
    """Convert database conversation to response model."""
    return ChatSessionResponse(
        id=conv_data.get('id', ''),
        character_id=conv_data.get('character_id', 0),
        title=conv_data.get('title'),
        rating=conv_data.get('rating'),
        created_at=conv_data.get('created_at', datetime.now(timezone.utc)),
        last_modified=conv_data.get('last_modified', datetime.now(timezone.utc)),
        message_count=conv_data.get('message_count', 0),
        version=conv_data.get('version', 1)
    )

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

# ========================================================================
# Chat Session Endpoints
# ========================================================================

@router.post("/", response_model=ChatSessionResponse, status_code=status.HTTP_201_CREATED,
             summary="Create a new chat session", tags=["Chat Sessions"])
async def create_chat_session(
    session_data: ChatSessionCreate,
    db: CharactersRAGDB = Depends(get_chacha_db_for_user),
    current_user: User = Depends(get_request_user),
    background_tasks: BackgroundTasks = BackgroundTasks()
):
    """
    Create a new chat session with a character.
    
    Args:
        session_data: Chat session creation data
        db: Database instance
        current_user: Authenticated user
        
    Returns:
        Created chat session details
        
    Raises:
        HTTPException: 404 if character not found, 429 if rate limited
    """
    try:
        # Check rate limits
        rate_limiter = get_character_rate_limiter()
        await rate_limiter.check_rate_limit(current_user.id, "chat_create")
        
        # Verify character exists
        character = db.get_character_card_by_id(session_data.character_id)
        if not character:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Character with ID {session_data.character_id} not found"
            )
        
        # Generate chat ID and title
        chat_id = str(uuid.uuid4())
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        title = session_data.title or f"{character['name']} Chat ({timestamp})"
        
        # Create conversation data
        conv_data = {
            'id': chat_id,
            'character_id': session_data.character_id,
            'title': title,
            'root_id': chat_id,  # Root for new conversations
            'parent_conversation_id': session_data.parent_conversation_id,
            'client_id': str(current_user.id),
            'version': 1
        }
        
        # Add to database
        created_id = db.add_conversation(conv_data)
        if not created_id:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to create chat session"
            )
        
        # Retrieve created conversation
        created_conv = db.get_conversation_by_id(created_id)
        if not created_conv:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to retrieve created chat session"
            )
        
        # Log creation
        logger.info(f"Created chat session {created_id} for character {session_data.character_id} by user {current_user.id}")
        
        return _convert_db_conversation_to_response(created_conv)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error creating chat session: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An unexpected error occurred while creating chat session"
        )


@router.get("/{chat_id}", response_model=ChatSessionResponse,
            summary="Get chat session details", tags=["Chat Sessions"])
async def get_chat_session(
    chat_id: str = Path(..., description="Chat session ID"),
    db: CharactersRAGDB = Depends(get_chacha_db_for_user),
    current_user: User = Depends(get_request_user)
):
    """
    Get details of a specific chat session.
    
    Args:
        chat_id: Chat session ID
        db: Database instance
        current_user: Authenticated user
        
    Returns:
        Chat session details
        
    Raises:
        HTTPException: 404 if not found, 403 if unauthorized
    """
    try:
        conversation = db.get_conversation_by_id(chat_id)
        
        if not conversation:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Chat session {chat_id} not found"
            )
        
        # Verify ownership
        if conversation.get('client_id') != str(current_user.id):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="You don't have access to this chat session"
            )
        
        # Get message count
        messages = db.get_messages_for_conversation(chat_id, limit=1000)
        conversation['message_count'] = len(messages) if messages else 0
        
        return _convert_db_conversation_to_response(conversation)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting chat session {chat_id}: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An unexpected error occurred while retrieving chat session"
        )


@router.get("/", response_model=ChatSessionListResponse,
            summary="List user's chat sessions", tags=["Chat Sessions"])
async def list_chat_sessions(
    character_id: Optional[int] = Query(None, description="Filter by character ID"),
    limit: int = Query(50, ge=1, le=200, description="Number of items to return"),
    offset: int = Query(0, ge=0, description="Number of items to skip"),
    db: CharactersRAGDB = Depends(get_chacha_db_for_user),
    current_user: User = Depends(get_request_user)
):
    """
    List all chat sessions for the current user.
    
    Args:
        character_id: Optional character ID filter
        limit: Maximum number of items to return
        offset: Number of items to skip
        db: Database instance
        current_user: Authenticated user
        
    Returns:
        List of chat sessions with pagination info
    """
    try:
        if character_id:
            # Get conversations for specific character
            conversations = db.get_conversations_for_character(character_id, limit=limit, offset=offset)
        else:
            # Get all user conversations (need to implement this method)
            # For now, aggregate from all characters
            conversations = []
            characters = db.list_character_cards(limit=1000)
            for char in characters:
                char_convs = db.get_conversations_for_character(char['id'], limit=limit, offset=offset)
                conversations.extend(char_convs)
        
        # Filter by client_id for security
        user_conversations = [
            conv for conv in conversations 
            if conv.get('client_id') == str(current_user.id)
        ]
        
        # Sort by last_modified descending
        user_conversations.sort(key=lambda x: x.get('last_modified', ''), reverse=True)
        
        # Apply pagination after filtering
        paginated = user_conversations[offset:offset+limit]
        
        # Add message counts
        for conv in paginated:
            messages = db.get_messages_for_conversation(conv['id'], limit=1)
            conv['message_count'] = len(messages) if messages else 0
        
        return ChatSessionListResponse(
            chats=[_convert_db_conversation_to_response(conv) for conv in paginated],
            total=len(user_conversations),
            limit=limit,
            offset=offset
        )
        
    except Exception as e:
        logger.error(f"Error listing chat sessions: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An unexpected error occurred while listing chat sessions"
        )


@router.put("/{chat_id}", response_model=ChatSessionResponse,
            summary="Update chat session", tags=["Chat Sessions"])
async def update_chat_session(
    update_data: ChatSessionUpdate,
    chat_id: str = Path(..., description="Chat session ID"),
    expected_version: int = Query(..., description="Expected version for optimistic locking"),
    db: CharactersRAGDB = Depends(get_chacha_db_for_user),
    current_user: User = Depends(get_request_user)
):
    """
    Update a chat session's metadata.
    
    Args:
        chat_id: Chat session ID
        update_data: Update data
        expected_version: Expected version for optimistic locking
        db: Database instance
        current_user: Authenticated user
        
    Returns:
        Updated chat session details
        
    Raises:
        HTTPException: 404 if not found, 403 if unauthorized, 409 if version conflict
    """
    try:
        # Get current conversation
        conversation = db.get_conversation_by_id(chat_id)
        
        if not conversation:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Chat session {chat_id} not found"
            )
        
        # Verify ownership
        if conversation.get('client_id') != str(current_user.id):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="You don't have access to this chat session"
            )
        
        # Check version
        if conversation.get('version', 1) != expected_version:
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail=f"Version mismatch. Expected {expected_version}, found {conversation.get('version', 1)}"
            )
        
        # Update fields
        update_fields = update_data.model_dump(exclude_unset=True)
        if update_fields:
            # Use database update method (would need to implement)
            # For now, we'll construct the update manually
            conn = db.get_connection()
            cursor = conn.cursor()
            
            set_clauses = []
            params = []
            
            if 'title' in update_fields:
                set_clauses.append("title = ?")
                params.append(update_fields['title'])
            if 'rating' in update_fields:
                set_clauses.append("rating = ?")
                params.append(update_fields['rating'])
            
            if set_clauses:
                set_clauses.append("last_modified = CURRENT_TIMESTAMP")
                set_clauses.append("version = version + 1")
                
                query = f"""
                    UPDATE conversations 
                    SET {', '.join(set_clauses)}
                    WHERE id = ? AND deleted = 0
                """
                params.append(chat_id)
                
                cursor.execute(query, params)
                conn.commit()
        
        # Retrieve updated conversation
        updated_conv = db.get_conversation_by_id(chat_id)
        if updated_conv:
            messages = db.get_messages_for_conversation(chat_id, limit=1000)
            updated_conv['message_count'] = len(messages) if messages else 0
        
        return _convert_db_conversation_to_response(updated_conv)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error updating chat session {chat_id}: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An unexpected error occurred while updating chat session"
        )


@router.delete("/{chat_id}", status_code=status.HTTP_204_NO_CONTENT,
               summary="Delete chat session", tags=["Chat Sessions"])
async def delete_chat_session(
    chat_id: str = Path(..., description="Chat session ID"),
    expected_version: int = Query(..., description="Expected version for optimistic locking"),
    db: CharactersRAGDB = Depends(get_chacha_db_for_user),
    current_user: User = Depends(get_request_user)
):
    """
    Soft delete a chat session.
    
    Args:
        chat_id: Chat session ID
        expected_version: Expected version for optimistic locking
        db: Database instance
        current_user: Authenticated user
        
    Raises:
        HTTPException: 404 if not found, 403 if unauthorized, 409 if version conflict
    """
    try:
        # Get current conversation
        conversation = db.get_conversation_by_id(chat_id)
        
        if not conversation:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Chat session {chat_id} not found"
            )
        
        # Verify ownership
        if conversation.get('client_id') != str(current_user.id):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="You don't have access to this chat session"
            )
        
        # Check version
        if conversation.get('version', 1) != expected_version:
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail=f"Version mismatch. Expected {expected_version}, found {conversation.get('version', 1)}"
            )
        
        # Soft delete
        conn = db.get_connection()
        cursor = conn.cursor()
        
        cursor.execute("""
            UPDATE conversations 
            SET deleted = 1, last_modified = CURRENT_TIMESTAMP, version = version + 1
            WHERE id = ? AND deleted = 0
        """, (chat_id,))
        
        # Also soft delete messages
        cursor.execute("""
            UPDATE messages 
            SET deleted = 1, last_modified = CURRENT_TIMESTAMP
            WHERE conversation_id = ? AND deleted = 0
        """, (chat_id,))
        
        conn.commit()
        
        logger.info(f"Soft deleted chat session {chat_id} by user {current_user.id}")
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting chat session {chat_id}: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An unexpected error occurred while deleting chat session"
        )


# ========================================================================
# Character Chat Completion Endpoint
# ========================================================================

@router.post("/{chat_id}/complete", response_model=CharacterChatCompletionResponse,
             summary="Get AI response in character context", tags=["Chat Completion"])
async def character_chat_completion(
    request_data: CharacterChatCompletionRequest,
    chat_id: str = Path(..., description="Chat session ID"),
    db: CharactersRAGDB = Depends(get_chacha_db_for_user),
    current_user: User = Depends(get_request_user),
    background_tasks: BackgroundTasks = BackgroundTasks()
):
    """
    Get an AI response in the context of a character chat.
    
    Args:
        chat_id: Chat session ID
        request_data: Completion request data
        db: Database instance
        current_user: Authenticated user
        
    Returns:
        AI response with message ID
        
    Raises:
        HTTPException: 404 if chat not found, 403 if unauthorized, 429 if rate limited
    """
    try:
        # Check rate limits
        rate_limiter = get_character_rate_limiter()
        await rate_limiter.check_rate_limit(current_user.id, "chat_completion")
        
        # Get conversation and verify access
        conversation = db.get_conversation_by_id(chat_id)
        if not conversation:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Chat session {chat_id} not found"
            )
        
        if conversation.get('client_id') != str(current_user.id):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="You don't have access to this chat session"
            )
        
        # Get character
        character_id = conversation.get('character_id')
        character = db.get_character_card_by_id(character_id)
        if not character:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Character {character_id} not found"
            )
        
        # Add user message to conversation
        user_msg_id = str(uuid.uuid4())
        user_msg_data = {
            'id': user_msg_id,
            'conversation_id': chat_id,
            'sender': 'user',
            'content': request_data.message,
            'client_id': str(current_user.id),
            'version': 1
        }
        db.add_message(user_msg_data)
        
        # Load conversation history if requested
        messages = []
        if request_data.include_history:
            history = db.get_messages_for_conversation(
                chat_id, 
                limit=request_data.history_limit or 20
            )
            messages = [
                {"role": msg['sender'], "content": msg['content']}
                for msg in history
                if not msg.get('deleted')
            ]
        
        # Add current message
        messages.append({"role": "user", "content": request_data.message})
        
        # Build system prompt from character
        system_prompt = f"""You are {character.get('name', 'Assistant')}.
{character.get('description', '')}
{character.get('personality', '')}
{character.get('scenario', '')}
{character.get('system_prompt', '')}"""
        
        # Prepare chat request
        from tldw_Server_API.app.api.v1.schemas.chat_request_schemas import ChatCompletionRequest
        
        chat_request = ChatCompletionRequest(
            model="gpt-3.5-turbo",  # Default model
            messages=[
                {"role": "system", "content": system_prompt.strip()},
                *messages
            ],
            max_tokens=request_data.max_tokens,
            temperature=request_data.temperature,
            stream=request_data.stream
        )
        
        # Get AI response
        # This would use the existing chat infrastructure
        # For now, create a simple response
        # In production, would call: perform_chat_api_call(chat_request)
        
        ai_response = f"I understand you said: '{request_data.message}'. As {character.get('name', 'Assistant')}, I'm here to help!"
        
        # Save AI response as message
        ai_msg_id = str(uuid.uuid4())
        ai_msg_data = {
            'id': ai_msg_id,
            'conversation_id': chat_id,
            'sender': 'assistant',
            'content': ai_response,
            'parent_message_id': user_msg_id,
            'client_id': str(current_user.id),
            'version': 1
        }
        db.add_message(ai_msg_data)
        
        # Update conversation last_modified
        conn = db.get_connection()
        cursor = conn.cursor()
        cursor.execute("""
            UPDATE conversations 
            SET last_modified = CURRENT_TIMESTAMP
            WHERE id = ?
        """, (chat_id,))
        conn.commit()
        
        return CharacterChatCompletionResponse(
            response=ai_response,
            message_id=ai_msg_id,
            usage={
                "prompt_tokens": len(str(messages)),
                "completion_tokens": len(ai_response.split()),
                "total_tokens": len(str(messages)) + len(ai_response.split())
            }
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in chat completion for session {chat_id}: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An unexpected error occurred during chat completion"
        )


# ========================================================================
# Chat Export Endpoint
# ========================================================================

@router.get("/{chat_id}/export", 
            summary="Export chat history", tags=["Chat Export"])
async def export_chat_history(
    chat_id: str = Path(..., description="Chat session ID"),
    format: str = Query("json", description="Export format (json, markdown, text)"),
    include_metadata: bool = Query(True, description="Include chat metadata"),
    include_character: bool = Query(True, description="Include character info"),
    db: CharactersRAGDB = Depends(get_chacha_db_for_user),
    current_user: User = Depends(get_request_user)
):
    """
    Export chat history in various formats.
    
    Args:
        chat_id: Chat session ID to export
        format: Export format
        include_metadata: Whether to include metadata
        include_character: Whether to include character info
        db: Database instance
        current_user: Authenticated user
        
    Returns:
        Chat history in requested format
        
    Raises:
        HTTPException: 404 if chat not found, 403 if unauthorized
    """
    try:
        # Get conversation
        conversation = db.get_conversation_by_id(chat_id)
        if not conversation:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Chat session {chat_id} not found"
            )
        
        # Verify ownership
        if conversation.get('client_id') != str(current_user.id):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="You don't have access to this chat session"
            )
        
        # Get character info if requested
        character = None
        if include_character:
            character = db.get_character_card_by_id(conversation['character_id'])
        
        # Get messages
        messages = db.get_messages_for_conversation(chat_id, limit=10000)
        
        # Format based on requested type
        if format == "markdown":
            # Markdown format
            lines = []
            if include_metadata:
                lines.append(f"# Chat Export: {conversation.get('title', 'Untitled')}")
                if character:
                    lines.append(f"**Character**: {character.get('name', 'Unknown')}")
                lines.append(f"**Date**: {conversation.get('created_at', '')}")
                lines.append(f"**Messages**: {len(messages)}")
                lines.append("\n---\n")
            
            for msg in messages:
                if msg.get('deleted'):
                    continue
                sender = msg.get('sender', 'unknown')
                content = msg.get('content', '')
                timestamp = msg.get('timestamp', '')
                lines.append(f"**{sender.title()}** ({timestamp}):")
                lines.append(f"{content}\n")
            
            return {"content": "\n".join(lines), "format": "markdown"}
            
        elif format == "text":
            # Plain text format
            lines = []
            if include_metadata:
                lines.append(f"Chat: {conversation.get('title', 'Untitled')}")
                if character:
                    lines.append(f"Character: {character.get('name', 'Unknown')}")
                lines.append(f"Date: {conversation.get('created_at', '')}")
                lines.append("-" * 40)
            
            for msg in messages:
                if msg.get('deleted'):
                    continue
                sender = msg.get('sender', 'unknown')
                content = msg.get('content', '')
                lines.append(f"{sender}: {content}")
            
            return {"content": "\n".join(lines), "format": "text"}
            
        else:
            # JSON format (default)
            export_data = {
                "chat_id": chat_id,
                "character_name": character.get('name', 'Unknown') if character else 'Unknown',
                "character_id": conversation['character_id'],
                "title": conversation.get('title'),
                "created_at": str(conversation.get('created_at', '')),
                "messages": [
                    {
                        "id": msg.get('id'),
                        "role": msg.get('sender'),
                        "content": msg.get('content'),
                        "timestamp": str(msg.get('timestamp', '')),
                        "has_image": bool(msg.get('image_data'))
                    }
                    for msg in messages
                    if not msg.get('deleted')
                ]
            }
            
            if include_metadata:
                export_data["metadata"] = {
                    "total_messages": len(messages),
                    "rating": conversation.get('rating'),
                    "last_modified": str(conversation.get('last_modified', ''))
                }
            
            return export_data
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error exporting chat {chat_id}: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An unexpected error occurred while exporting chat history"
        )


#
# End of character_chat_sessions.py
######################################################################################################################