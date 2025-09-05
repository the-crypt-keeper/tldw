# chat_helpers.py
# Description: Helper functions for chat endpoint to improve modularity and maintainability
#
# Imports
import asyncio
import datetime
import random
from typing import Any, Dict, List, Optional, Tuple
from loguru import logger

from tldw_Server_API.app.core.DB_Management.ChaChaNotes_DB import CharactersRAGDB
from tldw_Server_API.app.api.v1.schemas.chat_request_schemas import ChatCompletionRequest
from tldw_Server_API.app.api.v1.API_Deps.ChaCha_Notes_DB_Deps import DEFAULT_CHARACTER_NAME

#######################################################################################################################
#
# Request Validation Functions:

async def validate_request_payload(
    request_data: ChatCompletionRequest,
    max_messages: int = 1000,
    max_images: int = 10,
    max_text_length: int = 400_000
) -> Tuple[bool, Optional[str]]:
    """
    Validate the chat completion request payload.
    
    Args:
        request_data: The chat completion request
        max_messages: Maximum allowed messages
        max_images: Maximum allowed images
        max_text_length: Maximum text length per message
        
    Returns:
        Tuple of (is_valid, error_message)
    """
    # Check messages list
    if not request_data.messages:
        return False, "Messages list cannot be empty."
    
    if len(request_data.messages) > max_messages:
        return False, f"Too many messages (max {max_messages}, got {len(request_data.messages)})."
    
    # Count images across all messages
    total_image_parts = 0
    for msg_idx, msg_model in enumerate(request_data.messages):
        if isinstance(msg_model.content, list):
            for part_idx, part in enumerate(msg_model.content):
                if getattr(part, 'type', None) == 'image_url':
                    total_image_parts += 1
                elif getattr(part, 'type', None) == 'text':
                    text_content = getattr(part, 'text', None)
                    if isinstance(text_content, str) and len(text_content) > max_text_length:
                        return False, f"Text part at index {part_idx} in message {msg_idx} too long."
        elif isinstance(msg_model.content, str):
            if len(msg_model.content) > max_text_length:
                return False, f"Message at index {msg_idx} text too long."
    
    if total_image_parts > max_images:
        return False, f"Too many images in request (max {max_images}, found {total_image_parts})."
    
    return True, None


#######################################################################################################################
#
# Character and Conversation Context Functions:

async def get_or_create_character_context(
    db: CharactersRAGDB,
    character_id: Optional[str],
    loop
) -> Tuple[Optional[Dict[str, Any]], Optional[int]]:
    """
    Get or create character context for the chat.
    
    Args:
        db: Database instance
        character_id: Optional character ID (string or numeric)
        loop: Event loop for async operations
        
    Returns:
        Tuple of (character_card, character_db_id)
    """
    character_card = None
    final_character_db_id = None
    
    if character_id:
        try:
            # Try as integer first
            char_id_int = int(character_id)
            character_card = await loop.run_in_executor(None, db.get_character_card_by_id, char_id_int)
        except ValueError:
            # Not an integer, try by name
            character_card = await loop.run_in_executor(None, db.get_character_card_by_name, character_id)
        
        if character_card:
            final_character_db_id = character_card['id']
            logger.info(f"Loaded character '{character_card['name']}' (ID: {final_character_db_id})")
    
    # Fall back to default character if needed
    if not character_card:
        character_card = await loop.run_in_executor(None, db.get_character_card_by_name, DEFAULT_CHARACTER_NAME)
        if character_card:
            final_character_db_id = character_card['id']
            logger.info(f"Using default character '{DEFAULT_CHARACTER_NAME}' (ID: {final_character_db_id})")
    
    return character_card, final_character_db_id


async def get_or_create_conversation(
    db: CharactersRAGDB,
    conversation_id: Optional[str],
    character_id: int,
    character_name: str,
    client_id: str,
    loop
) -> Tuple[str, bool]:
    """
    Get existing conversation or create a new one with proper concurrency handling.
    
    Args:
        db: Database instance
        conversation_id: Optional existing conversation ID
        character_id: Character database ID
        character_name: Character name for title
        client_id: Client identifier
        loop: Event loop
        
    Returns:
        Tuple of (conversation_id, was_created)
    """
    was_created = False
    
    if conversation_id:
        # Verify existing conversation
        conv_details = await loop.run_in_executor(None, db.get_conversation_by_id, conversation_id)
        if conv_details:
            # Validate ownership and character match
            if (conv_details.get('character_id') == character_id and 
                conv_details.get('client_id') == client_id):
                return conversation_id, False
            else:
                logger.warning(
                    f"Conversation {conversation_id} mismatch - expected char:{character_id}, "
                    f"client:{client_id}, got char:{conv_details.get('character_id')}, "
                    f"client:{conv_details.get('client_id')}"
                )
                conversation_id = None
    
    # Create new conversation if needed with retry logic for race conditions
    if not conversation_id:
        timestamp = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%d_%H%M%S")
        title = f"{character_name} ({timestamp})"
        conv_data = {
            'character_id': character_id,
            'title': title,
            'client_id': client_id
        }
        
        # Try to create with retry on conflict
        max_retries = 3
        for attempt in range(max_retries):
            try:
                # Use transaction context for atomic creation
                from tldw_Server_API.app.core.DB_Management.transaction_utils import db_transaction
                async with db_transaction(db, max_retries=1):
                    conversation_id = await loop.run_in_executor(None, db.add_conversation, conv_data)
                    was_created = True
                    logger.info(f"Created new conversation '{conversation_id}' for character {character_id}")
                    break
            except Exception as e:
                if attempt < max_retries - 1:
                    # Add small random delay to reduce collision probability
                    import random
                    await asyncio.sleep(0.05 + random.random() * 0.1)
                    # Update timestamp for uniqueness
                    timestamp = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%d_%H%M%S_%f")
                    title = f"{character_name} ({timestamp})"
                    conv_data['title'] = title
                else:
                    logger.error(f"Failed to create conversation after {max_retries} attempts: {e}")
                    raise
    
    return conversation_id, was_created


#######################################################################################################################
#
# History Loading Functions:

async def load_conversation_history(
    db: CharactersRAGDB,
    conversation_id: str,
    character_card: Optional[Dict[str, Any]],
    limit: int = 20,
    loop = None
) -> List[Dict[str, Any]]:
    """
    Load conversation history from database.
    
    Args:
        db: Database instance
        conversation_id: Conversation ID
        character_card: Character card for context
        limit: Maximum messages to load
        loop: Event loop
        
    Returns:
        List of messages in OpenAI format
    """
    if not loop:
        import asyncio
        loop = asyncio.get_running_loop()
    
    historical_messages = []
    
    try:
        # Load messages from database
        raw_history = await loop.run_in_executor(
            None, 
            db.get_messages_for_conversation, 
            conversation_id, 
            limit, 
            0, 
            "ASC"
        )
        
        for db_msg in raw_history:
            role = "user" if db_msg.get("sender", "").lower() == "user" else "assistant"
            
            # Build message content
            msg_parts = []
            
            # Add text content
            text_content = db_msg.get("content", "")
            if text_content:
                # Apply placeholder replacement if using character context
                if character_card:
                    from tldw_Server_API.app.core.Character_Chat.Character_Chat_Lib import replace_placeholders
                    char_name = character_card.get('name', "Assistant")
                    text_content = replace_placeholders(text_content, char_name, "User")
                msg_parts.append({"type": "text", "text": text_content})
            
            # Add image if present
            img_data = db_msg.get("image_data")
            img_mime = db_msg.get("image_mime_type")
            if img_data and img_mime:
                try:
                    import base64
                    b64_img = await loop.run_in_executor(None, base64.b64encode, img_data)
                    msg_parts.append({
                        "type": "image_url",
                        "image_url": {"url": f"data:{img_mime};base64,{b64_img.decode('utf-8')}"}
                    })
                except Exception as e:
                    logger.warning(f"Error encoding image from history (msg_id {db_msg.get('id')}): {e}")
            
            # Create message entry
            if msg_parts:
                hist_entry = {
                    "role": role,
                    "content": msg_parts if len(msg_parts) > 1 else msg_parts[0].get("text", "")
                }
                
                # Add character name for assistant messages
                if role == "assistant" and character_card and character_card.get('name'):
                    hist_entry["name"] = character_card.get('name')
                
                historical_messages.append(hist_entry)
        
        logger.info(f"Loaded {len(historical_messages)} historical messages for conversation {conversation_id}")
        
    except Exception as e:
        logger.error(f"Error loading conversation history: {e}", exc_info=True)
    
    return historical_messages


#######################################################################################################################
#
# Message Processing Functions:

async def prepare_llm_messages(
    request_messages: List[Any],
    historical_messages: List[Dict[str, Any]],
    character_card: Optional[Dict[str, Any]] = None
) -> List[Dict[str, Any]]:
    """
    Prepare messages for LLM API call.
    
    Args:
        request_messages: Messages from the request
        historical_messages: Historical conversation messages
        character_card: Optional character context
        
    Returns:
        List of messages ready for LLM
    """
    llm_messages = []
    
    # Add historical messages
    llm_messages.extend(historical_messages)
    
    # Process current request messages
    for msg_model in request_messages:
        if msg_model.role == "system":
            continue  # System messages handled separately
        
        msg_dict = msg_model.model_dump(exclude_none=True)
        
        # Add character name for assistant messages (sanitized for OpenAI compatibility)
        if msg_model.role == "assistant" and character_card and character_card.get('name'):
            # OpenAI requires name to match pattern ^[^\\s<|\\\/>]+$ (no spaces or special chars)
            name = character_card.get('name', '').replace(' ', '_').replace('<', '').replace('>', '').replace('|', '').replace('\\', '').replace('/', '')
            if name:  # Only add if name is not empty after sanitization
                msg_dict["name"] = name
        
        llm_messages.append(msg_dict)
    
    return llm_messages


def extract_system_message(
    request_messages: List[Any],
    character_card: Optional[Dict[str, Any]] = None
) -> Optional[str]:
    """
    Extract system message from request or character card.
    
    Args:
        request_messages: Messages from the request
        character_card: Optional character card with system prompt
        
    Returns:
        System message string or None
    """
    # Check for explicit system message in request
    for msg in request_messages:
        if msg.role == 'system' and isinstance(msg.content, str):
            return msg.content
    
    # Fall back to character system prompt
    if character_card and character_card.get('system_prompt'):
        return character_card.get('system_prompt')
    
    return None


#######################################################################################################################
#
# Response Processing Functions:

def extract_response_content(llm_response: Any) -> Optional[str]:
    """
    Extract text content from LLM response.
    
    Args:
        llm_response: Response from LLM API
        
    Returns:
        Extracted text content or None
    """
    if isinstance(llm_response, str):
        return llm_response
    elif isinstance(llm_response, dict):
        # OpenAI-style response
        choices = llm_response.get("choices", [])
        if choices:
            message = choices[0].get("message", {})
            return message.get("content")
    
    return None


#######################################################################################################################
#
# Provider Validation Functions:

def validate_provider_configuration(
    provider: str,
    api_keys: Dict[str, str]
) -> Tuple[bool, Optional[str]]:
    """
    Validate that a provider is properly configured.
    
    Args:
        provider: Provider name
        api_keys: Dictionary of API keys
        
    Returns:
        Tuple of (is_valid, error_message)
    """
    providers_requiring_keys = [
        "openai", "anthropic", "cohere", "groq", "openrouter",
        "deepseek", "mistral", "google", "huggingface"
    ]
    
    if provider in providers_requiring_keys:
        api_key = api_keys.get(provider)
        if not api_key:
            return False, f"API key for provider '{provider}' is missing or not configured."
    
    return True, None


#
# End of chat_helpers.py
#######################################################################################################################