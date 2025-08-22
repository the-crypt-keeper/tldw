# Server_API/app/api/v1/endpoints/chat.py
# Description: This code provides a FastAPI endpoint for all Chat-related functionalities.
#
# Imports
from __future__ import annotations
# ---------------------------------------------------------------------------
# Imports
# ---------------------------------------------------------------------------
from tldw_Server_API.app.core.config import AUTH_BEARER_PREFIX
from tldw_Server_API.app.core.Auth.auth_utils import (
    extract_bearer_token,
    validate_api_token,
    get_expected_api_token,
    is_authentication_required
)
from tldw_Server_API.app.core.Utils.image_validation import (
    validate_data_uri,
    safe_decode_base64_image,
    validate_image_url
)
import asyncio
import base64
import datetime
import json
import logging
import os
import time
import uuid
from collections import deque
from functools import partial
from io import BytesIO
from typing import Any, AsyncIterator, Dict, Iterator, List, Optional, Tuple, Union

from fastapi import (
    APIRouter,
    BackgroundTasks,
    Body,
    Depends,
    File,
    Form,
    Header,
    HTTPException,
    Query,
    Request,
    Response,
    status,
    UploadFile
)
from fastapi.encoders import jsonable_encoder
from requests import RequestException, HTTPError

# Import new modules for integration
from tldw_Server_API.app.core.DB_Management.async_db_wrapper import create_async_db
from tldw_Server_API.app.core.Chat.provider_manager import get_provider_manager
from tldw_Server_API.app.core.Chat.rate_limiter import get_rate_limiter
from tldw_Server_API.app.core.Chat.request_queue import get_request_queue, RequestPriority
from tldw_Server_API.app.core.Audit.unified_audit_service import (
    get_unified_audit_service, 
    AuditEventType, 
    AuditContext
)
from tldw_Server_API.app.core.Utils.cpu_bound_handler import process_large_json_async
from tldw_Server_API.app.core.Utils.chunked_image_processor import get_image_processor

# API Rate Limiter/Caching via Redis
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.middleware import SlowAPIMiddleware
from slowapi.util import get_remote_address
from loguru import logger
from starlette.background import BackgroundTask
from starlette.responses import JSONResponse, StreamingResponse

from tldw_Server_API.app.api.v1.API_Deps.ChaCha_Notes_DB_Deps import (
    DEFAULT_CHARACTER_NAME,
    get_chacha_db_for_user,
)
from tldw_Server_API.app.api.v1.API_Deps.DB_Deps import get_media_db_for_user
from tldw_Server_API.app.api.v1.schemas.chat_request_schemas import (
    API_KEYS,
    ChatCompletionRequest,
    DEFAULT_LLM_PROVIDER,
)
from tldw_Server_API.app.core.Chat.Chat_Functions import (
    ChatAPIError,
    ChatAuthenticationError,
    ChatBadRequestError,
    ChatConfigurationError,
    ChatDictionary,
    ChatProviderError,
    ChatRateLimitError,
    chat_api_call as perform_chat_api_call,
    process_user_input,
    update_chat_content,
)
from tldw_Server_API.app.core.Chat.prompt_template_manager import (
    DEFAULT_RAW_PASSTHROUGH_TEMPLATE,
    PromptTemplate,
    apply_template_to_string,
    load_template,
)
from tldw_Server_API.app.core.DB_Management.ChaChaNotes_DB import (
    CharactersRAGDB,
    CharactersRAGDBError,
    ConflictError,
    InputError,
)
from tldw_Server_API.app.core.DB_Management.transaction_utils import (
    db_transaction,
    save_conversation_with_messages,
    update_conversation_with_rollback,
)
from tldw_Server_API.app.core.Chat.streaming_utils import (
    StreamingResponseHandler,
    create_streaming_response_with_timeout,
)
from tldw_Server_API.app.core.Chat.chat_helpers import (
    validate_request_payload,
    get_or_create_character_context,
    get_or_create_conversation,
    load_conversation_history,
    prepare_llm_messages,
    extract_system_message,
    extract_response_content,
)
from tldw_Server_API.app.api.v1.schemas.chat_validators import (
    validate_conversation_id,
    validate_character_id,
    validate_tool_definitions,
    validate_temperature,
    validate_max_tokens,
    validate_request_size,
)
from tldw_Server_API.app.core.Character_Chat.Character_Chat_Lib import replace_placeholders
from tldw_Server_API.app.core.Chat.chat_metrics import get_chat_metrics
#######################################################################################################################
#
# ---------------------------------------------------------------------------
# Constants & helpers
# ---------------------------------------------------------------------------

router = APIRouter()

# Load configuration values from config
from tldw_Server_API.app.core.config import load_comprehensive_config

_config = load_comprehensive_config()
# ConfigParser uses sections, check if Chat-Module section exists
_chat_config = {}
if _config and _config.has_section('Chat-Module'):
    _chat_config = dict(_config.items('Chat-Module'))

ALLOWED_IMAGE_MIME_TYPES: set[str] = {"image/png", "image/jpeg", "image/webp"}
MAX_BASE64_BYTES: int = int(_chat_config.get('max_base64_image_size_mb', 3)) * 1024 * 1024
MAX_TEXT_LENGTH: int = int(_chat_config.get('max_text_length_per_message', 400000))
MAX_MESSAGES_PER_REQUEST: int = int(_chat_config.get('max_messages_per_request', 1000))
MAX_IMAGES_PER_REQUEST: int = int(_chat_config.get('max_images_per_request', 10))

# --- Helper Functions ---

def _check_mime(mime: str) -> bool:
    return mime.lower() in ALLOWED_IMAGE_MIME_TYPES

async def _process_content_for_db_sync(
    content_iterable: Any, # Can be list of dicts or string
    conversation_id: str # For logging
) -> tuple[list[str], list[tuple[bytes, str]]]:
    """
    Synchronous helper to process message content, including base64 decoding.
    To be run in an executor.
    """
    text_parts_sync: list[str] = []
    images_sync: list[tuple[bytes, str]] = []   # (bytes, mime)

    processed_content_iterable: Any # Define type more specifically if possible
    if isinstance(content_iterable, str):
        processed_content_iterable = [{"type": "text", "text": content_iterable}]
    elif isinstance(content_iterable, list):
        processed_content_iterable = content_iterable
    else:
        logger.warning(
            "[DB SYNC] Unsupported content type=%s for conv=%s, treating as unsupported text.",
            type(content_iterable),
            conversation_id
        )
        processed_content_iterable = [{"type": "text", "text": f"<unsupported content type: {type(content_iterable).__name__}>"}]

    for part in processed_content_iterable:
        p_type = part.get("type")
        if p_type == "text":
            snippet = str(part.get("text", ""))[:MAX_TEXT_LENGTH + 1] # Ensure text is string
            if len(snippet) > MAX_TEXT_LENGTH:
                logger.info(
                    "[DB SYNC] Trimmed over-long text part (>%d chars) for conv=%s",
                    MAX_TEXT_LENGTH,
                    conversation_id
                )
                snippet = snippet[:MAX_TEXT_LENGTH]
            text_parts_sync.append(snippet)
        elif p_type == "image_url":
            url_dict = part.get("image_url", {})
            url_str = url_dict.get("url", "")

            if url_str.startswith("data:"):
                # Use chunked image processor for large images
                image_processor = get_image_processor()
                if image_processor and len(url_str) > 100000:  # Large image, use chunked processing
                    is_valid, decoded_bytes, mime_type, error_msg = await image_processor.process_image_url(
                        url_str, MAX_BASE64_BYTES
                    )
                    if is_valid and decoded_bytes:
                        images_sync.append((decoded_bytes, mime_type))
                        logger.debug("[DB SYNC] Successfully processed large image for conv=%s", conversation_id)
                    else:
                        logger.warning(
                            "[DB SYNC] Large image processing failed for conv=%s: %s",
                            conversation_id, error_msg
                        )
                        text_parts_sync.append(f"<Image failed: {error_msg}>")
                else:
                    # Small image or processor not available - use standard validation
                    is_valid, mime_type, decoded_bytes = validate_image_url(url_str)
                    if is_valid and decoded_bytes:
                        images_sync.append((decoded_bytes, mime_type))
                        logger.debug("[DB SYNC] Successfully validated and decoded image for conv=%s", conversation_id)
                    else:
                        logger.warning(
                            "[DB SYNC] Image validation failed for conv=%s, storing as text placeholder",
                            conversation_id
                        )
                        # Provide more context about the failed image
                        text_parts_sync.append(f"<Image failed validation: {mime_type if mime_type else 'unknown type'}>")
            else:
                logger.warning(
                    "[DB SYNC] image_url part was not a valid data URI or did not pass checks, storing as text placeholder. conv=%s, url_start='%.50s...'",
                    conversation_id, url_str
                )
                text_parts_sync.append(f"<Image URL (not processed): {url_str[:200]}>")
    return text_parts_sync, images_sync

async def _save_message_turn_to_db(
    db: CharactersRAGDB,
    conversation_id: str,
    message_obj: Dict[str, Any],
    use_transaction: bool = False
) -> Optional[str]:
    """
    Persist a single user/assistant message.
    - Validates size/format.
    - CPU-bound content processing (image decoding) is run in an executor.
    - DB write is run in an executor.
    - Logs only metadata, never raw content.
    - Can optionally use transactions for atomic operations.
    """
    metrics = get_chat_metrics()
    current_loop = asyncio.get_running_loop()
    role = message_obj.get("role")
    if role not in ("user", "assistant"):
        logger.warning("Skip DB save: invalid role='%s' for conv=%s", role, conversation_id)
        return None

    content = message_obj.get("content")

    try:
        # Track image processing if content contains images
        image_start_time = time.time()
        # Call async function directly instead of using run_in_executor
        text_parts, images = await _process_content_for_db_sync(content, conversation_id)
        
        # Track image processing metrics if images were processed
        if images:
            image_processing_time = time.time() - image_start_time
            for _, _ in images:
                metrics.track_image_processing(
                    size_bytes=len(images[0][0]) if images else 0,
                    validation_time=image_processing_time
                )
    except Exception as e_proc:
        logger.error(
            "Error processing message content in executor for DB save. conv=%s err_type=%s err=%s",
            conversation_id, type(e_proc).__name__, e_proc, exc_info=True
        )
        return None

    if not text_parts and not images: # Issue 1 Fix
        # Save a placeholder message to maintain conversation continuity
        # This ensures we don't lose track of failed image processing attempts
        logger.warning("Message with no valid content after processing for conv=%s, saving placeholder", conversation_id)
        text_parts = ["<Message processing failed - no valid content>"]
        # Continue to save the message with placeholder text

    db_payload = {
        "conversation_id": conversation_id,
        "sender": message_obj.get("name") or role,
        "content": "\n".join(text_parts) if text_parts else None,
        "images": [{"data": b, "mime": m} for b, m in images] or None,
        "client_id": db.client_id,
    }

    try:
        # Track database operation
        async with metrics.track_database_operation("save_message"):
            if use_transaction:
                # Track transaction metrics
                retries = 0
                try:
                    async with db_transaction(db) as transaction_ctx:
                        # Track retry count if available
                        if hasattr(transaction_ctx, 'retry_count'):
                            retries = transaction_ctx.retry_count
                        result = await current_loop.run_in_executor(None, db.add_message, db_payload)
                        metrics.track_transaction(success=True, retries=retries)
                        metrics.track_message_saved(conversation_id, role)
                        return result
                except Exception as e:
                    metrics.track_transaction(success=False, retries=retries)
                    raise
            else:
                result = await current_loop.run_in_executor(None, db.add_message, db_payload)
                metrics.track_message_saved(conversation_id, role)
                return result
    except (InputError, ConflictError, CharactersRAGDBError) as e_db:
        logger.error("DB error saving message for conv=%s: Type=%s, Msg='%s'", conversation_id, type(e_db).__name__, e_db)
        return None
    except Exception as e_unexpected_db:
        logger.error("Unexpected DB error saving message for conv=%s: %s", conversation_id, e_unexpected_db, exc_info=True)
        return None


@router.post(
    "/completions",
    summary="Creates a model response and manages conversation state.",
    tags=["Chat"],
    responses={
        status.HTTP_400_BAD_REQUEST: {"description": "Invalid request (e.g., empty messages, text too long, bad parameters)."},
        status.HTTP_401_UNAUTHORIZED: {"description": "Invalid authentication token."},
        status.HTTP_404_NOT_FOUND: {"description": "Resource not found (e.g., character)."},
        status.HTTP_409_CONFLICT: {"description": "Data conflict (e.g., version mismatch during DB operation)."},
        status.HTTP_413_REQUEST_ENTITY_TOO_LARGE: {"description": "Request payload too large (e.g., too many messages, too many images)."},
        status.HTTP_429_TOO_MANY_REQUESTS: {"description": "Rate limit exceeded."},
        status.HTTP_500_INTERNAL_SERVER_ERROR: {"description": "Internal server error."},
        status.HTTP_502_BAD_GATEWAY: {"description": "Error received from an upstream LLM provider."},
        status.HTTP_503_SERVICE_UNAVAILABLE: {"description": "Service temporarily unavailable or misconfigured (e.g., provider API key issue)."},
        status.HTTP_504_GATEWAY_TIMEOUT: {"description": "Upstream LLM provider timed out."},
    }
)
async def create_chat_completion(
    request_data: ChatCompletionRequest = Body(...),
    chat_db: CharactersRAGDB = Depends(get_chacha_db_for_user),
    Token: str = Header(None, description="Bearer token for authentication."),
    request: Request = None,  # Optional Request object for audit logging and rate limiting
    # background_tasks: BackgroundTasks = Depends(), # Replaced by starlette.background.BackgroundTask for StreamingResponse
):
    current_loop = asyncio.get_running_loop()
    
    # Generate unique request ID for tracking
    request_id = str(uuid.uuid4())
    
    # Wrap database with async wrapper for better performance
    async_db = create_async_db(chat_db)
    
    # Initialize metrics collector
    metrics = get_chat_metrics()
    
    # Get provider and model for metrics
    provider = (request_data.api_provider or DEFAULT_LLM_PROVIDER).lower()
    model = request_data.model or "unknown"
    client_id = getattr(chat_db, 'client_id', 'unknown_client')
    
    # Initialize audit logger with error handling
    audit_service = None
    try:
        audit_service = await get_unified_audit_service()
    except Exception as audit_error:
        logger.warning(f"Failed to initialize audit service: {audit_error}")
        # Continue without audit logging rather than failing the request
    
    # Get user ID for rate limiting and audit (extract from token or use client_id)
    user_id = client_id  # In production, extract from JWT token
    
    # Initialize audit context for logging
    context = None
    if audit_service:
        try:
            context = AuditContext(
                user_id=user_id,
                request_id=request_id,
                ip_address=request.client.host if request and hasattr(request, 'client') else None
            )
            await audit_service.log_event(
                event_type=AuditEventType.API_REQUEST,
                context=context,
                action="chat_completion_request",
                metadata={
                    "model": model,
                    "provider": provider,
                    "message_count": len(request_data.messages),
                    "streaming": request_data.stream,
                    "has_tools": bool(request_data.tools),
                    "conversation_id": request_data.conversation_id
                }
            )
        except Exception as log_error:
            logger.warning(f"Failed to log audit event: {log_error}")
            # Continue without logging rather than failing the request
    
    # Start tracking the request
    # Serialize request once and reuse
    request_json = json.dumps(request_data.model_dump())
    request_json_bytes = request_json.encode()
    
    async with metrics.track_request(
        provider=provider,
        model=model,
        streaming=request_data.stream,
        client_id=client_id
    ) as span:
        # Track request size
        metrics.metrics.request_size_bytes.record(len(request_json_bytes))

        # Secure authentication validation
        if is_authentication_required():
            if not Token:
                metrics.track_auth_failure("missing_token")
                logger.warning("Authentication required but no token provided")
                raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Missing authentication token.")
            
            extracted_token = extract_bearer_token(Token)
            if not extracted_token:
                metrics.track_auth_failure("invalid_token_format")
                logger.warning("Invalid token format provided")
                raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid authentication token format.")
            
            expected_token = get_expected_api_token()
            if not expected_token:
                logger.critical("Authentication required but API_BEARER not configured")
                raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Server authentication is misconfigured.")
            
            if not validate_api_token(extracted_token, expected_token):
                metrics.track_auth_failure("invalid_token")
                logger.warning("Invalid authentication token provided")
                raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid authentication token.")

        # Validate request payload using helper function
        validation_start = time.time()
        is_valid, error_message = await validate_request_payload(
            request_data, 
            max_messages=MAX_MESSAGES_PER_REQUEST,
            max_images=MAX_IMAGES_PER_REQUEST,
            max_text_length=MAX_TEXT_LENGTH
        )
        metrics.metrics.validation_duration.record(time.time() - validation_start)
        
        if not is_valid:
            metrics.track_validation_failure("payload", error_message)
            logger.warning(f"Request validation failed: {error_message}")
            if "too many" in error_message.lower() or "too long" in error_message.lower():
                raise HTTPException(status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE, detail=error_message)
            else:
                raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=error_message)
    
    # Validate specific fields with validators
    try:
        if request_data.conversation_id:
            request_data.conversation_id = validate_conversation_id(request_data.conversation_id)
        if request_data.character_id:
            request_data.character_id = validate_character_id(request_data.character_id)
        if request_data.tools:
            # Convert ToolDefinition objects to dictionaries for validation
            tools_as_dicts = [tool.model_dump(exclude_none=True) if hasattr(tool, 'model_dump') else tool 
                              for tool in request_data.tools]
            validated_tools = validate_tool_definitions(tools_as_dicts)
            # Keep the validated tools as dicts since that's what the LLM API expects
            request_data.tools = validated_tools
        if request_data.temperature is not None:
            request_data.temperature = validate_temperature(request_data.temperature)
        if request_data.max_tokens is not None:
            request_data.max_tokens = validate_max_tokens(request_data.max_tokens)
        
        # Validate overall request size (reuse cached JSON)
        validate_request_size(request_json)
        
        # Apply rate limiting
        rate_limiter = get_rate_limiter()
        if rate_limiter:
            # Estimate tokens for rate limiting
            estimated_tokens = len(request_json) // 4  # Rough estimate: 4 chars per token
            
            allowed, rate_error = await rate_limiter.check_rate_limit(
                user_id=user_id,
                conversation_id=request_data.conversation_id,
                estimated_tokens=estimated_tokens
            )
            
            if not allowed:
                metrics.track_rate_limit(user_id)
                if audit_service and context:
                    await audit_service.log_event(
                        event_type=AuditEventType.API_RATE_LIMITED,
                        context=context,
                        action="rate_limit_exceeded",
                        details={"reason": rate_error}
                    )
                raise HTTPException(
                    status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                    detail=rate_error or "Rate limit exceeded"
                )
    except ValueError as e:
        logger.warning(f"Input validation error: {e}")
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))

    provider = (request_data.api_provider or DEFAULT_LLM_PROVIDER).lower()
    user_identifier_for_log = getattr(chat_db, 'client_id', 'unknown_client') # Example from original
    logger.info(
        "Chat completion request. Provider=%s, Model=%s, User=%s, Stream=%s, ConvID=%s, CharID=%s",
        provider, request_data.model, user_identifier_for_log,
        request_data.stream, request_data.conversation_id, request_data.character_id
    )

    character_card_for_context: Optional[Dict[str, Any]] = None
    final_conversation_id: Optional[str] = request_data.conversation_id
    final_character_db_id: Optional[int] = None # Initialize

    try:
        target_api_provider = provider # Already determined
        provider_api_key = API_KEYS.get(target_api_provider) # API_KEYS should be up-to-date

        # Simplified list, actual check might be in Chat_Functions or per-provider
        # FIXME - This should be a more dynamic check based on the provider's requirements.
        providers_requiring_keys = ["openai", "anthropic", "cohere", "groq", "openrouter", "deepseek", "mistral", "google", "huggingface"]
        if target_api_provider in providers_requiring_keys and not provider_api_key:
            logger.error(f"API key for provider '{target_api_provider}' is missing or not configured.")
            raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=f"Service for '{target_api_provider}' is not configured (key missing).")

        conversation_created_this_turn = False

        # --- Character and Conversation Context ---
        # Use helper function to get or create character context
        character_card_for_context, final_character_db_id = await get_or_create_character_context(
            chat_db,
            request_data.character_id,
            current_loop
        )
        if character_card_for_context:
            logger.debug(f"Loaded character: {character_card_for_context.get('name')} with system_prompt: {character_card_for_context.get('system_prompt', 'None')[:50]}...")
        
        # Track character access
        if character_card_for_context:
            metrics.track_character_access(
                character_id=str(request_data.character_id or "default"),
                cache_hit=False  # Could be enhanced to track actual cache hits
            )
        
        if not character_card_for_context:
            logger.critical(f"CRITICAL: Neither requested character nor default character found in DB.")
            raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Character context is missing.")

        # Multi-User Security FIXME
        client_id_from_db = getattr(chat_db, 'client_id', None)
        # if not client_id_from_db: # Should be set by get_chacha_db_for_user
        #      logger.critical("Client ID missing on chat_db instance. This is a server configuration issue.")
        #      raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Server error: Client identification for DB operations failed.")

        # Use helper function to get or create conversation
        final_conversation_id, conversation_created_this_turn = await get_or_create_conversation(
            chat_db,
            final_conversation_id,
            final_character_db_id,
            character_card_for_context.get('name', 'Chat'),
            client_id_from_db,
            current_loop
        )
        
        # Track conversation creation/resumption
        if final_conversation_id:
            metrics.track_conversation(final_conversation_id, conversation_created_this_turn)
        
        if not final_conversation_id:
            raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Failed to create or retrieve conversation.")

        # --- History Loading ---
        historical_openai_messages: List[Dict[str, Any]] = []
        if not conversation_created_this_turn and final_conversation_id:
            # Limit history length (e.g., 20 messages = 10 turns)
            raw_hist = await current_loop.run_in_executor(None, chat_db.get_messages_for_conversation, final_conversation_id, 20, 0, "ASC")
            for db_msg in raw_hist:
                role = "user" if db_msg.get("sender", "").lower() == "user" else "assistant"
                char_name_hist = character_card_for_context.get('name', "Char") if character_card_for_context else "Char"

                text_content = db_msg.get("content", "")
                if text_content: # Apply placeholder replacement
                     text_content = replace_placeholders(text_content, char_name_hist, "User") # Assuming "User" for {{user}} placeholder

                msg_parts = []
                if text_content:
                    msg_parts.append({"type": "text", "text": text_content})

                img_data, img_mime = db_msg.get("image_data"), db_msg.get("image_mime_type")
                if img_data and img_mime:
                    try:
                        b64_img = await current_loop.run_in_executor(None, base64.b64encode, img_data)
                        msg_parts.append({"type": "image_url", "image_url": {"url": f"data:{img_mime};base64,{b64_img.decode('utf-8')}"}})
                    except Exception as e: logger.warning(f"Error encoding DB image for history (msg_id {db_msg.get('id')}): {e}")

                if msg_parts:
                    hist_entry = {"role": role, "content": msg_parts}
                    if role == "assistant" and character_card_for_context and character_card_for_context.get('name'):
                        hist_entry["name"] = character_card_for_context.get('name')
                    historical_openai_messages.append(hist_entry)
            logger.info(f"Loaded {len(historical_openai_messages)} historical messages for conv_id '{final_conversation_id}'.")

        # --- User Message Processing & DB Save ---
        current_turn_messages_for_llm: List[Dict[str, Any]] = []
        for msg_model in request_data.messages:
            if msg_model.role == "system": continue # Handled by templating

            msg_dict = msg_model.model_dump(exclude_none=True)
            msg_for_db = msg_dict.copy()
            if msg_model.role == "assistant" and character_card_for_context:
                msg_for_db["name"] = character_card_for_context.get('name', "Assistant")

            await _save_message_turn_to_db(chat_db, final_conversation_id, msg_for_db, use_transaction=True) # Use transaction for consistency

            msg_for_llm = msg_dict.copy()
            if msg_model.role == "assistant" and character_card_for_context and character_card_for_context.get('name'):
                msg_for_llm["name"] = character_card_for_context.get('name')
            current_turn_messages_for_llm.append(msg_for_llm)

        # --- Prompt Templating ---
        llm_payload_messages = historical_openai_messages + current_turn_messages_for_llm
        active_template = load_template(request_data.prompt_template_name or DEFAULT_RAW_PASSTHROUGH_TEMPLATE.name)
        template_data: Dict[str, Any] = {}
        if character_card_for_context:
            template_data.update({k: v for k, v in character_card_for_context.items() if isinstance(v, (str, int, float))}) # Basic fields
            template_data["char_name"] = character_card_for_context.get("name", "Character") # Ensure common alias
            # Add specific character fields used by templates if not covered by above
            template_data["character_system_prompt"] = character_card_for_context.get('system_prompt', "")

        sys_msg_from_req = next((m.content for m in request_data.messages if m.role == 'system' and isinstance(m.content, str)), None)
        template_data["original_system_message_from_request"] = sys_msg_from_req or ""

        final_system_message: Optional[str] = None
        logger.debug(f"sys_msg_from_req: {sys_msg_from_req}, active_template: {active_template}, character: {character_card_for_context.get('name') if character_card_for_context else None}")
        if active_template and active_template.system_message_template:
            final_system_message = apply_template_to_string(active_template.system_message_template, template_data)
            # If template produces empty string and we have a character with system prompt, use that instead
            if not final_system_message and character_card_for_context and character_card_for_context.get('system_prompt'):
                final_system_message = character_card_for_context.get('system_prompt')
                logger.debug(f"Template produced empty, using character system prompt: {final_system_message[:50]}...")
        elif sys_msg_from_req:
            final_system_message = sys_msg_from_req
        elif character_card_for_context and character_card_for_context.get('system_prompt'):
            # Use character's system prompt if no template and no system message in request
            final_system_message = character_card_for_context.get('system_prompt')
            logger.debug(f"Using character system prompt: {final_system_message[:50]}...")
        
        logger.debug(f"Final system message: {final_system_message}")

        templated_llm_payload: List[Dict[str, Any]] = []
        # FIXME
        # This logic should be efficient or offloaded if it becomes a bottleneck for large histories/contents.
        # For now, assume original logic is mostly sound but ensure it handles content lists correctly.
        if active_template:
            for msg in llm_payload_messages:
                templated_msg_content = msg.get("content")
                role = msg.get("role")
                content_template_str = None
                if role == "user" and active_template.user_message_content_template:
                    content_template_str = active_template.user_message_content_template
                elif role == "assistant" and active_template.assistant_message_content_template:
                    content_template_str = active_template.assistant_message_content_template

                if content_template_str:
                    new_content_parts = []
                    msg_template_data = template_data.copy()
                    if isinstance(templated_msg_content, str):
                        msg_template_data["message_content"] = templated_msg_content
                        new_text = apply_template_to_string(content_template_str, msg_template_data)
                        new_content_parts.append({"type": "text", "text": new_text or templated_msg_content})
                    elif isinstance(templated_msg_content, list):
                        for part in templated_msg_content:
                            if part.get("type") == "text":
                                msg_template_data["message_content"] = part.get("text", "")
                                new_text_part = apply_template_to_string(content_template_str, msg_template_data)
                                new_content_parts.append({"type": "text", "text": new_text_part or part.get("text", "")})
                            else:
                                new_content_parts.append(part) # Keep image parts
                    templated_llm_payload.append({**msg, "content": new_content_parts or templated_msg_content}) # type: ignore
                else:
                    templated_llm_payload.append(msg)
        else:
            templated_llm_payload = llm_payload_messages

        # --- LLM Call ---
        call_params = request_data.model_dump(
            exclude_none=True,
            exclude={"api_provider", "messages", "character_id", "conversation_id", "prompt_template_name", "stream"}
        )

        # Rename keys to match chat_api_call's signature for generic params
        if "temperature" in call_params:
            call_params["temp"] = call_params.pop("temperature")

        if "top_p" in call_params:
            top_p_value = call_params.pop("top_p")
            # chat_api_call has 'topp' and 'maxp' which both relate to top_p sampling.
            # Pass the value to both, let PROVIDER_PARAM_MAP in chat_api_call pick the relevant one.
            call_params["topp"] = top_p_value
            call_params["maxp"] = top_p_value

        if "user" in call_params:
            call_params["user_identifier"] = call_params.pop("user")

        # response_format, tools, tool_choice are already dict/list of dicts/str from model_dump if not None.
        # They match the expected names in chat_api_call signature.

        # Add other fixed arguments
        call_params.update({
            "api_endpoint": target_api_provider,
            "api_key": provider_api_key,
            "messages_payload": templated_llm_payload,
            "system_message": final_system_message,  # This can be None
            "streaming": request_data.stream,  # This is a boolean
        })

        # Filter out None values before making the call, as chat_api_call's defaults handle Nones.
        # The previous `cleaned_args` did this.
        # Keep system_message even if it's None - let the LLM call handle it
        cleaned_args = {k: v for k, v in call_params.items() if v is not None}
        if 'system_message' not in cleaned_args and final_system_message is not None:
            cleaned_args['system_message'] = final_system_message

        # Use provider manager for health checks and failover
        provider_manager = get_provider_manager()
        selected_provider = provider
        
        if provider_manager:
            # Get healthy provider
            healthy_provider = provider_manager.get_available_provider()
            if healthy_provider:
                selected_provider = healthy_provider
                logger.info(f"Using provider {selected_provider} (health check passed)")
            else:
                logger.warning(f"No healthy providers available, using {provider} anyway")
        
        # Update provider in cleaned_args
        # Note: chat_api_call expects 'api_endpoint', not 'api_provider'
        # This is already set in call_params as 'api_endpoint'
        # cleaned_args['api_provider'] = selected_provider  # REMOVED - causes error
        
        # TODO: Request Queue Integration (SHIM)
        # ------------------------------------------------------------------------
        # The request queue system has been initialized in main.py but is not yet
        # integrated here. Once the central scheduling/queue module is built, this
        # endpoint should enqueue requests rather than processing them directly.
        # 
        # Integration points:
        # 1. Get the request queue instance: queue = get_request_queue()
        # 2. Determine priority based on user/request type
        # 3. Enqueue the request with: 
        #    future = await queue.enqueue(
        #        request_id=request_id,
        #        request_data={'cleaned_args': cleaned_args, 'request': request_data},
        #        client_id=client_id,
        #        priority=priority,
        #        estimated_tokens=estimated_tokens
        #    )
        # 4. Await the future for the result
        # 5. The queue's worker would call perform_chat_api_call
        #
        # Benefits of queue integration:
        # - Prevents server overload with backpressure
        # - Allows priority-based processing (e.g., premium users)
        # - Better resource utilization with controlled concurrency
        # - Request timeout management
        # - Queue depth monitoring for scaling decisions
        #
        # Current implementation continues with direct processing:
        # ------------------------------------------------------------------------
        
        llm_call_func = partial(perform_chat_api_call, **cleaned_args)

        if request_data.stream:
            # Track LLM call
            llm_start_time = time.time()
            provider_call_start = time.time()
            try:
                raw_stream_iter = await current_loop.run_in_executor(None, llm_call_func)
                llm_latency = time.time() - llm_start_time
                metrics.track_llm_call(provider, model, llm_latency, success=True)
            except Exception as e:
                llm_latency = time.time() - llm_start_time
                metrics.track_llm_call(provider, model, llm_latency, success=False, error_type=type(e).__name__)
                raise
            if not (hasattr(raw_stream_iter, "__aiter__") or hasattr(raw_stream_iter, "__iter__")):
                raise HTTPException(status_code=status.HTTP_502_BAD_GATEWAY, detail="Provider did not return a valid stream.")

            # Use the improved streaming handler with timeout and heartbeat
            async def save_callback(full_reply: str):
                """Callback to save the assistant's reply after streaming completes."""
                if full_reply and final_conversation_id:
                    asst_name = character_card_for_context.get("name", "Assistant") if character_card_for_context else "Assistant"
                    logger.info(f"Saving assistant reply (len {len(full_reply)}) for conv_id {final_conversation_id}")
                    # Use transaction for atomic save
                    await _save_message_turn_to_db(
                        chat_db, 
                        final_conversation_id, 
                        {"role": "assistant", "name": asst_name, "content": full_reply},
                        use_transaction=True
                    )
                else:
                    logger.info(f"No assistant reply or conv_id to save for conv_id {final_conversation_id}.")

            # Create streaming response with timeout and heartbeat support
            # Wrap the generator with metrics tracking
            async def tracked_streaming_generator():
                async with metrics.track_streaming(final_conversation_id) as stream_tracker:
                    generator = create_streaming_response_with_timeout(
                        stream=raw_stream_iter,
                        conversation_id=final_conversation_id,
                        model_name=request_data.model,
                        save_callback=save_callback,
                        idle_timeout=300,  # 5 minutes
                        heartbeat_interval=30  # 30 seconds
                    )
                    async for chunk in generator:
                        # Track chunks and heartbeats
                        if "heartbeat" in chunk:
                            stream_tracker.add_heartbeat()
                        else:
                            stream_tracker.add_chunk()
                        yield chunk
            
            streaming_generator = tracked_streaming_generator()
            
            return StreamingResponse(
                streaming_generator, 
                media_type="text/event-stream",
                headers={
                    "Cache-Control": "no-cache",
                    "Connection": "keep-alive",
                    "X-Accel-Buffering": "no"  # Disable nginx buffering
                }
            )

        else: # Non-streaming
            # Track LLM call
            llm_start_time = time.time()
            try:
                llm_response = await current_loop.run_in_executor(None, llm_call_func)
                llm_latency = time.time() - llm_start_time
                metrics.track_llm_call(provider, model, llm_latency, success=True)
                
                # Record success with provider manager
                if provider_manager:
                    provider_manager.record_success(selected_provider, llm_latency)
                    
            except Exception as e:
                llm_latency = time.time() - llm_start_time
                metrics.track_llm_call(provider, model, llm_latency, success=False, error_type=type(e).__name__)
                
                # Record failure with provider manager
                if provider_manager:
                    provider_manager.record_failure(selected_provider, e)
                    
                    # Try failover provider on certain errors
                    if isinstance(e, (ChatProviderError, ChatAPIError)):
                        fallback_provider = provider_manager.get_available_provider(exclude=[selected_provider])
                        if fallback_provider:
                            logger.warning(f"Trying fallback provider {fallback_provider} after {selected_provider} failed")
                            # Fix: chat_api_call expects 'api_endpoint', not 'api_provider'
                            cleaned_args['api_endpoint'] = fallback_provider
                            llm_call_func = partial(perform_chat_api_call, **cleaned_args)
                            
                            try:
                                llm_response = await current_loop.run_in_executor(None, llm_call_func)
                                provider_manager.record_success(fallback_provider, time.time() - llm_start_time)
                                metrics.track_llm_call(fallback_provider, model, time.time() - llm_start_time, success=True)
                            except Exception as fallback_error:
                                provider_manager.record_failure(fallback_provider, fallback_error)
                                raise fallback_error
                        else:
                            raise
                else:
                    raise
            
            content_to_save: Optional[str] = None
            if isinstance(llm_response, dict): # OpenAI-like
                content_to_save = llm_response.get("choices", [{}])[0].get("message", {}).get("content")
                
                # Track token usage if available
                usage = llm_response.get("usage")
                if usage:
                    prompt_tokens = usage.get("prompt_tokens", 0)
                    completion_tokens = usage.get("completion_tokens", 0)
                    metrics.track_tokens(
                        prompt_tokens=prompt_tokens,
                        completion_tokens=completion_tokens,
                        model=model,
                        provider=provider
                    )
            elif isinstance(llm_response, str):
                content_to_save = llm_response

            if content_to_save:
                asst_name = character_card_for_context.get("name", "Assistant") if character_card_for_context else "Assistant"
                await _save_message_turn_to_db(chat_db, final_conversation_id, {"role": "assistant", "name": asst_name, "content": content_to_save}, use_transaction=True)

            # Use CPU-bound handler for large JSON encoding
            if isinstance(llm_response, dict) and len(str(llm_response)) > 10000:
                # Large response - use CPU handler
                encoded_json = await process_large_json_async(llm_response)
                encoded_payload = json.loads(encoded_json)
            else:
                # Small response - process inline
                encoded_payload = await current_loop.run_in_executor(None, jsonable_encoder, llm_response)
            if isinstance(encoded_payload, dict): # Ensure it's a dict to add custom fields
                encoded_payload["tldw_conversation_id"] = final_conversation_id
                
                # Track response size
                response_size = len(json.dumps(encoded_payload))
                metrics.metrics.response_size_bytes.record(
                    response_size,
                    {
                        "provider": provider,
                        "model": model,
                        "streaming": "false"
                    }
                )
            # Log successful response
            if audit_service and context:
                await audit_service.log_event(
                    event_type=AuditEventType.API_RESPONSE,
                    context=context,
                    action="chat_completion_success",
                    result="success",
                    metadata={
                        "conversation_id": final_conversation_id,
                        "provider": selected_provider,
                        "model": model,
                        "streaming": False
                    }
                )
            
            return JSONResponse(content=encoded_payload)

    # --- Exception Handling --- Issue 8 Fix: Mask internal details in 5xx
    except HTTPException as e_http:
        if e_http.status_code >= 500: logger.error(f"HTTPException (Server Error): {e_http.status_code} - {e_http.detail}", exc_info=True)
        else: logger.warning(f"HTTPException (Client Error): {e_http.status_code} - {e_http.detail}")
        raise e_http # Re-raise, details are assumed to be client-safe or intentionally set

    except (ChatAuthenticationError, ChatRateLimitError, ChatBadRequestError, ChatConfigurationError, ChatProviderError, ChatAPIError) as e_chat:
        # Log audit event for chat error
        if audit_service and context:
            await audit_service.log_event(
                event_type=AuditEventType.API_ERROR,
                context=context,
                action="chat_error",
                result="failure",
                metadata={
                    "error_type": type(e_chat).__name__,
                    "error_message": str(e_chat),
                    "provider": provider,
                    "model": model
                }
            )
        status_code_map = { ChatAuthenticationError: 401, ChatRateLimitError: 429, ChatBadRequestError: 400,
                            ChatConfigurationError: 503, ChatProviderError: getattr(e_chat, 'status_code', 502),
                            ChatAPIError: getattr(e_chat, 'status_code', 500) }
        err_status = status_code_map.get(type(e_chat), 500)
        logger.error(f"Chat Library Error: {type(e_chat).__name__} - '{e_chat.message}' (Provider: {e_chat.provider}, UpstreamStatus: {getattr(e_chat, 'status_code', 'N/A')})", exc_info=True)
        # Standardize error messages - never expose internal details for 5xx errors
        if err_status < 500:
            # Client errors can have more detail
            client_detail = e_chat.message
        else:
            # Server errors should be generic
            if err_status == 502:
                client_detail = "The chat service provider is currently unavailable."
            elif err_status == 503:
                client_detail = "The chat service is temporarily unavailable."
            elif err_status == 504:
                client_detail = "The chat service request timed out."
            else:
                client_detail = "An internal server error occurred."
        raise HTTPException(status_code=err_status, detail=client_detail)

    except (InputError, ConflictError, CharactersRAGDBError) as e_db:
        logger.error(f"Database Error: {type(e_db).__name__} - {str(e_db)}", exc_info=True)
        err_status = status.HTTP_400_BAD_REQUEST if isinstance(e_db, InputError) else \
                     status.HTTP_409_CONFLICT if isinstance(e_db, ConflictError) else \
                     status.HTTP_500_INTERNAL_SERVER_ERROR
        # Standardize database error messages
        if err_status < 500:
            client_detail = str(e_db)  # Client errors can have detail
        else:
            client_detail = "A database error occurred. Please try again later."
        raise HTTPException(status_code=err_status, detail=client_detail)

    except Exception as e_final:
        logger.critical(f"Unexpected Critical Error in /completions: {type(e_final).__name__} - {str(e_final)}", exc_info=True)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="An unexpected internal server error occurred.")


# ---------------------------------------------------------------------------
# Chat Dictionary Endpoints
# ---------------------------------------------------------------------------
from tldw_Server_API.app.api.v1.schemas.chat_dictionary_schemas import (
    ChatDictionaryCreate,
    ChatDictionaryUpdate,
    ChatDictionaryResponse,
    ChatDictionaryWithEntries,
    DictionaryEntryCreate,
    DictionaryEntryUpdate,
    DictionaryEntryResponse,
    ProcessTextRequest,
    ProcessTextResponse,
    ImportDictionaryRequest,
    ImportDictionaryResponse,
    ExportDictionaryResponse,
    DictionaryListResponse,
    EntryListResponse,
    DictionaryStatistics,
)
from tldw_Server_API.app.core.Character_Chat.chat_dictionary import (
    ChatDictionaryService,
    TokenBudgetExceededWarning,
)
import time
import warnings


@router.post("/dictionaries", response_model=ChatDictionaryResponse, status_code=status.HTTP_201_CREATED,
             summary="Create a new chat dictionary", tags=["Chat Dictionaries"])
async def create_chat_dictionary(
    dictionary: ChatDictionaryCreate,
    db: CharactersRAGDB = Depends(get_chacha_db_for_user)
) -> ChatDictionaryResponse:
    """
    Create a new chat dictionary for pattern-based text replacements.
    
    Dictionaries allow you to define patterns (literal or regex) that will be
    automatically replaced in chat messages.
    """
    try:
        service = ChatDictionaryService(db)
        dict_id = service.create_dictionary(dictionary.name, dictionary.description)
        
        # Fetch the created dictionary
        dict_data = service.get_dictionary(dict_id)
        if not dict_data:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Dictionary created but could not be retrieved"
            )
        
        # Get entry count
        entries = service.get_entries(dictionary_id=dict_id)
        dict_data['entry_count'] = len(entries)
        
        return ChatDictionaryResponse(**dict_data)
        
    except ConflictError as e:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=str(e))
    except Exception as e:
        logger.error(f"Error creating dictionary: {e}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))


@router.get("/dictionaries", response_model=DictionaryListResponse,
            summary="List all chat dictionaries", tags=["Chat Dictionaries"])
async def list_chat_dictionaries(
    include_inactive: bool = Query(False, description="Include inactive dictionaries"),
    db: CharactersRAGDB = Depends(get_chacha_db_for_user)
) -> DictionaryListResponse:
    """List all chat dictionaries for the current user."""
    try:
        service = ChatDictionaryService(db)
        dictionaries = service.list_dictionaries(include_inactive=include_inactive)
        
        # Add entry counts
        for dict_data in dictionaries:
            entries = service.get_entries(dictionary_id=dict_data['id'])
            dict_data['entry_count'] = len(entries)
        
        active_count = sum(1 for d in dictionaries if d.get('is_active', True))
        inactive_count = len(dictionaries) - active_count
        
        dict_responses = [ChatDictionaryResponse(**d) for d in dictionaries]
        
        return DictionaryListResponse(
            dictionaries=dict_responses,
            total=len(dictionaries),
            active_count=active_count,
            inactive_count=inactive_count
        )
        
    except Exception as e:
        logger.error(f"Error listing dictionaries: {e}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))


@router.get("/dictionaries/{dictionary_id}", response_model=ChatDictionaryWithEntries,
            summary="Get dictionary with entries", tags=["Chat Dictionaries"])
async def get_chat_dictionary(
    dictionary_id: int,
    db: CharactersRAGDB = Depends(get_chacha_db_for_user)
) -> ChatDictionaryWithEntries:
    """Get a specific dictionary with all its entries."""
    try:
        service = ChatDictionaryService(db)
        
        dict_data = service.get_dictionary(dictionary_id)
        if not dict_data:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Dictionary not found")
        
        entries = service.get_entries(dictionary_id=dictionary_id)
        dict_data['entry_count'] = len(entries)
        
        entry_responses = [DictionaryEntryResponse(
            id=e.entry_id,
            dictionary_id=dictionary_id,
            key=e.raw_key,
            content=e.content,
            probability=e.probability,
            group=e.group,
            timed_effects=e.timed_effects,
            max_replacements=e.max_replacements,
            is_regex=e.is_regex,
            created_at=datetime.now(),  # These would come from DB in production
            updated_at=datetime.now()
        ) for e in entries]
        
        return ChatDictionaryWithEntries(
            **dict_data,
            entries=entry_responses
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting dictionary: {e}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))


@router.put("/dictionaries/{dictionary_id}", response_model=ChatDictionaryResponse,
            summary="Update a dictionary", tags=["Chat Dictionaries"])
async def update_chat_dictionary(
    dictionary_id: int,
    update: ChatDictionaryUpdate,
    db: CharactersRAGDB = Depends(get_chacha_db_for_user)
) -> ChatDictionaryResponse:
    """Update a dictionary's metadata."""
    try:
        service = ChatDictionaryService(db)
        
        success = service.update_dictionary(
            dictionary_id,
            name=update.name,
            description=update.description,
            is_active=update.is_active
        )
        
        if not success:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Dictionary not found")
        
        dict_data = service.get_dictionary(dictionary_id)
        entries = service.get_entries(dictionary_id=dictionary_id)
        dict_data['entry_count'] = len(entries)
        
        return ChatDictionaryResponse(**dict_data)
        
    except ConflictError as e:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=str(e))
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error updating dictionary: {e}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))


@router.delete("/dictionaries/{dictionary_id}", status_code=status.HTTP_204_NO_CONTENT,
               summary="Delete a dictionary", tags=["Chat Dictionaries"])
async def delete_chat_dictionary(
    dictionary_id: int,
    hard_delete: bool = Query(False, description="Permanently delete instead of soft delete"),
    db: CharactersRAGDB = Depends(get_chacha_db_for_user)
):
    """Delete a dictionary (soft delete by default)."""
    try:
        service = ChatDictionaryService(db)
        success = service.delete_dictionary(dictionary_id, hard_delete=hard_delete)
        
        if not success:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Dictionary not found")
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting dictionary: {e}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))


# --- Dictionary Entry Endpoints ---

@router.post("/dictionaries/{dictionary_id}/entries", response_model=DictionaryEntryResponse,
             status_code=status.HTTP_201_CREATED, summary="Add entry to dictionary", tags=["Chat Dictionaries"])
async def add_dictionary_entry(
    dictionary_id: int,
    entry: DictionaryEntryCreate,
    db: CharactersRAGDB = Depends(get_chacha_db_for_user)
) -> DictionaryEntryResponse:
    """
    Add a new entry to a dictionary.
    
    The key can be:
    - A literal string: "hello"
    - A regex pattern: "/hel+o/i" (with optional flags: i=ignore case, m=multiline, s=dotall)
    """
    try:
        service = ChatDictionaryService(db)
        
        # Check dictionary exists
        dict_data = service.get_dictionary(dictionary_id)
        if not dict_data:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Dictionary not found")
        
        timed_effects_dict = entry.timed_effects.model_dump() if entry.timed_effects else None
        
        entry_id = service.add_entry(
            dictionary_id,
            entry.key,
            entry.content,
            entry.probability,
            entry.group,
            timed_effects_dict,
            entry.max_replacements
        )
        
        # Create response
        return DictionaryEntryResponse(
            id=entry_id,
            dictionary_id=dictionary_id,
            key=entry.key,
            content=entry.content,
            probability=entry.probability,
            group=entry.group,
            timed_effects=entry.timed_effects,
            max_replacements=entry.max_replacements,
            is_regex=entry.key.startswith("/") and "/" in entry.key[1:],
            created_at=datetime.now(),
            updated_at=datetime.now()
        )
        
    except InputError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error adding dictionary entry: {e}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))


@router.get("/dictionaries/{dictionary_id}/entries", response_model=EntryListResponse,
            summary="List dictionary entries", tags=["Chat Dictionaries"])
async def list_dictionary_entries(
    dictionary_id: int,
    group: Optional[str] = Query(None, description="Filter by group"),
    db: CharactersRAGDB = Depends(get_chacha_db_for_user)
) -> EntryListResponse:
    """List all entries in a dictionary, optionally filtered by group."""
    try:
        service = ChatDictionaryService(db)
        
        # Check dictionary exists
        dict_data = service.get_dictionary(dictionary_id)
        if not dict_data:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Dictionary not found")
        
        entries = service.get_entries(dictionary_id=dictionary_id, group=group)
        
        entry_responses = [DictionaryEntryResponse(
            id=e.entry_id,
            dictionary_id=dictionary_id,
            key=e.raw_key,
            content=e.content,
            probability=e.probability,
            group=e.group,
            timed_effects=e.timed_effects,
            max_replacements=e.max_replacements,
            is_regex=e.is_regex,
            created_at=datetime.now(),
            updated_at=datetime.now()
        ) for e in entries]
        
        return EntryListResponse(
            entries=entry_responses,
            total=len(entries),
            dictionary_id=dictionary_id,
            group=group
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error listing dictionary entries: {e}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))


@router.put("/dictionaries/entries/{entry_id}", response_model=DictionaryEntryResponse,
            summary="Update dictionary entry", tags=["Chat Dictionaries"])
async def update_dictionary_entry(
    entry_id: int,
    update: DictionaryEntryUpdate,
    db: CharactersRAGDB = Depends(get_chacha_db_for_user)
) -> DictionaryEntryResponse:
    """Update a dictionary entry."""
    try:
        service = ChatDictionaryService(db)
        
        timed_effects_dict = update.timed_effects.model_dump() if update.timed_effects else None
        
        success = service.update_entry(
            entry_id,
            key=update.key,
            content=update.content,
            probability=update.probability,
            group=update.group,
            timed_effects=timed_effects_dict,
            max_replacements=update.max_replacements
        )
        
        if not success:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Entry not found")
        
        # Fetch updated entry (simplified for now)
        # In production, we'd fetch from DB
        is_regex = update.key.startswith("/") and "/" in update.key[1:] if update.key else False
        
        return DictionaryEntryResponse(
            id=entry_id,
            dictionary_id=1,  # Would fetch from DB
            key=update.key or "updated",
            content=update.content or "updated",
            probability=update.probability or 100,
            group=update.group,
            timed_effects=update.timed_effects,
            max_replacements=update.max_replacements or 1,
            is_regex=is_regex,
            created_at=datetime.now(),
            updated_at=datetime.now()
        )
        
    except InputError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error updating dictionary entry: {e}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))


@router.delete("/dictionaries/entries/{entry_id}", status_code=status.HTTP_204_NO_CONTENT,
               summary="Delete dictionary entry", tags=["Chat Dictionaries"])
async def delete_dictionary_entry(
    entry_id: int,
    db: CharactersRAGDB = Depends(get_chacha_db_for_user)
):
    """Delete a dictionary entry."""
    try:
        service = ChatDictionaryService(db)
        success = service.delete_entry(entry_id)
        
        if not success:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Entry not found")
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting dictionary entry: {e}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))


# --- Text Processing Endpoint ---

@router.post("/dictionaries/process", response_model=ProcessTextResponse,
             summary="Process text through dictionaries", tags=["Chat Dictionaries"])
async def process_text_with_dictionaries(
    request: ProcessTextRequest,
    db: CharactersRAGDB = Depends(get_chacha_db_for_user)
) -> ProcessTextResponse:
    """
    Process text through active dictionaries to apply replacements.
    
    This endpoint applies pattern-based replacements defined in the user's
    active dictionaries to the provided text.
    """
    try:
        service = ChatDictionaryService(db)
        
        start_time = time.time()
        
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            
            processed_text, stats = service.process_text(
                request.text,
                dictionary_id=request.dictionary_id,
                group=request.group,
                max_iterations=request.max_iterations,
                token_budget=request.token_budget
            )
            
            # Check if token budget was exceeded
            token_budget_exceeded = any(
                issubclass(warning.category, TokenBudgetExceededWarning) for warning in w
            )
            if token_budget_exceeded:
                stats["token_budget_exceeded"] = True
        
        processing_time_ms = (time.time() - start_time) * 1000
        
        return ProcessTextResponse(
            original_text=request.text,
            processed_text=processed_text,
            replacements=stats.get("replacements", 0),
            iterations=stats.get("iterations", 0),
            entries_used=stats.get("entries_used", []),
            token_budget_exceeded=stats.get("token_budget_exceeded", False),
            processing_time_ms=processing_time_ms
        )
        
    except Exception as e:
        logger.error(f"Error processing text: {e}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))


# --- Import/Export Endpoints ---

@router.post("/dictionaries/import", response_model=ImportDictionaryResponse,
             status_code=status.HTTP_201_CREATED, summary="Import dictionary from markdown", 
             tags=["Chat Dictionaries"])
async def import_dictionary(
    import_request: ImportDictionaryRequest,
    db: CharactersRAGDB = Depends(get_chacha_db_for_user)
) -> ImportDictionaryResponse:
    """
    Import a dictionary from markdown format.
    
    Format example:
    ```
    key: value
    /regex/i: replacement
    
    ## Group Name
    grouped_key: grouped_value
    ```
    """
    try:
        service = ChatDictionaryService(db)
        
        # Write content to temporary file
        import tempfile
        with tempfile.NamedTemporaryFile(mode='w', suffix='.md', delete=False) as tmp:
            tmp.write(import_request.content)
            tmp_path = tmp.name
        
        try:
            dict_id = service.import_from_markdown(tmp_path, import_request.name)
            
            # Get statistics
            entries = service.get_entries(dictionary_id=dict_id)
            groups = list(set(e.group for e in entries if e.group))
            
            # Activate if requested
            if import_request.activate:
                service.update_dictionary(dict_id, is_active=True)
            
            return ImportDictionaryResponse(
                dictionary_id=dict_id,
                name=import_request.name,
                entries_imported=len(entries),
                groups_created=groups
            )
            
        finally:
            # Clean up temp file
            import os
            os.unlink(tmp_path)
            
    except InputError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))
    except ConflictError as e:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=str(e))
    except Exception as e:
        logger.error(f"Error importing dictionary: {e}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))


@router.get("/dictionaries/{dictionary_id}/export", response_model=ExportDictionaryResponse,
            summary="Export dictionary to markdown", tags=["Chat Dictionaries"])
async def export_dictionary(
    dictionary_id: int,
    db: CharactersRAGDB = Depends(get_chacha_db_for_user)
) -> ExportDictionaryResponse:
    """Export a dictionary to markdown format."""
    try:
        service = ChatDictionaryService(db)
        
        # Get dictionary info
        dict_data = service.get_dictionary(dictionary_id)
        if not dict_data:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Dictionary not found")
        
        # Export to temporary file
        import tempfile
        with tempfile.NamedTemporaryFile(mode='w', suffix='.md', delete=False) as tmp:
            tmp_path = tmp.name
        
        try:
            service.export_to_markdown(dictionary_id, tmp_path)
            
            # Read content
            with open(tmp_path, 'r') as f:
                content = f.read()
            
            # Get statistics
            entries = service.get_entries(dictionary_id=dictionary_id)
            groups = list(set(e.group for e in entries if e.group))
            
            return ExportDictionaryResponse(
                name=dict_data['name'],
                content=content,
                entry_count=len(entries),
                group_count=len(groups)
            )
            
        finally:
            # Clean up temp file
            import os
            os.unlink(tmp_path)
            
    except InputError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error exporting dictionary: {e}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))


@router.get("/dictionaries/{dictionary_id}/statistics", response_model=DictionaryStatistics,
            summary="Get dictionary statistics", tags=["Chat Dictionaries"])
async def get_dictionary_statistics(
    dictionary_id: int,
    db: CharactersRAGDB = Depends(get_chacha_db_for_user)
) -> DictionaryStatistics:
    """Get statistics for a dictionary."""
    try:
        service = ChatDictionaryService(db)
        
        dict_data = service.get_dictionary(dictionary_id)
        if not dict_data:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Dictionary not found")
        
        entries = service.get_entries(dictionary_id=dictionary_id)
        
        # Calculate statistics
        regex_count = sum(1 for e in entries if e.is_regex)
        literal_count = len(entries) - regex_count
        groups = list(set(e.group for e in entries if e.group))
        avg_probability = sum(e.probability for e in entries) / len(entries) if entries else 0
        
        return DictionaryStatistics(
            dictionary_id=dictionary_id,
            name=dict_data['name'],
            total_entries=len(entries),
            regex_entries=regex_count,
            literal_entries=literal_count,
            groups=groups,
            average_probability=avg_probability,
            total_usage_count=None,  # Would track in production
            last_used=None  # Would track in production
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting dictionary statistics: {e}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))


# ---------------------------------------------------------------------------
# Document Generator Endpoints
# ---------------------------------------------------------------------------
from tldw_Server_API.app.api.v1.schemas.document_generator_schemas import (
    DocumentType as DocType,
    GenerationStatus,
    GenerateDocumentRequest,
    GenerateDocumentResponse,
    AsyncGenerationResponse,
    JobStatusResponse,
    GeneratedDocument,
    DocumentListResponse,
    SavePromptConfigRequest,
    PromptConfigResponse,
    BulkGenerateRequest,
    BulkGenerateResponse,
    GenerationStatistics,
    DocumentGeneratorError
)
from tldw_Server_API.app.core.Chat.document_generator import (
    DocumentGeneratorService,
    DocumentType,
    GenerationStatus as GenStatus
)


@router.post("/documents/generate", response_model=Union[GenerateDocumentResponse, AsyncGenerationResponse],
             summary="Generate a document from conversation", tags=["Document Generator"])
async def generate_document(
    request: GenerateDocumentRequest,
    db: CharactersRAGDB = Depends(get_chacha_db_for_user)
) -> Union[GenerateDocumentResponse, AsyncGenerationResponse]:
    """Generate a document from a conversation."""
    try:
        service = DocumentGeneratorService(db)
        
        # Convert string enum to internal enum
        doc_type = DocumentType[request.document_type.value.upper()]
        
        if request.async_generation:
            # Create async job
            job_id = service.create_generation_job(
                conversation_id=request.conversation_id,
                document_type=doc_type,
                provider=request.provider,
                model=request.model,
                prompt_config={
                    "specific_message": request.specific_message,
                    "custom_prompt": request.custom_prompt
                }
            )
            
            return AsyncGenerationResponse(
                job_id=job_id,
                status=GenStatus.PENDING,
                conversation_id=request.conversation_id,
                document_type=request.document_type,
                created_at=datetime.datetime.utcnow(),
                message="Document generation job created"
            )
        else:
            # Synchronous generation
            content = service.generate_document(
                conversation_id=request.conversation_id,
                document_type=doc_type,
                provider=request.provider,
                model=request.model,
                api_key=request.api_key,
                specific_message=request.specific_message,
                custom_prompt=request.custom_prompt,
                stream=request.stream
            )
            
            if request.stream:
                # Return streaming response
                return StreamingResponse(content, media_type="text/plain")
            
            # Get the saved document
            docs = service.get_generated_documents(
                conversation_id=request.conversation_id,
                document_type=doc_type,
                limit=1
            )
            
            if not docs:
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail="Document generated but not saved"
                )
            
            doc = docs[0]
            return GenerateDocumentResponse(
                document_id=doc['id'],
                conversation_id=doc['conversation_id'],
                document_type=request.document_type,
                title=doc['title'],
                content=doc['content'],
                provider=doc['provider'],
                model=doc['model'],
                generation_time_ms=doc['generation_time_ms'],
                created_at=doc['created_at']
            )
            
    except InputError as e:
        logger.warning(f"Input error generating document: {e}")
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))
    except ChatAPIError as e:
        logger.error(f"API error generating document: {e}")
        raise HTTPException(status_code=status.HTTP_502_BAD_GATEWAY, detail=str(e))
    except Exception as e:
        logger.error(f"Unexpected error generating document: {e}", exc_info=True)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))


@router.get("/documents/jobs/{job_id}", response_model=JobStatusResponse,
            summary="Get generation job status", tags=["Document Generator"])
async def get_job_status(
    job_id: str,
    db: CharactersRAGDB = Depends(get_chacha_db_for_user)
) -> JobStatusResponse:
    """Get the status of a document generation job."""
    try:
        service = DocumentGeneratorService(db)
        job = service.get_job_status(job_id)
        
        if not job:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Job {job_id} not found"
            )
        
        # Calculate progress (simplified)
        progress = 0
        if job['status'] == GenStatus.PENDING.value:
            progress = 0
        elif job['status'] == GenStatus.IN_PROGRESS.value:
            progress = 50
        elif job['status'] in [GenStatus.COMPLETED.value, GenStatus.FAILED.value, GenStatus.CANCELLED.value]:
            progress = 100
        
        return JobStatusResponse(
            job_id=job['job_id'],
            conversation_id=job['conversation_id'],
            document_type=DocType(job['document_type']),
            status=GenStatus(job['status']),
            provider=job['provider'],
            model=job['model'],
            result_content=job['result_content'],
            error_message=job['error_message'],
            created_at=job['created_at'],
            started_at=job['started_at'],
            completed_at=job['completed_at'],
            progress_percentage=progress
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting job status: {e}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))


@router.delete("/documents/jobs/{job_id}", summary="Cancel generation job", tags=["Document Generator"])
async def cancel_job(
    job_id: str,
    db: CharactersRAGDB = Depends(get_chacha_db_for_user)
) -> Dict[str, str]:
    """Cancel a document generation job."""
    try:
        service = DocumentGeneratorService(db)
        
        job = service.get_job_status(job_id)
        if not job:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Job {job_id} not found"
            )
        
        if job['status'] in [GenStatus.COMPLETED.value, GenStatus.FAILED.value, GenStatus.CANCELLED.value]:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Job {job_id} is already {job['status']}"
            )
        
        success = service.update_job_status(job_id, GenStatus.CANCELLED)
        
        if not success:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to cancel job"
            )
        
        return {"message": f"Job {job_id} cancelled successfully"}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error cancelling job: {e}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))


@router.get("/documents", response_model=DocumentListResponse,
            summary="List generated documents", tags=["Document Generator"])
async def list_generated_documents(
    conversation_id: Optional[int] = Query(None, description="Filter by conversation ID"),
    document_type: Optional[DocType] = Query(None, description="Filter by document type"),
    limit: int = Query(50, ge=1, le=200, description="Maximum number of documents"),
    db: CharactersRAGDB = Depends(get_chacha_db_for_user)
) -> DocumentListResponse:
    """List previously generated documents."""
    try:
        service = DocumentGeneratorService(db)
        
        # Convert string enum to internal enum if provided
        doc_type = DocumentType[document_type.value.upper()] if document_type else None
        
        documents = service.get_generated_documents(
            conversation_id=conversation_id,
            document_type=doc_type,
            limit=limit
        )
        
        # Convert to response models
        doc_responses = [GeneratedDocument(**doc) for doc in documents]
        
        return DocumentListResponse(
            documents=doc_responses,
            total=len(doc_responses),
            conversation_id=conversation_id,
            document_type=document_type
        )
        
    except Exception as e:
        logger.error(f"Error listing generated documents: {e}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))


@router.get("/documents/{document_id}", response_model=GeneratedDocument,
            summary="Get generated document", tags=["Document Generator"])
async def get_generated_document(
    document_id: int,
    db: CharactersRAGDB = Depends(get_chacha_db_for_user)
) -> GeneratedDocument:
    """Get a specific generated document."""
    try:
        service = DocumentGeneratorService(db)
        
        documents = service.get_generated_documents(limit=1)
        
        # Find the specific document
        doc = next((d for d in documents if d['id'] == document_id), None)
        
        if not doc:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Document {document_id} not found"
            )
        
        return GeneratedDocument(**doc)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting document {document_id}: {e}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))


@router.delete("/documents/{document_id}", summary="Delete generated document", tags=["Document Generator"])
async def delete_generated_document(
    document_id: int,
    db: CharactersRAGDB = Depends(get_chacha_db_for_user)
) -> Dict[str, str]:
    """Delete a generated document."""
    try:
        service = DocumentGeneratorService(db)
        
        success = service.delete_generated_document(document_id)
        
        if not success:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Document {document_id} not found"
            )
        
        return {"message": f"Document {document_id} deleted successfully"}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting document {document_id}: {e}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))


@router.post("/documents/prompts", response_model=PromptConfigResponse,
             summary="Save custom prompt configuration", tags=["Document Generator"])
async def save_prompt_config(
    config: SavePromptConfigRequest,
    db: CharactersRAGDB = Depends(get_chacha_db_for_user)
) -> PromptConfigResponse:
    """Save a custom prompt configuration for a document type."""
    try:
        service = DocumentGeneratorService(db)
        
        # Convert string enum to internal enum
        doc_type = DocumentType[config.document_type.value.upper()]
        
        success = service.save_user_prompt_config(
            document_type=doc_type,
            system_prompt=config.system_prompt,
            user_prompt=config.user_prompt,
            temperature=config.temperature,
            max_tokens=config.max_tokens
        )
        
        if not success:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to save prompt configuration"
            )
        
        return PromptConfigResponse(
            document_type=config.document_type,
            system_prompt=config.system_prompt,
            user_prompt=config.user_prompt,
            temperature=config.temperature,
            max_tokens=config.max_tokens,
            is_custom=True,
            created_at=datetime.datetime.utcnow(),
            updated_at=datetime.datetime.utcnow()
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error saving prompt config: {e}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))


@router.get("/documents/prompts/{document_type}", response_model=PromptConfigResponse,
            summary="Get prompt configuration", tags=["Document Generator"])
async def get_prompt_config(
    document_type: DocType,
    db: CharactersRAGDB = Depends(get_chacha_db_for_user)
) -> PromptConfigResponse:
    """Get the prompt configuration for a document type."""
    try:
        service = DocumentGeneratorService(db)
        
        # Convert string enum to internal enum
        doc_type = DocumentType[document_type.value.upper()]
        
        config = service.get_user_prompt_config(doc_type)
        
        # Check if it's a custom config by trying to fetch from database
        is_custom = False
        try:
            with db.get_connection() as conn:
                cursor = conn.execute(
                    "SELECT 1 FROM user_prompts WHERE document_type = ? AND is_active = 1",
                    (doc_type.value,)
                )
                is_custom = cursor.fetchone() is not None
        except:
            pass
        
        return PromptConfigResponse(
            document_type=document_type,
            system_prompt=config['system'],
            user_prompt=config['user'],
            temperature=config['temperature'],
            max_tokens=config['max_tokens'],
            is_custom=is_custom,
            created_at=None,
            updated_at=None
        )
        
    except Exception as e:
        logger.error(f"Error getting prompt config: {e}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))


@router.post("/documents/bulk", response_model=BulkGenerateResponse,
             summary="Bulk generate documents", tags=["Document Generator"])
async def bulk_generate_documents(
    request: BulkGenerateRequest,
    db: CharactersRAGDB = Depends(get_chacha_db_for_user)
) -> BulkGenerateResponse:
    """Generate multiple documents in bulk (async)."""
    try:
        service = DocumentGeneratorService(db)
        
        job_ids = []
        total_jobs = len(request.conversation_ids) * len(request.document_types)
        
        for conv_id in request.conversation_ids:
            for doc_type_str in request.document_types:
                # Convert string enum to internal enum
                doc_type = DocumentType[doc_type_str.value.upper()]
                
                # Create job for each combination
                job_id = service.create_generation_job(
                    conversation_id=conv_id,
                    document_type=doc_type,
                    provider=request.provider,
                    model=request.model,
                    prompt_config={}
                )
                job_ids.append(job_id)
        
        # Estimate time (simplified - 10 seconds per document)
        estimated_time = total_jobs * 10
        
        return BulkGenerateResponse(
            total_jobs=total_jobs,
            job_ids=job_ids,
            estimated_time_seconds=estimated_time,
            message=f"Created {total_jobs} generation jobs"
        )
        
    except Exception as e:
        logger.error(f"Error creating bulk generation jobs: {e}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))


@router.get("/documents/statistics", response_model=GenerationStatistics,
            summary="Get generation statistics", tags=["Document Generator"])
async def get_generation_statistics(
    db: CharactersRAGDB = Depends(get_chacha_db_for_user)
) -> GenerationStatistics:
    """Get statistics about document generation."""
    try:
        service = DocumentGeneratorService(db)
        
        # Get all documents for statistics
        all_docs = service.get_generated_documents(limit=1000)
        
        if not all_docs:
            return GenerationStatistics(
                total_documents=0,
                by_type={},
                by_provider={},
                average_generation_time_ms=0,
                total_tokens_used=None,
                last_generated=None,
                most_used_model=None
            )
        
        # Calculate statistics
        by_type = {}
        by_provider = {}
        total_time = 0
        total_tokens = 0
        models = {}
        
        for doc in all_docs:
            # Count by type
            doc_type = doc['document_type']
            by_type[doc_type] = by_type.get(doc_type, 0) + 1
            
            # Count by provider
            provider = doc['provider']
            by_provider[provider] = by_provider.get(provider, 0) + 1
            
            # Sum generation time
            total_time += doc.get('generation_time_ms', 0)
            
            # Sum tokens
            if doc.get('token_count'):
                total_tokens += doc['token_count']
            
            # Count models
            model = doc['model']
            models[model] = models.get(model, 0) + 1
        
        # Find most used model
        most_used_model = max(models, key=models.get) if models else None
        
        # Get last generated
        last_doc = max(all_docs, key=lambda d: d['created_at'])
        
        return GenerationStatistics(
            total_documents=len(all_docs),
            by_type=by_type,
            by_provider=by_provider,
            average_generation_time_ms=total_time / len(all_docs) if all_docs else 0,
            total_tokens_used=total_tokens if total_tokens > 0 else None,
            last_generated=last_doc['created_at'],
            most_used_model=most_used_model
        )
        
    except Exception as e:
        logger.error(f"Error getting generation statistics: {e}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))


#
# End of chat.py
#######################################################################################################################