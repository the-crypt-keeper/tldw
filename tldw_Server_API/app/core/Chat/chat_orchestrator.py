# chat_orchestrator.py
# Description: Core chat orchestration functions for LLM interactions
"""
This module provides the core chat orchestration functionality, including
the main chat_api_call dispatcher and the chat function for multimodal
interactions with various LLM providers.
"""
#
# Imports
import logging
import os
import time
from typing import Any, Dict, List, Optional, Union
#
# 3rd-party Libraries
import requests
from loguru import logger
#
# Local Imports
from tldw_Server_API.app.api.v1.schemas.chat_request_schemas import ResponseFormat
from tldw_Server_API.app.core.Chat.Chat_Deps import (
    ChatAPIError,
    ChatAuthenticationError,
    ChatBadRequestError,
    ChatConfigurationError,
    ChatProviderError,
    ChatRateLimitError
)
from tldw_Server_API.app.core.Chat.provider_config import (
    API_CALL_HANDLERS,
    PROVIDER_PARAM_MAP
)
from tldw_Server_API.app.core.Metrics.metrics_logger import log_counter, log_histogram
from tldw_Server_API.app.core.config import load_and_log_configs
#
####################################################################################################
#
# Token Counting
#

def approximate_token_count(history):
    """
    Approximate the token count for a chat history.
    
    Args:
        history: Chat history in various formats
        
    Returns:
        Approximate token count
    """
    try:
        total_text = ''
        for user_msg, bot_msg in history:
            if user_msg:
                total_text += user_msg + ' '
            if bot_msg:
                total_text += bot_msg + ' '
        total_tokens = len(total_text.split())
        return total_tokens
    except Exception as e:
        logging.error(f"Error calculating token count: {str(e)}")
        return 0

#
####################################################################################################
#
# Main Chat API Call Dispatcher
#

def chat_api_call(
    api_endpoint: str,
    messages_payload: List[Dict[str, Any]], # CHANGED from input_data, prompt
    api_key: Optional[str] = None,
    temp: Optional[float] = None,
    system_message: Optional[str] = None, # Still passed separately, some providers might use it, others expect it in messages_payload
    streaming: Optional[bool] = None,
    minp: Optional[float] = None,
    maxp: Optional[float] = None, # Often maps to top_p
    model: Optional[str] = None,
    topk: Optional[int] = None,
    topp: Optional[float] = None, # Often maps to top_p
    logprobs: Optional[bool] = None,
    top_logprobs: Optional[int] = None,
    logit_bias: Optional[Dict[str, float]] = None,
    presence_penalty: Optional[float] = None,
    frequency_penalty: Optional[float] = None,
    tools: Optional[List[Dict[str, Any]]] = None,
    tool_choice: Optional[Union[str, Dict[str, Any]]] = None,
    max_tokens: Optional[int] = None,
    seed: Optional[int] = None,
    stop: Optional[Union[str, List[str]]] = None,
    response_format: Optional[Dict[str, str]] = None,  # Expects {'type': 'text' | 'json_object'}
    n: Optional[int] = None,
    user_identifier: Optional[str] = None  # Renamed from 'user' to avoid conflict with 'user' role in messages
    ):
    """
    Acts as a unified dispatcher to call various LLM API providers.

    This function routes chat requests to the appropriate LLM provider based on
    `api_endpoint`. It uses `API_CALL_HANDLERS` to find the correct handler
    function and `PROVIDER_PARAM_MAP` to translate generic parameters to
    provider-specific ones.

    Args:
        api_endpoint: The identifier for the target LLM provider (e.g., "openai", "anthropic").
        messages_payload: A list of message objects (OpenAI format: `{'role': ..., 'content': ...}`)
                          representing the conversation history and current user message.
        api_key: The API key for the specified provider.
        temp: Temperature for sampling, controlling randomness.
        system_message: An optional system-level instruction for the LLM. How this is
                        used depends on the provider; some prepend it to messages, others
                        have a dedicated parameter.
        streaming: Whether to stream the response from the LLM.
        minp: Minimum probability for token sampling (nucleus sampling related).
        maxp: Maximum probability for token sampling (often maps to `top_p`).
        model: The specific model to use for the LLM provider.
        topk: Top-K sampling parameter.
        topp: Top-P (nucleus) sampling parameter.
        logprobs: Whether to return log probabilities of tokens.
        top_logprobs: Number of top log probabilities to return.
        logit_bias: A dictionary to bias token generation probabilities.
        presence_penalty: Penalty for new tokens based on their presence in the text so far.
        frequency_penalty: Penalty for new tokens based on their frequency in the text so far.
        tools: A list of tools the model may call.
        tool_choice: Controls which tool the model should call.
        max_tokens: The maximum number of tokens to generate in the response.
        seed: A seed for deterministic generation, if supported.
        stop: A string or list of strings that, when generated, will cause the LLM to stop.
        response_format: Specifies the format of the response (e.g., `{'type': 'json_object'}`).
        n: The number of chat completion choices to generate.
        user_identifier: An identifier for the end-user, for tracking or moderation purposes.

    Returns:
        The LLM's response. This can be a string for non-streaming responses or
        a generator for streaming responses. The exact type depends on the
        underlying provider's handler function.

    Raises:
        ValueError: If the `api_endpoint` is unsupported or if there's a parameter issue.
        ChatAuthenticationError: If authentication with the provider fails (e.g., invalid API key).
        ChatRateLimitError: If the provider's rate limit is exceeded.
        ChatBadRequestError: If the request to the provider is malformed or invalid.
        ChatProviderError: If the provider's server returns an error or there's a network issue.
        ChatConfigurationError: If there's a configuration issue for the specified provider.
        ChatAPIError: For other unexpected API-related errors.
        requests.exceptions.HTTPError: Propagated from underlying HTTP requests if not caught and re-raised.
        requests.exceptions.RequestException: For network errors during the request.
    """
    endpoint_lower = api_endpoint.lower()
    logging.info(f"Chat API Call - Routing to endpoint: {endpoint_lower}")
    log_counter("chat_api_call_attempt", labels={"api_endpoint": endpoint_lower})
    start_time = time.time()

    handler = API_CALL_HANDLERS.get(endpoint_lower)
    if not handler:
        logging.error(f"Unsupported API endpoint requested: {api_endpoint}")
        raise ValueError(f"Unsupported API endpoint: {api_endpoint}")

    params_map = PROVIDER_PARAM_MAP.get(endpoint_lower, {})
    call_kwargs = {}

    # Construct kwargs for the handler function based on the map
    # This requires careful mapping and ensuring the handler functions are adapted.

    # Generic parameters available from chat_api_call signature
    available_generic_params = {
        'api_key': api_key,
        'messages_payload': messages_payload, # This is the core change
        'temp': temp,
        'system_message': system_message,
        'streaming': streaming,
        'minp': minp,
        'maxp': maxp, # Will be mapped to top_p by some providers
        'model': model,
        'topk': topk,
        'topp': topp, # Will be mapped to top_p by some providers
        'logprobs': logprobs,
        'top_logprobs': top_logprobs,
        'logit_bias': logit_bias,
        'presence_penalty': presence_penalty,
        'frequency_penalty': frequency_penalty,
        'tools': tools,
        'tool_choice': tool_choice,
        'max_tokens': max_tokens,
        'seed': seed,
        'stop': stop,
        'response_format': response_format,
        'n': n,
        'user_identifier': user_identifier
    }

    for generic_param_name, provider_param_name in params_map.items():
        if generic_param_name in available_generic_params and available_generic_params[generic_param_name] is not None:
            call_kwargs[provider_param_name] = available_generic_params[generic_param_name]
        if generic_param_name == 'prompt' and endpoint_lower == 'cohere':
             pass # Specific handling for Cohere's prompt is assumed to be within chat_with_cohere

    if call_kwargs.get(params_map.get('api_key', 'api_key')) and isinstance(call_kwargs.get(params_map.get('api_key', 'api_key')), str) and len(call_kwargs.get(params_map.get('api_key', 'api_key'))) > 8:
         logging.info(f"Debug - Chat API Call - API Key: {call_kwargs[params_map.get('api_key', 'api_key')][:4]}...{call_kwargs[params_map.get('api_key', 'api_key')][-4:]}")

    try:
        logging.debug(f"Calling handler {handler.__name__} with kwargs: { {k: (type(v) if k != params_map.get('api_key') else 'key_hidden') for k,v in call_kwargs.items()} }")
        response = handler(**call_kwargs)

        call_duration = time.time() - start_time
        log_histogram("chat_api_call_duration", call_duration, labels={"api_endpoint": endpoint_lower})
        log_counter("chat_api_call_success", labels={"api_endpoint": endpoint_lower})

        if isinstance(response, str):
             logging.debug(f"Debug - Chat API Call - Response (first 500 chars): {response[:500]}...")
        elif hasattr(response, '__iter__') and not isinstance(response, (str, bytes, dict)):
             logging.debug(f"Debug - Chat API Call - Response: Streaming Generator")
        else:
             logging.debug(f"Debug - Chat API Call - Response Type: {type(response)}")
        return response

    # --- Exception Mapping (copied from your original, ensure it's still relevant) ---
    except requests.exceptions.HTTPError as e:
        status_code = getattr(e.response, 'status_code', 500)
        error_text = getattr(e.response, 'text', str(e))
        log_message_base = f"{endpoint_lower} API call failed with status {status_code}"

        # Log safely first
        try:
            logging.error("%s. Details: %s", log_message_base, error_text[:500], exc_info=False)
        except Exception as log_e:
            logging.error(f"Error during logging HTTPError details: {log_e}")

        detail_message = f"API call to {endpoint_lower} failed with status {status_code}. Response: {error_text[:200]}"
        if status_code == 401:
            raise ChatAuthenticationError(provider=endpoint_lower,
                                          message=f"Authentication failed for {endpoint_lower}. Check API key. Detail: {error_text[:200]}")
        elif status_code == 429:
            raise ChatRateLimitError(provider=endpoint_lower,
                                     message=f"Rate limit exceeded for {endpoint_lower}. Detail: {error_text[:200]}")
        elif 400 <= status_code < 500:
            raise ChatBadRequestError(provider=endpoint_lower,
                                      message=f"Bad request to {endpoint_lower} (Status {status_code}). Detail: {error_text[:200]}")
        elif 500 <= status_code < 600:
            raise ChatProviderError(provider=endpoint_lower,
                                    message=f"Error from {endpoint_lower} server (Status {status_code}). Detail: {error_text[:200]}",
                                    status_code=status_code)
        else:
            raise ChatAPIError(provider=endpoint_lower,
                               message=f"Unexpected HTTP status {status_code} from {endpoint_lower}. Detail: {error_text[:200]}",
                               status_code=status_code)
    except requests.exceptions.RequestException as e:
        logging.error(f"Network error connecting to {endpoint_lower}: {e}", exc_info=False)
        raise ChatProviderError(provider=endpoint_lower, message=f"Network error: {e}", status_code=504)
    except (ChatAuthenticationError, ChatRateLimitError, ChatBadRequestError, ChatConfigurationError, ChatProviderError,
            ChatAPIError) as e_chat_direct:
        # This catches cases where the handler itself has already processed an error
        # (e.g. non-HTTP error, or it decided to raise a specific Chat*Error type)
        # and raises one of our custom exceptions.
        logging.error(
            f"Handler for {endpoint_lower} directly raised: {type(e_chat_direct).__name__} - {e_chat_direct.message}",
            exc_info=True if e_chat_direct.status_code >= 500 else False)
        raise e_chat_direct  # Re-raise the specific error
    except (ValueError, TypeError, KeyError) as e:
        logging.error(f"Value/Type/Key error during chat API call setup for {endpoint_lower}: {e}", exc_info=True)
        error_type = "Configuration/Parameter Error"
        if "Unsupported API endpoint" in str(e):
            raise ChatConfigurationError(provider=endpoint_lower, message=f"Unsupported API endpoint: {endpoint_lower}")
        else:
            raise ChatBadRequestError(provider=endpoint_lower, message=f"{error_type} for {endpoint_lower}: {e}")
    except Exception as e:
        logging.exception(
            f"Unexpected internal error in chat_api_call for {endpoint_lower}: {e}")
        raise ChatAPIError(provider=endpoint_lower,
                           message=f"An unexpected internal error occurred in chat_api_call for {endpoint_lower}: {str(e)}",
                           status_code=500)

#
####################################################################################################
#
# Main Chat Function
#

# Import needed for chat dictionary processing
# This creates a circular import that we'll fix in the refactored Chat_Functions.py
# For now, we'll import it conditionally inside the function
def process_user_input_wrapper(message, chatdict_entries, max_tokens=500, strategy="sorted_evenly"):
    """Wrapper to avoid circular import during module refactoring."""
    from tldw_Server_API.app.core.Chat.Chat_Functions import process_user_input
    return process_user_input(message, chatdict_entries, max_tokens, strategy)

def parse_user_dict_markdown_file_wrapper(file_path: str):
    """Wrapper to avoid circular import during module refactoring."""
    from tldw_Server_API.app.core.Chat.Chat_Functions import parse_user_dict_markdown_file, ChatDictionary
    parsed_dict_entries = parse_user_dict_markdown_file(file_path)
    if parsed_dict_entries:
        return [ChatDictionary(key=k, content=str(v)) for k, v in parsed_dict_entries.items()]
    return []

# FIXME - thing is fucking big.
# Break it down into smaller, logical async helper functions. For example:
#
#     _authenticate_request(Token)
#
#     _validate_request_payload(request_data)
#
#     _get_or_create_conversation_context(chat_db, request_data, character_card)
#
#     _load_chat_history(chat_db, conversation_id, character_card)
#
#     _prepare_llm_messages(request_data, historical_messages, character_card, template_name)
#
#     _handle_llm_call_and_response(loop, llm_call_func, request_data, chat_db, final_conversation_id, character_card)
def chat(
    message: str,
    history: List[Dict[str, Any]],
    media_content: Optional[Dict[str, str]],
    selected_parts: List[str],
    api_endpoint: str,
    api_key: Optional[str],
    custom_prompt: Optional[str],
    temperature: float,
    system_message: Optional[str] = None,
    streaming: bool = False,
    minp: Optional[float] = None,
    maxp: Optional[float] = None,
    model: Optional[str] = None,
    topp: Optional[float] = None,
    topk: Optional[int] = None,
    chatdict_entries: Optional[List[Any]] = None, # Should be List[ChatDictionary]
    max_tokens: int = 500,
    strategy: str = "sorted_evenly",
    current_image_input: Optional[Dict[str, str]] = None,
    image_history_mode: str = "tag_past",
    llm_max_tokens: Optional[int] = None,
    llm_seed: Optional[int] = None,
    llm_stop: Optional[Union[str, List[str]]] = None,
    llm_response_format: Optional[ResponseFormat] = None,
    llm_n: Optional[int] = None,
    llm_user_identifier: Optional[str] = None,
    llm_logprobs: Optional[bool] = None,
    llm_top_logprobs: Optional[int] = None,
    llm_logit_bias: Optional[Dict[str, float]] = None,
    llm_presence_penalty: Optional[float] = None,
    llm_frequency_penalty: Optional[float] = None,
    llm_tools: Optional[List[Dict[str, Any]]] = None,
    llm_tool_choice: Optional[Union[str, Dict[str, Any]]] = None
) -> Union[str, Any]: # Any for streaming generator
    """
    Orchestrates a chat interaction with an LLM, handling message processing,
    RAG, multimodal content, and chat dictionary features.

    This function prepares the `messages_payload` in OpenAI format, including
    history, current user message (with optional RAG and image), and then
    calls `chat_api_call` to get the LLM's response.

    Args:
        message: The current text message from the user.
        history: A list of previous messages in OpenAI format
                 (e.g., `[{'role': 'user', 'content': '...'}, {'role': 'assistant', 'content': '...'}]`).
                 Content can be simple text or a list of multimodal parts.
        media_content: A dictionary containing RAG content (e.g., `{'summary': '...', 'transcript': '...'}`).
        selected_parts: A list of keys from `media_content` to include as RAG.
        api_endpoint: Identifier for the target LLM provider.
        api_key: API key for the provider.
        custom_prompt: An additional prompt/instruction to prepend to the user's current message.
        temperature: LLM sampling temperature.
        system_message: A system-level instruction for the LLM. Passed to `chat_api_call`.
        streaming: Whether to stream the LLM response.
        minp: Min-P sampling parameter for the LLM.
        maxp: Max-P (often Top-P) sampling parameter for the LLM.
        model: The specific LLM model to use.
        topp: Top-P (nucleus) sampling parameter for the LLM.
        topk: Top-K sampling parameter for the LLM.
        chatdict_entries: A list of `ChatDictionary` objects for keyword replacement/expansion.
        max_tokens: Max tokens for chat dictionary content processing (not LLM response).
        strategy: Strategy for applying chat dictionary entries (e.g., "sorted_evenly").
        current_image_input: An optional dictionary for the current image being sent by the user,
                             in the format `{'base64_data': '...', 'mime_type': 'image/png'}`.
        image_history_mode: How to handle images from past messages:
                            "send_all": Send all past images.
                            "send_last_user_image": Send only the last image sent by a user.
                            "tag_past": Replace past images with a textual tag (e.g., "<image: prior_history.png>").
                            "ignore_past": Do not include any past images.
        llm_max_tokens: Max tokens for the LLM to generate in its response.
        llm_seed: Seed for LLM generation.
        llm_stop: Stop sequence(s) for LLM generation.
        llm_response_format: Desired response format from LLM (e.g., JSON object).
                             Pydantic `ResponseFormat` model instance.
        llm_n: Number of LLM completion choices to generate.
        llm_user_identifier: User identifier for LLM API call.
        llm_logprobs: Whether LLM should return log probabilities.
        llm_top_logprobs: Number of top log probabilities for LLM to return.
        llm_logit_bias: Logit bias for LLM token generation.
        llm_presence_penalty: Presence penalty for LLM generation.
        llm_frequency_penalty: Frequency penalty for LLM generation.
        llm_tools: Tools for LLM function calling.
        llm_tool_choice: Tool choice for LLM function calling.

    Returns:
        The LLM's response, either as a string (non-streaming) or a generator
        (streaming). In case of an error during chat processing, a string
        containing an error message is returned.

    Raises:
        Catches internal exceptions and returns an error message string.
        Exceptions from `chat_api_call` might propagate if not handled by its own try-except blocks.
    """
    log_counter("chat_attempt_multimodal", labels={"api_endpoint": api_endpoint, "image_mode": image_history_mode})
    start_time = time.time()

    try:
        logging.info(f"Debug - Chat Function - Input Text: '{message}', Image provided: {'Yes' if current_image_input else 'No'}")
        logging.info(f"Debug - Chat Function - History length: {len(history)}, Image History Mode: {image_history_mode}")

        # Ensure selected_parts is a list
        if not isinstance(selected_parts, (list, tuple)):
            selected_parts = [selected_parts] if selected_parts else []

        # Process message with Chat Dictionary (text only for now)
        processed_text_message = message
        if chatdict_entries and message:
            processed_text_message = process_user_input_wrapper(
                message, chatdict_entries, max_tokens=max_tokens, strategy=strategy
            )

        # --- Construct messages payload for the LLM API (OpenAI format) ---
        llm_messages_payload: List[Dict[str, Any]] = []

        # PHILOSOPHY:
        # `chat()` prepares the `llm_messages_payload` (user/assistant turns with multimodal content).
        # `chat()` also collects the `system_message`.
        # `chat_api_call()` receives both `llm_messages_payload` and the separate `system_message`.
        # `chat_api_call()` then dispatches these to the specific provider function (e.g., `chat_with_openai`).
        # The provider function (e.g., `chat_with_openai`) is responsible for:
        #   1. Taking the `messages` (which is `llm_messages_payload`).
        #   2. Taking the `system_message` parameter.
        #   3. If `system_message` is provided, *it* prepends `{"role": "system", "content": system_message}`
        #      to the `messages` list *if* that's how its API works (like OpenAI).
        #   4. Or, if its API takes system message as a separate top-level parameter (like Anthropic's `system_prompt`),
        #      it uses it directly there.
        # This way, `chat()` doesn't need to know the specifics of each API for system prompts


        # 2. Process History (now expecting list of OpenAI message dicts)
        last_user_image_url_from_history: Optional[str] = None

        for hist_msg_obj in history:
            role = hist_msg_obj.get("role")
            original_content = hist_msg_obj.get("content") # This can be str or list of parts

            processed_hist_content_parts = []

            if isinstance(original_content, str): # Simple text history message
                processed_hist_content_parts.append({"type": "text", "text": original_content})
            elif isinstance(original_content, list): # Already structured content
                for part in original_content:
                    if part.get("type") == "text":
                        processed_hist_content_parts.append(part)
                    elif part.get("type") == "image_url":
                        image_url_data = part.get("image_url", {}).get("url", "") # data URI
                        if image_history_mode == "send_all":
                            processed_hist_content_parts.append(part)
                            if role == "user": last_user_image_url_from_history = image_url_data
                        elif image_history_mode == "send_last_user_image" and role == "user":
                            last_user_image_url_from_history = image_url_data # Track, add later
                        elif image_history_mode == "tag_past":
                            mime_type_part = "image"
                            if image_url_data.startswith("data:image/") and ";base64," in image_url_data:
                                try: 
                                    mime_type_part = image_url_data.split(';base64,')[0].split('/')[-1]
                                except Exception as e: 
                                    logging.debug(f"Failed to extract MIME type from data URI: {e}")
                                    mime_type_part = "image"
                            processed_hist_content_parts.append({"type": "text", "text": f"<image: prior_history.{mime_type_part}>"})
                        # "ignore_past": do nothing, image part is skipped

            if processed_hist_content_parts: # Add if content remains
                llm_messages_payload.append({"role": role, "content": processed_hist_content_parts})

        # Handle "send_last_user_image" - append it to the last user message in payload if applicable
        if image_history_mode == "send_last_user_image" and last_user_image_url_from_history:
            appended_to_last = False
            for i in range(len(llm_messages_payload) -1, -1, -1): # Iterate backwards
                if llm_messages_payload[i]["role"] == "user":
                    # Ensure content is a list
                    if not isinstance(llm_messages_payload[i]["content"], list):
                        llm_messages_payload[i]["content"] = [{"type": "text", "text": str(llm_messages_payload[i]["content"])}]

                    # Avoid duplicates if already processed (e.g., if history was already "send_all" style)
                    is_duplicate = any(p.get("type") == "image_url" and p.get("image_url", {}).get("url") == last_user_image_url_from_history for p in llm_messages_payload[i]["content"])
                    if not is_duplicate:
                        llm_messages_payload[i]["content"].append({"type": "image_url", "image_url": {"url": last_user_image_url_from_history}})
                    appended_to_last = True
                    break
            if not appended_to_last: # No user message in history, or image already there
                 logging.debug(f"Could not append last_user_image_from_history, no suitable prior user message or already present. Image: {last_user_image_url_from_history[:60]}...")


        # 3. Add RAG Content (prepended to current user's text)
        rag_text_prefix = ""
        if media_content and selected_parts:
            rag_text_prefix = "\n\n".join(
                [f"{part.capitalize()}: {media_content.get(part, '')}" for part in selected_parts if media_content.get(part)]
            ).strip()
            if rag_text_prefix:
                rag_text_prefix += "\n\n---\n\n"

        # 4. Construct Current User Message (text + optional new image)
        current_user_content_parts: List[Dict[str, Any]] = []

        # Combine RAG, custom_prompt (if it's for current turn's text), and processed_text_message
        # Deciding where `custom_prompt` goes: if it's a direct instruction for *this* turn,
        # it should be part of the user's text. If it's more like a persona or ongoing rule,
        # it's better in `system_message`. Let's assume it's for this turn.
        final_text_for_current_message = processed_text_message
        if custom_prompt: # Prepend custom_prompt if it exists
            final_text_for_current_message = f"{custom_prompt}\n\n{final_text_for_current_message}"

        final_text_for_current_message = f"{rag_text_prefix}{final_text_for_current_message}".strip()

        if final_text_for_current_message:
            current_user_content_parts.append({"type": "text", "text": final_text_for_current_message})

        if current_image_input and current_image_input.get('base64_data') and current_image_input.get('mime_type'):
            image_url = f"data:{current_image_input['mime_type']};base64,{current_image_input['base64_data']}"
            current_user_content_parts.append({"type": "image_url", "image_url": {"url": image_url}})

        if not current_user_content_parts: # Should only happen if message, custom_prompt, RAG, and image are all empty/None
             logging.warning("Current user message has no text or image content parts. Sending a placeholder.")
             current_user_content_parts.append({"type": "text", "text": "(No user input for this turn)"})

        llm_messages_payload.append({"role": "user", "content": current_user_content_parts})

        # Temperature and other LLM params
        temperature_float = 0.7
        try: temperature_float = float(temperature) if temperature is not None else 0.7
        except ValueError: logging.warning(f"Invalid temperature '{temperature}', using 0.7.")

        logging.debug(f"Debug - Chat Function - Final LLM Payload (structure, image data truncated):")
        for i, msg_p in enumerate(llm_messages_payload):
            content_log = []
            if isinstance(msg_p.get("content"), list):
                for part_idx, part_c in enumerate(msg_p["content"]):
                    if part_c.get("type") == "text": content_log.append(f"text: '{part_c['text'][:30]}...'")
                    elif part_c.get("type") == "image_url": content_log.append(f"image: '{part_c['image_url']['url'][:40]}...'")
            logging.debug(f"  Msg {i}: Role: {msg_p['role']}, Content: [{', '.join(content_log)}]")

        logging.debug(f"Debug - Chat Function - Temperature: {temperature}")
        logging.debug(f"Debug - Chat Function - API Key: {api_key[:10]}")
        logging.debug(f"Debug - Chat Function - Prompt: {custom_prompt}")

        # --- Call the LLM via the updated chat_api_call ---
        response = chat_api_call(
            api_endpoint=api_endpoint,
            api_key=api_key,
            messages_payload=llm_messages_payload,
            temp=temperature_float,
            system_message=system_message,
            streaming=streaming,
            minp=minp, maxp=maxp, model=model, topp=topp, topk=topk,
            # Pass through new params from ChatCompletionRequest
            max_tokens=llm_max_tokens,
            seed=llm_seed,
            stop=llm_stop,
            response_format=llm_response_format.model_dump() if llm_response_format else None,
            n=llm_n,
            user_identifier=llm_user_identifier,
            logprobs=llm_logprobs,
            top_logprobs=llm_top_logprobs,
            logit_bias=llm_logit_bias,
            presence_penalty=llm_presence_penalty,
            frequency_penalty=llm_frequency_penalty,
            tools=llm_tools,
            tool_choice=llm_tool_choice
        )

        if streaming:
            logging.debug("Chat Function - Response: Streaming Generator")
            return response
        else:
            chat_duration = time.time() - start_time
            log_histogram("chat_duration_multimodal", chat_duration, labels={"api_endpoint": api_endpoint})
            log_counter("chat_success_multimodal", labels={"api_endpoint": api_endpoint})
            logging.debug(f"Chat Function - Response (first 500 chars): {str(response)[:500]}")

            loaded_config_data = load_and_log_configs()
            post_gen_replacement_config = loaded_config_data.get('chat_dictionaries', {}).get('post_gen_replacement')
            if post_gen_replacement_config and isinstance(response, str):
                post_gen_replacement_dict_path = loaded_config_data.get('chat_dictionaries', {}).get('post_gen_replacement_dict')
                if post_gen_replacement_dict_path and os.path.exists(post_gen_replacement_dict_path):
                    try:
                        post_gen_chat_dict_objects = parse_user_dict_markdown_file_wrapper(post_gen_replacement_dict_path)
                        if post_gen_chat_dict_objects:
                            response = process_user_input_wrapper(response, post_gen_chat_dict_objects)
                            # The original warning log can be removed or changed to a debug log if successfully applied.
                            logging.debug(
                                f"Response after post-gen replacement (first 500 chars): {str(response)[:500]}")
                        else:
                            logging.debug("Post-gen dictionary parsed but resulted in no ChatDictionary objects.")

                        logging.debug(f"Response after post-gen replacement (first 500 chars): {str(response)[:500]}")
                    except Exception as e_post_gen:
                        logging.error(f"Error during post-generation replacement: {e_post_gen}", exc_info=True)
                else:
                    logging.warning("Post-gen replacement enabled but dict file not found/configured.")
            return response

    except Exception as e:
        log_counter("chat_error_multimodal", labels={"api_endpoint": api_endpoint, "error": str(e)})
        logging.error(f"Error in multimodal chat function: {str(e)}", exc_info=True)
        # Consider if the error format should change from just a string
        return f"An error occurred in the chat function: {str(e)}"

#
# End of chat_orchestrator.py
####################################################################################################