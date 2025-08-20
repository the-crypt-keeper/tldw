# Chat_Functions.py
# Description: Chat functions for interacting with the LLMs as chatbots
"""
This module provides a comprehensive set of functions and classes for managing chat interactions,
character data, and chat dictionaries in a multimodal chatbot system. It includes functionality
for handling API calls, saving and loading chat history, managing character cards, and processing
user input with chat dictionaries.

Key Features:
- Chat API call handling with error management and multimodal support.
- Chat history saving and exporting to JSON.
- Character card management, including saving, loading, and updating character data.
- Chat dictionary processing for keyword-based text replacement and token budget management.
"""
#
# Imports
import base64
import json
import logging
import os
import random
import re
import tempfile
from ..Utils.secure_temp_files import create_secure_temp_file, secure_delete_file
import time
import warnings
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional, Union, Literal
#
# 3rd-party Libraries
from loguru import logger
import requests
from pydantic import BaseModel, Field

# Configure logger with context
logger = logger.bind(module="Chat_Functions")

#
# Local Imports
from .Chat_Deps import ChatBadRequestError, ChatConfigurationError, ChatAPIError, \
    ChatProviderError, ChatRateLimitError, ChatAuthenticationError
from tldw_chatbook.DB.ChaChaNotes_DB import CharactersRAGDB, InputError, ConflictError, CharactersRAGDBError
from tldw_chatbook.LLM_Calls.LLM_API_Calls import chat_with_openai, chat_with_anthropic, chat_with_cohere, \
    chat_with_groq, chat_with_openrouter, chat_with_deepseek, chat_with_mistral, chat_with_huggingface, chat_with_google, \
    chat_with_moonshot, chat_with_zai
from tldw_chatbook.LLM_Calls.LLM_API_Calls_Local import chat_with_aphrodite, chat_with_local_llm, chat_with_ollama, \
    chat_with_kobold, chat_with_llama, chat_with_oobabooga, chat_with_tabbyapi, chat_with_vllm, chat_with_custom_openai, \
    chat_with_custom_openai_2, chat_with_mlx_lm
from tldw_chatbook.Utils.Utils import generate_unique_filename, logging
from tldw_chatbook.Utils.path_validation import validate_path
from tldw_chatbook.Metrics.metrics_logger import log_counter, log_histogram
from tldw_chatbook.config import load_settings
#
####################################################################################################
#
# Functions:

# +++ Default Character Configuration +++
DEFAULT_CHARACTER_NAME = "Default Character"
DEFAULT_CHARACTER_DESCRIPTION = "This is a default character created by the system."

class ResponseFormat(BaseModel):
    type: Literal["text", "json_object"] = Field("text", description="Must be one of `text` or `json_object`.")

def approximate_token_count(history):
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
        logger.error(f"Error calculating token count: {str(e)}")
        return 0

# 1. Dispatch table for handler functions
API_CALL_HANDLERS = {
    'openai': chat_with_openai,
    'anthropic': chat_with_anthropic,
    'cohere': chat_with_cohere,
    'groq': chat_with_groq,
    'openrouter': chat_with_openrouter,
    'deepseek': chat_with_deepseek,
    'mistral': chat_with_mistral,
    'mistralai': chat_with_mistral,  # Handle both 'mistral' and 'MistralAI' API types
    'google': chat_with_google,
    'huggingface': chat_with_huggingface,
    'moonshot': chat_with_moonshot,
    'zai': chat_with_zai,
    'llama_cpp': chat_with_llama,
    'koboldcpp': chat_with_kobold,
    'oobabooga': chat_with_oobabooga,
    'tabbyapi': chat_with_tabbyapi,
    'vllm': chat_with_vllm,
    'local-llm': chat_with_local_llm,
    'ollama': chat_with_ollama,
    'aphrodite': chat_with_aphrodite,
    'custom-openai-api': chat_with_custom_openai,
    'custom-openai-api-2': chat_with_custom_openai_2,
    'mlx_lm': chat_with_mlx_lm,
    # Map local_* provider names to their handler functions
    'local_llamacpp': chat_with_llama,
    'local_llamafile': chat_with_llama,
    'local_ollama': chat_with_ollama,
    'local_vllm': chat_with_vllm,
    'local_mlx_lm': chat_with_mlx_lm,
}
"""
A dispatch table mapping API endpoint names (e.g., 'openai') to their
corresponding handler functions (e.g., `chat_with_openai`). This is used by
`chat_api_call` to route requests to the appropriate LLM provider.
FIXME: The mappings and handlers should be validated for correctness.
"""

# 2. Parameter mapping for each provider
# Maps generic chat_api_call param name to provider-specific param name
PROVIDER_PARAM_MAP = {
    'openai': {
        'api_key': 'api_key',
        'messages_payload': 'input_data',
        'prompt': 'custom_prompt_arg',
        'temp': 'temp',
        'system_message': 'system_message',
        'streaming': 'streaming',
        'maxp': 'maxp',
        'model': 'model',
        'tools': 'tools',
        'tool_choice': 'tool_choice',
        'logprobs': 'logprobs',
        'top_logprobs': 'top_logprobs',
        'logit_bias': 'logit_bias',
        'presence_penalty': 'presence_penalty',
        'frequency_penalty': 'frequency_penalty',
        'max_tokens': 'max_tokens',
        'seed': 'seed',
        'stop': 'stop',
        'response_format': 'response_format',
        'n': 'n',
        'user_identifier': 'user',
    },
    'anthropic': {
        'api_key': 'api_key',
        'messages_payload': 'input_data',
        'prompt': 'custom_prompt_arg',
        'temp': 'temp',
        'system_message': 'system_prompt',
        'streaming': 'streaming',
        'model': 'model',
        'topp': 'topp',
        'topk': 'topk',
        'tools': 'tools',
        #'tool_choice': 'tool_choice',
        'max_tokens': 'max_tokens',  # Anthropic uses max_tokens
        'stop': 'stop_sequences',  # Anthropic uses stop_sequences
    },
    'cohere': {
        'api_key': 'api_key',
        'messages_payload': 'input_data',
        'prompt': 'custom_prompt_arg',
        'temperature': 'temp',
        'system_message': 'system_prompt',
        'streaming': 'streaming',
        'model': 'model',
        'topp': 'topp',
        'topk': 'topk',
        'tools': 'tools',
        #'tool_choice': 'tool_choice',
        'max_tokens': 'max_tokens',
        'stop': 'stop_sequences',
        'seed': 'seed',
        'n': 'num_generations',
        'frequency_penalty': 'frequency_penalty',
        'presence_penalty': 'presence_penalty',
    },
    'groq': {
        'api_key': 'api_key',
        'messages_payload': 'input_data',
        'prompt': 'custom_prompt_arg',
        'temperature': 'temp',
        'system_message': 'system_message',
        'streaming': 'streaming',
        'maxp': 'maxp',
        'model':'model', # Groq also uses top_p, handled by chat_with_groq
        'logit_bias': 'logit_bias',
        'presence_penalty': 'presence_penalty',
        'frequency_penalty': 'frequency_penalty',
    },
    'openrouter': {
        'api_key': 'api_key',
        'messages_payload': 'input_data',
        'prompt': 'custom_prompt_arg',
        'temperature': 'temp',
        'system_message': 'system_message',
        'streaming': 'streaming',
        'topp': 'top_p',
        'topk': 'top_k',
        'minp': 'min_p',  # OpenRouter expects min_p not minp
        'model':'model',
        'max_tokens': 'max_tokens',
        'seed': 'seed',
        'stop': 'stop',
        'response_format': 'response_format',
        'n': 'n',
        'user_identifier': 'user',
        'tools': 'tools',
        'tool_choice': 'tool_choice',
        'logit_bias': 'logit_bias',
        'presence_penalty': 'presence_penalty',
        'frequency_penalty': 'frequency_penalty',
        'logprobs': 'logprobs',
        'top_logprobs': 'top_logprobs',
    },
    'deepseek': {
        'api_key': 'api_key',
        'messages_payload': 'input_data',
        'prompt': 'custom_prompt_arg',
        'temperature': 'temp',
        'system_message': 'system_message',
        'streaming': 'streaming',
        'topp': 'topp',
        'model':'model',
        'max_tokens': 'max_tokens',
        'seed': 'seed',
        'stop': 'stop',
        'logprobs': 'logprobs',
        'top_logprobs': 'top_logprobs',  # if supported
        'presence_penalty': 'presence_penalty',
        'frequency_penalty': 'frequency_penalty',
    },
    'mistral': {
        'api_key': 'api_key',
        'messages_payload': 'input_data',
        'prompt': 'custom_prompt_arg',
        'temperature': 'temp',
        'system_message': 'system_message',
        'streaming': 'streaming',
        'topp': 'topp',
        'tools': 'tools',
        'tool_choice': 'tool_choice',
        'model': 'model',
        'max_tokens': 'max_tokens',
        'seed': 'random_seed',  # Mistral uses random_seed
        'topk': 'top_k',  # Mistral uses top_k
    },
    'mistralai': {  # Same mapping as 'mistral'
        'api_key': 'api_key',
        'messages_payload': 'input_data',
        'prompt': 'custom_prompt_arg',
        'temperature': 'temp',
        'system_message': 'system_message',
        'streaming': 'streaming',
        'topp': 'topp',
        'tools': 'tools',
        'tool_choice': 'tool_choice',
        'model': 'model',
        'max_tokens': 'max_tokens',
        'seed': 'random_seed',  # Mistral uses random_seed
        'topk': 'top_k',  # Mistral uses top_k
    },
    'google': {
        'api_key': 'api_key',
        'messages_payload': 'input_data',
        'prompt': 'custom_prompt_arg',
        'temperature': 'temp',
        'system_message': 'system_message',
        'streaming': 'streaming',
        'topp': 'topp',
        'topk': 'topk',
        'tools': 'tools',
        #'tool_choice': 'tool_choice',
        'model':'model',
        'max_tokens': 'max_output_tokens',
        'stop': 'stop_sequences',  # List of strings
        'n': 'candidate_count',
    },
    'huggingface': {
        'api_key': 'api_key',
        'messages_payload': 'input_data',
        'prompt': 'custom_prompt_arg',
        'temperature': 'temp',
        'system_message': 'system_message',
        'streaming': 'streaming',
        'model':'model',
        'max_tokens': 'max_new_tokens',  # Common for TGI
        'topp': 'top_p',
        'topk': 'top_k',
        'seed': 'seed',
        'stop': 'stop',  # often 'stop_sequences'
    },
    'llama_cpp': { # Has api_url as a positional argument which needs special handling if not None
        'api_key': 'api_key',
        'messages_payload': 'input_data',
        'prompt': 'custom_prompt',
        'temperature': 'temperature',
        'system_message': 'system_prompt',
        'streaming': 'streaming',
        'topp': 'top_p',
        'topk': 'top_k',
        'minp': 'min_p',
        'model':'model',
        #'tools': 'tools',
        #'tool_choice': 'tool_choice',
        'max_tokens': 'n_predict', # Common for llama.cpp server
        'seed': 'seed',
        'stop': 'stop', # list of strings
        'response_format': 'response_format', # if OpenAI compatible endpoint
        'logit_bias': 'logit_bias',
        'n': 'n_probs', # FIXME: n_probs mapping might not be direct.
        'presence_penalty': 'presence_penalty',
        'frequency_penalty': 'frequency_penalty',
        'logprobs': 'logprobs',
        'top_logprobs': 'top_logprobs',
    },
    'koboldcpp': {
        'api_key': 'api_key',
        'messages_payload': 'input_data',
        'llm_fixed_tokens_kobold': 'fixed_tokens_mode', # Added
        'prompt': 'custom_prompt_input',
        'temperature': 'temp',
        'system_message': 'system_message',
        'streaming': 'streaming',
        'topp': 'top_p',
        'topk': 'top_k',
        'model':'model',
        'max_tokens': 'max_length',  # or 'max_context_length'
        'stop': 'stop_sequence',  # Often a list
        'n': 'num_responses',
        'seed': 'seed',
    },
    'oobabooga': { # api_url also a consideration like llama.cpp
        'api_key': 'api_key',
        'messages_payload': 'input_data',
        'prompt': 'custom_prompt',
        'temperature': 'temperature',
        'system_message': 'system_prompt', # often part of messages or specific param
        'streaming': 'streaming',
        'topp': 'top_p',
        'model':'model',
        'topk': 'top_k',
        'minp': 'min_p',
        'max_tokens': 'max_tokens', # or 'max_new_tokens'
        'seed': 'seed',
        'stop': 'stop',
        'response_format': 'response_format',
        'n': 'n',
        'user_identifier': 'user',
        'logit_bias': 'logit_bias',
        'presence_penalty': 'presence_penalty',
        'frequency_penalty': 'frequency_penalty',
    },
    'tabbyapi': {
        'api_key': 'api_key',
        'messages_payload': 'input_data',
        'prompt': 'custom_prompt_input',
        'temp': 'temperature',
        'system_message': 'system_message',
        'streaming': 'streaming',
        'topp': 'top_p',
        'topk': 'top_k',
        'minp': 'min_p',
        'model':'model',
        'max_tokens': 'max_tokens',
        'seed': 'seed',
        'stop': 'stop',
    },
    'vllm': { # vllm_api_url consideration
                'api_key': 'api_key', 'messages_payload': 'input_data', 'prompt': 'custom_prompt_input',
        'temp': 'temperature', 'system_message': 'system_prompt', 'streaming': 'streaming',
        'topp': 'top_p', 'topk': 'top_k', 'minp': 'min_p', 'model': 'model',
        'max_tokens': 'max_tokens',
        'seed': 'seed',
        'stop': 'stop',
        'response_format': 'response_format',
        'n': 'n',
        'logit_bias': 'logit_bias',
        'presence_penalty': 'presence_penalty',
        'frequency_penalty': 'frequency_penalty',
        'logprobs': 'logprobs',
        'top_logprobs': 'top_logprobs',
        'user_identifier': 'user',
    },
    'local-llm': {
        'messages_payload': 'input_data',
        'prompt': 'custom_prompt_arg',
        'temp': 'temperature',
        'system_message': 'system_message',
        'streaming': 'streaming',
        'topp': 'top_p',
        'topk': 'top_k',
        'minp': 'min_p',
        'model':'model',
        'max_tokens': 'max_tokens',
        'seed': 'seed',
        'stop': 'stop',
    },
    'ollama': { # api_url consideration
        'api_key': 'api_key', # api_key is not used by ollama directly, url is more important
        'messages_payload': 'input_data',
        'prompt': 'custom_prompt', # This is 'prompt' for generate, 'messages' for chat
        'temp': 'temperature',
        'system_message': 'system_message', # Part of request body
        'streaming': 'streaming',
        'topp': 'top_p',
        'topk': 'top_k',
        'model': 'model',
        'max_tokens': 'num_predict', # For generate endpoint, chat might be different
        'seed': 'seed',
        'stop': 'stop', # list of strings
        'response_format': 'format', # 'json' string
        'presence_penalty': 'presence_penalty',
        'frequency_penalty': 'frequency_penalty',
    },
    'aphrodite': {
        'api_key': 'api_key',
        'messages_payload': 'input_data',
        'prompt': 'custom_prompt',
        'temp': 'temperature',
        'system_message': 'system_message',
        'streaming': 'streaming',
        'topp': 'top_p',
        'topk': 'top_k',
        'minp': 'min_p',
        'model': 'model',
        'max_tokens': 'max_tokens',
        'seed': 'seed',
        'stop': 'stop',
        'response_format': 'response_format',
        'n': 'n',
        'logit_bias': 'logit_bias',
        'presence_penalty': 'presence_penalty',
        'frequency_penalty': 'frequency_penalty',
        'logprobs': 'logprobs',
        'user_identifier': 'user',
    },
    'custom-openai-api': {
        'api_key': 'api_key',
        'messages_payload': 'input_data',
        'prompt': 'custom_prompt_arg',
        'temp': 'temp',
        'system_message': 'system_message',
        'streaming': 'streaming',
        'maxp': 'maxp',
        'minp':'minp',
        'topk':'topk',
        'model': 'model',
        'max_tokens': 'max_tokens',
        'seed': 'seed',
        'stop': 'stop',
        'response_format': 'response_format',
        'n': 'n',
        'user_identifier': 'user',
        'tools': 'tools',
        'tool_choice': 'tool_choice',
        'logit_bias': 'logit_bias',
        'presence_penalty': 'presence_penalty',
        'frequency_penalty': 'frequency_penalty',
        'logprobs': 'logprobs',
        'top_logprobs': 'top_logprobs',
    },
    'custom-openai-api-2': {
        'api_key': 'api_key',
        'messages_payload': 'input_data',
        'prompt': 'custom_prompt_arg',
        'temp': 'temp',
        'system_message': 'system_message',
        'streaming': 'streaming',
        'model': 'model',
        'max_tokens': 'max_tokens',
        'seed': 'seed',
        'stop': 'stop',
        'response_format': 'response_format',
        'n': 'n',
        'user_identifier': 'user',
        'tools': 'tools',
        'tool_choice': 'tool_choice',
        'logit_bias': 'logit_bias',
        'presence_penalty': 'presence_penalty',
        'frequency_penalty': 'frequency_penalty',
        'logprobs': 'logprobs',
        'top_logprobs': 'top_logprobs',
    },
    'mlx_lm': {
        'api_key': 'api_key', # chat_with_mlx_lm doesn't use it, but map for consistency if passed via chat_api_call
        'messages_payload': 'input_data',
        'prompt': 'custom_prompt_arg', # This would be caught by **kwargs in chat_with_mlx_lm if passed
        'temp': 'temp',
        'system_message': 'system_message',
        'streaming': 'streaming',
        'model': 'model', # In chat_with_mlx_lm, 'model' parameter is the model_path
        'max_tokens': 'max_tokens',
        # chat_api_call uses 'topp', 'topk', 'minp' as its generic names.
        # chat_with_mlx_lm (via _chat_with_openai_compatible_local_server) expects 'top_p', 'top_k', 'min_p'.
        'topp': 'top_p',
        'topk': 'top_k',
        'minp': 'min_p',
        'stop': 'stop',
        'seed': 'seed',
        'response_format': 'response_format',
        'n': 'n',
        'presence_penalty': 'presence_penalty',
        'frequency_penalty': 'frequency_penalty',
        'logit_bias': 'logit_bias',
        'logprobs': 'logprobs',
        'top_logprobs': 'top_logprobs',
        'user_identifier': 'user_identifier',
        'tools': 'tools',
        'tool_choice': 'tool_choice',
        # api_url is a direct kwarg to chat_with_mlx_lm, not typically mapped from these generic chat_api_call args.
        # It's usually derived from config within the function itself or passed via UI directly to server start.
    },
    # Local provider mappings (same as their non-local counterparts)
    'local_llamacpp': {
        'api_key': 'api_key',
        'messages_payload': 'input_data',
        'prompt': 'custom_prompt',
        'temperature': 'temperature',
        'system_message': 'system_prompt',
        'streaming': 'streaming',
        'topp': 'top_p',
        'topk': 'top_k',
        'minp': 'min_p',
        'model':'model',
        'max_tokens': 'n_predict',
        'seed': 'seed',
        'stop': 'stop',
        'response_format': 'response_format',
        'logit_bias': 'logit_bias',
        'n': 'n_probs',
        'presence_penalty': 'presence_penalty',
        'frequency_penalty': 'frequency_penalty',
    },
    'local_llamafile': {
        'api_key': 'api_key',
        'messages_payload': 'input_data',
        'prompt': 'custom_prompt',
        'temperature': 'temperature',
        'system_message': 'system_prompt',
        'streaming': 'streaming',
        'topp': 'top_p',
        'topk': 'top_k',
        'minp': 'min_p',
        'model':'model',
        'max_tokens': 'n_predict',
        'seed': 'seed',
        'stop': 'stop',
        'response_format': 'response_format',
        'logit_bias': 'logit_bias',
        'n': 'n_probs',
        'presence_penalty': 'presence_penalty',
        'frequency_penalty': 'frequency_penalty',
    },
    'local_ollama': {
        'api_key': 'api_key',
        'messages_payload': 'input_data',
        'prompt': 'custom_prompt',
        'temp': 'temperature',
        'system_message': 'system_message',
        'streaming': 'streaming',
        'model': 'model',
        'topp': 'top_p',
        'topk': 'top_k',
        'max_tokens': 'num_predict',
        'seed': 'seed',
        'stop': 'stop',
        'response_format': 'format',
        'presence_penalty': 'presence_penalty',
        'frequency_penalty': 'frequency_penalty',
    },
    'local_vllm': {
        'api_key': 'api_key',
        'messages_payload': 'input_data',
        'prompt': 'custom_prompt_input',
        'temp': 'temperature',
        'system_message': 'system_prompt',
        'streaming': 'streaming',
        'model': 'model',
        'topk': 'top_k',
        'topp': 'top_p',
        'minp': 'min_p',
        'max_tokens': 'max_tokens',
        'seed': 'seed',
        'stop': 'stop',
        'response_format': 'response_format',
        'n': 'n',
        'logit_bias': 'logit_bias',
        'presence_penalty': 'presence_penalty',
        'frequency_penalty': 'frequency_penalty',
        'logprobs': 'logprobs',
        'user_identifier': 'user_identifier',
    },
    'local_mlx_lm': {
        'api_key': 'api_key',
        'messages_payload': 'input_data',
        'prompt': 'custom_prompt_arg',
        'temp': 'temp',
        'system_message': 'system_message',
        'streaming': 'streaming',
        'model': 'model',
        'max_tokens': 'max_tokens',
        'topp': 'top_p',
        'topk': 'top_k',
        'minp': 'min_p',
        'stop': 'stop',
        'seed': 'seed',
        'response_format': 'response_format',
        'n': 'n',
        'presence_penalty': 'presence_penalty',
        'frequency_penalty': 'frequency_penalty',
        'logit_bias': 'logit_bias',
        'logprobs': 'logprobs',
        'top_logprobs': 'top_logprobs',
        'user_identifier': 'user_identifier',
        'tools': 'tools',
        'tool_choice': 'tool_choice',
    },
    'moonshot': {
        'api_key': 'api_key',
        'messages_payload': 'input_data',
        'prompt': 'custom_prompt_arg',
        'temp': 'temp',
        'system_message': 'system_message',
        'streaming': 'streaming',
        'maxp': 'maxp',
        'model': 'model',
        'tools': 'tools',
        'tool_choice': 'tool_choice',
        'presence_penalty': 'presence_penalty',
        'frequency_penalty': 'frequency_penalty',
        'max_tokens': 'max_tokens',
        'seed': 'seed',
        'stop': 'stop',
        'response_format': 'response_format',
        'n': 'n',
        'user_identifier': 'user',
    },
    'zai': {
        'api_key': 'api_key',
        'messages_payload': 'input_data',
        'prompt': 'custom_prompt_arg',
        'temp': 'temp',
        'system_message': 'system_message',
        'streaming': 'streaming',
        'maxp': 'maxp',  # maps to top_p
        'model': 'model',
        'max_tokens': 'max_tokens',
        'tools': 'tools',
    },
    # Add other providers here
}
"""
Maps generic parameter names used in `chat_api_call` to provider-specific
parameter names for each LLM API. This allows `chat_api_call` to use a
consistent interface while adapting to the idiosyncrasies of different providers.
FIXME: The mappings should be validated for correctness and completeness for each provider.
"""

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
    user_identifier: Optional[str] = None,  # Renamed from 'user' to avoid conflict with 'user' role in messages
    llm_fixed_tokens_kobold: Optional[bool] = False # Added
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
        :rtype: str | dict | Generator[str, Any, None]
    """
    endpoint_lower = api_endpoint.lower()
    logger.info(f"Chat API Call - Routing to endpoint: {endpoint_lower}")
    log_counter("chat_api_call_attempt", labels={"api_endpoint": endpoint_lower})
    start_time = time.time()

    handler = API_CALL_HANDLERS.get(endpoint_lower)
    if not handler:
        logger.error(f"Unsupported API endpoint requested: {api_endpoint}")
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
        'user_identifier': user_identifier,
        'llm_fixed_tokens_kobold': llm_fixed_tokens_kobold # Added
    }

    for generic_param_name, provider_param_name in params_map.items():
        if generic_param_name in available_generic_params and available_generic_params[generic_param_name] is not None:
            call_kwargs[provider_param_name] = available_generic_params[generic_param_name]
        if generic_param_name == 'prompt' and endpoint_lower == 'cohere':
             pass # Specific handling for Cohere's prompt is assumed to be within chat_with_cohere

    if call_kwargs.get(params_map.get('api_key', 'api_key')) and isinstance(call_kwargs.get(params_map.get('api_key', 'api_key')), str) and len(call_kwargs.get(params_map.get('api_key', 'api_key'))) > 8:
         logger.info(f"Debug - Chat API Call - API Key: {call_kwargs[params_map.get('api_key', 'api_key')][:4]}...{call_kwargs[params_map.get('api_key', 'api_key')][-4:]}")

    # Add provider_name to kwargs only for handlers that support it
    # Some local providers use this for dynamic configuration loading
    PROVIDERS_WITH_PROVIDER_NAME = {
        'llama_cpp', 'vllm', 'ollama', 'mlx_lm', 'vllm_api', 'mlx',
        'local_llamacpp', 'local_llamafile', 'local_vllm', 'local_ollama', 'local_mlx_lm'
    }
    
    if endpoint_lower in PROVIDERS_WITH_PROVIDER_NAME:
        call_kwargs['provider_name'] = endpoint_lower

    try:
        logger.debug(f"Calling handler {handler.__name__} with kwargs: { {k: (type(v) if k != params_map.get('api_key') else 'key_hidden') for k,v in call_kwargs.items()} }")
        response = handler(**call_kwargs)

        call_duration = time.time() - start_time
        log_histogram("chat_api_call_duration", call_duration, labels={"api_endpoint": endpoint_lower})
        log_counter("chat_api_call_success", labels={"api_endpoint": endpoint_lower})

        if isinstance(response, str):
             logger.debug(f"Debug - Chat API Call - Response (first 500 chars): {response[:500]}...")
        elif hasattr(response, '__iter__') and not isinstance(response, (str, bytes, dict)):
             logger.debug(f"Debug - Chat API Call - Response: Streaming Generator")
        else:
             logger.debug(f"Debug - Chat API Call - Response Type: {type(response)}")
        return response

    # --- Exception Mapping (copied from your original, ensure it's still relevant) ---
    except requests.exceptions.HTTPError as e:
        status_code = getattr(e.response, 'status_code', 500)
        error_text = getattr(e.response, 'text', str(e))
        log_message_base = f"{endpoint_lower} API call failed with status {status_code}"

        # Log safely first
        try:
            logger.error("%s. Details: %s", log_message_base, error_text[:500], exc_info=False)
        except Exception as log_e:
            logger.error(f"Error during logging HTTPError details: {log_e}")

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
        logger.error(f"Network error connecting to {endpoint_lower}: {e}", exc_info=False)
        raise ChatProviderError(provider=endpoint_lower, message=f"Network error: {e}", status_code=504)
    except (ChatAuthenticationError, ChatRateLimitError, ChatBadRequestError, ChatConfigurationError, ChatProviderError,
            ChatAPIError) as e_chat_direct:
        # This catches cases where the handler itself has already processed an error
        # (e.g. non-HTTP error, or it decided to raise a specific Chat*Error type)
        # and raises one of our custom exceptions.
        logger.error(
            f"Handler for {endpoint_lower} directly raised: {type(e_chat_direct).__name__} - {e_chat_direct.message}",
            exc_info=True if e_chat_direct.status_code >= 500 else False)
        raise e_chat_direct  # Re-raise the specific error
    except (ValueError, TypeError, KeyError) as e:
        logger.error(f"Value/Type/Key error during chat API call setup for {endpoint_lower}: {e}", exc_info=True)
        error_type = "Configuration/Parameter Error"
        if "Unsupported API endpoint" in str(e):
            raise ChatConfigurationError(provider=endpoint_lower, message=f"Unsupported API endpoint: {endpoint_lower}")
        else:
            raise ChatBadRequestError(provider=endpoint_lower, message=f"{error_type} for {endpoint_lower}: {e}")
    except Exception as e:
        logger.exception(
            f"Unexpected internal error in chat_api_call for {endpoint_lower}: {e}")
        raise ChatAPIError(provider=endpoint_lower,
                           message=f"An unexpected internal error occurred in chat_api_call for {endpoint_lower}: {str(e)}",
                           status_code=500)


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
    llm_response_format: Optional[Dict[str, str]] = None,
    llm_n: Optional[int] = None,
    llm_user_identifier: Optional[str] = None,
    llm_logprobs: Optional[bool] = None,
    llm_top_logprobs: Optional[int] = None,
    llm_logit_bias: Optional[Dict[str, float]] = None,
    llm_presence_penalty: Optional[float] = None,
    llm_frequency_penalty: Optional[float] = None,
    llm_tools: Optional[List[Dict[str, Any]]] = None,
    llm_tool_choice: Optional[Union[str, Dict[str, Any]]] = None,
    llm_fixed_tokens_kobold: Optional[bool] = False, # Added
    strip_thinking_tags: bool = True # Added for thinking tag stripping
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
                             Passed as a dictionary, e.g., `{"type": "json_object"}`.
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
        if current_image_input:
            logging.info(f"DEBUG: current_image_input contents: {current_image_input.keys() if isinstance(current_image_input, dict) else type(current_image_input)}")
            if isinstance(current_image_input, dict):
                logging.info(f"DEBUG: has base64_data={bool(current_image_input.get('base64_data'))}, mime_type={current_image_input.get('mime_type')}")
        logging.info(f"Debug - Chat Function - History length: {len(history)}, Image History Mode: {image_history_mode}")
        logging.info(
            f"Debug - Chat Function - LLM Max Tokens: {llm_max_tokens}, LLM Seed: {llm_seed}, LLM Stop: {llm_stop}, LLM N: {llm_n}")
        logging.info(
            f"Debug - Chat Function - LLM User Identifier: {llm_user_identifier}, LLM Logprobs: {llm_logprobs}, LLM Top Logprobs: {llm_top_logprobs}")
        logging.info(
            f"Debug - Chat Function - LLM Logit Bias: {llm_logit_bias}, LLM Presence Penalty: {llm_presence_penalty}, LLM Frequency Penalty: {llm_frequency_penalty}")
        logging.info(
            f"Debug - Chat Function - LLM Tools: {llm_tools}, LLM Tool Choice: {llm_tool_choice}, LLM Response Format (dict): {llm_response_format}")

        # Ensure selected_parts is a list
        if not isinstance(selected_parts, (list, tuple)):
            selected_parts = [selected_parts] if selected_parts else []

        # Process message with Chat Dictionary (text only for now)
        processed_text_message = message
        if chatdict_entries and message:
            dict_start_time = time.time()
            processed_text_message = process_user_input(
                message, chatdict_entries, max_tokens=max_tokens, strategy=strategy
            )
            dict_duration = time.time() - dict_start_time
            log_histogram("chat_dictionary_processing_duration", dict_duration, labels={"api_endpoint": api_endpoint})
            log_counter("chat_dictionary_applied", labels={"api_endpoint": api_endpoint})

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
        
        history_start_time = time.time()
        history_message_count = len(history)
        history_image_count = 0

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
                        history_image_count += 1
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
                                except (IndexError, ValueError) as e:
                                    logger.debug(f"Failed to parse image MIME type from data URL: {e}")
                                    # mime_type_part remains "image"
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
        
        # Log history processing metrics
        history_duration = time.time() - history_start_time
        log_histogram("chat_history_processing_duration", history_duration, labels={"api_endpoint": api_endpoint})
        log_histogram("chat_history_message_count", history_message_count, labels={"api_endpoint": api_endpoint})
        log_histogram("chat_history_image_count", history_image_count, labels={
            "api_endpoint": api_endpoint, 
            "image_mode": image_history_mode
        })


        # 3. Add RAG Content (prepended to current user's text)
        rag_text_prefix = ""
        if media_content and selected_parts:
            rag_start_time = time.time()
            rag_text_prefix = "\n\n".join(
                [f"{part.capitalize()}: {media_content.get(part, '')}" for part in selected_parts if media_content.get(part)]
            ).strip()
            if rag_text_prefix:
                rag_text_prefix += "\n\n---\n\n"
                rag_duration = time.time() - rag_start_time
                log_histogram("chat_rag_processing_duration", rag_duration, labels={"api_endpoint": api_endpoint})
                log_histogram("chat_rag_content_length", len(rag_text_prefix), labels={"api_endpoint": api_endpoint})
                log_counter("chat_rag_content_added", labels={"api_endpoint": api_endpoint})

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
            logging.info(f"DEBUG: Adding image to current_user_content_parts: mime_type={current_image_input['mime_type']}, base64_data_length={len(current_image_input['base64_data'])}")
            image_url = f"data:{current_image_input['mime_type']};base64,{current_image_input['base64_data']}"
            current_user_content_parts.append({"type": "image_url", "image_url": {"url": image_url}})
            logging.info(f"✅ Successfully added image to message payload for {api_endpoint}/{model}")
        else:
            if current_image_input:
                logging.warning(f"Image input provided but missing required data: has_base64={bool(current_image_input.get('base64_data'))}, has_mime_type={bool(current_image_input.get('mime_type'))}")

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
        logging.debug(f"Debug - Chat Function - API Key: {api_key[:10] if api_key else 'None'}")
        logging.debug(f"Debug - Chat Function - Prompt: {custom_prompt}")

        #####################################################################
        # --- Adapt payload for specific provider requirements ---
        #####################################################################
        final_api_payload_for_provider = llm_messages_payload  # Default to the multimodal one

        if api_endpoint.lower() == 'deepseek':
            logging.info("Adapting message payload for DeepSeek (text-only string content).")
            adapted_payload_for_deepseek = []
            for msg_obj in llm_messages_payload:
                role = msg_obj.get("role")
                content_input = msg_obj.get("content")

                text_parts = []
                image_detected_and_ignored = False

                if isinstance(content_input, list):  # Standard multimodal format
                    for part in content_input:
                        if part.get("type") == "text":
                            text_parts.append(part.get("text", ""))
                        elif part.get("type") == "image_url":
                            image_detected_and_ignored = True
                            # You already log warnings about image handling during payload construction if current_image_input is present.
                            # This is an additional safeguard if history contains images.
                            logging.warning(
                                f"DeepSeek (API: {api_endpoint}) does not support images. Image part in role '{role}' will be ignored.")
                elif isinstance(content_input, str):  # If content somehow became a string already
                    text_parts.append(content_input)
                else:
                    logging.warning(
                        f"Unexpected content type for role '{role}' in payload for DeepSeek: {type(content_input)}. Attempting to coerce to string.")
                    text_parts.append(str(content_input))

                final_text_content = "\n".join(text_parts).strip()

                # If only an image was present and ignored, the content might be empty.
                # DeepSeek might not like empty content strings, decide on a placeholder if necessary.
                # For now, we'll pass the potentially empty string.
                if image_detected_and_ignored and not final_text_content:
                    logging.debug(
                        f"Message for role '{role}' contained only an image (ignored for DeepSeek), resulting in empty text content.")

                adapted_payload_for_deepseek.append({"role": role, "content": final_text_content})

            final_api_payload_for_provider = adapted_payload_for_deepseek

            # Optional: Re-log the adapted payload for DeepSeek to confirm its structure
            logging.debug("Debug - Chat Function - Adapted LLM Payload for DeepSeek:")
            for i, msg_p in enumerate(final_api_payload_for_provider):
                # Ensure content is logged even if it's long by truncating
                content_preview = str(msg_p.get('content', ''))[:100] + (
                    '...' if len(str(msg_p.get('content', ''))) > 100 else '')
                logging.debug(f"  Msg {i}: Role: {msg_p['role']}, Content: '{content_preview}'")

        # --- Call the LLM via the updated chat_api_call ---
        response = chat_api_call(
            api_endpoint=api_endpoint,
            api_key=api_key,
            messages_payload=final_api_payload_for_provider,
            temp=temperature_float,
            system_message=system_message,
            streaming=streaming,
            minp=minp, maxp=maxp, model=model, topp=topp, topk=topk,
            # Pass through new params from ChatCompletionRequest
            max_tokens=llm_max_tokens,
            seed=llm_seed,
            stop=llm_stop,
            response_format=llm_response_format,
            n=llm_n,
            user_identifier=llm_user_identifier,
            logprobs=llm_logprobs,
            top_logprobs=llm_top_logprobs,
            logit_bias=llm_logit_bias,
            presence_penalty=llm_presence_penalty,
            frequency_penalty=llm_frequency_penalty,
            tools=llm_tools,
            tool_choice=llm_tool_choice,
            llm_fixed_tokens_kobold=llm_fixed_tokens_kobold # Added
        )

        if streaming:
            logging.debug("Chat Function - Response: Streaming Generator")
            return response
        else:
            chat_duration = time.time() - start_time
            log_histogram("chat_duration_multimodal", chat_duration, labels={"api_endpoint": api_endpoint})
            log_counter("chat_success_multimodal", labels={"api_endpoint": api_endpoint})
            logging.debug(f"Chat Function - Response (first 500 chars): {str(response)[:500]}")

            loaded_config_data = load_settings()
            post_gen_replacement_config = loaded_config_data.get('chat_dictionaries', {}).get('post_gen_replacement')
            if post_gen_replacement_config and isinstance(response, str):
                post_gen_replacement_dict_path = loaded_config_data.get('chat_dictionaries', {}).get('post_gen_replacement_dict')
                if post_gen_replacement_dict_path and os.path.exists(post_gen_replacement_dict_path):
                    try:
                        parsed_dict_entries = parse_user_dict_markdown_file(post_gen_replacement_dict_path)
                        if parsed_dict_entries:
                            post_gen_chat_dict_objects = [
                                ChatDictionary(key=k, content=str(v)) for k, v in parsed_dict_entries.items()
                            ]
                            if post_gen_chat_dict_objects:
                                response = process_user_input(response, post_gen_chat_dict_objects)
                                # The original warning log can be removed or changed to a debug log if successfully applied.
                                logging.debug(
                                    f"Response after post-gen replacement (first 500 chars): {str(response)[:500]}")
                            else:
                                logging.debug("Post-gen dictionary parsed but resulted in no ChatDictionary objects.")
                        else:
                            logging.debug(
                                f"Post-gen replacement dictionary at {post_gen_replacement_dict_path} was empty or failed to parse.")

                        logging.debug(f"Response after post-gen replacement (first 500 chars): {str(response)[:500]}")
                    except Exception as e_post_gen:
                        logging.error(f"Error during post-generation replacement: {e_post_gen}", exc_info=True)
                else:
                    logging.warning("Post-gen replacement enabled but dict file not found/configured.")
            # For non-streaming, apply stripping logic if enabled
            logging.info(f"Non-streaming response - strip_thinking_tags setting: {strip_thinking_tags}")
            if not streaming and isinstance(response, str) and strip_thinking_tags:
                # Regex to find all <think>...</think> blocks, non-greedy
                # Match both <think> and <thinking> tags
                think_blocks = list(re.finditer(r"<think(?:ing)?>.*?</think(?:ing)?>", response, re.DOTALL))

                if len(think_blocks) > 1:
                    logging.debug(f"Processing thinking tags for non-streaming. Found {len(think_blocks)} blocks.")
                    # Keep only the last think block
                    text_parts = []
                    last_kept_block_end = 0
                    for i, block in enumerate(think_blocks):
                        if i < len(think_blocks) - 1: # This is a block to remove
                            text_parts.append(response[last_kept_block_end:block.start()]) # Text before this block
                            last_kept_block_end = block.end() # Skip this block
                    # Add the text after the last removed block, which includes the final think block and any subsequent text
                    text_parts.append(response[last_kept_block_end:])
                    response = "".join(text_parts)
                    logging.debug(f"Response after stripping all but last think block: {response[:200]}...")
                elif think_blocks: # Only one block, or stripping not needed / done
                    logging.info(f"Thinking tags: {len(think_blocks)} block(s) found, no stripping needed or already processed if only one.")
            elif not streaming and isinstance(response, str) and not strip_thinking_tags:
                logging.info(f"Not stripping thinking tags from non-streaming response - setting is disabled")

            # For streaming=True, stripping logic should be applied by the receiver
            # of the stream (e.g., in app.py's on_stream_done event handler).
            return response

    except Exception as e:
        log_counter("chat_error_multimodal", labels={"api_endpoint": api_endpoint, "error": str(e)})
        logging.error(f"Error in multimodal chat function: {str(e)}", exc_info=True)
        # Consider if the error format should change from just a string
        return f"An error occurred in the chat function: {str(e)}"


def parse_tool_calls_from_response(response: Union[str, Dict[str, Any]]) -> Optional[List[Dict[str, Any]]]:
    """
    Extract tool calls from LLM response, handling various provider formats.
    
    Args:
        response: The LLM response, either as a string or dict
        
    Returns:
        List of tool calls in OpenAI format, or None if no tool calls found
        
    Example tool call format:
    [{
        "id": "call_function_name_timestamp",
        "type": "function",
        "function": {
            "name": "function_name",
            "arguments": "json_string_of_arguments"
        }
    }]
    """
    # Handle string responses by attempting to parse as JSON
    if isinstance(response, str):
        try:
            response = json.loads(response)
        except json.JSONDecodeError:
            logging.debug("Response is not valid JSON, no tool calls to extract")
            return None
    
    if not isinstance(response, dict):
        return None
    
    # Check for tool calls in various locations
    tool_calls = None
    
    # OpenAI format: response.message.tool_calls
    if isinstance(response.get("message"), dict):
        tool_calls = response["message"].get("tool_calls")
    
    # Alternative: tool_calls at root level
    if not tool_calls:
        tool_calls = response.get("tool_calls")
    
    # Legacy format: single function_call
    if not tool_calls and "function_call" in response.get("message", {}):
        func_call = response["message"]["function_call"]
        tool_calls = [{
            "id": f"call_{func_call.get('name', 'unknown')}_{int(time.time())}",
            "type": "function",
            "function": func_call
        }]
    
    # Anthropic format: check for tool_use in stop_reason
    if not tool_calls and response.get("stop_reason") == "tool_use":
        # Tool calls might be in content blocks
        content = response.get("content", [])
        if isinstance(content, list):
            tool_calls = []
            for block in content:
                if block.get("type") == "tool_use":
                    tool_calls.append({
                        "id": block.get("id", f"call_{int(time.time())}"),
                        "type": "function",
                        "function": {
                            "name": block.get("name"),
                            "arguments": json.dumps(block.get("input", {}))
                        }
                    })
    
    # Validate tool calls format
    if tool_calls and isinstance(tool_calls, list):
        valid_calls = []
        for call in tool_calls:
            if isinstance(call, dict) and "function" in call:
                # Ensure required fields exist
                if not call.get("id"):
                    call["id"] = f"call_{int(time.time())}"
                if not call.get("type"):
                    call["type"] = "function"
                valid_calls.append(call)
        return valid_calls if valid_calls else None
    
    return None


def save_chat_history_to_db_wrapper(
    db: CharactersRAGDB,
    chatbot_history: List[Dict[str, Any]], # CHANGED: Expects List of OpenAI message objects
    conversation_id: Optional[str],
    # Renamed for clarity: these are for identifying the character for association,
    # not for the content of the messages themselves.
    media_content_for_char_assoc: Optional[Dict[str, Any]],
    media_name_for_char_assoc: Optional[str] = None,
    character_name_for_chat: Optional[str] = None
) -> Tuple[Optional[str], str]:
    """
    Saves or updates a chat conversation in the database.

    This function handles associating the chat with a character (either specified,
    derived from media, or a default character), creating a new conversation entry
    if `conversation_id` is None, or updating an existing conversation by
    soft-deleting old messages and adding new ones.

    The `chatbot_history` is expected in OpenAI's message format: a list of
    dictionaries, each with 'role' and 'content' keys. Multimodal content
    (text and images as base64 data URIs) within the 'content' field is supported.

    Args:
        db: An instance of `CharactersRAGDB` for database operations.
        chatbot_history: The chat history as a list of OpenAI message objects.
                         Each object is a dict: `{'role': str, 'content': Union[str, List[Dict]]}`.
                         Content lists support `{'type': 'text', 'text': str}` and
                         `{'type': 'image_url', 'image_url': {'url': 'data:image/...;base64,...'}}`.
        conversation_id: The ID of an existing conversation to update. If None,
                         a new conversation is created.
        media_content_for_char_assoc: Optional dictionary containing media details.
                                      Used to derive a character name for association
                                      if `character_name_for_chat` or `media_name_for_char_assoc`
                                      are not provided. Expected to have a 'content' key
                                      which might be a JSON string or dict with a 'title'.
        media_name_for_char_assoc: Optional name of media to associate with a character.
                                   Used if `character_name_for_chat` is not provided.
        character_name_for_chat: Optional name of the character for this chat.
                                 If provided, the chat is associated with this character.

    Returns:
        A tuple containing:
        - `Optional[str]`: The conversation ID (new or existing). None on critical failure
                           to create a conversation entry.
        - `str`: A status message indicating success or failure.
    """
    log_counter("save_chat_history_to_db_attempt")
    start_time = time.time()
    logging.info(f"Saving chat history (OpenAI format). Conversation ID: {conversation_id}, Character: {character_name_for_chat}, Num messages: {len(chatbot_history)}")

    try:
        # The DB connection is managed by the CharactersRAGDB instance (`db`)
        # No need for direct `get_db_connection` or manual sqlite3 corruption checks here.
        # The `db` instance methods will raise exceptions if issues occur.

        associated_character_id: Optional[int] = None
        final_character_name_for_title = "Unknown Character" # For conversation title

        # --- Character Association Logic (largely same as your provided version) ---
        char_lookup_name = character_name_for_chat
        if not char_lookup_name and media_name_for_char_assoc:
            char_lookup_name = media_name_for_char_assoc

        # Fallback to media_content_for_char_assoc to derive char_lookup_name if others are None
        if not char_lookup_name and media_content_for_char_assoc:
            content_details = media_content_for_char_assoc.get('content')
            if isinstance(content_details, str):
                try: content_details = json.loads(content_details)
                except json.JSONDecodeError: content_details = {}
            if isinstance(content_details, dict):
                char_lookup_name = content_details.get('title')

        if char_lookup_name:
            try:
                character = db.get_character_card_by_name(char_lookup_name)
                if character:
                    associated_character_id = character['id']
                    final_character_name_for_title = character['name']
                    logging.info(f"Chat will be associated with specific character '{final_character_name_for_title}' (ID: {associated_character_id}).")
                else:
                    logging.error(f"Intended specific character '{char_lookup_name}' not found in DB. Chat save aborted.")
                    return conversation_id, f"Error: Specific character '{char_lookup_name}' intended for this chat was not found. Cannot save chat."
            except CharactersRAGDBError as e:
                logging.error(f"DB error looking up specific character '{char_lookup_name}': {e}")
                return conversation_id, f"DB error finding specific character: {e}"
        else:
            logging.info("No specific character name for chat. Using Default Character.")
            try:
                default_char = db.get_character_card_by_name(DEFAULT_CHARACTER_NAME)
                if default_char:
                    associated_character_id = default_char['id']
                    final_character_name_for_title = default_char['name']
                    logging.info(f"Chat will be associated with '{DEFAULT_CHARACTER_NAME}' (ID: {associated_character_id}).")
                else:
                    # This is a critical state: no specific char, and default char is missing (should have been created by DB dep)
                    logging.error(f"'{DEFAULT_CHARACTER_NAME}' is missing from the DB and no specific character was provided. Chat save aborted.")
                    return conversation_id, f"Error: Critical - '{DEFAULT_CHARACTER_NAME}' is missing. Cannot save chat."
            except CharactersRAGDBError as e:
                logging.error(f"DB error looking up '{DEFAULT_CHARACTER_NAME}': {e}")
                return conversation_id, f"DB error finding '{DEFAULT_CHARACTER_NAME}': {e}"

        # Ensure we have a character_id to proceed
        if associated_character_id is None:
             # This should be an unreachable state if the logic above is correct.
             logging.critical(f"Logic error: associated_character_id is None after character lookup. Chat save aborted.")
             return conversation_id, "Critical internal error: Could not determine character for chat."
        # --- End Character Association ---

        current_conversation_id = conversation_id
        is_new_conversation = not current_conversation_id

        # --- Create or Prepare Conversation ---
        if is_new_conversation:
            conv_title_base = f"Chat with {final_character_name_for_title}"

            timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
            conversation_title = f"{conv_title_base} ({timestamp_str})"

            conv_data = {
                'character_id': associated_character_id,
                'title': conversation_title,
                # 'root_id' will be set to new conv_id by add_conversation if not provided
                'client_id': db.client_id  # Use client_id from the DB instance
            }
            try:
                current_conversation_id = db.add_conversation(conv_data)
                if not current_conversation_id:  # Should not happen if add_conversation raises on failure
                    return None, "Failed to create new conversation in DB."
                logging.info(f"Created new conv ID: {current_conversation_id} for char ID: {associated_character_id} ('{final_character_name_for_title}')")
            except (InputError, ConflictError, CharactersRAGDBError) as e:
                logging.error(f"Error creating new conversation: {e}", exc_info=True)
                return None, f"Error creating conversation: {e}"
        else: # Resaving existing conversation
            logging.info(f"Resaving history for existing conv ID: {current_conversation_id}. Char context ID: {associated_character_id} ('{final_character_name_for_title}')")
            try:
                with db.transaction():
                    existing_conv_details = db.get_conversation_by_id(current_conversation_id)
                    if not existing_conv_details:
                        logging.error(f"Cannot resave: Conversation {current_conversation_id} not found.")
                        return current_conversation_id, f"Error: Conversation {current_conversation_id} not found for resaving."

                    # Important: Ensure the existing conversation being updated belongs to the character context we're in.
                    # This prevents accidentally overwriting a chat from Character A if the current UI context is Character B (or Default).
                    if existing_conv_details.get('character_id') != associated_character_id:
                        # Fetch names for better logging
                        existing_char_of_conv = db.get_character_card_by_id(existing_conv_details.get('character_id'))
                        existing_char_name = existing_char_of_conv['name'] if existing_char_of_conv else "ID "+str(existing_conv_details.get('character_id'))
                        logging.error(f"Cannot resave: Conversation {current_conversation_id} (for char '{existing_char_name}') does not match current character context '{final_character_name_for_title}' (ID: {associated_character_id}).")
                        return current_conversation_id, "Error: Mismatch in character association for resaving chat. The conversation belongs to a different character."

                    existing_messages = db.get_messages_for_conversation(current_conversation_id, limit=10000, order_by_timestamp="ASC")
                    logging.info(f"Found {len(existing_messages)} existing messages to soft-delete for conv {current_conversation_id}.")
                    for msg in existing_messages:
                        db.soft_delete_message(msg['id'], msg['version'])
            except (InputError, ConflictError, CharactersRAGDBError) as e:
                logging.error(f"Error preparing existing conversation {current_conversation_id} for resave: {e}", exc_info=True)
                return current_conversation_id, f"Error during resave prep: {e}"
        # --- End Create or Prepare Conversation ---

        # --- Save Messages (Handles new OpenAI format) ---
        try:
            # Ensure transaction wraps message saving, especially for new conversations or full resaves.
            # For resaves, the transaction is already started above.
            with db.transaction() if is_new_conversation else db.transaction(): # No-op if already in transaction for resave
                message_save_count = 0
                for i, message_obj in enumerate(chatbot_history):
                    sender = message_obj.get("role")
                    if not sender or sender == "system": # Don't save system prompts as messages
                        logging.debug(f"Skipping message with role '{sender}' at index {i}")
                        continue

                    text_content_parts = []
                    image_data_bytes: Optional[bytes] = None
                    image_mime_type_str: Optional[str] = None

                    content_data = message_obj.get("content")

                    if isinstance(content_data, str): # Simple text content (e.g., from older history or some assistant responses)
                        text_content_parts.append(content_data)
                    elif isinstance(content_data, list): # OpenAI multimodal content list
                        for part in content_data:
                            part_type = part.get("type")
                            if part_type == "text":
                                text_content_parts.append(part.get("text", ""))
                            elif part_type == "image_url":
                                image_url_dict = part.get("image_url", {})
                                url_str = image_url_dict.get("url", "")
                                if url_str.startswith("data:") and ";base64," in url_str:
                                    try:
                                        header, b64_data = url_str.split(";base64,", 1)
                                        image_mime_type_str = header.split("data:", 1)[1] if "data:" in header else None
                                        if image_mime_type_str: # Ensure mime type was found
                                            image_data_bytes = base64.b64decode(b64_data)
                                            logging.debug(f"Decoded image for saving (MIME: {image_mime_type_str}, Size: {len(image_data_bytes) if image_data_bytes else 0}) for msg {i} in conv {current_conversation_id}")
                                        else:
                                            logging.warning(f"Could not parse MIME type from data URI: {url_str[:60]}...")
                                            text_content_parts.append("<Error: Malformed image data URI in history>")
                                    except Exception as e_b64:
                                        logging.error(f"Error decoding base64 image from history for msg {i} in conv {current_conversation_id}: {e_b64}")
                                        text_content_parts.append("<Error: Failed to decode image data from history>")
                                else:
                                    # If it's a non-data URL, store it as text.
                                    logging.debug(f"Storing non-data image URL as text: {url_str}")
                                    text_content_parts.append(f"<Image URL: {url_str}>")
                    else:
                        logging.warning(f"Unsupported message content type at index {i}: {type(content_data)}")
                        text_content_parts.append(f"<Unsupported content type: {type(content_data)}>")

                    final_text_content = "\n".join(text_content_parts).strip()

                    # A message must have either text or an image to be saved.
                    if not final_text_content and not image_data_bytes:
                        logging.warning(f"Skipping empty message (no text or decodable image) at index {i} for conv {current_conversation_id}")
                        continue

                    db.add_message({
                        'conversation_id': current_conversation_id,
                        'sender': sender, # 'user' or 'assistant'
                        'content': final_text_content,
                        'image_data': image_data_bytes,
                        'image_mime_type': image_mime_type_str,
                        'client_id': db.client_id
                        # timestamp, version, ranking, etc., handled by db.add_message
                    })
                    message_save_count +=1
                logging.info(f"Successfully saved {message_save_count} messages to conversation {current_conversation_id}.")

                # If resaving (not a new conversation), update conversation's last_modified and version
                if not is_new_conversation:
                    conv_details_for_update = db.get_conversation_by_id(current_conversation_id)
                    if conv_details_for_update:
                        db.update_conversation(
                            current_conversation_id,
                            {'title': conv_details_for_update.get('title')}, # Keep title, just bump version/timestamp
                            conv_details_for_update['version']
                        )
                    else:
                        logging.error(f"Conversation {current_conversation_id} disappeared before final metadata update during resave.")


        except (InputError, ConflictError, CharactersRAGDBError) as e:
            logging.error(f"Error saving messages to conversation {current_conversation_id}: {e}", exc_info=True)
            return current_conversation_id, f"Error saving messages: {e}"
        # --- End Save Messages ---

        save_duration = time.time() - start_time
        log_histogram("save_chat_history_to_db_duration", save_duration)
        log_counter("save_chat_history_to_db_success")

        return current_conversation_id, "Chat history saved successfully!"

    except Exception as e:
        log_counter("save_chat_history_to_db_error", labels={"error": str(e)})
        error_message = f"Failed to save chat history due to an unexpected error: {str(e)}"
        logging.error(error_message, exc_info=True)
        return conversation_id, error_message


# FIXME - turn into export function
def save_chat_history(
    history: List[Union[Tuple[Optional[str], Optional[str]], Dict[str, Any]]],
    conversation_id: Optional[str],
    media_content: Optional[Dict[str, Any]],
    db_instance: Optional[CharactersRAGDB] = None # Added for generate_chat_history_content
) -> Optional[str]:
    """
    Saves chat history to a uniquely named JSON file in a temporary directory.

    FIXME: This function is marked to be potentially turned into a more generic
    export function. Currently, it generates content using
    `generate_chat_history_content` and saves it locally.

    Args:
        history: The chat history. Can be a list of (user_msg, bot_msg) tuples
                 or a list of OpenAI message dicts.
        conversation_id: The ID of the conversation, used for naming the file
                         and potentially fetching details via `db_instance`.
        media_content: Optional dictionary containing media details, used by
                       `generate_chat_history_content` to derive a conversation name
                       if `conversation_id` doesn't yield one.
        db_instance: Optional `CharactersRAGDB` instance passed to
                     `generate_chat_history_content` to fetch conversation title.

    Returns:
        The absolute path to the saved JSON file, or None if an error occurred.
    """
    log_counter("save_chat_history_attempt")
    start_time = time.time()
    try:
        content, conversation_name = generate_chat_history_content(history, conversation_id, media_content)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        safe_conversation_name = re.sub(r'[^a-zA-Z0-9_-]', '_', conversation_name)
        base_filename = f"{safe_conversation_name}_{timestamp}.json"

        # Create a secure temporary file
        temp_file_path = create_secure_temp_file(content, suffix='.json', prefix='chat_export_')

        try:
            # Generate a unique filename
            unique_filename = generate_unique_filename(os.path.dirname(temp_file_path), base_filename)
            final_path = os.path.join(os.path.dirname(temp_file_path), unique_filename)

            # Rename the temporary file to the unique filename
            os.rename(temp_file_path, final_path)
        except Exception:
            # Clean up temp file if renaming fails
            secure_delete_file(temp_file_path)
            raise

        save_duration = time.time() - start_time
        log_histogram("save_chat_history_duration", save_duration)
        log_counter("save_chat_history_success")
        return final_path
    except Exception as e:
        log_counter("save_chat_history_error", labels={"error": str(e)})
        logging.error(f"Error saving chat history: {str(e)}")
        return None


def get_conversation_name(conversation_id: Optional[str], db_instance: Optional[CharactersRAGDB] = None) -> Optional[str]:
    """
    Retrieves the title of a conversation from the database.

    Args:
        conversation_id: The ID of the conversation.
        db_instance: An optional instance of `CharactersRAGDB` to use for DB lookup.

    Returns:
        The conversation title as a string if found, otherwise None.
    """
    if db_instance and conversation_id:
        try:
            conversation = db_instance.get_conversation_by_id(conversation_id)
            if conversation and conversation.get('title'):
                return conversation['title']
        except Exception as e:
            logging.warning(f"Could not fetch conversation title from DB for {conversation_id}: {e}")
    # Fallback or if no DB instance provided
    # This part of the original logic is unclear how it worked without DB.
    # For now, returning None if not found in DB.
    return None


def generate_chat_history_content(
    history: List[Union[Tuple[Optional[str], Optional[str]], Dict[str, Any]]],
    conversation_id: Optional[str],
    media_content: Optional[Dict[str, Any]],
    db_instance: Optional[CharactersRAGDB] = None
) -> Tuple[str, str]:
    """
    Generates JSON content representing the chat history and determines a conversation name.

    The conversation name is fetched from the database using `conversation_id` if
    `db_instance` is provided. Otherwise, it's derived from `media_content` or
    a timestamp. The history is formatted into a list of role-content dictionaries.

    Args:
        history: The chat history. Can be a list of (user_msg, bot_msg) tuples
                 or a list of OpenAI message dicts (`{'role': ..., 'content': ...}`).
        conversation_id: Optional ID of the conversation.
        media_content: Optional dictionary with media details, used for deriving
                       conversation name as a fallback.
        db_instance: Optional `CharactersRAGDB` instance for fetching conversation title.

    Returns:
        A tuple containing:
        - `str`: JSON string of the chat data.
        - `str`: The derived or fetched conversation name.
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Try to get conversation name from DB if possible
    conversation_name = None
    if conversation_id:
        conversation_name = get_conversation_name(conversation_id, db_instance)

    if not conversation_name:  # Fallback logic
        media_name_extracted = extract_media_name(media_content)  # media_content is the original complex object
        if media_name_extracted:
            conversation_name = f"{media_name_extracted}-chat-{timestamp}"
        else:
            conversation_name = f"chat-{timestamp}"

    chat_data = {
        "conversation_id": conversation_id,  # Can be None if new chat not yet saved to DB
        "conversation_name": conversation_name,
        "timestamp": timestamp,
        "history": [],
        # The original history format seemed to be a list of tuples (user, bot) or just a list of messages
        # The new DB stores messages individually. This JSON should reflect the 'chatbot' structure if it's for UI.
        # Assuming 'history' is like chatbot: List[Tuple[Optional[str], Optional[str]]]
    }

    current_turn = []
    for item in history:  # Iterating through the provided history structure
        if isinstance(item, tuple) and len(item) == 2:  # Expected (user_msg, bot_msg)
            user_msg, bot_msg = item
            if user_msg is not None:
                chat_data["history"].append({"role": "user", "content": user_msg})
            if bot_msg is not None:
                chat_data["history"].append(
                    {"role": "assistant", "content": bot_msg})  # Changed "bot" to "assistant" for consistency
        elif isinstance(item, dict) and "role" in item and "content" in item:  # Already in desired format
            chat_data["history"].append(item)
        else:
            logging.warning(f"Unexpected item format in history for JSON export: {item}")

    return json.dumps(chat_data, indent=2), conversation_name  # Return the derived/fetched name

def extract_media_name(media_content: Optional[Dict[str, Any]]) -> Optional[str]:
    """
    Extracts a name or title from a media content dictionary.

    It attempts to find a name by looking at common keys like 'title', 'name',
    'media_title', 'webpage_title' within the `media_content` dictionary itself
    or within a nested 'content' field (which might be a JSON string or a sub-dictionary).

    Args:
        media_content: A dictionary potentially containing media metadata.

    Returns:
        The extracted name as a string if found, otherwise None.
    """
    if not media_content or not isinstance(media_content, dict):
        return None

    # Try to get from 'content' which might be a JSON string or a dict
    content_field = media_content.get('content')
    parsed_content = None

    if isinstance(content_field, str):
        try:
            parsed_content = json.loads(content_field)
        except json.JSONDecodeError:
            logging.warning("Failed to parse media_content['content'] JSON string in extract_media_name")
            # It might be a plain string title itself, or not what we expect
            # For now, if it's a non-JSON string, we don't assume it's the name.
            parsed_content = {}
    elif isinstance(content_field, dict):
        parsed_content = content_field

    if isinstance(parsed_content, dict):
        # Check common keys for a title or name
        name = parsed_content.get('title') or \
               parsed_content.get('name') or \
               parsed_content.get('media_title') or \
               parsed_content.get('webpage_title')
        if name: return name

    # Fallback to top-level keys in media_content itself if 'content' didn't yield a name
    name_top_level = media_content.get('title') or \
                     media_content.get('name') or \
                     media_content.get('media_title')
    if name_top_level: return name_top_level

    logging.warning(f"Could not extract a clear media name from media_content: {str(media_content)[:200]}")
    return None

# FIXME
# update_chat_content Note Parsing:
#     Issue: raw_note_content_field can be a plain string or a JSON string. This dual nature can be brittle.
#     Improvement: Enforce a consistent structure for notes.content in the database if it's meant to hold structured data. If it's always JSON, then json.loads can be used directly (with error handling). If it can be either, the current logic is a necessary workaround but adds complexity.
def update_chat_content(
        selected_item: Optional[str],
        use_content: bool,
        use_summary: bool,
        use_prompt: bool,
        item_mapping: Dict[str, str],
        db_instance: CharactersRAGDB
) -> Tuple[Dict[str, str], List[str]]:
    """
    Fetches content from a database 'note' to be used as RAG context in a chat.

    The function retrieves a note by ID (derived from `selected_item` via
    `item_mapping`). It then extracts parts like 'content', 'summary', and 'prompt'
    from the note's data based on the boolean flags `use_content`, `use_summary`,
    and `use_prompt`.

    The note's 'content' field in the database can be a plain string or a JSON string
    containing structured data (e.g., `{"content": "...", "summary": "..."}`).
    This function attempts to parse it accordingly.

    FIXME: The dual nature of `note_data.content` (plain string or JSON string)
    can be brittle. Enforcing a consistent structure for notes.content in the
    database (e.g., always JSON if structured data is intended) would be an
    improvement.

    Args:
        selected_item: The display name of the item selected by the user.
        use_content: Boolean flag to include the main 'content' part.
        use_summary: Boolean flag to include the 'summary' part.
        use_prompt: Boolean flag to include the 'prompt' part.
        item_mapping: A dictionary mapping display names (like `selected_item`)
                      to note IDs (UUID strings).
        db_instance: An instance of `CharactersRAGDB` for database operations.

    Returns:
        A tuple containing:
        - `Dict[str, str]`: A dictionary where keys are part names (e.g., "content",
                            "summary") and values are their string content. This is
                            intended for the `media_content` argument of the `chat` function.
        - `List[str]`: A list of selected part names (e.g., ["content", "summary"]).
    """
    log_counter("update_chat_content_attempt")
    start_time = time.time()
    logging.debug(f"Debug - Update Chat Content - Selected Item: {selected_item}")

    # This function's purpose seems to be to fetch content (possibly from a 'note' in the new DB)
    # and prepare it for the 'chat' function's 'media_content' input.
    # The 'media_content' that 'chat' receives is a simple dict of strings: {'summary': '...', 'content': '...'}

    output_media_content_for_chat: Dict[str, str] = {}  # This will be passed to chat()
    selected_parts_names: List[str] = []

    if selected_item and selected_item in item_mapping:
        note_id = item_mapping[selected_item]  # Assuming media_id from mapping is a note_id (UUID string)

        try:
            note_data = db_instance.get_note_by_id(note_id)
        except CharactersRAGDBError as e:
            logging.error(f"Error fetching note {note_id} for chat content: {e}", exc_info=True)
            note_data = None
        except Exception as e_gen:  # Catch any other unexpected error during DB fetch
            logging.error(f"Unexpected error fetching note {note_id}: {e_gen}", exc_info=True)
            note_data = None

        if note_data:
            # The content of the note ('note_data.content') might be:
            # 1. A plain string (e.g., the main transcript/content).
            # 2. A JSON string containing structured data like {"content": "...", "summary": "...", "prompt": "..."}.

            raw_note_content_field = note_data.get('content', '')  # The actual text from notes.content
            structured_content_from_note: Dict[str, str] = {}

            # Try to parse raw_note_content_field as JSON
            if isinstance(raw_note_content_field, str) and \
                    raw_note_content_field.strip().startswith('{') and \
                    raw_note_content_field.strip().endswith('}'):
                try:
                    parsed_json = json.loads(raw_note_content_field)
                    if isinstance(parsed_json, dict):
                        # Filter to ensure only string values are taken for safety
                        structured_content_from_note = {k: str(v) for k, v in parsed_json.items() if
                                                        isinstance(v, (str, int, float, bool))}
                        logging.debug(f"Parsed note's content field (ID: {note_id}) as JSON.")
                    else:
                        # JSON, but not a dict. Treat main content as the raw string.
                        structured_content_from_note['content'] = raw_note_content_field
                        logging.debug(
                            f"Note's content field (ID: {note_id}) was JSON but not a dict. Using raw string for 'content'.")
                except json.JSONDecodeError:
                    # Not valid JSON, treat it as the main 'content' part
                    structured_content_from_note['content'] = raw_note_content_field
                    logging.debug(f"Note's content field (ID: {note_id}) is not JSON. Using raw string for 'content'.")
            else:  # Not a JSON string, treat as main 'content'
                structured_content_from_note['content'] = raw_note_content_field
                logging.debug(f"Note's content field (ID: {note_id}) is a plain string. Using for 'content'.")

            # Populate `output_media_content_for_chat` based on `use_` flags and what's in `structured_content_from_note`
            if use_content and "content" in structured_content_from_note:
                output_media_content_for_chat["content"] = structured_content_from_note["content"]
                selected_parts_names.append("content")

            if use_summary and "summary" in structured_content_from_note:
                output_media_content_for_chat["summary"] = structured_content_from_note["summary"]
                selected_parts_names.append("summary")
            elif use_summary and "content" in structured_content_from_note and "summary" not in output_media_content_for_chat:
                # Fallback: if summary requested but not explicitly present, use first N words of content as summary?
                # For now, only include if explicitly present.
                logging.debug("Summary requested but not found in structured note content.")

            if use_prompt and "prompt" in structured_content_from_note:
                output_media_content_for_chat["prompt"] = structured_content_from_note["prompt"]
                selected_parts_names.append("prompt")

            # Add note title as a part, if not already taken by 'content', 'summary', or 'prompt'
            # and if a relevant use_ flag is true (e.g., use_content implies use_title if no other content)
            # This part is a bit ambiguous from the original. Let's assume title is just metadata for now.
            # Or, if note_data['title'] is meaningful as a 'part':
            # if use_title_flag and "title" not in selected_parts_names:
            #    output_media_content_for_chat["title_from_note"] = note_data.get('title', '')
            #    selected_parts_names.append("title_from_note")

            # Debug logging of what was prepared
            logging.debug(f"Prepared media content for chat from note {note_id}:")
            for key, value in output_media_content_for_chat.items():
                logging.debug(f"  {key} (first 100 chars): {str(value)[:100]}")
            logging.debug(f"Selected part names for chat: {selected_parts_names}")

        else:  # Note not found
            logging.warning(f"Note ID {note_id} (from selected_item '{selected_item}') not found in DB.")
            # Return empty, as per original fallback
            output_media_content_for_chat = {}
            selected_parts_names = []

    else:  # No item selected or item not in mapping
        log_counter("update_chat_content_error", labels={"error": str("No item selected or item not in mapping")})
        logging.debug(f"Debug - Update Chat Content - No item selected or item not in mapping: {selected_item}")
        output_media_content_for_chat = {}
        selected_parts_names = []

    update_duration = time.time() - start_time
    log_histogram("update_chat_content_duration", update_duration)
    log_counter("update_chat_content_success" if selected_parts_names else "update_chat_content_noop")

    return output_media_content_for_chat, selected_parts_names

#
# End of Chat functions
#######################################################################################################################


#######################################################################################################################
#
# Chat Dictionary Functions - Moved to Character_Chat/Chat_Dictionary_Lib.py
# Import the functions for backward compatibility
from ..Character_Chat.Chat_Dictionary_Lib import (
    ChatDictionary,
    parse_user_dict_markdown_file,
    process_user_input,
    TokenBudgetExceededWarning
)

#
# End of Chat Dictionary functions
#######################################################################################################################


#######################################################################################################################
#
# Character Card Functions

def save_character(
        db: CharactersRAGDB,
        character_data: Dict[str, Any],
        expected_version: Optional[int] = None
) -> Optional[int]:
    """
    Saves (adds or updates) a character card in the database.

    If a character with the same name already exists, it attempts to update it.
    An `expected_version` can be provided for optimistic concurrency control during updates.
    If the character does not exist, a new one is added.

    Image data in `character_data['image']` is expected as a base64 encoded string;
    it will be decoded to bytes before saving.

    Args:
        db: An instance of `CharactersRAGDB` for database operations.
        character_data: A dictionary containing the character's attributes.
                        Required key: 'name'.
                        Optional keys: 'description', 'personality', 'scenario',
                        'system_prompt' (or 'system'), 'post_history_instructions' (or 'post_history'),
                        'first_message' (or 'mes_example_greeting'), 'message_example' (or 'mes_example'),
                        'creator_notes', 'alternate_greetings' (list/JSON), 'tags' (list/JSON),
                        'creator', 'character_version', 'extensions' (dict/JSON), 'image' (base64 string).
        expected_version: Optional integer. If provided for an update, the character's
                          current version in the DB must match this value.

    Returns:
        The character's ID (integer) if successfully saved or updated.
        Returns None if the name is missing, or if a DB error, conflict,
        or other exception occurs.
    """
    log_counter("save_character_attempt")
    start_time = time.time()

    char_name = character_data.get('name')
    if not char_name:
        logging.error("Character name is required to save.")
        return None

    db_card_data_full = { # Template for all possible fields for add_character_card
        'name': char_name,
        'description': character_data.get('description'),
        'personality': character_data.get('personality'),
        'scenario': character_data.get('scenario'),
        'system_prompt': character_data.get('system_prompt', character_data.get('system')),  # common alternative key
        'post_history_instructions': character_data.get('post_history_instructions',
                                                        character_data.get('post_history')),
        'first_message': character_data.get('first_message', character_data.get('mes_example_greeting')),
        'message_example': character_data.get('message_example', character_data.get('mes_example')),
        'creator_notes': character_data.get('creator_notes'),
        'alternate_greetings': character_data.get('alternate_greetings'),  # Should be list or JSON string
        'tags': character_data.get('tags'),  # Should be list or JSON string
        'creator': character_data.get('creator'),
        'character_version': character_data.get('character_version'),
        'extensions': character_data.get('extensions')  # Should be dict or JSON string
    }

    # Handle image: convert base64 to bytes if present
    if 'image' in character_data and character_data['image']:
        try:
            # Assuming character_data['image'] is base64 string. Remove data URL prefix if present.
            img_b64_data = character_data['image']
            if ',' in img_b64_data:  # e.g. data:image/png;base64,xxxxx
                img_b64_data = img_b64_data.split(',', 1)[1]
            db_card_data_full['image'] = base64.b64decode(img_b64_data)
        except Exception as e_img:
            logging.error(f"Error decoding character image for {char_name}: {e_img}")
            db_card_data_full['image'] = None
    try:
        # Check if character exists for an "upsert" like behavior
        existing_char = db.get_character_card_by_name(char_name)

        char_id = None
        if existing_char:
            logging.info(f"Character '{char_name}' found (ID: {existing_char['id']}). Attempting update.")
            current_db_version = existing_char['version']
            if expected_version is not None and expected_version != current_db_version:
                logging.error(
                    f"Version mismatch for character '{char_name}'. Expected {expected_version}, DB has {current_db_version}.")
                raise ConflictError(
                    f"Version mismatch for character '{char_name}'. Expected {expected_version}, DB has {current_db_version}",
                    entity="character_cards", entity_id=existing_char['id'])

            # Use current_db_version as expected_version for the update call
            # Merge: Ensure fields not in character_data but in existing_char are preserved if desired
            # For now, db_card_data contains all fields to be set.
            # If a field is None in db_card_data, it will be set to NULL in DB.
            # If character_data omits a field, it's None in db_card_data, so it updates to NULL.
            # This is standard update behavior. If you want partial updates (only update provided fields),
            # then db_card_data should only contain non-None values from character_data.

            # Let's make db_card_data only contain fields that are present in the input character_data
            # so it acts as a partial update for existing characters.
            update_payload = {}
            for key, input_value in character_data.items():
                if key == 'name': continue # Name is for lookup, not part of update payload itself
                if key == 'image': # Handle image separately due to b64 decode
                    update_payload['image'] = db_card_data_full['image'] # Use processed image
                elif key in db_card_data_full: # Check if it's a known/mapped field
                     # Use the value from character_data, allowing explicit nulls if desired by client
                    update_payload[key] = input_value
                # else: field not in db_card_data_full, so ignore

            # Map alternate keys if primary not in update_payload but alternate was
            if 'system' in character_data and 'system_prompt' not in character_data:
                update_payload['system_prompt'] = character_data.get('system')
            if 'post_history' in character_data and 'post_history_instructions' not in character_data:
                update_payload['post_history_instructions'] = character_data.get('post_history')
            if 'mes_example_greeting' in character_data and 'first_message' not in character_data:
                update_payload['first_message'] = character_data.get('mes_example_greeting')
            if 'mes_example' in character_data and 'message_example' not in character_data:
                update_payload['message_example'] = character_data.get('mes_example')


            if not update_payload:
                logging.info(
                    f"No updatable fields provided for existing character '{char_name}'. Skipping update, but returning ID.")
                char_id = existing_char['id']  # No actual update, but considered "saved"
            elif db.update_character_card(existing_char['id'], update_payload, current_db_version):
                char_id = existing_char['id']
                logging.info(f"Character '{char_name}' (ID: {char_id}) updated successfully.")
            # db.update_character_card should raise on failure rather than return False
        else:
            logging.info(f"Character '{char_name}' not found. Attempting to add new.")
            # Use db_card_data_full for adding, as it contains all potential fields
            char_id = db.add_character_card(db_card_data_full)
            if char_id:
                logging.info(f"Character '{char_name}' added successfully with ID: {char_id}.")
            else:  # add_character_card returned None (should raise on error)
                logging.error(f"Failed to add new character '{char_name}'.")

        save_duration = time.time() - start_time
        if char_id:
            log_histogram("save_character_duration", save_duration)
            log_counter("save_character_success")
            return char_id
        else:
            # This path means neither update nor add succeeded in setting char_id
            log_counter("save_character_error_unspecified")
            logging.error(f"Save character operation for '{char_name}' did not result in a character ID.")
            return None

    except ConflictError as e_conflict:
        log_counter("save_character_error_conflict", labels={"error": str(e_conflict)})
        logging.error(f"Conflict error saving character '{char_name}': {e_conflict}")
        # Re-raise or return None. For now, return None for simplicity in this wrapper.
        return None
    except (InputError, CharactersRAGDBError) as e_db:
        log_counter("save_character_error_db", labels={"error": str(e_db)})
        logging.error(f"Database error saving character '{char_name}': {e_db}", exc_info=True)
        return None
    except Exception as e_gen:
        log_counter("save_character_error_generic", labels={"error": str(e_gen)})
        logging.error(f"Generic error saving character '{char_name}': {e_gen}", exc_info=True)
        return None


def load_characters(db: CharactersRAGDB) -> Dict[str, Dict[str, Any]]:
    """
    Loads all character cards from the database.

    The image data (if present) is converted from a database BLOB to a
    base64 encoded string and stored in the 'image_base64' key of each
    character's dictionary.

    Args:
        db: An instance of `CharactersRAGDB` for database operations.

    Returns:
        A dictionary where keys are character names and values are dictionaries
        representing the character cards. Returns an empty dictionary if an
        error occurs or no characters are found.
    """
    log_counter("load_characters_attempt")
    start_time = time.time()
    characters_map: Dict[str, Dict[str, Any]] = {}
    try:
        # list_character_cards returns List[Dict[str, Any]]
        all_cards_list = db.list_character_cards(limit=10000)  # Assuming not too many cards for now

        for card_dict in all_cards_list:
            char_name = card_dict.get('name')
            if char_name:
                # Convert image BLOB back to base64 string for compatibility if needed by UI
                if 'image' in card_dict and isinstance(card_dict['image'], bytes):
                    try:
                        # You might want to store the image format or assume one (e.g., png)
                        card_dict['image_base64'] = base64.b64encode(card_dict['image']).decode('utf-8')
                        # del card_dict['image'] # Optionally remove the bytes version
                    except Exception as e_img_enc:
                        logging.warning(f"Could not encode image for character {char_name}: {e_img_enc}")
                        card_dict['image_base64'] = None

                # The old code had 'image_path'. This is not directly stored.
                # If 'image_path' is needed, it implies images are also saved to disk by save_character,
                # which the new DB logic doesn't do (it stores BLOB).
                # For now, 'image_path' won't be present unless explicitly added.

                characters_map[char_name] = card_dict
            else:
                logging.warning(f"Character card found with no name (ID: {card_dict.get('id')}). Skipping.")

        load_duration = time.time() - start_time
        log_histogram("load_characters_duration", load_duration)
        log_counter("load_characters_success", labels={"character_count": len(characters_map)})
        logging.info(f"Loaded {len(characters_map)} characters from DB.")
        return characters_map

    except CharactersRAGDBError as e_db:
        log_counter("load_characters_error_db", labels={"error": str(e_db)})
        logging.error(f"Database error loading characters: {e_db}", exc_info=True)
        return {}
    except Exception as e_gen:
        log_counter("load_characters_error_generic", labels={"error": str(e_gen)})
        logging.error(f"Generic error loading characters: {e_gen}", exc_info=True)
        return {}


def get_character_names(db: CharactersRAGDB) -> List[str]:
    """
    Retrieves a sorted list of all character names from the database.

    Args:
        db: An instance of `CharactersRAGDB` for database operations.

    Returns:
        A list of character names, sorted alphabetically.
        Returns an empty list if an error occurs or no characters are found.
    """
    log_counter("get_character_names_attempt")
    start_time = time.time()
    names: List[str] = []
    try:
        all_cards = db.list_character_cards(limit=10000)  # Fetch all, then extract names
        for card in all_cards:
            if card.get('name'):
                names.append(card['name'])

        names.sort()  # Optional: sort names alphabetically

        get_names_duration = time.time() - start_time
        log_histogram("get_character_names_duration", get_names_duration)
        log_counter("get_character_names_success", labels={"name_count": len(names)})
        return names
    except CharactersRAGDBError as e_db:
        log_counter("get_character_names_error_db", labels={"error": str(e_db)})
        logging.error(f"Database error getting character names: {e_db}", exc_info=True)
        return []
    except Exception as e_gen:
        log_counter("get_character_names_error_generic", labels={"error": str(e_gen)})
        logging.error(f"Generic error getting character names: {e_gen}", exc_info=True)
        return []

#
# End of Chat_Functions.py
##########################################################################################################################
