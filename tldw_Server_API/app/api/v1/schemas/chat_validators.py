# chat_validators.py
# Description: Advanced validators for chat request schemas
#
# Imports
import re
import uuid
from typing import Any, Optional
from pydantic import field_validator, model_validator, ValidationError
from loguru import logger

#######################################################################################################################
#
# Constants:

# Valid conversation ID pattern (UUID or alphanumeric with hyphens/underscores)
CONVERSATION_ID_PATTERN = re.compile(r'^[a-zA-Z0-9_-]{1,100}$')

# Valid character ID pattern (numeric or name)
CHARACTER_ID_PATTERN = re.compile(r'^(\d+|[a-zA-Z0-9_\- ]{1,100})$')

# Maximum tool definition size
MAX_TOOL_DEFINITION_SIZE = 10000  # characters

# Maximum total request size
MAX_REQUEST_SIZE = 1000000  # 1MB in characters

#######################################################################################################################
#
# Validation Functions:

def validate_conversation_id(conversation_id: Optional[str]) -> Optional[str]:
    """
    Validate conversation ID format.
    
    Args:
        conversation_id: Conversation ID to validate
        
    Returns:
        Validated conversation ID or None
        
    Raises:
        ValueError: If format is invalid
    """
    if conversation_id is None:
        return None
    
    # Check if it's a valid UUID
    try:
        uuid.UUID(conversation_id)
        return conversation_id
    except ValueError:
        pass
    
    # Check against pattern
    if not CONVERSATION_ID_PATTERN.match(conversation_id):
        raise ValueError(
            f"Invalid conversation_id format. Must be UUID or alphanumeric "
            f"with hyphens/underscores (max 100 chars): {conversation_id[:50]}"
        )
    
    return conversation_id


def validate_character_id(character_id: Optional[str]) -> Optional[str]:
    """
    Validate character ID format (can be numeric ID or character name).
    
    Args:
        character_id: Character ID to validate
        
    Returns:
        Validated character ID or None
        
    Raises:
        ValueError: If format is invalid
    """
    if character_id is None:
        return None
    
    if not CHARACTER_ID_PATTERN.match(character_id):
        raise ValueError(
            f"Invalid character_id format. Must be numeric or valid name "
            f"(alphanumeric with spaces, hyphens, underscores, max 100 chars): {character_id[:50]}"
        )
    
    return character_id


def validate_tool_definitions(tools: Optional[list]) -> Optional[list]:
    """
    Validate tool definitions for function calling.
    
    Args:
        tools: List of tool definitions
        
    Returns:
        Validated tools or None
        
    Raises:
        ValueError: If tools are invalid
    """
    if tools is None:
        return None
    
    if not isinstance(tools, list):
        raise ValueError("Tools must be a list")
    
    if len(tools) > 128:
        raise ValueError(f"Too many tools defined (max 128, got {len(tools)})")
    
    for idx, tool in enumerate(tools):
        if not isinstance(tool, dict):
            raise ValueError(f"Tool at index {idx} must be a dictionary")
        
        # Check required fields
        if 'type' not in tool:
            raise ValueError(f"Tool at index {idx} missing 'type' field")
        
        if tool['type'] != 'function':
            raise ValueError(f"Tool at index {idx} has invalid type '{tool['type']}' (only 'function' supported)")
        
        if 'function' not in tool:
            raise ValueError(f"Tool at index {idx} missing 'function' field")
        
        func = tool['function']
        if not isinstance(func, dict):
            raise ValueError(f"Tool at index {idx} function must be a dictionary")
        
        # Validate function definition
        if 'name' not in func:
            raise ValueError(f"Tool at index {idx} function missing 'name' field")
        
        name = func['name']
        if not isinstance(name, str) or not re.match(r'^[a-zA-Z0-9_-]{1,64}$', name):
            raise ValueError(
                f"Tool at index {idx} function name must be alphanumeric "
                f"with underscores/hyphens (max 64 chars): {name[:50]}"
            )
        
        # Check size
        import json
        tool_json = json.dumps(tool)
        if len(tool_json) > MAX_TOOL_DEFINITION_SIZE:
            raise ValueError(
                f"Tool at index {idx} definition too large "
                f"(max {MAX_TOOL_DEFINITION_SIZE} chars, got {len(tool_json)})"
            )
    
    return tools


def validate_temperature(temp: Optional[float]) -> Optional[float]:
    """
    Validate temperature parameter.
    
    Args:
        temp: Temperature value
        
    Returns:
        Validated temperature or None
        
    Raises:
        ValueError: If temperature is invalid
    """
    if temp is None:
        return None
    
    if not isinstance(temp, (int, float)):
        raise ValueError(f"Temperature must be a number, got {type(temp)}")
    
    if temp < 0.0 or temp > 2.0:
        raise ValueError(f"Temperature must be between 0.0 and 2.0, got {temp}")
    
    return float(temp)


def validate_max_tokens(max_tokens: Optional[int]) -> Optional[int]:
    """
    Validate max_tokens parameter.
    
    Args:
        max_tokens: Maximum tokens value
        
    Returns:
        Validated max_tokens or None
        
    Raises:
        ValueError: If max_tokens is invalid
    """
    if max_tokens is None:
        return None
    
    if not isinstance(max_tokens, int):
        try:
            max_tokens = int(max_tokens)
        except (ValueError, TypeError):
            raise ValueError(f"max_tokens must be an integer, got {type(max_tokens)}")
    
    if max_tokens < 1:
        raise ValueError(f"max_tokens must be at least 1, got {max_tokens}")
    
    if max_tokens > 128000:  # Most models have lower limits, but we'll be generous
        raise ValueError(f"max_tokens too large (max 128000, got {max_tokens})")
    
    return max_tokens


def validate_request_size(request_data: Any) -> bool:
    """
    Validate total request size.
    
    Args:
        request_data: Request data object
        
    Returns:
        True if valid
        
    Raises:
        ValueError: If request is too large
    """
    try:
        import json
        request_json = json.dumps(request_data.model_dump() if hasattr(request_data, 'model_dump') else request_data)
        
        if len(request_json) > MAX_REQUEST_SIZE:
            raise ValueError(
                f"Request too large (max {MAX_REQUEST_SIZE} chars, got {len(request_json)})"
            )
        
        return True
        
    except Exception as e:
        logger.error(f"Error validating request size: {e}")
        raise ValueError(f"Failed to validate request size: {str(e)}")


def validate_stop_sequences(stop: Optional[Any]) -> Optional[Any]:
    """
    Validate stop sequences parameter.
    
    Args:
        stop: Stop sequences (string or list of strings)
        
    Returns:
        Validated stop sequences or None
        
    Raises:
        ValueError: If stop sequences are invalid
    """
    if stop is None:
        return None
    
    if isinstance(stop, str):
        if len(stop) > 500:
            raise ValueError(f"Stop sequence too long (max 500 chars, got {len(stop)})")
        return stop
    
    if isinstance(stop, list):
        if len(stop) > 4:
            raise ValueError(f"Too many stop sequences (max 4, got {len(stop)})")
        
        for idx, seq in enumerate(stop):
            if not isinstance(seq, str):
                raise ValueError(f"Stop sequence at index {idx} must be a string")
            if len(seq) > 500:
                raise ValueError(f"Stop sequence at index {idx} too long (max 500 chars)")
        
        return stop
    
    raise ValueError(f"Stop sequences must be string or list of strings, got {type(stop)}")


def validate_model_name(model: Optional[str]) -> Optional[str]:
    """
    Validate model name format.
    
    Args:
        model: Model name
        
    Returns:
        Validated model name or None
        
    Raises:
        ValueError: If model name is invalid
    """
    if model is None:
        return None
    
    if not isinstance(model, str):
        raise ValueError(f"Model name must be a string, got {type(model)}")
    
    if len(model) > 100:
        raise ValueError(f"Model name too long (max 100 chars, got {len(model)})")
    
    # Basic sanity check for model name
    if not re.match(r'^[a-zA-Z0-9_\-./: ]+$', model):
        raise ValueError(f"Model name contains invalid characters: {model[:50]}")
    
    return model


def validate_provider_name(provider: Optional[str]) -> Optional[str]:
    """
    Validate provider name format.
    
    Args:
        provider: Provider name
        
    Returns:
        Validated provider name or None
        
    Raises:
        ValueError: If provider name is invalid
    """
    if provider is None:
        return None
    
    if not isinstance(provider, str):
        raise ValueError(f"Provider name must be a string, got {type(provider)}")
    
    # Convert to lowercase for consistency
    provider = provider.lower()
    
    # List of known providers (can be extended)
    known_providers = [
        "openai", "anthropic", "cohere", "groq", "openrouter",
        "deepseek", "mistral", "google", "huggingface",
        "llama.cpp", "kobold", "ollama", "ooba", "tabbyapi",
        "vllm", "local-llm", "aphrodite",
        "custom-openai-api", "custom-openai-api-2"
    ]
    
    if provider not in known_providers:
        logger.warning(f"Unknown provider '{provider}', proceeding anyway")
    
    return provider


#
# End of chat_validators.py
#######################################################################################################################