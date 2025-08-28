"""
Input validation and sanitization for LLM API calls.

This module provides security functions to validate and sanitize user inputs
before they are sent to LLM providers, preventing injection attacks and data leaks.
"""

import re
import base64
from typing import Any, Dict, List, Optional, Union
from urllib.parse import urlparse
from loguru import logger

# Maximum lengths for various inputs
MAX_PROMPT_LENGTH = 100000  # ~25k tokens
MAX_MESSAGE_LENGTH = 50000
MAX_SYSTEM_MESSAGE_LENGTH = 10000
MAX_MODEL_NAME_LENGTH = 100
MAX_URL_LENGTH = 2048
MAX_MESSAGES_COUNT = 100

# Regex patterns for validation
MODEL_NAME_PATTERN = re.compile(r'^[a-zA-Z0-9\-_\.\/]+$')
DATA_URL_PATTERN = re.compile(r'^data:([a-zA-Z0-9]+\/[a-zA-Z0-9\-\+\.]+)(;base64)?,(.*)$')

# Allowed MIME types for multimodal inputs
ALLOWED_IMAGE_TYPES = {
    'image/jpeg', 'image/jpg', 'image/png', 'image/gif', 
    'image/webp', 'image/svg+xml'
}
ALLOWED_AUDIO_TYPES = {
    'audio/mpeg', 'audio/wav', 'audio/ogg', 'audio/mp4'
}
ALLOWED_VIDEO_TYPES = {
    'video/mp4', 'video/mpeg', 'video/webm'
}

# Patterns that might indicate injection attempts
INJECTION_PATTERNS = [
    # Common prompt injection patterns
    re.compile(r'ignore\s+(previous|all|above)\s+(instructions?|prompts?)', re.IGNORECASE),
    re.compile(r'disregard\s+(previous|all)\s+(instructions?|prompts?)', re.IGNORECASE),
    re.compile(r'forget\s+(everything|all|previous)', re.IGNORECASE),
    re.compile(r'(system|admin)\s*:\s*override', re.IGNORECASE),
    re.compile(r'<\s*script\s*>', re.IGNORECASE),  # HTML injection
    re.compile(r'javascript\s*:', re.IGNORECASE),   # JavaScript protocol
    re.compile(r'data\s*:\s*text\/html', re.IGNORECASE),  # Data URL HTML injection
]


class ValidationError(Exception):
    """Raised when input validation fails."""
    pass


def sanitize_string(text: str, max_length: int = MAX_PROMPT_LENGTH, 
                    allow_newlines: bool = True) -> str:
    """
    Sanitize a string input for safe use with LLM APIs.
    
    Args:
        text: The input string to sanitize
        max_length: Maximum allowed length
        allow_newlines: Whether to allow newline characters
        
    Returns:
        Sanitized string
        
    Raises:
        ValidationError: If the input is invalid or potentially malicious
    """
    if not isinstance(text, str):
        raise ValidationError(f"Expected string, got {type(text).__name__}")
    
    # Check length
    if len(text) > max_length:
        logger.warning(f"Input text exceeds maximum length ({len(text)} > {max_length})")
        raise ValidationError(f"Input exceeds maximum length of {max_length} characters")
    
    # Remove null bytes and other control characters
    text = text.replace('\x00', '')
    text = ''.join(char for char in text if ord(char) >= 32 or char in '\n\r\t')
    
    # Optionally remove newlines
    if not allow_newlines:
        text = ' '.join(text.split())
    
    # Check for potential injection patterns
    for pattern in INJECTION_PATTERNS:
        if pattern.search(text):
            logger.warning(f"Potential injection pattern detected: {pattern.pattern}")
            # Don't raise error, just log - these might be legitimate use cases
            # Consider implementing a whitelist mode for trusted users
    
    return text.strip()


def validate_model_name(model: str) -> str:
    """
    Validate a model name string.
    
    Args:
        model: The model name to validate
        
    Returns:
        Validated model name
        
    Raises:
        ValidationError: If the model name is invalid
    """
    if not isinstance(model, str):
        raise ValidationError(f"Model name must be a string, got {type(model).__name__}")
    
    if len(model) > MAX_MODEL_NAME_LENGTH:
        raise ValidationError(f"Model name exceeds maximum length of {MAX_MODEL_NAME_LENGTH}")
    
    if not MODEL_NAME_PATTERN.match(model):
        raise ValidationError(f"Invalid model name format: {model}")
    
    return model


def validate_messages(messages: List[Dict[str, Any]], 
                     max_messages: int = MAX_MESSAGES_COUNT) -> List[Dict[str, Any]]:
    """
    Validate and sanitize a list of message objects for chat completion.
    
    Args:
        messages: List of message dictionaries
        max_messages: Maximum number of messages allowed
        
    Returns:
        Validated and sanitized messages
        
    Raises:
        ValidationError: If messages are invalid
    """
    if not isinstance(messages, list):
        raise ValidationError(f"Messages must be a list, got {type(messages).__name__}")
    
    if len(messages) > max_messages:
        raise ValidationError(f"Too many messages ({len(messages)} > {max_messages})")
    
    if len(messages) == 0:
        raise ValidationError("Messages list cannot be empty")
    
    validated_messages = []
    
    for i, msg in enumerate(messages):
        if not isinstance(msg, dict):
            raise ValidationError(f"Message {i} must be a dictionary")
        
        # Validate required fields
        if 'role' not in msg:
            raise ValidationError(f"Message {i} missing required 'role' field")
        
        if 'content' not in msg:
            raise ValidationError(f"Message {i} missing required 'content' field")
        
        # Validate role
        valid_roles = {'system', 'user', 'assistant', 'function', 'tool'}
        if msg['role'] not in valid_roles:
            raise ValidationError(f"Message {i} has invalid role: {msg['role']}")
        
        # Sanitize content based on type
        validated_msg = {'role': msg['role']}
        
        if isinstance(msg['content'], str):
            # Simple text content
            max_len = MAX_SYSTEM_MESSAGE_LENGTH if msg['role'] == 'system' else MAX_MESSAGE_LENGTH
            validated_msg['content'] = sanitize_string(msg['content'], max_length=max_len)
        
        elif isinstance(msg['content'], list):
            # Multimodal content (e.g., text + images)
            validated_content = []
            for item in msg['content']:
                if not isinstance(item, dict):
                    raise ValidationError(f"Message {i} content item must be a dictionary")
                
                if 'type' not in item:
                    raise ValidationError(f"Message {i} content item missing 'type' field")
                
                if item['type'] == 'text':
                    if 'text' not in item:
                        raise ValidationError(f"Message {i} text item missing 'text' field")
                    validated_content.append({
                        'type': 'text',
                        'text': sanitize_string(item['text'])
                    })
                
                elif item['type'] == 'image_url':
                    if 'image_url' not in item:
                        raise ValidationError(f"Message {i} image item missing 'image_url' field")
                    
                    image_url = validate_image_url(item['image_url'])
                    validated_content.append({
                        'type': 'image_url',
                        'image_url': image_url
                    })
                
                else:
                    logger.warning(f"Unknown content type in message {i}: {item['type']}")
            
            validated_msg['content'] = validated_content
        
        else:
            raise ValidationError(f"Message {i} content must be string or list")
        
        # Copy over other fields (like name, function_call, etc.)
        for key in msg:
            if key not in ('role', 'content'):
                validated_msg[key] = msg[key]
        
        validated_messages.append(validated_msg)
    
    return validated_messages


def validate_image_url(image_url: Union[str, Dict[str, Any]]) -> Dict[str, Any]:
    """
    Validate an image URL or data URL.
    
    Args:
        image_url: Either a URL string or a dictionary with 'url' field
        
    Returns:
        Validated image URL dictionary
        
    Raises:
        ValidationError: If the image URL is invalid
    """
    if isinstance(image_url, str):
        url = image_url
        detail = 'auto'
    elif isinstance(image_url, dict):
        if 'url' not in image_url:
            raise ValidationError("Image URL dictionary missing 'url' field")
        url = image_url['url']
        detail = image_url.get('detail', 'auto')
    else:
        raise ValidationError(f"Image URL must be string or dict, got {type(image_url).__name__}")
    
    # Validate detail level
    if detail not in ('auto', 'low', 'high'):
        raise ValidationError(f"Invalid image detail level: {detail}")
    
    # Check if it's a data URL
    if url.startswith('data:'):
        mime_type, data = validate_data_url(url)
        if mime_type not in ALLOWED_IMAGE_TYPES:
            raise ValidationError(f"Unsupported image type: {mime_type}")
    else:
        # Regular URL validation
        url = validate_url(url)
    
    return {'url': url, 'detail': detail}


def validate_data_url(data_url: str) -> tuple[str, str]:
    """
    Validate and parse a data URL.
    
    Args:
        data_url: The data URL to validate
        
    Returns:
        Tuple of (mime_type, base64_data)
        
    Raises:
        ValidationError: If the data URL is invalid
    """
    match = DATA_URL_PATTERN.match(data_url)
    if not match:
        raise ValidationError("Invalid data URL format")
    
    mime_type = match.group(1)
    is_base64 = match.group(2) is not None
    data = match.group(3)
    
    if is_base64:
        # Validate base64 encoding
        try:
            base64.b64decode(data, validate=True)
        except Exception as e:
            raise ValidationError(f"Invalid base64 data: {e}")
    
    # Check data size (prevent huge data URLs)
    if len(data) > 10 * 1024 * 1024:  # 10MB limit
        raise ValidationError("Data URL too large (max 10MB)")
    
    return mime_type, data


def validate_url(url: str) -> str:
    """
    Validate a regular URL.
    
    Args:
        url: The URL to validate
        
    Returns:
        Validated URL
        
    Raises:
        ValidationError: If the URL is invalid
    """
    if not isinstance(url, str):
        raise ValidationError(f"URL must be a string, got {type(url).__name__}")
    
    if len(url) > MAX_URL_LENGTH:
        raise ValidationError(f"URL exceeds maximum length of {MAX_URL_LENGTH}")
    
    # Parse and validate URL
    try:
        parsed = urlparse(url)
    except Exception as e:
        raise ValidationError(f"Invalid URL format: {e}")
    
    # Check scheme
    if parsed.scheme not in ('http', 'https'):
        raise ValidationError(f"Invalid URL scheme: {parsed.scheme}")
    
    # Check for localhost/private IPs (prevent SSRF)
    hostname = parsed.hostname
    if hostname:
        if hostname in ('localhost', '127.0.0.1', '0.0.0.0'):
            logger.warning(f"Attempt to access localhost: {url}")
            raise ValidationError("Access to localhost URLs not allowed")
        
        # Check for private IP ranges
        if hostname.startswith('192.168.') or hostname.startswith('10.') or hostname.startswith('172.'):
            logger.warning(f"Attempt to access private IP: {url}")
            raise ValidationError("Access to private network URLs not allowed")
    
    return url


def validate_temperature(temp: Optional[float]) -> Optional[float]:
    """
    Validate temperature parameter.
    
    Args:
        temp: Temperature value
        
    Returns:
        Validated temperature
        
    Raises:
        ValidationError: If temperature is invalid
    """
    if temp is None:
        return None
    
    try:
        temp = float(temp)
    except (ValueError, TypeError):
        raise ValidationError(f"Temperature must be a number, got {type(temp).__name__}")
    
    if not 0.0 <= temp <= 2.0:
        raise ValidationError(f"Temperature must be between 0.0 and 2.0, got {temp}")
    
    return temp


def validate_max_tokens(max_tokens: Optional[int]) -> Optional[int]:
    """
    Validate max_tokens parameter.
    
    Args:
        max_tokens: Maximum tokens value
        
    Returns:
        Validated max_tokens
        
    Raises:
        ValidationError: If max_tokens is invalid
    """
    if max_tokens is None:
        return None
    
    try:
        max_tokens = int(max_tokens)
    except (ValueError, TypeError):
        raise ValidationError(f"max_tokens must be an integer, got {type(max_tokens).__name__}")
    
    if max_tokens <= 0:
        raise ValidationError(f"max_tokens must be positive, got {max_tokens}")
    
    if max_tokens > 128000:  # Max for GPT-4
        raise ValidationError(f"max_tokens too large (max 128000), got {max_tokens}")
    
    return max_tokens


def validate_api_request(
    messages: List[Dict[str, Any]],
    model: Optional[str] = None,
    temperature: Optional[float] = None,
    max_tokens: Optional[int] = None,
    system_message: Optional[str] = None,
    **kwargs
) -> Dict[str, Any]:
    """
    Validate and sanitize a complete API request.
    
    Args:
        messages: List of message objects
        model: Model name
        temperature: Temperature parameter
        max_tokens: Maximum tokens
        system_message: System message
        **kwargs: Additional parameters
        
    Returns:
        Dictionary of validated parameters
        
    Raises:
        ValidationError: If any parameter is invalid
    """
    validated = {}
    
    # Validate messages
    validated['messages'] = validate_messages(messages)
    
    # Validate optional parameters
    if model is not None:
        validated['model'] = validate_model_name(model)
    
    if temperature is not None:
        validated['temperature'] = validate_temperature(temperature)
    
    if max_tokens is not None:
        validated['max_tokens'] = validate_max_tokens(max_tokens)
    
    if system_message is not None:
        validated['system_message'] = sanitize_string(
            system_message, 
            max_length=MAX_SYSTEM_MESSAGE_LENGTH
        )
    
    # Pass through other parameters (they'll be validated by specific providers)
    for key, value in kwargs.items():
        if key not in validated and value is not None:
            validated[key] = value
    
    return validated