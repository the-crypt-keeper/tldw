"""
Security module for LLM API calls.

Provides input validation, key management, and audit logging.
"""

from .input_validator import (
    ValidationError,
    sanitize_string,
    validate_model_name,
    validate_messages,
    validate_temperature,
    validate_max_tokens,
    validate_api_request,
)

from .key_manager import (
    KeyManager,
    get_key_manager,
    APIKeyInfo,
)

__all__ = [
    # Input validation
    'ValidationError',
    'sanitize_string',
    'validate_model_name',
    'validate_messages',
    'validate_temperature',
    'validate_max_tokens',
    'validate_api_request',
    # Key management
    'KeyManager',
    'get_key_manager',
    'APIKeyInfo',
]