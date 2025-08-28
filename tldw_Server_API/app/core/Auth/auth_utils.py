# auth_utils.py
# Description: Authentication utilities for secure token validation
#
# Imports
import hmac
import os
from typing import Optional
from loguru import logger

from tldw_Server_API.app.core.config import AUTH_BEARER_PREFIX

#######################################################################################################################
#
# Functions:

def constant_time_compare(val1: str, val2: str) -> bool:
    """
    Perform constant-time string comparison to prevent timing attacks.
    
    Args:
        val1: First string to compare
        val2: Second string to compare
        
    Returns:
        True if strings are equal, False otherwise
    """
    return hmac.compare_digest(val1.encode('utf-8'), val2.encode('utf-8'))


def extract_bearer_token(auth_header: Optional[str]) -> Optional[str]:
    """
    Extract bearer token from authorization header.
    
    Args:
        auth_header: Authorization header value (e.g., "Bearer token123")
        
    Returns:
        The extracted token without prefix, or None if invalid format
    """
    if not auth_header:
        return None
    
    if not auth_header.startswith(AUTH_BEARER_PREFIX):
        logger.warning("Invalid authorization header format - missing Bearer prefix")
        return None
    
    token = auth_header[len(AUTH_BEARER_PREFIX):].strip()
    if not token:
        logger.warning("Empty token after Bearer prefix")
        return None
        
    return token


def validate_api_token(provided_token: Optional[str], expected_token: Optional[str]) -> bool:
    """
    Validate an API token using constant-time comparison.
    
    Args:
        provided_token: Token provided by the client
        expected_token: Expected token value
        
    Returns:
        True if tokens match, False otherwise
    """
    if not provided_token or not expected_token:
        return False
    
    # Ensure both tokens are strings
    provided_str = str(provided_token)
    expected_str = str(expected_token)
    
    # Use constant-time comparison to prevent timing attacks
    return constant_time_compare(provided_str, expected_str)


def get_expected_api_token() -> Optional[str]:
    """
    Get the expected API token from environment variables.
    
    Returns:
        The expected API token, or None if not configured
    """
    token = os.getenv("API_BEARER")
    if not token:
        logger.warning("API_BEARER environment variable is not set")
    return token


def is_authentication_required() -> bool:
    """
    Check if authentication is required based on configuration.
    
    Returns:
        True if authentication is required, False otherwise
    """
    # Check if we're in multi-user mode
    auth_mode = os.getenv("AUTH_MODE", "single_user")
    if auth_mode == "multi_user":
        return True
    
    # In single-user mode, authentication is required if API_BEARER is set
    return bool(os.getenv("API_BEARER"))


#
# End of auth_utils.py
#######################################################################################################################
