# rate_limiting.py
# Centralized rate limiting configuration that respects TEST_MODE

import os
from slowapi import Limiter
from slowapi.util import get_remote_address as _original_get_remote_address

def get_test_aware_remote_address(request):
    """
    Custom key function for rate limiting that bypasses limits in TEST_MODE.
    Returns None in TEST_MODE to effectively disable rate limiting.
    """
    # ONLY check for TEST_MODE in environment - NEVER trust client headers
    if os.getenv("TEST_MODE") == "true":
        return None  # Bypass rate limiting in test mode
    
    return _original_get_remote_address(request)

def create_limiter():
    """
    Create a Limiter instance that respects TEST_MODE.
    """
    return Limiter(key_func=get_test_aware_remote_address)

# Global limiter instance that can be imported by endpoints
limiter = create_limiter()