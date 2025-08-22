#!/usr/bin/env python3
"""
Debug script to trace authentication during pytest run.
"""

import os
import sys
import pytest

# Add the tldw_Server_API to path
sys.path.insert(0, 'tldw_Server_API')

print("=" * 60)
print("Pytest Authentication Debug")
print("=" * 60)

# Check initial environment
print("\nInitial environment:")
print(f"  API_BEARER: {os.getenv('API_BEARER', 'NOT SET')}")
print(f"  SINGLE_USER_API_KEY: {os.getenv('SINGLE_USER_API_KEY', 'NOT SET')}")

# Patch the chat endpoint to print debug info
import unittest.mock as mock

original_chat_completions = None

def debug_chat_completions(*args, **kwargs):
    """Wrapper to debug chat_completions calls."""
    print("\n[DEBUG] chat_completions called:")
    print(f"  API_BEARER env: {os.getenv('API_BEARER', 'NOT SET')}")
    
    from tldw_Server_API.app.core.Auth.auth_utils import is_authentication_required
    print(f"  is_authentication_required(): {is_authentication_required()}")
    
    # Check the Token parameter
    token = kwargs.get('Token')
    print(f"  Token parameter: {token}")
    
    # Call original
    return original_chat_completions(*args, **kwargs)

# Run a single test with debugging
print("\nRunning test with debugging...")

# Monkey-patch the endpoint
with mock.patch('tldw_Server_API.app.api.v1.endpoints.chat.chat_completions', side_effect=debug_chat_completions) as mock_chat:
    # Store original
    from tldw_Server_API.app.api.v1.endpoints.chat import chat_completions
    original_chat_completions = chat_completions
    mock_chat.side_effect = debug_chat_completions
    
    # Run specific test
    exit_code = pytest.main([
        'tldw_Server_API/tests/Chat/test_chat_unit.py::TestChatUnit::test_chat_completion_basic',
        '-xvs',
        '--tb=short'
    ])

print("\n" + "=" * 60)
print(f"Test exit code: {exit_code}")
print("=" * 60)