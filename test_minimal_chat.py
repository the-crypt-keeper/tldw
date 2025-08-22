#!/usr/bin/env python3
"""
Minimal test to debug chat endpoint authentication issues.
Run with: python -m pytest test_minimal_chat.py -xvs
"""

import os
import sys
import pytest
from unittest.mock import patch

# Ensure we're using the right path
sys.path.insert(0, 'tldw_Server_API')


def test_chat_auth_without_fixtures():
    """Test chat endpoint directly without any fixtures."""
    print("\n" + "="*60)
    print("Testing without fixtures")
    print("="*60)
    
    # Import after path is set
    from fastapi.testclient import TestClient
    from app.main import app
    from app.core.AuthNZ.settings import get_settings
    
    settings = get_settings()
    client = TestClient(app)
    
    print(f"Settings AUTH_MODE: {settings.AUTH_MODE}")
    print(f"Settings SINGLE_USER_API_KEY: {settings.SINGLE_USER_API_KEY}")
    
    # Test without headers
    response = client.post('/api/v1/chat/completions', 
                           json={'messages': [{'role': 'user', 'content': 'test'}]})
    print(f"Without headers - Status: {response.status_code}, Response: {response.json()}")
    assert response.status_code == 401
    assert "X-API-KEY" in response.json()['detail']
    
    # Test with correct header
    response = client.post('/api/v1/chat/completions',
                           headers={'X-API-KEY': settings.SINGLE_USER_API_KEY},
                           json={'messages': [{'role': 'user', 'content': 'test'}]})
    print(f"With X-API-KEY - Status: {response.status_code}")
    # Should be 503 (service unavailable) not 401 (unauthorized)
    assert response.status_code == 503


def test_chat_auth_with_mock():
    """Test chat endpoint with mocked LLM to avoid 503."""
    print("\n" + "="*60)
    print("Testing with mock LLM")
    print("="*60)
    
    from fastapi.testclient import TestClient
    from app.main import app
    from app.core.AuthNZ.settings import get_settings
    
    settings = get_settings()
    
    # Mock the API keys to avoid 503
    with patch.dict("tldw_Server_API.app.api.v1.endpoints.chat.API_KEYS", {"openai": "sk-mock-key"}):
        # Mock the LLM call
        mock_response = {
            "id": "chatcmpl-test",
            "object": "chat.completion",
            "created": 1234567890,
            "model": "test-model",
            "choices": [{
                "index": 0,
                "message": {"role": "assistant", "content": "Test response"},
                "finish_reason": "stop"
            }],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15}
        }
        
        with patch("tldw_Server_API.app.api.v1.endpoints.chat.perform_chat_api_call", 
                   return_value=mock_response):
            client = TestClient(app)
            
            # Test with correct header - should work
            response = client.post('/api/v1/chat/completions',
                                   headers={'X-API-KEY': settings.SINGLE_USER_API_KEY},
                                   json={'messages': [{'role': 'user', 'content': 'test'}]})
            print(f"With mocked LLM - Status: {response.status_code}")
            assert response.status_code == 200
            assert response.json()['choices'][0]['message']['content'] == "Test response"


if __name__ == "__main__":
    # Run directly for debugging
    print("Running tests directly...")
    
    try:
        test_chat_auth_without_fixtures()
        print("✓ test_chat_auth_without_fixtures passed")
    except AssertionError as e:
        print(f"✗ test_chat_auth_without_fixtures failed: {e}")
    
    try:
        test_chat_auth_with_mock()
        print("✓ test_chat_auth_with_mock passed")
    except AssertionError as e:
        print(f"✗ test_chat_auth_with_mock failed: {e}")