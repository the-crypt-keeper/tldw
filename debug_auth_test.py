#!/usr/bin/env python3
"""
Debug script to test authentication issues with chat endpoint.
Run this to see what's happening with authentication.
"""

import os
import sys

# Add the tldw_Server_API to path
sys.path.insert(0, 'tldw_Server_API')

print("=" * 60)
print("Authentication Debug Test")
print("=" * 60)

# 1. Check environment variables
print("\n1. Environment Variables:")
print(f"   API_BEARER: {os.getenv('API_BEARER', 'NOT SET')}")
print(f"   SINGLE_USER_API_KEY: {os.getenv('SINGLE_USER_API_KEY', 'NOT SET')}")

# 2. Check auth_utils
from app.core.Auth.auth_utils import is_authentication_required, get_expected_api_token
print("\n2. Auth Utils:")
print(f"   is_authentication_required(): {is_authentication_required()}")
print(f"   get_expected_api_token(): {get_expected_api_token()}")

# 3. Check settings
from app.core.AuthNZ.settings import get_settings
settings = get_settings()
print("\n3. Settings:")
print(f"   AUTH_MODE: {settings.AUTH_MODE}")
print(f"   SINGLE_USER_API_KEY: {settings.SINGLE_USER_API_KEY}")

# 4. Test the endpoint
print("\n4. Testing Chat Endpoint:")
from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)

# Test without headers
print("\n   a) Without headers:")
response = client.post('/api/v1/chat/completions', 
                       json={'messages': [{'role': 'user', 'content': 'test'}]})
print(f"      Status: {response.status_code}")
print(f"      Response: {response.json()}")

# Test with X-API-KEY
print(f"\n   b) With X-API-KEY={settings.SINGLE_USER_API_KEY}:")
response = client.post('/api/v1/chat/completions',
                       headers={'X-API-KEY': settings.SINGLE_USER_API_KEY},
                       json={'messages': [{'role': 'user', 'content': 'test'}]})
print(f"      Status: {response.status_code}")
if response.status_code != 200:
    print(f"      Response: {response.json()}")
else:
    print("      Success!")

print("\n" + "=" * 60)
print("To fix the tests, ensure:")
print("1. API_BEARER environment variable is NOT set")
print("2. Tests use X-API-KEY header with value matching SINGLE_USER_API_KEY")
print("3. SINGLE_USER_API_KEY in settings matches what tests expect")
print("=" * 60)