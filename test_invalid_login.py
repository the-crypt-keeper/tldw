#!/usr/bin/env python3
"""Test invalid login credentials"""

import requests

BASE_URL = "http://localhost:8001/api/v1"

# Test invalid credentials
print("Testing invalid credentials...")
response = requests.post(
    f"{BASE_URL}/auth/login",
    data={
        "username": "nonexistent_user_xyz",
        "password": "WrongPassword123!"
    }
)

print(f"Status: {response.status_code}")
print(f"Response: {response.text}")

if response.status_code == 401:
    print("✅ Correctly returned 401 Unauthorized")
elif response.status_code == 500:
    print("❌ Returned 500 Internal Server Error")
    print("This suggests the error is happening inside a transaction")
else:
    print(f"Unexpected status: {response.status_code}")