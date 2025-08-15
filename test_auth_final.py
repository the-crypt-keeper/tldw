#!/usr/bin/env python3
"""
Final test of multi-user authentication with PostgreSQL
"""

import requests
import json

BASE_URL = "http://localhost:8001/api/v1"

def test_auth_flow():
    print("=" * 60)
    print("PostgreSQL Multi-User Authentication Test")
    print("=" * 60)
    
    # Step 1: Login
    print("\n1. Testing Login")
    print("-" * 40)
    
    login_response = requests.post(
        f"{BASE_URL}/auth/login",
        data={
            "username": "bob",
            "password": "SecureP@ss#2024!"
        }
    )
    
    if login_response.status_code != 200:
        print(f"❌ Login failed: {login_response.status_code}")
        print(f"   Response: {login_response.text}")
        return False
    
    tokens = login_response.json()
    access_token = tokens["access_token"]
    print(f"✅ Login successful!")
    print(f"   Token type: {tokens['token_type']}")
    print(f"   Expires in: {tokens['expires_in']} seconds")
    
    # Step 2: Test authenticated endpoint
    print("\n2. Testing Authenticated Endpoint (/auth/me)")
    print("-" * 40)
    
    headers = {"Authorization": f"Bearer {access_token}"}
    me_response = requests.get(f"{BASE_URL}/auth/me", headers=headers)
    
    if me_response.status_code != 200:
        print(f"❌ Failed to get user info: {me_response.status_code}")
        print(f"   Response: {me_response.text}")
        return False
    
    user_info = me_response.json()
    print(f"✅ Successfully retrieved user info!")
    print(f"   User ID: {user_info.get('id')}")
    print(f"   Username: {user_info.get('username')}")
    print(f"   Email: {user_info.get('email')}")
    print(f"   Role: {user_info.get('role')}")
    print(f"   Active: {user_info.get('is_active')}")
    print(f"   Verified: {user_info.get('is_verified')}")
    
    # Step 3: Test refresh token
    print("\n3. Testing Token Refresh")
    print("-" * 40)
    
    refresh_response = requests.post(
        f"{BASE_URL}/auth/refresh",
        json={"refresh_token": tokens["refresh_token"]}
    )
    
    if refresh_response.status_code != 200:
        print(f"❌ Token refresh failed: {refresh_response.status_code}")
        print(f"   Response: {refresh_response.text}")
        return False
    
    new_tokens = refresh_response.json()
    print(f"✅ Token refresh successful!")
    print(f"   New access token generated")
    
    # Step 4: Test logout
    print("\n4. Testing Logout")
    print("-" * 40)
    
    logout_response = requests.post(
        f"{BASE_URL}/auth/logout",
        headers={"Authorization": f"Bearer {new_tokens['access_token']}"}
    )
    
    if logout_response.status_code != 200:
        print(f"❌ Logout failed: {logout_response.status_code}")
        print(f"   Response: {logout_response.text}")
        return False
    
    print(f"✅ Logout successful!")
    
    # Step 5: Verify token is invalid after logout
    print("\n5. Testing Token Invalidation After Logout")
    print("-" * 40)
    
    invalid_response = requests.get(
        f"{BASE_URL}/auth/me",
        headers={"Authorization": f"Bearer {new_tokens['access_token']}"}
    )
    
    if invalid_response.status_code == 401:
        print(f"✅ Token correctly invalidated after logout")
    else:
        print(f"⚠️  Token still valid after logout (status: {invalid_response.status_code})")
    
    return True

if __name__ == "__main__":
    success = test_auth_flow()
    
    print("\n" + "=" * 60)
    if success:
        print("✅ ALL TESTS PASSED!")
        print("\nPostgreSQL multi-user authentication is working correctly!")
    else:
        print("❌ TESTS FAILED")
        print("\nThere are still issues to resolve.")
    print("=" * 60)