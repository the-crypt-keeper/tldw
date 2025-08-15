#!/usr/bin/env python3
"""
Test multi-user authentication with PostgreSQL
"""

import requests
import json
import sys
from datetime import datetime

BASE_URL = "http://localhost:8001/api/v1"

def test_registration():
    """Test user registration in multi-user mode"""
    print("\n1. Testing Registration")
    print("-" * 40)
    
    users = [
        {
            "username": "administrator",
            "email": "admin@example.com", 
            "password": "SecureP@ss#2024!",
            "role": "admin"
        },
        {
            "username": "johndoe",
            "email": "john@example.com",
            "password": "MyP@ssw0rd#2024!",
            "role": "user"
        }
    ]
    
    registered_users = []
    
    for user_data in users:
        response = requests.post(
            f"{BASE_URL}/auth/register",
            json={
                "username": user_data["username"],
                "email": user_data["email"],
                "password": user_data["password"]
            }
        )
        
        if response.status_code in [200, 201]:
            data = response.json()
            print(f"✅ Registered {user_data['username']}: User ID {data.get('user_id')}")
            registered_users.append(user_data)
        elif response.status_code == 409:
            print(f"ℹ️  User {user_data['username']} already exists")
            registered_users.append(user_data)
        else:
            print(f"❌ Failed to register {user_data['username']}: {response.status_code}")
            print(f"   Response: {response.text}")
    
    return registered_users

def test_login(username, password):
    """Test login"""
    print(f"\n2. Testing Login for {username}")
    print("-" * 40)
    
    response = requests.post(
        f"{BASE_URL}/auth/login",
        data={
            "username": username,
            "password": password
        }
    )
    
    if response.status_code == 200:
        data = response.json()
        print(f"✅ Login successful!")
        print(f"   Token type: {data.get('token_type')}")
        print(f"   Expires in: {data.get('expires_in')} seconds")
        return data.get("access_token"), data.get("refresh_token")
    else:
        print(f"❌ Login failed: {response.status_code}")
        print(f"   Response: {response.text}")
        return None, None

def test_authenticated_endpoints(token):
    """Test endpoints that require authentication"""
    print("\n3. Testing Authenticated Endpoints")
    print("-" * 40)
    
    headers = {"Authorization": f"Bearer {token}"}
    
    # Test /auth/me
    response = requests.get(f"{BASE_URL}/auth/me", headers=headers)
    if response.status_code == 200:
        user_data = response.json()
        print(f"✅ GET /auth/me:")
        print(f"   User: {user_data.get('username')} (ID: {user_data.get('id')})")
        print(f"   Email: {user_data.get('email')}")
        print(f"   Role: {user_data.get('role')}")
        print(f"   Storage: {user_data.get('storage_used_mb')}/{user_data.get('storage_quota_mb')} MB")
        return user_data
    else:
        print(f"❌ Failed to get user info: {response.status_code}")
        return None

def test_admin_endpoints(token):
    """Test admin-only endpoints"""
    print("\n4. Testing Admin Endpoints")
    print("-" * 40)
    
    headers = {"Authorization": f"Bearer {token}"}
    
    # Test listing users
    response = requests.get(f"{BASE_URL}/admin/users", headers=headers)
    if response.status_code == 200:
        data = response.json()
        print(f"✅ GET /admin/users:")
        print(f"   Total users: {data.get('total')}")
        for user in data.get('users', []):
            print(f"   - {user.get('username')} ({user.get('role')})")
    elif response.status_code == 403:
        print("ℹ️  Access denied (expected for non-admin)")
    else:
        print(f"❌ Unexpected response: {response.status_code}")
    
    # Test system stats
    response = requests.get(f"{BASE_URL}/admin/stats", headers=headers)
    if response.status_code == 200:
        data = response.json()
        print(f"✅ GET /admin/stats:")
        users_stats = data.get('users', {})
        print(f"   Total users: {users_stats.get('total')}")
        print(f"   Active users: {users_stats.get('active')}")
    elif response.status_code == 403:
        print("ℹ️  Stats access denied (expected for non-admin)")

def test_health_endpoints():
    """Test health monitoring endpoints"""
    print("\n5. Testing Health Endpoints")
    print("-" * 40)
    
    endpoints = ["/health", "/health/live", "/health/ready", "/health/metrics"]
    
    for endpoint in endpoints:
        response = requests.get(f"{BASE_URL}{endpoint}")
        if response.status_code in [200, 206]:
            print(f"✅ GET {endpoint}: {response.status_code}")
            if endpoint == "/health":
                data = response.json()
                print(f"   Status: {data.get('status')}")
                checks = data.get('checks', {})
                for check, info in checks.items():
                    if isinstance(info, dict):
                        print(f"   - {check}: {info.get('status')}")
                    else:
                        print(f"   - {check}: {info}")
        else:
            print(f"❌ GET {endpoint}: {response.status_code}")

def test_session_management(token):
    """Test session management"""
    print("\n6. Testing Session Management")
    print("-" * 40)
    
    headers = {"Authorization": f"Bearer {token}"}
    
    # Get sessions
    response = requests.get(f"{BASE_URL}/users/sessions", headers=headers)
    if response.status_code == 200:
        data = response.json()
        sessions = data.get('sessions', [])
        print(f"✅ Active sessions: {len(sessions)}")
        for session in sessions:
            print(f"   - IP: {session.get('ip_address')}, Created: {session.get('created_at')}")
    else:
        print(f"❌ Failed to get sessions: {response.status_code}")

def test_rate_limiting():
    """Test rate limiting (if enabled)"""
    print("\n7. Testing Rate Limiting")
    print("-" * 40)
    
    # Make multiple rapid requests
    for i in range(10):
        response = requests.post(
            f"{BASE_URL}/auth/login",
            data={"username": "nonexistent", "password": "wrong"}
        )
        if response.status_code == 429:
            print(f"✅ Rate limit triggered after {i+1} requests")
            return
    
    print("ℹ️  Rate limiting not triggered (may be disabled)")

def main():
    print("=" * 60)
    print("🚀 Multi-User Authentication Test Suite")
    print("=" * 60)
    print(f"Server: {BASE_URL}")
    print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Test health first
    test_health_endpoints()
    
    # Register users
    users = test_registration()
    
    if not users:
        print("\n❌ No users registered, cannot continue tests")
        return 1
    
    # Test with admin user
    print("\n" + "=" * 60)
    print("Testing Admin User")
    print("=" * 60)
    
    admin_token, _ = test_login("administrator", "SecureP@ss#2024!")
    if admin_token:
        user_info = test_authenticated_endpoints(admin_token)
        test_admin_endpoints(admin_token)
        test_session_management(admin_token)
    
    # Test with regular user
    print("\n" + "=" * 60)
    print("Testing Regular User")
    print("=" * 60)
    
    user_token, _ = test_login("johndoe", "MyP@ssw0rd#2024!")
    if user_token:
        test_authenticated_endpoints(user_token)
        test_admin_endpoints(user_token)  # Should be denied
    
    # Test rate limiting
    test_rate_limiting()
    
    print("\n" + "=" * 60)
    print("✅ Multi-User Authentication Testing Complete!")
    print("=" * 60)
    print("\nNext steps:")
    print("1. Check the API documentation at http://localhost:8000/docs")
    print("2. Query the PostgreSQL database to see the data")
    print("3. Check audit logs: docker exec -it tldw-postgres psql -U tldw_user -d tldw_multiuser -c 'SELECT * FROM audit_log;'")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())