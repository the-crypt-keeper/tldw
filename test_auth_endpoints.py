#!/usr/bin/env python3
"""
Test script to verify all authentication endpoints are working
"""

import requests
import json
import sys
from datetime import datetime
from typing import Dict, Any, Optional

# Configuration
BASE_URL = "http://127.0.0.1:8000/api/v1"
TEST_USER = {
    "username": "test_user_" + datetime.now().strftime("%Y%m%d_%H%M%S"),
    "email": f"test_{datetime.now().strftime('%Y%m%d_%H%M%S')}@example.com",
    "password": "TestPassword123!"
}

# Colors for terminal output
class Colors:
    GREEN = '\033[92m'
    RED = '\033[91m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    RESET = '\033[0m'
    BOLD = '\033[1m'

def print_test(name: str, passed: bool, details: str = ""):
    """Print test result with color"""
    status = f"{Colors.GREEN}✓ PASSED{Colors.RESET}" if passed else f"{Colors.RED}✗ FAILED{Colors.RESET}"
    print(f"{Colors.BOLD}[TEST]{Colors.RESET} {name}: {status}")
    if details:
        print(f"  └─ {details}")

def print_section(title: str):
    """Print section header"""
    print(f"\n{Colors.BLUE}{'='*60}{Colors.RESET}")
    print(f"{Colors.BLUE}{Colors.BOLD}{title}{Colors.RESET}")
    print(f"{Colors.BLUE}{'='*60}{Colors.RESET}")

def test_health_endpoints():
    """Test health monitoring endpoints"""
    print_section("Testing Health Endpoints")
    
    endpoints = [
        ("/health", "Main health check"),
        ("/health/live", "Liveness probe"),
        ("/health/ready", "Readiness probe"),
        ("/health/metrics", "Metrics endpoint")
    ]
    
    all_passed = True
    for endpoint, description in endpoints:
        try:
            response = requests.get(f"{BASE_URL}{endpoint}", timeout=5)
            passed = response.status_code in [200, 206]
            print_test(
                f"{description} ({endpoint})",
                passed,
                f"Status: {response.status_code}"
            )
            if not passed:
                all_passed = False
                print(f"  Response: {response.text[:200]}")
        except Exception as e:
            print_test(f"{description} ({endpoint})", False, str(e))
            all_passed = False
    
    return all_passed

def test_registration():
    """Test user registration"""
    print_section("Testing Registration")
    
    try:
        response = requests.post(
            f"{BASE_URL}/auth/register",
            json=TEST_USER,
            timeout=5
        )
        
        if response.status_code == 201:
            data = response.json()
            print_test(
                "User registration",
                True,
                f"User ID: {data.get('user_id')}, Username: {data.get('username')}"
            )
            return True
        else:
            print_test(
                "User registration",
                False,
                f"Status: {response.status_code}, Error: {response.text[:200]}"
            )
            return False
    except Exception as e:
        print_test("User registration", False, str(e))
        return False

def test_login() -> Optional[Dict[str, str]]:
    """Test user login"""
    print_section("Testing Login")
    
    try:
        # Test with form data (OAuth2 compatible)
        response = requests.post(
            f"{BASE_URL}/auth/login",
            data={
                "username": TEST_USER["username"],
                "password": TEST_USER["password"]
            },
            timeout=5
        )
        
        if response.status_code == 200:
            data = response.json()
            print_test(
                "User login",
                True,
                f"Token type: {data.get('token_type')}, Expires in: {data.get('expires_in')}s"
            )
            return {
                "access_token": data.get("access_token"),
                "refresh_token": data.get("refresh_token")
            }
        else:
            print_test(
                "User login",
                False,
                f"Status: {response.status_code}, Error: {response.text[:200]}"
            )
            return None
    except Exception as e:
        print_test("User login", False, str(e))
        return None

def test_authenticated_endpoints(tokens: Dict[str, str]):
    """Test endpoints that require authentication"""
    print_section("Testing Authenticated Endpoints")
    
    headers = {"Authorization": f"Bearer {tokens['access_token']}"}
    
    endpoints = [
        ("GET", "/auth/me", "Get current user (auth)", None),
        ("GET", "/users/me", "Get user profile", None),
        ("GET", "/users/sessions", "Get user sessions", None),
        ("GET", "/users/storage", "Get storage usage", None),
    ]
    
    all_passed = True
    for method, endpoint, description, data in endpoints:
        try:
            if method == "GET":
                response = requests.get(
                    f"{BASE_URL}{endpoint}",
                    headers=headers,
                    timeout=5
                )
            else:
                response = requests.post(
                    f"{BASE_URL}{endpoint}",
                    headers=headers,
                    json=data,
                    timeout=5
                )
            
            passed = response.status_code == 200
            print_test(
                f"{description} ({method} {endpoint})",
                passed,
                f"Status: {response.status_code}"
            )
            
            if passed and endpoint == "/auth/me":
                user_data = response.json()
                print(f"  User: {user_data.get('username')} (ID: {user_data.get('id')})")
                print(f"  Role: {user_data.get('role')}, Active: {user_data.get('is_active')}")
            
            if not passed:
                all_passed = False
                print(f"  Response: {response.text[:200]}")
                
        except Exception as e:
            print_test(f"{description} ({method} {endpoint})", False, str(e))
            all_passed = False
    
    return all_passed

def test_token_refresh(tokens: Dict[str, str]):
    """Test token refresh"""
    print_section("Testing Token Refresh")
    
    try:
        response = requests.post(
            f"{BASE_URL}/auth/refresh",
            json={"refresh_token": tokens["refresh_token"]},
            timeout=5
        )
        
        if response.status_code == 200:
            data = response.json()
            print_test(
                "Token refresh",
                True,
                "New access token received"
            )
            return data.get("access_token")
        else:
            print_test(
                "Token refresh",
                False,
                f"Status: {response.status_code}, Error: {response.text[:200]}"
            )
            return None
    except Exception as e:
        print_test("Token refresh", False, str(e))
        return None

def test_password_change(tokens: Dict[str, str]):
    """Test password change"""
    print_section("Testing Password Change")
    
    headers = {"Authorization": f"Bearer {tokens['access_token']}"}
    new_password = "NewTestPassword456!"
    
    try:
        response = requests.post(
            f"{BASE_URL}/users/change-password",
            headers=headers,
            json={
                "current_password": TEST_USER["password"],
                "new_password": new_password
            },
            timeout=5
        )
        
        if response.status_code == 200:
            print_test("Password change", True, "Password updated successfully")
            
            # Test login with new password
            login_response = requests.post(
                f"{BASE_URL}/auth/login",
                data={
                    "username": TEST_USER["username"],
                    "password": new_password
                },
                timeout=5
            )
            
            print_test(
                "Login with new password",
                login_response.status_code == 200,
                f"Status: {login_response.status_code}"
            )
            
            return login_response.status_code == 200
        else:
            print_test(
                "Password change",
                False,
                f"Status: {response.status_code}, Error: {response.text[:200]}"
            )
            return False
    except Exception as e:
        print_test("Password change", False, str(e))
        return False

def test_logout(tokens: Dict[str, str]):
    """Test logout"""
    print_section("Testing Logout")
    
    headers = {"Authorization": f"Bearer {tokens['access_token']}"}
    
    try:
        response = requests.post(
            f"{BASE_URL}/auth/logout",
            headers=headers,
            timeout=5
        )
        
        if response.status_code == 200:
            print_test("Logout", True, "Successfully logged out")
            
            # Verify token is invalid after logout
            verify_response = requests.get(
                f"{BASE_URL}/auth/me",
                headers=headers,
                timeout=5
            )
            
            print_test(
                "Token invalid after logout",
                verify_response.status_code == 401,
                f"Status: {verify_response.status_code}"
            )
            
            return True
        else:
            print_test(
                "Logout",
                False,
                f"Status: {response.status_code}, Error: {response.text[:200]}"
            )
            return False
    except Exception as e:
        print_test("Logout", False, str(e))
        return False

def test_admin_endpoints(tokens: Dict[str, str]):
    """Test admin endpoints (should fail for regular user)"""
    print_section("Testing Admin Access Control")
    
    headers = {"Authorization": f"Bearer {tokens['access_token']}"}
    
    admin_endpoints = [
        ("GET", "/admin/users", "List all users"),
        ("GET", "/admin/stats", "Get system stats"),
        ("GET", "/admin/audit-log", "View audit log"),
    ]
    
    all_correct = True
    for method, endpoint, description in admin_endpoints:
        try:
            response = requests.get(
                f"{BASE_URL}{endpoint}",
                headers=headers,
                timeout=5
            )
            
            # Should get 403 Forbidden for non-admin
            expected_forbidden = response.status_code == 403
            print_test(
                f"{description} - Access denied for non-admin",
                expected_forbidden,
                f"Status: {response.status_code}"
            )
            
            if not expected_forbidden:
                all_correct = False
                
        except Exception as e:
            print_test(f"{description}", False, str(e))
            all_correct = False
    
    return all_correct

def main():
    """Run all tests"""
    print(f"{Colors.BOLD}{Colors.BLUE}")
    print("="*60)
    print("  TLDW Authentication System Test Suite")
    print("="*60)
    print(f"{Colors.RESET}")
    
    print(f"Server: {BASE_URL}")
    print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    results = {}
    
    # Test health endpoints
    results["health"] = test_health_endpoints()
    
    # Test registration
    results["registration"] = test_registration()
    
    if results["registration"]:
        # Test login
        tokens = test_login()
        results["login"] = tokens is not None
        
        if tokens:
            # Test authenticated endpoints
            results["authenticated"] = test_authenticated_endpoints(tokens)
            
            # Test token refresh
            new_token = test_token_refresh(tokens)
            results["refresh"] = new_token is not None
            
            if new_token:
                tokens["access_token"] = new_token
            
            # Test password change
            results["password_change"] = test_password_change(tokens)
            
            # Test admin access control
            results["access_control"] = test_admin_endpoints(tokens)
            
            # Test logout
            results["logout"] = test_logout(tokens)
    
    # Print summary
    print_section("Test Summary")
    
    total_tests = len(results)
    passed_tests = sum(1 for v in results.values() if v)
    
    print(f"\nTotal Tests: {total_tests}")
    print(f"Passed: {Colors.GREEN}{passed_tests}{Colors.RESET}")
    print(f"Failed: {Colors.RED}{total_tests - passed_tests}{Colors.RESET}")
    
    if passed_tests == total_tests:
        print(f"\n{Colors.GREEN}{Colors.BOLD}✓ ALL TESTS PASSED!{Colors.RESET}")
        print("The authentication system is working correctly.")
        return 0
    else:
        print(f"\n{Colors.RED}{Colors.BOLD}✗ SOME TESTS FAILED{Colors.RESET}")
        print("Please review the failed tests above.")
        return 1

if __name__ == "__main__":
    sys.exit(main())