# test_auth_comprehensive.py
# Description: Comprehensive test suite for authentication and user management system
#
# Imports
import pytest
import asyncio
import os
from datetime import datetime, timedelta
from typing import Dict, Any, Optional
import json
import secrets
#
# 3rd-party imports
from httpx import AsyncClient
from fastapi.testclient import TestClient
#
# Local imports
from tldw_Server_API.app.main import app
from tldw_Server_API.app.core.AuthNZ.settings import Settings, get_settings
from tldw_Server_API.app.core.AuthNZ.database import DatabasePool, get_db_pool
from tldw_Server_API.app.core.AuthNZ.password_service import get_password_service
from tldw_Server_API.app.core.AuthNZ.jwt_service import get_jwt_service
from tldw_Server_API.app.services.registration_service import get_registration_service
from tldw_Server_API.app.services.audit_service import get_audit_service

#######################################################################################################################
#
# Test Fixtures

# Test database configuration
TEST_DB_NAME = "tldw_test"
TEST_DB_HOST = os.getenv("TEST_DB_HOST", "localhost")
TEST_DB_PORT = int(os.getenv("TEST_DB_PORT", "5432"))
TEST_DB_USER = os.getenv("TEST_DB_USER", "tldw_user")
TEST_DB_PASSWORD = os.getenv("TEST_DB_PASSWORD", "TestPassword123!")

# Note: We now use isolated_test_environment from conftest.py for true DB isolation
# Each test gets its own unique database that is created and destroyed per test

@pytest.fixture
async def test_user_data(isolated_test_environment):
    """Create test user data in isolated database"""
    import asyncpg
    import uuid
    from tldw_Server_API.app.core.AuthNZ.password_service import PasswordService
    
    client, db_name = isolated_test_environment
    
    # Connect to the unique test database
    conn = await asyncpg.connect(
        host=TEST_DB_HOST,
        port=TEST_DB_PORT,
        user=TEST_DB_USER,
        password=TEST_DB_PASSWORD,
        database=db_name
    )
    
    try:
        password_service = PasswordService()
        user_uuid = str(uuid.uuid4())
        password = "Test@Pass#2024!"
        password_hash = password_service.hash_password(password)
        
        user = await conn.fetchrow("""
            INSERT INTO users (
                uuid, username, email, password_hash, role,
                is_active, is_verified, storage_quota_mb, storage_used_mb
            )
            VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9)
            RETURNING id, uuid, username, email, role, is_active, is_verified
        """, user_uuid, "testuser", "test@example.com", password_hash,
            "user", True, True, 5120, 0.0)
        
        return {
            "id": user["id"],
            "uuid": str(user["uuid"]),
            "username": user["username"],
            "email": user["email"],
            "role": user["role"],
            "password": password
        }
    finally:
        await conn.close()

@pytest.fixture
async def admin_user_data(isolated_test_environment):
    """Create admin user data in isolated database"""
    import asyncpg
    import uuid
    from tldw_Server_API.app.core.AuthNZ.password_service import PasswordService
    
    client, db_name = isolated_test_environment
    
    # Connect to the unique test database
    conn = await asyncpg.connect(
        host=TEST_DB_HOST,
        port=TEST_DB_PORT,
        user=TEST_DB_USER,
        password=TEST_DB_PASSWORD,
        database=db_name
    )
    
    try:
        password_service = PasswordService()
        user_uuid = str(uuid.uuid4())
        password = "Admin@Pass#2024!"
        password_hash = password_service.hash_password(password)
        
        user = await conn.fetchrow("""
            INSERT INTO users (
                uuid, username, email, password_hash, role,
                is_active, is_verified, storage_quota_mb, storage_used_mb
            )
            VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9)
            RETURNING id, uuid, username, email, role, is_active, is_verified
        """, user_uuid, "admin", "admin@example.com", password_hash,
            "admin", True, True, 10240, 0.0)
        
        return {
            "id": user["id"],
            "uuid": str(user["uuid"]),
            "username": user["username"],
            "email": user["email"],
            "role": user["role"],
            "password": password
        }
    finally:
        await conn.close()


@pytest.fixture
async def async_client(isolated_test_environment):
    """Create async test client using isolated environment"""
    # Use the client from isolated_test_environment
    client, db_name = isolated_test_environment
    # For async tests, we use the same isolated environment
    from httpx import ASGITransport
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as async_client:
        yield async_client


# Use the test_user and admin_user fixtures from conftest.py
# These are properly configured with PostgreSQL test database


@pytest.fixture
async def auth_headers(isolated_test_environment, test_user_data):
    """Get authentication headers for a test user"""
    client, db_name = isolated_test_environment
    response = client.post(
        "/api/v1/auth/login",
        data={
            "username": test_user_data["username"],
            "password": test_user_data["password"]
        }
    )
    assert response.status_code == 200
    token = response.json()["access_token"]
    return {"Authorization": f"Bearer {token}"}


@pytest.fixture
async def admin_headers(isolated_test_environment, admin_user_data):
    """Get authentication headers for an admin user"""
    client, db_name = isolated_test_environment
    response = client.post(
        "/api/v1/auth/login",
        data={
            "username": admin_user_data["username"],
            "password": admin_user_data["password"]
        }
    )
    assert response.status_code == 200
    token = response.json()["access_token"]
    return {"Authorization": f"Bearer {token}"}


#######################################################################################################################
#
# Authentication Tests

class TestAuthentication:
    """Test authentication endpoints"""
    
    async def test_login_success(self, isolated_test_environment, test_user_data):
        """Test successful login"""
        client, db_name = isolated_test_environment
        response = client.post(
            "/api/v1/auth/login",
            data={
                "username": test_user_data["username"],
                "password": test_user_data["password"]
            }
        )
        assert response.status_code == 200
        data = response.json()
        assert "access_token" in data
        assert "refresh_token" in data
        assert data["token_type"] == "bearer"
        assert data["expires_in"] > 0
    
    async def test_login_invalid_username(self, isolated_test_environment):
        """Test login with invalid username"""
        client, db_name = isolated_test_environment
        response = client.post(
            "/api/v1/auth/login",
            data={
                "username": "nonexistent",
                "password": "Pass@Word#2024"
            }
        )
        assert response.status_code == 401
        assert "Incorrect username or password" in response.json()["detail"]
    
    async def test_login_invalid_password(self, isolated_test_environment, test_user_data):
        """Test login with invalid password"""
        client, db_name = isolated_test_environment
        response = client.post(
            "/api/v1/auth/login",
            data={
                "username": test_user_data["username"],
                "password": "wrongpassword"
            }
        )
        assert response.status_code == 401
        assert "Incorrect username or password" in response.json()["detail"]
    
    async def test_logout(self, isolated_test_environment, auth_headers):
        """Test logout endpoint"""
        client, db_name = isolated_test_environment
        response = client.post(
            "/api/v1/auth/logout",
            headers=auth_headers
        )
        assert response.status_code == 200
        assert "Successfully logged out" in response.json()["message"]
    
    async def test_refresh_token(self, isolated_test_environment, test_user_data):
        """Test token refresh"""
        client, db_name = isolated_test_environment
        # First login
        login_response = client.post(
            "/api/v1/auth/login",
            data={
                "username": test_user_data["username"],
                "password": test_user_data["password"]
            }
        )
        assert login_response.status_code == 200
        refresh_token = login_response.json()["refresh_token"]
        
        # Refresh token
        refresh_response = client.post(
            "/api/v1/auth/refresh",
            json={"refresh_token": refresh_token}
        )
        assert refresh_response.status_code == 200
        data = refresh_response.json()
        assert "access_token" in data
        assert "refresh_token" in data
    
    async def test_get_current_user(self, isolated_test_environment, auth_headers):
        """Test getting current user info"""
        client, db_name = isolated_test_environment
        response = client.get(
            "/api/v1/auth/me",
            headers=auth_headers
        )
        assert response.status_code == 200
        data = response.json()
        assert "id" in data
        assert "username" in data
        assert "email" in data
        assert "role" in data


#######################################################################################################################
#
# Registration Tests

class TestRegistration:
    """Test user registration"""
    
    async def test_register_success(self, isolated_test_environment):
        """Test successful registration"""
        client, db_name = isolated_test_environment
        response = client.post(
            "/api/v1/auth/register",
            json={
                "username": "newuser",
                "email": "newuser@example.com",
                "password": "New@Pass#2024!"
            }
        )
        assert response.status_code in [200, 201]
        data = response.json()
        assert data["username"] == "newuser"
        assert data["email"] == "newuser@example.com"
        assert "user_id" in data
    
    async def test_register_duplicate_username(self, isolated_test_environment, test_user_data):
        """Test registration with duplicate username"""
        client, db_name = isolated_test_environment
        response = client.post(
            "/api/v1/auth/register",
            json={
                "username": test_user_data["username"],
                "email": "different@example.com",
                "password": "Pass@Word#2024!"
            }
        )
        assert response.status_code == 409
        assert "already exists" in response.json()["detail"]
    
    async def test_register_duplicate_email(self, isolated_test_environment, test_user_data):
        """Test registration with duplicate email"""
        client, db_name = isolated_test_environment
        response = client.post(
            "/api/v1/auth/register",
            json={
                "username": "different",
                "email": test_user_data["email"],
                "password": "Pass@Word#2024!"
            }
        )
        assert response.status_code == 409
        assert "already exists" in response.json()["detail"]
    
    async def test_register_weak_password(self, isolated_test_environment):
        """Test registration with weak password"""
        client, db_name = isolated_test_environment
        response = client.post(
            "/api/v1/auth/register",
            json={
                "username": "weakpass",
                "email": "weak@example.com",
                "password": "weak"
            }
        )
        assert response.status_code in [400, 422]  # 422 for validation error, 400 for weak password
        # Either it's a Pydantic validation error (422) or our custom weak password error (400)
        # Both are acceptable for a weak password


#######################################################################################################################
#
# User Management Tests

class TestUserManagement:
    """Test user management endpoints"""
    
    async def test_get_user_profile(self, isolated_test_environment, auth_headers):
        """Test getting user profile"""
        client, db_name = isolated_test_environment
        response = client.get(
            "/api/v1/users/me",
            headers=auth_headers
        )
        assert response.status_code == 200
        data = response.json()
        assert "id" in data
        assert "username" in data
        assert "storage_quota_mb" in data
    
    async def test_change_password(self, isolated_test_environment, auth_headers):
        """Test password change"""
        client, db_name = isolated_test_environment
        response = client.post(
            "/api/v1/users/change-password",
            headers=auth_headers,
            json={
                "current_password": "Test@Pass#2024!",
                "new_password": "NewPass@Word2024!"
            }
        )
        assert response.status_code == 200
        assert "Password changed successfully" in response.json()["message"]
    
    async def test_get_sessions(self, isolated_test_environment, auth_headers):
        """Test getting user sessions"""
        client, db_name = isolated_test_environment
        response = client.get(
            "/api/v1/users/sessions",
            headers=auth_headers
        )
        assert response.status_code == 200
        data = response.json()
        assert "sessions" in data
        assert len(data["sessions"]) > 0


#######################################################################################################################
#
# Admin Tests

class TestAdminEndpoints:
    """Test admin endpoints"""
    
    async def test_list_users(self, isolated_test_environment, admin_headers):
        """Test listing users as admin"""
        client, db_name = isolated_test_environment
        response = client.get(
            "/api/v1/admin/users",
            headers=admin_headers
        )
        assert response.status_code == 200
        data = response.json()
        assert "users" in data
        assert "total" in data
        assert "page" in data
    
    async def test_update_user(self, isolated_test_environment, admin_headers, test_user_data):
        """Test updating user as admin"""
        client, db_name = isolated_test_environment
        response = client.put(
            f"/api/v1/admin/users/{test_user_data['id']}",
            headers=admin_headers,
            json={
                "is_verified": True,
                "storage_quota_mb": 10240
            }
        )
        assert response.status_code == 200
        assert "updated successfully" in response.json()["message"]
    
    async def test_create_registration_code(self, isolated_test_environment, admin_headers):
        """Test creating registration code"""
        client, db_name = isolated_test_environment
        response = client.post(
            "/api/v1/admin/registration-codes",
            headers=admin_headers,
            json={
                "max_uses": 5,
                "expiry_days": 7,
                "role_to_grant": "user"
            }
        )
        assert response.status_code == 200
        data = response.json()
        assert "code" in data
        assert data["max_uses"] == 5
        assert data["role_to_grant"] == "user"
    
    async def test_get_system_stats(self, isolated_test_environment, admin_headers):
        """Test getting system statistics"""
        client, db_name = isolated_test_environment
        response = client.get(
            "/api/v1/admin/stats",
            headers=admin_headers
        )
        assert response.status_code == 200
        data = response.json()
        assert "users" in data
        assert "storage" in data
        assert "sessions" in data
    
    async def test_get_audit_log(self, isolated_test_environment, admin_headers):
        """Test getting audit log"""
        client, db_name = isolated_test_environment
        response = client.get(
            "/api/v1/admin/audit-log",
            headers=admin_headers
        )
        assert response.status_code == 200
        data = response.json()
        assert "entries" in data


#######################################################################################################################
#
# Health Check Tests

class TestHealthEndpoints:
    """Test health monitoring endpoints"""
    
    async def test_health_check(self, isolated_test_environment):
        """Test main health check"""
        client, db_name = isolated_test_environment
        response = client.get("/api/v1/health")
        assert response.status_code in [200, 206, 503]
        data = response.json()
        assert "status" in data
        assert "checks" in data
        assert "timestamp" in data
    
    async def test_liveness_probe(self, isolated_test_environment):
        """Test liveness probe"""
        client, db_name = isolated_test_environment
        response = client.get("/api/v1/health/live")
        assert response.status_code == 200
        assert response.json()["status"] == "alive"
    
    async def test_readiness_probe(self, isolated_test_environment):
        """Test readiness probe"""
        client, db_name = isolated_test_environment
        response = client.get("/api/v1/health/ready")
        assert response.status_code in [200, 503]
        data = response.json()
        assert "status" in data
    
    async def test_metrics_endpoint(self, isolated_test_environment):
        """Test metrics endpoint"""
        client, db_name = isolated_test_environment
        response = client.get("/api/v1/health/metrics")
        assert response.status_code == 200
        data = response.json()
        assert "cpu" in data
        assert "memory" in data
        assert "disk" in data


#######################################################################################################################
#
# Security Tests

class TestSecurity:
    """Test security features"""
    
    async def test_unauthorized_access(self, isolated_test_environment):
        """Test accessing protected endpoint without auth"""
        client, db_name = isolated_test_environment
        response = client.get("/api/v1/users/me")
        assert response.status_code in [401, 403]  # Both unauthorized and forbidden are acceptable
    
    async def test_invalid_token(self, isolated_test_environment):
        """Test with invalid token"""
        client, db_name = isolated_test_environment
        response = client.get(
            "/api/v1/users/me",
            headers={"Authorization": "Bearer invalid-token"}
        )
        assert response.status_code == 401
    
    async def test_admin_endpoint_as_user(self, isolated_test_environment, auth_headers):
        """Test accessing admin endpoint as regular user"""
        client, db_name = isolated_test_environment
        response = client.get(
            "/api/v1/admin/users",
            headers=auth_headers
        )
        assert response.status_code == 403
        assert "Admin access required" in response.json()["detail"]
    
    @pytest.mark.skipif(
        get_settings().RATE_LIMIT_ENABLED == False,
        reason="Rate limiting disabled"
    )
    async def test_rate_limiting(self, isolated_test_environment):
        """Test rate limiting"""
        client, db_name = isolated_test_environment
        # Make many requests quickly
        for i in range(10):
            response = client.post(
                "/api/v1/auth/login",
                data={
                    "username": "test",
                    "password": "wrong"
                }
            )
        
        # Should eventually get rate limited
        assert response.status_code == 429
        assert "Too many" in response.json()["detail"]


#######################################################################################################################
#
# Performance Tests

class TestPerformance:
    """Test performance and concurrency"""
    
    @pytest.mark.asyncio
    async def test_concurrent_logins(self, async_client, test_user):
        """Test handling concurrent login requests"""
        async def login():
            response = await async_client.post(
                "/api/v1/auth/login",
                data={
                    "username": test_user["username"],
                    "password": test_user["password"]
                }
            )
            return response.status_code == 200
        
        # Run 10 concurrent logins
        tasks = [login() for _ in range(10)]
        results = await asyncio.gather(*tasks)
        
        # All should succeed
        assert all(results)
    
    @pytest.mark.asyncio
    async def test_session_cleanup(self, test_db_pool):
        """Test that expired sessions are cleaned up"""
        # Create expired session
        expired_time = datetime.utcnow() - timedelta(days=1)
        
        async with test_db_pool.transaction() as conn:
            if hasattr(conn, 'execute'):
                # PostgreSQL
                await conn.execute("""
                    INSERT INTO sessions (user_id, token_hash, expires_at)
                    VALUES ($1, $2, $3)
                """, 1, "expired-token-hash", expired_time)
            else:
                # SQLite
                await conn.execute("""
                    INSERT INTO sessions (user_id, token_hash, expires_at)
                    VALUES (?, ?, ?)
                """, (1, "expired-token-hash", expired_time.isoformat()))
                await conn.commit()
        
        # Run cleanup (normally done by scheduler)
        from tldw_Server_API.app.core.AuthNZ.session_manager import get_session_manager
        session_manager = await get_session_manager()
        await session_manager.cleanup_expired_sessions()
        
        # Check session was deleted
        async with test_db_pool.transaction() as conn:
            if hasattr(conn, 'fetchval'):
                count = await conn.fetchval(
                    "SELECT COUNT(*) FROM sessions WHERE token_hash = $1",
                    "expired-token-hash"
                )
            else:
                cursor = await conn.execute(
                    "SELECT COUNT(*) FROM sessions WHERE token_hash = ?",
                    ("expired-token-hash",)
                )
                count = (await cursor.fetchone())[0]
        
        assert count == 0


#######################################################################################################################
#
# Integration Tests

class TestIntegration:
    """Test full user flows"""
    
    async def test_full_user_lifecycle(self, isolated_test_environment, admin_headers):
        """Test complete user lifecycle from registration to deletion"""
        client, db_name = isolated_test_environment
        # 1. Register new user
        register_response = client.post(
            "/api/v1/auth/register",
            json={
                "username": "lifecycle_user",
                "email": "lifecycle@example.com",
                "password": "Life@Cycle#2024!"
            }
        )
        assert register_response.status_code in [200, 201]
        user_id = register_response.json()["user_id"]
        
        # 2. Login as new user
        login_response = client.post(
            "/api/v1/auth/login",
            data={
                "username": "lifecycle_user",
                "password": "Life@Cycle#2024!"
            }
        )
        assert login_response.status_code == 200
        token = login_response.json()["access_token"]
        user_headers = {"Authorization": f"Bearer {token}"}
        
        # 3. Get user profile
        profile_response = client.get(
            "/api/v1/users/me",
            headers=user_headers
        )
        assert profile_response.status_code == 200
        
        # 4. Change password
        password_response = client.post(
            "/api/v1/users/change-password",
            headers=user_headers,
            json={
                "current_password": "Life@Cycle#2024!",
                "new_password": "NewLifecyclePass456!"
            }
        )
        assert password_response.status_code == 200
        
        # 5. Admin updates user
        update_response = client.put(
            f"/api/v1/admin/users/{user_id}",
            headers=admin_headers,
            json={
                "storage_quota_mb": 20480
            }
        )
        assert update_response.status_code == 200
        
        # 6. Admin deactivates user
        deactivate_response = client.delete(
            f"/api/v1/admin/users/{user_id}",
            headers=admin_headers
        )
        assert deactivate_response.status_code == 200
        
        # 7. User can't login after deactivation
        failed_login = client.post(
            "/api/v1/auth/login",
            data={
                "username": "lifecycle_user",
                "password": "NewLifecyclePass456!"
            }
        )
        assert failed_login.status_code in [401, 403]


#
## End of test_auth_comprehensive.py
#######################################################################################################################