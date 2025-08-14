# test_auth_comprehensive.py
# Description: Comprehensive test suite for authentication and user management system
#
# Imports
import pytest
import asyncio
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

@pytest.fixture
def test_settings():
    """Override settings for testing"""
    settings = Settings(
        AUTH_MODE="single_user",  # Start with single-user for simplicity
        DATABASE_URL="sqlite:///./test_users.db",
        JWT_SECRET_KEY="test-secret-key-for-testing-only",
        ENABLE_REGISTRATION=True,
        REQUIRE_REGISTRATION_CODE=False,
        RATE_LIMIT_ENABLED=False  # Disable for testing
    )
    return settings


@pytest.fixture
async def test_db_pool(test_settings):
    """Create test database pool"""
    pool = DatabasePool(test_settings)
    await pool.initialize()
    yield pool
    await pool.close()


@pytest.fixture
def test_client():
    """Create test client"""
    return TestClient(app)


@pytest.fixture
async def async_client():
    """Create async test client"""
    async with AsyncClient(app=app, base_url="http://test") as client:
        yield client


@pytest.fixture
async def test_user(test_db_pool):
    """Create a test user"""
    password_service = get_password_service()
    password_hash = password_service.hash_password("TestPassword123!")
    
    async with test_db_pool.transaction() as conn:
        if hasattr(conn, 'fetchrow'):
            # PostgreSQL
            user = await conn.fetchrow("""
                INSERT INTO users (username, email, password_hash, role, is_active, is_verified)
                VALUES ($1, $2, $3, $4, $5, $6)
                RETURNING id, username, email, role
            """, "testuser", "test@example.com", password_hash, "user", True, True)
        else:
            # SQLite
            cursor = await conn.execute("""
                INSERT INTO users (username, email, password_hash, role, is_active, is_verified)
                VALUES (?, ?, ?, ?, ?, ?)
            """, ("testuser", "test@example.com", password_hash, "user", 1, 1))
            user_id = cursor.lastrowid
            await conn.commit()
            user = {
                "id": user_id,
                "username": "testuser",
                "email": "test@example.com",
                "role": "user"
            }
    
    return {
        **user,
        "password": "TestPassword123!"  # Include plain password for testing
    }


@pytest.fixture
async def admin_user(test_db_pool):
    """Create an admin test user"""
    password_service = get_password_service()
    password_hash = password_service.hash_password("AdminPassword123!")
    
    async with test_db_pool.transaction() as conn:
        if hasattr(conn, 'fetchrow'):
            # PostgreSQL
            user = await conn.fetchrow("""
                INSERT INTO users (username, email, password_hash, role, is_active, is_verified)
                VALUES ($1, $2, $3, $4, $5, $6)
                RETURNING id, username, email, role
            """, "admin", "admin@example.com", password_hash, "admin", True, True)
        else:
            # SQLite
            cursor = await conn.execute("""
                INSERT INTO users (username, email, password_hash, role, is_active, is_verified)
                VALUES (?, ?, ?, ?, ?, ?)
            """, ("admin", "admin@example.com", password_hash, "admin", 1, 1))
            user_id = cursor.lastrowid
            await conn.commit()
            user = {
                "id": user_id,
                "username": "admin",
                "email": "admin@example.com",
                "role": "admin"
            }
    
    return {
        **user,
        "password": "AdminPassword123!"
    }


@pytest.fixture
async def auth_headers(test_client, test_user):
    """Get authentication headers for a test user"""
    response = test_client.post(
        "/api/v1/auth/login",
        data={
            "username": test_user["username"],
            "password": test_user["password"]
        }
    )
    assert response.status_code == 200
    token = response.json()["access_token"]
    return {"Authorization": f"Bearer {token}"}


@pytest.fixture
async def admin_headers(test_client, admin_user):
    """Get authentication headers for an admin user"""
    response = test_client.post(
        "/api/v1/auth/login",
        data={
            "username": admin_user["username"],
            "password": admin_user["password"]
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
    
    def test_login_success(self, test_client, test_user):
        """Test successful login"""
        response = test_client.post(
            "/api/v1/auth/login",
            data={
                "username": test_user["username"],
                "password": test_user["password"]
            }
        )
        assert response.status_code == 200
        data = response.json()
        assert "access_token" in data
        assert "refresh_token" in data
        assert data["token_type"] == "bearer"
        assert data["expires_in"] > 0
    
    def test_login_invalid_username(self, test_client):
        """Test login with invalid username"""
        response = test_client.post(
            "/api/v1/auth/login",
            data={
                "username": "nonexistent",
                "password": "password123"
            }
        )
        assert response.status_code == 401
        assert "Incorrect username or password" in response.json()["detail"]
    
    def test_login_invalid_password(self, test_client, test_user):
        """Test login with invalid password"""
        response = test_client.post(
            "/api/v1/auth/login",
            data={
                "username": test_user["username"],
                "password": "wrongpassword"
            }
        )
        assert response.status_code == 401
        assert "Incorrect username or password" in response.json()["detail"]
    
    def test_logout(self, test_client, auth_headers):
        """Test logout endpoint"""
        response = test_client.post(
            "/api/v1/auth/logout",
            headers=auth_headers
        )
        assert response.status_code == 200
        assert "Successfully logged out" in response.json()["message"]
    
    def test_refresh_token(self, test_client, test_user):
        """Test token refresh"""
        # First login
        login_response = test_client.post(
            "/api/v1/auth/login",
            data={
                "username": test_user["username"],
                "password": test_user["password"]
            }
        )
        assert login_response.status_code == 200
        refresh_token = login_response.json()["refresh_token"]
        
        # Refresh token
        refresh_response = test_client.post(
            "/api/v1/auth/refresh",
            json={"refresh_token": refresh_token}
        )
        assert refresh_response.status_code == 200
        data = refresh_response.json()
        assert "access_token" in data
        assert "refresh_token" in data
    
    def test_get_current_user(self, test_client, auth_headers):
        """Test getting current user info"""
        response = test_client.get(
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
    
    def test_register_success(self, test_client):
        """Test successful registration"""
        response = test_client.post(
            "/api/v1/auth/register",
            json={
                "username": "newuser",
                "email": "newuser@example.com",
                "password": "NewPassword123!"
            }
        )
        assert response.status_code == 201
        data = response.json()
        assert data["username"] == "newuser"
        assert data["email"] == "newuser@example.com"
        assert "user_id" in data
    
    def test_register_duplicate_username(self, test_client, test_user):
        """Test registration with duplicate username"""
        response = test_client.post(
            "/api/v1/auth/register",
            json={
                "username": test_user["username"],
                "email": "different@example.com",
                "password": "Password123!"
            }
        )
        assert response.status_code == 409
        assert "already exists" in response.json()["detail"]
    
    def test_register_duplicate_email(self, test_client, test_user):
        """Test registration with duplicate email"""
        response = test_client.post(
            "/api/v1/auth/register",
            json={
                "username": "different",
                "email": test_user["email"],
                "password": "Password123!"
            }
        )
        assert response.status_code == 409
        assert "already exists" in response.json()["detail"]
    
    def test_register_weak_password(self, test_client):
        """Test registration with weak password"""
        response = test_client.post(
            "/api/v1/auth/register",
            json={
                "username": "weakpass",
                "email": "weak@example.com",
                "password": "weak"
            }
        )
        assert response.status_code == 400
        assert "password" in response.json()["detail"].lower()


#######################################################################################################################
#
# User Management Tests

class TestUserManagement:
    """Test user management endpoints"""
    
    def test_get_user_profile(self, test_client, auth_headers):
        """Test getting user profile"""
        response = test_client.get(
            "/api/v1/users/me",
            headers=auth_headers
        )
        assert response.status_code == 200
        data = response.json()
        assert "id" in data
        assert "username" in data
        assert "storage_quota_mb" in data
    
    def test_change_password(self, test_client, auth_headers):
        """Test password change"""
        response = test_client.post(
            "/api/v1/users/change-password",
            headers=auth_headers,
            json={
                "current_password": "TestPassword123!",
                "new_password": "NewPassword456!"
            }
        )
        assert response.status_code == 200
        assert "Password changed successfully" in response.json()["message"]
    
    def test_get_sessions(self, test_client, auth_headers):
        """Test getting user sessions"""
        response = test_client.get(
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
    
    def test_list_users(self, test_client, admin_headers):
        """Test listing users as admin"""
        response = test_client.get(
            "/api/v1/admin/users",
            headers=admin_headers
        )
        assert response.status_code == 200
        data = response.json()
        assert "users" in data
        assert "total" in data
        assert "page" in data
    
    def test_update_user(self, test_client, admin_headers, test_user):
        """Test updating user as admin"""
        response = test_client.put(
            f"/api/v1/admin/users/{test_user['id']}",
            headers=admin_headers,
            json={
                "is_verified": True,
                "storage_quota_mb": 10240
            }
        )
        assert response.status_code == 200
        assert "updated successfully" in response.json()["message"]
    
    def test_create_registration_code(self, test_client, admin_headers):
        """Test creating registration code"""
        response = test_client.post(
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
    
    def test_get_system_stats(self, test_client, admin_headers):
        """Test getting system statistics"""
        response = test_client.get(
            "/api/v1/admin/stats",
            headers=admin_headers
        )
        assert response.status_code == 200
        data = response.json()
        assert "users" in data
        assert "storage" in data
        assert "sessions" in data
    
    def test_get_audit_log(self, test_client, admin_headers):
        """Test getting audit log"""
        response = test_client.get(
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
    
    def test_health_check(self, test_client):
        """Test main health check"""
        response = test_client.get("/api/v1/health")
        assert response.status_code in [200, 206, 503]
        data = response.json()
        assert "status" in data
        assert "checks" in data
        assert "timestamp" in data
    
    def test_liveness_probe(self, test_client):
        """Test liveness probe"""
        response = test_client.get("/api/v1/health/live")
        assert response.status_code == 200
        assert response.json()["status"] == "alive"
    
    def test_readiness_probe(self, test_client):
        """Test readiness probe"""
        response = test_client.get("/api/v1/health/ready")
        assert response.status_code in [200, 503]
        data = response.json()
        assert "status" in data
    
    def test_metrics_endpoint(self, test_client):
        """Test metrics endpoint"""
        response = test_client.get("/api/v1/health/metrics")
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
    
    def test_unauthorized_access(self, test_client):
        """Test accessing protected endpoint without auth"""
        response = test_client.get("/api/v1/users/me")
        assert response.status_code == 401
    
    def test_invalid_token(self, test_client):
        """Test with invalid token"""
        response = test_client.get(
            "/api/v1/users/me",
            headers={"Authorization": "Bearer invalid-token"}
        )
        assert response.status_code == 401
    
    def test_admin_endpoint_as_user(self, test_client, auth_headers):
        """Test accessing admin endpoint as regular user"""
        response = test_client.get(
            "/api/v1/admin/users",
            headers=auth_headers
        )
        assert response.status_code == 403
        assert "Admin access required" in response.json()["detail"]
    
    @pytest.mark.skipif(
        get_settings().RATE_LIMIT_ENABLED == False,
        reason="Rate limiting disabled"
    )
    def test_rate_limiting(self, test_client):
        """Test rate limiting"""
        # Make many requests quickly
        for i in range(10):
            response = test_client.post(
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
    
    def test_full_user_lifecycle(self, test_client, admin_headers):
        """Test complete user lifecycle from registration to deletion"""
        # 1. Register new user
        register_response = test_client.post(
            "/api/v1/auth/register",
            json={
                "username": "lifecycle_user",
                "email": "lifecycle@example.com",
                "password": "LifecyclePass123!"
            }
        )
        assert register_response.status_code == 201
        user_id = register_response.json()["user_id"]
        
        # 2. Login as new user
        login_response = test_client.post(
            "/api/v1/auth/login",
            data={
                "username": "lifecycle_user",
                "password": "LifecyclePass123!"
            }
        )
        assert login_response.status_code == 200
        token = login_response.json()["access_token"]
        user_headers = {"Authorization": f"Bearer {token}"}
        
        # 3. Get user profile
        profile_response = test_client.get(
            "/api/v1/users/me",
            headers=user_headers
        )
        assert profile_response.status_code == 200
        
        # 4. Change password
        password_response = test_client.post(
            "/api/v1/users/change-password",
            headers=user_headers,
            json={
                "current_password": "LifecyclePass123!",
                "new_password": "NewLifecyclePass456!"
            }
        )
        assert password_response.status_code == 200
        
        # 5. Admin updates user
        update_response = test_client.put(
            f"/api/v1/admin/users/{user_id}",
            headers=admin_headers,
            json={
                "storage_quota_mb": 20480
            }
        )
        assert update_response.status_code == 200
        
        # 6. Admin deactivates user
        deactivate_response = test_client.delete(
            f"/api/v1/admin/users/{user_id}",
            headers=admin_headers
        )
        assert deactivate_response.status_code == 200
        
        # 7. User can't login after deactivation
        failed_login = test_client.post(
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