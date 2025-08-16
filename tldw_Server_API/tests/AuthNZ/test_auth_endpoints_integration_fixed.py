"""
Integration tests for authentication endpoints using real database.
"""

import pytest
import asyncio
from datetime import datetime, timedelta


class TestAuthEndpointsIntegration:
    """Integration tests for authentication endpoints with real database."""
    
    @pytest.mark.asyncio
    async def test_login_success(self, isolated_test_environment):
        """Test successful login."""
        client, db_name = isolated_test_environment
        
        # Register a user first
        register_response = client.post(
            "/api/v1/auth/register",
            json={
                "username": "loginuser",
                "email": "login@example.com",
                "password": "MyS3cur3P@ssw0rd!"
            }
        )
        assert register_response.status_code in [200, 201]
        
        # Login with correct credentials
        login_response = client.post(
            "/api/v1/auth/login",
            data={
                "username": "loginuser",
                "password": "MyS3cur3P@ssw0rd!"
            }
        )
        
        assert login_response.status_code == 200
        data = login_response.json()
        assert "access_token" in data
        assert "refresh_token" in data
        assert data["token_type"] == "bearer"
        assert data["expires_in"] > 0
    
    @pytest.mark.asyncio
    async def test_login_invalid_credentials(self, isolated_test_environment):
        """Test login with invalid credentials."""
        client, db_name = isolated_test_environment
        
        # Try to login with non-existent user
        response = client.post(
            "/api/v1/auth/login",
            data={
                "username": "nonexistent",
                "password": "wrongpass"
            }
        )
        
        assert response.status_code == 401
        assert "Incorrect username or password" in response.json()["detail"]
    
    @pytest.mark.asyncio
    async def test_login_inactive_account(self, isolated_test_environment):
        """Test login with inactive account."""
        client, db_name = isolated_test_environment
        
        # For this test, we need to create an inactive user
        # Since we can't directly create inactive users via API,
        # we'll need to use database connection
        import asyncpg
        import uuid
        from tldw_Server_API.app.core.AuthNZ.password_service import PasswordService
        
        # Connect to test database
        conn = await asyncpg.connect(
            host="localhost",
            port=5432,
            user="tldw_user",
            password="TestPassword123!",
            database=db_name
        )
        
        try:
            # Create inactive user directly in database
            password_service = PasswordService()
            user_uuid = str(uuid.uuid4())
            password_hash = password_service.hash_password("MyS3cur3P@ssw0rd!")
            
            await conn.execute("""
                INSERT INTO users (uuid, username, email, password_hash, role, is_active, is_verified)
                VALUES ($1, $2, $3, $4, $5, $6, $7)
            """, user_uuid, "inactiveuser", "inactive@example.com", password_hash, "user", False, True)
        finally:
            await conn.close()
        
        # Try to login with inactive account
        response = client.post(
            "/api/v1/auth/login",
            data={
                "username": "inactiveuser",
                "password": "MyS3cur3P@ssw0rd!"
            }
        )
        
        # The response could be 401 or 403 depending on implementation
        # Some systems return 401 to avoid leaking account status information
        assert response.status_code in [401, 403]
        detail = response.json().get("detail", "").lower()
        # Check for either inactive account message or generic auth failure
        assert "inactive" in detail or "incorrect" in detail or "unauthorized" in detail
    
    @pytest.mark.asyncio
    async def test_register_success(self, isolated_test_environment):
        """Test successful user registration."""
        client, db_name = isolated_test_environment
        
        response = client.post(
            "/api/v1/auth/register",
            json={
                "username": "newuser",
                "email": "new@example.com",
                "password": "S3cur3P@ssw0rd2024!"
            }
        )
        
        assert response.status_code in [200, 201]
        data = response.json()
        assert data["username"] == "newuser"
        assert data["email"] == "new@example.com"
        assert "user_id" in data or "id" in data
        
        # Verify can login with new account
        login_response = client.post(
            "/api/v1/auth/login",
            data={
                "username": "newuser",
                "password": "S3cur3P@ssw0rd2024!"
            }
        )
        assert login_response.status_code == 200
    
    @pytest.mark.asyncio
    async def test_register_duplicate_user(self, isolated_test_environment):
        """Test registration with duplicate username."""
        client, db_name = isolated_test_environment
        
        # Register first user with a secure password (avoiding sequential characters)
        first_response = client.post(
            "/api/v1/auth/register",
            json={
                "username": "duplicateuser",
                "email": "first@example.com",
                "password": "MyS3cur3P@ssw0rd!"  # Avoid sequential patterns
            }
        )
        # First registration should succeed
        assert first_response.status_code in [200, 201], f"First registration failed: {first_response.json()}"
        
        # Try to register with same username
        response = client.post(
            "/api/v1/auth/register",
            json={
                "username": "duplicateuser",
                "email": "second@example.com",
                "password": "An0th3rS3cur3P@ss!"  # Different password
            }
        )
        
        # Second registration should fail due to duplicate username
        assert response.status_code in [400, 409], f"Expected duplicate error, got: {response.status_code} - {response.json()}"
        detail = response.json().get("detail", "")
        assert any(word in detail.lower() for word in ["already", "duplicate", "exists"]), f"Expected duplicate error message, got: {detail}"
    
    @pytest.mark.asyncio
    async def test_register_weak_password(self, isolated_test_environment):
        """Test registration with weak password."""
        client, db_name = isolated_test_environment
        
        response = client.post(
            "/api/v1/auth/register",
            json={
                "username": "weakpassuser",
                "email": "weak@example.com",
                "password": "weak"
            }
        )
        
        # Should get validation error for weak password
        assert response.status_code in [400, 422]
        detail = str(response.json().get("detail", ""))
        assert "password" in detail.lower() or "weak" in detail.lower() or "characters" in detail.lower()
    
    @pytest.mark.asyncio
    async def test_refresh_token_success(self, isolated_test_environment):
        """Test successful token refresh."""
        client, db_name = isolated_test_environment
        
        # Register and login
        client.post(
            "/api/v1/auth/register",
            json={
                "username": "refreshuser",
                "email": "refresh@example.com",
                "password": "R3fr3shP@ssw0rd!"
            }
        )
        
        login_response = client.post(
            "/api/v1/auth/login",
            data={
                "username": "refreshuser",
                "password": "R3fr3shP@ssw0rd!"
            }
        )
        refresh_token = login_response.json()["refresh_token"]
        
        # Refresh the token
        refresh_response = client.post(
            "/api/v1/auth/refresh",
            json={"refresh_token": refresh_token}
        )
        
        assert refresh_response.status_code == 200
        data = refresh_response.json()
        assert "access_token" in data
        assert "refresh_token" in data
        assert data["token_type"] == "bearer"
    
    @pytest.mark.asyncio
    async def test_refresh_token_invalid(self, isolated_test_environment):
        """Test refresh with invalid token."""
        client, db_name = isolated_test_environment
        
        response = client.post(
            "/api/v1/auth/refresh",
            json={"refresh_token": "invalid.token.here"}
        )
        
        assert response.status_code == 401
    
    @pytest.mark.asyncio
    async def test_logout_success(self, isolated_test_environment):
        """Test successful logout."""
        client, db_name = isolated_test_environment
        
        # Register and login
        client.post(
            "/api/v1/auth/register",
            json={
                "username": "logoutuser",
                "email": "logout@example.com",
                "password": "L0g0utP@ssw0rd!"
            }
        )
        
        login_response = client.post(
            "/api/v1/auth/login",
            data={
                "username": "logoutuser",
                "password": "L0g0utP@ssw0rd!"
            }
        )
        token = login_response.json()["access_token"]
        
        # Logout
        logout_response = client.post(
            "/api/v1/auth/logout",
            headers={"Authorization": f"Bearer {token}"}
        )
        
        assert logout_response.status_code == 200
        assert "Successfully logged out" in logout_response.json()["message"]
    
    @pytest.mark.asyncio
    async def test_get_current_user_info(self, isolated_test_environment):
        """Test getting current user information."""
        client, db_name = isolated_test_environment
        
        # Register and login
        client.post(
            "/api/v1/auth/register",
            json={
                "username": "currentuser",
                "email": "current@example.com",
                "password": "Curr3ntP@ssw0rd!"
            }
        )
        
        login_response = client.post(
            "/api/v1/auth/login",
            data={
                "username": "currentuser",
                "password": "Curr3ntP@ssw0rd!"
            }
        )
        token = login_response.json()["access_token"]
        
        # Get current user info
        user_response = client.get(
            "/api/v1/auth/me",
            headers={"Authorization": f"Bearer {token}"}
        )
        
        assert user_response.status_code == 200
        data = user_response.json()
        assert data["username"] == "currentuser"
        assert data["email"] == "current@example.com"
        assert data["role"] == "user"


class TestAuthEndpointsValidation:
    """Validation tests for authentication endpoints."""
    
    @pytest.mark.asyncio
    async def test_register_invalid_email(self, isolated_test_environment):
        """Test registration with invalid email format."""
        client, db_name = isolated_test_environment
        
        response = client.post(
            "/api/v1/auth/register",
            json={
                "username": "invalidemail",
                "email": "not-an-email",
                "password": "V@lidP@ssw0rd2024!"
            }
        )
        
        # Should get validation error
        assert response.status_code == 422
    
    @pytest.mark.asyncio
    async def test_register_missing_fields(self, isolated_test_environment):
        """Test registration with missing fields."""
        client, db_name = isolated_test_environment
        
        # Missing password
        response = client.post(
            "/api/v1/auth/register",
            json={
                "username": "missingpass",
                "email": "missing@example.com"
            }
        )
        assert response.status_code == 422
        
        # Missing email
        response = client.post(
            "/api/v1/auth/register",
            json={
                "username": "missingemail",
                "password": "V@lidP@ssw0rd2024!"
            }
        )
        assert response.status_code == 422
    
    @pytest.mark.asyncio
    async def test_login_missing_credentials(self, isolated_test_environment):
        """Test login with missing credentials."""
        client, db_name = isolated_test_environment
        
        # Missing password
        response = client.post(
            "/api/v1/auth/login",
            data={"username": "user"}
        )
        assert response.status_code == 422
        
        # Missing username
        response = client.post(
            "/api/v1/auth/login",
            data={"password": "pass"}
        )
        assert response.status_code == 422
    
    @pytest.mark.asyncio
    async def test_multiple_login_sessions(self, isolated_test_environment):
        """Test creating multiple login sessions for same user."""
        client, db_name = isolated_test_environment
        
        # Register user
        client.post(
            "/api/v1/auth/register",
            json={
                "username": "multiuser",
                "email": "multi@example.com",
                "password": "Mult1P@ssw0rd!"
            }
        )
        
        # Login multiple times
        tokens = []
        for i in range(3):
            response = client.post(
                "/api/v1/auth/login",
                data={
                    "username": "multiuser",
                    "password": "Mult1P@ssw0rd!"
                }
            )
            assert response.status_code == 200
            tokens.append(response.json()["access_token"])
        
        # All tokens should be valid
        for token in tokens:
            response = client.get(
                "/api/v1/auth/me",
                headers={"Authorization": f"Bearer {token}"}
            )
            assert response.status_code == 200