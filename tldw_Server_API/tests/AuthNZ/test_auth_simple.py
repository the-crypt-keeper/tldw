"""
Simple authentication tests with PostgreSQL test database.
"""

import pytest
import asyncpg
import os
import uuid as uuid_lib

from tldw_Server_API.app.core.AuthNZ.password_service import PasswordService


# Test database configuration
TEST_DB_HOST = os.getenv("TEST_DB_HOST", "localhost")
TEST_DB_PORT = int(os.getenv("TEST_DB_PORT", "5432"))
TEST_DB_USER = os.getenv("TEST_DB_USER", "tldw_user")
TEST_DB_PASSWORD = os.getenv("TEST_DB_PASSWORD", "TestPassword123!")


class TestSimpleAuth:
    """Simple authentication tests."""
    
    @pytest.fixture(autouse=True)
    async def setup_test_user(self, isolated_test_environment):
        """Setup test user in the isolated database."""
        client, db_name = isolated_test_environment
        
        # Connect to the unique test database for this test
        conn = await asyncpg.connect(
            host=TEST_DB_HOST,
            port=TEST_DB_PORT,
            user=TEST_DB_USER,
            password=TEST_DB_PASSWORD,
            database=db_name
        )
        
        try:
            # Create test user
            password_service = PasswordService()
            user_uuid = str(uuid_lib.uuid4())
            password = "Test@Pass#2024!"
            password_hash = password_service.hash_password(password)
            
            await conn.execute("""
                INSERT INTO users (
                    uuid, username, email, password_hash, role,
                    is_active, is_verified, storage_quota_mb, storage_used_mb
                )
                VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9)
            """, user_uuid, "testuser", "test@example.com", password_hash,
                "user", True, True, 5120, 0.0)
            
            # Store for tests
            self.test_username = "testuser"
            self.test_password = password
            self.test_email = "test@example.com"
            self.client = client  # Store the TestClient
            
        finally:
            await conn.close()
    
    def test_login_success(self):
        """Test successful login with valid credentials."""
        response = self.client.post(
            "/api/v1/auth/login",
            data={
                "username": self.test_username,
                "password": self.test_password
            }
        )
        
        assert response.status_code == 200
        data = response.json()
        assert "access_token" in data
        assert "refresh_token" in data
        assert data["token_type"] == "bearer"
    
    def test_login_invalid_password(self):
        """Test login with invalid password."""
        response = self.client.post(
            "/api/v1/auth/login",
            data={
                "username": self.test_username,
                "password": "WrongPassword123!"
            }
        )
        
        assert response.status_code == 401
        assert "Incorrect username or password" in response.json()["detail"]
    
    def test_login_invalid_username(self):
        """Test login with non-existent username."""
        response = self.client.post(
            "/api/v1/auth/login",
            data={
                "username": "nonexistent",
                "password": self.test_password
            }
        )
        
        assert response.status_code == 401
        assert "Incorrect username or password" in response.json()["detail"]
    
    def test_register_new_user(self):
        """Test registering a new user."""
        response = self.client.post(
            "/api/v1/auth/register",
            json={
                "username": "newuser",
                "email": "newuser@example.com",
                "password": "New@User#Pass2024!"
            }
        )
        
        # Could be 200 or 201 depending on implementation
        assert response.status_code in [200, 201]
        data = response.json()
        assert data["username"] == "newuser"
        assert data["email"] == "newuser@example.com"
    
    def test_register_duplicate_username(self):
        """Test registering with existing username."""
        response = self.client.post(
            "/api/v1/auth/register",
            json={
                "username": self.test_username,  # Already exists
                "email": "different@example.com",
                "password": "Another@Pass#2024!"
            }
        )
        
        assert response.status_code == 409
        assert "already exists" in response.json()["detail"].lower()
    
    def test_get_current_user(self):
        """Test getting current user info with valid token."""
        # First login to get token
        login_response = self.client.post(
            "/api/v1/auth/login",
            data={
                "username": self.test_username,
                "password": self.test_password
            }
        )
        assert login_response.status_code == 200
        token = login_response.json()["access_token"]
        
        # Get current user info
        response = self.client.get(
            "/api/v1/auth/me",
            headers={"Authorization": f"Bearer {token}"}
        )
        
        assert response.status_code == 200
        data = response.json()
        assert data["username"] == self.test_username
        assert data["email"] == self.test_email
    
    def test_unauthorized_access(self):
        """Test accessing protected endpoint without auth."""
        response = self.client.get("/api/v1/auth/me")
        assert response.status_code == 401
    
    def test_invalid_token(self):
        """Test with invalid token."""
        response = self.client.get(
            "/api/v1/auth/me",
            headers={"Authorization": "Bearer invalid-token"}
        )
        assert response.status_code == 401