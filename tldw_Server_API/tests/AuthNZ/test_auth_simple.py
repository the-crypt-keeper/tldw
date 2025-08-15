"""
Simple authentication tests with PostgreSQL test database.
"""

import pytest
import asyncio
import asyncpg
import os
import uuid as uuid_lib
from httpx import AsyncClient
from fastapi.testclient import TestClient

from tldw_Server_API.app.main import app
from tldw_Server_API.app.core.AuthNZ.password_service import PasswordService


# Test database configuration
TEST_DB_HOST = os.getenv("TEST_DB_HOST", "localhost")
TEST_DB_PORT = int(os.getenv("TEST_DB_PORT", "5432"))
TEST_DB_USER = os.getenv("TEST_DB_USER", "tldw_user")
TEST_DB_PASSWORD = os.getenv("TEST_DB_PASSWORD", "TestPassword123!")
TEST_DB_NAME = "tldw_test"


class TestSimpleAuth:
    """Simple authentication tests."""
    
    @pytest.fixture(autouse=True)
    async def setup_database(self):
        """Setup test database before each test."""
        # Connect to test database
        conn = await asyncpg.connect(
            host=TEST_DB_HOST,
            port=TEST_DB_PORT,
            user=TEST_DB_USER,
            password=TEST_DB_PASSWORD,
            database=TEST_DB_NAME
        )
        
        try:
            # Clean tables
            await conn.execute("TRUNCATE TABLE users CASCADE")
            
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
            
            yield
            
        finally:
            # Cleanup
            await conn.execute("TRUNCATE TABLE users CASCADE")
            await conn.close()
    
    def test_login_success(self):
        """Test successful login with valid credentials."""
        with TestClient(app) as client:
            response = client.post(
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
        with TestClient(app) as client:
            response = client.post(
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
        with TestClient(app) as client:
            response = client.post(
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
        with TestClient(app) as client:
            response = client.post(
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
        with TestClient(app) as client:
            response = client.post(
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
        with TestClient(app) as client:
            # First login to get token
            login_response = client.post(
                "/api/v1/auth/login",
                data={
                    "username": self.test_username,
                    "password": self.test_password
                }
            )
            assert login_response.status_code == 200
            token = login_response.json()["access_token"]
            
            # Get current user info
            response = client.get(
                "/api/v1/auth/me",
                headers={"Authorization": f"Bearer {token}"}
            )
            
            assert response.status_code == 200
            data = response.json()
            assert data["username"] == self.test_username
            assert data["email"] == self.test_email
    
    def test_unauthorized_access(self):
        """Test accessing protected endpoint without auth."""
        with TestClient(app) as client:
            response = client.get("/api/v1/auth/me")
            assert response.status_code == 401
    
    def test_invalid_token(self):
        """Test with invalid token."""
        with TestClient(app) as client:
            response = client.get(
                "/api/v1/auth/me",
                headers={"Authorization": "Bearer invalid-token"}
            )
            assert response.status_code == 401