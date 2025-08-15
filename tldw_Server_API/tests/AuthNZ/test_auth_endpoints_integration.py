"""
Integration tests for authentication endpoints.
"""

import pytest
import asyncio
from datetime import datetime, timedelta
from unittest.mock import AsyncMock, MagicMock
from httpx import AsyncClient, ASGITransport
from hypothesis import given, strategies as st, settings as hypothesis_settings

from fastapi import FastAPI
from fastapi.testclient import TestClient

from tldw_Server_API.app.main import app
from tldw_Server_API.app.core.AuthNZ.exceptions import (
    InvalidCredentialsError,
    UserNotFoundError,
    AccountInactiveError,
    DuplicateUserError,
    WeakPasswordError
)


class TestAuthEndpointsIntegration:
    """Integration tests for authentication endpoints."""
    
    @pytest.mark.asyncio
    async def test_login_success(self, mock_db_pool, password_service, test_user):
        """Test successful login."""
        # Setup mock
        test_user_copy = test_user.copy()
        test_user_copy['password_hash'] = password_service.hash_password("Test@Pass#2024")
        
        mock_db_pool.fetchrow = AsyncMock(return_value=test_user_copy)
        mock_db_pool.execute = AsyncMock()
        
        # Override dependency
        from tldw_Server_API.app.api.v1.endpoints.auth import router
        from tldw_Server_API.app.api.v1.API_Deps.auth_deps import get_db_transaction
        
        app.dependency_overrides[get_db_transaction] = lambda: mock_db_pool
        
        async with AsyncClient(
            transport=ASGITransport(app=app),
            base_url="http://test"
        ) as client:
            response = await client.post(
                "/api/v1/auth/login",
                data={
                    "username": "testuser",
                    "password": "Test@Pass#2024"
                }
            )
        
        assert response.status_code == 200
        data = response.json()
        assert "access_token" in data
        assert "refresh_token" in data
        assert data["token_type"] == "bearer"
        assert data["expires_in"] > 0
        
        # Cleanup
        app.dependency_overrides.clear()
    
    @pytest.mark.asyncio
    async def test_login_invalid_credentials(self, mock_db_pool):
        """Test login with invalid credentials."""
        mock_db_pool.fetchrow = AsyncMock(return_value=None)
        
        from tldw_Server_API.app.api.v1.API_Deps.auth_deps import get_db_transaction
        app.dependency_overrides[get_db_transaction] = lambda: mock_db_pool
        
        async with AsyncClient(
            transport=ASGITransport(app=app),
            base_url="http://test"
        ) as client:
            response = await client.post(
                "/api/v1/auth/login",
                data={
                    "username": "nonexistent",
                    "password": "wrongpass"
                }
            )
        
        assert response.status_code == 401
        assert "Incorrect username or password" in response.json()["detail"]
        
        app.dependency_overrides.clear()
    
    @pytest.mark.asyncio
    async def test_login_inactive_account(self, mock_db_pool, password_service, inactive_user):
        """Test login with inactive account."""
        inactive_user_copy = inactive_user.copy()
        inactive_user_copy['password_hash'] = password_service.hash_password("testpass123")
        
        mock_db_pool.fetchrow = AsyncMock(return_value=inactive_user_copy)
        
        from tldw_Server_API.app.api.v1.API_Deps.auth_deps import get_db_transaction
        app.dependency_overrides[get_db_transaction] = lambda: mock_db_pool
        
        async with AsyncClient(
            transport=ASGITransport(app=app),
            base_url="http://test"
        ) as client:
            response = await client.post(
                "/api/v1/auth/login",
                data={
                    "username": "inactiveuser",
                    "password": "Test@Pass#2024"
                }
            )
        
        assert response.status_code == 403
        assert "Account is inactive" in response.json()["detail"]
        
        app.dependency_overrides.clear()
    
    @pytest.mark.asyncio
    async def test_register_success(self, mock_db_pool, registration_service):
        """Test successful user registration."""
        # Mock database to simulate no existing user
        mock_db_pool.fetchrow = AsyncMock(return_value=None)
        mock_db_pool.execute = AsyncMock()
        
        # Mock the registration service to return success
        registration_service.register_user = AsyncMock(return_value={
            'user_id': 1,
            'username': 'newuser',
            'email': 'new@example.com',
            'is_verified': False
        })
        
        from tldw_Server_API.app.api.v1.API_Deps.auth_deps import get_registration_service_dep
        app.dependency_overrides[get_registration_service_dep] = lambda: registration_service
        
        async with AsyncClient(
            transport=ASGITransport(app=app),
            base_url="http://test"
        ) as client:
            response = await client.post(
                "/api/v1/auth/register",
                json={
                    "username": "newuser",
                    "email": "new@example.com",
                    "password": "Secure@Pass#2024!"
                }
            )
        
        assert response.status_code == 200
        data = response.json()
        assert data["message"] == "Registration successful"
        assert data["username"] == "newuser"
        assert data["requires_verification"] == True
        
        app.dependency_overrides.clear()
    
    @pytest.mark.asyncio
    async def test_register_duplicate_user(self, mock_db_pool, registration_service):
        """Test registration with duplicate username."""
        registration_service.register_user = AsyncMock(
            side_effect=DuplicateUserError("Username already exists")
        )
        
        from tldw_Server_API.app.api.v1.API_Deps.auth_deps import get_registration_service_dep
        app.dependency_overrides[get_registration_service_dep] = lambda: registration_service
        
        async with AsyncClient(
            transport=ASGITransport(app=app),
            base_url="http://test"
        ) as client:
            response = await client.post(
                "/api/v1/auth/register",
                json={
                    "username": "existinguser",
                    "email": "existing@example.com",
                    "password": "Secure@Pass#2024!"
                }
            )
        
        assert response.status_code == 409
        assert "Username already exists" in response.json()["detail"]
        
        app.dependency_overrides.clear()
    
    @pytest.mark.asyncio
    async def test_register_weak_password(self, registration_service):
        """Test registration with weak password."""
        registration_service.register_user = AsyncMock(
            side_effect=WeakPasswordError("Password must be at least 8 characters")
        )
        
        from tldw_Server_API.app.api.v1.API_Deps.auth_deps import get_registration_service_dep
        app.dependency_overrides[get_registration_service_dep] = lambda: registration_service
        
        async with AsyncClient(
            transport=ASGITransport(app=app),
            base_url="http://test"
        ) as client:
            response = await client.post(
                "/api/v1/auth/register",
                json={
                    "username": "newuser",
                    "email": "new@example.com",
                    "password": "weak"
                }
            )
        
        assert response.status_code == 400
        assert "Password must be at least 8 characters" in response.json()["detail"]
        
        app.dependency_overrides.clear()
    
    @pytest.mark.asyncio
    async def test_refresh_token_success(self, mock_db_pool, jwt_service, test_user, valid_refresh_token):
        """Test successful token refresh."""
        mock_db_pool.fetchrow = AsyncMock(return_value=test_user)
        
        from tldw_Server_API.app.api.v1.API_Deps.auth_deps import (
            get_db_transaction,
            get_jwt_service_dep
        )
        app.dependency_overrides[get_db_transaction] = lambda: mock_db_pool
        app.dependency_overrides[get_jwt_service_dep] = lambda: jwt_service
        
        async with AsyncClient(
            transport=ASGITransport(app=app),
            base_url="http://test"
        ) as client:
            response = await client.post(
                "/api/v1/auth/refresh",
                json={"refresh_token": valid_refresh_token}
            )
        
        assert response.status_code == 200
        data = response.json()
        assert "access_token" in data
        assert "refresh_token" in data
        assert data["token_type"] == "bearer"
        
        app.dependency_overrides.clear()
    
    @pytest.mark.asyncio
    async def test_refresh_token_invalid(self, jwt_service):
        """Test refresh with invalid token."""
        from tldw_Server_API.app.api.v1.API_Deps.auth_deps import get_jwt_service_dep
        app.dependency_overrides[get_jwt_service_dep] = lambda: jwt_service
        
        async with AsyncClient(
            transport=ASGITransport(app=app),
            base_url="http://test"
        ) as client:
            response = await client.post(
                "/api/v1/auth/refresh",
                json={"refresh_token": "invalid.token.here"}
            )
        
        assert response.status_code == 401
        
        app.dependency_overrides.clear()
    
    @pytest.mark.asyncio
    async def test_logout_success(self, mock_db_pool, jwt_service, session_manager, test_user, valid_access_token):
        """Test successful logout."""
        from tldw_Server_API.app.api.v1.API_Deps.auth_deps import (
            get_current_user,
            get_session_manager_dep,
            get_jwt_service_dep
        )
        
        # Mock current user
        async def mock_get_current_user():
            return test_user
        
        app.dependency_overrides[get_current_user] = mock_get_current_user
        app.dependency_overrides[get_session_manager_dep] = lambda: session_manager
        app.dependency_overrides[get_jwt_service_dep] = lambda: jwt_service
        
        async with AsyncClient(
            transport=ASGITransport(app=app),
            base_url="http://test"
        ) as client:
            response = await client.post(
                "/api/v1/auth/logout",
                headers={"Authorization": f"Bearer {valid_access_token}"}
            )
        
        assert response.status_code == 200
        data = response.json()
        assert "Successfully logged out" in data["message"]
        
        app.dependency_overrides.clear()
    
    @pytest.mark.asyncio
    async def test_get_current_user_info(self, test_user, valid_access_token):
        """Test getting current user information."""
        from tldw_Server_API.app.api.v1.API_Deps.auth_deps import get_current_active_user
        
        async def mock_get_current_active_user():
            return test_user
        
        app.dependency_overrides[get_current_active_user] = mock_get_current_active_user
        
        async with AsyncClient(
            transport=ASGITransport(app=app),
            base_url="http://test"
        ) as client:
            response = await client.get(
                "/api/v1/auth/me",
                headers={"Authorization": f"Bearer {valid_access_token}"}
            )
        
        assert response.status_code == 200
        data = response.json()
        assert data["username"] == test_user["username"]
        assert data["email"] == test_user["email"]
        assert data["role"] == test_user["role"]
        
        app.dependency_overrides.clear()


class TestAuthEndpointsProperty:
    """Property-based tests for authentication endpoints."""
    
    @pytest.mark.asyncio
    @given(
        username=st.text(min_size=3, max_size=50).filter(lambda x: x.strip() and x.isalnum()),
        email=st.emails(),
        password=st.text(min_size=8, max_size=100).filter(lambda x: not x.isspace())
    )
    @hypothesis_settings(max_examples=10, deadline=5000)
    async def test_register_with_various_inputs(self, username, email, password, registration_service):
        """Test registration with various valid inputs."""
        registration_service.register_user = AsyncMock(return_value={
            'user_id': 1,
            'username': username,
            'email': email,
            'is_verified': False
        })
        
        from tldw_Server_API.app.api.v1.API_Deps.auth_deps import get_registration_service_dep
        app.dependency_overrides[get_registration_service_dep] = lambda: registration_service
        
        async with AsyncClient(
            transport=ASGITransport(app=app),
            base_url="http://test"
        ) as client:
            response = await client.post(
                "/api/v1/auth/register",
                json={
                    "username": username,
                    "email": email,
                    "password": password
                }
            )
        
        # Should either succeed or fail with specific validation errors
        assert response.status_code in [200, 400, 409]
        
        if response.status_code == 200:
            data = response.json()
            assert data["username"] == username
            assert data["email"] == email
        
        app.dependency_overrides.clear()
    
    @pytest.mark.asyncio
    @given(
        token_length=st.integers(min_value=10, max_value=1000)
    )
    @hypothesis_settings(max_examples=10, deadline=5000)
    async def test_refresh_with_various_token_lengths(self, token_length, jwt_service):
        """Test refresh endpoint with tokens of various lengths."""
        from tldw_Server_API.app.api.v1.API_Deps.auth_deps import get_jwt_service_dep
        app.dependency_overrides[get_jwt_service_dep] = lambda: jwt_service
        
        # Generate a token-like string of specified length
        fake_token = "a" * token_length
        
        async with AsyncClient(
            transport=ASGITransport(app=app),
            base_url="http://test"
        ) as client:
            response = await client.post(
                "/api/v1/auth/refresh",
                json={"refresh_token": fake_token}
            )
        
        # Should always return 401 for invalid tokens
        assert response.status_code == 401
        
        app.dependency_overrides.clear()