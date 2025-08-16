"""
Tests for user management endpoints.
"""

import pytest
from datetime import datetime, timedelta
from unittest.mock import AsyncMock
from httpx import AsyncClient, ASGITransport
from hypothesis import given, strategies as st, settings as hypothesis_settings, HealthCheck

from tldw_Server_API.app.main import app


class TestUserEndpoints:
    """Tests for user management endpoints."""
    
    @pytest.mark.asyncio
    async def test_get_user_profile(self, test_user, valid_access_token):
        """Test getting user profile."""
        from tldw_Server_API.app.api.v1.API_Deps.auth_deps import get_current_active_user
        
        async def mock_get_current_active_user():
            return test_user
        
        app.dependency_overrides[get_current_active_user] = mock_get_current_active_user
        
        async with AsyncClient(
            transport=ASGITransport(app=app),
            base_url="http://test"
        ) as client:
            response = await client.get(
                "/api/v1/users/me",
                headers={"Authorization": f"Bearer {valid_access_token}"}
            )
        
        assert response.status_code == 200
        data = response.json()
        assert data["username"] == test_user["username"]
        assert data["email"] == test_user["email"]
        assert data["role"] == test_user["role"]
        assert data["storage_quota_mb"] == test_user["storage_quota_mb"]
        assert data["storage_used_mb"] == test_user["storage_used_mb"]
        
        app.dependency_overrides.clear()
    
    @pytest.mark.asyncio
    async def test_update_user_profile(self, mock_db_pool, test_user, valid_access_token):
        """Test updating user profile."""
        # Setup mock connection with proper transaction context
        mock_conn = AsyncMock()
        mock_conn.execute = AsyncMock()
        mock_conn.fetchrow = AsyncMock(return_value={
            **test_user,
            'email': 'newemail@example.com'
        })
        mock_conn.commit = AsyncMock()
        
        # Mock the transaction context manager
        mock_db_pool.transaction.return_value.__aenter__.return_value = mock_conn
        mock_db_pool.transaction.return_value.__aexit__.return_value = None
        
        from tldw_Server_API.app.api.v1.API_Deps.auth_deps import (
            get_current_active_user,
            get_db_transaction
        )
        
        async def mock_get_current_active_user():
            return test_user
        
        app.dependency_overrides[get_current_active_user] = mock_get_current_active_user
        async def mock_get_db_transaction():
            yield mock_conn
        
        app.dependency_overrides[get_db_transaction] = mock_get_db_transaction
        
        async with AsyncClient(
            transport=ASGITransport(app=app),
            base_url="http://test"
        ) as client:
            response = await client.put(
                "/api/v1/users/me",
                headers={"Authorization": f"Bearer {valid_access_token}"},
                json={"email": "newemail@example.com"}
            )
        
        assert response.status_code == 200
        data = response.json()
        assert data["email"] == "newemail@example.com"
        
        app.dependency_overrides.clear()
    
    @pytest.mark.asyncio
    async def test_change_password(self, mock_db_pool, password_service, test_user, valid_access_token):
        """Test changing user password."""
        # Setup user with known password
        test_user_copy = test_user.copy()
        test_user_copy['password_hash'] = password_service.hash_password("Old@Pass#2024")
        
        # Setup mock connection with proper transaction context
        mock_conn = AsyncMock()
        mock_conn.fetchval = AsyncMock(return_value=test_user_copy['password_hash'])
        mock_conn.execute = AsyncMock()
        mock_conn.commit = AsyncMock()
        
        # Mock the transaction context manager
        mock_db_pool.transaction.return_value.__aenter__.return_value = mock_conn
        mock_db_pool.transaction.return_value.__aexit__.return_value = None
        
        from tldw_Server_API.app.api.v1.API_Deps.auth_deps import (
            get_current_active_user,
            get_db_transaction,
            get_password_service_dep
        )
        
        async def mock_get_current_active_user():
            return test_user_copy
        
        app.dependency_overrides[get_current_active_user] = mock_get_current_active_user
        async def mock_get_db_transaction():
            yield mock_conn
        
        app.dependency_overrides[get_db_transaction] = mock_get_db_transaction
        app.dependency_overrides[get_password_service_dep] = lambda: password_service
        
        async with AsyncClient(
            transport=ASGITransport(app=app),
            base_url="http://test"
        ) as client:
            response = await client.post(
                "/api/v1/users/change-password",
                headers={"Authorization": f"Bearer {valid_access_token}"},
                json={
                    "current_password": "Old@Pass#2024",
                    "new_password": "New@Secure#2024!"
                }
            )
        
        assert response.status_code == 200
        data = response.json()
        assert "Password changed successfully" in data["message"]
        
        app.dependency_overrides.clear()
    
    @pytest.mark.asyncio
    async def test_change_password_wrong_current(self, mock_db_pool, password_service, test_user, valid_access_token):
        """Test changing password with wrong current password."""
        test_user_copy = test_user.copy()
        test_user_copy['password_hash'] = password_service.hash_password("Old@Pass#2024")
        
        # Setup mock connection with proper transaction context
        mock_conn = AsyncMock()
        mock_conn.fetchval = AsyncMock(return_value=test_user_copy['password_hash'])
        
        # Mock the transaction context manager
        mock_db_pool.transaction.return_value.__aenter__.return_value = mock_conn
        mock_db_pool.transaction.return_value.__aexit__.return_value = None
        
        from tldw_Server_API.app.api.v1.API_Deps.auth_deps import (
            get_current_active_user,
            get_db_transaction,
            get_password_service_dep
        )
        
        async def mock_get_current_active_user():
            return test_user_copy
        
        app.dependency_overrides[get_current_active_user] = mock_get_current_active_user
        async def mock_get_db_transaction():
            yield mock_conn
        
        app.dependency_overrides[get_db_transaction] = mock_get_db_transaction
        app.dependency_overrides[get_password_service_dep] = lambda: password_service
        
        async with AsyncClient(
            transport=ASGITransport(app=app),
            base_url="http://test"
        ) as client:
            response = await client.post(
                "/api/v1/users/change-password",
                headers={"Authorization": f"Bearer {valid_access_token}"},
                json={
                    "current_password": "wrongpass",
                    "new_password": "New@Secure#2024!"
                }
            )
        
        # Should return 401 for incorrect password authentication
        assert response.status_code == 401
        assert "Current password is incorrect" in response.json()["detail"]
        
        app.dependency_overrides.clear()
    
    @pytest.mark.asyncio
    async def test_get_user_sessions(self, mock_db_pool, session_manager, test_user, valid_access_token):
        """Test getting user sessions."""
        mock_sessions = [
            {
                'id': 1,  # Changed to integer to match database schema
                'user_id': test_user['id'],
                'created_at': datetime.utcnow().isoformat(),
                'last_activity': datetime.utcnow().isoformat(),
                'ip_address': '127.0.0.1',
                'user_agent': 'TestClient/1.0',
                'device_id': None,
                'expires_at': (datetime.utcnow() + timedelta(hours=1)).isoformat()
            },
            {
                'id': 2,  # Changed to integer to match database schema
                'user_id': test_user['id'],
                'created_at': datetime.utcnow().isoformat(),
                'last_activity': datetime.utcnow().isoformat(),
                'ip_address': '192.168.1.1',
                'user_agent': 'Mozilla/5.0',
                'device_id': None,
                'expires_at': (datetime.utcnow() + timedelta(hours=1)).isoformat()
            }
        ]
        
        session_manager.get_user_sessions = AsyncMock(return_value=mock_sessions)
        session_manager.get_active_sessions = AsyncMock(return_value=mock_sessions)
        
        from tldw_Server_API.app.api.v1.API_Deps.auth_deps import (
            get_current_active_user,
            get_session_manager_dep
        )
        
        async def mock_get_current_active_user():
            return test_user
        
        app.dependency_overrides[get_current_active_user] = mock_get_current_active_user
        app.dependency_overrides[get_session_manager_dep] = lambda: session_manager
        
        async with AsyncClient(
            transport=ASGITransport(app=app),
            base_url="http://test"
        ) as client:
            response = await client.get(
                "/api/v1/users/sessions",
                headers={"Authorization": f"Bearer {valid_access_token}"}
            )
        
        assert response.status_code == 200
        data = response.json()
        assert len(data) == 2
        assert data[0]["id"] == 1
        assert data[1]["id"] == 2
        
        app.dependency_overrides.clear()
    
    @pytest.mark.asyncio
    async def test_revoke_session(self, session_manager, test_user, valid_access_token):
        """Test revoking a user session."""
        # Mock get_user_sessions to return a session with id 123
        session_manager.get_user_sessions = AsyncMock(return_value=[
            {'id': 123, 'user_id': test_user['id'], 'created_at': datetime.utcnow(), 'expires_at': datetime.utcnow() + timedelta(hours=1)}
        ])
        session_manager.revoke_session = AsyncMock()
        
        from tldw_Server_API.app.api.v1.API_Deps.auth_deps import (
            get_current_active_user,
            get_session_manager_dep
        )
        
        async def mock_get_current_active_user():
            return test_user
        
        app.dependency_overrides[get_current_active_user] = mock_get_current_active_user
        app.dependency_overrides[get_session_manager_dep] = lambda: session_manager
        
        async with AsyncClient(
            transport=ASGITransport(app=app),
            base_url="http://test"
        ) as client:
            response = await client.delete(
                "/api/v1/users/sessions/123",  # Use integer session ID
                headers={"Authorization": f"Bearer {valid_access_token}"}
            )
        
        assert response.status_code == 200
        data = response.json()
        assert "Session revoked successfully" in data["message"]
        
        # Verify the session was revoked
        session_manager.revoke_session.assert_called_once_with(
            123, 
            revoked_by=test_user['id'], 
            reason="User requested revocation"
        )
        
        app.dependency_overrides.clear()
    
    @pytest.mark.asyncio
    async def test_get_storage_quota(self, storage_service, test_user, valid_access_token):
        """Test getting storage quota information."""
        # Create a mock storage info dictionary matching the service's return format
        storage_info = {
            "user_id": test_user['id'],
            "total_mb": 100,
            "quota_mb": 1000,
            "available_mb": 900,
            "usage_percentage": 10.0,
            "user_data_mb": 100,
            "chromadb_mb": 0,
            "total_bytes": 104857600,
            "user_data_bytes": 104857600,
            "chromadb_bytes": 0,
            "calculated_at": "2024-01-01T00:00:00"
        }
        
        storage_service.calculate_user_storage = AsyncMock(return_value=storage_info)
        
        from tldw_Server_API.app.api.v1.API_Deps.auth_deps import (
            get_current_active_user,
            get_storage_service_dep
        )
        
        async def mock_get_current_active_user():
            return test_user
        
        app.dependency_overrides[get_current_active_user] = mock_get_current_active_user
        app.dependency_overrides[get_storage_service_dep] = lambda: storage_service
        
        async with AsyncClient(
            transport=ASGITransport(app=app),
            base_url="http://test"
        ) as client:
            response = await client.get(
                "/api/v1/users/storage",
                headers={"Authorization": f"Bearer {valid_access_token}"}
            )
        
        assert response.status_code == 200
        data = response.json()
        assert data["storage_quota_mb"] == 1000
        assert data["storage_used_mb"] == 100
        assert data["available_mb"] == 900
        assert data["usage_percentage"] == 10.0
        
        app.dependency_overrides.clear()


class TestUserEndpointsProperty:
    """Property-based tests for user endpoints."""
    
    @pytest.mark.asyncio
    @given(
        email=st.emails()
    )
    @hypothesis_settings(max_examples=10, deadline=5000, suppress_health_check=[HealthCheck.function_scoped_fixture])
    async def test_update_profile_with_various_emails(self, email, mock_db_pool, test_user, valid_access_token):
        """Test updating profile with various email formats."""
        updated_user = {**test_user, 'email': email}
        
        # Setup mock connection with proper transaction context
        mock_conn = AsyncMock()
        mock_conn.execute = AsyncMock()
        mock_conn.fetchrow = AsyncMock(return_value=updated_user)
        mock_conn.commit = AsyncMock()
        
        # Mock the transaction context manager
        mock_db_pool.transaction.return_value.__aenter__.return_value = mock_conn
        mock_db_pool.transaction.return_value.__aexit__.return_value = None
        
        from tldw_Server_API.app.api.v1.API_Deps.auth_deps import (
            get_current_active_user,
            get_db_transaction
        )
        
        async def mock_get_current_active_user():
            return test_user
        
        app.dependency_overrides[get_current_active_user] = mock_get_current_active_user
        async def mock_get_db_transaction():
            yield mock_conn
        
        app.dependency_overrides[get_db_transaction] = mock_get_db_transaction
        
        async with AsyncClient(
            transport=ASGITransport(app=app),
            base_url="http://test"
        ) as client:
            response = await client.put(
                "/api/v1/users/me",
                headers={"Authorization": f"Bearer {valid_access_token}"},
                json={"email": email}
            )
        
        # The endpoint may return 400 if the email is the same as current (no updates)
        if response.status_code == 400:
            # This is expected when the new email is the same as the current one
            assert "No updates provided" in response.json()["detail"]
        else:
            assert response.status_code == 200
            data = response.json()
            # Email should be lowercased by the endpoint
            assert data["email"] == email.lower()
        
        app.dependency_overrides.clear()
    
    @pytest.mark.asyncio
    @given(
        new_password=st.text(min_size=8, max_size=100).filter(
            lambda x: not x.isspace() and any(c.isdigit() for c in x)
        )
    )
    @hypothesis_settings(max_examples=5, deadline=5000, suppress_health_check=[HealthCheck.function_scoped_fixture])
    async def test_change_password_various_formats(
        self, new_password, mock_db_pool, password_service, test_user, valid_access_token
    ):
        """Test changing password with various valid formats."""
        test_user_copy = test_user.copy()
        # Use a strong password for the current password
        test_user_copy['password_hash'] = password_service.hash_password("Current@Pass#2024")
        
        # Setup mock connection with proper transaction context
        mock_conn = AsyncMock()
        mock_conn.fetchval = AsyncMock(return_value=test_user_copy['password_hash'])
        mock_conn.execute = AsyncMock()
        mock_conn.commit = AsyncMock()
        
        # Mock the transaction context manager
        mock_db_pool.transaction.return_value.__aenter__.return_value = mock_conn
        mock_db_pool.transaction.return_value.__aexit__.return_value = None
        
        from tldw_Server_API.app.api.v1.API_Deps.auth_deps import (
            get_current_active_user,
            get_db_transaction,
            get_password_service_dep
        )
        
        async def mock_get_current_active_user():
            return test_user_copy
        
        app.dependency_overrides[get_current_active_user] = mock_get_current_active_user
        async def mock_get_db_transaction():
            yield mock_conn
        
        app.dependency_overrides[get_db_transaction] = mock_get_db_transaction
        app.dependency_overrides[get_password_service_dep] = lambda: password_service
        
        async with AsyncClient(
            transport=ASGITransport(app=app),
            base_url="http://test"
        ) as client:
            response = await client.post(
                "/api/v1/users/change-password",
                headers={"Authorization": f"Bearer {valid_access_token}"},
                json={
                    "current_password": "Current@Pass#2024",
                    "new_password": new_password
                }
            )
        
        # Should either succeed or fail with validation error
        assert response.status_code in [200, 400, 422]
        
        if response.status_code == 200:
            assert "Password changed successfully" in response.json()["message"]
        
        app.dependency_overrides.clear()