"""
Tests for user management endpoints.
"""

import pytest
from datetime import datetime
from unittest.mock import AsyncMock
from httpx import AsyncClient, ASGITransport
from hypothesis import given, strategies as st, settings as hypothesis_settings

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
        mock_db_pool.execute = AsyncMock()
        mock_db_pool.fetchrow = AsyncMock(return_value={
            **test_user,
            'email': 'newemail@example.com'
        })
        
        from tldw_Server_API.app.api.v1.API_Deps.auth_deps import (
            get_current_active_user,
            get_db_transaction
        )
        
        async def mock_get_current_active_user():
            return test_user
        
        app.dependency_overrides[get_current_active_user] = mock_get_current_active_user
        app.dependency_overrides[get_db_transaction] = lambda: mock_db_pool
        
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
        
        mock_db_pool.fetchrow = AsyncMock(return_value=test_user_copy)
        mock_db_pool.execute = AsyncMock()
        
        from tldw_Server_API.app.api.v1.API_Deps.auth_deps import (
            get_current_active_user,
            get_db_transaction,
            get_password_service_dep
        )
        
        async def mock_get_current_active_user():
            return test_user_copy
        
        app.dependency_overrides[get_current_active_user] = mock_get_current_active_user
        app.dependency_overrides[get_db_transaction] = lambda: mock_db_pool
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
        
        mock_db_pool.fetchrow = AsyncMock(return_value=test_user_copy)
        
        from tldw_Server_API.app.api.v1.API_Deps.auth_deps import (
            get_current_active_user,
            get_db_transaction,
            get_password_service_dep
        )
        
        async def mock_get_current_active_user():
            return test_user_copy
        
        app.dependency_overrides[get_current_active_user] = mock_get_current_active_user
        app.dependency_overrides[get_db_transaction] = lambda: mock_db_pool
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
        
        assert response.status_code == 400
        assert "Current password is incorrect" in response.json()["detail"]
        
        app.dependency_overrides.clear()
    
    @pytest.mark.asyncio
    async def test_get_user_sessions(self, mock_db_pool, session_manager, test_user, valid_access_token):
        """Test getting user sessions."""
        mock_sessions = [
            {
                'id': 'session-1',
                'user_id': test_user['id'],
                'created_at': datetime.utcnow().isoformat(),
                'last_activity': datetime.utcnow().isoformat(),
                'ip_address': '127.0.0.1',
                'user_agent': 'TestClient/1.0',
                'is_active': True
            },
            {
                'id': 'session-2',
                'user_id': test_user['id'],
                'created_at': datetime.utcnow().isoformat(),
                'last_activity': datetime.utcnow().isoformat(),
                'ip_address': '192.168.1.1',
                'user_agent': 'Mozilla/5.0',
                'is_active': True
            }
        ]
        
        session_manager.get_user_sessions = AsyncMock(return_value=mock_sessions)
        
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
        assert data[0]["id"] == "session-1"
        assert data[1]["id"] == "session-2"
        
        app.dependency_overrides.clear()
    
    @pytest.mark.asyncio
    async def test_revoke_session(self, session_manager, test_user, valid_access_token):
        """Test revoking a user session."""
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
                "/api/v1/users/sessions/session-123",
                headers={"Authorization": f"Bearer {valid_access_token}"}
            )
        
        assert response.status_code == 200
        data = response.json()
        assert "Session revoked successfully" in data["message"]
        
        # Verify the session was revoked
        session_manager.revoke_session.assert_called_once_with("session-123", test_user['id'])
        
        app.dependency_overrides.clear()
    
    @pytest.mark.asyncio
    async def test_get_storage_quota(self, storage_service, test_user, valid_access_token):
        """Test getting storage quota information."""
        from tldw_Server_API.app.services.storage_quota_service import StorageQuotaInfo
        
        storage_info = StorageQuotaInfo(
            quota_mb=1000,
            used_mb=100,
            available_mb=900,
            percentage_used=10.0
        )
        
        storage_service.get_user_storage_info = AsyncMock(return_value=storage_info)
        
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
        assert data["quota_mb"] == 1000
        assert data["used_mb"] == 100
        assert data["available_mb"] == 900
        assert data["percentage_used"] == 10.0
        
        app.dependency_overrides.clear()


class TestUserEndpointsProperty:
    """Property-based tests for user endpoints."""
    
    @pytest.mark.asyncio
    @given(
        email=st.emails()
    )
    @hypothesis_settings(max_examples=10, deadline=5000)
    async def test_update_profile_with_various_emails(self, email, mock_db_pool, test_user, valid_access_token):
        """Test updating profile with various email formats."""
        updated_user = {**test_user, 'email': email}
        mock_db_pool.execute = AsyncMock()
        mock_db_pool.fetchrow = AsyncMock(return_value=updated_user)
        
        from tldw_Server_API.app.api.v1.API_Deps.auth_deps import (
            get_current_active_user,
            get_db_transaction
        )
        
        async def mock_get_current_active_user():
            return test_user
        
        app.dependency_overrides[get_current_active_user] = mock_get_current_active_user
        app.dependency_overrides[get_db_transaction] = lambda: mock_db_pool
        
        async with AsyncClient(
            transport=ASGITransport(app=app),
            base_url="http://test"
        ) as client:
            response = await client.put(
                "/api/v1/users/me",
                headers={"Authorization": f"Bearer {valid_access_token}"},
                json={"email": email}
            )
        
        assert response.status_code == 200
        data = response.json()
        assert data["email"] == email
        
        app.dependency_overrides.clear()
    
    @pytest.mark.asyncio
    @given(
        new_password=st.text(min_size=8, max_size=100).filter(
            lambda x: not x.isspace() and any(c.isdigit() for c in x)
        )
    )
    @hypothesis_settings(max_examples=5, deadline=5000)
    async def test_change_password_various_formats(
        self, new_password, mock_db_pool, password_service, test_user, valid_access_token
    ):
        """Test changing password with various valid formats."""
        test_user_copy = test_user.copy()
        test_user_copy['password_hash'] = password_service.hash_password("currentpass")
        
        mock_db_pool.fetchrow = AsyncMock(return_value=test_user_copy)
        mock_db_pool.execute = AsyncMock()
        
        from tldw_Server_API.app.api.v1.API_Deps.auth_deps import (
            get_current_active_user,
            get_db_transaction,
            get_password_service_dep
        )
        
        async def mock_get_current_active_user():
            return test_user_copy
        
        app.dependency_overrides[get_current_active_user] = mock_get_current_active_user
        app.dependency_overrides[get_db_transaction] = lambda: mock_db_pool
        app.dependency_overrides[get_password_service_dep] = lambda: password_service
        
        async with AsyncClient(
            transport=ASGITransport(app=app),
            base_url="http://test"
        ) as client:
            response = await client.post(
                "/api/v1/users/change-password",
                headers={"Authorization": f"Bearer {valid_access_token}"},
                json={
                    "current_password": "currentpass",
                    "new_password": new_password
                }
            )
        
        # Should either succeed or fail with validation error
        assert response.status_code in [200, 400]
        
        if response.status_code == 200:
            assert "Password changed successfully" in response.json()["message"]
        
        app.dependency_overrides.clear()