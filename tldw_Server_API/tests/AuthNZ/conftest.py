"""
Shared fixtures and configuration for AuthNZ tests.
"""

import pytest
import asyncio
import tempfile
import shutil
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, Any, AsyncGenerator
from unittest.mock import Mock, AsyncMock

from tldw_Server_API.app.core.AuthNZ.database import DatabasePool
from tldw_Server_API.app.core.AuthNZ.password_service import PasswordService
from tldw_Server_API.app.core.AuthNZ.jwt_service import JWTService
from tldw_Server_API.app.core.AuthNZ.settings import Settings
from tldw_Server_API.app.core.AuthNZ.session_manager import SessionManager
from tldw_Server_API.app.core.AuthNZ.rate_limiter import RateLimiter
from tldw_Server_API.app.services.registration_service import RegistrationService
from tldw_Server_API.app.core.config import settings


@pytest.fixture(scope="session")
def event_loop():
    """Create an instance of the default event loop for the test session."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


@pytest.fixture(scope="session")
async def test_db_dir():
    """Create a temporary directory for test databases."""
    temp_dir = tempfile.mkdtemp(prefix="auth_test_")
    yield Path(temp_dir)
    # Cleanup after tests
    shutil.rmtree(temp_dir, ignore_errors=True)


@pytest.fixture
async def mock_db_pool():
    """Create a mock database pool for testing."""
    pool = AsyncMock(spec=DatabasePool)
    
    # Mock connection context manager
    mock_conn = AsyncMock()
    pool.transaction.return_value.__aenter__.return_value = mock_conn
    pool.transaction.return_value.__aexit__.return_value = None
    
    # Mock fetchone for user queries
    pool.fetchone = AsyncMock()
    
    # Mock execute for updates
    pool.execute = AsyncMock()
    
    return pool


@pytest.fixture
def password_service():
    """Create a password service instance for testing."""
    return PasswordService()


@pytest.fixture
def jwt_settings():
    """Create JWT settings for testing."""
    return Settings(
        AUTH_MODE="multi_user",
        JWT_SECRET_KEY="test-secret-key-for-testing-only-needs-32-chars-minimum",
        JWT_ALGORITHM="HS256",
        ACCESS_TOKEN_EXPIRE_MINUTES=30,
        REFRESH_TOKEN_EXPIRE_DAYS=7,
        SESSION_CLEANUP_INTERVAL_HOURS=24,
        SESSION_MAX_AGE_DAYS=30,
        RATE_LIMIT_ENABLED=True,
        RATE_LIMIT_MAX_REQUESTS=100,
        RATE_LIMIT_WINDOW_SECONDS=60,
        PASSWORD_MIN_LENGTH=8,
        PASSWORD_REQUIRE_UPPERCASE=True,
        PASSWORD_REQUIRE_LOWERCASE=True,
        PASSWORD_REQUIRE_DIGIT=True,
        PASSWORD_REQUIRE_SPECIAL=False,
        REGISTRATION_ENABLED=True,
        REGISTRATION_REQUIRE_CODE=False,
        REGISTRATION_CODES=[],
        DEFAULT_USER_ROLE="user",
        DEFAULT_STORAGE_QUOTA_MB=1000,
        EMAIL_VERIFICATION_REQUIRED=False,
        CORS_ORIGINS=["*"],
        API_PREFIX="/api/v1"
    )


@pytest.fixture
def jwt_service(jwt_settings):
    """Create a JWT service instance for testing."""
    return JWTService(settings=jwt_settings)


@pytest.fixture
async def session_manager(mock_db_pool):
    """Create a session manager instance for testing."""
    manager = SessionManager(db_pool=mock_db_pool)
    yield manager
    # Cleanup
    await manager.shutdown()


@pytest.fixture
async def rate_limiter():
    """Create a rate limiter instance for testing."""
    limiter = RateLimiter()
    yield limiter
    # Cleanup
    await limiter.cleanup()


@pytest.fixture
async def registration_service(mock_db_pool, password_service):
    """Create a registration service instance for testing."""
    return RegistrationService(
        db_pool=mock_db_pool,
        password_service=password_service,
        require_registration_code=False
    )


@pytest.fixture
def test_user():
    """Create a test user dictionary."""
    return {
        'id': 1,
        'uuid': 'test-uuid-123',
        'username': 'testuser',
        'email': 'test@example.com',
        'password_hash': '$2b$12$test_hash',  # Mock hash
        'role': 'user',
        'is_active': True,
        'is_verified': True,
        'created_at': datetime.utcnow(),
        'updated_at': datetime.utcnow(),
        'last_login': datetime.utcnow(),
        'storage_quota_mb': 1000,
        'storage_used_mb': 100
    }


@pytest.fixture
def admin_user():
    """Create a test admin user dictionary."""
    return {
        'id': 2,
        'uuid': 'admin-uuid-456',
        'username': 'adminuser',
        'email': 'admin@example.com',
        'password_hash': '$2b$12$admin_hash',  # Mock hash
        'role': 'admin',
        'is_active': True,
        'is_verified': True,
        'created_at': datetime.utcnow(),
        'updated_at': datetime.utcnow(),
        'last_login': datetime.utcnow(),
        'storage_quota_mb': 10000,
        'storage_used_mb': 500
    }


@pytest.fixture
def inactive_user():
    """Create an inactive test user dictionary."""
    return {
        'id': 3,
        'uuid': 'inactive-uuid-789',
        'username': 'inactiveuser',
        'email': 'inactive@example.com',
        'password_hash': '$2b$12$inactive_hash',  # Mock hash
        'role': 'user',
        'is_active': False,
        'is_verified': True,
        'created_at': datetime.utcnow(),
        'updated_at': datetime.utcnow(),
        'last_login': None,
        'storage_quota_mb': 1000,
        'storage_used_mb': 0
    }


@pytest.fixture
def valid_access_token(jwt_service, test_user):
    """Create a valid access token for testing."""
    return jwt_service.create_access_token(
        user_id=test_user['id'],
        username=test_user['username'],
        role=test_user['role']
    )


@pytest.fixture
def valid_refresh_token(jwt_service, test_user):
    """Create a valid refresh token for testing."""
    return jwt_service.create_refresh_token(user_id=test_user['id'])


@pytest.fixture
def expired_access_token(jwt_service, test_user):
    """Create an expired access token for testing."""
    # Temporarily override expiry
    original_expire = jwt_service.settings.ACCESS_TOKEN_EXPIRE_MINUTES
    jwt_service.settings.ACCESS_TOKEN_EXPIRE_MINUTES = -1  # Expired
    token = jwt_service.create_access_token(
        user_id=test_user['id'],
        username=test_user['username'],
        role=test_user['role']
    )
    jwt_service.settings.ACCESS_TOKEN_EXPIRE_MINUTES = original_expire
    return token


@pytest.fixture
def auth_headers(valid_access_token):
    """Create authorization headers with valid token."""
    return {"Authorization": f"Bearer {valid_access_token}"}


@pytest.fixture
def api_key_headers():
    """Create API key headers for single-user mode."""
    return {"X-API-KEY": settings.get("SINGLE_USER_API_KEY", "test-api-key")}