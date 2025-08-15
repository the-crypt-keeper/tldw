"""
Shared fixtures and configuration for AuthNZ tests.
Provides PostgreSQL test database isolation with transaction rollback.
"""

import os
import pytest
import asyncio
import uuid
import tempfile
import shutil
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, Any, AsyncGenerator, Optional
from unittest.mock import Mock, AsyncMock, MagicMock

import asyncpg
from loguru import logger

from tldw_Server_API.app.core.AuthNZ.database import DatabasePool
from tldw_Server_API.app.core.AuthNZ.password_service import PasswordService
from tldw_Server_API.app.core.AuthNZ.jwt_service import JWTService
from tldw_Server_API.app.core.AuthNZ.settings import Settings
from tldw_Server_API.app.core.AuthNZ.session_manager import SessionManager
from tldw_Server_API.app.core.AuthNZ.rate_limiter import RateLimiter
from tldw_Server_API.app.services.registration_service import RegistrationService
from tldw_Server_API.app.services.audit_service import AuditService
from tldw_Server_API.app.services.storage_quota_service import StorageQuotaService
from tldw_Server_API.app.core.config import settings

# Test database configuration
TEST_DB_NAME = "tldw_test"
TEST_DB_HOST = os.getenv("TEST_DB_HOST", "localhost")
TEST_DB_PORT = int(os.getenv("TEST_DB_PORT", "5432"))
TEST_DB_USER = os.getenv("TEST_DB_USER", "tldw_user")
TEST_DB_PASSWORD = os.getenv("TEST_DB_PASSWORD", "TestPassword123!")

# Import TestClient for isolated environment
from fastapi.testclient import TestClient
from tldw_Server_API.app.main import app


@pytest.fixture(scope="session")
def event_loop():
    """Create an instance of the default event loop for the test session."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


@pytest.fixture(autouse=True)
async def reset_singletons():
    """Auto-reset all singletons before and after each test for clean state."""
    # Reset before test
    from tldw_Server_API.app.core.AuthNZ.database import reset_db_pool
    from tldw_Server_API.app.core.AuthNZ.session_manager import reset_session_manager
    from tldw_Server_API.app.core.AuthNZ.settings import reset_settings
    
    await reset_db_pool()
    await reset_session_manager()
    reset_settings()
    
    # Clear any FastAPI dependency overrides
    app.dependency_overrides.clear()
    
    yield
    
    # Reset after test
    await reset_db_pool()
    await reset_session_manager()
    reset_settings()
    app.dependency_overrides.clear()


@pytest.fixture
async def isolated_test_environment(monkeypatch):
    """Create isolated DB and app instance for each test - TRUE ONE DB PER TEST."""
    import uuid as uuid_lib
    
    # 1. Generate unique DB name for this test
    db_name = f"tldw_test_{uuid_lib.uuid4().hex[:8]}"
    logger.info(f"Creating isolated test database: {db_name}")
    
    # 2. Create the unique database
    conn = await asyncpg.connect(
        host=TEST_DB_HOST,
        port=TEST_DB_PORT,
        user=TEST_DB_USER,
        password=TEST_DB_PASSWORD,
        database="postgres"
    )
    
    try:
        # Drop if exists (cleanup from failed tests)
        await conn.execute(f"""
            SELECT pg_terminate_backend(pid) 
            FROM pg_stat_activity 
            WHERE datname = '{db_name}' AND pid <> pg_backend_pid()
        """)
        await conn.execute(f"DROP DATABASE IF EXISTS {db_name}")
        
        # Create new database
        await conn.execute(f"CREATE DATABASE {db_name}")
        logger.info(f"Created test database: {db_name}")
    finally:
        await conn.close()
    
    # 3. Create schema in the new database
    test_conn = await asyncpg.connect(
        host=TEST_DB_HOST,
        port=TEST_DB_PORT,
        user=TEST_DB_USER,
        password=TEST_DB_PASSWORD,
        database=db_name
    )
    
    try:
        # Create all required tables
        await test_conn.execute("""
            CREATE TABLE IF NOT EXISTS users (
                id SERIAL PRIMARY KEY,
                uuid UUID UNIQUE NOT NULL,
                username VARCHAR(255) UNIQUE NOT NULL,
                email VARCHAR(255) UNIQUE NOT NULL,
                password_hash TEXT NOT NULL,
                role VARCHAR(50) NOT NULL DEFAULT 'user',
                is_active BOOLEAN DEFAULT TRUE,
                is_verified BOOLEAN DEFAULT FALSE,
                storage_quota_mb INTEGER DEFAULT 5120,
                storage_used_mb FLOAT DEFAULT 0.0,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                last_login TIMESTAMP,
                email_verified_at TIMESTAMP,
                two_factor_enabled BOOLEAN DEFAULT FALSE,
                two_factor_secret TEXT,
                created_by INTEGER REFERENCES users(id)
            )
        """)
        
        await test_conn.execute("""
            CREATE TABLE IF NOT EXISTS sessions (
                id SERIAL PRIMARY KEY,
                user_id INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
                token_hash TEXT UNIQUE NOT NULL,
                refresh_token_hash TEXT UNIQUE,
                expires_at TIMESTAMP NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                last_activity TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                ip_address VARCHAR(45),
                user_agent TEXT,
                device_id VARCHAR(255),
                is_active BOOLEAN DEFAULT TRUE,
                is_revoked BOOLEAN DEFAULT FALSE,
                revoked_at TIMESTAMP,
                revoked_by INTEGER REFERENCES users(id),
                revoke_reason TEXT
            )
        """)
        
        await test_conn.execute("""
            CREATE TABLE IF NOT EXISTS password_history (
                id SERIAL PRIMARY KEY,
                user_id INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
                password_hash TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        await test_conn.execute("""
            CREATE TABLE IF NOT EXISTS registration_codes (
                id SERIAL PRIMARY KEY,
                code VARCHAR(255) UNIQUE NOT NULL,
                max_uses INTEGER DEFAULT 1,
                uses_count INTEGER DEFAULT 0,
                expires_at TIMESTAMP,
                created_by INTEGER REFERENCES users(id),
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                role_to_grant VARCHAR(50) DEFAULT 'user',
                is_active BOOLEAN DEFAULT TRUE
            )
        """)
        
        await test_conn.execute("""
            CREATE TABLE IF NOT EXISTS audit_log (
                id SERIAL PRIMARY KEY,
                user_id INTEGER REFERENCES users(id) ON DELETE SET NULL,
                action VARCHAR(255) NOT NULL,
                details JSONB,
                ip_address VARCHAR(45),
                user_agent TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # Create indexes
        await test_conn.execute("CREATE INDEX IF NOT EXISTS idx_users_email ON users(email)")
        await test_conn.execute("CREATE INDEX IF NOT EXISTS idx_users_username ON users(username)")
        await test_conn.execute("CREATE INDEX IF NOT EXISTS idx_sessions_user_id ON sessions(user_id)")
        await test_conn.execute("CREATE INDEX IF NOT EXISTS idx_sessions_token_hash ON sessions(token_hash)")
        await test_conn.execute("CREATE INDEX IF NOT EXISTS idx_audit_log_user_id ON audit_log(user_id)")
        
        logger.info(f"Created schema in test database: {db_name}")
    finally:
        await test_conn.close()
    
    # 4. Set environment variables for this test
    db_url = f"postgresql://{TEST_DB_USER}:{TEST_DB_PASSWORD}@{TEST_DB_HOST}:{TEST_DB_PORT}/{db_name}"
    monkeypatch.setenv("AUTH_MODE", "multi_user")
    monkeypatch.setenv("DATABASE_URL", db_url)
    monkeypatch.setenv("JWT_SECRET_KEY", "test-secret-key-for-testing-only")
    monkeypatch.setenv("ENABLE_REGISTRATION", "true")
    monkeypatch.setenv("REQUIRE_REGISTRATION_CODE", "false")
    monkeypatch.setenv("RATE_LIMIT_ENABLED", "false")
    
    # 5. Reset ALL singletons to force fresh initialization with new DB
    from tldw_Server_API.app.core.AuthNZ.database import reset_db_pool
    from tldw_Server_API.app.core.AuthNZ.session_manager import reset_session_manager
    from tldw_Server_API.app.core.AuthNZ.settings import reset_settings
    
    await reset_db_pool()
    await reset_session_manager()
    reset_settings()
    
    # 6. Clear app dependency overrides
    app.dependency_overrides.clear()
    
    # 7. Create TestClient (DB exists, singletons reset, env vars set)
    with TestClient(app) as client:
        yield client, db_name
    
    # 8. Cleanup: reset singletons again
    await reset_db_pool()
    await reset_session_manager()
    reset_settings()
    
    # 9. Drop the unique database
    cleanup_conn = await asyncpg.connect(
        host=TEST_DB_HOST,
        port=TEST_DB_PORT,
        user=TEST_DB_USER,
        password=TEST_DB_PASSWORD,
        database="postgres"
    )
    
    try:
        await cleanup_conn.execute(f"""
            SELECT pg_terminate_backend(pid) 
            FROM pg_stat_activity 
            WHERE datname = '{db_name}' AND pid <> pg_backend_pid()
        """)
        await cleanup_conn.execute(f"DROP DATABASE IF EXISTS {db_name}")
        logger.info(f"Dropped test database: {db_name}")
    finally:
        await cleanup_conn.close()


@pytest.fixture(scope="session")
async def setup_test_database():
    """Create and setup the test database for the test session."""
    # Connect to postgres database to create test database
    conn = await asyncpg.connect(
        host=TEST_DB_HOST,
        port=TEST_DB_PORT,
        user=TEST_DB_USER,
        password=TEST_DB_PASSWORD,
        database="postgres"
    )
    
    try:
        # Drop test database if it exists
        await conn.execute(f"""
            SELECT pg_terminate_backend(pid) 
            FROM pg_stat_activity 
            WHERE datname = '{TEST_DB_NAME}' AND pid <> pg_backend_pid()
        """)
        await conn.execute(f"DROP DATABASE IF EXISTS {TEST_DB_NAME}")
        
        # Create test database
        await conn.execute(f"CREATE DATABASE {TEST_DB_NAME}")
        logger.info(f"Created test database: {TEST_DB_NAME}")
        
    finally:
        await conn.close()
    
    # Connect to test database and create schema
    test_conn = await asyncpg.connect(
        host=TEST_DB_HOST,
        port=TEST_DB_PORT,
        user=TEST_DB_USER,
        password=TEST_DB_PASSWORD,
        database=TEST_DB_NAME
    )
    
    try:
        # Create tables
        await test_conn.execute("""
            CREATE TABLE IF NOT EXISTS users (
                id SERIAL PRIMARY KEY,
                uuid UUID UNIQUE NOT NULL,
                username VARCHAR(255) UNIQUE NOT NULL,
                email VARCHAR(255) UNIQUE NOT NULL,
                password_hash TEXT NOT NULL,
                role VARCHAR(50) NOT NULL DEFAULT 'user',
                is_active BOOLEAN DEFAULT TRUE,
                is_verified BOOLEAN DEFAULT FALSE,
                storage_quota_mb INTEGER DEFAULT 5120,
                storage_used_mb FLOAT DEFAULT 0.0,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                last_login TIMESTAMP,
                email_verified_at TIMESTAMP,
                two_factor_enabled BOOLEAN DEFAULT FALSE,
                two_factor_secret TEXT,
                created_by INTEGER REFERENCES users(id)
            )
        """)
        
        await test_conn.execute("""
            CREATE TABLE IF NOT EXISTS sessions (
                id SERIAL PRIMARY KEY,
                user_id INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
                token_hash TEXT UNIQUE NOT NULL,
                refresh_token_hash TEXT UNIQUE,
                expires_at TIMESTAMP NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                last_activity TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                ip_address VARCHAR(45),
                user_agent TEXT,
                device_id VARCHAR(255),
                is_active BOOLEAN DEFAULT TRUE,
                is_revoked BOOLEAN DEFAULT FALSE,
                revoked_at TIMESTAMP,
                revoked_by INTEGER REFERENCES users(id),
                revoke_reason TEXT
            )
        """)
        
        await test_conn.execute("""
            CREATE TABLE IF NOT EXISTS password_history (
                id SERIAL PRIMARY KEY,
                user_id INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
                password_hash TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        await test_conn.execute("""
            CREATE TABLE IF NOT EXISTS registration_codes (
                id SERIAL PRIMARY KEY,
                code VARCHAR(255) UNIQUE NOT NULL,
                max_uses INTEGER DEFAULT 1,
                uses_count INTEGER DEFAULT 0,
                expires_at TIMESTAMP,
                created_by INTEGER REFERENCES users(id),
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                role_to_grant VARCHAR(50) DEFAULT 'user',
                is_active BOOLEAN DEFAULT TRUE
            )
        """)
        
        await test_conn.execute("""
            CREATE TABLE IF NOT EXISTS audit_log (
                id SERIAL PRIMARY KEY,
                user_id INTEGER REFERENCES users(id) ON DELETE SET NULL,
                action VARCHAR(255) NOT NULL,
                details JSONB,
                ip_address VARCHAR(45),
                user_agent TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # Create indexes
        await test_conn.execute("CREATE INDEX IF NOT EXISTS idx_users_email ON users(email)")
        await test_conn.execute("CREATE INDEX IF NOT EXISTS idx_users_username ON users(username)")
        await test_conn.execute("CREATE INDEX IF NOT EXISTS idx_sessions_user_id ON sessions(user_id)")
        await test_conn.execute("CREATE INDEX IF NOT EXISTS idx_sessions_token_hash ON sessions(token_hash)")
        await test_conn.execute("CREATE INDEX IF NOT EXISTS idx_audit_log_user_id ON audit_log(user_id)")
        
        logger.info("Created test database schema")
        
    finally:
        await test_conn.close()
    
    yield
    
    # Cleanup: Drop test database after all tests
    cleanup_conn = await asyncpg.connect(
        host=TEST_DB_HOST,
        port=TEST_DB_PORT,
        user=TEST_DB_USER,
        password=TEST_DB_PASSWORD,
        database="postgres"
    )
    
    try:
        await cleanup_conn.execute(f"""
            SELECT pg_terminate_backend(pid) 
            FROM pg_stat_activity 
            WHERE datname = '{TEST_DB_NAME}' AND pid <> pg_backend_pid()
        """)
        await cleanup_conn.execute(f"DROP DATABASE IF EXISTS {TEST_DB_NAME}")
        logger.info(f"Dropped test database: {TEST_DB_NAME}")
    finally:
        await cleanup_conn.close()


@pytest.fixture(scope="function")
async def clean_database(setup_test_database):
    """Ensure database is clean before each test."""
    conn = await asyncpg.connect(
        host=TEST_DB_HOST,
        port=TEST_DB_PORT,
        user=TEST_DB_USER,
        password=TEST_DB_PASSWORD,
        database=TEST_DB_NAME
    )
    
    try:
        # Clean all tables in correct order (respecting foreign keys)
        await conn.execute("TRUNCATE TABLE audit_log CASCADE")
        await conn.execute("TRUNCATE TABLE registration_codes CASCADE")
        await conn.execute("TRUNCATE TABLE password_history CASCADE")
        await conn.execute("TRUNCATE TABLE sessions CASCADE")
        await conn.execute("TRUNCATE TABLE users CASCADE")
        
        logger.debug("Cleaned test database tables")
    finally:
        await conn.close()
    
    yield
    
    # Clean up after test
    cleanup_conn = await asyncpg.connect(
        host=TEST_DB_HOST,
        port=TEST_DB_PORT,
        user=TEST_DB_USER,
        password=TEST_DB_PASSWORD,
        database=TEST_DB_NAME
    )
    
    try:
        await cleanup_conn.execute("TRUNCATE TABLE audit_log CASCADE")
        await cleanup_conn.execute("TRUNCATE TABLE registration_codes CASCADE")
        await cleanup_conn.execute("TRUNCATE TABLE password_history CASCADE")
        await cleanup_conn.execute("TRUNCATE TABLE sessions CASCADE")
        await cleanup_conn.execute("TRUNCATE TABLE users CASCADE")
    finally:
        await cleanup_conn.close()


@pytest.fixture
async def test_db_pool(setup_test_database, clean_database):
    """Create a test database pool connected to the test PostgreSQL database."""
    test_database_url = f"postgresql://{TEST_DB_USER}:{TEST_DB_PASSWORD}@{TEST_DB_HOST}:{TEST_DB_PORT}/{TEST_DB_NAME}"
    
    test_settings = Settings(
        AUTH_MODE="multi_user",
        DATABASE_URL=test_database_url,
        JWT_SECRET_KEY="test-secret-key-for-testing-only",
        ENABLE_REGISTRATION=True,
        REQUIRE_REGISTRATION_CODE=False,
        RATE_LIMIT_ENABLED=False
    )
    
    pool = DatabasePool(test_settings)
    await pool.initialize()
    
    try:
        yield pool
    finally:
        await pool.close()


@pytest.fixture
async def mock_db_pool():
    """Create a mock database pool for unit testing."""
    pool = AsyncMock(spec=DatabasePool)
    
    # Mock connection context manager
    mock_conn = AsyncMock()
    pool.transaction.return_value.__aenter__.return_value = mock_conn
    pool.transaction.return_value.__aexit__.return_value = None
    
    # Mock fetchone for user queries
    pool.fetchone = AsyncMock()
    pool.fetchrow = AsyncMock()
    pool.fetch = AsyncMock()
    pool.fetchval = AsyncMock()
    
    # Mock execute for updates
    pool.execute = AsyncMock()
    pool.acquire = AsyncMock()
    pool.release = AsyncMock()
    
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
async def session_manager(test_db_pool):
    """Create a session manager instance for testing."""
    manager = SessionManager(db_pool=test_db_pool)
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
async def registration_service(test_db_pool, password_service):
    """Create a registration service instance for testing."""
    return RegistrationService(
        db_pool=test_db_pool,
        password_service=password_service
    )


@pytest.fixture
async def audit_service(test_db_pool):
    """Create an audit service instance for testing."""
    return AuditService(test_db_pool)


@pytest.fixture
async def storage_service(test_db_pool):
    """Create a storage quota service instance for testing."""
    return StorageQuotaService(test_db_pool)


@pytest.fixture
async def test_user(test_db_pool, password_service):
    """Create a test user in the database."""
    user_uuid = str(uuid.uuid4())
    password = "Test@Pass#2024!"
    password_hash = password_service.hash_password(password)
    
    async with test_db_pool.acquire() as conn:
        user = await conn.fetchrow("""
            INSERT INTO users (
                uuid, username, email, password_hash, role,
                is_active, is_verified, storage_quota_mb, storage_used_mb
            )
            VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9)
            RETURNING id, uuid, username, email, role, is_active, is_verified,
                      storage_quota_mb, storage_used_mb, created_at
        """, user_uuid, "testuser", "test@example.com", password_hash,
            "user", True, True, 5120, 0.0)
    
    return {
        "id": user["id"],
        "uuid": str(user["uuid"]),
        "username": user["username"],
        "email": user["email"],
        "role": user["role"],
        "is_active": user["is_active"],
        "is_verified": user["is_verified"],
        "storage_quota_mb": user["storage_quota_mb"],
        "storage_used_mb": user["storage_used_mb"],
        "created_at": user["created_at"],
        "password": password,
        "password_hash": password_hash
    }


@pytest.fixture
async def admin_user(test_db_pool, password_service):
    """Create an admin test user in the database."""
    user_uuid = str(uuid.uuid4())
    password = "Admin@Pass#2024!"
    password_hash = password_service.hash_password(password)
    
    async with test_db_pool.acquire() as conn:
        user = await conn.fetchrow("""
            INSERT INTO users (
                uuid, username, email, password_hash, role,
                is_active, is_verified, storage_quota_mb, storage_used_mb
            )
            VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9)
            RETURNING id, uuid, username, email, role, is_active, is_verified,
                      storage_quota_mb, storage_used_mb, created_at
        """, user_uuid, "admin", "admin@example.com", password_hash,
            "admin", True, True, 10240, 0.0)
    
    return {
        "id": user["id"],
        "uuid": str(user["uuid"]),
        "username": user["username"],
        "email": user["email"],
        "role": user["role"],
        "is_active": user["is_active"],
        "is_verified": user["is_verified"],
        "storage_quota_mb": user["storage_quota_mb"],
        "storage_used_mb": user["storage_used_mb"],
        "created_at": user["created_at"],
        "password": password,
        "password_hash": password_hash
    }


@pytest.fixture
async def inactive_user(test_db_pool, password_service):
    """Create an inactive test user in the database."""
    user_uuid = str(uuid.uuid4())
    password = "Inactive@Pass#2024!"
    password_hash = password_service.hash_password(password)
    
    async with test_db_pool.acquire() as conn:
        user = await conn.fetchrow("""
            INSERT INTO users (
                uuid, username, email, password_hash, role,
                is_active, is_verified, storage_quota_mb, storage_used_mb
            )
            VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9)
            RETURNING id, uuid, username, email, role, is_active, is_verified,
                      storage_quota_mb, storage_used_mb, created_at
        """, user_uuid, "inactiveuser", "inactive@example.com", password_hash,
            "user", False, True, 5120, 0.0)
    
    return {
        "id": user["id"],
        "uuid": str(user["uuid"]),
        "username": user["username"],
        "email": user["email"],
        "role": user["role"],
        "is_active": user["is_active"],
        "is_verified": user["is_verified"],
        "storage_quota_mb": user["storage_quota_mb"],
        "storage_used_mb": user["storage_used_mb"],
        "created_at": user["created_at"],
        "password": password,
        "password_hash": password_hash
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
    return jwt_service.create_refresh_token(
        user_id=test_user['id'],
        username=test_user['username']
    )


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


@pytest.fixture(autouse=True)
def clear_app_overrides():
    """Clear FastAPI app dependency overrides after each test."""
    yield
    from tldw_Server_API.app.main import app
    app.dependency_overrides.clear()