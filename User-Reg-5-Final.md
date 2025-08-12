# User Registration Module Implementation Plan - Final Version

## Overview
Implement a production-ready user registration and management system for tldw_server with:
1. PostgreSQL for multi-user mode, SQLite for single-user mode
2. JWT-based authentication with persistent secret management
3. Proper connection pooling and session cleanup
4. Rate limiting and health monitoring
5. Transactional consistency for all operations
6. Comprehensive error handling and logging

## Core Design Principles
- **Production Ready**: Handle edge cases and failures gracefully
- **Security First**: Secure by default with rate limiting and proper secret management
- **Observable**: Comprehensive logging and health monitoring
- **Performant**: Connection pooling, caching, and async operations
- **Maintainable**: Clear error handling and dependency injection

## Database Architecture

### Database Configuration with Connection Pooling
```python
# app/core/database.py
from contextlib import asynccontextmanager
import asyncpg
import sqlite3
from typing import Optional
import aiosqlite

class DatabasePool:
    """Database connection pool manager"""
    
    def __init__(self, settings: Settings):
        self.settings = settings
        self.pool: Optional[asyncpg.Pool] = None
        
    async def initialize(self):
        """Initialize database connection pool"""
        if self.settings.AUTH_MODE == "multi_user":
            # PostgreSQL with connection pooling
            self.pool = await asyncpg.create_pool(
                self.settings.DATABASE_URL,
                min_size=5,
                max_size=20,
                max_queries=50000,
                max_inactive_connection_lifetime=300,
                command_timeout=60
            )
            
            # Create schema if needed
            async with self.pool.acquire() as conn:
                await conn.execute(open("schema/postgresql.sql").read())
                
                # Set up automatic partition creation
                await self.setup_partition_management(conn)
        else:
            # SQLite for single-user mode
            self.db_path = self.settings.DATABASE_URL.replace("sqlite:///", "")
    
    async def setup_partition_management(self, conn):
        """Set up automatic monthly partition creation"""
        await conn.execute("""
            CREATE OR REPLACE FUNCTION create_monthly_partition()
            RETURNS void AS $$
            DECLARE
                partition_name text;
                start_date date;
                end_date date;
            BEGIN
                start_date := DATE_TRUNC('month', CURRENT_DATE);
                end_date := start_date + INTERVAL '1 month';
                partition_name := 'audit_log_' || TO_CHAR(start_date, 'YYYY_MM');
                
                -- Check if partition exists
                IF NOT EXISTS (
                    SELECT 1 FROM pg_class 
                    WHERE relname = partition_name
                ) THEN
                    EXECUTE format(
                        'CREATE TABLE %I PARTITION OF audit_log 
                         FOR VALUES FROM (%L) TO (%L)',
                        partition_name, start_date, end_date
                    );
                END IF;
            END;
            $$ LANGUAGE plpgsql;
            
            -- Schedule monthly execution
            SELECT cron.schedule(
                'create-audit-partition',
                '0 0 1 * *',  -- First day of each month
                'SELECT create_monthly_partition()'
            );
        """)
    
    @asynccontextmanager
    async def transaction(self):
        """Database transaction context manager"""
        if self.settings.AUTH_MODE == "multi_user":
            async with self.pool.acquire() as conn:
                async with conn.transaction():
                    yield conn
        else:
            # SQLite transaction
            async with aiosqlite.connect(self.db_path) as conn:
                await conn.execute("BEGIN")
                try:
                    yield conn
                    await conn.commit()
                except Exception:
                    await conn.rollback()
                    raise
    
    async def close(self):
        """Close database connections"""
        if self.pool:
            await self.pool.close()

# Dependency injection provider
_db_pool: Optional[DatabasePool] = None

async def get_db_pool() -> DatabasePool:
    """Get database pool instance"""
    global _db_pool
    if not _db_pool:
        _db_pool = DatabasePool(get_settings())
        await _db_pool.initialize()
    return _db_pool

async def get_db():
    """Get database connection for request"""
    pool = await get_db_pool()
    async with pool.transaction() as conn:
        yield conn
```

### Settings with Persistent JWT Secret
```python
# app/core/config.py
from pydantic import BaseSettings, Field, validator
from typing import Literal, Optional
import os
import secrets
from pathlib import Path

class Settings(BaseSettings):
    """Configuration with persistent secret management"""
    
    # Core Settings
    AUTH_MODE: Literal["single_user", "multi_user"] = "single_user"
    DATABASE_URL: str = Field(
        default="sqlite:///./Databases/users.db",
        description="PostgreSQL for multi-user: postgresql://user:pass@localhost/tldw"
    )
    
    # JWT Settings with persistent storage
    JWT_SECRET_KEY: Optional[str] = None
    JWT_SECRET_FILE: str = Field(
        default=".jwt_secret",
        description="File to store JWT secret"
    )
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 30
    REFRESH_TOKEN_EXPIRE_DAYS: int = 7
    
    # Password Settings
    PASSWORD_MIN_LENGTH: int = 10
    ARGON2_MEMORY_COST: int = 32768  # 32MB
    ARGON2_TIME_COST: int = 2
    ARGON2_PARALLELISM: int = 1
    
    # Redis (Optional)
    REDIS_URL: Optional[str] = None
    
    # Security
    ENABLE_REGISTRATION: bool = False
    REQUIRE_REGISTRATION_CODE: bool = True
    MAX_LOGIN_ATTEMPTS: int = 5
    LOCKOUT_DURATION_MINUTES: int = 15
    
    # Rate Limiting
    RATE_LIMIT_ENABLED: bool = True
    RATE_LIMIT_PER_MINUTE: int = 60
    RATE_LIMIT_BURST: int = 10
    
    # Storage
    DEFAULT_STORAGE_QUOTA_MB: int = 5120
    USER_DATA_BASE_PATH: str = "./user_databases"
    
    # Monitoring
    ENABLE_HEALTH_CHECK: bool = True
    ENABLE_METRICS: bool = True
    SESSION_CLEANUP_INTERVAL_HOURS: int = 1
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Load or generate JWT secret
        self._ensure_jwt_secret()
    
    def _ensure_jwt_secret(self):
        """Ensure JWT secret exists and is persistent"""
        secret_file = Path(self.JWT_SECRET_FILE)
        
        if self.JWT_SECRET_KEY:
            # Secret provided via environment
            return
        
        if secret_file.exists():
            # Load existing secret
            self.JWT_SECRET_KEY = secret_file.read_text().strip()
        else:
            # Generate and save new secret
            self.JWT_SECRET_KEY = secrets.token_urlsafe(64)
            secret_file.write_text(self.JWT_SECRET_KEY)
            # Secure file permissions (Unix-like systems)
            if os.name != 'nt':
                os.chmod(secret_file, 0o600)
    
    @validator("JWT_SECRET_KEY")
    def validate_jwt_secret(cls, v):
        if v and len(v) < 32:
            raise ValueError("JWT secret must be at least 32 characters")
        return v
    
    class Config:
        env_file = ".env"
        case_sensitive = False

# Singleton settings instance
_settings: Optional[Settings] = None

def get_settings() -> Settings:
    """Get settings singleton"""
    global _settings
    if not _settings:
        _settings = Settings()
    return _settings
```

### PostgreSQL Schema with Constraints
```sql
-- PostgreSQL schema with proper constraints and indexes
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS "pgcrypto";
CREATE EXTENSION IF NOT EXISTS "pg_cron";  -- For scheduled tasks

-- Users table
CREATE TABLE users (
    id SERIAL PRIMARY KEY,
    uuid UUID DEFAULT uuid_generate_v4() UNIQUE NOT NULL,
    username VARCHAR(50) UNIQUE NOT NULL,
    email VARCHAR(255) UNIQUE NOT NULL,
    password_hash TEXT NOT NULL,
    role VARCHAR(20) NOT NULL DEFAULT 'user' 
        CHECK (role IN ('user', 'admin', 'service')),
    is_active BOOLEAN DEFAULT TRUE,
    is_locked BOOLEAN DEFAULT FALSE,
    locked_until TIMESTAMP,
    failed_login_attempts INT DEFAULT 0,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    last_login TIMESTAMP,
    preferences JSONB DEFAULT '{}',
    storage_quota_mb INT DEFAULT 5120,
    storage_used_mb DECIMAL(10,2) DEFAULT 0.00
);

-- Optimized indexes
CREATE INDEX idx_users_username ON users(username) WHERE is_active = TRUE;
CREATE INDEX idx_users_email ON users(email) WHERE is_active = TRUE;
CREATE INDEX idx_users_role ON users(role) WHERE is_active = TRUE;
CREATE INDEX idx_users_locked ON users(id) WHERE is_locked = TRUE;

-- Sessions table with automatic cleanup
CREATE TABLE sessions (
    id SERIAL PRIMARY KEY,
    user_id INT NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    token_hash VARCHAR(64) NOT NULL UNIQUE,
    refresh_token_hash VARCHAR(64) UNIQUE,
    expires_at TIMESTAMP NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    ip_address INET,
    user_agent TEXT,
    is_active BOOLEAN DEFAULT TRUE
);

CREATE INDEX idx_sessions_token_hash ON sessions(token_hash) WHERE is_active = TRUE;
CREATE INDEX idx_sessions_user_id ON sessions(user_id) WHERE is_active = TRUE;
CREATE INDEX idx_sessions_expires ON sessions(expires_at) WHERE is_active = TRUE;
CREATE INDEX idx_sessions_cleanup ON sessions(expires_at) WHERE expires_at < CURRENT_TIMESTAMP;

-- Registration codes with race condition prevention
CREATE TABLE registration_codes (
    id SERIAL PRIMARY KEY,
    code VARCHAR(32) UNIQUE NOT NULL,
    max_uses INT DEFAULT 1 CHECK (max_uses > 0),
    times_used INT DEFAULT 0 CHECK (times_used >= 0),
    expires_at TIMESTAMP NOT NULL,
    created_by INT REFERENCES users(id),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    role_to_grant VARCHAR(20) DEFAULT 'user',
    -- Prevent race conditions with constraint
    CONSTRAINT usage_limit CHECK (times_used <= max_uses)
);

CREATE INDEX idx_registration_codes_code ON registration_codes(code);
CREATE INDEX idx_registration_codes_active 
    ON registration_codes(code) 
    WHERE times_used < max_uses AND expires_at > CURRENT_TIMESTAMP;

-- Rate limiting table
CREATE TABLE rate_limits (
    id SERIAL PRIMARY KEY,
    identifier VARCHAR(255) NOT NULL,  -- IP or user_id
    endpoint VARCHAR(255) NOT NULL,
    request_count INT DEFAULT 1,
    window_start TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(identifier, endpoint, window_start)
);

CREATE INDEX idx_rate_limits_lookup 
    ON rate_limits(identifier, endpoint, window_start);
CREATE INDEX idx_rate_limits_cleanup 
    ON rate_limits(window_start) 
    WHERE window_start < CURRENT_TIMESTAMP - INTERVAL '1 hour';

-- Audit log with automatic partitioning
CREATE TABLE audit_log (
    id BIGSERIAL,
    user_id INT,
    action VARCHAR(50) NOT NULL,
    details JSONB,
    ip_address INET,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
) PARTITION BY RANGE (created_at);

-- Create first partition
CREATE TABLE audit_log_current PARTITION OF audit_log
    FOR VALUES FROM (DATE_TRUNC('month', CURRENT_DATE))
    TO (DATE_TRUNC('month', CURRENT_DATE) + INTERVAL '1 month');

-- Scheduled cleanup function
CREATE OR REPLACE FUNCTION cleanup_expired_sessions()
RETURNS void AS $$
BEGIN
    DELETE FROM sessions 
    WHERE expires_at < CURRENT_TIMESTAMP - INTERVAL '1 day';
    
    DELETE FROM rate_limits 
    WHERE window_start < CURRENT_TIMESTAMP - INTERVAL '1 hour';
    
    DELETE FROM registration_codes 
    WHERE expires_at < CURRENT_TIMESTAMP - INTERVAL '30 days';
END;
$$ LANGUAGE plpgsql;

-- Schedule cleanup every hour
SELECT cron.schedule(
    'cleanup-expired-data',
    '0 * * * *',  -- Every hour
    'SELECT cleanup_expired_sessions()'
);
```

## Authentication Implementation

### Custom Exceptions
```python
# app/core/exceptions.py
class AuthenticationError(Exception):
    """Base authentication exception"""
    pass

class InvalidCredentialsError(AuthenticationError):
    """Invalid username or password"""
    pass

class AccountLockedException(AuthenticationError):
    """Account is locked due to failed attempts"""
    def __init__(self, locked_until: datetime):
        self.locked_until = locked_until
        super().__init__(f"Account locked until {locked_until}")

class RegistrationError(Exception):
    """Base registration exception"""
    pass

class InvalidRegistrationCodeError(RegistrationError):
    """Invalid or expired registration code"""
    pass

class DuplicateUserError(RegistrationError):
    """Username or email already exists"""
    def __init__(self, field: str):
        self.field = field
        super().__init__(f"{field} already exists")

class QuotaExceededError(Exception):
    """Storage quota exceeded"""
    pass
```

### Logging Configuration
```python
# app/core/logging_config.py
from loguru import logger
import sys
from pathlib import Path

def setup_logging(settings: Settings):
    """Configure application logging"""
    
    # Remove default handler
    logger.remove()
    
    # Console handler with color
    logger.add(
        sys.stdout,
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
        level="INFO" if settings.AUTH_MODE == "multi_user" else "DEBUG",
        colorize=True
    )
    
    # File handler with rotation
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    
    logger.add(
        log_dir / "tldw_{time:YYYY-MM-DD}.log",
        rotation="00:00",  # Daily rotation
        retention="30 days",  # Keep 30 days of logs
        compression="gz",   # Compress old logs
        level="DEBUG",
        format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {name}:{function}:{line} - {message}",
        enqueue=True  # Thread-safe
    )
    
    # Error file
    logger.add(
        log_dir / "errors.log",
        level="ERROR",
        rotation="10 MB",
        retention="90 days",
        format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {name}:{function}:{line} - {message}\n{exception}",
        backtrace=True,
        diagnose=True
    )
    
    # Audit log for security events
    logger.add(
        log_dir / "audit.log",
        filter=lambda record: "audit" in record["extra"],
        rotation="100 MB",
        retention="180 days",
        format="{time:YYYY-MM-DD HH:mm:ss} | AUDIT | {extra[user_id]} | {extra[action]} | {extra[ip]} - {message}"
    )
    
    return logger
```

### Rate Limiting Implementation
```python
# app/core/rate_limiting.py
from fastapi import Request, HTTPException
from datetime import datetime, timedelta
import hashlib
from typing import Optional

class RateLimiter:
    """Token bucket rate limiter with database backend"""
    
    def __init__(self, db_pool: DatabasePool, settings: Settings):
        self.db_pool = db_pool
        self.settings = settings
        self.enabled = settings.RATE_LIMIT_ENABLED
    
    async def check_rate_limit(
        self,
        identifier: str,
        endpoint: str,
        limit: Optional[int] = None,
        window_minutes: int = 1
    ) -> bool:
        """Check if request is within rate limit"""
        
        if not self.enabled:
            return True
        
        limit = limit or self.settings.RATE_LIMIT_PER_MINUTE
        window_start = datetime.utcnow().replace(second=0, microsecond=0)
        
        async with self.db_pool.transaction() as conn:
            # Try to increment counter atomically
            result = await conn.fetchval("""
                INSERT INTO rate_limits (identifier, endpoint, request_count, window_start)
                VALUES ($1, $2, 1, $3)
                ON CONFLICT (identifier, endpoint, window_start) 
                DO UPDATE SET request_count = rate_limits.request_count + 1
                RETURNING request_count
            """, identifier, endpoint, window_start)
            
            if result > limit:
                return False
            
            # Allow burst by checking previous window
            if result > limit - self.settings.RATE_LIMIT_BURST:
                prev_window = window_start - timedelta(minutes=window_minutes)
                prev_count = await conn.fetchval("""
                    SELECT request_count FROM rate_limits
                    WHERE identifier = $1 AND endpoint = $2 AND window_start = $3
                """, identifier, endpoint, prev_window)
                
                if prev_count and prev_count + result > limit + self.settings.RATE_LIMIT_BURST:
                    return False
            
            return True

# Middleware
async def rate_limit_middleware(request: Request, call_next):
    """Apply rate limiting to requests"""
    
    # Skip health checks
    if request.url.path in ["/health", "/metrics"]:
        return await call_next(request)
    
    # Get identifier (IP address or user ID)
    identifier = request.client.host
    if hasattr(request.state, "user_id"):
        identifier = f"user:{request.state.user_id}"
    
    # Check rate limit
    rate_limiter = request.app.state.rate_limiter
    endpoint = f"{request.method}:{request.url.path}"
    
    if not await rate_limiter.check_rate_limit(identifier, endpoint):
        raise HTTPException(
            status_code=429,
            detail="Rate limit exceeded. Please try again later."
        )
    
    return await call_next(request)
```

### User Registration with Transaction Safety
```python
# app/services/registration.py
from typing import Dict, Optional
import secrets
import string
import os
import shutil
from pathlib import Path
import asyncio
from concurrent.futures import ThreadPoolExecutor

class RegistrationService:
    def __init__(
        self,
        db_pool: DatabasePool,
        password_service: PasswordService,
        settings: Settings
    ):
        self.db_pool = db_pool
        self.password_service = password_service
        self.settings = settings
        self.executor = ThreadPoolExecutor(max_workers=2)
    
    def generate_registration_code(self) -> str:
        """Generate secure registration code"""
        alphabet = string.ascii_letters + string.digits
        return ''.join(secrets.choice(alphabet) for _ in range(24))
    
    def _create_user_directories(self, user_id: int) -> bool:
        """Create user directories (runs in thread pool)"""
        try:
            base_path = Path(self.settings.USER_DATA_BASE_PATH)
            user_dir = base_path / str(user_id)
            user_dir.mkdir(parents=True, exist_ok=True)
            (user_dir / "media").mkdir(exist_ok=True)
            (user_dir / "notes").mkdir(exist_ok=True)
            (user_dir / "embeddings").mkdir(exist_ok=True)
            
            # Set permissions on Unix-like systems
            if os.name != 'nt':
                os.chmod(user_dir, 0o750)
            
            return True
        except Exception as e:
            logger.error(f"Failed to create directories for user {user_id}: {e}")
            return False
    
    async def register_user(
        self,
        username: str,
        email: str,
        password: str,
        registration_code: Optional[str] = None
    ) -> Dict:
        """Register user with full transaction safety"""
        
        user_id = None
        directories_created = False
        
        try:
            async with self.db_pool.transaction() as conn:
                # Validate registration code if required
                role = "user"
                if self.settings.REQUIRE_REGISTRATION_CODE:
                    if not registration_code:
                        raise InvalidRegistrationCodeError("Registration code required")
                    
                    # Use SELECT FOR UPDATE to lock the row
                    code_row = await conn.fetchrow("""
                        SELECT id, role_to_grant, times_used, max_uses
                        FROM registration_codes
                        WHERE code = $1
                        AND times_used < max_uses
                        AND expires_at > CURRENT_TIMESTAMP
                        FOR UPDATE
                    """, registration_code)
                    
                    if not code_row:
                        raise InvalidRegistrationCodeError("Invalid registration code")
                    
                    # Update usage count
                    await conn.execute("""
                        UPDATE registration_codes
                        SET times_used = times_used + 1
                        WHERE id = $1
                    """, code_row['id'])
                    
                    role = code_row['role_to_grant']
                
                # Check for duplicate username/email
                existing = await conn.fetchrow("""
                    SELECT username, email
                    FROM users
                    WHERE username = $1 OR email = $2
                """, username, email)
                
                if existing:
                    if existing['username'] == username:
                        raise DuplicateUserError("username")
                    else:
                        raise DuplicateUserError("email")
                
                # Hash password
                password_hash = self.password_service.hash_password(password)
                
                # Create user
                user_id = await conn.fetchval("""
                    INSERT INTO users (username, email, password_hash, role)
                    VALUES ($1, $2, $3, $4)
                    RETURNING id
                """, username, email, password_hash, role)
                
                # Create directories before committing transaction
                loop = asyncio.get_event_loop()
                directories_created = await loop.run_in_executor(
                    self.executor,
                    self._create_user_directories,
                    user_id
                )
                
                if not directories_created:
                    raise Exception("Failed to create user directories")
                
                # Log registration
                await conn.execute("""
                    INSERT INTO audit_log (user_id, action, details)
                    VALUES ($1, 'user_registered', $2)
                """, user_id, {"username": username, "role": role})
                
                logger.info(f"User registered: {username} (ID: {user_id})")
                
                return {
                    "user_id": user_id,
                    "username": username,
                    "role": role
                }
                
        except Exception as e:
            # Clean up directories if they were created
            if user_id and directories_created:
                user_dir = Path(self.settings.USER_DATA_BASE_PATH) / str(user_id)
                shutil.rmtree(user_dir, ignore_errors=True)
            
            # Log error
            logger.error(f"Registration failed for {username}: {e}")
            raise
```

### Session Management with Cleanup
```python
# app/services/session_manager.py
from typing import Optional, Dict
import redis
import json
import hashlib
from datetime import datetime, timedelta
from apscheduler.schedulers.asyncio import AsyncIOScheduler

class SessionManager:
    def __init__(self, db_pool: DatabasePool, settings: Settings):
        self.db_pool = db_pool
        self.settings = settings
        self.redis_client = None
        self.scheduler = AsyncIOScheduler()
        
        # Initialize Redis if available
        if settings.REDIS_URL:
            try:
                self.redis_client = redis.from_url(
                    settings.REDIS_URL,
                    decode_responses=True,
                    socket_connect_timeout=1,
                    max_connections=50
                )
                self.redis_client.ping()
                logger.info("Redis connected for session caching")
            except (redis.ConnectionError, redis.TimeoutError):
                logger.warning("Redis unavailable, using database only")
                self.redis_client = None
        
        # Schedule session cleanup
        self.scheduler.add_job(
            self.cleanup_expired_sessions,
            'interval',
            hours=self.settings.SESSION_CLEANUP_INTERVAL_HOURS,
            id='session_cleanup',
            replace_existing=True
        )
        self.scheduler.start()
    
    async def cleanup_expired_sessions(self):
        """Clean up expired sessions"""
        try:
            async with self.db_pool.transaction() as conn:
                deleted = await conn.fetchval("""
                    DELETE FROM sessions
                    WHERE expires_at < CURRENT_TIMESTAMP
                    RETURNING COUNT(*)
                """)
                
                if deleted:
                    logger.info(f"Cleaned up {deleted} expired sessions")
                    
                # Clear Redis cache for expired sessions
                if self.redis_client:
                    pattern = "session:*"
                    cursor = 0
                    while True:
                        cursor, keys = self.redis_client.scan(
                            cursor, match=pattern, count=100
                        )
                        for key in keys:
                            ttl = self.redis_client.ttl(key)
                            if ttl == -1:  # No expiry set
                                self.redis_client.delete(key)
                        if cursor == 0:
                            break
                            
        except Exception as e:
            logger.error(f"Session cleanup failed: {e}")
    
    def hash_token(self, token: str) -> str:
        """Hash token for storage"""
        return hashlib.sha256(token.encode()).hexdigest()
    
    async def create_session(
        self,
        user_id: int,
        access_token: str,
        refresh_token: str,
        ip_address: str,
        user_agent: str
    ) -> Dict:
        """Create session with both tokens"""
        access_hash = self.hash_token(access_token)
        refresh_hash = self.hash_token(refresh_token)
        expires_at = datetime.utcnow() + timedelta(
            minutes=self.settings.ACCESS_TOKEN_EXPIRE_MINUTES
        )
        
        async with self.db_pool.transaction() as conn:
            session_id = await conn.fetchval("""
                INSERT INTO sessions (
                    user_id, token_hash, refresh_token_hash,
                    expires_at, ip_address, user_agent
                )
                VALUES ($1, $2, $3, $4, $5, $6)
                RETURNING id
            """, user_id, access_hash, refresh_hash, expires_at, 
                ip_address, user_agent)
            
            # Cache in Redis if available
            if self.redis_client:
                try:
                    cache_data = {
                        "user_id": user_id,
                        "session_id": session_id,
                        "expires_at": expires_at.isoformat()
                    }
                    ttl = self.settings.ACCESS_TOKEN_EXPIRE_MINUTES * 60
                    
                    pipe = self.redis_client.pipeline()
                    pipe.setex(f"session:{access_hash}", ttl, json.dumps(cache_data))
                    pipe.setex(f"user:{user_id}:session", ttl, session_id)
                    pipe.execute()
                except redis.RedisError as e:
                    logger.warning(f"Failed to cache session: {e}")
            
            return {"session_id": session_id, "expires_at": expires_at}
    
    async def validate_session(self, token: str) -> Optional[Dict]:
        """Validate session with caching"""
        token_hash = self.hash_token(token)
        
        # Try Redis cache first
        if self.redis_client:
            try:
                cached = self.redis_client.get(f"session:{token_hash}")
                if cached:
                    data = json.loads(cached)
                    expires = datetime.fromisoformat(data['expires_at'])
                    if expires > datetime.utcnow():
                        return data
                    else:
                        self.redis_client.delete(f"session:{token_hash}")
            except redis.RedisError:
                pass
        
        # Database lookup
        async with self.db_pool.pool.acquire() as conn:
            session = await conn.fetchrow("""
                SELECT id, user_id, expires_at
                FROM sessions
                WHERE token_hash = $1
                AND is_active = TRUE
                AND expires_at > CURRENT_TIMESTAMP
            """, token_hash)
            
            if session:
                result = dict(session)
                
                # Update cache
                if self.redis_client:
                    try:
                        ttl = int((session['expires_at'] - datetime.utcnow()).total_seconds())
                        if ttl > 0:
                            self.redis_client.setex(
                                f"session:{token_hash}",
                                ttl,
                                json.dumps({
                                    "user_id": result['user_id'],
                                    "session_id": result['id'],
                                    "expires_at": result['expires_at'].isoformat()
                                })
                            )
                    except redis.RedisError:
                        pass
                
                return result
        
        return None
```

### Health Check Endpoints
```python
# app/api/health.py
from fastapi import APIRouter, Response
from typing import Dict
import psutil
import aioredis

router = APIRouter(tags=["health"])

@router.get("/health")
async def health_check(
    db_pool = Depends(get_db_pool),
    settings = Depends(get_settings)
) -> Dict:
    """Comprehensive health check"""
    
    health = {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "checks": {}
    }
    
    # Database check
    try:
        if settings.AUTH_MODE == "multi_user":
            async with db_pool.pool.acquire() as conn:
                await conn.fetchval("SELECT 1")
        else:
            async with aiosqlite.connect(db_pool.db_path) as conn:
                await conn.execute("SELECT 1")
        health["checks"]["database"] = "ok"
    except Exception as e:
        health["checks"]["database"] = f"error: {str(e)}"
        health["status"] = "unhealthy"
    
    # Redis check
    if settings.REDIS_URL:
        try:
            redis_client = aioredis.from_url(settings.REDIS_URL)
            await redis_client.ping()
            await redis_client.close()
            health["checks"]["redis"] = "ok"
        except Exception as e:
            health["checks"]["redis"] = f"error: {str(e)}"
            # Redis is optional, don't mark as unhealthy
    
    # System resources
    health["checks"]["system"] = {
        "cpu_percent": psutil.cpu_percent(),
        "memory_percent": psutil.virtual_memory().percent,
        "disk_percent": psutil.disk_usage('/').percent
    }
    
    # High resource usage warning
    if health["checks"]["system"]["memory_percent"] > 90:
        health["status"] = "degraded"
        health["warnings"] = ["High memory usage"]
    
    status_code = 200 if health["status"] == "healthy" else 503
    return Response(
        content=json.dumps(health),
        status_code=status_code,
        media_type="application/json"
    )

@router.get("/health/live")
async def liveness_probe() -> Dict:
    """Simple liveness probe for Kubernetes"""
    return {"status": "alive"}

@router.get("/health/ready")
async def readiness_probe(
    db_pool = Depends(get_db_pool),
    settings = Depends(get_settings)
) -> Dict:
    """Readiness probe for Kubernetes"""
    
    try:
        # Check database connection
        if settings.AUTH_MODE == "multi_user":
            async with db_pool.pool.acquire() as conn:
                await conn.fetchval("SELECT 1")
        else:
            async with aiosqlite.connect(db_pool.db_path) as conn:
                await conn.execute("SELECT 1")
        
        return {"status": "ready"}
    except Exception:
        return Response(
            content=json.dumps({"status": "not ready"}),
            status_code=503,
            media_type="application/json"
        )
```

### Storage Quota with Async Operations
```python
# app/services/storage_quota.py
from pathlib import Path
import asyncio
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, Optional
from cachetools import TTLCache

class StorageQuotaService:
    def __init__(self, db_pool: DatabasePool, settings: Settings):
        self.db_pool = db_pool
        self.settings = settings
        self.executor = ThreadPoolExecutor(max_workers=4)
        # TTL cache for quota checks (5 minutes)
        self.quota_cache = TTLCache(maxsize=1000, ttl=300)
    
    async def check_quota(self, user_id: int, new_bytes: int) -> bool:
        """Check if user has quota for new content"""
        
        # Check cache first
        cache_key = f"quota:{user_id}"
        if cache_key in self.quota_cache:
            current_mb, quota_mb = self.quota_cache[cache_key]
        else:
            # Fetch from database
            async with self.db_pool.pool.acquire() as conn:
                row = await conn.fetchrow("""
                    SELECT storage_used_mb, storage_quota_mb
                    FROM users WHERE id = $1
                """, user_id)
                
                if not row:
                    return False
                
                current_mb = float(row['storage_used_mb'])
                quota_mb = row['storage_quota_mb']
                self.quota_cache[cache_key] = (current_mb, quota_mb)
        
        new_mb = new_bytes / (1024 * 1024)
        return (current_mb + new_mb) <= quota_mb
    
    async def update_usage(self, user_id: int, bytes_added: int):
        """Update storage usage atomically"""
        mb_added = bytes_added / (1024 * 1024)
        
        async with self.db_pool.transaction() as conn:
            new_usage = await conn.fetchval("""
                UPDATE users 
                SET storage_used_mb = storage_used_mb + $1
                WHERE id = $2
                RETURNING storage_used_mb
            """, mb_added, user_id)
            
            # Check if over quota
            quota = await conn.fetchval("""
                SELECT storage_quota_mb FROM users WHERE id = $1
            """, user_id)
            
            if new_usage > quota:
                # Rollback the update
                raise QuotaExceededError(
                    f"Storage quota exceeded: {new_usage:.2f}MB / {quota}MB"
                )
            
            # Invalidate cache
            cache_key = f"quota:{user_id}"
            self.quota_cache.pop(cache_key, None)
    
    def _calculate_directory_size(self, path: str) -> int:
        """Calculate directory size (runs in thread pool)"""
        total = 0
        path_obj = Path(path)
        
        if not path_obj.exists():
            return 0
        
        try:
            for entry in path_obj.rglob('*'):
                if entry.is_file():
                    try:
                        total += entry.stat().st_size
                    except OSError:
                        continue
        except OSError as e:
            logger.error(f"Error calculating size for {path}: {e}")
        
        return total
    
    async def calculate_user_storage(self, user_id: int) -> Dict:
        """Calculate actual storage usage asynchronously"""
        user_dir = Path(self.settings.USER_DATA_BASE_PATH) / str(user_id)
        
        # Run in thread pool to avoid blocking
        loop = asyncio.get_event_loop()
        size_bytes = await loop.run_in_executor(
            self.executor,
            self._calculate_directory_size,
            str(user_dir)
        )
        
        size_mb = size_bytes / (1024 * 1024)
        
        # Update database
        async with self.db_pool.transaction() as conn:
            await conn.execute("""
                UPDATE users
                SET storage_used_mb = $1
                WHERE id = $2
            """, size_mb, user_id)
        
        # Invalidate cache
        cache_key = f"quota:{user_id}"
        self.quota_cache.pop(cache_key, None)
        
        return {
            "user_id": user_id,
            "storage_bytes": size_bytes,
            "storage_mb": round(size_mb, 2)
        }
```

### Migration Script with Progress and Validation
```python
# migrate_to_multiuser.py
import asyncio
import asyncpg
import aiosqlite
from pathlib import Path
import shutil
from datetime import datetime
from typing import Optional
import click
from tqdm import tqdm

class SafeMigration:
    def __init__(
        self,
        postgres_url: str,
        backup_dir: str = "migration_backups",
        dry_run: bool = False
    ):
        self.postgres_url = postgres_url
        self.backup_dir = Path(backup_dir) / datetime.now().isoformat()
        self.dry_run = dry_run
    
    async def migrate(self) -> bool:
        """Safe migration with progress tracking"""
        
        steps = [
            ("Validating environment", self.validate_environment),
            ("Creating backup", self.create_backup),
            ("Testing PostgreSQL", self.test_postgres),
            ("Creating schema", self.create_postgres_schema),
            ("Creating admin user", self.create_admin_user),
            ("Migrating data", self.migrate_existing_data),
            ("Updating configuration", self.update_configuration),
            ("Verifying migration", self.verify_migration)
        ]
        
        if self.dry_run:
            print("🔍 DRY RUN MODE - No changes will be made")
        
        with tqdm(total=len(steps), desc="Migration Progress") as pbar:
            for step_name, step_func in steps:
                pbar.set_description(f"⚙️  {step_name}")
                
                try:
                    result = await step_func()
                    if not result:
                        print(f"\n❌ {step_name} failed")
                        if not self.dry_run:
                            await self.rollback()
                        return False
                    
                    pbar.update(1)
                    print(f"✅ {step_name} completed")
                    
                except Exception as e:
                    print(f"\n❌ {step_name} failed: {e}")
                    if not self.dry_run:
                        await self.rollback()
                    return False
        
        if self.dry_run:
            print("\n✅ Dry run completed successfully")
        else:
            print("\n✅ Migration completed successfully")
            print(f"📁 Backup location: {self.backup_dir}")
            
            # Save admin credentials securely
            creds_file = self.backup_dir / "admin_credentials.txt"
            creds_file.write_text(
                f"Username: admin\n"
                f"Password: {self.admin_password}\n"
                f"Created: {datetime.now().isoformat()}\n"
            )
            creds_file.chmod(0o600)
            print(f"🔐 Admin credentials saved to: {creds_file}")
        
        return True
    
    async def validate_environment(self) -> bool:
        """Validate migration prerequisites"""
        checks = []
        
        # Check disk space (need at least 2x current data)
        data_size = sum(
            f.stat().st_size for f in Path(".").rglob("*") if f.is_file()
        )
        free_space = shutil.disk_usage(".").free
        checks.append(("Disk space", free_space > data_size * 2))
        
        # Check if PostgreSQL client is available
        try:
            import asyncpg
            checks.append(("PostgreSQL client", True))
        except ImportError:
            checks.append(("PostgreSQL client", False))
        
        # Check current data integrity
        if Path("Databases/Media_DB_v2.db").exists():
            try:
                async with aiosqlite.connect("Databases/Media_DB_v2.db") as conn:
                    await conn.execute("PRAGMA integrity_check")
                checks.append(("Database integrity", True))
            except Exception:
                checks.append(("Database integrity", False))
        
        # Print validation results
        print("\nEnvironment Validation:")
        for check_name, passed in checks:
            status = "✅" if passed else "❌"
            print(f"  {status} {check_name}")
        
        return all(passed for _, passed in checks)
    
    async def create_backup(self) -> bool:
        """Create comprehensive backup with verification"""
        if self.dry_run:
            print("  Would create backup at:", self.backup_dir)
            return True
        
        self.backup_dir.mkdir(parents=True, exist_ok=True)
        
        # Calculate total size for progress bar
        backup_items = [
            ("Databases", Path("Databases")),
            ("User Data", Path("user_databases")),
            ("Configuration", Path("."))
        ]
        
        for name, source in backup_items:
            if source.is_dir():
                dest = self.backup_dir / source.name
                print(f"  Backing up {name}...")
                shutil.copytree(source, dest, dirs_exist_ok=True)
            elif source.is_file():
                shutil.copy2(source, self.backup_dir)
        
        # Verify backup
        backup_size = sum(
            f.stat().st_size for f in self.backup_dir.rglob("*") if f.is_file()
        )
        print(f"  Backup size: {backup_size / (1024**2):.2f} MB")
        
        return True
    
    async def rollback(self) -> bool:
        """Rollback migration with verification"""
        print("\n⏮️  Rolling back migration...")
        
        if not self.backup_dir.exists():
            print("❌ No backup found")
            return False
        
        # Restore each component
        for item in self.backup_dir.iterdir():
            if item.is_dir():
                dest = Path(item.name)
                if dest.exists():
                    shutil.rmtree(dest)
                shutil.copytree(item, dest)
                print(f"  Restored {item.name}")
            elif item.is_file() and item.name in [".env", "config.txt"]:
                shutil.copy2(item, item.name)
                print(f"  Restored {item.name}")
        
        print("✅ Rollback completed")
        return True

# CLI Interface
@click.command()
@click.option('--postgres-url', required=True, help='PostgreSQL connection URL')
@click.option('--backup-dir', default='migration_backups', help='Backup directory')
@click.option('--dry-run', is_flag=True, help='Perform dry run without changes')
def migrate_command(postgres_url: str, backup_dir: str, dry_run: bool):
    """Migrate tldw_server to multi-user mode"""
    
    migration = SafeMigration(postgres_url, backup_dir, dry_run)
    
    # Run migration
    success = asyncio.run(migration.migrate())
    
    if success:
        click.echo("\n✅ Migration successful!")
    else:
        click.echo("\n❌ Migration failed!")
        raise click.Abort()

if __name__ == "__main__":
    migrate_command()
```

## Main Application Setup
```python
# app/main.py
from fastapi import FastAPI, Request
from contextlib import asynccontextmanager
import uvicorn

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager"""
    
    # Startup
    settings = get_settings()
    
    # Setup logging
    logger = setup_logging(settings)
    logger.info("Starting tldw_server...")
    
    # Initialize database pool
    db_pool = await get_db_pool()
    app.state.db_pool = db_pool
    
    # Initialize services
    app.state.session_manager = SessionManager(db_pool, settings)
    app.state.rate_limiter = RateLimiter(db_pool, settings)
    app.state.storage_service = StorageQuotaService(db_pool, settings)
    
    logger.info("Services initialized")
    
    yield
    
    # Shutdown
    logger.info("Shutting down tldw_server...")
    
    # Close database connections
    await db_pool.close()
    
    # Stop schedulers
    app.state.session_manager.scheduler.shutdown()
    
    logger.info("Shutdown complete")

# Create FastAPI app
app = FastAPI(
    title="tldw_server",
    version="0.2.0",
    lifespan=lifespan
)

# Add middleware
app.add_middleware(rate_limit_middleware)

# Include routers
from app.api import auth, users, health

app.include_router(health.router)
app.include_router(auth.router)
app.include_router(users.router)

if __name__ == "__main__":
    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_config=None  # Use loguru instead
    )
```

## Testing Strategy (Comprehensive)

### Performance Tests
```python
# tests/test_performance.py
import pytest
import asyncio
from concurrent.futures import ThreadPoolExecutor

class TestPerformance:
    @pytest.mark.asyncio
    async def test_concurrent_logins(self, client, test_users):
        """Test system handles concurrent logins"""
        
        async def login(user):
            response = await client.post("/api/v1/auth/login", json={
                "username": user['username'],
                "password": user['password']
            })
            return response.status_code == 200
        
        # Test 100 concurrent logins
        tasks = [login(user) for user in test_users[:100]]
        results = await asyncio.gather(*tasks)
        
        success_rate = sum(results) / len(results)
        assert success_rate > 0.95  # 95% success rate
    
    @pytest.mark.asyncio
    async def test_registration_race_condition(self, client, db_pool):
        """Test registration code race condition prevention"""
        
        # Create a registration code with 1 use
        code = "TEST_SINGLE_USE_CODE"
        async with db_pool.transaction() as conn:
            await conn.execute("""
                INSERT INTO registration_codes 
                (code, max_uses, expires_at)
                VALUES ($1, 1, CURRENT_TIMESTAMP + INTERVAL '1 hour')
            """, code)
        
        # Try to register 10 users concurrently with same code
        async def register(i):
            response = await client.post("/api/v1/auth/register", json={
                "username": f"raceuser{i}",
                "email": f"race{i}@test.com",
                "password": "TestPass123!",
                "registration_code": code
            })
            return response.status_code == 201
        
        tasks = [register(i) for i in range(10)]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Only one should succeed
        successes = sum(1 for r in results if r is True)
        assert successes == 1
```

## Deployment Configuration

### Docker Compose (Development)
```yaml
# docker-compose.yml
version: '3.8'

services:
  postgres:
    image: postgres:15-alpine
    environment:
      POSTGRES_USER: tldw
      POSTGRES_PASSWORD: ${DB_PASSWORD}
      POSTGRES_DB: tldw_db
    volumes:
      - postgres_data:/var/lib/postgresql/data
    ports:
      - "5432:5432"
    healthcheck:
      test: ["CMD-SHELL", "pg_isready -U tldw"]
      interval: 10s
      timeout: 5s
      retries: 5

  redis:
    image: redis:7-alpine
    command: redis-server --appendonly yes
    volumes:
      - redis_data:/data
    ports:
      - "6379:6379"
    healthcheck:
      test: ["CMD", "redis-cli", "ping"]
      interval: 10s
      timeout: 5s
      retries: 5

  app:
    build: .
    environment:
      AUTH_MODE: multi_user
      DATABASE_URL: postgresql://tldw:${DB_PASSWORD}@postgres/tldw_db
      REDIS_URL: redis://redis:6379/0
      JWT_SECRET_KEY: ${JWT_SECRET_KEY}
    volumes:
      - ./user_databases:/app/user_databases
      - ./logs:/app/logs
    ports:
      - "8000:8000"
    depends_on:
      postgres:
        condition: service_healthy
      redis:
        condition: service_healthy

volumes:
  postgres_data:
  redis_data:
```

## Summary of Improvements

### Must Fix (All Addressed):
1. ✅ **JWT secret persistence** - Stored in file with proper permissions
2. ✅ **Directory creation transaction** - Created before commit with rollback
3. ✅ **Database connection pooling** - Proper asyncpg pool configuration
4. ✅ **Session cleanup** - Scheduled cleanup with Redis cache invalidation
5. ✅ **Registration code race condition** - Row-level locking with SELECT FOR UPDATE
6. ✅ **Rate limiting** - Database-backed token bucket implementation

### Should Fix (All Addressed):
1. ✅ **Health check endpoints** - Comprehensive health, liveness, and readiness probes
2. ✅ **Proper logging** - Loguru with rotation, compression, and audit logs
3. ✅ **Partition management** - Automatic monthly partition creation
4. ✅ **Dependency injection** - Proper providers and singletons
5. ✅ **Transaction context manager** - Implemented for both PostgreSQL and SQLite
6. ✅ **Connection pooling config** - Comprehensive pool settings

### Nice to Have (Implemented):
1. ✅ **Custom exceptions** - Specific exception types for better error handling
2. ✅ **Async file operations** - Thread pool executor for directory calculations
3. ✅ **User lookup caching** - Redis and in-memory TTL caching
4. ✅ **Migration progress** - tqdm progress bars and status updates
5. ✅ **Dry-run mode** - Full dry-run support for migration

## Implementation Timeline

### Week 1: Foundation
- Set up PostgreSQL and connection pooling
- Implement JWT with persistent secrets
- Create custom exceptions and logging
- Build health check endpoints
- Add rate limiting implementation

### Week 2: Core Features
- Implement user registration with transaction safety
- Add session management with cleanup
- Create authentication endpoints
- Build storage quota service
- Add audit logging

### Week 3: Integration
- Integrate with existing media system
- Implement user data isolation
- Add caching layers
- Create migration script with dry-run
- Test backward compatibility

### Week 4: Production Ready
- Performance testing and optimization
- Security audit and penetration testing
- Documentation and deployment guides
- Monitoring and alerting setup
- Production deployment

This final plan provides a production-ready, secure, and maintainable user registration system that addresses all identified issues while maintaining simplicity and performance.