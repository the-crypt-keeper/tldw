# settings.py
# Description: Pydantic settings for user registration system with persistent JWT secret management
#
# Imports
import os
import secrets
from pathlib import Path
from typing import Literal, Optional
#
# 3rd-party imports
from pydantic_settings import BaseSettings
from pydantic import Field, field_validator
#
# Local imports
from loguru import logger

#######################################################################################################################
#
# Settings Class

class Settings(BaseSettings):
    """Configuration with persistent secret management for user registration system"""
    
    # ===== Core Settings =====
    AUTH_MODE: Literal["single_user", "multi_user"] = Field(
        default="single_user",
        description="Authentication mode: single_user (API key) or multi_user (JWT)"
    )
    
    DATABASE_URL: str = Field(
        default="sqlite:///./Databases/users.db",
        description="Database URL - PostgreSQL for multi-user: postgresql://user:pass@localhost/tldw"
    )
    
    # ===== JWT Settings with secure storage =====
    JWT_SECRET_KEY: Optional[str] = Field(
        default=None,
        description="JWT signing key - MUST be set via environment variable in production"
    )
    
    ACCESS_TOKEN_EXPIRE_MINUTES: int = Field(
        default=30,
        description="Access token expiration time in minutes"
    )
    
    REFRESH_TOKEN_EXPIRE_DAYS: int = Field(
        default=7,
        description="Refresh token expiration time in days"
    )
    
    JWT_ALGORITHM: str = Field(
        default="HS256",
        description="JWT signing algorithm"
    )
    
    # ===== Password Settings =====
    PASSWORD_MIN_LENGTH: int = Field(
        default=10,
        ge=8,
        description="Minimum password length"
    )
    
    ARGON2_MEMORY_COST: int = Field(
        default=32768,  # 32MB
        description="Argon2 memory cost parameter"
    )
    
    ARGON2_TIME_COST: int = Field(
        default=2,
        description="Argon2 time cost (iterations)"
    )
    
    ARGON2_PARALLELISM: int = Field(
        default=1,
        description="Argon2 parallelism factor"
    )
    
    # ===== Redis Configuration (Optional) =====
    REDIS_URL: Optional[str] = Field(
        default=None,
        description="Redis URL for session caching (optional)"
    )
    
    REDIS_MAX_CONNECTIONS: int = Field(
        default=50,
        description="Maximum Redis connections"
    )
    
    # ===== Security Settings =====
    ENABLE_REGISTRATION: bool = Field(
        default=False,
        description="Enable user registration"
    )
    
    REQUIRE_REGISTRATION_CODE: bool = Field(
        default=True,
        description="Require registration code for new users"
    )
    
    MAX_LOGIN_ATTEMPTS: int = Field(
        default=5,
        ge=3,
        description="Maximum login attempts before lockout"
    )
    
    LOCKOUT_DURATION_MINUTES: int = Field(
        default=15,
        ge=5,
        description="Account lockout duration in minutes"
    )
    
    # ===== Rate Limiting =====
    RATE_LIMIT_ENABLED: bool = Field(
        default=True,
        description="Enable rate limiting"
    )
    
    RATE_LIMIT_PER_MINUTE: int = Field(
        default=60,
        ge=10,
        description="Requests allowed per minute"
    )
    
    RATE_LIMIT_BURST: int = Field(
        default=10,
        ge=5,
        description="Burst requests allowed"
    )
    
    # ===== Storage Settings =====
    DEFAULT_STORAGE_QUOTA_MB: int = Field(
        default=5120,  # 5GB
        ge=100,
        description="Default storage quota in MB"
    )
    
    USER_DATA_BASE_PATH: str = Field(
        default="./user_databases",
        description="Base path for user data directories"
    )
    
    CHROMADB_BASE_PATH: str = Field(
        default="./chromadb_data",
        description="Base path for ChromaDB data"
    )
    
    # ===== Registration Settings =====
    DEFAULT_USER_ROLE: str = Field(
        default="user",
        description="Default role for new users"
    )
    
    REGISTRATION_CODE_DEFAULT_EXPIRY_DAYS: int = Field(
        default=7,
        ge=1,
        description="Default registration code expiry in days"
    )
    
    REGISTRATION_CODE_DEFAULT_MAX_USES: int = Field(
        default=1,
        ge=1,
        description="Default maximum uses for registration code"
    )
    
    MAX_BATCH_REGISTRATION_CODES: int = Field(
        default=100,
        ge=1,
        le=1000,
        description="Maximum registration codes per batch"
    )
    
    # ===== Database Pool Settings =====
    DATABASE_POOL_MIN_SIZE: int = Field(
        default=5,
        ge=1,
        description="Minimum database pool size"
    )
    
    DATABASE_POOL_MAX_SIZE: int = Field(
        default=20,
        ge=5,
        description="Maximum database pool size"
    )
    
    DATABASE_MAX_QUERIES: int = Field(
        default=50000,
        description="Maximum queries before connection reset"
    )
    
    DATABASE_MAX_INACTIVE_CONNECTION_LIFETIME: int = Field(
        default=300,
        description="Maximum inactive connection lifetime in seconds"
    )
    
    # ===== Session Settings =====
    SESSION_COOKIE_SECURE: bool = Field(
        default=True,
        description="Use secure cookies (HTTPS only)"
    )
    
    SESSION_COOKIE_HTTPONLY: bool = Field(
        default=True,
        description="HTTP-only cookies (no JavaScript access)"
    )
    
    SESSION_COOKIE_SAMESITE: str = Field(
        default="lax",
        description="SameSite cookie attribute"
    )
    
    SESSION_CLEANUP_INTERVAL_HOURS: int = Field(
        default=1,
        ge=1,
        description="Session cleanup interval in hours"
    )
    
    # ===== Encryption Settings =====
    SESSION_ENCRYPTION_KEY: Optional[str] = Field(
        default=None,
        description="Base64-encoded Fernet key for session token encryption (auto-generated if not set)"
    )
    
    API_KEY_PEPPER: Optional[str] = Field(
        default=None,
        description="Additional secret for API key hashing (recommended for production)"
    )
    
    # ===== Monitoring =====
    ENABLE_HEALTH_CHECK: bool = Field(
        default=True,
        description="Enable health check endpoints"
    )
    
    ENABLE_METRICS: bool = Field(
        default=True,
        description="Enable Prometheus metrics"
    )
    
    METRICS_PORT: int = Field(
        default=9090,
        ge=1024,
        le=65535,
        description="Prometheus metrics port"
    )
    
    # ===== Data Retention =====
    AUDIT_LOG_RETENTION_DAYS: int = Field(
        default=180,
        ge=30,
        description="Audit log retention in days"
    )
    
    PASSWORD_HISTORY_RETENTION_COUNT: int = Field(
        default=5,
        ge=3,
        description="Number of previous passwords to remember"
    )
    
    SESSION_LOG_RETENTION_DAYS: int = Field(
        default=90,
        ge=7,
        description="Session log retention in days"
    )
    
    # ===== Single-User Mode Settings =====
    SINGLE_USER_API_KEY: Optional[str] = Field(
        default=None,
        description="API key for single-user mode - MUST be set via environment variable"
    )
    
    SINGLE_USER_FIXED_ID: int = Field(
        default=1,
        description="Fixed user ID for single-user mode"
    )
    
    # ===== Service Account Settings =====
    SERVICE_ACCOUNT_RATE_LIMIT: int = Field(
        default=1000,
        ge=100,
        description="Rate limit for service accounts per minute"
    )
    
    def __init__(self, **kwargs):
        """Initialize settings and ensure JWT secret persistence"""
        super().__init__(**kwargs)
        self._ensure_jwt_secret()
        self._validate_api_key()
    
    def _ensure_jwt_secret(self):
        """Ensure JWT secret exists with secure handling"""
        # Skip JWT secret handling in single-user mode
        if self.AUTH_MODE == "single_user":
            logger.debug("Single-user mode - skipping JWT secret initialization")
            return
        
        # Environment variable is REQUIRED for JWT secret
        if self.JWT_SECRET_KEY:
            # Secret provided via environment - validate it
            if len(self.JWT_SECRET_KEY) < 32:
                raise ValueError("JWT_SECRET_KEY must be at least 32 characters for security")
            logger.info("Using JWT secret from environment variable")
            return
        
        # No JWT secret available - this is a configuration error
        raise ValueError(
            "JWT_SECRET_KEY must be set via environment variable for multi-user mode. "
            "This is required for security - file-based storage is not supported."
        )
    
    def _validate_api_key(self):
        """Validate API key for single-user mode"""
        if self.AUTH_MODE == "single_user":
            if not self.SINGLE_USER_API_KEY:
                # Generate a secure random API key for first-time setup
                self.SINGLE_USER_API_KEY = secrets.token_urlsafe(32)
                logger.warning(
                    f"⚠️ Generated temporary API key for single-user mode: {self.SINGLE_USER_API_KEY}\n"
                    "Please set SINGLE_USER_API_KEY environment variable for production!"
                )
            elif self.SINGLE_USER_API_KEY == "change-me-in-production":
                raise ValueError(
                    "Default API key detected! Please set SINGLE_USER_API_KEY environment variable."
                )
            elif len(self.SINGLE_USER_API_KEY) < 16:
                raise ValueError("SINGLE_USER_API_KEY must be at least 16 characters")
    
    @field_validator("JWT_SECRET_KEY")
    @classmethod
    def validate_jwt_secret(cls, v, info):
        """Validate JWT secret strength"""
        # Skip validation in single-user mode
        if info.data.get("AUTH_MODE") == "single_user":
            return v
            
        if v and len(v) < 32:
            raise ValueError("JWT_SECRET_KEY must be at least 32 characters")
        return v
    
    @field_validator("DATABASE_URL")
    @classmethod
    def validate_database_url(cls, v, info):
        """Validate database URL based on auth mode"""
        auth_mode = info.data.get("AUTH_MODE", "single_user")
        
        if auth_mode == "multi_user" and v.startswith("sqlite"):
            logger.warning(
                "Using SQLite in multi-user mode is not recommended. "
                "Consider using PostgreSQL for better concurrency."
            )
        
        return v
    
    @field_validator("REDIS_URL")
    @classmethod
    def validate_redis_url(cls, v):
        """Validate Redis URL format if provided"""
        if v and not (v.startswith("redis://") or v.startswith("rediss://")):
            raise ValueError("REDIS_URL must start with redis:// or rediss://")
        return v
    
    model_config = {
        "env_file": ".env",
        "env_file_encoding": "utf-8",
        "case_sensitive": False,
        "extra": "allow"  # Allow extra fields for backward compatibility
    }


# ===== Singleton Settings Instance =====
_settings: Optional[Settings] = None


def get_settings() -> Settings:
    """Get settings singleton instance"""
    global _settings
    if not _settings:
        _settings = Settings()
        logger.info(f"Settings initialized - Auth mode: {_settings.AUTH_MODE}")
    return _settings


def reset_settings():
    """Reset settings singleton (mainly for testing)"""
    global _settings
    _settings = None


# ===== Utility Functions =====
def is_multi_user_mode() -> bool:
    """Check if system is in multi-user mode"""
    return get_settings().AUTH_MODE == "multi_user"


def is_single_user_mode() -> bool:
    """Check if system is in single-user mode"""
    return get_settings().AUTH_MODE == "single_user"


def get_database_url() -> str:
    """Get the configured database URL"""
    return get_settings().DATABASE_URL


def get_jwt_secret() -> str:
    """Get JWT secret key (multi-user mode only)"""
    settings = get_settings()
    if settings.AUTH_MODE != "multi_user":
        raise RuntimeError("JWT secret only available in multi-user mode")
    return settings.JWT_SECRET_KEY


#
# End of settings.py
#######################################################################################################################