"""
Secure configuration management for unified MCP module

All sensitive configuration is loaded from environment variables or secure config files.
No hardcoded secrets allowed.
"""

import os
import secrets
from typing import Optional, Dict, Any, List
from functools import lru_cache
from pydantic import Field, validator, SecretStr
from pydantic_settings import BaseSettings
from loguru import logger


class MCPConfig(BaseSettings):
    """
    MCP configuration with secure defaults and environment variable support.
    
    All sensitive values MUST come from environment variables or secure storage.
    """
    
    # Server Configuration
    server_name: str = Field(default="tldw-mcp-unified", env="MCP_SERVER_NAME")
    server_version: str = Field(default="3.0.0", env="MCP_SERVER_VERSION")
    debug_mode: bool = Field(default=False, env="MCP_DEBUG")
    
    # Security Configuration - CRITICAL: No hardcoded secrets!
    jwt_secret_key: SecretStr = Field(..., env="MCP_JWT_SECRET")
    jwt_algorithm: str = Field(default="HS256", env="MCP_JWT_ALGORITHM")
    jwt_access_token_expire_minutes: int = Field(default=30, env="MCP_JWT_ACCESS_EXPIRE")
    jwt_refresh_token_expire_days: int = Field(default=7, env="MCP_JWT_REFRESH_EXPIRE")
    
    # API Key Configuration
    api_key_salt: SecretStr = Field(..., env="MCP_API_KEY_SALT")
    api_key_iterations: int = Field(default=100000, env="MCP_API_KEY_ITERATIONS")
    
    # Database Configuration
    database_url: str = Field(
        default="sqlite+aiosqlite:///./Databases/mcp_unified.db",
        env="MCP_DATABASE_URL"
    )
    database_pool_size: int = Field(default=20, env="MCP_DATABASE_POOL_SIZE")
    database_pool_timeout: int = Field(default=30, env="MCP_DATABASE_POOL_TIMEOUT")
    database_pool_recycle: int = Field(default=3600, env="MCP_DATABASE_POOL_RECYCLE")
    database_echo: bool = Field(default=False, env="MCP_DATABASE_ECHO")
    
    # Redis Configuration (Optional - for distributed deployments)
    redis_url: Optional[str] = Field(default=None, env="MCP_REDIS_URL")
    redis_password: Optional[SecretStr] = Field(default=None, env="MCP_REDIS_PASSWORD")
    redis_ssl: bool = Field(default=False, env="MCP_REDIS_SSL")
    redis_pool_size: int = Field(default=10, env="MCP_REDIS_POOL_SIZE")
    
    # Rate Limiting Configuration
    rate_limit_enabled: bool = Field(default=True, env="MCP_RATE_LIMIT_ENABLED")
    rate_limit_requests_per_minute: int = Field(default=60, env="MCP_RATE_LIMIT_RPM")
    rate_limit_burst_size: int = Field(default=10, env="MCP_RATE_LIMIT_BURST")
    rate_limit_use_redis: bool = Field(default=False, env="MCP_RATE_LIMIT_USE_REDIS")
    
    # WebSocket Configuration
    ws_max_connections: int = Field(default=1000, env="MCP_WS_MAX_CONNECTIONS")
    ws_max_message_size: int = Field(default=1048576, env="MCP_WS_MAX_MESSAGE_SIZE")  # 1MB
    ws_ping_interval: int = Field(default=30, env="MCP_WS_PING_INTERVAL")
    ws_ping_timeout: int = Field(default=60, env="MCP_WS_PING_TIMEOUT")
    ws_close_timeout: int = Field(default=10, env="MCP_WS_CLOSE_TIMEOUT")
    
    # CORS Configuration
    cors_enabled: bool = Field(default=True, env="MCP_CORS_ENABLED")
    cors_origins: List[str] = Field(
        default=["http://localhost:3000", "http://localhost:8000"],
        env="MCP_CORS_ORIGINS"
    )
    cors_allow_credentials: bool = Field(default=True, env="MCP_CORS_CREDENTIALS")
    cors_allow_methods: List[str] = Field(
        default=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
        env="MCP_CORS_METHODS"
    )
    cors_allow_headers: List[str] = Field(
        default=["*"],
        env="MCP_CORS_HEADERS"
    )
    
    # Security Headers
    security_headers_enabled: bool = Field(default=True, env="MCP_SECURITY_HEADERS")
    csp_policy: str = Field(
        default="default-src 'self'; script-src 'self' 'unsafe-inline'",
        env="MCP_CSP_POLICY"
    )
    hsts_max_age: int = Field(default=31536000, env="MCP_HSTS_MAX_AGE")  # 1 year
    
    # Module Configuration
    module_timeout: int = Field(default=30, env="MCP_MODULE_TIMEOUT")
    module_max_retries: int = Field(default=3, env="MCP_MODULE_MAX_RETRIES")
    module_health_check_interval: int = Field(default=60, env="MCP_MODULE_HEALTH_INTERVAL")
    
    # Monitoring Configuration
    metrics_enabled: bool = Field(default=True, env="MCP_METRICS_ENABLED")
    metrics_port: int = Field(default=9090, env="MCP_METRICS_PORT")
    health_check_path: str = Field(default="/health", env="MCP_HEALTH_PATH")
    
    # Logging Configuration
    log_level: str = Field(default="INFO", env="MCP_LOG_LEVEL")
    log_format: str = Field(
        default="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
        env="MCP_LOG_FORMAT"
    )
    log_file: Optional[str] = Field(default=None, env="MCP_LOG_FILE")
    log_rotation: str = Field(default="100 MB", env="MCP_LOG_ROTATION")
    log_retention: str = Field(default="30 days", env="MCP_LOG_RETENTION")
    
    # Audit Logging
    audit_enabled: bool = Field(default=True, env="MCP_AUDIT_ENABLED")
    audit_log_file: str = Field(default="audit.log", env="MCP_AUDIT_LOG_FILE")
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False
    
    @validator("jwt_secret_key", pre=True)
    def validate_jwt_secret(cls, v):
        """Ensure JWT secret is set and strong enough"""
        if not v:
            # Generate a secure random secret if not provided
            logger.warning("JWT secret not provided, generating a random one (not suitable for production)")
            return secrets.token_urlsafe(32)
        
        if len(str(v)) < 32:
            raise ValueError("JWT secret must be at least 32 characters long")
        
        if str(v) == "your-secret-key-change-this-in-production":
            raise ValueError("Default JWT secret detected! Please set MCP_JWT_SECRET environment variable")
        
        return v
    
    @validator("api_key_salt", pre=True)
    def validate_api_key_salt(cls, v):
        """Ensure API key salt is set and secure"""
        if not v:
            logger.warning("API key salt not provided, generating a random one")
            return secrets.token_urlsafe(32)
        
        if len(str(v)) < 32:
            raise ValueError("API key salt must be at least 32 characters long")
        
        return v
    
    @validator("cors_origins", pre=True)
    def parse_cors_origins(cls, v):
        """Parse CORS origins from comma-separated string"""
        if isinstance(v, str):
            return [origin.strip() for origin in v.split(",")]
        return v
    
    @validator("database_url")
    def validate_database_url(cls, v):
        """Validate and potentially modify database URL"""
        if "sqlite" in v and not v.startswith("sqlite+aiosqlite"):
            # Ensure async SQLite driver is used
            v = v.replace("sqlite://", "sqlite+aiosqlite://")
        return v
    
    def get_redis_connection_params(self) -> Optional[Dict[str, Any]]:
        """Get Redis connection parameters if Redis is configured"""
        if not self.redis_url:
            return None
        
        params = {
            "url": self.redis_url,
            "encoding": "utf-8",
            "decode_responses": True,
            "max_connections": self.redis_pool_size,
        }
        
        if self.redis_password:
            params["password"] = self.redis_password.get_secret_value()
        
        if self.redis_ssl:
            params["ssl"] = True
            params["ssl_cert_reqs"] = "required"
        
        return params
    
    def get_database_connection_params(self) -> Dict[str, Any]:
        """Get database connection parameters"""
        return {
            "url": self.database_url,
            "pool_size": self.database_pool_size,
            "max_overflow": 20,
            "pool_timeout": self.database_pool_timeout,
            "pool_recycle": self.database_pool_recycle,
            "echo": self.database_echo,
        }
    
    def configure_logging(self):
        """Configure logging based on settings"""
        logger.remove()  # Remove default handler
        
        # Console logging
        logger.add(
            sink=os.sys.stderr,
            format=self.log_format,
            level=self.log_level,
            colorize=True
        )
        
        # File logging if configured
        if self.log_file:
            logger.add(
                sink=self.log_file,
                format=self.log_format,
                level=self.log_level,
                rotation=self.log_rotation,
                retention=self.log_retention,
                compression="zip"
            )
        
        # Audit logging if enabled
        if self.audit_enabled:
            logger.add(
                sink=self.audit_log_file,
                format="{time:YYYY-MM-DD HH:mm:ss} | AUDIT | {message}",
                level="INFO",
                filter=lambda record: "audit" in record["extra"],
                rotation="1 day",
                retention="90 days",
                compression="zip"
            )


@lru_cache()
def get_config() -> MCPConfig:
    """Get cached configuration instance"""
    try:
        config = MCPConfig()
        config.configure_logging()
        logger.info("MCP configuration loaded successfully")
        return config
    except Exception as e:
        logger.error(f"Failed to load configuration: {e}")
        raise


def validate_config() -> bool:
    """Validate configuration on startup"""
    try:
        config = get_config()
        
        # Check critical security settings
        if not config.jwt_secret_key:
            logger.error("JWT secret key not configured!")
            return False
        
        if not config.api_key_salt:
            logger.error("API key salt not configured!")
            return False
        
        # Validate database connection
        if not config.database_url:
            logger.error("Database URL not configured!")
            return False
        
        # Warn about development settings in production
        if not config.debug_mode:
            if "localhost" in config.cors_origins or "*" in config.cors_origins:
                logger.warning("Localhost or wildcard in CORS origins - not recommended for production")
            
            if config.database_echo:
                logger.warning("Database echo enabled - not recommended for production")
        
        logger.info("Configuration validation passed")
        return True
    
    except Exception as e:
        logger.error(f"Configuration validation failed: {e}")
        return False