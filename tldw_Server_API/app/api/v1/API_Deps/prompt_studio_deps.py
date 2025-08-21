# prompt_studio_deps.py
# FastAPI dependency injection for Prompt Studio feature

import logging
import threading
from pathlib import Path
from typing import Dict, Optional, Any
from functools import lru_cache

from fastapi import Depends, HTTPException, status, Header, Request
from cachetools import LRUCache
from loguru import logger

# Local imports
from tldw_Server_API.app.core.DB_Management.PromptStudioDatabase import (
    PromptStudioDatabase, DatabaseError, SchemaError, InputError, ConflictError
)
from tldw_Server_API.app.core.config import settings
from tldw_Server_API.app.api.v1.API_Deps.auth_deps import get_current_active_user
from tldw_Server_API.app.api.v1.schemas.prompt_studio_base import SecurityConfig

########################################################################################################################
# Configuration

DEFAULT_PROMPT_STUDIO_DB_SUBDIR = "prompt_studio_dbs"
MAIN_USER_DATA_BASE_DIR = settings.get("USER_DB_BASE_DIR")

if not MAIN_USER_DATA_BASE_DIR:
    logger.critical("CRITICAL: USER_DB_BASE_DIR is not configured in settings.")
    MAIN_USER_DATA_BASE_DIR = Path("./app_data/user_databases_fallback").resolve()
    logger.error(f"USER_DB_BASE_DIR missing, using fallback: {MAIN_USER_DATA_BASE_DIR}")

SERVER_CLIENT_ID = settings.get("SERVER_CLIENT_ID", "prompt_studio_server")

# Global cache for database instances
MAX_CACHED_INSTANCES = settings.get("MAX_CACHED_PROMPT_STUDIO_DB_INSTANCES", 20)
_db_instances_cache: LRUCache = LRUCache(maxsize=MAX_CACHED_INSTANCES)
_db_lock = threading.Lock()

########################################################################################################################
# Helper Functions

def _get_prompt_studio_db_path_for_user(user_id: str) -> Path:
    """
    Determines the Prompt Studio database file path for a given user.
    
    Args:
        user_id: User identifier
        
    Returns:
        Path to the user's Prompt Studio database
    """
    user_dir_name = str(user_id)
    user_specific_db_dir = MAIN_USER_DATA_BASE_DIR / user_dir_name / DEFAULT_PROMPT_STUDIO_DB_SUBDIR
    
    # Ensure directory exists
    user_specific_db_dir.mkdir(parents=True, exist_ok=True)
    
    db_file = user_specific_db_dir / "prompt_studio.db"
    return db_file

def _get_or_create_prompt_studio_db(user_id: str, client_id: str) -> PromptStudioDatabase:
    """
    Get or create a PromptStudioDatabase instance for a user.
    
    Args:
        user_id: User identifier
        client_id: Client identifier for sync logging
        
    Returns:
        PromptStudioDatabase instance
    """
    db_path = _get_prompt_studio_db_path_for_user(user_id)
    cache_key = str(db_path)
    
    with _db_lock:
        # Check cache first
        if cache_key in _db_instances_cache:
            logger.debug(f"Using cached PromptStudioDatabase for user {user_id}")
            return _db_instances_cache[cache_key]
        
        # Create new instance
        try:
            db_instance = PromptStudioDatabase(db_path, client_id)
            _db_instances_cache[cache_key] = db_instance
            logger.info(f"Created new PromptStudioDatabase instance for user {user_id}")
            return db_instance
        except Exception as e:
            logger.error(f"Failed to create PromptStudioDatabase for user {user_id}: {e}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to initialize database"
            )

########################################################################################################################
# User Context Dependencies

async def get_prompt_studio_user(
    request: Request,
    current_user: Optional[Dict] = Depends(get_current_active_user),
    x_client_id: Optional[str] = Header(None)
) -> Dict[str, Any]:
    """
    Extract user context for Prompt Studio operations.
    
    Args:
        request: FastAPI request object
        current_user: Current authenticated user
        x_client_id: Client ID from header
        
    Returns:
        User context dictionary
    """
    import os
    
    # Check for test mode bypass
    if os.getenv("TEST_MODE") == "true":
        user_context = {
            "user_id": "test-user",
            "client_id": x_client_id or "test-client",
            "is_authenticated": True,
            "is_admin": True,
            "permissions": ["all"]
        }
        request.state.user_context = user_context
        return user_context
    
    # Default user context
    user_context = {
        "user_id": "anonymous",
        "client_id": x_client_id or "web",
        "is_authenticated": False,
        "is_admin": False,
        "permissions": []
    }
    
    # Update with authenticated user info if available
    if current_user:
        user_context.update({
            "user_id": str(current_user.get("id", current_user.get("user_id", "anonymous"))),
            "is_authenticated": True,
            "is_admin": current_user.get("is_admin", False),
            "permissions": current_user.get("permissions", [])
        })
    
    # Store in request state for downstream use
    request.state.user_context = user_context
    
    return user_context

########################################################################################################################
# Database Dependencies

async def get_prompt_studio_db(
    user_context: Dict = Depends(get_prompt_studio_user)
) -> PromptStudioDatabase:
    """
    Get PromptStudioDatabase instance for the current user.
    
    Args:
        user_context: User context from authentication
        
    Returns:
        PromptStudioDatabase instance
    """
    user_id = user_context["user_id"]
    client_id = user_context["client_id"]
    
    if user_id == "anonymous" and not settings.get("ALLOW_ANONYMOUS_PROMPT_STUDIO", False):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication required for Prompt Studio"
        )
    
    return _get_or_create_prompt_studio_db(user_id, client_id)

########################################################################################################################
# Permission Dependencies

async def require_project_access(
    project_id: int,
    user_context: Dict = Depends(get_prompt_studio_user),
    db: PromptStudioDatabase = Depends(get_prompt_studio_db)
) -> bool:
    """
    Verify user has access to a specific project.
    
    Args:
        project_id: Project ID to check
        user_context: User context
        db: Database instance
        
    Returns:
        True if access granted
        
    Raises:
        HTTPException: If access denied
    """
    try:
        project = db.get_project(project_id)
        
        if not project:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Project {project_id} not found"
            )
        
        # Check ownership or admin status
        if project["user_id"] != user_context["user_id"] and not user_context["is_admin"]:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Access denied to this project"
            )
        
        return True
        
    except DatabaseError as e:
        logger.error(f"Database error checking project access: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Database error"
        )

async def require_project_write_access(
    project_id: int,
    user_context: Dict = Depends(get_prompt_studio_user),
    db: PromptStudioDatabase = Depends(get_prompt_studio_db)
) -> bool:
    """
    Verify user has write access to a project.
    Currently same as read access, but separated for future permission granularity.
    """
    return await require_project_access(project_id, user_context, db)

########################################################################################################################
# Security Configuration

@lru_cache()
def get_security_config() -> SecurityConfig:
    """
    Get security configuration for Prompt Studio.
    Cached for performance.
    
    Returns:
        SecurityConfig instance
    """
    return SecurityConfig(
        max_prompt_length=settings.get("PROMPT_STUDIO_MAX_PROMPT_LENGTH", 50000),
        max_test_cases=settings.get("PROMPT_STUDIO_MAX_TEST_CASES", 1000),
        max_concurrent_jobs=settings.get("PROMPT_STUDIO_MAX_CONCURRENT_JOBS", 10),
        enable_prompt_validation=settings.get("PROMPT_STUDIO_ENABLE_VALIDATION", True),
        enable_rate_limiting=settings.get("PROMPT_STUDIO_ENABLE_RATE_LIMITING", True)
    )

########################################################################################################################
# Rate Limiting

class PromptStudioRateLimiter:
    """Simple rate limiter for Prompt Studio operations."""
    
    def __init__(self):
        self.requests = {}
        self.lock = threading.Lock()
    
    def check_rate_limit(self, user_id: str, operation: str, limit: int = 100, window: int = 60) -> bool:
        """
        Check if user has exceeded rate limit.
        
        Args:
            user_id: User identifier
            operation: Operation being performed
            limit: Maximum requests in window
            window: Time window in seconds
            
        Returns:
            True if within limits
        """
        import time
        
        key = f"{user_id}:{operation}"
        current_time = time.time()
        
        with self.lock:
            if key not in self.requests:
                self.requests[key] = []
            
            # Remove old requests outside window
            self.requests[key] = [
                t for t in self.requests[key] 
                if current_time - t < window
            ]
            
            # Check limit
            if len(self.requests[key]) >= limit:
                return False
            
            # Add current request
            self.requests[key].append(current_time)
            return True

# Global rate limiter instance
_rate_limiter = PromptStudioRateLimiter()

async def check_rate_limit(
    operation: str = "default",
    user_context: Dict = Depends(get_prompt_studio_user),
    security_config: SecurityConfig = Depends(get_security_config)
) -> bool:
    """
    Check rate limit for current user and operation.
    
    Args:
        operation: Operation being performed
        user_context: User context
        security_config: Security configuration
        
    Returns:
        True if within limits
        
    Raises:
        HTTPException: If rate limit exceeded
    """
    if not security_config.enable_rate_limiting:
        return True
    
    user_id = user_context["user_id"]
    
    # Different limits for different operations
    limits = {
        "create_project": (10, 3600),  # 10 per hour
        "optimize": (5, 3600),  # 5 per hour
        "evaluate": (20, 3600),  # 20 per hour
        "generate": (30, 3600),  # 30 per hour
        "default": (100, 60)  # 100 per minute
    }
    
    limit, window = limits.get(operation, limits["default"])
    
    if not _rate_limiter.check_rate_limit(user_id, operation, limit, window):
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail=f"Rate limit exceeded for operation: {operation}"
        )
    
    return True

########################################################################################################################
# Cleanup

def shutdown_prompt_studio_deps():
    """
    Cleanup function to close all cached database connections.
    Should be called on application shutdown.
    """
    with _db_lock:
        for db_instance in _db_instances_cache.values():
            try:
                if hasattr(db_instance, 'close'):
                    db_instance.close()
            except Exception as e:
                logger.error(f"Error closing database instance: {e}")
        
        _db_instances_cache.clear()
        logger.info("Prompt Studio dependencies cleaned up")