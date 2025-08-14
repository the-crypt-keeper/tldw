# auth_deps.py
# Description: FastAPI dependency injection for authentication services
#
# Imports
from typing import Optional, Dict, Any
#
# 3rd-party imports
from fastapi import Depends, HTTPException, status, Request
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from loguru import logger
#
# Local imports
from tldw_Server_API.app.core.AuthNZ.database import DatabasePool, get_db_pool
from tldw_Server_API.app.core.AuthNZ.password_service import PasswordService, get_password_service
from tldw_Server_API.app.core.AuthNZ.jwt_service import JWTService, get_jwt_service
from tldw_Server_API.app.core.AuthNZ.session_manager import SessionManager, get_session_manager
from tldw_Server_API.app.core.AuthNZ.rate_limiter import RateLimiter, get_rate_limiter
from tldw_Server_API.app.services.registration_service import RegistrationService, get_registration_service
from tldw_Server_API.app.services.storage_quota_service import StorageQuotaService, get_storage_service
from tldw_Server_API.app.core.AuthNZ.exceptions import (
    AuthenticationError,
    InvalidTokenError,
    TokenExpiredError,
    UserNotFoundError,
    AccountInactiveError,
    InsufficientPermissionsError
)

#######################################################################################################################
#
# Security scheme for JWT bearer tokens

security = HTTPBearer()


#######################################################################################################################
#
# Service Dependency Functions

async def get_db_transaction():
    """Get database connection in transaction mode"""
    db_pool = await get_db_pool()
    async with db_pool.transaction() as conn:
        yield conn


async def get_password_service_dep() -> PasswordService:
    """Get password service dependency"""
    return get_password_service()


async def get_jwt_service_dep() -> JWTService:
    """Get JWT service dependency"""
    return get_jwt_service()


async def get_session_manager_dep() -> SessionManager:
    """Get session manager dependency"""
    return await get_session_manager()


async def get_rate_limiter_dep() -> RateLimiter:
    """Get rate limiter dependency"""
    return await get_rate_limiter()


async def get_registration_service_dep() -> RegistrationService:
    """Get registration service dependency"""
    return await get_registration_service()


async def get_storage_service_dep() -> StorageQuotaService:
    """Get storage service dependency"""
    return await get_storage_service()


#######################################################################################################################
#
# User Authentication Dependencies

async def get_current_user(
    request: Request,
    credentials: HTTPAuthorizationCredentials = Depends(security),
    jwt_service: JWTService = Depends(get_jwt_service_dep),
    session_manager: SessionManager = Depends(get_session_manager_dep),
    db_pool: DatabasePool = Depends(get_db_pool)
) -> Dict[str, Any]:
    """
    Get current authenticated user from JWT token
    
    Args:
        request: FastAPI request object
        credentials: Bearer token from Authorization header
        jwt_service: JWT service instance
        session_manager: Session manager instance
        db_pool: Database pool instance
        
    Returns:
        User dictionary with all user information
        
    Raises:
        HTTPException: If authentication fails
    """
    try:
        # Extract token
        token = credentials.credentials
        
        # Decode and validate JWT
        payload = jwt_service.decode_access_token(token)
        
        # Check if token is blacklisted
        if await session_manager.is_token_blacklisted(token):
            raise InvalidTokenError("Token has been revoked")
        
        # Get user from database
        user_id = payload.get("user_id")
        if not user_id:
            raise InvalidTokenError("Invalid token payload")
        
        # Fetch user from database
        user = await db_pool.fetchone(
            "SELECT * FROM users WHERE id = ? AND is_active = ?",
            user_id, True
        )
        
        if not user:
            raise UserNotFoundError(f"User {user_id}")
        
        # Update session activity
        session_id = payload.get("session_id")
        if session_id:
            client_ip = request.client.host if request.client else None
            user_agent = request.headers.get("User-Agent")
            await session_manager.update_session_activity(
                session_id, client_ip, user_agent
            )
        
        # Convert to dict if needed
        if hasattr(user, 'dict'):
            user = dict(user)
        
        return user
        
    except TokenExpiredError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Token has expired",
            headers={"WWW-Authenticate": "Bearer"}
        )
    except InvalidTokenError as e:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=str(e),
            headers={"WWW-Authenticate": "Bearer"}
        )
    except Exception as e:
        logger.error(f"Authentication error: {e}")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Could not validate credentials",
            headers={"WWW-Authenticate": "Bearer"}
        )


async def get_current_active_user(
    current_user: Dict[str, Any] = Depends(get_current_user)
) -> Dict[str, Any]:
    """
    Get current active user (verified and not locked)
    
    Args:
        current_user: Current authenticated user
        
    Returns:
        User dictionary if active and verified
        
    Raises:
        HTTPException: If user is inactive or unverified
    """
    if not current_user.get("is_active"):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="User account is inactive"
        )
    
    if not current_user.get("is_verified"):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Email verification required"
        )
    
    return current_user


async def require_admin(
    current_user: Dict[str, Any] = Depends(get_current_active_user)
) -> Dict[str, Any]:
    """
    Require admin role for access
    
    Args:
        current_user: Current active user
        
    Returns:
        User dictionary if admin
        
    Raises:
        HTTPException: If user is not admin
    """
    if current_user.get("role") != "admin":
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Admin access required"
        )
    
    return current_user


def require_role(role: str):
    """
    Create a dependency that requires a specific role
    
    Args:
        role: Required role name
        
    Returns:
        Dependency function that checks for the role
    """
    async def role_checker(
        current_user: Dict[str, Any] = Depends(get_current_active_user)
    ) -> Dict[str, Any]:
        user_role = current_user.get("role", "user")
        
        # Admin can access everything
        if user_role == "admin":
            return current_user
        
        # Check specific role
        if user_role != role:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Requires {role} role"
            )
        
        return current_user
    
    return role_checker


async def get_optional_current_user(
    request: Request,
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(security),
    jwt_service: JWTService = Depends(get_jwt_service_dep),
    session_manager: SessionManager = Depends(get_session_manager_dep),
    db_pool: DatabasePool = Depends(get_db_pool)
) -> Optional[Dict[str, Any]]:
    """
    Get current user if authenticated, None otherwise
    
    This is useful for endpoints that have different behavior
    for authenticated vs unauthenticated users
    
    Args:
        request: FastAPI request object
        credentials: Optional bearer token
        jwt_service: JWT service instance
        session_manager: Session manager instance
        db_pool: Database pool instance
        
    Returns:
        User dictionary if authenticated, None otherwise
    """
    if not credentials:
        return None
    
    try:
        return await get_current_user(
            request, credentials, jwt_service, session_manager, db_pool
        )
    except HTTPException:
        return None


#######################################################################################################################
#
# Rate Limiting Dependencies

async def check_rate_limit(
    request: Request,
    rate_limiter: RateLimiter = Depends(get_rate_limiter_dep)
):
    """
    Check rate limit for the current request
    
    Args:
        request: FastAPI request object
        rate_limiter: Rate limiter instance
        
    Raises:
        HTTPException: If rate limit exceeded
    """
    # Get client IP
    client_ip = request.client.host if request.client else "unknown"
    
    # Get endpoint key
    endpoint = f"{request.method}:{request.url.path}"
    
    # Check rate limit
    allowed, retry_after = await rate_limiter.check_rate_limit(client_ip, endpoint)
    
    if not allowed:
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail=f"Rate limit exceeded. Retry after {retry_after} seconds",
            headers={"Retry-After": str(retry_after)}
        )


async def check_auth_rate_limit(
    request: Request,
    rate_limiter: RateLimiter = Depends(get_rate_limiter_dep)
):
    """
    Check stricter rate limit for authentication endpoints
    
    Args:
        request: FastAPI request object
        rate_limiter: Rate limiter instance
        
    Raises:
        HTTPException: If rate limit exceeded
    """
    # Get client IP
    client_ip = request.client.host if request.client else "unknown"
    
    # Use stricter limits for auth endpoints
    allowed, metadata = await rate_limiter.check_rate_limit(
        client_ip, 
        "auth", 
        limit=5,  # Stricter limit (5 requests per minute)
        window_minutes=1
    )
    retry_after = metadata.get("retry_after", 60)
    
    if not allowed:
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail=f"Too many authentication attempts. Retry after {retry_after} seconds",
            headers={"Retry-After": str(retry_after)}
        )


#
# End of auth_deps.py
#######################################################################################################################