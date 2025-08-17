# csrf_protection.py
# Description: CSRF protection middleware for FastAPI
#
# Imports
import secrets
import hashlib
from typing import Optional, Set, Callable
from datetime import datetime, timedelta
#
# 3rd-party imports
from fastapi import Request, Response, HTTPException, status
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware
from loguru import logger
#
# Local imports
from tldw_Server_API.app.core.AuthNZ.settings import get_settings
from tldw_Server_API.app.core.config import settings as global_settings

#######################################################################################################################
#
# CSRF Token Manager

class CSRFTokenManager:
    """Manages CSRF tokens for session protection"""
    
    def __init__(self):
        """Initialize CSRF token manager"""
        self.settings = get_settings()
        self.token_header_name = "X-CSRF-Token"
        self.token_cookie_name = "csrf_token"
        self.token_length = 32
        
        # Methods that require CSRF protection
        self.protected_methods = {"POST", "PUT", "PATCH", "DELETE"}
        
        # Paths to exclude from CSRF protection
        self.excluded_paths = {
            "/api/v1/auth/login",  # Login needs to work without existing token
            "/api/v1/auth/refresh",  # Token refresh
            "/api/v1/health",  # Health checks
            "/docs",  # API documentation
            "/openapi.json",  # OpenAPI schema
            "/redoc",  # ReDoc documentation
        }
        
        # Content types that require CSRF protection
        self.protected_content_types = {
            "application/json",
            "application/x-www-form-urlencoded",
            "multipart/form-data",
            "text/plain",
        }
    
    def generate_token(self) -> str:
        """Generate a new CSRF token"""
        return secrets.token_urlsafe(self.token_length)
    
    def hash_token(self, token: str) -> str:
        """Create a hash of the token for comparison"""
        return hashlib.sha256(token.encode()).hexdigest()
    
    def validate_token(self, cookie_token: str, header_token: str) -> bool:
        """
        Validate CSRF token using double-submit cookie pattern
        
        Args:
            cookie_token: Token from cookie
            header_token: Token from header
            
        Returns:
            True if tokens match and are valid
        """
        if not cookie_token or not header_token:
            return False
        
        # Constant-time comparison to prevent timing attacks
        return secrets.compare_digest(cookie_token, header_token)
    
    def should_protect(self, request: Request) -> bool:
        """
        Determine if request should be protected by CSRF
        
        Args:
            request: The incoming request
            
        Returns:
            True if CSRF protection should be applied
        """
        # Skip if not a protected method
        if request.method not in self.protected_methods:
            return False
        
        # Skip if path is excluded
        path = str(request.url.path)
        if any(path.startswith(excluded) for excluded in self.excluded_paths):
            return False
        
        # Skip if single-user mode with API key auth
        if self.settings.AUTH_MODE == "single_user":
            # Check if API key is present
            api_key = request.headers.get("X-API-KEY")
            if api_key:
                return False  # API key auth doesn't need CSRF
        
        # Check content type
        content_type = request.headers.get("content-type", "").lower()
        if content_type:
            # Extract base content type (remove charset, etc.)
            content_type = content_type.split(";")[0].strip()
            
            # Skip if not a protected content type
            if not any(ct in content_type for ct in self.protected_content_types):
                return False
        
        return True
    
    def set_cookie(self, response: Response, token: str):
        """
        Set CSRF token cookie
        
        Args:
            response: Response to add cookie to
            token: CSRF token to set
        """
        response.set_cookie(
            key=self.token_cookie_name,
            value=token,
            max_age=3600 * 24,  # 24 hours
            httponly=False,  # Must be readable by JavaScript
            secure=self.settings.SESSION_COOKIE_SECURE,
            samesite=self.settings.SESSION_COOKIE_SAMESITE,
            path="/"
        )


#######################################################################################################################
#
# CSRF Protection Middleware

class CSRFProtectionMiddleware(BaseHTTPMiddleware):
    """
    CSRF Protection Middleware using double-submit cookie pattern
    
    This middleware implements CSRF protection by:
    1. Setting a CSRF token cookie on responses
    2. Requiring the token to be submitted in a header for protected requests
    3. Validating that the header token matches the cookie token
    """
    
    def __init__(self, app, enabled: bool = True):
        """
        Initialize CSRF protection middleware
        
        Args:
            app: FastAPI application
            enabled: Whether CSRF protection is enabled
        """
        super().__init__(app)
        self.enabled = enabled
        self.token_manager = CSRFTokenManager()
        logger.info(f"CSRF Protection Middleware initialized (enabled={enabled})")
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """
        Process request with CSRF protection
        
        Args:
            request: Incoming request
            call_next: Next middleware/handler
            
        Returns:
            Response with CSRF token cookie if needed
        """
        # Check runtime setting to allow test overrides
        # Use global settings which tests can modify
        runtime_csrf_enabled = global_settings.get('CSRF_ENABLED', None)
        
        # If CSRF_ENABLED is explicitly False, bypass protection
        if runtime_csrf_enabled is False or not self.enabled:
            return await call_next(request)
        
        # Check if this request needs CSRF protection
        if self.token_manager.should_protect(request):
            # Get tokens
            cookie_token = request.cookies.get(self.token_manager.token_cookie_name)
            header_token = request.headers.get(self.token_manager.token_header_name)
            
            # Validate tokens
            if not self.token_manager.validate_token(cookie_token, header_token):
                logger.warning(
                    f"CSRF token validation failed for {request.method} {request.url.path} "
                    f"from {request.client.host if request.client else 'unknown'}"
                )
                
                # Return 403 Forbidden
                return JSONResponse(
                    status_code=status.HTTP_403_FORBIDDEN,
                    content={"detail": "CSRF token validation failed"}
                )
        
        # Process request
        response = await call_next(request)
        
        # Set CSRF token cookie if not present
        if self.token_manager.token_cookie_name not in request.cookies:
            token = self.token_manager.generate_token()
            self.token_manager.set_cookie(response, token)
            
            # Also add token to response header for easy access
            response.headers[self.token_manager.token_header_name] = token
        
        return response


#######################################################################################################################
#
# Helper Functions

def get_csrf_token(request: Request) -> Optional[str]:
    """
    Get CSRF token from request cookie
    
    Args:
        request: FastAPI request
        
    Returns:
        CSRF token if present, None otherwise
    """
    return request.cookies.get("csrf_token")


def validate_csrf_token(request: Request) -> bool:
    """
    Validate CSRF token in request
    
    Args:
        request: FastAPI request
        
    Returns:
        True if CSRF token is valid
        
    Raises:
        HTTPException: If CSRF validation fails
    """
    manager = CSRFTokenManager()
    
    if manager.should_protect(request):
        cookie_token = request.cookies.get(manager.token_cookie_name)
        header_token = request.headers.get(manager.token_header_name)
        
        if not manager.validate_token(cookie_token, header_token):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="CSRF token validation failed"
            )
    
    return True


#######################################################################################################################
#
# FastAPI Integration

def add_csrf_protection(app):
    """
    Add CSRF protection middleware to FastAPI app
    
    Args:
        app: FastAPI application instance
    """
    settings = get_settings()
    
    # Check both AUTH_MODE and CSRF_ENABLED setting
    # CSRF_ENABLED can override the default behavior for testing
    csrf_enabled = global_settings.get('CSRF_ENABLED', None)
    
    if csrf_enabled is False:
        # Explicitly disabled (e.g., for testing)
        logger.info("CSRF Protection explicitly disabled via CSRF_ENABLED setting")
        app.add_middleware(
            CSRFProtectionMiddleware,
            enabled=False
        )
    elif settings.AUTH_MODE == "multi_user" or csrf_enabled is True:
        # Enable for multi-user mode or if explicitly enabled
        app.add_middleware(
            CSRFProtectionMiddleware,
            enabled=True
        )
        logger.info("CSRF Protection enabled")
    else:
        # Single-user mode and not explicitly enabled
        app.add_middleware(
            CSRFProtectionMiddleware,
            enabled=False
        )
        logger.info("CSRF Protection disabled for single-user mode")


#
# End of csrf_protection.py
#######################################################################################################################