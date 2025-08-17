# security_headers.py
# Description: Security headers middleware for FastAPI to protect against common web vulnerabilities
#
# Imports
from typing import Optional, Dict, Any
#
# 3rd-party imports
from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.types import ASGIApp
from loguru import logger
#
# Local imports
from tldw_Server_API.app.core.AuthNZ.settings import get_settings

#######################################################################################################################
#
# Security Headers Middleware
#

class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    """
    Middleware to add security headers to all responses.
    
    These headers help protect against various web vulnerabilities:
    - XSS (Cross-Site Scripting)
    - Clickjacking
    - MIME type sniffing
    - Protocol downgrade attacks
    """
    
    def __init__(
        self,
        app: ASGIApp,
        strict_transport_security: bool = True,
        content_type_options: bool = True,
        frame_options: str = "DENY",
        xss_protection: bool = True,
        content_security_policy: Optional[str] = None,
        referrer_policy: str = "strict-origin-when-cross-origin",
        permissions_policy: Optional[str] = None,
        custom_headers: Optional[Dict[str, str]] = None
    ):
        """
        Initialize security headers middleware.
        
        Args:
            app: The ASGI application
            strict_transport_security: Enable HSTS header
            content_type_options: Enable X-Content-Type-Options header
            frame_options: X-Frame-Options value (DENY, SAMEORIGIN, or ALLOW-FROM uri)
            xss_protection: Enable X-XSS-Protection header
            content_security_policy: CSP header value
            referrer_policy: Referrer-Policy header value
            permissions_policy: Permissions-Policy header value
            custom_headers: Additional custom headers to add
        """
        super().__init__(app)
        
        self.settings = get_settings()
        self.strict_transport_security = strict_transport_security
        self.content_type_options = content_type_options
        self.frame_options = frame_options
        self.xss_protection = xss_protection
        self.content_security_policy = content_security_policy
        self.referrer_policy = referrer_policy
        self.permissions_policy = permissions_policy
        self.custom_headers = custom_headers or {}
        
        logger.info("SecurityHeadersMiddleware initialized")
    
    async def dispatch(self, request: Request, call_next):
        """
        Process the request and add security headers to the response.
        
        Args:
            request: The incoming request
            call_next: The next middleware or route handler
            
        Returns:
            Response with security headers added
        """
        # Process the request
        response = await call_next(request)
        
        # Add security headers
        
        # 1. Strict-Transport-Security (HSTS)
        # Forces browsers to use HTTPS for future requests
        if self.strict_transport_security and self.settings.SESSION_COOKIE_SECURE:
            response.headers['Strict-Transport-Security'] = (
                'max-age=31536000; includeSubDomains; preload'
            )
        
        # 2. X-Content-Type-Options
        # Prevents MIME type sniffing
        if self.content_type_options:
            response.headers['X-Content-Type-Options'] = 'nosniff'
        
        # 3. X-Frame-Options
        # Prevents clickjacking attacks
        if self.frame_options:
            response.headers['X-Frame-Options'] = self.frame_options
        
        # 4. X-XSS-Protection
        # Enables browser's XSS filter (legacy, but still useful for older browsers)
        if self.xss_protection:
            response.headers['X-XSS-Protection'] = '1; mode=block'
        
        # 5. Content-Security-Policy
        # Controls which resources can be loaded
        if self.content_security_policy:
            response.headers['Content-Security-Policy'] = self.content_security_policy
        elif not self.content_security_policy:
            # Default restrictive CSP
            response.headers['Content-Security-Policy'] = (
                "default-src 'self'; "
                "script-src 'self' 'unsafe-inline' 'unsafe-eval'; "
                "style-src 'self' 'unsafe-inline'; "
                "img-src 'self' data: https:; "
                "font-src 'self' data:; "
                "connect-src 'self'; "
                "frame-ancestors 'none'; "
                "base-uri 'self'; "
                "form-action 'self'"
            )
        
        # 6. Referrer-Policy
        # Controls how much referrer information is sent
        if self.referrer_policy:
            response.headers['Referrer-Policy'] = self.referrer_policy
        
        # 7. Permissions-Policy (formerly Feature-Policy)
        # Controls which browser features can be used
        if self.permissions_policy:
            response.headers['Permissions-Policy'] = self.permissions_policy
        elif not self.permissions_policy:
            # Default restrictive permissions
            response.headers['Permissions-Policy'] = (
                "geolocation=(), "
                "microphone=(), "
                "camera=(), "
                "payment=(), "
                "usb=(), "
                "magnetometer=(), "
                "gyroscope=(), "
                "accelerometer=()"
            )
        
        # 8. Additional security headers
        # Remove server header if present (information disclosure)
        if 'Server' in response.headers:
            del response.headers['Server']
        
        # Add X-Permitted-Cross-Domain-Policies
        response.headers['X-Permitted-Cross-Domain-Policies'] = 'none'
        
        # Add custom headers if provided
        for header_name, header_value in self.custom_headers.items():
            response.headers[header_name] = header_value
        
        return response


#######################################################################################################################
#
# Factory Function
#

def create_security_headers_middleware(
    app: ASGIApp,
    development_mode: bool = False
) -> SecurityHeadersMiddleware:
    """
    Create security headers middleware with appropriate settings.
    
    Args:
        app: The ASGI application
        development_mode: If True, use less restrictive settings for development
        
    Returns:
        Configured SecurityHeadersMiddleware instance
    """
    settings = get_settings()
    
    if development_mode:
        # Less restrictive settings for development
        return SecurityHeadersMiddleware(
            app,
            strict_transport_security=False,  # No HSTS in development
            content_type_options=True,
            frame_options="SAMEORIGIN",  # Allow framing from same origin
            xss_protection=True,
            content_security_policy=None,  # Use default CSP
            referrer_policy="strict-origin-when-cross-origin",
            permissions_policy=None,  # Use default permissions
        )
    else:
        # Production settings
        return SecurityHeadersMiddleware(
            app,
            strict_transport_security=True,
            content_type_options=True,
            frame_options="DENY",  # No framing allowed
            xss_protection=True,
            content_security_policy=(
                "default-src 'self'; "
                "script-src 'self'; "
                "style-src 'self' 'unsafe-inline'; "  # Allow inline styles for compatibility
                "img-src 'self' data: https:; "
                "font-src 'self' data:; "
                "connect-src 'self'; "
                "frame-ancestors 'none'; "
                "base-uri 'self'; "
                "form-action 'self'; "
                "upgrade-insecure-requests"
            ),
            referrer_policy="strict-origin-when-cross-origin",
            permissions_policy=(
                "geolocation=(), "
                "microphone=(), "
                "camera=(), "
                "payment=(), "
                "usb=(), "
                "magnetometer=(), "
                "gyroscope=(), "
                "accelerometer=()"
            ),
        )


#######################################################################################################################
#
# Usage Instructions
#

"""
To use this middleware in your FastAPI application, add it in your main.py:

from tldw_Server_API.app.core.AuthNZ.security_headers import create_security_headers_middleware

app = FastAPI()

# Add security headers middleware
app.add_middleware(
    SecurityHeadersMiddleware,
    strict_transport_security=True,
    content_type_options=True,
    frame_options="DENY",
    xss_protection=True
)

# Or use the factory function
middleware = create_security_headers_middleware(app, development_mode=False)
app.add_middleware(middleware.__class__, **middleware.__dict__)
"""

#
# End of security_headers.py
#######################################################################################################################