# jwt_service.py
# Description: JWT token service with persistent secret management
#
# Imports
import hashlib
import secrets
from datetime import datetime, timedelta
from typing import Optional, Dict, Any
from uuid import uuid4
#
# 3rd-party imports
from jose import jwt, JWTError
from loguru import logger
#
# Local imports
from tldw_Server_API.app.core.AuthNZ.settings import Settings, get_settings
from tldw_Server_API.app.core.AuthNZ.exceptions import (
    InvalidTokenError,
    TokenExpiredError,
    ConfigurationError
)

#######################################################################################################################
#
# JWT Service Class

class JWTService:
    """Service for creating and verifying JWT tokens with persistent secret management"""
    
    def __init__(self, settings: Optional[Settings] = None):
        """Initialize JWT service"""
        self.settings = settings or get_settings()
        
        # Validate we're in multi-user mode
        if self.settings.AUTH_MODE != "multi_user":
            logger.warning("JWTService initialized in single-user mode - JWT features may not work correctly")
        
        # JWT configuration
        self.algorithm = self.settings.JWT_ALGORITHM
        # Note: We don't cache timedeltas to allow dynamic configuration changes during testing
        
        # Get the persistent secret key
        self.secret_key = self.settings.JWT_SECRET_KEY
        if not self.secret_key:
            raise ConfigurationError("JWT_SECRET_KEY", "JWT secret key not configured")
        
        logger.debug(f"JWTService initialized with algorithm: {self.algorithm}")
    
    def create_access_token(
        self,
        user_id: int,
        username: str,
        role: str,
        additional_claims: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Create an access token for a user
        
        Args:
            user_id: User's database ID
            username: User's username
            role: User's role
            additional_claims: Additional claims to include in token
            
        Returns:
            Encoded JWT access token
        """
        # Calculate expiration dynamically from settings
        expire = datetime.utcnow() + timedelta(minutes=self.settings.ACCESS_TOKEN_EXPIRE_MINUTES)
        
        # Build token payload
        payload = {
            "sub": str(user_id),  # Subject (user ID)
            "username": username,
            "role": role,
            "exp": expire,
            "iat": datetime.utcnow(),
            "jti": str(uuid4()),  # JWT ID for tracking
            "type": "access"
        }
        
        # Add any additional claims
        if additional_claims:
            payload.update(additional_claims)
        
        # Encode the token
        try:
            token = jwt.encode(payload, self.secret_key, algorithm=self.algorithm)
            logger.debug(f"Created access token for user {username} (ID: {user_id})")
            return token
            
        except Exception as e:
            logger.error(f"Failed to create access token: {e}")
            raise InvalidTokenError(f"Failed to create token: {e}")
    
    def create_refresh_token(
        self,
        user_id: int,
        username: str,
        additional_claims: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Create a refresh token for a user
        
        Args:
            user_id: User's database ID
            username: User's username
            additional_claims: Additional claims to include in token
            
        Returns:
            Encoded JWT refresh token
        """
        # Calculate expiration dynamically from settings
        expire = datetime.utcnow() + timedelta(days=self.settings.REFRESH_TOKEN_EXPIRE_DAYS)
        
        # Build token payload (minimal claims for refresh token)
        payload = {
            "sub": str(user_id),
            "username": username,
            "exp": expire,
            "iat": datetime.utcnow(),
            "jti": str(uuid4()),
            "type": "refresh"
        }
        
        # Add any additional claims
        if additional_claims:
            payload.update(additional_claims)
        
        # Encode the token
        try:
            token = jwt.encode(payload, self.secret_key, algorithm=self.algorithm)
            logger.debug(f"Created refresh token for user {username} (ID: {user_id})")
            return token
            
        except Exception as e:
            logger.error(f"Failed to create refresh token: {e}")
            raise InvalidTokenError(f"Failed to create token: {e}")
    
    def verify_token(self, token: str, token_type: Optional[str] = None) -> Dict[str, Any]:
        """
        Verify and decode a JWT token
        
        Args:
            token: JWT token to verify
            token_type: Expected token type ('access' or 'refresh')
            
        Returns:
            Decoded token payload
            
        Raises:
            InvalidTokenError: If token is invalid or malformed
            TokenExpiredError: If token has expired
        """
        try:
            # Decode the token
            payload = jwt.decode(
                token,
                self.secret_key,
                algorithms=[self.algorithm]
            )
            
            # Verify token type if specified
            if token_type and payload.get("type") != token_type:
                raise InvalidTokenError(f"Invalid token type. Expected {token_type}, got {payload.get('type')}")
            
            logger.debug(f"Token verified successfully for user ID: {payload.get('sub')}")
            return payload
            
        except jwt.ExpiredSignatureError:
            logger.debug("Token has expired")
            raise TokenExpiredError()
            
        except jwt.JWTClaimsError as e:
            logger.warning(f"JWT claims error: {e}")
            raise InvalidTokenError(f"Invalid token claims: {e}")
            
        except JWTError as e:
            logger.warning(f"JWT verification error: {e}")
            raise InvalidTokenError(f"Invalid token: {e}")
            
        except Exception as e:
            logger.error(f"Unexpected error verifying token: {e}")
            raise InvalidTokenError(f"Token verification failed: {e}")
    
    def decode_access_token(self, token: str) -> Dict[str, Any]:
        """
        Decode and verify an access token
        
        Args:
            token: Access token to decode
            
        Returns:
            Decoded token payload
            
        Raises:
            InvalidTokenError: If token is invalid
            ExpiredTokenError: If token has expired
        """
        return self.verify_token(token, token_type="access")
    
    def decode_refresh_token(self, token: str) -> Dict[str, Any]:
        """
        Decode and verify a refresh token
        
        Args:
            token: Refresh token to decode
            
        Returns:
            Decoded token payload
            
        Raises:
            InvalidTokenError: If token is invalid
            ExpiredTokenError: If token has expired
        """
        return self.verify_token(token, token_type="refresh")
    
    def hash_token(self, token: str) -> str:
        """
        Create a SHA256 hash of a token for storage
        
        Args:
            token: Token to hash
            
        Returns:
            SHA256 hash of the token
        """
        return hashlib.sha256(token.encode()).hexdigest()
    
    def extract_jti(self, token: str) -> Optional[str]:
        """
        Extract the JTI (JWT ID) from a token without full verification
        
        Args:
            token: JWT token
            
        Returns:
            JTI if present, None otherwise
        """
        try:
            # Decode without verification to get JTI
            unverified = jwt.get_unverified_claims(token)
            return unverified.get("jti")
        except Exception:
            return None
    
    def create_password_reset_token(
        self,
        user_id: int,
        email: str,
        expires_in_hours: int = 1
    ) -> str:
        """
        Create a password reset token
        
        Args:
            user_id: User's database ID
            email: User's email
            expires_in_hours: Token validity in hours
            
        Returns:
            Encoded password reset token
        """
        expire = datetime.utcnow() + timedelta(hours=expires_in_hours)
        
        payload = {
            "sub": str(user_id),
            "email": email,
            "exp": expire,
            "iat": datetime.utcnow(),
            "jti": str(uuid4()),
            "type": "password_reset"
        }
        
        try:
            token = jwt.encode(payload, self.secret_key, algorithm=self.algorithm)
            logger.info(f"Created password reset token for user {user_id}")
            return token
            
        except Exception as e:
            logger.error(f"Failed to create password reset token: {e}")
            raise InvalidTokenError(f"Failed to create token: {e}")
    
    def create_email_verification_token(
        self,
        user_id: int,
        email: str,
        expires_in_hours: int = 24
    ) -> str:
        """
        Create an email verification token
        
        Args:
            user_id: User's database ID
            email: Email to verify
            expires_in_hours: Token validity in hours
            
        Returns:
            Encoded email verification token
        """
        expire = datetime.utcnow() + timedelta(hours=expires_in_hours)
        
        payload = {
            "sub": str(user_id),
            "email": email,
            "exp": expire,
            "iat": datetime.utcnow(),
            "jti": str(uuid4()),
            "type": "email_verification"
        }
        
        try:
            token = jwt.encode(payload, self.secret_key, algorithm=self.algorithm)
            logger.info(f"Created email verification token for user {user_id}")
            return token
            
        except Exception as e:
            logger.error(f"Failed to create email verification token: {e}")
            raise InvalidTokenError(f"Failed to create token: {e}")
    
    def create_service_account_token(
        self,
        service_name: str,
        permissions: list,
        expires_in_days: int = 365
    ) -> str:
        """
        Create a long-lived token for service accounts
        
        Args:
            service_name: Name of the service
            permissions: List of permissions granted
            expires_in_days: Token validity in days
            
        Returns:
            Encoded service account token
        """
        expire = datetime.utcnow() + timedelta(days=expires_in_days)
        
        payload = {
            "sub": f"service:{service_name}",
            "service": service_name,
            "permissions": permissions,
            "exp": expire,
            "iat": datetime.utcnow(),
            "jti": str(uuid4()),
            "type": "service"
        }
        
        try:
            token = jwt.encode(payload, self.secret_key, algorithm=self.algorithm)
            logger.info(f"Created service account token for {service_name}")
            return token
            
        except Exception as e:
            logger.error(f"Failed to create service account token: {e}")
            raise InvalidTokenError(f"Failed to create token: {e}")
    
    def refresh_access_token(self, refresh_token: str) -> tuple[str, str]:
        """
        Create a new access token from a refresh token
        
        Args:
            refresh_token: Valid refresh token
            
        Returns:
            Tuple of (new_access_token, original_refresh_token)
            
        Raises:
            InvalidTokenError: If refresh token is invalid
        """
        # Verify the refresh token
        payload = self.verify_token(refresh_token, token_type="refresh")
        
        # Extract user information
        user_id = int(payload["sub"])
        username = payload.get("username", "")
        
        # Note: Role should be fetched from database in production
        # This is a simplified version
        role = payload.get("role", "user")
        
        # Create new access token
        new_access_token = self.create_access_token(
            user_id=user_id,
            username=username,
            role=role
        )
        
        logger.info(f"Refreshed access token for user {username}")
        
        # Return new access token and original refresh token
        # In production, you might want to rotate refresh tokens too
        return new_access_token, refresh_token
    
    def get_token_remaining_time(self, token: str) -> Optional[int]:
        """
        Get remaining time before token expires
        
        Args:
            token: JWT token
            
        Returns:
            Remaining seconds until expiration, None if invalid
        """
        try:
            payload = self.verify_token(token)
            exp = payload.get("exp")
            if exp:
                remaining = exp - datetime.utcnow().timestamp()
                return max(0, int(remaining))
            return None
            
        except (InvalidTokenError, TokenExpiredError):
            return None


#######################################################################################################################
#
# Module Functions for convenience

# Global instance
_jwt_service: Optional[JWTService] = None


def get_jwt_service() -> JWTService:
    """Get JWT service singleton instance"""
    global _jwt_service
    if not _jwt_service:
        _jwt_service = JWTService()
    return _jwt_service


def create_access_token(user_id: int, username: str, role: str) -> str:
    """Convenience function to create an access token"""
    return get_jwt_service().create_access_token(user_id, username, role)


def create_refresh_token(user_id: int, username: str) -> str:
    """Convenience function to create a refresh token"""
    return get_jwt_service().create_refresh_token(user_id, username)


def verify_token(token: str, token_type: Optional[str] = None) -> Dict[str, Any]:
    """Convenience function to verify a token"""
    return get_jwt_service().verify_token(token, token_type)


def hash_token(token: str) -> str:
    """Convenience function to hash a token"""
    return get_jwt_service().hash_token(token)


#
# End of jwt_service.py
#######################################################################################################################