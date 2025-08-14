"""
Unit tests for JWT service.
"""

import pytest
import jwt
from datetime import datetime, timedelta
from hypothesis import given, strategies as st, assume
from hypothesis.strategies import text, integers

from tldw_Server_API.app.core.AuthNZ.jwt_service import JWTService
from tldw_Server_API.app.core.AuthNZ.settings import Settings
from tldw_Server_API.app.core.AuthNZ.exceptions import (
    InvalidTokenError,
    TokenExpiredError
)


class TestJWTServiceUnit:
    """Unit tests for JWT service."""
    
    def test_create_access_token(self, jwt_service):
        """Test creating an access token."""
        token = jwt_service.create_access_token(
            user_id=1,
            username="testuser",
            role="user"
        )
        
        assert token is not None
        assert isinstance(token, str)
        
        # Decode and verify
        payload = jwt.decode(
            token,
            jwt_service.settings.JWT_SECRET_KEY,
            algorithms=[jwt_service.settings.JWT_ALGORITHM]
        )
        
        assert payload["sub"] == "1"  # JWT service stores user_id as string in 'sub'
        assert payload["username"] == "testuser"
        assert payload["role"] == "user"
        assert payload["type"] == "access"
        assert "exp" in payload
        assert "iat" in payload
        assert "jti" in payload
    
    def test_create_refresh_token(self, jwt_service):
        """Test creating a refresh token."""
        token = jwt_service.create_refresh_token(user_id=1, username="testuser")
        
        assert token is not None
        assert isinstance(token, str)
        
        # Decode and verify
        payload = jwt.decode(
            token,
            jwt_service.settings.JWT_SECRET_KEY,
            algorithms=[jwt_service.settings.JWT_ALGORITHM]
        )
        
        assert payload["sub"] == "1"  # JWT service stores user_id as string in 'sub'
        assert payload["type"] == "refresh"
        assert "exp" in payload
        assert "iat" in payload
        assert "jti" in payload
    
    def test_decode_access_token_valid(self, jwt_service):
        """Test decoding a valid access token."""
        token = jwt_service.create_access_token(
            user_id=1,
            username="testuser",
            role="user"
        )
        
        payload = jwt_service.decode_access_token(token)
        
        assert payload["sub"] == "1"  # JWT service stores user_id as string in 'sub'
        assert payload["username"] == "testuser"
        assert payload["role"] == "user"
        assert payload["type"] == "access"
    
    def test_decode_access_token_expired(self, jwt_service):
        """Test decoding an expired access token."""
        # Create expired token
        original_expire = jwt_service.settings.ACCESS_TOKEN_EXPIRE_MINUTES
        jwt_service.settings.ACCESS_TOKEN_EXPIRE_MINUTES = -1
        
        token = jwt_service.create_access_token(
            user_id=1,
            username="testuser",
            role="user"
        )
        
        jwt_service.settings.ACCESS_TOKEN_EXPIRE_MINUTES = original_expire
        
        with pytest.raises(TokenExpiredError):
            jwt_service.decode_access_token(token)
    
    def test_decode_access_token_invalid(self, jwt_service):
        """Test decoding an invalid access token."""
        with pytest.raises(InvalidTokenError):
            jwt_service.decode_access_token("invalid.token.here")
    
    def test_decode_refresh_token_as_access(self, jwt_service):
        """Test that refresh tokens cannot be used as access tokens."""
        refresh_token = jwt_service.create_refresh_token(user_id=1)
        
        with pytest.raises(InvalidTokenError, match="Invalid token type"):
            jwt_service.decode_access_token(refresh_token)
    
    def test_decode_refresh_token_valid(self, jwt_service):
        """Test decoding a valid refresh token."""
        token = jwt_service.create_refresh_token(user_id=1, username="testuser")
        
        payload = jwt_service.decode_refresh_token(token)
        
        assert payload["sub"] == "1"  # JWT service stores user_id as string in 'sub'
        assert payload["type"] == "refresh"
    
    def test_token_with_additional_claims(self, jwt_service):
        """Test creating tokens with additional claims."""
        token = jwt_service.create_access_token(
            user_id=1,
            username="testuser",
            role="user",
            additional_claims={
                "session_id": "session-123",
                "custom_claim": "custom_value"
            }
        )
        
        payload = jwt_service.decode_access_token(token)
        
        assert payload["session_id"] == "session-123"
        assert payload["custom_claim"] == "custom_value"


class TestJWTServiceProperty:
    """Property-based tests for JWT service."""
    
    @given(
        user_id=integers(min_value=1, max_value=2**31-1),
        username=text(min_size=1, max_size=50).filter(lambda x: x.strip()),
        role=st.sampled_from(["user", "admin", "moderator"])
    )
    def test_access_token_roundtrip(self, jwt_service, user_id, username, role):
        """Test that access tokens can be created and decoded correctly."""
        token = jwt_service.create_access_token(
            user_id=user_id,
            username=username,
            role=role
        )
        
        payload = jwt_service.decode_access_token(token)
        
        assert payload["sub"] == str(user_id)  # Stored as string
        assert payload["username"] == username
        assert payload["role"] == role
        assert payload["type"] == "access"
    
    @given(user_id=integers(min_value=1, max_value=2**31-1))
    def test_refresh_token_roundtrip(self, jwt_service, user_id):
        """Test that refresh tokens can be created and decoded correctly."""
        token = jwt_service.create_refresh_token(user_id=user_id, username="testuser")
        
        payload = jwt_service.decode_refresh_token(token)
        
        assert payload["sub"] == str(user_id)  # Stored as string
        assert payload["type"] == "refresh"
    
    @given(
        secret_key=text(min_size=32, max_size=64),
        algorithm=st.sampled_from(["HS256", "HS384", "HS512"])
    )
    def test_jwt_settings_configuration(self, secret_key, algorithm):
        """Test JWT service with different settings."""
        settings = JWTSettings(
            SECRET_KEY=secret_key,
            ALGORITHM=algorithm,
            ACCESS_TOKEN_EXPIRE_MINUTES=30,
            REFRESH_TOKEN_EXPIRE_DAYS=7
        )
        
        service = JWTService(settings=settings)
        
        token = service.create_access_token(
            user_id=1,
            username="testuser",
            role="user"
        )
        
        payload = service.decode_access_token(token)
        
        assert payload["sub"] == "1"  # JWT service stores user_id as string in 'sub'
        assert payload["username"] == "testuser"
    
    @given(
        expire_minutes=integers(min_value=1, max_value=60*24),  # 1 minute to 1 day
        expire_days=integers(min_value=1, max_value=365)
    )
    def test_token_expiry_settings(self, expire_minutes, expire_days):
        """Test tokens with different expiry settings."""
        settings = JWTSettings(
            SECRET_KEY="test-key",
            ALGORITHM="HS256",
            ACCESS_TOKEN_EXPIRE_MINUTES=expire_minutes,
            REFRESH_TOKEN_EXPIRE_DAYS=expire_days
        )
        
        service = JWTService(settings=settings)
        
        access_token = service.create_access_token(
            user_id=1,
            username="testuser",
            role="user"
        )
        refresh_token = service.create_refresh_token(user_id=1)
        
        # Decode to check expiry
        access_payload = jwt.decode(
            access_token,
            settings.SECRET_KEY,
            algorithms=[settings.ALGORITHM]
        )
        refresh_payload = jwt.decode(
            refresh_token,
            settings.SECRET_KEY,
            algorithms=[settings.ALGORITHM]
        )
        
        # Check expiry times are set correctly
        now = datetime.utcnow()
        access_exp = datetime.fromtimestamp(access_payload["exp"])
        refresh_exp = datetime.fromtimestamp(refresh_payload["exp"])
        
        # Allow 5 seconds tolerance for test execution time
        assert access_exp <= now + timedelta(minutes=expire_minutes, seconds=5)
        assert access_exp >= now + timedelta(minutes=expire_minutes, seconds=-5)
        
        assert refresh_exp <= now + timedelta(days=expire_days, seconds=5)
        assert refresh_exp >= now + timedelta(days=expire_days, seconds=-5)