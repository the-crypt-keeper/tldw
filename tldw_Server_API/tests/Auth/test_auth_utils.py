# test_auth_utils.py
# Unit tests for authentication utilities

import os
import pytest
from unittest.mock import patch, MagicMock

from tldw_Server_API.app.core.Auth.auth_utils import (
    constant_time_compare,
    extract_bearer_token,
    validate_api_token,
    get_expected_api_token,
    is_authentication_required,
)


class TestConstantTimeCompare:
    """Test constant-time string comparison function."""
    
    def test_equal_strings(self):
        """Test that equal strings return True."""
        assert constant_time_compare("test123", "test123") is True
        assert constant_time_compare("", "") is True
        assert constant_time_compare("a" * 1000, "a" * 1000) is True
    
    def test_different_strings(self):
        """Test that different strings return False."""
        assert constant_time_compare("test123", "test124") is False
        assert constant_time_compare("test", "test123") is False
        assert constant_time_compare("", "test") is False
    
    def test_unicode_strings(self):
        """Test that unicode strings are handled correctly."""
        assert constant_time_compare("tëst", "tëst") is True
        assert constant_time_compare("tëst", "test") is False
        assert constant_time_compare("🔐", "🔐") is True
        assert constant_time_compare("🔐", "🔑") is False


class TestExtractBearerToken:
    """Test bearer token extraction from authorization headers."""
    
    def test_valid_bearer_token(self):
        """Test extraction from valid Bearer header."""
        token = extract_bearer_token("Bearer test-token-123")
        assert token == "test-token-123"
    
    def test_bearer_with_extra_spaces(self):
        """Test extraction handles extra spaces."""
        token = extract_bearer_token("Bearer   test-token-123   ")
        assert token == "test-token-123"
    
    def test_missing_bearer_prefix(self):
        """Test that missing Bearer prefix returns None."""
        assert extract_bearer_token("test-token-123") is None
        assert extract_bearer_token("Basic test-token-123") is None
    
    def test_empty_token_after_bearer(self):
        """Test that empty token after Bearer returns None."""
        assert extract_bearer_token("Bearer ") is None
        assert extract_bearer_token("Bearer") is None
    
    def test_none_or_empty_header(self):
        """Test that None or empty header returns None."""
        assert extract_bearer_token(None) is None
        assert extract_bearer_token("") is None
    
    def test_case_sensitive_bearer(self):
        """Test that Bearer prefix is case-sensitive."""
        # The function should handle the exact case from config
        with patch('tldw_Server_API.app.core.Auth.auth_utils.AUTH_BEARER_PREFIX', 'Bearer '):
            assert extract_bearer_token("bearer test-token") is None
            assert extract_bearer_token("BEARER test-token") is None


class TestValidateApiToken:
    """Test API token validation."""
    
    def test_valid_matching_tokens(self):
        """Test that matching tokens validate correctly."""
        assert validate_api_token("token123", "token123") is True
    
    def test_different_tokens(self):
        """Test that different tokens fail validation."""
        assert validate_api_token("token123", "token456") is False
    
    def test_none_tokens(self):
        """Test that None tokens fail validation."""
        assert validate_api_token(None, "token123") is False
        assert validate_api_token("token123", None) is False
        assert validate_api_token(None, None) is False
    
    def test_empty_tokens(self):
        """Test that empty tokens fail validation."""
        assert validate_api_token("", "token123") is False
        assert validate_api_token("token123", "") is False
        # Empty strings should still fail even if both empty
        assert validate_api_token("", "") is False
    
    def test_numeric_tokens_converted(self):
        """Test that numeric tokens are converted to strings."""
        assert validate_api_token(123, "123") is True
        assert validate_api_token("123", 123) is True
        assert validate_api_token(123, 456) is False


class TestGetExpectedApiToken:
    """Test retrieval of expected API token from environment."""
    
    @patch.dict(os.environ, {"API_BEARER": "test-bearer-token"})
    def test_token_from_environment(self):
        """Test token retrieval when set in environment."""
        token = get_expected_api_token()
        assert token == "test-bearer-token"
    
    @patch.dict(os.environ, {}, clear=True)
    def test_no_token_in_environment(self):
        """Test token retrieval when not set in environment."""
        with patch('tldw_Server_API.app.core.Auth.auth_utils.logger') as mock_logger:
            token = get_expected_api_token()
            assert token is None
            mock_logger.warning.assert_called_once()
    
    @patch.dict(os.environ, {"API_BEARER": ""})
    def test_empty_token_in_environment(self):
        """Test token retrieval when set to empty string."""
        token = get_expected_api_token()
        assert token == ""


class TestIsAuthenticationRequired:
    """Test authentication requirement checking."""
    
    @patch.dict(os.environ, {"AUTH_MODE": "multi_user"})
    def test_multi_user_mode_requires_auth(self):
        """Test that multi-user mode requires authentication."""
        assert is_authentication_required() is True
    
    @patch.dict(os.environ, {"AUTH_MODE": "single_user", "API_BEARER": "token123"})
    def test_single_user_with_bearer_requires_auth(self):
        """Test that single-user mode with API_BEARER requires auth."""
        assert is_authentication_required() is True
    
    @patch.dict(os.environ, {"AUTH_MODE": "single_user"}, clear=True)
    def test_single_user_without_bearer_no_auth(self):
        """Test that single-user mode without API_BEARER doesn't require auth."""
        # Remove API_BEARER if it exists
        os.environ.pop("API_BEARER", None)
        assert is_authentication_required() is False
    
    @patch.dict(os.environ, {}, clear=True)
    def test_default_mode_without_bearer(self):
        """Test default mode (single_user) without bearer."""
        assert is_authentication_required() is False
    
    @patch.dict(os.environ, {"API_BEARER": "token123"}, clear=True)
    def test_default_mode_with_bearer(self):
        """Test default mode with bearer token set."""
        # Default mode is single_user when AUTH_MODE not set
        assert is_authentication_required() is True


class TestIntegration:
    """Integration tests for auth utilities working together."""
    
    @patch.dict(os.environ, {"API_BEARER": "secret-token-123", "AUTH_MODE": "multi_user"})
    def test_full_auth_flow_success(self):
        """Test successful authentication flow."""
        # Check auth is required
        assert is_authentication_required() is True
        
        # Extract token from header
        auth_header = "Bearer secret-token-123"
        extracted = extract_bearer_token(auth_header)
        assert extracted == "secret-token-123"
        
        # Get expected token
        expected = get_expected_api_token()
        assert expected == "secret-token-123"
        
        # Validate token
        assert validate_api_token(extracted, expected) is True
    
    @patch.dict(os.environ, {"API_BEARER": "secret-token-123", "AUTH_MODE": "multi_user"})
    def test_full_auth_flow_failure_wrong_token(self):
        """Test failed authentication with wrong token."""
        # Check auth is required
        assert is_authentication_required() is True
        
        # Extract wrong token from header
        auth_header = "Bearer wrong-token"
        extracted = extract_bearer_token(auth_header)
        assert extracted == "wrong-token"
        
        # Get expected token
        expected = get_expected_api_token()
        assert expected == "secret-token-123"
        
        # Validate token fails
        assert validate_api_token(extracted, expected) is False
    
    @patch.dict(os.environ, {"API_BEARER": "secret-token-123", "AUTH_MODE": "multi_user"})
    def test_full_auth_flow_failure_no_bearer(self):
        """Test failed authentication with missing Bearer prefix."""
        # Check auth is required
        assert is_authentication_required() is True
        
        # Try to extract without Bearer prefix
        auth_header = "secret-token-123"
        extracted = extract_bearer_token(auth_header)
        assert extracted is None
        
        # Get expected token
        expected = get_expected_api_token()
        assert expected == "secret-token-123"
        
        # Validation would fail with None token
        assert validate_api_token(extracted, expected) is False


if __name__ == "__main__":
    pytest.main([__file__, "-v"])