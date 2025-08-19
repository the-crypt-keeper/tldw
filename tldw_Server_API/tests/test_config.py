"""
Centralized test configuration for tldw_server tests.

This module provides secure test configuration without hardcoded secrets.
"""

import os
import secrets
from typing import Dict, Any


class TestConfig:
    """Test configuration with secure defaults."""
    
    # Generate unique test API key for this test session
    # Can be overridden with TEST_API_KEY environment variable
    TEST_API_KEY = os.environ.get("TEST_API_KEY", f"test-key-{secrets.token_hex(16)}")
    
    # Generate unique SK key for OpenAI compatibility testing
    TEST_SK_KEY = os.environ.get("TEST_SK_KEY", f"sk-test-{secrets.token_hex(16)}")
    
    # Test database path
    TEST_DB_PATH = os.environ.get("TEST_DB_PATH", ":memory:")
    
    # Test authentication mode
    AUTH_MODE = "single_user"
    
    # Disable CSRF for testing
    CSRF_ENABLED = False
    
    @classmethod
    def setup_test_environment(cls) -> None:
        """Set up environment variables for testing."""
        os.environ["AUTH_MODE"] = cls.AUTH_MODE
        os.environ["API_BEARER"] = cls.TEST_API_KEY
        os.environ["SINGLE_USER_API_KEY"] = cls.TEST_API_KEY
        
        # Disable CSRF for testing
        try:
            from tldw_Server_API.app.core.AuthNZ.csrf_protection import global_settings
            global_settings['CSRF_ENABLED'] = cls.CSRF_ENABLED
        except ImportError:
            pass
    
    @classmethod
    def get_auth_headers(cls) -> Dict[str, str]:
        """Get authentication headers for API testing."""
        return {"Authorization": f"Bearer {cls.TEST_API_KEY}"}
    
    @classmethod
    def get_sk_auth_headers(cls) -> Dict[str, str]:
        """Get OpenAI-style authentication headers for API testing."""
        return {"Authorization": f"Bearer {cls.TEST_SK_KEY}"}
    
    @classmethod
    def reset_settings(cls) -> None:
        """Reset authentication settings."""
        try:
            from tldw_Server_API.app.core.AuthNZ.settings import reset_settings
            reset_settings()
        except ImportError:
            pass


# Singleton instance
test_config = TestConfig()