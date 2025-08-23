"""
Configuration for character tests.
Ensures proper authentication setup for single-user mode.
"""

import os
import pytest

# IMPORTANT: Ensure API_BEARER is not set - it causes wrong authentication path in single-user mode
if "API_BEARER" in os.environ:
    del os.environ["API_BEARER"]

@pytest.fixture(autouse=True)
def ensure_no_api_bearer():
    """Ensure API_BEARER is not set for character tests."""
    import os
    if "API_BEARER" in os.environ:
        del os.environ["API_BEARER"]
    yield
    # Clean up after test as well
    if "API_BEARER" in os.environ:
        del os.environ["API_BEARER"]

@pytest.fixture(autouse=True)
def setup_test_auth():
    """Set up authentication for tests."""
    from tldw_Server_API.tests.test_config import TestConfig
    from tldw_Server_API.app.core.AuthNZ.settings import reset_settings
    
    # Reset settings singleton to ensure clean state
    reset_settings()
    
    # Apply test configuration
    TestConfig.setup_test_environment()
    
    # Ensure we're in single-user mode with proper API key
    os.environ["AUTH_MODE"] = "single_user"
    os.environ["SINGLE_USER_API_KEY"] = TestConfig.TEST_API_KEY
    
    # Make sure API_BEARER is not set
    if "API_BEARER" in os.environ:
        del os.environ["API_BEARER"]
    
    yield
    
    # Clean up
    if "API_BEARER" in os.environ:
        del os.environ["API_BEARER"]
    
    # Reset settings after test
    reset_settings()