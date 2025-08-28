"""
Pytest configuration for e2e tests.

Defines custom markers and test configuration for end-to-end testing
that simulates real user interactions with the application.
"""

import pytest
import os
from typing import Dict, Any, Optional

# Disable rate limiting for all e2e tests
@pytest.fixture(autouse=True, scope="session")
def disable_rate_limiting():
    """Disable rate limiting for all tests in e2e suite"""
    os.environ["TESTING"] = "true"  # For embeddings endpoint
    os.environ["TEST_MODE"] = "true"  # For regular limiter
    yield
    # Clean up after tests
    if "TESTING" in os.environ:
        del os.environ["TESTING"]
    if "TEST_MODE" in os.environ:
        del os.environ["TEST_MODE"]

# Custom markers for conditional test execution
pytest_markers = [
    "single_user: marks tests that only run in single-user mode",
    "multi_user: marks tests that only run in multi-user mode",
    "requires_llm: marks tests that require a configured LLM provider",
    "requires_transcription: marks tests that require transcription services",
    "requires_embeddings: marks tests that require embedding services",
    "slow: marks tests that take a long time to run (>30s)",
    "critical: marks tests that are critical for basic functionality",
    "benchmark: marks benchmark-related tests",
    "media_processing: marks media processing tests",
    "auth: marks authentication-related tests",
    "timeout: marks tests with explicit timeout limits (requires pytest-timeout plugin)",
]

def pytest_configure(config):
    """Register custom markers."""
    for marker in pytest_markers:
        config.addinivalue_line("markers", marker)


def pytest_collection_modifyitems(config, items):
    """Modify test collection based on environment and configuration."""
    
    # Check environment to determine which tests to skip
    auth_mode = os.getenv("E2E_AUTH_MODE", "auto")  # auto, single_user, multi_user
    skip_slow = config.getoption("--skip-slow", default=False)
    run_critical_only = config.getoption("--critical-only", default=False)
    
    # Get auth mode from API if auto
    if auth_mode == "auto":
        auth_mode = _detect_auth_mode()
    
    for item in items:
        # Skip tests based on auth mode
        if auth_mode == "single_user":
            if "multi_user" in item.keywords:
                item.add_marker(pytest.mark.skip(
                    reason="Multi-user test skipped in single-user mode"
                ))
        elif auth_mode == "multi_user":
            if "single_user" in item.keywords:
                item.add_marker(pytest.mark.skip(
                    reason="Single-user test skipped in multi-user mode"
                ))
        
        # Skip slow tests if requested
        if skip_slow and "slow" in item.keywords:
            item.add_marker(pytest.mark.skip(
                reason="Slow test skipped (use --run-slow to include)"
            ))
        
        # Run only critical tests if requested
        if run_critical_only and "critical" not in item.keywords:
            item.add_marker(pytest.mark.skip(
                reason="Non-critical test skipped (--critical-only mode)"
            ))


def _detect_auth_mode() -> str:
    """Detect auth mode from running API server."""
    try:
        import httpx
        base_url = os.getenv("E2E_TEST_BASE_URL", "http://localhost:8000")
        with httpx.Client(timeout=5) as client:
            response = client.get(f"{base_url}/api/v1/health")
            if response.status_code == 200:
                return response.json().get("auth_mode", "multi_user")
    except:
        pass
    return "multi_user"  # Default


def pytest_addoption(parser):
    """Add custom command-line options."""
    parser.addoption(
        "--skip-slow",
        action="store_true",
        default=False,
        help="Skip slow-running tests"
    )
    parser.addoption(
        "--critical-only",
        action="store_true",
        default=False,
        help="Run only critical tests"
    )
    parser.addoption(
        "--auth-mode",
        action="store",
        default="auto",
        choices=["auto", "single_user", "multi_user"],
        help="Force specific auth mode for testing"
    )


# Session-scoped fixture to store test results
@pytest.fixture(scope="session")
def test_results():
    """Store test results across the session for reporting."""
    results = {
        "passed": [],
        "failed": [],
        "skipped": [],
        "errors": [],
    }
    return results


# Session-scoped fixture for shared media state
@pytest.fixture(scope="session")
def shared_media_state():
    """
    Session-scoped fixture for sharing media across all tests.
    This maintains state across test classes and modules.
    """
    return {
        "uploaded_media": {},  # media_id -> media_data mapping
        "generated_embeddings": set(),  # Set of media_ids with embeddings
        "test_users": {},  # user_id -> user_data mapping
        "api_clients": {},  # client_id -> api_client mapping
    }


# Class-scoped fixture for test workflow state
@pytest.fixture(scope="class")
def test_workflow_state(shared_media_state):
    """
    Class-scoped fixture for test workflows.
    Provides an interface to both class-local and shared session state.
    """
    class WorkflowState:
        def __init__(self):
            self.media_items = []  # Local to this test class
            self.current_user = None
            self.api_client = None
            self._shared = shared_media_state  # Reference to session state
            
        def add_media(self, media_id: int, media_data: Dict[str, Any]):
            """Add media to both local and shared state."""
            self.media_items.append(media_data)
            self._shared["uploaded_media"][media_id] = media_data
            
        def get_any_media(self) -> Optional[Dict[str, Any]]:
            """Get any available media, preferring local then shared."""
            if self.media_items:
                return self.media_items[0]
            elif self._shared["uploaded_media"]:
                return next(iter(self._shared["uploaded_media"].values()))
            return None
            
        def get_media_by_id(self, media_id: int) -> Optional[Dict[str, Any]]:
            """Get specific media by ID from shared state."""
            return self._shared["uploaded_media"].get(media_id)
            
        def mark_embeddings_generated(self, media_id: int):
            """Mark that embeddings have been generated for this media."""
            self._shared["generated_embeddings"].add(media_id)
            
        def has_embeddings(self, media_id: int) -> bool:
            """Check if embeddings have been generated for this media."""
            return media_id in self._shared["generated_embeddings"]
            
        def get_or_create_user(self, user_id: str, user_data: Dict[str, Any] = None):
            """Get or create a test user."""
            if user_id not in self._shared["test_users"]:
                self._shared["test_users"][user_id] = user_data or {}
            return self._shared["test_users"][user_id]
            
        def set_api_client(self, client_id: str, api_client):
            """Store an API client for reuse."""
            self._shared["api_clients"][client_id] = api_client
            self.api_client = api_client
            
        def get_api_client(self, client_id: str = "default"):
            """Get a stored API client."""
            return self._shared["api_clients"].get(client_id, self.api_client)
    
    return WorkflowState()


# Function-scoped fixture for ensuring media has embeddings
@pytest.fixture
def ensure_embeddings(test_workflow_state):
    """
    Fixture that returns a function to ensure embeddings exist for media.
    Usage in tests:
        ensure_embeddings(api_client, media_id)
    """
    async def _ensure_embeddings(api_client, media_id: int) -> bool:
        """Ensure embeddings are generated for the given media."""
        if test_workflow_state.has_embeddings(media_id):
            return True
            
        try:
            # Generate embeddings
            response = api_client.post(
                f"/api/v1/media/{media_id}/embeddings",
                json={"force_regenerate": False}
            )
            
            if response.status_code == 200:
                test_workflow_state.mark_embeddings_generated(media_id)
                return True
            else:
                print(f"Failed to generate embeddings: {response.text}")
                return False
        except Exception as e:
            print(f"Error generating embeddings: {e}")
            return False
    
    return _ensure_embeddings


# Hook to capture test results
@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_makereport(item, call):
    """Capture test results for custom reporting."""
    outcome = yield
    report = outcome.get_result()
    
    if report.when == "call":
        # Get the test_results fixture if available
        if hasattr(item, "funcargs") and "test_results" in item.funcargs:
            results = item.funcargs["test_results"]
            
            test_info = {
                "name": item.name,
                "nodeid": item.nodeid,
                "duration": report.duration,
                "markers": [m.name for m in item.iter_markers()],
            }
            
            if report.passed:
                results["passed"].append(test_info)
            elif report.failed:
                test_info["error"] = str(report.longrepr)
                results["failed"].append(test_info)
            elif report.skipped:
                test_info["reason"] = str(report.longrepr)
                results["skipped"].append(test_info)


def pytest_sessionfinish(session, exitstatus):
    """Generate test summary report at end of session."""
    # Only generate report if tests were run
    if hasattr(session.config, "_test_results"):
        results = session.config._test_results
        
        print("\n" + "="*70)
        print("E2E TEST SUMMARY - Real User Simulation")
        print("="*70)
        
        print(f"✅ Passed: {len(results['passed'])}")
        print(f"❌ Failed: {len(results['failed'])}")
        print(f"⏭️  Skipped: {len(results['skipped'])}")
        
        if results['failed']:
            print("\n⚠️  Failed Tests:")
            for test in results['failed']:
                print(f"  - {test['name']}")
        
        print("="*70)