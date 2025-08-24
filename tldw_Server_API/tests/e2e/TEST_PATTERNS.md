# E2E Test Patterns and Best Practices

## Overview

This document outlines common patterns, best practices, and reusable code snippets used throughout the E2E test suite. These patterns ensure consistency, maintainability, and reliability across all tests.

## Table of Contents

1. [Authentication Patterns](#authentication-patterns)
2. [Response Handling Patterns](#response-handling-patterns)
3. [Error Handling Patterns](#error-handling-patterns)
4. [Resource Management Patterns](#resource-management-patterns)
5. [Test Data Patterns](#test-data-patterns)
6. [Assertion Patterns](#assertion-patterns)
7. [Performance Testing Patterns](#performance-testing-patterns)
8. [Anti-Patterns to Avoid](#anti-patterns-to-avoid)

## Authentication Patterns

### Pattern 1: Conditional Authentication Based on Mode

```python
@pytest.fixture(scope="session")
async def authenticated_client(api_client):
    """Get authenticated client based on auth mode"""
    # Check auth mode from health endpoint
    health = await api_client.get("/health")
    auth_mode = health.get("auth_mode", "single_user")
    
    if auth_mode == "single_user":
        # Use X-API-KEY for single-user mode
        from tldw_Server_API.app.core.AuthNZ.settings import get_settings
        settings = get_settings()
        api_key = settings.SINGLE_USER_API_KEY
        api_client.set_auth_token(api_key)
    else:
        # Multi-user mode: register and login
        user_data = {
            "username": f"test_{uuid.uuid4().hex[:8]}",
            "password": "TestPassword123!",
            "email": f"test_{uuid.uuid4().hex[:8]}@example.com"
        }
        
        # Register
        await api_client.post("/auth/register", json=user_data)
        
        # Login
        login_response = await api_client.post("/auth/login", json={
            "username": user_data["username"],
            "password": user_data["password"]
        })
        
        token = login_response["access_token"]
        api_client.set_auth_token(token)
    
    return api_client
```

### Pattern 2: Dual Header Authentication

Some endpoints require both X-API-KEY and Token headers:

```python
def set_auth_headers(self, token: str):
    """Set both authentication headers for maximum compatibility"""
    self.client.headers.update({
        "X-API-KEY": token,
        "Token": token  # Capital T is important for some endpoints
    })
```

### Pattern 3: Token Refresh Handling

```python
async def make_request_with_refresh(self, method, endpoint, **kwargs):
    """Make request with automatic token refresh on 401"""
    response = await self.request(method, endpoint, **kwargs)
    
    if response.status_code == 401 and self.refresh_token:
        # Refresh the token
        refresh_response = await self.post("/auth/refresh", json={
            "refresh_token": self.refresh_token
        })
        
        if refresh_response.status_code == 200:
            new_token = refresh_response.json()["access_token"]
            self.set_auth_token(new_token)
            # Retry the original request
            response = await self.request(method, endpoint, **kwargs)
    
    return response
```

## Response Handling Patterns

### Pattern 1: Adaptive Response Format Handling

APIs may return data in different formats. Handle multiple possibilities:

```python
async def extract_items_from_response(response):
    """Extract items from various response formats"""
    data = response.json()
    
    # Handle different response structures
    if isinstance(data, list):
        # Direct list response
        return data
    elif isinstance(data, dict):
        # Check for common wrapper keys
        for key in ['results', 'items', 'data', 'content']:
            if key in data:
                return data[key] if isinstance(data[key], list) else [data[key]]
        # Single item as dict
        return [data]
    else:
        return []
```

### Pattern 2: Safe Nested Value Extraction

```python
def safe_get(data, *keys, default=None):
    """Safely navigate nested dictionaries"""
    for key in keys:
        if isinstance(data, dict):
            data = data.get(key)
            if data is None:
                return default
        else:
            return default
    return data

# Usage
media_id = safe_get(response_data, 'results', 0, 'media_id')
```

### Pattern 3: Response Validation

```python
def validate_response_schema(response, expected_schema):
    """Validate response matches expected schema"""
    data = response.json()
    
    for field, field_type in expected_schema.items():
        assert field in data, f"Missing required field: {field}"
        assert isinstance(data[field], field_type), \
            f"Field {field} should be {field_type.__name__}, got {type(data[field]).__name__}"
    
    return True

# Usage
expected_schema = {
    'id': int,
    'title': str,
    'created_at': str,
    'status': str
}
validate_response_schema(response, expected_schema)
```

## Error Handling Patterns

### Pattern 1: Graceful Degradation

```python
async def test_with_fallback(self):
    """Test with fallback behavior on failure"""
    try:
        # Try primary operation
        response = await self.api_client.post("/media/transcribe", 
                                             json={"url": test_url})
        assert response.status_code == 200
        result = response.json()
    except AssertionError:
        # Fallback to alternative method
        self.logger.warning("Primary transcription failed, trying alternative")
        response = await self.api_client.post("/media/process",
                                             json={"url": test_url, "skip_transcription": True})
        assert response.status_code == 200
        result = response.json()
    
    return result
```

### Pattern 2: Detailed Error Reporting

```python
def assert_success_response(response, expected_status=200):
    """Assert response is successful with detailed error reporting"""
    if response.status_code != expected_status:
        # Provide detailed error information
        error_details = {
            "status_code": response.status_code,
            "headers": dict(response.headers),
            "body": response.text[:500] if response.text else None,
            "url": str(response.url),
            "method": response.request.method
        }
        
        pytest.fail(
            f"Expected status {expected_status}, got {response.status_code}\n"
            f"Details: {json.dumps(error_details, indent=2)}"
        )
```

### Pattern 3: Retry with Exponential Backoff

```python
async def retry_with_backoff(func, max_attempts=3, initial_delay=1):
    """Retry function with exponential backoff"""
    last_exception = None
    delay = initial_delay
    
    for attempt in range(max_attempts):
        try:
            return await func()
        except Exception as e:
            last_exception = e
            if attempt < max_attempts - 1:
                await asyncio.sleep(delay)
                delay *= 2  # Exponential backoff
            else:
                raise last_exception
```

## Resource Management Patterns

### Pattern 1: Resource Tracking for Cleanup

```python
class ResourceTracker:
    """Track created resources for cleanup"""
    
    def __init__(self):
        self.resources = defaultdict(list)
    
    def track(self, resource_type: str, resource_id: str, metadata: dict = None):
        """Track a created resource"""
        self.resources[resource_type].append({
            'id': resource_id,
            'created_at': datetime.now(),
            'metadata': metadata or {}
        })
    
    async def cleanup(self, api_client):
        """Clean up all tracked resources in reverse order"""
        cleanup_order = ['notes', 'media', 'prompts', 'users']
        
        for resource_type in cleanup_order:
            if resource_type in self.resources:
                for resource in reversed(self.resources[resource_type]):
                    try:
                        await api_client.delete(f"/{resource_type}/{resource['id']}")
                    except Exception as e:
                        logger.warning(f"Failed to cleanup {resource_type} {resource['id']}: {e}")
```

### Pattern 2: Context Manager for Temporary Resources

```python
@contextmanager
async def temporary_media(api_client, media_data):
    """Context manager for temporary media that auto-cleans"""
    media_id = None
    try:
        # Create media
        response = await api_client.post("/media/add", json=media_data)
        media_id = response.json()['media_id']
        yield media_id
    finally:
        # Cleanup
        if media_id:
            try:
                await api_client.delete(f"/media/{media_id}")
            except:
                pass  # Best effort cleanup

# Usage
async with temporary_media(api_client, test_data) as media_id:
    # Use media_id for testing
    response = await api_client.get(f"/media/{media_id}")
    assert response.status_code == 200
```

## Test Data Patterns

### Pattern 1: Deterministic Test Data Generation

```python
class TestDataBuilder:
    """Build test data with deterministic values"""
    
    def __init__(self, seed=None):
        self.seed = seed or "test"
        self.counter = 0
    
    def next_id(self):
        """Generate next deterministic ID"""
        self.counter += 1
        return f"{self.seed}_{self.counter}_{hashlib.md5(f'{self.seed}{self.counter}'.encode()).hexdigest()[:8]}"
    
    def build_document(self, **overrides):
        """Build test document with defaults"""
        doc_id = self.next_id()
        defaults = {
            'title': f'Test Document {doc_id}',
            'content': f'Test content for {doc_id}',
            'tags': ['test', 'automated'],
            'metadata': {
                'test_id': doc_id,
                'created_by': 'test_suite'
            }
        }
        defaults.update(overrides)
        return defaults
```

### Pattern 2: Parameterized Test Data

```python
@pytest.mark.parametrize("media_type,content,expected_status", [
    ("text", "Simple text content", 200),
    ("markdown", "# Markdown\n\nContent", 200),
    ("html", "<html><body>HTML content</body></html>", 200),
    ("invalid", "", 422),
])
async def test_media_types(api_client, media_type, content, expected_status):
    """Test different media types"""
    response = await api_client.post("/media/add", json={
        "type": media_type,
        "content": content
    })
    assert response.status_code == expected_status
```

## Assertion Patterns

### Pattern 1: Soft Assertions for Complete Validation

```python
class SoftAssertions:
    """Collect multiple assertion failures"""
    
    def __init__(self):
        self.failures = []
    
    def assert_equal(self, actual, expected, message=""):
        """Soft assert equality"""
        try:
            assert actual == expected, message
        except AssertionError as e:
            self.failures.append(str(e))
    
    def assert_all(self):
        """Raise all collected failures"""
        if self.failures:
            raise AssertionError("\n".join(self.failures))

# Usage
soft = SoftAssertions()
soft.assert_equal(response.status_code, 200, "Status code mismatch")
soft.assert_equal(len(data['items']), 10, "Item count mismatch")
soft.assert_equal(data['total'], 100, "Total count mismatch")
soft.assert_all()  # Raises with all failures
```

### Pattern 2: Custom Assertions

```python
def assert_datetime_recent(datetime_str, max_age_seconds=60):
    """Assert datetime string is recent"""
    dt = datetime.fromisoformat(datetime_str.replace('Z', '+00:00'))
    age = (datetime.now(timezone.utc) - dt).total_seconds()
    assert age <= max_age_seconds, \
        f"Datetime {datetime_str} is {age}s old, expected < {max_age_seconds}s"

def assert_contains_subset(actual, expected):
    """Assert actual dict contains all expected key-value pairs"""
    for key, value in expected.items():
        assert key in actual, f"Missing key: {key}"
        assert actual[key] == value, \
            f"Value mismatch for {key}: expected {value}, got {actual[key]}"
```

## Performance Testing Patterns

### Pattern 1: Response Time Tracking

```python
class PerformanceTracker:
    """Track and assert performance metrics"""
    
    def __init__(self):
        self.metrics = defaultdict(list)
    
    @contextmanager
    def measure(self, operation_name):
        """Measure operation duration"""
        start = time.perf_counter()
        try:
            yield
        finally:
            duration = time.perf_counter() - start
            self.metrics[operation_name].append(duration)
    
    def assert_performance(self, operation_name, max_duration):
        """Assert operation performance"""
        durations = self.metrics[operation_name]
        if durations:
            avg_duration = sum(durations) / len(durations)
            assert avg_duration <= max_duration, \
                f"{operation_name} avg duration {avg_duration:.2f}s exceeds {max_duration}s"

# Usage
tracker = PerformanceTracker()

with tracker.measure("media_upload"):
    response = await api_client.post("/media/add", files=files)

tracker.assert_performance("media_upload", max_duration=5.0)
```

### Pattern 2: Load Testing

```python
async def concurrent_requests(api_client, endpoint, num_requests=10):
    """Test concurrent request handling"""
    async def make_request(i):
        start = time.perf_counter()
        response = await api_client.get(f"{endpoint}?page={i}")
        duration = time.perf_counter() - start
        return response.status_code, duration
    
    # Execute concurrent requests
    tasks = [make_request(i) for i in range(num_requests)]
    results = await asyncio.gather(*tasks, return_exceptions=True)
    
    # Analyze results
    successful = sum(1 for r in results if isinstance(r, tuple) and r[0] == 200)
    avg_time = sum(r[1] for r in results if isinstance(r, tuple)) / len(results)
    
    assert successful >= num_requests * 0.95, \
        f"Only {successful}/{num_requests} requests succeeded"
    assert avg_time < 2.0, \
        f"Average response time {avg_time:.2f}s exceeds threshold"
```

## Anti-Patterns to Avoid

### Anti-Pattern 1: Hardcoded Wait Times

**Bad:**
```python
async def test_processing():
    response = await api_client.post("/media/process", json=data)
    await asyncio.sleep(10)  # Bad: arbitrary wait
    result = await api_client.get(f"/media/{media_id}")
```

**Good:**
```python
async def test_processing():
    response = await api_client.post("/media/process", json=data)
    
    # Poll with timeout
    timeout = time.time() + 30
    while time.time() < timeout:
        result = await api_client.get(f"/media/{media_id}/status")
        if result.json()['status'] == 'completed':
            break
        await asyncio.sleep(1)
    else:
        pytest.fail("Processing timeout")
```

### Anti-Pattern 2: Ignoring Cleanup Failures

**Bad:**
```python
def cleanup():
    try:
        api_client.delete(f"/media/{media_id}")
    except:
        pass  # Bad: silently ignoring all errors
```

**Good:**
```python
def cleanup():
    try:
        response = api_client.delete(f"/media/{media_id}")
        if response.status_code not in [200, 204, 404]:
            logger.warning(f"Unexpected cleanup status: {response.status_code}")
    except RequestException as e:
        logger.warning(f"Cleanup failed: {e}")
        # Don't fail the test, but log for investigation
```

### Anti-Pattern 3: Tight Coupling to Implementation

**Bad:**
```python
def test_internal_state():
    # Bad: testing internal implementation details
    response = await api_client.get("/internal/cache/size")
    assert response.json()['size'] == 1024
```

**Good:**
```python
def test_behavior():
    # Good: testing observable behavior
    response = await api_client.get("/media/search?q=test")
    assert response.status_code == 200
    assert len(response.json()['results']) > 0
```

### Anti-Pattern 4: Test Interdependencies

**Bad:**
```python
class TestSuite:
    def test_1_create(self):
        self.media_id = create_media()  # Bad: storing state
    
    def test_2_update(self):
        update_media(self.media_id)  # Bad: depends on test_1
```

**Good:**
```python
class TestSuite:
    @pytest.fixture
    def media_id(self, api_client):
        # Good: isolated fixture
        media_id = create_media(api_client)
        yield media_id
        cleanup_media(api_client, media_id)
    
    def test_update(self, media_id):
        # Good: independent test
        update_media(media_id)
```

## Best Practices Summary

### DO:
- ✅ Use fixtures for setup/teardown
- ✅ Make tests independent and idempotent
- ✅ Track resources for cleanup
- ✅ Handle multiple response formats
- ✅ Provide detailed error messages
- ✅ Use deterministic test data
- ✅ Test behavior, not implementation
- ✅ Use appropriate timeouts
- ✅ Log warnings for non-critical failures

### DON'T:
- ❌ Use hardcoded sleep/wait times
- ❌ Create test dependencies
- ❌ Ignore cleanup failures silently
- ❌ Test internal implementation details
- ❌ Use random data without seeds
- ❌ Assume response formats
- ❌ Mix test and production data
- ❌ Leave resources after tests

## Code Snippets Library

### Complete Test Template

```python
import pytest
from datetime import datetime
from typing import Dict, Any

class TestFeatureComplete:
    """Complete test class template"""
    
    @pytest.fixture(autouse=True)
    def setup(self, authenticated_client, data_tracker):
        """Setup for each test"""
        self.client = authenticated_client
        self.tracker = data_tracker
        self.test_data = TestDataBuilder()
    
    async def test_feature_workflow(self):
        """Test complete feature workflow"""
        # Arrange
        test_input = self.test_data.build_document()
        
        # Act - Create
        create_response = await self.client.post("/feature", json=test_input)
        assert create_response.status_code == 201
        feature_id = create_response.json()['id']
        self.tracker.track('feature', feature_id)
        
        # Act - Read
        read_response = await self.client.get(f"/feature/{feature_id}")
        assert read_response.status_code == 200
        feature_data = read_response.json()
        
        # Assert - Validate
        assert feature_data['id'] == feature_id
        assert_contains_subset(feature_data, test_input)
        assert_datetime_recent(feature_data['created_at'])
        
        # Act - Update
        update_data = {"status": "updated"}
        update_response = await self.client.patch(
            f"/feature/{feature_id}", 
            json=update_data
        )
        assert update_response.status_code == 200
        
        # Act - Delete
        delete_response = await self.client.delete(f"/feature/{feature_id}")
        assert delete_response.status_code in [200, 204]
        
        # Assert - Verify deletion
        verify_response = await self.client.get(f"/feature/{feature_id}")
        assert verify_response.status_code == 404
```

This patterns document provides reusable code and best practices that can be applied across all E2E tests, ensuring consistency and maintainability.