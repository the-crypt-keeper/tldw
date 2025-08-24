# Guide to Extending the E2E Test Suite

## Table of Contents
1. [Quick Start](#quick-start)
2. [Adding Tests for New Endpoints](#adding-tests-for-new-endpoints)
3. [Testing New Features](#testing-new-features)
4. [Test Data Management](#test-data-management)
5. [Advanced Test Scenarios](#advanced-test-scenarios)
6. [Integration Checklist](#integration-checklist)
7. [Templates and Examples](#templates-and-examples)

---

## Quick Start

### When to Add New Tests
Add E2E tests when:
- ✅ New API endpoint is stable and documented
- ✅ Feature is complete and ready for production
- ✅ API contract is finalized
- ✅ Authentication requirements are clear
- ✅ Response format is standardized

### Prerequisites
Before adding new tests:
1. Understand the existing test structure
2. Review authentication patterns
3. Identify the appropriate test phase
4. Plan resource cleanup strategy

---

## Adding Tests for New Endpoints

### Step-by-Step Process

#### 1. Analyze the Endpoint

First, understand the endpoint thoroughly:

```python
# Document the endpoint details
"""
Endpoint: POST /api/v1/analytics/generate
Purpose: Generate analytics report for media items
Authentication: Required (X-API-KEY header)
Request Body: {
    "media_ids": [1, 2, 3],
    "report_type": "summary|detailed|comparison",
    "date_range": {"start": "2024-01-01", "end": "2024-12-31"}
}
Response: {
    "report_id": "uuid",
    "status": "processing|completed",
    "data": {...}
}
"""
```

#### 2. Add API Client Method

Add the method to `fixtures.py`:

```python
# In APIClient class
def generate_analytics(
    self, 
    media_ids: List[int],
    report_type: str = "summary",
    date_range: Optional[Dict[str, str]] = None
) -> Dict[str, Any]:
    """Generate analytics report for media items."""
    data = {
        "media_ids": media_ids,
        "report_type": report_type
    }
    if date_range:
        data["date_range"] = date_range
    
    response = self.client.post(
        f"{API_PREFIX}/analytics/generate",
        json=data
    )
    response.raise_for_status()
    return response.json()

def get_analytics_report(self, report_id: str) -> Dict[str, Any]:
    """Get analytics report by ID."""
    response = self.client.get(f"{API_PREFIX}/analytics/{report_id}")
    response.raise_for_status()
    return response.json()
```

#### 3. Create Test Data Generator

Add to `test_data.py`:

```python
@staticmethod
def sample_analytics_request() -> Dict[str, Any]:
    """Generate sample analytics request data."""
    return {
        "report_type": random.choice(["summary", "detailed", "comparison"]),
        "date_range": {
            "start": (datetime.now() - timedelta(days=30)).isoformat(),
            "end": datetime.now().isoformat()
        },
        "options": {
            "include_transcripts": True,
            "include_metadata": True,
            "format": "json"
        }
    }

@staticmethod
def sample_analytics_filters() -> Dict[str, Any]:
    """Generate sample analytics filters."""
    return {
        "media_types": ["document", "audio", "video"],
        "tags": ["important", "reviewed"],
        "min_duration": 60,
        "max_duration": 3600
    }
```

#### 4. Write the Test

Add to appropriate test class:

```python
def test_85_analytics_generation(self, api_client, data_tracker):
    """Test analytics report generation."""
    # Prerequisite: Need media items
    if not self.media_items:
        pytest.skip("No media items available for analytics")
    
    # Get valid media IDs
    media_ids = []
    for item in self.media_items[:3]:  # Use first 3 items
        if "results" in item:
            for result in item["results"]:
                if result.get("db_id"):
                    media_ids.append(result["db_id"])
        elif item.get("id"):
            media_ids.append(item["id"])
    
    if not media_ids:
        pytest.skip("No valid media IDs for analytics")
    
    # Generate analytics request
    request_data = TestDataGenerator.sample_analytics_request()
    
    try:
        # Generate report
        response = api_client.generate_analytics(
            media_ids=media_ids,
            report_type=request_data["report_type"],
            date_range=request_data.get("date_range")
        )
        
        # Verify response
        assert "report_id" in response
        assert response.get("status") in ["processing", "completed"]
        
        # Track for cleanup
        if "report_id" in response:
            data_tracker.add_resource("analytics_report", response["report_id"])
        
        # Store for later tests
        self.analytics_reports.append(response)
        
        # If processing, wait and check status
        if response.get("status") == "processing":
            time.sleep(2)  # Wait for processing
            report = api_client.get_analytics_report(response["report_id"])
            assert report.get("status") == "completed"
            
    except httpx.HTTPStatusError as e:
        if e.response.status_code == 402:
            pytest.skip("Analytics feature requires premium subscription")
        else:
            raise
```

#### 5. Add Cleanup

Add to cleanup phase:

```python
def test_106_delete_analytics(self, api_client, data_tracker):
    """Clean up analytics reports."""
    for report in self.analytics_reports:
        report_id = report.get("report_id")
        if report_id:
            try:
                api_client.delete_analytics_report(report_id)
            except:
                pass  # Best effort cleanup
```

---

## Testing New Features

### Feature Stability Assessment

Before adding tests, evaluate feature stability:

```python
# Stability Checklist Class
class FeatureStability:
    def __init__(self, feature_name: str):
        self.feature_name = feature_name
        self.criteria = {
            "api_documented": False,
            "response_stable": False,
            "auth_clear": False,
            "errors_handled": False,
            "performance_acceptable": False,
            "backwards_compatible": False
        }
    
    def assess(self) -> bool:
        """Return True if feature is stable enough for E2E tests."""
        return all(self.criteria.values())
    
    def report(self) -> str:
        """Generate stability report."""
        stable = sum(self.criteria.values())
        total = len(self.criteria)
        return f"{self.feature_name}: {stable}/{total} criteria met"
```

### Integration Testing Strategy

#### Isolated Testing
First test the feature in isolation:

```python
class TestNewFeatureIsolated:
    """Test new feature in isolation before integration."""
    
    def test_feature_basic(self, api_client):
        """Test basic functionality."""
        response = api_client.new_feature(basic_params)
        assert response.get("success") == True
    
    def test_feature_error_handling(self, api_client):
        """Test error cases."""
        with pytest.raises(httpx.HTTPStatusError) as exc:
            api_client.new_feature(invalid_params)
        assert exc.value.response.status_code == 400
```

#### Integration Testing
Then integrate with main workflow:

```python
def test_86_new_feature_integrated(self, api_client):
    """Test new feature integrated with existing workflow."""
    # Use data from previous tests
    if not self.media_items:
        pytest.skip("Requires media items from earlier tests")
    
    # Test feature with real data
    response = api_client.new_feature(
        media_id=self.media_items[0]["id"],
        options=TestDataGenerator.sample_feature_options()
    )
    
    # Verify integration
    assert response.get("media_id") == self.media_items[0]["id"]
```

### Performance Testing

Add performance benchmarks:

```python
def test_87_feature_performance(self, api_client):
    """Test feature performance meets requirements."""
    start_time = time.time()
    
    # Perform operation
    response = api_client.new_feature(test_data)
    
    duration = time.time() - start_time
    
    # Check performance
    assert duration < 2.0, f"Operation too slow: {duration}s"
    
    # Store metric
    self.performance_metrics[f"new_feature"] = duration
```

---

## Test Data Management

### Dynamic Test Data Generation

Create flexible data generators:

```python
class DynamicTestData:
    """Generate test data based on context."""
    
    @staticmethod
    def generate_based_on_mode(auth_mode: str) -> Dict[str, Any]:
        """Generate data appropriate for auth mode."""
        if auth_mode == "single_user":
            return {
                "owner": "default",
                "shared": False,
                "permissions": ["all"]
            }
        else:
            return {
                "owner": f"user_{generate_unique_id()}",
                "shared": True,
                "permissions": ["read", "write"]
            }
    
    @staticmethod
    def generate_with_dependencies(media_items: List) -> Dict[str, Any]:
        """Generate data that depends on existing resources."""
        return {
            "media_ids": [item["id"] for item in media_items[:5]],
            "operation": "batch_process",
            "settings": {
                "quality": "high",
                "format": "json"
            }
        }
```

### Test Data Fixtures

Create reusable fixtures:

```python
@pytest.fixture(scope="class")
def sample_media_collection(api_client):
    """Create a collection of media items for testing."""
    items = []
    for i in range(5):
        content = TestDataGenerator.sample_text_content()
        file_path = create_test_file(content, suffix=f"_{i}.txt")
        
        response = api_client.upload_media(
            file_path=file_path,
            title=f"Test Media {i}",
            media_type="document"
        )
        items.append(response)
        cleanup_test_file(file_path)
    
    yield items
    
    # Cleanup
    for item in items:
        try:
            api_client.delete_media(item["id"])
        except:
            pass
```

### Data Cleanup Strategies

#### Immediate Cleanup
```python
def test_with_immediate_cleanup(self, api_client):
    """Test with immediate resource cleanup."""
    resource = None
    try:
        # Create resource
        resource = api_client.create_resource(data)
        
        # Test operations
        assert resource["id"] is not None
        
    finally:
        # Always cleanup
        if resource:
            api_client.delete_resource(resource["id"])
```

#### Batch Cleanup
```python
class TestWithBatchCleanup:
    resources_to_cleanup = []
    
    def test_create_resources(self, api_client):
        """Create multiple resources."""
        for i in range(10):
            resource = api_client.create_resource(f"Resource {i}")
            self.resources_to_cleanup.append(resource["id"])
    
    def test_999_cleanup_all(self, api_client):
        """Cleanup all resources at once."""
        for resource_id in self.resources_to_cleanup:
            try:
                api_client.delete_resource(resource_id)
            except:
                continue  # Best effort
```

---

## Advanced Test Scenarios

### Multi-Step Workflows

Test complex workflows:

```python
def test_advanced_workflow(self, api_client, data_tracker):
    """Test complex multi-step workflow."""
    workflow_state = {}
    
    # Step 1: Upload media
    media_response = api_client.upload_media(
        file_path=create_test_file("Content"),
        title="Workflow Test Media"
    )
    workflow_state["media_id"] = media_response["id"]
    data_tracker.add_media(media_response["id"])
    
    # Step 2: Process with AI
    analysis_response = api_client.analyze_media(
        media_id=workflow_state["media_id"],
        prompt="Summarize this content"
    )
    workflow_state["analysis_id"] = analysis_response["id"]
    
    # Step 3: Create note from analysis
    note_response = api_client.create_note(
        title=f"Analysis of {workflow_state['media_id']}",
        content=analysis_response["summary"]
    )
    workflow_state["note_id"] = note_response["id"]
    data_tracker.add_note(note_response["id"])
    
    # Step 4: Generate report
    report_response = api_client.generate_report(
        media_ids=[workflow_state["media_id"]],
        note_ids=[workflow_state["note_id"]],
        include_analysis=True
    )
    
    # Verify workflow completed
    assert report_response["status"] == "completed"
    assert len(report_response["sections"]) >= 2
```

### Concurrent Operations

Test parallel operations:

```python
import asyncio
import httpx

async def test_concurrent_operations():
    """Test concurrent API operations."""
    async with httpx.AsyncClient(base_url=BASE_URL) as client:
        tasks = []
        
        # Create 10 concurrent requests
        for i in range(10):
            task = client.post(
                f"{API_PREFIX}/notes/",
                json={"title": f"Concurrent Note {i}", "content": "Test"},
                headers={"X-API-KEY": "test-api-key-12345"}
            )
            tasks.append(task)
        
        # Execute concurrently
        responses = await asyncio.gather(*tasks)
        
        # Verify all succeeded
        for response in responses:
            assert response.status_code == 201
```

### Error Recovery

Test error handling and recovery:

```python
def test_error_recovery(self, api_client):
    """Test system recovery from errors."""
    
    # Cause an error
    with pytest.raises(httpx.HTTPStatusError) as exc:
        api_client.upload_media(
            file_path="/nonexistent/file.txt",
            title="Will Fail"
        )
    
    # Verify system still responsive
    health = api_client.health_check()
    assert health["status"] == "healthy"
    
    # Verify can still perform operations
    response = api_client.create_note(
        title="After Error",
        content="System recovered"
    )
    assert "id" in response
```

### Load Testing

Basic load testing:

```python
def test_load_handling(self, api_client):
    """Test system under load."""
    results = {
        "success": 0,
        "failure": 0,
        "times": []
    }
    
    # Send 50 requests
    for i in range(50):
        start = time.time()
        try:
            response = api_client.create_note(
                title=f"Load Test {i}",
                content="Testing load handling"
            )
            results["success"] += 1
            results["times"].append(time.time() - start)
        except:
            results["failure"] += 1
    
    # Analyze results
    success_rate = results["success"] / 50
    avg_time = sum(results["times"]) / len(results["times"])
    
    assert success_rate >= 0.95, f"Success rate too low: {success_rate}"
    assert avg_time < 1.0, f"Average response time too high: {avg_time}s"
```

---

## Integration Checklist

### Before Integration

- [ ] Feature is documented in API specs
- [ ] Authentication requirements are clear
- [ ] Response format is finalized
- [ ] Error codes are defined
- [ ] Rate limits are known

### Implementation Checklist

- [ ] API client method added
- [ ] Test data generator created
- [ ] Main test implemented
- [ ] Error cases covered
- [ ] Performance tracked
- [ ] Resources tracked for cleanup
- [ ] Documentation updated

### After Integration

- [ ] Tests pass locally
- [ ] Tests pass in CI/CD
- [ ] Performance meets requirements
- [ ] No test interdependencies
- [ ] Cleanup verified
- [ ] Documentation complete

---

## Templates and Examples

### Basic Endpoint Test Template

```python
def test_XX_feature_name(self, api_client, data_tracker):
    """Test feature_name functionality.
    
    This test verifies that:
    1. Feature accepts valid input
    2. Returns expected response format
    3. Handles errors appropriately
    4. Cleans up resources
    """
    # Setup test data
    test_data = TestDataGenerator.sample_feature_data()
    
    try:
        # Call API
        response = api_client.feature_endpoint(
            param1=test_data["param1"],
            param2=test_data.get("param2")
        )
        
        # Verify response structure
        assert "expected_field" in response
        assert response.get("status") in ["valid", "values"]
        
        # Track resources
        if "id" in response:
            data_tracker.add_resource("feature", response["id"])
        
        # Store for later tests
        self.feature_results.append(response)
        
    except httpx.HTTPStatusError as e:
        # Handle expected errors
        if e.response.status_code == 404:
            pytest.skip("Feature not available")
        else:
            raise
```

### CRUD Operations Template

```python
class TestFeatureCRUD:
    """Test CRUD operations for new feature."""
    
    created_ids = []
    
    def test_create(self, api_client):
        """Test CREATE operation."""
        data = TestDataGenerator.sample_data()
        response = api_client.create_feature(data)
        assert "id" in response
        self.created_ids.append(response["id"])
    
    def test_read(self, api_client):
        """Test READ operation."""
        if not self.created_ids:
            pytest.skip("No items to read")
        response = api_client.get_feature(self.created_ids[0])
        assert response["id"] == self.created_ids[0]
    
    def test_update(self, api_client):
        """Test UPDATE operation."""
        if not self.created_ids:
            pytest.skip("No items to update")
        response = api_client.update_feature(
            self.created_ids[0],
            {"field": "new_value"}
        )
        assert response["field"] == "new_value"
    
    def test_delete(self, api_client):
        """Test DELETE operation."""
        if not self.created_ids:
            pytest.skip("No items to delete")
        api_client.delete_feature(self.created_ids[0])
        
        # Verify deleted
        with pytest.raises(httpx.HTTPStatusError) as exc:
            api_client.get_feature(self.created_ids[0])
        assert exc.value.response.status_code == 404
```

### Async Operations Template

```python
def test_async_operation(self, api_client):
    """Test asynchronous operation with polling."""
    # Start async operation
    response = api_client.start_async_operation(data)
    operation_id = response["operation_id"]
    
    # Poll for completion
    max_attempts = 30
    for attempt in range(max_attempts):
        status = api_client.get_operation_status(operation_id)
        
        if status["state"] == "completed":
            break
        elif status["state"] == "failed":
            pytest.fail(f"Operation failed: {status.get('error')}")
        
        time.sleep(1)  # Wait before next poll
    else:
        pytest.fail(f"Operation timed out after {max_attempts} seconds")
    
    # Get results
    results = api_client.get_operation_results(operation_id)
    assert results["success"] == True
```

---

## Best Practices Summary

### DO's
✅ Use data generators for test data
✅ Track all created resources
✅ Handle both auth modes
✅ Include performance metrics
✅ Write clear docstrings
✅ Test error cases
✅ Clean up resources
✅ Make tests idempotent

### DON'Ts
❌ Hardcode test data
❌ Depend on test order
❌ Skip cleanup
❌ Ignore errors
❌ Use production data
❌ Make external API calls
❌ Leave debug code
❌ Create flaky tests

---

## Troubleshooting New Tests

### Common Issues

1. **Test Fails in CI but Passes Locally**
   - Check environment variables
   - Verify server configuration
   - Look for timing issues
   - Check resource limits

2. **Intermittent Failures**
   - Add retry logic
   - Increase timeouts
   - Check for race conditions
   - Verify cleanup between tests

3. **Authentication Issues**
   - Verify both headers are set
   - Check token expiration
   - Confirm auth mode
   - Review endpoint requirements

---

## Contributing

When contributing new tests:

1. Follow the patterns in this guide
2. Update documentation
3. Add to coverage matrix
4. Include in CI/CD
5. Request review from team

---

*Last Updated: August 2024*
*See also: [E2E_TEST_DOCUMENTATION.md](E2E_TEST_DOCUMENTATION.md) | [TEST_PATTERNS.md](TEST_PATTERNS.md)*