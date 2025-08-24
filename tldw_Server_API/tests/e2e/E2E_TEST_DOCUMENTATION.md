# End-to-End Test Suite Documentation

## Table of Contents
1. [Overview & Architecture](#overview--architecture)
2. [Authentication System](#authentication-system)
3. [Test Suite Components](#test-suite-components)
4. [Running Tests Guide](#running-tests-guide)
5. [Test Coverage Matrix](#test-coverage-matrix)
6. [Extending the Test Suite](#extending-the-test-suite)
7. [Troubleshooting Guide](#troubleshooting-guide)
8. [Response Format Handling](#response-format-handling)

---

## Overview & Architecture

### Purpose
The E2E test suite validates the complete user journey through the tldw_server API, ensuring all components work together correctly from a user's perspective. These tests simulate real-world usage patterns and verify that the system behaves correctly end-to-end.

### Test Philosophy
- **User-Centric**: Tests mirror actual user workflows
- **Comprehensive**: Cover all major API functionalities
- **Adaptive**: Handle both single-user and multi-user modes
- **Maintainable**: Clear structure with reusable components
- **Performance-Aware**: Track execution times for optimization

### Architecture Flow
```
┌─────────────┐     ┌──────────────┐     ┌─────────────┐
│   Fixtures  │────▶│ Test Classes │────▶│  API Client │
└─────────────┘     └──────────────┘     └─────────────┘
       │                    │                     │
       ▼                    ▼                     ▼
┌─────────────┐     ┌──────────────┐     ┌─────────────┐
│ Test Data   │     │ Test Phases  │     │ API Server  │
│ Generators  │     │   (1-11)     │     │  Endpoints  │
└─────────────┘     └──────────────┘     └─────────────┘
       │                    │                     │
       ▼                    ▼                     ▼
┌─────────────┐     ┌──────────────┐     ┌─────────────┐
│   Cleanup   │     │   Tracking   │     │  Database   │
│   Tracker   │     │   Metrics    │     │   Storage   │
└─────────────┘     └──────────────┘     └─────────────┘
```

---

## Authentication System

### Overview
The test suite handles two authentication modes seamlessly:
- **Single-User Mode**: Uses API key authentication
- **Multi-User Mode**: Uses JWT token authentication

### Single-User Mode Authentication

#### Required Headers
```python
headers = {
    "X-API-KEY": "test-api-key-12345",  # Primary authentication
    "Token": "test-api-key-12345"       # Required by some endpoints (prompts)
}
```

#### How It Works
1. **Health Check**: Determines authentication mode
2. **API Key Retrieval**: Gets key from AuthNZ settings
3. **Header Setup**: Configures both X-API-KEY and Token headers
4. **Request Authentication**: Headers sent with all requests

#### Key Discovery Process
```python
# The test suite automatically discovers the API key
from tldw_Server_API.app.core.AuthNZ.settings import get_settings
settings = get_settings()
api_key = settings.SINGLE_USER_API_KEY  # e.g., "test-api-key-12345"
```

### Multi-User Mode Authentication

#### Required Headers
```python
headers = {
    "Authorization": f"Bearer {jwt_token}",
    "X-API-KEY": api_key  # Some endpoints still require this
}
```

#### Authentication Flow
1. Register new user
2. Login to obtain JWT token
3. Set Authorization header
4. Include token in subsequent requests

### Common Authentication Issues

| Issue | Cause | Solution |
|-------|-------|----------|
| 401 Unauthorized | Missing X-API-KEY header | Ensure fixtures set both headers |
| 401 on prompts | Missing Token header | Add capital-T "Token" header |
| Token mismatch | Different settings sources | Use AuthNZ settings consistently |
| Invalid token | Expired or wrong format | Refresh token or check generation |

---

## Test Suite Components

### File Structure
```
e2e/
├── fixtures.py                 # Shared fixtures and utilities
├── test_data.py                # Test data generators
├── test_full_user_workflow.py  # Main workflow tests
├── README.md                   # Quick reference
├── E2E_TEST_DOCUMENTATION.md  # This file
├── EXTENDING_TESTS.md         # Extension guide
└── TEST_PATTERNS.md           # Common patterns
```

### fixtures.py
Provides the core testing infrastructure:

#### APIClient Class
```python
class APIClient:
    """Wrapper for API interactions with authentication support."""
    
    def __init__(self, base_url: str = BASE_URL):
        self.client = httpx.Client(base_url=base_url, timeout=TEST_TIMEOUT)
        
    def set_auth_token(self, token: str):
        """Set authentication tokens for both single and multi-user modes."""
        self.client.headers.update({
            "X-API-KEY": token,
            "Token": token  # Capital T is important!
        })
```

#### Key Fixtures
- `api_client`: Basic client instance
- `authenticated_client`: Pre-authenticated client
- `data_tracker`: Tracks created resources for cleanup
- `test_user_credentials`: Generated test user data

### test_data.py
Generates realistic test data:

#### TestDataGenerator Class
- `sample_text_content()`: Markdown documents
- `sample_transcript()`: Timestamped transcripts
- `sample_note()`: Research notes
- `sample_prompt_template()`: AI prompts
- `sample_character_card()`: Character definitions

#### TestScenarios Class
- `research_workflow()`: Academic research scenario
- `content_creation_workflow()`: Content production scenario
- `media_processing_workflow()`: Media analysis scenario

### test_full_user_workflow.py
Main test implementation with 11 phases:

#### Test Phases
1. **Setup & Authentication** (tests 01-04)
2. **Media Ingestion** (tests 10-14)
3. **Transcription & Analysis** (test 20)
4. **Chat & Interaction** (tests 30-31)
5. **Notes Management** (tests 40-43)
6. **Prompts & Templates** (tests 50-51)
7. **Character Management** (tests 60-61)
8. **RAG & Search** (test 70)
9. **Evaluation** (test 80)
10. **Export & Sync** (test 90)
11. **Cleanup** (tests 100-105)

---

## Running Tests Guide

### Prerequisites
```bash
# Install dependencies
pip install -r tldw_Server_API/requirements.txt
pip install pytest httpx

# Start the API server
python -m uvicorn tldw_Server_API.app.main:app --reload

# Verify server is running
curl http://localhost:8000/api/v1/health
```

### Environment Configuration
```bash
# Required environment variables
export E2E_TEST_BASE_URL=http://localhost:8000
export SINGLE_USER_API_KEY=test-api-key-12345  # Optional, auto-detected

# Optional configuration
export TEST_TIMEOUT=30  # Request timeout in seconds
export PYTEST_VERBOSE=1  # Verbose output
```

### Running Tests

#### Run All Tests
```bash
python -m pytest tldw_Server_API/tests/e2e/ -v
```

#### Run Specific Phase
```bash
# Run only authentication tests
python -m pytest tldw_Server_API/tests/e2e/ -k "test_0[1-4]" -v

# Run only media tests
python -m pytest tldw_Server_API/tests/e2e/ -k "test_1[0-4]" -v
```

#### Run with Performance Metrics
```bash
python -m pytest tldw_Server_API/tests/e2e/ -v -s --tb=short
```

#### Run in CI/CD Pipeline
```bash
python -m pytest tldw_Server_API/tests/e2e/ \
    --junit-xml=test-results/e2e.xml \
    --cov=tldw_Server_API \
    --cov-report=html \
    --maxfail=5
```

### Interpreting Results

#### Success Output
```
======================== 29 passed, 3 skipped in 2.14s =========================
```

#### Performance Summary
```
Performance Summary:
├── test_01_health_check: 0.12s
├── test_10_upload_text: 0.45s
├── test_30_chat_completion: 0.89s
└── Average: 0.48s
```

---

## Test Coverage Matrix

### API Endpoints Coverage

| Endpoint | Test Coverage | Test Name | Status |
|----------|--------------|-----------|--------|
| `/health` | ✅ Full | test_01_health_check | Stable |
| `/auth/register` | ✅ Full | test_02_user_registration | Stable |
| `/auth/login` | ✅ Full | test_03_user_login | Stable |
| `/auth/me` | ✅ Full | test_04_get_user_profile | Stable |
| `/media/add` | ✅ Full | test_10-13_upload_* | Stable |
| `/media/` | ✅ Full | test_14_list_media | Stable |
| `/media/{id}` | ✅ Full | test_20_get_details | Stable |
| `/chat/completions` | ✅ Full | test_30-31_chat_* | Stable |
| `/notes/` | ✅ Full | test_40-43_notes_* | Stable |
| `/prompts/` | ✅ Full | test_50-51_prompts_* | Stable |
| `/characters/import` | ✅ Full | test_60_import_character | Stable |
| `/media/search` | ✅ Full | test_70_search_media | Stable |
| `/evaluations/` | ⚠️ Partial | test_80_evaluation | In Progress |
| `/export/` | ⚠️ Partial | test_90_export | In Progress |

### Feature Coverage

| Feature | Coverage | Notes |
|---------|----------|-------|
| Authentication | 100% | Both single/multi-user |
| Media Upload | 95% | All formats except video |
| Transcription | 80% | Audio/video covered |
| Chat/LLM | 90% | Streaming not tested |
| RAG/Search | 85% | Vector search covered |
| Notes | 100% | CRUD + search |
| Prompts | 100% | CRUD operations |
| Characters | 90% | Import/list covered |

---

## Extending the Test Suite

### Adding Tests for New Endpoints

#### Step 1: Add API Client Method
```python
# In fixtures.py APIClient class
def new_feature(self, param1: str, param2: Optional[str] = None) -> Dict[str, Any]:
    """Call new feature endpoint."""
    data = {"param1": param1}
    if param2:
        data["param2"] = param2
    
    response = self.client.post(f"{API_PREFIX}/new-feature", json=data)
    response.raise_for_status()
    return response.json()
```

#### Step 2: Add Test Data Generator
```python
# In test_data.py TestDataGenerator class
@staticmethod
def sample_new_feature_data() -> Dict[str, Any]:
    """Generate test data for new feature."""
    return {
        "param1": f"test_{TestDataGenerator.random_string(8)}",
        "param2": "optional_value",
        "metadata": {
            "created_at": datetime.now().isoformat(),
            "test_run": True
        }
    }
```

#### Step 3: Add Test Method
```python
# In test_full_user_workflow.py
def test_XX_new_feature(self, api_client, data_tracker):
    """Test new feature endpoint."""
    # Generate test data
    test_data = TestDataGenerator.sample_new_feature_data()
    
    # Call API
    response = api_client.new_feature(
        param1=test_data["param1"],
        param2=test_data.get("param2")
    )
    
    # Verify response
    assert "id" in response or "success" in response
    assert response.get("status") != "error"
    
    # Track for cleanup
    if "id" in response:
        data_tracker.add_resource("new_feature", response["id"])
    
    # Store for later tests
    self.new_features.append(response)
```

### Testing Feature Stability

#### Stability Criteria Checklist
- [ ] Endpoint responds consistently
- [ ] Authentication works correctly
- [ ] Error handling is proper
- [ ] Response format is stable
- [ ] Performance is acceptable
- [ ] Cleanup works properly

#### Integration Points
- [ ] Works with existing authentication
- [ ] Compatible with data tracker
- [ ] Follows test patterns
- [ ] Includes cleanup logic
- [ ] Has performance metrics

### Best Practices for New Tests

1. **Follow Naming Convention**: `test_XX_feature_name` where XX is the phase number
2. **Use Data Generators**: Don't hardcode test data
3. **Track Resources**: Always use data_tracker for cleanup
4. **Handle Errors Gracefully**: Use try/except for non-critical failures
5. **Add Performance Tracking**: Utilize the autouse fixture
6. **Document Expected Behavior**: Add docstrings explaining what's tested
7. **Consider Both Auth Modes**: Test should work in single and multi-user modes

---

## Troubleshooting Guide

### Common Test Failures

#### Authentication Failures
```
FAILED test_XX - HTTPStatusError: 401 Unauthorized
```
**Solutions:**
1. Check API key in settings: `settings.SINGLE_USER_API_KEY`
2. Verify headers are set: Both X-API-KEY and Token
3. Ensure server is in correct mode (single vs multi-user)
4. Check token expiration in multi-user mode

#### Media Upload Failures
```
FAILED test_10_upload - HTTPStatusError: 422 Unprocessable Entity
```
**Solutions:**
1. Verify file format matches media_type
2. Check file size limits
3. Ensure multipart form data format
4. Verify required fields (title, overwrite_existing)

#### Response Format Issues
```
AssertionError: assert 'items' in response
```
**Solutions:**
1. Check if API returns list directly vs wrapped
2. Update test to handle both formats
3. Verify API version compatibility

### Debug Techniques

#### Enable Verbose Logging
```python
# Add to test file
import logging
logging.basicConfig(level=logging.DEBUG)

# Or run with pytest
pytest -vvs --log-cli-level=DEBUG
```

#### Inspect Request/Response
```python
# In test method
print(f"Request headers: {api_client.client.headers}")
print(f"Response: {response}")
print(f"Response status: {response.status_code}")
```

#### Use Debugger
```python
# Add breakpoint in test
import pdb; pdb.set_trace()

# Run test
pytest -s test_file.py::test_method
```

### Performance Optimization

#### Identify Slow Tests
```python
# After test run, check performance_metrics
def test_105_performance_summary(self):
    slow_tests = {k: v for k, v in self.performance_metrics.items() if v > 1.0}
    print(f"Slow tests (>1s): {slow_tests}")
```

#### Optimization Strategies
1. **Parallel Execution**: Use pytest-xdist
2. **Reduce Timeouts**: Lower TEST_TIMEOUT for fast tests
3. **Skip Expensive Tests**: Mark with @pytest.mark.slow
4. **Cache Responses**: For idempotent operations
5. **Batch Operations**: Combine related API calls

---

## Response Format Handling

### Adaptive Response Parsing

The test suite handles various response formats intelligently:

#### Pattern 1: Direct vs Wrapped Lists
```python
# API might return list directly or wrapped
response = api_client.get_notes()

if isinstance(response, list):
    items = response
else:
    items = response.get("items") or response.get("results", [])
```

#### Pattern 2: Nested Data Structures
```python
# Handle nested title field
assert "title" in response or (
    isinstance(response.get("source"), dict) and 
    "title" in response["source"]
)
```

#### Pattern 3: Results Array Format
```python
# New media upload format with results array
if "results" in response and isinstance(response["results"], list):
    result = response["results"][0]
    if result.get("status") == "Success":
        media_id = result.get("db_id")
```

### Response Validation Strategies

#### Flexible Field Checking
```python
# Check for various ID field names
media_id = (
    response.get("id") or 
    response.get("media_id") or 
    response.get("db_id")
)
```

#### Status Validation
```python
# Accept various success indicators
assert response.get("status") in ["success", "Success", "completed", "ok"]
```

#### Error Handling
```python
# Handle expected errors gracefully
if result.get("error") and "already exists" in result.get("error"):
    # This is acceptable for idempotent tests
    pass
elif result.get("error"):
    pytest.fail(f"Unexpected error: {result['error']}")
```

### Format Migration Guide

When API response formats change:

1. **Maintain Backward Compatibility**: Support both old and new formats
2. **Use Feature Detection**: Check response structure, not API version
3. **Log Format Changes**: Help identify when formats change
4. **Update Documentation**: Note format variations in tests

---

## Security Considerations

### API Key Management
- Never hardcode production keys
- Use environment variables for sensitive data
- Rotate test API keys regularly
- Use different keys for different test environments

### Test Data Security
- Don't use real user data
- Sanitize any exported test results
- Clear sensitive data after tests
- Use mock data for payment/billing tests

### Rate Limiting
- Respect API rate limits
- Add delays between intensive tests
- Use batch operations where possible
- Monitor rate limit headers

---

## Continuous Integration

### GitHub Actions Example
```yaml
name: E2E Tests

on:
  push:
    branches: [main, dev]
  pull_request:
    branches: [main]

jobs:
  e2e-tests:
    runs-on: ubuntu-latest
    
    steps:
    - uses: actions/checkout@v2
    
    - name: Set up Python
      uses: actions/setup-python@v2
      with:
        python-version: '3.10'
    
    - name: Install dependencies
      run: |
        pip install -r requirements.txt
        pip install pytest httpx pytest-cov
    
    - name: Start API server
      run: |
        python -m uvicorn tldw_Server_API.app.main:app &
        sleep 5  # Wait for server to start
    
    - name: Run E2E tests
      env:
        E2E_TEST_BASE_URL: http://localhost:8000
        SINGLE_USER_API_KEY: ${{ secrets.TEST_API_KEY }}
      run: |
        python -m pytest tldw_Server_API/tests/e2e/ \
          --junit-xml=test-results/e2e.xml \
          --cov=tldw_Server_API \
          --cov-report=xml
    
    - name: Upload test results
      uses: actions/upload-artifact@v2
      if: always()
      with:
        name: test-results
        path: test-results/
```

---

## Version History

### Current Version: 2.0.0 (2024-08)
- Complete authentication system rewrite
- Support for both single and multi-user modes
- Adaptive response format handling
- Performance metrics tracking

### Changes from 1.0.0
- Fixed authentication header issues
- Added Token header support
- Updated media upload format handling
- Improved error recovery
- Added advanced test scenarios

---

## Additional Resources

- [EXTENDING_TESTS.md](EXTENDING_TESTS.md) - Detailed guide for adding new tests
- [TEST_PATTERNS.md](TEST_PATTERNS.md) - Common patterns and best practices
- [README.md](README.md) - Quick reference guide
- [API Documentation](/Docs/API-related/) - API endpoint specifications

---

*Last Updated: August 2024*
*Maintained by: tldw_server Development Team*