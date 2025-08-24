# API Issues Discovered by E2E Tests

## Date: August 2024
## Status: Investigation Complete

## Summary
The E2E tests revealed 3 API issues that need to be addressed. These are not test problems but actual API implementation issues.

## Issues Found

### 1. ✅ User Registration (HTTP 400) - Expected Behavior
**Test**: `test_01_user_registration`
**Issue**: Registration endpoint returns HTTP 400 Bad Request
**Investigation Result**: This is EXPECTED BEHAVIOR
- The API is running in single-user mode (SINGLE_USER_API_KEY is set)
- Registration is intentionally disabled in single-user mode
- The API returns 400 with message: "User registration is disabled in single-user mode"
**Fix Required**: None - this is correct behavior
**Test Fix**: Test should check for single-user mode and skip registration test

### 2. ❌ Media Upload Response Field Mismatch
**Test**: `test_10_upload_text_document` and others
**Issue**: Upload endpoint returns `db_id` instead of `media_id`
**Investigation Result**: FIELD NAME MISMATCH
```python
# Test expects:
response = {"media_id": 123, ...}

# API returns:
response = {"db_id": 123, ...}
```
**Location**: `/api/v1/media/add` endpoint response
**Fix Required**: Either:
  - Option A: Update API to return `media_id` field (breaking change)
  - Option B: Update tests to use `db_id` field (non-breaking)
**Recommendation**: Option B - update tests to use `db_id`

### 3. ❌ Missing Generic Media Process Endpoint
**Test**: `test_12_process_web_content`
**Issue**: Test calls `/api/v1/media/process` which doesn't exist (HTTP 405)
**Investigation Result**: ENDPOINT DOESN'T EXIST
- Test is calling: `POST /api/v1/media/process`
- The API has two patterns for media processing:
  
  **For ephemeral processing (no DB storage):**
  - `POST /api/v1/media/process-videos` - Process videos, return results
  - `POST /api/v1/media/process-audios` - Process audio files
  - `POST /api/v1/media/process-documents` - Process text/HTML documents
  - `POST /api/v1/media/process-pdfs` - Process PDF files
  - `POST /api/v1/media/process-ebooks` - Process EPUB files
  
  **For persistent storage (saves to DB):**
  - `POST /api/v1/media/add` - Add any media type with processing
  - `POST /api/v1/media/ingest-web-content` - Ingest web content with advanced options

**Fix Required**: 
- Option A: Create a generic `/api/v1/media/process` endpoint that routes to the appropriate processor
- Option B: Update test to use the correct specific endpoint based on content type
**Recommendation**: The test should actually test BOTH patterns:
  1. Use `/api/v1/media/process-documents` for ephemeral processing
  2. Use `/api/v1/media/add` or `/api/v1/media/ingest-web-content` for persistent storage

## Fixes Needed

### Test Fixes (Non-Breaking)

1. **Update fixtures.py to support both processing patterns**:
```python
def process_media(self, url: Optional[str] = None, file_path: Optional[str] = None,
                 title: Optional[str] = None, custom_prompt: Optional[str] = None,
                 persist: bool = True) -> Dict[str, Any]:
    """Process media from URL or file, with option for ephemeral or persistent storage."""
    
    if persist:
        # Use /add endpoint for persistent storage
        data = {}
        if url:
            data["urls"] = [url]  # Note: urls is a list
            data["media_type"] = "article"  # or detect from URL
        if title:
            data["title"] = title
        # ... other fields
        
        response = self.client.post(
            f"{API_PREFIX}/media/add",
            data=data,
            files=files if file_path else None
        )
    else:
        # Use process-documents for ephemeral processing (web content)
        data = {}
        if url:
            data["urls"] = [url]
        if title:
            data["titles"] = [title]
        
        response = self.client.post(
            f"{API_PREFIX}/media/process-documents",
            data=data,
            files=files if file_path else None
        )
    
    response.raise_for_status()
    return response.json()
```

2. **Update all tests to use `db_id` instead of `media_id`**:
```python
# Throughout test files, change:
media_id = response.get("media_id")
# To:
media_id = response.get("db_id")
```

3. **Skip registration test in single-user mode**:
```python
def test_01_user_registration(self, api_client):
    # Check if in single-user mode
    if os.getenv("SINGLE_USER_API_KEY"):
        pytest.skip("Registration disabled in single-user mode")
    # ... rest of test
```

### Alternative: API Fixes (Breaking Changes)

If we want to fix at the API level instead:

1. **Add `/api/v1/media/process` endpoint** - create new endpoint that matches test expectations
2. **Change `db_id` to `media_id`** in response schemas - breaking change for existing clients
3. **Keep registration behavior** - already correct

## Impact Assessment

### Current State
- 60 tests pass, 4 fail (89.5% pass rate)
- Failures are due to API/test mismatches, not functionality issues
- API is working correctly, just differently than tests expect

### After Test Fixes
- Expected: 100% pass rate
- No API changes needed
- Tests will properly validate actual API behavior

### After API Fixes (if chosen)
- Would be breaking changes
- Need to update all existing clients
- Not recommended unless part of major version bump

## Recommendations

1. **Immediate Action**: Fix tests to match actual API behavior
   - Update tests to handle both ephemeral and persistent processing patterns
   - Use `db_id` instead of `media_id` 
   - Skip registration in single-user mode
   
2. **Enhanced Testing**: Tests should validate BOTH patterns:
   - Ephemeral processing (process-* endpoints) - verify data is returned but NOT stored
   - Persistent processing (/add, /ingest-web-content) - verify data is stored and retrievable
   
3. **Documentation**: Update API documentation to clarify:
   - Two distinct processing patterns (ephemeral vs persistent)
   - Registration behavior in single-user mode
   - Field names in responses (`db_id` not `media_id`)
   - Correct endpoints for different media types
   
4. **Future Consideration**: 
   - Consider adding a generic `/api/v1/media/process` endpoint that auto-detects media type
   - Consider API v2 that addresses naming inconsistencies

## Test Command

After fixes, run:
```bash
python -m pytest tldw_Server_API/tests/e2e/test_full_user_workflow.py -v
```

Expected result: All tests pass or skip appropriately.