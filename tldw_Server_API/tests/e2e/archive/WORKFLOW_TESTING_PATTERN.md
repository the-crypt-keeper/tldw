# E2E Workflow Testing Pattern Documentation

## Overview

The E2E test suite uses a **Sequential Workflow Testing Pattern** that simulates a real user's journey through the API from initial setup to final cleanup. This pattern is intentionally designed to test how operations work together in a realistic sequence.

## Key Design Principles

### 1. Sequential Execution
Tests are numbered (test_01, test_02, etc.) to ensure they run in order:
```python
def test_01_health_check(self, api_client):
def test_02_user_registration(self, api_client):
def test_10_upload_text_document(self, api_client):
```

### 2. Shared State
Class-level variables store data that flows between test phases:
```python
class TestFullUserWorkflow:
    # Shared state for workflow continuity
    user_data = {}
    media_items = []
    notes = []
    prompts = []
    characters = []
    chats = []
```

This is **intentional** - each test builds on the previous one's results.

### 3. Phase-Based Organization
The workflow is divided into logical phases:

1. **Setup & Authentication** (test_01-09)
2. **Media Upload** (test_10-18) 
3. **Transcription & Analysis** (test_20-28)
4. **Chat & Interaction** (test_30-38)
5. **Notes & Knowledge Management** (test_40-48)
6. **Prompts & Templates** (test_50-58)
7. **Character Management** (test_60-68)
8. **RAG & Search** (test_70-78)
9. **Evaluation** (test_80-88)
10. **Export & Sync** (test_90-98)
11. **Cleanup** (test_100-105)

### 4. Verification Checkpoints
Between phases, checkpoint tests verify data integrity:
```python
def test_16_verify_upload_phase_complete(self, api_client):
    """CHECKPOINT: Verify all uploads from phase 2 are accessible."""
    # Verify previous phase completed successfully
    # Check data is ready for next phase
```

## How It Works

### Example Flow
1. **test_01** checks API health and determines auth mode
2. **test_02** registers user (if multi-user mode)
3. **test_10** uploads a document, stores media_id in class variable
4. **test_16** verifies upload succeeded (checkpoint)
5. **test_30** uses uploaded content for chat context
6. **test_40** creates notes referencing the chat
7. **test_100** cleans up all created resources

### Data Flow Example
```python
# Test 10: Upload stores media ID
TestFullUserWorkflow.media_items.append({
    "media_id": media_id,
    "response": response,
    "original_content": content
})

# Test 30: Chat uses uploaded media for context
if TestFullUserWorkflow.media_items:
    media_id = TestFullUserWorkflow.media_items[0]["media_id"]
    # Use media_id for context-aware chat
```

## Best Practices

### 1. Strengthened Assertions
Use helper classes for meaningful validations:
```python
from workflow_helpers import WorkflowAssertions

# Don't just check existence
assert "id" in response  # ❌ Weak

# Validate actual values
media_id = WorkflowAssertions.assert_valid_upload(response)  # ✅ Strong
```

### 2. Proper Error Handling
Distinguish between expected and unexpected failures:
```python
from workflow_helpers import WorkflowErrorHandler

try:
    response = api_client.upload_media(file_path)
except Exception as e:
    WorkflowErrorHandler.handle_api_error(e, "media upload")
    # Automatically skips for 501 (not implemented)
    # Fails for actual errors
```

### 3. Verification After Operations
Always verify operations succeeded:
```python
# After upload
response = api_client.upload_media(file_path)
media_id = WorkflowAssertions.assert_valid_upload(response)

# Verify retrievable
retrieved = api_client.get_media_item(media_id)
assert retrieved is not None
```

### 4. Checkpoint Implementation
Add checkpoints between major phases:
```python
def test_29_verify_ready_for_interaction(self, api_client):
    """CHECKPOINT: Verify system ready for chat phase."""
    print(f"\\n=== PRE-PHASE 4 VERIFICATION ===")
    
    # Check prerequisites
    has_media = len(TestFullUserWorkflow.media_items) > 0
    assert has_media or skip_if_no_media, "Need media for context"
    
    print("=== Proceeding to Phase 4 ===")
```

## Running the Tests

### Full Workflow
Run all tests in sequence:
```bash
python -m pytest tldw_Server_API/tests/e2e/test_full_user_workflow.py -v
```

### With Detailed Output
See checkpoint messages and progress:
```bash
python -m pytest tldw_Server_API/tests/e2e/test_full_user_workflow.py -xvs
```

### Specific Phase
Run tests for a specific phase:
```bash
# Run upload phase only
python -m pytest tldw_Server_API/tests/e2e/test_full_user_workflow.py -k "test_1" -xvs
```

### From Specific Test
Start from a particular test:
```bash
# Start from test_30 onwards
python -m pytest tldw_Server_API/tests/e2e/test_full_user_workflow.py::TestFullUserWorkflow::test_30_simple_chat_completion -xvs
```

## Adding New Tests

### 1. Choose Correct Position
Place test in appropriate phase based on dependencies:
```python
def test_35_new_chat_feature(self, api_client):
    """Test new chat feature - requires uploaded media."""
    # This goes in Phase 4 (Chat) after basic chat works
```

### 2. Use Shared State
Access data from previous tests:
```python
def test_45_enhanced_notes(self, api_client):
    # Use existing media
    if TestFullUserWorkflow.media_items:
        media_id = TestFullUserWorkflow.media_items[0]["media_id"]
        # Create note referencing media
```

### 3. Store Results for Later
Add results to class variables:
```python
# Store for use in later tests
TestFullUserWorkflow.notes.append({
    "note_id": note_id,
    "content": content
})
```

### 4. Handle Dependencies Gracefully
Skip if prerequisites missing:
```python
def test_75_advanced_search(self, api_client):
    if not TestFullUserWorkflow.media_items:
        pytest.skip("No media available for search test")
```

## Common Patterns

### Pattern 1: Create-Verify-Store
```python
# Create resource
response = api_client.create_note(title, content)

# Verify creation
note_id = WorkflowAssertions.assert_valid_note(response)

# Verify retrievable
retrieved = api_client.get_note(note_id)
assert retrieved["title"] == title

# Store for later
TestFullUserWorkflow.notes.append({"note_id": note_id})
```

### Pattern 2: Batch Operations
```python
# Create multiple items
created_ids = []
for i in range(5):
    response = api_client.create_item(f"Item {i}")
    item_id = response["id"]
    created_ids.append(item_id)

# Store batch
TestFullUserWorkflow.items.extend(created_ids)
```

### Pattern 3: Cleanup Pattern
```python
def test_100_cleanup_items(self, api_client):
    """Clean up all created items."""
    for item in TestFullUserWorkflow.items:
        try:
            api_client.delete_item(item["id"])
        except Exception as e:
            print(f"Cleanup failed for {item['id']}: {e}")
            # Don't fail test on cleanup errors
```

## Troubleshooting

### Issue: Tests Fail When Run Individually
**Cause**: Test depends on state from previous tests
**Solution**: Either run full suite or mock required state in setup

### Issue: Flaky Tests
**Cause**: Race conditions or timing issues
**Solution**: Add proper waits or use AsyncOperationHandler

### Issue: Cleanup Incomplete
**Cause**: Tests failed before cleanup phase
**Solution**: Use pytest fixtures with proper teardown

### Issue: State Pollution Between Runs
**Cause**: Class variables persist
**Solution**: Clear state in first test or use fixture

## Benefits of This Pattern

1. **Realistic Testing**: Simulates actual user workflows
2. **Integration Testing**: Tests how features work together
3. **State Verification**: Ensures data persists correctly
4. **Progressive Complexity**: Later tests build on earlier ones
5. **Clear Dependencies**: Obvious what each test requires

## Limitations

1. **Test Independence**: Individual tests can't run in isolation
2. **Debugging Difficulty**: Failures might be caused by earlier tests
3. **Longer Execution**: Must run entire workflow
4. **State Management**: Requires careful handling of shared state

## Alternative Patterns

For independent tests, consider:
- **Fixture-based setup**: Each test gets fresh state
- **Factory pattern**: Create test data on demand
- **Mocking**: Mock dependencies instead of creating them

However, these don't test the actual user journey like the workflow pattern does.

## Conclusion

The Sequential Workflow Testing Pattern is intentionally designed for E2E testing of API flows. It trades test independence for realistic user journey validation. When properly implemented with checkpoints and verification, it provides confidence that the system works correctly for real users.