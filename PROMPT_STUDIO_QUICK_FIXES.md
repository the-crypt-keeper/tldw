# Prompt Studio Quick Fix Guide

## Immediate Actions (Can be done in 1 day)

### 1. Fix Database Schema (15 min)
```sql
-- Add to PromptStudioDatabase._create_tables()
ALTER TABLE prompt_studio_job_queue ADD COLUMN project_id INTEGER;
```

### 2. Register Missing Endpoints (30 min)
```python
# In app/main.py, add:
from app.api.v1.endpoints import prompt_studio

app.include_router(
    prompt_studio.router,
    prefix="/api/v1/prompt-studio",
    tags=["prompt-studio"]
)
```

### 3. Fix Authentication in Tests (30 min)
```python
# In tests/prompt_studio/integration/test_api_endpoints.py
# Add to test methods that need auth:

from unittest.mock import patch

@patch('app.api.v1.API_Deps.auth_deps.get_current_user')
def test_create_project(self, mock_user, client):
    mock_user.return_value = {"id": "test-user", "username": "test"}
    # ... rest of test
```

### 4. Fix SQL Binding Error (15 min)
```python
# In test_database.py line with binding error:
# Change from:
cursor.execute("SELECT * FROM table WHERE is_golden = ?", ())
# To:
cursor.execute("SELECT * FROM table WHERE is_golden = ?", (1,))
```

### 5. Add Missing Prompt Generation Methods (1 hour)
```python
# In prompt_generator.py:

def generate(self, prompt_type, task, **kwargs):
    if prompt_type == PromptType.CHAIN_OF_THOUGHT:
        return f"{task}\n\nLet's think step by step:"
    
    elif prompt_type == PromptType.FEW_SHOT:
        examples = kwargs.get('examples', [])
        result = "\n\nExamples:\n"
        for ex in examples:
            result += f"Input: {ex.get('input')}\n"
            result += f"Output: {ex.get('output')}\n\n"
        return result + task
    
    elif prompt_type == PromptType.REACT:
        return f"Task: {task}\n\nThought:\nAction:\nObservation:"
    
    # Add strategy support
    if kwargs.get('strategy') == GenerationStrategy.DETAILED:
        additions = "\n\nProvide comprehensive details and explanations."
        return f"{task}{additions}"
    
    return task
```

### 6. Fix Mock Issues in Tests (30 min)
```python
# In test_test_case_manager.py:
# Replace direct attribute access with dict-style:

def test_get_test_case(self, manager, mock_db):
    mock_row = {"id": 1, "name": "Test", "inputs": '{}'}
    mock_db.row_to_dict.return_value = mock_row
    
    result = manager.get_test_case(1)
    assert result["name"] == "Test"  # Use dict access
```

## Expected Results After Quick Fixes

- **Before**: 114/159 tests passing (71.7%)
- **After Phase 1**: ~136/159 tests passing (85.5%)
- **Time Required**: 3-4 hours

## Test Commands

```bash
# Run all tests
python -m pytest tldw_Server_API/tests/prompt_studio/ -v

# Run only unit tests (faster)
python -m pytest tldw_Server_API/tests/prompt_studio/unit/ -v

# Run with coverage
python -m pytest tldw_Server_API/tests/prompt_studio/ --cov=tldw_Server_API.app.core.Prompt_Management.prompt_studio

# Run specific test file
python -m pytest tldw_Server_API/tests/prompt_studio/test_database.py -v
```

## Priority Order

1. **Database schema** - Fixes 3 tests immediately
2. **SQL binding** - Fixes 1 test immediately  
3. **Mock issues** - Fixes 5 tests immediately
4. **Authentication** - Fixes 11 tests immediately
5. **Prompt generation** - Fixes 8 tests
6. **Endpoint registration** - Fixes 6 tests

Total: 34 tests fixed in Phase 1 quick fixes!
