# End-to-End Tests

## Overview

This directory contains comprehensive end-to-end tests for the tldw_server API. These tests validate the complete user journey through all core functionality, ensuring that the entire system works correctly from a user's perspective.

## Test Structure

```
e2e/
├── fixtures.py                 # Shared fixtures and utilities
├── test_data.py                # Test data generators
├── test_full_user_workflow.py  # Main workflow tests
├── E2E_TEST_DOCUMENTATION.md   # Comprehensive test documentation
├── EXTENDING_TESTS.md          # Guide for extending test coverage
├── TEST_PATTERNS.md            # Common patterns and best practices
└── README.md                   # This file
```

## Documentation

For detailed information about the E2E test suite:

- **[E2E Test Documentation](./E2E_TEST_DOCUMENTATION.md)** - Comprehensive guide covering all aspects of the test suite including architecture, authentication, troubleshooting, and test coverage matrix
- **[Extending Tests Guide](./EXTENDING_TESTS.md)** - Step-by-step instructions for adding new tests, including templates, examples, and integration checklists
- **[Test Patterns](./TEST_PATTERNS.md)** - Common patterns, best practices, and reusable code snippets used throughout the test suite

## Running the Tests

### Basic Usage

```bash
# Run all e2e tests
python -m pytest tldw_Server_API/tests/e2e/ -v

# Run with performance tracking
python -m pytest tldw_Server_API/tests/e2e/ -v -s

# Run specific test phase
python -m pytest tldw_Server_API/tests/e2e/test_full_user_workflow.py::TestFullUserWorkflow::test_01_health_check -v
```

### Configuration

Tests can be configured via environment variables:

```bash
# Set API endpoint (default: http://localhost:8000)
export E2E_TEST_BASE_URL=http://localhost:8000

# Run tests
python -m pytest tldw_Server_API/tests/e2e/ -v
```

## Test Phases

The tests are organized into 11 sequential phases:

### Phase 1: Setup & Authentication (tests 01-04)
- Health check
- User registration (if multi-user mode)
- Login and token management
- Profile verification

### Phase 2: Media Ingestion (tests 10-14)
- Upload text document
- Upload PDF
- Process web content
- Upload audio file
- List media items

### Phase 3: Transcription & Analysis (test 20)
- Get media details
- Verify transcription/extraction

### Phase 4: Chat & Interaction (tests 30-31)
- Simple chat completion
- Chat with context (RAG)

### Phase 5: Notes & Knowledge Management (tests 40-43)
- Create notes
- List notes
- Update notes
- Search notes

### Phase 6: Prompts & Templates (tests 50-51)
- Create prompt templates
- List prompts

### Phase 7: Character Management (tests 60-61)
- Import character cards
- List characters

### Phase 8: RAG & Search (test 70)
- Search across media content
- Vector and text search

### Phase 9: Evaluation (test 80)
- Placeholder for evaluation tests

### Phase 10: Export & Sync (test 90)
- Placeholder for export functionality

### Phase 11: Cleanup (tests 100-105)
- Delete all created resources
- Logout
- Performance summary

## Test Data

Test data is generated using the `TestDataGenerator` class which provides:
- Sample documents and transcripts
- Character cards
- Prompt templates
- Chat conversations
- Search queries

## Fixtures

Key fixtures provided:
- `api_client`: Basic API client
- `authenticated_client`: Logged-in API client
- `data_tracker`: Tracks created resources for cleanup
- `test_user_credentials`: Test user credentials

## Performance Tracking

Tests automatically track execution time and provide a summary at the end:
- Individual test duration
- Total execution time
- Average test time

## Troubleshooting

### Common Issues

1. **Authentication Failures**
   - Ensure the API server is running
   - Check if single-user or multi-user mode

2. **Media Upload Failures**
   - Verify ffmpeg is installed
   - Check file size limits

3. **Search Tests Failing**
   - Ensure embeddings are enabled
   - Check if ChromaDB is configured

### Debug Mode

Run with verbose output:
```bash
python -m pytest tldw_Server_API/tests/e2e/ -vvs --tb=short
```

## CI/CD Integration

These tests are designed to run in CI/CD pipelines:

```yaml
# Example GitHub Actions workflow
- name: Run E2E Tests
  run: |
    python -m pytest tldw_Server_API/tests/e2e/ \
      --junit-xml=test-results/e2e.xml \
      --cov=tldw_Server_API \
      --cov-report=xml
```

## Extending the Tests

To add new test scenarios:

1. Add test data generators to `test_data.py`
2. Add test methods to `TestFullUserWorkflow` class
3. Use appropriate `@pytest.mark.order()` decorators
4. Track created resources with `data_tracker`

## Requirements

- Python 3.8+
- pytest
- httpx
- All tldw_server dependencies

## Notes

- Tests are designed to be idempotent
- All created resources are cleaned up
- Tests handle both single-user and multi-user modes
- Performance metrics help identify bottlenecks