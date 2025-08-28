# End-to-End Tests

## Quick Start

```bash
# Run all e2e tests
python -m pytest tldw_Server_API/tests/e2e/ -v

# Run specific test
python -m pytest tldw_Server_API/tests/e2e/test_full_user_workflow.py::TestFullUserWorkflow::test_01_health_check -v

# Run with detailed output
python -m pytest tldw_Server_API/tests/e2e/ -xvs
```

## Documentation

📚 **[View the Complete E2E Testing Guide](./E2E_TESTING_GUIDE.md)**

The comprehensive guide covers:
- Architecture & Design
- Authentication System
- Test Organization (11 phases)
- Running Tests
- Writing New Tests
- Common Patterns & Best Practices
- Workflow Testing Pattern
- Troubleshooting
- Performance Testing
- CI/CD Integration
- Complete API Reference

## Test Files

| File | Purpose |
|------|---------|
| `E2E_TESTING_GUIDE.md` | Complete documentation (start here) |
| `fixtures.py` | Shared fixtures and utilities |
| `test_data.py` | Test data generators |
| `workflow_helpers.py` | Workflow testing helpers |
| `test_full_user_workflow.py` | Main workflow tests (11 phases) |
| `test_evaluations_workflow.py` | Evaluation endpoint tests |
| `test_concurrent_operations.py` | Concurrency tests |
| `test_negative_scenarios.py` | Error handling tests |
| `test_custom_benchmark.py` | Performance benchmarks |

## Test Phases Overview

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

## Requirements

- Python 3.8+
- pytest
- httpx
- All tldw_server dependencies

---

For complete documentation, examples, and patterns, see **[E2E_TESTING_GUIDE.md](./E2E_TESTING_GUIDE.md)**