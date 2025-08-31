# E2E Test Setup Instructions

## Quick Start

1. **Start the test server with proper configuration:**
   ```bash
   cd tldw_Server_API/tests/e2e
   ./start_test_server.sh
   ```
   This script will:
   - Set `TEST_MODE=true` to enable test features
   - Set `TESTING=true` to disable rate limiting
   - Set a fixed `SINGLE_USER_API_KEY` for consistent authentication
   - Start the server and wait for it to be ready

2. **Run the E2E tests:**
   ```bash
   # Run all E2E tests
   python -m pytest tests/e2e/ -v

   # Run specific test files
   python -m pytest tests/e2e/test_database_operations.py -v
   python -m pytest tests/e2e/test_concurrent_operations.py -v
   ```

## Important Notes

- The server MUST be started with the `start_test_server.sh` script for tests to pass
- The script sets a fixed API key: `test-api-key-for-e2e-testing-12345`
- In TEST_MODE, the health endpoint exposes the API key for test fixtures to retrieve
- Tests will automatically skip if the server is not running

## Troubleshooting

If tests are failing with 401 Unauthorized:
1. Make sure the server was started with `./start_test_server.sh`
2. Check that `TEST_MODE=true` is set in the server environment
3. Verify the health endpoint returns `test_api_key` field when TEST_MODE is active

## Manual Server Start

If you prefer to start the server manually:
```bash
export TEST_MODE=true
export TESTING=true
export SINGLE_USER_API_KEY=test-api-key-for-e2e-testing-12345
python -m uvicorn tldw_Server_API.app.main:app --reload
```