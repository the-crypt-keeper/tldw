#!/bin/bash
# Script to start the test server with proper environment variables

echo "Starting test server with TEST_MODE enabled..."

# Set test environment variables
export TEST_MODE=true
export TESTING=true
export SINGLE_USER_API_KEY=test-api-key-for-e2e-testing-12345

# Start the server
echo "Starting server with API key: $SINGLE_USER_API_KEY"
cd ../..
python -m uvicorn app.main:app --reload --port 8000 &

# Wait for server to be ready
echo "Waiting for server to be ready..."
for i in {1..30}; do
    if curl -s http://localhost:8000/api/v1/health > /dev/null 2>&1; then
        echo "✅ Server is ready!"
        echo ""
        echo "You can now run the E2E tests with:"
        echo "  python -m pytest tests/e2e/"
        echo ""
        echo "API Key for tests: $SINGLE_USER_API_KEY"
        exit 0
    fi
    echo -n "."
    sleep 1
done

echo ""
echo "❌ Server failed to start within 30 seconds"
exit 1