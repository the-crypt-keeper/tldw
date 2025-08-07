#!/bin/bash
# Script to test GitHub Actions workflow using Docker

set -e

echo "=== Testing GitHub Actions Workflow with Docker ==="
echo

# Colors for output
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m'

# Function to run Docker test
run_docker_test() {
    local python_version=$1
    echo -e "\n${YELLOW}Testing with Python $python_version in Docker${NC}"
    
    # Build and run the Docker container
    docker build \
        --build-arg PYTHON_VERSION=$python_version \
        -f test-workflow/Dockerfile.ubuntu \
        -t tldw-test:py$python_version \
        .
    
    # Run the test
    docker run --rm \
        -v $(pwd):/app \
        -e PYTHONPATH=/app/tldw_Server_API \
        -e OPENAI_API_KEY=test-key \
        -e ANTHROPIC_API_KEY=test-key \
        tldw-test:py$python_version \
        bash -c "
            cd /app
            echo '=== Installing dependencies with pyproject.toml ==='
            python$python_version -m pip install --upgrade pip
            pip install -e '.[dev]'
            
            echo -e '\n=== Verifying installations ==='
            python$python_version -c 'import fastapi; print(f\"FastAPI: {fastapi.__version__}\")'
            python$python_version -c 'import pytest; print(f\"pytest: {pytest.__version__}\")'
            python$python_version -c 'import black; print(f\"black: {black.__version__}\")'
            python$python_version -c 'import ruff; print(\"ruff installed\")'
            python$python_version -c 'import mypy; print(f\"mypy: {mypy.__version__}\")'
            
            echo -e '\n=== Checking pytest markers ==='
            cd tldw_Server_API
            pytest --markers | grep -E 'unit|integration|external_api' || echo 'Markers not found'
            
            echo -e '\n=== Test discovery ==='
            pytest -v -m 'unit' --collect-only | head -20
            
            echo -e '\n=== Code quality tools ==='
            black --version
            ruff --version
            mypy --version
        "
}

# Test with different Python versions
for version in 3.10 3.11 3.12; do
    run_docker_test $version
done

echo -e "\n${GREEN}=== Docker-based workflow validation complete ===${NC}"