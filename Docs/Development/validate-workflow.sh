#!/bin/bash
# Quick validation script for GitHub Actions workflow

set -e

echo "=== Validating GitHub Actions Workflow Configuration ==="
echo

# Colors for output
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m'

# Create a test virtual environment
echo -e "${YELLOW}Creating test environment...${NC}"
python3 -m venv test_venv
source test_venv/bin/activate

# Test 1: Install with pyproject.toml
echo -e "\n${YELLOW}Test 1: Installing package with pyproject.toml${NC}"
python -m pip install --upgrade pip
pip install -e ".[dev]"

if [ $? -eq 0 ]; then
    echo -e "${GREEN}✓ Package installation successful${NC}"
else
    echo -e "${RED}✗ Package installation failed${NC}"
    deactivate
    rm -rf test_venv
    exit 1
fi

# Test 2: Verify key dependencies
echo -e "\n${YELLOW}Test 2: Verifying key dependencies${NC}"
python << EOF
import sys
try:
    import fastapi
    print("✓ FastAPI installed")
except ImportError:
    print("✗ FastAPI not found")
    sys.exit(1)

try:
    import pytest
    print("✓ pytest installed")
except ImportError:
    print("✗ pytest not found")
    sys.exit(1)

try:
    import black
    print("✓ black installed")
except ImportError:
    print("✗ black not found")
    sys.exit(1)

try:
    import ruff
    print("✓ ruff installed")
except ImportError:
    print("✗ ruff not found")
    sys.exit(1)

try:
    import mypy
    print("✓ mypy installed")
except ImportError:
    print("✗ mypy not found")
    sys.exit(1)

try:
    import pytest_cov
    print("✓ pytest-cov installed")
except ImportError:
    print("✗ pytest-cov not found")
    sys.exit(1)

try:
    import pytest_asyncio
    print("✓ pytest-asyncio installed")
except ImportError:
    print("✗ pytest-asyncio not found")
    sys.exit(1)
EOF

# Test 3: Check pytest configuration
echo -e "\n${YELLOW}Test 3: Checking pytest configuration${NC}"
cd tldw_Server_API
pytest --markers | grep -E "unit|integration|external_api" > /dev/null 2>&1
if [ $? -eq 0 ]; then
    echo -e "${GREEN}✓ Pytest markers configured correctly${NC}"
    pytest --markers | grep -E "unit|integration|external_api"
else
    echo -e "${RED}✗ Pytest markers not found${NC}"
fi

# Test 4: Test discovery
echo -e "\n${YELLOW}Test 4: Test discovery${NC}"
echo "Collecting tests with 'unit' marker..."
test_count=$(pytest --collect-only -q -m "unit" 2>/dev/null | grep -c "<" || echo "0")
echo "Found $test_count unit tests"

if [ $test_count -gt 0 ]; then
    echo -e "${GREEN}✓ Test discovery working${NC}"
else
    echo -e "${YELLOW}⚠ No unit tests found (might need to add markers to tests)${NC}"
fi

# Test 5: Code quality tools
echo -e "\n${YELLOW}Test 5: Testing code quality tools${NC}"

# Black
echo "Testing black..."
black --version > /dev/null 2>&1
if [ $? -eq 0 ]; then
    echo -e "${GREEN}✓ Black is available${NC}"
else
    echo -e "${RED}✗ Black not available${NC}"
fi

# Ruff
echo "Testing ruff..."
ruff --version > /dev/null 2>&1
if [ $? -eq 0 ]; then
    echo -e "${GREEN}✓ Ruff is available${NC}"
else
    echo -e "${RED}✗ Ruff not available${NC}"
fi

# MyPy
echo "Testing mypy..."
mypy --version > /dev/null 2>&1
if [ $? -eq 0 ]; then
    echo -e "${GREEN}✓ MyPy is available${NC}"
else
    echo -e "${RED}✗ MyPy not available${NC}"
fi

cd ..

# Cleanup
deactivate
rm -rf test_venv

echo -e "\n${GREEN}=== Validation complete ===${NC}"
echo
echo "Summary:"
echo "- pyproject.toml installation: ✓"
echo "- Development dependencies: ✓"
echo "- Pytest configuration: ✓"
echo "- Code quality tools: ✓"
echo
echo -e "${YELLOW}Note: Some tests might show warnings about missing test markers.${NC}"
echo -e "${YELLOW}You may need to add pytest markers to your test files.${NC}"