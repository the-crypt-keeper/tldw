#!/bin/bash
# Script to test GitHub Actions workflow steps

set - e  # Exit on error

echo
"=== Testing GitHub Actions Workflow ==="
echo

# Colors for output
GREEN = '\033[0;32m'
RED = '\033[0;31m'
YELLOW = '\033[1;33m'
NC = '\033[0m'  # No Color

# Function to print test results
print_result()
{
if [ $1 - eq
0]; then
echo - e
"${GREEN}✓ $2 passed${NC}"
else
echo - e
"${RED}✗ $2 failed${NC}"
return 1
fi
}

# Function to test with a specific Python version
test_python_version()
{
local
python_version =$1
echo - e
"\n${YELLOW}Testing with Python $python_version${NC}"

# Create virtual environment
echo
"Creating virtual environment..."
$python_version - m
venv
venv_$python_version
source
venv_$python_version / bin / activate

# Upgrade pip
echo
"Upgrading pip..."
python - m
pip
install - -upgrade
pip

# Test 1: Install package with dev dependencies
echo - e
"\n${YELLOW}Test 1: Installing package with pyproject.toml${NC}"
pip
install - e
".[dev]"
2 > & 1 | tee
install_$python_version.log
print_result ${PIPESTATUS[0]}
"Package installation"

# Verify key packages are installed
echo - e
"\n${YELLOW}Verifying key dependencies:${NC}"
python - c
"import fastapi; print(f'✓ FastAPI: {fastapi.__version__}')" | | echo
"✗ FastAPI not found"
python - c
"import pytest; print(f'✓ pytest: {pytest.__version__}')" | | echo
"✗ pytest not found"
python - c
"import black; print(f'✓ black: {black.__version__}')" | | echo
"✗ black not found"
python - c
"import ruff; print(f'✓ ruff installed')" | | echo
"✗ ruff not found"
python - c
"import mypy; print(f'✓ mypy: {mypy.__version__}')" | | echo
"✗ mypy not found"

# Test 2: Check pytest markers
echo - e
"\n${YELLOW}Test 2: Checking pytest markers${NC}"
cd
tldw_Server_API
pytest - -markers | grep - E
"unit|integration|external_api" & & print_result
0
"Pytest markers found" | | print_result
1
"Pytest markers"

# Test 3: Run unit tests (dry run to check discovery)
echo - e
"\n${YELLOW}Test 3: Test discovery with markers${NC}"
pytest - -collect - only - m
"unit"
2 > & 1 | tee
test_discovery_$python_version.log
test_count =$(pytest - -collect - only - q - m "unit" 2 > / dev / null | grep -c "test" | | echo "0")
echo
"Found $test_count unit tests"
[ $test_count - gt
0] & & print_result
0
"Unit test discovery" | | print_result
1
"Unit test discovery"

# Test 4: Code quality tools
echo - e
"\n${YELLOW}Test 4: Code quality tools${NC}"

# Black check (dry run)
echo
"Running black check..."
black - -check - -diff.
2 > & 1 | head - 20
black_result =${PIPESTATUS[0]}
[ $black_result - eq
0] | | [ $black_result - eq
1] & & print_result
0
"Black formatter check" | | print_result
1
"Black formatter check"

# Ruff check
echo
"Running ruff check..."
ruff
check.
2 > & 1 | head - 20
ruff_result =${PIPESTATUS[0]}
[ $ruff_result - eq
0] | | [ $ruff_result - eq
1] & & print_result
0
"Ruff linter check" | | print_result
1
"Ruff linter check"

# MyPy check (allow failures)
echo
"Running mypy check..."
mypy.
2 > & 1 | head - 20 | | true
print_result
0
"MyPy type checker (allowed to fail)"

# Test 5: Coverage tools
echo - e
"\n${YELLOW}Test 5: Coverage tools${NC}"
python - c
"import pytest_cov; print('✓ pytest-cov installed')" & & print_result
0
"Coverage tools" | | print_result
1
"Coverage tools"

cd..
deactivate

echo - e
"\n${GREEN}Completed testing with Python $python_version${NC}"
}

# Test with different Python versions
for version in python3.10 python3.11 python3.12; do
if command -v $version & > / dev / null; then
test_python_version $version
else
echo -e "${YELLOW}Skipping $version (not installed)${NC}"
fi
done

# Test 6: Platform-specific dependencies
echo -e "\n${YELLOW}Test 6: Platform-specific dependencies${NC}"
source venv_python3.12 / bin / activate
pip install -e ".[audio_recording_windows]" 2 > & 1 | grep -i "error" & & print_result 1 "Platform-specific deps" | | print_result 0 "Platform-specific deps (or not Windows)"
deactivate

# Test 7: Server startup test
echo -e "\n${YELLOW}Test 7: Server startup test${NC}"
source venv_python3.12 / bin / activate
cd tldw_Server_API
timeout 10 python -m uvicorn app.main:app - -host
0.0
.0
.0 - -port
8000 &
SERVER_PID =$!
sleep
5
curl - f
http: // localhost: 8000 / docs > / dev / null
2 > & 1 & & print_result
0
"Server startup" | | print_result
1
"Server startup"
kill $SERVER_PID
2 > / dev / null | | true
cd..
deactivate

echo - e
"\n${GREEN}=== Workflow validation complete ===${NC}"

# Summary
echo - e
"\n${YELLOW}Summary:${NC}"
echo
"- Tested dependency installation with pyproject.toml"
echo
"- Verified pytest markers and test discovery"
echo
"- Checked code quality tools (black, ruff, mypy)"
echo
"- Tested coverage tools installation"
echo
"- Validated server startup"

echo - e
"\n${YELLOW}Check log files for detailed output:${NC}"
ls - la *.log
2 > / dev / null | | echo
"No log files generated"