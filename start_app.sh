#!/bin/bash
# Startup script for tldw application with error handling

echo "========================================="
echo "Starting tldw Application"
echo "========================================="

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Function to check command existence
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Check Python version
echo -e "${YELLOW}Checking Python version...${NC}"
if command_exists python3; then
    PYTHON_VERSION=$(python3 -c 'import sys; print(".".join(map(str, sys.version_info[:2])))')
    echo -e "${GREEN}✓ Python $PYTHON_VERSION found${NC}"
    PYTHON_CMD="python3"
elif command_exists python; then
    PYTHON_VERSION=$(python -c 'import sys; print(".".join(map(str, sys.version_info[:2])))')
    echo -e "${GREEN}✓ Python $PYTHON_VERSION found${NC}"
    PYTHON_CMD="python"
else
    echo -e "${RED}✗ Python not found. Please install Python 3.8 or higher.${NC}"
    exit 1
fi

# Check ffmpeg
echo -e "${YELLOW}Checking ffmpeg...${NC}"
if command_exists ffmpeg; then
    echo -e "${GREEN}✓ ffmpeg found${NC}"
else
    echo -e "${YELLOW}⚠ ffmpeg not found. Audio/video processing may not work.${NC}"
    echo "Install with: brew install ffmpeg (macOS) or sudo apt-get install ffmpeg (Linux)"
fi

# Create necessary directories
echo -e "${YELLOW}Creating directories...${NC}"
mkdir -p Databases Logs Config_Files
echo -e "${GREEN}✓ Directories created${NC}"

# Check for virtual environment
if [ -n "$VIRTUAL_ENV" ]; then
    echo -e "${GREEN}✓ Virtual environment active: $VIRTUAL_ENV${NC}"
else
    echo -e "${YELLOW}⚠ No virtual environment detected${NC}"
    echo "Consider creating one with: python3 -m venv venv && source venv/bin/activate"
fi

# Install/update dependencies
echo -e "${YELLOW}Checking dependencies...${NC}"
if [ -f "requirements.txt" ]; then
    echo "Installing/updating dependencies..."
    $PYTHON_CMD -m pip install --quiet --upgrade pip
    $PYTHON_CMD -m pip install --quiet -r requirements.txt
    echo -e "${GREEN}✓ Dependencies installed${NC}"
else
    echo -e "${RED}✗ requirements.txt not found${NC}"
fi

# Run health check
echo -e "${YELLOW}Running health check...${NC}"
if [ -f "check_app_health.py" ]; then
    $PYTHON_CMD check_app_health.py
else
    echo -e "${YELLOW}⚠ Health check script not found${NC}"
fi

# Start the application
echo -e "\n${GREEN}Starting tldw GUI...${NC}"
echo "========================================="

# Check if fixed version exists
if [ -f "app_fixed.py" ]; then
    echo "Using enhanced launcher..."
    $PYTHON_CMD app_fixed.py -gui "$@"
else
    echo "Using standard launcher..."
    $PYTHON_CMD summarize.py -gui "$@"
fi