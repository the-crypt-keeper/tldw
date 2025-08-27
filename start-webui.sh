#!/bin/bash

# Simple startup script for TLDW WebUI
# Run this from the project root directory

echo "===========================================" 
echo "      TLDW Server - WebUI Launcher"
echo "==========================================="
echo ""

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Check if we're in the right directory
if [ ! -f "tldw_Server_API/app/main.py" ]; then
    echo -e "${RED}Error:${NC} Please run this script from the tldw_server project root directory."
    exit 1
fi

# Function to check if port is in use
check_port() {
    if lsof -Pi :8000 -sTCP:LISTEN -t >/dev/null 2>&1; then
        return 0
    else
        return 1
    fi
}

# Check if API server is already running
echo "Checking API server status..."
http_code=$(curl -s -o /dev/null -w "%{http_code}" http://localhost:8000/health 2>/dev/null || echo "000")
if [ "$http_code" = "200" ]; then
    echo -e "${GREEN}✓${NC} API server is already running"
    SERVER_PID=""
else
    echo -e "${YELLOW}○${NC} API server is not running"
    echo ""
    
    # Check if port is in use but not responding to health check
    if check_port; then
        echo -e "${RED}Warning:${NC} Port 8000 is in use but API is not responding."
        echo "You may need to stop the existing process first."
        echo ""
        read -p "Try to start the API server anyway? (y/n) " -n 1 -r
        echo
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            exit 1
        fi
    fi
    
    echo "Starting API server..."
    
    # Start the server in background
    SINGLE_USER_API_KEY="${SINGLE_USER_API_KEY}" AUTH_MODE="${AUTH_MODE:-single_user}" \
        python -m uvicorn tldw_Server_API.app.main:app --host 0.0.0.0 --port 8000 --reload > /tmp/tldw_server.log 2>&1 &
    SERVER_PID=$!
    
    echo -e "API server starting (PID: ${SERVER_PID})..."
    
    # Wait for server to be ready
    echo -n "Waiting for API to be ready"
    for i in {1..30}; do
        http_code=$(curl -s -o /dev/null -w "%{http_code}" http://localhost:8000/health 2>/dev/null || echo "000")
        if [ "$http_code" = "200" ]; then
            echo -e " ${GREEN}✓${NC}"
            break
        fi
        echo -n "."
        sleep 1
    done
    
    # Check if server started successfully
    http_code=$(curl -s -o /dev/null -w "%{http_code}" http://localhost:8000/health 2>/dev/null || echo "000")
    if [ "$http_code" != "200" ]; then
        echo -e " ${RED}✗${NC}"
        echo ""
        echo -e "${RED}Error:${NC} Failed to start API server (HTTP code: $http_code)."
        echo "Check the logs at /tmp/tldw_server.log for details."
        exit 1
    fi
fi

echo ""
echo "==========================================="
echo -e "${GREEN}Server Status:${NC}"
echo "==========================================="
echo ""

# Display URLs
echo -e "${BLUE}Access URLs:${NC}"
echo -e "  WebUI:    ${GREEN}http://localhost:8000/webui/${NC}"
echo -e "  API Docs: ${GREEN}http://localhost:8000/docs${NC}"
echo -e "  ReDoc:    ${GREEN}http://localhost:8000/redoc${NC}"
echo ""

# Display authentication info
if [ ! -z "$SINGLE_USER_API_KEY" ]; then
    echo -e "${BLUE}Authentication:${NC}"
    echo -e "  Mode: ${GREEN}Single-User (Auto-configured)${NC}"
    echo -e "  The WebUI will automatically use your API key."
    echo ""
    # Show first 10 chars of API key for confirmation
    KEY_PREVIEW="${SINGLE_USER_API_KEY:0:10}..."
    echo -e "  API Key Preview: ${YELLOW}${KEY_PREVIEW}${NC}"
elif [ "$AUTH_MODE" = "multi_user" ]; then
    echo -e "${BLUE}Authentication:${NC}"
    echo -e "  Mode: ${YELLOW}Multi-User${NC}"
    echo -e "  Please log in through the WebUI to obtain a token."
else
    echo -e "${BLUE}Authentication:${NC}"
    echo -e "  Mode: ${YELLOW}Manual Configuration Required${NC}"
    echo ""
    echo "  To enable auto-configuration:"
    echo -e "    ${YELLOW}export SINGLE_USER_API_KEY='your-api-key'${NC}"
    echo -e "    ${YELLOW}./start-webui.sh${NC}"
fi

echo ""
echo "==========================================="
echo ""

# Open browser
echo -e "${GREEN}Opening WebUI in your default browser...${NC}"

# Detect OS and open browser
if [[ "$OSTYPE" == "linux-gnu"* ]]; then
    xdg-open "http://localhost:8000/webui/" 2>/dev/null &
elif [[ "$OSTYPE" == "darwin"* ]]; then
    open "http://localhost:8000/webui/" 2>/dev/null &
elif [[ "$OSTYPE" == "cygwin" ]] || [[ "$OSTYPE" == "msys" ]] || [[ "$OSTYPE" == "win32" ]]; then
    start "http://localhost:8000/webui/" 2>/dev/null &
else
    echo "Please manually open: http://localhost:8000/webui/"
fi

echo ""

# If we started the server, offer to monitor it
if [ ! -z "$SERVER_PID" ]; then
    echo "==========================================="
    echo -e "${YELLOW}Server Management:${NC}"
    echo "==========================================="
    echo ""
    echo "The API server is running in the background."
    echo ""
    echo -e "To stop the server:  ${YELLOW}kill $SERVER_PID${NC}"
    echo -e "To view logs:        ${YELLOW}tail -f /tmp/tldw_server.log${NC}"
    echo ""
    echo "Press Ctrl+C to exit this launcher (server will keep running)."
    echo ""
    
    # Keep script running but don't block
    trap "echo ''; echo 'Launcher exiting. Server still running (PID: $SERVER_PID)'; exit 0" INT
    
    # Wait indefinitely
    while true; do
        sleep 1
    done
else
    echo "Press Ctrl+C to exit."
    echo ""
fi