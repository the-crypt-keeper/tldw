#!/bin/bash

# Start script for TLDW WebUI
# This script starts a local HTTP server to serve the WebUI files
# This avoids CORS issues when accessing the API

echo "Starting TLDW WebUI..."
echo "========================================"
echo ""

# Check if Python 3 is installed
if ! command -v python3 &> /dev/null; then
    echo "Error: Python 3 is not installed"
    echo "Please install Python 3 to continue"
    exit 1
fi

# Get the directory of this script
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# Change to the WebUI directory
cd "$DIR"

echo "Starting local web server on port 8081..."
echo "WebUI will be available at: http://localhost:8081"
echo ""
echo "Make sure the TLDW API server is running on http://localhost:8000"
echo ""
echo "Press Ctrl+C to stop the server"
echo "========================================"
echo ""

# Start Python HTTP server
python3 -m http.server 8081