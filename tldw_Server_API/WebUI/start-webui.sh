#!/bin/bash

echo "Testing TLDW WebUI..."
echo "====================="

# Check if all required JS files exist
echo "Checking JavaScript files..."
for file in utils.js components.js api-client.js endpoint-helper.js chat-ui.js main.js; do
    if [ -f "js/$file" ]; then
        echo "✓ js/$file exists"
    else
        echo "✗ js/$file missing!"
        exit 1
    fi
done

# Check if CSS exists
echo -e "\nChecking CSS..."
if [ -f "css/styles.css" ]; then
    echo "✓ css/styles.css exists"
else
    echo "✗ css/styles.css missing!"
    exit 1
fi

# Check if index.html exists
echo -e "\nChecking main HTML..."
if [ -f "index.html" ]; then
    echo "✓ index.html exists"
else
    echo "✗ index.html missing!"
    exit 1
fi

# Check tab content files
echo -e "\nChecking tab content files..."
for file in tabs/*.html; do
    if [ -f "$file" ]; then
        # Check for DOCTYPE in tab files (shouldn't exist)
        if grep -q "<!DOCTYPE html>" "$file"; then
            echo "✗ $(basename $file) has full HTML structure (should be fragment only)"
        else
            echo "✓ $(basename $file) is properly formatted"
        fi
    fi
done

# Check if Python 3 is installed
if ! command -v python3 &> /dev/null; then
    echo "Error: Python 3 is not installed"
    echo "Please install Python 3 to continue"
    exit 1
fi

# Auto-configure from environment if available
echo -e "\n====================="
echo "Checking for API configuration..."
if [ ! -z "$SINGLE_USER_API_KEY" ]; then
    echo "Found SINGLE_USER_API_KEY in environment, updating WebUI config..."
    cat > "webui-config.json" << EOF
{
  "apiUrl": "${API_URL:-http://localhost:8000}",
  "apiKey": "$SINGLE_USER_API_KEY",
  "_comment": "Auto-generated from environment variables. Do not commit to version control."
}
EOF
    echo "✓ WebUI config updated with API key"
else
    echo "No SINGLE_USER_API_KEY found in environment"
    echo "The WebUI will prompt for manual API key entry"
    echo ""
    echo "Tip: Set your API key with:"
    echo "  export SINGLE_USER_API_KEY='your-api-key'"
    echo "  Then restart this script"
fi

# Start a simple HTTP server
echo -e "\n====================="
echo "Starting WebUI server on http://localhost:8080"
echo "API endpoint: ${API_URL:-http://localhost:8000}"
if [ ! -z "$SINGLE_USER_API_KEY" ]; then
    echo "API Key: [AUTO-CONFIGURED]"
else
    echo "API Key: [MANUAL ENTRY REQUIRED]"
fi
echo ""
echo "Open http://localhost:8080 in your browser"
echo "Press Ctrl+C to stop the server"
echo "====================="

# Use Python's built-in HTTP server
python3 -m http.server 8080