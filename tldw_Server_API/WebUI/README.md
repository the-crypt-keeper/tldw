# TLDW Server WebUI

A browser-based interface for testing and interacting with the TLDW Server API. This tool serves as an internal API documentation and testing platform, similar to Swagger or Postman, designed for single-user maintenance and development use.

## Quick Start

### Prerequisites
- TLDW Server API running (default: `http://localhost:8000`)
- Modern web browser (Chrome, Firefox, Safari, Edge)
- Python 3.x (for serving the WebUI)

### Starting the WebUI

1. **Start the API Server** (in one terminal):
   ```bash
   cd /path/to/tldw_server
   # Set your API key (if using single-user mode)
   export SINGLE_USER_API_KEY="your-secret-api-key"
   python -m uvicorn tldw_Server_API.app.main:app --reload
   ```
   The API will be available at http://localhost:8000

2. **Start the WebUI** (in another terminal):
   
   **Option A: With Auto-Configuration (Recommended)**
   ```bash
   cd tldw_Server_API/WebUI
   # The script will auto-detect SINGLE_USER_API_KEY from environment
   ./Start-WebUI.sh
   ```
   
   **Option B: Manual Configuration**
   ```bash
   cd tldw_Server_API/WebUI
   python3 -m http.server 8080
   # You'll need to enter the API key manually in the UI
   ```
   
   **Option C: With Custom API URL**
   ```bash
   cd tldw_Server_API/WebUI
   export API_URL="http://your-server:8000"
   export SINGLE_USER_API_KEY="your-api-key"
   ./Start-WebUI.sh
   ```

3. **Open your browser** and navigate to:
   ```
   http://localhost:8080
   ```

⚠️ **Important**: Do NOT open `index.html` directly in your browser (file:// protocol) as this will cause CORS errors. Always use an HTTP server.

## Overview

The WebUI is a comprehensive testing interface for the TLDW Server API, providing:
- **Interactive API Testing**: Send requests and view responses for all API endpoints
- **Documentation**: See available endpoints with example payloads
- **Maintenance Tools**: Database operations, cleanup, backup/restore
- **Development Aid**: cURL command generation, request history, JSON viewer

### Use Case
This tool is designed for:
- Single-user operation on private networks
- API functionality verification
- Development and debugging
- System maintenance tasks
- Internal documentation reference

## Features

### Core Functionality
- **17 API Sections**: Complete coverage of all API endpoints
- **Request Builder**: Form-based inputs with validation
- **Response Viewer**: Syntax-highlighted JSON with collapsible sections
- **Request History**: Track and replay previous API calls
- **cURL Generation**: Export requests as cURL commands
- **Theme Support**: Dark/light mode with persistent preference

### API Sections Available
- **General**: Global settings and debug tools
- **Media**: Media management, versioning, and processing
- **Chat**: OpenAI-compatible chat completions
- **RAG**: Search and retrieval-augmented generation
- **Prompts**: Prompt library management
- **Notes**: Note-taking and knowledge management
- **Evaluations**: Model evaluation tools
- **Embeddings**: Vector embedding generation
- **Research**: ArXiv and Semantic Scholar integration
- **Audio**: Text-to-speech functionality
- **Admin**: User management and system administration
- **Sync**: Synchronization operations
- **Health**: System health monitoring and diagnostics
- **MCP**: Model Context Protocol tools
- **Llama.cpp**: Local LLM server management
- **Web Scraping**: Web content ingestion service
- **Maintenance**: Database maintenance and batch operations

### Recent Improvements

**v1.2.0 - Auto-Configuration Update**
- ✅ Added automatic API key detection from environment variables
- ✅ WebUI auto-populates credentials when running alongside server
- ✅ Configuration file support (webui-config.json)
- ✅ Visual indicators for auto-configured settings
- ✅ Simplified startup for local installations

**v1.1.0 - Stability Fixes**
- ✅ Fixed component initialization errors
- ✅ Removed debug/test files from production
- ✅ Cleaned up duplicate HTML structure in tab files
- ✅ Enhanced connection status indicator with response times
- ✅ Added fallback error handling for missing components
- ✅ Improved error messages and user feedback

## File Structure

```
WebUI/
├── index.html                 # Main application entry point
├── api-endpoints-config.json  # API endpoint documentation
├── webui-config.json         # Auto-generated configuration (gitignored)
├── Start-WebUI.sh            # Start script with auto-configuration
├── test-ui.sh                # Testing and verification script
├── css/
│   └── styles.css            # Application styles with theme support
├── js/
│   ├── api-client.js         # API communication layer
│   ├── chat-ui.js            # Chat interface functionality
│   ├── components.js         # Reusable UI components (Toast, Modal, etc.)
│   ├── endpoint-helper.js    # Endpoint configuration helper
│   ├── main.js              # Main application logic
│   └── utils.js             # Utility functions with security features
└── tabs/                     # Tab content HTML fragments
    ├── admin_content.html
    ├── audio_content.html
    ├── chat_content.html
    ├── embeddings_content.html
    ├── evaluations_content.html
    ├── general_content.html
    ├── health_content.html
    ├── llamacpp_content.html
    ├── maintenance_content.html
    ├── mcp_content.html
    ├── media_content.html
    ├── notes_content.html
    ├── prompts_content.html
    ├── rag_content.html
    ├── research_content.html
    ├── sync_content.html
    └── webscraping_content.html
```

## Configuration

### Auto-Configuration (New in v1.2.0)
The WebUI now supports automatic configuration when running alongside a TLDW server installation:

1. **Environment Variables**: Set these before starting the WebUI:
   - `SINGLE_USER_API_KEY`: Your API authentication token
   - `API_URL`: Custom API server URL (optional, defaults to http://localhost:8000)

2. **Auto-Detection**: When using `Start-WebUI.sh`, the script will:
   - Check for `SINGLE_USER_API_KEY` in environment
   - Generate a `webui-config.json` file automatically
   - Pre-populate the API key in the UI
   - Show "✓ Auto-configured" indicator in the UI

3. **Configuration File**: The `webui-config.json` file:
   - Created automatically from environment variables
   - Loaded by the WebUI on startup
   - Excluded from version control (.gitignore)
   - Can be edited manually if needed

### Manual Configuration
If not using auto-configuration:
1. Open the WebUI in your browser
2. Navigate to the **General** tab (opens by default)
3. Configure your API settings:
   - **API Base URL**: Default is `http://localhost:8000`
   - **API Token**: Enter your authentication token
4. Click **Test Connection** to verify API connectivity

### Connection Status Indicators
- 🟢 **Green**: Connected and responsive
- 🟠 **Orange**: Connected but slow (>1000ms response time)
- 🔴 **Red**: Disconnected or error
- Hover over the status for detailed information

## Troubleshooting

### Common Issues

**"CORS error" in console**
- You're opening the file directly. Use the HTTP server method (see Quick Start)

**"Connection refused" or "API Unreachable"**
- Ensure the API server is running
- Check the API URL in Global Settings
- Verify: http://localhost:8000/docs shows the FastAPI documentation

**"404 Not Found" errors**
- Check you're using the correct ports:
  - API: http://localhost:8000
  - WebUI: http://localhost:8080 (or your chosen port)

**Tabs not loading**
- Clear browser cache: Ctrl+Shift+R (Windows/Linux) or Cmd+Shift+R (Mac)
- Check browser console (F12) for JavaScript errors
- Verify all JS files are loading in Network tab

**"Auth Failed" status**
- Check your API token in Global Settings
- Ensure the token matches your API configuration

### Browser Console
Press F12 to open Developer Tools and check:
- **Console tab**: JavaScript errors or warnings
- **Network tab**: Failed requests or slow responses
- **Application tab**: Local storage and session data

## Security Notes

For single-user internal use, the current security measures are adequate:
- No hardcoded API tokens
- XSS protection through HTML escaping
- Safe DOM manipulation utilities
- Input validation on forms

For multi-user or public deployment, additional security measures would be needed (authentication flow, CSRF protection, etc.).

## Development

### Running Tests
```bash
cd tldw_Server_API/WebUI
./test-ui.sh
```
This script verifies:
- All required files exist
- Tab HTML files are properly formatted
- No debug files remain in production

### Making Changes
When modifying the WebUI:
1. Test against the current API implementation
2. Update `api-endpoints-config.json` for API changes
3. Use the safe utility functions in `utils.js` for DOM manipulation
4. Ensure no sensitive information is hardcoded
5. Run the test script to verify structure

### API Compatibility
The WebUI is designed to work with the TLDW Server API v0.1.0+. Check `/api/v1/docs` for the current API specification.

## Tips for Effective Use

1. **Save Common Requests**: Use browser bookmarks to save specific tab states
2. **Export Requests**: Use the cURL generation to save complex requests
3. **Monitor Performance**: Watch the connection status for API response times
4. **Use Request History**: Access previous requests with Ctrl+Shift+H
5. **Keyboard Shortcuts**:
   - `Ctrl/Cmd + K`: Search endpoints
   - `Ctrl/Cmd + Shift + D`: Toggle dark mode
   - `Ctrl/Cmd + Shift + H`: Show request history
   - `Escape`: Close modals

## Support

For issues or questions:
1. Check the browser console for errors
2. Verify the API is running and accessible
3. Review the troubleshooting section above
4. Check the main TLDW Server documentation

## License

This module is part of the TLDW Server project and follows the same licensing terms (AGPL-3.0 / Commercial dual license).

---

*Version 1.2.0 - Auto-configuration support for local installations*