# TLDW Server WebUI Module

## ⚠️ IMPORTANT SECURITY NOTICE

This WebUI module is currently **NOT PRODUCTION READY** and requires significant security improvements before deployment to customers.

## Overview

The WebUI module provides a browser-based testing interface for the TLDW Server API. It was developed by an external contractor and is intended as a development and testing tool.

## Current Status

### Security Issues (RESOLVED)
- ✅ Removed hardcoded API tokens from JavaScript
- ✅ Fixed XSS vulnerabilities by implementing proper HTML escaping
- ✅ Added secure utility functions for DOM manipulation

### Remaining Issues
- [ ] API endpoint compatibility needs verification with recent API changes
- [ ] Missing endpoints for new features (OpenAI evaluations, etc.)
- [ ] No production deployment configuration
- [ ] No authentication flow implementation
- [ ] Duplicate code across multiple HTML files needs consolidation

## Features

- Multi-tab interface for different API sections
- API request builder with form inputs
- Response viewer with JSON syntax highlighting
- Request history tracking
- Dark/light theme support
- cURL command generation

## Setup

1. **Configure API Connection**:
   - Open the WebUI in a browser
   - Navigate to "Global Settings" tab
   - Enter your API base URL (default: `http://localhost:8000`)
   - Enter your API token (no default provided for security)

2. **CORS Configuration**:
   - Ensure the API server has appropriate CORS settings to allow WebUI connections
   - Current CORS configuration in `app/main.py` allows `["*"]` for development

## File Structure

```
WebUI/
├── index.html              # Main application entry point
├── api-endpoints-config.json # API endpoint documentation
├── css/
│   └── styles.css         # Application styles
├── js/
│   ├── api-client.js      # API communication layer
│   ├── chat-ui.js         # Chat interface functionality
│   ├── components.js      # Reusable UI components
│   ├── main.js           # Main application logic
│   └── utils.js          # Utility functions (includes security fixes)
└── tabs/                  # Tab content HTML files
    ├── admin_content.html
    ├── audio_content.html
    ├── chat_content.html
    ├── embeddings_content.html
    ├── evaluations_content.html
    ├── general_content.html
    ├── media_content.html
    ├── notes_content.html
    ├── prompts_content.html
    ├── rag_content.html
    ├── research_content.html
    └── sync_content.html
```

## Security Improvements Made

1. **Removed Hardcoded Secrets**: 
   - API tokens are no longer hardcoded in JavaScript
   - Users must provide their own tokens

2. **XSS Prevention**:
   - Implemented proper HTML escaping in `utils.js`
   - Replaced dangerous `innerHTML` usage with safe DOM manipulation
   - Added `safeSetHTML()` and `createSafeElement()` utility functions

3. **Input Validation**:
   - Added type checking for string inputs
   - Implemented attribute whitelisting for DOM elements

## Recommended Next Steps

### High Priority
1. **API Compatibility Testing**:
   - Verify all endpoints against current API implementation
   - Update endpoints that have changed
   - Add missing endpoints for new features

2. **Authentication Implementation**:
   - Add proper login flow
   - Implement JWT token handling
   - Add session management

3. **Production Configuration**:
   - Remove development defaults
   - Add environment-specific configurations
   - Implement proper error handling for production

### Medium Priority
1. **Code Refactoring**:
   - Consolidate duplicate code from tab HTML files
   - Implement a proper build system (webpack/vite)
   - Add TypeScript for better type safety

2. **Testing**:
   - Add unit tests for JavaScript modules
   - Add integration tests for API interactions
   - Implement E2E testing

3. **Documentation**:
   - Add JSDoc comments to all functions
   - Create user guide for the WebUI
   - Document API integration points

### Low Priority
1. **UI Improvements**:
   - Modernize the design
   - Add loading states for all operations
   - Improve mobile responsiveness

2. **Features**:
   - Add API response caching
   - Implement request templates
   - Add export functionality for request history

## Usage Warning

⚠️ **This WebUI should only be used in development environments** until all security issues are resolved and production configuration is implemented.

## Development

To run the WebUI locally:

1. Start the TLDW Server API:
   ```bash
   python -m uvicorn tldw_Server_API.app.main:app --reload
   ```

2. Open `index.html` in a web browser or serve it with a local HTTP server:
   ```bash
   python -m http.server 8080 --directory tldw_Server_API/WebUI
   ```

3. Navigate to `http://localhost:8080` in your browser

## Contributing

When making changes to the WebUI:

1. Follow the security guidelines in this README
2. Test all changes against the current API
3. Update the `api-endpoints-config.json` when API changes occur
4. Ensure no sensitive information is hardcoded
5. Use the safe utility functions for DOM manipulation

## License

This module is part of the TLDW Server project and follows the same licensing terms.