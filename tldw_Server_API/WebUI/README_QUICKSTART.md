# WebUI Quick Start Guide

## How to Start the WebUI

### Option 1: Using the Start Script (Recommended)

1. **Start the API Server** (in one terminal):
   ```bash
   cd /Users/appledev/Working/tldw_server
   python -m uvicorn tldw_Server_API.app.main:app --reload
   ```
   The API should be running at http://localhost:8000

2. **Start the WebUI** (in another terminal):
   ```bash
   cd /Users/appledev/Working/tldw_server/tldw_Server_API/WebUI
   ./start-webui.sh
   ```
   
3. **Open your browser** and go to:
   ```
   http://localhost:8081
   ```

### Option 2: Using Python Directly

```bash
cd /Users/appledev/Working/tldw_server/tldw_Server_API/WebUI
python3 -m http.server 8081
```

Then open http://localhost:8081 in your browser.

## Important Notes

⚠️ **DO NOT** open the index.html file directly in your browser (file:// protocol) as this will cause CORS errors!

✅ **ALWAYS** use a local HTTP server (the methods above) to serve the WebUI files.

## Troubleshooting

### If tabs aren't working:

1. **Check Browser Console** (F12):
   - Look for JavaScript errors
   - The components.js error has been fixed

2. **Clear Browser Cache**:
   - Hard refresh: Ctrl+Shift+R (Windows/Linux) or Cmd+Shift+R (Mac)
   - Or open Developer Tools → Network tab → check "Disable cache"

3. **Verify API is Running**:
   - Go to http://localhost:8000/docs
   - You should see the FastAPI documentation

4. **Check Network Tab**:
   - Open Developer Tools → Network tab
   - Look for failed requests
   - The /health endpoint should return 200 OK

### Common Issues:

1. **"CORS error"**: You're opening the file directly. Use the HTTP server method above.

2. **"Connection refused"**: The API server isn't running. Start it first.

3. **"404 Not Found"**: You're on the wrong port. Make sure you're using:
   - API: http://localhost:8000
   - WebUI: http://localhost:8081

4. **Tabs not switching**: Clear cache and reload. The JavaScript error has been fixed.

## First Time Setup

1. When the WebUI loads, you should see the "General" tab active
2. Enter your API token in the "API Token" field (if you have authentication enabled)
3. Click "Test Connection" to verify the API is reachable
4. Navigate to other tabs using the top navigation

## Available Tabs

- **General**: API configuration and settings
- **Media**: Media management and processing
- **Chat**: Chat completions interface
- **RAG**: Search and retrieval features
- **Prompts**: Prompt library management
- **Notes**: Note-taking system
- **Evaluations**: Model evaluation tools
- **Embeddings**: Embedding generation
- **Research**: ArXiv and Semantic Scholar search
- **Audio**: Text-to-speech
- **Admin**: User management
- **Sync**: Sync operations
- **Health**: System health monitoring
- **MCP**: Model Context Protocol tools
- **Llama.cpp**: Local LLM management
- **Web Scraping**: Web scraping service
- **Maintenance**: Database maintenance tools