# CORS Solution for TLDW WebUI

## The Problem
When the WebUI is served from a different origin (different protocol, domain, or port) than the API server, browsers block requests due to CORS (Cross-Origin Resource Sharing) policy.

**Common scenarios that cause CORS errors:**
- Opening `index.html` directly as a file (`file://` protocol)
- Serving WebUI from port 8080 while API is on port 8000
- Using different domains (e.g., `localhost` vs `127.0.0.1`)

## The Solution: Same-Origin Serving

The WebUI is now served directly from the FastAPI server at the same origin, completely eliminating CORS issues.

## How to Access the WebUI

### Method 1: Automatic (Recommended)
```bash
cd tldw_Server_API/WebUI
./Start-WebUI-SameOrigin.sh
```
This script will:
- Check if the API server is running
- Open the WebUI in your browser at http://localhost:8000/webui/
- Auto-configure API keys if set in environment

### Method 2: Manual
1. Start the API server:
   ```bash
   python -m uvicorn tldw_Server_API.app.main:app --reload
   ```

2. Open your browser to: http://localhost:8000/webui/

## Benefits of Same-Origin Serving

✅ **No CORS Issues** - All requests are same-origin  
✅ **Better Security** - No need to allow cross-origin requests  
✅ **Simpler Configuration** - No CORS headers needed  
✅ **Single Port** - Everything runs on port 8000  
✅ **Production Ready** - This is how it should be deployed  

## API Key Configuration

### Option 1: Environment Variable (Recommended)
```bash
export SINGLE_USER_API_KEY='your-api-key-here'
./Start-WebUI-SameOrigin.sh
```

### Option 2: Manual Entry
Enter your API key in the WebUI's Global Settings tab.

## Alternative: Enable CORS (Not Recommended)

If you must serve the WebUI from a different origin, you need to configure CORS properly in `main.py`:

```python
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:8080"],  # Specific origin
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
```

**Security Warning**: Using `allow_origins=["*"]` is insecure and should never be used in production.

## URLs When Using Same-Origin

- **WebUI**: http://localhost:8000/webui/
- **API Docs**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc
- **API Base**: http://localhost:8000/api/v1/

## Troubleshooting

### WebUI Not Loading
1. Check that the API server is running
2. Clear browser cache
3. Check browser console for errors

### API Key Issues
1. Ensure the key is set correctly in environment or WebUI
2. Check that the API server has authentication enabled

### Still Getting CORS Errors
1. Make sure you're accessing via http://localhost:8000/webui/ (not file://)
2. Clear browser cache and cookies
3. Check that no browser extensions are interfering

## Development vs Production

### Development
Same-origin serving works perfectly for development and is the recommended approach.

### Production
In production, you should:
1. Serve everything through a reverse proxy (nginx, Apache)
2. Use HTTPS for security
3. Configure proper authentication
4. Never expose the API directly to the internet

## Summary

The CORS issue has been solved by serving the WebUI directly from the FastAPI server. This is the simplest, most secure, and most reliable solution. Just access the WebUI at:

**http://localhost:8000/webui/**

No additional configuration needed! 🎉