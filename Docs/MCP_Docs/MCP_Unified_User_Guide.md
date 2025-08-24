# MCP Unified - User Guide

## Table of Contents
1. [Overview](#overview)
2. [Getting Started](#getting-started)
3. [Using the WebSocket Interface](#using-the-websocket-interface)
4. [Using the HTTP API](#using-the-http-api)
5. [Available Tools](#available-tools)
6. [Authentication](#authentication)
7. [Examples](#examples)
8. [Troubleshooting](#troubleshooting)

## Overview

The Model Context Protocol (MCP) Unified module provides a standardized interface for interacting with TLDW's media processing, search, and analysis capabilities. It supports both WebSocket (for real-time communication) and HTTP (for simpler integrations).

### Key Features
- 🔍 **Media Search**: Search through ingested content
- 📝 **Transcription Access**: Retrieve and search transcripts
- 🤖 **AI Integration**: Leverage LLM capabilities for analysis
- 📊 **RAG Search**: Advanced retrieval-augmented generation
- 🔐 **Secure Access**: JWT-based authentication with role-based permissions

## Getting Started

### Prerequisites
- TLDW server running with MCP Unified module enabled
- API credentials (if authentication is enabled)
- WebSocket or HTTP client

### Quick Start

1. **Check Server Health**
   ```bash
   curl http://localhost:8000/api/v1/mcp/health
   ```
   Response: `{"status": "healthy"}`

2. **List Available Tools**
   ```bash
   curl http://localhost:8000/api/v1/mcp/tools
   ```

## Using the WebSocket Interface

### Connection
Connect to the WebSocket endpoint:
```
ws://localhost:8000/api/v1/mcp/ws
```

Optional query parameters:
- `client_id`: Your client identifier
- `token`: JWT authentication token

### Example WebSocket Client (JavaScript)
```javascript
const ws = new WebSocket('ws://localhost:8000/api/v1/mcp/ws?client_id=my-app');

ws.onopen = () => {
    // Initialize connection
    ws.send(JSON.stringify({
        jsonrpc: "2.0",
        method: "initialize",
        params: {
            clientInfo: {
                name: "My Application",
                version: "1.0.0"
            }
        },
        id: 1
    }));
};

ws.onmessage = (event) => {
    const response = JSON.parse(event.data);
    console.log('Received:', response);
};

// Call a tool
function callTool(toolName, args) {
    ws.send(JSON.stringify({
        jsonrpc: "2.0",
        method: "tools/call",
        params: {
            name: toolName,
            arguments: args
        },
        id: Date.now()
    }));
}
```

### Example WebSocket Client (Python)
```python
import websocket
import json

def on_message(ws, message):
    response = json.loads(message)
    print(f"Received: {response}")

def on_open(ws):
    # Initialize connection
    ws.send(json.dumps({
        "jsonrpc": "2.0",
        "method": "initialize",
        "params": {
            "clientInfo": {
                "name": "Python Client",
                "version": "1.0.0"
            }
        },
        "id": 1
    }))

ws = websocket.WebSocketApp("ws://localhost:8000/api/v1/mcp/ws",
                            on_open=on_open,
                            on_message=on_message)
ws.run_forever()
```

## Using the HTTP API

### Basic Request Format
All HTTP requests use POST to `/api/v1/mcp/request`:

```bash
curl -X POST http://localhost:8000/api/v1/mcp/request \
  -H "Content-Type: application/json" \
  -d '{
    "jsonrpc": "2.0",
    "method": "tools/list",
    "params": {},
    "id": 1
  }'
```

### Convenience Endpoints

#### List Tools
```bash
GET /api/v1/mcp/tools
```

#### Execute Tool
```bash
POST /api/v1/mcp/tools/execute
{
    "tool_name": "media.search",
    "arguments": {
        "query": "machine learning",
        "limit": 10
    }
}
```

#### Server Status
```bash
GET /api/v1/mcp/status
```

## Available Tools

### Media Tools

#### media.search
Search for media content in the database.
```json
{
    "tool_name": "media.search",
    "arguments": {
        "query": "your search query",
        "limit": 10,
        "media_type": "video"  // optional: video, audio, document
    }
}
```

#### media.get_transcript
Retrieve transcript for a specific media item.
```json
{
    "tool_name": "media.get_transcript",
    "arguments": {
        "media_id": 123,
        "format": "text"  // or "srt", "vtt"
    }
}
```

### RAG Tools

#### rag.search
Perform semantic search using RAG.
```json
{
    "tool_name": "rag.search",
    "arguments": {
        "query": "explain quantum computing",
        "collection": "media_content",
        "top_k": 5
    }
}
```

#### rag.hybrid_search
Combine keyword and semantic search.
```json
{
    "tool_name": "rag.hybrid_search",
    "arguments": {
        "query": "climate change effects",
        "keywords": ["temperature", "CO2"],
        "limit": 10
    }
}
```

### Chat Tools

#### chat.complete
Get AI-powered completions.
```json
{
    "tool_name": "chat.complete",
    "arguments": {
        "messages": [
            {"role": "user", "content": "Summarize this transcript"}
        ],
        "context": "transcript_text_here"
    }
}
```

## Authentication

### Getting a Token
```bash
POST /api/v1/mcp/auth/token
{
    "username": "your_username",
    "password": "your_password"
}
```

Response:
```json
{
    "access_token": "eyJ...",
    "token_type": "bearer",
    "expires_in": 1800,
    "refresh_token": "..."
}
```

### Using the Token

#### WebSocket
Include as query parameter:
```
ws://localhost:8000/api/v1/mcp/ws?token=eyJ...
```

#### HTTP
Include in Authorization header:
```bash
curl -H "Authorization: Bearer eyJ..." \
  http://localhost:8000/api/v1/mcp/tools
```

## Examples

### Example 1: Search and Summarize Media

```python
import requests
import json

# Search for content
response = requests.post('http://localhost:8000/api/v1/mcp/tools/execute',
    json={
        "tool_name": "media.search",
        "arguments": {
            "query": "artificial intelligence",
            "limit": 5
        }
    }
)
results = response.json()

# Get transcript of first result
media_id = results['result'][0]['id']
transcript_response = requests.post('http://localhost:8000/api/v1/mcp/tools/execute',
    json={
        "tool_name": "media.get_transcript",
        "arguments": {
            "media_id": media_id,
            "format": "text"
        }
    }
)

# Summarize the transcript
summary_response = requests.post('http://localhost:8000/api/v1/mcp/tools/execute',
    json={
        "tool_name": "chat.complete",
        "arguments": {
            "messages": [
                {"role": "user", "content": "Summarize this content in 3 bullet points"}
            ],
            "context": transcript_response.json()['result']['transcript']
        }
    }
)

print(summary_response.json()['result'])
```

### Example 2: Real-time Monitoring

```javascript
const ws = new WebSocket('ws://localhost:8000/api/v1/mcp/ws');

// Subscribe to updates
ws.onopen = () => {
    ws.send(JSON.stringify({
        jsonrpc: "2.0",
        method: "subscribe",
        params: {
            events: ["media.added", "transcript.completed"]
        },
        id: 1
    }));
};

// Handle notifications
ws.onmessage = (event) => {
    const msg = JSON.parse(event.data);
    if (msg.method === "notification") {
        console.log(`Event: ${msg.params.event}`, msg.params.data);
    }
};
```

## Troubleshooting

### Common Issues

#### Connection Refused
- **Check**: Is the server running?
- **Solution**: Start the server with MCP module enabled
- **Verify**: `curl http://localhost:8000/api/v1/mcp/health`

#### Authentication Failed
- **Check**: Are environment variables set?
- **Solution**: Set `MCP_JWT_SECRET` and `MCP_API_KEY_SALT`
- **Verify**: Check server logs for authentication errors

#### Tool Not Found
- **Check**: Is the module registered?
- **Solution**: Verify module initialization in server logs
- **List tools**: `GET /api/v1/mcp/tools`

#### Rate Limit Exceeded
- **Check**: Current rate limits
- **Solution**: Implement exponential backoff
- **Headers**: Check `X-RateLimit-Remaining` in response

### Debug Mode
Enable debug logging:
```bash
export MCP_LOG_LEVEL=DEBUG
```

### Support
- Check server logs for detailed error messages
- Review API documentation at `http://localhost:8000/docs`
- Consult the Developer Guide for advanced usage