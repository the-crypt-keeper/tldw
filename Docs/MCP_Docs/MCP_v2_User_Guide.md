# MCP v2 User Guide

## Table of Contents
1. [Introduction](#introduction)
2. [What is MCP?](#what-is-mcp)
3. [Getting Started](#getting-started)
4. [Available Modules](#available-modules)
5. [Authentication](#authentication)
6. [Using the API](#using-the-api)
7. [WebSocket Connection](#websocket-connection)
8. [Rate Limits](#rate-limits)
9. [Common Use Cases](#common-use-cases)
10. [Troubleshooting](#troubleshooting)

## Introduction

The Model Context Protocol (MCP) v2 is a powerful interface that allows AI assistants and applications to interact with tldw_server's comprehensive media processing, knowledge management, and AI capabilities through a standardized protocol.

MCP v2 provides a unified way to:
- Search and retrieve media content
- Manage notes and documentation
- Work with prompt templates
- Perform transcriptions
- Generate embeddings and perform RAG operations
- Interact with various LLM providers

## What is MCP?

MCP (Model Context Protocol) is an open protocol that enables seamless integration between AI systems and application capabilities. It uses JSON-RPC 2.0 for communication and supports both HTTP and WebSocket connections.

### Key Benefits:
- **Standardized Interface**: Consistent API across all modules
- **Tool Discovery**: Automatically discover available capabilities
- **Resource Management**: Access and manage various data resources
- **Real-time Communication**: WebSocket support for streaming operations
- **Security**: Built-in authentication and authorization

## Getting Started

### Prerequisites
1. tldw_server running (default: `http://localhost:8001`)
2. API credentials (if authentication is enabled)
3. HTTP client or WebSocket library for your programming language

### Quick Start

1. **Check Server Status**:
```bash
curl http://localhost:8001/api/v1/mcp/v2/status
```

2. **List Available Tools**:
```bash
curl http://localhost:8001/api/v1/mcp/v2/tools
```

3. **Make Your First Request**:
```bash
curl -X POST http://localhost:8001/api/v1/mcp/v2/request \
  -H "Content-Type: application/json" \
  -d '{
    "jsonrpc": "2.0",
    "method": "tools/call",
    "params": {
      "name": "media.search_media",
      "arguments": {
        "query": "machine learning",
        "limit": 5
      }
    },
    "id": "1"
  }'
```

## Available Modules

### 1. Media Module
Manage and search your media library.

**Key Tools:**
- `search_media` - Search content using keywords or semantic search
- `get_transcript` - Retrieve transcripts for media items
- `get_media_metadata` - Get detailed metadata
- `ingest_media` - Add new media from URLs
- `get_media_summary` - Generate or retrieve summaries
- `list_recent_media` - View recently added content

### 2. RAG Module
Advanced retrieval-augmented generation capabilities.

**Key Tools:**
- `vector_search` - Semantic search using embeddings
- `hybrid_search` - Combined keyword and vector search
- `get_context` - Retrieve context for prompts
- `create_embedding` - Generate embeddings for text
- `rerank_results` - Re-rank search results
- `contextual_retrieval` - Context-aware retrieval

### 3. Notes Module
Personal knowledge management system.

**Key Tools:**
- `create_note` - Create new notes with markdown support
- `update_note` - Modify existing notes
- `search_notes` - Search by keywords or tags
- `get_note` - Retrieve specific notes
- `delete_note` - Remove notes (with recovery option)
- `list_notes` - Browse notes with pagination
- `export_note` - Export in various formats (MD, HTML, PDF)

### 4. Prompts Module
Manage reusable prompt templates.

**Key Tools:**
- `create_prompt` - Create prompt templates
- `get_prompt` - Retrieve and fill templates
- `search_prompts` - Find prompts by category or tags
- `update_prompt` - Modify templates
- `delete_prompt` - Remove prompts
- `import_prompts` - Import from JSON
- `export_prompts` - Export prompt libraries

### 5. Transcription Module
Audio and video transcription services.

**Key Tools:**
- `transcribe_audio` - Transcribe local audio/video files
- `transcribe_url` - Transcribe from URLs
- `detect_language` - Identify audio language
- `translate_transcript` - Translate transcripts
- `generate_subtitles` - Create SRT/VTT files
- `improve_transcript` - Post-process transcripts
- `split_transcript` - Segment transcripts

### 6. Chat Module
LLM interaction and conversation management.

**Key Tools:**
- `chat_completion` - Generate completions
- `create_conversation` - Start new conversations
- `add_message` - Add to conversations
- `get_conversation` - Retrieve chat history
- `list_providers` - View available LLM providers
- `stream_completion` - Streaming responses
- `manage_context` - Context window management

## Authentication

### Using JWT Tokens

If authentication is enabled, you'll need to include a JWT token in your requests:

```bash
curl -X POST http://localhost:8001/api/v1/mcp/v2/request \
  -H "Authorization: Bearer YOUR_JWT_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"jsonrpc": "2.0", "method": "tools/list", "id": "1"}'
```

### Obtaining Tokens

1. **Login** (if authentication system is configured):
```bash
curl -X POST http://localhost:8001/api/v1/auth/login \
  -H "Content-Type: application/json" \
  -d '{"username": "your_username", "password": "your_password"}'
```

2. **Token Refresh**:
```bash
curl -X POST http://localhost:8001/api/v1/auth/refresh \
  -H "Authorization: Bearer YOUR_REFRESH_TOKEN"
```

## Using the API

### Request Format

All MCP requests follow the JSON-RPC 2.0 format:

```json
{
  "jsonrpc": "2.0",
  "method": "method_name",
  "params": {
    // method-specific parameters
  },
  "id": "unique_request_id"
}
```

### Common Methods

#### 1. List Tools
```json
{
  "jsonrpc": "2.0",
  "method": "tools/list",
  "params": {},
  "id": "1"
}
```

#### 2. Call a Tool
```json
{
  "jsonrpc": "2.0",
  "method": "tools/call",
  "params": {
    "name": "module.tool_name",
    "arguments": {
      // tool-specific arguments
    }
  },
  "id": "2"
}
```

#### 3. List Resources
```json
{
  "jsonrpc": "2.0",
  "method": "resources/list",
  "params": {},
  "id": "3"
}
```

#### 4. Read a Resource
```json
{
  "jsonrpc": "2.0",
  "method": "resources/read",
  "params": {
    "uri": "resource://path"
  },
  "id": "4"
}
```

### Response Format

Successful responses:
```json
{
  "jsonrpc": "2.0",
  "result": {
    // method-specific result
  },
  "id": "request_id"
}
```

Error responses:
```json
{
  "jsonrpc": "2.0",
  "error": {
    "code": -32601,
    "message": "Method not found",
    "data": {
      // additional error details
    }
  },
  "id": "request_id"
}
```

## WebSocket Connection

For real-time communication and streaming operations:

### JavaScript Example
```javascript
const ws = new WebSocket('ws://localhost:8001/api/v1/mcp/v2/ws');

ws.onopen = () => {
  // Send initialization
  ws.send(JSON.stringify({
    jsonrpc: "2.0",
    method: "initialize",
    params: {
      clientInfo: {
        name: "My App",
        version: "1.0.0"
      }
    },
    id: "init"
  }));
};

ws.onmessage = (event) => {
  const response = JSON.parse(event.data);
  console.log('Received:', response);
};

// Call a tool
ws.send(JSON.stringify({
  jsonrpc: "2.0",
  method: "tools/call",
  params: {
    name: "chat.stream_completion",
    arguments: {
      messages: [
        {role: "user", content: "Hello!"}
      ],
      stream: true
    }
  },
  id: "chat-1"
}));
```

### Python Example
```python
import asyncio
import websockets
import json

async def mcp_client():
    uri = "ws://localhost:8001/api/v1/mcp/v2/ws"
    
    async with websockets.connect(uri) as websocket:
        # Initialize
        await websocket.send(json.dumps({
            "jsonrpc": "2.0",
            "method": "initialize",
            "params": {
                "clientInfo": {
                    "name": "Python Client",
                    "version": "1.0.0"
                }
            },
            "id": "init"
        }))
        
        # Receive initialization response
        response = await websocket.recv()
        print(f"Init response: {response}")
        
        # Call a tool
        await websocket.send(json.dumps({
            "jsonrpc": "2.0",
            "method": "tools/call",
            "params": {
                "name": "media.search_media",
                "arguments": {
                    "query": "python",
                    "limit": 3
                }
            },
            "id": "search-1"
        }))
        
        # Get result
        result = await websocket.recv()
        print(f"Search result: {result}")

asyncio.run(mcp_client())
```

## Rate Limits

MCP v2 implements rate limiting to ensure fair usage:

- **Default Limits**: 100 requests per minute
- **Burst Allowance**: Up to 10 requests in quick succession
- **Per-User Tracking**: Limits apply per authenticated user

### Rate Limit Headers

Responses include rate limit information:
- `X-RateLimit-Limit`: Maximum requests allowed
- `X-RateLimit-Remaining`: Requests remaining
- `X-RateLimit-Reset`: Time when limit resets (Unix timestamp)

### Handling Rate Limits

When rate limited, you'll receive a 429 status code:
```json
{
  "error": {
    "code": 429,
    "message": "Rate limit exceeded",
    "data": {
      "retry_after": 30
    }
  }
}
```

## Common Use Cases

### 1. Building a Research Assistant
```python
# Search for content
search_result = mcp_request("tools/call", {
    "name": "media.search_media",
    "arguments": {"query": "quantum computing", "limit": 10}
})

# Get detailed transcripts
for item in search_result['results']:
    transcript = mcp_request("tools/call", {
        "name": "media.get_transcript",
        "arguments": {"media_id": item['id']}
    })
    
# Create a summary note
note = mcp_request("tools/call", {
    "name": "notes.create_note",
    "arguments": {
        "title": "Quantum Computing Research",
        "content": summary_content,
        "tags": ["quantum", "research"]
    }
})
```

### 2. Content Processing Pipeline
```python
# Ingest new media
media = mcp_request("tools/call", {
    "name": "media.ingest_media",
    "arguments": {
        "url": "https://youtube.com/watch?v=...",
        "process_type": "transcribe"
    }
})

# Generate embeddings
embeddings = mcp_request("tools/call", {
    "name": "rag.create_embedding",
    "arguments": {
        "text": transcript_text,
        "collection": "research"
    }
})

# Create searchable context
context = mcp_request("tools/call", {
    "name": "rag.contextual_retrieval",
    "arguments": {
        "query": "key concepts",
        "media_ids": [media['id']]
    }
})
```

### 3. Interactive Chat with Context
```python
# Search for relevant context
context = mcp_request("tools/call", {
    "name": "rag.hybrid_search",
    "arguments": {
        "query": user_question,
        "top_k": 5
    }
})

# Create conversation with context
conversation = mcp_request("tools/call", {
    "name": "chat.create_conversation",
    "arguments": {
        "title": "Research Discussion",
        "context": context['results']
    }
})

# Generate response
response = mcp_request("tools/call", {
    "name": "chat.chat_completion",
    "arguments": {
        "conversation_id": conversation['id'],
        "messages": [
            {"role": "system", "content": "You are a research assistant."},
            {"role": "user", "content": user_question}
        ],
        "context": context['results']
    }
})
```

## Troubleshooting

### Common Issues

#### 1. Connection Refused
- **Check**: Is tldw_server running?
- **Solution**: Start the server with `python -m uvicorn tldw_Server_API.app.main:app`

#### 2. Authentication Failed
- **Check**: Is your token valid and not expired?
- **Solution**: Refresh your token or obtain a new one

#### 3. Method Not Found
- **Check**: Is the module registered and healthy?
- **Solution**: Check `/status` endpoint for module health

#### 4. Rate Limit Exceeded
- **Check**: Are you making too many requests?
- **Solution**: Implement exponential backoff or request batching

#### 5. WebSocket Disconnects
- **Check**: Network stability and timeout settings
- **Solution**: Implement reconnection logic with backoff

### Debug Mode

Enable debug logging for detailed information:

```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

Or set environment variable:
```bash
export MCP_DEBUG=true
```

### Getting Help

1. **Check Status**: `/api/v1/mcp/v2/status`
2. **List Tools**: `/api/v1/mcp/v2/tools`
3. **Module Health**: `/api/v1/mcp/v2/modules/{module}/health`
4. **API Documentation**: `/docs` (FastAPI automatic docs)
5. **GitHub Issues**: Report bugs or request features

## Best Practices

1. **Use Batch Operations**: Combine multiple operations when possible
2. **Cache Results**: Store frequently accessed data locally
3. **Handle Errors Gracefully**: Implement retry logic with backoff
4. **Monitor Rate Limits**: Track usage to avoid hitting limits
5. **Use WebSockets for Streaming**: Better performance for real-time operations
6. **Implement Timeouts**: Prevent hanging requests
7. **Validate Input**: Check parameters before sending requests
8. **Log Interactions**: Keep audit trail for debugging

## Security Considerations

1. **Use HTTPS in Production**: Encrypt data in transit
2. **Secure Token Storage**: Never expose tokens in code or logs
3. **Implement Token Rotation**: Regularly refresh tokens
4. **Validate Permissions**: Check user has required access
5. **Sanitize Input**: Prevent injection attacks
6. **Monitor Usage**: Track unusual patterns
7. **Rate Limit Clients**: Prevent abuse
8. **Audit Logs**: Maintain security audit trail

---

For more information, see the [Developer Documentation](./MCP_v2_Developer_Guide.md) or visit the [API Reference](http://localhost:8001/docs).