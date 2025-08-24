# MCP v2 for tldw - Model Context Protocol Server

## Overview

tldw now includes a powerful Model Context Protocol (MCP) server that enables AI assistants and other clients to interact with your media library through a standardized, modular interface. This enterprise-grade implementation provides secure, scalable access to tldw's features.

## What is MCP?

The Model Context Protocol is a standard that allows AI assistants (like Claude, ChatGPT, or custom agents) to:
- Execute tools and actions in external systems
- Access resources and data
- Use predefined prompts
- Maintain context across interactions

## Architecture

```
┌─────────────────┐
│   AI Assistant  │
│  (Claude, etc)  │
└────────┬────────┘
         │ MCP Protocol
         ▼
┌─────────────────┐
│   MCP Server    │
│   (tldw v2)     │
└────────┬────────┘
         │
    ┌────┴────┬──────────┬──────────┐
    ▼         ▼          ▼          ▼
┌────────┐┌────────┐┌────────┐┌────────┐
│ Media  ││  RAG   ││  Chat  ││ Notes  │
│ Module ││ Module ││ Module ││ Module │
└────────┘└────────┘└────────┘└────────┘
```

## Available Modules

### 1. Media Module
Handles all media-related operations:
- **Tools**:
  - `media.search_media` - Search content with keywords or semantic search
  - `media.get_transcript` - Retrieve media transcripts
  - `media.get_media_metadata` - Get media information
  - `media.ingest_media` - Add new media from URLs
  - `media.get_media_summary` - Get or generate summaries
  - `media.list_recent_media` - List recently added content

### 2. RAG Module  
Provides advanced search and retrieval:
- **Tools**:
  - `rag.vector_search` - Semantic similarity search
  - `rag.hybrid_search` - Combined BM25 + vector search
  - `rag.get_context` - Retrieve relevant context for queries
  - `rag.rerank_results` - Re-rank search results
  - `rag.generate_embedding` - Generate text embeddings
  - `rag.index_content` - Index new content for search

### 3. Coming Soon
- **Chat Module** - Chat completions and conversation management
- **Notes Module** - Note-taking and knowledge management
- **Prompts Module** - Prompt library and templates
- **Transcription Module** - Advanced transcription services

## API Endpoints

### WebSocket Endpoint (Recommended)
```
ws://localhost:8000/mcp/v2/ws?client_id=your_client_id
```

Full bidirectional MCP protocol support for real-time communication.

### HTTP Endpoints
```
POST /mcp/v2/request       - Send MCP requests via HTTP
GET  /mcp/v2/status        - Server and module status
GET  /mcp/v2/tools         - List all available tools
GET  /mcp/v2/modules       - List registered modules
GET  /mcp/v2/modules/{id}/health - Module health check
```

## Quick Start

### 1. Start the Server
The MCP server starts automatically with tldw:
```bash
python -m uvicorn tldw_Server_API.app.main:app --reload
```

### 2. Test the Server
```bash
# Check status
curl http://localhost:8000/mcp/v2/status

# List available tools
curl http://localhost:8000/mcp/v2/tools

# Test echo
curl -X POST http://localhost:8000/mcp/v2/test/echo \
  -H "Content-Type: application/json" \
  -d '{"message": "Hello MCP!"}'
```

### 3. Connect with WebSocket (Python Example)
```python
import websocket
import json

ws = websocket.WebSocket()
ws.connect("ws://localhost:8000/mcp/v2/ws?client_id=python_client")

# Initialize connection
ws.send(json.dumps({
    "jsonrpc": "2.0",
    "method": "initialize",
    "params": {
        "clientInfo": {
            "name": "Python Client",
            "version": "1.0"
        }
    },
    "id": "1"
}))

response = json.loads(ws.recv())
print("Initialized:", response)

# Search media
ws.send(json.dumps({
    "jsonrpc": "2.0",
    "method": "tools/call",
    "params": {
        "name": "media.search_media",
        "arguments": {
            "query": "machine learning",
            "limit": 5
        }
    },
    "id": "2"
}))

results = json.loads(ws.recv())
print("Search results:", results)
```

### 4. HTTP Request Example
```python
import requests

# Send MCP request via HTTP
response = requests.post(
    "http://localhost:8000/mcp/v2/request",
    json={
        "jsonrpc": "2.0",
        "method": "tools/list",
        "params": {},
        "id": "1"
    },
    params={"client_id": "http_client"}
)

print(response.json())
```

## MCP Protocol Format

All requests follow JSON-RPC 2.0 format:

### Request Structure
```json
{
    "jsonrpc": "2.0",
    "method": "method_name",
    "params": {
        // Method parameters
    },
    "id": "unique_request_id"
}
```

### Response Structure
```json
{
    "jsonrpc": "2.0",
    "result": {
        // Method result
    },
    "id": "unique_request_id"
}
```

### Error Response
```json
{
    "jsonrpc": "2.0",
    "error": {
        "code": -32600,
        "message": "Error description"
    },
    "id": "unique_request_id"
}
```

## Available Methods

### Core Methods
- `initialize` - Initialize connection
- `tools/list` - List available tools
- `tools/call` - Execute a tool
- `resources/list` - List available resources
- `resources/read` - Read a resource
- `prompts/list` - List available prompts
- `prompts/get` - Get a specific prompt

### Tool Execution
```json
{
    "method": "tools/call",
    "params": {
        "name": "media.search_media",
        "arguments": {
            "query": "your search",
            "search_type": "semantic",
            "limit": 10
        }
    }
}
```

## Use Cases

### 1. AI Assistant Integration
Connect Claude, ChatGPT, or other AI assistants to your media library:
- Search and retrieve content
- Generate summaries
- Answer questions using your data
- Manage transcriptions

### 2. Automation
Build automated workflows:
- Scheduled media ingestion
- Automatic summarization
- Content categorization
- Alert on specific content

### 3. Custom Applications
Build applications that leverage tldw's capabilities:
- Search interfaces
- Content browsers
- Analytics dashboards
- Mobile apps

## Configuration

The MCP server can be configured through module settings:

```python
# In server.py
media_config = ModuleConfig(
    name="media",
    description="Media module",
    version="1.0.0",
    department="media",
    settings={
        "db_path": "./Databases/Media_DB_v2.db",
        "max_results": 100
    }
)
```

## Security

### Authentication
- Client ID required for connections
- JWT support for advanced authentication (coming soon)

### Permissions
- Tool-level access control
- Module-based permissions
- Rate limiting per client

### Audit
- All operations logged
- Request tracking with IDs
- Error logging for debugging

## Extending with New Modules

### Creating a Custom Module

1. Create a new module file:
```python
from tldw_Server_API.app.core.MCP_v2.modules.base import BaseModule

class CustomModule(BaseModule):
    async def get_tools(self):
        return [
            {
                "name": "my_tool",
                "description": "My custom tool",
                "inputSchema": {...}
            }
        ]
    
    async def execute_tool(self, tool_name, arguments):
        if tool_name == "my_tool":
            # Tool implementation
            return result
```

2. Register in server initialization:
```python
await register_module("custom", CustomModule, config)
```

## Performance

- Async/await throughout for high concurrency
- Module lazy loading
- Connection pooling for databases
- Efficient JSON serialization
- WebSocket for low-latency communication

## Troubleshooting

### Connection Issues
- Ensure tldw server is running
- Check firewall settings
- Verify client_id is provided

### Tool Not Found
- Tools are namespaced: use `module.tool_name`
- Check module is registered and healthy
- Verify tool name spelling

### Performance Issues
- Check module health: `/mcp/v2/modules/{id}/health`
- Review server logs for errors
- Consider rate limiting settings

## Examples

### Search and Summarize
```python
# Search for content
search_result = mcp_call("tools/call", {
    "name": "media.search_media",
    "arguments": {"query": "climate change"}
})

# Get summary of first result
media_id = search_result["results"][0]["id"]
summary = mcp_call("tools/call", {
    "name": "media.get_media_summary",
    "arguments": {"media_id": media_id}
})
```

### Ingest and Process
```python
# Ingest new media
ingest_result = mcp_call("tools/call", {
    "name": "media.ingest_media",
    "arguments": {
        "url": "https://youtube.com/watch?v=...",
        "process_type": "both"
    }
})

# Get transcript
transcript = mcp_call("tools/call", {
    "name": "media.get_transcript",
    "arguments": {"media_id": ingest_result["media_id"]}
})
```

## Support

- Report issues with the `mcp-v2` tag
- Check logs in `server.log`
- Module-specific logs available
- Debug mode available with `--debug` flag

## Roadmap

- [ ] Additional modules (Chat, Notes, Prompts)
- [ ] Enhanced authentication (OAuth, API keys)
- [ ] WebRTC support for real-time features
- [ ] Module marketplace
- [ ] Client SDKs (Python, JS, Go)
- [ ] Prometheus metrics export
- [ ] Admin dashboard UI

## License

Part of tldw project - see main LICENSE file.