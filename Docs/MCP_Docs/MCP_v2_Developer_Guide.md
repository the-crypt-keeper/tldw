# MCP v2 Developer Documentation

## Table of Contents
1. [Architecture Overview](#architecture-overview)
2. [Core Components](#core-components)
3. [Module Development](#module-development)
4. [API Reference](#api-reference)
5. [Protocol Specification](#protocol-specification)
6. [Authentication & Security](#authentication--security)
7. [Database Schema](#database-schema)
8. [Testing](#testing)
9. [Deployment](#deployment)
10. [Contributing](#contributing)

## Architecture Overview

### System Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                         Clients                              │
│  (AI Assistants, Web Apps, CLI Tools, IDE Extensions)       │
└─────────────┬───────────────────────┬───────────────────────┘
              │                       │
              ▼                       ▼
        ┌──────────┐           ┌──────────┐
        │   HTTP   │           │WebSocket │
        │ Endpoint │           │ Endpoint │
        └─────┬────┘           └────┬─────┘
              │                      │
              ▼                      ▼
┌──────────────────────────────────────────────────────────────┐
│                    MCP v2 Protocol Layer                      │
│                     (JSON-RPC 2.0)                            │
├────────────────────────────────────────────────────────────────┤
│   ┌────────────┐  ┌─────────────┐  ┌──────────────┐         │
│   │   Auth     │  │Rate Limiter │  │     RBAC     │         │
│   │   (JWT)    │  │             │  │   Policies   │         │
│   └────────────┘  └─────────────┘  └──────────────┘         │
├────────────────────────────────────────────────────────────────┤
│                     Module Registry                           │
│  ┌──────────────────────────────────────────────────┐       │
│  │ Registered Modules:                               │       │
│  │  • Media Module    • RAG Module                  │       │
│  │  • Notes Module    • Prompts Module              │       │
│  │  • Transcription   • Chat Module                 │       │
│  └──────────────────────────────────────────────────┘       │
└────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌────────────────────────────────────────────────────────────────┐
│                    Core Services Layer                        │
│  ┌─────────────┐  ┌──────────────┐  ┌─────────────┐        │
│  │  Database   │  │  Embeddings  │  │     LLM     │        │
│  │  Management │  │   Service    │  │  Providers  │        │
│  └─────────────┘  └──────────────┘  └─────────────┘        │
└────────────────────────────────────────────────────────────────┘
```

### Directory Structure

```
tldw_Server_API/app/
├── core/
│   └── MCP_v2/
│       ├── __init__.py           # Server initialization
│       ├── schemas.py            # Pydantic models
│       ├── core/
│       │   ├── protocol.py       # Protocol implementation
│       │   ├── registry.py       # Module registry
│       │   └── server.py         # Main server class
│       ├── modules/
│       │   ├── base.py           # Base module interface
│       │   ├── media_module.py   # Media operations
│       │   ├── rag_module.py     # RAG operations
│       │   ├── notes_module.py   # Notes management
│       │   ├── prompts_module.py # Prompt templates
│       │   ├── transcription_module.py # Transcription
│       │   └── chat_module.py    # Chat completions
│       └── auth/
│           ├── jwt_auth.py       # JWT authentication
│           ├── rate_limiter.py   # Rate limiting
│           └── rbac.py           # Role-based access
└── api/v1/endpoints/
    └── mcp_v2_endpoint.py        # FastAPI routes
```

## Core Components

### 1. MCP Server (`core/server.py`)

The main server orchestrates all MCP operations:

```python
class MCPServer:
    def __init__(self):
        self.registry = ModuleRegistry()
        self.protocol = MCPProtocol(self.registry)
        self.auth_manager = AuthManager()
        self.rate_limiter = RateLimiter()
        
    async def initialize(self):
        """Initialize all modules"""
        await self._register_modules()
        await self._initialize_modules()
        
    async def handle_request(self, request: MCPRequest) -> MCPResponse:
        """Process MCP requests"""
        # Validate request
        # Check authentication
        # Apply rate limiting
        # Route to protocol handler
        # Return response
```

### 2. Module Registry (`core/registry.py`)

Manages module lifecycle and routing:

```python
class ModuleRegistry:
    def __init__(self):
        self.modules: Dict[str, BaseModule] = {}
        self.registrations: Dict[str, ModuleRegistration] = {}
        
    async def register_module(
        self,
        module_id: str,
        module: BaseModule,
        registration: ModuleRegistration
    ):
        """Register a new module"""
        
    async def find_module_for_tool(self, tool_name: str) -> Optional[BaseModule]:
        """Find module that provides a specific tool"""
        
    async def get_all_tools(self) -> List[Dict[str, Any]]:
        """Aggregate tools from all modules"""
```

### 3. Protocol Handler (`core/protocol.py`)

Implements MCP protocol methods:

```python
class MCPProtocol:
    async def process_request(self, request: MCPRequest) -> MCPResponse:
        """Route requests to appropriate handlers"""
        
        handlers = {
            "initialize": self.handle_initialize,
            "tools/list": self.handle_tools_list,
            "tools/call": self.handle_tool_call,
            "resources/list": self.handle_resources_list,
            "resources/read": self.handle_resource_read,
            "prompts/list": self.handle_prompts_list,
            "prompts/get": self.handle_prompt_get,
        }
        
        handler = handlers.get(request.method)
        if handler:
            return await handler(request.params)
        else:
            return error_response("Method not found", -32601)
```

## Module Development

### Creating a New Module

1. **Extend BaseModule**:

```python
from typing import Dict, Any, List
from loguru import logger
from ..modules.base import BaseModule, create_tool_definition
from ..schemas import ModuleConfig

class CustomModule(BaseModule):
    """Custom module implementation"""
    
    def __init__(self, config: ModuleConfig):
        super().__init__(config)
        # Initialize module-specific resources
        
    async def on_initialize(self) -> None:
        """Called when module is initialized"""
        logger.info(f"Initializing {self.name} module")
        # Setup connections, load resources, etc.
        
    async def on_shutdown(self) -> None:
        """Called when module is shutting down"""
        # Cleanup resources
        
    async def check_health(self) -> bool:
        """Health check for monitoring"""
        return True
        
    async def get_tools(self) -> List[Dict[str, Any]]:
        """Define available tools"""
        return [
            create_tool_definition(
                name="my_tool",
                description="Tool description",
                parameters={
                    "properties": {
                        "param1": {
                            "type": "string",
                            "description": "Parameter description"
                        }
                    },
                    "required": ["param1"]
                },
                department=self.name
            )
        ]
        
    async def execute_tool(self, tool_name: str, arguments: Dict[str, Any]) -> Any:
        """Execute a tool"""
        if tool_name == "my_tool":
            return await self._my_tool_implementation(arguments)
        raise ValueError(f"Unknown tool: {tool_name}")
        
    async def get_resources(self) -> List[Dict[str, Any]]:
        """Define available resources"""
        return []
        
    async def read_resource(self, uri: str) -> Dict[str, Any]:
        """Read a resource"""
        raise ValueError(f"Unknown resource: {uri}")
```

2. **Register the Module**:

```python
# In server.py
async def _register_modules(self):
    # ... existing modules ...
    
    # Register custom module
    custom_module = CustomModule(ModuleConfig(
        name="custom",
        version="1.0.0",
        settings={}
    ))
    
    await self.registry.register_module(
        module_id="custom",
        module=custom_module,
        registration=ModuleRegistration(
            id="custom",
            name="Custom Module",
            version="1.0.0",
            capabilities=["tools", "resources"]
        )
    )
```

### Module Configuration

Modules can be configured via `ModuleConfig`:

```python
config = ModuleConfig(
    name="module_name",
    version="1.0.0",
    settings={
        "database_path": "./databases/module.db",
        "api_key": "optional_api_key",
        "max_connections": 10,
        "timeout": 30
    }
)
```

### Tool Definition Format

```python
tool = {
    "name": "module.tool_name",
    "description": "Clear description of what the tool does",
    "inputSchema": {
        "type": "object",
        "properties": {
            "param1": {
                "type": "string",
                "description": "Parameter description",
                "enum": ["option1", "option2"]  # Optional
            },
            "param2": {
                "type": "integer",
                "description": "Another parameter",
                "default": 10
            }
        },
        "required": ["param1"]
    }
}
```

## API Reference

### HTTP Endpoints

#### POST `/api/v1/mcp/v2/request`
Process MCP request via HTTP.

**Request Body:**
```json
{
  "jsonrpc": "2.0",
  "method": "string",
  "params": {},
  "id": "string|number"
}
```

**Response:**
```json
{
  "jsonrpc": "2.0",
  "result": {},
  "id": "string|number"
}
```

#### GET `/api/v1/mcp/v2/status`
Get server status and module health.

**Response:**
```json
{
  "status": "healthy",
  "version": "2.0.0",
  "uptime_seconds": 1234.5,
  "modules": {
    "total": 6,
    "active": 6,
    "registrations": [...]
  },
  "health": {...}
}
```

#### GET `/api/v1/mcp/v2/tools`
List all available tools.

**Response:**
```json
{
  "tools": [
    {
      "name": "module.tool_name",
      "description": "Tool description",
      "inputSchema": {...}
    }
  ],
  "count": 42
}
```

#### GET `/api/v1/mcp/v2/modules/{module_id}/health`
Check specific module health.

**Response:**
```json
{
  "module": "module_id",
  "status": "healthy",
  "last_check": "2025-01-01T00:00:00Z",
  "version": "1.0.0"
}
```

### WebSocket Endpoint

#### WS `/api/v1/mcp/v2/ws`
WebSocket connection for real-time MCP communication.

**Connection:**
```javascript
ws = new WebSocket('ws://localhost:8001/api/v1/mcp/v2/ws?client_id=my-client')
```

**Message Format:**
Same as HTTP request/response format but over WebSocket frames.

## Protocol Specification

### JSON-RPC 2.0 Methods

#### Core Methods

1. **initialize**
   - Initialize client session
   - Parameters: `clientInfo`, `capabilities`

2. **ping**
   - Health check
   - Parameters: none

#### Tool Methods

3. **tools/list**
   - List available tools
   - Parameters: `filter` (optional)

4. **tools/call**
   - Execute a tool
   - Parameters: `name`, `arguments`

#### Resource Methods

5. **resources/list**
   - List available resources
   - Parameters: none

6. **resources/read**
   - Read resource content
   - Parameters: `uri`

7. **resources/subscribe**
   - Subscribe to resource changes (WebSocket only)
   - Parameters: `uri`

#### Prompt Methods

8. **prompts/list**
   - List available prompts
   - Parameters: none

9. **prompts/get**
   - Get prompt with arguments
   - Parameters: `name`, `arguments`

### Error Codes

| Code | Message | Description |
|------|---------|-------------|
| -32700 | Parse error | Invalid JSON |
| -32600 | Invalid request | Not valid JSON-RPC |
| -32601 | Method not found | Method doesn't exist |
| -32602 | Invalid params | Invalid parameters |
| -32603 | Internal error | Server error |
| -32000 | Server error | Generic server error |
| -32001 | Resource not found | Resource doesn't exist |
| -32002 | Resource error | Error reading resource |
| -32003 | Tool not found | Tool doesn't exist |
| -32004 | Tool execution error | Tool failed |
| -32005 | Authentication required | Missing auth |
| -32006 | Permission denied | Insufficient permissions |
| -32007 | Rate limit exceeded | Too many requests |

## Authentication & Security

### JWT Authentication

#### Token Structure
```json
{
  "sub": "user_id",
  "username": "username",
  "roles": ["user", "admin"],
  "department": "engineering",
  "permissions": ["tools:execute", "resources:read"],
  "exp": 1234567890,
  "type": "access"
}
```

#### Configuration
```python
# In jwt_auth.py
JWT_CONFIG = {
    "SECRET_KEY": os.getenv("JWT_SECRET", "change-in-production"),
    "ALGORITHM": "HS256",
    "ACCESS_TOKEN_EXPIRE_MINUTES": 60,
    "REFRESH_TOKEN_EXPIRE_DAYS": 7
}
```

### Rate Limiting

#### Algorithms

1. **Token Bucket**
   - Capacity: 100 tokens
   - Refill rate: 100/minute
   - Burst allowance: 10

2. **Sliding Window**
   - Window: 60 seconds
   - Max requests: 100

#### Configuration
```python
RATE_LIMIT_CONFIG = {
    "default": {
        "rate": 100,
        "per": 60,  # seconds
        "burst": 10
    },
    "authenticated": {
        "rate": 1000,
        "per": 60,
        "burst": 50
    }
}
```

### RBAC Policies

#### Role Definitions
```python
ROLES = {
    "admin": {
        "permissions": ["*"],  # All permissions
        "departments": ["*"]   # All departments
    },
    "user": {
        "permissions": [
            "tools:execute",
            "resources:read",
            "prompts:read"
        ],
        "departments": ["general"]
    },
    "viewer": {
        "permissions": ["resources:read"],
        "departments": ["general"]
    }
}
```

#### Permission Check
```python
def check_permission(user: MCPUser, action: str, resource: str) -> bool:
    """Check if user has permission for action on resource"""
    # Check role permissions
    # Check department access
    # Check resource-specific rules
    return authorized
```

## Database Schema

### Module Databases

Each module may have its own database. Common patterns:

#### Media Database (SQLite)
```sql
-- Media items
CREATE TABLE media (
    id INTEGER PRIMARY KEY,
    title TEXT NOT NULL,
    url TEXT,
    content TEXT,
    transcript TEXT,
    metadata JSON,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP
);

-- Full-text search
CREATE VIRTUAL TABLE media_fts USING fts5(
    title, content, transcript,
    content='media'
);
```

#### Notes Database (SQLite)
```sql
-- Notes
CREATE TABLE notes (
    id TEXT PRIMARY KEY,
    title TEXT NOT NULL,
    content TEXT,
    tags JSON,
    version INTEGER DEFAULT 1,
    deleted BOOLEAN DEFAULT 0,
    created_at TIMESTAMP,
    updated_at TIMESTAMP
);

-- Note search
CREATE VIRTUAL TABLE notes_fts USING fts5(
    title, content,
    content='notes'
);
```

#### Vector Database (ChromaDB)
```python
# Collection schema
collection = {
    "name": "tldw_media",
    "metadata": {
        "description": "Media embeddings",
        "embedding_model": "all-MiniLM-L6-v2"
    },
    "embedding_function": embedding_function
}
```

## Testing

### Unit Tests

```python
# tests/test_mcp_protocol.py
import pytest
from tldw_Server_API.app.core.MCP_v2.core.protocol import MCPProtocol

@pytest.fixture
def protocol():
    return MCPProtocol(mock_registry)

async def test_tools_list(protocol):
    request = MCPRequest(
        jsonrpc="2.0",
        method="tools/list",
        params={},
        id="test-1"
    )
    response = await protocol.process_request(request)
    assert response.result is not None
    assert "tools" in response.result
```

### Integration Tests

```python
# tests/test_mcp_integration.py
import httpx
import pytest

@pytest.mark.asyncio
async def test_full_workflow():
    async with httpx.AsyncClient() as client:
        # Initialize
        response = await client.post(
            "http://localhost:8001/api/v1/mcp/v2/request",
            json={
                "jsonrpc": "2.0",
                "method": "initialize",
                "params": {"clientInfo": {"name": "test"}},
                "id": "1"
            }
        )
        assert response.status_code == 200
        
        # Call tool
        response = await client.post(
            "http://localhost:8001/api/v1/mcp/v2/request",
            json={
                "jsonrpc": "2.0",
                "method": "tools/call",
                "params": {
                    "name": "media.search_media",
                    "arguments": {"query": "test"}
                },
                "id": "2"
            }
        )
        assert response.status_code == 200
```

### Load Testing

```python
# tests/test_load.py
import asyncio
import httpx
import time

async def load_test(num_requests=100):
    async with httpx.AsyncClient() as client:
        start = time.time()
        
        tasks = []
        for i in range(num_requests):
            task = client.post(
                "http://localhost:8001/api/v1/mcp/v2/request",
                json={
                    "jsonrpc": "2.0",
                    "method": "tools/list",
                    "id": f"load-{i}"
                }
            )
            tasks.append(task)
        
        responses = await asyncio.gather(*tasks)
        elapsed = time.time() - start
        
        success = sum(1 for r in responses if r.status_code == 200)
        print(f"Completed {success}/{num_requests} in {elapsed:.2f}s")
        print(f"RPS: {num_requests/elapsed:.2f}")

asyncio.run(load_test())
```

## Deployment

### Docker Deployment

```dockerfile
# Dockerfile
FROM python:3.11-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install -r requirements.txt

# Copy application
COPY tldw_Server_API ./tldw_Server_API

# Environment
ENV PYTHONPATH=/app
ENV MCP_HOST=0.0.0.0
ENV MCP_PORT=8001

# Health check
HEALTHCHECK --interval=30s --timeout=3s --start-period=5s --retries=3 \
  CMD curl -f http://localhost:8001/api/v1/mcp/v2/status || exit 1

# Run
CMD ["uvicorn", "tldw_Server_API.app.main:app", "--host", "0.0.0.0", "--port", "8001"]
```

### Docker Compose

```yaml
# docker-compose.yml
version: '3.8'

services:
  mcp-server:
    build: .
    ports:
      - "8001:8001"
    environment:
      - JWT_SECRET=${JWT_SECRET}
      - DATABASE_PATH=/data/databases
      - REDIS_URL=redis://redis:6379
    volumes:
      - ./data:/data
      - ./config:/app/config
    depends_on:
      - redis
    restart: unless-stopped

  redis:
    image: redis:alpine
    ports:
      - "6379:6379"
    volumes:
      - redis-data:/data

volumes:
  redis-data:
```

### Production Configuration

```python
# config/production.py
import os

class ProductionConfig:
    # Server
    HOST = "0.0.0.0"
    PORT = 8001
    WORKERS = os.cpu_count() * 2 + 1
    
    # Security
    JWT_SECRET = os.environ["JWT_SECRET"]
    CORS_ORIGINS = ["https://app.example.com"]
    HTTPS_ONLY = True
    
    # Database
    DATABASE_PATH = "/data/databases"
    DATABASE_POOL_SIZE = 20
    
    # Rate Limiting
    RATE_LIMIT_ENABLED = True
    RATE_LIMIT_STORAGE = "redis"
    
    # Monitoring
    METRICS_ENABLED = True
    LOGGING_LEVEL = "INFO"
    SENTRY_DSN = os.environ.get("SENTRY_DSN")
```

### Nginx Configuration

```nginx
# /etc/nginx/sites-available/mcp
server {
    listen 443 ssl http2;
    server_name mcp.example.com;

    ssl_certificate /etc/ssl/certs/cert.pem;
    ssl_certificate_key /etc/ssl/private/key.pem;

    # WebSocket support
    location /api/v1/mcp/v2/ws {
        proxy_pass http://localhost:8001;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        
        # Timeouts
        proxy_connect_timeout 7d;
        proxy_send_timeout 7d;
        proxy_read_timeout 7d;
    }

    # HTTP endpoints
    location /api/v1/mcp/v2/ {
        proxy_pass http://localhost:8001;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        
        # Buffering
        proxy_buffering off;
        
        # Timeouts
        proxy_connect_timeout 60s;
        proxy_send_timeout 60s;
        proxy_read_timeout 60s;
    }
}
```

## Contributing

### Development Setup

1. **Clone Repository**:
```bash
git clone https://github.com/yourusername/tldw_server.git
cd tldw_server
```

2. **Create Virtual Environment**:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install Dependencies**:
```bash
pip install -r requirements.txt
pip install -r requirements-dev.txt
```

4. **Run Tests**:
```bash
pytest tests/
```

5. **Start Development Server**:
```bash
uvicorn tldw_Server_API.app.main:app --reload --port 8001
```

### Code Style

- Follow PEP 8
- Use type hints
- Add docstrings to all functions/classes
- Run formatters:
```bash
black tldw_Server_API/
isort tldw_Server_API/
flake8 tldw_Server_API/
```

### Pull Request Process

1. Fork the repository
2. Create feature branch: `git checkout -b feature/amazing-feature`
3. Make changes and test
4. Commit: `git commit -m 'Add amazing feature'`
5. Push: `git push origin feature/amazing-feature`
6. Open Pull Request

### Module Contribution Guidelines

When contributing a new module:

1. **Follow the Module Template**
2. **Include Tests**: Unit and integration tests
3. **Document Tools**: Clear descriptions and examples
4. **Handle Errors**: Graceful error handling
5. **Add Examples**: Usage examples in documentation
6. **Performance**: Consider rate limits and resource usage
7. **Security**: Validate inputs, sanitize outputs

---

For questions or support, please open an issue on GitHub or contact the maintainers.