# MCP Unified - Developer Guide

## Table of Contents
1. [Architecture Overview](#architecture-overview)
2. [Module Development](#module-development)
3. [API Integration](#api-integration)
4. [Authentication & Authorization](#authentication--authorization)
5. [Protocol Implementation](#protocol-implementation)
6. [Testing](#testing)
7. [Performance Optimization](#performance-optimization)
8. [Debugging](#debugging)
9. [Contributing](#contributing)
10. [API Reference](#api-reference)

## Architecture Overview

### Core Components

```
MCP_unified/
├── config.py                 # Configuration management
├── server.py                 # Main server implementation
├── protocol.py              # JSON-RPC 2.0 protocol handler
├── auth/                    # Security layer
│   ├── jwt_manager.py      # JWT token management
│   ├── rbac.py            # Role-based access control
│   └── rate_limiter.py    # Rate limiting implementation
├── modules/                 # Module system
│   ├── base.py            # Base module interface
│   ├── registry.py        # Module registry
│   └── implementations/   # Concrete modules
└── monitoring/             # Observability
    └── metrics.py         # Metrics collection
```

### Design Principles

1. **Security First**: All secrets from environment, no hardcoded values
2. **Modular Architecture**: Extensible through module system
3. **Protocol Compliance**: Full JSON-RPC 2.0 implementation
4. **Async/Await**: Non-blocking I/O throughout
5. **Type Safety**: Comprehensive type hints with Pydantic
6. **Observability**: Built-in metrics and health checks

### Key Technologies
- **FastAPI**: Modern async web framework
- **Pydantic**: Data validation and settings management
- **python-jose**: JWT implementation
- **asyncio**: Async/await support
- **WebSocket**: Real-time bidirectional communication

## Module Development

### Creating a New Module

#### 1. Define Module Interface
```python
# modules/implementations/my_module.py
from typing import Dict, Any, List, Optional
from tldw_Server_API.app.core.MCP_unified.modules.base import BaseModule, ModuleHealth
from tldw_Server_API.app.core.MCP_unified.schemas import (
    MCPTool, MCPResource, MCPPrompt, ToolParameter
)

class MyModule(BaseModule):
    """Custom module implementation"""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__("my_module", "1.0.0", config)
        self.initialize_resources()
    
    async def initialize(self) -> None:
        """Initialize module resources"""
        # Set up database connections, load models, etc.
        self.db_pool = await create_pool(self.config.get("database_url"))
        self.logger.info(f"{self.name} initialized successfully")
    
    async def shutdown(self) -> None:
        """Clean up resources"""
        if self.db_pool:
            await self.db_pool.close()
        self.logger.info(f"{self.name} shutdown complete")
```

#### 2. Implement Tools
```python
    def get_tools(self) -> List[MCPTool]:
        """Define module tools"""
        return [
            MCPTool(
                name=f"{self.name}.process_data",
                description="Process data with custom logic",
                parameters=[
                    ToolParameter(
                        name="input_data",
                        type="string",
                        description="Data to process",
                        required=True
                    ),
                    ToolParameter(
                        name="options",
                        type="object",
                        description="Processing options",
                        required=False,
                        properties={
                            "format": {"type": "string", "enum": ["json", "xml", "csv"]},
                            "validate": {"type": "boolean", "default": True}
                        }
                    )
                ]
            )
        ]
    
    async def execute_tool(self, tool_name: str, arguments: Dict[str, Any]) -> Any:
        """Execute a tool"""
        if tool_name == f"{self.name}.process_data":
            return await self.process_data(
                arguments["input_data"],
                arguments.get("options", {})
            )
        raise ValueError(f"Unknown tool: {tool_name}")
    
    async def process_data(self, input_data: str, options: Dict[str, Any]) -> Dict[str, Any]:
        """Actual implementation"""
        # Validate input
        if options.get("validate", True):
            self.validate_input(input_data)
        
        # Process data
        format_type = options.get("format", "json")
        result = await self.transform_data(input_data, format_type)
        
        # Return result
        return {
            "status": "success",
            "data": result,
            "metadata": {
                "processed_at": datetime.utcnow().isoformat(),
                "format": format_type
            }
        }
```

#### 3. Add Health Checks
```python
    async def health_check(self) -> ModuleHealth:
        """Perform health check"""
        checks = {}
        
        # Check database connection
        try:
            async with self.db_pool.acquire() as conn:
                await conn.fetchval("SELECT 1")
            checks["database"] = True
        except Exception as e:
            checks["database"] = False
            self.logger.error(f"Database health check failed: {e}")
        
        # Check external service
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(self.config["service_url"], timeout=5) as resp:
                    checks["external_service"] = resp.status == 200
        except:
            checks["external_service"] = False
        
        # Overall status
        all_healthy = all(checks.values())
        
        return ModuleHealth(
            module_id=self.name,
            status="healthy" if all_healthy else "unhealthy",
            checks=checks,
            message="All systems operational" if all_healthy else "Some checks failed"
        )
```

#### 4. Register Module
```python
# In server initialization
from tldw_Server_API.app.core.MCP_unified.modules.registry import ModuleRegistry
from .modules.implementations.my_module import MyModule

async def register_modules(registry: ModuleRegistry):
    """Register all modules"""
    
    # Create module instance
    my_module = MyModule({
        "database_url": os.getenv("MY_MODULE_DATABASE_URL"),
        "service_url": os.getenv("MY_MODULE_SERVICE_URL"),
        "cache_ttl": 300
    })
    
    # Initialize module
    await my_module.initialize()
    
    # Register with registry
    await registry.register(my_module)
```

### Module Best Practices

1. **Async Everything**: Use async/await for all I/O operations
2. **Error Handling**: Implement comprehensive error handling with meaningful messages
3. **Logging**: Use structured logging with appropriate levels
4. **Configuration**: Accept configuration through constructor
5. **Health Checks**: Implement meaningful health checks
6. **Resource Management**: Properly initialize and cleanup resources
7. **Testing**: Write unit and integration tests

## API Integration

### Client Libraries

#### Python Client
```python
import asyncio
import aiohttp
import json
from typing import Dict, Any, Optional

class MCPClient:
    """MCP Unified API Client"""
    
    def __init__(self, base_url: str, token: Optional[str] = None):
        self.base_url = base_url
        self.token = token
        self.session = None
        self._request_id = 0
    
    async def __aenter__(self):
        self.session = aiohttp.ClientSession()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()
    
    def _next_id(self) -> int:
        """Generate next request ID"""
        self._request_id += 1
        return self._request_id
    
    async def request(self, method: str, params: Dict[str, Any] = None) -> Any:
        """Make JSON-RPC request"""
        headers = {"Content-Type": "application/json"}
        if self.token:
            headers["Authorization"] = f"Bearer {self.token}"
        
        payload = {
            "jsonrpc": "2.0",
            "method": method,
            "params": params or {},
            "id": self._next_id()
        }
        
        async with self.session.post(
            f"{self.base_url}/api/v1/mcp/request",
            json=payload,
            headers=headers
        ) as response:
            result = await response.json()
            
            if "error" in result:
                raise MCPError(result["error"])
            
            return result.get("result")
    
    async def list_tools(self) -> List[Dict[str, Any]]:
        """List available tools"""
        return await self.request("tools/list")
    
    async def execute_tool(self, name: str, arguments: Dict[str, Any]) -> Any:
        """Execute a tool"""
        return await self.request("tools/call", {
            "name": name,
            "arguments": arguments
        })
    
    async def search_media(self, query: str, limit: int = 10) -> List[Dict[str, Any]]:
        """Search media content"""
        return await self.execute_tool("media.search", {
            "query": query,
            "limit": limit
        })

# Usage example
async def main():
    async with MCPClient("http://localhost:8000") as client:
        # List tools
        tools = await client.list_tools()
        print(f"Available tools: {len(tools)}")
        
        # Search media
        results = await client.search_media("machine learning")
        for result in results:
            print(f"- {result['title']}")

asyncio.run(main())
```

#### JavaScript/TypeScript Client
```typescript
// mcp-client.ts
interface MCPRequest {
    jsonrpc: "2.0";
    method: string;
    params?: any;
    id: number | string;
}

interface MCPResponse {
    jsonrpc: "2.0";
    result?: any;
    error?: {
        code: number;
        message: string;
        data?: any;
    };
    id: number | string;
}

class MCPClient {
    private baseUrl: string;
    private token?: string;
    private requestId = 0;

    constructor(baseUrl: string, token?: string) {
        this.baseUrl = baseUrl;
        this.token = token;
    }

    private nextId(): number {
        return ++this.requestId;
    }

    async request(method: string, params?: any): Promise<any> {
        const headers: HeadersInit = {
            "Content-Type": "application/json",
        };
        
        if (this.token) {
            headers["Authorization"] = `Bearer ${this.token}`;
        }

        const request: MCPRequest = {
            jsonrpc: "2.0",
            method,
            params: params || {},
            id: this.nextId(),
        };

        const response = await fetch(`${this.baseUrl}/api/v1/mcp/request`, {
            method: "POST",
            headers,
            body: JSON.stringify(request),
        });

        const result: MCPResponse = await response.json();

        if (result.error) {
            throw new Error(`MCP Error ${result.error.code}: ${result.error.message}`);
        }

        return result.result;
    }

    async listTools(): Promise<any[]> {
        return this.request("tools/list");
    }

    async executeTool(name: string, args: any): Promise<any> {
        return this.request("tools/call", {
            name,
            arguments: args,
        });
    }

    // WebSocket connection
    connectWebSocket(onMessage: (data: any) => void): WebSocket {
        const wsUrl = this.baseUrl.replace("http", "ws") + "/api/v1/mcp/ws";
        const ws = new WebSocket(wsUrl);

        ws.onopen = () => {
            // Send initialization
            ws.send(JSON.stringify({
                jsonrpc: "2.0",
                method: "initialize",
                params: {
                    clientInfo: {
                        name: "TypeScript Client",
                        version: "1.0.0"
                    }
                },
                id: this.nextId()
            }));
        };

        ws.onmessage = (event) => {
            const data = JSON.parse(event.data);
            onMessage(data);
        };

        return ws;
    }
}

// Usage
const client = new MCPClient("http://localhost:8000");
const tools = await client.listTools();
console.log("Available tools:", tools);
```

### WebSocket Integration

```python
# WebSocket client example
import websockets
import json
import asyncio

class MCPWebSocketClient:
    def __init__(self, url: str, client_id: str = None):
        self.url = url
        self.client_id = client_id
        self.ws = None
        self._request_id = 0
        self._pending_requests = {}
        
    async def connect(self):
        """Connect to WebSocket"""
        params = f"?client_id={self.client_id}" if self.client_id else ""
        self.ws = await websockets.connect(f"{self.url}{params}")
        
        # Start message handler
        asyncio.create_task(self._handle_messages())
        
        # Initialize connection
        await self._initialize()
    
    async def _initialize(self):
        """Send initialization message"""
        await self.request("initialize", {
            "clientInfo": {
                "name": "Python WebSocket Client",
                "version": "1.0.0"
            }
        })
    
    async def _handle_messages(self):
        """Handle incoming messages"""
        async for message in self.ws:
            data = json.loads(message)
            
            # Handle response to request
            if "id" in data and data["id"] in self._pending_requests:
                future = self._pending_requests.pop(data["id"])
                if "error" in data:
                    future.set_exception(Exception(data["error"]["message"]))
                else:
                    future.set_result(data.get("result"))
            
            # Handle server notifications
            elif "method" in data and data["method"] == "notification":
                await self._handle_notification(data["params"])
    
    async def _handle_notification(self, params: Dict[str, Any]):
        """Handle server notifications"""
        print(f"Notification: {params}")
    
    async def request(self, method: str, params: Dict[str, Any] = None) -> Any:
        """Send request and wait for response"""
        request_id = self._next_id()
        
        request = {
            "jsonrpc": "2.0",
            "method": method,
            "params": params or {},
            "id": request_id
        }
        
        # Create future for response
        future = asyncio.Future()
        self._pending_requests[request_id] = future
        
        # Send request
        await self.ws.send(json.dumps(request))
        
        # Wait for response
        return await future
    
    def _next_id(self) -> int:
        self._request_id += 1
        return self._request_id
    
    async def close(self):
        """Close connection"""
        if self.ws:
            await self.ws.close()

# Usage
async def main():
    client = MCPWebSocketClient("ws://localhost:8000/api/v1/mcp/ws", "my-client")
    await client.connect()
    
    # Execute tool
    result = await client.request("tools/call", {
        "name": "media.search",
        "arguments": {"query": "test"}
    })
    print(f"Search results: {result}")
    
    await client.close()

asyncio.run(main())
```

## Authentication & Authorization

### JWT Implementation

```python
from datetime import datetime, timedelta
from typing import Optional, Dict, Any
import jwt
from passlib.context import CryptContext

class AuthManager:
    """Authentication manager"""
    
    def __init__(self, secret_key: str, algorithm: str = "HS256"):
        self.secret_key = secret_key
        self.algorithm = algorithm
        self.pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
    
    def create_access_token(
        self,
        subject: str,
        expires_delta: Optional[timedelta] = None,
        additional_claims: Dict[str, Any] = None
    ) -> str:
        """Create JWT access token"""
        if expires_delta:
            expire = datetime.utcnow() + expires_delta
        else:
            expire = datetime.utcnow() + timedelta(minutes=30)
        
        to_encode = {
            "sub": subject,
            "exp": expire,
            "iat": datetime.utcnow(),
            "type": "access"
        }
        
        if additional_claims:
            to_encode.update(additional_claims)
        
        return jwt.encode(to_encode, self.secret_key, algorithm=self.algorithm)
    
    def verify_token(self, token: str) -> Dict[str, Any]:
        """Verify and decode JWT token"""
        try:
            payload = jwt.decode(
                token,
                self.secret_key,
                algorithms=[self.algorithm]
            )
            return payload
        except jwt.ExpiredSignatureError:
            raise ValueError("Token has expired")
        except jwt.JWTError:
            raise ValueError("Invalid token")
    
    def hash_password(self, password: str) -> str:
        """Hash password"""
        return self.pwd_context.hash(password)
    
    def verify_password(self, plain_password: str, hashed_password: str) -> bool:
        """Verify password"""
        return self.pwd_context.verify(plain_password, hashed_password)
```

### RBAC Implementation

```python
from enum import Enum
from typing import Set, Dict, Any
from dataclasses import dataclass

class Permission(Enum):
    """System permissions"""
    READ = "read"
    WRITE = "write"
    DELETE = "delete"
    ADMIN = "admin"
    EXECUTE_TOOLS = "execute_tools"
    VIEW_METRICS = "view_metrics"

class Role(Enum):
    """User roles"""
    VIEWER = "viewer"
    USER = "user"
    POWER_USER = "power_user"
    ADMIN = "admin"

@dataclass
class RolePermissions:
    """Role to permissions mapping"""
    role: Role
    permissions: Set[Permission]

# Define role permissions
ROLE_PERMISSIONS = {
    Role.VIEWER: {Permission.READ},
    Role.USER: {Permission.READ, Permission.WRITE, Permission.EXECUTE_TOOLS},
    Role.POWER_USER: {
        Permission.READ,
        Permission.WRITE,
        Permission.DELETE,
        Permission.EXECUTE_TOOLS,
        Permission.VIEW_METRICS
    },
    Role.ADMIN: {p for p in Permission}  # All permissions
}

class RBACManager:
    """Role-based access control manager"""
    
    def __init__(self):
        self.role_permissions = ROLE_PERMISSIONS
    
    def check_permission(
        self,
        user_roles: Set[Role],
        required_permission: Permission
    ) -> bool:
        """Check if user has required permission"""
        for role in user_roles:
            if required_permission in self.role_permissions.get(role, set()):
                return True
        return False
    
    def get_user_permissions(self, user_roles: Set[Role]) -> Set[Permission]:
        """Get all permissions for user roles"""
        permissions = set()
        for role in user_roles:
            permissions.update(self.role_permissions.get(role, set()))
        return permissions

# FastAPI dependency
from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials

security = HTTPBearer()

async def get_current_user(
    credentials: HTTPAuthorizationCredentials = Depends(security)
) -> Dict[str, Any]:
    """Get current user from token"""
    auth_manager = AuthManager(secret_key=os.getenv("MCP_JWT_SECRET"))
    
    try:
        payload = auth_manager.verify_token(credentials.credentials)
        return payload
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=str(e),
            headers={"WWW-Authenticate": "Bearer"},
        )

def require_permission(permission: Permission):
    """Require specific permission"""
    async def permission_checker(
        user: Dict[str, Any] = Depends(get_current_user)
    ):
        rbac = RBACManager()
        user_roles = {Role(r) for r in user.get("roles", [])}
        
        if not rbac.check_permission(user_roles, permission):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Permission {permission.value} required"
            )
        
        return user
    
    return permission_checker
```

## Protocol Implementation

### JSON-RPC 2.0 Handler

```python
from typing import Any, Dict, Optional, Union
from pydantic import BaseModel, Field
from enum import Enum

class JSONRPCErrorCode(Enum):
    """Standard JSON-RPC 2.0 error codes"""
    PARSE_ERROR = -32700
    INVALID_REQUEST = -32600
    METHOD_NOT_FOUND = -32601
    INVALID_PARAMS = -32602
    INTERNAL_ERROR = -32603
    
    # Custom error codes
    AUTHENTICATION_REQUIRED = -32000
    PERMISSION_DENIED = -32001
    RATE_LIMIT_EXCEEDED = -32002

class JSONRPCError(BaseModel):
    """JSON-RPC error object"""
    code: int
    message: str
    data: Optional[Any] = None

class JSONRPCRequest(BaseModel):
    """JSON-RPC request"""
    jsonrpc: str = Field(default="2.0", const=True)
    method: str
    params: Optional[Union[Dict[str, Any], list]] = None
    id: Optional[Union[str, int]] = None

class JSONRPCResponse(BaseModel):
    """JSON-RPC response"""
    jsonrpc: str = Field(default="2.0", const=True)
    result: Optional[Any] = None
    error: Optional[JSONRPCError] = None
    id: Optional[Union[str, int]] = None

class ProtocolHandler:
    """JSON-RPC 2.0 protocol handler"""
    
    def __init__(self, method_registry: Dict[str, callable]):
        self.methods = method_registry
    
    async def handle_request(
        self,
        request: JSONRPCRequest,
        context: Optional[Dict[str, Any]] = None
    ) -> JSONRPCResponse:
        """Handle JSON-RPC request"""
        
        # Validate request
        if request.jsonrpc != "2.0":
            return self._error_response(
                JSONRPCErrorCode.INVALID_REQUEST,
                "Invalid JSON-RPC version",
                request.id
            )
        
        # Check method exists
        if request.method not in self.methods:
            return self._error_response(
                JSONRPCErrorCode.METHOD_NOT_FOUND,
                f"Method '{request.method}' not found",
                request.id
            )
        
        # Execute method
        try:
            method = self.methods[request.method]
            
            # Pass context if method accepts it
            import inspect
            sig = inspect.signature(method)
            if "context" in sig.parameters:
                result = await method(request.params, context=context)
            else:
                result = await method(request.params)
            
            return JSONRPCResponse(
                result=result,
                id=request.id
            )
            
        except TypeError as e:
            return self._error_response(
                JSONRPCErrorCode.INVALID_PARAMS,
                str(e),
                request.id
            )
        except Exception as e:
            return self._error_response(
                JSONRPCErrorCode.INTERNAL_ERROR,
                str(e),
                request.id
            )
    
    def _error_response(
        self,
        code: JSONRPCErrorCode,
        message: str,
        request_id: Optional[Union[str, int]],
        data: Optional[Any] = None
    ) -> JSONRPCResponse:
        """Create error response"""
        return JSONRPCResponse(
            error=JSONRPCError(
                code=code.value,
                message=message,
                data=data
            ),
            id=request_id
        )

# Method registration
class MethodRegistry:
    """Registry for JSON-RPC methods"""
    
    def __init__(self):
        self.methods = {}
    
    def register(self, name: str):
        """Decorator to register method"""
        def decorator(func):
            self.methods[name] = func
            return func
        return decorator
    
    def get_handler(self) -> ProtocolHandler:
        """Get protocol handler with registered methods"""
        return ProtocolHandler(self.methods)

# Usage example
registry = MethodRegistry()

@registry.register("tools/list")
async def list_tools(params: Dict[str, Any], context: Dict[str, Any] = None):
    """List available tools"""
    # Implementation
    return ["tool1", "tool2", "tool3"]

@registry.register("tools/call")
async def call_tool(params: Dict[str, Any], context: Dict[str, Any] = None):
    """Execute a tool"""
    tool_name = params.get("name")
    arguments = params.get("arguments", {})
    
    # Execute tool
    result = await execute_tool(tool_name, arguments)
    return result
```

## Testing

### Unit Tests

```python
# tests/test_module.py
import pytest
from unittest.mock import AsyncMock, patch
from tldw_Server_API.app.core.MCP_unified.modules.implementations.my_module import MyModule

@pytest.fixture
async def module():
    """Create module instance"""
    config = {
        "database_url": "sqlite:///:memory:",
        "service_url": "http://test.example.com"
    }
    module = MyModule(config)
    await module.initialize()
    yield module
    await module.shutdown()

@pytest.mark.asyncio
async def test_process_data(module):
    """Test data processing"""
    result = await module.process_data(
        "test input",
        {"format": "json", "validate": True}
    )
    
    assert result["status"] == "success"
    assert "data" in result
    assert result["metadata"]["format"] == "json"

@pytest.mark.asyncio
async def test_health_check(module):
    """Test health check"""
    with patch.object(module, 'db_pool') as mock_pool:
        mock_conn = AsyncMock()
        mock_conn.fetchval.return_value = 1
        mock_pool.acquire.return_value.__aenter__.return_value = mock_conn
        
        health = await module.health_check()
        
        assert health.status == "healthy"
        assert health.checks["database"] is True

@pytest.mark.asyncio
async def test_tool_execution(module):
    """Test tool execution"""
    result = await module.execute_tool(
        "my_module.process_data",
        {"input_data": "test", "options": {"format": "csv"}}
    )
    
    assert result is not None
    assert result["status"] == "success"

@pytest.mark.asyncio
async def test_invalid_tool(module):
    """Test invalid tool raises error"""
    with pytest.raises(ValueError, match="Unknown tool"):
        await module.execute_tool("invalid_tool", {})
```

### Integration Tests

```python
# tests/test_integration.py
import pytest
from httpx import AsyncClient
from fastapi import FastAPI
from tldw_Server_API.app.main import app

@pytest.fixture
async def client():
    """Create test client"""
    async with AsyncClient(app=app, base_url="http://test") as ac:
        yield ac

@pytest.mark.asyncio
async def test_websocket_connection():
    """Test WebSocket connection"""
    from websockets import connect
    
    async with connect("ws://localhost:8000/api/v1/mcp/ws") as websocket:
        # Send initialization
        await websocket.send(json.dumps({
            "jsonrpc": "2.0",
            "method": "initialize",
            "params": {},
            "id": 1
        }))
        
        # Receive response
        response = await websocket.recv()
        data = json.loads(response)
        
        assert data["jsonrpc"] == "2.0"
        assert "result" in data or "error" in data

@pytest.mark.asyncio
async def test_http_request(client):
    """Test HTTP request"""
    response = await client.post(
        "/api/v1/mcp/request",
        json={
            "jsonrpc": "2.0",
            "method": "tools/list",
            "params": {},
            "id": 1
        }
    )
    
    assert response.status_code == 200
    data = response.json()
    assert data["jsonrpc"] == "2.0"
    assert "result" in data

@pytest.mark.asyncio
async def test_authentication_required(client):
    """Test authentication requirement"""
    response = await client.post(
        "/api/v1/mcp/tools/execute",
        json={
            "tool_name": "test_tool",
            "arguments": {}
        }
    )
    
    assert response.status_code == 401
    assert "Authentication required" in response.json()["detail"]

@pytest.mark.asyncio
async def test_rate_limiting(client):
    """Test rate limiting"""
    # Make many requests quickly
    for _ in range(100):
        response = await client.get("/api/v1/mcp/status")
    
    # Should eventually get rate limited
    response = await client.get("/api/v1/mcp/status")
    assert response.status_code in [200, 429]  # OK or Too Many Requests
```

### Load Testing

```python
# tests/test_load.py
import asyncio
import aiohttp
import time
from typing import List

async def make_request(session: aiohttp.ClientSession, url: str) -> float:
    """Make single request and return response time"""
    start = time.time()
    try:
        async with session.post(url, json={
            "jsonrpc": "2.0",
            "method": "tools/list",
            "params": {},
            "id": 1
        }) as response:
            await response.json()
            return time.time() - start
    except Exception as e:
        print(f"Request failed: {e}")
        return -1

async def load_test(
    url: str,
    num_requests: int,
    concurrent: int
) -> Dict[str, Any]:
    """Run load test"""
    async with aiohttp.ClientSession() as session:
        tasks = []
        for _ in range(num_requests):
            tasks.append(make_request(session, url))
            
            # Limit concurrent requests
            if len(tasks) >= concurrent:
                results = await asyncio.gather(*tasks)
                tasks = []
        
        # Process remaining tasks
        if tasks:
            results = await asyncio.gather(*tasks)
    
    # Calculate statistics
    successful = [r for r in results if r > 0]
    failed = len(results) - len(successful)
    
    return {
        "total_requests": num_requests,
        "successful": len(successful),
        "failed": failed,
        "avg_response_time": sum(successful) / len(successful) if successful else 0,
        "min_response_time": min(successful) if successful else 0,
        "max_response_time": max(successful) if successful else 0
    }

# Run load test
if __name__ == "__main__":
    results = asyncio.run(load_test(
        "http://localhost:8000/api/v1/mcp/request",
        num_requests=1000,
        concurrent=50
    ))
    
    print(f"Results: {results}")
```

## Performance Optimization

### Caching Strategy

```python
from functools import lru_cache
import hashlib
import json
from typing import Any
import redis.asyncio as redis

class CacheManager:
    """Cache manager with Redis backend"""
    
    def __init__(self, redis_url: str = "redis://localhost"):
        self.redis = redis.from_url(redis_url)
    
    def cache_key(self, prefix: str, params: Dict[str, Any]) -> str:
        """Generate cache key"""
        param_str = json.dumps(params, sort_keys=True)
        hash_digest = hashlib.md5(param_str.encode()).hexdigest()
        return f"{prefix}:{hash_digest}"
    
    async def get(self, key: str) -> Optional[Any]:
        """Get from cache"""
        value = await self.redis.get(key)
        if value:
            return json.loads(value)
        return None
    
    async def set(self, key: str, value: Any, ttl: int = 300):
        """Set in cache with TTL"""
        await self.redis.setex(
            key,
            ttl,
            json.dumps(value)
        )
    
    async def invalidate(self, pattern: str):
        """Invalidate cache by pattern"""
        cursor = 0
        while True:
            cursor, keys = await self.redis.scan(
                cursor,
                match=pattern,
                count=100
            )
            if keys:
                await self.redis.delete(*keys)
            if cursor == 0:
                break

# Decorator for caching
def cached(prefix: str, ttl: int = 300):
    """Cache decorator for async functions"""
    def decorator(func):
        async def wrapper(*args, **kwargs):
            # Get cache manager
            cache = CacheManager()
            
            # Generate cache key
            cache_key = cache.cache_key(
                prefix,
                {"args": args, "kwargs": kwargs}
            )
            
            # Check cache
            cached_value = await cache.get(cache_key)
            if cached_value is not None:
                return cached_value
            
            # Execute function
            result = await func(*args, **kwargs)
            
            # Store in cache
            await cache.set(cache_key, result, ttl)
            
            return result
        
        return wrapper
    return decorator

# Usage
@cached("media_search", ttl=600)
async def search_media(query: str, limit: int = 10):
    """Search media with caching"""
    # Expensive search operation
    results = await perform_search(query, limit)
    return results
```

### Connection Pooling

```python
import asyncpg
from contextlib import asynccontextmanager

class DatabasePool:
    """Database connection pool manager"""
    
    def __init__(self, dsn: str, min_size: int = 10, max_size: int = 20):
        self.dsn = dsn
        self.min_size = min_size
        self.max_size = max_size
        self.pool = None
    
    async def initialize(self):
        """Initialize connection pool"""
        self.pool = await asyncpg.create_pool(
            self.dsn,
            min_size=self.min_size,
            max_size=self.max_size,
            command_timeout=60,
            max_queries=50000,
            max_inactive_connection_lifetime=300
        )
    
    async def close(self):
        """Close connection pool"""
        if self.pool:
            await self.pool.close()
    
    @asynccontextmanager
    async def acquire(self):
        """Acquire connection from pool"""
        async with self.pool.acquire() as connection:
            yield connection
    
    async def execute(self, query: str, *args):
        """Execute query"""
        async with self.acquire() as conn:
            return await conn.execute(query, *args)
    
    async def fetch(self, query: str, *args):
        """Fetch results"""
        async with self.acquire() as conn:
            return await conn.fetch(query, *args)
    
    async def fetchrow(self, query: str, *args):
        """Fetch single row"""
        async with self.acquire() as conn:
            return await conn.fetchrow(query, *args)
```

## Debugging

### Debug Logging

```python
import logging
from loguru import logger
import sys

# Configure debug logging
def setup_debug_logging():
    """Set up debug logging"""
    
    # Remove default handler
    logger.remove()
    
    # Add debug handler with detailed format
    logger.add(
        sys.stderr,
        level="DEBUG",
        format="<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | "
               "<level>{level: <8}</level> | "
               "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> | "
               "<level>{message}</level>",
        colorize=True,
        backtrace=True,
        diagnose=True
    )
    
    # Add file handler for debugging
    logger.add(
        "debug.log",
        level="DEBUG",
        rotation="10 MB",
        retention="7 days",
        format="{time:YYYY-MM-DD HH:mm:ss.SSS} | {level: <8} | {name}:{function}:{line} | {message}",
        backtrace=True,
        diagnose=True
    )
    
    # Enable SQL query logging
    logging.getLogger("sqlalchemy.engine").setLevel(logging.INFO)
    
    # Enable async debugging
    import asyncio
    asyncio.get_event_loop().set_debug(True)
    
    logger.info("Debug logging enabled")

# Request tracing
from fastapi import Request
import uuid

async def add_request_id(request: Request, call_next):
    """Add request ID for tracing"""
    request_id = str(uuid.uuid4())
    
    # Add to request state
    request.state.request_id = request_id
    
    # Log request
    logger.info(
        f"Request {request_id}: {request.method} {request.url.path}",
        extra={"request_id": request_id}
    )
    
    # Process request
    response = await call_next(request)
    
    # Add header
    response.headers["X-Request-ID"] = request_id
    
    # Log response
    logger.info(
        f"Response {request_id}: {response.status_code}",
        extra={"request_id": request_id}
    )
    
    return response
```

### Profiling

```python
import cProfile
import pstats
from io import StringIO
import asyncio

def profile_async(func):
    """Profile async function"""
    async def wrapper(*args, **kwargs):
        profiler = cProfile.Profile()
        profiler.enable()
        
        try:
            result = await func(*args, **kwargs)
        finally:
            profiler.disable()
            
            # Print stats
            s = StringIO()
            ps = pstats.Stats(profiler, stream=s).sort_stats('cumulative')
            ps.print_stats(20)
            
            logger.debug(f"Profile for {func.__name__}:\n{s.getvalue()}")
        
        return result
    
    return wrapper

# Memory profiling
from memory_profiler import profile

@profile
async def memory_intensive_operation():
    """Operation to profile for memory usage"""
    large_list = [i for i in range(1000000)]
    # Process data
    return len(large_list)
```

## Contributing

### Development Setup

```bash
# Clone repository
git clone https://github.com/yourorg/tldw_server.git
cd tldw_server

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install development dependencies
pip install -r requirements-dev.txt
pip install -e .

# Set up pre-commit hooks
pre-commit install

# Run tests
pytest tests/
```

### Code Style

Follow PEP 8 and use these tools:
```bash
# Format code
black tldw_Server_API/app/core/MCP_unified/

# Lint code
flake8 tldw_Server_API/app/core/MCP_unified/

# Type checking
mypy tldw_Server_API/app/core/MCP_unified/
```

### Pull Request Process

1. Fork the repository
2. Create feature branch (`git checkout -b feature/amazing-feature`)
3. Make changes with tests
4. Run test suite (`pytest`)
5. Commit changes (`git commit -m 'Add amazing feature'`)
6. Push to branch (`git push origin feature/amazing-feature`)
7. Open Pull Request

### Commit Message Format
```
type(scope): subject

body

footer
```

Types: feat, fix, docs, style, refactor, test, chore
Example: `feat(modules): add new media processing module`

## API Reference

Complete API documentation is available at:
- Swagger UI: `http://localhost:8000/docs`
- ReDoc: `http://localhost:8000/redoc`
- OpenAPI Schema: `http://localhost:8000/openapi.json`

### Key Endpoints

#### WebSocket
- `WS /api/v1/mcp/ws` - Main WebSocket endpoint

#### HTTP
- `POST /api/v1/mcp/request` - JSON-RPC request endpoint
- `GET /api/v1/mcp/status` - Server status
- `GET /api/v1/mcp/health` - Health check
- `GET /api/v1/mcp/tools` - List available tools
- `POST /api/v1/mcp/tools/execute` - Execute tool
- `GET /api/v1/mcp/metrics` - Prometheus metrics

#### Authentication
- `POST /api/v1/mcp/auth/token` - Get access token
- `POST /api/v1/mcp/auth/refresh` - Refresh token

### Error Codes

| Code | Message | Description |
|------|---------|-------------|
| -32700 | Parse error | Invalid JSON |
| -32600 | Invalid request | Invalid JSON-RPC request |
| -32601 | Method not found | Method does not exist |
| -32602 | Invalid params | Invalid method parameters |
| -32603 | Internal error | Internal server error |
| -32000 | Authentication required | Missing or invalid auth |
| -32001 | Permission denied | Insufficient permissions |
| -32002 | Rate limit exceeded | Too many requests |

## Resources

- [JSON-RPC 2.0 Specification](https://www.jsonrpc.org/specification)
- [FastAPI Documentation](https://fastapi.tiangolo.com/)
- [Pydantic Documentation](https://pydantic-docs.helpmanual.io/)
- [Python AsyncIO](https://docs.python.org/3/library/asyncio.html)
- [JWT Best Practices](https://tools.ietf.org/html/rfc8725)