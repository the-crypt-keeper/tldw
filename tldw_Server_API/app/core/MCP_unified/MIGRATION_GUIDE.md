# Migration Guide: MCP v1/v2 to Unified MCP

## Overview
This guide helps you migrate from the existing MCP v1/v2 implementations to the new unified MCP module.

## Key Improvements in Unified MCP

### Security Fixes
- ✅ No hardcoded secrets - all from environment variables
- ✅ JWT token rotation and revocation
- ✅ Fine-grained RBAC permissions
- ✅ Rate limiting (in-memory and Redis)
- ✅ Input validation and sanitization

### Production Features
- ✅ Health monitoring with circuit breakers
- ✅ Metrics collection
- ✅ Connection pooling
- ✅ Graceful degradation
- ✅ Distributed deployment support

## Migration Steps

### 1. Environment Setup

Create a `.env` file with required variables:

```bash
# Required Security Settings
MCP_JWT_SECRET=$(openssl rand -base64 32)
MCP_API_KEY_SALT=$(openssl rand -base64 32)

# Optional Settings
MCP_DATABASE_URL=sqlite+aiosqlite:///./Databases/mcp_unified.db
MCP_RATE_LIMIT_ENABLED=true
MCP_RATE_LIMIT_RPM=60
MCP_LOG_LEVEL=INFO
```

### 2. Update Imports

Replace old imports with unified imports:

```python
# Old (MCP v1)
from tldw_Server_API.app.core.MCP import MCPServer, MCPTool

# Old (MCP v2)
from tldw_Server_API.app.core.MCP_v2 import MCPServer, BaseModule

# New (Unified)
from tldw_Server_API.app.core.MCP_unified import (
    MCPServer,
    BaseModule,
    get_mcp_server,
    ModuleConfig
)
```

### 3. Migrate Modules

#### Example: Migrating Media Module

**Old Module (MCP v2):**
```python
class MediaModule(BaseModule):
    def __init__(self, config: ModuleConfig):
        super().__init__(config)
        self.db = MediaDatabase(config.settings.get("db_path"))
    
    async def on_initialize(self):
        # Simple initialization
        pass
    
    async def get_tools(self):
        return [{"name": "search_media", ...}]
```

**New Module (Unified):**
```python
from tldw_Server_API.app.core.MCP_unified.modules import BaseModule, ModuleConfig
from typing import Dict, Any, List

class MediaModule(BaseModule):
    async def on_initialize(self) -> None:
        """Initialize with error handling"""
        try:
            self.db = MediaDatabase(self.config.settings.get("db_path"))
            await self.db.connect()  # Async connection
        except Exception as e:
            logger.error(f"Failed to initialize database: {e}")
            raise
    
    async def on_shutdown(self) -> None:
        """Graceful shutdown"""
        if hasattr(self, 'db'):
            await self.db.close()
    
    async def check_health(self) -> Dict[str, bool]:
        """Required health checks"""
        checks = {
            "database": False,
            "service": True
        }
        
        try:
            # Check database connection
            await self.db.ping()
            checks["database"] = True
        except:
            pass
        
        return checks
    
    async def get_tools(self) -> List[Dict[str, Any]]:
        """Tools with proper schema"""
        from tldw_Server_API.app.core.MCP_unified.modules.base import create_tool_definition
        
        return [
            create_tool_definition(
                name="search_media",
                description="Search media content",
                parameters={
                    "properties": {
                        "query": {"type": "string"},
                        "limit": {"type": "integer", "default": 10}
                    },
                    "required": ["query"]
                }
            )
        ]
    
    async def execute_tool(self, tool_name: str, arguments: Dict[str, Any]) -> Any:
        """Execute with validation"""
        # Sanitize inputs
        arguments = self.sanitize_input(arguments)
        
        if tool_name == "search_media":
            query = arguments.get("query", "")
            limit = arguments.get("limit", 10)
            
            # Validate
            if not query or len(query) > 1000:
                raise ValueError("Invalid query")
            
            # Execute with circuit breaker
            return await self.execute_with_circuit_breaker(
                self._search_media, query, limit
            )
        
        raise ValueError(f"Unknown tool: {tool_name}")
    
    async def _search_media(self, query: str, limit: int):
        """Actual search implementation"""
        results = await self.db.search(query, limit=limit)
        return results
```

### 4. Update API Endpoints

**Old Endpoint:**
```python
from tldw_Server_API.app.core.MCP import get_mcp_server

@router.websocket("/mcp/ws")
async def websocket_endpoint(websocket: WebSocket):
    server = get_mcp_server()
    await server.handle_connection(websocket)
```

**New Endpoint:**
```python
from tldw_Server_API.app.core.MCP_unified import get_mcp_server

@router.websocket("/mcp/ws")
async def websocket_endpoint(
    websocket: WebSocket,
    client_id: Optional[str] = Query(None),
    token: Optional[str] = Query(None)  # Authentication support
):
    server = get_mcp_server()
    
    # Initialize if needed
    if not server.initialized:
        await server.initialize()
    
    # Handle with authentication
    await server.handle_websocket(
        websocket,
        client_id=client_id,
        auth_token=token
    )
```

### 5. Update Main Application

In `main.py`:

```python
from fastapi import FastAPI
from tldw_Server_API.app.core.MCP_unified.server import lifespan

# Use lifespan for proper initialization/shutdown
app = FastAPI(lifespan=lifespan)

# Include unified router
from tldw_Server_API.app.api.v1.endpoints import mcp_unified_endpoint
app.include_router(mcp_unified_endpoint.router, prefix="/api/v1")
```

## Module Registration

Register modules during server initialization:

```python
from tldw_Server_API.app.core.MCP_unified import (
    get_module_registry,
    ModuleConfig
)
from your_modules import MediaModule, RAGModule, NotesModule

async def register_modules():
    registry = get_module_registry()
    
    # Media Module
    await registry.register_module(
        "media",
        MediaModule,
        ModuleConfig(
            name="media",
            version="1.0.0",
            description="Media management",
            settings={"db_path": "./Databases/Media_DB.db"}
        )
    )
    
    # RAG Module
    await registry.register_module(
        "rag",
        RAGModule,
        ModuleConfig(
            name="rag",
            version="1.0.0",
            settings={"collection": "tldw_media"}
        )
    )
```

## Authentication Integration

### Creating Tokens
```python
from tldw_Server_API.app.core.MCP_unified.auth import create_access_token

# Create token for user
token = create_access_token(
    subject=user_id,
    username=username,
    roles=["user"],
    permissions=["tools:execute", "resources:read"]
)
```

### Protecting Endpoints
```python
from tldw_Server_API.app.core.MCP_unified.auth.jwt_manager import get_jwt_manager

async def require_auth(token: str):
    manager = get_jwt_manager()
    return manager.verify_token(token)
```

## Testing Your Migration

### 1. Unit Tests
```python
import pytest
from your_module import YourModule
from tldw_Server_API.app.core.MCP_unified import ModuleConfig

@pytest.mark.asyncio
async def test_module_health():
    config = ModuleConfig(name="test")
    module = YourModule(config)
    
    await module.initialize()
    health = await module.health_check()
    
    assert health.is_operational
    
    await module.shutdown()
```

### 2. Integration Tests
```python
@pytest.mark.asyncio
async def test_tool_execution():
    server = MCPServer()
    await server.initialize()
    
    # Register your module
    await server.module_registry.register_module(
        "test", YourModule, ModuleConfig(name="test")
    )
    
    # Execute tool
    request = MCPRequest(
        method="tools/call",
        params={"name": "your_tool", "arguments": {}}
    )
    
    response = await server.handle_http_request(request)
    assert response.error is None
    
    await server.shutdown()
```

## Common Issues and Solutions

### Issue: "JWT secret key not configured"
**Solution:** Set `MCP_JWT_SECRET` environment variable

### Issue: Module health check fails
**Solution:** Implement all required health checks in `check_health()`

### Issue: Rate limiting blocking requests
**Solution:** Adjust `MCP_RATE_LIMIT_RPM` or disable for development

### Issue: Database connection errors
**Solution:** Use async database drivers (e.g., `aiosqlite` for SQLite)

## Rollback Plan

If you need to rollback:

1. Keep old MCP modules intact during migration
2. Use feature flags to switch between implementations
3. Run both in parallel during transition
4. Gradually migrate traffic using load balancer

## Performance Considerations

### Before Migration
- Measure current performance metrics
- Document baseline latencies
- Note memory usage patterns

### After Migration
- Compare metrics with baseline
- Monitor circuit breaker triggers
- Watch error rates
- Check cache hit rates

## Support

For migration assistance:
1. Check test examples in `/tests/`
2. Review module examples in `/modules/implementations/`
3. Consult `IMPLEMENTATION_STATUS.md` for progress
4. File issues for specific problems

## Checklist

- [ ] Environment variables configured
- [ ] Dependencies installed (`pip install -r requirements.txt`)
- [ ] Modules migrated to new base class
- [ ] Health checks implemented
- [ ] API endpoints updated
- [ ] Authentication integrated
- [ ] Tests passing
- [ ] Performance validated
- [ ] Documentation updated
- [ ] Deployment plan ready