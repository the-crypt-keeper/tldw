"""
MCP v2 API Endpoints for tldw

Provides both WebSocket and HTTP endpoints for the modular MCP server.
"""

from typing import Dict, Any, List, Optional
from fastapi import APIRouter, WebSocket, HTTPException, Depends, Query, Security
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from loguru import logger

from tldw_Server_API.app.core.MCP_v2 import (
    get_mcp_server,
    MCPRequest,
    MCPResponse,
    MCPServer
)
from tldw_Server_API.app.core.MCP_v2.auth import (
    get_current_user,
    get_optional_user,
    require_roles,
    rate_limit,
    RateLimitMiddleware
)
from tldw_Server_API.app.core.MCP_v2.schemas import MCPUser, UserRole

# Create router
router = APIRouter(prefix="/mcp/v2", tags=["MCP v2"])


# Request/Response models
class MCPStatusResponse(BaseModel):
    """Server status response"""
    status: str
    version: str
    uptime_seconds: float
    modules: Dict[str, Any]
    health: Dict[str, Any]


class ToolListResponse(BaseModel):
    """Tool list response"""
    tools: List[Dict[str, Any]]
    count: int


class ModuleListResponse(BaseModel):
    """Module list response"""
    modules: List[Dict[str, Any]]
    count: int


# WebSocket endpoint for MCP protocol
@router.websocket("/ws")
async def mcp_websocket(
    websocket: WebSocket,
    client_id: Optional[str] = Query(None, description="Client identifier")
):
    """
    WebSocket endpoint for MCP v2 protocol.
    
    Supports full MCP protocol over WebSocket including:
    - Tool discovery and execution
    - Resource listing and reading
    - Prompt management
    - Real-time bidirectional communication
    """
    server = get_mcp_server()
    
    # Ensure server is initialized
    if not server.initialized:
        await server.initialize()
    
    await server.handle_websocket(websocket, client_id=client_id)


# HTTP endpoint for MCP protocol (for simpler integrations)
@router.post("/request", response_model=MCPResponse, dependencies=[Depends(rate_limit(rate=100, per=60))])
async def mcp_http_request(
    request: MCPRequest,
    client_id: Optional[str] = Query(None, description="Client identifier"),
    current_user: Optional[MCPUser] = Depends(get_optional_user)
):
    """
    HTTP endpoint for MCP v2 protocol.
    
    Accepts MCP requests via HTTP POST for simpler integrations
    that don't require WebSocket.
    """
    server = get_mcp_server()
    
    # Ensure server is initialized
    if not server.initialized:
        await server.initialize()
    
    try:
        response = await server.handle_http_request(request, client_id=client_id)
        return response
    except Exception as e:
        logger.error(f"Error processing MCP request: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# Management endpoints

@router.get("/status", response_model=MCPStatusResponse)
async def get_status():
    """
    Get MCP server status including module health.
    """
    server = get_mcp_server()
    
    # Ensure server is initialized
    if not server.initialized:
        await server.initialize()
    
    status = await server.get_status()
    return MCPStatusResponse(**status)


@router.get("/tools", response_model=ToolListResponse)
async def list_tools(
    module: Optional[str] = Query(None, description="Filter by module"),
    tag: Optional[str] = Query(None, description="Filter by tag")
):
    """
    List all available tools across all modules.
    """
    server = get_mcp_server()
    
    # Ensure server is initialized
    if not server.initialized:
        await server.initialize()
    
    tools = await server.list_tools()
    
    # Apply filters if provided
    if module:
        tools = [t for t in tools if t.get("module") == module]
    if tag:
        tools = [t for t in tools if tag in t.get("tags", [])]
    
    return ToolListResponse(tools=tools, count=len(tools))


@router.get("/modules", response_model=ModuleListResponse)
async def list_modules():
    """
    List all registered modules.
    """
    server = get_mcp_server()
    
    # Ensure server is initialized
    if not server.initialized:
        await server.initialize()
    
    registrations = await server.registry.list_registrations()
    
    modules = [
        {
            "id": r.module_id,
            "name": r.name,
            "version": r.version,
            "department": r.department,
            "status": r.status,
            "capabilities": r.capabilities,
            "registered_at": r.registered_at.isoformat()
        }
        for r in registrations
    ]
    
    return ModuleListResponse(modules=modules, count=len(modules))


@router.get("/modules/{module_id}/health")
async def get_module_health(module_id: str):
    """
    Get health status for a specific module.
    """
    server = get_mcp_server()
    
    # Ensure server is initialized
    if not server.initialized:
        await server.initialize()
    
    module = await server.registry.get_module(module_id)
    if not module:
        raise HTTPException(status_code=404, detail=f"Module {module_id} not found")
    
    health = await module.health_check()
    return JSONResponse(content=health)


@router.post("/modules/{module_id}/reload")
async def reload_module(module_id: str):
    """
    Reload a specific module.
    """
    server = get_mcp_server()
    
    # Ensure server is initialized
    if not server.initialized:
        await server.initialize()
    
    try:
        # Get module registration
        registration = await server.registry.get_module_registration(module_id)
        if not registration:
            raise HTTPException(status_code=404, detail=f"Module {module_id} not found")
        
        # Unregister and re-register the module
        await server.registry.unregister_module(module_id)
        
        # Re-register based on module type
        # This would need to be expanded as more modules are added
        if module_id == "media":
            from tldw_Server_API.app.core.MCP_v2.modules.media_module import MediaModule
            from tldw_Server_API.app.core.MCP_v2.schemas import ModuleConfig, ModuleCapability
            
            config = ModuleConfig(
                name="media",
                description="Media ingestion and management module",
                version="1.0.0",
                department="media",
                enabled=True,
                capabilities=[ModuleCapability.TOOLS, ModuleCapability.RESOURCES],
                settings={"db_path": "./Databases/Media_DB_v2.db"}
            )
            await server.registry.register_module(module_id, MediaModule(config), registration)
            
        elif module_id == "rag":
            from tldw_Server_API.app.core.MCP_v2.modules.rag_module import RAGModule
            from tldw_Server_API.app.core.MCP_v2.schemas import ModuleConfig, ModuleCapability
            
            config = ModuleConfig(
                name="rag",
                description="RAG and vector search module",
                version="1.0.0",
                department="rag",
                enabled=True,
                capabilities=[ModuleCapability.TOOLS, ModuleCapability.RESOURCES],
                settings={
                    "collection_name": "tldw_media",
                    "embedding_model": "all-MiniLM-L6-v2"
                }
            )
            await server.registry.register_module(module_id, RAGModule(config), registration)
        else:
            raise HTTPException(status_code=400, detail=f"Unknown module type: {module_id}")
        
        return JSONResponse(content={"message": f"Module {module_id} reloaded successfully"})
        
    except Exception as e:
        logger.error(f"Error reloading module {module_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# Quick test endpoint
@router.post("/test/echo")
async def test_echo(message: str):
    """
    Test endpoint to verify MCP v2 is working.
    """
    server = get_mcp_server()
    
    # Ensure server is initialized
    if not server.initialized:
        await server.initialize()
    
    # Create a simple echo request
    request = MCPRequest(
        method="tools/call",
        params={
            "name": "media.search_media",
            "arguments": {"query": message, "limit": 1}
        }
    )
    
    response = await server.handle_http_request(request, client_id="test")
    
    return {
        "input": message,
        "mcp_response": response.dict()
    }