# mcp_endpoint.py - MCP Server API Endpoints
"""
API endpoints for Model Context Protocol (MCP) server.

Provides WebSocket endpoint for MCP clients and REST endpoints for management.
"""

from typing import Dict, Any, List, Optional
import logging

from fastapi import APIRouter, WebSocket, HTTPException, Depends, Query
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from tldw_Server_API.app.core.MCP import (
    get_mcp_server,
    MCPServer,
    MCPTool,
    MCPContext,
    mcp_tool
)

logger = logging.getLogger(__name__)

# Create router
router = APIRouter(prefix="/mcp", tags=["MCP"])


# REST API Models
class ToolInfo(BaseModel):
    """Tool information response"""
    name: str
    description: str
    parameters: List[Dict[str, Any]]
    tags: List[str]
    examples: Optional[List[Dict[str, Any]]] = None


class ContextCreateRequest(BaseModel):
    """Context creation request"""
    name: Optional[str] = None
    content: Dict[str, Any] = {}
    metadata: Dict[str, Any] = {}
    ttl: Optional[int] = None


class ContextUpdateRequest(BaseModel):
    """Context update request"""
    updates: Dict[str, Any]
    merge: bool = True


class ServerStatus(BaseModel):
    """Server status response"""
    status: str
    version: str
    active_sessions: int
    active_contexts: int
    available_tools: int
    capabilities: Dict[str, Any]


# WebSocket endpoint
@router.websocket("/ws")
async def mcp_websocket(websocket: WebSocket):
    """
    WebSocket endpoint for MCP clients.
    
    The MCP protocol uses WebSocket for real-time bidirectional communication.
    Clients must send a CONNECT message first to establish a session.
    """
    server = get_mcp_server()
    await server.handle_websocket(websocket)


# REST endpoints for management and debugging

@router.get("/status", response_model=ServerStatus)
async def get_server_status():
    """Get MCP server status"""
    server = get_mcp_server()
    
    # Count active contexts
    contexts = await server.context_manager.list_contexts()
    
    return ServerStatus(
        status="running",
        version="0.1.0",
        active_sessions=len(server._sessions),
        active_contexts=len(contexts),
        available_tools=len(server.tool_registry.list_tools()),
        capabilities=server._capabilities.dict()
    )


@router.get("/tools", response_model=List[ToolInfo])
async def list_tools(
    tag: Optional[str] = Query(None, description="Filter tools by tag")
):
    """List available MCP tools"""
    server = get_mcp_server()
    
    tags = [tag] if tag else None
    tools = server.tool_registry.list_tools(tags=tags)
    
    return [
        ToolInfo(
            name=tool.name,
            description=tool.description,
            parameters=[p.dict() for p in tool.parameters],
            tags=tool.tags,
            examples=tool.examples
        )
        for tool in tools
    ]


@router.get("/tools/{tool_name}", response_model=ToolInfo)
async def get_tool_info(tool_name: str):
    """Get information about a specific tool"""
    server = get_mcp_server()
    
    tool = server.tool_registry.get_tool(tool_name)
    if not tool:
        raise HTTPException(status_code=404, detail=f"Tool '{tool_name}' not found")
    
    return ToolInfo(
        name=tool.name,
        description=tool.description,
        parameters=[p.dict() for p in tool.parameters],
        tags=tool.tags,
        examples=tool.examples
    )


@router.post("/contexts", response_model=Dict[str, Any])
async def create_context(request: ContextCreateRequest):
    """Create a new context"""
    server = get_mcp_server()
    
    context = await server.context_manager.create_context(
        name=request.name,
        content=request.content,
        metadata=request.metadata,
        ttl=request.ttl
    )
    
    return {
        "id": context.id,
        "name": context.name,
        "created_at": context.created_at.isoformat(),
        "expires_at": context.expires_at.isoformat() if context.expires_at else None
    }


@router.get("/contexts", response_model=List[Dict[str, Any]])
async def list_contexts(
    include_expired: bool = Query(False, description="Include expired contexts")
):
    """List all contexts"""
    server = get_mcp_server()
    
    contexts = await server.context_manager.list_contexts(
        include_expired=include_expired
    )
    
    return [
        {
            "id": ctx.id,
            "name": ctx.name,
            "created_at": ctx.created_at.isoformat(),
            "updated_at": ctx.updated_at.isoformat(),
            "expires_at": ctx.expires_at.isoformat() if ctx.expires_at else None,
            "size": len(str(ctx.content))
        }
        for ctx in contexts
    ]


@router.get("/contexts/{context_id}")
async def get_context(context_id: str):
    """Get a specific context"""
    server = get_mcp_server()
    
    context = await server.context_manager.get_context(context_id)
    if not context:
        raise HTTPException(status_code=404, detail=f"Context '{context_id}' not found")
    
    return context.dict()


@router.put("/contexts/{context_id}")
async def update_context(context_id: str, request: ContextUpdateRequest):
    """Update a context"""
    server = get_mcp_server()
    
    context = await server.context_manager.update_context(
        context_id,
        request.updates,
        request.merge
    )
    
    if not context:
        raise HTTPException(status_code=404, detail=f"Context '{context_id}' not found")
    
    return {"status": "updated", "context_id": context_id}


@router.delete("/contexts/{context_id}")
async def delete_context(context_id: str):
    """Delete a context"""
    server = get_mcp_server()
    
    await server.context_manager.delete_context(context_id)
    
    return {"status": "deleted", "context_id": context_id}


# Register some tldw-specific tools
@mcp_tool(
    description="Search media content in tldw database",
    tags=["tldw", "search"]
)
async def search_media(
    query: str,
    limit: int = 10,
    media_type: Optional[str] = None,
    _context=None
) -> List[Dict[str, Any]]:
    """Search for media content in the tldw database"""
    # This would integrate with the existing media search functionality
    # For now, return a placeholder
    return [
        {
            "id": 1,
            "title": f"Sample result for: {query}",
            "type": media_type or "video",
            "url": "https://example.com/video1"
        }
    ]


@mcp_tool(
    description="Get transcript for a media item",
    tags=["tldw", "transcript"]
)
async def get_transcript(
    media_id: int,
    format: str = "text",
    _context=None
) -> Dict[str, Any]:
    """Get the transcript for a specific media item"""
    # This would integrate with the existing transcript functionality
    # For now, return a placeholder
    return {
        "media_id": media_id,
        "format": format,
        "transcript": "This is a sample transcript for the requested media."
    }


@mcp_tool(
    description="Summarize media content",
    tags=["tldw", "summarization"]
)
async def summarize_media(
    media_id: int,
    prompt: Optional[str] = None,
    max_length: int = 500,
    _context=None
) -> Dict[str, Any]:
    """Generate a summary for media content"""
    # This would integrate with the existing summarization functionality
    # For now, return a placeholder
    return {
        "media_id": media_id,
        "summary": "This is a sample summary of the media content.",
        "length": 42
    }


# Health check endpoint
@router.get("/health")
async def health_check():
    """Check MCP server health"""
    server = get_mcp_server()
    
    return {
        "status": "healthy",
        "server_running": True,
        "version": "0.1.0"
    }


# Add this router to your main FastAPI app in main.py:
# from tldw_Server_API.app.api.v1.endpoints import mcp_endpoint
# app.include_router(mcp_endpoint.router, prefix="/api/v1")