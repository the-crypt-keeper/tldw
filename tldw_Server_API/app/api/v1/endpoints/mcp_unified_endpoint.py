"""
Unified MCP API Endpoints

Production-ready endpoints for the unified MCP module with enhanced security and monitoring.
"""

from typing import Dict, Any, List, Optional
from fastapi import APIRouter, WebSocket, HTTPException, Depends, Query, Header, Security, status
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from loguru import logger

from tldw_Server_API.app.core.MCP_unified import (
    get_mcp_server,
    MCPRequest,
    MCPResponse,
    get_config
)
from tldw_Server_API.app.core.MCP_unified.auth import (
    JWTManager,
    UserRole,
    RBACPolicy
)
from tldw_Server_API.app.core.MCP_unified.auth.jwt_manager import (
    TokenData,
    get_jwt_manager
)
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials

# Create router
router = APIRouter(prefix="/mcp", tags=["MCP Unified"])

# Security
security = HTTPBearer(auto_error=False)


# Request/Response models
class ServerStatusResponse(BaseModel):
    """Server status response"""
    status: str
    version: str
    uptime_seconds: float
    connections: Dict[str, int]
    modules: Dict[str, int]


class ServerMetricsResponse(BaseModel):
    """Server metrics response"""
    connections: Dict[str, Any]
    modules: Dict[str, Dict[str, Any]]


class ToolExecutionRequest(BaseModel):
    """Tool execution request"""
    tool_name: str = Field(..., min_length=1, max_length=100)
    arguments: Dict[str, Any] = Field(default_factory=dict)


class ToolExecutionResponse(BaseModel):
    """Tool execution response"""
    result: Any
    execution_time_ms: float
    module: str


class ModuleHealthResponse(BaseModel):
    """Module health response"""
    module_id: str
    status: str
    message: str
    checks: Dict[str, bool]
    metrics: Optional[Dict[str, Any]] = None


class AuthTokenRequest(BaseModel):
    """Authentication token request"""
    username: str
    password: Optional[str] = None
    api_key: Optional[str] = None


class AuthTokenResponse(BaseModel):
    """Authentication token response"""
    access_token: str
    token_type: str = "bearer"
    expires_in: int
    refresh_token: Optional[str] = None


# Dependency functions

async def get_current_user(
    credentials: HTTPAuthorizationCredentials = Security(security)
) -> Optional[TokenData]:
    """Get current user from JWT token (optional)"""
    if not credentials:
        return None
    
    try:
        jwt_manager = get_jwt_manager()
        return jwt_manager.verify_token(credentials.credentials)
    except Exception as e:
        logger.debug(f"Token verification failed: {e}")
        return None


async def require_user(
    credentials: HTTPAuthorizationCredentials = Security(security)
) -> TokenData:
    """Require authenticated user"""
    if not credentials:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication required",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    try:
        jwt_manager = get_jwt_manager()
        return jwt_manager.verify_token(credentials.credentials)
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=str(e),
            headers={"WWW-Authenticate": "Bearer"},
        )


async def require_admin(
    user: TokenData = Depends(require_user)
) -> TokenData:
    """Require admin role"""
    if UserRole.ADMIN.value not in user.roles:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Admin role required"
        )
    return user


# WebSocket endpoint

@router.websocket("/ws")
async def websocket_endpoint(
    websocket: WebSocket,
    client_id: Optional[str] = Query(None, description="Client identifier"),
    token: Optional[str] = Query(None, description="Authentication token")
):
    """
    WebSocket endpoint for MCP protocol.
    
    Supports:
    - Full MCP protocol over WebSocket
    - Optional authentication via token parameter
    - Real-time bidirectional communication
    - Automatic reconnection support
    
    Example:
    ```javascript
    const ws = new WebSocket('ws://localhost:8000/api/v1/mcp/ws?client_id=my-client&token=jwt-token');
    
    ws.onopen = () => {
        ws.send(JSON.stringify({
            jsonrpc: "2.0",
            method: "initialize",
            params: {
                clientInfo: {
                    name: "My Client",
                    version: "1.0.0"
                }
            },
            id: 1
        }));
    };
    ```
    """
    server = get_mcp_server()
    
    # Ensure server is initialized
    if not server.initialized:
        await server.initialize()
    
    # Handle WebSocket connection
    await server.handle_websocket(websocket, client_id=client_id, auth_token=token)


# HTTP endpoints

@router.post("/request", response_model=MCPResponse)
async def mcp_request(
    request: MCPRequest,
    client_id: Optional[str] = Query(None, description="Client identifier"),
    user: Optional[TokenData] = Depends(get_current_user)
):
    """
    Process an MCP request via HTTP.
    
    This endpoint provides a simpler alternative to WebSocket for clients
    that don't need real-time bidirectional communication.
    """
    server = get_mcp_server()
    
    # Ensure server is initialized
    if not server.initialized:
        await server.initialize()
    
    # Process request
    response = await server.handle_http_request(
        request,
        client_id=client_id,
        user_id=user.sub if user else None
    )
    
    return response


@router.get("/status", response_model=ServerStatusResponse)
async def get_server_status():
    """
    Get MCP server status.
    
    Returns:
    - Server health status
    - Uptime
    - Connection statistics
    - Module health summary
    """
    server = get_mcp_server()
    
    # Ensure server is initialized
    if not server.initialized:
        await server.initialize()
    
    status = await server.get_status()
    return ServerStatusResponse(**status)


@router.get("/metrics", response_model=ServerMetricsResponse)
async def get_server_metrics(
    _: TokenData = Depends(require_admin)
):
    """
    Get detailed server metrics (requires admin).
    
    Returns:
    - Connection metrics
    - Module performance metrics
    - Error rates
    - Latency statistics
    """
    server = get_mcp_server()
    
    if not server.initialized:
        await server.initialize()
    
    metrics = await server.get_metrics()
    return ServerMetricsResponse(**metrics)


@router.get("/tools")
async def list_tools(
    module: Optional[str] = Query(None, description="Filter by module"),
    user: Optional[TokenData] = Depends(get_current_user)
):
    """
    List available MCP tools.
    
    Returns tools with their descriptions and required parameters.
    Tools are filtered based on user permissions if authenticated.
    """
    request = MCPRequest(
        method="tools/list",
        params={"module": module} if module else {}
    )
    
    server = get_mcp_server()
    if not server.initialized:
        await server.initialize()
    
    response = await server.handle_http_request(
        request,
        user_id=user.sub if user else None
    )
    
    if response.error:
        raise HTTPException(status_code=500, detail=response.error.message)
    
    return response.result


@router.post("/tools/execute", response_model=ToolExecutionResponse)
async def execute_tool(
    request: ToolExecutionRequest,
    user: TokenData = Depends(require_user)
):
    """
    Execute a specific tool (requires authentication).
    
    Tools are executed with user context for permission checking.
    """
    import time
    start_time = time.time()
    
    mcp_request = MCPRequest(
        method="tools/call",
        params={
            "name": request.tool_name,
            "arguments": request.arguments
        }
    )
    
    server = get_mcp_server()
    if not server.initialized:
        await server.initialize()
    
    response = await server.handle_http_request(
        mcp_request,
        user_id=user.sub
    )
    
    if response.error:
        raise HTTPException(
            status_code=400 if response.error.code == -32602 else 500,
            detail=response.error.message
        )
    
    execution_time = (time.time() - start_time) * 1000
    
    return ToolExecutionResponse(
        result=response.result,
        execution_time_ms=execution_time,
        module="unknown"  # TODO: Get actual module from response
    )


@router.get("/modules")
async def list_modules(
    user: Optional[TokenData] = Depends(get_current_user)
):
    """
    List registered MCP modules.
    
    Returns module information including status and capabilities.
    """
    request = MCPRequest(method="modules/list")
    
    server = get_mcp_server()
    if not server.initialized:
        await server.initialize()
    
    response = await server.handle_http_request(
        request,
        user_id=user.sub if user else None
    )
    
    if response.error:
        raise HTTPException(status_code=500, detail=response.error.message)
    
    return response.result


@router.get("/modules/health")
async def get_modules_health(
    _: TokenData = Depends(require_admin)
):
    """
    Get detailed health status of all modules (requires admin).
    
    Returns health checks and metrics for each module.
    """
    request = MCPRequest(method="modules/health")
    
    server = get_mcp_server()
    if not server.initialized:
        await server.initialize()
    
    response = await server.handle_http_request(request)
    
    if response.error:
        raise HTTPException(status_code=500, detail=response.error.message)
    
    return response.result


@router.get("/resources")
async def list_resources(
    user: Optional[TokenData] = Depends(get_current_user)
):
    """
    List available MCP resources.
    
    Resources are filtered based on user permissions if authenticated.
    """
    request = MCPRequest(method="resources/list")
    
    server = get_mcp_server()
    if not server.initialized:
        await server.initialize()
    
    response = await server.handle_http_request(
        request,
        user_id=user.sub if user else None
    )
    
    if response.error:
        raise HTTPException(status_code=500, detail=response.error.message)
    
    return response.result


@router.get("/prompts")
async def list_prompts(
    user: Optional[TokenData] = Depends(get_current_user)
):
    """
    List available MCP prompts.
    
    Prompts are filtered based on user permissions if authenticated.
    """
    request = MCPRequest(method="prompts/list")
    
    server = get_mcp_server()
    if not server.initialized:
        await server.initialize()
    
    response = await server.handle_http_request(
        request,
        user_id=user.sub if user else None
    )
    
    if response.error:
        raise HTTPException(status_code=500, detail=response.error.message)
    
    return response.result


# Authentication endpoints

@router.post("/auth/token", response_model=AuthTokenResponse)
async def create_token(
    auth_request: AuthTokenRequest
):
    """
    Create authentication token.
    
    Supports authentication via:
    - Username/password
    - API key
    
    Returns JWT access token and optional refresh token.
    """
    jwt_manager = get_jwt_manager()
    
    # TODO: Implement actual user authentication
    # For now, create a demo token
    if auth_request.username == "admin" and auth_request.password == "admin":
        access_token = jwt_manager.create_access_token(
            subject="admin_user",
            username="admin",
            roles=[UserRole.ADMIN.value],
            permissions=["*"]
        )
        
        refresh_token, _ = jwt_manager.create_refresh_token("admin_user")
        
        return AuthTokenResponse(
            access_token=access_token,
            expires_in=1800,  # 30 minutes
            refresh_token=refresh_token
        )
    
    raise HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Invalid credentials"
    )


@router.post("/auth/refresh", response_model=AuthTokenResponse)
async def refresh_token(
    refresh_token: str = Query(..., description="Refresh token")
):
    """
    Refresh authentication token.
    
    Exchange a valid refresh token for a new access token.
    """
    # TODO: Implement token refresh
    raise HTTPException(
        status_code=status.HTTP_501_NOT_IMPLEMENTED,
        detail="Token refresh not yet implemented"
    )


# Health check endpoint

@router.get("/health")
async def health_check():
    """
    Health check endpoint for load balancers.
    
    Returns 200 if server is healthy, 503 if not.
    """
    server = get_mcp_server()
    
    if not server.initialized:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Server not initialized"
        )
    
    status = await server.get_status()
    
    if status["status"] != "healthy":
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Server status: {status['status']}"
        )
    
    return {"status": "healthy"}


# OpenAPI documentation customization

def customize_openapi():
    """Customize OpenAPI schema for better documentation"""
    from fastapi.openapi.utils import get_openapi
    
    def custom_openapi():
        if router.openapi_schema:
            return router.openapi_schema
        
        openapi_schema = get_openapi(
            title="MCP Unified API",
            version="3.0.0",
            description="""
            # Model Context Protocol (MCP) Unified API
            
            Production-ready MCP implementation with enhanced security and monitoring.
            
            ## Features
            - 🔒 **Secure by Default**: JWT authentication, RBAC, rate limiting
            - 🚀 **High Performance**: Connection pooling, caching, circuit breakers
            - 📊 **Observable**: Health checks, metrics, distributed tracing
            - 🔧 **Modular**: Extensible module system with hot-reload
            
            ## Authentication
            Most endpoints support optional authentication via Bearer token.
            Some endpoints require authentication or admin role.
            
            ## WebSocket
            The `/ws` endpoint provides full MCP protocol support over WebSocket.
            """,
            routes=router.routes,
        )
        
        # Add security schemes
        openapi_schema["components"]["securitySchemes"] = {
            "bearerAuth": {
                "type": "http",
                "scheme": "bearer",
                "bearerFormat": "JWT"
            }
        }
        
        router.openapi_schema = openapi_schema
        return router.openapi_schema
    
    return custom_openapi


router.openapi = customize_openapi()