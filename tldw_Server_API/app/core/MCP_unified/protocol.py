"""
MCP Protocol implementation for unified module

Implements JSON-RPC 2.0 with enhanced error handling and request routing.
"""

import uuid
import json
from typing import Dict, Any, Optional, List, Union, Callable, Literal
from datetime import datetime, timezone
from enum import IntEnum
from pydantic import BaseModel, Field, validator
from loguru import logger

from .modules.registry import get_module_registry
from .auth.rbac import get_rbac_policy, Resource, Action
from .auth.rate_limiter import get_rate_limiter


# JSON-RPC 2.0 Error Codes
class ErrorCode(IntEnum):
    """Standard JSON-RPC 2.0 error codes"""
    PARSE_ERROR = -32700
    INVALID_REQUEST = -32600
    METHOD_NOT_FOUND = -32601
    INVALID_PARAMS = -32602
    INTERNAL_ERROR = -32603
    
    # Custom error codes (must be -32000 to -32099)
    AUTHENTICATION_ERROR = -32000
    AUTHORIZATION_ERROR = -32001
    RATE_LIMIT_ERROR = -32002
    MODULE_ERROR = -32003
    TIMEOUT_ERROR = -32004


class MCPRequest(BaseModel):
    """MCP request following JSON-RPC 2.0 specification"""
    jsonrpc: Literal["2.0"] = Field(default="2.0")
    method: str = Field(..., min_length=1, max_length=100)
    params: Optional[Dict[str, Any]] = None
    id: Optional[Union[str, int]] = None
    
    @validator("method")
    def validate_method(cls, v):
        """Validate method name"""
        # Prevent potential injection attacks
        if any(char in v for char in ["'", '"', ';', '--', '/*', '*/']):
            raise ValueError("Invalid characters in method name")
        return v
    
    @validator("params")
    def validate_params(cls, v):
        """Validate and sanitize parameters"""
        if v is not None and not isinstance(v, dict):
            raise ValueError("Params must be a dictionary")
        return v


class MCPError(BaseModel):
    """MCP error structure"""
    code: int
    message: str
    data: Optional[Any] = None


class MCPResponse(BaseModel):
    """MCP response following JSON-RPC 2.0 specification"""
    jsonrpc: Literal["2.0"] = Field(default="2.0")
    result: Optional[Any] = None
    error: Optional[MCPError] = None
    id: Optional[Union[str, int]] = None
    
    @validator("error")
    def validate_error_result(cls, v, values):
        """Ensure either result or error is set, not both"""
        if v is not None and values.get("result") is not None:
            raise ValueError("Response cannot have both result and error")
        return v


class RequestContext:
    """Context for request processing"""
    def __init__(
        self,
        request_id: str,
        user_id: Optional[str] = None,
        client_id: Optional[str] = None,
        session_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ):
        self.request_id = request_id
        self.user_id = user_id
        self.client_id = client_id
        self.session_id = session_id
        self.metadata = metadata or {}
        self.start_time = datetime.now(timezone.utc)
        
        # Add request ID to logger context
        logger.contextualize(request_id=request_id)


class MCPProtocol:
    """
    MCP Protocol handler with enhanced security and error handling.
    
    Features:
    - JSON-RPC 2.0 compliance
    - Request validation and sanitization
    - Authentication and authorization
    - Rate limiting
    - Request routing
    - Error handling with proper codes
    - Request tracing
    """
    
    def __init__(self):
        self.module_registry = get_module_registry()
        self.rbac_policy = get_rbac_policy()
        self.rate_limiter = get_rate_limiter()
        self.protocol_version = "2024-11-05"
        
        # Method handlers
        self.handlers: Dict[str, Callable] = {
            "initialize": self._handle_initialize,
            "ping": self._handle_ping,
            "tools/list": self._handle_tools_list,
            "tools/call": self._handle_tools_call,
            "resources/list": self._handle_resources_list,
            "resources/read": self._handle_resources_read,
            "prompts/list": self._handle_prompts_list,
            "prompts/get": self._handle_prompts_get,
            "modules/list": self._handle_modules_list,
            "modules/health": self._handle_modules_health,
        }
        
        logger.info("MCP Protocol handler initialized")
    
    async def process_request(
        self,
        request: Union[Dict[str, Any], MCPRequest],
        context: Optional[RequestContext] = None
    ) -> MCPResponse:
        """
        Process an MCP request and return response.
        
        Args:
            request: MCP request (dict or MCPRequest object)
            context: Request context with user/session info
        
        Returns:
            MCP response
        """
        # Parse request if dict
        if isinstance(request, dict):
            try:
                request = MCPRequest(**request)
            except Exception as e:
                return self._error_response(
                    ErrorCode.INVALID_REQUEST,
                    f"Invalid request format: {str(e)}",
                    request.get("id") if isinstance(request, dict) else None
                )
        
        # Create context if not provided
        if context is None:
            context = RequestContext(
                request_id=str(uuid.uuid4()),
                client_id="unknown"
            )
        
        # Log request
        logger.info(
            f"MCP request: method={request.method}, "
            f"user={context.user_id}, client={context.client_id}",
            extra={"audit": True}
        )
        
        try:
            # Check rate limit
            if context.user_id:
                await self.rate_limiter.check_rate_limit(f"user:{context.user_id}")
            elif context.client_id:
                await self.rate_limiter.check_rate_limit(f"client:{context.client_id}")
            
            # Validate JSON-RPC version
            if request.jsonrpc != "2.0":
                return self._error_response(
                    ErrorCode.INVALID_REQUEST,
                    "Invalid JSON-RPC version",
                    request.id
                )
            
            # Find handler
            handler = self.handlers.get(request.method)
            if not handler:
                return self._error_response(
                    ErrorCode.METHOD_NOT_FOUND,
                    f"Method not found: {request.method}",
                    request.id
                )
            
            # Check authorization
            if not await self._check_authorization(request.method, context):
                return self._error_response(
                    ErrorCode.AUTHORIZATION_ERROR,
                    "Insufficient permissions",
                    request.id
                )
            
            # Execute handler
            result = await handler(request.params or {}, context)
            
            # Log success
            elapsed = (datetime.now(timezone.utc) - context.start_time).total_seconds()
            logger.info(
                f"MCP request completed: method={request.method}, "
                f"elapsed={elapsed:.3f}s",
                extra={"audit": True}
            )
            
            # Return success response
            return MCPResponse(
                result=result,
                id=request.id
            )
            
        except Exception as e:
            # Log error
            logger.error(
                f"MCP request failed: method={request.method}, error={str(e)}",
                extra={"audit": True}
            )
            
            # Return error response
            return self._error_response(
                ErrorCode.INTERNAL_ERROR,
                str(e),
                request.id
            )
    
    def _error_response(
        self,
        code: ErrorCode,
        message: str,
        request_id: Optional[Union[str, int]] = None,
        data: Optional[Any] = None
    ) -> MCPResponse:
        """Create an error response"""
        return MCPResponse(
            error=MCPError(
                code=code,
                message=message,
                data=data
            ),
            id=request_id
        )
    
    async def _check_authorization(
        self,
        method: str,
        context: RequestContext
    ) -> bool:
        """Check if user is authorized for method"""
        # Public methods that don't require auth
        public_methods = ["initialize", "ping"]
        if method in public_methods:
            return True
        
        # No user context means no auth
        if not context.user_id:
            return False
        
        # Map methods to resources and actions
        method_permissions = {
            "tools/list": (Resource.TOOL, Action.READ),
            "tools/call": (Resource.TOOL, Action.EXECUTE),
            "resources/list": (Resource.RESOURCE, Action.READ),
            "resources/read": (Resource.RESOURCE, Action.READ),
            "prompts/list": (Resource.PROMPT, Action.READ),
            "prompts/get": (Resource.PROMPT, Action.READ),
            "modules/list": (Resource.MODULE, Action.READ),
            "modules/health": (Resource.MODULE, Action.READ),
        }
        
        if method in method_permissions:
            resource, action = method_permissions[method]
            return self.rbac_policy.check_permission(
                context.user_id,
                resource,
                action
            )
        
        # Unknown method - deny by default
        return False
    
    # Protocol method handlers
    
    async def _handle_initialize(
        self,
        params: Dict[str, Any],
        context: RequestContext
    ) -> Dict[str, Any]:
        """Handle initialize request"""
        client_info = params.get("clientInfo", {})
        
        logger.info(f"Client initializing: {client_info}")
        
        # Get server capabilities
        modules = await self.module_registry.get_all_modules()
        
        capabilities = {
            "tools": {"available": bool(modules)},
            "resources": {"available": bool(modules)},
            "prompts": {"available": bool(modules)}
        }
        
        return {
            "protocolVersion": self.protocol_version,
            "capabilities": capabilities,
            "serverInfo": {
                "name": "tldw-mcp-unified",
                "version": "3.0.0"
            }
        }
    
    async def _handle_ping(
        self,
        params: Dict[str, Any],
        context: RequestContext
    ) -> Dict[str, Any]:
        """Handle ping request"""
        return {"pong": True, "timestamp": datetime.now(timezone.utc).isoformat()}
    
    async def _handle_tools_list(
        self,
        params: Dict[str, Any],
        context: RequestContext
    ) -> Dict[str, Any]:
        """List available tools"""
        tools = []
        modules = await self.module_registry.get_all_modules()
        
        for module_id, module in modules.items():
            try:
                module_tools = await module.get_tools()
                
                # Add module context to each tool
                for tool in module_tools:
                    tool_copy = tool.copy()
                    tool_copy["module"] = module_id
                    
                    # Check if user can execute this tool
                    if context.user_id:
                        can_execute = self.rbac_policy.check_permission(
                            context.user_id,
                            Resource.TOOL,
                            Action.EXECUTE,
                            tool["name"]
                        )
                        tool_copy["canExecute"] = can_execute
                    
                    tools.append(tool_copy)
                    
            except Exception as e:
                logger.error(f"Error getting tools from module {module_id}: {e}")
        
        return {"tools": tools}
    
    async def _handle_tools_call(
        self,
        params: Dict[str, Any],
        context: RequestContext
    ) -> Dict[str, Any]:
        """Execute a tool"""
        tool_name = params.get("name")
        tool_args = params.get("arguments", {})
        
        if not tool_name:
            raise ValueError("Tool name is required")
        
        # Sanitize tool name
        if any(char in tool_name for char in ["'", '"', ';', '--']):
            raise ValueError("Invalid tool name")
        
        # Find module for tool
        module = await self.module_registry.find_module_for_tool(tool_name)
        if not module:
            raise ValueError(f"Tool not found: {tool_name}")
        
        # Check specific tool permission
        if context.user_id:
            if not self.rbac_policy.check_permission(
                context.user_id,
                Resource.TOOL,
                Action.EXECUTE,
                tool_name
            ):
                raise PermissionError(f"Permission denied for tool: {tool_name}")
        
        # Execute tool with circuit breaker
        try:
            result = await module.execute_with_circuit_breaker(
                module.execute_tool,
                tool_name,
                tool_args
            )
            
            # Format result
            if isinstance(result, str):
                content = [{"type": "text", "text": result}]
            elif isinstance(result, list):
                content = result
            else:
                content = [{"type": "text", "text": str(result)}]
            
            return {"content": content}
            
        except Exception as e:
            logger.error(f"Tool execution failed: {tool_name} - {e}")
            raise
    
    async def _handle_resources_list(
        self,
        params: Dict[str, Any],
        context: RequestContext
    ) -> Dict[str, Any]:
        """List available resources"""
        resources = []
        modules = await self.module_registry.get_all_modules()
        
        for module_id, module in modules.items():
            try:
                module_resources = await module.get_resources()
                
                # Add module context
                for resource in module_resources:
                    resource_copy = resource.copy()
                    resource_copy["module"] = module_id
                    resources.append(resource_copy)
                    
            except Exception as e:
                logger.error(f"Error getting resources from module {module_id}: {e}")
        
        return {"resources": resources}
    
    async def _handle_resources_read(
        self,
        params: Dict[str, Any],
        context: RequestContext
    ) -> Dict[str, Any]:
        """Read a resource"""
        uri = params.get("uri")
        if not uri:
            raise ValueError("Resource URI is required")
        
        # Find module for resource
        module = await self.module_registry.find_module_for_resource(uri)
        if not module:
            raise ValueError(f"Resource not found: {uri}")
        
        # Read resource
        content = await module.read_resource(uri)
        
        return {"contents": [content]}
    
    async def _handle_prompts_list(
        self,
        params: Dict[str, Any],
        context: RequestContext
    ) -> Dict[str, Any]:
        """List available prompts"""
        prompts = []
        modules = await self.module_registry.get_all_modules()
        
        for module_id, module in modules.items():
            try:
                module_prompts = await module.get_prompts()
                
                # Add module context
                for prompt in module_prompts:
                    prompt_copy = prompt.copy()
                    prompt_copy["module"] = module_id
                    prompts.append(prompt_copy)
                    
            except Exception as e:
                logger.error(f"Error getting prompts from module {module_id}: {e}")
        
        return {"prompts": prompts}
    
    async def _handle_prompts_get(
        self,
        params: Dict[str, Any],
        context: RequestContext
    ) -> Dict[str, Any]:
        """Get a specific prompt"""
        name = params.get("name")
        if not name:
            raise ValueError("Prompt name is required")
        
        arguments = params.get("arguments", {})
        
        # Find module for prompt
        module = await self.module_registry.find_module_for_prompt(name)
        if not module:
            raise ValueError(f"Prompt not found: {name}")
        
        # Get prompt
        prompt = await module.get_prompt(name, arguments)
        
        return prompt
    
    async def _handle_modules_list(
        self,
        params: Dict[str, Any],
        context: RequestContext
    ) -> Dict[str, Any]:
        """List registered modules"""
        registrations = await self.module_registry.list_registrations()
        return {"modules": registrations}
    
    async def _handle_modules_health(
        self,
        params: Dict[str, Any],
        context: RequestContext
    ) -> Dict[str, Any]:
        """Get module health status"""
        health_results = await self.module_registry.check_all_health()
        
        # Convert to serializable format
        health_data = {}
        for module_id, health in health_results.items():
            health_data[module_id] = {
                "status": health.status.value,
                "message": health.message,
                "checks": health.checks,
                "last_check": health.last_check.isoformat()
            }
        
        return {"health": health_data}


# Convenience function
async def process_mcp_request(
    request: Union[Dict[str, Any], MCPRequest],
    context: Optional[RequestContext] = None
) -> MCPResponse:
    """Process an MCP request"""
    protocol = MCPProtocol()
    return await protocol.process_request(request, context)