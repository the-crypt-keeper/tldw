"""
MCP Protocol implementation for tldw following the MCP specification
"""

from typing import Dict, Any, Optional, List, Callable
from datetime import datetime
import uuid
from loguru import logger

from ..schemas import MCPRequest, MCPResponse, MCPError


# MCP Protocol version
PROTOCOL_VERSION = "2024-11-05"

# Error codes following JSON-RPC 2.0
ERROR_PARSE = -32700
ERROR_INVALID_REQUEST = -32600
ERROR_METHOD_NOT_FOUND = -32601
ERROR_INVALID_PARAMS = -32602
ERROR_INTERNAL = -32603


async def handle_initialize(params: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
    """Handle MCP initialize request"""
    logger.info(f"MCP initialize: {params.get('clientInfo')}")
    
    return {
        "protocolVersion": PROTOCOL_VERSION,
        "capabilities": {
            "tools": {},
            "resources": {},
            "prompts": {}
        },
        "serverInfo": {
            "name": "tldw-mcp-server",
            "version": "2.0.0"
        }
    }


async def handle_tools_list(params: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
    """List available tools from all modules"""
    module_registry = context.get("module_registry")
    if not module_registry:
        return {"tools": []}
    
    tools = []
    modules = await module_registry.get_all_modules()
    
    for module_id, module in modules.items():
        try:
            module_tools = await module.get_tools()
            # Add module prefix to tool names for namespacing
            for tool in module_tools:
                tool_copy = tool.copy()
                tool_copy["name"] = f"{module_id}.{tool['name']}"
                tools.append(tool_copy)
        except Exception as e:
            logger.error(f"Error getting tools from module {module_id}: {e}")
    
    logger.info(f"Listed {len(tools)} tools")
    return {"tools": tools}


async def handle_tools_call(params: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
    """Execute a tool call"""
    tool_name = params.get("name")
    tool_params = params.get("arguments", {})
    
    if not tool_name:
        raise ValueError("Tool name is required")
    
    module_registry = context.get("module_registry")
    if not module_registry:
        raise ValueError("Module registry not available")
    
    # Parse module prefix if present
    if "." in tool_name:
        module_id, actual_tool_name = tool_name.split(".", 1)
        module = await module_registry.get_module(module_id)
        
        if not module:
            # Try finding without prefix
            module = await module_registry.find_module_for_tool(tool_name)
            actual_tool_name = tool_name
    else:
        # Search all modules for the tool
        module = await module_registry.find_module_for_tool(tool_name)
        actual_tool_name = tool_name
    
    if not module:
        raise ValueError(f"Tool not found: {tool_name}")
    
    # Check permissions if auth is enabled
    user = context.get("user")
    if user and not await check_tool_permission(tool_name, user, context):
        raise PermissionError(f"Access denied to tool: {tool_name}")
    
    # Execute tool
    try:
        logger.info(f"Executing tool: {tool_name}")
        result = await module.execute_tool(actual_tool_name, tool_params)
        
        # Log execution if audit service is available
        await log_tool_execution(tool_name, user, "success", context)
        
        return {"content": [{"type": "text", "text": str(result)}] if not isinstance(result, list) else result}
    except Exception as e:
        await log_tool_execution(tool_name, user, "failure", context, error=str(e))
        raise


async def handle_resources_list(params: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
    """List available resources"""
    module_registry = context.get("module_registry")
    if not module_registry:
        return {"resources": []}
    
    resources = []
    modules = await module_registry.get_all_modules()
    
    for module_id, module in modules.items():
        try:
            module_resources = await module.get_resources()
            resources.extend(module_resources)
        except Exception as e:
            logger.error(f"Error getting resources from module {module_id}: {e}")
    
    return {"resources": resources}


async def handle_resources_read(params: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
    """Read a resource"""
    uri = params.get("uri")
    if not uri:
        raise ValueError("Resource URI is required")
    
    module_registry = context.get("module_registry")
    if not module_registry:
        raise ValueError("Module registry not available")
    
    # Find module that provides this resource
    module = await module_registry.find_module_for_resource(uri)
    if not module:
        raise ValueError(f"Resource not found: {uri}")
    
    content = await module.read_resource(uri)
    return {"contents": [content]}


async def handle_prompts_list(params: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
    """List available prompts"""
    module_registry = context.get("module_registry")
    if not module_registry:
        return {"prompts": []}
    
    prompts = []
    modules = await module_registry.get_all_modules()
    
    for module_id, module in modules.items():
        try:
            module_prompts = await module.get_prompts()
            prompts.extend(module_prompts)
        except Exception as e:
            logger.error(f"Error getting prompts from module {module_id}: {e}")
    
    return {"prompts": prompts}


async def handle_prompts_get(params: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
    """Get a specific prompt"""
    name = params.get("name")
    if not name:
        raise ValueError("Prompt name is required")
    
    module_registry = context.get("module_registry")
    if not module_registry:
        raise ValueError("Module registry not available")
    
    # Find module that provides this prompt
    module = await module_registry.find_module_for_prompt(name)
    if not module:
        raise ValueError(f"Prompt not found: {name}")
    
    prompt = await module.get_prompt(name, params.get("arguments", {}))
    return prompt


# Method handlers registry
METHOD_HANDLERS: Dict[str, Callable] = {
    "initialize": handle_initialize,
    "tools/list": handle_tools_list,
    "tools/call": handle_tools_call,
    "resources/list": handle_resources_list,
    "resources/read": handle_resources_read,
    "prompts/list": handle_prompts_list,
    "prompts/get": handle_prompts_get,
}


async def process_mcp_request(request: MCPRequest, context: Dict[str, Any]) -> MCPResponse:
    """Process an MCP request and return response"""
    request_id = request.id or str(uuid.uuid4())
    
    # Add request ID to context for tracing
    context["request_id"] = request_id
    
    logger.info(f"MCP request: {request.method} (id: {request_id})")
    
    try:
        # Validate request
        if request.jsonrpc != "2.0":
            return MCPResponse(
                error={"code": ERROR_INVALID_REQUEST, "message": "Invalid JSON-RPC version"},
                id=request_id
            )
        
        # Find handler
        handler = METHOD_HANDLERS.get(request.method)
        if not handler:
            return MCPResponse(
                error={
                    "code": ERROR_METHOD_NOT_FOUND,
                    "message": f"Method not found: {request.method}"
                },
                id=request_id
            )
        
        # Execute handler
        result = await handler(request.params or {}, context)
        
        logger.info(f"MCP response: {request.method} (id: {request_id}) - success")
        
        return MCPResponse(result=result, id=request_id)
        
    except ValueError as e:
        logger.error(f"MCP error: {request.method} (id: {request_id}) - {str(e)}")
        return MCPResponse(
            error={
                "code": ERROR_INVALID_PARAMS,
                "message": str(e)
            },
            id=request_id
        )
    except PermissionError as e:
        logger.error(f"MCP permission error: {request.method} (id: {request_id}) - {str(e)}")
        return MCPResponse(
            error={
                "code": ERROR_INTERNAL,
                "message": str(e)
            },
            id=request_id
        )
    except Exception as e:
        logger.exception(f"MCP internal error: {request.method} (id: {request_id})")
        return MCPResponse(
            error={
                "code": ERROR_INTERNAL,
                "message": "Internal server error"
            },
            id=request_id
        )


async def check_tool_permission(tool_name: str, user: Dict[str, Any], context: Dict[str, Any]) -> bool:
    """Check if user has permission to use a tool"""
    if not user:
        # Allow anonymous access for basic tools (configurable)
        basic_tools = ["search_media", "list_tools", "get_timestamp"]
        return tool_name in basic_tools
    
    # Import RBAC here to avoid circular import
    from ..auth.rbac import rbac_policy, ResourceType, Action
    from ..schemas import MCPUser, UserRole
    
    # Convert user dict to MCPUser if needed
    if isinstance(user, dict):
        mcp_user = MCPUser(
            id=user.get("id", "unknown"),
            username=user.get("username", "unknown"),
            roles=[UserRole(r) for r in user.get("roles", ["user"])],
            department=user.get("department"),
            permissions=user.get("permissions", [])
        )
    else:
        mcp_user = user
    
    # Check RBAC permission
    return rbac_policy.check_permission(
        mcp_user,
        ResourceType.TOOL,
        tool_name,
        Action.EXECUTE
    )


async def log_tool_execution(
    tool_name: str,
    user: Optional[Dict[str, Any]],
    status: str,
    context: Dict[str, Any],
    error: Optional[str] = None
):
    """Log tool execution for audit (if audit service is available)"""
    # In tldw, we could log to the database or use existing logging
    logger.info(f"Tool execution: {tool_name} by {user.get('username') if user else 'anonymous'} - {status}")
    if error:
        logger.error(f"Tool execution error: {error}")