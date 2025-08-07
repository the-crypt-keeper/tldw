# mcp_tools.py - MCP Tool Registry and Management
"""
Tool registration and execution system for MCP server.

This module provides:
- Tool registry for managing available tools
- Tool validation and execution
- Built-in tools for common operations
- Plugin system for custom tools
"""

import asyncio
import inspect
import time
from typing import Dict, Any, Callable, Optional, List, Union, Type
from functools import wraps
import logging

from .mcp_protocol import (
    MCPTool,
    MCPToolParameter,
    MCPToolExecutionRequest,
    MCPToolExecutionResult,
    MCPError,
    MCPErrorCode
)

logger = logging.getLogger(__name__)


class ToolExecutionError(Exception):
    """Raised when tool execution fails"""
    pass


class ToolRegistry:
    """Registry for MCP tools"""
    
    def __init__(self):
        self._tools: Dict[str, MCPTool] = {}
        self._handlers: Dict[str, Callable] = {}
        self._middleware: List[Callable] = []
        
        # Register built-in tools
        self._register_builtin_tools()
    
    def register(
        self,
        name: str,
        handler: Callable,
        description: str,
        parameters: Optional[List[MCPToolParameter]] = None,
        returns: Optional[Dict[str, Any]] = None,
        tags: Optional[List[str]] = None,
        examples: Optional[List[Dict[str, Any]]] = None
    ):
        """Register a tool with the registry"""
        if name in self._tools:
            raise ValueError(f"Tool '{name}' already registered")
        
        # Validate handler
        if not callable(handler):
            raise ValueError(f"Handler for tool '{name}' must be callable")
        
        # Create tool definition
        tool = MCPTool(
            name=name,
            description=description,
            parameters=parameters or [],
            returns=returns,
            tags=tags or [],
            examples=examples or []
        )
        
        self._tools[name] = tool
        self._handlers[name] = handler
        
        logger.info(f"Registered tool: {name}")
    
    def register_decorator(
        self,
        name: Optional[str] = None,
        description: Optional[str] = None,
        **kwargs
    ):
        """Decorator for registering tools"""
        def decorator(func: Callable):
            tool_name = name or func.__name__
            tool_desc = description or func.__doc__ or "No description"
            
            # Extract parameters from function signature
            sig = inspect.signature(func)
            parameters = []
            
            for param_name, param in sig.parameters.items():
                if param_name in ['self', 'cls']:
                    continue
                    
                param_type = "string"  # Default type
                if param.annotation != inspect.Parameter.empty:
                    # Try to infer type from annotation
                    if param.annotation == int:
                        param_type = "integer"
                    elif param.annotation == float:
                        param_type = "number"
                    elif param.annotation == bool:
                        param_type = "boolean"
                    elif param.annotation == list:
                        param_type = "array"
                    elif param.annotation == dict:
                        param_type = "object"
                
                parameters.append(MCPToolParameter(
                    name=param_name,
                    type=param_type,
                    required=param.default == inspect.Parameter.empty
                ))
            
            self.register(
                name=tool_name,
                handler=func,
                description=tool_desc,
                parameters=parameters,
                **kwargs
            )
            
            return func
        
        return decorator
    
    def unregister(self, name: str):
        """Unregister a tool"""
        if name not in self._tools:
            raise ValueError(f"Tool '{name}' not found")
        
        del self._tools[name]
        del self._handlers[name]
        
        logger.info(f"Unregistered tool: {name}")
    
    def get_tool(self, name: str) -> Optional[MCPTool]:
        """Get tool definition by name"""
        return self._tools.get(name)
    
    def list_tools(self, tags: Optional[List[str]] = None) -> List[MCPTool]:
        """List all registered tools, optionally filtered by tags"""
        tools = list(self._tools.values())
        
        if tags:
            tools = [
                tool for tool in tools
                if any(tag in tool.tags for tag in tags)
            ]
        
        return tools
    
    async def execute(
        self,
        request: MCPToolExecutionRequest,
        context: Optional[Dict[str, Any]] = None
    ) -> MCPToolExecutionResult:
        """Execute a tool"""
        start_time = time.time()
        
        # Check if tool exists
        if request.tool_name not in self._tools:
            return MCPToolExecutionResult(
                tool_name=request.tool_name,
                success=False,
                error=f"Tool '{request.tool_name}' not found",
                execution_time=time.time() - start_time
            )
        
        tool = self._tools[request.tool_name]
        handler = self._handlers[request.tool_name]
        
        try:
            # Validate arguments
            self._validate_arguments(tool, request.arguments)
            
            # Apply middleware
            for middleware in self._middleware:
                request, context = await self._apply_middleware(
                    middleware, request, context
                )
            
            # Execute handler
            if asyncio.iscoroutinefunction(handler):
                result = await handler(**request.arguments, _context=context)
            else:
                result = handler(**request.arguments, _context=context)
            
            return MCPToolExecutionResult(
                tool_name=request.tool_name,
                success=True,
                result=result,
                execution_time=time.time() - start_time
            )
            
        except Exception as e:
            logger.error(f"Tool execution failed: {request.tool_name}", exc_info=True)
            return MCPToolExecutionResult(
                tool_name=request.tool_name,
                success=False,
                error=str(e),
                execution_time=time.time() - start_time
            )
    
    def add_middleware(self, middleware: Callable):
        """Add middleware for tool execution"""
        if not callable(middleware):
            raise ValueError("Middleware must be callable")
        self._middleware.append(middleware)
    
    def _validate_arguments(self, tool: MCPTool, arguments: Dict[str, Any]):
        """Validate tool arguments"""
        # Check required parameters
        for param in tool.parameters:
            if param.required and param.name not in arguments:
                raise ValueError(f"Missing required parameter: {param.name}")
            
            # Validate enum values
            if param.enum and arguments.get(param.name) not in param.enum:
                raise ValueError(
                    f"Invalid value for {param.name}. "
                    f"Must be one of: {param.enum}"
                )
    
    async def _apply_middleware(
        self,
        middleware: Callable,
        request: MCPToolExecutionRequest,
        context: Optional[Dict[str, Any]]
    ) -> tuple:
        """Apply middleware to request"""
        if asyncio.iscoroutinefunction(middleware):
            return await middleware(request, context)
        else:
            return middleware(request, context)
    
    def _register_builtin_tools(self):
        """Register built-in MCP tools"""
        
        @self.register_decorator(
            name="echo",
            description="Echo back the provided message",
            tags=["builtin", "test"]
        )
        async def echo(message: str, _context=None) -> str:
            """Echo the provided message"""
            return message
        
        @self.register_decorator(
            name="get_timestamp",
            description="Get current timestamp",
            tags=["builtin", "utility"]
        )
        async def get_timestamp(_context=None) -> dict:
            """Get current timestamp in various formats"""
            now = time.time()
            return {
                "unix": now,
                "iso": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(now)),
                "readable": time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime(now))
            }
        
        @self.register_decorator(
            name="list_available_tools",
            description="List all available tools",
            tags=["builtin", "discovery"]
        )
        async def list_available_tools(tag: str = None, _context=None) -> list:
            """List available tools with optional tag filter"""
            tools = self.list_tools(tags=[tag] if tag else None)
            return [
                {
                    "name": tool.name,
                    "description": tool.description,
                    "tags": tool.tags,
                    "parameters": [p.dict() for p in tool.parameters]
                }
                for tool in tools
            ]


# Global tool registry instance
tool_registry = ToolRegistry()


# Convenience decorator for registering tools
def mcp_tool(name: Optional[str] = None, **kwargs):
    """Decorator for registering MCP tools"""
    return tool_registry.register_decorator(name=name, **kwargs)


# Example custom tool
@mcp_tool(
    description="Calculate the sum of two numbers",
    tags=["math", "example"],
    examples=[
        {"arguments": {"a": 5, "b": 3}, "result": 8},
        {"arguments": {"a": -2, "b": 10}, "result": 8}
    ]
)
async def add_numbers(a: float, b: float, _context=None) -> float:
    """Add two numbers together"""
    return a + b


__all__ = [
    'ToolRegistry',
    'ToolExecutionError',
    'tool_registry',
    'mcp_tool'
]