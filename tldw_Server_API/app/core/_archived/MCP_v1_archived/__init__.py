# MCP (Model Context Protocol) Server Implementation
"""
Model Context Protocol (MCP) server implementation for tldw_server.

This module provides:
- MCP server for handling client connections
- Tool registration and execution
- Context management for LLM interactions
- WebSocket and HTTP transport support

Based on MCP specification v1.0
"""

from .mcp_server import MCPServer, get_mcp_server
from .mcp_protocol import (
    MCPMessage,
    MCPRequest,
    MCPResponse,
    MCPError,
    MCPTool,
    MCPContext
)
from .mcp_tools import ToolRegistry, mcp_tool
from .mcp_context import ContextManager

__all__ = [
    'MCPServer',
    'get_mcp_server',
    'MCPMessage',
    'MCPRequest', 
    'MCPResponse',
    'MCPError',
    'MCPTool',
    'MCPContext',
    'ToolRegistry',
    'ContextManager',
    'mcp_tool'
]

__version__ = "0.1.0"