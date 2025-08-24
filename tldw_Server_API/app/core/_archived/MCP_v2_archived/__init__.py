"""
MCP v2 - Enterprise-grade Model Context Protocol implementation for tldw

This module provides a modular, extensible MCP server with:
- Department/feature-based modules
- Enterprise security (JWT, RBAC, audit logging)
- Full MCP specification compliance
- Production-ready features
"""

from .core.server import MCPServer, get_mcp_server
from .core.protocol import process_mcp_request
from .core.registry import ModuleRegistry, get_module_registry
from .modules.base import BaseModule
from .schemas import (
    MCPRequest,
    MCPResponse,
    MCPError,
    ModuleConfig,
    ModuleRegistration
)

__all__ = [
    'MCPServer',
    'get_mcp_server',
    'process_mcp_request',
    'ModuleRegistry',
    'get_module_registry',
    'BaseModule',
    'MCPRequest',
    'MCPResponse',
    'MCPError',
    'ModuleConfig',
    'ModuleRegistration'
]