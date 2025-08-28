"""
Unified MCP (Model Context Protocol) implementation for tldw_server

This module combines the best features of MCP v1 and v2 with enhanced security,
performance, and production-readiness.
"""

from .server import MCPServer, get_mcp_server
from .protocol import MCPProtocol, MCPRequest, MCPResponse
from .modules.base import BaseModule
from .modules.registry import ModuleRegistry, get_module_registry
from .auth.jwt_manager import JWTManager
from .auth.rbac import RBACPolicy, UserRole, Permission
from .config import get_config

__version__ = "3.0.0"

__all__ = [
    "MCPServer",
    "get_mcp_server",
    "MCPProtocol",
    "MCPRequest",
    "MCPResponse",
    "BaseModule",
    "ModuleRegistry",
    "get_module_registry",
    "JWTManager",
    "RBACPolicy",
    "UserRole",
    "Permission",
    "get_config",
]