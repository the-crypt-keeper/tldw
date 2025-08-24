"""
MCP v2 Schemas - Pydantic models for MCP protocol and modules
"""

from typing import Dict, Any, Optional, List, Union
from datetime import datetime
from pydantic import BaseModel, Field
from enum import Enum


# MCP Protocol Schemas

class MCPRequest(BaseModel):
    """MCP request following JSON-RPC 2.0"""
    jsonrpc: str = "2.0"
    method: str
    params: Optional[Dict[str, Any]] = None
    id: Optional[Union[str, int]] = None


class MCPResponse(BaseModel):
    """MCP response following JSON-RPC 2.0"""
    jsonrpc: str = "2.0"
    result: Optional[Any] = None
    error: Optional[Dict[str, Any]] = None
    id: Optional[Union[str, int]] = None


class MCPError(BaseModel):
    """MCP error structure"""
    code: int
    message: str
    data: Optional[Any] = None


# Module Configuration Schemas

class ModuleCapability(str, Enum):
    """Module capabilities"""
    TOOLS = "tools"
    RESOURCES = "resources"
    PROMPTS = "prompts"


class ModuleConfig(BaseModel):
    """Configuration for a module"""
    name: str
    description: Optional[str] = None
    version: str = "1.0.0"
    department: str  # In tldw context, this could be feature area (media, rag, chat, etc.)
    enabled: bool = True
    capabilities: List[ModuleCapability] = Field(default_factory=lambda: [ModuleCapability.TOOLS])
    settings: Dict[str, Any] = Field(default_factory=dict)


class ModuleRegistration(BaseModel):
    """Module registration information"""
    module_id: str
    name: str
    version: str
    department: str
    capabilities: List[str]
    status: str = "pending"  # pending, active, inactive, error
    registered_at: datetime = Field(default_factory=datetime.utcnow)
    health_check_url: Optional[str] = None
    module_metadata: Dict[str, Any] = Field(default_factory=dict)


# Tool Schemas

class ToolParameter(BaseModel):
    """Tool parameter definition"""
    name: str
    type: str  # string, number, boolean, object, array
    description: Optional[str] = None
    required: bool = False
    default: Optional[Any] = None
    enum: Optional[List[Any]] = None


class ToolDefinition(BaseModel):
    """Tool definition for MCP"""
    name: str
    description: str
    inputSchema: Dict[str, Any]  # JSON Schema format
    metadata: Optional[Dict[str, Any]] = None


# Resource Schemas

class ResourceDefinition(BaseModel):
    """Resource definition for MCP"""
    uri: str
    name: str
    description: Optional[str] = None
    mimeType: str = "application/json"
    metadata: Optional[Dict[str, Any]] = None


# Prompt Schemas

class PromptArgument(BaseModel):
    """Prompt argument definition"""
    name: str
    description: Optional[str] = None
    required: bool = False


class PromptDefinition(BaseModel):
    """Prompt definition for MCP"""
    name: str
    description: Optional[str] = None
    arguments: List[PromptArgument] = Field(default_factory=list)


# User/Auth Schemas (adapted for tldw)

class UserRole(str, Enum):
    """User roles in the system"""
    ADMIN = "admin"
    USER = "user"
    API_CLIENT = "api_client"
    GUEST = "guest"


class MCPUser(BaseModel):
    """User context for MCP operations"""
    id: str
    username: str
    roles: List[UserRole] = Field(default_factory=lambda: [UserRole.USER])
    department: Optional[str] = None  # For tldw, this could be feature access level
    permissions: List[str] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)