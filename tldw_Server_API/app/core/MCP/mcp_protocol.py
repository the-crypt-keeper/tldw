# mcp_protocol.py - MCP Protocol Definitions
"""
Model Context Protocol (MCP) message definitions and protocol structures.

This module defines the core protocol messages, requests, responses, and
data structures used in MCP communication.
"""

from enum import Enum
from typing import Dict, Any, Optional, List, Union
from datetime import datetime
from pydantic import BaseModel, Field, validator
import uuid


class MCPMessageType(str, Enum):
    """MCP message types"""
    # Connection management
    CONNECT = "connect"
    DISCONNECT = "disconnect"
    PING = "ping"
    PONG = "pong"
    
    # Tool operations
    LIST_TOOLS = "list_tools"
    EXECUTE_TOOL = "execute_tool"
    TOOL_RESULT = "tool_result"
    
    # Context operations
    GET_CONTEXT = "get_context"
    UPDATE_CONTEXT = "update_context"
    CLEAR_CONTEXT = "clear_context"
    CONTEXT_UPDATED = "context_updated"
    
    # Resource operations
    LIST_RESOURCES = "list_resources"
    GET_RESOURCE = "get_resource"
    RESOURCE_UPDATED = "resource_updated"
    
    # Error handling
    ERROR = "error"
    
    # System
    CAPABILITIES = "capabilities"
    STATUS = "status"


class MCPErrorCode(str, Enum):
    """Standard MCP error codes"""
    INVALID_REQUEST = "invalid_request"
    METHOD_NOT_FOUND = "method_not_found"
    INVALID_PARAMS = "invalid_params"
    INTERNAL_ERROR = "internal_error"
    TOOL_NOT_FOUND = "tool_not_found"
    TOOL_EXECUTION_ERROR = "tool_execution_error"
    UNAUTHORIZED = "unauthorized"
    RATE_LIMITED = "rate_limited"
    CONTEXT_TOO_LARGE = "context_too_large"
    RESOURCE_NOT_FOUND = "resource_not_found"


class MCPMessage(BaseModel):
    """Base MCP message structure"""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    type: MCPMessageType
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    version: str = "1.0"
    
    class Config:
        use_enum_values = True


class MCPRequest(MCPMessage):
    """MCP request message"""
    method: str
    params: Optional[Dict[str, Any]] = None
    
    @validator('method')
    def validate_method(cls, v):
        if not v:
            raise ValueError("Method cannot be empty")
        return v


class MCPResponse(MCPMessage):
    """MCP response message"""
    request_id: str
    result: Optional[Any] = None
    error: Optional['MCPError'] = None
    
    @validator('error')
    def validate_error_result(cls, v, values):
        if v and values.get('result'):
            raise ValueError("Response cannot have both result and error")
        return v


class MCPError(BaseModel):
    """MCP error structure"""
    code: MCPErrorCode
    message: str
    data: Optional[Dict[str, Any]] = None
    
    class Config:
        use_enum_values = True


class MCPToolParameter(BaseModel):
    """Tool parameter definition"""
    name: str
    type: str  # JSON Schema type
    description: Optional[str] = None
    required: bool = True
    default: Optional[Any] = None
    enum: Optional[List[Any]] = None


class MCPTool(BaseModel):
    """MCP tool definition"""
    name: str
    description: str
    parameters: List[MCPToolParameter] = []
    returns: Optional[Dict[str, Any]] = None  # JSON Schema
    examples: Optional[List[Dict[str, Any]]] = None
    tags: List[str] = []
    
    @validator('name')
    def validate_name(cls, v):
        if not v or not v.replace('_', '').replace('-', '').isalnum():
            raise ValueError("Tool name must be alphanumeric with underscores/hyphens")
        return v


class MCPContext(BaseModel):
    """MCP context structure"""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    name: Optional[str] = None
    content: Dict[str, Any] = {}
    metadata: Dict[str, Any] = {}
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    expires_at: Optional[datetime] = None
    
    def update(self, updates: Dict[str, Any]):
        """Update context content"""
        self.content.update(updates)
        self.updated_at = datetime.utcnow()
    
    def clear(self):
        """Clear context content"""
        self.content = {}
        self.updated_at = datetime.utcnow()
    
    def is_expired(self) -> bool:
        """Check if context has expired"""
        if self.expires_at:
            return datetime.utcnow() > self.expires_at
        return False


class MCPResource(BaseModel):
    """MCP resource definition"""
    uri: str
    name: str
    type: str  # mime type
    description: Optional[str] = None
    metadata: Dict[str, Any] = {}
    
    @validator('uri')
    def validate_uri(cls, v):
        if not v:
            raise ValueError("Resource URI cannot be empty")
        return v


class MCPCapabilities(BaseModel):
    """Server capabilities"""
    tools: bool = True
    context: bool = True
    resources: bool = True
    streaming: bool = True
    max_context_size: int = 1024 * 1024  # 1MB default
    supported_encodings: List[str] = ["json", "msgpack"]
    protocol_version: str = "1.0"
    server_info: Dict[str, Any] = {}


class MCPConnectRequest(BaseModel):
    """Connection request"""
    client_id: str
    client_info: Dict[str, Any] = {}
    requested_capabilities: Optional[List[str]] = None


class MCPConnectResponse(BaseModel):
    """Connection response"""
    session_id: str
    capabilities: MCPCapabilities
    welcome_message: Optional[str] = None


class MCPToolExecutionRequest(BaseModel):
    """Tool execution request"""
    tool_name: str
    arguments: Dict[str, Any] = {}
    context_id: Optional[str] = None
    timeout: Optional[int] = None  # seconds


class MCPToolExecutionResult(BaseModel):
    """Tool execution result"""
    tool_name: str
    success: bool
    result: Optional[Any] = None
    error: Optional[str] = None
    execution_time: float  # seconds
    metadata: Dict[str, Any] = {}


# Re-export commonly used types
__all__ = [
    'MCPMessageType',
    'MCPErrorCode',
    'MCPMessage',
    'MCPRequest',
    'MCPResponse',
    'MCPError',
    'MCPToolParameter',
    'MCPTool',
    'MCPContext',
    'MCPResource',
    'MCPCapabilities',
    'MCPConnectRequest',
    'MCPConnectResponse',
    'MCPToolExecutionRequest',
    'MCPToolExecutionResult'
]