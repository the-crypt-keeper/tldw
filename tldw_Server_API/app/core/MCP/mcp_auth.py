# mcp_auth.py - MCP Authentication and Authorization Module
"""
Authentication and authorization for MCP (Model Context Protocol) server.

Implements:
- JWT-based authentication
- API key authentication
- Role-based access control (RBAC)
- Permission system for tools and resources
- Rate limiting per client
"""

import jwt
import hashlib
import secrets
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Optional, Set, Any
from dataclasses import dataclass, field
from enum import Enum
import asyncio
from loguru import logger
from fastapi import HTTPException, status, Depends, Header
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
import redis
from pydantic import BaseModel

from tldw_Server_API.app.core.config import load_and_log_configs
from tldw_Server_API.app.core.Security.Security import JWT_SECRET_KEY


# Security scheme
security = HTTPBearer()


class MCPRole(str, Enum):
    """MCP user roles"""
    ADMIN = "admin"
    USER = "user"
    SERVICE = "service"  # For service-to-service communication
    GUEST = "guest"


class MCPPermission(str, Enum):
    """MCP permissions"""
    # Tool permissions
    TOOLS_READ = "tools:read"
    TOOLS_EXECUTE = "tools:execute"
    TOOLS_MANAGE = "tools:manage"
    
    # Resource permissions
    RESOURCES_READ = "resources:read"
    RESOURCES_WRITE = "resources:write"
    RESOURCES_DELETE = "resources:delete"
    
    # Context permissions
    CONTEXT_READ = "context:read"
    CONTEXT_WRITE = "context:write"
    
    # Admin permissions
    ADMIN_USERS = "admin:users"
    ADMIN_TOOLS = "admin:tools"
    ADMIN_ALL = "admin:*"


# Role-permission mapping
ROLE_PERMISSIONS: Dict[MCPRole, Set[MCPPermission]] = {
    MCPRole.ADMIN: {
        MCPPermission.TOOLS_READ,
        MCPPermission.TOOLS_EXECUTE,
        MCPPermission.TOOLS_MANAGE,
        MCPPermission.RESOURCES_READ,
        MCPPermission.RESOURCES_WRITE,
        MCPPermission.RESOURCES_DELETE,
        MCPPermission.CONTEXT_READ,
        MCPPermission.CONTEXT_WRITE,
        MCPPermission.ADMIN_USERS,
        MCPPermission.ADMIN_TOOLS,
        MCPPermission.ADMIN_ALL,
    },
    MCPRole.USER: {
        MCPPermission.TOOLS_READ,
        MCPPermission.TOOLS_EXECUTE,
        MCPPermission.RESOURCES_READ,
        MCPPermission.RESOURCES_WRITE,
        MCPPermission.CONTEXT_READ,
        MCPPermission.CONTEXT_WRITE,
    },
    MCPRole.SERVICE: {
        MCPPermission.TOOLS_READ,
        MCPPermission.TOOLS_EXECUTE,
        MCPPermission.RESOURCES_READ,
        MCPPermission.CONTEXT_READ,
    },
    MCPRole.GUEST: {
        MCPPermission.TOOLS_READ,
        MCPPermission.RESOURCES_READ,
        MCPPermission.CONTEXT_READ,
    }
}


@dataclass
class MCPClient:
    """Represents an authenticated MCP client"""
    client_id: str
    name: str
    role: MCPRole
    permissions: Set[MCPPermission] = field(default_factory=set)
    api_key_hash: Optional[str] = None
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    last_seen: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    rate_limit: int = 100  # Requests per minute
    allowed_tools: Optional[List[str]] = None  # None means all tools
    metadata: Dict[str, Any] = field(default_factory=dict)


class MCPAuthRequest(BaseModel):
    """Authentication request"""
    api_key: Optional[str] = None
    client_id: Optional[str] = None
    client_secret: Optional[str] = None


class MCPAuthResponse(BaseModel):
    """Authentication response"""
    access_token: str
    token_type: str = "bearer"
    expires_in: int = 3600
    client_id: str
    role: str
    permissions: List[str]


class MCPTokenData(BaseModel):
    """Token payload data"""
    client_id: str
    role: str
    permissions: List[str]
    exp: float
    iat: float


class MCPAuthManager:
    """Manages MCP authentication and authorization"""
    
    def __init__(self):
        self.config = load_and_log_configs()
        self.jwt_secret = JWT_SECRET_KEY
        self.jwt_algorithm = "HS256"
        self.token_expiry = timedelta(hours=24)
        
        # In-memory client store (should be replaced with database in production)
        self.clients: Dict[str, MCPClient] = {}
        
        # Redis for rate limiting (optional)
        self.redis_client = None
        self._init_redis()
        
        # Initialize default clients
        self._init_default_clients()
    
    def _init_redis(self):
        """Initialize Redis connection for rate limiting"""
        try:
            redis_config = self.config.get('redis', {})
            if redis_config.get('enabled', False):
                self.redis_client = redis.Redis(
                    host=redis_config.get('host', 'localhost'),
                    port=redis_config.get('port', 6379),
                    db=redis_config.get('db', 0),
                    decode_responses=True
                )
                self.redis_client.ping()
                logger.info("Redis connected for MCP rate limiting")
        except Exception as e:
            logger.warning(f"Redis not available for MCP rate limiting: {e}")
            self.redis_client = None
    
    def _init_default_clients(self):
        """Initialize default service clients"""
        # Admin client
        admin_client = MCPClient(
            client_id="mcp-admin",
            name="MCP Administrator",
            role=MCPRole.ADMIN,
            permissions=ROLE_PERMISSIONS[MCPRole.ADMIN],
            api_key_hash=self._hash_api_key("admin-secret-key")  # Change in production
        )
        self.clients[admin_client.client_id] = admin_client
        
        # Default service client
        service_client = MCPClient(
            client_id="mcp-service",
            name="MCP Service Account",
            role=MCPRole.SERVICE,
            permissions=ROLE_PERMISSIONS[MCPRole.SERVICE],
            api_key_hash=self._hash_api_key("service-secret-key")  # Change in production
        )
        self.clients[service_client.client_id] = service_client
        
        logger.info(f"Initialized {len(self.clients)} default MCP clients")
    
    def _hash_api_key(self, api_key: str) -> str:
        """Hash an API key for storage"""
        return hashlib.sha256(api_key.encode()).hexdigest()
    
    def create_client(
        self,
        name: str,
        role: MCPRole = MCPRole.USER,
        permissions: Optional[Set[MCPPermission]] = None,
        allowed_tools: Optional[List[str]] = None,
        rate_limit: int = 100
    ) -> tuple[MCPClient, str]:
        """
        Create a new MCP client.
        
        Returns:
            Tuple of (client, api_key)
        """
        client_id = f"mcp-{secrets.token_urlsafe(16)}"
        api_key = secrets.token_urlsafe(32)
        
        if permissions is None:
            permissions = ROLE_PERMISSIONS.get(role, set())
        
        client = MCPClient(
            client_id=client_id,
            name=name,
            role=role,
            permissions=permissions,
            api_key_hash=self._hash_api_key(api_key),
            allowed_tools=allowed_tools,
            rate_limit=rate_limit
        )
        
        self.clients[client_id] = client
        logger.info(f"Created MCP client: {client_id} with role {role}")
        
        return client, api_key
    
    def authenticate_api_key(self, api_key: str) -> Optional[MCPClient]:
        """Authenticate using API key"""
        api_key_hash = self._hash_api_key(api_key)
        
        for client in self.clients.values():
            if client.api_key_hash == api_key_hash:
                client.last_seen = datetime.now(timezone.utc)
                return client
        
        return None
    
    def authenticate_client_credentials(
        self, 
        client_id: str, 
        client_secret: str
    ) -> Optional[MCPClient]:
        """Authenticate using client ID and secret"""
        client = self.clients.get(client_id)
        if client and client.api_key_hash == self._hash_api_key(client_secret):
            client.last_seen = datetime.now(timezone.utc)
            return client
        
        return None
    
    def generate_token(self, client: MCPClient) -> str:
        """Generate JWT token for authenticated client"""
        now = datetime.now(timezone.utc)
        exp = now + self.token_expiry
        
        payload = {
            "client_id": client.client_id,
            "role": client.role.value,
            "permissions": [p.value for p in client.permissions],
            "exp": exp.timestamp(),
            "iat": now.timestamp()
        }
        
        token = jwt.encode(payload, self.jwt_secret, algorithm=self.jwt_algorithm)
        return token
    
    def verify_token(self, token: str) -> Optional[MCPTokenData]:
        """Verify and decode JWT token"""
        try:
            payload = jwt.decode(
                token, 
                self.jwt_secret, 
                algorithms=[self.jwt_algorithm]
            )
            return MCPTokenData(**payload)
        except jwt.ExpiredSignatureError:
            logger.warning("MCP token expired")
            return None
        except jwt.InvalidTokenError as e:
            logger.warning(f"Invalid MCP token: {e}")
            return None
    
    def check_permission(
        self, 
        client: MCPClient, 
        permission: MCPPermission
    ) -> bool:
        """Check if client has specific permission"""
        return permission in client.permissions or MCPPermission.ADMIN_ALL in client.permissions
    
    def check_tool_access(self, client: MCPClient, tool_name: str) -> bool:
        """Check if client can access specific tool"""
        if client.allowed_tools is None:
            # None means all tools allowed
            return True
        return tool_name in client.allowed_tools
    
    async def check_rate_limit(self, client_id: str) -> bool:
        """Check if client is within rate limit"""
        if not self.redis_client:
            # No rate limiting without Redis
            return True
        
        try:
            key = f"mcp:rate:{client_id}"
            current = await asyncio.to_thread(self.redis_client.incr, key)
            
            if current == 1:
                # First request, set expiry
                await asyncio.to_thread(self.redis_client.expire, key, 60)
            
            client = self.clients.get(client_id)
            limit = client.rate_limit if client else 100
            
            return current <= limit
            
        except Exception as e:
            logger.error(f"Rate limit check failed: {e}")
            return True  # Allow on error
    
    def get_client(self, client_id: str) -> Optional[MCPClient]:
        """Get client by ID"""
        return self.clients.get(client_id)
    
    def list_clients(self) -> List[MCPClient]:
        """List all clients"""
        return list(self.clients.values())
    
    def update_client(
        self, 
        client_id: str,
        name: Optional[str] = None,
        role: Optional[MCPRole] = None,
        permissions: Optional[Set[MCPPermission]] = None,
        allowed_tools: Optional[List[str]] = None,
        rate_limit: Optional[int] = None
    ) -> Optional[MCPClient]:
        """Update client properties"""
        client = self.clients.get(client_id)
        if not client:
            return None
        
        if name is not None:
            client.name = name
        if role is not None:
            client.role = role
        if permissions is not None:
            client.permissions = permissions
        if allowed_tools is not None:
            client.allowed_tools = allowed_tools
        if rate_limit is not None:
            client.rate_limit = rate_limit
        
        logger.info(f"Updated MCP client: {client_id}")
        return client
    
    def delete_client(self, client_id: str) -> bool:
        """Delete a client"""
        if client_id in self.clients:
            del self.clients[client_id]
            logger.info(f"Deleted MCP client: {client_id}")
            return True
        return False


# Global auth manager instance
auth_manager = MCPAuthManager()


# FastAPI dependencies
async def get_current_client(
    credentials: HTTPAuthorizationCredentials = Depends(security)
) -> MCPClient:
    """Get current authenticated client from JWT token"""
    token = credentials.credentials
    token_data = auth_manager.verify_token(token)
    
    if not token_data:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired token",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    client = auth_manager.get_client(token_data.client_id)
    if not client:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Client not found",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    # Check rate limit
    if not await auth_manager.check_rate_limit(client.client_id):
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail="Rate limit exceeded"
        )
    
    return client


def require_permission(permission: MCPPermission):
    """Dependency to require specific permission"""
    async def permission_checker(client: MCPClient = Depends(get_current_client)):
        if not auth_manager.check_permission(client, permission):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Permission denied: {permission.value}"
            )
        return client
    return permission_checker


def require_tool_access(tool_name: str):
    """Dependency to require access to specific tool"""
    async def tool_access_checker(client: MCPClient = Depends(get_current_client)):
        if not auth_manager.check_tool_access(client, tool_name):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Access denied to tool: {tool_name}"
            )
        return client
    return tool_access_checker


# Optional: API key authentication for simpler integrations
async def get_client_from_api_key(
    x_api_key: Optional[str] = Header(None)
) -> Optional[MCPClient]:
    """Get client from API key header"""
    if not x_api_key:
        return None
    
    client = auth_manager.authenticate_api_key(x_api_key)
    if client and await auth_manager.check_rate_limit(client.client_id):
        return client
    
    return None