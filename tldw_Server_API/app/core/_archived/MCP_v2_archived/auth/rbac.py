"""
Role-Based Access Control (RBAC) for MCP v2
"""

from typing import Dict, List, Set, Optional, Any
from enum import Enum
from dataclasses import dataclass, field
from loguru import logger

from ..schemas import UserRole, MCPUser


class ResourceType(str, Enum):
    """Types of resources in the system"""
    MODULE = "module"
    TOOL = "tool"
    RESOURCE = "resource"
    PROMPT = "prompt"
    MEDIA = "media"
    NOTE = "note"
    CONVERSATION = "conversation"
    TRANSCRIPT = "transcript"


class Action(str, Enum):
    """Actions that can be performed on resources"""
    READ = "read"
    WRITE = "write"
    EXECUTE = "execute"
    DELETE = "delete"
    ADMIN = "admin"


@dataclass
class Permission:
    """A permission defines an action on a resource"""
    resource_type: ResourceType
    resource_id: Optional[str] = None  # None means all resources of this type
    action: Action = Action.READ
    conditions: Dict[str, Any] = field(default_factory=dict)
    
    def __str__(self):
        if self.resource_id:
            return f"{self.resource_type.value}:{self.resource_id}:{self.action.value}"
        return f"{self.resource_type.value}:*:{self.action.value}"
    
    def matches(self, resource_type: str, resource_id: str, action: str) -> bool:
        """Check if this permission matches a request"""
        # Check resource type
        if self.resource_type.value != resource_type:
            return False
        
        # Check resource ID (None means all)
        if self.resource_id and self.resource_id != resource_id:
            return False
        
        # Check action
        if self.action.value != action:
            return False
        
        return True


@dataclass
class Role:
    """A role is a collection of permissions"""
    name: str
    description: str
    permissions: List[Permission] = field(default_factory=list)
    inherits_from: Optional[List[str]] = field(default_factory=list)
    
    def has_permission(self, resource_type: str, resource_id: str, action: str) -> bool:
        """Check if role has a specific permission"""
        for perm in self.permissions:
            if perm.matches(resource_type, resource_id, action):
                return True
        return False


class RBACPolicy:
    """RBAC Policy manager"""
    
    def __init__(self):
        self.roles: Dict[str, Role] = {}
        self.user_roles: Dict[str, Set[str]] = {}
        self.user_permissions: Dict[str, Set[Permission]] = {}
        
        # Initialize default roles
        self._init_default_roles()
    
    def _init_default_roles(self):
        """Initialize default system roles"""
        
        # Admin role - full access
        admin_role = Role(
            name=UserRole.ADMIN.value,
            description="Administrator with full system access",
            permissions=[
                Permission(ResourceType.MODULE, None, Action.ADMIN),
                Permission(ResourceType.TOOL, None, Action.ADMIN),
                Permission(ResourceType.RESOURCE, None, Action.ADMIN),
                Permission(ResourceType.PROMPT, None, Action.ADMIN),
                Permission(ResourceType.MEDIA, None, Action.ADMIN),
                Permission(ResourceType.NOTE, None, Action.ADMIN),
                Permission(ResourceType.CONVERSATION, None, Action.ADMIN),
                Permission(ResourceType.TRANSCRIPT, None, Action.ADMIN),
            ]
        )
        
        # User role - standard access
        user_role = Role(
            name=UserRole.USER.value,
            description="Standard user with read/write access to own content",
            permissions=[
                # Can execute most tools
                Permission(ResourceType.TOOL, None, Action.EXECUTE),
                # Can read resources
                Permission(ResourceType.RESOURCE, None, Action.READ),
                # Can use prompts
                Permission(ResourceType.PROMPT, None, Action.READ),
                Permission(ResourceType.PROMPT, None, Action.EXECUTE),
                # Can manage own media
                Permission(ResourceType.MEDIA, None, Action.READ),
                Permission(ResourceType.MEDIA, None, Action.WRITE),
                # Can manage own notes
                Permission(ResourceType.NOTE, None, Action.READ),
                Permission(ResourceType.NOTE, None, Action.WRITE),
                Permission(ResourceType.NOTE, None, Action.DELETE),
                # Can manage own conversations
                Permission(ResourceType.CONVERSATION, None, Action.READ),
                Permission(ResourceType.CONVERSATION, None, Action.WRITE),
                Permission(ResourceType.CONVERSATION, None, Action.DELETE),
                # Can read transcripts
                Permission(ResourceType.TRANSCRIPT, None, Action.READ),
            ]
        )
        
        # API Client role - programmatic access
        api_client_role = Role(
            name=UserRole.API_CLIENT.value,
            description="API client with limited programmatic access",
            permissions=[
                # Can execute specific tools
                Permission(ResourceType.TOOL, "search_media", Action.EXECUTE),
                Permission(ResourceType.TOOL, "get_transcript", Action.EXECUTE),
                Permission(ResourceType.TOOL, "chat_completion", Action.EXECUTE),
                # Read-only access to resources
                Permission(ResourceType.RESOURCE, None, Action.READ),
                Permission(ResourceType.MEDIA, None, Action.READ),
                Permission(ResourceType.PROMPT, None, Action.READ),
            ]
        )
        
        # Guest role - minimal access
        guest_role = Role(
            name=UserRole.GUEST.value,
            description="Guest with minimal read-only access",
            permissions=[
                # Can only read public resources
                Permission(ResourceType.RESOURCE, None, Action.READ),
                # Can search media
                Permission(ResourceType.TOOL, "search_media", Action.EXECUTE),
                # Can view media metadata
                Permission(ResourceType.MEDIA, None, Action.READ),
            ]
        )
        
        # Department-specific roles
        media_admin_role = Role(
            name="media_admin",
            description="Administrator for media module",
            permissions=[
                Permission(ResourceType.MODULE, "media", Action.ADMIN),
                Permission(ResourceType.MEDIA, None, Action.ADMIN),
                Permission(ResourceType.TRANSCRIPT, None, Action.ADMIN),
            ],
            inherits_from=[UserRole.USER.value]
        )
        
        notes_admin_role = Role(
            name="notes_admin",
            description="Administrator for notes module",
            permissions=[
                Permission(ResourceType.MODULE, "notes", Action.ADMIN),
                Permission(ResourceType.NOTE, None, Action.ADMIN),
            ],
            inherits_from=[UserRole.USER.value]
        )
        
        chat_moderator_role = Role(
            name="chat_moderator",
            description="Moderator for chat conversations",
            permissions=[
                Permission(ResourceType.MODULE, "chat", Action.ADMIN),
                Permission(ResourceType.CONVERSATION, None, Action.READ),
                Permission(ResourceType.CONVERSATION, None, Action.DELETE),
            ],
            inherits_from=[UserRole.USER.value]
        )
        
        # Register all roles
        self.roles = {
            admin_role.name: admin_role,
            user_role.name: user_role,
            api_client_role.name: api_client_role,
            guest_role.name: guest_role,
            media_admin_role.name: media_admin_role,
            notes_admin_role.name: notes_admin_role,
            chat_moderator_role.name: chat_moderator_role,
        }
    
    def add_role(self, role: Role):
        """Add a new role to the system"""
        self.roles[role.name] = role
        logger.info(f"Added role: {role.name}")
    
    def assign_role(self, user_id: str, role_name: str):
        """Assign a role to a user"""
        if role_name not in self.roles:
            raise ValueError(f"Role {role_name} does not exist")
        
        if user_id not in self.user_roles:
            self.user_roles[user_id] = set()
        
        self.user_roles[user_id].add(role_name)
        logger.info(f"Assigned role {role_name} to user {user_id}")
    
    def revoke_role(self, user_id: str, role_name: str):
        """Revoke a role from a user"""
        if user_id in self.user_roles:
            self.user_roles[user_id].discard(role_name)
            logger.info(f"Revoked role {role_name} from user {user_id}")
    
    def grant_permission(self, user_id: str, permission: Permission):
        """Grant a specific permission to a user"""
        if user_id not in self.user_permissions:
            self.user_permissions[user_id] = set()
        
        self.user_permissions[user_id].add(permission)
        logger.info(f"Granted permission {permission} to user {user_id}")
    
    def revoke_permission(self, user_id: str, permission: Permission):
        """Revoke a specific permission from a user"""
        if user_id in self.user_permissions:
            self.user_permissions[user_id].discard(permission)
            logger.info(f"Revoked permission {permission} from user {user_id}")
    
    def check_permission(
        self,
        user: MCPUser,
        resource_type: ResourceType,
        resource_id: str,
        action: Action
    ) -> bool:
        """
        Check if a user has permission to perform an action on a resource
        
        Args:
            user: The user to check
            resource_type: Type of resource
            resource_id: ID of the specific resource
            action: Action to perform
        
        Returns:
            True if permitted, False otherwise
        """
        # Admin users have all permissions
        if UserRole.ADMIN in user.roles:
            return True
        
        # Check user's direct permissions
        if user.id in self.user_permissions:
            for perm in self.user_permissions[user.id]:
                if perm.matches(resource_type.value, resource_id, action.value):
                    return True
        
        # Check permissions from user's roles
        user_role_names = set()
        
        # Add roles from MCPUser object
        for role in user.roles:
            user_role_names.add(role.value)
        
        # Add assigned roles
        if user.id in self.user_roles:
            user_role_names.update(self.user_roles[user.id])
        
        # Check each role's permissions (including inherited)
        checked_roles = set()
        roles_to_check = list(user_role_names)
        
        while roles_to_check:
            role_name = roles_to_check.pop()
            
            if role_name in checked_roles:
                continue
            
            checked_roles.add(role_name)
            
            if role_name not in self.roles:
                continue
            
            role = self.roles[role_name]
            
            # Check role's permissions
            if role.has_permission(resource_type.value, resource_id, action.value):
                return True
            
            # Add inherited roles to check
            if role.inherits_from:
                roles_to_check.extend(role.inherits_from)
        
        return False
    
    def get_user_permissions(self, user: MCPUser) -> List[Permission]:
        """Get all permissions for a user"""
        permissions = []
        
        # Admin gets all permissions
        if UserRole.ADMIN in user.roles:
            for role in self.roles.values():
                permissions.extend(role.permissions)
            return permissions
        
        # Collect direct permissions
        if user.id in self.user_permissions:
            permissions.extend(self.user_permissions[user.id])
        
        # Collect permissions from roles
        user_role_names = set()
        
        # Add roles from MCPUser object
        for role in user.roles:
            user_role_names.add(role.value)
        
        # Add assigned roles
        if user.id in self.user_roles:
            user_role_names.update(self.user_roles[user.id])
        
        # Collect from each role (including inherited)
        checked_roles = set()
        roles_to_check = list(user_role_names)
        
        while roles_to_check:
            role_name = roles_to_check.pop()
            
            if role_name in checked_roles:
                continue
            
            checked_roles.add(role_name)
            
            if role_name not in self.roles:
                continue
            
            role = self.roles[role_name]
            permissions.extend(role.permissions)
            
            if role.inherits_from:
                roles_to_check.extend(role.inherits_from)
        
        return permissions
    
    def get_user_roles(self, user_id: str) -> Set[str]:
        """Get all roles for a user"""
        return self.user_roles.get(user_id, set())
    
    def can_access_module(self, user: MCPUser, module_id: str) -> bool:
        """Check if user can access a module"""
        return self.check_permission(
            user,
            ResourceType.MODULE,
            module_id,
            Action.READ
        )
    
    def can_execute_tool(self, user: MCPUser, tool_name: str) -> bool:
        """Check if user can execute a tool"""
        return self.check_permission(
            user,
            ResourceType.TOOL,
            tool_name,
            Action.EXECUTE
        )
    
    def can_read_resource(self, user: MCPUser, resource_uri: str) -> bool:
        """Check if user can read a resource"""
        return self.check_permission(
            user,
            ResourceType.RESOURCE,
            resource_uri,
            Action.READ
        )


# Global RBAC policy instance
rbac_policy = RBACPolicy()


# Helper functions for common permission checks

def check_module_access(user: MCPUser, module_id: str) -> bool:
    """Check if user can access a module"""
    return rbac_policy.can_access_module(user, module_id)


def check_tool_permission(user: MCPUser, tool_name: str) -> bool:
    """Check if user can execute a tool"""
    return rbac_policy.can_execute_tool(user, tool_name)


def check_resource_permission(user: MCPUser, resource_uri: str) -> bool:
    """Check if user can read a resource"""
    return rbac_policy.can_read_resource(user, resource_uri)


def require_module_permission(module_id: str, action: Action = Action.READ):
    """Dependency to require module permission"""
    from .jwt_auth import get_current_active_user
    from fastapi import Security, HTTPException, status
    
    async def permission_checker(
        current_user: MCPUser = Security(get_current_active_user)
    ) -> MCPUser:
        if not rbac_policy.check_permission(
            current_user,
            ResourceType.MODULE,
            module_id,
            action
        ):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"No permission for {action.value} on module {module_id}"
            )
        return current_user
    
    return permission_checker