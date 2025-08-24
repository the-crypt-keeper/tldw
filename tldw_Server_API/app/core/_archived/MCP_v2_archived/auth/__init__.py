"""
Authentication and Authorization for MCP v2
"""

from .jwt_auth import (
    JWTAuth,
    jwt_auth,
    get_current_user,
    get_current_active_user,
    get_optional_user,
    require_roles,
    require_permissions
)

from .rate_limiter import (
    RateLimiter,
    SlidingWindowRateLimiter,
    RateLimitManager,
    RateLimitMiddleware,
    rate_limit_manager,
    rate_limit,
    cleanup_rate_limits
)

from .rbac import (
    ResourceType,
    Action,
    Permission,
    Role,
    RBACPolicy,
    rbac_policy,
    check_module_access,
    check_tool_permission,
    check_resource_permission,
    require_module_permission
)

__all__ = [
    # JWT Auth
    'JWTAuth',
    'jwt_auth',
    'get_current_user',
    'get_current_active_user',
    'get_optional_user',
    'require_roles',
    'require_permissions',
    
    # Rate Limiting
    'RateLimiter',
    'SlidingWindowRateLimiter',
    'RateLimitManager',
    'RateLimitMiddleware',
    'rate_limit_manager',
    'rate_limit',
    'cleanup_rate_limits',
    
    # RBAC
    'ResourceType',
    'Action',
    'Permission',
    'Role',
    'RBACPolicy',
    'rbac_policy',
    'check_module_access',
    'check_tool_permission',
    'check_resource_permission',
    'require_module_permission'
]