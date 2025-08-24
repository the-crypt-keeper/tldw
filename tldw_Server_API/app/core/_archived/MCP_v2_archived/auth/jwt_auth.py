"""
JWT Authentication for MCP v2
"""

from typing import Optional, Dict, Any, List
from datetime import datetime, timedelta
import jwt
from passlib.context import CryptContext
from fastapi import HTTPException, Security, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from loguru import logger

from ..schemas import MCPUser, UserRole


# Configuration
SECRET_KEY = "your-secret-key-change-this-in-production"  # Should be in env/config
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30
REFRESH_TOKEN_EXPIRE_DAYS = 7

# Password hashing
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

# Bearer token security
security = HTTPBearer()


class JWTAuth:
    """JWT Authentication manager for MCP v2"""
    
    def __init__(self, secret_key: str = SECRET_KEY, algorithm: str = ALGORITHM):
        self.secret_key = secret_key
        self.algorithm = algorithm
        self.pwd_context = pwd_context
    
    def verify_password(self, plain_password: str, hashed_password: str) -> bool:
        """Verify a password against its hash"""
        return self.pwd_context.verify(plain_password, hashed_password)
    
    def get_password_hash(self, password: str) -> str:
        """Hash a password"""
        return self.pwd_context.hash(password)
    
    def create_access_token(
        self,
        data: Dict[str, Any],
        expires_delta: Optional[timedelta] = None
    ) -> str:
        """Create a JWT access token"""
        to_encode = data.copy()
        
        if expires_delta:
            expire = datetime.utcnow() + expires_delta
        else:
            expire = datetime.utcnow() + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
        
        to_encode.update({"exp": expire, "type": "access"})
        encoded_jwt = jwt.encode(to_encode, self.secret_key, algorithm=self.algorithm)
        
        return encoded_jwt
    
    def create_refresh_token(
        self,
        data: Dict[str, Any],
        expires_delta: Optional[timedelta] = None
    ) -> str:
        """Create a JWT refresh token"""
        to_encode = data.copy()
        
        if expires_delta:
            expire = datetime.utcnow() + expires_delta
        else:
            expire = datetime.utcnow() + timedelta(days=REFRESH_TOKEN_EXPIRE_DAYS)
        
        to_encode.update({"exp": expire, "type": "refresh"})
        encoded_jwt = jwt.encode(to_encode, self.secret_key, algorithm=self.algorithm)
        
        return encoded_jwt
    
    def decode_token(self, token: str) -> Dict[str, Any]:
        """Decode and validate a JWT token"""
        try:
            payload = jwt.decode(token, self.secret_key, algorithms=[self.algorithm])
            return payload
        except jwt.ExpiredSignatureError:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Token has expired",
                headers={"WWW-Authenticate": "Bearer"},
            )
        except jwt.JWTError:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Could not validate credentials",
                headers={"WWW-Authenticate": "Bearer"},
            )
    
    async def get_current_user(
        self,
        credentials: HTTPAuthorizationCredentials = Security(security)
    ) -> MCPUser:
        """Get the current user from JWT token"""
        token = credentials.credentials
        
        try:
            payload = self.decode_token(token)
            
            # Check token type
            if payload.get("type") != "access":
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="Invalid token type",
                    headers={"WWW-Authenticate": "Bearer"},
                )
            
            # Extract user information
            user_id = payload.get("sub")
            username = payload.get("username")
            roles = payload.get("roles", ["user"])
            department = payload.get("department")
            permissions = payload.get("permissions", [])
            
            if not user_id:
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="Invalid token payload",
                    headers={"WWW-Authenticate": "Bearer"},
                )
            
            # Create MCPUser object
            user = MCPUser(
                id=user_id,
                username=username or user_id,
                roles=[UserRole(role) for role in roles],
                department=department,
                permissions=permissions,
                metadata={"token_issued": payload.get("iat")}
            )
            
            return user
            
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Error decoding token: {e}")
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Could not validate credentials",
                headers={"WWW-Authenticate": "Bearer"},
            )
    
    def create_api_key(self, user_id: str, name: str, permissions: List[str]) -> str:
        """Create an API key for a user"""
        # API keys don't expire by default
        data = {
            "sub": user_id,
            "type": "api_key",
            "name": name,
            "permissions": permissions,
            "created": datetime.utcnow().isoformat()
        }
        
        api_key = jwt.encode(data, self.secret_key, algorithm=self.algorithm)
        return api_key
    
    def validate_api_key(self, api_key: str) -> Dict[str, Any]:
        """Validate an API key"""
        try:
            payload = jwt.decode(api_key, self.secret_key, algorithms=[self.algorithm])
            
            if payload.get("type") != "api_key":
                raise ValueError("Invalid API key type")
            
            return payload
        except Exception as e:
            logger.error(f"Invalid API key: {e}")
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid API key"
            )


# Global JWT auth instance
jwt_auth = JWTAuth()


# Dependency functions for FastAPI

async def get_current_user(
    credentials: HTTPAuthorizationCredentials = Security(security)
) -> MCPUser:
    """FastAPI dependency to get current user"""
    return await jwt_auth.get_current_user(credentials)


async def get_current_active_user(
    current_user: MCPUser = Security(get_current_user)
) -> MCPUser:
    """FastAPI dependency to get current active user"""
    # Could check if user is active/not banned here
    return current_user


def require_roles(*roles: UserRole):
    """Dependency to require specific roles"""
    async def role_checker(
        current_user: MCPUser = Security(get_current_active_user)
    ) -> MCPUser:
        user_roles = set(current_user.roles)
        required_roles = set(roles)
        
        if not required_roles.intersection(user_roles):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Required roles: {', '.join(r.value for r in roles)}"
            )
        
        return current_user
    
    return role_checker


def require_permissions(*permissions: str):
    """Dependency to require specific permissions"""
    async def permission_checker(
        current_user: MCPUser = Security(get_current_active_user)
    ) -> MCPUser:
        user_permissions = set(current_user.permissions)
        required_permissions = set(permissions)
        
        # Admin bypasses all permission checks
        if UserRole.ADMIN in current_user.roles:
            return current_user
        
        if not required_permissions.issubset(user_permissions):
            missing = required_permissions - user_permissions
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Missing permissions: {', '.join(missing)}"
            )
        
        return current_user
    
    return permission_checker


# Optional authentication (returns None if no auth provided)
async def get_optional_user(
    credentials: Optional[HTTPAuthorizationCredentials] = Security(HTTPBearer(auto_error=False))
) -> Optional[MCPUser]:
    """Get user if authenticated, None otherwise"""
    if not credentials:
        return None
    
    try:
        return await jwt_auth.get_current_user(credentials)
    except:
        return None