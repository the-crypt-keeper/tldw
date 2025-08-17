# admin_schemas.py
# Description: Pydantic schemas for admin endpoints
#
# Imports
from typing import Optional, Dict, Any, List
from datetime import datetime
from pydantic import BaseModel, Field, EmailStr, ConfigDict

#######################################################################################################################
#
# User Management Schemas

class UserUpdateRequest(BaseModel):
    """Request to update user information"""
    email: Optional[EmailStr] = None
    role: Optional[str] = Field(None, pattern="^(user|admin|service)$")
    is_active: Optional[bool] = None
    is_verified: Optional[bool] = None
    is_locked: Optional[bool] = None
    storage_quota_mb: Optional[int] = Field(None, ge=100)
    
    model_config = ConfigDict(from_attributes=True)


class UserSummary(BaseModel):
    """Summary information about a user"""
    id: int
    uuid: str
    username: str
    email: str
    role: str
    is_active: bool
    is_verified: bool
    created_at: datetime
    last_login: Optional[datetime] = None
    storage_quota_mb: int
    storage_used_mb: float
    
    model_config = ConfigDict(from_attributes=True)


class UserListResponse(BaseModel):
    """Response for user list endpoint"""
    users: List[UserSummary]
    total: int
    page: int
    limit: int
    pages: int
    
    model_config = ConfigDict(from_attributes=True)


class UserQuotaUpdateRequest(BaseModel):
    """Request to update user storage quota"""
    storage_quota_mb: int = Field(..., ge=100, le=1000000)  # 100MB to 1TB
    
    model_config = ConfigDict(from_attributes=True)


#######################################################################################################################
#
# Registration Code Schemas

class RegistrationCodeRequest(BaseModel):
    """Request to create a registration code"""
    max_uses: int = Field(1, ge=1, le=100)
    expiry_days: int = Field(7, ge=1, le=365)
    role_to_grant: str = Field("user", pattern="^(user|admin|service)$")
    metadata: Optional[Dict[str, Any]] = None
    
    model_config = ConfigDict(from_attributes=True)


class RegistrationCodeResponse(BaseModel):
    """Response with registration code details"""
    id: int
    code: str
    max_uses: int
    times_used: int
    expires_at: datetime
    created_at: datetime
    role_to_grant: str
    is_valid: Optional[bool] = None
    
    model_config = ConfigDict(from_attributes=True)
    
    def __init__(self, **data):
        super().__init__(**data)
        # Calculate is_valid if not provided
        if self.is_valid is None:
            self.is_valid = (
                self.times_used < self.max_uses and 
                self.expires_at > datetime.utcnow()
            )


class RegistrationCodeListResponse(BaseModel):
    """Response for registration code list"""
    codes: List[RegistrationCodeResponse]
    
    model_config = ConfigDict(from_attributes=True)


#######################################################################################################################
#
# System Statistics Schemas

class UserStats(BaseModel):
    """User statistics"""
    total: int
    active: int
    verified: int
    admins: int
    new_last_30d: int
    
    model_config = ConfigDict(from_attributes=True)


class StorageStats(BaseModel):
    """Storage statistics"""
    total_used_mb: float
    total_quota_mb: float
    average_used_mb: float
    max_used_mb: float
    
    model_config = ConfigDict(from_attributes=True)


class SessionStats(BaseModel):
    """Session statistics"""
    active: int
    unique_users: int
    
    model_config = ConfigDict(from_attributes=True)


class SystemStatsResponse(BaseModel):
    """System statistics response"""
    users: UserStats
    storage: StorageStats
    sessions: SessionStats
    
    model_config = ConfigDict(from_attributes=True)


#######################################################################################################################
#
# Audit Log Schemas

class AuditLogEntry(BaseModel):
    """Single audit log entry"""
    id: int
    user_id: Optional[int] = None
    username: Optional[str] = None
    action: str
    details: Optional[Any] = None
    ip_address: Optional[str] = None
    created_at: datetime
    
    model_config = ConfigDict(from_attributes=True)


class AuditLogResponse(BaseModel):
    """Response for audit log endpoint"""
    entries: List[AuditLogEntry]
    
    model_config = ConfigDict(from_attributes=True)


#######################################################################################################################
#
# Batch Operation Schemas

class BatchUserOperation(BaseModel):
    """Batch operation on multiple users"""
    user_ids: List[int]
    operation: str = Field(..., pattern="^(activate|deactivate|verify|lock|unlock|delete)$")
    
    model_config = ConfigDict(from_attributes=True)


class BatchOperationResponse(BaseModel):
    """Response for batch operations"""
    success_count: int
    failed_count: int
    failed_ids: List[int] = []
    message: str
    
    model_config = ConfigDict(from_attributes=True)


#
## End of admin_schemas.py
#######################################################################################################################