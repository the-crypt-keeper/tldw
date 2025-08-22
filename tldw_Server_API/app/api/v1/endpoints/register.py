# register.py
# Description: User registration endpoint with proper validation and security
#
# Imports
from typing import Dict, Any, Optional
from datetime import datetime
from uuid import uuid4
import re
#
# 3rd-party imports
from fastapi import APIRouter, Depends, HTTPException, status, Request
from loguru import logger
from pydantic import BaseModel, EmailStr, field_validator, ConfigDict
#
# Local imports
from tldw_Server_API.app.api.v1.schemas.auth_schemas import (
    RegisterRequest,
    RegistrationResponse,
    MessageResponse
)
from tldw_Server_API.app.api.v1.API_Deps.auth_deps import (
    get_db_transaction,
    get_password_service_dep,
    get_registration_service_dep,
    check_auth_rate_limit
)
from tldw_Server_API.app.core.AuthNZ.password_service import PasswordService
from tldw_Server_API.app.services.registration_service import RegistrationService
from tldw_Server_API.app.core.AuthNZ.settings import get_settings
from tldw_Server_API.app.core.AuthNZ.exceptions import (
    RegistrationError,
    DuplicateUserError,
    WeakPasswordError,
    InvalidRegistrationCodeError,
    RegistrationDisabledError
)
from tldw_Server_API.app.core.Audit.unified_audit_service import (
    get_unified_audit_service,
    AuditEventType,
    AuditContext
)

#######################################################################################################################
#
# Router Configuration

router = APIRouter(
    prefix="/register",
    tags=["registration"],
    responses={404: {"description": "Not found"}}
)


#######################################################################################################################
#
# Input Validation Schemas

class UserCreateRequest(BaseModel):
    """Request model for user registration with validation"""
    
    username: str
    email: EmailStr
    password: str
    confirm_password: str
    registration_code: Optional[str] = None
    
    model_config = ConfigDict(
        str_strip_whitespace=True,
        str_min_length=1
    )
    
    @field_validator('username')
    @classmethod
    def validate_username(cls, v: str) -> str:
        """Validate username format and length"""
        if not v or len(v) < 3:
            raise ValueError('Username must be at least 3 characters long')
        if len(v) > 50:
            raise ValueError('Username must not exceed 50 characters')
        
        # Allow only alphanumeric, underscore, and dash
        if not re.match(r'^[a-zA-Z0-9_-]+$', v):
            raise ValueError('Username can only contain letters, numbers, underscore and dash')
        
        # Prevent potentially problematic usernames
        reserved_names = ['admin', 'root', 'system', 'api', 'null', 'undefined']
        if v.lower() in reserved_names:
            raise ValueError('This username is reserved')
        
        return v
    
    @field_validator('email')
    @classmethod
    def validate_email(cls, v: str) -> str:
        """Additional email validation"""
        if len(v) > 255:
            raise ValueError('Email must not exceed 255 characters')
        
        # Basic check for obviously invalid emails
        if v.count('@') != 1:
            raise ValueError('Invalid email format')
        
        local, domain = v.split('@')
        if not local or not domain:
            raise ValueError('Invalid email format')
        
        # Prevent some common test emails in production
        test_domains = ['example.com', 'test.com', 'localhost']
        if domain.lower() in test_domains:
            logger.warning(f"Registration attempt with test email domain: {domain}")
        
        return v.lower()  # Normalize to lowercase
    
    @field_validator('password')
    @classmethod
    def validate_password_format(cls, v: str) -> str:
        """Basic password format validation"""
        if not v or len(v) < 10:
            raise ValueError('Password must be at least 10 characters long')
        if len(v) > 128:
            raise ValueError('Password must not exceed 128 characters')
        
        # Check for NULL bytes or other control characters
        if '\x00' in v or any(ord(c) < 32 for c in v):
            raise ValueError('Password contains invalid characters')
        
        return v
    
    @field_validator('confirm_password')
    @classmethod
    def passwords_match(cls, v: str, info) -> str:
        """Ensure password confirmation matches"""
        if 'password' in info.data and v != info.data['password']:
            raise ValueError('Passwords do not match')
        return v


#######################################################################################################################
#
# Registration Endpoint

@router.post(
    "/user",
    response_model=RegistrationResponse,
    status_code=status.HTTP_201_CREATED,
    dependencies=[Depends(check_auth_rate_limit)],
    summary="Register a new user",
    description="Create a new user account with the provided credentials"
)
async def register_user(
    request: Request,
    user_data: UserCreateRequest,
    db=Depends(get_db_transaction),
    password_service: PasswordService = Depends(get_password_service_dep),
    registration_service: RegistrationService = Depends(get_registration_service_dep),
    audit_service=Depends(get_unified_audit_service)
) -> RegistrationResponse:
    """
    Register a new user account
    
    This endpoint creates a new user account with the provided credentials.
    Registration may require a valid registration code depending on system configuration.
    
    Args:
        user_data: User registration information
        
    Returns:
        RegistrationResponse with success message
        
    Raises:
        HTTPException: Various status codes for different error conditions
    """
    settings = get_settings()
    
    # Check if registration is enabled
    if not settings.ENABLE_REGISTRATION:
        logger.warning(f"Registration attempt while disabled from IP: {request.client.host}")
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="User registration is currently disabled"
        )
    
    # Initialize registration service
    await registration_service.initialize()
    
    # Log registration attempt
    client_ip = request.client.host if request.client else "unknown"
    logger.info(f"Registration attempt for username: {user_data.username} from IP: {client_ip}")
    
    try:
        # Validate password strength with username context
        password_service.validate_password_strength(
            user_data.password,
            username=user_data.username
        )
        
        # Check for duplicate username
        existing_user = await db.fetchone(
            "SELECT id FROM users WHERE username = ? OR email = ?",
            user_data.username, user_data.email
        )
        
        if existing_user:
            # Don't reveal which field is duplicate (security)
            logger.warning(f"Duplicate registration attempt for: {user_data.username}")
            raise DuplicateUserError("username or email")
        
        # Validate registration code if required
        if settings.REQUIRE_REGISTRATION_CODE:
            if not user_data.registration_code:
                raise InvalidRegistrationCodeError("Registration code is required")
            
            # Verify registration code
            code_valid = await registration_service.validate_registration_code(
                user_data.registration_code,
                db_connection=db
            )
            
            if not code_valid:
                logger.warning(f"Invalid registration code used: {user_data.registration_code[:8]}...")
                raise InvalidRegistrationCodeError()
        
        # Hash the password
        password_hash = password_service.hash_password(user_data.password)
        
        # Generate user UUID
        user_uuid = str(uuid4())
        
        # Create user in database
        if hasattr(db, 'fetchval'):
            # PostgreSQL
            user_id = await db.fetchval(
                """
                INSERT INTO users (uuid, username, email, password_hash, role, is_active, is_verified)
                VALUES ($1, $2, $3, $4, $5, $6, $7)
                RETURNING id
                """,
                user_uuid, user_data.username, user_data.email, password_hash,
                settings.DEFAULT_USER_ROLE, True, False
            )
        else:
            # SQLite
            cursor = await db.execute(
                """
                INSERT INTO users (uuid, username, email, password_hash, role, is_active, is_verified)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (user_uuid, user_data.username, user_data.email, password_hash,
                 settings.DEFAULT_USER_ROLE, 1, 0)
            )
            user_id = cursor.lastrowid
        
        # Create user directories
        try:
            await registration_service.create_user_resources(user_id, db)
        except Exception as e:
            logger.error(f"Failed to create user resources for {user_id}: {e}")
            # Continue - user is created but might have limited functionality
        
        # Mark registration code as used if applicable
        if settings.REQUIRE_REGISTRATION_CODE and user_data.registration_code:
            await registration_service.mark_code_used(
                user_data.registration_code,
                user_id,
                db
            )
        
        # Audit log
        context = AuditContext(
            user_id=str(user_id),
            ip_address=client_ip
        )
        await audit_service.log_event(
            event_type=AuditEventType.USER_CREATED,
            context=context,
            metadata={
                "username": user_data.username,
                "email": user_data.email
            }
        )
        
        logger.info(f"Successfully registered user: {user_data.username} (ID: {user_id})")
        
        return RegistrationResponse(
            success=True,
            message="Registration successful. Please check your email to verify your account.",
            user_id=user_id,
            username=user_data.username
        )
        
    except (DuplicateUserError, WeakPasswordError, InvalidRegistrationCodeError) as e:
        # These are expected errors - log at warning level
        logger.warning(f"Registration failed for {user_data.username}: {e}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    except RegistrationDisabledError:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Registration is currently disabled"
        )
    except Exception as e:
        # Unexpected error - log at error level
        logger.error(f"Unexpected error during registration for {user_data.username}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An error occurred during registration. Please try again later."
        )


#######################################################################################################################
#
# Additional Endpoints

@router.post(
    "/validate-code",
    response_model=MessageResponse,
    summary="Validate a registration code",
    description="Check if a registration code is valid without using it"
)
async def validate_registration_code(
    code: str,
    registration_service: RegistrationService = Depends(get_registration_service_dep),
    db=Depends(get_db_transaction)
) -> MessageResponse:
    """
    Validate a registration code without consuming it
    
    Args:
        code: Registration code to validate
        
    Returns:
        MessageResponse indicating if code is valid
    """
    settings = get_settings()
    
    if not settings.ENABLE_REGISTRATION:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Registration is currently disabled"
        )
    
    if not settings.REQUIRE_REGISTRATION_CODE:
        return MessageResponse(
            success=True,
            message="Registration codes are not required"
        )
    
    try:
        await registration_service.initialize()
        is_valid = await registration_service.validate_registration_code(
            code,
            db_connection=db,
            check_only=True  # Don't consume the code
        )
        
        if is_valid:
            return MessageResponse(
                success=True,
                message="Registration code is valid"
            )
        else:
            return MessageResponse(
                success=False,
                message="Registration code is invalid or expired"
            )
            
    except Exception as e:
        logger.error(f"Error validating registration code: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Error validating registration code"
        )


@router.get(
    "/check-availability",
    response_model=MessageResponse,
    summary="Check username/email availability"
)
async def check_availability(
    username: Optional[str] = None,
    email: Optional[str] = None,
    db=Depends(get_db_transaction)
) -> MessageResponse:
    """
    Check if a username or email is available for registration
    
    Args:
        username: Username to check
        email: Email to check
        
    Returns:
        MessageResponse indicating availability
    """
    if not username and not email:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Must provide username or email to check"
        )
    
    try:
        if username:
            # Validate username format
            validator = UserCreateRequest.__fields__['username'].validator
            try:
                validator(username)
            except ValueError as e:
                return MessageResponse(
                    success=False,
                    message=str(e)
                )
            
            existing = await db.fetchone(
                "SELECT id FROM users WHERE username = ?",
                username
            )
            
            if existing:
                return MessageResponse(
                    success=False,
                    message="Username is already taken"
                )
        
        if email:
            # Normalize email
            email = email.lower()
            
            existing = await db.fetchone(
                "SELECT id FROM users WHERE email = ?",
                email
            )
            
            if existing:
                return MessageResponse(
                    success=False,
                    message="Email is already registered"
                )
        
        return MessageResponse(
            success=True,
            message="Available for registration"
        )
        
    except Exception as e:
        logger.error(f"Error checking availability: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Error checking availability"
        )


#
# End of register.py
#######################################################################################################################