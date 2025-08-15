# auth.py
# Description: Authentication endpoints for user login, logout, refresh, and registration
#
# Imports
from typing import Dict, Any, Optional
from datetime import datetime
#
# 3rd-party imports
from fastapi import APIRouter, Depends, HTTPException, status, Request, Response
from fastapi.security import OAuth2PasswordRequestForm
from loguru import logger
#
# Local imports
from tldw_Server_API.app.api.v1.schemas.auth_schemas import (
    LoginRequest,
    TokenResponse,
    RefreshTokenRequest,
    RegisterRequest,
    RegistrationResponse,
    MessageResponse,
    UserResponse
)
from tldw_Server_API.app.api.v1.API_Deps.auth_deps import (
    get_db_transaction,
    get_password_service_dep,
    get_jwt_service_dep,
    get_session_manager_dep,
    get_rate_limiter_dep,
    get_registration_service_dep,
    get_current_user,
    get_current_active_user,
    check_auth_rate_limit
)
from tldw_Server_API.app.core.AuthNZ.database import get_db_pool
from tldw_Server_API.app.core.AuthNZ.password_service import PasswordService
from tldw_Server_API.app.core.AuthNZ.jwt_service import JWTService, get_jwt_service
from tldw_Server_API.app.core.AuthNZ.session_manager import SessionManager
from tldw_Server_API.app.core.AuthNZ.rate_limiter import RateLimiter
from tldw_Server_API.app.services.registration_service import RegistrationService
from tldw_Server_API.app.core.AuthNZ.settings import Settings, get_settings
from tldw_Server_API.app.services.audit_service import get_audit_service, AuditAction
from tldw_Server_API.app.core.AuthNZ.exceptions import (
    AuthenticationError,
    InvalidCredentialsError,
    UserNotFoundError,
    AccountInactiveError,
    InvalidTokenError,
    TokenExpiredError,
    RegistrationError,
    DuplicateUserError,
    WeakPasswordError,
    InvalidRegistrationCodeError
)

#######################################################################################################################
#
# Router Configuration

router = APIRouter(
    prefix="/auth",
    tags=["authentication"],
    responses={404: {"description": "Not found"}}
)


#######################################################################################################################
#
# Login Endpoint

@router.post("/login", response_model=TokenResponse, dependencies=[Depends(check_auth_rate_limit)])
async def login(
    request: Request,
    form_data: OAuth2PasswordRequestForm = Depends(),
    db=Depends(get_db_transaction),
    password_service: PasswordService = Depends(get_password_service_dep),
    session_manager: SessionManager = Depends(get_session_manager_dep),
    rate_limiter: RateLimiter = Depends(get_rate_limiter_dep),
    settings: Settings = Depends(get_settings)
) -> TokenResponse:
    """
    OAuth2 compatible login endpoint
    
    Authenticates user with username/password and returns JWT tokens.
    Compatible with OAuth2PasswordRequestForm for standard OAuth2 clients.
    
    Args:
        form_data: OAuth2 form with username and password
        
    Returns:
        TokenResponse with access_token, refresh_token, and expiry
        
    Raises:
        HTTPException: 401 if credentials invalid, 403 if account inactive
    """
    try:
        # Get client info
        client_ip = request.client.host if request.client else "unknown"
        user_agent = request.headers.get("User-Agent", "Unknown")
        logger.info(f"Login attempt for user: {form_data.username} from IP: {client_ip}")
        
        # Get audit service
        audit_service = await get_audit_service()
        
        # Fetch user from database
        user = None
        if hasattr(db, 'fetchrow'):
            # PostgreSQL
            user = await db.fetchrow(
                "SELECT * FROM users WHERE lower(username) = $1 OR lower(email) = $1",
                form_data.username.lower()
            )
        else:
            # SQLite
            cursor = await db.execute(
                "SELECT * FROM users WHERE lower(username) = ? OR lower(email) = ?",
                (form_data.username.lower(), form_data.username.lower())
            )
            user = await cursor.fetchone()
        
        # Check if user exists
        if not user:
            # Log failed attempt
            logger.warning(f"Failed login: User not found - {form_data.username}")
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Incorrect username or password",
                headers={"WWW-Authenticate": "Bearer"}
            )
        
        # Convert to dict if needed
        if not isinstance(user, dict):
            if hasattr(user, 'keys'):
                user = dict(user)
            else:
                # SQLite returns tuple
                columns = ['id', 'uuid', 'username', 'email', 'password_hash', 'role',
                          'is_active', 'is_verified', 'created_at', 'updated_at',
                          'last_login', 'storage_quota_mb', 'storage_used_mb']
                user = dict(zip(columns[:len(user)], user))
        
        # Verify password
        is_valid, needs_rehash = password_service.verify_password(form_data.password, user['password_hash'])
        if not is_valid:
            # Log failed attempt
            logger.warning(f"Failed login: Invalid password for user {user['username']}")
            
            # Audit log failed login
            await audit_service.log_login(
                user_id=user['id'],
                username=user['username'],
                ip_address=client_ip,
                user_agent=user_agent,
                success=False
            )
            
            # TODO: Implement failed attempt tracking in rate limiter
            # await rate_limiter.record_failed_attempt(client_ip, "login")
            
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Incorrect username or password",
                headers={"WWW-Authenticate": "Bearer"}
            )
        
        # Check if account is active
        if not user['is_active']:
            logger.warning(f"Failed login: Inactive account - {user['username']}")
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Account is inactive. Please contact support."
            )
        
        # If password needs rehashing, update it
        if needs_rehash:
            new_hash = password_service.hash_password(form_data.password)
            if hasattr(db, 'execute'):
                # PostgreSQL
                await db.execute(
                    "UPDATE users SET password_hash = $1 WHERE id = $2",
                    new_hash, user['id']
                )
            else:
                # SQLite
                await db.execute(
                    "UPDATE users SET password_hash = ? WHERE id = ?",
                    (new_hash, user['id'])
                )
                await db.commit()
            logger.info(f"Updated password hash for user {user['username']} with new parameters")
        
        # Create session first to get session_id
        user_agent = request.headers.get("User-Agent", "Unknown")
        
        # Generate tokens based on auth mode
        if settings.AUTH_MODE == "single_user":
            # For single-user mode, return simple tokens
            access_token = f"single-user-token-{user['id']}"
            refresh_token = f"single-user-refresh-{user['id']}"
            
            # Create session with tokens
            session_info = await session_manager.create_session(
                user_id=user['id'],
                access_token=access_token,
                refresh_token=refresh_token,
                ip_address=client_ip,
                user_agent=user_agent
            )
        else:
            # For multi-user mode, create a temporary session first
            # Use unique placeholders to avoid duplicate token hash constraints
            import secrets
            temp_access = f"temp_access_{secrets.token_urlsafe(16)}"
            temp_refresh = f"temp_refresh_{secrets.token_urlsafe(16)}"
            
            temp_session_info = await session_manager.create_session(
                user_id=user['id'],
                access_token=temp_access,  # Will update with actual token
                refresh_token=temp_refresh,  # Will update with actual token
                ip_address=client_ip,
                user_agent=user_agent
            )
            
            session_id = temp_session_info['session_id']
            
            # Create JWT tokens with session_id
            jwt_service = get_jwt_service()
            access_token = jwt_service.create_access_token(
                user_id=user['id'],
                username=user['username'],
                role=user['role'],
                additional_claims={"session_id": session_id}
            )
            
            refresh_token = jwt_service.create_refresh_token(
                user_id=user['id'],
                username=user['username'],
                additional_claims={"session_id": session_id}
            )
            
            # Update session with actual tokens
            await session_manager.update_session_tokens(
                session_id=session_id,
                access_token=access_token,
                refresh_token=refresh_token
            )
            
            session_info = temp_session_info
        
        # Update last login time
        if hasattr(db, 'execute'):
            # PostgreSQL
            await db.execute(
                "UPDATE users SET last_login = $1 WHERE id = $2",
                datetime.utcnow(), user['id']
            )
        else:
            # SQLite
            await db.execute(
                "UPDATE users SET last_login = ? WHERE id = ?",
                (datetime.utcnow().isoformat(), user['id'])
            )
            await db.commit()
        
        # Log successful login
        logger.info(f"Successful login for user: {user['username']} (ID: {user['id']})")
        
        # Audit log successful login
        await audit_service.log_login(
            user_id=user['id'],
            username=user['username'],
            ip_address=client_ip,
            user_agent=user_agent,
            success=True
        )
        
        return TokenResponse(
            access_token=access_token,
            refresh_token=refresh_token,
            token_type="bearer",
            expires_in=settings.ACCESS_TOKEN_EXPIRE_MINUTES * 60
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Login error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An error occurred during login"
        )


#######################################################################################################################
#
# Logout Endpoint

@router.post("/logout", response_model=MessageResponse)
async def logout(
    current_user: Dict[str, Any] = Depends(get_current_user),
    session_manager: SessionManager = Depends(get_session_manager_dep),
    jwt_service: JWTService = Depends(get_jwt_service_dep)
) -> MessageResponse:
    """
    Logout current user and invalidate tokens
    
    Invalidates the current session and blacklists the tokens.
    
    Returns:
        MessageResponse confirming logout
    """
    try:
        # Get session from current token
        # Note: We'll need to pass session_id in JWT payload
        # For now, invalidate all sessions for the user
        await session_manager.revoke_all_user_sessions(current_user['id'])
        
        logger.info(f"User logged out: {current_user['username']} (ID: {current_user['id']})")
        
        return MessageResponse(
            message="Successfully logged out",
            details={"user_id": current_user['id']}
        )
        
    except Exception as e:
        logger.error(f"Logout error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An error occurred during logout"
        )


#######################################################################################################################
#
# Token Refresh Endpoint

@router.post("/refresh", response_model=TokenResponse)
async def refresh_token(
    request: RefreshTokenRequest,
    session_manager: SessionManager = Depends(get_session_manager_dep),
    db=Depends(get_db_transaction),
    settings: Settings = Depends(get_settings)
) -> TokenResponse:
    """
    Refresh access token using refresh token
    
    Args:
        request: RefreshTokenRequest with refresh token
        
    Returns:
        TokenResponse with new access_token
        
    Raises:
        HTTPException: 401 if refresh token invalid or expired
    """
    try:
        # Handle based on auth mode
        if settings.AUTH_MODE == "single_user":
            # Simple token validation for single-user mode
            if not request.refresh_token.startswith("single-user-refresh-"):
                raise InvalidTokenError("Invalid refresh token format")
            user_id = int(request.refresh_token.split("-")[-1])
        else:
            # JWT validation for multi-user mode
            jwt_service = get_jwt_service()
            payload = jwt_service.decode_refresh_token(request.refresh_token)
            
            # Check if token is blacklisted
            if await session_manager.is_token_blacklisted(request.refresh_token):
                raise InvalidTokenError("Refresh token has been revoked")
            
            # JWT standard uses 'sub' for subject (user ID)
            user_id = payload.get("sub") or payload.get("user_id")
            if not user_id:
                raise InvalidTokenError("Invalid refresh token payload")
            
            # Convert to int if it's a string
            try:
                user_id = int(user_id)
            except (ValueError, TypeError):
                raise InvalidTokenError("Invalid user ID in refresh token")
        
        # Fetch user
        user = None
        if hasattr(db, 'fetchrow'):
            # PostgreSQL
            user = await db.fetchrow(
                "SELECT * FROM users WHERE id = $1 AND is_active = $2",
                user_id, True
            )
        else:
            # SQLite
            cursor = await db.execute(
                "SELECT * FROM users WHERE id = ? AND is_active = ?",
                (user_id, 1)
            )
            user = await cursor.fetchone()
        
        if not user:
            raise UserNotFoundError(f"User {user_id}")
        
        # Convert to dict
        if not isinstance(user, dict):
            if hasattr(user, 'keys'):
                user = dict(user)
            else:
                columns = ['id', 'uuid', 'username', 'email', 'password_hash', 'role']
                user = dict(zip(columns[:len(user)], user))
        
        # Generate new access token based on auth mode
        if settings.AUTH_MODE == "single_user":
            new_access_token = f"single-user-token-{user['id']}"
        else:
            jwt_service = get_jwt_service()
            new_access_token = jwt_service.create_access_token(
                user_id=user['id'],
                username=user['username'],
                role=user['role']
            )
        
        # Keep the same refresh token
        # Optionally, you could rotate refresh tokens here
        
        logger.info(f"Token refreshed for user: {user['username']} (ID: {user['id']})")
        
        return TokenResponse(
            access_token=new_access_token,
            refresh_token=request.refresh_token,
            token_type="bearer",
            expires_in=settings.ACCESS_TOKEN_EXPIRE_MINUTES * 60
        )
        
    except (TokenExpiredError, InvalidTokenError) as e:
        logger.warning(f"Token refresh failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=str(e),
            headers={"WWW-Authenticate": "Bearer"}
        )
    except Exception as e:
        logger.error(f"Token refresh error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An error occurred during token refresh"
        )


#######################################################################################################################
#
# Registration Endpoint

@router.post("/register", response_model=RegistrationResponse, dependencies=[Depends(check_auth_rate_limit)])
async def register(
    request: RegisterRequest,
    registration_service: RegistrationService = Depends(get_registration_service_dep)
) -> RegistrationResponse:
    """
    Register a new user
    
    Creates a new user account with the provided credentials.
    May require a registration code if configured.
    
    Args:
        request: RegisterRequest with user details
        
    Returns:
        RegistrationResponse with user information
        
    Raises:
        HTTPException: 400 if validation fails, 409 if user exists
    """
    try:
        # Register the user
        user_info = await registration_service.register_user(
            username=request.username,
            email=request.email,
            password=request.password,
            registration_code=request.registration_code
        )
        
        logger.info(f"New user registered: {user_info['username']} (ID: {user_info['user_id']})")
        
        return RegistrationResponse(
            message="Registration successful",
            user_id=user_info['user_id'],
            username=user_info['username'],
            email=user_info['email'],
            requires_verification=not user_info['is_verified']
        )
        
    except DuplicateUserError as e:
        logger.warning(f"Registration failed - duplicate user: {e}")
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=str(e)
        )
    except WeakPasswordError as e:
        logger.warning(f"Registration failed - weak password: {e}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    except InvalidRegistrationCodeError as e:
        logger.warning(f"Registration failed - invalid code: {e}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    except RegistrationError as e:
        logger.error(f"Registration error: {e}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    except Exception as e:
        # Check if it's a transaction error wrapping a duplicate user error
        error_msg = str(e).lower()
        if "username already exists" in error_msg or "email already exists" in error_msg:
            logger.warning(f"Registration failed - duplicate user (wrapped): {e}")
            if "username" in error_msg:
                detail = "Username already exists"
            else:
                detail = "Email already exists"
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail=detail
            )
        
        logger.error(f"Unexpected registration error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An error occurred during registration"
        )


#######################################################################################################################
#
# Test Endpoint

@router.get("/me", response_model=UserResponse)
async def get_current_user_info(
    current_user: Dict[str, Any] = Depends(get_current_active_user)
) -> UserResponse:
    """
    Get current user information
    
    Returns:
        UserResponse with current user details
    """
    # Ensure UUID is a string
    user_uuid = current_user.get('uuid')
    if user_uuid and not isinstance(user_uuid, str):
        user_uuid = str(user_uuid)
    
    return UserResponse(
        id=current_user['id'],
        uuid=user_uuid or '',  # Provide empty string if missing
        username=current_user['username'],
        email=current_user['email'],
        role=current_user['role'],
        is_active=current_user.get('is_active', True),
        is_verified=current_user.get('is_verified', True),
        created_at=current_user.get('created_at', datetime.utcnow()),
        last_login=current_user.get('last_login'),
        storage_quota_mb=current_user.get('storage_quota_mb', 1000),
        storage_used_mb=current_user.get('storage_used_mb', 0.0)
    )


#
# End of auth.py
#######################################################################################################################