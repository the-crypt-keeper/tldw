# users.py
# Description: User management endpoints for profile, password, and session management
#
# Imports
from typing import Dict, Any, List, Optional
from datetime import datetime
#
# 3rd-party imports
from fastapi import APIRouter, Depends, HTTPException, status, Request
from loguru import logger
#
# Local imports
from tldw_Server_API.app.api.v1.schemas.auth_schemas import (
    UserResponse,
    PasswordChangeRequest,
    MessageResponse,
    SessionResponse,
    StorageQuotaResponse
)
from tldw_Server_API.app.api.v1.API_Deps.auth_deps import (
    get_current_active_user,
    get_db_transaction,
    get_password_service_dep,
    get_session_manager_dep,
    get_storage_service_dep
)
from tldw_Server_API.app.core.AuthNZ.password_service import PasswordService
from tldw_Server_API.app.core.AuthNZ.session_manager import SessionManager
from tldw_Server_API.app.services.storage_quota_service import StorageQuotaService
from tldw_Server_API.app.core.AuthNZ.exceptions import (
    InvalidCredentialsError,
    WeakPasswordError,
    SessionError
)

#######################################################################################################################
#
# Router Configuration

router = APIRouter(
    prefix="/users",
    tags=["users"],
    responses={404: {"description": "Not found"}}
)


#######################################################################################################################
#
# User Profile Endpoints

@router.get("/me", response_model=UserResponse)
async def get_current_user_profile(
    current_user: Dict[str, Any] = Depends(get_current_active_user)
) -> UserResponse:
    """
    Get current user profile
    
    Returns the authenticated user's profile information.
    
    Returns:
        UserResponse with user details
    """
    return UserResponse(
        id=current_user['id'],
        uuid=current_user.get('uuid', ''),
        username=current_user['username'],
        email=current_user.get('email', ''),
        role=current_user.get('role', 'user'),
        is_active=current_user.get('is_active', True),
        is_verified=current_user.get('is_verified', False),
        created_at=current_user.get('created_at', datetime.utcnow()),
        last_login=current_user.get('last_login'),
        storage_quota_mb=current_user.get('storage_quota_mb', 5120),
        storage_used_mb=current_user.get('storage_used_mb', 0.0)
    )


@router.put("/me", response_model=UserResponse)
async def update_user_profile(
    email: Optional[str] = None,
    current_user: Dict[str, Any] = Depends(get_current_active_user),
    db=Depends(get_db_transaction)
) -> UserResponse:
    """
    Update current user profile
    
    Allows users to update their email address.
    Username changes are not allowed for security reasons.
    
    Args:
        email: New email address (optional)
        
    Returns:
        Updated UserResponse
    """
    try:
        updates_made = False
        
        if email and email != current_user.get('email'):
            # Update email
            if hasattr(db, 'execute'):
                # PostgreSQL
                await db.execute(
                    "UPDATE users SET email = $1, updated_at = CURRENT_TIMESTAMP WHERE id = $2",
                    email.lower(), current_user['id']
                )
            else:
                # SQLite
                await db.execute(
                    "UPDATE users SET email = ?, updated_at = datetime('now') WHERE id = ?",
                    (email.lower(), current_user['id'])
                )
                await db.commit()
            
            updates_made = True
            current_user['email'] = email.lower()
            logger.info(f"Updated email for user {current_user['username']} (ID: {current_user['id']})")
        
        if not updates_made:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="No updates provided"
            )
        
        # Return updated user info
        return await get_current_user_profile(current_user)
        
    except Exception as e:
        logger.error(f"Failed to update user profile: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to update profile"
        )


#######################################################################################################################
#
# Password Management

@router.post("/change-password", response_model=MessageResponse)
async def change_password(
    request: PasswordChangeRequest,
    current_user: Dict[str, Any] = Depends(get_current_active_user),
    password_service: PasswordService = Depends(get_password_service_dep),
    db=Depends(get_db_transaction)
) -> MessageResponse:
    """
    Change user password
    
    Allows users to change their password by providing the current password.
    
    Args:
        request: PasswordChangeRequest with current and new passwords
        
    Returns:
        MessageResponse confirming password change
        
    Raises:
        HTTPException: 401 if current password is incorrect, 400 if new password is weak
    """
    try:
        # Verify current password
        if not password_service.verify_password(
            request.current_password, 
            current_user['password_hash']
        ):
            logger.warning(f"Failed password change attempt for user {current_user['username']}")
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Current password is incorrect"
            )
        
        # Validate new password strength
        try:
            password_service.validate_password_strength(
                request.new_password,
                current_user['username']
            )
        except WeakPasswordError as e:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=str(e)
            )
        
        # Hash new password
        new_hash = password_service.hash_password(request.new_password)
        
        # Update password in database
        if hasattr(db, 'execute'):
            # PostgreSQL
            await db.execute(
                """
                UPDATE users 
                SET password_hash = $1, 
                    password_changed_at = CURRENT_TIMESTAMP,
                    updated_at = CURRENT_TIMESTAMP
                WHERE id = $2
                """,
                new_hash, current_user['id']
            )
            
            # Add to password history
            await db.execute(
                "INSERT INTO password_history (user_id, password_hash) VALUES ($1, $2)",
                current_user['id'], new_hash
            )
        else:
            # SQLite
            await db.execute(
                """
                UPDATE users 
                SET password_hash = ?, 
                    password_changed_at = datetime('now'),
                    updated_at = datetime('now')
                WHERE id = ?
                """,
                (new_hash, current_user['id'])
            )
            
            # Add to password history
            await db.execute(
                "INSERT INTO password_history (user_id, password_hash) VALUES (?, ?)",
                (current_user['id'], new_hash)
            )
            await db.commit()
        
        logger.info(f"Password changed for user {current_user['username']} (ID: {current_user['id']})")
        
        return MessageResponse(
            message="Password changed successfully",
            details={"user_id": current_user['id']}
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to change password: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to change password"
        )


#######################################################################################################################
#
# Session Management

@router.get("/sessions", response_model=List[SessionResponse])
async def list_user_sessions(
    current_user: Dict[str, Any] = Depends(get_current_active_user),
    session_manager: SessionManager = Depends(get_session_manager_dep)
) -> List[SessionResponse]:
    """
    List all active sessions for the current user
    
    Returns:
        List of SessionResponse objects
    """
    try:
        sessions = await session_manager.get_user_sessions(current_user['id'])
        
        return [
            SessionResponse(
                id=session['id'],
                ip_address=session.get('ip_address'),
                user_agent=session.get('user_agent'),
                created_at=session['created_at'],
                last_activity=session['last_activity'],
                expires_at=session['expires_at']
            )
            for session in sessions
        ]
        
    except Exception as e:
        logger.error(f"Failed to list user sessions: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve sessions"
        )


@router.delete("/sessions/{session_id}", response_model=MessageResponse)
async def revoke_session(
    session_id: int,
    current_user: Dict[str, Any] = Depends(get_current_active_user),
    session_manager: SessionManager = Depends(get_session_manager_dep)
) -> MessageResponse:
    """
    Revoke a specific session
    
    Allows users to log out specific sessions (e.g., on other devices).
    
    Args:
        session_id: ID of the session to revoke
        
    Returns:
        MessageResponse confirming revocation
        
    Raises:
        HTTPException: 404 if session not found or doesn't belong to user
    """
    try:
        # Get session to verify ownership
        sessions = await session_manager.get_user_sessions(current_user['id'])
        session_ids = [s['id'] for s in sessions]
        
        if session_id not in session_ids:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Session not found or does not belong to current user"
            )
        
        # Revoke the session
        await session_manager.revoke_session(
            session_id,
            revoked_by=current_user['id'],
            reason="User requested revocation"
        )
        
        logger.info(f"User {current_user['username']} revoked session {session_id}")
        
        return MessageResponse(
            message="Session revoked successfully",
            details={"session_id": session_id}
        )
        
    except HTTPException:
        raise
    except SessionError as e:
        logger.error(f"Failed to revoke session: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to revoke session"
        )
    except Exception as e:
        logger.error(f"Unexpected error revoking session: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An error occurred while revoking the session"
        )


@router.post("/sessions/revoke-all", response_model=MessageResponse)
async def revoke_all_sessions(
    current_user: Dict[str, Any] = Depends(get_current_active_user),
    session_manager: SessionManager = Depends(get_session_manager_dep)
) -> MessageResponse:
    """
    Revoke all sessions for the current user
    
    Logs out the user from all devices.
    
    Returns:
        MessageResponse confirming revocation
    """
    try:
        count = await session_manager.revoke_all_user_sessions(
            current_user['id'],
            reason="User requested logout from all devices"
        )
        
        logger.info(f"User {current_user['username']} revoked all {count} sessions")
        
        return MessageResponse(
            message=f"Successfully revoked {count} sessions",
            details={"sessions_revoked": count}
        )
        
    except Exception as e:
        logger.error(f"Failed to revoke all sessions: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to revoke sessions"
        )


#######################################################################################################################
#
# Storage Management

@router.get("/storage", response_model=StorageQuotaResponse)
async def get_storage_quota(
    current_user: Dict[str, Any] = Depends(get_current_active_user),
    storage_service: StorageQuotaService = Depends(get_storage_service_dep)
) -> StorageQuotaResponse:
    """
    Get storage quota information for current user
    
    Returns:
        StorageQuotaResponse with usage details
    """
    try:
        # Get storage info from service
        storage_info = await storage_service.calculate_user_storage(
            current_user['id'],
            update_database=False  # Don't update unless explicitly requested
        )
        
        return StorageQuotaResponse(
            user_id=current_user['id'],
            storage_used_mb=storage_info['total_mb'],
            storage_quota_mb=storage_info['quota_mb'],
            available_mb=storage_info['available_mb'],
            usage_percentage=storage_info['usage_percentage']
        )
        
    except Exception as e:
        logger.error(f"Failed to get storage quota: {e}")
        # Return from database values if calculation fails
        return StorageQuotaResponse(
            user_id=current_user['id'],
            storage_used_mb=current_user.get('storage_used_mb', 0.0),
            storage_quota_mb=current_user.get('storage_quota_mb', 5120),
            available_mb=max(0, current_user.get('storage_quota_mb', 5120) - current_user.get('storage_used_mb', 0.0)),
            usage_percentage=round((current_user.get('storage_used_mb', 0.0) / current_user.get('storage_quota_mb', 5120) * 100) if current_user.get('storage_quota_mb', 5120) > 0 else 0, 1)
        )


@router.post("/storage/recalculate", response_model=StorageQuotaResponse)
async def recalculate_storage(
    current_user: Dict[str, Any] = Depends(get_current_active_user),
    storage_service: StorageQuotaService = Depends(get_storage_service_dep)
) -> StorageQuotaResponse:
    """
    Recalculate storage usage for current user
    
    Forces a recalculation of actual disk usage and updates the database.
    
    Returns:
        StorageQuotaResponse with updated usage details
    """
    try:
        # Recalculate and update database
        storage_info = await storage_service.calculate_user_storage(
            current_user['id'],
            update_database=True
        )
        
        logger.info(f"Recalculated storage for user {current_user['username']}: {storage_info['total_mb']:.2f}MB")
        
        return StorageQuotaResponse(
            user_id=current_user['id'],
            storage_used_mb=storage_info['total_mb'],
            storage_quota_mb=storage_info['quota_mb'],
            available_mb=storage_info['available_mb'],
            usage_percentage=storage_info['usage_percentage']
        )
        
    except Exception as e:
        logger.error(f"Failed to recalculate storage: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to recalculate storage"
        )


#
# End of users.py
#######################################################################################################################