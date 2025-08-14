# admin.py
# Description: Admin endpoints for user management, registration codes, and system administration
#
# Imports
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta
import secrets
import string
#
# 3rd-party imports
from fastapi import APIRouter, Depends, HTTPException, status, Query
from loguru import logger
#
# Local imports
from tldw_Server_API.app.api.v1.schemas.admin_schemas import (
    UserListResponse,
    UserUpdateRequest,
    RegistrationCodeRequest,
    RegistrationCodeResponse,
    RegistrationCodeListResponse,
    SystemStatsResponse,
    AuditLogResponse,
    UserQuotaUpdateRequest
)
from tldw_Server_API.app.api.v1.API_Deps.auth_deps import (
    require_admin,
    get_db_transaction,
    get_storage_service_dep
)
from tldw_Server_API.app.core.AuthNZ.settings import Settings, get_settings
from tldw_Server_API.app.services.storage_quota_service import StorageQuotaService
from tldw_Server_API.app.core.AuthNZ.exceptions import (
    UserNotFoundError,
    DuplicateUserError,
    QuotaExceededError
)

#######################################################################################################################
#
# Router Configuration

router = APIRouter(
    prefix="/admin",
    tags=["admin"],
    dependencies=[Depends(require_admin)],  # All endpoints require admin role
    responses={403: {"description": "Not authorized"}}
)


#######################################################################################################################
#
# User Management Endpoints

@router.get("/users", response_model=UserListResponse)
async def list_users(
    page: int = Query(1, ge=1),
    limit: int = Query(20, ge=1, le=100),
    role: Optional[str] = None,
    is_active: Optional[bool] = None,
    search: Optional[str] = None,
    db=Depends(get_db_transaction)
) -> UserListResponse:
    """
    List all users with pagination and filters
    
    Args:
        page: Page number (1-based)
        limit: Items per page
        role: Filter by role
        is_active: Filter by active status
        search: Search in username/email
        
    Returns:
        Paginated list of users
    """
    try:
        offset = (page - 1) * limit
        
        # Build query conditions
        conditions = []
        params = []
        param_count = 0
        
        if role:
            param_count += 1
            conditions.append(f"role = ${param_count}" if hasattr(db, 'fetchrow') else "role = ?")
            params.append(role)
        
        if is_active is not None:
            param_count += 1
            conditions.append(f"is_active = ${param_count}" if hasattr(db, 'fetchrow') else "is_active = ?")
            params.append(is_active)
        
        if search:
            param_count += 1
            search_pattern = f"%{search}%"
            if hasattr(db, 'fetchrow'):
                conditions.append(f"(username ILIKE ${param_count} OR email ILIKE ${param_count})")
            else:
                conditions.append("(username LIKE ? OR email LIKE ?)")
                params.append(search_pattern)  # Add twice for SQLite
            params.append(search_pattern)
        
        where_clause = " WHERE " + " AND ".join(conditions) if conditions else ""
        
        # Get total count
        if hasattr(db, 'fetchrow'):
            # PostgreSQL
            count_query = f"SELECT COUNT(*) FROM users{where_clause}"
            total = await db.fetchval(count_query, *params)
            
            # Get users
            query = f"""
                SELECT id, uuid, username, email, role, is_active, is_verified,
                       created_at, last_login, storage_quota_mb, storage_used_mb
                FROM users{where_clause}
                ORDER BY created_at DESC
                LIMIT ${param_count + 1} OFFSET ${param_count + 2}
            """
            params.extend([limit, offset])
            rows = await db.fetch(query, *params)
        else:
            # SQLite
            count_query = f"SELECT COUNT(*) FROM users{where_clause}"
            cursor = await db.execute(count_query, params)
            total = (await cursor.fetchone())[0]
            
            # Get users
            query = f"""
                SELECT id, uuid, username, email, role, is_active, is_verified,
                       created_at, last_login, storage_quota_mb, storage_used_mb
                FROM users{where_clause}
                ORDER BY created_at DESC
                LIMIT ? OFFSET ?
            """
            params.extend([limit, offset])
            cursor = await db.execute(query, params)
            rows = await cursor.fetchall()
        
        # Convert to list of dicts
        users = []
        for row in rows:
            if isinstance(row, dict):
                users.append(row)
            else:
                user_dict = {
                    "id": row[0],
                    "uuid": row[1],
                    "username": row[2],
                    "email": row[3],
                    "role": row[4],
                    "is_active": bool(row[5]),
                    "is_verified": bool(row[6]),
                    "created_at": row[7],
                    "last_login": row[8],
                    "storage_quota_mb": row[9],
                    "storage_used_mb": float(row[10]) if row[10] else 0
                }
                users.append(user_dict)
        
        return UserListResponse(
            users=users,
            total=total,
            page=page,
            limit=limit,
            pages=(total + limit - 1) // limit
        )
        
    except Exception as e:
        logger.error(f"Failed to list users: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve users"
        )


@router.get("/users/{user_id}")
async def get_user_details(
    user_id: int,
    db=Depends(get_db_transaction)
) -> Dict[str, Any]:
    """
    Get detailed information about a specific user
    
    Args:
        user_id: User ID
        
    Returns:
        User details including all fields
    """
    try:
        if hasattr(db, 'fetchrow'):
            # PostgreSQL
            user = await db.fetchrow(
                "SELECT * FROM users WHERE id = $1",
                user_id
            )
        else:
            # SQLite
            cursor = await db.execute(
                "SELECT * FROM users WHERE id = ?",
                (user_id,)
            )
            user = await cursor.fetchone()
        
        if not user:
            raise UserNotFoundError(f"User {user_id}")
        
        # Convert to dict
        if not isinstance(user, dict):
            columns = ['id', 'uuid', 'username', 'email', 'password_hash', 'role',
                      'is_active', 'is_verified', 'is_locked', 'locked_until',
                      'failed_login_attempts', 'created_at', 'updated_at',
                      'last_login', 'email_verified_at', 'password_changed_at',
                      'preferences', 'storage_quota_mb', 'storage_used_mb']
            user = dict(zip(columns[:len(user)], user))
        
        # Remove sensitive fields
        user.pop('password_hash', None)
        
        return user
        
    except UserNotFoundError:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"User {user_id} not found"
        )
    except Exception as e:
        logger.error(f"Failed to get user {user_id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve user details"
        )


@router.put("/users/{user_id}")
async def update_user(
    user_id: int,
    request: UserUpdateRequest,
    db=Depends(get_db_transaction)
) -> Dict[str, str]:
    """
    Update user information
    
    Args:
        user_id: User ID
        request: Update request with fields to modify
        
    Returns:
        Success message
    """
    try:
        # Build update query dynamically
        updates = []
        params = []
        param_count = 0
        
        if request.email is not None:
            param_count += 1
            updates.append(f"email = ${param_count}" if hasattr(db, 'fetchrow') else "email = ?")
            params.append(request.email)
        
        if request.role is not None:
            param_count += 1
            updates.append(f"role = ${param_count}" if hasattr(db, 'fetchrow') else "role = ?")
            params.append(request.role)
        
        if request.is_active is not None:
            param_count += 1
            updates.append(f"is_active = ${param_count}" if hasattr(db, 'fetchrow') else "is_active = ?")
            params.append(request.is_active)
        
        if request.is_verified is not None:
            param_count += 1
            updates.append(f"is_verified = ${param_count}" if hasattr(db, 'fetchrow') else "is_verified = ?")
            params.append(request.is_verified)
        
        if request.is_locked is not None:
            param_count += 1
            updates.append(f"is_locked = ${param_count}" if hasattr(db, 'fetchrow') else "is_locked = ?")
            params.append(request.is_locked)
            
            if not request.is_locked:
                # Unlock user - reset failed attempts
                param_count += 1
                updates.append(f"failed_login_attempts = ${param_count}" if hasattr(db, 'fetchrow') else "failed_login_attempts = ?")
                params.append(0)
                updates.append("locked_until = NULL")
        
        if request.storage_quota_mb is not None:
            param_count += 1
            updates.append(f"storage_quota_mb = ${param_count}" if hasattr(db, 'fetchrow') else "storage_quota_mb = ?")
            params.append(request.storage_quota_mb)
        
        if not updates:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="No fields to update"
            )
        
        # Add updated_at
        param_count += 1
        if hasattr(db, 'fetchrow'):
            updates.append("updated_at = CURRENT_TIMESTAMP")
            params.append(user_id)
            query = f"UPDATE users SET {', '.join(updates)} WHERE id = ${param_count}"
        else:
            updates.append("updated_at = datetime('now')")
            params.append(user_id)
            query = f"UPDATE users SET {', '.join(updates)} WHERE id = ?"
        
        # Execute update
        if hasattr(db, 'fetchrow'):
            await db.execute(query, *params)
        else:
            await db.execute(query, params)
            await db.commit()
        
        logger.info(f"Admin updated user {user_id}")
        
        return {"message": f"User {user_id} updated successfully"}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to update user {user_id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to update user"
        )


@router.delete("/users/{user_id}")
async def delete_user(
    user_id: int,
    current_user: Dict[str, Any] = Depends(require_admin),
    db=Depends(get_db_transaction)
) -> Dict[str, str]:
    """
    Delete a user (soft delete by default)
    
    Args:
        user_id: User ID to delete
        
    Returns:
        Success message
    """
    try:
        # Prevent self-deletion
        if user_id == current_user["id"]:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Cannot delete your own account"
            )
        
        # Soft delete - just mark as inactive
        if hasattr(db, 'fetchrow'):
            # PostgreSQL
            result = await db.execute(
                "UPDATE users SET is_active = FALSE, updated_at = CURRENT_TIMESTAMP WHERE id = $1",
                user_id
            )
        else:
            # SQLite
            await db.execute(
                "UPDATE users SET is_active = 0, updated_at = datetime('now') WHERE id = ?",
                (user_id,)
            )
            await db.commit()
        
        logger.info(f"Admin soft-deleted user {user_id}")
        
        return {"message": f"User {user_id} has been deactivated"}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to delete user {user_id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to delete user"
        )


#######################################################################################################################
#
# Registration Code Management

@router.post("/registration-codes", response_model=RegistrationCodeResponse)
async def create_registration_code(
    request: RegistrationCodeRequest,
    current_user: Dict[str, Any] = Depends(require_admin),
    db=Depends(get_db_transaction)
) -> RegistrationCodeResponse:
    """
    Create a new registration code
    
    Args:
        request: Registration code configuration
        
    Returns:
        Created registration code details
    """
    try:
        # Generate secure code
        code = ''.join(secrets.choice(string.ascii_uppercase + string.digits) for _ in range(24))
        
        # Calculate expiration
        expires_at = datetime.utcnow() + timedelta(days=request.expiry_days)
        
        if hasattr(db, 'fetchrow'):
            # PostgreSQL
            result = await db.fetchrow("""
                INSERT INTO registration_codes 
                (code, max_uses, expires_at, created_by, role_to_grant, metadata)
                VALUES ($1, $2, $3, $4, $5, $6)
                RETURNING id, code, max_uses, times_used, expires_at, created_at, role_to_grant
            """, code, request.max_uses, expires_at, current_user["id"],
                request.role_to_grant, __import__('json').dumps(request.metadata or {}))
        else:
            # SQLite
            cursor = await db.execute("""
                INSERT INTO registration_codes 
                (code, max_uses, expires_at, created_by, role_to_grant, metadata)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (code, request.max_uses, expires_at.isoformat(), current_user["id"],
                  request.role_to_grant, __import__('json').dumps(request.metadata or {})))
            
            code_id = cursor.lastrowid
            await db.commit()
            
            # Fetch the created code
            cursor = await db.execute(
                "SELECT * FROM registration_codes WHERE id = ?",
                (code_id,)
            )
            result = await cursor.fetchone()
        
        logger.info(f"Admin created registration code: {code[:8]}...")
        
        return RegistrationCodeResponse(
            id=result[0] if isinstance(result, tuple) else result['id'],
            code=code,
            max_uses=request.max_uses,
            times_used=0,
            expires_at=expires_at,
            created_at=datetime.utcnow(),
            role_to_grant=request.role_to_grant
        )
        
    except Exception as e:
        logger.error(f"Failed to create registration code: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to create registration code"
        )


@router.get("/registration-codes", response_model=RegistrationCodeListResponse)
async def list_registration_codes(
    include_expired: bool = Query(False),
    db=Depends(get_db_transaction)
) -> RegistrationCodeListResponse:
    """
    List all registration codes
    
    Args:
        include_expired: Include expired codes in the list
        
    Returns:
        List of registration codes
    """
    try:
        if hasattr(db, 'fetchrow'):
            # PostgreSQL
            if include_expired:
                query = """
                    SELECT id, code, max_uses, times_used, expires_at, 
                           created_at, created_by, role_to_grant
                    FROM registration_codes
                    ORDER BY created_at DESC
                """
            else:
                query = """
                    SELECT id, code, max_uses, times_used, expires_at,
                           created_at, created_by, role_to_grant
                    FROM registration_codes
                    WHERE expires_at > CURRENT_TIMESTAMP
                    ORDER BY created_at DESC
                """
            rows = await db.fetch(query)
        else:
            # SQLite
            if include_expired:
                query = """
                    SELECT id, code, max_uses, times_used, expires_at,
                           created_at, created_by, role_to_grant
                    FROM registration_codes
                    ORDER BY created_at DESC
                """
            else:
                query = """
                    SELECT id, code, max_uses, times_used, expires_at,
                           created_at, created_by, role_to_grant
                    FROM registration_codes
                    WHERE datetime(expires_at) > datetime('now')
                    ORDER BY created_at DESC
                """
            cursor = await db.execute(query)
            rows = await cursor.fetchall()
        
        codes = []
        for row in rows:
            if isinstance(row, dict):
                codes.append(row)
            else:
                code_dict = {
                    "id": row[0],
                    "code": row[1],
                    "max_uses": row[2],
                    "times_used": row[3],
                    "expires_at": row[4],
                    "created_at": row[5],
                    "created_by": row[6],
                    "role_to_grant": row[7],
                    "is_valid": row[3] < row[2] and (
                        row[4] > datetime.utcnow() if isinstance(row[4], datetime)
                        else datetime.fromisoformat(row[4]) > datetime.utcnow()
                    )
                }
                codes.append(code_dict)
        
        return RegistrationCodeListResponse(codes=codes)
        
    except Exception as e:
        logger.error(f"Failed to list registration codes: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve registration codes"
        )


@router.delete("/registration-codes/{code_id}")
async def delete_registration_code(
    code_id: int,
    db=Depends(get_db_transaction)
) -> Dict[str, str]:
    """
    Delete a registration code
    
    Args:
        code_id: Registration code ID
        
    Returns:
        Success message
    """
    try:
        if hasattr(db, 'fetchrow'):
            # PostgreSQL
            await db.execute(
                "DELETE FROM registration_codes WHERE id = $1",
                code_id
            )
        else:
            # SQLite
            await db.execute(
                "DELETE FROM registration_codes WHERE id = ?",
                (code_id,)
            )
            await db.commit()
        
        logger.info(f"Admin deleted registration code {code_id}")
        
        return {"message": f"Registration code {code_id} deleted"}
        
    except Exception as e:
        logger.error(f"Failed to delete registration code {code_id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to delete registration code"
        )


#######################################################################################################################
#
# System Statistics and Monitoring

@router.get("/stats", response_model=SystemStatsResponse)
async def get_system_stats(
    db=Depends(get_db_transaction)
) -> SystemStatsResponse:
    """
    Get system statistics
    
    Returns:
        System-wide statistics including user counts, storage usage, etc.
    """
    try:
        stats = {}
        
        if hasattr(db, 'fetchrow'):
            # PostgreSQL
            # User stats
            user_stats = await db.fetchrow("""
                SELECT 
                    COUNT(*) as total_users,
                    COUNT(*) FILTER (WHERE is_active = TRUE) as active_users,
                    COUNT(*) FILTER (WHERE is_verified = TRUE) as verified_users,
                    COUNT(*) FILTER (WHERE role = 'admin') as admin_users,
                    COUNT(*) FILTER (WHERE created_at > CURRENT_TIMESTAMP - INTERVAL '30 days') as new_users_30d
                FROM users
            """)
            
            # Storage stats
            storage_stats = await db.fetchrow("""
                SELECT 
                    SUM(storage_used_mb) as total_used_mb,
                    SUM(storage_quota_mb) as total_quota_mb,
                    AVG(storage_used_mb) as avg_used_mb,
                    MAX(storage_used_mb) as max_used_mb
                FROM users
                WHERE is_active = TRUE
            """)
            
            # Session stats
            session_stats = await db.fetchrow("""
                SELECT 
                    COUNT(*) as active_sessions,
                    COUNT(DISTINCT user_id) as unique_users
                FROM sessions
                WHERE is_active = TRUE AND expires_at > CURRENT_TIMESTAMP
            """)
            
        else:
            # SQLite
            cursor = await db.execute("""
                SELECT 
                    COUNT(*) as total_users,
                    SUM(CASE WHEN is_active = 1 THEN 1 ELSE 0 END) as active_users,
                    SUM(CASE WHEN is_verified = 1 THEN 1 ELSE 0 END) as verified_users,
                    SUM(CASE WHEN role = 'admin' THEN 1 ELSE 0 END) as admin_users,
                    SUM(CASE WHEN datetime(created_at) > datetime('now', '-30 days') THEN 1 ELSE 0 END) as new_users_30d
                FROM users
            """)
            user_stats = await cursor.fetchone()
            
            cursor = await db.execute("""
                SELECT 
                    SUM(storage_used_mb) as total_used_mb,
                    SUM(storage_quota_mb) as total_quota_mb,
                    AVG(storage_used_mb) as avg_used_mb,
                    MAX(storage_used_mb) as max_used_mb
                FROM users
                WHERE is_active = 1
            """)
            storage_stats = await cursor.fetchone()
            
            cursor = await db.execute("""
                SELECT 
                    COUNT(*) as active_sessions,
                    COUNT(DISTINCT user_id) as unique_users
                FROM sessions
                WHERE is_active = 1 AND datetime(expires_at) > datetime('now')
            """)
            session_stats = await cursor.fetchone()
        
        # Convert to response model
        return SystemStatsResponse(
            users={
                "total": user_stats[0] or 0,
                "active": user_stats[1] or 0,
                "verified": user_stats[2] or 0,
                "admins": user_stats[3] or 0,
                "new_last_30d": user_stats[4] or 0
            },
            storage={
                "total_used_mb": float(storage_stats[0] or 0),
                "total_quota_mb": float(storage_stats[1] or 0),
                "average_used_mb": float(storage_stats[2] or 0),
                "max_used_mb": float(storage_stats[3] or 0)
            },
            sessions={
                "active": session_stats[0] or 0,
                "unique_users": session_stats[1] or 0
            }
        )
        
    except Exception as e:
        logger.error(f"Failed to get system stats: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve system statistics"
        )


@router.get("/audit-log", response_model=AuditLogResponse)
async def get_audit_log(
    user_id: Optional[int] = None,
    action: Optional[str] = None,
    days: int = Query(7, ge=1, le=90),
    limit: int = Query(100, ge=1, le=1000),
    db=Depends(get_db_transaction)
) -> AuditLogResponse:
    """
    Get audit log entries
    
    Args:
        user_id: Filter by user ID
        action: Filter by action type
        days: Number of days to look back
        limit: Maximum entries to return
        
    Returns:
        Audit log entries
    """
    try:
        conditions = []
        params = []
        param_count = 0
        
        if user_id:
            param_count += 1
            conditions.append(f"user_id = ${param_count}" if hasattr(db, 'fetchrow') else "user_id = ?")
            params.append(user_id)
        
        if action:
            param_count += 1
            conditions.append(f"action = ${param_count}" if hasattr(db, 'fetchrow') else "action = ?")
            params.append(action)
        
        # Date filter
        if hasattr(db, 'fetchrow'):
            conditions.append(f"created_at > CURRENT_TIMESTAMP - INTERVAL '{days} days'")
        else:
            conditions.append("datetime(created_at) > datetime('now', ? || ' days')")
            params.append(f"-{days}")
        
        where_clause = " WHERE " + " AND ".join(conditions) if conditions else ""
        
        if hasattr(db, 'fetchrow'):
            # PostgreSQL
            query = f"""
                SELECT a.id, a.user_id, u.username, a.action, a.details,
                       a.ip_address, a.created_at
                FROM audit_log a
                LEFT JOIN users u ON a.user_id = u.id
                {where_clause}
                ORDER BY a.created_at DESC
                LIMIT ${param_count + 1}
            """
            params.append(limit)
            rows = await db.fetch(query, *params)
        else:
            # SQLite
            query = f"""
                SELECT a.id, a.user_id, u.username, a.action, a.details,
                       a.ip_address, a.created_at
                FROM audit_log a
                LEFT JOIN users u ON a.user_id = u.id
                {where_clause}
                ORDER BY a.created_at DESC
                LIMIT ?
            """
            params.append(limit)
            cursor = await db.execute(query, params)
            rows = await cursor.fetchall()
        
        entries = []
        for row in rows:
            if isinstance(row, dict):
                entries.append(row)
            else:
                entry = {
                    "id": row[0],
                    "user_id": row[1],
                    "username": row[2],
                    "action": row[3],
                    "details": row[4],
                    "ip_address": row[5],
                    "created_at": row[6]
                }
                entries.append(entry)
        
        return AuditLogResponse(entries=entries)
        
    except Exception as e:
        logger.error(f"Failed to get audit log: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve audit log"
        )


#
## End of admin.py
#######################################################################################################################