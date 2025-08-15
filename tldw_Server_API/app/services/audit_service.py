# audit_service.py
# Description: Audit logging service for tracking security-relevant events
#
# Imports
from typing import Optional, Dict, Any
from datetime import datetime
import json
from enum import Enum
#
# 3rd-party imports
from loguru import logger
#
# Local imports
from tldw_Server_API.app.core.AuthNZ.database import DatabasePool, get_db_pool
from tldw_Server_API.app.core.AuthNZ.settings import Settings, get_settings

#######################################################################################################################
#
# Audit Action Types

class AuditAction(str, Enum):
    """Enumeration of audit action types"""
    # Authentication events
    USER_LOGIN = "user_login"
    USER_LOGOUT = "user_logout"
    LOGIN_FAILED = "login_failed"
    TOKEN_REFRESH = "token_refresh"
    SESSION_EXPIRED = "session_expired"
    
    # User management
    USER_REGISTERED = "user_registered"
    USER_UPDATED = "user_updated"
    USER_DELETED = "user_deleted"
    USER_ACTIVATED = "user_activated"
    USER_DEACTIVATED = "user_deactivated"
    USER_LOCKED = "user_locked"
    USER_UNLOCKED = "user_unlocked"
    
    # Password events
    PASSWORD_CHANGED = "password_changed"
    PASSWORD_RESET_REQUESTED = "password_reset_requested"
    PASSWORD_RESET_COMPLETED = "password_reset_completed"
    
    # Admin actions
    ADMIN_USER_UPDATE = "admin_user_update"
    ADMIN_USER_DELETE = "admin_user_delete"
    ADMIN_CODE_CREATED = "admin_code_created"
    ADMIN_CODE_DELETED = "admin_code_deleted"
    ADMIN_QUOTA_CHANGED = "admin_quota_changed"
    
    # Security events
    RATE_LIMIT_EXCEEDED = "rate_limit_exceeded"
    INVALID_TOKEN = "invalid_token"
    UNAUTHORIZED_ACCESS = "unauthorized_access"
    SUSPICIOUS_ACTIVITY = "suspicious_activity"
    
    # Data access
    DATA_EXPORTED = "data_exported"
    DATA_IMPORTED = "data_imported"
    DATA_DELETED = "data_deleted"
    
    # System events
    SYSTEM_STARTUP = "system_startup"
    SYSTEM_SHUTDOWN = "system_shutdown"
    CONFIG_CHANGED = "config_changed"
    MIGRATION_STARTED = "migration_started"
    MIGRATION_COMPLETED = "migration_completed"


#######################################################################################################################
#
# Audit Service Class

class AuditService:
    """Service for logging audit events to database and log files"""
    
    def __init__(self, db_pool: Optional[DatabasePool] = None, settings: Optional[Settings] = None):
        """Initialize audit service"""
        self.db_pool = db_pool
        self.settings = settings or get_settings()
        self._initialized = False
    
    async def initialize(self):
        """Initialize database connection if needed"""
        if not self._initialized:
            if not self.db_pool:
                self.db_pool = await get_db_pool()
            self._initialized = True
    
    async def log_event(
        self,
        action: AuditAction,
        user_id: Optional[int] = None,
        details: Optional[Dict[str, Any]] = None,
        ip_address: Optional[str] = None,
        user_agent: Optional[str] = None,
        success: bool = True
    ) -> bool:
        """
        Log an audit event
        
        Args:
            action: Type of action being logged
            user_id: ID of user performing action (if applicable)
            details: Additional details about the event
            ip_address: Client IP address
            user_agent: Client user agent string
            success: Whether the action was successful
            
        Returns:
            True if event was logged successfully
        """
        try:
            # Ensure we're initialized
            await self.initialize()
            
            # Add success flag to details
            if details is None:
                details = {}
            details['success'] = success
            
            # Log to database
            if self.settings.AUTH_MODE == "multi_user" or user_id is not None:
                await self._log_to_database(
                    action=action,
                    user_id=user_id,
                    details=details,
                    ip_address=ip_address,
                    user_agent=user_agent
                )
            
            # Also log to file with loguru
            self._log_to_file(
                action=action,
                user_id=user_id,
                details=details,
                ip_address=ip_address,
                success=success
            )
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to log audit event {action}: {e}")
            return False
    
    async def _log_to_database(
        self,
        action: str,
        user_id: Optional[int],
        details: Dict[str, Any],
        ip_address: Optional[str],
        user_agent: Optional[str]
    ):
        """Log event to database"""
        try:
            async with self.db_pool.transaction() as conn:
                if hasattr(conn, 'execute'):
                    # PostgreSQL
                    await conn.execute("""
                        INSERT INTO audit_log (user_id, action, details, ip_address, user_agent)
                        VALUES ($1, $2, $3, $4, $5)
                    """, user_id, action, json.dumps(details), ip_address, user_agent)
                else:
                    # SQLite
                    await conn.execute("""
                        INSERT INTO audit_log (user_id, action, details, ip_address, user_agent)
                        VALUES (?, ?, ?, ?, ?)
                    """, (user_id, action, json.dumps(details), ip_address, user_agent))
                    await conn.commit()
                    
        except Exception as e:
            logger.error(f"Failed to log audit event to database: {e}")
            # Don't raise - we still want to log to file
    
    def _log_to_file(
        self,
        action: str,
        user_id: Optional[int],
        details: Dict[str, Any],
        ip_address: Optional[str],
        success: bool
    ):
        """Log event to file using loguru"""
        log_level = "INFO" if success else "WARNING"
        
        # Create audit log entry
        log_message = f"AUDIT: {action}"
        if user_id:
            log_message += f" | User: {user_id}"
        if ip_address:
            log_message += f" | IP: {ip_address}"
        
        # Add structured data for filtering
        logger.bind(
            audit=True,
            user_id=user_id,
            action=action,
            ip=ip_address
        ).log(log_level, log_message)
        
        # Log details separately if present
        if details:
            logger.bind(audit=True).debug(f"Audit details: {json.dumps(details)}")
    
    async def log_login(
        self,
        user_id: int,
        username: str,
        ip_address: str,
        user_agent: Optional[str] = None,
        success: bool = True
    ):
        """Convenience method for logging login attempts"""
        action = AuditAction.USER_LOGIN if success else AuditAction.LOGIN_FAILED
        await self.log_event(
            action=action,
            user_id=user_id if success else None,
            details={"username": username},
            ip_address=ip_address,
            user_agent=user_agent,
            success=success
        )
    
    async def log_logout(
        self,
        user_id: int,
        ip_address: Optional[str] = None
    ):
        """Convenience method for logging logout"""
        await self.log_event(
            action=AuditAction.USER_LOGOUT,
            user_id=user_id,
            ip_address=ip_address
        )
    
    async def log_registration(
        self,
        user_id: int,
        username: str,
        email: str,
        ip_address: Optional[str] = None,
        registration_code: Optional[str] = None
    ):
        """Convenience method for logging user registration"""
        details = {
            "username": username,
            "email": email
        }
        if registration_code:
            details["used_code"] = registration_code[:8] + "..."  # Don't log full code
        
        await self.log_event(
            action=AuditAction.USER_REGISTERED,
            user_id=user_id,
            details=details,
            ip_address=ip_address
        )
    
    async def log_admin_action(
        self,
        admin_id: int,
        action: AuditAction,
        target_user_id: Optional[int] = None,
        details: Optional[Dict[str, Any]] = None,
        ip_address: Optional[str] = None
    ):
        """Convenience method for logging admin actions"""
        if details is None:
            details = {}
        if target_user_id:
            details["target_user_id"] = target_user_id
        
        await self.log_event(
            action=action,
            user_id=admin_id,
            details=details,
            ip_address=ip_address
        )
    
    async def log_security_event(
        self,
        action: AuditAction,
        details: Dict[str, Any],
        ip_address: Optional[str] = None,
        user_id: Optional[int] = None
    ):
        """Convenience method for logging security events"""
        # Security events are always logged at WARNING level
        await self.log_event(
            action=action,
            user_id=user_id,
            details=details,
            ip_address=ip_address,
            success=False  # Security events indicate issues
        )
    
    async def get_user_audit_trail(
        self,
        user_id: int,
        limit: int = 100,
        days: int = 30
    ) -> list:
        """
        Get audit trail for a specific user
        
        Args:
            user_id: User ID to get trail for
            limit: Maximum number of entries
            days: Number of days to look back
            
        Returns:
            List of audit events
        """
        try:
            await self.initialize()
            
            async with self.db_pool.transaction() as conn:
                if hasattr(conn, 'fetch'):
                    # PostgreSQL
                    rows = await conn.fetch("""
                        SELECT id, action, details, ip_address, created_at
                        FROM audit_log
                        WHERE user_id = $1
                        AND created_at > CURRENT_TIMESTAMP - INTERVAL '%s days'
                        ORDER BY created_at DESC
                        LIMIT $2
                    """ % days, user_id, limit)
                else:
                    # SQLite
                    cursor = await conn.execute("""
                        SELECT id, action, details, ip_address, created_at
                        FROM audit_log
                        WHERE user_id = ?
                        AND datetime(created_at) > datetime('now', ? || ' days')
                        ORDER BY created_at DESC
                        LIMIT ?
                    """, (user_id, f"-{days}", limit))
                    rows = await cursor.fetchall()
                
                # Convert to list of dicts
                events = []
                for row in rows:
                    if isinstance(row, dict):
                        events.append(row)
                    else:
                        events.append({
                            "id": row[0],
                            "action": row[1],
                            "details": json.loads(row[2]) if row[2] else {},
                            "ip_address": row[3],
                            "created_at": row[4]
                        })
                
                return events
                
        except Exception as e:
            logger.error(f"Failed to get audit trail for user {user_id}: {e}")
            return []


#######################################################################################################################
#
# Global Audit Service Instance

_audit_service: Optional[AuditService] = None

async def get_audit_service() -> AuditService:
    """Get audit service singleton"""
    global _audit_service
    if not _audit_service:
        _audit_service = AuditService()
        await _audit_service.initialize()
    return _audit_service


async def reset_audit_service():
    """Reset audit service singleton (for testing)"""
    global _audit_service
    if _audit_service:
        await _audit_service.shutdown()
        _audit_service = None


#######################################################################################################################
#
# Convenience Functions

async def audit_log(
    action: AuditAction,
    user_id: Optional[int] = None,
    details: Optional[Dict[str, Any]] = None,
    ip_address: Optional[str] = None,
    user_agent: Optional[str] = None,
    success: bool = True
) -> bool:
    """Quick function to log an audit event"""
    service = await get_audit_service()
    return await service.log_event(
        action=action,
        user_id=user_id,
        details=details,
        ip_address=ip_address,
        user_agent=user_agent,
        success=success
    )


#
## End of audit_service.py
#######################################################################################################################