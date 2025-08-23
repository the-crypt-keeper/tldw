"""
Comprehensive audit logging for Evaluations module.

Provides secure, detailed logging of all evaluation operations,
authentication events, and security-relevant activities for compliance
and security monitoring.
"""

import json
import sqlite3
import uuid
from datetime import datetime, timezone
from typing import Dict, Any, Optional, List
from contextlib import contextmanager
from dataclasses import dataclass, asdict
from enum import Enum
from pathlib import Path
from loguru import logger
import asyncio
import threading


class AuditEventType(Enum):
    """Types of audit events."""
    # Authentication events
    AUTHENTICATION_SUCCESS = "auth.success"
    AUTHENTICATION_FAILURE = "auth.failure"
    AUTHORIZATION_FAILURE = "auth.authorization_failure"
    
    # Evaluation operations
    EVALUATION_CREATE = "eval.create"
    EVALUATION_READ = "eval.read"
    EVALUATION_UPDATE = "eval.update"
    EVALUATION_DELETE = "eval.delete"
    EVALUATION_RUN = "eval.run"
    
    # Rate limiting events
    RATE_LIMIT_EXCEEDED = "rate_limit.exceeded"
    RATE_LIMIT_WARNING = "rate_limit.warning"
    
    # Security events
    INPUT_VALIDATION_FAILURE = "security.input_validation_failure"
    SUSPICIOUS_ACTIVITY = "security.suspicious_activity"
    CIRCUIT_BREAKER_OPENED = "security.circuit_breaker_opened"
    
    # Webhook events
    WEBHOOK_REGISTER = "webhook.register"
    WEBHOOK_UNREGISTER = "webhook.unregister"
    WEBHOOK_DELIVERY_FAILURE = "webhook.delivery_failure"
    
    # Configuration changes
    CONFIG_CHANGE = "config.change"
    TIER_UPGRADE = "config.tier_upgrade"
    
    # Data events
    DATA_EXPORT = "data.export"
    DATA_DELETION = "data.deletion"


class AuditSeverity(Enum):
    """Severity levels for audit events."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class AuditEvent:
    """Audit event data structure."""
    event_id: str
    timestamp: str
    event_type: str
    severity: str
    user_id: Optional[str]
    session_id: Optional[str]
    ip_address: Optional[str]
    user_agent: Optional[str]
    endpoint: Optional[str]
    method: Optional[str]
    resource_id: Optional[str]
    resource_type: Optional[str]
    action: str
    outcome: str  # "success", "failure", "warning"
    details: Dict[str, Any]
    metadata: Dict[str, Any]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return asdict(self)


class AuditLogger:
    """Comprehensive audit logger for security and compliance."""
    
    def __init__(self, db_path: Optional[str] = None):
        """
        Initialize audit logger.
        
        Args:
            db_path: Path to audit database (defaults to evaluations.db)
        """
        if db_path is None:
            db_dir = Path(__file__).parent.parent.parent.parent / "Databases"
            db_dir.mkdir(parents=True, exist_ok=True)
            db_path = db_dir / "evaluations.db"
        
        self.db_path = str(db_path)
        self._lock = threading.RLock()
        self._init_database()
    
    def _init_database(self):
        """Initialize audit logging tables."""
        with sqlite3.connect(self.db_path) as conn:
            # Audit events table
            conn.execute("""
                CREATE TABLE IF NOT EXISTS audit_events (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    event_id TEXT UNIQUE NOT NULL,
                    timestamp TIMESTAMP NOT NULL,
                    event_type TEXT NOT NULL,
                    severity TEXT NOT NULL,
                    user_id TEXT,
                    session_id TEXT,
                    ip_address TEXT,
                    user_agent TEXT,
                    endpoint TEXT,
                    method TEXT,
                    resource_id TEXT,
                    resource_type TEXT,
                    action TEXT NOT NULL,
                    outcome TEXT NOT NULL,
                    details TEXT,
                    metadata TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # Create indexes for efficient querying
            indexes = [
                "CREATE INDEX IF NOT EXISTS idx_audit_timestamp ON audit_events(timestamp)",
                "CREATE INDEX IF NOT EXISTS idx_audit_event_type ON audit_events(event_type)",
                "CREATE INDEX IF NOT EXISTS idx_audit_user_id ON audit_events(user_id)",
                "CREATE INDEX IF NOT EXISTS idx_audit_severity ON audit_events(severity)",
                "CREATE INDEX IF NOT EXISTS idx_audit_outcome ON audit_events(outcome)",
                "CREATE INDEX IF NOT EXISTS idx_audit_ip ON audit_events(ip_address)",
                "CREATE INDEX IF NOT EXISTS idx_audit_resource ON audit_events(resource_type, resource_id)"
            ]
            
            for index_sql in indexes:
                try:
                    conn.execute(index_sql)
                except sqlite3.OperationalError:
                    pass  # Index might already exist
            
            conn.commit()
    
    @contextmanager
    def get_connection(self):
        """Context manager for database connections."""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
        finally:
            conn.close()
    
    def log_event(
        self,
        event_type: AuditEventType,
        action: str,
        outcome: str = "success",
        severity: AuditSeverity = AuditSeverity.LOW,
        user_id: Optional[str] = None,
        session_id: Optional[str] = None,
        ip_address: Optional[str] = None,
        user_agent: Optional[str] = None,
        endpoint: Optional[str] = None,
        method: Optional[str] = None,
        resource_id: Optional[str] = None,
        resource_type: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """
        Log an audit event.
        
        Args:
            event_type: Type of event
            action: Description of action performed
            outcome: Result of action ("success", "failure", "warning")
            severity: Severity level
            user_id: User identifier
            session_id: Session identifier
            ip_address: Client IP address
            user_agent: Client user agent
            endpoint: API endpoint
            method: HTTP method
            resource_id: ID of affected resource
            resource_type: Type of affected resource
            details: Additional event details
            metadata: Additional metadata
        """
        event = AuditEvent(
            event_id=str(uuid.uuid4()),
            timestamp=datetime.now(timezone.utc).isoformat(),
            event_type=event_type.value,
            severity=severity.value,
            user_id=user_id,
            session_id=session_id,
            ip_address=ip_address,
            user_agent=user_agent,
            endpoint=endpoint,
            method=method,
            resource_id=resource_id,
            resource_type=resource_type,
            action=action,
            outcome=outcome,
            details=details or {},
            metadata=metadata or {}
        )
        
        self._store_event(event)
    
    def _store_event(self, event: AuditEvent):
        """Store audit event in database."""
        with self._lock:
            try:
                with self.get_connection() as conn:
                    conn.execute("""
                        INSERT INTO audit_events (
                            event_id, timestamp, event_type, severity, user_id,
                            session_id, ip_address, user_agent, endpoint, method,
                            resource_id, resource_type, action, outcome, details, metadata
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """, (
                        event.event_id,
                        event.timestamp,
                        event.event_type,
                        event.severity,
                        event.user_id,
                        event.session_id,
                        event.ip_address,
                        event.user_agent,
                        event.endpoint,
                        event.method,
                        event.resource_id,
                        event.resource_type,
                        event.action,
                        event.outcome,
                        json.dumps(event.details),
                        json.dumps(event.metadata)
                    ))
                    conn.commit()
                    
                # Log to application logs for immediate visibility
                self._log_to_app_logger(event)
                
            except Exception as e:
                logger.error(f"Failed to store audit event: {e}")
                # Fallback to application logs only
                self._log_to_app_logger(event)
    
    def _log_to_app_logger(self, event: AuditEvent):
        """Log audit event to application logger."""
        log_message = (
            f"AUDIT [{event.severity.upper()}] {event.event_type}: {event.action} "
            f"({event.outcome}) - User: {event.user_id or 'anonymous'}, "
            f"IP: {event.ip_address or 'unknown'}"
        )
        
        if event.severity in [AuditSeverity.HIGH.value, AuditSeverity.CRITICAL.value]:
            logger.warning(log_message)
        else:
            logger.info(log_message)
    
    def get_events(
        self,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        event_type: Optional[AuditEventType] = None,
        user_id: Optional[str] = None,
        severity: Optional[AuditSeverity] = None,
        outcome: Optional[str] = None,
        limit: int = 1000,
        offset: int = 0
    ) -> List[Dict[str, Any]]:
        """
        Retrieve audit events with filtering.
        
        Args:
            start_time: Start time filter
            end_time: End time filter
            event_type: Event type filter
            user_id: User ID filter
            severity: Severity filter
            outcome: Outcome filter
            limit: Maximum results
            offset: Results offset
            
        Returns:
            List of audit events
        """
        with self.get_connection() as conn:
            query = "SELECT * FROM audit_events WHERE 1=1"
            params = []
            
            if start_time:
                query += " AND timestamp >= ?"
                params.append(start_time.isoformat())
            
            if end_time:
                query += " AND timestamp <= ?"
                params.append(end_time.isoformat())
            
            if event_type:
                query += " AND event_type = ?"
                params.append(event_type.value)
            
            if user_id:
                query += " AND user_id = ?"
                params.append(user_id)
            
            if severity:
                query += " AND severity = ?"
                params.append(severity.value)
            
            if outcome:
                query += " AND outcome = ?"
                params.append(outcome)
            
            query += " ORDER BY timestamp DESC LIMIT ? OFFSET ?"
            params.extend([limit, offset])
            
            cursor = conn.execute(query, params)
            events = []
            
            for row in cursor:
                event = dict(row)
                event["details"] = json.loads(event["details"] or "{}")
                event["metadata"] = json.loads(event["metadata"] or "{}")
                events.append(event)
            
            return events
    
    def get_security_summary(
        self,
        hours: int = 24
    ) -> Dict[str, Any]:
        """
        Get security event summary for the last N hours.
        
        Args:
            hours: Number of hours to look back
            
        Returns:
            Security summary statistics
        """
        start_time = datetime.now(timezone.utc).replace(
            hour=datetime.now(timezone.utc).hour - hours
        )
        
        with self.get_connection() as conn:
            # Get event counts by severity
            cursor = conn.execute("""
                SELECT severity, COUNT(*) as count
                FROM audit_events
                WHERE timestamp >= ? AND severity IN ('high', 'critical')
                GROUP BY severity
            """, (start_time.isoformat(),))
            
            severity_counts = {row[0]: row[1] for row in cursor}
            
            # Get failure counts by event type
            cursor = conn.execute("""
                SELECT event_type, COUNT(*) as count
                FROM audit_events
                WHERE timestamp >= ? AND outcome = 'failure'
                GROUP BY event_type
            """, (start_time.isoformat(),))
            
            failure_counts = {row[0]: row[1] for row in cursor}
            
            # Get unique users with security events
            cursor = conn.execute("""
                SELECT COUNT(DISTINCT user_id) as unique_users
                FROM audit_events
                WHERE timestamp >= ? AND severity IN ('high', 'critical')
            """, (start_time.isoformat(),))
            
            unique_security_users = cursor.fetchone()[0]
            
            # Get top IP addresses with failures
            cursor = conn.execute("""
                SELECT ip_address, COUNT(*) as count
                FROM audit_events
                WHERE timestamp >= ? AND outcome = 'failure'
                AND ip_address IS NOT NULL
                GROUP BY ip_address
                ORDER BY count DESC
                LIMIT 10
            """, (start_time.isoformat(),))
            
            top_failing_ips = [{"ip": row[0], "count": row[1]} for row in cursor]
            
            return {
                "time_range_hours": hours,
                "severity_counts": severity_counts,
                "failure_counts": failure_counts,
                "unique_security_users": unique_security_users,
                "top_failing_ips": top_failing_ips,
                "generated_at": datetime.now(timezone.utc).isoformat()
            }
    
    async def cleanup_old_events(self, retention_days: int = 90):
        """
        Clean up old audit events beyond retention period.
        
        Args:
            retention_days: Number of days to retain events
        """
        cutoff_date = datetime.now(timezone.utc).replace(
            day=datetime.now(timezone.utc).day - retention_days
        )
        
        def _cleanup():
            with self.get_connection() as conn:
                cursor = conn.execute("""
                    DELETE FROM audit_events
                    WHERE timestamp < ?
                """, (cutoff_date.isoformat(),))
                
                deleted_count = cursor.rowcount
                conn.commit()
                
                logger.info(f"Cleaned up {deleted_count} audit events older than {retention_days} days")
                return deleted_count
        
        # Run cleanup in thread to avoid blocking
        return await asyncio.to_thread(_cleanup)


# Global audit logger instance
audit_logger = AuditLogger()


# Convenience functions for common audit operations
def log_authentication_success(user_id: str, ip_address: str, user_agent: str):
    """Log successful authentication."""
    audit_logger.log_event(
        event_type=AuditEventType.AUTHENTICATION_SUCCESS,
        action=f"User {user_id} authenticated successfully",
        user_id=user_id,
        ip_address=ip_address,
        user_agent=user_agent,
        severity=AuditSeverity.LOW
    )


def log_authentication_failure(attempted_user_id: str, ip_address: str, reason: str):
    """Log failed authentication attempt."""
    audit_logger.log_event(
        event_type=AuditEventType.AUTHENTICATION_FAILURE,
        action=f"Authentication failed for user {attempted_user_id}: {reason}",
        outcome="failure",
        user_id=attempted_user_id,
        ip_address=ip_address,
        severity=AuditSeverity.MEDIUM,
        details={"failure_reason": reason}
    )


def log_rate_limit_exceeded(user_id: str, endpoint: str, ip_address: str, limit_type: str):
    """Log rate limit violation."""
    audit_logger.log_event(
        event_type=AuditEventType.RATE_LIMIT_EXCEEDED,
        action=f"Rate limit exceeded for {limit_type}",
        outcome="failure",
        user_id=user_id,
        ip_address=ip_address,
        endpoint=endpoint,
        severity=AuditSeverity.MEDIUM,
        details={"limit_type": limit_type}
    )


def log_evaluation_operation(
    operation: str,
    user_id: str,
    resource_id: str,
    endpoint: str,
    method: str,
    outcome: str = "success",
    details: Optional[Dict[str, Any]] = None
):
    """Log evaluation CRUD operation."""
    event_type_map = {
        "create": AuditEventType.EVALUATION_CREATE,
        "read": AuditEventType.EVALUATION_READ,
        "update": AuditEventType.EVALUATION_UPDATE,
        "delete": AuditEventType.EVALUATION_DELETE,
        "run": AuditEventType.EVALUATION_RUN
    }
    
    audit_logger.log_event(
        event_type=event_type_map.get(operation, AuditEventType.EVALUATION_READ),
        action=f"Evaluation {operation} operation",
        user_id=user_id,
        resource_id=resource_id,
        resource_type="evaluation",
        endpoint=endpoint,
        method=method,
        outcome=outcome,
        severity=AuditSeverity.LOW,
        details=details
    )


def log_security_event(
    event_type: AuditEventType,
    action: str,
    user_id: Optional[str] = None,
    ip_address: Optional[str] = None,
    details: Optional[Dict[str, Any]] = None
):
    """Log security-related event."""
    audit_logger.log_event(
        event_type=event_type,
        action=action,
        outcome="warning",
        user_id=user_id,
        ip_address=ip_address,
        severity=AuditSeverity.HIGH,
        details=details
    )