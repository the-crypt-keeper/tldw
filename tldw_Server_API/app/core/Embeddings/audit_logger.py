# audit_logger.py
# Audit logging for sensitive operations in the Embeddings module

import json
import time
from datetime import datetime
from typing import Any, Dict, Optional
from enum import Enum
from pathlib import Path
import threading

from loguru import logger


class AuditEventType(Enum):
    """Types of audit events"""
    # Security events
    PATH_TRAVERSAL_ATTEMPT = "path_traversal_attempt"
    INVALID_USER_ID = "invalid_user_id"
    INVALID_MODEL_ID = "invalid_model_id"
    UNAUTHORIZED_ACCESS = "unauthorized_access"
    
    # Resource management events
    MODEL_LOADED = "model_loaded"
    MODEL_EVICTED = "model_evicted"
    MEMORY_LIMIT_EXCEEDED = "memory_limit_exceeded"
    
    # Admin operations
    CACHE_CLEARED = "cache_cleared"
    CIRCUIT_BREAKER_RESET = "circuit_breaker_reset"
    CONFIG_CHANGED = "config_changed"
    
    # User operations
    EMBEDDING_CREATED = "embedding_created"
    BATCH_EMBEDDING_CREATED = "batch_embedding_created"
    RATE_LIMIT_EXCEEDED = "rate_limit_exceeded"


class AuditLogger:
    """
    Centralized audit logging for security and compliance.
    Logs sensitive operations to a separate audit log file.
    """
    
    def __init__(self, audit_log_path: Optional[str] = None):
        """
        Initialize the audit logger.
        
        Args:
            audit_log_path: Path to the audit log file. If None, uses default.
        """
        if audit_log_path:
            self.audit_log_path = Path(audit_log_path)
        else:
            # Default to a logs directory
            self.audit_log_path = Path("./logs/embeddings_audit.jsonl")
        
        # Ensure the directory exists
        self.audit_log_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Thread lock for file writing
        self._lock = threading.Lock()
        
        # Add a separate logger for audit events
        audit_logger_id = logger.add(
            str(self.audit_log_path),
            format="{message}",
            rotation="100 MB",
            retention="90 days",
            compression="gz",
            serialize=True,
            enqueue=True  # Thread-safe
        )
        self.audit_logger_id = audit_logger_id
        
        logger.info(f"Audit logger initialized. Writing to: {self.audit_log_path}")
    
    def log_event(
        self,
        event_type: AuditEventType,
        user_id: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None,
        severity: str = "INFO",
        ip_address: Optional[str] = None,
        session_id: Optional[str] = None
    ) -> None:
        """
        Log an audit event.
        
        Args:
            event_type: Type of the audit event
            user_id: ID of the user involved (if applicable)
            details: Additional details about the event
            severity: Severity level (INFO, WARNING, ERROR, CRITICAL)
            ip_address: IP address of the request
            session_id: Session identifier
        """
        audit_entry = {
            "timestamp": datetime.utcnow().isoformat(),
            "event_type": event_type.value,
            "severity": severity,
            "user_id": user_id,
            "ip_address": ip_address,
            "session_id": session_id,
            "details": details or {},
            "unix_timestamp": time.time()
        }
        
        # Log to the audit file
        with self._lock:
            try:
                with open(self.audit_log_path, 'a') as f:
                    f.write(json.dumps(audit_entry) + '\n')
            except Exception as e:
                logger.error(f"Failed to write audit log: {e}")
        
        # Also log to the main logger based on severity
        log_message = f"AUDIT: {event_type.value} - User: {user_id}"
        if details:
            log_message += f" - Details: {json.dumps(details)}"
        
        if severity == "CRITICAL":
            logger.critical(log_message)
        elif severity == "ERROR":
            logger.error(log_message)
        elif severity == "WARNING":
            logger.warning(log_message)
        else:
            logger.info(log_message)
    
    def log_security_event(
        self,
        event_type: AuditEventType,
        user_id: Optional[str] = None,
        attempted_value: Optional[str] = None,
        ip_address: Optional[str] = None,
        **kwargs
    ) -> None:
        """
        Log a security-related event.
        
        Args:
            event_type: Type of security event
            user_id: User involved
            attempted_value: The value that triggered the security event
            ip_address: IP address of the request
            **kwargs: Additional details
        """
        details = {
            "attempted_value": attempted_value,
            **kwargs
        }
        
        self.log_event(
            event_type=event_type,
            user_id=user_id,
            details=details,
            severity="WARNING" if event_type != AuditEventType.UNAUTHORIZED_ACCESS else "ERROR",
            ip_address=ip_address
        )
    
    def log_resource_event(
        self,
        event_type: AuditEventType,
        model_id: str,
        memory_usage_gb: Optional[float] = None,
        **kwargs
    ) -> None:
        """
        Log a resource management event.
        
        Args:
            event_type: Type of resource event
            model_id: Model identifier
            memory_usage_gb: Memory usage in GB
            **kwargs: Additional details
        """
        details = {
            "model_id": model_id,
            "memory_usage_gb": memory_usage_gb,
            **kwargs
        }
        
        self.log_event(
            event_type=event_type,
            details=details,
            severity="INFO"
        )
    
    def log_admin_operation(
        self,
        event_type: AuditEventType,
        admin_id: str,
        operation_details: Dict[str, Any],
        ip_address: Optional[str] = None
    ) -> None:
        """
        Log an admin operation.
        
        Args:
            event_type: Type of admin operation
            admin_id: ID of the admin user
            operation_details: Details about the operation
            ip_address: IP address of the admin
        """
        self.log_event(
            event_type=event_type,
            user_id=admin_id,
            details=operation_details,
            severity="INFO",
            ip_address=ip_address
        )
    
    def get_recent_events(
        self,
        event_type: Optional[AuditEventType] = None,
        user_id: Optional[str] = None,
        limit: int = 100
    ) -> list:
        """
        Retrieve recent audit events (for monitoring/analysis).
        
        Args:
            event_type: Filter by event type
            user_id: Filter by user ID
            limit: Maximum number of events to return
            
        Returns:
            List of audit events
        """
        events = []
        
        try:
            with self._lock:
                with open(self.audit_log_path, 'r') as f:
                    # Read lines in reverse order for recent events
                    lines = f.readlines()
                    for line in reversed(lines[-limit*2:]):  # Read more than limit to account for filtering
                        try:
                            event = json.loads(line.strip())
                            
                            # Apply filters
                            if event_type and event.get('event_type') != event_type.value:
                                continue
                            if user_id and event.get('user_id') != user_id:
                                continue
                            
                            events.append(event)
                            
                            if len(events) >= limit:
                                break
                        except json.JSONDecodeError:
                            continue
        except FileNotFoundError:
            logger.warning("Audit log file not found")
        except Exception as e:
            logger.error(f"Error reading audit log: {e}")
        
        return events


# Global audit logger instance
_audit_logger: Optional[AuditLogger] = None


def get_audit_logger() -> AuditLogger:
    """Get or create the global audit logger instance."""
    global _audit_logger
    if _audit_logger is None:
        _audit_logger = AuditLogger()
    return _audit_logger


def audit_log(
    event_type: AuditEventType,
    **kwargs
) -> None:
    """
    Convenience function for logging audit events.
    
    Args:
        event_type: Type of audit event
        **kwargs: Additional parameters for the log event
    """
    logger_instance = get_audit_logger()
    logger_instance.log_event(event_type, **kwargs)