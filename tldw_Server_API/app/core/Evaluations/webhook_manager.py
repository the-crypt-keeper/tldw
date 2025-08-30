"""
Webhook management for Evaluations module.

Provides webhook registration, delivery, and retry logic for
asynchronous evaluation notifications.
"""

import json
import hmac
import hashlib
import asyncio
import aiohttp
from datetime import datetime, timedelta, timezone
from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass, asdict
from enum import Enum
import sqlite3
from pathlib import Path
from loguru import logger
import secrets

from tldw_Server_API.app.core.Chatbooks.chatbook_service import audit_logger
# Import security enhancements
from tldw_Server_API.app.core.Evaluations.webhook_security import (
    webhook_validator,
    WebhookPermissionManager,
    WebhookValidationResult
)
from tldw_Server_API.app.core.Evaluations.config_manager import get_config
from tldw_Server_API.app.core.Evaluations.audit_logger import (
    AuditEventType,
    AuditSeverity
)
from tldw_Server_API.app.core.Evaluations.metrics import get_metrics
from tldw_Server_API.app.core.Evaluations.connection_pool import get_connection


class WebhookEvent(Enum):
    """Webhook event types."""
    EVALUATION_STARTED = "evaluation.started"
    EVALUATION_PROGRESS = "evaluation.progress"
    EVALUATION_COMPLETED = "evaluation.completed"
    EVALUATION_FAILED = "evaluation.failed"
    EVALUATION_CANCELLED = "evaluation.cancelled"
    BATCH_STARTED = "batch.started"
    BATCH_COMPLETED = "batch.completed"
    BATCH_FAILED = "batch.failed"


@dataclass
class WebhookPayload:
    """Webhook payload structure."""
    event: str
    evaluation_id: str
    timestamp: str
    data: Dict[str, Any]
    
    def to_json(self) -> str:
        """Convert to JSON string."""
        return json.dumps(asdict(self), default=str)


class WebhookManager:
    """Manages webhook registrations and deliveries."""
    
    def __init__(self, db_path: Optional[str] = None):
        """
        Initialize webhook manager with enhanced security.
        
        Args:
            db_path: Path to database
        """
        if db_path is None:
            db_dir = Path(__file__).parent.parent.parent.parent / "Databases"
            db_dir.mkdir(parents=True, exist_ok=True)
            db_path = db_dir / "evaluations.db"
        
        self.db_path = str(db_path)
        self._init_database()
        
        # Load delivery configuration from external config
        delivery_config = get_config("webhooks.delivery", {})
        self.max_retries = delivery_config.get("max_retries", 3)
        self.retry_delays = delivery_config.get("retry_delays", [1, 5, 15])
        self.timeout = delivery_config.get("timeout_seconds", 30)
        self.batch_size = delivery_config.get("batch_size", 10)
        
        # Security components
        self.permission_manager = WebhookPermissionManager(self.db_path)
        
        # Metrics
        self.metrics = get_metrics()
        
        # Background task for retries
        self._retry_task = None
    
    def _init_database(self):
        """Initialize webhook tables."""
        with get_connection() as conn:
            # These tables are created in the migration
            # but we ensure they exist here for standalone usage
            conn.execute("""
                CREATE TABLE IF NOT EXISTS webhook_registrations (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id TEXT NOT NULL,
                    url TEXT NOT NULL,
                    secret TEXT NOT NULL,
                    events TEXT NOT NULL,
                    active BOOLEAN DEFAULT 1,
                    retry_count INTEGER DEFAULT 3,
                    timeout_seconds INTEGER DEFAULT 30,
                    total_deliveries INTEGER DEFAULT 0,
                    successful_deliveries INTEGER DEFAULT 0,
                    failed_deliveries INTEGER DEFAULT 0,
                    last_delivery_at TIMESTAMP,
                    last_error TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE(user_id, url)
                )
            """)
            
            conn.execute("""
                CREATE TABLE IF NOT EXISTS webhook_deliveries (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    webhook_id INTEGER NOT NULL,
                    evaluation_id TEXT NOT NULL,
                    event_type TEXT NOT NULL,
                    payload TEXT NOT NULL,
                    signature TEXT NOT NULL,
                    status_code INTEGER,
                    response_body TEXT,
                    response_time_ms INTEGER,
                    delivered BOOLEAN DEFAULT 0,
                    retry_count INTEGER DEFAULT 0,
                    error_message TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    delivered_at TIMESTAMP,
                    next_retry_at TIMESTAMP,
                    FOREIGN KEY (webhook_id) REFERENCES webhook_registrations(id)
                )
            """)
            
            conn.commit()
    
    async def register_webhook(
        self,
        user_id: str,
        url: str,
        events: List[WebhookEvent],
        secret: Optional[str] = None,
        skip_validation: bool = False
    ) -> Dict[str, Any]:
        """
        Register a webhook for a user with enhanced security validation.
        
        Args:
            user_id: User identifier
            url: Webhook URL
            events: List of events to subscribe to
            secret: Optional secret for HMAC signature (generated if not provided)
            skip_validation: Skip URL validation (for testing)
            
        Returns:
            Webhook registration details with validation results
            
        Raises:
            ValueError: If validation fails or permissions are insufficient
        """
        start_time = asyncio.get_event_loop().time()
        
        try:
            # Check permissions first
            has_permission, permission_error = await self.permission_manager.check_webhook_permissions(
                user_id=user_id,
                url=url,
                action="register"
            )
            
            if not has_permission:
                audit_logger.log_event(
                    event_type=AuditEventType.WEBHOOK_REGISTER,
                    action="Webhook registration denied - permission check failed",
                    user_id=user_id,
                    outcome="failure",
                    severity=AuditSeverity.MEDIUM,
                    details={
                        "url": url[:100] + "..." if len(url) > 100 else url,
                        "error": permission_error,
                        "events": [e.value for e in events]
                    }
                )
                raise ValueError(f"Registration denied: {permission_error}")
            
            # URL security validation
            validation_result = None
            if not skip_validation:
                validation_result = await webhook_validator.validate_webhook_url(
                    url=url,
                    user_id=user_id,
                    check_connectivity=True
                )
                
                if not validation_result.valid:
                    error_messages = [error.message for error in validation_result.errors]
                    audit_logger.log_event(
                        event_type=AuditEventType.WEBHOOK_REGISTER,
                        action="Webhook registration failed - URL validation errors",
                        user_id=user_id,
                        outcome="failure",
                        severity=AuditSeverity.HIGH,
                        details={
                            "url": url[:100] + "..." if len(url) > 100 else url,
                            "validation_errors": error_messages,
                            "security_score": validation_result.security_score
                        }
                    )
                    raise ValueError(f"URL validation failed: {'; '.join(error_messages)}")
            
            # Generate secret if not provided
            if not secret:
                secret = secrets.token_hex(32)
            
            # Convert events to JSON
            events_json = json.dumps([e.value for e in events])
            
            # Register webhook in database
            with get_connection() as conn:
                cursor = conn.cursor()
                
                # Check if webhook already exists
                cursor.execute("""
                    SELECT id, secret, events FROM webhook_registrations
                    WHERE user_id = ? AND url = ?
                """, (user_id, url))
                
                existing = cursor.fetchone()
                webhook_id = None
                
                if existing:
                    webhook_id = existing[0]
                    existing_secret = existing[1]
                    existing_events = existing[2]
                    
                    # Update existing webhook
                    cursor.execute("""
                        UPDATE webhook_registrations
                        SET events = ?, active = 1, updated_at = CURRENT_TIMESTAMP
                        WHERE id = ?
                    """, (events_json, webhook_id))
                    
                    # Use existing secret if not provided
                    if not secret:
                        secret = existing_secret
                    
                    action = "Updated"
                    logger.info(f"Updated webhook {webhook_id} for user {user_id}")
                else:
                    # Create new webhook
                    cursor.execute("""
                        INSERT INTO webhook_registrations (
                            user_id, url, secret, events, 
                            retry_count, timeout_seconds
                        ) VALUES (?, ?, ?, ?, ?, ?)
                    """, (
                        user_id, url, secret, events_json,
                        self.max_retries, self.timeout
                    ))
                    
                    webhook_id = cursor.lastrowid
                    action = "Registered"
                    logger.info(f"Registered webhook {webhook_id} for user {user_id}")
                
                conn.commit()
            
            # Record metrics
            processing_time = asyncio.get_event_loop().time() - start_time
            self.metrics.record_webhook_delivery(
                event_type="registration",
                outcome="success",
                response_time=processing_time
            )
            
            # Audit log successful registration
            audit_logger.log_event(
                event_type=AuditEventType.WEBHOOK_REGISTER,
                action=f"Webhook {action.lower()} successfully",
                user_id=user_id,
                resource_id=str(webhook_id),
                resource_type="webhook",
                outcome="success",
                severity=AuditSeverity.LOW,
                details={
                    "url": url[:100] + "..." if len(url) > 100 else url,
                    "events": [e.value for e in events],
                    "security_score": validation_result.security_score if validation_result else None,
                    "processing_time_ms": int(processing_time * 1000)
                }
            )
            
            # Prepare response
            response = {
                "webhook_id": webhook_id,
                "url": url,
                "events": [e.value for e in events],
                "secret": secret if not existing else "***hidden***",
                "active": True,
                "action": action.lower()
            }
            
            # Include validation results if available
            if validation_result:
                response["validation"] = {
                    "security_score": validation_result.security_score,
                    "warnings": [w.to_dict() for w in validation_result.warnings],
                    "connectivity": validation_result.metadata.get("connectivity", {})
                }
            
            return response
            
        except ValueError:
            # Re-raise validation errors
            raise
        except Exception as e:
            # Log unexpected errors
            processing_time = asyncio.get_event_loop().time() - start_time
            self.metrics.record_webhook_delivery(
                event_type="registration",
                outcome="failure",
                response_time=processing_time
            )
            
            audit_logger.log_event(
                event_type=AuditEventType.WEBHOOK_REGISTER,
                action="Webhook registration failed - system error",
                user_id=user_id,
                outcome="failure",
                severity=AuditSeverity.HIGH,
                details={
                    "url": url[:100] + "..." if len(url) > 100 else url,
                    "error": str(e),
                    "processing_time_ms": int(processing_time * 1000)
                }
            )
            
            logger.error(f"Failed to register webhook: {e}")
            raise ValueError(f"Webhook registration failed: {str(e)}")
    
    async def unregister_webhook(self, user_id: str, url: str) -> Dict[str, Any]:
        """
        Unregister a webhook with permission checks.
        
        Args:
            user_id: User identifier
            url: Webhook URL
            
        Returns:
            Dict with operation result and details
        """
        try:
            # Check permissions
            has_permission, permission_error = await self.permission_manager.check_webhook_permissions(
                user_id=user_id,
                url=url,
                action="delete"
            )
            
            if not has_permission:
                audit_logger.log_event(
                    event_type=AuditEventType.WEBHOOK_UNREGISTER,
                    action="Webhook unregistration denied - permission check failed",
                    user_id=user_id,
                    outcome="failure",
                    severity=AuditSeverity.MEDIUM,
                    details={
                        "url": url[:100] + "..." if len(url) > 100 else url,
                        "error": permission_error
                    }
                )
                return {
                    "success": False,
                    "error": f"Unregistration denied: {permission_error}"
                }
            
            # Perform unregistration
            with get_connection() as conn:
                cursor = conn.cursor()
                
                # Get webhook details before unregistering
                cursor.execute("""
                    SELECT id, events FROM webhook_registrations
                    WHERE user_id = ? AND url = ? AND active = 1
                """, (user_id, url))
                
                webhook_data = cursor.fetchone()
                if not webhook_data:
                    return {
                        "success": False,
                        "error": "Webhook not found or already inactive"
                    }
                
                webhook_id, events_json = webhook_data
                
                # Deactivate webhook
                cursor.execute("""
                    UPDATE webhook_registrations
                    SET active = 0, updated_at = CURRENT_TIMESTAMP
                    WHERE user_id = ? AND url = ?
                """, (user_id, url))
                
                if cursor.rowcount > 0:
                    conn.commit()
                    
                    # Audit log successful unregistration
                    audit_logger.log_event(
                        event_type=AuditEventType.WEBHOOK_UNREGISTER,
                        action="Webhook unregistered successfully",
                        user_id=user_id,
                        resource_id=str(webhook_id),
                        resource_type="webhook",
                        outcome="success",
                        severity=AuditSeverity.LOW,
                        details={
                            "url": url[:100] + "..." if len(url) > 100 else url,
                            "events": json.loads(events_json) if events_json else []
                        }
                    )
                    
                    logger.info(f"Unregistered webhook {webhook_id} for user {user_id}: {url}")
                    return {
                        "success": True,
                        "webhook_id": webhook_id,
                        "message": "Webhook unregistered successfully"
                    }
                
                return {
                    "success": False,
                    "error": "Failed to update webhook status"
                }
                
        except Exception as e:
            audit_logger.log_event(
                event_type=AuditEventType.WEBHOOK_UNREGISTER,
                action="Webhook unregistration failed - system error",
                user_id=user_id,
                outcome="failure",
                severity=AuditSeverity.MEDIUM,
                details={
                    "url": url[:100] + "..." if len(url) > 100 else url,
                    "error": str(e)
                }
            )
            
            logger.error(f"Failed to unregister webhook: {e}")
            return {
                "success": False,
                "error": f"Unregistration failed: {str(e)}"
            }
    
    async def send_webhook(
        self,
        user_id: str,
        event: WebhookEvent,
        evaluation_id: str,
        data: Dict[str, Any]
    ):
        """
        Send webhook notification to all registered endpoints.
        
        Args:
            user_id: User identifier
            event: Event type
            evaluation_id: Evaluation ID
            data: Event data
        """
        # Get active webhooks for user
        webhooks = await self._get_webhooks(user_id, event)
        
        if not webhooks:
            return
        
        # Create payload
        payload = WebhookPayload(
            event=event.value,
            evaluation_id=evaluation_id,
            timestamp=datetime.now(timezone.utc).isoformat(),
            data=data
        )
        
        # Send to each webhook
        tasks = []
        for webhook in webhooks:
            task = asyncio.create_task(
                self._deliver_webhook(webhook, payload)
            )
            tasks.append(task)
        
        # Wait for all deliveries
        await asyncio.gather(*tasks, return_exceptions=True)
    
    async def _get_webhooks(
        self,
        user_id: str,
        event: WebhookEvent
    ) -> List[Dict[str, Any]]:
        """Get active webhooks for user and event."""
        with get_connection() as conn:
            cursor = conn.cursor()
            
            cursor.execute("""
                SELECT id, url, secret, retry_count, timeout_seconds
                FROM webhook_registrations
                WHERE user_id = ? AND active = 1
                AND events LIKE ?
            """, (user_id, f'%"{event.value}"%'))
            
            webhooks = []
            for row in cursor.fetchall():
                webhooks.append({
                    "id": row[0],
                    "url": row[1],
                    "secret": row[2],
                    "retry_count": row[3],
                    "timeout_seconds": row[4]
                })
            
            return webhooks
    
    async def _deliver_webhook(
        self,
        webhook: Dict[str, Any],
        payload: WebhookPayload
    ):
        """Deliver webhook with retry logic."""
        webhook_id = webhook["id"]
        url = webhook["url"]
        secret = webhook["secret"]
        
        # Validate URL before delivery to prevent SSRF
        from urllib.parse import urlparse
        import ipaddress
        import socket
        
        try:
            parsed_url = urlparse(url)
            if parsed_url.hostname:
                # Check if hostname is an IP address
                try:
                    ip_addr = ipaddress.ip_address(parsed_url.hostname)
                    # Check against private networks
                    private_networks = [
                        ipaddress.IPv4Network("10.0.0.0/8"),
                        ipaddress.IPv4Network("172.16.0.0/12"),
                        ipaddress.IPv4Network("192.168.0.0/16"),
                        ipaddress.IPv4Network("169.254.0.0/16"),
                        ipaddress.IPv4Network("127.0.0.0/8"),
                        ipaddress.IPv6Network("::1/128"),
                        ipaddress.IPv6Network("fc00::/7"),
                        ipaddress.IPv6Network("fe80::/10"),
                    ]
                    for network in private_networks:
                        if ip_addr in network:
                            logger.error(f"Webhook URL points to private network: {url}")
                            self._update_webhook_stats(webhook_id, success=False, error="URL points to private network")
                            return
                except ValueError:
                    # Not an IP, resolve hostname
                    try:
                        resolved_ips = socket.getaddrinfo(parsed_url.hostname, None)
                        for addr_info in resolved_ips:
                            ip_str = addr_info[4][0]
                            try:
                                ip_addr = ipaddress.ip_address(ip_str)
                                private_networks = [
                                    ipaddress.IPv4Network("10.0.0.0/8"),
                                    ipaddress.IPv4Network("172.16.0.0/12"),
                                    ipaddress.IPv4Network("192.168.0.0/16"),
                                    ipaddress.IPv4Network("169.254.0.0/16"),
                                    ipaddress.IPv4Network("127.0.0.0/8"),
                                    ipaddress.IPv6Network("::1/128"),
                                    ipaddress.IPv6Network("fc00::/7"),
                                    ipaddress.IPv6Network("fe80::/10"),
                                ]
                                for network in private_networks:
                                    if ip_addr in network:
                                        logger.error(f"Webhook hostname resolves to private IP: {url}")
                                        self._update_webhook_stats(webhook_id, success=False, error="Hostname resolves to private IP")
                                        return
                            except ValueError:
                                pass
                    except socket.gaierror:
                        logger.error(f"Failed to resolve webhook hostname: {url}")
                        self._update_webhook_stats(webhook_id, success=False, error="DNS resolution failed")
                        return
        except Exception as e:
            logger.error(f"URL validation failed: {e}")
            self._update_webhook_stats(webhook_id, success=False, error=f"URL validation failed: {str(e)}")
            return
        
        # Generate signature
        payload_json = payload.to_json()
        signature = self._generate_signature(payload_json, secret)
        
        # Create delivery record
        delivery_id = self._create_delivery_record(
            webhook_id,
            payload.evaluation_id,
            payload.event,
            payload_json,
            signature
        )
        
        # Attempt delivery
        success = False
        for attempt in range(webhook["retry_count"]):
            try:
                # Configure session to prevent SSRF
                connector = aiohttp.TCPConnector(
                    force_close=True,
                    enable_cleanup_closed=True
                )
                
                # Send webhook
                async with aiohttp.ClientSession(
                    connector=connector,
                    trust_env=False  # Disable automatic proxy detection
                ) as session:
                    headers = {
                        "Content-Type": "application/json",
                        "X-Webhook-Signature": signature,
                        "X-Webhook-Event": payload.event,
                        "X-Webhook-Delivery": str(delivery_id)
                    }
                    
                    start_time = datetime.now()
                    
                    async with session.post(
                        url,
                        data=payload_json,
                        headers=headers,
                        timeout=aiohttp.ClientTimeout(total=webhook["timeout_seconds"]),
                        allow_redirects=False  # Prevent SSRF via redirects
                    ) as response:
                        response_time = (datetime.now() - start_time).total_seconds() * 1000
                        response_body = await response.text()
                        
                        # Update delivery record
                        self._update_delivery_record(
                            delivery_id,
                            status_code=response.status,
                            response_body=response_body[:1000],  # Limit stored response
                            response_time_ms=int(response_time),
                            delivered=response.status < 400,
                            retry_count=attempt
                        )
                        
                        if response.status < 400:
                            success = True
                            self._update_webhook_stats(webhook_id, success=True)
                            logger.info(f"Webhook delivered successfully: {delivery_id}")
                            break
                        else:
                            logger.warning(f"Webhook delivery failed with status {response.status}: {delivery_id}")
                            
            except asyncio.TimeoutError:
                error = "Request timeout"
                logger.warning(f"Webhook delivery timeout: {delivery_id}")
                
            except Exception as e:
                error = str(e)
                logger.error(f"Webhook delivery error: {delivery_id} - {error}")
            
            # If not last attempt, wait before retry
            if attempt < webhook["retry_count"] - 1 and not success:
                delay = self.retry_delays[min(attempt, len(self.retry_delays) - 1)]
                await asyncio.sleep(delay)
        
        # Final update if failed
        if not success:
            self._update_delivery_record(
                delivery_id,
                delivered=False,
                retry_count=webhook["retry_count"],
                error_message=error if 'error' in locals() else "Max retries exceeded"
            )
            self._update_webhook_stats(webhook_id, success=False, error=error if 'error' in locals() else None)
    
    def _generate_signature(self, payload: str, secret: str) -> str:
        """Generate HMAC signature for payload."""
        signature = hmac.new(
            secret.encode(),
            payload.encode(),
            hashlib.sha256
        ).hexdigest()
        return f"sha256={signature}"
    
    def _create_delivery_record(
        self,
        webhook_id: int,
        evaluation_id: str,
        event_type: str,
        payload: str,
        signature: str
    ) -> int:
        """Create delivery record in database."""
        with get_connection() as conn:
            cursor = conn.cursor()
            
            cursor.execute("""
                INSERT INTO webhook_deliveries (
                    webhook_id, evaluation_id, event_type, payload, signature
                ) VALUES (?, ?, ?, ?, ?)
            """, (webhook_id, evaluation_id, event_type, payload, signature))
            
            delivery_id = cursor.lastrowid
            conn.commit()
            
            return delivery_id
    
    def _update_delivery_record(
        self,
        delivery_id: int,
        status_code: Optional[int] = None,
        response_body: Optional[str] = None,
        response_time_ms: Optional[int] = None,
        delivered: bool = False,
        retry_count: int = 0,
        error_message: Optional[str] = None
    ):
        """Update delivery record."""
        with get_connection() as conn:
            cursor = conn.cursor()
            
            cursor.execute("""
                UPDATE webhook_deliveries
                SET status_code = ?, response_body = ?, response_time_ms = ?,
                    delivered = ?, retry_count = ?, error_message = ?,
                    delivered_at = CASE WHEN ? THEN CURRENT_TIMESTAMP ELSE NULL END
                WHERE id = ?
            """, (
                status_code, response_body, response_time_ms,
                delivered, retry_count, error_message,
                delivered, delivery_id
            ))
            
            conn.commit()
    
    def _update_webhook_stats(
        self,
        webhook_id: int,
        success: bool,
        error: Optional[str] = None
    ):
        """Update webhook statistics."""
        with get_connection() as conn:
            cursor = conn.cursor()
            
            if success:
                cursor.execute("""
                    UPDATE webhook_registrations
                    SET total_deliveries = total_deliveries + 1,
                        successful_deliveries = successful_deliveries + 1,
                        last_delivery_at = CURRENT_TIMESTAMP,
                        last_error = NULL
                    WHERE id = ?
                """, (webhook_id,))
            else:
                cursor.execute("""
                    UPDATE webhook_registrations
                    SET total_deliveries = total_deliveries + 1,
                        failed_deliveries = failed_deliveries + 1,
                        last_delivery_at = CURRENT_TIMESTAMP,
                        last_error = ?
                    WHERE id = ?
                """, (error, webhook_id))
            
            conn.commit()
    
    async def get_webhook_status(
        self,
        user_id: str,
        url: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Get webhook status for user.
        
        Args:
            user_id: User identifier
            url: Optional specific webhook URL
            
        Returns:
            List of webhook status information
        """
        with get_connection() as conn:
            cursor = conn.cursor()
            
            if url:
                cursor.execute("""
                    SELECT id, url, events, active, total_deliveries,
                           successful_deliveries, failed_deliveries,
                           last_delivery_at, last_error, created_at
                    FROM webhook_registrations
                    WHERE user_id = ? AND url = ?
                """, (user_id, url))
            else:
                cursor.execute("""
                    SELECT id, url, events, active, total_deliveries,
                           successful_deliveries, failed_deliveries,
                           last_delivery_at, last_error, created_at
                    FROM webhook_registrations
                    WHERE user_id = ?
                """, (user_id,))
            
            webhooks = []
            for row in cursor.fetchall():
                webhooks.append({
                    "id": row[0],
                    "url": row[1],
                    "events": json.loads(row[2]),
                    "active": bool(row[3]),
                    "statistics": {
                        "total_deliveries": row[4],
                        "successful_deliveries": row[5],
                        "failed_deliveries": row[6],
                        "success_rate": row[5] / row[4] if row[4] > 0 else 0
                    },
                    "last_delivery_at": row[7],
                    "last_error": row[8],
                    "created_at": row[9]
                })
            
            return webhooks
    
    async def test_webhook(
        self,
        user_id: str,
        url: str
    ) -> Dict[str, Any]:
        """
        Send a test webhook.
        
        Args:
            user_id: User identifier
            url: Webhook URL to test
            
        Returns:
            Test result
        """
        # Get webhook details
        with get_connection() as conn:
            cursor = conn.cursor()
            
            cursor.execute("""
                SELECT id, secret FROM webhook_registrations
                WHERE user_id = ? AND url = ?
            """, (user_id, url))
            
            row = cursor.fetchone()
            
            if not row:
                return {
                    "success": False,
                    "error": "Webhook not found"
                }
            
            webhook_id, secret = row
        
        # Create test payload
        payload = WebhookPayload(
            event="test",
            evaluation_id="test_eval_123",
            timestamp=datetime.now(timezone.utc).isoformat(),
            data={
                "message": "This is a test webhook delivery",
                "user_id": user_id
            }
        )
        
        # Validate URL before test to prevent SSRF
        validation_result = await webhook_validator.validate_webhook_url(
            url=url,
            user_id=user_id,
            check_connectivity=False  # We'll test connectivity ourselves
        )
        
        if not validation_result.valid:
            return {
                "success": False,
                "error": "URL validation failed",
                "validation_errors": [error.to_dict() for error in validation_result.errors]
            }
        
        # Send test webhook with DNS rebinding prevention
        try:
            # Parse URL to get hostname and port
            from urllib.parse import urlparse, urlunparse
            import socket
            import ipaddress
            
            parsed_url = urlparse(url)
            hostname = parsed_url.hostname
            port = parsed_url.port or (443 if parsed_url.scheme == "https" else 80)
            
            # Resolve hostname to IP and validate it's not private
            safe_ip = None
            try:
                # Check if it's already an IP
                ip_addr = ipaddress.ip_address(hostname)
                safe_ip = str(ip_addr)
            except ValueError:
                # Resolve hostname
                try:
                    ip_addresses = socket.getaddrinfo(hostname, port, socket.AF_UNSPEC, socket.SOCK_STREAM)
                    if ip_addresses:
                        # Use the first resolved IP (already validated by webhook_validator)
                        safe_ip = ip_addresses[0][4][0]
                except socket.gaierror:
                    return {
                        "success": False,
                        "error": "Failed to resolve webhook hostname"
                    }
            
            if not safe_ip:
                return {
                    "success": False,
                    "error": "Could not determine target IP address"
                }
            
            # Reconstruct URL with IP to prevent DNS rebinding
            if parsed_url.port:
                netloc = f"[{safe_ip}]:{parsed_url.port}" if ":" in safe_ip else f"{safe_ip}:{parsed_url.port}"
            else:
                netloc = f"[{safe_ip}]" if ":" in safe_ip else safe_ip
            
            ip_url = urlunparse((
                parsed_url.scheme,
                netloc,
                parsed_url.path,
                parsed_url.params,
                parsed_url.query,
                parsed_url.fragment
            ))
            
            # Configure session to prevent SSRF
            connector = aiohttp.TCPConnector(
                force_close=True,
                enable_cleanup_closed=True
            )
            
            async with aiohttp.ClientSession(
                connector=connector,
                trust_env=False  # Disable automatic proxy detection
            ) as session:
                payload_json = payload.to_json()
                signature = self._generate_signature(payload_json, secret)
                
                headers = {
                    "Content-Type": "application/json",
                    "X-Webhook-Signature": signature,
                    "X-Webhook-Event": "test",
                    "X-Webhook-Test": "true",
                    "Host": hostname  # Add original hostname for virtual hosting
                }
                
                start_time = datetime.now()
                
                async with session.post(
                    ip_url,  # Use IP-based URL to prevent DNS rebinding
                    data=payload_json,
                    headers=headers,
                    timeout=aiohttp.ClientTimeout(total=10),
                    allow_redirects=False  # Prevent SSRF via redirects
                ) as response:
                    response_time = (datetime.now() - start_time).total_seconds() * 1000
                    response_body = await response.text()
                    
                    return {
                        "success": response.status < 400,
                        "status_code": response.status,
                        "response_time_ms": int(response_time),
                        "response_body": response_body[:500]
                    }
                    
        except asyncio.TimeoutError:
            return {
                "success": False,
                "error": "Request timeout"
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }


# Global instance
webhook_manager = WebhookManager()