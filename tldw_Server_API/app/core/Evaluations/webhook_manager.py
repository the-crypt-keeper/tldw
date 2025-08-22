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
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, asdict
from enum import Enum
import sqlite3
from pathlib import Path
from loguru import logger
import secrets


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
        Initialize webhook manager.
        
        Args:
            db_path: Path to database
        """
        if db_path is None:
            db_dir = Path(__file__).parent.parent.parent.parent / "Databases"
            db_dir.mkdir(parents=True, exist_ok=True)
            db_path = db_dir / "evaluations.db"
        
        self.db_path = str(db_path)
        self._init_database()
        
        # Delivery configuration
        self.max_retries = 3
        self.retry_delays = [1, 5, 15]  # seconds
        self.timeout = 30  # seconds
        
        # Background task for retries
        self._retry_task = None
    
    def _init_database(self):
        """Initialize webhook tables."""
        with sqlite3.connect(self.db_path) as conn:
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
        secret: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Register a webhook for a user.
        
        Args:
            user_id: User identifier
            url: Webhook URL
            events: List of events to subscribe to
            secret: Optional secret for HMAC signature (generated if not provided)
            
        Returns:
            Webhook registration details
        """
        # Generate secret if not provided
        if not secret:
            secret = secrets.token_hex(32)
        
        # Convert events to JSON
        events_json = json.dumps([e.value for e in events])
        
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Check if webhook already exists
                cursor.execute("""
                    SELECT id, secret FROM webhook_registrations
                    WHERE user_id = ? AND url = ?
                """, (user_id, url))
                
                existing = cursor.fetchone()
                
                if existing:
                    webhook_id = existing[0]
                    # Update existing webhook
                    cursor.execute("""
                        UPDATE webhook_registrations
                        SET events = ?, active = 1, updated_at = CURRENT_TIMESTAMP
                        WHERE id = ?
                    """, (events_json, webhook_id))
                    
                    logger.info(f"Updated webhook {webhook_id} for user {user_id}")
                else:
                    # Create new webhook
                    cursor.execute("""
                        INSERT INTO webhook_registrations (user_id, url, secret, events)
                        VALUES (?, ?, ?, ?)
                    """, (user_id, url, secret, events_json))
                    
                    webhook_id = cursor.lastrowid
                    logger.info(f"Registered webhook {webhook_id} for user {user_id}")
                
                conn.commit()
            
            return {
                "webhook_id": webhook_id,
                "url": url,
                "events": [e.value for e in events],
                "secret": secret if not existing else "***hidden***",
                "active": True
            }
            
        except Exception as e:
            logger.error(f"Failed to register webhook: {e}")
            raise
    
    async def unregister_webhook(self, user_id: str, url: str) -> bool:
        """
        Unregister a webhook.
        
        Args:
            user_id: User identifier
            url: Webhook URL
            
        Returns:
            True if unregistered successfully
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                cursor.execute("""
                    UPDATE webhook_registrations
                    SET active = 0, updated_at = CURRENT_TIMESTAMP
                    WHERE user_id = ? AND url = ?
                """, (user_id, url))
                
                if cursor.rowcount > 0:
                    conn.commit()
                    logger.info(f"Unregistered webhook for user {user_id}: {url}")
                    return True
                
                return False
                
        except Exception as e:
            logger.error(f"Failed to unregister webhook: {e}")
            return False
    
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
        with sqlite3.connect(self.db_path) as conn:
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
                # Send webhook
                async with aiohttp.ClientSession() as session:
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
                        timeout=aiohttp.ClientTimeout(total=webhook["timeout_seconds"])
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
        with sqlite3.connect(self.db_path) as conn:
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
        with sqlite3.connect(self.db_path) as conn:
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
        with sqlite3.connect(self.db_path) as conn:
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
        with sqlite3.connect(self.db_path) as conn:
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
        with sqlite3.connect(self.db_path) as conn:
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
        
        # Send test webhook
        try:
            async with aiohttp.ClientSession() as session:
                payload_json = payload.to_json()
                signature = self._generate_signature(payload_json, secret)
                
                headers = {
                    "Content-Type": "application/json",
                    "X-Webhook-Signature": signature,
                    "X-Webhook-Event": "test",
                    "X-Webhook-Test": "true"
                }
                
                start_time = datetime.now()
                
                async with session.post(
                    url,
                    data=payload_json,
                    headers=headers,
                    timeout=aiohttp.ClientTimeout(total=10)
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