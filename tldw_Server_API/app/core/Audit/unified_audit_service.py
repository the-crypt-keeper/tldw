"""
Unified Audit Service for tldw_server

This module consolidates all audit logging functionality into a single, 
consistent service that handles authentication, RAG, evaluations, and 
general audit events.

Features:
- Async-first design with proper connection pooling
- Unified event schema across all modules
- Correlation IDs for request tracking
- PII detection and redaction
- Risk scoring and anomaly detection
- Configurable retention and rotation policies
- Export capabilities for compliance
"""

import asyncio
import hashlib
import json
import re
import sqlite3
import time
from contextlib import asynccontextmanager
from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Union
from uuid import uuid4

import aiosqlite
from loguru import logger


# ============================================================================
# Event Types
# ============================================================================

class AuditEventCategory(Enum):
    """High-level audit event categories"""
    AUTHENTICATION = "authentication"
    AUTHORIZATION = "authorization"
    DATA_ACCESS = "data_access"
    DATA_MODIFICATION = "data_modification"
    SYSTEM = "system"
    SECURITY = "security"
    COMPLIANCE = "compliance"
    API_CALL = "api_call"
    EVALUATION = "evaluation"
    RAG = "rag"


class AuditEventType(Enum):
    """Detailed audit event types"""
    # Authentication
    AUTH_LOGIN_SUCCESS = "auth.login.success"
    AUTH_LOGIN_FAILURE = "auth.login.failure"
    AUTH_LOGOUT = "auth.logout"
    AUTH_TOKEN_CREATED = "auth.token.created"
    AUTH_TOKEN_REFRESHED = "auth.token.refreshed"
    AUTH_TOKEN_REVOKED = "auth.token.revoked"
    AUTH_SESSION_EXPIRED = "auth.session.expired"
    
    # User Management
    USER_CREATED = "user.created"
    USER_UPDATED = "user.updated"
    USER_DELETED = "user.deleted"
    USER_ACTIVATED = "user.activated"
    USER_DEACTIVATED = "user.deactivated"
    USER_PASSWORD_CHANGED = "user.password.changed"
    USER_PASSWORD_RESET = "user.password.reset"
    
    # Data Operations
    DATA_READ = "data.read"
    DATA_WRITE = "data.write"
    DATA_UPDATE = "data.update"
    DATA_DELETE = "data.delete"
    DATA_EXPORT = "data.export"
    DATA_IMPORT = "data.import"
    
    # RAG Operations
    RAG_SEARCH = "rag.search"
    RAG_RETRIEVAL = "rag.retrieval"
    RAG_GENERATION = "rag.generation"
    RAG_INDEXING = "rag.indexing"
    RAG_EMBEDDING = "rag.embedding"
    
    # Evaluation Operations
    EVAL_STARTED = "eval.started"
    EVAL_COMPLETED = "eval.completed"
    EVAL_FAILED = "eval.failed"
    EVAL_COST_TRACKED = "eval.cost.tracked"
    
    # API Operations
    API_REQUEST = "api.request"
    API_RESPONSE = "api.response"
    API_ERROR = "api.error"
    API_RATE_LIMITED = "api.rate_limited"
    
    # Security Events
    SECURITY_VIOLATION = "security.violation"
    SECURITY_SCAN = "security.scan"
    PERMISSION_DENIED = "permission.denied"
    SUSPICIOUS_ACTIVITY = "suspicious.activity"
    PII_DETECTED = "pii.detected"
    
    # System Events
    SYSTEM_START = "system.start"
    SYSTEM_STOP = "system.stop"
    SYSTEM_ERROR = "system.error"
    CONFIG_CHANGED = "config.changed"
    MIGRATION_RUN = "migration.run"


class AuditSeverity(Enum):
    """Severity levels for audit events"""
    DEBUG = "debug"
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


# ============================================================================
# Data Models
# ============================================================================

@dataclass
class AuditContext:
    """Context information for audit events"""
    request_id: str = field(default_factory=lambda: str(uuid4()))
    correlation_id: Optional[str] = None
    session_id: Optional[str] = None
    user_id: Optional[str] = None
    api_key_hash: Optional[str] = None
    ip_address: Optional[str] = None
    user_agent: Optional[str] = None
    endpoint: Optional[str] = None
    method: Optional[str] = None


@dataclass
class AuditEvent:
    """Unified audit event structure"""
    # Core fields
    event_id: str = field(default_factory=lambda: str(uuid4()))
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    category: AuditEventCategory = AuditEventCategory.SYSTEM
    event_type: AuditEventType = AuditEventType.SYSTEM_START
    severity: AuditSeverity = AuditSeverity.INFO
    
    # Context
    context: AuditContext = field(default_factory=AuditContext)
    
    # Event details
    resource_type: Optional[str] = None
    resource_id: Optional[str] = None
    action: Optional[str] = None
    result: str = "success"  # success, failure, error
    error_message: Optional[str] = None
    
    # Metrics
    duration_ms: Optional[float] = None
    tokens_used: Optional[int] = None
    estimated_cost: Optional[float] = None
    result_count: Optional[int] = None
    
    # Risk and compliance
    risk_score: int = 0  # 0-100
    pii_detected: bool = False
    compliance_flags: List[str] = field(default_factory=list)
    
    # Additional metadata
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for storage"""
        data = {
            "event_id": self.event_id,
            "timestamp": self.timestamp.isoformat(),
            "category": self.category.value,
            "event_type": self.event_type.value,
            "severity": self.severity.value,
            "resource_type": self.resource_type,
            "resource_id": self.resource_id,
            "action": self.action,
            "result": self.result,
            "error_message": self.error_message,
            "duration_ms": self.duration_ms,
            "tokens_used": self.tokens_used,
            "estimated_cost": self.estimated_cost,
            "result_count": self.result_count,
            "risk_score": self.risk_score,
            "pii_detected": self.pii_detected,
            "compliance_flags": json.dumps(self.compliance_flags) if self.compliance_flags else None,
            "metadata": json.dumps(self.metadata) if self.metadata else None,
        }
        
        # Add context fields
        context_dict = asdict(self.context)
        for key, value in context_dict.items():
            data[f"context_{key}"] = value
        
        return data


# ============================================================================
# PII Detection
# ============================================================================

class PIIDetector:
    """Enhanced PII detection with multiple pattern types"""
    
    # Comprehensive PII patterns
    PII_PATTERNS = {
        "ssn": re.compile(r'\b\d{3}-\d{2}-\d{4}\b'),
        "credit_card": re.compile(r'\b(?:\d{4}[\s-]?){3}\d{4}\b'),
        "email": re.compile(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'),
        "phone": re.compile(r'\b(?:\+?1[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b'),
        "ip_address": re.compile(r'\b(?:\d{1,3}\.){3}\d{1,3}\b'),
        "passport": re.compile(r'\b[A-Z]{1,2}[0-9]{6,9}\b'),
        "driver_license": re.compile(r'\b[A-Z]{1,2}[\s-]?\d{6,8}\b'),
        "bank_account": re.compile(r'\b\d{8,17}\b'),
        "iban": re.compile(r'\b[A-Z]{2}\d{2}[A-Z0-9]{4}\d{7}([A-Z0-9]?){0,16}\b'),
        "api_key": re.compile(r'\b(sk|pk|api[_-]?key)[_-]?[A-Za-z0-9]{32,}\b', re.IGNORECASE),
        "jwt_token": re.compile(r'\beyJ[A-Za-z0-9_-]+\.[A-Za-z0-9_-]+\.[A-Za-z0-9_-]+\b'),
    }
    
    def detect(self, text: str) -> Dict[str, List[str]]:
        """Detect PII in text"""
        if not text:
            return {}
        
        found_pii = {}
        for pii_type, pattern in self.PII_PATTERNS.items():
            matches = pattern.findall(text)
            if matches:
                found_pii[pii_type] = matches
        
        return found_pii
    
    def redact(self, text: str, placeholder_format: str = "[{type}_REDACTED]") -> str:
        """Redact PII from text"""
        if not text:
            return text
        
        redacted = text
        for pii_type, pattern in self.PII_PATTERNS.items():
            placeholder = placeholder_format.format(type=pii_type.upper())
            redacted = pattern.sub(placeholder, redacted)
        
        return redacted


# ============================================================================
# Risk Scoring
# ============================================================================

class RiskScorer:
    """Calculate risk scores for audit events"""
    
    # High-risk operations
    HIGH_RISK_OPERATIONS = {
        "delete", "drop", "truncate", "export", "download",
        "change_password", "reset_password", "grant", "revoke",
        "modify_permissions", "create_admin", "delete_user"
    }
    
    # Suspicious patterns
    SUSPICIOUS_PATTERNS = {
        "rapid_requests": 100,  # More than 100 requests per minute
        "failed_auth": 5,       # More than 5 failed auth attempts
        "data_export": 1000,    # Exporting more than 1000 records
        "after_hours": True,     # Activity outside business hours
        "unusual_location": True # Access from unusual location
    }
    
    def calculate_risk_score(self, event: AuditEvent) -> int:
        """Calculate risk score for an event (0-100)"""
        score = 0
        
        # Event type risk
        if event.event_type in [
            AuditEventType.SECURITY_VIOLATION,
            AuditEventType.PERMISSION_DENIED,
            AuditEventType.SUSPICIOUS_ACTIVITY
        ]:
            score += 50
        elif event.event_type in [
            AuditEventType.AUTH_LOGIN_FAILURE,
            AuditEventType.DATA_DELETE,
            AuditEventType.CONFIG_CHANGED
        ]:
            score += 30
        
        # Failed operations
        if event.result == "failure":
            score += 20
        elif event.result == "error":
            score += 10
        
        # High-risk operations
        if event.action and any(op in event.action.lower() for op in self.HIGH_RISK_OPERATIONS):
            score += 30
        
        # PII detection
        if event.pii_detected:
            score += 25
        
        # Time-based risk (after hours)
        hour = event.timestamp.hour
        if hour < 6 or hour > 22:
            score += 10
        
        # Weekend activity
        if event.timestamp.weekday() >= 5:
            score += 5
        
        # Multiple consecutive failures (from metadata)
        if event.metadata.get("consecutive_failures", 0) > 3:
            score += 20
        
        # Large data operations
        if event.result_count and event.result_count > 1000:
            score += 15
        
        return min(score, 100)


# ============================================================================
# Unified Audit Service
# ============================================================================

class UnifiedAuditService:
    """
    Unified audit service with async operations, connection pooling,
    and comprehensive event tracking.
    """
    
    def __init__(
        self,
        db_path: Optional[str] = None,
        retention_days: int = 90,
        enable_pii_detection: bool = True,
        enable_risk_scoring: bool = True,
        buffer_size: int = 1000,
        flush_interval: float = 10.0
    ):
        """
        Initialize unified audit service.
        
        Args:
            db_path: Path to audit database
            retention_days: Days to retain audit logs
            enable_pii_detection: Enable PII detection
            enable_risk_scoring: Enable risk scoring
            buffer_size: Maximum events to buffer before flush
            flush_interval: Seconds between automatic flushes
        """
        # Configuration
        if db_path is None:
            db_dir = Path("./Databases")
            db_dir.mkdir(parents=True, exist_ok=True)
            db_path = db_dir / "unified_audit.db"
        
        self.db_path = Path(db_path)
        self.retention_days = retention_days
        self.enable_pii_detection = enable_pii_detection
        self.enable_risk_scoring = enable_risk_scoring
        self.buffer_size = buffer_size
        self.flush_interval = flush_interval
        
        # Components
        self.pii_detector = PIIDetector() if enable_pii_detection else None
        self.risk_scorer = RiskScorer() if enable_risk_scoring else None
        
        # Event buffer
        self.event_buffer: List[AuditEvent] = []
        self.buffer_lock = asyncio.Lock()
        
        # Background tasks
        self._flush_task: Optional[asyncio.Task] = None
        self._cleanup_task: Optional[asyncio.Task] = None
        
        # Connection pool
        self._db_pool: Optional[aiosqlite.Connection] = None
        self._pool_lock = asyncio.Lock()
        
        # Statistics
        self.stats = {
            "events_logged": 0,
            "events_flushed": 0,
            "flush_failures": 0,
            "high_risk_events": 0
        }
    
    async def initialize(self):
        """Initialize database and start background tasks"""
        await self._init_database()
        await self.start_background_tasks()
    
    async def _init_database(self):
        """Initialize database schema"""
        async with aiosqlite.connect(self.db_path) as db:
            # Main audit table with all fields
            await db.execute("""
                CREATE TABLE IF NOT EXISTS audit_events (
                    event_id TEXT PRIMARY KEY,
                    timestamp TIMESTAMP NOT NULL,
                    category TEXT NOT NULL,
                    event_type TEXT NOT NULL,
                    severity TEXT NOT NULL,
                    
                    -- Context fields
                    context_request_id TEXT,
                    context_correlation_id TEXT,
                    context_session_id TEXT,
                    context_user_id TEXT,
                    context_api_key_hash TEXT,
                    context_ip_address TEXT,
                    context_user_agent TEXT,
                    context_endpoint TEXT,
                    context_method TEXT,
                    
                    -- Event details
                    resource_type TEXT,
                    resource_id TEXT,
                    action TEXT,
                    result TEXT,
                    error_message TEXT,
                    
                    -- Metrics
                    duration_ms REAL,
                    tokens_used INTEGER,
                    estimated_cost REAL,
                    result_count INTEGER,
                    
                    -- Risk and compliance
                    risk_score INTEGER,
                    pii_detected BOOLEAN,
                    compliance_flags TEXT,
                    
                    -- Metadata
                    metadata TEXT
                )
            """)
            
            # Create indexes for common queries
            await db.execute("CREATE INDEX IF NOT EXISTS idx_timestamp ON audit_events(timestamp)")
            await db.execute("CREATE INDEX IF NOT EXISTS idx_user_id ON audit_events(context_user_id)")
            await db.execute("CREATE INDEX IF NOT EXISTS idx_request_id ON audit_events(context_request_id)")
            await db.execute("CREATE INDEX IF NOT EXISTS idx_correlation_id ON audit_events(context_correlation_id)")
            await db.execute("CREATE INDEX IF NOT EXISTS idx_event_type ON audit_events(event_type)")
            await db.execute("CREATE INDEX IF NOT EXISTS idx_category ON audit_events(category)")
            await db.execute("CREATE INDEX IF NOT EXISTS idx_severity ON audit_events(severity)")
            await db.execute("CREATE INDEX IF NOT EXISTS idx_risk_score ON audit_events(risk_score)")
            
            # Daily statistics table
            await db.execute("""
                CREATE TABLE IF NOT EXISTS audit_daily_stats (
                    date DATE NOT NULL,
                    category TEXT NOT NULL,
                    total_events INTEGER DEFAULT 0,
                    high_risk_events INTEGER DEFAULT 0,
                    failed_events INTEGER DEFAULT 0,
                    total_cost REAL DEFAULT 0.0,
                    total_tokens INTEGER DEFAULT 0,
                    avg_duration_ms REAL,
                    PRIMARY KEY (date, category)
                )
            """)
            
            await db.commit()
    
    async def start_background_tasks(self):
        """Start background flush and cleanup tasks"""
        if not self._flush_task:
            self._flush_task = asyncio.create_task(self._flush_loop())
        if not self._cleanup_task:
            self._cleanup_task = asyncio.create_task(self._cleanup_loop())
    
    async def stop(self):
        """Stop background tasks and flush remaining events"""
        # Cancel background tasks
        if self._flush_task:
            self._flush_task.cancel()
            try:
                await self._flush_task
            except asyncio.CancelledError:
                pass
        
        if self._cleanup_task:
            self._cleanup_task.cancel()
            try:
                await self._cleanup_task
            except asyncio.CancelledError:
                pass
        
        # Final flush
        await self.flush()
        
        # Close connection pool
        if self._db_pool:
            await self._db_pool.close()
    
    async def _flush_loop(self):
        """Background task to periodically flush events"""
        while True:
            try:
                await asyncio.sleep(self.flush_interval)
                await self.flush()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in audit flush loop: {e}")
    
    async def _cleanup_loop(self):
        """Background task to clean up old logs"""
        while True:
            try:
                await asyncio.sleep(86400)  # Daily
                await self.cleanup_old_logs()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in audit cleanup loop: {e}")
    
    async def log_event(
        self,
        event_type: AuditEventType,
        context: Optional[AuditContext] = None,
        category: Optional[AuditEventCategory] = None,
        severity: Optional[AuditSeverity] = None,
        resource_type: Optional[str] = None,
        resource_id: Optional[str] = None,
        action: Optional[str] = None,
        result: str = "success",
        error_message: Optional[str] = None,
        duration_ms: Optional[float] = None,
        tokens_used: Optional[int] = None,
        estimated_cost: Optional[float] = None,
        result_count: Optional[int] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Log an audit event.
        
        Returns:
            Event ID of the logged event
        """
        # Auto-determine category if not provided
        if category is None:
            category = self._determine_category(event_type)
        
        # Auto-determine severity if not provided
        if severity is None:
            severity = self._determine_severity(event_type, result)
        
        # Create context if not provided
        if context is None:
            context = AuditContext()
        
        # Create event
        event = AuditEvent(
            category=category,
            event_type=event_type,
            severity=severity,
            context=context,
            resource_type=resource_type,
            resource_id=resource_id,
            action=action,
            result=result,
            error_message=error_message,
            duration_ms=duration_ms,
            tokens_used=tokens_used,
            estimated_cost=estimated_cost,
            result_count=result_count,
            metadata=metadata or {}
        )
        
        # PII detection
        if self.enable_pii_detection and metadata:
            metadata_str = json.dumps(metadata)
            found_pii = self.pii_detector.detect(metadata_str)
            if found_pii:
                event.pii_detected = True
                event.compliance_flags.append("pii_detected")
                # Redact PII from metadata
                redacted_str = self.pii_detector.redact(metadata_str)
                event.metadata = json.loads(redacted_str)
        
        # Risk scoring
        if self.enable_risk_scoring:
            event.risk_score = self.risk_scorer.calculate_risk_score(event)
            if event.risk_score >= 70:
                self.stats["high_risk_events"] += 1
                logger.warning(
                    f"High-risk event: {event_type.value} "
                    f"(risk: {event.risk_score}, user: {context.user_id})"
                )
        
        # Add to buffer
        async with self.buffer_lock:
            self.event_buffer.append(event)
            self.stats["events_logged"] += 1
            
            # Flush if buffer is full or high-risk event
            if len(self.event_buffer) >= self.buffer_size or event.risk_score >= 80:
                asyncio.create_task(self.flush())
        
        return event.event_id
    
    async def flush(self):
        """Flush buffered events to database"""
        async with self.buffer_lock:
            if not self.event_buffer:
                return
            
            events = self.event_buffer.copy()
            self.event_buffer.clear()
        
        try:
            async with aiosqlite.connect(self.db_path) as db:
                # Prepare batch data
                records = [event.to_dict() for event in events]
                
                # Batch insert
                await db.executemany("""
                    INSERT OR IGNORE INTO audit_events (
                        event_id, timestamp, category, event_type, severity,
                        context_request_id, context_correlation_id, context_session_id,
                        context_user_id, context_api_key_hash, context_ip_address,
                        context_user_agent, context_endpoint, context_method,
                        resource_type, resource_id, action, result, error_message,
                        duration_ms, tokens_used, estimated_cost, result_count,
                        risk_score, pii_detected, compliance_flags, metadata
                    ) VALUES (
                        :event_id, :timestamp, :category, :event_type, :severity,
                        :context_request_id, :context_correlation_id, :context_session_id,
                        :context_user_id, :context_api_key_hash, :context_ip_address,
                        :context_user_agent, :context_endpoint, :context_method,
                        :resource_type, :resource_id, :action, :result, :error_message,
                        :duration_ms, :tokens_used, :estimated_cost, :result_count,
                        :risk_score, :pii_detected, :compliance_flags, :metadata
                    )
                """, records)
                
                # Update daily statistics
                await self._update_daily_stats(db, events)
                
                await db.commit()
                
                self.stats["events_flushed"] += len(events)
                logger.debug(f"Flushed {len(events)} audit events to database")
                
        except Exception as e:
            logger.error(f"Failed to flush audit events: {e}")
            self.stats["flush_failures"] += 1
            
            # Re-add events to buffer (with limit to prevent memory issues)
            async with self.buffer_lock:
                max_buffer = self.buffer_size * 2
                self.event_buffer = (events + self.event_buffer)[:max_buffer]
    
    async def _update_daily_stats(self, db: aiosqlite.Connection, events: List[AuditEvent]):
        """Update daily statistics"""
        from collections import defaultdict
        
        # Aggregate by date and category
        stats = defaultdict(lambda: {
            "total": 0, "high_risk": 0, "failed": 0,
            "cost": 0.0, "tokens": 0, "durations": []
        })
        
        for event in events:
            date = event.timestamp.date()
            key = (date, event.category.value)
            
            stats[key]["total"] += 1
            if event.risk_score >= 70:
                stats[key]["high_risk"] += 1
            if event.result != "success":
                stats[key]["failed"] += 1
            if event.estimated_cost:
                stats[key]["cost"] += event.estimated_cost
            if event.tokens_used:
                stats[key]["tokens"] += event.tokens_used
            if event.duration_ms:
                stats[key]["durations"].append(event.duration_ms)
        
        # Update database
        for (date, category), data in stats.items():
            avg_duration = (
                sum(data["durations"]) / len(data["durations"])
                if data["durations"] else None
            )
            
            await db.execute("""
                INSERT INTO audit_daily_stats (
                    date, category, total_events, high_risk_events,
                    failed_events, total_cost, total_tokens, avg_duration_ms
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(date, category) DO UPDATE SET
                    total_events = total_events + excluded.total_events,
                    high_risk_events = high_risk_events + excluded.high_risk_events,
                    failed_events = failed_events + excluded.failed_events,
                    total_cost = total_cost + excluded.total_cost,
                    total_tokens = total_tokens + excluded.total_tokens,
                    avg_duration_ms = COALESCE(
                        (avg_duration_ms * total_events + excluded.avg_duration_ms * excluded.total_events) 
                        / (total_events + excluded.total_events),
                        excluded.avg_duration_ms
                    )
            """, (
                date, category, data["total"], data["high_risk"],
                data["failed"], data["cost"], data["tokens"], avg_duration
            ))
    
    async def cleanup_old_logs(self):
        """Remove logs older than retention period"""
        cutoff = datetime.now(timezone.utc) - timedelta(days=self.retention_days)
        
        try:
            async with aiosqlite.connect(self.db_path) as db:
                # Delete old events
                cursor = await db.execute(
                    "DELETE FROM audit_events WHERE timestamp < ?",
                    (cutoff.isoformat(),)
                )
                deleted = cursor.rowcount
                
                # Delete old stats
                await db.execute(
                    "DELETE FROM audit_daily_stats WHERE date < ?",
                    (cutoff.date(),)
                )
                
                await db.commit()
                
                if deleted > 0:
                    logger.info(f"Cleaned up {deleted} audit events older than {self.retention_days} days")
                    
        except Exception as e:
            logger.error(f"Failed to cleanup old audit logs: {e}")
    
    async def query_events(
        self,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        event_types: Optional[List[AuditEventType]] = None,
        categories: Optional[List[AuditEventCategory]] = None,
        user_id: Optional[str] = None,
        request_id: Optional[str] = None,
        correlation_id: Optional[str] = None,
        min_risk_score: Optional[int] = None,
        limit: int = 100,
        offset: int = 0
    ) -> List[Dict[str, Any]]:
        """Query audit events with filters"""
        query = "SELECT * FROM audit_events WHERE 1=1"
        params = []
        
        if start_time:
            query += " AND timestamp >= ?"
            params.append(start_time.isoformat())
        
        if end_time:
            query += " AND timestamp <= ?"
            params.append(end_time.isoformat())
        
        if event_types:
            placeholders = ",".join("?" * len(event_types))
            query += f" AND event_type IN ({placeholders})"
            params.extend([et.value for et in event_types])
        
        if categories:
            placeholders = ",".join("?" * len(categories))
            query += f" AND category IN ({placeholders})"
            params.extend([c.value for c in categories])
        
        if user_id:
            query += " AND context_user_id = ?"
            params.append(user_id)
        
        if request_id:
            query += " AND context_request_id = ?"
            params.append(request_id)
        
        if correlation_id:
            query += " AND context_correlation_id = ?"
            params.append(correlation_id)
        
        if min_risk_score is not None:
            query += " AND risk_score >= ?"
            params.append(min_risk_score)
        
        query += " ORDER BY timestamp DESC LIMIT ? OFFSET ?"
        params.extend([limit, offset])
        
        try:
            async with aiosqlite.connect(self.db_path) as db:
                db.row_factory = aiosqlite.Row
                async with db.execute(query, params) as cursor:
                    rows = await cursor.fetchall()
                    return [dict(row) for row in rows]
        except Exception as e:
            logger.error(f"Failed to query audit events: {e}")
            return []
    
    def _determine_category(self, event_type: AuditEventType) -> AuditEventCategory:
        """Auto-determine category from event type"""
        type_name = event_type.name.lower()
        
        if type_name.startswith("auth_"):
            return AuditEventCategory.AUTHENTICATION
        elif type_name.startswith("user_"):
            return AuditEventCategory.AUTHORIZATION
        elif type_name.startswith("data_"):
            return AuditEventCategory.DATA_ACCESS
        elif type_name.startswith("rag_"):
            return AuditEventCategory.RAG
        elif type_name.startswith("eval_"):
            return AuditEventCategory.EVALUATION
        elif type_name.startswith("api_"):
            return AuditEventCategory.API_CALL
        elif type_name.startswith("security_"):
            return AuditEventCategory.SECURITY
        elif type_name.startswith("system_"):
            return AuditEventCategory.SYSTEM
        else:
            return AuditEventCategory.SYSTEM
    
    def _determine_severity(self, event_type: AuditEventType, result: str) -> AuditSeverity:
        """Auto-determine severity from event type and result"""
        if result == "error":
            return AuditSeverity.ERROR
        elif result == "failure":
            return AuditSeverity.WARNING
        
        # Critical events
        if event_type in [
            AuditEventType.SECURITY_VIOLATION,
            AuditEventType.SUSPICIOUS_ACTIVITY
        ]:
            return AuditSeverity.CRITICAL
        
        # Warning events
        elif event_type in [
            AuditEventType.AUTH_LOGIN_FAILURE,
            AuditEventType.PERMISSION_DENIED,
            AuditEventType.API_RATE_LIMITED
        ]:
            return AuditSeverity.WARNING
        
        # Debug events
        elif event_type in [
            AuditEventType.SYSTEM_START,
            AuditEventType.SYSTEM_STOP
        ]:
            return AuditSeverity.DEBUG
        
        # Default to INFO
        else:
            return AuditSeverity.INFO
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get current statistics"""
        return {
            "events_logged": self.stats["events_logged"],
            "events_flushed": self.stats["events_flushed"],
            "events_buffered": len(self.event_buffer),
            "flush_failures": self.stats["flush_failures"],
            "high_risk_events": self.stats["high_risk_events"],
            "db_path": str(self.db_path),
            "retention_days": self.retention_days,
            "pii_detection_enabled": self.enable_pii_detection,
            "risk_scoring_enabled": self.enable_risk_scoring
        }


# ============================================================================
# Context Manager for Audit Operations
# ============================================================================

@asynccontextmanager
async def audit_operation(
    service: UnifiedAuditService,
    event_type: AuditEventType,
    context: AuditContext,
    **kwargs
):
    """Context manager for auditing operations with automatic timing"""
    start_time = time.perf_counter()
    event_id = None
    
    try:
        # Log start event if it's a long operation
        if "STARTED" in event_type.name:
            event_id = await service.log_event(
                event_type=event_type,
                context=context,
                result="started",
                **kwargs
            )
        
        yield event_id
        
        # Log success
        duration_ms = (time.perf_counter() - start_time) * 1000
        await service.log_event(
            event_type=event_type,
            context=context,
            result="success",
            duration_ms=duration_ms,
            **kwargs
        )
        
    except Exception as e:
        # Log failure
        duration_ms = (time.perf_counter() - start_time) * 1000
        await service.log_event(
            event_type=event_type,
            context=context,
            result="failure",
            error_message=str(e),
            duration_ms=duration_ms,
            **kwargs
        )
        raise


# ============================================================================
# Global Service Instance
# ============================================================================

_unified_audit_service: Optional[UnifiedAuditService] = None


async def get_unified_audit_service() -> UnifiedAuditService:
    """Get or create the global unified audit service"""
    global _unified_audit_service
    
    if _unified_audit_service is None:
        _unified_audit_service = UnifiedAuditService()
        await _unified_audit_service.initialize()
    
    return _unified_audit_service


async def shutdown_audit_service():
    """Shutdown the global audit service"""
    global _unified_audit_service
    
    if _unified_audit_service:
        await _unified_audit_service.stop()
        _unified_audit_service = None