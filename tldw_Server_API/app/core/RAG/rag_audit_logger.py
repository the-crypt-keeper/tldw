"""
RAG-specific audit logging.

Provides comprehensive audit logging for all RAG operations including
searches, retrievals, generations, and user interactions.
"""

import json
import sqlite3
import hashlib
import os
from datetime import datetime, timezone
from typing import Dict, Any, Optional, List
from pathlib import Path
from dataclasses import dataclass, asdict
from enum import Enum
import asyncio
from contextlib import asynccontextmanager

from loguru import logger


class RAGEventType(Enum):
    """Types of RAG events to audit"""
    # Search operations
    SEARCH_REQUEST = "search_request"
    SEARCH_SUCCESS = "search_success"
    SEARCH_FAILURE = "search_failure"
    
    # Agent operations
    AGENT_REQUEST = "agent_request"
    AGENT_GENERATION = "agent_generation"
    AGENT_SUCCESS = "agent_success"
    AGENT_FAILURE = "agent_failure"
    
    # Retrieval operations
    RETRIEVAL_START = "retrieval_start"
    RETRIEVAL_SUCCESS = "retrieval_success"
    RETRIEVAL_FAILURE = "retrieval_failure"
    
    # Rate limiting
    RATE_LIMIT_CHECK = "rate_limit_check"
    RATE_LIMIT_EXCEEDED = "rate_limit_exceeded"
    
    # Performance
    SLOW_QUERY = "slow_query"
    HIGH_TOKEN_USAGE = "high_token_usage"
    
    # Security
    UNAUTHORIZED_ACCESS = "unauthorized_access"
    SUSPICIOUS_ACTIVITY = "suspicious_activity"


@dataclass
class RAGAuditEntry:
    """Single audit log entry"""
    timestamp: datetime
    event_type: RAGEventType
    user_id: str
    request_id: str
    endpoint: str
    
    # Operation details
    operation: str
    query: Optional[str] = None
    databases_searched: Optional[List[str]] = None
    search_type: Optional[str] = None
    
    # Results
    status: str = "success"
    error_message: Optional[str] = None
    result_count: Optional[int] = None
    
    # Performance metrics
    latency_ms: Optional[float] = None
    tokens_used: Optional[int] = None
    estimated_cost: Optional[float] = None
    
    # Context
    ip_address: Optional[str] = None
    user_agent: Optional[str] = None
    api_key_hash: Optional[str] = None
    conversation_id: Optional[str] = None
    
    # Additional metadata
    metadata: Optional[Dict[str, Any]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for storage"""
        data = asdict(self)
        data['timestamp'] = self.timestamp.isoformat()
        data['event_type'] = self.event_type.value
        if self.databases_searched:
            data['databases_searched'] = json.dumps(self.databases_searched)
        if self.metadata:
            data['metadata'] = json.dumps(self.metadata)
        return data


class RAGAuditLogger:
    """Audit logger for RAG operations"""
    
    def __init__(self, db_path: Optional[str] = None, enabled: bool = True):
        """
        Initialize RAG audit logger.
        
        Args:
            db_path: Path to audit database
            enabled: Whether audit logging is enabled
        """
        self.enabled = enabled
        if not enabled:
            logger.info("RAG audit logging is disabled")
            return
        
        if db_path is None:
            db_dir = Path(os.getenv("AUDIT_LOG_PATH", "./Databases"))
            db_dir.mkdir(parents=True, exist_ok=True)
            db_path = db_dir / "rag_audit.db"
        
        self.db_path = str(db_path)
        self._init_database()
        self._write_queue = None  # Will be created when started
        self._writer_task = None
        logger.info(f"RAG audit logger initialized at {self.db_path}")
    
    def _init_database(self):
        """Initialize audit database with optimized schema"""
        with sqlite3.connect(self.db_path) as conn:
            # Main audit log table
            conn.execute("""
                CREATE TABLE IF NOT EXISTS rag_audit_log (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TIMESTAMP NOT NULL,
                    event_type TEXT NOT NULL,
                    user_id TEXT NOT NULL,
                    request_id TEXT NOT NULL,
                    endpoint TEXT NOT NULL,
                    
                    -- Operation details
                    operation TEXT NOT NULL,
                    query TEXT,
                    databases_searched TEXT,
                    search_type TEXT,
                    
                    -- Results
                    status TEXT NOT NULL,
                    error_message TEXT,
                    result_count INTEGER,
                    
                    -- Performance
                    latency_ms REAL,
                    tokens_used INTEGER,
                    estimated_cost REAL,
                    
                    -- Context
                    ip_address TEXT,
                    user_agent TEXT,
                    api_key_hash TEXT,
                    conversation_id TEXT,
                    
                    -- Metadata
                    metadata TEXT
                )
            """)
            
            # Create indexes separately
            conn.execute("CREATE INDEX IF NOT EXISTS idx_timestamp ON rag_audit_log(timestamp)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_user_id ON rag_audit_log(user_id)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_request_id ON rag_audit_log(request_id)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_event_type ON rag_audit_log(event_type)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_endpoint ON rag_audit_log(endpoint)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_conversation_id ON rag_audit_log(conversation_id)")
            
            # Daily aggregates for reporting
            conn.execute("""
                CREATE TABLE IF NOT EXISTS rag_daily_stats (
                    date DATE NOT NULL,
                    user_id TEXT NOT NULL,
                    endpoint TEXT NOT NULL,
                    
                    total_requests INTEGER DEFAULT 0,
                    successful_requests INTEGER DEFAULT 0,
                    failed_requests INTEGER DEFAULT 0,
                    
                    total_tokens INTEGER DEFAULT 0,
                    total_cost REAL DEFAULT 0.0,
                    
                    avg_latency_ms REAL,
                    p95_latency_ms REAL,
                    max_latency_ms REAL,
                    
                    PRIMARY KEY (date, user_id, endpoint)
                )
            """)
            
            # Suspicious activity tracking
            conn.execute("""
                CREATE TABLE IF NOT EXISTS rag_security_events (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TIMESTAMP NOT NULL,
                    user_id TEXT,
                    ip_address TEXT,
                    event_type TEXT NOT NULL,
                    severity TEXT NOT NULL,
                    description TEXT,
                    metadata TEXT,
                    resolved BOOLEAN DEFAULT FALSE
                )
            """)
            
            # Create security events indexes
            conn.execute("CREATE INDEX IF NOT EXISTS idx_sec_timestamp ON rag_security_events(timestamp)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_sec_severity ON rag_security_events(severity)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_sec_resolved ON rag_security_events(resolved)")
            
            conn.commit()
    
    async def start(self):
        """Start the async writer task"""
        if self.enabled and not self._writer_task:
            # Create queue in the current event loop
            if self._write_queue is None:
                self._write_queue = asyncio.Queue()
            self._writer_task = asyncio.create_task(self._writer_loop())
    
    async def stop(self):
        """Stop the async writer task"""
        if self._writer_task and self._write_queue:
            try:
                await self._write_queue.put(None)  # Sentinel
                await asyncio.wait_for(self._writer_task, timeout=5.0)
            except asyncio.TimeoutError:
                logger.warning("Audit logger writer task did not stop gracefully")
                self._writer_task.cancel()
            except Exception as e:
                logger.error(f"Error stopping audit logger: {e}")
            finally:
                self._writer_task = None
                self._write_queue = None
    
    async def _writer_loop(self):
        """Async loop for batch writing audit entries"""
        batch = []
        batch_size = 100
        flush_interval = 5.0  # seconds
        last_flush = asyncio.get_event_loop().time()
        
        while True:
            try:
                # Wait for entries with timeout
                timeout = flush_interval - (asyncio.get_event_loop().time() - last_flush)
                entry = await asyncio.wait_for(
                    self._write_queue.get(),
                    timeout=max(0.1, timeout)
                )
                
                if entry is None:  # Sentinel
                    break
                
                batch.append(entry)
                
                # Flush if batch is full
                if len(batch) >= batch_size:
                    await self._flush_batch(batch)
                    batch = []
                    last_flush = asyncio.get_event_loop().time()
                    
            except asyncio.TimeoutError:
                # Flush on timeout
                if batch:
                    await self._flush_batch(batch)
                    batch = []
                last_flush = asyncio.get_event_loop().time()
        
        # Final flush
        if batch:
            await self._flush_batch(batch)
    
    async def _flush_batch(self, batch: List[RAGAuditEntry]):
        """Write batch of entries to database"""
        if not batch:
            return
        
        try:
            with sqlite3.connect(self.db_path) as conn:
                # Prepare batch data
                records = [entry.to_dict() for entry in batch]
                
                # Batch insert
                conn.executemany("""
                    INSERT INTO rag_audit_log (
                        timestamp, event_type, user_id, request_id, endpoint,
                        operation, query, databases_searched, search_type,
                        status, error_message, result_count,
                        latency_ms, tokens_used, estimated_cost,
                        ip_address, user_agent, api_key_hash, conversation_id,
                        metadata
                    ) VALUES (
                        :timestamp, :event_type, :user_id, :request_id, :endpoint,
                        :operation, :query, :databases_searched, :search_type,
                        :status, :error_message, :result_count,
                        :latency_ms, :tokens_used, :estimated_cost,
                        :ip_address, :user_agent, :api_key_hash, :conversation_id,
                        :metadata
                    )
                """, records)
                
                # Update daily stats
                self._update_daily_stats(conn, batch)
                
                conn.commit()
                
        except Exception as e:
            logger.error(f"Failed to write audit batch: {e}")
    
    def _update_daily_stats(self, conn: sqlite3.Connection, batch: List[RAGAuditEntry]):
        """Update daily statistics from batch"""
        from collections import defaultdict
        
        # Aggregate by date, user, endpoint
        stats = defaultdict(lambda: {
            'requests': 0, 'success': 0, 'failed': 0,
            'tokens': 0, 'cost': 0.0, 'latencies': []
        })
        
        for entry in batch:
            date = entry.timestamp.date()
            key = (date, entry.user_id, entry.endpoint)
            
            stats[key]['requests'] += 1
            if entry.status == 'success':
                stats[key]['success'] += 1
            else:
                stats[key]['failed'] += 1
            
            if entry.tokens_used:
                stats[key]['tokens'] += entry.tokens_used
            if entry.estimated_cost:
                stats[key]['cost'] += entry.estimated_cost
            if entry.latency_ms:
                stats[key]['latencies'].append(entry.latency_ms)
        
        # Update database
        for (date, user_id, endpoint), data in stats.items():
            latencies = data['latencies']
            avg_latency = sum(latencies) / len(latencies) if latencies else None
            max_latency = max(latencies) if latencies else None
            p95_latency = sorted(latencies)[int(len(latencies) * 0.95)] if len(latencies) > 1 else max_latency
            
            conn.execute("""
                INSERT INTO rag_daily_stats (
                    date, user_id, endpoint,
                    total_requests, successful_requests, failed_requests,
                    total_tokens, total_cost,
                    avg_latency_ms, p95_latency_ms, max_latency_ms
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(date, user_id, endpoint) DO UPDATE SET
                    total_requests = total_requests + excluded.total_requests,
                    successful_requests = successful_requests + excluded.successful_requests,
                    failed_requests = failed_requests + excluded.failed_requests,
                    total_tokens = total_tokens + excluded.total_tokens,
                    total_cost = total_cost + excluded.total_cost,
                    avg_latency_ms = (avg_latency_ms * total_requests + excluded.avg_latency_ms * excluded.total_requests) 
                                     / (total_requests + excluded.total_requests),
                    p95_latency_ms = MAX(p95_latency_ms, excluded.p95_latency_ms),
                    max_latency_ms = MAX(max_latency_ms, excluded.max_latency_ms)
            """, (
                date, user_id, endpoint,
                data['requests'], data['success'], data['failed'],
                data['tokens'], data['cost'],
                avg_latency, p95_latency, max_latency
            ))
    
    async def log(self, entry: RAGAuditEntry):
        """Log an audit entry asynchronously"""
        if not self.enabled:
            return
        
        if self._write_queue is None:
            # Queue not yet initialized, create it
            self._write_queue = asyncio.Queue()
        
        await self._write_queue.put(entry)
    
    def log_sync(self, entry: RAGAuditEntry):
        """Log an audit entry synchronously (for use in sync contexts)"""
        if not self.enabled:
            return
        
        try:
            asyncio.create_task(self.log(entry))
        except RuntimeError:
            # Not in async context, write directly
            with sqlite3.connect(self.db_path) as conn:
                record = entry.to_dict()
                conn.execute("""
                    INSERT INTO rag_audit_log (
                        timestamp, event_type, user_id, request_id, endpoint,
                        operation, query, databases_searched, search_type,
                        status, error_message, result_count,
                        latency_ms, tokens_used, estimated_cost,
                        ip_address, user_agent, api_key_hash, conversation_id,
                        metadata
                    ) VALUES (
                        :timestamp, :event_type, :user_id, :request_id, :endpoint,
                        :operation, :query, :databases_searched, :search_type,
                        :status, :error_message, :result_count,
                        :latency_ms, :tokens_used, :estimated_cost,
                        :ip_address, :user_agent, :api_key_hash, :conversation_id,
                        :metadata
                    )
                """, record)
                conn.commit()
    
    def log_security_event(
        self,
        event_type: str,
        severity: str,
        description: str,
        user_id: Optional[str] = None,
        ip_address: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """Log a security event"""
        if not self.enabled:
            return
        
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                INSERT INTO rag_security_events (
                    timestamp, user_id, ip_address, event_type,
                    severity, description, metadata
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (
                datetime.now(timezone.utc),
                user_id,
                ip_address,
                event_type,
                severity,
                description,
                json.dumps(metadata) if metadata else None
            ))
            conn.commit()
    
    def get_user_stats(self, user_id: str, days: int = 30) -> Dict[str, Any]:
        """Get usage statistics for a user"""
        if not self.enabled:
            return {}
        
        with sqlite3.connect(self.db_path) as conn:
            # Get aggregated stats
            cursor = conn.execute("""
                SELECT 
                    COUNT(*) as total_requests,
                    SUM(CASE WHEN status = 'success' THEN 1 ELSE 0 END) as successful_requests,
                    SUM(tokens_used) as total_tokens,
                    SUM(estimated_cost) as total_cost,
                    AVG(latency_ms) as avg_latency,
                    MAX(latency_ms) as max_latency
                FROM rag_audit_log
                WHERE user_id = ? 
                AND timestamp > datetime('now', '-' || ? || ' days')
            """, (user_id, days))
            
            row = cursor.fetchone()
            if row:
                return {
                    'total_requests': row[0],
                    'successful_requests': row[1],
                    'total_tokens': row[2],
                    'total_cost': row[3],
                    'avg_latency_ms': row[4],
                    'max_latency_ms': row[5]
                }
        
        return {}
    
    def get_suspicious_activity(self, hours: int = 24) -> List[Dict[str, Any]]:
        """Get recent suspicious activity"""
        if not self.enabled:
            return []
        
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("""
                SELECT * FROM rag_security_events
                WHERE timestamp > datetime('now', '-' || ? || ' hours')
                AND resolved = FALSE
                ORDER BY timestamp DESC
            """, (hours,))
            
            columns = [desc[0] for desc in cursor.description]
            return [dict(zip(columns, row)) for row in cursor.fetchall()]


# Global singleton instance
_rag_audit_logger = None


def get_rag_audit_logger() -> RAGAuditLogger:
    """Get the global RAG audit logger instance"""
    global _rag_audit_logger
    if _rag_audit_logger is None:
        enabled = os.getenv("AUDIT_LOGGING_ENABLED", "true").lower() == "true"
        _rag_audit_logger = RAGAuditLogger(enabled=enabled)
    return _rag_audit_logger


@asynccontextmanager
async def audit_context(
    event_type: RAGEventType,
    user_id: str,
    request_id: str,
    endpoint: str,
    operation: str,
    **kwargs
):
    """Context manager for auditing operations with automatic timing"""
    start_time = datetime.now(timezone.utc)
    start_perf = asyncio.get_event_loop().time()
    
    entry = RAGAuditEntry(
        timestamp=start_time,
        event_type=event_type,
        user_id=user_id,
        request_id=request_id,
        endpoint=endpoint,
        operation=operation,
        **kwargs
    )
    
    try:
        yield entry
        entry.status = "success"
    except Exception as e:
        entry.status = "failure"
        entry.error_message = str(e)
        raise
    finally:
        # Calculate latency
        entry.latency_ms = (asyncio.get_event_loop().time() - start_perf) * 1000
        
        # Log the entry
        logger = get_rag_audit_logger()
        await logger.log(entry)