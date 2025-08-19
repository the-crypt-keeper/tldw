"""
Audit logging for Evaluations module.

Provides secure audit logging for all evaluation operations,
tracking user actions, API usage, and costs.
"""

import json
import sqlite3
from datetime import datetime, timezone
from typing import Dict, Any, Optional, List
from pathlib import Path
from loguru import logger
import hashlib
import os


class EvaluationAuditLogger:
    """Audit logger for evaluation operations."""
    
    def __init__(self, db_path: Optional[str] = None):
        """
        Initialize audit logger.
        
        Args:
            db_path: Path to audit database. If None, uses default.
        """
        if db_path is None:
            # Use evaluations database directory
            db_dir = Path(__file__).parent.parent.parent.parent / "Databases"
            db_dir.mkdir(parents=True, exist_ok=True)
            db_path = db_dir / "evaluation_audit.db"
        
        self.db_path = str(db_path)
        self._init_database()
    
    def _init_database(self):
        """Initialize audit database."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS audit_log (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TIMESTAMP NOT NULL,
                    event_type TEXT NOT NULL,
                    user_id TEXT,
                    api_key_hash TEXT,
                    evaluation_type TEXT,
                    evaluation_id TEXT,
                    request_id TEXT,
                    ip_address TEXT,
                    endpoint TEXT,
                    method TEXT,
                    status_code INTEGER,
                    error_message TEXT,
                    api_provider TEXT,
                    api_model TEXT,
                    token_count INTEGER,
                    estimated_cost REAL,
                    processing_time REAL,
                    metadata TEXT,
                    INDEX idx_timestamp (timestamp),
                    INDEX idx_user_id (user_id),
                    INDEX idx_evaluation_id (evaluation_id),
                    INDEX idx_event_type (event_type)
                )
            """)
            
            # Create cost tracking table
            conn.execute("""
                CREATE TABLE IF NOT EXISTS cost_tracking (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    date DATE NOT NULL,
                    user_id TEXT,
                    api_provider TEXT NOT NULL,
                    api_model TEXT NOT NULL,
                    total_tokens INTEGER DEFAULT 0,
                    total_cost REAL DEFAULT 0.0,
                    request_count INTEGER DEFAULT 0,
                    UNIQUE(date, user_id, api_provider, api_model)
                )
            """)
            
            conn.commit()
    
    def log_evaluation_request(
        self,
        event_type: str,
        user_id: Optional[str] = None,
        api_key: Optional[str] = None,
        evaluation_type: Optional[str] = None,
        evaluation_id: Optional[str] = None,
        request_id: Optional[str] = None,
        ip_address: Optional[str] = None,
        endpoint: Optional[str] = None,
        method: Optional[str] = None,
        status_code: Optional[int] = None,
        error_message: Optional[str] = None,
        api_provider: Optional[str] = None,
        api_model: Optional[str] = None,
        token_count: Optional[int] = None,
        estimated_cost: Optional[float] = None,
        processing_time: Optional[float] = None,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """
        Log an evaluation request for audit purposes.
        
        Args:
            event_type: Type of event (e.g., "evaluation_started", "evaluation_completed")
            user_id: User identifier
            api_key: API key (will be hashed)
            evaluation_type: Type of evaluation
            evaluation_id: Unique evaluation ID
            request_id: Request ID for correlation
            ip_address: Client IP address
            endpoint: API endpoint
            method: HTTP method
            status_code: HTTP status code
            error_message: Error message if failed
            api_provider: LLM provider used
            api_model: Model used
            token_count: Number of tokens used
            estimated_cost: Estimated cost in USD
            processing_time: Processing time in seconds
            metadata: Additional metadata
        """
        try:
            # Hash API key for security
            api_key_hash = None
            if api_key:
                api_key_hash = hashlib.sha256(api_key.encode()).hexdigest()[:16]
            
            # Prepare metadata
            metadata_json = json.dumps(metadata) if metadata else None
            
            # Insert audit log
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("""
                    INSERT INTO audit_log (
                        timestamp, event_type, user_id, api_key_hash,
                        evaluation_type, evaluation_id, request_id,
                        ip_address, endpoint, method, status_code,
                        error_message, api_provider, api_model,
                        token_count, estimated_cost, processing_time,
                        metadata
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    datetime.now(timezone.utc),
                    event_type,
                    user_id,
                    api_key_hash,
                    evaluation_type,
                    evaluation_id,
                    request_id,
                    ip_address,
                    endpoint,
                    method,
                    status_code,
                    error_message,
                    api_provider,
                    api_model,
                    token_count,
                    estimated_cost,
                    processing_time,
                    metadata_json
                ))
                
                # Update cost tracking if applicable
                if estimated_cost and estimated_cost > 0:
                    self._update_cost_tracking(
                        conn,
                        user_id,
                        api_provider,
                        api_model,
                        token_count,
                        estimated_cost
                    )
                
                conn.commit()
            
            # Log to standard logger for immediate visibility
            logger.info(
                f"Audit: {event_type} | User: {user_id} | "
                f"Type: {evaluation_type} | ID: {evaluation_id} | "
                f"Status: {status_code} | Cost: ${estimated_cost or 0:.4f}"
            )
            
        except Exception as e:
            logger.error(f"Failed to write audit log: {e}")
    
    def _update_cost_tracking(
        self,
        conn: sqlite3.Connection,
        user_id: Optional[str],
        api_provider: Optional[str],
        api_model: Optional[str],
        token_count: Optional[int],
        cost: float
    ):
        """Update cost tracking table."""
        if not api_provider or not api_model:
            return
        
        today = datetime.now(timezone.utc).date()
        
        conn.execute("""
            INSERT INTO cost_tracking (
                date, user_id, api_provider, api_model,
                total_tokens, total_cost, request_count
            ) VALUES (?, ?, ?, ?, ?, ?, 1)
            ON CONFLICT(date, user_id, api_provider, api_model) DO UPDATE SET
                total_tokens = total_tokens + ?,
                total_cost = total_cost + ?,
                request_count = request_count + 1
        """, (
            today, user_id, api_provider, api_model,
            token_count or 0, cost,
            token_count or 0, cost
        ))
    
    def get_cost_summary(
        self,
        user_id: Optional[str] = None,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None
    ) -> Dict[str, Any]:
        """
        Get cost summary for a user or all users.
        
        Args:
            user_id: Optional user ID to filter by
            start_date: Start date for filtering
            end_date: End date for filtering
            
        Returns:
            Cost summary with breakdown by provider and model
        """
        query = "SELECT * FROM cost_tracking WHERE 1=1"
        params = []
        
        if user_id:
            query += " AND user_id = ?"
            params.append(user_id)
        
        if start_date:
            query += " AND date >= ?"
            params.append(start_date.date())
        
        if end_date:
            query += " AND date <= ?"
            params.append(end_date.date())
        
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.execute(query, params)
            rows = cursor.fetchall()
        
        # Aggregate results
        summary = {
            "total_cost": 0.0,
            "total_tokens": 0,
            "total_requests": 0,
            "by_provider": {},
            "by_date": {}
        }
        
        for row in rows:
            row_dict = dict(row)
            provider = row_dict["api_provider"]
            model = row_dict["api_model"]
            date = row_dict["date"]
            
            # Update totals
            summary["total_cost"] += row_dict["total_cost"]
            summary["total_tokens"] += row_dict["total_tokens"]
            summary["total_requests"] += row_dict["request_count"]
            
            # By provider breakdown
            if provider not in summary["by_provider"]:
                summary["by_provider"][provider] = {
                    "total_cost": 0.0,
                    "total_tokens": 0,
                    "models": {}
                }
            
            summary["by_provider"][provider]["total_cost"] += row_dict["total_cost"]
            summary["by_provider"][provider]["total_tokens"] += row_dict["total_tokens"]
            
            if model not in summary["by_provider"][provider]["models"]:
                summary["by_provider"][provider]["models"][model] = {
                    "cost": 0.0,
                    "tokens": 0,
                    "requests": 0
                }
            
            summary["by_provider"][provider]["models"][model]["cost"] += row_dict["total_cost"]
            summary["by_provider"][provider]["models"][model]["tokens"] += row_dict["total_tokens"]
            summary["by_provider"][provider]["models"][model]["requests"] += row_dict["request_count"]
            
            # By date breakdown
            if date not in summary["by_date"]:
                summary["by_date"][date] = {
                    "cost": 0.0,
                    "tokens": 0,
                    "requests": 0
                }
            
            summary["by_date"][date]["cost"] += row_dict["total_cost"]
            summary["by_date"][date]["tokens"] += row_dict["total_tokens"]
            summary["by_date"][date]["requests"] += row_dict["request_count"]
        
        return summary
    
    def get_audit_logs(
        self,
        user_id: Optional[str] = None,
        evaluation_id: Optional[str] = None,
        event_type: Optional[str] = None,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        limit: int = 100,
        offset: int = 0
    ) -> List[Dict[str, Any]]:
        """
        Retrieve audit logs with filtering.
        
        Args:
            user_id: Filter by user ID
            evaluation_id: Filter by evaluation ID
            event_type: Filter by event type
            start_time: Start time for filtering
            end_time: End time for filtering
            limit: Maximum number of records
            offset: Offset for pagination
            
        Returns:
            List of audit log entries
        """
        query = "SELECT * FROM audit_log WHERE 1=1"
        params = []
        
        if user_id:
            query += " AND user_id = ?"
            params.append(user_id)
        
        if evaluation_id:
            query += " AND evaluation_id = ?"
            params.append(evaluation_id)
        
        if event_type:
            query += " AND event_type = ?"
            params.append(event_type)
        
        if start_time:
            query += " AND timestamp >= ?"
            params.append(start_time)
        
        if end_time:
            query += " AND timestamp <= ?"
            params.append(end_time)
        
        query += " ORDER BY timestamp DESC LIMIT ? OFFSET ?"
        params.extend([limit, offset])
        
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.execute(query, params)
            rows = cursor.fetchall()
        
        return [dict(row) for row in rows]


# Global audit logger instance
audit_logger = EvaluationAuditLogger()


def estimate_cost(
    provider: str,
    model: str,
    token_count: int
) -> float:
    """
    Estimate cost based on provider, model, and token count.
    
    Args:
        provider: LLM provider
        model: Model name
        token_count: Number of tokens
        
    Returns:
        Estimated cost in USD
    """
    # Cost per 1K tokens (approximate)
    cost_table = {
        "openai": {
            "gpt-4": 0.03,
            "gpt-4-turbo": 0.01,
            "gpt-3.5-turbo": 0.001,
            "text-embedding-3-small": 0.00002,
            "text-embedding-3-large": 0.00013
        },
        "anthropic": {
            "claude-3-opus": 0.015,
            "claude-3-sonnet": 0.003,
            "claude-3-haiku": 0.00025
        },
        "google": {
            "gemini-pro": 0.00025,
            "gemini-ultra": 0.007
        },
        "cohere": {
            "command": 0.0015,
            "embed": 0.0001
        }
    }
    
    # Get cost per 1K tokens
    provider_costs = cost_table.get(provider.lower(), {})
    cost_per_1k = provider_costs.get(model.lower(), 0.001)  # Default to $0.001 per 1K
    
    # Calculate total cost
    return (token_count / 1000.0) * cost_per_1k