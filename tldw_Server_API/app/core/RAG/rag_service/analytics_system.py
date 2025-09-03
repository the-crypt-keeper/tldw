# analytics_system.py
"""
Analytics and feedback system for RAG with dual storage:
1. Analytics.db - Server-side QA and anonymized metrics
2. ChaChaNotes_DB - User-specific feedback linked to conversations

This module replaces the old feedback_system.py with proper separation
of concerns between analytics and user data.
"""

import asyncio
import json
import time
import hashlib
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Set
from collections import defaultdict, deque
import sqlite3
from pathlib import Path
from contextlib import asynccontextmanager
import statistics

from loguru import logger
import numpy as np

from .analytics_db import get_analytics_db, AnalyticsDatabase


class FeedbackType(Enum):
    """Types of feedback that can be collected."""
    RELEVANCE = "relevance"  # 1-5 star rating
    HELPFUL = "helpful"  # Yes/No
    CLICK = "click"  # User clicked on result
    DWELL_TIME = "dwell_time"  # Time spent on result
    COPY = "copy"  # User copied text from result
    REPORT = "report"  # User reported an issue
    CITATION_USED = "citation_used"  # Citation was used in answer


class AnalyticsEventType(Enum):
    """Types of analytics events."""
    SEARCH = "search"
    FEEDBACK = "feedback"
    GENERATION = "generation"
    CITATION = "citation"
    CACHE_HIT = "cache_hit"
    CACHE_MISS = "cache_miss"
    ERROR = "error"
    PERFORMANCE = "performance"


@dataclass
class AnalyticsEvent:
    """An analytics event for server-side QA."""
    event_type: AnalyticsEventType
    timestamp: datetime = field(default_factory=datetime.now)
    query_hash: Optional[str] = None  # Hashed for privacy
    metrics: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for storage."""
        return {
            "event_type": self.event_type.value,
            "timestamp": self.timestamp.isoformat(),
            "query_hash": self.query_hash,
            "metrics": json.dumps(self.metrics)
        }


@dataclass
class UserFeedback:
    """User feedback linked to conversation."""
    conversation_id: str
    message_id: Optional[str]
    query: str
    document_ids: List[str]
    chunk_ids: List[str]
    relevance_score: Optional[int]  # 1-5
    helpful: Optional[bool]
    user_notes: Optional[str]
    created_at: datetime = field(default_factory=datetime.now)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for storage."""
        return {
            "conversation_id": self.conversation_id,
            "message_id": self.message_id,
            "query": self.query,
            "document_ids": json.dumps(self.document_ids),
            "chunk_ids": json.dumps(self.chunk_ids),
            "relevance_score": self.relevance_score,
            "helpful": self.helpful,
            "user_notes": self.user_notes,
            "created_at": self.created_at.isoformat()
        }


class AnalyticsStore:
    """
    Server-side analytics storage for QA and system improvement.
    No PII is stored - only anonymized metrics.
    Uses the new AnalyticsDatabase for storage.
    """
    
    def __init__(self, db_path: str = "Analytics.db"):
        """
        Initialize analytics store.
        
        Args:
            db_path: Path to Analytics database
        """
        self.db = get_analytics_db(db_path)
    
    async def record_search(self, search_data: Dict[str, Any]) -> bool:
        """
        Record search analytics.
        
        Args:
            search_data: Dictionary containing search metrics
            
        Returns:
            Success status
        """
        try:
            # Use sync method from AnalyticsDatabase
            await asyncio.get_event_loop().run_in_executor(
                None, self.db.record_search, search_data
            )
            return True
        except Exception as e:
            logger.error(f"Failed to record search analytics: {e}")
            return False
    
    async def record_feedback(self, feedback_data: Dict[str, Any]) -> bool:
        """
        Record anonymized feedback analytics.
        
        Args:
            feedback_data: Dictionary containing feedback data
            
        Returns:
            Success status
        """
        try:
            await asyncio.get_event_loop().run_in_executor(
                None, self.db.record_feedback, feedback_data
            )
            return True
        except Exception as e:
            logger.error(f"Failed to record feedback: {e}")
            return False
    
    async def record_document_performance(self, doc_data: Dict[str, Any]) -> bool:
        """
        Record document performance metrics.
        
        Args:
            doc_data: Dictionary containing document metrics
            
        Returns:
            Success status
        """
        try:
            await asyncio.get_event_loop().run_in_executor(
                None, self.db.record_document_performance, doc_data
            )
            return True
        except Exception as e:
            logger.error(f"Failed to record document performance: {e}")
            return False
    
    async def record_error(self, error_data: Dict[str, Any]) -> bool:
        """
        Record error tracking information.
        
        Args:
            error_data: Dictionary containing error information
            
        Returns:
            Success status
        """
        try:
            await asyncio.get_event_loop().run_in_executor(
                None, self.db.record_error, error_data
            )
            return True
        except Exception as e:
            logger.error(f"Failed to record error: {e}")
            return False
    
    async def record_feature_usage(self, feature_data: Dict[str, Any]) -> bool:
        """
        Record feature usage statistics.
        
        Args:
            feature_data: Dictionary containing feature usage data
            
        Returns:
            Success status
        """
        try:
            await asyncio.get_event_loop().run_in_executor(
                None, self.db.record_feature_usage, feature_data
            )
            return True
        except Exception as e:
            logger.error(f"Failed to record feature usage: {e}")
            return False
    
    async def get_analytics_summary(self, days: int = 7) -> Dict[str, Any]:
        """
        Get analytics summary for the specified period.
        
        Args:
            days: Number of days to include in summary
            
        Returns:
            Dictionary containing analytics summary
        """
        try:
            summary = await asyncio.get_event_loop().run_in_executor(
                None, self.db.get_analytics_summary, days
            )
            return summary
        except Exception as e:
            logger.error(f"Failed to get analytics summary: {e}")
            return {}
    
    async def cleanup_old_data(self, days_to_keep: int = 90) -> int:
        """
        Clean up old analytics data.
        
        Args:
            days_to_keep: Number of days of data to retain
            
        Returns:
            Number of records deleted
        """
        try:
            deleted = await asyncio.get_event_loop().run_in_executor(
                None, self.db.cleanup_old_data, days_to_keep
            )
            return deleted
        except Exception as e:
            logger.error(f"Failed to cleanup old data: {e}")
            return 0


class UserFeedbackStore:
    """
    Stores user-specific feedback in ChaChaNotes_DB.
    Links feedback to conversations for context.
    """
    
    def __init__(self, chacha_db):
        """
        Initialize user feedback store.
        
        Args:
            chacha_db: ChaChaNotes database instance
        """
        self.db = chacha_db
        self._init_schema()
    
    def _init_schema(self):
        """Ensure feedback tables exist in ChaChaNotes_DB."""
        try:
            with sqlite3.connect(self.db.db_path) as conn:
                # Create conversation feedback table
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS conversation_feedback (
                        id TEXT PRIMARY KEY,
                        conversation_id TEXT NOT NULL,
                        message_id TEXT,
                        query TEXT,
                        document_ids TEXT,  -- JSON array
                        chunk_ids TEXT,  -- JSON array
                        relevance_score INTEGER CHECK(relevance_score BETWEEN 1 AND 5),
                        helpful INTEGER,  -- 0 or 1 for false/true
                        user_notes TEXT,
                        created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                        FOREIGN KEY (conversation_id) REFERENCES conversations(id)
                    )
                """)
                
                # Create indexes
                conn.execute("CREATE INDEX IF NOT EXISTS idx_feedback_conv ON conversation_feedback(conversation_id)")
                conn.execute("CREATE INDEX IF NOT EXISTS idx_feedback_created ON conversation_feedback(created_at)")
                
                conn.commit()
                
        except Exception as e:
            logger.error(f"Failed to initialize feedback schema: {e}")
    
    async def add_feedback(
        self,
        conversation_id: str,
        query: str,
        document_ids: List[str],
        chunk_ids: List[str],
        relevance_score: Optional[int] = None,
        helpful: Optional[bool] = None,
        user_notes: Optional[str] = None,
        message_id: Optional[str] = None
    ) -> str:
        """
        Add feedback for a conversation.
        
        Returns:
            Feedback ID
        """
        feedback_id = f"fb_{int(time.time() * 1000)}_{hashlib.md5(query.encode()).hexdigest()[:8]}"
        
        try:
            with sqlite3.connect(self.db.db_path) as conn:
                conn.execute(
                    """
                    INSERT INTO conversation_feedback 
                    (id, conversation_id, message_id, query, document_ids, chunk_ids,
                     relevance_score, helpful, user_notes)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        feedback_id,
                        conversation_id,
                        message_id,
                        query,
                        json.dumps(document_ids),
                        json.dumps(chunk_ids),
                        relevance_score,
                        1 if helpful else 0 if helpful is not None else None,
                        user_notes
                    )
                )
                conn.commit()
                
                logger.info(f"Added feedback {feedback_id} for conversation {conversation_id}")
                return feedback_id
                
        except Exception as e:
            logger.error(f"Failed to add feedback: {e}")
            raise
    
    async def get_conversation_feedback(
        self,
        conversation_id: str
    ) -> List[Dict[str, Any]]:
        """Get all feedback for a conversation."""
        try:
            with sqlite3.connect(self.db.db_path) as conn:
                conn.row_factory = sqlite3.Row
                cursor = conn.execute(
                    """
                    SELECT * FROM conversation_feedback 
                    WHERE conversation_id = ?
                    ORDER BY created_at DESC
                    """,
                    (conversation_id,)
                )
                
                feedback = []
                for row in cursor.fetchall():
                    fb = dict(row)
                    # Parse JSON fields
                    fb["document_ids"] = json.loads(fb["document_ids"])
                    fb["chunk_ids"] = json.loads(fb["chunk_ids"])
                    fb["helpful"] = bool(fb["helpful"]) if fb["helpful"] is not None else None
                    feedback.append(fb)
                
                return feedback
                
        except Exception as e:
            logger.error(f"Failed to get conversation feedback: {e}")
            return []


class UnifiedFeedbackSystem:
    """
    Unified feedback system that manages both:
    1. Analytics.db for server-side QA
    2. ChaChaNotes_DB for user-specific feedback
    """
    
    def __init__(
        self,
        analytics_db_path: str = "Analytics.db",
        chacha_db = None,
        enable_analytics: bool = True
    ):
        """
        Initialize unified feedback system.
        
        Args:
            analytics_db_path: Path to analytics database
            chacha_db: ChaChaNotes database instance
            enable_analytics: Whether to enable server-side analytics
        """
        self.enable_analytics = enable_analytics
        
        if enable_analytics:
            self.analytics = AnalyticsStore(analytics_db_path)
        else:
            self.analytics = None
        
        if chacha_db:
            self.user_feedback = UserFeedbackStore(chacha_db)
        else:
            self.user_feedback = None
        
        # Performance tracking
        self._performance_buffer = deque(maxlen=100)
        self._error_buffer = deque(maxlen=50)
    
    async def submit_feedback(
        self,
        conversation_id: str,
        query: str,
        document_ids: List[str],
        chunk_ids: List[str],
        relevance_score: Optional[int] = None,
        helpful: Optional[bool] = None,
        user_notes: Optional[str] = None,
        user_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Submit feedback to both stores.
        
        Returns:
            Result with feedback ID and status
        """
        result = {"success": False, "feedback_id": None, "errors": []}
        
        # Store in user's conversation DB
        if self.user_feedback and conversation_id:
            try:
                feedback_id = await self.user_feedback.add_feedback(
                    conversation_id=conversation_id,
                    query=query,
                    document_ids=document_ids,
                    chunk_ids=chunk_ids,
                    relevance_score=relevance_score,
                    helpful=helpful,
                    user_notes=user_notes
                )
                result["feedback_id"] = feedback_id
                result["success"] = True
                
            except Exception as e:
                result["errors"].append(f"User feedback error: {str(e)}")
        
        # Store anonymized metrics in Analytics DB
        if self.enable_analytics and self.analytics:
            try:
                # Hash query for privacy
                query_hash = hashlib.sha256(query.encode()).hexdigest()
                
                # Record search quality
                if relevance_score:
                    await self.analytics.record_search_quality(
                        query_hash=query_hash,
                        relevance_score=relevance_score / 5.0,  # Normalize to 0-1
                        clicked=len(chunk_ids) > 0
                    )
                
                # Record document performance
                for doc_id in document_ids:
                    await self.analytics.record_document_performance(
                        document_id=doc_id,
                        relevance_score=relevance_score / 5.0 if relevance_score else None,
                        positive_feedback=helpful is True,
                        negative_feedback=helpful is False
                    )
                
                # Record feedback event
                event = AnalyticsEvent(
                    event_type=AnalyticsEventType.FEEDBACK,
                    query_hash=query_hash,
                    metrics={
                        "relevance": relevance_score,
                        "helpful": helpful,
                        "chunks_used": len(chunk_ids),
                        "has_notes": bool(user_notes)
                    }
                )
                await self.analytics.record_event(event)
                
            except Exception as e:
                result["errors"].append(f"Analytics error: {str(e)}")
        
        return result
    
    async def record_search(
        self,
        query: str,
        results_count: int,
        cache_hit: bool,
        latency_ms: float
    ):
        """Record a search event."""
        if self.enable_analytics and self.analytics:
            query_hash = hashlib.sha256(query.encode()).hexdigest()
            
            event = AnalyticsEvent(
                event_type=AnalyticsEventType.SEARCH,
                query_hash=query_hash,
                metrics={
                    "results_count": results_count,
                    "cache_hit": cache_hit,
                    "latency_ms": latency_ms
                }
            )
            await self.analytics.record_event(event)
            
            # Record performance metric
            await self.analytics.record_performance_metric(
                metric_type="search_latency",
                value=latency_ms,
                metadata={"cache_hit": cache_hit}
            )
    
    async def record_citation_usage(
        self,
        document_ids: List[str],
        chunk_ids: List[str],
        citation_style: str
    ):
        """Record citation usage."""
        if self.enable_analytics and self.analytics:
            # Record citation event
            event = AnalyticsEvent(
                event_type=AnalyticsEventType.CITATION,
                metrics={
                    "documents_cited": len(document_ids),
                    "chunks_cited": len(chunk_ids),
                    "style": citation_style
                }
            )
            await self.analytics.record_event(event)
            
            # Update document performance
            for doc_id in document_ids:
                await self.analytics.record_document_performance(
                    document_id=doc_id,
                    cited=True
                )
    
    async def record_error(
        self,
        error_type: str,
        error_message: str,
        context: Optional[Dict[str, Any]] = None
    ):
        """Record an error occurrence."""
        if self.enable_analytics and self.analytics:
            await self.analytics.record_error(
                error_type=error_type,
                error_message=error_message,
                metadata=context
            )
    
    async def get_analytics_dashboard(self) -> Dict[str, Any]:
        """Get analytics dashboard data."""
        if not self.enable_analytics or not self.analytics:
            return {"error": "Analytics not enabled"}
        
        return await self.analytics.get_analytics_summary()
    
    async def get_conversation_insights(
        self,
        conversation_id: str
    ) -> Dict[str, Any]:
        """Get insights for a specific conversation."""
        insights = {
            "feedback_count": 0,
            "average_relevance": 0,
            "helpful_percentage": 0,
            "feedback_history": []
        }
        
        if self.user_feedback:
            feedback = await self.user_feedback.get_conversation_feedback(conversation_id)
            insights["feedback_count"] = len(feedback)
            insights["feedback_history"] = feedback
            
            if feedback:
                relevance_scores = [f["relevance_score"] for f in feedback if f["relevance_score"]]
                if relevance_scores:
                    insights["average_relevance"] = statistics.mean(relevance_scores)
                
                helpful_votes = [f["helpful"] for f in feedback if f["helpful"] is not None]
                if helpful_votes:
                    insights["helpful_percentage"] = sum(helpful_votes) / len(helpful_votes) * 100
        
        return insights


# Global instance management
_feedback_system: Optional[UnifiedFeedbackSystem] = None


def get_feedback_system(
    analytics_db_path: str = "Analytics.db",
    chacha_db = None,
    enable_analytics: bool = True
) -> UnifiedFeedbackSystem:
    """Get or create global feedback system instance."""
    global _feedback_system
    if _feedback_system is None:
        _feedback_system = UnifiedFeedbackSystem(
            analytics_db_path=analytics_db_path,
            chacha_db=chacha_db,
            enable_analytics=enable_analytics
        )
    return _feedback_system


# Pipeline integration functions

async def collect_feedback(context: Any, **kwargs) -> Any:
    """Collect feedback in RAG pipeline."""
    if not context.config.get("feedback", {}).get("enabled", False):
        return context
    
    feedback_system = get_feedback_system(
        chacha_db=context.config.get("chacha_db"),
        enable_analytics=context.config.get("enable_analytics", True)
    )
    
    # Submit feedback if provided
    if "feedback_data" in context.metadata:
        fb = context.metadata["feedback_data"]
        
        result = await feedback_system.submit_feedback(
            conversation_id=fb.get("conversation_id", ""),
            query=context.query,
            document_ids=[doc.id for doc in context.documents],
            chunk_ids=[doc.id for doc in context.documents],  # Assuming doc.id is chunk_id
            relevance_score=fb.get("relevance_score"),
            helpful=fb.get("helpful"),
            user_notes=fb.get("user_notes"),
            user_id=fb.get("user_id")
        )
        
        context.metadata["feedback_result"] = result
    
    # Record search metrics
    await feedback_system.record_search(
        query=context.query,
        results_count=len(context.documents),
        cache_hit=context.cache_hit,
        latency_ms=context.timings.get("total", 0) * 1000
    )
    
    return context


async def apply_feedback_boost(context: Any, **kwargs) -> Any:
    """Apply feedback-based boosting to search results."""
    if not context.config.get("feedback", {}).get("apply_boost", False):
        return context
    
    # This would require accessing historical feedback data
    # to boost documents that have received positive feedback
    # Implementation depends on specific requirements
    
    return context