"""
Analytics database schema and management for server-side QA metrics.

This module handles the creation and management of the Analytics.db database
for storing anonymized metrics and quality assurance data.
"""

import sqlite3
import hashlib
import json
from datetime import datetime
from typing import Dict, Any, List, Optional, Tuple
from pathlib import Path
import threading
from contextlib import contextmanager

from loguru import logger


class AnalyticsDatabase:
    """
    Manages the Analytics.db database for server-side QA metrics.
    
    This database stores anonymized metrics for improving the RAG system,
    including search quality, document performance, and error tracking.
    """
    
    def __init__(self, db_path: str = "Analytics.db"):
        """
        Initialize the Analytics database.
        
        Args:
            db_path: Path to the Analytics.db file
        """
        self.db_path = db_path
        self._local = threading.local()
        self._lock = threading.RLock()
        
        # Create database and tables
        self._initialize_database()
    
    @property
    def connection(self) -> sqlite3.Connection:
        """Get thread-local database connection."""
        if not hasattr(self._local, 'connection'):
            self._local.connection = sqlite3.connect(
                self.db_path,
                check_same_thread=False,
                isolation_level=None  # Autocommit mode
            )
            self._local.connection.row_factory = sqlite3.Row
            # Enable foreign keys
            self._local.connection.execute("PRAGMA foreign_keys = ON")
            # Optimize for concurrent access
            self._local.connection.execute("PRAGMA journal_mode = WAL")
            self._local.connection.execute("PRAGMA synchronous = NORMAL")
        return self._local.connection
    
    @contextmanager
    def transaction(self):
        """Context manager for database transactions."""
        conn = self.connection
        try:
            conn.execute("BEGIN")
            yield conn
            conn.execute("COMMIT")
        except Exception as e:
            conn.execute("ROLLBACK")
            logger.error(f"Transaction failed: {e}")
            raise
    
    def _initialize_database(self):
        """Create database tables if they don't exist."""
        with self._lock:
            conn = self.connection
            cursor = conn.cursor()
            
            # Search Analytics table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS search_analytics (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    query_hash TEXT NOT NULL,
                    query_length INTEGER,
                    query_complexity TEXT,
                    search_type TEXT,
                    results_count INTEGER,
                    max_score REAL,
                    avg_score REAL,
                    response_time_ms INTEGER,
                    cache_hit BOOLEAN,
                    reranking_used BOOLEAN,
                    expansion_used BOOLEAN,
                    filters_used TEXT,
                    error_occurred BOOLEAN DEFAULT FALSE,
                    error_type TEXT
                )
            """)
            
            # Document Performance table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS document_performance (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    document_hash TEXT NOT NULL,
                    document_type TEXT,
                    chunk_size INTEGER,
                    retrieval_count INTEGER DEFAULT 0,
                    avg_relevance_score REAL,
                    citation_count INTEGER DEFAULT 0,
                    feedback_positive INTEGER DEFAULT 0,
                    feedback_negative INTEGER DEFAULT 0,
                    last_accessed TIMESTAMP
                )
            """)
            
            # User Feedback Analytics table (anonymized)
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS feedback_analytics (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    session_hash TEXT NOT NULL,
                    query_hash TEXT NOT NULL,
                    feedback_type TEXT,
                    rating INTEGER,
                    response_quality TEXT,
                    retrieval_accuracy TEXT,
                    response_time_acceptable BOOLEAN,
                    categories TEXT,
                    improvement_areas TEXT
                )
            """)
            
            # Citation Analytics table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS citation_analytics (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    citation_style TEXT,
                    document_count INTEGER,
                    chunk_count INTEGER,
                    verification_requested BOOLEAN,
                    format_errors INTEGER DEFAULT 0,
                    generation_time_ms INTEGER
                )
            """)
            
            # Error Tracking table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS error_tracking (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    error_type TEXT NOT NULL,
                    error_category TEXT,
                    error_hash TEXT,
                    component TEXT,
                    severity TEXT,
                    frequency INTEGER DEFAULT 1,
                    resolved BOOLEAN DEFAULT FALSE,
                    resolution_time TIMESTAMP,
                    stack_trace_hash TEXT
                )
            """)
            
            # System Performance table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS system_performance (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    metric_type TEXT NOT NULL,
                    metric_value REAL,
                    component TEXT,
                    operation TEXT,
                    duration_ms INTEGER,
                    memory_mb REAL,
                    cpu_percent REAL
                )
            """)
            
            # Feature Usage table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS feature_usage (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    feature_name TEXT NOT NULL,
                    usage_count INTEGER DEFAULT 1,
                    success_count INTEGER DEFAULT 0,
                    failure_count INTEGER DEFAULT 0,
                    avg_execution_time_ms INTEGER,
                    last_used TIMESTAMP
                )
            """)
            
            # Query Patterns table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS query_patterns (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    pattern_hash TEXT UNIQUE NOT NULL,
                    pattern_type TEXT,
                    frequency INTEGER DEFAULT 1,
                    avg_results_quality REAL,
                    avg_response_time_ms INTEGER,
                    first_seen TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    last_seen TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # A/B Testing table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS ab_testing (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    test_name TEXT NOT NULL,
                    variant TEXT NOT NULL,
                    session_hash TEXT,
                    metric_name TEXT,
                    metric_value REAL,
                    conversion BOOLEAN
                )
            """)
            
            # Create indexes for performance
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_search_timestamp 
                ON search_analytics(timestamp DESC)
            """)
            
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_document_hash 
                ON document_performance(document_hash)
            """)
            
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_feedback_session 
                ON feedback_analytics(session_hash)
            """)
            
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_error_type 
                ON error_tracking(error_type, timestamp DESC)
            """)
            
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_feature_name 
                ON feature_usage(feature_name, timestamp DESC)
            """)
            
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_query_pattern 
                ON query_patterns(pattern_hash)
            """)
            
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_ab_test 
                ON ab_testing(test_name, variant)
            """)
            
            logger.info(f"Analytics database initialized at {self.db_path}")
    
    def record_search(self, search_data: Dict[str, Any]) -> None:
        """
        Record search analytics.
        
        Args:
            search_data: Dictionary containing search metrics
        """
        try:
            with self.transaction() as conn:
                cursor = conn.cursor()
                
                # Hash the query for privacy
                query_hash = hashlib.sha256(
                    search_data.get('query', '').encode()
                ).hexdigest()[:16]
                
                cursor.execute("""
                    INSERT INTO search_analytics (
                        query_hash, query_length, query_complexity, search_type,
                        results_count, max_score, avg_score, response_time_ms,
                        cache_hit, reranking_used, expansion_used, filters_used,
                        error_occurred, error_type
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    query_hash,
                    search_data.get('query_length'),
                    search_data.get('query_complexity'),
                    search_data.get('search_type'),
                    search_data.get('results_count'),
                    search_data.get('max_score'),
                    search_data.get('avg_score'),
                    search_data.get('response_time_ms'),
                    search_data.get('cache_hit', False),
                    search_data.get('reranking_used', False),
                    search_data.get('expansion_used', False),
                    json.dumps(search_data.get('filters_used', [])),
                    search_data.get('error_occurred', False),
                    search_data.get('error_type')
                ))
                
                logger.debug(f"Recorded search analytics for query hash: {query_hash}")
        except Exception as e:
            logger.error(f"Failed to record search analytics: {e}")
    
    def record_document_performance(self, doc_data: Dict[str, Any]) -> None:
        """
        Record document performance metrics.
        
        Args:
            doc_data: Dictionary containing document metrics
        """
        try:
            with self.transaction() as conn:
                cursor = conn.cursor()
                
                # Hash document ID for privacy
                doc_hash = hashlib.sha256(
                    str(doc_data.get('document_id', '')).encode()
                ).hexdigest()[:16]
                
                # Check if document already exists
                cursor.execute("""
                    SELECT id, retrieval_count, citation_count, 
                           feedback_positive, feedback_negative
                    FROM document_performance 
                    WHERE document_hash = ?
                """, (doc_hash,))
                
                existing = cursor.fetchone()
                
                if existing:
                    # Update existing record
                    cursor.execute("""
                        UPDATE document_performance 
                        SET retrieval_count = retrieval_count + ?,
                            citation_count = citation_count + ?,
                            feedback_positive = feedback_positive + ?,
                            feedback_negative = feedback_negative + ?,
                            avg_relevance_score = 
                                (avg_relevance_score * retrieval_count + ?) / 
                                (retrieval_count + 1),
                            last_accessed = CURRENT_TIMESTAMP
                        WHERE document_hash = ?
                    """, (
                        1 if doc_data.get('retrieved') else 0,
                        1 if doc_data.get('cited') else 0,
                        1 if doc_data.get('feedback') == 'positive' else 0,
                        1 if doc_data.get('feedback') == 'negative' else 0,
                        doc_data.get('relevance_score', 0),
                        doc_hash
                    ))
                else:
                    # Insert new record
                    cursor.execute("""
                        INSERT INTO document_performance (
                            document_hash, document_type, chunk_size,
                            retrieval_count, avg_relevance_score, citation_count,
                            feedback_positive, feedback_negative, last_accessed
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
                    """, (
                        doc_hash,
                        doc_data.get('document_type'),
                        doc_data.get('chunk_size'),
                        1 if doc_data.get('retrieved') else 0,
                        doc_data.get('relevance_score', 0),
                        1 if doc_data.get('cited') else 0,
                        1 if doc_data.get('feedback') == 'positive' else 0,
                        1 if doc_data.get('feedback') == 'negative' else 0
                    ))
                
                logger.debug(f"Recorded document performance for hash: {doc_hash}")
        except Exception as e:
            logger.error(f"Failed to record document performance: {e}")
    
    def record_feedback(self, feedback_data: Dict[str, Any]) -> None:
        """
        Record anonymized feedback analytics.
        
        Args:
            feedback_data: Dictionary containing feedback data
        """
        try:
            with self.transaction() as conn:
                cursor = conn.cursor()
                
                # Hash identifiers for privacy
                session_hash = hashlib.sha256(
                    str(feedback_data.get('session_id', '')).encode()
                ).hexdigest()[:16]
                
                query_hash = hashlib.sha256(
                    feedback_data.get('query', '').encode()
                ).hexdigest()[:16]
                
                cursor.execute("""
                    INSERT INTO feedback_analytics (
                        session_hash, query_hash, feedback_type, rating,
                        response_quality, retrieval_accuracy, 
                        response_time_acceptable, categories, improvement_areas
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    session_hash,
                    query_hash,
                    feedback_data.get('feedback_type'),
                    feedback_data.get('rating'),
                    feedback_data.get('response_quality'),
                    feedback_data.get('retrieval_accuracy'),
                    feedback_data.get('response_time_acceptable'),
                    json.dumps(feedback_data.get('categories', [])),
                    json.dumps(feedback_data.get('improvement_areas', []))
                ))
                
                logger.debug(f"Recorded feedback for session: {session_hash}")
        except Exception as e:
            logger.error(f"Failed to record feedback: {e}")
    
    def record_error(self, error_data: Dict[str, Any]) -> None:
        """
        Record error tracking information.
        
        Args:
            error_data: Dictionary containing error information
        """
        try:
            with self.transaction() as conn:
                cursor = conn.cursor()
                
                # Hash error for deduplication
                error_hash = hashlib.sha256(
                    f"{error_data.get('error_type', '')}:"
                    f"{error_data.get('component', '')}".encode()
                ).hexdigest()[:16]
                
                # Hash stack trace if present
                stack_hash = None
                if error_data.get('stack_trace'):
                    stack_hash = hashlib.sha256(
                        error_data['stack_trace'].encode()
                    ).hexdigest()[:16]
                
                # Check if error already exists
                cursor.execute("""
                    SELECT id, frequency FROM error_tracking 
                    WHERE error_hash = ? AND resolved = FALSE
                """, (error_hash,))
                
                existing = cursor.fetchone()
                
                if existing:
                    # Update frequency
                    cursor.execute("""
                        UPDATE error_tracking 
                        SET frequency = frequency + 1,
                            timestamp = CURRENT_TIMESTAMP
                        WHERE id = ?
                    """, (existing['id'],))
                else:
                    # Insert new error
                    cursor.execute("""
                        INSERT INTO error_tracking (
                            error_type, error_category, error_hash, component,
                            severity, frequency, stack_trace_hash
                        ) VALUES (?, ?, ?, ?, ?, 1, ?)
                    """, (
                        error_data.get('error_type'),
                        error_data.get('error_category'),
                        error_hash,
                        error_data.get('component'),
                        error_data.get('severity', 'medium'),
                        stack_hash
                    ))
                
                logger.debug(f"Recorded error: {error_hash}")
        except Exception as e:
            logger.error(f"Failed to record error: {e}")
    
    def record_feature_usage(self, feature_data: Dict[str, Any]) -> None:
        """
        Record feature usage statistics.
        
        Args:
            feature_data: Dictionary containing feature usage data
        """
        try:
            with self.transaction() as conn:
                cursor = conn.cursor()
                
                feature_name = feature_data.get('feature_name')
                
                # Check if feature exists
                cursor.execute("""
                    SELECT id FROM feature_usage 
                    WHERE feature_name = ? 
                    AND DATE(timestamp) = DATE('now')
                """, (feature_name,))
                
                existing = cursor.fetchone()
                
                if existing:
                    # Update existing record for today
                    cursor.execute("""
                        UPDATE feature_usage 
                        SET usage_count = usage_count + 1,
                            success_count = success_count + ?,
                            failure_count = failure_count + ?,
                            avg_execution_time_ms = 
                                (avg_execution_time_ms * usage_count + ?) / 
                                (usage_count + 1),
                            last_used = CURRENT_TIMESTAMP
                        WHERE id = ?
                    """, (
                        1 if feature_data.get('success') else 0,
                        1 if not feature_data.get('success') else 0,
                        feature_data.get('execution_time_ms', 0),
                        existing['id']
                    ))
                else:
                    # Insert new record
                    cursor.execute("""
                        INSERT INTO feature_usage (
                            feature_name, usage_count, success_count, 
                            failure_count, avg_execution_time_ms, last_used
                        ) VALUES (?, 1, ?, ?, ?, CURRENT_TIMESTAMP)
                    """, (
                        feature_name,
                        1 if feature_data.get('success') else 0,
                        1 if not feature_data.get('success') else 0,
                        feature_data.get('execution_time_ms', 0)
                    ))
                
                logger.debug(f"Recorded feature usage: {feature_name}")
        except Exception as e:
            logger.error(f"Failed to record feature usage: {e}")
    
    def get_analytics_summary(self, days: int = 7) -> Dict[str, Any]:
        """
        Get analytics summary for the specified period.
        
        Args:
            days: Number of days to include in summary
            
        Returns:
            Dictionary containing analytics summary
        """
        try:
            with self._lock:
                cursor = self.connection.cursor()
                
                # Search analytics
                cursor.execute("""
                    SELECT 
                        COUNT(*) as total_searches,
                        AVG(response_time_ms) as avg_response_time,
                        AVG(results_count) as avg_results,
                        SUM(cache_hit) * 100.0 / COUNT(*) as cache_hit_rate,
                        SUM(error_occurred) * 100.0 / COUNT(*) as error_rate
                    FROM search_analytics
                    WHERE timestamp > datetime('now', '-' || ? || ' days')
                """, (days,))
                search_stats = dict(cursor.fetchone())
                
                # Document performance
                cursor.execute("""
                    SELECT 
                        COUNT(*) as total_documents,
                        SUM(retrieval_count) as total_retrievals,
                        SUM(citation_count) as total_citations,
                        AVG(avg_relevance_score) as avg_relevance,
                        SUM(feedback_positive) as positive_feedback,
                        SUM(feedback_negative) as negative_feedback
                    FROM document_performance
                    WHERE last_accessed > datetime('now', '-' || ? || ' days')
                """, (days,))
                doc_stats = dict(cursor.fetchone())
                
                # Feature usage
                cursor.execute("""
                    SELECT 
                        feature_name,
                        SUM(usage_count) as total_usage,
                        SUM(success_count) * 100.0 / SUM(usage_count) as success_rate
                    FROM feature_usage
                    WHERE timestamp > datetime('now', '-' || ? || ' days')
                    GROUP BY feature_name
                    ORDER BY total_usage DESC
                    LIMIT 10
                """, (days,))
                top_features = [dict(row) for row in cursor.fetchall()]
                
                # Error summary
                cursor.execute("""
                    SELECT 
                        error_type,
                        SUM(frequency) as total_occurrences,
                        COUNT(DISTINCT component) as affected_components
                    FROM error_tracking
                    WHERE timestamp > datetime('now', '-' || ? || ' days')
                        AND resolved = FALSE
                    GROUP BY error_type
                    ORDER BY total_occurrences DESC
                    LIMIT 5
                """, (days,))
                top_errors = [dict(row) for row in cursor.fetchall()]
                
                return {
                    'period_days': days,
                    'search_analytics': search_stats,
                    'document_performance': doc_stats,
                    'top_features': top_features,
                    'top_errors': top_errors,
                    'generated_at': datetime.now().isoformat()
                }
        except Exception as e:
            logger.error(f"Failed to get analytics summary: {e}")
            return {}
    
    def cleanup_old_data(self, days_to_keep: int = 90) -> int:
        """
        Clean up old analytics data.
        
        Args:
            days_to_keep: Number of days of data to retain
            
        Returns:
            Number of records deleted
        """
        try:
            with self.transaction() as conn:
                cursor = conn.cursor()
                total_deleted = 0
                
                tables = [
                    'search_analytics', 'document_performance', 
                    'feedback_analytics', 'citation_analytics',
                    'error_tracking', 'system_performance',
                    'feature_usage', 'ab_testing'
                ]
                
                for table in tables:
                    cursor.execute(f"""
                        DELETE FROM {table}
                        WHERE timestamp < datetime('now', '-' || ? || ' days')
                    """, (days_to_keep,))
                    
                    deleted = cursor.rowcount
                    total_deleted += deleted
                    
                    if deleted > 0:
                        logger.info(f"Deleted {deleted} old records from {table}")
                
                # Vacuum to reclaim space
                conn.execute("VACUUM")
                
                logger.info(f"Cleanup complete. Deleted {total_deleted} total records")
                return total_deleted
        except Exception as e:
            logger.error(f"Failed to cleanup old data: {e}")
            return 0
    
    def close(self):
        """Close database connection."""
        if hasattr(self._local, 'connection'):
            self._local.connection.close()
            delattr(self._local, 'connection')
            logger.debug("Analytics database connection closed")


# Singleton instance
_analytics_db: Optional[AnalyticsDatabase] = None
_analytics_lock = threading.Lock()


def get_analytics_db(db_path: str = "Analytics.db") -> AnalyticsDatabase:
    """
    Get or create the singleton Analytics database instance.
    
    Args:
        db_path: Path to the Analytics.db file
        
    Returns:
        AnalyticsDatabase instance
    """
    global _analytics_db
    
    with _analytics_lock:
        if _analytics_db is None:
            _analytics_db = AnalyticsDatabase(db_path)
        return _analytics_db