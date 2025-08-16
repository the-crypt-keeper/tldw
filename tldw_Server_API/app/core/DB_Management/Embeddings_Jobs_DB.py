# Embeddings_Jobs_DB.py
"""
Database schema and management for embedding jobs and user quotas.

This module provides the database layer for tracking embedding jobs,
user quotas, and job history.
"""

import sqlite3
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from contextlib import contextmanager
from pathlib import Path

from loguru import logger


class EmbeddingsJobsDatabase:
    """Database manager for embedding jobs and quotas"""
    
    def __init__(self, db_path: str = "./Databases/embeddings_jobs.db"):
        """Initialize the database connection and create tables if needed"""
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize_database()
    
    def _initialize_database(self):
        """Create database tables if they don't exist"""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            
            # Embedding jobs table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS embedding_jobs (
                    job_id TEXT PRIMARY KEY,
                    user_id TEXT NOT NULL,
                    media_id INTEGER NOT NULL,
                    status TEXT NOT NULL,
                    priority INTEGER DEFAULT 50,
                    user_tier TEXT DEFAULT 'free',
                    
                    -- Progress tracking
                    progress_percentage REAL DEFAULT 0.0,
                    chunks_processed INTEGER DEFAULT 0,
                    total_chunks INTEGER DEFAULT 0,
                    current_stage TEXT,
                    
                    -- Timing
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    started_at TIMESTAMP,
                    completed_at TIMESTAMP,
                    
                    -- Error handling
                    error_message TEXT,
                    retry_count INTEGER DEFAULT 0,
                    max_retries INTEGER DEFAULT 3,
                    
                    -- Metadata
                    model_name TEXT,
                    chunking_config TEXT,  -- JSON
                    metadata TEXT,  -- JSON
                    
                    -- Resource tracking
                    processing_time_ms INTEGER,
                    gpu_time_ms INTEGER,
                    
                    -- Indexes for common queries
                    INDEX idx_user_id (user_id),
                    INDEX idx_media_id (media_id),
                    INDEX idx_status (status),
                    INDEX idx_created_at (created_at),
                    INDEX idx_user_status (user_id, status)
                )
            """)
            
            # User quotas table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS user_quotas (
                    user_id TEXT PRIMARY KEY,
                    user_tier TEXT DEFAULT 'free',
                    
                    -- Daily limits
                    daily_chunks_limit INTEGER DEFAULT 1000,
                    daily_chunks_used INTEGER DEFAULT 0,
                    daily_reset_time TIMESTAMP,
                    
                    -- Concurrent job limits
                    concurrent_jobs_limit INTEGER DEFAULT 2,
                    concurrent_jobs_active INTEGER DEFAULT 0,
                    
                    -- Total usage tracking
                    total_chunks_processed INTEGER DEFAULT 0,
                    total_jobs_completed INTEGER DEFAULT 0,
                    total_jobs_failed INTEGER DEFAULT 0,
                    
                    -- Account standing
                    quota_exceeded_count INTEGER DEFAULT 0,
                    last_quota_exceeded TIMESTAMP,
                    is_suspended BOOLEAN DEFAULT FALSE,
                    suspension_reason TEXT,
                    
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # Job history table (for analytics and debugging)
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS job_history (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    job_id TEXT NOT NULL,
                    user_id TEXT NOT NULL,
                    media_id INTEGER NOT NULL,
                    event_type TEXT NOT NULL,  -- created, started, progress, completed, failed, cancelled
                    event_data TEXT,  -- JSON
                    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    
                    INDEX idx_job_id (job_id),
                    INDEX idx_timestamp (timestamp)
                )
            """)
            
            # Queue metrics table (for monitoring)
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS queue_metrics (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    queue_name TEXT NOT NULL,
                    queue_depth INTEGER,
                    processing_rate REAL,
                    error_rate REAL,
                    avg_processing_time_ms INTEGER,
                    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    
                    INDEX idx_queue_timestamp (queue_name, timestamp)
                )
            """)
            
            conn.commit()
            logger.info("Embeddings jobs database initialized")
    
    @contextmanager
    def _get_connection(self):
        """Get a database connection with proper error handling"""
        conn = sqlite3.connect(str(self.db_path), timeout=30.0)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
        finally:
            conn.close()
    
    # Job Management Methods
    
    def create_job(self, job_data: Dict[str, Any]) -> bool:
        """Create a new embedding job"""
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                
                # Insert job
                cursor.execute("""
                    INSERT INTO embedding_jobs (
                        job_id, user_id, media_id, status, priority, user_tier,
                        model_name, chunking_config, metadata
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    job_data['job_id'],
                    job_data['user_id'],
                    job_data['media_id'],
                    job_data.get('status', 'pending'),
                    job_data.get('priority', 50),
                    job_data.get('user_tier', 'free'),
                    job_data.get('model_name'),
                    job_data.get('chunking_config'),
                    job_data.get('metadata')
                ))
                
                # Record in history
                cursor.execute("""
                    INSERT INTO job_history (job_id, user_id, media_id, event_type, event_data)
                    VALUES (?, ?, ?, 'created', ?)
                """, (
                    job_data['job_id'],
                    job_data['user_id'],
                    job_data['media_id'],
                    job_data.get('metadata')
                ))
                
                # Update user's concurrent job count
                cursor.execute("""
                    UPDATE user_quotas 
                    SET concurrent_jobs_active = concurrent_jobs_active + 1,
                        updated_at = CURRENT_TIMESTAMP
                    WHERE user_id = ?
                """, (job_data['user_id'],))
                
                conn.commit()
                return True
                
        except Exception as e:
            logger.error(f"Failed to create job: {e}")
            return False
    
    def update_job_status(self, job_id: str, status: str, **kwargs) -> bool:
        """Update job status and related fields"""
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                
                # Build update query
                updates = ["status = ?", "updated_at = CURRENT_TIMESTAMP"]
                params = [status]
                
                if 'progress_percentage' in kwargs:
                    updates.append("progress_percentage = ?")
                    params.append(kwargs['progress_percentage'])
                
                if 'chunks_processed' in kwargs:
                    updates.append("chunks_processed = ?")
                    params.append(kwargs['chunks_processed'])
                
                if 'total_chunks' in kwargs:
                    updates.append("total_chunks = ?")
                    params.append(kwargs['total_chunks'])
                
                if 'current_stage' in kwargs:
                    updates.append("current_stage = ?")
                    params.append(kwargs['current_stage'])
                
                if 'error_message' in kwargs:
                    updates.append("error_message = ?")
                    params.append(kwargs['error_message'])
                
                if status == 'chunking' and 'started_at' not in kwargs:
                    updates.append("started_at = CURRENT_TIMESTAMP")
                
                if status in ['completed', 'failed', 'cancelled']:
                    updates.append("completed_at = CURRENT_TIMESTAMP")
                
                params.append(job_id)
                
                query = f"UPDATE embedding_jobs SET {', '.join(updates)} WHERE job_id = ?"
                cursor.execute(query, params)
                
                # Update concurrent jobs if completed
                if status in ['completed', 'failed', 'cancelled']:
                    cursor.execute("""
                        UPDATE user_quotas 
                        SET concurrent_jobs_active = MAX(0, concurrent_jobs_active - 1),
                            updated_at = CURRENT_TIMESTAMP
                        WHERE user_id = (SELECT user_id FROM embedding_jobs WHERE job_id = ?)
                    """, (job_id,))
                
                conn.commit()
                return True
                
        except Exception as e:
            logger.error(f"Failed to update job status: {e}")
            return False
    
    def get_job(self, job_id: str) -> Optional[Dict[str, Any]]:
        """Get job information by ID"""
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT * FROM embedding_jobs WHERE job_id = ?", (job_id,))
                row = cursor.fetchone()
                return dict(row) if row else None
        except Exception as e:
            logger.error(f"Failed to get job: {e}")
            return None
    
    def list_user_jobs(self, user_id: str, status: Optional[str] = None, 
                       limit: int = 20, offset: int = 0) -> List[Dict[str, Any]]:
        """List jobs for a user with optional filtering"""
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                
                query = "SELECT * FROM embedding_jobs WHERE user_id = ?"
                params = [user_id]
                
                if status:
                    query += " AND status = ?"
                    params.append(status)
                
                query += " ORDER BY created_at DESC LIMIT ? OFFSET ?"
                params.extend([limit, offset])
                
                cursor.execute(query, params)
                return [dict(row) for row in cursor.fetchall()]
                
        except Exception as e:
            logger.error(f"Failed to list user jobs: {e}")
            return []
    
    # Quota Management Methods
    
    def get_or_create_user_quota(self, user_id: str, user_tier: str = 'free') -> Dict[str, Any]:
        """Get or create user quota record"""
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                
                # Try to get existing quota
                cursor.execute("SELECT * FROM user_quotas WHERE user_id = ?", (user_id,))
                row = cursor.fetchone()
                
                if row:
                    return dict(row)
                
                # Create new quota record
                tier_limits = {
                    'free': {'daily_chunks': 1000, 'concurrent_jobs': 2},
                    'premium': {'daily_chunks': 10000, 'concurrent_jobs': 5},
                    'enterprise': {'daily_chunks': 100000, 'concurrent_jobs': 20}
                }
                
                limits = tier_limits.get(user_tier, tier_limits['free'])
                
                cursor.execute("""
                    INSERT INTO user_quotas (
                        user_id, user_tier, daily_chunks_limit, concurrent_jobs_limit,
                        daily_reset_time
                    ) VALUES (?, ?, ?, ?, ?)
                """, (
                    user_id,
                    user_tier,
                    limits['daily_chunks'],
                    limits['concurrent_jobs'],
                    (datetime.utcnow() + timedelta(days=1)).replace(hour=0, minute=0, second=0)
                ))
                
                conn.commit()
                
                cursor.execute("SELECT * FROM user_quotas WHERE user_id = ?", (user_id,))
                return dict(cursor.fetchone())
                
        except Exception as e:
            logger.error(f"Failed to get/create user quota: {e}")
            return {}
    
    def check_and_update_quota(self, user_id: str, chunks_requested: int) -> bool:
        """Check if user has quota available and update if so"""
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                
                # Get current quota
                quota = self.get_or_create_user_quota(user_id)
                
                # Check if daily reset is needed
                if datetime.fromisoformat(quota['daily_reset_time']) < datetime.utcnow():
                    cursor.execute("""
                        UPDATE user_quotas 
                        SET daily_chunks_used = 0,
                            daily_reset_time = ?,
                            updated_at = CURRENT_TIMESTAMP
                        WHERE user_id = ?
                    """, (
                        (datetime.utcnow() + timedelta(days=1)).replace(hour=0, minute=0, second=0),
                        user_id
                    ))
                    quota['daily_chunks_used'] = 0
                
                # Check if quota available
                if quota['daily_chunks_used'] + chunks_requested > quota['daily_chunks_limit']:
                    return False
                
                # Update quota
                cursor.execute("""
                    UPDATE user_quotas 
                    SET daily_chunks_used = daily_chunks_used + ?,
                        total_chunks_processed = total_chunks_processed + ?,
                        updated_at = CURRENT_TIMESTAMP
                    WHERE user_id = ?
                """, (chunks_requested, chunks_requested, user_id))
                
                conn.commit()
                return True
                
        except Exception as e:
            logger.error(f"Failed to check/update quota: {e}")
            return False
    
    def record_queue_metrics(self, metrics: Dict[str, Any]) -> bool:
        """Record queue metrics for monitoring"""
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                
                cursor.execute("""
                    INSERT INTO queue_metrics (
                        queue_name, queue_depth, processing_rate, 
                        error_rate, avg_processing_time_ms
                    ) VALUES (?, ?, ?, ?, ?)
                """, (
                    metrics['queue_name'],
                    metrics['queue_depth'],
                    metrics.get('processing_rate', 0),
                    metrics.get('error_rate', 0),
                    metrics.get('avg_processing_time_ms', 0)
                ))
                
                conn.commit()
                return True
                
        except Exception as e:
            logger.error(f"Failed to record queue metrics: {e}")
            return False
    
    def cleanup_old_jobs(self, days_to_keep: int = 30) -> int:
        """Clean up old completed jobs"""
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                
                cutoff_date = datetime.utcnow() - timedelta(days=days_to_keep)
                
                cursor.execute("""
                    DELETE FROM embedding_jobs 
                    WHERE status IN ('completed', 'failed', 'cancelled')
                    AND completed_at < ?
                """, (cutoff_date,))
                
                deleted_count = cursor.rowcount
                
                # Also clean up old history
                cursor.execute("""
                    DELETE FROM job_history 
                    WHERE timestamp < ?
                """, (cutoff_date,))
                
                conn.commit()
                logger.info(f"Cleaned up {deleted_count} old jobs")
                return deleted_count
                
        except Exception as e:
            logger.error(f"Failed to cleanup old jobs: {e}")
            return 0