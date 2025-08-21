# job_manager.py
# Job queue management for Prompt Studio

import json
import sqlite3
import uuid
import asyncio
from typing import List, Dict, Any, Optional, Callable
from datetime import datetime, timedelta
from enum import Enum
from loguru import logger

from tldw_Server_API.app.core.DB_Management.PromptStudioDatabase import (
    PromptStudioDatabase, DatabaseError
)

########################################################################################################################
# Job Types and Status

class JobType(str, Enum):
    EVALUATION = "evaluation"
    OPTIMIZATION = "optimization"
    GENERATION = "generation"

class JobStatus(str, Enum):
    QUEUED = "queued"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"

########################################################################################################################
# Job Manager

class JobManager:
    """Manages job queue for Prompt Studio background operations."""
    
    def __init__(self, db: PromptStudioDatabase):
        """
        Initialize JobManager.
        
        Args:
            db: PromptStudioDatabase instance
        """
        self.db = db
        self.client_id = db.client_id
        self._job_handlers: Dict[JobType, Callable] = {}
        self._is_processing = False
        self._processing_task = None
    
    ####################################################################################################################
    # Job Creation and Management
    
    def create_job(self, job_type: JobType, entity_id: int, payload: Dict[str, Any],
                  project_id: Optional[int] = None, priority: int = 5, max_retries: int = 3) -> Dict[str, Any]:
        """
        Create a new job in the queue.
        
        Args:
            job_type: Type of job
            entity_id: ID of related entity (evaluation, optimization, etc.)
            payload: Job-specific data
            project_id: Optional project ID
            priority: Job priority (1-10, higher = more priority)
            max_retries: Maximum retry attempts
            
        Returns:
            Created job record
        """
        try:
            conn = self.db.get_connection()
            cursor = conn.cursor()
            
            job_uuid = str(uuid.uuid4())
            
            cursor.execute("""
                INSERT INTO prompt_studio_job_queue (
                    uuid, job_type, entity_id, project_id, priority, status,
                    payload, max_retries, client_id
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                job_uuid, job_type, entity_id, project_id, priority, JobStatus.QUEUED.value,
                json.dumps(payload), max_retries, self.client_id
            ))
            
            job_id = cursor.lastrowid
            conn.commit()
            
            logger.info(f"Created {job_type} job {job_id} for entity {entity_id}")
            
            return self.get_job(job_id)
            
        except Exception as e:
            logger.error(f"Failed to create job: {e}")
            raise DatabaseError(f"Failed to create job: {e}")
    
    def get_job(self, job_id: int) -> Optional[Dict[str, Any]]:
        """
        Get a job by ID.
        
        Args:
            job_id: Job ID
            
        Returns:
            Job record or None
        """
        conn = self.db.get_connection()
        cursor = conn.cursor()
        
        cursor.execute("SELECT * FROM prompt_studio_job_queue WHERE id = ?", (job_id,))
        row = cursor.fetchone()
        
        if row:
            return self.db._row_to_dict(cursor, row)
        return None
    
    def get_job_by_uuid(self, job_uuid: str) -> Optional[Dict[str, Any]]:
        """
        Get a job by UUID.
        
        Args:
            job_uuid: Job UUID
            
        Returns:
            Job record or None
        """
        conn = self.db.get_connection()
        cursor = conn.cursor()
        
        cursor.execute("SELECT * FROM prompt_studio_job_queue WHERE uuid = ?", (job_uuid,))
        row = cursor.fetchone()
        
        if row:
            return self.db._row_to_dict(cursor, row)
        return None
    
    def list_jobs(self, status: Optional[JobStatus] = None, 
                 job_type: Optional[JobType] = None,
                 limit: int = 100) -> List[Dict[str, Any]]:
        """
        List jobs with optional filtering.
        
        Args:
            status: Filter by status
            job_type: Filter by job type
            limit: Maximum results
            
        Returns:
            List of jobs
        """
        conn = self.db.get_connection()
        cursor = conn.cursor()
        
        query = "SELECT * FROM prompt_studio_job_queue WHERE 1=1"
        params = []
        
        if status:
            query += " AND status = ?"
            params.append(status.value if isinstance(status, JobStatus) else status)
        
        if job_type:
            query += " AND job_type = ?"
            params.append(job_type.value if isinstance(job_type, JobType) else job_type)
        
        query += " ORDER BY priority DESC, created_at ASC LIMIT ?"
        params.append(limit)
        
        cursor.execute(query, params)
        
        return [self.db._row_to_dict(cursor, row) for row in cursor.fetchall()]
    
    def update_job_status(self, job_id: int, status: JobStatus, 
                         error_message: Optional[str] = None,
                         result: Optional[Dict[str, Any]] = None) -> bool:
        """
        Update job status.
        
        Args:
            job_id: Job ID
            status: New status
            error_message: Error message if failed
            result: Result data if completed
            
        Returns:
            True if updated
        """
        conn = self.db.get_connection()
        cursor = conn.cursor()
        
        updates = ["status = ?"]
        params = [status.value]
        
        if status == JobStatus.PROCESSING:
            updates.append("started_at = CURRENT_TIMESTAMP")
        elif status in [JobStatus.COMPLETED, JobStatus.FAILED, JobStatus.CANCELLED]:
            updates.append("completed_at = CURRENT_TIMESTAMP")
        
        if error_message:
            updates.append("error_message = ?")
            params.append(error_message)
        
        if result:
            updates.append("result = ?")
            params.append(json.dumps(result))
        
        params.append(job_id)
        
        cursor.execute(f"""
            UPDATE prompt_studio_job_queue
            SET {', '.join(updates)}
            WHERE id = ?
        """, params)
        
        success = cursor.rowcount > 0
        if success:
            conn.commit()
            logger.info(f"Updated job {job_id} status to {status.value}")
        
        return success
    
    def cancel_job(self, job_id: int, reason: Optional[str] = None) -> bool:
        """
        Cancel a job.
        
        Args:
            job_id: Job ID
            reason: Cancellation reason
            
        Returns:
            True if cancelled
        """
        job = self.get_job(job_id)
        if not job:
            return False
        
        if job["status"] in [JobStatus.COMPLETED.value, JobStatus.FAILED.value]:
            logger.warning(f"Cannot cancel job {job_id} with status {job['status']}")
            return False
        
        return self.update_job_status(
            job_id, 
            JobStatus.CANCELLED,
            error_message=reason or "Job cancelled by user"
        )
    
    ####################################################################################################################
    # Job Processing
    
    def get_next_job(self) -> Optional[Dict[str, Any]]:
        """
        Get the next job to process from the queue.
        
        Returns:
            Next job or None
        """
        conn = self.db.get_connection()
        cursor = conn.cursor()
        
        # Get highest priority queued job
        cursor.execute("""
            SELECT * FROM prompt_studio_job_queue
            WHERE status = ?
            ORDER BY priority DESC, created_at ASC
            LIMIT 1
        """, (JobStatus.QUEUED.value,))
        
        row = cursor.fetchone()
        if row:
            job = self.db._row_to_dict(cursor, row)
            
            # Mark as processing
            self.update_job_status(job["id"], JobStatus.PROCESSING)
            
            return job
        
        return None
    
    def retry_job(self, job_id: int) -> bool:
        """
        Retry a failed job.
        
        Args:
            job_id: Job ID
            
        Returns:
            True if retry scheduled
        """
        conn = self.db.get_connection()
        cursor = conn.cursor()
        
        # Get job
        job = self.get_job(job_id)
        if not job:
            return False
        
        # Check if can retry
        if job["retry_count"] >= job["max_retries"]:
            logger.warning(f"Job {job_id} has reached max retries")
            return False
        
        # Update retry count and reset status
        cursor.execute("""
            UPDATE prompt_studio_job_queue
            SET status = ?, retry_count = retry_count + 1,
                error_message = NULL, started_at = NULL, completed_at = NULL
            WHERE id = ?
        """, (JobStatus.QUEUED.value, job_id))
        
        success = cursor.rowcount > 0
        if success:
            conn.commit()
            logger.info(f"Scheduled retry for job {job_id} (attempt {job['retry_count'] + 1})")
        
        return success
    
    def register_handler(self, job_type: JobType, handler: Callable):
        """
        Register a handler function for a job type.
        
        Args:
            job_type: Type of job
            handler: Async function to handle the job
        """
        self._job_handlers[job_type] = handler
        logger.info(f"Registered handler for {job_type.value} jobs")
    
    async def process_job(self, job: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process a single job.
        
        Args:
            job: Job record
            
        Returns:
            Job result
        """
        job_type = JobType(job["job_type"])
        handler = self._job_handlers.get(job_type)
        
        if not handler:
            raise ValueError(f"No handler registered for job type {job_type.value}")
        
        try:
            logger.info(f"Processing {job_type.value} job {job['id']}")
            
            # Execute handler
            result = await handler(job["payload"], job["entity_id"])
            
            # Update job as completed
            self.update_job_status(job["id"], JobStatus.COMPLETED, result=result)
            
            logger.info(f"Completed {job_type.value} job {job['id']}")
            return result
            
        except Exception as e:
            logger.error(f"Job {job['id']} failed: {e}")
            
            # Check if should retry
            if job["retry_count"] < job["max_retries"]:
                self.retry_job(job["id"])
            else:
                self.update_job_status(
                    job["id"], 
                    JobStatus.FAILED,
                    error_message=str(e)
                )
            
            raise
    
    async def start_processing(self, max_concurrent: int = 3):
        """
        Start processing jobs from the queue.
        
        Args:
            max_concurrent: Maximum concurrent jobs
        """
        if self._is_processing:
            logger.warning("Job processing already running")
            return
        
        self._is_processing = True
        logger.info(f"Starting job processor with max {max_concurrent} concurrent jobs")
        
        try:
            while self._is_processing:
                # Get next jobs up to max_concurrent
                jobs = []
                for _ in range(max_concurrent):
                    job = self.get_next_job()
                    if job:
                        jobs.append(job)
                
                if not jobs:
                    # No jobs to process, wait
                    await asyncio.sleep(5)
                    continue
                
                # Process jobs concurrently
                tasks = [self.process_job(job) for job in jobs]
                results = await asyncio.gather(*tasks, return_exceptions=True)
                
                # Log results
                for job, result in zip(jobs, results):
                    if isinstance(result, Exception):
                        logger.error(f"Job {job['id']} failed with exception: {result}")
                    else:
                        logger.debug(f"Job {job['id']} completed successfully")
                
                # Small delay before next batch
                await asyncio.sleep(1)
                
        except Exception as e:
            logger.error(f"Job processor error: {e}")
        finally:
            self._is_processing = False
            logger.info("Job processor stopped")
    
    def stop_processing(self):
        """Stop processing jobs."""
        self._is_processing = False
        logger.info("Stopping job processor")
    
    ####################################################################################################################
    # Job Statistics
    
    def get_job_stats(self) -> Dict[str, Any]:
        """
        Get job queue statistics.
        
        Returns:
            Statistics dictionary
        """
        conn = self.db.get_connection()
        cursor = conn.cursor()
        
        stats = {}
        
        # Count by status
        cursor.execute("""
            SELECT status, COUNT(*) as count
            FROM prompt_studio_job_queue
            GROUP BY status
        """)
        
        stats["by_status"] = {row[0]: row[1] for row in cursor.fetchall()}
        
        # Count by type
        cursor.execute("""
            SELECT job_type, COUNT(*) as count
            FROM prompt_studio_job_queue
            GROUP BY job_type
        """)
        
        stats["by_type"] = {row[0]: row[1] for row in cursor.fetchall()}
        
        # Average processing time
        cursor.execute("""
            SELECT AVG(
                CAST((julianday(completed_at) - julianday(started_at)) * 24 * 60 * 60 AS INTEGER)
            )
            FROM prompt_studio_job_queue
            WHERE status = 'completed' 
                AND started_at IS NOT NULL 
                AND completed_at IS NOT NULL
        """)
        
        avg_time = cursor.fetchone()[0]
        stats["avg_processing_time_seconds"] = avg_time if avg_time else 0
        
        # Success rate
        cursor.execute("""
            SELECT 
                COUNT(CASE WHEN status = 'completed' THEN 1 END) * 100.0 / 
                COUNT(CASE WHEN status IN ('completed', 'failed') THEN 1 END)
            FROM prompt_studio_job_queue
            WHERE status IN ('completed', 'failed')
        """)
        
        success_rate = cursor.fetchone()[0]
        stats["success_rate"] = success_rate if success_rate else 0
        
        # Queue depth
        stats["queue_depth"] = stats["by_status"].get(JobStatus.QUEUED.value, 0)
        
        # Currently processing
        stats["processing"] = stats["by_status"].get(JobStatus.PROCESSING.value, 0)
        
        return stats
    
    def cleanup_old_jobs(self, days: int = 30) -> int:
        """
        Clean up old completed/failed jobs.
        
        Args:
            days: Delete jobs older than this many days
            
        Returns:
            Number of jobs deleted
        """
        conn = self.db.get_connection()
        cursor = conn.cursor()
        
        cutoff_date = datetime.utcnow() - timedelta(days=days)
        
        cursor.execute("""
            DELETE FROM prompt_studio_job_queue
            WHERE status IN ('completed', 'failed', 'cancelled')
                AND completed_at < ?
        """, (cutoff_date.isoformat(),))
        
        deleted = cursor.rowcount
        if deleted > 0:
            conn.commit()
            logger.info(f"Cleaned up {deleted} old jobs")
        
        return deleted