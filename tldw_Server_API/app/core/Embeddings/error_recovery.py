# error_recovery.py
# Error recovery and dead letter queue for failed embedding jobs

import asyncio
import json
import time
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, List, Callable
from enum import Enum
from dataclasses import dataclass, asdict
import pickle
from pathlib import Path

from loguru import logger
from tldw_Server_API.app.core.Embeddings.audit_logger import audit_log, AuditEventType


class FailureReason(Enum):
    """Types of failures that can occur"""
    PROVIDER_ERROR = "provider_error"
    RATE_LIMIT = "rate_limit"
    INVALID_INPUT = "invalid_input"
    TIMEOUT = "timeout"
    NETWORK_ERROR = "network_error"
    MODEL_NOT_FOUND = "model_not_found"
    QUOTA_EXCEEDED = "quota_exceeded"
    UNKNOWN = "unknown"


@dataclass
class FailedJob:
    """Represents a failed embedding job"""
    job_id: str
    user_id: Optional[str]
    input_text: str
    model: str
    provider: str
    failure_reason: FailureReason
    error_message: str
    attempt_count: int
    first_failed_at: datetime
    last_failed_at: datetime
    metadata: Dict[str, Any]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        result = asdict(self)
        result['failure_reason'] = self.failure_reason.value
        result['first_failed_at'] = self.first_failed_at.isoformat()
        result['last_failed_at'] = self.last_failed_at.isoformat()
        return result
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'FailedJob':
        """Create from dictionary"""
        data['failure_reason'] = FailureReason(data['failure_reason'])
        data['first_failed_at'] = datetime.fromisoformat(data['first_failed_at'])
        data['last_failed_at'] = datetime.fromisoformat(data['last_failed_at'])
        return cls(**data)


class DeadLetterQueue:
    """
    Dead Letter Queue for failed embedding jobs.
    Stores failed jobs for manual review or retry.
    """
    
    def __init__(
        self,
        storage_path: Optional[str] = None,
        max_size: int = 10000,
        retention_days: int = 30
    ):
        """
        Initialize the Dead Letter Queue.
        
        Args:
            storage_path: Path to store failed jobs
            max_size: Maximum number of jobs to store
            retention_days: Days to retain failed jobs
        """
        self.storage_path = Path(storage_path or "./dlq/embeddings")
        self.storage_path.mkdir(parents=True, exist_ok=True)
        
        self.max_size = max_size
        self.retention_days = retention_days
        self.failed_jobs: List[FailedJob] = []
        
        # Load existing failed jobs
        self._load_from_disk()
        
        # Statistics
        self.stats = {
            'total_failures': 0,
            'recovered': 0,
            'expired': 0,
            'by_reason': {}
        }
        
        logger.info(f"Dead Letter Queue initialized at {self.storage_path}")
    
    def add_failed_job(
        self,
        job_id: str,
        input_text: str,
        model: str,
        provider: str,
        error: Exception,
        user_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Add a failed job to the DLQ.
        
        Args:
            job_id: Unique job identifier
            input_text: Original input that failed
            model: Model that was requested
            provider: Provider that failed
            error: The exception that occurred
            user_id: Optional user identifier
            metadata: Additional metadata
            
        Returns:
            DLQ entry ID
        """
        # Determine failure reason
        failure_reason = self._classify_error(error)
        
        # Check if job already exists
        existing_job = self._find_job(job_id)
        
        if existing_job:
            # Update existing job
            existing_job.attempt_count += 1
            existing_job.last_failed_at = datetime.utcnow()
            existing_job.error_message = str(error)
        else:
            # Create new failed job
            failed_job = FailedJob(
                job_id=job_id,
                user_id=user_id,
                input_text=input_text[:1000],  # Truncate very long texts
                model=model,
                provider=provider,
                failure_reason=failure_reason,
                error_message=str(error),
                attempt_count=1,
                first_failed_at=datetime.utcnow(),
                last_failed_at=datetime.utcnow(),
                metadata=metadata or {}
            )
            
            self.failed_jobs.append(failed_job)
            
            # Enforce max size
            if len(self.failed_jobs) > self.max_size:
                self.failed_jobs.pop(0)  # Remove oldest
        
        # Update statistics
        self.stats['total_failures'] += 1
        self.stats['by_reason'][failure_reason.value] = \
            self.stats['by_reason'].get(failure_reason.value, 0) + 1
        
        # Persist to disk
        self._save_to_disk()
        
        # Log the failure
        logger.error(
            f"Job {job_id} added to DLQ: "
            f"reason={failure_reason.value}, "
            f"provider={provider}, "
            f"model={model}"
        )
        
        return job_id
    
    def _classify_error(self, error: Exception) -> FailureReason:
        """Classify the error type based on exception"""
        error_str = str(error).lower()
        error_type = type(error).__name__.lower()
        
        if 'rate' in error_str or 'limit' in error_str or '429' in error_str:
            return FailureReason.RATE_LIMIT
        elif 'timeout' in error_type or 'timeout' in error_str:
            return FailureReason.TIMEOUT
        elif 'network' in error_str or 'connection' in error_str:
            return FailureReason.NETWORK_ERROR
        elif 'model' in error_str and 'not found' in error_str:
            return FailureReason.MODEL_NOT_FOUND
        elif 'quota' in error_str or 'exceeded' in error_str:
            return FailureReason.QUOTA_EXCEEDED
        elif 'invalid' in error_str or 'validation' in error_str:
            return FailureReason.INVALID_INPUT
        elif any(x in error_str for x in ['500', '502', '503', 'server']):
            return FailureReason.PROVIDER_ERROR
        else:
            return FailureReason.UNKNOWN
    
    def _find_job(self, job_id: str) -> Optional[FailedJob]:
        """Find a job by ID"""
        for job in self.failed_jobs:
            if job.job_id == job_id:
                return job
        return None
    
    def get_retryable_jobs(
        self,
        max_attempts: int = 3,
        older_than_minutes: int = 5
    ) -> List[FailedJob]:
        """
        Get jobs that can be retried.
        
        Args:
            max_attempts: Maximum attempts before giving up
            older_than_minutes: Wait time before retry
            
        Returns:
            List of jobs that can be retried
        """
        retryable = []
        cutoff_time = datetime.utcnow() - timedelta(minutes=older_than_minutes)
        
        for job in self.failed_jobs:
            # Skip if too many attempts
            if job.attempt_count >= max_attempts:
                continue
            
            # Skip if too recent
            if job.last_failed_at > cutoff_time:
                continue
            
            # Skip non-retryable failures
            if job.failure_reason in [
                FailureReason.INVALID_INPUT,
                FailureReason.MODEL_NOT_FOUND
            ]:
                continue
            
            retryable.append(job)
        
        return retryable
    
    def remove_job(self, job_id: str) -> bool:
        """
        Remove a job from the DLQ (e.g., after successful retry).
        
        Args:
            job_id: Job identifier
            
        Returns:
            True if removed, False if not found
        """
        for i, job in enumerate(self.failed_jobs):
            if job.job_id == job_id:
                self.failed_jobs.pop(i)
                self.stats['recovered'] += 1
                self._save_to_disk()
                logger.info(f"Job {job_id} removed from DLQ (recovered)")
                return True
        return False
    
    def cleanup_expired(self) -> int:
        """
        Remove jobs older than retention period.
        
        Returns:
            Number of jobs removed
        """
        cutoff_date = datetime.utcnow() - timedelta(days=self.retention_days)
        
        original_count = len(self.failed_jobs)
        self.failed_jobs = [
            job for job in self.failed_jobs
            if job.first_failed_at > cutoff_date
        ]
        
        removed = original_count - len(self.failed_jobs)
        
        if removed > 0:
            self.stats['expired'] += removed
            self._save_to_disk()
            logger.info(f"Cleaned up {removed} expired jobs from DLQ")
        
        return removed
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get DLQ statistics"""
        by_provider = {}
        by_model = {}
        
        for job in self.failed_jobs:
            by_provider[job.provider] = by_provider.get(job.provider, 0) + 1
            by_model[job.model] = by_model.get(job.model, 0) + 1
        
        return {
            'current_size': len(self.failed_jobs),
            'max_size': self.max_size,
            'total_failures': self.stats['total_failures'],
            'recovered': self.stats['recovered'],
            'expired': self.stats['expired'],
            'by_reason': self.stats['by_reason'],
            'by_provider': by_provider,
            'by_model': by_model,
            'oldest_job': (
                min(self.failed_jobs, key=lambda x: x.first_failed_at).first_failed_at.isoformat()
                if self.failed_jobs else None
            )
        }
    
    def _save_to_disk(self):
        """Persist failed jobs to disk"""
        try:
            file_path = self.storage_path / "failed_jobs.json"
            data = {
                'jobs': [job.to_dict() for job in self.failed_jobs],
                'stats': self.stats,
                'saved_at': datetime.utcnow().isoformat()
            }
            
            with open(file_path, 'w') as f:
                json.dump(data, f, indent=2)
                
        except Exception as e:
            logger.error(f"Failed to save DLQ to disk: {e}")
    
    def _load_from_disk(self):
        """Load failed jobs from disk"""
        try:
            file_path = self.storage_path / "failed_jobs.json"
            
            if file_path.exists():
                with open(file_path, 'r') as f:
                    data = json.load(f)
                
                self.failed_jobs = [
                    FailedJob.from_dict(job_data)
                    for job_data in data.get('jobs', [])
                ]
                
                self.stats = data.get('stats', self.stats)
                
                logger.info(f"Loaded {len(self.failed_jobs)} jobs from DLQ")
                
        except Exception as e:
            logger.error(f"Failed to load DLQ from disk: {e}")


class ErrorRecoveryManager:
    """
    Manages error recovery strategies for embedding operations.
    """
    
    def __init__(self, dlq: Optional[DeadLetterQueue] = None):
        """
        Initialize error recovery manager.
        
        Args:
            dlq: Dead letter queue instance
        """
        self.dlq = dlq or DeadLetterQueue()
        self.recovery_strategies: Dict[FailureReason, Callable] = {}
        self._setup_default_strategies()
    
    def _setup_default_strategies(self):
        """Setup default recovery strategies"""
        
        async def rate_limit_strategy(job: FailedJob) -> bool:
            """Wait and retry for rate limit errors"""
            wait_time = min(60 * (2 ** job.attempt_count), 300)  # Exponential backoff, max 5 min
            logger.info(f"Rate limit recovery: waiting {wait_time}s before retry")
            await asyncio.sleep(wait_time)
            return True  # Can retry
        
        async def network_error_strategy(job: FailedJob) -> bool:
            """Brief wait for network errors"""
            await asyncio.sleep(5)
            return True  # Can retry
        
        async def provider_error_strategy(job: FailedJob) -> bool:
            """Log and don't retry provider errors immediately"""
            logger.error(f"Provider error for job {job.job_id}: {job.error_message}")
            return False  # Don't retry automatically
        
        self.recovery_strategies = {
            FailureReason.RATE_LIMIT: rate_limit_strategy,
            FailureReason.NETWORK_ERROR: network_error_strategy,
            FailureReason.TIMEOUT: network_error_strategy,
            FailureReason.PROVIDER_ERROR: provider_error_strategy,
        }
    
    async def handle_failure(
        self,
        job_id: str,
        error: Exception,
        context: Dict[str, Any]
    ) -> bool:
        """
        Handle a failure with appropriate recovery strategy.
        
        Args:
            job_id: Job identifier
            error: The exception that occurred
            context: Context information (input, model, provider, etc.)
            
        Returns:
            True if should retry, False otherwise
        """
        # Add to DLQ
        self.dlq.add_failed_job(
            job_id=job_id,
            input_text=context.get('input', ''),
            model=context.get('model', 'unknown'),
            provider=context.get('provider', 'unknown'),
            error=error,
            user_id=context.get('user_id'),
            metadata=context.get('metadata', {})
        )
        
        # Get the failed job
        job = self.dlq._find_job(job_id)
        if not job:
            return False
        
        # Apply recovery strategy
        strategy = self.recovery_strategies.get(job.failure_reason)
        if strategy:
            return await strategy(job)
        
        return False  # No strategy, don't retry
    
    async def process_dlq_retries(self) -> int:
        """
        Process retryable jobs from DLQ.
        
        Returns:
            Number of jobs processed
        """
        retryable = self.dlq.get_retryable_jobs()
        processed = 0
        
        for job in retryable:
            try:
                # Apply recovery strategy
                strategy = self.recovery_strategies.get(job.failure_reason)
                if strategy:
                    should_retry = await strategy(job)
                    if should_retry:
                        # Mark for retry (actual retry would be done by caller)
                        logger.info(f"Job {job.job_id} marked for retry")
                        processed += 1
            except Exception as e:
                logger.error(f"Error processing DLQ job {job.job_id}: {e}")
        
        return processed


# Global error recovery manager
_recovery_manager: Optional[ErrorRecoveryManager] = None


def get_recovery_manager() -> ErrorRecoveryManager:
    """Get or create the global error recovery manager."""
    global _recovery_manager
    if _recovery_manager is None:
        _recovery_manager = ErrorRecoveryManager()
    return _recovery_manager