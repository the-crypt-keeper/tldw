"""
Task model for the scheduler system.
Defines the complete lifecycle and metadata for tasks.
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional
import uuid


class TaskStatus(Enum):
    """Task lifecycle states"""
    PENDING = "pending"          # Created but not queued
    QUEUED = "queued"           # In queue waiting for worker
    RUNNING = "running"         # Being processed by worker
    COMPLETED = "completed"     # Successfully completed
    FAILED = "failed"          # Failed after all retries
    CANCELLED = "cancelled"    # Cancelled by user
    DEAD = "dead"             # Moved to dead letter queue


class TaskPriority(Enum):
    """Task priority levels (lower value = higher priority)"""
    CRITICAL = 0    # Highest priority
    HIGH = 1
    NORMAL = 2
    LOW = 3
    BACKGROUND = 4  # Lowest priority


@dataclass
class Task:
    """
    Complete task definition with metadata and lifecycle tracking.
    
    This class represents a unit of work in the scheduler system.
    Tasks can have dependencies, priorities, and retry policies.
    """
    
    # Identifiers
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    queue_name: str = "default"
    idempotency_key: Optional[str] = None
    
    # Handler
    handler: str = ""  # Registry key for handler function
    payload: Dict[str, Any] = field(default_factory=dict)
    
    # Scheduling
    priority: int = TaskPriority.NORMAL.value
    scheduled_at: Optional[datetime] = None
    expires_at: Optional[datetime] = None
    
    # Execution control
    max_retries: int = 3
    retry_count: int = 0
    retry_delay: int = 60  # seconds, with exponential backoff
    timeout: int = 300  # seconds
    
    # Dependencies
    depends_on: List[str] = field(default_factory=list)
    
    # State
    status: TaskStatus = TaskStatus.PENDING
    result: Optional[Any] = None
    error: Optional[str] = None
    
    # Metadata
    created_at: datetime = field(default_factory=datetime.utcnow)
    queued_at: Optional[datetime] = None
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    
    # Tracking
    worker_id: Optional[str] = None
    lease_id: Optional[str] = None
    execution_time: Optional[float] = None
    
    # Payload reference (for large payloads)
    payload_ref: Optional[str] = None
    result_ref: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert task to dictionary for serialization"""
        return {
            'id': self.id,
            'queue_name': self.queue_name,
            'idempotency_key': self.idempotency_key,
            'handler': self.handler,
            'payload': self.payload,
            'priority': self.priority,
            'scheduled_at': self.scheduled_at.isoformat() if self.scheduled_at else None,
            'expires_at': self.expires_at.isoformat() if self.expires_at else None,
            'max_retries': self.max_retries,
            'retry_count': self.retry_count,
            'retry_delay': self.retry_delay,
            'timeout': self.timeout,
            'depends_on': self.depends_on,
            'status': self.status.value,
            'result': self.result,
            'error': self.error,
            'created_at': self.created_at.isoformat(),
            'queued_at': self.queued_at.isoformat() if self.queued_at else None,
            'started_at': self.started_at.isoformat() if self.started_at else None,
            'completed_at': self.completed_at.isoformat() if self.completed_at else None,
            'worker_id': self.worker_id,
            'lease_id': self.lease_id,
            'execution_time': self.execution_time,
            'payload_ref': self.payload_ref,
            'result_ref': self.result_ref
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Task':
        """Create task from dictionary"""
        task = cls()
        
        # Simple fields
        for field_name in ['id', 'queue_name', 'idempotency_key', 'handler',
                          'payload', 'priority', 'max_retries', 'retry_count',
                          'retry_delay', 'timeout', 'depends_on', 'result',
                          'error', 'worker_id', 'lease_id', 'execution_time',
                          'payload_ref', 'result_ref']:
            if field_name in data:
                setattr(task, field_name, data[field_name])
        
        # Status enum
        if 'status' in data:
            task.status = TaskStatus(data['status'])
        
        # Datetime fields
        datetime_fields = ['scheduled_at', 'expires_at', 'created_at',
                          'queued_at', 'started_at', 'completed_at']
        for field_name in datetime_fields:
            if field_name in data and data[field_name]:
                value = data[field_name]
                if isinstance(value, str):
                    setattr(task, field_name, datetime.fromisoformat(value))
                elif isinstance(value, datetime):
                    setattr(task, field_name, value)
        
        return task
    
    def is_ready(self) -> bool:
        """Check if task is ready to run (no pending dependencies)"""
        return not self.depends_on or len(self.depends_on) == 0
    
    def is_expired(self) -> bool:
        """Check if task has expired"""
        if not self.expires_at:
            return False
        return datetime.utcnow() > self.expires_at
    
    def is_scheduled(self) -> bool:
        """Check if task is scheduled for future execution"""
        if not self.scheduled_at:
            return False
        return datetime.utcnow() < self.scheduled_at
    
    def should_retry(self) -> bool:
        """Check if task should be retried after failure"""
        return self.retry_count < self.max_retries
    
    def calculate_retry_delay(self) -> int:
        """Calculate delay before next retry (exponential backoff)"""
        base_delay = self.retry_delay
        return base_delay * (2 ** self.retry_count)