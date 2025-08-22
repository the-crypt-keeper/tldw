"""
Base abstractions for the scheduler system.
"""

from .task import Task, TaskStatus, TaskPriority
from .queue_backend import QueueBackend
from .registry import TaskRegistry, get_registry, task
from .exceptions import (
    SchedulerError,
    QueueError,
    TaskError,
    BackendError,
    QueueFullError,
    QueueEmptyError,
    DuplicateTaskError,
    TaskNotFoundError,
    TaskExpiredError,
    TaskDependencyError,
    TaskTimeoutError,
    TaskCancelledError,
    HandlerNotFoundError,
    HandlerExecutionError,
    ConnectionError,
    TransactionError,
    SchemaError,
    MigrationError,
    LeaseError,
    LeaseExpiredError,
    LeaseConflictError,
    WorkerError,
    WorkerShutdownError,
    WorkerOverloadError,
    ConfigurationError,
    BufferError,
    BufferClosedError,
    BufferFlushError
)

__all__ = [
    # Task
    'Task',
    'TaskStatus',
    'TaskPriority',
    
    # Backend
    'QueueBackend',
    
    # Registry
    'TaskRegistry',
    'get_registry',
    'task',
    
    # Exceptions
    'SchedulerError',
    'QueueError',
    'TaskError',
    'BackendError',
    'QueueFullError',
    'QueueEmptyError',
    'DuplicateTaskError',
    'TaskNotFoundError',
    'TaskExpiredError',
    'TaskDependencyError',
    'TaskTimeoutError',
    'TaskCancelledError',
    'HandlerNotFoundError',
    'HandlerExecutionError',
    'ConnectionError',
    'TransactionError',
    'SchemaError',
    'MigrationError',
    'LeaseError',
    'LeaseExpiredError',
    'LeaseConflictError',
    'WorkerError',
    'WorkerShutdownError',
    'WorkerOverloadError',
    'ConfigurationError',
    'BufferError',
    'BufferClosedError',
    'BufferFlushError'
]