"""
Custom exceptions for the scheduler system.
"""


class SchedulerError(Exception):
    """Base exception for all scheduler errors"""
    pass


class QueueError(SchedulerError):
    """Base exception for queue-related errors"""
    pass


class TaskError(SchedulerError):
    """Base exception for task-related errors"""
    pass


class BackendError(SchedulerError):
    """Base exception for backend-related errors"""
    pass


# Queue-specific exceptions

class QueueFullError(QueueError):
    """Raised when queue has reached maximum size"""
    pass


class QueueEmptyError(QueueError):
    """Raised when attempting to dequeue from empty queue"""
    pass


class DuplicateTaskError(QueueError):
    """Raised when task with same idempotency key already exists"""
    pass


# Task-specific exceptions

class TaskNotFoundError(TaskError):
    """Raised when task cannot be found"""
    pass


class TaskExpiredError(TaskError):
    """Raised when task has expired"""
    pass


class TaskDependencyError(TaskError):
    """Raised when task dependencies are invalid or create cycles"""
    pass


class TaskTimeoutError(TaskError):
    """Raised when task execution exceeds timeout"""
    pass


class TaskCancelledError(TaskError):
    """Raised when task has been cancelled"""
    pass


class HandlerNotFoundError(TaskError):
    """Raised when task handler is not registered"""
    pass


class HandlerExecutionError(TaskError):
    """Raised when task handler execution fails"""
    pass


# Backend-specific exceptions

class ConnectionError(BackendError):
    """Raised when backend connection fails"""
    pass


class TransactionError(BackendError):
    """Raised when database transaction fails"""
    pass


class SchemaError(BackendError):
    """Raised when schema operations fail"""
    pass


class MigrationError(BackendError):
    """Raised when schema migration fails"""
    pass


# Lease-specific exceptions

class LeaseError(SchedulerError):
    """Base exception for lease-related errors"""
    pass


class LeaseExpiredError(LeaseError):
    """Raised when lease has expired"""
    pass


class LeaseConflictError(LeaseError):
    """Raised when lease cannot be acquired due to conflict"""
    pass


# Worker-specific exceptions

class WorkerError(SchedulerError):
    """Base exception for worker-related errors"""
    pass


class WorkerShutdownError(WorkerError):
    """Raised when worker is shutting down"""
    pass


class WorkerOverloadError(WorkerError):
    """Raised when worker is overloaded"""
    pass


# Configuration exceptions

class ConfigurationError(SchedulerError):
    """Raised when configuration is invalid"""
    pass


# Buffer exceptions

class BufferError(SchedulerError):
    """Base exception for buffer-related errors"""
    pass


class BufferClosedError(BufferError):
    """Raised when attempting to use closed buffer"""
    pass


class BufferFlushError(BufferError):
    """Raised when buffer flush fails"""
    pass