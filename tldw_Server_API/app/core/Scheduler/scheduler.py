"""
Main Scheduler class that orchestrates all components.
"""

import asyncio
from typing import Optional, List, Dict, Any, Callable
from datetime import datetime
from loguru import logger

from .base import Task, TaskStatus, TaskPriority
from .base.registry import TaskRegistry, get_registry
from .base.exceptions import SchedulerError
from .backends import create_backend, BackendManager
from .config import SchedulerConfig, get_config
from .core.write_buffer import SafeWriteBuffer
from .core.worker_pool import WorkerPool
from .core.leader_election import LeaderElection
from .services import LeaseService, DependencyService, PayloadService


class Scheduler:
    """
    Main scheduler that orchestrates task queuing and execution.
    
    This is the primary interface for the scheduler system, providing:
    - Task submission and management
    - Worker pool management
    - Leader election for distributed deployments
    - Monitoring and health checks
    """
    
    def __init__(self, config: Optional[SchedulerConfig] = None):
        """
        Initialize scheduler.
        
        Args:
            config: Scheduler configuration (uses global if not provided)
        """
        self.config = config or get_config()
        self.registry = get_registry()
        
        # Core components (initialized in start())
        self.backend_manager = BackendManager(self.config)
        self.backend = None
        self.write_buffer: Optional[SafeWriteBuffer] = None
        self.worker_pool: Optional[WorkerPool] = None
        self.leader_election: Optional[LeaderElection] = None
        
        # Services
        self.lease_service: Optional[LeaseService] = None
        self.dependency_service: Optional[DependencyService] = None
        self.payload_service: Optional[PayloadService] = None
        
        # Background tasks
        self._cleanup_task: Optional[asyncio.Task] = None
        self._monitor_task: Optional[asyncio.Task] = None
        
        # State
        self._started = False
        self._stopping = False
        
        logger.info(f"Scheduler initialized with backend: {self.config.database_url}")
    
    async def start(self, start_workers: bool = True) -> None:
        """
        Start the scheduler.
        
        Args:
            start_workers: Whether to start worker pool
        """
        if self._started:
            logger.warning("Scheduler already started")
            return
        
        logger.info("Starting scheduler...")
        
        try:
            # Connect to backend
            self.backend = await self.backend_manager.connect()
            
            # Initialize write buffer
            self.write_buffer = SafeWriteBuffer(self.backend, self.config)
            
            # Initialize services
            self.lease_service = LeaseService(self.backend, self.config.lease_duration_seconds)
            self.dependency_service = DependencyService(self.backend)
            self.payload_service = PayloadService(self.backend, self.config)
            
            # Start lease reaper
            await self.lease_service.start_reaper(self.config.lease_reaper_interval)
            
            # Initialize leader election
            self.leader_election = LeaderElection(self.backend, self.config)
            
            # Start worker pool if requested
            if start_workers:
                self.worker_pool = WorkerPool(self.backend, self.registry, self.config)
                await self.worker_pool.start()
            
            # Start background tasks
            self._cleanup_task = asyncio.create_task(self._cleanup_loop())
            self._monitor_task = asyncio.create_task(self._monitor_loop())
            
            # Try to become leader for cleanup tasks
            await self.leader_election.maintain_leadership(
                "scheduler:cleanup",
                callback=self._on_become_cleanup_leader
            )
            
            self._started = True
            logger.info("Scheduler started successfully")
            
        except Exception as e:
            logger.error(f"Failed to start scheduler: {e}")
            await self.stop()
            raise SchedulerError(f"Scheduler start failed: {e}")
    
    async def stop(self) -> None:
        """Stop the scheduler gracefully."""
        if self._stopping:
            return
        
        self._stopping = True
        logger.info("Stopping scheduler...")
        
        # Stop background tasks
        for task in [self._cleanup_task, self._monitor_task]:
            if task:
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass
        
        # Stop worker pool
        if self.worker_pool:
            await self.worker_pool.stop()
        
        # Stop leader election
        if self.leader_election:
            await self.leader_election.stop_all()
        
        # Stop lease reaper
        if self.lease_service:
            await self.lease_service.stop_reaper()
        
        # Flush write buffer
        if self.write_buffer:
            await self.write_buffer.close()
        
        # Disconnect backend
        await self.backend_manager.disconnect()
        
        self._started = False
        self._stopping = False
        logger.info("Scheduler stopped")
    
    async def submit(self, 
                     handler: str,
                     payload: Optional[Any] = None,
                     priority: int = TaskPriority.NORMAL.value,
                     queue_name: Optional[str] = None,
                     depends_on: Optional[List[str]] = None,
                     idempotency_key: Optional[str] = None,
                     metadata: Optional[Dict[str, Any]] = None) -> str:
        """
        Submit a task to the scheduler.
        
        Args:
            handler: Task handler name (must be registered)
            payload: Task payload
            priority: Task priority
            queue_name: Target queue
            depends_on: Task dependencies
            idempotency_key: Idempotency key for deduplication
            metadata: Additional metadata
            
        Returns:
            Task ID
        """
        if not self._started:
            raise SchedulerError("Scheduler not started")
        
        # Validate handler
        if handler not in self.registry:
            raise ValueError(f"Handler '{handler}' not registered")
        
        # Prepare payload
        if self.payload_service:
            payload = self.payload_service.prepare_payload(payload)
        
        # Create task
        task = Task(
            handler=handler,
            payload=payload,
            priority=priority,
            queue_name=queue_name or self.config.default_queue_name,
            depends_on=depends_on,
            idempotency_key=idempotency_key,
            metadata=metadata
        )
        
        # Check for circular dependencies
        if depends_on and self.dependency_service:
            if await self.dependency_service.detect_circular_dependencies(task.id):
                raise ValueError("Circular dependency detected")
        
        # Add to write buffer
        task_id = await self.write_buffer.add(task)
        
        logger.debug(f"Task {task_id} submitted to queue {task.queue_name}")
        return task_id
    
    async def submit_batch(self, tasks: List[Dict[str, Any]]) -> List[str]:
        """
        Submit multiple tasks efficiently.
        
        Args:
            tasks: List of task specifications
            
        Returns:
            List of task IDs
        """
        task_ids = []
        for task_spec in tasks:
            task_id = await self.submit(**task_spec)
            task_ids.append(task_id)
        return task_ids
    
    async def get_task(self, task_id: str) -> Optional[Task]:
        """
        Get task by ID.
        
        Args:
            task_id: Task ID
            
        Returns:
            Task if found
        """
        if not self._started:
            raise SchedulerError("Scheduler not started")
        
        return await self.backend.get_task(task_id)
    
    async def cancel_task(self, task_id: str) -> bool:
        """
        Cancel a pending task.
        
        Args:
            task_id: Task ID
            
        Returns:
            True if cancelled
        """
        if not self._started:
            raise SchedulerError("Scheduler not started")
        
        task = await self.backend.get_task(task_id)
        if not task:
            return False
        
        if task.status in [TaskStatus.PENDING, TaskStatus.QUEUED]:
            # Update status to cancelled
            # This would need to be added to the backend interface
            logger.warning("Task cancellation not fully implemented")
            return False
        
        return False
    
    async def wait_for_task(self, task_id: str, timeout: Optional[float] = None) -> Optional[Task]:
        """
        Wait for a task to complete.
        
        Args:
            task_id: Task ID
            timeout: Maximum wait time in seconds
            
        Returns:
            Completed task or None if timeout
        """
        start = datetime.utcnow()
        
        while True:
            task = await self.backend.get_task(task_id)
            
            if not task:
                return None
            
            if task.status in [TaskStatus.COMPLETED, TaskStatus.FAILED, TaskStatus.DEAD]:
                return task
            
            # Check timeout
            if timeout:
                elapsed = (datetime.utcnow() - start).total_seconds()
                if elapsed >= timeout:
                    return None
            
            # Wait before checking again
            await asyncio.sleep(1)
    
    async def get_queue_status(self, queue_name: Optional[str] = None) -> Dict[str, Any]:
        """
        Get queue status.
        
        Args:
            queue_name: Queue name (all queues if None)
            
        Returns:
            Status dictionary
        """
        if not self._started:
            raise SchedulerError("Scheduler not started")
        
        if queue_name:
            size = await self.backend.get_queue_size(queue_name)
            return {
                'queue': queue_name,
                'size': size
            }
        
        # Get all queue statuses
        # This would need backend support for listing queues
        return {
            'default': await self.backend.get_queue_size('default')
        }
    
    async def scale_workers(self, target: int, queue_name: str = "default") -> int:
        """
        Scale workers for a queue.
        
        Args:
            target: Target worker count
            queue_name: Queue name
            
        Returns:
            Actual worker count
        """
        if not self._started or not self.worker_pool:
            raise SchedulerError("Worker pool not available")
        
        return await self.worker_pool.scale_to(target, queue_name)
    
    def get_status(self) -> Dict[str, Any]:
        """
        Get scheduler status.
        
        Returns:
            Status dictionary
        """
        status = {
            'started': self._started,
            'backend': self.backend.get_status() if self.backend else None,
            'write_buffer': self.write_buffer.get_status() if self.write_buffer else None,
            'worker_pool': self.worker_pool.get_status() if self.worker_pool else None,
            'leader_election': self.leader_election.get_status() if self.leader_election else None,
            'registry': {
                'handlers': len(self.registry),
                'handler_names': self.registry.list_handlers()
            }
        }
        
        return status
    
    async def _cleanup_loop(self) -> None:
        """Background task for cleanup operations."""
        while self._started and not self._stopping:
            try:
                await asyncio.sleep(self.config.cleanup_interval_seconds)
                
                # Only run if we're the leader
                if self.leader_election and await self.leader_election.is_leader("scheduler:cleanup"):
                    # Reclaim expired leases
                    reclaimed = await self.backend.reclaim_expired_leases()
                    if reclaimed > 0:
                        logger.info(f"Reclaimed {reclaimed} expired leases")
                    
                    # Clean up old payloads
                    if self.payload_service:
                        deleted = await self.payload_service.cleanup_old_payloads()
                        if deleted > 0:
                            logger.info(f"Cleaned up {deleted} old payloads")
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Cleanup loop error: {e}")
    
    async def _monitor_loop(self) -> None:
        """Background task for monitoring."""
        while self._started and not self._stopping:
            try:
                await asyncio.sleep(self.config.health_check_interval)
                
                # Log status periodically
                status = self.get_status()
                logger.debug(f"Scheduler status: {status}")
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Monitor loop error: {e}")
    
    async def _on_become_cleanup_leader(self) -> None:
        """Callback when becoming cleanup leader."""
        logger.info("This instance is now the cleanup leader")
    
    async def __aenter__(self):
        """Async context manager entry."""
        await self.start()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.stop()


# Convenience functions
async def create_scheduler(config: Optional[SchedulerConfig] = None,
                          start_workers: bool = True) -> Scheduler:
    """
    Create and start a scheduler.
    
    Args:
        config: Scheduler configuration
        start_workers: Whether to start workers
        
    Returns:
        Started scheduler instance
    """
    scheduler = Scheduler(config)
    await scheduler.start(start_workers)
    return scheduler