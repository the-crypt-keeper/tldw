"""
Leader election coordinator for distributed deployments.
Uses database-backed locks for coordination.
"""

import asyncio
import uuid
from typing import Optional, Callable, Any
from datetime import datetime, timedelta
from loguru import logger

from ..base.queue_backend import QueueBackend
from ..config import SchedulerConfig


class LeaderElection:
    """
    Coordinates leader election for distributed scheduler instances.
    
    Uses database locks (advisory locks for PostgreSQL, TTL-based for SQLite)
    to ensure only one instance acts as leader for specific resources.
    
    This is completely stateless - leadership status is always checked
    against the database.
    """
    
    def __init__(self, 
                 backend: QueueBackend,
                 config: SchedulerConfig,
                 instance_id: Optional[str] = None):
        """
        Initialize leader election coordinator.
        
        Args:
            backend: Queue backend for coordination
            config: Scheduler configuration
            instance_id: Unique instance identifier (auto-generated if not provided)
        """
        self.backend = backend
        self.config = config
        self.instance_id = instance_id or str(uuid.uuid4())
        self._leadership_tasks: dict[str, asyncio.Task] = {}
        
        logger.info(f"Leader election initialized for instance {self.instance_id}")
    
    async def acquire_leadership(self, 
                                 resource: str,
                                 ttl: Optional[int] = None) -> bool:
        """
        Try to acquire leadership for a resource.
        
        Args:
            resource: Resource name (e.g., "scheduler", "cleanup", "monitor")
            ttl: Time-to-live in seconds (defaults to config)
            
        Returns:
            True if leadership acquired
        """
        if ttl is None:
            ttl = self.config.leader_ttl_seconds
        
        try:
            acquired = await self.backend.acquire_leader(
                resource, self.instance_id, ttl
            )
            
            if acquired:
                logger.info(f"Instance {self.instance_id} acquired leadership for {resource}")
            else:
                logger.debug(f"Instance {self.instance_id} failed to acquire {resource}")
            
            return acquired
            
        except Exception as e:
            logger.error(f"Failed to acquire leadership for {resource}: {e}")
            return False
    
    async def release_leadership(self, resource: str) -> bool:
        """
        Release leadership for a resource.
        
        Args:
            resource: Resource name
            
        Returns:
            True if released successfully
        """
        try:
            # Cancel renewal task if exists
            if resource in self._leadership_tasks:
                self._leadership_tasks[resource].cancel()
                try:
                    await self._leadership_tasks[resource]
                except asyncio.CancelledError:
                    pass
                del self._leadership_tasks[resource]
            
            # Release in database
            released = await self.backend.release_leader(resource, self.instance_id)
            
            if released:
                logger.info(f"Instance {self.instance_id} released leadership for {resource}")
            
            return released
            
        except Exception as e:
            logger.error(f"Failed to release leadership for {resource}: {e}")
            return False
    
    async def is_leader(self, resource: str) -> bool:
        """
        Check if this instance is the leader for a resource.
        
        This is a stateless check against the database.
        
        Args:
            resource: Resource name
            
        Returns:
            True if this instance is the leader
        """
        # For now, we try to renew - if it succeeds, we're the leader
        # A more efficient method would be to add a check method to the backend
        try:
            return await self.backend.renew_leader(
                resource, self.instance_id, self.config.leader_ttl_seconds
            )
        except Exception:
            return False
    
    async def maintain_leadership(self,
                                  resource: str,
                                  callback: Optional[Callable[[], Any]] = None,
                                  ttl: Optional[int] = None,
                                  renew_interval: Optional[int] = None) -> asyncio.Task:
        """
        Start a task to maintain leadership with automatic renewal.
        
        Args:
            resource: Resource name
            callback: Optional callback when leadership is acquired
            ttl: Leadership TTL in seconds
            renew_interval: Renewal interval in seconds
            
        Returns:
            Asyncio task handling leadership
        """
        if resource in self._leadership_tasks:
            logger.warning(f"Leadership task already exists for {resource}")
            return self._leadership_tasks[resource]
        
        if ttl is None:
            ttl = self.config.leader_ttl_seconds
        
        if renew_interval is None:
            renew_interval = max(10, ttl // 3)  # Renew at 1/3 of TTL
        
        async def leadership_loop():
            """Maintain leadership with automatic renewal."""
            is_leader = False
            
            while True:
                try:
                    if not is_leader:
                        # Try to acquire leadership
                        acquired = await self.acquire_leadership(resource, ttl)
                        if acquired:
                            is_leader = True
                            logger.info(f"Became leader for {resource}")
                            
                            # Run callback if provided
                            if callback:
                                try:
                                    if asyncio.iscoroutinefunction(callback):
                                        await callback()
                                    else:
                                        callback()
                                except Exception as e:
                                    logger.error(f"Leadership callback error: {e}")
                    else:
                        # Try to renew leadership
                        renewed = await self.backend.renew_leader(
                            resource, self.instance_id, ttl
                        )
                        if not renewed:
                            is_leader = False
                            logger.warning(f"Lost leadership for {resource}")
                    
                    await asyncio.sleep(renew_interval)
                    
                except asyncio.CancelledError:
                    if is_leader:
                        await self.release_leadership(resource)
                    break
                except Exception as e:
                    logger.error(f"Leadership maintenance error for {resource}: {e}")
                    await asyncio.sleep(renew_interval)
        
        task = asyncio.create_task(leadership_loop())
        self._leadership_tasks[resource] = task
        return task
    
    async def stop_all(self) -> None:
        """
        Stop all leadership tasks and release all held resources.
        """
        logger.info(f"Stopping all leadership tasks for instance {self.instance_id}")
        
        # Cancel all tasks
        for resource, task in list(self._leadership_tasks.items()):
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass
            
            # Try to release leadership
            try:
                await self.backend.release_leader(resource, self.instance_id)
            except Exception as e:
                logger.error(f"Failed to release {resource}: {e}")
        
        self._leadership_tasks.clear()
    
    def get_status(self) -> dict:
        """
        Get leadership status for monitoring.
        
        Returns:
            Status dictionary
        """
        return {
            'instance_id': self.instance_id,
            'active_resources': list(self._leadership_tasks.keys()),
            'task_count': len(self._leadership_tasks)
        }


class LeaderTask:
    """
    Decorator for tasks that should only run on the leader instance.
    """
    
    def __init__(self, 
                 election: LeaderElection,
                 resource: str,
                 check_interval: int = 10):
        """
        Initialize leader-only task decorator.
        
        Args:
            election: Leader election coordinator
            resource: Resource name for leadership
            check_interval: How often to check leadership (seconds)
        """
        self.election = election
        self.resource = resource
        self.check_interval = check_interval
    
    def __call__(self, func: Callable) -> Callable:
        """
        Wrap function to only execute on leader.
        
        Args:
            func: Function to wrap
            
        Returns:
            Wrapped function
        """
        async def wrapper(*args, **kwargs):
            """Check leadership before executing."""
            # Check if we're the leader
            is_leader = await self.election.is_leader(self.resource)
            
            if not is_leader:
                logger.debug(
                    f"Skipping {func.__name__} - not leader for {self.resource}"
                )
                return None
            
            # Execute the function
            return await func(*args, **kwargs)
        
        wrapper.__name__ = func.__name__
        wrapper.__doc__ = func.__doc__
        return wrapper


class DistributedLock:
    """
    Distributed lock using leader election mechanism.
    
    Useful for ensuring only one instance performs certain operations.
    """
    
    def __init__(self,
                 backend: QueueBackend,
                 resource: str,
                 instance_id: Optional[str] = None,
                 ttl: int = 60):
        """
        Initialize distributed lock.
        
        Args:
            backend: Queue backend
            resource: Lock resource name
            instance_id: Instance ID
            ttl: Lock TTL in seconds
        """
        self.backend = backend
        self.resource = f"lock:{resource}"
        self.instance_id = instance_id or str(uuid.uuid4())
        self.ttl = ttl
        self._acquired = False
    
    async def __aenter__(self):
        """Acquire lock on context entry."""
        max_attempts = 10
        for attempt in range(max_attempts):
            self._acquired = await self.backend.acquire_leader(
                self.resource, self.instance_id, self.ttl
            )
            if self._acquired:
                return self
            
            # Exponential backoff
            await asyncio.sleep(0.1 * (2 ** attempt))
        
        raise TimeoutError(f"Failed to acquire lock {self.resource}")
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Release lock on context exit."""
        if self._acquired:
            await self.backend.release_leader(self.resource, self.instance_id)
            self._acquired = False