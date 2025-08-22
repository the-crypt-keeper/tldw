"""
Stateless lease management service.
All operations query the database directly.
"""

import asyncio
from typing import Optional, List
from datetime import datetime, timedelta
from loguru import logger

from ..base.queue_backend import QueueBackend
from ..base.exceptions import LeaseError


class LeaseService:
    """
    Manages task leases without maintaining any state.
    
    All lease information is stored in the database and queried
    on-demand. This service provides convenience methods for
    lease operations but holds no state itself.
    """
    
    def __init__(self, backend: QueueBackend, lease_duration: int = 300):
        """
        Initialize lease service.
        
        Args:
            backend: Queue backend for database operations
            lease_duration: Default lease duration in seconds
        """
        self.backend = backend
        self.lease_duration = lease_duration
        self._reaper_task: Optional[asyncio.Task] = None
    
    async def start_reaper(self, interval: int = 60) -> None:
        """
        Start background task to reclaim expired leases.
        
        Args:
            interval: Check interval in seconds
        """
        if self._reaper_task:
            logger.warning("Lease reaper already running")
            return
        
        self._reaper_task = asyncio.create_task(
            self._reaper_loop(interval)
        )
        logger.info(f"Lease reaper started with {interval}s interval")
    
    async def stop_reaper(self) -> None:
        """
        Stop the lease reaper task.
        """
        if self._reaper_task:
            self._reaper_task.cancel()
            try:
                await self._reaper_task
            except asyncio.CancelledError:
                pass
            self._reaper_task = None
            logger.info("Lease reaper stopped")
    
    async def _reaper_loop(self, interval: int) -> None:
        """
        Background loop to reclaim expired leases.
        
        Args:
            interval: Check interval in seconds
        """
        while True:
            try:
                await asyncio.sleep(interval)
                
                # Reclaim expired leases
                reclaimed = await self.backend.reclaim_expired_leases()
                
                if reclaimed > 0:
                    logger.info(f"Reclaimed {reclaimed} expired leases")
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Lease reaper error: {e}")
    
    async def renew_lease(self, task_id: str, lease_id: str) -> bool:
        """
        Renew a task lease.
        
        Args:
            task_id: Task ID
            lease_id: Current lease ID
            
        Returns:
            True if renewed successfully
        """
        try:
            return await self.backend.renew_lease(task_id, lease_id)
        except Exception as e:
            logger.error(f"Failed to renew lease for task {task_id}: {e}")
            return False
    
    async def auto_renew(self, 
                         task_id: str, 
                         lease_id: str,
                         renew_interval: Optional[int] = None) -> asyncio.Task:
        """
        Start auto-renewal for a task lease.
        
        Args:
            task_id: Task ID
            lease_id: Lease ID
            renew_interval: Renewal interval (defaults to lease_duration/10)
            
        Returns:
            Asyncio task handling renewal
        """
        if renew_interval is None:
            renew_interval = max(10, self.lease_duration // 10)
        
        async def renew_loop():
            while True:
                await asyncio.sleep(renew_interval)
                success = await self.renew_lease(task_id, lease_id)
                if not success:
                    logger.warning(f"Failed to renew lease for task {task_id}")
                    break
        
        return asyncio.create_task(renew_loop())
    
    async def get_expired_tasks(self) -> List[str]:
        """
        Get list of task IDs with expired leases.
        
        This is a query operation, no state is maintained.
        
        Returns:
            List of task IDs
        """
        # This would be implemented differently for each backend
        # For now, we rely on the backend's reclaim method
        # In a real implementation, we'd add a query method to the backend
        logger.warning("get_expired_tasks not fully implemented")
        return []
    
    async def release_lease(self, task_id: str) -> bool:
        """
        Release a task lease early.
        
        Args:
            task_id: Task ID
            
        Returns:
            True if released
        """
        # This would require updating the task status
        # For now, we'd need to add this to the backend interface
        logger.warning("release_lease not fully implemented")
        return False
    
    def get_stats(self) -> dict:
        """
        Get lease statistics.
        
        Returns:
            Statistics dictionary
        """
        return {
            'lease_duration': self.lease_duration,
            'reaper_running': self._reaper_task is not None
        }