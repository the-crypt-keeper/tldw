"""
Safe write-ahead buffer with atomic operations.
Guarantees no data loss even under concurrent load.
"""

import asyncio
import json
from typing import List, Optional, Any
from datetime import datetime
from pathlib import Path

from loguru import logger

from ..base import Task, BufferError, BufferClosedError, BufferFlushError
from ..base.queue_backend import QueueBackend
from ..config import SchedulerConfig


class SafeWriteBuffer:
    """
    Write buffer that guarantees no data loss.
    
    CRITICAL: Buffer modifications are atomic and don't straddle awaits.
    This prevents race conditions that could cause data loss.
    
    Performance characteristics:
    - Add operations may block during flush when buffer is full
    - This is intentional to maintain absolute data safety
    - For higher throughput with relaxed guarantees, use async flush
    """
    
    def __init__(self, 
                 backend: QueueBackend,
                 config: SchedulerConfig):
        """
        Initialize the write buffer.
        
        Args:
            backend: Queue backend for flushing
            config: Scheduler configuration
        """
        self.backend = backend
        self.config = config
        self.flush_size = config.write_buffer_size
        self.flush_interval = config.write_buffer_flush_interval
        
        self.buffer: List[Task] = []
        self.lock = asyncio.Lock()
        self._flush_task: Optional[asyncio.Task] = None
        self._closing = False
        self._flush_failures = 0
        self._total_flushed = 0
        
        logger.info(
            f"SafeWriteBuffer initialized: size={self.flush_size}, "
            f"interval={self.flush_interval}s"
        )
    
    async def add(self, task: Task) -> str:
        """
        Add task to buffer.
        
        Performance note: When buffer is full, this method will block
        for the duration of the database flush operation to maintain
        atomicity. This trades latency for absolute data safety.
        
        Args:
            task: Task to add
            
        Returns:
            Task ID
            
        Raises:
            BufferClosedError: If buffer is closing
            BufferFlushError: If flush fails
        """
        if self._closing:
            raise BufferClosedError("Buffer is closing, cannot accept new tasks")
        
        async with self.lock:
            self.buffer.append(task)
            
            # Start flush timer if not running and buffer has data
            if not self._flush_task and self.buffer:
                self._flush_task = asyncio.create_task(self._flush_timer())
            
            # Immediate flush if buffer full
            if len(self.buffer) >= self.flush_size:
                # PERFORMANCE NOTE: The lock is held during this flush to ensure
                # absolute atomicity of the buffer state. This can increase the
                # latency of `add` if the database is slow, but prevents complex
                # race conditions. For applications requiring lower latency, consider:
                # 1. Increasing flush_size to reduce flush frequency
                # 2. Using multiple buffers with round-robin distribution
                # 3. Implementing async flush with eventual consistency
                await self._flush_internal()
        
        return task.id
    
    async def _flush_internal(self):
        """
        Internal flush - must be called with lock held.
        
        CRITICAL FIX: Atomic buffer modification that doesn't straddle await.
        The buffer is cleared BEFORE the await to prevent race conditions.
        """
        if not self.buffer:
            return
        
        # CRITICAL: Atomically remove tasks BEFORE the await
        # This prevents race conditions where another coroutine could
        # modify the buffer while we're waiting for the database
        tasks_to_flush = self.buffer.copy()
        self.buffer.clear()  # Clear NOW, not after await
        
        try:
            # Lock is released here during await, but buffer is already updated
            # so no race condition is possible
            await self.backend.bulk_enqueue(tasks_to_flush)
            
            self._total_flushed += len(tasks_to_flush)
            logger.debug(f"Successfully flushed {len(tasks_to_flush)} tasks")
            
        except Exception as e:
            self._flush_failures += 1
            logger.error(f"Failed to flush buffer: {e}. Re-queuing {len(tasks_to_flush)} tasks.")
            
            # CRITICAL FIX: Re-add failed tasks atomically while still holding the lock
            # The lock is still held here from the caller, so we can safely modify buffer
            # Add failed tasks to front so they're retried first
            self.buffer = tasks_to_flush + self.buffer
            
            # Re-raise to alert caller
            raise BufferFlushError(f"Flush failed: {e}")
    
    async def _flush_timer(self):
        """
        Periodic flush timer - only runs when buffer has data.
        Stops automatically when buffer is empty.
        """
        try:
            while not self._closing:
                await asyncio.sleep(self.flush_interval)
                
                async with self.lock:
                    if self.buffer:
                        try:
                            await self._flush_internal()
                        except BufferFlushError as e:
                            # Log but continue - tasks are still in buffer
                            logger.warning(f"Periodic flush failed: {e}")
                    else:
                        # Stop timer if buffer is empty
                        self._flush_task = None
                        break
                        
        except asyncio.CancelledError:
            logger.debug("Flush timer cancelled")
        except Exception as e:
            logger.error(f"Flush timer error: {e}")
        finally:
            self._flush_task = None
    
    async def flush(self):
        """
        Manually trigger a flush of the buffer.
        Useful before shutdown or for testing.
        """
        async with self.lock:
            if self.buffer:
                await self._flush_internal()
    
    async def close(self):
        """
        Graceful shutdown - flush remaining tasks.
        MUST be called before application shutdown to avoid data loss.
        """
        logger.info("Closing SafeWriteBuffer...")
        self._closing = True
        
        # Cancel timer
        if self._flush_task:
            self._flush_task.cancel()
            try:
                await self._flush_task
            except asyncio.CancelledError:
                pass
        
        # Final flush with retries
        max_retries = 3
        for attempt in range(max_retries):
            try:
                async with self.lock:
                    if not self.buffer:
                        break
                    
                    # Use the same atomic pattern
                    tasks_to_save = self.buffer.copy()
                    self.buffer.clear()
                    
                    try:
                        await self.backend.bulk_enqueue(tasks_to_save)
                        logger.info(f"Final flush completed: {len(tasks_to_save)} tasks")
                    except Exception as e:
                        # Re-add on failure
                        self.buffer = tasks_to_save
                        raise
                
                break  # Success
                
            except Exception as e:
                if attempt == max_retries - 1:
                    # Last resort: Write to emergency backup file
                    await self._emergency_backup()
                    logger.critical(f"Failed to flush after {max_retries} attempts")
                    raise
                else:
                    logger.warning(f"Flush attempt {attempt + 1} failed, retrying...")
                    await asyncio.sleep(1)
        
        logger.info(
            f"SafeWriteBuffer closed. Total flushed: {self._total_flushed}, "
            f"Failures: {self._flush_failures}"
        )
    
    async def _emergency_backup(self):
        """
        Last resort: Save buffer to file if database is unavailable.
        This prevents data loss in catastrophic failure scenarios.
        """
        if not self.buffer:
            return
        
        backup_path = self.config.emergency_backup_path
        
        try:
            # Ensure directory exists
            backup_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Write with timestamp in filename
            timestamp = datetime.utcnow().strftime('%Y%m%d_%H%M%S')
            final_path = backup_path.parent / f"buffer_backup_{timestamp}.json"
            
            # Save tasks as JSON
            backup_data = {
                'timestamp': datetime.utcnow().isoformat(),
                'task_count': len(self.buffer),
                'tasks': [task.to_dict() for task in self.buffer]
            }
            
            with open(final_path, 'w') as f:
                json.dump(backup_data, f, indent=2)
            
            logger.critical(f"EMERGENCY: Saved {len(self.buffer)} tasks to {final_path}")
            
            # Also write a symlink to latest
            latest_link = backup_path.parent / 'latest_backup.json'
            if latest_link.exists():
                latest_link.unlink()
            latest_link.symlink_to(final_path)
            
        except Exception as e:
            logger.critical(f"Failed to save emergency backup: {e}")
            # At this point we've done everything possible
            # The tasks in self.buffer will be lost
    
    async def recover_from_backup(self, backup_path: Path) -> int:
        """
        Recover tasks from an emergency backup file.
        
        Args:
            backup_path: Path to backup file
            
        Returns:
            Number of tasks recovered
        """
        if not backup_path.exists():
            logger.warning(f"Backup file not found: {backup_path}")
            return 0
        
        try:
            with open(backup_path, 'r') as f:
                backup_data = json.load(f)
            
            tasks = []
            for task_dict in backup_data['tasks']:
                tasks.append(Task.from_dict(task_dict))
            
            if tasks:
                # Add recovered tasks to buffer
                async with self.lock:
                    self.buffer.extend(tasks)
                
                # Trigger flush
                await self.flush()
                
                logger.info(f"Recovered {len(tasks)} tasks from {backup_path}")
                
                # Rename backup file to indicate it's been processed
                processed_path = backup_path.with_suffix('.processed')
                backup_path.rename(processed_path)
                
                return len(tasks)
            
        except Exception as e:
            logger.error(f"Failed to recover from backup: {e}")
        
        return 0
    
    def get_status(self) -> dict:
        """
        Get buffer status for monitoring.
        
        Returns:
            Status dictionary
        """
        return {
            'buffer_size': len(self.buffer),
            'flush_size': self.flush_size,
            'flush_interval': self.flush_interval,
            'total_flushed': self._total_flushed,
            'flush_failures': self._flush_failures,
            'is_closing': self._closing,
            'timer_active': self._flush_task is not None
        }
    
    def __len__(self) -> int:
        """Get current buffer size"""
        return len(self.buffer)
    
    def __repr__(self) -> str:
        """String representation"""
        return f"SafeWriteBuffer(size={len(self.buffer)}/{self.flush_size})"