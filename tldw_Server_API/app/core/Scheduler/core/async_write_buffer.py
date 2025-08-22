"""
Asynchronous write buffer with non-blocking flush operations.

This implementation uses a separate flush queue to avoid blocking
add operations during database writes.
"""

import asyncio
import json
from typing import List, Optional, Any, Deque
from collections import deque
from datetime import datetime
from pathlib import Path
from dataclasses import dataclass
from enum import Enum

from loguru import logger

from ..base import Task, BufferError, BufferClosedError, BufferFlushError
from ..base.queue_backend import QueueBackend
from ..config import SchedulerConfig


class FlushStrategy(Enum):
    """Flush strategy for handling full buffers"""
    BLOCK = "block"  # Block until space available (safest)
    DROP_OLDEST = "drop_oldest"  # Drop oldest tasks (lossy)
    SPILL_TO_DISK = "spill_to_disk"  # Spill to disk (slower but safe)
    REJECT = "reject"  # Reject new tasks (fail fast)


@dataclass
class BufferMetrics:
    """Metrics for monitoring buffer performance"""
    total_added: int = 0
    total_flushed: int = 0
    total_dropped: int = 0
    flush_operations: int = 0
    flush_failures: int = 0
    avg_flush_time_ms: float = 0.0
    max_flush_time_ms: float = 0.0
    buffer_overflows: int = 0


class AsyncWriteBuffer:
    """
    Non-blocking write buffer with separate flush queue.
    
    Key improvements over SafeWriteBuffer:
    - Non-blocking add operations (uses separate flush queue)
    - Multiple flush strategies for handling overload
    - Better metrics and monitoring
    - Adaptive flush intervals based on load
    - Backpressure handling
    """
    
    def __init__(self,
                 backend: QueueBackend,
                 config: SchedulerConfig,
                 flush_strategy: FlushStrategy = FlushStrategy.BLOCK,
                 max_queue_size: int = 10000):
        """
        Initialize async write buffer.
        
        Args:
            backend: Queue backend for flushing
            config: Scheduler configuration
            flush_strategy: Strategy for handling full buffers
            max_queue_size: Maximum tasks in flush queue
        """
        self.backend = backend
        self.config = config
        self.flush_strategy = flush_strategy
        self.max_queue_size = max_queue_size
        
        # Configuration
        self.flush_size = config.write_buffer_size
        self.base_flush_interval = config.write_buffer_flush_interval
        self.current_flush_interval = self.base_flush_interval
        
        # Active buffer for new tasks
        self.active_buffer: List[Task] = []
        self.active_lock = asyncio.Lock()
        
        # Flush queue for tasks ready to be written
        self.flush_queue: Deque[List[Task]] = deque()
        self.flush_queue_size = 0  # Track total tasks in queue
        self.flush_event = asyncio.Event()
        
        # Disk spill for overflow (if using SPILL_TO_DISK strategy)
        self.spill_path = config.base_path / 'buffer_spill'
        self.spill_files: List[Path] = []
        
        # Background tasks
        self._flush_worker: Optional[asyncio.Task] = None
        self._timer_task: Optional[asyncio.Task] = None
        
        # State
        self._closing = False
        self._closed = False
        self.metrics = BufferMetrics()
        
        # Backpressure handling
        self._backpressure_event = asyncio.Event()
        self._backpressure_event.set()  # Start unblocked
        
        logger.info(
            f"AsyncWriteBuffer initialized: size={self.flush_size}, "
            f"interval={self.base_flush_interval}s, strategy={flush_strategy.value}"
        )
    
    async def start(self):
        """Start background workers"""
        if self._flush_worker:
            return
        
        self._flush_worker = asyncio.create_task(self._flush_worker_loop())
        self._timer_task = asyncio.create_task(self._flush_timer_loop())
        logger.debug("AsyncWriteBuffer workers started")
    
    async def add(self, task: Task) -> str:
        """
        Add task to buffer (non-blocking).
        
        This method is designed to be fast and non-blocking.
        It only holds the lock long enough to add to the active buffer.
        
        Args:
            task: Task to add
            
        Returns:
            Task ID
            
        Raises:
            BufferClosedError: If buffer is closed
            BufferError: If buffer is full and strategy is REJECT
        """
        if self._closed:
            raise BufferClosedError("Buffer is closed")
        
        # Handle backpressure based on strategy
        if self.flush_queue_size >= self.max_queue_size:
            if self.flush_strategy == FlushStrategy.BLOCK:
                # Wait for space in queue
                logger.debug("Buffer full, waiting for space...")
                await self._backpressure_event.wait()
                
            elif self.flush_strategy == FlushStrategy.DROP_OLDEST:
                # Drop oldest batch from queue
                if self.flush_queue:
                    dropped = self.flush_queue.popleft()
                    self.flush_queue_size -= len(dropped)
                    self.metrics.total_dropped += len(dropped)
                    logger.warning(f"Dropped {len(dropped)} oldest tasks due to overflow")
                    
            elif self.flush_strategy == FlushStrategy.SPILL_TO_DISK:
                # Spill oldest batch to disk
                await self._spill_to_disk()
                
            elif self.flush_strategy == FlushStrategy.REJECT:
                self.metrics.buffer_overflows += 1
                raise BufferError(f"Buffer full, rejecting task (queue size: {self.flush_queue_size})")
        
        # Add to active buffer (fast operation)
        async with self.active_lock:
            self.active_buffer.append(task)
            self.metrics.total_added += 1
            
            # Check if we should trigger flush
            if len(self.active_buffer) >= self.flush_size:
                # Move active buffer to flush queue
                await self._queue_for_flush()
        
        return task.id
    
    async def _queue_for_flush(self):
        """
        Move active buffer to flush queue.
        Must be called with active_lock held.
        """
        if not self.active_buffer:
            return
        
        # Swap buffers atomically
        batch = self.active_buffer
        self.active_buffer = []
        
        # Add to flush queue
        self.flush_queue.append(batch)
        self.flush_queue_size += len(batch)
        
        # Signal flush worker
        self.flush_event.set()
        
        # Update backpressure
        if self.flush_queue_size >= self.max_queue_size:
            self._backpressure_event.clear()
    
    async def _flush_worker_loop(self):
        """
        Background worker that flushes batches to the database.
        Runs continuously until buffer is closed.
        """
        logger.debug("Flush worker started")
        
        while not self._closing:
            try:
                # Wait for work
                await self.flush_event.wait()
                
                # Process all available batches
                while self.flush_queue and not self._closing:
                    batch = self.flush_queue.popleft()
                    self.flush_queue_size -= len(batch)
                    
                    # Update backpressure
                    if self.flush_queue_size < self.max_queue_size * 0.8:
                        self._backpressure_event.set()
                    
                    # Flush to database
                    start_time = asyncio.get_event_loop().time()
                    try:
                        await self.backend.bulk_enqueue(batch)
                        
                        # Update metrics
                        flush_time_ms = (asyncio.get_event_loop().time() - start_time) * 1000
                        self.metrics.total_flushed += len(batch)
                        self.metrics.flush_operations += 1
                        self._update_flush_time_metrics(flush_time_ms)
                        
                        logger.debug(f"Flushed {len(batch)} tasks in {flush_time_ms:.2f}ms")
                        
                        # Adaptive flush interval based on performance
                        self._adapt_flush_interval(flush_time_ms)
                        
                    except Exception as e:
                        self.metrics.flush_failures += 1
                        logger.error(f"Flush failed: {e}. Re-queuing {len(batch)} tasks")
                        
                        # Re-queue at front for retry
                        self.flush_queue.appendleft(batch)
                        self.flush_queue_size += len(batch)
                        
                        # Back off on failure
                        await asyncio.sleep(1)
                
                # Clear event if queue is empty
                if not self.flush_queue:
                    self.flush_event.clear()
                    
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Flush worker error: {e}")
                await asyncio.sleep(1)
        
        logger.debug("Flush worker stopped")
    
    async def _flush_timer_loop(self):
        """
        Timer that triggers periodic flushes of partial buffers.
        """
        logger.debug("Flush timer started")
        
        while not self._closing:
            try:
                await asyncio.sleep(self.current_flush_interval)
                
                # Flush active buffer if it has data
                async with self.active_lock:
                    if self.active_buffer:
                        await self._queue_for_flush()
                        
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Flush timer error: {e}")
        
        logger.debug("Flush timer stopped")
    
    def _update_flush_time_metrics(self, flush_time_ms: float):
        """Update flush time metrics"""
        if self.metrics.flush_operations == 1:
            self.metrics.avg_flush_time_ms = flush_time_ms
        else:
            # Exponential moving average
            alpha = 0.1
            self.metrics.avg_flush_time_ms = (
                alpha * flush_time_ms + 
                (1 - alpha) * self.metrics.avg_flush_time_ms
            )
        
        self.metrics.max_flush_time_ms = max(
            self.metrics.max_flush_time_ms,
            flush_time_ms
        )
    
    def _adapt_flush_interval(self, flush_time_ms: float):
        """
        Adapt flush interval based on performance.
        
        If flushes are fast, we can flush more frequently.
        If they're slow, we should batch more.
        """
        if flush_time_ms < 50:  # Very fast
            self.current_flush_interval = max(
                self.base_flush_interval * 0.5,
                0.05  # Minimum 50ms
            )
        elif flush_time_ms > 500:  # Slow
            self.current_flush_interval = min(
                self.base_flush_interval * 2,
                5.0  # Maximum 5 seconds
            )
        else:
            self.current_flush_interval = self.base_flush_interval
    
    async def _spill_to_disk(self):
        """
        Spill oldest batch to disk when memory buffer is full.
        Used with SPILL_TO_DISK strategy.
        """
        if not self.flush_queue:
            return
        
        batch = self.flush_queue.popleft()
        self.flush_queue_size -= len(batch)
        
        try:
            # Ensure spill directory exists
            self.spill_path.mkdir(parents=True, exist_ok=True)
            
            # Write batch to disk
            timestamp = datetime.utcnow().strftime('%Y%m%d_%H%M%S_%f')
            spill_file = self.spill_path / f"spill_{timestamp}.json"
            
            spill_data = {
                'timestamp': datetime.utcnow().isoformat(),
                'task_count': len(batch),
                'tasks': [task.to_dict() for task in batch]
            }
            
            with open(spill_file, 'w') as f:
                json.dump(spill_data, f)
            
            self.spill_files.append(spill_file)
            logger.info(f"Spilled {len(batch)} tasks to {spill_file}")
            
        except Exception as e:
            logger.error(f"Failed to spill to disk: {e}")
            # Re-queue if spill fails
            self.flush_queue.appendleft(batch)
            self.flush_queue_size += len(batch)
    
    async def _recover_from_spill(self):
        """Recover tasks from spill files"""
        for spill_file in self.spill_files:
            try:
                with open(spill_file, 'r') as f:
                    spill_data = json.load(f)
                
                tasks = []
                for task_dict in spill_data['tasks']:
                    tasks.append(Task.from_dict(task_dict))
                
                if tasks:
                    # Add to flush queue
                    self.flush_queue.append(tasks)
                    self.flush_queue_size += len(tasks)
                    self.flush_event.set()
                
                # Delete spill file after recovery
                spill_file.unlink()
                logger.info(f"Recovered {len(tasks)} tasks from {spill_file}")
                
            except Exception as e:
                logger.error(f"Failed to recover from spill file {spill_file}: {e}")
        
        self.spill_files.clear()
    
    async def flush(self):
        """
        Manually trigger flush of active buffer.
        """
        async with self.active_lock:
            if self.active_buffer:
                await self._queue_for_flush()
    
    async def close(self):
        """
        Gracefully close the buffer.
        Ensures all tasks are flushed before closing.
        """
        if self._closed:
            return
        
        logger.info("Closing AsyncWriteBuffer...")
        self._closing = True
        
        # Stop accepting new tasks
        self._backpressure_event.clear()
        
        # Flush active buffer
        await self.flush()
        
        # Stop timer
        if self._timer_task:
            self._timer_task.cancel()
            try:
                await self._timer_task
            except asyncio.CancelledError:
                pass
        
        # Recover any spilled tasks
        if self.spill_files:
            await self._recover_from_spill()
        
        # Wait for flush queue to empty (with timeout)
        timeout = 30  # seconds
        start_time = asyncio.get_event_loop().time()
        
        while self.flush_queue:
            if asyncio.get_event_loop().time() - start_time > timeout:
                logger.error(f"Timeout waiting for flush queue to empty. {self.flush_queue_size} tasks remaining")
                # Emergency backup
                await self._emergency_backup()
                break
            
            await asyncio.sleep(0.1)
        
        # Stop flush worker
        if self._flush_worker:
            self._flush_worker.cancel()
            try:
                await self._flush_worker
            except asyncio.CancelledError:
                pass
        
        self._closed = True
        
        logger.info(
            f"AsyncWriteBuffer closed. Metrics: "
            f"added={self.metrics.total_added}, "
            f"flushed={self.metrics.total_flushed}, "
            f"dropped={self.metrics.total_dropped}, "
            f"failures={self.metrics.flush_failures}"
        )
    
    async def _emergency_backup(self):
        """Emergency backup of unflushed tasks"""
        all_tasks = []
        
        # Collect all tasks
        for batch in self.flush_queue:
            all_tasks.extend(batch)
        
        if self.active_buffer:
            all_tasks.extend(self.active_buffer)
        
        if not all_tasks:
            return
        
        try:
            backup_path = self.config.emergency_backup_path
            backup_path.parent.mkdir(parents=True, exist_ok=True)
            
            timestamp = datetime.utcnow().strftime('%Y%m%d_%H%M%S')
            final_path = backup_path.parent / f"async_buffer_backup_{timestamp}.json"
            
            backup_data = {
                'timestamp': datetime.utcnow().isoformat(),
                'task_count': len(all_tasks),
                'tasks': [task.to_dict() for task in all_tasks]
            }
            
            with open(final_path, 'w') as f:
                json.dump(backup_data, f, indent=2)
            
            logger.critical(f"EMERGENCY: Saved {len(all_tasks)} tasks to {final_path}")
            
        except Exception as e:
            logger.critical(f"Failed to save emergency backup: {e}")
    
    def get_status(self) -> dict:
        """Get buffer status for monitoring"""
        return {
            'active_buffer_size': len(self.active_buffer),
            'flush_queue_size': self.flush_queue_size,
            'flush_queue_batches': len(self.flush_queue),
            'max_queue_size': self.max_queue_size,
            'flush_strategy': self.flush_strategy.value,
            'current_flush_interval': self.current_flush_interval,
            'is_closing': self._closing,
            'is_closed': self._closed,
            'backpressure_active': not self._backpressure_event.is_set(),
            'metrics': {
                'total_added': self.metrics.total_added,
                'total_flushed': self.metrics.total_flushed,
                'total_dropped': self.metrics.total_dropped,
                'flush_operations': self.metrics.flush_operations,
                'flush_failures': self.metrics.flush_failures,
                'avg_flush_time_ms': round(self.metrics.avg_flush_time_ms, 2),
                'max_flush_time_ms': round(self.metrics.max_flush_time_ms, 2),
                'buffer_overflows': self.metrics.buffer_overflows
            }
        }
    
    def __repr__(self) -> str:
        """String representation"""
        return (
            f"AsyncWriteBuffer(active={len(self.active_buffer)}, "
            f"queue={self.flush_queue_size}/{self.max_queue_size})"
        )