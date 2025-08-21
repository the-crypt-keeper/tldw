"""
Test reliability improvements for the scheduler module.

Tests the async write buffer, improved worker pool, and resource management.
"""

import pytest
import asyncio
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, AsyncMock, patch
from datetime import datetime, timedelta

from ..config import SchedulerConfig
from ..core.async_write_buffer import AsyncWriteBuffer, FlushStrategy, BufferMetrics
from ..core.improved_worker_pool import (
    ImprovedWorker, ImprovedWorkerPool, WorkerState,
    ResourceTracker, WorkerMetrics
)
from ..base import Task, TaskStatus, TaskPriority
from ..base.registry import TaskRegistry


class TestAsyncWriteBuffer:
    """Test the async write buffer implementation."""
    
    @pytest.mark.asyncio
    async def test_non_blocking_add(self):
        """Test that add operations don't block during flush."""
        # Create mock backend
        backend = MagicMock()
        flush_event = asyncio.Event()
        
        async def slow_flush(tasks):
            """Simulate slow database flush"""
            flush_event.set()
            await asyncio.sleep(0.5)  # Simulate slow operation
            return [t.id for t in tasks]
        
        backend.bulk_enqueue = AsyncMock(side_effect=slow_flush)
        
        # Create buffer with small size to trigger flush
        config = SchedulerConfig(
            database_url=':memory:',
            write_buffer_size=2,
            write_buffer_flush_interval=10.0,
            base_path=Path('/tmp/test')
        )
        
        buffer = AsyncWriteBuffer(backend, config, FlushStrategy.BLOCK)
        await buffer.start()
        
        try:
            # Add tasks to trigger flush
            task1 = Task(handler='test', payload={'id': 1})
            task2 = Task(handler='test', payload={'id': 2})
            task3 = Task(handler='test', payload={'id': 3})
            
            # First two should trigger flush
            await buffer.add(task1)
            await buffer.add(task2)
            
            # Wait for flush to start
            await flush_event.wait()
            
            # This should not block (or block very briefly)
            start_time = asyncio.get_event_loop().time()
            await buffer.add(task3)
            add_time = asyncio.get_event_loop().time() - start_time
            
            # Add should be fast even though flush is slow
            assert add_time < 0.1, f"Add operation took {add_time}s, should be non-blocking"
            
            # Verify all tasks are eventually flushed
            await asyncio.sleep(1)
            assert buffer.metrics.total_flushed >= 2
            
        finally:
            await buffer.close()
    
    @pytest.mark.asyncio
    async def test_flush_strategies(self):
        """Test different flush strategies."""
        backend = MagicMock()
        backend.bulk_enqueue = AsyncMock(return_value=['task-1', 'task-2'])
        
        config = SchedulerConfig(
            database_url=':memory:',
            write_buffer_size=2,
            write_buffer_flush_interval=10.0,
            base_path=Path('/tmp/test')
        )
        
        # Test DROP_OLDEST strategy
        buffer = AsyncWriteBuffer(
            backend, config,
            FlushStrategy.DROP_OLDEST,
            max_queue_size=4
        )
        await buffer.start()
        
        try:
            # Fill up the queue
            for i in range(10):
                task = Task(handler='test', payload={'id': i})
                await buffer.add(task)
            
            # Some tasks should be dropped
            assert buffer.metrics.total_dropped > 0
            
        finally:
            await buffer.close()
        
        # Test REJECT strategy
        buffer2 = AsyncWriteBuffer(
            backend, config,
            FlushStrategy.REJECT,
            max_queue_size=2
        )
        await buffer2.start()
        
        try:
            # Fill up the queue
            for i in range(5):
                task = Task(handler='test', payload={'id': i})
                if i < 4:
                    await buffer2.add(task)
                else:
                    # Should reject when full
                    with pytest.raises(Exception, match="Buffer full"):
                        await buffer2.add(task)
            
        finally:
            await buffer2.close()
    
    @pytest.mark.asyncio
    async def test_adaptive_flush_interval(self):
        """Test that flush interval adapts to performance."""
        backend = MagicMock()
        
        # First flush is fast
        async def fast_flush(tasks):
            await asyncio.sleep(0.01)  # 10ms
            return [t.id for t in tasks]
        
        backend.bulk_enqueue = AsyncMock(side_effect=fast_flush)
        
        config = SchedulerConfig(
            database_url=':memory:',
            write_buffer_size=2,
            write_buffer_flush_interval=1.0,
            base_path=Path('/tmp/test')
        )
        
        buffer = AsyncWriteBuffer(backend, config)
        await buffer.start()
        
        try:
            # Add tasks to trigger flush
            for i in range(4):
                task = Task(handler='test', payload={'id': i})
                await buffer.add(task)
            
            await asyncio.sleep(0.1)
            
            # Interval should decrease for fast flushes
            assert buffer.current_flush_interval < config.write_buffer_flush_interval
            
            # Now make flushes slow
            async def slow_flush(tasks):
                await asyncio.sleep(0.6)  # 600ms
                return [t.id for t in tasks]
            
            backend.bulk_enqueue = AsyncMock(side_effect=slow_flush)
            
            # Add more tasks
            for i in range(4):
                task = Task(handler='test', payload={'id': i + 4})
                await buffer.add(task)
            
            await asyncio.sleep(1)
            
            # Interval should increase for slow flushes
            assert buffer.current_flush_interval > config.write_buffer_flush_interval
            
        finally:
            await buffer.close()
    
    @pytest.mark.asyncio
    async def test_emergency_backup_on_failure(self, tmp_path):
        """Test that emergency backup works when database fails."""
        backend = MagicMock()
        backend.bulk_enqueue = AsyncMock(side_effect=Exception("Database error"))
        
        config = SchedulerConfig(
            database_url=':memory:',
            write_buffer_size=100,
            write_buffer_flush_interval=10.0,
            base_path=tmp_path / 'scheduler',
            emergency_backup_path=tmp_path / 'scheduler' / 'emergency' / 'backup.json'
        )
        
        buffer = AsyncWriteBuffer(backend, config)
        await buffer.start()
        
        try:
            # Add tasks
            tasks = []
            for i in range(5):
                task = Task(handler='test', payload={'id': i})
                tasks.append(task)
                await buffer.add(task)
            
        finally:
            # Close should trigger emergency backup
            await buffer.close()
        
        # Check that backup file was created
        backup_files = list((tmp_path / 'scheduler' / 'emergency').glob('*.json'))
        assert len(backup_files) > 0, "Emergency backup file should be created"


class TestImprovedWorkerPool:
    """Test the improved worker pool implementation."""
    
    @pytest.mark.asyncio
    async def test_worker_resource_cleanup(self):
        """Test that worker properly cleans up resources."""
        backend = MagicMock()
        registry = TaskRegistry()
        
        # Register a test handler
        @registry.task(name='test_handler')
        async def test_handler(payload):
            await asyncio.sleep(0.1)
            return {'result': 'success'}
        
        config = SchedulerConfig(
            database_url=':memory:',
            lease_duration_seconds=10,
            lease_renewal_interval=2,
            default_task_timeout=5
        )
        
        worker = ImprovedWorker(
            worker_id='test-worker',
            backend=backend,
            registry=registry,
            config=config
        )
        
        # Track resources
        initial_tasks = len(asyncio.all_tasks())
        
        await worker.start()
        
        # Stop worker
        await worker.stop(timeout=2)
        
        # Check that resources were cleaned up
        await asyncio.sleep(0.1)  # Let cleanup complete
        final_tasks = len(asyncio.all_tasks())
        
        # Should have roughly the same number of tasks (allowing for test framework tasks)
        assert abs(final_tasks - initial_tasks) <= 2, "Worker should clean up all tasks"
        assert worker.state == WorkerState.STOPPED
        assert len(worker.resource_tracker.tasks) == 0
        assert len(worker.resource_tracker.leases) == 0
    
    @pytest.mark.asyncio
    async def test_worker_health_checks(self):
        """Test worker health check functionality."""
        backend = MagicMock()
        backend.dequeue_atomic = AsyncMock(return_value=None)
        
        registry = TaskRegistry()
        config = SchedulerConfig(database_url=':memory:')
        
        worker = ImprovedWorker(
            worker_id='test-worker',
            backend=backend,
            registry=registry,
            config=config
        )
        
        # Healthy worker
        assert worker.is_healthy() == True
        
        # Simulate errors
        worker._consecutive_errors = 5
        assert worker.is_healthy() == False
        
        # Simulate stale heartbeat
        worker._consecutive_errors = 0
        worker._last_heartbeat = datetime.utcnow() - timedelta(minutes=2)
        assert worker.is_healthy() == False
        
        # Error state
        worker._last_heartbeat = datetime.utcnow()
        worker.state = WorkerState.ERROR
        assert worker.is_healthy() == False
    
    @pytest.mark.asyncio
    async def test_worker_recycling(self):
        """Test that workers are recycled properly."""
        backend = MagicMock()
        backend.dequeue_atomic = AsyncMock(return_value=None)
        backend.get_queue_size = AsyncMock(return_value=0)
        
        registry = TaskRegistry()
        
        config = SchedulerConfig(
            database_url=':memory:',
            min_workers=1,
            max_workers=5,
            worker_recycle_after_tasks=10
        )
        
        pool = ImprovedWorkerPool(backend, registry, config)
        await pool.start()
        
        try:
            # Get initial worker
            assert len(pool.workers) == 1
            initial_worker_id = list(pool.workers.keys())[0]
            worker = pool.workers[initial_worker_id]
            
            # Simulate processing enough tasks to trigger recycle
            worker.metrics.tasks_processed = 10
            worker.state = WorkerState.RECYCLING
            
            # Put worker in recycle queue
            await pool._recycle_queue.put(initial_worker_id)
            
            # Wait for recycle to complete
            await asyncio.sleep(0.5)
            
            # Should have a new worker
            assert len(pool.workers) == 1
            new_worker_id = list(pool.workers.keys())[0]
            assert new_worker_id != initial_worker_id
            
        finally:
            await pool.stop()
    
    @pytest.mark.asyncio
    async def test_pool_graceful_shutdown(self):
        """Test that pool shuts down gracefully."""
        backend = MagicMock()
        backend.dequeue_atomic = AsyncMock(return_value=None)
        backend.get_queue_size = AsyncMock(return_value=0)
        
        registry = TaskRegistry()
        
        config = SchedulerConfig(
            database_url=':memory:',
            min_workers=2,
            max_workers=5
        )
        
        pool = ImprovedWorkerPool(backend, registry, config)
        await pool.start()
        
        # Verify workers started
        assert len(pool.workers) == 2
        
        # Track initial tasks
        initial_tasks = len(asyncio.all_tasks())
        
        # Stop pool
        await pool.stop(timeout=5)
        
        # Verify cleanup
        assert len(pool.workers) == 0
        assert pool._stop_event.is_set()
        assert pool._stopped_event.is_set()
        
        # Check that background tasks were cleaned up
        await asyncio.sleep(0.1)
        final_tasks = len(asyncio.all_tasks())
        
        # Should have cleaned up background tasks
        assert abs(final_tasks - initial_tasks) <= 3, "Pool should clean up all background tasks"
    
    @pytest.mark.asyncio
    async def test_worker_force_stop_on_timeout(self):
        """Test that workers are force-stopped on timeout."""
        backend = MagicMock()
        registry = TaskRegistry()
        
        # Register a slow handler
        @registry.task(name='slow_handler')
        async def slow_handler(payload):
            await asyncio.sleep(10)  # Very slow
            return {'result': 'success'}
        
        config = SchedulerConfig(
            database_url=':memory:',
            default_task_timeout=20
        )
        
        worker = ImprovedWorker(
            worker_id='test-worker',
            backend=backend,
            registry=registry,
            config=config
        )
        
        # Mock task processing
        task = Task(handler='slow_handler', payload={'test': 'data'})
        worker.current_task = task
        worker.state = WorkerState.BUSY
        
        # Create a slow-running task
        async def slow_task():
            await asyncio.sleep(10)
        
        worker._main_task = asyncio.create_task(slow_task())
        
        # Stop with short timeout
        await worker.stop(timeout=0.5)
        
        # Worker should be stopped even though task was slow
        assert worker.state == WorkerState.STOPPED
        assert worker._main_task.cancelled()
    
    @pytest.mark.asyncio
    async def test_auto_scaling(self):
        """Test that pool scales based on load."""
        backend = MagicMock()
        backend.dequeue_atomic = AsyncMock(return_value=None)
        
        registry = TaskRegistry()
        
        config = SchedulerConfig(
            database_url=':memory:',
            min_workers=1,
            max_workers=5
        )
        
        pool = ImprovedWorkerPool(backend, registry, config)
        await pool.start()
        
        try:
            # Initially should have min workers
            assert len(pool.workers) == 1
            
            # Simulate high load
            backend.get_queue_size = AsyncMock(return_value=50)
            
            # Trigger scaling loop
            await pool._scaling_loop()
            
            # Should have scaled up
            assert len(pool.workers) > 1
            
            # Simulate low load
            backend.get_queue_size = AsyncMock(return_value=0)
            
            # Mark workers as idle
            for worker in pool.workers.values():
                worker.state = WorkerState.IDLE
            
            # Trigger scaling loop
            await pool._scaling_loop()
            
            # Should scale down but maintain minimum
            assert len(pool.workers) >= config.min_workers
            
        finally:
            await pool.stop()


class TestResourceTracker:
    """Test the resource tracker."""
    
    @pytest.mark.asyncio
    async def test_task_tracking(self):
        """Test that tasks are properly tracked and cleaned up."""
        tracker = ResourceTracker()
        
        # Create some tasks
        async def dummy_task():
            await asyncio.sleep(0.1)
        
        task1 = asyncio.create_task(dummy_task())
        task2 = asyncio.create_task(dummy_task())
        
        await tracker.register_task(task1)
        await tracker.register_task(task2)
        
        assert len(tracker.tasks) == 2
        
        # Clean up
        await tracker.cleanup_all(timeout=1)
        
        assert len(tracker.tasks) == 0
        assert task1.cancelled() or task1.done()
        assert task2.cancelled() or task2.done()
    
    @pytest.mark.asyncio
    async def test_lease_tracking(self):
        """Test that leases are properly tracked and cleaned up."""
        tracker = ResourceTracker()
        
        # Create lease tasks
        async def lease_renewal():
            while True:
                await asyncio.sleep(1)
        
        lease1 = asyncio.create_task(lease_renewal())
        lease2 = asyncio.create_task(lease_renewal())
        
        await tracker.register_lease('task-1', lease1)
        await tracker.register_lease('task-2', lease2)
        
        assert len(tracker.leases) == 2
        
        # Unregister one lease
        await tracker.unregister_lease('task-1')
        assert len(tracker.leases) == 1
        assert lease1.cancelled()
        
        # Clean up remaining
        await tracker.cleanup_all(timeout=1)
        assert len(tracker.leases) == 0
        assert lease2.cancelled()


if __name__ == '__main__':
    pytest.main([__file__, '-v'])