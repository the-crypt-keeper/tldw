"""
Basic tests for the scheduler components.
"""

import asyncio
import pytest
from datetime import datetime
import tempfile
from pathlib import Path

from ..base import Task, TaskStatus, TaskPriority
from ..backends.sqlite_backend import SQLiteBackend
from ..core.write_buffer import SafeWriteBuffer
from ..config import SchedulerConfig


@pytest.mark.asyncio
async def test_sqlite_backend_basic():
    """Test basic SQLite backend operations"""
    # Create temporary database
    with tempfile.TemporaryDirectory() as tmpdir:
        config = SchedulerConfig(
            database_url=f"sqlite:///{tmpdir}/test.db",
            base_path=Path(tmpdir)
        )
        
        backend = SQLiteBackend(config)
        await backend.connect()
        
        try:
            # Create a task
            task = Task(
                handler="test_handler",
                payload={"test": "data"},
                queue_name="test_queue",
                priority=TaskPriority.NORMAL.value
            )
            
            # Enqueue task
            task_id = await backend.enqueue(task)
            assert task_id == task.id
            
            # Get task
            retrieved = await backend.get_task(task_id)
            assert retrieved is not None
            assert retrieved.handler == "test_handler"
            assert retrieved.payload == {"test": "data"}
            
            # Check queue size
            size = await backend.get_queue_size("test_queue")
            assert size == 1
            
            # Dequeue task
            dequeued = await backend.dequeue_atomic("test_queue", "worker1")
            assert dequeued is not None
            assert dequeued.id == task_id
            assert dequeued.status == TaskStatus.RUNNING
            assert dequeued.worker_id == "worker1"
            
            # Queue should be empty now
            size = await backend.get_queue_size("test_queue")
            assert size == 0
            
            # Acknowledge task
            ack_result = await backend.ack(task_id, {"result": "success"})
            assert ack_result is True
            
            # Task should be completed
            completed = await backend.get_task(task_id)
            assert completed.status == TaskStatus.COMPLETED
            
        finally:
            await backend.disconnect()


@pytest.mark.asyncio
async def test_bulk_enqueue_with_idempotency():
    """Test bulk enqueue with idempotency"""
    with tempfile.TemporaryDirectory() as tmpdir:
        config = SchedulerConfig(
            database_url=f"sqlite:///{tmpdir}/test.db",
            base_path=Path(tmpdir)
        )
        
        backend = SQLiteBackend(config)
        await backend.connect()
        
        try:
            # Create tasks with some having idempotency keys
            tasks = [
                Task(handler="handler1", idempotency_key="unique1"),
                Task(handler="handler2", idempotency_key="unique2"),
                Task(handler="handler3"),  # No idempotency key
                Task(handler="handler4", idempotency_key="unique1"),  # Duplicate!
            ]
            
            # Bulk enqueue
            result = await backend.bulk_enqueue(tasks)
            assert len(result) == 4  # All task IDs returned
            
            # Check that duplicate was ignored
            size = await backend.get_queue_size("default")
            assert size == 3  # Only 3 unique tasks
            
        finally:
            await backend.disconnect()


@pytest.mark.asyncio
async def test_dependency_resolution():
    """Test efficient 2-query dependency resolution"""
    with tempfile.TemporaryDirectory() as tmpdir:
        config = SchedulerConfig(
            database_url=f"sqlite:///{tmpdir}/test.db",
            base_path=Path(tmpdir)
        )
        
        backend = SQLiteBackend(config)
        await backend.connect()
        
        try:
            # Create tasks with dependencies
            task1 = Task(id="task1", handler="handler1")
            task2 = Task(id="task2", handler="handler2", depends_on=["task1"])
            task3 = Task(id="task3", handler="handler3", depends_on=["task1", "task2"])
            
            # Enqueue all tasks
            await backend.enqueue(task1)
            await backend.enqueue(task2)
            await backend.enqueue(task3)
            
            # Initially only task1 should be ready (no dependencies)
            ready = await backend.get_ready_tasks()
            assert ready == ["task1"]
            
            # Complete task1
            await backend.execute(
                "UPDATE tasks SET status = 'completed' WHERE id = ?",
                "task1"
            )
            
            # Now task2 should be ready
            ready = await backend.get_ready_tasks()
            assert ready == ["task2"]
            
            # Complete task2
            await backend.execute(
                "UPDATE tasks SET status = 'completed' WHERE id = ?",
                "task2"
            )
            
            # Now task3 should be ready
            ready = await backend.get_ready_tasks()
            assert ready == ["task3"]
            
        finally:
            await backend.disconnect()


@pytest.mark.asyncio
async def test_write_buffer_atomic_operations():
    """Test that write buffer handles race conditions correctly"""
    with tempfile.TemporaryDirectory() as tmpdir:
        config = SchedulerConfig(
            database_url=f"sqlite:///{tmpdir}/test.db",
            base_path=Path(tmpdir),
            write_buffer_size=10,
            write_buffer_flush_interval=0.5
        )
        
        backend = SQLiteBackend(config)
        await backend.connect()
        
        buffer = SafeWriteBuffer(backend, config)
        
        try:
            # Add tasks concurrently
            tasks = []
            for i in range(20):
                task = Task(handler=f"handler{i}", payload={"id": i})
                tasks.append(task)
            
            # Add tasks concurrently to test race conditions
            async def add_task(task):
                return await buffer.add(task)
            
            results = await asyncio.gather(*[add_task(t) for t in tasks])
            assert len(results) == 20
            
            # Force flush
            await buffer.flush()
            
            # Check all tasks made it to database
            size = await backend.get_queue_size("default")
            assert size == 20
            
            # Test graceful close
            await buffer.close()
            
            # Verify buffer status
            status = buffer.get_status()
            assert status['total_flushed'] >= 20
            assert status['buffer_size'] == 0
            
        finally:
            await backend.disconnect()


@pytest.mark.asyncio
async def test_write_buffer_recovery():
    """Test emergency backup and recovery"""
    with tempfile.TemporaryDirectory() as tmpdir:
        config = SchedulerConfig(
            database_url=f"sqlite:///{tmpdir}/test.db",
            base_path=Path(tmpdir),
            write_buffer_size=10
        )
        
        # Create backend that will fail
        backend = SQLiteBackend(config)
        await backend.connect()
        
        buffer = SafeWriteBuffer(backend, config)
        
        # Add some tasks
        tasks = [Task(handler=f"handler{i}") for i in range(5)]
        for task in tasks:
            await buffer.add(task)
        
        # Simulate database failure by disconnecting
        await backend.disconnect()
        
        # Try to close buffer - should create emergency backup
        try:
            await buffer.close()
        except:
            pass  # Expected to fail
        
        # Check for backup file
        backup_files = list(config.emergency_backup_path.parent.glob("buffer_backup_*.json"))
        assert len(backup_files) > 0
        
        # Create new buffer and recover
        await backend.connect()
        new_buffer = SafeWriteBuffer(backend, config)
        
        recovered = await new_buffer.recover_from_backup(backup_files[0])
        assert recovered == 5
        
        # Check tasks were recovered
        size = await backend.get_queue_size("default")
        assert size == 5
        
        await backend.disconnect()


if __name__ == "__main__":
    # Run basic test
    asyncio.run(test_sqlite_backend_basic())
    print("✓ Basic SQLite backend test passed")
    
    asyncio.run(test_bulk_enqueue_with_idempotency())
    print("✓ Bulk enqueue with idempotency test passed")
    
    asyncio.run(test_dependency_resolution())
    print("✓ Dependency resolution test passed")
    
    asyncio.run(test_write_buffer_atomic_operations())
    print("✓ Write buffer atomic operations test passed")
    
    asyncio.run(test_write_buffer_recovery())
    print("✓ Write buffer recovery test passed")
    
    print("\n✅ All tests passed!")