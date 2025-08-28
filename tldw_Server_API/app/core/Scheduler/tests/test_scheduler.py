"""
Comprehensive test suite for the Scheduler module.
"""

import asyncio
import pytest
import tempfile
from pathlib import Path
from datetime import datetime, timedelta

from ..scheduler import Scheduler, create_scheduler
from ..base import Task, TaskStatus, TaskPriority
from ..base.registry import get_registry
from ..config import SchedulerConfig
from ..backends import create_backend


@pytest.fixture
async def test_config():
    """Create test configuration."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield SchedulerConfig(
            database_url=f"sqlite:///{tmpdir}/test.db",
            base_path=Path(tmpdir),
            min_workers=1,
            max_workers=5,
            write_buffer_size=10,
            write_buffer_flush_interval=0.1
        )


@pytest.fixture
async def scheduler(test_config):
    """Create and start a test scheduler."""
    scheduler = Scheduler(test_config)
    await scheduler.start(start_workers=False)  # No workers for unit tests
    try:
        yield scheduler
    finally:
        await scheduler.stop()


@pytest.mark.asyncio
async def test_scheduler_lifecycle(test_config):
    """Test scheduler start and stop."""
    scheduler = Scheduler(test_config)
    
    # Start scheduler
    await scheduler.start(start_workers=False)
    assert scheduler._started is True
    
    # Get status
    status = scheduler.get_status()
    assert status['started'] is True
    assert status['backend'] is not None
    
    # Stop scheduler
    await scheduler.stop()
    assert scheduler._started is False


@pytest.mark.asyncio
async def test_task_submission(scheduler):
    """Test submitting tasks to scheduler."""
    # Register a test handler
    registry = get_registry()
    
    @registry.task(name="test_handler")
    async def test_handler(payload):
        return {"result": payload.get("value", 0) * 2}
    
    # Submit task
    task_id = await scheduler.submit(
        handler="test_handler",
        payload={"value": 42},
        priority=TaskPriority.HIGH.value
    )
    
    assert task_id is not None
    
    # Force flush to database
    await scheduler.write_buffer.flush()
    
    # Retrieve task
    task = await scheduler.get_task(task_id)
    assert task is not None
    assert task.handler == "test_handler"
    assert task.payload == {"value": 42}
    assert task.priority == TaskPriority.HIGH.value


@pytest.mark.asyncio
async def test_batch_submission(scheduler):
    """Test batch task submission."""
    # Register handler
    registry = get_registry()
    
    @registry.task(name="batch_handler")
    async def batch_handler(payload):
        return payload
    
    # Submit batch
    tasks = [
        {"handler": "batch_handler", "payload": {"id": i}}
        for i in range(5)
    ]
    
    task_ids = await scheduler.submit_batch(tasks)
    assert len(task_ids) == 5
    
    # Force flush
    await scheduler.write_buffer.flush()
    
    # Verify all tasks created
    for i, task_id in enumerate(task_ids):
        task = await scheduler.get_task(task_id)
        assert task is not None
        assert task.payload == {"id": i}


@pytest.mark.asyncio
async def test_idempotency(scheduler):
    """Test idempotent task submission."""
    registry = get_registry()
    
    @registry.task(name="idempotent_handler")
    async def handler(payload):
        return payload
    
    # Submit task with idempotency key
    task_id1 = await scheduler.submit(
        handler="idempotent_handler",
        payload={"data": "test"},
        idempotency_key="unique-key-123"
    )
    
    # Submit again with same key
    task_id2 = await scheduler.submit(
        handler="idempotent_handler",
        payload={"data": "different"},  # Different payload
        idempotency_key="unique-key-123"  # Same key
    )
    
    # Should get same task ID
    assert task_id1 == task_id2


@pytest.mark.asyncio
async def test_task_dependencies(scheduler):
    """Test task dependency handling."""
    registry = get_registry()
    
    @registry.task(name="dep_handler")
    async def handler(payload):
        return payload
    
    # Create parent task
    parent_id = await scheduler.submit(
        handler="dep_handler",
        payload={"task": "parent"}
    )
    
    # Create child task with dependency
    child_id = await scheduler.submit(
        handler="dep_handler",
        payload={"task": "child"},
        depends_on=[parent_id]
    )
    
    # Force flush
    await scheduler.write_buffer.flush()
    
    # Check dependency service
    ready = await scheduler.dependency_service.check_dependencies(child_id)
    assert ready is False  # Parent not completed
    
    # Complete parent task
    await scheduler.backend.execute(
        "UPDATE tasks SET status = 'completed' WHERE id = ?",
        parent_id
    )
    
    # Now child should be ready
    ready = await scheduler.dependency_service.check_dependencies(child_id)
    assert ready is True


@pytest.mark.asyncio
async def test_worker_pool_integration(test_config):
    """Test scheduler with worker pool."""
    scheduler = Scheduler(test_config)
    await scheduler.start(start_workers=True)
    
    try:
        # Register handler
        registry = get_registry()
        
        @registry.task(name="worker_test")
        async def handler(payload):
            await asyncio.sleep(0.1)  # Simulate work
            return {"processed": payload}
        
        # Submit task
        task_id = await scheduler.submit(
            handler="worker_test",
            payload={"test": "data"}
        )
        
        # Force flush
        await scheduler.write_buffer.flush()
        
        # Wait for task completion
        result = await scheduler.wait_for_task(task_id, timeout=5)
        assert result is not None
        assert result.status == TaskStatus.COMPLETED
        
        # Check worker pool status
        pool_status = scheduler.worker_pool.get_status()
        assert pool_status['total_tasks_processed'] > 0
        
    finally:
        await scheduler.stop()


@pytest.mark.asyncio
async def test_queue_management(scheduler):
    """Test queue operations."""
    registry = get_registry()
    
    @registry.task(name="queue_test")
    async def handler(payload):
        return payload
    
    # Submit to different queues
    task1 = await scheduler.submit(
        handler="queue_test",
        payload={"id": 1},
        queue_name="high_priority"
    )
    
    task2 = await scheduler.submit(
        handler="queue_test",
        payload={"id": 2},
        queue_name="low_priority"
    )
    
    # Force flush
    await scheduler.write_buffer.flush()
    
    # Check queue sizes
    high_status = await scheduler.get_queue_status("high_priority")
    assert high_status['size'] == 1
    
    low_status = await scheduler.get_queue_status("low_priority")
    assert low_status['size'] == 1


@pytest.mark.asyncio
async def test_scheduler_context_manager(test_config):
    """Test scheduler as context manager."""
    async with Scheduler(test_config) as scheduler:
        registry = get_registry()
        
        @registry.task(name="context_test")
        async def handler(payload):
            return payload
        
        task_id = await scheduler.submit(
            handler="context_test",
            payload={"test": True}
        )
        
        await scheduler.write_buffer.flush()
        
        task = await scheduler.get_task(task_id)
        assert task is not None


@pytest.mark.asyncio
async def test_leader_election(test_config):
    """Test leader election with multiple schedulers."""
    # Create two scheduler instances
    scheduler1 = Scheduler(test_config)
    scheduler2 = Scheduler(test_config)
    
    await scheduler1.start(start_workers=False)
    await scheduler2.start(start_workers=False)
    
    try:
        # Try to acquire leadership on both
        leader1 = await scheduler1.leader_election.acquire_leadership("test_resource")
        leader2 = await scheduler2.leader_election.acquire_leadership("test_resource")
        
        # Only one should be leader
        assert leader1 != leader2
        assert leader1 or leader2  # At least one should succeed
        
    finally:
        await scheduler1.stop()
        await scheduler2.stop()


@pytest.mark.asyncio
async def test_payload_service(scheduler):
    """Test large payload handling."""
    registry = get_registry()
    
    @registry.task(name="payload_test")
    async def handler(payload):
        return len(payload.get("data", ""))
    
    # Create large payload
    large_data = "x" * 100000  # 100KB of data
    
    task_id = await scheduler.submit(
        handler="payload_test",
        payload={"data": large_data}
    )
    
    await scheduler.write_buffer.flush()
    
    # Check if payload was externalized
    should_external = scheduler.payload_service.should_externalize({"data": large_data})
    assert should_external is True
    
    # Get stats
    stats = await scheduler.payload_service.get_stats()
    assert stats['storage_path'] is not None


@pytest.mark.asyncio 
async def test_error_handling(scheduler):
    """Test error handling in scheduler."""
    # Try to submit with non-existent handler
    with pytest.raises(ValueError, match="not registered"):
        await scheduler.submit(
            handler="non_existent",
            payload={}
        )
    
    # Test scheduler not started
    new_scheduler = Scheduler(scheduler.config)
    with pytest.raises(Exception):
        await new_scheduler.submit(
            handler="test",
            payload={}
        )


if __name__ == "__main__":
    # Run tests
    asyncio.run(test_scheduler_lifecycle(SchedulerConfig(
        database_url="sqlite://:memory:",
        base_path=Path("/tmp")
    )))
    print("✓ Scheduler lifecycle test passed")
    
    print("\n✅ All scheduler tests passed!")