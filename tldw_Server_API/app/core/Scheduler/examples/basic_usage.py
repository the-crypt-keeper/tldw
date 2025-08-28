"""
Basic usage example for the Scheduler module.
"""

import asyncio
from pathlib import Path
from datetime import datetime, timedelta

from tldw_Server_API.app.core.Scheduler import (
    Scheduler,
    task,
    SchedulerConfig,
    TaskPriority
)


# Register task handlers using decorator
@task(max_retries=3, timeout=60)
async def send_email(data):
    """Send an email notification."""
    print(f"Sending email to {data['to']}: {data['subject']}")
    await asyncio.sleep(1)  # Simulate sending
    return {"sent": True, "timestamp": datetime.utcnow().isoformat()}


@task(max_retries=5, timeout=300, queue="heavy")
async def process_video(video_info):
    """Process a video file."""
    print(f"Processing video: {video_info['url']}")
    await asyncio.sleep(2)  # Simulate processing
    return {
        "status": "completed",
        "duration": video_info.get("duration", 0),
        "processed_at": datetime.utcnow().isoformat()
    }


@task(queue="analytics")
async def analyze_metrics(metrics_data):
    """Analyze metrics data."""
    print(f"Analyzing metrics for period: {metrics_data['period']}")
    total = sum(metrics_data.get("values", []))
    return {
        "total": total,
        "average": total / len(metrics_data.get("values", [1]))
    }


async def main():
    """Main example function."""
    
    # Configure scheduler
    config = SchedulerConfig(
        database_url="sqlite:///scheduler_example.db",
        base_path=Path("/tmp/scheduler_example"),
        min_workers=2,
        max_workers=10,
        write_buffer_size=100,
        write_buffer_flush_interval=1.0
    )
    
    # Create and start scheduler
    async with Scheduler(config) as scheduler:
        print("Scheduler started!")
        
        # Submit a simple task
        email_task = await scheduler.submit(
            handler="__main__.send_email",
            payload={
                "to": "user@example.com",
                "subject": "Welcome!",
                "body": "Thanks for signing up"
            },
            priority=TaskPriority.HIGH.value
        )
        print(f"Submitted email task: {email_task}")
        
        # Submit a task with dependencies
        video_task = await scheduler.submit(
            handler="__main__.process_video",
            payload={
                "url": "https://example.com/video.mp4",
                "duration": 120
            },
            queue_name="heavy"
        )
        
        # Task that depends on video processing
        analytics_task = await scheduler.submit(
            handler="__main__.analyze_metrics",
            payload={
                "period": "daily",
                "values": [10, 20, 30, 40, 50]
            },
            queue_name="analytics",
            depends_on=[video_task]  # Will only run after video_task completes
        )
        
        # Submit with idempotency key (prevents duplicates)
        dedup_task = await scheduler.submit(
            handler="__main__.send_email",
            payload={"to": "admin@example.com", "subject": "Report"},
            idempotency_key="daily-report-2024-01-01"
        )
        
        # Try to submit duplicate (will return same task ID)
        duplicate = await scheduler.submit(
            handler="__main__.send_email",
            payload={"to": "different@example.com", "subject": "Different"},
            idempotency_key="daily-report-2024-01-01"  # Same key!
        )
        
        assert dedup_task == duplicate
        print(f"Idempotency works: {dedup_task} == {duplicate}")
        
        # Schedule a task for the future
        future_time = datetime.utcnow() + timedelta(seconds=5)
        scheduled_task = await scheduler.submit(
            handler="__main__.send_email",
            payload={"to": "future@example.com", "subject": "Scheduled"},
            metadata={"scheduled_at": future_time.isoformat()}
        )
        
        # Wait for a task to complete
        print(f"Waiting for email task {email_task} to complete...")
        result = await scheduler.wait_for_task(email_task, timeout=10)
        if result:
            print(f"Email task completed with status: {result.status}")
            print(f"Result: {result.result}")
        
        # Get queue status
        status = await scheduler.get_queue_status()
        print(f"Queue status: {status}")
        
        # Get scheduler status
        scheduler_status = scheduler.get_status()
        print(f"Scheduler status: {scheduler_status}")
        
        # Scale workers for heavy queue
        if scheduler.worker_pool:
            new_count = await scheduler.scale_workers(5, "heavy")
            print(f"Scaled heavy queue to {new_count} workers")


async def example_with_batch_processing():
    """Example of batch task processing."""
    
    config = SchedulerConfig(
        database_url="sqlite:///batch_example.db",
        base_path=Path("/tmp/batch_example")
    )
    
    async with Scheduler(config) as scheduler:
        # Submit a batch of tasks
        tasks = [
            {
                "handler": "__main__.send_email",
                "payload": {
                    "to": f"user{i}@example.com",
                    "subject": f"Message {i}"
                },
                "priority": TaskPriority.NORMAL.value if i % 2 == 0 else TaskPriority.LOW.value
            }
            for i in range(10)
        ]
        
        task_ids = await scheduler.submit_batch(tasks)
        print(f"Submitted {len(task_ids)} tasks in batch")
        
        # Wait for all to complete
        results = []
        for task_id in task_ids:
            result = await scheduler.wait_for_task(task_id, timeout=30)
            if result:
                results.append(result)
        
        print(f"Completed {len(results)} out of {len(task_ids)} tasks")


async def example_with_error_handling():
    """Example with error handling and retries."""
    
    # Task that might fail
    @task(max_retries=3, timeout=10)
    async def unreliable_task(data):
        """Task that fails sometimes."""
        import random
        if random.random() < 0.5:
            raise Exception("Random failure!")
        return {"success": True, "data": data}
    
    config = SchedulerConfig(
        database_url="sqlite:///error_example.db",
        base_path=Path("/tmp/error_example")
    )
    
    async with Scheduler(config) as scheduler:
        # Submit task that might fail
        task_id = await scheduler.submit(
            handler="__main__.unreliable_task",
            payload={"important": "data"}
        )
        
        # Wait and check result
        result = await scheduler.wait_for_task(task_id, timeout=60)
        if result:
            if result.status.value == "completed":
                print(f"Task succeeded: {result.result}")
            elif result.status.value == "failed":
                print(f"Task failed after retries: {result.error}")
        else:
            print("Task timed out")


if __name__ == "__main__":
    print("=" * 60)
    print("SCHEDULER EXAMPLE")
    print("=" * 60)
    
    # Run main example
    asyncio.run(main())
    
    print("\n" + "=" * 60)
    print("BATCH PROCESSING EXAMPLE")
    print("=" * 60)
    
    # Run batch example
    asyncio.run(example_with_batch_processing())
    
    print("\n" + "=" * 60)
    print("ERROR HANDLING EXAMPLE")
    print("=" * 60)
    
    # Run error handling example
    asyncio.run(example_with_error_handling())