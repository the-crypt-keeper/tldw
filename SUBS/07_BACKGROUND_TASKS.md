# Background Tasks - Periodic Checking Design

## Overview

This document details the design and implementation of the background task system for periodic subscription checking. The system must be reliable, scalable, and efficient while integrating seamlessly with the existing tldw_server architecture.

## Architecture Options

### Option 1: Embedded Scheduler (Recommended)

Run the scheduler within the FastAPI application using `asyncio` and `apscheduler`.

**Pros:**
- Simple deployment (single process)
- Shares database connections
- Easy state management
- Minimal infrastructure

**Cons:**
- Restarts affect scheduler
- Limited horizontal scaling
- Coupled to API process

**Implementation:**
```python
# Location: /app/services/subscription_scheduler.py

from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.executors.asyncio import AsyncIOExecutor
from contextlib import asynccontextmanager
from fastapi import FastAPI

class SubscriptionSchedulerService:
    def __init__(self):
        self.scheduler = AsyncIOScheduler(
            executors={
                'default': AsyncIOExecutor(),
            },
            job_defaults={
                'coalesce': True,
                'max_instances': 1,
                'misfire_grace_time': 300  # 5 minutes
            }
        )
        self.is_running = False
        
    async def start(self):
        """Start the scheduler and load jobs"""
        if not self.is_running:
            self.scheduler.start()
            self.is_running = True
            await self._load_subscription_jobs()
            logger.info("Subscription scheduler started")
            
    async def shutdown(self):
        """Gracefully shutdown scheduler"""
        if self.is_running:
            self.scheduler.shutdown(wait=True)
            self.is_running = False
            logger.info("Subscription scheduler stopped")

# FastAPI lifespan integration
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    scheduler = SubscriptionSchedulerService()
    await scheduler.start()
    app.state.scheduler = scheduler
    
    yield
    
    # Shutdown
    await scheduler.shutdown()

app = FastAPI(lifespan=lifespan)
```

### Option 2: Separate Worker Process

Run scheduler as a separate process/service.

**Pros:**
- Independent scaling
- Fault isolation
- Can use different technologies

**Cons:**
- Complex deployment
- Inter-process communication
- State synchronization

### Option 3: External Cron

Use system cron or cloud scheduler to trigger checks.

**Pros:**
- Battle-tested reliability
- No custom scheduler code
- Easy monitoring

**Cons:**
- Less flexible
- Requires API endpoints
- No dynamic scheduling

## Detailed Implementation

### Core Scheduler Class

```python
# Location: /app/services/subscription_scheduler.py

import asyncio
from typing import Dict, Optional, Set
from datetime import datetime, timedelta
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.interval import IntervalTrigger
from apscheduler.triggers.cron import CronTrigger
from apscheduler.job import Job

class SubscriptionScheduler:
    """
    Manages periodic checking of subscriptions with dynamic scheduling
    """
    
    def __init__(self, subscription_service, max_concurrent_checks: int = 5):
        self.subscription_service = subscription_service
        self.scheduler = AsyncIOScheduler()
        self.max_concurrent = max_concurrent_checks
        self.check_semaphore = asyncio.Semaphore(max_concurrent_checks)
        self.active_checks: Set[int] = set()
        self.job_mapping: Dict[int, str] = {}  # subscription_id -> job_id
        
    async def start(self):
        """Initialize and start the scheduler"""
        # Configure scheduler
        self.scheduler.configure(
            job_defaults={
                'coalesce': True,
                'max_instances': 1,
                'misfire_grace_time': 300
            },
            timezone='UTC'
        )
        
        # Add job listeners
        self.scheduler.add_listener(
            self._job_executed,
            EVENT_JOB_EXECUTED | EVENT_JOB_ERROR
        )
        
        # Start scheduler
        self.scheduler.start()
        
        # Load existing subscriptions
        await self._load_all_subscriptions()
        
        # Schedule periodic maintenance
        self.scheduler.add_job(
            self._maintenance_task,
            'interval',
            hours=1,
            id='maintenance',
            replace_existing=True
        )
        
        logger.info(f"Scheduler started with {len(self.job_mapping)} subscriptions")
    
    async def _load_all_subscriptions(self):
        """Load and schedule all active subscriptions"""
        try:
            subscriptions = await self.subscription_service.get_active_subscriptions()
            
            for subscription in subscriptions:
                await self.schedule_subscription(subscription)
                
            logger.info(f"Loaded {len(subscriptions)} active subscriptions")
            
        except Exception as e:
            logger.error(f"Failed to load subscriptions: {e}")
    
    async def schedule_subscription(self, subscription):
        """Schedule or reschedule a subscription check"""
        job_id = f"subscription_{subscription.id}"
        
        # Remove existing job if any
        if subscription.id in self.job_mapping:
            self.remove_subscription(subscription.id)
        
        # Skip if not active
        if not subscription.is_active:
            return
        
        # Create trigger based on check interval
        trigger = IntervalTrigger(
            seconds=subscription.check_interval,
            start_date=self._calculate_next_check(subscription)
        )
        
        # Add job
        job = self.scheduler.add_job(
            self._check_subscription_wrapper,
            trigger,
            args=[subscription.id],
            id=job_id,
            name=f"Check {subscription.name}",
            replace_existing=True
        )
        
        self.job_mapping[subscription.id] = job_id
        
        logger.debug(f"Scheduled subscription {subscription.id} with interval {subscription.check_interval}s")
    
    def remove_subscription(self, subscription_id: int):
        """Remove a subscription from the scheduler"""
        job_id = self.job_mapping.get(subscription_id)
        if job_id:
            try:
                self.scheduler.remove_job(job_id)
                del self.job_mapping[subscription_id]
                logger.debug(f"Removed subscription {subscription_id} from scheduler")
            except Exception as e:
                logger.error(f"Failed to remove job {job_id}: {e}")
    
    async def _check_subscription_wrapper(self, subscription_id: int):
        """Wrapper for subscription checking with concurrency control"""
        # Skip if already checking
        if subscription_id in self.active_checks:
            logger.warning(f"Subscription {subscription_id} check already in progress")
            return
        
        # Acquire semaphore for concurrency control
        async with self.check_semaphore:
            self.active_checks.add(subscription_id)
            try:
                await self._check_subscription(subscription_id)
            finally:
                self.active_checks.discard(subscription_id)
    
    async def _check_subscription(self, subscription_id: int):
        """Perform the actual subscription check"""
        start_time = datetime.utcnow()
        check_id = None
        
        try:
            # Get subscription details
            subscription = await self.subscription_service.get_subscription(subscription_id)
            if not subscription:
                logger.error(f"Subscription {subscription_id} not found")
                self.remove_subscription(subscription_id)
                return
            
            # Create check record
            check_id = await self.subscription_service.create_check_record(
                subscription_id,
                status='started'
            )
            
            # Perform the check
            logger.info(f"Checking subscription {subscription_id}: {subscription.name}")
            result = await self.subscription_service.check_subscription(subscription_id)
            
            # Update check record
            duration_ms = int((datetime.utcnow() - start_time).total_seconds() * 1000)
            await self.subscription_service.update_check_record(
                check_id,
                status='success',
                items_found=result.total_items,
                new_items=result.new_items,
                duration_ms=duration_ms
            )
            
            # Update subscription last_checked
            await self.subscription_service.update_last_checked(subscription_id)
            
            logger.info(
                f"Subscription {subscription_id} check completed: "
                f"{result.new_items} new items found"
            )
            
            # Handle auto-import if configured
            if subscription.auto_import and result.new_items > 0:
                await self._handle_auto_import(subscription_id, result.new_item_ids)
            
        except Exception as e:
            logger.error(f"Error checking subscription {subscription_id}: {e}")
            
            # Update check record with error
            if check_id:
                await self.subscription_service.update_check_record(
                    check_id,
                    status='failed',
                    error_message=str(e),
                    duration_ms=int((datetime.utcnow() - start_time).total_seconds() * 1000)
                )
            
            # Handle persistent failures
            await self._handle_check_failure(subscription_id, e)
    
    async def _handle_auto_import(self, subscription_id: int, item_ids: List[int]):
        """Handle automatic import of new items"""
        try:
            logger.info(f"Auto-importing {len(item_ids)} items for subscription {subscription_id}")
            
            # Import items in batches
            batch_size = 10
            for i in range(0, len(item_ids), batch_size):
                batch = item_ids[i:i + batch_size]
                await self.subscription_service.import_items(batch, auto=True)
                
        except Exception as e:
            logger.error(f"Auto-import failed for subscription {subscription_id}: {e}")
    
    async def _handle_check_failure(self, subscription_id: int, error: Exception):
        """Handle subscription check failures"""
        try:
            # Increment failure counter
            subscription = await self.subscription_service.get_subscription(subscription_id)
            failures = subscription.consecutive_failures + 1
            
            await self.subscription_service.update_subscription(
                subscription_id,
                consecutive_failures=failures
            )
            
            # Implement exponential backoff
            if failures >= 3:
                # Double the check interval after 3 failures
                new_interval = min(
                    subscription.check_interval * 2,
                    86400  # Max 24 hours
                )
                
                logger.warning(
                    f"Subscription {subscription_id} has failed {failures} times. "
                    f"Increasing interval to {new_interval}s"
                )
                
                await self.subscription_service.update_subscription(
                    subscription_id,
                    check_interval=new_interval
                )
                
                # Reschedule with new interval
                subscription.check_interval = new_interval
                await self.schedule_subscription(subscription)
            
            # Disable after too many failures
            if failures >= 10:
                logger.error(
                    f"Subscription {subscription_id} has failed {failures} times. Disabling."
                )
                
                await self.subscription_service.update_subscription(
                    subscription_id,
                    is_active=False
                )
                
                self.remove_subscription(subscription_id)
                
        except Exception as e:
            logger.error(f"Failed to handle check failure: {e}")
    
    def _calculate_next_check(self, subscription) -> datetime:
        """Calculate when the next check should occur"""
        if not subscription.last_checked:
            # Never checked, start immediately
            return datetime.utcnow()
        
        # Calculate based on last check + interval
        next_check = subscription.last_checked + timedelta(seconds=subscription.check_interval)
        
        # If overdue, start soon but add jitter to avoid thundering herd
        if next_check < datetime.utcnow():
            jitter = asyncio.create_task(asyncio.sleep(0))  # 0-1 second random delay
            return datetime.utcnow() + timedelta(seconds=hash(subscription.id) % 60)
        
        return next_check
    
    async def _maintenance_task(self):
        """Periodic maintenance and health checks"""
        try:
            logger.info("Running scheduler maintenance")
            
            # Check for orphaned jobs
            job_ids = {job.id for job in self.scheduler.get_jobs()}
            for sub_id, job_id in list(self.job_mapping.items()):
                if job_id not in job_ids:
                    logger.warning(f"Orphaned job mapping for subscription {sub_id}")
                    del self.job_mapping[sub_id]
            
            # Reload any missing subscriptions
            active_subs = await self.subscription_service.get_active_subscriptions()
            for sub in active_subs:
                if sub.id not in self.job_mapping:
                    logger.info(f"Scheduling missing subscription {sub.id}")
                    await self.schedule_subscription(sub)
            
            # Log statistics
            stats = {
                'scheduled_jobs': len(self.scheduler.get_jobs()),
                'active_checks': len(self.active_checks),
                'job_mappings': len(self.job_mapping)
            }
            logger.info(f"Scheduler stats: {stats}")
            
        except Exception as e:
            logger.error(f"Maintenance task failed: {e}")
    
    def get_next_check_time(self, subscription_id: int) -> Optional[datetime]:
        """Get the next scheduled check time for a subscription"""
        job_id = self.job_mapping.get(subscription_id)
        if job_id:
            job = self.scheduler.get_job(job_id)
            if job:
                return job.next_run_time
        return None
    
    def get_scheduler_status(self) -> Dict:
        """Get current scheduler status and statistics"""
        jobs = self.scheduler.get_jobs()
        
        return {
            'running': self.scheduler.running,
            'total_jobs': len(jobs),
            'active_checks': len(self.active_checks),
            'pending_jobs': len([j for j in jobs if j.pending]),
            'next_run': min((j.next_run_time for j in jobs), default=None),
            'concurrent_limit': self.max_concurrent
        }
```

### Job Persistence

```python
# Location: /app/services/subscription_job_store.py

from apscheduler.jobstores.sqlalchemy import SQLAlchemyJobStore
from apscheduler.jobstores.memory import MemoryJobStore

class PersistentJobStore:
    """
    Persistent job storage for scheduler recovery
    """
    
    @staticmethod
    def create_job_store(db_path: str, persistent: bool = True):
        """Create appropriate job store"""
        if persistent:
            # Use SQLite for persistence
            return SQLAlchemyJobStore(
                url=f'sqlite:///{db_path}/scheduler_jobs.db',
                tablename='subscription_jobs'
            )
        else:
            # Use memory store for development/testing
            return MemoryJobStore()
    
    @staticmethod
    def migrate_job_store(old_store, new_store):
        """Migrate jobs between stores"""
        jobs = old_store.get_all_jobs()
        for job in jobs:
            new_store.add_job(job)
```

### Check Coordination

```python
# Location: /app/services/subscription_check_coordinator.py

from typing import List, Dict, Set
import asyncio
from collections import defaultdict

class CheckCoordinator:
    """
    Coordinates subscription checks to optimize resource usage
    """
    
    def __init__(self, max_concurrent_domains: int = 3):
        self.domain_locks: Dict[str, asyncio.Semaphore] = defaultdict(
            lambda: asyncio.Semaphore(1)
        )
        self.global_semaphore = asyncio.Semaphore(max_concurrent_domains)
        self.check_queue: asyncio.Queue = asyncio.Queue()
        self.workers: List[asyncio.Task] = []
        
    async def start_workers(self, num_workers: int = 5):
        """Start worker tasks for processing checks"""
        for i in range(num_workers):
            worker = asyncio.create_task(self._worker(f"worker-{i}"))
            self.workers.append(worker)
            
    async def stop_workers(self):
        """Stop all worker tasks"""
        # Cancel all workers
        for worker in self.workers:
            worker.cancel()
            
        # Wait for cancellation
        await asyncio.gather(*self.workers, return_exceptions=True)
        self.workers.clear()
    
    async def _worker(self, name: str):
        """Worker task that processes checks from queue"""
        logger.info(f"Check worker {name} started")
        
        try:
            while True:
                check_task = await self.check_queue.get()
                
                try:
                    await self._process_check(check_task)
                except Exception as e:
                    logger.error(f"Worker {name} error: {e}")
                finally:
                    self.check_queue.task_done()
                    
        except asyncio.CancelledError:
            logger.info(f"Check worker {name} stopped")
            raise
    
    async def _process_check(self, check_task: Dict):
        """Process a single check with domain limiting"""
        subscription = check_task['subscription']
        domain = self._extract_domain(subscription.url)
        
        # Acquire global semaphore
        async with self.global_semaphore:
            # Acquire domain-specific lock
            async with self.domain_locks[domain]:
                # Add delay between requests to same domain
                await asyncio.sleep(1.0)
                
                # Perform the actual check
                await check_task['callback'](subscription.id)
    
    def _extract_domain(self, url: str) -> str:
        """Extract domain from URL for rate limiting"""
        from urllib.parse import urlparse
        return urlparse(url).netloc
    
    async def queue_check(self, subscription, callback):
        """Queue a subscription check"""
        await self.check_queue.put({
            'subscription': subscription,
            'callback': callback
        })
```

### Monitoring and Metrics

```python
# Location: /app/services/subscription_metrics.py

from dataclasses import dataclass
from typing import Dict, List
from datetime import datetime, timedelta
import asyncio

@dataclass
class CheckMetrics:
    subscription_id: int
    duration_ms: int
    items_found: int
    new_items: int
    success: bool
    timestamp: datetime

class SchedulerMetrics:
    """
    Collect and report scheduler metrics
    """
    
    def __init__(self, window_size: int = 3600):  # 1 hour window
        self.window_size = window_size
        self.metrics: List[CheckMetrics] = []
        self.lock = asyncio.Lock()
        
    async def record_check(self, metrics: CheckMetrics):
        """Record check metrics"""
        async with self.lock:
            self.metrics.append(metrics)
            # Clean old metrics
            cutoff = datetime.utcnow() - timedelta(seconds=self.window_size)
            self.metrics = [m for m in self.metrics if m.timestamp > cutoff]
    
    async def get_statistics(self) -> Dict:
        """Get current statistics"""
        async with self.lock:
            if not self.metrics:
                return {
                    'total_checks': 0,
                    'successful_checks': 0,
                    'failed_checks': 0,
                    'average_duration_ms': 0,
                    'total_items_found': 0,
                    'total_new_items': 0,
                    'checks_per_minute': 0
                }
            
            successful = [m for m in self.metrics if m.success]
            failed = [m for m in self.metrics if not m.success]
            
            # Calculate time range
            time_range = (
                datetime.utcnow() - min(m.timestamp for m in self.metrics)
            ).total_seconds() / 60  # minutes
            
            return {
                'total_checks': len(self.metrics),
                'successful_checks': len(successful),
                'failed_checks': len(failed),
                'average_duration_ms': (
                    sum(m.duration_ms for m in successful) / len(successful)
                    if successful else 0
                ),
                'total_items_found': sum(m.items_found for m in successful),
                'total_new_items': sum(m.new_items for m in successful),
                'checks_per_minute': len(self.metrics) / max(time_range, 1),
                'success_rate': len(successful) / len(self.metrics) * 100
            }
```

### Advanced Scheduling Features

```python
# Location: /app/services/subscription_advanced_scheduler.py

class AdvancedSchedulingFeatures:
    """
    Advanced scheduling capabilities
    """
    
    @staticmethod
    def create_cron_trigger(expression: str) -> CronTrigger:
        """
        Create cron trigger from expression
        Examples:
        - "0 */6 * * *" - Every 6 hours
        - "0 9 * * MON" - Every Monday at 9 AM
        """
        parts = expression.split()
        return CronTrigger(
            minute=parts[0],
            hour=parts[1],
            day=parts[2],
            month=parts[3],
            day_of_week=parts[4]
        )
    
    @staticmethod
    def calculate_adaptive_interval(
        subscription,
        recent_checks: List[Dict]
    ) -> int:
        """
        Calculate adaptive check interval based on activity
        """
        if not recent_checks:
            return subscription.check_interval
        
        # Calculate average new items per check
        avg_new_items = sum(
            c['new_items'] for c in recent_checks
        ) / len(recent_checks)
        
        # Adjust interval based on activity
        if avg_new_items > 10:
            # Very active, check more frequently
            return max(subscription.check_interval // 2, 900)  # Min 15 min
        elif avg_new_items < 1:
            # Low activity, check less frequently
            return min(subscription.check_interval * 2, 86400)  # Max 24 hours
        else:
            return subscription.check_interval
    
    @staticmethod
    def distribute_check_times(
        subscriptions: List,
        time_window: int = 3600
    ) -> Dict[int, datetime]:
        """
        Distribute subscription checks evenly across time window
        """
        if not subscriptions:
            return {}
        
        interval = time_window / len(subscriptions)
        base_time = datetime.utcnow()
        
        return {
            sub.id: base_time + timedelta(seconds=i * interval)
            for i, sub in enumerate(subscriptions)
        }
```

## Integration with FastAPI

### API Endpoints for Scheduler Control

```python
# Location: /app/api/v1/endpoints/scheduler.py

from fastapi import APIRouter, Depends, HTTPException
from app.services.subscription_scheduler import SubscriptionScheduler

router = APIRouter(prefix="/scheduler", tags=["scheduler"])

@router.get("/status")
async def get_scheduler_status(
    scheduler: SubscriptionScheduler = Depends(get_scheduler)
):
    """Get current scheduler status"""
    return scheduler.get_scheduler_status()

@router.post("/pause")
async def pause_scheduler(
    scheduler: SubscriptionScheduler = Depends(get_scheduler)
):
    """Pause all scheduled checks"""
    scheduler.scheduler.pause()
    return {"status": "paused"}

@router.post("/resume")
async def resume_scheduler(
    scheduler: SubscriptionScheduler = Depends(get_scheduler)
):
    """Resume scheduled checks"""
    scheduler.scheduler.resume()
    return {"status": "resumed"}

@router.get("/jobs")
async def list_scheduled_jobs(
    scheduler: SubscriptionScheduler = Depends(get_scheduler)
):
    """List all scheduled jobs"""
    jobs = scheduler.scheduler.get_jobs()
    return {
        "jobs": [
            {
                "id": job.id,
                "name": job.name,
                "next_run": job.next_run_time.isoformat() if job.next_run_time else None,
                "pending": job.pending
            }
            for job in jobs
        ]
    }

@router.post("/subscriptions/{subscription_id}/reschedule")
async def reschedule_subscription(
    subscription_id: int,
    scheduler: SubscriptionScheduler = Depends(get_scheduler)
):
    """Manually reschedule a subscription"""
    subscription = await get_subscription(subscription_id)
    await scheduler.schedule_subscription(subscription)
    return {"status": "rescheduled"}
```

## Deployment Considerations

### Production Configuration

```python
# Location: /app/core/config.py

class SchedulerConfig:
    # Scheduler settings
    MAX_CONCURRENT_CHECKS = int(os.getenv("MAX_CONCURRENT_CHECKS", "5"))
    MAX_CONCURRENT_DOMAINS = int(os.getenv("MAX_CONCURRENT_DOMAINS", "3"))
    
    # Job settings
    JOB_MISFIRE_GRACE_TIME = int(os.getenv("JOB_MISFIRE_GRACE_TIME", "300"))
    JOB_MAX_INSTANCES = int(os.getenv("JOB_MAX_INSTANCES", "1"))
    JOB_COALESCE = os.getenv("JOB_COALESCE", "true").lower() == "true"
    
    # Performance settings
    DOMAIN_REQUEST_DELAY = float(os.getenv("DOMAIN_REQUEST_DELAY", "1.0"))
    CHECK_TIMEOUT = int(os.getenv("CHECK_TIMEOUT", "30"))
    
    # Persistence
    PERSIST_JOBS = os.getenv("PERSIST_JOBS", "true").lower() == "true"
    JOB_STORE_PATH = os.getenv("JOB_STORE_PATH", "./data/scheduler")
```

### Monitoring and Alerting

```yaml
# prometheus metrics
subscription_checks_total:
  type: counter
  help: Total number of subscription checks performed

subscription_check_duration_seconds:
  type: histogram
  help: Duration of subscription checks in seconds

active_subscription_checks:
  type: gauge
  help: Number of currently active subscription checks

scheduled_jobs_total:
  type: gauge
  help: Total number of scheduled jobs
```

### Graceful Shutdown

```python
async def shutdown_handler():
    """Graceful shutdown of scheduler"""
    logger.info("Shutting down scheduler...")
    
    # Stop accepting new jobs
    scheduler.scheduler.pause()
    
    # Wait for active checks to complete
    timeout = 30
    start = time.time()
    while scheduler.active_checks and time.time() - start < timeout:
        await asyncio.sleep(1)
    
    # Force shutdown if needed
    if scheduler.active_checks:
        logger.warning(f"Forcing shutdown with {len(scheduler.active_checks)} active checks")
    
    # Shutdown scheduler
    scheduler.scheduler.shutdown(wait=False)
```