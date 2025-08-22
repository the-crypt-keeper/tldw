# event_broadcaster.py
# Event broadcasting system for Prompt Studio real-time updates

import json
import asyncio
from typing import Dict, Any, Optional, List, Set
from datetime import datetime
from enum import Enum
from loguru import logger

from tldw_Server_API.app.core.DB_Management.PromptStudioDatabase import PromptStudioDatabase

########################################################################################################################
# Event Types

class EventType(str, Enum):
    """Types of events that can be broadcast."""
    
    # Job events
    JOB_CREATED = "job_created"
    JOB_STARTED = "job_started"
    JOB_PROGRESS = "job_progress"
    JOB_COMPLETED = "job_completed"
    JOB_FAILED = "job_failed"
    JOB_CANCELLED = "job_cancelled"
    JOB_RETRYING = "job_retrying"
    
    # Evaluation events
    EVALUATION_STARTED = "evaluation_started"
    EVALUATION_PROGRESS = "evaluation_progress"
    EVALUATION_COMPLETED = "evaluation_completed"
    
    # Optimization events
    OPTIMIZATION_STARTED = "optimization_started"
    OPTIMIZATION_ITERATION = "optimization_iteration"
    OPTIMIZATION_COMPLETED = "optimization_completed"
    
    # Test case events
    TEST_CASE_CREATED = "test_case_created"
    TEST_CASE_UPDATED = "test_case_updated"
    TEST_CASE_DELETED = "test_case_deleted"
    TEST_CASES_GENERATED = "test_cases_generated"
    
    # Project events
    PROJECT_UPDATED = "project_updated"
    PROJECT_DELETED = "project_deleted"
    
    # System events
    SYSTEM_MESSAGE = "system_message"
    SYSTEM_WARNING = "system_warning"
    SYSTEM_ERROR = "system_error"

########################################################################################################################
# Event Broadcaster

class EventBroadcaster:
    """Broadcasts events to connected WebSocket clients."""
    
    def __init__(self, connection_manager, db: PromptStudioDatabase):
        """
        Initialize EventBroadcaster.
        
        Args:
            connection_manager: WebSocket connection manager
            db: Database instance
        """
        self.connection_manager = connection_manager
        self.db = db
        self.client_id = db.client_id
        
        # Track subscriptions
        self.subscriptions: Dict[str, Set[str]] = {}
        
        # Event queue for async processing
        self.event_queue: asyncio.Queue = asyncio.Queue()
        self._processing = False
        self._process_task = None
    
    ####################################################################################################################
    # Broadcasting Methods
    
    async def broadcast_event(self, event_type: EventType, data: Dict[str, Any],
                             client_ids: Optional[List[str]] = None,
                             project_id: Optional[int] = None):
        """
        Broadcast an event to clients.
        
        Args:
            event_type: Type of event
            data: Event data
            client_ids: Specific clients to broadcast to (None = all)
            project_id: Associated project ID
        """
        event = {
            "type": event_type.value,
            "data": data,
            "timestamp": datetime.utcnow().isoformat(),
            "project_id": project_id
        }
        
        # Log event to database
        await self._log_event(event_type, data, project_id)
        
        # Convert to JSON
        message = json.dumps(event)
        
        # Broadcast to specified clients or all
        if client_ids:
            for client_id in client_ids:
                await self.connection_manager.broadcast_to_client(client_id, message)
        else:
            await self.connection_manager.broadcast_to_all(message)
        
        logger.debug(f"Broadcast {event_type.value} to {client_ids or 'all'}")
    
    async def broadcast_job_event(self, job_id: int, event_type: EventType,
                                 additional_data: Optional[Dict] = None):
        """
        Broadcast a job-related event.
        
        Args:
            job_id: Job ID
            event_type: Event type
            additional_data: Additional event data
        """
        # Get job details
        from .job_manager import JobManager
        job_manager = JobManager(self.db)
        job = job_manager.get_job(job_id)
        
        if not job:
            logger.warning(f"Job {job_id} not found for event broadcast")
            return
        
        # Prepare event data
        data = {
            "job_id": job_id,
            "job_type": job["job_type"],
            "entity_id": job["entity_id"],
            "status": job["status"],
            "progress": job.get("progress", 0)
        }
        
        if additional_data:
            data.update(additional_data)
        
        # Get project ID from entity
        project_id = await self._get_project_for_job(job)
        
        # Broadcast event
        await self.broadcast_event(
            event_type=event_type,
            data=data,
            project_id=project_id
        )
    
    async def broadcast_progress(self, job_id: int, progress: float,
                                message: Optional[str] = None):
        """
        Broadcast job progress update.
        
        Args:
            job_id: Job ID
            progress: Progress percentage (0-100)
            message: Optional progress message
        """
        data = {
            "progress": progress,
            "message": message
        }
        
        await self.broadcast_job_event(
            job_id=job_id,
            event_type=EventType.JOB_PROGRESS,
            additional_data=data
        )
    
    async def broadcast_evaluation_progress(self, evaluation_id: int,
                                          tests_completed: int, total_tests: int,
                                          current_test: Optional[str] = None):
        """
        Broadcast evaluation progress.
        
        Args:
            evaluation_id: Evaluation ID
            tests_completed: Number of tests completed
            total_tests: Total number of tests
            current_test: Currently running test name
        """
        progress = (tests_completed / total_tests * 100) if total_tests > 0 else 0
        
        data = {
            "evaluation_id": evaluation_id,
            "tests_completed": tests_completed,
            "total_tests": total_tests,
            "progress": progress,
            "current_test": current_test
        }
        
        await self.broadcast_event(
            event_type=EventType.EVALUATION_PROGRESS,
            data=data
        )
    
    async def broadcast_optimization_iteration(self, optimization_id: int,
                                              iteration: int, max_iterations: int,
                                              current_metric: float,
                                              best_metric: float):
        """
        Broadcast optimization iteration update.
        
        Args:
            optimization_id: Optimization ID
            iteration: Current iteration number
            max_iterations: Maximum iterations
            current_metric: Current iteration metric
            best_metric: Best metric so far
        """
        data = {
            "optimization_id": optimization_id,
            "iteration": iteration,
            "max_iterations": max_iterations,
            "current_metric": current_metric,
            "best_metric": best_metric,
            "progress": (iteration / max_iterations * 100) if max_iterations > 0 else 0
        }
        
        await self.broadcast_event(
            event_type=EventType.OPTIMIZATION_ITERATION,
            data=data
        )
    
    ####################################################################################################################
    # Subscription Management
    
    def subscribe(self, client_id: str, entity_type: str, entity_id: int):
        """
        Subscribe a client to entity updates.
        
        Args:
            client_id: Client ID
            entity_type: Type of entity (job, evaluation, etc.)
            entity_id: Entity ID
        """
        key = f"{entity_type}:{entity_id}"
        
        if key not in self.subscriptions:
            self.subscriptions[key] = set()
        
        self.subscriptions[key].add(client_id)
        logger.debug(f"Client {client_id} subscribed to {key}")
    
    def unsubscribe(self, client_id: str, entity_type: str, entity_id: int):
        """
        Unsubscribe a client from entity updates.
        
        Args:
            client_id: Client ID
            entity_type: Type of entity
            entity_id: Entity ID
        """
        key = f"{entity_type}:{entity_id}"
        
        if key in self.subscriptions:
            self.subscriptions[key].discard(client_id)
            
            # Clean up empty subscriptions
            if not self.subscriptions[key]:
                del self.subscriptions[key]
        
        logger.debug(f"Client {client_id} unsubscribed from {key}")
    
    def get_subscribers(self, entity_type: str, entity_id: int) -> Set[str]:
        """
        Get subscribers for an entity.
        
        Args:
            entity_type: Type of entity
            entity_id: Entity ID
            
        Returns:
            Set of client IDs
        """
        key = f"{entity_type}:{entity_id}"
        return self.subscriptions.get(key, set())
    
    ####################################################################################################################
    # Event Processing
    
    async def start_processing(self):
        """Start processing events from the queue."""
        if self._processing:
            logger.warning("Event processing already running")
            return
        
        self._processing = True
        self._process_task = asyncio.create_task(self._process_events())
        logger.info("Started event processor")
    
    async def stop_processing(self):
        """Stop processing events."""
        self._processing = False
        
        if self._process_task:
            self._process_task.cancel()
            try:
                await self._process_task
            except asyncio.CancelledError:
                pass
        
        logger.info("Stopped event processor")
    
    async def _process_events(self):
        """Process events from the queue."""
        while self._processing:
            try:
                # Get event from queue with timeout
                event = await asyncio.wait_for(
                    self.event_queue.get(),
                    timeout=1.0
                )
                
                # Process event
                await self._handle_queued_event(event)
                
            except asyncio.TimeoutError:
                continue
            except Exception as e:
                logger.error(f"Error processing event: {e}")
    
    async def _handle_queued_event(self, event: Dict[str, Any]):
        """
        Handle a queued event.
        
        Args:
            event: Event data
        """
        event_type = event.get("type")
        data = event.get("data")
        client_ids = event.get("client_ids")
        project_id = event.get("project_id")
        
        await self.broadcast_event(
            event_type=EventType(event_type),
            data=data,
            client_ids=client_ids,
            project_id=project_id
        )
    
    async def queue_event(self, event_type: EventType, data: Dict[str, Any],
                         client_ids: Optional[List[str]] = None,
                         project_id: Optional[int] = None):
        """
        Queue an event for async processing.
        
        Args:
            event_type: Type of event
            data: Event data
            client_ids: Target client IDs
            project_id: Associated project ID
        """
        event = {
            "type": event_type.value,
            "data": data,
            "client_ids": client_ids,
            "project_id": project_id
        }
        
        await self.event_queue.put(event)
    
    ####################################################################################################################
    # Helper Methods
    
    async def _get_project_for_job(self, job: Dict[str, Any]) -> Optional[int]:
        """
        Get project ID for a job.
        
        Args:
            job: Job data
            
        Returns:
            Project ID or None
        """
        job_type = job["job_type"]
        entity_id = job["entity_id"]
        
        conn = self.db.get_connection()
        cursor = conn.cursor()
        
        try:
            if job_type == "evaluation":
                cursor.execute(
                    "SELECT project_id FROM prompt_studio_evaluations WHERE id = ?",
                    (entity_id,)
                )
            elif job_type == "optimization":
                cursor.execute(
                    "SELECT project_id FROM prompt_studio_optimizations WHERE id = ?",
                    (entity_id,)
                )
            else:
                return None
            
            row = cursor.fetchone()
            return row[0] if row else None
            
        except Exception as e:
            logger.error(f"Error getting project for job: {e}")
            return None
    
    async def _log_event(self, event_type: EventType, data: Dict[str, Any],
                        project_id: Optional[int] = None):
        """
        Log event to database for history/audit.
        
        Args:
            event_type: Event type
            data: Event data
            project_id: Associated project ID
        """
        try:
            conn = self.db.get_connection()
            cursor = conn.cursor()
            
            # Store in sync_log for now (could add dedicated event log table)
            cursor.execute("""
                INSERT INTO sync_log (
                    table_name, operation, entity_id, 
                    entity_data, client_id, synced_at
                ) VALUES (?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
            """, (
                "prompt_studio_events",
                event_type.value,
                project_id or 0,
                json.dumps(data),
                self.client_id
            ))
            
            conn.commit()
            
        except Exception as e:
            logger.error(f"Failed to log event: {e}")

########################################################################################################################
# Event Hooks for Integration

class EventHooks:
    """Hooks for integrating event broadcasting into operations."""
    
    def __init__(self, broadcaster: EventBroadcaster):
        """
        Initialize EventHooks.
        
        Args:
            broadcaster: EventBroadcaster instance
        """
        self.broadcaster = broadcaster
    
    async def on_job_created(self, job_id: int):
        """Hook for job creation."""
        await self.broadcaster.broadcast_job_event(
            job_id=job_id,
            event_type=EventType.JOB_CREATED
        )
    
    async def on_job_started(self, job_id: int):
        """Hook for job start."""
        await self.broadcaster.broadcast_job_event(
            job_id=job_id,
            event_type=EventType.JOB_STARTED
        )
    
    async def on_job_completed(self, job_id: int, result: Dict[str, Any]):
        """Hook for job completion."""
        await self.broadcaster.broadcast_job_event(
            job_id=job_id,
            event_type=EventType.JOB_COMPLETED,
            additional_data={"result": result}
        )
    
    async def on_job_failed(self, job_id: int, error: str):
        """Hook for job failure."""
        await self.broadcaster.broadcast_job_event(
            job_id=job_id,
            event_type=EventType.JOB_FAILED,
            additional_data={"error": error}
        )
    
    async def on_test_cases_generated(self, project_id: int, count: int):
        """Hook for test case generation."""
        await self.broadcaster.broadcast_event(
            event_type=EventType.TEST_CASES_GENERATED,
            data={"count": count},
            project_id=project_id
        )