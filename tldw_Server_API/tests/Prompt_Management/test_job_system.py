# test_job_system.py
# Tests for Prompt Studio job queue and processing system

import pytest
import asyncio
import json
from unittest.mock import MagicMock, AsyncMock, patch
from datetime import datetime, timedelta

from tldw_Server_API.app.core.Prompt_Management.prompt_studio.job_manager import (
    JobManager, JobType, JobStatus
)
from tldw_Server_API.app.core.Prompt_Management.prompt_studio.job_processor import JobProcessor
from tldw_Server_API.app.core.Prompt_Management.prompt_studio.event_broadcaster import (
    EventBroadcaster, EventType
)

########################################################################################################################
# Test JobManager

class TestJobManager:
    """Test cases for JobManager."""
    
    @pytest.fixture
    def job_manager(self, prompt_studio_db):
        """Create JobManager instance."""
        return JobManager(prompt_studio_db)
    
    def test_create_job(self, job_manager):
        """Test creating a job."""
        job = job_manager.create_job(
            job_type=JobType.GENERATION,
            entity_id=1,
            payload={"test": "data"},
            priority=5
        )
        
        assert job is not None
        assert job["job_type"] == JobType.GENERATION.value
        assert job["entity_id"] == 1
        assert job["status"] == JobStatus.QUEUED.value
        assert job["priority"] == 5
        assert json.loads(job["payload"]) == {"test": "data"}
    
    def test_get_job(self, job_manager):
        """Test getting a job by ID."""
        # Create job
        created = job_manager.create_job(
            job_type=JobType.EVALUATION,
            entity_id=2,
            payload={"eval": "params"}
        )
        
        # Get job
        job = job_manager.get_job(created["id"])
        assert job is not None
        assert job["id"] == created["id"]
        assert job["job_type"] == JobType.EVALUATION.value
    
    def test_get_job_by_uuid(self, job_manager):
        """Test getting a job by UUID."""
        # Create job
        created = job_manager.create_job(
            job_type=JobType.OPTIMIZATION,
            entity_id=3,
            payload={"opt": "config"}
        )
        
        # Get by UUID
        job = job_manager.get_job_by_uuid(created["uuid"])
        assert job is not None
        assert job["uuid"] == created["uuid"]
    
    def test_list_jobs(self, job_manager):
        """Test listing jobs with filters."""
        # Create multiple jobs
        job_manager.create_job(JobType.GENERATION, 1, {}, priority=1)
        job_manager.create_job(JobType.EVALUATION, 2, {}, priority=5)
        job_manager.create_job(JobType.OPTIMIZATION, 3, {}, priority=10)
        
        # List all jobs
        all_jobs = job_manager.list_jobs()
        assert len(all_jobs) >= 3
        
        # List by type
        eval_jobs = job_manager.list_jobs(job_type=JobType.EVALUATION)
        assert all(j["job_type"] == JobType.EVALUATION.value for j in eval_jobs)
        
        # List by status
        queued_jobs = job_manager.list_jobs(status=JobStatus.QUEUED)
        assert all(j["status"] == JobStatus.QUEUED.value for j in queued_jobs)
    
    def test_update_job_status(self, job_manager):
        """Test updating job status."""
        # Create job
        job = job_manager.create_job(JobType.GENERATION, 1, {})
        
        # Update to processing
        success = job_manager.update_job_status(
            job["id"],
            JobStatus.PROCESSING
        )
        assert success
        
        # Verify update
        updated = job_manager.get_job(job["id"])
        assert updated["status"] == JobStatus.PROCESSING.value
        assert updated["started_at"] is not None
        
        # Update to completed with result
        result = {"generated": 5}
        success = job_manager.update_job_status(
            job["id"],
            JobStatus.COMPLETED,
            result=result
        )
        assert success
        
        # Verify completion
        completed = job_manager.get_job(job["id"])
        assert completed["status"] == JobStatus.COMPLETED.value
        assert completed["completed_at"] is not None
        assert json.loads(completed["result"]) == result
    
    def test_cancel_job(self, job_manager):
        """Test cancelling a job."""
        # Create job
        job = job_manager.create_job(JobType.EVALUATION, 1, {})
        
        # Cancel job
        success = job_manager.cancel_job(job["id"], "User cancelled")
        assert success
        
        # Verify cancellation
        cancelled = job_manager.get_job(job["id"])
        assert cancelled["status"] == JobStatus.CANCELLED.value
        assert "User cancelled" in cancelled["error_message"]
        
        # Can't cancel completed job
        job_manager.update_job_status(job["id"], JobStatus.COMPLETED)
        success = job_manager.cancel_job(job["id"])
        assert not success
    
    def test_get_next_job(self, job_manager):
        """Test getting next job from queue."""
        # Create jobs with different priorities
        low = job_manager.create_job(JobType.GENERATION, 1, {}, priority=1)
        high = job_manager.create_job(JobType.GENERATION, 2, {}, priority=10)
        med = job_manager.create_job(JobType.GENERATION, 3, {}, priority=5)
        
        # Should get highest priority first
        next_job = job_manager.get_next_job()
        assert next_job["id"] == high["id"]
        assert next_job["status"] == JobStatus.PROCESSING.value
        
        # Should get medium priority next
        next_job = job_manager.get_next_job()
        assert next_job["id"] == med["id"]
    
    def test_retry_job(self, job_manager):
        """Test retrying a failed job."""
        # Create job with max_retries=3
        job = job_manager.create_job(
            JobType.OPTIMIZATION,
            1,
            {},
            max_retries=3
        )
        
        # Fail the job
        job_manager.update_job_status(
            job["id"],
            JobStatus.FAILED,
            error_message="First failure"
        )
        
        # Retry job
        success = job_manager.retry_job(job["id"])
        assert success
        
        # Verify retry
        retried = job_manager.get_job(job["id"])
        assert retried["status"] == JobStatus.QUEUED.value
        assert retried["retry_count"] == 1
        assert retried["error_message"] is None
        
        # Fail and retry until max
        for i in range(2, 4):
            job_manager.update_job_status(job["id"], JobStatus.FAILED)
            if i < 3:
                success = job_manager.retry_job(job["id"])
                assert success
            else:
                # Should fail on 4th attempt
                success = job_manager.retry_job(job["id"])
                assert not success
    
    def test_job_stats(self, job_manager):
        """Test getting job statistics."""
        # Create various jobs
        job_manager.create_job(JobType.GENERATION, 1, {})
        job_manager.create_job(JobType.EVALUATION, 2, {})
        
        job = job_manager.create_job(JobType.OPTIMIZATION, 3, {})
        job_manager.update_job_status(job["id"], JobStatus.PROCESSING)
        job_manager.update_job_status(job["id"], JobStatus.COMPLETED)
        
        # Get stats
        stats = job_manager.get_job_stats()
        
        assert "by_status" in stats
        assert "by_type" in stats
        assert "queue_depth" in stats
        assert "processing" in stats
        assert "success_rate" in stats
    
    def test_cleanup_old_jobs(self, job_manager):
        """Test cleaning up old jobs."""
        # Create old job (simulate by updating timestamp directly)
        job = job_manager.create_job(JobType.GENERATION, 1, {})
        job_manager.update_job_status(job["id"], JobStatus.COMPLETED)
        
        # Manually set completed_at to old date
        conn = job_manager.db.get_connection()
        cursor = conn.cursor()
        old_date = (datetime.utcnow() - timedelta(days=35)).isoformat()
        cursor.execute(
            "UPDATE prompt_studio_job_queue SET completed_at = ? WHERE id = ?",
            (old_date, job["id"])
        )
        conn.commit()
        
        # Clean up jobs older than 30 days
        deleted = job_manager.cleanup_old_jobs(days=30)
        assert deleted > 0
        
        # Job should be gone
        assert job_manager.get_job(job["id"]) is None

########################################################################################################################
# Test JobProcessor

class TestJobProcessor:
    """Test cases for JobProcessor."""
    
    @pytest.fixture
    def job_processor(self, prompt_studio_db):
        """Create JobProcessor instance."""
        job_manager = JobManager(prompt_studio_db)
        return JobProcessor(prompt_studio_db, job_manager)
    
    @pytest.mark.asyncio
    async def test_process_generation_job(self, job_processor, test_project):
        """Test processing a generation job."""
        # Create signature for generation
        conn = job_processor.db.get_connection()
        cursor = conn.cursor()
        cursor.execute("""
            INSERT INTO prompt_studio_signatures (
                uuid, project_id, name, input_schema, output_schema, client_id
            ) VALUES (?, ?, ?, ?, ?, ?)
        """, (
            "sig-uuid", test_project["id"], "Test Signature",
            json.dumps([{"name": "input", "type": "string"}]),
            json.dumps([{"name": "output", "type": "string"}]),
            job_processor.db.client_id
        ))
        signature_id = cursor.lastrowid
        conn.commit()
        
        # Process generation job
        payload = {
            "type": "diverse",
            "signature_id": signature_id,
            "num_cases": 3
        }
        
        result = await job_processor.process_generation_job(
            payload,
            test_project["id"]
        )
        
        assert result["generated_count"] == 3
        assert len(result["test_case_ids"]) == 3
    
    @pytest.mark.asyncio
    async def test_process_evaluation_job(self, job_processor, test_project):
        """Test processing an evaluation job."""
        # Create test case
        test_manager = job_processor.test_manager
        test_case = test_manager.create_test_case(
            project_id=test_project["id"],
            name="Test Case",
            inputs={"text": "test"},
            expected_outputs={"result": "expected"}
        )
        
        # Create evaluation
        conn = job_processor.db.get_connection()
        cursor = conn.cursor()
        cursor.execute("""
            INSERT INTO prompt_studio_evaluations (
                uuid, project_id, name, prompt_id, test_case_ids,
                model_configs, status, client_id
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            "eval-uuid", test_project["id"], "Test Eval", 1,
            json.dumps([test_case["id"]]),
            json.dumps([{"model": "gpt-3.5-turbo"}]),
            "pending", job_processor.db.client_id
        ))
        evaluation_id = cursor.lastrowid
        conn.commit()
        
        # Process evaluation job
        payload = {
            "prompt_id": 1,
            "test_case_ids": [test_case["id"]],
            "model_configs": [{"model": "gpt-3.5-turbo"}]
        }
        
        result = await job_processor.process_evaluation_job(
            payload,
            evaluation_id
        )
        
        assert result["evaluation_id"] == evaluation_id
        assert result["test_runs"] == 1
        assert result["status"] == "completed"
        assert "aggregate_metrics" in result
    
    @pytest.mark.asyncio
    async def test_process_optimization_job(self, job_processor, test_project):
        """Test processing an optimization job."""
        # Create optimization
        conn = job_processor.db.get_connection()
        cursor = conn.cursor()
        cursor.execute("""
            INSERT INTO prompt_studio_optimizations (
                uuid, project_id, name, initial_prompt_id,
                optimizer_type, max_iterations, status, client_id
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            "opt-uuid", test_project["id"], "Test Opt", 1,
            "basic", 5, "pending", job_processor.db.client_id
        ))
        optimization_id = cursor.lastrowid
        conn.commit()
        
        # Process optimization job
        payload = {
            "initial_prompt_id": 1,
            "optimizer_type": "basic",
            "max_iterations": 5
        }
        
        result = await job_processor.process_optimization_job(
            payload,
            optimization_id
        )
        
        assert result["optimization_id"] == optimization_id
        assert result["iterations_completed"] <= 5
        assert result["status"] == "completed"
        assert "improvement_percentage" in result

########################################################################################################################
# Test Event Broadcasting

class TestEventBroadcaster:
    """Test cases for EventBroadcaster."""
    
    @pytest.fixture
    def mock_connection_manager(self):
        """Create mock connection manager."""
        manager = MagicMock()
        manager.broadcast_to_client = AsyncMock()
        manager.broadcast_to_all = AsyncMock()
        return manager
    
    @pytest.fixture
    def event_broadcaster(self, prompt_studio_db, mock_connection_manager):
        """Create EventBroadcaster instance."""
        return EventBroadcaster(mock_connection_manager, prompt_studio_db)
    
    @pytest.mark.asyncio
    async def test_broadcast_event(self, event_broadcaster, mock_connection_manager):
        """Test broadcasting an event."""
        await event_broadcaster.broadcast_event(
            event_type=EventType.JOB_CREATED,
            data={"job_id": 1},
            project_id=1
        )
        
        # Should broadcast to all
        mock_connection_manager.broadcast_to_all.assert_called_once()
        
        # Check message format
        call_args = mock_connection_manager.broadcast_to_all.call_args[0][0]
        message = json.loads(call_args)
        assert message["type"] == EventType.JOB_CREATED.value
        assert message["data"]["job_id"] == 1
        assert message["project_id"] == 1
    
    @pytest.mark.asyncio
    async def test_broadcast_to_specific_clients(self, event_broadcaster, mock_connection_manager):
        """Test broadcasting to specific clients."""
        client_ids = ["client1", "client2"]
        
        await event_broadcaster.broadcast_event(
            event_type=EventType.EVALUATION_STARTED,
            data={"eval_id": 1},
            client_ids=client_ids
        )
        
        # Should broadcast to specific clients
        assert mock_connection_manager.broadcast_to_client.call_count == 2
    
    def test_subscription_management(self, event_broadcaster):
        """Test subscription management."""
        # Subscribe
        event_broadcaster.subscribe("client1", "job", 1)
        event_broadcaster.subscribe("client2", "job", 1)
        event_broadcaster.subscribe("client1", "evaluation", 2)
        
        # Get subscribers
        job_subs = event_broadcaster.get_subscribers("job", 1)
        assert "client1" in job_subs
        assert "client2" in job_subs
        
        eval_subs = event_broadcaster.get_subscribers("evaluation", 2)
        assert "client1" in eval_subs
        assert "client2" not in eval_subs
        
        # Unsubscribe
        event_broadcaster.unsubscribe("client1", "job", 1)
        job_subs = event_broadcaster.get_subscribers("job", 1)
        assert "client1" not in job_subs
        assert "client2" in job_subs
    
    @pytest.mark.asyncio
    async def test_progress_broadcasting(self, event_broadcaster, mock_connection_manager):
        """Test broadcasting progress updates."""
        # Create a mock job
        conn = event_broadcaster.db.get_connection()
        cursor = conn.cursor()
        cursor.execute("""
            INSERT INTO prompt_studio_job_queue (
                uuid, job_type, entity_id, priority, status, 
                payload, client_id
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (
            "job-uuid", JobType.GENERATION.value, 1, 5,
            JobStatus.PROCESSING.value, "{}", event_broadcaster.db.client_id
        ))
        job_id = cursor.lastrowid
        conn.commit()
        
        # Broadcast progress
        await event_broadcaster.broadcast_progress(
            job_id=job_id,
            progress=50.0,
            message="Halfway done"
        )
        
        # Check broadcast
        mock_connection_manager.broadcast_to_all.assert_called()
        call_args = mock_connection_manager.broadcast_to_all.call_args[0][0]
        message = json.loads(call_args)
        assert message["type"] == EventType.JOB_PROGRESS.value
        assert message["data"]["progress"] == 50.0
        assert message["data"]["message"] == "Halfway done"

########################################################################################################################
# Integration Tests

class TestJobSystemIntegration:
    """Integration tests for the complete job system."""
    
    @pytest.mark.asyncio
    async def test_end_to_end_job_processing(self, prompt_studio_db, test_project):
        """Test complete job lifecycle."""
        # Setup
        job_manager = JobManager(prompt_studio_db)
        job_processor = JobProcessor(prompt_studio_db, job_manager)
        
        # Create job
        job = job_manager.create_job(
            job_type=JobType.GENERATION,
            entity_id=test_project["id"],
            payload={
                "type": "description",
                "description": "Test generation",
                "num_cases": 2
            }
        )
        
        # Process job
        result = await job_processor.process_job(job)
        
        # Verify result
        assert result["generated_count"] == 2
        
        # Verify job status
        processed_job = job_manager.get_job(job["id"])
        assert processed_job["status"] == JobStatus.COMPLETED.value
        assert processed_job["result"] is not None
    
    @pytest.mark.asyncio
    async def test_concurrent_job_processing(self, prompt_studio_db, test_project):
        """Test processing multiple jobs concurrently."""
        job_manager = JobManager(prompt_studio_db)
        job_processor = JobProcessor(prompt_studio_db, job_manager)
        
        # Create multiple jobs
        jobs = []
        for i in range(3):
            job = job_manager.create_job(
                job_type=JobType.GENERATION,
                entity_id=test_project["id"],
                payload={
                    "type": "description",
                    "description": f"Test {i}",
                    "num_cases": 1
                }
            )
            jobs.append(job)
        
        # Process concurrently
        tasks = [job_processor.process_job(job) for job in jobs]
        results = await asyncio.gather(*tasks)
        
        # Verify all completed
        assert len(results) == 3
        for result in results:
            assert result["generated_count"] == 1