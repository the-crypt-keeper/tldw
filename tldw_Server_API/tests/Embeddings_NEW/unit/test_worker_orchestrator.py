"""
Unit tests for the Worker Orchestrator.

Tests the orchestration of embedding, chunking, and storage workers
with minimal mocking.
"""

import pytest
from unittest.mock import Mock, AsyncMock, patch, MagicMock, call
import asyncio
import numpy as np
from typing import List, Dict, Any
from datetime import datetime
import uuid

from tldw_Server_API.app.core.Embeddings.worker_orchestrator import WorkerOrchestrator
from tldw_Server_API.app.core.Embeddings.queue_schemas import (
    JobRequest,
    JobStatus,
    JobResult,
    JobType
)

# ========================================================================
# Orchestrator Initialization Tests
# ========================================================================

class TestOrchestratorInitialization:
    """Test orchestrator initialization and setup."""
    
    @pytest.mark.unit
    def test_orchestrator_initialization(self):
        """Test basic orchestrator initialization."""
        orchestrator = WorkerOrchestrator()
        
        assert orchestrator is not None
        assert hasattr(orchestrator, 'embedding_worker')
        assert hasattr(orchestrator, 'chunking_worker')
        assert hasattr(orchestrator, 'storage_worker')
        assert hasattr(orchestrator, 'job_queue')
    
    @pytest.mark.unit
    async def test_orchestrator_startup(self):
        """Test orchestrator startup process."""
        orchestrator = WorkerOrchestrator()
        
        with patch.object(orchestrator, '_initialize_workers') as mock_init:
            mock_init.return_value = None
            await orchestrator.start()
            
            mock_init.assert_called_once()
            assert orchestrator.is_running
    
    @pytest.mark.unit
    async def test_orchestrator_shutdown(self):
        """Test orchestrator shutdown process."""
        orchestrator = WorkerOrchestrator()
        await orchestrator.start()
        
        await orchestrator.shutdown()
        
        assert not orchestrator.is_running
        assert orchestrator.job_queue.empty()
    
    @pytest.mark.unit
    def test_orchestrator_configuration(self):
        """Test orchestrator configuration options."""
        config = {
            "max_workers": 4,
            "queue_size": 1000,
            "batch_timeout": 5.0,
            "retry_limit": 3
        }
        
        orchestrator = WorkerOrchestrator(config=config)
        
        assert orchestrator.max_workers == 4
        assert orchestrator.queue_size == 1000
        assert orchestrator.batch_timeout == 5.0
        assert orchestrator.retry_limit == 3

# ========================================================================
# Job Queue Management Tests
# ========================================================================

class TestJobQueueManagement:
    """Test job queue management functionality."""
    
    @pytest.mark.unit
    async def test_submit_single_job(self, sample_job_request):
        """Test submitting a single job to the queue."""
        orchestrator = WorkerOrchestrator()
        
        job_id = await orchestrator.submit_job(sample_job_request)
        
        assert job_id == sample_job_request.job_id
        assert not orchestrator.job_queue.empty()
    
    @pytest.mark.unit
    async def test_submit_batch_jobs(self, batch_job_requests):
        """Test submitting multiple jobs in batch."""
        orchestrator = WorkerOrchestrator()
        
        job_ids = await orchestrator.submit_batch(batch_job_requests)
        
        assert len(job_ids) == len(batch_job_requests)
        assert orchestrator.job_queue.qsize() == len(batch_job_requests)
    
    @pytest.mark.unit
    async def test_priority_queue_ordering(self):
        """Test that jobs are processed by priority."""
        orchestrator = WorkerOrchestrator()
        
        # Submit jobs with different priorities
        high_priority = JobRequest(
            job_id="high",
            job_type=JobType.EMBEDDING,
            media_id=1,
            priority=10,
            data={"text": "high priority"}
        )
        
        low_priority = JobRequest(
            job_id="low",
            job_type=JobType.EMBEDDING,
            media_id=2,
            priority=1,
            data={"text": "low priority"}
        )
        
        await orchestrator.submit_job(low_priority)
        await orchestrator.submit_job(high_priority)
        
        # High priority should be processed first
        next_job = orchestrator.job_queue.get_nowait()
        assert next_job.job_id == "high"
    
    @pytest.mark.unit
    async def test_queue_overflow_handling(self):
        """Test handling of queue overflow."""
        orchestrator = WorkerOrchestrator(config={"queue_size": 5})
        
        # Try to submit more jobs than queue size
        jobs = []
        for i in range(10):
            job = JobRequest(
                job_id=f"job_{i}",
                job_type=JobType.EMBEDDING,
                media_id=i,
                data={"text": f"text {i}"}
            )
            jobs.append(job)
        
        # Should handle overflow gracefully
        with pytest.raises(Exception) as exc_info:
            for job in jobs:
                await orchestrator.submit_job(job)
        
        assert "queue" in str(exc_info.value).lower()

# ========================================================================
# Worker Coordination Tests
# ========================================================================

class TestWorkerCoordination:
    """Test coordination between different workers."""
    
    @pytest.mark.unit
    async def test_embedding_to_storage_pipeline(self, mock_embedding_worker, mock_storage_worker):
        """Test pipeline from embedding to storage."""
        orchestrator = WorkerOrchestrator()
        orchestrator.embedding_worker = mock_embedding_worker
        orchestrator.storage_worker = mock_storage_worker
        
        # Mock the pipeline
        mock_embedding_worker.process.return_value = [0.1, 0.2, 0.3]
        mock_storage_worker.store.return_value = {"status": "success"}
        
        job = JobRequest(
            job_id="test",
            job_type=JobType.EMBEDDING,
            media_id=1,
            data={"text": "test text"}
        )
        
        result = await orchestrator.process_embedding_job(job)
        
        assert result.status == JobStatus.COMPLETED
        mock_embedding_worker.process.assert_called_once()
        mock_storage_worker.store.assert_called_once()
    
    @pytest.mark.unit
    async def test_chunking_to_embedding_pipeline(self, mock_chunking_worker, mock_embedding_worker):
        """Test pipeline from chunking to embedding."""
        orchestrator = WorkerOrchestrator()
        orchestrator.chunking_worker = mock_chunking_worker
        orchestrator.embedding_worker = mock_embedding_worker
        
        # Mock the pipeline
        mock_chunking_worker.process.return_value = [
            {"text": "chunk1", "metadata": {}},
            {"text": "chunk2", "metadata": {}}
        ]
        mock_embedding_worker.process.return_value = [[0.1, 0.2], [0.3, 0.4]]
        
        job = JobRequest(
            job_id="test",
            job_type=JobType.CHUNKING,
            media_id=1,
            data={"text": "long text to chunk"}
        )
        
        result = await orchestrator.process_chunking_job(job)
        
        assert result.status == JobStatus.COMPLETED
        mock_chunking_worker.process.assert_called_once()
        mock_embedding_worker.process.assert_called()
    
    @pytest.mark.unit
    async def test_full_pipeline_execution(self, mock_chunking_worker, mock_embedding_worker, mock_storage_worker):
        """Test full pipeline from chunking to storage."""
        orchestrator = WorkerOrchestrator()
        orchestrator.chunking_worker = mock_chunking_worker
        orchestrator.embedding_worker = mock_embedding_worker
        orchestrator.storage_worker = mock_storage_worker
        
        # Mock the full pipeline
        chunks = [{"text": f"chunk{i}", "metadata": {"id": i}} for i in range(3)]
        mock_chunking_worker.process.return_value = chunks
        mock_embedding_worker.process.return_value = np.random.randn(3, 384).tolist()
        mock_storage_worker.store.return_value = {"stored": 3}
        
        job = JobRequest(
            job_id="full_pipeline",
            job_type=JobType.FULL_PIPELINE,
            media_id=1,
            data={"text": "Process this through full pipeline"}
        )
        
        result = await orchestrator.process_full_pipeline(job)
        
        assert result.status == JobStatus.COMPLETED
        assert result.result["chunks_processed"] == 3
        assert result.result["embeddings_generated"] == 3

# ========================================================================
# Batch Processing Tests
# ========================================================================

class TestBatchProcessing:
    """Test batch processing capabilities."""
    
    @pytest.mark.unit
    async def test_batch_job_processing(self, batch_job_requests, mock_embedding_worker):
        """Test processing multiple jobs in batch."""
        orchestrator = WorkerOrchestrator()
        orchestrator.embedding_worker = mock_embedding_worker
        
        # Mock batch processing
        mock_embedding_worker.process_batch = AsyncMock(
            return_value=[JobResult(job_id=j.job_id, status=JobStatus.COMPLETED) 
                         for j in batch_job_requests]
        )
        
        results = await orchestrator.process_batch(batch_job_requests)
        
        assert len(results) == len(batch_job_requests)
        assert all(r.status == JobStatus.COMPLETED for r in results)
        mock_embedding_worker.process_batch.assert_called_once()
    
    @pytest.mark.unit
    async def test_batch_size_optimization(self):
        """Test dynamic batch size optimization."""
        orchestrator = WorkerOrchestrator(config={"optimal_batch_size": 10})
        
        # Create jobs of varying sizes
        small_batch = [JobRequest(job_id=f"s{i}", job_type=JobType.EMBEDDING, media_id=i, data={}) 
                      for i in range(3)]
        optimal_batch = [JobRequest(job_id=f"o{i}", job_type=JobType.EMBEDDING, media_id=i, data={}) 
                        for i in range(10)]
        large_batch = [JobRequest(job_id=f"l{i}", job_type=JobType.EMBEDDING, media_id=i, data={}) 
                      for i in range(50)]
        
        # Test batch formation
        batches = orchestrator._form_batches(large_batch)
        
        assert len(batches) == 5  # 50 jobs / 10 optimal size
        assert all(len(b) <= 10 for b in batches)
    
    @pytest.mark.unit
    async def test_mixed_job_type_batching(self, mock_embedding_worker, mock_chunking_worker):
        """Test batching with mixed job types."""
        orchestrator = WorkerOrchestrator()
        orchestrator.embedding_worker = mock_embedding_worker
        orchestrator.chunking_worker = mock_chunking_worker
        
        # Create mixed job types
        jobs = []
        for i in range(10):
            job_type = JobType.EMBEDDING if i % 2 == 0 else JobType.CHUNKING
            jobs.append(JobRequest(
                job_id=f"job_{i}",
                job_type=job_type,
                media_id=i,
                data={"text": f"text {i}"}
            ))
        
        # Process mixed batch
        await orchestrator.submit_batch(jobs)
        
        # Jobs should be routed to appropriate workers
        embedding_jobs = [j for j in jobs if j.job_type == JobType.EMBEDDING]
        chunking_jobs = [j for j in jobs if j.job_type == JobType.CHUNKING]
        
        assert len(embedding_jobs) == 5
        assert len(chunking_jobs) == 5

# ========================================================================
# Error Handling and Recovery Tests
# ========================================================================

class TestErrorHandlingAndRecovery:
    """Test error handling and recovery mechanisms."""
    
    @pytest.mark.unit
    async def test_worker_failure_recovery(self, mock_embedding_worker):
        """Test recovery from worker failure."""
        orchestrator = WorkerOrchestrator()
        orchestrator.embedding_worker = mock_embedding_worker
        
        # Simulate worker failure then recovery
        call_count = 0
        def side_effect(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise Exception("Worker failed")
            return [0.1, 0.2, 0.3]
        
        mock_embedding_worker.process.side_effect = side_effect
        
        job = JobRequest(
            job_id="test",
            job_type=JobType.EMBEDDING,
            media_id=1,
            data={"text": "test"}
        )
        
        result = await orchestrator.process_with_retry(job, max_retries=3)
        
        assert result.status == JobStatus.COMPLETED
        assert call_count == 2  # Failed once, succeeded on retry
    
    @pytest.mark.unit
    async def test_job_timeout_handling(self):
        """Test handling of job timeouts."""
        orchestrator = WorkerOrchestrator(config={"job_timeout": 1.0})
        
        async def slow_process(*args):
            await asyncio.sleep(5)  # Longer than timeout
            return [0.1, 0.2, 0.3]
        
        orchestrator.embedding_worker = MagicMock()
        orchestrator.embedding_worker.process = slow_process
        
        job = JobRequest(
            job_id="timeout_test",
            job_type=JobType.EMBEDDING,
            media_id=1,
            data={"text": "test"}
        )
        
        result = await orchestrator.process_with_timeout(job)
        
        assert result.status == JobStatus.FAILED
        assert "timeout" in result.error.lower()
    
    @pytest.mark.unit
    async def test_partial_batch_failure(self, mock_embedding_worker):
        """Test handling of partial batch failures."""
        orchestrator = WorkerOrchestrator()
        
        # Mock partial failure
        def batch_process(jobs):
            results = []
            for i, job in enumerate(jobs):
                if i == 2:  # Third job fails
                    results.append(JobResult(
                        job_id=job.job_id,
                        status=JobStatus.FAILED,
                        error="Processing failed"
                    ))
                else:
                    results.append(JobResult(
                        job_id=job.job_id,
                        status=JobStatus.COMPLETED,
                        result={"embedding": [0.1, 0.2]}
                    ))
            return results
        
        mock_embedding_worker.process_batch = batch_process
        orchestrator.embedding_worker = mock_embedding_worker
        
        jobs = [JobRequest(job_id=f"job_{i}", job_type=JobType.EMBEDDING, media_id=i, data={})
               for i in range(5)]
        
        results = await orchestrator.process_batch(jobs)
        
        assert sum(1 for r in results if r.status == JobStatus.COMPLETED) == 4
        assert sum(1 for r in results if r.status == JobStatus.FAILED) == 1

# ========================================================================
# Performance Monitoring Tests
# ========================================================================

class TestPerformanceMonitoring:
    """Test performance monitoring and metrics."""
    
    @pytest.mark.unit
    async def test_job_metrics_collection(self):
        """Test collection of job processing metrics."""
        orchestrator = WorkerOrchestrator()
        
        # Process some jobs and collect metrics
        jobs_processed = []
        for i in range(10):
            job = JobRequest(
                job_id=f"job_{i}",
                job_type=JobType.EMBEDDING,
                media_id=i,
                data={"text": f"text {i}"}
            )
            await orchestrator.submit_job(job)
            jobs_processed.append(job)
        
        metrics = orchestrator.get_metrics()
        
        assert "jobs_submitted" in metrics
        assert metrics["jobs_submitted"] == 10
        assert "queue_size" in metrics
    
    @pytest.mark.unit
    async def test_throughput_monitoring(self):
        """Test monitoring of processing throughput."""
        orchestrator = WorkerOrchestrator()
        
        with patch.object(orchestrator, '_calculate_throughput') as mock_throughput:
            mock_throughput.return_value = 100.0  # jobs per second
            
            throughput = orchestrator.get_throughput()
            
            assert throughput == 100.0
            mock_throughput.assert_called_once()
    
    @pytest.mark.unit
    async def test_resource_usage_tracking(self):
        """Test tracking of resource usage."""
        orchestrator = WorkerOrchestrator()
        
        # Mock resource monitoring
        with patch.object(orchestrator, '_get_resource_usage') as mock_resources:
            mock_resources.return_value = {
                "cpu_percent": 45.5,
                "memory_mb": 1024,
                "gpu_percent": 80.0
            }
            
            resources = orchestrator.get_resource_usage()
            
            assert resources["cpu_percent"] == 45.5
            assert resources["memory_mb"] == 1024
            assert resources["gpu_percent"] == 80.0