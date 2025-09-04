"""
Unit tests for embedding worker classes.

Tests chunking, embedding generation, and storage workers
with mocked dependencies.
"""

import pytest
from unittest.mock import Mock, MagicMock, patch, call
from queue import Queue, Empty
import time
import threading
from typing import List

from tldw_Server_API.app.core.Embeddings.workers.chunking_worker import ChunkingWorker
from tldw_Server_API.app.core.Embeddings.workers.embedding_worker import EmbeddingWorker
from tldw_Server_API.app.core.Embeddings.workers.storage_worker import StorageWorker
from tldw_Server_API.app.core.Embeddings.workers.base_worker import BaseWorker
from tldw_Server_API.app.core.Embeddings.queue_schemas import (
    ChunkingTask,
    EmbeddingTask,
    StorageTask,
    JobStatus
)


@pytest.mark.unit
class TestBaseWorker:
    """Test base worker functionality."""
    
    def test_base_worker_initialization(self):
        """Test base worker initialization."""
        input_queue = Queue()
        output_queue = Queue()
        
        worker = BaseWorker(
            worker_id="test_worker",
            input_queue=input_queue,
            output_queue=output_queue
        )
        
        assert worker.worker_id == "test_worker"
        assert worker.input_queue == input_queue
        assert worker.output_queue == output_queue
        assert worker.running is False
        assert worker.metrics["tasks_processed"] == 0
        assert worker.metrics["tasks_failed"] == 0
    
    def test_base_worker_start_stop(self):
        """Test starting and stopping base worker."""
        worker = BaseWorker(
            worker_id="test_worker",
            input_queue=Queue(),
            output_queue=Queue()
        )
        
        # Start worker
        worker.start()
        assert worker.running is True
        assert worker.thread is not None
        assert worker.thread.is_alive()
        
        # Stop worker
        worker.stop()
        assert worker.running is False
        worker.thread.join(timeout=2)
        assert not worker.thread.is_alive()
    
    def test_base_worker_metrics_tracking(self):
        """Test metrics tracking in base worker."""
        worker = BaseWorker(
            worker_id="test_worker",
            input_queue=Queue(),
            output_queue=Queue()
        )
        
        # Update metrics
        worker._update_metrics("tasks_processed", 1)
        assert worker.metrics["tasks_processed"] == 1
        
        worker._update_metrics("tasks_failed", 2)
        assert worker.metrics["tasks_failed"] == 2
        
        # Get metrics
        metrics = worker.get_metrics()
        assert metrics["tasks_processed"] == 1
        assert metrics["tasks_failed"] == 2
        assert metrics["worker_id"] == "test_worker"
    
    def test_base_worker_error_handling(self):
        """Test error handling in base worker."""
        input_queue = Queue()
        worker = BaseWorker(
            worker_id="test_worker",
            input_queue=input_queue,
            output_queue=Queue()
        )
        
        # Mock process_task to raise exception
        with patch.object(worker, 'process_task') as mock_process:
            mock_process.side_effect = Exception("Processing failed")
            
            # Add task and start worker
            input_queue.put("test_task")
            worker.start()
            
            # Wait briefly for processing
            time.sleep(0.1)
            
            # Stop worker
            worker.stop()
            
            # Check error was handled
            assert worker.metrics["tasks_failed"] > 0


@pytest.mark.unit
class TestChunkingWorker:
    """Test chunking worker functionality."""
    
    def test_chunking_worker_initialization(self):
        """Test chunking worker initialization."""
        input_queue = Queue()
        output_queue = Queue()
        
        worker = ChunkingWorker(
            worker_id="chunking_1",
            input_queue=input_queue,
            output_queue=output_queue,
            chunk_size=1000,
            overlap=100
        )
        
        assert worker.worker_id == "chunking_1"
        assert worker.chunk_size == 1000
        assert worker.overlap == 100
    
    @patch('tldw_Server_API.app.core.Embeddings.workers.chunking_worker.improved_chunking_process')
    def test_process_chunking_task(self, mock_chunking):
        """Test processing a chunking task."""
        input_queue = Queue()
        output_queue = Queue()
        
        worker = ChunkingWorker(
            worker_id="chunking_1",
            input_queue=input_queue,
            output_queue=output_queue
        )
        
        # Setup mock
        mock_chunking.return_value = ["chunk1", "chunk2", "chunk3"]
        
        # Create task
        task = ChunkingTask(
            job_id="job_1",
            content="Long content to be chunked",
            chunk_size=500,
            overlap=50,
            metadata={"source": "test"}
        )
        
        # Process task
        worker.process_task(task)
        
        # Verify chunking was called
        mock_chunking.assert_called_once_with(
            task.content,
            chunk_size=500,
            overlap=50
        )
        
        # Verify output was queued
        assert not output_queue.empty()
        result = output_queue.get()
        assert result.job_id == "job_1"
        assert result.chunks == ["chunk1", "chunk2", "chunk3"]
        assert result.metadata == {"source": "test"}
    
    @patch('tldw_Server_API.app.core.Embeddings.workers.chunking_worker.improved_chunking_process')
    def test_chunking_with_default_params(self, mock_chunking):
        """Test chunking with default parameters."""
        worker = ChunkingWorker(
            worker_id="chunking_1",
            input_queue=Queue(),
            output_queue=Queue(),
            chunk_size=2000,
            overlap=200
        )
        
        mock_chunking.return_value = ["chunk1"]
        
        task = ChunkingTask(
            job_id="job_1",
            content="Content",
            chunk_size=None,  # Use defaults
            overlap=None
        )
        
        worker.process_task(task)
        
        # Should use worker defaults
        mock_chunking.assert_called_with(
            "Content",
            chunk_size=2000,
            overlap=200
        )
    
    @patch('tldw_Server_API.app.core.Embeddings.workers.chunking_worker.improved_chunking_process')
    def test_chunking_error_handling(self, mock_chunking):
        """Test error handling in chunking worker."""
        output_queue = Queue()
        worker = ChunkingWorker(
            worker_id="chunking_1",
            input_queue=Queue(),
            output_queue=output_queue
        )
        
        # Make chunking fail
        mock_chunking.side_effect = Exception("Chunking failed")
        
        task = ChunkingTask(
            job_id="job_1",
            content="Content",
            chunk_size=500,
            overlap=50
        )
        
        # Process should not raise but queue error
        worker.process_task(task)
        
        # Check error was queued
        assert not output_queue.empty()
        result = output_queue.get()
        assert result.error is not None
        assert "Chunking failed" in str(result.error)
    
    def test_chunking_worker_full_pipeline(self):
        """Test full chunking pipeline with multiple tasks."""
        input_queue = Queue()
        output_queue = Queue()
        
        worker = ChunkingWorker(
            worker_id="chunking_1",
            input_queue=input_queue,
            output_queue=output_queue
        )
        
        with patch('tldw_Server_API.app.core.Embeddings.workers.chunking_worker.improved_chunking_process') as mock_chunk:
            mock_chunk.side_effect = [
                ["chunk1_1", "chunk1_2"],
                ["chunk2_1", "chunk2_2", "chunk2_3"],
                ["chunk3_1"]
            ]
            
            # Add multiple tasks
            for i in range(3):
                task = ChunkingTask(
                    job_id=f"job_{i}",
                    content=f"Content {i}",
                    chunk_size=500,
                    overlap=50
                )
                input_queue.put(task)
            
            # Start worker
            worker.start()
            
            # Wait for processing
            time.sleep(0.2)
            
            # Stop worker
            worker.stop()
            
            # Verify all tasks processed
            assert worker.metrics["tasks_processed"] >= 3
            assert output_queue.qsize() >= 3


@pytest.mark.unit
class TestEmbeddingWorker:
    """Test embedding generation worker."""
    
    def test_embedding_worker_initialization(self):
        """Test embedding worker initialization."""
        worker = EmbeddingWorker(
            worker_id="embedding_1",
            input_queue=Queue(),
            output_queue=Queue(),
            provider="openai",
            model="text-embedding-ada-002"
        )
        
        assert worker.worker_id == "embedding_1"
        assert worker.provider == "openai"
        assert worker.model == "text-embedding-ada-002"
    
    @patch('tldw_Server_API.app.core.Embeddings.workers.embedding_worker.create_embeddings_batch')
    def test_process_embedding_task(self, mock_create_embeddings):
        """Test processing an embedding task."""
        output_queue = Queue()
        worker = EmbeddingWorker(
            worker_id="embedding_1",
            input_queue=Queue(),
            output_queue=output_queue
        )
        
        # Setup mock
        mock_create_embeddings.return_value = [[0.1, 0.2], [0.3, 0.4]]
        
        # Create task
        task = EmbeddingTask(
            job_id="job_1",
            chunks=["text1", "text2"],
            provider="openai",
            model="text-embedding-ada-002",
            metadata={"source": "test"}
        )
        
        # Process task
        worker.process_task(task)
        
        # Verify embedding creation
        mock_create_embeddings.assert_called_once_with(
            ["text1", "text2"],
            provider="openai",
            model="text-embedding-ada-002"
        )
        
        # Verify output
        assert not output_queue.empty()
        result = output_queue.get()
        assert result.job_id == "job_1"
        assert result.embeddings == [[0.1, 0.2], [0.3, 0.4]]
        assert result.metadata == {"source": "test"}
    
    @patch('tldw_Server_API.app.core.Embeddings.workers.embedding_worker.create_embeddings_batch')
    def test_embedding_batch_processing(self, mock_create_embeddings):
        """Test batch processing of embeddings."""
        worker = EmbeddingWorker(
            worker_id="embedding_1",
            input_queue=Queue(),
            output_queue=Queue(),
            batch_size=2
        )
        
        mock_create_embeddings.return_value = [[0.1], [0.2]]
        
        # Task with more chunks than batch size
        task = EmbeddingTask(
            job_id="job_1",
            chunks=["text1", "text2", "text3", "text4", "text5"],
            provider="openai",
            model="text-embedding-ada-002"
        )
        
        with patch.object(worker, '_process_batch') as mock_process_batch:
            mock_process_batch.return_value = [[0.1]] * 5
            
            worker.process_task(task)
            
            # Should process in batches
            assert mock_process_batch.call_count == 3  # 5 items with batch size 2
    
    @patch('tldw_Server_API.app.core.Embeddings.workers.embedding_worker.create_embeddings_batch')
    def test_embedding_error_recovery(self, mock_create_embeddings):
        """Test error recovery in embedding generation."""
        output_queue = Queue()
        worker = EmbeddingWorker(
            worker_id="embedding_1",
            input_queue=Queue(),
            output_queue=output_queue
        )
        
        # First call fails, second succeeds
        mock_create_embeddings.side_effect = [
            Exception("API error"),
            [[0.1, 0.2]]
        ]
        
        task = EmbeddingTask(
            job_id="job_1",
            chunks=["text1"],
            provider="openai",
            model="text-embedding-ada-002"
        )
        
        with patch('time.sleep'):  # Speed up test
            worker.process_task(task)
        
        # Should retry and succeed
        assert mock_create_embeddings.call_count == 2
        
        result = output_queue.get()
        assert result.embeddings == [[0.1, 0.2]]
    
    @patch('tldw_Server_API.app.core.Embeddings.workers.embedding_worker.create_embeddings_batch')
    def test_embedding_caching(self, mock_create_embeddings):
        """Test embedding caching functionality."""
        worker = EmbeddingWorker(
            worker_id="embedding_1",
            input_queue=Queue(),
            output_queue=Queue(),
            enable_cache=True
        )
        
        mock_create_embeddings.return_value = [[0.1, 0.2]]
        
        # Process same text twice
        task1 = EmbeddingTask(
            job_id="job_1",
            chunks=["cached_text"],
            provider="openai",
            model="text-embedding-ada-002"
        )
        
        task2 = EmbeddingTask(
            job_id="job_2",
            chunks=["cached_text"],
            provider="openai",
            model="text-embedding-ada-002"
        )
        
        worker.process_task(task1)
        worker.process_task(task2)
        
        # Should only call embedding creation once due to cache
        assert mock_create_embeddings.call_count == 1


@pytest.mark.unit
class TestStorageWorker:
    """Test storage worker functionality."""
    
    def test_storage_worker_initialization(self):
        """Test storage worker initialization."""
        worker = StorageWorker(
            worker_id="storage_1",
            input_queue=Queue(),
            db_path="/tmp/test.db",
            chroma_path="/tmp/chroma"
        )
        
        assert worker.worker_id == "storage_1"
        assert worker.db_path == "/tmp/test.db"
        assert worker.chroma_path == "/tmp/chroma"
    
    @patch('tldw_Server_API.app.core.Embeddings.workers.storage_worker.ChromaDBManager')
    @patch('tldw_Server_API.app.core.Embeddings.workers.storage_worker.MediaDatabase')
    def test_process_storage_task(self, mock_media_db, mock_chroma_manager):
        """Test processing a storage task."""
        worker = StorageWorker(
            worker_id="storage_1",
            input_queue=Queue(),
            db_path="/tmp/test.db"
        )
        
        # Setup mocks
        mock_manager_instance = MagicMock()
        mock_manager_instance.store_in_chroma.return_value = True
        mock_chroma_manager.return_value = mock_manager_instance
        
        mock_db_instance = MagicMock()
        mock_media_db.return_value = mock_db_instance
        
        # Create task
        task = StorageTask(
            job_id="job_1",
            collection_name="test_collection",
            texts=["text1", "text2"],
            embeddings=[[0.1, 0.2], [0.3, 0.4]],
            ids=["id1", "id2"],
            metadata=[{"index": 0}, {"index": 1}]
        )
        
        # Process task
        worker.process_task(task)
        
        # Verify storage
        mock_manager_instance.store_in_chroma.assert_called_once_with(
            collection_name="test_collection",
            texts=["text1", "text2"],
            embeddings=[[0.1, 0.2], [0.3, 0.4]],
            ids=["id1", "id2"],
            metadata=[{"index": 0}, {"index": 1}]
        )
    
    @patch('tldw_Server_API.app.core.Embeddings.workers.storage_worker.ChromaDBManager')
    def test_storage_with_database_update(self, mock_chroma_manager):
        """Test storage with database metadata update."""
        worker = StorageWorker(
            worker_id="storage_1",
            input_queue=Queue(),
            db_path="/tmp/test.db"
        )
        
        mock_manager_instance = MagicMock()
        mock_chroma_manager.return_value = mock_manager_instance
        
        with patch.object(worker, '_update_media_metadata') as mock_update:
            task = StorageTask(
                job_id="job_1",
                collection_name="test",
                texts=["text"],
                embeddings=[[0.1]],
                ids=["id1"],
                media_id="media_123"
            )
            
            worker.process_task(task)
            
            # Should update media metadata
            mock_update.assert_called_once_with("media_123", mock.ANY)
    
    @patch('tldw_Server_API.app.core.Embeddings.workers.storage_worker.ChromaDBManager')
    def test_storage_error_handling(self, mock_chroma_manager):
        """Test error handling in storage worker."""
        worker = StorageWorker(
            worker_id="storage_1",
            input_queue=Queue(),
            db_path="/tmp/test.db"
        )
        
        # Make storage fail
        mock_manager_instance = MagicMock()
        mock_manager_instance.store_in_chroma.side_effect = Exception("Storage failed")
        mock_chroma_manager.return_value = mock_manager_instance
        
        task = StorageTask(
            job_id="job_1",
            collection_name="test",
            texts=["text"],
            embeddings=[[0.1]],
            ids=["id1"]
        )
        
        # Should handle error gracefully
        with patch.object(worker, '_handle_storage_error') as mock_handle:
            worker.process_task(task)
            mock_handle.assert_called_once()
    
    @patch('tldw_Server_API.app.core.Embeddings.workers.storage_worker.ChromaDBManager')
    def test_storage_transaction_rollback(self, mock_chroma_manager):
        """Test transaction rollback on storage failure."""
        worker = StorageWorker(
            worker_id="storage_1",
            input_queue=Queue(),
            db_path="/tmp/test.db"
        )
        
        mock_manager_instance = MagicMock()
        mock_chroma_manager.return_value = mock_manager_instance
        
        # Simulate partial success then failure
        mock_manager_instance.store_in_chroma.side_effect = [
            True,  # First chunk succeeds
            Exception("Storage failed")  # Second chunk fails
        ]
        
        task = StorageTask(
            job_id="job_1",
            collection_name="test",
            texts=["text1", "text2"],
            embeddings=[[0.1], [0.2]],
            ids=["id1", "id2"],
            requires_transaction=True
        )
        
        with patch.object(worker, '_rollback_transaction') as mock_rollback:
            worker.process_task(task)
            
            # Should rollback on failure
            mock_rollback.assert_called_once()


@pytest.mark.unit
class TestWorkerCoordination:
    """Test coordination between multiple workers."""
    
    def test_pipeline_coordination(self):
        """Test full pipeline with all workers."""
        chunking_queue = Queue()
        embedding_queue = Queue()
        storage_queue = Queue()
        
        # Create workers
        chunking_worker = ChunkingWorker(
            worker_id="chunk_1",
            input_queue=chunking_queue,
            output_queue=embedding_queue
        )
        
        embedding_worker = EmbeddingWorker(
            worker_id="embed_1",
            input_queue=embedding_queue,
            output_queue=storage_queue
        )
        
        storage_worker = StorageWorker(
            worker_id="store_1",
            input_queue=storage_queue,
            db_path="/tmp/test.db"
        )
        
        with patch('tldw_Server_API.app.core.Embeddings.workers.chunking_worker.improved_chunking_process') as mock_chunk:
            with patch('tldw_Server_API.app.core.Embeddings.workers.embedding_worker.create_embeddings_batch') as mock_embed:
                with patch('tldw_Server_API.app.core.Embeddings.workers.storage_worker.ChromaDBManager') as mock_store:
                    
                    # Setup mocks
                    mock_chunk.return_value = ["chunk1", "chunk2"]
                    mock_embed.return_value = [[0.1], [0.2]]
                    mock_store_instance = MagicMock()
                    mock_store.return_value = mock_store_instance
                    
                    # Start workers
                    storage_worker.start()
                    embedding_worker.start()
                    chunking_worker.start()
                    
                    # Add initial task
                    chunking_task = ChunkingTask(
                        job_id="job_1",
                        content="Content to process",
                        chunk_size=500,
                        overlap=50
                    )
                    chunking_queue.put(chunking_task)
                    
                    # Wait for pipeline to complete
                    time.sleep(0.5)
                    
                    # Stop workers
                    chunking_worker.stop()
                    embedding_worker.stop()
                    storage_worker.stop()
                    
                    # Verify full pipeline executed
                    assert mock_chunk.called
                    assert mock_embed.called
                    assert mock_store_instance.store_in_chroma.called
    
    def test_worker_failure_propagation(self):
        """Test error propagation through pipeline."""
        input_queue = Queue()
        output_queue = Queue()
        error_queue = Queue()
        
        worker = ChunkingWorker(
            worker_id="chunk_1",
            input_queue=input_queue,
            output_queue=output_queue,
            error_queue=error_queue
        )
        
        with patch('tldw_Server_API.app.core.Embeddings.workers.chunking_worker.improved_chunking_process') as mock_chunk:
            mock_chunk.side_effect = Exception("Chunking failed")
            
            task = ChunkingTask(
                job_id="job_1",
                content="Content",
                chunk_size=500,
                overlap=50
            )
            
            worker.process_task(task)
            
            # Error should be queued
            assert not error_queue.empty()
            error = error_queue.get()
            assert error.job_id == "job_1"
            assert "Chunking failed" in str(error.error)
    
    def test_worker_backpressure_handling(self):
        """Test handling of queue backpressure."""
        input_queue = Queue(maxsize=2)
        output_queue = Queue(maxsize=2)
        
        worker = ChunkingWorker(
            worker_id="chunk_1",
            input_queue=input_queue,
            output_queue=output_queue
        )
        
        # Fill output queue to create backpressure
        output_queue.put("item1")
        output_queue.put("item2")
        
        with patch('tldw_Server_API.app.core.Embeddings.workers.chunking_worker.improved_chunking_process') as mock_chunk:
            mock_chunk.return_value = ["chunk1", "chunk2", "chunk3"]
            
            task = ChunkingTask(
                job_id="job_1",
                content="Content",
                chunk_size=500,
                overlap=50
            )
            
            # Start worker in thread to avoid blocking
            def process_with_timeout():
                worker.process_task(task)
            
            thread = threading.Thread(target=process_with_timeout)
            thread.start()
            
            # Should handle backpressure without deadlock
            time.sleep(0.1)
            output_queue.get()  # Free up space
            
            thread.join(timeout=1)
            assert not thread.is_alive()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])