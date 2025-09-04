"""
Unit tests for the Embedding Worker.

Tests the embedding generation worker with minimal mocking - only external
services like HuggingFace or OpenAI APIs are mocked.
"""

import pytest
from unittest.mock import Mock, AsyncMock, patch, MagicMock
import numpy as np
import asyncio
from typing import List, Dict, Any

from tldw_Server_API.app.core.Embeddings.workers.embedding_worker import EmbeddingWorker
from tldw_Server_API.app.core.Embeddings.queue_schemas import (
    JobRequest,
    JobStatus,
    JobResult,
    JobType
)

# ========================================================================
# Worker Initialization Tests
# ========================================================================

class TestEmbeddingWorkerInitialization:
    """Test embedding worker initialization and configuration."""
    
    @pytest.mark.unit
    def test_worker_initialization_default_model(self):
        """Test worker initialization with default model."""
        worker = EmbeddingWorker()
        
        assert worker is not None
        assert hasattr(worker, 'model_name')
        assert hasattr(worker, 'batch_size')
        assert worker.model_name == "sentence-transformers/all-MiniLM-L6-v2"
    
    @pytest.mark.unit
    def test_worker_initialization_custom_model(self):
        """Test worker initialization with custom model."""
        worker = EmbeddingWorker(
            model_name="BAAI/bge-small-en-v1.5",
            batch_size=64
        )
        
        assert worker.model_name == "BAAI/bge-small-en-v1.5"
        assert worker.batch_size == 64
    
    @pytest.mark.unit
    @patch('sentence_transformers.SentenceTransformer')
    def test_model_loading(self, mock_transformer):
        """Test model loading process."""
        mock_model = MagicMock()
        mock_transformer.return_value = mock_model
        
        worker = EmbeddingWorker()
        worker.initialize()
        
        mock_transformer.assert_called_once_with("sentence-transformers/all-MiniLM-L6-v2")
        assert worker.model == mock_model
    
    @pytest.mark.unit
    def test_worker_configuration_validation(self):
        """Test worker configuration validation."""
        # Test with invalid batch size
        with pytest.raises(ValueError):
            worker = EmbeddingWorker(batch_size=-1)
        
        # Test with empty model name
        with pytest.raises(ValueError):
            worker = EmbeddingWorker(model_name="")

# ========================================================================
# Embedding Generation Tests
# ========================================================================

class TestEmbeddingGeneration:
    """Test embedding generation functionality."""
    
    @pytest.mark.unit
    @patch('sentence_transformers.SentenceTransformer')
    async def test_single_text_embedding(self, mock_transformer):
        """Test embedding generation for single text."""
        mock_model = MagicMock()
        mock_model.encode.return_value = np.random.randn(384)
        mock_transformer.return_value = mock_model
        
        worker = EmbeddingWorker()
        worker.initialize()
        
        text = "This is a test sentence for embedding."
        embedding = await worker.generate_embedding(text)
        
        assert isinstance(embedding, list)
        assert len(embedding) == 384
        mock_model.encode.assert_called_once()
    
    @pytest.mark.unit
    @patch('sentence_transformers.SentenceTransformer')
    async def test_batch_text_embedding(self, mock_transformer):
        """Test batch embedding generation."""
        batch_size = 5
        mock_model = MagicMock()
        mock_model.encode.return_value = np.random.randn(batch_size, 384)
        mock_transformer.return_value = mock_model
        
        worker = EmbeddingWorker()
        worker.initialize()
        
        texts = [f"Sentence {i}" for i in range(batch_size)]
        embeddings = await worker.generate_embeddings_batch(texts)
        
        assert isinstance(embeddings, list)
        assert len(embeddings) == batch_size
        assert all(len(emb) == 384 for emb in embeddings)
    
    @pytest.mark.unit
    @patch('sentence_transformers.SentenceTransformer')
    async def test_empty_text_handling(self, mock_transformer):
        """Test handling of empty text input."""
        mock_model = MagicMock()
        mock_transformer.return_value = mock_model
        
        worker = EmbeddingWorker()
        worker.initialize()
        
        # Empty string should still generate an embedding (model dependent)
        embedding = await worker.generate_embedding("")
        
        # Should handle gracefully - either generate or raise appropriate error
        assert embedding is not None or mock_model.encode.side_effect is not None
    
    @pytest.mark.unit
    @patch('sentence_transformers.SentenceTransformer')
    async def test_long_text_truncation(self, mock_transformer):
        """Test handling of text exceeding max tokens."""
        mock_model = MagicMock()
        mock_model.encode.return_value = np.random.randn(384)
        mock_model.max_seq_length = 256
        mock_transformer.return_value = mock_model
        
        worker = EmbeddingWorker()
        worker.initialize()
        
        # Create text that would exceed token limit
        long_text = " ".join(["word"] * 1000)
        embedding = await worker.generate_embedding(long_text)
        
        assert isinstance(embedding, list)
        assert len(embedding) == 384
        # Text should be truncated internally

# ========================================================================
# Job Processing Tests
# ========================================================================

class TestJobProcessing:
    """Test job processing functionality."""
    
    @pytest.mark.unit
    @patch('sentence_transformers.SentenceTransformer')
    async def test_process_embedding_job(self, mock_transformer, sample_job_request):
        """Test processing a single embedding job."""
        mock_model = MagicMock()
        mock_model.encode.return_value = np.random.randn(384)
        mock_transformer.return_value = mock_model
        
        worker = EmbeddingWorker()
        worker.initialize()
        
        sample_job_request.job_type = JobType.EMBEDDING
        sample_job_request.data = {"text": "Process this text"}
        
        result = await worker.process_job(sample_job_request)
        
        assert result.status == JobStatus.COMPLETED
        assert "embedding" in result.result
        assert len(result.result["embedding"]) == 384
    
    @pytest.mark.unit
    @patch('sentence_transformers.SentenceTransformer')
    async def test_process_batch_job(self, mock_transformer, batch_job_requests):
        """Test processing multiple jobs in batch."""
        mock_model = MagicMock()
        mock_model.encode.return_value = np.random.randn(5, 384)
        mock_transformer.return_value = mock_model
        
        worker = EmbeddingWorker()
        worker.initialize()
        
        # Filter for embedding jobs only
        embedding_jobs = [j for j in batch_job_requests if j.job_type == JobType.EMBEDDING]
        
        results = await worker.process_batch(embedding_jobs)
        
        assert len(results) == len(embedding_jobs)
        assert all(r.status == JobStatus.COMPLETED for r in results)
    
    @pytest.mark.unit
    @patch('sentence_transformers.SentenceTransformer')
    async def test_job_error_handling(self, mock_transformer, sample_job_request):
        """Test error handling during job processing."""
        mock_model = MagicMock()
        mock_model.encode.side_effect = Exception("Model error")
        mock_transformer.return_value = mock_model
        
        worker = EmbeddingWorker()
        worker.initialize()
        
        result = await worker.process_job(sample_job_request)
        
        assert result.status == JobStatus.FAILED
        assert result.error is not None
        assert "Model error" in result.error

# ========================================================================
# Performance and Optimization Tests
# ========================================================================

class TestPerformanceOptimization:
    """Test performance optimizations in the worker."""
    
    @pytest.mark.unit
    @patch('sentence_transformers.SentenceTransformer')
    async def test_batch_size_optimization(self, mock_transformer):
        """Test dynamic batch size optimization."""
        mock_model = MagicMock()
        mock_transformer.return_value = mock_model
        
        worker = EmbeddingWorker(batch_size=32)
        worker.initialize()
        
        # Test with various input sizes
        small_batch = ["text"] * 5
        medium_batch = ["text"] * 32
        large_batch = ["text"] * 100
        
        # Worker should handle different sizes efficiently
        mock_model.encode.return_value = np.random.randn(5, 384)
        await worker.generate_embeddings_batch(small_batch)
        
        mock_model.encode.return_value = np.random.randn(32, 384)
        await worker.generate_embeddings_batch(medium_batch)
        
        # Large batch should be processed in chunks
        mock_model.encode.return_value = np.random.randn(32, 384)
        await worker.generate_embeddings_batch(large_batch)
        
        # Should have been called multiple times for large batch
        assert mock_model.encode.call_count >= 3
    
    @pytest.mark.unit
    @patch('sentence_transformers.SentenceTransformer')
    async def test_memory_efficient_processing(self, mock_transformer, large_text_corpus):
        """Test memory-efficient processing of large corpus."""
        mock_model = MagicMock()
        
        def mock_encode(texts, *args, **kwargs):
            # Simulate memory-efficient encoding
            batch_size = len(texts) if isinstance(texts, list) else 1
            return np.random.randn(batch_size, 384) if batch_size > 1 else np.random.randn(384)
        
        mock_model.encode.side_effect = mock_encode
        mock_transformer.return_value = mock_model
        
        worker = EmbeddingWorker(batch_size=32)
        worker.initialize()
        
        # Process large corpus
        embeddings = []
        for i in range(0, len(large_text_corpus), worker.batch_size):
            batch = large_text_corpus[i:i + worker.batch_size]
            batch_embeddings = await worker.generate_embeddings_batch(batch)
            embeddings.extend(batch_embeddings)
        
        assert len(embeddings) == len(large_text_corpus)
        assert all(len(emb) == 384 for emb in embeddings)
    
    @pytest.mark.unit
    async def test_concurrent_job_processing(self):
        """Test concurrent processing of multiple jobs."""
        with patch('sentence_transformers.SentenceTransformer') as mock_transformer:
            mock_model = MagicMock()
            mock_model.encode.return_value = np.random.randn(384)
            mock_transformer.return_value = mock_model
            
            worker = EmbeddingWorker()
            worker.initialize()
            
            # Create multiple concurrent tasks
            tasks = []
            for i in range(10):
                text = f"Concurrent text {i}"
                task = asyncio.create_task(worker.generate_embedding(text))
                tasks.append(task)
            
            results = await asyncio.gather(*tasks)
            
            assert len(results) == 10
            assert all(len(r) == 384 for r in results)

# ========================================================================
# Model-Specific Tests
# ========================================================================

class TestModelSpecificBehavior:
    """Test model-specific behaviors and configurations."""
    
    @pytest.mark.unit
    @patch('sentence_transformers.SentenceTransformer')
    def test_minilm_model_configuration(self, mock_transformer):
        """Test MiniLM model specific configuration."""
        mock_model = MagicMock()
        mock_model.get_sentence_embedding_dimension.return_value = 384
        mock_transformer.return_value = mock_model
        
        worker = EmbeddingWorker(model_name="sentence-transformers/all-MiniLM-L6-v2")
        worker.initialize()
        
        assert worker.embedding_dimension == 384
        assert worker.model_name == "sentence-transformers/all-MiniLM-L6-v2"
    
    @pytest.mark.unit
    @patch('sentence_transformers.SentenceTransformer')
    def test_mpnet_model_configuration(self, mock_transformer):
        """Test MPNet model specific configuration."""
        mock_model = MagicMock()
        mock_model.get_sentence_embedding_dimension.return_value = 768
        mock_transformer.return_value = mock_model
        
        worker = EmbeddingWorker(model_name="sentence-transformers/all-mpnet-base-v2")
        worker.initialize()
        
        assert worker.embedding_dimension == 768
    
    @pytest.mark.unit
    @patch('httpx.AsyncClient.post')
    async def test_openai_embedding_fallback(self, mock_post):
        """Test fallback to OpenAI embeddings."""
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "data": [{"embedding": np.random.randn(1536).tolist()}]
        }
        mock_post.return_value = mock_response
        
        worker = EmbeddingWorker(
            model_name="openai/text-embedding-ada-002",
            use_api=True,
            api_key="test-key"
        )
        
        embedding = await worker.generate_embedding("Test text")
        
        assert len(embedding) == 1536
        mock_post.assert_called_once()

# ========================================================================
# Error Recovery Tests
# ========================================================================

class TestErrorRecovery:
    """Test error recovery mechanisms."""
    
    @pytest.mark.unit
    @patch('sentence_transformers.SentenceTransformer')
    async def test_retry_on_transient_error(self, mock_transformer):
        """Test retry logic for transient errors."""
        mock_model = MagicMock()
        call_count = 0
        
        def mock_encode(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise RuntimeError("Transient error")
            return np.random.randn(384)
        
        mock_model.encode.side_effect = mock_encode
        mock_transformer.return_value = mock_model
        
        worker = EmbeddingWorker(max_retries=3)
        worker.initialize()
        
        embedding = await worker.generate_embedding("Test text")
        
        assert embedding is not None
        assert call_count == 3
    
    @pytest.mark.unit
    @patch('sentence_transformers.SentenceTransformer')
    async def test_fallback_on_model_failure(self, mock_transformer):
        """Test fallback to alternative model on failure."""
        # Primary model fails
        primary_model = MagicMock()
        primary_model.encode.side_effect = Exception("Primary model failed")
        
        # Fallback model works
        fallback_model = MagicMock()
        fallback_model.encode.return_value = np.random.randn(384)
        
        mock_transformer.side_effect = [primary_model, fallback_model]
        
        worker = EmbeddingWorker(
            model_name="primary-model",
            fallback_model="fallback-model"
        )
        worker.initialize()
        
        embedding = await worker.generate_embedding("Test text")
        
        assert embedding is not None
        assert len(embedding) == 384
    
    @pytest.mark.unit
    @patch('sentence_transformers.SentenceTransformer')
    async def test_graceful_degradation(self, mock_transformer):
        """Test graceful degradation when resources are limited."""
        mock_model = MagicMock()
        
        def mock_encode(texts, *args, **kwargs):
            # Simulate resource constraints
            if len(texts) > 16:
                raise MemoryError("Batch too large")
            return np.random.randn(len(texts), 384)
        
        mock_model.encode.side_effect = mock_encode
        mock_transformer.return_value = mock_model
        
        worker = EmbeddingWorker(batch_size=32)
        worker.initialize()
        
        # Should automatically reduce batch size
        texts = ["text"] * 32
        embeddings = await worker.generate_embeddings_batch(texts)
        
        assert len(embeddings) == 32
        # Should have processed in smaller batches