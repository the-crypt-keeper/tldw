"""
Embeddings Module Test Configuration and Fixtures

Provides fixtures for testing embeddings functionality including
vector generation, worker orchestration, and ChromaDB integration.
"""

import os
import tempfile
from pathlib import Path
from typing import Dict, Any, List, Generator, Optional
from unittest.mock import MagicMock, AsyncMock, Mock
import json
import numpy as np
from datetime import datetime
import uuid

import pytest
from fastapi.testclient import TestClient
import chromadb
from chromadb.config import Settings

# Import actual embeddings components for integration tests
from tldw_Server_API.app.core.Embeddings.ChromaDB_Library import ChromaDBManager
from tldw_Server_API.app.core.Embeddings.worker_orchestrator import WorkerOrchestrator
from tldw_Server_API.app.core.Embeddings.workers.embedding_worker import EmbeddingWorker
from tldw_Server_API.app.core.Embeddings.workers.chunking_worker import ChunkingWorker
from tldw_Server_API.app.core.Embeddings.workers.storage_worker import StorageWorker
from tldw_Server_API.app.core.Embeddings.queue_schemas import (
    JobRequest,
    JobStatus,
    JobResult,
    JobType
)
from tldw_Server_API.app.core.DB_Management.Media_DB_v2 import MediaDatabase

# =====================================================================
# Test Markers
# =====================================================================

def pytest_configure(config):
    """Register custom markers for test categorization."""
    config.addinivalue_line("markers", "unit: Unit tests with minimal mocking")
    config.addinivalue_line("markers", "integration: Integration tests with real components")
    config.addinivalue_line("markers", "property: Property-based tests")
    config.addinivalue_line("markers", "slow: Tests that take > 1 second")
    config.addinivalue_line("markers", "requires_gpu: Tests requiring GPU/CUDA")
    config.addinivalue_line("markers", "worker: Worker-specific tests")

# =====================================================================
# Environment Configuration
# =====================================================================

@pytest.fixture(scope="session")
def test_env_vars():
    """Set up test environment variables."""
    original_env = os.environ.copy()
    
    # Set test mode
    os.environ["TEST_MODE"] = "true"
    os.environ["EMBEDDING_MODEL"] = "sentence-transformers/all-MiniLM-L6-v2"
    os.environ["CHROMA_BATCH_SIZE"] = "100"
    os.environ["EMBEDDING_BATCH_SIZE"] = "32"
    os.environ["MAX_WORKERS"] = "2"
    
    yield
    
    # Restore original environment
    os.environ.clear()
    os.environ.update(original_env)

# =====================================================================
# Vector/Embedding Fixtures
# =====================================================================

@pytest.fixture
def sample_embedding() -> List[float]:
    """Generate a sample embedding vector."""
    # Standard 384-dimensional vector for all-MiniLM-L6-v2
    np.random.seed(42)
    return np.random.randn(384).tolist()

@pytest.fixture
def sample_embeddings_batch() -> List[List[float]]:
    """Generate a batch of embedding vectors."""
    np.random.seed(42)
    return [np.random.randn(384).tolist() for _ in range(10)]

@pytest.fixture
def text_chunks() -> List[str]:
    """Sample text chunks for embedding."""
    return [
        "This is the first chunk of text about machine learning.",
        "The second chunk discusses natural language processing.",
        "Deep learning models require significant computational resources.",
        "Transfer learning helps reduce training time and improves accuracy.",
        "Fine-tuning pre-trained models is a common practice.",
        "Embeddings capture semantic meaning in vector space.",
        "Similarity search finds related documents using vector distance.",
        "ChromaDB is a vector database for storing embeddings.",
        "RAG systems combine retrieval and generation.",
        "Context window limits affect model performance."
    ]

@pytest.fixture
def document_metadata() -> List[Dict[str, Any]]:
    """Sample metadata for documents."""
    return [
        {"source": "doc1.pdf", "page": 1, "chunk_id": 0},
        {"source": "doc1.pdf", "page": 2, "chunk_id": 1},
        {"source": "doc2.txt", "page": 1, "chunk_id": 0},
        {"source": "doc2.txt", "page": 1, "chunk_id": 1},
        {"source": "video1.mp4", "timestamp": 120, "chunk_id": 0},
        {"source": "video1.mp4", "timestamp": 240, "chunk_id": 1},
        {"source": "article.html", "section": "intro", "chunk_id": 0},
        {"source": "article.html", "section": "main", "chunk_id": 1},
        {"source": "notes.md", "heading": "Overview", "chunk_id": 0},
        {"source": "notes.md", "heading": "Details", "chunk_id": 1}
    ]

# =====================================================================
# ChromaDB Fixtures
# =====================================================================

@pytest.fixture
def chroma_client():
    """Create an in-memory ChromaDB client for testing."""
    return chromadb.Client(Settings(
        is_persistent=False,
        anonymized_telemetry=False
    ))

@pytest.fixture
def chroma_collection(chroma_client):
    """Create a test collection in ChromaDB."""
    collection_name = f"test_collection_{uuid.uuid4().hex[:8]}"
    return chroma_client.create_collection(
        name=collection_name,
        metadata={"hnsw:space": "cosine"}
    )

@pytest.fixture
def populated_chroma_collection(chroma_collection, sample_embeddings_batch, text_chunks, document_metadata):
    """Create a ChromaDB collection with test data."""
    ids = [f"doc_{i}" for i in range(len(text_chunks))]
    
    chroma_collection.add(
        embeddings=sample_embeddings_batch,
        documents=text_chunks,
        metadatas=document_metadata,
        ids=ids
    )
    
    return chroma_collection

@pytest.fixture
def chromadb_manager(chroma_client):
    """Create a ChromaDBManager instance for testing."""
    manager = ChromaDBManager()
    manager.client = chroma_client
    manager.collection = None
    return manager

# =====================================================================
# Worker Fixtures
# =====================================================================

@pytest.fixture
def mock_embedding_worker():
    """Create a mock embedding worker."""
    worker = MagicMock(spec=EmbeddingWorker)
    worker.process = AsyncMock(return_value=np.random.randn(10, 384).tolist())
    worker.batch_size = 32
    worker.model_name = "sentence-transformers/all-MiniLM-L6-v2"
    return worker

@pytest.fixture
def mock_chunking_worker():
    """Create a mock chunking worker."""
    worker = MagicMock(spec=ChunkingWorker)
    worker.process = AsyncMock(return_value=[
        {"text": "chunk1", "metadata": {"chunk_id": 0}},
        {"text": "chunk2", "metadata": {"chunk_id": 1}}
    ])
    worker.chunk_size = 500
    worker.chunk_overlap = 50
    return worker

@pytest.fixture
def mock_storage_worker():
    """Create a mock storage worker."""
    worker = MagicMock(spec=StorageWorker)
    worker.store = AsyncMock(return_value={"status": "success", "stored_count": 10})
    worker.retrieve = AsyncMock(return_value=[])
    return worker

@pytest.fixture
def worker_orchestrator(mock_embedding_worker, mock_chunking_worker, mock_storage_worker):
    """Create a worker orchestrator with mock workers."""
    orchestrator = WorkerOrchestrator()
    orchestrator.embedding_worker = mock_embedding_worker
    orchestrator.chunking_worker = mock_chunking_worker
    orchestrator.storage_worker = mock_storage_worker
    orchestrator.is_initialized = True
    return orchestrator

# =====================================================================
# Job/Queue Fixtures
# =====================================================================

@pytest.fixture
def sample_job_request() -> JobRequest:
    """Create a sample job request."""
    return JobRequest(
        job_id=str(uuid.uuid4()),
        job_type=JobType.EMBEDDING,
        media_id=123,
        collection_name="test_collection",
        data={
            "text": "Sample text for embedding",
            "metadata": {"source": "test.txt"}
        },
        priority=5,
        created_at=datetime.utcnow()
    )

@pytest.fixture
def batch_job_requests() -> List[JobRequest]:
    """Create multiple job requests."""
    jobs = []
    for i in range(5):
        jobs.append(JobRequest(
            job_id=str(uuid.uuid4()),
            job_type=JobType.EMBEDDING if i % 2 == 0 else JobType.CHUNKING,
            media_id=100 + i,
            collection_name="test_collection",
            data={
                "text": f"Text for job {i}",
                "metadata": {"source": f"doc{i}.txt"}
            },
            priority=i,
            created_at=datetime.utcnow()
        ))
    return jobs

@pytest.fixture
def job_result() -> JobResult:
    """Create a sample job result."""
    return JobResult(
        job_id=str(uuid.uuid4()),
        status=JobStatus.COMPLETED,
        result={
            "embeddings_generated": 10,
            "chunks_processed": 5,
            "time_taken": 1.23
        },
        error=None,
        completed_at=datetime.utcnow()
    )

# =====================================================================
# Database Fixtures
# =====================================================================

@pytest.fixture
def media_database() -> Generator[MediaDatabase, None, None]:
    """Create a test media database."""
    with tempfile.TemporaryDirectory() as temp_dir:
        db_path = Path(temp_dir) / "test_media.db"
        db = MediaDatabase(db_path=str(db_path), client_id="test_client")
        db.initialize_db()
        yield db
        db.close()

@pytest.fixture
def populated_media_database(media_database) -> MediaDatabase:
    """Create a media database with test data."""
    # Add test media items
    for i in range(5):
        media_database.add_media(
            title=f"Test Document {i}",
            content=f"This is the content of document {i}. It contains various information about topic {i}.",
            media_type="document",
            author=f"Author {i}",
            metadata={
                "page_count": 10 + i,
                "word_count": 1000 + i * 100
            }
        )
    
    return media_database

# =====================================================================
# Model Configuration Fixtures
# =====================================================================

@pytest.fixture
def embedding_models():
    """Available embedding models configuration."""
    return {
        "sentence-transformers/all-MiniLM-L6-v2": {
            "dimension": 384,
            "max_tokens": 256,
            "batch_size": 32
        },
        "sentence-transformers/all-mpnet-base-v2": {
            "dimension": 768,
            "max_tokens": 384,
            "batch_size": 16
        },
        "BAAI/bge-small-en-v1.5": {
            "dimension": 384,
            "max_tokens": 512,
            "batch_size": 32
        },
        "openai/text-embedding-ada-002": {
            "dimension": 1536,
            "max_tokens": 8191,
            "batch_size": 100
        }
    }

@pytest.fixture
def chunking_strategies():
    """Available chunking strategies."""
    return {
        "fixed": {
            "chunk_size": 500,
            "chunk_overlap": 50
        },
        "sentence": {
            "sentences_per_chunk": 5,
            "min_chunk_size": 100
        },
        "semantic": {
            "similarity_threshold": 0.7,
            "max_chunk_size": 1000
        },
        "token": {
            "tokens_per_chunk": 256,
            "token_overlap": 25
        }
    }

# =====================================================================
# Mock API Responses
# =====================================================================

@pytest.fixture
def mock_openai_embedding_response():
    """Mock OpenAI embedding API response."""
    return {
        "object": "list",
        "data": [
            {
                "object": "embedding",
                "index": 0,
                "embedding": np.random.randn(1536).tolist()
            }
        ],
        "model": "text-embedding-ada-002",
        "usage": {
            "prompt_tokens": 8,
            "total_tokens": 8
        }
    }

@pytest.fixture
def mock_huggingface_embedding_response():
    """Mock HuggingFace embedding API response."""
    return np.random.randn(5, 384).tolist()  # Batch of 5 embeddings

# =====================================================================
# Performance Testing Fixtures
# =====================================================================

@pytest.fixture
def large_text_corpus() -> List[str]:
    """Generate a large corpus for performance testing."""
    base_texts = [
        "Machine learning algorithms can identify patterns in data.",
        "Natural language processing enables computers to understand text.",
        "Deep neural networks have revolutionized computer vision.",
        "Reinforcement learning trains agents through trial and error.",
        "Transfer learning leverages knowledge from pre-trained models."
    ]
    
    # Replicate and vary to create larger corpus
    corpus = []
    for i in range(100):
        for base in base_texts:
            corpus.append(f"{base} Variation {i}: {base.replace('.', f' in iteration {i}.')}")
    
    return corpus

@pytest.fixture
def performance_metrics():
    """Track performance metrics during tests."""
    return {
        "embedding_time": [],
        "storage_time": [],
        "retrieval_time": [],
        "memory_usage": [],
        "batch_processing_time": []
    }

# =====================================================================
# Circuit Breaker and Rate Limiting Fixtures
# =====================================================================

@pytest.fixture
def mock_circuit_breaker():
    """Mock circuit breaker for testing."""
    breaker = MagicMock()
    breaker.is_open = False
    breaker.call = AsyncMock(side_effect=lambda func, *args, **kwargs: func(*args, **kwargs))
    breaker.record_success = Mock()
    breaker.record_failure = Mock()
    return breaker

@pytest.fixture
def mock_rate_limiter():
    """Mock rate limiter for testing."""
    limiter = MagicMock()
    limiter.check_rate_limit = AsyncMock(return_value=True)
    limiter.consume = AsyncMock()
    limiter.reset = Mock()
    return limiter

# =====================================================================
# API Client Fixtures
# =====================================================================

@pytest.fixture
def test_client(test_env_vars):
    """Create a test client for the FastAPI app."""
    from tldw_Server_API.app.main import app
    return TestClient(app)

@pytest.fixture
def auth_headers():
    """Authentication headers for API requests."""
    return {
        "Authorization": "Bearer test-api-key",
        "Content-Type": "application/json"
    }

# =====================================================================
# Cleanup Fixtures
# =====================================================================

@pytest.fixture(autouse=True)
def cleanup_after_test():
    """Cleanup after each test."""
    yield
    # Cleanup any temporary files or resources
    import gc
    gc.collect()