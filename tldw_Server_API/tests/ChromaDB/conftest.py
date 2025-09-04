"""
ChromaDB Test Configuration and Fixtures

This module provides comprehensive fixtures for testing ChromaDB functionality,
including both unit tests (with mocking) and integration tests (with real ChromaDB).
"""

import os
import shutil
import tempfile
import uuid
from pathlib import Path
from typing import Dict, List, Any, Generator, Optional
from unittest.mock import Mock, MagicMock, AsyncMock, patch
import json
import sqlite3
from queue import Queue

import pytest
import chromadb
from chromadb.config import Settings
from chromadb.api import API
import numpy as np

# Import the MediaDatabase class for proper database creation
from tldw_Server_API.app.core.DB_Management.Media_DB_v2 import MediaDatabase

# Import the modules we're testing
from tldw_Server_API.app.core.Embeddings.ChromaDB_Library import ChromaDBManager
from tldw_Server_API.app.core.Embeddings.Embeddings_Create import (
    HuggingFaceEmbedder,
    ONNXEmbedder,
    create_embeddings_batch
)
from tldw_Server_API.app.core.Embeddings.job_manager import EmbeddingJobManager
from tldw_Server_API.app.core.Embeddings.worker_orchestrator import WorkerOrchestrator
from tldw_Server_API.app.core.Embeddings.queue_schemas import (
    ChunkingTask,
    EmbeddingTask, 
    StorageTask,
    JobStatus
)
from tldw_Server_API.app.core.Embeddings.workers.base_worker import BaseWorker
from tldw_Server_API.app.core.Embeddings.workers.embedding_worker import EmbeddingWorker
from tldw_Server_API.app.core.Embeddings.workers.chunking_worker import ChunkingWorker
from tldw_Server_API.app.core.Embeddings.workers.storage_worker import StorageWorker
from tldw_Server_API.app.core.Embeddings.connection_pool import ConnectionPool
from tldw_Server_API.app.core.Embeddings.error_recovery import CircuitBreaker, ErrorRecoveryManager
from tldw_Server_API.app.core.Embeddings.audit_logger import AuditLogger

# =====================================================================
# Test Markers
# =====================================================================

def pytest_configure(config):
    """Register custom markers for test categorization."""
    config.addinivalue_line("markers", "unit: Unit tests with mocking")
    config.addinivalue_line("markers", "integration: Integration tests with real ChromaDB")
    config.addinivalue_line("markers", "property: Property-based tests")
    config.addinivalue_line("markers", "slow: Tests that take > 1 second")
    config.addinivalue_line("markers", "requires_model: Tests that need ML models")
    config.addinivalue_line("markers", "concurrent: Tests with concurrent operations")

# =====================================================================
# Database Fixtures using MediaDatabase
# =====================================================================

@pytest.fixture
def temp_media_db():
    """Create a temporary MediaDatabase instance for testing."""
    # Create temporary directory for database
    db_dir = tempfile.mkdtemp()
    db_path = os.path.join(db_dir, "test_media.db")
    
    # Initialize MediaDatabase with test client ID
    db = MediaDatabase(
        db_path=db_path,
        client_id="test_client"
    )
    
    # Initialize the database schema
    db.initialize_db()
    
    yield db_path
    
    # Cleanup
    try:
        db.close_connection()
    except:
        pass
    shutil.rmtree(db_dir, ignore_errors=True)

@pytest.fixture
def media_database(temp_media_db):
    """Create a MediaDatabase instance for testing."""
    db = MediaDatabase(
        db_path=temp_media_db,
        client_id="test_client"
    )
    yield db
    db.close_connection()

# =====================================================================
# ChromaDB Fixtures
# =====================================================================

@pytest.fixture
def temp_chroma_path():
    """Create a temporary directory for ChromaDB storage."""
    temp_dir = tempfile.mkdtemp(prefix="chroma_test_")
    yield temp_dir
    shutil.rmtree(temp_dir, ignore_errors=True)

@pytest.fixture
def chroma_client(temp_chroma_path):
    """Create a ChromaDB client for integration tests."""
    settings = Settings(
        persist_directory=temp_chroma_path,
        anonymized_telemetry=False,
        allow_reset=True
    )
    client = chromadb.PersistentClient(path=temp_chroma_path, settings=settings)
    yield client
    # Cleanup handled by temp_chroma_path fixture

@pytest.fixture
def mock_chroma_client():
    """Create a mock ChromaDB client for unit tests."""
    mock_client = MagicMock()
    mock_collection = MagicMock()
    
    # Setup collection behavior
    mock_collection.count.return_value = 0
    mock_collection.add.return_value = None
    mock_collection.query.return_value = {
        "ids": [["test_id_1", "test_id_2"]],
        "embeddings": None,
        "documents": [["Test document 1", "Test document 2"]],
        "metadatas": [[{"source": "test"}, {"source": "test"}]],
        "distances": [[0.1, 0.2]]
    }
    mock_collection.delete.return_value = None
    mock_collection.get.return_value = {
        "ids": ["test_id_1"],
        "embeddings": None,
        "documents": ["Test document 1"],
        "metadatas": [{"source": "test"}]
    }
    
    mock_client.get_or_create_collection.return_value = mock_collection
    mock_client.list_collections.return_value = []
    mock_client.delete_collection.return_value = None
    mock_client.reset.return_value = None
    
    return mock_client

@pytest.fixture
def chromadb_manager(mock_chroma_client, temp_media_db):
    """Create a ChromaDBManager instance with mocked dependencies."""
    with patch('tldw_Server_API.app.core.Embeddings.ChromaDB_Library.chromadb') as mock_chromadb:
        mock_chromadb.PersistentClient.return_value = mock_chroma_client
        manager = ChromaDBManager(
            user_id="test_user",
            user_embedding_config={"provider": "openai", "model": "text-embedding-ada-002"}
        )
        manager.db_path = temp_media_db
        yield manager

@pytest.fixture
def real_chromadb_manager(chroma_client, temp_media_db):
    """Create a ChromaDBManager instance with real ChromaDB for integration tests."""
    with patch.object(ChromaDBManager, '_initialize_client') as mock_init:
        mock_init.return_value = chroma_client
        manager = ChromaDBManager(
            user_id="test_user",
            user_embedding_config={"provider": "huggingface", "model": "sentence-transformers/all-MiniLM-L6-v2"}
        )
        manager.client = chroma_client
        manager.db_path = temp_media_db
        yield manager

# =====================================================================
# Embedding Fixtures
# =====================================================================

@pytest.fixture
def mock_embeddings():
    """Generate mock embeddings for testing."""
    def _create_embeddings(texts: List[str], dim: int = 384) -> List[List[float]]:
        """Create deterministic mock embeddings based on text content."""
        embeddings = []
        for text in texts:
            # Create a deterministic embedding based on text hash
            seed = hash(text) % 10000
            np.random.seed(seed)
            embedding = np.random.randn(dim).tolist()
            # Normalize to unit vector
            norm = np.linalg.norm(embedding)
            if norm > 0:
                embedding = (np.array(embedding) / norm).tolist()
            embeddings.append(embedding)
        return embeddings
    return _create_embeddings

@pytest.fixture
def mock_embedding_provider(mock_embeddings):
    """Mock embedding provider for unit tests."""
    mock_provider = MagicMock()
    mock_provider.create_embeddings = Mock(side_effect=lambda texts: mock_embeddings(texts))
    mock_provider.embedding_dimension = 384
    mock_provider.max_batch_size = 100
    return mock_provider

@pytest.fixture
def huggingface_embedder():
    """Create a HuggingFace embedder for testing."""
    embedder = HuggingFaceEmbedder(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        device="cpu",
        auto_unload_time=60
    )
    yield embedder
    # Ensure model is unloaded
    if hasattr(embedder, 'model') and embedder.model is not None:
        embedder.unload_model()

# =====================================================================
# Worker and Job Fixtures
# =====================================================================

@pytest.fixture
def job_manager(temp_media_db):
    """Create an EmbeddingJobManager for testing."""
    manager = EmbeddingJobManager(db_path=temp_media_db)
    yield manager
    # Cleanup
    manager.cleanup()

@pytest.fixture
def mock_job_manager():
    """Create a mock job manager for unit tests."""
    mock_manager = MagicMock(spec=EmbeddingJobManager)
    mock_manager.create_job.return_value = "test_job_id"
    mock_manager.get_job_status.return_value = JobStatus(
        job_id="test_job_id",
        status="pending",
        progress=0,
        created_at="2024-01-01T00:00:00"
    )
    mock_manager.cancel_job.return_value = True
    mock_manager.get_user_jobs.return_value = []
    mock_manager.get_queue_stats.return_value = {
        "pending": 0,
        "processing": 0,
        "completed": 0,
        "failed": 0
    }
    return mock_manager

@pytest.fixture
def worker_config():
    """Worker configuration for testing."""
    return {
        "chunking_workers": 2,
        "embedding_workers": 2,
        "storage_workers": 1,
        "max_queue_size": 100,
        "batch_size": 10,
        "auto_scale": False
    }

@pytest.fixture
def mock_queue():
    """Create a mock queue for worker testing."""
    return Queue()

@pytest.fixture
def chunking_worker(mock_queue):
    """Create a chunking worker for testing."""
    output_queue = Queue()
    worker = ChunkingWorker(
        worker_id="chunking_test_1",
        input_queue=mock_queue,
        output_queue=output_queue
    )
    yield worker
    worker.stop()

@pytest.fixture
def embedding_worker(mock_queue, mock_embedding_provider):
    """Create an embedding worker for testing."""
    output_queue = Queue()
    with patch('tldw_Server_API.app.core.Embeddings.workers.embedding_worker.create_embeddings_batch') as mock_create:
        mock_create.return_value = [[0.1] * 384]
        worker = EmbeddingWorker(
            worker_id="embedding_test_1",
            input_queue=mock_queue,
            output_queue=output_queue,
            embedding_provider=mock_embedding_provider
        )
        yield worker
        worker.stop()

@pytest.fixture
def storage_worker(mock_queue, mock_chroma_client, temp_media_db):
    """Create a storage worker for testing."""
    with patch('tldw_Server_API.app.core.Embeddings.workers.storage_worker.ChromaDBManager') as mock_manager_class:
        mock_instance = MagicMock()
        mock_instance.store_in_chroma.return_value = True
        mock_manager_class.return_value = mock_instance
        
        worker = StorageWorker(
            worker_id="storage_test_1",
            input_queue=mock_queue,
            chroma_client=mock_chroma_client,
            db_path=temp_media_db
        )
        yield worker
        worker.stop()

# =====================================================================
# Error Recovery Fixtures
# =====================================================================

@pytest.fixture
def circuit_breaker():
    """Create a circuit breaker for testing."""
    return CircuitBreaker(
        failure_threshold=3,
        recovery_timeout=1.0,
        half_open_attempts=1
    )

@pytest.fixture
def error_recovery_manager():
    """Create an error recovery manager for testing."""
    return ErrorRecoveryManager(
        max_retries=3,
        base_delay=0.1,
        max_delay=1.0
    )

# =====================================================================
# Test Data Fixtures
# =====================================================================

@pytest.fixture
def sample_texts():
    """Sample texts for testing."""
    return [
        "The quick brown fox jumps over the lazy dog.",
        "Machine learning is transforming how we process information.",
        "ChromaDB provides efficient vector similarity search.",
        "Testing is essential for maintaining code quality.",
        "Python is a versatile programming language."
    ]

@pytest.fixture
def sample_documents():
    """Sample documents with metadata for testing."""
    return [
        {
            "id": "doc_1",
            "text": "Introduction to vector databases and their applications.",
            "metadata": {"source": "tutorial", "chapter": 1, "author": "TestAuthor"}
        },
        {
            "id": "doc_2",
            "text": "ChromaDB architecture and performance optimization.",
            "metadata": {"source": "documentation", "version": "1.0", "tags": ["chromadb", "performance"]}
        },
        {
            "id": "doc_3",
            "text": "Best practices for embedding generation and storage.",
            "metadata": {"source": "guide", "difficulty": "intermediate", "updated": "2024-01-01"}
        }
    ]

@pytest.fixture
def sample_media_content():
    """Sample media content for end-to-end testing."""
    return {
        "media_id": str(uuid.uuid4()),
        "title": "Test Media Content",
        "content": """This is a comprehensive test document for ChromaDB integration.
        It contains multiple paragraphs to test chunking functionality.
        
        The document includes various topics to test semantic search:
        - Vector databases and their importance
        - Machine learning applications
        - Natural language processing
        - Information retrieval systems
        
        This content should be properly chunked and embedded for testing.""",
        "author": "Test Suite",
        "media_type": "document",
        "metadata": {
            "test": True,
            "version": "1.0",
            "tags": ["test", "chromadb", "embeddings"]
        }
    }

@pytest.fixture
def chunking_task():
    """Sample chunking task for worker testing."""
    return ChunkingTask(
        job_id="test_job_1",
        content="This is a test content that needs to be chunked into smaller pieces for processing.",
        chunk_size=50,
        overlap=10,
        metadata={"source": "test"}
    )

@pytest.fixture
def embedding_task(sample_texts):
    """Sample embedding task for worker testing."""
    return EmbeddingTask(
        job_id="test_job_1",
        chunks=sample_texts,
        provider="openai",
        model="text-embedding-ada-002",
        metadata={"source": "test"}
    )

@pytest.fixture
def storage_task(sample_texts, mock_embeddings):
    """Sample storage task for worker testing."""
    embeddings = mock_embeddings(sample_texts)
    return StorageTask(
        job_id="test_job_1",
        collection_name="test_collection",
        texts=sample_texts,
        embeddings=embeddings,
        ids=[f"chunk_{i}" for i in range(len(sample_texts))],
        metadata=[{"chunk_index": i} for i in range(len(sample_texts))]
    )

# =====================================================================
# Connection Pool Fixtures
# =====================================================================

@pytest.fixture
def connection_pool(temp_media_db):
    """Create a connection pool for testing."""
    pool = ConnectionPool(
        db_path=temp_media_db,
        min_connections=1,
        max_connections=5,
        timeout=1.0
    )
    yield pool
    pool.close()

# =====================================================================
# Audit and Metrics Fixtures
# =====================================================================

@pytest.fixture
def audit_logger(temp_media_db):
    """Create an audit logger for testing."""
    return AuditLogger(db_path=temp_media_db)

@pytest.fixture
def mock_metrics():
    """Mock metrics collector for testing."""
    mock_metrics = MagicMock()
    mock_metrics.record_embedding_generation.return_value = None
    mock_metrics.record_search_query.return_value = None
    mock_metrics.record_storage_operation.return_value = None
    mock_metrics.record_error.return_value = None
    return mock_metrics

# =====================================================================
# Configuration Fixtures
# =====================================================================

@pytest.fixture
def test_config():
    """Test configuration for ChromaDB module."""
    return {
        "chroma_base_path": "/tmp/chroma_test",
        "embedding_providers": {
            "openai": {
                "api_key": "test_key",
                "model": "text-embedding-ada-002",
                "dimension": 1536
            },
            "huggingface": {
                "model": "sentence-transformers/all-MiniLM-L6-v2",
                "device": "cpu",
                "dimension": 384
            }
        },
        "cache_config": {
            "enabled": True,
            "ttl": 3600,
            "max_size": 1000
        },
        "resource_limits": {
            "max_memory_gb": 2,
            "max_collections_per_user": 10,
            "max_items_per_collection": 100000
        },
        "security": {
            "validate_user_ids": True,
            "sanitize_inputs": True,
            "audit_logging": True
        }
    }

# =====================================================================
# Utility Functions
# =====================================================================

@pytest.fixture
def create_test_collection():
    """Factory fixture for creating test collections."""
    def _create(client, name="test_collection", metadata=None):
        """Create a test collection with optional metadata."""
        return client.get_or_create_collection(
            name=name,
            metadata=metadata or {"test": True}
        )
    return _create

@pytest.fixture
def populate_collection():
    """Factory fixture for populating collections with test data."""
    def _populate(collection, num_items=10, embedding_dim=384):
        """Populate a collection with test data."""
        ids = [f"item_{i}" for i in range(num_items)]
        texts = [f"Test document {i} with some content" for i in range(num_items)]
        embeddings = np.random.randn(num_items, embedding_dim).tolist()
        metadatas = [{"index": i, "test": True} for i in range(num_items)]
        
        collection.add(
            ids=ids,
            documents=texts,
            embeddings=embeddings,
            metadatas=metadatas
        )
        return ids, texts, embeddings, metadatas
    return _populate

@pytest.fixture
def assert_embeddings_similar():
    """Utility to assert embeddings are similar within tolerance."""
    def _assert(emb1: List[float], emb2: List[float], tolerance: float = 0.01):
        """Assert two embeddings are similar."""
        assert len(emb1) == len(emb2), f"Embedding dimensions don't match: {len(emb1)} vs {len(emb2)}"
        
        # Calculate cosine similarity
        dot_product = sum(a * b for a, b in zip(emb1, emb2))
        norm1 = sum(a * a for a in emb1) ** 0.5
        norm2 = sum(b * b for b in emb2) ** 0.5
        
        if norm1 > 0 and norm2 > 0:
            similarity = dot_product / (norm1 * norm2)
            assert similarity > (1 - tolerance), f"Embeddings not similar enough: {similarity}"
    return _assert

# =====================================================================
# Cleanup and Resource Management
# =====================================================================

@pytest.fixture(autouse=True)
def cleanup_environment():
    """Automatically cleanup test environment after each test."""
    yield
    # Cleanup any temporary files or processes
    import gc
    gc.collect()

@pytest.fixture(scope="session")
def test_session_cleanup():
    """Session-level cleanup for all ChromaDB tests."""
    yield
    # Final cleanup
    temp_dirs = Path("/tmp").glob("chroma_test_*")
    for temp_dir in temp_dirs:
        try:
            shutil.rmtree(temp_dir)
        except:
            pass