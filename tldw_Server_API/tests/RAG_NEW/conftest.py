"""
Simplified RAG Module Test Configuration

Focused on testing only the unified pipeline that's actually in use.
"""

import tempfile
from pathlib import Path
from typing import Dict, Any, Generator
from unittest.mock import MagicMock, AsyncMock

import pytest

# Import actual MediaDatabase for integration tests
from tldw_Server_API.app.core.DB_Management.Media_DB_v2 import MediaDatabase

# =====================================================================
# Test Markers
# =====================================================================

def pytest_configure(config):
    """Register custom markers for test categorization."""
    config.addinivalue_line("markers", "unit: Unit tests with minimal mocking")
    config.addinivalue_line("markers", "integration: Integration tests with real components")

# =====================================================================
# Database Fixtures
# =====================================================================

@pytest.fixture
def temp_db_path() -> Generator[Path, None, None]:
    """Create a temporary database path."""
    with tempfile.TemporaryDirectory() as temp_dir:
        db_path = Path(temp_dir) / "test_media.db"
        yield db_path

@pytest.fixture
def media_database(temp_db_path) -> MediaDatabase:
    """Create a real MediaDatabase instance for testing."""
    db = MediaDatabase(
        db_path=str(temp_db_path),
        client_id="test_client"
    )
    db.initialize_db()
    return db

@pytest.fixture
def populated_media_db(media_database) -> MediaDatabase:
    """Create a MediaDatabase with test data."""
    # Add test media items
    from datetime import datetime
    from uuid import uuid4
    
    test_items = [
        {
            "media_id": str(uuid4()),
            "title": "Introduction to RAG Systems",
            "content": "Retrieval-Augmented Generation (RAG) combines large language models with external knowledge retrieval.",
            "media_type": "article",
            "author": "AI Research Team",
            "ingestion_date": datetime.now().isoformat()
        },
        {
            "media_id": str(uuid4()),
            "title": "Vector Database Tutorial",
            "content": "Vector databases are essential for semantic search in RAG systems.",
            "media_type": "video",
            "author": "Database Expert",
            "ingestion_date": datetime.now().isoformat()
        }
    ]
    
    for item in test_items:
        media_database.add_media(**item)
    
    return media_database

# =====================================================================
# Mock Fixtures for Unit Tests
# =====================================================================

@pytest.fixture
def mock_llm():
    """Mock LLM for unit tests."""
    mock_llm = AsyncMock()
    mock_llm.generate.return_value = "This is a generated response based on the retrieved context."
    return mock_llm

# =====================================================================
# RAG Configuration Fixtures
# =====================================================================

@pytest.fixture
def minimal_rag_config() -> Dict[str, Any]:
    """Minimal RAG configuration for testing."""
    return {
        "query": "What is RAG?",
        "top_k": 5
    }

@pytest.fixture
def typical_rag_config() -> Dict[str, Any]:
    """Typical production RAG configuration."""
    return {
        "query": "How does retrieval-augmented generation work?",
        "top_k": 10,
        "enable_cache": True,
        "enable_reranking": True,
        "temperature": 0.7
    }