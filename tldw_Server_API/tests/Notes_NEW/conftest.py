"""
Notes Module Test Configuration and Fixtures

Provides fixtures for testing notes functionality including
note management, keyword tagging, and search capabilities.
"""

import os
import tempfile
from pathlib import Path
from typing import Dict, Any, List, Generator, Optional
from unittest.mock import MagicMock, AsyncMock, Mock, patch
from datetime import datetime, timedelta
import uuid
import json
import sqlite3

import pytest
from fastapi.testclient import TestClient

# Import actual notes components for integration tests
from tldw_Server_API.app.core.Notes.Notes_Library import NotesInteropService
from tldw_Server_API.app.core.DB_Management.ChaChaNotes_DB import (
    CharactersRAGDB,
    CharactersRAGDBError,
    InputError,
    ConflictError
)

# =====================================================================
# Test Markers
# =====================================================================

def pytest_configure(config):
    """Register custom markers for test categorization."""
    config.addinivalue_line("markers", "unit: Unit tests with minimal mocking")
    config.addinivalue_line("markers", "integration: Integration tests with real components")
    config.addinivalue_line("markers", "property: Property-based tests")
    config.addinivalue_line("markers", "slow: Tests that take > 1 second")
    config.addinivalue_line("markers", "concurrent: Tests for concurrent operations")
    config.addinivalue_line("markers", "search: Tests for search functionality")

# =====================================================================
# Environment Configuration
# =====================================================================

@pytest.fixture(scope="session")
def test_env_vars():
    """Set up test environment variables."""
    original_env = os.environ.copy()
    
    # Set test mode
    os.environ["TEST_MODE"] = "true"
    os.environ["RATE_LIMIT_PER_MINUTE"] = "30"
    
    yield
    
    # Restore original environment
    os.environ.clear()
    os.environ.update(original_env)

# =====================================================================
# Database Fixtures
# =====================================================================

@pytest.fixture
def test_db_path() -> Generator[Path, None, None]:
    """Create a temporary database file that gets cleaned up."""
    with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as tmp_file:
        db_path = Path(tmp_file.name)
    
    yield db_path
    
    # Cleanup
    try:
        if db_path.exists():
            db_path.unlink()
    except Exception as e:
        print(f"Warning: Could not delete test database: {e}")

@pytest.fixture
def test_chacha_db(test_db_path) -> Generator[CharactersRAGDB, None, None]:
    """Create a real CharactersRAGDB instance for testing."""
    db = CharactersRAGDB(
        db_path=str(test_db_path),
        client_id="test_client"
    )
    
    # Initialize the database schema
    db.initialize_db()
    
    yield db
    
    # Cleanup
    try:
        db.close()
    except:
        pass

@pytest.fixture
def populated_chacha_db(test_chacha_db) -> CharactersRAGDB:
    """Create a CharactersRAGDB with test data."""
    db = test_chacha_db
    
    # Create test notes
    note1_id = db.create_note(
        title="Test Note 1",
        content="This is the content of test note 1.",
        user_id="test_user"
    )
    
    note2_id = db.create_note(
        title="Test Note 2",
        content="This is the content of test note 2 with more details.",
        user_id="test_user"
    )
    
    note3_id = db.create_note(
        title="Python Tutorial",
        content="This note contains information about Python programming.",
        user_id="test_user"
    )
    
    # Create test keywords
    kw1_id = db.create_keyword("python", user_id="test_user")
    kw2_id = db.create_keyword("testing", user_id="test_user")
    kw3_id = db.create_keyword("tutorial", user_id="test_user")
    
    # Link notes to keywords
    db.link_note_keyword(note3_id, kw1_id)
    db.link_note_keyword(note3_id, kw3_id)
    db.link_note_keyword(note1_id, kw2_id)
    
    return db

@pytest.fixture
def test_notes_service(test_db_path) -> Generator[NotesInteropService, None, None]:
    """Create a NotesInteropService with a real test database."""
    # Create a temporary directory for the service
    with tempfile.TemporaryDirectory() as temp_dir:
        service = NotesInteropService(
            base_db_directory=temp_dir,
            api_client_id="test_client"
        )
        
        # Create a test user database
        test_user_db = CharactersRAGDB(
            db_path=str(test_db_path),
            client_id="test_client"
        )
        test_user_db.initialize_db()
        
        # Inject the test database
        service._db_instances["test_user"] = test_user_db
        
        yield service
        
        # Cleanup
        service.close_all_user_connections()

@pytest.fixture
def mock_chacha_db():
    """Create a mock CharactersRAGDB for unit tests."""
    db = MagicMock(spec=CharactersRAGDB)
    
    # Mock note methods
    db.create_note = Mock(return_value=1)
    db.get_note = Mock(return_value={
        'id': 1,
        'title': 'Test Note',
        'content': 'Test content',
        'created_at': datetime.utcnow().isoformat(),
        'updated_at': datetime.utcnow().isoformat(),
        'version': 1,
        'is_deleted': 0
    })
    db.list_notes = Mock(return_value=[])
    db.update_note = Mock(return_value={'rows_affected': 1})
    db.delete_note = Mock(return_value={'success': True})
    db.search_notes = Mock(return_value=[])
    
    # Mock keyword methods
    db.create_keyword = Mock(return_value=1)
    db.get_keyword = Mock(return_value={
        'id': 1,
        'keyword': 'test-keyword',
        'created_at': datetime.utcnow().isoformat()
    })
    db.list_keywords = Mock(return_value=[])
    db.delete_keyword = Mock(return_value={'success': True})
    db.search_keywords = Mock(return_value=[])
    
    # Mock linking methods
    db.link_note_keyword = Mock(return_value={'success': True})
    db.unlink_note_keyword = Mock(return_value={'success': True})
    db.get_keywords_for_note = Mock(return_value=[])
    db.get_notes_for_keyword = Mock(return_value=[])
    
    return db

@pytest.fixture
def mock_notes_service(mock_chacha_db):
    """Create a NotesInteropService with mocked database for unit tests."""
    with tempfile.TemporaryDirectory() as temp_dir:
        with patch('tldw_Server_API.app.core.Notes.Notes_Library.CharactersRAGDB', return_value=mock_chacha_db):
            service = NotesInteropService(
                base_db_directory=temp_dir,
                api_client_id="test_client"
            )
            # Override the internal DB instance
            service._db_instances["test_user"] = mock_chacha_db
            yield service

# =====================================================================
# Note Data Fixtures
# =====================================================================

@pytest.fixture
def sample_note():
    """Sample note data."""
    return {
        'title': 'Test Note',
        'content': 'This is a test note with some content.',
        'user_id': 'test_user'
    }

@pytest.fixture
def sample_notes():
    """Multiple sample notes."""
    return [
        {
            'title': 'Note 1',
            'content': 'Content for note 1.',
            'user_id': 'test_user'
        },
        {
            'title': 'Note 2',
            'content': 'Content for note 2 with more details.',
            'user_id': 'test_user'
        },
        {
            'title': 'Note 3',
            'content': 'Content for note 3. This has search terms.',
            'user_id': 'test_user'
        }
    ]

@pytest.fixture
def long_note():
    """Long note for testing content limits."""
    return {
        'title': 'Long Note',
        'content': ' '.join(['This is sentence number {}.'.format(i) for i in range(1000)]),
        'user_id': 'test_user'
    }

@pytest.fixture
def markdown_note():
    """Note with markdown formatting."""
    return {
        'title': 'Markdown Note',
        'content': """# Header 1
        
## Header 2        
        
- Bullet 1
- Bullet 2
        
**Bold text** and *italic text*
        
[Link](https://example.com)
        
```python
def hello():
    print("Hello")
```
        """,
        'user_id': 'test_user'
    }

# =====================================================================
# Keyword Data Fixtures
# =====================================================================

@pytest.fixture
def sample_keyword():
    """Sample keyword data."""
    return {
        'keyword': 'test-keyword',
        'user_id': 'test_user'
    }

@pytest.fixture
def sample_keywords():
    """Multiple sample keywords."""
    return [
        {'keyword': 'python', 'user_id': 'test_user'},
        {'keyword': 'testing', 'user_id': 'test_user'},
        {'keyword': 'fastapi', 'user_id': 'test_user'},
        {'keyword': 'database', 'user_id': 'test_user'},
        {'keyword': 'api', 'user_id': 'test_user'}
    ]

@pytest.fixture
def hierarchical_keywords():
    """Keywords with hierarchical structure."""
    return [
        {'keyword': 'programming', 'user_id': 'test_user'},
        {'keyword': 'programming/python', 'user_id': 'test_user'},
        {'keyword': 'programming/python/testing', 'user_id': 'test_user'},
        {'keyword': 'programming/javascript', 'user_id': 'test_user'}
    ]

# =====================================================================
# Version Conflict Fixtures
# =====================================================================

@pytest.fixture
def version_conflict_scenario():
    """Setup for version conflict testing."""
    return {
        'note_id': 1,
        'original_version': 1,
        'user1_update': {
            'content': 'User 1 updated content',
            'version': 1
        },
        'user2_update': {
            'content': 'User 2 updated content',
            'version': 1  # Same version - should conflict
        }
    }

# =====================================================================
# Search Query Fixtures
# =====================================================================

@pytest.fixture
def search_queries():
    """Various search query patterns."""
    return {
        'simple': 'test',
        'phrase': '"exact phrase"',
        'boolean_and': 'python AND testing',
        'boolean_or': 'python OR javascript',
        'boolean_not': 'python NOT java',
        'wildcard': 'test*',
        'complex': '(python OR javascript) AND testing NOT java'
    }

# =====================================================================
# Rate Limiting Fixtures
# =====================================================================

@pytest.fixture
def rate_limiter_config():
    """Rate limiter configuration."""
    return {
        'max_requests_per_minute': 30,
        'window_size': 60,  # seconds
        'burst_size': 5
    }

@pytest.fixture
def mock_rate_limiter():
    """Mock rate limiter for testing."""
    from tldw_Server_API.app.api.v1.endpoints.notes import SimpleRateLimiter
    limiter = SimpleRateLimiter(max_requests_per_minute=30)
    return limiter

# =====================================================================
# API Request/Response Fixtures
# =====================================================================

@pytest.fixture
def note_create_request():
    """Note creation API request."""
    return {
        'title': 'API Test Note',
        'content': 'Content created via API',
        'keywords': ['api-test', 'automated']
    }

@pytest.fixture
def note_update_request():
    """Note update API request."""
    return {
        'title': 'Updated Title',
        'content': 'Updated content',
        'version': 1
    }

@pytest.fixture
def expected_note_response():
    """Expected note API response structure."""
    return {
        'id': int,
        'title': str,
        'content': str,
        'created_at': str,
        'updated_at': str,
        'version': int,
        'keywords': list
    }

# =====================================================================
# User Fixtures
# =====================================================================

@pytest.fixture
def test_users():
    """Multiple test users for isolation testing."""
    return [
        {'id': 'user1', 'name': 'Test User 1'},
        {'id': 'user2', 'name': 'Test User 2'},
        {'id': 'user3', 'name': 'Test User 3'}
    ]

@pytest.fixture
def user_headers():
    """Headers for different test users."""
    return {
        'user1': {'X-User-ID': 'user1', 'Authorization': 'Bearer token1'},
        'user2': {'X-User-ID': 'user2', 'Authorization': 'Bearer token2'},
        'user3': {'X-User-ID': 'user3', 'Authorization': 'Bearer token3'}
    }

# =====================================================================
# Performance Testing Fixtures
# =====================================================================

@pytest.fixture
def large_note_collection():
    """Generate a large collection of notes for performance testing."""
    notes = []
    for i in range(1000):
        notes.append({
            'title': f'Note {i}',
            'content': f'Content for note {i}. ' * 10,
            'user_id': 'test_user'
        })
    return notes

@pytest.fixture
def performance_metrics():
    """Track performance metrics during tests."""
    return {
        'create_times': [],
        'read_times': [],
        'update_times': [],
        'search_times': [],
        'memory_usage': []
    }

# =====================================================================
# Concurrent Operation Fixtures
# =====================================================================

@pytest.fixture
def concurrent_operations():
    """Setup for concurrent operation testing."""
    return {
        'num_threads': 5,
        'operations_per_thread': 10,
        'operation_types': ['create', 'read', 'update', 'delete'],
        'delay_between_ops': 0.01  # seconds
    }

# =====================================================================
# Error Injection Fixtures
# =====================================================================

@pytest.fixture
def error_scenarios():
    """Various error scenarios for testing."""
    return {
        'database_locked': sqlite3.OperationalError("database is locked"),
        'constraint_violation': InputError("UNIQUE constraint failed"),
        'version_conflict': ConflictError("Version mismatch"),
        'not_found': CharactersRAGDBError("Note not found"),
        'invalid_input': InputError("Invalid input data")
    }

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

# =====================================================================
# Helper Functions
# =====================================================================

def create_test_note(db, **kwargs):
    """Helper to create a test note."""
    note_data = {
        'title': 'Test Note',
        'content': 'Test content',
        'user_id': 'test_user'
    }
    note_data.update(kwargs)
    return db.create_note(**note_data)

def create_test_keyword(db, **kwargs):
    """Helper to create a test keyword."""
    keyword_data = {
        'keyword': 'test-keyword',
        'user_id': 'test_user'
    }
    keyword_data.update(kwargs)
    return db.create_keyword(**keyword_data)