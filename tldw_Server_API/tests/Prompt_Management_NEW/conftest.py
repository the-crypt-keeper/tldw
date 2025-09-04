"""
Prompt Management Module Test Configuration and Fixtures

Provides fixtures for testing prompt management functionality including
prompt templates, versions, import/export, and the job system.
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

# Import actual prompt management components for integration tests
from tldw_Server_API.app.core.Prompt_Management.Prompts_Interop import PromptsInteropService
from tldw_Server_API.app.core.DB_Management.Prompts_DB_V2 import PromptsDB
from tldw_Server_API.app.core.DB_Management.Job_System import JobSystem

# =====================================================================
# Test Markers
# =====================================================================

def pytest_configure(config):
    """Register custom markers for test categorization."""
    config.addinivalue_line("markers", "unit: Unit tests with minimal mocking")
    config.addinivalue_line("markers", "integration: Integration tests with real components")
    config.addinivalue_line("markers", "property: Property-based tests")
    config.addinivalue_line("markers", "slow: Tests that take > 1 second")
    config.addinivalue_line("markers", "job_system: Tests for the job system")
    config.addinivalue_line("markers", "import_export: Tests for import/export functionality")

# =====================================================================
# Environment Configuration
# =====================================================================

@pytest.fixture(scope="session")
def test_env_vars():
    """Set up test environment variables."""
    original_env = os.environ.copy()
    
    # Set test mode
    os.environ["TEST_MODE"] = "true"
    os.environ["PROMPTS_DB_PATH"] = ":memory:"
    os.environ["MAX_PROMPT_LENGTH"] = "10000"
    
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
def prompts_db(test_db_path) -> Generator[PromptsDB, None, None]:
    """Create a real PromptsDB instance for testing."""
    db = PromptsDB(db_path=str(test_db_path))
    db.initialize_db()
    
    yield db
    
    # Cleanup
    try:
        db.close()
    except:
        pass

@pytest.fixture
def populated_prompts_db(prompts_db) -> PromptsDB:
    """Create a PromptsDB with test data."""
    db = prompts_db
    
    # Create test prompts
    prompt1_id = db.create_prompt(
        name="Test Prompt 1",
        content="You are a helpful assistant. {{user_input}}",
        author="test_user",
        keywords=["test", "assistant"]
    )
    
    prompt2_id = db.create_prompt(
        name="Summarization Prompt",
        content="Summarize the following text: {{text}}",
        author="test_user",
        keywords=["summarization", "text"]
    )
    
    prompt3_id = db.create_prompt(
        name="Code Review Prompt",
        content="Review this code for bugs: {{code}}",
        author="dev_user",
        keywords=["code", "review", "debugging"]
    )
    
    # Create versions for first prompt
    db.update_prompt(
        prompt_id=prompt1_id,
        content="You are a helpful AI assistant. {{user_input}}",
        version_comment="Improved clarity"
    )
    
    return db

@pytest.fixture
def prompts_service(test_db_path) -> Generator[PromptsInteropService, None, None]:
    """Create a PromptsInteropService with a real test database."""
    with tempfile.TemporaryDirectory() as temp_dir:
        service = PromptsInteropService(
            db_directory=temp_dir,
            client_id="test_client"
        )
        
        # Create a test database
        test_db = PromptsDB(db_path=str(test_db_path))
        test_db.initialize_db()
        
        # Inject the test database
        service._db_instance = test_db
        
        yield service
        
        # Cleanup
        service.close()

@pytest.fixture
def mock_prompts_db():
    """Create a mock PromptsDB for unit tests."""
    db = MagicMock(spec=PromptsDB)
    
    # Mock prompt methods
    db.create_prompt = Mock(return_value=1)
    db.get_prompt = Mock(return_value={
        'id': 1,
        'name': 'Test Prompt',
        'content': 'Test content {{variable}}',
        'author': 'test_user',
        'created_at': datetime.utcnow().isoformat(),
        'updated_at': datetime.utcnow().isoformat(),
        'version': 1,
        'is_deleted': 0
    })
    db.list_prompts = Mock(return_value=[])
    db.update_prompt = Mock(return_value={'success': True})
    db.delete_prompt = Mock(return_value={'success': True})
    db.search_prompts = Mock(return_value=[])
    
    # Mock version methods
    db.get_prompt_versions = Mock(return_value=[])
    db.restore_version = Mock(return_value={'success': True})
    
    return db

@pytest.fixture
def mock_prompts_service(mock_prompts_db):
    """Create a PromptsInteropService with mocked database for unit tests."""
    with tempfile.TemporaryDirectory() as temp_dir:
        with patch('tldw_Server_API.app.core.Prompt_Management.Prompts_Interop.PromptsDB', return_value=mock_prompts_db):
            service = PromptsInteropService(
                db_directory=temp_dir,
                client_id="test_client"
            )
            service._db_instance = mock_prompts_db
            yield service

# =====================================================================
# Job System Fixtures
# =====================================================================

@pytest.fixture
def job_system(test_db_path) -> Generator[JobSystem, None, None]:
    """Create a JobSystem instance for testing."""
    job_sys = JobSystem(db_path=str(test_db_path))
    job_sys.initialize_db()
    
    yield job_sys
    
    # Cleanup
    try:
        job_sys.close()
    except:
        pass

@pytest.fixture
def mock_job_system():
    """Create a mock JobSystem for unit tests."""
    job_sys = MagicMock(spec=JobSystem)
    
    job_sys.create_job = Mock(return_value="job-123")
    job_sys.get_job = Mock(return_value={
        'id': 'job-123',
        'type': 'prompt_processing',
        'status': 'pending',
        'created_at': datetime.utcnow().isoformat()
    })
    job_sys.update_job_status = Mock(return_value={'success': True})
    job_sys.list_jobs = Mock(return_value=[])
    
    return job_sys

# =====================================================================
# Prompt Data Fixtures
# =====================================================================

@pytest.fixture
def sample_prompt():
    """Sample prompt data."""
    return {
        'name': 'Test Prompt',
        'content': 'You are a {{role}}. Please {{task}}.',
        'author': 'test_user',
        'keywords': ['test', 'template'],
        'description': 'A test prompt template'
    }

@pytest.fixture
def sample_prompts():
    """Multiple sample prompts."""
    return [
        {
            'name': 'Assistant Prompt',
            'content': 'You are a helpful assistant.',
            'author': 'test_user',
            'keywords': ['assistant', 'general']
        },
        {
            'name': 'Code Helper',
            'content': 'Help debug this code: {{code}}',
            'author': 'dev_user',
            'keywords': ['code', 'debugging']
        },
        {
            'name': 'Writing Assistant',
            'content': 'Improve this text: {{text}}',
            'author': 'writer_user',
            'keywords': ['writing', 'editing']
        }
    ]

@pytest.fixture
def complex_prompt():
    """Complex prompt with multiple variables and sections."""
    return {
        'name': 'Complex Analysis Prompt',
        'content': """# System Instructions
        You are an expert {{expertise_area}} analyst.
        
        ## Context
        {{context}}
        
        ## Task
        Analyze the following {{data_type}}:
        {{data}}
        
        ## Requirements
        - {{requirement_1}}
        - {{requirement_2}}
        - {{requirement_3}}
        
        ## Output Format
        {{output_format}}
        """,
        'author': 'expert_user',
        'keywords': ['analysis', 'complex', 'template'],
        'variables': {
            'expertise_area': 'data science',
            'context': 'Q4 analysis',
            'data_type': 'sales data',
            'data': '[data placeholder]',
            'requirement_1': 'Include trends',
            'requirement_2': 'Identify anomalies',
            'requirement_3': 'Provide recommendations',
            'output_format': 'structured report'
        }
    }

# =====================================================================
# Prompt Categories and Collections
# =====================================================================

@pytest.fixture
def prompt_categories():
    """Prompt categories for organization."""
    return [
        {'name': 'General', 'description': 'General purpose prompts'},
        {'name': 'Code', 'description': 'Programming and code-related prompts'},
        {'name': 'Writing', 'description': 'Writing and editing prompts'},
        {'name': 'Analysis', 'description': 'Data analysis prompts'},
        {'name': 'Creative', 'description': 'Creative writing prompts'}
    ]

@pytest.fixture
def prompt_collection():
    """A collection of related prompts."""
    return {
        'name': 'Development Toolkit',
        'description': 'Prompts for software development',
        'prompts': [
            {'id': 1, 'name': 'Code Review'},
            {'id': 2, 'name': 'Bug Analysis'},
            {'id': 3, 'name': 'Test Generation'},
            {'id': 4, 'name': 'Documentation'}
        ]
    }

# =====================================================================
# Import/Export Fixtures
# =====================================================================

@pytest.fixture
def export_data():
    """Sample export data structure."""
    return {
        'version': '1.0',
        'exported_at': datetime.utcnow().isoformat(),
        'prompts': [
            {
                'name': 'Exported Prompt 1',
                'content': 'Content 1',
                'author': 'user1',
                'keywords': ['export', 'test']
            },
            {
                'name': 'Exported Prompt 2',
                'content': 'Content 2 with {{variable}}',
                'author': 'user2',
                'keywords': ['export', 'template']
            }
        ],
        'metadata': {
            'source': 'test_system',
            'count': 2
        }
    }

@pytest.fixture
def import_file(export_data, tmp_path):
    """Create a temporary import file."""
    file_path = tmp_path / "prompts_import.json"
    file_path.write_text(json.dumps(export_data))
    return file_path

# =====================================================================
# Versioning Fixtures
# =====================================================================

@pytest.fixture
def versioned_prompt():
    """Prompt with multiple versions."""
    return {
        'prompt_id': 1,
        'versions': [
            {
                'version': 1,
                'content': 'Original content',
                'created_at': '2024-01-01T00:00:00',
                'comment': 'Initial version'
            },
            {
                'version': 2,
                'content': 'Updated content',
                'created_at': '2024-01-02T00:00:00',
                'comment': 'Fixed typo'
            },
            {
                'version': 3,
                'content': 'Final content',
                'created_at': '2024-01-03T00:00:00',
                'comment': 'Production ready'
            }
        ]
    }

# =====================================================================
# Search and Filter Fixtures
# =====================================================================

@pytest.fixture
def search_queries():
    """Various search query patterns."""
    return {
        'simple': 'code',
        'phrase': '"code review"',
        'author': 'author:dev_user',
        'keyword': 'keyword:debugging',
        'combined': 'code AND review',
        'complex': '(code OR debug) AND author:dev_user'
    }

@pytest.fixture
def filter_criteria():
    """Filter criteria for prompts."""
    return {
        'by_author': {'author': 'test_user'},
        'by_keyword': {'keywords': ['code', 'review']},
        'by_date': {
            'created_after': '2024-01-01',
            'created_before': '2024-12-31'
        },
        'by_status': {'is_deleted': False},
        'combined': {
            'author': 'test_user',
            'keywords': ['test'],
            'is_deleted': False
        }
    }

# =====================================================================
# Performance Testing Fixtures
# =====================================================================

@pytest.fixture
def large_prompt_collection():
    """Generate a large collection of prompts for performance testing."""
    prompts = []
    for i in range(1000):
        prompts.append({
            'name': f'Prompt {i}',
            'content': f'Content for prompt {i} with {{variable_{i}}}',
            'author': f'user_{i % 10}',
            'keywords': [f'keyword_{i % 5}', f'tag_{i % 3}']
        })
    return prompts

@pytest.fixture
def performance_metrics():
    """Track performance metrics during tests."""
    return {
        'create_times': [],
        'read_times': [],
        'update_times': [],
        'search_times': [],
        'export_times': [],
        'import_times': []
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

def create_test_prompt(db, **kwargs):
    """Helper to create a test prompt."""
    prompt_data = {
        'name': 'Test Prompt',
        'content': 'Test content',
        'author': 'test_user',
        'keywords': ['test']
    }
    prompt_data.update(kwargs)
    return db.create_prompt(**prompt_data)

def create_prompt_with_versions(db, num_versions=3):
    """Helper to create a prompt with multiple versions."""
    prompt_id = create_test_prompt(db)
    
    for i in range(1, num_versions):
        db.update_prompt(
            prompt_id=prompt_id,
            content=f'Version {i+1} content',
            version_comment=f'Update {i}'
        )
    
    return prompt_id