# conftest.py
# Test fixtures for Prompt Studio tests

import pytest
import tempfile
import shutil
from pathlib import Path
from typing import Generator, Dict, Any
import uuid

from tldw_Server_API.app.core.DB_Management.PromptStudioDatabase import PromptStudioDatabase
from tldw_Server_API.app.api.v1.schemas.prompt_studio_base import ProjectStatus
from tldw_Server_API.app.api.v1.schemas.prompt_studio_project import (
    ProjectCreate, PromptCreate, SignatureCreate
)

########################################################################################################################
# Database Fixtures

@pytest.fixture
def temp_db_path() -> Generator[Path, None, None]:
    """Create a temporary database file for testing."""
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as tmp_file:
        db_path = Path(tmp_file.name)
    
    yield db_path
    
    # Cleanup
    if db_path.exists():
        db_path.unlink()

@pytest.fixture
def test_db(temp_db_path: Path) -> PromptStudioDatabase:
    """Create a test database instance."""
    client_id = "test_client_" + str(uuid.uuid4())[:8]
    db = PromptStudioDatabase(temp_db_path, client_id)
    return db

@pytest.fixture
def populated_db(test_db: PromptStudioDatabase) -> PromptStudioDatabase:
    """Create a database with some test data."""
    # Create test projects
    project1 = test_db.create_project(
        name="Test Project 1",
        description="First test project",
        status="active"
    )
    
    project2 = test_db.create_project(
        name="Test Project 2",
        description="Second test project",
        status="draft"
    )
    
    # Add more test data as needed
    conn = test_db.get_connection()
    cursor = conn.cursor()
    
    # Add a test signature
    cursor.execute("""
        INSERT INTO prompt_studio_signatures (
            uuid, project_id, name, input_schema, output_schema, client_id
        ) VALUES (?, ?, ?, ?, ?, ?)
    """, (
        str(uuid.uuid4()),
        project1["id"],
        "Test Signature",
        '[{"name": "input", "type": "string", "required": true}]',
        '[{"name": "output", "type": "string", "required": true}]',
        test_db.client_id
    ))
    
    # Add a test prompt
    cursor.execute("""
        INSERT INTO prompt_studio_prompts (
            uuid, project_id, name, system_prompt, user_prompt, 
            version_number, client_id
        ) VALUES (?, ?, ?, ?, ?, ?, ?)
    """, (
        str(uuid.uuid4()),
        project1["id"],
        "Test Prompt",
        "You are a helpful assistant.",
        "Please help with: {input}",
        1,
        test_db.client_id
    ))
    
    conn.commit()
    
    return test_db

########################################################################################################################
# User Context Fixtures

@pytest.fixture
def anonymous_user() -> Dict[str, Any]:
    """Anonymous user context."""
    return {
        "user_id": "anonymous",
        "client_id": "test_client",
        "is_authenticated": False,
        "is_admin": False,
        "permissions": []
    }

@pytest.fixture
def authenticated_user() -> Dict[str, Any]:
    """Authenticated regular user context."""
    return {
        "user_id": "user_123",
        "client_id": "test_client",
        "is_authenticated": True,
        "is_admin": False,
        "permissions": ["read", "write"]
    }

@pytest.fixture
def admin_user() -> Dict[str, Any]:
    """Admin user context."""
    return {
        "user_id": "admin_456",
        "client_id": "admin_client",
        "is_authenticated": True,
        "is_admin": True,
        "permissions": ["read", "write", "delete", "admin"]
    }

########################################################################################################################
# Test Data Fixtures

@pytest.fixture
def sample_project_data() -> ProjectCreate:
    """Sample project creation data."""
    return ProjectCreate(
        name=f"Test Project {uuid.uuid4().hex[:8]}",
        description="A test project for unit testing",
        status=ProjectStatus.DRAFT,
        metadata={"test": True, "version": "1.0"}
    )

@pytest.fixture
def sample_prompt_data() -> Dict[str, Any]:
    """Sample prompt creation data."""
    return {
        "name": f"Test Prompt {uuid.uuid4().hex[:8]}",
        "system_prompt": "You are a helpful AI assistant.",
        "user_prompt": "Please help the user with: {task}",
        "few_shot_examples": [
            {
                "inputs": {"task": "Write a poem"},
                "outputs": {"response": "Here's a beautiful poem..."},
                "explanation": "Example of creative writing"
            }
        ],
        "modules_config": [
            {
                "type": "chain_of_thought",
                "enabled": True,
                "config": {"steps": 3}
            }
        ],
        "change_description": "Initial version"
    }

@pytest.fixture
def sample_signature_data() -> Dict[str, Any]:
    """Sample signature data."""
    return {
        "name": f"Test Signature {uuid.uuid4().hex[:8]}",
        "input_schema": [
            {
                "name": "text",
                "type": "string",
                "description": "Input text",
                "required": True
            }
        ],
        "output_schema": [
            {
                "name": "result",
                "type": "string",
                "description": "Processing result",
                "required": True
            }
        ],
        "constraints": [
            {
                "type": "length",
                "field": "text",
                "value": 1000,
                "message": "Text must be less than 1000 characters"
            }
        ]
    }

@pytest.fixture
def sample_test_case_data() -> Dict[str, Any]:
    """Sample test case data."""
    return {
        "name": f"Test Case {uuid.uuid4().hex[:8]}",
        "description": "A sample test case",
        "inputs": {"text": "Hello, world!"},
        "expected_outputs": {"result": "Processed: Hello, world!"},
        "tags": ["sample", "test"],
        "is_golden": False
    }

########################################################################################################################
# Mock Fixtures

@pytest.fixture
def mock_llm_response():
    """Mock LLM response for testing."""
    def _mock_response(prompt: str) -> str:
        return f"Mock response for: {prompt[:50]}..."
    return _mock_response

@pytest.fixture
def mock_security_config():
    """Mock security configuration."""
    from tldw_Server_API.app.api.v1.schemas.prompt_studio_base import SecurityConfig
    return SecurityConfig(
        max_prompt_length=10000,
        max_test_cases=100,
        max_concurrent_jobs=5,
        enable_prompt_validation=True,
        enable_rate_limiting=False  # Disable for testing
    )

########################################################################################################################
# Cleanup Fixtures

@pytest.fixture(autouse=True)
def cleanup_test_files():
    """Cleanup any test files created during tests."""
    test_dir = Path("test_prompt_studio_temp")
    
    yield
    
    # Cleanup after test
    if test_dir.exists():
        shutil.rmtree(test_dir)

########################################################################################################################
# Helper Functions for Tests

def create_test_project(db: PromptStudioDatabase, name: str = None) -> Dict[str, Any]:
    """Helper to create a test project."""
    if name is None:
        name = f"Test Project {uuid.uuid4().hex[:8]}"
    
    return db.create_project(
        name=name,
        description="Test project",
        status="draft"
    )

def create_test_prompt(db: PromptStudioDatabase, project_id: int, name: str = None) -> Dict[str, Any]:
    """Helper to create a test prompt."""
    if name is None:
        name = f"Test Prompt {uuid.uuid4().hex[:8]}"
    
    conn = db.get_connection()
    cursor = conn.cursor()
    
    cursor.execute("""
        INSERT INTO prompt_studio_prompts (
            uuid, project_id, name, system_prompt, user_prompt,
            version_number, client_id
        ) VALUES (?, ?, ?, ?, ?, ?, ?)
    """, (
        str(uuid.uuid4()),
        project_id,
        name,
        "Test system prompt",
        "Test user prompt",
        1,
        db.client_id
    ))
    
    prompt_id = cursor.lastrowid
    conn.commit()
    
    cursor.execute("SELECT * FROM prompt_studio_prompts WHERE id = ?", (prompt_id,))
    return db._row_to_dict(cursor, cursor.fetchone())