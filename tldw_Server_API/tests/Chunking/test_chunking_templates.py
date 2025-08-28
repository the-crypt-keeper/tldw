# test_chunking_templates.py
"""
Comprehensive tests for chunking template functionality.
Tests database operations, API endpoints, and template application.
"""

import json
import pytest
import tempfile
from pathlib import Path
from typing import Dict, Any, List
from unittest.mock import patch, MagicMock

from fastapi.testclient import TestClient
from fastapi import FastAPI

from tldw_Server_API.app.core.DB_Management.Media_DB_v2 import MediaDatabase
from tldw_Server_API.app.core.Chunking.templates import TemplateProcessor, ChunkingTemplate, TemplateStage
from tldw_Server_API.app.core.Chunking.template_initialization import (
    load_builtin_templates,
    initialize_chunking_templates,
    ensure_templates_initialized
)
from tldw_Server_API.app.api.v1.endpoints.chunking_templates import router as templates_router
from tldw_Server_API.app.api.v1.schemas.chunking_templates_schemas import (
    ChunkingTemplateCreate,
    ChunkingTemplateResponse,
    TemplateConfig
)


# Fixtures
@pytest.fixture
def temp_db():
    """Create a temporary database for testing."""
    with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as tmp:
        db_path = tmp.name
    
    # Initialize database with schema
    db = MediaDatabase(db_path=db_path, client_id='test_client')
    
    # Run migration to create chunking templates table
    conn = db.get_connection()
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS ChunkingTemplates (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            uuid TEXT UNIQUE NOT NULL,
            name TEXT NOT NULL,
            description TEXT,
            template_json TEXT NOT NULL,
            is_builtin BOOLEAN DEFAULT 0 NOT NULL,
            tags TEXT,
            created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
            last_modified DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
            version INTEGER NOT NULL DEFAULT 1,
            client_id TEXT NOT NULL,
            user_id TEXT,
            deleted BOOLEAN NOT NULL DEFAULT 0,
            prev_version INTEGER,
            merge_parent_uuid TEXT
        );
        
        CREATE UNIQUE INDEX IF NOT EXISTS idx_unique_template_name 
        ON ChunkingTemplates(name) WHERE deleted = 0;
        
        CREATE TABLE IF NOT EXISTS sync_log (
            change_id INTEGER PRIMARY KEY AUTOINCREMENT,
            entity_uuid TEXT NOT NULL,
            entity_type TEXT NOT NULL,
            action TEXT NOT NULL,
            timestamp DATETIME NOT NULL,
            client_id TEXT NOT NULL,
            details TEXT
        );
    """)
    conn.commit()
    conn.close()
    
    yield db, db_path
    
    # Cleanup
    import os
    try:
        os.unlink(db_path)
    except:
        pass


@pytest.fixture
def test_client():
    """Create a test client for API testing."""
    app = FastAPI()
    app.include_router(templates_router, prefix="/api/v1")
    
    with TestClient(app) as client:
        yield client


@pytest.fixture
def sample_template():
    """Create a sample template for testing."""
    return {
        "name": "test_template",
        "description": "Test template for unit tests",
        "tags": ["test", "sample"],
        "preprocessing": [
            {
                "operation": "normalize_whitespace",
                "config": {"max_line_breaks": 2}
            }
        ],
        "chunking": {
            "method": "sentences",
            "config": {
                "max_size": 5,
                "overlap": 1
            }
        },
        "postprocessing": [
            {
                "operation": "filter_empty",
                "config": {"min_length": 10}
            }
        ]
    }


# Database Tests
class TestDatabaseOperations:
    """Test database CRUD operations for chunking templates."""
    
    def test_create_template(self, temp_db):
        """Test creating a new template."""
        db, _ = temp_db
        
        template = db.create_chunking_template(
            name="test_template",
            template_json='{"chunking": {"method": "words", "config": {"max_size": 100}}}',
            description="Test template",
            tags=["test"],
            user_id="test_user"
        )
        
        assert template is not None
        assert template['name'] == "test_template"
        assert template['description'] == "Test template"
        assert template['is_builtin'] == False
        assert template['tags'] == ["test"]
        assert template['user_id'] == "test_user"
        assert template['version'] == 1
    
    def test_create_duplicate_template(self, temp_db):
        """Test that creating duplicate template names raises error."""
        db, _ = temp_db
        
        # Create first template
        db.create_chunking_template(
            name="duplicate_test",
            template_json='{"chunking": {"method": "words"}}',
            description="First template"
        )
        
        # Try to create duplicate
        with pytest.raises(Exception) as exc_info:
            db.create_chunking_template(
                name="duplicate_test",
                template_json='{"chunking": {"method": "sentences"}}',
                description="Duplicate template"
            )
        
        assert "already exists" in str(exc_info.value)
    
    def test_get_template(self, temp_db):
        """Test retrieving a template by name, ID, or UUID."""
        db, _ = temp_db
        
        # Create template
        created = db.create_chunking_template(
            name="get_test",
            template_json='{"chunking": {"method": "words"}}',
            description="Get test"
        )
        
        # Get by name
        by_name = db.get_chunking_template(name="get_test")
        assert by_name is not None
        assert by_name['name'] == "get_test"
        
        # Get by ID
        by_id = db.get_chunking_template(template_id=created['id'])
        assert by_id is not None
        assert by_id['id'] == created['id']
        
        # Get by UUID
        by_uuid = db.get_chunking_template(uuid=created['uuid'])
        assert by_uuid is not None
        assert by_uuid['uuid'] == created['uuid']
        
        # Get non-existent
        none_result = db.get_chunking_template(name="non_existent")
        assert none_result is None
    
    def test_list_templates(self, temp_db):
        """Test listing templates with various filters."""
        db, _ = temp_db
        
        # Create custom templates
        db.create_chunking_template(
            name="custom1",
            template_json='{"chunking": {"method": "words"}}',
            description="Custom 1",
            tags=["custom", "test"],
            user_id="user1"
        )
        
        db.create_chunking_template(
            name="custom2",
            template_json='{"chunking": {"method": "sentences"}}',
            description="Custom 2",
            tags=["custom"],
            user_id="user2"
        )
        
        # Create built-in template
        db.create_chunking_template(
            name="builtin1",
            template_json='{"chunking": {"method": "paragraphs"}}',
            description="Built-in 1",
            is_builtin=True,
            tags=["builtin"]
        )
        
        # List all
        all_templates = db.list_chunking_templates()
        assert len(all_templates) == 3
        
        # List only custom
        custom_only = db.list_chunking_templates(include_builtin=False)
        assert len(custom_only) == 2
        assert all(not t['is_builtin'] for t in custom_only)
        
        # List only built-in
        builtin_only = db.list_chunking_templates(include_custom=False)
        assert len(builtin_only) == 1
        assert all(t['is_builtin'] for t in builtin_only)
        
        # Filter by user
        user1_templates = db.list_chunking_templates(user_id="user1")
        assert len(user1_templates) == 1
        assert user1_templates[0]['name'] == "custom1"
        
        # Filter by tags
        test_tagged = db.list_chunking_templates(tags=["test"])
        assert len(test_tagged) == 1
        assert "test" in test_tagged[0]['tags']
    
    def test_update_template(self, temp_db):
        """Test updating a template."""
        db, _ = temp_db
        
        # Create template
        created = db.create_chunking_template(
            name="update_test",
            template_json='{"chunking": {"method": "words"}}',
            description="Original description",
            tags=["original"]
        )
        
        # Update template
        success = db.update_chunking_template(
            name="update_test",
            description="Updated description",
            tags=["updated", "modified"]
        )
        assert success is True
        
        # Get updated template
        updated = db.get_chunking_template(name="update_test")
        assert updated['description'] == "Updated description"
        assert updated['tags'] == ["updated", "modified"]
        assert updated['version'] == 2  # Version should increment
    
    def test_cannot_update_builtin(self, temp_db):
        """Test that built-in templates cannot be updated."""
        db, _ = temp_db
        
        # Create built-in template
        db.create_chunking_template(
            name="builtin_test",
            template_json='{"chunking": {"method": "words"}}',
            description="Built-in template",
            is_builtin=True
        )
        
        # Try to update
        with pytest.raises(Exception) as exc_info:
            db.update_chunking_template(
                name="builtin_test",
                description="Try to update"
            )
        
        assert "Cannot modify built-in templates" in str(exc_info.value)
    
    def test_delete_template(self, temp_db):
        """Test soft and hard delete of templates."""
        db, _ = temp_db
        
        # Create template
        db.create_chunking_template(
            name="delete_test",
            template_json='{"chunking": {"method": "words"}}',
            description="To be deleted"
        )
        
        # Soft delete
        success = db.delete_chunking_template(name="delete_test", hard_delete=False)
        assert success is True
        
        # Should not appear in normal list
        templates = db.list_chunking_templates()
        assert not any(t['name'] == "delete_test" for t in templates)
        
        # Should still exist with include_deleted
        deleted = db.get_chunking_template(name="delete_test", include_deleted=True)
        assert deleted is not None
        assert deleted['deleted'] is True
        
        # Create another for hard delete test
        db.create_chunking_template(
            name="hard_delete_test",
            template_json='{"chunking": {"method": "words"}}',
            description="To be hard deleted"
        )
        
        # Hard delete
        success = db.delete_chunking_template(name="hard_delete_test", hard_delete=True)
        assert success is True
        
        # Should not exist even with include_deleted
        gone = db.get_chunking_template(name="hard_delete_test", include_deleted=True)
        assert gone is None
    
    def test_cannot_delete_builtin(self, temp_db):
        """Test that built-in templates cannot be deleted."""
        db, _ = temp_db
        
        # Create built-in template
        db.create_chunking_template(
            name="builtin_delete_test",
            template_json='{"chunking": {"method": "words"}}',
            description="Built-in template",
            is_builtin=True
        )
        
        # Try to delete
        with pytest.raises(Exception) as exc_info:
            db.delete_chunking_template(name="builtin_delete_test")
        
        assert "Cannot delete built-in templates" in str(exc_info.value)


# Template Initialization Tests
class TestTemplateInitialization:
    """Test template initialization and seeding."""
    
    def test_load_builtin_templates(self):
        """Test loading built-in templates from JSON files."""
        import tempfile
        import json
        from pathlib import Path
        
        # Create a temporary directory with test templates
        with tempfile.TemporaryDirectory() as tmpdir:
            template_dir = Path(tmpdir) / "template_library"
            template_dir.mkdir()
            
            # Create test template files
            template1 = {
                "name": "test_template1",
                "description": "Test 1",
                "tags": ["test"],
                "chunking": {"method": "words", "config": {}}
            }
            template2 = {
                "name": "test_template2",
                "description": "Test 2",
                "tags": ["test"],
                "chunking": {"method": "sentences", "config": {}}
            }
            
            (template_dir / "test1.json").write_text(json.dumps(template1))
            (template_dir / "test2.json").write_text(json.dumps(template2))
            
            # Patch the Path to use our temp directory
            with patch('tldw_Server_API.app.core.Chunking.template_initialization.Path') as mock_path:
                mock_path.return_value.parent = Path(tmpdir)
                
                # Load templates
                templates = load_builtin_templates()
                
                assert len(templates) == 2
                names = [t['name'] for t in templates]
                assert "test_template1" in names
                assert "test_template2" in names
    
    def test_seed_builtin_templates(self, temp_db):
        """Test seeding built-in templates into database."""
        db, _ = temp_db
        
        templates = [
            {
                'name': 'seed_test1',
                'description': 'Seed test 1',
                'tags': ['test'],
                'template': {
                    'chunking': {'method': 'words', 'config': {'max_size': 100}}
                }
            },
            {
                'name': 'seed_test2',
                'description': 'Seed test 2',
                'tags': ['test'],
                'template': {
                    'chunking': {'method': 'sentences', 'config': {'max_size': 5}}
                }
            }
        ]
        
        # Seed templates
        count = db.seed_builtin_templates(templates)
        assert count == 2
        
        # Verify they were created as built-in
        seeded = db.list_chunking_templates(include_custom=False)
        assert len(seeded) == 2
        assert all(t['is_builtin'] for t in seeded)
        
        # Test idempotency - seeding again should not duplicate
        count = db.seed_builtin_templates(templates)
        assert count == 0  # No new templates created
        
        seeded = db.list_chunking_templates(include_custom=False)
        assert len(seeded) == 2  # Still only 2


# API Endpoint Tests
class TestAPIEndpoints:
    """Test REST API endpoints for template management."""
    
    @patch('tldw_Server_API.app.api.v1.endpoints.chunking_templates.get_database')
    def test_list_templates_endpoint(self, mock_get_db, test_client, temp_db):
        """Test GET /api/v1/chunking/templates endpoint."""
        db, _ = temp_db
        mock_get_db.return_value = db
        
        # Create test templates
        db.create_chunking_template(
            name="api_test1",
            template_json='{"chunking": {"method": "words"}}',
            description="API test 1",
            tags=["api", "test"]
        )
        
        # Test listing
        response = test_client.get("/api/v1/chunking/templates")
        assert response.status_code == 200
        
        data = response.json()
        assert "templates" in data
        assert "total" in data
        assert data["total"] >= 1
        
        # Test filtering
        response = test_client.get("/api/v1/chunking/templates?tags=api")
        assert response.status_code == 200
        data = response.json()
        assert all("api" in t["tags"] for t in data["templates"])
    
    @patch('tldw_Server_API.app.api.v1.endpoints.chunking_templates.get_database')
    def test_get_template_endpoint(self, mock_get_db, test_client, temp_db):
        """Test GET /api/v1/chunking/templates/{name} endpoint."""
        db, _ = temp_db
        mock_get_db.return_value = db
        
        # Create template
        db.create_chunking_template(
            name="get_api_test",
            template_json='{"chunking": {"method": "words"}}',
            description="Get API test"
        )
        
        # Get existing template
        response = test_client.get("/api/v1/chunking/templates/get_api_test")
        assert response.status_code == 200
        
        data = response.json()
        assert data["name"] == "get_api_test"
        assert data["description"] == "Get API test"
        
        # Get non-existent template
        response = test_client.get("/api/v1/chunking/templates/non_existent")
        assert response.status_code == 404
    
    @patch('tldw_Server_API.app.api.v1.endpoints.chunking_templates.get_database')
    def test_create_template_endpoint(self, mock_get_db, test_client, temp_db, sample_template):
        """Test POST /api/v1/chunking/templates endpoint."""
        db, _ = temp_db
        mock_get_db.return_value = db
        
        # Create template via API
        request_data = {
            "name": sample_template["name"],
            "description": sample_template["description"],
            "tags": sample_template["tags"],
            "template": {
                "preprocessing": sample_template["preprocessing"],
                "chunking": sample_template["chunking"],
                "postprocessing": sample_template["postprocessing"]
            }
        }
        
        response = test_client.post("/api/v1/chunking/templates", json=request_data)
        assert response.status_code == 201
        
        data = response.json()
        assert data["name"] == sample_template["name"]
        assert data["description"] == sample_template["description"]
        assert data["is_builtin"] is False
        
        # Try to create duplicate
        response = test_client.post("/api/v1/chunking/templates", json=request_data)
        assert response.status_code == 409
    
    @patch('tldw_Server_API.app.api.v1.endpoints.chunking_templates.get_database')
    def test_update_template_endpoint(self, mock_get_db, test_client, temp_db):
        """Test PUT /api/v1/chunking/templates/{name} endpoint."""
        db, _ = temp_db
        mock_get_db.return_value = db
        
        # Create template
        db.create_chunking_template(
            name="update_api_test",
            template_json='{"chunking": {"method": "words"}}',
            description="Original"
        )
        
        # Update via API
        update_data = {
            "description": "Updated via API",
            "tags": ["updated", "api"]
        }
        
        response = test_client.put("/api/v1/chunking/templates/update_api_test", json=update_data)
        assert response.status_code == 200
        
        data = response.json()
        assert data["description"] == "Updated via API"
        assert data["tags"] == ["updated", "api"]
        
        # Update non-existent
        response = test_client.put("/api/v1/chunking/templates/non_existent", json=update_data)
        assert response.status_code == 404
    
    @patch('tldw_Server_API.app.api.v1.endpoints.chunking_templates.get_database')
    def test_delete_template_endpoint(self, mock_get_db, test_client, temp_db):
        """Test DELETE /api/v1/chunking/templates/{name} endpoint."""
        db, _ = temp_db
        mock_get_db.return_value = db
        
        # Create template
        db.create_chunking_template(
            name="delete_api_test",
            template_json='{"chunking": {"method": "words"}}',
            description="To delete"
        )
        
        # Delete via API
        response = test_client.delete("/api/v1/chunking/templates/delete_api_test")
        assert response.status_code == 204
        
        # Verify deleted
        response = test_client.get("/api/v1/chunking/templates/delete_api_test")
        assert response.status_code == 404
        
        # Delete non-existent
        response = test_client.delete("/api/v1/chunking/templates/non_existent")
        assert response.status_code == 404
    
    @patch('tldw_Server_API.app.api.v1.endpoints.chunking_templates.get_database')
    def test_validate_template_endpoint(self, mock_get_db, test_client):
        """Test POST /api/v1/chunking/templates/validate endpoint."""
        mock_get_db.return_value = MagicMock()
        
        # Valid template
        valid_template = {
            "chunking": {
                "method": "words",
                "config": {"max_size": 100}
            },
            "preprocessing": [
                {"operation": "normalize_whitespace", "config": {}}
            ]
        }
        
        response = test_client.post("/api/v1/chunking/templates/validate", json=valid_template)
        assert response.status_code == 200
        data = response.json()
        assert data["valid"] is True
        assert data["errors"] is None
        
        # Invalid template (missing chunking)
        invalid_template = {
            "preprocessing": [
                {"operation": "normalize_whitespace"}
            ]
        }
        
        response = test_client.post("/api/v1/chunking/templates/validate", json=invalid_template)
        assert response.status_code == 200
        data = response.json()
        assert data["valid"] is False
        assert data["errors"] is not None
        assert len(data["errors"]) > 0


# Template Processing Tests
class TestTemplateProcessing:
    """Test template processing and application."""
    
    def test_template_processor_operations(self):
        """Test TemplateProcessor built-in operations."""
        processor = TemplateProcessor()
        
        # Test normalize_whitespace
        text = "This  has   multiple\n\n\n\nspaces"
        result = processor._normalize_whitespace(text, {"max_line_breaks": 2})
        assert "\n\n\n\n" not in result
        
        # Test filter_empty
        chunks = ["", "valid chunk", "  ", "another valid", "\n\t"]
        result = processor._filter_empty(chunks, {"min_length": 5})
        assert len(result) == 2
        assert "" not in result
        assert "valid chunk" in result
        
        # Test merge_small
        chunks = ["a", "b", "longer chunk", "c", "another long chunk"]
        result = processor._merge_small(chunks, {"min_size": 10, "separator": " "})
        assert all(len(chunk) >= 10 or chunk == result[-1] for chunk in result)
    
    def test_process_template(self):
        """Test processing text through a template."""
        processor = TemplateProcessor()
        
        # Create a simple template
        template = ChunkingTemplate(
            name="test_processing",
            description="Test template",
            base_method="sentences",
            stages=[
                TemplateStage("preprocess", [
                    {"type": "normalize_whitespace", "params": {"max_line_breaks": 1}}
                ]),
                TemplateStage("chunk", [
                    {"method": "sentences", "max_size": 2}
                ]),
                TemplateStage("postprocess", [
                    {"type": "filter_empty", "params": {"min_length": 5}}
                ])
            ],
            default_options={"max_size": 2}
        )
        
        # Process text
        text = "This is sentence one. This is sentence two. Short. This is sentence three."
        
        with patch.object(processor, '_get_chunker') as mock_chunker:
            mock_chunker_instance = MagicMock()
            mock_chunker_instance.chunk.return_value = [
                "This is sentence one. This is sentence two.",
                "Short.",
                "This is sentence three."
            ]
            mock_chunker.return_value = mock_chunker_instance
            
            chunks = processor.process_template(text, template)
            
            # Short chunk should be filtered out
            assert "Short." not in chunks
            assert len(chunks) == 2


# Integration Tests
class TestIntegration:
    """Test integration between components."""
    
    @patch('tldw_Server_API.app.api.v1.endpoints.chunking_templates.get_database')
    def test_apply_template_endpoint(self, mock_get_db, test_client, temp_db):
        """Test applying a template to text via API."""
        db, _ = temp_db
        mock_get_db.return_value = db
        
        # Create a template
        template_config = {
            "chunking": {
                "method": "words",
                "config": {"max_size": 10, "overlap": 2}
            }
        }
        
        db.create_chunking_template(
            name="apply_test",
            template_json=json.dumps(template_config),
            description="Template for apply test"
        )
        
        # Apply template
        request_data = {
            "template_name": "apply_test",
            "text": "This is a test text that should be chunked according to the template configuration."
        }
        
        with patch('tldw_Server_API.app.api.v1.endpoints.chunking_templates.TemplateProcessor') as mock_processor:
            mock_instance = MagicMock()
            mock_instance.process_template.return_value = [
                "This is a test text",
                "that should be chunked",
                "according to the template",
                "configuration."
            ]
            mock_processor.return_value = mock_instance
            
            response = test_client.post("/api/v1/chunking/templates/apply", json=request_data)
            assert response.status_code == 200
            
            data = response.json()
            assert data["template_name"] == "apply_test"
            assert "chunks" in data
            assert len(data["chunks"]) == 4
            assert "metadata" in data
    
    def test_template_in_chunking_endpoint(self):
        """Test using templates in the main chunking endpoint."""
        from tldw_Server_API.app.api.v1.endpoints.chunking import process_text_for_chunking_json
        from tldw_Server_API.app.api.v1.schemas.chunking_schema import ChunkingTextRequest, ChunkingOptionsRequest
        
        # This would require more complex mocking of the chunking endpoint
        # For now, we just verify the schema accepts template_name
        options = ChunkingOptionsRequest(
            template_name="test_template",
            method="words",  # This should be overridden by template
            max_size=100
        )
        
        assert options.template_name == "test_template"
        assert hasattr(options, 'template_name')


if __name__ == "__main__":
    pytest.main([__file__, "-v"])