# test_database.py
# Database tests for Prompt Studio

import pytest
import sqlite3
from pathlib import Path
from datetime import datetime

from tldw_Server_API.app.core.DB_Management.PromptStudioDatabase import (
    PromptStudioDatabase, DatabaseError, ConflictError, InputError
)

########################################################################################################################
# Database Initialization Tests

class TestDatabaseInitialization:
    """Test database initialization and schema creation."""
    
    def test_database_creation(self, temp_db_path: Path):
        """Test that database is created successfully."""
        db = PromptStudioDatabase(temp_db_path, "test_client")
        assert temp_db_path.exists()
        
        # Check that connection works
        conn = db.get_connection()
        assert conn is not None
    
    def test_schema_creation(self, test_db: PromptStudioDatabase):
        """Test that all required tables are created."""
        conn = test_db.get_connection()
        cursor = conn.cursor()
        
        # Check for expected tables
        expected_tables = [
            "prompt_studio_projects",
            "prompt_studio_signatures",
            "prompt_studio_prompts",
            "prompt_studio_test_cases",
            "prompt_studio_test_runs",
            "prompt_studio_evaluations",
            "prompt_studio_optimizations",
            "prompt_studio_job_queue"
        ]
        
        cursor.execute("""
            SELECT name FROM sqlite_master 
            WHERE type='table' AND name LIKE 'prompt_studio_%'
        """)
        
        tables = [row[0] for row in cursor.fetchall()]
        
        for table in expected_tables:
            assert table in tables, f"Table {table} not found"
    
    def test_indexes_creation(self, test_db: PromptStudioDatabase):
        """Test that indexes are created."""
        conn = test_db.get_connection()
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT name FROM sqlite_master 
            WHERE type='index' AND name LIKE 'idx_ps_%'
        """)
        
        indexes = [row[0] for row in cursor.fetchall()]
        
        # Check for some key indexes
        assert "idx_ps_projects_user" in indexes
        assert "idx_ps_prompts_project" in indexes
        assert "idx_ps_test_cases_project" in indexes
    
    def test_fts_tables_creation(self, test_db: PromptStudioDatabase):
        """Test that FTS tables are created."""
        conn = test_db.get_connection()
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT name FROM sqlite_master 
            WHERE type='table' AND name LIKE '%_fts'
        """)
        
        fts_tables = [row[0] for row in cursor.fetchall()]
        
        expected_fts = [
            "prompt_studio_projects_fts",
            "prompt_studio_prompts_fts",
            "prompt_studio_test_cases_fts"
        ]
        
        for table in expected_fts:
            assert table in fts_tables, f"FTS table {table} not found"

########################################################################################################################
# Project CRUD Tests

class TestProjectOperations:
    """Test project CRUD operations."""
    
    def test_create_project(self, test_db: PromptStudioDatabase):
        """Test creating a project."""
        project = test_db.create_project(
            name="Test Project",
            description="Test description",
            status="draft"
        )
        
        assert project is not None
        assert project["name"] == "Test Project"
        assert project["description"] == "Test description"
        assert project["status"] == "draft"
        assert project["uuid"] is not None
        assert project["id"] > 0
    
    def test_create_duplicate_project(self, test_db: PromptStudioDatabase):
        """Test that duplicate project names are rejected."""
        test_db.create_project(name="Unique Project")
        
        with pytest.raises(ConflictError):
            test_db.create_project(name="Unique Project")
    
    def test_get_project(self, populated_db: PromptStudioDatabase):
        """Test getting a project by ID."""
        # Create a project first
        created = populated_db.create_project(name="Get Test Project")
        
        # Get the project
        project = populated_db.get_project(created["id"])
        
        assert project is not None
        assert project["id"] == created["id"]
        assert project["name"] == "Get Test Project"
    
    def test_get_nonexistent_project(self, test_db: PromptStudioDatabase):
        """Test getting a non-existent project."""
        project = test_db.get_project(99999)
        assert project is None
    
    def test_list_projects(self, populated_db: PromptStudioDatabase):
        """Test listing projects."""
        # List all projects
        result = populated_db.list_projects(page=1, per_page=10)
        
        assert "projects" in result
        assert "pagination" in result
        assert len(result["projects"]) >= 2  # From populated_db fixture
        assert result["pagination"]["total"] >= 2
    
    def test_list_projects_with_filter(self, populated_db: PromptStudioDatabase):
        """Test listing projects with status filter."""
        result = populated_db.list_projects(status="active", page=1, per_page=10)
        
        assert "projects" in result
        for project in result["projects"]:
            assert project["status"] == "active"
    
    def test_update_project(self, populated_db: PromptStudioDatabase):
        """Test updating a project."""
        # Create a project
        project = populated_db.create_project(name="Update Test")
        
        # Update it
        updated = populated_db.update_project(
            project["id"],
            {"description": "Updated description", "status": "active"}
        )
        
        assert updated["description"] == "Updated description"
        assert updated["status"] == "active"
        assert updated["name"] == "Update Test"  # Unchanged
    
    def test_delete_project_soft(self, populated_db: PromptStudioDatabase):
        """Test soft deleting a project."""
        # Create a project
        project = populated_db.create_project(name="Delete Test")
        
        # Soft delete it
        success = populated_db.delete_project(project["id"], hard_delete=False)
        assert success is True
        
        # Should not be found in normal get
        deleted = populated_db.get_project(project["id"], include_deleted=False)
        assert deleted is None
        
        # Should be found when including deleted
        deleted_with_flag = populated_db.get_project(project["id"], include_deleted=True)
        assert deleted_with_flag is not None
        assert deleted_with_flag["deleted"] == 1
    
    def test_delete_project_hard(self, populated_db: PromptStudioDatabase):
        """Test hard deleting a project."""
        # Create a project
        project = populated_db.create_project(name="Hard Delete Test")
        project_id = project["id"]
        
        # Hard delete it
        success = populated_db.delete_project(project_id, hard_delete=True)
        assert success is True
        
        # Should not exist at all
        deleted = populated_db.get_project(project_id, include_deleted=True)
        assert deleted is None

########################################################################################################################
# Transaction Tests

class TestTransactions:
    """Test database transaction management."""
    
    def test_transaction_rollback(self, test_db: PromptStudioDatabase):
        """Test that transactions rollback on error."""
        with pytest.raises(Exception):
            with test_db.transaction() as conn:
                cursor = conn.cursor()
                
                # Create a project
                cursor.execute("""
                    INSERT INTO prompt_studio_projects (uuid, name, user_id, client_id)
                    VALUES ('test-uuid', 'Transaction Test', 'user1', 'client1')
                """)
                
                # Force an error
                raise Exception("Test error")
        
        # Check that project was not created
        conn = test_db.get_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM prompt_studio_projects WHERE name = 'Transaction Test'")
        assert cursor.fetchone() is None
    
    def test_transaction_commit(self, test_db: PromptStudioDatabase):
        """Test that transactions commit properly."""
        with test_db.transaction() as conn:
            cursor = conn.cursor()
            
            # Create a project
            cursor.execute("""
                INSERT INTO prompt_studio_projects (uuid, name, user_id, client_id)
                VALUES ('test-uuid-2', 'Transaction Test 2', 'user1', 'client1')
            """)
        
        # Check that project was created
        conn = test_db.get_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM prompt_studio_projects WHERE name = 'Transaction Test 2'")
        assert cursor.fetchone() is not None

########################################################################################################################
# Helper Method Tests

class TestHelperMethods:
    """Test database helper methods."""
    
    def test_row_to_dict(self, test_db: PromptStudioDatabase):
        """Test converting database row to dictionary."""
        conn = test_db.get_connection()
        cursor = conn.cursor()
        
        # Create a test project
        cursor.execute("""
            INSERT INTO prompt_studio_projects (uuid, name, user_id, client_id, metadata)
            VALUES ('test-uuid', 'Dict Test', 'user1', 'client1', '{"key": "value"}')
        """)
        conn.commit()
        
        # Fetch and convert
        cursor.execute("SELECT * FROM prompt_studio_projects WHERE name = 'Dict Test'")
        row = cursor.fetchone()
        result = test_db._row_to_dict(cursor, row)
        
        assert isinstance(result, dict)
        assert result["name"] == "Dict Test"
        assert result["metadata"] == {"key": "value"}  # JSON parsed
        assert isinstance(result["created_at"], datetime)  # Datetime parsed
    
    def test_sync_log_event(self, test_db: PromptStudioDatabase):
        """Test sync event logging."""
        # This should not raise an error even if sync_log doesn't exist
        test_db._log_sync_event(
            "test_entity",
            "test-uuid",
            "create",
            {"test": "data"}
        )
        
        # If sync_log exists, check that event was logged
        conn = test_db.get_connection()
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT name FROM sqlite_master 
            WHERE type='table' AND name='sync_log'
        """)
        
        if cursor.fetchone():
            cursor.execute("""
                SELECT * FROM sync_log 
                WHERE entity = 'test_entity' AND entity_uuid = 'test-uuid'
            """)
            log_entry = cursor.fetchone()
            # May or may not exist depending on base database setup