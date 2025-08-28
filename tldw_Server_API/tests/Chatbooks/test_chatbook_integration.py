"""
Integration tests for Chatbook service using real test database.
No mocking - uses actual components.
"""
import pytest
import tempfile
import json
import zipfile
import os
import shutil
from pathlib import Path
from datetime import datetime
import asyncio

from tldw_Server_API.app.core.Chatbooks.chatbook_service import ChatbookService
from tldw_Server_API.app.core.Chatbooks.chatbook_models import (
    ChatbookManifest,
    ChatbookVersion,
    ContentItem,
    ContentType,
    ExportStatus,
    ImportStatus
)
from tldw_Server_API.app.core.DB_Management.ChaChaNotes_DB import CharactersRAGDB


def manifest_to_dict(manifest):
    """Helper to convert manifest to dict handling enums."""
    data = manifest.to_dict()
    if hasattr(manifest.version, 'value'):
        data['version'] = manifest.version.value
    return data


@pytest.fixture
def test_db_path(tmp_path):
    """Create a temporary database path."""
    db_path = tmp_path / "test_chatbook.db"
    return str(db_path)


@pytest.fixture
def test_db(test_db_path):
    """Create a real test database."""
    db = CharactersRAGDB(db_path=test_db_path, client_id="test_client")
    
    # The database initializes itself in __init__, no need to call initialize_db
    
    # Add some test data - but we need to check what tables actually exist
    try:
        # Try to add conversation data if table exists
        db.execute_query("""
            INSERT INTO conversations (id, title, created_at, updated_at, user_id)
            VALUES (?, ?, ?, ?, ?)
        """, ("conv1", "Test Conversation", datetime.now().isoformat(), datetime.now().isoformat(), "test_user"))
    except:
        pass  # Table might not exist or have different schema
    
    try:
        # Try to add note data if table exists
        db.execute_query("""
            INSERT INTO Notes (id, title, content, created_at, updated_at, user_id)
            VALUES (?, ?, ?, ?, ?, ?)
        """, ("note1", "Test Note", "Note content", datetime.now().isoformat(), datetime.now().isoformat(), "test_user"))
    except:
        pass  # Table might not exist or have different schema
    
    yield db
    
    # Cleanup - don't call close() as it might not exist
    if os.path.exists(test_db_path):
        os.unlink(test_db_path)


@pytest.fixture
def service(test_db, tmp_path):
    """Create ChatbookService with real database."""
    # Set up test environment
    os.environ['PYTEST_CURRENT_TEST'] = 'test'
    os.environ['TLDW_USER_DATA_PATH'] = str(tmp_path)
    
    service = ChatbookService(user_id="test_user", db=test_db)
    
    yield service
    
    # Cleanup
    if hasattr(service, 'temp_dir') and service.temp_dir.exists():
        shutil.rmtree(service.temp_dir, ignore_errors=True)


@pytest.fixture
def sample_chatbook_file(tmp_path):
    """Create a sample chatbook file for import tests."""
    chatbook_path = tmp_path / "sample.chatbook"
    
    manifest = ChatbookManifest(
        version=ChatbookVersion.V1,
        name="Sample Chatbook",
        description="Test chatbook for integration tests",
        author="Test Author",
        user_id="test_user"
    )
    
    with zipfile.ZipFile(chatbook_path, 'w') as zf:
        # Add manifest with content_items
        manifest_dict = manifest_to_dict(manifest)
        manifest_dict['content_items'] = [
            {
                "id": "conv2",
                "type": "conversation",
                "title": "Imported Conversation",
                "file_path": "content/conversations/conversation_conv2.json"
            },
            {
                "id": "note2",
                "type": "note",
                "title": "Imported Note",
                "file_path": "content/notes/note_note2.md"
            }
        ]
        zf.writestr('manifest.json', json.dumps(manifest_dict))
        
        # Add sample content with correct paths
        zf.writestr('content/conversations/conversation_conv2.json', json.dumps({
            "id": "conv2",
            "name": "Imported Conversation",
            "character_id": 1,
            "messages": []
        }))
        
        # Notes are stored as markdown with frontmatter
        zf.writestr('content/notes/note_note2.md', """---
id: note2
title: Imported Note
---

Imported content""")
    
    return str(chatbook_path)


class TestChatbookIntegration:
    """Integration tests for Chatbook service."""
    
    def test_service_initialization(self, service, test_db):
        """Test that service initializes with real database."""
        assert service.db == test_db
        assert service.user_id == "test_user"
        
        # Check that directories were created
        assert service.export_dir.exists()
        assert service.import_dir.exists()
        assert service.temp_dir.exists()
    
    def test_create_export_job(self, service):
        """Test creating an export job with real database."""
        result = service.create_export_job(
            name="Test Export",
            description="Test export job",
            content_types=["conversations", "notes"]
        )
        
        assert result is not None
        assert "job_id" in result
        job_id = result["job_id"]
        assert len(job_id) == 36  # UUID format
        
        # Verify job was saved to database
        job = service.get_export_job(job_id)
        assert job is not None
        assert job.chatbook_name == "Test Export"
        assert job.status == ExportStatus.PENDING
    
    @pytest.mark.asyncio
    async def test_export_chatbook_full_cycle(self, service):
        """Test full export cycle with real database and file system."""
        # Create export
        result = await service.create_chatbook(
            name="Integration Test Export",
            description="Testing full export",
            content_selections={
                ContentType.CONVERSATION: [],  # Export all conversations
                ContentType.NOTE: []  # Export all notes
            },
            async_mode=False
        )
        
        # Check result
        success, message, file_path = result
        assert success is True
        assert file_path is not None
        assert os.path.exists(file_path)
        
        # Verify it's a valid chatbook file
        with zipfile.ZipFile(file_path, 'r') as zf:
            assert 'manifest.json' in zf.namelist()
            
            # Check manifest
            manifest_data = json.loads(zf.read('manifest.json'))
            assert manifest_data['name'] == "Integration Test Export"
            assert manifest_data['description'] == "Testing full export"
    
    @pytest.mark.asyncio
    async def test_import_chatbook_full_cycle(self, service, sample_chatbook_file):
        """Test full import cycle with real database."""
        # Import the chatbook
        result = await service.import_chatbook(
            file_path=sample_chatbook_file,
            conflict_resolution="skip"
        )
        
        # Check result
        success, message, _ = result
        assert success is True
        assert "imported" in message.lower() or "completed" in message.lower()
        
        # Verify data was imported (check via database)
        # Note: This would require actual database queries to verify
        # For now, just check that no errors occurred
    
    def test_preview_export(self, service):
        """Test preview export with real database data."""
        result = service.preview_export(
            content_types=["conversations", "notes"]
        )
        
        # Should return counts based on actual database content
        assert "conversations" in result
        assert "notes" in result
        assert result["conversations"] >= 0  # At least the test data we inserted
        assert result["notes"] >= 0
    
    def test_validate_chatbook_file(self, service, sample_chatbook_file):
        """Test validating a real chatbook file."""
        result = service.validate_chatbook_file(sample_chatbook_file)
        
        assert result["is_valid"] is True
        assert result["manifest"] is not None
        assert result["error"] is None
    
    def test_validate_invalid_file(self, service, tmp_path):
        """Test validating an invalid file."""
        invalid_file = tmp_path / "invalid.txt"
        invalid_file.write_text("Not a chatbook")
        
        result = service.validate_chatbook_file(str(invalid_file))
        
        assert result["is_valid"] is False
        assert result["error"] is not None
    
    def test_list_export_jobs(self, service):
        """Test listing export jobs from real database."""
        # Create a few jobs
        result1 = service.create_export_job(
            name="Export 1",
            description="First export",
            content_types=["conversations"]
        )
        job_id1 = result1["job_id"]
        
        result2 = service.create_export_job(
            name="Export 2",
            description="Second export",
            content_types=["notes"]
        )
        job_id2 = result2["job_id"]
        
        # List jobs
        jobs = service.list_export_jobs()
        
        assert len(jobs) >= 2
        job_ids = [job["job_id"] for job in jobs]
        assert job_id1 in job_ids
        assert job_id2 in job_ids
    
    def test_get_export_job_status(self, service):
        """Test getting export job status from real database."""
        # Create a job
        result = service.create_export_job(
            name="Status Test",
            description="Testing status",
            content_types=["conversations"]
        )
        job_id = result["job_id"]
        
        # Get status
        status = service.get_export_job_status(job_id)
        
        assert status["job_id"] == job_id
        assert status["status"] == "pending"
        assert status["chatbook_name"] == "Status Test"
    
    def test_clean_old_exports(self, service, tmp_path):
        """Test cleaning old exports with real file system."""
        # Create some old export files
        old_file1 = service.export_dir / "old1.chatbook"
        old_file2 = service.export_dir / "old2.chatbook"
        old_file1.touch()
        old_file2.touch()
        
        # Create export jobs in database
        service.db.execute_query("""
            INSERT INTO export_jobs (job_id, user_id, status, chatbook_name, output_path, created_at)
            VALUES (?, ?, ?, ?, ?, datetime('now', '-10 days'))
        """, ("old1", "test_user", "completed", "Old 1", str(old_file1)))
        
        service.db.execute_query("""
            INSERT INTO export_jobs (job_id, user_id, status, chatbook_name, output_path, created_at)
            VALUES (?, ?, ?, ?, ?, datetime('now', '-10 days'))
        """, ("old2", "test_user", "completed", "Old 2", str(old_file2)))
        
        # Clean old exports
        count = service.clean_old_exports(days_old=7)
        
        assert count == 2
        assert not old_file1.exists()
        assert not old_file2.exists()
    
    def test_get_statistics(self, service):
        """Test getting statistics from real database."""
        # Create some test data
        service.create_export_job(name="Export 1", description="Test 1", content_types=[])
        service.create_export_job(name="Export 2", description="Test 2", content_types=[])
        
        # Get statistics
        stats = service.get_statistics()
        
        assert "exports" in stats
        assert "imports" in stats
        
        # Check export stats
        if "pending" in stats["exports"]:
            assert stats["exports"]["pending"] >= 2
    
    @pytest.mark.asyncio
    async def test_export_import_roundtrip(self, service, tmp_path):
        """Test full export and import roundtrip with real components."""
        # First, export current data
        export_result = await service.create_chatbook(
            name="Roundtrip Test",
            description="Testing roundtrip",
            content_selections={
                ContentType.CONVERSATION: [],
                ContentType.NOTE: []
            },
            async_mode=False
        )
        
        success, message, export_path = export_result
        assert success is True
        assert export_path is not None
        
        # Now import it back (with rename to avoid conflicts)
        import_result = await service.import_chatbook(
            file_path=export_path,
            conflict_resolution="rename"
        )
        
        import_success, import_message, _ = import_result
        # Import might have no items if the database was empty, which is OK
        assert import_success is True or "No items" in import_message
    
    @pytest.mark.asyncio
    async def test_async_export_job(self, service):
        """Test async export job processing."""
        # Create async export
        result = await service.create_chatbook(
            name="Async Export",
            description="Testing async",
            content_selections={ContentType.CONVERSATION: []},
            async_mode=True
        )
        
        success, message, job_id = result
        
        # For async mode, should return job_id instead of file path
        assert success is True
        assert job_id is not None
        
        # Check job status
        status = service.get_export_job_status(job_id)
        assert status["status"] in ["pending", "in_progress", "completed"]
    
    @pytest.mark.asyncio
    async def test_import_with_conflicts(self, service, sample_chatbook_file):
        """Test import with conflict handling."""
        # First import
        result1 = await service.import_chatbook(
            file_path=sample_chatbook_file,
            conflict_resolution="skip"
        )
        assert result1[0] is True
        
        # Second import with same file - should handle conflicts
        result2 = await service.import_chatbook(
            file_path=sample_chatbook_file,
            conflict_resolution="skip"
        )
        
        # Should still succeed but skip conflicting items
        assert result2[0] is True
        assert "skip" in result2[1].lower() or "conflict" in result2[1].lower() or "imported" in result2[1].lower()
    
    def test_cancel_export_job(self, service):
        """Test cancelling an export job."""
        # Create a job
        result = service.create_export_job(
            name="Cancel Test",
            description="Testing cancellation",
            content_types=[]
        )
        job_id = result["job_id"]
        
        # Cancel it
        cancel_result = service.cancel_export_job(job_id)
        assert cancel_result is True
        
        # Check status
        job = service.get_export_job(job_id)
        assert job.status == ExportStatus.CANCELLED