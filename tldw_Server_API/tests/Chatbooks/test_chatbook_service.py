# test_chatbook_service.py
# Description: Unit tests for the ChatbookService (combined import/export)
#
"""
Chatbook Service Tests
----------------------

Comprehensive unit tests for the chatbook import/export functionality including
job management, conflict resolution, and user isolation.
"""

import pytest
import json
import tempfile
import zipfile
from unittest.mock import MagicMock, patch, AsyncMock, call
from datetime import datetime
from uuid import uuid4
from pathlib import Path

from tldw_Server_API.app.core.Chatbooks.chatbook_service import ChatbookService
from tldw_Server_API.app.core.Chatbooks.chatbook_models import (
    ChatbookManifest,
    ContentItem,
    ExportJob,
    ImportJob,
    ExportStatus,
    ImportStatus
)
from tldw_Server_API.app.core.DB_Management.ChaChaNotes_DB import CharactersRAGDB


@pytest.fixture
def mock_db():
    """Create a mock database instance."""
    mock = MagicMock()
    mock.execute_query = MagicMock()
    mock.execute_many = MagicMock()
    return mock


@pytest.fixture
def service(mock_db):
    """Create a ChatbookService instance with mocked database."""
    return ChatbookService(mock_db, user_id="test_user")


@pytest.fixture
def sample_manifest():
    """Create a sample chatbook manifest."""
    return ChatbookManifest(
        version="1.0.0",
        exported_at=datetime.now().isoformat(),
        user_id="test_user",
        name="Test Chatbook",
        description="Test chatbook for unit tests",
        content_summary={
            "conversations": 2,
            "characters": 1,
            "world_books": 1,
            "dictionaries": 1,
            "notes": 2,
            "prompts": 3
        },
        metadata={"test": "data"}
    )


@pytest.fixture
def sample_content_items():
    """Create sample content items."""
    return [
        ContentItem(
            id="conv1",
            type="conversation",
            name="Test Conversation",
            content={"messages": ["Hello", "Hi"]},
            metadata={"created_at": "2024-01-01"},
            user_id="test_user"
        ),
        ContentItem(
            id="char1",
            type="character",
            name="Test Character",
            content={"description": "A test character"},
            metadata={},
            user_id="test_user"
        )
    ]


class TestChatbookService:
    """Test suite for ChatbookService."""
    
    def test_init_creates_tables(self, mock_db):
        """Test that initialization creates necessary tables."""
        service = ChatbookService(mock_db, "test_user")
        
        # Should create export and import job tables
        assert mock_db.execute_query.call_count >= 2
        
        # Check table creation
        calls = mock_db.execute_query.call_args_list
        sql_statements = [call[0][0] for call in calls]
        assert any("CREATE TABLE IF NOT EXISTS export_jobs" in sql for sql in sql_statements)
        assert any("CREATE TABLE IF NOT EXISTS import_jobs" in sql for sql in sql_statements)
    
    def test_create_export_job(self, service, mock_db):
        """Test creating an export job."""
        job_id = str(uuid4())
        
        with patch('uuid.uuid4', return_value=job_id):
            mock_db.execute_query.return_value = None
            
            result = service.create_export_job(
                name="Test Export",
                description="Test export job",
                content_types=["conversations", "characters"]
            )
        
        assert result["job_id"] == job_id
        assert result["status"] == "pending"
        
        # Check job was saved
        call_args = mock_db.execute_query.call_args[0]
        assert "INSERT INTO export_jobs" in call_args[0]
    
    @pytest.mark.asyncio
    async def test_export_chatbook_sync(self, service, mock_db):
        """Test synchronous chatbook export."""
        # Mock content retrieval
        mock_db.execute_query.side_effect = [
            [{"id": "conv1", "title": "Test Conv", "created_at": "2024-01-01"}],  # Conversations
            [],  # Characters  
            [],  # World books
            [],  # Dictionaries
            [],  # Notes
            []   # Prompts
        ]
        
        with patch.object(service, '_create_chatbook_archive') as mock_archive:
            mock_archive.return_value = "/tmp/test.chatbook"
            
            result = await service.export_chatbook(
                name="Test Export",
                content_types=["conversations"]
            )
        
        assert result["success"] == True
        assert result["file_path"] == "/tmp/test.chatbook"
        assert result["content_summary"]["conversations"] == 1
    
    @pytest.mark.asyncio
    async def test_export_chatbook_async_job(self, service, mock_db):
        """Test asynchronous chatbook export with job management."""
        job_id = str(uuid4())
        
        with patch('uuid.uuid4', return_value=job_id):
            mock_db.execute_query.return_value = None
            
            result = await service.export_chatbook(
                name="Test Export",
                content_types=["conversations"],
                async_job=True
            )
        
        assert result["job_id"] == job_id
        assert result["status"] == "pending"
        assert result["message"] == "Export job created successfully"
    
    def test_preview_export(self, service, mock_db):
        """Test previewing export content."""
        mock_db.execute_query.side_effect = [
            [{"id": "conv1"}, {"id": "conv2"}],  # Conversations
            [{"id": "char1"}],  # Characters
            [],  # World books
            [],  # Dictionaries
            [{"id": "note1"}, {"id": "note2"}],  # Notes
            []   # Prompts
        ]
        
        result = service.preview_export(
            content_types=["conversations", "characters", "notes"]
        )
        
        assert result["conversations"] == 2
        assert result["characters"] == 1
        assert result["notes"] == 2
        assert result["world_books"] == 0
    
    def test_get_export_job_status(self, service, mock_db):
        """Test retrieving export job status."""
        mock_db.execute_query.return_value = [{
            "job_id": "job123",
            "status": "completed",
            "file_path": "/tmp/export.chatbook",
            "content_summary": json.dumps({"conversations": 5}),
            "created_at": "2024-01-01T00:00:00",
            "completed_at": "2024-01-01T00:05:00",
            "error_message": None
        }]
        
        result = service.get_export_job_status("job123")
        
        assert result["job_id"] == "job123"
        assert result["status"] == "completed"
        assert result["file_path"] == "/tmp/export.chatbook"
        assert result["content_summary"]["conversations"] == 5
    
    def test_cancel_export_job(self, service, mock_db):
        """Test cancelling an export job."""
        mock_db.execute_query.return_value = None
        
        result = service.cancel_export_job("job123")
        
        assert result == True
        
        # Check job was updated
        call_args = mock_db.execute_query.call_args[0]
        assert "UPDATE export_jobs SET status = ?" in call_args[0]
        assert "cancelled" in call_args[1]
    
    @pytest.mark.asyncio
    async def test_import_chatbook_with_conflicts(self, service, mock_db, sample_manifest):
        """Test importing a chatbook with conflict resolution."""
        # Create a test chatbook file
        with tempfile.NamedTemporaryFile(suffix='.chatbook', delete=False) as tmp:
            with zipfile.ZipFile(tmp.name, 'w') as zf:
                zf.writestr('manifest.json', json.dumps(sample_manifest.__dict__))
                zf.writestr('conversations/conv1.json', json.dumps({
                    "id": "conv1",
                    "title": "Test Conversation",
                    "messages": []
                }))
            
            # Mock checking for conflicts
            mock_db.execute_query.side_effect = [
                [{"id": "conv1"}],  # Existing conversation (conflict)
                None  # Import operation
            ]
            
            result = await service.import_chatbook(
                file_path=tmp.name,
                conflict_strategy="skip"
            )
        
        assert result["success"] == True
        assert result["conflicts_found"] > 0
        assert result["conflicts_resolved"]["skipped"] > 0
    
    @pytest.mark.asyncio
    async def test_import_chatbook_replace_strategy(self, service, mock_db, sample_manifest):
        """Test importing with replace conflict strategy."""
        with tempfile.NamedTemporaryFile(suffix='.chatbook', delete=False) as tmp:
            with zipfile.ZipFile(tmp.name, 'w') as zf:
                zf.writestr('manifest.json', json.dumps(sample_manifest.__dict__))
                zf.writestr('characters/char1.json', json.dumps({
                    "id": "char1",
                    "name": "New Character"
                }))
            
            mock_db.execute_query.side_effect = [
                [{"id": "char1"}],  # Existing character
                None  # Replace operation
            ]
            
            result = await service.import_chatbook(
                file_path=tmp.name,
                conflict_strategy="replace"
            )
        
        assert result["success"] == True
        assert result["conflicts_resolved"]["replaced"] > 0
    
    @pytest.mark.asyncio
    async def test_import_chatbook_rename_strategy(self, service, mock_db, sample_manifest):
        """Test importing with rename conflict strategy."""
        with tempfile.NamedTemporaryFile(suffix='.chatbook', delete=False) as tmp:
            with zipfile.ZipFile(tmp.name, 'w') as zf:
                zf.writestr('manifest.json', json.dumps(sample_manifest.__dict__))
                zf.writestr('notes/note1.json', json.dumps({
                    "id": "note1",
                    "title": "Test Note"
                }))
            
            mock_db.execute_query.side_effect = [
                [{"id": "note1"}],  # Existing note
                None  # Import with new name
            ]
            
            result = await service.import_chatbook(
                file_path=tmp.name,
                conflict_strategy="rename"
            )
        
        assert result["success"] == True
        assert result["conflicts_resolved"]["renamed"] > 0
    
    def test_create_import_job(self, service, mock_db):
        """Test creating an import job."""
        job_id = str(uuid4())
        
        with patch('uuid.uuid4', return_value=job_id):
            mock_db.execute_query.return_value = None
            
            result = service.create_import_job(
                file_path="/tmp/test.chatbook",
                conflict_strategy="skip"
            )
        
        assert result["job_id"] == job_id
        assert result["status"] == "pending"
    
    def test_get_import_job_status(self, service, mock_db):
        """Test retrieving import job status."""
        mock_db.execute_query.return_value = [{
            "job_id": "job456",
            "status": "completed",
            "file_path": "/tmp/import.chatbook",
            "conflict_strategy": "skip",
            "items_imported": 10,
            "conflicts_found": 2,
            "conflicts_resolved": json.dumps({"skipped": 2}),
            "created_at": "2024-01-01T00:00:00",
            "completed_at": "2024-01-01T00:10:00",
            "error_message": None
        }]
        
        result = service.get_import_job_status("job456")
        
        assert result["job_id"] == "job456"
        assert result["status"] == "completed"
        assert result["items_imported"] == 10
        assert result["conflicts_found"] == 2
        assert result["conflicts_resolved"]["skipped"] == 2
    
    def test_list_export_jobs(self, service, mock_db):
        """Test listing export jobs."""
        mock_db.execute_query.return_value = [
            {"job_id": "job1", "status": "completed", "name": "Export 1"},
            {"job_id": "job2", "status": "pending", "name": "Export 2"}
        ]
        
        results = service.list_export_jobs(status="all")
        
        assert len(results) == 2
        assert results[0]["name"] == "Export 1"
        assert results[1]["status"] == "pending"
    
    def test_list_import_jobs(self, service, mock_db):
        """Test listing import jobs."""
        mock_db.execute_query.return_value = [
            {"job_id": "job3", "status": "completed", "items_imported": 5},
            {"job_id": "job4", "status": "failed", "error_message": "File not found"}
        ]
        
        results = service.list_import_jobs(limit=10)
        
        assert len(results) == 2
        assert results[0]["items_imported"] == 5
        assert results[1]["error_message"] == "File not found"
    
    def test_clean_old_exports(self, service, mock_db):
        """Test cleaning old export files."""
        mock_db.execute_query.return_value = [
            {"file_path": "/tmp/old1.chatbook"},
            {"file_path": "/tmp/old2.chatbook"}
        ]
        
        with patch('os.path.exists', return_value=True):
            with patch('os.remove') as mock_remove:
                result = service.clean_old_exports(days_old=30)
        
        assert result["files_deleted"] == 2
        assert mock_remove.call_count == 2
    
    def test_validate_chatbook_file(self, service, sample_manifest):
        """Test validating a chatbook file structure."""
        with tempfile.NamedTemporaryFile(suffix='.chatbook', delete=False) as tmp:
            with zipfile.ZipFile(tmp.name, 'w') as zf:
                zf.writestr('manifest.json', json.dumps(sample_manifest.__dict__))
                zf.writestr('conversations/test.json', '{}')
            
            result = service.validate_chatbook(tmp.name)
        
        assert result["valid"] == True
        assert result["version"] == "1.0.0"
        assert "conversations" in result["content_types"]
    
    def test_validate_invalid_chatbook(self, service):
        """Test validating an invalid chatbook file."""
        with tempfile.NamedTemporaryFile(suffix='.txt', delete=False) as tmp:
            tmp.write(b"Not a zip file")
            tmp.flush()
            
            result = service.validate_chatbook(tmp.name)
        
        assert result["valid"] == False
        assert "error" in result
    
    def test_get_statistics(self, service, mock_db):
        """Test getting import/export statistics."""
        mock_db.execute_query.side_effect = [
            [{"total_exports": 50, "completed_exports": 45}],
            [{"total_imports": 30, "completed_imports": 28}],
            [{"total_items_exported": 500}],
            [{"total_items_imported": 350}]
        ]
        
        stats = service.get_statistics()
        
        assert stats["total_exports"] == 50
        assert stats["export_success_rate"] == 45 / 50
        assert stats["total_imports"] == 30
        assert stats["import_success_rate"] == 28 / 30
        assert stats["total_items_exported"] == 500
        assert stats["total_items_imported"] == 350
    
    @pytest.mark.asyncio
    async def test_error_handling_during_export(self, service, mock_db):
        """Test error handling during export."""
        mock_db.execute_query.side_effect = Exception("Database error")
        
        result = await service.export_chatbook(
            name="Test Export",
            content_types=["conversations"]
        )
        
        assert result["success"] == False
        assert "error" in result
        assert "Database error" in result["error"]
    
    @pytest.mark.asyncio
    async def test_error_handling_during_import(self, service):
        """Test error handling during import."""
        result = await service.import_chatbook(
            file_path="/nonexistent/file.chatbook",
            conflict_strategy="skip"
        )
        
        assert result["success"] == False
        assert "error" in result
    
    def test_user_isolation(self, service, mock_db):
        """Test that operations are isolated to the current user."""
        # Test export - should only get current user's content
        mock_db.execute_query.return_value = []
        
        service.preview_export(content_types=["conversations"])
        
        # Check that user_id is in the query
        call_args = mock_db.execute_query.call_args[0]
        assert "WHERE user_id = ?" in call_args[0] or "test_user" in str(call_args)
    
    def test_create_chatbook_archive(self, service, sample_manifest, sample_content_items):
        """Test creating a chatbook archive file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            file_path = Path(tmpdir) / "test.chatbook"
            
            # Mock the internal method
            with patch.object(service, '_write_content_to_archive'):
                result = service._create_chatbook_archive(
                    manifest=sample_manifest,
                    content_items=sample_content_items,
                    output_path=str(file_path)
                )
            
            assert result == str(file_path)
            assert file_path.exists()
            
            # Verify it's a valid zip file
            with zipfile.ZipFile(file_path, 'r') as zf:
                assert 'manifest.json' in zf.namelist()
    
    def test_process_import_items(self, service, mock_db, sample_content_items):
        """Test processing individual import items."""
        mock_db.execute_query.return_value = []  # No conflicts
        
        results = service._process_import_items(
            items=sample_content_items,
            conflict_strategy="skip"
        )
        
        assert results["imported"] == len(sample_content_items)
        assert results["skipped"] == 0
        assert results["replaced"] == 0
        assert results["renamed"] == 0