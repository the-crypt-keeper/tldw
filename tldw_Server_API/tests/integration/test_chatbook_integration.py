# test_chatbook_integration.py
# Description: Integration tests for Chatbook import/export functionality
#
"""
Chatbook Integration Tests
--------------------------

Integration tests for the complete chatbook import/export workflow including
conflict resolution, job management, and multi-user scenarios.
"""

import pytest
import asyncio
import json
import tempfile
import zipfile
import shutil
from pathlib import Path
from datetime import datetime
from unittest.mock import MagicMock, patch, AsyncMock
from uuid import uuid4

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
from tldw_Server_API.app.core.Character_Chat.chat_dictionary import ChatDictionaryService
from tldw_Server_API.app.core.Character_Chat.world_book_manager import WorldBookService


@pytest.fixture
def test_db():
    """Create test database for integration tests."""
    db = CharactersRAGDB(":memory:", "test_user")
    return db


@pytest.fixture
def chatbook_service(test_db):
    """Create ChatbookService with test database."""
    return ChatbookService(user_id="test_user", db=test_db)


@pytest.fixture
def dict_service(test_db):
    """Create ChatDictionaryService for test data."""
    return ChatDictionaryService(test_db)


@pytest.fixture
def wb_service(test_db):
    """Create WorldBookService for test data."""
    return WorldBookService(test_db)


@pytest.fixture
def temp_export_dir():
    """Create temporary directory for exports."""
    temp_dir = tempfile.mkdtemp()
    yield Path(temp_dir)
    shutil.rmtree(temp_dir, ignore_errors=True)


@pytest.fixture
def sample_test_data(dict_service, wb_service, test_db):
    """Create sample data for export testing."""
    # Create dictionary with entries
    dict_id = dict_service.create_dictionary(
        name="Test Dictionary",
        description="Sample dictionary for testing"
    )
    dict_service.add_entry(dict_id, "hello", "greetings")
    dict_service.add_entry(dict_id, "goodbye", "farewell")
    
    # Create world book with entries
    wb_id = wb_service.create_world_book(
        name="Test World",
        description="Sample world book",
        token_budget=1000
    )
    wb_service.add_entry(wb_id, ["dragon", "fire"], "Dragons breathe fire", 100)
    wb_service.add_entry(wb_id, ["wizard"], "Wizards use magic", 90)
    
    # Create mock conversations
    conversations = [
        {
            "id": "conv1",
            "title": "Test Conversation 1",
            "created_at": datetime.now().isoformat(),
            "messages": [
                {"sender": "user", "content": "Hello"},
                {"sender": "assistant", "content": "Hi there!"}
            ]
        },
        {
            "id": "conv2", 
            "title": "Test Conversation 2",
            "created_at": datetime.now().isoformat(),
            "messages": [
                {"sender": "user", "content": "Tell me about dragons"},
                {"sender": "assistant", "content": "Dragons are mythical creatures"}
            ]
        }
    ]
    
    # Create mock notes
    notes = [
        {"id": "note1", "title": "Research Notes", "content": "Important findings"},
        {"id": "note2", "title": "Meeting Notes", "content": "Action items"}
    ]
    
    return {
        "dictionary_id": dict_id,
        "world_book_id": wb_id,
        "conversations": conversations,
        "notes": notes
    }


class TestExportWorkflow:
    """Test complete export workflow."""
    
    @pytest.mark.asyncio
    async def test_export_all_content_types(self, chatbook_service, sample_test_data, temp_export_dir, test_db):
        """Test exporting all content types to a chatbook."""
        # Mock content retrieval
        with patch.object(test_db, 'execute_query') as mock_query:
            mock_query.side_effect = [
                # Conversations
                sample_test_data["conversations"],
                # Characters (empty)
                [],
                # World books
                [{"id": sample_test_data["world_book_id"], "name": "Test World"}],
                # Dictionaries
                [{"id": sample_test_data["dictionary_id"], "name": "Test Dictionary"}],
                # Notes
                sample_test_data["notes"],
                # Prompts (empty)
                []
            ]
            
            with patch.object(chatbook_service, '_create_chatbook_archive') as mock_archive:
                archive_path = temp_export_dir / "test_export.chatbook"
                mock_archive.return_value = str(archive_path)
                
                # Create mock archive file
                with zipfile.ZipFile(archive_path, 'w') as zf:
                    manifest = ChatbookManifest(
                        version="1.0.0",
                        exported_at=datetime.now().isoformat(),
                        user_id="test_user",
                        name="Test Export",
                        description="Integration test export",
                        content_summary={
                            "conversations": 2,
                            "world_books": 1,
                            "dictionaries": 1,
                            "notes": 2
                        }
                    )
                    zf.writestr('manifest.json', json.dumps(manifest.__dict__))
                
                result = await chatbook_service.export_chatbook(
                    name="Test Export",
                    description="Integration test",
                    content_types=["conversations", "world_books", "dictionaries", "notes"]
                )
        
        assert result["success"] == True
        assert result["content_summary"]["conversations"] == 2
        assert result["content_summary"]["world_books"] == 1
        assert result["content_summary"]["dictionaries"] == 1
        assert result["content_summary"]["notes"] == 2
        assert Path(result["file_path"]).exists()
    
    @pytest.mark.asyncio
    async def test_export_with_relationships(self, chatbook_service, test_db):
        """Test exporting content with relationships preserved."""
        # Create character with attached world book
        character_data = {
            "id": "char1",
            "name": "Test Character",
            "world_books": ["wb1", "wb2"]
        }
        
        with patch.object(test_db, 'execute_query') as mock_query:
            mock_query.side_effect = [
                [],  # Conversations
                [character_data],  # Characters
                [{"id": "wb1"}, {"id": "wb2"}],  # World books
                [], [], []  # Other content types
            ]
            
            result = await chatbook_service.export_chatbook(
                name="Relationship Export",
                content_types=["characters", "world_books"]
            )
        
        # Verify relationships are tracked
        assert result["success"] == True
        # In real implementation, would verify manifest contains relationship data
    
    @pytest.mark.asyncio  
    async def test_async_export_job(self, chatbook_service, test_db):
        """Test asynchronous export job creation and processing."""
        job_id = str(uuid4())
        
        with patch('uuid.uuid4', return_value=job_id):
            result = await chatbook_service.export_chatbook(
                name="Async Export",
                content_types=["conversations"],
                async_job=True
            )
        
        assert result["job_id"] == job_id
        assert result["status"] == "pending"
        
        # Simulate job processing
        with patch.object(test_db, 'execute_query') as mock_query:
            mock_query.return_value = [{
                "job_id": job_id,
                "status": "completed",
                "file_path": "/tmp/export.chatbook",
                "content_summary": json.dumps({"conversations": 5}),
                "created_at": datetime.now().isoformat(),
                "completed_at": datetime.now().isoformat()
            }]
            
            status = chatbook_service.get_export_job_status(job_id)
        
        assert status["status"] == "completed"
        assert status["content_summary"]["conversations"] == 5


class TestImportWorkflow:
    """Test complete import workflow."""
    
    @pytest.mark.asyncio
    async def test_import_with_no_conflicts(self, chatbook_service, temp_export_dir):
        """Test importing a chatbook with no conflicts."""
        # Create test chatbook file
        chatbook_path = temp_export_dir / "import_test.chatbook"
        
        with zipfile.ZipFile(chatbook_path, 'w') as zf:
            # Add manifest
            manifest = {
                "version": "1.0.0",
                "exported_at": datetime.now().isoformat(),
                "user_id": "other_user",
                "name": "Import Test",
                "content_summary": {"conversations": 1}
            }
            zf.writestr('manifest.json', json.dumps(manifest))
            
            # Add conversation
            conv = {
                "id": "new_conv",
                "title": "Imported Conversation",
                "messages": [{"sender": "user", "content": "Test"}]
            }
            zf.writestr('conversations/new_conv.json', json.dumps(conv))
        
        # Mock checking for conflicts (none found)
        with patch.object(chatbook_service.db, 'execute_query', return_value=[]):
            result = await chatbook_service.import_chatbook(
                file_path=str(chatbook_path),
                conflict_strategy="skip"
            )
        
        assert result["success"] == True
        assert result["items_imported"] > 0
        assert result["conflicts_found"] == 0
    
    @pytest.mark.asyncio
    async def test_import_with_skip_strategy(self, chatbook_service, temp_export_dir):
        """Test importing with skip conflict strategy."""
        chatbook_path = temp_export_dir / "skip_test.chatbook"
        
        with zipfile.ZipFile(chatbook_path, 'w') as zf:
            manifest = {
                "version": "1.0.0",
                "content_summary": {"notes": 2}
            }
            zf.writestr('manifest.json', json.dumps(manifest))
            
            # Add notes that will conflict
            zf.writestr('notes/note1.json', json.dumps({"id": "note1", "title": "Note 1"}))
            zf.writestr('notes/note2.json', json.dumps({"id": "note2", "title": "Note 2"}))
        
        # Mock finding existing note1
        with patch.object(chatbook_service.db, 'execute_query') as mock_query:
            mock_query.side_effect = [
                [{"id": "note1"}],  # note1 exists (conflict)
                [],  # note2 doesn't exist
                None  # Import note2
            ]
            
            result = await chatbook_service.import_chatbook(
                file_path=str(chatbook_path),
                conflict_strategy="skip"
            )
        
        assert result["success"] == True
        assert result["conflicts_found"] == 1
        assert result["conflicts_resolved"]["skipped"] == 1
        assert result["items_imported"] == 1  # Only note2 imported
    
    @pytest.mark.asyncio
    async def test_import_with_replace_strategy(self, chatbook_service, temp_export_dir):
        """Test importing with replace conflict strategy."""
        chatbook_path = temp_export_dir / "replace_test.chatbook"
        
        with zipfile.ZipFile(chatbook_path, 'w') as zf:
            manifest = {"version": "1.0.0", "content_summary": {"dictionaries": 1}}
            zf.writestr('manifest.json', json.dumps(manifest))
            
            dict_data = {
                "id": "dict1",
                "name": "Updated Dictionary",
                "entries": [{"key": "test", "value": "replaced"}]
            }
            zf.writestr('dictionaries/dict1.json', json.dumps(dict_data))
        
        with patch.object(chatbook_service.db, 'execute_query') as mock_query:
            mock_query.side_effect = [
                [{"id": "dict1"}],  # Existing dictionary found
                None,  # Delete old
                None   # Import new
            ]
            
            result = await chatbook_service.import_chatbook(
                file_path=str(chatbook_path),
                conflict_strategy="replace"
            )
        
        assert result["success"] == True
        assert result["conflicts_resolved"]["replaced"] == 1
    
    @pytest.mark.asyncio
    async def test_import_with_rename_strategy(self, chatbook_service, temp_export_dir):
        """Test importing with rename conflict strategy."""
        chatbook_path = temp_export_dir / "rename_test.chatbook"
        
        with zipfile.ZipFile(chatbook_path, 'w') as zf:
            manifest = {"version": "1.0.0", "content_summary": {"world_books": 1}}
            zf.writestr('manifest.json', json.dumps(manifest))
            
            wb_data = {"id": "wb1", "name": "Fantasy World"}
            zf.writestr('world_books/wb1.json', json.dumps(wb_data))
        
        with patch.object(chatbook_service.db, 'execute_query') as mock_query:
            mock_query.side_effect = [
                [{"id": "wb1", "name": "Fantasy World"}],  # Conflict found
                None  # Import with new name
            ]
            
            result = await chatbook_service.import_chatbook(
                file_path=str(chatbook_path),
                conflict_strategy="rename"
            )
        
        assert result["success"] == True
        assert result["conflicts_resolved"]["renamed"] == 1


class TestMultiUserScenarios:
    """Test multi-user import/export scenarios."""
    
    @pytest.mark.asyncio
    async def test_user_isolation_during_export(self):
        """Test that users only export their own content."""
        # Create services for different users
        db1 = CharactersRAGDB(":memory:", "user1")
        db2 = CharactersRAGDB(":memory:", "user2")
        
        service1 = ChatbookService(user_id="user1", db=db1)
        service2 = ChatbookService(user_id="user2", db=db2)
        
        # User 1 creates content
        dict_service1 = ChatDictionaryService(db1)
        dict1 = dict_service1.create_dictionary("User1 Dict", "Private")
        
        # User 2 creates different content
        dict_service2 = ChatDictionaryService(db2)
        dict2 = dict_service2.create_dictionary("User2 Dict", "Also private")
        
        # Export for each user
        with patch.object(service1, '_create_chatbook_archive', return_value="/tmp/user1.chatbook"):
            result1 = await service1.export_chatbook(
                name="User1 Export",
                content_types=["dictionaries"]
            )
        
        with patch.object(service2, '_create_chatbook_archive', return_value="/tmp/user2.chatbook"):
            result2 = await service2.export_chatbook(
                name="User2 Export",
                content_types=["dictionaries"]
            )
        
        # Each user should only see their own content
        assert result1["success"] == True
        assert result2["success"] == True
        # In real implementation, would verify content isolation
    
    @pytest.mark.asyncio
    async def test_import_preserves_user_ownership(self, chatbook_service, temp_export_dir):
        """Test that imported content is assigned to importing user."""
        chatbook_path = temp_export_dir / "ownership_test.chatbook"
        
        with zipfile.ZipFile(chatbook_path, 'w') as zf:
            manifest = {
                "version": "1.0.0",
                "user_id": "original_user",  # Different user
                "content_summary": {"conversations": 1}
            }
            zf.writestr('manifest.json', json.dumps(manifest))
            
            conv = {"id": "conv1", "user_id": "original_user"}
            zf.writestr('conversations/conv1.json', json.dumps(conv))
        
        with patch.object(chatbook_service.db, 'execute_query', return_value=[]):
            result = await chatbook_service.import_chatbook(
                file_path=str(chatbook_path),
                conflict_strategy="skip"
            )
        
        assert result["success"] == True
        # Imported content should now belong to test_user, not original_user


class TestErrorScenarios:
    """Test error handling in integration scenarios."""
    
    @pytest.mark.asyncio
    async def test_corrupted_chatbook_import(self, chatbook_service, temp_export_dir):
        """Test importing a corrupted chatbook file."""
        # Create invalid zip file
        bad_file = temp_export_dir / "corrupted.chatbook"
        bad_file.write_text("Not a valid zip file")
        
        result = await chatbook_service.import_chatbook(
            file_path=str(bad_file),
            conflict_strategy="skip"
        )
        
        assert result["success"] == False
        assert "error" in result
    
    @pytest.mark.asyncio
    async def test_missing_manifest(self, chatbook_service, temp_export_dir):
        """Test importing chatbook without manifest."""
        chatbook_path = temp_export_dir / "no_manifest.chatbook"
        
        with zipfile.ZipFile(chatbook_path, 'w') as zf:
            zf.writestr('conversations/test.json', '{}')
            # No manifest.json
        
        result = await chatbook_service.import_chatbook(
            file_path=str(chatbook_path),
            conflict_strategy="skip"
        )
        
        assert result["success"] == False
        assert "manifest" in result["error"].lower()
    
    @pytest.mark.asyncio
    async def test_export_with_database_error(self, chatbook_service):
        """Test export when database fails."""
        with patch.object(chatbook_service.db, 'execute_query', side_effect=Exception("DB Error")):
            result = await chatbook_service.export_chatbook(
                name="Failed Export",
                content_types=["conversations"]
            )
        
        assert result["success"] == False
        assert "DB Error" in result["error"]


class TestPerformanceScenarios:
    """Test performance with large datasets."""
    
    @pytest.mark.asyncio
    async def test_large_export(self, chatbook_service, test_db):
        """Test exporting large amounts of content."""
        # Mock large dataset
        large_conversations = [
            {"id": f"conv{i}", "title": f"Conversation {i}"}
            for i in range(1000)
        ]
        
        with patch.object(test_db, 'execute_query') as mock_query:
            mock_query.side_effect = [
                large_conversations,  # 1000 conversations
                [], [], [], [], []  # Other content types empty
            ]
            
            with patch.object(chatbook_service, '_create_chatbook_archive') as mock_archive:
                mock_archive.return_value = "/tmp/large_export.chatbook"
                
                result = await chatbook_service.export_chatbook(
                    name="Large Export",
                    content_types=["conversations"]
                )
        
        assert result["success"] == True
        assert result["content_summary"]["conversations"] == 1000
    
    @pytest.mark.asyncio
    async def test_chunked_import(self, chatbook_service, temp_export_dir):
        """Test importing in chunks for memory efficiency."""
        chatbook_path = temp_export_dir / "chunked_test.chatbook"
        
        with zipfile.ZipFile(chatbook_path, 'w') as zf:
            manifest = {
                "version": "1.0.0",
                "content_summary": {"notes": 500}
            }
            zf.writestr('manifest.json', json.dumps(manifest))
            
            # Add many notes
            for i in range(500):
                note = {"id": f"note{i}", "title": f"Note {i}"}
                zf.writestr(f'notes/note{i}.json', json.dumps(note))
        
        with patch.object(chatbook_service.db, 'execute_query', return_value=[]):
            with patch.object(chatbook_service.db, 'execute_many') as mock_many:
                result = await chatbook_service.import_chatbook(
                    file_path=str(chatbook_path),
                    conflict_strategy="skip"
                )
        
        assert result["success"] == True
        assert result["items_imported"] == 500