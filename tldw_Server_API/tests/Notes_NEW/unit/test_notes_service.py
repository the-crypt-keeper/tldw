"""
Unit tests for NotesInteropService.

Tests the core notes service functionality with minimal mocking -
only the database layer is mocked.
"""

import pytest
from unittest.mock import Mock, MagicMock, patch, call
from datetime import datetime
import sqlite3

from tldw_Server_API.app.core.Notes.Notes_Library import NotesInteropService
from tldw_Server_API.app.core.DB_Management.ChaChaNotes_DB import (
    CharactersRAGDBError,
    InputError,
    ConflictError
)

# ========================================================================
# Service Initialization Tests
# ========================================================================

class TestServiceInitialization:
    """Test service initialization and setup."""
    
    @pytest.mark.unit
    def test_service_initialization(self, test_db_path):
        """Test basic service initialization."""
        service = NotesInteropService(
            base_db_directory=str(test_db_path.parent),
            api_client_id="test_client"
        )
        
        assert service is not None
        assert service.api_client_id == "test_client"
        assert service.base_db_directory == test_db_path.parent
        assert service._db_instances == {}
        assert service._lock is not None
    
    @pytest.mark.unit
    def test_service_get_db_for_user(self, mock_notes_service):
        """Test getting database instance for a user."""
        service = mock_notes_service
        
        # First call should create instance
        db1 = service._get_db("user1")
        assert db1 is not None
        
        # Second call should return same instance
        db2 = service._get_db("user1")
        assert db1 is db2
    
    @pytest.mark.unit
    def test_service_thread_safety(self, mock_notes_service):
        """Test thread-safe database access."""
        service = mock_notes_service
        
        import threading
        results = []
        
        def get_db(user_id):
            db = service._get_db(user_id)
            results.append(db)
        
        threads = []
        for i in range(5):
            t = threading.Thread(target=get_db, args=(f"user{i}",))
            threads.append(t)
            t.start()
        
        for t in threads:
            t.join()
        
        # Should have 5 different DB instances
        assert len(set(id(r) for r in results)) == 5

# ========================================================================
# Note CRUD Operations Tests
# ========================================================================

class TestNoteCRUDOperations:
    """Test note creation, reading, updating, and deletion."""
    
    @pytest.mark.unit
    def test_create_note(self, mock_notes_service, sample_note):
        """Test note creation."""
        service = mock_notes_service
        
        note_id = service.create_note(**sample_note)
        
        assert note_id == 1
        service._db_instances["test_user"].create_note.assert_called_once_with(
            title=sample_note['title'],
            content=sample_note['content'],
            user_id=sample_note['user_id']
        )
    
    @pytest.mark.unit
    def test_get_note(self, mock_notes_service):
        """Test getting a note."""
        service = mock_notes_service
        
        note = service.get_note(note_id=1, user_id="test_user")
        
        assert note is not None
        assert note['id'] == 1
        assert note['title'] == 'Test Note'
        service._db_instances["test_user"].get_note.assert_called_once_with(1)
    
    @pytest.mark.unit
    def test_list_notes(self, mock_notes_service):
        """Test listing notes."""
        service = mock_notes_service
        mock_db = service._db_instances["test_user"]
        
        mock_db.list_notes.return_value = [
            {'id': 1, 'title': 'Note 1'},
            {'id': 2, 'title': 'Note 2'}
        ]
        
        notes = service.list_notes(user_id="test_user", limit=10, offset=0)
        
        assert len(notes) == 2
        assert notes[0]['title'] == 'Note 1'
        mock_db.list_notes.assert_called_once()
    
    @pytest.mark.unit
    def test_update_note(self, mock_notes_service):
        """Test updating a note."""
        service = mock_notes_service
        
        result = service.update_note(
            note_id=1,
            user_id="test_user",
            title="Updated Title",
            content="Updated content",
            expected_version=1
        )
        
        assert result['rows_affected'] == 1
        service._db_instances["test_user"].update_note.assert_called_once()
    
    @pytest.mark.unit
    def test_delete_note(self, mock_notes_service):
        """Test deleting a note."""
        service = mock_notes_service
        
        result = service.delete_note(note_id=1, user_id="test_user")
        
        assert result['success'] is True
        service._db_instances["test_user"].delete_note.assert_called_once_with(1)

# ========================================================================
# Keyword Operations Tests
# ========================================================================

class TestKeywordOperations:
    """Test keyword management operations."""
    
    @pytest.mark.unit
    def test_create_keyword(self, mock_notes_service, sample_keyword):
        """Test keyword creation."""
        service = mock_notes_service
        
        keyword_id = service.create_keyword(**sample_keyword)
        
        assert keyword_id == 1
        service._db_instances["test_user"].create_keyword.assert_called_once()
    
    @pytest.mark.unit
    def test_get_keyword(self, mock_notes_service):
        """Test getting a keyword."""
        service = mock_notes_service
        
        keyword = service.get_keyword(keyword_id=1, user_id="test_user")
        
        assert keyword is not None
        assert keyword['id'] == 1
        assert keyword['keyword'] == 'test-keyword'
    
    @pytest.mark.unit
    def test_list_keywords(self, mock_notes_service):
        """Test listing keywords."""
        service = mock_notes_service
        mock_db = service._db_instances["test_user"]
        
        mock_db.list_keywords.return_value = [
            {'id': 1, 'keyword': 'python'},
            {'id': 2, 'keyword': 'testing'}
        ]
        
        keywords = service.list_keywords(user_id="test_user")
        
        assert len(keywords) == 2
        assert keywords[0]['keyword'] == 'python'
    
    @pytest.mark.unit
    def test_delete_keyword(self, mock_notes_service):
        """Test deleting a keyword."""
        service = mock_notes_service
        
        result = service.delete_keyword(keyword_id=1, user_id="test_user")
        
        assert result['success'] is True
        service._db_instances["test_user"].delete_keyword.assert_called_once_with(1)

# ========================================================================
# Note-Keyword Linking Tests
# ========================================================================

class TestNoteKeywordLinking:
    """Test linking notes to keywords."""
    
    @pytest.mark.unit
    def test_link_note_keyword(self, mock_notes_service):
        """Test linking a note to a keyword."""
        service = mock_notes_service
        
        result = service.link_note_keyword(
            note_id=1,
            keyword_id=2,
            user_id="test_user"
        )
        
        assert result['success'] is True
        service._db_instances["test_user"].link_note_keyword.assert_called_once_with(1, 2)
    
    @pytest.mark.unit
    def test_unlink_note_keyword(self, mock_notes_service):
        """Test unlinking a note from a keyword."""
        service = mock_notes_service
        
        result = service.unlink_note_keyword(
            note_id=1,
            keyword_id=2,
            user_id="test_user"
        )
        
        assert result['success'] is True
        service._db_instances["test_user"].unlink_note_keyword.assert_called_once_with(1, 2)
    
    @pytest.mark.unit
    def test_get_keywords_for_note(self, mock_notes_service):
        """Test getting keywords for a note."""
        service = mock_notes_service
        mock_db = service._db_instances["test_user"]
        
        mock_db.get_keywords_for_note.return_value = [
            {'id': 1, 'keyword': 'python'},
            {'id': 2, 'keyword': 'testing'}
        ]
        
        keywords = service.get_keywords_for_note(note_id=1, user_id="test_user")
        
        assert len(keywords) == 2
        assert keywords[0]['keyword'] == 'python'
    
    @pytest.mark.unit
    def test_get_notes_for_keyword(self, mock_notes_service):
        """Test getting notes for a keyword."""
        service = mock_notes_service
        mock_db = service._db_instances["test_user"]
        
        mock_db.get_notes_for_keyword.return_value = [
            {'id': 1, 'title': 'Note 1'},
            {'id': 2, 'title': 'Note 2'}
        ]
        
        notes = service.get_notes_for_keyword(keyword_id=1, user_id="test_user")
        
        assert len(notes) == 2
        assert notes[0]['title'] == 'Note 1'

# ========================================================================
# Search Operations Tests
# ========================================================================

class TestSearchOperations:
    """Test search functionality."""
    
    @pytest.mark.unit
    def test_search_notes(self, mock_notes_service):
        """Test searching notes."""
        service = mock_notes_service
        mock_db = service._db_instances["test_user"]
        
        mock_db.search_notes.return_value = [
            {'id': 1, 'title': 'Python Tutorial', 'snippet': '...python...'},
            {'id': 2, 'title': 'Python Guide', 'snippet': '...python...'}
        ]
        
        results = service.search_notes(
            query="python",
            user_id="test_user",
            limit=10
        )
        
        assert len(results) == 2
        assert 'Python' in results[0]['title']
        mock_db.search_notes.assert_called_once_with(
            query="python",
            limit=10,
            offset=0
        )
    
    @pytest.mark.unit
    def test_search_keywords(self, mock_notes_service):
        """Test searching keywords."""
        service = mock_notes_service
        mock_db = service._db_instances["test_user"]
        
        mock_db.search_keywords.return_value = [
            {'id': 1, 'keyword': 'python'},
            {'id': 2, 'keyword': 'python-testing'}
        ]
        
        results = service.search_keywords(
            query="python",
            user_id="test_user"
        )
        
        assert len(results) == 2
        assert 'python' in results[0]['keyword']

# ========================================================================
# Error Handling Tests
# ========================================================================

class TestErrorHandling:
    """Test error handling in the service."""
    
    @pytest.mark.unit
    def test_handle_input_error(self, mock_notes_service):
        """Test handling of input errors."""
        service = mock_notes_service
        mock_db = service._db_instances["test_user"]
        
        mock_db.create_note.side_effect = InputError("Invalid title")
        
        with pytest.raises(InputError) as exc_info:
            service.create_note(
                title="",  # Invalid
                content="Content",
                user_id="test_user"
            )
        
        assert "Invalid title" in str(exc_info.value)
    
    @pytest.mark.unit
    def test_handle_conflict_error(self, mock_notes_service):
        """Test handling of version conflicts."""
        service = mock_notes_service
        mock_db = service._db_instances["test_user"]
        
        mock_db.update_note.side_effect = ConflictError("Version mismatch")
        
        with pytest.raises(ConflictError) as exc_info:
            service.update_note(
                note_id=1,
                user_id="test_user",
                title="Updated",
                expected_version=1
            )
        
        assert "Version mismatch" in str(exc_info.value)
    
    @pytest.mark.unit
    def test_handle_not_found(self, mock_notes_service):
        """Test handling of not found errors."""
        service = mock_notes_service
        mock_db = service._db_instances["test_user"]
        
        mock_db.get_note.return_value = None
        
        result = service.get_note(note_id=999, user_id="test_user")
        
        assert result is None
    
    @pytest.mark.unit
    def test_handle_database_locked(self, mock_notes_service):
        """Test handling of database lock errors."""
        service = mock_notes_service
        mock_db = service._db_instances["test_user"]
        
        mock_db.create_note.side_effect = sqlite3.OperationalError("database is locked")
        
        with pytest.raises(sqlite3.OperationalError) as exc_info:
            service.create_note(
                title="Test",
                content="Content",
                user_id="test_user"
            )
        
        assert "database is locked" in str(exc_info.value)

# ========================================================================
# User Isolation Tests
# ========================================================================

class TestUserIsolation:
    """Test user isolation in the service."""
    
    @pytest.mark.unit
    def test_separate_user_databases(self, mock_notes_service):
        """Test that different users get different database instances."""
        service = mock_notes_service
        
        # Mock different databases for different users
        mock_db1 = MagicMock()
        mock_db2 = MagicMock()
        
        with patch('tldw_Server_API.app.core.Notes.Notes_Library.CharactersRAGDB') as mock_class:
            mock_class.side_effect = [mock_db1, mock_db2]
            
            db1 = service._get_db("user1")
            db2 = service._get_db("user2")
            
            assert db1 is not db2
    
    @pytest.mark.unit
    def test_user_cannot_access_other_notes(self, mock_notes_service):
        """Test that users cannot access other users' notes."""
        service = mock_notes_service
        
        # Create note as user1
        mock_db_user1 = MagicMock()
        service._db_instances["user1"] = mock_db_user1
        
        # Try to get note as user2
        mock_db_user2 = MagicMock()
        mock_db_user2.get_note.return_value = None
        service._db_instances["user2"] = mock_db_user2
        
        result = service.get_note(note_id=1, user_id="user2")
        
        assert result is None
        mock_db_user2.get_note.assert_called_once_with(1)

# ========================================================================
# Connection Management Tests
# ========================================================================

class TestConnectionManagement:
    """Test database connection management."""
    
    @pytest.mark.unit
    def test_close_user_connection(self, mock_notes_service):
        """Test closing a user's database connection."""
        service = mock_notes_service
        mock_db = MagicMock()
        service._db_instances["test_user"] = mock_db
        
        service.close_user_connection("test_user")
        
        mock_db.close.assert_called_once()
        assert "test_user" not in service._db_instances
    
    @pytest.mark.unit
    def test_close_all_connections(self, mock_notes_service):
        """Test closing all database connections."""
        service = mock_notes_service
        
        # Add multiple mock databases
        mock_db1 = MagicMock()
        mock_db2 = MagicMock()
        service._db_instances["user1"] = mock_db1
        service._db_instances["user2"] = mock_db2
        
        service.close_all_user_connections()
        
        mock_db1.close.assert_called_once()
        mock_db2.close.assert_called_once()
        assert len(service._db_instances) == 0
    
    @pytest.mark.unit
    def test_connection_cleanup_on_error(self, mock_notes_service):
        """Test connection cleanup when errors occur."""
        service = mock_notes_service
        mock_db = MagicMock()
        mock_db.close.side_effect = Exception("Close failed")
        service._db_instances["test_user"] = mock_db
        
        # Should not raise even if close fails
        service.close_user_connection("test_user")
        
        assert "test_user" not in service._db_instances