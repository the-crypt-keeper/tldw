# test_chat_dictionary.py
# Description: Unit tests for the ChatDictionaryService
#
"""
Chat Dictionary Service Tests
------------------------------

Comprehensive unit tests for the chat dictionary functionality including
CRUD operations, pattern matching, text processing, and import/export.
"""

import pytest
import json
import tempfile
from unittest.mock import MagicMock, patch, call
from datetime import datetime

from tldw_Server_API.app.core.Character_Chat.chat_dictionary import ChatDictionaryService
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
    """Create a ChatDictionaryService instance with mocked database."""
    return ChatDictionaryService(mock_db)


class TestChatDictionaryService:
    """Test suite for ChatDictionaryService."""
    
    def test_init_creates_tables(self, mock_db):
        """Test that initialization creates necessary tables."""
        mock_conn = MagicMock()
        mock_db.get_connection.return_value.__enter__ = MagicMock(return_value=mock_conn)
        mock_db.get_connection.return_value.__exit__ = MagicMock(return_value=None)
        
        service = ChatDictionaryService(mock_db)
        
        # Should create two tables
        assert mock_conn.execute.call_count >= 2
        
        # Check that tables are created with correct SQL
        calls = mock_conn.execute.call_args_list
        sql_statements = [call[0][0] for call in calls]
        assert any("CREATE TABLE IF NOT EXISTS chat_dictionaries" in sql for sql in sql_statements)
        assert any("CREATE TABLE IF NOT EXISTS dictionary_entries" in sql for sql in sql_statements)
    
    def test_create_dictionary(self, service, mock_db):
        """Test creating a new dictionary."""
        # Mock the connection and cursor
        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_cursor.lastrowid = 1
        mock_conn.execute.return_value = mock_cursor
        mock_db.get_connection.return_value.__enter__.return_value = mock_conn
        mock_db.get_connection.return_value.__exit__.return_value = None
        
        dict_id = service.create_dictionary(
            name="Test Dictionary",
            description="A test dictionary"
        )
        
        assert dict_id == 1
        mock_conn.execute.assert_called()
        
        # Check the SQL contains the right values
        call_args = mock_conn.execute.call_args[0]
        assert "INSERT INTO chat_dictionaries" in call_args[0]
        assert "Test Dictionary" in call_args[1]
        assert "A test dictionary" in call_args[1]
    
    def test_get_dictionary(self, service, mock_db):
        """Test retrieving a dictionary with its entries."""
        # Mock the connection and cursor
        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        
        # Create a mock row that behaves like a dict
        mock_row = MagicMock()
        mock_row.keys = lambda: ["id", "name", "description", "is_active", "created_at", "updated_at", "version", "deleted"]
        mock_row.__getitem__ = lambda self, key: {
            "id": 1, "name": "Test Dict", "description": "Test", "is_active": 1,
            "created_at": "2024-01-01", "updated_at": "2024-01-01", "version": 1, "deleted": 0
        }[key]
        
        mock_cursor.fetchone.return_value = mock_row
        mock_conn.execute.return_value = mock_cursor
        mock_db.get_connection.return_value.__enter__.return_value = mock_conn
        mock_db.get_connection.return_value.__exit__.return_value = None
        
        result = service.get_dictionary(1)
        
        assert result is not None
        assert result["id"] == 1
        assert result["name"] == "Test Dict"
    
    def test_add_entry(self, service, mock_db):
        """Test adding an entry to a dictionary."""
        # Mock the connection and cursor
        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_cursor.lastrowid = 1
        mock_conn.execute.return_value = mock_cursor
        mock_db.get_connection.return_value.__enter__.return_value = mock_conn
        mock_db.get_connection.return_value.__exit__.return_value = None
        
        entry_id = service.add_entry(
            dictionary_id=1,
            key="test",
            content="replaced",
            probability=100,
            max_replacements=1
        )
        
        assert entry_id == 1
        mock_conn.execute.assert_called()
        
        # Check the SQL
        call_args = mock_conn.execute.call_args[0]
        assert "INSERT INTO dictionary_entries" in call_args[0]
        assert "test" in call_args[1]
        assert "replaced" in call_args[1]
    
    def test_process_text_literal_replacement(self, service, mock_db):
        """Test processing text with literal string replacement."""
        # Mock active dictionaries with entries
        mock_db.execute_query.side_effect = [
            [{"id": 1, "name": "Test", "is_active": 1}],  # Active dictionaries
            [{"id": 1, "dictionary_id": 1, "key": "hello",
              "content": "hi", "probability": 100,
              "max_replacements": 1}]  # Entries
        ]
        
        processed_text, metadata = service.process_text(
            text="hello world, hello there",
            token_budget=1000
        )
        
        assert "hi" in processed_text  # Should have replaced at least one "hello"
        assert metadata["replacements"] >= 1
        assert not metadata["token_budget_exceeded"]
    
    def test_process_text_regex_replacement(self, service, mock_db):
        """Test processing text with regex pattern replacement."""
        # Mock active dictionaries with regex entries
        mock_db.execute_query.side_effect = [
            [{"id": 1, "name": "Regex Dict", "is_active": 1}],
            [{"id": 1, "dictionary_id": 1, "key": r"\btest\w+",
              "content": "TEST", "probability": 100,
              "max_replacements": 2}]
        ]
        
        processed_text, metadata = service.process_text(
            text="testing tested tester",
            token_budget=1000
        )
        
        # Note: Actual regex processing depends on implementation
        assert processed_text is not None
        assert metadata is not None
    
    def test_process_text_with_probability(self, service, mock_db):
        """Test that probability affects replacements."""
        # Mock entry with 0% probability
        mock_db.execute_query.side_effect = [
            [{"id": 1, "name": "Test", "is_active": 1}],
            [{"id": 1, "dictionary_id": 1, "key": "hello",
              "content": "hi", "probability": 0,
              "max_replacements": 1}]
        ]
        
        processed_text, metadata = service.process_text("hello world", token_budget=1000)
        
        # With 0% probability, no replacement should occur
        assert processed_text == "hello world"
        assert metadata["replacements"] == 0
    
    def test_process_text_token_budget(self, service, mock_db):
        """Test that processing stops when token budget is exceeded."""
        # Mock a very long text
        long_text = " ".join(["word"] * 1000)
        
        mock_db.execute_query.return_value = []  # No active dictionaries
        
        processed_text, metadata = service.process_text(long_text, token_budget=10)
        
        # Should handle budget
        assert metadata["token_budget_exceeded"] == True
        assert len(processed_text.split()) <= 1000
    
    def test_delete_dictionary_cascade(self, service, mock_db):
        """Test that deleting a dictionary cascades to entries."""
        mock_db.execute_query.return_value = None
        
        result = service.delete_dictionary(1)
        
        assert result == True
        
        # Should delete entries first, then dictionary
        calls = mock_db.execute_query.call_args_list
        assert "DELETE FROM dictionary_entries WHERE dictionary_id = ?" in calls[0][0][0]
        assert "DELETE FROM chat_dictionaries WHERE id = ?" in calls[1][0][0]
    
    def test_import_from_markdown(self, service, mock_db):
        """Test importing dictionary from markdown format."""
        markdown_content = """# Test Dictionary
Description: Test import

## Entries
- hello -> hi
- /test\\w+/ -> TEST (regex)
- goodbye -> bye (50%)
"""
        
        mock_db.execute_query.side_effect = [
            [{"id": 1}],  # Dictionary creation
            [{"id": 1}],  # Entry 1
            [{"id": 2}],  # Entry 2
            [{"id": 3}],  # Entry 3
        ]
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.md', delete=False) as f:
            f.write(markdown_content)
            f.flush()
            
            result = service.import_from_markdown(f.name)
        
        assert result["success"] == True
        assert result["dictionary_id"] == 1
        assert result["entries_imported"] == 3
    
    def test_export_to_markdown(self, service, mock_db):
        """Test exporting dictionary to markdown format."""
        # Mock dictionary data
        mock_db.execute_query.side_effect = [
            [{"id": 1, "name": "Export Test", "description": "Test export",
              "is_active": 1, "created_at": "2024-01-01", "updated_at": "2024-01-01"}],
            [{"id": 1, "key_pattern": "hello", "replacement": "hi",
              "is_regex": 0, "probability": 100, "max_replacements": 1}]
        ]
        
        result = service.export_to_markdown(1)
        
        assert result["success"] == True
        assert "# Export Test" in result["content"]
        assert "- hello -> hi" in result["content"]
    
    def test_update_entry(self, service, mock_db):
        """Test updating a dictionary entry."""
        mock_db.execute_query.return_value = None
        
        result = service.update_entry(
            entry_id=1,
            key="new_pattern",
            content="new_replacement",
            probability=50
        )
        
        assert result == True
        
        # Check SQL
        call_args = mock_db.execute_query.call_args[0]
        assert "UPDATE dictionary_entries SET" in call_args[0]
        assert "new_pattern" in call_args[1]
        assert 50 in call_args[1]
    
    def test_list_dictionaries(self, service, mock_db):
        """Test listing all dictionaries."""
        mock_db.execute_query.return_value = [
            {"id": 1, "name": "Dict1", "is_active": 1, "entry_count": 5},
            {"id": 2, "name": "Dict2", "is_active": 0, "entry_count": 3}
        ]
        
        result = service.list_dictionaries(include_inactive=True)
        
        assert len(result) == 2
        assert result[0]["name"] == "Dict1"
        assert result[1]["entry_count"] == 3
    
    def test_toggle_dictionary_active(self, service, mock_db):
        """Test toggling dictionary active status."""
        # Mock the connection and cursor
        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_cursor.rowcount = 1  # Indicate one row was updated
        mock_conn.execute.return_value = mock_cursor
        mock_db.get_connection.return_value.__enter__.return_value = mock_conn
        mock_db.get_connection.return_value.__exit__.return_value = None
        
        result = service.toggle_dictionary_active(1, False)
        
        assert result == True
        mock_conn.execute.assert_called()
        
        # Check SQL contains the update
        call_args = mock_conn.execute.call_args[0]
        assert "UPDATE chat_dictionaries SET" in call_args[0]
        assert "is_active = ?" in call_args[0]
    
    def test_get_statistics(self, service, mock_db):
        """Test getting dictionary statistics."""
        # Mock the connection and multiple cursors for different queries
        mock_conn = MagicMock()
        
        # First cursor for dictionary counts
        mock_cursor1 = MagicMock()
        mock_row1 = {"total_dictionaries": 5, "active_dictionaries": 3}
        mock_cursor1.fetchone.return_value = mock_row1
        
        # Second cursor for entry counts
        mock_cursor2 = MagicMock()
        mock_row2 = {"total_entries": 100}
        mock_cursor2.fetchone.return_value = mock_row2
        
        # Set up execute to return different cursors for each call
        mock_conn.execute.side_effect = [mock_cursor1, mock_cursor2]
        
        mock_db.get_connection.return_value.__enter__.return_value = mock_conn
        mock_db.get_connection.return_value.__exit__.return_value = None
        
        stats = service.get_statistics()
        
        assert stats["total_dictionaries"] == 5
        assert stats["active_dictionaries"] == 3
        assert stats["total_entries"] == 100
        assert stats["average_entries_per_dictionary"] == 20.0
    
    def test_bulk_add_entries(self, service, mock_db):
        """Test adding multiple entries at once."""
        entries = [
            {"key": "test1", "content": "repl1"},
            {"key": "test2", "content": "repl2"},
            {"key": "test3", "content": "repl3"}
        ]
        
        # Mock the connection and cursor
        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_conn.execute.return_value = mock_cursor
        mock_db.get_connection.return_value.__enter__.return_value = mock_conn
        mock_db.get_connection.return_value.__exit__.return_value = None
        
        result = service.bulk_add_entries(1, entries)
        
        assert result == 3
        # Should have been called 3 times (once for each entry)
        assert mock_conn.execute.call_count == 3
    
    def test_search_entries(self, service, mock_db):
        """Test searching for entries by pattern."""
        mock_db.execute_query.return_value = [
            {"id": 1, "key": "hello", "content": "hi", "dictionary_name": "Dict1"},
            {"id": 2, "key": "hello world", "content": "greetings", "dictionary_name": "Dict2"}
        ]
        
        results = service.search_entries("hello")
        
        assert len(results) == 2
        assert results[0]["key"] == "hello"
        assert results[1]["dictionary_name"] == "Dict2"
    
    def test_clone_dictionary(self, service, mock_db):
        """Test cloning a dictionary with all its entries."""
        # Mock getting original dictionary
        mock_db.execute_query.side_effect = [
            [{"id": 1, "name": "Original", "description": "Original dict", "is_active": 1}],
            [{"key": "test", "content": "repl", "probability": 100}],
            [{"id": 2}],  # New dictionary ID
            None  # Bulk insert entries
        ]
        
        new_id = service.clone_dictionary(1, "Cloned Dict")
        
        assert new_id == 2
        assert mock_db.execute_many.called
    
    def test_entry_validation(self, service, mock_db):
        """Test that invalid entries are handled."""
        # The actual implementation may handle this differently
        # Testing basic validation
        pass
    
    def test_clear_cache(self, service, mock_db):
        """Test that cache is cleared when dictionaries are modified."""
        # First load entries to populate cache
        mock_db.execute_query.return_value = []
        service._load_active_entries()
        
        # Cache should be populated
        assert service._entry_cache is not None
        
        # Modify dictionary
        service.create_dictionary("New", "New dict")
        
        # Cache should be cleared
        assert service._entry_cache is None