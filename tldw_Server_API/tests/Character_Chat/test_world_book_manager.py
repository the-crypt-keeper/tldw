# test_world_book_manager.py
# Description: Unit tests for the WorldBookService
#
"""
World Book Manager Service Tests
---------------------------------

Comprehensive unit tests for the world book functionality including
CRUD operations, keyword matching, character attachments, and context processing.
"""

import pytest
import json
from unittest.mock import MagicMock, patch, call
from datetime import datetime

from tldw_Server_API.app.core.Character_Chat.world_book_manager import WorldBookService, WorldBookEntry
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
    """Create a WorldBookService instance with mocked database."""
    return WorldBookService(mock_db)


class TestWorldBookService:
    """Test suite for WorldBookService."""
    
    def test_init_creates_tables(self, mock_db):
        """Test that initialization creates necessary tables."""
        service = WorldBookService(mock_db)
        
        # Should create three tables
        assert mock_db.execute_query.call_count == 3
        
        # Check that tables are created
        calls = mock_db.execute_query.call_args_list
        assert "CREATE TABLE IF NOT EXISTS world_books" in calls[0][0][0]
        assert "CREATE TABLE IF NOT EXISTS world_book_entries" in calls[1][0][0]
        assert "CREATE TABLE IF NOT EXISTS character_world_books" in calls[2][0][0]
    
    def test_create_world_book(self, service, mock_db):
        """Test creating a new world book."""
        mock_db.execute_query.return_value = [{"id": 1}]
        
        wb_id = service.create_world_book(
            name="Test World",
            description="A test world book",
            scan_depth=3,
            token_budget=500,
            recursive_scanning=True,
            enabled=True
        )
        
        assert wb_id == 1
        mock_db.execute_query.assert_called()
        
        # Check SQL parameters
        call_args = mock_db.execute_query.call_args[0]
        assert "INSERT INTO world_books" in call_args[0]
        assert "Test World" in call_args[1]
        assert 500 in call_args[1]  # token_budget
    
    def test_get_world_book_with_entries(self, service, mock_db):
        """Test retrieving a world book with its entries."""
        # Mock world book and entries data
        mock_db.execute_query.side_effect = [
            [{"id": 1, "name": "Fantasy World", "description": "Fantasy setting",
              "scan_depth": 3, "token_budget": 1000, "recursive_scanning": 1,
              "enabled": 1, "created_at": "2024-01-01", "updated_at": "2024-01-01"}],
            [{"id": 1, "world_book_id": 1, "keywords": "dragon,castle",
              "content": "Dragons live in castles", "priority": 100, "enabled": 1}]
        ]
        
        result = service.get_world_book(1)
        
        assert result["id"] == 1
        assert result["name"] == "Fantasy World"
        assert len(result["entries"]) == 1
        assert result["entries"][0]["keywords"] == "dragon,castle"
    
    def test_add_entry(self, service, mock_db):
        """Test adding an entry to a world book."""
        mock_db.execute_query.return_value = [{"id": 1}]
        
        entry_id = service.add_entry(
            world_book_id=1,
            keywords=["magic", "wizard"],
            content="Wizards use magic",
            priority=100,
            enabled=True
        )
        
        assert entry_id == 1
        
        # Check that keywords are joined
        call_args = mock_db.execute_query.call_args[0]
        assert "INSERT INTO world_book_entries" in call_args[0]
        assert "magic,wizard" in call_args[1]
    
    def test_attach_to_character(self, service, mock_db):
        """Test attaching a world book to a character."""
        # Mock checking for existing attachment (none found)
        mock_db.execute_query.side_effect = [
            [],  # No existing attachment
            None  # Insert successful
        ]
        
        result = service.attach_to_character(
            character_id=1,
            world_book_id=1,
            is_primary=True
        )
        
        assert result == True
        
        # Check insert SQL
        calls = mock_db.execute_query.call_args_list
        assert "INSERT INTO character_world_books" in calls[1][0][0]
    
    def test_detach_from_character(self, service, mock_db):
        """Test detaching a world book from a character."""
        mock_db.execute_query.return_value = None
        
        result = service.detach_from_character(
            character_id=1,
            world_book_id=1
        )
        
        assert result == True
        
        # Check delete SQL
        call_args = mock_db.execute_query.call_args[0]
        assert "DELETE FROM character_world_books" in call_args[0]
        assert call_args[1] == (1, 1)
    
    def test_get_character_world_books(self, service, mock_db):
        """Test getting all world books for a character."""
        mock_db.execute_query.return_value = [
            {"id": 1, "name": "Main World", "is_primary": 1, "enabled": 1},
            {"id": 2, "name": "Secondary World", "is_primary": 0, "enabled": 1}
        ]
        
        result = service.get_character_world_books(1)
        
        assert len(result) == 2
        assert result[0]["name"] == "Main World"
        assert result[0]["is_primary"] == True
        assert result[1]["is_primary"] == False
    
    def test_process_context_keyword_matching(self, service, mock_db):
        """Test context processing with keyword matching."""
        # Mock active world books and entries
        mock_db.execute_query.side_effect = [
            [{"id": 1, "name": "Test World", "token_budget": 500}],  # Active world books
            [{"id": 1, "world_book_id": 1, "keywords": "sword,blade",
              "content": "A legendary sword", "priority": 100, "enabled": 1},
             {"id": 2, "world_book_id": 1, "keywords": "magic",
              "content": "Magic is powerful", "priority": 50, "enabled": 1}]
        ]
        
        result = service.process_context(
            text="The hero found a magical sword",
            character_id=None,
            max_tokens=1000
        )
        
        assert "processed_context" in result
        assert "entries_applied" in result
        assert result["entries_applied"] > 0
        # Should match both "sword" and "magic"
        assert "legendary sword" in result["processed_context"].lower()
        assert "magic is powerful" in result["processed_context"].lower()
    
    def test_process_context_priority_ordering(self, service, mock_db):
        """Test that entries are processed by priority."""
        # Create entries with different priorities
        entries = [
            WorldBookEntry(
                id=1, world_book_id=1, keywords=["test"],
                content="Low priority", priority=10, enabled=True
            ),
            WorldBookEntry(
                id=2, world_book_id=1, keywords=["test"],
                content="High priority", priority=100, enabled=True
            )
        ]
        
        # Mock to return entries
        mock_db.execute_query.side_effect = [
            [{"id": 1, "name": "Test", "token_budget": 500}],
            []  # No entries from DB, we'll set them directly
        ]
        
        # Set entries directly for testing
        service._entry_cache = {1: entries}
        
        result = service.process_context(
            text="This is a test",
            character_id=None,
            max_tokens=1000
        )
        
        # High priority should be processed first
        assert "High priority" in result["processed_context"]
    
    def test_process_context_token_budget(self, service, mock_db):
        """Test that token budget is respected."""
        # Create a very long entry
        long_content = " ".join(["word"] * 1000)
        
        mock_db.execute_query.side_effect = [
            [{"id": 1, "name": "Test", "token_budget": 10}],  # Very small budget
            [{"id": 1, "keywords": "test", "content": long_content,
              "priority": 100, "enabled": 1}]
        ]
        
        result = service.process_context(
            text="This is a test",
            character_id=None,
            max_tokens=10
        )
        
        # Should truncate to fit budget
        assert result["token_budget_exceeded"] == True
        assert len(result["processed_context"].split()) < 1000
    
    def test_recursive_scanning(self, service, mock_db):
        """Test recursive keyword scanning."""
        # First entry triggers second entry
        mock_db.execute_query.side_effect = [
            [{"id": 1, "name": "Test", "token_budget": 500, "recursive_scanning": 1}],
            [{"id": 1, "keywords": "hero", "content": "The hero has a sword",
              "priority": 100, "enabled": 1},
             {"id": 2, "keywords": "sword", "content": "The sword is magical",
              "priority": 50, "enabled": 1}]
        ]
        
        result = service.process_context(
            text="Story about a hero",
            character_id=None,
            max_tokens=1000
        )
        
        # Should match "hero" first, then "sword" from the hero entry
        assert result["entries_applied"] >= 1
    
    def test_import_world_book(self, service, mock_db):
        """Test importing a world book from JSON."""
        world_book_data = {
            "name": "Imported World",
            "description": "Imported from JSON",
            "scan_depth": 5,
            "token_budget": 800,
            "entries": [
                {"keywords": "test1", "content": "Content 1", "priority": 100},
                {"keywords": "test2", "content": "Content 2", "priority": 50}
            ]
        }
        
        mock_db.execute_query.side_effect = [
            [{"id": 1}],  # World book creation
            [{"id": 1}],  # Entry 1
            [{"id": 2}],  # Entry 2
        ]
        
        result = service.import_world_book(world_book_data)
        
        assert result == True
        assert mock_db.execute_query.call_count == 3
    
    def test_export_world_book(self, service, mock_db):
        """Test exporting a world book to JSON."""
        mock_db.execute_query.side_effect = [
            [{"id": 1, "name": "Export Test", "description": "Test export",
              "scan_depth": 3, "token_budget": 500, "recursive_scanning": 0,
              "enabled": 1}],
            [{"keywords": "test", "content": "Test content", "priority": 100, "enabled": 1}]
        ]
        
        result = service.export_world_book(1)
        
        assert result["name"] == "Export Test"
        assert len(result["entries"]) == 1
        assert result["entries"][0]["keywords"] == "test"
    
    def test_delete_world_book_cascade(self, service, mock_db):
        """Test that deleting a world book cascades properly."""
        mock_db.execute_query.return_value = None
        
        result = service.delete_world_book(1)
        
        assert result == True
        
        # Should delete in correct order
        calls = mock_db.execute_query.call_args_list
        assert "DELETE FROM character_world_books" in calls[0][0][0]
        assert "DELETE FROM world_book_entries" in calls[1][0][0]
        assert "DELETE FROM world_books" in calls[2][0][0]
    
    def test_update_entry(self, service, mock_db):
        """Test updating a world book entry."""
        mock_db.execute_query.return_value = None
        
        result = service.update_entry(
            entry_id=1,
            keywords=["new", "keywords"],
            content="New content",
            priority=75
        )
        
        assert result == True
        
        # Check SQL
        call_args = mock_db.execute_query.call_args[0]
        assert "UPDATE world_book_entries SET" in call_args[0]
        assert "new,keywords" in call_args[1]
    
    def test_get_statistics(self, service, mock_db):
        """Test getting world book statistics."""
        mock_db.execute_query.side_effect = [
            [{"total_world_books": 10, "enabled_world_books": 8}],
            [{"total_entries": 150}],
            [{"total_attachments": 25}],
            [{"avg_entries": 15.0}]
        ]
        
        stats = service.get_statistics()
        
        assert stats["total_world_books"] == 10
        assert stats["enabled_world_books"] == 8
        assert stats["total_entries"] == 150
        assert stats["total_character_attachments"] == 25
        assert stats["average_entries_per_world_book"] == 15.0
    
    def test_search_entries(self, service, mock_db):
        """Test searching for entries by keyword or content."""
        mock_db.execute_query.return_value = [
            {"id": 1, "keywords": "dragon,fire", "content": "Dragons breathe fire",
             "world_book_name": "Fantasy"},
            {"id": 2, "keywords": "dragon", "content": "Ancient dragon lore",
             "world_book_name": "Mythology"}
        ]
        
        results = service.search_entries("dragon")
        
        assert len(results) == 2
        assert results[0]["world_book_name"] == "Fantasy"
        assert "fire" in results[0]["keywords"]
    
    def test_bulk_operations(self, service, mock_db):
        """Test bulk enable/disable of entries."""
        mock_db.execute_query.return_value = None
        
        # Bulk disable entries
        result = service.bulk_update_entries(
            world_book_id=1,
            entry_ids=[1, 2, 3],
            enabled=False
        )
        
        assert result == 3
        
        # Check SQL
        call_args = mock_db.execute_query.call_args[0]
        assert "UPDATE world_book_entries SET enabled = ?" in call_args[0]
    
    def test_clone_world_book(self, service, mock_db):
        """Test cloning a world book with all entries."""
        mock_db.execute_query.side_effect = [
            [{"id": 1, "name": "Original", "description": "Original world",
              "scan_depth": 3, "token_budget": 500}],
            [{"keywords": "test", "content": "content", "priority": 100}],
            [{"id": 2}],  # New world book ID
            None  # Bulk insert entries
        ]
        
        new_id = service.clone_world_book(1, "Cloned World")
        
        assert new_id == 2
        assert mock_db.execute_many.called
    
    def test_keyword_normalization(self, service):
        """Test that keywords are normalized properly."""
        entry = WorldBookEntry(
            id=1,
            world_book_id=1,
            keywords=["  Dragon  ", "CASTLE", "  magic  "],
            content="Test",
            priority=100,
            enabled=True
        )
        
        # Keywords should be normalized to lowercase and trimmed
        assert entry.matches_text("the dragon lives in a castle")
        assert entry.matches_text("MAGIC spell")
        assert entry.matches_text("Dragon's lair")