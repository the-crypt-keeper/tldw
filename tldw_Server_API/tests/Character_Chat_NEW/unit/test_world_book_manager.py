"""
Unit tests for WorldBookService.

Tests the world book functionality for context management.
"""

import pytest
from unittest.mock import Mock, MagicMock, patch
import json

from tldw_Server_API.app.core.Character_Chat.world_book_manager import WorldBookService

# ========================================================================
# World Book Management Tests
# ========================================================================

class TestWorldBookManagement:
    """Test world book CRUD operations."""
    
    @pytest.mark.unit
    def test_create_world_book(self, world_book_service, sample_world_book):
        """Test creating a world book."""
        service = world_book_service
        
        wb_id = service.create_world_book(
            name=sample_world_book['name'],
            description=sample_world_book['description']
        )
        
        assert wb_id is not None
        assert wb_id > 0
    
    @pytest.mark.unit
    def test_get_world_book(self, world_book_service, sample_world_book):
        """Test getting a world book."""
        service = world_book_service
        
        wb_id = service.create_world_book(
            name=sample_world_book['name'],
            description=sample_world_book['description']
        )
        
        world_book = service.get_world_book(wb_id)
        
        assert world_book is not None
        assert world_book['name'] == sample_world_book['name']
        assert world_book['description'] == sample_world_book['description']
    
    @pytest.mark.unit
    def test_list_world_books(self, world_book_service):
        """Test listing all world books."""
        service = world_book_service
        
        # Create multiple world books
        for i in range(3):
            service.create_world_book(
                name=f"World Book {i}",
                description=f"Description {i}"
            )
        
        world_books = service.list_world_books()
        
        assert len(world_books) >= 3
        assert all('name' in wb for wb in world_books)
    
    @pytest.mark.unit
    def test_update_world_book(self, world_book_service):
        """Test updating a world book."""
        service = world_book_service
        
        wb_id = service.create_world_book(
            name="Original",
            description="Original description"
        )
        
        service.update_world_book(
            world_book_id=wb_id,
            name="Updated",
            description="Updated description"
        )
        
        updated = service.get_world_book(wb_id)
        assert updated['name'] == "Updated"
        assert updated['description'] == "Updated description"
    
    @pytest.mark.unit
    def test_delete_world_book(self, world_book_service):
        """Test deleting a world book."""
        service = world_book_service
        
        wb_id = service.create_world_book(name="To Delete")
        
        service.delete_world_book(wb_id)
        
        # Should not find deleted world book
        deleted = service.get_world_book(wb_id)
        assert deleted is None or deleted.get('is_deleted', False)
    
    @pytest.mark.unit
    def test_delete_cascade(self, world_book_service, sample_world_book):
        """Test cascade deletion of entries."""
        service = world_book_service
        
        wb_id = service.create_world_book(name="Cascade Test")
        
        # Add entries
        for entry in sample_world_book['entries']:
            service.add_entry(wb_id, **entry)
        
        # Delete world book
        service.delete_world_book(wb_id, cascade=True)
        
        # Entries should also be deleted
        entries = service.get_entries(wb_id)
        assert len(entries) == 0

# ========================================================================
# Entry Management Tests
# ========================================================================

class TestEntryManagement:
    """Test world book entry operations."""
    
    @pytest.mark.unit
    def test_add_entry(self, world_book_service):
        """Test adding an entry."""
        service = world_book_service
        
        wb_id = service.create_world_book(name="Entry Test")
        
        entry_id = service.add_entry(
            world_book_id=wb_id,
            keywords=['dragon', 'dragons'],
            content='Dragons are magical creatures.',
            priority=100
        )
        
        assert entry_id is not None
        entries = service.get_entries(wb_id)
        assert len(entries) == 1
        assert 'dragon' in entries[0]['keywords']
    
    @pytest.mark.unit
    def test_add_recursive_entry(self, world_book_service):
        """Test adding recursive scanning entry."""
        service = world_book_service
        
        wb_id = service.create_world_book(name="Recursive Test")
        
        entry_id = service.add_entry(
            world_book_id=wb_id,
            keywords=['hero'],
            content='The hero wields a sword.',
            priority=100,
            recursive_scanning=True
        )
        
        entries = service.get_entries(wb_id)
        assert entries[0]['recursive_scanning'] is True
    
    @pytest.mark.unit
    def test_update_entry(self, world_book_service):
        """Test updating an entry."""
        service = world_book_service
        
        wb_id = service.create_world_book(name="Update Test")
        entry_id = service.add_entry(
            world_book_id=wb_id,
            keywords=['old'],
            content='Old content'
        )
        
        service.update_entry(
            entry_id=entry_id,
            keywords=['new', 'updated'],
            content='New content',
            priority=90
        )
        
        entries = service.get_entries(wb_id)
        assert 'new' in entries[0]['keywords']
        assert entries[0]['content'] == 'New content'
        assert entries[0]['priority'] == 90
    
    @pytest.mark.unit
    def test_delete_entry(self, world_book_service):
        """Test deleting an entry."""
        service = world_book_service
        
        wb_id = service.create_world_book(name="Delete Test")
        entry_id = service.add_entry(
            world_book_id=wb_id,
            keywords=['test'],
            content='Test content'
        )
        
        service.delete_entry(entry_id)
        
        entries = service.get_entries(wb_id)
        assert len(entries) == 0
    
    @pytest.mark.unit
    def test_toggle_entry_enabled(self, world_book_service):
        """Test toggling entry enabled status."""
        service = world_book_service
        
        wb_id = service.create_world_book(name="Toggle Test")
        entry_id = service.add_entry(
            world_book_id=wb_id,
            keywords=['test'],
            content='Test'
        )
        
        # Initially enabled
        entries = service.get_entries(wb_id)
        initial_status = entries[0].get('enabled', True)
        
        # Toggle
        service.toggle_entry_enabled(entry_id)
        
        # Check toggled
        entries = service.get_entries(wb_id)
        assert entries[0].get('enabled', True) != initial_status
    
    @pytest.mark.unit
    def test_bulk_operations(self, world_book_service, sample_world_book):
        """Test bulk entry operations."""
        service = world_book_service
        
        wb_id = service.create_world_book(name="Bulk Test")
        
        # Bulk add
        results = service.bulk_add_entries(
            world_book_id=wb_id,
            entries=sample_world_book['entries']
        )
        
        assert results['added'] == len(sample_world_book['entries'])
        
        entries = service.get_entries(wb_id)
        assert len(entries) == len(sample_world_book['entries'])

# ========================================================================
# Character Association Tests
# ========================================================================

class TestCharacterAssociation:
    """Test world book to character associations."""
    
    @pytest.mark.unit
    def test_attach_to_character(self, world_book_service):
        """Test attaching world book to character."""
        service = world_book_service
        
        wb_id = service.create_world_book(name="Character World")
        character_id = 1  # Mock character ID
        
        result = service.attach_to_character(wb_id, character_id)
        
        assert result['success'] is True
        
        # Check association
        character_books = service.get_character_world_books(character_id)
        assert any(wb['id'] == wb_id for wb in character_books)
    
    @pytest.mark.unit
    def test_detach_from_character(self, world_book_service):
        """Test detaching world book from character."""
        service = world_book_service
        
        wb_id = service.create_world_book(name="Detach Test")
        character_id = 1
        
        # Attach first
        service.attach_to_character(wb_id, character_id)
        
        # Then detach
        result = service.detach_from_character(wb_id, character_id)
        
        assert result['success'] is True
        
        # Check no longer associated
        character_books = service.get_character_world_books(character_id)
        assert not any(wb['id'] == wb_id for wb in character_books)
    
    @pytest.mark.unit
    def test_get_character_world_books(self, world_book_service):
        """Test getting all world books for a character."""
        service = world_book_service
        
        character_id = 1
        
        # Create and attach multiple world books
        wb_ids = []
        for i in range(3):
            wb_id = service.create_world_book(name=f"Character Book {i}")
            service.attach_to_character(wb_id, character_id)
            wb_ids.append(wb_id)
        
        character_books = service.get_character_world_books(character_id)
        
        assert len(character_books) >= 3
        for wb_id in wb_ids:
            assert any(wb['id'] == wb_id for wb in character_books)

# ========================================================================
# Context Processing Tests
# ========================================================================

class TestContextProcessing:
    """Test context processing with world book entries."""
    
    @pytest.mark.unit
    def test_process_context_keyword_matching(self, world_book_service, sample_world_book):
        """Test keyword matching in context."""
        service = world_book_service
        
        wb_id = service.create_world_book(name="Context Test")
        for entry in sample_world_book['entries']:
            service.add_entry(wb_id, **entry)
        
        context = "Tell me about dragons and magic in the kingdom."
        
        activated = service.process_context(context, wb_id)
        
        # Should activate all three entries based on keywords
        assert len(activated) == 3
        assert any('Dragons are ancient' in e['content'] for e in activated)
        assert any('Magic flows' in e['content'] for e in activated)
        assert any('kingdom spans' in e['content'] for e in activated)
    
    @pytest.mark.unit
    def test_process_context_priority_ordering(self, world_book_service):
        """Test entries ordered by priority."""
        service = world_book_service
        
        wb_id = service.create_world_book(name="Priority Test")
        
        # Add entries with different priorities
        service.add_entry(wb_id, ['test'], 'Low priority', priority=50)
        service.add_entry(wb_id, ['test'], 'High priority', priority=100)
        service.add_entry(wb_id, ['test'], 'Medium priority', priority=75)
        
        context = "This is a test."
        activated = service.process_context(context, wb_id)
        
        # Should be ordered by priority (highest first)
        assert activated[0]['content'] == 'High priority'
        assert activated[1]['content'] == 'Medium priority'
        assert activated[2]['content'] == 'Low priority'
    
    @pytest.mark.unit
    def test_process_context_disabled_entries(self, world_book_service):
        """Test disabled entries are not activated."""
        service = world_book_service
        
        wb_id = service.create_world_book(name="Disabled Test")
        
        entry1 = service.add_entry(wb_id, ['active'], 'Active entry')
        entry2 = service.add_entry(wb_id, ['disabled'], 'Disabled entry')
        
        # Disable second entry
        service.toggle_entry_enabled(entry2)
        
        context = "Test active and disabled keywords."
        activated = service.process_context(context, wb_id)
        
        # Only active entry should be included
        assert len(activated) == 1
        assert activated[0]['content'] == 'Active entry'
    
    @pytest.mark.unit
    def test_recursive_scanning(self, world_book_service, complex_world_book):
        """Test recursive scanning of entries."""
        service = world_book_service
        
        wb_id = service.create_world_book(name="Recursive Test")
        for entry in complex_world_book['entries']:
            service.add_entry(wb_id, **entry)
        
        # Initial context only mentions 'hero'
        context = "The hero arrives."
        
        activated = service.process_context(
            context,
            wb_id,
            recursive_depth=2
        )
        
        # Should recursively activate: hero -> sword -> smiths
        assert len(activated) >= 2
        assert any('hero' in str(e['keywords']) for e in activated)
        assert any('sword' in e['content'] for e in activated)
        # With depth=2, might also get smiths entry
    
    @pytest.mark.unit
    def test_token_budget_limit(self, world_book_service, mock_tokenizer):
        """Test respecting token budget."""
        service = world_book_service
        
        wb_id = service.create_world_book(name="Budget Test")
        
        # Add many entries
        for i in range(10):
            service.add_entry(
                wb_id,
                [f'keyword{i}'],
                f'This is a long content for entry {i} ' * 10,
                priority=100 - i
            )
        
        context = ' '.join([f'keyword{i}' for i in range(10)])
        
        with patch.object(service, 'count_tokens', side_effect=lambda x: len(x.split())):
            activated = service.process_context(
                context,
                wb_id,
                max_tokens=100
            )
        
        # Should limit entries to stay within budget
        total_tokens = sum(len(e['content'].split()) for e in activated)
        assert total_tokens <= 120  # Allow small overhead

# ========================================================================
# Search and Filter Tests
# ========================================================================

class TestSearchAndFilter:
    """Test search and filter functionality."""
    
    @pytest.mark.unit
    def test_search_entries(self, world_book_service):
        """Test searching world book entries."""
        service = world_book_service
        
        wb_id = service.create_world_book(name="Search Test")
        service.add_entry(wb_id, ['dragon'], 'Dragon content')
        service.add_entry(wb_id, ['wizard'], 'Wizard content')
        service.add_entry(wb_id, ['dragon', 'wizard'], 'Both content')
        
        results = service.search_entries(wb_id, query="dragon")
        
        assert len(results) == 2
        assert all('dragon' in str(r['keywords']) for r in results)
    
    @pytest.mark.unit
    def test_filter_by_priority(self, world_book_service):
        """Test filtering entries by priority."""
        service = world_book_service
        
        wb_id = service.create_world_book(name="Priority Filter")
        service.add_entry(wb_id, ['low'], 'Low', priority=25)
        service.add_entry(wb_id, ['medium'], 'Medium', priority=50)
        service.add_entry(wb_id, ['high'], 'High', priority=100)
        
        high_priority = service.filter_entries(wb_id, min_priority=75)
        
        assert len(high_priority) == 1
        assert high_priority[0]['content'] == 'High'
    
    @pytest.mark.unit
    def test_filter_recursive_entries(self, world_book_service):
        """Test filtering recursive scanning entries."""
        service = world_book_service
        
        wb_id = service.create_world_book(name="Recursive Filter")
        service.add_entry(wb_id, ['normal'], 'Normal', recursive_scanning=False)
        service.add_entry(wb_id, ['recursive'], 'Recursive', recursive_scanning=True)
        
        recursive = service.filter_entries(wb_id, recursive_only=True)
        
        assert len(recursive) == 1
        assert recursive[0]['content'] == 'Recursive'

# ========================================================================
# Import/Export Tests
# ========================================================================

class TestImportExport:
    """Test import/export functionality."""
    
    @pytest.mark.unit
    def test_export_world_book(self, world_book_service, sample_world_book):
        """Test exporting world book."""
        service = world_book_service
        
        wb_id = service.create_world_book(
            name=sample_world_book['name'],
            description=sample_world_book['description']
        )
        for entry in sample_world_book['entries']:
            service.add_entry(wb_id, **entry)
        
        exported = service.export_world_book(wb_id)
        
        assert exported['name'] == sample_world_book['name']
        assert 'entries' in exported
        assert len(exported['entries']) == len(sample_world_book['entries'])
    
    @pytest.mark.unit
    def test_import_world_book(self, world_book_service, sample_world_book):
        """Test importing world book."""
        service = world_book_service
        
        wb_id = service.import_world_book(sample_world_book)
        
        assert wb_id is not None
        
        world_book = service.get_world_book(wb_id)
        assert world_book['name'] == sample_world_book['name']
        
        entries = service.get_entries(wb_id)
        assert len(entries) == len(sample_world_book['entries'])
    
    @pytest.mark.unit
    def test_export_to_lorebook_format(self, world_book_service, sample_world_book):
        """Test exporting to lorebook format."""
        service = world_book_service
        
        wb_id = service.create_world_book(name=sample_world_book['name'])
        for entry in sample_world_book['entries']:
            service.add_entry(wb_id, **entry)
        
        lorebook = service.export_to_lorebook_format(wb_id)
        
        assert 'entries' in lorebook
        assert all('key' in e for e in lorebook['entries'])
        assert all('content' in e for e in lorebook['entries'])

# ========================================================================
# Statistics Tests
# ========================================================================

class TestStatistics:
    """Test statistics and analytics."""
    
    @pytest.mark.unit
    def test_get_statistics(self, world_book_service, sample_world_book):
        """Test getting world book statistics."""
        service = world_book_service
        
        wb_id = service.create_world_book(name="Stats Test")
        for entry in sample_world_book['entries']:
            service.add_entry(wb_id, **entry)
        
        stats = service.get_statistics(wb_id)
        
        assert stats['total_entries'] == len(sample_world_book['entries'])
        assert stats['total_keywords'] > 0
        assert 'avg_priority' in stats
        assert 'recursive_entries' in stats
    
    @pytest.mark.unit
    def test_get_activation_statistics(self, world_book_service):
        """Test getting activation statistics."""
        service = world_book_service
        
        wb_id = service.create_world_book(name="Activation Stats")
        service.add_entry(wb_id, ['test'], 'Test content')
        
        # Process context multiple times
        for _ in range(5):
            service.process_context("test context", wb_id)
        
        stats = service.get_activation_statistics(wb_id)
        
        assert stats.get('total_activations', 0) >= 5

# ========================================================================
# Cloning Tests
# ========================================================================

class TestCloning:
    """Test world book cloning."""
    
    @pytest.mark.unit
    def test_clone_world_book(self, world_book_service, sample_world_book):
        """Test cloning a world book."""
        service = world_book_service
        
        # Create original
        original_id = service.create_world_book(name="Original")
        for entry in sample_world_book['entries']:
            service.add_entry(original_id, **entry)
        
        # Clone
        cloned_id = service.clone_world_book(original_id, new_name="Clone")
        
        assert cloned_id != original_id
        
        # Check clone has same entries
        original_entries = service.get_entries(original_id)
        cloned_entries = service.get_entries(cloned_id)
        
        assert len(cloned_entries) == len(original_entries)
        
        # Check entries match (excluding IDs)
        for orig, clone in zip(original_entries, cloned_entries):
            assert orig['keywords'] == clone['keywords']
            assert orig['content'] == clone['content']
            assert orig['priority'] == clone['priority']

# ========================================================================
# Keyword Processing Tests
# ========================================================================

class TestKeywordProcessing:
    """Test keyword normalization and processing."""
    
    @pytest.mark.unit
    def test_keyword_normalization(self, world_book_service):
        """Test keyword normalization."""
        service = world_book_service
        
        # Test various normalizations
        assert service.normalize_keyword("  Test  ") == "test"
        assert service.normalize_keyword("TEST") == "test"
        assert service.normalize_keyword("test-keyword") == "test-keyword"
        assert service.normalize_keyword("test_keyword") == "test_keyword"
    
    @pytest.mark.unit
    def test_keyword_matching_case_insensitive(self, world_book_service):
        """Test case-insensitive keyword matching."""
        service = world_book_service
        
        wb_id = service.create_world_book(name="Case Test")
        service.add_entry(wb_id, ['Dragon'], 'Dragon entry')
        
        # Should match regardless of case
        context_lower = "dragon appears"
        context_upper = "DRAGON appears"
        context_mixed = "DrAgOn appears"
        
        for context in [context_lower, context_upper, context_mixed]:
            activated = service.process_context(context, wb_id)
            assert len(activated) == 1
            assert activated[0]['content'] == 'Dragon entry'

# ========================================================================
# Error Handling Tests
# ========================================================================

class TestErrorHandling:
    """Test error handling."""
    
    @pytest.mark.unit
    def test_world_book_not_found(self, world_book_service):
        """Test handling world book not found."""
        service = world_book_service
        
        result = service.get_world_book(999999)
        assert result is None
    
    @pytest.mark.unit
    def test_entry_validation(self, world_book_service):
        """Test entry validation."""
        service = world_book_service
        
        wb_id = service.create_world_book(name="Validation Test")
        
        # Empty keywords should fail
        with pytest.raises(ValueError):
            service.add_entry(wb_id, [], "Content")
        
        # Empty content is allowed but might warn
        entry_id = service.add_entry(wb_id, ['test'], "")
        assert entry_id is not None
    
    @pytest.mark.unit
    def test_invalid_priority(self, world_book_service):
        """Test invalid priority handling."""
        service = world_book_service
        
        wb_id = service.create_world_book(name="Priority Test")
        
        # Negative priority should be clamped to 0
        entry_id = service.add_entry(
            wb_id,
            ['test'],
            'Content',
            priority=-10
        )
        
        entries = service.get_entries(wb_id)
        assert entries[0]['priority'] >= 0
        
        # Priority over 100 should be clamped to 100
        entry_id2 = service.add_entry(
            wb_id,
            ['test2'],
            'Content2',
            priority=200
        )
        
        entries = service.get_entries(wb_id)
        entry = next(e for e in entries if 'test2' in e['keywords'])
        assert entry['priority'] <= 100