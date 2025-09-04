# world_book_manager.py
# Description: World Book/Lorebook Manager for character chat with multi-user support
# Adapted from single-user to multi-user architecture with per-user database isolation
#
"""
World Book Manager for Multi-User Environment
---------------------------------------------

This module provides management for world books (lorebooks) that can be used
independently of characters, allowing shared lorebooks across conversations.

Key Adaptations from Single-User:
- Per-user database isolation (each user has their own database)
- Stateless service instances (no global state or singletons)
- Request-scoped processing (instantiated per API request)
- No thread-local storage (incompatible with async API)
- Database-backed persistence with user isolation

Features:
- CRUD operations for world books and entries
- Keyword matching and activation
- Token budget management
- Priority-based entry selection
- Scan depth control
- Recursive scanning support
"""

import json
import re
import sqlite3
from typing import List, Dict, Any, Optional, Set
from datetime import datetime
from pathlib import Path

from loguru import logger

# Local imports
from tldw_Server_API.app.core.DB_Management.ChaChaNotes_DB import (
    CharactersRAGDB,
    CharactersRAGDBError,
    InputError,
    ConflictError
)


class WorldBookEntry:
    """
    Individual world book entry with keyword matching capabilities.
    
    This is a stateless data class that handles keyword matching and content.
    No global state or thread-local storage is used.
    """
    
    def __init__(
        self,
        entry_id: Optional[int] = None,
        world_book_id: int = None,
        keywords: List[str] = None,
        content: str = "",
        priority: int = 0,
        enabled: bool = True,
        case_sensitive: bool = False,
        regex_match: bool = False,
        whole_word_match: bool = True,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize a world book entry.
        
        Args:
            entry_id: Database ID
            world_book_id: Parent world book ID
            keywords: List of keywords to match
            content: Content to inject when matched
            priority: Priority for ordering (higher = more important)
            enabled: Whether entry is active
            case_sensitive: Whether keyword matching is case-sensitive
            regex_match: Whether keywords are regex patterns
            whole_word_match: Whether to match whole words only
            metadata: Additional metadata
        """
        self.entry_id = entry_id
        self.world_book_id = world_book_id
        self.keywords = keywords or []
        self.content = content
        self.priority = priority
        self.enabled = enabled
        self.case_sensitive = case_sensitive
        self.regex_match = regex_match
        self.whole_word_match = whole_word_match
        self.metadata = metadata or {}
        
        # Compile patterns for efficient matching
        self._patterns = self._compile_patterns()
    
    def _compile_patterns(self) -> List[re.Pattern]:
        """Compile keyword patterns for matching."""
        patterns = []
        
        for keyword in self.keywords:
            if self.regex_match:
                # Use keyword as regex directly
                try:
                    flags = 0 if self.case_sensitive else re.IGNORECASE
                    pattern = re.compile(keyword, flags)
                    patterns.append(pattern)
                except re.error as e:
                    logger.warning(f"Invalid regex pattern '{keyword}': {e}")
            else:
                # Build pattern for literal matching
                escaped = re.escape(keyword)
                if self.whole_word_match:
                    pattern_str = r'\b' + escaped + r'\b'
                else:
                    pattern_str = escaped
                
                flags = 0 if self.case_sensitive else re.IGNORECASE
                patterns.append(re.compile(pattern_str, flags))
        
        return patterns
    
    def matches(self, text: str) -> bool:
        """
        Check if any keyword matches the given text.
        
        Args:
            text: Text to search for keywords
            
        Returns:
            True if any keyword matches
        """
        if not self.enabled or not self._patterns:
            return False
        
        for pattern in self._patterns:
            if pattern.search(text):
                return True
        
        return False
    
    def get_match_count(self, text: str) -> int:
        """
        Count how many keywords match in the text.
        
        Args:
            text: Text to search
            
        Returns:
            Number of matching keywords
        """
        if not self.enabled or not self._patterns:
            return 0
        
        count = 0
        for pattern in self._patterns:
            if pattern.search(text):
                count += 1
        
        return count
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for database storage."""
        return {
            'id': self.entry_id,
            'world_book_id': self.world_book_id,
            'keywords': json.dumps(self.keywords),
            'content': self.content,
            'priority': self.priority,
            'enabled': self.enabled,
            'case_sensitive': self.case_sensitive,
            'regex_match': self.regex_match,
            'whole_word_match': self.whole_word_match,
            'metadata': json.dumps(self.metadata)
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'WorldBookEntry':
        """Create instance from database dictionary."""
        keywords = data.get('keywords')
        if isinstance(keywords, str):
            keywords = json.loads(keywords)
        
        metadata = data.get('metadata')
        if isinstance(metadata, str):
            metadata = json.loads(metadata)
        
        return cls(
            entry_id=data.get('id'),
            world_book_id=data.get('world_book_id'),
            keywords=keywords,
            content=data.get('content', ''),
            priority=data.get('priority', 0),
            enabled=data.get('enabled', True),
            case_sensitive=data.get('case_sensitive', False),
            regex_match=data.get('regex_match', False),
            whole_word_match=data.get('whole_word_match', True),
            metadata=metadata
        )


class WorldBookService:
    """
    Service class for managing world books in a multi-user environment.
    
    This is a request-scoped service that is instantiated per API request.
    It works with the per-user database model where each user has their own
    separate database instance.
    """
    
    def __init__(self, db: CharactersRAGDB):
        """
        Initialize the service with a user-specific database connection.
        
        Args:
            db: User-specific database instance from dependency injection
        """
        self.db = db
        self._init_tables()
        
        # Request-scoped cache
        self._entry_cache: Optional[Dict[int, List[WorldBookEntry]]] = {}
        self._book_cache: Optional[Dict[int, Dict[str, Any]]] = {}
    
    def _init_tables(self):
        """Initialize world book tables in the user's database if they don't exist."""
        try:
            with self.db.get_connection() as conn:
                # Create world_books table
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS world_books (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        name TEXT NOT NULL UNIQUE,
                        description TEXT,
                        scan_depth INTEGER DEFAULT 3,
                        token_budget INTEGER DEFAULT 500,
                        recursive_scanning BOOLEAN DEFAULT 0,
                        enabled BOOLEAN DEFAULT 1,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        last_modified TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        version INTEGER DEFAULT 1,
                        deleted BOOLEAN DEFAULT 0
                    )
                """)
                
                # Create world_book_entries table
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS world_book_entries (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        world_book_id INTEGER NOT NULL,
                        keywords TEXT NOT NULL,
                        content TEXT NOT NULL,
                        priority INTEGER DEFAULT 0,
                        enabled BOOLEAN DEFAULT 1,
                        case_sensitive BOOLEAN DEFAULT 0,
                        regex_match BOOLEAN DEFAULT 0,
                        whole_word_match BOOLEAN DEFAULT 1,
                        metadata TEXT DEFAULT '{}',
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        last_modified TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        FOREIGN KEY (world_book_id) REFERENCES world_books(id) ON DELETE CASCADE
                    )
                """)
                
                # Create character_world_books linking table
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS character_world_books (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        character_id INTEGER NOT NULL,
                        world_book_id INTEGER NOT NULL,
                        enabled BOOLEAN DEFAULT 1,
                        priority INTEGER DEFAULT 0,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        UNIQUE(character_id, world_book_id),
                        FOREIGN KEY (character_id) REFERENCES character_cards(id) ON DELETE CASCADE,
                        FOREIGN KEY (world_book_id) REFERENCES world_books(id) ON DELETE CASCADE
                    )
                """)
                
                # Create indexes
                conn.execute("CREATE INDEX IF NOT EXISTS idx_wb_entries_book_id ON world_book_entries(world_book_id)")
                conn.execute("CREATE INDEX IF NOT EXISTS idx_wb_entries_priority ON world_book_entries(priority DESC)")
                conn.execute("CREATE INDEX IF NOT EXISTS idx_char_wb_char_id ON character_world_books(character_id)")
                conn.execute("CREATE INDEX IF NOT EXISTS idx_char_wb_book_id ON character_world_books(world_book_id)")
                
                conn.commit()
                logger.info("World book tables initialized")
        except Exception as e:
            logger.error(f"Failed to initialize world book tables: {e}")
            raise CharactersRAGDBError(f"Failed to initialize world book tables: {e}")
    
    # --- World Book CRUD Operations ---
    
    def create_world_book(
        self,
        name: str,
        description: Optional[str] = None,
        scan_depth: int = 3,
        token_budget: int = 500,
        recursive_scanning: bool = False,
        enabled: bool = True
    ) -> int:
        """
        Create a new world book.
        
        Args:
            name: Unique name for the world book
            description: Optional description
            scan_depth: How many messages to scan for keywords
            token_budget: Maximum tokens to use for world info
            recursive_scanning: Whether to scan matched entries for more keywords
            enabled: Whether the world book is active
            
        Returns:
            The ID of the created world book
            
        Raises:
            InputError: If name is empty or invalid
            ConflictError: If a world book with this name already exists
        """
        if not name or not name.strip():
            raise InputError("World book name cannot be empty")
        
        try:
            with self.db.get_connection() as conn:
                cursor = conn.execute(
                    """
                    INSERT INTO world_books 
                    (name, description, scan_depth, token_budget, recursive_scanning, enabled)
                    VALUES (?, ?, ?, ?, ?, ?)
                    """,
                    (name.strip(), description, scan_depth, token_budget, recursive_scanning, enabled)
                )
                world_book_id = cursor.lastrowid
                conn.commit()
                
                logger.info(f"Created world book '{name}' with ID {world_book_id}")
                self._invalidate_cache()
                return world_book_id
                
        except sqlite3.IntegrityError as e:
            if "UNIQUE constraint failed" in str(e):
                raise ConflictError(f"World book with name '{name}' already exists", "world_books", name)
            raise CharactersRAGDBError(f"Database error creating world book: {e}")
    
    def get_world_book(self, world_book_id: Optional[int] = None, name: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """
        Get a world book by ID or name.
        
        Args:
            world_book_id: Optional world book ID
            name: Optional world book name
            
        Returns:
            Dictionary with world book data or None if not found
        """
        # Check cache first
        if world_book_id and world_book_id in self._book_cache:
            return self._book_cache[world_book_id]
        
        try:
            with self.db.get_connection() as conn:
                if world_book_id:
                    cursor = conn.execute(
                        """
                        SELECT * FROM world_books 
                        WHERE id = ? AND deleted = 0
                        """,
                        (world_book_id,)
                    )
                elif name:
                    cursor = conn.execute(
                        """
                        SELECT * FROM world_books 
                        WHERE name = ? AND deleted = 0
                        """,
                        (name,)
                    )
                else:
                    return None
                
                row = cursor.fetchone()
                if row:
                    book_data = dict(row)
                    if world_book_id:
                        self._book_cache[world_book_id] = book_data
                    return book_data
                return None
                
        except Exception as e:
            logger.error(f"Error fetching world book: {e}")
            raise CharactersRAGDBError(f"Error fetching world book: {e}")
    
    def list_world_books(self, include_disabled: bool = False) -> List[Dict[str, Any]]:
        """
        List all world books for the user.
        
        Args:
            include_disabled: Whether to include disabled world books
            
        Returns:
            List of world book data dictionaries
        """
        try:
            with self.db.get_connection() as conn:
                query = "SELECT * FROM world_books WHERE deleted = 0"
                if not include_disabled:
                    query += " AND enabled = 1"
                query += " ORDER BY name"
                
                cursor = conn.execute(query)
                books = [dict(row) for row in cursor.fetchall()]
                
                # Cache the books
                for book in books:
                    self._book_cache[book['id']] = book
                
                return books
                
        except Exception as e:
            logger.error(f"Error listing world books: {e}")
            raise CharactersRAGDBError(f"Error listing world books: {e}")
    
    def update_world_book(
        self,
        world_book_id: int,
        name: Optional[str] = None,
        description: Optional[str] = None,
        scan_depth: Optional[int] = None,
        token_budget: Optional[int] = None,
        recursive_scanning: Optional[bool] = None,
        enabled: Optional[bool] = None
    ) -> bool:
        """
        Update a world book's settings.
        
        Args:
            world_book_id: World book ID
            Various optional fields to update
            
        Returns:
            True if updated successfully
            
        Raises:
            ConflictError: If the new name conflicts with an existing world book
        """
        try:
            updates = []
            params = []
            
            if name is not None:
                updates.append("name = ?")
                params.append(name.strip())
            if description is not None:
                updates.append("description = ?")
                params.append(description)
            if scan_depth is not None:
                updates.append("scan_depth = ?")
                params.append(scan_depth)
            if token_budget is not None:
                updates.append("token_budget = ?")
                params.append(token_budget)
            if recursive_scanning is not None:
                updates.append("recursive_scanning = ?")
                params.append(int(recursive_scanning))
            if enabled is not None:
                updates.append("enabled = ?")
                params.append(int(enabled))
            
            if not updates:
                return True
            
            updates.append("last_modified = CURRENT_TIMESTAMP")
            updates.append("version = version + 1")
            params.append(world_book_id)
            
            with self.db.get_connection() as conn:
                cursor = conn.execute(
                    f"UPDATE world_books SET {', '.join(updates)} WHERE id = ? AND deleted = 0",
                    params
                )
                conn.commit()
                
                if cursor.rowcount > 0:
                    logger.info(f"Updated world book {world_book_id}")
                    self._invalidate_cache()
                    return True
                return False
                
        except sqlite3.IntegrityError as e:
            if "UNIQUE constraint failed" in str(e):
                raise ConflictError(f"World book name '{name}' already exists", "world_books", name)
            raise CharactersRAGDBError(f"Database error updating world book: {e}")
    
    def delete_world_book(self, world_book_id: int, hard_delete: bool = False) -> bool:
        """
        Delete a world book (soft delete by default).
        
        Args:
            world_book_id: World book ID
            hard_delete: If True, permanently delete; otherwise soft delete
            
        Returns:
            True if deleted successfully
        """
        try:
            with self.db.get_connection() as conn:
                if hard_delete:
                    cursor = conn.execute(
                        "DELETE FROM world_books WHERE id = ?",
                        (world_book_id,)
                    )
                else:
                    cursor = conn.execute(
                        "UPDATE world_books SET deleted = 1, last_modified = CURRENT_TIMESTAMP WHERE id = ?",
                        (world_book_id,)
                    )
                conn.commit()
                
                if cursor.rowcount > 0:
                    logger.info(f"{'Hard' if hard_delete else 'Soft'} deleted world book {world_book_id}")
                    self._invalidate_cache()
                    return True
                return False
                
        except Exception as e:
            logger.error(f"Error deleting world book: {e}")
            raise CharactersRAGDBError(f"Error deleting world book: {e}")
    
    # --- Entry CRUD Operations ---
    
    def add_entry(
        self,
        world_book_id: int,
        keywords: List[str],
        content: str,
        priority: int = 0,
        enabled: bool = True,
        case_sensitive: bool = False,
        regex_match: bool = False,
        whole_word_match: bool = True,
        metadata: Optional[Dict[str, Any]] = None
    ) -> int:
        """
        Add an entry to a world book.
        
        Args:
            world_book_id: World book ID
            keywords: List of keywords to match
            content: Content to inject when matched
            priority: Priority for ordering (higher = more important)
            enabled: Whether entry is active
            case_sensitive: Whether keyword matching is case-sensitive
            regex_match: Whether keywords are regex patterns
            whole_word_match: Whether to match whole words only
            metadata: Additional metadata
            
        Returns:
            The ID of the created entry
        """
        if not keywords:
            raise InputError("Entry must have at least one keyword")
        if not content:
            raise InputError("Entry content cannot be empty")
        
        # Validate regex patterns if regex_match is True
        if regex_match:
            for keyword in keywords:
                try:
                    re.compile(keyword)
                except re.error as e:
                    raise InputError(f"Invalid regex pattern '{keyword}': {e}")
        
        keywords_json = json.dumps(keywords)
        metadata_json = json.dumps(metadata or {})
        
        try:
            with self.db.get_connection() as conn:
                cursor = conn.execute(
                    """
                    INSERT INTO world_book_entries 
                    (world_book_id, keywords, content, priority, enabled, 
                     case_sensitive, regex_match, whole_word_match, metadata)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (world_book_id, keywords_json, content, priority, enabled,
                     case_sensitive, regex_match, whole_word_match, metadata_json)
                )
                entry_id = cursor.lastrowid
                conn.commit()
                
                logger.info(f"Added entry {entry_id} to world book {world_book_id}")
                self._invalidate_cache()
                return entry_id
                
        except Exception as e:
            logger.error(f"Error adding world book entry: {e}")
            raise CharactersRAGDBError(f"Error adding world book entry: {e}")
    
    def get_entries(
        self,
        world_book_id: Optional[int] = None,
        enabled_only: bool = True
    ) -> List[WorldBookEntry]:
        """
        Get world book entries.
        
        Args:
            world_book_id: Optional specific world book ID
            enabled_only: Only return enabled entries
            
        Returns:
            List of WorldBookEntry instances
        """
        # Check cache first
        if world_book_id and world_book_id in self._entry_cache:
            entries = self._entry_cache[world_book_id]
            if enabled_only:
                entries = [e for e in entries if e.enabled]
            return entries
        
        try:
            with self.db.get_connection() as conn:
                query = """
                    SELECT e.*, wb.enabled as book_enabled
                    FROM world_book_entries e
                    JOIN world_books wb ON e.world_book_id = wb.id
                    WHERE wb.deleted = 0
                """
                params = []
                
                if world_book_id:
                    query += " AND e.world_book_id = ?"
                    params.append(world_book_id)
                if enabled_only:
                    query += " AND e.enabled = 1 AND wb.enabled = 1"
                
                query += " ORDER BY e.priority DESC, e.id"
                
                cursor = conn.execute(query, params)
                entries = []
                
                for row in cursor.fetchall():
                    entry_data = dict(row)
                    entry = WorldBookEntry.from_dict(entry_data)
                    entries.append(entry)
                
                # Cache if fetching for specific book
                if world_book_id:
                    self._entry_cache[world_book_id] = entries
                
                return entries
                
        except Exception as e:
            logger.error(f"Error fetching world book entries: {e}")
            raise CharactersRAGDBError(f"Error fetching world book entries: {e}")
    
    def update_entry(
        self,
        entry_id: int,
        keywords: Optional[List[str]] = None,
        content: Optional[str] = None,
        priority: Optional[int] = None,
        enabled: Optional[bool] = None,
        case_sensitive: Optional[bool] = None,
        regex_match: Optional[bool] = None,
        whole_word_match: Optional[bool] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> bool:
        """
        Update a world book entry.
        
        Args:
            entry_id: Entry ID
            Various optional fields to update
            
        Returns:
            True if updated successfully
        """
        try:
            updates = []
            params = []
            
            if keywords is not None:
                if not keywords:
                    raise InputError("Entry must have at least one keyword")
                if regex_match:
                    for keyword in keywords:
                        try:
                            re.compile(keyword)
                        except re.error as e:
                            raise InputError(f"Invalid regex pattern '{keyword}': {e}")
                updates.append("keywords = ?")
                params.append(json.dumps(keywords))
                
            if content is not None:
                if not content:
                    raise InputError("Entry content cannot be empty")
                updates.append("content = ?")
                params.append(content)
                
            if priority is not None:
                updates.append("priority = ?")
                params.append(priority)
                
            if enabled is not None:
                updates.append("enabled = ?")
                params.append(int(enabled))
                
            if case_sensitive is not None:
                updates.append("case_sensitive = ?")
                params.append(int(case_sensitive))
                
            if regex_match is not None:
                updates.append("regex_match = ?")
                params.append(int(regex_match))
                
            if whole_word_match is not None:
                updates.append("whole_word_match = ?")
                params.append(int(whole_word_match))
                
            if metadata is not None:
                updates.append("metadata = ?")
                params.append(json.dumps(metadata))
            
            if not updates:
                return True
            
            updates.append("last_modified = CURRENT_TIMESTAMP")
            params.append(entry_id)
            
            with self.db.get_connection() as conn:
                cursor = conn.execute(
                    f"UPDATE world_book_entries SET {', '.join(updates)} WHERE id = ?",
                    params
                )
                conn.commit()
                
                if cursor.rowcount > 0:
                    logger.info(f"Updated world book entry {entry_id}")
                    self._invalidate_cache()
                    return True
                return False
                
        except InputError:
            raise
        except Exception as e:
            logger.error(f"Error updating world book entry: {e}")
            raise CharactersRAGDBError(f"Error updating world book entry: {e}")
    
    def delete_entry(self, entry_id: int) -> bool:
        """
        Delete a world book entry.
        
        Args:
            entry_id: Entry ID
            
        Returns:
            True if deleted successfully
        """
        try:
            with self.db.get_connection() as conn:
                cursor = conn.execute(
                    "DELETE FROM world_book_entries WHERE id = ?",
                    (entry_id,)
                )
                conn.commit()
                
                if cursor.rowcount > 0:
                    logger.info(f"Deleted world book entry {entry_id}")
                    self._invalidate_cache()
                    return True
                return False
                
        except Exception as e:
            logger.error(f"Error deleting world book entry: {e}")
            raise CharactersRAGDBError(f"Error deleting world book entry: {e}")
    
    # --- Character Association ---
    
    def attach_to_character(
        self,
        character_id: int,
        world_book_id: int,
        enabled: bool = True,
        priority: int = 0
    ) -> bool:
        """
        Attach a world book to a character.
        
        Args:
            character_id: Character ID
            world_book_id: World book ID
            enabled: Whether the attachment is active
            priority: Priority for this character (higher = more important)
            
        Returns:
            True if attached successfully
        """
        try:
            with self.db.get_connection() as conn:
                conn.execute(
                    """
                    INSERT OR REPLACE INTO character_world_books 
                    (character_id, world_book_id, enabled, priority)
                    VALUES (?, ?, ?, ?)
                    """,
                    (character_id, world_book_id, enabled, priority)
                )
                conn.commit()
                
                logger.info(f"Attached world book {world_book_id} to character {character_id}")
                return True
                
        except Exception as e:
            logger.error(f"Error attaching world book to character: {e}")
            raise CharactersRAGDBError(f"Error attaching world book to character: {e}")
    
    def detach_from_character(self, character_id: int, world_book_id: int) -> bool:
        """
        Detach a world book from a character.
        
        Args:
            character_id: Character ID
            world_book_id: World book ID
            
        Returns:
            True if detached successfully
        """
        try:
            with self.db.get_connection() as conn:
                cursor = conn.execute(
                    """
                    DELETE FROM character_world_books 
                    WHERE character_id = ? AND world_book_id = ?
                    """,
                    (character_id, world_book_id)
                )
                conn.commit()
                
                if cursor.rowcount > 0:
                    logger.info(f"Detached world book {world_book_id} from character {character_id}")
                    return True
                return False
                
        except Exception as e:
            logger.error(f"Error detaching world book from character: {e}")
            raise CharactersRAGDBError(f"Error detaching world book from character: {e}")
    
    def get_character_world_books(self, character_id: int, enabled_only: bool = True) -> List[Dict[str, Any]]:
        """
        Get all world books attached to a character.
        
        Args:
            character_id: Character ID
            enabled_only: Only return enabled attachments
            
        Returns:
            List of world book data with attachment info
        """
        try:
            with self.db.get_connection() as conn:
                query = """
                    SELECT wb.*, cwb.enabled as attachment_enabled, cwb.priority as attachment_priority
                    FROM world_books wb
                    JOIN character_world_books cwb ON wb.id = cwb.world_book_id
                    WHERE cwb.character_id = ? AND wb.deleted = 0
                """
                params = [character_id]
                
                if enabled_only:
                    query += " AND cwb.enabled = 1 AND wb.enabled = 1"
                
                query += " ORDER BY cwb.priority DESC, wb.name"
                
                cursor = conn.execute(query, params)
                return [dict(row) for row in cursor.fetchall()]
                
        except Exception as e:
            logger.error(f"Error fetching character world books: {e}")
            raise CharactersRAGDBError(f"Error fetching character world books: {e}")
    
    # --- Content Processing ---
    
    def process_context(
        self,
        text: str,
        world_book_ids: Optional[List[int]] = None,
        character_id: Optional[int] = None,
        scan_depth: int = 3,
        token_budget: int = 500,
        recursive_scanning: bool = False
    ) -> Dict[str, Any]:
        """
        Process text to find and inject relevant world info.
        
        Args:
            text: Text to scan for keywords (usually recent messages)
            world_book_ids: Specific world books to use (optional)
            character_id: Character whose world books to use (optional)
            scan_depth: Override for scan depth
            token_budget: Maximum tokens to inject
            recursive_scanning: Whether to scan matched entries for more keywords
            
        Returns:
            Dictionary with processed_context and statistics
        """
        # Gather applicable world books
        books_to_use = []
        
        if world_book_ids:
            for book_id in world_book_ids:
                book = self.get_world_book(book_id)
                if book and book.get('enabled', True):
                    books_to_use.append(book)
        
        if character_id:
            char_books = self.get_character_world_books(character_id, enabled_only=True)
            books_to_use.extend(char_books)
        
        if not books_to_use:
            return {
                "processed_context": "",
                "entries_matched": 0,
                "tokens_used": 0,
                "books_used": 0
            }
        
        # Gather all entries from applicable books
        all_entries = []
        for book in books_to_use:
            entries = self.get_entries(book['id'], enabled_only=True)
            all_entries.extend(entries)
        
        # Sort by priority (highest first)
        all_entries.sort(key=lambda e: e.priority, reverse=True)
        
        # Find matching entries
        matched_entries = []
        tokens_used = 0
        
        for entry in all_entries:
            if entry.matches(text):
                # Estimate tokens (simple approximation)
                entry_tokens = len(entry.content.split())
                if tokens_used + entry_tokens <= token_budget:
                    matched_entries.append(entry)
                    tokens_used += entry_tokens
                else:
                    break  # Token budget exceeded
        
        # Handle recursive scanning
        if recursive_scanning and matched_entries:
            combined_content = " ".join(e.content for e in matched_entries)
            additional_entries = []
            
            for entry in all_entries:
                if entry not in matched_entries and entry.matches(combined_content):
                    entry_tokens = len(entry.content.split())
                    if tokens_used + entry_tokens <= token_budget:
                        additional_entries.append(entry)
                        tokens_used += entry_tokens
            
            matched_entries.extend(additional_entries)
        
        # Build injected content
        if matched_entries:
            # Sort by priority for final output
            matched_entries.sort(key=lambda e: e.priority, reverse=True)
            injected_content = "\n\n".join(e.content for e in matched_entries)
        else:
            injected_content = ""
        
        return {
            "processed_context": injected_content,
            "entries_matched": len(matched_entries),
            "tokens_used": tokens_used,
            "books_used": len(set(e.world_book_id for e in matched_entries)) if matched_entries else 0,
            "entry_ids": [e.entry_id for e in matched_entries]
        }
    
    # --- Import/Export ---
    
    def export_world_book(self, world_book_id: int) -> Dict[str, Any]:
        """
        Export a world book to a dictionary format.
        
        Args:
            world_book_id: World book ID
            
        Returns:
            Dictionary with world book data and entries
        """
        book = self.get_world_book(world_book_id)
        if not book:
            raise InputError(f"World book {world_book_id} not found")
        
        entries = self.get_entries(world_book_id, enabled_only=False)
        
        return {
            "world_book": book,
            "entries": [e.to_dict() for e in entries]
        }
    
    def import_world_book(self, data: Dict[str, Any], merge_on_conflict: bool = False) -> int:
        """
        Import a world book from dictionary format.
        
        Args:
            data: Dictionary with world book data and entries
            merge_on_conflict: If True, merge with existing book of same name
            
        Returns:
            ID of the imported/merged world book
        """
        book_data = data.get("world_book", {})
        entries_data = data.get("entries", [])
        
        if not book_data.get("name"):
            raise InputError("World book must have a name")
        
        # Check for existing book
        existing = self.get_world_book(name=book_data["name"])
        
        if existing:
            if not merge_on_conflict:
                raise ConflictError(f"World book '{book_data['name']}' already exists")
            world_book_id = existing["id"]
            logger.info(f"Merging into existing world book {world_book_id}")
        else:
            # Create new world book
            world_book_id = self.create_world_book(
                name=book_data["name"],
                description=book_data.get("description"),
                scan_depth=book_data.get("scan_depth", 3),
                token_budget=book_data.get("token_budget", 500),
                recursive_scanning=book_data.get("recursive_scanning", False),
                enabled=book_data.get("enabled", True)
            )
        
        # Import entries
        for entry_data in entries_data:
            self.add_entry(
                world_book_id=world_book_id,
                keywords=entry_data.get("keywords", []),
                content=entry_data.get("content", ""),
                priority=entry_data.get("priority", 0),
                enabled=entry_data.get("enabled", True),
                case_sensitive=entry_data.get("case_sensitive", False),
                regex_match=entry_data.get("regex_match", False),
                whole_word_match=entry_data.get("whole_word_match", True),
                metadata=entry_data.get("metadata", {})
            )
        
        logger.info(f"Imported world book with {len(entries_data)} entries")
        return world_book_id
    
    # --- Helper Methods ---
    
    def _invalidate_cache(self):
        """Invalidate the request-scoped cache."""
        self._entry_cache = {}
        self._book_cache = {}
    
    def _estimate_tokens(self, text: str) -> int:
        """
        Estimate token count for text.
        
        Simple approximation: 1 token ≈ 4 characters or 0.75 words
        """
        return max(len(text) // 4, len(text.split()) * 3 // 4)
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        Get world book usage statistics.
        
        Returns:
            Dictionary containing statistics
        """
        try:
            with self.db.get_connection() as conn:
                # Get world book counts
                cursor = conn.execute("""
                    SELECT 
                        COUNT(*) as total_world_books,
                        SUM(CASE WHEN enabled = 1 THEN 1 ELSE 0 END) as enabled_world_books
                    FROM world_books 
                    WHERE deleted = 0
                """)
                book_stats = dict(cursor.fetchone())
                
                # Get entry counts
                cursor = conn.execute("""
                    SELECT COUNT(*) as total_entries
                    FROM world_book_entries e
                    JOIN world_books w ON e.world_book_id = w.id
                    WHERE w.deleted = 0
                """)
                entry_stats = dict(cursor.fetchone())
                
                # Get character attachment counts
                cursor = conn.execute("""
                    SELECT COUNT(DISTINCT character_id) as total_attachments
                    FROM character_world_books c
                    JOIN world_books w ON c.world_book_id = w.id
                    WHERE w.deleted = 0
                """)
                attachment_stats = dict(cursor.fetchone())
                
                # Calculate average entries per world book
                avg_entries = 0
                if book_stats['total_world_books'] > 0:
                    avg_entries = entry_stats['total_entries'] / book_stats['total_world_books']
                
                return {
                    "total_world_books": book_stats['total_world_books'],
                    "enabled_world_books": book_stats['enabled_world_books'],
                    "total_entries": entry_stats['total_entries'],
                    "total_character_attachments": attachment_stats['total_attachments'],
                    "average_entries_per_world_book": avg_entries
                }
                
        except Exception as e:
            logger.error(f"Error getting statistics: {e}")
            raise CharactersRAGDBError(f"Error getting statistics: {e}")
    
    def search_entries(self, search_term: str) -> List[Dict[str, Any]]:
        """
        Search for entries by keyword or content.
        
        Args:
            search_term: Search term to look for
            
        Returns:
            List of matching entries with world book info
        """
        try:
            with self.db.get_connection() as conn:
                cursor = conn.execute("""
                    SELECT 
                        e.id,
                        e.keywords,
                        e.content,
                        e.priority,
                        e.enabled,
                        w.name as world_book_name,
                        w.id as world_book_id
                    FROM world_book_entries e
                    JOIN world_books w ON e.world_book_id = w.id
                    WHERE w.deleted = 0
                    AND (e.keywords LIKE ? OR e.content LIKE ?)
                    ORDER BY w.name, e.priority DESC
                """, (f'%{search_term}%', f'%{search_term}%'))
                
                results = []
                for row in cursor.fetchall():
                    results.append(dict(row))
                
                return results
                
        except Exception as e:
            logger.error(f"Error searching entries: {e}")
            raise CharactersRAGDBError(f"Error searching entries: {e}")
    
    def bulk_update_entries(
        self,
        world_book_id: int,
        entry_ids: List[int],
        enabled: Optional[bool] = None,
        priority: Optional[int] = None
    ) -> int:
        """
        Bulk update entries.
        
        Args:
            world_book_id: World book ID
            entry_ids: List of entry IDs to update
            enabled: New enabled status (optional)
            priority: New priority (optional)
            
        Returns:
            Number of entries updated
        """
        try:
            if not entry_ids:
                return 0
            
            updates = []
            params = []
            
            if enabled is not None:
                updates.append("enabled = ?")
                params.append(int(enabled))
            
            if priority is not None:
                updates.append("priority = ?")
                params.append(priority)
            
            if not updates:
                return 0
            
            updates.append("updated_at = CURRENT_TIMESTAMP")
            
            with self.db.get_connection() as conn:
                # Build the IN clause for entry IDs
                placeholders = ','.join('?' * len(entry_ids))
                params.extend(entry_ids)
                params.append(world_book_id)
                
                cursor = conn.execute(
                    f"""
                    UPDATE world_book_entries 
                    SET {', '.join(updates)}
                    WHERE id IN ({placeholders})
                    AND world_book_id = ?
                    """,
                    params
                )
                conn.commit()
                
                updated_count = cursor.rowcount
                logger.info(f"Updated {updated_count} entries in world book {world_book_id}")
                self._invalidate_cache()
                return updated_count
                
        except Exception as e:
            logger.error(f"Error bulk updating entries: {e}")
            raise CharactersRAGDBError(f"Error bulk updating entries: {e}")
    
    def clone_world_book(self, source_wb_id: int, new_name: str) -> int:
        """
        Create a copy of a world book with all its entries.
        
        Args:
            source_wb_id: Source world book ID
            new_name: Name for the cloned world book
            
        Returns:
            ID of the new world book
        """
        try:
            # Get source world book
            source_wb = self.get_world_book(source_wb_id)
            if not source_wb:
                raise InputError(f"Source world book {source_wb_id} not found")
            
            # Create new world book
            new_wb_id = self.create_world_book(
                name=new_name,
                description=f"Cloned from {source_wb['name']}",
                scan_depth=source_wb.get('scan_depth', 3),
                token_budget=source_wb.get('token_budget', 500),
                recursive_scanning=source_wb.get('recursive_scanning', False),
                enabled=source_wb.get('enabled', True)
            )
            
            # Get all entries from source world book
            entries = self.get_entries(source_wb_id, enabled_only=False)
            
            # Add entries to new world book
            if entries:
                with self.db.get_connection() as conn:
                    for entry in entries:
                        metadata_json = json.dumps(entry.metadata) if entry.metadata else '{}'
                        
                        conn.execute(
                            """
                            INSERT INTO world_book_entries 
                            (world_book_id, keywords, content, priority, enabled, 
                             case_sensitive, regex_match, whole_word_match, metadata)
                            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                            """,
                            (new_wb_id, ','.join(entry.keywords), entry.content, 
                             entry.priority, int(entry.enabled), int(entry.case_sensitive),
                             int(entry.regex_match), int(entry.whole_word_match), metadata_json)
                        )
                    conn.commit()
            
            logger.info(f"Cloned world book {source_wb_id} to new world book {new_wb_id} with {len(entries)} entries")
            self._invalidate_cache()
            return new_wb_id
            
        except ConflictError:
            raise
        except Exception as e:
            logger.error(f"Error cloning world book: {e}")
            raise CharactersRAGDBError(f"Error cloning world book: {e}")
    
    def close(self):
        """
        Close the service and clean up resources.
        
        This method is provided for compatibility with test fixtures
        that expect a close method. Since the database connection is
        managed by CharactersRAGDB, this is a no-op.
        """
        # Database connections are managed by CharactersRAGDB
        # Nothing to close here
        pass


# Export main classes
__all__ = [
    'WorldBookEntry',
    'WorldBookService'
]