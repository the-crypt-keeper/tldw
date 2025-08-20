"""
World Book Manager - CRUD operations for independent world books/lorebooks.

This module provides functions for managing world books that can be used
independently of characters, allowing shared lorebooks across conversations.
"""

import json
import logging
import sqlite3
from typing import List, Dict, Any, Optional, Tuple, Union
from datetime import datetime, timezone

from loguru import logger

from tldw_chatbook.DB.ChaChaNotes_DB import CharactersRAGDB, InputError, ConflictError, CharactersRAGDBError


class WorldBookManager:
    """Manages world books and their entries in the database."""
    
    def __init__(self, db: CharactersRAGDB):
        """
        Initialize the WorldBookManager with a database connection.
        
        Args:
            db: CharactersRAGDB instance for database operations
        """
        self.db = db
    
    # --- World Book CRUD Operations ---
    
    def create_world_book(self, name: str, description: Optional[str] = None,
                         scan_depth: int = 3, token_budget: int = 500,
                         recursive_scanning: bool = False, enabled: bool = True) -> int:
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
        
        query = """
        INSERT INTO world_books (name, description, scan_depth, token_budget, 
                                recursive_scanning, enabled, client_id)
        VALUES (?, ?, ?, ?, ?, ?, ?)
        """
        
        try:
            with self.db.transaction() as cursor:
                cursor.execute(query, (
                    name.strip(), description, scan_depth, token_budget,
                    recursive_scanning, enabled, self.db.client_id
                ))
                world_book_id = cursor.lastrowid
                logger.info(f"Created world book '{name}' with ID {world_book_id}")
                return world_book_id
        except sqlite3.IntegrityError as e:
            if "UNIQUE constraint failed" in str(e):
                raise ConflictError(f"World book with name '{name}' already exists")
            raise CharactersRAGDBError(f"Database error creating world book: {e}")
    
    def get_world_book(self, world_book_id: int) -> Optional[Dict[str, Any]]:
        """
        Get a world book by ID.
        
        Args:
            world_book_id: The ID of the world book
            
        Returns:
            Dictionary with world book data or None if not found
        """
        query = """
        SELECT id, name, description, scan_depth, token_budget, recursive_scanning,
               enabled, created_at, last_modified, deleted, client_id, version
        FROM world_books
        WHERE id = ? AND deleted = 0
        """
        
        with self.db.transaction() as cursor:
            cursor.execute(query, (world_book_id,))
            row = cursor.fetchone()
            
            if row:
                return {
                    'id': row[0],
                    'name': row[1],
                    'description': row[2],
                    'scan_depth': row[3],
                    'token_budget': row[4],
                    'recursive_scanning': bool(row[5]),
                    'enabled': bool(row[6]),
                    'created_at': row[7],
                    'last_modified': row[8],
                    'deleted': bool(row[9]),
                    'client_id': row[10],
                    'version': row[11]
                }
            return None
    
    def get_world_book_by_name(self, name: str) -> Optional[Dict[str, Any]]:
        """
        Get a world book by name.
        
        Args:
            name: The name of the world book
            
        Returns:
            Dictionary with world book data or None if not found
        """
        query = """
        SELECT id, name, description, scan_depth, token_budget, recursive_scanning,
               enabled, created_at, last_modified, deleted, client_id, version
        FROM world_books
        WHERE name = ? AND deleted = 0
        """
        
        with self.db.transaction() as cursor:
            cursor.execute(query, (name,))
            row = cursor.fetchone()
            
            if row:
                return {
                    'id': row[0],
                    'name': row[1],
                    'description': row[2],
                    'scan_depth': row[3],
                    'token_budget': row[4],
                    'recursive_scanning': bool(row[5]),
                    'enabled': bool(row[6]),
                    'created_at': row[7],
                    'last_modified': row[8],
                    'deleted': bool(row[9]),
                    'client_id': row[10],
                    'version': row[11]
                }
            return None
    
    def list_world_books(self, include_disabled: bool = False) -> List[Dict[str, Any]]:
        """
        List all world books.
        
        Args:
            include_disabled: Whether to include disabled world books
            
        Returns:
            List of world book dictionaries
        """
        query = """
        SELECT id, name, description, scan_depth, token_budget, recursive_scanning,
               enabled, created_at, last_modified, deleted, client_id, version
        FROM world_books
        WHERE deleted = 0
        """
        
        if not include_disabled:
            query += " AND enabled = 1"
        
        query += " ORDER BY name"
        
        with self.db.transaction() as cursor:
            cursor.execute(query)
            
            world_books = []
            for row in cursor.fetchall():
                world_books.append({
                    'id': row[0],
                    'name': row[1],
                    'description': row[2],
                    'scan_depth': row[3],
                    'token_budget': row[4],
                    'recursive_scanning': bool(row[5]),
                    'enabled': bool(row[6]),
                    'created_at': row[7],
                    'last_modified': row[8],
                    'deleted': bool(row[9]),
                    'client_id': row[10],
                    'version': row[11]
                })
            
            return world_books
    
    def update_world_book(self, world_book_id: int, name: Optional[str] = None,
                         description: Optional[str] = None, scan_depth: Optional[int] = None,
                         token_budget: Optional[int] = None, recursive_scanning: Optional[bool] = None,
                         enabled: Optional[bool] = None, expected_version: Optional[int] = None) -> bool:
        """
        Update a world book.
        
        Args:
            world_book_id: The ID of the world book to update
            name: New name (optional)
            description: New description (optional)
            scan_depth: New scan depth (optional)
            token_budget: New token budget (optional)
            recursive_scanning: New recursive scanning setting (optional)
            enabled: New enabled status (optional)
            expected_version: Expected version for optimistic locking (optional)
            
        Returns:
            True if updated successfully
            
        Raises:
            ConflictError: If version mismatch or name already exists
        """
        # Build update query dynamically
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
            params.append(recursive_scanning)
        if enabled is not None:
            updates.append("enabled = ?")
            params.append(enabled)
        
        if not updates:
            return True  # Nothing to update
        
        # Add standard update fields
        updates.extend(["last_modified = CURRENT_TIMESTAMP", "version = version + 1", "client_id = ?"])
        params.append(self.db.client_id)
        
        query = f"UPDATE world_books SET {', '.join(updates)} WHERE id = ? AND deleted = 0"
        params.append(world_book_id)
        
        if expected_version is not None:
            query += " AND version = ?"
            params.append(expected_version)
        
        try:
            with self.db.transaction() as cursor:
                cursor.execute(query, params)
                if cursor.rowcount == 0:
                    if expected_version is not None:
                        raise ConflictError(f"Version mismatch updating world book {world_book_id}")
                    return False
                return True
        except sqlite3.IntegrityError as e:
            if "UNIQUE constraint failed" in str(e):
                raise ConflictError(f"World book with name '{name}' already exists")
            raise
    
    def delete_world_book(self, world_book_id: int, expected_version: Optional[int] = None) -> bool:
        """
        Soft delete a world book.
        
        Args:
            world_book_id: The ID of the world book to delete
            expected_version: Expected version for optimistic locking (optional)
            
        Returns:
            True if deleted successfully
            
        Raises:
            ConflictError: If version mismatch
        """
        query = """
        UPDATE world_books 
        SET deleted = 1, last_modified = CURRENT_TIMESTAMP, version = version + 1, client_id = ?
        WHERE id = ? AND deleted = 0
        """
        
        params = [self.db.client_id, world_book_id]
        
        if expected_version is not None:
            query += " AND version = ?"
            params.append(expected_version)
        
        with self.db.transaction() as cursor:
            cursor.execute(query, params)
            if cursor.rowcount == 0:
                if expected_version is not None:
                    raise ConflictError(f"Version mismatch deleting world book {world_book_id}")
                return False
            return True
    
    # --- World Book Entry CRUD Operations ---
    
    def create_world_book_entry(self, world_book_id: int, keys: List[str], content: str,
                               enabled: bool = True, position: str = 'before_char',
                               insertion_order: int = 0, selective: bool = False,
                               secondary_keys: Optional[List[str]] = None,
                               case_sensitive: bool = False,
                               extensions: Optional[Dict[str, Any]] = None) -> int:
        """
        Create a new world book entry.
        
        Args:
            world_book_id: The ID of the world book this entry belongs to
            keys: List of keywords that trigger this entry
            content: The content to inject when triggered
            enabled: Whether the entry is active
            position: Where to inject (before_char, after_char, at_start, at_end)
            insertion_order: Order for multiple entries
            selective: Whether to require secondary keys
            secondary_keys: Additional keys required if selective
            case_sensitive: Whether keyword matching is case sensitive
            extensions: Additional data for future features
            
        Returns:
            The ID of the created entry
            
        Raises:
            InputError: If keys or content are empty
        """
        if not keys or not any(k.strip() for k in keys):
            raise InputError("Entry must have at least one non-empty key")
        if not content or not content.strip():
            raise InputError("Entry content cannot be empty")
        
        # Clean and validate keys
        clean_keys = [k.strip() for k in keys if k.strip()]
        clean_secondary = [k.strip() for k in (secondary_keys or [])] if secondary_keys else []
        
        query = """
        INSERT INTO world_book_entries (world_book_id, keys, content, enabled, position,
                                       insertion_order, selective, secondary_keys,
                                       case_sensitive, extensions)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """
        
        with self.db.transaction() as cursor:
            cursor.execute(query, (
                world_book_id,
                json.dumps(clean_keys),
                content.strip(),
                enabled,
                position,
                insertion_order,
                selective,
                json.dumps(clean_secondary) if clean_secondary else None,
                case_sensitive,
                json.dumps(extensions) if extensions else None
            ))
            entry_id = cursor.lastrowid
            logger.info(f"Created world book entry {entry_id} for book {world_book_id}")
            return entry_id
    
    def get_world_book_entries(self, world_book_id: int, enabled_only: bool = False) -> List[Dict[str, Any]]:
        """
        Get all entries for a world book.
        
        Args:
            world_book_id: The ID of the world book
            enabled_only: Whether to return only enabled entries
            
        Returns:
            List of entry dictionaries
        """
        query = """
        SELECT id, world_book_id, keys, content, enabled, position, insertion_order,
               selective, secondary_keys, case_sensitive, extensions, created_at, last_modified
        FROM world_book_entries
        WHERE world_book_id = ?
        """
        
        if enabled_only:
            query += " AND enabled = 1"
        
        query += " ORDER BY insertion_order, id"
        
        with self.db.transaction() as cursor:
            cursor.execute(query, (world_book_id,))
            
            entries = []
            for row in cursor.fetchall():
                entries.append({
                    'id': row[0],
                    'world_book_id': row[1],
                    'keys': json.loads(row[2]) if row[2] else [],
                    'content': row[3],
                    'enabled': bool(row[4]),
                    'position': row[5],
                    'insertion_order': row[6],
                    'selective': bool(row[7]),
                    'secondary_keys': json.loads(row[8]) if row[8] else [],
                    'case_sensitive': bool(row[9]),
                    'extensions': json.loads(row[10]) if row[10] else {},
                    'created_at': row[11],
                    'last_modified': row[12]
                })
            
            return entries
    
    def update_world_book_entry(self, entry_id: int, **kwargs) -> bool:
        """
        Update a world book entry.
        
        Args:
            entry_id: The ID of the entry to update
            **kwargs: Fields to update (keys, content, enabled, position, etc.)
            
        Returns:
            True if updated successfully
        """
        # Build update query dynamically
        updates = []
        params = []
        
        # Handle each possible field
        for field in ['keys', 'content', 'enabled', 'position', 'insertion_order',
                     'selective', 'secondary_keys', 'case_sensitive', 'extensions']:
            if field in kwargs:
                value = kwargs[field]
                if field in ['keys', 'secondary_keys', 'extensions']:
                    value = json.dumps(value) if value else None
                updates.append(f"{field} = ?")
                params.append(value)
        
        if not updates:
            return True  # Nothing to update
        
        # Add last_modified update
        updates.append("last_modified = CURRENT_TIMESTAMP")
        
        query = f"UPDATE world_book_entries SET {', '.join(updates)} WHERE id = ?"
        params.append(entry_id)
        
        with self.db.transaction() as cursor:
            cursor.execute(query, params)
            return cursor.rowcount > 0
    
    def delete_world_book_entry(self, entry_id: int) -> bool:
        """
        Delete a world book entry.
        
        Args:
            entry_id: The ID of the entry to delete
            
        Returns:
            True if deleted successfully
        """
        query = "DELETE FROM world_book_entries WHERE id = ?"
        
        with self.db.transaction() as cursor:
            cursor.execute(query, (entry_id,))
            return cursor.rowcount > 0
    
    # --- Conversation Association Functions ---
    
    def associate_world_book_with_conversation(self, conversation_id: int, world_book_id: int,
                                              priority: int = 0) -> bool:
        """
        Associate a world book with a conversation.
        
        Args:
            conversation_id: The ID of the conversation
            world_book_id: The ID of the world book
            priority: Priority for ordering multiple world books
            
        Returns:
            True if associated successfully
        """
        query = """
        INSERT OR REPLACE INTO conversation_world_books (conversation_id, world_book_id, priority)
        VALUES (?, ?, ?)
        """
        
        with self.db.transaction() as cursor:
            cursor.execute(query, (conversation_id, world_book_id, priority))
            return True
    
    def disassociate_world_book_from_conversation(self, conversation_id: int, world_book_id: int) -> bool:
        """
        Remove association between a world book and conversation.
        
        Args:
            conversation_id: The ID of the conversation
            world_book_id: The ID of the world book
            
        Returns:
            True if disassociated successfully
        """
        query = """
        DELETE FROM conversation_world_books
        WHERE conversation_id = ? AND world_book_id = ?
        """
        
        with self.db.transaction() as cursor:
            cursor.execute(query, (conversation_id, world_book_id))
            return cursor.rowcount > 0
    
    def get_world_books_for_conversation(self, conversation_id: int, enabled_only: bool = True) -> List[Dict[str, Any]]:
        """
        Get all world books associated with a conversation.
        
        Args:
            conversation_id: The ID of the conversation
            enabled_only: Whether to return only enabled world books
            
        Returns:
            List of world book dictionaries with their entries
        """
        query = """
        SELECT wb.id, wb.name, wb.description, wb.scan_depth, wb.token_budget,
               wb.recursive_scanning, wb.enabled, cwb.priority
        FROM world_books wb
        JOIN conversation_world_books cwb ON wb.id = cwb.world_book_id
        WHERE cwb.conversation_id = ? AND wb.deleted = 0
        """
        
        if enabled_only:
            query += " AND wb.enabled = 1"
        
        query += " ORDER BY cwb.priority DESC, wb.name"
        
        with self.db.transaction() as cursor:
            cursor.execute(query, (conversation_id,))
            
            world_books = []
            for row in cursor.fetchall():
                world_book = {
                    'id': row[0],
                    'name': row[1],
                    'description': row[2],
                    'scan_depth': row[3],
                    'token_budget': row[4],
                    'recursive_scanning': bool(row[5]),
                    'enabled': bool(row[6]),
                    'priority': row[7],
                    'entries': self.get_world_book_entries(row[0], enabled_only=enabled_only)
                }
                world_books.append(world_book)
            
            return world_books
    
    # --- Import/Export Functions ---
    
    def export_world_book(self, world_book_id: int) -> Dict[str, Any]:
        """
        Export a world book in a format compatible with other tools.
        
        Args:
            world_book_id: The ID of the world book to export
            
        Returns:
            Dictionary with world book data in standard format
        """
        world_book = self.get_world_book(world_book_id)
        if not world_book:
            raise InputError(f"World book {world_book_id} not found")
        
        entries = self.get_world_book_entries(world_book_id)
        
        # Format for export (compatible with SillyTavern character book format)
        export_data = {
            'name': world_book['name'],
            'description': world_book['description'],
            'scan_depth': world_book['scan_depth'],
            'token_budget': world_book['token_budget'],
            'recursive_scanning': world_book['recursive_scanning'],
            'entries': []
        }
        
        for entry in entries:
            export_data['entries'].append({
                'keys': entry['keys'],
                'content': entry['content'],
                'enabled': entry['enabled'],
                'position': entry['position'],
                'insertion_order': entry['insertion_order'],
                'selective': entry['selective'],
                'secondary_keys': entry['secondary_keys'],
                'case_sensitive': entry['case_sensitive'],
                'extensions': entry['extensions']
            })
        
        return export_data
    
    def import_world_book(self, data: Dict[str, Any], name_override: Optional[str] = None) -> int:
        """
        Import a world book from external data.
        
        Args:
            data: World book data in standard format
            name_override: Override the name to avoid conflicts
            
        Returns:
            The ID of the imported world book
        """
        # Extract world book metadata
        name = name_override or data.get('name', 'Imported World Book')
        description = data.get('description', '')
        scan_depth = data.get('scan_depth', 3)
        token_budget = data.get('token_budget', 500)
        recursive_scanning = data.get('recursive_scanning', False)
        
        # Create the world book
        world_book_id = self.create_world_book(
            name=name,
            description=description,
            scan_depth=scan_depth,
            token_budget=token_budget,
            recursive_scanning=recursive_scanning
        )
        
        # Import entries
        entries = data.get('entries', [])
        for i, entry in enumerate(entries):
            self.create_world_book_entry(
                world_book_id=world_book_id,
                keys=entry.get('keys', []),
                content=entry.get('content', ''),
                enabled=entry.get('enabled', True),
                position=entry.get('position', 'before_char'),
                insertion_order=entry.get('insertion_order', i),
                selective=entry.get('selective', False),
                secondary_keys=entry.get('secondary_keys', []),
                case_sensitive=entry.get('case_sensitive', False),
                extensions=entry.get('extensions', {})
            )
        
        logger.info(f"Imported world book '{name}' with {len(entries)} entries")
        return world_book_id