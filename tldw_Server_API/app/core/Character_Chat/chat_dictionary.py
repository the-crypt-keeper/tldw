# chat_dictionary.py
# Description: Chat Dictionary System with pattern-based text replacement
# Adapted from single-user to multi-user architecture with per-user database isolation
#
"""
Chat Dictionary Service for Multi-User Environment
--------------------------------------------------

This module provides a pattern-based text replacement system for chat conversations.
Adapted from the single-user TUI version to work with the multi-user API architecture.

Key Adaptations:
- Per-user database isolation (each user has their own database)
- Stateless service instances (no global state or singletons)
- Request-scoped processing (instantiated per API request)
- No thread-local storage (incompatible with async API)
- Database-backed persistence instead of in-memory storage

Features:
- Regex and literal pattern matching
- Probability-based replacements
- Token budget management
- Group-based dictionary organization
- Max replacement limits
- Timed effects support (sticky, cooldown, delay)
"""

import json
import logging
import random
import re
import warnings
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Union, Any, Tuple, Set
from pathlib import Path

from loguru import logger

# Local imports
from tldw_Server_API.app.core.DB_Management.ChaChaNotes_DB import (
    CharactersRAGDB, 
    CharactersRAGDBError, 
    InputError, 
    ConflictError
)


class TokenBudgetExceededWarning(Warning):
    """Custom warning for token budget issues"""
    pass


class ChatDictionaryEntry:
    """
    Individual dictionary entry with pattern matching capabilities.
    
    This is a stateless data class that handles pattern compilation and matching.
    No global state or thread-local storage is used.
    """
    
    def __init__(
        self, 
        key: str, 
        content: str, 
        probability: int = 100, 
        group: Optional[str] = None,
        timed_effects: Optional[Dict[str, int]] = None, 
        max_replacements: int = 1,
        entry_id: Optional[int] = None
    ):
        """
        Initialize a dictionary entry.
        
        Args:
            key: Pattern to match (can be regex with /pattern/flags format)
            content: Replacement text
            probability: Chance of replacement (0-100)
            group: Optional group name for organization
            timed_effects: Dictionary with sticky, cooldown, delay settings
            max_replacements: Maximum number of replacements per processing
            entry_id: Database ID (for persistence)
        """
        self.entry_id = entry_id
        self.raw_key = key
        self.content = content
        self.probability = probability
        self.group = group
        self.timed_effects = timed_effects or {"sticky": 0, "cooldown": 0, "delay": 0}
        self.max_replacements = max_replacements
        
        # Pattern compilation
        self.is_regex = False
        self.key_pattern_str = ""
        self.key_flags = 0
        self.key = self._compile_key(key)
        
        # Runtime state (not persisted)
        self.last_triggered: Optional[datetime] = None
        self.trigger_count = 0
    
    def _compile_key(self, key_str: str) -> Union[re.Pattern, str]:
        """
        Compile the key pattern, detecting regex format.
        
        Supports /pattern/flags format for regex patterns.
        """
        self.is_regex = False
        self.key_flags = 0
        pattern_to_compile = key_str
        
        # Check for /pattern/flags format
        if key_str.startswith("/") and len(key_str) > 1:
            last_slash_idx = key_str.rfind("/")
            if last_slash_idx > 0:
                pattern_to_compile = key_str[1:last_slash_idx]
                flag_chars = key_str[last_slash_idx+1:]
                
                # Parse regex flags
                if 'i' in flag_chars: 
                    self.key_flags |= re.IGNORECASE
                if 'm' in flag_chars: 
                    self.key_flags |= re.MULTILINE
                if 's' in flag_chars: 
                    self.key_flags |= re.DOTALL
                if 'x' in flag_chars:
                    self.key_flags |= re.VERBOSE
                    
                self.is_regex = True
            elif key_str.endswith("/") and len(key_str) > 2:
                pattern_to_compile = key_str[1:-1]
                self.is_regex = True
        
        self.key_pattern_str = pattern_to_compile
        
        if self.is_regex:
            try:
                if not pattern_to_compile:
                    logger.warning(f"Empty regex pattern from key '{self.raw_key}'. Treating as literal.")
                    self.is_regex = False
                    return self.raw_key
                return re.compile(pattern_to_compile, self.key_flags)
            except re.error as e:
                logger.warning(
                    f"Invalid regex '{pattern_to_compile}' (from key '{self.raw_key}'): {e}. "
                    f"Treating as literal string."
                )
                self.is_regex = False
                return self.raw_key
        else:
            return key_str
    
    def matches(self, text: str) -> bool:
        """Check if this entry's pattern matches the given text."""
        if self.is_regex and isinstance(self.key, re.Pattern):
            return bool(self.key.search(text))
        elif not self.is_regex and isinstance(self.key, str):
            return self.key in text
        return False
    
    def should_apply(self) -> bool:
        """Check probability and timed effects to determine if replacement should occur."""
        # Check probability
        if self.probability < 100:
            if random.randint(1, 100) > self.probability:
                return False
        
        # Check timed effects
        if self.last_triggered:
            now = datetime.now()
            
            # Cooldown check
            cooldown = self.timed_effects.get('cooldown', 0)
            if cooldown > 0:
                if (now - self.last_triggered) < timedelta(seconds=cooldown):
                    return False
            
            # Delay check (initial delay before first trigger)
            delay = self.timed_effects.get('delay', 0)
            if delay > 0 and self.trigger_count == 0:
                if (now - self.last_triggered) < timedelta(seconds=delay):
                    return False
        
        return True
    
    def apply_replacement(self, text: str) -> Tuple[str, int]:
        """
        Apply the replacement to the text.
        
        Returns:
            Tuple of (modified text, number of replacements made)
        """
        if not self.should_apply():
            return text, 0
        
        replacement_count = 0
        
        if self.is_regex and isinstance(self.key, re.Pattern):
            # For regex patterns, use sub with count limit
            def replacer(match):
                nonlocal replacement_count
                if replacement_count >= self.max_replacements:
                    return match.group(0)
                replacement_count += 1
                return self.content
            
            text = self.key.sub(replacer, text)
        else:
            # For literal patterns, replace up to max_replacements
            parts = text.split(self.key)
            if len(parts) > 1:
                replacements_to_make = min(len(parts) - 1, self.max_replacements)
                replacement_count = replacements_to_make
                
                # Reconstruct with replacements
                result = []
                for i, part in enumerate(parts):
                    result.append(part)
                    if i < len(parts) - 1:
                        if replacements_to_make > 0:
                            result.append(self.content)
                            replacements_to_make -= 1
                        else:
                            result.append(self.key)
                text = ''.join(result)
        
        if replacement_count > 0:
            self.last_triggered = datetime.now()
            self.trigger_count += replacement_count
        
        return text, replacement_count
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for database storage."""
        return {
            'id': self.entry_id,
            'key': self.raw_key,
            'content': self.content,
            'probability': self.probability,
            'group': self.group,
            'timed_effects': json.dumps(self.timed_effects),
            'max_replacements': self.max_replacements,
            'is_regex': self.is_regex
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ChatDictionaryEntry':
        """Create instance from database dictionary."""
        timed_effects = data.get('timed_effects')
        if isinstance(timed_effects, str):
            timed_effects = json.loads(timed_effects)
        
        return cls(
            key=data['key'],
            content=data['content'],
            probability=data.get('probability', 100),
            group=data.get('group'),
            timed_effects=timed_effects,
            max_replacements=data.get('max_replacements', 1),
            entry_id=data.get('id')
        )


class ChatDictionaryService:
    """
    Service class for managing chat dictionaries in a multi-user environment.
    
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
        
        # Request-scoped cache (not shared between requests)
        self._entry_cache: Optional[List[ChatDictionaryEntry]] = None
        self._cache_timestamp: Optional[datetime] = None
        self._cache_ttl = timedelta(seconds=60)  # Cache for 1 minute within request
    
    def _init_tables(self):
        """Initialize dictionary tables in the user's database if they don't exist."""
        try:
            with self.db.get_connection() as conn:
                # Create chat_dictionaries table
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS chat_dictionaries (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        name TEXT NOT NULL UNIQUE,
                        description TEXT,
                        is_active BOOLEAN DEFAULT 1,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        version INTEGER DEFAULT 1,
                        deleted BOOLEAN DEFAULT 0
                    )
                """)
                
                # Create dictionary_entries table
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS dictionary_entries (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        dictionary_id INTEGER NOT NULL,
                        key TEXT NOT NULL,
                        content TEXT NOT NULL,
                        is_regex BOOLEAN DEFAULT 0,
                        probability INTEGER DEFAULT 100,
                        max_replacements INTEGER DEFAULT 1,
                        group_name TEXT,
                        timed_effects TEXT DEFAULT '{"sticky": 0, "cooldown": 0, "delay": 0}',
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        FOREIGN KEY (dictionary_id) REFERENCES chat_dictionaries(id) ON DELETE CASCADE
                    )
                """)
                
                # Create indexes
                conn.execute("CREATE INDEX IF NOT EXISTS idx_dict_entries_dict_id ON dictionary_entries(dictionary_id)")
                conn.execute("CREATE INDEX IF NOT EXISTS idx_dict_entries_group ON dictionary_entries(group_name)")
                
                conn.commit()
                logger.info("Chat dictionary tables initialized")
        except Exception as e:
            logger.error(f"Failed to initialize dictionary tables: {e}")
            raise CharactersRAGDBError(f"Failed to initialize dictionary tables: {e}")
    
    # --- Dictionary CRUD Operations ---
    
    def create_dictionary(self, name: str, description: Optional[str] = None) -> int:
        """
        Create a new chat dictionary for the user.
        
        Args:
            name: Unique name for the dictionary
            description: Optional description
            
        Returns:
            The ID of the created dictionary
            
        Raises:
            ConflictError: If a dictionary with this name already exists
        """
        try:
            with self.db.get_connection() as conn:
                cursor = conn.execute(
                    """
                    INSERT INTO chat_dictionaries (name, description)
                    VALUES (?, ?)
                    """,
                    (name, description)
                )
                dictionary_id = cursor.lastrowid
                conn.commit()
                
                logger.info(f"Created dictionary '{name}' with ID {dictionary_id}")
                self._invalidate_cache()
                return dictionary_id
                
        except sqlite3.IntegrityError as e:
            if "UNIQUE constraint failed" in str(e):
                raise ConflictError(f"Dictionary '{name}' already exists", "chat_dictionaries", name)
            raise CharactersRAGDBError(f"Database error creating dictionary: {e}")
    
    def get_dictionary(self, dictionary_id: Optional[int] = None, name: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """
        Get a dictionary by ID or name.
        
        Args:
            dictionary_id: Optional dictionary ID
            name: Optional dictionary name
            
        Returns:
            Dictionary data or None if not found
        """
        try:
            with self.db.get_connection() as conn:
                if dictionary_id:
                    cursor = conn.execute(
                        "SELECT * FROM chat_dictionaries WHERE id = ? AND deleted = 0",
                        (dictionary_id,)
                    )
                elif name:
                    cursor = conn.execute(
                        "SELECT * FROM chat_dictionaries WHERE name = ? AND deleted = 0",
                        (name,)
                    )
                else:
                    return None
                
                row = cursor.fetchone()
                if row:
                    return dict(row)
                return None
                
        except Exception as e:
            logger.error(f"Error fetching dictionary: {e}")
            raise CharactersRAGDBError(f"Error fetching dictionary: {e}")
    
    def list_dictionaries(self, include_inactive: bool = False) -> List[Dict[str, Any]]:
        """
        List all dictionaries for the user.
        
        Args:
            include_inactive: Whether to include inactive dictionaries
            
        Returns:
            List of dictionary data
        """
        try:
            with self.db.get_connection() as conn:
                query = "SELECT * FROM chat_dictionaries WHERE deleted = 0"
                if not include_inactive:
                    query += " AND is_active = 1"
                query += " ORDER BY name"
                
                cursor = conn.execute(query)
                return [dict(row) for row in cursor.fetchall()]
                
        except Exception as e:
            logger.error(f"Error listing dictionaries: {e}")
            raise CharactersRAGDBError(f"Error listing dictionaries: {e}")
    
    def update_dictionary(
        self, 
        dictionary_id: int, 
        name: Optional[str] = None,
        description: Optional[str] = None,
        is_active: Optional[bool] = None
    ) -> bool:
        """
        Update a dictionary's metadata.
        
        Args:
            dictionary_id: Dictionary ID
            name: New name (optional)
            description: New description (optional)
            is_active: Active status (optional)
            
        Returns:
            True if updated successfully
            
        Raises:
            ConflictError: If the new name conflicts with an existing dictionary
        """
        try:
            updates = []
            params = []
            
            if name is not None:
                updates.append("name = ?")
                params.append(name)
            if description is not None:
                updates.append("description = ?")
                params.append(description)
            if is_active is not None:
                updates.append("is_active = ?")
                params.append(int(is_active))
            
            if not updates:
                return True
            
            updates.append("updated_at = CURRENT_TIMESTAMP")
            updates.append("version = version + 1")
            params.append(dictionary_id)
            
            with self.db.get_connection() as conn:
                cursor = conn.execute(
                    f"UPDATE chat_dictionaries SET {', '.join(updates)} WHERE id = ? AND deleted = 0",
                    params
                )
                conn.commit()
                
                if cursor.rowcount > 0:
                    logger.info(f"Updated dictionary {dictionary_id}")
                    self._invalidate_cache()
                    return True
                return False
                
        except sqlite3.IntegrityError as e:
            if "UNIQUE constraint failed" in str(e):
                raise ConflictError(f"Dictionary name '{name}' already exists", "chat_dictionaries", name)
            raise CharactersRAGDBError(f"Database error updating dictionary: {e}")
    
    def delete_dictionary(self, dictionary_id: int, hard_delete: bool = False) -> bool:
        """
        Delete a dictionary (soft delete by default).
        
        Args:
            dictionary_id: Dictionary ID
            hard_delete: If True, permanently delete; otherwise soft delete
            
        Returns:
            True if deleted successfully
        """
        try:
            with self.db.get_connection() as conn:
                if hard_delete:
                    cursor = conn.execute(
                        "DELETE FROM chat_dictionaries WHERE id = ?",
                        (dictionary_id,)
                    )
                else:
                    cursor = conn.execute(
                        "UPDATE chat_dictionaries SET deleted = 1, updated_at = CURRENT_TIMESTAMP WHERE id = ?",
                        (dictionary_id,)
                    )
                conn.commit()
                
                if cursor.rowcount > 0:
                    logger.info(f"{'Hard' if hard_delete else 'Soft'} deleted dictionary {dictionary_id}")
                    self._invalidate_cache()
                    return True
                return False
                
        except Exception as e:
            logger.error(f"Error deleting dictionary: {e}")
            raise CharactersRAGDBError(f"Error deleting dictionary: {e}")
    
    # --- Entry CRUD Operations ---
    
    def add_entry(
        self, 
        dictionary_id: int,
        key: str,
        content: str,
        probability: int = 100,
        group: Optional[str] = None,
        timed_effects: Optional[Dict[str, int]] = None,
        max_replacements: int = 1
    ) -> int:
        """
        Add an entry to a dictionary.
        
        Args:
            dictionary_id: Dictionary ID
            key: Pattern to match
            content: Replacement text
            probability: Chance of replacement (0-100)
            group: Optional group name
            timed_effects: Timing effects dictionary
            max_replacements: Max replacements per processing
            
        Returns:
            The ID of the created entry
        """
        try:
            # Validate probability
            if not 0 <= probability <= 100:
                raise InputError("Probability must be between 0 and 100")
            
            # Compile pattern to check validity
            entry = ChatDictionaryEntry(key, content, probability, group, timed_effects, max_replacements)
            
            timed_effects_json = json.dumps(timed_effects or {"sticky": 0, "cooldown": 0, "delay": 0})
            
            with self.db.get_connection() as conn:
                cursor = conn.execute(
                    """
                    INSERT INTO dictionary_entries 
                    (dictionary_id, key, content, is_regex, probability, max_replacements, group_name, timed_effects)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (dictionary_id, key, content, entry.is_regex, probability, max_replacements, group, timed_effects_json)
                )
                entry_id = cursor.lastrowid
                conn.commit()
                
                logger.info(f"Added entry {entry_id} to dictionary {dictionary_id}")
                self._invalidate_cache()
                return entry_id
                
        except Exception as e:
            logger.error(f"Error adding dictionary entry: {e}")
            raise CharactersRAGDBError(f"Error adding dictionary entry: {e}")
    
    def get_entries(
        self, 
        dictionary_id: Optional[int] = None,
        group: Optional[str] = None,
        active_only: bool = True
    ) -> List[ChatDictionaryEntry]:
        """
        Get dictionary entries.
        
        Args:
            dictionary_id: Optional specific dictionary ID
            group: Optional group filter
            active_only: Only return entries from active dictionaries
            
        Returns:
            List of ChatDictionaryEntry instances
        """
        # Check cache first
        if self._entry_cache is not None and self._cache_timestamp:
            if datetime.now() - self._cache_timestamp < self._cache_ttl:
                # Filter cached entries
                entries = self._entry_cache
                if dictionary_id:
                    entries = [e for e in entries if self._get_entry_dict_id(e.entry_id) == dictionary_id]
                if group:
                    entries = [e for e in entries if e.group == group]
                return entries
        
        try:
            with self.db.get_connection() as conn:
                query = """
                    SELECT e.*, d.is_active 
                    FROM dictionary_entries e
                    JOIN chat_dictionaries d ON e.dictionary_id = d.id
                    WHERE d.deleted = 0
                """
                params = []
                
                if active_only:
                    query += " AND d.is_active = 1"
                if dictionary_id:
                    query += " AND e.dictionary_id = ?"
                    params.append(dictionary_id)
                if group:
                    query += " AND e.group_name = ?"
                    params.append(group)
                
                query += " ORDER BY e.id"
                
                cursor = conn.execute(query, params)
                entries = []
                for row in cursor.fetchall():
                    entry_data = dict(row)
                    entries.append(ChatDictionaryEntry.from_dict(entry_data))
                
                # Update cache if fetching all active entries
                if not dictionary_id and not group and active_only:
                    self._entry_cache = entries
                    self._cache_timestamp = datetime.now()
                
                return entries
                
        except Exception as e:
            logger.error(f"Error fetching dictionary entries: {e}")
            raise CharactersRAGDBError(f"Error fetching dictionary entries: {e}")
    
    def update_entry(
        self,
        entry_id: int,
        key: Optional[str] = None,
        content: Optional[str] = None,
        probability: Optional[int] = None,
        group: Optional[str] = None,
        timed_effects: Optional[Dict[str, int]] = None,
        max_replacements: Optional[int] = None
    ) -> bool:
        """
        Update a dictionary entry.
        
        Args:
            entry_id: Entry ID
            Various optional fields to update
            
        Returns:
            True if updated successfully
        """
        try:
            updates = []
            params = []
            
            if key is not None:
                # Validate pattern
                entry = ChatDictionaryEntry(key, "test")
                updates.append("key = ?")
                updates.append("is_regex = ?")
                params.extend([key, entry.is_regex])
                
            if content is not None:
                updates.append("content = ?")
                params.append(content)
                
            if probability is not None:
                if not 0 <= probability <= 100:
                    raise InputError("Probability must be between 0 and 100")
                updates.append("probability = ?")
                params.append(probability)
                
            if group is not None:
                updates.append("group_name = ?")
                params.append(group)
                
            if timed_effects is not None:
                updates.append("timed_effects = ?")
                params.append(json.dumps(timed_effects))
                
            if max_replacements is not None:
                updates.append("max_replacements = ?")
                params.append(max_replacements)
            
            if not updates:
                return True
            
            updates.append("updated_at = CURRENT_TIMESTAMP")
            params.append(entry_id)
            
            with self.db.get_connection() as conn:
                cursor = conn.execute(
                    f"UPDATE dictionary_entries SET {', '.join(updates)} WHERE id = ?",
                    params
                )
                conn.commit()
                
                if cursor.rowcount > 0:
                    logger.info(f"Updated dictionary entry {entry_id}")
                    self._invalidate_cache()
                    return True
                return False
                
        except Exception as e:
            logger.error(f"Error updating dictionary entry: {e}")
            raise CharactersRAGDBError(f"Error updating dictionary entry: {e}")
    
    def delete_entry(self, entry_id: int) -> bool:
        """
        Delete a dictionary entry.
        
        Args:
            entry_id: Entry ID
            
        Returns:
            True if deleted successfully
        """
        try:
            with self.db.get_connection() as conn:
                cursor = conn.execute(
                    "DELETE FROM dictionary_entries WHERE id = ?",
                    (entry_id,)
                )
                conn.commit()
                
                if cursor.rowcount > 0:
                    logger.info(f"Deleted dictionary entry {entry_id}")
                    self._invalidate_cache()
                    return True
                return False
                
        except Exception as e:
            logger.error(f"Error deleting dictionary entry: {e}")
            raise CharactersRAGDBError(f"Error deleting dictionary entry: {e}")
    
    # --- Text Processing ---
    
    def process_text(
        self,
        text: str,
        dictionary_id: Optional[int] = None,
        group: Optional[str] = None,
        max_iterations: int = 5,
        token_budget: Optional[int] = None
    ) -> Tuple[str, Dict[str, Any]]:
        """
        Process text through dictionary replacements.
        
        Args:
            text: Input text
            dictionary_id: Optional specific dictionary to use
            group: Optional group filter
            max_iterations: Maximum processing iterations
            token_budget: Optional token limit
            
        Returns:
            Tuple of (processed text, statistics dictionary)
        """
        # Get applicable entries
        entries = self.get_entries(dictionary_id, group, active_only=True)
        
        if not entries:
            return text, {"replacements": 0, "iterations": 0, "entries_used": []}
        
        stats = {
            "replacements": 0,
            "iterations": 0,
            "entries_used": [],
            "token_budget_exceeded": False
        }
        
        original_text = text
        
        for iteration in range(max_iterations):
            iteration_replacements = 0
            
            for entry in entries:
                if entry.matches(text):
                    new_text, count = entry.apply_replacement(text)
                    if count > 0:
                        text = new_text
                        iteration_replacements += count
                        stats["replacements"] += count
                        
                        if entry.entry_id not in stats["entries_used"]:
                            stats["entries_used"].append(entry.entry_id)
                        
                        # Check token budget
                        if token_budget and self._estimate_tokens(text) > token_budget:
                            warnings.warn(
                                f"Token budget ({token_budget}) exceeded after {stats['replacements']} replacements",
                                TokenBudgetExceededWarning
                            )
                            stats["token_budget_exceeded"] = True
                            return text, stats
            
            stats["iterations"] += 1
            
            # Stop if no replacements were made in this iteration
            if iteration_replacements == 0:
                break
        
        return text, stats
    
    def import_from_markdown(self, file_path: Union[str, Path], dictionary_name: str) -> int:
        """
        Import dictionary entries from a markdown file.
        
        File format:
        ```
        key: value
        /regex/: replacement
        /pattern/i: case-insensitive replacement
        
        ## Group Name
        grouped_key: grouped_value
        ```
        
        Args:
            file_path: Path to markdown file
            dictionary_name: Name for the new dictionary
            
        Returns:
            Dictionary ID
        """
        file_path = Path(file_path)
        if not file_path.exists():
            raise InputError(f"File not found: {file_path}")
        
        # Create dictionary
        dict_id = self.create_dictionary(
            dictionary_name,
            f"Imported from {file_path.name}"
        )
        
        current_group = None
        entries_added = 0
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    
                    # Skip empty lines and comments
                    if not line or line.startswith('#'):
                        # Check for group headers (## Group Name)
                        if line.startswith('## '):
                            current_group = line[3:].strip()
                        continue
                    
                    # Parse key: value format
                    if ':' in line:
                        key, value = line.split(':', 1)
                        key = key.strip()
                        value = value.strip()
                        
                        if key and value:
                            self.add_entry(
                                dict_id,
                                key,
                                value,
                                group=current_group
                            )
                            entries_added += 1
            
            logger.info(f"Imported {entries_added} entries into dictionary '{dictionary_name}'")
            return dict_id
            
        except Exception as e:
            # Rollback by deleting the dictionary
            self.delete_dictionary(dict_id, hard_delete=True)
            raise CharactersRAGDBError(f"Failed to import dictionary: {e}")
    
    def export_to_markdown(self, dictionary_id: int, file_path: Union[str, Path]) -> bool:
        """
        Export a dictionary to markdown format.
        
        Args:
            dictionary_id: Dictionary to export
            file_path: Output file path
            
        Returns:
            True if exported successfully
        """
        file_path = Path(file_path)
        
        # Get dictionary info
        dict_info = self.get_dictionary(dictionary_id)
        if not dict_info:
            raise InputError(f"Dictionary {dictionary_id} not found")
        
        # Get entries
        entries = self.get_entries(dictionary_id)
        
        try:
            with open(file_path, 'w', encoding='utf-8') as f:
                # Write header
                f.write(f"# {dict_info['name']}\n\n")
                if dict_info.get('description'):
                    f.write(f"{dict_info['description']}\n\n")
                
                # Group entries
                grouped = {}
                ungrouped = []
                
                for entry in entries:
                    if entry.group:
                        if entry.group not in grouped:
                            grouped[entry.group] = []
                        grouped[entry.group].append(entry)
                    else:
                        ungrouped.append(entry)
                
                # Write ungrouped entries
                if ungrouped:
                    for entry in ungrouped:
                        f.write(f"{entry.raw_key}: {entry.content}\n")
                    f.write("\n")
                
                # Write grouped entries
                for group_name, group_entries in grouped.items():
                    f.write(f"## {group_name}\n\n")
                    for entry in group_entries:
                        f.write(f"{entry.raw_key}: {entry.content}\n")
                    f.write("\n")
            
            logger.info(f"Exported dictionary to {file_path}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to export dictionary: {e}")
            raise CharactersRAGDBError(f"Failed to export dictionary: {e}")
    
    # --- Helper Methods ---
    
    def _invalidate_cache(self):
        """Invalidate the entry cache."""
        self._entry_cache = None
        self._cache_timestamp = None
    
    def _get_entry_dict_id(self, entry_id: int) -> Optional[int]:
        """Get the dictionary ID for an entry."""
        try:
            with self.db.get_connection() as conn:
                cursor = conn.execute(
                    "SELECT dictionary_id FROM dictionary_entries WHERE id = ?",
                    (entry_id,)
                )
                row = cursor.fetchone()
                return row['dictionary_id'] if row else None
        except Exception:
            return None
    
    def _estimate_tokens(self, text: str) -> int:
        """
        Estimate token count for text.
        
        Simple approximation: 1 token ≈ 4 characters
        """
        return len(text) // 4
    
    def toggle_dictionary_active(self, dictionary_id: int, is_active: bool) -> bool:
        """
        Toggle dictionary active status.
        
        Args:
            dictionary_id: Dictionary ID
            is_active: New active status
            
        Returns:
            True if updated successfully
        """
        return self.update_dictionary(dictionary_id, is_active=is_active)
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        Get dictionary usage statistics.
        
        Returns:
            Dictionary containing statistics
        """
        try:
            with self.db.get_connection() as conn:
                # Get dictionary counts
                cursor = conn.execute("""
                    SELECT 
                        COUNT(*) as total_dictionaries,
                        SUM(CASE WHEN is_active = 1 THEN 1 ELSE 0 END) as active_dictionaries
                    FROM chat_dictionaries 
                    WHERE deleted = 0
                """)
                dict_stats = dict(cursor.fetchone())
                
                # Get entry counts
                cursor = conn.execute("""
                    SELECT COUNT(*) as total_entries
                    FROM dictionary_entries e
                    JOIN chat_dictionaries d ON e.dictionary_id = d.id
                    WHERE d.deleted = 0
                """)
                entry_stats = dict(cursor.fetchone())
                
                # Calculate average entries per dictionary
                avg_entries = 0
                if dict_stats['total_dictionaries'] > 0:
                    avg_entries = entry_stats['total_entries'] / dict_stats['total_dictionaries']
                
                return {
                    "total_dictionaries": dict_stats['total_dictionaries'],
                    "active_dictionaries": dict_stats['active_dictionaries'],
                    "total_entries": entry_stats['total_entries'],
                    "average_entries_per_dictionary": avg_entries
                }
                
        except Exception as e:
            logger.error(f"Error getting statistics: {e}")
            raise CharactersRAGDBError(f"Error getting statistics: {e}")
    
    def bulk_add_entries(
        self,
        dictionary_id: int,
        entries: List[Dict[str, Any]]
    ) -> int:
        """
        Add multiple entries at once.
        
        Args:
            dictionary_id: Dictionary ID
            entries: List of entry dictionaries with keys: key, content, probability, group, etc.
            
        Returns:
            Number of entries added
        """
        try:
            added_count = 0
            with self.db.get_connection() as conn:
                for entry in entries:
                    # Validate and prepare each entry
                    key = entry.get('key', '')
                    content = entry.get('content', '')
                    probability = entry.get('probability', 100)
                    group = entry.get('group')
                    timed_effects = entry.get('timed_effects', {"sticky": 0, "cooldown": 0, "delay": 0})
                    max_replacements = entry.get('max_replacements', 1)
                    
                    if not key or not content:
                        continue
                    
                    # Validate probability
                    if not 0 <= probability <= 100:
                        probability = 100
                    
                    # Compile pattern to check validity and get is_regex flag
                    test_entry = ChatDictionaryEntry(key, content)
                    
                    timed_effects_json = json.dumps(timed_effects)
                    
                    cursor = conn.execute(
                        """
                        INSERT INTO dictionary_entries 
                        (dictionary_id, key, content, is_regex, probability, max_replacements, group_name, timed_effects)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                        """,
                        (dictionary_id, key, content, test_entry.is_regex, probability, max_replacements, group, timed_effects_json)
                    )
                    added_count += 1
                
                conn.commit()
                
            logger.info(f"Added {added_count} entries to dictionary {dictionary_id}")
            self._invalidate_cache()
            return added_count
            
        except Exception as e:
            logger.error(f"Error adding bulk entries: {e}")
            raise CharactersRAGDBError(f"Error adding bulk entries: {e}")
    
    def search_entries(self, search_term: str) -> List[Dict[str, Any]]:
        """
        Search for entries by pattern.
        
        Args:
            search_term: Search term to look for in keys and content
            
        Returns:
            List of matching entries with dictionary info
        """
        try:
            with self.db.get_connection() as conn:
                cursor = conn.execute("""
                    SELECT 
                        e.id,
                        e.key,
                        e.content,
                        e.probability,
                        e.group_name,
                        d.name as dictionary_name,
                        d.id as dictionary_id
                    FROM dictionary_entries e
                    JOIN chat_dictionaries d ON e.dictionary_id = d.id
                    WHERE d.deleted = 0
                    AND (e.key LIKE ? OR e.content LIKE ?)
                    ORDER BY d.name, e.key
                """, (f'%{search_term}%', f'%{search_term}%'))
                
                results = []
                for row in cursor.fetchall():
                    results.append(dict(row))
                
                return results
                
        except Exception as e:
            logger.error(f"Error searching entries: {e}")
            raise CharactersRAGDBError(f"Error searching entries: {e}")
    
    def clone_dictionary(self, source_dict_id: int, new_name: str) -> int:
        """
        Create a copy of a dictionary with all its entries.
        
        Args:
            source_dict_id: Source dictionary ID
            new_name: Name for the cloned dictionary
            
        Returns:
            ID of the new dictionary
        """
        try:
            # Get source dictionary
            source_dict = self.get_dictionary(source_dict_id)
            if not source_dict:
                raise InputError(f"Source dictionary {source_dict_id} not found")
            
            # Create new dictionary
            new_dict_id = self.create_dictionary(
                name=new_name,
                description=f"Cloned from {source_dict['name']}"
            )
            
            # Get all entries from source dictionary
            entries = self.get_entries(source_dict_id)
            
            # Add entries to new dictionary
            if entries:
                with self.db.get_connection() as conn:
                    for entry in entries:
                        timed_effects_json = json.dumps(entry.timed_effects)
                        
                        conn.execute(
                            """
                            INSERT INTO dictionary_entries 
                            (dictionary_id, key, content, is_regex, probability, max_replacements, group_name, timed_effects)
                            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                            """,
                            (new_dict_id, entry.raw_key, entry.content, entry.is_regex, 
                             entry.probability, entry.max_replacements, entry.group, timed_effects_json)
                        )
                    conn.commit()
            
            logger.info(f"Cloned dictionary {source_dict_id} to new dictionary {new_dict_id} with {len(entries)} entries")
            self._invalidate_cache()
            return new_dict_id
            
        except ConflictError:
            raise
        except Exception as e:
            logger.error(f"Error cloning dictionary: {e}")
            raise CharactersRAGDBError(f"Error cloning dictionary: {e}")


# Import handling to prevent breaking changes
import sqlite3
__all__ = [
    'ChatDictionaryEntry',
    'ChatDictionaryService',
    'TokenBudgetExceededWarning'
]