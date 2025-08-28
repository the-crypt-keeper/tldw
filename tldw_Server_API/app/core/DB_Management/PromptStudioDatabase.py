# PromptStudioDatabase.py
# Database management for Prompt Studio feature
# Extends PromptsDatabase to add Prompt Studio specific functionality

import json
import sqlite3
import threading
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Tuple, Dict, Any, Optional, Union

from loguru import logger

# Local imports
from .Prompts_DB import PromptsDatabase, DatabaseError, SchemaError, InputError, ConflictError

########################################################################################################################
# Prompt Studio Database Class

class PromptStudioDatabase(PromptsDatabase):
    """
    Extends PromptsDatabase with Prompt Studio specific functionality.
    Manages projects, signatures, test cases, evaluations, and optimizations.
    """
    
    _PROMPT_STUDIO_SCHEMA_VERSION = 1
    
    def __init__(self, db_path: Union[str, Path], client_id: str):
        """
        Initialize PromptStudioDatabase with path and client ID.
        
        Args:
            db_path: Path to the database file
            client_id: Client identifier for sync logging
        """
        # Initialize parent class
        super().__init__(db_path, client_id)
        
        # Create a write lock for serializing write operations
        self._write_lock = threading.RLock()
        
        # Initialize prompt studio schema
        self._init_prompt_studio_schema()
        
        # Set pragmas for better reliability
        try:
            conn = self.get_connection()
            cursor = conn.cursor()
            # Don't use WAL mode as it can cause corruption in some cases
            cursor.execute("PRAGMA busy_timeout=30000")  # 30 second timeout for locked database
            conn.commit()
        except Exception as e:
            logger.debug(f"Could not set pragmas: {e}")
        
        logger.info(f"PromptStudioDatabase initialized for {db_path} with client {client_id}")
    
    def _init_prompt_studio_schema(self):
        """Initialize Prompt Studio specific schema."""
        try:
            conn = self.get_connection()
            cursor = conn.cursor()
            
            # Check if prompt studio tables exist
            cursor.execute("""
                SELECT name FROM sqlite_master 
                WHERE type='table' AND name='prompt_studio_projects'
            """)
            
            if not cursor.fetchone():
                logger.info("Initializing Prompt Studio schema...")
                self._apply_prompt_studio_migrations(conn)
                
        except Exception as e:
            logger.error(f"Error initializing Prompt Studio schema: {e}")
            raise SchemaError(f"Failed to initialize Prompt Studio schema: {e}")
    
    def _apply_prompt_studio_migrations(self, conn: sqlite3.Connection):
        """Apply Prompt Studio migration scripts."""
        migrations_dir = Path(__file__).parent / "migrations"
        
        # List of migration files in order
        migration_files = [
            "001_prompt_studio_schema.sql",
            "002_prompt_studio_indexes.sql",
            "003_prompt_studio_triggers.sql",
            "004_prompt_studio_fts.sql"
        ]
        
        for migration_file in migration_files:
            migration_path = migrations_dir / migration_file
            if migration_path.exists():
                logger.info(f"Applying migration: {migration_file}")
                with open(migration_path, 'r') as f:
                    migration_sql = f.read()
                    
                # Execute migration statements
                try:
                    conn.executescript(migration_sql)
                    conn.commit()
                    logger.info(f"Successfully applied {migration_file}")
                except Exception as e:
                    logger.error(f"Failed to apply {migration_file}: {e}")
                    raise SchemaError(f"Migration {migration_file} failed: {e}")
            else:
                logger.warning(f"Migration file not found: {migration_path}")
    
    ####################################################################################################################
    # Project Management
    
    def create_project(self, name: str, description: Optional[str] = None, 
                      status: str = "draft", metadata: Optional[Dict] = None,
                      user_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Create a new prompt studio project.
        
        Args:
            name: Project name
            description: Project description
            status: Project status (draft, active, archived)
            metadata: Additional metadata
            
        Returns:
            Created project record
        """
        import time
        import sqlite3
        import random
        
        project_id = None
        # Get connection before acquiring lock to avoid deadlock
        conn = self.get_connection()
        
        max_retries = 5
        base_delay = 0.1  # 100ms
        
        for attempt in range(max_retries):
            should_retry = False
            retry_delay = 0
            
            # Use write lock to serialize write operations
            with self._write_lock:
                try:
                    cursor = conn.cursor()
                    
                    # Generate UUID
                    project_uuid = str(uuid.uuid4())
                    
                    # Insert project
                    cursor.execute("""
                        INSERT INTO prompt_studio_projects 
                        (uuid, name, description, user_id, client_id, status, metadata)
                        VALUES (?, ?, ?, ?, ?, ?, ?)
                    """, (project_uuid, name, description, user_id or self.client_id, self.client_id, 
                          status, json.dumps(metadata) if metadata else None))
                    
                    project_id = cursor.lastrowid
                    conn.commit()
                    
                    # Log to sync_log
                    self._log_sync_event("prompt_studio_project", project_uuid, "create", {
                        "name": name,
                        "description": description,
                        "status": status
                    })
                    
                    logger.info(f"Created project: {name} (ID: {project_id})")
                    break  # Success, exit retry loop
                    
                except sqlite3.OperationalError as e:
                    if "database is locked" in str(e) and attempt < max_retries - 1:
                        # Database locked, will retry
                        should_retry = True
                        retry_delay = base_delay * (2 ** attempt) * (0.5 + random.random())
                    else:
                        raise DatabaseError(f"Failed to create project: {e}")
                except sqlite3.IntegrityError as e:
                    if "UNIQUE" in str(e):
                        raise ConflictError(f"Project with name '{name}' already exists for this user")
                    raise DatabaseError(f"Failed to create project: {e}")
                except Exception as e:
                    raise DatabaseError(f"Failed to create project: {e}")
            
            # Sleep outside the lock if we need to retry
            if should_retry:
                time.sleep(retry_delay)
        
        # Get the project after releasing the lock
        return self.get_project(project_id)
    
    def get_project(self, project_id: int, include_deleted: bool = False) -> Optional[Dict[str, Any]]:
        """
        Get a project by ID.
        
        Args:
            project_id: Project ID
            include_deleted: Include soft-deleted projects
            
        Returns:
            Project record or None
        """
        conn = self.get_connection()
        cursor = conn.cursor()
        
        query = """
            SELECT 
                id, uuid, name, description, user_id, client_id, status,
                deleted, deleted_at, created_at, updated_at, last_modified,
                version, metadata
            FROM prompt_studio_projects
            WHERE id = ?
        """
        
        if not include_deleted:
            query += " AND deleted = 0"
        
        cursor.execute(query, (project_id,))
        row = cursor.fetchone()
        
        if row:
            return self._row_to_dict(cursor, row)
        return None
    
    def list_projects(self, user_id: Optional[str] = None, status: Optional[str] = None,
                     include_deleted: bool = False, page: int = 1, per_page: int = 20) -> Dict[str, Any]:
        """
        List projects with optional filtering.
        
        Args:
            user_id: Filter by user ID
            status: Filter by status
            include_deleted: Include soft-deleted projects
            page: Page number
            per_page: Items per page
            
        Returns:
            Dictionary with projects list and pagination metadata
        """
        conn = self.get_connection()
        cursor = conn.cursor()
        
        # Build query
        conditions = []
        params = []
        
        if not include_deleted:
            conditions.append("deleted = 0")
        
        if user_id:
            conditions.append("user_id = ?")
            params.append(user_id)
        
        if status:
            conditions.append("status = ?")
            params.append(status)
        
        where_clause = " WHERE " + " AND ".join(conditions) if conditions else ""
        
        # Count total
        count_query = f"SELECT COUNT(*) FROM prompt_studio_projects{where_clause}"
        cursor.execute(count_query, params)
        total = cursor.fetchone()[0]
        
        # Get projects with pagination
        offset = (page - 1) * per_page
        query = f"""
            SELECT 
                p.*,
                (SELECT COUNT(*) FROM prompt_studio_prompts WHERE project_id = p.id AND deleted = 0) as prompt_count,
                (SELECT COUNT(*) FROM prompt_studio_test_cases WHERE project_id = p.id AND deleted = 0) as test_case_count
            FROM prompt_studio_projects p
            {where_clause}
            ORDER BY p.updated_at DESC
            LIMIT ? OFFSET ?
        """
        params.extend([per_page, offset])
        
        cursor.execute(query, params)
        projects = [self._row_to_dict(cursor, row) for row in cursor.fetchall()]
        
        return {
            "projects": projects,
            "pagination": {
                "page": page,
                "per_page": per_page,
                "total": total,
                "total_pages": (total + per_page - 1) // per_page
            }
        }
    
    def update_project(self, project_id: int, updates: Dict[str, Any]) -> Dict[str, Any]:
        """
        Update a project.
        
        Args:
            project_id: Project ID
            updates: Fields to update
            
        Returns:
            Updated project record
        """
        import time
        import sqlite3
        import random
        
        project_uuid = None
        # Get connection before acquiring lock to avoid deadlock
        conn = self.get_connection()
        
        # Use write lock to serialize write operations
        with self._write_lock:
            try:
                cursor = conn.cursor()
                
                # Build update query
                allowed_fields = ["name", "description", "status", "metadata"]
                set_clauses = []
                params = []
                
                for field in allowed_fields:
                    if field in updates:
                        set_clauses.append(f"{field} = ?")
                        value = updates[field]
                        if field == "metadata" and value is not None:
                            value = json.dumps(value)
                        params.append(value)
                
                if not set_clauses:
                    return self.get_project(project_id)
                
                # Add updated_at
                set_clauses.append("updated_at = CURRENT_TIMESTAMP")
                params.append(project_id)
                
                query = f"""
                    UPDATE prompt_studio_projects
                    SET {', '.join(set_clauses)}
                    WHERE id = ? AND deleted = 0
                """
                
                cursor.execute(query, params)
                
                if cursor.rowcount == 0:
                    raise InputError(f"Project {project_id} not found or already deleted")
                
                # Get UUID for logging before committing
                cursor.execute("SELECT uuid FROM prompt_studio_projects WHERE id = ?", (project_id,))
                row = cursor.fetchone()
                if row:
                    project_uuid = row[0]
                
                conn.commit()
                
                # Log sync event with UUID we captured
                if project_uuid:
                    self._log_sync_event("prompt_studio_project", project_uuid, "update", updates)
                
            except sqlite3.IntegrityError as e:
                if "UNIQUE" in str(e):
                    raise ConflictError(f"Project with name already exists")
                raise DatabaseError(f"Failed to update project: {e}")
            except Exception as e:
                raise DatabaseError(f"Failed to update project: {e}")
        
        # Get the updated project after releasing the lock  
        return self.get_project(project_id)
    
    def delete_project(self, project_id: int, hard_delete: bool = False) -> bool:
        """
        Delete a project (soft delete by default).
        
        Args:
            project_id: Project ID
            hard_delete: Permanently delete if True
            
        Returns:
            True if deleted
        """
        conn = self.get_connection()
        cursor = conn.cursor()
        
        if hard_delete:
            # Cascade delete all related data
            cursor.execute("DELETE FROM prompt_studio_projects WHERE id = ?", (project_id,))
        else:
            # Soft delete
            cursor.execute("""
                UPDATE prompt_studio_projects
                SET deleted = 1, deleted_at = CURRENT_TIMESTAMP
                WHERE id = ? AND deleted = 0
            """, (project_id,))
        
        success = cursor.rowcount > 0
        if success:
            conn.commit()
            logger.info(f"{'Hard' if hard_delete else 'Soft'} deleted project {project_id}")
        
        return success
    
    ####################################################################################################################
    # Helper Methods
    
    def _row_to_dict(self, cursor: sqlite3.Cursor, row: tuple) -> Dict[str, Any]:
        """Convert a database row to dictionary."""
        if not row:
            return None
        
        columns = [description[0] for description in cursor.description]
        result = dict(zip(columns, row))
        
        # Parse JSON fields
        json_fields = ["metadata", "input_schema", "output_schema", "constraints", 
                      "validation_rules", "few_shot_examples", "modules_config",
                      "model_params", "inputs", "outputs", "expected_outputs",
                      "actual_outputs", "scores", "test_case_ids", "test_run_ids",
                      "aggregate_metrics", "model_configs", "payload", "result",
                      "initial_metrics", "final_metrics", "optimization_config"]
        
        for field in json_fields:
            if field in result and result[field]:
                try:
                    result[field] = json.loads(result[field])
                except (json.JSONDecodeError, TypeError):
                    pass
        
        # Parse datetime fields
        datetime_fields = ["created_at", "updated_at", "deleted_at", "last_modified",
                          "started_at", "completed_at"]
        
        for field in datetime_fields:
            if field in result and result[field]:
                try:
                    if isinstance(result[field], str):
                        result[field] = datetime.fromisoformat(result[field])
                except (ValueError, TypeError):
                    pass
        
        return result
    
    def _log_sync_event(self, entity: str, entity_uuid: str, operation: str, payload: Dict[str, Any]):
        """Log an event to sync_log table if it exists."""
        try:
            conn = self.get_connection()
            cursor = conn.cursor()
            
            # Check if sync_log table exists
            cursor.execute("""
                SELECT name FROM sqlite_master 
                WHERE type='table' AND name='sync_log'
            """)
            
            if cursor.fetchone():
                cursor.execute("""
                    INSERT INTO sync_log (entity, entity_uuid, operation, client_id, version, payload)
                    VALUES (?, ?, ?, ?, 1, ?)
                """, (entity, entity_uuid, operation, self.client_id, json.dumps(payload)))
                conn.commit()
        except Exception as e:
            logger.debug(f"Could not log sync event: {e}")
    
    ####################################################################################################################
    # Test Case Methods
    
    def get_golden_test_cases(self, project_id: int, limit: int = 100, offset: int = 0) -> List[Dict[str, Any]]:
        """
        Get golden test cases for a project.
        
        Args:
            project_id: Project ID
            limit: Maximum number of results
            offset: Number of results to skip
            
        Returns:
            List of golden test case records
        """
        try:
            conn = self.get_connection()
            cursor = conn.cursor()
            
            cursor.execute("""
                SELECT 
                    id, uuid, project_id, signature_id, name, description,
                    inputs, expected_outputs, actual_outputs, tags,
                    is_golden, is_generated, client_id, deleted,
                    created_at, updated_at
                FROM prompt_studio_test_cases
                WHERE project_id = ? AND is_golden = 1 AND deleted = 0
                ORDER BY created_at DESC
                LIMIT ? OFFSET ?
            """, (project_id, limit, offset))
            
            test_cases = []
            for row in cursor.fetchall():
                test_case = self.row_to_dict(row, cursor)
                # Parse JSON fields
                if test_case.get('inputs'):
                    test_case['inputs'] = json.loads(test_case['inputs'])
                if test_case.get('expected_outputs'):
                    test_case['expected_outputs'] = json.loads(test_case['expected_outputs'])
                if test_case.get('actual_outputs'):
                    test_case['actual_outputs'] = json.loads(test_case['actual_outputs'])
                test_cases.append(test_case)
            
            return test_cases
            
        except Exception as e:
            logger.error(f"Failed to get golden test cases: {e}")
            raise DatabaseError(f"Failed to get golden test cases: {e}")
    
    ####################################################################################################################
    # Transaction Management
    
    @contextmanager
    def transaction(self):
        """
        Context manager for database transactions.
        Ensures atomic operations with automatic rollback on error.
        """
        conn = self.get_connection()
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise