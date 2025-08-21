# chunking_interop_library.py
# Description: This module provides a service layer for managing chunking templates and configurations
#
# Imports
import json
import logging
import threading
from pathlib import Path
from typing import List, Dict, Optional, Any, Union, Tuple
from datetime import datetime
import sqlite3

# Third-Party Imports
from loguru import logger

# Local Imports
from ..DB.Client_Media_DB_v2 import MediaDatabase, DatabaseError, InputError, ConflictError
from ..Metrics.metrics_logger import log_counter, log_histogram

#######################################################################################################################
#
# Functions:

logger = logging.getLogger(__name__)


class ChunkingTemplateError(Exception):
    """Base exception for chunking template related errors."""
    pass


class TemplateNotFoundError(ChunkingTemplateError):
    """Exception raised when a template is not found."""
    pass


class SystemTemplateError(ChunkingTemplateError):
    """Exception raised when trying to modify a system template."""
    pass


class ChunkingInteropService:
    """
    Service layer for managing chunking templates and per-document configurations.
    Provides a clean interface for CRUD operations on templates and configurations.
    """
    
    def __init__(self, media_db: MediaDatabase):
        """
        Initialize the chunking interop service.
        
        Args:
            media_db: MediaDatabase instance to use for operations
        """
        self.media_db = media_db
        self._template_cache = {}
        self._cache_lock = threading.Lock()
        logger.info("ChunkingInteropService initialized")
    
    # --- Template Management Methods ---
    
    def get_all_templates(self, include_system: bool = True) -> List[Dict[str, Any]]:
        """
        Retrieve all chunking templates.
        
        Args:
            include_system: Whether to include system templates
            
        Returns:
            List of template dictionaries
        """
        try:
            query = "SELECT * FROM ChunkingTemplates"
            params = []
            
            if not include_system:
                query += " WHERE is_system = 0"
            
            query += " ORDER BY is_system DESC, name ASC"
            
            conn = self.media_db.get_connection()
            cursor = conn.execute(query, params)
            
            templates = []
            for row in cursor:
                template = self._row_to_template_dict(row)
                templates.append(template)
                # Update cache
                with self._cache_lock:
                    self._template_cache[template['id']] = template
            
            log_counter("chunking_templates_fetched", len(templates))
            return templates
            
        except Exception as e:
            logger.error(f"Error fetching templates: {e}")
            raise ChunkingTemplateError(f"Failed to fetch templates: {str(e)}")
    
    def get_template_by_id(self, template_id: int) -> Dict[str, Any]:
        """
        Retrieve a specific template by ID.
        
        Args:
            template_id: Template ID
            
        Returns:
            Template dictionary
            
        Raises:
            TemplateNotFoundError: If template not found
        """
        # Check cache first
        with self._cache_lock:
            if template_id in self._template_cache:
                return self._template_cache[template_id].copy()
        
        try:
            conn = self.media_db.get_connection()
            cursor = conn.execute(
                "SELECT * FROM ChunkingTemplates WHERE id = ?",
                (template_id,)
            )
            
            row = cursor.fetchone()
            if not row:
                raise TemplateNotFoundError(f"Template with ID {template_id} not found")
            
            template = self._row_to_template_dict(row)
            
            # Update cache
            with self._cache_lock:
                self._template_cache[template_id] = template
            
            return template
            
        except TemplateNotFoundError:
            raise
        except Exception as e:
            logger.error(f"Error fetching template {template_id}: {e}")
            raise ChunkingTemplateError(f"Failed to fetch template: {str(e)}")
    
    def get_template_by_name(self, name: str) -> Optional[Dict[str, Any]]:
        """
        Retrieve a template by name.
        
        Args:
            name: Template name
            
        Returns:
            Template dictionary or None if not found
        """
        try:
            conn = self.media_db.get_connection()
            cursor = conn.execute(
                "SELECT * FROM ChunkingTemplates WHERE name = ?",
                (name,)
            )
            
            row = cursor.fetchone()
            if not row:
                return None
            
            template = self._row_to_template_dict(row)
            
            # Update cache
            with self._cache_lock:
                self._template_cache[template['id']] = template
            
            return template
            
        except Exception as e:
            logger.error(f"Error fetching template by name '{name}': {e}")
            raise ChunkingTemplateError(f"Failed to fetch template: {str(e)}")
    
    def create_template(
        self,
        name: str,
        description: str,
        template_json: Union[str, Dict[str, Any]],
        is_system: bool = False
    ) -> int:
        """
        Create a new chunking template.
        
        Args:
            name: Template name (must be unique)
            description: Template description
            template_json: Template configuration as JSON string or dict
            is_system: Whether this is a system template
            
        Returns:
            ID of created template
            
        Raises:
            InputError: If validation fails
            ChunkingTemplateError: If creation fails
        """
        # Validate inputs
        if not name or not name.strip():
            raise InputError("Template name cannot be empty")
        
        if not description or not description.strip():
            raise InputError("Template description cannot be empty")
        
        # Convert dict to JSON string if needed
        if isinstance(template_json, dict):
            try:
                template_json_str = json.dumps(template_json)
            except (TypeError, ValueError) as e:
                raise InputError(f"Invalid template JSON: {str(e)}")
        else:
            template_json_str = template_json
            # Validate JSON
            try:
                json.loads(template_json_str)
            except json.JSONDecodeError as e:
                raise InputError(f"Invalid JSON format: {str(e)}")
        
        try:
            # Check if name already exists
            existing = self.get_template_by_name(name)
            if existing:
                raise InputError(f"Template with name '{name}' already exists")
            
            # Insert template
            conn = self.media_db.get_connection()
            cursor = conn.execute("""
                INSERT INTO ChunkingTemplates 
                (name, description, template_json, is_system, created_at, updated_at)
                VALUES (?, ?, ?, ?, CURRENT_TIMESTAMP, CURRENT_TIMESTAMP)
            """, (name.strip(), description.strip(), template_json_str, int(is_system)))
            
            template_id = cursor.lastrowid
            conn.commit()
            
            # Clear cache to force refresh
            with self._cache_lock:
                self._template_cache.clear()
            
            log_counter("chunking_template_created", 1)
            logger.info(f"Created chunking template '{name}' with ID {template_id}")
            
            return template_id
            
        except InputError:
            raise
        except Exception as e:
            logger.error(f"Error creating template: {e}")
            raise ChunkingTemplateError(f"Failed to create template: {str(e)}")
    
    def update_template(
        self,
        template_id: int,
        name: Optional[str] = None,
        description: Optional[str] = None,
        template_json: Optional[Union[str, Dict[str, Any]]] = None
    ) -> None:
        """
        Update an existing template.
        
        Args:
            template_id: Template ID
            name: New name (optional)
            description: New description (optional)
            template_json: New template JSON (optional)
            
        Raises:
            TemplateNotFoundError: If template not found
            SystemTemplateError: If trying to modify system template
            InputError: If validation fails
        """
        # Get existing template
        template = self.get_template_by_id(template_id)
        
        # Check if system template
        if template['is_system']:
            raise SystemTemplateError("Cannot modify system templates")
        
        # Build update query
        updates = []
        params = []
        
        if name is not None:
            if not name.strip():
                raise InputError("Template name cannot be empty")
            # Check if new name conflicts
            existing = self.get_template_by_name(name)
            if existing and existing['id'] != template_id:
                raise InputError(f"Template with name '{name}' already exists")
            updates.append("name = ?")
            params.append(name.strip())
        
        if description is not None:
            if not description.strip():
                raise InputError("Template description cannot be empty")
            updates.append("description = ?")
            params.append(description.strip())
        
        if template_json is not None:
            # Convert and validate JSON
            if isinstance(template_json, dict):
                try:
                    template_json_str = json.dumps(template_json)
                except (TypeError, ValueError) as e:
                    raise InputError(f"Invalid template JSON: {str(e)}")
            else:
                template_json_str = template_json
                try:
                    json.loads(template_json_str)
                except json.JSONDecodeError as e:
                    raise InputError(f"Invalid JSON format: {str(e)}")
            
            updates.append("template_json = ?")
            params.append(template_json_str)
        
        if not updates:
            return  # Nothing to update
        
        # Add updated_at
        updates.append("updated_at = CURRENT_TIMESTAMP")
        
        # Execute update
        try:
            query = f"UPDATE ChunkingTemplates SET {', '.join(updates)} WHERE id = ?"
            params.append(template_id)
            
            conn = self.media_db.get_connection()
            conn.execute(query, params)
            conn.commit()
            
            # Clear from cache
            with self._cache_lock:
                self._template_cache.pop(template_id, None)
            
            log_counter("chunking_template_updated", 1)
            logger.info(f"Updated chunking template ID {template_id}")
            
        except Exception as e:
            logger.error(f"Error updating template: {e}")
            raise ChunkingTemplateError(f"Failed to update template: {str(e)}")
    
    def delete_template(self, template_id: int) -> None:
        """
        Delete a template.
        
        Args:
            template_id: Template ID
            
        Raises:
            TemplateNotFoundError: If template not found
            SystemTemplateError: If trying to delete system template
        """
        # Get existing template
        template = self.get_template_by_id(template_id)
        
        # Check if system template
        if template['is_system']:
            raise SystemTemplateError("Cannot delete system templates")
        
        try:
            conn = self.media_db.get_connection()
            conn.execute("DELETE FROM ChunkingTemplates WHERE id = ?", (template_id,))
            conn.commit()
            
            # Remove from cache
            with self._cache_lock:
                self._template_cache.pop(template_id, None)
            
            log_counter("chunking_template_deleted", 1)
            logger.info(f"Deleted chunking template ID {template_id}")
            
        except Exception as e:
            logger.error(f"Error deleting template: {e}")
            raise ChunkingTemplateError(f"Failed to delete template: {str(e)}")
    
    def duplicate_template(
        self,
        template_id: int,
        new_name: str,
        new_description: Optional[str] = None
    ) -> int:
        """
        Duplicate an existing template.
        
        Args:
            template_id: Source template ID
            new_name: Name for the duplicate
            new_description: Description for duplicate (optional)
            
        Returns:
            ID of created duplicate
        """
        # Get source template
        source = self.get_template_by_id(template_id)
        
        # Use source description if not provided
        if new_description is None:
            new_description = f"Copy of {source['description']}"
        
        # Create duplicate (always as custom template)
        return self.create_template(
            name=new_name,
            description=new_description,
            template_json=source['template_json'],
            is_system=False
        )
    
    # --- Document Configuration Methods ---
    
    def get_document_config(self, media_id: int) -> Optional[Dict[str, Any]]:
        """
        Get chunking configuration for a specific document.
        
        Args:
            media_id: Media document ID
            
        Returns:
            Configuration dict or None if not configured
        """
        try:
            conn = self.media_db.get_connection()
            cursor = conn.execute(
                "SELECT chunking_config FROM Media WHERE id = ?",
                (media_id,)
            )
            
            row = cursor.fetchone()
            if not row or not row['chunking_config']:
                return None
            
            return json.loads(row['chunking_config'])
            
        except json.JSONDecodeError as e:
            logger.error(f"Invalid JSON in chunking_config for media {media_id}: {e}")
            return None
        except Exception as e:
            logger.error(f"Error fetching document config: {e}")
            raise ChunkingTemplateError(f"Failed to fetch document config: {str(e)}")
    
    def set_document_config(
        self,
        media_id: int,
        config: Dict[str, Any]
    ) -> None:
        """
        Set chunking configuration for a document.
        
        Args:
            media_id: Media document ID
            config: Configuration dictionary
        """
        try:
            config_json = json.dumps(config)
            
            conn = self.media_db.get_connection()
            conn.execute(
                "UPDATE Media SET chunking_config = ? WHERE id = ?",
                (config_json, media_id)
            )
            conn.commit()
            
            log_counter("document_chunking_config_set", 1)
            logger.info(f"Set chunking config for media {media_id}")
            
        except Exception as e:
            logger.error(f"Error setting document config: {e}")
            raise ChunkingTemplateError(f"Failed to set document config: {str(e)}")
    
    def clear_document_config(self, media_id: int) -> None:
        """
        Clear chunking configuration for a document.
        
        Args:
            media_id: Media document ID
        """
        try:
            conn = self.media_db.get_connection()
            conn.execute(
                "UPDATE Media SET chunking_config = NULL WHERE id = ?",
                (media_id,)
            )
            conn.commit()
            
            log_counter("document_chunking_config_cleared", 1)
            logger.info(f"Cleared chunking config for media {media_id}")
            
        except Exception as e:
            logger.error(f"Error clearing document config: {e}")
            raise ChunkingTemplateError(f"Failed to clear document config: {str(e)}")
    
    def get_documents_using_template(self, template_name: str) -> List[Dict[str, Any]]:
        """
        Get all documents using a specific template.
        
        Args:
            template_name: Template name to search for
            
        Returns:
            List of media items using the template
        """
        try:
            conn = self.media_db.get_connection()
            cursor = conn.execute("""
                SELECT id, title, author, type, chunking_config
                FROM Media
                WHERE chunking_config LIKE ?
                AND deleted = 0
            """, (f'%"template": "{template_name}"%',))
            
            documents = []
            for row in cursor:
                doc = {
                    'id': row['id'],
                    'title': row['title'],
                    'author': row['author'],
                    'type': row['type'],
                    'config': json.loads(row['chunking_config']) if row['chunking_config'] else None
                }
                documents.append(doc)
            
            return documents
            
        except Exception as e:
            logger.error(f"Error fetching documents using template: {e}")
            raise ChunkingTemplateError(f"Failed to fetch documents: {str(e)}")
    
    # --- Import/Export Methods ---
    
    def export_template(self, template_id: int) -> Dict[str, Any]:
        """
        Export a template for sharing.
        
        Args:
            template_id: Template ID
            
        Returns:
            Export dictionary with template data
        """
        template = self.get_template_by_id(template_id)
        
        # Parse template JSON
        template_data = json.loads(template['template_json'])
        
        export_data = {
            'name': template['name'],
            'description': template['description'],
            'template_json': template_data,
            'exported_at': datetime.now().isoformat(),
            'version': '1.0',
            'source': 'tldw_chatbook'
        }
        
        return export_data
    
    def import_template(
        self,
        import_data: Dict[str, Any],
        name_suffix: str = " (Imported)"
    ) -> int:
        """
        Import a template from export data.
        
        Args:
            import_data: Template export data
            name_suffix: Suffix to add if name conflicts
            
        Returns:
            ID of imported template
        """
        # Validate required fields
        required_fields = ['name', 'description', 'template_json']
        for field in required_fields:
            if field not in import_data:
                raise InputError(f"Missing required field: {field}")
        
        name = import_data['name']
        
        # Check for name conflict
        existing = self.get_template_by_name(name)
        if existing:
            name = f"{name}{name_suffix}"
        
        # Create template
        return self.create_template(
            name=name,
            description=import_data['description'],
            template_json=import_data['template_json'],
            is_system=False
        )
    
    # --- Statistics Methods ---
    
    def get_template_statistics(self) -> Dict[str, Any]:
        """
        Get statistics about template usage.
        
        Returns:
            Dictionary with statistics
        """
        try:
            conn = self.media_db.get_connection()
            
            # Count templates
            cursor = conn.execute("""
                SELECT 
                    COUNT(*) as total,
                    SUM(CASE WHEN is_system = 1 THEN 1 ELSE 0 END) as system_count,
                    SUM(CASE WHEN is_system = 0 THEN 1 ELSE 0 END) as custom_count
                FROM ChunkingTemplates
            """)
            
            template_stats = cursor.fetchone()
            
            # Count configured documents
            cursor = conn.execute("""
                SELECT COUNT(*) as configured_docs
                FROM Media
                WHERE chunking_config IS NOT NULL
                AND deleted = 0
            """)
            
            doc_stats = cursor.fetchone()
            
            # Get most used templates
            cursor = conn.execute("""
                SELECT 
                    json_extract(chunking_config, '$.template') as template_name,
                    COUNT(*) as usage_count
                FROM Media
                WHERE chunking_config IS NOT NULL
                AND deleted = 0
                GROUP BY template_name
                ORDER BY usage_count DESC
                LIMIT 5
            """)
            
            most_used = []
            for row in cursor:
                if row['template_name']:
                    most_used.append({
                        'template': row['template_name'],
                        'count': row['usage_count']
                    })
            
            return {
                'total_templates': template_stats['total'],
                'system_templates': template_stats['system_count'],
                'custom_templates': template_stats['custom_count'],
                'configured_documents': doc_stats['configured_docs'],
                'most_used_templates': most_used
            }
            
        except Exception as e:
            logger.error(f"Error getting template statistics: {e}")
            return {
                'total_templates': 0,
                'system_templates': 0,
                'custom_templates': 0,
                'configured_documents': 0,
                'most_used_templates': []
            }
    
    # --- Helper Methods ---
    
    def _row_to_template_dict(self, row: sqlite3.Row) -> Dict[str, Any]:
        """Convert a database row to a template dictionary."""
        return {
            'id': row['id'],
            'name': row['name'],
            'description': row['description'],
            'template_json': row['template_json'],
            'is_system': bool(row['is_system']),
            'created_at': row['created_at'],
            'updated_at': row['updated_at']
        }
    
    def validate_template_json(self, template_json: Union[str, Dict[str, Any]]) -> Tuple[bool, Optional[str]]:
        """
        Validate template JSON structure.
        
        Args:
            template_json: Template JSON to validate
            
        Returns:
            Tuple of (is_valid, error_message)
        """
        try:
            # Parse JSON if string
            if isinstance(template_json, str):
                template_data = json.loads(template_json)
            else:
                template_data = template_json
            
            # Check required fields
            required_fields = ['name', 'base_method', 'pipeline']
            for field in required_fields:
                if field not in template_data:
                    return False, f"Missing required field: {field}"
            
            # Validate pipeline structure
            if not isinstance(template_data['pipeline'], list):
                return False, "Pipeline must be a list"
            
            if not template_data['pipeline']:
                return False, "Pipeline cannot be empty"
            
            # Validate each stage
            for i, stage in enumerate(template_data['pipeline']):
                if not isinstance(stage, dict):
                    return False, f"Pipeline stage {i} must be a dictionary"
                
                if 'stage' not in stage:
                    return False, f"Pipeline stage {i} missing 'stage' field"
                
                if stage['stage'] not in ['preprocess', 'chunk', 'postprocess']:
                    return False, f"Pipeline stage {i} has invalid stage type: {stage['stage']}"
            
            return True, None
            
        except json.JSONDecodeError as e:
            return False, f"Invalid JSON: {str(e)}"
        except Exception as e:
            return False, f"Validation error: {str(e)}"


# --- Convenience Functions ---

def get_chunking_service(media_db: MediaDatabase) -> ChunkingInteropService:
    """
    Get a ChunkingInteropService instance.
    
    Args:
        media_db: MediaDatabase instance
        
    Returns:
        ChunkingInteropService instance
    """
    return ChunkingInteropService(media_db)


# End of chunking_interop_library.py
#######################################################################################################################