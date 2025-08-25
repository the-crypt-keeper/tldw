# template_initialization.py
"""
Initialize built-in chunking templates in the database.
This module loads templates from JSON files and seeds them into the database.
"""

import json
import os
from pathlib import Path
from typing import List, Dict, Any
from loguru import logger

from tldw_Server_API.app.core.DB_Management.Media_DB_v2 import MediaDatabase


def load_builtin_templates() -> List[Dict[str, Any]]:
    """
    Load all built-in templates from the template_library directory.
    
    Returns:
        List of template dictionaries
    """
    templates = []
    template_dir = Path(__file__).parent / "template_library"
    
    if not template_dir.exists():
        logger.warning(f"Template library directory not found: {template_dir}")
        return templates
    
    # Load all JSON files from template_library
    for template_file in template_dir.glob("*.json"):
        try:
            with open(template_file, 'r') as f:
                template_data = json.load(f)
                
                # Ensure required fields
                if 'name' not in template_data:
                    logger.error(f"Template file {template_file} missing 'name' field")
                    continue
                
                templates.append({
                    'name': template_data['name'],
                    'description': template_data.get('description', ''),
                    'tags': template_data.get('tags', []),
                    'template': template_data  # Store the entire template config
                })
                
                logger.info(f"Loaded template: {template_data['name']} from {template_file}")
                
        except json.JSONDecodeError as e:
            logger.error(f"Invalid JSON in template file {template_file}: {e}")
        except Exception as e:
            logger.error(f"Error loading template file {template_file}: {e}")
    
    return templates


def initialize_chunking_templates(db_path: str = None, client_id: str = 'system') -> int:
    """
    Initialize built-in chunking templates in the database.
    
    Args:
        db_path: Path to the database file (uses default if None)
        client_id: Client ID for database operations
        
    Returns:
        Number of templates successfully seeded
    """
    try:
        # Get database path from config if not provided
        if db_path is None:
            import os
            db_path = os.environ.get('TLDW_DB_PATH', 'Databases/Media_DB_v2.db')
        
        # Create database instance
        db = MediaDatabase(db_path=db_path, client_id=client_id)
        
        # Load built-in templates
        templates = load_builtin_templates()
        
        if not templates:
            logger.warning("No built-in templates found to seed")
            return 0
        
        # Seed templates into database
        count = db.seed_builtin_templates(templates)
        
        logger.info(f"Successfully seeded {count} built-in templates into database")
        return count
        
    except Exception as e:
        logger.error(f"Error initializing chunking templates: {e}")
        return 0


def update_builtin_templates(db_path: str = None, client_id: str = 'system', force: bool = False) -> int:
    """
    Update existing built-in templates with latest definitions.
    
    Args:
        db_path: Path to the database file
        client_id: Client ID for database operations
        force: If True, overwrites existing templates even if not changed
        
    Returns:
        Number of templates updated
    """
    try:
        if db_path is None:
            import os
            db_path = os.environ.get('TLDW_DB_PATH', 'Databases/Media_DB_v2.db')
        
        db = MediaDatabase(db_path=db_path, client_id=client_id)
        templates = load_builtin_templates()
        
        updated_count = 0
        
        for template in templates:
            existing = db.get_chunking_template(name=template['name'])
            
            if existing and existing['is_builtin']:
                # Compare template content
                existing_template = json.loads(existing['template_json'])
                new_template = template['template']
                
                if force or existing_template != new_template:
                    # Update the template
                    success = db.update_chunking_template(
                        name=template['name'],
                        template_json=json.dumps(new_template),
                        description=template.get('description'),
                        tags=template.get('tags')
                    )
                    
                    if success:
                        updated_count += 1
                        logger.info(f"Updated built-in template: {template['name']}")
        
        return updated_count
        
    except Exception as e:
        logger.error(f"Error updating built-in templates: {e}")
        return 0


# Convenience function to be called during application startup
def ensure_templates_initialized(db_path: str = None) -> bool:
    """
    Ensure built-in templates are initialized in the database.
    Called during application startup.
    
    Args:
        db_path: Path to the database file
        
    Returns:
        True if templates are properly initialized
    """
    try:
        count = initialize_chunking_templates(db_path=db_path)
        
        if count > 0:
            logger.info(f"Initialized {count} chunking templates on startup")
        else:
            # Check if templates already exist
            import os
            if db_path is None:
                db_path = os.environ.get('TLDW_DB_PATH', 'Databases/Media_DB_v2.db')
            
            db = MediaDatabase(db_path=db_path, client_id='system')
            existing = db.list_chunking_templates(include_builtin=True, include_custom=False)
            
            if existing:
                logger.debug(f"Found {len(existing)} existing built-in templates")
            else:
                logger.warning("No chunking templates found in database")
        
        return True
        
    except Exception as e:
        logger.error(f"Failed to ensure templates initialized: {e}")
        return False


if __name__ == "__main__":
    # If run directly, initialize templates
    import sys
    
    db_path = sys.argv[1] if len(sys.argv) > 1 else None
    count = initialize_chunking_templates(db_path)
    print(f"Initialized {count} templates")