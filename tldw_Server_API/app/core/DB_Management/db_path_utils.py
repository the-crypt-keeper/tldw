# db_path_utils.py
"""
Centralized database path management utilities.
Ensures consistent database file locations across the application.
"""

import os
from pathlib import Path
from typing import Optional, Dict
import logging

from tldw_Server_API.app.core.config import settings

logger = logging.getLogger(__name__)


class DatabasePaths:
    """Centralized database path management."""
    
    # Database file names
    MEDIA_DB_NAME = "Media_DB_v2.db"
    CHACHA_DB_NAME = "ChaChaNotes.db"
    PROMPTS_DB_NAME = "user_prompts_v2.sqlite"
    AUDIT_DB_NAME = "unified_audit.db"
    EVALUATIONS_DB_NAME = "evaluations.db"
    
    # Subdirectories
    PROMPTS_SUBDIR = "prompts_user_dbs"
    AUDIT_SUBDIR = "audit"
    EVALUATIONS_SUBDIR = "evaluations"
    
    @staticmethod
    def get_user_base_directory(user_id: int) -> Path:
        """
        Get the base directory for a specific user's databases.
        
        Args:
            user_id: The user's ID
            
        Returns:
            Path to the user's database directory
        """
        user_db_base = settings.get("USER_DB_BASE_DIR")
        if not user_db_base:
            # Fallback to default location
            project_root = Path(__file__).parent.parent.parent.parent.parent
            user_db_base = project_root / "Databases" / "user_databases"
            logger.warning(f"USER_DB_BASE_DIR not configured, using fallback: {user_db_base}")
        
        user_dir = Path(user_db_base) / str(user_id)
        
        # Ensure directory exists
        try:
            user_dir.mkdir(parents=True, exist_ok=True)
            logger.debug(f"Ensured user directory exists: {user_dir}")
        except OSError as e:
            logger.error(f"Failed to create user directory {user_dir}: {e}")
            raise
        
        return user_dir
    
    @staticmethod
    def get_media_db_path(user_id: int) -> Path:
        """Get the path to the user's media database."""
        user_dir = DatabasePaths.get_user_base_directory(user_id)
        return user_dir / DatabasePaths.MEDIA_DB_NAME
    
    @staticmethod
    def get_chacha_db_path(user_id: int) -> Path:
        """Get the path to the user's ChaChaNotes database."""
        user_dir = DatabasePaths.get_user_base_directory(user_id)
        return user_dir / DatabasePaths.CHACHA_DB_NAME
    
    @staticmethod
    def get_prompts_db_path(user_id: int) -> Path:
        """Get the path to the user's prompts database."""
        user_dir = DatabasePaths.get_user_base_directory(user_id)
        prompts_dir = user_dir / DatabasePaths.PROMPTS_SUBDIR
        
        # Ensure prompts subdirectory exists
        try:
            prompts_dir.mkdir(parents=True, exist_ok=True)
        except OSError as e:
            logger.error(f"Failed to create prompts directory {prompts_dir}: {e}")
            raise
        
        return prompts_dir / DatabasePaths.PROMPTS_DB_NAME
    
    @staticmethod
    def get_audit_db_path(user_id: int) -> Path:
        """Get the path to the user's audit database."""
        user_dir = DatabasePaths.get_user_base_directory(user_id)
        audit_dir = user_dir / DatabasePaths.AUDIT_SUBDIR
        
        # Ensure audit subdirectory exists
        try:
            audit_dir.mkdir(parents=True, exist_ok=True)
        except OSError as e:
            logger.error(f"Failed to create audit directory {audit_dir}: {e}")
            raise
        
        return audit_dir / DatabasePaths.AUDIT_DB_NAME
    
    @staticmethod
    def get_evaluations_db_path(user_id: int) -> Path:
        """Get the path to the user's evaluations database."""
        user_dir = DatabasePaths.get_user_base_directory(user_id)
        eval_dir = user_dir / DatabasePaths.EVALUATIONS_SUBDIR
        
        # Ensure evaluations subdirectory exists
        try:
            eval_dir.mkdir(parents=True, exist_ok=True)
        except OSError as e:
            logger.error(f"Failed to create evaluations directory {eval_dir}: {e}")
            raise
        
        return eval_dir / DatabasePaths.EVALUATIONS_DB_NAME
    
    @staticmethod
    def get_all_user_db_paths(user_id: int) -> Dict[str, Path]:
        """
        Get all database paths for a user.
        
        Returns:
            Dictionary mapping database types to their paths
        """
        return {
            "media": DatabasePaths.get_media_db_path(user_id),
            "chacha": DatabasePaths.get_chacha_db_path(user_id),
            "prompts": DatabasePaths.get_prompts_db_path(user_id),
            "audit": DatabasePaths.get_audit_db_path(user_id),
            "evaluations": DatabasePaths.get_evaluations_db_path(user_id),
        }
    
    @staticmethod
    def validate_database_structure(user_id: int) -> bool:
        """
        Validate that all required directories exist for a user.
        
        Args:
            user_id: The user's ID
            
        Returns:
            True if all directories exist or were created successfully
        """
        try:
            paths = DatabasePaths.get_all_user_db_paths(user_id)
            logger.info(f"Database structure validated for user {user_id}")
            return True
        except Exception as e:
            logger.error(f"Failed to validate database structure for user {user_id}: {e}")
            return False
    
    @staticmethod
    def get_single_user_id() -> int:
        """
        Get the user ID for single-user mode.
        
        Returns:
            The configured single-user ID (typically 1)
        """
        return int(settings.get("SINGLE_USER_FIXED_ID", "1"))


# Convenience functions for backward compatibility
def get_user_media_db_path(user_id: int) -> str:
    """Get the media database path for a user (returns string for compatibility)."""
    return str(DatabasePaths.get_media_db_path(user_id))


def get_user_chacha_db_path(user_id: int) -> str:
    """Get the ChaChaNotes database path for a user (returns string for compatibility)."""
    return str(DatabasePaths.get_chacha_db_path(user_id))


def get_user_prompts_db_path(user_id: int) -> str:
    """Get the prompts database path for a user (returns string for compatibility)."""
    return str(DatabasePaths.get_prompts_db_path(user_id))


def ensure_user_database_structure(user_id: int) -> bool:
    """
    Ensure all database directories exist for a user.
    
    Args:
        user_id: The user's ID
        
    Returns:
        True if successful
    """
    return DatabasePaths.validate_database_structure(user_id)