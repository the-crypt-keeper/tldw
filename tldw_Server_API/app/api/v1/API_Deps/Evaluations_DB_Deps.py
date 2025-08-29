# Evaluations_DB_Deps.py
"""
Manages user-specific evaluation audit logger instances for dependency injection.
"""

import threading
from pathlib import Path
import logging
from typing import Dict, Optional

from fastapi import Depends, HTTPException, status
try:
    from cachetools import LRUCache
    _HAS_CACHETOOLS = True
except ImportError:
    _HAS_CACHETOOLS = False
    logging.warning("cachetools not found. Evaluations logger cache will grow indefinitely. "
                    "Install with: pip install cachetools")

# Local Imports
from tldw_Server_API.app.core.config import settings
from tldw_Server_API.app.core.AuthNZ.User_DB_Handling import get_request_user, User
from tldw_Server_API.app.core.Evaluations.audit_logger import AuditLogger
from tldw_Server_API.app.core.DB_Management.db_path_utils import DatabasePaths

#######################################################################################################################

# --- Configuration ---
MAX_CACHED_EVAL_LOGGERS = settings.get("MAX_CACHED_EVAL_LOGGERS", 20)

if _HAS_CACHETOOLS:
    # Keyed by user ID (int)
    _user_eval_loggers: LRUCache = LRUCache(maxsize=MAX_CACHED_EVAL_LOGGERS)
    logging.info(f"Using LRUCache for evaluation logger instances (maxsize={MAX_CACHED_EVAL_LOGGERS}).")
else:
    # Keyed by user ID (int)
    _user_eval_loggers: Dict[int, AuditLogger] = {}

_eval_logger_lock = threading.Lock()

#######################################################################################################################

# --- Helper Functions ---

def _create_eval_logger_for_user(user_id: int) -> AuditLogger:
    """
    Create a new evaluation audit logger instance for a specific user.
    
    Args:
        user_id: The user's ID
        
    Returns:
        Initialized AuditLogger instance
    """
    # Get the user-specific evaluations database path
    db_path = DatabasePaths.get_evaluations_db_path(user_id)
    
    logging.info(f"Creating evaluation audit logger for user {user_id} at path: {db_path}")
    
    # Create the logger with user-specific database
    logger = AuditLogger(db_path=str(db_path))
    
    logging.info(f"Evaluation audit logger initialized successfully for user {user_id}")
    return logger

# --- Main Dependency Function ---

def get_evaluations_logger_for_user(
    current_user: User = Depends(get_request_user)
) -> AuditLogger:
    """
    FastAPI dependency to get the AuditLogger instance for evaluations for the identified user.
    
    Handles caching and initialization.
    Uses configuration values from the 'settings' dictionary.
    
    Args:
        current_user: The User object provided by `get_request_user`.
        
    Returns:
        An AuditLogger instance for the user's evaluations.
        
    Raises:
        HTTPException: If the logger cannot be initialized.
    """
    if not current_user or not isinstance(current_user.id, int):
        logging.error("get_evaluations_logger_for_user called without a valid User object/ID.")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, 
            detail="User identification failed for evaluations logger."
        )
    
    user_id = current_user.id
    logger_instance: Optional[AuditLogger] = None
    
    # Check cache
    with _eval_logger_lock:
        logger_instance = _user_eval_loggers.get(user_id)
    
    if logger_instance:
        logging.debug(f"Using cached evaluation logger instance for user_id: {user_id}")
        return logger_instance
    
    # Instance not cached: create new one
    logging.info(f"No cached evaluation logger found for user_id: {user_id}. Initializing.")
    
    with _eval_logger_lock:
        # Double-check cache in case another thread created it
        logger_instance = _user_eval_loggers.get(user_id)
        if logger_instance:
            logging.debug(f"Evaluation logger for user {user_id} created concurrently.")
            return logger_instance
        
        try:
            # Create the logger
            logger_instance = _create_eval_logger_for_user(user_id)
            
            # Store in cache
            _user_eval_loggers[user_id] = logger_instance
            
            logging.info(f"Evaluation logger created and cached successfully for user {user_id}")
            
        except Exception as e:
            logging.error(f"Failed to initialize evaluation logger for user {user_id}: {e}", exc_info=True)
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Could not initialize evaluation logger: {str(e)}"
            ) from e
    
    return logger_instance

# --- Cleanup Functions ---

def close_user_eval_logger(user_id: int):
    """
    Close evaluation logger for a specific user.
    
    Args:
        user_id: The user's ID
    """
    with _eval_logger_lock:
        logger = _user_eval_loggers.pop(user_id, None)
    
    if logger:
        try:
            # AuditLogger doesn't have an explicit close method,
            # but we remove it from cache to allow garbage collection
            logging.info(f"Closed evaluation logger for user {user_id}")
        except Exception as e:
            logging.error(f"Error closing evaluation logger for user {user_id}: {e}", exc_info=True)

def close_all_eval_loggers():
    """
    Close all cached evaluation logger instances.
    Useful for application shutdown.
    """
    with _eval_logger_lock:
        user_ids = list(_user_eval_loggers.keys())
        logging.info(f"Closing evaluation loggers for {len(user_ids)} users...")
    
    for user_id in user_ids:
        close_user_eval_logger(user_id)
    
    logging.info("All evaluation loggers closed successfully.")

# Example of how to register for shutdown event in FastAPI:
# from fastapi import FastAPI
# app = FastAPI()
# @app.on_event("shutdown")
# def shutdown_event():
#     close_all_eval_loggers()

#
# End of Evaluations_DB_Deps.py
########################################################################################################################