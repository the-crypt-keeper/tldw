# Audit_DB_Deps.py
"""
Manages user-specific audit service instances for dependency injection.
"""

import asyncio
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
    logging.warning("cachetools not found. Audit service cache will grow indefinitely. "
                    "Install with: pip install cachetools")

# Local Imports
from tldw_Server_API.app.core.config import settings
from tldw_Server_API.app.core.AuthNZ.User_DB_Handling import get_request_user, User
from tldw_Server_API.app.core.Audit.unified_audit_service import UnifiedAuditService
from tldw_Server_API.app.core.DB_Management.db_path_utils import DatabasePaths

#######################################################################################################################

# --- Configuration ---
MAX_CACHED_AUDIT_INSTANCES = settings.get("MAX_CACHED_AUDIT_INSTANCES", 20)

if _HAS_CACHETOOLS:
    # Keyed by user ID (int)
    _user_audit_instances: LRUCache = LRUCache(maxsize=MAX_CACHED_AUDIT_INSTANCES)
    logging.info(f"Using LRUCache for audit service instances (maxsize={MAX_CACHED_AUDIT_INSTANCES}).")
else:
    # Keyed by user ID (int)
    _user_audit_instances: Dict[int, UnifiedAuditService] = {}

_audit_service_lock = threading.Lock()
_service_initialization_lock = asyncio.Lock()

#######################################################################################################################

# --- Helper Functions ---

async def _create_audit_service_for_user(user_id: int) -> UnifiedAuditService:
    """
    Create a new audit service instance for a specific user.
    
    Args:
        user_id: The user's ID
        
    Returns:
        Initialized UnifiedAuditService instance
    """
    # Get the user-specific audit database path
    db_path = DatabasePaths.get_audit_db_path(user_id)
    
    logging.info(f"Creating audit service for user {user_id} at path: {db_path}")
    
    # Create the service with user-specific database
    service = UnifiedAuditService(
        db_path=str(db_path),
        retention_days=settings.get("AUDIT_RETENTION_DAYS", 30),
        enable_pii_detection=settings.get("AUDIT_ENABLE_PII_DETECTION", True),
        enable_risk_scoring=settings.get("AUDIT_ENABLE_RISK_SCORING", True),
        buffer_size=settings.get("AUDIT_BUFFER_SIZE", 100),
        flush_interval=settings.get("AUDIT_FLUSH_INTERVAL", 5.0)
    )
    
    # Initialize the service (creates database, starts background tasks)
    await service.initialize()
    
    logging.info(f"Audit service initialized successfully for user {user_id}")
    return service

# --- Main Dependency Function ---

async def get_audit_service_for_user(
    current_user: User = Depends(get_request_user)
) -> UnifiedAuditService:
    """
    FastAPI dependency to get the UnifiedAuditService instance for the identified user.
    
    Handles caching, initialization, and lifecycle management.
    Uses configuration values from the 'settings' dictionary.
    
    Args:
        current_user: The User object provided by `get_request_user`.
        
    Returns:
        A UnifiedAuditService instance for the user.
        
    Raises:
        HTTPException: If the service cannot be initialized.
    """
    if not current_user or not isinstance(current_user.id, int):
        logging.error("get_audit_service_for_user called without a valid User object/ID.")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, 
            detail="User identification failed for audit service."
        )
    
    user_id = current_user.id
    service_instance: Optional[UnifiedAuditService] = None
    
    # Check cache
    with _audit_service_lock:
        service_instance = _user_audit_instances.get(user_id)
    
    if service_instance:
        # Verify the service is still healthy
        # The UnifiedAuditService should handle its own health checks
        logging.debug(f"Using cached audit service instance for user_id: {user_id}")
        return service_instance
    
    # Instance not cached or unhealthy: create new one
    logging.info(f"No cached audit service found for user_id: {user_id}. Initializing.")
    
    # Use async lock for initialization to prevent concurrent creation
    async with _service_initialization_lock:
        # Double-check cache in case another request created it
        with _audit_service_lock:
            service_instance = _user_audit_instances.get(user_id)
            if service_instance:
                logging.debug(f"Audit service for user {user_id} created concurrently.")
                return service_instance
        
        try:
            # Create and initialize the service
            service_instance = await _create_audit_service_for_user(user_id)
            
            # Store in cache
            with _audit_service_lock:
                _user_audit_instances[user_id] = service_instance
            
            logging.info(f"Audit service created and cached successfully for user {user_id}")
            
        except Exception as e:
            logging.error(f"Failed to initialize audit service for user {user_id}: {e}", exc_info=True)
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Could not initialize audit service: {str(e)}"
            ) from e
    
    return service_instance

# --- Cleanup Functions ---

async def shutdown_user_audit_service(user_id: int):
    """
    Shutdown audit service for a specific user.
    
    Args:
        user_id: The user's ID
    """
    with _audit_service_lock:
        service = _user_audit_instances.pop(user_id, None)
    
    if service:
        try:
            await service.shutdown()
            logging.info(f"Shut down audit service for user {user_id}")
        except Exception as e:
            logging.error(f"Error shutting down audit service for user {user_id}: {e}", exc_info=True)

async def shutdown_all_audit_services():
    """
    Shutdown all cached audit service instances.
    Useful for application shutdown.
    """
    with _audit_service_lock:
        user_ids = list(_user_audit_instances.keys())
        logging.info(f"Shutting down audit services for {len(user_ids)} users...")
    
    for user_id in user_ids:
        await shutdown_user_audit_service(user_id)
    
    logging.info("All audit services shut down successfully.")

# Example of how to register for shutdown event in FastAPI:
# from fastapi import FastAPI
# app = FastAPI()
# @app.on_event("shutdown")
# async def shutdown_event():
#     await shutdown_all_audit_services()

#
# End of Audit_DB_Deps.py
########################################################################################################################