"""
Logging configuration for RAG services.

This module provides centralized logging configuration to control
sensitive information logging in production environments.
"""

import os
from loguru import logger
from typing import Optional, Dict, Any


# Log levels for different components
LOG_LEVELS = {
    "default": os.environ.get("RAG_LOG_LEVEL", "INFO"),
    "queries": os.environ.get("RAG_LOG_QUERIES_LEVEL", "WARNING"),  # Higher level for query logging
    "content": os.environ.get("RAG_LOG_CONTENT_LEVEL", "WARNING"),  # Higher level for content logging
    "metadata": os.environ.get("RAG_LOG_METADATA_LEVEL", "INFO"),
    "performance": os.environ.get("RAG_LOG_PERFORMANCE_LEVEL", "INFO"),
}

# Sensitive data masking
MASK_SENSITIVE = os.environ.get("RAG_MASK_SENSITIVE", "true").lower() == "true"

# Query truncation length
QUERY_TRUNCATE_LENGTH = int(os.environ.get("RAG_QUERY_TRUNCATE_LENGTH", "50"))


def should_log_queries() -> bool:
    """Check if query logging is enabled based on log level."""
    import loguru
    current_level = logger._core.min_level
    query_level = loguru.logger.level(LOG_LEVELS["queries"]).no
    return current_level <= query_level


def should_log_content() -> bool:
    """Check if content logging is enabled based on log level."""
    import loguru
    current_level = logger._core.min_level
    content_level = loguru.logger.level(LOG_LEVELS["content"]).no
    return current_level <= content_level


def truncate_query(query: str, max_length: Optional[int] = None) -> str:
    """
    Truncate query for logging.
    
    Args:
        query: Query string to truncate
        max_length: Maximum length (defaults to QUERY_TRUNCATE_LENGTH)
        
    Returns:
        Truncated query with ellipsis if needed
    """
    if max_length is None:
        max_length = QUERY_TRUNCATE_LENGTH
    
    if len(query) <= max_length:
        return query
    
    return f"{query[:max_length]}..."


def mask_sensitive_metadata(metadata: Dict[str, Any]) -> Dict[str, Any]:
    """
    Mask sensitive fields in metadata.
    
    Args:
        metadata: Metadata dictionary
        
    Returns:
        Metadata with sensitive fields masked
    """
    if not MASK_SENSITIVE:
        return metadata
    
    sensitive_keys = {
        "api_key", "password", "secret", "token", "auth", "credential",
        "private", "ssn", "email", "phone", "address"
    }
    
    masked_metadata = metadata.copy()
    for key in masked_metadata:
        if any(sensitive in key.lower() for sensitive in sensitive_keys):
            masked_metadata[key] = "***MASKED***"
    
    return masked_metadata


def log_query_event(level: str, message: str, query: Optional[str] = None, **kwargs):
    """
    Log a query-related event with appropriate truncation.
    
    Args:
        level: Log level (debug, info, warning, error)
        message: Log message
        query: Optional query to include
        **kwargs: Additional context
    """
    if query and should_log_queries():
        truncated_query = truncate_query(query)
        message = f"{message} [query: '{truncated_query}']"
    
    # Mask any metadata in kwargs
    if "metadata" in kwargs and MASK_SENSITIVE:
        kwargs["metadata"] = mask_sensitive_metadata(kwargs["metadata"])
    
    getattr(logger, level)(message, **kwargs)


def configure_logging():
    """
    Configure logging settings for the RAG system.
    
    This should be called once at startup.
    """
    # Set default log level
    logger.remove()  # Remove default handler
    logger.add(
        sink=os.environ.get("RAG_LOG_FILE", "rag_service.log"),
        level=LOG_LEVELS["default"],
        rotation="100 MB",
        retention="7 days",
        compression="zip"
    )
    
    # Add console handler with appropriate level
    if os.environ.get("RAG_LOG_CONSOLE", "true").lower() == "true":
        logger.add(
            sink=os.sys.stderr,
            level=LOG_LEVELS["default"],
            colorize=True
        )
    
    logger.info(f"RAG logging configured: levels={LOG_LEVELS}, mask_sensitive={MASK_SENSITIVE}")


# Example usage patterns for safe logging:
"""
# Instead of:
logger.info(f"Processing query: {query}")

# Use:
log_query_event("info", "Processing query", query=query)

# Instead of:
logger.debug(f"Metadata: {metadata}")

# Use:
logger.debug(f"Metadata: {mask_sensitive_metadata(metadata)}")
"""