# chat_exceptions.py
# Description: Custom exception classes for the Chat module with proper error handling
#
# Imports
import traceback
import uuid
from datetime import datetime
from enum import Enum
from typing import Optional, Dict, Any
from contextvars import ContextVar
from loguru import logger

#######################################################################################################################
#
# Context Variables for Request Tracking
#######################################################################################################################

# Thread-safe context variable for request ID
request_id_var: ContextVar[str] = ContextVar('request_id', default='')

def set_request_id(request_id: Optional[str] = None) -> str:
    """
    Set a request ID for the current context.
    
    Args:
        request_id: Optional request ID. If None, generates a new UUID.
        
    Returns:
        The request ID that was set.
    """
    if request_id is None:
        request_id = str(uuid.uuid4())
    request_id_var.set(request_id)
    return request_id

def get_request_id() -> str:
    """Get the current request ID from context."""
    return request_id_var.get() or str(uuid.uuid4())

#######################################################################################################################
#
# Error Codes
#######################################################################################################################

class ChatErrorCode(Enum):
    """Standardized error codes for the Chat module."""
    
    # Authentication errors (AUTH_xxx)
    AUTH_MISSING_TOKEN = "AUTH_001"
    AUTH_INVALID_TOKEN = "AUTH_002"
    AUTH_EXPIRED_TOKEN = "AUTH_003"
    AUTH_INSUFFICIENT_PERMISSIONS = "AUTH_004"
    
    # Validation errors (VAL_xxx)
    VAL_INVALID_REQUEST = "VAL_001"
    VAL_MESSAGE_TOO_LONG = "VAL_002"
    VAL_TOO_MANY_MESSAGES = "VAL_003"
    VAL_INVALID_IMAGE = "VAL_004"
    VAL_FILE_TOO_LARGE = "VAL_005"
    VAL_INVALID_FILE_TYPE = "VAL_006"
    
    # Database errors (DB_xxx)
    DB_CONNECTION_ERROR = "DB_001"
    DB_QUERY_ERROR = "DB_002"
    DB_TRANSACTION_ERROR = "DB_003"
    DB_INTEGRITY_ERROR = "DB_004"
    DB_NOT_FOUND = "DB_005"
    
    # External API errors (EXT_xxx)
    EXT_PROVIDER_ERROR = "EXT_001"
    EXT_RATE_LIMITED = "EXT_002"
    EXT_TIMEOUT = "EXT_003"
    EXT_INVALID_RESPONSE = "EXT_004"
    EXT_API_KEY_ERROR = "EXT_005"
    
    # Internal errors (INT_xxx)
    INT_PROCESSING_ERROR = "INT_001"
    INT_CONFIGURATION_ERROR = "INT_002"
    INT_UNEXPECTED_ERROR = "INT_999"
    
    # Rate limiting (RATE_xxx)
    RATE_LIMIT_EXCEEDED = "RATE_001"
    RATE_QUOTA_EXCEEDED = "RATE_002"

#######################################################################################################################
#
# Base Exception Class
#######################################################################################################################

class ChatModuleException(Exception):
    """
    Base exception class for all Chat module errors.
    Provides structured error information and logging.
    """
    
    def __init__(
        self,
        code: ChatErrorCode,
        message: str,
        details: Optional[Dict[str, Any]] = None,
        cause: Optional[Exception] = None,
        user_message: Optional[str] = None
    ):
        """
        Initialize a Chat module exception.
        
        Args:
            code: Error code from ChatErrorCode enum
            message: Internal error message (for logging)
            details: Additional error details (for debugging)
            cause: The underlying exception that caused this error
            user_message: Safe message to show to end users
        """
        self.code = code
        self.message = message
        self.details = details or {}
        self.cause = cause
        self.user_message = user_message or "An error occurred processing your request"
        self.request_id = get_request_id()
        self.timestamp = datetime.utcnow()
        
        # Include traceback if there's a cause
        if cause:
            self.traceback = traceback.format_exc()
        else:
            self.traceback = None
            
        super().__init__(message)
    
    def to_log_dict(self) -> Dict[str, Any]:
        """
        Convert exception to a dictionary for structured logging.
        Includes all internal details for debugging.
        """
        return {
            "error_code": self.code.value,
            "message": self.message,
            "request_id": self.request_id,
            "timestamp": self.timestamp.isoformat(),
            "details": self.details,
            "traceback": self.traceback,
            "cause": str(self.cause) if self.cause else None,
            "cause_type": type(self.cause).__name__ if self.cause else None
        }
    
    def to_response_dict(self) -> Dict[str, Any]:
        """
        Convert exception to a safe dictionary for API responses.
        Excludes sensitive internal details.
        """
        return {
            "error": {
                "code": self.code.value,
                "message": self.user_message,
                "request_id": self.request_id,
                "timestamp": self.timestamp.isoformat()
            }
        }
    
    def log(self, level: str = "error"):
        """
        Log the exception with full context.
        
        Args:
            level: Log level (debug, info, warning, error, critical)
        """
        log_func = getattr(logger, level, logger.error)
        log_func(
            f"Chat module error: {self.code.value} - {self.message}",
            **self.to_log_dict()
        )

#######################################################################################################################
#
# Specific Exception Classes
#######################################################################################################################

class ChatAuthenticationError(ChatModuleException):
    """Authentication-related errors."""
    
    def __init__(self, message: str, details: Optional[Dict] = None, cause: Optional[Exception] = None):
        super().__init__(
            code=ChatErrorCode.AUTH_INVALID_TOKEN,
            message=message,
            details=details,
            cause=cause,
            user_message="Authentication failed. Please check your credentials."
        )

class ChatValidationError(ChatModuleException):
    """Request validation errors."""
    
    def __init__(self, message: str, details: Optional[Dict] = None, validation_errors: Optional[list] = None):
        if details is None:
            details = {}
        if validation_errors:
            details["validation_errors"] = validation_errors
            
        super().__init__(
            code=ChatErrorCode.VAL_INVALID_REQUEST,
            message=message,
            details=details,
            user_message="Invalid request. Please check your input."
        )

class ChatDatabaseError(ChatModuleException):
    """Database operation errors."""
    
    def __init__(self, message: str, operation: str, details: Optional[Dict] = None, cause: Optional[Exception] = None):
        if details is None:
            details = {}
        details["operation"] = operation
        
        super().__init__(
            code=ChatErrorCode.DB_QUERY_ERROR,
            message=message,
            details=details,
            cause=cause,
            user_message="A database error occurred. Please try again later."
        )

class ChatProviderError(ChatModuleException):
    """External LLM provider errors."""
    
    def __init__(self, provider: str, message: str, status_code: Optional[int] = None, cause: Optional[Exception] = None):
        details = {
            "provider": provider,
            "status_code": status_code
        }
        
        super().__init__(
            code=ChatErrorCode.EXT_PROVIDER_ERROR,
            message=message,
            details=details,
            cause=cause,
            user_message=f"The chat service is temporarily unavailable. Please try again later."
        )

class ChatRateLimitError(ChatModuleException):
    """Rate limiting errors."""
    
    def __init__(self, limit: int, window: str, retry_after: Optional[int] = None):
        details = {
            "limit": limit,
            "window": window,
            "retry_after": retry_after
        }
        
        user_msg = f"Rate limit exceeded. Please wait {retry_after} seconds before trying again." if retry_after else "Rate limit exceeded. Please try again later."
        
        super().__init__(
            code=ChatErrorCode.RATE_LIMIT_EXCEEDED,
            message=f"Rate limit exceeded: {limit} requests per {window}",
            details=details,
            user_message=user_msg
        )

class ChatFileError(ChatModuleException):
    """File operation errors."""
    
    def __init__(self, message: str, filename: Optional[str] = None, cause: Optional[Exception] = None):
        details = {}
        if filename:
            details["filename"] = filename
            
        super().__init__(
            code=ChatErrorCode.VAL_INVALID_FILE_TYPE,
            message=message,
            details=details,
            cause=cause,
            user_message="File processing failed. Please check the file and try again."
        )

#######################################################################################################################
#
# Error Handler Utilities
#######################################################################################################################

def handle_database_error(operation: str, error: Exception, return_default: Any = None) -> Any:
    """
    Handle database errors consistently.
    
    Args:
        operation: Description of the database operation
        error: The exception that occurred
        return_default: Default value to return on error
        
    Returns:
        The default value after logging the error
    """
    import sqlite3
    
    # Determine error type and create appropriate exception
    if isinstance(error, sqlite3.IntegrityError):
        chat_error = ChatDatabaseError(
            message=f"Database integrity error during {operation}",
            operation=operation,
            cause=error
        )
        chat_error.code = ChatErrorCode.DB_INTEGRITY_ERROR
    elif isinstance(error, sqlite3.OperationalError):
        chat_error = ChatDatabaseError(
            message=f"Database operational error during {operation}",
            operation=operation,
            cause=error
        )
        chat_error.code = ChatErrorCode.DB_CONNECTION_ERROR
    else:
        chat_error = ChatDatabaseError(
            message=f"Unexpected database error during {operation}",
            operation=operation,
            cause=error
        )
    
    # Log the error
    chat_error.log()
    
    # Return safe default
    return return_default

def handle_provider_error(provider: str, error: Exception, status_code: Optional[int] = None) -> None:
    """
    Handle LLM provider errors consistently.
    
    Args:
        provider: Name of the LLM provider
        error: The exception that occurred
        status_code: HTTP status code if available
        
    Raises:
        ChatProviderError: Always raises after logging
    """
    chat_error = ChatProviderError(
        provider=provider,
        message=str(error),
        status_code=status_code,
        cause=error
    )
    chat_error.log()
    raise chat_error

def sanitize_error_message(error: Exception) -> str:
    """
    Sanitize an error message for safe display to users.
    Removes sensitive information like file paths, server details, etc.
    
    Args:
        error: The exception to sanitize
        
    Returns:
        A safe error message for users
    """
    error_str = str(error)
    
    # Remove file paths
    import re
    error_str = re.sub(r'[/\\][\w\-_./\\]+', '[path]', error_str)
    
    # Remove IP addresses
    error_str = re.sub(r'\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b', '[ip]', error_str)
    
    # Remove potential secrets (anything that looks like a key/token)
    # Redact: API keys starting with sk_, hex strings > 32 chars, mixed case+digit strings > 40 chars
    # But don't redact simple repeated characters
    error_str = re.sub(r'\bsk_[A-Za-z0-9_\-]{20,}\b', '[redacted]', error_str)  # API keys
    error_str = re.sub(r'\b[A-Fa-f0-9]{32,}\b', '[redacted]', error_str)  # Hex tokens
    error_str = re.sub(r'\b(?=.*[A-Z])(?=.*[a-z])(?=.*[0-9])[A-Za-z0-9_\-]{40,}\b', '[redacted]', error_str)  # Mixed long tokens
    
    # Truncate if too long
    if len(error_str) > 200:
        error_str = error_str[:197] + "..."
    
    return error_str

#######################################################################################################################
#
# Context Manager for Error Handling
#######################################################################################################################

class ErrorHandler:
    """
    Context manager for consistent error handling in the Chat module.
    
    Usage:
        with ErrorHandler("operation_name") as handler:
            # Your code here
            pass
    """
    
    def __init__(self, operation: str, default_return: Any = None):
        """
        Initialize error handler context.
        
        Args:
            operation: Name of the operation being performed
            default_return: Default value to return on error
        """
        self.operation = operation
        self.default_return = default_return
        self.request_id = set_request_id()
        
    def __enter__(self):
        """Enter the context."""
        logger.debug(f"Starting operation: {self.operation} (request_id: {self.request_id})")
        return self
    
    def __exit__(self, exc_type, exc_value, traceback_obj):
        """
        Handle any exceptions that occurred.
        
        Returns:
            True to suppress the exception, False to propagate it
        """
        if exc_type is None:
            logger.debug(f"Operation completed successfully: {self.operation}")
            return False
        
        # Create appropriate Chat exception
        if isinstance(exc_value, ChatModuleException):
            # Already a Chat exception, just log it
            exc_value.log()
            return False
        
        # Wrap in ChatModuleException
        chat_error = ChatModuleException(
            code=ChatErrorCode.INT_UNEXPECTED_ERROR,
            message=f"Unexpected error in {self.operation}",
            cause=exc_value,
            user_message="An unexpected error occurred. Please try again."
        )
        chat_error.log()
        
        # Suppress the original exception
        return True

#
# End of chat_exceptions.py
#######################################################################################################################