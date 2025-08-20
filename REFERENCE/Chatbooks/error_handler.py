# error_handler.py
# Description: Error handling for chatbook operations
#
"""
Chatbook Error Handler
----------------------

Provides comprehensive error handling for chatbook operations with:
- Specific exception types
- User-friendly error messages
- Recovery suggestions
- Logging integration
"""

from typing import Optional, Dict, Any, Callable
from enum import Enum
from loguru import logger


class ChatbookErrorType(Enum):
    """Types of chatbook errors."""
    FILE_NOT_FOUND = "file_not_found"
    INVALID_FORMAT = "invalid_format"
    DATABASE_ERROR = "database_error"
    PERMISSION_ERROR = "permission_error"
    DISK_SPACE_ERROR = "disk_space_error"
    NETWORK_ERROR = "network_error"
    VALIDATION_ERROR = "validation_error"
    IMPORT_CONFLICT = "import_conflict"
    EXPORT_ERROR = "export_error"
    UNKNOWN_ERROR = "unknown_error"


class ChatbookError(Exception):
    """Base exception for chatbook operations."""
    
    def __init__(
        self,
        error_type: ChatbookErrorType,
        message: str,
        details: Optional[str] = None,
        recovery_suggestions: Optional[list] = None,
        original_exception: Optional[Exception] = None
    ):
        super().__init__(message)
        self.error_type = error_type
        self.message = message
        self.details = details
        self.recovery_suggestions = recovery_suggestions or []
        self.original_exception = original_exception
        
        # Log the error
        logger.error(f"ChatbookError [{error_type.value}]: {message}")
        if details:
            logger.error(f"Details: {details}")
        if original_exception:
            logger.exception(f"Original exception: {original_exception}")
    
    def get_user_message(self) -> str:
        """Get a user-friendly error message."""
        return self.message
    
    def get_full_message(self) -> str:
        """Get full error message with details."""
        parts = [self.message]
        if self.details:
            parts.append(f"\nDetails: {self.details}")
        if self.recovery_suggestions:
            parts.append("\nSuggestions:")
            for suggestion in self.recovery_suggestions:
                parts.append(f"  • {suggestion}")
        return "\n".join(parts)


class ChatbookErrorHandler:
    """Handles errors in chatbook operations."""
    
    # Error message templates
    ERROR_MESSAGES = {
        ChatbookErrorType.FILE_NOT_FOUND: "Chatbook file not found",
        ChatbookErrorType.INVALID_FORMAT: "Invalid chatbook format",
        ChatbookErrorType.DATABASE_ERROR: "Database operation failed",
        ChatbookErrorType.PERMISSION_ERROR: "Permission denied",
        ChatbookErrorType.DISK_SPACE_ERROR: "Insufficient disk space",
        ChatbookErrorType.NETWORK_ERROR: "Network error occurred",
        ChatbookErrorType.VALIDATION_ERROR: "Validation failed",
        ChatbookErrorType.IMPORT_CONFLICT: "Import conflict detected",
        ChatbookErrorType.EXPORT_ERROR: "Export failed",
        ChatbookErrorType.UNKNOWN_ERROR: "An unexpected error occurred"
    }
    
    # Recovery suggestions
    RECOVERY_SUGGESTIONS = {
        ChatbookErrorType.FILE_NOT_FOUND: [
            "Check if the file path is correct",
            "Ensure the file hasn't been moved or deleted",
            "Try browsing for the file again"
        ],
        ChatbookErrorType.INVALID_FORMAT: [
            "Ensure the file is a valid chatbook (.zip)",
            "The file may be corrupted - try downloading again",
            "Check if the file was created with a compatible version"
        ],
        ChatbookErrorType.DATABASE_ERROR: [
            "Check database connectivity",
            "Ensure database files are not locked",
            "Try restarting the application"
        ],
        ChatbookErrorType.PERMISSION_ERROR: [
            "Check file and folder permissions",
            "Run the application with appropriate privileges",
            "Choose a different location with write access"
        ],
        ChatbookErrorType.DISK_SPACE_ERROR: [
            "Free up disk space",
            "Choose a different export location",
            "Reduce the size of the export"
        ],
        ChatbookErrorType.NETWORK_ERROR: [
            "Check your internet connection",
            "Try again later",
            "Check proxy settings if applicable"
        ],
        ChatbookErrorType.VALIDATION_ERROR: [
            "Review the validation errors",
            "Ensure all required fields are filled",
            "Check for invalid characters or data"
        ],
        ChatbookErrorType.IMPORT_CONFLICT: [
            "Choose a different conflict resolution strategy",
            "Review existing items before importing",
            "Consider renaming conflicting items"
        ],
        ChatbookErrorType.EXPORT_ERROR: [
            "Check available disk space",
            "Ensure all selected content exists",
            "Try exporting fewer items"
        ],
        ChatbookErrorType.UNKNOWN_ERROR: [
            "Check the application logs for details",
            "Try the operation again",
            "Contact support if the issue persists"
        ]
    }
    
    @classmethod
    def handle_error(
        cls,
        exception: Exception,
        operation: str = "operation",
        context: Optional[Dict[str, Any]] = None
    ) -> ChatbookError:
        """
        Handle an exception and convert it to a ChatbookError.
        
        Args:
            exception: The exception to handle
            operation: Description of the operation that failed
            context: Additional context about the error
            
        Returns:
            ChatbookError with appropriate type and messages
        """
        error_type = cls._determine_error_type(exception)
        base_message = cls.ERROR_MESSAGES.get(error_type, "An error occurred")
        
        # Build detailed message
        message = f"{base_message} during {operation}"
        details = str(exception)
        
        # Add context if provided
        if context:
            context_parts = []
            for key, value in context.items():
                context_parts.append(f"{key}: {value}")
            if context_parts:
                details += f"\nContext: {', '.join(context_parts)}"
        
        # Get recovery suggestions
        suggestions = cls.RECOVERY_SUGGESTIONS.get(error_type, [])
        
        return ChatbookError(
            error_type=error_type,
            message=message,
            details=details,
            recovery_suggestions=suggestions,
            original_exception=exception
        )
    
    @classmethod
    def _determine_error_type(cls, exception: Exception) -> ChatbookErrorType:
        """Determine the error type from an exception."""
        error_str = str(exception).lower()
        exception_type = type(exception).__name__
        
        # Check for specific error patterns
        if isinstance(exception, FileNotFoundError) or "not found" in error_str:
            return ChatbookErrorType.FILE_NOT_FOUND
        
        elif isinstance(exception, PermissionError) or "permission" in error_str:
            return ChatbookErrorType.PERMISSION_ERROR
        
        elif "disk space" in error_str or "no space" in error_str:
            return ChatbookErrorType.DISK_SPACE_ERROR
        
        elif "database" in error_str or "sqlite" in error_str:
            return ChatbookErrorType.DATABASE_ERROR
        
        elif "network" in error_str or "connection" in error_str:
            return ChatbookErrorType.NETWORK_ERROR
        
        elif "validation" in error_str or "invalid" in error_str:
            return ChatbookErrorType.VALIDATION_ERROR
        
        elif "conflict" in error_str:
            return ChatbookErrorType.IMPORT_CONFLICT
        
        elif "export" in error_str:
            return ChatbookErrorType.EXPORT_ERROR
        
        elif "format" in error_str or "corrupt" in error_str:
            return ChatbookErrorType.INVALID_FORMAT
        
        else:
            return ChatbookErrorType.UNKNOWN_ERROR
    
    @classmethod
    def wrap_operation(
        cls,
        operation: Callable,
        operation_name: str,
        context: Optional[Dict[str, Any]] = None
    ):
        """
        Wrap an operation with error handling.
        
        Args:
            operation: The operation to wrap
            operation_name: Name of the operation for error messages
            context: Additional context for error reporting
            
        Returns:
            Result of the operation
            
        Raises:
            ChatbookError: If the operation fails
        """
        try:
            return operation()
        except ChatbookError:
            # Re-raise ChatbookErrors as-is
            raise
        except Exception as e:
            # Convert other exceptions to ChatbookError
            raise cls.handle_error(e, operation_name, context)
    
    @classmethod
    def create_error(
        cls,
        error_type: ChatbookErrorType,
        message: Optional[str] = None,
        details: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None
    ) -> ChatbookError:
        """
        Create a ChatbookError directly.
        
        Args:
            error_type: Type of error
            message: Custom message (uses default if not provided)
            details: Additional details
            context: Additional context
            
        Returns:
            ChatbookError instance
        """
        if not message:
            message = cls.ERROR_MESSAGES.get(error_type, "An error occurred")
        
        # Add context to details if provided
        if context and details:
            context_parts = [f"{k}: {v}" for k, v in context.items()]
            details += f"\nContext: {', '.join(context_parts)}"
        elif context:
            details = f"Context: {', '.join(f'{k}: {v}' for k, v in context.items())}"
        
        suggestions = cls.RECOVERY_SUGGESTIONS.get(error_type, [])
        
        return ChatbookError(
            error_type=error_type,
            message=message,
            details=details,
            recovery_suggestions=suggestions
        )


def safe_chatbook_operation(operation_name: str):
    """
    Decorator for safe chatbook operations with error handling.
    
    Usage:
        @safe_chatbook_operation("export chatbook")
        def export_chatbook(...):
            ...
    """
    def decorator(func):
        async def async_wrapper(*args, **kwargs):
            try:
                return await func(*args, **kwargs)
            except ChatbookError:
                raise
            except Exception as e:
                raise ChatbookErrorHandler.handle_error(e, operation_name)
        
        def sync_wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except ChatbookError:
                raise
            except Exception as e:
                raise ChatbookErrorHandler.handle_error(e, operation_name)
        
        # Return appropriate wrapper based on function type
        import asyncio
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        else:
            return sync_wrapper
    
    return decorator