# test_error_handling.py
# Description: Tests for improved error handling in the Chat module
#
# Imports
import pytest
import sqlite3
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime
import uuid

# Import the modules we're testing
from tldw_Server_API.app.core.Chat.chat_exceptions import (
    ChatModuleException,
    ChatAuthenticationError,
    ChatValidationError,
    ChatDatabaseError,
    ChatProviderError,
    ChatRateLimitError,
    ChatFileError,
    ChatErrorCode,
    set_request_id,
    get_request_id,
    handle_database_error,
    sanitize_error_message,
    ErrorHandler
)

#######################################################################################################################
#
# Test Custom Exception Classes
#######################################################################################################################

class TestChatExceptions:
    """Test custom exception classes."""
    
    def test_chat_module_exception_creation(self):
        """Test creating a ChatModuleException with all fields."""
        error = ChatModuleException(
            code=ChatErrorCode.INT_UNEXPECTED_ERROR,
            message="Test error message",
            details={"key": "value"},
            cause=ValueError("Original error"),
            user_message="User-friendly message"
        )
        
        assert error.code == ChatErrorCode.INT_UNEXPECTED_ERROR
        assert error.message == "Test error message"
        assert error.details == {"key": "value"}
        assert isinstance(error.cause, ValueError)
        assert error.user_message == "User-friendly message"
        assert error.request_id  # Should have a request ID
        assert isinstance(error.timestamp, datetime)
    
    def test_exception_to_log_dict(self):
        """Test converting exception to log dictionary."""
        original_error = ValueError("Original error")
        error = ChatModuleException(
            code=ChatErrorCode.DB_QUERY_ERROR,
            message="Database query failed",
            details={"query": "SELECT * FROM users"},
            cause=original_error
        )
        
        log_dict = error.to_log_dict()
        
        assert log_dict["error_code"] == "DB_002"
        assert log_dict["message"] == "Database query failed"
        assert log_dict["details"]["query"] == "SELECT * FROM users"
        assert log_dict["cause"] == "Original error"
        assert log_dict["cause_type"] == "ValueError"
        assert "request_id" in log_dict
        assert "timestamp" in log_dict
    
    def test_exception_to_response_dict(self):
        """Test converting exception to safe response dictionary."""
        error = ChatModuleException(
            code=ChatErrorCode.AUTH_INVALID_TOKEN,
            message="Token validation failed - secret details",
            details={"secret": "should_not_appear"},
            user_message="Authentication failed"
        )
        
        response_dict = error.to_response_dict()
        
        # Should only contain safe information
        assert "error" in response_dict
        assert response_dict["error"]["code"] == "AUTH_002"
        assert response_dict["error"]["message"] == "Authentication failed"
        assert "secret" not in str(response_dict)
        assert "secret details" not in str(response_dict)
        assert "request_id" in response_dict["error"]
    
    def test_specific_exception_classes(self):
        """Test specific exception subclasses."""
        # Authentication error
        auth_error = ChatAuthenticationError("Invalid token")
        assert auth_error.code == ChatErrorCode.AUTH_INVALID_TOKEN
        assert auth_error.user_message == "Authentication failed. Please check your credentials."
        
        # Validation error
        val_error = ChatValidationError(
            "Invalid input",
            validation_errors=["field1 required", "field2 too long"]
        )
        assert val_error.code == ChatErrorCode.VAL_INVALID_REQUEST
        assert val_error.details["validation_errors"] == ["field1 required", "field2 too long"]
        
        # Database error
        db_error = ChatDatabaseError(
            "Query failed",
            operation="SELECT",
            cause=sqlite3.OperationalError("database locked")
        )
        assert db_error.code == ChatErrorCode.DB_QUERY_ERROR
        assert db_error.details["operation"] == "SELECT"
        
        # Provider error
        provider_error = ChatProviderError(
            provider="openai",
            message="API error",
            status_code=500
        )
        assert provider_error.code == ChatErrorCode.EXT_PROVIDER_ERROR
        assert provider_error.details["provider"] == "openai"
        assert provider_error.details["status_code"] == 500
        
        # Rate limit error
        rate_error = ChatRateLimitError(
            limit=100,
            window="minute",
            retry_after=60
        )
        assert rate_error.code == ChatErrorCode.RATE_LIMIT_EXCEEDED
        assert "60 seconds" in rate_error.user_message

#######################################################################################################################
#
# Test Request ID Context
#######################################################################################################################

class TestRequestIDContext:
    """Test request ID context management."""
    
    def test_set_and_get_request_id(self):
        """Test setting and getting request ID."""
        # Set a specific request ID
        request_id = "test-request-123"
        set_id = set_request_id(request_id)
        
        assert set_id == request_id
        assert get_request_id() == request_id
    
    def test_auto_generate_request_id(self):
        """Test automatic request ID generation."""
        # Don't provide an ID
        request_id = set_request_id()
        
        assert request_id is not None
        assert len(request_id) == 36  # UUID format
        assert get_request_id() == request_id
    
    def test_request_id_in_exception(self):
        """Test that exceptions capture the current request ID."""
        request_id = set_request_id("exception-test-123")
        
        error = ChatModuleException(
            code=ChatErrorCode.INT_UNEXPECTED_ERROR,
            message="Test error"
        )
        
        assert error.request_id == "exception-test-123"

#######################################################################################################################
#
# Test Error Handler Utilities
#######################################################################################################################

class TestErrorHandlerUtilities:
    """Test error handler utility functions."""
    
    def test_handle_database_error_integrity(self):
        """Test handling database integrity errors."""
        error = sqlite3.IntegrityError("UNIQUE constraint failed")
        
        result = handle_database_error(
            operation="insert_user",
            error=error,
            return_default="default_value"
        )
        
        assert result == "default_value"
    
    def test_handle_database_error_operational(self):
        """Test handling database operational errors."""
        error = sqlite3.OperationalError("database is locked")
        
        result = handle_database_error(
            operation="query_data",
            error=error,
            return_default=None
        )
        
        assert result is None
    
    def test_handle_database_error_generic(self):
        """Test handling generic database errors."""
        error = Exception("Unknown database error")
        
        result = handle_database_error(
            operation="update_record",
            error=error,
            return_default=[]
        )
        
        assert result == []
    
    def test_sanitize_error_message_removes_paths(self):
        """Test that file paths are sanitized from error messages."""
        error = Exception("Error in /home/user/project/secret/file.py at line 42")
        
        sanitized = sanitize_error_message(error)
        
        assert "/home/user/project" not in sanitized
        assert "[path]" in sanitized
        assert "at line 42" in sanitized
    
    def test_sanitize_error_message_removes_ips(self):
        """Test that IP addresses are sanitized from error messages."""
        error = Exception("Connection failed to 192.168.1.100:8080")
        
        sanitized = sanitize_error_message(error)
        
        assert "192.168.1.100" not in sanitized
        assert "[ip]" in sanitized
        assert ":8080" in sanitized
    
    def test_sanitize_error_message_removes_secrets(self):
        """Test that potential secrets are sanitized from error messages."""
        error = Exception("API key sk_test_abcdef1234567890abcdef1234567890 is invalid")
        
        sanitized = sanitize_error_message(error)
        
        assert "sk_test_abcdef1234567890abcdef1234567890" not in sanitized
        assert "[redacted]" in sanitized
        assert "is invalid" in sanitized
    
    def test_sanitize_error_message_truncates_long(self):
        """Test that long error messages are truncated."""
        long_message = "Error: " + "x" * 300
        error = Exception(long_message)
        
        sanitized = sanitize_error_message(error)
        
        assert len(sanitized) == 200
        assert sanitized.endswith("...")

#######################################################################################################################
#
# Test Error Handler Context Manager
#######################################################################################################################

class TestErrorHandlerContext:
    """Test ErrorHandler context manager."""
    
    def test_error_handler_success(self):
        """Test ErrorHandler with successful operation."""
        with ErrorHandler("test_operation") as handler:
            # Successful operation
            result = 1 + 1
        
        assert result == 2
        # No exception should be raised
    
    def test_error_handler_catches_exception(self):
        """Test ErrorHandler catches and wraps exceptions."""
        with ErrorHandler("test_operation", default_return="default"):
            # This will raise an exception
            raise ValueError("Test error")
        
        # Context manager should suppress the exception
        # (returns True from __exit__)
    
    def test_error_handler_preserves_chat_exceptions(self):
        """Test ErrorHandler preserves ChatModuleException."""
        with pytest.raises(ChatAuthenticationError):
            with ErrorHandler("test_operation"):
                raise ChatAuthenticationError("Auth failed")
    
    def test_error_handler_sets_request_id(self):
        """Test ErrorHandler sets a request ID."""
        initial_id = get_request_id()
        
        with ErrorHandler("test_operation"):
            new_id = get_request_id()
        
        # Should have set a new request ID
        assert new_id != initial_id
        assert new_id != ""

#######################################################################################################################
#
# Integration Tests
#######################################################################################################################

class TestErrorHandlingIntegration:
    """Integration tests for error handling in real scenarios."""
    
    @patch('tldw_Server_API.app.core.Chat.chat_exceptions.logger')
    def test_database_error_logging(self, mock_logger):
        """Test that database errors are properly logged."""
        error = ChatDatabaseError(
            message="Failed to save message",
            operation="save_message",
            details={"conversation_id": "conv-123"},
            cause=sqlite3.OperationalError("database locked")
        )
        
        error.log()
        
        # Check that logger was called with correct information
        mock_logger.error.assert_called_once()
        call_args = mock_logger.error.call_args
        
        assert "DB_002" in str(call_args)
        assert "Failed to save message" in str(call_args)
        assert "conv-123" in str(call_args)
    
    @patch('tldw_Server_API.app.core.Chat.chat_exceptions.logger')
    def test_critical_error_logging(self, mock_logger):
        """Test that critical errors are logged at correct level."""
        error = ChatModuleException(
            code=ChatErrorCode.INT_UNEXPECTED_ERROR,
            message="Critical system failure",
            cause=MemoryError("Out of memory")
        )
        
        error.log(level="critical")
        
        # Should use critical log level
        mock_logger.critical.assert_called_once()
    
    def test_error_chain_with_cause(self):
        """Test error chaining with cause tracking."""
        original_error = ValueError("Original problem")
        
        db_error = ChatDatabaseError(
            message="Query failed",
            operation="select",
            cause=original_error
        )
        
        wrapper_error = ChatModuleException(
            code=ChatErrorCode.INT_PROCESSING_ERROR,
            message="Processing failed",
            cause=db_error
        )
        
        # Check error chain
        assert wrapper_error.cause == db_error
        assert db_error.cause == original_error
        assert wrapper_error.traceback is not None  # Should have traceback

#######################################################################################################################
#
# End of test_error_handling.py
#######################################################################################################################