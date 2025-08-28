# exceptions.py
"""
Custom exceptions for the chunking module.
"""

from typing import Optional, Any, Dict


class ChunkingError(Exception):
    """Base exception for all chunking-related errors."""
    
    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        """
        Initialize chunking error.
        
        Args:
            message: Error message
            details: Optional error details
        """
        super().__init__(message)
        self.message = message
        self.details = details or {}


class InvalidInputError(ChunkingError):
    """Exception raised when input validation fails."""
    pass


class InvalidChunkingMethodError(ChunkingError):
    """Exception raised when an invalid chunking method is specified."""
    pass


class TokenizerError(ChunkingError):
    """Exception raised for tokenizer-related errors."""
    pass


class TemplateError(ChunkingError):
    """Exception raised for template-related errors."""
    
    def __init__(self, message: str, template_name: Optional[str] = None, **kwargs):
        """
        Initialize template error.
        
        Args:
            message: Error message
            template_name: Name of the template that caused the error
            **kwargs: Additional error details
        """
        details = {'template_name': template_name, **kwargs}
        super().__init__(message, details)


class LanguageNotSupportedError(ChunkingError):
    """Exception raised when a language is not supported."""
    
    def __init__(self, language: str, available_languages: Optional[list] = None):
        """
        Initialize language not supported error.
        
        Args:
            language: The unsupported language code
            available_languages: List of supported languages
        """
        message = f"Language '{language}' is not supported"
        if available_languages:
            message += f". Supported languages: {', '.join(available_languages)}"
        
        super().__init__(message, {'language': language, 'available': available_languages})


class ChunkSizeError(ChunkingError):
    """Exception raised for invalid chunk size parameters."""
    
    def __init__(self, message: str, max_size: Optional[int] = None, 
                 overlap: Optional[int] = None, **kwargs):
        """
        Initialize chunk size error.
        
        Args:
            message: Error message
            max_size: Maximum chunk size that caused the error
            overlap: Overlap size that caused the error
            **kwargs: Additional error details
        """
        details = {'max_size': max_size, 'overlap': overlap, **kwargs}
        super().__init__(message, details)


class ProcessingError(ChunkingError):
    """Exception raised during text processing."""
    
    def __init__(self, message: str, stage: Optional[str] = None, 
                 operation: Optional[str] = None, **kwargs):
        """
        Initialize processing error.
        
        Args:
            message: Error message
            stage: Processing stage where error occurred
            operation: Specific operation that failed
            **kwargs: Additional error details
        """
        details = {'stage': stage, 'operation': operation, **kwargs}
        super().__init__(message, details)


class ConfigurationError(ChunkingError):
    """Exception raised for configuration-related errors."""
    pass


class CacheError(ChunkingError):
    """Exception raised for cache-related errors."""
    pass


# For backward compatibility with existing code
ChunkerError = ChunkingError  # Alias


__all__ = [
    'ChunkingError',
    'InvalidInputError',
    'InvalidChunkingMethodError',
    'TokenizerError',
    'TemplateError',
    'LanguageNotSupportedError',
    'ChunkSizeError',
    'ProcessingError',
    'ConfigurationError',
    'CacheError',
    'ChunkerError',  # Backward compatibility alias
]