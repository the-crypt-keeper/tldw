# Chatbooks module - Knowledge pack creation and management
"""
Chatbooks Module
----------------

This module provides functionality for creating, managing, and sharing
knowledge packs (chatbooks) that contain curated content from multiple
databases in the tldw_chatbook application.

Main components:
- chatbook_creator.py: Package creation logic
- chatbook_importer.py: Import and validation
- chatbook_models.py: Data structures and schemas
- conflict_resolver.py: Handle duplicate content during import
- error_handler.py: Comprehensive error handling
"""

from .chatbook_creator import ChatbookCreator
from .chatbook_importer import ChatbookImporter
from .chatbook_models import Chatbook, ChatbookManifest, ChatbookContent
from .error_handler import ChatbookError, ChatbookErrorHandler, ChatbookErrorType

__all__ = [
    'ChatbookCreator',
    'ChatbookImporter',
    'Chatbook',
    'ChatbookManifest',
    'ChatbookContent',
    'ChatbookError',
    'ChatbookErrorHandler',
    'ChatbookErrorType'
]