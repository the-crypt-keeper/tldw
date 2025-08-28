"""
RAG (Retrieval-Augmented Generation) Service

This package provides a functional pipeline implementation of RAG functionality
for the tldw_server application. It uses composable functions that can be
combined into custom pipelines for different use cases.

Main components:
- functional_pipeline.py: Core pipeline functions and presets
- config.py: Configuration management
- types.py: Type definitions
- database_retrievers.py: Database retrieval strategies
- query_expansion.py: Query expansion strategies
- advanced_reranking.py: Document reranking
- Various feature modules for caching, monitoring, etc.
"""

from .config import RAGConfig
from .types import DataSource, Document, SearchResult

__all__ = ['RAGConfig', 'DataSource', 'Document', 'SearchResult']