"""
Vector store adapters for the RAG service.

This package provides a unified interface for different vector databases,
allowing easy switching between ChromaDB, Pinecone, Weaviate, and others.
"""

from .base import (
    VectorStoreAdapter,
    VectorStoreConfig,
    VectorSearchResult,
    VectorStoreType
)
from .chromadb_adapter import ChromaDBAdapter
from .factory import VectorStoreFactory

__all__ = [
    "VectorStoreAdapter",
    "VectorStoreConfig",
    "VectorSearchResult",
    "VectorStoreType",
    "ChromaDBAdapter",
    "VectorStoreFactory"
]