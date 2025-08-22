# RAG Module Exports
#
# This module provides the main RAG (Retrieval Augmented Generation) functionality
# for the tldw_server application using a functional pipeline architecture.

from .rag_service.config import RAGConfig
from .rag_service.types import DataSource, Document, SearchResult
from .rag_service.functional_pipeline import (
    RAGPipelineContext,
    minimal_pipeline,
    standard_pipeline,
    quality_pipeline,
    enhanced_pipeline,
    custom_pipeline,
    build_pipeline,
    # Individual pipeline functions
    expand_query,
    check_cache,
    retrieve_documents,
    optimize_chromadb_search,
    process_tables,
    rerank_documents,
    store_in_cache,
    analyze_performance,
)

__all__ = [
    # Configuration and types
    'RAGConfig',
    'DataSource',
    'Document',
    'SearchResult',
    'RAGPipelineContext',
    # Pre-built pipelines
    'minimal_pipeline',
    'standard_pipeline',
    'quality_pipeline',
    'enhanced_pipeline',
    'custom_pipeline',
    # Pipeline builder
    'build_pipeline',
    # Individual functions
    'expand_query',
    'check_cache',
    'retrieve_documents',
    'optimize_chromadb_search',
    'process_tables',
    'rerank_documents',
    'store_in_cache',
    'analyze_performance',
]