"""
Adapter layer for backward compatibility with existing RAG functions.

This module provides adapters that allow the existing monolithic search functions
to work with the new functional pipeline system. It enables gradual migration
while maintaining full backward compatibility.
"""

from typing import List, Dict, Any, Optional, Tuple
from loguru import logger

from .pipeline_core import (
    PipelineContext, PipelineConfig, PipelineStep, StepType,
    SearchResult, Success, PipelineResources
)
from .pipeline_builder import build_pipeline_from_dict
from .pipeline_resources import get_global_resource_manager


# ==============================================================================
# Legacy Function Adapters
# ==============================================================================

async def perform_plain_rag_search_v2(
    app: Any,
    query: str,
    sources: Dict[str, bool],
    top_k: int = 10,
    max_context_length: int = 10000,
    enable_rerank: bool = False,
    reranker_model: str = "flashrank",
    keyword_filter_list: Optional[List[str]] = None
) -> Tuple[List[Dict[str, Any]], str]:
    """
    Adapter for perform_plain_rag_search using the new pipeline system.
    
    This function maintains the exact same interface as the original
    but uses the new functional pipeline system internally.
    """
    # Build pipeline configuration
    steps = [
        {
            'type': 'retrieve',
            'function': 'retrieve_fts5',
            'config': {
                'top_k': top_k * 2 if enable_rerank else top_k,
                'keyword_filter': keyword_filter_list or []
            }
        }
    ]
    
    # Add deduplication
    steps.append({
        'type': 'process',
        'function': 'deduplicate_results',
        'config': {'strategy': 'content_hash'}
    })
    
    # Add reranking if enabled
    if enable_rerank:
        steps.append({
            'type': 'process',
            'function': 'rerank_results',
            'config': {
                'model': reranker_model,
                'top_k': top_k
            }
        })
    else:
        # Just limit results
        steps.append({
            'type': 'process',
            'function': 'filter_by_score',
            'config': {'min_score': 0.0}
        })
    
    # Add formatting
    steps.append({
        'type': 'format',
        'function': 'format_as_context',
        'config': {
            'max_length': max_context_length,
            'include_citations': True,
            'separator': '\n---\n'
        }
    })
    
    # Create pipeline config
    pipeline_config = {
        'id': 'plain_rag_adapter',
        'name': 'Plain RAG Search (Adapter)',
        'description': 'Backward compatible plain RAG search',
        'steps': steps,
        'cache_results': False  # Disable caching for compatibility
    }
    
    # Build and execute pipeline
    pipeline = build_pipeline_from_dict(pipeline_config)
    
    # Get resources
    resource_manager = get_global_resource_manager(app)
    resources = resource_manager.get_resources()
    
    # Create context
    context: PipelineContext = {
        'query': query,
        'sources': sources,
        'resources': resources,
        'params': {
            'top_k': top_k,
            'max_context_length': max_context_length
        }
    }
    
    # Execute pipeline
    result, effects = await pipeline(context)
    
    # Convert results to legacy format
    if isinstance(result, Success):
        # Convert SearchResult objects to dicts
        legacy_results = []
        for r in result.value:
            legacy_results.append({
                'source': r.source,
                'id': r.id,
                'title': r.title,
                'content': r.content,
                'score': r.score,
                'metadata': r.metadata
            })
        
        # Get formatted context
        formatted_context = context.get('formatted_output', '')
        
        return legacy_results, formatted_context
    else:
        # Return empty results on error
        logger.error(f"Pipeline execution failed: {result.error.message}")
        return [], ""


async def perform_full_rag_pipeline_v2(
    app: Any,
    query: str,
    sources: Dict[str, bool],
    top_k: int = 10,
    max_context_length: int = 10000,
    chunk_size: int = 400,
    chunk_overlap: int = 100,
    chunk_type: str = "words",
    include_metadata: bool = True,
    enable_rerank: bool = False,
    reranker_model: str = "flashrank",
    keyword_filter_list: Optional[List[str]] = None
) -> Tuple[List[Dict[str, Any]], str]:
    """
    Adapter for perform_full_rag_pipeline using the new pipeline system.
    
    This provides semantic search functionality using the new system.
    """
    # Build pipeline configuration
    steps = [
        {
            'type': 'retrieve',
            'function': 'retrieve_semantic',
            'config': {
                'top_k': top_k * 2 if enable_rerank else top_k,
                'score_threshold': 0.0,
                'include_citations': True,
                'chunk_size': chunk_size,
                'chunk_overlap': chunk_overlap
            }
        }
    ]
    
    # Add reranking if enabled
    if enable_rerank:
        steps.append({
            'type': 'process',
            'function': 'rerank_results',
            'config': {
                'model': reranker_model,
                'top_k': top_k
            }
        })
    
    # Add formatting
    steps.append({
        'type': 'format',
        'function': 'format_as_context',
        'config': {
            'max_length': max_context_length,
            'include_citations': include_metadata
        }
    })
    
    # Create pipeline config
    pipeline_config = {
        'id': 'semantic_rag_adapter',
        'name': 'Semantic RAG Pipeline (Adapter)',
        'description': 'Backward compatible semantic RAG pipeline',
        'steps': steps,
        'cache_results': True,  # Enable caching for semantic search
        'cache_ttl_seconds': 3600
    }
    
    # Build and execute pipeline
    pipeline = build_pipeline_from_dict(pipeline_config)
    
    # Get resources
    resource_manager = get_global_resource_manager(app)
    resources = resource_manager.get_resources()
    
    # Ensure RAG service is initialized
    await resource_manager.get_rag_service()
    
    # Create context
    context: PipelineContext = {
        'query': query,
        'sources': sources,
        'resources': resources,
        'params': {
            'top_k': top_k,
            'max_context_length': max_context_length,
            'chunk_size': chunk_size,
            'chunk_overlap': chunk_overlap
        }
    }
    
    # Execute pipeline
    result, effects = await pipeline(context)
    
    # Convert results to legacy format
    if isinstance(result, Success):
        legacy_results = []
        for r in result.value:
            legacy_result = {
                'source': r.source,
                'id': r.id,
                'title': r.title,
                'content': r.content,
                'score': r.score,
                'metadata': r.metadata
            }
            
            # Add citations if available
            if r.citations:
                legacy_result['citations'] = r.citations
            
            legacy_results.append(legacy_result)
        
        formatted_context = context.get('formatted_output', '')
        return legacy_results, formatted_context
    else:
        logger.error(f"Pipeline execution failed: {result.error.message}")
        return [], ""


async def perform_hybrid_rag_search_v2(
    app: Any,
    query: str,
    sources: Dict[str, bool],
    top_k: int = 10,
    max_context_length: int = 10000,
    enable_rerank: bool = False,
    reranker_model: str = "flashrank",
    chunk_size: int = 400,
    chunk_overlap: int = 100,
    chunk_type: str = "words",
    bm25_weight: float = 0.5,
    vector_weight: float = 0.5,
    keyword_filter_list: Optional[List[str]] = None
) -> Tuple[List[Dict[str, Any]], str]:
    """
    Adapter for perform_hybrid_rag_search using the new pipeline system.
    
    This combines FTS5 and semantic search with configurable weights.
    """
    # Normalize weights
    total_weight = bm25_weight + vector_weight
    if total_weight > 0:
        bm25_weight = bm25_weight / total_weight
        vector_weight = vector_weight / total_weight
    else:
        bm25_weight = 0.5
        vector_weight = 0.5
    
    # Build pipeline configuration
    steps = [
        {
            'type': 'parallel',
            'functions': [
                {
                    'function': 'retrieve_fts5',
                    'config': {
                        'top_k': top_k * 2,
                        'keyword_filter': keyword_filter_list or []
                    }
                },
                {
                    'function': 'retrieve_semantic',
                    'config': {
                        'top_k': top_k * 2,
                        'score_threshold': 0.0,
                        'include_citations': True,
                        'chunk_size': chunk_size,
                        'chunk_overlap': chunk_overlap
                    }
                }
            ]
        },
        {
            'type': 'merge',
            'function': 'weighted_merge',
            'config': {
                'weights': [bm25_weight, vector_weight]
            }
        },
        {
            'type': 'process',
            'function': 'deduplicate_results',
            'config': {'strategy': 'content_hash'}
        }
    ]
    
    # Add reranking if enabled
    if enable_rerank:
        steps.append({
            'type': 'process',
            'function': 'rerank_results',
            'config': {
                'model': reranker_model,
                'top_k': top_k
            }
        })
    else:
        # Just limit to top_k
        steps.append({
            'type': 'process',
            'function': 'filter_by_score',
            'config': {'min_score': 0.0}
        })
    
    # Add formatting
    steps.append({
        'type': 'format',
        'function': 'format_as_context',
        'config': {
            'max_length': max_context_length,
            'include_citations': True
        }
    })
    
    # Create pipeline config
    pipeline_config = {
        'id': 'hybrid_rag_adapter',
        'name': 'Hybrid RAG Search (Adapter)',
        'description': 'Backward compatible hybrid RAG search',
        'steps': steps,
        'cache_results': True,
        'cache_ttl_seconds': 1800  # 30 minutes
    }
    
    # Build and execute pipeline
    pipeline = build_pipeline_from_dict(pipeline_config)
    
    # Get resources
    resource_manager = get_global_resource_manager(app)
    resources = resource_manager.get_resources()
    
    # Ensure RAG service is initialized for semantic search
    await resource_manager.get_rag_service()
    
    # Create context
    context: PipelineContext = {
        'query': query,
        'sources': sources,
        'resources': resources,
        'params': {
            'top_k': top_k,
            'max_context_length': max_context_length,
            'chunk_size': chunk_size,
            'chunk_overlap': chunk_overlap
        }
    }
    
    # Execute pipeline
    result, effects = await pipeline(context)
    
    # Convert results to legacy format
    if isinstance(result, Success):
        legacy_results = []
        for r in result.value:
            legacy_result = {
                'source': r.source,
                'id': r.id,
                'title': r.title,
                'content': r.content,
                'score': r.score,
                'metadata': r.metadata
            }
            if r.citations:
                legacy_result['citations'] = r.citations
            legacy_results.append(legacy_result)
        
        formatted_context = context.get('formatted_output', '')
        return legacy_results, formatted_context
    else:
        logger.error(f"Pipeline execution failed: {result.error.message}")
        return [], ""


# ==============================================================================
# Migration Helpers
# ==============================================================================

def is_using_v2_pipelines() -> bool:
    """Check if the system is configured to use v2 pipelines."""
    import os
    return os.environ.get('USE_V2_PIPELINES', 'false').lower() in ('true', '1', 'yes')


def get_search_function(function_name: str):
    """Get the appropriate search function based on configuration."""
    if is_using_v2_pipelines():
        # Use v2 adapters
        if function_name == 'perform_plain_rag_search':
            return perform_plain_rag_search_v2
        elif function_name == 'perform_full_rag_pipeline':
            return perform_full_rag_pipeline_v2
        elif function_name == 'perform_hybrid_rag_search':
            return perform_hybrid_rag_search_v2
    
    # Use original functions
    from ..Event_Handlers.Chat_Events import chat_rag_events
    return getattr(chat_rag_events, function_name)


# ==============================================================================
# Gradual Migration Support
# ==============================================================================

class MigrationMonitor:
    """Monitor usage of v1 vs v2 pipeline functions."""
    
    def __init__(self):
        self.v1_calls = 0
        self.v2_calls = 0
        self.call_history = []
    
    def record_v1_call(self, function_name: str):
        """Record a call to v1 function."""
        self.v1_calls += 1
        self.call_history.append({
            'version': 'v1',
            'function': function_name,
            'timestamp': time.time()
        })
    
    def record_v2_call(self, function_name: str):
        """Record a call to v2 function."""
        self.v2_calls += 1
        self.call_history.append({
            'version': 'v2',
            'function': function_name,
            'timestamp': time.time()
        })
    
    def get_stats(self) -> Dict[str, Any]:
        """Get migration statistics."""
        total_calls = self.v1_calls + self.v2_calls
        v2_percentage = (self.v2_calls / total_calls * 100) if total_calls > 0 else 0
        
        return {
            'v1_calls': self.v1_calls,
            'v2_calls': self.v2_calls,
            'total_calls': total_calls,
            'v2_percentage': v2_percentage,
            'recent_calls': self.call_history[-10:]  # Last 10 calls
        }


# Global migration monitor
_migration_monitor = MigrationMonitor()


def get_migration_stats() -> Dict[str, Any]:
    """Get current migration statistics."""
    return _migration_monitor.get_stats()


# ==============================================================================
# Monkey Patching Support (for gradual migration)
# ==============================================================================

def enable_v2_pipelines():
    """
    Enable v2 pipelines by monkey patching the existing functions.
    
    This allows gradual migration without changing any calling code.
    """
    from ..Event_Handlers.Chat_Events import chat_rag_events
    
    # Store original functions
    if not hasattr(chat_rag_events, '_original_perform_plain_rag_search'):
        chat_rag_events._original_perform_plain_rag_search = chat_rag_events.perform_plain_rag_search
        chat_rag_events._original_perform_full_rag_pipeline = chat_rag_events.perform_full_rag_pipeline
        chat_rag_events._original_perform_hybrid_rag_search = chat_rag_events.perform_hybrid_rag_search
    
    # Replace with v2 adapters
    chat_rag_events.perform_plain_rag_search = perform_plain_rag_search_v2
    chat_rag_events.perform_full_rag_pipeline = perform_full_rag_pipeline_v2
    chat_rag_events.perform_hybrid_rag_search = perform_hybrid_rag_search_v2
    
    logger.info("V2 pipelines enabled via monkey patching")


def disable_v2_pipelines():
    """
    Disable v2 pipelines and restore original functions.
    """
    from ..Event_Handlers.Chat_Events import chat_rag_events
    
    # Restore original functions
    if hasattr(chat_rag_events, '_original_perform_plain_rag_search'):
        chat_rag_events.perform_plain_rag_search = chat_rag_events._original_perform_plain_rag_search
        chat_rag_events.perform_full_rag_pipeline = chat_rag_events._original_perform_full_rag_pipeline
        chat_rag_events.perform_hybrid_rag_search = chat_rag_events._original_perform_hybrid_rag_search
    
    logger.info("V2 pipelines disabled, original functions restored")


# ==============================================================================
# Performance Comparison
# ==============================================================================

import time
from typing import Any


async def compare_pipeline_performance(
    app: Any,
    query: str,
    sources: Dict[str, bool],
    **kwargs
) -> Dict[str, Any]:
    """
    Compare performance between v1 and v2 pipelines.
    
    Useful for validating that v2 pipelines perform as well as v1.
    """
    from ..Event_Handlers.Chat_Events import chat_rag_events
    
    results = {}
    
    # Test v1 plain search
    start = time.time()
    v1_plain_results, v1_plain_context = await chat_rag_events.perform_plain_rag_search(
        app, query, sources, **kwargs
    )
    v1_plain_time = (time.time() - start) * 1000
    
    # Test v2 plain search
    start = time.time()
    v2_plain_results, v2_plain_context = await perform_plain_rag_search_v2(
        app, query, sources, **kwargs
    )
    v2_plain_time = (time.time() - start) * 1000
    
    results['plain'] = {
        'v1': {
            'time_ms': v1_plain_time,
            'result_count': len(v1_plain_results),
            'context_length': len(v1_plain_context)
        },
        'v2': {
            'time_ms': v2_plain_time,
            'result_count': len(v2_plain_results),
            'context_length': len(v2_plain_context)
        },
        'speedup': v1_plain_time / v2_plain_time if v2_plain_time > 0 else 0
    }
    
    return results