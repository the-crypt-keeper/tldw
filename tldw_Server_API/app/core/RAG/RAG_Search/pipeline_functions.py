"""
Pure pipeline functions for the RAG system.

This module contains all the pure functions used in RAG pipelines. Each function
has clear inputs/outputs and minimal side effects. Functions are organized by
their role: retrieval, processing, formatting, and combination.
"""

import asyncio
from typing import List, Dict, Any, Optional, Tuple, Callable, Awaitable, Set
from dataclasses import dataclass
import hashlib
import time
from loguru import logger

from .pipeline_core import (
    SearchResult, PipelineContext, PipelineError, PipelineErrorType,
    Result, Success, Failure, TypedEffect, EffectType
)
from ..DB.Client_Media_DB_v2 import DatabaseError
from ..Utils.optional_deps import DEPENDENCIES_AVAILABLE

# Check for optional dependencies
RERANK_AVAILABLE = DEPENDENCIES_AVAILABLE.get('rerank', False)
COHERE_AVAILABLE = DEPENDENCIES_AVAILABLE.get('cohere', False)

# ==============================================================================
# Helper Functions
# ==============================================================================

def matches_keyword_filter(item_keywords: List[str], filter_keywords: List[str]) -> bool:
    """
    Check if item has any of the filter keywords (case-insensitive).
    
    Args:
        item_keywords: Keywords/tags associated with the item
        filter_keywords: Keywords to filter by
        
    Returns:
        True if item matches filter (or no filter specified)
    """
    if not filter_keywords:
        return True
    
    # Convert to lowercase for case-insensitive comparison
    item_keywords_lower = [kw.lower() for kw in item_keywords]
    filter_keywords_lower = [kw.lower() for kw in filter_keywords]
    
    # Check if any filter keyword matches any item keyword
    return any(filter_kw in item_keywords_lower for filter_kw in filter_keywords_lower)


def create_search_result(
    source: str,
    id: str,
    title: str,
    content: str,
    score: float = 1.0,
    metadata: Optional[Dict[str, Any]] = None
) -> SearchResult:
    """Create a standardized search result."""
    return SearchResult(
        source=source,
        id=id,
        title=title,
        content=content,
        score=score,
        metadata=metadata or {}
    )


# ==============================================================================
# Retrieval Functions
# ==============================================================================

async def retrieve_fts5(
    context: PipelineContext,
    config: Dict[str, Any]
) -> Result[List[SearchResult], PipelineError]:
    """
    Pure FTS5/BM25 retrieval function.
    
    Searches across media, conversations, and notes using SQLite FTS5.
    
    Args:
        context: Pipeline execution context
        config: Configuration including top_k, keyword_filter, etc.
        
    Returns:
        Success with search results or Failure with error
    """
    try:
        query = context['query']
        sources = context['sources']
        resources = context['resources']
        
        top_k = config.get('top_k', 10)
        keyword_filter = config.get('keyword_filter', [])
        
        all_results = []
        effects = []
        
        # Log the search
        effects.append(TypedEffect.log(
            'info',
            f"Performing FTS5 search for query: '{query}'",
            sources=sources
        ))
        
        # Search Media Items
        if sources.get('media', False) and resources.has_media_db():
            media_results, media_effects = await _search_media_fts5(
                resources.media_db, query, top_k, keyword_filter
            )
            all_results.extend(media_results)
            effects.extend(media_effects)
        
        # Search Conversations
        if sources.get('conversations', False) and resources.has_conversations_db():
            conv_results, conv_effects = await _search_conversations_fts5(
                resources.conversations_db, query, top_k, keyword_filter
            )
            all_results.extend(conv_results)
            effects.extend(conv_effects)
        
        # Search Notes
        if sources.get('notes', False) and resources.has_notes_service():
            notes_results, notes_effects = await _search_notes_fts5(
                resources.notes_service, query, top_k, keyword_filter, context
            )
            all_results.extend(notes_results)
            effects.extend(notes_effects)
        
        # Log metrics
        effects.append(TypedEffect.metric(
            'rag_fts5_search_results',
            len(all_results),
            'gauge'
        ))
        
        # Store effects in context for executor
        context['effects'] = context.get('effects', []) + effects
        
        return Success(all_results)
        
    except Exception as e:
        logger.error(f"FTS5 retrieval error: {e}", exc_info=True)
        return Failure(PipelineError(
            error_type=PipelineErrorType.RETRIEVAL_ERROR,
            message=f"FTS5 search failed: {str(e)}",
            step_name="retrieve_fts5",
            cause=e
        ))


async def _search_media_fts5(
    media_db: Any,
    query: str,
    top_k: int,
    keyword_filter: List[str]
) -> Tuple[List[SearchResult], List[TypedEffect]]:
    """Search media database using FTS5."""
    results = []
    effects = []
    
    try:
        # Search media database
        media_results = await asyncio.to_thread(
            media_db.search_media_db,
            search_query=query,
            search_fields=['title', 'content'],
            page=1,
            results_per_page=top_k * 2,  # Get more for filtering
            include_trash=False
        )
        
        # Extract results list
        if isinstance(media_results, tuple):
            media_items = media_results[0]
        else:
            media_items = media_results
        
        # Fetch keywords in batch
        keywords_map = {}
        if media_items:
            media_ids = [item.get('id') for item in media_items if item.get('id')]
            if media_ids:
                keywords_map = await asyncio.to_thread(
                    media_db.fetch_keywords_for_media_batch,
                    media_ids
                )
        
        # Process results
        for item in media_items:
            item_keywords = keywords_map.get(item.get('id'), [])
            
            if matches_keyword_filter(item_keywords, keyword_filter):
                result = create_search_result(
                    source='media',
                    id=str(item.get('id', '')),
                    title=item.get('title', 'Untitled'),
                    content=item.get('content', ''),
                    score=1.0,
                    metadata={
                        'type': item.get('type', 'unknown'),
                        'author': item.get('author', 'Unknown'),
                        'ingestion_date': item.get('ingestion_date', ''),
                        'keywords': item_keywords
                    }
                )
                results.append(result)
        
        effects.append(TypedEffect.log(
            'debug',
            f"Media search found {len(results)} results"
        ))
        
    except DatabaseError as e:
        effects.append(TypedEffect.log(
            'error',
            f"Media database error: {e}"
        ))
    except Exception as e:
        effects.append(TypedEffect.log(
            'error',
            f"Unexpected error searching media: {e}",
            exc_info=True
        ))
    
    return results, effects


async def _search_conversations_fts5(
    conversations_db: Any,
    query: str,
    top_k: int,
    keyword_filter: List[str]
) -> Tuple[List[SearchResult], List[TypedEffect]]:
    """Search conversations database using FTS5."""
    results = []
    effects = []
    
    try:
        # Search conversations
        conv_results = await asyncio.to_thread(
            conversations_db.search_conversations_by_content,
            search_query=query,
            limit=top_k * 2
        )
        
        for conv in conv_results:
            # Get messages for context
            messages = await asyncio.to_thread(
                conversations_db.get_messages_for_conversation,
                conversation_id=conv['id'],
                limit=5  # Last 5 messages
            )
            
            # Combine messages
            content_parts = []
            for msg in messages:
                content_parts.append(f"{msg['sender']}: {msg['content']}")
            
            # Extract keywords
            item_keywords = []
            if 'keywords' in conv:
                item_keywords.extend(conv.get('keywords', []))
            if 'tags' in conv:
                item_keywords.extend(conv.get('tags', []))
            
            if matches_keyword_filter(item_keywords, keyword_filter):
                result = create_search_result(
                    source='conversation',
                    id=str(conv['id']),
                    title=conv.get('title', 'Untitled Conversation'),
                    content="\n".join(content_parts),
                    score=1.0,
                    metadata={
                        'character_id': conv.get('character_id'),
                        'created_at': conv.get('created_at'),
                        'updated_at': conv.get('updated_at'),
                        'keywords': item_keywords
                    }
                )
                results.append(result)
        
        effects.append(TypedEffect.log(
            'debug',
            f"Conversation search found {len(results)} results"
        ))
        
    except Exception as e:
        effects.append(TypedEffect.log(
            'error',
            f"Error searching conversations: {e}",
            exc_info=True
        ))
    
    return results, effects


async def _search_notes_fts5(
    notes_service: Any,
    query: str,
    top_k: int,
    keyword_filter: List[str],
    context: PipelineContext
) -> Tuple[List[SearchResult], List[TypedEffect]]:
    """Search notes using FTS5."""
    results = []
    effects = []
    
    try:
        # Get user ID from context or app
        user_id = context.get('user_id')
        if not user_id and 'resources' in context:
            app = context['resources'].app
            if hasattr(app, 'notes_user_id'):
                user_id = app.notes_user_id
        
        # Search notes
        note_results = await asyncio.to_thread(
            notes_service.search_notes,
            user_id=user_id,
            search_term=query,
            limit=top_k * 2
        )
        
        for note in note_results:
            # Extract keywords
            item_keywords = []
            if 'keywords' in note:
                item_keywords.extend(note.get('keywords', []))
            if 'tags' in note:
                item_keywords.extend(note.get('tags', []))
            
            if matches_keyword_filter(item_keywords, keyword_filter):
                result = create_search_result(
                    source='note',
                    id=str(note['id']),
                    title=note.get('title', 'Untitled Note'),
                    content=note.get('content', ''),
                    score=1.0,
                    metadata={
                        'created_at': note.get('created_at'),
                        'updated_at': note.get('updated_at'),
                        'tags': note.get('tags', []),
                        'keywords': item_keywords
                    }
                )
                results.append(result)
        
        effects.append(TypedEffect.log(
            'debug',
            f"Notes search found {len(results)} results"
        ))
        
    except Exception as e:
        effects.append(TypedEffect.log(
            'error',
            f"Error searching notes: {e}",
            exc_info=True
        ))
    
    return results, effects


async def retrieve_semantic(
    context: PipelineContext,
    config: Dict[str, Any]
) -> Result[List[SearchResult], PipelineError]:
    """
    Pure semantic/vector retrieval function.
    
    Uses embeddings and vector similarity search.
    """
    try:
        query = context['query']
        resources = context['resources']
        
        # Get or initialize RAG service
        rag_service = resources.embeddings_service
        if not rag_service:
            # Get from resource manager
            from .pipeline_resources import ResourceManager
            if hasattr(resources.app, '_resource_manager'):
                rag_service = await resources.app._resource_manager.get_rag_service()
            else:
                return Failure(PipelineError(
                    error_type=PipelineErrorType.RESOURCE_ERROR,
                    message="RAG service not available",
                    step_name="retrieve_semantic"
                ))
        
        # Perform semantic search
        from .simplified import SearchResultWithCitations
        results = await rag_service.search(
            query=query,
            search_type="semantic",
            top_k=config.get('top_k', 10),
            score_threshold=config.get('score_threshold', 0.0),
            include_citations=config.get('include_citations', True)
        )
        
        # Convert to standard SearchResult format
        search_results = []
        for result in results:
            if isinstance(result, SearchResultWithCitations):
                search_results.append(SearchResult(
                    source=result.source,
                    id=result.id,
                    title=result.title,
                    content=result.content,
                    score=result.score,
                    metadata=result.metadata,
                    citations=[c.to_dict() for c in result.citations] if result.citations else None
                ))
            else:
                # Handle plain result
                search_results.append(SearchResult(**result.to_dict()))
        
        return Success(search_results)
        
    except Exception as e:
        logger.error(f"Semantic retrieval error: {e}", exc_info=True)
        return Failure(PipelineError(
            error_type=PipelineErrorType.RETRIEVAL_ERROR,
            message=f"Semantic search failed: {str(e)}",
            step_name="retrieve_semantic",
            cause=e
        ))


# ==============================================================================
# Processing Functions
# ==============================================================================

def rerank_results(
    results: List[SearchResult],
    context: PipelineContext,
    config: Dict[str, Any]
) -> Result[List[SearchResult], PipelineError]:
    """
    Rerank results using specified model.
    
    Pure function - returns new list without modifying input.
    """
    try:
        if not results:
            return Success(results)
        
        model = config.get('model', 'flashrank')
        top_k = min(config.get('top_k', len(results)), len(results))
        query = context['query']
        
        if model == 'flashrank' and RERANK_AVAILABLE:
            return _rerank_with_flashrank(results, query, top_k)
        elif model == 'cohere' and COHERE_AVAILABLE:
            return _rerank_with_cohere(results, query, top_k, context, config)
        else:
            # No reranking, just limit and sort by score
            sorted_results = sorted(results, key=lambda x: x.score, reverse=True)
            return Success(sorted_results[:top_k])
            
    except Exception as e:
        logger.error(f"Reranking error: {e}", exc_info=True)
        return Failure(PipelineError(
            error_type=PipelineErrorType.PROCESSING_ERROR,
            message=f"Reranking failed: {str(e)}",
            step_name="rerank_results",
            cause=e
        ))


def _rerank_with_flashrank(
    results: List[SearchResult],
    query: str,
    top_k: int
) -> Result[List[SearchResult], PipelineError]:
    """Rerank using FlashRank model."""
    try:
        from flashrank import RerankRequest, Ranker
        
        # Initialize ranker (cached internally)
        ranker = Ranker(model_name="ms-marco-MiniLM-L-12-v2", cache_dir="/tmp")
        
        # Prepare passages
        passages = []
        for result in results:
            text = f"{result.title}\n{result.content[:1000]}"
            passages.append({"text": text})
        
        # Rerank
        rerank_req = RerankRequest(query=query, passages=passages)
        ranked_results = ranker.rerank(rerank_req)
        
        # Build reranked list
        reranked = []
        for ranked in ranked_results[:top_k]:
            idx = getattr(ranked, 'index', 0)
            score = getattr(ranked, 'score', 0.0)
            
            if idx < len(results):
                # Create new result with updated score
                result = results[idx]
                reranked_result = SearchResult(
                    **{**result.dict(), 'score': float(score)}
                )
                reranked.append(reranked_result)
        
        return Success(reranked)
        
    except Exception as e:
        logger.error(f"FlashRank error: {e}")
        # Fallback to original order
        return Success(results[:top_k])


def _rerank_with_cohere(
    results: List[SearchResult],
    query: str,
    top_k: int,
    context: PipelineContext,
    config: Dict[str, Any]
) -> Result[List[SearchResult], PipelineError]:
    """Rerank using Cohere API."""
    try:
        import cohere
        
        # Get API key
        api_key = config.get('cohere_api_key')
        if not api_key:
            # Try from app config
            resources = context.get('resources')
            if resources and resources.app:
                api_key = getattr(resources.app, 'config_dict', {}).get('API', {}).get('cohere_api_key')
        
        if not api_key:
            logger.warning("Cohere API key not found, skipping reranking")
            return Success(results[:top_k])
        
        co = cohere.Client(api_key)
        
        # Prepare documents
        documents = []
        for result in results:
            text = f"{result.title}\n{result.content[:1000]}"
            documents.append(text)
        
        # Rerank
        response = co.rerank(
            query=query,
            documents=documents,
            top_n=top_k,
            model='rerank-english-v2.0'
        )
        
        # Build reranked list
        reranked = []
        for hit in response:
            idx = hit.index
            if idx < len(results):
                result = results[idx]
                reranked_result = SearchResult(
                    **{**result.dict(), 'score': float(hit.relevance_score)}
                )
                reranked.append(reranked_result)
        
        return Success(reranked)
        
    except Exception as e:
        logger.error(f"Cohere reranking error: {e}")
        return Success(results[:top_k])


def deduplicate_results(
    results: List[SearchResult],
    context: PipelineContext,
    config: Dict[str, Any]
) -> Result[List[SearchResult], PipelineError]:
    """
    Remove duplicate results based on content similarity.
    
    Pure function - returns new list.
    """
    try:
        strategy = config.get('strategy', 'content_hash')
        threshold = config.get('threshold', 0.9)
        
        if strategy == 'content_hash':
            seen: Set[int] = set()
            deduped = []
            
            for result in results:
                # Hash first 200 chars of content
                content_hash = hash(result.content[:200])
                if content_hash not in seen:
                    seen.add(content_hash)
                    deduped.append(result)
            
            return Success(deduped)
            
        elif strategy == 'fuzzy':
            # TODO: Implement fuzzy deduplication
            return Success(results)
            
        else:
            return Success(results)
            
    except Exception as e:
        return Failure(PipelineError(
            error_type=PipelineErrorType.PROCESSING_ERROR,
            message=f"Deduplication failed: {str(e)}",
            step_name="deduplicate_results",
            cause=e
        ))


def filter_by_score(
    results: List[SearchResult],
    context: PipelineContext,
    config: Dict[str, Any]
) -> Result[List[SearchResult], PipelineError]:
    """Filter results by minimum score threshold."""
    try:
        min_score = config.get('min_score', 0.0)
        filtered = [r for r in results if r.score >= min_score]
        return Success(filtered)
    except Exception as e:
        return Failure(PipelineError(
            error_type=PipelineErrorType.PROCESSING_ERROR,
            message=f"Score filtering failed: {str(e)}",
            step_name="filter_by_score",
            cause=e
        ))


# ==============================================================================
# Formatting Functions
# ==============================================================================

def format_as_context(
    results: List[SearchResult],
    context: PipelineContext,
    config: Dict[str, Any]
) -> Result[str, PipelineError]:
    """
    Format results as LLM context string.
    
    Pure function - no side effects.
    """
    try:
        max_length = config.get('max_length', 10000)
        include_citations = config.get('include_citations', True)
        separator = config.get('separator', '\n---\n')
        
        context_parts = []
        total_chars = 0
        
        for i, result in enumerate(results):
            # Format header
            if include_citations:
                header = f"[{result.source.upper()} - {result.title}]"
            else:
                header = ""
            
            # Calculate available space
            remaining = max_length - total_chars - len(header) - len(separator)
            if remaining <= 0:
                break
            
            # Truncate content if needed
            content = result.content
            if len(content) > remaining:
                content = content[:remaining] + "..."
            
            # Build result text
            if header:
                result_text = f"{header}\n{content}"
            else:
                result_text = content
            
            context_parts.append(result_text)
            total_chars += len(result_text) + len(separator)
            
            if total_chars >= max_length:
                break
        
        formatted = separator.join(context_parts)
        return Success(formatted)
        
    except Exception as e:
        return Failure(PipelineError(
            error_type=PipelineErrorType.FORMATTING_ERROR,
            message=f"Context formatting failed: {str(e)}",
            step_name="format_as_context",
            cause=e
        ))


def format_as_json(
    results: List[SearchResult],
    context: PipelineContext,
    config: Dict[str, Any]
) -> Result[str, PipelineError]:
    """Format results as JSON string."""
    try:
        import json
        max_results = config.get('max_results', len(results))
        
        # Convert to dict format
        result_dicts = []
        for result in results[:max_results]:
            result_dicts.append(result.dict())
        
        formatted = json.dumps(result_dicts, indent=2)
        return Success(formatted)
        
    except Exception as e:
        return Failure(PipelineError(
            error_type=PipelineErrorType.FORMATTING_ERROR,
            message=f"JSON formatting failed: {str(e)}",
            step_name="format_as_json",
            cause=e
        ))


# ==============================================================================
# Combinator Functions
# ==============================================================================

def parallel(*funcs) -> Callable:
    """
    Create a function that runs functions in parallel.
    
    Returns a retrieval function that executes all provided functions
    in parallel and combines their results.
    """
    async def parallel_executor(
        context: PipelineContext,
        config: Dict[str, Any]
    ) -> Result[List[SearchResult], PipelineError]:
        try:
            # Create tasks for each function
            tasks = []
            for func in funcs:
                task = func(context, config)
                tasks.append(task)
            
            # Execute in parallel
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Combine results
            all_results = []
            errors = []
            
            for i, result in enumerate(results):
                if isinstance(result, Exception):
                    errors.append(f"Function {i} failed: {str(result)}")
                elif isinstance(result, Success):
                    all_results.extend(result.value)
                elif isinstance(result, Failure):
                    errors.append(f"Function {i} error: {result.error.message}")
            
            if errors and not all_results:
                # All functions failed
                return Failure(PipelineError(
                    error_type=PipelineErrorType.RETRIEVAL_ERROR,
                    message=f"Parallel execution failed: {'; '.join(errors)}",
                    step_name="parallel"
                ))
            
            return Success(all_results)
            
        except Exception as e:
            return Failure(PipelineError(
                error_type=PipelineErrorType.RETRIEVAL_ERROR,
                message=f"Parallel execution error: {str(e)}",
                step_name="parallel",
                cause=e
            ))
    
    return parallel_executor


def sequential(*funcs) -> Callable:
    """
    Create a function that runs functions in sequence.
    
    First function gets standard inputs, subsequent functions
    get the previous function's output as results.
    """
    async def sequential_executor(
        context: PipelineContext,
        config: Dict[str, Any]
    ) -> Result[Any, PipelineError]:
        try:
            result = None
            
            for i, func in enumerate(funcs):
                if i == 0:
                    # First function gets normal inputs
                    if asyncio.iscoroutinefunction(func):
                        result = await func(context, config)
                    else:
                        result = func(context, config)
                else:
                    # Subsequent functions process previous results
                    if isinstance(result, Success):
                        if asyncio.iscoroutinefunction(func):
                            result = await func(result.value, context, config)
                        else:
                            result = func(result.value, context, config)
                    else:
                        # Previous step failed
                        return result
            
            return result
            
        except Exception as e:
            return Failure(PipelineError(
                error_type=PipelineErrorType.PROCESSING_ERROR,
                message=f"Sequential execution error: {str(e)}",
                step_name="sequential",
                cause=e
            ))
    
    return sequential_executor


async def weighted_merge(
    results_lists: List[List[SearchResult]],
    weights: List[float]
) -> List[SearchResult]:
    """
    Merge multiple result lists with weighted scores.
    
    Pure function that combines results from multiple sources
    with configurable weights.
    """
    if len(results_lists) != len(weights):
        raise ValueError("Number of result lists must match number of weights")
    
    if not weights:
        return []
    
    # Normalize weights
    total_weight = sum(weights)
    if total_weight == 0:
        return []
    
    weights = [w / total_weight for w in weights]
    
    # Merge with weighted scores
    merged: Dict[str, SearchResult] = {}
    
    for results, weight in zip(results_lists, weights):
        for result in results:
            # Use content prefix as key
            key = f"{result.source}:{result.id}"
            
            if key in merged:
                # Update score with weighted average
                existing = merged[key]
                new_score = existing.score + (result.score * weight)
                merged[key] = SearchResult(
                    **{**existing.dict(), 'score': new_score}
                )
            else:
                # New result with weighted score
                merged[key] = SearchResult(
                    **{**result.dict(), 'score': result.score * weight}
                )
    
    # Sort by final score
    final_results = list(merged.values())
    final_results.sort(key=lambda x: x.score, reverse=True)
    
    return final_results


# ==============================================================================
# Function Registry
# ==============================================================================

PIPELINE_FUNCTIONS = {
    # Retrievers
    'retrieve_fts5': retrieve_fts5,
    'retrieve_semantic': retrieve_semantic,
    
    # Processors
    'rerank_results': rerank_results,
    'deduplicate_results': deduplicate_results,
    'filter_by_score': filter_by_score,
    
    # Formatters
    'format_as_context': format_as_context,
    'format_as_json': format_as_json,
    
    # Combinators
    'parallel': parallel,
    'sequential': sequential,
    'weighted_merge': weighted_merge,
}


def get_function(name: str) -> Callable:
    """Get a pipeline function by name."""
    if name not in PIPELINE_FUNCTIONS:
        raise ValueError(f"Unknown pipeline function: {name}")
    return PIPELINE_FUNCTIONS[name]


def register_function(name: str, func: Callable) -> None:
    """Register a custom pipeline function."""
    if name in PIPELINE_FUNCTIONS:
        logger.warning(f"Overriding existing function: {name}")
    PIPELINE_FUNCTIONS[name] = func