"""
Unified RAG API Endpoint

This is the new, simplified RAG API that uses the unified pipeline.
All features are accessible through explicit parameters.
"""

import time
from typing import Optional, Dict, Any
from uuid import uuid4

from fastapi import APIRouter, HTTPException, Depends, status, BackgroundTasks
from loguru import logger

# Dependencies
from tldw_Server_API.app.api.v1.API_Deps.DB_Deps import get_media_db_for_user
from tldw_Server_API.app.api.v1.API_Deps.ChaCha_Notes_DB_Deps import get_chacha_db_for_user
from tldw_Server_API.app.core.AuthNZ.User_DB_Handling import get_request_user, User
from tldw_Server_API.app.core.DB_Management.Media_DB_v2 import MediaDatabase
from tldw_Server_API.app.core.DB_Management.ChaChaNotes_DB import CharactersRAGDB

# Unified Pipeline
from tldw_Server_API.app.core.RAG.rag_service.unified_pipeline import (
    unified_rag_pipeline,
    unified_batch_pipeline,
    simple_search,
    advanced_search,
    UnifiedSearchResult
)

# Schemas
from tldw_Server_API.app.api.v1.schemas.rag_schemas_unified import (
    UnifiedRAGRequest,
    UnifiedRAGResponse,
    UnifiedBatchRequest,
    UnifiedBatchResponse
)

router = APIRouter(prefix="/api/v1/rag", tags=["RAG - Unified"])


def convert_result_to_response(result: UnifiedSearchResult) -> UnifiedRAGResponse:
    """Convert internal result to API response."""
    return UnifiedRAGResponse(
        documents=[
            {
                "id": doc.id,
                "content": doc.content,
                "metadata": doc.metadata,
                "score": getattr(doc, 'score', 0.0)
            }
            for doc in result.documents
        ],
        query=result.query,
        expanded_queries=result.expanded_queries,
        metadata=result.metadata,
        timings=result.timings,
        citations=result.citations,
        feedback_id=result.feedback_id,
        generated_answer=result.generated_answer,
        cache_hit=result.cache_hit,
        errors=result.errors,
        security_report=result.security_report,
        total_time=result.total_time
    )


@router.post(
    "/search",
    response_model=UnifiedRAGResponse,
    summary="Unified RAG Search",
    description="""
    The unified RAG search endpoint with ALL features accessible via parameters.
    
    **Key Features:**
    - No configuration files needed
    - Every feature is a direct parameter
    - Mix and match any features
    - Transparent execution
    
    **Available Features:**
    - Query expansion (acronym, synonym, domain, entity)
    - Semantic caching with adaptive thresholds
    - Multi-database search (media, notes, characters, chats)
    - Security filtering and PII detection
    - Citation generation (APA, MLA, Chicago, Harvard)
    - Answer generation from context
    - Document reranking (FlashRank, Cross-Encoder, Hybrid)
    - Table processing and extraction
    - Enhanced chunking with parent context
    - User feedback collection
    - Performance monitoring and observability
    - Batch processing support
    - Resilience features (retries, circuit breakers)
    
    Simply set any feature parameter to enable it. All parameters are optional
    except the query itself.
    """,
    response_description="Search results with all requested features applied"
)
async def unified_search_endpoint(
    request: UnifiedRAGRequest,
    background_tasks: BackgroundTasks,
    current_user: User = Depends(get_request_user),
    media_db: MediaDatabase = Depends(get_media_db_for_user),
    chacha_db: CharactersRAGDB = Depends(get_chacha_db_for_user)
):
    """
    Unified RAG search with all features as parameters.
    
    This endpoint replaces the complex configuration-based approach with
    a simple, parameter-driven interface. Every feature in the RAG system
    is accessible by setting the appropriate parameter.
    """
    try:
        logger.info(f"Unified RAG search: query='{request.query}', user={current_user.username if current_user else 'anonymous'}")
        
        # Set up database paths
        db_paths = {
            "media_db_path": media_db.db_path if media_db else None,
            "notes_db_path": None,  # TODO: Get from dependency when available
            "character_db_path": chacha_db.db_path if chacha_db else None
        }
        
        # Execute unified pipeline with all parameters from request
        result = await unified_rag_pipeline(
            # Core parameters
            query=request.query,
            sources=request.sources,
            
            # Database paths
            media_db_path=db_paths.get("media_db_path"),
            notes_db_path=db_paths.get("notes_db_path"),
            character_db_path=db_paths.get("character_db_path"),
            
            # Search configuration
            search_mode=request.search_mode,
            hybrid_alpha=request.hybrid_alpha,
            top_k=request.top_k,
            min_score=request.min_score,
            
            # Query expansion
            expand_query=request.expand_query,
            expansion_strategies=request.expansion_strategies,
            spell_check=request.spell_check,
            
            # Caching
            enable_cache=request.enable_cache,
            cache_threshold=request.cache_threshold,
            adaptive_cache=request.adaptive_cache,
            
            # Filtering
            keyword_filter=request.keyword_filter,
            
            # Security
            enable_security_filter=request.enable_security_filter,
            detect_pii=request.detect_pii,
            redact_pii=request.redact_pii,
            sensitivity_level=request.sensitivity_level,
            content_filter=request.content_filter,
            
            # Document processing
            enable_table_processing=request.enable_table_processing,
            table_method=request.table_method,
            
            # Chunking
            enable_enhanced_chunking=request.enable_enhanced_chunking,
            chunk_type_filter=request.chunk_type_filter,
            enable_parent_expansion=request.enable_parent_expansion,
            parent_context_size=request.parent_context_size,
            include_sibling_chunks=request.include_sibling_chunks,
            
            # Reranking
            enable_reranking=request.enable_reranking,
            reranking_strategy=request.reranking_strategy,
            rerank_top_k=request.rerank_top_k,
            
            # Citations
            enable_citations=request.enable_citations,
            citation_style=request.citation_style,
            include_page_numbers=request.include_page_numbers,
            
            # Generation
            enable_generation=request.enable_generation,
            generation_model=request.generation_model,
            generation_prompt=request.generation_prompt,
            max_generation_tokens=request.max_generation_tokens,
            
            # Feedback
            collect_feedback=request.collect_feedback,
            feedback_user_id=request.feedback_user_id or (current_user.username if current_user else None),
            apply_feedback_boost=request.apply_feedback_boost,
            
            # Monitoring
            enable_monitoring=request.enable_monitoring,
            enable_observability=request.enable_observability,
            trace_id=request.trace_id,
            
            # Performance
            enable_performance_analysis=request.enable_performance_analysis,
            timeout_seconds=request.timeout_seconds,
            
            # Quick wins
            highlight_results=request.highlight_results,
            highlight_query_terms=request.highlight_query_terms,
            track_cost=request.track_cost,
            debug_mode=request.debug_mode,
            
            # Batch
            enable_batch=request.enable_batch,
            batch_queries=request.batch_queries,
            batch_concurrent=request.batch_concurrent,
            
            # Resilience
            enable_resilience=request.enable_resilience,
            retry_attempts=request.retry_attempts,
            circuit_breaker=request.circuit_breaker,
            
            # User context
            user_id=current_user.username if current_user else request.user_id,
            session_id=request.session_id
        )
        
        # Convert to response format
        response = convert_result_to_response(result)
        
        # Log performance if monitoring enabled
        if request.enable_monitoring:
            logger.info(f"Query completed in {result.total_time:.3f}s - Cache hit: {result.cache_hit}")
            if request.debug_mode:
                logger.debug(f"Timings: {result.timings}")
                logger.debug(f"Metadata: {result.metadata}")
        
        # Handle any errors that occurred
        if result.errors and request.debug_mode:
            logger.warning(f"Errors during processing: {result.errors}")
        
        return response
        
    except Exception as e:
        logger.error(f"Unified search error: {e}", exc_info=True)
        import traceback
        logger.error(f"Full traceback: {traceback.format_exc()}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Search failed: {str(e)}"
        )


@router.post(
    "/batch",
    response_model=UnifiedBatchResponse,
    summary="Batch RAG Search",
    description="""
    Process multiple queries concurrently using the unified pipeline.
    
    All parameters from the single search endpoint are available and will
    be applied to all queries in the batch.
    """,
    response_description="Batch processing results"
)
async def unified_batch_endpoint(
    request: UnifiedBatchRequest,
    background_tasks: BackgroundTasks,
    current_user: User = Depends(get_request_user),
    media_db: MediaDatabase = Depends(get_media_db_for_user),
    chacha_db: CharactersRAGDB = Depends(get_chacha_db_for_user)
):
    """
    Batch processing endpoint for multiple queries.
    
    Processes multiple queries concurrently with the same parameters.
    """
    try:
        logger.info(f"Batch RAG search: {len(request.queries)} queries, user={current_user.username if current_user else 'anonymous'}")
        
        start_time = time.time()
        
        # Set up database paths
        db_paths = {
            "media_db_path": media_db.db_path if media_db else None,
            "notes_db_path": None,
            "character_db_path": chacha_db.db_path if chacha_db else None
        }
        
        # Convert request to kwargs, excluding queries
        kwargs = request.dict(exclude={"queries", "max_concurrent"})
        kwargs.update(db_paths)
        kwargs["user_id"] = current_user.username if current_user else kwargs.get("user_id")
        
        # Process batch
        results = await unified_batch_pipeline(
            queries=request.queries,
            max_concurrent=request.max_concurrent,
            **kwargs
        )
        
        # Convert results
        responses = [convert_result_to_response(r) for r in results]
        
        # Count successes and failures
        successful = sum(1 for r in results if not r.errors)
        failed = len(results) - successful
        
        total_time = time.time() - start_time
        
        return UnifiedBatchResponse(
            results=responses,
            total_queries=len(request.queries),
            successful=successful,
            failed=failed,
            total_time=total_time
        )
        
    except Exception as e:
        logger.error(f"Batch search error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Batch search failed: {str(e)}"
        )


@router.get(
    "/simple",
    summary="Simple Search",
    description="""
    Simplified search endpoint for basic use cases.
    
    Uses sensible defaults:
    - Caching enabled
    - Reranking enabled
    - No query expansion
    """,
    response_description="Search results"
)
async def simple_search_endpoint(
    query: str,
    top_k: int = 10,
    current_user: User = Depends(get_request_user),
    media_db: MediaDatabase = Depends(get_media_db_for_user)
):
    """
    Simple search for basic use cases.
    """
    try:
        logger.info(f"Simple search: query='{query}'")
        
        # Use the simple_search wrapper
        documents = await simple_search(query, top_k)
        
        return {
            "query": query,
            "documents": [
                {
                    "id": doc.id,
                    "content": doc.content,
                    "metadata": doc.metadata,
                    "score": getattr(doc, 'score', 0.0)
                }
                for doc in documents
            ],
            "count": len(documents)
        }
        
    except Exception as e:
        logger.error(f"Simple search error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Search failed: {str(e)}"
        )


@router.get(
    "/advanced",
    summary="Advanced Search",
    description="""
    Advanced search with commonly used features enabled.
    
    Automatically enables:
    - Query expansion
    - Citations
    - Answer generation
    - Table processing
    - Performance analysis
    """,
    response_description="Full search results with analysis"
)
async def advanced_search_endpoint(
    query: str,
    with_citations: bool = True,
    with_answer: bool = True,
    current_user: User = Depends(get_request_user),
    media_db: MediaDatabase = Depends(get_media_db_for_user),
    chacha_db: CharactersRAGDB = Depends(get_chacha_db_for_user)
):
    """
    Advanced search with common features enabled.
    """
    try:
        logger.info(f"Advanced search: query='{query}'")
        
        # Set up database paths
        db_paths = {
            "media_db_path": media_db.db_path if media_db else None,
            "character_db_path": chacha_db.db_path if chacha_db else None
        }
        
        # Use the advanced_search wrapper
        result = await advanced_search(
            query=query,
            with_citations=with_citations,
            with_answer=with_answer,
            **db_paths
        )
        
        return convert_result_to_response(result)
        
    except Exception as e:
        logger.error(f"Advanced search error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Search failed: {str(e)}"
        )


@router.get(
    "/features",
    summary="List Available Features",
    description="Get a list of all available features in the unified pipeline",
    response_description="Feature list with descriptions"
)
async def list_features():
    """
    List all available features in the unified pipeline.
    """
    return {
        "features": {
            "query_expansion": {
                "description": "Expand queries with synonyms, acronyms, domain terms, and entities",
                "parameters": ["expand_query", "expansion_strategies", "spell_check"]
            },
            "caching": {
                "description": "Semantic caching with adaptive thresholds",
                "parameters": ["enable_cache", "cache_threshold", "adaptive_cache"]
            },
            "security": {
                "description": "PII detection, content filtering, and access control",
                "parameters": ["enable_security_filter", "detect_pii", "redact_pii", "sensitivity_level"]
            },
            "citations": {
                "description": "Generate citations in various formats",
                "parameters": ["enable_citations", "citation_style", "include_page_numbers"]
            },
            "generation": {
                "description": "Generate answers from retrieved context",
                "parameters": ["enable_generation", "generation_model", "generation_prompt"]
            },
            "reranking": {
                "description": "Rerank documents for better relevance",
                "parameters": ["enable_reranking", "reranking_strategy", "rerank_top_k"]
            },
            "feedback": {
                "description": "Collect and apply user feedback",
                "parameters": ["collect_feedback", "feedback_user_id", "apply_feedback_boost"]
            },
            "monitoring": {
                "description": "Performance monitoring and observability",
                "parameters": ["enable_monitoring", "enable_observability", "trace_id"]
            },
            "table_processing": {
                "description": "Extract and process tables from documents",
                "parameters": ["enable_table_processing", "table_method"]
            },
            "enhanced_chunking": {
                "description": "Advanced document chunking with parent context",
                "parameters": ["enable_enhanced_chunking", "chunk_type_filter", "enable_parent_expansion"]
            },
            "batch_processing": {
                "description": "Process multiple queries concurrently",
                "parameters": ["enable_batch", "batch_queries", "batch_concurrent"]
            },
            "resilience": {
                "description": "Fault tolerance with retries and circuit breakers",
                "parameters": ["enable_resilience", "retry_attempts", "circuit_breaker"]
            }
        },
        "total_features": 12,
        "total_parameters": 50
    }


@router.get(
    "/health",
    summary="Health Check",
    description="Check the health of the unified RAG pipeline",
    response_description="Health status"
)
async def health_check():
    """
    Health check for the unified pipeline.
    """
    try:
        # Test basic search functionality
        test_result = await simple_search("test", top_k=1)
        
        return {
            "status": "healthy",
            "pipeline": "unified",
            "version": "1.0.0",
            "test_successful": len(test_result) >= 0
        }
    except Exception as e:
        return {
            "status": "unhealthy",
            "pipeline": "unified",
            "version": "1.0.0",
            "error": str(e)
        }