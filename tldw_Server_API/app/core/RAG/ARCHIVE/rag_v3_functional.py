# rag_v3_functional.py - RAG API with Functional Pipeline
"""
RAG API Endpoints using the new functional pipeline architecture.

This module provides:
- Simple, composable function-based pipeline
- Runtime pipeline selection (minimal, standard, quality)
- Custom pipeline configuration
- Full integration with all RAG modules
"""

import asyncio
import time
from typing import Optional, Dict, Any, List, Literal
from uuid import uuid4
from pathlib import Path

from fastapi import APIRouter, HTTPException, Depends, Query, status
from fastapi.responses import StreamingResponse
from loguru import logger
from pydantic import BaseModel, Field

# Dependencies
from tldw_Server_API.app.api.v1.API_Deps.DB_Deps import get_media_db_for_user
from tldw_Server_API.app.api.v1.API_Deps.ChaCha_Notes_DB_Deps import get_chacha_db_for_user
from tldw_Server_API.app.core.AuthNZ.User_DB_Handling import get_request_user, User
from tldw_Server_API.app.core.DB_Management.Media_DB_v2 import MediaDatabase
from tldw_Server_API.app.core.DB_Management.ChaChaNotes_DB import CharactersRAGDB

# Functional Pipeline
from tldw_Server_API.app.core.RAG.rag_service.functional_pipeline import (
    minimal_pipeline,
    standard_pipeline,
    quality_pipeline,
    custom_pipeline,
    build_pipeline,
    RAGPipelineContext,
    # Individual functions for custom pipelines
    expand_query,
    check_cache,
    retrieve_documents,
    optimize_chromadb_search,
    process_tables,
    rerank_documents,
    store_in_cache,
    analyze_performance,
)

from tldw_Server_API.app.core.RAG.rag_service.types import DataSource

# ============= Request/Response Models =============

class SearchRequest(BaseModel):
    """Search request with pipeline selection."""
    query: str = Field(..., description="Search query", min_length=1, max_length=1000)
    pipeline: Literal["minimal", "standard", "quality", "enhanced", "custom"] = Field(
        "standard",
        description="Pipeline to use: minimal (fast), standard (balanced), quality (best), enhanced (with chunking), custom"
    )
    databases: List[str] = Field(
        default=["media_db"],
        description="Databases to search: media_db, notes, characters, chat_history"
    )
    limit: int = Field(10, ge=1, le=100, description="Maximum results to return")
    
    # Configuration for pipeline functions
    config: Dict[str, Any] = Field(
        default_factory=dict,
        description="Pipeline configuration options"
    )
    
    # For custom pipeline
    custom_functions: Optional[List[str]] = Field(
        None,
        description="Function names for custom pipeline (e.g., ['expand_query', 'retrieve_documents'])"
    )
    
    class Config:
        schema_extra = {
            "example": {
                "query": "What is machine learning?",
                "pipeline": "standard",
                "databases": ["media_db", "notes"],
                "limit": 10,
                "config": {
                    "enable_cache": True,
                    "expansion_strategies": ["acronym", "semantic"],
                    "reranking_strategy": "hybrid",
                    "enable_hybrid_search": True,
                    "hybrid_alpha": 0.7
                }
            }
        }


class SearchResult(BaseModel):
    """Individual search result."""
    id: str
    title: str
    content: str
    score: float
    source: str
    metadata: Dict[str, Any] = Field(default_factory=dict)


class SearchResponse(BaseModel):
    """Search response with results and metadata."""
    query: str
    results: List[SearchResult]
    total_results: int
    pipeline_used: str
    cache_hit: bool
    processing_time: float
    metadata: Dict[str, Any] = Field(default_factory=dict)
    
    class Config:
        schema_extra = {
            "example": {
                "query": "machine learning",
                "results": [
                    {
                        "id": "doc_123",
                        "title": "Introduction to ML",
                        "content": "Machine learning is...",
                        "score": 0.95,
                        "source": "media_db",
                        "metadata": {"author": "John Doe"}
                    }
                ],
                "total_results": 1,
                "pipeline_used": "standard",
                "cache_hit": False,
                "processing_time": 0.234,
                "metadata": {
                    "expanded_queries": ["machine learning", "ML"],
                    "reranking_applied": True
                }
            }
        }


class PipelineInfoResponse(BaseModel):
    """Information about available pipelines."""
    available_pipelines: List[str]
    available_functions: List[str]
    recommended_configs: Dict[str, Dict[str, Any]]


# ============= Router Configuration =============

router = APIRouter(
    prefix="/rag/v3",
    tags=["RAG v3 - Functional"],
    responses={
        400: {"description": "Bad Request"},
        500: {"description": "Internal Server Error"},
    }
)

# ============= Helper Functions =============

def map_databases_to_sources(databases: List[str]) -> List[DataSource]:
    """Map database names to DataSource enums."""
    mapping = {
        "media_db": DataSource.MEDIA_DB,
        "media": DataSource.MEDIA_DB,
        "notes": DataSource.NOTES,
        "characters": DataSource.CHARACTER_CARDS,
        "character_cards": DataSource.CHARACTER_CARDS,
        "chat_history": DataSource.CHAT_HISTORY,
        "chats": DataSource.CHAT_HISTORY,
    }
    
    sources = []
    for db in databases:
        if db.lower() in mapping:
            sources.append(mapping[db.lower()])
    
    return sources if sources else [DataSource.MEDIA_DB]


def prepare_pipeline_config(request: SearchRequest, user_id: int) -> Dict[str, Any]:
    """Prepare configuration for pipeline execution."""
    config = request.config.copy()
    
    # Add standard configurations
    config.update({
        "sources": map_databases_to_sources(request.databases),
        "top_k": request.limit,
        "user_id": user_id,
    })
    
    # Set defaults if not provided
    config.setdefault("enable_cache", True)
    config.setdefault("enable_monitoring", True)
    config.setdefault("expansion_strategies", ["acronym", "semantic"])
    config.setdefault("reranking_strategy", "flashrank")
    config.setdefault("table_serialize_method", "hybrid")
    
    # Pipeline-specific defaults
    if request.pipeline == "quality":
        config.setdefault("expansion_strategies", ["acronym", "semantic", "domain", "entity"])
        config.setdefault("reranking_strategy", "hybrid")
        config.setdefault("enable_hybrid_search", True)
        config.setdefault("cache_threshold", 0.9)
    elif request.pipeline == "minimal":
        config["enable_cache"] = False
        config["enable_monitoring"] = False
        config.setdefault("reranking_strategy", "flashrank")
    
    return config


def format_context_results(context: RAGPipelineContext) -> List[SearchResult]:
    """Format pipeline context into search results."""
    results = []
    
    for doc in context.documents:
        results.append(SearchResult(
            id=doc.id or str(uuid4()),
            title=doc.metadata.get("title", "Untitled"),
            content=doc.content[:500],  # Limit content length
            score=doc.score,
            source=doc.source.value if hasattr(doc.source, 'value') else str(doc.source),
            metadata=doc.metadata
        ))
    
    return results


# ============= Available Functions Registry =============

# Import enhanced chunking functions if available
try:
    from tldw_Server_API.app.core.RAG.rag_service.enhanced_chunking_integration import (
        enhanced_chunk_documents,
        filter_chunks_by_type,
        expand_with_parent_context,
        prioritize_by_chunk_type
    )
    ENHANCED_CHUNKING_FUNCS = {
        "enhanced_chunk_documents": enhanced_chunk_documents,
        "filter_chunks_by_type": filter_chunks_by_type,
        "expand_with_parent_context": expand_with_parent_context,
        "prioritize_by_chunk_type": prioritize_by_chunk_type,
    }
except ImportError:
    ENHANCED_CHUNKING_FUNCS = {}
    logger.warning("Enhanced chunking functions not available")

AVAILABLE_FUNCTIONS = {
    "expand_query": expand_query,
    "check_cache": check_cache,
    "retrieve_documents": retrieve_documents,
    "optimize_chromadb_search": optimize_chromadb_search,
    "process_tables": process_tables,
    "rerank_documents": rerank_documents,
    "store_in_cache": store_in_cache,
    "analyze_performance": analyze_performance,
    **ENHANCED_CHUNKING_FUNCS  # Add enhanced chunking functions if available
}


# ============= Main Search Endpoint =============

@router.post(
    "/search",
    response_model=SearchResponse,
    summary="Execute RAG search with functional pipeline",
    description="""
    Perform a RAG search using the functional pipeline architecture.
    
    **Pipeline Options:**
    - `minimal`: Fast, basic search (retrieval + reranking only)
    - `standard`: Balanced with caching and query expansion
    - `quality`: All enhancements for best results
    - `custom`: Build your own pipeline from available functions
    
    **Available Functions for Custom Pipeline:**
    - `expand_query`: Query expansion with multiple strategies
    - `check_cache`: Semantic cache lookup
    - `retrieve_documents`: Document retrieval from databases
    - `optimize_chromadb_search`: ChromaDB hybrid search optimization
    - `process_tables`: Table serialization and processing
    - `rerank_documents`: Advanced document reranking
    - `store_in_cache`: Store results in cache
    - `analyze_performance`: Performance analysis
    
    **Configuration Options:**
    - `enable_cache`: Enable semantic caching (default: true)
    - `expansion_strategies`: List of strategies ["acronym", "semantic", "domain", "entity"]
    - `reranking_strategy`: Strategy for reranking ["flashrank", "diversity", "hybrid"]
    - `enable_hybrid_search`: Enable ChromaDB hybrid search (default: true)
    - `hybrid_alpha`: Balance between vector and FTS (0-1, default: 0.7)
    - `cache_threshold`: Similarity threshold for cache hits (0-1, default: 0.85)
    - `table_serialize_method`: Method for table serialization ["entities", "sentences", "hybrid"]
    """
)
async def search(
    request: SearchRequest,
    current_user: User = Depends(get_request_user),
    media_db: MediaDatabase = Depends(get_media_db_for_user),
    chacha_db: CharactersRAGDB = Depends(get_chacha_db_for_user),
) -> SearchResponse:
    """Execute RAG search with selected pipeline."""
    
    start_time = time.time()
    
    try:
        # Prepare configuration
        config = prepare_pipeline_config(request, current_user.id)
        
        # Add database paths to config
        config["media_db_path"] = Path(media_db.db_path)
        config["chacha_db_path"] = Path(chacha_db.db_path)
        
        # Execute appropriate pipeline
        if request.pipeline == "minimal":
            context = await minimal_pipeline(request.query, config)
            pipeline_used = "minimal"
            
        elif request.pipeline == "standard":
            context = await standard_pipeline(request.query, config)
            pipeline_used = "standard"
            
        elif request.pipeline == "quality":
            context = await quality_pipeline(request.query, config)
            pipeline_used = "quality"
            
        elif request.pipeline == "enhanced":
            # Import enhanced pipeline
            from tldw_Server_API.app.core.RAG.rag_service.functional_pipeline import enhanced_pipeline
            context = await enhanced_pipeline(request.query, config)
            pipeline_used = "enhanced"
            
        elif request.pipeline == "custom":
            if not request.custom_functions:
                raise HTTPException(
                    status_code=400,
                    detail="custom_functions must be specified for custom pipeline"
                )
            
            # Build custom pipeline from function names
            functions = []
            for func_name in request.custom_functions:
                if func_name not in AVAILABLE_FUNCTIONS:
                    raise HTTPException(
                        status_code=400,
                        detail=f"Unknown function: {func_name}"
                    )
                functions.append(AVAILABLE_FUNCTIONS[func_name])
            
            context = await custom_pipeline(request.query, functions, config)
            pipeline_used = f"custom[{','.join(request.custom_functions[:3])}...]"
            
        else:
            raise HTTPException(
                status_code=400,
                detail=f"Unknown pipeline: {request.pipeline}"
            )
        
        # Format results
        results = format_context_results(context)
        
        # Build response
        response = SearchResponse(
            query=request.query,
            results=results,
            total_results=len(results),
            pipeline_used=pipeline_used,
            cache_hit=context.cache_hit,
            processing_time=time.time() - start_time,
            metadata=context.metadata
        )
        
        logger.info(
            f"Search completed for user {current_user.id}: "
            f"query='{request.query[:50]}...', pipeline={pipeline_used}, "
            f"results={len(results)}, cache_hit={context.cache_hit}, "
            f"time={response.processing_time:.3f}s"
        )
        
        return response
        
    except Exception as e:
        logger.error(f"Search failed for user {current_user.id}: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Search failed: {str(e)}"
        )


# ============= Pipeline Information Endpoint =============

@router.get(
    "/pipelines",
    response_model=PipelineInfoResponse,
    summary="Get available pipelines and functions",
    description="Get information about available pipelines and functions for custom pipeline building."
)
async def get_pipeline_info() -> PipelineInfoResponse:
    """Get information about available pipelines and functions."""
    
    return PipelineInfoResponse(
        available_pipelines=["minimal", "standard", "quality", "enhanced", "custom"],
        available_functions=list(AVAILABLE_FUNCTIONS.keys()),
        recommended_configs={
            "minimal": {
                "enable_cache": False,
                "reranking_strategy": "flashrank",
                "top_k": 10
            },
            "standard": {
                "enable_cache": True,
                "expansion_strategies": ["acronym", "semantic"],
                "reranking_strategy": "flashrank",
                "cache_threshold": 0.85,
                "top_k": 10
            },
            "quality": {
                "enable_cache": True,
                "expansion_strategies": ["acronym", "semantic", "domain", "entity"],
                "reranking_strategy": "hybrid",
                "enable_hybrid_search": True,
                "hybrid_alpha": 0.7,
                "cache_threshold": 0.9,
                "table_serialize_method": "hybrid",
                "top_k": 20
            },
            "enhanced": {
                "enable_cache": True,
                "expansion_strategies": ["acronym", "semantic", "domain", "entity"],
                "reranking_strategy": "hybrid",
                "enable_hybrid_search": True,
                "clean_pdf_artifacts": True,
                "preserve_code_blocks": True,
                "preserve_tables": True,
                "structure_aware": True,
                "track_positions": True,
                "chunk_type_filter": False,
                "expand_parent_context": False,
                "chunk_size": 512,
                "chunk_overlap": 128,
                "top_k": 20
            },
            "custom_example": {
                "custom_functions": [
                    "expand_query",
                    "check_cache",
                    "retrieve_documents",
                    "process_tables",
                    "rerank_documents",
                    "analyze_performance"
                ],
                "config": {
                    "enable_cache": True,
                    "expansion_strategies": ["semantic"],
                    "reranking_strategy": "diversity"
                }
            }
        }
    )


# ============= Advanced Search Endpoint =============

@router.post(
    "/search/advanced",
    response_model=SearchResponse,
    summary="Advanced search with full control",
    description="Advanced search endpoint with full control over pipeline composition and configuration."
)
async def advanced_search(
    query: str = Query(..., description="Search query"),
    functions: List[str] = Query(
        default=["expand_query", "check_cache", "retrieve_documents", "rerank_documents"],
        description="Pipeline functions to execute in order"
    ),
    enable_cache: bool = Query(True, description="Enable semantic caching"),
    expansion_strategies: List[str] = Query(
        default=["acronym", "semantic"],
        description="Query expansion strategies"
    ),
    reranking_strategy: str = Query("flashrank", description="Reranking strategy"),
    enable_hybrid_search: bool = Query(True, description="Enable hybrid search"),
    hybrid_alpha: float = Query(0.7, ge=0, le=1, description="Hybrid search alpha"),
    databases: List[str] = Query(default=["media_db"], description="Databases to search"),
    limit: int = Query(10, ge=1, le=100, description="Result limit"),
    current_user: User = Depends(get_request_user),
    media_db: MediaDatabase = Depends(get_media_db_for_user),
    chacha_db: CharactersRAGDB = Depends(get_chacha_db_for_user),
) -> SearchResponse:
    """Execute advanced search with full control."""
    
    start_time = time.time()
    
    # Build configuration
    config = {
        "enable_cache": enable_cache,
        "expansion_strategies": expansion_strategies,
        "reranking_strategy": reranking_strategy,
        "enable_hybrid_search": enable_hybrid_search,
        "hybrid_alpha": hybrid_alpha,
        "sources": map_databases_to_sources(databases),
        "top_k": limit,
        "user_id": current_user.id,
        "media_db_path": Path(media_db.db_path),
        "chacha_db_path": Path(chacha_db.db_path),
    }
    
    # Build function list
    pipeline_functions = []
    for func_name in functions:
        if func_name not in AVAILABLE_FUNCTIONS:
            raise HTTPException(
                status_code=400,
                detail=f"Unknown function: {func_name}"
            )
        pipeline_functions.append(AVAILABLE_FUNCTIONS[func_name])
    
    # Execute pipeline
    context = await custom_pipeline(query, pipeline_functions, config)
    
    # Format results
    results = format_context_results(context)
    
    return SearchResponse(
        query=query,
        results=results,
        total_results=len(results),
        pipeline_used=f"advanced[{','.join(functions[:3])}...]",
        cache_hit=context.cache_hit,
        processing_time=time.time() - start_time,
        metadata=context.metadata
    )


# ============= Streaming Search Endpoint =============

@router.post(
    "/search/stream",
    summary="Streaming search results",
    description="Execute search and stream results as they become available."
)
async def stream_search(
    request: SearchRequest,
    current_user: User = Depends(get_request_user),
    media_db: MediaDatabase = Depends(get_media_db_for_user),
    chacha_db: CharactersRAGDB = Depends(get_chacha_db_for_user),
):
    """Stream search results as they become available."""
    
    async def generate():
        """Generate streaming results."""
        config = prepare_pipeline_config(request, current_user.id)
        config["media_db_path"] = Path(media_db.db_path)
        config["chacha_db_path"] = Path(chacha_db.db_path)
        
        # Execute pipeline
        if request.pipeline == "minimal":
            context = await minimal_pipeline(request.query, config)
        elif request.pipeline == "standard":
            context = await standard_pipeline(request.query, config)
        elif request.pipeline == "quality":
            context = await quality_pipeline(request.query, config)
        else:
            context = await standard_pipeline(request.query, config)
        
        # Stream results
        for doc in context.documents:
            result = SearchResult(
                id=doc.id or str(uuid4()),
                title=doc.metadata.get("title", "Untitled"),
                content=doc.content[:500],
                score=doc.score,
                source=str(doc.source),
                metadata=doc.metadata
            )
            yield f"data: {result.json()}\n\n"
            await asyncio.sleep(0.01)  # Small delay for streaming effect
        
        # Send completion marker
        yield f"data: {{'complete': true, 'total': {len(context.documents)}}}\n\n"
    
    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
        }
    )


# ============= Health Check =============

@router.get(
    "/health",
    summary="Health check",
    description="Check if the RAG service is operational."
)
async def health_check():
    """Check RAG service health."""
    return {
        "status": "healthy",
        "version": "3.0",
        "pipeline": "functional",
        "available_pipelines": ["minimal", "standard", "quality", "custom"],
        "timestamp": time.time()
    }