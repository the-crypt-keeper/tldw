# rag_api.py - Simple and Complex RAG API Endpoints
"""
Production RAG API with both simple and complex interfaces.

Simple API: Easy to use with common parameters
Complex API: Full control with extensibility for future options
"""

import asyncio
import time
from typing import Optional, Dict, Any, List, Union
from uuid import uuid4
from pathlib import Path

from fastapi import APIRouter, HTTPException, Depends, Body, status, Request
from loguru import logger
from pydantic import BaseModel, Field, validator
from slowapi import Limiter
from slowapi.util import get_remote_address

# Dependencies
from tldw_Server_API.app.api.v1.API_Deps.DB_Deps import get_media_db_for_user
from tldw_Server_API.app.api.v1.API_Deps.ChaCha_Notes_DB_Deps import get_chacha_db_for_user
from tldw_Server_API.app.core.AuthNZ.User_DB_Handling import get_request_user, User
from tldw_Server_API.app.core.DB_Management.Media_DB_v2 import MediaDatabase
from tldw_Server_API.app.core.DB_Management.ChaChaNotes_DB import CharactersRAGDB

# Functional Pipeline
from tldw_Server_API.app.core.RAG.rag_service.functional_pipeline import (
    RAGPipelineContext,
    build_pipeline,
    expand_query,
    check_cache,
    retrieve_documents,
    optimize_chromadb_search,
    process_tables,
    rerank_documents,
    store_in_cache,
    analyze_performance,
)
from tldw_Server_API.app.core.RAG.rag_service.enhanced_chunking_integration import (
    expand_with_parent_context
)

from tldw_Server_API.app.core.RAG.rag_service.types import DataSource, Document

# ============= Simple API Models =============

class SimpleSearchRequest(BaseModel):
    """Simple search request with essential parameters."""
    
    # Required
    query: str = Field(
        ...,
        description="Search query",
        min_length=1,
        max_length=1000,
        example="What is machine learning?"
    )
    
    # Database selection
    databases: List[str] = Field(
        default=["media"],
        description="Databases to search: media, notes, characters, chats",
        example=["media", "notes"]
    )
    
    # Core options with defaults
    max_context_size: int = Field(
        default=4000,
        ge=100,
        le=50000,
        description="Maximum total size of returned content in characters",
        example=4000
    )
    
    top_k: int = Field(
        default=10,
        ge=1,
        le=100,
        description="Number of top results to return",
        example=10
    )
    
    enable_reranking: bool = Field(
        default=True,
        description="Enable document reranking for better relevance",
        example=True
    )
    
    enable_citations: bool = Field(
        default=False,
        description="Include citations with source information",
        example=True
    )
    
    # Keyword filtering
    keywords: Optional[List[str]] = Field(
        default=None,
        description="Filter results by keywords (optional)",
        example=["python", "api"]
    )
    
    # Contextual Retrieval Options
    enable_contextual_retrieval: bool = Field(
        default=False,
        description="Enable contextual retrieval with parent document expansion",
        example=False
    )
    
    parent_expansion_size: Optional[int] = Field(
        default=None,
        ge=100,
        le=2000,
        description="Size of parent context to include (in characters) when contextual retrieval is enabled",
        example=500
    )
    
    include_sibling_chunks: bool = Field(
        default=False,
        description="Include adjacent chunks from the same document for continuity",
        example=False
    )
    
    class Config:
        schema_extra = {
            "example": {
                "query": "explain neural networks",
                "databases": ["media", "notes"],
                "max_context_size": 4000,
                "top_k": 10,
                "enable_reranking": True,
                "enable_citations": True,
                "keywords": ["deep learning", "AI"]
            }
        }


class Citation(BaseModel):
    """Citation information for a document."""
    source_id: str
    source_type: str  # media, note, character, chat
    title: Optional[str]
    author: Optional[str]
    timestamp: Optional[str]
    url: Optional[str]
    page: Optional[int]
    metadata: Dict[str, Any] = Field(default_factory=dict)


class SimpleSearchResult(BaseModel):
    """Simple search result with optional citations."""
    content: str
    score: float
    source: str
    citation: Optional[Citation] = None
    
    class Config:
        schema_extra = {
            "example": {
                "content": "Neural networks are computational models...",
                "score": 0.95,
                "source": "media",
                "citation": {
                    "source_id": "doc_123",
                    "source_type": "media",
                    "title": "Introduction to Neural Networks",
                    "author": "John Doe",
                    "timestamp": "2024-01-15T10:30:00Z"
                }
            }
        }


class SimpleSearchResponse(BaseModel):
    """Simple search response."""
    query: str
    results: List[SimpleSearchResult]
    total_context_size: int
    databases_searched: List[str]
    processing_time: float
    
    class Config:
        schema_extra = {
            "example": {
                "query": "neural networks",
                "results": [{
                    "content": "Neural networks are...",
                    "score": 0.95,
                    "source": "media",
                    "citation": None
                }],
                "total_context_size": 1523,
                "databases_searched": ["media", "notes"],
                "processing_time": 0.234
            }
        }


# ============= Complex API Models =============

class ComplexSearchRequest(BaseModel):
    """Complex search request with full configurability."""
    
    # Query parameters
    query: str = Field(..., description="Search query")
    query_expansion: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Query expansion configuration",
        example={
            "enabled": True,
            "strategies": ["acronym", "semantic", "domain", "entity"],
            "max_expansions": 5
        }
    )
    
    # Database configuration
    databases: Dict[str, Dict[str, Any]] = Field(
        default={"media": {"enabled": True}},
        description="Database configuration with per-database options",
        example={
            "media": {"enabled": True, "weight": 1.0},
            "notes": {"enabled": True, "weight": 0.8},
            "characters": {"enabled": False}
        }
    )
    
    # Retrieval configuration
    retrieval: Dict[str, Any] = Field(
        default_factory=dict,
        description="Retrieval configuration",
        example={
            "vector_search": {
                "enabled": True,
                "top_k": 50,
                "similarity_threshold": 0.7
            },
            "fts_search": {
                "enabled": True,
                "top_k": 50,
                "bm25_params": {"k1": 1.2, "b": 0.75}
            },
            "hybrid_search": {
                "enabled": True,
                "alpha": 0.7,
                "normalization": "min-max"
            }
        }
    )
    
    # Processing configuration
    processing: Dict[str, Any] = Field(
        default_factory=dict,
        description="Document processing configuration",
        example={
            "max_context_size": 10000,
            "chunking": {
                "enabled": True,
                "chunk_size": 500,
                "overlap": 50
            },
            "table_processing": {
                "enabled": True,
                "method": "hybrid"
            },
            "deduplication": {
                "enabled": True,
                "threshold": 0.9
            },
            "contextual_retrieval": {
                "enabled": False,
                "parent_expansion_size": 500,
                "include_siblings": False,
                "context_window_size": 500
            }
        }
    )
    
    # Reranking configuration
    reranking: Dict[str, Any] = Field(
        default_factory=dict,
        description="Reranking configuration",
        example={
            "enabled": True,
            "strategy": "hybrid",
            "top_k": 10,
            "diversity_weight": 0.3,
            "relevance_weight": 0.7,
            "use_llm_scoring": False
        }
    )
    
    # Filtering configuration
    filters: Dict[str, Any] = Field(
        default_factory=dict,
        description="Filtering configuration",
        example={
            "keywords": {
                "include": ["python", "api"],
                "exclude": ["deprecated"],
                "mode": "any"  # any, all
            },
            "date_range": {
                "start": "2024-01-01",
                "end": "2024-12-31"
            },
            "metadata": {
                "author": "John Doe",
                "type": "tutorial"
            }
        }
    )
    
    # Caching configuration
    caching: Dict[str, Any] = Field(
        default_factory=dict,
        description="Caching configuration",
        example={
            "enabled": True,
            "semantic_cache": {
                "enabled": True,
                "threshold": 0.85,
                "ttl": 3600
            },
            "result_cache": {
                "enabled": True,
                "ttl": 1800
            }
        }
    )
    
    # Citation configuration
    citations: Dict[str, Any] = Field(
        default_factory=dict,
        description="Citation configuration",
        example={
            "enabled": True,
            "format": "detailed",  # minimal, standard, detailed
            "include_metadata": True,
            "include_snippets": True,
            "snippet_length": 200
        }
    )
    
    # Performance configuration
    performance: Dict[str, Any] = Field(
        default_factory=dict,
        description="Performance configuration",
        example={
            "timeout": 30,
            "parallel_processing": True,
            "batch_size": 100,
            "enable_monitoring": True,
            "enable_profiling": False
        }
    )
    
    # Extensible options for future features
    extensions: Dict[str, Any] = Field(
        default_factory=dict,
        description="Extension configuration for future features"
    )
    
    class Config:
        schema_extra = {
            "example": {
                "query": "advanced machine learning techniques",
                "query_expansion": {
                    "enabled": True,
                    "strategies": ["semantic", "domain"]
                },
                "databases": {
                    "media": {"enabled": True, "weight": 1.0},
                    "notes": {"enabled": True, "weight": 0.8}
                },
                "retrieval": {
                    "hybrid_search": {
                        "enabled": True,
                        "alpha": 0.7
                    }
                },
                "reranking": {
                    "enabled": True,
                    "strategy": "hybrid",
                    "top_k": 10
                },
                "citations": {
                    "enabled": True,
                    "format": "detailed"
                }
            }
        }


class ComplexSearchResult(BaseModel):
    """Complex search result with full metadata."""
    id: str
    content: str
    score: float
    source: str
    metadata: Dict[str, Any]
    citation: Optional[Citation]
    processing_info: Dict[str, Any]
    debug_info: Optional[Dict[str, Any]] = None


class ComplexSearchResponse(BaseModel):
    """Complex search response with detailed information."""
    request_id: str
    query: str
    results: List[ComplexSearchResult]
    metadata: Dict[str, Any]
    performance: Dict[str, Any]
    debug: Optional[Dict[str, Any]] = None


# ============= Router Configuration =============

router = APIRouter(
    prefix="/api/v1/rag",
    tags=["RAG API"],
    responses={
        400: {"description": "Bad Request"},
        500: {"description": "Internal Server Error"},
    }
)

# Initialize rate limiter
limiter = Limiter(key_func=get_remote_address)

# ============= Helper Functions =============

def map_databases(databases: Union[List[str], Dict[str, Dict[str, Any]]]) -> List[DataSource]:
    """Map database names to DataSource enums."""
    mapping = {
        "media": DataSource.MEDIA_DB,
        "media_db": DataSource.MEDIA_DB,
        "notes": DataSource.NOTES,
        "characters": DataSource.CHARACTER_CARDS,
        "character_cards": DataSource.CHARACTER_CARDS,
        "chats": DataSource.CHAT_HISTORY,
        "chat_history": DataSource.CHAT_HISTORY,
    }
    
    if isinstance(databases, list):
        sources = []
        for db in databases:
            if db.lower() in mapping:
                sources.append(mapping[db.lower()])
        return sources if sources else [DataSource.MEDIA_DB]
    else:
        # Handle dict format from complex API
        sources = []
        for db_name, config in databases.items():
            if config.get("enabled", True) and db_name.lower() in mapping:
                sources.append(mapping[db_name.lower()])
        return sources if sources else [DataSource.MEDIA_DB]


def truncate_to_context_size(documents: List[Document], max_size: int) -> List[Document]:
    """Truncate documents to fit within max context size."""
    total_size = 0
    truncated_docs = []
    
    for doc in documents:
        doc_size = len(doc.content)
        if total_size + doc_size <= max_size:
            truncated_docs.append(doc)
            total_size += doc_size
        else:
            # Partial inclusion of last document
            remaining = max_size - total_size
            if remaining > 100:  # Only include if meaningful amount
                doc.content = doc.content[:remaining]
                truncated_docs.append(doc)
            break
    
    return truncated_docs


def create_citation(doc: Document, include_metadata: bool = True) -> Citation:
    """Create citation from document."""
    return Citation(
        source_id=doc.id or str(uuid4()),
        source_type=doc.source.value if hasattr(doc.source, 'value') else str(doc.source),
        title=doc.metadata.get("title"),
        author=doc.metadata.get("author"),
        timestamp=doc.metadata.get("timestamp"),
        url=doc.metadata.get("url"),
        page=doc.metadata.get("page"),
        metadata=doc.metadata if include_metadata else {}
    )


# ============= Simple API Endpoint =============

@router.post(
    "/search/simple",
    response_model=SimpleSearchResponse,
    summary="Simple RAG search",
    description="""
    Simple RAG search with essential parameters.
    
    **Features:**
    - Multi-database search (media, notes, characters, chats)
    - Automatic context size limiting
    - Optional reranking for better relevance
    - Optional citations with source information
    - Keyword filtering support
    
    This endpoint provides a balance between simplicity and control,
    suitable for most use cases without complex configuration.
    """
)
@limiter.limit("30/minute")  # Rate limit: 30 simple searches per minute
async def simple_search(
    request: SimpleSearchRequest,
    req: Request,  # Required for rate limiter
    current_user: User = Depends(get_request_user),
    media_db: MediaDatabase = Depends(get_media_db_for_user),
    chacha_db: CharactersRAGDB = Depends(get_chacha_db_for_user),
) -> SimpleSearchResponse:
    """Execute simple RAG search."""
    
    start_time = time.time()
    
    try:
        # Build configuration
        config = {
            "sources": map_databases(request.databases),
            "top_k": request.top_k * 3 if request.enable_reranking else request.top_k,  # Get more for reranking
            "enable_cache": True,
            "expansion_strategies": ["acronym", "semantic"],
            "enable_hybrid_search": True,
            "hybrid_alpha": 0.7,
            "user_id": current_user.id,
            "media_db_path": Path(media_db.db_path),
            "chacha_db_path": Path(chacha_db.db_path),
        }
        
        # Add contextual retrieval options
        if request.enable_contextual_retrieval:
            config["expand_parent_context"] = True
            config["parent_expansion_size"] = request.parent_expansion_size or 500
            config["include_siblings"] = request.include_sibling_chunks
        
        # Add keyword filters if provided
        if request.keywords:
            config["keyword_filters"] = request.keywords
        
        # Build pipeline based on options
        pipeline_functions = [
            expand_query,
            check_cache,
        ]
        
        # Add retrieval with optimization
        pipeline_functions.extend([
            optimize_chromadb_search,
            retrieve_documents,
        ])
        
        # Add contextual expansion if enabled
        if request.enable_contextual_retrieval:
            pipeline_functions.append(expand_with_parent_context)
        
        # Add reranking if enabled
        if request.enable_reranking:
            config["reranking_strategy"] = "flashrank"  # Fast reranking for simple API
            config["top_k"] = request.top_k
            pipeline_functions.append(rerank_documents)
        
        # Add cache storage
        pipeline_functions.append(store_in_cache)
        
        # Build and execute pipeline
        pipeline = build_pipeline(*pipeline_functions)
        context = await pipeline(request.query, config)
        
        # Truncate to context size
        truncated_docs = truncate_to_context_size(
            context.documents,
            request.max_context_size
        )
        
        # Format results
        results = []
        total_size = 0
        
        for doc in truncated_docs:
            result = SimpleSearchResult(
                content=doc.content,
                score=doc.score,
                source=doc.source.value if hasattr(doc.source, 'value') else str(doc.source),
                citation=create_citation(doc) if request.enable_citations else None
            )
            results.append(result)
            total_size += len(doc.content)
        
        return SimpleSearchResponse(
            query=request.query,
            results=results,
            total_context_size=total_size,
            databases_searched=request.databases,
            processing_time=time.time() - start_time
        )
        
    except Exception as e:
        logger.error(f"Simple search failed: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Search failed: {str(e)}"
        )


# ============= Complex API Endpoint =============

@router.post(
    "/search/complex",
    response_model=ComplexSearchResponse,
    summary="Complex RAG search with full control",
    description="""
    Complex RAG search with complete configurability.
    
    **Features:**
    - Full control over all pipeline components
    - Per-database configuration
    - Advanced retrieval options (vector, FTS, hybrid)
    - Multiple reranking strategies
    - Comprehensive filtering options
    - Detailed citations and metadata
    - Performance monitoring and profiling
    - Extensible for future features
    
    This endpoint is designed for advanced use cases requiring
    fine-grained control over the search process.
    """
)
@limiter.limit("10/minute")  # Rate limit: 10 complex searches per minute (more resource intensive)
async def complex_search(
    request: ComplexSearchRequest,
    req: Request,  # Required for rate limiter
    current_user: User = Depends(get_request_user),
    media_db: MediaDatabase = Depends(get_media_db_for_user),
    chacha_db: CharactersRAGDB = Depends(get_chacha_db_for_user),
) -> ComplexSearchResponse:
    """Execute complex RAG search with full configurability."""
    
    start_time = time.time()
    request_id = str(uuid4())
    
    try:
        # Build comprehensive configuration
        config = {
            "request_id": request_id,
            "sources": map_databases(request.databases),
            "user_id": current_user.id,
            "media_db_path": Path(media_db.db_path),
            "chacha_db_path": Path(chacha_db.db_path),
        }
        
        # Query expansion configuration
        if request.query_expansion and request.query_expansion.get("enabled", False):
            config["expansion_strategies"] = request.query_expansion.get(
                "strategies", ["acronym", "semantic"]
            )
            config["max_expansions"] = request.query_expansion.get("max_expansions", 5)
        
        # Retrieval configuration
        if request.retrieval:
            if request.retrieval.get("hybrid_search", {}).get("enabled", True):
                config["enable_hybrid_search"] = True
                config["hybrid_alpha"] = request.retrieval["hybrid_search"].get("alpha", 0.7)
            
            if request.retrieval.get("vector_search", {}).get("enabled", True):
                config["vector_top_k"] = request.retrieval["vector_search"].get("top_k", 50)
                config["similarity_threshold"] = request.retrieval["vector_search"].get(
                    "similarity_threshold", 0.7
                )
            
            if request.retrieval.get("fts_search", {}).get("enabled", True):
                config["fts_top_k"] = request.retrieval["fts_search"].get("top_k", 50)
        
        # Processing configuration
        if request.processing:
            config["max_context_size"] = request.processing.get("max_context_size", 10000)
            
            if request.processing.get("table_processing", {}).get("enabled", True):
                config["table_serialize_method"] = request.processing["table_processing"].get(
                    "method", "hybrid"
                )
            
            # Contextual retrieval configuration
            if request.processing.get("contextual_retrieval", {}).get("enabled", False):
                config["expand_parent_context"] = True
                config["parent_expansion_size"] = request.processing["contextual_retrieval"].get(
                    "parent_expansion_size", 500
                )
                config["include_siblings"] = request.processing["contextual_retrieval"].get(
                    "include_siblings", False
                )
        
        # Reranking configuration
        if request.reranking and request.reranking.get("enabled", True):
            config["reranking_strategy"] = request.reranking.get("strategy", "hybrid")
            config["top_k"] = request.reranking.get("top_k", 10)
            config["diversity_weight"] = request.reranking.get("diversity_weight", 0.3)
            config["relevance_weight"] = request.reranking.get("relevance_weight", 0.7)
        
        # Filtering configuration
        if request.filters:
            if request.filters.get("keywords"):
                config["keyword_filters"] = request.filters["keywords"]
            if request.filters.get("metadata"):
                config["metadata_filters"] = request.filters["metadata"]
        
        # Caching configuration
        if request.caching:
            config["enable_cache"] = request.caching.get("enabled", True)
            if request.caching.get("semantic_cache", {}).get("enabled", True):
                config["cache_threshold"] = request.caching["semantic_cache"].get(
                    "threshold", 0.85
                )
        
        # Performance configuration
        if request.performance:
            config["enable_monitoring"] = request.performance.get("enable_monitoring", True)
            config["enable_profiling"] = request.performance.get("enable_profiling", False)
            config["parallel_processing"] = request.performance.get("parallel_processing", True)
        
        # Build dynamic pipeline based on configuration
        pipeline_functions = []
        
        # Query expansion
        if config.get("expansion_strategies"):
            pipeline_functions.append(expand_query)
        
        # Caching
        if config.get("enable_cache", True):
            pipeline_functions.append(check_cache)
        
        # Retrieval with optimization
        if config.get("enable_hybrid_search"):
            pipeline_functions.append(optimize_chromadb_search)
        pipeline_functions.append(retrieve_documents)
        
        # Table processing
        if config.get("table_serialize_method"):
            pipeline_functions.append(process_tables)
        
        # Contextual expansion
        if config.get("expand_parent_context"):
            pipeline_functions.append(expand_with_parent_context)
        
        # Reranking
        if config.get("reranking_strategy"):
            pipeline_functions.append(rerank_documents)
        
        # Cache storage
        if config.get("enable_cache", True):
            pipeline_functions.append(store_in_cache)
        
        # Performance analysis
        if config.get("enable_monitoring"):
            pipeline_functions.append(analyze_performance)
        
        # Execute pipeline
        pipeline = build_pipeline(*pipeline_functions)
        context = await pipeline(request.query, config)
        
        # Process results based on configuration
        max_context = request.processing.get("max_context_size", 10000) if request.processing else 10000
        truncated_docs = truncate_to_context_size(context.documents, max_context)
        
        # Format results with full metadata
        results = []
        for doc in truncated_docs:
            result = ComplexSearchResult(
                id=doc.id or str(uuid4()),
                content=doc.content,
                score=doc.score,
                source=doc.source.value if hasattr(doc.source, 'value') else str(doc.source),
                metadata=doc.metadata,
                citation=create_citation(
                    doc,
                    request.citations.get("include_metadata", True)
                ) if request.citations.get("enabled", False) else None,
                processing_info={
                    "was_reranked": context.metadata.get("reranking_applied", False),
                    "was_expanded": context.metadata.get("query_expanded", False),
                    "from_cache": context.cache_hit,
                },
                debug_info=context.metadata if request.performance.get("enable_profiling") else None
            )
            results.append(result)
        
        # Build response metadata
        response_metadata = {
            "total_documents_found": len(context.documents),
            "documents_returned": len(results),
            "databases_searched": [s.value for s in config["sources"]],
            "cache_hit": context.cache_hit,
            "pipeline_stages": [f.__name__ for f in pipeline_functions],
        }
        
        # Add performance metrics
        performance_data = {
            "total_time": time.time() - start_time,
            "stage_timings": context.timings,
        }
        
        # Add debug information if profiling enabled
        debug_data = None
        if request.performance.get("enable_profiling"):
            debug_data = {
                "expanded_queries": context.metadata.get("expanded_queries", []),
                "errors": context.errors,
                "full_metadata": context.metadata,
            }
        
        return ComplexSearchResponse(
            request_id=request_id,
            query=request.query,
            results=results,
            metadata=response_metadata,
            performance=performance_data,
            debug=debug_data
        )
        
    except Exception as e:
        logger.error(f"Complex search failed for request {request_id}: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Search failed: {str(e)}"
        )


# ============= Health Check =============

@router.get(
    "/health",
    summary="RAG API health check",
    description="Check if the RAG API is operational."
)
async def health_check():
    """Check RAG API health."""
    return {
        "status": "healthy",
        "api_version": "1.0",
        "endpoints": {
            "simple": "/api/v1/rag/search/simple",
            "complex": "/api/v1/rag/search/complex"
        },
        "timestamp": time.time()
    }