# rag_v2.py - Simplified RAG API Endpoints
"""
Refactored RAG API with developer-friendly endpoints.
Provides simple and advanced variants for both search and agent functionality.

This module follows FastAPI best practices with:
- Clear separation between simple and advanced usage
- Comprehensive error handling
- Async-first design
- Structured logging
- OpenAPI documentation
"""

import asyncio
import json
import time
from typing import Optional, Dict, Any, List
from uuid import uuid4
from pathlib import Path

from fastapi import APIRouter, HTTPException, Depends, status
from fastapi.responses import StreamingResponse
from loguru import logger

# Schemas
from tldw_Server_API.app.api.v1.schemas.rag_schemas_simple import (
    SimpleSearchRequest,
    SimpleSearchResponse,
    SearchResult,
    AdvancedSearchRequest,
    AdvancedSearchResponse,
    SimpleAgentRequest,
    SimpleAgentResponse,
    Source,
    AdvancedAgentRequest,
    AdvancedAgentResponse,
    ErrorResponse,
    SearchType,
    AgentMode,
)

# Dependencies
from tldw_Server_API.app.api.v1.API_Deps.DB_Deps import get_media_db_for_user
from tldw_Server_API.app.api.v1.API_Deps.ChaCha_Notes_DB_Deps import get_chacha_db_for_user
from tldw_Server_API.app.core.AuthNZ.User_DB_Handling import get_request_user, User
from tldw_Server_API.app.core.DB_Management.Media_DB_v2 import MediaDatabase
from tldw_Server_API.app.core.DB_Management.ChaChaNotes_DB import CharactersRAGDB

# RAG Service
from tldw_Server_API.app.core.RAG.rag_service.integration import RAGService
from tldw_Server_API.app.core.RAG.rag_service.types import DataSource
from tldw_Server_API.app.core.RAG.rag_service.config import RAGConfig
from tldw_Server_API.app.core.config import settings, RAG_SERVICE_CONFIG

# ============= Router Configuration =============

router = APIRouter(
    tags=["RAG"],
    responses={
        400: {"model": ErrorResponse, "description": "Bad Request"},
        401: {"model": ErrorResponse, "description": "Unauthorized"},
        404: {"model": ErrorResponse, "description": "Not Found"},
        500: {"model": ErrorResponse, "description": "Internal Server Error"},
    }
)

# ============= Database Mapping =============

DATABASE_MAPPING = {
    "media_db": DataSource.MEDIA_DB,
    "media": DataSource.MEDIA_DB,  # Alias
    "notes": DataSource.NOTES,
    "characters": DataSource.CHARACTER_CARDS,
    "chat_history": DataSource.CHAT_HISTORY,
    "chats": DataSource.CHAT_HISTORY,  # Alias
}

# ============= RAG Service Cache =============

class RAGServiceManager:
    """Manages RAG services for users with caching"""
    
    def __init__(self, ttl_seconds: int = 3600):
        self._cache: Dict[int, Dict[str, Any]] = {}
        self._lock = asyncio.Lock()
        self._ttl = ttl_seconds
    
    async def get_or_create(
        self,
        user_id: int,
        media_db_path: Path,
        chacha_db_path: Path
    ) -> RAGService:
        """Get cached service or create new one"""
        async with self._lock:
            # Check cache
            if user_id in self._cache:
                entry = self._cache[user_id]
                if time.time() - entry['timestamp'] < self._ttl:
                    return entry['service']
                # Expired - cleanup
                await self._cleanup_service(entry['service'])
                del self._cache[user_id]
            
            # Create new service
            service = await self._create_service(user_id, media_db_path, chacha_db_path)
            self._cache[user_id] = {
                'service': service,
                'timestamp': time.time()
            }
            return service
    
    async def _create_service(
        self,
        user_id: int,
        media_db_path: Path,
        chacha_db_path: Path
    ) -> RAGService:
        """Create new RAG service for user"""
        user_dir = Path(settings.get("USER_DB_BASE_DIR")) / str(user_id)
        chroma_path = user_dir / "chroma"
        
        # Create RAG config
        config = RAGConfig()
        for key, value in RAG_SERVICE_CONFIG.items():
            if hasattr(config, key):
                if isinstance(value, dict):
                    config_attr = getattr(config, key)
                    for nested_key, nested_value in value.items():
                        if hasattr(config_attr, nested_key):
                            setattr(config_attr, nested_key, nested_value)
                else:
                    setattr(config, key, value)
        
        # Create service
        service = RAGService(
            config=config,
            media_db_path=media_db_path,
            chachanotes_db_path=chacha_db_path,
            chroma_path=chroma_path,
            llm_handler=None  # Will be configured based on request
        )
        
        await service.initialize()
        logger.info(f"Created RAG service for user {user_id}")
        return service
    
    async def _cleanup_service(self, service: RAGService):
        """Cleanup RAG service resources"""
        try:
            service.cleanup()
        except Exception as e:
            logger.warning(f"Error cleaning up RAG service: {e}")
    
    async def cleanup_expired(self):
        """Clean up all expired services"""
        async with self._lock:
            current_time = time.time()
            expired = []
            
            for user_id, entry in self._cache.items():
                if current_time - entry['timestamp'] > self._ttl:
                    expired.append(user_id)
                    await self._cleanup_service(entry['service'])
            
            for user_id in expired:
                del self._cache[user_id]
            
            if expired:
                logger.info(f"Cleaned up {len(expired)} expired RAG services")

# Global service manager
rag_service_manager = RAGServiceManager()

# ============= Dependencies =============

async def get_rag_service(
    current_user: User = Depends(get_request_user),
    media_db: MediaDatabase = Depends(get_media_db_for_user),
    chacha_db: CharactersRAGDB = Depends(get_chacha_db_for_user)
) -> RAGService:
    """Get or create RAG service for current user"""
    return await rag_service_manager.get_or_create(
        user_id=current_user.id,
        media_db_path=Path(media_db.db_path),
        chacha_db_path=Path(chacha_db.db_path)
    )

# ============= Helper Functions =============

def map_databases_to_sources(databases: List[str]) -> List[str]:
    """Map database names to DataSource strings"""
    sources = []
    for db in databases:
        if db in DATABASE_MAPPING:
            sources.append(DATABASE_MAPPING[db].name)  # Get the string name of the enum
    return sources if sources else [DataSource.MEDIA_DB.name]  # Default to media_db

def format_search_results(raw_results: List[Dict], search_type: SearchType) -> List[SearchResult]:
    """Format raw search results into response model"""
    formatted = []
    for result in raw_results:
        formatted.append(SearchResult(
            id=result.get("id", str(uuid4())),
            title=result.get("title", "Untitled"),
            content=result.get("content", "")[:500],  # Limit snippet length
            score=result.get("score", 0.0),
            source=result.get("source", "unknown"),
            metadata=result.get("metadata", {})
        ))
    return formatted

def create_search_config(request: SimpleSearchRequest) -> Dict[str, Any]:
    """Create search configuration from simple request"""
    config = {
        "limit": request.limit,
        "sources": map_databases_to_sources(request.databases),
    }
    
    # Add search type configuration
    if request.search_type == SearchType.HYBRID:
        config["use_hybrid_search"] = True
        config["use_semantic_search"] = True
        config["use_fulltext_search"] = True
    elif request.search_type == SearchType.SEMANTIC:
        config["use_hybrid_search"] = False
        config["use_semantic_search"] = True
        config["use_fulltext_search"] = False
    else:  # FULLTEXT
        config["use_hybrid_search"] = False
        config["use_semantic_search"] = False
        config["use_fulltext_search"] = True
    
    # Add keyword filters if provided
    if request.keywords:
        config["filters"] = {"keywords": {"$in": request.keywords}}
    
    return config

# ============= Simple Search Endpoint =============

@router.post(
    "/search",
    response_model=SimpleSearchResponse,
    status_code=status.HTTP_200_OK,
    summary="Simple search across databases",
    description="""
    Perform a simple search with essential parameters only.
    
    - **query**: The search query string
    - **search_type**: Choose between hybrid, semantic, or fulltext search
    - **limit**: Maximum number of results (1-100)
    - **databases**: Which databases to search (media_db, notes, characters, chat_history)
    - **keywords**: Optional keywords to filter results
    """
)
async def simple_search(
    request: SimpleSearchRequest,
    rag_service: RAGService = Depends(get_rag_service),
    current_user: User = Depends(get_request_user)
) -> SimpleSearchResponse:
    """Execute a simple search request"""
    logger.info(f"User {current_user.id}: Simple search for '{request.query[:50]}...'")
    
    try:
        # Create search configuration
        search_config = create_search_config(request)
        
        # Execute search
        logger.debug(f"Search config: {search_config}")
        raw_results = await rag_service.search(
            query=request.query,
            **search_config
        )
        
        # Format results
        results = format_search_results(raw_results, request.search_type)
        
        return SimpleSearchResponse(
            results=results,
            total_results=len(results),
            query_id=str(uuid4()),
            search_type_used=request.search_type
        )
        
    except Exception as e:
        logger.error(f"Search failed for user {current_user.id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Search operation failed: {str(e)}"
        )

# ============= Advanced Search Endpoint =============

@router.post(
    "/search/advanced",
    response_model=AdvancedSearchResponse,
    status_code=status.HTTP_200_OK,
    summary="Advanced search with full control",
    description="""
    Advanced search endpoint with complete configuration options.
    
    Supports:
    - Complex metadata filters
    - Date range filtering
    - Custom search strategies (vanilla, query_fusion, hyde)
    - Fine-tuned hybrid search weights
    - Semantic similarity thresholds
    - Full content retrieval
    """
)
async def advanced_search(
    request: AdvancedSearchRequest,
    rag_service: RAGService = Depends(get_rag_service),
    current_user: User = Depends(get_request_user)
) -> AdvancedSearchResponse:
    """Execute an advanced search request with full configuration"""
    logger.info(f"User {current_user.id}: Advanced search for '{request.query[:50]}...'")
    
    try:
        # Build comprehensive search configuration
        config = {
            "limit": request.search_config.limit,
            "offset": request.search_config.offset,
            "sources": map_databases_to_sources(request.search_config.databases),
            "include_scores": request.search_config.include_scores,
            "search_strategy": request.strategy.value,
        }
        
        # Configure search type
        if request.search_config.search_type == SearchType.HYBRID:
            config["use_hybrid_search"] = True
            if request.hybrid_config:
                config["hybrid_settings"] = {
                    "semantic_weight": request.hybrid_config.semantic_weight,
                    "fulltext_weight": request.hybrid_config.fulltext_weight,
                    "rrf_k": request.hybrid_config.rrf_k
                }
        elif request.search_config.search_type == SearchType.SEMANTIC:
            config["use_semantic_search"] = True
            config["use_fulltext_search"] = False
            if request.semantic_config:
                config["similarity_threshold"] = request.semantic_config.similarity_threshold
                config["rerank"] = request.semantic_config.rerank
        else:  # FULLTEXT
            config["use_semantic_search"] = False
            config["use_fulltext_search"] = True
        
        # Add filters
        filters = {}
        if request.search_config.keywords:
            filters["keywords"] = {"$in": request.search_config.keywords}
        if request.search_config.metadata_filters:
            filters.update(request.search_config.metadata_filters)
        if request.search_config.date_range:
            filters["date"] = {
                "$gte": request.search_config.date_range.get("start"),
                "$lte": request.search_config.date_range.get("end")
            }
        if filters:
            config["filters"] = filters
        
        # Execute search
        raw_results = await rag_service.search(
            query=request.query,
            **config
        )
        
        # Format results with full content if requested
        results = []
        for result in raw_results:
            content = result.get("content", "")
            if not request.search_config.include_full_content:
                content = content[:500]  # Limit to snippet
            
            results.append(SearchResult(
                id=result.get("id", str(uuid4())),
                title=result.get("title", "Untitled"),
                content=content,
                score=result.get("score", 0.0),
                source=result.get("source", "unknown"),
                metadata=result.get("metadata", {})
            ))
        
        return AdvancedSearchResponse(
            results=results,
            total_results=len(results),
            query_id=str(uuid4()),
            search_type_used=request.search_config.search_type,
            strategy_used=request.strategy,
            search_config=config,
            debug_info={"execution_time": time.time()} if settings.get("DEBUG") else None
        )
        
    except Exception as e:
        logger.error(f"Advanced search failed for user {current_user.id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Advanced search operation failed: {str(e)}"
        )

# ============= Simple Agent Endpoint =============

@router.post(
    "/agent",
    response_model=SimpleAgentResponse,
    status_code=status.HTTP_200_OK,
    summary="Simple Q&A agent with retrieval",
    description="""
    Simple agent for question answering with automatic context retrieval.
    
    - **message**: Your question or message
    - **conversation_id**: Optional ID to continue a conversation
    - **search_databases**: Which databases to search for context
    - **model**: Optional model selection (uses default if not specified)
    """
)
async def simple_agent(
    request: SimpleAgentRequest,
    rag_service: RAGService = Depends(get_rag_service),
    current_user: User = Depends(get_request_user),
    chacha_db: CharactersRAGDB = Depends(get_chacha_db_for_user)
) -> SimpleAgentResponse:
    """Simple agent endpoint for Q&A with retrieval"""
    logger.info(f"User {current_user.id}: Agent request - '{request.message[:50]}...'")
    
    try:
        # Load conversation context if provided
        conversation_history = []
        if request.conversation_id:
            try:
                messages = chacha_db.get_messages_for_conversation(
                    request.conversation_id,
                    limit=20,  # Reasonable context window
                    order_by_timestamp="ASC"
                )
                for msg in messages:
                    if msg.get('content') != request.message:  # Skip current message
                        conversation_history.append({
                            "role": msg.get('sender', 'user').lower(),
                            "content": msg.get('content', '')
                        })
                logger.info(f"Loaded {len(conversation_history)} messages from conversation")
            except Exception as e:
                logger.warning(f"Failed to load conversation history: {e}")
        
        # Generate or use conversation ID
        conversation_id = request.conversation_id or str(uuid4())
        
        # Search for relevant context
        search_config = {
            "limit": 5,  # Fixed for simple mode
            "sources": map_databases_to_sources(request.search_databases),
            "use_hybrid_search": True  # Best for Q&A
        }
        
        context_results = await rag_service.search(
            query=request.message,
            **search_config
        )
        
        # Format context for generation
        context = "\n\n".join([
            f"[{r.get('source', 'unknown')}] {r.get('title', 'Untitled')}:\n{r.get('content', '')[:500]}"
            for r in context_results[:3]  # Top 3 results
        ])
        
        # Generate response
        generation_config = {
            "model": request.model or settings.get("DEFAULT_LLM_MODEL", "gpt-3.5-turbo"),
            "temperature": 0.7,
            "max_tokens": 1024,
            "messages": conversation_history + [
                {"role": "system", "content": "You are a helpful assistant. Use the provided context to answer questions accurately."},
                {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {request.message}"}
            ]
        }
        
        response = await rag_service.generate(**generation_config)
        
        # Format sources
        sources = [
            Source(
                title=r.get("title", "Untitled"),
                content=r.get("content", "")[:200],
                database=r.get("source", "unknown"),
                relevance_score=r.get("score", 0.0)
            )
            for r in context_results[:3]
        ]
        
        # Save conversation
        if request.conversation_id:
            try:
                chacha_db.save_message(
                    conversation_id=conversation_id,
                    sender="assistant",
                    content=response,
                    metadata={"sources": len(sources)}
                )
            except Exception as e:
                logger.warning(f"Failed to save conversation: {e}")
        
        return SimpleAgentResponse(
            response=response,
            conversation_id=conversation_id,
            sources=sources
        )
        
    except Exception as e:
        logger.error(f"Agent request failed for user {current_user.id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Agent operation failed: {str(e)}"
        )

# ============= Advanced Agent Endpoint =============

@router.post(
    "/agent/advanced",
    response_model=AdvancedAgentResponse,
    status_code=status.HTTP_200_OK,
    summary="Advanced agent with research capabilities",
    description="""
    Advanced agent with full control over generation and search.
    
    Supports:
    - RAG mode for simple Q&A
    - Research mode for multi-step reasoning
    - Custom system prompts
    - Tool usage (web search, reasoning, etc.)
    - Streaming responses
    - Fine-tuned generation parameters
    """
)
async def advanced_agent(
    request: AdvancedAgentRequest,
    rag_service: RAGService = Depends(get_rag_service),
    current_user: User = Depends(get_request_user),
    chacha_db: CharactersRAGDB = Depends(get_chacha_db_for_user)
) -> AdvancedAgentResponse:
    """Advanced agent with research capabilities"""
    logger.info(f"User {current_user.id}: Advanced agent in {request.mode} mode")
    
    try:
        # Load conversation context
        conversation_history = []
        if request.conversation_id:
            try:
                messages = chacha_db.get_messages_for_conversation(
                    request.conversation_id,
                    limit=50,
                    order_by_timestamp="ASC"
                )
                for msg in messages:
                    if msg.get('content') != request.message:
                        conversation_history.append({
                            "role": msg.get('sender', 'user').lower(),
                            "content": msg.get('content', '')
                        })
            except Exception as e:
                logger.warning(f"Failed to load conversation: {e}")
        
        conversation_id = request.conversation_id or str(uuid4())
        
        # Configure search
        search_config = {}
        if request.search_config:
            search_config = {
                "limit": request.search_config.limit,
                "sources": map_databases_to_sources(request.search_config.databases),
                "use_hybrid_search": request.search_config.search_type == SearchType.HYBRID,
                "use_semantic_search": request.search_config.search_type == SearchType.SEMANTIC,
                "use_fulltext_search": request.search_config.search_type == SearchType.FULLTEXT,
            }
            if request.search_config.keywords:
                search_config["filters"] = {"keywords": {"$in": request.search_config.keywords}}
        else:
            # Default search config
            search_config = {
                "limit": 10,
                "sources": [DataSource.MEDIA_DB.name],
                "use_hybrid_search": True
            }
        
        # Retrieve context
        context_results = await rag_service.search(
            query=request.message,
            **search_config
        )
        
        # Configure generation
        gen_config = request.generation_config or {}
        generation_config = {
            "model": gen_config.model or settings.get("DEFAULT_LLM_MODEL", "gpt-4"),
            "temperature": gen_config.temperature if hasattr(gen_config, 'temperature') else 0.7,
            "max_tokens": gen_config.max_tokens if hasattr(gen_config, 'max_tokens') else 2048,
            "stream": gen_config.stream if hasattr(gen_config, 'stream') else False,
        }
        
        # Build messages
        system_prompt = request.system_prompt or (
            "You are an advanced research assistant. "
            "Use the provided context and tools to answer questions thoroughly and accurately."
        )
        
        context = "\n\n".join([
            f"[{r.get('source', 'unknown')}] {r.get('title', 'Untitled')}:\n{r.get('content', '')[:1000]}"
            for r in context_results[:5]
        ])
        
        messages = conversation_history + [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Context:\n{context}\n\n{request.message}"}
        ]
        
        # Handle different modes
        tools_used = []
        if request.mode == AgentMode.RESEARCH and request.tools:
            # Research mode with tools
            logger.info(f"Using research mode with tools: {request.tools}")
            # TODO: Implement tool usage for research mode
            tools_used = [t.value for t in request.tools]
        
        # Generate response
        if generation_config.get("stream"):
            # Streaming response
            async def generate_stream():
                async for chunk in rag_service.generate_stream(messages=messages, **generation_config):
                    yield f"data: {json.dumps({'content': chunk})}\n\n"
                yield f"data: {json.dumps({'done': True})}\n\n"
            
            return StreamingResponse(
                generate_stream(),
                media_type="text/event-stream"
            )
        else:
            # Regular response
            response = await rag_service.generate(messages=messages, **generation_config)
            
            # Format sources
            sources = [
                Source(
                    title=r.get("title", "Untitled"),
                    content=r.get("content", "")[:500],
                    database=r.get("source", "unknown"),
                    relevance_score=r.get("score", 0.0)
                )
                for r in context_results[:5]
            ]
            
            # Save to conversation
            try:
                chacha_db.save_message(
                    conversation_id=conversation_id,
                    sender="assistant",
                    content=response,
                    metadata={
                        "mode": request.mode.value,
                        "tools_used": tools_used,
                        "sources": len(sources)
                    }
                )
            except Exception as e:
                logger.warning(f"Failed to save conversation: {e}")
            
            return AdvancedAgentResponse(
                response=response,
                conversation_id=conversation_id,
                sources=sources,
                mode_used=request.mode,
                tools_used=tools_used if tools_used else None,
                search_stats={
                    "total_results": len(context_results),
                    "databases_searched": search_config.get("sources", [])
                },
                generation_stats={
                    "model": generation_config["model"],
                    "temperature": generation_config["temperature"],
                    "max_tokens": generation_config["max_tokens"]
                }
            )
    
    except Exception as e:
        logger.error(f"Advanced agent failed for user {current_user.id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Advanced agent operation failed: {str(e)}"
        )

# ============= Health Check =============

@router.get(
    "/health",
    summary="RAG service health check",
    description="Check if the RAG service is operational"
)
async def health_check():
    """Health check endpoint for RAG service"""
    return {
        "status": "healthy",
        "service": "rag_v2",
        "timestamp": time.time()
    }

# ============= Cleanup Task =============

async def cleanup_expired_services():
    """Background task to clean up expired RAG services"""
    while True:
        await asyncio.sleep(3600)  # Run every hour
        await rag_service_manager.cleanup_expired()