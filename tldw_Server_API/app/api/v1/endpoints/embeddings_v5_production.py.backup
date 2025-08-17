# embeddings_v5_production.py - Production-ready embeddings API
"""
Production-ready OpenAI-compatible embeddings API with multiple provider support.

Key improvements over v4:
- No placeholder/fake embeddings - fails explicitly
- Proper authentication and authorization
- Thread-safe caching with TTL and cleanup
- Connection pooling and retry logic
- Comprehensive error handling
- Production monitoring and metrics
"""

import asyncio
import base64
import hashlib
import time
from datetime import datetime, timedelta
from typing import List, Union, Optional, Dict, Any, Tuple
from enum import Enum
import numpy as np
from functools import lru_cache

from fastapi import APIRouter, HTTPException, Body, Depends, status, BackgroundTasks, Request, Query
from fastapi.responses import JSONResponse
import tiktoken
from loguru import logger
from asyncio import Lock
import aiohttp
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

# Schemas
from tldw_Server_API.app.api.v1.schemas.embeddings_models import (
    CreateEmbeddingRequest,
    CreateEmbeddingResponse,
    EmbeddingData,
    EmbeddingUsage
)

# Authentication
from tldw_Server_API.app.core.AuthNZ.User_DB_Handling import get_request_user, User

# Configuration
from tldw_Server_API.app.core.config import settings

# Rate limiting
from slowapi import Limiter
from slowapi.util import get_remote_address

# Monitoring
from prometheus_client import Counter, Histogram, Gauge
import structlog

# Configure structured logging
log = structlog.get_logger()

# ============================================================================
# CRITICAL: Embeddings Implementation Import with Explicit Failure
# ============================================================================

try:
    from tldw_Server_API.app.core.Embeddings.Embeddings_Server.Embeddings_Create import (
        create_embeddings_batch,
        EmbeddingConfigSchema,
        HFModelCfg,
        ONNXModelCfg,
        OpenAIModelCfg,
        LocalAPICfg
    )
    EMBEDDINGS_AVAILABLE = True
except ImportError as e:
    # CRITICAL: Do NOT provide placeholder implementation
    # Fail fast and explicitly in production
    logger.error(f"CRITICAL: Failed to import embeddings implementation: {e}")
    logger.error("Embeddings service cannot start without proper dependencies")
    EMBEDDINGS_AVAILABLE = False
    
    # Raise error immediately - don't allow service to start
    raise RuntimeError(
        f"Embeddings service dependencies not available: {e}. "
        "Please install required packages: transformers, sentence-transformers, onnxruntime"
    )

# ============================================================================
# Metrics and Monitoring
# ============================================================================

# Prometheus metrics - handle duplicate registration gracefully
from prometheus_client import REGISTRY, CollectorRegistry

try:
    embedding_requests_total = Counter(
        'embedding_requests_total',
        'Total number of embedding requests',
        ['provider', 'model', 'status']
    )
except ValueError:
    # Metric already registered, get existing one
    embedding_requests_total = REGISTRY._names_to_collectors.get('embedding_requests_total')
    if not embedding_requests_total:
        # If still not found, use the one from Embeddings_Create
        from tldw_Server_API.app.core.Embeddings.Embeddings_Server.Embeddings_Create import EMBEDDINGS_REQUESTS as embedding_requests_total

try:
    embedding_request_duration = Histogram(
        'embedding_request_duration_seconds',
        'Duration of embedding requests',
        ['provider', 'model']
    )
except ValueError:
    embedding_request_duration = REGISTRY._names_to_collectors.get('embedding_request_duration_seconds')

try:
    embedding_cache_hits = Counter(
        'embedding_cache_hits_total',
        'Number of cache hits',
        ['provider', 'model']
    )
except ValueError:
    embedding_cache_hits = REGISTRY._names_to_collectors.get('embedding_cache_hits_total')

try:
    embedding_cache_size = Gauge(
        'embedding_cache_size',
        'Current size of embedding cache'
    )
except ValueError:
    embedding_cache_size = REGISTRY._names_to_collectors.get('embedding_cache_size')

try:
    active_embedding_requests = Gauge(
        'active_embedding_requests',
        'Number of active embedding requests'
    )
except ValueError:
    active_embedding_requests = REGISTRY._names_to_collectors.get('active_embedding_requests')

# ============================================================================
# Configuration and Constants
# ============================================================================

class EmbeddingProvider(str, Enum):
    OPENAI = "openai"
    HUGGINGFACE = "huggingface"
    ONNX = "onnx"
    LOCAL_API = "local_api"
    COHERE = "cohere"
    VOYAGE = "voyage"
    GOOGLE = "google"
    MISTRAL = "mistral"

# Production configuration
MAX_BATCH_SIZE = 100
MAX_CACHE_SIZE = 5000  # Reduced for production
CACHE_TTL_SECONDS = 3600  # 1 hour
CACHE_CLEANUP_INTERVAL = 300  # 5 minutes
CONNECTION_POOL_SIZE = 20
REQUEST_TIMEOUT = 30
MAX_RETRIES = 3

# Provider models configuration
PROVIDER_MODELS = {
    EmbeddingProvider.OPENAI: [
        "text-embedding-ada-002",
        "text-embedding-3-small", 
        "text-embedding-3-large"
    ],
    EmbeddingProvider.COHERE: [
        "embed-english-v3.0",
        "embed-multilingual-v3.0"
    ],
    EmbeddingProvider.HUGGINGFACE: [
        "sentence-transformers/all-MiniLM-L6-v2",
        "sentence-transformers/all-mpnet-base-v2"
    ]
}

# ============================================================================
# Thread-Safe Cache Implementation with TTL
# ============================================================================

class TTLCache:
    """Thread-safe cache with TTL support and automatic cleanup"""
    
    def __init__(self, max_size: int = MAX_CACHE_SIZE, ttl_seconds: int = CACHE_TTL_SECONDS):
        self.max_size = max_size
        self.ttl_seconds = ttl_seconds
        self.cache: Dict[str, Dict[str, Any]] = {}
        self.lock = Lock()
        self.cleanup_task = None
        
    async def start_cleanup_task(self):
        """Start background cleanup task"""
        if self.cleanup_task is None:
            self.cleanup_task = asyncio.create_task(self._cleanup_loop())
            
    async def stop_cleanup_task(self):
        """Stop background cleanup task"""
        if self.cleanup_task:
            self.cleanup_task.cancel()
            try:
                await self.cleanup_task
            except asyncio.CancelledError:
                pass
            self.cleanup_task = None
            
    async def _cleanup_loop(self):
        """Background task to clean up expired entries"""
        while True:
            try:
                await asyncio.sleep(CACHE_CLEANUP_INTERVAL)
                await self.cleanup_expired()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in cache cleanup: {e}")
                
    async def cleanup_expired(self):
        """Remove expired entries from cache"""
        async with self.lock:
            current_time = time.time()
            expired_keys = [
                key for key, value in self.cache.items()
                if current_time - value['timestamp'] > self.ttl_seconds
            ]
            
            for key in expired_keys:
                del self.cache[key]
                
            if expired_keys:
                logger.info(f"Cleaned up {len(expired_keys)} expired cache entries")
                embedding_cache_size.set(len(self.cache))
                
    async def get(self, key: str) -> Optional[Any]:
        """Get value from cache if not expired"""
        async with self.lock:
            if key in self.cache:
                entry = self.cache[key]
                if time.time() - entry['timestamp'] <= self.ttl_seconds:
                    # Update access time for LRU
                    entry['last_access'] = time.time()
                    return entry['value']
                else:
                    # Entry expired, remove it
                    del self.cache[key]
                    embedding_cache_size.set(len(self.cache))
            return None
            
    async def set(self, key: str, value: Any):
        """Set value in cache with TTL"""
        async with self.lock:
            # Implement LRU eviction if cache is full
            if len(self.cache) >= self.max_size:
                # Find least recently used entry
                lru_key = min(
                    self.cache.keys(),
                    key=lambda k: self.cache[k].get('last_access', 0)
                )
                del self.cache[lru_key]
                
            self.cache[key] = {
                'value': value,
                'timestamp': time.time(),
                'last_access': time.time()
            }
            embedding_cache_size.set(len(self.cache))
            
    async def clear(self):
        """Clear all cache entries"""
        async with self.lock:
            self.cache.clear()
            embedding_cache_size.set(0)
            
    def stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        return {
            'size': len(self.cache),
            'max_size': self.max_size,
            'ttl_seconds': self.ttl_seconds
        }

# ============================================================================
# Connection Pool Manager
# ============================================================================

class ConnectionPoolManager:
    """Manages connection pools for different providers"""
    
    def __init__(self):
        self.pools: Dict[str, aiohttp.ClientSession] = {}
        self.lock = Lock()
        
    async def get_session(self, provider: str) -> aiohttp.ClientSession:
        """Get or create session for provider"""
        async with self.lock:
            if provider not in self.pools:
                connector = aiohttp.TCPConnector(
                    limit=CONNECTION_POOL_SIZE,
                    limit_per_host=CONNECTION_POOL_SIZE
                )
                timeout = aiohttp.ClientTimeout(total=REQUEST_TIMEOUT)
                self.pools[provider] = aiohttp.ClientSession(
                    connector=connector,
                    timeout=timeout
                )
            return self.pools[provider]
            
    async def close_all(self):
        """Close all connection pools"""
        async with self.lock:
            for session in self.pools.values():
                await session.close()
            self.pools.clear()

# ============================================================================
# Global Instances
# ============================================================================

# Initialize cache
embedding_cache = TTLCache()

# Initialize connection pool manager
connection_manager = ConnectionPoolManager()

# Rate limiter
limiter = Limiter(key_func=get_remote_address)

# Router configuration
router = APIRouter(
    tags=["Embeddings"],
    responses={
        401: {"description": "Unauthorized"},
        403: {"description": "Forbidden"},
        429: {"description": "Rate limit exceeded"},
        500: {"description": "Internal server error"},
        503: {"description": "Service unavailable"}
    }
)

# ============================================================================
# Startup and Shutdown Events
# ============================================================================

@router.on_event("startup")
async def startup_event():
    """Initialize services on startup"""
    logger.info("Starting embeddings service v5 (production)")
    
    # Start cache cleanup task
    await embedding_cache.start_cleanup_task()
    
    # Verify embeddings implementation
    if not EMBEDDINGS_AVAILABLE:
        logger.error("Embeddings implementation not available - service will not function")
        
    logger.info("Embeddings service started successfully")

@router.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown"""
    logger.info("Shutting down embeddings service")
    
    # Stop cache cleanup
    await embedding_cache.stop_cleanup_task()
    
    # Close connection pools
    await connection_manager.close_all()
    
    logger.info("Embeddings service shutdown complete")

# ============================================================================
# Helper Functions
# ============================================================================

@lru_cache(maxsize=128)
def get_tokenizer(model_name: str):
    """Get or create a tokenizer for the model"""
    try:
        return tiktoken.encoding_for_model(model_name)
    except KeyError:
        logger.warning(f"No tokenizer for model '{model_name}', using cl100k_base")
        return tiktoken.get_encoding("cl100k_base")

def count_tokens(text: str, model_name: str) -> int:
    """Count tokens in a string"""
    try:
        encoding = get_tokenizer(model_name)
        return len(encoding.encode(text))
    except Exception as e:
        logger.warning(f"Token counting failed: {e}, estimating")
        # Better estimation: ~4 characters per token
        return len(text) // 4

def get_cache_key(text: str, provider: str, model: str, dimensions: Optional[int] = None) -> str:
    """Generate cache key for embedding"""
    key_parts = [text, provider, model]
    if dimensions:
        key_parts.append(str(dimensions))
    key_string = "|".join(key_parts)
    return hashlib.sha256(key_string.encode()).hexdigest()

# ============================================================================
# Provider Configuration Builders
# ============================================================================

def build_provider_config(
    provider: EmbeddingProvider,
    model: str,
    api_key: Optional[str] = None,
    api_url: Optional[str] = None,
    dimensions: Optional[int] = None
) -> Dict[str, Any]:
    """Build provider-specific configuration"""
    
    # Get API keys from environment if not provided
    if provider == EmbeddingProvider.OPENAI:
        return {
            "provider": "openai",
            "model_name_or_path": model,
            "api_key": api_key or settings.get("OPENAI_API_KEY"),
        }
    elif provider == EmbeddingProvider.HUGGINGFACE:
        return {
            "provider": "huggingface",
            "model_name_or_path": model,
            "trust_remote_code": False,
            "hf_cache_dir_subpath": "huggingface_cache",
        }
    # Add other providers...
    else:
        raise ValueError(f"Unknown provider: {provider}")

# ============================================================================
# Core Embedding Function with Retry Logic
# ============================================================================

@retry(
    stop=stop_after_attempt(MAX_RETRIES),
    wait=wait_exponential(multiplier=1, min=2, max=10),
    retry=retry_if_exception_type((ConnectionError, TimeoutError))
)
async def create_embeddings_with_retry(
    texts: List[str],
    config: Dict[str, Any],
    model_id: Optional[str] = None
) -> List[List[float]]:
    """Create embeddings with retry logic"""
    
    # Run in thread pool to avoid blocking
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(
        None,  # Use default executor
        create_embeddings_batch,
        texts,
        config,
        model_id
    )

async def create_embeddings_batch_async(
    texts: List[str],
    provider: str,
    model_id: Optional[str] = None,
    dimensions: Optional[int] = None,
    api_key: Optional[str] = None,
    api_url: Optional[str] = None
) -> List[List[float]]:
    """Async wrapper for embeddings with caching and batching"""
    
    embeddings = []
    uncached_texts = []
    uncached_indices = []
    
    # Check cache
    for i, text in enumerate(texts):
        cache_key = get_cache_key(text, provider, model_id or "default", dimensions)
        cached = await embedding_cache.get(cache_key)
        
        if cached:
            embeddings.append(cached)
            embedding_cache_hits.labels(provider=provider, model=model_id).inc()
        else:
            embeddings.append(None)
            uncached_texts.append(text)
            uncached_indices.append(i)
    
    # Process uncached texts
    if uncached_texts:
        config = build_provider_config(
            EmbeddingProvider(provider),
            model_id,
            api_key,
            api_url,
            dimensions
        )
        
        # Process in batches
        all_new_embeddings = []
        for batch_start in range(0, len(uncached_texts), MAX_BATCH_SIZE):
            batch_end = min(batch_start + MAX_BATCH_SIZE, len(uncached_texts))
            batch_texts = uncached_texts[batch_start:batch_end]
            
            try:
                batch_embeddings = await create_embeddings_with_retry(
                    batch_texts,
                    config,
                    model_id
                )
                all_new_embeddings.extend(batch_embeddings)
            except Exception as e:
                logger.error(f"Failed to create embeddings for batch: {e}")
                raise HTTPException(
                    status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                    detail=f"Embedding service temporarily unavailable: {str(e)}"
                )
        
        # Update results and cache
        for i, (idx, text) in enumerate(zip(uncached_indices, uncached_texts)):
            embedding = all_new_embeddings[i]
            embeddings[idx] = embedding
            
            # Cache asynchronously
            cache_key = get_cache_key(text, provider, model_id or "default", dimensions)
            await embedding_cache.set(cache_key, embedding)
    
    return embeddings

# ============================================================================
# Authorization Helpers
# ============================================================================

def require_admin(user: User) -> None:
    """Require admin privileges for endpoint"""
    if not user or not getattr(user, 'is_admin', False):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Admin privileges required"
        )

# ============================================================================
# API Endpoints
# ============================================================================

@router.post(
    "/embeddings",
    response_model=CreateEmbeddingResponse,
    status_code=status.HTTP_200_OK,
    summary="Create embeddings (production-ready)"
)
@limiter.limit("60/minute")
async def create_embedding_endpoint(
    request: Request,
    embedding_request: CreateEmbeddingRequest = Body(...),
    current_user: User = Depends(get_request_user),
    background_tasks: BackgroundTasks = BackgroundTasks(),
    x_provider: Optional[str] = None
):
    """Create embeddings with production-grade error handling and monitoring"""
    
    # Track active requests
    active_embedding_requests.inc()
    start_time = time.time()
    
    try:
        # Validate provider
        provider = x_provider or "openai"
        model = embedding_request.model
        
        # Extract provider from model if specified
        if ":" in model:
            parts = model.split(":", 1)
            provider = parts[0]
            model = parts[1]
        
        try:
            provider_enum = EmbeddingProvider(provider.lower())
        except ValueError:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Unknown provider: {provider}"
            )
        
        # Parse and validate input
        texts_to_embed: List[str] = []
        
        if isinstance(embedding_request.input, str):
            if not embedding_request.input.strip():
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="Input cannot be empty"
                )
            texts_to_embed = [embedding_request.input]
        elif isinstance(embedding_request.input, list):
            if not embedding_request.input:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="Input list cannot be empty"
                )
            if len(embedding_request.input) > 2048:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="Maximum 2048 inputs allowed"
                )
            
            # Validate all strings
            if not all(isinstance(item, str) for item in embedding_request.input):
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="All inputs must be strings"
                )
            texts_to_embed = embedding_request.input
        else:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Invalid input type"
            )
        
        # Create embeddings
        try:
            embeddings = await create_embeddings_batch_async(
                texts=texts_to_embed,
                provider=provider,
                model_id=model,
                dimensions=embedding_request.dimensions
            )
        except Exception as e:
            embedding_requests_total.labels(
                provider=provider,
                model=model,
                status="error"
            ).inc()
            logger.error(f"Embedding creation failed: {e}", exc_info=True)
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to create embeddings"
            )
        
        # Format response
        output_data = []
        for i, embedding in enumerate(embeddings):
            if embedding_request.encoding_format == "base64":
                byte_array = np.array(embedding, dtype=np.float32).tobytes()
                processed_value = base64.b64encode(byte_array).decode('utf-8')
            else:
                processed_value = embedding
            
            output_data.append(
                EmbeddingData(
                    embedding=processed_value,
                    index=i
                )
            )
        
        # Calculate token usage
        num_tokens = sum(count_tokens(text, model) for text in texts_to_embed)
        
        # Track metrics
        duration = time.time() - start_time
        embedding_request_duration.labels(
            provider=provider,
            model=model
        ).observe(duration)
        
        embedding_requests_total.labels(
            provider=provider,
            model=model,
            status="success"
        ).inc()
        
        logger.info(
            f"Created {len(output_data)} embeddings",
            extra={
                "user_id": current_user.id,
                "provider": provider,
                "model": model,
                "duration": duration
            }
        )
        
        return CreateEmbeddingResponse(
            data=output_data,
            model=f"{provider}:{model}" if provider != "openai" else model,
            usage=EmbeddingUsage(
                prompt_tokens=num_tokens,
                total_tokens=num_tokens
            )
        )
        
    finally:
        active_embedding_requests.dec()

@router.delete(
    "/embeddings/cache",
    summary="Clear embedding cache (admin only)"
)
async def clear_cache(
    current_user: User = Depends(get_request_user)
):
    """Clear the embedding cache - requires admin privileges"""
    
    # SECURITY: Proper admin check
    require_admin(current_user)
    
    cache_stats = embedding_cache.stats()
    await embedding_cache.clear()
    
    logger.info(
        f"Cache cleared by admin",
        extra={
            "admin_id": current_user.id,
            "entries_cleared": cache_stats['size']
        }
    )
    
    return {
        "message": "Cache cleared successfully",
        "entries_removed": cache_stats['size']
    }

@router.get(
    "/embeddings/health",
    summary="Health check"
)
async def health_check():
    """Production health check endpoint"""
    
    health_status = {
        "status": "healthy" if EMBEDDINGS_AVAILABLE else "degraded",
        "service": "embeddings_v5_production",
        "timestamp": datetime.utcnow().isoformat(),
        "cache_stats": embedding_cache.stats(),
        "active_requests": active_embedding_requests._value.get()
    }
    
    if not EMBEDDINGS_AVAILABLE:
        return JSONResponse(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            content=health_status
        )
    
    return health_status

@router.get(
    "/embeddings/metrics",
    summary="Get service metrics (admin only)"
)
async def get_metrics(
    current_user: User = Depends(get_request_user)
):
    """Get detailed service metrics - requires admin privileges"""
    
    require_admin(current_user)
    
    return {
        "cache": embedding_cache.stats(),
        "active_requests": active_embedding_requests._value.get(),
        "total_requests": {
            "success": embedding_requests_total.labels(
                provider="all", model="all", status="success"
            )._value.get(),
            "error": embedding_requests_total.labels(
                provider="all", model="all", status="error"
            )._value.get()
        }
    }