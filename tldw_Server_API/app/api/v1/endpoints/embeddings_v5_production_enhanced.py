# embeddings_v5_production_enhanced.py
# Enhanced version with circuit breaker pattern and improved error recovery
"""
Production-ready OpenAI-compatible embeddings API with circuit breaker.

Key enhancements over v5:
- Circuit breaker pattern for fault tolerance
- Improved connection cleanup on failures
- Better error recovery mechanisms
- Enhanced monitoring and observability
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
import atexit
import os

from fastapi import APIRouter, HTTPException, Body, Depends, status, BackgroundTasks, Request, Query, Header
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

# Circuit Breaker
from tldw_Server_API.app.core.Embeddings.circuit_breaker import (
    CircuitBreaker,
    CircuitBreakerError,
    circuit_breaker,
    registry as circuit_breaker_registry
)

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
    logger.error(f"CRITICAL: Failed to import embeddings implementation: {e}")
    logger.error("Embeddings service cannot start without proper dependencies")
    EMBEDDINGS_AVAILABLE = False
    
    raise RuntimeError(
        f"Embeddings service dependencies not available: {e}. "
        "Please install required packages: transformers, sentence-transformers, onnxruntime"
    )

# ============================================================================
# Metrics and Monitoring
# ============================================================================

from prometheus_client import REGISTRY

# Safely get or create metrics
def get_or_create_counter(name, description, labelnames):
    """Get existing counter or create new one"""
    try:
        # Check if metric already exists
        if name in REGISTRY._names_to_collectors:
            collector = REGISTRY._names_to_collectors[name]
            # Verify it's a Counter with matching labels
            if hasattr(collector, '_labelnames') and set(collector._labelnames) == set(labelnames):
                return collector
            # If labels don't match, unregister the old one
            REGISTRY.unregister(collector)
        return Counter(name, description, labelnames)
    except Exception as e:
        # Try to create new counter, handling any registration issues
        try:
            return Counter(name, description, labelnames)
        except Exception:
            # Return existing if we can't create new
            if name in REGISTRY._names_to_collectors:
                return REGISTRY._names_to_collectors[name]
            raise

def get_or_create_histogram(name, description, labelnames):
    """Get existing histogram or create new one"""
    try:
        if name in REGISTRY._names_to_collectors:
            collector = REGISTRY._names_to_collectors[name]
            # Verify it's a Histogram with matching labels
            if hasattr(collector, '_labelnames') and set(collector._labelnames) == set(labelnames):
                return collector
            # If labels don't match, unregister the old one
            REGISTRY.unregister(collector)
        return Histogram(name, description, labelnames)
    except Exception:
        # Try to create new, or return existing
        try:
            return Histogram(name, description, labelnames)
        except Exception:
            if name in REGISTRY._names_to_collectors:
                return REGISTRY._names_to_collectors[name]
            raise

def get_or_create_gauge(name, description, labelnames=None):
    """Get existing gauge or create new one"""
    try:
        if name in REGISTRY._names_to_collectors:
            collector = REGISTRY._names_to_collectors[name]
            expected_labels = set(labelnames) if labelnames else set()
            existing_labels = set(collector._labelnames) if hasattr(collector, '_labelnames') else set()
            if expected_labels == existing_labels:
                return collector
            # If labels don't match, unregister the old one
            REGISTRY.unregister(collector)
        if labelnames:
            return Gauge(name, description, labelnames)
        return Gauge(name, description)
    except Exception:
        # Try to create new, or return existing
        try:
            if labelnames:
                return Gauge(name, description, labelnames)
            return Gauge(name, description)
        except Exception:
            if name in REGISTRY._names_to_collectors:
                return REGISTRY._names_to_collectors[name]
            raise

# Create metrics using safe getters
embedding_requests_total = get_or_create_counter(
    'embedding_requests_total',
    'Total number of embedding requests',
    ['provider', 'model', 'status']
)

embedding_request_duration = get_or_create_histogram(
    'embedding_request_duration_seconds',
    'Duration of embedding requests',
    ['provider', 'model']
)

embedding_cache_hits = get_or_create_counter(
    'embedding_cache_hits_total',
    'Number of cache hits',
    ['provider', 'model']
)

embedding_cache_size = get_or_create_gauge(
    'embedding_cache_size',
    'Current size of embedding cache'
)

active_embedding_requests = get_or_create_gauge(
    'active_embedding_requests',
    'Number of active embedding requests'
)

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
MAX_CACHE_SIZE = 5000
CACHE_TTL_SECONDS = 3600
CACHE_CLEANUP_INTERVAL = 300
CONNECTION_POOL_SIZE = 20
REQUEST_TIMEOUT = 30
MAX_RETRIES = 3

# Circuit breaker configuration
CIRCUIT_BREAKER_FAILURE_THRESHOLD = 5
CIRCUIT_BREAKER_RECOVERY_TIMEOUT = 60
CIRCUIT_BREAKER_SUCCESS_THRESHOLD = 2

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
# Enhanced TTL Cache with Better Cleanup
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
                    entry['last_access'] = time.time()
                    return entry['value']
                else:
                    del self.cache[key]
                    embedding_cache_size.set(len(self.cache))
            return None
            
    async def set(self, key: str, value: Any):
        """Set value in cache with TTL"""
        async with self.lock:
            if len(self.cache) >= self.max_size:
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
# Enhanced Connection Pool Manager with Cleanup
# ============================================================================

class ConnectionPoolManager:
    """Manages connection pools with proper cleanup"""
    
    def __init__(self):
        self.pools: Dict[str, aiohttp.ClientSession] = {}
        self.lock = Lock()
        self._closed = False
        
    async def get_session(self, provider: str) -> aiohttp.ClientSession:
        """Get or create session for provider"""
        if self._closed:
            raise RuntimeError("ConnectionPoolManager has been closed")
            
        async with self.lock:
            if provider not in self.pools:
                connector = aiohttp.TCPConnector(
                    limit=CONNECTION_POOL_SIZE,
                    limit_per_host=CONNECTION_POOL_SIZE,
                    force_close=True  # Force close connections on errors
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
            self._closed = True
            for session in self.pools.values():
                await session.close()
            self.pools.clear()
            
    async def remove_provider(self, provider: str):
        """Remove and close specific provider's session"""
        async with self.lock:
            if provider in self.pools:
                await self.pools[provider].close()
                del self.pools[provider]

# ============================================================================
# Initialize Circuit Breakers for Each Provider
# ============================================================================

def get_or_create_circuit_breaker(provider: str) -> CircuitBreaker:
    """Get or create circuit breaker for provider"""
    breaker_name = f"embeddings_{provider}"
    breaker = circuit_breaker_registry.get(breaker_name)
    
    if not breaker:
        breaker = CircuitBreaker(
            name=breaker_name,
            failure_threshold=CIRCUIT_BREAKER_FAILURE_THRESHOLD,
            recovery_timeout=CIRCUIT_BREAKER_RECOVERY_TIMEOUT,
            expected_exception=(ConnectionError, TimeoutError, aiohttp.ClientError),
            success_threshold=CIRCUIT_BREAKER_SUCCESS_THRESHOLD
        )
        circuit_breaker_registry.register(breaker)
    
    return breaker

# ============================================================================
# Global Instances
# ============================================================================

embedding_cache = TTLCache()
connection_manager = ConnectionPoolManager()
limiter = Limiter(key_func=get_remote_address)

# Helper to conditionally apply rate limiting
def apply_rate_limit(limit_string: str):
    """Apply rate limiting unless we're in test mode"""
    if os.getenv("TESTING", "").lower() == "true":
        # In test mode, return a no-op decorator
        return lambda f: f
    else:
        # In production, apply the actual rate limit
        return limiter.limit(limit_string)

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
    logger.info("Starting embeddings service v5 enhanced (with circuit breaker)")
    
    await embedding_cache.start_cleanup_task()
    
    if not EMBEDDINGS_AVAILABLE:
        logger.error("Embeddings implementation not available - service will not function")
        
    logger.info("Embeddings service started successfully")

@router.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown"""
    logger.info("Shutting down embeddings service")
    
    await embedding_cache.stop_cleanup_task()
    await connection_manager.close_all()
    
    logger.info("Embeddings service shutdown complete")

# Register cleanup on process exit
def cleanup_on_exit():
    """Synchronous cleanup for process exit"""
    try:
        loop = asyncio.get_event_loop()
        if not loop.is_closed():
            loop.run_until_complete(connection_manager.close_all())
    except Exception as e:
        logger.error(f"Error during exit cleanup: {e}")

atexit.register(cleanup_on_exit)

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
    elif provider == EmbeddingProvider.COHERE:
        return {
            "provider": "cohere",
            "model_name_or_path": model,
            "api_key": api_key or settings.get("COHERE_API_KEY"),
        }
    elif provider == EmbeddingProvider.VOYAGE:
        return {
            "provider": "voyage",
            "model_name_or_path": model,
            "api_key": api_key or settings.get("VOYAGE_API_KEY"),
        }
    elif provider == EmbeddingProvider.GOOGLE:
        return {
            "provider": "google",
            "model_name_or_path": model,
            "api_key": api_key or settings.get("GOOGLE_API_KEY"),
        }
    elif provider == EmbeddingProvider.MISTRAL:
        return {
            "provider": "mistral",
            "model_name_or_path": model,
            "api_key": api_key or settings.get("MISTRAL_API_KEY"),
        }
    elif provider == EmbeddingProvider.ONNX:
        return {
            "provider": "onnx",
            "model_name_or_path": model,
        }
    elif provider == EmbeddingProvider.LOCAL_API:
        return {
            "provider": "local_api",
            "model_name_or_path": model,
            "api_url": api_url or settings.get("LOCAL_API_URL"),
        }
    else:
        raise ValueError(f"Unknown provider: {provider}")

# ============================================================================
# Enhanced Embedding Function with Circuit Breaker
# ============================================================================

async def create_embeddings_with_circuit_breaker(
    texts: List[str],
    provider: str,
    model_id: str,
    config: Dict[str, Any]
) -> List[List[float]]:
    """Create embeddings with circuit breaker protection"""
    
    breaker = get_or_create_circuit_breaker(provider)
    
    try:
        # Use circuit breaker to protect the call
        async def _create():
            # Build proper ModelCfg based on provider
            model_cfg = {
                "provider": provider,
                "model_name_or_path": config.get("model_name_or_path", model_id),
            }
            
            # Add provider-specific fields
            if provider == "huggingface":
                model_cfg["trust_remote_code"] = config.get("trust_remote_code", False)
                model_cfg["hf_cache_dir_subpath"] = config.get("hf_cache_dir_subpath", "huggingface_cache")
                model_cfg["device"] = config.get("device", "cpu")
            elif provider == "openai":
                model_cfg["api_key"] = config.get("api_key")
            elif provider == "onnx":
                model_cfg["onnx_storage_dir_subpath"] = config.get("onnx_storage_dir_subpath", "onnx_models")
            else:
                # For other providers, just pass through config
                model_cfg.update(config)
            
            # Wrap config in expected structure for create_embeddings_batch
            app_config = {
                "embedding_config": {
                    "default_model_id": model_id,
                    "model_storage_base_dir": "./embedding_models_data/",
                    "models": {
                        model_id: model_cfg
                    }
                }
            }
            
            loop = asyncio.get_event_loop()
            return await loop.run_in_executor(
                None,
                create_embeddings_batch,
                texts,
                app_config,  # Pass wrapped config
                model_id
            )
        
        return await breaker.call_async(_create)
        
    except CircuitBreakerError as e:
        logger.warning(f"Circuit breaker open for {provider}: {e}")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Service temporarily unavailable for provider {provider}. Please try again later."
        )
    except Exception as e:
        logger.error(f"Failed to create embeddings with {provider}: {e}")
        raise

async def create_embeddings_batch_async(
    texts: List[str],
    provider: str,
    model_id: Optional[str] = None,
    dimensions: Optional[int] = None,
    api_key: Optional[str] = None,
    api_url: Optional[str] = None
) -> List[List[float]]:
    """Async wrapper for embeddings with caching and circuit breaker"""
    
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
        try:
            provider_enum = EmbeddingProvider(provider)
        except ValueError:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Unknown provider: {provider}"
            )
        
        config = build_provider_config(
            provider_enum,
            model_id,
            api_key,
            api_url,
            dimensions
        )
        
        # Process in batches with circuit breaker
        all_new_embeddings = []
        for batch_start in range(0, len(uncached_texts), MAX_BATCH_SIZE):
            batch_end = min(batch_start + MAX_BATCH_SIZE, len(uncached_texts))
            batch_texts = uncached_texts[batch_start:batch_end]
            
            try:
                batch_embeddings = await create_embeddings_with_circuit_breaker(
                    batch_texts,
                    provider,
                    model_id,
                    config
                )
                all_new_embeddings.extend(batch_embeddings)
            except HTTPException:
                raise
            except Exception as e:
                logger.error(f"Failed to create embeddings for batch: {e}")
                
                # Try to close and recreate connection for this provider
                await connection_manager.remove_provider(provider)
                
                raise HTTPException(
                    status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                    detail=f"Embedding service error: {str(e)}"
                )
        
        # Update results and cache
        for i, (idx, text) in enumerate(zip(uncached_indices, uncached_texts)):
            embedding = all_new_embeddings[i]
            embeddings[idx] = embedding
            
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
    summary="Create embeddings (enhanced with circuit breaker)"
)
@apply_rate_limit("60/minute")
async def create_embedding_endpoint(
    request: Request,
    embedding_request: CreateEmbeddingRequest = Body(...),
    current_user: User = Depends(get_request_user),
    background_tasks: BackgroundTasks = BackgroundTasks(),
    x_provider: Optional[str] = Header(None, alias="x-provider")
):
    """Create embeddings with circuit breaker protection and enhanced error recovery"""
    
    active_embedding_requests.inc()
    start_time = time.time()
    
    try:
        # Validate provider
        provider = x_provider or "openai"
        model = embedding_request.model
        
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
        
        # Create embeddings with circuit breaker
        try:
            embeddings = await create_embeddings_batch_async(
                texts=texts_to_embed,
                provider=provider,
                model_id=model,
                dimensions=embedding_request.dimensions
            )
        except HTTPException:
            raise
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
    summary="Health check with circuit breaker status"
)
async def health_check():
    """Enhanced health check with circuit breaker status"""
    
    # Get circuit breaker status for all providers
    breaker_status = {}
    for provider in EmbeddingProvider:
        breaker_name = f"embeddings_{provider.value}"
        breaker = circuit_breaker_registry.get(breaker_name)
        if breaker:
            status_info = breaker.get_status()
            breaker_status[provider.value] = {
                "state": status_info["state"],
                "failure_count": status_info["failure_count"],
                "last_failure": status_info["last_failure_time"]
            }
    
    health_status = {
        "status": "healthy" if EMBEDDINGS_AVAILABLE else "degraded",
        "service": "embeddings_v5_production_enhanced",
        "timestamp": datetime.utcnow().isoformat(),
        "cache_stats": embedding_cache.stats(),
        "active_requests": active_embedding_requests._value.get(),
        "circuit_breakers": breaker_status
    }
    
    if not EMBEDDINGS_AVAILABLE:
        return JSONResponse(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            content=health_status
        )
    
    return health_status

@router.get(
    "/embeddings/circuit-breakers",
    summary="Get circuit breaker status (admin only)"
)
async def get_circuit_breakers(
    current_user: User = Depends(get_request_user)
):
    """Get detailed circuit breaker status - requires admin privileges"""
    
    require_admin(current_user)
    
    return circuit_breaker_registry.get_all_status()

@router.post(
    "/embeddings/circuit-breakers/{provider}/reset",
    summary="Reset circuit breaker (admin only)"
)
async def reset_circuit_breaker(
    provider: str,
    current_user: User = Depends(get_request_user)
):
    """Reset specific circuit breaker - requires admin privileges"""
    
    require_admin(current_user)
    
    breaker_name = f"embeddings_{provider}"
    breaker = circuit_breaker_registry.get(breaker_name)
    
    if not breaker:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Circuit breaker for provider '{provider}' not found"
        )
    
    breaker.reset()
    
    logger.info(
        f"Circuit breaker reset by admin",
        extra={
            "admin_id": current_user.id,
            "provider": provider
        }
    )
    
    return {
        "message": f"Circuit breaker for '{provider}' reset successfully"
    }

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
        "circuit_breakers": circuit_breaker_registry.get_all_status(),
        "total_requests": {
            "success": embedding_requests_total.labels(
                provider="all", model="all", status="success"
            )._value.get(),
            "error": embedding_requests_total.labels(
                provider="all", model="all", status="error"
            )._value.get()
        }
    }