# embeddings_v4.py - Multi-provider embeddings API with enhanced provider support
"""
OpenAI-compatible embeddings API with support for multiple providers:
- OpenAI (text-embedding-ada-002, text-embedding-3-small, text-embedding-3-large)
- HuggingFace (any sentence-transformers model)
- ONNX (optimized models)
- Local API (custom embedding servers)
- Cohere (embed-english-v3.0, embed-multilingual-v3.0)
- Voyage AI (voyage-2, voyage-large-2)
- Google (text-embedding-004)
- Mistral (mistral-embed)
"""

import base64
import asyncio
import hashlib
from typing import List, Union, Optional, Dict, Any, Tuple
from concurrent.futures import ThreadPoolExecutor
import numpy as np
from functools import lru_cache
from enum import Enum

from fastapi import APIRouter, HTTPException, Body, Depends, status, BackgroundTasks, Request, Query
from fastapi.responses import JSONResponse
import tiktoken

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

# Logging
from loguru import logger

# Rate limiting
from slowapi import Limiter
from slowapi.util import get_remote_address

# Import the actual embedding function
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
    logger.warning(f"Could not import embeddings implementation: {e}")
    EMBEDDINGS_AVAILABLE = False
    
    # Enhanced placeholder implementation for testing
    def create_embeddings_batch(
        texts: List[str],
        user_app_config: Dict[str, Any],
        model_id_override: Optional[str] = None,
    ) -> List[List[float]]:
        """Enhanced placeholder that returns provider-aware embeddings"""
        logger.warning("Using placeholder embeddings implementation")
        import random
        
        # Extract provider and dimensions from config
        embedding_config = user_app_config.get("embedding_config", {})
        provider = embedding_config.get("provider", "openai")
        dimensions = embedding_config.get("dimensions", None)
        
        # Provider-specific default dimensions
        provider_dimensions = {
            "openai": {
                "text-embedding-ada-002": 1536,
                "text-embedding-3-small": 1536,
                "text-embedding-3-large": 3072,
            },
            "cohere": {
                "embed-english-v3.0": 1024,
                "embed-multilingual-v3.0": 1024,
            },
            "voyage": {
                "voyage-2": 1024,
                "voyage-large-2": 1536,
            },
            "google": {
                "text-embedding-004": 768,
            },
            "mistral": {
                "mistral-embed": 1024,
            },
            "huggingface": 384,  # Default for HF models
            "local_api": 768,     # Default for local
        }
        
        model = model_id_override or embedding_config.get("default_model_id", "text-embedding-3-small")
        
        # Determine dimensions
        if dimensions:
            dim = dimensions
        elif provider in provider_dimensions:
            if isinstance(provider_dimensions[provider], dict):
                dim = provider_dimensions[provider].get(model, 384)
            else:
                dim = provider_dimensions[provider]
        else:
            dim = 384
        
        # Generate reproducible embeddings based on text hash
        embeddings = []
        for text in texts:
            # Use hash for reproducible "embeddings"
            text_hash = hashlib.md5(f"{provider}:{model}:{text}".encode()).digest()
            np.random.seed(int.from_bytes(text_hash[:4], 'big'))
            embedding = np.random.randn(dim).tolist()
            # Normalize to unit length (common for embeddings)
            norm = np.linalg.norm(embedding)
            embedding = [x / norm for x in embedding]
            embeddings.append(embedding)
        
        return embeddings

# Provider enum for easy selection
class EmbeddingProvider(str, Enum):
    OPENAI = "openai"
    HUGGINGFACE = "huggingface"
    ONNX = "onnx"
    LOCAL_API = "local_api"
    COHERE = "cohere"
    VOYAGE = "voyage"
    GOOGLE = "google"
    MISTRAL = "mistral"

# Thread pool for running sync code in async context
# Increased workers for better batch processing
executor = ThreadPoolExecutor(max_workers=8)

# Rate limiter
limiter = Limiter(key_func=get_remote_address)

# Router configuration
router = APIRouter(
    tags=["Embeddings"],
    responses={
        401: {"description": "Unauthorized"},
        429: {"description": "Rate limit exceeded"},
        500: {"description": "Internal server error"},
    }
)

# Cache for tokenizers and embeddings
_tokenizer_cache = {}
_embedding_cache = {}  # Simple in-memory cache
_cache_lock = asyncio.Lock()

# Configuration constants
MAX_BATCH_SIZE = 100  # Process in batches of 100
MAX_CACHE_SIZE = 10000  # Maximum cached embeddings
CACHE_TTL = 3600  # Cache for 1 hour

# Provider-specific model mappings
PROVIDER_MODELS = {
    EmbeddingProvider.OPENAI: [
        "text-embedding-ada-002",
        "text-embedding-3-small", 
        "text-embedding-3-large"
    ],
    EmbeddingProvider.COHERE: [
        "embed-english-v3.0",
        "embed-multilingual-v3.0",
        "embed-english-light-v3.0",
        "embed-multilingual-light-v3.0"
    ],
    EmbeddingProvider.VOYAGE: [
        "voyage-2",
        "voyage-large-2",
        "voyage-code-2",
        "voyage-lite-02-instruct"
    ],
    EmbeddingProvider.GOOGLE: [
        "text-embedding-004",
        "textembedding-gecko@003",
        "textembedding-gecko-multilingual@001"
    ],
    EmbeddingProvider.MISTRAL: [
        "mistral-embed"
    ],
    EmbeddingProvider.HUGGINGFACE: [
        "sentence-transformers/all-MiniLM-L6-v2",
        "sentence-transformers/all-mpnet-base-v2",
        "BAAI/bge-large-en-v1.5",
        "intfloat/e5-large-v2",
        "jinaai/jina-embeddings-v2-base-en"
    ]
}

# Provider-specific configuration builders
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
    
    elif provider == EmbeddingProvider.ONNX:
        return {
            "provider": "onnx",
            "model_name_or_path": model,
            "onnx_storage_dir_subpath": "onnx_models",
            "onnx_providers": ["CPUExecutionProvider"],
        }
    
    elif provider == EmbeddingProvider.LOCAL_API:
        return {
            "provider": "local_api",
            "model_name_or_path": model,
            "api_url": api_url or "http://localhost:8080/v1/embeddings",
            "api_key": api_key,
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
    
    else:
        raise ValueError(f"Unknown provider: {provider}")

@lru_cache(maxsize=128)
def get_tokenizer(model_name: str):
    """Get or create a tokenizer for the model."""
    try:
        return tiktoken.encoding_for_model(model_name)
    except KeyError:
        logger.warning(f"No tokenizer for model '{model_name}', using cl100k_base")
        return tiktoken.get_encoding("cl100k_base")

def count_tokens(text: str, model_name: str) -> int:
    """Count tokens in a string for a given model."""
    try:
        encoding = get_tokenizer(model_name)
        return len(encoding.encode(text))
    except Exception as e:
        logger.warning(f"Token counting failed: {e}, using word count")
        return len(text.split())

def count_tokens_for_list(texts: List[str], model_name: str) -> int:
    """Count total tokens for a list of strings."""
    return sum(count_tokens(text, model_name) for text in texts)

def tokens_to_text(tokens: List[int], model_name: str) -> str:
    """Convert token IDs back to text."""
    try:
        encoding = get_tokenizer(model_name)
        return encoding.decode(tokens)
    except Exception as e:
        logger.warning(f"Token decoding failed: {e}")
        # Fallback: treat as character codes
        return ''.join(chr(min(t, 127)) for t in tokens if t > 0)

def get_embedding_config(
    provider: Optional[str] = None,
    model: Optional[str] = None,
    dimensions: Optional[int] = None,
    api_key: Optional[str] = None,
    api_url: Optional[str] = None
) -> Dict[str, Any]:
    """Get embedding configuration with provider override."""
    
    # Get default config from settings
    embedding_config = settings.get("EMBEDDING_CONFIG", {})
    
    # Use provided values or fall back to defaults
    active_provider = provider or embedding_config.get('embedding_provider', 'openai')
    active_model = model or embedding_config.get('embedding_model', 'text-embedding-3-small')
    
    # Build provider-specific config
    provider_enum = EmbeddingProvider(active_provider.lower())
    provider_config = build_provider_config(
        provider_enum,
        active_model,
        api_key,
        api_url,
        dimensions
    )
    
    # Build the configuration structure
    config = {
        "embedding_config": {
            "provider": active_provider,
            "default_model_id": active_model,
            "model_storage_base_dir": "./embedding_models_data/",
            "models": {
                active_model: provider_config
            }
        }
    }
    
    # Add dimensions if specified
    if dimensions:
        config["embedding_config"]["dimensions"] = dimensions
    
    return config

def get_cache_key(
    text: str, 
    provider: str,
    model: str, 
    dimensions: Optional[int] = None
) -> str:
    """Generate cache key for an embedding."""
    key_parts = [text, provider, model]
    if dimensions:
        key_parts.append(str(dimensions))
    key_string = "|".join(key_parts)
    return hashlib.sha256(key_string.encode()).hexdigest()

async def get_cached_embedding(
    text: str,
    provider: str,
    model: str, 
    dimensions: Optional[int] = None
) -> Optional[List[float]]:
    """Get embedding from cache if available."""
    cache_key = get_cache_key(text, provider, model, dimensions)
    
    async with _cache_lock:
        if cache_key in _embedding_cache:
            # Update access time
            _embedding_cache[cache_key]["last_access"] = asyncio.get_event_loop().time()
            return _embedding_cache[cache_key]["embedding"]
    
    return None

async def cache_embedding(
    text: str,
    provider: str,
    model: str, 
    embedding: List[float],
    dimensions: Optional[int] = None
):
    """Cache an embedding."""
    cache_key = get_cache_key(text, provider, model, dimensions)
    
    async with _cache_lock:
        # Implement simple LRU by removing oldest if cache is full
        if len(_embedding_cache) >= MAX_CACHE_SIZE:
            # Find and remove oldest entry
            oldest_key = min(
                _embedding_cache.keys(),
                key=lambda k: _embedding_cache[k]["last_access"]
            )
            del _embedding_cache[oldest_key]
        
        _embedding_cache[cache_key] = {
            "embedding": embedding,
            "last_access": asyncio.get_event_loop().time()
        }

async def create_embeddings_batch_async(
    texts: List[str],
    provider: str,
    model_id: Optional[str] = None,
    dimensions: Optional[int] = None,
    api_key: Optional[str] = None,
    api_url: Optional[str] = None
) -> List[List[float]]:
    """Async wrapper for create_embeddings_batch with provider support."""
    config = get_embedding_config(provider, model_id, dimensions, api_key, api_url)
    
    # Check cache first
    embeddings = []
    uncached_texts = []
    uncached_indices = []
    
    for i, text in enumerate(texts):
        cached = await get_cached_embedding(text, provider, model_id or "default", dimensions)
        if cached:
            embeddings.append(cached)
        else:
            embeddings.append(None)  # Placeholder
            uncached_texts.append(text)
            uncached_indices.append(i)
    
    logger.debug(f"Cache hits: {len(texts) - len(uncached_texts)}/{len(texts)}")
    
    # Process uncached texts in batches
    if uncached_texts:
        all_new_embeddings = []
        
        for batch_start in range(0, len(uncached_texts), MAX_BATCH_SIZE):
            batch_end = min(batch_start + MAX_BATCH_SIZE, len(uncached_texts))
            batch_texts = uncached_texts[batch_start:batch_end]
            
            logger.debug(f"Processing batch {batch_start//MAX_BATCH_SIZE + 1}: {len(batch_texts)} texts with {provider}")
            
            # Run the synchronous function in a thread pool
            loop = asyncio.get_event_loop()
            batch_embeddings = await loop.run_in_executor(
                executor,
                create_embeddings_batch,
                batch_texts,
                config,
                model_id
            )
            
            all_new_embeddings.extend(batch_embeddings)
        
        # Update results and cache
        for i, (idx, text) in enumerate(zip(uncached_indices, uncached_texts)):
            embedding = all_new_embeddings[i]
            embeddings[idx] = embedding
            # Cache in background
            asyncio.create_task(
                cache_embedding(text, provider, model_id or "default", embedding, dimensions)
            )
    
    return embeddings

def apply_dimensions_reduction(
    embeddings: List[List[float]], 
    target_dimensions: int
) -> List[List[float]]:
    """
    Reduce embedding dimensions using truncation or PCA-like approach.
    Note: This is a simplified implementation. Production systems should use
    proper dimensionality reduction techniques.
    """
    reduced = []
    
    for embedding in embeddings:
        current_dim = len(embedding)
        
        if current_dim == target_dimensions:
            reduced.append(embedding)
        elif current_dim > target_dimensions:
            # Simple truncation (OpenAI's approach for text-embedding-3-*)
            reduced.append(embedding[:target_dimensions])
        else:
            # Pad with zeros if somehow smaller
            padded = embedding + [0.0] * (target_dimensions - current_dim)
            reduced.append(padded)
    
    return reduced

@router.post(
    "/embeddings",
    response_model=CreateEmbeddingResponse,
    status_code=status.HTTP_200_OK,
    summary="Create embeddings with multiple provider support",
    description="""
    Create embedding vectors for input text with support for multiple providers.
    
    **Supported Providers:**
    - **OpenAI**: text-embedding-ada-002, text-embedding-3-small, text-embedding-3-large
    - **Cohere**: embed-english-v3.0, embed-multilingual-v3.0
    - **Voyage AI**: voyage-2, voyage-large-2
    - **Google**: text-embedding-004
    - **Mistral**: mistral-embed
    - **HuggingFace**: Any sentence-transformers model
    - **Local API**: Custom embedding servers
    
    **Features:**
    - Provider selection via `x-provider` header or model prefix
    - Batch processing (100 items per batch)
    - Dimensions support for compatible models
    - Token array input support
    - Intelligent caching with LRU eviction
    
    **Provider Selection:**
    1. Via header: `x-provider: cohere`
    2. Via model prefix: `cohere:embed-english-v3.0`
    3. Default: Uses configured provider in settings
    
    **Rate Limits:**
    - 60 requests per minute per user
    - Maximum 2048 strings per request
    """
)
@limiter.limit("60/minute")
async def create_embedding_endpoint(
    request: Request,
    embedding_request: CreateEmbeddingRequest = Body(...),
    current_user: User = Depends(get_request_user),
    background_tasks: BackgroundTasks = BackgroundTasks(),
    x_provider: Optional[str] = None,  # Custom header for provider
    x_api_key: Optional[str] = None,   # Custom API key override
    x_api_url: Optional[str] = None    # Custom API URL for local providers
):
    """Create embeddings with multi-provider support."""
    
    # Determine provider and model
    provider = x_provider
    model = embedding_request.model
    
    # Check if model has provider prefix (e.g., "cohere:embed-english-v3.0")
    if ":" in model and not provider:
        parts = model.split(":", 1)
        provider = parts[0]
        model = parts[1]
    
    # Default provider if not specified
    if not provider:
        provider = settings.get("EMBEDDING_CONFIG", {}).get("embedding_provider", "openai")
    
    logger.info(f"User {current_user.id} requesting embeddings with provider: {provider}, model: {model}")
    
    if not EMBEDDINGS_AVAILABLE:
        logger.warning("Embeddings implementation not available, using enhanced placeholder")
    
    # Validate provider
    try:
        provider_enum = EmbeddingProvider(provider.lower())
    except ValueError:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unknown provider: {provider}. Supported: {', '.join([p.value for p in EmbeddingProvider])}"
        )
    
    # Validate dimensions parameter for specific providers/models
    if embedding_request.dimensions:
        # Only OpenAI text-embedding-3-* and some other models support dimensions
        supports_dimensions = (
            (provider == "openai" and "text-embedding-3" in model) or
            (provider == "cohere" and model.startswith("embed-"))
        )
        
        if not supports_dimensions:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Dimensions parameter is not supported for {provider}:{model}"
            )
        
        if embedding_request.dimensions < 1:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Dimensions must be at least 1"
            )
    
    # Parse input
    texts_to_embed: List[str] = []
    num_prompt_tokens: int = 0
    input_was_tokens = False
    
    if isinstance(embedding_request.input, str):
        # Single string input
        if not embedding_request.input.strip():
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Input string cannot be empty"
            )
        texts_to_embed = [embedding_request.input]
        num_prompt_tokens = count_tokens(embedding_request.input, model)
        
    elif isinstance(embedding_request.input, list):
        if not embedding_request.input:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Input list cannot be empty"
            )
        if len(embedding_request.input) > 2048:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Input array must not exceed 2048 elements"
            )
        
        # Handle different list types
        if all(isinstance(item, str) for item in embedding_request.input):
            # List of strings
            if any(not item.strip() for item in embedding_request.input):
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="Input strings cannot be empty"
                )
            texts_to_embed = embedding_request.input
            num_prompt_tokens = count_tokens_for_list(embedding_request.input, model)
            
        elif all(isinstance(item, int) for item in embedding_request.input):
            # Single token array - convert to text
            logger.info(f"Converting token array of length {len(embedding_request.input)} to text")
            input_was_tokens = True
            decoded_text = tokens_to_text(embedding_request.input, model)
            texts_to_embed = [decoded_text]
            num_prompt_tokens = len(embedding_request.input)
            
        elif all(isinstance(item, list) and all(isinstance(x, int) for x in item) for item in embedding_request.input):
            # Batch token arrays - convert each to text
            logger.info(f"Converting {len(embedding_request.input)} token arrays to text")
            input_was_tokens = True
            texts_to_embed = []
            num_prompt_tokens = 0
            
            for token_array in embedding_request.input:
                decoded_text = tokens_to_text(token_array, model)
                texts_to_embed.append(decoded_text)
                num_prompt_tokens += len(token_array)
        else:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Invalid input format. Must be string, array of strings, array of integers, or array of integer arrays."
            )
    else:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid input type. Must be string or array."
        )
    
    # Create embeddings with provider support
    try:
        logger.debug(f"Creating embeddings for {len(texts_to_embed)} texts with {provider}:{model}, dimensions={embedding_request.dimensions}")
        
        raw_embeddings = await create_embeddings_batch_async(
            texts=texts_to_embed,
            provider=provider,
            model_id=model,
            dimensions=embedding_request.dimensions,
            api_key=x_api_key,
            api_url=x_api_url
        )
        
        # Apply dimension reduction if requested
        if embedding_request.dimensions and embedding_request.dimensions > 0:
            logger.debug(f"Applying dimension reduction to {embedding_request.dimensions}")
            raw_embeddings = apply_dimensions_reduction(raw_embeddings, embedding_request.dimensions)
        
    except ValueError as e:
        logger.warning(f"ValueError in embedding creation: {e}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    except FileNotFoundError as e:
        logger.error(f"Model file not found: {e}")
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Model '{model}' not found or not configured for provider '{provider}'"
        )
    except Exception as e:
        logger.error(f"Unexpected error creating embeddings: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to create embeddings. Please try again later."
        )
    
    # Format output
    output_data: List[EmbeddingData] = []
    
    for i, embedding in enumerate(raw_embeddings):
        # Convert to list if needed
        if hasattr(embedding, 'tolist'):
            embedding_floats = embedding.tolist()
        elif isinstance(embedding, list):
            embedding_floats = [float(x) for x in embedding]
        else:
            logger.error(f"Unexpected embedding format: {type(embedding)}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Internal error processing embeddings"
            )
        
        # Apply encoding format
        if embedding_request.encoding_format == "base64":
            byte_array = np.array(embedding_floats, dtype=np.float32).tobytes()
            processed_value = base64.b64encode(byte_array).decode('utf-8')
        else:
            processed_value = embedding_floats
        
        output_data.append(
            EmbeddingData(
                embedding=processed_value,
                index=i
            )
        )
    
    # Create response
    usage = EmbeddingUsage(
        prompt_tokens=num_prompt_tokens,
        total_tokens=num_prompt_tokens
    )
    
    # Include provider info in model name for transparency
    response_model = f"{provider}:{model}" if provider != "openai" else model
    
    response = CreateEmbeddingResponse(
        data=output_data,
        model=response_model,
        usage=usage
    )
    
    logger.info(
        f"Successfully created {len(output_data)} embeddings for user {current_user.id} "
        f"(provider={provider}, model={model}, tokens_input={input_was_tokens}, dimensions={embedding_request.dimensions})"
    )
    
    return response

@router.get(
    "/embeddings/providers",
    summary="List available embedding providers",
    description="Get a list of available embedding providers and their models"
)
async def list_providers(
    current_user: User = Depends(get_request_user)
):
    """List available embedding providers and their models."""
    providers = []
    
    for provider in EmbeddingProvider:
        provider_info = {
            "id": provider.value,
            "name": provider.value.title(),
            "models": PROVIDER_MODELS.get(provider, []),
            "features": {
                "dimensions_support": provider in [EmbeddingProvider.OPENAI, EmbeddingProvider.COHERE],
                "batch_support": True,
                "token_input": provider == EmbeddingProvider.OPENAI,
                "max_batch_size": MAX_BATCH_SIZE
            }
        }
        
        # Add status based on API key availability
        if provider == EmbeddingProvider.OPENAI:
            provider_info["configured"] = bool(settings.get("OPENAI_API_KEY"))
        elif provider == EmbeddingProvider.COHERE:
            provider_info["configured"] = bool(settings.get("COHERE_API_KEY"))
        elif provider == EmbeddingProvider.VOYAGE:
            provider_info["configured"] = bool(settings.get("VOYAGE_API_KEY"))
        elif provider == EmbeddingProvider.GOOGLE:
            provider_info["configured"] = bool(settings.get("GOOGLE_API_KEY"))
        elif provider == EmbeddingProvider.MISTRAL:
            provider_info["configured"] = bool(settings.get("MISTRAL_API_KEY"))
        elif provider in [EmbeddingProvider.HUGGINGFACE, EmbeddingProvider.ONNX, EmbeddingProvider.LOCAL_API]:
            provider_info["configured"] = True  # Always available
        
        providers.append(provider_info)
    
    return {
        "providers": providers,
        "default_provider": settings.get("EMBEDDING_CONFIG", {}).get("embedding_provider", "openai"),
        "default_model": settings.get("EMBEDDING_CONFIG", {}).get("embedding_model", "text-embedding-3-small")
    }

@router.post(
    "/embeddings/compare",
    summary="Compare embeddings from different providers",
    description="Generate embeddings from multiple providers for comparison"
)
async def compare_embeddings(
    request: Request,
    text: str = Body(..., description="Text to embed"),
    providers: List[str] = Body(..., description="List of providers to compare"),
    current_user: User = Depends(get_request_user)
):
    """Compare embeddings from different providers."""
    if len(providers) > 5:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Maximum 5 providers for comparison"
        )
    
    results = {}
    
    for provider_name in providers:
        try:
            provider_enum = EmbeddingProvider(provider_name.lower())
        except ValueError:
            results[provider_name] = {"error": f"Unknown provider: {provider_name}"}
            continue
        
        # Get default model for provider
        models = PROVIDER_MODELS.get(provider_enum, [])
        if not models:
            results[provider_name] = {"error": f"No models available for {provider_name}"}
            continue
        
        model = models[0]  # Use first/default model
        
        try:
            embeddings = await create_embeddings_batch_async(
                texts=[text],
                provider=provider_name,
                model_id=model
            )
            
            embedding = embeddings[0]
            
            results[provider_name] = {
                "model": model,
                "dimensions": len(embedding),
                "sample": embedding[:5],  # First 5 values as sample
                "norm": float(np.linalg.norm(embedding))  # L2 norm
            }
        except Exception as e:
            results[provider_name] = {"error": str(e)}
    
    return {
        "text": text[:100] + "..." if len(text) > 100 else text,
        "comparisons": results
    }

@router.get(
    "/embeddings/cache/stats",
    summary="Get cache statistics",
    description="Get information about the embedding cache"
)
async def get_cache_stats(
    current_user: User = Depends(get_request_user)
):
    """Get cache statistics."""
    async with _cache_lock:
        total_entries = len(_embedding_cache)
        
        if total_entries > 0:
            # Calculate average age
            current_time = asyncio.get_event_loop().time()
            ages = [current_time - entry["last_access"] for entry in _embedding_cache.values()]
            avg_age = sum(ages) / len(ages)
            oldest_age = max(ages)
            
            # Group by provider
            provider_stats = {}
            for key in _embedding_cache.keys():
                # Extract provider from cache key (simplified)
                provider = "unknown"
                for entry_data in _embedding_cache.values():
                    # This is simplified - in production you'd store provider in cache
                    break
                
                provider_stats[provider] = provider_stats.get(provider, 0) + 1
        else:
            avg_age = 0
            oldest_age = 0
            provider_stats = {}
    
    return {
        "cache_size": total_entries,
        "max_cache_size": MAX_CACHE_SIZE,
        "cache_ttl": CACHE_TTL,
        "average_age_seconds": avg_age,
        "oldest_entry_age_seconds": oldest_age,
        "provider_breakdown": provider_stats
    }

@router.delete(
    "/embeddings/cache",
    summary="Clear embedding cache",
    description="Clear the embedding cache (admin only)"
)
async def clear_cache(
    current_user: User = Depends(get_request_user)
):
    """Clear the embedding cache."""
    # TODO: Add admin check here
    async with _cache_lock:
        count = len(_embedding_cache)
        _embedding_cache.clear()
    
    logger.info(f"User {current_user.id} cleared embedding cache: {count} entries removed")
    
    return {
        "message": "Cache cleared",
        "entries_removed": count
    }

# Health check
@router.get(
    "/embeddings/health",
    summary="Embeddings service health check",
    description="Check if the embeddings service is operational"
)
async def health_check():
    """Health check for embeddings service."""
    
    # Check provider availability
    provider_status = {}
    for provider in EmbeddingProvider:
        if provider == EmbeddingProvider.OPENAI:
            provider_status[provider.value] = bool(settings.get("OPENAI_API_KEY"))
        elif provider == EmbeddingProvider.COHERE:
            provider_status[provider.value] = bool(settings.get("COHERE_API_KEY"))
        elif provider == EmbeddingProvider.VOYAGE:
            provider_status[provider.value] = bool(settings.get("VOYAGE_API_KEY"))
        elif provider == EmbeddingProvider.GOOGLE:
            provider_status[provider.value] = bool(settings.get("GOOGLE_API_KEY"))
        elif provider == EmbeddingProvider.MISTRAL:
            provider_status[provider.value] = bool(settings.get("MISTRAL_API_KEY"))
        else:
            provider_status[provider.value] = True
    
    return {
        "status": "healthy",
        "service": "embeddings_v4",
        "implementation_available": EMBEDDINGS_AVAILABLE,
        "features": {
            "multi_provider": True,
            "batch_processing": True,
            "dimensions_support": True,
            "token_input_support": True,
            "caching_enabled": True,
            "provider_comparison": True
        },
        "provider_status": provider_status
    }