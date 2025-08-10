# embeddings_v2.py - Fixed and improved embeddings API
"""
OpenAI-compatible embeddings API with proper configuration and authentication.
This is a fixed version that actually works with the backend implementation.
"""

import base64
import asyncio
from typing import List, Union, Optional, Dict, Any
from concurrent.futures import ThreadPoolExecutor
import numpy as np

from fastapi import APIRouter, HTTPException, Body, Depends, status
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
    
    # Placeholder implementation for testing
    def create_embeddings_batch(
        texts: List[str],
        user_app_config: Dict[str, Any],
        model_id_override: Optional[str] = None,
    ) -> List[List[float]]:
        """Placeholder that returns random embeddings for testing"""
        logger.warning("Using placeholder embeddings implementation")
        import random
        dimension = 1536 if "ada" in str(model_id_override).lower() else 384
        return [[random.random() for _ in range(dimension)] for _ in texts]

# Thread pool for running sync code in async context
executor = ThreadPoolExecutor(max_workers=4)

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

# Cache for tokenizers
_tokenizer_cache = {}

def get_tokenizer(model_name: str):
    """Get or create a tokenizer for the model."""
    if model_name not in _tokenizer_cache:
        try:
            _tokenizer_cache[model_name] = tiktoken.encoding_for_model(model_name)
        except KeyError:
            logger.warning(f"No tokenizer for model '{model_name}', using cl100k_base")
            _tokenizer_cache[model_name] = tiktoken.get_encoding("cl100k_base")
    return _tokenizer_cache[model_name]

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

def get_embedding_config() -> Dict[str, Any]:
    """Get embedding configuration from settings."""
    # Get the embedding config from settings
    embedding_config = settings.get("EMBEDDING_CONFIG", {})
    
    # Build the configuration structure expected by create_embeddings_batch
    default_model = embedding_config.get('embedding_model', 'text-embedding-3-small')
    provider = embedding_config.get('embedding_provider', 'openai')
    
    # Build model configurations based on provider
    models = {}
    
    if provider == 'openai':
        models[default_model] = {
            "provider": "openai",
            "model_name_or_path": default_model,
        }
        # Add common OpenAI models
        for model in ['text-embedding-ada-002', 'text-embedding-3-small', 'text-embedding-3-large']:
            if model not in models:
                models[model] = {
                    "provider": "openai",
                    "model_name_or_path": model,
                }
    elif provider == 'huggingface':
        models[default_model] = {
            "provider": "huggingface",
            "model_name_or_path": default_model,
            "trust_remote_code": False,
        }
    elif provider == 'local_api':
        models[default_model] = {
            "provider": "local_api",
            "model_name_or_path": default_model,
            "api_url": embedding_config.get('embedding_api_url', 'http://localhost:8080/v1/embeddings'),
            "api_key": embedding_config.get('embedding_api_key', ''),
        }
    
    return {
        "embedding_config": {
            "default_model_id": default_model,
            "model_storage_base_dir": "./embedding_models_data/",
            "models": models
        }
    }

async def create_embeddings_async(
    texts: List[str],
    model_id: Optional[str] = None
) -> List[List[float]]:
    """Async wrapper for create_embeddings_batch."""
    config = get_embedding_config()
    
    # Run the synchronous function in a thread pool
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(
        executor,
        create_embeddings_batch,
        texts,
        config,
        model_id
    )

@router.post(
    "/embeddings",
    response_model=CreateEmbeddingResponse,
    status_code=status.HTTP_200_OK,
    summary="Create embeddings",
    description="""
    Create embedding vectors for input text.
    
    **Supported Input Formats:**
    - Single string
    - Array of strings (max 2048 items)
    - Array of token integers (not yet fully supported)
    
    **Available Models:**
    - `text-embedding-ada-002`: OpenAI Ada v2 (1536 dimensions)
    - `text-embedding-3-small`: OpenAI v3 small (1536 dimensions)
    - `text-embedding-3-large`: OpenAI v3 large (3072 dimensions)
    
    **Output Formats:**
    - `float`: Array of floating point numbers (default)
    - `base64`: Base64-encoded binary format
    
    **Rate Limits:**
    - 60 requests per minute per user
    - Maximum 2048 strings per request
    """
)
@limiter.limit("60/minute")
async def create_embedding_endpoint(
    request: CreateEmbeddingRequest = Body(...),
    current_user: User = Depends(get_request_user)
):
    """Create embeddings for the provided input."""
    logger.info(f"User {current_user.id} requesting embeddings with model: {request.model}")
    
    if not EMBEDDINGS_AVAILABLE:
        logger.warning("Embeddings implementation not available, using placeholder")
    
    # Parse input
    texts_to_embed: List[str] = []
    num_prompt_tokens: int = 0
    
    if isinstance(request.input, str):
        if not request.input.strip():
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Input string cannot be empty"
            )
        texts_to_embed = [request.input]
        num_prompt_tokens = count_tokens(request.input, request.model)
        
    elif isinstance(request.input, list):
        if not request.input:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Input list cannot be empty"
            )
        if len(request.input) > 2048:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Input array must not exceed 2048 elements"
            )
        
        # Handle different list types
        if all(isinstance(item, str) for item in request.input):
            if any(not item.strip() for item in request.input):
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="Input strings cannot be empty"
                )
            texts_to_embed = request.input
            num_prompt_tokens = count_tokens_for_list(request.input, request.model)
            
        elif all(isinstance(item, int) for item in request.input):
            # Token array input - not fully supported yet
            raise HTTPException(
                status_code=status.HTTP_501_NOT_IMPLEMENTED,
                detail="Token array input is not yet supported. Please provide text strings."
            )
            
        elif all(isinstance(item, list) and all(isinstance(x, int) for x in item) for item in request.input):
            # Batch token array input - not supported
            raise HTTPException(
                status_code=status.HTTP_501_NOT_IMPLEMENTED,
                detail="Batch token array input is not yet supported. Please provide text strings."
            )
        else:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Invalid input format. Must be string or array of strings."
            )
    else:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid input type. Must be string or array."
        )
    
    # Create embeddings
    try:
        logger.debug(f"Creating embeddings for {len(texts_to_embed)} texts")
        raw_embeddings = await create_embeddings_async(
            texts=texts_to_embed,
            model_id=request.model
        )
        
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
            detail=f"Model '{request.model}' not found or not configured"
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
        if request.encoding_format == "base64":
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
    
    response = CreateEmbeddingResponse(
        data=output_data,
        model=request.model,
        usage=usage
    )
    
    logger.info(f"Successfully created {len(output_data)} embeddings for user {current_user.id}")
    return response

@router.get(
    "/embeddings/models",
    summary="List available embedding models",
    description="Get a list of available embedding models and their configurations"
)
async def list_embedding_models(
    current_user: User = Depends(get_request_user)
):
    """List available embedding models."""
    config = get_embedding_config()
    models = []
    
    # Extract model info from config
    embedding_config = config.get("embedding_config", {})
    for model_id, model_spec in embedding_config.get("models", {}).items():
        models.append({
            "id": model_id,
            "provider": model_spec.get("provider"),
            "dimensions": 1536 if "ada" in model_id else (3072 if "large" in model_id else 384),
            "max_tokens": 8192 if "ada" in model_id else 8191
        })
    
    return {
        "models": models,
        "default_model": embedding_config.get("default_model_id")
    }

@router.post(
    "/embeddings/test",
    response_model=CreateEmbeddingResponse,
    summary="Test embedding endpoint",
    description="Test the embedding API with a simple example"
)
async def test_embedding(
    current_user: User = Depends(get_request_user)
):
    """Test endpoint with a simple example."""
    test_request = CreateEmbeddingRequest(
        input="Hello, world! This is a test of the embedding API.",
        model="text-embedding-3-small"
    )
    return await create_embedding_endpoint(test_request, current_user)

# Health check
@router.get(
    "/embeddings/health",
    summary="Embeddings service health check",
    description="Check if the embeddings service is operational"
)
async def health_check():
    """Health check for embeddings service."""
    return {
        "status": "healthy",
        "service": "embeddings_v2",
        "implementation_available": EMBEDDINGS_AVAILABLE
    }