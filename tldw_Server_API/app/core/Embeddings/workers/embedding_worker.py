# embedding_worker.py
# Worker for generating embeddings from chunks

import hashlib
import json
import time
from typing import Any, Dict, List, Optional, Union, Tuple
from collections import OrderedDict
from datetime import datetime, timedelta

import numpy as np
from loguru import logger

from ..Embeddings_Server.Embeddings_Create import (
    HFModelCfg,
    ONNXModelCfg,
    OpenAIModelCfg,
    LocalAPICfg,
    create_embeddings_batch
)
from ..queue_schemas import (
    EmbeddingData,
    EmbeddingMessage,
    JobStatus,
    StorageMessage,
)
from .base_worker import BaseWorker, WorkerConfig


class EmbeddingWorkerConfig(WorkerConfig):
    """Extended configuration for embedding workers"""
    default_model_provider: str = "huggingface"
    default_model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    max_batch_size: int = 32
    gpu_id: Optional[int] = None
    
    # Multi-model support configuration
    enable_model_selection: bool = True
    enable_caching: bool = True
    cache_ttl_seconds: int = 3600  # 1 hour default
    cache_max_size: int = 10000  # Maximum cached embeddings
    
    # Model selection thresholds
    long_text_threshold: int = 512  # Use different model for long texts
    multilingual_detection: bool = True
    
    # Available models by category
    models_by_category: Dict[str, Dict[str, str]] = {
        "general": {
            "provider": "huggingface",
            "model": "sentence-transformers/all-MiniLM-L6-v2"
        },
        "multilingual": {
            "provider": "huggingface", 
            "model": "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
        },
        "long_context": {
            "provider": "openai",
            "model": "text-embedding-3-large"
        },
        "high_quality": {
            "provider": "openai",
            "model": "text-embedding-3-large"
        }
    }


class EmbeddingCache:
    """LRU cache for embeddings with TTL support"""
    
    def __init__(self, max_size: int = 10000, ttl_seconds: int = 3600):
        self.max_size = max_size
        self.ttl_seconds = ttl_seconds
        self.cache: OrderedDict[str, Tuple[List[float], datetime]] = OrderedDict()
        self.hits = 0
        self.misses = 0
        
    def _get_cache_key(self, text: str, model: str) -> str:
        """Generate cache key from text and model"""
        content = f"{model}:{text}"
        return hashlib.sha256(content.encode()).hexdigest()
    
    def get(self, text: str, model: str) -> Optional[List[float]]:
        """Get embedding from cache if exists and not expired"""
        key = self._get_cache_key(text, model)
        
        if key in self.cache:
            embedding, timestamp = self.cache[key]
            
            # Check TTL
            if datetime.now() - timestamp < timedelta(seconds=self.ttl_seconds):
                # Move to end (most recently used)
                self.cache.move_to_end(key)
                self.hits += 1
                return embedding
            else:
                # Expired, remove it
                del self.cache[key]
        
        self.misses += 1
        return None
    
    def put(self, text: str, model: str, embedding: List[float]):
        """Store embedding in cache"""
        key = self._get_cache_key(text, model)
        
        # Remove oldest if at capacity
        if len(self.cache) >= self.max_size:
            self.cache.popitem(last=False)
        
        self.cache[key] = (embedding, datetime.now())
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        total = self.hits + self.misses
        hit_rate = self.hits / total if total > 0 else 0
        
        return {
            "size": len(self.cache),
            "max_size": self.max_size,
            "hits": self.hits,
            "misses": self.misses,
            "hit_rate": hit_rate,
            "ttl_seconds": self.ttl_seconds
        }
    
    def clear(self):
        """Clear all cache entries"""
        self.cache.clear()
        self.hits = 0
        self.misses = 0


class EmbeddingWorker(BaseWorker):
    """Worker that generates embeddings from text chunks"""
    
    def __init__(self, config: EmbeddingWorkerConfig):
        super().__init__(config)
        self.embedding_config = config
        self.storage_queue = config.queue_name.replace("embedding", "storage")
        
        # Initialize embedding cache
        self.cache = EmbeddingCache(
            max_size=config.cache_max_size,
            ttl_seconds=config.cache_ttl_seconds
        ) if config.enable_caching else None
        
        # Model configuration cache
        self.model_configs = {
            "huggingface": HFModelCfg(
                model_name_or_path=config.default_model_name,
                trust_remote_code=False
            ),
            "openai": OpenAIModelCfg(
                model_name_or_path="text-embedding-3-small"
            ),
            # Add more default configs as needed
        }
        
        # Performance tracking
        self.model_usage_stats = {}
        self.model_performance = {}
        
        # Set GPU if specified
        if config.gpu_id is not None:
            import os
            os.environ["CUDA_VISIBLE_DEVICES"] = str(config.gpu_id)
    
    def _parse_message(self, data: Dict[str, Any]) -> EmbeddingMessage:
        """Parse raw message data into EmbeddingMessage"""
        return EmbeddingMessage(**data)
    
    def _detect_language(self, text: str) -> str:
        """Detect language of text"""
        try:
            # Simple heuristic - check for non-ASCII characters
            # In production, use langdetect or polyglot
            non_ascii_count = sum(1 for c in text if ord(c) > 127)
            text_length = len(text)
            
            # If more than 10% non-ASCII, likely non-English
            if text_length > 0 and non_ascii_count / text_length > 0.1:
                return "multilingual"
            
            # Check for common non-English patterns
            multilingual_indicators = [
                'à', 'è', 'ì', 'ò', 'ù',  # Italian/French
                'ñ', 'á', 'é', 'í', 'ó', 'ú',  # Spanish
                'ä', 'ö', 'ü', 'ß',  # German
                'å', 'æ', 'ø',  # Nordic
                'ą', 'ć', 'ę', 'ł', 'ń', 'ś', 'ź', 'ż',  # Polish
                'а', 'б', 'в', 'г', 'д',  # Cyrillic
                '中', '文', '日', '本',  # CJK
            ]
            
            if any(char in text.lower() for char in multilingual_indicators):
                return "multilingual"
            
            return "english"
            
        except Exception as e:
            logger.warning(f"Language detection failed: {e}")
            return "english"  # Default to English
    
    def _select_model(self, text: str, metadata: Dict[str, Any]) -> Tuple[str, str]:
        """
        Select appropriate model based on text characteristics.
        
        Returns:
            Tuple of (provider, model_name)
        """
        if not self.embedding_config.enable_model_selection:
            return (
                self.embedding_config.default_model_provider,
                self.embedding_config.default_model_name
            )
        
        # Check for explicit model request in metadata
        if "preferred_model" in metadata:
            model_info = metadata["preferred_model"]
            if isinstance(model_info, dict):
                return (model_info.get("provider"), model_info.get("model"))
        
        # Determine model category based on text characteristics
        text_length = len(text)
        category = "general"
        
        # Check for long text
        if text_length > self.embedding_config.long_text_threshold:
            category = "long_context"
            logger.debug(f"Selected long_context model for text with {text_length} chars")
        
        # Check for multilingual content
        elif self.embedding_config.multilingual_detection:
            language = self._detect_language(text)
            if language == "multilingual":
                category = "multilingual"
                logger.debug("Selected multilingual model for non-English text")
        
        # Check for high quality request
        elif metadata.get("high_quality", False):
            category = "high_quality"
            logger.debug("Selected high quality model per request")
        
        # Get model for category
        model_info = self.embedding_config.models_by_category.get(
            category,
            self.embedding_config.models_by_category["general"]
        )
        
        provider = model_info.get("provider", self.embedding_config.default_model_provider)
        model = model_info.get("model", self.embedding_config.default_model_name)
        
        # Track model usage
        model_key = f"{provider}:{model}"
        self.model_usage_stats[model_key] = self.model_usage_stats.get(model_key, 0) + 1
        
        return (provider, model)
    
    async def process_message(self, message: EmbeddingMessage) -> Optional[StorageMessage]:
        """Process embedding message and generate embeddings"""
        logger.info(f"Processing embedding job {message.job_id} with {len(message.chunks)} chunks")
        
        start_time = time.time()
        cache_hits = 0
        cache_misses = 0
        
        try:
            # Update job status
            await self._update_job_status(message.job_id, JobStatus.EMBEDDING)
            
            # Process chunks in batches
            embedding_data_list = []
            batch_size = message.batch_size or self.embedding_config.max_batch_size
            
            # Track models used
            models_used = set()
            
            for i in range(0, len(message.chunks), batch_size):
                batch_chunks = message.chunks[i:i + batch_size]
                
                # Process each chunk individually for cache lookup and model selection
                for chunk in batch_chunks:
                    embedding = None
                    model_provider = None
                    model_name = None
                    
                    # Select appropriate model for this chunk
                    if self.embedding_config.enable_model_selection:
                        model_provider, model_name = self._select_model(
                            chunk.content,
                            chunk.metadata or {}
                        )
                    else:
                        # Use message-level or default configuration
                        if message.model_provider and message.model_config:
                            model_provider = message.model_provider
                            model_name = message.model_config.get("model_name_or_path", self.embedding_config.default_model_name)
                        else:
                            model_config = self._get_model_config(message)
                            model_provider = message.model_provider or self.embedding_config.default_model_provider
                            model_name = model_config.model_name_or_path
                    
                    models_used.add(f"{model_provider}:{model_name}")
                    
                    # Check cache first
                    if self.cache and self.embedding_config.enable_caching:
                        cached_embedding = self.cache.get(chunk.content, model_name)
                        if cached_embedding:
                            embedding = cached_embedding
                            cache_hits += 1
                            logger.debug(f"Cache hit for chunk {chunk.chunk_id}")
                        else:
                            cache_misses += 1
                    
                    # Generate embedding if not cached
                    if embedding is None:
                        # Get appropriate model config
                        if model_provider == "huggingface":
                            model_config = HFModelCfg(
                                model_name_or_path=model_name,
                                trust_remote_code=False
                            )
                        elif model_provider == "openai":
                            model_config = OpenAIModelCfg(
                                model_name_or_path=model_name
                            )
                        else:
                            model_config = self._get_model_config(message)
                        
                        # Generate embedding
                        embeddings = await self._generate_embeddings(
                            [chunk.content],
                            model_config,
                            model_provider
                        )
                        embedding = embeddings[0]
                        
                        # Cache the embedding
                        if self.cache and self.embedding_config.enable_caching:
                            embedding_list = embedding.tolist() if isinstance(embedding, np.ndarray) else embedding
                            self.cache.put(chunk.content, model_name, embedding_list)
                    
                    # Create embedding data object
                    embedding_data = EmbeddingData(
                        chunk_id=chunk.chunk_id,
                        embedding=embedding if isinstance(embedding, list) else embedding.tolist(),
                        model_used=model_name,
                        dimensions=len(embedding) if isinstance(embedding, list) else len(embedding.tolist()),
                        metadata={
                            **(chunk.metadata or {}),
                            "model_provider": model_provider,
                            "cached": embedding is not None and cache_hits > cache_misses
                        }
                    )
                    embedding_data_list.append(embedding_data)
                
                # Update progress
                progress = 25 + (50 * (i + len(batch_chunks)) / len(message.chunks))
                await self._update_job_progress(
                    message.job_id,
                    progress,
                    chunks_processed=i + len(batch_chunks)
                )
            
            processing_time_ms = int((time.time() - start_time) * 1000)
            
            # Log cache statistics
            if self.cache:
                cache_stats = self.cache.get_stats()
                logger.info(
                    f"Cache stats for job {message.job_id}: "
                    f"Hits: {cache_hits}, Misses: {cache_misses}, "
                    f"Total hit rate: {cache_stats['hit_rate']:.2%}"
                )
            
            # Create storage message
            storage_message = StorageMessage(
                job_id=message.job_id,
                user_id=message.user_id,
                media_id=message.media_id,
                priority=message.priority,
                user_tier=message.user_tier,
                created_at=message.created_at,
                embeddings=embedding_data_list,
                collection_name=f"user_{message.user_id}_media_{message.media_id}",
                total_chunks=len(message.chunks),
                processing_time_ms=processing_time_ms,
                metadata={
                    "models_used": list(models_used),
                    "cache_hits": cache_hits,
                    "cache_misses": cache_misses,
                    "model_selection_enabled": self.embedding_config.enable_model_selection
                }
            )
            
            logger.info(
                f"Generated {len(embedding_data_list)} embeddings for job {message.job_id} "
                f"in {processing_time_ms}ms using models: {', '.join(models_used)}"
            )
            return storage_message
            
        except Exception as e:
            logger.error(f"Error generating embeddings for job {message.job_id}: {e}")
            raise
    
    async def _send_to_next_stage(self, result: StorageMessage):
        """Send embeddings to storage queue"""
        await self.redis_client.xadd(
            self.storage_queue,
            result.dict()
        )
        logger.debug(f"Sent job {result.job_id} to storage queue")
    
    def _get_model_config(self, message: EmbeddingMessage) -> Union[HFModelCfg, ONNXModelCfg, OpenAIModelCfg, LocalAPICfg]:
        """Get or create model configuration"""
        if message.model_config:
            # Use provided config
            provider = message.model_provider
            if provider == "huggingface":
                return HFModelCfg(**message.model_config)
            elif provider == "onnx":
                return ONNXModelCfg(**message.model_config)
            elif provider == "openai":
                return OpenAIModelCfg(**message.model_config)
            elif provider == "local_api":
                return LocalAPICfg(**message.model_config)
        
        # Use default config based on user tier
        if message.user_tier == "enterprise":
            return self.model_configs.get("openai", self.model_configs["huggingface"])
        else:
            return self.model_configs["huggingface"]
    
    async def _generate_embeddings(
        self,
        texts: List[str],
        config: Union[HFModelCfg, ONNXModelCfg, OpenAIModelCfg, LocalAPICfg],
        provider: str
    ) -> List[np.ndarray]:
        """Generate embeddings for a batch of texts"""
        # Use the existing create_embeddings_batch function
        # This runs in a thread pool to avoid blocking the event loop
        import asyncio
        
        loop = asyncio.get_event_loop()
        embeddings = await loop.run_in_executor(
            None,
            create_embeddings_batch,
            texts,
            config.model_name_or_path,
            provider,
            config.api_url if hasattr(config, 'api_url') else None,
            config.api_key if hasattr(config, 'api_key') else None
        )
        
        return embeddings
    
    async def _update_job_progress(self, job_id: str, percentage: float, chunks_processed: int):
        """Update job progress information"""
        job_key = f"job:{job_id}"
        await self.redis_client.hset(
            job_key,
            mapping={
                "progress_percentage": percentage,
                "chunks_processed": chunks_processed
            }
        )
    
    async def _calculate_load(self) -> float:
        """Calculate current worker load based on GPU utilization"""
        try:
            import pynvml
            pynvml.nvmlInit()
            
            if self.embedding_config.gpu_id is not None:
                handle = pynvml.nvmlDeviceGetHandleByIndex(self.embedding_config.gpu_id)
                util = pynvml.nvmlDeviceGetUtilizationRates(handle)
                return util.gpu / 100.0
            
        except Exception:
            # Fallback to base implementation if GPU monitoring fails
            pass
        
        return await super()._calculate_load()
    
    def get_model_stats(self) -> Dict[str, Any]:
        """Get model usage and performance statistics"""
        stats = {
            "model_usage": self.model_usage_stats,
            "model_performance": self.model_performance,
            "cache_stats": self.cache.get_stats() if self.cache else None,
            "models_available": list(self.embedding_config.models_by_category.keys()),
            "model_selection_enabled": self.embedding_config.enable_model_selection,
            "caching_enabled": self.embedding_config.enable_caching
        }
        
        # Calculate most used model
        if self.model_usage_stats:
            most_used = max(self.model_usage_stats.items(), key=lambda x: x[1])
            stats["most_used_model"] = {
                "model": most_used[0],
                "count": most_used[1]
            }
        
        return stats
    
    async def clear_cache(self):
        """Clear the embedding cache"""
        if self.cache:
            self.cache.clear()
            logger.info("Embedding cache cleared")
            return {"status": "success", "message": "Cache cleared"}
        else:
            return {"status": "info", "message": "Caching is disabled"}
    
    async def warm_cache(self, texts: List[str], model: Optional[str] = None):
        """Pre-generate embeddings for frequently used texts"""
        if not self.cache:
            return {"status": "error", "message": "Caching is disabled"}
        
        model_to_use = model or self.embedding_config.default_model_name
        warmed_count = 0
        
        for text in texts:
            # Check if already cached
            if not self.cache.get(text, model_to_use):
                try:
                    # Generate embedding
                    config = self._get_model_config_for_name(model_to_use)
                    embeddings = await self._generate_embeddings(
                        [text],
                        config,
                        self.embedding_config.default_model_provider
                    )
                    
                    # Cache it
                    embedding_list = embeddings[0].tolist() if isinstance(embeddings[0], np.ndarray) else embeddings[0]
                    self.cache.put(text, model_to_use, embedding_list)
                    warmed_count += 1
                    
                except Exception as e:
                    logger.error(f"Failed to warm cache for text: {e}")
        
        return {
            "status": "success",
            "warmed_count": warmed_count,
            "total_requested": len(texts)
        }
    
    def _get_model_config_for_name(self, model_name: str):
        """Get model configuration for a specific model name"""
        # Check if it's in our known models
        for category, model_info in self.embedding_config.models_by_category.items():
            if model_info.get("model") == model_name:
                provider = model_info.get("provider")
                if provider == "huggingface":
                    return HFModelCfg(model_name_or_path=model_name, trust_remote_code=False)
                elif provider == "openai":
                    return OpenAIModelCfg(model_name_or_path=model_name)
        
        # Default to HuggingFace
        return HFModelCfg(model_name_or_path=model_name, trust_remote_code=False)