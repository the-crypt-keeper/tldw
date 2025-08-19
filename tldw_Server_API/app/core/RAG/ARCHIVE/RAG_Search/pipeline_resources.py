"""
Resource management for functional RAG pipelines.

This module handles the lifecycle management of stateful resources like database
connections, embedding models, vector stores, and thread pools. It ensures proper
initialization, sharing, and cleanup of resources across pipeline executions.
"""

import asyncio
from typing import Optional, Any, Dict, TYPE_CHECKING
from dataclasses import dataclass, field
from contextlib import asynccontextmanager, contextmanager
from concurrent.futures import ThreadPoolExecutor
import threading
import weakref
from pathlib import Path
from loguru import logger
import psutil

from .pipeline_core import PipelineResources, PipelineError, PipelineErrorType
from .simplified.config import RAGConfig
from .simplified.simple_cache import SimpleRAGCache, get_rag_cache
from .simplified.db_connection_pool import get_connection_pool, ConnectionPoolManager
from ..DB.Client_Media_DB_v2 import MediaDatabase
from ..Utils.paths import get_user_data_dir

if TYPE_CHECKING:
    from ..app import TldwCli
    from .simplified import RAGService, EmbeddingsService
    from .simplified.vector_store import VectorStore


# ==============================================================================
# Resource Manager Implementation
# ==============================================================================

class ResourceManager:
    """
    Manages the lifecycle of all pipeline resources.
    
    This class ensures that expensive resources like embedding models and database
    connections are properly initialized, shared across pipelines, and cleaned up.
    """
    
    def __init__(self, app: 'TldwCli', config: Optional[RAGConfig] = None):
        """
        Initialize the resource manager.
        
        Args:
            app: The main application instance
            config: RAG configuration (uses defaults if None)
        """
        self.app = app
        self.config = config or RAGConfig.from_settings()
        self._lock = threading.RLock()
        self._resources: Optional[PipelineResources] = None
        self._resource_refs: Dict[str, weakref.ref] = {}
        self._is_initialized = False
        self._cleanup_callbacks = []
        
        # Thread pool for CPU-bound operations
        self._thread_pool: Optional[ThreadPoolExecutor] = None
        
        # Connection pools
        self._connection_pools: Dict[str, ConnectionPoolManager] = {}
        
        # Caches
        self._caches: Dict[str, SimpleRAGCache] = {}
        
        # RAG services (lazy initialized)
        self._rag_service: Optional['RAGService'] = None
        self._embeddings_service: Optional['EmbeddingsService'] = None
        self._vector_store: Optional['VectorStore'] = None
        
        logger.info("ResourceManager initialized", extra={
            "embedding_model": self.config.embedding_model,
            "vector_store_type": self.config.vector_store_type
        })
    
    def get_resources(self) -> PipelineResources:
        """
        Get or create the pipeline resources.
        
        This method is thread-safe and ensures resources are initialized only once.
        
        Returns:
            PipelineResources instance with all necessary resources
        """
        with self._lock:
            if not self._is_initialized:
                self._initialize_resources()
            return self._resources
    
    def _initialize_resources(self) -> None:
        """Initialize all resources."""
        logger.info("Initializing pipeline resources")
        
        try:
            # Initialize thread pool
            self._thread_pool = ThreadPoolExecutor(
                max_workers=self.config.pipeline.max_concurrent_pipelines,
                thread_name_prefix="pipeline-worker"
            )
            
            # Initialize connection pools
            self._initialize_connection_pools()
            
            # Initialize caches
            self._initialize_caches()
            
            # Create resources object
            self._resources = PipelineResources(
                app=self.app,
                media_db=self._get_media_db(),
                conversations_db=self._get_conversations_db(),
                notes_service=self._get_notes_service(),
                vector_store=None,  # Lazy initialized
                embeddings_service=None,  # Lazy initialized
                connection_pool=self._connection_pools.get('main'),
                thread_pool=self._thread_pool,
                cache=self._caches.get('main')
            )
            
            self._is_initialized = True
            logger.info("Pipeline resources initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize resources: {e}")
            raise PipelineError(
                error_type=PipelineErrorType.RESOURCE_ERROR,
                message=f"Resource initialization failed: {str(e)}",
                cause=e
            )
    
    def _initialize_connection_pools(self) -> None:
        """Initialize database connection pools."""
        # Main connection pool for FTS5 searches
        main_pool_size = self.config.search.fts5_connection_pool_size
        self._connection_pools['main'] = get_connection_pool(
            pool_size=main_pool_size,
            pool_name="pipeline-main"
        )
        
        # Additional pools can be added here
        logger.debug(f"Initialized connection pool with size {main_pool_size}")
    
    def _initialize_caches(self) -> None:
        """Initialize caches."""
        # Main cache for search results
        self._caches['main'] = get_rag_cache(
            ttl=self.config.search.cache_ttl,
            max_size=self.config.search.cache_size
        )
        
        # Specialized caches for different search types
        if self.config.search.semantic_cache_ttl:
            self._caches['semantic'] = SimpleRAGCache(
                ttl=self.config.search.semantic_cache_ttl,
                max_size=self.config.search.cache_size
            )
        
        if self.config.search.keyword_cache_ttl:
            self._caches['keyword'] = SimpleRAGCache(
                ttl=self.config.search.keyword_cache_ttl,
                max_size=self.config.search.cache_size
            )
    
    def _get_media_db(self) -> Optional[MediaDatabase]:
        """Get media database if available."""
        if hasattr(self.app, 'client_media_db'):
            return self.app.client_media_db
        return None
    
    def _get_conversations_db(self) -> Optional[Any]:
        """Get conversations database if available."""
        if hasattr(self.app, 'chachanotes_db'):
            return self.app.chachanotes_db
        return None
    
    def _get_notes_service(self) -> Optional[Any]:
        """Get notes service if available."""
        if hasattr(self.app, 'notes_service'):
            return self.app.notes_service
        return None
    
    async def get_rag_service(self) -> 'RAGService':
        """
        Get or create the RAG service (lazy initialization).
        
        This is async because RAG service initialization may involve async operations.
        """
        if self._rag_service is None:
            from .simplified import RAGService
            self._rag_service = RAGService(self.config)
            # Initialize the service
            await self._rag_service.initialize()
            
            # Update resources with initialized services
            self._resources = PipelineResources(
                **{**self._resources.__dict__,
                   'vector_store': self._rag_service.vector_store,
                   'embeddings_service': self._rag_service.embeddings}
            )
        
        return self._rag_service
    
    async def get_embeddings_service(self) -> 'EmbeddingsService':
        """Get embeddings service (may trigger RAG service initialization)."""
        rag_service = await self.get_rag_service()
        return rag_service.embeddings
    
    def get_cache(self, cache_type: str = 'main') -> SimpleRAGCache:
        """
        Get a specific cache by type.
        
        Args:
            cache_type: Type of cache ('main', 'semantic', 'keyword')
            
        Returns:
            Cache instance
        """
        return self._caches.get(cache_type, self._caches['main'])
    
    def register_cleanup(self, callback: callable) -> None:
        """Register a cleanup callback to be called on shutdown."""
        self._cleanup_callbacks.append(callback)
    
    def cleanup(self) -> None:
        """Clean up all resources."""
        logger.info("Cleaning up pipeline resources")
        
        with self._lock:
            # Execute cleanup callbacks
            for callback in self._cleanup_callbacks:
                try:
                    callback()
                except Exception as e:
                    logger.error(f"Cleanup callback failed: {e}")
            
            # Shutdown thread pool
            if self._thread_pool:
                self._thread_pool.shutdown(wait=True)
                self._thread_pool = None
            
            # Close connection pools
            for pool in self._connection_pools.values():
                try:
                    pool.close_all()
                except Exception as e:
                    logger.error(f"Failed to close connection pool: {e}")
            
            # Clear caches
            for cache in self._caches.values():
                cache.clear()
            
            # Cleanup RAG service
            if self._rag_service:
                try:
                    self._rag_service.cleanup()
                except Exception as e:
                    logger.error(f"Failed to cleanup RAG service: {e}")
            
            self._is_initialized = False
            self._resources = None
            
            logger.info("Pipeline resources cleaned up")
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit with cleanup."""
        self.cleanup()
    
    async def __aenter__(self):
        """Async context manager entry."""
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit with cleanup."""
        self.cleanup()


# ==============================================================================
# Resource Context Managers
# ==============================================================================

@contextmanager
def with_resource_manager(app: 'TldwCli', config: Optional[RAGConfig] = None):
    """
    Context manager for resource management.
    
    Usage:
        with with_resource_manager(app) as manager:
            resources = manager.get_resources()
            # Use resources
    """
    manager = ResourceManager(app, config)
    try:
        yield manager
    finally:
        manager.cleanup()


@asynccontextmanager
async def async_with_resource_manager(app: 'TldwCli', config: Optional[RAGConfig] = None):
    """
    Async context manager for resource management.
    
    Usage:
        async with async_with_resource_manager(app) as manager:
            resources = manager.get_resources()
            rag_service = await manager.get_rag_service()
            # Use resources
    """
    manager = ResourceManager(app, config)
    try:
        yield manager
    finally:
        manager.cleanup()


# ==============================================================================
# Resource Health Monitoring
# ==============================================================================

class ResourceHealthMonitor:
    """Monitors health of pipeline resources."""
    
    def __init__(self, manager: ResourceManager):
        self.manager = manager
        self._last_check = 0
        self._check_interval = 60  # seconds
    
    def check_health(self) -> Dict[str, bool]:
        """
        Check health of all resources.
        
        Returns:
            Dict mapping resource name to health status
        """
        import time
        
        # Throttle health checks
        now = time.time()
        if now - self._last_check < self._check_interval:
            return {}
        
        self._last_check = now
        health = {}
        
        # Check memory
        memory = psutil.virtual_memory()
        health['memory_ok'] = memory.available > 500 * 1024 * 1024  # 500MB
        
        # Check thread pool
        if self.manager._thread_pool:
            health['thread_pool_ok'] = not self.manager._thread_pool._shutdown
        
        # Check databases
        resources = self.manager.get_resources()
        health['media_db_ok'] = resources.has_media_db()
        health['conversations_db_ok'] = resources.has_conversations_db()
        health['notes_service_ok'] = resources.has_notes_service()
        
        # Check caches
        health['cache_ok'] = len(self.manager._caches) > 0
        
        return health
    
    def get_resource_stats(self) -> Dict[str, Any]:
        """Get detailed resource statistics."""
        stats = {}
        
        # Memory stats
        memory = psutil.virtual_memory()
        stats['memory'] = {
            'total_mb': memory.total / 1024 / 1024,
            'available_mb': memory.available / 1024 / 1024,
            'percent_used': memory.percent
        }
        
        # Thread pool stats
        if self.manager._thread_pool:
            stats['thread_pool'] = {
                'max_workers': self.manager._thread_pool._max_workers,
                'active_threads': len(self.manager._thread_pool._threads)
            }
        
        # Cache stats
        cache_stats = {}
        for name, cache in self.manager._caches.items():
            cache_info = cache.cache_info()
            cache_stats[name] = {
                'hits': cache_info['hits'],
                'misses': cache_info['misses'],
                'size': cache_info['current_size'],
                'hit_rate': cache_info['hits'] / max(1, cache_info['hits'] + cache_info['misses'])
            }
        stats['caches'] = cache_stats
        
        return stats


# ==============================================================================
# Resource Pool for Pipeline Executors
# ==============================================================================

class ResourcePool:
    """
    Pool of resource managers for concurrent pipeline execution.
    
    This allows multiple pipelines to execute concurrently while sharing
    underlying resources efficiently.
    """
    
    def __init__(self, app: 'TldwCli', pool_size: int = 3):
        self.app = app
        self.pool_size = pool_size
        self._managers = []
        self._lock = threading.Lock()
        self._semaphore = threading.Semaphore(pool_size)
        
        # Pre-initialize managers
        for _ in range(pool_size):
            manager = ResourceManager(app)
            self._managers.append(manager)
    
    @contextmanager
    def acquire(self) -> ResourceManager:
        """Acquire a resource manager from the pool."""
        self._semaphore.acquire()
        try:
            with self._lock:
                manager = self._managers.pop()
            yield manager
        finally:
            with self._lock:
                self._managers.append(manager)
            self._semaphore.release()
    
    def cleanup(self):
        """Clean up all managers in the pool."""
        with self._lock:
            for manager in self._managers:
                manager.cleanup()
            self._managers.clear()


# ==============================================================================
# Global Resource Manager Instance
# ==============================================================================

_global_resource_manager: Optional[ResourceManager] = None
_manager_lock = threading.Lock()


def get_global_resource_manager(app: 'TldwCli', config: Optional[RAGConfig] = None) -> ResourceManager:
    """
    Get or create the global resource manager.
    
    This ensures a single resource manager is used across the application.
    """
    global _global_resource_manager
    
    with _manager_lock:
        if _global_resource_manager is None:
            _global_resource_manager = ResourceManager(app, config)
        return _global_resource_manager


def cleanup_global_resources():
    """Clean up global resources."""
    global _global_resource_manager
    
    with _manager_lock:
        if _global_resource_manager is not None:
            _global_resource_manager.cleanup()
            _global_resource_manager = None