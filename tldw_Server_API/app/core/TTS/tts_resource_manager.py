# tts_resource_manager.py
# Description: Resource management for TTS operations including connection pooling, memory management, and cleanup
#
# Imports
import asyncio
import gc
import psutil
import time
import weakref
from contextlib import asynccontextmanager
from typing import Dict, Any, List, Optional, AsyncGenerator, Set, Callable, TYPE_CHECKING
import threading
from dataclasses import dataclass, field
from enum import Enum
#
# Third-party Imports
import httpx
from loguru import logger
#
# Local Imports
from .tts_exceptions import (
    TTSResourceError,
    TTSInsufficientMemoryError,
    TTSInsufficientStorageError,
    TTSModelLoadError,
    TTSNetworkError
)
#
# Conditional imports for type checking
if TYPE_CHECKING:
    from .adapters.base import TTSAdapter
#
#######################################################################################################################
#
# Resource Management System

class ResourceType(Enum):
    """Types of resources managed by the system"""
    HTTP_CONNECTION = "http_connection"
    MODEL_INSTANCE = "model_instance" 
    STREAMING_SESSION = "streaming_session"
    TEMP_FILE = "temp_file"
    MEMORY_BUFFER = "memory_buffer"


@dataclass
class ResourceMetrics:
    """Metrics for resource usage tracking"""
    created_at: float
    last_used: float
    use_count: int = 0
    memory_usage: int = 0  # bytes
    is_active: bool = True
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def update_usage(self):
        """Update usage statistics"""
        self.last_used = time.time()
        self.use_count += 1


class StreamingSession:
    """Manages a streaming TTS session with proper cleanup"""
    
    def __init__(
        self, 
        session_id: str, 
        provider: str, 
        cleanup_callback: Optional[Callable] = None
    ):
        """
        Initialize streaming session.
        
        Args:
            session_id: Unique session identifier
            provider: TTS provider name
            cleanup_callback: Optional cleanup function
        """
        self.session_id = session_id
        self.provider = provider
        self.cleanup_callback = cleanup_callback
        self.created_at = time.time()
        self.last_activity = time.time()
        self.is_active = True
        self.chunks_sent = 0
        self.bytes_sent = 0
        self.error_count = 0
        self._cleanup_tasks: Set[asyncio.Task] = set()
        
    async def track_activity(self, chunk_size: int = 0):
        """Track session activity"""
        self.last_activity = time.time()
        if chunk_size > 0:
            self.chunks_sent += 1
            self.bytes_sent += chunk_size
    
    async def add_cleanup_task(self, coro):
        """Add cleanup task to be executed when session closes"""
        task = asyncio.create_task(coro)
        self._cleanup_tasks.add(task)
        task.add_done_callback(self._cleanup_tasks.discard)
    
    async def close(self):
        """Close the streaming session and cleanup resources"""
        if not self.is_active:
            return
        
        self.is_active = False
        
        try:
            # Execute cleanup callback
            if self.cleanup_callback:
                if asyncio.iscoroutinefunction(self.cleanup_callback):
                    await self.cleanup_callback()
                else:
                    self.cleanup_callback()
            
            # Wait for cleanup tasks
            if self._cleanup_tasks:
                await asyncio.gather(*self._cleanup_tasks, return_exceptions=True)
                
            logger.debug(
                f"Streaming session {self.session_id} closed: "
                f"chunks={self.chunks_sent}, bytes={self.bytes_sent}, "
                f"duration={time.time() - self.created_at:.2f}s"
            )
            
        except Exception as e:
            logger.error(f"Error closing streaming session {self.session_id}: {e}")
    
    def is_expired(self, timeout: float = 300) -> bool:
        """Check if session has expired"""
        return time.time() - self.last_activity > timeout


class HTTPConnectionPool:
    """HTTP connection pool for API-based TTS providers"""
    
    def __init__(
        self,
        max_connections: int = 10,
        max_keepalive_connections: int = 5,
        keepalive_expiry: float = 30.0,
        timeout: float = 60.0
    ):
        """
        Initialize HTTP connection pool.
        
        Args:
            max_connections: Maximum total connections
            max_keepalive_connections: Maximum keep-alive connections
            keepalive_expiry: Keep-alive timeout in seconds
            timeout: Request timeout in seconds
        """
        self.max_connections = max_connections
        self.max_keepalive_connections = max_keepalive_connections
        self.keepalive_expiry = keepalive_expiry
        self.timeout = timeout
        
        # Connection pools per provider
        self._pools: Dict[str, httpx.AsyncClient] = {}
        self._pool_metrics: Dict[str, ResourceMetrics] = {}
        self._lock = asyncio.Lock()
        
    async def get_client(self, provider: str, base_url: Optional[str] = None) -> httpx.AsyncClient:
        """
        Get or create HTTP client for provider.
        
        Args:
            provider: Provider name
            base_url: Optional base URL for the client
            
        Returns:
            HTTP client instance
        """
        async with self._lock:
            if provider not in self._pools:
                client = httpx.AsyncClient(
                    limits=httpx.Limits(
                        max_connections=self.max_connections,
                        max_keepalive_connections=self.max_keepalive_connections,
                        keepalive_expiry=self.keepalive_expiry
                    ),
                    timeout=httpx.Timeout(self.timeout),
                    base_url=base_url
                )
                
                self._pools[provider] = client
                self._pool_metrics[provider] = ResourceMetrics(
                    created_at=time.time(),
                    last_used=time.time(),
                    metadata={"provider": provider, "base_url": base_url}
                )
                
                logger.debug(f"Created HTTP connection pool for {provider}")
            
            # Update metrics
            metrics = self._pool_metrics[provider]
            metrics.update_usage()
            
            return self._pools[provider]
    
    async def close_pool(self, provider: str):
        """Close connection pool for specific provider"""
        async with self._lock:
            if provider in self._pools:
                await self._pools[provider].aclose()
                del self._pools[provider]
                del self._pool_metrics[provider]
                logger.debug(f"Closed HTTP connection pool for {provider}")
    
    async def close_all(self):
        """Close all connection pools"""
        async with self._lock:
            tasks = []
            for provider, client in self._pools.items():
                tasks.append(client.aclose())
            
            if tasks:
                await asyncio.gather(*tasks, return_exceptions=True)
            
            self._pools.clear()
            self._pool_metrics.clear()
            logger.info("Closed all HTTP connection pools")
    
    def get_stats(self) -> Dict[str, Dict[str, Any]]:
        """Get connection pool statistics"""
        stats = {}
        for provider, metrics in self._pool_metrics.items():
            stats[provider] = {
                "created_at": metrics.created_at,
                "last_used": metrics.last_used,
                "use_count": metrics.use_count,
                "is_active": metrics.is_active,
                "metadata": metrics.metadata
            }
        return stats


class MemoryMonitor:
    """Memory monitoring and management for local TTS models"""
    
    def __init__(
        self,
        memory_threshold: float = 0.85,  # 85% of total memory
        check_interval: float = 30.0,    # Check every 30 seconds
        cleanup_threshold: float = 0.90   # Force cleanup at 90%
    ):
        """
        Initialize memory monitor.
        
        Args:
            memory_threshold: Memory usage threshold (0.0-1.0)
            check_interval: Monitoring check interval in seconds
            cleanup_threshold: Force cleanup threshold (0.0-1.0)
        """
        self.memory_threshold = memory_threshold
        self.check_interval = check_interval
        self.cleanup_threshold = cleanup_threshold
        
        self._monitoring = False
        self._monitor_task: Optional[asyncio.Task] = None
        self._model_references: Set[weakref.ReferenceType] = set()
        self._cleanup_callbacks: List[Callable] = []
        
        # Get system memory info
        self.total_memory = psutil.virtual_memory().total
    
    def register_model(self, model_instance: Any, cleanup_callback: Optional[Callable] = None):
        """
        Register a model instance for memory monitoring.
        
        Args:
            model_instance: Model instance to monitor
            cleanup_callback: Optional cleanup function for the model
        """
        # Use weak reference to avoid circular references
        weak_ref = weakref.ref(model_instance)
        self._model_references.add(weak_ref)
        
        if cleanup_callback:
            self._cleanup_callbacks.append(cleanup_callback)
        
        logger.debug(f"Registered model for memory monitoring: {type(model_instance).__name__}")
    
    def get_memory_usage(self) -> Dict[str, Any]:
        """Get current memory usage statistics"""
        memory = psutil.virtual_memory()
        return {
            "total": memory.total,
            "available": memory.available,
            "used": memory.used,
            "percent": memory.percent,
            "free": memory.free,
            "threshold": self.memory_threshold * 100,
            "cleanup_threshold": self.cleanup_threshold * 100
        }
    
    def is_memory_critical(self) -> bool:
        """Check if memory usage is critical"""
        usage = psutil.virtual_memory().percent / 100.0
        return usage > self.cleanup_threshold
    
    def is_memory_high(self) -> bool:
        """Check if memory usage is high"""
        usage = psutil.virtual_memory().percent / 100.0
        return usage > self.memory_threshold
    
    async def force_cleanup(self):
        """Force memory cleanup"""
        logger.warning("Forcing memory cleanup due to high usage")
        
        # Run cleanup callbacks
        for callback in self._cleanup_callbacks:
            try:
                if asyncio.iscoroutinefunction(callback):
                    await callback()
                else:
                    callback()
            except Exception as e:
                logger.error(f"Error in cleanup callback: {e}")
        
        # Clean up dead references
        self._model_references = {ref for ref in self._model_references if ref() is not None}
        
        # Force garbage collection
        gc.collect()
        
        # Check if cleanup was effective
        if self.is_memory_critical():
            raise TTSInsufficientMemoryError(
                "Memory usage remains critical after cleanup",
                details=self.get_memory_usage()
            )
    
    async def start_monitoring(self):
        """Start memory monitoring"""
        if self._monitoring:
            return
        
        self._monitoring = True
        self._monitor_task = asyncio.create_task(self._monitor_loop())
        logger.info("Memory monitoring started")
    
    async def stop_monitoring(self):
        """Stop memory monitoring"""
        self._monitoring = False
        if self._monitor_task:
            self._monitor_task.cancel()
            try:
                await self._monitor_task
            except asyncio.CancelledError:
                pass
        logger.info("Memory monitoring stopped")
    
    async def _monitor_loop(self):
        """Memory monitoring loop"""
        while self._monitoring:
            try:
                if self.is_memory_critical():
                    await self.force_cleanup()
                elif self.is_memory_high():
                    logger.warning(f"High memory usage: {psutil.virtual_memory().percent:.1f}%")
                
                await asyncio.sleep(self.check_interval)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in memory monitoring: {e}")
                await asyncio.sleep(self.check_interval)


class StreamingSessionManager:
    """Manages streaming audio sessions"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.sessions: Dict[str, StreamingSession] = {}
        self.max_sessions = self.config.get("max_streaming_sessions", 10)
        self._lock = threading.Lock()
    
    def create_session(self, session_id: str, **kwargs) -> StreamingSession:
        """Create a new streaming session"""
        with self._lock:
            if len(self.sessions) >= self.max_sessions:
                # Clean up old sessions
                self._cleanup_old_sessions()
                
            if len(self.sessions) >= self.max_sessions:
                raise TTSResourceError("Maximum streaming sessions reached")
            
            session = StreamingSession(session_id=session_id, **kwargs)
            self.sessions[session_id] = session
            return session
    
    def get_session(self, session_id: str) -> Optional[StreamingSession]:
        """Get an existing session"""
        return self.sessions.get(session_id)
    
    def end_session(self, session_id: str) -> Optional[Dict[str, Any]]:
        """End a streaming session and return stats"""
        with self._lock:
            session = self.sessions.pop(session_id, None)
            if session:
                session.end()
                return session.get_stats()
            return None
    
    def cleanup_expired_sessions(self):
        """Clean up expired sessions"""
        with self._lock:
            self._cleanup_old_sessions()
    
    def _cleanup_old_sessions(self):
        """Internal method to clean up old sessions"""
        current_time = time.time()
        expired = []
        
        for sid, session in self.sessions.items():
            if current_time - session.start_time > 3600:  # 1 hour timeout
                expired.append(sid)
        
        for sid in expired:
            session = self.sessions.pop(sid)
            session.end()
    
    def get_active_sessions(self) -> int:
        """Get number of active sessions"""
        return len(self.sessions)


class TTSResourceManager:
    """Central resource manager for TTS operations"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize TTS resource manager.
        
        Args:
            config: Resource management configuration
        """
        self.config = config or {}
        
        # Initialize components
        self.connection_pool = HTTPConnectionPool(
            max_connections=self.config.get("max_http_connections", 10),
            max_keepalive_connections=self.config.get("max_keepalive_connections", 5),
            keepalive_expiry=self.config.get("keepalive_expiry", 30.0),
            timeout=self.config.get("http_timeout", 60.0)
        )
        
        self.memory_monitor = MemoryMonitor(
            memory_threshold=self.config.get("memory_threshold", 0.85),
            check_interval=self.config.get("memory_check_interval", 30.0),
            cleanup_threshold=self.config.get("memory_cleanup_threshold", 0.90)
        )
        
        # Streaming session management
        self._streaming_sessions: Dict[str, StreamingSession] = {}
        self._session_cleanup_task: Optional[asyncio.Task] = None
        self._session_timeout = self.config.get("streaming_session_timeout", 300)  # 5 minutes
        
        # Model instance tracking
        self._model_instances: Dict[str, weakref.ReferenceType] = {}
        
        # Resource cleanup
        self._cleanup_handlers: Dict[ResourceType, List[Callable]] = {}
        
        logger.info("TTS Resource Manager initialized")
    
    async def initialize(self):
        """Initialize resource management"""
        # Start memory monitoring
        await self.memory_monitor.start_monitoring()
        
        # Start session cleanup task
        self._session_cleanup_task = asyncio.create_task(self._cleanup_expired_sessions())
        
        logger.info("TTS Resource Manager started")
    
    async def shutdown(self):
        """Shutdown resource manager and cleanup all resources"""
        logger.info("Shutting down TTS Resource Manager")
        
        # Stop monitoring
        await self.memory_monitor.stop_monitoring()
        
        # Stop session cleanup
        if self._session_cleanup_task:
            self._session_cleanup_task.cancel()
            try:
                await self._session_cleanup_task
            except asyncio.CancelledError:
                pass
        
        # Close all streaming sessions
        sessions = list(self._streaming_sessions.values())
        for session in sessions:
            await session.close()
        self._streaming_sessions.clear()
        
        # Close connection pools
        await self.connection_pool.close_all()
        
        # Run cleanup handlers
        for resource_type, handlers in self._cleanup_handlers.items():
            for handler in handlers:
                try:
                    if asyncio.iscoroutinefunction(handler):
                        await handler()
                    else:
                        handler()
                except Exception as e:
                    logger.error(f"Error in {resource_type} cleanup handler: {e}")
        
        logger.info("TTS Resource Manager shutdown complete")
    
    @asynccontextmanager
    async def streaming_session(self, session_id: str, provider: str, cleanup_callback: Optional[Callable] = None):
        """
        Context manager for streaming sessions.
        
        Args:
            session_id: Unique session identifier
            provider: TTS provider name
            cleanup_callback: Optional cleanup callback
        """
        session = StreamingSession(session_id, provider, cleanup_callback)
        self._streaming_sessions[session_id] = session
        
        try:
            logger.debug(f"Started streaming session {session_id} for {provider}")
            yield session
        finally:
            await session.close()
            self._streaming_sessions.pop(session_id, None)
            logger.debug(f"Closed streaming session {session_id}")
    
    async def get_http_client(self, provider: str, base_url: Optional[str] = None) -> httpx.AsyncClient:
        """Get HTTP client for provider"""
        return await self.connection_pool.get_client(provider, base_url)
    
    def register_model(self, provider: str, model_instance: Any, cleanup_callback: Optional[Callable] = None):
        """Register model instance for resource management"""
        self._model_instances[provider] = weakref.ref(model_instance)
        self.memory_monitor.register_model(model_instance, cleanup_callback)
    
    def register_cleanup_handler(self, resource_type: ResourceType, handler: Callable):
        """Register cleanup handler for resource type"""
        if resource_type not in self._cleanup_handlers:
            self._cleanup_handlers[resource_type] = []
        self._cleanup_handlers[resource_type].append(handler)
    
    async def _cleanup_expired_sessions(self):
        """Cleanup expired streaming sessions"""
        while True:
            try:
                current_time = time.time()
                expired_sessions = []
                
                for session_id, session in self._streaming_sessions.items():
                    if session.is_expired(self._session_timeout):
                        expired_sessions.append(session_id)
                
                for session_id in expired_sessions:
                    session = self._streaming_sessions.pop(session_id, None)
                    if session:
                        await session.close()
                        logger.info(f"Cleaned up expired streaming session {session_id}")
                
                # Sleep for cleanup interval
                await asyncio.sleep(60)  # Check every minute
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in session cleanup: {e}")
                await asyncio.sleep(60)
    
    def get_resource_stats(self) -> Dict[str, Any]:
        """Get resource usage statistics"""
        return {
            "http_connections": self.connection_pool.get_stats(),
            "memory": self.memory_monitor.get_memory_usage(),
            "streaming_sessions": {
                "active": len(self._streaming_sessions),
                "sessions": [
                    {
                        "id": session.session_id,
                        "provider": session.provider,
                        "duration": time.time() - session.created_at,
                        "chunks_sent": session.chunks_sent,
                        "bytes_sent": session.bytes_sent
                    }
                    for session in self._streaming_sessions.values()
                ]
            },
            "model_instances": {
                provider: ref() is not None 
                for provider, ref in self._model_instances.items()
            }
        }


# Global resource manager instance
_resource_manager: Optional[TTSResourceManager] = None
_manager_lock = asyncio.Lock()


async def get_resource_manager(config: Optional[Dict[str, Any]] = None) -> TTSResourceManager:
    """
    Get or create the global TTS resource manager.
    
    Args:
        config: Configuration for resource management
        
    Returns:
        TTSResourceManager instance
    """
    global _resource_manager
    
    if _resource_manager is None:
        async with _manager_lock:
            if _resource_manager is None:
                _resource_manager = TTSResourceManager(config)
                await _resource_manager.initialize()
                logger.info("Global TTS Resource Manager created")
    
    return _resource_manager


async def close_resource_manager():
    """Close the global resource manager"""
    global _resource_manager
    
    if _resource_manager:
        await _resource_manager.shutdown()
        _resource_manager = None
        logger.info("Global TTS Resource Manager closed")


# Alias for compatibility
reset_resource_manager = close_resource_manager


# Context manager for resource management
@asynccontextmanager
async def managed_resources(config: Optional[Dict[str, Any]] = None):
    """Context manager for TTS resource management"""
    manager = await get_resource_manager(config)
    try:
        yield manager
    finally:
        # Don't close the global manager, just ensure it's cleaned up properly
        pass

#
# End of tts_resource_manager.py
#######################################################################################################################