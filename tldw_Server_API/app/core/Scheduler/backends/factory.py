"""
Backend factory for automatic detection and instantiation.
"""

from typing import Optional
from loguru import logger

from ..base.queue_backend import QueueBackend
from ..base.exceptions import BackendError
from ..config import SchedulerConfig, get_config


def create_backend(config: Optional[SchedulerConfig] = None) -> QueueBackend:
    """
    Create appropriate backend based on configuration.
    
    Automatically detects backend type from database URL and
    instantiates the correct implementation.
    
    Args:
        config: Scheduler configuration (uses global if not provided)
        
    Returns:
        Configured backend instance
        
    Raises:
        BackendError: If backend cannot be created
    """
    if config is None:
        from ..config import get_config
        config = get_config()
    
    # Detect backend type from URL
    url = config.database_url.lower()
    
    if config.is_memory:
        # In-memory backend for testing
        logger.info("Creating in-memory backend")
        from .memory_backend import MemoryBackend
        return MemoryBackend(config)
    
    elif config.is_postgresql:
        # PostgreSQL backend for production
        logger.info("Creating PostgreSQL backend")
        
        # Check if asyncpg is available
        try:
            import asyncpg
        except ImportError:
            raise BackendError(
                "PostgreSQL backend requires asyncpg. "
                "Install with: pip install asyncpg"
            )
        
        from .postgresql_backend import PostgreSQLBackend
        return PostgreSQLBackend(config)
    
    elif config.is_sqlite:
        # SQLite backend for development
        logger.info("Creating SQLite backend")
        
        # Check if aiosqlite is available
        try:
            import aiosqlite
        except ImportError:
            raise BackendError(
                "SQLite backend requires aiosqlite. "
                "Install with: pip install aiosqlite"
            )
        
        from .sqlite_backend import SQLiteBackend
        return SQLiteBackend(config)
    
    else:
        # Unknown backend type
        raise BackendError(
            f"Unknown backend type from URL: {config.database_url}. "
            f"Supported: postgresql://, sqlite://, memory://"
        )


async def test_backend_connection(config: Optional[SchedulerConfig] = None) -> bool:
    """
    Test if backend can connect successfully.
    
    Args:
        config: Scheduler configuration
        
    Returns:
        True if connection successful, False otherwise
    """
    try:
        backend = create_backend(config)
        await backend.connect()
        
        # Try a simple operation
        from ..base import Task
        test_task = Task(handler="test", payload={"test": True})
        task_id = await backend.enqueue(test_task)
        
        # Verify task was created
        task = await backend.get_task(task_id)
        success = task is not None
        
        await backend.disconnect()
        return success
        
    except Exception as e:
        logger.error(f"Backend connection test failed: {e}")
        return False


def get_backend_info(config: Optional[SchedulerConfig] = None) -> dict:
    """
    Get information about the configured backend.
    
    Args:
        config: Scheduler configuration
        
    Returns:
        Dictionary with backend information
    """
    if config is None:
        from ..config import get_config
        config = get_config()
    
    info = {
        "type": "unknown",
        "url": config._safe_database_url(),
        "features": [],
        "limitations": []
    }
    
    if config.is_memory:
        info.update({
            "type": "memory",
            "features": [
                "Fast operations",
                "No persistence",
                "Good for testing"
            ],
            "limitations": [
                "Data lost on restart",
                "Single process only",
                "No distributed support"
            ]
        })
    
    elif config.is_postgresql:
        info.update({
            "type": "postgresql",
            "features": [
                "SKIP LOCKED for atomic dequeue",
                "NOTIFY/LISTEN for real-time updates",
                "Advisory locks for leader election",
                "Full ACID compliance",
                "Distributed support",
                "High concurrency"
            ],
            "limitations": [
                "Requires PostgreSQL server",
                "Higher resource usage"
            ]
        })
    
    elif config.is_sqlite:
        info.update({
            "type": "sqlite",
            "features": [
                "No server required",
                "Low resource usage",
                "Good for development",
                "File-based persistence"
            ],
            "limitations": [
                "Limited concurrency",
                "No SKIP LOCKED",
                "Single machine only",
                "Write lock contention"
            ]
        })
    
    return info


class BackendManager:
    """
    Manages backend lifecycle and provides connection pooling.
    """
    
    def __init__(self, config: Optional[SchedulerConfig] = None):
        """
        Initialize backend manager.
        
        Args:
            config: Scheduler configuration
        """
        self.config = config or get_config()
        self._backend: Optional[QueueBackend] = None
        self._connected = False
    
    async def __aenter__(self) -> QueueBackend:
        """
        Async context manager entry.
        """
        await self.connect()
        return self._backend
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """
        Async context manager exit.
        """
        await self.disconnect()
    
    async def connect(self) -> QueueBackend:
        """
        Connect to backend.
        
        Returns:
            Connected backend instance
        """
        if self._connected:
            return self._backend
        
        self._backend = create_backend(self.config)
        await self._backend.connect()
        self._connected = True
        
        return self._backend
    
    async def disconnect(self) -> None:
        """
        Disconnect from backend.
        """
        if self._connected and self._backend:
            await self._backend.disconnect()
            self._connected = False
            self._backend = None
    
    @property
    def backend(self) -> QueueBackend:
        """
        Get backend instance.
        
        Returns:
            Backend instance
            
        Raises:
            BackendError: If not connected
        """
        if not self._connected or not self._backend:
            raise BackendError("Backend not connected. Call connect() first.")
        return self._backend
    
    @property
    def is_connected(self) -> bool:
        """
        Check if backend is connected.
        
        Returns:
            True if connected
        """
        return self._connected