"""
Database connection pooling for improved performance.

This module provides connection pooling for SQLite and other databases
to reduce connection overhead and improve concurrent access.
"""

import sqlite3
import asyncio
from typing import Dict, Any, Optional, List
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timedelta
import threading
from queue import Queue, Empty, Full
import time

from loguru import logger


@dataclass
class PooledConnection:
    """Wrapper for a pooled database connection."""
    connection: sqlite3.Connection
    created_at: datetime
    last_used: datetime
    in_use: bool = False
    _id: int = field(default_factory=lambda: id(object()))  # Unique ID for hashing
    
    def __hash__(self):
        """Make PooledConnection hashable using its unique ID."""
        return hash(self._id)
    
    def __eq__(self, other):
        """Compare PooledConnections by their unique ID."""
        if not isinstance(other, PooledConnection):
            return False
        return self._id == other._id
    
    def is_expired(self, max_age_seconds: int = 3600) -> bool:
        """Check if connection is too old."""
        age = datetime.now() - self.created_at
        return age > timedelta(seconds=max_age_seconds)
    
    def is_idle(self, idle_timeout_seconds: int = 300) -> bool:
        """Check if connection has been idle too long."""
        idle_time = datetime.now() - self.last_used
        return idle_time > timedelta(seconds=idle_timeout_seconds)


class SQLiteConnectionPool:
    """
    Thread-safe connection pool for SQLite databases.
    
    Features:
    - Configurable pool size
    - Connection health checking
    - Automatic connection recycling
    - Thread-safe operation
    """
    
    def __init__(
        self,
        database_path: str,
        min_connections: int = 2,
        max_connections: int = 10,
        connection_timeout: float = 5.0,
        max_connection_age: int = 3600,
        idle_timeout: int = 300
    ):
        """
        Initialize connection pool.
        
        Args:
            database_path: Path to SQLite database
            min_connections: Minimum connections to maintain
            max_connections: Maximum connections allowed
            connection_timeout: Timeout for getting a connection
            max_connection_age: Maximum age of a connection in seconds
            idle_timeout: Timeout for idle connections
        """
        self.database_path = database_path
        self.min_connections = min_connections
        self.max_connections = max_connections
        self.connection_timeout = connection_timeout
        self.max_connection_age = max_connection_age
        self.idle_timeout = idle_timeout
        
        # Thread-safe queue for available connections
        self._available = Queue(maxsize=max_connections)
        self._in_use = set()
        self._lock = threading.Lock()
        self._created_connections = 0
        
        # Initialize minimum connections
        self._initialize_pool()
        
        logger.info(
            f"Initialized SQLite connection pool for {database_path} "
            f"with min={min_connections}, max={max_connections}"
        )
    
    def _initialize_pool(self):
        """Create initial connections for the pool."""
        for _ in range(self.min_connections):
            try:
                conn = self._create_connection()
                self._available.put(conn)
                self._created_connections += 1
            except Exception as e:
                logger.error(f"Failed to create initial connection: {e}")
    
    def _create_connection(self) -> PooledConnection:
        """Create a new database connection."""
        connection = sqlite3.connect(
            self.database_path,
            check_same_thread=False,  # Allow multi-threaded access
            timeout=30.0
        )
        
        # Enable optimizations
        connection.execute("PRAGMA journal_mode=WAL")  # Write-Ahead Logging
        connection.execute("PRAGMA cache_size=10000")  # Larger cache
        connection.execute("PRAGMA synchronous=NORMAL")  # Balanced durability
        
        # Enable FTS5 if available
        connection.execute("PRAGMA compile_options")
        
        return PooledConnection(
            connection=connection,
            created_at=datetime.now(),
            last_used=datetime.now()
        )
    
    @contextmanager
    def get_connection(self):
        """
        Get a connection from the pool.
        
        Yields:
            sqlite3.Connection: A database connection
            
        Raises:
            TimeoutError: If unable to get connection within timeout
        """
        pooled_conn = None
        start_time = time.time()
        
        try:
            # Try to get an available connection
            while time.time() - start_time < self.connection_timeout:
                try:
                    pooled_conn = self._available.get_nowait()
                    
                    # Check if connection is still valid
                    if pooled_conn.is_expired(self.max_connection_age):
                        pooled_conn.connection.close()
                        pooled_conn = self._create_connection()
                    elif pooled_conn.is_idle(self.idle_timeout):
                        # Ping to check if connection is alive
                        try:
                            pooled_conn.connection.execute("SELECT 1")
                        except sqlite3.Error:
                            pooled_conn.connection.close()
                            pooled_conn = self._create_connection()
                    
                    pooled_conn.in_use = True
                    pooled_conn.last_used = datetime.now()
                    
                    with self._lock:
                        self._in_use.add(pooled_conn)
                    
                    yield pooled_conn.connection
                    return
                    
                except Empty:
                    # No available connections, try to create a new one
                    with self._lock:
                        if self._created_connections < self.max_connections:
                            pooled_conn = self._create_connection()
                            self._created_connections += 1
                            pooled_conn.in_use = True
                            self._in_use.add(pooled_conn)
                            
                            yield pooled_conn.connection
                            return
                    
                    # Wait a bit before retrying
                    time.sleep(0.1)
            
            raise TimeoutError(f"Could not get connection within {self.connection_timeout} seconds")
            
        finally:
            # Return connection to pool
            if pooled_conn:
                pooled_conn.in_use = False
                pooled_conn.last_used = datetime.now()
                
                with self._lock:
                    self._in_use.discard(pooled_conn)
                
                try:
                    self._available.put_nowait(pooled_conn)
                except Full:
                    # Pool is full, close the connection
                    pooled_conn.connection.close()
                    with self._lock:
                        self._created_connections -= 1
    
    def close_all(self):
        """Close all connections in the pool."""
        # Close available connections
        while not self._available.empty():
            try:
                pooled_conn = self._available.get_nowait()
                pooled_conn.connection.close()
            except Empty:
                break
        
        # Note: Can't close in-use connections, they should be returned first
        logger.info(f"Closed connection pool for {self.database_path}")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get pool statistics."""
        with self._lock:
            return {
                "database": self.database_path,
                "created_connections": self._created_connections,
                "available_connections": self._available.qsize(),
                "in_use_connections": len(self._in_use),
                "min_connections": self.min_connections,
                "max_connections": self.max_connections
            }


class ConnectionPoolManager:
    """
    Manages multiple connection pools for different databases.
    
    This is useful when the RAG system needs to access multiple
    databases (media_db, notes_db, etc.).
    """
    
    def __init__(self, default_config: Optional[Dict[str, Any]] = None):
        """
        Initialize connection pool manager.
        
        Args:
            default_config: Default configuration for new pools
        """
        self.pools: Dict[str, SQLiteConnectionPool] = {}
        self.default_config = default_config or {
            "min_connections": 2,
            "max_connections": 10,
            "connection_timeout": 5.0,
            "max_connection_age": 3600,
            "idle_timeout": 300
        }
        self._lock = threading.Lock()
    
    def get_pool(
        self,
        database_path: str,
        config: Optional[Dict[str, Any]] = None
    ) -> SQLiteConnectionPool:
        """
        Get or create a connection pool for a database.
        
        Args:
            database_path: Path to the database
            config: Optional pool configuration
            
        Returns:
            Connection pool for the database
        """
        with self._lock:
            if database_path not in self.pools:
                pool_config = {**self.default_config, **(config or {})}
                self.pools[database_path] = SQLiteConnectionPool(
                    database_path,
                    **pool_config
                )
            
            return self.pools[database_path]
    
    @contextmanager
    def get_connection(self, database_path: str):
        """
        Get a connection from the appropriate pool.
        
        Args:
            database_path: Path to the database
            
        Yields:
            Database connection
        """
        pool = self.get_pool(database_path)
        with pool.get_connection() as conn:
            yield conn
    
    def close_all(self):
        """Close all connection pools."""
        with self._lock:
            for pool in self.pools.values():
                pool.close_all()
            self.pools.clear()
    
    def get_all_stats(self) -> Dict[str, Dict[str, Any]]:
        """Get statistics for all pools."""
        with self._lock:
            return {
                path: pool.get_stats()
                for path, pool in self.pools.items()
            }


# Global connection pool manager instance
_global_pool_manager = None
_manager_lock = threading.Lock()


def get_pool_manager() -> ConnectionPoolManager:
    """Get the global connection pool manager."""
    global _global_pool_manager
    
    if _global_pool_manager is None:
        with _manager_lock:
            if _global_pool_manager is None:
                _global_pool_manager = ConnectionPoolManager()
    
    return _global_pool_manager


def cleanup_pools():
    """Clean up all connection pools."""
    global _global_pool_manager
    
    if _global_pool_manager:
        _global_pool_manager.close_all()
        _global_pool_manager = None