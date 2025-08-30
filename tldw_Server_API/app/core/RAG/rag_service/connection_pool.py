"""
Database connection pooling for RAG service.

This module provides thread-safe connection pooling for SQLite databases
to improve performance and reduce connection overhead.
"""

import sqlite3
import threading
import queue
import time
from typing import Dict, Optional, Any, Callable
from contextlib import contextmanager
from pathlib import Path
import logging

from loguru import logger


class ConnectionPool:
    """
    Thread-safe connection pool for SQLite databases.
    
    Features:
    - Configurable pool size
    - Connection health checks
    - Automatic reconnection
    - Connection lifecycle management
    - Performance metrics
    """
    
    def __init__(
        self,
        db_path: str,
        min_connections: int = 2,
        max_connections: int = 10,
        connection_timeout: float = 5.0,
        max_idle_time: float = 300.0,  # 5 minutes
        enable_wal: bool = True
    ):
        """
        Initialize connection pool.
        
        Args:
            db_path: Path to SQLite database
            min_connections: Minimum number of connections to maintain
            max_connections: Maximum number of connections allowed
            connection_timeout: Timeout for acquiring a connection
            max_idle_time: Maximum idle time before closing a connection
            enable_wal: Enable WAL mode for better concurrency
        """
        self.db_path = db_path
        self.min_connections = min_connections
        self.max_connections = max_connections
        self.connection_timeout = connection_timeout
        self.max_idle_time = max_idle_time
        self.enable_wal = enable_wal
        
        # Pool state
        self._pool = queue.Queue(maxsize=max_connections)
        self._all_connections: Dict[int, Dict[str, Any]] = {}
        self._lock = threading.RLock()
        self._closed = False
        
        # Statistics
        self._stats = {
            "connections_created": 0,
            "connections_reused": 0,
            "connections_closed": 0,
            "wait_time_total": 0.0,
            "active_connections": 0,
            "pool_exhausted_count": 0
        }
        
        # Initialize minimum connections
        self._initialize_pool()
    
    def _initialize_pool(self):
        """Initialize the connection pool with minimum connections."""
        for _ in range(self.min_connections):
            conn = self._create_connection()
            if conn:
                self._pool.put(conn)
    
    def _create_connection(self) -> Optional[sqlite3.Connection]:
        """
        Create a new database connection.
        
        Returns:
            New SQLite connection or None if failed
        """
        try:
            conn = sqlite3.connect(
                self.db_path,
                check_same_thread=False,
                isolation_level=None  # Autocommit mode
            )
            
            # Configure connection
            conn.row_factory = sqlite3.Row
            
            if self.enable_wal:
                conn.execute("PRAGMA journal_mode = WAL")
                conn.execute("PRAGMA synchronous = NORMAL")
            
            # Enable foreign keys
            conn.execute("PRAGMA foreign_keys = ON")
            
            # Track connection
            conn_id = id(conn)
            with self._lock:
                self._all_connections[conn_id] = {
                    "connection": conn,
                    "created_at": time.time(),
                    "last_used": time.time(),
                    "in_use": False
                }
                self._stats["connections_created"] += 1
            
            logger.debug(f"Created new connection {conn_id} for {self.db_path}")
            return conn
            
        except Exception as e:
            logger.error(f"Failed to create connection to {self.db_path}: {e}")
            return None
    
    def _is_connection_valid(self, conn: sqlite3.Connection) -> bool:
        """
        Check if a connection is still valid.
        
        Args:
            conn: Connection to check
            
        Returns:
            True if connection is valid
        """
        try:
            conn.execute("SELECT 1")
            return True
        except:
            return False
    
    @contextmanager
    def get_connection(self):
        """
        Get a connection from the pool.
        
        Yields:
            SQLite connection
            
        Raises:
            TimeoutError: If unable to acquire connection within timeout
        """
        if self._closed:
            raise RuntimeError("Connection pool is closed")
        
        conn = None
        wait_start = time.time()
        
        try:
            # Try to get existing connection
            try:
                conn = self._pool.get(timeout=self.connection_timeout)
                
                # Validate connection
                if not self._is_connection_valid(conn):
                    logger.debug("Invalid connection found, creating new one")
                    self._close_connection(conn)
                    conn = self._create_connection()
                    if not conn:
                        raise RuntimeError("Failed to create replacement connection")
                
                with self._lock:
                    self._stats["connections_reused"] += 1
                    
            except queue.Empty:
                # Pool exhausted, try to create new connection
                with self._lock:
                    active_count = len([c for c in self._all_connections.values() if c["in_use"]])
                    
                    if active_count < self.max_connections:
                        conn = self._create_connection()
                        if not conn:
                            raise RuntimeError("Failed to create new connection")
                    else:
                        self._stats["pool_exhausted_count"] += 1
                        raise TimeoutError(
                            f"Connection pool exhausted (max={self.max_connections})"
                        )
            
            # Update connection state
            conn_id = id(conn)
            with self._lock:
                if conn_id in self._all_connections:
                    self._all_connections[conn_id]["in_use"] = True
                    self._all_connections[conn_id]["last_used"] = time.time()
                self._stats["active_connections"] += 1
                self._stats["wait_time_total"] += time.time() - wait_start
            
            yield conn
            
        finally:
            # Return connection to pool
            if conn:
                conn_id = id(conn)
                with self._lock:
                    if conn_id in self._all_connections:
                        self._all_connections[conn_id]["in_use"] = False
                    self._stats["active_connections"] -= 1
                
                # Check if connection should be recycled
                if conn_id in self._all_connections:
                    conn_info = self._all_connections[conn_id]
                    idle_time = time.time() - conn_info["last_used"]
                    
                    if idle_time > self.max_idle_time:
                        logger.debug(f"Closing idle connection {conn_id}")
                        self._close_connection(conn)
                    else:
                        self._pool.put(conn)
    
    def _close_connection(self, conn: sqlite3.Connection):
        """
        Close a connection and remove from tracking.
        
        Args:
            conn: Connection to close
        """
        try:
            conn_id = id(conn)
            conn.close()
            
            with self._lock:
                if conn_id in self._all_connections:
                    del self._all_connections[conn_id]
                self._stats["connections_closed"] += 1
                
        except Exception as e:
            logger.error(f"Error closing connection: {e}")
    
    def close_idle_connections(self):
        """Close connections that have been idle too long."""
        current_time = time.time()
        connections_to_close = []
        
        with self._lock:
            for conn_id, info in self._all_connections.items():
                if not info["in_use"]:
                    idle_time = current_time - info["last_used"]
                    if idle_time > self.max_idle_time:
                        connections_to_close.append(info["connection"])
        
        for conn in connections_to_close:
            try:
                # Try to remove from pool
                try:
                    self._pool.get_nowait()
                except queue.Empty:
                    pass
                
                self._close_connection(conn)
                logger.debug(f"Closed idle connection")
                
            except Exception as e:
                logger.error(f"Error closing idle connection: {e}")
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get pool statistics.
        
        Returns:
            Dictionary of statistics
        """
        with self._lock:
            stats = self._stats.copy()
            stats["total_connections"] = len(self._all_connections)
            stats["idle_connections"] = self._pool.qsize()
            
            if stats["connections_reused"] > 0:
                stats["avg_wait_time"] = (
                    stats["wait_time_total"] / 
                    (stats["connections_created"] + stats["connections_reused"])
                )
            else:
                stats["avg_wait_time"] = 0
            
            return stats
    
    def close(self):
        """Close all connections and shutdown the pool."""
        self._closed = True
        
        # Close all connections
        while not self._pool.empty():
            try:
                conn = self._pool.get_nowait()
                self._close_connection(conn)
            except queue.Empty:
                break
        
        # Close any remaining connections
        with self._lock:
            for info in list(self._all_connections.values()):
                try:
                    info["connection"].close()
                except:
                    pass
            
            self._all_connections.clear()
        
        logger.info(f"Connection pool closed. Final stats: {self._stats}")


class MultiDatabasePool:
    """
    Manages connection pools for multiple databases.
    
    This is useful when working with Media_DB, ChaChaNotes_DB, etc.
    """
    
    def __init__(self, default_config: Optional[Dict[str, Any]] = None):
        """
        Initialize multi-database pool manager.
        
        Args:
            default_config: Default configuration for new pools
        """
        self._pools: Dict[str, ConnectionPool] = {}
        self._lock = threading.RLock()
        self._default_config = default_config or {
            "min_connections": 2,
            "max_connections": 10,
            "connection_timeout": 5.0,
            "max_idle_time": 300.0,
            "enable_wal": True
        }
    
    def get_pool(self, db_path: str, **config) -> ConnectionPool:
        """
        Get or create a connection pool for a database.
        
        Args:
            db_path: Path to database
            **config: Optional configuration overrides
            
        Returns:
            ConnectionPool for the database
        """
        db_path = str(Path(db_path).resolve())
        
        with self._lock:
            if db_path not in self._pools:
                # Merge configurations
                pool_config = self._default_config.copy()
                pool_config.update(config)
                
                # Create new pool
                self._pools[db_path] = ConnectionPool(
                    db_path=db_path,
                    **pool_config
                )
                logger.info(f"Created connection pool for {db_path}")
            
            return self._pools[db_path]
    
    @contextmanager
    def get_connection(self, db_path: str, **config):
        """
        Get a connection from the appropriate pool.
        
        Args:
            db_path: Path to database
            **config: Optional configuration for new pools
            
        Yields:
            Database connection
        """
        pool = self.get_pool(db_path, **config)
        with pool.get_connection() as conn:
            yield conn
    
    def close_idle_connections(self):
        """Close idle connections in all pools."""
        with self._lock:
            for pool in self._pools.values():
                pool.close_idle_connections()
    
    def get_all_stats(self) -> Dict[str, Dict[str, Any]]:
        """
        Get statistics for all pools.
        
        Returns:
            Dictionary mapping database paths to statistics
        """
        with self._lock:
            return {
                db_path: pool.get_stats()
                for db_path, pool in self._pools.items()
            }
    
    def close_all(self):
        """Close all connection pools."""
        with self._lock:
            for pool in self._pools.values():
                pool.close()
            self._pools.clear()
        
        logger.info("All connection pools closed")


# Global pool manager instance
_global_pool_manager: Optional[MultiDatabasePool] = None
_manager_lock = threading.Lock()


def get_global_pool_manager() -> MultiDatabasePool:
    """
    Get or create the global connection pool manager.
    
    Returns:
        Global MultiDatabasePool instance
    """
    global _global_pool_manager
    
    with _manager_lock:
        if _global_pool_manager is None:
            _global_pool_manager = MultiDatabasePool()
        return _global_pool_manager


def close_all_pools():
    """Close all connection pools globally."""
    global _global_pool_manager
    
    with _manager_lock:
        if _global_pool_manager:
            _global_pool_manager.close_all()
            _global_pool_manager = None