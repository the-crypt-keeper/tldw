"""
Database connection pool for RAG services.

Provides thread-safe connection pooling for SQLite databases to prevent
connection leaks and improve performance.
"""

import sqlite3
import threading
import queue
import time
from typing import Optional, Dict, Any, ContextManager
from pathlib import Path
from contextlib import contextmanager
from loguru import logger



class ConnectionPool:
    """Thread-safe SQLite connection pool."""
    
    def __init__(self, db_path: str, pool_size: int = 5, timeout: float = 30.0):
        """
        Initialize connection pool.
        
        Args:
            db_path: Path to SQLite database
            pool_size: Number of connections to maintain
            timeout: Timeout for getting connections
        """
        self.db_path = db_path
        self.pool_size = pool_size
        self.timeout = timeout
        self._pool = queue.Queue(maxsize=pool_size)
        self._all_connections = []
        self._lock = threading.Lock()
        self._closed = False
        
        # Initialize the pool
        self._initialize_pool()
        
    def _initialize_pool(self):
        """Create initial connections."""
        for _ in range(self.pool_size):
            conn = self._create_connection()
            self._pool.put(conn)
            self._all_connections.append(conn)
            
    def _create_connection(self) -> sqlite3.Connection:
        """Create a new database connection."""
        conn = sqlite3.connect(self.db_path, check_same_thread=False)
        conn.row_factory = sqlite3.Row
        # Enable WAL mode for better concurrency
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=NORMAL")
        return conn
    
    @contextmanager
    def get_connection(self) -> ContextManager[sqlite3.Connection]:
        """
        Get a connection from the pool.
        
        Usage:
            with pool.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT * FROM table")
        """
        if self._closed:
            raise RuntimeError("Connection pool is closed")
            
        conn = None
        try:
            # Get connection from pool with timeout
            conn = self._pool.get(timeout=self.timeout)
            yield conn
        except queue.Empty:
            raise TimeoutError(f"Could not get connection within {self.timeout} seconds")
        finally:
            # Return connection to pool
            if conn is not None:
                try:
                    # Test if connection is still valid
                    conn.execute("SELECT 1")
                    self._pool.put(conn)
                except sqlite3.Error:
                    # Connection is broken, create a new one
                    logger.warning("Replacing broken database connection")
                    with self._lock:
                        try:
                            self._all_connections.remove(conn)
                        except ValueError:
                            # Connection was already removed
                            logger.debug("Broken connection was already removed from tracking")
                        new_conn = self._create_connection()
                        self._pool.put(new_conn)
                        self._all_connections.append(new_conn)
    
    @contextmanager
    def transaction(self) -> ContextManager[sqlite3.Connection]:
        """
        Get a connection from the pool with automatic transaction management.
        
        Automatically commits on success and rolls back on exception.
        
        Usage:
            with pool.transaction() as conn:
                cursor = conn.cursor()
                cursor.execute("INSERT INTO table VALUES (?)", (value,))
                # Automatically commits if no exception
        """
        if self._closed:
            raise RuntimeError("Connection pool is closed")
        
        conn = None
        try:
            # Get connection from pool
            conn = self._pool.get(timeout=self.timeout)
            
            # Start explicit transaction
            conn.execute("BEGIN")
            
            yield conn
            
            # Commit on success
            conn.commit()
            logger.debug("Transaction committed successfully")
            
        except queue.Empty:
            raise TimeoutError(f"Could not get connection within {self.timeout} seconds")
        except Exception as e:
            # Rollback on any error
            if conn is not None:
                try:
                    conn.rollback()
                    logger.warning(f"Transaction rolled back due to error: {e}")
                except sqlite3.Error as rollback_error:
                    logger.error(f"Failed to rollback transaction: {rollback_error}")
            raise
        finally:
            # Return connection to pool
            if conn is not None:
                try:
                    # Ensure we're not in a transaction
                    conn.rollback()  # This is safe even if already rolled back
                    # Test if connection is still valid
                    conn.execute("SELECT 1")
                    self._pool.put(conn)
                except sqlite3.Error:
                    # Connection is broken, create a new one
                    logger.warning("Replacing broken database connection after transaction")
                    with self._lock:
                        if conn in self._all_connections:
                            self._all_connections.remove(conn)
                        new_conn = self._create_connection()
                        self._pool.put(new_conn)
                        self._all_connections.append(new_conn)
    
    def close(self):
        """Close all connections in the pool."""
        with self._lock:
            self._closed = True
            
            # Close all connections
            while not self._pool.empty():
                try:
                    conn = self._pool.get_nowait()
                    conn.close()
                except queue.Empty:
                    break
                    
            # Close any connections that might be in use
            # Use a copy of the list to avoid modification during iteration
            connections_to_close = list(self._all_connections)
            for conn in connections_to_close:
                try:
                    conn.close()
                except Exception as e:
                    logger.error(f"Error closing connection: {e}")
                finally:
                    # Always remove from list, even if close fails
                    try:
                        self._all_connections.remove(conn)
                    except ValueError:
                        # Connection was already removed
                        pass
                    
            # Clear any remaining connections (shouldn't be any, but just in case)
            self._all_connections.clear()
            
    def __enter__(self):
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()


class ConnectionPoolManager:
    """Manages connection pools for different databases."""
    
    _instance = None
    _lock = threading.Lock()
    _pools: Dict[str, ConnectionPool] = {}
    
    def __new__(cls):
        """Singleton pattern to ensure one manager instance."""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
        return cls._instance
    
    def get_pool(self, db_path: str, pool_size: int = 5) -> ConnectionPool:
        """
        Get or create a connection pool for a database.
        
        Args:
            db_path: Path to the database
            pool_size: Size of the connection pool
            
        Returns:
            ConnectionPool instance
        """
        db_path = str(Path(db_path).resolve())
        
        with self._lock:
            if db_path not in self._pools:
                logger.info(f"Creating new connection pool for {db_path} with size {pool_size}")
                self._pools[db_path] = ConnectionPool(db_path, pool_size)
                
            return self._pools[db_path]
    
    def close_pool(self, db_path: str):
        """Close a specific connection pool."""
        db_path = str(Path(db_path).resolve())
        
        with self._lock:
            if db_path in self._pools:
                self._pools[db_path].close()
                del self._pools[db_path]
                logger.info(f"Closed connection pool for {db_path}")
    
    def close_all(self):
        """Close all connection pools."""
        with self._lock:
            for db_path, pool in self._pools.items():
                pool.close()
                logger.info(f"Closed connection pool for {db_path}")
            self._pools.clear()


# Global connection pool manager
_pool_manager = ConnectionPoolManager()


def get_connection_pool(db_path: str, pool_size: int = 5) -> ConnectionPool:
    """
    Get a connection pool for a database.
    
    Args:
        db_path: Path to the database
        pool_size: Size of the connection pool
        
    Returns:
        ConnectionPool instance
    """
    return _pool_manager.get_pool(db_path, pool_size)


def close_all_pools():
    """Close all connection pools."""
    _pool_manager.close_all()