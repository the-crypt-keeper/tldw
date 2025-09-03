"""
Database connection pool for Evaluations module.

Provides connection pooling and management for SQLite operations,
preparing for future aiosqlite migration.
"""

import asyncio
import sqlite3
import threading
from typing import Optional, Callable, Any, Dict, List, AsyncContextManager
from contextlib import contextmanager, asynccontextmanager
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from loguru import logger
import time
from collections import deque
from threading import Lock, Condition
import weakref

from tldw_Server_API.app.core.Evaluations.config_manager import get_config
from tldw_Server_API.app.core.Evaluations.metrics import get_metrics


@dataclass
class ConnectionStats:
    """Connection pool statistics."""
    total_connections: int = 0
    active_connections: int = 0
    idle_connections: int = 0
    created_connections: int = 0
    closed_connections: int = 0
    checkout_count: int = 0
    total_checkout_time: float = 0.0
    max_checkout_time: float = 0.0
    connection_errors: int = 0
    pool_exhausted_count: int = 0
    
    @property
    def avg_checkout_time(self) -> float:
        """Average connection checkout time in seconds."""
        return self.total_checkout_time / max(1, self.checkout_count)


class PooledConnection:
    """Wrapper for a pooled database connection."""
    
    def __init__(self, connection: sqlite3.Connection, pool: 'ConnectionPool'):
        self.connection = connection
        self.pool = pool
        self.created_at = time.time()
        self.last_used = time.time()
        self.checkout_time: Optional[float] = None
        self.in_use = False
        self.connection_id = id(connection)
        self._lock = threading.RLock()  # Thread safety for shared connections
        
        # Configure connection for optimal performance
        self._configure_connection()
    
    def _configure_connection(self):
        """Configure connection with optimal settings."""
        with self._lock:
            try:
                # Enable WAL mode for better concurrency
                self.connection.execute("PRAGMA journal_mode=WAL")
                
                # Set reasonable timeout
                self.connection.execute("PRAGMA busy_timeout=30000")  # 30 seconds
                
                # Optimize for performance
                self.connection.execute("PRAGMA synchronous=NORMAL")
                self.connection.execute("PRAGMA temp_store=MEMORY")
                self.connection.execute("PRAGMA mmap_size=268435456")  # 256MB
                
                # Enable foreign keys
                self.connection.execute("PRAGMA foreign_keys=ON")
                
                self.connection.commit()
                
            except sqlite3.Error as e:
                logger.warning(f"Failed to configure connection {self.connection_id}: {e}")
    
    def execute(self, query: str, parameters: tuple = ()) -> sqlite3.Cursor:
        """Execute query on the connection."""
        with self._lock:
            self.last_used = time.time()
            return self.connection.execute(query, parameters)
    
    def executemany(self, query: str, parameters: List[tuple]) -> sqlite3.Cursor:
        """Execute query multiple times."""
        with self._lock:
            self.last_used = time.time()
            return self.connection.executemany(query, parameters)
    
    def commit(self):
        """Commit transaction."""
        with self._lock:
            self.connection.commit()
    
    def rollback(self):
        """Rollback transaction."""
        with self._lock:
            self.connection.rollback()
    
    def close(self):
        """Close the underlying connection."""
        with self._lock:
            try:
                self.connection.close()
            except sqlite3.Error as e:
                logger.warning(f"Error closing connection {self.connection_id}: {e}")
    
    def is_stale(self, max_age_seconds: int = 3600) -> bool:
        """Check if connection is stale."""
        return (time.time() - self.created_at) > max_age_seconds
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        # Return connection to pool
        self.pool._return_connection(self)


class ConnectionPool:
    """
    Database connection pool with monitoring and health management.
    
    Prepares for future aiosqlite migration by providing async-compatible interface.
    """
    
    def __init__(
        self,
        db_path: str,
        pool_size: int = 10,
        max_overflow: int = 20,
        pool_timeout: float = 30.0,
        pool_recycle: int = 3600,
        enable_monitoring: bool = True
    ):
        """
        Initialize connection pool.
        
        Args:
            db_path: Path to database file
            pool_size: Core pool size
            max_overflow: Maximum additional connections
            pool_timeout: Timeout for getting connection (seconds)
            pool_recycle: Connection max age (seconds)
            enable_monitoring: Enable pool monitoring
        """
        self.db_path = Path(db_path)
        self.pool_size = pool_size
        self.max_overflow = max_overflow
        self.pool_timeout = pool_timeout
        self.pool_recycle = pool_recycle
        self.enable_monitoring = enable_monitoring
        
        # Connection management
        self._pool: deque = deque()
        self._overflow_connections: Dict[int, PooledConnection] = {}
        self._lock = Lock()
        self._condition = Condition(self._lock)
        
        # Statistics
        self._stats = ConnectionStats()
        self._created_connections: weakref.WeakSet = weakref.WeakSet()
        
        # Metrics
        self._metrics = get_metrics() if enable_monitoring else None
        
        # Background maintenance
        self._maintenance_task = None
        self._shutdown = False
        
        # Initialize pool
        self._initialize_pool()
        
        if enable_monitoring:
            self._start_maintenance()
    
    def _initialize_pool(self):
        """Initialize the connection pool."""
        logger.info(f"Initializing connection pool: size={self.pool_size}, max_overflow={self.max_overflow}")
        
        # Ensure database directory exists
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Pre-create core pool connections
        for _ in range(self.pool_size):
            try:
                conn = self._create_connection()
                self._pool.append(conn)
            except Exception as e:
                logger.error(f"Failed to create initial connection: {e}")
                break
        
        logger.info(f"Connection pool initialized with {len(self._pool)} connections")
    
    def _create_connection(self) -> PooledConnection:
        """Create a new database connection."""
        try:
            # Create SQLite connection
            # For thread safety in a connection pool, we need to allow sharing
            # but ensure proper synchronization at the pool level
            conn = sqlite3.connect(
                str(self.db_path),
                timeout=30.0,
                check_same_thread=False  # Allow sharing between threads
            )
            
            # Row factory for dict-like access
            conn.row_factory = sqlite3.Row
            
            # Create pooled wrapper
            pooled_conn = PooledConnection(conn, self)
            
            # Track statistics
            self._stats.created_connections += 1
            self._stats.total_connections += 1
            self._created_connections.add(pooled_conn)
            
            logger.debug(f"Created new database connection {pooled_conn.connection_id}")
            
            if self._metrics:
                self._metrics.record_database_query("connection_created", "pool", 0.0, True)
            
            return pooled_conn
            
        except sqlite3.Error as e:
            self._stats.connection_errors += 1
            if self._metrics:
                self._metrics.record_database_query("connection_error", "pool", 0.0, False, str(e))
            
            logger.error(f"Failed to create database connection: {e}")
            raise
    
    @contextmanager
    def get_connection(self) -> PooledConnection:
        """
        Get a connection from the pool (synchronous).
        
        Returns:
            PooledConnection: Connection wrapper
        """
        start_time = time.time()
        connection = None
        
        try:
            with self._condition:
                # Try to get from pool
                while len(self._pool) == 0:
                    # Check if we can create overflow connection
                    if len(self._overflow_connections) < self.max_overflow:
                        connection = self._create_connection()
                        connection.in_use = True
                        connection.checkout_time = start_time
                        self._overflow_connections[connection.connection_id] = connection
                        self._stats.active_connections += 1
                        break
                    
                    # Pool exhausted, wait
                    self._stats.pool_exhausted_count += 1
                    if not self._condition.wait(timeout=self.pool_timeout):
                        raise TimeoutError(f"Timeout waiting for database connection after {self.pool_timeout}s")
                
                if connection is None:
                    # Get from pool
                    connection = self._pool.popleft()
                    connection.in_use = True
                    connection.checkout_time = start_time
                    self._stats.active_connections += 1
                    self._stats.idle_connections -= 1
            
            # Update statistics
            checkout_time = time.time() - start_time
            self._stats.checkout_count += 1
            self._stats.total_checkout_time += checkout_time
            self._stats.max_checkout_time = max(self._stats.max_checkout_time, checkout_time)
            
            if self._metrics:
                self._metrics.record_database_query("connection_checkout", "pool", checkout_time)
            
            logger.debug(f"Checked out connection {connection.connection_id} in {checkout_time:.3f}s")
            
            yield connection
            
        except Exception as e:
            logger.error(f"Error getting database connection: {e}")
            if self._metrics:
                self._metrics.record_database_query("connection_checkout_error", "pool", 0.0, False, str(e))
            raise
        
        finally:
            if connection:
                self._return_connection(connection)
    
    async def get_connection_async(self) -> AsyncContextManager[PooledConnection]:
        """
        Get a connection asynchronously (future aiosqlite compatibility).
        
        Returns:
            AsyncContextManager[PooledConnection]: Connection wrapper
        """
        return self._async_connection_context()
    
    @asynccontextmanager
    async def _async_connection_context(self) -> PooledConnection:
        """Async context manager for connections."""
        # For now, run sync version in thread pool
        # In future aiosqlite migration, this will be truly async
        loop = asyncio.get_event_loop()
        
        # Use thread pool to avoid blocking
        connection = await loop.run_in_executor(None, self._get_connection_sync)
        
        try:
            yield connection
        finally:
            self._return_connection(connection)
    
    def _get_connection_sync(self) -> PooledConnection:
        """Synchronous connection getter for async wrapper."""
        with self.get_connection() as conn:
            return conn
    
    def _return_connection(self, connection: PooledConnection):
        """Return a connection to the pool."""
        if not connection or not connection.in_use:
            return
        
        checkout_time = time.time() - (connection.checkout_time or time.time())
        
        with self._condition:
            connection.in_use = False
            connection.checkout_time = None
            self._stats.active_connections -= 1
            
            # Check if connection is stale or we have overflow
            if (connection.is_stale(self.pool_recycle) or 
                connection.connection_id in self._overflow_connections):
                
                # Remove overflow connection or close stale connection
                if connection.connection_id in self._overflow_connections:
                    del self._overflow_connections[connection.connection_id]
                
                self._close_connection(connection)
                self._condition.notify()
                
            elif len(self._pool) < self.pool_size:
                # Return to pool
                self._pool.append(connection)
                self._stats.idle_connections += 1
                self._condition.notify()
            else:
                # Pool is full, close connection
                self._close_connection(connection)
        
        if self._metrics:
            self._metrics.record_database_query("connection_return", "pool", checkout_time)
        
        logger.debug(f"Returned connection {connection.connection_id} after {checkout_time:.3f}s")
    
    def _close_connection(self, connection: PooledConnection):
        """Close a connection and update statistics."""
        try:
            connection.close()
            self._stats.closed_connections += 1
            self._stats.total_connections -= 1
            
            if self._metrics:
                self._metrics.record_database_query("connection_closed", "pool", 0.0)
            
            logger.debug(f"Closed connection {connection.connection_id}")
            
        except Exception as e:
            logger.error(f"Error closing connection {connection.connection_id}: {e}")
    
    def _start_maintenance(self):
        """Start background maintenance task."""
        if self._maintenance_task is not None:
            return
        
        def maintenance_worker():
            """Background maintenance worker."""
            while not self._shutdown:
                try:
                    self._perform_maintenance()
                    time.sleep(60)  # Run every minute
                except Exception as e:
                    logger.error(f"Pool maintenance error: {e}")
                    time.sleep(30)
        
        self._maintenance_task = threading.Thread(target=maintenance_worker, daemon=True)
        self._maintenance_task.start()
        
        logger.info("Started connection pool maintenance")
    
    def _perform_maintenance(self):
        """Perform pool maintenance tasks."""
        now = time.time()
        
        with self._lock:
            # Remove stale connections from pool
            fresh_pool = deque()
            for conn in self._pool:
                if conn.is_stale(self.pool_recycle):
                    self._close_connection(conn)
                    self._stats.idle_connections -= 1
                else:
                    fresh_pool.append(conn)
            
            self._pool = fresh_pool
            
            # Check overflow connections
            stale_overflow = []
            for conn_id, conn in self._overflow_connections.items():
                if not conn.in_use and conn.is_stale(self.pool_recycle):
                    stale_overflow.append(conn_id)
            
            for conn_id in stale_overflow:
                conn = self._overflow_connections.pop(conn_id)
                self._close_connection(conn)
            
            # Ensure minimum pool size
            while len(self._pool) < self.pool_size:
                try:
                    conn = self._create_connection()
                    self._pool.append(conn)
                    self._stats.idle_connections += 1
                except Exception as e:
                    logger.error(f"Failed to create maintenance connection: {e}")
                    break
        
        # Log statistics periodically
        if self.enable_monitoring and int(now) % 300 == 0:  # Every 5 minutes
            self._log_statistics()
        
        # Record metrics
        if self._metrics:
            self._metrics.set_database_connections(self._stats.total_connections)
            # Record pool statistics with a custom gauge update
            self._metrics.database_query_duration.labels(operation="pool_stats", table="pool").observe(0.0)
    
    def _log_statistics(self):
        """Log pool statistics."""
        stats = self.get_statistics()
        logger.info(
            f"Connection pool stats: "
            f"total={stats.total_connections}, "
            f"active={stats.active_connections}, "
            f"idle={stats.idle_connections}, "
            f"checkout_count={stats.checkout_count}, "
            f"avg_checkout_time={stats.avg_checkout_time:.3f}s, "
            f"pool_exhausted={stats.pool_exhausted_count}"
        )
    
    def get_statistics(self) -> ConnectionStats:
        """Get current pool statistics."""
        with self._lock:
            # Update current counts
            stats = ConnectionStats(
                total_connections=self._stats.total_connections,
                active_connections=self._stats.active_connections,
                idle_connections=len(self._pool),
                created_connections=self._stats.created_connections,
                closed_connections=self._stats.closed_connections,
                checkout_count=self._stats.checkout_count,
                total_checkout_time=self._stats.total_checkout_time,
                max_checkout_time=self._stats.max_checkout_time,
                connection_errors=self._stats.connection_errors,
                pool_exhausted_count=self._stats.pool_exhausted_count
            )
            
            return stats
    
    def get_health_status(self) -> Dict[str, Any]:
        """Get pool health status."""
        stats = self.get_statistics()
        
        # Calculate health indicators
        pool_utilization = stats.active_connections / max(1, stats.total_connections)
        error_rate = stats.connection_errors / max(1, stats.checkout_count)
        
        health_score = 100.0
        issues = []
        
        # Check for issues
        if pool_utilization > 0.9:
            health_score -= 20
            issues.append("High pool utilization")
        
        if error_rate > 0.1:
            health_score -= 30
            issues.append("High connection error rate")
        
        if stats.pool_exhausted_count > 0:
            health_score -= 25
            issues.append("Pool exhaustion detected")
        
        if stats.avg_checkout_time > 1.0:
            health_score -= 15
            issues.append("Slow connection checkout")
        
        return {
            "health_score": max(0, health_score),
            "status": "healthy" if health_score >= 80 else "degraded" if health_score >= 50 else "unhealthy",
            "statistics": {
                "total_connections": stats.total_connections,
                "active_connections": stats.active_connections,
                "idle_connections": stats.idle_connections,
                "pool_utilization": pool_utilization,
                "checkout_count": stats.checkout_count,
                "avg_checkout_time_ms": int(stats.avg_checkout_time * 1000),
                "max_checkout_time_ms": int(stats.max_checkout_time * 1000),
                "connection_errors": stats.connection_errors,
                "error_rate": error_rate,
                "pool_exhausted_count": stats.pool_exhausted_count
            },
            "issues": issues,
            "configuration": {
                "pool_size": self.pool_size,
                "max_overflow": self.max_overflow,
                "pool_timeout": self.pool_timeout,
                "pool_recycle": self.pool_recycle
            }
        }
    
    def shutdown(self):
        """Shutdown the connection pool."""
        logger.info("Shutting down connection pool")
        
        self._shutdown = True
        
        with self._condition:
            # Close all pooled connections
            while self._pool:
                conn = self._pool.popleft()
                self._close_connection(conn)
            
            # Close overflow connections
            for conn in list(self._overflow_connections.values()):
                self._close_connection(conn)
            
            self._overflow_connections.clear()
            self._condition.notify_all()
        
        # Wait for maintenance task to finish
        if self._maintenance_task and self._maintenance_task.is_alive():
            self._maintenance_task.join(timeout=5)
        
        logger.info("Connection pool shutdown complete")
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.shutdown()


class EvaluationsConnectionManager:
    """
    Connection manager for the Evaluations module.
    
    Provides centralized connection management with configuration integration.
    """
    
    def __init__(self, db_path: Optional[str] = None):
        """
        Initialize connection manager.
        
        Args:
            db_path: Path to database file
        """
        if db_path is None:
            db_dir = Path(__file__).parent.parent.parent.parent / "Databases"
            db_path = db_dir / "evaluations.db"
        
        # Load pool configuration
        db_config = get_config("database.connection", {})
        
        self._pool = ConnectionPool(
            db_path=str(db_path),
            pool_size=db_config.get("pool_size", 10),
            max_overflow=db_config.get("max_overflow", 20),
            pool_timeout=db_config.get("pool_timeout", 30),
            pool_recycle=db_config.get("pool_recycle", 3600),
            enable_monitoring=True
        )
        
        logger.info(f"Initialized Evaluations connection manager for {db_path}")
    
    def get_connection(self) -> PooledConnection:
        """Get a database connection (synchronous)."""
        return self._pool.get_connection()
    
    async def get_connection_async(self) -> AsyncContextManager[PooledConnection]:
        """Get a database connection (asynchronous)."""
        return await self._pool.get_connection_async()
    
    def get_health_status(self) -> Dict[str, Any]:
        """Get connection manager health status."""
        return self._pool.get_health_status()
    
    def get_statistics(self) -> ConnectionStats:
        """Get connection pool statistics."""
        return self._pool.get_statistics()
    
    def shutdown(self):
        """Shutdown connection manager."""
        self._pool.shutdown()


# Global connection manager instance
connection_manager = EvaluationsConnectionManager()


# Convenience functions for easy access
def get_connection():
    """Get a database connection from the global pool."""
    return connection_manager.get_connection()


async def get_connection_async():
    """Get an async database connection from the global pool."""
    return await connection_manager.get_connection_async()


def get_connection_health() -> Dict[str, Any]:
    """Get connection pool health status."""
    return connection_manager.get_health_status()


def get_connection_stats() -> ConnectionStats:
    """Get connection pool statistics."""
    return connection_manager.get_statistics()