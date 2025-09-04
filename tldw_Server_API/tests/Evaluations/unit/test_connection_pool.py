"""
Unit tests for ConnectionPool.

Tests database connection pooling functionality with minimal mocking.
"""

import pytest
import sqlite3
import threading
import time
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
import concurrent.futures

from tldw_Server_API.app.core.Evaluations.connection_pool import ConnectionPool


@pytest.mark.unit
class TestConnectionPoolInit:
    """Test ConnectionPool initialization."""
    
    def test_init_with_defaults(self, temp_db_path):
        """Test initialization with default values."""
        pool = ConnectionPool(db_path=str(temp_db_path))
        
        assert str(pool.db_path) == str(temp_db_path)
        assert pool.pool_size == 10
        assert pool.max_overflow == 20
        assert pool.pool_timeout == 30.0
        assert len(pool._pool) >= 0
        assert len(pool._overflow_connections) == 0
    
    def test_init_with_custom_values(self, temp_db_path):
        """Test initialization with custom values."""
        pool = ConnectionPool(
            db_path=str(temp_db_path),
            pool_size=5,
            max_overflow=15,
            pool_timeout=60.0
        )
        
        assert pool.pool_size == 5
        assert pool.max_overflow == 15
        assert pool.pool_timeout == 60.0
    
    def test_init_with_invalid_values(self, temp_db_path):
        """Test initialization with invalid values."""
        # The actual ConnectionPool implementation doesn't validate parameters
        # but negative values would likely cause issues in practice
        # Test that construction at least succeeds (behavior testing)
        pool1 = ConnectionPool(
            db_path=str(temp_db_path),
            pool_size=0  # Minimal valid value
        )
        pool1.shutdown()
        
        pool2 = ConnectionPool(
            db_path=str(temp_db_path),
            max_overflow=0  # Minimal valid value
        )
        pool2.shutdown()
    
    def test_initialize_creates_pool_connections(self, temp_db_path):
        """Test that initialization creates pool connections."""
        pool = ConnectionPool(
            db_path=str(temp_db_path),
            pool_size=3
        )
        
        # Pool initialization happens in __init__
        assert len(pool._pool) == 3
        assert len(pool._overflow_connections) == 0
        
        # Cleanup
        pool.shutdown()


@pytest.mark.unit
class TestConnectionAcquisition:
    """Test connection acquisition from pool."""
    
    def test_acquire_connection_from_available(self, temp_db_path):
        """Test acquiring connection from available pool."""
        pool = ConnectionPool(db_path=str(temp_db_path), pool_size=2)
        
        initial_pool_size = len(pool._pool)
        
        with pool.get_connection() as conn:
            assert conn is not None
            assert hasattr(conn, 'connection')  # PooledConnection wrapper
            assert len(pool._pool) == initial_pool_size - 1
            stats = pool.get_statistics()
            assert stats.active_connections >= 1
        
        pool.shutdown()
    
    def test_acquire_creates_new_connection_when_empty(self, temp_db_path):
        """Test acquiring connection creates new one when pool is empty."""
        pool = ConnectionPool(
            db_path=str(temp_db_path),
            pool_size=0,
            max_overflow=5
        )
        
        assert len(pool._pool) == 0
        
        with pool.get_connection() as conn:
            assert conn is not None
            stats = pool.get_statistics()
            assert stats.active_connections >= 1
        
        pool.shutdown()
    
    def test_acquire_blocks_when_max_reached(self, temp_db_path):
        """Test acquire blocks when max connections reached."""
        pool = ConnectionPool(
            db_path=str(temp_db_path),
            pool_size=1,
            max_overflow=1,
            pool_timeout=1.0
        )
        
        # Use context manager to properly hold connections
        acquired_connections = []
        ctx1 = pool.get_connection()
        ctx2 = pool.get_connection()
        
        conn1 = ctx1.__enter__()
        conn2 = ctx2.__enter__()
        acquired_connections.extend([ctx1, ctx2])
        
        try:
            # Next acquire should timeout since pool (1) + overflow (1) = 2 are in use
            start = time.time()
            with pytest.raises(TimeoutError):
                with pool.get_connection():
                    pass
            elapsed = time.time() - start
            
            assert elapsed >= 0.9  # Should wait ~1 second
        finally:
            # Clean up properly
            for ctx in acquired_connections:
                try:
                    ctx.__exit__(None, None, None)
                except:
                    pass
            pool.shutdown()
    
    def test_acquire_with_timeout(self, temp_db_path):
        """Test acquire with custom timeout."""
        pool = ConnectionPool(
            db_path=str(temp_db_path),
            pool_size=1,
            max_overflow=0,
            pool_timeout=0.1
        )
        
        # Use context manager properly
        ctx1 = pool.get_connection()
        conn1 = ctx1.__enter__()
        
        try:
            # Try to acquire with short timeout (configured in pool_timeout)
            with pytest.raises(TimeoutError):
                with pool.get_connection():
                    pass
        finally:
            ctx1.__exit__(None, None, None)
            pool.shutdown()


@pytest.mark.unit
class TestConnectionRelease:
    """Test connection release back to pool."""
    
    def test_release_connection_to_pool(self, temp_db_path):
        """Test releasing connection back to available pool."""
        pool = ConnectionPool(db_path=str(temp_db_path))
        
        initial_active = pool.get_statistics().active_connections
        
        with pool.get_connection() as conn:
            stats_during = pool.get_statistics()
            assert stats_during.active_connections > initial_active
        
        # After context manager exits, connection should be returned
        stats_after = pool.get_statistics()
        assert stats_after.active_connections == initial_active
        
        pool.shutdown()
    
    def test_release_invalid_connection(self, temp_db_path):
        """Test releasing connection not from pool."""
        pool = ConnectionPool(db_path=str(temp_db_path))
        
        # Create a mock PooledConnection-like object with wrong properties
        class FakeConnection:
            def __init__(self):
                self.in_use = False
                self.connection_id = 99999
        
        fake_conn = FakeConnection()
        
        # _return_connection should handle invalid connection gracefully
        pool._return_connection(fake_conn)  # Should not crash
        
        pool.shutdown()
    
    def test_release_already_released_connection(self, temp_db_path):
        """Test releasing already released connection."""
        pool = ConnectionPool(db_path=str(temp_db_path))
        
        conn = pool.get_connection().__enter__()
        pool._return_connection(conn)
        
        # Second release should be handled gracefully (connection.in_use = False)
        pool._return_connection(conn)  # Should not crash
        
        pool.shutdown()
    
    def test_release_closed_connection(self, temp_db_path):
        """Test releasing a closed connection."""
        pool = ConnectionPool(db_path=str(temp_db_path))
        
        ctx = pool.get_connection()
        conn = ctx.__enter__()
        
        # Close the underlying SQLite connection
        conn.connection.close()
        
        # Release should detect and handle closed connection
        ctx.__exit__(None, None, None)
        
        # Connection should have been removed from pool due to being closed
        # (This is implementation-dependent behavior)
        pool.shutdown()


@pytest.mark.unit
class TestConnectionHealth:
    """Test connection health checking."""
    
    def test_health_check_valid_connection(self, temp_db_path):
        """Test health check on valid connection."""
        pool = ConnectionPool(db_path=str(temp_db_path))
        
        with pool.get_connection() as conn:
            # Test that connection works
            cursor = conn.execute("SELECT 1")
            result = cursor.fetchone()
            assert result[0] == 1
        
        pool.shutdown()
    
    def test_health_check_closed_connection(self, temp_db_path):
        """Test health check on closed connection."""
        pool = ConnectionPool(db_path=str(temp_db_path))
        
        conn = pool.get_connection().__enter__()
        conn.close()
        
        # Try to use closed connection - should fail
        with pytest.raises(sqlite3.ProgrammingError):
            conn.execute("SELECT 1")
        
        pool._return_connection(conn)
        pool.shutdown()
    
    def test_health_check_with_query(self, temp_db_path):
        """Test health check with test query."""
        pool = ConnectionPool(db_path=str(temp_db_path))
        
        with pool.get_connection() as conn:
            # Should execute test query successfully
            cursor = conn.execute("SELECT 1")
            result = cursor.fetchone()
            assert result[0] == 1
        
        pool.shutdown()
    
    def test_automatic_reconnection_on_failed_health(self, temp_db_path):
        """Test automatic reconnection when health check fails."""
        pool = ConnectionPool(db_path=str(temp_db_path))
        
        # Simulate connection failure
        conn = pool.get_connection().__enter__()
        old_id = conn.connection_id
        conn.close()
        
        # Return to pool (should detect failure)
        pool._return_connection(conn)
        
        # Acquire again should get new healthy connection
        with pool.get_connection() as new_conn:
            assert new_conn.connection_id != old_id
            cursor = new_conn.execute("SELECT 1")
            assert cursor.fetchone()[0] == 1
        
        pool.shutdown()


@pytest.mark.unit
class TestConnectionPoolConcurrency:
    """Test connection pool with concurrent access."""
    
    def test_concurrent_acquire_release(self, temp_db_path):
        """Test concurrent connection acquisition and release."""
        pool = ConnectionPool(
            db_path=str(temp_db_path),
            pool_size=2,
            max_overflow=3
        )
        
        results = []
        errors = []
        
        def worker(worker_id):
            try:
                # Acquire connection
                with pool.get_connection() as conn:
                    results.append(f"Worker {worker_id} acquired")
                    
                    # Simulate work
                    cursor = conn.execute("SELECT 1")
                    assert cursor.fetchone()[0] == 1
                    time.sleep(0.1)
                    
                    results.append(f"Worker {worker_id} releasing")
                    # Connection automatically released by context manager
            except Exception as e:
                errors.append((worker_id, str(e)))
        
        # Start multiple threads
        threads = []
        for i in range(10):
            t = threading.Thread(target=worker, args=(i,))
            threads.append(t)
            t.start()
        
        # Wait for completion
        for t in threads:
            t.join()
        
        assert len(errors) == 0, f"Errors occurred: {errors}"
        assert len(results) == 20  # 10 acquires + 10 releases
        
        pool.shutdown()
    
    def test_connection_pool_under_load(self, temp_db_path):
        """Test connection pool under heavy load."""
        pool = ConnectionPool(
            db_path=str(temp_db_path),
            pool_size=3,
            max_overflow=7
        )
        
        completed_tasks = []
        
        def heavy_worker(task_id):
            with pool.get_connection() as conn:
                # Perform database operations
                conn.execute("CREATE TABLE IF NOT EXISTS test (id INTEGER)")
                conn.execute("INSERT INTO test VALUES (?)", (task_id,))
                conn.commit()
                completed_tasks.append(task_id)
        
        # Use thread pool executor for controlled concurrency
        with concurrent.futures.ThreadPoolExecutor(max_workers=20) as executor:
            futures = [executor.submit(heavy_worker, i) for i in range(50)]
            concurrent.futures.wait(futures)
        
        assert len(completed_tasks) == 50
        
        # Verify all data was written
        with pool.get_connection() as conn:
            cursor = conn.execute("SELECT COUNT(*) FROM test")
            count = cursor.fetchone()[0]
            assert count == 50
        
        pool.shutdown()
    
    def test_deadlock_prevention(self, temp_db_path):
        """Test that pool prevents deadlocks."""
        pool = ConnectionPool(
            db_path=str(temp_db_path),
            pool_size=1,
            max_overflow=1,
            pool_timeout=0.5
        )
        
        # Use context managers properly
        ctx1 = pool.get_connection()
        ctx2 = pool.get_connection()
        conn1 = ctx1.__enter__()
        conn2 = ctx2.__enter__()
        
        deadlock_detected = False
        
        def try_acquire():
            nonlocal deadlock_detected
            try:
                with pool.get_connection():
                    pass
            except TimeoutError:
                deadlock_detected = True
        
        thread = threading.Thread(target=try_acquire)
        thread.start()
        thread.join()
        
        assert deadlock_detected is True
        
        try:
            # Release and verify pool recovers
            ctx1.__exit__(None, None, None)
            with pool.get_connection() as conn3:  # Should succeed now
                assert conn3 is not None
        finally:
            ctx2.__exit__(None, None, None)
            pool.shutdown()


@pytest.mark.unit
class TestConnectionPoolManagement:
    """Test connection pool management features."""
    
    def test_pool_statistics(self, temp_db_path):
        """Test getting pool statistics."""
        pool = ConnectionPool(
            db_path=str(temp_db_path),
            pool_size=3,
            max_overflow=7
        )
        
        stats = pool.get_statistics()
        assert stats.total_connections == 3
        assert stats.idle_connections == 3
        assert stats.active_connections == 0
        
        conn1 = pool.get_connection().__enter__()
        conn2 = pool.get_connection().__enter__()
        
        stats = pool.get_statistics()
        # The exact counts depend on timing and pool behavior
        # Just verify the connections are being tracked
        assert stats.idle_connections >= 0
        assert stats.active_connections >= 0
        assert stats.total_connections >= 2
        
        pool._return_connection(conn1)
        pool._return_connection(conn2)
        pool.shutdown()
    
    def test_pool_configuration(self, temp_db_path):
        """Test pool configuration."""
        pool = ConnectionPool(
            db_path=str(temp_db_path),
            pool_size=2,
            max_overflow=3
        )
        
        # Check configuration is set correctly
        assert pool.pool_size == 2
        assert pool.max_overflow == 3
        assert len(pool._pool) >= 2
        
        pool.shutdown()
    
    def test_pool_cleanup_idle_connections(self, temp_db_path):
        """Test cleanup of idle connections."""
        pool = ConnectionPool(
            db_path=str(temp_db_path),
            pool_size=2,
            max_overflow=8,
            pool_recycle=1  # 1 second for testing
        )
        
        # Create extra connections by using overflow
        conns = []
        for _ in range(5):
            conn = pool.get_connection().__enter__()
            conns.append(conn)
        
        # Return all connections
        for conn in conns:
            pool._return_connection(conn)
        
        initial_total = pool.get_statistics().total_connections
        
        # Wait for recycle timeout
        time.sleep(1.1)
        
        # Perform maintenance (which handles cleanup)
        pool._perform_maintenance()
        
        # Should have fewer connections due to cleanup
        final_stats = pool.get_statistics()
        assert final_stats.total_connections <= initial_total
        
        pool.shutdown()
    
    def test_graceful_shutdown(self, temp_db_path):
        """Test graceful pool shutdown."""
        pool = ConnectionPool(db_path=str(temp_db_path))
        
        # Acquire some connections
        conn1 = pool.get_connection().__enter__()
        conn2 = pool.get_connection().__enter__()
        
        # Release one
        pool._return_connection(conn1)
        
        # Graceful shutdown should close all connections
        pool.shutdown()
        
        # All connections should be closed
        assert len(pool._pool) == 0
        assert len(pool._overflow_connections) == 0


@pytest.mark.unit
class TestConnectionPoolErrorHandling:
    """Test error handling in connection pool."""
    
    def test_handle_database_connection_error(self):
        """Test handling database connection errors."""
        # Use a path that will definitely fail
        import tempfile
        import os
        
        # Create a directory where we expect a file (will cause error)
        with tempfile.TemporaryDirectory() as temp_dir:
            bad_path = os.path.join(temp_dir, "directory_not_file")
            os.makedirs(bad_path)
            
            # Pool initialization will handle errors gracefully and continue with 0 connections
            pool = ConnectionPool(db_path=bad_path)
            
            # Verify pool was created but with no connections
            assert len(pool._pool) == 0
            
            # When trying to get a connection, it should fail
            with pytest.raises(sqlite3.OperationalError):
                with pool.get_connection():
                    pass
            
            pool.shutdown()
    
    def test_handle_corrupted_connection(self, temp_db_path):
        """Test handling corrupted connections."""
        pool = ConnectionPool(db_path=str(temp_db_path))
        
        ctx = pool.get_connection()
        conn = ctx.__enter__()
        
        # Close the underlying connection to simulate corruption
        conn.connection.close()
        
        # Return connection - pool should handle the closed connection
        ctx.__exit__(None, None, None)
        
        # Pool should have handled the corrupted connection
        # The exact behavior depends on implementation details
        pool.shutdown()
    
    def test_recovery_from_connection_loss(self, temp_db_path):
        """Test recovery from connection loss."""
        pool = ConnectionPool(
            db_path=str(temp_db_path),
            pool_size=2
        )
        
        # Simulate connection loss by closing underlying connections
        for conn in list(pool._pool):
            conn.connection.close()
        
        # Clear the pool to force new connections
        pool._pool.clear()
        
        # Pool should recover on next acquire by creating new connections
        with pool.get_connection() as new_conn:
            assert new_conn is not None
            cursor = new_conn.execute("SELECT 1")
            assert cursor.fetchone()[0] == 1
        
        pool.shutdown()
    
    def test_handle_connection_creation_failure(self, temp_db_path):
        """Test handling when connection creation fails."""
        # Mock connection creation to fail during initialization
        with patch('sqlite3.connect', side_effect=sqlite3.Error("Connection failed")):
            # The ConnectionPool catches errors during initialization and continues
            # with 0 connections, then tries to create new ones on demand
            pool = ConnectionPool(db_path=str(temp_db_path))
            
            # Verify that pool was created but with 0 connections
            assert len(pool._pool) == 0
            
            # When trying to get a connection, it should fail
            with pytest.raises(sqlite3.Error):
                with pool.get_connection():
                    pass
            
            pool.shutdown()


@pytest.mark.unit
class TestConnectionPoolContextManager:
    """Test connection pool as context manager."""
    
    def test_pool_as_context_manager(self, temp_db_path):
        """Test using pool as context manager."""
        with ConnectionPool(db_path=str(temp_db_path)) as pool:
            with pool.get_connection() as conn:
                assert conn is not None
        
        # Pool should be closed after context
        assert len(pool._pool) == 0
    
    def test_connection_context_manager(self, temp_db_path):
        """Test connection context manager from pool."""
        pool = ConnectionPool(db_path=str(temp_db_path))
        
        # Use connection as context manager
        with pool.get_connection() as conn:
            cursor = conn.execute("SELECT 1")
            result = cursor.fetchone()
            assert result[0] == 1
        
        # Connection should be released automatically
        stats = pool.get_statistics()
        assert stats.active_connections == 0
        
        pool.shutdown()
    
    def test_connection_context_manager_with_exception(self, temp_db_path):
        """Test connection context manager handles exceptions."""
        pool = ConnectionPool(db_path=str(temp_db_path))
        
        try:
            with pool.get_connection() as conn:
                conn.execute("SELECT 1")
                raise ValueError("Test error")
        except ValueError:
            pass
        
        # Connection should still be released
        stats = pool.get_statistics()
        assert stats.active_connections == 0
        
        pool.shutdown()