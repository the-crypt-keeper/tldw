"""
SQLite backend implementation for the scheduler.
Optimized for development and single-user deployments.
"""

import sqlite3
import json
import asyncio
import aiosqlite
from contextlib import asynccontextmanager
from typing import List, Optional, Dict, Any, AsyncContextManager
from datetime import datetime, timedelta
from pathlib import Path
import uuid

from loguru import logger

from ..base import (
    Task, TaskStatus, QueueBackend,
    DuplicateTaskError, TaskNotFoundError,
    ConnectionError, TransactionError
)
from ..config import SchedulerConfig


class SQLiteBackend(QueueBackend):
    """
    SQLite backend implementation with optimizations:
    - INSERT OR IGNORE for idempotency
    - 2-query dependency resolution (no N+1)
    - Write-ahead logging for concurrency
    - JSON text for compatibility
    """
    
    def __init__(self, config: SchedulerConfig):
        """
        Initialize SQLite backend.
        
        Args:
            config: Scheduler configuration
        """
        self.config = config
        self.db_path = self._extract_db_path(config.database_url)
        self._connection: Optional[aiosqlite.Connection] = None
        self._transaction_conn: Optional[aiosqlite.Connection] = None
        self._lock = asyncio.Lock()
        
        logger.info(f"Initializing SQLite backend with database: {self.db_path}")
    
    def _extract_db_path(self, url: str) -> str:
        """Extract database path from URL"""
        if url.startswith('sqlite:///'):
            return url[10:]  # Remove 'sqlite:///'
        elif url == ':memory:':
            return ':memory:'
        else:
            return url
    
    async def connect(self) -> None:
        """Initialize database connection and create schema"""
        if self._connection:
            return
        
        # Ensure directory exists for file-based database
        if self.db_path != ':memory:':
            Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        
        try:
            self._connection = await aiosqlite.connect(
                self.db_path,
                isolation_level=None  # Autocommit mode
            )
            
            # Enable optimizations
            await self._connection.execute("PRAGMA journal_mode=WAL")  # Write-ahead logging
            await self._connection.execute("PRAGMA synchronous=NORMAL")  # Balance safety/speed
            await self._connection.execute("PRAGMA cache_size=10000")  # Larger cache
            await self._connection.execute("PRAGMA temp_store=MEMORY")  # Temp tables in memory
            await self._connection.execute("PRAGMA busy_timeout=5000")  # 5 second timeout
            
            # Create schema
            await self.create_schema()
            
            logger.info("SQLite backend connected and initialized")
            
        except Exception as e:
            logger.error(f"Failed to connect to SQLite: {e}")
            raise ConnectionError(f"SQLite connection failed: {e}")
    
    async def disconnect(self) -> None:
        """Close database connection"""
        if self._connection:
            await self._connection.close()
            self._connection = None
            logger.info("SQLite backend disconnected")
    
    async def create_schema(self) -> None:
        """Create database schema (idempotent)"""
        schema = """
        -- Main tasks table
        CREATE TABLE IF NOT EXISTS tasks (
            id TEXT PRIMARY KEY,
            queue_name TEXT NOT NULL,
            handler TEXT NOT NULL,
            payload TEXT,  -- JSON text
            priority INTEGER DEFAULT 2,
            status TEXT NOT NULL DEFAULT 'queued',
            
            -- Scheduling
            scheduled_at TEXT,  -- ISO format timestamp
            expires_at TEXT,
            
            -- Execution control
            max_retries INTEGER DEFAULT 3,
            retry_count INTEGER DEFAULT 0,
            retry_delay INTEGER DEFAULT 60,
            timeout INTEGER DEFAULT 300,
            
            -- Dependencies as JSON array
            depends_on TEXT,
            idempotency_key TEXT,
            
            -- Timestamps
            created_at TEXT DEFAULT CURRENT_TIMESTAMP,
            queued_at TEXT,
            started_at TEXT,
            completed_at TEXT,
            
            -- Tracking
            worker_id TEXT,
            lease_id TEXT,
            execution_time REAL,  -- Seconds
            error TEXT,
            result TEXT,  -- JSON
            
            -- References for large payloads
            payload_ref TEXT,
            result_ref TEXT,
            
            CHECK (status IN ('pending', 'queued', 'running', 'completed', 'failed', 'cancelled', 'dead'))
        );
        
        -- Critical indexes for performance
        CREATE INDEX IF NOT EXISTS idx_dequeue 
            ON tasks(queue_name, status, priority DESC, created_at)
            WHERE status = 'queued';
        
        CREATE INDEX IF NOT EXISTS idx_scheduled 
            ON tasks(scheduled_at) 
            WHERE scheduled_at IS NOT NULL AND status = 'queued';
        
        -- CRITICAL: Unique index for idempotency with INSERT OR IGNORE
        CREATE UNIQUE INDEX IF NOT EXISTS idx_idempotency 
            ON tasks(idempotency_key) 
            WHERE idempotency_key IS NOT NULL;
        
        -- For efficient keyset pagination during migration
        CREATE INDEX IF NOT EXISTS idx_migration 
            ON tasks(id) 
            WHERE status IN ('queued', 'scheduled');
        
        -- Task leases with unique constraint
        CREATE TABLE IF NOT EXISTS task_leases (
            lease_id TEXT PRIMARY KEY,
            task_id TEXT NOT NULL,
            worker_id TEXT NOT NULL,
            acquired_at TEXT DEFAULT CURRENT_TIMESTAMP,
            expires_at TEXT NOT NULL,
            renewal_count INTEGER DEFAULT 0,
            
            -- Ensures only one lease per task
            UNIQUE(task_id),
            FOREIGN KEY (task_id) REFERENCES tasks(id) ON DELETE CASCADE
        );
        
        CREATE INDEX IF NOT EXISTS idx_lease_expiry 
            ON task_leases(expires_at);
        
        -- Service leaders with TTL support
        CREATE TABLE IF NOT EXISTS service_leaders (
            service_name TEXT PRIMARY KEY,
            leader_id TEXT NOT NULL,
            acquired_at TEXT DEFAULT CURRENT_TIMESTAMP
        );
        
        -- Index for efficient stale lock detection
        CREATE INDEX IF NOT EXISTS idx_leader_stale 
            ON service_leaders(acquired_at);
        
        -- Dead letter queue
        CREATE TABLE IF NOT EXISTS dead_letter_queue (
            id TEXT PRIMARY KEY,
            original_task_id TEXT NOT NULL,
            queue_name TEXT NOT NULL,
            handler TEXT NOT NULL,
            payload TEXT,
            error_count INTEGER DEFAULT 1,
            last_error TEXT,
            moved_at TEXT DEFAULT CURRENT_TIMESTAMP,
            original_created_at TEXT,
            metadata TEXT
        );
        
        -- Schema version for migrations
        CREATE TABLE IF NOT EXISTS schema_version (
            version INTEGER PRIMARY KEY,
            applied_at TEXT DEFAULT CURRENT_TIMESTAMP
        );
        
        -- Insert initial version if not exists
        INSERT OR IGNORE INTO schema_version (version) VALUES (1);
        """
        
        async with self._lock:
            await self._connection.executescript(schema)
            await self._connection.commit()
    
    async def enqueue(self, task: Task) -> str:
        """Add a task to the queue"""
        # Check idempotency
        if task.idempotency_key:
            existing = await self.fetchval(
                "SELECT id FROM tasks WHERE idempotency_key = ?",
                task.idempotency_key
            )
            if existing:
                logger.debug(f"Task with idempotency_key {task.idempotency_key} already exists")
                return existing
        
        # Insert task
        await self.execute("""
            INSERT INTO tasks (
                id, queue_name, handler, payload, priority, status,
                scheduled_at, expires_at, max_retries, retry_count,
                retry_delay, timeout, depends_on, idempotency_key,
                created_at, queued_at, payload_ref
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, 
            task.id,
            task.queue_name,
            task.handler,
            json.dumps(task.payload) if task.payload else None,
            task.priority,
            TaskStatus.QUEUED.value,
            task.scheduled_at.isoformat() if task.scheduled_at else None,
            task.expires_at.isoformat() if task.expires_at else None,
            task.max_retries,
            task.retry_count,
            task.retry_delay,
            task.timeout,
            json.dumps(task.depends_on) if task.depends_on else None,
            task.idempotency_key,
            datetime.utcnow().isoformat(),
            datetime.utcnow().isoformat(),
            task.payload_ref
        )
        
        return task.id
    
    async def bulk_enqueue(self, tasks: List[Task]) -> List[str]:
        """
        Efficiently enqueue multiple tasks using INSERT OR IGNORE.
        This avoids N+1 queries for idempotency checks.
        """
        if not tasks:
            return []
        
        # Prepare values for bulk insert
        values = []
        for task in tasks:
            values.append((
                task.id,
                task.queue_name,
                task.handler,
                json.dumps(task.payload) if task.payload else None,
                task.priority,
                TaskStatus.QUEUED.value,
                task.scheduled_at.isoformat() if task.scheduled_at else None,
                task.expires_at.isoformat() if task.expires_at else None,
                task.max_retries,
                task.retry_count,
                task.retry_delay,
                task.timeout,
                json.dumps(task.depends_on) if task.depends_on else None,
                task.idempotency_key,
                datetime.utcnow().isoformat(),
                datetime.utcnow().isoformat(),
                task.payload_ref
            ))
        
        # Single atomic bulk insert with conflict handling
        async with self._lock:
            cursor = await self._connection.executemany("""
                INSERT OR IGNORE INTO tasks (
                    id, queue_name, handler, payload, priority, status,
                    scheduled_at, expires_at, max_retries, retry_count,
                    retry_delay, timeout, depends_on, idempotency_key,
                    created_at, queued_at, payload_ref
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, values)
            
            await self._connection.commit()
            
            inserted_count = cursor.rowcount
            logger.info(f"Bulk inserted {inserted_count} tasks (duplicates ignored)")
            
            # Return IDs of tasks (SQLite doesn't tell us which were ignored)
            return [task.id for task in tasks]
    
    async def dequeue_atomic(self, queue_name: str, worker_id: str) -> Optional[Task]:
        """
        Atomically dequeue next available task.
        SQLite limitation: Single writer, so this is a bottleneck.
        """
        async with self._lock:
            # Begin immediate transaction for write lock
            await self._connection.execute("BEGIN IMMEDIATE")
            
            try:
                # Find next available task
                row = await self._connection.execute_fetchone("""
                    SELECT * FROM tasks
                    WHERE queue_name = ?
                      AND status = 'queued'
                      AND (scheduled_at IS NULL OR scheduled_at <= datetime('now'))
                      AND (expires_at IS NULL OR expires_at > datetime('now'))
                      AND (depends_on IS NULL OR depends_on = '[]')
                    ORDER BY priority ASC, created_at ASC
                    LIMIT 1
                """, (queue_name,))
                
                if not row:
                    await self._connection.rollback()
                    return None
                
                task = await self._row_to_task(row)
                
                # Create lease
                lease_id = str(uuid.uuid4())
                lease_duration = task.timeout or self.config.lease_duration_seconds
                expires_at = datetime.utcnow() + timedelta(seconds=lease_duration)
                
                await self._connection.execute("""
                    INSERT INTO task_leases (lease_id, task_id, worker_id, expires_at)
                    VALUES (?, ?, ?, ?)
                """, (lease_id, task.id, worker_id, expires_at.isoformat()))
                
                # Update task status
                await self._connection.execute("""
                    UPDATE tasks
                    SET status = 'running',
                        worker_id = ?,
                        lease_id = ?,
                        started_at = datetime('now')
                    WHERE id = ?
                """, (worker_id, lease_id, task.id))
                
                await self._connection.commit()
                
                task.status = TaskStatus.RUNNING
                task.worker_id = worker_id
                task.lease_id = lease_id
                task.started_at = datetime.utcnow()
                
                return task
                
            except Exception as e:
                await self._connection.rollback()
                raise TransactionError(f"Dequeue failed: {e}")
    
    async def get_ready_tasks(self, queue_name: Optional[str] = None) -> List[str]:
        """
        Efficient dependency resolution with only 2 queries.
        Avoids N+1 query pattern.
        """
        # Step 1: Get all queued tasks in a single query
        query = "SELECT id, depends_on FROM tasks WHERE status = 'queued'"
        params = []
        if queue_name:
            query += " AND queue_name = ?"
            params.append(queue_name)
        
        rows = await self.fetch(query, *params)
        
        ready = []
        tasks_with_deps = {}
        all_dependency_ids = set()
        
        # Process tasks and collect all dependency IDs
        for row in rows:
            task_id = row['id']
            deps = json.loads(row['depends_on']) if row['depends_on'] else []
            
            if not deps:
                # No dependencies, task is ready
                ready.append(task_id)
            else:
                # Has dependencies, need to check them
                tasks_with_deps[task_id] = set(deps)
                all_dependency_ids.update(deps)
        
        if not all_dependency_ids:
            # No dependencies to check
            return ready
        
        # Step 2: Get status of ALL dependencies in a SINGLE query
        placeholders = ','.join(['?' for _ in all_dependency_ids])
        completed_deps_query = f"""
            SELECT id FROM tasks 
            WHERE status = 'completed' 
            AND id IN ({placeholders})
        """
        completed_deps_rows = await self.fetch(
            completed_deps_query, 
            *list(all_dependency_ids)
        )
        completed_deps_set = {row['id'] for row in completed_deps_rows}
        
        # Step 3: Check dependencies in memory (fast!)
        for task_id, deps in tasks_with_deps.items():
            if deps.issubset(completed_deps_set):
                ready.append(task_id)
        
        return ready
    
    async def ack(self, task_id: str, result: Optional[Any] = None) -> bool:
        """Acknowledge task completion"""
        result_json = json.dumps(result) if result is not None else None
        
        affected = await self.execute("""
            UPDATE tasks
            SET status = 'completed',
                completed_at = datetime('now'),
                result = ?,
                execution_time = (julianday('now') - julianday(started_at)) * 86400
            WHERE id = ? AND status = 'running'
        """, result_json, task_id)
        
        if affected:
            # Delete lease
            await self.execute("DELETE FROM task_leases WHERE task_id = ?", task_id)
        
        return affected > 0
    
    async def nack(self, task_id: str, error: str, retry: bool = True) -> bool:
        """Handle task failure"""
        task = await self.get_task(task_id)
        if not task:
            return False
        
        if retry and task.retry_count < task.max_retries:
            # Schedule retry
            retry_delay = task.calculate_retry_delay()
            scheduled_at = datetime.utcnow() + timedelta(seconds=retry_delay)
            
            await self.execute("""
                UPDATE tasks
                SET status = 'queued',
                    retry_count = retry_count + 1,
                    scheduled_at = ?,
                    error = ?,
                    worker_id = NULL,
                    lease_id = NULL
                WHERE id = ?
            """, scheduled_at.isoformat(), error, task_id)
        else:
            # Move to failed or DLQ
            await self.execute("""
                UPDATE tasks
                SET status = 'failed',
                    completed_at = datetime('now'),
                    error = ?
                WHERE id = ?
            """, error, task_id)
        
        # Delete lease
        await self.execute("DELETE FROM task_leases WHERE task_id = ?", task_id)
        
        return True
    
    async def _row_to_task(self, row: sqlite3.Row) -> Task:
        """Convert database row to Task object"""
        task_dict = dict(row)
        
        # Parse JSON fields
        if task_dict.get('payload'):
            task_dict['payload'] = json.loads(task_dict['payload'])
        if task_dict.get('depends_on'):
            task_dict['depends_on'] = json.loads(task_dict['depends_on'])
        if task_dict.get('result'):
            task_dict['result'] = json.loads(task_dict['result'])
        
        # Convert status string to enum
        if task_dict.get('status'):
            task_dict['status'] = task_dict['status']  # Keep as string for from_dict
        
        return Task.from_dict(task_dict)
    
    # Implement remaining abstract methods...
    
    async def get_task(self, task_id: str) -> Optional[Task]:
        """Get task by ID"""
        row = await self.fetchrow("SELECT * FROM tasks WHERE id = ?", task_id)
        if row:
            return await self._row_to_task(row)
        return None
    
    async def update_task(self, task: Task) -> bool:
        """Update task"""
        affected = await self.execute("""
            UPDATE tasks
            SET status = ?, error = ?, result = ?, worker_id = ?, lease_id = ?
            WHERE id = ?
        """, 
            task.status.value,
            task.error,
            json.dumps(task.result) if task.result else None,
            task.worker_id,
            task.lease_id,
            task.id
        )
        return affected > 0
    
    async def get_queue_size(self, queue_name: str) -> int:
        """Get queue size"""
        return await self.fetchval(
            "SELECT COUNT(*) FROM tasks WHERE queue_name = ? AND status = 'queued'",
            queue_name
        ) or 0
    
    async def clear_queue(self, queue_name: str) -> int:
        """Clear queue"""
        affected = await self.execute(
            "DELETE FROM tasks WHERE queue_name = ? AND status = 'queued'",
            queue_name
        )
        return affected
    
    async def get_dead_letter_queue(self) -> List[Task]:
        """Get DLQ tasks"""
        rows = await self.fetch("SELECT * FROM dead_letter_queue")
        # Convert to tasks (simplified)
        return []  # TODO: Implement if needed
    
    async def move_to_dlq(self, task_id: str, reason: str) -> bool:
        """Move task to DLQ"""
        task = await self.get_task(task_id)
        if not task:
            return False
        
        await self.execute("""
            INSERT INTO dead_letter_queue (
                id, original_task_id, queue_name, handler, payload, last_error
            ) VALUES (?, ?, ?, ?, ?, ?)
        """,
            str(uuid.uuid4()),
            task.id,
            task.queue_name,
            task.handler,
            json.dumps(task.payload),
            reason
        )
        
        await self.execute("UPDATE tasks SET status = 'dead' WHERE id = ?", task_id)
        return True
    
    # Lease management
    
    async def create_lease(self, lease_id: str, task_id: str, 
                          worker_id: str, expires_at: datetime) -> bool:
        """Create lease"""
        try:
            await self.execute("""
                INSERT INTO task_leases (lease_id, task_id, worker_id, expires_at)
                VALUES (?, ?, ?, ?)
            """, lease_id, task_id, worker_id, expires_at.isoformat())
            return True
        except:
            return False
    
    async def renew_lease(self, lease_id: str, expires_at: datetime) -> bool:
        """Renew lease"""
        affected = await self.execute("""
            UPDATE task_leases
            SET expires_at = ?, renewal_count = renewal_count + 1
            WHERE lease_id = ?
        """, expires_at.isoformat(), lease_id)
        return affected > 0
    
    async def delete_lease(self, lease_id: str) -> bool:
        """Delete lease"""
        affected = await self.execute(
            "DELETE FROM task_leases WHERE lease_id = ?",
            lease_id
        )
        return affected > 0
    
    async def get_expired_leases(self) -> List[Dict[str, Any]]:
        """Get expired leases"""
        return await self.fetch(
            "SELECT * FROM task_leases WHERE expires_at < datetime('now')"
        )
    
    # Transaction support
    
    @asynccontextmanager
    async def transaction(self):
        """Transaction context manager"""
        async with self._lock:
            await self._connection.execute("BEGIN")
            try:
                yield self
                await self._connection.commit()
            except Exception:
                await self._connection.rollback()
                raise
    
    # Utility methods
    
    async def execute(self, query: str, *args) -> int:
        """Execute query and return affected rows"""
        async with self._lock:
            cursor = await self._connection.execute(query, args)
            await self._connection.commit()
            return cursor.rowcount
    
    async def fetch(self, query: str, *args) -> List[Dict[str, Any]]:
        """Fetch multiple rows"""
        async with self._lock:
            cursor = await self._connection.execute(query, args)
            rows = await cursor.fetchall()
            if rows:
                # Convert to dictionaries
                columns = [desc[0] for desc in cursor.description]
                return [dict(zip(columns, row)) for row in rows]
            return []
    
    async def fetchval(self, query: str, *args) -> Any:
        """Fetch single value"""
        async with self._lock:
            cursor = await self._connection.execute(query, args)
            row = await cursor.fetchone()
            return row[0] if row else None
    
    async def fetchrow(self, query: str, *args) -> Optional[Dict[str, Any]]:
        """Fetch single row"""
        rows = await self.fetch(query, *args)
        return rows[0] if rows else None
    
    # Schema management
    
    async def get_schema_version(self) -> int:
        """Get schema version"""
        return await self.fetchval("SELECT MAX(version) FROM schema_version") or 0
    
    async def migrate_schema(self, target_version: int) -> None:
        """Migrate schema (placeholder)"""
        current = await self.get_schema_version()
        if current < target_version:
            # TODO: Implement migrations
            pass