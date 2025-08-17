"""
Database migration system for handling schema changes safely.

This module provides a simple migration framework to handle database schema changes
without losing data or causing runtime detection issues.
"""

import sqlite3
from pathlib import Path
from typing import List, Dict, Any, Optional, Callable
from datetime import datetime
from loguru import logger


class Migration:
    """Represents a single database migration."""
    
    def __init__(self, version: int, name: str, up: Callable, down: Optional[Callable] = None):
        """
        Initialize a migration.
        
        Args:
            version: Migration version number
            name: Descriptive name for the migration
            up: Function to apply the migration
            down: Optional function to rollback the migration
        """
        self.version = version
        self.name = name
        self.up = up
        self.down = down
    
    def apply(self, conn: sqlite3.Connection) -> None:
        """Apply this migration."""
        logger.info(f"Applying migration {self.version}: {self.name}")
        self.up(conn)
    
    def rollback(self, conn: sqlite3.Connection) -> None:
        """Rollback this migration."""
        if self.down:
            logger.info(f"Rolling back migration {self.version}: {self.name}")
            self.down(conn)
        else:
            logger.warning(f"No rollback defined for migration {self.version}: {self.name}")


class MigrationManager:
    """Manages database migrations."""
    
    def __init__(self, db_path: Path):
        """
        Initialize migration manager.
        
        Args:
            db_path: Path to the database file
        """
        self.db_path = db_path
        self.migrations: List[Migration] = []
        self._init_migration_table()
    
    def _init_migration_table(self) -> None:
        """Create migration tracking table if it doesn't exist."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS schema_migrations (
                    version INTEGER PRIMARY KEY,
                    name TEXT NOT NULL,
                    applied_at TIMESTAMP NOT NULL
                )
            """)
            conn.commit()
    
    def add_migration(self, migration: Migration) -> None:
        """Add a migration to the manager."""
        self.migrations.append(migration)
        # Keep migrations sorted by version
        self.migrations.sort(key=lambda m: m.version)
    
    def get_current_version(self) -> int:
        """Get the current schema version."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT MAX(version) FROM schema_migrations
            """)
            result = cursor.fetchone()
            return result[0] if result and result[0] else 0
    
    def get_pending_migrations(self) -> List[Migration]:
        """Get list of migrations that haven't been applied yet."""
        current_version = self.get_current_version()
        return [m for m in self.migrations if m.version > current_version]
    
    def migrate(self, target_version: Optional[int] = None) -> None:
        """
        Apply migrations up to target version.
        
        Args:
            target_version: Version to migrate to (None = latest)
        """
        current_version = self.get_current_version()
        
        if target_version is None:
            target_version = max([m.version for m in self.migrations]) if self.migrations else 0
        
        if current_version >= target_version:
            logger.info(f"Database already at version {current_version}, no migrations needed")
            return
        
        with sqlite3.connect(self.db_path) as conn:
            # Apply migrations in order
            for migration in self.migrations:
                if current_version < migration.version <= target_version:
                    try:
                        # Begin transaction for each migration
                        conn.execute("BEGIN")
                        
                        # Apply migration
                        migration.apply(conn)
                        
                        # Record migration
                        conn.execute("""
                            INSERT INTO schema_migrations (version, name, applied_at)
                            VALUES (?, ?, ?)
                        """, (migration.version, migration.name, datetime.utcnow()))
                        
                        # Commit transaction
                        conn.commit()
                        logger.info(f"Successfully applied migration {migration.version}")
                        
                    except Exception as e:
                        # Rollback on error
                        conn.rollback()
                        logger.error(f"Failed to apply migration {migration.version}: {e}")
                        raise
        
        logger.info(f"Database migrated to version {target_version}")
    
    def rollback(self, target_version: int = 0) -> None:
        """
        Rollback migrations to target version.
        
        Args:
            target_version: Version to rollback to (0 = initial state)
        """
        current_version = self.get_current_version()
        
        if current_version <= target_version:
            logger.info(f"Database already at version {current_version}, no rollback needed")
            return
        
        with sqlite3.connect(self.db_path) as conn:
            # Rollback migrations in reverse order
            for migration in reversed(self.migrations):
                if target_version < migration.version <= current_version:
                    try:
                        # Begin transaction for each rollback
                        conn.execute("BEGIN")
                        
                        # Rollback migration
                        migration.rollback(conn)
                        
                        # Remove migration record
                        conn.execute("""
                            DELETE FROM schema_migrations WHERE version = ?
                        """, (migration.version,))
                        
                        # Commit transaction
                        conn.commit()
                        logger.info(f"Successfully rolled back migration {migration.version}")
                        
                    except Exception as e:
                        # Rollback on error
                        conn.rollback()
                        logger.error(f"Failed to rollback migration {migration.version}: {e}")
                        raise
        
        logger.info(f"Database rolled back to version {target_version}")


# ============= Evaluation Database Migrations =============

def create_evaluations_migrations(db_path: Optional[Path] = None) -> MigrationManager:
    """Create migrations for the evaluations database.
    
    Args:
        db_path: Optional path to database (uses default if not provided)
    """
    if db_path is None:
        db_path = Path("Databases/evaluations.db")
    
    def migration_001_initial_schema(conn: sqlite3.Connection):
        """Create initial evaluation tables."""
        # Create internal_evaluations table (separate from OpenAI evaluations)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS internal_evaluations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                evaluation_id TEXT UNIQUE NOT NULL,
                evaluation_type TEXT NOT NULL,
                created_at TIMESTAMP NOT NULL,
                input_data TEXT NOT NULL,
                results TEXT NOT NULL,
                metadata TEXT
            )
        """)
        
        # Create indexes
        conn.execute("CREATE INDEX IF NOT EXISTS idx_type ON internal_evaluations(evaluation_type)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_created ON internal_evaluations(created_at)")
        
        # Create metrics table
        conn.execute("""
            CREATE TABLE IF NOT EXISTS evaluation_metrics (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                evaluation_id TEXT NOT NULL,
                metric_name TEXT NOT NULL,
                score REAL NOT NULL,
                created_at TIMESTAMP NOT NULL,
                FOREIGN KEY (evaluation_id) REFERENCES internal_evaluations(evaluation_id)
            )
        """)
        
        # Create indexes for metrics table
        conn.execute("CREATE INDEX IF NOT EXISTS idx_eval_id ON evaluation_metrics(evaluation_id)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_metric ON evaluation_metrics(metric_name)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_metric_created ON evaluation_metrics(created_at)")
    
    def migration_002_add_user_support(conn: sqlite3.Connection):
        """Add user_id column for multi-user support."""
        # Check if column already exists
        cursor = conn.cursor()
        cursor.execute("PRAGMA table_info(internal_evaluations)")
        columns = [col[1] for col in cursor.fetchall()]
        
        if 'user_id' not in columns:
            conn.execute("ALTER TABLE internal_evaluations ADD COLUMN user_id TEXT")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_user_id ON internal_evaluations(user_id)")
    
    def migration_003_add_status_tracking(conn: sqlite3.Connection):
        """Add status and error tracking columns."""
        cursor = conn.cursor()
        cursor.execute("PRAGMA table_info(internal_evaluations)")
        columns = [col[1] for col in cursor.fetchall()]
        
        if 'status' not in columns:
            conn.execute("ALTER TABLE internal_evaluations ADD COLUMN status TEXT DEFAULT 'completed'")
        
        if 'error_message' not in columns:
            conn.execute("ALTER TABLE internal_evaluations ADD COLUMN error_message TEXT")
        
        if 'completed_at' not in columns:
            conn.execute("ALTER TABLE internal_evaluations ADD COLUMN completed_at TIMESTAMP")
        
        # Add index for status
        conn.execute("CREATE INDEX IF NOT EXISTS idx_status ON internal_evaluations(status)")
    
    def migration_004_add_embeddings_config(conn: sqlite3.Connection):
        """Add embeddings configuration to evaluations."""
        cursor = conn.cursor()
        cursor.execute("PRAGMA table_info(internal_evaluations)")
        columns = [col[1] for col in cursor.fetchall()]
        
        if 'embedding_provider' not in columns:
            conn.execute("ALTER TABLE internal_evaluations ADD COLUMN embedding_provider TEXT")
        
        if 'embedding_model' not in columns:
            conn.execute("ALTER TABLE internal_evaluations ADD COLUMN embedding_model TEXT")
    
    # Create manager and add migrations
    manager = MigrationManager(db_path)
    
    manager.add_migration(Migration(
        1, "initial_schema", 
        migration_001_initial_schema
    ))
    
    manager.add_migration(Migration(
        2, "add_user_support",
        migration_002_add_user_support
    ))
    
    manager.add_migration(Migration(
        3, "add_status_tracking",
        migration_003_add_status_tracking
    ))
    
    manager.add_migration(Migration(
        4, "add_embeddings_config",
        migration_004_add_embeddings_config
    ))
    
    return manager


def migrate_evaluations_database(db_path: Optional[Path] = None) -> None:
    """
    Apply all pending migrations to the evaluations database.
    
    Args:
        db_path: Optional path to database (uses default if not provided)
    """
    if db_path is None:
        db_path = Path("Databases/evaluations.db")
    
    # Ensure directory exists
    db_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Create migration manager with the specific path
    manager = MigrationManager(db_path)
    
    # Add all migrations
    migrations = create_evaluations_migrations(db_path)
    for migration in migrations.migrations:
        manager.add_migration(migration)
    
    # Apply pending migrations
    manager.migrate()
    
    logger.info(f"Evaluations database at version {manager.get_current_version()}")


if __name__ == "__main__":
    # Example usage
    migrate_evaluations_database()