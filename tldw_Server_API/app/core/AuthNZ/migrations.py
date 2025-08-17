# migrations.py
# Description: Database migrations for AuthNZ module tables
#
# Imports
from typing import List
import sqlite3
from pathlib import Path
#
# 3rd-party imports
from loguru import logger
#
# Local imports
from tldw_Server_API.app.core.DB_Management.migrations import Migration, MigrationManager

#######################################################################################################################
#
# AuthNZ Migrations
#

def migration_001_create_users_table(conn: sqlite3.Connection) -> None:
    """Create the users table for authentication"""
    conn.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            email TEXT UNIQUE NOT NULL,
            password_hash TEXT NOT NULL,
            is_active INTEGER DEFAULT 1,
            is_superuser INTEGER DEFAULT 0,
            role TEXT DEFAULT 'user',
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            last_login TIMESTAMP,
            email_verified INTEGER DEFAULT 0,
            storage_quota_mb INTEGER DEFAULT 5120,
            storage_used_mb INTEGER DEFAULT 0
        )
    """)
    
    # Create indexes
    conn.execute("CREATE INDEX IF NOT EXISTS idx_users_username ON users(username)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_users_email ON users(email)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_users_role ON users(role)")
    
    conn.commit()
    logger.info("Migration 001: Created users table")


def migration_002_create_sessions_table(conn: sqlite3.Connection) -> None:
    """Create the sessions table for session management"""
    conn.execute("""
        CREATE TABLE IF NOT EXISTS sessions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL,
            token_hash TEXT NOT NULL,
            refresh_token_hash TEXT,
            encrypted_token TEXT,
            encrypted_refresh TEXT,
            expires_at TIMESTAMP NOT NULL,
            ip_address TEXT,
            user_agent TEXT,
            device_id TEXT,
            is_revoked INTEGER DEFAULT 0,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
        )
    """)
    
    # Create indexes
    conn.execute("CREATE INDEX IF NOT EXISTS idx_sessions_token_hash ON sessions(token_hash)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_sessions_user_id ON sessions(user_id)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_sessions_expires_at ON sessions(expires_at)")
    
    conn.commit()
    logger.info("Migration 002: Created sessions table")


def migration_003_create_api_keys_table(conn: sqlite3.Connection) -> None:
    """Create the api_keys table for API key management"""
    conn.execute("""
        CREATE TABLE IF NOT EXISTS api_keys (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL,
            key_hash TEXT UNIQUE NOT NULL,
            key_prefix TEXT NOT NULL,
            name TEXT,
            description TEXT,
            scope TEXT DEFAULT 'read',
            status TEXT DEFAULT 'active',
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            expires_at TIMESTAMP,
            last_used_at TIMESTAMP,
            last_used_ip TEXT,
            usage_count INTEGER DEFAULT 0,
            rate_limit INTEGER,
            allowed_ips TEXT,
            metadata TEXT,
            rotated_from INTEGER REFERENCES api_keys(id),
            rotated_to INTEGER REFERENCES api_keys(id),
            revoked_at TIMESTAMP,
            revoked_by INTEGER,
            revoke_reason TEXT,
            FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
        )
    """)
    
    # Create indexes
    conn.execute("CREATE INDEX IF NOT EXISTS idx_api_keys_user_id ON api_keys(user_id)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_api_keys_key_hash ON api_keys(key_hash)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_api_keys_status ON api_keys(status)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_api_keys_expires_at ON api_keys(expires_at)")
    
    conn.commit()
    logger.info("Migration 003: Created api_keys table")


def migration_004_create_api_key_audit_log(conn: sqlite3.Connection) -> None:
    """Create the api_key_audit_log table for tracking API key actions"""
    conn.execute("""
        CREATE TABLE IF NOT EXISTS api_key_audit_log (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            api_key_id INTEGER NOT NULL,
            action TEXT NOT NULL,
            user_id INTEGER,
            ip_address TEXT,
            user_agent TEXT,
            details TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (api_key_id) REFERENCES api_keys(id) ON DELETE CASCADE
        )
    """)
    
    # Create index
    conn.execute("CREATE INDEX IF NOT EXISTS idx_api_key_audit_log_api_key_id ON api_key_audit_log(api_key_id)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_api_key_audit_log_created_at ON api_key_audit_log(created_at)")
    
    conn.commit()
    logger.info("Migration 004: Created api_key_audit_log table")


def migration_005_create_rate_limits_table(conn: sqlite3.Connection) -> None:
    """Create the rate_limits table for rate limiting"""
    conn.execute("""
        CREATE TABLE IF NOT EXISTS rate_limits (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            identifier TEXT NOT NULL,
            endpoint TEXT NOT NULL,
            request_count INTEGER DEFAULT 1,
            window_start TIMESTAMP NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            UNIQUE(identifier, endpoint, window_start)
        )
    """)
    
    # Create indexes
    conn.execute("CREATE INDEX IF NOT EXISTS idx_rate_limits_identifier ON rate_limits(identifier)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_rate_limits_window_start ON rate_limits(window_start)")
    
    conn.commit()
    logger.info("Migration 005: Created rate_limits table")


def migration_006_create_registration_codes_table(conn: sqlite3.Connection) -> None:
    """Create the registration_codes table for controlled registration"""
    conn.execute("""
        CREATE TABLE IF NOT EXISTS registration_codes (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            code TEXT UNIQUE NOT NULL,
            created_by INTEGER,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            expires_at TIMESTAMP,
            max_uses INTEGER DEFAULT 1,
            uses_count INTEGER DEFAULT 0,
            is_active INTEGER DEFAULT 1,
            metadata TEXT,
            FOREIGN KEY (created_by) REFERENCES users(id)
        )
    """)
    
    # Create indexes
    conn.execute("CREATE INDEX IF NOT EXISTS idx_registration_codes_code ON registration_codes(code)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_registration_codes_expires_at ON registration_codes(expires_at)")
    
    conn.commit()
    logger.info("Migration 006: Created registration_codes table")


def migration_007_create_audit_logs_table(conn: sqlite3.Connection) -> None:
    """Create the audit_logs table for security auditing"""
    conn.execute("""
        CREATE TABLE IF NOT EXISTS audit_logs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER,
            action TEXT NOT NULL,
            resource_type TEXT,
            resource_id INTEGER,
            ip_address TEXT,
            user_agent TEXT,
            status TEXT,
            details TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (user_id) REFERENCES users(id)
        )
    """)
    
    # Create indexes
    conn.execute("CREATE INDEX IF NOT EXISTS idx_audit_logs_user_id ON audit_logs(user_id)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_audit_logs_action ON audit_logs(action)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_audit_logs_created_at ON audit_logs(created_at)")
    
    conn.commit()
    logger.info("Migration 007: Created audit_logs table")


def migration_008_add_password_history_table(conn: sqlite3.Connection) -> None:
    """Create the password_history table to prevent password reuse"""
    conn.execute("""
        CREATE TABLE IF NOT EXISTS password_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL,
            password_hash TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
        )
    """)
    
    # Create indexes
    conn.execute("CREATE INDEX IF NOT EXISTS idx_password_history_user_id ON password_history(user_id)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_password_history_created_at ON password_history(created_at)")
    
    conn.commit()
    logger.info("Migration 008: Created password_history table")


def migration_009_add_session_encryption_columns(conn: sqlite3.Connection) -> None:
    """Add encryption columns to sessions table if they don't exist"""
    # Check if columns already exist
    cursor = conn.execute("PRAGMA table_info(sessions)")
    columns = [row[1] for row in cursor.fetchall()]
    
    if 'encrypted_token' not in columns:
        conn.execute("ALTER TABLE sessions ADD COLUMN encrypted_token TEXT")
        logger.info("Added encrypted_token column to sessions table")
    
    if 'encrypted_refresh' not in columns:
        conn.execute("ALTER TABLE sessions ADD COLUMN encrypted_refresh TEXT")
        logger.info("Added encrypted_refresh column to sessions table")
    
    conn.commit()
    logger.info("Migration 009: Added session encryption columns")


def migration_010_add_2fa_columns(conn: sqlite3.Connection) -> None:
    """Add two-factor authentication columns to users table"""
    # Check if columns already exist
    cursor = conn.execute("PRAGMA table_info(users)")
    columns = [row[1] for row in cursor.fetchall()]
    
    if 'totp_secret' not in columns:
        conn.execute("ALTER TABLE users ADD COLUMN totp_secret TEXT")
        logger.info("Added totp_secret column to users table")
    
    if 'two_factor_enabled' not in columns:
        conn.execute("ALTER TABLE users ADD COLUMN two_factor_enabled INTEGER DEFAULT 0")
        logger.info("Added two_factor_enabled column to users table")
    
    if 'backup_codes' not in columns:
        conn.execute("ALTER TABLE users ADD COLUMN backup_codes TEXT")
        logger.info("Added backup_codes column to users table")
    
    conn.commit()
    logger.info("Migration 010: Added 2FA columns to users table")


# Rollback functions (for migrations that can be rolled back)

def rollback_001_drop_users_table(conn: sqlite3.Connection) -> None:
    """Rollback: Drop the users table"""
    conn.execute("DROP TABLE IF EXISTS users")
    conn.commit()
    logger.info("Rollback 001: Dropped users table")


def rollback_002_drop_sessions_table(conn: sqlite3.Connection) -> None:
    """Rollback: Drop the sessions table"""
    conn.execute("DROP TABLE IF EXISTS sessions")
    conn.commit()
    logger.info("Rollback 002: Dropped sessions table")


def rollback_003_drop_api_keys_table(conn: sqlite3.Connection) -> None:
    """Rollback: Drop the api_keys table"""
    conn.execute("DROP TABLE IF EXISTS api_keys")
    conn.commit()
    logger.info("Rollback 003: Dropped api_keys table")


#######################################################################################################################
#
# Migration Registry
#

def get_authnz_migrations() -> List[Migration]:
    """Get all AuthNZ migrations in order"""
    return [
        Migration(1, "Create users table", migration_001_create_users_table, rollback_001_drop_users_table),
        Migration(2, "Create sessions table", migration_002_create_sessions_table, rollback_002_drop_sessions_table),
        Migration(3, "Create api_keys table", migration_003_create_api_keys_table, rollback_003_drop_api_keys_table),
        Migration(4, "Create api_key_audit_log table", migration_004_create_api_key_audit_log),
        Migration(5, "Create rate_limits table", migration_005_create_rate_limits_table),
        Migration(6, "Create registration_codes table", migration_006_create_registration_codes_table),
        Migration(7, "Create audit_logs table", migration_007_create_audit_logs_table),
        Migration(8, "Add password_history table", migration_008_add_password_history_table),
        Migration(9, "Add session encryption columns", migration_009_add_session_encryption_columns),
        Migration(10, "Add 2FA columns to users", migration_010_add_2fa_columns),
    ]


def apply_authnz_migrations(db_path: Path, target_version: int = None) -> None:
    """
    Apply AuthNZ migrations to a database
    
    Args:
        db_path: Path to the database file
        target_version: Target migration version (None = latest)
    """
    manager = MigrationManager(db_path)
    
    # Add all migrations to the manager
    for migration in get_authnz_migrations():
        manager.add_migration(migration)
    
    # Apply migrations
    manager.migrate(target_version)
    
    logger.info(f"Applied AuthNZ migrations to {db_path}")


def rollback_authnz_migrations(db_path: Path, target_version: int = 0) -> None:
    """
    Rollback AuthNZ migrations
    
    Args:
        db_path: Path to the database file
        target_version: Target migration version to rollback to
    """
    manager = MigrationManager(db_path)
    
    # Add all migrations to the manager
    for migration in get_authnz_migrations():
        manager.add_migration(migration)
    
    # Rollback migrations
    manager.rollback(target_version)
    
    logger.info(f"Rolled back AuthNZ migrations to version {target_version}")


#######################################################################################################################
#
# Utility Functions
#

def check_migration_status(db_path: Path) -> dict:
    """
    Check the migration status of a database
    
    Args:
        db_path: Path to the database file
        
    Returns:
        Dictionary with migration status information
    """
    manager = MigrationManager(db_path)
    
    # Add all migrations
    for migration in get_authnz_migrations():
        manager.add_migration(migration)
    
    current_version = manager.get_current_version()
    pending = manager.get_pending_migrations()
    
    return {
        "current_version": current_version,
        "latest_version": len(get_authnz_migrations()),
        "pending_migrations": [{"version": m.version, "name": m.name} for m in pending],
        "is_up_to_date": len(pending) == 0
    }


def ensure_authnz_tables(db_path: Path) -> None:
    """
    Ensure all AuthNZ tables exist in the database
    
    Args:
        db_path: Path to the database file
    """
    # Check current status
    status = check_migration_status(db_path)
    
    if not status["is_up_to_date"]:
        logger.info(f"Database needs migrations. Current: {status['current_version']}, Latest: {status['latest_version']}")
        apply_authnz_migrations(db_path)
    else:
        logger.debug("AuthNZ tables are up to date")


#
# End of migrations.py
#######################################################################################################################