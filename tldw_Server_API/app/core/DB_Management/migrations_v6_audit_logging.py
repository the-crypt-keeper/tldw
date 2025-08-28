"""
Database migration v6: Audit Logging Enhancement

Adds comprehensive audit logging tables and indexes for security monitoring
and compliance tracking in the Evaluations module.
"""

import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Optional
from loguru import logger


def apply_audit_logging_migration(db_path: str) -> bool:
    """
    Apply audit logging migration to database.
    
    Args:
        db_path: Path to database file
        
    Returns:
        True if migration applied successfully
    """
    try:
        with sqlite3.connect(db_path) as conn:
            cursor = conn.cursor()
            
            # Check if migration already applied
            cursor.execute("""
                SELECT name FROM sqlite_master 
                WHERE type='table' AND name='audit_events'
            """)
            
            if cursor.fetchone():
                logger.info("Audit logging migration already applied")
                return True
            
            logger.info("Applying audit logging migration v6...")
            
            # Create audit_events table with comprehensive structure
            cursor.execute("""
                CREATE TABLE audit_events (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    event_id TEXT UNIQUE NOT NULL,
                    timestamp TIMESTAMP NOT NULL,
                    event_type TEXT NOT NULL,
                    severity TEXT NOT NULL,
                    user_id TEXT,
                    session_id TEXT,
                    ip_address TEXT,
                    user_agent TEXT,
                    endpoint TEXT,
                    method TEXT,
                    resource_id TEXT,
                    resource_type TEXT,
                    action TEXT NOT NULL,
                    outcome TEXT NOT NULL,
                    details TEXT,
                    metadata TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # Create comprehensive indexes for efficient querying
            audit_indexes = [
                "CREATE INDEX idx_audit_timestamp ON audit_events(timestamp)",
                "CREATE INDEX idx_audit_event_type ON audit_events(event_type)",
                "CREATE INDEX idx_audit_user_id ON audit_events(user_id)",
                "CREATE INDEX idx_audit_severity ON audit_events(severity)",
                "CREATE INDEX idx_audit_outcome ON audit_events(outcome)",
                "CREATE INDEX idx_audit_ip ON audit_events(ip_address)",
                "CREATE INDEX idx_audit_resource ON audit_events(resource_type, resource_id)",
                "CREATE INDEX idx_audit_endpoint ON audit_events(endpoint)",
                "CREATE INDEX idx_audit_session ON audit_events(session_id)",
                "CREATE INDEX idx_audit_user_time ON audit_events(user_id, timestamp)",
                "CREATE INDEX idx_audit_security ON audit_events(severity, event_type, timestamp)"
            ]
            
            for index_sql in audit_indexes:
                cursor.execute(index_sql)
            
            # Create audit configuration table for retention policies
            cursor.execute("""
                CREATE TABLE audit_configuration (
                    id INTEGER PRIMARY KEY,
                    retention_days INTEGER DEFAULT 90,
                    log_level TEXT DEFAULT 'INFO',
                    enable_real_time_alerts BOOLEAN DEFAULT 1,
                    max_events_per_user_per_hour INTEGER DEFAULT 1000,
                    suspicious_ip_threshold INTEGER DEFAULT 100,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # Insert default configuration
            cursor.execute("""
                INSERT INTO audit_configuration (
                    retention_days, log_level, enable_real_time_alerts,
                    max_events_per_user_per_hour, suspicious_ip_threshold
                ) VALUES (90, 'INFO', 1, 1000, 100)
            """)
            
            # Create audit statistics view for reporting
            cursor.execute("""
                CREATE VIEW audit_statistics AS
                SELECT 
                    DATE(timestamp) as audit_date,
                    event_type,
                    severity,
                    outcome,
                    COUNT(*) as event_count,
                    COUNT(DISTINCT user_id) as unique_users,
                    COUNT(DISTINCT ip_address) as unique_ips
                FROM audit_events
                GROUP BY DATE(timestamp), event_type, severity, outcome
            """)
            
            # Create security alerts view for high-priority events
            cursor.execute("""
                CREATE VIEW security_alerts AS
                SELECT 
                    event_id,
                    timestamp,
                    event_type,
                    severity,
                    user_id,
                    ip_address,
                    action,
                    details,
                    CASE 
                        WHEN severity = 'critical' THEN 1
                        WHEN severity = 'high' THEN 2
                        WHEN severity = 'medium' THEN 3
                        ELSE 4
                    END as priority
                FROM audit_events
                WHERE severity IN ('critical', 'high')
                ORDER BY priority, timestamp DESC
            """)
            
            # Add audit logging trigger for user_rate_limits table changes
            cursor.execute("""
                CREATE TRIGGER audit_rate_limit_changes
                AFTER UPDATE ON user_rate_limits
                BEGIN
                    INSERT INTO audit_events (
                        event_id, timestamp, event_type, severity,
                        user_id, action, outcome, resource_type, resource_id,
                        details
                    ) VALUES (
                        lower(hex(randomblob(16))),
                        datetime('now'),
                        'config.tier_upgrade',
                        'medium',
                        NEW.user_id,
                        'User tier updated from ' || OLD.tier || ' to ' || NEW.tier,
                        'success',
                        'user_tier',
                        NEW.user_id,
                        json_object(
                            'old_tier', OLD.tier,
                            'new_tier', NEW.tier,
                            'old_daily_limit', OLD.evaluations_per_day,
                            'new_daily_limit', NEW.evaluations_per_day
                        )
                    );
                END
            """)
            
            # Create audit retention cleanup function (will be called by scheduled job)
            cursor.execute("""
                CREATE TABLE audit_cleanup_log (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    cleanup_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    records_deleted INTEGER,
                    retention_days INTEGER
                )
            """)
            
            # Update database version tracking
            cursor.execute("""
                INSERT OR REPLACE INTO migrations (
                    version, description, applied_at, checksum
                ) VALUES (
                    6,
                    'Audit logging system with comprehensive security monitoring',
                    datetime('now'),
                    'audit_logging_v6_' || hex(randomblob(8))
                )
            """)
            
            conn.commit()
            logger.info("Audit logging migration v6 applied successfully")
            return True
            
    except Exception as e:
        logger.error(f"Failed to apply audit logging migration: {e}")
        return False


def rollback_audit_logging_migration(db_path: str) -> bool:
    """
    Rollback audit logging migration.
    
    Args:
        db_path: Path to database file
        
    Returns:
        True if rollback successful
    """
    try:
        with sqlite3.connect(db_path) as conn:
            cursor = conn.cursor()
            
            logger.info("Rolling back audit logging migration v6...")
            
            # Drop audit-related tables and views
            audit_objects = [
                "DROP VIEW IF EXISTS security_alerts",
                "DROP VIEW IF EXISTS audit_statistics", 
                "DROP TRIGGER IF EXISTS audit_rate_limit_changes",
                "DROP TABLE IF EXISTS audit_cleanup_log",
                "DROP TABLE IF EXISTS audit_configuration",
                "DROP TABLE IF EXISTS audit_events"
            ]
            
            for sql in audit_objects:
                cursor.execute(sql)
            
            # Remove migration record
            cursor.execute("DELETE FROM migrations WHERE version = 6")
            
            conn.commit()
            logger.info("Audit logging migration v6 rolled back successfully")
            return True
            
    except Exception as e:
        logger.error(f"Failed to rollback audit logging migration: {e}")
        return False


def verify_audit_migration(db_path: str) -> bool:
    """
    Verify audit logging migration is properly applied.
    
    Args:
        db_path: Path to database file
        
    Returns:
        True if migration verified successfully
    """
    try:
        with sqlite3.connect(db_path) as conn:
            cursor = conn.cursor()
            
            # Check for required tables
            required_tables = ['audit_events', 'audit_configuration', 'audit_cleanup_log']
            for table in required_tables:
                cursor.execute("""
                    SELECT name FROM sqlite_master 
                    WHERE type='table' AND name=?
                """, (table,))
                
                if not cursor.fetchone():
                    logger.error(f"Missing required table: {table}")
                    return False
            
            # Check for required views
            required_views = ['audit_statistics', 'security_alerts']
            for view in required_views:
                cursor.execute("""
                    SELECT name FROM sqlite_master 
                    WHERE type='view' AND name=?
                """, (view,))
                
                if not cursor.fetchone():
                    logger.error(f"Missing required view: {view}")
                    return False
            
            # Check for required indexes
            cursor.execute("""
                SELECT name FROM sqlite_master 
                WHERE type='index' AND name LIKE 'idx_audit_%'
            """)
            
            indexes = cursor.fetchall()
            if len(indexes) < 10:  # Should have at least 10 audit indexes
                logger.error(f"Missing audit indexes. Found {len(indexes)}, expected at least 10")
                return False
            
            # Test basic functionality
            cursor.execute("""
                INSERT INTO audit_events (
                    event_id, timestamp, event_type, severity, action, outcome
                ) VALUES (
                    'test_' || hex(randomblob(8)),
                    datetime('now'),
                    'test.migration_verify',
                    'low',
                    'Testing audit migration',
                    'success'
                )
            """)
            
            # Clean up test record
            cursor.execute("DELETE FROM audit_events WHERE event_type = 'test.migration_verify'")
            
            conn.commit()
            logger.info("Audit logging migration verification passed")
            return True
            
    except Exception as e:
        logger.error(f"Audit migration verification failed: {e}")
        return False


# Integration with existing migration system
def migrate_audit_logging(db_path: str) -> bool:
    """
    Apply audit logging migration with verification.
    
    Args:
        db_path: Path to database file
        
    Returns:
        True if migration successful
    """
    # Ensure migrations table exists
    with sqlite3.connect(db_path) as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS migrations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                version INTEGER UNIQUE NOT NULL,
                description TEXT NOT NULL,
                applied_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                checksum TEXT
            )
        """)
        conn.commit()
    
    # Apply migration
    if apply_audit_logging_migration(db_path):
        # Verify migration
        if verify_audit_migration(db_path):
            return True
        else:
            # Rollback on verification failure
            logger.error("Migration verification failed, rolling back...")
            rollback_audit_logging_migration(db_path)
            return False
    
    return False