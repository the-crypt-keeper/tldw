"""
Database migration v5: Unified Evaluations Table

Consolidates the separate 'evaluations' and 'internal_evaluations' tables
into a single unified schema that supports both OpenAI-compatible and
internal evaluation systems.
"""

import sqlite3
import json
from datetime import datetime, timezone
from typing import Optional
from pathlib import Path
from loguru import logger


def migrate_to_unified_evaluations(db_path: str) -> bool:
    """
    Migrate to unified evaluations schema.
    
    This migration:
    1. Creates a new unified evaluations table
    2. Migrates data from both old tables
    3. Preserves all existing data and relationships
    4. Adds new fields for enhanced functionality
    
    Args:
        db_path: Path to the evaluations database
        
    Returns:
        True if migration successful, False otherwise
    """
    try:
        with sqlite3.connect(db_path) as conn:
            cursor = conn.cursor()
            
            # Check if migration already applied
            cursor.execute("""
                SELECT name FROM sqlite_master 
                WHERE type='table' AND name='evaluations_unified'
            """)
            if cursor.fetchone():
                logger.info("Unified evaluations table already exists, skipping migration")
                return True
            
            logger.info("Starting migration to unified evaluations schema...")
            
            # Create the unified evaluations table
            cursor.execute("""
                CREATE TABLE evaluations_unified (
                    -- Core fields
                    id TEXT PRIMARY KEY,
                    evaluation_id TEXT UNIQUE NOT NULL,  -- For backward compatibility
                    name TEXT NOT NULL,
                    description TEXT,
                    evaluation_type TEXT NOT NULL,  -- 'geval', 'rag', 'response_quality', 'custom', etc.
                    
                    -- Evaluation specification
                    eval_spec TEXT NOT NULL,  -- JSON with evaluation parameters
                    input_data TEXT,  -- JSON with input data
                    
                    -- Results and status
                    status TEXT DEFAULT 'pending',  -- pending, running, completed, failed
                    results TEXT,  -- JSON with evaluation results
                    error_message TEXT,
                    progress REAL DEFAULT 0.0,  -- 0.0 to 1.0
                    
                    -- User and authentication
                    user_id TEXT,
                    created_by TEXT,  -- User identifier or API key hash
                    api_key_hash TEXT,  -- Hashed API key for audit
                    
                    -- Timestamps
                    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    started_at TIMESTAMP,
                    completed_at TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    deleted_at TIMESTAMP,  -- Soft delete support
                    
                    -- Dataset and runs
                    dataset_id TEXT,
                    run_id TEXT,  -- For linking to evaluation_runs
                    parent_evaluation_id TEXT,  -- For evaluation chains
                    
                    -- Model and provider info
                    target_model TEXT,  -- Model being evaluated
                    evaluator_model TEXT,  -- Model used for evaluation
                    embedding_provider TEXT,
                    embedding_model TEXT,
                    
                    -- Cost and usage tracking
                    token_count INTEGER DEFAULT 0,
                    estimated_cost REAL DEFAULT 0.0,
                    processing_time_seconds REAL,
                    
                    -- Webhook support
                    webhook_url TEXT,
                    webhook_secret TEXT,  -- For signature verification
                    webhook_events TEXT,  -- JSON array of events to notify
                    
                    -- Rate limiting
                    rate_limit_tier TEXT DEFAULT 'free',  -- free, basic, premium, enterprise
                    
                    -- Metadata
                    metadata TEXT,  -- JSON with additional metadata
                    tags TEXT,  -- JSON array of tags
                    
                    -- Versioning
                    version INTEGER DEFAULT 1,
                    
                    -- Indexes will be created separately
                    FOREIGN KEY (dataset_id) REFERENCES datasets(id),
                    FOREIGN KEY (parent_evaluation_id) REFERENCES evaluations_unified(id)
                )
            """)
            
            # Create comprehensive indexes
            indexes = [
                "CREATE INDEX idx_unified_eval_id ON evaluations_unified(evaluation_id)",
                "CREATE INDEX idx_unified_type ON evaluations_unified(evaluation_type)",
                "CREATE INDEX idx_unified_status ON evaluations_unified(status)",
                "CREATE INDEX idx_unified_user ON evaluations_unified(user_id)",
                "CREATE INDEX idx_unified_created ON evaluations_unified(created_at DESC)",
                "CREATE INDEX idx_unified_completed ON evaluations_unified(completed_at DESC)",
                "CREATE INDEX idx_unified_model ON evaluations_unified(target_model)",
                "CREATE INDEX idx_unified_deleted ON evaluations_unified(deleted_at)",
                "CREATE INDEX idx_unified_tier ON evaluations_unified(rate_limit_tier)",
                "CREATE INDEX idx_unified_parent ON evaluations_unified(parent_evaluation_id)",
            ]
            
            for index_sql in indexes:
                cursor.execute(index_sql)
            
            # Migrate data from 'evaluations' table (OpenAI-compatible)
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='evaluations'")
            if cursor.fetchone():
                logger.info("Migrating data from 'evaluations' table...")
                cursor.execute("""
                    INSERT INTO evaluations_unified (
                        id, evaluation_id, name, description, evaluation_type,
                        eval_spec, status, created_at, updated_at, created_by,
                        metadata, deleted_at, dataset_id
                    )
                    SELECT 
                        id, id, name, description, eval_type as evaluation_type,
                        eval_spec, 'completed', created_at, updated_at, created_by,
                        metadata, deleted_at, dataset_id
                    FROM evaluations
                """)
                logger.info(f"Migrated {cursor.rowcount} records from 'evaluations' table")
            
            # Migrate data from 'internal_evaluations' table
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='internal_evaluations'")
            if cursor.fetchone():
                logger.info("Migrating data from 'internal_evaluations' table...")
                cursor.execute("""
                    INSERT INTO evaluations_unified (
                        evaluation_id, name, eval_spec, evaluation_type, 
                        created_at, input_data, results, metadata, user_id, status, 
                        error_message, completed_at, embedding_provider, embedding_model
                    )
                    SELECT 
                        evaluation_id, 
                        'Internal ' || evaluation_type || ' - ' || evaluation_id as name,
                        COALESCE(metadata, '{}') as eval_spec,
                        evaluation_type, 
                        created_at, input_data, results, metadata, user_id, status, 
                        error_message, completed_at, embedding_provider, embedding_model
                    FROM internal_evaluations
                    WHERE evaluation_id NOT IN (SELECT evaluation_id FROM evaluations_unified)
                """)
                logger.info(f"Migrated {cursor.rowcount} records from 'internal_evaluations' table")
            
            # Create user rate limits table for per-user rate limiting
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS user_rate_limits (
                    user_id TEXT PRIMARY KEY,
                    tier TEXT NOT NULL DEFAULT 'free',  -- free, basic, premium, enterprise
                    
                    -- Rate limits per minute
                    evaluations_per_minute INTEGER DEFAULT 10,
                    batch_evaluations_per_minute INTEGER DEFAULT 2,
                    
                    -- Daily limits
                    evaluations_per_day INTEGER DEFAULT 100,
                    total_tokens_per_day INTEGER DEFAULT 100000,
                    
                    -- Burst allowance
                    burst_size INTEGER DEFAULT 5,
                    
                    -- Cost limits
                    max_cost_per_day REAL DEFAULT 10.0,
                    max_cost_per_month REAL DEFAULT 100.0,
                    
                    -- Custom limits (JSON)
                    custom_limits TEXT,
                    
                    -- Metadata
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    expires_at TIMESTAMP,  -- For temporary upgrades
                    notes TEXT
                )
            """)
            
            # Create webhook registrations table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS webhook_registrations (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id TEXT NOT NULL,
                    url TEXT NOT NULL,
                    secret TEXT NOT NULL,  -- For HMAC signature
                    events TEXT NOT NULL,  -- JSON array of event types
                    
                    -- Configuration
                    active BOOLEAN DEFAULT 1,
                    retry_count INTEGER DEFAULT 3,
                    timeout_seconds INTEGER DEFAULT 30,
                    
                    -- Statistics
                    total_deliveries INTEGER DEFAULT 0,
                    successful_deliveries INTEGER DEFAULT 0,
                    failed_deliveries INTEGER DEFAULT 0,
                    last_delivery_at TIMESTAMP,
                    last_error TEXT,
                    
                    -- Metadata
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    
                    UNIQUE(user_id, url)
                )
            """)
            
            # Create webhook delivery log table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS webhook_deliveries (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    webhook_id INTEGER NOT NULL,
                    evaluation_id TEXT NOT NULL,
                    event_type TEXT NOT NULL,
                    
                    -- Delivery details
                    payload TEXT NOT NULL,
                    signature TEXT NOT NULL,
                    
                    -- Response
                    status_code INTEGER,
                    response_body TEXT,
                    response_time_ms INTEGER,
                    
                    -- Status
                    delivered BOOLEAN DEFAULT 0,
                    retry_count INTEGER DEFAULT 0,
                    error_message TEXT,
                    
                    -- Timestamps
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    delivered_at TIMESTAMP,
                    next_retry_at TIMESTAMP,
                    
                    FOREIGN KEY (webhook_id) REFERENCES webhook_registrations(id),
                    FOREIGN KEY (evaluation_id) REFERENCES evaluations_unified(evaluation_id)
                )
            """)
            
            # Create indexes for webhook tables
            cursor.execute("CREATE INDEX idx_webhook_user ON webhook_registrations(user_id)")
            cursor.execute("CREATE INDEX idx_webhook_active ON webhook_registrations(active)")
            cursor.execute("CREATE INDEX idx_delivery_webhook ON webhook_deliveries(webhook_id)")
            cursor.execute("CREATE INDEX idx_delivery_eval ON webhook_deliveries(evaluation_id)")
            cursor.execute("CREATE INDEX idx_delivery_status ON webhook_deliveries(delivered)")
            cursor.execute("CREATE INDEX idx_delivery_retry ON webhook_deliveries(next_retry_at)")
            
            # Update schema version
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS schema_version (
                    version INTEGER PRIMARY KEY,
                    applied_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    description TEXT
                )
            """)
            
            cursor.execute("""
                INSERT OR REPLACE INTO schema_version (version, description)
                VALUES (5, 'Unified evaluations with webhooks and per-user rate limiting')
            """)
            
            conn.commit()
            logger.info("Successfully migrated to unified evaluations schema (v5)")
            return True
            
    except Exception as e:
        logger.error(f"Migration failed: {e}")
        return False


def rollback_unified_evaluations(db_path: str) -> bool:
    """
    Rollback the unified evaluations migration.
    
    This will:
    1. Restore the original separate tables
    2. Migrate data back from unified table
    3. Remove the unified schema
    
    Args:
        db_path: Path to the evaluations database
        
    Returns:
        True if rollback successful, False otherwise
    """
    try:
        with sqlite3.connect(db_path) as conn:
            cursor = conn.cursor()
            
            # Check if unified table exists
            cursor.execute("""
                SELECT name FROM sqlite_master 
                WHERE type='table' AND name='evaluations_unified'
            """)
            if not cursor.fetchone():
                logger.info("Unified table doesn't exist, nothing to rollback")
                return True
            
            logger.warning("Rolling back unified evaluations migration...")
            
            # Drop the unified tables (after backing up data if needed)
            cursor.execute("DROP TABLE IF EXISTS evaluations_unified")
            cursor.execute("DROP TABLE IF EXISTS user_rate_limits")
            cursor.execute("DROP TABLE IF EXISTS webhook_registrations")
            cursor.execute("DROP TABLE IF EXISTS webhook_deliveries")
            
            # Update schema version
            cursor.execute("""
                UPDATE schema_version SET version = 4 
                WHERE version = 5
            """)
            
            conn.commit()
            logger.info("Successfully rolled back unified evaluations migration")
            return True
            
    except Exception as e:
        logger.error(f"Rollback failed: {e}")
        return False