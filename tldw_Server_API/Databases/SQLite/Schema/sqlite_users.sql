-- SQLite Schema for User Registration System (Single-User Mode Fallback)
-- Version: 1.0.0
-- Description: Minimal schema for single-user mode and development

-- Enable foreign keys (must be done per connection in SQLite)
PRAGMA foreign_keys = ON;

-- ============================================
-- Users Table (Simplified for single-user mode)
-- ============================================
CREATE TABLE IF NOT EXISTS users (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    uuid TEXT UNIQUE NOT NULL,
    username TEXT UNIQUE NOT NULL,
    email TEXT UNIQUE NOT NULL,
    password_hash TEXT NOT NULL,
    
    -- Role
    role TEXT NOT NULL DEFAULT 'user' 
        CHECK (role IN ('user', 'moderator', 'admin', 'root', 'service')),
    
    -- Status flags
    is_active INTEGER DEFAULT 1,  -- SQLite uses 0/1 for boolean
    is_verified INTEGER DEFAULT 0,
    is_locked INTEGER DEFAULT 0,
    must_change_password INTEGER DEFAULT 0,
    
    -- Security
    locked_until TEXT,  -- ISO8601 timestamp string
    failed_login_attempts INTEGER DEFAULT 0,
    last_failed_login TEXT,
    
    -- Metadata
    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
    updated_at TEXT DEFAULT CURRENT_TIMESTAMP,
    last_login TEXT,
    created_by INTEGER REFERENCES users(id),
    deleted_at TEXT,
    deleted_by INTEGER REFERENCES users(id),
    
    -- Settings (JSON string)
    preferences TEXT DEFAULT '{}',
    
    -- Storage
    storage_quota_mb INTEGER DEFAULT 5120,
    storage_used_mb REAL DEFAULT 0.00 CHECK (storage_used_mb >= 0)
);

-- Indexes for performance
CREATE INDEX IF NOT EXISTS idx_users_username ON users(username);
CREATE INDEX IF NOT EXISTS idx_users_email ON users(email);
CREATE INDEX IF NOT EXISTS idx_users_role ON users(role);
CREATE INDEX IF NOT EXISTS idx_users_active ON users(is_active);

-- ============================================
-- API Keys Table (for single-user mode)
-- ============================================
CREATE TABLE IF NOT EXISTS api_keys (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    key_hash TEXT UNIQUE NOT NULL,
    name TEXT,
    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
    last_used TEXT,
    is_active INTEGER DEFAULT 1
);

CREATE INDEX IF NOT EXISTS idx_api_keys_hash ON api_keys(key_hash);
CREATE INDEX IF NOT EXISTS idx_api_keys_active ON api_keys(is_active);

-- ============================================
-- Sessions Table (minimal for single-user)
-- ============================================
CREATE TABLE IF NOT EXISTS sessions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    token_hash TEXT NOT NULL UNIQUE,
    refresh_token_hash TEXT UNIQUE,
    encrypted_token TEXT,
    encrypted_refresh TEXT,
    
    -- Session info
    ip_address TEXT,
    user_agent TEXT,
    
    -- Timestamps
    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
    last_activity TEXT DEFAULT CURRENT_TIMESTAMP,
    expires_at TEXT NOT NULL,
    
    -- Status
    is_active INTEGER DEFAULT 1,
    revoked_at TEXT
);

CREATE INDEX IF NOT EXISTS idx_sessions_token_hash ON sessions(token_hash);
CREATE INDEX IF NOT EXISTS idx_sessions_user_id ON sessions(user_id);
CREATE INDEX IF NOT EXISTS idx_sessions_expires ON sessions(expires_at);

-- ============================================
-- Registration Codes Table (for future multi-user migration)
-- ============================================
CREATE TABLE IF NOT EXISTS registration_codes (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    code TEXT UNIQUE NOT NULL,
    
    -- Usage limits
    max_uses INTEGER DEFAULT 1 CHECK (max_uses > 0),
    times_used INTEGER DEFAULT 0 CHECK (times_used >= 0),
    
    -- Validity
    expires_at TEXT NOT NULL,
    is_active INTEGER DEFAULT 1,
    
    -- Metadata
    created_by INTEGER NOT NULL REFERENCES users(id),
    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
    
    -- Settings
    role_to_grant TEXT DEFAULT 'user',
    allowed_email_domain TEXT,
    description TEXT,
    
    -- Usage tracking (JSON string)
    used_by TEXT DEFAULT '[]',
    
    -- Prevent race conditions with constraint
    CHECK (times_used <= max_uses)
);

CREATE INDEX IF NOT EXISTS idx_registration_codes_code ON registration_codes(code);
CREATE INDEX IF NOT EXISTS idx_registration_codes_active ON registration_codes(is_active);

-- ============================================
-- Rate Limits Table (simplified)
-- ============================================
CREATE TABLE IF NOT EXISTS rate_limits (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    identifier TEXT NOT NULL,  -- IP address or user_id
    endpoint TEXT NOT NULL,
    request_count INTEGER DEFAULT 1,
    window_start TEXT DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(identifier, endpoint, window_start)
);

CREATE INDEX IF NOT EXISTS idx_rate_limits_lookup ON rate_limits(identifier, endpoint, window_start);

-- ============================================
-- Audit Log Table (simplified, no partitioning)
-- ============================================
CREATE TABLE IF NOT EXISTS audit_log (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id INTEGER REFERENCES users(id),
    action TEXT NOT NULL,
    target_type TEXT,
    target_id INTEGER,
    
    -- Request details
    ip_address TEXT,
    user_agent TEXT,
    request_method TEXT,
    request_path TEXT,
    
    -- Results
    success INTEGER DEFAULT 1,
    error_message TEXT,
    
    -- Additional data (JSON string)
    details TEXT,
    created_at TEXT DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_audit_log_user_id ON audit_log(user_id);
CREATE INDEX IF NOT EXISTS idx_audit_log_action ON audit_log(action);
CREATE INDEX IF NOT EXISTS idx_audit_log_created_at ON audit_log(created_at);

-- ============================================
-- Security Logs Table (simplified)
-- ============================================
CREATE TABLE IF NOT EXISTS security_logs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    event_type TEXT NOT NULL,
    severity TEXT NOT NULL CHECK (severity IN ('info', 'warning', 'error', 'critical')),
    user_id INTEGER REFERENCES users(id),
    ip_address TEXT,
    user_agent TEXT,
    details TEXT,  -- JSON string
    created_at TEXT DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_security_logs_event_type ON security_logs(event_type);
CREATE INDEX IF NOT EXISTS idx_security_logs_severity ON security_logs(severity);
CREATE INDEX IF NOT EXISTS idx_security_logs_created_at ON security_logs(created_at);

-- ============================================
-- Password History Table
-- ============================================
CREATE TABLE IF NOT EXISTS password_history (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    password_hash TEXT NOT NULL,
    created_at TEXT DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_password_history_user_id ON password_history(user_id);

-- ============================================
-- Triggers for SQLite
-- ============================================

-- Trigger to update the updated_at timestamp
CREATE TRIGGER IF NOT EXISTS update_users_updated_at 
AFTER UPDATE ON users
FOR EACH ROW
BEGIN
    UPDATE users SET updated_at = CURRENT_TIMESTAMP WHERE id = NEW.id;
END;

-- ============================================
-- Initial Data for Single-User Mode
-- ============================================

-- Create default single user if it doesn't exist
-- Password hash should be properly generated by the application
INSERT OR IGNORE INTO users (
    id, 
    uuid, 
    username, 
    email, 
    password_hash, 
    role, 
    is_active,
    is_verified
) VALUES (
    1,
    'default-single-user-uuid',
    'single_user',
    'user@localhost',
    'PLACEHOLDER_HASH',  -- Will be updated by application
    'admin',
    1,
    1
);

-- Create default API key entry (hash will be set by application)
INSERT OR IGNORE INTO api_keys (
    id,
    key_hash,
    name,
    is_active
) VALUES (
    1,
    'PLACEHOLDER_HASH',  -- Will be updated by application
    'Default API Key',
    1
);