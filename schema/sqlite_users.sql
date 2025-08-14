-- SQLite schema for tldw_server user registration system
-- Supports single-user and basic multi-user modes

-- Users table
CREATE TABLE IF NOT EXISTS users (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    uuid TEXT UNIQUE NOT NULL DEFAULT (lower(hex(randomblob(16)))),
    username TEXT UNIQUE NOT NULL,
    email TEXT UNIQUE NOT NULL,
    password_hash TEXT NOT NULL,
    role TEXT NOT NULL DEFAULT 'user' CHECK (role IN ('user', 'admin', 'service')),
    is_active INTEGER DEFAULT 1,
    is_verified INTEGER DEFAULT 0,
    is_locked INTEGER DEFAULT 0,
    locked_until TEXT,  -- ISO format timestamp
    failed_login_attempts INTEGER DEFAULT 0,
    created_at TEXT DEFAULT (datetime('now')),
    updated_at TEXT DEFAULT (datetime('now')),
    last_login TEXT,
    email_verified_at TEXT,
    password_changed_at TEXT,
    preferences TEXT DEFAULT '{}',  -- JSON string
    storage_quota_mb INTEGER DEFAULT 5120,
    storage_used_mb REAL DEFAULT 0.0,
    CHECK (storage_used_mb >= 0),
    CHECK (storage_quota_mb > 0)
);

-- Indexes for users
CREATE INDEX IF NOT EXISTS idx_users_username ON users(username) WHERE is_active = 1;
CREATE INDEX IF NOT EXISTS idx_users_email ON users(email) WHERE is_active = 1;
CREATE INDEX IF NOT EXISTS idx_users_role ON users(role) WHERE is_active = 1;
CREATE INDEX IF NOT EXISTS idx_users_locked ON users(id) WHERE is_locked = 1;
CREATE INDEX IF NOT EXISTS idx_users_created ON users(created_at);

-- Sessions table
CREATE TABLE IF NOT EXISTS sessions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    token_hash TEXT NOT NULL UNIQUE,
    refresh_token_hash TEXT UNIQUE,
    expires_at TEXT NOT NULL,  -- ISO format timestamp
    created_at TEXT DEFAULT (datetime('now')),
    ip_address TEXT,
    user_agent TEXT,
    is_active INTEGER DEFAULT 1,
    last_accessed TEXT DEFAULT (datetime('now'))
);

-- Session indexes
CREATE INDEX IF NOT EXISTS idx_sessions_token_hash ON sessions(token_hash) WHERE is_active = 1;
CREATE INDEX IF NOT EXISTS idx_sessions_user_id ON sessions(user_id) WHERE is_active = 1;
CREATE INDEX IF NOT EXISTS idx_sessions_expires ON sessions(expires_at) WHERE is_active = 1;

-- Registration codes
CREATE TABLE IF NOT EXISTS registration_codes (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    code TEXT UNIQUE NOT NULL,
    max_uses INTEGER DEFAULT 1 CHECK (max_uses > 0),
    times_used INTEGER DEFAULT 0 CHECK (times_used >= 0),
    expires_at TEXT NOT NULL,  -- ISO format timestamp
    created_by INTEGER REFERENCES users(id),
    created_at TEXT DEFAULT (datetime('now')),
    role_to_grant TEXT DEFAULT 'user' CHECK (role_to_grant IN ('user', 'admin', 'service')),
    metadata TEXT DEFAULT '{}',  -- JSON string
    CHECK (times_used <= max_uses)
);

-- Registration code indexes
CREATE INDEX IF NOT EXISTS idx_registration_codes_code ON registration_codes(code);
CREATE INDEX IF NOT EXISTS idx_registration_codes_active 
    ON registration_codes(code) 
    WHERE times_used < max_uses AND datetime(expires_at) > datetime('now');
CREATE INDEX IF NOT EXISTS idx_registration_codes_created_by ON registration_codes(created_by);

-- Rate limiting table
CREATE TABLE IF NOT EXISTS rate_limits (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    identifier TEXT NOT NULL,  -- IP or user_id
    endpoint TEXT NOT NULL,
    request_count INTEGER DEFAULT 1,
    window_start TEXT DEFAULT (datetime('now')),
    UNIQUE(identifier, endpoint, window_start)
);

-- Rate limit indexes
CREATE INDEX IF NOT EXISTS idx_rate_limits_lookup 
    ON rate_limits(identifier, endpoint, window_start);
CREATE INDEX IF NOT EXISTS idx_rate_limits_cleanup 
    ON rate_limits(window_start) 
    WHERE datetime(window_start) < datetime('now', '-1 hour');

-- Audit log table
CREATE TABLE IF NOT EXISTS audit_log (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id INTEGER,
    action TEXT NOT NULL,
    details TEXT,  -- JSON string
    ip_address TEXT,
    user_agent TEXT,
    created_at TEXT DEFAULT (datetime('now'))
);

-- Audit log indexes
CREATE INDEX IF NOT EXISTS idx_audit_log_user_id ON audit_log(user_id);
CREATE INDEX IF NOT EXISTS idx_audit_log_action ON audit_log(action);
CREATE INDEX IF NOT EXISTS idx_audit_log_created ON audit_log(created_at);

-- Password history table
CREATE TABLE IF NOT EXISTS password_history (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    password_hash TEXT NOT NULL,
    changed_at TEXT DEFAULT (datetime('now')),
    changed_by INTEGER REFERENCES users(id),
    change_reason TEXT
);

CREATE INDEX IF NOT EXISTS idx_password_history_user ON password_history(user_id);
CREATE INDEX IF NOT EXISTS idx_password_history_date ON password_history(changed_at);

-- Email verification tokens
CREATE TABLE IF NOT EXISTS email_verification_tokens (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    token TEXT UNIQUE NOT NULL,
    expires_at TEXT NOT NULL,  -- ISO format timestamp
    created_at TEXT DEFAULT (datetime('now')),
    used_at TEXT
);

CREATE INDEX IF NOT EXISTS idx_email_tokens_token ON email_verification_tokens(token);
CREATE INDEX IF NOT EXISTS idx_email_tokens_user ON email_verification_tokens(user_id);

-- User preferences table (for complex preferences)
CREATE TABLE IF NOT EXISTS user_preferences (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    preference_key TEXT NOT NULL,
    preference_value TEXT NOT NULL,  -- JSON string
    created_at TEXT DEFAULT (datetime('now')),
    updated_at TEXT DEFAULT (datetime('now')),
    UNIQUE(user_id, preference_key)
);

CREATE INDEX IF NOT EXISTS idx_user_prefs_user ON user_preferences(user_id);
CREATE INDEX IF NOT EXISTS idx_user_prefs_key ON user_preferences(preference_key);

-- Trigger to update updated_at timestamp (SQLite version)
CREATE TRIGGER IF NOT EXISTS update_users_updated_at
AFTER UPDATE ON users
FOR EACH ROW
BEGIN
    UPDATE users SET updated_at = datetime('now') WHERE id = NEW.id;
END;

CREATE TRIGGER IF NOT EXISTS update_user_preferences_updated_at
AFTER UPDATE ON user_preferences
FOR EACH ROW
BEGIN
    UPDATE user_preferences SET updated_at = datetime('now') WHERE id = NEW.id;
END;

-- Create default admin user for single-user mode
-- Note: This is a placeholder hash - the application will update it on first login
INSERT OR IGNORE INTO users (
    uuid,
    username, 
    email, 
    password_hash, 
    role, 
    is_active, 
    is_verified
) VALUES (
    'default-single-user-uuid',
    'single_user',
    'user@localhost',
    'PLACEHOLDER_HASH',  -- Will be updated on first login
    'admin',
    1,
    1
);

-- Views for easier querying

-- Active users view
CREATE VIEW IF NOT EXISTS active_users AS
SELECT 
    id,
    username,
    email,
    role,
    created_at,
    last_login,
    storage_quota_mb,
    storage_used_mb
FROM users
WHERE is_active = 1 AND is_verified = 1;

-- Active sessions view
CREATE VIEW IF NOT EXISTS active_sessions AS
SELECT 
    s.id,
    s.user_id,
    u.username,
    s.created_at,
    s.expires_at,
    s.ip_address,
    s.user_agent
FROM sessions s
JOIN users u ON s.user_id = u.id
WHERE s.is_active = 1 
    AND datetime(s.expires_at) > datetime('now');

-- Recent audit entries view
CREATE VIEW IF NOT EXISTS recent_audit_log AS
SELECT 
    a.id,
    a.user_id,
    u.username,
    a.action,
    a.details,
    a.ip_address,
    a.created_at
FROM audit_log a
LEFT JOIN users u ON a.user_id = u.id
WHERE datetime(a.created_at) > datetime('now', '-7 days')
ORDER BY a.created_at DESC;

-- Storage usage summary view
CREATE VIEW IF NOT EXISTS storage_usage_summary AS
SELECT 
    COUNT(*) as total_users,
    SUM(storage_used_mb) as total_used_mb,
    SUM(storage_quota_mb) as total_quota_mb,
    AVG(storage_used_mb) as avg_used_mb,
    MAX(storage_used_mb) as max_used_mb
FROM users
WHERE is_active = 1;

-- Helper indexes for FTS (Full Text Search) if needed
-- These would be used for searching user content, notes, etc.
-- CREATE VIRTUAL TABLE IF NOT EXISTS users_fts USING fts5(
--     username,
--     email,
--     content=users,
--     content_rowid=id
-- );

-- Cleanup old data (manual execution or scheduled job)
-- This is a template - adjust retention periods as needed
/*
DELETE FROM sessions WHERE datetime(expires_at) < datetime('now', '-7 days');
DELETE FROM rate_limits WHERE datetime(window_start) < datetime('now', '-1 day');
DELETE FROM registration_codes WHERE datetime(expires_at) < datetime('now', '-30 days');
DELETE FROM email_verification_tokens WHERE datetime(expires_at) < datetime('now', '-7 days');
DELETE FROM audit_log WHERE datetime(created_at) < datetime('now', '-180 days');
*/

-- Version tracking (useful for migrations)
CREATE TABLE IF NOT EXISTS schema_version (
    version INTEGER PRIMARY KEY,
    applied_at TEXT DEFAULT (datetime('now')),
    description TEXT
);

INSERT OR IGNORE INTO schema_version (version, description) 
VALUES (1, 'Initial user registration schema');