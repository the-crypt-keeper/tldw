-- PostgreSQL schema for tldw_server user registration system
-- Supports multi-user mode with advanced features

-- Enable required extensions
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS "pgcrypto";

-- Users table
CREATE TABLE IF NOT EXISTS users (
    id SERIAL PRIMARY KEY,
    uuid UUID DEFAULT uuid_generate_v4() UNIQUE NOT NULL,
    username VARCHAR(50) UNIQUE NOT NULL,
    email VARCHAR(255) UNIQUE NOT NULL,
    password_hash TEXT NOT NULL,
    role VARCHAR(20) NOT NULL DEFAULT 'user' 
        CHECK (role IN ('user', 'admin', 'service')),
    is_active BOOLEAN DEFAULT TRUE,
    is_verified BOOLEAN DEFAULT FALSE,
    is_locked BOOLEAN DEFAULT FALSE,
    locked_until TIMESTAMP,
    failed_login_attempts INT DEFAULT 0,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    last_login TIMESTAMP,
    email_verified_at TIMESTAMP,
    password_changed_at TIMESTAMP,
    preferences JSONB DEFAULT '{}',
    storage_quota_mb INT DEFAULT 5120,
    storage_used_mb DECIMAL(10,2) DEFAULT 0.00,
    CONSTRAINT positive_storage CHECK (storage_used_mb >= 0),
    CONSTRAINT valid_quota CHECK (storage_quota_mb > 0)
);

-- Optimized indexes for users
CREATE INDEX IF NOT EXISTS idx_users_username ON users(username) WHERE is_active = TRUE;
CREATE INDEX IF NOT EXISTS idx_users_email ON users(email) WHERE is_active = TRUE;
CREATE INDEX IF NOT EXISTS idx_users_role ON users(role) WHERE is_active = TRUE;
CREATE INDEX IF NOT EXISTS idx_users_locked ON users(id) WHERE is_locked = TRUE;
CREATE INDEX IF NOT EXISTS idx_users_created ON users(created_at);

-- Sessions table with automatic cleanup
CREATE TABLE IF NOT EXISTS sessions (
    id SERIAL PRIMARY KEY,
    user_id INT NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    token_hash VARCHAR(64) NOT NULL UNIQUE,
    refresh_token_hash VARCHAR(64) UNIQUE,
    encrypted_token TEXT,
    encrypted_refresh TEXT,
    expires_at TIMESTAMP NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    ip_address INET,
    user_agent TEXT,
    is_active BOOLEAN DEFAULT TRUE,
    last_accessed TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    CONSTRAINT valid_expiry CHECK (expires_at > created_at)
);

-- Session indexes
CREATE INDEX IF NOT EXISTS idx_sessions_token_hash ON sessions(token_hash) WHERE is_active = TRUE;
CREATE INDEX IF NOT EXISTS idx_sessions_user_id ON sessions(user_id) WHERE is_active = TRUE;
CREATE INDEX IF NOT EXISTS idx_sessions_expires ON sessions(expires_at) WHERE is_active = TRUE;
CREATE INDEX IF NOT EXISTS idx_sessions_cleanup ON sessions(expires_at) WHERE expires_at < CURRENT_TIMESTAMP;

-- Registration codes with race condition prevention
CREATE TABLE IF NOT EXISTS registration_codes (
    id SERIAL PRIMARY KEY,
    code VARCHAR(32) UNIQUE NOT NULL,
    max_uses INT DEFAULT 1 CHECK (max_uses > 0),
    times_used INT DEFAULT 0 CHECK (times_used >= 0),
    expires_at TIMESTAMP NOT NULL,
    created_by INT REFERENCES users(id),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    role_to_grant VARCHAR(20) DEFAULT 'user',
    metadata JSONB DEFAULT '{}',
    -- Prevent race conditions with constraint
    CONSTRAINT usage_limit CHECK (times_used <= max_uses),
    CONSTRAINT valid_role CHECK (role_to_grant IN ('user', 'admin', 'service')),
    CONSTRAINT future_expiry CHECK (expires_at > created_at)
);

-- Registration code indexes
CREATE INDEX IF NOT EXISTS idx_registration_codes_code ON registration_codes(code);
CREATE INDEX IF NOT EXISTS idx_registration_codes_active 
    ON registration_codes(code) 
    WHERE times_used < max_uses AND expires_at > CURRENT_TIMESTAMP;
CREATE INDEX IF NOT EXISTS idx_registration_codes_created_by ON registration_codes(created_by);

-- Rate limiting table
CREATE TABLE IF NOT EXISTS rate_limits (
    id SERIAL PRIMARY KEY,
    identifier VARCHAR(255) NOT NULL,  -- IP or user_id
    endpoint VARCHAR(255) NOT NULL,
    request_count INT DEFAULT 1,
    window_start TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(identifier, endpoint, window_start)
);

-- Rate limit indexes
CREATE INDEX IF NOT EXISTS idx_rate_limits_lookup 
    ON rate_limits(identifier, endpoint, window_start);
CREATE INDEX IF NOT EXISTS idx_rate_limits_cleanup 
    ON rate_limits(window_start) 
    WHERE window_start < CURRENT_TIMESTAMP - INTERVAL '1 hour';

-- Audit log table (partitioned by month)
CREATE TABLE IF NOT EXISTS audit_log (
    id BIGSERIAL NOT NULL,
    user_id INT,
    action VARCHAR(50) NOT NULL,
    details JSONB,
    ip_address INET,
    user_agent TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (id, created_at)
) PARTITION BY RANGE (created_at);

-- Create initial partition for current month
DO $$
DECLARE
    partition_name text;
    start_date date;
    end_date date;
BEGIN
    start_date := DATE_TRUNC('month', CURRENT_DATE);
    end_date := start_date + INTERVAL '1 month';
    partition_name := 'audit_log_' || TO_CHAR(start_date, 'YYYY_MM');
    
    -- Check if partition exists
    IF NOT EXISTS (
        SELECT 1 FROM pg_class 
        WHERE relname = partition_name
    ) THEN
        EXECUTE format(
            'CREATE TABLE %I PARTITION OF audit_log 
             FOR VALUES FROM (%L) TO (%L)',
            partition_name, start_date, end_date
        );
    END IF;
END $$;

-- Audit log indexes
CREATE INDEX IF NOT EXISTS idx_audit_log_user_id ON audit_log(user_id);
CREATE INDEX IF NOT EXISTS idx_audit_log_action ON audit_log(action);
CREATE INDEX IF NOT EXISTS idx_audit_log_created ON audit_log(created_at);

-- Password history table
CREATE TABLE IF NOT EXISTS password_history (
    id SERIAL PRIMARY KEY,
    user_id INT NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    password_hash TEXT NOT NULL,
    changed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    changed_by INT REFERENCES users(id),
    change_reason VARCHAR(100)
);

CREATE INDEX IF NOT EXISTS idx_password_history_user ON password_history(user_id);
CREATE INDEX IF NOT EXISTS idx_password_history_date ON password_history(changed_at);

-- Email verification tokens
CREATE TABLE IF NOT EXISTS email_verification_tokens (
    id SERIAL PRIMARY KEY,
    user_id INT NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    token VARCHAR(64) UNIQUE NOT NULL,
    expires_at TIMESTAMP NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    used_at TIMESTAMP,
    CONSTRAINT token_not_expired CHECK (expires_at > created_at)
);

CREATE INDEX IF NOT EXISTS idx_email_tokens_token ON email_verification_tokens(token);
CREATE INDEX IF NOT EXISTS idx_email_tokens_user ON email_verification_tokens(user_id);

-- User preferences table (for complex preferences)
CREATE TABLE IF NOT EXISTS user_preferences (
    id SERIAL PRIMARY KEY,
    user_id INT NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    preference_key VARCHAR(100) NOT NULL,
    preference_value JSONB NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(user_id, preference_key)
);

CREATE INDEX IF NOT EXISTS idx_user_prefs_user ON user_preferences(user_id);
CREATE INDEX IF NOT EXISTS idx_user_prefs_key ON user_preferences(preference_key);

-- Function to update updated_at timestamp
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = CURRENT_TIMESTAMP;
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

-- Triggers for updated_at
CREATE TRIGGER update_users_updated_at BEFORE UPDATE ON users
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_user_preferences_updated_at BEFORE UPDATE ON user_preferences
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

-- Cleanup function for expired data
CREATE OR REPLACE FUNCTION cleanup_expired_data()
RETURNS void AS $$
BEGIN
    -- Delete expired sessions
    DELETE FROM sessions 
    WHERE expires_at < CURRENT_TIMESTAMP - INTERVAL '1 day';
    
    -- Delete expired rate limits
    DELETE FROM rate_limits 
    WHERE window_start < CURRENT_TIMESTAMP - INTERVAL '1 hour';
    
    -- Delete expired registration codes
    DELETE FROM registration_codes 
    WHERE expires_at < CURRENT_TIMESTAMP - INTERVAL '30 days';
    
    -- Delete expired email verification tokens
    DELETE FROM email_verification_tokens
    WHERE expires_at < CURRENT_TIMESTAMP - INTERVAL '7 days';
    
    -- Archive old audit logs (older than 6 months)
    -- In production, you might want to move these to an archive table
    DELETE FROM audit_log
    WHERE created_at < CURRENT_TIMESTAMP - INTERVAL '180 days';
END;
$$ LANGUAGE plpgsql;

-- Function to create monthly partitions
CREATE OR REPLACE FUNCTION create_monthly_partition()
RETURNS void AS $$
DECLARE
    partition_name text;
    start_date date;
    end_date date;
BEGIN
    -- Create next month's partition
    start_date := DATE_TRUNC('month', CURRENT_DATE + INTERVAL '1 month');
    end_date := start_date + INTERVAL '1 month';
    partition_name := 'audit_log_' || TO_CHAR(start_date, 'YYYY_MM');
    
    -- Check if partition exists
    IF NOT EXISTS (
        SELECT 1 FROM pg_class 
        WHERE relname = partition_name
    ) THEN
        EXECUTE format(
            'CREATE TABLE %I PARTITION OF audit_log 
             FOR VALUES FROM (%L) TO (%L)',
            partition_name, start_date, end_date
        );
    END IF;
END;
$$ LANGUAGE plpgsql;

-- Default admin user (password: admin123 - CHANGE THIS!)
-- Password hash is for demonstration only - use proper hashing in production
INSERT INTO users (username, email, password_hash, role, is_active, is_verified)
VALUES ('admin', 'admin@localhost', 
        '$argon2id$v=19$m=32768,t=2,p=1$PLACEHOLDER_SALT$PLACEHOLDER_HASH',
        'admin', TRUE, TRUE)
ON CONFLICT (username) DO NOTHING;

-- Comments for documentation
COMMENT ON TABLE users IS 'Main user accounts table';
COMMENT ON TABLE sessions IS 'Active user sessions with JWT token hashes';
COMMENT ON TABLE registration_codes IS 'Registration codes for controlled user signup';
COMMENT ON TABLE rate_limits IS 'Rate limiting tracking per identifier/endpoint';
COMMENT ON TABLE audit_log IS 'Security and action audit log, partitioned by month';
COMMENT ON TABLE password_history IS 'Password change history for security compliance';
COMMENT ON TABLE email_verification_tokens IS 'Email verification token storage';
COMMENT ON TABLE user_preferences IS 'Complex user preferences stored as JSON';

-- Grant permissions (adjust for your specific database user)
-- GRANT ALL ON ALL TABLES IN SCHEMA public TO tldw_user;
-- GRANT ALL ON ALL SEQUENCES IN SCHEMA public TO tldw_user;
-- GRANT EXECUTE ON ALL FUNCTIONS IN SCHEMA public TO tldw_user;