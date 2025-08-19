-- PostgreSQL Schema for User Registration System
-- Version: 1.0.0
-- Description: Production-ready schema with proper constraints, indexes, and partitioning

-- Enable required extensions
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS "pgcrypto";

-- Drop tables if they exist (for clean installation)
-- Comment these out in production after initial setup
-- DROP TABLE IF EXISTS audit_log CASCADE;
-- DROP TABLE IF EXISTS security_logs CASCADE;
-- DROP TABLE IF EXISTS rate_limits CASCADE;
-- DROP TABLE IF EXISTS sessions CASCADE;
-- DROP TABLE IF EXISTS registration_codes CASCADE;
-- DROP TABLE IF EXISTS password_history CASCADE;
-- DROP TABLE IF EXISTS service_accounts CASCADE;
-- DROP TABLE IF EXISTS users CASCADE;

-- ============================================
-- Users Table
-- ============================================
CREATE TABLE IF NOT EXISTS users (
    id SERIAL PRIMARY KEY,
    uuid UUID DEFAULT uuid_generate_v4() UNIQUE NOT NULL,
    username VARCHAR(50) UNIQUE NOT NULL,
    email VARCHAR(255) UNIQUE NOT NULL,
    password_hash TEXT NOT NULL,
    
    -- Role (simple enum for now, can be expanded)
    role VARCHAR(20) NOT NULL DEFAULT 'user' 
        CHECK (role IN ('user', 'moderator', 'admin', 'root', 'service')),
    
    -- Status flags
    is_active BOOLEAN DEFAULT TRUE,
    is_verified BOOLEAN DEFAULT FALSE,
    is_locked BOOLEAN DEFAULT FALSE,
    must_change_password BOOLEAN DEFAULT FALSE,
    
    -- Security
    locked_until TIMESTAMP,
    failed_login_attempts INT DEFAULT 0,
    last_failed_login TIMESTAMP,
    
    -- Metadata
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    last_login TIMESTAMP,
    created_by INT REFERENCES users(id),
    deleted_at TIMESTAMP,
    deleted_by INT REFERENCES users(id),
    
    -- Settings (JSON for flexibility)
    preferences JSONB DEFAULT '{}',
    
    -- Storage
    storage_quota_mb INT DEFAULT 5120, -- 5GB default
    storage_used_mb DECIMAL(10,2) DEFAULT 0.00 CHECK (storage_used_mb >= 0)
);

-- Indexes for performance
CREATE INDEX IF NOT EXISTS idx_users_username ON users(username) WHERE is_active = TRUE;
CREATE INDEX IF NOT EXISTS idx_users_email ON users(email) WHERE is_active = TRUE;
CREATE INDEX IF NOT EXISTS idx_users_role ON users(role) WHERE is_active = TRUE;
CREATE INDEX IF NOT EXISTS idx_users_locked ON users(id) WHERE is_locked = TRUE;
CREATE INDEX IF NOT EXISTS idx_users_deleted ON users(deleted_at) WHERE deleted_at IS NOT NULL;

-- ============================================
-- Password History Table
-- ============================================
CREATE TABLE IF NOT EXISTS password_history (
    id SERIAL PRIMARY KEY,
    user_id INT NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    password_hash TEXT NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_password_history_user_id ON password_history(user_id);
CREATE INDEX IF NOT EXISTS idx_password_history_created ON password_history(created_at);

-- ============================================
-- Service Accounts Table
-- ============================================
CREATE TABLE IF NOT EXISTS service_accounts (
    id SERIAL PRIMARY KEY,
    name VARCHAR(100) UNIQUE NOT NULL,
    api_key_hash VARCHAR(64) UNIQUE NOT NULL,
    created_by INT REFERENCES users(id),
    permissions JSONB DEFAULT '{}',
    rate_limit_per_minute INT DEFAULT 1000,
    is_active BOOLEAN DEFAULT TRUE,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    last_used TIMESTAMP,
    expires_at TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_service_accounts_api_key ON service_accounts(api_key_hash);
CREATE INDEX IF NOT EXISTS idx_service_accounts_active ON service_accounts(is_active);
CREATE INDEX IF NOT EXISTS idx_service_accounts_expires ON service_accounts(expires_at) WHERE expires_at IS NOT NULL;

-- ============================================
-- Sessions Table with automatic cleanup
-- ============================================
CREATE TABLE IF NOT EXISTS sessions (
    id SERIAL PRIMARY KEY,
    user_id INT NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    token_hash VARCHAR(64) NOT NULL UNIQUE,
    refresh_token_hash VARCHAR(64) UNIQUE,
    encrypted_token TEXT,
    encrypted_refresh TEXT,
    
    -- Session info
    ip_address INET,
    user_agent TEXT,
    device_id VARCHAR(100),
    
    -- Timestamps
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    last_activity TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    expires_at TIMESTAMP NOT NULL,
    
    -- Status
    is_active BOOLEAN DEFAULT TRUE,
    revoked_at TIMESTAMP,
    revoked_by INT REFERENCES users(id),
    revoke_reason TEXT
);

-- Performance indexes
CREATE INDEX IF NOT EXISTS idx_sessions_token_hash ON sessions(token_hash) WHERE is_active = TRUE;
CREATE INDEX IF NOT EXISTS idx_sessions_refresh_hash ON sessions(refresh_token_hash) WHERE is_active = TRUE;
CREATE INDEX IF NOT EXISTS idx_sessions_user_id ON sessions(user_id) WHERE is_active = TRUE;
CREATE INDEX IF NOT EXISTS idx_sessions_expires ON sessions(expires_at) WHERE is_active = TRUE;
CREATE INDEX IF NOT EXISTS idx_sessions_cleanup ON sessions(expires_at) WHERE expires_at < CURRENT_TIMESTAMP;

-- ============================================
-- Registration Codes Table with race condition prevention
-- ============================================
CREATE TABLE IF NOT EXISTS registration_codes (
    id SERIAL PRIMARY KEY,
    code VARCHAR(32) UNIQUE NOT NULL,
    
    -- Usage limits
    max_uses INT DEFAULT 1 CHECK (max_uses > 0),
    times_used INT DEFAULT 0 CHECK (times_used >= 0),
    
    -- Validity
    expires_at TIMESTAMP NOT NULL,
    is_active BOOLEAN DEFAULT TRUE,
    
    -- Metadata
    created_by INT NOT NULL REFERENCES users(id),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    
    -- Settings
    role_to_grant VARCHAR(20) DEFAULT 'user',
    allowed_email_domain VARCHAR(100),
    description TEXT,
    
    -- Usage tracking (JSON array)
    used_by JSONB DEFAULT '[]',
    
    -- Prevent race conditions with constraint
    CONSTRAINT usage_limit CHECK (times_used <= max_uses)
);

CREATE INDEX IF NOT EXISTS idx_registration_codes_code ON registration_codes(code);
CREATE INDEX IF NOT EXISTS idx_registration_codes_active 
    ON registration_codes(code) 
    WHERE is_active = TRUE AND times_used < max_uses AND expires_at > CURRENT_TIMESTAMP;
CREATE INDEX IF NOT EXISTS idx_registration_codes_created_by ON registration_codes(created_by);

-- ============================================
-- Rate Limiting Table
-- ============================================
CREATE TABLE IF NOT EXISTS rate_limits (
    id SERIAL PRIMARY KEY,
    identifier VARCHAR(255) NOT NULL,  -- IP address or user_id
    endpoint VARCHAR(255) NOT NULL,
    request_count INT DEFAULT 1,
    window_start TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(identifier, endpoint, window_start)
);

CREATE INDEX IF NOT EXISTS idx_rate_limits_lookup 
    ON rate_limits(identifier, endpoint, window_start);
CREATE INDEX IF NOT EXISTS idx_rate_limits_cleanup 
    ON rate_limits(window_start) 
    WHERE window_start < CURRENT_TIMESTAMP - INTERVAL '1 hour';

-- ============================================
-- Audit Log Table (Partitioned by month)
-- ============================================
CREATE TABLE IF NOT EXISTS audit_log (
    id BIGSERIAL,
    user_id INT REFERENCES users(id),
    action VARCHAR(50) NOT NULL,
    target_type VARCHAR(50),
    target_id INT,
    
    -- Request details
    ip_address INET,
    user_agent TEXT,
    request_method VARCHAR(10),
    request_path TEXT,
    
    -- Results
    success BOOLEAN NOT NULL DEFAULT TRUE,
    error_message TEXT,
    
    -- Additional data
    details JSONB,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
) PARTITION BY RANGE (created_at);

-- Create indexes on parent table (inherited by partitions)
CREATE INDEX IF NOT EXISTS idx_audit_log_user_id ON audit_log(user_id);
CREATE INDEX IF NOT EXISTS idx_audit_log_action ON audit_log(action);
CREATE INDEX IF NOT EXISTS idx_audit_log_created_at ON audit_log(created_at);

-- Create first partition (current month)
DO $$
DECLARE
    partition_name TEXT;
    start_date DATE;
    end_date DATE;
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

-- ============================================
-- Security Logs Table
-- ============================================
CREATE TABLE IF NOT EXISTS security_logs (
    id SERIAL PRIMARY KEY,
    event_type VARCHAR(50) NOT NULL, -- failed_login, brute_force, privilege_escalation, etc.
    severity VARCHAR(20) NOT NULL CHECK (severity IN ('info', 'warning', 'error', 'critical')),
    user_id INT REFERENCES users(id),
    ip_address INET,
    user_agent TEXT,
    details JSONB,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_security_logs_event_type ON security_logs(event_type);
CREATE INDEX IF NOT EXISTS idx_security_logs_severity ON security_logs(severity);
CREATE INDEX IF NOT EXISTS idx_security_logs_user_id ON security_logs(user_id);
CREATE INDEX IF NOT EXISTS idx_security_logs_created_at ON security_logs(created_at);

-- ============================================
-- Functions and Triggers
-- ============================================

-- Function to update the updated_at timestamp
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = CURRENT_TIMESTAMP;
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

-- Trigger to automatically update updated_at
CREATE TRIGGER update_users_updated_at BEFORE UPDATE ON users
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

-- Function to clean up expired sessions
CREATE OR REPLACE FUNCTION cleanup_expired_sessions()
RETURNS void AS $$
BEGIN
    DELETE FROM sessions 
    WHERE expires_at < CURRENT_TIMESTAMP - INTERVAL '1 day';
    
    DELETE FROM rate_limits 
    WHERE window_start < CURRENT_TIMESTAMP - INTERVAL '1 hour';
    
    DELETE FROM registration_codes 
    WHERE expires_at < CURRENT_TIMESTAMP - INTERVAL '30 days'
        AND is_active = FALSE;
END;
$$ LANGUAGE plpgsql;

-- Function to create monthly audit log partitions
CREATE OR REPLACE FUNCTION create_monthly_audit_partition()
RETURNS void AS $$
DECLARE
    partition_name TEXT;
    start_date DATE;
    end_date DATE;
BEGIN
    -- Create partition for next month
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

-- ============================================
-- Initial Data (Optional)
-- ============================================

-- Create root user if it doesn't exist (password should be changed immediately)
-- This is commented out by default for security
-- INSERT INTO users (username, email, password_hash, role, is_active, must_change_password)
-- VALUES ('root', 'root@localhost', 'CHANGE_ME_IMMEDIATELY', 'root', true, true)
-- ON CONFLICT (username) DO NOTHING;