-- Prompt Studio Database Schema Migration
-- Version: 001
-- Description: Initial schema for Prompt Studio feature
-- Date: 2024

-- Projects table with full tracking
CREATE TABLE IF NOT EXISTS prompt_studio_projects (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    uuid TEXT UNIQUE NOT NULL DEFAULT (lower(hex(randomblob(16)))),
    name TEXT NOT NULL,
    description TEXT,
    user_id TEXT NOT NULL,
    client_id TEXT NOT NULL,
    status TEXT DEFAULT 'draft' CHECK (status IN ('draft', 'active', 'archived')),
    deleted INTEGER DEFAULT 0,
    deleted_at TIMESTAMP,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    last_modified TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    version INTEGER DEFAULT 1,
    metadata JSON,
    UNIQUE(name, user_id, deleted)
);

-- Signatures (contracts) - Created before prompts to resolve dependency
CREATE TABLE IF NOT EXISTS prompt_studio_signatures (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    uuid TEXT UNIQUE NOT NULL DEFAULT (lower(hex(randomblob(16)))),
    project_id INTEGER NOT NULL REFERENCES prompt_studio_projects(id) ON DELETE CASCADE,
    name TEXT NOT NULL,
    input_schema JSON NOT NULL,
    output_schema JSON NOT NULL,
    constraints JSON,
    validation_rules JSON,
    client_id TEXT NOT NULL,
    deleted INTEGER DEFAULT 0,
    deleted_at TIMESTAMP,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(project_id, name, deleted)
);

-- Prompts within projects (versioned)
CREATE TABLE IF NOT EXISTS prompt_studio_prompts (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    uuid TEXT UNIQUE NOT NULL DEFAULT (lower(hex(randomblob(16)))),
    project_id INTEGER NOT NULL REFERENCES prompt_studio_projects(id) ON DELETE CASCADE,
    signature_id INTEGER REFERENCES prompt_studio_signatures(id) ON DELETE SET NULL,
    version_number INTEGER NOT NULL DEFAULT 1,
    name TEXT NOT NULL,
    system_prompt TEXT,
    user_prompt TEXT,
    few_shot_examples JSON,
    modules_config JSON,
    parent_version_id INTEGER REFERENCES prompt_studio_prompts(id),
    change_description TEXT,
    client_id TEXT NOT NULL,
    deleted INTEGER DEFAULT 0,
    deleted_at TIMESTAMP,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(project_id, name, version_number)
);

-- Test cases with proper tracking
CREATE TABLE IF NOT EXISTS prompt_studio_test_cases (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    uuid TEXT UNIQUE NOT NULL DEFAULT (lower(hex(randomblob(16)))),
    project_id INTEGER NOT NULL REFERENCES prompt_studio_projects(id) ON DELETE CASCADE,
    signature_id INTEGER REFERENCES prompt_studio_signatures(id) ON DELETE SET NULL,
    name TEXT,
    description TEXT,
    inputs JSON NOT NULL,
    expected_outputs JSON,
    actual_outputs JSON,
    tags TEXT,
    is_golden INTEGER DEFAULT 0,
    is_generated INTEGER DEFAULT 0,
    client_id TEXT NOT NULL,
    deleted INTEGER DEFAULT 0,
    deleted_at TIMESTAMP,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Test runs (execution history)
CREATE TABLE IF NOT EXISTS prompt_studio_test_runs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    uuid TEXT UNIQUE NOT NULL DEFAULT (lower(hex(randomblob(16)))),
    project_id INTEGER NOT NULL REFERENCES prompt_studio_projects(id) ON DELETE CASCADE,
    prompt_id INTEGER NOT NULL REFERENCES prompt_studio_prompts(id) ON DELETE CASCADE,
    test_case_id INTEGER NOT NULL REFERENCES prompt_studio_test_cases(id) ON DELETE CASCADE,
    model_name TEXT NOT NULL,
    model_params JSON,
    inputs JSON NOT NULL,
    outputs JSON NOT NULL,
    expected_outputs JSON,
    scores JSON,
    execution_time_ms INTEGER,
    tokens_used INTEGER,
    cost_estimate REAL,
    error_message TEXT,
    client_id TEXT NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Evaluations (batch test runs)
CREATE TABLE IF NOT EXISTS prompt_studio_evaluations (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    uuid TEXT UNIQUE NOT NULL DEFAULT (lower(hex(randomblob(16)))),
    project_id INTEGER NOT NULL REFERENCES prompt_studio_projects(id) ON DELETE CASCADE,
    prompt_id INTEGER NOT NULL REFERENCES prompt_studio_prompts(id) ON DELETE CASCADE,
    name TEXT,
    description TEXT,
    test_case_ids JSON NOT NULL,
    test_run_ids JSON,
    aggregate_metrics JSON,
    model_configs JSON,
    total_tokens INTEGER,
    total_cost REAL,
    status TEXT DEFAULT 'pending' CHECK (status IN ('pending', 'running', 'completed', 'failed', 'cancelled')),
    error_message TEXT,
    client_id TEXT NOT NULL,
    started_at TIMESTAMP,
    completed_at TIMESTAMP,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Optimization runs
CREATE TABLE IF NOT EXISTS prompt_studio_optimizations (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    uuid TEXT UNIQUE NOT NULL DEFAULT (lower(hex(randomblob(16)))),
    project_id INTEGER NOT NULL REFERENCES prompt_studio_projects(id) ON DELETE CASCADE,
    initial_prompt_id INTEGER NOT NULL REFERENCES prompt_studio_prompts(id),
    optimized_prompt_id INTEGER REFERENCES prompt_studio_prompts(id),
    optimizer_type TEXT NOT NULL,
    optimization_config JSON NOT NULL,
    initial_metrics JSON,
    final_metrics JSON,
    improvement_percentage REAL,
    iterations_completed INTEGER,
    max_iterations INTEGER,
    bootstrap_samples INTEGER,
    status TEXT DEFAULT 'pending' CHECK (status IN ('pending', 'running', 'completed', 'failed', 'cancelled')),
    error_message TEXT,
    total_tokens INTEGER,
    total_cost REAL,
    client_id TEXT NOT NULL,
    started_at TIMESTAMP,
    completed_at TIMESTAMP,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Optimization job queue
CREATE TABLE IF NOT EXISTS prompt_studio_job_queue (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    uuid TEXT UNIQUE NOT NULL DEFAULT (lower(hex(randomblob(16)))),
    job_type TEXT NOT NULL CHECK (job_type IN ('evaluation', 'optimization', 'generation')),
    entity_id INTEGER NOT NULL,
    priority INTEGER DEFAULT 5,
    status TEXT DEFAULT 'queued' CHECK (status IN ('queued', 'processing', 'completed', 'failed', 'cancelled')),
    payload JSON NOT NULL,
    result JSON,
    error_message TEXT,
    retry_count INTEGER DEFAULT 0,
    max_retries INTEGER DEFAULT 3,
    client_id TEXT NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    started_at TIMESTAMP,
    completed_at TIMESTAMP
);