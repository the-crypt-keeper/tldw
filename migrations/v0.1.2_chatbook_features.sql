-- Migration Script for v0.1.2 - Chatbook Features
-- This migration adds tables for Chat Dictionary, World Books, Document Generation, and Chatbook Import/Export

-- ============================================
-- Chat Dictionary Tables
-- ============================================

-- Chat dictionaries table
CREATE TABLE IF NOT EXISTS chat_dictionaries (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id TEXT NOT NULL,
    name TEXT NOT NULL,
    description TEXT,
    is_active INTEGER DEFAULT 1,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(user_id, name)
);

-- Dictionary entries table
CREATE TABLE IF NOT EXISTS dictionary_entries (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    dictionary_id INTEGER NOT NULL,
    key_pattern TEXT NOT NULL,
    replacement TEXT NOT NULL,
    is_regex INTEGER DEFAULT 0,
    probability INTEGER DEFAULT 100,
    max_replacements INTEGER DEFAULT 1,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (dictionary_id) REFERENCES chat_dictionaries(id) ON DELETE CASCADE
);

-- Indexes for dictionary tables
CREATE INDEX IF NOT EXISTS idx_chat_dictionaries_user_id ON chat_dictionaries(user_id);
CREATE INDEX IF NOT EXISTS idx_chat_dictionaries_active ON chat_dictionaries(is_active);
CREATE INDEX IF NOT EXISTS idx_dictionary_entries_dictionary_id ON dictionary_entries(dictionary_id);

-- ============================================
-- World Book Tables
-- ============================================

-- World books table
CREATE TABLE IF NOT EXISTS world_books (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id TEXT NOT NULL,
    name TEXT NOT NULL,
    description TEXT,
    scan_depth INTEGER DEFAULT 3,
    token_budget INTEGER DEFAULT 1000,
    recursive_scanning INTEGER DEFAULT 0,
    enabled INTEGER DEFAULT 1,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(user_id, name)
);

-- World book entries table
CREATE TABLE IF NOT EXISTS world_book_entries (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    world_book_id INTEGER NOT NULL,
    keywords TEXT NOT NULL,  -- Comma-separated keywords
    content TEXT NOT NULL,
    priority INTEGER DEFAULT 50,
    enabled INTEGER DEFAULT 1,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (world_book_id) REFERENCES world_books(id) ON DELETE CASCADE
);

-- Character-world book attachments table
CREATE TABLE IF NOT EXISTS character_world_books (
    character_id INTEGER NOT NULL,
    world_book_id INTEGER NOT NULL,
    is_primary INTEGER DEFAULT 0,
    attached_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (character_id, world_book_id),
    FOREIGN KEY (world_book_id) REFERENCES world_books(id) ON DELETE CASCADE
);

-- Indexes for world book tables
CREATE INDEX IF NOT EXISTS idx_world_books_user_id ON world_books(user_id);
CREATE INDEX IF NOT EXISTS idx_world_books_enabled ON world_books(enabled);
CREATE INDEX IF NOT EXISTS idx_world_book_entries_world_book_id ON world_book_entries(world_book_id);
CREATE INDEX IF NOT EXISTS idx_world_book_entries_keywords ON world_book_entries(keywords);
CREATE INDEX IF NOT EXISTS idx_character_world_books_character_id ON character_world_books(character_id);

-- ============================================
-- Document Generation Tables
-- ============================================

-- Generated documents table
CREATE TABLE IF NOT EXISTS generated_documents (
    id TEXT PRIMARY KEY,  -- UUID
    user_id TEXT NOT NULL,
    conversation_id TEXT NOT NULL,
    document_type TEXT NOT NULL,  -- timeline, study_guide, briefing, summary, qa_pairs, meeting_notes
    title TEXT NOT NULL,
    content TEXT NOT NULL,
    metadata TEXT,  -- JSON
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Generation jobs table
CREATE TABLE IF NOT EXISTS generation_jobs (
    job_id TEXT PRIMARY KEY,  -- UUID
    user_id TEXT NOT NULL,
    conversation_id TEXT NOT NULL,
    document_type TEXT NOT NULL,
    status TEXT NOT NULL,  -- pending, processing, completed, failed, cancelled
    document_id TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    started_at TIMESTAMP,
    completed_at TIMESTAMP,
    error_message TEXT,
    FOREIGN KEY (document_id) REFERENCES generated_documents(id) ON DELETE SET NULL
);

-- User prompt configurations table
CREATE TABLE IF NOT EXISTS user_prompt_configs (
    user_id TEXT NOT NULL,
    document_type TEXT NOT NULL,
    custom_prompt TEXT NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (user_id, document_type)
);

-- Indexes for document generation tables
CREATE INDEX IF NOT EXISTS idx_generated_documents_user_id ON generated_documents(user_id);
CREATE INDEX IF NOT EXISTS idx_generated_documents_conversation_id ON generated_documents(conversation_id);
CREATE INDEX IF NOT EXISTS idx_generated_documents_type ON generated_documents(document_type);
CREATE INDEX IF NOT EXISTS idx_generation_jobs_user_id ON generation_jobs(user_id);
CREATE INDEX IF NOT EXISTS idx_generation_jobs_status ON generation_jobs(status);

-- ============================================
-- Chatbook Import/Export Tables
-- ============================================

-- Export jobs table
CREATE TABLE IF NOT EXISTS export_jobs (
    job_id TEXT PRIMARY KEY,  -- UUID
    user_id TEXT NOT NULL,
    name TEXT NOT NULL,
    description TEXT,
    content_types TEXT NOT NULL,  -- JSON array
    filters TEXT,  -- JSON
    status TEXT NOT NULL,  -- pending, processing, completed, failed, cancelled
    file_path TEXT,
    file_size INTEGER,
    content_summary TEXT,  -- JSON
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    started_at TIMESTAMP,
    completed_at TIMESTAMP,
    error_message TEXT
);

-- Import jobs table
CREATE TABLE IF NOT EXISTS import_jobs (
    job_id TEXT PRIMARY KEY,  -- UUID
    user_id TEXT NOT NULL,
    file_path TEXT NOT NULL,
    conflict_strategy TEXT NOT NULL,  -- skip, replace, rename
    status TEXT NOT NULL,  -- pending, processing, completed, failed, cancelled
    items_imported INTEGER DEFAULT 0,
    conflicts_found INTEGER DEFAULT 0,
    conflicts_resolved TEXT,  -- JSON
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    started_at TIMESTAMP,
    completed_at TIMESTAMP,
    error_message TEXT
);

-- Import conflicts tracking table
CREATE TABLE IF NOT EXISTS import_conflicts (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    job_id TEXT NOT NULL,
    content_type TEXT NOT NULL,
    item_id TEXT NOT NULL,
    conflict_type TEXT NOT NULL,  -- duplicate_id, duplicate_name
    resolution TEXT NOT NULL,  -- skipped, replaced, renamed
    original_name TEXT,
    new_name TEXT,
    resolved_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (job_id) REFERENCES import_jobs(job_id) ON DELETE CASCADE
);

-- Indexes for chatbook tables
CREATE INDEX IF NOT EXISTS idx_export_jobs_user_id ON export_jobs(user_id);
CREATE INDEX IF NOT EXISTS idx_export_jobs_status ON export_jobs(status);
CREATE INDEX IF NOT EXISTS idx_import_jobs_user_id ON import_jobs(user_id);
CREATE INDEX IF NOT EXISTS idx_import_jobs_status ON import_jobs(status);
CREATE INDEX IF NOT EXISTS idx_import_conflicts_job_id ON import_conflicts(job_id);

-- ============================================
-- Version Tracking
-- ============================================

-- Update version in migrations table (create if doesn't exist)
CREATE TABLE IF NOT EXISTS migrations (
    version TEXT PRIMARY KEY,
    applied_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    description TEXT
);

INSERT OR REPLACE INTO migrations (version, description)
VALUES ('0.1.2', 'Added Chatbook features: Chat Dictionary, World Books, Document Generation, Import/Export');

-- ============================================
-- Data Migration (if needed)
-- ============================================

-- Note: No data migration needed as these are new features

-- ============================================
-- Cleanup and Optimization
-- ============================================

-- Analyze tables for query optimization
ANALYZE chat_dictionaries;
ANALYZE dictionary_entries;
ANALYZE world_books;
ANALYZE world_book_entries;
ANALYZE character_world_books;
ANALYZE generated_documents;
ANALYZE generation_jobs;
ANALYZE export_jobs;
ANALYZE import_jobs;