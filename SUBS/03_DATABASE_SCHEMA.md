# Subscriptions Feature - Database Schema Design

## Overview

The database schema for the Subscriptions feature follows the established patterns in tldw_server:
- UUID-based synchronization support
- Optimistic concurrency control with version numbers
- Soft deletes (deleted flag)
- Comprehensive audit logging via sync_log
- Full-text search capabilities where appropriate
- Foreign key relationships with proper cascading

## Core Tables

### 1. Subscriptions Table

Stores user subscriptions to various content sources.

```sql
CREATE TABLE IF NOT EXISTS Subscriptions (
    -- Primary key
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    
    -- Core fields
    url TEXT NOT NULL,
    name TEXT NOT NULL,
    type TEXT NOT NULL CHECK(type IN ('youtube_channel', 'youtube_playlist', 'rss_feed', 'atom_feed', 'podcast', 'other')),
    
    -- Configuration
    check_interval INTEGER NOT NULL DEFAULT 3600, -- seconds between checks
    auto_import BOOLEAN NOT NULL DEFAULT 0,
    is_active BOOLEAN NOT NULL DEFAULT 1,
    
    -- Metadata
    description TEXT,
    thumbnail_url TEXT,
    author TEXT,
    language TEXT,
    
    -- Tracking
    last_checked DATETIME,
    last_successful_check DATETIME,
    consecutive_failures INTEGER DEFAULT 0,
    total_items_found INTEGER DEFAULT 0,
    total_items_imported INTEGER DEFAULT 0,
    
    -- User association (for future multi-user support)
    user_id INTEGER DEFAULT 1,
    
    -- Sync support fields
    uuid TEXT UNIQUE NOT NULL DEFAULT (lower(hex(randomblob(16)))),
    version INTEGER NOT NULL DEFAULT 1,
    last_modified DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
    client_id TEXT NOT NULL DEFAULT 'default',
    deleted BOOLEAN NOT NULL DEFAULT 0,
    
    -- Indexes for performance
    CHECK (deleted IN (0, 1))
);

-- Indexes
CREATE INDEX idx_subscriptions_url ON Subscriptions(url) WHERE deleted = 0;
CREATE INDEX idx_subscriptions_type ON Subscriptions(type) WHERE deleted = 0;
CREATE INDEX idx_subscriptions_user ON Subscriptions(user_id) WHERE deleted = 0;
CREATE INDEX idx_subscriptions_active ON Subscriptions(is_active) WHERE deleted = 0;
CREATE INDEX idx_subscriptions_last_checked ON Subscriptions(last_checked) WHERE deleted = 0;
CREATE INDEX idx_subscriptions_uuid ON Subscriptions(uuid);

-- Triggers for sync support
CREATE TRIGGER subscriptions_update_last_modified
AFTER UPDATE ON Subscriptions
FOR EACH ROW
WHEN OLD.last_modified = NEW.last_modified
BEGIN
    UPDATE Subscriptions SET last_modified = CURRENT_TIMESTAMP WHERE id = NEW.id;
END;

CREATE TRIGGER subscriptions_increment_version
AFTER UPDATE ON Subscriptions
FOR EACH ROW
WHEN OLD.version = NEW.version
BEGIN
    UPDATE Subscriptions SET version = OLD.version + 1 WHERE id = NEW.id;
END;
```

### 2. SubscriptionItems Table

Tracks individual content items discovered from subscriptions.

```sql
CREATE TABLE IF NOT EXISTS SubscriptionItems (
    -- Primary key
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    
    -- Relationships
    subscription_id INTEGER NOT NULL,
    media_id INTEGER, -- NULL until imported
    
    -- Item identification
    item_url TEXT NOT NULL,
    item_guid TEXT, -- RSS/Atom GUID if available
    
    -- Item metadata
    title TEXT NOT NULL,
    description TEXT,
    author TEXT,
    published_date DATETIME,
    updated_date DATETIME,
    duration INTEGER, -- seconds, for video/audio
    thumbnail_url TEXT,
    
    -- Processing status
    status TEXT NOT NULL DEFAULT 'new' CHECK(status IN ('new', 'reviewed', 'importing', 'imported', 'failed', 'skipped')),
    discovered_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
    reviewed_at DATETIME,
    imported_at DATETIME,
    error_message TEXT,
    
    -- Content type hints
    content_type TEXT, -- video, audio, article, etc.
    estimated_size INTEGER, -- bytes
    
    -- Additional metadata (JSON)
    extra_metadata TEXT, -- JSON field for platform-specific data
    
    -- Sync support fields
    uuid TEXT UNIQUE NOT NULL DEFAULT (lower(hex(randomblob(16)))),
    version INTEGER NOT NULL DEFAULT 1,
    last_modified DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
    client_id TEXT NOT NULL DEFAULT 'default',
    deleted BOOLEAN NOT NULL DEFAULT 0,
    
    -- Foreign keys
    FOREIGN KEY (subscription_id) REFERENCES Subscriptions(id) ON DELETE CASCADE,
    FOREIGN KEY (media_id) REFERENCES Media(id) ON DELETE SET NULL,
    
    -- Constraints
    CHECK (deleted IN (0, 1)),
    UNIQUE(subscription_id, item_url) -- Prevent duplicates per subscription
);

-- Indexes
CREATE INDEX idx_subscription_items_subscription ON SubscriptionItems(subscription_id) WHERE deleted = 0;
CREATE INDEX idx_subscription_items_status ON SubscriptionItems(status) WHERE deleted = 0;
CREATE INDEX idx_subscription_items_discovered ON SubscriptionItems(discovered_at) WHERE deleted = 0;
CREATE INDEX idx_subscription_items_url ON SubscriptionItems(item_url) WHERE deleted = 0;
CREATE INDEX idx_subscription_items_guid ON SubscriptionItems(item_guid) WHERE deleted = 0 AND item_guid IS NOT NULL;
CREATE INDEX idx_subscription_items_media ON SubscriptionItems(media_id) WHERE deleted = 0 AND media_id IS NOT NULL;
CREATE INDEX idx_subscription_items_uuid ON SubscriptionItems(uuid);

-- Triggers
CREATE TRIGGER subscription_items_update_last_modified
AFTER UPDATE ON SubscriptionItems
FOR EACH ROW
WHEN OLD.last_modified = NEW.last_modified
BEGIN
    UPDATE SubscriptionItems SET last_modified = CURRENT_TIMESTAMP WHERE id = NEW.id;
END;

CREATE TRIGGER subscription_items_increment_version
AFTER UPDATE ON SubscriptionItems
FOR EACH ROW
WHEN OLD.version = NEW.version
BEGIN
    UPDATE SubscriptionItems SET version = OLD.version + 1 WHERE id = NEW.id;
END;
```

### 3. SubscriptionChecks Table

Logs history of subscription checks for monitoring and debugging.

```sql
CREATE TABLE IF NOT EXISTS SubscriptionChecks (
    -- Primary key
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    
    -- Relationship
    subscription_id INTEGER NOT NULL,
    
    -- Check details
    check_start DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
    check_end DATETIME,
    check_duration_ms INTEGER,
    
    -- Results
    status TEXT NOT NULL CHECK(status IN ('started', 'success', 'partial', 'failed', 'timeout')),
    items_found INTEGER DEFAULT 0,
    new_items INTEGER DEFAULT 0,
    error_message TEXT,
    
    -- Performance metrics
    bytes_downloaded INTEGER,
    api_calls_made INTEGER,
    
    -- Sync support fields
    uuid TEXT UNIQUE NOT NULL DEFAULT (lower(hex(randomblob(16)))),
    
    -- Foreign key
    FOREIGN KEY (subscription_id) REFERENCES Subscriptions(id) ON DELETE CASCADE
);

-- Indexes
CREATE INDEX idx_subscription_checks_subscription ON SubscriptionChecks(subscription_id);
CREATE INDEX idx_subscription_checks_start ON SubscriptionChecks(check_start);
CREATE INDEX idx_subscription_checks_status ON SubscriptionChecks(status);
```

### 4. ImportRules Table

Stores automated import rules for subscriptions.

```sql
CREATE TABLE IF NOT EXISTS ImportRules (
    -- Primary key
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    
    -- Relationship
    subscription_id INTEGER NOT NULL,
    
    -- Rule definition
    rule_name TEXT NOT NULL,
    rule_type TEXT NOT NULL CHECK(rule_type IN ('keyword', 'author', 'date_range', 'regex', 'all')),
    rule_value TEXT NOT NULL, -- JSON encoded rule parameters
    
    -- Rule configuration
    is_active BOOLEAN NOT NULL DEFAULT 1,
    priority INTEGER NOT NULL DEFAULT 50,
    action TEXT NOT NULL DEFAULT 'import' CHECK(action IN ('import', 'skip', 'flag')),
    
    -- Statistics
    times_matched INTEGER DEFAULT 0,
    last_matched DATETIME,
    
    -- Sync support fields
    uuid TEXT UNIQUE NOT NULL DEFAULT (lower(hex(randomblob(16)))),
    version INTEGER NOT NULL DEFAULT 1,
    last_modified DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
    client_id TEXT NOT NULL DEFAULT 'default',
    deleted BOOLEAN NOT NULL DEFAULT 0,
    
    -- Foreign key
    FOREIGN KEY (subscription_id) REFERENCES Subscriptions(id) ON DELETE CASCADE,
    
    -- Constraints
    CHECK (deleted IN (0, 1))
);

-- Indexes
CREATE INDEX idx_import_rules_subscription ON ImportRules(subscription_id) WHERE deleted = 0;
CREATE INDEX idx_import_rules_active ON ImportRules(is_active) WHERE deleted = 0;
CREATE INDEX idx_import_rules_type ON ImportRules(rule_type) WHERE deleted = 0;
```

### 5. SubscriptionTags Junction Table

Links subscriptions to existing keyword tags.

```sql
CREATE TABLE IF NOT EXISTS SubscriptionTags (
    -- Composite primary key
    subscription_id INTEGER NOT NULL,
    keyword_id INTEGER NOT NULL,
    
    -- Metadata
    added_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
    
    -- Foreign keys
    FOREIGN KEY (subscription_id) REFERENCES Subscriptions(id) ON DELETE CASCADE,
    FOREIGN KEY (keyword_id) REFERENCES Keywords(id) ON DELETE CASCADE,
    
    -- Constraints
    PRIMARY KEY (subscription_id, keyword_id)
);

-- Indexes
CREATE INDEX idx_subscription_tags_subscription ON SubscriptionTags(subscription_id);
CREATE INDEX idx_subscription_tags_keyword ON SubscriptionTags(keyword_id);
```

## Integration with Existing Schema

### Relationships to Core Tables

1. **Media Table**
   - SubscriptionItems.media_id links to Media.id
   - Preserves subscription source in Media metadata

2. **Keywords Table**
   - Subscriptions can be tagged using existing keywords
   - Imported media inherits subscription tags

3. **sync_log Table**
   - All subscription tables participate in sync
   - Changes logged automatically via triggers

### Data Flow

```
Subscription → Check → SubscriptionItems → Import → Media
                ↓                                      ↓
         SubscriptionChecks                      MediaChunks
                                                       ↓
                                                  Embeddings
```

## Migration Strategy

### Initial Setup

```sql
-- Migration script for adding subscription tables
BEGIN TRANSACTION;

-- Create tables (use SQL from above)
-- ... 

-- Add subscription source to Media table
ALTER TABLE Media ADD COLUMN subscription_id INTEGER REFERENCES Subscriptions(id);
ALTER TABLE Media ADD COLUMN subscription_item_id INTEGER REFERENCES SubscriptionItems(id);

-- Create indexes for new Media columns
CREATE INDEX idx_media_subscription ON Media(subscription_id) WHERE subscription_id IS NOT NULL;
CREATE INDEX idx_media_subscription_item ON Media(subscription_item_id) WHERE subscription_item_id IS NOT NULL;

-- Add sync_log entries for new tables
INSERT INTO sync_log (table_name, operation, uuid, client_id)
VALUES 
    ('Subscriptions', 'CREATE_TABLE', lower(hex(randomblob(16))), 'migration'),
    ('SubscriptionItems', 'CREATE_TABLE', lower(hex(randomblob(16))), 'migration'),
    ('SubscriptionChecks', 'CREATE_TABLE', lower(hex(randomblob(16))), 'migration'),
    ('ImportRules', 'CREATE_TABLE', lower(hex(randomblob(16))), 'migration'),
    ('SubscriptionTags', 'CREATE_TABLE', lower(hex(randomblob(16))), 'migration');

COMMIT;
```

### Rollback Plan

```sql
-- Rollback script
BEGIN TRANSACTION;

-- Remove foreign key columns from Media
ALTER TABLE Media DROP COLUMN subscription_id;
ALTER TABLE Media DROP COLUMN subscription_item_id;

-- Drop tables in reverse order
DROP TABLE IF EXISTS SubscriptionTags;
DROP TABLE IF EXISTS ImportRules;
DROP TABLE IF EXISTS SubscriptionChecks;
DROP TABLE IF EXISTS SubscriptionItems;
DROP TABLE IF EXISTS Subscriptions;

-- Log rollback
INSERT INTO sync_log (table_name, operation, uuid, client_id)
VALUES ('Subscriptions_Rollback', 'ROLLBACK', lower(hex(randomblob(16))), 'migration');

COMMIT;
```

## Example Queries

### Get Active Subscriptions Due for Checking

```sql
SELECT s.*
FROM Subscriptions s
WHERE s.deleted = 0 
  AND s.is_active = 1
  AND (s.last_checked IS NULL 
       OR datetime('now') >= datetime(s.last_checked, '+' || s.check_interval || ' seconds'))
ORDER BY s.last_checked ASC NULLS FIRST;
```

### Get Unprocessed Items for Review

```sql
SELECT si.*, s.name as subscription_name
FROM SubscriptionItems si
JOIN Subscriptions s ON si.subscription_id = s.id
WHERE si.deleted = 0 
  AND si.status = 'new'
  AND s.user_id = ?
ORDER BY si.discovered_at DESC
LIMIT 100;
```

### Get Subscription Statistics

```sql
SELECT 
    s.id,
    s.name,
    s.type,
    COUNT(DISTINCT si.id) as total_items,
    COUNT(DISTINCT CASE WHEN si.status = 'imported' THEN si.id END) as imported_items,
    COUNT(DISTINCT CASE WHEN si.status = 'new' THEN si.id END) as new_items,
    MAX(si.discovered_at) as last_item_date
FROM Subscriptions s
LEFT JOIN SubscriptionItems si ON s.id = si.subscription_id AND si.deleted = 0
WHERE s.deleted = 0 AND s.user_id = ?
GROUP BY s.id, s.name, s.type;
```

### Apply Import Rules

```sql
SELECT ir.*, si.*
FROM SubscriptionItems si
JOIN ImportRules ir ON si.subscription_id = ir.subscription_id
WHERE si.deleted = 0 
  AND ir.deleted = 0 
  AND ir.is_active = 1
  AND si.status = 'new'
  AND (
    (ir.rule_type = 'keyword' AND si.title LIKE '%' || json_extract(ir.rule_value, '$.keyword') || '%')
    OR (ir.rule_type = 'author' AND si.author = json_extract(ir.rule_value, '$.author'))
    OR (ir.rule_type = 'all')
  )
ORDER BY ir.priority DESC, si.published_date DESC;
```

## Performance Considerations

### Indexing Strategy
- Primary indexes on foreign keys
- Composite indexes for common query patterns
- Partial indexes excluding deleted records
- UUID indexes for sync operations

### Query Optimization
- Use covering indexes where possible
- Batch operations for bulk imports
- Prepared statements for repeated queries
- Connection pooling for concurrent access

### Data Retention
- Periodic cleanup of old check logs
- Archive completed items after threshold
- Vacuum database regularly
- Monitor table sizes and growth