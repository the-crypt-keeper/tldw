-- Prompt Studio Database Triggers
-- Version: 003
-- Description: Triggers for auto-update timestamps and sync logging
-- Date: 2024

-- Auto-update timestamp triggers

-- Projects table update trigger
DROP TRIGGER IF EXISTS prompt_studio_projects_update;
CREATE TRIGGER prompt_studio_projects_update 
AFTER UPDATE ON prompt_studio_projects
FOR EACH ROW
WHEN OLD.updated_at = NEW.updated_at OR OLD.last_modified = NEW.last_modified
BEGIN
    UPDATE prompt_studio_projects 
    SET updated_at = CURRENT_TIMESTAMP, 
        last_modified = CURRENT_TIMESTAMP,
        version = version + 1
    WHERE id = NEW.id;
END;

-- Signatures table update trigger
DROP TRIGGER IF EXISTS prompt_studio_signatures_update;
CREATE TRIGGER prompt_studio_signatures_update
AFTER UPDATE ON prompt_studio_signatures
FOR EACH ROW
WHEN OLD.updated_at = NEW.updated_at
BEGIN
    UPDATE prompt_studio_signatures
    SET updated_at = CURRENT_TIMESTAMP
    WHERE id = NEW.id;
END;

-- Prompts table update trigger
DROP TRIGGER IF EXISTS prompt_studio_prompts_update;
CREATE TRIGGER prompt_studio_prompts_update
AFTER UPDATE ON prompt_studio_prompts
FOR EACH ROW
WHEN OLD.updated_at = NEW.updated_at
BEGIN
    UPDATE prompt_studio_prompts
    SET updated_at = CURRENT_TIMESTAMP
    WHERE id = NEW.id;
END;

-- Test cases table update trigger
DROP TRIGGER IF EXISTS prompt_studio_test_cases_update;
CREATE TRIGGER prompt_studio_test_cases_update
AFTER UPDATE ON prompt_studio_test_cases
FOR EACH ROW
WHEN OLD.updated_at = NEW.updated_at
BEGIN
    UPDATE prompt_studio_test_cases
    SET updated_at = CURRENT_TIMESTAMP
    WHERE id = NEW.id;
END;

-- Sync log triggers (conditionally created if sync_log table exists)
-- Note: These triggers will only work if sync_log table exists
-- They are wrapped to fail silently if the table doesn't exist

-- Projects sync triggers
DROP TRIGGER IF EXISTS prompt_studio_projects_sync_insert;
CREATE TRIGGER prompt_studio_projects_sync_insert
AFTER INSERT ON prompt_studio_projects
FOR EACH ROW
WHEN (SELECT COUNT(*) FROM sqlite_master WHERE type='table' AND name='sync_log') > 0
BEGIN
    INSERT INTO sync_log (entity, entity_uuid, operation, client_id, version, payload)
    SELECT 'prompt_studio_project', NEW.uuid, 'create', NEW.client_id, NEW.version, 
            json_object('name', NEW.name, 'description', NEW.description, 'status', NEW.status)
    WHERE EXISTS (SELECT 1 FROM sqlite_master WHERE type='table' AND name='sync_log');
END;

DROP TRIGGER IF EXISTS prompt_studio_projects_sync_update;
CREATE TRIGGER prompt_studio_projects_sync_update
AFTER UPDATE ON prompt_studio_projects
FOR EACH ROW
WHEN NEW.deleted = 0 AND (SELECT COUNT(*) FROM sqlite_master WHERE type='table' AND name='sync_log') > 0
BEGIN
    INSERT INTO sync_log (entity, entity_uuid, operation, client_id, version, payload)
    SELECT 'prompt_studio_project', NEW.uuid, 'update', NEW.client_id, NEW.version,
            json_object('name', NEW.name, 'description', NEW.description, 'status', NEW.status)
    WHERE EXISTS (SELECT 1 FROM sqlite_master WHERE type='table' AND name='sync_log');
END;

DROP TRIGGER IF EXISTS prompt_studio_projects_sync_delete;
CREATE TRIGGER prompt_studio_projects_sync_delete
AFTER UPDATE ON prompt_studio_projects
FOR EACH ROW
WHEN NEW.deleted = 1 AND OLD.deleted = 0 AND (SELECT COUNT(*) FROM sqlite_master WHERE type='table' AND name='sync_log') > 0
BEGIN
    INSERT INTO sync_log (entity, entity_uuid, operation, client_id, version, payload)
    SELECT 'prompt_studio_project', NEW.uuid, 'delete', NEW.client_id, NEW.version,
            json_object('deleted_at', NEW.deleted_at)
    WHERE EXISTS (SELECT 1 FROM sqlite_master WHERE type='table' AND name='sync_log');
END;

-- Prompts sync triggers
DROP TRIGGER IF EXISTS prompt_studio_prompts_sync_insert;
CREATE TRIGGER prompt_studio_prompts_sync_insert
AFTER INSERT ON prompt_studio_prompts
FOR EACH ROW
WHEN (SELECT COUNT(*) FROM sqlite_master WHERE type='table' AND name='sync_log') > 0
BEGIN
    INSERT INTO sync_log (entity, entity_uuid, operation, client_id, version, payload)
    SELECT 'prompt_studio_prompt', NEW.uuid, 'create', NEW.client_id, NEW.version_number,
            json_object('project_id', NEW.project_id, 'name', NEW.name, 'version', NEW.version_number)
    WHERE EXISTS (SELECT 1 FROM sqlite_master WHERE type='table' AND name='sync_log');
END;

DROP TRIGGER IF EXISTS prompt_studio_prompts_sync_update;
CREATE TRIGGER prompt_studio_prompts_sync_update
AFTER UPDATE ON prompt_studio_prompts
FOR EACH ROW
WHEN NEW.deleted = 0 AND (SELECT COUNT(*) FROM sqlite_master WHERE type='table' AND name='sync_log') > 0
BEGIN
    INSERT INTO sync_log (entity, entity_uuid, operation, client_id, version, payload)
    SELECT 'prompt_studio_prompt', NEW.uuid, 'update', NEW.client_id, NEW.version_number,
            json_object('project_id', NEW.project_id, 'name', NEW.name, 'version', NEW.version_number)
    WHERE EXISTS (SELECT 1 FROM sqlite_master WHERE type='table' AND name='sync_log');
END;

-- Test cases sync triggers
DROP TRIGGER IF EXISTS prompt_studio_test_cases_sync_insert;
CREATE TRIGGER prompt_studio_test_cases_sync_insert
AFTER INSERT ON prompt_studio_test_cases
FOR EACH ROW
WHEN (SELECT COUNT(*) FROM sqlite_master WHERE type='table' AND name='sync_log') > 0
BEGIN
    INSERT INTO sync_log (entity, entity_uuid, operation, client_id, version, payload)
    SELECT 'prompt_studio_test_case', NEW.uuid, 'create', NEW.client_id, 1,
            json_object('project_id', NEW.project_id, 'name', NEW.name, 'is_golden', NEW.is_golden)
    WHERE EXISTS (SELECT 1 FROM sqlite_master WHERE type='table' AND name='sync_log');
END;

DROP TRIGGER IF EXISTS prompt_studio_test_cases_sync_update;
CREATE TRIGGER prompt_studio_test_cases_sync_update
AFTER UPDATE ON prompt_studio_test_cases
FOR EACH ROW
WHEN NEW.deleted = 0 AND (SELECT COUNT(*) FROM sqlite_master WHERE type='table' AND name='sync_log') > 0
BEGIN
    INSERT INTO sync_log (entity, entity_uuid, operation, client_id, version, payload)
    SELECT 'prompt_studio_test_case', NEW.uuid, 'update', NEW.client_id, 1,
            json_object('project_id', NEW.project_id, 'name', NEW.name, 'is_golden', NEW.is_golden)
    WHERE EXISTS (SELECT 1 FROM sqlite_master WHERE type='table' AND name='sync_log');
END;