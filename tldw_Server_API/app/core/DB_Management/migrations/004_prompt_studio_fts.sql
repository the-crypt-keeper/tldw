-- Prompt Studio FTS (Full-Text Search) Tables
-- Version: 004
-- Description: Full-text search virtual tables for Prompt Studio
-- Date: 2024

-- FTS for projects (search by name and description)
DROP TABLE IF EXISTS prompt_studio_projects_fts;
CREATE VIRTUAL TABLE IF NOT EXISTS prompt_studio_projects_fts USING fts5(
    name, 
    description, 
    content=prompt_studio_projects,
    content_rowid=id
);

-- Populate FTS table with existing data
INSERT INTO prompt_studio_projects_fts(rowid, name, description)
SELECT id, name, description FROM prompt_studio_projects WHERE deleted = 0;

-- Triggers to keep FTS in sync with main table
DROP TRIGGER IF EXISTS prompt_studio_projects_fts_insert;
CREATE TRIGGER prompt_studio_projects_fts_insert 
AFTER INSERT ON prompt_studio_projects
FOR EACH ROW
WHEN NEW.deleted = 0
BEGIN
    INSERT INTO prompt_studio_projects_fts(rowid, name, description)
    VALUES (NEW.id, NEW.name, NEW.description);
END;

DROP TRIGGER IF EXISTS prompt_studio_projects_fts_update;
CREATE TRIGGER prompt_studio_projects_fts_update
AFTER UPDATE ON prompt_studio_projects
FOR EACH ROW
BEGIN
    DELETE FROM prompt_studio_projects_fts WHERE rowid = OLD.id;
    INSERT INTO prompt_studio_projects_fts(rowid, name, description)
    SELECT id, name, description FROM prompt_studio_projects 
    WHERE id = NEW.id AND deleted = 0;
END;

DROP TRIGGER IF EXISTS prompt_studio_projects_fts_delete;
CREATE TRIGGER prompt_studio_projects_fts_delete
AFTER DELETE ON prompt_studio_projects
FOR EACH ROW
BEGIN
    DELETE FROM prompt_studio_projects_fts WHERE rowid = OLD.id;
END;

-- FTS for prompts (search by name, system_prompt, user_prompt)
DROP TABLE IF EXISTS prompt_studio_prompts_fts;
CREATE VIRTUAL TABLE IF NOT EXISTS prompt_studio_prompts_fts USING fts5(
    name,
    system_prompt,
    user_prompt,
    content=prompt_studio_prompts,
    content_rowid=id
);

-- Populate FTS table with existing data
INSERT INTO prompt_studio_prompts_fts(rowid, name, system_prompt, user_prompt)
SELECT id, name, system_prompt, user_prompt FROM prompt_studio_prompts WHERE deleted = 0;

-- Triggers to keep FTS in sync
DROP TRIGGER IF EXISTS prompt_studio_prompts_fts_insert;
CREATE TRIGGER prompt_studio_prompts_fts_insert
AFTER INSERT ON prompt_studio_prompts
FOR EACH ROW
WHEN NEW.deleted = 0
BEGIN
    INSERT INTO prompt_studio_prompts_fts(rowid, name, system_prompt, user_prompt)
    VALUES (NEW.id, NEW.name, NEW.system_prompt, NEW.user_prompt);
END;

DROP TRIGGER IF EXISTS prompt_studio_prompts_fts_update;
CREATE TRIGGER prompt_studio_prompts_fts_update
AFTER UPDATE ON prompt_studio_prompts
FOR EACH ROW
BEGIN
    DELETE FROM prompt_studio_prompts_fts WHERE rowid = OLD.id;
    INSERT INTO prompt_studio_prompts_fts(rowid, name, system_prompt, user_prompt)
    SELECT id, name, system_prompt, user_prompt FROM prompt_studio_prompts
    WHERE id = NEW.id AND deleted = 0;
END;

DROP TRIGGER IF EXISTS prompt_studio_prompts_fts_delete;
CREATE TRIGGER prompt_studio_prompts_fts_delete
AFTER DELETE ON prompt_studio_prompts
FOR EACH ROW
BEGIN
    DELETE FROM prompt_studio_prompts_fts WHERE rowid = OLD.id;
END;

-- FTS for test cases (search by name, description, tags)
DROP TABLE IF EXISTS prompt_studio_test_cases_fts;
CREATE VIRTUAL TABLE IF NOT EXISTS prompt_studio_test_cases_fts USING fts5(
    name,
    description,
    tags,
    content=prompt_studio_test_cases,
    content_rowid=id
);

-- Populate FTS table with existing data
INSERT INTO prompt_studio_test_cases_fts(rowid, name, description, tags)
SELECT id, name, description, tags FROM prompt_studio_test_cases WHERE deleted = 0;

-- Triggers to keep FTS in sync
DROP TRIGGER IF EXISTS prompt_studio_test_cases_fts_insert;
CREATE TRIGGER prompt_studio_test_cases_fts_insert
AFTER INSERT ON prompt_studio_test_cases
FOR EACH ROW
WHEN NEW.deleted = 0
BEGIN
    INSERT INTO prompt_studio_test_cases_fts(rowid, name, description, tags)
    VALUES (NEW.id, NEW.name, NEW.description, NEW.tags);
END;

DROP TRIGGER IF EXISTS prompt_studio_test_cases_fts_update;
CREATE TRIGGER prompt_studio_test_cases_fts_update
AFTER UPDATE ON prompt_studio_test_cases
FOR EACH ROW
BEGIN
    DELETE FROM prompt_studio_test_cases_fts WHERE rowid = OLD.id;
    INSERT INTO prompt_studio_test_cases_fts(rowid, name, description, tags)
    SELECT id, name, description, tags FROM prompt_studio_test_cases
    WHERE id = NEW.id AND deleted = 0;
END;

DROP TRIGGER IF EXISTS prompt_studio_test_cases_fts_delete;
CREATE TRIGGER prompt_studio_test_cases_fts_delete
AFTER DELETE ON prompt_studio_test_cases
FOR EACH ROW
BEGIN
    DELETE FROM prompt_studio_test_cases_fts WHERE rowid = OLD.id;
END;