-- Prompt Studio Database Indexes
-- Version: 002
-- Description: Performance indexes for Prompt Studio tables
-- Date: 2024

-- Projects table indexes
CREATE INDEX IF NOT EXISTS idx_ps_projects_user ON prompt_studio_projects(user_id);
CREATE INDEX IF NOT EXISTS idx_ps_projects_deleted ON prompt_studio_projects(deleted);
CREATE INDEX IF NOT EXISTS idx_ps_projects_status ON prompt_studio_projects(status);
CREATE INDEX IF NOT EXISTS idx_ps_projects_updated ON prompt_studio_projects(updated_at);
CREATE INDEX IF NOT EXISTS idx_ps_projects_user_status ON prompt_studio_projects(user_id, status, deleted);

-- Signatures table indexes
CREATE INDEX IF NOT EXISTS idx_ps_signatures_project ON prompt_studio_signatures(project_id);
CREATE INDEX IF NOT EXISTS idx_ps_signatures_deleted ON prompt_studio_signatures(deleted);
CREATE INDEX IF NOT EXISTS idx_ps_signatures_name ON prompt_studio_signatures(name);

-- Prompts table indexes
CREATE INDEX IF NOT EXISTS idx_ps_prompts_project ON prompt_studio_prompts(project_id);
CREATE INDEX IF NOT EXISTS idx_ps_prompts_parent ON prompt_studio_prompts(parent_version_id);
CREATE INDEX IF NOT EXISTS idx_ps_prompts_deleted ON prompt_studio_prompts(deleted);
CREATE INDEX IF NOT EXISTS idx_ps_prompts_signature ON prompt_studio_prompts(signature_id);
CREATE INDEX IF NOT EXISTS idx_ps_prompts_name ON prompt_studio_prompts(name);
CREATE INDEX IF NOT EXISTS idx_ps_prompts_project_name ON prompt_studio_prompts(project_id, name, deleted);

-- Test cases table indexes
CREATE INDEX IF NOT EXISTS idx_ps_test_cases_project ON prompt_studio_test_cases(project_id);
CREATE INDEX IF NOT EXISTS idx_ps_test_cases_golden ON prompt_studio_test_cases(is_golden);
CREATE INDEX IF NOT EXISTS idx_ps_test_cases_tags ON prompt_studio_test_cases(tags);
CREATE INDEX IF NOT EXISTS idx_ps_test_cases_signature ON prompt_studio_test_cases(signature_id);
CREATE INDEX IF NOT EXISTS idx_ps_test_cases_deleted ON prompt_studio_test_cases(deleted);
CREATE INDEX IF NOT EXISTS idx_ps_test_cases_project_golden ON prompt_studio_test_cases(project_id, is_golden, deleted);

-- Test runs table indexes
CREATE INDEX IF NOT EXISTS idx_ps_test_runs_project ON prompt_studio_test_runs(project_id);
CREATE INDEX IF NOT EXISTS idx_ps_test_runs_prompt ON prompt_studio_test_runs(prompt_id);
CREATE INDEX IF NOT EXISTS idx_ps_test_runs_test_case ON prompt_studio_test_runs(test_case_id);
CREATE INDEX IF NOT EXISTS idx_ps_test_runs_created ON prompt_studio_test_runs(created_at);
CREATE INDEX IF NOT EXISTS idx_ps_test_runs_model ON prompt_studio_test_runs(model_name);
CREATE INDEX IF NOT EXISTS idx_ps_test_runs_client ON prompt_studio_test_runs(client_id);

-- Evaluations table indexes
CREATE INDEX IF NOT EXISTS idx_ps_evaluations_project ON prompt_studio_evaluations(project_id);
CREATE INDEX IF NOT EXISTS idx_ps_evaluations_prompt ON prompt_studio_evaluations(prompt_id);
CREATE INDEX IF NOT EXISTS idx_ps_evaluations_status ON prompt_studio_evaluations(status);

-- Optimizations table indexes
CREATE INDEX IF NOT EXISTS idx_ps_optimizations_project ON prompt_studio_optimizations(project_id);
CREATE INDEX IF NOT EXISTS idx_ps_optimizations_status ON prompt_studio_optimizations(status);

-- Job queue table indexes
CREATE INDEX IF NOT EXISTS idx_ps_job_queue_status ON prompt_studio_job_queue(status, priority);
CREATE INDEX IF NOT EXISTS idx_ps_job_queue_type ON prompt_studio_job_queue(job_type);
CREATE INDEX IF NOT EXISTS idx_ps_job_queue_entity ON prompt_studio_job_queue(entity_id, job_type);
CREATE INDEX IF NOT EXISTS idx_ps_job_queue_client ON prompt_studio_job_queue(client_id);
CREATE INDEX IF NOT EXISTS idx_ps_job_queue_created ON prompt_studio_job_queue(created_at);