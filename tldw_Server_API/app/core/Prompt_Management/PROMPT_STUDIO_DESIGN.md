# Prompt Studio Design Document V2 (Revised)

## Overview
A structured prompt engineering platform that extends the existing tldw_server infrastructure, combining DSPy's programmatic optimization with Anthropic Console's testing features while maintaining consistency with existing patterns.

## Architecture Principles

### Consistency with Existing System
1. **Database Patterns**: Use existing soft delete, UUID, sync_log patterns
2. **API Conventions**: Follow `/api/v1/{resource}/{action}` pattern
3. **Authentication**: Leverage existing JWT/token system
4. **Error Handling**: Use existing exception classes
5. **Logging**: Integrate with loguru patterns

### Core Design Decisions
1. **Extend, Don't Replace**: Build on top of `PromptsDatabase`
2. **Async First**: Use async/await for all I/O operations
3. **Queue Long Operations**: Use background jobs for optimization
4. **Progressive Enhancement**: Start simple, iterate based on usage
5. **Security by Default**: Input validation, rate limiting, sandboxing

## Database Design (Corrected)

### Schema with Proper Patterns

```sql
-- Projects table with full tracking
CREATE TABLE prompt_studio_projects (
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
CREATE INDEX idx_ps_projects_user ON prompt_studio_projects(user_id);
CREATE INDEX idx_ps_projects_deleted ON prompt_studio_projects(deleted);
CREATE INDEX idx_ps_projects_status ON prompt_studio_projects(status);
CREATE INDEX idx_ps_projects_updated ON prompt_studio_projects(updated_at);
CREATE INDEX idx_ps_projects_user_status ON prompt_studio_projects(user_id, status, deleted);

-- FTS for projects
CREATE VIRTUAL TABLE prompt_studio_projects_fts USING fts5(
    name, description, content=prompt_studio_projects
);

-- Signatures (contracts) - MOVED BEFORE prompts to resolve dependency
CREATE TABLE prompt_studio_signatures (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    uuid TEXT UNIQUE NOT NULL DEFAULT (lower(hex(randomblob(16)))),
    project_id INTEGER NOT NULL REFERENCES prompt_studio_projects(id),
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
CREATE INDEX idx_ps_signatures_project ON prompt_studio_signatures(project_id);
CREATE INDEX idx_ps_signatures_deleted ON prompt_studio_signatures(deleted);
CREATE INDEX idx_ps_signatures_name ON prompt_studio_signatures(name);

-- Prompts within projects (versioned)
CREATE TABLE prompt_studio_prompts (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    uuid TEXT UNIQUE NOT NULL DEFAULT (lower(hex(randomblob(16)))),
    project_id INTEGER NOT NULL REFERENCES prompt_studio_projects(id),
    signature_id INTEGER REFERENCES prompt_studio_signatures(id),
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
CREATE INDEX idx_ps_prompts_project ON prompt_studio_prompts(project_id);
CREATE INDEX idx_ps_prompts_parent ON prompt_studio_prompts(parent_version_id);
CREATE INDEX idx_ps_prompts_deleted ON prompt_studio_prompts(deleted);
CREATE INDEX idx_ps_prompts_signature ON prompt_studio_prompts(signature_id);
CREATE INDEX idx_ps_prompts_name ON prompt_studio_prompts(name);
CREATE INDEX idx_ps_prompts_project_name ON prompt_studio_prompts(project_id, name, deleted);

-- Test cases with proper tracking
CREATE TABLE prompt_studio_test_cases (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    uuid TEXT UNIQUE NOT NULL DEFAULT (lower(hex(randomblob(16)))),
    project_id INTEGER NOT NULL REFERENCES prompt_studio_projects(id),
    signature_id INTEGER REFERENCES prompt_studio_signatures(id),
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
CREATE INDEX idx_ps_test_cases_project ON prompt_studio_test_cases(project_id);
CREATE INDEX idx_ps_test_cases_golden ON prompt_studio_test_cases(is_golden);
CREATE INDEX idx_ps_test_cases_tags ON prompt_studio_test_cases(tags);
CREATE INDEX idx_ps_test_cases_signature ON prompt_studio_test_cases(signature_id);
CREATE INDEX idx_ps_test_cases_deleted ON prompt_studio_test_cases(deleted);
CREATE INDEX idx_ps_test_cases_project_golden ON prompt_studio_test_cases(project_id, is_golden, deleted);

-- Test runs (execution history)
CREATE TABLE prompt_studio_test_runs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    uuid TEXT UNIQUE NOT NULL DEFAULT (lower(hex(randomblob(16)))),
    project_id INTEGER NOT NULL REFERENCES prompt_studio_projects(id),
    prompt_id INTEGER NOT NULL REFERENCES prompt_studio_prompts(id),
    test_case_id INTEGER NOT NULL REFERENCES prompt_studio_test_cases(id),
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
CREATE INDEX idx_ps_test_runs_project ON prompt_studio_test_runs(project_id);
CREATE INDEX idx_ps_test_runs_prompt ON prompt_studio_test_runs(prompt_id);
CREATE INDEX idx_ps_test_runs_test_case ON prompt_studio_test_runs(test_case_id);
CREATE INDEX idx_ps_test_runs_created ON prompt_studio_test_runs(created_at);
CREATE INDEX idx_ps_test_runs_model ON prompt_studio_test_runs(model_name);
CREATE INDEX idx_ps_test_runs_client ON prompt_studio_test_runs(client_id);

-- Evaluations (batch test runs)
CREATE TABLE prompt_studio_evaluations (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    uuid TEXT UNIQUE NOT NULL DEFAULT (lower(hex(randomblob(16)))),
    project_id INTEGER NOT NULL REFERENCES prompt_studio_projects(id),
    prompt_id INTEGER NOT NULL REFERENCES prompt_studio_prompts(id),
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
CREATE INDEX idx_ps_evaluations_project ON prompt_studio_evaluations(project_id);
CREATE INDEX idx_ps_evaluations_prompt ON prompt_studio_evaluations(prompt_id);
CREATE INDEX idx_ps_evaluations_status ON prompt_studio_evaluations(status);

-- Optimization runs
CREATE TABLE prompt_studio_optimizations (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    uuid TEXT UNIQUE NOT NULL DEFAULT (lower(hex(randomblob(16)))),
    project_id INTEGER NOT NULL REFERENCES prompt_studio_projects(id),
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
CREATE INDEX idx_ps_optimizations_project ON prompt_studio_optimizations(project_id);
CREATE INDEX idx_ps_optimizations_status ON prompt_studio_optimizations(status);

-- Optimization job queue
CREATE TABLE prompt_studio_job_queue (
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
CREATE INDEX idx_ps_job_queue_status ON prompt_studio_job_queue(status, priority);
CREATE INDEX idx_ps_job_queue_type ON prompt_studio_job_queue(job_type);
CREATE INDEX idx_ps_job_queue_entity ON prompt_studio_job_queue(entity_id, job_type);
CREATE INDEX idx_ps_job_queue_client ON prompt_studio_job_queue(client_id);
CREATE INDEX idx_ps_job_queue_created ON prompt_studio_job_queue(created_at);

-- Sync log extension (following existing pattern)
-- Uses existing sync_log table structure, just new entity types
```

### Triggers for Sync and Audit

```sql
-- Auto-update timestamps
CREATE TRIGGER prompt_studio_projects_update 
AFTER UPDATE ON prompt_studio_projects
BEGIN
    UPDATE prompt_studio_projects 
    SET updated_at = CURRENT_TIMESTAMP, 
        last_modified = CURRENT_TIMESTAMP,
        version = version + 1
    WHERE id = NEW.id;
END;

-- Similar triggers for other tables...

-- Sync log triggers (following existing pattern)
CREATE TRIGGER prompt_studio_projects_sync_insert
AFTER INSERT ON prompt_studio_projects
BEGIN
    INSERT INTO sync_log (entity, entity_uuid, operation, client_id, version, payload)
    VALUES ('prompt_studio_project', NEW.uuid, 'create', NEW.client_id, NEW.version, 
            json_object('name', NEW.name, 'description', NEW.description));
END;
```

## API Design (Following Conventions)

### Endpoint Structure
Following the existing pattern: `/api/v1/{resource}/{action}`

```python
# Project Management
POST   /api/v1/prompt_studio/projects/create
GET    /api/v1/prompt_studio/projects/list
GET    /api/v1/prompt_studio/projects/get/{project_id}
PUT    /api/v1/prompt_studio/projects/update/{project_id}
DELETE /api/v1/prompt_studio/projects/delete/{project_id}
POST   /api/v1/prompt_studio/projects/archive/{project_id}
POST   /api/v1/prompt_studio/projects/unarchive/{project_id}

# Prompt Management within Projects
POST   /api/v1/prompt_studio/prompts/create
GET    /api/v1/prompt_studio/prompts/list/{project_id}
GET    /api/v1/prompt_studio/prompts/get/{prompt_id}
PUT    /api/v1/prompt_studio/prompts/update/{prompt_id}
GET    /api/v1/prompt_studio/prompts/history/{prompt_id}
POST   /api/v1/prompt_studio/prompts/revert/{prompt_id}/{version}

# Signature Management
POST   /api/v1/prompt_studio/signatures/create
GET    /api/v1/prompt_studio/signatures/list/{project_id}
PUT    /api/v1/prompt_studio/signatures/update/{signature_id}
POST   /api/v1/prompt_studio/signatures/validate

# Test Case Management
POST   /api/v1/prompt_studio/test_cases/create
POST   /api/v1/prompt_studio/test_cases/generate
GET    /api/v1/prompt_studio/test_cases/list/{project_id}
POST   /api/v1/prompt_studio/test_cases/import
GET    /api/v1/prompt_studio/test_cases/export/{project_id}
PUT    /api/v1/prompt_studio/test_cases/update/{test_case_id}
DELETE /api/v1/prompt_studio/test_cases/delete/{test_case_id}

# Testing & Evaluation
POST   /api/v1/prompt_studio/test/run
POST   /api/v1/prompt_studio/test/batch
GET    /api/v1/prompt_studio/test/results/{test_run_id}
POST   /api/v1/prompt_studio/evaluate/start
GET    /api/v1/prompt_studio/evaluate/status/{evaluation_id}
GET    /api/v1/prompt_studio/evaluate/results/{evaluation_id}
POST   /api/v1/prompt_studio/evaluate/compare

# Optimization
POST   /api/v1/prompt_studio/optimize/start
GET    /api/v1/prompt_studio/optimize/status/{optimization_id}
GET    /api/v1/prompt_studio/optimize/results/{optimization_id}
POST   /api/v1/prompt_studio/optimize/cancel/{optimization_id}

# Generation & Improvement
POST   /api/v1/prompt_studio/generate/prompt
POST   /api/v1/prompt_studio/improve/prompt
POST   /api/v1/prompt_studio/generate/examples

# Export & Integration
POST   /api/v1/prompt_studio/export/prompt/{prompt_id}
POST   /api/v1/prompt_studio/export/project/{project_id}
POST   /api/v1/prompt_studio/import/project

# Job Management
GET    /api/v1/prompt_studio/jobs/status/{job_id}
GET    /api/v1/prompt_studio/jobs/list
POST   /api/v1/prompt_studio/jobs/cancel/{job_id}
```

### Request/Response Patterns

All endpoints follow consistent patterns:

```python
# Standard Request Headers
Token: str  # Bearer token for authentication
X-Client-Id: str  # Client identifier for sync logging

# Standard Query Parameters for List Endpoints
page: int = Query(1, ge=1)
per_page: int = Query(20, ge=1, le=100)
include_deleted: bool = Query(False)
sort_by: str = Query("updated_at")
sort_order: str = Query("desc", regex="^(asc|desc)$")

# Standard Response Format
{
    "success": bool,
    "data": Any,  # Actual response data
    "error": Optional[str],
    "metadata": {
        "page": int,
        "per_page": int,
        "total": int,
        "total_pages": int
    }
}

# Error Response Format
{
    "success": false,
    "error": str,
    "error_code": str,
    "details": Optional[Dict],
    "request_id": str
}
```

## Integration Architecture

### Authentication Integration
```python
from app.core.AuthNZ.auth_utils import get_current_user, verify_token
from app.api.v1.API_Deps.Auth_Deps import get_auth_status

# Dependency for authenticated endpoints
async def get_prompt_studio_user(
    auth_status: dict = Depends(get_auth_status),
    token: str = Header(None)
) -> dict:
    """Extract user context from existing auth system"""
    if not auth_status.get("authenticated", False):
        raise HTTPException(status_code=401, detail="Authentication required")
    
    user_id = auth_status.get("user_id", "anonymous")
    client_id = auth_status.get("client_id", "web")
    
    return {
        "user_id": user_id,
        "client_id": client_id,
        "permissions": auth_status.get("permissions", []),
        "is_admin": auth_status.get("is_admin", False)
    }

# Permission checking
def require_project_access(
    project_id: str,
    user: dict = Depends(get_prompt_studio_user),
    db: PromptStudioDatabase = Depends(get_database)
) -> bool:
    """Verify user has access to project"""
    project = db.get_project(project_id)
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")
    
    # Check ownership or admin
    if project["user_id"] != user["user_id"] and not user["is_admin"]:
        raise HTTPException(status_code=403, detail="Access denied")
    
    return True
```

### Database Layer
```python
class PromptStudioDatabase(PromptsDatabase):
    """Extends existing PromptsDatabase with studio functionality"""
    
    def __init__(self, db_path: str, client_id: str):
        super().__init__(db_path, client_id)
        self._init_studio_schema()
    
    # Inherits all existing patterns:
    # - Thread-local connections
    # - Transaction management
    # - Sync logging
    # - Soft deletes
    # - FTS management
```

### Background Job System
```python
class PromptStudioJobProcessor:
    """Processes long-running optimization and evaluation jobs"""
    
    async def process_job(self, job_id: str):
        # 1. Fetch job from queue
        # 2. Update status to 'processing'
        # 3. Execute job based on type
        # 4. Store results
        # 5. Update status to 'completed'
        # 6. Send notification (webhook/websocket)
```

### Real-time Updates
```python
# WebSocket for live updates during optimization
@router.websocket("/ws/prompt_studio/{project_id}")
async def optimization_updates(websocket: WebSocket, project_id: str):
    await websocket.accept()
    # Send progress updates during optimization
    # Send intermediate results
    # Handle client disconnection gracefully
```

## Security & Safety

### Security Shim (Placeholder for Future Enhancement)
```python
class PromptStudioSecurity:
    """Basic security layer - to be expanded in future iterations"""
    
    def __init__(self):
        self.max_prompt_length = 50000  # Character limit
        self.max_test_cases = 1000  # Per project
        self.max_concurrent_jobs = 10  # Per user
    
    def validate_input(self, text: str) -> bool:
        """Basic input validation - expand later with injection detection"""
        if not text or len(text) > self.max_prompt_length:
            return False
        # TODO: Add prompt injection detection
        # TODO: Add PII detection
        # TODO: Add malicious pattern detection
        return True
    
    def check_rate_limit(self, user_id: str, endpoint: str) -> bool:
        """Placeholder for rate limiting - integrate with existing system"""
        # TODO: Integrate with existing rate limiting
        return True
    
    def estimate_cost(self, config: Dict) -> float:
        """Basic cost estimation"""
        # TODO: Implement accurate cost calculation
        return 0.0
    
    def sanitize_output(self, output: str) -> str:
        """Basic output sanitization"""
        # TODO: Add output filtering
        return output
```

### Future Security Enhancements
- Prompt injection detection using known patterns
- PII/sensitive data detection and masking
- Sandboxed execution environment
- Advanced rate limiting per operation type
- Cost caps and budget enforcement
- Output filtering and sanitization

## Performance Optimizations

### Caching Strategy
```python
# Redis-based caching for expensive operations
cache_keys = {
    "prompt_generation": "ps:gen:{task_hash}",
    "test_results": "ps:test:{prompt_id}:{test_case_id}",
    "evaluation_metrics": "ps:eval:{evaluation_id}",
    "optimization_state": "ps:opt:{optimization_id}"
}
```

### Batch Processing
```python
async def batch_evaluate(prompt_id: str, test_case_ids: List[str]):
    # Group by model for efficiency
    # Process in parallel where possible
    # Use connection pooling for LLM calls
    # Aggregate results efficiently
```

### Database Optimizations
1. **Prepared Statements**: For frequently used queries
2. **Connection Pooling**: Reuse database connections
3. **Batch Inserts**: For test results and metrics
4. **Materialized Views**: For complex aggregations

## Monitoring & Observability

### Metrics to Track
```python
metrics = {
    "optimization_duration": Histogram,
    "evaluation_accuracy": Gauge,
    "api_latency": Histogram,
    "token_usage": Counter,
    "cost_per_optimization": Histogram,
    "cache_hit_rate": Gauge,
    "job_queue_depth": Gauge,
    "error_rate": Counter
}
```

### Logging Standards
```python
# Following existing loguru patterns
logger.info("Starting optimization", extra={
    "project_id": project_id,
    "optimizer_type": optimizer_type,
    "user_id": user_id,
    "request_id": request_id
})
```

## Migration Strategy

### From Existing Prompts
```python
async def migrate_prompt_to_studio(prompt_id: str) -> str:
    # 1. Create new project
    # 2. Import prompt as v1
    # 3. Generate signature from prompt
    # 4. Create sample test cases
    # 5. Return project_id
```

### Backward Compatibility
- Studio prompts can be exported to regular prompts
- Existing prompt API remains unchanged
- Gradual migration path for users

## Implementation Phases (Realistic)

### Phase 1: Foundation (Weeks 1-2)
- Database schema with migrations
- Basic CRUD operations
- Authentication integration
- Initial test suite

### Phase 2: Core Features (Weeks 3-4)
- Test case management
- Basic prompt generation
- Simple evaluation
- Job queue system

### Phase 3: Optimization (Weeks 5-6)
- Bootstrap implementation
- Basic MIPRO optimizer
- Cost tracking
- Performance monitoring

### Phase 4: Advanced Features (Weeks 7-8)
- Module system
- Advanced comparisons
- WebSocket updates
- Full documentation

### Phase 5: Production Ready (Weeks 9-10)
- Performance optimization
- Security hardening
- Migration tools
- Gradual rollout

## Success Criteria

### Technical Metrics
- API response time < 200ms (p95)
- Optimization convergence in < 50 iterations
- Test coverage > 80%
- Zero security vulnerabilities

### Business Metrics
- 30% improvement in prompt quality
- 50% reduction in development time
- 90% user satisfaction score
- 25% cost reduction through optimization

## Risk Mitigation

### Technical Risks
- **Complexity**: Incremental development, feature flags
- **Performance**: Caching, async processing, monitoring
- **Cost Overruns**: Budget limits, cost estimates
- **Integration Issues**: Extensive testing, gradual rollout

### Operational Risks
- **User Adoption**: Training, documentation, support
- **Data Loss**: Regular backups, transaction logs
- **Service Degradation**: Circuit breakers, fallbacks
- **Security Breaches**: Regular audits, penetration testing