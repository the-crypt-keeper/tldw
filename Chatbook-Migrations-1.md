# Chatbook Migration Plan: Single-User TUI to Multi-User API Architecture

## Executive Summary

This document outlines the migration plan for adapting features from reference modules located in `/REFERENCE/` folder to the multi-user, API-first tldw_server architecture. The reference modules come from a single-user TUI application (tldw_chatbook) and contain valuable features that can enhance the current system, but require significant architectural adaptations to support multiple users, stateless API design, and proper resource isolation.

## File Structure Overview

### Source Files (Reference Modules)
```
/REFERENCE/
├── Character_Chat/
│   ├── Character_Chat_Lib.py      # Core character management
│   ├── Chat_Dictionary_Lib.py     # Pattern-based text replacement
│   ├── ccv3_parser.py             # Character card format parsing
│   ├── character_card_formats.py  # Card format definitions
│   ├── world_book_manager.py      # World book CRUD operations
│   └── world_info_processor.py    # World info injection logic
├── Chat/
│   ├── Chat_Functions.py          # Chat operations
│   ├── Chat_Deps.py              # Chat dependencies
│   ├── chat_models.py            # Session data models
│   ├── document_generator.py     # Document generation service
│   ├── prompt_template_manager.py # Template management
│   └── tabs/
│       ├── tab_context.py        # Tab context management
│       └── tab_state_manager.py  # Thread-safe tab states
├── Chatbooks/
│   ├── chatbook_models.py        # Chatbook data structures
│   ├── chatbook_creator.py       # Export functionality
│   ├── chatbook_importer.py      # Import functionality
│   ├── conflict_resolver.py      # Conflict resolution logic
│   └── error_handler.py          # Error handling utilities
└── LLM_Calls/
    ├── LLM_API_Calls.py          # LLM API integrations
    ├── LLM_API_Calls_Local.py   # Local LLM support
    └── huggingface_api.py        # HuggingFace model browser
```

### Target Structure (tldw_server)
```
/tldw_Server_API/
├── app/
│   ├── api/v1/
│   │   ├── endpoints/         # API endpoints
│   │   └── schemas/           # Pydantic models
│   └── core/
│       ├── Character_Chat/    # Existing character module
│       ├── Chat/             # Existing chat module
│       ├── DB_Management/     # Database operations
│       └── LLM_Calls/        # Existing LLM integrations
```

## Architecture Comparison

### Source Architecture (Reference Modules in `/REFERENCE/`)
- **Type**: Single-user TUI application
- **State Management**: Thread-local storage, global singletons
- **Storage**: Direct file system access with user home directories
- **Sessions**: Persistent UI tabs with in-memory state
- **Authentication**: None (single user assumed)
- **Concurrency**: Thread-based for UI operations

### Target Architecture (tldw_server in `/tldw_Server_API/`)
- **Type**: Multi-user API server (FastAPI)
- **State Management**: Stateless REST API with database persistence
- **Storage**: User-isolated database records with user_id/client_id
- **Sessions**: Token-based authentication with session management
- **Authentication**: JWT-based with user isolation (see `/app/core/AuthNZ/`)
- **Concurrency**: Async/await with multiple workers

## Module Analysis

### 1. Character_Chat Module (`/REFERENCE/Character_Chat/`)

#### Features to Migrate

##### World Book Manager ✅
**Source File:** `/REFERENCE/Character_Chat/world_book_manager.py`
**Current Implementation:**
- Single global instance managing world books/lorebooks
- Direct database access without user context
- Shared across all operations

**Target Location:** `/tldw_Server_API/app/core/Character_Chat/world_book_manager.py` (new)
**Required Adaptations:**
- Add `user_id` column to world book tables
- Pass user context through all database operations
- Implement user-based access control
- Create user-scoped CRUD operations
- No global instances - instantiate per request

##### Chat Dictionary System ✅
**Source File:** `/REFERENCE/Character_Chat/Chat_Dictionary_Lib.py`
**Current Implementation:**
- In-memory pattern matching with regex support
- Thread-local state for processing
- Global dictionary storage

**Target Location:** `/tldw_Server_API/app/core/Character_Chat/chat_dictionary.py` (new)
**Required Adaptations:**
- Store dictionaries in database with user association
- Implement per-user caching layer (Redis/in-memory with TTL)
- Pass user context through processing pipeline
- Per-user token budgets and limits
- Request-scoped processing instances

##### World Info Processor ⚠️
**Source File:** `/REFERENCE/Character_Chat/world_info_processor.py`
**Current Implementation:**
- Processes world info with shared state
- Caches compiled patterns globally

**Target Location:** `/tldw_Server_API/app/core/Character_Chat/world_info_processor.py` (new)
**Required Adaptations:**
- Instantiate per-request, not as singleton
- Pass user context explicitly through all methods
- No shared state between requests
- Cache patterns per-user with expiration

#### Features NOT Compatible
- Thread-local storage patterns → Use request context instead
- Global character defaults → Implement per-user defaults
- Direct file I/O for images → Use user-isolated object storage

### 2. Chat Module (`/REFERENCE/Chat/`)

#### Features to Migrate

##### Document Generator ✅
**Source File:** `/REFERENCE/Chat/document_generator.py`
**Current Implementation:**
- Direct LLM API calls
- Hardcoded file paths for templates
- Single-user prompt configurations

**Target Location:** `/tldw_Server_API/app/core/Chat/document_generator.py` (new)
**Required Adaptations:**
- Instantiate service per-request with user context
- User-specific prompt configurations from database
- Store generated documents with user_id
- Implement per-user rate limiting
- Track usage for billing/quotas

##### Chat Session Models ⚠️
**Source File:** `/REFERENCE/Chat/chat_models.py`
**Current Implementation:**
- In-memory session state
- Thread-local storage for context
- Persistent worker references

**Target Location:** `/tldw_Server_API/app/api/v1/schemas/chat_session_schemas.py` (new)
**Required Adaptations:**
- Store sessions in database/Redis with user_id
- Use session tokens instead of tab_ids
- Implement session expiry and cleanup
- Remove worker references (stateless API)
- Add session validation middleware

#### Features NOT Compatible

##### Tab State Manager ❌
**Source Files:** `/REFERENCE/Chat/tabs/tab_state_manager.py`, `/REFERENCE/Chat/tabs/tab_context.py`
**Why it doesn't work:**
- Uses thread-local storage (breaks across API requests)
- Assumes persistent UI tabs (API is stateless)
- Global active tab tracking (meaningless in multi-user)
- Holds references to UI widgets

**Alternative Solution:**
- Implement session management with Redis/database
- Use session tokens for state tracking
- WebSockets for real-time features if needed

### 3. Chatbooks Module (`/REFERENCE/Chatbooks/`)

#### Features to Migrate

##### Chatbook Models & Creation ✅
**Source Files:** 
- `/REFERENCE/Chatbooks/chatbook_models.py` - Data structures
- `/REFERENCE/Chatbooks/chatbook_creator.py` - Export logic

**Current Implementation:**
- File system based with home directory access
- Temporary directories in user home
- Direct zip file creation

**Target Location:** `/tldw_Server_API/app/core/Chatbooks/` (new directory)
**Required Adaptations:**
- User-isolated export directories (`/tmp/{user_id}/chatbooks/`)
- Add user_id to all chatbook metadata
- Implement access control for import/export
- Use background jobs for large exports
- Store exports in object storage (S3/MinIO)

##### Chatbook Import & Conflict Resolution ✅
**Source Files:**
- `/REFERENCE/Chatbooks/chatbook_importer.py` - Import logic
- `/REFERENCE/Chatbooks/conflict_resolver.py` - Conflict handling

**Current Implementation:**
- Single-user conflict handling
- Direct database modifications

**Target Location:** `/tldw_Server_API/app/core/Chatbooks/` (new files)
**Required Adaptations:**
- User-scoped conflict detection
- Prevent cross-user data leakage
- Audit logging of all imports/exports
- Validate user permissions for imported content
- Sanitize paths and content

### 4. LLM_Calls Module (`/REFERENCE/LLM_Calls/`)

#### Features to Migrate

##### HuggingFace API Integration ✅
**Source File:** `/REFERENCE/LLM_Calls/huggingface_api.py`
**Current Implementation:**
- Downloads models to local filesystem
- Stores in user home directory
- No access control

**Target Location:** `/tldw_Server_API/app/core/LLM_Calls/huggingface_api.py` (new)
**Required Adaptations:**
- Central model cache with access control
- Admin-only download endpoints
- Track user access permissions for models
- Implement per-user rate limiting
- Model usage quotas and billing

## Implementation Plan

### Phase 1: Core Features with User Isolation (Week 1-2)

#### 1.1 Chat Dictionary System
**Source:** `/REFERENCE/Character_Chat/Chat_Dictionary_Lib.py`
**Target:** `/tldw_Server_API/app/core/Character_Chat/chat_dictionary.py`

```python
# Database Schema (add to ChaChaNotes_DB.py)
CREATE TABLE chat_dictionaries (
    id INTEGER PRIMARY KEY,
    user_id INTEGER NOT NULL,
    name TEXT NOT NULL,
    description TEXT,
    is_active BOOLEAN DEFAULT TRUE,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (user_id) REFERENCES users(id),
    UNIQUE(user_id, name)
);

CREATE TABLE dictionary_entries (
    id INTEGER PRIMARY KEY,
    dictionary_id INTEGER NOT NULL,
    key_pattern TEXT NOT NULL,
    replacement TEXT NOT NULL,
    is_regex BOOLEAN DEFAULT FALSE,
    probability INTEGER DEFAULT 100,
    max_replacements INTEGER DEFAULT 1,
    metadata JSON,
    FOREIGN KEY (dictionary_id) REFERENCES chat_dictionaries(id)
);
```

**API Endpoints:** (add to `/tldw_Server_API/app/api/v1/endpoints/`)
- `POST /api/v1/chat/dictionaries` - Create dictionary
- `GET /api/v1/chat/dictionaries` - List user's dictionaries
- `PUT /api/v1/chat/dictionaries/{id}` - Update dictionary
- `DELETE /api/v1/chat/dictionaries/{id}` - Delete dictionary
- `POST /api/v1/chat/dictionaries/{id}/process` - Process text

#### 1.2 World Book Manager
**Source:** `/REFERENCE/Character_Chat/world_book_manager.py`
**Target:** `/tldw_Server_API/app/core/Character_Chat/world_book_manager.py`

```python
# Database Schema (add to ChaChaNotes_DB.py)
CREATE TABLE world_books (
    id INTEGER PRIMARY KEY,
    user_id INTEGER NOT NULL,
    name TEXT NOT NULL,
    description TEXT,
    scan_depth INTEGER DEFAULT 3,
    token_budget INTEGER DEFAULT 500,
    recursive_scanning BOOLEAN DEFAULT FALSE,
    enabled BOOLEAN DEFAULT TRUE,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (user_id) REFERENCES users(id),
    UNIQUE(user_id, name)
);

CREATE TABLE world_book_entries (
    id INTEGER PRIMARY KEY,
    world_book_id INTEGER NOT NULL,
    keywords TEXT NOT NULL,
    content TEXT NOT NULL,
    priority INTEGER DEFAULT 0,
    enabled BOOLEAN DEFAULT TRUE,
    metadata JSON,
    FOREIGN KEY (world_book_id) REFERENCES world_books(id)
);
```

**API Endpoints:** (add to `/tldw_Server_API/app/api/v1/endpoints/characters.py`)
- `POST /api/v1/characters/world-books` - Create world book
- `GET /api/v1/characters/world-books` - List user's world books
- `PUT /api/v1/characters/world-books/{id}` - Update world book
- `DELETE /api/v1/characters/world-books/{id}` - Delete world book
- `POST /api/v1/characters/world-books/{id}/entries` - Add entry

#### 1.3 Document Generator
**Source:** `/REFERENCE/Chat/document_generator.py`
**Target:** `/tldw_Server_API/app/core/Chat/document_generator.py`

**Implementation:**
- Stateless service class instantiated per request
- User-specific configurations from database
- Background job support for large documents

**API Endpoints:** (add to `/tldw_Server_API/app/api/v1/endpoints/chat.py`)
- `POST /api/v1/chat/documents/timeline` - Generate timeline
- `POST /api/v1/chat/documents/study-guide` - Generate study guide
- `POST /api/v1/chat/documents/briefing` - Generate briefing
- `GET /api/v1/chat/documents/{job_id}/status` - Check generation status

### Phase 2: Adapted Features (Week 3-4)

#### 2.1 Chatbooks with User Isolation
**Source:** `/REFERENCE/Chatbooks/` (all files)
**Target:** `/tldw_Server_API/app/core/Chatbooks/` (new directory)

**Implementation:**
- User-isolated temporary directories
- Background job processing with Celery/RQ
- S3/MinIO for storage
- Signed URLs for downloads

**API Endpoints:** (new file: `/tldw_Server_API/app/api/v1/endpoints/chatbooks.py`)
- `POST /api/v1/chatbooks/export` - Start export job
- `GET /api/v1/chatbooks/export/{job_id}` - Check export status
- `POST /api/v1/chatbooks/import` - Upload and import
- `GET /api/v1/chatbooks` - List user's chatbooks

#### 2.2 Session Management (Replacing Tabs)
**Replaces:** `/REFERENCE/Chat/tabs/` (incompatible with stateless API)
**Target:** `/tldw_Server_API/app/core/Chat/session_manager.py` (new)

**Implementation:**
- Redis-backed session storage
- Session tokens in API responses
- Automatic cleanup of expired sessions
- WebSocket support for streaming

**API Endpoints:** (add to `/tldw_Server_API/app/api/v1/endpoints/chat.py`)
- `POST /api/v1/chat/sessions` - Create session
- `GET /api/v1/chat/sessions/{id}` - Get session state
- `PUT /api/v1/chat/sessions/{id}` - Update session
- `DELETE /api/v1/chat/sessions/{id}` - End session
- `WS /api/v1/chat/sessions/{id}/stream` - WebSocket streaming

### Phase 3: Shared Resources (Week 5)

#### 3.1 HuggingFace Model Management
**Source:** `/REFERENCE/LLM_Calls/huggingface_api.py`
**Target:** `/tldw_Server_API/app/core/LLM_Calls/huggingface_api.py`

**Implementation:**
- Admin-only download endpoints
- Shared model cache with permissions
- User quotas and usage tracking

**API Endpoints:** (add to `/tldw_Server_API/app/api/v1/endpoints/models.py` - new)
- `POST /api/v1/admin/models/download` - Download model (admin)
- `GET /api/v1/models` - List available models
- `POST /api/v1/models/{id}/access` - Request model access

## Multi-User Considerations

### Authentication & Authorization
- Every endpoint must validate user authentication
- Add user_id to all database operations
- Implement row-level security where applicable
- Use dependency injection for user context

### State Management
```python
# Example: Request-scoped service (in endpoint files)
# Location: /tldw_Server_API/app/api/v1/endpoints/chat.py
async def get_dictionary_service(
    user: User = Depends(get_current_user),  # from AuthNZ/User_DB_Handling.py
    db: Database = Depends(get_db)  # from API_Deps/DB_Deps.py
) -> ChatDictionaryService:
    return ChatDictionaryService(db, user.id)
```

### Resource Isolation
- User-specific paths: `/data/users/{user_id}/`
- Separate temp directories: `/tmp/tldw/{user_id}/`
- No shared mutable state between requests
- Use connection pooling for databases

### Security Measures
- Input validation on all endpoints
- Path traversal prevention
- Rate limiting per user
- Audit logging for sensitive operations
- CORS configuration for API access

### Scalability Design
- Stateless API design for horizontal scaling
- Database connection pooling
- Redis for caching and sessions
- Background jobs for heavy operations
- CDN/object storage for static files

## Testing Strategy

### Test File Locations
All tests should be added to `/tldw_Server_API/tests/` following the existing structure:
- `/tests/Character_Chat/` - World book and dictionary tests
- `/tests/Chat/` - Document generator and session tests
- `/tests/Chatbooks/` - Import/export tests (new directory)
- `/tests/LLM_Calls/` - HuggingFace API tests

### Unit Tests
- Test each service with mocked dependencies
- Verify user isolation in all operations
- Test permission checks
- Use existing fixtures from `/tests/conftest.py`

### Integration Tests
- Multi-user scenarios
- Concurrent request handling
- Session management across requests
- Rate limiting verification

### Security Tests
- Attempt cross-user data access
- Path traversal attempts
- SQL injection tests
- Token validation

## Migration Timeline

### Week 1-2: Phase 1 Implementation
- Chat Dictionary System
- World Book Manager
- Document Generator
- Unit tests

### Week 3-4: Phase 2 Implementation
- Chatbooks adaptation
- Session management
- Integration tests

### Week 5: Phase 3 & Testing
- HuggingFace integration
- Security testing
- Performance testing
- Documentation

### Week 6: Deployment
- Staging deployment
- User acceptance testing
- Production deployment
- Monitoring setup

## Risk Mitigation

### Technical Risks
- **Risk**: State management complexity
  - **Mitigation**: Use proven patterns (Redis sessions)
  
- **Risk**: Performance degradation with user isolation
  - **Mitigation**: Implement caching, use indexes

- **Risk**: Data migration errors
  - **Mitigation**: Comprehensive testing, rollback plans

### Security Risks
- **Risk**: Cross-user data leakage
  - **Mitigation**: Strict user_id validation, security tests

- **Risk**: Resource exhaustion attacks
  - **Mitigation**: Rate limiting, quotas, monitoring

## Success Criteria

1. All migrated features support multiple concurrent users
2. No cross-user data leakage in security tests
3. API response times remain under 200ms for 95th percentile
4. Zero data loss during migration
5. All features have comprehensive test coverage (>80%)
6. Documentation updated for all new endpoints

## File Migration Checklist

### Phase 1 Files to Create/Modify
- [ ] `/tldw_Server_API/app/core/Character_Chat/chat_dictionary.py` (new - from `/REFERENCE/Character_Chat/Chat_Dictionary_Lib.py`)
- [ ] `/tldw_Server_API/app/core/Character_Chat/world_book_manager.py` (new - from `/REFERENCE/Character_Chat/world_book_manager.py`)
- [ ] `/tldw_Server_API/app/core/Character_Chat/world_info_processor.py` (new - from `/REFERENCE/Character_Chat/world_info_processor.py`)
- [ ] `/tldw_Server_API/app/core/Chat/document_generator.py` (new - from `/REFERENCE/Chat/document_generator.py`)
- [ ] `/tldw_Server_API/app/core/DB_Management/ChaChaNotes_DB.py` (modify - add new tables)
- [ ] `/tldw_Server_API/app/api/v1/endpoints/characters.py` (modify - add world book endpoints)
- [ ] `/tldw_Server_API/app/api/v1/endpoints/chat.py` (modify - add dictionary and document endpoints)
- [ ] `/tldw_Server_API/app/api/v1/schemas/character_schemas.py` (modify - add world book schemas)
- [ ] `/tldw_Server_API/app/api/v1/schemas/chat_schemas.py` (modify - add dictionary schemas)

### Phase 2 Files to Create
- [ ] `/tldw_Server_API/app/core/Chatbooks/` (new directory)
- [ ] `/tldw_Server_API/app/core/Chatbooks/chatbook_models.py` (from `/REFERENCE/Chatbooks/chatbook_models.py`)
- [ ] `/tldw_Server_API/app/core/Chatbooks/chatbook_creator.py` (from `/REFERENCE/Chatbooks/chatbook_creator.py`)
- [ ] `/tldw_Server_API/app/core/Chatbooks/chatbook_importer.py` (from `/REFERENCE/Chatbooks/chatbook_importer.py`)
- [ ] `/tldw_Server_API/app/core/Chatbooks/conflict_resolver.py` (from `/REFERENCE/Chatbooks/conflict_resolver.py`)
- [ ] `/tldw_Server_API/app/core/Chat/session_manager.py` (new - replacement for tabs)
- [ ] `/tldw_Server_API/app/api/v1/endpoints/chatbooks.py` (new)
- [ ] `/tldw_Server_API/app/api/v1/schemas/chatbook_schemas.py` (new)
- [ ] `/tldw_Server_API/app/api/v1/schemas/chat_session_schemas.py` (new)

### Phase 3 Files to Create
- [ ] `/tldw_Server_API/app/core/LLM_Calls/huggingface_api.py` (from `/REFERENCE/LLM_Calls/huggingface_api.py`)
- [ ] `/tldw_Server_API/app/api/v1/endpoints/models.py` (new)
- [ ] `/tldw_Server_API/app/api/v1/schemas/model_schemas.py` (new)

### Test Files to Create
- [ ] `/tldw_Server_API/tests/Character_Chat/test_world_books.py`
- [ ] `/tldw_Server_API/tests/Character_Chat/test_chat_dictionary.py`
- [ ] `/tldw_Server_API/tests/Chat/test_document_generator.py`
- [ ] `/tldw_Server_API/tests/Chat/test_session_manager.py`
- [ ] `/tldw_Server_API/tests/Chatbooks/` (new directory)
- [ ] `/tldw_Server_API/tests/Chatbooks/test_chatbook_creator.py`
- [ ] `/tldw_Server_API/tests/Chatbooks/test_chatbook_importer.py`
- [ ] `/tldw_Server_API/tests/LLM_Calls/test_huggingface_api.py`

## Conclusion

This migration plan transforms valuable single-user features from the `/REFERENCE/` folder into robust multi-user capabilities for the tldw_server API. The plan provides specific file paths for both source and target locations, ensuring clear guidance for implementation. The phased approach allows for incremental delivery and testing, reducing risk and ensuring quality throughout the migration process.