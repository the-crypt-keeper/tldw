# Domain-Driven Design Refactoring Plan for tldw_server

## Executive Summary

This document outlines a comprehensive plan to refactor the tldw_server codebase from its current layered architecture to a domain-driven design (DDD) approach, inspired by the Polar project structure. This refactoring will improve maintainability, scalability, and team collaboration capabilities.

## Current vs. Target Architecture

### Current Structure (Layered Architecture)
```
tldw_Server_API/
├── app/
│   ├── api/v1/
│   │   ├── endpoints/       # All API endpoints grouped
│   │   ├── schemas/         # All Pydantic models grouped
│   │   └── API_Deps/        # Shared dependencies
│   └── core/
│       ├── DB_Management/   # Database operations
│       ├── LLM_Calls/       # LLM integrations
│       ├── RAG/             # RAG service
│       └── Embeddings/      # Embeddings logic
```

### Target Structure (Domain-Driven Design)
```
tldw_Server_API/
├── media/
│   ├── endpoints.py         # Media API routes
│   ├── schemas.py           # Media Pydantic models
│   ├── service.py           # Media business logic
│   ├── repository.py        # Media data access
│   └── __init__.py
├── chat/
│   ├── endpoints.py         # Chat API routes
│   ├── schemas.py           # Chat Pydantic models
│   ├── service.py           # Chat business logic
│   ├── repository.py        # Chat data access
│   └── __init__.py
├── rag/
│   ├── endpoints.py         # RAG API routes
│   ├── schemas.py           # RAG Pydantic models
│   ├── service.py           # RAG business logic
│   ├── repository.py        # RAG data access
│   └── __init__.py
├── embeddings/
│   ├── endpoints.py         # Embeddings API routes
│   ├── schemas.py           # Embeddings Pydantic models
│   ├── service.py           # Embeddings business logic
│   ├── providers/           # Provider-specific implementations
│   │   ├── openai.py
│   │   ├── cohere.py
│   │   └── huggingface.py
│   └── __init__.py
├── shared/
│   ├── database/            # Shared database utilities
│   ├── auth/                # Authentication/authorization
│   ├── config/              # Configuration management
│   └── exceptions/          # Shared exceptions
├── models/                  # SQLAlchemy/Database models
│   ├── media.py
│   ├── chat.py
│   ├── notes.py
│   └── base.py
└── main.py                  # FastAPI application

```

## Benefits of This Refactoring

### 1. **Improved Maintainability**
- Each domain is self-contained with clear boundaries
- Changes in one domain don't cascade to others
- Easier to locate and fix bugs

### 2. **Better Scalability**
- Domains can grow independently
- Easy to add new features within a domain
- Microservice-ready architecture

### 3. **Enhanced Team Collaboration**
- Different developers/teams can own different domains
- Reduced merge conflicts
- Clear ownership boundaries

### 4. **Simplified Testing**
- Domain-specific test suites
- Easier to mock dependencies
- Better test isolation

### 5. **Clearer Business Logic**
- Business logic is colocated with its domain
- Service layer pattern enforces clean separation
- Repository pattern for data access

## Implementation Phases

### Phase 0: Preparation (Day 1)
- [ ] Create this refactoring plan document
- [ ] Set up git branch for refactoring: `refactor/domain-driven-design`
- [ ] Create backup of current structure
- [ ] Document all current API endpoints and their dependencies
- [ ] Identify shared utilities and cross-domain dependencies

### Phase 1: Core Infrastructure (Days 2-3)
- [ ] Create new directory structure skeleton
- [ ] Set up `shared/` module with:
  - Database connection management
  - Authentication/authorization utilities
  - Configuration management
  - Base exceptions
  - Common decorators and middleware
- [ ] Create `models/` directory and base model classes
- [ ] Set up new `main.py` with router registration framework

### Phase 2: Media Domain Migration (Days 4-5)
- [ ] Create `media/` domain structure
- [ ] Extract media endpoints from `app/api/v1/endpoints/media.py`
- [ ] Extract media schemas from various schema files
- [ ] Create `media/service.py` from core business logic
- [ ] Create `media/repository.py` for data access
- [ ] Extract media-specific utilities
- [ ] Write integration tests for media domain
- [ ] Update imports and test

### Phase 3: Chat Domain Migration (Days 6-7)
- [ ] Create `chat/` domain structure
- [ ] Extract chat endpoints and character endpoints
- [ ] Consolidate chat-related schemas
- [ ] Create chat service layer
- [ ] Create chat repository layer
- [ ] Handle character cards as subdomain
- [ ] Write integration tests for chat domain
- [ ] Update imports and test

### Phase 4: RAG Domain Migration (Days 8-9)
- [ ] Create `rag/` domain structure
- [ ] Migrate `rag_v2.py` endpoints
- [ ] Migrate `rag_schemas_simple.py`
- [ ] Extract RAG service from `core/RAG/`
- [ ] Create RAG repository for data access
- [ ] Handle RAG configuration
- [ ] Write integration tests for RAG domain
- [ ] Update imports and test

### Phase 5: Embeddings Domain Migration (Days 10-11)
- [ ] Create `embeddings/` domain structure
- [ ] Migrate `embeddings_v4.py` endpoints
- [ ] Create provider subdirectory structure
- [ ] Extract provider-specific logic into separate modules
- [ ] Create embeddings service layer
- [ ] Handle caching and batch processing
- [ ] Write integration tests for embeddings domain
- [ ] Update imports and test

### Phase 6: Notes Domain Migration (Day 12)
- [ ] Create `notes/` domain structure
- [ ] Extract notes endpoints
- [ ] Extract notes schemas
- [ ] Create notes service and repository
- [ ] Write integration tests for notes domain
- [ ] Update imports and test

### Phase 7: Prompts Domain Migration (Day 13)
- [ ] Create `prompts/` domain structure
- [ ] Extract prompts endpoints
- [ ] Extract prompts schemas
- [ ] Create prompts service and repository
- [ ] Write integration tests for prompts domain
- [ ] Update imports and test

### Phase 8: Additional Domains (Days 14-15)
- [ ] Create `audio/` domain
- [ ] Create `research/` domain
- [ ] Create `tools/` domain
- [ ] Create `sync/` domain
- [ ] Migrate remaining endpoints

### Phase 9: Cleanup and Optimization (Days 16-17)
- [ ] Remove old directory structure
- [ ] Update all import statements
- [ ] Fix circular dependencies
- [ ] Optimize shared utilities
- [ ] Update configuration files
- [ ] Update Docker configurations

### Phase 10: Testing and Validation (Days 18-19)
- [ ] Run full test suite
- [ ] Fix any broken tests
- [ ] Performance testing
- [ ] API compatibility testing
- [ ] Load testing
- [ ] Security audit

### Phase 11: Documentation Update (Day 20)
- [ ] Update README.md
- [ ] Update CLAUDE.md with new structure
- [ ] Update API documentation
- [ ] Create domain-specific documentation
- [ ] Update deployment guides
- [ ] Create migration guide for developers

## Migration Strategy for Each Domain

### Standard Migration Pattern

For each domain, follow this pattern:

1. **Create Domain Structure**
```python
# domain/__init__.py
from .endpoints import router
from .service import DomainService
from .repository import DomainRepository

__all__ = ["router", "DomainService", "DomainRepository"]
```

2. **Extract Endpoints**
```python
# domain/endpoints.py
from fastapi import APIRouter, Depends, HTTPException
from .schemas import RequestModel, ResponseModel
from .service import DomainService

router = APIRouter()

@router.post("/action")
async def action(
    request: RequestModel,
    service: DomainService = Depends(get_domain_service)
) -> ResponseModel:
    return await service.perform_action(request)
```

3. **Define Schemas**
```python
# domain/schemas.py
from pydantic import BaseModel, Field
from typing import Optional, List

class RequestModel(BaseModel):
    field: str = Field(..., description="Field description")
    
class ResponseModel(BaseModel):
    result: str
    metadata: Optional[dict] = None
```

4. **Implement Service Layer**
```python
# domain/service.py
from .repository import DomainRepository
from .schemas import RequestModel, ResponseModel

class DomainService:
    def __init__(self, repository: DomainRepository):
        self.repository = repository
    
    async def perform_action(self, request: RequestModel) -> ResponseModel:
        # Business logic here
        data = await self.repository.get_data(request.field)
        # Process data
        return ResponseModel(result=processed_data)
```

5. **Implement Repository Layer**
```python
# domain/repository.py
from sqlalchemy.ext.asyncio import AsyncSession
from typing import Optional

class DomainRepository:
    def __init__(self, session: AsyncSession):
        self.session = session
    
    async def get_data(self, field: str) -> Optional[dict]:
        # Database operations here
        pass
```

## Dependency Management

### Shared Dependencies
Create a central dependency injection module:

```python
# shared/dependencies.py
from fastapi import Depends
from sqlalchemy.ext.asyncio import AsyncSession
from .database import get_session
from .auth import get_current_user

async def get_base_dependencies(
    session: AsyncSession = Depends(get_session),
    user: User = Depends(get_current_user)
):
    return {
        "session": session,
        "user": user
    }
```

### Domain-Specific Dependencies
Each domain can extend base dependencies:

```python
# media/dependencies.py
from shared.dependencies import get_base_dependencies
from .service import MediaService
from .repository import MediaRepository

async def get_media_service(
    deps: dict = Depends(get_base_dependencies)
) -> MediaService:
    repository = MediaRepository(deps["session"])
    return MediaService(repository, deps["user"])
```

## Router Registration

Update main.py to use domain routers:

```python
# main.py
from fastapi import FastAPI
from media import router as media_router
from chat import router as chat_router
from rag import router as rag_router
from embeddings import router as embeddings_router

app = FastAPI(title="tldw API", version="0.1.0")

# Register domain routers
app.include_router(media_router, prefix="/api/v1/media", tags=["media"])
app.include_router(chat_router, prefix="/api/v1/chat", tags=["chat"])
app.include_router(rag_router, prefix="/api/v1/rag", tags=["rag"])
app.include_router(embeddings_router, prefix="/api/v1/embeddings", tags=["embeddings"])
```

## Testing Strategy

### Domain-Specific Tests
```
tests/
├── media/
│   ├── test_endpoints.py
│   ├── test_service.py
│   └── test_repository.py
├── chat/
│   ├── test_endpoints.py
│   ├── test_service.py
│   └── test_repository.py
└── shared/
    └── test_dependencies.py
```

### Test Isolation
Each domain should have:
- Unit tests for service logic
- Integration tests for endpoints
- Repository tests with test database
- Mock tests for external dependencies

## Risk Mitigation

### Potential Risks and Mitigation Strategies

1. **Breaking Changes**
   - Mitigation: Maintain API compatibility layer during transition
   - Create facade endpoints that redirect to new structure

2. **Import Errors**
   - Mitigation: Gradual migration with compatibility imports
   - Use `__init__.py` files to maintain backward compatibility

3. **Database Migration Issues**
   - Mitigation: No database schema changes, only code reorganization
   - Test thoroughly with production data copy

4. **Performance Regression**
   - Mitigation: Benchmark before and after each phase
   - Profile critical paths

5. **Team Disruption**
   - Mitigation: Migrate one domain at a time
   - Maintain clear communication about changes

## Success Criteria

The refactoring will be considered successful when:

1. **All tests pass** - 100% of existing tests still pass
2. **API compatibility** - No breaking changes to external API
3. **Performance maintained** - No degradation in response times
4. **Code coverage** - Maintain or improve test coverage
5. **Documentation complete** - All domains documented
6. **Team trained** - All developers understand new structure

## Rollback Plan

If issues arise:

1. **Git reversion** - All changes in feature branch
2. **Staged rollout** - Deploy to staging first
3. **Feature flags** - Use flags to switch between old/new
4. **Gradual migration** - Keep both structures temporarily
5. **Database backup** - Full backup before deployment

## Timeline

- **Total Duration**: 20 working days (4 weeks)
- **Buffer Time**: Additional 5 days for unexpected issues
- **Review Checkpoints**: After each phase
- **Go/No-Go Decision**: Day 15

## Next Steps

1. Review and approve this plan
2. Create feature branch
3. Set up tracking dashboard
4. Begin Phase 0 preparation
5. Schedule daily standup for progress tracking

---

*This refactoring will transform tldw_server into a more maintainable, scalable, and developer-friendly codebase following domain-driven design principles.*