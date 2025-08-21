# Audit Module Deprecation Notice

## Overview
As of this update, all legacy audit modules have been removed and replaced with a unified audit service.

## Removed Modules
The following deprecated modules have been removed from the codebase:

1. **`app/core/Audit/audit_logger.py`** - General audit logging (REMOVED)
2. **`app/core/RAG/rag_audit_logger.py`** - RAG-specific audit logging (REMOVED)
3. **`app/core/Evaluations/audit_logger.py`** - Evaluations audit logging (REMOVED)
4. **`app/services/audit_service.py`** - Auth/admin audit service (REMOVED)

## Replacement
All audit functionality is now consolidated in:
- **`app/core/Audit/unified_audit_service.py`** - Unified audit service for all operations

## Migration Guide

### Old Code (Deprecated):
```python
# Old imports - NO LONGER WORK
from tldw_Server_API.app.core.Audit.audit_logger import get_audit_logger, AuditEventType
from tldw_Server_API.app.services.audit_service import get_audit_service, AuditAction
from tldw_Server_API.app.core.RAG.rag_audit_logger import get_rag_audit_logger
from tldw_Server_API.app.core.Evaluations.audit_logger import audit_logger

# Old usage
audit_logger = get_audit_logger()
await audit_logger.log_event(
    event_type=AuditEventType.AUTH_SUCCESS,
    action="login",
    user_id=user_id
)
```

### New Code (Current):
```python
# New unified imports
from tldw_Server_API.app.core.Audit.unified_audit_service import (
    get_unified_audit_service,
    AuditEventType,
    AuditContext
)

# New usage
audit_service = await get_unified_audit_service()

context = AuditContext(
    user_id=user_id,
    ip_address=request.client.host,
    session_id=session_id
)

await audit_service.log_event(
    event_type=AuditEventType.AUTH_LOGIN_SUCCESS,
    context=context,
    metadata={"browser": "Chrome"}
)
```

## Key Changes

### 1. Unified Event Types
All event types are now in a single enum `AuditEventType`:
- Authentication: `AUTH_LOGIN_SUCCESS`, `AUTH_LOGIN_FAILURE`, etc.
- Data operations: `DATA_READ`, `DATA_WRITE`, `DATA_DELETE`, etc.
- RAG operations: `RAG_SEARCH`, `RAG_RETRIEVAL`, `RAG_GENERATION`, etc.
- API operations: `API_REQUEST`, `API_RESPONSE`, `API_ERROR`, etc.

### 2. Context Object
All audit events now use an `AuditContext` object for tracking:
- `request_id` - Unique request identifier
- `correlation_id` - For tracking related events
- `session_id` - User session tracking
- `user_id` - User identifier
- `ip_address` - Client IP
- `user_agent` - Client user agent

### 3. Enhanced Features
The unified service provides:
- **PII Detection**: Automatic detection and redaction of 11+ PII types
- **Risk Scoring**: Intelligent 0-100 risk scoring
- **Performance Metrics**: Built-in timing and cost tracking
- **Async Operations**: Full async/await support with connection pooling
- **Correlation Tracking**: Track related events across services

### 4. Database Changes
- Single unified database: `unified_audit.db`
- Comprehensive schema with all event types
- Daily statistics aggregation
- Configurable retention policies

## Files Updated
The following files have been updated to use the new unified audit service:
- `app/main.py` - Application startup/shutdown
- `app/api/v1/endpoints/chat.py` - Chat endpoint audit logging
- `app/api/v1/endpoints/register.py` - User registration audit
- `app/api/v1/endpoints/auth.py` - Authentication audit imports
- `tests/AuthNZ/conftest.py` - Test configuration
- `tests/AuthNZ/test_auth_comprehensive.py` - Auth tests

## Testing
All functionality is tested in:
- `tests/Audit/test_unified_audit_service.py`

Run tests with:
```bash
python -m pytest tldw_Server_API/tests/Audit/test_unified_audit_service.py -v
```

## Benefits of Migration
1. **Single source of truth** - One audit system for all operations
2. **Consistent logging** - Same format and fields across all events
3. **Better performance** - Async operations, batching, connection pooling
4. **Enhanced security** - PII detection, risk scoring, full hash storage
5. **Easier maintenance** - One codebase to maintain instead of four
6. **Better compliance** - Comprehensive audit trails with retention

## Support
For questions about the migration or issues with the new audit service, please refer to:
- `app/core/Audit/unified_audit_service.py` - Source code with documentation
- `app/core/Audit/AUDIT_IMPROVEMENTS_SUMMARY.md` - Detailed improvements documentation
- `tests/Audit/test_unified_audit_service.py` - Usage examples in tests

## Important Note
⚠️ **The old audit modules have been permanently removed. Any code still referencing them will fail. Please update immediately to use the unified audit service.**