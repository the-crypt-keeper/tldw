# User Registration Module Implementation Plan - Revised

## Overview
Implement a comprehensive user registration and management system for multi-user instances with:
1. Root user auto-generation on first run
2. Manual user registration by admins
3. UUID-based registration codes with usage limits
4. Full RBAC foundation for future expansion
5. Session management with JWT tokens
6. User data isolation and archival on deletion

## Core Requirements

### Root User Management
- **Auto-generation**: On first multi-user startup, generate root user with strong random password
- **Credential Storage**: Save root credentials in `.env` file in application directory (added to .gitignore)
- **Format**: 
  ```env
  ROOT_USER_EMAIL=root@localhost
  ROOT_USER_PASSWORD=<generated-32-char-password>
  ROOT_USER_CREATED_AT=<timestamp>
  ```
- **Security**: File permissions set to 600 (owner read/write only)
- **User ID**: Root user will have ID 1 (first user in database)

### User Database Design (Full Implementation)

#### Database Configuration
- **WAL Mode**: Enable Write-Ahead Logging for better concurrency
- **Connection Management**: Use existing MediaDatabase connection patterns
- **Retry Logic**: Exponential backoff for database locks
- **Queue System**: Use FastAPI BackgroundTasks for registration queue
- **Prepared Statements**: Use prepared statements for all queries

#### Users Table
```sql
CREATE TABLE users (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    username TEXT UNIQUE NOT NULL,
    email TEXT UNIQUE NOT NULL,
    hashed_password TEXT NOT NULL,
    
    -- Role & Permissions (RBAC Foundation)
    role TEXT NOT NULL DEFAULT 'user', -- root, admin, moderator, user
    permissions JSON, -- Future: granular permissions
    
    -- Status Fields
    is_active BOOLEAN DEFAULT TRUE,
    is_verified BOOLEAN DEFAULT FALSE,
    is_deleted BOOLEAN DEFAULT FALSE,
    must_change_password BOOLEAN DEFAULT FALSE,
    
    -- Account Security
    failed_login_attempts INTEGER DEFAULT 0,
    locked_until TIMESTAMP,
    last_failed_login TIMESTAMP,
    
    -- Password Reset
    password_reset_token TEXT,
    password_reset_expires TIMESTAMP,
    
    -- Metadata
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    last_login TIMESTAMP,
    created_by INTEGER REFERENCES users(id),
    deleted_at TIMESTAMP,
    deleted_by INTEGER REFERENCES users(id),
    
    -- User Settings
    preferences JSON, -- UI preferences, default models, etc.
    storage_quota_mb INTEGER DEFAULT 10240, -- 10GB default
    storage_used_mb INTEGER DEFAULT 0,
    
    -- API Key Compatibility
    api_key TEXT UNIQUE, -- For backwards compatibility
    api_key_created_at TIMESTAMP,
    
    -- Indices
    INDEX idx_username (username),
    INDEX idx_email (email),
    INDEX idx_role (role),
    INDEX idx_is_active (is_active),
    INDEX idx_api_key (api_key)
);
```

#### Registration Codes Table
```sql
CREATE TABLE registration_codes (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    code TEXT UNIQUE NOT NULL, -- UUID format
    
    -- Usage Limits
    max_uses INTEGER NOT NULL DEFAULT 1,
    current_uses INTEGER DEFAULT 0,
    
    -- Validity
    expires_at TIMESTAMP NOT NULL,
    is_active BOOLEAN DEFAULT TRUE,
    
    -- Metadata
    created_by INTEGER NOT NULL REFERENCES users(id),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    
    -- Usage Tracking
    used_by JSON, -- Array of {user_id, used_at, ip_address}
    
    -- Optional Constraints
    allowed_email_domain TEXT, -- e.g., "@company.com"
    role_granted TEXT DEFAULT 'user', -- Role assigned to users who register with this code
    
    -- Notes
    description TEXT, -- Admin notes about code purpose
    
    INDEX idx_code (code),
    INDEX idx_expires_at (expires_at),
    INDEX idx_is_active (is_active)
);
```

#### Sessions Table (for JWT management)
```sql
CREATE TABLE sessions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id INTEGER NOT NULL REFERENCES users(id),
    
    -- Token Management (store only JTI, not full tokens)
    access_token_jti TEXT UNIQUE NOT NULL, -- JWT ID for access token
    refresh_token_jti TEXT UNIQUE NOT NULL, -- JWT ID for refresh token
    
    -- Session Info
    ip_address TEXT,
    user_agent TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    last_activity TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    expires_at TIMESTAMP NOT NULL,
    
    -- Status
    is_active BOOLEAN DEFAULT TRUE,
    revoked_at TIMESTAMP,
    revoked_by INTEGER REFERENCES users(id),
    revoke_reason TEXT,
    
    INDEX idx_user_id (user_id),
    INDEX idx_access_token_jti (access_token_jti),
    INDEX idx_refresh_token_jti (refresh_token_jti),
    INDEX idx_expires_at (expires_at),
    INDEX idx_is_active (is_active)
);
```

#### Audit Log Table (180-day retention)
```sql
CREATE TABLE audit_logs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id INTEGER REFERENCES users(id),
    action TEXT NOT NULL, -- login, logout, register, update_user, delete_user, etc.
    target_type TEXT, -- user, registration_code, session, etc.
    target_id INTEGER,
    
    -- Details
    ip_address TEXT,
    user_agent TEXT,
    request_method TEXT,
    request_path TEXT,
    
    -- Results
    success BOOLEAN NOT NULL,
    error_message TEXT,
    
    -- Metadata
    metadata JSON, -- Additional context
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    
    INDEX idx_user_id (user_id),
    INDEX idx_action (action),
    INDEX idx_created_at (created_at)
);
```

#### Security Log Table (separate security events)
```sql
CREATE TABLE security_logs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    event_type TEXT NOT NULL, -- failed_login, brute_force, privilege_escalation, etc.
    severity TEXT NOT NULL, -- info, warning, error, critical
    user_id INTEGER REFERENCES users(id),
    ip_address TEXT,
    user_agent TEXT,
    details JSON,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    
    INDEX idx_event_type (event_type),
    INDEX idx_severity (severity),
    INDEX idx_created_at (created_at)
);
```

### User Data Management

#### Directory Structure
```
user_databases/
├── 1/                          # Root user (ID: 1)
│   ├── media.db
│   ├── prompts.db
│   ├── uploads/
│   └── exports/
├── 2/                          # User ID: 2
│   ├── media.db
│   ├── prompts.db
│   ├── uploads/
│   └── exports/
└── deleted/                    # Archived deleted users
    └── DELETED-johndoe-2024-01-15-FILES.zip
```

#### User Deletion Process
1. Mark user as deleted in database (soft delete)
2. Revoke all active sessions
3. Create zip archive of user directory
4. Move archive to `user_databases/deleted/DELETED-<username>-<date>-FILES.zip`
5. Remove original user directory
6. Log deletion in audit log
7. GDPR compliance: Permanent deletion after retention period

#### User Directory Creation
- Create directory structure atomically using os.makedirs with exist_ok=False
- Set appropriate permissions (700 - owner only)
- Verify creation before confirming registration
- Use try/except for race condition handling

### Session Management Implementation

#### Token Strategy
- **Access Token**: Short-lived (15 minutes), contains user info, validated cryptographically
- **Refresh Token**: Long-lived (7 days), used to get new access token
- **Token Validation**: Validate JWTs cryptographically using signature, store only JTI in database
- **Token Rotation**: New refresh token issued on each refresh
- **Blacklisting**: Track revoked tokens by JTI in sessions table

#### Refresh Token Flow
1. Client sends refresh token to `/auth/refresh`
2. Server validates token signature and expiry cryptographically
3. Server checks if JTI exists and is active in sessions table
4. If valid, issue new access token and rotate refresh token
5. Update session record with new JTIs

### Registration Code Management

#### Code Generation
- **Format**: UUID v4 (e.g., `550e8400-e29b-41d4-a716-446655440000`)
- **Default Expiry**: 7 days
- **Default Uses**: 1
- **Batch Generation**: Max 100 codes per request
- **Rate Limiting**: Max 10 batch requests per hour per admin

#### Security Features
- **Rate Limiting**: Max 5 registration attempts per IP per hour
- **Code Validation**: Check expiry, usage count, domain restrictions
- **Atomic Usage**: Use database transaction with row locking
- **Audit Trail**: Log all registration attempts (success and failure)
- **Generic Errors**: Return "Invalid registration data" for all validation failures (prevent enumeration)
- **CSRF Protection**: Implement double-submit cookie pattern

### Account Security

#### Password Requirements
```python
PASSWORD_REQUIREMENTS = {
    'min_length': 12,
    'require_uppercase': True,
    'require_lowercase': True,
    'require_numbers': True,
    'require_special': True,
    'check_common_passwords': True,
    'check_user_info': True  # Password shouldn't contain username/email
}
```

#### Account Lockout
```python
LOCKOUT_POLICY = {
    'max_attempts': 5,
    'lockout_duration_minutes': 30,
    'reset_attempts_after_minutes': 60
}
```

#### Password Reset Flow
1. Admin initiates reset for user
2. Generate secure random token, store with expiry (1 hour)
3. Return temporary password to admin in API response
4. Set `must_change_password` flag on user
5. Force password change on next login

### API Endpoints Design (v1)

#### Public Authentication (`/api/v1/auth/`)
```yaml
POST /register:
  body:
    username: string
    email: string
    password: string
    registration_code: uuid
  responses:
    201: User created
    400: Invalid registration data  # Generic error
    429: Rate limited

POST /login:
  body:
    username: string
    password: string
  responses:
    200: { access_token, refresh_token }
    401: Invalid credentials
    423: Account locked
    429: Too many attempts

POST /login-api-key:  # Backwards compatibility
  headers:
    X-API-KEY: string
  responses:
    200: { access_token, refresh_token }
    401: Invalid API key

POST /refresh:
  body:
    refresh_token: string
  responses:
    200: { access_token, refresh_token }
    401: Invalid/expired token

POST /logout:
  headers:
    Authorization: Bearer <token>
  responses:
    200: Logged out
    401: Invalid token

GET /validate-code:
  query:
    code: uuid
  responses:
    200: { valid: true, role: string }
    404: Invalid code
```

#### User Management (`/api/v1/users/`)
```yaml
GET /me:
  responses:
    200: Current user info

PATCH /me:
  body:
    email?: string
    current_password: string  # Required for any change
    new_password?: string
    preferences?: object
  responses:
    200: Updated user

DELETE /me:
  body:
    password: string  # Confirm deletion
  responses:
    200: Account scheduled for deletion

GET /me/sessions:
  responses:
    200: List of active sessions

POST /me/change-password:
  body:
    current_password: string
    new_password: string
  responses:
    200: Password changed
    400: Password requirements not met

GET /me/api-key:  # Backwards compatibility
  responses:
    200: { api_key: string }
    404: No API key set

POST /me/api-key/generate:  # Backwards compatibility
  responses:
    200: { api_key: string }
```

#### Admin Endpoints (`/api/v1/admin/`)
```yaml
# User Management
GET /users:
  query:
    page?: number
    limit?: number
    role?: string
    is_active?: boolean
  responses:
    200: Paginated user list

POST /users:
  body:
    username: string
    email: string
    password: string
    role: string
    must_change_password?: boolean
  responses:
    201: User created

GET /users/{id}:
  responses:
    200: User details
    404: User not found

PATCH /users/{id}:
  body:
    role?: string
    is_active?: boolean
    storage_quota_mb?: number
    must_change_password?: boolean
  responses:
    200: User updated

DELETE /users/{id}:
  responses:
    200: User deleted (archived)

POST /users/{id}/reset-password:
  responses:
    200: { 
      temporary_password: string,
      expires_in_minutes: 60
    }

POST /users/{id}/revoke-sessions:
  responses:
    200: All sessions revoked

POST /users/{id}/unlock:
  responses:
    200: Account unlocked

# Registration Code Management
POST /registration-codes:
  body:
    count?: number (max: 100)
    max_uses?: number (default: 1)
    expires_in_days?: number (default: 7)
    role_granted?: string (default: 'user')
    allowed_email_domain?: string
    description?: string
  responses:
    201: { codes: [uuid] }
    429: Rate limited

GET /registration-codes:
  query:
    is_active?: boolean
    include_expired?: boolean
  responses:
    200: Code list with usage stats

GET /registration-codes/{code}:
  responses:
    200: Code details with usage history

DELETE /registration-codes/{code}:
  responses:
    200: Code deactivated

# System Management
GET /audit-logs:
  query:
    user_id?: number
    action?: string
    start_date?: datetime
    end_date?: datetime
  responses:
    200: Paginated audit logs

GET /security-logs:
  query:
    event_type?: string
    severity?: string
    start_date?: datetime
    end_date?: datetime
  responses:
    200: Paginated security logs

GET /system/stats:
  responses:
    200: {
      total_users: number,
      active_users: number,
      locked_accounts: number,
      total_storage_used_gb: number,
      active_sessions: number
    }
```

#### Health & Monitoring (`/api/v1/`)
```yaml
GET /health:
  responses:
    200: {
      status: "healthy",
      components: {
        database: "up",
        auth: "up",
        storage: "up"
      }
    }

GET /metrics:
  responses:
    200: Prometheus/OpenTelemetry format metrics
```

### Role-Based Access Control (RBAC) Foundation

#### Role Hierarchy
```python
ROLES = {
    'root': {
        'level': 100,
        'description': 'System root user',
        'inherits': ['admin']
    },
    'admin': {
        'level': 90,
        'description': 'System administrator',
        'inherits': ['moderator']
    },
    'moderator': {
        'level': 50,
        'description': 'Content moderator',
        'inherits': ['user']
    },
    'user': {
        'level': 10,
        'description': 'Regular user',
        'inherits': []
    }
}
```

#### Permission System (Future)
```python
# Foundation for future granular permissions
PERMISSIONS = {
    'users.create': ['root', 'admin'],
    'users.read': ['root', 'admin', 'moderator'],
    'users.update': ['root', 'admin'],
    'users.delete': ['root', 'admin'],
    'codes.create': ['root', 'admin'],
    'codes.read': ['root', 'admin'],
    'codes.delete': ['root', 'admin'],
    'media.moderate': ['root', 'admin', 'moderator'],
    # ... extensible for future features
}
```

### Storage Quota Enforcement

```python
class StorageQuotaService:
    def check_quota(user_id: int, size_bytes: int) -> bool:
        """Check if operation would exceed quota"""
        
    def update_usage(user_id: int, delta_bytes: int):
        """Update storage usage after operation"""
        
    def calculate_usage(user_id: int) -> int:
        """Recalculate actual usage from filesystem"""
        
    # Background job to periodically sync usage
    async def sync_all_usage():
        """Run daily to ensure accuracy"""
```

### Database Operations

#### Connection Management
```python
class UserDatabase:
    """Follow MediaDatabase pattern from existing codebase"""
    def __init__(self, db_path: str):
        self.db_path = db_path
        self._enable_wal_mode()
        
    def get_connection(self):
        """Get database connection with retry logic"""
        max_retries = 3
        for attempt in range(max_retries):
            try:
                conn = sqlite3.connect(self.db_path)
                conn.row_factory = sqlite3.Row
                return conn
            except sqlite3.OperationalError:
                if attempt == max_retries - 1:
                    raise
                time.sleep(2 ** attempt)
        
    def _enable_wal_mode(self):
        with self.get_connection() as conn:
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA busy_timeout=5000")
```

#### Prepared Statements
```python
PREPARED_STATEMENTS = {
    'get_user_by_username': "SELECT * FROM users WHERE username = ? AND is_deleted = FALSE",
    'increment_code_usage': "UPDATE registration_codes SET current_uses = current_uses + 1 WHERE code = ?",
    'check_session_valid': "SELECT * FROM sessions WHERE refresh_token_jti = ? AND is_active = TRUE",
    # ... more statements
}
```

#### Registration Queue (Using FastAPI BackgroundTasks)
```python
from fastapi import BackgroundTasks

async def register_endpoint(
    registration_data: RegistrationForm,
    background_tasks: BackgroundTasks,
    db: UserDatabase = Depends(get_user_db)
):
    # Validate immediately
    if not validate_registration_data(registration_data):
        raise HTTPException(400, "Invalid registration data")
    
    # Queue the actual registration
    background_tasks.add_task(
        process_registration,
        registration_data,
        db
    )
    
    return {"message": "Registration queued"}

async def process_registration(data: RegistrationForm, db: UserDatabase):
    """Process registration with retry logic"""
    for attempt in range(3):
        try:
            # Create user
            user_id = create_user(db, data)
            # Create user directories
            create_user_directories(user_id)
            # Send confirmation
            break
        except sqlite3.OperationalError:
            if attempt == 2:
                log_error("Registration failed after 3 attempts")
            await asyncio.sleep(2 ** attempt)
```

### Migration from Single-User Mode

#### Migration Script (`migrate_to_multiuser.py`)
```python
def migrate_single_to_multi():
    """
    1. Backup existing databases and config
    2. Create Users database with schema
    3. Generate root user with credentials
    4. Save credentials to .env file
    5. Migrate single-user data to root account (ID: 1)
    6. Update config.txt to multi-user mode
    7. Move existing data to user_databases/1/
    8. Verify migration success
    9. Create migration report
    10. Create rollback script
    """

def create_rollback_script():
    """
    Generate script to revert to single-user mode if needed
    - Restore original config
    - Move data back to original location
    - Remove users database
    """
```

### Audit Log Retention

```python
class AuditLogManager:
    RETENTION_DAYS = 180
    
    async def cleanup_old_logs(self):
        """Run daily to remove logs older than retention period"""
        cutoff_date = datetime.now() - timedelta(days=self.RETENTION_DAYS)
        
        # Archive old logs before deletion
        await self.archive_logs(cutoff_date)
        
        # Delete from database
        await self.delete_logs_before(cutoff_date)
        
    async def archive_logs(self, before_date: datetime):
        """Export logs to compressed JSON file"""
        filename = f"audit_archive_{before_date.isoformat()}.json.gz"
        # ... export and compress logic
```

### Testing Strategy

#### Unit Tests
```python
# test_user_db.py
- test_create_user()
- test_duplicate_user()
- test_user_deletion_with_archival()
- test_role_validation()
- test_password_requirements()
- test_account_lockout()
- test_api_key_compatibility()

# test_registration_codes.py
- test_code_generation()
- test_code_validation()
- test_code_expiry()
- test_concurrent_code_usage()
- test_batch_generation_limit()

# test_sessions.py
- test_token_creation()
- test_token_refresh()
- test_token_revocation()
- test_session_expiry()
- test_jwt_validation()

# test_storage_quota.py
- test_quota_check()
- test_usage_calculation()
- test_quota_enforcement()
```

#### Integration Tests
```python
# test_registration_flow.py
- test_full_registration_with_code()
- test_registration_rate_limiting()
- test_user_directory_creation()
- test_invalid_code_handling()
- test_registration_queue()

# test_admin_operations.py
- test_user_creation_by_admin()
- test_user_deletion_with_archival()
- test_batch_code_generation()
- test_audit_log_creation()
- test_password_reset_flow()

# test_migration.py
- test_single_to_multi_migration()
- test_data_preservation()
- test_root_user_creation()
- test_migration_rollback()
```

#### Stress Testing
```python
# test_load.py
- test_concurrent_registrations(users=100)
- test_concurrent_logins(users=1000)
- test_database_locks()
- test_queue_overflow()
```

#### Security Testing
```python
# test_security.py
- test_sql_injection()
- test_csrf_protection()
- test_rate_limiting()
- test_account_lockout()
- test_user_enumeration()
- test_password_brute_force()
```

#### Migration Testing
```python
# test_migration.py
- test_single_user_to_multi_user()
- test_data_integrity_after_migration()
- test_rollback_capability()
- test_api_key_migration()
```

#### Edge Cases Testing
```python
# test_edge_cases.py
- test_root_user_deletion_prevented()
- test_corrupt_jwt_handling()
- test_database_recovery_after_crash()
- test_directory_creation_failures()
- test_storage_quota_edge_cases()
```

### Error Handling Strategy

#### Error Codes
```python
ERROR_CODES = {
    'AUTH001': 'Invalid credentials',
    'AUTH002': 'Account locked',
    'AUTH003': 'Session expired',
    'REG001': 'Invalid registration code',
    'REG002': 'Code expired',
    'REG003': 'Code usage limit reached',
    'REG004': 'Email domain not allowed',
    'REG005': 'Invalid registration data',  # Generic for other registration errors
    'REG006': 'Registration temporarily unavailable',
    'USER001': 'User operation failed',  # Generic for user errors
    'USER002': 'Invalid user data',
    'USER003': 'Password requirements not met',
    'PERM001': 'Insufficient permissions',
    'RATE001': 'Rate limit exceeded',
    'QUOTA001': 'Storage quota exceeded'
}
```

#### Error Recovery
- Database transaction rollback on failure
- Directory cleanup on failed registration
- Cleanup jobs for orphaned records
- Retry logic with exponential backoff

### Monitoring & Logging

#### Key Metrics
- Registration attempts (success/failure)
- Login attempts (success/failure)  
- Active sessions count
- Storage usage by user
- API response times
- Error rates by endpoint
- Database lock contentions
- Queue depth and processing time

#### Log Levels
- **ERROR**: Authentication failures, system errors
- **WARNING**: Rate limit hits, invalid codes
- **INFO**: Successful registrations, logins, user actions
- **DEBUG**: Token validation, permission checks

#### Security Event Logging
- All authentication attempts
- Permission escalation attempts
- Account lockouts
- Password changes
- Admin actions
- Suspicious patterns

### Documentation Requirements

#### Deployment Guide
1. Prerequisites (SQLite 3.9+, Python 3.8+)
2. Installation steps
3. Initial configuration
4. Root user setup
5. Migration from single-user
6. Backup procedures
7. Security hardening checklist
8. Rollback procedures

#### API Documentation
1. OpenAPI/Swagger spec
2. Authentication flow diagrams
3. Example requests/responses
4. Error code reference
5. Rate limit documentation
6. Client SDK examples
7. API key migration guide

#### Administrator Guide
1. User management procedures
2. Code generation best practices
3. Security configuration
4. Monitoring and alerts
5. Troubleshooting common issues
6. Incident response procedures
7. GDPR compliance procedures

#### Security Best Practices
1. Password policies
2. Session management
3. Audit log review
4. Incident response
5. Backup and recovery
6. Regular security audits

### Configuration Updates

#### New Environment Variables
```env
# Authentication
JWT_SECRET_KEY=<generated-64-char-secret>
JWT_ACCESS_TOKEN_EXPIRE_MINUTES=15
JWT_REFRESH_TOKEN_EXPIRE_DAYS=7

# Registration
REGISTRATION_ENABLED=true
REGISTRATION_REQUIRE_CODE=true
DEFAULT_USER_ROLE=user
DEFAULT_STORAGE_QUOTA_MB=10240

# Security
PASSWORD_MIN_LENGTH=12
MAX_LOGIN_ATTEMPTS=5
LOCKOUT_DURATION_MINUTES=30

# Admin
ADMIN_EMAIL_DOMAINS=admin.local,company.com

# Database
ENABLE_WAL_MODE=true

# Monitoring
ENABLE_METRICS_ENDPOINT=true
ENABLE_HEALTH_ENDPOINT=true

# GDPR
DATA_RETENTION_DAYS=365
ENABLE_RIGHT_TO_DELETION=true
```

### Implementation Phases

#### Phase 1: Database Foundation (Week 1)
- [ ] Create UserDatabase class following MediaDatabase pattern
- [ ] Implement all table schemas
- [ ] Enable WAL mode and retry logic
- [ ] Create root user generation script
- [ ] Implement migration script with rollback capability

#### Phase 2: Core Authentication (Week 1-2)
- [ ] Implement JWT with refresh tokens
- [ ] Create session management
- [ ] Build login/logout endpoints
- [ ] Add password hashing and validation
- [ ] Implement account lockout
- [ ] Add CSRF protection
- [ ] API key backwards compatibility

#### Phase 3: User Registration (Week 2)
- [ ] Create registration code system
- [ ] Implement registration with FastAPI BackgroundTasks
- [ ] Add rate limiting
- [ ] Create user directory structure on registration
- [ ] Add audit and security logging

#### Phase 4: Admin Interface (Week 3)
- [ ] Build user management endpoints
- [ ] Create code generation endpoints (with limits)
- [ ] Implement user deletion with archival
- [ ] Add system statistics endpoint
- [ ] Create audit log viewer
- [ ] Implement password reset flow

#### Phase 5: User Experience (Week 3-4)
- [ ] Add user profile management
- [ ] Implement password change
- [ ] Create session management UI
- [ ] Add user preferences system
- [ ] Build storage quota tracking and enforcement

#### Phase 6: Testing & Documentation (Week 4)
- [ ] Write comprehensive unit tests
- [ ] Create integration tests
- [ ] Perform stress testing
- [ ] Security testing
- [ ] Migration testing
- [ ] Edge case testing
- [ ] Document API endpoints
- [ ] Create deployment guide
- [ ] Write admin documentation

#### Phase 7: Compliance & Operations (Week 5)
- [ ] Implement GDPR compliance features
- [ ] Create data retention policies
- [ ] Add health check endpoint
- [ ] Create metrics endpoint (Prometheus/OTEL)
- [ ] Document rollback procedures

### Future Enhancements

1. **Email Verification** (Low Priority)
   - SMTP configuration
   - Verification token generation
   - Email templates

2. **OAuth/SSO Integration**
   - OAuth2 providers (Google, GitHub)
   - SAML support
   - LDAP/AD integration

3. **Advanced RBAC**
   - Custom roles
   - Granular permissions
   - Resource-based permissions

4. **PostgreSQL Migration**
   - Multi-tenant schema
   - Connection pooling
   - Full-text search

5. **2FA Support**
   - TOTP implementation
   - Backup codes
   - WebAuthn support

6. **Backup System Integration**
   - User data backup scheduling
   - Encrypted backups
   - Cross-region replication

### Known Limitations & TODOs

1. **Connection Pooling**: Using simple retry logic instead of true pooling initially
2. **File Locking**: Relying on atomic operations and exception handling rather than explicit locks
3. **Backup Strategy**: To be integrated with existing backup system (documented elsewhere)
4. **Monitoring Details**: Specific metrics to be defined during implementation
5. **Performance Benchmarks**: To be established during testing phase