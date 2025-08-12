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
- **Credential Storage**: Save root credentials in `.env` file in application directory
- **Format**: 
  ```env
  ROOT_USER_EMAIL=root@localhost
  ROOT_USER_PASSWORD=<generated-32-char-password>
  ROOT_USER_CREATED_AT=<timestamp>
  ```
- **Security**: File permissions set to 600 (owner read/write only)

### User Database Design (Full Implementation)

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
    
    -- Indices
    INDEX idx_username (username),
    INDEX idx_email (email),
    INDEX idx_role (role),
    INDEX idx_is_active (is_active)
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
    
    -- Token Management
    access_token_jti TEXT UNIQUE NOT NULL, -- JWT ID for access token
    refresh_token_jti TEXT UNIQUE NOT NULL, -- JWT ID for refresh token
    refresh_token_hash TEXT NOT NULL, -- Hashed refresh token for validation
    
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

#### Audit Log Table (for compliance and security)
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

### User Data Management

#### Directory Structure
```
user_databases/
├── 0/                          # Root user (ID: 0)
│   ├── media.db
│   ├── prompts.db
│   ├── uploads/
│   └── exports/
├── 1/                          # User ID: 1
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

### Session Management Implementation

#### Token Strategy
- **Access Token**: Short-lived (15 minutes), contains user info
- **Refresh Token**: Long-lived (7 days), used to get new access token
- **Token Rotation**: New refresh token issued on each refresh
- **Blacklisting**: Revoked tokens tracked in sessions table

#### Session Endpoints
```
POST   /api/v1/auth/login          # Create session
POST   /api/v1/auth/refresh        # Refresh access token
POST   /api/v1/auth/logout         # Revoke current session
POST   /api/v1/auth/logout-all     # Revoke all user sessions
GET    /api/v1/auth/sessions       # List active sessions
DELETE /api/v1/auth/sessions/{id}  # Revoke specific session
```

### Registration Code Management

#### Code Generation
- **Format**: UUID v4 (e.g., `550e8400-e29b-41d4-a716-446655440000`)
- **Default Expiry**: 7 days
- **Default Uses**: 1
- **Batch Generation**: Support creating multiple codes at once

#### Security Features
- **Rate Limiting**: Max 5 registration attempts per IP per hour
- **Code Validation**: Check expiry, usage count, domain restrictions
- **Atomic Usage**: Use database transaction to prevent race conditions
- **Audit Trail**: Log all registration attempts (success and failure)

### API Endpoints Design

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
    400: Invalid code/data
    429: Rate limited

POST /login:
  body:
    username: string
    password: string
  responses:
    200: { access_token, refresh_token }
    401: Invalid credentials
    429: Too many attempts

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
    password?: string
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
  responses:
    200: User updated

DELETE /users/{id}:
  responses:
    200: User deleted

POST /users/{id}/reset-password:
  responses:
    200: { temporary_password: string }

POST /users/{id}/revoke-sessions:
  responses:
    200: All sessions revoked

# Registration Code Management
POST /registration-codes:
  body:
    count?: number (default: 1)
    max_uses?: number (default: 1)
    expires_in_days?: number (default: 7)
    role_granted?: string (default: 'user')
    allowed_email_domain?: string
    description?: string
  responses:
    201: { codes: [uuid] }

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

GET /system/stats:
  responses:
    200: {
      total_users: number,
      active_users: number,
      total_storage_used_gb: number,
      active_sessions: number
    }
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

### Security Implementation

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

#### Rate Limiting
```python
RATE_LIMITS = {
    'registration': '5/hour',
    'login': '10/hour',
    'password_reset': '3/hour',
    'api_general': '100/minute'
}
```

#### Security Headers
```python
SECURITY_HEADERS = {
    'X-Frame-Options': 'DENY',
    'X-Content-Type-Options': 'nosniff',
    'X-XSS-Protection': '1; mode=block',
    'Strict-Transport-Security': 'max-age=31536000; includeSubDomains'
}
```

### Implementation Phases

#### Phase 1: Database Foundation (Week 1)
- [ ] Create UserDatabase class similar to MediaDatabase
- [ ] Implement all table schemas
- [ ] Create database initialization script
- [ ] Implement root user generation on first run
- [ ] Write database migration system

#### Phase 2: Core Authentication (Week 1-2)
- [ ] Implement JWT with refresh tokens
- [ ] Create session management
- [ ] Build login/logout endpoints
- [ ] Add password hashing and validation
- [ ] Implement get_current_user dependency

#### Phase 3: User Registration (Week 2)
- [ ] Create registration code system
- [ ] Implement public registration endpoint
- [ ] Add rate limiting
- [ ] Create user directory structure on registration
- [ ] Add audit logging

#### Phase 4: Admin Interface (Week 3)
- [ ] Build user management endpoints
- [ ] Create code generation endpoints
- [ ] Implement user deletion with archival
- [ ] Add system statistics endpoint
- [ ] Create audit log viewer

#### Phase 5: User Experience (Week 3-4)
- [ ] Add user profile management
- [ ] Implement password change
- [ ] Create session management UI
- [ ] Add user preferences system
- [ ] Build storage quota tracking

#### Phase 6: Testing & Documentation (Week 4)
- [ ] Write comprehensive unit tests
- [ ] Create integration tests
- [ ] Test concurrent registration scenarios
- [ ] Document API endpoints
- [ ] Create admin guide

### Testing Strategy

#### Unit Tests
```python
# test_user_db.py
- test_create_user()
- test_duplicate_user()
- test_user_deletion()
- test_role_validation()

# test_registration_codes.py
- test_code_generation()
- test_code_validation()
- test_code_expiry()
- test_concurrent_code_usage()

# test_sessions.py
- test_token_creation()
- test_token_refresh()
- test_token_revocation()
- test_session_expiry()
```

#### Integration Tests
```python
# test_registration_flow.py
- test_full_registration_with_code()
- test_registration_rate_limiting()
- test_user_directory_creation()
- test_invalid_code_handling()

# test_admin_operations.py
- test_user_creation_by_admin()
- test_user_deletion_with_archival()
- test_batch_code_generation()
- test_audit_log_creation()
```

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
    'USER001': 'Username already exists',
    'USER002': 'Email already exists',
    'USER003': 'Password requirements not met',
    'PERM001': 'Insufficient permissions',
    'RATE001': 'Rate limit exceeded'
}
```

### Monitoring & Logging

#### Key Metrics
- Registration attempts (success/failure)
- Login attempts (success/failure)
- Active sessions count
- Storage usage by user
- API response times
- Error rates by endpoint

#### Log Levels
- **ERROR**: Authentication failures, system errors
- **WARNING**: Rate limit hits, invalid codes
- **INFO**: Successful registrations, logins, user actions
- **DEBUG**: Token validation, permission checks

### Future Enhancements

1. **Email Verification**
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