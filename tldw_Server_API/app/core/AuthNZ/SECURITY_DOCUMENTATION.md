# AuthNZ Security Features - Comprehensive Documentation

## Table of Contents
1. [Overview](#overview)
2. [Security Architecture](#security-architecture)
3. [Authentication Flow](#authentication-flow)
4. [Multi-Factor Authentication (MFA)](#multi-factor-authentication-mfa)
5. [Password Management](#password-management)
6. [Token Management](#token-management)
7. [Email Services](#email-services)
8. [API Endpoints](#api-endpoints)
9. [Configuration Guide](#configuration-guide)
10. [Testing Guide](#testing-guide)
11. [Security Best Practices](#security-best-practices)
12. [Troubleshooting](#troubleshooting)

---

## Overview

The AuthNZ module provides enterprise-grade authentication and authorization for the TLDW Server. This documentation covers the enhanced security features implemented to protect user accounts and data.

### Key Features
- **JWT-based authentication** with token blacklisting
- **Multi-Factor Authentication (MFA)** using TOTP
- **Password reset flow** with secure tokens
- **Email verification** for new accounts
- **Rate limiting** and account lockout protection
- **Input validation** to prevent injection attacks
- **Audit logging** for security events
- **Support for both SQLite and PostgreSQL**

### Security Improvements Summary
- ✅ Removed file-based JWT secret storage vulnerability
- ✅ Implemented comprehensive input validation
- ✅ Added failed login tracking with account lockout
- ✅ Token blacklist for secure logout
- ✅ MFA/2FA with TOTP and backup codes
- ✅ Secure password reset flow
- ✅ Email verification system

---

## Security Architecture

### Component Overview

```
┌─────────────────────────────────────────────────────────┐
│                     API Layer                            │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐  │
│  │ auth.py      │  │auth_enhanced │  │ Middleware   │  │
│  │ (Basic Auth) │  │(Advanced)    │  │ (Rate Limit) │  │
│  └──────────────┘  └──────────────┘  └──────────────┘  │
└─────────────────────────────────────────────────────────┘
                            │
┌─────────────────────────────────────────────────────────┐
│                   Service Layer                          │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐  │
│  │ JWT Service  │  │ MFA Service  │  │Email Service │  │
│  └──────────────┘  └──────────────┘  └──────────────┘  │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐  │
│  │Token Blacklist│ │Password Svc  │  │Input Valid.  │  │
│  └──────────────┘  └──────────────┘  └──────────────┘  │
└─────────────────────────────────────────────────────────┘
                            │
┌─────────────────────────────────────────────────────────┐
│                   Data Layer                             │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐  │
│  │ PostgreSQL   │  │   SQLite     │  │    Redis     │  │
│  │  (Primary)   │  │ (Alternative)│  │   (Cache)    │  │
│  └──────────────┘  └──────────────┘  └──────────────┘  │
└─────────────────────────────────────────────────────────┘
```

### Security Layers

1. **Input Layer**: Validates and sanitizes all user input
2. **Authentication Layer**: Verifies user credentials and MFA
3. **Authorization Layer**: Checks permissions and roles
4. **Session Layer**: Manages JWT tokens and blacklisting
5. **Audit Layer**: Logs all security-relevant events

---

## Authentication Flow

### Standard Login Flow

```mermaid
sequenceDiagram
    participant User
    participant API
    participant Validator
    participant Auth
    participant MFA
    participant JWT
    participant DB
    
    User->>API: POST /auth/login
    API->>Validator: Validate input
    Validator->>API: Input safe
    API->>Auth: Check credentials
    Auth->>DB: Query user
    DB->>Auth: User data
    Auth->>Auth: Verify password
    
    alt MFA Enabled
        Auth->>User: Request MFA token
        User->>API: Provide TOTP
        API->>MFA: Verify TOTP
        MFA->>API: Valid
    end
    
    Auth->>JWT: Create tokens
    JWT->>DB: Store session
    JWT->>User: Access + Refresh tokens
```

### Login with Account Lockout

```python
# Example: Login endpoint with all security features
@router.post("/login")
async def login(request: LoginRequest):
    # 1. Check IP lockout
    if await rate_limiter.check_lockout(client_ip):
        raise HTTPException(429, "Account locked")
    
    # 2. Validate input
    is_valid, error = validator.validate_username(request.username)
    if not is_valid:
        await rate_limiter.record_failed_attempt(client_ip)
        raise HTTPException(400, "Invalid credentials")
    
    # 3. Verify credentials
    user = await db.get_user(request.username)
    if not password_service.verify_password(request.password, user.password_hash):
        await rate_limiter.record_failed_attempt(client_ip, username)
        raise HTTPException(401, "Invalid credentials")
    
    # 4. Check MFA if enabled
    if user.two_factor_enabled:
        if not mfa_service.verify_totp(user.totp_secret, request.mfa_token):
            raise HTTPException(401, "Invalid MFA token")
    
    # 5. Generate tokens
    access_token = jwt_service.create_access_token(user)
    refresh_token = jwt_service.create_refresh_token(user)
    
    # 6. Reset failed attempts
    await rate_limiter.reset_failed_attempts(client_ip, username)
    
    return {"access_token": access_token, "refresh_token": refresh_token}
```

---

## Multi-Factor Authentication (MFA)

### TOTP Implementation

The MFA system uses Time-based One-Time Passwords (TOTP) compatible with standard authenticator apps:
- Google Authenticator
- Microsoft Authenticator
- Authy
- 1Password
- Bitwarden

### MFA Setup Process

1. **Initialize MFA Setup**
   ```bash
   POST /auth/mfa/setup
   Authorization: Bearer <access_token>
   
   Response:
   {
     "secret": "JBSWY3DPEHPK3PXP",
     "qr_code": "data:image/png;base64,...",
     "backup_codes": [
       "ABCD-1234",
       "EFGH-5678",
       ...
     ]
   }
   ```

2. **Verify Setup with TOTP**
   ```bash
   POST /auth/mfa/verify
   Authorization: Bearer <access_token>
   X-MFA-Secret: JBSWY3DPEHPK3PXP
   
   {
     "token": "123456"
   }
   ```

3. **MFA is now enabled for the account**

### Backup Codes

- 8 single-use recovery codes generated during setup
- Format: `XXXX-XXXX` (alphanumeric)
- Stored encrypted in database
- Can be regenerated if lost

### Using MFA During Login

```json
POST /auth/login
{
  "username": "user@example.com",
  "password": "SecurePassword123!",
  "mfa_token": "123456"  // 6-digit TOTP or backup code
}
```

---

## Password Management

### Password Requirements

Default password policy (configurable):
- Minimum 8 characters
- At least one uppercase letter
- At least one lowercase letter
- At least one number
- At least one special character
- No common passwords (checked against list)
- Not similar to username/email

### Password Reset Flow

1. **Request Reset**
   ```bash
   POST /auth/forgot-password
   {
     "email": "user@example.com"
   }
   ```
   - Always returns success (security best practice)
   - Sends email if account exists
   - Token valid for 1 hour

2. **Reset Password**
   ```bash
   POST /auth/reset-password
   {
     "token": "eyJhbGciOiJIUzI1NiIs...",
     "new_password": "NewSecurePassword123!"
   }
   ```
   - Validates token
   - Checks password requirements
   - Revokes all existing sessions
   - Marks token as used

### Password Hashing

- Algorithm: **Argon2id** (most secure)
- Fallback: **bcrypt** (if Argon2 unavailable)
- Automatic rehashing on login if needed
- Salt automatically generated and stored

---

## Token Management

### JWT Token Structure

```json
{
  "sub": "123",           // User ID
  "username": "john.doe",
  "role": "user",
  "exp": 1234567890,      // Expiration
  "iat": 1234567800,      // Issued at
  "jti": "uuid-here",     // JWT ID for blacklisting
  "type": "access"        // Token type
}
```

### Token Types

1. **Access Token**
   - Short-lived (30 minutes default)
   - Used for API requests
   - Contains user info and permissions

2. **Refresh Token**
   - Long-lived (7 days default)
   - Used to get new access tokens
   - Minimal claims

3. **Password Reset Token**
   - Valid for 1 hour
   - Single use only
   - Specific to password reset

4. **Email Verification Token**
   - Valid for 24 hours
   - Used to verify email addresses

### Token Blacklisting

The blacklist system allows for immediate token revocation:

```python
# Revoke a single token
await token_blacklist.revoke_token(
    jti="token-id",
    expires_at=datetime.utcnow() + timedelta(hours=1),
    user_id=123,
    reason="User logout"
)

# Revoke all user tokens (logout from all devices)
await token_blacklist.revoke_all_user_tokens(
    user_id=123,
    reason="Password changed"
)

# Check if token is blacklisted
is_revoked = await token_blacklist.is_blacklisted("token-id")
```

### Storage Strategy

- **Redis** (preferred): Fast lookups, automatic expiration
- **Database** (fallback): Persistent storage with cleanup job
- **Local Cache**: 5-minute TTL for performance

---

## Email Services

### Email Provider Configuration

The system supports multiple email providers:

1. **Mock Provider** (Development/Testing)
   ```bash
   EMAIL_PROVIDER=mock
   EMAIL_MOCK_OUTPUT=console  # console, file, or both
   EMAIL_MOCK_FILE_PATH=./mock_emails/
   ```

2. **SMTP Provider** (Production)
   ```bash
   EMAIL_PROVIDER=smtp
   EMAIL_SMTP_HOST=smtp.gmail.com
   EMAIL_SMTP_PORT=587
   EMAIL_SMTP_USERNAME=your-email@gmail.com
   EMAIL_SMTP_PASSWORD=your-app-password
   EMAIL_SMTP_USE_TLS=true
   EMAIL_FROM_ADDRESS=noreply@yourdomain.com
   EMAIL_FROM_NAME=TLDW Server
   ```

### Email Templates

All emails use responsive HTML templates with:
- Professional styling
- Clear call-to-action buttons
- Security information (IP address, timestamp)
- Plain text fallback

#### Available Templates

1. **Password Reset Email**
   - Reset link with token
   - 1-hour expiration notice
   - IP address of request

2. **Email Verification**
   - Verification link
   - Welcome message
   - Next steps

3. **MFA Enabled Notification**
   - Backup codes
   - Security recommendations
   - Device information

4. **Security Alert**
   - Suspicious activity notification
   - Action required steps

### Mock Email Output

During development, emails are output to console/files:

```
========================================
📧 MOCK EMAIL
========================================
To: user@example.com
Subject: Password Reset Request
From: TLDW Server <noreply@tldw.local>
Date: 2024-01-15 10:30:00
----------------------------------------
[HTML and plain text content]
========================================
```

---

## API Endpoints

### Authentication Endpoints

#### Basic Authentication
- `POST /auth/register` - Create new account
- `POST /auth/login` - Login with credentials
- `POST /auth/refresh` - Refresh access token
- `GET /auth/me` - Get current user info

#### Enhanced Authentication
- `POST /auth/forgot-password` - Request password reset
- `POST /auth/reset-password` - Reset password with token
- `GET /auth/verify-email` - Verify email address
- `POST /auth/resend-verification` - Resend verification email
- `POST /auth/logout` - Logout (blacklist tokens)

#### MFA Endpoints
- `POST /auth/mfa/setup` - Initialize MFA setup
- `POST /auth/mfa/verify` - Complete MFA setup
- `POST /auth/mfa/disable` - Disable MFA
- `POST /auth/mfa/backup-codes` - Regenerate backup codes

### Request/Response Examples

#### Login Request
```json
POST /auth/login
{
  "username": "john.doe",
  "password": "SecurePass123!",
  "mfa_token": "123456"  // Optional, required if MFA enabled
}
```

#### Login Response
```json
{
  "access_token": "eyJhbGciOiJIUzI1NiIs...",
  "refresh_token": "eyJhbGciOiJIUzI1NiIs...",
  "token_type": "bearer",
  "expires_in": 1800
}
```

#### Error Response
```json
{
  "detail": "Invalid credentials",
  "status_code": 401,
  "headers": {
    "X-RateLimit-Remaining": "4",
    "X-RateLimit-Reset": "1234567890"
  }
}
```

---

## Configuration Guide

### Environment Variables

```bash
# Required for production
JWT_SECRET_KEY=your-very-long-secret-key-min-32-chars
DATABASE_URL=postgresql://user:pass@localhost/dbname

# Optional security settings
AUTH_MODE=multi_user
ACCESS_TOKEN_EXPIRE_MINUTES=30
REFRESH_TOKEN_EXPIRE_DAYS=7
PASSWORD_MIN_LENGTH=8
PASSWORD_REQUIRE_UPPERCASE=true
PASSWORD_REQUIRE_LOWERCASE=true
PASSWORD_REQUIRE_NUMBERS=true
PASSWORD_REQUIRE_SPECIAL=true
MAX_LOGIN_ATTEMPTS=5
LOCKOUT_DURATION_MINUTES=15

# Email configuration
EMAIL_PROVIDER=smtp
EMAIL_SMTP_HOST=smtp.gmail.com
EMAIL_SMTP_PORT=587
EMAIL_SMTP_USERNAME=your-email@gmail.com
EMAIL_SMTP_PASSWORD=your-app-password

# MFA settings
MFA_ISSUER_NAME=TLDW Server
MFA_TOTP_DIGITS=6
MFA_TOTP_INTERVAL=30
MFA_BACKUP_CODES_COUNT=8

# Redis (optional, for performance)
REDIS_URL=redis://localhost:6379/0
```

### Database Migrations

Run migrations to create required tables:

```python
from tldw_Server_API.app.core.AuthNZ.migrations import apply_authnz_migrations

# Apply all migrations
apply_authnz_migrations(db_path="path/to/database.db")

# Check migration status
status = check_migration_status(db_path="path/to/database.db")
print(f"Current version: {status['current_version']}")
print(f"Latest version: {status['latest_version']}")
```

### Security Headers

Configure security headers in FastAPI:

```python
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware

app = FastAPI()

# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://yourdomain.com"],
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["Authorization", "Content-Type"],
)

# Trusted host validation
app.add_middleware(
    TrustedHostMiddleware,
    allowed_hosts=["yourdomain.com", "*.yourdomain.com"]
)

# Security headers
@app.middleware("http")
async def add_security_headers(request, call_next):
    response = await call_next(request)
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["X-Frame-Options"] = "DENY"
    response.headers["X-XSS-Protection"] = "1; mode=block"
    response.headers["Strict-Transport-Security"] = "max-age=31536000"
    return response
```

---

## Testing Guide

### Testing with Mock Services

The system includes mock implementations for testing without external dependencies:

#### Mock Email Testing

```python
# Set environment variable
os.environ["EMAIL_PROVIDER"] = "mock"
os.environ["EMAIL_MOCK_OUTPUT"] = "both"  # Console and file

# Emails will be:
# 1. Printed to console
# 2. Saved to ./mock_emails/email_TIMESTAMP.html
```

#### Testing MFA Without Real Device

```python
import pyotp
from tldw_Server_API.app.core.AuthNZ.mfa_service import get_mfa_service

# Generate secret
mfa_service = get_mfa_service()
secret = mfa_service.generate_secret()

# Generate token programmatically
totp = pyotp.TOTP(secret)
current_token = totp.now()

# Verify token
is_valid = mfa_service.verify_totp(secret, current_token)
print(f"Token {current_token} is valid: {is_valid}")
```

### Unit Test Examples

```python
import pytest
from httpx import AsyncClient

@pytest.mark.asyncio
async def test_login_with_mfa(client: AsyncClient):
    # Setup MFA for test user
    response = await client.post("/auth/mfa/setup")
    secret = response.json()["secret"]
    
    # Generate valid TOTP
    import pyotp
    totp = pyotp.TOTP(secret)
    token = totp.now()
    
    # Login with MFA
    response = await client.post("/auth/login", json={
        "username": "testuser",
        "password": "TestPass123!",
        "mfa_token": token
    })
    
    assert response.status_code == 200
    assert "access_token" in response.json()

@pytest.mark.asyncio
async def test_password_reset_flow(client: AsyncClient):
    # Request reset
    response = await client.post("/auth/forgot-password", json={
        "email": "test@example.com"
    })
    assert response.status_code == 200
    
    # Get token from mock email
    # In tests, you might intercept the email service
    reset_token = get_reset_token_from_mock()
    
    # Reset password
    response = await client.post("/auth/reset-password", json={
        "token": reset_token,
        "new_password": "NewPass123!"
    })
    assert response.status_code == 200
```

### Load Testing

Test rate limiting and performance:

```bash
# Using Apache Bench
ab -n 1000 -c 10 -H "Authorization: Bearer TOKEN" \
   http://localhost:8000/api/v1/protected

# Using locust
locust -f tests/load/auth_test.py --host=http://localhost:8000
```

---

## Security Best Practices

### Deployment Checklist

- [ ] **Environment Variables**
  - [ ] JWT_SECRET_KEY is set and secure (min 32 chars)
  - [ ] Database credentials are secure
  - [ ] Email credentials are configured
  - [ ] Redis is configured (if available)

- [ ] **HTTPS Configuration**
  - [ ] SSL/TLS certificate installed
  - [ ] HTTP to HTTPS redirect configured
  - [ ] HSTS header enabled

- [ ] **Database Security**
  - [ ] Database encrypted at rest
  - [ ] Regular backups configured
  - [ ] Connection uses SSL/TLS

- [ ] **Rate Limiting**
  - [ ] Configured for all endpoints
  - [ ] Account lockout enabled
  - [ ] DDoS protection in place

- [ ] **Monitoring**
  - [ ] Security event logging enabled
  - [ ] Failed login attempts monitored
  - [ ] Alerting configured for suspicious activity

### Security Recommendations

1. **Password Policy**
   - Enforce strong passwords (min 12 chars in production)
   - Implement password history to prevent reuse
   - Require periodic password changes for sensitive accounts

2. **Session Management**
   - Use short-lived access tokens (15-30 minutes)
   - Implement sliding session expiration
   - Clear sessions on password change

3. **MFA Enforcement**
   - Require MFA for admin accounts
   - Consider mandatory MFA for all users
   - Provide multiple MFA methods (TOTP, SMS, Email)

4. **Audit Logging**
   - Log all authentication attempts
   - Log permission changes
   - Log sensitive data access
   - Store logs securely and immutably

5. **Input Validation**
   - Validate all user input
   - Use parameterized queries
   - Sanitize output to prevent XSS
   - Implement CSRF protection

6. **Network Security**
   - Use VPN for admin access
   - Implement IP allowlisting for sensitive operations
   - Use Web Application Firewall (WAF)

---

## Troubleshooting

### Common Issues and Solutions

#### Issue: "JWT secret key not configured"
**Solution**: Set `JWT_SECRET_KEY` environment variable
```bash
export JWT_SECRET_KEY=$(openssl rand -hex 32)
```

#### Issue: "Account locked due to failed attempts"
**Solution**: Wait for lockout duration or manually reset in database
```sql
DELETE FROM failed_attempts WHERE identifier = 'username_or_ip';
DELETE FROM account_lockouts WHERE identifier = 'username_or_ip';
```

#### Issue: "MFA token invalid"
**Causes**:
1. Time sync issue between server and device
2. Wrong secret key
3. Token already used (replay attack prevention)

**Solution**: 
- Verify system time: `date`
- Increase validation window: `window=2`
- Use backup code if available

#### Issue: "Email not sending"
**Solution**: Check email configuration
```python
# Test email configuration
from tldw_Server_API.app.core.AuthNZ.email_service import get_email_service

email_service = get_email_service()
await email_service.send_test_email("test@example.com")
```

#### Issue: "Token blacklist not working"
**Solution**: Ensure blacklist service is initialized
```python
# Check blacklist status
from tldw_Server_API.app.core.AuthNZ.token_blacklist import get_token_blacklist

blacklist = get_token_blacklist()
stats = await blacklist.get_statistics()
print(f"Total blacklisted: {stats['total_blacklisted']}")
```

### Debug Mode

Enable debug logging for troubleshooting:

```python
import logging
from loguru import logger

# Set debug level
logger.add("auth_debug.log", level="DEBUG", rotation="10 MB")

# Enable SQL query logging
logging.getLogger('sqlalchemy.engine').setLevel(logging.INFO)
```

### Performance Optimization

1. **Use Redis for token blacklist** (10x faster than database)
2. **Enable connection pooling** for database
3. **Implement caching** for user lookups
4. **Use async operations** throughout
5. **Batch operations** where possible

---

## Migration Guide

### Migrating from Single-User to Multi-User Mode

1. **Update Configuration**
   ```bash
   AUTH_MODE=multi_user
   JWT_SECRET_KEY=<generate-secure-key>
   ```

2. **Run Migrations**
   ```python
   from tldw_Server_API.app.core.AuthNZ.migrations import apply_authnz_migrations
   apply_authnz_migrations(db_path)
   ```

3. **Create Admin User**
   ```python
   # Create first admin user
   await user_service.create_user(
       username="admin",
       email="admin@example.com",
       password="SecureAdminPass123!",
       role="admin",
       is_superuser=True
   )
   ```

4. **Enable Security Features**
   - Configure email service
   - Enable MFA for admin
   - Set up monitoring

### Backward Compatibility

The system maintains backward compatibility:
- Single-user mode still supported with API keys
- Existing sessions remain valid until expiration
- Database schema additions are non-breaking
- API endpoints maintain same interface

---

## Appendix

### Database Schema

```sql
-- Users table
CREATE TABLE users (
    id INTEGER PRIMARY KEY,
    username TEXT UNIQUE NOT NULL,
    email TEXT UNIQUE NOT NULL,
    password_hash TEXT NOT NULL,
    is_active BOOLEAN DEFAULT true,
    is_superuser BOOLEAN DEFAULT false,
    role TEXT DEFAULT 'user',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    last_login TIMESTAMP,
    email_verified BOOLEAN DEFAULT false,
    totp_secret TEXT,
    two_factor_enabled BOOLEAN DEFAULT false,
    backup_codes TEXT
);

-- Token blacklist table
CREATE TABLE token_blacklist (
    id INTEGER PRIMARY KEY,
    jti TEXT UNIQUE NOT NULL,
    user_id INTEGER REFERENCES users(id),
    token_type TEXT,
    revoked_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    expires_at TIMESTAMP NOT NULL,
    reason TEXT
);

-- Password reset tokens
CREATE TABLE password_reset_tokens (
    id INTEGER PRIMARY KEY,
    user_id INTEGER REFERENCES users(id),
    token_hash TEXT UNIQUE NOT NULL,
    expires_at TIMESTAMP NOT NULL,
    used_at TIMESTAMP,
    ip_address TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Failed login attempts
CREATE TABLE failed_attempts (
    id INTEGER PRIMARY KEY,
    identifier TEXT NOT NULL,
    attempt_type TEXT NOT NULL,
    attempt_count INTEGER DEFAULT 1,
    window_start TIMESTAMP NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
```

### API Response Codes

| Code | Meaning | Example |
|------|---------|---------|
| 200 | Success | Login successful |
| 201 | Created | User registered |
| 400 | Bad Request | Invalid input |
| 401 | Unauthorized | Invalid credentials |
| 403 | Forbidden | Insufficient permissions |
| 404 | Not Found | User not found |
| 429 | Too Many Requests | Rate limit exceeded |
| 500 | Server Error | Internal error |

### Glossary

- **JWT**: JSON Web Token - Secure token format for authentication
- **MFA/2FA**: Multi-Factor/Two-Factor Authentication
- **TOTP**: Time-based One-Time Password
- **Argon2id**: Password hashing algorithm (recommended by OWASP)
- **CSRF**: Cross-Site Request Forgery
- **XSS**: Cross-Site Scripting
- **OWASP**: Open Web Application Security Project

---

## Support and Resources

### External Resources
- [OWASP Authentication Cheat Sheet](https://cheatsheetseries.owasp.org/cheatsheets/Authentication_Cheat_Sheet.html)
- [JWT Best Practices](https://tools.ietf.org/html/rfc8725)
- [TOTP RFC 6238](https://tools.ietf.org/html/rfc6238)
- [Argon2 Documentation](https://github.com/P-H-C/phc-winner-argon2)

### Internal Documentation
- [API Documentation](/docs) - Interactive API documentation
- [Project Guidelines](Project_Guidelines.md) - Development guidelines
- [CLAUDE.md](CLAUDE.md) - AI assistant instructions

---

*Last Updated: January 2025*
*Version: 1.0.0*