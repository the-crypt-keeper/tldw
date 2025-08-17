# AuthNZ API Guide

## Table of Contents
- [Overview](#overview)
- [Authentication Modes](#authentication-modes)
- [API Endpoints](#api-endpoints)
- [Request Examples](#request-examples)
- [Response Formats](#response-formats)
- [Error Handling](#error-handling)
- [Rate Limiting](#rate-limiting)
- [Best Practices](#best-practices)

## Overview

The AuthNZ module provides comprehensive authentication and authorization for the tldw_server API. It supports both single-user (API key) and multi-user (JWT) modes with enterprise-grade security features.

### Base URL
```
http://localhost:8000/api/v1
```

### Authentication Headers

#### Single-User Mode
```http
X-API-KEY: your-api-key-here
```

#### Multi-User Mode
```http
Authorization: Bearer your-jwt-token-here
```

## Authentication Modes

### Single-User Mode

Simple API key authentication for personal deployments:

- **Configuration**: Set `AUTH_MODE=single_user` in `.env`
- **API Key**: Auto-generated on first run, displayed in console
- **Header**: `X-API-KEY`
- **Use Case**: Personal installations, development

### Multi-User Mode

JWT-based authentication with user management:

- **Configuration**: Set `AUTH_MODE=multi_user` in `.env`
- **Authentication**: Username/password login returns JWT tokens
- **Header**: `Authorization: Bearer <token>`
- **Features**: User registration, roles, permissions, API key management
- **Use Case**: Team deployments, production environments

## API Endpoints

### Authentication Endpoints

#### Login (Multi-User Mode)
```http
POST /api/v1/auth/login
```

**Request Body:**
```json
{
  "username": "user@example.com",
  "password": "secure_password"
}
```

**Response:**
```json
{
  "access_token": "eyJ0eXAiOiJKV1QiLCJhbGc...",
  "refresh_token": "eyJ0eXAiOiJKV1QiLCJhbGc...",
  "token_type": "bearer",
  "expires_in": 1800,
  "user": {
    "id": 1,
    "username": "user@example.com",
    "email": "user@example.com",
    "role": "user"
  }
}
```

#### Refresh Token
```http
POST /api/v1/auth/refresh
```

**Request Body:**
```json
{
  "refresh_token": "eyJ0eXAiOiJKV1QiLCJhbGc..."
}
```

**Response:**
```json
{
  "access_token": "eyJ0eXAiOiJKV1QiLCJhbGc...",
  "token_type": "bearer",
  "expires_in": 1800
}
```

#### Logout
```http
POST /api/v1/auth/logout
```

**Headers:**
```http
Authorization: Bearer your-jwt-token
```

**Response:**
```json
{
  "message": "Successfully logged out"
}
```

### User Management Endpoints

#### Register New User (If Enabled)
```http
POST /api/v1/auth/register
```

**Request Body:**
```json
{
  "username": "newuser",
  "email": "newuser@example.com",
  "password": "SecurePassword123!",
  "registration_code": "INVITE-CODE-123"
}
```

**Response:**
```json
{
  "id": 2,
  "username": "newuser",
  "email": "newuser@example.com",
  "role": "user",
  "created_at": "2024-01-15T10:30:00Z"
}
```

#### Get Current User
```http
GET /api/v1/auth/me
```

**Headers:**
```http
Authorization: Bearer your-jwt-token
```

**Response:**
```json
{
  "id": 1,
  "username": "user@example.com",
  "email": "user@example.com",
  "role": "user",
  "is_active": true,
  "storage_quota_mb": 5120,
  "storage_used_mb": 1024,
  "created_at": "2024-01-01T00:00:00Z",
  "last_login": "2024-01-15T09:00:00Z"
}
```

#### Update Password
```http
PUT /api/v1/auth/password
```

**Headers:**
```http
Authorization: Bearer your-jwt-token
```

**Request Body:**
```json
{
  "current_password": "OldPassword123!",
  "new_password": "NewSecurePassword456!"
}
```

**Response:**
```json
{
  "message": "Password updated successfully"
}
```

### API Key Management (Multi-User Mode)

#### List API Keys
```http
GET /api/v1/auth/api-keys
```

**Headers:**
```http
Authorization: Bearer your-jwt-token
```

**Response:**
```json
{
  "api_keys": [
    {
      "id": 1,
      "name": "Production Key",
      "key_prefix": "tldw_ak_1234",
      "scope": "read",
      "status": "active",
      "created_at": "2024-01-01T00:00:00Z",
      "last_used_at": "2024-01-15T08:30:00Z",
      "expires_at": "2024-04-01T00:00:00Z"
    }
  ]
}
```

#### Create API Key
```http
POST /api/v1/auth/api-keys
```

**Headers:**
```http
Authorization: Bearer your-jwt-token
```

**Request Body:**
```json
{
  "name": "CI/CD Pipeline Key",
  "description": "Used for automated deployments",
  "scope": "write",
  "expires_in_days": 90
}
```

**Response:**
```json
{
  "id": 2,
  "key": "tldw_ak_5678_AbCdEfGhIjKlMnOpQrStUvWxYz",
  "name": "CI/CD Pipeline Key",
  "scope": "write",
  "expires_at": "2024-04-15T10:30:00Z",
  "message": "Save this key - it won't be shown again"
}
```

#### Rotate API Key
```http
POST /api/v1/auth/api-keys/{key_id}/rotate
```

**Headers:**
```http
Authorization: Bearer your-jwt-token
```

**Response:**
```json
{
  "old_key_id": 2,
  "new_key_id": 3,
  "new_key": "tldw_ak_9012_ZyXwVuTsRqPoNmLkJiHgFeDcBa",
  "expires_at": "2024-07-15T10:30:00Z",
  "message": "Key rotated successfully. Old key will remain active for 24 hours."
}
```

#### Revoke API Key
```http
DELETE /api/v1/auth/api-keys/{key_id}
```

**Headers:**
```http
Authorization: Bearer your-jwt-token
```

**Request Body (Optional):**
```json
{
  "reason": "Key compromised"
}
```

**Response:**
```json
{
  "message": "API key revoked successfully",
  "revoked_at": "2024-01-15T10:30:00Z"
}
```

### Session Management

#### List Active Sessions
```http
GET /api/v1/auth/sessions
```

**Headers:**
```http
Authorization: Bearer your-jwt-token
```

**Response:**
```json
{
  "sessions": [
    {
      "id": "sess_123",
      "ip_address": "192.168.1.100",
      "user_agent": "Mozilla/5.0...",
      "created_at": "2024-01-15T08:00:00Z",
      "last_activity": "2024-01-15T10:00:00Z",
      "is_current": true
    }
  ]
}
```

#### Revoke Session
```http
DELETE /api/v1/auth/sessions/{session_id}
```

**Headers:**
```http
Authorization: Bearer your-jwt-token
```

**Response:**
```json
{
  "message": "Session revoked successfully"
}
```

## Request Examples

### JavaScript/TypeScript

#### Single-User Mode
```javascript
// Simple API call with API key
const response = await fetch('http://localhost:8000/api/v1/media/search', {
  headers: {
    'X-API-KEY': 'your-api-key-here',
    'Content-Type': 'application/json'
  }
});
const data = await response.json();
```

#### Multi-User Mode
```javascript
// Login and get token
async function login(username, password) {
  const response = await fetch('http://localhost:8000/api/v1/auth/login', {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json'
    },
    body: JSON.stringify({ username, password })
  });
  
  if (!response.ok) {
    throw new Error('Login failed');
  }
  
  const data = await response.json();
  // Store tokens securely
  localStorage.setItem('access_token', data.access_token);
  localStorage.setItem('refresh_token', data.refresh_token);
  
  return data;
}

// Make authenticated request
async function makeAuthenticatedRequest(endpoint) {
  const token = localStorage.getItem('access_token');
  
  const response = await fetch(`http://localhost:8000/api/v1${endpoint}`, {
    headers: {
      'Authorization': `Bearer ${token}`,
      'Content-Type': 'application/json'
    }
  });
  
  if (response.status === 401) {
    // Token expired, try refresh
    await refreshToken();
    return makeAuthenticatedRequest(endpoint);
  }
  
  return response.json();
}

// Refresh token
async function refreshToken() {
  const refreshToken = localStorage.getItem('refresh_token');
  
  const response = await fetch('http://localhost:8000/api/v1/auth/refresh', {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json'
    },
    body: JSON.stringify({ refresh_token: refreshToken })
  });
  
  if (!response.ok) {
    // Refresh failed, redirect to login
    window.location.href = '/login';
    return;
  }
  
  const data = await response.json();
  localStorage.setItem('access_token', data.access_token);
}
```

### Python

#### Single-User Mode
```python
import requests

# Set up session with API key
session = requests.Session()
session.headers.update({
    'X-API-KEY': 'your-api-key-here'
})

# Make requests
response = session.get('http://localhost:8000/api/v1/media/search')
data = response.json()
```

#### Multi-User Mode
```python
import requests
from datetime import datetime, timedelta

class TLDWClient:
    def __init__(self, base_url='http://localhost:8000'):
        self.base_url = base_url
        self.session = requests.Session()
        self.access_token = None
        self.refresh_token = None
        self.token_expires = None
    
    def login(self, username, password):
        """Login and store tokens"""
        response = self.session.post(
            f'{self.base_url}/api/v1/auth/login',
            json={'username': username, 'password': password}
        )
        response.raise_for_status()
        
        data = response.json()
        self.access_token = data['access_token']
        self.refresh_token = data['refresh_token']
        self.token_expires = datetime.now() + timedelta(seconds=data['expires_in'])
        
        # Set authorization header
        self.session.headers.update({
            'Authorization': f'Bearer {self.access_token}'
        })
        
        return data
    
    def refresh_access_token(self):
        """Refresh the access token"""
        response = self.session.post(
            f'{self.base_url}/api/v1/auth/refresh',
            json={'refresh_token': self.refresh_token}
        )
        response.raise_for_status()
        
        data = response.json()
        self.access_token = data['access_token']
        self.token_expires = datetime.now() + timedelta(seconds=data['expires_in'])
        
        # Update authorization header
        self.session.headers.update({
            'Authorization': f'Bearer {self.access_token}'
        })
    
    def request(self, method, endpoint, **kwargs):
        """Make authenticated request with automatic token refresh"""
        # Check if token needs refresh
        if self.token_expires and datetime.now() >= self.token_expires:
            self.refresh_access_token()
        
        response = self.session.request(
            method,
            f'{self.base_url}/api/v1{endpoint}',
            **kwargs
        )
        
        # If unauthorized, try refreshing token once
        if response.status_code == 401:
            self.refresh_access_token()
            response = self.session.request(
                method,
                f'{self.base_url}/api/v1{endpoint}',
                **kwargs
            )
        
        response.raise_for_status()
        return response.json()

# Usage
client = TLDWClient()
client.login('user@example.com', 'password')

# Make authenticated requests
media = client.request('GET', '/media/search', params={'query': 'machine learning'})
```

### cURL

#### Single-User Mode
```bash
# Simple request with API key
curl -H "X-API-KEY: your-api-key-here" \
     http://localhost:8000/api/v1/media/search?query=test

# POST request with API key
curl -X POST \
     -H "X-API-KEY: your-api-key-here" \
     -H "Content-Type: application/json" \
     -d '{"url": "https://example.com/video"}' \
     http://localhost:8000/api/v1/media/process
```

#### Multi-User Mode
```bash
# Login and save tokens
response=$(curl -X POST \
     -H "Content-Type: application/json" \
     -d '{"username": "user@example.com", "password": "password"}' \
     http://localhost:8000/api/v1/auth/login)

# Extract token (requires jq)
token=$(echo $response | jq -r '.access_token')

# Make authenticated request
curl -H "Authorization: Bearer $token" \
     http://localhost:8000/api/v1/media/search?query=test

# Create API key
curl -X POST \
     -H "Authorization: Bearer $token" \
     -H "Content-Type: application/json" \
     -d '{"name": "My API Key", "scope": "read"}' \
     http://localhost:8000/api/v1/auth/api-keys
```

## Response Formats

### Success Response
```json
{
  "status": "success",
  "data": {
    // Response data here
  },
  "message": "Operation completed successfully"
}
```

### Error Response
```json
{
  "status": "error",
  "error": {
    "code": "UNAUTHORIZED",
    "message": "Invalid or expired token",
    "details": "Token expired at 2024-01-15T10:00:00Z"
  },
  "request_id": "req_123456"
}
```

### Paginated Response
```json
{
  "status": "success",
  "data": {
    "items": [...],
    "pagination": {
      "page": 1,
      "per_page": 20,
      "total": 100,
      "total_pages": 5,
      "has_next": true,
      "has_prev": false
    }
  }
}
```

## Error Handling

### HTTP Status Codes

| Code | Meaning | Description |
|------|---------|-------------|
| 200 | OK | Request succeeded |
| 201 | Created | Resource created successfully |
| 204 | No Content | Request succeeded with no response body |
| 400 | Bad Request | Invalid request parameters |
| 401 | Unauthorized | Missing or invalid authentication |
| 403 | Forbidden | Authenticated but not authorized |
| 404 | Not Found | Resource not found |
| 409 | Conflict | Resource conflict (e.g., duplicate username) |
| 422 | Unprocessable Entity | Validation error |
| 429 | Too Many Requests | Rate limit exceeded |
| 500 | Internal Server Error | Server error |

### Error Codes

| Code | Description | Resolution |
|------|-------------|------------|
| `INVALID_CREDENTIALS` | Wrong username/password | Check credentials |
| `TOKEN_EXPIRED` | JWT token has expired | Refresh token |
| `TOKEN_INVALID` | JWT token is malformed | Re-authenticate |
| `API_KEY_INVALID` | API key not found or revoked | Check API key |
| `RATE_LIMIT_EXCEEDED` | Too many requests | Wait and retry |
| `INSUFFICIENT_PERMISSIONS` | User lacks required permissions | Check user role |
| `REGISTRATION_DISABLED` | Registration is not enabled | Contact admin |
| `PASSWORD_TOO_WEAK` | Password doesn't meet requirements | Use stronger password |
| `SESSION_EXPIRED` | Session has expired | Login again |

### Error Handling Best Practices

1. **Always check HTTP status codes**
2. **Parse error responses for details**
3. **Implement exponential backoff for rate limits**
4. **Handle token expiration gracefully**
5. **Log errors for debugging**

## Rate Limiting

### Default Limits

- **Anonymous**: 10 requests/minute
- **Authenticated**: 60 requests/minute
- **Service Accounts**: 1000 requests/minute

### Rate Limit Headers

```http
X-RateLimit-Limit: 60
X-RateLimit-Remaining: 45
X-RateLimit-Reset: 1705318800
X-RateLimit-Reset-After: 30
```

### Handling Rate Limits

```javascript
async function makeRequestWithRetry(url, options, maxRetries = 3) {
  for (let i = 0; i < maxRetries; i++) {
    const response = await fetch(url, options);
    
    if (response.status === 429) {
      // Get retry delay from header
      const retryAfter = response.headers.get('X-RateLimit-Reset-After');
      const delay = retryAfter ? parseInt(retryAfter) * 1000 : (i + 1) * 1000;
      
      console.log(`Rate limited. Retrying after ${delay}ms...`);
      await new Promise(resolve => setTimeout(resolve, delay));
      continue;
    }
    
    return response;
  }
  
  throw new Error('Max retries exceeded');
}
```

## Best Practices

### Security

1. **Never expose tokens in URLs** - Use headers or request body
2. **Store tokens securely** - Use secure storage, not localStorage for sensitive apps
3. **Implement token refresh** - Don't wait for expiration
4. **Use HTTPS in production** - Required for secure cookies
5. **Rotate API keys regularly** - Set up scheduled rotation
6. **Monitor failed authentication** - Watch for attacks

### Performance

1. **Cache tokens** - Don't login for every request
2. **Batch requests** - Use bulk endpoints when available
3. **Handle rate limits gracefully** - Implement backoff
4. **Use connection pooling** - Reuse HTTP connections
5. **Implement request timeouts** - Prevent hanging requests

### Error Recovery

1. **Implement retry logic** - For transient failures
2. **Handle token refresh** - Automatically refresh expired tokens
3. **Fallback strategies** - Have backup plans for failures
4. **User feedback** - Show meaningful error messages
5. **Logging** - Log errors for debugging

### Code Organization

```javascript
// Good: Centralized auth handling
class AuthService {
  constructor() {
    this.token = null;
    this.refreshToken = null;
  }
  
  async authenticate() {
    // Handle login, token storage, refresh
  }
  
  async makeAuthenticatedRequest(endpoint, options) {
    // Handle auth headers, refresh, retry
  }
}

// Bad: Scattered auth logic
async function getMedia() {
  const token = localStorage.getItem('token');
  // Auth logic repeated everywhere
}
```

## Migration from Gradio

If migrating from the old Gradio interface:

1. **API Keys**: Generate new API keys using the AuthNZ system
2. **Endpoints**: Update to new `/api/v1` endpoints
3. **Authentication**: Switch from session-based to token-based
4. **Headers**: Use standard authentication headers
5. **Response Format**: Handle new standardized JSON responses

## Testing Authentication

### Test Endpoints

```bash
# Test single-user mode
curl -H "X-API-KEY: your-key" http://localhost:8000/api/v1/auth/test

# Test multi-user mode (after login)
curl -H "Authorization: Bearer your-token" http://localhost:8000/api/v1/auth/test
```

### Postman Collection

Import this collection for testing:

```json
{
  "info": {
    "name": "TLDW AuthNZ",
    "schema": "https://schema.getpostman.com/json/collection/v2.1.0/collection.json"
  },
  "auth": {
    "type": "bearer",
    "bearer": [
      {
        "key": "token",
        "value": "{{access_token}}",
        "type": "string"
      }
    ]
  },
  "variable": [
    {
      "key": "base_url",
      "value": "http://localhost:8000/api/v1"
    },
    {
      "key": "access_token",
      "value": ""
    }
  ],
  "item": [
    {
      "name": "Login",
      "request": {
        "method": "POST",
        "header": [],
        "body": {
          "mode": "raw",
          "raw": "{\n  \"username\": \"admin\",\n  \"password\": \"password\"\n}",
          "options": {
            "raw": {
              "language": "json"
            }
          }
        },
        "url": {
          "raw": "{{base_url}}/auth/login",
          "host": ["{{base_url}}"],
          "path": ["auth", "login"]
        }
      },
      "event": [
        {
          "listen": "test",
          "script": {
            "exec": [
              "const response = pm.response.json();",
              "pm.collectionVariables.set('access_token', response.access_token);"
            ]
          }
        }
      ]
    }
  ]
}
```

## Support

- **Documentation**: See [AuthNZ Developer Guide](../Development/AuthNZ-Developer-Guide.md)
- **Issues**: Report at [GitHub Issues](https://github.com/rmusser01/tldw_server/issues)
- **Security**: Report security issues privately to the maintainers