# AuthNZ API Integration Guide

## Quick Start

### 1. Basic Authentication Flow

```python
import httpx
import pyotp

# Initialize client
client = httpx.AsyncClient(base_url="http://localhost:8000/api/v1")

# Register a new user
async def register_user():
    response = await client.post("/auth/register", json={
        "username": "john.doe",
        "email": "john@example.com",
        "password": "SecurePass123!"
    })
    return response.json()

# Login
async def login(username: str, password: str, mfa_token: str = None):
    data = {
        "username": username,
        "password": password
    }
    if mfa_token:
        data["mfa_token"] = mfa_token
    
    response = await client.post("/auth/login", json=data)
    return response.json()

# Use authenticated endpoints
async def get_protected_data(access_token: str):
    headers = {"Authorization": f"Bearer {access_token}"}
    response = await client.get("/protected", headers=headers)
    return response.json()
```

### 2. Implementing Token Refresh

```python
class AuthenticatedClient:
    def __init__(self, base_url: str):
        self.client = httpx.AsyncClient(base_url=base_url)
        self.access_token = None
        self.refresh_token = None
    
    async def login(self, username: str, password: str):
        response = await self.client.post("/auth/login", json={
            "username": username,
            "password": password
        })
        data = response.json()
        self.access_token = data["access_token"]
        self.refresh_token = data["refresh_token"]
        return data
    
    async def refresh_access_token(self):
        response = await self.client.post("/auth/refresh", json={
            "refresh_token": self.refresh_token
        })
        data = response.json()
        self.access_token = data["access_token"]
        return data
    
    async def make_request(self, method: str, path: str, **kwargs):
        headers = kwargs.pop("headers", {})
        headers["Authorization"] = f"Bearer {self.access_token}"
        
        response = await self.client.request(
            method, path, headers=headers, **kwargs
        )
        
        # Auto-refresh on 401
        if response.status_code == 401:
            await self.refresh_access_token()
            headers["Authorization"] = f"Bearer {self.access_token}"
            response = await self.client.request(
                method, path, headers=headers, **kwargs
            )
        
        return response
```

---

## JavaScript/TypeScript Integration

### Basic Setup

```typescript
// auth-client.ts
class AuthClient {
    private baseURL: string;
    private accessToken: string | null = null;
    private refreshToken: string | null = null;

    constructor(baseURL: string = 'http://localhost:8000/api/v1') {
        this.baseURL = baseURL;
    }

    async login(username: string, password: string, mfaToken?: string): Promise<AuthResponse> {
        const response = await fetch(`${this.baseURL}/auth/login`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                username,
                password,
                mfa_token: mfaToken
            })
        });

        if (!response.ok) {
            throw new Error(`Login failed: ${response.statusText}`);
        }

        const data = await response.json();
        this.accessToken = data.access_token;
        this.refreshToken = data.refresh_token;
        
        // Store tokens securely
        this.storeTokens(data.access_token, data.refresh_token);
        
        return data;
    }

    async makeAuthenticatedRequest(
        path: string,
        options: RequestInit = {}
    ): Promise<Response> {
        const headers = new Headers(options.headers);
        headers.set('Authorization', `Bearer ${this.accessToken}`);

        let response = await fetch(`${this.baseURL}${path}`, {
            ...options,
            headers
        });

        // Handle token refresh
        if (response.status === 401 && this.refreshToken) {
            await this.refreshAccessToken();
            headers.set('Authorization', `Bearer ${this.accessToken}`);
            response = await fetch(`${this.baseURL}${path}`, {
                ...options,
                headers
            });
        }

        return response;
    }

    private async refreshAccessToken(): Promise<void> {
        const response = await fetch(`${this.baseURL}/auth/refresh`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                refresh_token: this.refreshToken
            })
        });

        if (!response.ok) {
            throw new Error('Token refresh failed');
        }

        const data = await response.json();
        this.accessToken = data.access_token;
        this.storeTokens(data.access_token, this.refreshToken!);
    }

    private storeTokens(accessToken: string, refreshToken: string): void {
        // Use secure storage in production
        // For web: Consider httpOnly cookies or secure localStorage
        // For mobile: Use secure keychain/keystore
        localStorage.setItem('access_token', accessToken);
        localStorage.setItem('refresh_token', refreshToken);
    }
}
```

### React Hook Example

```typescript
// useAuth.tsx
import { useState, useEffect, createContext, useContext } from 'react';

interface AuthContextType {
    user: User | null;
    login: (username: string, password: string, mfaToken?: string) => Promise<void>;
    logout: () => Promise<void>;
    isAuthenticated: boolean;
}

const AuthContext = createContext<AuthContextType | null>(null);

export function AuthProvider({ children }: { children: React.ReactNode }) {
    const [user, setUser] = useState<User | null>(null);
    const [authClient] = useState(() => new AuthClient());

    const login = async (username: string, password: string, mfaToken?: string) => {
        try {
            const response = await authClient.login(username, password, mfaToken);
            // Fetch user details
            const userResponse = await authClient.makeAuthenticatedRequest('/auth/me');
            const userData = await userResponse.json();
            setUser(userData);
        } catch (error) {
            console.error('Login failed:', error);
            throw error;
        }
    };

    const logout = async () => {
        try {
            await authClient.makeAuthenticatedRequest('/auth/logout', {
                method: 'POST',
                body: JSON.stringify({ all_devices: false })
            });
        } finally {
            setUser(null);
            localStorage.removeItem('access_token');
            localStorage.removeItem('refresh_token');
        }
    };

    return (
        <AuthContext.Provider value={{
            user,
            login,
            logout,
            isAuthenticated: !!user
        }}>
            {children}
        </AuthContext.Provider>
    );
}

export const useAuth = () => {
    const context = useContext(AuthContext);
    if (!context) {
        throw new Error('useAuth must be used within AuthProvider');
    }
    return context;
};
```

---

## MFA Integration

### Setting Up MFA

```python
# Python example
async def setup_mfa(access_token: str):
    """Complete MFA setup process"""
    
    # 1. Initialize MFA setup
    headers = {"Authorization": f"Bearer {access_token}"}
    response = await client.post("/auth/mfa/setup", headers=headers)
    setup_data = response.json()
    
    secret = setup_data["secret"]
    qr_code = setup_data["qr_code"]
    backup_codes = setup_data["backup_codes"]
    
    # 2. Display QR code to user
    print(f"Scan this QR code with your authenticator app:")
    print(f"Or manually enter this secret: {secret}")
    
    # 3. Get TOTP from user
    user_token = input("Enter the 6-digit code from your app: ")
    
    # 4. Verify and enable MFA
    headers["X-MFA-Secret"] = secret
    response = await client.post("/auth/mfa/verify", 
        json={"token": user_token},
        headers=headers
    )
    
    if response.status_code == 200:
        print("MFA enabled successfully!")
        print(f"Save these backup codes: {backup_codes}")
        return True
    return False
```

### JavaScript MFA Setup

```javascript
class MFASetup {
    constructor(authClient) {
        this.authClient = authClient;
    }

    async initializeSetup() {
        const response = await this.authClient.makeAuthenticatedRequest(
            '/auth/mfa/setup',
            { method: 'POST' }
        );
        
        const data = await response.json();
        this.secret = data.secret;
        this.qrCode = data.qr_code;
        this.backupCodes = data.backup_codes;
        
        return data;
    }

    async verifyAndEnable(totpToken) {
        const response = await this.authClient.makeAuthenticatedRequest(
            '/auth/mfa/verify',
            {
                method: 'POST',
                headers: {
                    'X-MFA-Secret': this.secret,
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({ token: totpToken })
            }
        );
        
        if (response.ok) {
            // Save backup codes securely
            this.saveBackupCodes(this.backupCodes);
            return true;
        }
        return false;
    }

    saveBackupCodes(codes) {
        // In production, show to user and ensure they save them
        console.log('Backup codes:', codes);
        alert('Please save your backup codes in a secure location!');
    }
}
```

---

## Password Reset Flow

### Implementing Password Reset

```python
# Python implementation
class PasswordResetFlow:
    def __init__(self, client):
        self.client = client
    
    async def request_reset(self, email: str):
        """Request password reset email"""
        response = await self.client.post("/auth/forgot-password", json={
            "email": email
        })
        # Always returns success for security
        return response.status_code == 200
    
    async def reset_password(self, token: str, new_password: str):
        """Reset password with token from email"""
        response = await self.client.post("/auth/reset-password", json={
            "token": token,
            "new_password": new_password
        })
        return response.json()

# Usage
reset_flow = PasswordResetFlow(client)

# Request reset
await reset_flow.request_reset("user@example.com")
print("Check your email for reset instructions")

# User clicks link in email, extracts token
reset_token = "eyJhbGciOiJIUzI1NiIs..."

# Reset password
result = await reset_flow.reset_password(reset_token, "NewSecurePass123!")
print(result["message"])
```

### React Password Reset Component

```tsx
// PasswordReset.tsx
import React, { useState } from 'react';

export function PasswordResetRequest() {
    const [email, setEmail] = useState('');
    const [submitted, setSubmitted] = useState(false);

    const handleSubmit = async (e: React.FormEvent) => {
        e.preventDefault();
        
        const response = await fetch('/api/v1/auth/forgot-password', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ email })
        });

        if (response.ok) {
            setSubmitted(true);
        }
    };

    if (submitted) {
        return (
            <div className="alert alert-success">
                If an account exists with that email, you will receive 
                password reset instructions.
            </div>
        );
    }

    return (
        <form onSubmit={handleSubmit}>
            <input
                type="email"
                value={email}
                onChange={(e) => setEmail(e.target.value)}
                placeholder="Enter your email"
                required
            />
            <button type="submit">Request Password Reset</button>
        </form>
    );
}

export function PasswordResetConfirm({ token }: { token: string }) {
    const [password, setPassword] = useState('');
    const [confirmPassword, setConfirmPassword] = useState('');
    const [success, setSuccess] = useState(false);
    const [error, setError] = useState('');

    const handleSubmit = async (e: React.FormEvent) => {
        e.preventDefault();
        
        if (password !== confirmPassword) {
            setError('Passwords do not match');
            return;
        }

        const response = await fetch('/api/v1/auth/reset-password', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                token,
                new_password: password
            })
        });

        if (response.ok) {
            setSuccess(true);
        } else {
            const data = await response.json();
            setError(data.detail || 'Reset failed');
        }
    };

    if (success) {
        return (
            <div className="alert alert-success">
                Password reset successful! You can now login with your new password.
            </div>
        );
    }

    return (
        <form onSubmit={handleSubmit}>
            {error && <div className="alert alert-danger">{error}</div>}
            
            <input
                type="password"
                value={password}
                onChange={(e) => setPassword(e.target.value)}
                placeholder="New password"
                minLength={8}
                required
            />
            
            <input
                type="password"
                value={confirmPassword}
                onChange={(e) => setConfirmPassword(e.target.value)}
                placeholder="Confirm password"
                minLength={8}
                required
            />
            
            <button type="submit">Reset Password</button>
        </form>
    );
}
```

---

## Error Handling

### Comprehensive Error Handling

```python
# Python error handling
from enum import Enum
from typing import Optional

class AuthError(Enum):
    INVALID_CREDENTIALS = "invalid_credentials"
    ACCOUNT_LOCKED = "account_locked"
    MFA_REQUIRED = "mfa_required"
    MFA_INVALID = "mfa_invalid"
    TOKEN_EXPIRED = "token_expired"
    TOKEN_INVALID = "token_invalid"
    RATE_LIMITED = "rate_limited"
    SERVER_ERROR = "server_error"

class AuthException(Exception):
    def __init__(self, error_type: AuthError, message: str, retry_after: Optional[int] = None):
        self.error_type = error_type
        self.message = message
        self.retry_after = retry_after
        super().__init__(message)

async def handle_auth_response(response):
    """Handle authentication API responses with proper error handling"""
    
    if response.status_code == 200:
        return response.json()
    
    error_data = response.json() if response.content else {}
    detail = error_data.get("detail", "Unknown error")
    
    if response.status_code == 401:
        if "MFA" in detail:
            raise AuthException(AuthError.MFA_REQUIRED, detail)
        raise AuthException(AuthError.INVALID_CREDENTIALS, detail)
    
    elif response.status_code == 429:
        retry_after = response.headers.get("X-RateLimit-Reset")
        raise AuthException(
            AuthError.RATE_LIMITED,
            detail,
            retry_after=int(retry_after) if retry_after else None
        )
    
    elif response.status_code == 423:
        raise AuthException(AuthError.ACCOUNT_LOCKED, detail)
    
    elif response.status_code >= 500:
        raise AuthException(AuthError.SERVER_ERROR, detail)
    
    raise AuthException(AuthError.SERVER_ERROR, detail)

# Usage with retry logic
async def login_with_retry(username: str, password: str, max_retries: int = 3):
    for attempt in range(max_retries):
        try:
            response = await client.post("/auth/login", json={
                "username": username,
                "password": password
            })
            return await handle_auth_response(response)
            
        except AuthException as e:
            if e.error_type == AuthError.RATE_LIMITED and e.retry_after:
                print(f"Rate limited. Waiting {e.retry_after} seconds...")
                await asyncio.sleep(e.retry_after)
                continue
            
            if e.error_type == AuthError.SERVER_ERROR and attempt < max_retries - 1:
                print(f"Server error. Retrying in {2 ** attempt} seconds...")
                await asyncio.sleep(2 ** attempt)
                continue
            
            raise
```

### JavaScript Error Handling

```typescript
// Error handling utilities
class AuthError extends Error {
    constructor(
        public code: string,
        message: string,
        public statusCode: number,
        public retryAfter?: number
    ) {
        super(message);
        this.name = 'AuthError';
    }
}

class AuthErrorHandler {
    static async handleResponse(response: Response): Promise<any> {
        if (response.ok) {
            return response.json();
        }

        const errorData = await response.json().catch(() => ({}));
        const detail = errorData.detail || 'Unknown error';

        switch (response.status) {
            case 401:
                if (detail.includes('MFA')) {
                    throw new AuthError('MFA_REQUIRED', detail, 401);
                }
                throw new AuthError('INVALID_CREDENTIALS', detail, 401);

            case 429:
                const retryAfter = response.headers.get('X-RateLimit-Reset');
                throw new AuthError(
                    'RATE_LIMITED',
                    detail,
                    429,
                    retryAfter ? parseInt(retryAfter) : undefined
                );

            case 423:
                throw new AuthError('ACCOUNT_LOCKED', detail, 423);

            default:
                throw new AuthError('SERVER_ERROR', detail, response.status);
        }
    }

    static async withRetry<T>(
        fn: () => Promise<T>,
        maxRetries: number = 3
    ): Promise<T> {
        let lastError: AuthError | undefined;

        for (let attempt = 0; attempt < maxRetries; attempt++) {
            try {
                return await fn();
            } catch (error) {
                if (!(error instanceof AuthError)) {
                    throw error;
                }

                lastError = error;

                // Handle rate limiting
                if (error.code === 'RATE_LIMITED' && error.retryAfter) {
                    await new Promise(resolve => 
                        setTimeout(resolve, error.retryAfter! * 1000)
                    );
                    continue;
                }

                // Exponential backoff for server errors
                if (error.code === 'SERVER_ERROR' && attempt < maxRetries - 1) {
                    await new Promise(resolve => 
                        setTimeout(resolve, Math.pow(2, attempt) * 1000)
                    );
                    continue;
                }

                throw error;
            }
        }

        throw lastError;
    }
}
```

---

## Rate Limiting

### Handling Rate Limits

```python
# Python rate limit handler
import time
from typing import Dict, Optional

class RateLimitHandler:
    def __init__(self):
        self.limits: Dict[str, dict] = {}
    
    def update_limits(self, endpoint: str, headers: dict):
        """Update rate limit info from response headers"""
        self.limits[endpoint] = {
            "remaining": int(headers.get("X-RateLimit-Remaining", 0)),
            "reset": int(headers.get("X-RateLimit-Reset", 0)),
            "limit": int(headers.get("X-RateLimit-Limit", 100))
        }
    
    def should_wait(self, endpoint: str) -> Optional[int]:
        """Check if we should wait before making request"""
        if endpoint not in self.limits:
            return None
        
        limit_info = self.limits[endpoint]
        if limit_info["remaining"] <= 0:
            wait_time = limit_info["reset"] - int(time.time())
            return max(0, wait_time)
        
        return None
    
    async def make_request_with_limits(self, client, method: str, url: str, **kwargs):
        """Make request with automatic rate limit handling"""
        endpoint = url.split("/")[-1]  # Simple endpoint extraction
        
        # Check if we need to wait
        wait_time = self.should_wait(endpoint)
        if wait_time:
            print(f"Rate limited. Waiting {wait_time} seconds...")
            await asyncio.sleep(wait_time)
        
        # Make request
        response = await client.request(method, url, **kwargs)
        
        # Update limits
        self.update_limits(endpoint, response.headers)
        
        # Handle 429 response
        if response.status_code == 429:
            retry_after = int(response.headers.get("Retry-After", 60))
            print(f"Rate limited. Retrying after {retry_after} seconds...")
            await asyncio.sleep(retry_after)
            return await self.make_request_with_limits(client, method, url, **kwargs)
        
        return response
```

---

## Testing Authentication

### Unit Tests

```python
# test_auth.py
import pytest
from httpx import AsyncClient
import pyotp

@pytest.fixture
async def auth_client():
    async with AsyncClient(app=app, base_url="http://test") as client:
        yield client

@pytest.mark.asyncio
async def test_complete_auth_flow(auth_client):
    """Test complete authentication flow including MFA"""
    
    # 1. Register user
    register_response = await auth_client.post("/auth/register", json={
        "username": "testuser",
        "email": "test@example.com",
        "password": "TestPass123!"
    })
    assert register_response.status_code == 201
    
    # 2. Login
    login_response = await auth_client.post("/auth/login", json={
        "username": "testuser",
        "password": "TestPass123!"
    })
    assert login_response.status_code == 200
    tokens = login_response.json()
    access_token = tokens["access_token"]
    
    # 3. Setup MFA
    headers = {"Authorization": f"Bearer {access_token}"}
    mfa_setup = await auth_client.post("/auth/mfa/setup", headers=headers)
    assert mfa_setup.status_code == 200
    
    secret = mfa_setup.json()["secret"]
    backup_codes = mfa_setup.json()["backup_codes"]
    
    # 4. Verify MFA with valid TOTP
    totp = pyotp.TOTP(secret)
    verify_response = await auth_client.post(
        "/auth/mfa/verify",
        json={"token": totp.now()},
        headers={**headers, "X-MFA-Secret": secret}
    )
    assert verify_response.status_code == 200
    
    # 5. Logout
    logout_response = await auth_client.post(
        "/auth/logout",
        json={"all_devices": False},
        headers=headers
    )
    assert logout_response.status_code == 200
    
    # 6. Login with MFA
    login_mfa_response = await auth_client.post("/auth/login", json={
        "username": "testuser",
        "password": "TestPass123!",
        "mfa_token": totp.now()
    })
    assert login_mfa_response.status_code == 200

@pytest.mark.asyncio
async def test_password_reset_flow(auth_client):
    """Test password reset flow"""
    
    # Request reset
    reset_request = await auth_client.post("/auth/forgot-password", json={
        "email": "test@example.com"
    })
    assert reset_request.status_code == 200
    
    # In test environment, capture the token from mock email
    # This would be retrieved from the email in production
    reset_token = get_reset_token_from_mock_email()
    
    # Reset password
    reset_response = await auth_client.post("/auth/reset-password", json={
        "token": reset_token,
        "new_password": "NewTestPass123!"
    })
    assert reset_response.status_code == 200
    
    # Verify old password doesn't work
    old_login = await auth_client.post("/auth/login", json={
        "username": "testuser",
        "password": "TestPass123!"
    })
    assert old_login.status_code == 401
    
    # Verify new password works
    new_login = await auth_client.post("/auth/login", json={
        "username": "testuser",
        "password": "NewTestPass123!"
    })
    assert new_login.status_code == 200
```

---

## Security Considerations

### Token Storage Best Practices

```javascript
// Secure token storage for web applications
class SecureTokenStorage {
    constructor() {
        this.storage = this.detectStorageMethod();
    }

    detectStorageMethod() {
        // For web apps, prefer httpOnly cookies
        // For mobile/desktop apps, use secure storage
        if (typeof window !== 'undefined') {
            // Web environment
            return {
                store: this.storeInMemory.bind(this),
                retrieve: this.retrieveFromMemory.bind(this),
                clear: this.clearMemory.bind(this)
            };
        }
        // Node.js or other environments
        return {
            store: this.storeInKeychain.bind(this),
            retrieve: this.retrieveFromKeychain.bind(this),
            clear: this.clearKeychain.bind(this)
        };
    }

    // In-memory storage (most secure for web)
    private tokens = new Map();

    storeInMemory(key: string, value: string) {
        this.tokens.set(key, value);
    }

    retrieveFromMemory(key: string): string | null {
        return this.tokens.get(key) || null;
    }

    clearMemory() {
        this.tokens.clear();
    }

    // For production, implement platform-specific secure storage
    // iOS: Keychain
    // Android: Keystore
    // Desktop: OS credential manager
}
```

### CORS Configuration

```python
# FastAPI CORS configuration
from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://yourdomain.com",
        "https://app.yourdomain.com"
    ],
    allow_credentials=True,  # Required for auth cookies
    allow_methods=["GET", "POST", "PUT", "DELETE"],
    allow_headers=["Authorization", "Content-Type", "X-MFA-Secret"],
    expose_headers=["X-RateLimit-Remaining", "X-RateLimit-Reset"],
    max_age=3600  # Cache preflight requests
)
```

---

## Monitoring and Logging

### Authentication Event Logging

```python
# Structured logging for authentication events
import structlog
from datetime import datetime

logger = structlog.get_logger()

class AuthEventLogger:
    @staticmethod
    def log_login_attempt(username: str, ip: str, success: bool, mfa_used: bool = False):
        logger.info(
            "login_attempt",
            username=username,
            ip_address=ip,
            success=success,
            mfa_used=mfa_used,
            timestamp=datetime.utcnow().isoformat()
        )
    
    @staticmethod
    def log_mfa_setup(user_id: int, success: bool):
        logger.info(
            "mfa_setup",
            user_id=user_id,
            success=success,
            timestamp=datetime.utcnow().isoformat()
        )
    
    @staticmethod
    def log_password_reset(email: str, ip: str):
        logger.info(
            "password_reset_requested",
            email=email,
            ip_address=ip,
            timestamp=datetime.utcnow().isoformat()
        )
    
    @staticmethod
    def log_suspicious_activity(user_id: int, activity_type: str, details: dict):
        logger.warning(
            "suspicious_activity",
            user_id=user_id,
            activity_type=activity_type,
            details=details,
            timestamp=datetime.utcnow().isoformat()
        )
```

---

## Migration from Legacy Systems

### Migrating from Basic Auth to JWT

```python
# Migration script for existing users
async def migrate_to_jwt_auth(legacy_db, new_db):
    """Migrate users from basic auth to JWT system"""
    
    # Get all users from legacy system
    legacy_users = await legacy_db.fetch_all("SELECT * FROM users")
    
    for user in legacy_users:
        # Hash password if stored in plain text (NEVER do this in production!)
        if is_plain_text(user["password"]):
            password_hash = password_service.hash_password(user["password"])
        else:
            password_hash = user["password"]
        
        # Create user in new system
        await new_db.execute("""
            INSERT INTO users (
                username, email, password_hash, created_at, is_active
            ) VALUES ($1, $2, $3, $4, $5)
        """, user["username"], user["email"], password_hash, 
            user["created_at"], True)
        
        # Send email about migration
        await email_service.send_migration_notice(
            user["email"],
            user["username"]
        )
    
    print(f"Migrated {len(legacy_users)} users")
```

---

## Support Resources

- API Documentation: `/docs` (Swagger UI)
- ReDoc Documentation: `/redoc`
- Health Check: `/health`
- Version Info: `/version`

For issues or questions, consult the main SECURITY_DOCUMENTATION.md file or contact the security team.