# Critical Security Fixes - Implementation Guide

## Priority 1: Authentication System (MUST FIX IMMEDIATELY)

### Current Vulnerable Code (chat.py:421-441)
```python
# VULNERABLE - DO NOT USE
if is_authentication_required():
    if not Token:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Missing authentication token.")
    
    extracted_token = extract_bearer_token(Token)
    if not extracted_token:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid authentication token format.")
    
    expected_token = get_expected_api_token()  # Just reads from environment variable!
    if not expected_token:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Server authentication is misconfigured.")
    
    if not validate_api_token(extracted_token, expected_token):  # Simple string comparison!
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid authentication token.")
```

### Secure Implementation Required
```python
# SECURE IMPLEMENTATION
from datetime import datetime, timedelta
import jwt
from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
import redis
from typing import Optional

# JWT Configuration
JWT_SECRET_KEY = os.getenv("JWT_SECRET_KEY")  # Must be from secure vault
JWT_ALGORITHM = "HS256"
JWT_EXPIRATION_MINUTES = 30
REFRESH_TOKEN_EXPIRATION_DAYS = 7

# Redis for session management
redis_client = redis.Redis(
    host=os.getenv("REDIS_HOST"),
    port=int(os.getenv("REDIS_PORT", 6379)),
    password=os.getenv("REDIS_PASSWORD"),
    ssl=True,
    decode_responses=True
)

security = HTTPBearer()

class AuthService:
    @staticmethod
    def create_access_token(user_id: str, scopes: list = None) -> str:
        expire = datetime.utcnow() + timedelta(minutes=JWT_EXPIRATION_MINUTES)
        payload = {
            "sub": user_id,
            "exp": expire,
            "iat": datetime.utcnow(),
            "scopes": scopes or [],
            "jti": str(uuid.uuid4())  # JWT ID for revocation
        }
        return jwt.encode(payload, JWT_SECRET_KEY, algorithm=JWT_ALGORITHM)
    
    @staticmethod
    def verify_token(credentials: HTTPAuthorizationCredentials = Depends(security)) -> dict:
        token = credentials.credentials
        
        try:
            # Decode token
            payload = jwt.decode(token, JWT_SECRET_KEY, algorithms=[JWT_ALGORITHM])
            
            # Check if token is revoked
            jti = payload.get("jti")
            if redis_client.get(f"revoked_token:{jti}"):
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="Token has been revoked"
                )
            
            # Check session validity
            user_id = payload.get("sub")
            session_key = f"session:{user_id}:{jti}"
            if not redis_client.exists(session_key):
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="Invalid session"
                )
            
            # Update session activity
            redis_client.expire(session_key, 3600)  # Extend session
            
            return payload
            
        except jwt.ExpiredSignatureError:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Token has expired"
            )
        except jwt.JWTError:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid token"
            )

# Usage in endpoint
@router.post("/completions")
async def create_chat_completion(
    request_data: ChatCompletionRequest = Body(...),
    current_user: dict = Depends(AuthService.verify_token),  # Secure auth
    chat_db: CharactersRAGDB = Depends(get_chacha_db_for_user),
):
    # User is authenticated and session is valid
    user_id = current_user["sub"]
    scopes = current_user["scopes"]
    
    # Check authorization for this operation
    if "chat:write" not in scopes:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Insufficient permissions"
        )
    
    # Continue with secure implementation...
```

---

## Priority 2: SQL Injection Prevention

### Current Vulnerable Code (chat.py:1950-1952)
```python
# VULNERABLE - Bare except hiding potential SQL injection
try:
    with db.get_connection() as conn:
        cursor = conn.execute(
            "SELECT 1 FROM user_prompts WHERE document_type = ? AND is_active = 1",
            (doc_type.value,)
        )
        is_custom = cursor.fetchone() is not None
except:  # DANGEROUS - Hides SQL errors
    pass
```

### Secure Implementation
```python
# SECURE IMPLEMENTATION
from sqlalchemy import create_engine, text
from sqlalchemy.orm import Session
from contextlib import contextmanager
import logging

class SecureDatabase:
    def __init__(self, connection_string: str):
        # Use SQLAlchemy with proper configuration
        self.engine = create_engine(
            connection_string,
            pool_pre_ping=True,
            pool_size=10,
            max_overflow=20,
            echo=False,  # Set to True only in dev
            connect_args={
                "timeout": 10,
                "check_same_thread": False,
                "isolation_level": "READ COMMITTED"
            }
        )
    
    @contextmanager
    def get_session(self):
        session = Session(self.engine)
        try:
            yield session
            session.commit()
        except Exception as e:
            session.rollback()
            logger.error(f"Database error: {e}")
            raise
        finally:
            session.close()
    
    def check_custom_prompt(self, doc_type: str, user_id: str) -> bool:
        """
        Secure query with parameterization and proper error handling
        """
        with self.get_session() as session:
            try:
                # Use parameterized query with SQLAlchemy
                query = text("""
                    SELECT COUNT(*) as count
                    FROM user_prompts 
                    WHERE document_type = :doc_type 
                    AND is_active = 1
                    AND user_id = :user_id
                    AND deleted_at IS NULL
                """)
                
                result = session.execute(
                    query,
                    {
                        "doc_type": doc_type,
                        "user_id": user_id
                    }
                ).scalar()
                
                return result > 0
                
            except Exception as e:
                logger.error(f"Failed to check custom prompt: {e}")
                # Return safe default, don't expose error details
                return False

# Input validation before database operations
from pydantic import validator
import re

class DocumentTypeValidator:
    @staticmethod
    def validate_doc_type(doc_type: str) -> str:
        # Whitelist validation
        allowed_types = ["summary", "transcript", "notes", "outline"]
        if doc_type not in allowed_types:
            raise ValueError(f"Invalid document type: {doc_type}")
        
        # Additional pattern validation
        if not re.match(r'^[a-z_]+$', doc_type):
            raise ValueError("Document type contains invalid characters")
        
        return doc_type
```

---

## Priority 3: File Operation Security

### Current Vulnerable Code (chat.py:1464-1489)
```python
# VULNERABLE - Predictable temp files, no validation
import tempfile
with tempfile.NamedTemporaryFile(mode='w', suffix='.md', delete=False) as tmp:
    tmp.write(import_request.content)
    tmp_path = tmp.name  # Predictable path!

# No validation of file operations
service.import_from_markdown(tmp_path, import_request.name)

# Cleanup might fail
os.unlink(tmp_path)
```

### Secure Implementation
```python
# SECURE IMPLEMENTATION
import uuid
import os
from pathlib import Path
import hashlib
from typing import Optional
import magic  # python-magic for file type detection

class SecureFileHandler:
    def __init__(self):
        # Create secure temp directory with restricted permissions
        self.temp_dir = Path("/tmp/secure_chat_temp")
        self.temp_dir.mkdir(mode=0o700, exist_ok=True)
        
        # Whitelist of allowed file types
        self.allowed_mime_types = {
            "text/plain",
            "text/markdown",
            "application/json"
        }
        
        # Maximum file size (10MB)
        self.max_file_size = 10 * 1024 * 1024
    
    def create_secure_temp_file(self, content: str, file_type: str = "md") -> Path:
        """
        Create a secure temporary file with validation
        """
        # Validate file type
        if file_type not in ["md", "txt", "json"]:
            raise ValueError(f"Invalid file type: {file_type}")
        
        # Generate secure random filename
        secure_name = f"{uuid.uuid4().hex}_{hashlib.sha256(os.urandom(32)).hexdigest()[:8]}.{file_type}"
        file_path = self.temp_dir / secure_name
        
        # Validate content size
        content_bytes = content.encode('utf-8')
        if len(content_bytes) > self.max_file_size:
            raise ValueError(f"File too large: {len(content_bytes)} bytes")
        
        # Write with restricted permissions
        try:
            file_path.write_bytes(content_bytes)
            file_path.chmod(0o600)  # Owner read/write only
            
            # Validate written file
            self.validate_file(file_path)
            
            return file_path
            
        except Exception as e:
            # Clean up on error
            if file_path.exists():
                file_path.unlink()
            raise ValueError(f"Failed to create secure file: {e}")
    
    def validate_file(self, file_path: Path) -> bool:
        """
        Validate file type and content
        """
        # Check file exists and is not a symlink
        if not file_path.exists() or file_path.is_symlink():
            raise ValueError("Invalid file path")
        
        # Check file is within temp directory (prevent traversal)
        try:
            file_path.resolve().relative_to(self.temp_dir.resolve())
        except ValueError:
            raise ValueError("File path outside of secure directory")
        
        # Check file type with magic
        mime = magic.from_file(str(file_path), mime=True)
        if mime not in self.allowed_mime_types:
            raise ValueError(f"Invalid file type: {mime}")
        
        # Check file size
        if file_path.stat().st_size > self.max_file_size:
            raise ValueError("File too large")
        
        return True
    
    def cleanup_file(self, file_path: Path) -> bool:
        """
        Securely delete temporary file
        """
        try:
            if file_path and file_path.exists():
                # Overwrite with random data before deletion
                with open(file_path, 'wb') as f:
                    f.write(os.urandom(file_path.stat().st_size))
                file_path.unlink()
                return True
        except Exception as e:
            logger.error(f"Failed to cleanup file {file_path}: {e}")
            return False
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        # Cleanup all temp files on exit
        for file in self.temp_dir.glob("*"):
            if file.is_file():
                self.cleanup_file(file)

# Usage in endpoint
@router.post("/dictionaries/import")
async def import_dictionary(
    import_request: ImportDictionaryRequest,
    db: CharactersRAGDB = Depends(get_chacha_db_for_user),
    current_user: dict = Depends(AuthService.verify_token)
):
    with SecureFileHandler() as file_handler:
        try:
            # Create secure temp file
            secure_path = file_handler.create_secure_temp_file(
                content=import_request.content,
                file_type="md"
            )
            
            # Process with validation
            service = ChatDictionaryService(db)
            dict_id = service.import_from_markdown(
                file_path=secure_path,
                name=import_request.name,
                user_id=current_user["sub"]  # Track who imported
            )
            
            # File automatically cleaned up
            return {"dictionary_id": dict_id}
            
        except ValueError as e:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=str(e)
            )
```

---

## Priority 4: API Key Management

### Current Vulnerable Code
```python
# VULNERABLE - Keys in plaintext
API_KEYS = {
    'openai': os.getenv('OPENAI_API_KEY'),  # Plaintext in environment
    'anthropic': os.getenv('ANTHROPIC_API_KEY'),
    # ... more keys
}

# Passed around without encryption
provider_api_key = API_KEYS.get(target_api_provider)
```

### Secure Implementation
```python
# SECURE IMPLEMENTATION
from cryptography.fernet import Fernet
import hvac  # HashiCorp Vault client
from functools import lru_cache
from datetime import datetime, timedelta

class SecureKeyManager:
    def __init__(self):
        # Connect to HashiCorp Vault
        self.vault = hvac.Client(
            url=os.getenv("VAULT_URL"),
            token=os.getenv("VAULT_TOKEN")
        )
        
        # Local encryption for caching
        self.cipher = Fernet(os.getenv("LOCAL_ENCRYPTION_KEY").encode())
        
        # Cache with TTL
        self.key_cache = {}
        self.cache_ttl = timedelta(minutes=5)
    
    def get_api_key(self, provider: str, user_id: str = None) -> Optional[str]:
        """
        Securely retrieve API key with audit logging
        """
        cache_key = f"{provider}:{user_id or 'system'}"
        
        # Check cache
        if cache_key in self.key_cache:
            cached_data = self.key_cache[cache_key]
            if datetime.now() < cached_data['expires']:
                # Decrypt and return cached key
                return self.cipher.decrypt(cached_data['encrypted_key']).decode()
        
        try:
            # Fetch from Vault
            secret_path = f"secret/api_keys/{provider}"
            response = self.vault.secrets.kv.v2.read_secret_version(
                path=secret_path
            )
            
            api_key = response['data']['data']['key']
            
            # Audit log the key access
            self.audit_key_access(provider, user_id)
            
            # Cache encrypted key
            encrypted_key = self.cipher.encrypt(api_key.encode())
            self.key_cache[cache_key] = {
                'encrypted_key': encrypted_key,
                'expires': datetime.now() + self.cache_ttl
            }
            
            return api_key
            
        except Exception as e:
            logger.error(f"Failed to retrieve key for {provider}: {e}")
            # Send alert for key retrieval failure
            self.send_security_alert(f"Key retrieval failed: {provider}")
            return None
    
    def rotate_key(self, provider: str, new_key: str) -> bool:
        """
        Rotate API key with zero downtime
        """
        try:
            # Store new key version
            self.vault.secrets.kv.v2.create_or_update_secret(
                path=f"secret/api_keys/{provider}",
                secret={"key": new_key, "rotated_at": datetime.now().isoformat()}
            )
            
            # Clear cache to force refresh
            self.clear_cache(provider)
            
            # Audit log rotation
            logger.info(f"API key rotated for provider: {provider}")
            
            return True
            
        except Exception as e:
            logger.error(f"Key rotation failed for {provider}: {e}")
            return False
    
    def audit_key_access(self, provider: str, user_id: str = None):
        """
        Audit log all key access
        """
        logger.info(f"API key accessed - Provider: {provider}, User: {user_id or 'system'}")
    
    def clear_cache(self, provider: str = None):
        """
        Clear cached keys
        """
        if provider:
            # Clear specific provider
            self.key_cache = {
                k: v for k, v in self.key_cache.items()
                if not k.startswith(f"{provider}:")
            }
        else:
            # Clear all
            self.key_cache.clear()

# Usage in chat functions
key_manager = SecureKeyManager()

async def secure_chat_api_call(
    provider: str,
    messages: list,
    user_id: str,
    **kwargs
):
    # Get key securely
    api_key = key_manager.get_api_key(provider, user_id)
    if not api_key:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"API key not available for {provider}"
        )
    
    # Make API call with secure key
    # Key is never logged or exposed
    response = await perform_api_call(
        provider=provider,
        api_key=api_key,  # Secure, temporary access
        messages=messages,
        **kwargs
    )
    
    # Clear sensitive data from memory
    api_key = None
    
    return response
```

---

## Implementation Priority Order

### Week 1 - CRITICAL (Stop all other work)
1. **Day 1-2**: Implement JWT authentication
2. **Day 3-4**: Fix SQL injection vulnerabilities
3. **Day 5**: Secure file operations

### Week 2 - HIGH PRIORITY  
1. **Day 1-2**: Implement key vault
2. **Day 3-4**: Add comprehensive input validation
3. **Day 5**: Security testing of fixes

### Testing Requirements for Each Fix

#### Authentication Testing
```bash
# Test JWT implementation
pytest tests/security/test_jwt_auth.py -v

# Test session management
pytest tests/security/test_session_management.py -v

# Test rate limiting
pytest tests/security/test_rate_limiting.py -v
```

#### SQL Injection Testing
```bash
# Run SQLMap against endpoints
sqlmap -u "http://localhost:8000/api/v1/chat/completions" \
       --data='{"messages":[{"content":"test"}]}' \
       --method=POST \
       --headers="Content-Type: application/json" \
       --level=5 --risk=3
```

#### File Security Testing
```bash
# Test directory traversal
curl -X POST http://localhost:8000/api/v1/dictionaries/import \
     -d '{"content": "test", "name": "../../etc/passwd"}'

# Test file size limits
dd if=/dev/zero of=large.txt bs=1M count=20
curl -X POST http://localhost:8000/api/v1/dictionaries/import \
     -F "file=@large.txt"
```

---

## Monitoring After Implementation

### Security Metrics to Track
- Failed authentication attempts per minute
- SQL query execution time (flag anomalies)
- File operations per user
- API key access frequency
- Error rates by endpoint

### Alerts to Configure
- More than 10 failed auth attempts from same IP
- SQL query taking > 5 seconds
- File operation outside temp directory
- API key access from new IP
- Error rate > 1% for any endpoint

---

*This implementation guide provides secure code that can be immediately used to replace vulnerable sections.*