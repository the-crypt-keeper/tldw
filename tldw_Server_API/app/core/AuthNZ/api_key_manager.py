# api_key_manager.py
# Description: API key management with rotation, expiration, and revocation capabilities
#
# Imports
import secrets
import hashlib
from datetime import datetime, timedelta
from typing import Optional, Dict, Any, List
from enum import Enum
#
# 3rd-party imports
from loguru import logger
#
# Local imports
from tldw_Server_API.app.core.AuthNZ.database import DatabasePool, get_db_pool
from tldw_Server_API.app.core.AuthNZ.exceptions import DatabaseError, InvalidTokenError
from tldw_Server_API.app.core.AuthNZ.settings import get_settings

#######################################################################################################################
#
# Enums and Constants
#

class APIKeyStatus(Enum):
    """API key status states"""
    ACTIVE = "active"
    EXPIRED = "expired"
    REVOKED = "revoked"
    ROTATED = "rotated"

class APIKeyScope(Enum):
    """API key permission scopes"""
    READ = "read"
    WRITE = "write"
    ADMIN = "admin"
    SERVICE = "service"

#######################################################################################################################
#
# API Key Manager Class
#

class APIKeyManager:
    """Manages API keys with rotation, expiration, and revocation capabilities"""
    
    def __init__(self, db_pool: Optional[DatabasePool] = None):
        """Initialize API key manager"""
        self.db_pool = db_pool
        self._initialized = False
        self.settings = get_settings()
        self.key_prefix = "tldw_"  # Prefix for identifying our API keys
        self.key_length = 32  # Length of random part
        
    async def initialize(self):
        """Initialize database connection and ensure tables exist"""
        if self._initialized:
            return
            
        # Get database pool
        if not self.db_pool:
            self.db_pool = await get_db_pool()
        
        # Create API keys table if it doesn't exist
        await self._create_tables()
        
        self._initialized = True
        logger.info("APIKeyManager initialized")
    
    async def _create_tables(self):
        """Create API keys and related tables if they don't exist"""
        try:
            async with self.db_pool.transaction() as conn:
                if hasattr(conn, 'execute'):
                    # PostgreSQL
                    await conn.execute("""
                        CREATE TABLE IF NOT EXISTS api_keys (
                            id SERIAL PRIMARY KEY,
                            user_id INTEGER NOT NULL,
                            key_hash VARCHAR(64) UNIQUE NOT NULL,
                            key_prefix VARCHAR(16) NOT NULL,
                            name VARCHAR(255),
                            description TEXT,
                            scope VARCHAR(50) DEFAULT 'read',
                            status VARCHAR(20) DEFAULT 'active',
                            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                            expires_at TIMESTAMP,
                            last_used_at TIMESTAMP,
                            last_used_ip VARCHAR(45),
                            usage_count INTEGER DEFAULT 0,
                            rate_limit INTEGER,
                            allowed_ips TEXT,
                            metadata JSONB,
                            rotated_from INTEGER REFERENCES api_keys(id),
                            rotated_to INTEGER REFERENCES api_keys(id),
                            revoked_at TIMESTAMP,
                            revoked_by INTEGER,
                            revoke_reason TEXT,
                            FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
                        )
                    """)
                    
                    # Create indexes
                    await conn.execute("CREATE INDEX IF NOT EXISTS idx_api_keys_user_id ON api_keys(user_id)")
                    await conn.execute("CREATE INDEX IF NOT EXISTS idx_api_keys_key_hash ON api_keys(key_hash)")
                    await conn.execute("CREATE INDEX IF NOT EXISTS idx_api_keys_status ON api_keys(status)")
                    await conn.execute("CREATE INDEX IF NOT EXISTS idx_api_keys_expires_at ON api_keys(expires_at)")
                    
                    # Create API key audit log table
                    await conn.execute("""
                        CREATE TABLE IF NOT EXISTS api_key_audit_log (
                            id SERIAL PRIMARY KEY,
                            api_key_id INTEGER NOT NULL,
                            action VARCHAR(50) NOT NULL,
                            user_id INTEGER,
                            ip_address VARCHAR(45),
                            user_agent TEXT,
                            details JSONB,
                            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                            FOREIGN KEY (api_key_id) REFERENCES api_keys(id) ON DELETE CASCADE
                        )
                    """)
                    
                else:
                    # SQLite
                    await conn.execute("""
                        CREATE TABLE IF NOT EXISTS api_keys (
                            id INTEGER PRIMARY KEY AUTOINCREMENT,
                            user_id INTEGER NOT NULL,
                            key_hash TEXT UNIQUE NOT NULL,
                            key_prefix TEXT NOT NULL,
                            name TEXT,
                            description TEXT,
                            scope TEXT DEFAULT 'read',
                            status TEXT DEFAULT 'active',
                            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                            expires_at TIMESTAMP,
                            last_used_at TIMESTAMP,
                            last_used_ip TEXT,
                            usage_count INTEGER DEFAULT 0,
                            rate_limit INTEGER,
                            allowed_ips TEXT,
                            metadata TEXT,
                            rotated_from INTEGER REFERENCES api_keys(id),
                            rotated_to INTEGER REFERENCES api_keys(id),
                            revoked_at TIMESTAMP,
                            revoked_by INTEGER,
                            revoke_reason TEXT,
                            FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
                        )
                    """)
                    
                    # Create indexes
                    await conn.execute("CREATE INDEX IF NOT EXISTS idx_api_keys_user_id ON api_keys(user_id)")
                    await conn.execute("CREATE INDEX IF NOT EXISTS idx_api_keys_key_hash ON api_keys(key_hash)")
                    await conn.execute("CREATE INDEX IF NOT EXISTS idx_api_keys_status ON api_keys(status)")
                    await conn.execute("CREATE INDEX IF NOT EXISTS idx_api_keys_expires_at ON api_keys(expires_at)")
                    
                    # Create API key audit log table
                    await conn.execute("""
                        CREATE TABLE IF NOT EXISTS api_key_audit_log (
                            id INTEGER PRIMARY KEY AUTOINCREMENT,
                            api_key_id INTEGER NOT NULL,
                            action TEXT NOT NULL,
                            user_id INTEGER,
                            ip_address TEXT,
                            user_agent TEXT,
                            details TEXT,
                            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                            FOREIGN KEY (api_key_id) REFERENCES api_keys(id) ON DELETE CASCADE
                        )
                    """)
                    
                    await conn.commit()
                    
                logger.debug("API keys tables and indexes created/verified")
                
        except Exception as e:
            logger.error(f"Failed to create API keys tables: {e}")
            raise DatabaseError(f"Failed to create API keys tables: {e}")
    
    def generate_api_key(self) -> tuple[str, str]:
        """
        Generate a new API key
        
        Returns:
            Tuple of (full_key, key_hash)
            - full_key: The complete API key to give to the user
            - key_hash: The hash to store in the database
        """
        # Generate random key
        random_part = secrets.token_urlsafe(self.key_length)
        full_key = f"{self.key_prefix}{random_part}"
        
        # Hash for storage
        key_hash = hashlib.sha256(full_key.encode()).hexdigest()
        
        return full_key, key_hash
    
    def hash_api_key(self, api_key: str) -> str:
        """Hash an API key for comparison"""
        return hashlib.sha256(api_key.encode()).hexdigest()
    
    async def create_api_key(
        self,
        user_id: int,
        name: Optional[str] = None,
        description: Optional[str] = None,
        scope: str = "read",
        expires_in_days: Optional[int] = 90,
        rate_limit: Optional[int] = None,
        allowed_ips: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Create a new API key for a user
        
        Args:
            user_id: User ID who owns the key
            name: Optional name for the key
            description: Optional description
            scope: Permission scope (read, write, admin, service)
            expires_in_days: Days until expiration (None = no expiration)
            rate_limit: Custom rate limit for this key
            allowed_ips: List of allowed IP addresses
            metadata: Additional metadata
            
        Returns:
            Dictionary with key information including the actual key (only shown once)
        """
        if not self._initialized:
            await self.initialize()
        
        # Generate the key
        full_key, key_hash = self.generate_api_key()
        key_prefix = full_key[:10] + "..."  # Store prefix for identification
        
        # Calculate expiration
        expires_at = None
        if expires_in_days:
            expires_at = datetime.utcnow() + timedelta(days=expires_in_days)
        
        try:
            async with self.db_pool.transaction() as conn:
                if hasattr(conn, 'fetchval'):
                    # PostgreSQL
                    import json
                    key_id = await conn.fetchval(
                        """
                        INSERT INTO api_keys (
                            user_id, key_hash, key_prefix, name, description,
                            scope, expires_at, rate_limit, allowed_ips, metadata
                        )
                        VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10)
                        RETURNING id
                        """,
                        user_id, key_hash, key_prefix, name, description,
                        scope, expires_at, rate_limit,
                        json.dumps(allowed_ips) if allowed_ips else None,
                        json.dumps(metadata) if metadata else None
                    )
                else:
                    # SQLite
                    import json
                    cursor = await conn.execute(
                        """
                        INSERT INTO api_keys (
                            user_id, key_hash, key_prefix, name, description,
                            scope, expires_at, rate_limit, allowed_ips, metadata
                        )
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                        """,
                        (user_id, key_hash, key_prefix, name, description,
                         scope, expires_at.isoformat() if expires_at else None,
                         rate_limit,
                         json.dumps(allowed_ips) if allowed_ips else None,
                         json.dumps(metadata) if metadata else None)
                    )
                    key_id = cursor.lastrowid
                    await conn.commit()
                
                # Log the creation
                await self._log_action(key_id, "created", user_id)
                
                logger.info(f"Created API key {key_id} for user {user_id}")
                
                return {
                    "id": key_id,
                    "key": full_key,  # Only returned on creation!
                    "key_prefix": key_prefix,
                    "name": name,
                    "scope": scope,
                    "expires_at": expires_at.isoformat() if expires_at else None,
                    "created_at": datetime.utcnow().isoformat(),
                    "message": "Store this key securely - it will not be shown again"
                }
                
        except Exception as e:
            logger.error(f"Failed to create API key: {e}")
            raise DatabaseError(f"Failed to create API key: {e}")
    
    async def validate_api_key(
        self,
        api_key: str,
        required_scope: Optional[str] = None,
        ip_address: Optional[str] = None
    ) -> Optional[Dict[str, Any]]:
        """
        Validate an API key and return its information
        
        Args:
            api_key: The API key to validate
            required_scope: Required permission scope
            ip_address: Client IP address for validation and logging
            
        Returns:
            Key information if valid, None if invalid
        """
        if not self._initialized:
            await self.initialize()
        
        key_hash = self.hash_api_key(api_key)
        
        try:
            # Get key information
            result = await self.db_pool.fetchone(
                """
                SELECT id, user_id, name, scope, status, expires_at,
                       rate_limit, allowed_ips, usage_count
                FROM api_keys
                WHERE key_hash = ? AND status = ?
                """,
                key_hash, APIKeyStatus.ACTIVE.value
            )
            
            if not result:
                return None
            
            key_info = dict(result)
            
            # Check expiration
            if key_info['expires_at']:
                expires_at = datetime.fromisoformat(key_info['expires_at']) if isinstance(key_info['expires_at'], str) else key_info['expires_at']
                if expires_at < datetime.utcnow():
                    await self._mark_expired(key_info['id'])
                    return None
            
            # Check IP restrictions
            if key_info['allowed_ips'] and ip_address:
                import json
                allowed_ips = json.loads(key_info['allowed_ips'])
                if ip_address not in allowed_ips:
                    logger.warning(f"API key {key_info['id']} used from unauthorized IP: {ip_address}")
                    return None
            
            # Check scope
            if required_scope:
                key_scope = key_info['scope']
                if not self._has_scope(key_scope, required_scope):
                    return None
            
            # Update usage statistics
            await self._update_usage(key_info['id'], ip_address)
            
            return key_info
            
        except Exception as e:
            logger.error(f"Failed to validate API key: {e}")
            return None
    
    async def rotate_api_key(
        self,
        key_id: int,
        user_id: int,
        expires_in_days: Optional[int] = 90
    ) -> Dict[str, Any]:
        """
        Rotate an API key - create new one and revoke old one
        
        Args:
            key_id: ID of the key to rotate
            user_id: User requesting rotation (for authorization)
            expires_in_days: Expiration for new key
            
        Returns:
            New key information
        """
        if not self._initialized:
            await self.initialize()
        
        try:
            # Get existing key info
            old_key = await self.db_pool.fetchone(
                "SELECT * FROM api_keys WHERE id = ? AND user_id = ?",
                key_id, user_id
            )
            
            if not old_key:
                raise ValueError("API key not found or unauthorized")
            
            old_key = dict(old_key)
            
            # Create new key with same settings
            new_key_result = await self.create_api_key(
                user_id=user_id,
                name=f"{old_key['name']} (rotated)" if old_key['name'] else "Rotated key",
                description=old_key['description'],
                scope=old_key['scope'],
                expires_in_days=expires_in_days,
                rate_limit=old_key['rate_limit'],
                allowed_ips=json.loads(old_key['allowed_ips']) if old_key['allowed_ips'] else None,
                metadata=json.loads(old_key['metadata']) if old_key['metadata'] else None
            )
            
            # Update rotation references
            async with self.db_pool.transaction() as conn:
                # Mark old key as rotated
                if hasattr(conn, 'execute'):
                    await conn.execute(
                        """
                        UPDATE api_keys 
                        SET status = $1, rotated_to = $2, revoked_at = $3,
                            revoke_reason = $4
                        WHERE id = $5
                        """,
                        APIKeyStatus.ROTATED.value, new_key_result['id'],
                        datetime.utcnow(), "Key rotation", key_id
                    )
                    
                    # Update new key with rotation reference
                    await conn.execute(
                        "UPDATE api_keys SET rotated_from = $1 WHERE id = $2",
                        key_id, new_key_result['id']
                    )
                else:
                    await conn.execute(
                        """
                        UPDATE api_keys 
                        SET status = ?, rotated_to = ?, revoked_at = ?,
                            revoke_reason = ?
                        WHERE id = ?
                        """,
                        (APIKeyStatus.ROTATED.value, new_key_result['id'],
                         datetime.utcnow().isoformat(), "Key rotation", key_id)
                    )
                    
                    await conn.execute(
                        "UPDATE api_keys SET rotated_from = ? WHERE id = ?",
                        (key_id, new_key_result['id'])
                    )
                    
                    await conn.commit()
            
            # Log the rotation
            await self._log_action(key_id, "rotated", user_id)
            await self._log_action(new_key_result['id'], "created_from_rotation", user_id)
            
            logger.info(f"Rotated API key {key_id} to {new_key_result['id']}")
            
            return new_key_result
            
        except Exception as e:
            logger.error(f"Failed to rotate API key: {e}")
            raise DatabaseError(f"Failed to rotate API key: {e}")
    
    async def revoke_api_key(
        self,
        key_id: int,
        user_id: int,
        reason: Optional[str] = None
    ) -> bool:
        """
        Revoke an API key
        
        Args:
            key_id: ID of the key to revoke
            user_id: User requesting revocation
            reason: Reason for revocation
            
        Returns:
            True if successful
        """
        if not self._initialized:
            await self.initialize()
        
        try:
            async with self.db_pool.transaction() as conn:
                if hasattr(conn, 'execute'):
                    result = await conn.execute(
                        """
                        UPDATE api_keys 
                        SET status = $1, revoked_at = $2, revoked_by = $3,
                            revoke_reason = $4
                        WHERE id = $5 AND user_id = $6 AND status = $7
                        """,
                        APIKeyStatus.REVOKED.value, datetime.utcnow(), user_id,
                        reason or "Manual revocation", key_id, user_id,
                        APIKeyStatus.ACTIVE.value
                    )
                    success = result != "UPDATE 0"
                else:
                    cursor = await conn.execute(
                        """
                        UPDATE api_keys 
                        SET status = ?, revoked_at = ?, revoked_by = ?,
                            revoke_reason = ?
                        WHERE id = ? AND user_id = ? AND status = ?
                        """,
                        (APIKeyStatus.REVOKED.value, datetime.utcnow().isoformat(),
                         user_id, reason or "Manual revocation", key_id, user_id,
                         APIKeyStatus.ACTIVE.value)
                    )
                    success = cursor.rowcount > 0
                    await conn.commit()
            
            if success:
                await self._log_action(key_id, "revoked", user_id, {"reason": reason})
                logger.info(f"Revoked API key {key_id}")
            
            return success
            
        except Exception as e:
            logger.error(f"Failed to revoke API key: {e}")
            raise DatabaseError(f"Failed to revoke API key: {e}")
    
    async def list_user_keys(
        self,
        user_id: int,
        include_revoked: bool = False
    ) -> List[Dict[str, Any]]:
        """
        List all API keys for a user
        
        Args:
            user_id: User ID
            include_revoked: Include revoked/expired keys
            
        Returns:
            List of key information (without actual keys)
        """
        if not self._initialized:
            await self.initialize()
        
        try:
            if include_revoked:
                query = "SELECT * FROM api_keys WHERE user_id = ? ORDER BY created_at DESC"
            else:
                query = """
                    SELECT * FROM api_keys 
                    WHERE user_id = ? AND status = ?
                    ORDER BY created_at DESC
                """
                
            if include_revoked:
                results = await self.db_pool.fetchall(query, user_id)
            else:
                results = await self.db_pool.fetchall(query, user_id, APIKeyStatus.ACTIVE.value)
            
            keys = []
            for row in results:
                key_dict = dict(row)
                # Never return the actual hash
                key_dict.pop('key_hash', None)
                keys.append(key_dict)
            
            return keys
            
        except Exception as e:
            logger.error(f"Failed to list user keys: {e}")
            raise DatabaseError(f"Failed to list user keys: {e}")
    
    async def cleanup_expired_keys(self):
        """Mark expired keys as expired"""
        if not self._initialized:
            await self.initialize()
        
        try:
            async with self.db_pool.transaction() as conn:
                if hasattr(conn, 'execute'):
                    result = await conn.execute(
                        """
                        UPDATE api_keys 
                        SET status = $1
                        WHERE status = $2 AND expires_at < $3
                        """,
                        APIKeyStatus.EXPIRED.value,
                        APIKeyStatus.ACTIVE.value,
                        datetime.utcnow()
                    )
                else:
                    await conn.execute(
                        """
                        UPDATE api_keys 
                        SET status = ?
                        WHERE status = ? AND expires_at < ?
                        """,
                        (APIKeyStatus.EXPIRED.value,
                         APIKeyStatus.ACTIVE.value,
                         datetime.utcnow().isoformat())
                    )
                    await conn.commit()
            
            logger.debug("Cleaned up expired API keys")
            
        except Exception as e:
            logger.error(f"Failed to cleanup expired keys: {e}")
    
    def _has_scope(self, key_scope: str, required_scope: str) -> bool:
        """Check if key scope satisfies required scope"""
        scope_hierarchy = {
            "read": 0,
            "write": 1,
            "admin": 2,
            "service": 3
        }
        
        key_level = scope_hierarchy.get(key_scope, 0)
        required_level = scope_hierarchy.get(required_scope, 0)
        
        return key_level >= required_level
    
    async def _update_usage(self, key_id: int, ip_address: Optional[str] = None):
        """Update usage statistics for a key"""
        try:
            async with self.db_pool.transaction() as conn:
                if hasattr(conn, 'execute'):
                    await conn.execute(
                        """
                        UPDATE api_keys 
                        SET usage_count = usage_count + 1,
                            last_used_at = $1,
                            last_used_ip = $2
                        WHERE id = $3
                        """,
                        datetime.utcnow(), ip_address, key_id
                    )
                else:
                    await conn.execute(
                        """
                        UPDATE api_keys 
                        SET usage_count = usage_count + 1,
                            last_used_at = ?,
                            last_used_ip = ?
                        WHERE id = ?
                        """,
                        (datetime.utcnow().isoformat(), ip_address, key_id)
                    )
                    await conn.commit()
        except Exception as e:
            logger.error(f"Failed to update usage: {e}")
    
    async def _mark_expired(self, key_id: int):
        """Mark a key as expired"""
        try:
            async with self.db_pool.transaction() as conn:
                if hasattr(conn, 'execute'):
                    await conn.execute(
                        "UPDATE api_keys SET status = $1 WHERE id = $2",
                        APIKeyStatus.EXPIRED.value, key_id
                    )
                else:
                    await conn.execute(
                        "UPDATE api_keys SET status = ? WHERE id = ?",
                        (APIKeyStatus.EXPIRED.value, key_id)
                    )
                    await conn.commit()
        except Exception as e:
            logger.error(f"Failed to mark key as expired: {e}")
    
    async def _log_action(
        self,
        key_id: int,
        action: str,
        user_id: Optional[int] = None,
        details: Optional[Dict[str, Any]] = None
    ):
        """Log an action in the audit log"""
        try:
            import json
            async with self.db_pool.transaction() as conn:
                if hasattr(conn, 'execute'):
                    await conn.execute(
                        """
                        INSERT INTO api_key_audit_log (api_key_id, action, user_id, details)
                        VALUES ($1, $2, $3, $4)
                        """,
                        key_id, action, user_id,
                        json.dumps(details) if details else None
                    )
                else:
                    await conn.execute(
                        """
                        INSERT INTO api_key_audit_log (api_key_id, action, user_id, details)
                        VALUES (?, ?, ?, ?)
                        """,
                        (key_id, action, user_id,
                         json.dumps(details) if details else None)
                    )
                    await conn.commit()
        except Exception as e:
            logger.error(f"Failed to log action: {e}")


#######################################################################################################################
#
# Module Functions
#

# Global instance
_api_key_manager: Optional[APIKeyManager] = None

async def get_api_key_manager() -> APIKeyManager:
    """Get APIKeyManager singleton instance"""
    global _api_key_manager
    if not _api_key_manager:
        _api_key_manager = APIKeyManager()
        await _api_key_manager.initialize()
    return _api_key_manager

#
# End of api_key_manager.py
#######################################################################################################################