# registration_service.py
# Description: User registration service with transaction safety and directory management
#
# Imports
import os
import json
import shutil
import secrets
import string
import asyncio
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, Optional, List, Any
from concurrent.futures import ThreadPoolExecutor
from uuid import uuid4
#
# 3rd-party imports
from loguru import logger
#
# Local imports
from tldw_Server_API.app.core.AuthNZ.settings import Settings, get_settings
from tldw_Server_API.app.core.AuthNZ.database import DatabasePool, get_db_pool
from tldw_Server_API.app.core.AuthNZ.password_service import PasswordService, get_password_service
from tldw_Server_API.app.core.AuthNZ.exceptions import (
    RegistrationError,
    InvalidRegistrationCodeError,
    RegistrationCodeExpiredError,
    RegistrationCodeExhaustedError,
    DuplicateUserError,
    RegistrationDisabledError,
    WeakPasswordError,
    DirectoryCreationError,
    TransactionError,
    DatabaseError
)

#######################################################################################################################
#
# Registration Service Class

class RegistrationService:
    """Service for user registration with full transaction safety"""
    
    def __init__(
        self,
        db_pool: Optional[DatabasePool] = None,
        password_service: Optional[PasswordService] = None,
        settings: Optional[Settings] = None
    ):
        """Initialize registration service"""
        self.settings = settings or get_settings()
        self.db_pool = db_pool
        self.password_service = password_service or get_password_service()
        
        # Thread pool for directory operations
        self.executor = ThreadPoolExecutor(max_workers=2)
        
        # Check if registration is enabled
        self.registration_enabled = self.settings.ENABLE_REGISTRATION
        self.require_code = self.settings.REQUIRE_REGISTRATION_CODE
        
        logger.info(
            f"RegistrationService initialized (enabled={self.registration_enabled}, "
            f"require_code={self.require_code})"
        )
    
    async def initialize(self):
        """Initialize the registration service"""
        if not self.db_pool:
            self.db_pool = await get_db_pool()
    
    def generate_registration_code(self, length: int = 24) -> str:
        """
        Generate a secure registration code
        
        Args:
            length: Length of the code
            
        Returns:
            Secure random alphanumeric code
        """
        alphabet = string.ascii_letters + string.digits
        return ''.join(secrets.choice(alphabet) for _ in range(length))
    
    def _create_user_directories(self, user_id: int) -> bool:
        """
        Create user directories (runs in thread pool)
        
        Args:
            user_id: User's database ID
            
        Returns:
            True if successful, False otherwise
        """
        try:
            base_path = Path(self.settings.USER_DATA_BASE_PATH)
            user_dir = base_path / str(user_id)
            
            # Create main user directory
            user_dir.mkdir(parents=True, exist_ok=True)
            
            # Create subdirectories
            subdirs = ["media", "notes", "embeddings", "exports", "temp"]
            for subdir in subdirs:
                (user_dir / subdir).mkdir(exist_ok=True)
            
            # Create user-specific ChromaDB directory if configured
            if self.settings.CHROMADB_BASE_PATH:
                chroma_path = Path(self.settings.CHROMADB_BASE_PATH) / str(user_id)
                chroma_path.mkdir(parents=True, exist_ok=True)
            
            # Set permissions on Unix-like systems
            if os.name != 'nt':
                os.chmod(user_dir, 0o750)
                for subdir in subdirs:
                    os.chmod(user_dir / subdir, 0o750)
            
            logger.debug(f"Created directories for user {user_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to create directories for user {user_id}: {e}")
            return False
    
    def _cleanup_user_directories(self, user_id: int):
        """
        Clean up user directories (for rollback)
        
        Args:
            user_id: User's database ID
        """
        try:
            # Remove user data directory
            user_dir = Path(self.settings.USER_DATA_BASE_PATH) / str(user_id)
            if user_dir.exists():
                shutil.rmtree(user_dir, ignore_errors=True)
            
            # Remove ChromaDB directory
            if self.settings.CHROMADB_BASE_PATH:
                chroma_dir = Path(self.settings.CHROMADB_BASE_PATH) / str(user_id)
                if chroma_dir.exists():
                    shutil.rmtree(chroma_dir, ignore_errors=True)
            
            logger.debug(f"Cleaned up directories for user {user_id}")
            
        except Exception as e:
            logger.error(f"Error cleaning up directories for user {user_id}: {e}")
    
    async def register_user(
        self,
        username: str,
        email: str,
        password: str,
        registration_code: Optional[str] = None,
        created_by: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Register a new user with full transaction safety
        
        Args:
            username: Username (must be unique)
            email: Email address (must be unique)
            password: Plain text password
            registration_code: Optional registration code
            created_by: ID of user creating this account (for admin creation)
            
        Returns:
            Dictionary with user information
            
        Raises:
            Various registration exceptions
        """
        # Check if registration is enabled
        if not self.registration_enabled and not created_by:
            raise RegistrationDisabledError()
        
        # Initialize if needed
        if not self.db_pool:
            await self.initialize()
        
        # Validate password strength
        self.password_service.validate_password_strength(password, username)
        
        user_id = None
        directories_created = False
        
        try:
            async with self.db_pool.transaction() as conn:
                # Check for duplicate username/email
                if hasattr(conn, 'fetchrow'):
                    # PostgreSQL
                    existing = await conn.fetchrow(
                        """
                        SELECT username, email
                        FROM users
                        WHERE username = $1 OR email = $2
                        """,
                        username.lower(), email.lower()
                    )
                else:
                    # SQLite
                    cursor = await conn.execute(
                        """
                        SELECT username, email
                        FROM users
                        WHERE lower(username) = ? OR lower(email) = ?
                        """,
                        (username.lower(), email.lower())
                    )
                    existing = await cursor.fetchone()
                
                if existing:
                    if existing[0] and existing[0].lower() == username.lower():
                        raise DuplicateUserError("username")
                    else:
                        raise DuplicateUserError("email")
                
                # Validate registration code if required
                role = self.settings.DEFAULT_USER_ROLE
                storage_quota = self.settings.DEFAULT_STORAGE_QUOTA_MB
                
                if self.require_code and not created_by:
                    if not registration_code:
                        raise InvalidRegistrationCodeError("Registration code required")
                    
                    # Validate and use registration code
                    code_info = await self._validate_and_use_registration_code(
                        registration_code, conn
                    )
                    role = code_info.get('role_to_grant', role)
                    
                    # Check if code specifies storage quota
                    if 'storage_quota_mb' in code_info:
                        storage_quota = code_info['storage_quota_mb']
                
                # Hash the password
                password_hash = self.password_service.hash_password(password)
                
                # Generate UUID
                user_uuid = str(uuid4())
                
                # Create user
                if hasattr(conn, 'fetchval'):
                    # PostgreSQL
                    user_id = await conn.fetchval(
                        """
                        INSERT INTO users (
                            uuid, username, email, password_hash, role,
                            is_active, is_verified, created_by, storage_quota_mb
                        )
                        VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9)
                        RETURNING id
                        """,
                        user_uuid, username, email, password_hash, role,
                        True, not self.require_code, created_by, storage_quota
                    )
                else:
                    # SQLite
                    cursor = await conn.execute(
                        """
                        INSERT INTO users (
                            uuid, username, email, password_hash, role,
                            is_active, is_verified, created_by, storage_quota_mb
                        )
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                        """,
                        (user_uuid, username, email, password_hash, role,
                         1, 0 if self.require_code else 1, created_by, storage_quota)
                    )
                    user_id = cursor.lastrowid
                
                # Add password to history
                await self._add_password_to_history(user_id, password_hash, conn)
                
                # Create user directories (before committing transaction)
                loop = asyncio.get_event_loop()
                directories_created = await loop.run_in_executor(
                    self.executor,
                    self._create_user_directories,
                    user_id
                )
                
                if not directories_created:
                    raise DirectoryCreationError(
                        f"user_databases/{user_id}",
                        "Failed to create user directories"
                    )
                
                # Log registration in audit log
                await self._log_registration(
                    user_id, username, email, role, 
                    created_by, registration_code is not None, conn
                )
                
                # If we get here, everything succeeded
                logger.info(
                    f"Successfully registered user: {username} (ID: {user_id}, Role: {role})"
                )
                
                return {
                    "user_id": user_id,
                    "uuid": user_uuid,
                    "username": username,
                    "email": email,
                    "role": role,
                    "is_verified": not self.require_code,
                    "storage_quota_mb": storage_quota
                }
                
        except Exception as e:
            # Rollback: Clean up directories if they were created
            if user_id and directories_created:
                self._cleanup_user_directories(user_id)
            
            # Log the error
            logger.error(f"Registration failed for {username}: {e}")
            
            # Re-raise the exception
            raise
    
    async def _validate_and_use_registration_code(
        self,
        code: str,
        conn
    ) -> Dict[str, Any]:
        """
        Validate and consume a registration code
        
        Args:
            code: Registration code
            conn: Database connection (in transaction)
            
        Returns:
            Code information
            
        Raises:
            InvalidRegistrationCodeError: If code is invalid
        """
        if hasattr(conn, 'fetchrow'):
            # PostgreSQL - Use SELECT FOR UPDATE to lock the row
            code_row = await conn.fetchrow(
                """
                SELECT id, role_to_grant, times_used, max_uses,
                       expires_at, is_active, description
                FROM registration_codes
                WHERE code = $1
                FOR UPDATE
                """,
                code
            )
            
            if not code_row:
                raise InvalidRegistrationCodeError("Invalid registration code")
            
            # Check if active
            if not code_row['is_active']:
                raise InvalidRegistrationCodeError("Registration code is inactive")
            
            # Check expiration
            if code_row['expires_at'] < datetime.utcnow():
                raise RegistrationCodeExpiredError()
            
            # Check usage limit
            if code_row['times_used'] >= code_row['max_uses']:
                raise RegistrationCodeExhaustedError()
            
            # Update usage count
            await conn.execute(
                """
                UPDATE registration_codes
                SET times_used = times_used + 1
                WHERE id = $1
                """,
                code_row['id']
            )
            
            return dict(code_row)
            
        else:
            # SQLite - Manual locking with transaction
            cursor = await conn.execute(
                """
                SELECT id, role_to_grant, times_used, max_uses,
                       expires_at, is_active, description
                FROM registration_codes
                WHERE code = ?
                """,
                (code,)
            )
            code_row = await cursor.fetchone()
            
            if not code_row:
                raise InvalidRegistrationCodeError("Invalid registration code")
            
            # Convert to dict for easier access
            code_info = {
                "id": code_row[0],
                "role_to_grant": code_row[1],
                "times_used": code_row[2],
                "max_uses": code_row[3],
                "expires_at": datetime.fromisoformat(code_row[4]),
                "is_active": code_row[5],
                "description": code_row[6]
            }
            
            # Validate
            if not code_info['is_active']:
                raise InvalidRegistrationCodeError("Registration code is inactive")
            
            if code_info['expires_at'] < datetime.utcnow():
                raise RegistrationCodeExpiredError()
            
            if code_info['times_used'] >= code_info['max_uses']:
                raise RegistrationCodeExhaustedError()
            
            # Update usage
            await conn.execute(
                """
                UPDATE registration_codes
                SET times_used = times_used + 1
                WHERE id = ?
                """,
                (code_info['id'],)
            )
            
            return code_info
    
    async def _add_password_to_history(
        self,
        user_id: int,
        password_hash: str,
        conn
    ):
        """Add password to user's password history"""
        try:
            if hasattr(conn, 'execute'):
                # PostgreSQL
                await conn.execute(
                    """
                    INSERT INTO password_history (user_id, password_hash)
                    VALUES ($1, $2)
                    """,
                    user_id, password_hash
                )
            else:
                # SQLite
                await conn.execute(
                    """
                    INSERT INTO password_history (user_id, password_hash)
                    VALUES (?, ?)
                    """,
                    (user_id, password_hash)
                )
        except Exception as e:
            logger.error(f"Failed to add password to history: {e}")
            # Don't fail registration for this
    
    async def _log_registration(
        self,
        user_id: int,
        username: str,
        email: str,
        role: str,
        created_by: Optional[int],
        used_code: bool,
        conn
    ):
        """Log registration in audit log"""
        try:
            details = {
                "username": username,
                "email": email,
                "role": role,
                "used_registration_code": used_code,
                "created_by": created_by
            }
            
            if hasattr(conn, 'execute'):
                # PostgreSQL
                await conn.execute(
                    """
                    INSERT INTO audit_log (
                        user_id, action, target_type, target_id,
                        success, details
                    )
                    VALUES ($1, $2, $3, $4, $5, $6)
                    """,
                    user_id, 'user_registered', 'user', user_id,
                    True, json.dumps(details)
                )
            else:
                # SQLite
                await conn.execute(
                    """
                    INSERT INTO audit_log (
                        user_id, action, target_type, target_id,
                        success, details
                    )
                    VALUES (?, ?, ?, ?, ?, ?)
                    """,
                    (user_id, 'user_registered', 'user', user_id,
                     1, json.dumps(details))
                )
        except Exception as e:
            logger.error(f"Failed to log registration: {e}")
            # Don't fail registration for this
    
    async def create_registration_code(
        self,
        created_by: int,
        max_uses: int = 1,
        expires_in_days: Optional[int] = None,
        role_to_grant: str = "user",
        description: Optional[str] = None,
        allowed_email_domain: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Create a new registration code
        
        Args:
            created_by: ID of user creating the code
            max_uses: Maximum number of uses
            expires_in_days: Days until expiration
            role_to_grant: Role to grant to users
            description: Optional description
            allowed_email_domain: Restrict to email domain
            
        Returns:
            Registration code information
        """
        if not self.db_pool:
            await self.initialize()
        
        code = self.generate_registration_code()
        expires_at = datetime.utcnow() + timedelta(
            days=expires_in_days or self.settings.REGISTRATION_CODE_DEFAULT_EXPIRY_DAYS
        )
        
        try:
            async with self.db_pool.transaction() as conn:
                if hasattr(conn, 'fetchval'):
                    # PostgreSQL
                    code_id = await conn.fetchval(
                        """
                        INSERT INTO registration_codes (
                            code, max_uses, expires_at, created_by,
                            role_to_grant, description, allowed_email_domain
                        )
                        VALUES ($1, $2, $3, $4, $5, $6, $7)
                        RETURNING id
                        """,
                        code, max_uses, expires_at, created_by,
                        role_to_grant, description, allowed_email_domain
                    )
                else:
                    # SQLite
                    cursor = await conn.execute(
                        """
                        INSERT INTO registration_codes (
                            code, max_uses, expires_at, created_by,
                            role_to_grant, description, allowed_email_domain
                        )
                        VALUES (?, ?, ?, ?, ?, ?, ?)
                        """,
                        (code, max_uses, expires_at.isoformat(), created_by,
                         role_to_grant, description, allowed_email_domain)
                    )
                    code_id = cursor.lastrowid
                    await conn.commit()
                
                logger.info(
                    f"Created registration code {code[:8]}... "
                    f"(max_uses={max_uses}, role={role_to_grant})"
                )
                
                return {
                    "id": code_id,
                    "code": code,
                    "max_uses": max_uses,
                    "expires_at": expires_at.isoformat(),
                    "role_to_grant": role_to_grant,
                    "description": description
                }
                
        except Exception as e:
            logger.error(f"Failed to create registration code: {e}")
            raise RegistrationError(f"Failed to create registration code: {e}")
    
    async def list_registration_codes(
        self,
        active_only: bool = True,
        created_by: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """
        List registration codes
        
        Args:
            active_only: Only show active codes
            created_by: Filter by creator
            
        Returns:
            List of registration codes
        """
        if not self.db_pool:
            await self.initialize()
        
        query_parts = ["SELECT * FROM registration_codes WHERE 1=1"]
        params = []
        
        if active_only:
            query_parts.append("AND is_active = ?")
            params.append(1 if self.db_pool.pool else True)
            query_parts.append("AND expires_at > ?")
            params.append(datetime.utcnow())
        
        if created_by:
            query_parts.append("AND created_by = ?")
            params.append(created_by)
        
        query_parts.append("ORDER BY created_at DESC")
        query = " ".join(query_parts)
        
        codes = await self.db_pool.fetchall(query, *params)
        return codes
    
    async def revoke_registration_code(self, code_id: int):
        """Revoke a registration code"""
        if not self.db_pool:
            await self.initialize()
        
        await self.db_pool.execute(
            "UPDATE registration_codes SET is_active = ? WHERE id = ?",
            False, code_id
        )
        logger.info(f"Revoked registration code {code_id}")
    
    async def shutdown(self):
        """Shutdown the registration service"""
        self.executor.shutdown(wait=False)


#######################################################################################################################
#
# Module Functions

# Global instance
_registration_service: Optional[RegistrationService] = None


async def get_registration_service() -> RegistrationService:
    """Get registration service singleton"""
    global _registration_service
    if not _registration_service:
        _registration_service = RegistrationService()
        await _registration_service.initialize()
    return _registration_service


async def reset_registration_service():
    """Reset registration service singleton (for testing)"""
    global _registration_service
    if _registration_service:
        await _registration_service.shutdown()
        _registration_service = None


#
# End of registration_service.py
#######################################################################################################################