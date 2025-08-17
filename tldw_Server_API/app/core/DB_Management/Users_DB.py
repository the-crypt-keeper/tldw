# Users_DB.py
# Description: Database operations for user management in multi-user mode
#
# Imports
from typing import Optional, Dict, Any, List
from datetime import datetime, timedelta
import hashlib
#
# 3rd-party imports
from loguru import logger
#
# Local imports
from tldw_Server_API.app.core.AuthNZ.database import DatabasePool, get_db_pool
from tldw_Server_API.app.core.AuthNZ.exceptions import DatabaseError
from tldw_Server_API.app.core.AuthNZ.settings import get_settings

#######################################################################################################################
#
# Exceptions
#

class UserNotFoundError(Exception):
    """Raised when a user is not found in the database"""
    pass

class DuplicateUserError(Exception):
    """Raised when attempting to create a user that already exists"""
    pass

#######################################################################################################################
#
# Users Database Class
#

class UsersDB:
    """Handles all database operations for user management"""
    
    def __init__(self, db_pool: Optional[DatabasePool] = None):
        """Initialize Users database handler"""
        self.db_pool = db_pool
        self._initialized = False
        self.settings = get_settings()
        
    async def initialize(self):
        """Initialize database connection and ensure tables exist"""
        if self._initialized:
            return
            
        # Get database pool
        if not self.db_pool:
            self.db_pool = await get_db_pool()
        
        # Create users table if it doesn't exist
        await self._create_tables()
        
        self._initialized = True
        logger.info("UsersDB initialized")
    
    async def _create_tables(self):
        """Create users and related tables if they don't exist"""
        try:
            async with self.db_pool.transaction() as conn:
                if hasattr(conn, 'execute'):
                    # PostgreSQL
                    await conn.execute("""
                        CREATE TABLE IF NOT EXISTS users (
                            id SERIAL PRIMARY KEY,
                            username VARCHAR(50) UNIQUE NOT NULL,
                            email VARCHAR(255) UNIQUE NOT NULL,
                            password_hash TEXT NOT NULL,
                            is_active BOOLEAN DEFAULT TRUE,
                            is_superuser BOOLEAN DEFAULT FALSE,
                            role VARCHAR(50) DEFAULT 'user',
                            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                            last_login TIMESTAMP,
                            email_verified BOOLEAN DEFAULT FALSE,
                            storage_quota_mb INTEGER DEFAULT 5120,
                            storage_used_mb INTEGER DEFAULT 0
                        )
                    """)
                    
                    # Create indexes
                    await conn.execute("CREATE INDEX IF NOT EXISTS idx_users_username ON users(username)")
                    await conn.execute("CREATE INDEX IF NOT EXISTS idx_users_email ON users(email)")
                    await conn.execute("CREATE INDEX IF NOT EXISTS idx_users_role ON users(role)")
                    
                else:
                    # SQLite
                    await conn.execute("""
                        CREATE TABLE IF NOT EXISTS users (
                            id INTEGER PRIMARY KEY AUTOINCREMENT,
                            username TEXT UNIQUE NOT NULL,
                            email TEXT UNIQUE NOT NULL,
                            password_hash TEXT NOT NULL,
                            is_active INTEGER DEFAULT 1,
                            is_superuser INTEGER DEFAULT 0,
                            role TEXT DEFAULT 'user',
                            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                            last_login TIMESTAMP,
                            email_verified INTEGER DEFAULT 0,
                            storage_quota_mb INTEGER DEFAULT 5120,
                            storage_used_mb INTEGER DEFAULT 0
                        )
                    """)
                    
                    # Create indexes
                    await conn.execute("CREATE INDEX IF NOT EXISTS idx_users_username ON users(username)")
                    await conn.execute("CREATE INDEX IF NOT EXISTS idx_users_email ON users(email)")
                    await conn.execute("CREATE INDEX IF NOT EXISTS idx_users_role ON users(role)")
                    
                    await conn.commit()
                    
                logger.debug("Users table and indexes created/verified")
                
        except Exception as e:
            logger.error(f"Failed to create users table: {e}")
            raise DatabaseError(f"Failed to create users table: {e}")
    
    async def get_user_by_id(self, user_id: int) -> Optional[Dict[str, Any]]:
        """
        Get user by ID
        
        Args:
            user_id: User's database ID
            
        Returns:
            User data dictionary or None if not found
            
        Raises:
            UserNotFoundError: If user doesn't exist
        """
        if not self._initialized:
            await self.initialize()
        
        try:
            result = await self.db_pool.fetchone(
                "SELECT * FROM users WHERE id = ?",
                user_id
            )
            
            if not result:
                raise UserNotFoundError(f"User with ID {user_id} not found")
            
            # Convert to dictionary
            user_dict = dict(result)
            
            # Convert boolean fields for SQLite
            if not hasattr(self.db_pool, 'fetchval'):  # SQLite
                user_dict['is_active'] = bool(user_dict.get('is_active', 1))
                user_dict['is_superuser'] = bool(user_dict.get('is_superuser', 0))
                user_dict['email_verified'] = bool(user_dict.get('email_verified', 0))
            
            return user_dict
            
        except UserNotFoundError:
            raise
        except Exception as e:
            logger.error(f"Failed to get user by ID {user_id}: {e}")
            raise DatabaseError(f"Failed to get user: {e}")
    
    async def get_user_by_username(self, username: str) -> Optional[Dict[str, Any]]:
        """
        Get user by username
        
        Args:
            username: Username to search for
            
        Returns:
            User data dictionary or None if not found
        """
        if not self._initialized:
            await self.initialize()
        
        try:
            result = await self.db_pool.fetchone(
                "SELECT * FROM users WHERE username = ?",
                username
            )
            
            if not result:
                return None
            
            # Convert to dictionary
            user_dict = dict(result)
            
            # Convert boolean fields for SQLite
            if not hasattr(self.db_pool, 'fetchval'):  # SQLite
                user_dict['is_active'] = bool(user_dict.get('is_active', 1))
                user_dict['is_superuser'] = bool(user_dict.get('is_superuser', 0))
                user_dict['email_verified'] = bool(user_dict.get('email_verified', 0))
            
            return user_dict
            
        except Exception as e:
            logger.error(f"Failed to get user by username {username}: {e}")
            raise DatabaseError(f"Failed to get user: {e}")
    
    async def get_user_by_email(self, email: str) -> Optional[Dict[str, Any]]:
        """
        Get user by email
        
        Args:
            email: Email to search for
            
        Returns:
            User data dictionary or None if not found
        """
        if not self._initialized:
            await self.initialize()
        
        try:
            result = await self.db_pool.fetchone(
                "SELECT * FROM users WHERE email = ?",
                email.lower()
            )
            
            if not result:
                return None
            
            # Convert to dictionary
            user_dict = dict(result)
            
            # Convert boolean fields for SQLite
            if not hasattr(self.db_pool, 'fetchval'):  # SQLite
                user_dict['is_active'] = bool(user_dict.get('is_active', 1))
                user_dict['is_superuser'] = bool(user_dict.get('is_superuser', 0))
                user_dict['email_verified'] = bool(user_dict.get('email_verified', 0))
            
            return user_dict
            
        except Exception as e:
            logger.error(f"Failed to get user by email {email}: {e}")
            raise DatabaseError(f"Failed to get user: {e}")
    
    async def create_user(
        self,
        username: str,
        email: str,
        password_hash: str,
        role: str = "user",
        is_active: bool = True,
        is_superuser: bool = False,
        storage_quota_mb: int = 5120
    ) -> Dict[str, Any]:
        """
        Create a new user
        
        Args:
            username: Unique username
            email: User's email address
            password_hash: Hashed password
            role: User role (default: "user")
            is_active: Whether user is active
            is_superuser: Whether user is a superuser
            storage_quota_mb: Storage quota in MB
            
        Returns:
            Created user data
            
        Raises:
            DuplicateUserError: If username or email already exists
        """
        if not self._initialized:
            await self.initialize()
        
        # Check for existing user
        existing = await self.get_user_by_username(username)
        if existing:
            raise DuplicateUserError(f"Username '{username}' already exists")
        
        existing = await self.get_user_by_email(email)
        if existing:
            raise DuplicateUserError(f"Email '{email}' already exists")
        
        try:
            async with self.db_pool.transaction() as conn:
                if hasattr(conn, 'fetchval'):
                    # PostgreSQL
                    user_id = await conn.fetchval(
                        """
                        INSERT INTO users (
                            username, email, password_hash, role,
                            is_active, is_superuser, storage_quota_mb
                        )
                        VALUES ($1, $2, $3, $4, $5, $6, $7)
                        RETURNING id
                        """,
                        username, email.lower(), password_hash, role,
                        is_active, is_superuser, storage_quota_mb
                    )
                else:
                    # SQLite
                    cursor = await conn.execute(
                        """
                        INSERT INTO users (
                            username, email, password_hash, role,
                            is_active, is_superuser, storage_quota_mb
                        )
                        VALUES (?, ?, ?, ?, ?, ?, ?)
                        """,
                        (username, email.lower(), password_hash, role,
                         int(is_active), int(is_superuser), storage_quota_mb)
                    )
                    user_id = cursor.lastrowid
                    await conn.commit()
                
                logger.info(f"Created user: {username} (ID: {user_id})")
                
                # Return the created user
                return await self.get_user_by_id(user_id)
                
        except Exception as e:
            logger.error(f"Failed to create user {username}: {e}")
            raise DatabaseError(f"Failed to create user: {e}")
    
    async def update_user(
        self,
        user_id: int,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Update user information
        
        Args:
            user_id: User ID to update
            **kwargs: Fields to update
            
        Returns:
            Updated user data
        """
        if not self._initialized:
            await self.initialize()
        
        # Ensure user exists
        user = await self.get_user_by_id(user_id)
        if not user:
            raise UserNotFoundError(f"User with ID {user_id} not found")
        
        # Filter allowed fields
        allowed_fields = {
            'email', 'password_hash', 'is_active', 'is_superuser',
            'role', 'last_login', 'email_verified', 'storage_quota_mb',
            'storage_used_mb'
        }
        
        updates = {k: v for k, v in kwargs.items() if k in allowed_fields}
        
        if not updates:
            return user  # Nothing to update
        
        try:
            # Build update query
            set_clause = ", ".join([f"{k} = ?" for k in updates.keys()])
            values = list(updates.values())
            values.append(user_id)
            
            async with self.db_pool.transaction() as conn:
                if hasattr(conn, 'execute'):
                    # PostgreSQL - adjust placeholders
                    set_clause = ", ".join([f"{k} = ${i+1}" for i, k in enumerate(updates.keys())])
                    query = f"""
                        UPDATE users 
                        SET {set_clause}, updated_at = CURRENT_TIMESTAMP
                        WHERE id = ${len(values)}
                    """
                else:
                    # SQLite
                    # Convert booleans to integers for SQLite
                    for key in ['is_active', 'is_superuser', 'email_verified']:
                        if key in updates:
                            updates[key] = int(updates[key])
                    
                    values = list(updates.values())
                    values.append(user_id)
                    
                    query = f"""
                        UPDATE users 
                        SET {set_clause}, updated_at = CURRENT_TIMESTAMP
                        WHERE id = ?
                    """
                    
                await conn.execute(query, *values if hasattr(conn, 'execute') else values)
                
                if not hasattr(conn, 'execute'):
                    await conn.commit()
                
                logger.info(f"Updated user {user_id}: {list(updates.keys())}")
                
                # Return updated user
                return await self.get_user_by_id(user_id)
                
        except Exception as e:
            logger.error(f"Failed to update user {user_id}: {e}")
            raise DatabaseError(f"Failed to update user: {e}")
    
    async def delete_user(self, user_id: int) -> bool:
        """
        Delete a user (soft delete by marking inactive)
        
        Args:
            user_id: User ID to delete
            
        Returns:
            True if successful
        """
        if not self._initialized:
            await self.initialize()
        
        try:
            # Soft delete - just mark as inactive
            await self.update_user(user_id, is_active=False)
            logger.info(f"Soft deleted user {user_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to delete user {user_id}: {e}")
            raise DatabaseError(f"Failed to delete user: {e}")
    
    async def update_last_login(self, user_id: int):
        """Update user's last login timestamp"""
        await self.update_user(user_id, last_login=datetime.utcnow())
    
    async def list_users(
        self,
        offset: int = 0,
        limit: int = 100,
        role: Optional[str] = None,
        is_active: Optional[bool] = None
    ) -> List[Dict[str, Any]]:
        """
        List users with optional filtering
        
        Args:
            offset: Pagination offset
            limit: Maximum results
            role: Filter by role
            is_active: Filter by active status
            
        Returns:
            List of user dictionaries
        """
        if not self._initialized:
            await self.initialize()
        
        try:
            # Build query with filters
            query = "SELECT * FROM users WHERE 1=1"
            params = []
            
            if role is not None:
                query += " AND role = ?"
                params.append(role)
            
            if is_active is not None:
                query += " AND is_active = ?"
                params.append(int(is_active) if not hasattr(self.db_pool, 'fetchval') else is_active)
            
            query += " ORDER BY created_at DESC LIMIT ? OFFSET ?"
            params.extend([limit, offset])
            
            results = await self.db_pool.fetchall(query, *params)
            
            users = []
            for row in results:
                user_dict = dict(row)
                
                # Convert boolean fields for SQLite
                if not hasattr(self.db_pool, 'fetchval'):  # SQLite
                    user_dict['is_active'] = bool(user_dict.get('is_active', 1))
                    user_dict['is_superuser'] = bool(user_dict.get('is_superuser', 0))
                    user_dict['email_verified'] = bool(user_dict.get('email_verified', 0))
                
                users.append(user_dict)
            
            return users
            
        except Exception as e:
            logger.error(f"Failed to list users: {e}")
            raise DatabaseError(f"Failed to list users: {e}")


#######################################################################################################################
#
# Module Functions
#

# Global instance
_users_db: Optional[UsersDB] = None

async def get_users_db() -> UsersDB:
    """Get UsersDB singleton instance"""
    global _users_db
    if not _users_db:
        _users_db = UsersDB()
        await _users_db.initialize()
    return _users_db

# Convenience functions for backward compatibility
async def get_user_by_id(user_id: int) -> Optional[Dict[str, Any]]:
    """Get user by ID (convenience function)"""
    db = await get_users_db()
    return await db.get_user_by_id(user_id)

async def create_user(username: str, email: str, password_hash: str, **kwargs) -> Dict[str, Any]:
    """Create user (convenience function)"""
    db = await get_users_db()
    return await db.create_user(username, email, password_hash, **kwargs)

async def get_user_by_username(username: str) -> Optional[Dict[str, Any]]:
    """Get user by username (convenience function)"""
    db = await get_users_db()
    return await db.get_user_by_username(username)


#######################################################################################################################
#
# Per-User Database Path Management
# Each user gets their own SQLite database for their media/content
#

def get_user_db_path(user_id: int, db_name: str = "media") -> str:
    """
    Construct the file path for a user's personal database.
    
    Args:
        user_id: The user's ID
        db_name: Name of the database (default: "media")
        
    Returns:
        Path to the user's database file
        
    Example:
        >>> get_user_db_path(123, "media")
        './user_databases/user_123/media.db'
    """
    settings = get_settings()
    base_path = settings.USER_DATA_BASE_PATH
    return f"{base_path}/user_{user_id}/{db_name}.db"


def get_user_chromadb_path(user_id: int) -> str:
    """
    Construct the path for a user's ChromaDB data.
    
    Args:
        user_id: The user's ID
        
    Returns:
        Path to the user's ChromaDB directory
    """
    settings = get_settings()
    base_path = settings.CHROMADB_BASE_PATH
    return f"{base_path}/user_{user_id}"


async def get_user_media_db(user_id: int, db_name: str = "media"):
    """
    Get a MediaDatabase instance for a specific user.
    
    Args:
        user_id: The user's ID
        db_name: Name of the database
        
    Returns:
        MediaDatabase instance for the user
        
    Note:
        This creates the user directory structure if it doesn't exist.
    """
    import os
    from pathlib import Path
    
    # Get the database path
    db_path = get_user_db_path(user_id, db_name)
    
    # Ensure the directory exists
    db_dir = Path(db_path).parent
    db_dir.mkdir(parents=True, exist_ok=True)
    
    # Import MediaDatabase (avoid circular import)
    try:
        from tldw_Server_API.app.core.DB_Management.DB_Manager import MediaDatabase
        
        # Create and return the database instance
        # The MediaDatabase class should handle initialization
        db_instance = MediaDatabase(db_path)
        return db_instance
        
    except ImportError as e:
        logger.error(f"Failed to import MediaDatabase: {e}")
        raise ImportError("MediaDatabase class not available")


async def ensure_user_directories(user_id: int):
    """
    Ensure all necessary directories exist for a user.
    
    Args:
        user_id: The user's ID
    """
    from pathlib import Path
    
    settings = get_settings()
    
    # Create user database directory
    db_dir = Path(f"{settings.USER_DATA_BASE_PATH}/user_{user_id}")
    db_dir.mkdir(parents=True, exist_ok=True)
    
    # Create user ChromaDB directory
    chroma_dir = Path(f"{settings.CHROMADB_BASE_PATH}/user_{user_id}")
    chroma_dir.mkdir(parents=True, exist_ok=True)
    
    logger.debug(f"Ensured directories exist for user {user_id}")


async def cleanup_user_data(user_id: int):
    """
    Clean up all data associated with a user (for deletion).
    
    Args:
        user_id: The user's ID
        
    Warning:
        This permanently deletes all user data!
    """
    import shutil
    from pathlib import Path
    
    settings = get_settings()
    
    # Remove user database directory
    db_dir = Path(f"{settings.USER_DATA_BASE_PATH}/user_{user_id}")
    if db_dir.exists():
        shutil.rmtree(db_dir)
        logger.info(f"Removed database directory for user {user_id}")
    
    # Remove user ChromaDB directory
    chroma_dir = Path(f"{settings.CHROMADB_BASE_PATH}/user_{user_id}")
    if chroma_dir.exists():
        shutil.rmtree(chroma_dir)
        logger.info(f"Removed ChromaDB directory for user {user_id}")


#
# End of Users_DB.py
#######################################################################################################################