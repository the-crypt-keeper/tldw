# password_service.py
# Description: Password hashing and validation service using Argon2
#
# Imports
import re
import secrets
import string
from typing import Tuple, Optional, List
#
# 3rd-party imports
from argon2 import PasswordHasher
from argon2.exceptions import VerifyMismatchError, VerificationError
from loguru import logger
#
# Local imports
from tldw_Server_API.app.core.AuthNZ.settings import Settings, get_settings
from tldw_Server_API.app.core.AuthNZ.exceptions import WeakPasswordError

#######################################################################################################################
#
# Password Service Class

class PasswordService:
    """Service for password hashing, validation, and strength checking using Argon2"""
    
    def __init__(self, settings: Optional[Settings] = None):
        """Initialize password service with Argon2 hasher"""
        self.settings = settings or get_settings()
        
        # Initialize Argon2 hasher with configured parameters
        self.hasher = PasswordHasher(
            time_cost=self.settings.ARGON2_TIME_COST,
            memory_cost=self.settings.ARGON2_MEMORY_COST,
            parallelism=self.settings.ARGON2_PARALLELISM,
            hash_len=32,
            salt_len=16
        )
        
        # Password requirements
        self.min_length = self.settings.PASSWORD_MIN_LENGTH
        
        # Common weak passwords to check against
        self.common_passwords = {
            "password", "123456", "password123", "admin", "letmein",
            "qwerty", "monkey", "dragon", "master", "superman",
            "password1", "123456789", "qwertyuiop", "1234567890",
            "welcome", "welcome123", "admin123", "root", "toor",
            "pass", "pass123", "password1234", "qwerty123"
        }
        
        logger.debug(
            f"PasswordService initialized with Argon2 (time_cost={self.settings.ARGON2_TIME_COST}, "
            f"memory_cost={self.settings.ARGON2_MEMORY_COST}KB, parallelism={self.settings.ARGON2_PARALLELISM})"
        )
    
    def hash_password(self, password: str) -> str:
        """
        Hash a password using Argon2id
        
        Args:
            password: Plain text password to hash
            
        Returns:
            Argon2 hash string
            
        Raises:
            WeakPasswordError: If password doesn't meet requirements
        """
        # Validate password strength first
        self.validate_password_strength(password)
        
        try:
            # Hash the password
            password_hash = self.hasher.hash(password)
            logger.debug("Password hashed successfully")
            return password_hash
            
        except Exception as e:
            logger.error(f"Failed to hash password: {e}")
            raise
    
    def verify_password(self, password: str, password_hash: str) -> Tuple[bool, bool]:
        """
        Verify a password against its hash
        
        Args:
            password: Plain text password to verify
            password_hash: Argon2 hash to verify against
            
        Returns:
            Tuple of (is_valid, needs_rehash)
            - is_valid: True if password matches
            - needs_rehash: True if hash parameters have changed
        """
        try:
            # Verify the password
            self.hasher.verify(password_hash, password)
            
            # Check if rehashing is needed (parameters changed)
            needs_rehash = self.hasher.check_needs_rehash(password_hash)
            
            if needs_rehash:
                logger.info("Password hash needs rehashing due to parameter changes")
            
            return True, needs_rehash
            
        except VerifyMismatchError:
            # Password doesn't match
            logger.debug("Password verification failed - mismatch")
            return False, False
            
        except VerificationError as e:
            # Hash is malformed or corrupted
            logger.error(f"Password verification error - invalid hash: {e}")
            return False, False
            
        except Exception as e:
            # Unexpected error
            logger.error(f"Unexpected error during password verification: {e}")
            return False, False
    
    def validate_password_strength(self, password: str, username: Optional[str] = None) -> None:
        """
        Validate password meets security requirements
        
        Args:
            password: Password to validate
            username: Username to check password doesn't contain
            
        Raises:
            WeakPasswordError: If password doesn't meet requirements
        """
        errors = []
        
        # Check minimum length
        if len(password) < self.min_length:
            errors.append(f"Password must be at least {self.min_length} characters long")
        
        # Check for at least one uppercase letter
        if not re.search(r'[A-Z]', password):
            errors.append("Password must contain at least one uppercase letter")
        
        # Check for at least one lowercase letter
        if not re.search(r'[a-z]', password):
            errors.append("Password must contain at least one lowercase letter")
        
        # Check for at least one digit
        if not re.search(r'\d', password):
            errors.append("Password must contain at least one number")
        
        # Check for at least one special character
        if not re.search(r'[!@#$%^&*(),.?":{}|<>]', password):
            errors.append("Password must contain at least one special character")
        
        # Check against common passwords
        if password.lower() in self.common_passwords:
            errors.append("Password is too common, please choose a more unique password")
        
        # Check password doesn't contain username
        if username and username.lower() in password.lower():
            errors.append("Password must not contain your username")
        
        # Check for sequential or repeated characters
        if self._has_sequential_chars(password) or self._has_repeated_chars(password):
            errors.append("Password must not contain sequential or repeated characters")
        
        # If there are any errors, raise exception
        if errors:
            raise WeakPasswordError("; ".join(errors))
    
    def _has_sequential_chars(self, password: str, max_sequence: int = 3) -> bool:
        """Check if password has sequential characters (e.g., 'abc', '123')
        but allow year patterns like 2024"""
        import re
        
        for i in range(len(password) - max_sequence + 1):
            substring = password[i:i + max_sequence]
            
            # Skip if this looks like a year (2020-2099)
            if len(substring) >= 4 and re.match(r'20[2-9]\d', substring[:4]):
                continue
            
            # Check ascending sequence
            if all(ord(substring[j+1]) - ord(substring[j]) == 1 for j in range(len(substring) - 1)):
                return True
            
            # Check descending sequence
            if all(ord(substring[j]) - ord(substring[j+1]) == 1 for j in range(len(substring) - 1)):
                return True
        
        return False
    
    def _has_repeated_chars(self, password: str, max_repeat: int = 3) -> bool:
        """Check if password has repeated characters (e.g., 'aaa', '111')"""
        for i in range(len(password) - max_repeat + 1):
            if len(set(password[i:i + max_repeat])) == 1:
                return True
        return False
    
    def generate_secure_password(self, length: int = 16) -> str:
        """
        Generate a cryptographically secure random password
        
        Args:
            length: Length of password to generate (minimum 12)
            
        Returns:
            Secure random password string
        """
        if length < 12:
            length = 12
        
        # Character sets to use
        lowercase = string.ascii_lowercase
        uppercase = string.ascii_uppercase
        digits = string.digits
        special = "!@#$%^&*()_+-=[]{}|;:,.<>?"
        
        # Ensure at least one character from each set
        password = [
            secrets.choice(lowercase),
            secrets.choice(uppercase),
            secrets.choice(digits),
            secrets.choice(special)
        ]
        
        # Fill the rest with random characters from all sets
        all_chars = lowercase + uppercase + digits + special
        for _ in range(length - 4):
            password.append(secrets.choice(all_chars))
        
        # Shuffle the password
        secrets.SystemRandom().shuffle(password)
        
        return ''.join(password)
    
    def generate_registration_code(self, length: int = 24) -> str:
        """
        Generate a secure registration code
        
        Args:
            length: Length of code to generate
            
        Returns:
            Secure random code string (alphanumeric)
        """
        alphabet = string.ascii_letters + string.digits
        return ''.join(secrets.choice(alphabet) for _ in range(length))
    
    async def check_password_history(
        self,
        user_id: int,
        new_password: str,
        db_connection
    ) -> bool:
        """
        Check if password was recently used by the user
        
        Args:
            user_id: User ID to check history for
            new_password: New password to check
            db_connection: Database connection
            
        Returns:
            True if password is acceptable (not in history), False otherwise
        """
        try:
            # Get recent password hashes
            history_count = self.settings.PASSWORD_HISTORY_RETENTION_COUNT
            
            if hasattr(db_connection, 'fetch'):
                # PostgreSQL
                rows = await db_connection.fetch(
                    """
                    SELECT password_hash FROM password_history
                    WHERE user_id = $1
                    ORDER BY created_at DESC
                    LIMIT $2
                    """,
                    user_id, history_count
                )
                password_hashes = [row['password_hash'] for row in rows]
            else:
                # SQLite
                cursor = await db_connection.execute(
                    """
                    SELECT password_hash FROM password_history
                    WHERE user_id = ?
                    ORDER BY created_at DESC
                    LIMIT ?
                    """,
                    (user_id, history_count)
                )
                rows = await cursor.fetchall()
                password_hashes = [row['password_hash'] for row in rows]
            
            # Check if new password matches any in history
            for old_hash in password_hashes:
                is_match, _ = self.verify_password(new_password, old_hash)
                if is_match:
                    logger.warning(f"User {user_id} attempted to reuse a recent password")
                    return False
            
            return True
            
        except Exception as e:
            logger.error(f"Error checking password history: {e}")
            # On error, allow the password (don't block user)
            return True
    
    async def add_to_password_history(
        self,
        user_id: int,
        password_hash: str,
        db_connection
    ) -> None:
        """
        Add a password hash to user's password history
        
        Args:
            user_id: User ID
            password_hash: Password hash to add to history
            db_connection: Database connection
        """
        try:
            if hasattr(db_connection, 'execute'):
                # PostgreSQL
                await db_connection.execute(
                    """
                    INSERT INTO password_history (user_id, password_hash)
                    VALUES ($1, $2)
                    """,
                    user_id, password_hash
                )
                
                # Clean up old entries
                await db_connection.execute(
                    """
                    DELETE FROM password_history
                    WHERE user_id = $1 AND id NOT IN (
                        SELECT id FROM password_history
                        WHERE user_id = $1
                        ORDER BY created_at DESC
                        LIMIT $2
                    )
                    """,
                    user_id, self.settings.PASSWORD_HISTORY_RETENTION_COUNT
                )
            else:
                # SQLite
                await db_connection.execute(
                    """
                    INSERT INTO password_history (user_id, password_hash)
                    VALUES (?, ?)
                    """,
                    (user_id, password_hash)
                )
                
                # Clean up old entries
                await db_connection.execute(
                    """
                    DELETE FROM password_history
                    WHERE user_id = ? AND id NOT IN (
                        SELECT id FROM password_history
                        WHERE user_id = ?
                        ORDER BY created_at DESC
                        LIMIT ?
                    )
                    """,
                    (user_id, user_id, self.settings.PASSWORD_HISTORY_RETENTION_COUNT)
                )
                
            logger.debug(f"Added password to history for user {user_id}")
            
        except Exception as e:
            logger.error(f"Error adding to password history: {e}")
            # Don't raise - this shouldn't block password changes


#######################################################################################################################
#
# Module Functions for convenience

# Global instance
_password_service: Optional[PasswordService] = None


def get_password_service() -> PasswordService:
    """Get password service singleton instance"""
    global _password_service
    if not _password_service:
        _password_service = PasswordService()
    return _password_service


def hash_password(password: str) -> str:
    """Convenience function to hash a password"""
    return get_password_service().hash_password(password)


def verify_password(password: str, password_hash: str) -> Tuple[bool, bool]:
    """Convenience function to verify a password"""
    return get_password_service().verify_password(password, password_hash)


def generate_secure_password(length: int = 16) -> str:
    """Convenience function to generate a secure password"""
    return get_password_service().generate_secure_password(length)


def generate_registration_code(length: int = 24) -> str:
    """Convenience function to generate a registration code"""
    return get_password_service().generate_registration_code(length)


#
# End of password_service.py
#######################################################################################################################