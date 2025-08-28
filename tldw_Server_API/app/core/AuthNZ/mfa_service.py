# mfa_service.py
# Description: Multi-Factor Authentication service with TOTP support
#
# Imports
import base64
import secrets
import json
import qrcode
from io import BytesIO
from datetime import datetime, timedelta
from typing import Optional, List, Tuple, Dict, Any
#
# 3rd-party imports
import pyotp
from loguru import logger
#
# Local imports
from tldw_Server_API.app.core.AuthNZ.settings import Settings, get_settings
from tldw_Server_API.app.core.AuthNZ.database import DatabasePool, get_db_pool
from tldw_Server_API.app.core.AuthNZ.exceptions import (
    AuthenticationError,
    InvalidTokenError,
    DatabaseError
)

#######################################################################################################################
#
# MFA Service Class
#

class MFAService:
    """
    Multi-Factor Authentication service supporting TOTP (Time-based One-Time Passwords)
    
    Features:
    - TOTP generation and validation
    - QR code generation for authenticator apps
    - Backup codes generation and validation
    - Recovery options
    """
    
    def __init__(
        self,
        db_pool: Optional[DatabasePool] = None,
        settings: Optional[Settings] = None
    ):
        """Initialize MFA service"""
        self.settings = settings or get_settings()
        self.db_pool = db_pool
        self._initialized = False
        
        # TOTP configuration
        self.issuer_name = self.settings.APP_NAME if hasattr(self.settings, 'APP_NAME') else "TLDW Server"
        self.totp_digits = 6
        self.totp_interval = 30  # seconds
        self.backup_codes_count = 8
        
        # Window for TOTP validation (allows for time drift)
        self.validation_window = 1  # Allow 1 interval before/after
        
    async def initialize(self):
        """Initialize MFA service"""
        if self._initialized:
            return
        
        # Get database pool
        if not self.db_pool:
            self.db_pool = await get_db_pool()
        
        self._initialized = True
        logger.info("MFAService initialized")
    
    def generate_secret(self) -> str:
        """
        Generate a new TOTP secret
        
        Returns:
            Base32-encoded secret key
        """
        # Generate 20 bytes (160 bits) of random data
        random_bytes = secrets.token_bytes(20)
        # Encode as base32 for TOTP compatibility
        secret = base64.b32encode(random_bytes).decode('utf-8')
        return secret
    
    def generate_totp_uri(
        self,
        secret: str,
        username: str,
        issuer: Optional[str] = None
    ) -> str:
        """
        Generate TOTP URI for QR code
        
        Args:
            secret: Base32-encoded secret
            username: User's username/email
            issuer: Application name
            
        Returns:
            TOTP URI string
        """
        issuer = issuer or self.issuer_name
        totp = pyotp.TOTP(secret, issuer=issuer)
        return totp.provisioning_uri(
            name=username,
            issuer_name=issuer
        )
    
    def generate_qr_code(self, totp_uri: str) -> bytes:
        """
        Generate QR code image for TOTP URI
        
        Args:
            totp_uri: TOTP URI string
            
        Returns:
            PNG image bytes
        """
        qr = qrcode.QRCode(
            version=1,
            error_correction=qrcode.constants.ERROR_CORRECT_L,
            box_size=10,
            border=4,
        )
        qr.add_data(totp_uri)
        qr.make(fit=True)
        
        img = qr.make_image(fill_color="black", back_color="white")
        
        # Convert to bytes
        buffer = BytesIO()
        img.save(buffer, format='PNG')
        return buffer.getvalue()
    
    def generate_backup_codes(self, count: Optional[int] = None) -> List[str]:
        """
        Generate backup codes for account recovery
        
        Args:
            count: Number of codes to generate
            
        Returns:
            List of backup codes
        """
        count = count or self.backup_codes_count
        codes = []
        
        for _ in range(count):
            # Generate 8-character alphanumeric codes
            code = ''.join(secrets.choice('ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789') for _ in range(8))
            # Format as XXXX-XXXX for readability
            formatted_code = f"{code[:4]}-{code[4:]}"
            codes.append(formatted_code)
        
        return codes
    
    def verify_totp(
        self,
        secret: str,
        token: str,
        window: Optional[int] = None
    ) -> bool:
        """
        Verify a TOTP token
        
        Args:
            secret: User's TOTP secret
            token: 6-digit token to verify
            window: Validation window (intervals before/after)
            
        Returns:
            True if token is valid
        """
        if not secret or not token:
            return False
        
        # Remove any spaces or hyphens from token
        token = token.replace(' ', '').replace('-', '')
        
        # Validate token format
        if not token.isdigit() or len(token) != self.totp_digits:
            return False
        
        try:
            totp = pyotp.TOTP(secret)
            # Use custom window or default
            validation_window = window if window is not None else self.validation_window
            
            # Verify with time window to account for clock drift
            return totp.verify(token, valid_window=validation_window)
            
        except Exception as e:
            logger.error(f"TOTP verification error: {e}")
            return False
    
    async def enable_mfa(
        self,
        user_id: int,
        secret: str,
        backup_codes: List[str]
    ) -> bool:
        """
        Enable MFA for a user
        
        Args:
            user_id: User's ID
            secret: TOTP secret
            backup_codes: List of backup codes
            
        Returns:
            True if successfully enabled
        """
        if not self._initialized:
            await self.initialize()
        
        try:
            # Store encrypted secret and backup codes
            # In production, encrypt these values
            backup_codes_json = json.dumps(backup_codes)
            
            async with self.db_pool.transaction() as conn:
                if hasattr(conn, 'execute'):
                    # PostgreSQL
                    await conn.execute("""
                        UPDATE users 
                        SET totp_secret = $1, 
                            two_factor_enabled = true,
                            backup_codes = $2,
                            updated_at = $3
                        WHERE id = $4
                    """, secret, backup_codes_json, datetime.utcnow(), user_id)
                else:
                    # SQLite
                    await conn.execute("""
                        UPDATE users 
                        SET totp_secret = ?, 
                            two_factor_enabled = 1,
                            backup_codes = ?,
                            updated_at = ?
                        WHERE id = ?
                    """, (secret, backup_codes_json, datetime.utcnow().isoformat(), user_id))
                    await conn.commit()
            
            logger.info(f"MFA enabled for user {user_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to enable MFA: {e}")
            return False
    
    async def disable_mfa(self, user_id: int) -> bool:
        """
        Disable MFA for a user
        
        Args:
            user_id: User's ID
            
        Returns:
            True if successfully disabled
        """
        if not self._initialized:
            await self.initialize()
        
        try:
            async with self.db_pool.transaction() as conn:
                if hasattr(conn, 'execute'):
                    # PostgreSQL
                    await conn.execute("""
                        UPDATE users 
                        SET totp_secret = NULL, 
                            two_factor_enabled = false,
                            backup_codes = NULL,
                            updated_at = $1
                        WHERE id = $2
                    """, datetime.utcnow(), user_id)
                else:
                    # SQLite
                    await conn.execute("""
                        UPDATE users 
                        SET totp_secret = NULL, 
                            two_factor_enabled = 0,
                            backup_codes = NULL,
                            updated_at = ?
                        WHERE id = ?
                    """, (datetime.utcnow().isoformat(), user_id))
                    await conn.commit()
            
            logger.info(f"MFA disabled for user {user_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to disable MFA: {e}")
            return False
    
    async def get_user_mfa_status(self, user_id: int) -> Dict[str, Any]:
        """
        Get MFA status for a user
        
        Args:
            user_id: User's ID
            
        Returns:
            Dictionary with MFA status information
        """
        if not self._initialized:
            await self.initialize()
        
        try:
            async with self.db_pool.acquire() as conn:
                if hasattr(conn, 'fetchrow'):
                    # PostgreSQL
                    result = await conn.fetchrow("""
                        SELECT two_factor_enabled, totp_secret IS NOT NULL as has_secret,
                               backup_codes IS NOT NULL as has_backup_codes
                        FROM users WHERE id = $1
                    """, user_id)
                else:
                    # SQLite
                    cursor = await conn.execute("""
                        SELECT two_factor_enabled, 
                               totp_secret IS NOT NULL as has_secret,
                               backup_codes IS NOT NULL as has_backup_codes
                        FROM users WHERE id = ?
                    """, (user_id,))
                    result = await cursor.fetchone()
                
                if result:
                    return {
                        "enabled": bool(result[0] if isinstance(result, tuple) else result['two_factor_enabled']),
                        "has_secret": bool(result[1] if isinstance(result, tuple) else result['has_secret']),
                        "has_backup_codes": bool(result[2] if isinstance(result, tuple) else result['has_backup_codes']),
                        "method": "totp" if result[0] else None
                    }
                    
        except Exception as e:
            logger.error(f"Failed to get MFA status: {e}")
        
        return {
            "enabled": False,
            "has_secret": False,
            "has_backup_codes": False,
            "method": None
        }
    
    async def verify_backup_code(
        self,
        user_id: int,
        code: str
    ) -> bool:
        """
        Verify and consume a backup code
        
        Args:
            user_id: User's ID
            code: Backup code to verify
            
        Returns:
            True if code is valid and was consumed
        """
        if not self._initialized:
            await self.initialize()
        
        # Normalize code format
        code = code.strip().upper()
        
        try:
            async with self.db_pool.transaction() as conn:
                # Get backup codes
                if hasattr(conn, 'fetchval'):
                    # PostgreSQL
                    backup_codes_json = await conn.fetchval(
                        "SELECT backup_codes FROM users WHERE id = $1",
                        user_id
                    )
                else:
                    # SQLite
                    cursor = await conn.execute(
                        "SELECT backup_codes FROM users WHERE id = ?",
                        (user_id,)
                    )
                    result = await cursor.fetchone()
                    backup_codes_json = result[0] if result else None
                
                if not backup_codes_json:
                    return False
                
                backup_codes = json.loads(backup_codes_json)
                
                # Check if code exists
                if code not in backup_codes:
                    return False
                
                # Remove used code
                backup_codes.remove(code)
                updated_codes_json = json.dumps(backup_codes)
                
                # Update database
                if hasattr(conn, 'execute'):
                    # PostgreSQL
                    await conn.execute(
                        "UPDATE users SET backup_codes = $1 WHERE id = $2",
                        updated_codes_json, user_id
                    )
                else:
                    # SQLite
                    await conn.execute(
                        "UPDATE users SET backup_codes = ? WHERE id = ?",
                        (updated_codes_json, user_id)
                    )
                    await conn.commit()
                
                logger.info(f"Backup code used for user {user_id}")
                return True
                
        except Exception as e:
            logger.error(f"Failed to verify backup code: {e}")
            return False
    
    async def regenerate_backup_codes(
        self,
        user_id: int
    ) -> Optional[List[str]]:
        """
        Generate new backup codes for a user
        
        Args:
            user_id: User's ID
            
        Returns:
            List of new backup codes or None on failure
        """
        if not self._initialized:
            await self.initialize()
        
        try:
            # Generate new codes
            new_codes = self.generate_backup_codes()
            backup_codes_json = json.dumps(new_codes)
            
            async with self.db_pool.transaction() as conn:
                if hasattr(conn, 'execute'):
                    # PostgreSQL
                    await conn.execute(
                        "UPDATE users SET backup_codes = $1, updated_at = $2 WHERE id = $3",
                        backup_codes_json, datetime.utcnow(), user_id
                    )
                else:
                    # SQLite
                    await conn.execute(
                        "UPDATE users SET backup_codes = ?, updated_at = ? WHERE id = ?",
                        (backup_codes_json, datetime.utcnow().isoformat(), user_id)
                    )
                    await conn.commit()
            
            logger.info(f"Regenerated backup codes for user {user_id}")
            return new_codes
            
        except Exception as e:
            logger.error(f"Failed to regenerate backup codes: {e}")
            return None


#######################################################################################################################
#
# Module Functions for convenience
#

# Global instance
_mfa_service: Optional[MFAService] = None


def get_mfa_service() -> MFAService:
    """Get MFA service singleton instance"""
    global _mfa_service
    if not _mfa_service:
        _mfa_service = MFAService()
    return _mfa_service


def generate_totp_secret() -> str:
    """Generate a new TOTP secret"""
    return get_mfa_service().generate_secret()


def verify_totp_token(secret: str, token: str) -> bool:
    """Verify a TOTP token"""
    return get_mfa_service().verify_totp(secret, token)


#
# End of mfa_service.py
#######################################################################################################################