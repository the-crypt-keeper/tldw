# input_validation.py
# Description: Input validation for authentication fields with security focus
#
# Imports
import re
import unicodedata
from typing import Optional, List, Tuple
from email_validator import validate_email, EmailNotValidError
#
# 3rd-party imports
from loguru import logger
#
# Local imports
from tldw_Server_API.app.core.AuthNZ.exceptions import RegistrationError

#######################################################################################################################
#
# Input Validation Service
#

class InputValidator:
    """Service for validating and sanitizing user input in authentication flows"""
    
    def __init__(self):
        """Initialize input validator with security rules"""
        
        # Username constraints
        self.username_min_length = 3
        self.username_max_length = 30
        # Only alphanumeric, underscore, hyphen, and dot allowed
        self.username_pattern = re.compile(r'^[a-zA-Z0-9._-]+$')
        # Cannot start or end with special characters
        self.username_edge_pattern = re.compile(r'^[a-zA-Z0-9].*[a-zA-Z0-9]$')
        
        # Email constraints
        self.email_max_length = 254  # RFC 5321
        
        # Blocked username patterns (security sensitive)
        self.blocked_usernames = {
            'admin', 'administrator', 'root', 'system', 'superuser',
            'moderator', 'operator', 'user', 'test', 'demo', 'guest',
            'api', 'bot', 'service', 'webhook', 'support', 'help',
            'info', 'contact', 'noreply', 'no-reply', 'postmaster',
            'webmaster', 'abuse', 'security', 'null', 'undefined'
        }
        
        # Dangerous patterns to prevent injection attacks
        self.dangerous_patterns = [
            r'<script',  # XSS attempt
            r'javascript:',  # XSS attempt
            r'data:text/html',  # Data URI XSS
            r'vbscript:',  # VBScript injection
            r'on\w+=',  # Event handler injection
            r'\.\./',  # Path traversal
            r'%00',  # Null byte injection
            r'\\x00',  # Null byte (hex)
            r'[<>]',  # HTML tags
        ]
        
        logger.debug("InputValidator initialized with security rules")
    
    def validate_username(self, username: str) -> Tuple[bool, Optional[str]]:
        """
        Validate username against security rules
        
        Args:
            username: Username to validate
            
        Returns:
            Tuple of (is_valid, error_message)
        """
        if not username:
            return False, "Username is required"
        
        # Normalize unicode to prevent homograph attacks
        username = unicodedata.normalize('NFKC', username)
        
        # Length check
        if len(username) < self.username_min_length:
            return False, f"Username must be at least {self.username_min_length} characters"
        
        if len(username) > self.username_max_length:
            return False, f"Username must not exceed {self.username_max_length} characters"
        
        # Character validation
        if not self.username_pattern.match(username):
            return False, "Username can only contain letters, numbers, dots, hyphens, and underscores"
        
        # Edge character validation
        if len(username) > 1 and not self.username_edge_pattern.match(username):
            return False, "Username must start and end with a letter or number"
        
        # Check against blocked list (case-insensitive)
        if username.lower() in self.blocked_usernames:
            return False, "This username is reserved and cannot be used"
        
        # Check for dangerous patterns
        for pattern in self.dangerous_patterns:
            if re.search(pattern, username, re.IGNORECASE):
                logger.warning(f"Dangerous pattern detected in username: {pattern}")
                return False, "Username contains invalid characters or patterns"
        
        # Check for consecutive special characters
        if re.search(r'[._-]{2,}', username):
            return False, "Username cannot contain consecutive special characters"
        
        # Check for confusing characters (e.g., l vs 1, O vs 0)
        if self._has_confusing_characters(username):
            return False, "Username contains potentially confusing characters"
        
        return True, None
    
    def validate_email(self, email: str) -> Tuple[bool, Optional[str]]:
        """
        Validate email address with security focus
        
        Args:
            email: Email address to validate
            
        Returns:
            Tuple of (is_valid, error_message)
        """
        if not email:
            return False, "Email is required"
        
        # Normalize unicode
        email = unicodedata.normalize('NFKC', email)
        
        # Length check
        if len(email) > self.email_max_length:
            return False, f"Email must not exceed {self.email_max_length} characters"
        
        # Use email-validator library for comprehensive validation
        try:
            # This validates format, DNS, and deliverability
            validation = validate_email(email, check_deliverability=False)
            normalized_email = validation.email
            
            # Additional security checks
            local_part = normalized_email.split('@')[0]
            
            # Check for dangerous patterns in local part
            for pattern in self.dangerous_patterns:
                if re.search(pattern, local_part, re.IGNORECASE):
                    logger.warning(f"Dangerous pattern detected in email: {pattern}")
                    return False, "Email contains invalid characters or patterns"
            
            # Check for subdomain abuse (e.g., admin@fake.example.com)
            domain = normalized_email.split('@')[1]
            if domain.count('.') > 3:  # Unusual number of subdomains
                logger.warning(f"Suspicious domain in email: {domain}")
                return False, "Email domain appears invalid"
            
            return True, None
            
        except EmailNotValidError as e:
            return False, str(e)
        except Exception as e:
            logger.error(f"Email validation error: {e}")
            return False, "Email validation failed"
    
    def sanitize_input(self, text: str, max_length: int = 1000) -> str:
        """
        Sanitize general text input
        
        Args:
            text: Text to sanitize
            max_length: Maximum allowed length
            
        Returns:
            Sanitized text
        """
        if not text:
            return ""
        
        # Normalize unicode
        text = unicodedata.normalize('NFKC', text)
        
        # Remove null bytes
        text = text.replace('\x00', '')
        
        # Truncate to max length
        text = text[:max_length]
        
        # Remove control characters except newline and tab
        allowed_chars = {'\n', '\t', '\r'}
        sanitized = []
        for char in text:
            if unicodedata.category(char) == 'Cc' and char not in allowed_chars:
                continue
            sanitized.append(char)
        
        return ''.join(sanitized)
    
    def validate_password_reset_token(self, token: str) -> Tuple[bool, Optional[str]]:
        """
        Validate password reset token format
        
        Args:
            token: Reset token to validate
            
        Returns:
            Tuple of (is_valid, error_message)
        """
        if not token:
            return False, "Reset token is required"
        
        # Check length (typical JWT or random token)
        if len(token) < 20 or len(token) > 2000:
            return False, "Invalid reset token format"
        
        # Check for suspicious patterns
        if re.search(r'[<>\'"]', token):
            return False, "Invalid reset token format"
        
        return True, None
    
    def _has_confusing_characters(self, username: str) -> bool:
        """
        Check for visually confusing character combinations
        
        Args:
            username: Username to check
            
        Returns:
            True if confusing characters detected
        """
        # Confusing pairs that could be used for impersonation
        confusing_sets = [
            {'l', '1', 'I', '|'},  # Lowercase L, one, uppercase i, pipe
            {'O', '0', 'o'},  # Letters O and zero
            {'S', '5', '$'},  # S, five, dollar
            {'Z', '2'},  # Z and two
            {'B', '8'},  # B and eight
            {'G', '6'},  # G and six
        ]
        
        username_chars = set(username)
        
        for confusing_set in confusing_sets:
            # If username contains multiple characters from a confusing set
            if len(username_chars.intersection(confusing_set)) > 1:
                return True
        
        return False
    
    def validate_registration_code(self, code: str) -> Tuple[bool, Optional[str]]:
        """
        Validate registration code format
        
        Args:
            code: Registration code to validate
            
        Returns:
            Tuple of (is_valid, error_message)
        """
        if not code:
            return False, "Registration code is required"
        
        # Expected format: alphanumeric, specific length
        if not re.match(r'^[A-Za-z0-9]{16,32}$', code):
            return False, "Invalid registration code format"
        
        return True, None


#######################################################################################################################
#
# Module Functions for convenience
#

# Global instance
_input_validator: Optional[InputValidator] = None


def get_input_validator() -> InputValidator:
    """Get input validator singleton instance"""
    global _input_validator
    if not _input_validator:
        _input_validator = InputValidator()
    return _input_validator


def validate_username(username: str) -> Tuple[bool, Optional[str]]:
    """Convenience function to validate username"""
    return get_input_validator().validate_username(username)


def validate_email(email: str) -> Tuple[bool, Optional[str]]:
    """Convenience function to validate email"""
    return get_input_validator().validate_email(email)


def sanitize_input(text: str, max_length: int = 1000) -> str:
    """Convenience function to sanitize input"""
    return get_input_validator().sanitize_input(text, max_length)


#
# End of input_validation.py
#######################################################################################################################