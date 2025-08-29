"""
Secure API key management for LLM providers.

This module provides secure handling of API keys, including:
- Safe retrieval from configuration
- Key validation without logging
- Audit logging without exposing keys
- Key rotation support
"""

import hashlib
import hmac
import os
import secrets
import time
import json
import base64
from typing import Dict, Optional, Any, Set, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from pathlib import Path
from loguru import logger

from tldw_Server_API.app.core.config import load_and_log_configs


@dataclass
class APIKeyInfo:
    """Metadata about an API key (without the actual key)."""
    provider: str
    key_id: str  # Hash of the key for identification
    key_hash: str  # Full hash for verification
    salt: str  # Unique salt for this key
    created_at: datetime
    last_used: Optional[datetime] = None
    usage_count: int = 0
    is_valid: bool = True
    iterations: int = 100000  # PBKDF2 iterations


class KeyManager:
    """
    Secure management of API keys for LLM providers.
    
    This class handles:
    - Safe retrieval of API keys from configuration
    - Key validation and verification
    - Audit logging without exposing sensitive data
    - Usage tracking and monitoring
    """
    
    def __init__(self):
        """Initialize the key manager."""
        self._keys: Dict[str, str] = {}
        self._key_info: Dict[str, APIKeyInfo] = {}
        self._config: Optional[Dict[str, Any]] = None
        self._key_cache_ttl = 300  # 5 minutes
        self._last_config_load = 0
        self._blocked_keys: Set[str] = set()
        self._master_key: bytes = self._get_or_create_master_key()
        self._key_store_path = Path("./key_hashes.json")  # Store for persistent key info
        self._load_key_store()
        
    def _load_config_if_needed(self) -> Dict[str, Any]:
        """Load configuration if not cached or expired."""
        current_time = time.time()
        if (self._config is None or 
            current_time - self._last_config_load > self._key_cache_ttl):
            self._config = load_and_log_configs()
            self._last_config_load = current_time
            logger.debug("Reloaded configuration for API keys")
        return self._config
    
    def _get_or_create_master_key(self) -> bytes:
        """
        Get or create a master key for HMAC operations.
        
        Returns:
            Master key bytes
        """
        # Try to get from environment first
        env_key = os.environ.get('API_KEY_MASTER_KEY')
        if env_key:
            return base64.b64decode(env_key.encode())
        
        # Generate a secure random key if not set
        master_key_file = Path(".master_key")
        if master_key_file.exists():
            with open(master_key_file, 'rb') as f:
                return f.read()
        else:
            # Generate new master key
            new_key = secrets.token_bytes(32)
            with open(master_key_file, 'wb') as f:
                f.write(new_key)
            # Set restrictive permissions (Unix-like systems)
            try:
                os.chmod(master_key_file, 0o600)
            except:
                pass  # Windows doesn't support chmod
            logger.info("Generated new master key for API key hashing")
            return new_key
    
    def _load_key_store(self):
        """Load persisted key information from disk."""
        if self._key_store_path.exists():
            try:
                with open(self._key_store_path, 'r') as f:
                    data = json.load(f)
                    for key_id, info_dict in data.items():
                        # Convert datetime strings back to datetime objects
                        info_dict['created_at'] = datetime.fromisoformat(info_dict['created_at'])
                        if info_dict.get('last_used'):
                            info_dict['last_used'] = datetime.fromisoformat(info_dict['last_used'])
                        self._key_info[key_id] = APIKeyInfo(**info_dict)
            except Exception as e:
                logger.error(f"Error loading key store: {e}")
    
    def _save_key_store(self):
        """Save key information to disk."""
        try:
            data = {}
            for key_id, info in self._key_info.items():
                info_dict = asdict(info)
                # Convert datetime objects to ISO format strings
                info_dict['created_at'] = info.created_at.isoformat()
                if info.last_used:
                    info_dict['last_used'] = info.last_used.isoformat()
                data[key_id] = info_dict
            
            with open(self._key_store_path, 'w') as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            logger.error(f"Error saving key store: {e}")
    
    def _generate_salt(self) -> str:
        """
        Generate a cryptographically secure random salt.
        
        Returns:
            Base64-encoded salt string
        """
        return base64.b64encode(secrets.token_bytes(32)).decode('utf-8')
    
    def _hash_key_pbkdf2(self, api_key: str, salt: str, iterations: int = 100000) -> str:
        """
        Create a secure hash of an API key using PBKDF2-HMAC-SHA256.
        
        Args:
            api_key: The API key to hash
            salt: Unique salt for this key
            iterations: Number of PBKDF2 iterations (default 100000)
            
        Returns:
            Base64-encoded hash string
        """
        # Use PBKDF2 with HMAC-SHA256 for key stretching
        dk = hashlib.pbkdf2_hmac(
            'sha256',
            api_key.encode('utf-8'),
            base64.b64decode(salt.encode('utf-8')),
            iterations,
            dklen=32
        )
        return base64.b64encode(dk).decode('utf-8')
    
    def _hash_key_hmac(self, api_key: str) -> str:
        """
        Create a quick HMAC-SHA256 hash for key identification.
        
        Args:
            api_key: The API key to hash
            
        Returns:
            Hex string of the HMAC hash
        """
        return hmac.new(
            self._master_key,
            api_key.encode('utf-8'),
            hashlib.sha256
        ).hexdigest()
    
    def _hash_key(self, api_key: str) -> str:
        """
        Create a secure hash of an API key for identification.
        This uses HMAC-SHA256 for quick lookups.
        
        Args:
            api_key: The API key to hash
            
        Returns:
            Hex string of the hashed key
        """
        return self._hash_key_hmac(api_key)
    
    def _create_key_info(self, provider: str, api_key: str) -> APIKeyInfo:
        """
        Create a new APIKeyInfo entry with secure hashing.
        
        Args:
            provider: The provider name
            api_key: The API key
            
        Returns:
            APIKeyInfo object with secure hash
        """
        salt = self._generate_salt()
        key_hash = self._hash_key_pbkdf2(api_key, salt)
        key_id = self._hash_key_hmac(api_key)[:16]  # Short ID for identification
        
        return APIKeyInfo(
            provider=provider,
            key_id=key_id,
            key_hash=key_hash,
            salt=salt,
            created_at=datetime.now(),
            iterations=100000
        )
    
    def _verify_key(self, api_key: str, key_info: APIKeyInfo) -> bool:
        """
        Verify an API key against stored hash.
        
        Args:
            api_key: The API key to verify
            key_info: The stored key information
            
        Returns:
            True if the key matches
        """
        computed_hash = self._hash_key_pbkdf2(
            api_key, 
            key_info.salt, 
            key_info.iterations
        )
        # Use constant-time comparison to prevent timing attacks
        return hmac.compare_digest(computed_hash, key_info.key_hash)
    
    def get_api_key(self, provider: str, api_key: Optional[str] = None) -> Optional[str]:
        """
        Securely retrieve an API key for a provider.
        
        Args:
            provider: The LLM provider name (e.g., 'openai', 'anthropic')
            api_key: Optional API key provided by user
            
        Returns:
            The API key if found, None otherwise
        """
        # If API key is provided, validate and return it
        if api_key:
            if self._validate_api_key_format(provider, api_key):
                self._track_key_usage(provider, api_key)
                return api_key
            else:
                logger.warning(f"Invalid API key format for provider: {provider}")
                return None
        
        # Load from configuration
        config = self._load_config_if_needed()
        
        # Map provider names to config sections
        provider_config_map = {
            'openai': 'openai_api',
            'anthropic': 'anthropic_api',
            'cohere': 'cohere_api',
            'groq': 'groq_api',
            'mistral': 'mistral_api',
            'deepseek': 'deepseek_api',
            'google': 'google_api',
            'openrouter': 'openrouter_api',
            'huggingface': 'huggingface_api',
            'moonshot': 'moonshot_api',
            'zai': 'zai_api',
            # Local providers
            'llama': 'llama_api',
            'kobold': 'kobold_api',
            'oobabooga': 'oobabooga_api',
            'tabbyapi': 'tabbyapi_api',
            'vllm': 'vllm_api',
            'ollama': 'ollama_api',
            'aphrodite': 'aphrodite_api',
            'custom_openai': 'custom_openai_api',
        }
        
        config_section = provider_config_map.get(provider)
        if not config_section:
            logger.warning(f"Unknown provider: {provider}")
            return None
        
        provider_config = config.get(config_section, {})
        api_key = provider_config.get('api_key')
        
        if api_key:
            # Check if key is blocked
            key_id = self._hash_key(api_key)
            if key_id in self._blocked_keys:
                logger.error(f"Blocked API key attempted for provider: {provider}")
                return None
            
            # Check if key is marked as invalid
            if key_id in self._key_info and not self._key_info[key_id].is_valid:
                logger.error(f"Invalid API key attempted for provider: {provider}")
                return None
            
            self._track_key_usage(provider, api_key)
            return api_key
        
        logger.debug(f"No API key found for provider: {provider}")
        return None
    
    def _validate_api_key_format(self, provider: str, api_key: str) -> bool:
        """
        Validate the format of an API key for a specific provider.
        
        Args:
            provider: The provider name
            api_key: The API key to validate
            
        Returns:
            True if the key format is valid
        """
        if not api_key or not isinstance(api_key, str):
            return False
        
        # Remove whitespace
        api_key = api_key.strip()
        
        # Provider-specific validation
        validations = {
            'openai': lambda k: k.startswith('sk-') and len(k) > 20,
            'anthropic': lambda k: k.startswith('sk-ant-') and len(k) > 30,
            'cohere': lambda k: len(k) > 20,
            'groq': lambda k: k.startswith('gsk_') and len(k) > 30,
            'mistral': lambda k: len(k) > 20,
            'deepseek': lambda k: k.startswith('sk-') and len(k) > 20,
            'google': lambda k: len(k) > 30,
            'openrouter': lambda k: len(k) > 20,
            'huggingface': lambda k: k.startswith('hf_') and len(k) > 20,
            'moonshot': lambda k: k.startswith('sk-') and len(k) > 20,
            'zai': lambda k: len(k) > 20,
        }
        
        # Local providers typically don't need API keys or have simple formats
        local_providers = {
            'llama', 'kobold', 'oobabooga', 'tabbyapi', 
            'vllm', 'ollama', 'aphrodite', 'custom_openai'
        }
        
        if provider in local_providers:
            # For local providers, any non-empty key is valid
            return len(api_key) > 0
        
        validator = validations.get(provider)
        if validator:
            try:
                return validator(api_key)
            except Exception as e:
                logger.error(f"Error validating API key format for {provider}: {e}")
                return False
        
        # Default: accept any key with reasonable length
        return 10 <= len(api_key) <= 500
    
    def _track_key_usage(self, provider: str, api_key: str):
        """
        Track usage of an API key without logging the key itself.
        
        Args:
            provider: The provider name
            api_key: The API key being used
        """
        key_id = self._hash_key(api_key)
        
        if key_id not in self._key_info:
            # Create new key info with secure hashing
            self._key_info[key_id] = self._create_key_info(provider, api_key)
        
        info = self._key_info[key_id]
        info.last_used = datetime.now()
        info.usage_count += 1
        
        # Save updated info to disk
        self._save_key_store()
        
        # Log usage without exposing the key
        logger.debug(f"API key used - Provider: {provider}, Key ID: {info.key_id}, Usage: {info.usage_count}")
    
    def validate_key_availability(self, provider: str, api_key: Optional[str] = None) -> bool:
        """
        Check if a valid API key is available for a provider.
        
        Args:
            provider: The provider name
            api_key: Optional API key to check
            
        Returns:
            True if a valid key is available
        """
        key = self.get_api_key(provider, api_key)
        return key is not None and len(key) > 0
    
    def block_key(self, api_key: str, reason: str = "Unknown"):
        """
        Block an API key from being used.
        
        Args:
            api_key: The API key to block
            reason: Reason for blocking
        """
        key_id = self._hash_key(api_key)
        self._blocked_keys.add(key_id)
        
        # Mark as invalid in key info if it exists
        if key_id in self._key_info:
            self._key_info[key_id].is_valid = False
            self._save_key_store()
        
        logger.warning(f"API key blocked - ID: {key_id[:16]}, Reason: {reason}")
    
    def unblock_key(self, api_key: str):
        """
        Unblock a previously blocked API key.
        
        Args:
            api_key: The API key to unblock
        """
        key_id = self._hash_key(api_key)
        if key_id in self._blocked_keys:
            self._blocked_keys.remove(key_id)
            
            # Mark as valid in key info if it exists
            if key_id in self._key_info:
                self._key_info[key_id].is_valid = True
                self._save_key_store()
            
            logger.info(f"API key unblocked - ID: {key_id[:16]}")
    
    def get_usage_stats(self) -> Dict[str, Dict[str, Any]]:
        """
        Get usage statistics for all tracked API keys.
        
        Returns:
            Dictionary of usage stats by provider
        """
        stats = {}
        
        for key_hash, info in self._key_info.items():
            if info.provider not in stats:
                stats[info.provider] = {
                    'total_keys': 0,
                    'total_usage': 0,
                    'active_keys': 0,
                    'blocked_keys': 0,
                }
            
            provider_stats = stats[info.provider]
            provider_stats['total_keys'] += 1
            provider_stats['total_usage'] += info.usage_count
            
            # Check if key was used recently (last 24 hours)
            if info.last_used:
                if datetime.now() - info.last_used < timedelta(hours=24):
                    provider_stats['active_keys'] += 1
            
            if key_hash in self._blocked_keys:
                provider_stats['blocked_keys'] += 1
        
        return stats
    
    def audit_log(self, provider: str, action: str, success: bool, 
                  error: Optional[str] = None, metadata: Optional[Dict] = None):
        """
        Create an audit log entry for API key usage.
        
        Args:
            provider: The provider name
            action: The action performed (e.g., 'chat_completion')
            success: Whether the action was successful
            error: Error message if applicable
            metadata: Additional metadata to log
        """
        log_entry = {
            'timestamp': datetime.now().isoformat(),
            'provider': provider,
            'action': action,
            'success': success,
        }
        
        if error:
            log_entry['error'] = error
        
        if metadata:
            # Filter out sensitive data from metadata
            safe_metadata = {
                k: v for k, v in metadata.items() 
                if k not in ('api_key', 'key', 'token', 'secret', 'password')
            }
            log_entry['metadata'] = safe_metadata
        
        if success:
            logger.info(f"API action successful: {log_entry}")
        else:
            logger.error(f"API action failed: {log_entry}")
    
    def rotate_key(self, provider: str, old_key: str, new_key: str) -> bool:
        """
        Rotate an API key for a provider.
        
        Args:
            provider: The provider name
            old_key: The old API key
            new_key: The new API key
            
        Returns:
            True if rotation was successful
        """
        # Validate new key format
        if not self._validate_api_key_format(provider, new_key):
            logger.error(f"Invalid new API key format for provider: {provider}")
            return False
        
        # Block old key
        self.block_key(old_key, reason="Key rotation")
        
        # Track new key
        self._track_key_usage(provider, new_key)
        
        logger.info(f"API key rotated for provider: {provider}")
        return True


# Global key manager instance
_key_manager = None


def get_key_manager() -> KeyManager:
    """Get the global key manager instance."""
    global _key_manager
    if _key_manager is None:
        _key_manager = KeyManager()
    return _key_manager