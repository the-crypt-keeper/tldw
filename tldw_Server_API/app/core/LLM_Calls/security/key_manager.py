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
import time
from typing import Dict, Optional, Any, Set
from dataclasses import dataclass
from datetime import datetime, timedelta
from loguru import logger

from tldw_Server_API.app.core.config import load_and_log_configs


@dataclass
class APIKeyInfo:
    """Metadata about an API key (without the actual key)."""
    provider: str
    key_id: str  # Hash of the key for identification
    created_at: datetime
    last_used: Optional[datetime] = None
    usage_count: int = 0
    is_valid: bool = True


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
        
    def _load_config_if_needed(self) -> Dict[str, Any]:
        """Load configuration if not cached or expired."""
        current_time = time.time()
        if (self._config is None or 
            current_time - self._last_config_load > self._key_cache_ttl):
            self._config = load_and_log_configs()
            self._last_config_load = current_time
            logger.debug("Reloaded configuration for API keys")
        return self._config
    
    def _hash_key(self, api_key: str) -> str:
        """
        Create a secure hash of an API key for identification.
        
        Args:
            api_key: The API key to hash
            
        Returns:
            Hex string of the hashed key
        """
        # Use SHA-256 with a salt for better security
        salt = os.environ.get('API_KEY_SALT', 'default_salt_change_me')
        return hashlib.sha256(f"{salt}{api_key}".encode()).hexdigest()
    
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
            key_hash = self._hash_key(api_key)
            if key_hash in self._blocked_keys:
                logger.error(f"Blocked API key attempted for provider: {provider}")
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
        key_hash = self._hash_key(api_key)
        
        if key_hash not in self._key_info:
            self._key_info[key_hash] = APIKeyInfo(
                provider=provider,
                key_id=key_hash[:8],  # Short ID for logging
                created_at=datetime.now(),
            )
        
        info = self._key_info[key_hash]
        info.last_used = datetime.now()
        info.usage_count += 1
        
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
        key_hash = self._hash_key(api_key)
        self._blocked_keys.add(key_hash)
        logger.warning(f"API key blocked - Hash: {key_hash[:8]}, Reason: {reason}")
    
    def unblock_key(self, api_key: str):
        """
        Unblock a previously blocked API key.
        
        Args:
            api_key: The API key to unblock
        """
        key_hash = self._hash_key(api_key)
        if key_hash in self._blocked_keys:
            self._blocked_keys.remove(key_hash)
            logger.info(f"API key unblocked - Hash: {key_hash[:8]}")
    
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