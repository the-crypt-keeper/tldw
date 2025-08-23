"""
Configuration management for Evaluations module.

Provides dynamic configuration loading and reloading from YAML files,
environment-specific overrides, and runtime configuration updates.
"""

import os
import yaml
import asyncio
from typing import Dict, Any, Optional, List
from pathlib import Path
from dataclasses import dataclass, field
from datetime import datetime, timezone
from loguru import logger
import threading
import hashlib


@dataclass
class RateLimitTierConfig:
    """Rate limit configuration for a user tier."""
    tier: str
    evaluations_per_minute: int
    batch_evaluations_per_minute: int
    evaluations_per_day: int
    total_tokens_per_day: int
    burst_size: int
    max_cost_per_day: float
    max_cost_per_month: float
    description: str = ""
    
    @classmethod
    def from_dict(cls, tier_name: str, data: Dict[str, Any]) -> "RateLimitTierConfig":
        """Create config from dictionary data."""
        return cls(
            tier=tier_name,
            evaluations_per_minute=data.get("evaluations_per_minute", 10),
            batch_evaluations_per_minute=data.get("batch_evaluations_per_minute", 2),
            evaluations_per_day=data.get("evaluations_per_day", 100),
            total_tokens_per_day=data.get("total_tokens_per_day", 100000),
            burst_size=data.get("burst_size", 5),
            max_cost_per_day=data.get("max_cost_per_day", 1.0),
            max_cost_per_month=data.get("max_cost_per_month", 10.0),
            description=data.get("description", "")
        )


@dataclass
class CircuitBreakerConfig:
    """Circuit breaker configuration for a provider."""
    provider: str
    failure_threshold: int
    success_threshold: int
    timeout_seconds: float
    recovery_timeout_seconds: float
    expected_exceptions: List[str] = field(default_factory=list)
    
    @classmethod
    def from_dict(cls, provider_name: str, data: Dict[str, Any]) -> "CircuitBreakerConfig":
        """Create config from dictionary data."""
        return cls(
            provider=provider_name,
            failure_threshold=data.get("failure_threshold", 5),
            success_threshold=data.get("success_threshold", 2),
            timeout_seconds=data.get("timeout_seconds", 30.0),
            recovery_timeout_seconds=data.get("recovery_timeout_seconds", 60.0),
            expected_exceptions=data.get("expected_exceptions", ["Exception"])
        )


class EvaluationsConfigManager:
    """
    Manages configuration for the Evaluations module.
    
    Supports dynamic loading, environment overrides, and hot reloading.
    """
    
    def __init__(
        self,
        config_path: Optional[str] = None,
        environment: Optional[str] = None,
        enable_hot_reload: bool = False
    ):
        """
        Initialize configuration manager.
        
        Args:
            config_path: Path to YAML configuration file
            environment: Environment name for overrides (dev, test, prod)
            enable_hot_reload: Whether to watch for config file changes
        """
        if config_path is None:
            config_dir = Path(__file__).parent.parent.parent.parent / "Config_Files"
            config_path = config_dir / "evaluations_config.yaml"
        
        self.config_path = Path(config_path)
        self.environment = environment or os.getenv("ENVIRONMENT", "development")
        self.enable_hot_reload = enable_hot_reload
        
        # Configuration cache
        self._config: Dict[str, Any] = {}
        self._config_hash: Optional[str] = None
        self._last_loaded: Optional[datetime] = None
        self._lock = threading.RLock()
        
        # Parsed configurations
        self._rate_limit_tiers: Dict[str, RateLimitTierConfig] = {}
        self._circuit_breaker_configs: Dict[str, CircuitBreakerConfig] = {}
        
        # Hot reload task
        self._reload_task: Optional[asyncio.Task] = None
        
        # Load initial configuration
        self.load_config()
        
        if enable_hot_reload:
            asyncio.create_task(self._start_hot_reload())
    
    def load_config(self) -> bool:
        """
        Load configuration from file.
        
        Returns:
            True if configuration loaded successfully
        """
        try:
            if not self.config_path.exists():
                logger.error(f"Configuration file not found: {self.config_path}")
                return False
            
            # Calculate file hash to detect changes
            with open(self.config_path, 'rb') as f:
                content = f.read()
                new_hash = hashlib.md5(content).hexdigest()
            
            # Skip if file hasn't changed
            if self._config_hash == new_hash:
                return True
            
            # Load YAML configuration
            with open(self.config_path, 'r') as f:
                config = yaml.safe_load(f)
            
            if not config:
                logger.error("Empty or invalid configuration file")
                return False
            
            with self._lock:
                self._config = config
                self._config_hash = new_hash
                self._last_loaded = datetime.now(timezone.utc)
                
                # Apply environment-specific overrides
                self._apply_environment_overrides()
                
                # Parse specialized configurations
                self._parse_rate_limit_configs()
                self._parse_circuit_breaker_configs()
            
            logger.info(f"Configuration loaded successfully from {self.config_path}")
            logger.info(f"Environment: {self.environment}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load configuration: {e}")
            return False
    
    def _apply_environment_overrides(self):
        """Apply environment-specific configuration overrides."""
        environments = self._config.get("environments", {})
        env_config = environments.get(self.environment, {})
        
        if env_config:
            logger.info(f"Applying {self.environment} environment overrides")
            self._deep_merge(self._config, env_config)
    
    def _deep_merge(self, base: Dict[str, Any], override: Dict[str, Any]):
        """Recursively merge override config into base config."""
        for key, value in override.items():
            if key in base and isinstance(base[key], dict) and isinstance(value, dict):
                self._deep_merge(base[key], value)
            else:
                base[key] = value
    
    def _parse_rate_limit_configs(self):
        """Parse rate limiting tier configurations."""
        rate_config = self._config.get("rate_limiting", {})
        tiers_config = rate_config.get("tiers", {})
        
        self._rate_limit_tiers = {}
        for tier_name, tier_data in tiers_config.items():
            try:
                tier_config = RateLimitTierConfig.from_dict(tier_name, tier_data)
                self._rate_limit_tiers[tier_name] = tier_config
            except Exception as e:
                logger.error(f"Invalid rate limit config for tier '{tier_name}': {e}")
        
        logger.info(f"Loaded {len(self._rate_limit_tiers)} rate limit tier configurations")
    
    def _parse_circuit_breaker_configs(self):
        """Parse circuit breaker configurations."""
        cb_config = self._config.get("circuit_breakers", {})
        providers_config = cb_config.get("providers", {})
        
        self._circuit_breaker_configs = {}
        for provider_name, provider_data in providers_config.items():
            try:
                cb_config = CircuitBreakerConfig.from_dict(provider_name, provider_data)
                self._circuit_breaker_configs[provider_name] = cb_config
            except Exception as e:
                logger.error(f"Invalid circuit breaker config for provider '{provider_name}': {e}")
        
        logger.info(f"Loaded {len(self._circuit_breaker_configs)} circuit breaker configurations")
    
    async def _start_hot_reload(self):
        """Start hot reload task to watch for configuration changes."""
        logger.info("Starting configuration hot reload")
        
        while True:
            try:
                await asyncio.sleep(5)  # Check every 5 seconds
                
                if self.config_path.exists():
                    # Check if file has been modified
                    with open(self.config_path, 'rb') as f:
                        content = f.read()
                        current_hash = hashlib.md5(content).hexdigest()
                    
                    if current_hash != self._config_hash:
                        logger.info("Configuration file changed, reloading...")
                        if self.load_config():
                            logger.info("Configuration reloaded successfully")
                        else:
                            logger.error("Failed to reload configuration")
                
            except Exception as e:
                logger.error(f"Hot reload error: {e}")
                await asyncio.sleep(30)  # Wait longer on error
    
    def get_config(self, path: str, default: Any = None) -> Any:
        """
        Get configuration value by dot-separated path.
        
        Args:
            path: Configuration path (e.g., "rate_limiting.global.default_tier")
            default: Default value if path not found
            
        Returns:
            Configuration value or default
        """
        with self._lock:
            current = self._config
            
            for key in path.split('.'):
                if isinstance(current, dict) and key in current:
                    current = current[key]
                else:
                    return default
            
            return current
    
    def get_rate_limit_tier_config(self, tier: str) -> Optional[RateLimitTierConfig]:
        """
        Get rate limit configuration for a tier.
        
        Args:
            tier: Tier name
            
        Returns:
            Tier configuration or None if not found
        """
        return self._rate_limit_tiers.get(tier)
    
    def get_all_rate_limit_tiers(self) -> Dict[str, RateLimitTierConfig]:
        """Get all rate limit tier configurations."""
        return self._rate_limit_tiers.copy()
    
    def get_circuit_breaker_config(self, provider: str) -> Optional[CircuitBreakerConfig]:
        """
        Get circuit breaker configuration for a provider.
        
        Args:
            provider: Provider name
            
        Returns:
            Circuit breaker configuration or default if not found
        """
        config = self._circuit_breaker_configs.get(provider)
        if config is None:
            # Return default configuration
            default_data = self._config.get("circuit_breakers", {}).get("providers", {}).get("default", {})
            if default_data:
                config = CircuitBreakerConfig.from_dict(provider, default_data)
        
        return config
    
    def get_all_circuit_breaker_configs(self) -> Dict[str, CircuitBreakerConfig]:
        """Get all circuit breaker configurations."""
        return self._circuit_breaker_configs.copy()
    
    def update_tier_config(
        self,
        tier: str,
        updates: Dict[str, Any],
        persist: bool = False
    ) -> bool:
        """
        Update rate limit tier configuration at runtime.
        
        Args:
            tier: Tier name
            updates: Configuration updates
            persist: Whether to save changes to file
            
        Returns:
            True if update successful
        """
        try:
            with self._lock:
                if tier not in self._rate_limit_tiers:
                    logger.error(f"Unknown tier: {tier}")
                    return False
                
                # Update in-memory configuration
                tier_data = self._config.get("rate_limiting", {}).get("tiers", {}).get(tier, {})
                tier_data.update(updates)
                
                # Update parsed configuration
                self._rate_limit_tiers[tier] = RateLimitTierConfig.from_dict(tier, tier_data)
                
                if persist:
                    return self._persist_config()
                
                logger.info(f"Updated tier '{tier}' configuration: {updates}")
                return True
                
        except Exception as e:
            logger.error(f"Failed to update tier configuration: {e}")
            return False
    
    def _persist_config(self) -> bool:
        """Persist current configuration to file."""
        try:
            with open(self.config_path, 'w') as f:
                yaml.dump(self._config, f, default_flow_style=False, indent=2)
            
            logger.info(f"Configuration persisted to {self.config_path}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to persist configuration: {e}")
            return False
    
    def get_status(self) -> Dict[str, Any]:
        """
        Get configuration manager status.
        
        Returns:
            Status information
        """
        with self._lock:
            return {
                "config_path": str(self.config_path),
                "environment": self.environment,
                "last_loaded": self._last_loaded.isoformat() if self._last_loaded else None,
                "hot_reload_enabled": self.enable_hot_reload,
                "config_hash": self._config_hash,
                "rate_limit_tiers": list(self._rate_limit_tiers.keys()),
                "circuit_breaker_providers": list(self._circuit_breaker_configs.keys()),
                "config_sections": list(self._config.keys()) if self._config else []
            }
    
    def reload(self) -> bool:
        """Force reload configuration from file."""
        self._config_hash = None  # Force reload
        return self.load_config()
    
    def validate_config(self) -> List[str]:
        """
        Validate current configuration.
        
        Returns:
            List of validation errors (empty if valid)
        """
        errors = []
        
        try:
            # Validate rate limiting configuration
            if "rate_limiting" not in self._config:
                errors.append("Missing 'rate_limiting' section")
            else:
                rate_config = self._config["rate_limiting"]
                
                if "tiers" not in rate_config:
                    errors.append("Missing 'rate_limiting.tiers' section")
                else:
                    for tier_name, tier_data in rate_config["tiers"].items():
                        if not isinstance(tier_data, dict):
                            errors.append(f"Invalid tier configuration: {tier_name}")
                            continue
                        
                        required_fields = [
                            "evaluations_per_minute", "evaluations_per_day",
                            "total_tokens_per_day", "max_cost_per_day"
                        ]
                        
                        for field in required_fields:
                            if field not in tier_data:
                                errors.append(f"Missing field '{field}' in tier '{tier_name}'")
                            elif not isinstance(tier_data[field], (int, float)):
                                errors.append(f"Invalid type for '{field}' in tier '{tier_name}'")
            
            # Validate circuit breaker configuration
            if "circuit_breakers" in self._config:
                cb_config = self._config["circuit_breakers"]
                if "providers" in cb_config:
                    for provider_name, provider_data in cb_config["providers"].items():
                        if not isinstance(provider_data, dict):
                            errors.append(f"Invalid circuit breaker configuration: {provider_name}")
                            continue
                        
                        required_fields = ["failure_threshold", "timeout_seconds"]
                        for field in required_fields:
                            if field not in provider_data:
                                errors.append(f"Missing field '{field}' in circuit breaker '{provider_name}'")
            
        except Exception as e:
            errors.append(f"Configuration validation error: {e}")
        
        return errors


# Global configuration manager instance
config_manager = EvaluationsConfigManager(
    environment=os.getenv("ENVIRONMENT", "development"),
    enable_hot_reload=os.getenv("ENABLE_CONFIG_HOT_RELOAD", "false").lower() == "true"
)


# Convenience functions
def get_rate_limit_config(tier: str) -> Optional[RateLimitTierConfig]:
    """Get rate limit configuration for a tier."""
    return config_manager.get_rate_limit_tier_config(tier)


def get_circuit_breaker_config(provider: str) -> Optional[CircuitBreakerConfig]:
    """Get circuit breaker configuration for a provider."""
    return config_manager.get_circuit_breaker_config(provider)


def get_config(path: str, default: Any = None) -> Any:
    """Get configuration value by path."""
    return config_manager.get_config(path, default)


def reload_config() -> bool:
    """Reload configuration from file."""
    return config_manager.reload()


def validate_config() -> List[str]:
    """Validate current configuration."""
    return config_manager.validate_config()