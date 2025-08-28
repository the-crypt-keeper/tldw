# simplified_config.py
# Simplified configuration system for embeddings module

import os
import yaml
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, field, asdict
from pathlib import Path

from loguru import logger
from pydantic import BaseModel, Field, validator


@dataclass
class ProviderConfig:
    """Configuration for a single embedding provider"""
    name: str
    priority: int = 1
    enabled: bool = True
    api_key: Optional[str] = None
    api_url: Optional[str] = None
    models: List[str] = field(default_factory=list)
    max_connections: int = 10
    timeout_seconds: int = 30
    rate_limit: Optional[str] = None  # e.g., "100/min"
    fallback_provider: Optional[str] = None
    
    def __post_init__(self):
        # Load API key from environment if not provided
        if not self.api_key:
            env_key = f"{self.name.upper()}_API_KEY"
            self.api_key = os.getenv(env_key)


@dataclass
class CacheConfig:
    """Cache configuration"""
    enabled: bool = True
    ttl_seconds: int = 3600
    max_size: int = 10000
    cleanup_interval: int = 300


@dataclass
class ResourceConfig:
    """Resource management configuration"""
    max_models_in_memory: int = 3
    max_memory_gb: float = 8.0
    model_ttl_seconds: int = 3600
    enable_model_warmup: bool = False
    warmup_models: List[str] = field(default_factory=list)


@dataclass
class BatchingConfig:
    """Request batching configuration"""
    enabled: bool = True
    max_batch_size: int = 32
    batch_timeout_ms: int = 100
    adaptive_batching: bool = True


@dataclass
class SecurityConfig:
    """Security configuration"""
    enable_request_signing: bool = True
    enable_audit_logging: bool = True
    enable_rate_limiting: bool = True
    max_input_length: int = 100000
    validate_embeddings: bool = True


@dataclass
class MonitoringConfig:
    """Monitoring configuration"""
    enable_metrics: bool = True
    metrics_port: int = 9090
    enable_health_check: bool = True
    health_check_interval: int = 30


@dataclass
class EmbeddingsConfig:
    """Main configuration for embeddings module"""
    providers: List[ProviderConfig] = field(default_factory=list)
    cache: CacheConfig = field(default_factory=CacheConfig)
    resources: ResourceConfig = field(default_factory=ResourceConfig)
    batching: BatchingConfig = field(default_factory=BatchingConfig)
    security: SecurityConfig = field(default_factory=SecurityConfig)
    monitoring: MonitoringConfig = field(default_factory=MonitoringConfig)
    
    # Global settings
    default_provider: str = "openai"
    default_model: str = "text-embedding-3-small"
    chunk_size: int = 400
    chunk_overlap: int = 200
    
    @classmethod
    def from_yaml(cls, path: str) -> 'EmbeddingsConfig':
        """Load configuration from YAML file"""
        with open(path, 'r') as f:
            data = yaml.safe_load(f)
        return cls.from_dict(data)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'EmbeddingsConfig':
        """Create configuration from dictionary"""
        # Parse providers
        providers = []
        for provider_data in data.get('providers', []):
            providers.append(ProviderConfig(**provider_data))
        
        # Parse sub-configurations
        config = cls(
            providers=providers,
            cache=CacheConfig(**data.get('cache', {})),
            resources=ResourceConfig(**data.get('resources', {})),
            batching=BatchingConfig(**data.get('batching', {})),
            security=SecurityConfig(**data.get('security', {})),
            monitoring=MonitoringConfig(**data.get('monitoring', {})),
            default_provider=data.get('default_provider', 'openai'),
            default_model=data.get('default_model', 'text-embedding-3-small'),
            chunk_size=data.get('chunk_size', 400),
            chunk_overlap=data.get('chunk_overlap', 200)
        )
        
        return config
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            'providers': [asdict(p) for p in self.providers],
            'cache': asdict(self.cache),
            'resources': asdict(self.resources),
            'batching': asdict(self.batching),
            'security': asdict(self.security),
            'monitoring': asdict(self.monitoring),
            'default_provider': self.default_provider,
            'default_model': self.default_model,
            'chunk_size': self.chunk_size,
            'chunk_overlap': self.chunk_overlap
        }
    
    def to_yaml(self, path: str):
        """Save configuration to YAML file"""
        with open(path, 'w') as f:
            yaml.dump(self.to_dict(), f, default_flow_style=False)
    
    def get_provider(self, name: str) -> Optional[ProviderConfig]:
        """Get provider configuration by name"""
        for provider in self.providers:
            if provider.name == name:
                return provider
        return None
    
    def get_enabled_providers(self) -> List[ProviderConfig]:
        """Get list of enabled providers sorted by priority"""
        enabled = [p for p in self.providers if p.enabled]
        return sorted(enabled, key=lambda x: x.priority)
    
    def validate(self) -> List[str]:
        """Validate configuration and return list of issues"""
        issues = []
        
        # Check for at least one enabled provider
        if not self.get_enabled_providers():
            issues.append("No enabled providers configured")
        
        # Check for API keys
        for provider in self.providers:
            if provider.enabled and not provider.api_key and provider.name != "local":
                issues.append(f"Provider {provider.name} is enabled but has no API key")
        
        # Check resource limits
        if self.resources.max_models_in_memory < 1:
            issues.append("max_models_in_memory must be at least 1")
        
        if self.resources.max_memory_gb < 0.5:
            issues.append("max_memory_gb should be at least 0.5 GB")
        
        # Check cache settings
        if self.cache.enabled and self.cache.max_size < 100:
            issues.append("Cache size should be at least 100 entries")
        
        return issues


def create_default_config() -> EmbeddingsConfig:
    """Create a default configuration"""
    return EmbeddingsConfig(
        providers=[
            ProviderConfig(
                name="openai",
                priority=1,
                models=["text-embedding-3-small", "text-embedding-3-large"],
                rate_limit="3000/min",
                fallback_provider="huggingface"
            ),
            ProviderConfig(
                name="huggingface",
                priority=2,
                models=["sentence-transformers/all-MiniLM-L6-v2"],
                rate_limit="1000/min"
            ),
            ProviderConfig(
                name="local",
                priority=3,
                api_url="http://localhost:8080/v1/embeddings",
                enabled=False
            )
        ]
    )


def load_config(path: Optional[str] = None) -> EmbeddingsConfig:
    """
    Load configuration from file or create default.
    
    Args:
        path: Optional path to configuration file
        
    Returns:
        EmbeddingsConfig instance
    """
    # Try loading from path
    if path and Path(path).exists():
        try:
            config = EmbeddingsConfig.from_yaml(path)
            logger.info(f"Loaded embeddings config from {path}")
            return config
        except Exception as e:
            logger.error(f"Failed to load config from {path}: {e}")
    
    # Try loading from environment variable
    env_path = os.getenv("EMBEDDINGS_CONFIG_PATH")
    if env_path and Path(env_path).exists():
        try:
            config = EmbeddingsConfig.from_yaml(env_path)
            logger.info(f"Loaded embeddings config from environment: {env_path}")
            return config
        except Exception as e:
            logger.error(f"Failed to load config from environment: {e}")
    
    # Try default locations
    default_paths = [
        "./embeddings_config.yaml",
        "./config/embeddings.yaml",
        "./Config_Files/embeddings_config.yaml"
    ]
    
    for default_path in default_paths:
        if Path(default_path).exists():
            try:
                config = EmbeddingsConfig.from_yaml(default_path)
                logger.info(f"Loaded embeddings config from {default_path}")
                return config
            except Exception as e:
                logger.error(f"Failed to load config from {default_path}: {e}")
    
    # Create default config
    logger.info("Using default embeddings configuration")
    return create_default_config()


# Example YAML configuration
EXAMPLE_CONFIG_YAML = """
# Embeddings Configuration

# Provider settings
providers:
  - name: openai
    priority: 1
    enabled: true
    models:
      - text-embedding-3-small
      - text-embedding-3-large
    max_connections: 10
    timeout_seconds: 30
    rate_limit: "3000/min"
    fallback_provider: huggingface
    
  - name: huggingface
    priority: 2
    enabled: true
    models:
      - sentence-transformers/all-MiniLM-L6-v2
      - sentence-transformers/all-mpnet-base-v2
    rate_limit: "1000/min"
    
  - name: cohere
    priority: 3
    enabled: false
    models:
      - embed-english-v3.0
    rate_limit: "1000/min"
    
  - name: local
    priority: 4
    enabled: false
    api_url: http://localhost:8080/v1/embeddings

# Cache settings
cache:
  enabled: true
  ttl_seconds: 3600
  max_size: 10000
  cleanup_interval: 300

# Resource management
resources:
  max_models_in_memory: 3
  max_memory_gb: 8.0
  model_ttl_seconds: 3600
  enable_model_warmup: false
  warmup_models: []

# Request batching
batching:
  enabled: true
  max_batch_size: 32
  batch_timeout_ms: 100
  adaptive_batching: true

# Security
security:
  enable_request_signing: true
  enable_audit_logging: true
  enable_rate_limiting: true
  max_input_length: 100000
  validate_embeddings: true

# Monitoring
monitoring:
  enable_metrics: true
  metrics_port: 9090
  enable_health_check: true
  health_check_interval: 30

# Global settings
default_provider: openai
default_model: text-embedding-3-small
chunk_size: 400
chunk_overlap: 200
"""


def create_example_config(path: str = "./embeddings_config.yaml"):
    """Create an example configuration file"""
    with open(path, 'w') as f:
        f.write(EXAMPLE_CONFIG_YAML)
    logger.info(f"Created example configuration at {path}")


# Global configuration instance
_config: Optional[EmbeddingsConfig] = None


def get_config() -> EmbeddingsConfig:
    """Get or load the global configuration"""
    global _config
    if _config is None:
        _config = load_config()
    return _config


def reload_config(path: Optional[str] = None):
    """Reload configuration from file"""
    global _config
    _config = load_config(path)
    
    # Validate configuration
    issues = _config.validate()
    if issues:
        logger.warning(f"Configuration validation issues: {issues}")
    
    return _config