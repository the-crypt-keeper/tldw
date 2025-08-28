# advanced_config.py
"""
Advanced configuration management for the RAG service.

This module provides dynamic configuration, A/B testing, feature flags,
and configuration validation for the RAG pipeline.
"""

import json
import os
import hashlib
from dataclasses import dataclass, field, asdict
from enum import Enum
from typing import Any, Dict, List, Optional, Union, Callable
from pathlib import Path
import yaml
import toml
from datetime import datetime

from loguru import logger


class ConfigFormat(Enum):
    """Supported configuration formats."""
    JSON = "json"
    YAML = "yaml"
    TOML = "toml"
    ENV = "env"


class FeatureState(Enum):
    """Feature flag states."""
    ENABLED = "enabled"
    DISABLED = "disabled"
    EXPERIMENT = "experiment"  # A/B testing


@dataclass
class ConfigProfile:
    """A configuration profile for different scenarios."""
    name: str
    description: str
    settings: Dict[str, Any]
    priority: int = 0
    conditions: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class FeatureFlag:
    """A feature flag configuration."""
    name: str
    state: FeatureState
    rollout_percentage: float = 100.0
    conditions: Dict[str, Any] = field(default_factory=dict)
    variants: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ConfigValidation:
    """Configuration validation rules."""
    field: str
    type: type
    required: bool = True
    min_value: Optional[Union[int, float]] = None
    max_value: Optional[Union[int, float]] = None
    allowed_values: Optional[List[Any]] = None
    validator: Optional[Callable] = None


class ConfigManager:
    """Manages RAG configuration with advanced features."""
    
    def __init__(
        self,
        base_config: Optional[Dict[str, Any]] = None,
        config_path: Optional[str] = None
    ):
        """
        Initialize configuration manager.
        
        Args:
            base_config: Base configuration dictionary
            config_path: Path to configuration file
        """
        self.base_config = base_config or self._get_default_config()
        self.config_path = config_path
        self.profiles: Dict[str, ConfigProfile] = {}
        self.feature_flags: Dict[str, FeatureFlag] = {}
        self.overrides: Dict[str, Any] = {}
        self.validation_rules: List[ConfigValidation] = []
        
        # Load configuration from file if provided
        if config_path:
            self.load_from_file(config_path)
        
        # Initialize validation rules
        self._init_validation_rules()
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default RAG configuration."""
        return {
            # Retrieval settings
            "retrieval": {
                "top_k": 10,
                "min_score": 0.0,
                "use_hybrid": True,
                "hybrid_alpha": 0.7,
                "sources": ["media_db", "notes"]
            },
            
            # Chunking settings
            "chunking": {
                "method": "semantic",
                "chunk_size": 500,
                "overlap": 50,
                "min_chunk_size": 100,
                "max_chunk_size": 1000
            },
            
            # Query processing
            "query": {
                "enable_expansion": True,
                "expansion_strategies": ["acronym", "semantic"],
                "enable_rewriting": True,
                "rewrite_strategies": ["synonym"],
                "enable_routing": True
            },
            
            # Caching
            "cache": {
                "enabled": True,
                "ttl": 3600,
                "max_size": 1000,
                "multi_level": True
            },
            
            # Generation
            "generation": {
                "provider": "openai",
                "model": "gpt-4",
                "temperature": 0.7,
                "max_tokens": 1024,
                "streaming": False
            },
            
            # Reranking
            "reranking": {
                "enabled": True,
                "strategy": "flashrank",
                "top_k": 10
            },
            
            # Citations
            "citations": {
                "enabled": False,
                "max_citations": 10,
                "format": "numbered"
            },
            
            # Performance
            "performance": {
                "enable_monitoring": True,
                "enable_circuit_breaker": True,
                "max_retries": 3,
                "timeout": 30
            }
        }
    
    def _init_validation_rules(self):
        """Initialize configuration validation rules."""
        self.validation_rules = [
            # Retrieval validations
            ConfigValidation(
                field="retrieval.top_k",
                type=int,
                min_value=1,
                max_value=100
            ),
            ConfigValidation(
                field="retrieval.min_score",
                type=float,
                min_value=0.0,
                max_value=1.0
            ),
            ConfigValidation(
                field="retrieval.hybrid_alpha",
                type=float,
                min_value=0.0,
                max_value=1.0
            ),
            
            # Chunking validations
            ConfigValidation(
                field="chunking.chunk_size",
                type=int,
                min_value=50,
                max_value=5000
            ),
            ConfigValidation(
                field="chunking.method",
                type=str,
                allowed_values=["words", "sentences", "semantic", "fixed"]
            ),
            
            # Generation validations
            ConfigValidation(
                field="generation.temperature",
                type=float,
                min_value=0.0,
                max_value=2.0
            ),
            ConfigValidation(
                field="generation.max_tokens",
                type=int,
                min_value=1,
                max_value=8000
            ),
            
            # Cache validations
            ConfigValidation(
                field="cache.ttl",
                type=(int, float),
                min_value=0
            ),
            ConfigValidation(
                field="cache.max_size",
                type=int,
                min_value=1
            )
        ]
    
    def load_from_file(self, path: str) -> None:
        """Load configuration from file."""
        path_obj = Path(path)
        
        if not path_obj.exists():
            logger.warning(f"Configuration file not found: {path}")
            return
        
        # Determine format from extension
        extension = path_obj.suffix.lower()
        
        with open(path, 'r') as f:
            if extension == '.json':
                config = json.load(f)
            elif extension in ['.yaml', '.yml']:
                config = yaml.safe_load(f)
            elif extension == '.toml':
                config = toml.load(f)
            else:
                logger.error(f"Unsupported configuration format: {extension}")
                return
        
        # Merge with base config
        self.base_config = self._deep_merge(self.base_config, config)
        
        # Load profiles if present
        if "profiles" in config:
            for profile_data in config["profiles"]:
                profile = ConfigProfile(**profile_data)
                self.add_profile(profile)
        
        # Load feature flags if present
        if "feature_flags" in config:
            for flag_data in config["feature_flags"]:
                flag = FeatureFlag(
                    name=flag_data["name"],
                    state=FeatureState(flag_data["state"]),
                    rollout_percentage=flag_data.get("rollout_percentage", 100.0),
                    conditions=flag_data.get("conditions", {}),
                    variants=flag_data.get("variants", {})
                )
                self.add_feature_flag(flag)
        
        logger.info(f"Loaded configuration from {path}")
    
    def save_to_file(self, path: str, format: ConfigFormat = ConfigFormat.JSON) -> None:
        """Save configuration to file."""
        config = self.get_config()
        
        # Add profiles and feature flags
        config["profiles"] = [asdict(p) for p in self.profiles.values()]
        config["feature_flags"] = [
            {
                "name": f.name,
                "state": f.state.value,
                "rollout_percentage": f.rollout_percentage,
                "conditions": f.conditions,
                "variants": f.variants
            }
            for f in self.feature_flags.values()
        ]
        
        with open(path, 'w') as f:
            if format == ConfigFormat.JSON:
                json.dump(config, f, indent=2)
            elif format == ConfigFormat.YAML:
                yaml.dump(config, f, default_flow_style=False)
            elif format == ConfigFormat.TOML:
                toml.dump(config, f)
            else:
                raise ValueError(f"Unsupported format: {format}")
        
        logger.info(f"Saved configuration to {path}")
    
    def add_profile(self, profile: ConfigProfile) -> None:
        """Add a configuration profile."""
        self.profiles[profile.name] = profile
        logger.debug(f"Added profile: {profile.name}")
    
    def add_feature_flag(self, flag: FeatureFlag) -> None:
        """Add a feature flag."""
        self.feature_flags[flag.name] = flag
        logger.debug(f"Added feature flag: {flag.name}")
    
    def set_override(self, key: str, value: Any) -> None:
        """Set a configuration override."""
        self.overrides[key] = value
        logger.debug(f"Set override: {key} = {value}")
    
    def get_config(
        self,
        profile: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Get configuration with profile and overrides applied.
        
        Args:
            profile: Profile name to apply
            context: Context for conditional configuration
            
        Returns:
            Merged configuration
        """
        # Start with base config
        config = self.base_config.copy()
        
        # Apply profile if specified
        if profile and profile in self.profiles:
            profile_obj = self.profiles[profile]
            config = self._deep_merge(config, profile_obj.settings)
        
        # Apply conditional profiles based on context
        if context:
            for profile_obj in sorted(self.profiles.values(), key=lambda p: p.priority):
                if self._check_conditions(profile_obj.conditions, context):
                    config = self._deep_merge(config, profile_obj.settings)
        
        # Apply overrides
        for key, value in self.overrides.items():
            self._set_nested(config, key, value)
        
        # Apply feature flags
        config = self._apply_feature_flags(config, context)
        
        # Validate configuration
        self._validate_config(config)
        
        return config
    
    def _apply_feature_flags(
        self,
        config: Dict[str, Any],
        context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Apply feature flags to configuration."""
        for flag_name, flag in self.feature_flags.items():
            if flag.state == FeatureState.DISABLED:
                continue
            
            if flag.state == FeatureState.EXPERIMENT:
                # A/B testing logic
                if context and "user_id" in context:
                    # Deterministic assignment based on user ID
                    user_hash = int(hashlib.md5(
                        str(context["user_id"]).encode()
                    ).hexdigest(), 16)
                    
                    if (user_hash % 100) < flag.rollout_percentage:
                        # User is in experiment group
                        variant = self._select_variant(flag, user_hash)
                        if variant and variant in flag.variants:
                            config = self._deep_merge(config, flag.variants[variant])
            
            elif flag.state == FeatureState.ENABLED:
                # Check conditions
                if not flag.conditions or (context and self._check_conditions(flag.conditions, context)):
                    # Apply default variant if exists
                    if "default" in flag.variants:
                        config = self._deep_merge(config, flag.variants["default"])
        
        return config
    
    def _select_variant(self, flag: FeatureFlag, user_hash: int) -> Optional[str]:
        """Select variant for A/B testing."""
        if not flag.variants:
            return None
        
        # Simple variant selection based on hash
        variants = list(flag.variants.keys())
        variant_index = user_hash % len(variants)
        return variants[variant_index]
    
    def _check_conditions(
        self,
        conditions: Dict[str, Any],
        context: Dict[str, Any]
    ) -> bool:
        """Check if conditions are met."""
        for key, expected_value in conditions.items():
            if key not in context:
                return False
            
            actual_value = context[key]
            
            # Handle different condition types
            if isinstance(expected_value, dict):
                # Complex condition
                if "$gt" in expected_value and actual_value <= expected_value["$gt"]:
                    return False
                if "$lt" in expected_value and actual_value >= expected_value["$lt"]:
                    return False
                if "$in" in expected_value and actual_value not in expected_value["$in"]:
                    return False
                if "$regex" in expected_value:
                    import re
                    if not re.match(expected_value["$regex"], str(actual_value)):
                        return False
            else:
                # Simple equality check
                if actual_value != expected_value:
                    return False
        
        return True
    
    def _validate_config(self, config: Dict[str, Any]) -> None:
        """Validate configuration against rules."""
        for rule in self.validation_rules:
            value = self._get_nested(config, rule.field)
            
            if value is None:
                if rule.required:
                    raise ValueError(f"Required field missing: {rule.field}")
                continue
            
            # Type check
            if not isinstance(value, rule.type):
                raise TypeError(
                    f"Field {rule.field} must be {rule.type}, got {type(value)}"
                )
            
            # Range checks
            if rule.min_value is not None and value < rule.min_value:
                raise ValueError(
                    f"Field {rule.field} must be >= {rule.min_value}, got {value}"
                )
            
            if rule.max_value is not None and value > rule.max_value:
                raise ValueError(
                    f"Field {rule.field} must be <= {rule.max_value}, got {value}"
                )
            
            # Allowed values check
            if rule.allowed_values is not None and value not in rule.allowed_values:
                raise ValueError(
                    f"Field {rule.field} must be one of {rule.allowed_values}, got {value}"
                )
            
            # Custom validator
            if rule.validator and not rule.validator(value):
                raise ValueError(f"Field {rule.field} failed custom validation")
    
    def _deep_merge(self, base: Dict, overlay: Dict) -> Dict:
        """Deep merge two dictionaries."""
        result = base.copy()
        
        for key, value in overlay.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = self._deep_merge(result[key], value)
            else:
                result[key] = value
        
        return result
    
    def _get_nested(self, data: Dict, path: str) -> Any:
        """Get nested value from dictionary using dot notation."""
        keys = path.split('.')
        value = data
        
        for key in keys:
            if isinstance(value, dict) and key in value:
                value = value[key]
            else:
                return None
        
        return value
    
    def _set_nested(self, data: Dict, path: str, value: Any) -> None:
        """Set nested value in dictionary using dot notation."""
        keys = path.split('.')
        
        # Navigate to the parent
        current = data
        for key in keys[:-1]:
            if key not in current:
                current[key] = {}
            current = current[key]
        
        # Set the value
        current[keys[-1]] = value


class ConfigExporter:
    """Export configuration in various formats."""
    
    @staticmethod
    def export_env_vars(config: Dict[str, Any], prefix: str = "RAG_") -> str:
        """Export configuration as environment variables."""
        lines = []
        
        def flatten(data: Dict, parent_key: str = ""):
            for key, value in data.items():
                full_key = f"{parent_key}_{key}".upper() if parent_key else key.upper()
                
                if isinstance(value, dict):
                    flatten(value, full_key)
                elif isinstance(value, list):
                    lines.append(f"{prefix}{full_key}={json.dumps(value)}")
                elif isinstance(value, bool):
                    lines.append(f"{prefix}{full_key}={'true' if value else 'false'}")
                else:
                    lines.append(f"{prefix}{full_key}={value}")
        
        flatten(config)
        return "\n".join(lines)
    
    @staticmethod
    def export_markdown(config: Dict[str, Any]) -> str:
        """Export configuration as markdown documentation."""
        lines = ["# RAG Configuration\n"]
        
        def format_section(data: Dict, level: int = 2):
            for key, value in data.items():
                header = "#" * level
                lines.append(f"\n{header} {key.replace('_', ' ').title()}\n")
                
                if isinstance(value, dict):
                    format_section(value, level + 1)
                elif isinstance(value, list):
                    lines.append("- " + "\n- ".join(str(v) for v in value))
                else:
                    lines.append(f"`{value}`")
        
        format_section(config)
        return "\n".join(lines)


# Pipeline integration functions

async def load_dynamic_config(context: Any, **kwargs) -> Any:
    """Load dynamic configuration for pipeline."""
    config_manager = ConfigManager()
    
    # Load from file if specified
    config_path = kwargs.get("config_path")
    if config_path:
        config_manager.load_from_file(config_path)
    
    # Get context for conditional configuration
    config_context = {
        "query_length": len(context.query),
        "user_id": context.metadata.get("user_id"),
        "source": context.metadata.get("source"),
        "timestamp": datetime.now().isoformat()
    }
    
    # Get profile from context or kwargs
    profile = kwargs.get("profile") or context.metadata.get("config_profile")
    
    # Get merged configuration
    config = config_manager.get_config(profile=profile, context=config_context)
    
    # Apply to context
    context.config = config
    context.metadata["config_loaded"] = True
    context.metadata["config_profile"] = profile
    
    logger.info(f"Loaded dynamic configuration{f' with profile {profile}' if profile else ''}")
    
    return context


async def apply_feature_flags(context: Any, **kwargs) -> Any:
    """Apply feature flags to pipeline context."""
    if not hasattr(context, "config"):
        return context
    
    # Example feature flags application
    flags = context.config.get("feature_flags", {})
    
    for flag_name, flag_config in flags.items():
        if flag_config.get("enabled"):
            # Apply feature flag effects
            if flag_name == "enhanced_retrieval":
                context.config["retrieval"]["use_hybrid"] = True
                context.config["retrieval"]["top_k"] = 20
            elif flag_name == "streaming_generation":
                context.config["generation"]["streaming"] = True
            elif flag_name == "aggressive_caching":
                context.config["cache"]["ttl"] = 7200
                context.config["cache"]["max_size"] = 5000
    
    context.metadata["feature_flags_applied"] = list(flags.keys())
    
    return context