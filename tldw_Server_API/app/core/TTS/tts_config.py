# tts_config.py
# Description: Unified TTS configuration management system
#
# Imports
import os
import yaml
import configparser
from typing import Dict, Any, Optional, List
from pathlib import Path
#
# Third-party Imports
from loguru import logger
from pydantic import BaseModel, Field, validator
#
# Local Imports
from .adapters.base import AudioFormat
#
#######################################################################################################################
#
# TTS Configuration Schema

class ProviderConfig(BaseModel):
    """Configuration for a single TTS provider"""
    enabled: bool = False
    api_key: Optional[str] = None
    base_url: Optional[str] = None
    model: Optional[str] = None
    model_path: Optional[str] = None
    device: str = "cpu"
    timeout: int = 60
    max_retries: int = 3
    sample_rate: int = 24000
    use_fp16: bool = False
    use_bf16: bool = False
    use_onnx: bool = False
    batch_size: int = 1
    extra_params: Dict[str, Any] = Field(default_factory=dict)
    
    @validator('api_key', pre=True)
    def resolve_api_key(cls, v):
        """Resolve API key from environment variables"""
        if v and v.startswith('${') and v.endswith('}'):
            env_var = v[2:-1]
            return os.getenv(env_var)
        return v


class VoiceMappingConfig(BaseModel):
    """Voice mapping configuration"""
    generic: Dict[str, Dict[str, str]] = Field(default_factory=dict)
    emotions: Dict[str, Dict[str, str]] = Field(default_factory=dict)


class PerformanceConfig(BaseModel):
    """Performance settings"""
    max_concurrent_generations: int = 4
    cache_enabled: bool = False
    cache_ttl_seconds: int = 3600
    stream_chunk_size: int = 1024
    memory_warning_threshold: int = 80
    memory_critical_threshold: int = 90
    max_connections_per_provider: int = 5
    connection_timeout: float = 30.0


class FallbackConfig(BaseModel):
    """Fallback settings"""
    enabled: bool = True
    max_attempts: int = 3
    retry_delay_ms: int = 1000
    exclude_providers: List[str] = Field(default_factory=list)


class LoggingConfig(BaseModel):
    """Logging settings"""
    level: str = "INFO"
    log_requests: bool = True
    log_responses: bool = False
    log_performance_metrics: bool = True


class TTSConfig(BaseModel):
    """Complete TTS configuration"""
    provider_priority: List[str] = Field(default_factory=lambda: ["openai", "kokoro"])
    providers: Dict[str, ProviderConfig] = Field(default_factory=dict)
    voice_mappings: VoiceMappingConfig = Field(default_factory=VoiceMappingConfig)
    format_preferences: Dict[str, List[str]] = Field(default_factory=dict)
    performance: PerformanceConfig = Field(default_factory=PerformanceConfig)
    fallback: FallbackConfig = Field(default_factory=FallbackConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)
    
    # Settings from config.txt
    default_provider: Optional[str] = None
    default_voice: Optional[str] = None
    default_speed: float = 1.0
    local_device: str = "cpu"


class TTSConfigManager:
    """
    Manages TTS configuration from multiple sources.
    Priority order: Environment variables > config.txt > YAML > defaults
    """
    
    def __init__(self, 
                 yaml_path: Optional[Path] = None,
                 config_txt_path: Optional[Path] = None):
        """
        Initialize configuration manager.
        
        Args:
            yaml_path: Path to YAML configuration file
            config_txt_path: Path to config.txt file
        """
        self.yaml_path = yaml_path or self._find_yaml_config()
        self.config_txt_path = config_txt_path or self._find_config_txt()
        self._config: Optional[TTSConfig] = None
        self._env_overrides: Dict[str, Any] = {}
        
        # Load configurations
        self.reload()
    
    def _find_yaml_config(self) -> Optional[Path]:
        """Find YAML configuration file"""
        search_paths = [
            Path(__file__).parent / "tts_providers_config.yaml",
            Path.cwd() / "tts_providers_config.yaml",
            Path.home() / ".config" / "tldw" / "tts_providers_config.yaml"
        ]
        
        for path in search_paths:
            if path.exists():
                logger.info(f"Found TTS YAML config at: {path}")
                return path
        
        logger.warning("No TTS YAML configuration file found")
        return None
    
    def _find_config_txt(self) -> Optional[Path]:
        """Find config.txt file"""
        search_paths = [
            Path.cwd() / "Config_Files" / "config.txt",
            Path(__file__).parent.parent.parent.parent / "Config_Files" / "config.txt",
            Path.home() / ".config" / "tldw" / "config.txt"
        ]
        
        for path in search_paths:
            if path.exists():
                logger.info(f"Found config.txt at: {path}")
                return path
        
        logger.warning("No config.txt file found")
        return None
    
    def reload(self):
        """Reload configuration from all sources"""
        config_dict = {}
        
        # 1. Load YAML configuration
        if self.yaml_path and self.yaml_path.exists():
            config_dict.update(self._load_yaml())
        
        # 2. Load config.txt settings
        if self.config_txt_path and self.config_txt_path.exists():
            config_dict.update(self._load_config_txt())
        
        # 3. Apply environment variable overrides
        config_dict.update(self._load_env_overrides())
        
        # 4. Create configuration object
        self._config = TTSConfig(**config_dict)
        
        logger.info(f"TTS configuration loaded with {len(self._config.providers)} providers")
    
    def _load_yaml(self) -> Dict[str, Any]:
        """Load YAML configuration"""
        try:
            with open(self.yaml_path, 'r') as f:
                yaml_config = yaml.safe_load(f)
            
            # Convert provider configs to ProviderConfig objects
            if 'providers' in yaml_config:
                for provider_name, provider_cfg in yaml_config['providers'].items():
                    if isinstance(provider_cfg, dict):
                        yaml_config['providers'][provider_name] = ProviderConfig(**provider_cfg)
            
            return yaml_config
        except Exception as e:
            logger.error(f"Error loading YAML config: {e}")
            return {}
    
    def _load_config_txt(self) -> Dict[str, Any]:
        """Load settings from config.txt"""
        config_dict = {}
        
        try:
            config = configparser.ConfigParser()
            config.read(self.config_txt_path)
            
            # Check for TTS-Settings section
            if 'TTS-Settings' in config:
                tts_section = config['TTS-Settings']
                
                # Map config.txt settings to our schema
                if 'default_tts_provider' in tts_section:
                    config_dict['default_provider'] = tts_section['default_tts_provider']
                
                if 'default_tts_voice' in tts_section:
                    config_dict['default_voice'] = tts_section['default_tts_voice']
                
                if 'default_tts_speed' in tts_section:
                    try:
                        config_dict['default_speed'] = float(tts_section['default_tts_speed'])
                    except ValueError:
                        pass
                
                if 'local_tts_device' in tts_section:
                    config_dict['local_device'] = tts_section['local_tts_device']
                    
                    # Apply device setting to local providers
                    if 'providers' not in config_dict:
                        config_dict['providers'] = {}
                    
                    for provider in ['kokoro', 'higgs', 'dia', 'chatterbox', 'vibevoice']:
                        if provider not in config_dict['providers']:
                            config_dict['providers'][provider] = {}
                        config_dict['providers'][provider]['device'] = tts_section['local_tts_device']
            
            # Check for API keys in main section
            for key in config:
                if key != 'TTS-Settings':
                    section = config[key]
                    # Look for API keys
                    if 'openai_api_key' in section:
                        if 'providers' not in config_dict:
                            config_dict['providers'] = {}
                        if 'openai' not in config_dict['providers']:
                            config_dict['providers']['openai'] = {}
                        config_dict['providers']['openai']['api_key'] = section['openai_api_key']
                    
                    if 'elevenlabs_api_key' in section:
                        if 'providers' not in config_dict:
                            config_dict['providers'] = {}
                        if 'elevenlabs' not in config_dict['providers']:
                            config_dict['providers']['elevenlabs'] = {}
                        config_dict['providers']['elevenlabs']['api_key'] = section['elevenlabs_api_key']
            
            return config_dict
            
        except Exception as e:
            logger.error(f"Error loading config.txt: {e}")
            return {}
    
    def _load_env_overrides(self) -> Dict[str, Any]:
        """Load environment variable overrides"""
        config_dict = {}
        
        # Check for provider API keys
        api_key_env_vars = {
            'OPENAI_API_KEY': ('openai', 'api_key'),
            'ELEVENLABS_API_KEY': ('elevenlabs', 'api_key'),
            'ANTHROPIC_API_KEY': ('anthropic', 'api_key'),
        }
        
        for env_var, (provider, field) in api_key_env_vars.items():
            value = os.getenv(env_var)
            if value:
                if 'providers' not in config_dict:
                    config_dict['providers'] = {}
                if provider not in config_dict['providers']:
                    config_dict['providers'][provider] = {}
                config_dict['providers'][provider][field] = value
        
        # Check for default settings
        if os.getenv('TTS_DEFAULT_PROVIDER'):
            config_dict['default_provider'] = os.getenv('TTS_DEFAULT_PROVIDER')
        
        if os.getenv('TTS_DEFAULT_VOICE'):
            config_dict['default_voice'] = os.getenv('TTS_DEFAULT_VOICE')
        
        if os.getenv('TTS_DEVICE'):
            config_dict['local_device'] = os.getenv('TTS_DEVICE')
        
        return config_dict
    
    def get_config(self) -> TTSConfig:
        """Get the current configuration"""
        if self._config is None:
            self.reload()
        return self._config
    
    def get_provider_config(self, provider: str) -> Optional[ProviderConfig]:
        """Get configuration for a specific provider"""
        config = self.get_config()
        return config.providers.get(provider)
    
    def is_provider_enabled(self, provider: str) -> bool:
        """Check if a provider is enabled"""
        provider_config = self.get_provider_config(provider)
        return provider_config.enabled if provider_config else False
    
    def get_enabled_providers(self) -> List[str]:
        """Get list of enabled providers"""
        config = self.get_config()
        return [
            name for name, cfg in config.providers.items()
            if cfg.enabled
        ]
    
    def get_provider_priority(self) -> List[str]:
        """Get provider priority order"""
        config = self.get_config()
        # Filter to only enabled providers
        enabled = self.get_enabled_providers()
        return [p for p in config.provider_priority if p in enabled]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary"""
        return self.get_config().dict()
    
    def save_yaml(self, path: Optional[Path] = None):
        """Save current configuration to YAML file"""
        path = path or self.yaml_path
        if not path:
            raise ValueError("No YAML path specified")
        
        config_dict = self.to_dict()
        
        # Convert ProviderConfig objects to dicts
        if 'providers' in config_dict:
            for provider_name in config_dict['providers']:
                if isinstance(config_dict['providers'][provider_name], ProviderConfig):
                    config_dict['providers'][provider_name] = config_dict['providers'][provider_name].dict()
        
        with open(path, 'w') as f:
            yaml.dump(config_dict, f, default_flow_style=False, sort_keys=False)
        
        logger.info(f"Saved TTS configuration to {path}")


# Singleton instance
_config_manager: Optional[TTSConfigManager] = None


def get_tts_config() -> TTSConfig:
    """Get the global TTS configuration"""
    global _config_manager
    if _config_manager is None:
        _config_manager = TTSConfigManager()
    return _config_manager.get_config()


def get_tts_config_manager() -> TTSConfigManager:
    """Get the global TTS configuration manager"""
    global _config_manager
    if _config_manager is None:
        _config_manager = TTSConfigManager()
    return _config_manager


def reload_tts_config():
    """Reload TTS configuration from all sources"""
    manager = get_tts_config_manager()
    manager.reload()
    logger.info("TTS configuration reloaded")