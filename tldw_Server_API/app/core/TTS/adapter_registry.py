# adapter_registry.py
# Description: Registry and factory for TTS adapters
#
# Imports
import asyncio
from typing import Dict, List, Optional, Type, Any, Set
from enum import Enum
#
# Third-party Imports
from loguru import logger
#
# Local Imports
from .adapters.base import TTSAdapter, TTSCapabilities, ProviderStatus, AudioFormat
from .adapters.openai_adapter import OpenAIAdapter
from .adapters.kokoro_adapter import KokoroAdapter
from .adapters.higgs_adapter import HiggsAdapter
from .adapters.dia_adapter import DiaAdapter
from .adapters.chatterbox_adapter import ChatterboxAdapter
from .adapters.elevenlabs_adapter import ElevenLabsAdapter
from .adapters.vibevoice_adapter import VibeVoiceAdapter
from .tts_exceptions import (
    TTSProviderNotConfiguredError,
    TTSProviderInitializationError,
    TTSModelNotFoundError
)
from .tts_resource_manager import get_resource_manager
from .tts_config import get_tts_config_manager, TTSConfig
#
#######################################################################################################################
#
# TTS Adapter Registry and Factory

class TTSProvider(Enum):
    """Enumeration of available TTS providers"""
    OPENAI = "openai"
    KOKORO = "kokoro"
    HIGGS = "higgs"
    DIA = "dia"
    CHATTERBOX = "chatterbox"
    ELEVENLABS = "elevenlabs"
    VIBEVOICE = "vibevoice"
    # Additional providers
    ALLTALK = "alltalk"  # TODO: Implement AllTalk adapter


class TTSAdapterRegistry:
    """
    Registry for TTS adapters.
    Manages registration, initialization, and access to TTS providers.
    """
    
    # Default adapter mappings
    DEFAULT_ADAPTERS: Dict[TTSProvider, Type[TTSAdapter]] = {
        TTSProvider.OPENAI: OpenAIAdapter,
        TTSProvider.KOKORO: KokoroAdapter,
        TTSProvider.HIGGS: HiggsAdapter,
        TTSProvider.DIA: DiaAdapter,
        TTSProvider.CHATTERBOX: ChatterboxAdapter,
        TTSProvider.ELEVENLABS: ElevenLabsAdapter,
        TTSProvider.VIBEVOICE: VibeVoiceAdapter
    }
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the registry.
        
        Args:
            config: Configuration dictionary for all adapters
        """
        # Use unified configuration system
        self.config_manager = get_tts_config_manager()
        self.tts_config = self.config_manager.get_config()
        
        # Legacy config support
        self.config = config or self.tts_config.dict()
        
        self._adapters: Dict[TTSProvider, TTSAdapter] = {}
        self._adapter_classes: Dict[TTSProvider, Type[TTSAdapter]] = self.DEFAULT_ADAPTERS.copy()
        self._init_lock = asyncio.Lock()
        self._initialized_providers: Set[TTSProvider] = set()
    
    def register_adapter(self, provider: TTSProvider, adapter_class: Type[TTSAdapter]):
        """
        Register a custom adapter class for a provider.
        
        Args:
            provider: The provider enum
            adapter_class: The adapter class to register
        """
        self._adapter_classes[provider] = adapter_class
        logger.info(f"Registered adapter {adapter_class.__name__} for provider {provider.value}")
    
    async def get_adapter(self, provider: TTSProvider) -> Optional[TTSAdapter]:
        """
        Get an adapter instance for the specified provider.
        
        Args:
            provider: The TTS provider
            
        Returns:
            Initialized adapter instance or None if unavailable
            
        Raises:
            TTSProviderNotConfiguredError: If provider is not registered
        """
        if provider not in self._adapter_classes:
            error_msg = f"No adapter registered for provider {provider.value}"
            logger.error(error_msg)
            raise TTSProviderNotConfiguredError(
                error_msg,
                provider=provider.value
            )
        
        # Check if adapter already exists
        if provider in self._adapters:
            adapter = self._adapters[provider]
            if adapter.status == ProviderStatus.AVAILABLE:
                return adapter
        
        # Initialize adapter if needed
        async with self._init_lock:
            if provider not in self._adapters:
                await self._initialize_adapter(provider)
            
            adapter = self._adapters.get(provider)
            if adapter and adapter.status == ProviderStatus.AVAILABLE:
                return adapter
            else:
                logger.warning(f"Adapter for {provider.value} is not available (status: {adapter.status if adapter else 'None'})")
                return None
    
    async def _initialize_adapter(self, provider: TTSProvider) -> bool:
        """
        Initialize an adapter for a provider.
        
        Args:
            provider: The TTS provider
            
        Returns:
            True if initialization successful
            
        Raises:
            TTSProviderInitializationError: If initialization fails
        """
        try:
            # Get adapter class
            adapter_class = self._adapter_classes[provider]
            
            # Get provider-specific config
            provider_config = self._get_provider_config(provider)
            
            # Check if provider is enabled using unified config
            if not self.config_manager.is_provider_enabled(provider.value):
                logger.info(f"Provider {provider.value} is disabled in configuration")
                return False
            
            # Get resource manager for monitoring
            resource_manager = await get_resource_manager()
            
            # Check memory before initializing new adapter
            if resource_manager.memory_monitor.is_memory_critical():
                logger.warning(f"Skipping {provider.value} initialization due to memory constraints")
                return False
            
            # Create adapter instance
            logger.info(f"Initializing {provider.value} adapter...")
            adapter = adapter_class(config=provider_config)
            
            # Initialize the adapter
            success = await adapter.ensure_initialized()
            
            if success:
                self._adapters[provider] = adapter
                self._initialized_providers.add(provider)
                logger.info(f"Successfully initialized {provider.value} adapter")
                return True
            else:
                error_msg = f"Failed to initialize {provider.value} adapter"
                logger.error(error_msg)
                raise TTSProviderInitializationError(
                    error_msg,
                    provider=provider.value
                )
                
        except TTSProviderInitializationError:
            raise
        except Exception as e:
            logger.error(f"Error initializing {provider.value} adapter: {e}")
            raise TTSProviderInitializationError(
                f"Unexpected error initializing {provider.value}",
                provider=provider.value,
                details={"error": str(e), "error_type": type(e).__name__}
            )
    
    def _get_provider_config(self, provider: TTSProvider) -> Dict[str, Any]:
        """
        Get configuration for a specific provider.
        
        Args:
            provider: The TTS provider
            
        Returns:
            Provider-specific configuration dictionary
        """
        # Use unified configuration system
        provider_cfg = self.config_manager.get_provider_config(provider.value)
        
        if provider_cfg:
            # Convert to dict for adapter consumption
            return provider_cfg.dict()
        
        # Fallback to legacy config
        provider_config = self.config.copy()
        
        # Add provider-specific overrides
        provider_key = f"{provider.value}_config"
        if provider_key in self.config:
            provider_config.update(self.config[provider_key])
        
        return provider_config
    
    async def get_all_capabilities(self) -> Dict[TTSProvider, TTSCapabilities]:
        """
        Get capabilities of all available adapters.
        
        Returns:
            Dictionary mapping providers to their capabilities
        """
        capabilities = {}
        
        for provider in TTSProvider:
            adapter = await self.get_adapter(provider)
            if adapter:
                caps = adapter.capabilities
                if caps:
                    capabilities[provider] = caps
        
        return capabilities
    
    async def find_adapter_for_requirements(
        self,
        language: Optional[str] = None,
        format: Optional[AudioFormat] = None,
        supports_streaming: Optional[bool] = None,
        supports_emotion: Optional[bool] = None,
        supports_voice_cloning: Optional[bool] = None,
        supports_multi_speaker: Optional[bool] = None
    ) -> Optional[TTSAdapter]:
        """
        Find an adapter that meets specific requirements.
        
        Args:
            language: Required language support
            format: Required audio format
            supports_streaming: Requires streaming support
            supports_emotion: Requires emotion control
            supports_voice_cloning: Requires voice cloning
            supports_multi_speaker: Requires multi-speaker support
            
        Returns:
            First adapter that meets all requirements, or None
        """
        for provider in self._get_provider_priority():
            adapter = await self.get_adapter(provider)
            if not adapter or not adapter.capabilities:
                continue
            
            caps = adapter.capabilities
            
            # Check requirements
            if language and language not in caps.supported_languages:
                continue
            if format and format not in caps.supported_formats:
                continue
            if supports_streaming is not None and caps.supports_streaming != supports_streaming:
                continue
            if supports_emotion is not None and caps.supports_emotion_control != supports_emotion:
                continue
            if supports_voice_cloning is not None and caps.supports_voice_cloning != supports_voice_cloning:
                continue
            if supports_multi_speaker is not None and caps.supports_multi_speaker != supports_multi_speaker:
                continue
            
            return adapter
        
        return None
    
    def _get_provider_priority(self) -> List[TTSProvider]:
        """
        Get provider priority order.
        Can be customized via configuration.
        
        Returns:
            Ordered list of providers to try
        """
        # Use unified configuration priority
        priority_names = self.config_manager.get_provider_priority()
        
        priority = []
        for provider_name in priority_names:
            try:
                provider = TTSProvider(provider_name)
                priority.append(provider)
            except ValueError:
                logger.warning(f"Unknown provider in priority list: {provider_name}")
        
        # Fallback to default if no valid providers
        if not priority:
            priority = [
                TTSProvider.OPENAI,
                TTSProvider.KOKORO,
                TTSProvider.CHATTERBOX,
                TTSProvider.DIA,
                TTSProvider.HIGGS
            ]
        
        return priority
    
    async def close_all(self):
        """Close all initialized adapters and clean up resources"""
        logger.info("Closing all TTS adapters...")
        
        # Get resource manager for cleanup
        try:
            resource_manager = await get_resource_manager()
        except Exception as e:
            logger.warning(f"Could not get resource manager for cleanup: {e}")
            resource_manager = None
        
        tasks = []
        for provider, adapter in self._adapters.items():
            logger.info(f"Closing {provider.value} adapter...")
            tasks.append(adapter.close())
            
            # Unregister from resource manager if available
            if resource_manager:
                try:
                    await resource_manager.unregister_model(provider.value)
                except Exception as e:
                    logger.warning(f"Error unregistering {provider.value} from resource manager: {e}")
        
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
        
        self._adapters.clear()
        self._initialized_providers.clear()
        
        # Clean up resource manager connections
        if resource_manager:
            try:
                await resource_manager.cleanup_all()
            except Exception as e:
                logger.warning(f"Error during resource manager cleanup: {e}")
        
        logger.info("All TTS adapters closed")
    
    def get_status_summary(self) -> Dict[str, Any]:
        """
        Get status summary of all adapters.
        
        Returns:
            Dictionary with status information
        """
        summary = {
            "total_providers": len(TTSProvider),
            "initialized": len(self._initialized_providers),
            "available": 0,
            "providers": {}
        }
        
        for provider in TTSProvider:
            if provider in self._adapters:
                adapter = self._adapters[provider]
                status = adapter.status.value
                if adapter.status == ProviderStatus.AVAILABLE:
                    summary["available"] += 1
            else:
                status = "not_initialized"
            
            summary["providers"][provider.value] = status
        
        return summary


class TTSAdapterFactory:
    """
    Factory for creating and managing TTS adapters.
    Provides high-level interface for TTS operations.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the factory.
        
        Args:
            config: Configuration for all adapters
        """
        self.registry = TTSAdapterRegistry(config)
    
    async def get_adapter_by_model(self, model: str) -> Optional[TTSAdapter]:
        """
        Get adapter based on model name.
        Maps model names to providers.
        
        Args:
            model: Model name (e.g., "tts-1", "kokoro", "higgs")
            
        Returns:
            Appropriate adapter or None
        """
        # Model to provider mapping
        model_mapping = {
            # OpenAI models
            "tts-1": TTSProvider.OPENAI,
            "tts-1-hd": TTSProvider.OPENAI,
            
            # Kokoro models
            "kokoro": TTSProvider.KOKORO,
            "kokoro-v0_19": TTSProvider.KOKORO,
            "kokoro-onnx": TTSProvider.KOKORO,
            
            # Higgs models
            "higgs": TTSProvider.HIGGS,
            "higgs-v2": TTSProvider.HIGGS,
            "higgs-audio-v2": TTSProvider.HIGGS,
            
            # ElevenLabs models
            "elevenlabs": TTSProvider.ELEVENLABS,
            "eleven_monolingual_v1": TTSProvider.ELEVENLABS,
            "eleven_multilingual_v1": TTSProvider.ELEVENLABS,
            "eleven_multilingual_v2": TTSProvider.ELEVENLABS,
            "eleven_turbo_v2": TTSProvider.ELEVENLABS,
            
            # Dia models
            "dia": TTSProvider.DIA,
            "dia-1.6b": TTSProvider.DIA,
            
            # Chatterbox models
            "chatterbox": TTSProvider.CHATTERBOX,
            "chatterbox-emotion": TTSProvider.CHATTERBOX,
            
            # VibeVoice models
            "vibevoice": TTSProvider.VIBEVOICE,
            "vibevoice-1.5b": TTSProvider.VIBEVOICE,
            "vibevoice-7b": TTSProvider.VIBEVOICE,
            "microsoft/VibeVoice-1.5B": TTSProvider.VIBEVOICE,
            "WestZhang/VibeVoice-Large-pt": TTSProvider.VIBEVOICE
        }
        
        # Get provider from model name
        provider = model_mapping.get(model.lower())
        if not provider:
            logger.warning(f"Unknown model: {model}")
            return None
        
        return await self.registry.get_adapter(provider)
    
    async def get_best_adapter(self, **requirements) -> Optional[TTSAdapter]:
        """
        Get the best adapter for given requirements.
        
        Args:
            **requirements: Requirements for the adapter
            
        Returns:
            Best matching adapter or None
        """
        return await self.registry.find_adapter_for_requirements(**requirements)
    
    async def close(self):
        """Close all adapters"""
        await self.registry.close_all()
    
    def get_status(self) -> Dict[str, Any]:
        """Get factory status"""
        return self.registry.get_status_summary()


# Singleton instance management
_factory_instance: Optional[TTSAdapterFactory] = None
_factory_lock = asyncio.Lock()


async def get_tts_factory(config: Optional[Dict[str, Any]] = None) -> TTSAdapterFactory:
    """
    Get or create the TTS adapter factory singleton.
    
    Args:
        config: Configuration for the factory
        
    Returns:
        TTSAdapterFactory instance
    """
    global _factory_instance
    
    if _factory_instance is None:
        async with _factory_lock:
            if _factory_instance is None:
                _factory_instance = TTSAdapterFactory(config)
                logger.info("TTS Adapter Factory initialized")
    
    return _factory_instance


async def close_tts_factory():
    """Close the TTS factory and all adapters"""
    global _factory_instance
    
    if _factory_instance:
        await _factory_instance.close()
        _factory_instance = None
        logger.info("TTS Adapter Factory closed")

#
# End of adapter_registry.py
#######################################################################################################################