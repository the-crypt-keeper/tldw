# tts_service_v2.py
# Description: Enhanced TTS service using the adapter pattern
#
# Imports
import asyncio
from typing import AsyncGenerator, Optional, Dict, Any, List
#
# Third-party Imports
from loguru import logger
#
# Local Imports
from tldw_Server_API.app.api.v1.schemas.audio_schemas import OpenAISpeechRequest
from .adapter_registry import (
    get_tts_factory,
    close_tts_factory,
    TTSAdapterFactory,
    TTSProvider
)
from .adapters.base import (
    TTSAdapter,
    TTSRequest,
    TTSResponse,
    AudioFormat
)
#
#######################################################################################################################
#
# Enhanced TTS Service with Adapter Pattern

class TTSServiceV2:
    """
    Enhanced TTS service that uses the adapter pattern for multiple providers.
    Provides intelligent provider selection and fallback capabilities.
    """
    
    def __init__(self, factory: TTSAdapterFactory):
        """
        Initialize the TTS service.
        
        Args:
            factory: TTS adapter factory instance
        """
        self.factory = factory
        self._semaphore = asyncio.Semaphore(4)  # Limit concurrent generations
    
    async def generate_speech(
        self,
        request: OpenAISpeechRequest,
        provider: Optional[str] = None,
        fallback: bool = True
    ) -> AsyncGenerator[bytes, None]:
        """
        Generate speech from text using the best available provider.
        
        Args:
            request: OpenAI-compatible speech request
            provider: Optional specific provider to use
            fallback: Whether to fallback to other providers on failure
            
        Yields:
            Audio chunks in the requested format
        """
        # Convert OpenAI request to unified TTSRequest
        tts_request = self._convert_request(request)
        
        # Get adapter
        adapter = await self._get_adapter(request.model, provider)
        
        if not adapter:
            if fallback:
                # Try to find any available adapter
                adapter = await self._get_fallback_adapter(tts_request)
            
            if not adapter:
                error_msg = f"No TTS adapter available for model '{request.model}'"
                logger.error(error_msg)
                yield error_msg.encode()
                return
        
        # Generate speech with comprehensive error handling
        try:
            async with self._semaphore:
                logger.info(f"Generating speech with {adapter.provider_name}")
                
                # Generate response
                response = await adapter.generate(tts_request)
                
                # Stream audio
                if response.audio_stream:
                    async for chunk in response.audio_stream:
                        yield chunk
                elif response.audio_data:
                    # Non-streaming response
                    yield response.audio_data
                else:
                    error_msg = f"No audio data returned by {adapter.provider_name}"
                    logger.error(error_msg)
                    if fallback:
                        await self._handle_provider_fallback(tts_request, adapter.provider_name, error_msg)
                        async for chunk in self._try_fallback_providers(tts_request, [adapter.provider_name]):
                            yield chunk
                    else:
                        yield f"ERROR: {error_msg}".encode()
                    
        except Exception as e:
            error_msg = f"Error generating speech with {adapter.provider_name}: {str(e)}"
            logger.error(error_msg, exc_info=True)
            
            # Categorize error type for better handling
            error_type = self._categorize_error(e)
            logger.debug(f"Error categorized as: {error_type}")
            
            if fallback and error_type in ['network', 'api_limit', 'provider_error']:
                # Try fallback for recoverable errors
                logger.info(f"Attempting fallback due to {error_type} error...")
                async for chunk in self._try_fallback_providers(tts_request, [adapter.provider_name]):
                    yield chunk
            else:
                # For non-recoverable errors or when fallback is disabled
                yield f"ERROR: {error_msg}".encode()
                if not fallback:
                    raise
    
    async def _generate_with_adapter(
        self,
        adapter: TTSAdapter,
        request: TTSRequest
    ) -> AsyncGenerator[bytes, None]:
        """Generate audio with a specific adapter"""
        try:
            response = await adapter.generate(request)
            
            if response.audio_stream:
                async for chunk in response.audio_stream:
                    yield chunk
            elif response.audio_data:
                yield response.audio_data
                
        except Exception as e:
            logger.error(f"Fallback generation failed: {e}")
            yield f"ERROR: All providers failed - {str(e)}".encode()
    
    def _convert_request(self, request: OpenAISpeechRequest) -> TTSRequest:
        """Convert OpenAI request to unified TTS request"""
        # Map format
        format_mapping = {
            "mp3": AudioFormat.MP3,
            "opus": AudioFormat.OPUS,
            "aac": AudioFormat.AAC,
            "flac": AudioFormat.FLAC,
            "wav": AudioFormat.WAV,
            "pcm": AudioFormat.PCM
        }
        
        audio_format = format_mapping.get(
            request.response_format.lower(),
            AudioFormat.MP3
        )
        
        return TTSRequest(
            text=request.input,
            voice=request.voice,
            format=audio_format,
            speed=request.speed,
            stream=request.stream if hasattr(request, 'stream') else True,
            # Additional parameters can be added via extra_params
            extra_params={}
        )
    
    async def _get_adapter(
        self,
        model: str,
        provider: Optional[str] = None
    ) -> Optional[TTSAdapter]:
        """Get appropriate adapter for the request"""
        if provider:
            # Specific provider requested
            try:
                provider_enum = TTSProvider(provider.lower())
                return await self.factory.registry.get_adapter(provider_enum)
            except ValueError:
                logger.warning(f"Unknown provider: {provider}")
        
        # Get adapter by model name
        return await self.factory.get_adapter_by_model(model)
    
    async def _get_fallback_adapter(
        self,
        request: TTSRequest,
        exclude: Optional[List[str]] = None
    ) -> Optional[TTSAdapter]:
        """Get a fallback adapter that can handle the request"""
        exclude = exclude or []
        
        # Find adapter that supports the requirements
        adapter = await self.factory.get_best_adapter(
            language=request.language,
            format=request.format,
            supports_streaming=request.stream
        )
        
        # Check if adapter is not in exclude list
        if adapter and adapter.provider_name not in exclude:
            return adapter
        
        # Try any available adapter
        for provider in TTSProvider:
            if provider.value in exclude:
                continue
            
            adapter = await self.factory.registry.get_adapter(provider)
            if adapter:
                # Validate if it can handle the request
                is_valid, _ = await adapter.validate_request(request)
                if is_valid:
                    return adapter
        
        return None
    
    def _categorize_error(self, error: Exception) -> str:
        """
        Categorize error types for better error handling decisions.
        
        Args:
            error: Exception that occurred
            
        Returns:
            Error category string
        """
        error_str = str(error).lower()
        error_type = type(error).__name__.lower()
        
        # Network-related errors
        if any(keyword in error_str for keyword in ['connection', 'timeout', 'network', 'unreachable']):
            return 'network'
        
        # API rate limiting or quota errors
        if any(keyword in error_str for keyword in ['rate limit', 'quota', 'too many requests', '429']):
            return 'api_limit'
        
        # Authentication/authorization errors
        if any(keyword in error_str for keyword in ['unauthorized', 'api key', 'authentication', '401', '403']):
            return 'auth'
        
        # Provider-specific errors (likely recoverable with fallback)
        if any(keyword in error_str for keyword in ['provider', 'model', 'service unavailable', '503', '502']):
            return 'provider_error'
        
        # Input validation errors (not recoverable with different provider)
        if any(keyword in error_str for keyword in ['invalid input', 'text too long', 'unsupported format']):
            return 'input_error'
        
        # Configuration errors
        if any(keyword in error_str for keyword in ['not configured', 'missing config', 'initialization']):
            return 'config_error'
        
        # Default to unknown
        return 'unknown'
    
    async def _handle_provider_fallback(self, request: TTSRequest, failed_provider: str, error_msg: str):
        """
        Handle provider fallback logging and potential circuit breaker logic.
        
        Args:
            request: Original TTS request
            failed_provider: Name of the provider that failed
            error_msg: Error message from the failed provider
        """
        logger.warning(f"Provider {failed_provider} failed: {error_msg}")
        logger.info(f"Attempting fallback for request: text_length={len(request.text)}, voice={request.voice}")
        
        # TODO: Implement circuit breaker pattern here
        # - Track failure rates per provider
        # - Temporarily disable providers with high failure rates
        # - Implement exponential backoff
    
    async def _try_fallback_providers(
        self, 
        request: TTSRequest, 
        exclude_providers: List[str]
    ) -> AsyncGenerator[bytes, None]:
        """
        Try fallback providers in priority order.
        
        Args:
            request: TTS request to fulfill
            exclude_providers: List of provider names to exclude
            
        Yields:
            Audio chunks from successful provider
        """
        fallback_adapter = await self._get_fallback_adapter(request, exclude_providers)
        
        if fallback_adapter:
            try:
                async for chunk in self._generate_with_adapter(fallback_adapter, request):
                    yield chunk
                logger.info(f"Successfully fell back to {fallback_adapter.provider_name}")
            except Exception as e:
                logger.error(f"Fallback provider {fallback_adapter.provider_name} also failed: {e}")
                # Try one more fallback if available
                exclude_providers.append(fallback_adapter.provider_name)
                final_fallback = await self._get_fallback_adapter(request, exclude_providers)
                
                if final_fallback:
                    try:
                        async for chunk in self._generate_with_adapter(final_fallback, request):
                            yield chunk
                        logger.info(f"Final fallback to {final_fallback.provider_name} succeeded")
                    except Exception as final_e:
                        error_msg = f"All providers failed. Last error: {str(final_e)}"
                        logger.error(error_msg)
                        yield f"ERROR: {error_msg}".encode()
                else:
                    yield f"ERROR: All fallback providers exhausted".encode()
        else:
            yield f"ERROR: No fallback providers available".encode()
    
    async def list_voices(self) -> Dict[str, List[Dict[str, Any]]]:
        """
        List all available voices from all providers.
        
        Returns:
            Dictionary mapping provider names to voice lists
        """
        voices = {}
        
        for provider in TTSProvider:
            adapter = await self.factory.registry.get_adapter(provider)
            if adapter and adapter.capabilities:
                provider_voices = []
                for voice in adapter.capabilities.supported_voices:
                    provider_voices.append({
                        "id": voice.id,
                        "name": voice.name,
                        "gender": voice.gender,
                        "language": voice.language,
                        "description": voice.description
                    })
                voices[provider.value] = provider_voices
        
        return voices
    
    async def get_capabilities(self) -> Dict[str, Any]:
        """
        Get capabilities of all available providers.
        
        Returns:
            Dictionary with capability information
        """
        capabilities = await self.factory.registry.get_all_capabilities()
        
        result = {}
        for provider, caps in capabilities.items():
            result[provider.value] = {
                "languages": list(caps.supported_languages),
                "formats": [fmt.value for fmt in caps.supported_formats],
                "max_text_length": caps.max_text_length,
                "supports_streaming": caps.supports_streaming,
                "supports_voice_cloning": caps.supports_voice_cloning,
                "supports_emotion_control": caps.supports_emotion_control,
                "supports_multi_speaker": caps.supports_multi_speaker,
                "latency_ms": caps.latency_ms,
                "sample_rate": caps.sample_rate
            }
        
        return result
    
    def get_status(self) -> Dict[str, Any]:
        """Get service status"""
        return self.factory.get_status()


# Singleton management
_service_instance: Optional[TTSServiceV2] = None
_service_lock = asyncio.Lock()


async def get_tts_service_v2(config: Optional[Dict[str, Any]] = None) -> TTSServiceV2:
    """
    Get or create the enhanced TTS service singleton.
    
    Args:
        config: Configuration for the service
        
    Returns:
        TTSServiceV2 instance
    """
    global _service_instance
    
    if _service_instance is None:
        async with _service_lock:
            if _service_instance is None:
                # Load configuration if not provided
                if config is None:
                    from tldw_Server_API.app.core.config import load_comprehensive_config_with_tts
                    config_obj = load_comprehensive_config_with_tts()
                    config = config_obj.get_tts_config()
                
                # Get factory
                factory = await get_tts_factory(config)
                
                # Create service
                _service_instance = TTSServiceV2(factory)
                logger.info("Enhanced TTS Service (V2) initialized")
    
    return _service_instance


async def close_tts_service_v2():
    """Close the enhanced TTS service"""
    global _service_instance
    
    if _service_instance:
        await close_tts_factory()
        _service_instance = None
        logger.info("Enhanced TTS Service (V2) closed")


# Backwards compatibility wrapper
class TTSServiceAdapter:
    """
    Adapter to make V2 service compatible with existing code.
    Maps old interface to new service.
    """
    
    def __init__(self):
        self.service_v2: Optional[TTSServiceV2] = None
    
    async def generate_audio_stream(
        self,
        request: OpenAISpeechRequest,
        internal_model_id: str
    ) -> AsyncGenerator[bytes, None]:
        """
        Generate audio stream (backwards compatible interface).
        
        Args:
            request: OpenAI speech request
            internal_model_id: Internal model identifier
            
        Yields:
            Audio chunks
        """
        # Get V2 service
        if not self.service_v2:
            self.service_v2 = await get_tts_service_v2()
        
        # Map internal model ID to provider/model
        # This maintains compatibility with existing code
        model_mapping = {
            "openai_official_tts-1": "tts-1",
            "openai_official_tts-1-hd": "tts-1-hd",
            "local_kokoro_default_onnx": "kokoro",
            "elevenlabs_english_v1": "elevenlabs",  # TODO: Implement
            "alltalk_api_backend": "alltalk"  # TODO: Implement
        }
        
        # Update request model if needed
        if internal_model_id in model_mapping:
            request.model = model_mapping[internal_model_id]
        else:
            request.model = internal_model_id
        
        # Generate with V2 service
        async for chunk in self.service_v2.generate_speech(request, fallback=True):
            yield chunk


# Export the adapter for backwards compatibility
TTSService = TTSServiceAdapter

#
# End of tts_service_v2.py
#######################################################################################################################