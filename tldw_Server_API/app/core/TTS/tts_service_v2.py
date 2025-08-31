# tts_service_v2.py
# Description: Enhanced TTS service using the adapter pattern
#
# Imports
import asyncio
import time
from typing import AsyncGenerator, Optional, Dict, Any, List
#
# Third-party Imports
from loguru import logger
#
# Local Imports
from tldw_Server_API.app.api.v1.schemas.audio_schemas import OpenAISpeechRequest
from tldw_Server_API.app.core.Metrics import get_metrics_registry
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
from .circuit_breaker import (
    get_circuit_manager,
    CircuitBreakerManager,
    CircuitOpenError
)
from .tts_exceptions import (
    TTSError,
    TTSProviderNotConfiguredError,
    TTSProviderInitializationError,
    TTSModelNotFoundError,
    TTSGenerationError,
    TTSValidationError,
    TTSAuthenticationError,
    TTSRateLimitError,
    TTSNetworkError,
    TTSTimeoutError,
    TTSProviderError,
    TTSResourceError,
    TTSInsufficientMemoryError,
    TTSGPUError,
    categorize_error,
    is_retryable_error
)
from .tts_validation import validate_tts_request
from .tts_resource_manager import get_resource_manager
#
#######################################################################################################################
#
# Enhanced TTS Service with Adapter Pattern

class TTSServiceV2:
    """
    Enhanced TTS service that uses the adapter pattern for multiple providers.
    Provides intelligent provider selection and fallback capabilities.
    """
    
    def __init__(self, factory: TTSAdapterFactory, circuit_manager: Optional[CircuitBreakerManager] = None):
        """
        Initialize the TTS service.
        
        Args:
            factory: TTS adapter factory instance
            circuit_manager: Optional circuit breaker manager
        """
        self.factory = factory
        self.circuit_manager = circuit_manager
        self._semaphore = asyncio.Semaphore(4)  # Limit concurrent generations
        
        # Initialize metrics
        self.metrics = get_metrics_registry()
        self._register_tts_metrics()
    
    def _register_tts_metrics(self):
        """Register TTS-specific metrics"""
        # TTS request metrics
        self.metrics.register_counter(
            name="tts_requests_total",
            description="Total number of TTS requests",
            labels=["provider", "model", "voice", "format", "status"]
        )
        
        self.metrics.register_histogram(
            name="tts_request_duration_seconds",
            description="TTS request duration in seconds",
            unit="s",
            labels=["provider", "model", "voice"],
            buckets=[0.1, 0.25, 0.5, 1, 2, 5, 10, 30, 60]
        )
        
        self.metrics.register_histogram(
            name="tts_text_length_characters",
            description="Length of text processed",
            unit="characters",
            labels=["provider"],
            buckets=[10, 50, 100, 250, 500, 1000, 2500, 5000]
        )
        
        self.metrics.register_histogram(
            name="tts_audio_size_bytes",
            description="Size of generated audio",
            unit="bytes",
            labels=["provider", "format"],
            buckets=[1024, 10240, 102400, 1048576, 10485760]  # 1KB, 10KB, 100KB, 1MB, 10MB
        )
        
        self.metrics.register_gauge(
            name="tts_active_requests",
            description="Number of active TTS requests",
            labels=["provider"]
        )
        
        self.metrics.register_counter(
            name="tts_fallback_attempts",
            description="Number of fallback attempts",
            labels=["from_provider", "to_provider", "success"]
        )
    
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
        
        # Validate the request first
        try:
            validate_tts_request(tts_request, provider=provider.lower() if provider else None)
        except TTSValidationError as e:
            logger.error(f"TTS request validation failed: {e}")
            yield f"ERROR: {str(e)}".encode()
            return
        
        # Get adapter
        adapter = await self._get_adapter(request.model, provider)
        
        if not adapter:
            if fallback:
                # Try to find any available adapter
                adapter = await self._get_fallback_adapter(tts_request)
            
            if not adapter:
                error = TTSProviderNotConfiguredError(
                    f"No TTS adapter available for model '{request.model}'",
                    provider=provider
                )
                logger.error(str(error))
                yield f"ERROR: {str(error)}".encode()
                return
        
        # Track metrics
        start_time = time.time()
        audio_size = 0
        chunks_count = 0
        
        # Update active requests gauge
        self.metrics.gauge_set(
            "tts_active_requests",
            1,
            labels={"provider": adapter.provider_name}
        )
        
        # Generate speech with circuit breaker and comprehensive error handling
        try:
            async with self._semaphore:
                logger.info(f"Generating speech with {adapter.provider_name}")
                
                # Get circuit breaker if available
                circuit_breaker = None
                if self.circuit_manager:
                    circuit_breaker = await self.circuit_manager.get_breaker(adapter.provider_name)
                
                # Generate response (with or without circuit breaker)
                if circuit_breaker:
                    try:
                        response = await circuit_breaker.call(adapter.generate, tts_request)
                    except CircuitOpenError as e:
                        logger.warning(f"Circuit open for {adapter.provider_name}: {e}")
                        if fallback:
                            # Record fallback attempt
                            self.metrics.counter_increment(
                                "tts_fallback_attempts",
                                labels={"from_provider": adapter.provider_name, "to_provider": "any", "success": "pending"}
                            )
                            async for chunk in self._try_fallback_providers(tts_request, [adapter.provider_name]):
                                yield chunk
                            return
                        else:
                            raise TTSProviderError(
                                f"Circuit open for {adapter.provider_name}",
                                provider=adapter.provider_name,
                                details={"circuit_state": "open"}
                            )
                else:
                    response = await adapter.generate(tts_request)
                
                # Stream audio
                if response.audio_stream:
                    async for chunk in response.audio_stream:
                        chunks_count += 1
                        audio_size += len(chunk)
                        yield chunk
                elif response.audio_data:
                    # Non-streaming response
                    chunks_count = 1
                    audio_size = len(response.audio_data)
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
                
                # Record success metrics
                self._record_tts_metrics(
                    provider=adapter.provider_name,
                    model=tts_request.model or "default",
                    voice=tts_request.voice or "default",
                    format=tts_request.format.value,
                    text_length=len(tts_request.text),
                    audio_size=audio_size,
                    duration=time.time() - start_time,
                    success=True
                )
                    
        except TTSError as e:
            # Handle TTS-specific errors with proper categorization
            error_msg = f"Error generating speech with {adapter.provider_name}: {str(e)}"
            logger.error(error_msg)
            
            # Record failure metrics
            self._record_tts_metrics(
                provider=adapter.provider_name,
                model=tts_request.model or "default", 
                voice=tts_request.voice or "default",
                format=tts_request.format.value,
                text_length=len(tts_request.text),
                audio_size=audio_size,
                duration=time.time() - start_time,
                success=False,
                error=str(e)
            )
            
            # Check if error is retryable and fallback is enabled
            if fallback and is_retryable_error(e):
                logger.info(f"Attempting fallback due to retryable error: {type(e).__name__}")
                self.metrics.counter_increment(
                    "tts_fallback_attempts",
                    labels={"from_provider": adapter.provider_name, "to_provider": "any", "success": "pending"}
                )
                async for chunk in self._try_fallback_providers(tts_request, [adapter.provider_name]):
                    yield chunk
            else:
                # For non-recoverable errors or when fallback is disabled
                yield f"ERROR: {error_msg}".encode()
                if not fallback:
                    raise
        except Exception as e:
            # Handle unexpected errors
            error_msg = f"Unexpected error generating speech with {adapter.provider_name}: {str(e)}"
            logger.error(error_msg, exc_info=True)
            
            # Record failure metrics
            self._record_tts_metrics(
                provider=adapter.provider_name,
                model=tts_request.model or "default", 
                voice=tts_request.voice or "default",
                format=tts_request.format.value,
                text_length=len(tts_request.text),
                audio_size=audio_size,
                duration=time.time() - start_time,
                success=False,
                error=str(e)
            )
            
            # Wrap in TTS error for consistency
            tts_error = TTSGenerationError(
                f"Unexpected error in {adapter.provider_name}",
                provider=adapter.provider_name,
                details={"error": str(e), "error_type": type(e).__name__}
            )
            
            if fallback:
                logger.info("Attempting fallback due to unexpected error")
                async for chunk in self._try_fallback_providers(tts_request, [adapter.provider_name]):
                    yield chunk
            else:
                yield f"ERROR: {error_msg}".encode()
                raise tts_error
        finally:
            # Update active requests gauge
            self.metrics.gauge_add(
                "tts_active_requests",
                -1,
                labels={"provider": adapter.provider_name}
            )
    
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
    
    def _record_tts_metrics(
        self,
        provider: str,
        model: str,
        voice: str,
        format: str,
        text_length: int,
        audio_size: int,
        duration: float,
        success: bool,
        error: Optional[str] = None
    ):
        """Record TTS request metrics"""
        # Record request counter
        self.metrics.counter_increment(
            "tts_requests_total",
            labels={
                "provider": provider,
                "model": model,
                "voice": voice,
                "format": format,
                "status": "success" if success else "failure"
            }
        )
        
        # Record duration histogram
        self.metrics.histogram_observe(
            "tts_request_duration_seconds",
            duration,
            labels={"provider": provider, "model": model, "voice": voice}
        )
        
        # Record text length histogram
        self.metrics.histogram_observe(
            "tts_text_length_characters",
            text_length,
            labels={"provider": provider}
        )
        
        # Record audio size if successful
        if success and audio_size > 0:
            self.metrics.histogram_observe(
                "tts_audio_size_bytes",
                audio_size,
                labels={"provider": provider, "format": format}
            )
        
        # Log performance metrics
        if success:
            chars_per_second = text_length / duration if duration > 0 else 0
            logger.info(
                f"TTS metrics: provider={provider}, duration={duration:.2f}s, "
                f"text_length={text_length}, audio_size={audio_size}, "
                f"chars/sec={chars_per_second:.1f}"
            )
        else:
            logger.warning(
                f"TTS failed: provider={provider}, duration={duration:.2f}s, "
                f"error={error}"
            )
    
    def _categorize_error(self, error: Exception) -> str:
        """
        Categorize error types for better error handling decisions.
        Uses the new exception system's categorization.
        
        Args:
            error: Exception that occurred
            
        Returns:
            Error category string
        """
        # Use the new exception system's categorization
        return categorize_error(error)
    
    async def _handle_provider_fallback(self, request: TTSRequest, failed_provider: str, error_msg: str):
        """
        Handle provider fallback logging and circuit breaker updates.
        
        Args:
            request: Original TTS request
            failed_provider: Name of the provider that failed
            error_msg: Error message from the failed provider
        """
        logger.warning(f"Provider {failed_provider} failed: {error_msg}")
        logger.info(f"Attempting fallback for request: text_length={len(request.text)}, voice={request.voice}")
        
        # Update circuit breaker state if available
        if self.circuit_manager:
            breaker = await self.circuit_manager.get_breaker(failed_provider)
            if breaker:
                # The circuit breaker will track the failure internally
                logger.debug(f"Circuit breaker updated for {failed_provider}")
    
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
                # Record successful fallback
                self.metrics.counter_increment(
                    "tts_fallback_attempts",
                    labels={
                        "from_provider": exclude_providers[0] if exclude_providers else "unknown",
                        "to_provider": fallback_adapter.provider_name,
                        "success": "true"
                    }
                )
            except TTSError as e:
                logger.error(f"Fallback provider {fallback_adapter.provider_name} also failed: {e}")
                # Record failed fallback
                self.metrics.counter_increment(
                    "tts_fallback_attempts",
                    labels={
                        "from_provider": exclude_providers[0] if exclude_providers else "unknown",
                        "to_provider": fallback_adapter.provider_name,
                        "success": "false"
                    }
                )
                
                # Try one more fallback if available and error is retryable
                if is_retryable_error(e):
                    exclude_providers.append(fallback_adapter.provider_name)
                    final_fallback = await self._get_fallback_adapter(request, exclude_providers)
                    
                    if final_fallback:
                        try:
                            async for chunk in self._generate_with_adapter(final_fallback, request):
                                yield chunk
                            logger.info(f"Final fallback to {final_fallback.provider_name} succeeded")
                            # Record successful final fallback
                            self.metrics.counter_increment(
                                "tts_fallback_attempts",
                                labels={
                                    "from_provider": fallback_adapter.provider_name,
                                    "to_provider": final_fallback.provider_name,
                                    "success": "true"
                                }
                            )
                        except Exception as final_e:
                            # Wrap non-TTS errors
                            if not isinstance(final_e, TTSError):
                                final_e = TTSGenerationError(
                                    f"Final fallback failed",
                                    provider=final_fallback.provider_name,
                                    details={"error": str(final_e)}
                                )
                            error_msg = f"All providers failed. Last error: {str(final_e)}"
                            logger.error(error_msg)
                            yield f"ERROR: {error_msg}".encode()
                    else:
                        yield f"ERROR: All fallback providers exhausted".encode()
                else:
                    # Non-retryable error, don't attempt more fallbacks
                    yield f"ERROR: {str(e)} (non-retryable)".encode()
            except Exception as e:
                # Handle unexpected errors
                logger.error(f"Unexpected error in fallback: {e}", exc_info=True)
                yield f"ERROR: Unexpected error during fallback: {str(e)}".encode()
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
        status = self.factory.get_status()
        
        # Add circuit breaker status if available
        if self.circuit_manager:
            status["circuit_breakers"] = self.circuit_manager.get_all_status()
        
        return status


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
                
                # Get circuit breaker manager
                circuit_manager = await get_circuit_manager(config)
                
                # Create service
                _service_instance = TTSServiceV2(factory, circuit_manager)
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