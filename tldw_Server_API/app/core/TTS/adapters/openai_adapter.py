# openai_adapter.py
# Description: OpenAI TTS API adapter implementation
#
# Imports
import os
from typing import Optional, Dict, Any, AsyncGenerator, Set
#
# Third-party Imports
import httpx
from loguru import logger
#
# Local Imports
from .base import (
    TTSAdapter,
    TTSCapabilities,
    TTSRequest,
    TTSResponse,
    AudioFormat,
    VoiceInfo,
    ProviderStatus
)
from ..tts_exceptions import (
    TTSProviderNotConfiguredError,
    TTSProviderInitializationError,
    TTSAuthenticationError,
    TTSRateLimitError,
    TTSNetworkError,
    TTSTimeoutError,
    TTSProviderError,
    auth_error,
    rate_limit_error,
    network_error,
    timeout_error
)
from ..tts_validation import validate_tts_request
from ..tts_resource_manager import get_resource_manager
#
#######################################################################################################################
#
# OpenAI TTS Adapter Implementation

class OpenAIAdapter(TTSAdapter):
    """Adapter for OpenAI's TTS API"""
    
    # OpenAI voice definitions
    VOICES = {
        "alloy": VoiceInfo(
            id="alloy",
            name="Alloy",
            gender="neutral",
            description="Neutral and balanced voice"
        ),
        "echo": VoiceInfo(
            id="echo",
            name="Echo",
            gender="male",
            description="Male voice with clarity"
        ),
        "fable": VoiceInfo(
            id="fable",
            name="Fable",
            gender="neutral",
            description="Expressive and dynamic voice"
        ),
        "onyx": VoiceInfo(
            id="onyx",
            name="Onyx",
            gender="male",
            description="Deep and authoritative male voice"
        ),
        "nova": VoiceInfo(
            id="nova",
            name="Nova",
            gender="female",
            description="Warm and friendly female voice"
        ),
        "shimmer": VoiceInfo(
            id="shimmer",
            name="Shimmer",
            gender="female",
            description="Soft and pleasant female voice"
        )
    }
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        self.api_key = self.config.get("openai_api_key") or os.getenv("OPENAI_API_KEY")
        self.base_url = self.config.get("openai_base_url", "https://api.openai.com/v1/audio/speech")
        self.model = self.config.get("openai_model", "tts-1")  # or "tts-1-hd"
        self.client: Optional[httpx.AsyncClient] = None
        
        if not self.api_key:
            logger.warning(f"{self.provider_name}: API key not configured")
            self._status = ProviderStatus.NOT_CONFIGURED
    
    async def initialize(self) -> bool:
        """Initialize the OpenAI adapter"""
        try:
            if not self.api_key:
                error_msg = f"{self.provider_name}: Cannot initialize without API key"
                logger.error(error_msg)
                self._status = ProviderStatus.NOT_CONFIGURED
                raise TTSProviderNotConfiguredError(error_msg, provider=self.provider_name)
            
            # Get HTTP client from resource manager
            resource_manager = await get_resource_manager()
            self.client = await resource_manager.get_http_client(
                provider=self.provider_name.lower(),
                base_url=self.base_url
            )
            
            # Test the API key with a minimal request
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            }
            
            # Quick validation - we'll do a proper test in production
            logger.info(f"{self.provider_name}: Initialized successfully")
            self._status = ProviderStatus.AVAILABLE
            return True
            
        except TTSProviderNotConfiguredError:
            return False
        except Exception as e:
            logger.error(f"{self.provider_name}: Initialization failed: {e}")
            self._status = ProviderStatus.ERROR
            raise TTSProviderInitializationError(
                f"Failed to initialize {self.provider_name}",
                provider=self.provider_name,
                details={"error": str(e)}
            )
    
    async def get_capabilities(self) -> TTSCapabilities:
        """Get OpenAI TTS capabilities"""
        return TTSCapabilities(
            provider_name="OpenAI",
            supported_languages={"en"},  # OpenAI TTS primarily supports English
            supported_voices=list(self.VOICES.values()),
            supported_formats={
                AudioFormat.MP3,
                AudioFormat.OPUS,
                AudioFormat.AAC,
                AudioFormat.FLAC,
                AudioFormat.WAV,
                AudioFormat.PCM
            },
            max_text_length=4096,
            supports_streaming=True,
            supports_voice_cloning=False,
            supports_emotion_control=False,
            supports_speech_rate=True,
            supports_pitch_control=False,
            supports_volume_control=False,
            supports_ssml=False,
            supports_phonemes=False,
            supports_multi_speaker=False,
            supports_background_audio=False,
            latency_ms=200,  # Approximate
            sample_rate=24000,
            default_format=AudioFormat.MP3
        )
    
    async def generate(self, request: TTSRequest) -> TTSResponse:
        """Generate speech using OpenAI TTS API"""
        if not await self.ensure_initialized():
            raise TTSProviderNotConfiguredError(
                f"{self.provider_name} not initialized",
                provider=self.provider_name
            )
        
        # Validate request using new validation system
        try:
            validate_tts_request(request, provider=self.provider_name.lower())
        except Exception as e:
            logger.error(f"{self.provider_name} request validation failed: {e}")
            raise
        
        # Map voice if needed
        voice = self.map_voice(request.voice or "alloy")
        
        # Prepare request payload
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        payload = {
            "model": self.model,
            "input": self.preprocess_text(request.text),
            "voice": voice,
            "response_format": request.format.value,
            "speed": request.speed
        }
        
        logger.info(
            f"{self.provider_name}: Generating speech with model={self.model}, "
            f"voice={voice}, format={request.format.value}"
        )
        
        try:
            if request.stream:
                # Return streaming response
                return TTSResponse(
                    audio_stream=self._stream_audio(headers, payload),
                    format=request.format,
                    sample_rate=24000,
                    channels=1,
                    voice_used=voice,
                    provider=self.provider_name
                )
            else:
                # Return complete audio
                audio_data = await self._generate_complete(headers, payload)
                return TTSResponse(
                    audio_data=audio_data,
                    format=request.format,
                    sample_rate=24000,
                    channels=1,
                    voice_used=voice,
                    provider=self.provider_name
                )
                
        except httpx.HTTPStatusError as e:
            error_content = await e.response.aread()
            error_msg = error_content.decode()
            logger.error(f"{self.provider_name} API error: {e.response.status_code} - {error_msg}")
            
            if e.response.status_code == 401:
                raise auth_error("Invalid OpenAI API key", self.provider_name)
            elif e.response.status_code == 429:
                # Try to extract retry-after header
                retry_after = e.response.headers.get('retry-after')
                raise rate_limit_error(
                    self.provider_name, 
                    retry_after=int(retry_after) if retry_after else None
                )
            elif e.response.status_code == 400:
                raise TTSProviderError(
                    f"Invalid request to OpenAI: {error_msg}",
                    provider=self.provider_name,
                    error_code="BAD_REQUEST"
                )
            else:
                raise TTSProviderError(
                    f"OpenAI API error: {error_msg}",
                    provider=self.provider_name,
                    error_code=str(e.response.status_code)
                )
                
        except httpx.TimeoutException as e:
            logger.error(f"{self.provider_name} timeout error: {e}")
            raise timeout_error(self.provider_name, timeout_duration=60.0)
            
        except httpx.NetworkError as e:
            logger.error(f"{self.provider_name} network error: {e}")
            raise network_error(f"Network error communicating with {self.provider_name}", self.provider_name)
                
        except Exception as e:
            if not isinstance(e, (TTSProviderError, TTSAuthenticationError, TTSRateLimitError, TTSNetworkError, TTSTimeoutError)):
                logger.error(f"{self.provider_name} unexpected error: {e}")
                raise TTSProviderError(
                    f"Unexpected error in {self.provider_name}",
                    provider=self.provider_name,
                    details={"error": str(e), "error_type": type(e).__name__}
                )
            raise
    
    async def _stream_audio(
        self,
        headers: Dict[str, str],
        payload: Dict[str, Any]
    ) -> AsyncGenerator[bytes, None]:
        """Stream audio from OpenAI API"""
        try:
            async with self.client.stream("POST", self.base_url, headers=headers, json=payload) as response:
                response.raise_for_status()
                total_bytes = 0
                async for chunk in response.aiter_bytes(chunk_size=1024):
                    total_bytes += len(chunk)
                    yield chunk
                logger.debug(f"{self.provider_name}: Streamed {total_bytes} bytes")
        except Exception as e:
            logger.error(f"{self.provider_name} streaming error: {e}")
            raise
    
    async def _generate_complete(
        self,
        headers: Dict[str, str],
        payload: Dict[str, Any]
    ) -> bytes:
        """Generate complete audio from OpenAI API"""
        response = await self.client.post(self.base_url, headers=headers, json=payload)
        response.raise_for_status()
        return response.content
    
    async def _cleanup_resources(self):
        """Clean up OpenAI adapter resources"""
        # Note: HTTP clients are now managed by the resource manager
        # No need to manually close them as they use connection pooling
        try:
            # Clear reference to client
            self.client = None
            logger.debug(f"{self.provider_name}: Resources cleaned up")
        except Exception as e:
            logger.warning(f"{self.provider_name}: Error during cleanup: {e}")
    
    def map_voice(self, voice_id: str) -> str:
        """Map generic voice ID to OpenAI voice"""
        # Check if it's already a valid OpenAI voice
        if voice_id in self.VOICES:
            return voice_id
        
        # Try common mappings
        voice_mappings = {
            "male": "onyx",
            "female": "nova",
            "neutral": "alloy",
            "deep": "onyx",
            "soft": "shimmer",
            "expressive": "fable"
        }
        
        return voice_mappings.get(voice_id.lower(), "alloy")

#
# End of openai_adapter.py
#######################################################################################################################