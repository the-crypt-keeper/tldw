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
                logger.error(f"{self.provider_name}: Cannot initialize without API key")
                self._status = ProviderStatus.NOT_CONFIGURED
                return False
            
            # Create HTTP client
            self.client = httpx.AsyncClient(timeout=60.0)
            
            # Test the API key with a minimal request
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            }
            
            # Quick validation - we'll do a proper test in production
            logger.info(f"{self.provider_name}: Initialized successfully")
            self._status = ProviderStatus.AVAILABLE
            return True
            
        except Exception as e:
            logger.error(f"{self.provider_name}: Initialization failed: {e}")
            self._status = ProviderStatus.ERROR
            return False
    
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
            raise ValueError(f"{self.provider_name} not initialized")
        
        # Validate request
        is_valid, error = await self.validate_request(request)
        if not is_valid:
            raise ValueError(error)
        
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
                raise ValueError("Invalid OpenAI API key")
            elif e.response.status_code == 429:
                raise ValueError("OpenAI API rate limit exceeded")
            elif e.response.status_code == 400:
                raise ValueError(f"Invalid request to OpenAI: {error_msg}")
            else:
                raise ValueError(f"OpenAI API error: {error_msg}")
                
        except Exception as e:
            logger.error(f"{self.provider_name} error: {e}")
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
    
    async def close(self):
        """Clean up resources"""
        if self.client:
            await self.client.aclose()
        await super().close()
    
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