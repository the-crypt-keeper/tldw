# elevenlabs_adapter.py
# Description: ElevenLabs TTS adapter implementation
#
# Imports
import os
from typing import Optional, Dict, Any, AsyncGenerator, Set, List
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
# ElevenLabs TTS Adapter Implementation

class ElevenLabsAdapter(TTSAdapter):
    """Adapter for ElevenLabs TTS API"""
    
    # ElevenLabs API endpoints
    BASE_URL = "https://api.elevenlabs.io/v1"
    
    # Default voice IDs (can be customized)
    DEFAULT_VOICES = {
        "rachel": VoiceInfo(
            id="21m00Tcm4TlvDq8ikWAM",
            name="Rachel",
            gender="female",
            language="en",
            description="American female voice"
        ),
        "drew": VoiceInfo(
            id="29vD33N1CtxCmqQRPOHJ",
            name="Drew",
            gender="male",
            language="en",
            description="American male voice"
        ),
        "clyde": VoiceInfo(
            id="2EiwWnXFnvU5JabPnv8n",
            name="Clyde",
            gender="male",
            language="en",
            description="American male voice"
        ),
        "paul": VoiceInfo(
            id="5Q0t7uMcjvnagumLfvZi",
            name="Paul",
            gender="male",
            language="en",
            description="American male voice"
        ),
        "domi": VoiceInfo(
            id="AZnzlk1XvdvUeBnXmlld",
            name="Domi",
            gender="female",
            language="en",
            description="American female voice"
        ),
        "dave": VoiceInfo(
            id="CYw3kZ02Hs0563khs1Fj",
            name="Dave",
            gender="male",
            language="en-gb",
            description="British male voice"
        ),
        "fin": VoiceInfo(
            id="D38z5RcWu1voky8WS1ja",
            name="Fin",
            gender="male",
            language="en",
            description="Irish male voice"
        ),
        "bella": VoiceInfo(
            id="EXAVITQu4vr4xnSDxMaL",
            name="Bella",
            gender="female",
            language="en",
            description="American female voice"
        ),
        "antoni": VoiceInfo(
            id="ErXwobaYiN019PkySvjV",
            name="Antoni",
            gender="male",
            language="en",
            description="American male voice"
        ),
        "thomas": VoiceInfo(
            id="GBv7mTt0atIp3Br8iCZE",
            name="Thomas",
            gender="male",
            language="en",
            description="American male voice"
        )
    }
    
    # Model configurations
    MODELS = {
        "eleven_monolingual_v1": {
            "model_id": "eleven_monolingual_v1",
            "languages": ["en"],
            "description": "English only, fastest"
        },
        "eleven_multilingual_v1": {
            "model_id": "eleven_multilingual_v1",
            "languages": ["en", "de", "pl", "es", "it", "fr", "pt", "hi", "ar"],
            "description": "V1 with multiple languages"
        },
        "eleven_multilingual_v2": {
            "model_id": "eleven_multilingual_v2",
            "languages": ["en", "de", "pl", "es", "it", "fr", "pt", "hi", "ar", "ja", "ko", "zh", "nl", "tr", "sv", "id", "fil", "ms", "ro", "uk", "el"],
            "description": "V2 with extensive language support"
        },
        "eleven_turbo_v2": {
            "model_id": "eleven_turbo_v2",
            "languages": ["en"],
            "description": "Turbo model for low latency English"
        }
    }
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        
        # API configuration
        self.api_key = self.config.get("elevenlabs_api_key") or os.getenv("ELEVENLABS_API_KEY")
        self.base_url = self.config.get("elevenlabs_base_url", self.BASE_URL)
        
        # Model selection
        self.default_model = self.config.get("elevenlabs_model", "eleven_monolingual_v1")
        
        # Voice settings
        self.stability = self.config.get("elevenlabs_stability", 0.5)
        self.similarity_boost = self.config.get("elevenlabs_similarity_boost", 0.5)
        self.style = self.config.get("elevenlabs_style", 0.0)
        self.use_speaker_boost = self.config.get("elevenlabs_speaker_boost", True)
        
        # HTTP client
        self.client: Optional[httpx.AsyncClient] = None
        
        # Cache for user voices
        self._user_voices: List[VoiceInfo] = []
    
    async def initialize(self) -> bool:
        """Initialize the ElevenLabs adapter"""
        try:
            if not self.api_key:
                logger.warning(f"{self.provider_name}: No API key provided")
                self._status = ProviderStatus.NOT_CONFIGURED
                return False
            
            # Initialize HTTP client
            self.client = httpx.AsyncClient(
                timeout=httpx.Timeout(30.0, connect=5.0),
                limits=httpx.Limits(max_keepalive_connections=5)
            )
            
            # Test API connection and fetch user voices
            await self._fetch_user_voices()
            
            logger.info(f"{self.provider_name}: Initialized successfully")
            self._status = ProviderStatus.AVAILABLE
            return True
            
        except Exception as e:
            logger.error(f"{self.provider_name}: Initialization failed: {e}")
            self._status = ProviderStatus.ERROR
            return False
    
    async def _fetch_user_voices(self):
        """Fetch available voices from ElevenLabs API"""
        try:
            headers = {"xi-api-key": self.api_key}
            response = await self.client.get(f"{self.base_url}/voices", headers=headers)
            
            if response.status_code == 200:
                data = response.json()
                self._user_voices = []
                
                for voice in data.get("voices", []):
                    voice_info = VoiceInfo(
                        id=voice["voice_id"],
                        name=voice["name"],
                        gender=voice.get("labels", {}).get("gender", "unknown"),
                        language=voice.get("labels", {}).get("language", "en"),
                        description=voice.get("labels", {}).get("description", ""),
                        use_case=voice.get("labels", {}).get("use_case", [])
                    )
                    self._user_voices.append(voice_info)
                
                logger.info(f"{self.provider_name}: Loaded {len(self._user_voices)} voices")
            else:
                logger.warning(f"{self.provider_name}: Failed to fetch voices: {response.status_code}")
                
        except Exception as e:
            logger.error(f"{self.provider_name}: Error fetching voices: {e}")
    
    async def get_capabilities(self) -> TTSCapabilities:
        """Get ElevenLabs TTS capabilities"""
        # Determine supported languages based on selected model
        model_config = self.MODELS.get(self.default_model, self.MODELS["eleven_monolingual_v1"])
        supported_languages = set(model_config["languages"])
        
        # Combine default and user voices
        all_voices = list(self.DEFAULT_VOICES.values()) + self._user_voices
        
        return TTSCapabilities(
            provider_name="ElevenLabs",
            supported_languages=supported_languages,
            supported_voices=all_voices,
            supported_formats={
                AudioFormat.MP3,
                AudioFormat.PCM,
                AudioFormat.ULAW
            },
            max_text_length=5000,  # ElevenLabs character limit
            supports_streaming=True,
            supports_voice_cloning=True,  # Pro feature
            supports_emotion_control=True,  # Via voice settings
            supports_speech_rate=False,  # Not directly supported
            supports_pitch_control=False,
            supports_volume_control=False,
            supports_ssml=False,
            supports_phonemes=False,
            supports_multi_speaker=False,
            supports_background_audio=False,
            latency_ms=300,  # Typical latency
            sample_rate=44100,  # Default output sample rate
            default_format=AudioFormat.MP3
        )
    
    async def generate(self, request: TTSRequest) -> TTSResponse:
        """Generate speech using ElevenLabs TTS"""
        if not await self.ensure_initialized():
            raise ValueError(f"{self.provider_name} not initialized")
        
        # Validate request
        is_valid, error = await self.validate_request(request)
        if not is_valid:
            raise ValueError(error)
        
        # Prepare voice ID
        voice_id = self._get_voice_id(request.voice or "rachel")
        
        # Select model
        model_id = self._select_model(request)
        
        logger.info(
            f"{self.provider_name}: Generating speech with voice={voice_id}, "
            f"model={model_id}, format={request.format.value}"
        )
        
        try:
            if request.stream:
                # Return streaming response
                return TTSResponse(
                    audio_stream=self._stream_audio_elevenlabs(
                        text=request.text,
                        voice_id=voice_id,
                        model_id=model_id,
                        request=request
                    ),
                    format=request.format,
                    sample_rate=44100,
                    channels=1,
                    voice_used=request.voice or "rachel",
                    provider=self.provider_name
                )
            else:
                # Generate complete audio
                audio_data = await self._generate_complete_elevenlabs(
                    text=request.text,
                    voice_id=voice_id,
                    model_id=model_id,
                    request=request
                )
                return TTSResponse(
                    audio_data=audio_data,
                    format=request.format,
                    sample_rate=44100,
                    channels=1,
                    voice_used=request.voice or "rachel",
                    provider=self.provider_name
                )
                
        except Exception as e:
            logger.error(f"{self.provider_name} generation error: {e}")
            raise
    
    async def _stream_audio_elevenlabs(
        self,
        text: str,
        voice_id: str,
        model_id: str,
        request: TTSRequest
    ) -> AsyncGenerator[bytes, None]:
        """Stream audio from ElevenLabs API"""
        url = f"{self.base_url}/text-to-speech/{voice_id}/stream"
        
        headers = {
            "Accept": self._get_accept_header(request.format),
            "xi-api-key": self.api_key,
            "Content-Type": "application/json"
        }
        
        # Prepare voice settings
        voice_settings = {
            "stability": request.extra_params.get("stability", self.stability),
            "similarity_boost": request.extra_params.get("similarity_boost", self.similarity_boost),
            "style": request.extra_params.get("style", self.style),
            "use_speaker_boost": request.extra_params.get("speaker_boost", self.use_speaker_boost)
        }
        
        payload = {
            "text": text,
            "model_id": model_id,
            "voice_settings": voice_settings
        }
        
        try:
            async with self.client.stream("POST", url, headers=headers, json=payload) as response:
                response.raise_for_status()
                
                chunk_count = 0
                async for chunk in response.aiter_bytes(chunk_size=1024):
                    if chunk:
                        chunk_count += 1
                        yield chunk
                        
                        if chunk_count % 10 == 0:
                            logger.debug(f"{self.provider_name}: Streamed {chunk_count} chunks")
                
                logger.info(f"{self.provider_name}: Successfully streamed {chunk_count} chunks")
                
        except httpx.HTTPStatusError as e:
            logger.error(f"{self.provider_name} HTTP error: {e.response.status_code} - {e.response.text}")
            raise
        except Exception as e:
            logger.error(f"{self.provider_name} streaming error: {e}")
            raise
    
    async def _generate_complete_elevenlabs(
        self,
        text: str,
        voice_id: str,
        model_id: str,
        request: TTSRequest
    ) -> bytes:
        """Generate complete audio from ElevenLabs"""
        # Collect all streamed chunks
        all_audio = b""
        async for chunk in self._stream_audio_elevenlabs(text, voice_id, model_id, request):
            all_audio += chunk
        return all_audio
    
    def _get_voice_id(self, voice_name: str) -> str:
        """Get ElevenLabs voice ID from voice name"""
        # Check if it's already a voice ID (21 characters)
        if len(voice_name) == 20 and voice_name.isalnum():
            return voice_name
        
        # Check default voices
        voice_lower = voice_name.lower()
        if voice_lower in self.DEFAULT_VOICES:
            return self.DEFAULT_VOICES[voice_lower].id
        
        # Check user voices
        for voice in self._user_voices:
            if voice.name.lower() == voice_lower:
                return voice.id
        
        # Default to Rachel
        logger.warning(f"{self.provider_name}: Voice '{voice_name}' not found, using default")
        return self.DEFAULT_VOICES["rachel"].id
    
    def _select_model(self, request: TTSRequest) -> str:
        """Select appropriate ElevenLabs model"""
        # Check if model specified in extra params
        if "model" in request.extra_params:
            return request.extra_params["model"]
        
        # Select based on language
        if request.language and request.language != "en":
            # Use multilingual model for non-English
            return "eleven_multilingual_v2"
        
        # Use configured default
        return self.default_model
    
    def _get_accept_header(self, format: AudioFormat) -> str:
        """Get Accept header for requested format"""
        format_map = {
            AudioFormat.MP3: "audio/mpeg",
            AudioFormat.PCM: "audio/pcm",
            AudioFormat.ULAW: "audio/ulaw"
        }
        return format_map.get(format, "audio/mpeg")
    
    async def _cleanup_resources(self):
        """Clean up ElevenLabs adapter resources"""
        if self.client:
            try:
                await self.client.aclose()
                self.client = None
                logger.debug(f"{self.provider_name}: HTTP client closed")
            except Exception as e:
                logger.warning(f"{self.provider_name}: Error closing HTTP client: {e}")
    
    def map_voice(self, voice_id: str) -> str:
        """Map generic voice ID to ElevenLabs voice"""
        # Common mappings
        voice_mappings = {
            "female": "rachel",
            "male": "drew",
            "british": "dave",
            "irish": "fin",
            "young_female": "bella",
            "young_male": "antoni"
        }
        
        return voice_mappings.get(voice_id.lower(), voice_id)
    
    def preprocess_text(self, text: str, **kwargs) -> str:
        """Preprocess text for ElevenLabs"""
        # ElevenLabs handles most text normalization internally
        # Just ensure text isn't too long
        if len(text) > 5000:
            logger.warning(f"{self.provider_name}: Text truncated to 5000 characters")
            text = text[:5000]
        
        return text.strip()

#
# End of elevenlabs_adapter.py
#######################################################################################################################