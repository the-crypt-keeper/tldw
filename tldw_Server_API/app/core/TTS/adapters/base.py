# base.py
# Description: Base adapter classes and interfaces for TTS providers
#
# Imports
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import (
    AsyncGenerator,
    Dict,
    List,
    Optional,
    Any,
    Set,
    Tuple,
    Union
)
import asyncio
#
# Third-party Imports
import numpy as np
from loguru import logger
#
# Local Imports
#
#######################################################################################################################
#
# Enums and Data Classes

class AudioFormat(Enum):
    """Supported audio output formats"""
    MP3 = "mp3"
    WAV = "wav"
    OPUS = "opus"
    FLAC = "flac"
    AAC = "aac"
    PCM = "pcm"
    OGG = "ogg"
    WEBM = "webm"
    ULAW = "ulaw"  # μ-law encoding used by telephony systems

class ProviderStatus(Enum):
    """Provider availability status"""
    AVAILABLE = "available"
    INITIALIZING = "initializing"
    ERROR = "error"
    DISABLED = "disabled"
    NOT_CONFIGURED = "not_configured"

@dataclass
class VoiceInfo:
    """Information about an available voice"""
    id: str
    name: str
    gender: Optional[str] = None
    language: str = "en"
    accent: Optional[str] = None
    age: Optional[str] = None
    description: Optional[str] = None
    preview_url: Optional[str] = None
    styles: List[str] = field(default_factory=list)
    use_case: List[str] = field(default_factory=list)

@dataclass
class TTSCapabilities:
    """Capabilities of a TTS provider"""
    provider_name: str
    supported_languages: Set[str]
    supported_voices: List[VoiceInfo]
    supported_formats: Set[AudioFormat]
    max_text_length: int
    supports_streaming: bool
    supports_voice_cloning: bool = False
    supports_emotion_control: bool = False
    supports_speech_rate: bool = True
    supports_pitch_control: bool = False
    supports_volume_control: bool = False
    supports_ssml: bool = False
    supports_phonemes: bool = False
    supports_multi_speaker: bool = False
    supports_background_audio: bool = False
    latency_ms: Optional[int] = None  # Average latency in milliseconds
    sample_rate: int = 24000
    default_format: AudioFormat = AudioFormat.MP3

@dataclass
class TTSRequest:
    """Unified TTS request format"""
    text: str
    voice: Optional[str] = None
    language: Optional[str] = "en"
    format: AudioFormat = AudioFormat.MP3
    speed: float = 1.0
    pitch: float = 1.0
    volume: float = 1.0
    emotion: Optional[str] = None
    emotion_intensity: float = 1.0
    style: Optional[str] = None
    seed: Optional[int] = None
    voice_reference: Optional[bytes] = None  # For voice cloning
    ssml: bool = False
    stream: bool = True
    # Multi-speaker dialogue support
    speakers: Optional[Dict[str, str]] = None  # Speaker ID to voice mapping
    # Additional provider-specific parameters
    extra_params: Dict[str, Any] = field(default_factory=dict)

@dataclass
class TTSResponse:
    """Response from TTS generation"""
    audio_data: Optional[bytes] = None
    audio_stream: Optional[AsyncGenerator[bytes, None]] = None
    format: AudioFormat = AudioFormat.MP3
    sample_rate: int = 24000
    channels: int = 1
    duration_seconds: Optional[float] = None
    text_processed: Optional[str] = None
    voice_used: Optional[str] = None
    provider: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


class TTSAdapter(ABC):
    """
    Abstract base class for TTS provider adapters.
    All TTS providers must implement this interface.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the adapter with configuration.
        
        Args:
            config: Provider-specific configuration dictionary
        """
        self.config = config or {}
        self._status = ProviderStatus.NOT_CONFIGURED
        self._capabilities: Optional[TTSCapabilities] = None
        self._initialized = False
        self._init_lock = asyncio.Lock()
    
    @property
    def status(self) -> ProviderStatus:
        """Get current provider status"""
        return self._status
    
    @property
    def capabilities(self) -> Optional[TTSCapabilities]:
        """Get provider capabilities"""
        return self._capabilities
    
    @property
    def provider_name(self) -> str:
        """Get provider name"""
        return self.__class__.__name__.replace('Adapter', '')
    
    @abstractmethod
    async def initialize(self) -> bool:
        """
        Initialize the TTS provider (load models, connect to API, etc).
        
        Returns:
            bool: True if initialization successful, False otherwise
        """
        pass
    
    @abstractmethod
    async def generate(self, request: TTSRequest) -> TTSResponse:
        """
        Generate speech from text.
        
        Args:
            request: TTSRequest object with generation parameters
            
        Returns:
            TTSResponse object with audio data or stream
        """
        pass
    
    @abstractmethod
    async def get_capabilities(self) -> TTSCapabilities:
        """
        Get the capabilities of this provider.
        
        Returns:
            TTSCapabilities object describing what this provider supports
        """
        pass
    
    async def validate_request(self, request: TTSRequest) -> Tuple[bool, Optional[str]]:
        """
        Validate if the request can be handled by this provider.
        
        Args:
            request: TTSRequest to validate
            
        Returns:
            Tuple of (is_valid, error_message)
        """
        if not self._initialized:
            return False, f"{self.provider_name} not initialized"
        
        if not self._capabilities:
            return False, f"{self.provider_name} capabilities not available"
        
        # Check text length
        if len(request.text) > self._capabilities.max_text_length:
            return False, f"Text exceeds maximum length of {self._capabilities.max_text_length}"
        
        # Check format support
        if request.format not in self._capabilities.supported_formats:
            return False, f"Format {request.format.value} not supported"
        
        # Check language support
        if request.language and request.language not in self._capabilities.supported_languages:
            return False, f"Language {request.language} not supported"
        
        # Check streaming support
        if request.stream and not self._capabilities.supports_streaming:
            return False, "Streaming not supported"
        
        # Check voice cloning
        if request.voice_reference and not self._capabilities.supports_voice_cloning:
            return False, "Voice cloning not supported"
        
        # Check emotion control
        if request.emotion and not self._capabilities.supports_emotion_control:
            return False, "Emotion control not supported"
        
        return True, None
    
    async def ensure_initialized(self) -> bool:
        """
        Ensure the provider is initialized (thread-safe).
        
        Returns:
            bool: True if initialized successfully
        """
        if self._initialized:
            return True
        
        async with self._init_lock:
            if self._initialized:
                return True
            
            try:
                self._status = ProviderStatus.INITIALIZING
                success = await self.initialize()
                if success:
                    self._capabilities = await self.get_capabilities()
                    self._status = ProviderStatus.AVAILABLE
                    self._initialized = True
                else:
                    self._status = ProviderStatus.ERROR
                return success
            except Exception as e:
                logger.error(f"{self.provider_name} initialization failed: {e}")
                self._status = ProviderStatus.ERROR
                return False
    
    async def convert_audio_format(
        self,
        audio_data: np.ndarray,
        source_format: AudioFormat,
        target_format: AudioFormat,
        sample_rate: int = 24000
    ) -> bytes:
        """
        Convert audio between formats.
        
        Args:
            audio_data: Audio data as numpy array
            source_format: Source audio format
            target_format: Target audio format
            sample_rate: Sample rate of the audio
            
        Returns:
            Converted audio as bytes
        """
        # This will use the existing StreamingAudioWriter
        from tldw_Server_API.app.core.TTS.streaming_audio_writer import (
            StreamingAudioWriter,
            AudioNormalizer
        )
        
        normalizer = AudioNormalizer()
        writer = StreamingAudioWriter(
            format=target_format.value,
            sample_rate=sample_rate,
            channels=1
        )
        
        try:
            # Normalize to int16 if needed
            if audio_data.dtype != np.int16:
                audio_data = normalizer.normalize(audio_data, target_dtype=np.int16)
            
            # Write and finalize
            writer.write_chunk(audio_data)
            final_bytes = writer.write_chunk(finalize=True)
            return final_bytes
        finally:
            writer.close()
    
    async def close(self):
        """Clean up resources"""
        try:
            # Perform adapter-specific cleanup
            await self._cleanup_resources()
            self._initialized = False
            self._status = ProviderStatus.DISABLED
            logger.info(f"{self.provider_name} adapter closed")
        except Exception as e:
            logger.error(f"Error closing {self.provider_name} adapter: {e}")
    
    async def _cleanup_resources(self):
        """Override this method for adapter-specific cleanup"""
        pass
    
    def map_voice(self, voice_id: str) -> str:
        """
        Map a generic voice ID to provider-specific voice.
        
        Args:
            voice_id: Generic voice identifier
            
        Returns:
            Provider-specific voice identifier
        """
        # Default implementation returns the same ID
        # Subclasses can override for custom mapping
        return voice_id
    
    def preprocess_text(self, text: str, **kwargs) -> str:
        """
        Preprocess text before sending to TTS engine.
        
        Args:
            text: Input text
            **kwargs: Additional preprocessing options
            
        Returns:
            Preprocessed text
        """
        # Default implementation does minimal preprocessing
        # Subclasses can override for provider-specific needs
        return text.strip()
    
    def parse_dialogue(self, text: str) -> List[Tuple[str, str]]:
        """
        Parse multi-speaker dialogue from text.
        
        Args:
            text: Text potentially containing speaker markers
            
        Returns:
            List of (speaker, text) tuples
        """
        # Simple implementation for dialogue parsing
        # Format: "Speaker1: Hello there! Speaker2: Hi!"
        import re
        
        pattern = r'([A-Za-z0-9]+):\s*([^:]+?)(?=(?:[A-Za-z0-9]+:|$))'
        matches = re.findall(pattern, text)
        # Strip whitespace from dialogue text
        matches = [(speaker, dialogue.strip()) for speaker, dialogue in matches]
        
        if matches:
            return matches
        else:
            return [("default", text)]
    
    async def stream_audio(
        self,
        audio_chunks: AsyncGenerator[bytes, None]
    ) -> AsyncGenerator[bytes, None]:
        """
        Stream audio chunks with any necessary processing.
        
        Args:
            audio_chunks: Async generator of audio chunks
            
        Yields:
            Processed audio chunks
        """
        async for chunk in audio_chunks:
            if chunk:
                yield chunk

#
# End of base.py
#######################################################################################################################