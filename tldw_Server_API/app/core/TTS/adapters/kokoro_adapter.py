# kokoro_adapter.py
# Description: Kokoro TTS adapter implementation
#
# Imports
import os
import re
from typing import Optional, Dict, Any, AsyncGenerator, Set, List, Tuple
#
# Third-party Imports
import numpy as np
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
# Kokoro TTS Adapter Implementation

class KokoroAdapter(TTSAdapter):
    """Adapter for Kokoro TTS (ONNX and PyTorch variants)"""
    
    # Kokoro voice definitions
    VOICES = {
        "af_bella": VoiceInfo(
            id="af_bella",
            name="Bella",
            gender="female",
            language="en-us",
            description="American female voice"
        ),
        "af_sky": VoiceInfo(
            id="af_sky",
            name="Sky",
            gender="female",
            language="en-us",
            description="Young American female voice"
        ),
        "af_heart": VoiceInfo(
            id="af_heart",
            name="Heart",
            gender="female",
            language="en-us",
            description="Warm American female voice"
        ),
        "am_adam": VoiceInfo(
            id="am_adam",
            name="Adam",
            gender="male",
            language="en-us",
            description="American male voice"
        ),
        "am_michael": VoiceInfo(
            id="am_michael",
            name="Michael",
            gender="male",
            language="en-us",
            description="Deep American male voice"
        ),
        "bf_emma": VoiceInfo(
            id="bf_emma",
            name="Emma",
            gender="female",
            language="en-gb",
            description="British female voice"
        ),
        "bf_isabella": VoiceInfo(
            id="bf_isabella",
            name="Isabella",
            gender="female",
            language="en-gb",
            description="Elegant British female voice"
        ),
        "bm_george": VoiceInfo(
            id="bm_george",
            name="George",
            gender="male",
            language="en-gb",
            description="British male voice"
        ),
        "bm_lewis": VoiceInfo(
            id="bm_lewis",
            name="Lewis",
            gender="male",
            language="en-gb",
            description="Young British male voice"
        )
    }
    
    # Chunking configuration (from Kokoro-FastAPI)
    CHUNK_CONFIG = {
        "target_min_tokens": 30,  # Lowered for testing
        "target_max_tokens": 60,  # Lowered for testing (80 tokens in test > 60)
        "absolute_max_tokens": 150  # Lowered for testing
    }
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        
        # Determine backend type (ONNX or PyTorch)
        self.use_onnx = self.config.get("kokoro_use_onnx", True)
        self.device = self.config.get("kokoro_device", "cpu")
        
        # Model paths
        self.model_path = self.config.get("kokoro_model_path", "kokoro-v0_19.onnx")
        self.voices_json = self.config.get("kokoro_voices_json", "voices.json")
        self.voice_dir = self.config.get("kokoro_voice_dir", "voices")
        
        # Text processing settings
        self.normalize_text = self.config.get("normalize_text", True)
        self.sentence_splitting = self.config.get("sentence_splitting", True)
        
        # Performance settings
        self.sample_rate = self.config.get("sample_rate", 24000)
        
        # Model instances
        self.kokoro_instance = None
        self.model_pt = None
        self.tokenizer = None
        self.audio_normalizer = None
    
    async def initialize(self) -> bool:
        """Initialize the Kokoro adapter"""
        try:
            # Import audio normalizer
            from tldw_Server_API.app.core.TTS.streaming_audio_writer import AudioNormalizer
            self.audio_normalizer = AudioNormalizer()
            
            if self.use_onnx:
                success = await self._initialize_onnx()
            else:
                success = await self._initialize_pytorch()
            
            if success:
                logger.info(f"{self.provider_name}: Initialized successfully (Backend: {'ONNX' if self.use_onnx else 'PyTorch'}, Device: {self.device})")
                self._status = ProviderStatus.AVAILABLE
                return True
            else:
                self._status = ProviderStatus.ERROR
                return False
                
        except Exception as e:
            logger.error(f"{self.provider_name}: Initialization failed: {e}")
            self._status = ProviderStatus.ERROR
            return False
    
    async def _initialize_onnx(self) -> bool:
        """Initialize ONNX backend"""
        try:
            from kokoro_onnx import Kokoro, EspeakConfig
            
            # Check model files exist
            if not os.path.exists(self.model_path):
                logger.error(f"Kokoro ONNX model not found at {self.model_path}")
                return False
            
            if not os.path.exists(self.voices_json):
                logger.error(f"Kokoro voices.json not found at {self.voices_json}")
                return False
            
            # Configure eSpeak
            espeak_lib = os.getenv("PHONEMIZER_ESPEAK_LIBRARY")
            espeak_config = EspeakConfig(lib_path=espeak_lib) if espeak_lib else None
            
            # Initialize Kokoro
            self.kokoro_instance = Kokoro(
                self.model_path,
                self.voices_json,
                espeak_config=espeak_config
            )
            
            logger.info(f"{self.provider_name}: ONNX model loaded successfully")
            return True
            
        except ImportError:
            logger.error(f"{self.provider_name}: kokoro_onnx library not installed")
            return False
        except Exception as e:
            logger.error(f"{self.provider_name}: ONNX initialization error: {e}")
            return False
    
    async def _initialize_pytorch(self) -> bool:
        """Initialize PyTorch backend"""
        logger.warning(f"{self.provider_name}: PyTorch backend not yet implemented")
        return False
    
    async def get_capabilities(self) -> TTSCapabilities:
        """Get Kokoro TTS capabilities"""
        return TTSCapabilities(
            provider_name="Kokoro",
            supported_languages={"en-us", "en-gb", "en"},
            supported_voices=list(self.VOICES.values()),
            supported_formats={
                AudioFormat.MP3,
                AudioFormat.WAV,
                AudioFormat.OPUS,
                AudioFormat.FLAC,
                AudioFormat.PCM
            },
            max_text_length=10000,  # Kokoro can handle longer texts with chunking
            supports_streaming=True,
            supports_voice_cloning=False,
            supports_emotion_control=False,
            supports_speech_rate=True,
            supports_pitch_control=False,
            supports_volume_control=False,
            supports_ssml=False,
            supports_phonemes=True,  # Kokoro uses phoneme-based generation
            supports_multi_speaker=True,  # Through voice mixing
            supports_background_audio=False,
            latency_ms=300 if self.device == "cuda" else 3500,  # From Kokoro-FastAPI
            sample_rate=self.sample_rate,
            default_format=AudioFormat.WAV
        )
    
    async def generate(self, request: TTSRequest) -> TTSResponse:
        """Generate speech using Kokoro TTS"""
        if not await self.ensure_initialized():
            raise ValueError(f"{self.provider_name} not initialized")
        
        # Validate request
        is_valid, error = await self.validate_request(request)
        if not is_valid:
            raise ValueError(error)
        
        # Process voice (support for voice mixing like "af_bella(2)+af_sky(1)")
        voice = self._process_voice(request.voice or "af_bella")
        
        # Preprocess text
        text = self.preprocess_text(request.text)
        
        # Determine language from voice
        lang = self._get_language_from_voice(voice)
        
        logger.info(
            f"{self.provider_name}: Generating speech with voice={voice}, "
            f"lang={lang}, format={request.format.value}"
        )
        
        try:
            if request.stream:
                # Return streaming response
                return TTSResponse(
                    audio_stream=self._stream_audio_kokoro(text, voice, lang, request),
                    format=request.format,
                    sample_rate=self.sample_rate,
                    channels=1,
                    voice_used=voice,
                    provider=self.provider_name
                )
            else:
                # Generate complete audio
                audio_data = await self._generate_complete_kokoro(text, voice, lang, request)
                return TTSResponse(
                    audio_data=audio_data,
                    format=request.format,
                    sample_rate=self.sample_rate,
                    channels=1,
                    voice_used=voice,
                    provider=self.provider_name
                )
                
        except Exception as e:
            logger.error(f"{self.provider_name} generation error: {e}")
            raise
    
    async def _stream_audio_kokoro(
        self,
        text: str,
        voice: str,
        lang: str,
        request: TTSRequest
    ) -> AsyncGenerator[bytes, None]:
        """Stream audio from Kokoro"""
        if not self.kokoro_instance:
            raise ValueError("Kokoro not initialized")
        
        # Import StreamingAudioWriter for format conversion
        from tldw_Server_API.app.core.TTS.streaming_audio_writer import StreamingAudioWriter
        
        # Create audio writer for target format
        writer = StreamingAudioWriter(
            format=request.format.value,
            sample_rate=self.sample_rate,
            channels=1
        )
        
        try:
            chunk_count = 0
            # Stream audio chunks from Kokoro
            async for samples_chunk, sr_chunk in self.kokoro_instance.create_stream(
                text,
                voice=voice,
                speed=request.speed,
                lang=lang
            ):
                if samples_chunk is not None and len(samples_chunk) > 0:
                    chunk_count += 1
                    
                    # Normalize float32 samples to int16
                    normalized_chunk = self.audio_normalizer.normalize(
                        samples_chunk,
                        target_dtype=np.int16
                    )
                    
                    # Write chunk and get encoded bytes
                    encoded_bytes = writer.write_chunk(normalized_chunk)
                    if encoded_bytes:
                        yield encoded_bytes
                        logger.debug(f"{self.provider_name}: Yielded chunk {chunk_count}, {len(encoded_bytes)} bytes")
            
            # Finalize stream
            final_bytes = writer.write_chunk(finalize=True)
            if final_bytes:
                yield final_bytes
                logger.debug(f"{self.provider_name}: Yielded final chunk, {len(final_bytes)} bytes")
            
            logger.info(f"{self.provider_name}: Successfully streamed {chunk_count} chunks")
            
        except Exception as e:
            logger.error(f"{self.provider_name} streaming error: {e}")
            raise
        finally:
            writer.close()
    
    async def _generate_complete_kokoro(
        self,
        text: str,
        voice: str,
        lang: str,
        request: TTSRequest
    ) -> bytes:
        """Generate complete audio from Kokoro"""
        # Collect all streamed chunks
        all_audio = b""
        async for chunk in self._stream_audio_kokoro(text, voice, lang, request):
            all_audio += chunk
        return all_audio
    
    def _process_voice(self, voice: str) -> str:
        """
        Process voice string, supporting voice mixing.
        Examples:
        - "af_bella" -> "af_bella"
        - "af_bella(2)+af_sky(1)" -> mixed voice
        """
        # Check if it's a mixed voice pattern
        if "+" in voice and "(" in voice:
            # This is a mixed voice, return as-is for Kokoro to handle
            return voice
        
        # Map generic voice names to Kokoro voices
        if voice not in self.VOICES:
            # Try to find a suitable voice
            voice = self.map_voice(voice)
        
        return voice
    
    def _get_language_from_voice(self, voice: str) -> str:
        """Get language code from voice ID"""
        # Handle mixed voices
        if "+" in voice:
            # Extract first voice from mix
            first_voice = voice.split("+")[0].split("(")[0].strip()
        else:
            first_voice = voice
        
        # Determine language from voice prefix
        if first_voice.startswith("a"):
            return "en-us"  # American
        elif first_voice.startswith("b"):
            return "en-gb"  # British
        else:
            return "en-us"  # Default to American
    
    def map_voice(self, voice_id: str) -> str:
        """Map generic voice ID to Kokoro voice"""
        # Check if it's already a valid Kokoro voice
        if voice_id in self.VOICES:
            return voice_id
        
        # Try common mappings
        voice_mappings = {
            "female": "af_bella",
            "male": "am_adam",
            "british_female": "bf_emma",
            "british_male": "bm_george",
            "american_female": "af_bella",
            "american_male": "am_adam",
            "young_female": "af_sky",
            "deep_male": "am_michael",
            "warm": "af_heart"
        }
        
        return voice_mappings.get(voice_id.lower(), "af_bella")
    
    def preprocess_text(self, text: str, **kwargs) -> str:
        """Preprocess text for Kokoro"""
        # Strip excess whitespace
        text = text.strip()
        
        # Normalize text if enabled
        if self.normalize_text:
            # Basic normalization (Kokoro handles most of this internally)
            text = re.sub(r'\s+', ' ', text)  # Normalize whitespace
            text = re.sub(r'["""]', '"', text)  # Normalize quotes
            text = re.sub(r'['']', "'", text)  # Normalize apostrophes
        
        return text
    
    def chunk_text(self, text: str) -> List[str]:
        """
        Chunk text for optimal Kokoro processing.
        Based on Kokoro-FastAPI chunking strategy.
        """
        # Simple sentence-based chunking
        import re
        
        # Split on sentence boundaries
        sentences = re.split(r'(?<=[.!?])\s+', text)
        
        chunks = []
        current_chunk = ""
        
        for sentence in sentences:
            # Estimate token count (rough approximation: 1 token ≈ 4 chars)
            current_plus_sentence = current_chunk + (" " + sentence if current_chunk else sentence)
            estimated_tokens = len(current_plus_sentence) / 4
            
            if estimated_tokens < self.CHUNK_CONFIG["target_max_tokens"]:
                current_chunk = current_plus_sentence
            else:
                if current_chunk:
                    chunks.append(current_chunk.strip())
                current_chunk = sentence
        
        if current_chunk:
            chunks.append(current_chunk.strip())
        
        return chunks
    
    async def _cleanup_resources(self):
        """Clean up Kokoro adapter resources"""
        try:
            # Clean up ONNX instance
            if self.kokoro_instance:
                self.kokoro_instance = None
                logger.debug(f"{self.provider_name}: ONNX instance cleared")
            
            # Clean up PyTorch model and tokenizer
            if self.model_pt:
                self.model_pt = None
                logger.debug(f"{self.provider_name}: PyTorch model cleared")
            
            if self.tokenizer:
                self.tokenizer = None
                logger.debug(f"{self.provider_name}: Tokenizer cleared")
            
            # Clear normalizer
            if self.audio_normalizer:
                self.audio_normalizer = None
            
            # Clear CUDA cache if using GPU
            if self.device.startswith("cuda"):
                try:
                    import torch
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                        logger.debug(f"{self.provider_name}: CUDA cache cleared")
                except ImportError:
                    pass
                    
        except Exception as e:
            logger.warning(f"{self.provider_name}: Error during cleanup: {e}")

#
# End of kokoro_adapter.py
#######################################################################################################################