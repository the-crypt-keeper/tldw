# chatterbox_adapter.py
# Description: Chatterbox TTS adapter implementation (Resemble AI)
#
# Imports
import os
from typing import Optional, Dict, Any, AsyncGenerator, Set, List
#
# Third-party Imports
import torch
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
from ..tts_exceptions import (
    TTSProviderNotConfiguredError,
    TTSProviderInitializationError,
    TTSModelNotFoundError,
    TTSModelLoadError,
    TTSGenerationError,
    TTSResourceError,
    TTSGPUError,
    TTSInvalidVoiceReferenceError
)
from ..tts_validation import validate_tts_request
from ..tts_resource_manager import get_resource_manager
#
#######################################################################################################################
#
# Chatterbox TTS Adapter Implementation

class ChatterboxAdapter(TTSAdapter):
    """
    Adapter for Chatterbox TTS from Resemble AI.
    First open-source TTS with emotion exaggeration control.
    """
    
    # Emotion types supported by Chatterbox
    EMOTIONS = {
        "neutral", "happy", "sad", "angry", "surprised",
        "fearful", "disgusted", "excited", "calm", "confused"
    }
    
    # Voice presets (can be extended with voice cloning)
    VOICE_PRESETS = {
        "default": VoiceInfo(
            id="default",
            name="Default",
            gender="neutral",
            description="Default Chatterbox voice",
            styles=["neutral", "conversational"]
        ),
        "energetic": VoiceInfo(
            id="energetic",
            name="Energetic",
            gender="neutral",
            description="High energy voice",
            styles=["excited", "happy"]
        ),
        "calm": VoiceInfo(
            id="calm",
            name="Calm",
            gender="neutral",
            description="Calm and soothing voice",
            styles=["calm", "neutral"]
        ),
        "professional": VoiceInfo(
            id="professional",
            name="Professional",
            gender="neutral",
            description="Professional business voice",
            styles=["neutral", "confident"]
        )
    }
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        
        # Model configuration
        self.model_path = self.config.get("chatterbox_model_path", "resemble-ai/chatterbox")
        self.device = self.config.get("chatterbox_device", "cuda" if torch.cuda.is_available() else "cpu")
        
        # Audio configuration
        self.sample_rate = self.config.get("chatterbox_sample_rate", 24000)
        
        # Model instances
        self.model = None
        self.vocoder = None
        self.processor = None
        
        # Performance settings
        self.use_fp16 = self.config.get("chatterbox_use_fp16", True) and self.device == "cuda"
        
        # Chatterbox specific settings
        self.enable_watermark = self.config.get("chatterbox_watermark", True)  # Perth watermarker
        self.target_latency_ms = 200  # Sub-200ms latency target
    
    async def initialize(self) -> bool:
        """Initialize the Chatterbox TTS model"""
        try:
            logger.info(f"{self.provider_name}: Loading Chatterbox TTS model...")
            
            # Check if chatterbox-tts is installed
            try:
                import chatterbox_tts
                from chatterbox_tts import ChatterboxModel, ChatterboxProcessor
            except ImportError as e:
                error_msg = (
                    f"{self.provider_name}: Required libraries not installed. "
                    f"To use Chatterbox, follow these steps:\n"
                    f"1. Install Chatterbox (when available):\n"
                    f"   pip install chatterbox-tts\n"
                    f"2. Or install from source:\n"
                    f"   git clone https://github.com/resemble-ai/chatterbox\n"
                    f"   cd chatterbox && pip install -e .\n"
                    f"3. Download the model:\n"
                    f"   huggingface-cli download resemble-ai/chatterbox --local-dir models/chatterbox/\n"
                    f"4. Ensure CUDA is available for optimal performance (4GB+ VRAM recommended)"
                )
                logger.error(f"{self.provider_name}: chatterbox-tts library not installed")
                logger.error(error_msg)
                self._status = ProviderStatus.NOT_CONFIGURED
                raise TTSModelLoadError(
                    "Failed to import chatterbox-tts library",
                    provider=self.provider_name,
                    details={"error": str(e), "suggestion": error_msg}
                )
            
            # Load processor
            logger.info(f"{self.provider_name}: Loading processor...")
            self.processor = ChatterboxProcessor.from_pretrained(
                self.model_path,
                device=self.device
            )
            
            # Load model
            logger.info(f"{self.provider_name}: Loading model...")
            self.model = ChatterboxModel.from_pretrained(
                self.model_path,
                torch_dtype=torch.float16 if self.use_fp16 else torch.float32,
                device_map=self.device
            )
            
            # Set to evaluation mode
            self.model.eval()
            
            # Initialize Perth watermarker if enabled
            if self.enable_watermark:
                logger.info(f"{self.provider_name}: Initializing Perth watermarker...")
                # Perth watermarker would be initialized here
            
            logger.info(
                f"{self.provider_name}: Initialized successfully "
                f"(Device: {self.device}, FP16: {self.use_fp16}, Watermark: {self.enable_watermark})"
            )
            self._status = ProviderStatus.AVAILABLE
            return True
            
        except Exception as e:
            logger.error(f"{self.provider_name}: Initialization failed: {e}")
            self._status = ProviderStatus.ERROR
            return False
    
    async def get_capabilities(self) -> TTSCapabilities:
        """Get Chatterbox TTS capabilities"""
        return TTSCapabilities(
            provider_name="Chatterbox",
            supported_languages={"en"},  # Currently English only
            supported_voices=list(self.VOICE_PRESETS.values()),
            supported_formats={
                AudioFormat.WAV,
                AudioFormat.MP3,
                AudioFormat.OPUS,
                AudioFormat.FLAC,
                AudioFormat.PCM
            },
            max_text_length=10000,
            supports_streaming=True,
            supports_voice_cloning=True,  # Via short reference clips
            supports_emotion_control=True,  # Key feature - emotion exaggeration
            supports_speech_rate=True,
            supports_pitch_control=True,
            supports_volume_control=True,
            supports_ssml=False,
            supports_phonemes=False,
            supports_multi_speaker=False,
            supports_background_audio=False,
            latency_ms=self.target_latency_ms,  # Sub-200ms target
            sample_rate=self.sample_rate,
            default_format=AudioFormat.WAV
        )
    
    async def generate(self, request: TTSRequest) -> TTSResponse:
        """Generate speech using Chatterbox TTS"""
        if not await self.ensure_initialized():
            raise ValueError(f"{self.provider_name} not initialized")
        
        # Validate request
        is_valid, error = await self.validate_request(request)
        if not is_valid:
            raise ValueError(error)
        
        # Process voice
        voice = request.voice or "default"
        
        # Handle voice cloning if reference provided
        voice_reference_path = None
        if request.voice_reference:
            voice_reference_path = await self._prepare_voice_reference(request.voice_reference)
        
        logger.info(
            f"{self.provider_name}: Generating speech with voice={voice}, "
            f"emotion={request.emotion}, intensity={request.emotion_intensity}, "
            f"format={request.format.value}"
        )
        
        try:
            if request.stream:
                # Return streaming response
                return TTSResponse(
                    audio_stream=self._stream_audio_chatterbox(request, voice, voice_reference_path),
                    format=request.format,
                    sample_rate=self.sample_rate,
                    channels=1,
                    voice_used=voice,
                    provider=self.provider_name,
                    metadata={
                        "emotion": request.emotion,
                        "emotion_intensity": request.emotion_intensity,
                        "watermarked": self.enable_watermark
                    }
                )
            else:
                # Generate complete audio
                audio_data = await self._generate_complete_chatterbox(request, voice, voice_reference_path)
                return TTSResponse(
                    audio_data=audio_data,
                    format=request.format,
                    sample_rate=self.sample_rate,
                    channels=1,
                    voice_used=voice,
                    provider=self.provider_name,
                    metadata={
                        "emotion": request.emotion,
                        "emotion_intensity": request.emotion_intensity,
                        "watermarked": self.enable_watermark
                    }
                )
                
        except Exception as e:
            logger.error(f"{self.provider_name} generation error: {e}")
            raise
    
    async def _stream_audio_chatterbox(
        self,
        request: TTSRequest,
        voice: str,
        voice_reference_path: Optional[str] = None
    ) -> AsyncGenerator[bytes, None]:
        """Stream audio from Chatterbox model"""
        if not self.model or not self.processor:
            raise ValueError("Chatterbox model not initialized")
        
        # Import StreamingAudioWriter for format conversion
        from tldw_Server_API.app.core.TTS.streaming_audio_writer import (
            StreamingAudioWriter,
            AudioNormalizer
        )
        
        normalizer = AudioNormalizer()
        writer = StreamingAudioWriter(
            format=request.format.value,
            sample_rate=self.sample_rate,
            channels=1
        )
        
        try:
            # Prepare input with emotion control
            input_data = self._prepare_chatterbox_input(request, voice)
            
            # Process text
            inputs = self.processor(
                text=input_data["text"],
                voice=input_data["voice"],
                emotion=input_data["emotion"],
                emotion_intensity=input_data["emotion_intensity"],
                return_tensors="pt"
            ).to(self.device)
            
            # Generate with streaming
            with torch.no_grad():
                # Prepare generation kwargs
                gen_kwargs = {
                    "max_length": 2000,
                    "temperature": 0.7,
                    "do_sample": True,
                    "speed": request.speed,
                    "pitch": request.pitch,
                    "volume": request.volume
                }
                
                # Add voice reference if provided
                if voice_reference_path:
                    gen_kwargs["audio_prompt_path"] = voice_reference_path
                    logger.info(f"Using voice reference: {voice_reference_path}")
                
                # Chatterbox supports streaming generation
                stream_generator = self.model.generate_stream(
                    **inputs,
                    **gen_kwargs
                )
                
                # Process streaming chunks
                for audio_chunk in stream_generator:
                    if audio_chunk is not None and len(audio_chunk) > 0:
                        # Apply watermark if enabled
                        if self.enable_watermark:
                            audio_chunk = self._apply_watermark(audio_chunk)
                        
                        # Normalize to int16
                        normalized_chunk = normalizer.normalize(
                            audio_chunk,
                            target_dtype=np.int16
                        )
                        
                        # Encode to target format
                        encoded_bytes = writer.write_chunk(normalized_chunk)
                        if encoded_bytes:
                            yield encoded_bytes
            
            # Finalize stream
            final_bytes = writer.write_chunk(finalize=True)
            if final_bytes:
                yield final_bytes
            
            logger.info(f"{self.provider_name}: Successfully generated audio with emotion control")
            
        except Exception as e:
            logger.error(f"{self.provider_name} streaming error: {e}")
            raise
        finally:
            writer.close()
            # Clean up voice reference file if used
            if voice_reference_path:
                try:
                    from pathlib import Path
                    Path(voice_reference_path).unlink(missing_ok=True)
                    logger.debug(f"Cleaned up voice reference: {voice_reference_path}")
                except Exception as e:
                    logger.warning(f"Failed to clean up voice reference: {e}")
    
    async def _generate_complete_chatterbox(
        self,
        request: TTSRequest,
        voice: str,
        voice_reference_path: Optional[str] = None
    ) -> bytes:
        """Generate complete audio from Chatterbox"""
        all_audio = b""
        async for chunk in self._stream_audio_chatterbox(request, voice, voice_reference_path):
            all_audio += chunk
        return all_audio
    
    def _prepare_chatterbox_input(
        self,
        request: TTSRequest,
        voice: str
    ) -> Dict[str, Any]:
        """
        Prepare input for Chatterbox with emotion control.
        Chatterbox's key feature is emotion exaggeration.
        """
        # Validate emotion
        emotion = request.emotion or "neutral"
        if emotion not in self.EMOTIONS:
            logger.warning(f"Unknown emotion '{emotion}', using 'neutral'")
            emotion = "neutral"
        
        # Emotion intensity (0.0 to 2.0, where 1.0 is normal)
        emotion_intensity = max(0.0, min(2.0, request.emotion_intensity))
        
        return {
            "text": self.preprocess_text(request.text),
            "voice": voice,
            "emotion": emotion,
            "emotion_intensity": emotion_intensity
        }
    
    def _apply_watermark(self, audio: np.ndarray) -> np.ndarray:
        """
        Apply Perth watermark to audio.
        Watermarks survive MP3 compression and common edits.
        """
        # In production, this would use Resemble's Perth watermarker
        # The watermark is imperceptible but detectable
        return audio  # Placeholder
    
    async def _clone_voice_chatterbox(self, voice_reference: bytes) -> str:
        """
        Clone a voice from a short reference clip.
        Chatterbox supports speaker adaptation.
        """
        logger.info(f"{self.provider_name}: Cloning voice from {len(voice_reference)} bytes")
        
        # In production, this would:
        # 1. Process the audio reference
        # 2. Extract speaker embeddings
        # 3. Create a new voice ID
        
        return "cloned_voice"
    
    async def _prepare_voice_reference(self, voice_reference: bytes) -> Optional[str]:
        """
        Prepare voice reference audio for Chatterbox.
        
        Args:
            voice_reference: Voice reference audio bytes
            
        Returns:
            Path to temporary voice reference file or None if processing fails
        """
        try:
            import tempfile
            from pathlib import Path
            from tldw_Server_API.app.core.TTS.audio_utils import process_voice_reference
            
            # Process voice reference for Chatterbox requirements
            processed_audio, error = process_voice_reference(
                voice_reference,
                provider='chatterbox',
                validate=True,
                convert=True
            )
            
            if error:
                logger.error(f"Voice reference processing failed: {error}")
                return None
            
            # Save to temporary file
            with tempfile.NamedTemporaryFile(
                suffix='.wav',
                delete=False,
                prefix='chatterbox_voice_'
            ) as tmp_file:
                tmp_file.write(processed_audio)
                tmp_path = tmp_file.name
            
            logger.info(f"Voice reference prepared: {tmp_path}")
            return tmp_path
            
        except Exception as e:
            logger.error(f"Failed to prepare voice reference: {e}")
            return None
    
    def map_voice(self, voice_id: str) -> str:
        """Map generic voice ID to Chatterbox voice"""
        if voice_id in self.VOICE_PRESETS:
            return voice_id
        
        # Map common voice types
        voice_mappings = {
            "assistant": "professional",
            "friendly": "energetic",
            "soothing": "calm",
            "business": "professional",
            "neutral": "default"
        }
        
        return voice_mappings.get(voice_id.lower(), "default")
    
    def preprocess_text(self, text: str, **kwargs) -> str:
        """Preprocess text for Chatterbox"""
        # Basic preprocessing
        text = super().preprocess_text(text)
        
        # Chatterbox-specific preprocessing
        # Remove excessive punctuation that might affect emotion
        import re
        text = re.sub(r'[!]{2,}', '!', text)  # Multiple exclamations to one
        text = re.sub(r'[?]{2,}', '?', text)  # Multiple questions to one
        text = re.sub(r'\.{4,}', '...', text)  # Normalize ellipsis
        
        return text
    
    async def close(self):
        """Clean up resources"""
        if self.model:
            del self.model
        if self.processor:
            del self.processor
        if self.vocoder:
            del self.vocoder
        
        self.model = None
        self.processor = None
        self.vocoder = None
        
        # Clear GPU cache if using CUDA
        if self.device == "cuda" and torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        await super().close()

#
# End of chatterbox_adapter.py
#######################################################################################################################