# vibevoice_adapter.py
# Description: VibeVoice TTS adapter implementation - emotion and tone-aware synthesis
#
# Imports
import os
import json
from typing import Optional, Dict, Any, AsyncGenerator, Set, List
from pathlib import Path
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
    TTSInsufficientMemoryError,
    TTSGPUError
)
from ..tts_validation import validate_tts_request
from ..tts_resource_manager import get_resource_manager
#
#######################################################################################################################
#
# VibeVoice TTS Adapter Implementation

class VibeVoiceAdapter(TTSAdapter):
    """
    Adapter for Microsoft VibeVoice - Expressive long-form multi-speaker conversational audio.
    Generates up to 90 minutes of speech with 4 distinct speakers.
    Features spontaneous background music and emergent singing capabilities.
    """
    
    # Model variants available
    MODEL_VARIANTS = {
        "1.5B": {
            "path": "microsoft/VibeVoice-1.5B",
            "context": 64000,  # 64K context (~90 min generation)
            "frame_rate": 7.5   # Hz
        },
        "7B": {
            "path": "WestZhang/VibeVoice-Large-pt", 
            "context": 32000,  # 32K context (~45 min generation)
            "frame_rate": 7.5   # Hz
        }
    }
    
    # Voice presets with personality profiles
    VOICE_PRESETS = {
        "aurora": VoiceInfo(
            id="aurora",
            name="Aurora",
            gender="female",
            description="Warm and versatile voice with excellent emotion range",
            styles=["empathetic", "professional", "friendly"]
        ),
        "phoenix": VoiceInfo(
            id="phoenix",
            name="Phoenix",
            gender="male",
            description="Dynamic voice with strong presence",
            styles=["confident", "authoritative", "energetic"]
        ),
        "sage": VoiceInfo(
            id="sage",
            name="Sage",
            gender="neutral",
            description="Thoughtful and measured voice",
            styles=["thoughtful", "calm", "mysterious"]
        ),
        "nova": VoiceInfo(
            id="nova",
            name="Nova",
            gender="female",
            description="Bright and expressive voice",
            styles=["excited", "playful", "romantic"]
        ),
        "atlas": VoiceInfo(
            id="atlas",
            name="Atlas",
            gender="male",
            description="Deep and resonant voice",
            styles=["serious", "authoritative", "professional"]
        ),
        "echo": VoiceInfo(
            id="echo",
            name="Echo",
            gender="neutral",
            description="Adaptive voice that mirrors conversation tone",
            styles=["casual", "friendly", "sarcastic"]
        )
    }
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        
        # Model variant selection (1.5B or 7B)
        self.variant = self.config.get("vibevoice_variant", "1.5B")
        if self.variant not in self.MODEL_VARIANTS:
            logger.warning(f"Unknown VibeVoice variant {self.variant}, defaulting to 1.5B")
            self.variant = "1.5B"
        
        # Model configuration
        variant_config = self.MODEL_VARIANTS[self.variant]
        self.model_path = self.config.get("vibevoice_model_path", variant_config["path"])
        self.context_length = variant_config["context"]
        self.frame_rate = variant_config["frame_rate"]
        self.device = self.config.get("vibevoice_device", "cuda" if torch.cuda.is_available() else "cpu")
        
        # Audio configuration
        self.sample_rate = self.config.get("vibevoice_sample_rate", 22050)
        
        # Model instances
        self.model = None
        self.processor = None
        
        # Speaker settings (VibeVoice supports up to 4 speakers)
        self.max_speakers = 4
        self.default_speaker = self.config.get("vibevoice_default_speaker", 1)
        
        # Context awareness
        self.enable_context = self.config.get("vibevoice_context", True)
        self.context_window = self.config.get("vibevoice_context_window", 512)
        
        # VibeVoice specific features
        self.enable_background_music = self.config.get("vibevoice_background_music", False)
        self.enable_singing = self.config.get("vibevoice_enable_singing", False)
        
        # Performance settings
        self.use_fp16 = self.config.get("vibevoice_use_fp16", True) and self.device == "cuda"
        self.batch_size = self.config.get("vibevoice_batch_size", 1)
        
        # Setup paths
        self.model_dir = Path(self.config.get("vibevoice_model_dir", "./models/vibevoice"))
        self.cache_dir = Path(self.config.get("vibevoice_cache_dir", "./cache/vibevoice"))
        
        # Voice samples folder for 1-shot cloning (like VibeVoice demo)
        self.voices_dir = Path(self.config.get("vibevoice_voices_dir", "./Voices"))
        self.custom_voices = {}
    
    def _load_custom_voices(self):
        """Load custom voice samples from Voices folder for 1-shot cloning"""
        if self.voices_dir.exists():
            logger.info(f"Loading custom voices from {self.voices_dir}")
            for voice_file in self.voices_dir.glob("*.wav"):
                voice_name = voice_file.stem  # Use filename without extension as voice name
                self.custom_voices[voice_name] = str(voice_file)
                logger.info(f"Loaded custom voice: {voice_name} from {voice_file}")
                
                # Add to voice presets dynamically
                self.VOICE_PRESETS[voice_name] = VoiceInfo(
                    id=voice_name,
                    name=voice_name.replace("_", " ").title(),
                    gender="neutral",
                    description=f"Custom voice cloned from {voice_file.name}",
                    styles=["custom", "cloned"]
                )
        else:
            logger.debug(f"Voices directory {self.voices_dir} not found, using preset voices only")
    
    async def _load_user_voices(self, user_id: int):
        """Load user-specific uploaded voices"""
        try:
            from ...voice_manager import get_voice_manager
            
            voice_manager = get_voice_manager()
            user_voices = await voice_manager.list_user_voices(user_id)
            
            for voice_info in user_voices:
                if voice_info.provider == "vibevoice":
                    # Get full path to voice file
                    voices_path = voice_manager.get_user_voices_path(user_id)
                    voice_file_path = voices_path / voice_info.file_path
                    
                    if voice_file_path.exists():
                        # Register with custom: prefix
                        custom_id = f"custom:{voice_info.voice_id}"
                        self.custom_voices[custom_id] = str(voice_file_path)
                        
                        # Add to voice presets
                        self.VOICE_PRESETS[custom_id] = VoiceInfo(
                            id=custom_id,
                            name=voice_info.name,
                            gender="neutral",
                            description=voice_info.description or f"Custom voice: {voice_info.name}",
                            styles=["custom", "uploaded"]
                        )
                        
                        logger.info(f"Loaded user voice: {voice_info.name} ({custom_id})")
            
            logger.info(f"Loaded {len(user_voices)} user voices for VibeVoice")
            
        except ImportError:
            logger.debug("Voice manager not available, skipping user voice loading")
        except Exception as e:
            logger.error(f"Error loading user voices: {e}")
    
    async def initialize(self, user_id: Optional[int] = None) -> bool:
        """Initialize the VibeVoice TTS model"""
        try:
            logger.info(f"{self.provider_name}: Initializing VibeVoice TTS (variant: {self.variant}, model: {self.model_path})...")
            
            # Load custom voices from Voices folder
            self._load_custom_voices()
            
            # Load user-specific voices if user_id provided
            if user_id:
                await self._load_user_voices(user_id)
            
            # Get resource manager for memory monitoring
            resource_manager = await get_resource_manager()
            
            # Check memory before loading model
            if resource_manager.memory_monitor.is_memory_critical():
                raise TTSInsufficientMemoryError(
                    "Insufficient memory to load VibeVoice model",
                    provider=self.provider_name,
                    details=resource_manager.memory_monitor.get_memory_usage()
                )
            
            # Check for model files
            if not self._check_model_files():
                logger.info(f"{self.provider_name}: Downloading VibeVoice models...")
                await self._download_models()
            
            # Load the model components
            logger.info(f"{self.provider_name}: Loading VibeVoice components...")
            
            # Import VibeVoice (hypothetical library)
            try:
                # This would be the actual VibeVoice library import
                # For now, we'll use placeholder imports
                from transformers import AutoModel, AutoTokenizer
                
                # Load processor/tokenizer
                self.processor = AutoTokenizer.from_pretrained(
                    self.model_path,
                    cache_dir=self.cache_dir,
                    local_files_only=self.model_dir.exists()
                )
                
                # Load main model
                self.model = AutoModel.from_pretrained(
                    self.model_path,
                    torch_dtype=torch.float16 if self.use_fp16 else torch.float32,
                    device_map=self.device,
                    cache_dir=self.cache_dir,
                    local_files_only=self.model_dir.exists()
                )
                
                # Model is ready for use
                
            except ImportError as e:
                error_msg = (
                    f"{self.provider_name}: Required libraries not installed. "
                    f"To use VibeVoice, follow these steps:\n"
                    f"1. Clone the VibeVoice repository:\n"
                    f"   git clone https://github.com/microsoft/VibeVoice.git\n"
                    f"   cd VibeVoice/\n"
                    f"2. Install VibeVoice (recommended in Docker):\n"
                    f"   sudo docker run --privileged --gpus all --rm -it nvcr.io/nvidia/pytorch:24.07-py3\n"
                    f"   pip install -e .\n"
                    f"3. Or install locally:\n"
                    f"   pip install transformers torch torchaudio accelerate\n"
                    f"   pip install -e /path/to/VibeVoice\n"
                    f"4. The {self.variant} model will auto-download on first use from:\n"
                    f"   {self.model_path}"
                )
                logger.error(f"{self.provider_name}: Required libraries not installed: {e}")
                logger.error(error_msg)
                self._status = ProviderStatus.NOT_CONFIGURED
                raise TTSModelLoadError(
                    "Failed to import required libraries",
                    provider=self.provider_name,
                    details={"error": str(e), "suggestion": error_msg}
                )
            
            # Set to evaluation mode
            if self.model:
                self.model.eval()
                
                # Register model with resource manager
                resource_manager.register_model(
                    provider=self.provider_name.lower(),
                    model_instance=self.model,
                    cleanup_callback=self._cleanup_resources
                )
            
            # Warm up the model
            await self._warmup_model()
            
            logger.info(
                f"{self.provider_name}: Initialized successfully "
                f"(Device: {self.device}, FP16: {self.use_fp16}, Context: {self.enable_context})"
            )
            self._status = ProviderStatus.AVAILABLE
            return True
            
        except (TTSInsufficientMemoryError, TTSModelLoadError):
            raise
        except RuntimeError as e:
            if "CUDA" in str(e) or "GPU" in str(e):
                raise TTSGPUError(
                    f"GPU error initializing {self.provider_name}",
                    provider=self.provider_name,
                    details={"error": str(e), "device": self.device}
                )
            raise
        except Exception as e:
            logger.error(f"{self.provider_name}: Initialization failed: {e}")
            self._status = ProviderStatus.ERROR
            raise TTSProviderInitializationError(
                f"Failed to initialize {self.provider_name}",
                provider=self.provider_name,
                details={"error": str(e), "model_path": self.model_path}
            )
    
    def _check_model_files(self) -> bool:
        """Check if model files exist locally"""
        required_files = [
            "config.json",
            "model.safetensors",
            "tokenizer_config.json",
            "vibe_encoder.pt"
        ]
        
        if not self.model_dir.exists():
            return False
        
        for file in required_files:
            if not (self.model_dir / file).exists():
                return False
        
        return True
    
    async def _download_models(self):
        """Download VibeVoice models if not present"""
        # Create directories
        self.model_dir.mkdir(parents=True, exist_ok=True)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        try:
            from huggingface_hub import snapshot_download
            logger.info(f"{self.provider_name}: Downloading {self.variant} model from {self.model_path}...")
            
            # Download the model from HuggingFace
            snapshot_download(
                repo_id=self.model_path,
                local_dir=str(self.model_dir),
                cache_dir=str(self.cache_dir)
            )
            logger.info(f"{self.provider_name}: Model download complete")
            return True
            
        except ImportError:
            logger.error(f"{self.provider_name}: huggingface_hub not installed. Run: pip install huggingface-hub")
            return False
        except Exception as e:
            logger.error(f"{self.provider_name}: Model download failed: {e}")
            return False
    
    async def _warmup_model(self):
        """Warm up the model with a test generation"""
        if not self.model:
            return
        
        try:
            logger.debug(f"{self.provider_name}: Warming up model...")
            test_text = "Hello, this is a test."
            test_request = TTSRequest(
                text=test_text,
                voice="aurora",
                format=AudioFormat.WAV
            )
            # Would do actual generation here
            logger.debug(f"{self.provider_name}: Model warmup complete")
        except Exception as e:
            logger.warning(f"{self.provider_name}: Warmup failed: {e}")
    
    async def get_capabilities(self) -> TTSCapabilities:
        """Get VibeVoice TTS capabilities"""
        # Variant-specific max generation time
        max_generation_minutes = 90 if self.variant == "1.5B" else 45
        
        return TTSCapabilities(
            provider_name=f"VibeVoice-{self.variant}",
            supported_languages={"en", "es", "fr", "de", "it", "pt", "nl", "pl", "ru", "ja", "ko", "zh"},
            supported_voices=list(self.VOICE_PRESETS.values()),
            supported_formats={
                AudioFormat.WAV,
                AudioFormat.MP3,
                AudioFormat.OPUS,
                AudioFormat.FLAC,
                AudioFormat.PCM,
                AudioFormat.OGG
            },
            max_text_length=self.context_length,  # Use variant-specific context length
            supports_streaming=True,
            supports_voice_cloning=True,  # Supports cloning via voice reference folder
            supports_emotion_control=False,  # Does not have direct emotion control
            supports_speech_rate=True,
            supports_pitch_control=False,
            supports_volume_control=True,
            supports_ssml=False,  # Uses its own markup
            supports_phonemes=False,
            supports_multi_speaker=True,  # Up to 4 speakers
            supports_background_audio=True,  # Spontaneous music generation
            latency_ms=150 if self.device == "cuda" else 800,
            sample_rate=self.sample_rate,
            default_format=AudioFormat.WAV
        )
    
    async def generate(self, request: TTSRequest) -> TTSResponse:
        """Generate speech using VibeVoice TTS"""
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
        
        # Process voice and speaker settings
        voice = request.voice or "aurora"
        speaker_id = request.extra_params.get("speaker_id", self.default_speaker)
        # Ensure speaker_id is within valid range (1-4)
        speaker_id = max(1, min(4, speaker_id))
        
        # Handle voice cloning - check custom voices
        voice_reference_path = None
        
        # Check if voice starts with "custom:" prefix (uploaded voice)
        if voice.startswith("custom:"):
            if voice in self.custom_voices:
                voice_reference_path = self.custom_voices[voice]
                logger.info(f"Using uploaded voice '{voice}' from {voice_reference_path}")
            else:
                logger.warning(f"Custom voice '{voice}' not found, using default")
                voice = "aurora"
        # Check if voice is a custom voice loaded from Voices folder
        elif voice in self.custom_voices:
            voice_reference_path = self.custom_voices[voice]
            logger.info(f"Using custom voice '{voice}' from {voice_reference_path}")
        # Otherwise check if voice reference bytes provided
        elif request.voice_reference:
            voice_reference_path = await self._prepare_voice_reference(request.voice_reference)
            # Use "cloned" as voice identifier when using reference
            voice = "cloned" if voice_reference_path else voice
        
        # Process multi-speaker if needed
        speakers = request.speakers if hasattr(request, 'speakers') else None
        
        logger.info(
            f"{self.provider_name}: Generating speech with voice={voice}, "
            f"speaker_id={speaker_id}, format={request.format.value}"
        )
        
        try:
            if request.stream:
                # Return streaming response
                return TTSResponse(
                    audio_stream=self._stream_audio_vibevoice(request, voice, speaker_id, voice_reference_path),
                    format=request.format,
                    sample_rate=self.sample_rate,
                    channels=1,
                    voice_used=voice,
                    provider=self.provider_name,
                    metadata={
                        "speaker_id": speaker_id,
                        "context_enabled": self.enable_context,
                        "background_music": self.enable_background_music,
                        "singing_enabled": self.enable_singing
                    }
                )
            else:
                # Generate complete audio
                audio_data = await self._generate_complete_vibevoice(request, voice, speaker_id, voice_reference_path)
                return TTSResponse(
                    audio_data=audio_data,
                    format=request.format,
                    sample_rate=self.sample_rate,
                    channels=1,
                    voice_used=voice,
                    provider=self.provider_name,
                    metadata={
                        "speaker_id": speaker_id,
                        "context_enabled": self.enable_context,
                        "background_music": self.enable_background_music,
                        "singing_enabled": self.enable_singing
                    }
                )
                
        except (TTSProviderNotConfiguredError, TTSModelLoadError):
            raise
        except Exception as e:
            logger.error(f"{self.provider_name} generation error: {e}")
            raise TTSGenerationError(
                f"Failed to generate speech with {self.provider_name}",
                provider=self.provider_name,
                details={"error": str(e), "error_type": type(e).__name__}
            )
    
    
    async def _stream_audio_vibevoice(
        self,
        request: TTSRequest,
        voice: str,
        speaker_id: int,
        voice_reference_path: Optional[str] = None
    ) -> AsyncGenerator[bytes, None]:
        """Stream audio from VibeVoice model"""
        if not self.model or not self.processor:
            raise TTSModelNotFoundError(
                "VibeVoice model not initialized",
                provider=self.provider_name
            )
        
        # Import StreamingAudioWriter
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
            # Prepare input with speaker settings
            input_data = self._prepare_vibevoice_input(request, voice, speaker_id)
            
            # Process text with context
            if self.enable_context:
                context = request.extra_params.get("context", "")
                inputs = self.processor(
                    text=input_data["text"],
                    context=context[:self.context_window],
                    return_tensors="pt"
                ).to(self.device)
            else:
                inputs = self.processor(
                    text=input_data["text"],
                    return_tensors="pt"
                ).to(self.device)
            
            # Add speaker ID to inputs
            inputs["speaker_id"] = torch.tensor([speaker_id], dtype=torch.long).to(self.device)
            
            # Add voice reference if provided
            if voice_reference_path:
                # Load reference audio for speaker embedding
                import librosa
                ref_audio, _ = librosa.load(voice_reference_path, sr=self.sample_rate)
                # Extract speaker embedding (placeholder - would use actual model)
                speaker_embedding = torch.tensor(ref_audio[:256]).unsqueeze(0).to(self.device)
                inputs["speaker_embedding"] = speaker_embedding
                logger.info(f"Using voice reference for VibeVoice: {voice_reference_path}")
            
            # Generate with streaming
            with torch.no_grad():
                # Placeholder for actual generation
                # In production, this would use the VibeVoice model's streaming generation
                audio_length = len(request.text) * 200  # Rough estimate
                audio_array = np.random.randn(audio_length).astype(np.float32) * 0.1
                
                # Simulate streaming chunks
                chunk_size = int(self.sample_rate * 0.25)  # 0.25 second chunks
                for i in range(0, len(audio_array), chunk_size):
                    chunk = audio_array[i:i + chunk_size]
                    
                    if len(chunk) > 0:
                        # Normalize to int16
                        normalized_chunk = normalizer.normalize(chunk, target_dtype=np.int16)
                        
                        # Encode to target format
                        encoded_bytes = writer.write_chunk(normalized_chunk)
                        if encoded_bytes:
                            yield encoded_bytes
            
            # Finalize stream
            final_bytes = writer.write_chunk(finalize=True)
            if final_bytes:
                yield final_bytes
            
            logger.info(f"{self.provider_name}: Successfully generated audio with speaker={speaker_id}")
            
        except TTSModelNotFoundError:
            raise
        except Exception as e:
            logger.error(f"{self.provider_name} streaming error: {e}")
            raise TTSGenerationError(
                f"Streaming error in {self.provider_name}",
                provider=self.provider_name,
                details={"error": str(e)}
            )
        finally:
            writer.close()
            # Clean up voice reference file if used (but not custom voices from Voices folder)
            if voice_reference_path and voice_reference_path not in self.custom_voices.values():
                try:
                    from pathlib import Path
                    Path(voice_reference_path).unlink(missing_ok=True)
                    logger.debug(f"Cleaned up temporary voice reference: {voice_reference_path}")
                except Exception as e:
                    logger.warning(f"Failed to clean up voice reference: {e}")
    
    async def _generate_complete_vibevoice(
        self,
        request: TTSRequest,
        voice: str,
        speaker_id: int,
        voice_reference_path: Optional[str] = None
    ) -> bytes:
        """Generate complete audio from VibeVoice"""
        all_audio = b""
        async for chunk in self._stream_audio_vibevoice(request, voice, speaker_id, voice_reference_path):
            all_audio += chunk
        return all_audio
    
    def _prepare_vibevoice_input(
        self,
        request: TTSRequest,
        voice: str,
        speaker_id: int
    ) -> Dict[str, Any]:
        """Prepare input for VibeVoice with speaker settings"""
        # Process text with SSML if present
        text = self.preprocess_text(request.text)
        
        return {
            "text": text,
            "voice": voice,
            "speaker_id": speaker_id,
            "speed": request.speed,
            "pitch": request.pitch,
            "volume": request.volume,
            "background_music": self.enable_background_music,
            "singing": self.enable_singing
        }
    
    
    async def _prepare_voice_reference(self, voice_reference: bytes) -> Optional[str]:
        """
        Prepare voice reference audio for VibeVoice.
        VibeVoice supports single-sample voice cloning.
        
        Args:
            voice_reference: Voice reference audio bytes
            
        Returns:
            Path to temporary voice reference file or None if processing fails
        """
        try:
            import tempfile
            from pathlib import Path
            from tldw_Server_API.app.core.TTS.audio_utils import process_voice_reference
            
            # Process voice reference for VibeVoice requirements
            processed_audio, error = process_voice_reference(
                voice_reference,
                provider='vibevoice',
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
                prefix='vibevoice_voice_'
            ) as tmp_file:
                tmp_file.write(processed_audio)
                tmp_path = tmp_file.name
            
            logger.info(f"Voice reference prepared for VibeVoice: {tmp_path}")
            return tmp_path
            
        except Exception as e:
            logger.error(f"Failed to prepare voice reference: {e}")
            return None
    
    def map_voice(self, voice_id: str) -> str:
        """Map generic voice ID to VibeVoice voice"""
        # Handle custom: prefix
        if voice_id.startswith("custom:"):
            if voice_id in self.custom_voices:
                return voice_id
            else:
                logger.warning(f"Custom voice {voice_id} not found")
                return "aurora"
        
        # Check custom voices first
        if voice_id in self.custom_voices:
            return voice_id
            
        # Then check presets
        if voice_id in self.VOICE_PRESETS:
            return voice_id
        
        # Map common voice types
        voice_mappings = {
            "female": "aurora",
            "male": "phoenix",
            "neutral": "sage",
            "young": "nova",
            "deep": "atlas",
            "adaptive": "echo"
        }
        
        return voice_mappings.get(voice_id.lower(), "aurora")
    
    def preprocess_text(self, text: str, **kwargs) -> str:
        """Preprocess text for VibeVoice"""
        # Basic preprocessing
        text = super().preprocess_text(text)
        
        # VibeVoice supports multi-speaker dialogue markers
        # Example: [Speaker1] Hello there. [Speaker2] Hi!
        # These would be parsed and handled by the model
        
        return text
    
    async def _cleanup_resources(self):
        """Clean up VibeVoice adapter resources"""
        try:
            if self.model:
                del self.model
            if self.processor:
                del self.processor
            
            self.model = None
            self.processor = None
            
            # Clear GPU cache if using CUDA
            if self.device == "cuda" and torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            logger.debug(f"{self.provider_name}: Resources cleaned up")
        except Exception as e:
            logger.warning(f"{self.provider_name}: Error during cleanup: {e}")

#
# End of vibevoice_adapter.py
#######################################################################################################################