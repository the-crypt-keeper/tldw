# higgs_adapter.py
# Description: Higgs Audio V2 TTS adapter implementation
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
    TTSInsufficientMemoryError,
    TTSGPUError
)
from ..tts_validation import validate_tts_request
from ..tts_resource_manager import get_resource_manager
#
#######################################################################################################################
#
# Higgs Audio V2 TTS Adapter Implementation

class HiggsAdapter(TTSAdapter):
    """Adapter for Higgs Audio V2 TTS model from Boson AI"""
    
    # Supported languages (50+ languages)
    SUPPORTED_LANGUAGES = {
        "en", "zh", "es", "fr", "de", "ja", "ko", "ru", "it", "pt",
        "nl", "pl", "tr", "ar", "hi", "id", "ms", "th", "vi", "sv",
        "da", "no", "fi", "cs", "el", "he", "hu", "ro", "sk", "uk",
        "bg", "hr", "sr", "sl", "lt", "lv", "et", "sq", "mk", "ca",
        "eu", "gl", "is", "ga", "cy", "mt", "lb", "yi", "ur", "fa"
    }
    
    # Voice presets (can be extended with voice cloning)
    VOICE_PRESETS = {
        "narrator": VoiceInfo(
            id="narrator",
            name="Narrator",
            gender="neutral",
            description="Professional narration voice",
            use_case=["audiobook", "documentation"]
        ),
        "conversational": VoiceInfo(
            id="conversational",
            name="Conversational",
            gender="neutral",
            description="Natural conversational voice",
            use_case=["dialogue", "chat"]
        ),
        "expressive": VoiceInfo(
            id="expressive",
            name="Expressive",
            gender="neutral",
            description="Emotionally expressive voice",
            use_case=["storytelling", "drama"]
        ),
        "melodic": VoiceInfo(
            id="melodic",
            name="Melodic",
            gender="neutral",
            description="Musical and melodic voice",
            use_case=["singing", "humming"]
        )
    }
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        
        # Model configuration
        self.model_path = self.config.get(
            "higgs_model_path",
            "bosonai/higgs-audio-v2-generation-3B-base"
        )
        self.tokenizer_path = self.config.get(
            "higgs_tokenizer_path",
            "bosonai/higgs-audio-v2-tokenizer"
        )
        self.device = self.config.get("higgs_device", "cuda" if torch.cuda.is_available() else "cpu")
        
        # Audio configuration (24kHz for Higgs V2)
        self.sample_rate = 24000
        self.frame_rate = 25  # 25 frames per second tokenizer
        
        # Model instances
        self.model = None
        self.tokenizer = None
        self.audio_tokenizer = None
        
        # Performance settings
        self.use_fp16 = self.config.get("higgs_use_fp16", True) and self.device == "cuda"
        self.batch_size = self.config.get("higgs_batch_size", 1)
    
    async def initialize(self) -> bool:
        """Initialize the Higgs Audio V2 model"""
        try:
            logger.info(f"{self.provider_name}: Loading Higgs Audio V2 model...")
            
            # Get resource manager for memory monitoring
            resource_manager = await get_resource_manager()
            
            # Check memory before loading model
            if resource_manager.memory_monitor.is_memory_critical():
                raise TTSInsufficientMemoryError(
                    "Insufficient memory to load Higgs model",
                    provider=self.provider_name,
                    details=resource_manager.memory_monitor.get_memory_usage()
                )
            
            # Check if boson_multimodal library is available
            try:
                from boson_multimodal.serve.serve_engine import HiggsAudioServeEngine
            except ImportError as e:
                logger.error(f"{self.provider_name}: boson_multimodal library not installed")
                logger.info("Install Higgs Audio dependencies from: https://github.com/boson-ai/higgs-audio")
                self._status = ProviderStatus.NOT_CONFIGURED
                raise TTSModelLoadError(
                    "Failed to import boson_multimodal library",
                    provider=self.provider_name,
                    details={"error": str(e), "suggestion": "Install from https://github.com/boson-ai/higgs-audio"}
                )
            
            # Initialize HiggsAudioServeEngine
            logger.info(f"{self.provider_name}: Initializing HiggsAudioServeEngine...")
            self.serve_engine = HiggsAudioServeEngine(
                model_path=self.model_path,
                audio_tokenizer_path=self.tokenizer_path,
                device=self.device
            )
            
            # Register model with resource manager
            if self.serve_engine:
                resource_manager.register_model(
                    provider=self.provider_name.lower(),
                    model_instance=self.serve_engine,
                    cleanup_callback=self._cleanup_resources
                )
            
            logger.info(
                f"{self.provider_name}: Initialized successfully "
                f"(Device: {self.device})"
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
    
    async def get_capabilities(self) -> TTSCapabilities:
        """Get Higgs Audio V2 capabilities"""
        return TTSCapabilities(
            provider_name="Higgs",
            supported_languages=self.SUPPORTED_LANGUAGES,
            supported_voices=list(self.VOICE_PRESETS.values()),
            supported_formats={
                AudioFormat.WAV,
                AudioFormat.MP3,
                AudioFormat.OPUS,
                AudioFormat.FLAC,
                AudioFormat.PCM
            },
            max_text_length=50000,  # Higgs can handle very long texts
            supports_streaming=True,
            supports_voice_cloning=True,  # 3-10 seconds of audio needed
            supports_emotion_control=True,  # Advanced emotion control
            supports_speech_rate=True,
            supports_pitch_control=True,
            supports_volume_control=True,
            supports_ssml=False,
            supports_phonemes=True,
            supports_multi_speaker=True,  # Natural multi-speaker dialogues
            supports_background_audio=True,  # Can generate speech with background music
            latency_ms=200 if self.device == "cuda" else 2000,
            sample_rate=24000,
            default_format=AudioFormat.WAV
        )
    
    async def generate(self, request: TTSRequest) -> TTSResponse:
        """Generate speech using Higgs Audio V2"""
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
        
        # Prepare generation parameters
        voice = request.voice or "conversational"
        
        # Handle voice cloning if reference provided
        voice_reference_path = None
        if request.voice_reference:
            voice_reference_path = await self._prepare_voice_reference(request.voice_reference)
            # Use "cloned" as voice identifier when using reference
            voice = "cloned" if voice_reference_path else voice
        
        logger.info(
            f"{self.provider_name}: Generating speech with voice={voice}, "
            f"language={request.language}, format={request.format.value}"
        )
        
        try:
            if request.stream:
                # Return streaming response
                return TTSResponse(
                    audio_stream=self._stream_audio_higgs(request, voice_reference_path),
                    format=request.format,
                    sample_rate=self.sample_rate,
                    channels=1,
                    voice_used=voice,
                    provider=self.provider_name
                )
            else:
                # Generate complete audio
                audio_data = await self._generate_complete_higgs(request, voice_reference_path)
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
    
    async def _stream_audio_higgs(
        self,
        request: TTSRequest,
        voice_reference_path: Optional[str] = None
    ) -> AsyncGenerator[bytes, None]:
        """Stream audio from Higgs model"""
        if not hasattr(self, 'serve_engine') or not self.serve_engine:
            raise ValueError("Higgs serve engine not initialized")
        
        # Import required modules
        from boson_multimodal.serve.serve_engine import HiggsAudioResponse
        from tldw_Server_API.app.core.TTS.streaming_audio_writer import (
            StreamingAudioWriter,
            AudioNormalizer
        )
        import torchaudio
        
        normalizer = AudioNormalizer()
        writer = StreamingAudioWriter(
            format=request.format.value,
            sample_rate=self.sample_rate,
            channels=1
        )
        
        try:
            # Prepare ChatML format for Higgs
            chat_ml_sample = self._prepare_higgs_chat_ml(request, voice_reference_path)
            
            # Generate with HiggsAudioServeEngine
            logger.info(f"{self.provider_name}: Starting generation...")
            output: HiggsAudioResponse = self.serve_engine.generate(
                chat_ml_sample=chat_ml_sample,
                max_new_tokens=1024,
                temperature=request.extra_params.get("temperature", 1.0),
                top_p=request.extra_params.get("top_p", 0.95),
                top_k=request.extra_params.get("top_k", 50),
                stop_strings=["<|end_of_text|>", "<|eot_id|>"]
            )
            
            # Convert numpy audio to tensor and back to numpy for processing
            audio_array = output.audio
            
            # Process audio in chunks for streaming
            chunk_size = int(self.sample_rate * 0.5)  # 0.5 second chunks
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
            
            logger.info(f"{self.provider_name}: Successfully generated audio")
            logger.debug(f"Generated text: {output.generated_text}")
            
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
    
    async def _generate_complete_higgs(
        self,
        request: TTSRequest,
        voice_reference_path: Optional[str] = None
    ) -> bytes:
        """Generate complete audio from Higgs"""
        # Collect all streamed chunks
        all_audio = b""
        async for chunk in self._stream_audio_higgs(request, voice_reference_path):
            all_audio += chunk
        return all_audio
    
    def _prepare_higgs_chat_ml(
        self,
        request: TTSRequest,
        voice_reference_path: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Prepare ChatML format input for Higgs.
        Higgs uses a specific format for generation.
        """
        # Build the ChatML structure
        messages = []
        
        # System message if needed
        if request.extra_params.get("system_prompt"):
            messages.append({
                "role": "system",
                "content": request.extra_params["system_prompt"]
            })
        
        # User message with the text to generate
        user_content = request.text
        
        # Add language instruction if not English
        if request.language and request.language != "en":
            user_content = f"Please generate speech in {request.language}: {user_content}"
        
        # Add emotion instruction if specified
        if request.emotion:
            intensity_desc = "strongly" if request.emotion_intensity > 1.5 else "moderately" if request.emotion_intensity > 0.5 else "slightly"
            user_content = f"Say this {intensity_desc} {request.emotion}: {user_content}"
        
        # Add style instruction
        if request.style:
            user_content = f"In a {request.style} style: {user_content}"
        
        # Handle multi-speaker dialogue
        if request.speakers:
            user_content = f"Generate a dialogue with multiple speakers: {user_content}"
        
        messages.append({
            "role": "user", 
            "content": user_content
        })
        
        # Prepare the result
        result = {
            "messages": messages,
            "voice": request.voice or "narrator",
            "speed": request.speed,
            "seed": request.seed
        }
        
        # Add voice reference if provided
        if voice_reference_path:
            result["reference_audio_path"] = voice_reference_path
            result["voice"] = "cloned"  # Override voice when using reference
            logger.info(f"Added voice reference to Higgs ChatML: {voice_reference_path}")
        
        return result
    
    async def _prepare_voice_reference(self, voice_reference: bytes) -> Optional[str]:
        """
        Prepare voice reference audio for Higgs.
        Higgs needs 3-10 seconds of audio at 24kHz.
        
        Args:
            voice_reference: Voice reference audio bytes
            
        Returns:
            Path to temporary voice reference file or None if processing fails
        """
        try:
            import tempfile
            from pathlib import Path
            from tldw_Server_API.app.core.TTS.audio_utils import process_voice_reference
            
            # Process voice reference for Higgs requirements
            processed_audio, error = process_voice_reference(
                voice_reference,
                provider='higgs',
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
                prefix='higgs_voice_'
            ) as tmp_file:
                tmp_file.write(processed_audio)
                tmp_path = tmp_file.name
            
            logger.info(f"Voice reference prepared for Higgs: {tmp_path}")
            return tmp_path
            
        except Exception as e:
            logger.error(f"Failed to prepare voice reference: {e}")
            return None
    
    def map_voice(self, voice_id: str) -> str:
        """Map generic voice ID to Higgs voice preset"""
        if voice_id in self.VOICE_PRESETS:
            return voice_id
        
        # Map common voice types
        voice_mappings = {
            "default": "conversational",
            "assistant": "conversational",
            "narrator": "narrator",
            "expressive": "expressive",
            "emotional": "expressive",
            "singing": "melodic",
            "musical": "melodic"
        }
        
        return voice_mappings.get(voice_id.lower(), "conversational")
    
    async def close(self):
        """Clean up resources"""
        if hasattr(self, 'serve_engine') and self.serve_engine:
            del self.serve_engine
            self.serve_engine = None
        
        # Clear GPU cache if using CUDA
        if self.device == "cuda" and torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        await super().close()

#
# End of higgs_adapter.py
#######################################################################################################################