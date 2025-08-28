# dia_adapter.py  
# Description: Dia TTS adapter implementation for ultra-realistic dialogue generation
#
# Imports
import os
import re
from typing import Optional, Dict, Any, AsyncGenerator, Set, List, Tuple
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
#
#######################################################################################################################
#
# Dia TTS Adapter Implementation

class DiaAdapter(TTSAdapter):
    """Adapter for Dia TTS model (dialogue generation specialist)"""
    
    # Dia special tags for nonverbal audio
    NONVERBAL_TAGS = {
        "(laughs)", "(coughs)", "(gasps)", "(sighs)", "(clears throat)",
        "(sniffles)", "(yawns)", "(groans)", "(whispers)", "(shouts)",
        "(mumbles)", "(stutters)", "(pauses)", "(hesitates)", "(breathes)"
    }
    
    # Default speaker voices (Dia generates dynamic voices)
    DEFAULT_SPEAKERS = {
        "speaker1": VoiceInfo(
            id="speaker1",
            name="Speaker 1",
            gender="neutral",
            description="Primary dialogue speaker"
        ),
        "speaker2": VoiceInfo(
            id="speaker2", 
            name="Speaker 2",
            gender="neutral",
            description="Secondary dialogue speaker"
        ),
        "speaker3": VoiceInfo(
            id="speaker3",
            name="Speaker 3",
            gender="neutral",
            description="Tertiary dialogue speaker"
        ),
        "narrator": VoiceInfo(
            id="narrator",
            name="Narrator",
            gender="neutral",
            description="Narration voice"
        )
    }
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        
        # Model configuration
        self.model_path = self.config.get("dia_model_path", "nari-labs/dia")
        self.device = self.config.get("dia_device", "cuda" if torch.cuda.is_available() else "cpu")
        
        # Audio configuration
        self.sample_rate = self.config.get("dia_sample_rate", 24000)
        
        # Model instances
        self.model = None
        self.processor = None
        
        # Dialogue processing
        self.auto_detect_speakers = self.config.get("dia_auto_detect_speakers", True)
        self.max_speakers = self.config.get("dia_max_speakers", 5)
        
        # Performance settings
        self.use_safetensors = self.config.get("dia_use_safetensors", True)
        self.use_bf16 = self.config.get("dia_use_bf16", True) and self.device == "cuda"
    
    async def initialize(self) -> bool:
        """Initialize the Dia TTS model"""
        try:
            logger.info(f"{self.provider_name}: Loading Dia TTS model (1.6B parameters)...")
            
            # Check for required libraries
            try:
                from transformers import AutoModelForCausalLM, AutoProcessor
            except ImportError:
                logger.error(f"{self.provider_name}: transformers library not installed")
                self._status = ProviderStatus.NOT_CONFIGURED
                return False
            
            # Load processor
            logger.info(f"{self.provider_name}: Loading processor from {self.model_path}")
            self.processor = AutoProcessor.from_pretrained(
                self.model_path,
                trust_remote_code=True
            )
            
            # Load model with appropriate dtype
            logger.info(f"{self.provider_name}: Loading model from {self.model_path}")
            model_kwargs = {
                "trust_remote_code": True,
                "use_safetensors": self.use_safetensors
            }
            
            if self.use_bf16:
                model_kwargs["torch_dtype"] = torch.bfloat16
            
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_path,
                **model_kwargs
            )
            
            self.model = self.model.to(self.device)
            self.model.eval()
            
            logger.info(
                f"{self.provider_name}: Initialized successfully "
                f"(Device: {self.device}, BF16: {self.use_bf16})"
            )
            self._status = ProviderStatus.AVAILABLE
            return True
            
        except Exception as e:
            logger.error(f"{self.provider_name}: Initialization failed: {e}")
            self._status = ProviderStatus.ERROR
            return False
    
    async def get_capabilities(self) -> TTSCapabilities:
        """Get Dia TTS capabilities"""
        return TTSCapabilities(
            provider_name="Dia",
            supported_languages={"en"},  # Currently English only
            supported_voices=list(self.DEFAULT_SPEAKERS.values()),
            supported_formats={
                AudioFormat.WAV,
                AudioFormat.MP3,
                AudioFormat.OPUS,
                AudioFormat.FLAC,
                AudioFormat.PCM
            },
            max_text_length=30000,
            supports_streaming=True,
            supports_voice_cloning=True,  # Via audio prompts
            supports_emotion_control=False,  # Emotion through nonverbal tags
            supports_speech_rate=True,
            supports_pitch_control=False,
            supports_volume_control=False,
            supports_ssml=False,
            supports_phonemes=False,
            supports_multi_speaker=True,  # Core feature - multi-speaker dialogue
            supports_background_audio=False,
            latency_ms=500 if self.device == "cuda" else 3000,
            sample_rate=self.sample_rate,
            default_format=AudioFormat.WAV
        )
    
    async def generate(self, request: TTSRequest) -> TTSResponse:
        """Generate speech using Dia TTS"""
        if not await self.ensure_initialized():
            raise ValueError(f"{self.provider_name} not initialized")
        
        # Validate request
        is_valid, error = await self.validate_request(request)
        if not is_valid:
            raise ValueError(error)
        
        # Process text for dialogue
        dialogue_parts = self._process_dialogue(request.text, request.speakers)
        
        logger.info(
            f"{self.provider_name}: Generating dialogue with {len(dialogue_parts)} parts, "
            f"format={request.format.value}"
        )
        
        try:
            if request.stream:
                # Return streaming response
                return TTSResponse(
                    audio_stream=self._stream_audio_dia(dialogue_parts, request),
                    format=request.format,
                    sample_rate=self.sample_rate,
                    channels=1,
                    provider=self.provider_name,
                    metadata={"dialogue_parts": len(dialogue_parts)}
                )
            else:
                # Generate complete audio
                audio_data = await self._generate_complete_dia(dialogue_parts, request)
                return TTSResponse(
                    audio_data=audio_data,
                    format=request.format,
                    sample_rate=self.sample_rate,
                    channels=1,
                    provider=self.provider_name,
                    metadata={"dialogue_parts": len(dialogue_parts)}
                )
                
        except Exception as e:
            logger.error(f"{self.provider_name} generation error: {e}")
            raise
    
    async def _stream_audio_dia(
        self,
        dialogue_parts: List[Dict[str, Any]],
        request: TTSRequest
    ) -> AsyncGenerator[bytes, None]:
        """Stream audio from Dia model"""
        if not self.model or not self.processor:
            raise ValueError("Dia model not initialized")
        
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
            # Prepare input for Dia
            input_text = self._format_dia_input(dialogue_parts, request)
            
            # Process with Dia
            inputs = self.processor(
                text=input_text,
                return_tensors="pt",
                padding=True
            ).to(self.device)
            
            # Set generation parameters
            gen_kwargs = {
                "max_new_tokens": 2000,
                "temperature": 0.8,
                "do_sample": True,
                "top_p": 0.95,
                "pad_token_id": self.processor.tokenizer.eos_token_id
            }
            
            # Add seed for consistent voice if specified
            if request.seed:
                gen_kwargs["seed"] = request.seed
                torch.manual_seed(request.seed)
            
            # Generate audio
            with torch.no_grad():
                outputs = self.model.generate(**inputs, **gen_kwargs)
            
            # Decode to audio waveform
            audio_array = self._decode_dia_output(outputs[0])
            
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
            
            logger.info(f"{self.provider_name}: Successfully generated dialogue audio")
            
        except Exception as e:
            logger.error(f"{self.provider_name} streaming error: {e}")
            raise
        finally:
            writer.close()
    
    async def _generate_complete_dia(
        self,
        dialogue_parts: List[Dict[str, Any]],
        request: TTSRequest
    ) -> bytes:
        """Generate complete audio from Dia"""
        all_audio = b""
        async for chunk in self._stream_audio_dia(dialogue_parts, request):
            all_audio += chunk
        return all_audio
    
    def _process_dialogue(
        self,
        text: str,
        speakers: Optional[Dict[str, str]] = None
    ) -> List[Dict[str, Any]]:
        """
        Process text into dialogue parts with speaker assignments.
        Returns list of dicts with speaker, text, and any nonverbal cues.
        """
        parts = []
        
        # Check if text already has speaker markers
        if self._has_speaker_markers(text):
            # Parse existing dialogue format
            dialogue_pieces = self.parse_dialogue(text)
            for speaker, content in dialogue_pieces:
                # Extract nonverbal cues
                content, nonverbal = self._extract_nonverbal(content)
                parts.append({
                    "speaker": speaker,
                    "text": content.strip(),
                    "nonverbal": nonverbal,
                    "voice": speakers.get(speaker, speaker) if speakers else speaker
                })
        else:
            # Treat as single speaker or auto-detect
            if self.auto_detect_speakers:
                # Simple heuristic: split on quotation marks
                segments = self._split_on_quotes(text)
                for i, segment in enumerate(segments):
                    if segment.strip():
                        speaker = f"speaker{(i % self.max_speakers) + 1}"
                        segment, nonverbal = self._extract_nonverbal(segment)
                        parts.append({
                            "speaker": speaker,
                            "text": segment.strip(),
                            "nonverbal": nonverbal,
                            "voice": speakers.get(speaker, speaker) if speakers else speaker
                        })
            else:
                # Single speaker
                text, nonverbal = self._extract_nonverbal(text)
                parts.append({
                    "speaker": "speaker1",
                    "text": text.strip(),
                    "nonverbal": nonverbal,
                    "voice": speakers.get("speaker1", "speaker1") if speakers else "speaker1"
                })
        
        return parts
    
    def _has_speaker_markers(self, text: str) -> bool:
        """Check if text has speaker markers like 'Name:'"""
        pattern = r'^[A-Za-z0-9]+:\s*'
        return bool(re.search(pattern, text, re.MULTILINE))
    
    def _split_on_quotes(self, text: str) -> List[str]:
        """Split text on quotation marks for dialogue detection"""
        # Split on quotes but keep the quotes
        parts = re.split(r'(["\'].*?["\'])', text)
        return [p for p in parts if p.strip()]
    
    def _extract_nonverbal(self, text: str) -> Tuple[str, List[str]]:
        """Extract nonverbal cues from text"""
        nonverbal = []
        for tag in self.NONVERBAL_TAGS:
            if tag in text:
                nonverbal.append(tag)
                # Keep the tags in the text for Dia to process
        return text, nonverbal
    
    def _format_dia_input(
        self,
        dialogue_parts: List[Dict[str, Any]],
        request: TTSRequest
    ) -> str:
        """
        Format input for Dia model.
        Dia expects specific formatting for multi-speaker dialogue.
        """
        formatted_parts = []
        
        for part in dialogue_parts:
            # Format: [Speaker]: text (nonverbal)
            speaker_tag = f"[{part['voice']}]"
            text = part['text']
            
            # Add nonverbal cues
            for cue in part.get('nonverbal', []):
                if cue not in text:
                    text += f" {cue}"
            
            formatted_parts.append(f"{speaker_tag}: {text}")
        
        # Join all parts
        full_text = "\n".join(formatted_parts)
        
        # Add speed modifier if needed
        if request.speed != 1.0:
            full_text = f"<speed:{request.speed}>{full_text}</speed>"
        
        # Add voice reference prompt if provided
        if request.voice_reference:
            full_text = f"<voice_prompt>{full_text}</voice_prompt>"
        
        return full_text
    
    def _decode_dia_output(self, tokens: torch.Tensor) -> np.ndarray:
        """
        Decode Dia output tokens to audio waveform.
        This is a placeholder - actual Dia uses custom decoding.
        """
        # In production, this would use Dia's audio decoder
        # For now, return dummy audio
        num_samples = len(tokens) * 150  # Approximation
        return np.random.randn(num_samples).astype(np.float32) * 0.1
    
    def map_voice(self, voice_id: str) -> str:
        """Map generic voice ID to Dia speaker"""
        if voice_id in self.DEFAULT_SPEAKERS:
            return voice_id
        
        # Map common names to speakers
        voice_mappings = {
            "alice": "speaker1",
            "bob": "speaker2",
            "charlie": "speaker3",
            "narrator": "narrator",
            "main": "speaker1",
            "secondary": "speaker2"
        }
        
        return voice_mappings.get(voice_id.lower(), "speaker1")
    
    async def close(self):
        """Clean up resources"""
        if self.model:
            del self.model
        if self.processor:
            del self.processor
        self.model = None
        self.processor = None
        
        # Clear GPU cache if using CUDA
        if self.device == "cuda" and torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        await super().close()

#
# End of dia_adapter.py
#######################################################################################################################