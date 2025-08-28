# tts_backends.py
# Description: File contains
#
# Imports
import os
from abc import ABC, abstractmethod
from typing import Optional, Dict, Any, AsyncGenerator
#
# Third Party Libraries
import httpx
import numpy as np
from loguru import logger
#
# Local Libraries
from tldw_Server_API.app.api.v1.schemas.audio_schemas import OpenAISpeechRequest
#
#######################################################################################################################
#
# Functions

# Import StreamingAudioWriter for audio format conversion
from tldw_Server_API.app.core.TTS.streaming_audio_writer import StreamingAudioWriter, AudioNormalizer

# --- Abstract Base Class for TTS Backends ---
class TTSBackendBase(ABC):
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.client = httpx.AsyncClient(timeout=60.0) # Shared client for API backends

    @abstractmethod
    async def initialize(self):
        """Async initialization for the backend (e.g., load models)."""
        pass

    @abstractmethod
    async def generate_speech_stream(
        self, request: OpenAISpeechRequest
    ) -> AsyncGenerator[bytes, None]:
        """
        Generates audio for the given text and streams it.
        Should yield bytes of the audio in the request.response_format.
        """
        pass

    async def close(self):
        """Clean up resources, like closing the httpx client."""
        await self.client.aclose()


# --- Concrete Backend for OpenAI Official API ---
class OpenAIAPIBackend(TTSBackendBase):
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        # Try to get API key from config dict or environment variable
        self.api_key = self.config.get("openai_api_key") or os.getenv("OPENAI_API_KEY")
        self.base_url = "https://api.openai.com/v1/audio/speech"
        if not self.api_key:
            logger.error("OpenAIAPIBackend: API key not configured!")

    async def initialize(self):
        logger.info("OpenAIAPIBackend initialized.")
        if not self.api_key:
             logger.warning("OpenAIAPIBackend: API key is missing. Requests will fail.")


    async def generate_speech_stream(
        self, request: OpenAISpeechRequest
    ) -> AsyncGenerator[bytes, None]:
        if not self.api_key:
            logger.error("OpenAIAPIBackend: Cannot generate speech, API key missing.")
            raise ValueError("OpenAI API key not configured")

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        
        # Map the internal model name to OpenAI's expected format
        # The request.model might be our internal name, need to map to OpenAI's name
        openai_model = request.model
        if request.model in ["openai_official_tts-1", "openai_official_tts-1-hd"]:
            # Strip our internal prefix
            openai_model = request.model.replace("openai_official_", "")
        elif request.model not in ["tts-1", "tts-1-hd"]:
            # Default to tts-1 if unknown model
            logger.warning(f"Unknown model {request.model}, defaulting to tts-1")
            openai_model = "tts-1"
        
        payload = {
            "model": openai_model,
            "input": request.input,
            "voice": request.voice,
            "response_format": request.response_format,
            "speed": request.speed,
        }
        logger.info(f"OpenAIAPIBackend: Requesting TTS with model={openai_model}, voice={request.voice}")
        logger.debug(f"Full payload: {payload}")

        try:
            async with self.client.stream("POST", self.base_url, headers=headers, json=payload) as response:
                response.raise_for_status()
                total_bytes = 0
                async for chunk in response.aiter_bytes(chunk_size=1024):
                    total_bytes += len(chunk)
                    yield chunk
                logger.info(f"OpenAIAPIBackend: Successfully streamed {total_bytes} bytes")
        except httpx.HTTPStatusError as e:
            error_content = await e.response.aread()
            error_msg = error_content.decode()
            logger.error(f"OpenAI API error: {e.response.status_code} - {error_msg}")
            if e.response.status_code == 401:
                raise ValueError("Invalid OpenAI API key")
            elif e.response.status_code == 429:
                raise ValueError("OpenAI API rate limit exceeded")
            elif e.response.status_code == 400:
                raise ValueError(f"Invalid request to OpenAI: {error_msg}")
            else:
                raise ValueError(f"OpenAI API error: {error_msg}")
        except httpx.RequestError as e:
            logger.error(f"OpenAIAPIBackend: Network error: {e}", exc_info=True)
            raise ValueError(f"Network error connecting to OpenAI: {str(e)}")
        except Exception as e:
            logger.error(f"OpenAIAPIBackend: Unexpected error: {e}", exc_info=True)
            raise


# --- Concrete Backend for Your Local ro ---
class LocalKokoroBackend(TTSBackendBase):
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        # Config for your local Kokoro: model path, voice dir, ONNX/PyTorch, device
        self.use_onnx = self.config.get("KOKORO_USE_ONNX", True)
        self.kokoro_model_path = self.config.get("KOKORO_MODEL_PATH", "kokoro-v0_19.onnx") # or .pth
        self.kokoro_voices_json = self.config.get("KOKORO_VOICES_JSON_PATH", "voices.json") # For ONNX
        self.kokoro_voice_dir = self.config.get("KOKORO_VOICE_DIR_PT", "App_Function_Libraries/TTS/Kokoro/voices") # For PyTorch
        self.device = self.config.get("KOKORO_DEVICE", "cpu")
        self.kokoro_instance = None # For ONNX Kokoro library
        self.kokoro_model_pt = None # For PyTorch model
        self.tokenizer = None
        self.audio_normalizer = AudioNormalizer()


    async def initialize(self):
        logger.info(f"LocalKokoroBackend: Initializing (ONNX: {self.use_onnx}, Device: {self.device})")
        if self.use_onnx:
            try:
                from kokoro_onnx import Kokoro, EspeakConfig
                # Ensure model files exist, download if not (like in your TTS_Providers_Local.py)
                if not os.path.exists(self.kokoro_model_path):
                    logger.error(f"Kokoro ONNX model not found at {self.kokoro_model_path}")
                    # Add download logic here if needed
                    return
                if not os.path.exists(self.kokoro_voices_json):
                    logger.error(f"Kokoro voices.json not found at {self.kokoro_voices_json}")
                    # Add download logic here if needed
                    return

                espeak_lib = os.getenv("PHONEMIZER_ESPEAK_LIBRARY")
                self.kokoro_instance = Kokoro(
                    self.kokoro_model_path,
                    self.kokoro_voices_json,
                    espeak_config=EspeakConfig(lib_path=espeak_lib) if espeak_lib else None
                )
                logger.info("LocalKokoroBackend: ONNX instance created.")
            except ImportError:
                logger.error("LocalKokoroBackend: kokoro_onnx library not found. ONNX mode unavailable.")
                self.use_onnx = False # Fallback or error
            except Exception as e:
                logger.error(f"LocalKokoroBackend: Error initializing ONNX Kokoro: {e}", exc_info=True)
                self.use_onnx = False
        else: # PyTorch mode (adapt your existing logic)
            try:
                # from PoC_Version.App_Function_Libraries.TTS.Kokoro.models import build_model # Your model loader
                # from transformers import AutoTokenizer # For chunking
                # import torch

                # self.kokoro_model_pt = get_kokoro_model(device=self.device) # Your existing function
                # self.tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
                logger.info("LocalKokoroBackend: PyTorch model/tokenizer placeholder setup.")
                # Actual loading of PyTorch model would happen here using your existing functions
                pass
            except ImportError:
                logger.error("LocalKokoroBackend: PyTorch dependencies (e.g., torch, transformers) not found.")
            except Exception as e:
                logger.error(f"LocalKokoroBackend: Error initializing PyTorch Kokoro: {e}", exc_info=True)
        logger.info("LocalKokoroBackend initialized.")


    async def _generate_with_kokoro_onnx(self, request: OpenAISpeechRequest) -> AsyncGenerator[bytes, None]:
        if not self.kokoro_instance:
            logger.error("LocalKokoroBackend (ONNX): Not initialized.")
            raise ValueError("Kokoro ONNX backend not initialized")

        # Determine language from voice name (simple heuristic)
        lang = 'en-us'  # Default
        if request.voice and len(request.voice) > 0:
            if request.voice[0].lower() == 'a':
                lang = 'en-us'
            elif request.voice[0].lower() == 'b':
                lang = 'en-gb'
        
        # Use lang_code from request if provided
        if hasattr(request, 'lang_code') and request.lang_code:
            lang = request.lang_code

        logger.info(f"Kokoro ONNX: Generating speech for voice '{request.voice}', lang '{lang}', format '{request.response_format}'")
        
        # Initialize StreamingAudioWriter for the target format
        # Kokoro typically outputs at 24kHz
        sample_rate = 24000
        saw = StreamingAudioWriter(format=request.response_format, sample_rate=sample_rate, channels=1)
        
        try:
            # Stream audio chunks from Kokoro
            chunk_count = 0
            async for samples_chunk, sr_chunk in self.kokoro_instance.create_stream(
                request.input, voice=request.voice, speed=request.speed, lang=lang
            ):
                if samples_chunk is not None and len(samples_chunk) > 0:
                    chunk_count += 1
                    # Normalize float32 samples to int16
                    normalized_chunk = self.audio_normalizer.normalize(samples_chunk, target_dtype=np.int16)
                    
                    # Write chunk and get encoded bytes
                    encoded_bytes = saw.write_chunk(normalized_chunk)
                    if encoded_bytes:
                        yield encoded_bytes
                        logger.debug(f"Kokoro ONNX: Yielded chunk {chunk_count}, {len(encoded_bytes)} bytes")
            
            # Finalize the stream and get any remaining bytes
            final_bytes = saw.write_chunk(finalize=True)
            if final_bytes:
                yield final_bytes
                logger.debug(f"Kokoro ONNX: Yielded final chunk, {len(final_bytes)} bytes")
            
            logger.info(f"Kokoro ONNX: Successfully generated {chunk_count} chunks")
            
        except ImportError as e:
            logger.error(f"Kokoro ONNX import error: {e}")
            raise ValueError("Kokoro ONNX library not properly installed")
        except Exception as e:
            logger.error(f"LocalKokoroBackend (ONNX) error during generation: {e}", exc_info=True)
            raise ValueError(f"Kokoro ONNX generation failed: {str(e)}")
        finally:
            # Ensure cleanup
            saw.close()


    async def _generate_with_kokoro_pytorch(self, request: OpenAISpeechRequest) -> AsyncGenerator[bytes, None]:
        # PyTorch implementation would go here
        # For now, we'll focus on ONNX implementation
        logger.warning("LocalKokoroBackend (PyTorch): Not implemented. Use ONNX mode instead.")
        raise NotImplementedError("PyTorch Kokoro backend not yet implemented. Please use ONNX mode.")


    async def generate_speech_stream(
        self, request: OpenAISpeechRequest
    ) -> AsyncGenerator[bytes, None]:
        logger.info(f"LocalKokoroBackend: Generating speech. ONNX: {self.use_onnx}")
        if self.use_onnx:
            async for chunk in self._generate_with_kokoro_onnx(request):
                yield chunk
        else:
            async for chunk in self._generate_with_kokoro_pytorch(request):
                yield chunk

# Additional backends can be added here (ElevenLabs, AllTalk, etc.)


# --- Backend Manager ---
class TTSBackendManager:
    def __init__(self, app_config: Any):  # app_config is ConfigParser
        self.app_config = app_config
        self._backends: Dict[str, TTSBackendBase] = {}
        self._initialized_backends: set[str] = set()
        self._config_dict = self._parse_config(app_config)
    
    def _parse_config(self, config_parser) -> Dict[str, Any]:
        """Extract relevant config values from ConfigParser"""
        config_dict = {}
        
        # Extract API keys from API section
        if hasattr(config_parser, 'has_section') and config_parser.has_section('API'):
            if config_parser.has_option('API', 'openai_api_key'):
                config_dict['openai_api_key'] = config_parser.get('API', 'openai_api_key')
            if config_parser.has_option('API', 'elevenlabs_api_key'):
                config_dict['elevenlabs_api_key'] = config_parser.get('API', 'elevenlabs_api_key')
        
        # Extract TTS-specific settings if they exist
        if hasattr(config_parser, 'has_section') and config_parser.has_section('TTS'):
            for key, value in config_parser.items('TTS'):
                config_dict[key] = value
        
        # Also check environment variables as fallback
        config_dict['openai_api_key'] = config_dict.get('openai_api_key') or os.getenv('OPENAI_API_KEY')
        config_dict['elevenlabs_api_key'] = config_dict.get('elevenlabs_api_key') or os.getenv('ELEVENLABS_API_KEY')
        
        return config_dict

    async def get_backend(self, backend_id: str) -> Optional[TTSBackendBase]:
        if backend_id not in self._backends:
            logger.info(f"TTSBackendManager: Creating backend for ID: {backend_id}")
            
            # Use the parsed config dictionary
            base_config = self._config_dict.copy()

            if backend_id in ["openai_official_tts-1", "openai_official_tts-1-hd"]:
                self._backends[backend_id] = OpenAIAPIBackend(config=base_config)
            elif backend_id == "local_kokoro_default_onnx":
                # Kokoro-specific config
                kokoro_cfg = base_config.copy()
                kokoro_cfg.update({
                    "KOKORO_USE_ONNX": True,
                    "KOKORO_MODEL_PATH": base_config.get("kokoro_model_path", "kokoro-v0_19.onnx"),
                    "KOKORO_VOICES_JSON_PATH": base_config.get("kokoro_voices_json", "voices.json"),
                    "KOKORO_DEVICE": base_config.get("kokoro_device", "cpu"),
                })
                self._backends[backend_id] = LocalKokoroBackend(config=kokoro_cfg)
            # Add more backends as needed
            # elif backend_id == "elevenlabs_english_v1":
            #     self._backends[backend_id] = ElevenLabsBackend(config=base_config)
            else:
                logger.error(f"TTSBackendManager: Unknown backend ID: {backend_id}")
                return None

        backend = self._backends[backend_id]
        if backend_id not in self._initialized_backends:
            logger.info(f"TTSBackendManager: Initializing backend: {backend_id}")
            await backend.initialize()
            self._initialized_backends.add(backend_id)
        return backend

    async def close_all_backends(self):
        logger.info("TTSBackendManager: Closing all backends.")
        for backend_id, backend in self._backends.items():
            try:
                logger.info(f"Closing backend: {backend_id}")
                await backend.close()
            except Exception as e:
                logger.error(f"Error closing backend {backend_id}: {e}")
        self._backends.clear()
        self._initialized_backends.clear()







#
# End of tts_backends.py
#######################################################################################################################
