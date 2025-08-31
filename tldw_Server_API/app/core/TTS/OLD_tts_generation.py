# tts_generation.py
# Description: This module handles the text-to-speech (TTS) generation process.
#
# Imports
from typing import AsyncGenerator, Optional, Dict, Any
#
# Third-party Imports
import asyncio # For semaphore
#
# Local Imports
from tldw_Server_API.app.api.v1.schemas.audio_schemas import OpenAISpeechRequest
from tldw_Server_API.app.core.TTS.tts_backends import TTSBackendManager, TTSBackendBase
#
#######################################################################################################################
#
# Functions:


# For logging
from loguru import logger

class TTSService:
    # Limit concurrent calls to a single backend's generate method if needed
    _backend_semaphore = asyncio.Semaphore(4) # Example: 4 concurrent generations per backend instance

    def __init__(self, backend_manager: TTSBackendManager):
        self.backend_manager = backend_manager

    async def generate_audio_stream(
        self, request: OpenAISpeechRequest, internal_model_id: str
    ) -> AsyncGenerator[bytes, None]:
        """
        Orchestrates fetching the backend and calling its stream generation.
        Handles text processing (chunking, normalization) if the backend expects it.
        Handles audio format conversion if the backend produces raw audio.
        """
        backend: Optional[TTSBackendBase] = await self.backend_manager.get_backend(internal_model_id)
        if not backend:
            logger.error(f"TTSService: No backend found for internal_model_id: {internal_model_id}")
            # This case should ideally be caught by the router's mapping check,
            # but as a safeguard:
            yield f"ERROR: Backend for model '{request.model}' not configured.".encode()
            return

        logger.info(f"TTSService: Using backend {type(backend).__name__} for model '{request.model}' (internal: {internal_model_id})")
        
        # Stream audio from the backend
        try:
            async with self._backend_semaphore:
                 async for audio_bytes_chunk in backend.generate_speech_stream(request):
                    yield audio_bytes_chunk
        except Exception as e:
            logger.error(f"TTSService: Error streaming from backend {type(backend).__name__}: {e}", exc_info=True)
            # Decide how to propagate: re-raise, or yield an error marker if the protocol supports it.
            # Raising here will likely lead to the StreamingResponse stopping and client getting an error.
            raise # Re-raise to be caught by the main endpoint handler

# --- Singleton pattern for TTSService and its manager ---
_tts_service_instance: Optional[TTSService] = None
_tts_backend_manager_instance: Optional[TTSBackendManager] = None
_init_lock = asyncio.Lock()

async def get_tts_service(app_config: Optional[Dict[str, Any]] = None) -> TTSService: # app_config for initialization
    global _tts_service_instance, _tts_backend_manager_instance
    if not _tts_service_instance:
        async with _init_lock:
            if not _tts_service_instance:
                if app_config is None:
                    # Load default configuration
                    from tldw_Server_API.app.core.config import load_comprehensive_config
                    app_config = load_comprehensive_config()
                    logger.info("TTSService: Loaded default configuration")

                if not _tts_backend_manager_instance:
                    _tts_backend_manager_instance = TTSBackendManager(app_config=app_config)
                _tts_service_instance = TTSService(backend_manager=_tts_backend_manager_instance)
                logger.info("TTSService initialized.")
    return _tts_service_instance

async def close_tts_resources():
    """Call this during application shutdown (e.g., FastAPI lifespan event)."""
    global _tts_backend_manager_instance, _tts_service_instance
    async with _init_lock:
        if _tts_backend_manager_instance:
            logger.info("Closing TTS backend resources...")
            await _tts_backend_manager_instance.close_all_backends()
            _tts_backend_manager_instance = None
            _tts_service_instance = None
            logger.info("TTS backend resources closed.")

#
# End of tts_generation.py
#######################################################################################################################
