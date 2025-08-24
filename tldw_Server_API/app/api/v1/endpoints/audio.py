# audio.py
# Description: This file contains the API endpoints for audio processing.
#
# Imports
import asyncio
import json
import os
from typing import AsyncGenerator, Optional # Add this import
#
# Third-party libraries
from fastapi import APIRouter, Depends, HTTPException, Request, Header
from fastapi.responses import StreamingResponse, Response # Add Response
from starlette import status # For status codes
from slowapi import Limiter
from slowapi.util import get_remote_address
#
# Local imports
from tldw_Server_API.app.api.v1.schemas.audio_schemas import OpenAISpeechRequest
from tldw_Server_API.app.core.config import AUTH_BEARER_PREFIX
from tldw_Server_API.app.core.Auth.auth_utils import (
    extract_bearer_token,
    validate_api_token,
    get_expected_api_token,
    is_authentication_required
)
# from your_project.services.tts_service import TTSService, get_tts_service

# For logging (if you use the same logger as in your PDF endpoint)
import logging # or from your_project.utils import logger
logger = logging.getLogger(__name__)

# Initialize rate limiter
limiter = Limiter(key_func=get_remote_address)


router = APIRouter(
    prefix="/v1/audio", # Standard OpenAI prefix
    tags=["TTS (OpenAI Compatible)"],
    responses={
        404: {"description": "Not found"},
        401: {"description": "Unauthorized"},
        429: {"description": "Rate limit exceeded"}
    },
)



# --- Placeholder for TTSService and Mappings ---
# We will define these in subsequent steps.
# This is just to get the endpoint structure.

_openai_mappings = { # Load this from a JSON file later
    "models": {
        "tts-1": "openai_official_tts-1", # Maps to your backend identifier
        "tts-1-hd": "openai_official_tts-1-hd",
        "eleven_english_v1": "elevenlabs_english_v1",
        "kokoro_local": "local_kokoro_default_onnx"
    },
    "voices": { # This part is more complex and often backend-specific
        # OpenAI voices
        "alloy": "alloy", "echo": "echo", "fable": "fable",
        "onyx": "onyx", "nova": "nova", "shimmer": "shimmer",
        # ElevenLabs (IDs or names)
        "Rachel": "21m00Tcm4TlvDq8ikWAM", # Example ID
        # Kokoro local voices
        "bella": "af_bella", # Example mapping
    }
}

# Import the real TTS service
from tldw_Server_API.app.core.TTS.tts_generation import TTSService, get_tts_service as get_real_tts_service

async def get_tts_service() -> TTSService:
    """Get the TTS service instance."""
    return await get_real_tts_service()

# --- End of Placeholder ---


@router.post("/speech", summary="Generates audio from text input.")
@limiter.limit("10/minute")  # Rate limit: 10 requests per minute per IP
async def create_speech(
    request_data: OpenAISpeechRequest, # FastAPI will parse JSON body into this
    request: Request, # Required for rate limiter and to check for client disconnects
    tts_service: TTSService = Depends(get_tts_service),
    authorization: Optional[str] = Header(None),
):
    """
    Generates audio from the input text.
    
    Requires authentication via Bearer token in Authorization header.
    Rate limited to 10 requests per minute per IP address.
    """
    
    # Authentication check
    if is_authentication_required():
        # Extract and validate token
        token = extract_bearer_token(authorization)
        if not token:
            logger.warning("TTS request without authentication token")
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Authentication required. Please provide a valid Bearer token.",
                headers={"WWW-Authenticate": f"{AUTH_BEARER_PREFIX}"},
            )
        
        # Validate the token
        expected_token = get_expected_api_token()
        if not validate_api_token(token, expected_token):
            logger.warning(f"TTS request with invalid token")
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid authentication token.",
                headers={"WWW-Authenticate": f"{AUTH_BEARER_PREFIX}"},
            )
    
    # Input validation
    if not request_data.input or len(request_data.input.strip()) == 0:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Input text cannot be empty."
        )
    
    # Limit input length to prevent abuse
    MAX_INPUT_LENGTH = 4096  # Maximum characters
    if len(request_data.input) > MAX_INPUT_LENGTH:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Input text exceeds maximum length of {MAX_INPUT_LENGTH} characters."
        )
    logger.info(f"Received speech request: model={request_data.model}, voice={request_data.voice}, format={request_data.response_format}")

    # 1. Map OpenAI model name to your internal backend identifier
    internal_model_id = _openai_mappings["models"].get(request_data.model)
    if not internal_model_id:
        logger.warning(f"Unsupported model requested: {request_data.model}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={
                "error": "invalid_model",
                "message": f"The model `{request_data.model}` does not exist or you do not have access to it.",
                "type": "invalid_request_error",
            },
        )

    # 2. Basic voice validation (can be enhanced per backend later)
    # For now, we assume the voice name is passed as is to the backend.
    # More sophisticated mapping might be needed if OpenAI voice names
    # differ significantly from how your backends expect them.
    # internal_voice_id = _openai_mappings["voices"].get(request_data.voice, request_data.voice)
    # request_data.voice = internal_voice_id # Update request_data if you map voices

    # 3. Determine Content-Type
    content_type_map = {
        "mp3": "audio/mpeg",
        "opus": "audio/opus",
        "aac": "audio/aac",
        "flac": "audio/flac",
        "wav": "audio/wav",
        "pcm": "audio/L16; rate=24000; channels=1", # Example for raw PCM
    }
    content_type = content_type_map.get(request_data.response_format)
    if not content_type:
        logger.warning(f"Unsupported response format: {request_data.response_format}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unsupported response_format: {request_data.response_format}. Supported formats are: {', '.join(content_type_map.keys())}",
        )

    # 4. Streaming Logic (simplified from target app)
    async def audio_chunk_generator():
        try:
            async for audio_chunk_bytes in tts_service.generate_audio_stream(request_data, internal_model_id):
                if await request.is_disconnected():
                    logger.info("Client disconnected, stopping audio generation.")
                    break
                yield audio_chunk_bytes
        except HTTPException: # Re-raise HTTPExceptions directly
            raise
        except Exception as e:
            logger.error(f"Error during audio streaming: {e}", exc_info=True)
            # Important: Don't yield anything here if an error occurs,
            # let FastAPI handle the error response.
            # For a production system, you might want to yield a specific error chunk
            # if the protocol requires it, but for simple streaming, just raising is often enough.
            # If you raise HTTPException here, it should be caught by FastAPI.
            # If you raise a standard Python exception, it will result in a 500.
            # Consider how to signal errors in the stream if the client expects it.
            # For now, we'll let it become a 500 or be handled by the `tts_service` itself.
            raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))


    if request_data.stream:
        return StreamingResponse(
            audio_chunk_generator(),
            media_type=content_type,
            headers={
                "Content-Disposition": f"attachment; filename=speech.{request_data.response_format}",
                "X-Accel-Buffering": "no", # Useful for Nginx
                "Cache-Control": "no-cache",
            },
        )
    else:
        # Non-streaming: Collect all chunks and send as a single response
        all_audio_bytes = b""
        try:
            async for chunk in audio_chunk_generator():
                all_audio_bytes += chunk
            # Remove any "final boundary" placeholder if it was added by the dummy service
            all_audio_bytes = all_audio_bytes.replace(b"--final_boundary_for_non_streamed--", b"")

            if not all_audio_bytes: # Handle case where generation yielded nothing
                 logger.error("Non-streaming generation resulted in empty audio data.")
                 raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Audio generation failed to produce data.")

            return Response(
                content=all_audio_bytes,
                media_type=content_type,
                headers={
                    "Content-Disposition": f"attachment; filename=speech.{request_data.response_format}",
                    "Cache-Control": "no-cache",
                },
            )
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Error during non-streaming audio generation: {e}", exc_info=True)
            raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))


# Add other OpenAI compatible endpoints like /models, /voices later
# For now, this is the core.

#
# End of audio.py
#######################################################################################################################
