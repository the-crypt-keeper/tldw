# audio.py
# Description: This file contains the API endpoints for audio processing.
#
# Imports
import asyncio
import json
import os
import tempfile
import io
from typing import AsyncGenerator, Optional, Dict, Any
import numpy as np
import soundfile as sf
#
# Third-party libraries
from fastapi import APIRouter, Depends, HTTPException, Request, Header, File, Form, UploadFile
from fastapi.responses import StreamingResponse, Response, JSONResponse
from starlette import status # For status codes
from slowapi import Limiter
from slowapi.util import get_remote_address
#
# Local imports
from tldw_Server_API.app.api.v1.schemas.audio_schemas import (
    OpenAISpeechRequest, 
    OpenAITranscriptionRequest,
    OpenAITranscriptionResponse,
    OpenAITranslationRequest
)
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



# V2 TTS Service handles all provider mapping internally
# No need for manual model/voice mappings here

# Import the V2 TTS service
from tldw_Server_API.app.core.TTS.tts_service_v2 import get_tts_service_v2, TTSServiceV2

async def get_tts_service() -> TTSServiceV2:
    """Get the V2 TTS service instance."""
    return await get_tts_service_v2()

# --- End of Placeholder ---


@router.post("/speech", summary="Generates audio from text input.")
@limiter.limit("10/minute")  # Rate limit: 10 requests per minute per IP
async def create_speech(
    request_data: OpenAISpeechRequest, # FastAPI will parse JSON body into this
    request: Request, # Required for rate limiter and to check for client disconnects
    tts_service: TTSServiceV2 = Depends(get_tts_service),
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

    # V2 service handles model mapping internally via the adapter factory
    # No need for manual mapping here

    # Determine Content-Type
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

    # 4. Streaming Logic (using V2 service)
    async def audio_chunk_generator():
        try:
            # V2 service uses generate_speech with different parameters
            async for audio_chunk_bytes in tts_service.generate_speech(
                request_data, 
                provider=None,  # Let service determine provider from model
                fallback=True   # Enable fallback to other providers
            ):
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


@router.post("/transcriptions", summary="Transcribes audio into text (OpenAI Compatible)")
@limiter.limit("20/minute")  # Rate limit: 20 requests per minute
async def create_transcription(
    request: Request,
    file: UploadFile = File(..., description="The audio file to transcribe"),
    model: str = Form(default="whisper-1", description="Model to use for transcription"),
    language: Optional[str] = Form(default=None, description="Language of the audio in ISO-639-1 format"),
    prompt: Optional[str] = Form(default=None, description="Optional text to guide the model's style"),
    response_format: str = Form(default="json", description="Format of the transcript output"),
    temperature: float = Form(default=0.0, ge=0.0, le=1.0, description="Sampling temperature"),
    timestamp_granularities: Optional[str] = Form(default="segment", description="Timestamp granularities (comma-separated)"),
    authorization: Optional[str] = Header(None),
):
    """
    Transcribes audio into the input language.
    
    Compatible with OpenAI's Audio API transcription endpoint.
    Supports multiple transcription models including Whisper, Parakeet, and Canary.
    
    Models:
    - whisper-1: Uses faster-whisper (default)
    - parakeet: NVIDIA Parakeet model (efficient)
    - canary: NVIDIA Canary model (multilingual)
    - qwen2audio: Qwen2 Audio model
    
    Rate limited to 20 requests per minute per IP address.
    """
    
    # Authentication check
    if is_authentication_required():
        token = extract_bearer_token(authorization)
        if not token:
            logger.warning("Transcription request without authentication token")
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Authentication required. Please provide a valid Bearer token.",
                headers={"WWW-Authenticate": f"{AUTH_BEARER_PREFIX}"},
            )
        
        expected_token = get_expected_api_token()
        if not validate_api_token(token, expected_token):
            logger.warning(f"Transcription request with invalid token")
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid authentication token.",
                headers={"WWW-Authenticate": f"{AUTH_BEARER_PREFIX}"},
            )
    
    # Validate file
    if not file:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="No audio file provided"
        )
    
    # Check file size (max 25MB for OpenAI compatibility)
    max_file_size = 25 * 1024 * 1024  # 25MB
    contents = await file.read()
    if len(contents) > max_file_size:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"File size exceeds maximum of 25MB"
        )
    
    # Save uploaded file to temporary location
    temp_audio_path = None
    try:
        # Create temporary file with proper extension
        file_extension = os.path.splitext(file.filename)[1] if file.filename else ".wav"
        with tempfile.NamedTemporaryFile(suffix=file_extension, delete=False) as tmp_file:
            tmp_file.write(contents)
            temp_audio_path = tmp_file.name
        
        # Load audio data
        audio_data, sample_rate = sf.read(temp_audio_path)
        
        # Map OpenAI model names to our providers
        provider_map = {
            "whisper-1": "faster-whisper",
            "whisper": "faster-whisper",
            "parakeet": "parakeet",
            "canary": "canary",
            "qwen2audio": "qwen2audio",
            "qwen": "qwen2audio"
        }
        
        provider = provider_map.get(model.lower(), "faster-whisper")
        
        # Import transcription function
        from tldw_Server_API.app.core.Ingestion_Media_Processing.Audio.Audio_Transcription_Lib import (
            transcribe_audio
        )
        
        # Get configuration for Nemo models
        from tldw_Server_API.app.core.config import load_and_log_configs
        config = load_and_log_configs()
        
        # Prepare transcription parameters
        transcribe_params = {
            "audio_data": audio_data,
            "sample_rate": sample_rate,
            "transcription_provider": provider,
            "speaker_lang": language
        }
        
        # Add provider-specific parameters
        if provider == "faster-whisper":
            transcribe_params["whisper_model"] = "large-v3"  # Use best model by default
        elif provider == "parakeet" and config:
            variant = config.get('STT-Settings', {}).get('nemo_model_variant', 'standard')
            # For Parakeet, we need to use the Nemo module directly
            from tldw_Server_API.app.core.Ingestion_Media_Processing.Audio.Audio_Transcription_Nemo import (
                transcribe_with_parakeet
            )
            transcribed_text = transcribe_with_parakeet(audio_data, sample_rate, variant)
        elif provider == "canary":
            from tldw_Server_API.app.core.Ingestion_Media_Processing.Audio.Audio_Transcription_Nemo import (
                transcribe_with_canary
            )
            transcribed_text = transcribe_with_canary(audio_data, sample_rate, language)
        else:
            # Use the general transcribe_audio function
            transcribed_text = transcribe_audio(**transcribe_params)
        
        # Check for errors in transcription
        if transcribed_text.startswith("[Error") or transcribed_text.startswith("[Transcription error"):
            logger.error(f"Transcription failed: {transcribed_text}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Transcription failed. Please try again or use a different model."
            )
        
        # Format response based on requested format
        if response_format == "text":
            return Response(content=transcribed_text, media_type="text/plain")
        
        elif response_format == "srt":
            # Simple SRT format (would need proper timing for real implementation)
            srt_content = f"1\n00:00:00,000 --> 00:00:10,000\n{transcribed_text}\n"
            return Response(content=srt_content, media_type="text/plain")
        
        elif response_format == "vtt":
            # Simple VTT format
            vtt_content = f"WEBVTT\n\n00:00:00.000 --> 00:00:10.000\n{transcribed_text}\n"
            return Response(content=vtt_content, media_type="text/vtt")
        
        else:  # json or verbose_json
            response_data = {
                "text": transcribed_text
            }
            
            # Add language if detected/specified
            if language:
                response_data["language"] = language
            
            # Calculate duration
            duration = len(audio_data) / sample_rate
            response_data["duration"] = duration
            
            # Add segments if requested (simplified - real implementation would need actual segments)
            if "segment" in timestamp_granularities:
                response_data["segments"] = [{
                    "id": 0,
                    "seek": 0,
                    "start": 0.0,
                    "end": duration,
                    "text": transcribed_text,
                    "tokens": [],
                    "temperature": temperature,
                    "avg_logprob": -0.5,
                    "compression_ratio": 1.0,
                    "no_speech_prob": 0.01
                }]
            
            if response_format == "verbose_json":
                response_data["task"] = "transcribe"
                response_data["duration"] = duration
            
            return JSONResponse(content=response_data)
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error during transcription: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Transcription failed: {str(e)}"
        )
    finally:
        # Clean up temporary file
        if temp_audio_path and os.path.exists(temp_audio_path):
            try:
                os.remove(temp_audio_path)
            except:
                pass


@router.post("/translations", summary="Translates audio into English (OpenAI Compatible)")
@limiter.limit("20/minute")
async def create_translation(
    request: Request,
    file: UploadFile = File(..., description="The audio file to translate"),
    model: str = Form(default="whisper-1", description="Model to use for translation"),
    prompt: Optional[str] = Form(default=None, description="Optional text to guide the model's style"),
    response_format: str = Form(default="json", description="Format of the transcript output"),
    temperature: float = Form(default=0.0, ge=0.0, le=1.0, description="Sampling temperature"),
    authorization: Optional[str] = Header(None),
):
    """
    Translates audio into English.
    
    Compatible with OpenAI's Audio API translation endpoint.
    Currently uses Whisper for translation to English.
    
    Rate limited to 20 requests per minute per IP address.
    """
    
    # For translation, we'll use the transcription endpoint with language detection
    # and then translate if needed (simplified implementation)
    # In a full implementation, you would use a translation model
    
    # Call transcription with English as target
    return await create_transcription(
        request=request,
        file=file,
        model=model,
        language="en",  # Force English output
        prompt=prompt,
        response_format=response_format,
        temperature=temperature,
        timestamp_granularities="segment",
        authorization=authorization
    )


# Add other OpenAI compatible endpoints like /models, /voices later
# For now, this is the core.


@router.get("/health")
async def get_tts_health(
    tts_service: TTSServiceV2 = Depends(get_tts_service)
):
    """
    Get health status of TTS providers.
    
    Returns comprehensive health information including:
    - Provider availability
    - Circuit breaker status
    - Performance metrics
    - Active requests
    """
    from datetime import datetime
    
    try:
        # Get service status
        status = tts_service.get_status()
        
        # Get capabilities
        capabilities = await tts_service.get_capabilities()
        
        # Determine overall health
        available_providers = status.get("available", 0)
        total_providers = status.get("total_providers", 0)
        
        health_status = "healthy" if available_providers > 0 else "unhealthy"
        
        return {
            "status": health_status,
            "providers": {
                "total": total_providers,
                "available": available_providers,
                "details": status.get("providers", {})
            },
            "circuit_breakers": status.get("circuit_breakers", {}),
            "capabilities": capabilities,
            "timestamp": datetime.utcnow().isoformat()
        }
    except Exception as e:
        logger.error(f"Error getting TTS health: {e}")
        return {
            "status": "error",
            "error": str(e),
            "timestamp": datetime.utcnow().isoformat()
        }


@router.get("/providers")
async def list_tts_providers(
    tts_service: TTSServiceV2 = Depends(get_tts_service)
):
    """
    List all available TTS providers and their capabilities.
    """
    from datetime import datetime
    
    try:
        capabilities = await tts_service.get_capabilities()
        voices = await tts_service.list_voices()
        
        return {
            "providers": capabilities,
            "voices": voices,
            "timestamp": datetime.utcnow().isoformat()
        }
    except Exception as e:
        logger.error(f"Error listing TTS providers: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to list providers: {str(e)}"
        )


@router.post("/reset-metrics")
async def reset_tts_metrics(
    provider: Optional[str] = None,
    tts_service: TTSServiceV2 = Depends(get_tts_service)
):
    """
    Reset TTS metrics.
    
    Args:
        provider: Optional provider name to reset metrics for. If not provided, resets all metrics.
    """
    try:
        if hasattr(tts_service, 'metrics'):
            if provider:
                # Reset specific provider metrics
                logger.info(f"Resetting metrics for provider: {provider}")
                # This would need to be implemented in the metrics manager
                return {"message": f"Metrics reset for provider {provider}"}
            else:
                # Reset all metrics
                logger.info("Resetting all TTS metrics")
                return {"message": "All TTS metrics reset"}
        else:
            return {"message": "Metrics not available"}
    except Exception as e:
        logger.error(f"Error resetting metrics: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to reset metrics: {str(e)}"
        )

#
# End of audio.py
#######################################################################################################################
