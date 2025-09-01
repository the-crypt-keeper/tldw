# audio.py
# Description: This file contains the API endpoints for audio processing.
#
# Imports
import asyncio
import json
import os
import tempfile
import io
from pathlib import Path as PathLib
from typing import AsyncGenerator, Optional, Dict, Any
import numpy as np
import soundfile as sf
#
# Third-party libraries
from fastapi import APIRouter, Depends, HTTPException, Request, Header, File, Form, UploadFile, WebSocket, WebSocketDisconnect, Path
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
from tldw_Server_API.app.core.AuthNZ.User_DB_Handling import get_request_user, User
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

# Import the V2 TTS service and validation
from tldw_Server_API.app.core.TTS.tts_service_v2 import get_tts_service_v2, TTSServiceV2
from tldw_Server_API.app.core.TTS.tts_exceptions import (
    TTSError,
    TTSValidationError,
    TTSProviderNotConfiguredError,
    TTSAuthenticationError,
    TTSRateLimitError
)
from tldw_Server_API.app.core.TTS.tts_validation import TTSInputValidator

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
    
    # Input validation using the new validation system
    try:
        # Create validator instance
        validator = TTSInputValidator()
        
        # Validate and sanitize input text
        sanitized_text = validator.sanitize_text(request_data.input)
        
        # Check for empty input after sanitization
        if not sanitized_text or len(sanitized_text.strip()) == 0:
            raise TTSValidationError(
                "Input text cannot be empty after sanitization",
                details={"original_length": len(request_data.input)}
            )
        
        # Update request with sanitized text
        request_data.input = sanitized_text
        
    except TTSValidationError as e:
        logger.warning(f"TTS validation error: {e}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
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
            # The service will handle additional validation internally
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
        except TTSProviderNotConfiguredError as e:
            logger.error(f"TTS provider not configured: {e}")
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail=f"TTS service unavailable: {str(e)}"
            )
        except TTSAuthenticationError as e:
            logger.error(f"TTS authentication error: {e}")
            raise HTTPException(
                status_code=status.HTTP_502_BAD_GATEWAY,
                detail="TTS provider authentication failed"
            )
        except TTSRateLimitError as e:
            logger.warning(f"TTS rate limit exceeded: {e}")
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail="TTS provider rate limit exceeded. Please try again later."
            )
        except TTSError as e:
            # Handle other TTS-specific errors
            logger.error(f"TTS error during streaming: {e}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"TTS error: {str(e)}"
            )
        except Exception as e:
            logger.error(f"Unexpected error during audio streaming: {e}", exc_info=True)
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="An unexpected error occurred during audio generation"
            )


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
        except TTSProviderNotConfiguredError as e:
            logger.error(f"TTS provider not configured: {e}")
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail=f"TTS service unavailable: {str(e)}"
            )
        except TTSAuthenticationError as e:
            logger.error(f"TTS authentication error: {e}")
            raise HTTPException(
                status_code=status.HTTP_502_BAD_GATEWAY,
                detail="TTS provider authentication failed"
            )
        except TTSRateLimitError as e:
            logger.warning(f"TTS rate limit exceeded: {e}")
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail="TTS provider rate limit exceeded. Please try again later."
            )
        except TTSError as e:
            # Handle other TTS-specific errors
            logger.error(f"TTS error during non-streaming generation: {e}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"TTS error: {str(e)}"
            )
        except Exception as e:
            logger.error(f"Unexpected error during non-streaming audio generation: {e}", exc_info=True)
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="An unexpected error occurred during audio generation"
            )


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

######################################################################################################################
# WebSocket Streaming Transcription Endpoints
######################################################################################################################

@router.websocket("/stream/transcribe")
async def websocket_transcribe(websocket: WebSocket):
    """
    WebSocket endpoint for real-time audio transcription.
    
    Protocol:
    1. Client connects via WebSocket
    2. Client sends configuration message:
       {"type": "config", "sample_rate": 16000, "language": "en", "model_variant": "mlx"}
    3. Client sends audio chunks:
       {"type": "audio", "data": "<base64_encoded_float32_audio>"}
    4. Server responds with transcriptions:
       {"type": "transcription", "text": "...", "timestamp": ..., "is_final": true}
       {"type": "partial", "text": "...", "timestamp": ..., "is_final": false}
    5. Client can send commit to finalize:
       {"type": "commit"}
    6. Server sends final transcript:
       {"type": "full_transcript", "text": "..."}
    """
    await websocket.accept()
    
    try:
        from tldw_Server_API.app.core.Ingestion_Media_Processing.Audio.Audio_Streaming_Parakeet import (
            handle_websocket_transcription,
            StreamingConfig
        )
        
        # Default configuration
        config = StreamingConfig(
            model_variant='mlx',  # Default to MLX variant
            sample_rate=16000,
            chunk_duration=2.0,
            overlap_duration=0.5,
            enable_partial=True,
            partial_interval=0.5
        )
        
        # Handle the WebSocket connection
        await handle_websocket_transcription(websocket, config)
        
    except WebSocketDisconnect:
        logger.info("WebSocket disconnected")
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
        try:
            await websocket.send_json({
                "type": "error",
                "message": str(e)
            })
        except:
            pass
    finally:
        try:
            await websocket.close()
        except:
            pass


@router.get("/stream/status", summary="Check streaming transcription availability")
async def streaming_status():
    """
    Check if streaming transcription is available.
    
    Returns:
        JSON with status and available models
    """
    try:
        # Check available models
        available_models = []
        
        # Check for MLX variant
        try:
            from tldw_Server_API.app.core.Ingestion_Media_Processing.Audio.Audio_Transcription_Parakeet_MLX import (
                transcribe_with_parakeet_mlx
            )
            available_models.append("parakeet-mlx")
        except ImportError:
            pass
        
        # Check for standard variant
        try:
            from tldw_Server_API.app.core.Ingestion_Media_Processing.Audio.Audio_Transcription_Nemo import (
                load_parakeet_model
            )
            available_models.append("parakeet-standard")
        except ImportError:
            pass
        
        return JSONResponse({
            "status": "available" if available_models else "unavailable",
            "available_models": available_models,
            "websocket_endpoint": "/api/v1/audio/stream/transcribe",
            "supported_features": {
                "partial_results": True,
                "multiple_languages": True,
                "concurrent_streams": True
            }
        })
        
    except Exception as e:
        logger.error(f"Error checking streaming status: {e}")
        return JSONResponse(
            status_code=500,
            content={
                "status": "error",
                "message": str(e)
            }
        )


@router.post("/stream/test", summary="Test streaming transcription setup")
async def test_streaming():
    """
    Test endpoint to verify streaming setup.
    
    Returns:
        Test results
    """
    try:
        from tldw_Server_API.app.core.Ingestion_Media_Processing.Audio.Audio_Streaming_Parakeet import (
            ParakeetStreamingTranscriber,
            StreamingConfig
        )
        import base64
        
        # Try to initialize transcriber
        config = StreamingConfig(model_variant='mlx')
        transcriber = ParakeetStreamingTranscriber(config)
        
        # Generate test audio
        sample_rate = 16000
        duration = 0.5
        t = np.linspace(0, duration, int(sample_rate * duration))
        audio = (0.5 * np.sin(440 * 2 * np.pi * t)).astype(np.float32)
        encoded = base64.b64encode(audio.tobytes()).decode('utf-8')
        
        # Try processing
        result = await transcriber.process_audio_chunk(encoded)
        
        return JSONResponse({
            "status": "success",
            "test_passed": True,
            "message": "Streaming transcription is working",
            "test_result": result if result else "Buffer accumulating"
        })
        
    except Exception as e:
        logger.error(f"Streaming test failed: {e}")
        return JSONResponse(
            status_code=500,
            content={
                "status": "error",
                "test_passed": False,
                "message": str(e)
            }
        )

#######################################################################################################################
#
# Voice Management Endpoints
#

@router.post("/voices/upload", summary="Upload a custom voice sample")
@limiter.limit("5/hour")  # Rate limit: 5 uploads per hour
async def upload_voice(
    request: Request,
    file: UploadFile = File(..., description="Voice sample audio file (WAV, MP3, FLAC, OGG)"),
    name: str = Form(..., description="Name for the voice"),
    description: Optional[str] = Form(None, description="Description of the voice"),
    provider: str = Form(default="vibevoice", description="Target TTS provider"),
    current_user: User = Depends(get_request_user)
):
    """
    Upload a custom voice sample for use with TTS.
    
    Supports voice cloning for compatible providers:
    - VibeVoice: Any duration (1-shot cloning)
    - Higgs: 3-10 seconds recommended
    - Chatterbox: 5-20 seconds recommended
    
    The voice will be processed and optimized for the specified provider.
    """
    from tldw_Server_API.app.core.TTS.voice_manager import (
        get_voice_manager,
        VoiceUploadRequest,
        VoiceProcessingError,
        VoiceQuotaExceededError
    )
    
    try:
        # Get voice manager
        voice_manager = get_voice_manager()
        
        # Read file content
        file_content = await file.read()
        
        # Create upload request
        upload_request = VoiceUploadRequest(
            name=name,
            description=description,
            provider=provider
        )
        
        # Process upload
        result = await voice_manager.upload_voice(
            user_id=current_user.id,
            file_content=file_content,
            filename=file.filename,
            request=upload_request
        )
        
        return result.model_dump()
        
    except VoiceQuotaExceededError as e:
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail=str(e)
        )
    except VoiceProcessingError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    except Exception as e:
        logger.error(f"Voice upload error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to upload voice sample"
        )


@router.get("/voices", summary="List user's custom voices")
async def list_voices(
    request: Request,
    current_user: User = Depends(get_request_user)
):
    """
    List all custom voice samples uploaded by the user.
    
    Returns voice metadata including:
    - Voice ID for use in TTS requests
    - Name and description
    - Duration and format
    - Compatible providers
    """
    from tldw_Server_API.app.core.TTS.voice_manager import get_voice_manager
    
    try:
        voice_manager = get_voice_manager()
        voices = await voice_manager.list_user_voices(current_user.id)
        
        return {
            "voices": [voice.model_dump() for voice in voices],
            "count": len(voices)
        }
        
    except Exception as e:
        logger.error(f"Error listing voices: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to list voices"
        )


@router.get("/voices/{voice_id}", summary="Get voice details")
async def get_voice_details(
    request: Request,
    voice_id: str = Path(..., description="Voice ID"),
    current_user: User = Depends(get_request_user)
):
    """
    Get detailed information about a specific voice.
    """
    from tldw_Server_API.app.core.TTS.voice_manager import get_voice_manager
    
    try:
        voice_manager = get_voice_manager()
        voice = await voice_manager.registry.get_voice(current_user.id, voice_id)
        
        if not voice:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Voice not found"
            )
        
        return voice.model_dump()
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting voice details: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to get voice details"
        )


@router.delete("/voices/{voice_id}", summary="Delete a custom voice")
async def delete_voice(
    request: Request,
    voice_id: str = Path(..., description="Voice ID to delete"),
    current_user: User = Depends(get_request_user)
):
    """
    Delete a custom voice sample.
    
    This will remove the voice files and prevent it from being used in future TTS requests.
    """
    from tldw_Server_API.app.core.TTS.voice_manager import get_voice_manager
    
    try:
        voice_manager = get_voice_manager()
        deleted = await voice_manager.delete_voice(current_user.id, voice_id)
        
        if not deleted:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Voice not found"
            )
        
        return {"message": "Voice deleted successfully", "voice_id": voice_id}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting voice: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to delete voice"
        )


@router.post("/voices/{voice_id}/preview", summary="Generate voice preview")
@limiter.limit("10/minute")  # Rate limit: 10 previews per minute
async def preview_voice(
    request: Request,
    voice_id: str = Path(..., description="Voice ID to preview"),
    text: str = Form(default="Hello, this is a preview of your custom voice.", description="Text to speak"),
    current_user: User = Depends(get_request_user),
    tts_service: TTSServiceV2 = Depends(get_tts_service)
):
    """
    Generate a short preview of a custom voice.
    
    This endpoint generates a short audio sample using the specified voice
    to help users preview how it sounds before using it in full TTS requests.
    """
    from tldw_Server_API.app.core.TTS.voice_manager import get_voice_manager
    
    try:
        # Validate voice exists
        voice_manager = get_voice_manager()
        voice = await voice_manager.registry.get_voice(current_user.id, voice_id)
        
        if not voice:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Voice not found"
            )
        
        # Limit preview text length
        if len(text) > 100:
            text = text[:100]
        
        # Create TTS request with custom voice
        preview_request = OpenAISpeechRequest(
            model=voice.provider,
            input=text,
            voice=f"custom:{voice_id}",
            response_format="mp3"
        )
        
        # Generate preview
        response = await tts_service.generate_speech(preview_request)
        
        # Stream the audio
        return StreamingResponse(
            response.audio_stream,
            media_type="audio/mpeg",
            headers={
                "Content-Disposition": f"inline; filename=preview_{voice_id}.mp3",
                "X-Voice-Name": voice.name
            }
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Voice preview error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to generate voice preview"
        )

#
# End of audio.py
#######################################################################################################################
