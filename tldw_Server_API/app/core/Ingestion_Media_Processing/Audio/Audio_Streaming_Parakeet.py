# Audio_Streaming_Parakeet.py
#########################################
# Real-time Streaming Transcription with Parakeet Models
# This module provides WebSocket-based real-time transcription using Parakeet models
# with support for all variants (standard, ONNX, MLX) and chunked processing.
#
####################
# Function List
#
# 1. ParakeetStreamingTranscriber - Main class for streaming transcription
# 2. create_websocket_server() - Create WebSocket server for real-time transcription
# 3. process_audio_stream() - Process streaming audio chunks
#
####################

import asyncio
import base64
import json
import logging
import time
from typing import Optional, AsyncGenerator, Dict, Any, Callable
from dataclasses import dataclass, field
import numpy as np
import soundfile as sf
import tempfile
from pathlib import Path

# Import transcription functions
from .Audio_Transcription_Nemo import (
    transcribe_with_parakeet,
    load_parakeet_model
)
from .Audio_Transcription_Parakeet_MLX import (
    transcribe_with_parakeet_mlx
)

logger = logging.getLogger(__name__)


@dataclass
class StreamingConfig:
    """Configuration for streaming transcription."""
    model_variant: str = 'mlx'  # 'standard', 'onnx', 'mlx'
    sample_rate: int = 16000
    chunk_duration: float = 2.0  # Seconds of audio to accumulate before transcribing
    overlap_duration: float = 0.5  # Overlap between chunks
    max_buffer_duration: float = 30.0  # Maximum buffer size in seconds
    enable_partial: bool = True  # Send partial results
    partial_interval: float = 0.5  # How often to send partials
    language: Optional[str] = None


@dataclass
class AudioBuffer:
    """Manages audio buffering for streaming."""
    sample_rate: int
    max_duration: float
    data: list = field(default_factory=list)
    
    def add(self, audio_chunk: np.ndarray):
        """Add audio chunk to buffer."""
        self.data.append(audio_chunk)
        
        # Trim if buffer exceeds max duration
        total_samples = sum(len(chunk) for chunk in self.data)
        max_samples = int(self.sample_rate * self.max_duration)
        
        if total_samples > max_samples:
            # Remove oldest chunks
            while total_samples > max_samples and self.data:
                removed = self.data.pop(0)
                total_samples -= len(removed)
    
    def get_duration(self) -> float:
        """Get current buffer duration in seconds."""
        if not self.data:
            return 0.0
        total_samples = sum(len(chunk) for chunk in self.data)
        return total_samples / self.sample_rate
    
    def get_audio(self, duration: Optional[float] = None) -> Optional[np.ndarray]:
        """Get audio from buffer."""
        if not self.data:
            return None
        
        combined = np.concatenate(self.data)
        
        if duration is not None:
            samples_needed = int(duration * self.sample_rate)
            if len(combined) >= samples_needed:
                return combined[:samples_needed]
            return None
        
        return combined
    
    def consume(self, duration: float, overlap: float = 0.0):
        """Consume duration from buffer, keeping overlap."""
        if not self.data:
            return
        
        combined = np.concatenate(self.data)
        samples_to_consume = int((duration - overlap) * self.sample_rate)
        
        if samples_to_consume > 0 and len(combined) > samples_to_consume:
            # Keep the remaining audio
            remaining = combined[samples_to_consume:]
            self.data = [remaining]
        else:
            self.data.clear()
    
    def clear(self):
        """Clear the buffer."""
        self.data.clear()


class ParakeetStreamingTranscriber:
    """
    Real-time streaming transcriber using Parakeet models.
    
    Supports WebSocket-based streaming with chunked processing
    and partial results for all Parakeet variants.
    """
    
    def __init__(self, config: Optional[StreamingConfig] = None):
        """Initialize the streaming transcriber."""
        self.config = config or StreamingConfig()
        self.buffer = AudioBuffer(
            sample_rate=self.config.sample_rate,
            max_duration=self.config.max_buffer_duration
        )
        self.model = None
        self.is_running = False
        self.last_partial_time = 0
        self.transcription_history = []
        
    def initialize(self):
        """Load the model based on configuration."""
        if self.config.model_variant == 'mlx':
            # MLX model is loaded on-demand in transcribe function
            logger.info("Using Parakeet MLX variant")
        else:
            # Load standard or ONNX variant
            self.model = load_parakeet_model(self.config.model_variant)
            if self.model is None:
                raise RuntimeError(f"Failed to load Parakeet {self.config.model_variant} model")
            logger.info(f"Loaded Parakeet {self.config.model_variant} model")
    
    async def process_audio_chunk(self, audio_data: bytes) -> Optional[Dict[str, Any]]:
        """
        Process a chunk of audio data.
        
        Args:
            audio_data: Base64-encoded audio data
        
        Returns:
            Transcription result or None if not ready
        """
        try:
            # Decode base64 audio
            audio_bytes = base64.b64decode(audio_data)
            
            # Convert to numpy array (assuming float32 samples)
            audio_array = np.frombuffer(audio_bytes, dtype=np.float32)
            
            # Add to buffer
            self.buffer.add(audio_array)
            
            result = None
            
            # Check if we have enough audio for transcription
            if self.buffer.get_duration() >= self.config.chunk_duration:
                # Get audio chunk
                audio_chunk = self.buffer.get_audio(self.config.chunk_duration)
                
                if audio_chunk is not None:
                    # Transcribe
                    text = await self._transcribe_chunk(audio_chunk)
                    
                    if text and not text.startswith("["):
                        # Consume buffer with overlap
                        self.buffer.consume(
                            self.config.chunk_duration,
                            self.config.overlap_duration
                        )
                        
                        # Add to history
                        self.transcription_history.append(text)
                        
                        result = {
                            "type": "transcription",
                            "text": text,
                            "timestamp": time.time(),
                            "is_final": True
                        }
            
            # Send partial if enabled and enough time has passed
            elif self.config.enable_partial:
                now = time.time()
                if now - self.last_partial_time >= self.config.partial_interval:
                    partial_audio = self.buffer.get_audio()
                    if partial_audio is not None and len(partial_audio) > 0:
                        partial_text = await self._transcribe_chunk(partial_audio)
                        if partial_text and not partial_text.startswith("["):
                            self.last_partial_time = now
                            result = {
                                "type": "partial",
                                "text": partial_text,
                                "timestamp": now,
                                "is_final": False
                            }
            
            return result
            
        except Exception as e:
            logger.error(f"Error processing audio chunk: {e}")
            return {"type": "error", "message": str(e)}
    
    async def _transcribe_chunk(self, audio_chunk: np.ndarray) -> Optional[str]:
        """Transcribe an audio chunk using the appropriate variant."""
        try:
            if self.config.model_variant == 'mlx':
                # Use MLX transcription
                result = await asyncio.to_thread(
                    transcribe_with_parakeet_mlx,
                    audio_chunk,
                    sample_rate=self.config.sample_rate,
                    language=self.config.language
                )
            else:
                # Use standard or ONNX transcription
                result = await asyncio.to_thread(
                    transcribe_with_parakeet,
                    audio_chunk,
                    sample_rate=self.config.sample_rate,
                    variant=self.config.model_variant
                )
            
            return result
            
        except Exception as e:
            logger.error(f"Transcription error: {e}")
            return None
    
    async def flush(self) -> Optional[Dict[str, Any]]:
        """Process any remaining audio in the buffer."""
        if self.buffer.get_duration() > 0:
            audio = self.buffer.get_audio()
            if audio is not None:
                text = await self._transcribe_chunk(audio)
                self.buffer.clear()
                
                if text and not text.startswith("["):
                    self.transcription_history.append(text)
                    return {
                        "type": "final",
                        "text": text,
                        "timestamp": time.time(),
                        "is_final": True
                    }
        return None
    
    def get_full_transcript(self) -> str:
        """Get the complete transcript so far."""
        return " ".join(self.transcription_history)
    
    def reset(self):
        """Reset the transcriber state."""
        self.buffer.clear()
        self.transcription_history.clear()
        self.last_partial_time = 0


async def handle_websocket_transcription(
    websocket,
    config: Optional[StreamingConfig] = None
):
    """
    Handle WebSocket connection for real-time transcription.
    
    Args:
        websocket: WebSocket connection
        config: Streaming configuration
    """
    transcriber = ParakeetStreamingTranscriber(config)
    transcriber.initialize()
    
    try:
        async for message in websocket:
            try:
                data = json.loads(message)
                
                if data.get("type") == "audio":
                    # Process audio chunk
                    result = await transcriber.process_audio_chunk(data["data"])
                    if result:
                        await websocket.send(json.dumps(result))
                
                elif data.get("type") == "commit":
                    # Flush and finalize
                    result = await transcriber.flush()
                    if result:
                        await websocket.send(json.dumps(result))
                    
                    # Send full transcript
                    full_transcript = transcriber.get_full_transcript()
                    await websocket.send(json.dumps({
                        "type": "full_transcript",
                        "text": full_transcript,
                        "timestamp": time.time()
                    }))
                    
                    # Reset for next session
                    transcriber.reset()
                
                elif data.get("type") == "ping":
                    # Keep-alive
                    await websocket.send(json.dumps({"type": "pong"}))
                    
            except json.JSONDecodeError:
                await websocket.send(json.dumps({
                    "type": "error",
                    "message": "Invalid JSON"
                }))
            except Exception as e:
                logger.error(f"Error handling message: {e}")
                await websocket.send(json.dumps({
                    "type": "error",
                    "message": str(e)
                }))
                
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
    finally:
        # Clean up
        final_result = await transcriber.flush()
        if final_result:
            try:
                await websocket.send(json.dumps(final_result))
            except:
                pass


def create_streaming_generator(
    audio_source: Callable[[], bytes],
    config: Optional[StreamingConfig] = None
) -> AsyncGenerator[str, None]:
    """
    Create an async generator for streaming transcription.
    
    Args:
        audio_source: Callable that returns audio bytes
        config: Streaming configuration
    
    Yields:
        Transcribed text segments
    """
    transcriber = ParakeetStreamingTranscriber(config)
    transcriber.initialize()
    
    async def generate():
        try:
            while True:
                # Get audio from source
                audio_bytes = await asyncio.to_thread(audio_source)
                
                if audio_bytes is None:
                    # End of stream
                    final = await transcriber.flush()
                    if final:
                        yield final["text"]
                    break
                
                # Convert to base64 for processing
                audio_b64 = base64.b64encode(audio_bytes).decode('utf-8')
                
                # Process chunk
                result = await transcriber.process_audio_chunk(audio_b64)
                
                if result and result.get("is_final"):
                    yield result["text"]
                    
        except Exception as e:
            logger.error(f"Streaming error: {e}")
            yield f"[Error: {str(e)}]"
    
    return generate()


#######################################################################################################################
# End of Audio_Streaming_Parakeet.py
#######################################################################################################################