# Audio_Live_Transcription_Nemo.py
#########################################
# Live/Streaming Transcription with Nemo Models
# This module provides real-time transcription capabilities using Nemo models
# with support for streaming audio, VAD, and continuous transcription.
#
####################
# Function List
#
# 1. NemoLiveTranscriber - Main class for live transcription with Nemo models
# 2. NemoStreamingTranscriber - Streaming transcription for long audio streams
# 3. create_live_transcriber() - Factory function to create appropriate transcriber
#
####################

import queue
import threading
import time
import logging
import numpy as np
from typing import Optional, Callable, List, Dict, Any, Union
from dataclasses import dataclass
from enum import Enum
import pyaudio
import torch

# Import Nemo transcription functions
from tldw_Server_API.app.core.Ingestion_Media_Processing.Audio.Audio_Transcription_Nemo import (
    load_canary_model,
    load_parakeet_model,
    transcribe_with_nemo
)
from tldw_Server_API.app.core.config import load_and_log_configs, loaded_config_data


class TranscriptionMode(Enum):
    """Transcription mode for live audio processing."""
    CONTINUOUS = "continuous"  # Continuous transcription without pause detection
    VAD_BASED = "vad_based"    # Voice Activity Detection based
    SILENCE_BASED = "silence_based"  # Simple silence-based segmentation


@dataclass
class LiveTranscriptionConfig:
    """Configuration for live transcription."""
    model: str = 'parakeet'  # 'parakeet' or 'canary'
    variant: str = 'standard'  # For Parakeet: 'standard', 'onnx', 'mlx'
    language: Optional[str] = None  # For Canary: 'en', 'es', 'de', 'fr'
    sample_rate: int = 16000
    chunk_size: int = 1024
    mode: TranscriptionMode = TranscriptionMode.SILENCE_BASED
    silence_threshold: float = 0.01
    silence_duration: float = 1.5
    buffer_duration: float = 30.0  # Maximum buffer duration in seconds
    enable_partial: bool = True  # Enable partial transcriptions
    partial_interval: float = 3.0  # Interval for partial transcriptions
    device: Optional[str] = None  # 'cpu' or 'cuda', None for auto-detect


class NemoLiveTranscriber:
    """
    Live audio transcriber using Nemo models.
    
    This class provides real-time transcription with support for:
    - Continuous streaming transcription
    - VAD-based segmentation
    - Silence-based segmentation
    - Partial transcriptions for real-time feedback
    """
    
    def __init__(
        self,
        config: Optional[LiveTranscriptionConfig] = None,
        on_transcription: Optional[Callable[[str], None]] = None,
        on_partial: Optional[Callable[[str], None]] = None
    ):
        """
        Initialize the live transcriber.
        
        Args:
            config: Configuration for live transcription
            on_transcription: Callback for final transcriptions
            on_partial: Callback for partial transcriptions
        """
        self.config = config or LiveTranscriptionConfig()
        self.on_transcription = on_transcription or self._default_handler
        self.on_partial = on_partial or self._default_partial_handler
        
        # Audio setup
        self.pa = pyaudio.PyAudio()
        self.stream = None
        self.audio_queue = queue.Queue()
        self.is_recording = False
        self.stop_event = threading.Event()
        
        # Threads
        self.listener_thread = None
        self.partial_thread = None
        
        # Audio buffer management
        self.audio_buffer = []
        self.partial_buffer = []
        self.silence_start_time = None
        self.last_partial_time = time.time()
        
        # Model loading (lazy)
        self.model = None
        self.model_loaded = False
        
        # VAD setup (if needed)
        self.vad = None
        if self.config.mode == TranscriptionMode.VAD_BASED:
            self._setup_vad()
    
    def _setup_vad(self):
        """Setup Voice Activity Detection."""
        try:
            import webrtcvad
            self.vad = webrtcvad.Vad(2)  # Aggressiveness level 2
            logging.info("VAD initialized for live transcription")
        except ImportError:
            logging.warning("webrtcvad not installed, falling back to silence-based detection")
            self.config.mode = TranscriptionMode.SILENCE_BASED
    
    def _load_model(self):
        """Load the Nemo model if not already loaded."""
        if self.model_loaded:
            return
        
        try:
            if self.config.model.lower() == 'canary':
                self.model = load_canary_model()
            else:  # parakeet
                self.model = load_parakeet_model(self.config.variant)
            
            if self.model is not None:
                self.model_loaded = True
                logging.info(f"Loaded {self.config.model} model for live transcription")
            else:
                logging.error(f"Failed to load {self.config.model} model")
        except Exception as e:
            logging.error(f"Error loading model: {e}")
    
    def _default_handler(self, text: str):
        """Default handler for transcriptions."""
        print(f"[Transcription]: {text}")
    
    def _default_partial_handler(self, text: str):
        """Default handler for partial transcriptions."""
        print(f"[Partial]: {text}")
    
    def audio_callback(self, in_data, frame_count, time_info, status):
        """PyAudio callback for audio streaming."""
        if status:
            logging.debug(f"Stream status: {status}")
        
        if not self.is_recording:
            return (in_data, pyaudio.paContinue)
        
        # Convert to numpy array
        if self.config.mode == TranscriptionMode.VAD_BASED:
            # For VAD, we need int16
            audio_data = np.frombuffer(in_data, dtype=np.int16)
        else:
            # For amplitude-based, use float32
            audio_data = np.frombuffer(in_data, dtype=np.float32)
        
        self.audio_queue.put(audio_data.copy())
        return (in_data, pyaudio.paContinue)
    
    def start(self):
        """Start live transcription."""
        if self.is_recording:
            logging.warning("Already recording")
            return
        
        # Load model
        self._load_model()
        if not self.model_loaded:
            logging.error("Cannot start transcription - model not loaded")
            return
        
        self.is_recording = True
        self.stop_event.clear()
        
        # Open audio stream
        format_type = pyaudio.paInt16 if self.config.mode == TranscriptionMode.VAD_BASED else pyaudio.paFloat32
        
        self.stream = self.pa.open(
            format=format_type,
            channels=1,
            rate=self.config.sample_rate,
            input=True,
            frames_per_buffer=self.config.chunk_size,
            stream_callback=self.audio_callback
        )
        
        self.stream.start_stream()
        
        # Start processing threads
        self.listener_thread = threading.Thread(target=self._listen_loop, daemon=True)
        self.listener_thread.start()
        
        if self.config.enable_partial:
            self.partial_thread = threading.Thread(target=self._partial_loop, daemon=True)
            self.partial_thread.start()
        
        logging.info(f"Started live transcription with {self.config.model} model")
    
    def stop(self):
        """Stop live transcription."""
        if not self.is_recording:
            return
        
        self.is_recording = False
        self.stop_event.set()
        
        # Stop stream
        if self.stream:
            self.stream.stop_stream()
            self.stream.close()
            self.stream = None
        
        # Wait for threads
        if self.listener_thread:
            self.listener_thread.join(timeout=2)
        if self.partial_thread:
            self.partial_thread.join(timeout=2)
        
        # Process any remaining audio
        if self.audio_buffer:
            self._process_buffer(self.audio_buffer)
            self.audio_buffer.clear()
        
        logging.info("Stopped live transcription")
    
    def _listen_loop(self):
        """Main listening loop for processing audio."""
        while not self.stop_event.is_set():
            try:
                # Get audio chunk
                try:
                    chunk = self.audio_queue.get(timeout=0.1)
                except queue.Empty:
                    continue
                
                # Add to buffers
                self.audio_buffer.append(chunk)
                if self.config.enable_partial:
                    self.partial_buffer.append(chunk)
                
                # Process based on mode
                if self.config.mode == TranscriptionMode.CONTINUOUS:
                    self._process_continuous()
                elif self.config.mode == TranscriptionMode.VAD_BASED:
                    self._process_vad()
                else:  # SILENCE_BASED
                    self._process_silence_based(chunk)
                
            except Exception as e:
                logging.error(f"Error in listen loop: {e}")
    
    def _process_continuous(self):
        """Process audio in continuous mode."""
        # Check buffer duration
        buffer_samples = sum(len(c) for c in self.audio_buffer)
        buffer_duration = buffer_samples / self.config.sample_rate
        
        if buffer_duration >= self.config.buffer_duration:
            # Process and clear buffer
            self._process_buffer(self.audio_buffer)
            self.audio_buffer.clear()
    
    def _process_vad(self):
        """Process audio using VAD."""
        if not self.vad:
            self._process_silence_based(self.audio_buffer[-1] if self.audio_buffer else None)
            return
        
        # Check if we have enough data for VAD (needs specific frame sizes)
        # WebRTC VAD expects 10, 20, or 30 ms frames at 8, 16, 32, or 48 kHz
        frame_duration_ms = 30
        frame_size = int(self.config.sample_rate * frame_duration_ms / 1000)
        
        if len(self.audio_buffer) * self.config.chunk_size >= frame_size:
            # Concatenate buffer
            audio_data = np.concatenate(self.audio_buffer)
            
            # Process frames
            is_speech = False
            for i in range(0, len(audio_data) - frame_size, frame_size):
                frame = audio_data[i:i+frame_size].tobytes()
                is_speech = self.vad.is_speech(frame, self.config.sample_rate)
                if is_speech:
                    break
            
            if not is_speech:
                # Silence detected, process buffer
                if len(self.audio_buffer) > 0:
                    self._process_buffer(self.audio_buffer)
                    self.audio_buffer.clear()
    
    def _process_silence_based(self, chunk):
        """Process audio using silence detection."""
        if chunk is None:
            return
        
        # Calculate amplitude
        if chunk.dtype == np.int16:
            # Convert to float for amplitude calculation
            chunk_float = chunk.astype(np.float32) / 32768.0
            amplitude = np.abs(chunk_float).mean()
        else:
            amplitude = np.abs(chunk).mean()
        
        if amplitude < self.config.silence_threshold:
            # Silence detected
            if self.silence_start_time is None:
                self.silence_start_time = time.time()
            else:
                elapsed = time.time() - self.silence_start_time
                if elapsed >= self.config.silence_duration and len(self.audio_buffer) > 0:
                    # Process buffer
                    self._process_buffer(self.audio_buffer)
                    self.audio_buffer.clear()
                    self.silence_start_time = None
        else:
            # Speech detected, reset silence timer
            self.silence_start_time = None
    
    def _process_buffer(self, buffer: List[np.ndarray]):
        """Process audio buffer for transcription."""
        if not buffer:
            return
        
        try:
            # Concatenate buffer
            audio_data = np.concatenate(buffer)
            
            # Convert to float32 if needed
            if audio_data.dtype == np.int16:
                audio_data = audio_data.astype(np.float32) / 32768.0
            
            # Transcribe using Nemo
            text = transcribe_with_nemo(
                audio_data,
                sample_rate=self.config.sample_rate,
                model=self.config.model,
                variant=self.config.variant,
                language=self.config.language
            )
            
            if text and not text.startswith("[Error"):
                self.on_transcription(text)
        
        except Exception as e:
            logging.error(f"Error processing buffer: {e}")
    
    def _partial_loop(self):
        """Loop for generating partial transcriptions."""
        while not self.stop_event.is_set():
            try:
                time.sleep(0.5)  # Check every 500ms
                
                now = time.time()
                if now - self.last_partial_time < self.config.partial_interval:
                    continue
                
                if self.partial_buffer:
                    # Process partial buffer
                    self._process_partial()
                    self.last_partial_time = now
            
            except Exception as e:
                logging.error(f"Error in partial loop: {e}")
    
    def _process_partial(self):
        """Process partial buffer for interim results."""
        if not self.partial_buffer:
            return
        
        try:
            # Take a copy of current buffer
            buffer_copy = self.partial_buffer.copy()
            
            # Concatenate
            audio_data = np.concatenate(buffer_copy)
            
            # Convert if needed
            if audio_data.dtype == np.int16:
                audio_data = audio_data.astype(np.float32) / 32768.0
            
            # Limit partial buffer size
            max_partial_samples = int(self.config.sample_rate * 10)  # Max 10 seconds
            if len(audio_data) > max_partial_samples:
                audio_data = audio_data[-max_partial_samples:]
            
            # Transcribe
            text = transcribe_with_nemo(
                audio_data,
                sample_rate=self.config.sample_rate,
                model=self.config.model,
                variant=self.config.variant,
                language=self.config.language
            )
            
            if text and not text.startswith("[Error"):
                self.on_partial(text)
            
            # Clear old partial buffer data
            self.partial_buffer = self.partial_buffer[-5:]  # Keep last 5 chunks
        
        except Exception as e:
            logging.error(f"Error processing partial: {e}")
    
    def __enter__(self):
        """Context manager entry."""
        self.start()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.stop()
        if self.pa:
            self.pa.terminate()


class NemoStreamingTranscriber:
    """
    Streaming transcriber for processing long audio streams or files.
    Optimized for streaming scenarios where audio is fed in chunks.
    """
    
    def __init__(
        self,
        model: str = 'parakeet',
        variant: str = 'standard',
        language: Optional[str] = None,
        chunk_duration: float = 5.0,
        overlap_duration: float = 0.5
    ):
        """
        Initialize streaming transcriber.
        
        Args:
            model: Model to use ('parakeet' or 'canary')
            variant: Model variant for Parakeet
            language: Language for Canary
            chunk_duration: Duration of each chunk to process
            overlap_duration: Overlap between chunks for context
        """
        self.model_name = model
        self.variant = variant
        self.language = language
        self.chunk_duration = chunk_duration
        self.overlap_duration = overlap_duration
        
        self.model = None
        self.buffer = []
        self.transcriptions = []
        self.sample_rate = 16000
    
    def initialize(self, sample_rate: int = 16000):
        """Initialize the model and settings."""
        self.sample_rate = sample_rate
        
        # Load model
        if self.model_name.lower() == 'canary':
            self.model = load_canary_model()
        else:
            self.model = load_parakeet_model(self.variant)
        
        if self.model is None:
            raise RuntimeError(f"Failed to load {self.model_name} model")
        
        logging.info(f"Initialized streaming transcriber with {self.model_name}")
    
    def process_chunk(self, audio_chunk: np.ndarray) -> Optional[str]:
        """
        Process an audio chunk and return transcription if available.
        
        Args:
            audio_chunk: Audio data chunk
        
        Returns:
            Transcription text if a segment is complete, None otherwise
        """
        self.buffer.append(audio_chunk)
        
        # Check if we have enough data
        buffer_samples = sum(len(c) for c in self.buffer)
        buffer_duration = buffer_samples / self.sample_rate
        
        if buffer_duration >= self.chunk_duration:
            # Process buffer
            audio_data = np.concatenate(self.buffer)
            
            # Keep overlap for context
            overlap_samples = int(self.overlap_duration * self.sample_rate)
            
            # Transcribe current chunk
            chunk_to_process = audio_data[:int(self.chunk_duration * self.sample_rate)]
            
            text = transcribe_with_nemo(
                chunk_to_process,
                sample_rate=self.sample_rate,
                model=self.model_name,
                variant=self.variant,
                language=self.language
            )
            
            # Update buffer - keep overlap
            if len(audio_data) > overlap_samples:
                remaining = audio_data[-overlap_samples:]
                self.buffer = [remaining]
            else:
                self.buffer.clear()
            
            if text and not text.startswith("[Error"):
                self.transcriptions.append(text)
                return text
        
        return None
    
    def flush(self) -> Optional[str]:
        """
        Process any remaining audio in the buffer.
        
        Returns:
            Final transcription text if buffer has audio, None otherwise
        """
        if self.buffer:
            audio_data = np.concatenate(self.buffer)
            self.buffer.clear()
            
            text = transcribe_with_nemo(
                audio_data,
                sample_rate=self.sample_rate,
                model=self.model_name,
                variant=self.variant,
                language=self.language
            )
            
            if text and not text.startswith("[Error"):
                self.transcriptions.append(text)
                return text
        
        return None
    
    def get_full_transcription(self) -> str:
        """Get the complete transcription of all processed chunks."""
        return " ".join(self.transcriptions)
    
    def reset(self):
        """Reset the transcriber state."""
        self.buffer.clear()
        self.transcriptions.clear()


def create_live_transcriber(
    model: str = 'parakeet',
    mode: str = 'silence_based',
    on_transcription: Optional[Callable[[str], None]] = None,
    on_partial: Optional[Callable[[str], None]] = None,
    **kwargs
) -> NemoLiveTranscriber:
    """
    Factory function to create a live transcriber.
    
    Args:
        model: Model to use ('parakeet' or 'canary')
        mode: Transcription mode ('continuous', 'vad_based', 'silence_based')
        on_transcription: Callback for final transcriptions
        on_partial: Callback for partial transcriptions
        **kwargs: Additional configuration parameters
    
    Returns:
        Configured NemoLiveTranscriber instance
    """
    # Load configuration from system if available
    system_config = loaded_config_data or load_and_log_configs()
    
    # Build configuration
    config = LiveTranscriptionConfig(
        model=model,
        mode=TranscriptionMode[mode.upper()],
        **kwargs
    )
    
    # Override with system config if available
    if system_config and 'STT-Settings' in system_config:
        stt_config = system_config['STT-Settings']
        if 'nemo_model_variant' in stt_config:
            config.variant = stt_config['nemo_model_variant']
        if 'nemo_device' in stt_config:
            config.device = stt_config['nemo_device']
    
    return NemoLiveTranscriber(config, on_transcription, on_partial)


# Example usage functions
def example_live_transcription():
    """Example of using live transcription with Nemo."""
    
    def handle_transcription(text: str):
        print(f"\n[Final]: {text}")
    
    def handle_partial(text: str):
        print(f"\r[Partial]: {text[:50]}...", end='')
    
    # Create transcriber
    transcriber = create_live_transcriber(
        model='parakeet',
        mode='silence_based',
        on_transcription=handle_transcription,
        on_partial=handle_partial,
        variant='standard',
        silence_duration=1.5
    )
    
    # Use as context manager
    print("Starting live transcription. Speak into your microphone...")
    print("Press Ctrl+C to stop.")
    
    try:
        with transcriber:
            while True:
                time.sleep(0.1)
    except KeyboardInterrupt:
        print("\nStopping transcription...")


def example_streaming_transcription(audio_file_path: str):
    """Example of streaming transcription for audio files."""
    import soundfile as sf
    
    # Create streaming transcriber
    transcriber = NemoStreamingTranscriber(
        model='parakeet',
        variant='standard',
        chunk_duration=5.0,
        overlap_duration=0.5
    )
    
    # Load audio file
    audio_data, sample_rate = sf.read(audio_file_path)
    transcriber.initialize(sample_rate)
    
    # Process in chunks
    chunk_size = int(sample_rate * 2)  # 2-second chunks
    
    print("Processing audio file in streaming mode...")
    for i in range(0, len(audio_data), chunk_size):
        chunk = audio_data[i:i+chunk_size]
        text = transcriber.process_chunk(chunk)
        if text:
            print(f"Chunk {i//chunk_size}: {text}")
    
    # Process remaining
    final_text = transcriber.flush()
    if final_text:
        print(f"Final: {final_text}")
    
    # Get complete transcription
    full_text = transcriber.get_full_transcription()
    print(f"\nComplete transcription:\n{full_text}")


if __name__ == "__main__":
    # Run example
    example_live_transcription()