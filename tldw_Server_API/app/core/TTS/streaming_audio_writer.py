# streaming_audio_writer.py
# Description: Handles streaming audio format conversions for TTS
# Based on Kokoro-FastAPI implementation
#
# Imports
import struct
from io import BytesIO
from typing import Optional
#
# Third-party Imports
import av
import numpy as np
from loguru import logger
#
# Local Imports
#
#######################################################################################################################
#
# Functions:

class StreamingAudioWriter:
    """Handles streaming audio format conversions for TTS output"""

    def __init__(self, format: str, sample_rate: int, channels: int = 1):
        """
        Initialize the streaming audio writer.
        
        Args:
            format: Target audio format (wav, mp3, opus, flac, aac, pcm)
            sample_rate: Sample rate in Hz
            channels: Number of audio channels (1 for mono, 2 for stereo)
        """
        self.format = format.lower()
        self.sample_rate = sample_rate
        self.channels = channels
        self.bytes_written = 0
        self.pts = 0  # Presentation timestamp for audio frames

        # Map formats to codecs
        codec_map = {
            "wav": "pcm_s16le",
            "mp3": "mp3",
            "opus": "libopus",
            "flac": "flac",
            "aac": "aac",
        }
        
        # Format-specific setup
        if self.format in ["wav", "flac", "mp3", "pcm", "aac", "opus"]:
            if self.format != "pcm":
                self.output_buffer = BytesIO()
                container_options = {}
                
                # Disable Xing VBR header for MP3 to fix iOS timeline reading issues
                if self.format == 'mp3':
                    container_options = {'write_xing': '0'}
                    logger.debug("Disabling Xing VBR header for MP3 encoding.")

                # Open the container for writing
                self.container = av.open(
                    self.output_buffer,
                    mode="w",
                    format=self.format if self.format != "aac" else "adts",
                    options=container_options
                )
                
                # Add audio stream with appropriate codec
                self.stream = self.container.add_stream(
                    codec_map[self.format],
                    rate=self.sample_rate,
                    layout="mono" if self.channels == 1 else "stereo",
                )
                
                # Set bit rate for applicable codecs
                if self.format in ['mp3', 'aac', 'opus']:
                    # Use reasonable default bitrates
                    if self.format == 'mp3':
                        self.stream.bit_rate = 128000  # 128 kbps
                    elif self.format == 'aac':
                        self.stream.bit_rate = 96000   # 96 kbps
                    elif self.format == 'opus':
                        self.stream.bit_rate = 64000   # 64 kbps
        else:
            raise ValueError(f"Unsupported audio format: {self.format}")

    def write_chunk(
        self, audio_data: Optional[np.ndarray] = None, finalize: bool = False
    ) -> bytes:
        """
        Write a chunk of audio data and return bytes in the target format.
        
        Args:
            audio_data: NumPy array of audio samples (int16)
            finalize: If True, flush and finalize the stream
            
        Returns:
            Bytes of encoded audio in the target format
        """
        if finalize:
            if self.format != "pcm":
                # Flush stream encoder
                packets = self.stream.encode(None)
                for packet in packets:
                    self.container.mux(packet)
                
                # Close the container to write final data
                self.container.close()
                
                # Get the final bytes from the buffer
                self.output_buffer.seek(0)
                data = self.output_buffer.read()
                self.output_buffer.close()
                return data
            else:
                return b""

        if audio_data is None or len(audio_data) == 0:
            return b""

        if self.format == "pcm":
            # For PCM, just return raw bytes
            return audio_data.tobytes()
        else:
            # Create audio frame from numpy array
            frame = av.AudioFrame.from_ndarray(
                audio_data.reshape(1, -1),
                format="s16",
                layout="mono" if self.channels == 1 else "stereo"
            )
            frame.sample_rate = self.sample_rate
            frame.pts = self.pts
            self.pts += frame.samples

            # Encode the frame
            packets = self.stream.encode(frame)
            for packet in packets:
                self.container.mux(packet)

            # For streaming, we need to read what's been written so far
            # This is not ideal for all formats but works for streaming
            current_pos = self.output_buffer.tell()
            self.output_buffer.seek(self.bytes_written)
            data = self.output_buffer.read(current_pos - self.bytes_written)
            self.bytes_written = current_pos
            return data

    def close(self):
        """Clean up resources."""
        if hasattr(self, "container"):
            try:
                if not self.container.closed:
                    self.container.close()
            except Exception as e:
                logger.error(f"Error closing container: {e}")

        if hasattr(self, "output_buffer"):
            try:
                self.output_buffer.close()
            except Exception as e:
                logger.error(f"Error closing output buffer: {e}")


class AudioNormalizer:
    """Normalizes audio data for consistent output"""
    
    def normalize(self, audio_data: np.ndarray, target_dtype=np.int16) -> np.ndarray:
        """
        Normalize audio data to target dtype with proper scaling.
        
        Args:
            audio_data: Input audio as numpy array (typically float32)
            target_dtype: Target data type (default int16)
            
        Returns:
            Normalized audio array
        """
        if audio_data.dtype == target_dtype:
            return audio_data
        
        # Handle float to int16 conversion
        if audio_data.dtype in [np.float32, np.float64]:
            # Clip to [-1, 1] range
            audio_data = np.clip(audio_data, -1.0, 1.0)
            
            if target_dtype == np.int16:
                # Scale to int16 range
                return (audio_data * 32767).astype(np.int16)
            elif target_dtype == np.int32:
                # Scale to int32 range
                return (audio_data * 2147483647).astype(np.int32)
        
        # Handle int to float conversion
        elif target_dtype in [np.float32, np.float64]:
            if audio_data.dtype == np.int16:
                return audio_data.astype(target_dtype) / 32767.0
            elif audio_data.dtype == np.int32:
                return audio_data.astype(target_dtype) / 2147483647.0
        
        # Default: just cast
        return audio_data.astype(target_dtype)

#
# End of streaming_audio_writer.py
#######################################################################################################################