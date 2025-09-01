# audio_converter.py
# Description: Audio conversion and processing utilities for voice samples
#
# Imports
import asyncio
import os
import subprocess
import tempfile
from pathlib import Path
from typing import Optional, Tuple, Dict, Any
#
# Third-party Imports
import numpy as np
from loguru import logger
#
# Local Imports
from .tts_exceptions import TTSError
#
#######################################################################################################################
#
# Audio Conversion Service

class AudioConversionError(TTSError):
    """Error during audio conversion"""
    pass


class AudioConverter:
    """Audio conversion and processing utilities"""
    
    # Common audio formats and their codecs
    AUDIO_CODECS = {
        'wav': 'pcm_s16le',
        'mp3': 'libmp3lame',
        'flac': 'flac',
        'ogg': 'libvorbis',
        'opus': 'libopus',
        'm4a': 'aac'
    }
    
    @staticmethod
    async def convert_to_wav(
        input_path: Path,
        output_path: Path,
        sample_rate: int = 22050,
        channels: int = 1,
        bit_depth: int = 16
    ) -> bool:
        """
        Convert audio file to WAV format with specified parameters.
        
        Args:
            input_path: Path to input audio file
            output_path: Path for output WAV file
            sample_rate: Target sample rate in Hz
            channels: Number of audio channels (1=mono, 2=stereo)
            bit_depth: Bit depth (16 or 24)
            
        Returns:
            True if conversion successful, False otherwise
        """
        try:
            # Ensure output directory exists
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Build ffmpeg command
            cmd = [
                'ffmpeg', '-y',  # Overwrite output
                '-i', str(input_path),  # Input file
                '-ar', str(sample_rate),  # Sample rate
                '-ac', str(channels),  # Audio channels
                '-sample_fmt', f's{bit_depth}',  # Bit depth
                '-c:a', 'pcm_s16le',  # PCM codec for WAV
                str(output_path)  # Output file
            ]
            
            # Run conversion
            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            
            stdout, stderr = await process.communicate()
            
            if process.returncode != 0:
                logger.error(f"FFmpeg conversion failed: {stderr.decode()}")
                return False
            
            logger.info(f"Converted {input_path.name} to WAV ({sample_rate}Hz, {channels}ch, {bit_depth}bit)")
            return True
            
        except FileNotFoundError:
            logger.error("FFmpeg not found. Please install FFmpeg.")
            raise AudioConversionError("FFmpeg is required for audio conversion")
        except Exception as e:
            logger.error(f"Audio conversion error: {e}")
            return False
    
    @staticmethod
    async def convert_format(
        input_path: Path,
        output_path: Path,
        target_format: str,
        **kwargs
    ) -> bool:
        """
        Convert audio between formats.
        
        Args:
            input_path: Path to input audio file
            output_path: Path for output file
            target_format: Target format (wav, mp3, flac, etc.)
            **kwargs: Additional ffmpeg parameters
            
        Returns:
            True if successful
        """
        try:
            # Get codec for target format
            codec = AudioConverter.AUDIO_CODECS.get(target_format.lower())
            if not codec:
                logger.error(f"Unsupported format: {target_format}")
                return False
            
            # Ensure output has correct extension
            output_path = output_path.with_suffix(f".{target_format.lower()}")
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Build command
            cmd = ['ffmpeg', '-y', '-i', str(input_path)]
            
            # Add optional parameters
            if 'sample_rate' in kwargs:
                cmd.extend(['-ar', str(kwargs['sample_rate'])])
            if 'channels' in kwargs:
                cmd.extend(['-ac', str(kwargs['channels'])])
            if 'bitrate' in kwargs:
                cmd.extend(['-b:a', str(kwargs['bitrate'])])
            
            # Add codec and output
            cmd.extend(['-c:a', codec, str(output_path)])
            
            # Run conversion
            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            
            stdout, stderr = await process.communicate()
            
            if process.returncode != 0:
                logger.error(f"Format conversion failed: {stderr.decode()}")
                return False
            
            logger.info(f"Converted {input_path.name} to {target_format}")
            return True
            
        except Exception as e:
            logger.error(f"Format conversion error: {e}")
            return False
    
    @staticmethod
    async def validate_duration(
        file_path: Path,
        min_seconds: float = 0,
        max_seconds: float = float('inf')
    ) -> Tuple[bool, float]:
        """
        Check if audio duration is within specified range.
        
        Args:
            file_path: Path to audio file
            min_seconds: Minimum duration in seconds
            max_seconds: Maximum duration in seconds
            
        Returns:
            Tuple of (is_valid, actual_duration)
        """
        try:
            duration = await AudioConverter.get_duration(file_path)
            
            is_valid = min_seconds <= duration <= max_seconds
            
            if not is_valid:
                if duration < min_seconds:
                    logger.warning(f"Audio duration {duration:.1f}s is below minimum {min_seconds}s")
                else:
                    logger.warning(f"Audio duration {duration:.1f}s exceeds maximum {max_seconds}s")
            
            return is_valid, duration
            
        except Exception as e:
            logger.error(f"Duration validation error: {e}")
            return False, 0.0
    
    @staticmethod
    async def get_duration(file_path: Path) -> float:
        """
        Get audio file duration in seconds.
        
        Args:
            file_path: Path to audio file
            
        Returns:
            Duration in seconds
        """
        try:
            cmd = [
                'ffprobe', '-v', 'error',
                '-show_entries', 'format=duration',
                '-of', 'default=noprint_wrappers=1:nokey=1',
                str(file_path)
            ]
            
            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            
            stdout, stderr = await process.communicate()
            
            if process.returncode == 0 and stdout:
                return float(stdout.decode().strip())
            else:
                logger.error(f"Could not get duration: {stderr.decode()}")
                return 0.0
                
        except Exception as e:
            logger.error(f"Error getting duration: {e}")
            return 0.0
    
    @staticmethod
    async def get_audio_info(file_path: Path) -> Dict[str, Any]:
        """
        Get detailed audio file information.
        
        Args:
            file_path: Path to audio file
            
        Returns:
            Dictionary with audio properties
        """
        info = {
            'duration': 0.0,
            'sample_rate': 0,
            'channels': 0,
            'codec': '',
            'bitrate': 0,
            'format': file_path.suffix[1:] if file_path.suffix else ''
        }
        
        try:
            # Get duration
            info['duration'] = await AudioConverter.get_duration(file_path)
            
            # Get detailed stream info
            cmd = [
                'ffprobe', '-v', 'error',
                '-select_streams', 'a:0',
                '-show_entries', 'stream=codec_name,sample_rate,channels,bit_rate',
                '-of', 'json',
                str(file_path)
            ]
            
            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            
            stdout, stderr = await process.communicate()
            
            if process.returncode == 0 and stdout:
                import json
                data = json.loads(stdout.decode())
                if data.get('streams'):
                    stream = data['streams'][0]
                    info['codec'] = stream.get('codec_name', '')
                    info['sample_rate'] = int(stream.get('sample_rate', 0))
                    info['channels'] = int(stream.get('channels', 0))
                    info['bitrate'] = int(stream.get('bit_rate', 0))
            
            return info
            
        except Exception as e:
            logger.error(f"Error getting audio info: {e}")
            return info
    
    @staticmethod
    async def normalize_audio(
        input_path: Path,
        output_path: Path,
        target_level: float = -23.0
    ) -> bool:
        """
        Normalize audio loudness using EBU R128 standard.
        
        Args:
            input_path: Path to input audio
            output_path: Path for normalized output
            target_level: Target loudness in LUFS (default -23)
            
        Returns:
            True if successful
        """
        try:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            # First pass: analyze loudness
            cmd_analyze = [
                'ffmpeg', '-i', str(input_path),
                '-af', 'loudnorm=print_format=json',
                '-f', 'null', '-'
            ]
            
            process = await asyncio.create_subprocess_exec(
                *cmd_analyze,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            
            stdout, stderr = await process.communicate()
            
            # Parse loudness info from stderr (ffmpeg outputs to stderr)
            import json
            import re
            
            # Extract JSON from stderr
            json_match = re.search(r'\{[^}]+\}', stderr.decode())
            if not json_match:
                logger.error("Could not analyze audio loudness")
                return False
            
            loudness_info = json.loads(json_match.group())
            
            # Second pass: apply normalization
            cmd_normalize = [
                'ffmpeg', '-y', '-i', str(input_path),
                '-af', f"loudnorm=I={target_level}:TP=-1.5:LRA=11",
                str(output_path)
            ]
            
            process = await asyncio.create_subprocess_exec(
                *cmd_normalize,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            
            stdout, stderr = await process.communicate()
            
            if process.returncode != 0:
                logger.error(f"Normalization failed: {stderr.decode()}")
                return False
            
            logger.info(f"Normalized audio to {target_level} LUFS")
            return True
            
        except Exception as e:
            logger.error(f"Audio normalization error: {e}")
            return False
    
    @staticmethod
    async def trim_silence(
        input_path: Path,
        output_path: Path,
        threshold: float = -40.0,
        duration: float = 0.5
    ) -> bool:
        """
        Trim silence from beginning and end of audio.
        
        Args:
            input_path: Path to input audio
            output_path: Path for trimmed output
            threshold: Silence threshold in dB
            duration: Minimum silence duration to trim (seconds)
            
        Returns:
            True if successful
        """
        try:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Use silenceremove filter
            cmd = [
                'ffmpeg', '-y', '-i', str(input_path),
                '-af', f"silenceremove=start_periods=1:start_duration={duration}:start_threshold={threshold}dB:"
                       f"stop_periods=1:stop_duration={duration}:stop_threshold={threshold}dB",
                str(output_path)
            ]
            
            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            
            stdout, stderr = await process.communicate()
            
            if process.returncode != 0:
                logger.error(f"Silence trimming failed: {stderr.decode()}")
                return False
            
            logger.info(f"Trimmed silence from audio (threshold: {threshold}dB)")
            return True
            
        except Exception as e:
            logger.error(f"Silence trimming error: {e}")
            return False
    
    @staticmethod
    async def extract_segment(
        input_path: Path,
        output_path: Path,
        start_time: float,
        duration: float
    ) -> bool:
        """
        Extract a segment from audio file.
        
        Args:
            input_path: Path to input audio
            output_path: Path for segment output
            start_time: Start time in seconds
            duration: Segment duration in seconds
            
        Returns:
            True if successful
        """
        try:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            cmd = [
                'ffmpeg', '-y',
                '-ss', str(start_time),  # Seek to start
                '-i', str(input_path),
                '-t', str(duration),  # Duration
                '-c', 'copy',  # Copy codec (no re-encoding)
                str(output_path)
            ]
            
            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            
            stdout, stderr = await process.communicate()
            
            if process.returncode != 0:
                logger.error(f"Segment extraction failed: {stderr.decode()}")
                return False
            
            logger.info(f"Extracted {duration}s segment starting at {start_time}s")
            return True
            
        except Exception as e:
            logger.error(f"Segment extraction error: {e}")
            return False
    
    @staticmethod
    async def resample_audio(
        input_path: Path,
        output_path: Path,
        target_sample_rate: int
    ) -> bool:
        """
        Resample audio to target sample rate.
        
        Args:
            input_path: Path to input audio
            output_path: Path for resampled output
            target_sample_rate: Target sample rate in Hz
            
        Returns:
            True if successful
        """
        try:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            cmd = [
                'ffmpeg', '-y', '-i', str(input_path),
                '-ar', str(target_sample_rate),
                '-c:a', 'pcm_s16le' if output_path.suffix == '.wav' else 'copy',
                str(output_path)
            ]
            
            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            
            stdout, stderr = await process.communicate()
            
            if process.returncode != 0:
                logger.error(f"Resampling failed: {stderr.decode()}")
                return False
            
            logger.info(f"Resampled audio to {target_sample_rate}Hz")
            return True
            
        except Exception as e:
            logger.error(f"Resampling error: {e}")
            return False
    
    @staticmethod
    def check_ffmpeg_installed() -> bool:
        """Check if FFmpeg is installed and available."""
        try:
            result = subprocess.run(
                ['ffmpeg', '-version'],
                capture_output=True,
                text=True,
                timeout=5
            )
            return result.returncode == 0
        except (FileNotFoundError, subprocess.TimeoutExpired):
            return False
    
    @staticmethod
    def check_ffprobe_installed() -> bool:
        """Check if FFprobe is installed and available."""
        try:
            result = subprocess.run(
                ['ffprobe', '-version'],
                capture_output=True,
                text=True,
                timeout=5
            )
            return result.returncode == 0
        except (FileNotFoundError, subprocess.TimeoutExpired):
            return False


# Utility function for quick checks
async def validate_audio_tools() -> Tuple[bool, str]:
    """
    Validate that required audio processing tools are installed.
    
    Returns:
        Tuple of (all_tools_available, error_message)
    """
    errors = []
    
    if not AudioConverter.check_ffmpeg_installed():
        errors.append("FFmpeg is not installed or not in PATH")
    
    if not AudioConverter.check_ffprobe_installed():
        errors.append("FFprobe is not installed or not in PATH")
    
    if errors:
        return False, "; ".join(errors)
    
    return True, ""