# audio_utils.py
# Description: Audio processing utilities for TTS voice cloning/reference
#
# Imports
import base64
import io
import tempfile
from pathlib import Path
from typing import Optional, Tuple, Dict, Any
#
# Third-party Imports
import numpy as np
from loguru import logger
#
#######################################################################################################################
#
# Audio Processing Utilities for Voice Cloning

class AudioProcessor:
    """Process and validate audio for voice cloning/reference"""
    
    # Supported audio formats
    SUPPORTED_FORMATS = {'.wav', '.mp3', '.flac', '.ogg', '.m4a'}
    
    # Provider-specific requirements
    PROVIDER_REQUIREMENTS = {
        'higgs': {
            'min_duration': 3.0,
            'max_duration': 10.0,
            'preferred_sample_rate': 24000,
            'formats': {'.wav', '.mp3', '.flac'}
        },
        'chatterbox': {
            'min_duration': 5.0,
            'max_duration': 20.0,
            'preferred_sample_rate': 24000,
            'formats': {'.wav', '.mp3'}
        },
        'vibevoice': {
            'min_duration': 3.0,
            'max_duration': 30.0,
            'preferred_sample_rate': 22050,
            'formats': {'.wav', '.mp3'}
        }
    }
    
    def __init__(self):
        """Initialize audio processor"""
        self.ffmpeg_available = self._check_ffmpeg()
        self.librosa_available = self._check_librosa()
    
    def _check_ffmpeg(self) -> bool:
        """Check if ffmpeg is available"""
        try:
            import subprocess
            result = subprocess.run(['ffmpeg', '-version'], capture_output=True, text=True)
            return result.returncode == 0
        except:
            logger.warning("ffmpeg not found - audio conversion will be limited")
            return False
    
    def _check_librosa(self) -> bool:
        """Check if librosa is available for audio processing"""
        try:
            import librosa
            return True
        except ImportError:
            logger.warning("librosa not installed - advanced audio processing unavailable")
            return False
    
    def decode_base64_audio(self, base64_data: str) -> bytes:
        """
        Decode base64-encoded audio data.
        
        Args:
            base64_data: Base64-encoded audio string
            
        Returns:
            Raw audio bytes
            
        Raises:
            ValueError: If decoding fails
        """
        try:
            # Remove data URL prefix if present
            if ',' in base64_data:
                base64_data = base64_data.split(',')[1]
            
            # Decode base64
            audio_bytes = base64.b64decode(base64_data)
            return audio_bytes
            
        except Exception as e:
            raise ValueError(f"Failed to decode base64 audio: {e}")
    
    def encode_audio_base64(self, audio_bytes: bytes) -> str:
        """
        Encode audio bytes to base64.
        
        Args:
            audio_bytes: Raw audio bytes
            
        Returns:
            Base64-encoded string
        """
        return base64.b64encode(audio_bytes).decode('utf-8')
    
    def validate_audio(
        self,
        audio_bytes: bytes,
        provider: str,
        check_duration: bool = True,
        check_quality: bool = False
    ) -> Tuple[bool, Optional[str], Dict[str, Any]]:
        """
        Validate audio for a specific provider.
        
        Args:
            audio_bytes: Raw audio bytes
            provider: TTS provider name
            check_duration: Whether to check duration requirements
            check_quality: Whether to check audio quality (noise, etc.)
            
        Returns:
            Tuple of (is_valid, error_message, audio_info)
        """
        info = {}
        
        if provider.lower() not in self.PROVIDER_REQUIREMENTS:
            return True, None, info  # No specific requirements
        
        requirements = self.PROVIDER_REQUIREMENTS[provider.lower()]
        
        if not self.librosa_available:
            logger.warning("Cannot validate audio without librosa")
            return True, None, info
        
        try:
            import librosa
            import soundfile as sf
            
            # Write to temporary file for processing
            with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp_file:
                tmp_file.write(audio_bytes)
                tmp_path = tmp_file.name
            
            try:
                # Load audio
                audio_data, sample_rate = librosa.load(tmp_path, sr=None)
                
                # Get audio info
                info['sample_rate'] = sample_rate
                info['duration'] = len(audio_data) / sample_rate
                info['channels'] = 1 if audio_data.ndim == 1 else audio_data.shape[0]
                
                # Check duration
                if check_duration:
                    if info['duration'] < requirements['min_duration']:
                        return False, f"Audio too short: {info['duration']:.1f}s (minimum {requirements['min_duration']}s)", info
                    if info['duration'] > requirements['max_duration']:
                        return False, f"Audio too long: {info['duration']:.1f}s (maximum {requirements['max_duration']}s)", info
                
                # Check quality (optional)
                if check_quality:
                    # Check for silence
                    rms = np.sqrt(np.mean(audio_data**2))
                    if rms < 0.001:
                        return False, "Audio appears to be silent", info
                    
                    # Check for clipping
                    if np.any(np.abs(audio_data) > 0.99):
                        info['has_clipping'] = True
                        logger.warning("Audio may have clipping")
                
                return True, None, info
                
            finally:
                # Clean up temp file
                Path(tmp_path).unlink(missing_ok=True)
                
        except Exception as e:
            logger.error(f"Audio validation error: {e}")
            return False, f"Failed to validate audio: {e}", info
    
    def convert_audio(
        self,
        audio_bytes: bytes,
        target_format: str = 'wav',
        target_sample_rate: Optional[int] = None,
        provider: Optional[str] = None
    ) -> bytes:
        """
        Convert audio to target format and sample rate.
        
        Args:
            audio_bytes: Input audio bytes
            target_format: Target format (wav, mp3, flac)
            target_sample_rate: Target sample rate (Hz)
            provider: Provider name for specific requirements
            
        Returns:
            Converted audio bytes
        """
        if not self.ffmpeg_available and not self.librosa_available:
            logger.warning("No audio conversion libraries available")
            return audio_bytes
        
        # Get provider requirements if specified
        if provider and provider.lower() in self.PROVIDER_REQUIREMENTS:
            requirements = self.PROVIDER_REQUIREMENTS[provider.lower()]
            if not target_sample_rate:
                target_sample_rate = requirements['preferred_sample_rate']
        
        try:
            if self.librosa_available:
                import librosa
                import soundfile as sf
                
                # Write input to temp file
                with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as input_file:
                    input_file.write(audio_bytes)
                    input_path = input_file.name
                
                # Output temp file
                output_suffix = f'.{target_format.lower()}'
                with tempfile.NamedTemporaryFile(suffix=output_suffix, delete=False) as output_file:
                    output_path = output_file.name
                
                try:
                    # Load and resample
                    audio_data, original_sr = librosa.load(input_path, sr=None)
                    
                    if target_sample_rate and target_sample_rate != original_sr:
                        audio_data = librosa.resample(
                            audio_data,
                            orig_sr=original_sr,
                            target_sr=target_sample_rate
                        )
                        sample_rate = target_sample_rate
                    else:
                        sample_rate = original_sr
                    
                    # Write output
                    sf.write(output_path, audio_data, sample_rate, format=target_format.upper())
                    
                    # Read converted audio
                    with open(output_path, 'rb') as f:
                        converted_bytes = f.read()
                    
                    return converted_bytes
                    
                finally:
                    # Clean up temp files
                    Path(input_path).unlink(missing_ok=True)
                    Path(output_path).unlink(missing_ok=True)
                    
            elif self.ffmpeg_available:
                import subprocess
                
                # Use ffmpeg for conversion
                with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as input_file:
                    input_file.write(audio_bytes)
                    input_path = input_file.name
                
                output_suffix = f'.{target_format.lower()}'
                with tempfile.NamedTemporaryFile(suffix=output_suffix, delete=False) as output_file:
                    output_path = output_file.name
                
                try:
                    # Build ffmpeg command
                    cmd = ['ffmpeg', '-i', input_path, '-y']
                    
                    if target_sample_rate:
                        cmd.extend(['-ar', str(target_sample_rate)])
                    
                    cmd.append(output_path)
                    
                    # Run conversion
                    result = subprocess.run(cmd, capture_output=True, text=True)
                    if result.returncode != 0:
                        raise RuntimeError(f"ffmpeg conversion failed: {result.stderr}")
                    
                    # Read converted audio
                    with open(output_path, 'rb') as f:
                        converted_bytes = f.read()
                    
                    return converted_bytes
                    
                finally:
                    # Clean up temp files
                    Path(input_path).unlink(missing_ok=True)
                    Path(output_path).unlink(missing_ok=True)
                    
        except Exception as e:
            logger.error(f"Audio conversion failed: {e}")
            return audio_bytes  # Return original if conversion fails
    
    def extract_clean_segment(
        self,
        audio_bytes: bytes,
        target_duration: float = 10.0
    ) -> bytes:
        """
        Extract a clean segment from audio (remove silence, find best part).
        
        Args:
            audio_bytes: Input audio bytes
            target_duration: Target duration in seconds
            
        Returns:
            Extracted audio segment bytes
        """
        if not self.librosa_available:
            return audio_bytes
        
        try:
            import librosa
            import soundfile as sf
            
            # Write to temp file
            with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp_file:
                tmp_file.write(audio_bytes)
                tmp_path = tmp_file.name
            
            try:
                # Load audio
                audio_data, sample_rate = librosa.load(tmp_path, sr=None)
                
                # Remove silence from beginning and end
                audio_trimmed, _ = librosa.effects.trim(audio_data, top_db=20)
                
                # If still too long, take the middle portion
                if len(audio_trimmed) / sample_rate > target_duration:
                    target_samples = int(target_duration * sample_rate)
                    start = (len(audio_trimmed) - target_samples) // 2
                    audio_trimmed = audio_trimmed[start:start + target_samples]
                
                # Write output
                output_path = tmp_path + '_clean.wav'
                sf.write(output_path, audio_trimmed, sample_rate)
                
                # Read cleaned audio
                with open(output_path, 'rb') as f:
                    clean_bytes = f.read()
                
                Path(output_path).unlink(missing_ok=True)
                return clean_bytes
                
            finally:
                Path(tmp_path).unlink(missing_ok=True)
                
        except Exception as e:
            logger.error(f"Failed to extract clean segment: {e}")
            return audio_bytes


def process_voice_reference(
    base64_audio: str,
    provider: str,
    validate: bool = True,
    convert: bool = True
) -> Tuple[Optional[bytes], Optional[str]]:
    """
    Process voice reference audio for a specific provider.
    
    Args:
        base64_audio: Base64-encoded audio data
        provider: TTS provider name
        validate: Whether to validate audio
        convert: Whether to convert to optimal format
        
    Returns:
        Tuple of (processed_audio_bytes, error_message)
    """
    processor = AudioProcessor()
    
    try:
        # Decode base64
        audio_bytes = processor.decode_base64_audio(base64_audio)
        
        # Validate if requested
        if validate:
            is_valid, error_msg, info = processor.validate_audio(
                audio_bytes,
                provider,
                check_duration=True,
                check_quality=True
            )
            
            if not is_valid:
                return None, error_msg
            
            logger.info(f"Voice reference validated: {info}")
        
        # Convert if requested
        if convert and provider.lower() in processor.PROVIDER_REQUIREMENTS:
            requirements = processor.PROVIDER_REQUIREMENTS[provider.lower()]
            audio_bytes = processor.convert_audio(
                audio_bytes,
                target_format='wav',
                provider=provider
            )
            logger.info(f"Voice reference converted for {provider}")
        
        return audio_bytes, None
        
    except Exception as e:
        logger.error(f"Failed to process voice reference: {e}")
        return None, str(e)

#
# End of audio_utils.py
#######################################################################################################################