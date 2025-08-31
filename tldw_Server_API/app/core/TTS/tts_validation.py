# tts_validation.py
# Description: Input validation and sanitization for TTS requests
#
# Imports
import re
import html
import unicodedata
from typing import Dict, Any, List, Optional, Tuple, Union
from pathlib import Path
import mimetypes
#
# Third-party Imports
from loguru import logger
#
# Local Imports
from .tts_exceptions import (
    TTSValidationError,
    TTSInvalidInputError,
    TTSTextTooLongError,
    TTSUnsupportedFormatError,
    TTSUnsupportedLanguageError,
    TTSVoiceNotFoundError,
    TTSInvalidVoiceReferenceError,
    validation_error
)
from .adapters.base import AudioFormat, TTSRequest
#
#######################################################################################################################
#
# Provider Limits


class ProviderLimits:
    """Provider-specific limits and constraints"""
    
    LIMITS = {
        "openai": {
            "max_text_length": 4096,
            "valid_voices": {"alloy", "echo", "fable", "onyx", "nova", "shimmer"},
            "valid_formats": {"mp3", "opus", "aac", "flac", "wav", "pcm"},
            "min_speed": 0.25,
            "max_speed": 4.0
        },
        "elevenlabs": {
            "max_text_length": 5000,
            "valid_formats": {"mp3", "pcm", "ulaw"},
            "min_stability": 0.0,
            "max_stability": 1.0,
            "min_similarity": 0.0,
            "max_similarity": 1.0
        },
        "kokoro": {
            "max_text_length": 10000,
            "valid_formats": {"wav", "mp3"},
            "min_speed": 0.5,
            "max_speed": 2.0
        },
        "higgs": {
            "max_text_length": 8000,
            "valid_formats": {"wav", "mp3", "opus"},
            "min_speed": 0.5,
            "max_speed": 2.0
        },
        "dia": {
            "max_text_length": 10000,
            "valid_formats": {"wav", "mp3"},
            "min_speed": 0.5,
            "max_speed": 2.0,
            "max_speakers": 4
        },
        "chatterbox": {
            "max_text_length": 10000,
            "valid_formats": {"wav", "mp3"},
            "min_speed": 0.5,
            "max_speed": 2.0
        },
        "vibevoice": {
            "max_text_length": 10000,
            "valid_formats": {"wav", "mp3"},
            "min_speed": 0.5,
            "max_speed": 2.0,
            "max_speakers": 4
        }
    }
    
    @classmethod
    def get_limits(cls, provider: str) -> Dict[str, Any]:
        """Get limits for a specific provider"""
        return cls.LIMITS.get(provider, {})
    
    @classmethod
    def get_max_text_length(cls, provider: str) -> int:
        """Get maximum text length for provider"""
        limits = cls.get_limits(provider)
        return limits.get("max_text_length", 5000)  # Default 5000
    
    @classmethod
    def is_valid_voice(cls, provider: str, voice: str) -> bool:
        """Check if voice is valid for provider"""
        limits = cls.get_limits(provider)
        valid_voices = limits.get("valid_voices")
        if valid_voices is None:
            return True  # No restriction
        return voice in valid_voices
    
    @classmethod
    def is_valid_format(cls, provider: str, format: str) -> bool:
        """Check if format is valid for provider"""
        limits = cls.get_limits(provider)
        valid_formats = limits.get("valid_formats", {"mp3", "wav"})
        return format.lower() in valid_formats


#
# Input Validation and Sanitization

class TTSInputValidator:
    """
    Comprehensive input validator for TTS requests.
    Handles text sanitization, format validation, and security checks.
    """
    
    # Security patterns to detect potential injection attacks
    DANGEROUS_PATTERNS = [
        r'<script[^>]*>.*?</script>',  # Script tags
        r'javascript:',                # JavaScript URLs
        r'vbscript:',                 # VBScript URLs
        r'on\w+\s*=',                 # Event handlers
        r'expression\s*\(',           # CSS expressions
        r'@import',                   # CSS imports
        r'\\x[0-9a-fA-F]{2}',        # Hex escapes
        r'\\u[0-9a-fA-F]{4}',        # Unicode escapes
        r'&#[0-9]+;',                 # HTML numeric entities
        r'&#x[0-9a-fA-F]+;',         # HTML hex entities
    ]
    
    # Compiled regex patterns for performance
    DANGEROUS_REGEX = [re.compile(pattern, re.IGNORECASE | re.DOTALL) for pattern in DANGEROUS_PATTERNS]
    
    # Maximum text length per provider (characters)
    MAX_TEXT_LENGTHS = {
        "openai": 4096,
        "elevenlabs": 5000,
        "kokoro": 1000,
        "higgs": 2000,
        "dia": 1500,
        "chatterbox": 2000,
        "vibevoice": 3000,
        "default": 1000
    }
    
    # Supported languages by provider
    SUPPORTED_LANGUAGES = {
        "openai": {"en", "es", "fr", "de", "it", "pt", "ru", "ja", "ko", "zh", "ar", "hi"},
        "elevenlabs": {"en", "es", "fr", "de", "it", "pt", "pl", "hi", "ar", "zh", "ja", "ko"},
        "kokoro": {"en"},
        "higgs": {"en", "es", "fr", "de", "it", "pt", "ru", "ja", "ko", "zh"},
        "dia": {"en"},
        "chatterbox": {"en"},
        "vibevoice": {"en", "es", "fr", "de", "it", "pt", "ru", "ja", "ko", "zh", "ar", "hi"}
    }
    
    # Supported audio formats by provider
    SUPPORTED_FORMATS = {
        "openai": {AudioFormat.MP3, AudioFormat.OPUS, AudioFormat.AAC, AudioFormat.FLAC, AudioFormat.WAV, AudioFormat.PCM},
        "elevenlabs": {AudioFormat.MP3, AudioFormat.WAV, AudioFormat.OPUS},
        "kokoro": {AudioFormat.MP3, AudioFormat.WAV, AudioFormat.OPUS},
        "higgs": {AudioFormat.MP3, AudioFormat.WAV, AudioFormat.FLAC},
        "dia": {AudioFormat.MP3, AudioFormat.WAV},
        "chatterbox": {AudioFormat.MP3, AudioFormat.WAV, AudioFormat.OPUS},
        "vibevoice": {AudioFormat.MP3, AudioFormat.WAV, AudioFormat.FLAC, AudioFormat.OPUS}
    }
    
    # Voice reference file validation
    VOICE_REF_MAX_SIZE = 50 * 1024 * 1024  # 50MB
    VOICE_REF_MAX_DURATION = 300  # 5 minutes
    VOICE_REF_ALLOWED_FORMATS = {".mp3", ".wav", ".flac", ".opus", ".m4a", ".ogg"}
    VOICE_REF_ALLOWED_MIME_TYPES = {
        "audio/mpeg", "audio/wav", "audio/x-wav", "audio/flac",
        "audio/opus", "audio/ogg", "audio/mp4", "audio/x-m4a"
    }
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the validator with configuration.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config or {}
        self.strict_mode = self.config.get("strict_validation", True)
        self.allow_html = self.config.get("allow_html", False)
        self.max_text_length_override = self.config.get("max_text_length")
        logger.debug(f"TTSInputValidator initialized (strict_mode={self.strict_mode})")
    
    def sanitize_text(self, text: str, provider: Optional[str] = None) -> str:
        """
        Sanitize input text for TTS generation.
        
        Args:
            text: Input text to sanitize
            provider: TTS provider name for provider-specific rules
            
        Returns:
            Sanitized text
            
        Raises:
            TTSInvalidInputError: If text contains dangerous content
        """
        if not text or not text.strip():
            raise TTSInvalidInputError("Text cannot be empty or whitespace only")
        
        original_text = text
        
        # 1. Normalize Unicode characters
        text = unicodedata.normalize('NFKC', text)
        
        # 2. Check for dangerous patterns
        if self.strict_mode:
            for pattern in self.DANGEROUS_REGEX:
                if pattern.search(text):
                    logger.warning(f"Dangerous pattern detected in text: {pattern.pattern[:50]}")
                    raise TTSInvalidInputError(
                        "Text contains potentially dangerous content",
                        details={"pattern": pattern.pattern[:50]}
                    )
        
        # 3. HTML handling
        if not self.allow_html:
            # Escape HTML characters
            text = html.escape(text, quote=True)
        else:
            # Allow HTML but sanitize dangerous tags
            text = self._sanitize_html(text)
        
        # 4. Remove or replace problematic characters
        text = self._clean_control_characters(text)
        
        # 5. Provider-specific sanitization
        if provider:
            text = self._provider_specific_sanitization(text, provider)
        
        # 6. Final validation
        if len(text.strip()) == 0:
            raise TTSInvalidInputError("Text became empty after sanitization")
        
        logger.debug(f"Text sanitized: {len(original_text)} -> {len(text)} chars")
        return text.strip()
    
    def validate_text_length(self, text: str, provider: Optional[str] = None):
        """Public method to validate text length"""
        return self._validate_text(text, provider)
    
    def validate_language(self, language: str, provider: Optional[str] = None):
        """Public method to validate language"""
        return self._validate_language(language, provider)
    
    def validate_format(self, format: AudioFormat, provider: Optional[str] = None):
        """Public method to validate format"""
        return self._validate_format(format, provider)
    
    def validate_parameters(self, request: TTSRequest):
        """Public method to validate parameters"""
        return self._validate_parameters(request)
    
    def validate_voice_reference(self, voice_ref_data: bytes):
        """Public method to validate voice reference"""
        return self._validate_voice_reference(voice_ref_data)
    
    def validate_request(self, request: TTSRequest, provider: Optional[str] = None) -> Tuple[bool, Optional[str]]:
        """
        Validate a complete TTS request.
        
        Args:
            request: TTS request to validate
            provider: TTS provider name
            
        Returns:
            Tuple of (is_valid, error_message)
        """
        try:
            # Validate text
            self._validate_text(request.text, provider)
            
            # Validate format
            if request.format:
                self._validate_format(request.format, provider)
            
            # Validate language
            if request.language:
                self._validate_language(request.language, provider)
            
            # Validate voice
            if request.voice:
                self._validate_voice(request.voice, provider)
            
            # Validate parameters
            self._validate_parameters(request)
            
            # Validate voice reference if provided
            if request.voice_reference:
                self._validate_voice_reference(request.voice_reference)
            
            return True, None
            
        except TTSValidationError as e:
            return False, str(e)
        except Exception as e:
            logger.error(f"Unexpected validation error: {e}")
            return False, f"Validation failed: {str(e)}"
    
    def _validate_text(self, text: str, provider: Optional[str] = None):
        """Validate text content"""
        if not text or not text.strip():
            raise TTSInvalidInputError("Text cannot be empty")
        
        # Check length limits
        max_length = self.max_text_length_override or self.MAX_TEXT_LENGTHS.get(provider, self.MAX_TEXT_LENGTHS["default"])
        
        if len(text) > max_length:
            raise TTSTextTooLongError(
                f"Text length ({len(text)}) exceeds maximum of {max_length} characters",
                provider=provider,
                details={"text_length": len(text), "max_length": max_length}
            )
        
        # Check for excessive repetition (potential abuse)
        if self._has_excessive_repetition(text):
            raise TTSInvalidInputError(
                "Text contains excessive repetition",
                provider=provider
            )
    
    def _validate_format(self, format: AudioFormat, provider: Optional[str] = None):
        """Validate audio format"""
        if provider and provider in self.SUPPORTED_FORMATS:
            if format not in self.SUPPORTED_FORMATS[provider]:
                supported = [fmt.value for fmt in self.SUPPORTED_FORMATS[provider]]
                raise TTSUnsupportedFormatError(
                    f"Format '{format.value}' not supported by {provider}. Supported: {supported}",
                    provider=provider,
                    details={"requested_format": format.value, "supported_formats": supported}
                )
    
    def _validate_language(self, language: str, provider: Optional[str] = None):
        """Validate language code"""
        if provider and provider in self.SUPPORTED_LANGUAGES:
            if language not in self.SUPPORTED_LANGUAGES[provider]:
                supported = list(self.SUPPORTED_LANGUAGES[provider])
                raise TTSUnsupportedLanguageError(
                    f"Language '{language}' not supported by {provider}. Supported: {supported}",
                    provider=provider,
                    details={"requested_language": language, "supported_languages": supported}
                )
    
    def _validate_voice(self, voice: str, provider: Optional[str] = None):
        """Validate voice selection"""
        # Basic voice name validation
        if not re.match(r'^[a-zA-Z0-9_-]+$', voice):
            raise TTSVoiceNotFoundError(
                f"Invalid voice name format: {voice}",
                provider=provider
            )
        
        # Length check
        if len(voice) > 100:
            raise TTSVoiceNotFoundError(
                "Voice name too long",
                provider=provider
            )
    
    def _validate_parameters(self, request: TTSRequest):
        """Validate TTS parameters"""
        # Speed validation
        if request.speed < 0.1 or request.speed > 3.0:
            raise TTSInvalidInputError(
                f"Speed must be between 0.1 and 3.0, got {request.speed}"
            )
        
        # Pitch validation
        if request.pitch < -20.0 or request.pitch > 20.0:
            raise TTSInvalidInputError(
                f"Pitch must be between -20.0 and 20.0, got {request.pitch}"
            )
        
        # Volume validation
        if request.volume < 0.0 or request.volume > 2.0:
            raise TTSInvalidInputError(
                f"Volume must be between 0.0 and 2.0, got {request.volume}"
            )
        
        # Emotion intensity validation
        if request.emotion_intensity < 0.0 or request.emotion_intensity > 2.0:
            raise TTSInvalidInputError(
                f"Emotion intensity must be between 0.0 and 2.0, got {request.emotion_intensity}"
            )
    
    def _validate_voice_reference(self, voice_ref_data: bytes):
        """Validate voice reference audio for cloning"""
        if len(voice_ref_data) == 0:
            raise TTSInvalidVoiceReferenceError("Voice reference data is empty")
        
        if len(voice_ref_data) > self.VOICE_REF_MAX_SIZE:
            raise TTSInvalidVoiceReferenceError(
                f"Voice reference file too large: {len(voice_ref_data)} bytes (max: {self.VOICE_REF_MAX_SIZE})",
                details={"file_size": len(voice_ref_data), "max_size": self.VOICE_REF_MAX_SIZE}
            )
        
        # Basic file type validation (check magic bytes)
        if not self._is_valid_audio_file(voice_ref_data):
            raise TTSInvalidVoiceReferenceError(
                "Voice reference file is not a valid audio format"
            )
    
    def _sanitize_html(self, text: str) -> str:
        """Sanitize HTML content while preserving safe tags"""
        # For now, just escape everything - can be enhanced with a proper HTML sanitizer
        return html.escape(text, quote=True)
    
    def _clean_control_characters(self, text: str) -> str:
        """Remove or replace control characters"""
        # Remove most control characters but keep common whitespace
        cleaned = re.sub(r'[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]', '', text)
        
        # Replace multiple spaces/tabs with single space but preserve newlines
        cleaned = re.sub(r'[ \t]+', ' ', cleaned)
        
        # Replace multiple newlines with double newline
        cleaned = re.sub(r'\n{3,}', '\n\n', cleaned)
        
        return cleaned
    
    def _provider_specific_sanitization(self, text: str, provider: str) -> str:
        """Apply provider-specific text sanitization"""
        if provider == "openai":
            # OpenAI specific rules
            return text
        elif provider == "elevenlabs":
            # ElevenLabs specific rules
            return text
        elif provider in ["kokoro", "higgs", "dia", "chatterbox", "vibevoice"]:
            # Local model specific rules - more conservative
            # Remove URLs and email addresses
            text = re.sub(r'https?://\S+', '[URL]', text)
            text = re.sub(r'\S+@\S+\.\S+', '[EMAIL]', text)
            return text
        
        return text
    
    def _has_excessive_repetition(self, text: str) -> bool:
        """Check for excessive character or word repetition"""
        # Check for repeated characters (like "aaaaaaa")
        if re.search(r'(.)\1{10,}', text):
            return True
        
        # Check for repeated words
        words = text.lower().split()
        if len(words) > 10:
            word_counts = {}
            for word in words:
                word_counts[word] = word_counts.get(word, 0) + 1
                # If any word appears more than 30% of the time, it's excessive
                if word_counts[word] > len(words) * 0.3:
                    return True
        
        return False
    
    def _is_valid_audio_file(self, data: bytes) -> bool:
        """Check if data starts with valid audio file magic bytes"""
        if len(data) < 4:
            return False
        
        # Check common audio file signatures
        signatures = [
            b'ID3',      # MP3 with ID3
            b'\xff\xfb', # MP3
            b'\xff\xf3', # MP3
            b'\xff\xf2', # MP3
            b'RIFF',     # WAV
            b'fLaC',     # FLAC
            b'OggS',     # OGG/OPUS
            b'FORM',     # AIFF
        ]
        
        for sig in signatures:
            if data.startswith(sig):
                return True
        
        # Check for MP4/M4A (more complex)
        if len(data) >= 8 and data[4:8] == b'ftyp':
            return True
        
        return False


# Convenience validation functions
def validate_text_input(text: str, provider: Optional[str] = None, config: Optional[Dict[str, Any]] = None) -> str:
    """
    Validate and sanitize text input for TTS.
    
    Args:
        text: Input text
        provider: TTS provider name
        config: Validation configuration
        
    Returns:
        Sanitized text
        
    Raises:
        TTSInvalidInputError: If text is invalid
    """
    validator = TTSInputValidator(config)
    return validator.sanitize_text(text, provider)


def validate_tts_request(request: TTSRequest, provider: Optional[str] = None, config: Optional[Dict[str, Any]] = None) -> None:
    """
    Validate complete TTS request.
    
    Args:
        request: TTS request to validate
        provider: TTS provider name
        config: Validation configuration
        
    Raises:
        TTSValidationError: If request is invalid
    """
    validator = TTSInputValidator(config)
    is_valid, error_message = validator.validate_request(request, provider)
    
    if not is_valid:
        raise TTSValidationError(error_message, provider=provider)


def validate_voice_reference(voice_ref_data: bytes, config: Optional[Dict[str, Any]] = None) -> None:
    """
    Validate voice reference audio data.
    
    Args:
        voice_ref_data: Voice reference audio bytes
        config: Validation configuration
        
    Raises:
        TTSInvalidVoiceReferenceError: If voice reference is invalid
    """
    validator = TTSInputValidator(config)
    validator._validate_voice_reference(voice_ref_data)

#
# End of tts_validation.py
#######################################################################################################################