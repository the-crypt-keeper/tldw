# voice_manager.py
# Description: Voice management service for handling custom voice uploads and processing
#
# Imports
import asyncio
import hashlib
import json
import os
import shutil
import tempfile
import uuid
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime, timedelta
#
# Third-party Imports
import aiofiles
from loguru import logger
from pydantic import BaseModel, Field, field_validator
#
# Local Imports
from .tts_exceptions import (
    TTSError,
    TTSInvalidInputError,
    TTSResourceError
)
#
#######################################################################################################################
#
# Voice Management Classes and Service

# Provider requirements for voice samples
PROVIDER_REQUIREMENTS = {
    "vibevoice": {
        "formats": [".wav", ".mp3", ".flac", ".ogg"],
        "max_size_mb": 50,
        "duration": {"min": 0.1, "max": 600},  # 0.1s to 10 minutes
        "sample_rate": 22050,
        "convert_to": "wav"
    },
    "higgs": {
        "formats": [".wav", ".mp3"],
        "max_size_mb": 10,
        "duration": {"min": 3, "max": 10},
        "sample_rate": 16000,
        "convert_to": "wav"
    },
    "chatterbox": {
        "formats": [".wav", ".mp3"],
        "max_size_mb": 20,
        "duration": {"min": 5, "max": 20},
        "sample_rate": 22050,
        "convert_to": "wav"
    },
    "elevenlabs": {
        "formats": [".wav", ".mp3"],
        "max_size_mb": 10,
        "duration": {"min": 1, "max": 30},
        "sample_rate": 44100,
        "convert_to": "mp3"
    }
}

# Rate limiting configuration
VOICE_RATE_LIMITS = {
    "upload_per_hour": 5,
    "total_storage_mb": 500,
    "concurrent_processing": 2,
    "max_voices_per_user": 50
}


class VoiceUploadRequest(BaseModel):
    """Request model for voice upload"""
    name: str = Field(..., min_length=1, max_length=100)
    description: Optional[str] = Field(None, max_length=500)
    provider: str = Field(default="vibevoice")
    
    @field_validator('name')
    @classmethod
    def validate_name(cls, v):
        # Remove any path traversal attempts
        if any(char in v for char in ['/', '\\', '..', '~']):
            raise ValueError("Voice name contains invalid characters")
        return v.strip()


class VoiceInfo(BaseModel):
    """Information about a voice sample"""
    voice_id: str
    name: str
    description: Optional[str] = None
    file_path: str
    format: str
    duration: float
    sample_rate: Optional[int] = None
    size_bytes: int
    provider: str
    created_at: datetime
    file_hash: str


class VoiceUploadResponse(BaseModel):
    """Response model for voice upload"""
    voice_id: str
    name: str
    file_path: str
    duration: float
    format: str
    provider_compatible: bool
    warnings: List[str] = []
    info: str = ""


class VoiceProcessingError(TTSError):
    """Base exception for voice processing errors"""
    pass


class VoiceFormatError(VoiceProcessingError):
    """Invalid audio format error"""
    pass


class VoiceDurationError(VoiceProcessingError):
    """Voice duration outside acceptable range"""
    pass


class VoiceQuotaExceededError(VoiceProcessingError):
    """User exceeded voice storage quota"""
    pass


class VoiceFileValidator:
    """Validates voice file uploads"""
    
    ALLOWED_EXTENSIONS = {'.wav', '.mp3', '.flac', '.ogg', '.m4a', '.opus'}
    ALLOWED_MIME_TYPES = {
        'audio/wav', 'audio/x-wav', 'audio/wave',
        'audio/mpeg', 'audio/mp3',
        'audio/flac', 'audio/x-flac',
        'audio/ogg', 'application/ogg',
        'audio/mp4', 'audio/x-m4a',
        'audio/opus'
    }
    MAX_FILE_SIZE = 50 * 1024 * 1024  # 50MB default
    
    @staticmethod
    def validate_filename(filename: str) -> Tuple[bool, str]:
        """Validate and sanitize filename"""
        if not filename:
            return False, "No filename provided"
        
        # Get extension
        ext = Path(filename).suffix.lower()
        if ext not in VoiceFileValidator.ALLOWED_EXTENSIONS:
            return False, f"File type {ext} not allowed. Allowed types: {', '.join(VoiceFileValidator.ALLOWED_EXTENSIONS)}"
        
        # Sanitize filename
        safe_name = "".join(c for c in filename if c.isalnum() or c in '._- ')
        safe_name = safe_name.replace(' ', '_')
        
        return True, safe_name
    
    @staticmethod
    def validate_file_size(size_bytes: int, provider: str = "vibevoice") -> Tuple[bool, str]:
        """Validate file size for provider"""
        max_size = PROVIDER_REQUIREMENTS.get(provider, {}).get("max_size_mb", 50) * 1024 * 1024
        
        if size_bytes > max_size:
            return False, f"File size {size_bytes / 1024 / 1024:.1f}MB exceeds limit of {max_size / 1024 / 1024}MB"
        
        if size_bytes == 0:
            return False, "File is empty"
        
        return True, ""
    
    @staticmethod
    def sanitize_path(base_path: Path, filename: str) -> Path:
        """Ensure path is safe and within base directory"""
        # Remove any path components from filename
        safe_name = Path(filename).name
        full_path = base_path / safe_name
        
        # Resolve to absolute path and check it's within base
        try:
            full_path = full_path.resolve()
            base_path = base_path.resolve()
            if not str(full_path).startswith(str(base_path)):
                raise ValueError("Path traversal attempt detected")
        except (ValueError, RuntimeError) as e:
            raise VoiceProcessingError(f"Invalid file path: {e}")
        
        return full_path


class VoiceRegistry:
    """In-memory registry for voice samples"""
    
    def __init__(self):
        self.user_voices: Dict[int, Dict[str, VoiceInfo]] = {}
        self._lock = asyncio.Lock()
    
    async def register_voice(self, user_id: int, voice_info: VoiceInfo):
        """Register a voice in the runtime registry"""
        async with self._lock:
            if user_id not in self.user_voices:
                self.user_voices[user_id] = {}
            
            self.user_voices[user_id][voice_info.voice_id] = voice_info
            logger.info(f"Registered voice {voice_info.name} ({voice_info.voice_id}) for user {user_id}")
    
    async def get_voice(self, user_id: int, voice_id: str) -> Optional[VoiceInfo]:
        """Get voice info from registry"""
        async with self._lock:
            return self.user_voices.get(user_id, {}).get(voice_id)
    
    async def list_voices(self, user_id: int) -> List[VoiceInfo]:
        """List all voices for a user"""
        async with self._lock:
            return list(self.user_voices.get(user_id, {}).values())
    
    async def remove_voice(self, user_id: int, voice_id: str) -> bool:
        """Remove voice from registry"""
        async with self._lock:
            if user_id in self.user_voices and voice_id in self.user_voices[user_id]:
                del self.user_voices[user_id][voice_id]
                logger.info(f"Removed voice {voice_id} for user {user_id}")
                return True
            return False
    
    async def clear_user_voices(self, user_id: int):
        """Clear all voices for a user"""
        async with self._lock:
            if user_id in self.user_voices:
                del self.user_voices[user_id]
                logger.info(f"Cleared all voices for user {user_id}")


class VoiceManager:
    """Main voice management service"""
    
    def __init__(self):
        self.registry = VoiceRegistry()
        self.processing_queue = asyncio.Queue()
        self.cleanup_interval = 3600  # 1 hour
        self.user_upload_counts: Dict[int, List[datetime]] = {}
        self._processing_tasks: Dict[str, asyncio.Task] = {}
    
    def get_user_voices_path(self, user_id: int) -> Path:
        """Get the voices directory path for a user"""
        base_path = Path("./Databases/user_databases") / str(user_id) / "voices"
        base_path.mkdir(parents=True, exist_ok=True)
        
        # Create subdirectories
        (base_path / "uploads").mkdir(exist_ok=True)
        (base_path / "processed").mkdir(exist_ok=True)
        (base_path / "temp").mkdir(exist_ok=True)
        
        return base_path
    
    async def check_rate_limits(self, user_id: int) -> Tuple[bool, str]:
        """Check if user is within rate limits"""
        now = datetime.utcnow()
        hour_ago = now - timedelta(hours=1)
        
        # Clean old entries and count recent uploads
        if user_id not in self.user_upload_counts:
            self.user_upload_counts[user_id] = []
        
        self.user_upload_counts[user_id] = [
            dt for dt in self.user_upload_counts[user_id]
            if dt > hour_ago
        ]
        
        if len(self.user_upload_counts[user_id]) >= VOICE_RATE_LIMITS["upload_per_hour"]:
            return False, f"Rate limit exceeded: {VOICE_RATE_LIMITS['upload_per_hour']} uploads per hour"
        
        # Check total storage
        voices_path = self.get_user_voices_path(user_id)
        total_size = sum(
            f.stat().st_size for f in voices_path.rglob("*") if f.is_file()
        )
        
        max_storage = VOICE_RATE_LIMITS["total_storage_mb"] * 1024 * 1024
        if total_size > max_storage:
            return False, f"Storage quota exceeded: {total_size / 1024 / 1024:.1f}MB / {VOICE_RATE_LIMITS['total_storage_mb']}MB"
        
        # Check voice count
        voice_count = len(await self.registry.list_voices(user_id))
        if voice_count >= VOICE_RATE_LIMITS["max_voices_per_user"]:
            return False, f"Maximum voice limit reached: {VOICE_RATE_LIMITS['max_voices_per_user']} voices"
        
        return True, ""
    
    async def upload_voice(
        self,
        user_id: int,
        file_content: bytes,
        filename: str,
        request: VoiceUploadRequest
    ) -> VoiceUploadResponse:
        """Process a voice upload"""
        
        # Check rate limits
        can_upload, error_msg = await self.check_rate_limits(user_id)
        if not can_upload:
            raise VoiceQuotaExceededError(error_msg)
        
        # Validate filename
        is_valid, safe_filename = VoiceFileValidator.validate_filename(filename)
        if not is_valid:
            raise VoiceFormatError(safe_filename)
        
        # Validate file size
        is_valid, error_msg = VoiceFileValidator.validate_file_size(len(file_content), request.provider)
        if not is_valid:
            raise VoiceProcessingError(error_msg)
        
        # Generate unique ID
        voice_id = str(uuid.uuid4())
        
        # Save original file
        voices_path = self.get_user_voices_path(user_id)
        upload_path = voices_path / "uploads" / f"{voice_id}_{safe_filename}"
        
        try:
            async with aiofiles.open(upload_path, 'wb') as f:
                await f.write(file_content)
            
            logger.info(f"Saved uploaded voice file: {upload_path}")
            
            # Calculate file hash
            file_hash = hashlib.sha256(file_content).hexdigest()
            
            # Get audio duration (simplified - in production use ffprobe or similar)
            duration = await self._get_audio_duration(upload_path)
            
            # Validate duration for provider
            provider_reqs = PROVIDER_REQUIREMENTS.get(request.provider, {})
            min_duration = provider_reqs.get("duration", {}).get("min", 0)
            max_duration = provider_reqs.get("duration", {}).get("max", float('inf'))
            
            warnings = []
            if duration < min_duration:
                warnings.append(f"Audio duration {duration:.1f}s is less than recommended {min_duration}s for {request.provider}")
            elif duration > max_duration:
                warnings.append(f"Audio duration {duration:.1f}s exceeds maximum {max_duration}s for {request.provider}")
            
            # Process for provider (convert format if needed)
            processed_path = await self._process_for_provider(
                upload_path, 
                voices_path / "processed" / f"{voice_id}.wav",
                request.provider
            )
            
            # Create voice info
            voice_info = VoiceInfo(
                voice_id=voice_id,
                name=request.name,
                description=request.description,
                file_path=str(processed_path.relative_to(voices_path)),
                format=processed_path.suffix[1:],
                duration=duration,
                sample_rate=provider_reqs.get("sample_rate"),
                size_bytes=processed_path.stat().st_size,
                provider=request.provider,
                created_at=datetime.utcnow(),
                file_hash=file_hash
            )
            
            # Register voice
            await self.registry.register_voice(user_id, voice_info)
            
            # Update upload count
            self.user_upload_counts[user_id].append(datetime.utcnow())
            
            return VoiceUploadResponse(
                voice_id=voice_id,
                name=request.name,
                file_path=str(processed_path),
                duration=duration,
                format=voice_info.format,
                provider_compatible=len(warnings) == 0,
                warnings=warnings,
                info=f"Voice '{request.name}' uploaded successfully for {request.provider}"
            )
            
        except Exception as e:
            # Clean up on error
            if upload_path.exists():
                upload_path.unlink()
            logger.error(f"Failed to upload voice: {e}")
            raise VoiceProcessingError(f"Failed to process voice upload: {str(e)}")
    
    async def _get_audio_duration(self, file_path: Path) -> float:
        """Get audio file duration using ffprobe"""
        try:
            import subprocess
            result = subprocess.run(
                [
                    'ffprobe', '-v', 'error',
                    '-show_entries', 'format=duration',
                    '-of', 'default=noprint_wrappers=1:nokey=1',
                    str(file_path)
                ],
                capture_output=True,
                text=True,
                timeout=10
            )
            
            if result.returncode == 0 and result.stdout:
                return float(result.stdout.strip())
            else:
                logger.warning(f"Could not determine audio duration for {file_path}")
                return 0.0
                
        except (subprocess.TimeoutExpired, FileNotFoundError, ValueError) as e:
            logger.error(f"Error getting audio duration: {e}")
            return 0.0
    
    async def _process_for_provider(self, input_path: Path, output_path: Path, provider: str) -> Path:
        """Process audio file for specific provider requirements"""
        provider_reqs = PROVIDER_REQUIREMENTS.get(provider, {})
        target_format = provider_reqs.get("convert_to", "wav")
        target_sr = provider_reqs.get("sample_rate", 22050)
        
        # If already in correct format and sample rate, just copy
        if input_path.suffix[1:] == target_format:
            shutil.copy2(input_path, output_path)
            return output_path
        
        # Convert using ffmpeg
        try:
            import subprocess
            
            # Ensure output has correct extension
            output_path = output_path.with_suffix(f".{target_format}")
            
            cmd = [
                'ffmpeg', '-y', '-i', str(input_path),
                '-ar', str(target_sr),  # Sample rate
                '-ac', '1',  # Mono
                '-c:a', 'pcm_s16le' if target_format == 'wav' else 'libmp3lame',
                str(output_path)
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
            
            if result.returncode != 0:
                logger.error(f"FFmpeg conversion failed: {result.stderr}")
                # Fall back to copying original
                shutil.copy2(input_path, output_path)
            else:
                logger.info(f"Converted audio to {target_format} at {target_sr}Hz")
            
            return output_path
            
        except Exception as e:
            logger.error(f"Audio conversion failed: {e}")
            # Fall back to copying original
            shutil.copy2(input_path, output_path)
            return output_path
    
    async def list_user_voices(self, user_id: int) -> List[VoiceInfo]:
        """List all voices for a user"""
        # Get from registry
        voices = await self.registry.list_voices(user_id)
        
        # If empty, scan filesystem
        if not voices:
            voices = await self._scan_user_voices(user_id)
        
        return voices
    
    async def _scan_user_voices(self, user_id: int) -> List[VoiceInfo]:
        """Scan filesystem for user's voices"""
        voices = []
        voices_path = self.get_user_voices_path(user_id)
        processed_path = voices_path / "processed"
        
        if processed_path.exists():
            for voice_file in processed_path.glob("*"):
                if voice_file.is_file() and voice_file.suffix in VoiceFileValidator.ALLOWED_EXTENSIONS:
                    try:
                        # Extract voice ID from filename
                        voice_id = voice_file.stem
                        
                        # Get file info
                        stat = voice_file.stat()
                        duration = await self._get_audio_duration(voice_file)
                        
                        # Create voice info
                        voice_info = VoiceInfo(
                            voice_id=voice_id,
                            name=voice_id,  # Use ID as name if not stored
                            file_path=str(voice_file.relative_to(voices_path)),
                            format=voice_file.suffix[1:],
                            duration=duration,
                            size_bytes=stat.st_size,
                            provider="vibevoice",  # Default provider
                            created_at=datetime.fromtimestamp(stat.st_ctime),
                            file_hash=""  # Would need to calculate
                        )
                        
                        voices.append(voice_info)
                        
                        # Register in memory
                        await self.registry.register_voice(user_id, voice_info)
                        
                    except Exception as e:
                        logger.error(f"Error scanning voice file {voice_file}: {e}")
        
        return voices
    
    async def delete_voice(self, user_id: int, voice_id: str) -> bool:
        """Delete a voice"""
        # Get voice info
        voice_info = await self.registry.get_voice(user_id, voice_id)
        if not voice_info:
            return False
        
        # Delete files
        voices_path = self.get_user_voices_path(user_id)
        
        # Delete processed file
        processed_file = voices_path / voice_info.file_path
        if processed_file.exists():
            processed_file.unlink()
        
        # Delete original upload if exists
        for upload_file in (voices_path / "uploads").glob(f"{voice_id}_*"):
            upload_file.unlink()
        
        # Remove from registry
        await self.registry.remove_voice(user_id, voice_id)
        
        logger.info(f"Deleted voice {voice_id} for user {user_id}")
        return True
    
    async def cleanup_temp_files(self):
        """Clean up old temporary files"""
        try:
            # Clean all user temp directories
            base_path = Path("./Databases/user_databases")
            if base_path.exists():
                for user_dir in base_path.iterdir():
                    if user_dir.is_dir():
                        temp_dir = user_dir / "voices" / "temp"
                        if temp_dir.exists():
                            # Remove files older than 1 hour
                            cutoff_time = datetime.utcnow().timestamp() - 3600
                            for temp_file in temp_dir.iterdir():
                                if temp_file.is_file() and temp_file.stat().st_mtime < cutoff_time:
                                    temp_file.unlink()
                                    logger.debug(f"Cleaned up temp file: {temp_file}")
        
        except Exception as e:
            logger.error(f"Error during temp file cleanup: {e}")
    
    async def start_background_tasks(self):
        """Start background processing tasks"""
        # Start cleanup task
        asyncio.create_task(self._cleanup_worker())
        logger.info("Voice manager background tasks started")
    
    async def _cleanup_worker(self):
        """Background worker for cleanup"""
        while True:
            try:
                await asyncio.sleep(self.cleanup_interval)
                await self.cleanup_temp_files()
            except Exception as e:
                logger.error(f"Cleanup worker error: {e}")


# Global instance
_voice_manager: Optional[VoiceManager] = None


def get_voice_manager() -> VoiceManager:
    """Get or create the global voice manager instance"""
    global _voice_manager
    if _voice_manager is None:
        _voice_manager = VoiceManager()
    return _voice_manager


async def init_voice_manager():
    """Initialize the voice manager and start background tasks"""
    manager = get_voice_manager()
    await manager.start_background_tasks()
    logger.info("Voice manager initialized")