# chatbook_validators.py
# Description: Input validation and sanitization for chatbook operations
#
"""
Chatbook Input Validators
-------------------------

Provides comprehensive validation and sanitization for chatbook operations
to prevent security vulnerabilities and ensure data integrity.
"""

import re
import os
import zipfile
from pathlib import Path
from typing import Optional, Tuple, Dict, Any, List
from loguru import logger


class ChatbookValidator:
    """Validator for chatbook operations."""
    
    # File size limits
    MAX_UPLOAD_SIZE = 100 * 1024 * 1024  # 100MB compressed
    MAX_UNCOMPRESSED_SIZE = 500 * 1024 * 1024  # 500MB uncompressed
    MAX_FILE_IN_ARCHIVE = 50 * 1024 * 1024  # 50MB per file
    
    # Filename patterns
    SAFE_FILENAME_PATTERN = re.compile(r'^[a-zA-Z0-9._\-]+$')
    VALID_EXTENSIONS = {'.zip', '.chatbook'}
    
    # Security patterns
    PATH_TRAVERSAL_PATTERN = re.compile(r'(\.\./|\.\.\\|^/|^~)')
    DANGEROUS_PATHS = {'etc', 'usr', 'bin', 'sbin', 'var', 'proc', 'sys', 'dev'}
    
    # Content validation
    MAX_NAME_LENGTH = 255
    MAX_DESCRIPTION_LENGTH = 5000
    MAX_TAGS = 50
    MAX_TAG_LENGTH = 50
    
    @classmethod
    def validate_filename(cls, filename: str) -> Tuple[bool, Optional[str], str]:
        """
        Validate and sanitize a filename.
        
        Args:
            filename: The filename to validate
            
        Returns:
            Tuple of (is_valid, error_message, sanitized_filename)
        """
        if not filename:
            return False, "No filename provided", ""
        
        # Extract basename and remove path components
        base_filename = os.path.basename(filename)
        
        # Check length
        if len(base_filename) > cls.MAX_NAME_LENGTH:
            return False, f"Filename too long (max {cls.MAX_NAME_LENGTH} characters)", ""
        
        # Check extension
        extension = Path(base_filename).suffix.lower()
        if extension not in cls.VALID_EXTENSIONS:
            return False, f"Invalid file type. Allowed: {', '.join(cls.VALID_EXTENSIONS)}", ""
        
        # Sanitize filename - remove dangerous characters
        sanitized = re.sub(r'[^a-zA-Z0-9._\-]', '_', base_filename)
        
        # Prevent double extensions that could bypass filters
        if sanitized.count('.') > 1:
            # Keep only the last extension
            parts = sanitized.rsplit('.', 1)
            if len(parts) == 2:
                name_part = parts[0].replace('.', '_')
                sanitized = f"{name_part}.{parts[1]}"
        
        # Ensure it still has valid extension after sanitization
        if not any(sanitized.lower().endswith(ext) for ext in cls.VALID_EXTENSIONS):
            sanitized = sanitized.rsplit('.', 1)[0] + '.zip'
        
        return True, None, sanitized
    
    @classmethod
    def validate_file_size(cls, file_size: int, compressed: bool = True) -> Tuple[bool, Optional[str]]:
        """
        Validate file size.
        
        Args:
            file_size: Size in bytes
            compressed: Whether this is compressed size
            
        Returns:
            Tuple of (is_valid, error_message)
        """
        max_size = cls.MAX_UPLOAD_SIZE if compressed else cls.MAX_UNCOMPRESSED_SIZE
        max_mb = max_size / (1024 * 1024)
        
        if file_size > max_size:
            size_mb = file_size / (1024 * 1024)
            return False, f"File too large ({size_mb:.1f}MB). Maximum: {max_mb:.0f}MB"
        
        return True, None
    
    @classmethod
    def validate_zip_file(cls, file_path: str) -> Tuple[bool, Optional[str]]:
        """
        Comprehensive validation of a ZIP file.
        
        Args:
            file_path: Path to the ZIP file
            
        Returns:
            Tuple of (is_valid, error_message)
        """
        try:
            file_path_obj = Path(file_path)
            
            # Check file exists
            if not file_path_obj.exists():
                return False, "File does not exist"
            
            # Check file size
            file_size = file_path_obj.stat().st_size
            valid, error = cls.validate_file_size(file_size, compressed=True)
            if not valid:
                return False, error
            
            # Verify ZIP magic number
            with open(file_path, 'rb') as f:
                magic = f.read(4)
                if magic[:2] != b'PK':
                    return False, "Not a valid ZIP file"
            
            # Open and validate ZIP contents
            with zipfile.ZipFile(file_path, 'r') as zf:
                # Test ZIP integrity
                result = zf.testzip()
                if result is not None:
                    return False, f"Corrupted ZIP file: {result}"
                
                # Check total uncompressed size
                total_uncompressed = sum(info.file_size for info in zf.filelist)
                valid, error = cls.validate_file_size(total_uncompressed, compressed=False)
                if not valid:
                    return False, error
                
                # Validate each file in archive
                for info in zf.filelist:
                    # Check for path traversal
                    if cls._is_path_traversal(info.filename):
                        return False, f"Unsafe path in archive: {info.filename}"
                    
                    # Check individual file size
                    if info.file_size > cls.MAX_FILE_IN_ARCHIVE:
                        size_mb = info.file_size / (1024 * 1024)
                        return False, f"File too large in archive: {info.filename} ({size_mb:.1f}MB)"
                    
                    # Check for suspicious file types
                    if cls._is_dangerous_file(info.filename):
                        return False, f"Potentially dangerous file type: {info.filename}"
                
                # Check for required files
                if 'manifest.json' not in zf.namelist():
                    return False, "Invalid chatbook: manifest.json not found"
            
            return True, None
            
        except zipfile.BadZipFile:
            return False, "Invalid or corrupted ZIP file"
        except Exception as e:
            logger.error(f"Error validating ZIP file: {e}")
            return False, f"Error validating file: {str(e)}"
    
    @classmethod
    def validate_chatbook_metadata(cls, 
                                 name: str,
                                 description: str,
                                 tags: List[str] = None,
                                 categories: List[str] = None) -> Tuple[bool, Optional[str]]:
        """
        Validate chatbook metadata.
        
        Args:
            name: Chatbook name
            description: Chatbook description
            tags: List of tags
            categories: List of categories
            
        Returns:
            Tuple of (is_valid, error_message)
        """
        # Validate name
        if not name or not name.strip():
            return False, "Name is required"
        
        if len(name) > cls.MAX_NAME_LENGTH:
            return False, f"Name too long (max {cls.MAX_NAME_LENGTH} characters)"
        
        if not cls._is_safe_text(name):
            return False, "Name contains invalid characters"
        
        # Validate description
        if description and len(description) > cls.MAX_DESCRIPTION_LENGTH:
            return False, f"Description too long (max {cls.MAX_DESCRIPTION_LENGTH} characters)"
        
        # Validate tags
        if tags:
            if len(tags) > cls.MAX_TAGS:
                return False, f"Too many tags (max {cls.MAX_TAGS})"
            
            for tag in tags:
                if len(tag) > cls.MAX_TAG_LENGTH:
                    return False, f"Tag too long: {tag[:20]}... (max {cls.MAX_TAG_LENGTH} characters)"
                if not cls._is_safe_text(tag):
                    return False, f"Tag contains invalid characters: {tag}"
        
        # Validate categories
        if categories:
            if len(categories) > 20:  # Reasonable limit for categories
                return False, "Too many categories (max 20)"
            
            for category in categories:
                if len(category) > cls.MAX_TAG_LENGTH:
                    return False, f"Category too long: {category[:20]}..."
                if not cls._is_safe_text(category):
                    return False, f"Category contains invalid characters: {category}"
        
        return True, None
    
    @classmethod
    def sanitize_path(cls, path: str) -> str:
        """
        Sanitize a path to prevent traversal attacks.
        
        Args:
            path: Path to sanitize
            
        Returns:
            Sanitized path
        """
        # Convert to Path object for normalization
        path_obj = Path(path)
        
        # Get only the name component (no directory traversal)
        safe_name = path_obj.name
        
        # Remove any remaining dangerous patterns
        safe_name = re.sub(r'[^\w\s\-._]', '_', safe_name)
        
        # Ensure no double dots
        safe_name = safe_name.replace('..', '_')
        
        return safe_name
    
    @classmethod
    def validate_content_selections(cls, 
                                   content_selections: Dict[str, List[str]]) -> Tuple[bool, Optional[str]]:
        """
        Validate content selection dictionary.
        
        Args:
            content_selections: Dictionary of content types to IDs
            
        Returns:
            Tuple of (is_valid, error_message)
        """
        if not content_selections:
            return False, "No content selected for export"
        
        # Validate structure
        if not isinstance(content_selections, dict):
            return False, "Invalid content selections format"
        
        # Check for valid content types and IDs
        valid_types = {'conversation', 'note', 'character', 'world_book', 
                      'dictionary', 'generated_document', 'media', 'embedding'}
        
        for content_type, ids in content_selections.items():
            if content_type not in valid_types:
                return False, f"Invalid content type: {content_type}"
            
            if not isinstance(ids, list):
                return False, f"IDs for {content_type} must be a list"
            
            # Validate each ID
            for item_id in ids:
                if not cls._is_valid_id(item_id):
                    return False, f"Invalid ID format: {item_id}"
        
        return True, None
    
    @classmethod
    def _is_path_traversal(cls, path: str) -> bool:
        """Check if path contains traversal attempts."""
        # Normalize path
        normalized = os.path.normpath(path)
        
        # Check for absolute paths
        if os.path.isabs(normalized):
            return True
        
        # Check for parent directory references
        if '..' in normalized:
            return True
        
        # Check for home directory reference
        if normalized.startswith('~'):
            return True
        
        # Check against pattern
        if cls.PATH_TRAVERSAL_PATTERN.search(path):
            return True
        
        # Check for dangerous directory names
        parts = Path(normalized).parts
        if any(part in cls.DANGEROUS_PATHS for part in parts):
            return True
        
        return False
    
    @classmethod
    def _is_dangerous_file(cls, filename: str) -> bool:
        """Check if file type is potentially dangerous."""
        dangerous_extensions = {
            '.exe', '.dll', '.so', '.dylib', '.app',  # Executables
            '.sh', '.bat', '.cmd', '.ps1', '.vbs',     # Scripts
            '.com', '.scr', '.msi', '.jar',            # More executables
            '.lnk', '.url', '.website',                # Shortcuts
            '.reg', '.inf',                             # System files
        }
        
        extension = Path(filename).suffix.lower()
        return extension in dangerous_extensions
    
    @classmethod
    def _is_safe_text(cls, text: str) -> bool:
        """Check if text contains only safe characters."""
        # Allow alphanumeric, spaces, and common punctuation
        safe_pattern = re.compile(r'^[\w\s\-.,!?\'\"()\[\]{}@#$%^&*+=/:;<>|`~]+$')
        return bool(safe_pattern.match(text))
    
    @classmethod
    def _is_valid_id(cls, item_id: str) -> bool:
        """Check if ID has valid format."""
        if not item_id:
            return False
        
        # Allow alphanumeric, hyphens, underscores (common ID formats)
        # Also allow UUIDs
        id_pattern = re.compile(r'^[\w\-]+$')
        return bool(id_pattern.match(str(item_id)))
    
    @classmethod
    def validate_job_id(cls, job_id: str) -> bool:
        """
        Validate job ID format (should be UUID).
        
        Args:
            job_id: Job ID to validate
            
        Returns:
            True if valid UUID format
        """
        uuid_pattern = re.compile(
            r'^[a-f0-9]{8}-[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{12}$',
            re.IGNORECASE
        )
        return bool(uuid_pattern.match(job_id))