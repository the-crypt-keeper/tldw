# image_validation.py
# Description: Image validation utilities for secure image processing
#
# Imports
import base64
import re
from typing import Optional, Tuple
from loguru import logger

#######################################################################################################################
#
# Constants:

# Maximum allowed size for base64-encoded image data (3 MB)
MAX_BASE64_BYTES = 3 * 1024 * 1024

# Maximum size of base64 string (4/3 of binary size due to encoding)
MAX_BASE64_STRING_LENGTH = int(MAX_BASE64_BYTES * 4 / 3) + 100  # Adding buffer for padding

# Allowed MIME types for images
ALLOWED_IMAGE_MIME_TYPES = {"image/png", "image/jpeg", "image/webp"}

# Regex pattern for data URI validation
DATA_URI_PATTERN = re.compile(r'^data:([^;]+);base64,(.+)$')

#######################################################################################################################
#
# Functions:

def validate_mime_type(mime_type: str) -> bool:
    """
    Validate if the MIME type is allowed.
    
    Args:
        mime_type: MIME type to validate
        
    Returns:
        True if MIME type is allowed, False otherwise
    """
    return mime_type.lower() in ALLOWED_IMAGE_MIME_TYPES


def estimate_decoded_size(base64_string: str) -> int:
    """
    Estimate the decoded size of a base64 string without actually decoding it.
    
    Args:
        base64_string: Base64-encoded string
        
    Returns:
        Estimated size in bytes of the decoded data
    """
    # Remove padding characters
    base64_string = base64_string.rstrip('=')
    # Each base64 character represents 6 bits, so 4 characters = 3 bytes
    return int(len(base64_string) * 3 / 4)


def validate_data_uri(data_uri: str) -> Tuple[bool, Optional[str], Optional[str]]:
    """
    Validate a data URI and extract its components safely.
    
    Args:
        data_uri: Data URI string to validate
        
    Returns:
        Tuple of (is_valid, mime_type, base64_data)
    """
    # Check if it starts with 'data:'
    if not data_uri.startswith('data:'):
        return False, None, None
    
    # Parse the data URI
    match = DATA_URI_PATTERN.match(data_uri)
    if not match:
        logger.warning("Invalid data URI format")
        return False, None, None
    
    mime_type = match.group(1)
    base64_data = match.group(2)
    
    # Validate MIME type
    if not validate_mime_type(mime_type):
        logger.warning(f"Disallowed MIME type: {mime_type}")
        return False, mime_type, None
    
    # Check base64 string length BEFORE decoding
    if len(base64_data) > MAX_BASE64_STRING_LENGTH:
        logger.warning(f"Base64 string too long: {len(base64_data)} > {MAX_BASE64_STRING_LENGTH}")
        return False, mime_type, None
    
    # Estimate decoded size
    estimated_size = estimate_decoded_size(base64_data)
    if estimated_size > MAX_BASE64_BYTES:
        logger.warning(f"Estimated decoded size too large: {estimated_size} > {MAX_BASE64_BYTES}")
        return False, mime_type, None
    
    return True, mime_type, base64_data


def safe_decode_base64_image(base64_data: str, mime_type: str) -> Optional[bytes]:
    """
    Safely decode a base64-encoded image with size validation.
    
    Args:
        base64_data: Base64-encoded image data
        mime_type: MIME type of the image
        
    Returns:
        Decoded bytes if valid, None otherwise
    """
    try:
        # Final validation of MIME type
        if not validate_mime_type(mime_type):
            logger.warning(f"Invalid MIME type for decoding: {mime_type}")
            return None
        
        # Decode the base64 data
        decoded_data = base64.b64decode(base64_data, validate=True)
        
        # Final size check on decoded data
        if len(decoded_data) > MAX_BASE64_BYTES:
            logger.warning(f"Decoded image too large: {len(decoded_data)} > {MAX_BASE64_BYTES}")
            return None
        
        # TODO: Add optional virus/malware scanning here
        # if ENABLE_VIRUS_SCAN:
        #     if not scan_for_malware(decoded_data, mime_type):
        #         logger.warning("Image failed malware scan")
        #         return None
        
        return decoded_data
        
    except base64.binascii.Error as e:
        logger.warning(f"Invalid base64 data: {e}")
        return None
    except Exception as e:
        logger.error(f"Error decoding base64 image: {e}")
        return None


def validate_image_url(url: str) -> Tuple[bool, Optional[str], Optional[bytes]]:
    """
    Validate and process an image URL (data URI or HTTP URL).
    
    Args:
        url: Image URL to validate
        
    Returns:
        Tuple of (is_valid, mime_type, decoded_bytes)
    """
    if url.startswith('data:'):
        # Handle data URI
        is_valid, mime_type, base64_data = validate_data_uri(url)
        if not is_valid:
            return False, mime_type, None
        
        decoded_bytes = safe_decode_base64_image(base64_data, mime_type)
        if decoded_bytes is None:
            return False, mime_type, None
        
        return True, mime_type, decoded_bytes
    else:
        # For now, we don't support external URLs for security reasons
        logger.warning(f"External image URLs not supported: {url[:100]}")
        return False, None, None


#
# End of image_validation.py
#######################################################################################################################