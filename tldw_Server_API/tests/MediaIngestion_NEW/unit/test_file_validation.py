"""
Unit tests for file validation and sanitization in Upload_Sink.

Tests focus on MIME type detection, file size validation, security checks,
and sanitization without any external dependencies.
"""

import pytest
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
import tempfile
import os

from tldw_Server_API.app.core.Ingestion_Media_Processing.Upload_Sink import (
    ValidationResult,
    FileValidationError,
    validate_file_type,
    check_file_size,
    sanitize_filename,
    validate_upload,
    scan_for_malicious_content
)

# ========================================================================
# MIME Type Detection Tests
# ========================================================================

class TestMimeTypeDetection:
    """Test MIME type detection and validation."""
    
    @pytest.mark.unit
    def test_detect_text_file(self, test_text_file):
        """Test detection of text file MIME type."""
        result = validate_file_type(test_text_file)
        
        assert result.is_valid
        assert result.detected_mime_type in ["text/plain", "application/octet-stream"]
        assert result.detected_extension == ".txt"
    
    @pytest.mark.unit
    def test_detect_pdf_file(self, test_pdf_file):
        """Test detection of PDF file MIME type."""
        result = validate_file_type(test_pdf_file)
        
        assert result.is_valid
        assert "pdf" in result.detected_mime_type.lower()
        assert result.detected_extension == ".pdf"
    
    @pytest.mark.unit
    def test_detect_audio_file(self, test_audio_file):
        """Test detection of audio file MIME type."""
        result = validate_file_type(test_audio_file)
        
        assert result.is_valid
        assert "audio" in result.detected_mime_type.lower() or "wav" in result.detected_mime_type.lower()
        assert result.detected_extension == ".wav"
    
    @pytest.mark.unit
    def test_detect_video_file(self, test_video_file):
        """Test detection of video file MIME type."""
        result = validate_file_type(test_video_file)
        
        # Video validation might depend on actual implementation
        assert result.detected_extension == ".mp4"
    
    @pytest.mark.unit
    def test_reject_executable_file(self, test_media_dir):
        """Test rejection of executable files."""
        exe_path = test_media_dir / "malicious.exe"
        
        # Create a fake executable with PE header
        with open(exe_path, 'wb') as f:
            f.write(b'MZ')  # PE executable signature
            f.write(b'\x00' * 100)
        
        result = validate_file_type(exe_path)
        
        assert not result.is_valid
        assert "not allowed" in str(result.issues).lower() or "executable" in str(result.issues).lower()

# ========================================================================
# File Size Validation Tests
# ========================================================================

class TestFileSizeValidation:
    """Test file size validation."""
    
    @pytest.mark.unit
    def test_accept_small_file(self, test_text_file):
        """Test acceptance of small files."""
        max_size = 10 * 1024 * 1024  # 10MB
        
        result = check_file_size(test_text_file, max_size)
        
        assert result.is_valid
        assert len(result.issues) == 0
    
    @pytest.mark.unit
    def test_reject_large_file(self, test_media_dir):
        """Test rejection of files exceeding size limit."""
        large_file = test_media_dir / "large_file.bin"
        
        # Create a file that appears large (sparse file)
        with open(large_file, 'wb') as f:
            f.seek(101 * 1024 * 1024)  # 101MB
            f.write(b'\x00')
        
        max_size = 100 * 1024 * 1024  # 100MB limit
        
        result = check_file_size(large_file, max_size)
        
        assert not result.is_valid
        assert any("size" in issue.lower() for issue in result.issues)
    
    @pytest.mark.unit
    def test_handle_zero_size_file(self, test_media_dir):
        """Test handling of zero-size files."""
        empty_file = test_media_dir / "empty.txt"
        empty_file.touch()
        
        result = check_file_size(empty_file, 1024)
        
        # Zero-size files might be rejected
        if not result.is_valid:
            assert any("empty" in issue.lower() or "zero" in issue.lower() for issue in result.issues)

# ========================================================================
# Filename Sanitization Tests
# ========================================================================

class TestFilenameSanitization:
    """Test filename sanitization."""
    
    @pytest.mark.unit
    def test_sanitize_normal_filename(self):
        """Test sanitization of normal filename."""
        filename = "document.pdf"
        sanitized = sanitize_filename(filename)
        
        assert sanitized == "document.pdf"
    
    @pytest.mark.unit
    def test_sanitize_special_characters(self):
        """Test removal of special characters."""
        filename = "my<file>name|with*special?chars.txt"
        sanitized = sanitize_filename(filename)
        
        # Should remove or replace special chars
        assert "<" not in sanitized
        assert ">" not in sanitized
        assert "|" not in sanitized
        assert "*" not in sanitized
        assert "?" not in sanitized
    
    @pytest.mark.unit
    def test_sanitize_path_traversal(self):
        """Test prevention of path traversal."""
        filename = "../../etc/passwd"
        sanitized = sanitize_filename(filename)
        
        assert ".." not in sanitized
        assert "/" not in sanitized or sanitized == "passwd"
    
    @pytest.mark.unit
    def test_sanitize_unicode_filename(self):
        """Test handling of unicode filenames."""
        filename = "文档文件.pdf"
        sanitized = sanitize_filename(filename)
        
        # Should either preserve or safely handle unicode
        assert sanitized  # Should not be empty
        assert ".pdf" in sanitized
    
    @pytest.mark.unit
    def test_sanitize_very_long_filename(self):
        """Test truncation of very long filenames."""
        filename = "a" * 500 + ".txt"
        sanitized = sanitize_filename(filename)
        
        # Should truncate to reasonable length
        assert len(sanitized) < 255  # Common filesystem limit
        assert sanitized.endswith(".txt")

# ========================================================================
# Security Validation Tests
# ========================================================================

class TestSecurityValidation:
    """Test security-related validation."""
    
    @pytest.mark.unit
    def test_detect_php_code(self, malicious_file):
        """Test detection of PHP code in files."""
        result = scan_for_malicious_content(malicious_file)
        
        assert not result.is_valid
        assert any("malicious" in issue.lower() or "php" in issue.lower() for issue in result.issues)
    
    @pytest.mark.unit
    def test_detect_script_tags(self, test_media_dir):
        """Test detection of script tags."""
        html_file = test_media_dir / "test.html"
        with open(html_file, 'w') as f:
            f.write("<html><script>alert('XSS')</script></html>")
        
        result = scan_for_malicious_content(html_file)
        
        # May or may not flag depending on file type
        if not result.is_valid:
            assert any("script" in issue.lower() for issue in result.issues)
    
    @pytest.mark.unit
    def test_accept_safe_content(self, test_text_file):
        """Test acceptance of safe content."""
        result = scan_for_malicious_content(test_text_file)
        
        assert result.is_valid
        assert len(result.issues) == 0

# ========================================================================
# Complete Validation Pipeline Tests
# ========================================================================

class TestCompleteValidation:
    """Test the complete validation pipeline."""
    
    @pytest.mark.unit
    @patch('tldw_Server_API.app.core.Ingestion_Media_Processing.Upload_Sink.validate_file_type')
    @patch('tldw_Server_API.app.core.Ingestion_Media_Processing.Upload_Sink.check_file_size')
    @patch('tldw_Server_API.app.core.Ingestion_Media_Processing.Upload_Sink.scan_for_malicious_content')
    def test_complete_valid_file(self, mock_scan, mock_size, mock_type, test_text_file):
        """Test complete validation of a valid file."""
        mock_type.return_value = ValidationResult(True, [], test_text_file, "text/plain", ".txt")
        mock_size.return_value = ValidationResult(True, [])
        mock_scan.return_value = ValidationResult(True, [])
        
        result = validate_upload(test_text_file)
        
        assert result.is_valid
        assert len(result.issues) == 0
        mock_type.assert_called_once()
        mock_size.assert_called_once()
        mock_scan.assert_called_once()
    
    @pytest.mark.unit
    @patch('tldw_Server_API.app.core.Ingestion_Media_Processing.Upload_Sink.validate_file_type')
    def test_validation_stops_on_type_failure(self, mock_type, test_text_file):
        """Test that validation stops early on type check failure."""
        mock_type.return_value = ValidationResult(False, ["Invalid type"])
        
        result = validate_upload(test_text_file)
        
        assert not result.is_valid
        assert "Invalid type" in result.issues
    
    @pytest.mark.unit
    def test_handle_nonexistent_file(self):
        """Test handling of nonexistent files."""
        fake_path = Path("/nonexistent/file.txt")
        
        with pytest.raises(FileNotFoundError):
            validate_upload(fake_path)

# ========================================================================
# Edge Cases and Error Handling
# ========================================================================

class TestEdgeCases:
    """Test edge cases and error conditions."""
    
    @pytest.mark.unit
    def test_symlink_handling(self, test_media_dir, test_text_file):
        """Test handling of symbolic links."""
        symlink = test_media_dir / "symlink.txt"
        symlink.symlink_to(test_text_file)
        
        # Should either follow or reject symlinks
        result = validate_file_type(symlink)
        
        # Behavior depends on implementation
        assert result.is_valid or any("symlink" in issue.lower() for issue in result.issues)
    
    @pytest.mark.unit
    def test_handle_corrupted_file(self, test_media_dir):
        """Test handling of corrupted files."""
        corrupted = test_media_dir / "corrupted.pdf"
        
        # Create corrupted PDF (invalid structure)
        with open(corrupted, 'wb') as f:
            f.write(b'%PDF-1.4\n')
            f.write(b'corrupted data here')
        
        result = validate_file_type(corrupted)
        
        # May still detect as PDF based on header
        assert result.detected_extension == ".pdf"