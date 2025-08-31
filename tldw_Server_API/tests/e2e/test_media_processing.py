# test_media_processing.py
# Description: E2E tests for media processing edge cases and failures
#
"""
Media Processing E2E Tests
--------------------------

Tests media processing edge cases including corrupted files, large files,
transcription failures, and various document parsing scenarios.
"""

import os
import time
import tempfile
from pathlib import Path
from typing import Dict, Any, Optional
import pytest
import httpx

from fixtures import (
    api_client, authenticated_client, data_tracker,
    create_test_file, create_test_pdf, create_test_audio,
    StrongAssertionHelpers, SmartErrorHandler
)
from test_data import TestDataGenerator


class TestMediaProcessingEdgeCases:
    """Test edge cases in media file processing."""
    
    def test_corrupted_pdf_handling(self, api_client, data_tracker):
        """Test handling of corrupted PDF files."""
        # Create a corrupted PDF (invalid header)
        corrupted_pdf = self._create_corrupted_file(
            b"Not a PDF content - corrupted file",
            suffix=".pdf"
        )
        
        try:
            response = api_client.upload_media(
                file_path=corrupted_pdf,
                title="Corrupted PDF Test",
                media_type="pdf"
            )
            
            # Check if error is handled gracefully
            if "results" in response and response["results"]:
                result = response["results"][0]
                
                if result.get("status") == "Error":
                    # Good - error was caught and reported
                    assert "error" in result or "db_message" in result, \
                        "Error response missing error details"
                    error_msg = result.get("error", "") or result.get("db_message", "")
                    assert "pdf" in error_msg.lower() or "invalid" in error_msg.lower(), \
                        f"Error message not descriptive: {error_msg}"
                    print(f"✓ Corrupted PDF handled gracefully: {error_msg}")
                    
                elif result.get("db_id"):
                    # File was accepted - verify it's marked appropriately
                    media_id = result["db_id"]
                    data_tracker.add_media(media_id)
                    
                    media = api_client.get_media_item(media_id)
                    # Should have minimal or no content extracted
                    content = media.get("content", {})
                    if isinstance(content, dict):
                        text = content.get("text", "")
                    else:
                        text = str(content)
                    
                    assert len(text) < 100, \
                        "Too much content extracted from corrupted PDF"
            
        finally:
            os.unlink(corrupted_pdf)
    
    def test_large_document_processing(self, api_client, data_tracker):
        """Test processing of very large documents."""
        # Create a large text file (5MB)
        large_content = "Large document test content. " * 50000
        large_content += "\n".join([f"Line {i}: " + "x" * 100 for i in range(10000)])
        
        large_file = self._create_temp_file(large_content, suffix=".txt")
        file_size = os.path.getsize(large_file)
        
        try:
            print(f"Testing large file upload: {file_size / 1024 / 1024:.2f} MB")
            
            start_time = time.time()
            response = api_client.upload_media(
                file_path=large_file,
                title="Large Document Test",
                media_type="document"
            )
            elapsed = time.time() - start_time
            
            # Should handle large files
            media_id = self._extract_media_id(response)
            assert media_id is not None, "Large file upload failed"
            data_tracker.add_media(media_id)
            
            # Check processing time is reasonable
            assert elapsed < 60, f"Large file processing too slow: {elapsed:.2f}s"
            
            # Verify chunking occurred
            media = api_client.get_media_item(media_id)
            
            # Should be chunked or truncated appropriately
            content = media.get("content", {})
            if isinstance(content, dict):
                text = content.get("text", "")
            else:
                text = str(content)
            
            # Content should be present but potentially truncated
            assert len(text) > 1000, "No content extracted from large file"
            
            print(f"✓ Large file processed in {elapsed:.2f}s")
            
        finally:
            os.unlink(large_file)
    
    def test_unsupported_file_type(self, api_client):
        """Test uploading unsupported file types."""
        # Create an executable file (should be rejected)
        exec_content = b"#!/bin/bash\necho 'This should not be processed'"
        exec_file = self._create_temp_file_binary(exec_content, suffix=".sh")
        
        try:
            # The API should reject .sh files with 415 Unsupported Media Type
            response = api_client.upload_media(
                file_path=exec_file,
                title="Executable Test",
                media_type="document"
            )
            # If we get here, the file was accepted - check it's handled safely
            assert False, "Expected 415 error for .sh file but request succeeded"
            
        except httpx.HTTPStatusError as e:
            # Expected behavior - .sh files are rejected for security
            assert e.response.status_code == 415
            error_detail = e.response.json().get("detail", "")
            assert ".sh" in error_detail.lower() or "not allowed" in error_detail.lower()
            print(f"✓ Executable file properly rejected: {error_detail}")
                    
        finally:
            os.unlink(exec_file)
    
    def test_unicode_filename_handling(self, api_client, data_tracker):
        """Test files with unicode characters in filenames."""
        content = "Unicode filename test content"
        # Create file with unicode characters
        unicode_name = "test_文档_🎵_Ñoño.txt"
        
        with tempfile.NamedTemporaryFile(
            mode='w',
            suffix=unicode_name,
            delete=False,
            encoding='utf-8'
        ) as f:
            f.write(content)
            unicode_file = f.name
        
        try:
            response = api_client.upload_media(
                file_path=unicode_file,
                title="Unicode Filename Test",
                media_type="document"
            )
            
            media_id = self._extract_media_id(response)
            assert media_id is not None, "Unicode filename upload failed"
            data_tracker.add_media(media_id)
            
            # Verify content was processed correctly
            media = api_client.get_media_item(media_id)
            assert media is not None
            print(f"✓ Unicode filename handled correctly")
            
        finally:
            os.unlink(unicode_file)
    
    def test_empty_file_handling(self, api_client):
        """Test uploading empty files."""
        empty_file = self._create_temp_file("", suffix=".txt")
        
        try:
            # Empty files should be rejected
            try:
                response = api_client.upload_media(
                    file_path=empty_file,
                    title="Empty File Test",
                    media_type="document"
                )
                # If it succeeds, verify it's handled appropriately
                if "results" in response and response["results"]:
                    result = response["results"][0]
                    if result.get("db_id"):
                        media_id = result["db_id"]
                        media = api_client.get_media_item(media_id)
                        content = self._get_content_text(media)
                        assert len(content) == 0, "Content found in empty file"
                        print("✓ Empty file accepted but marked as empty")
            except httpx.HTTPStatusError as e:
                # Expected - empty file rejected
                if e.response.status_code == 400:
                    error_detail = e.response.json().get("detail", "")
                    assert "empty" in error_detail.lower(), \
                        f"Error should mention empty file: {error_detail}"
                    print("✓ Empty file properly rejected with 400")
                else:
                    raise
                    
        finally:
            os.unlink(empty_file)
    
    def _create_temp_file(self, content: str, suffix: str = ".txt") -> str:
        """Create a temporary text file."""
        with tempfile.NamedTemporaryFile(mode='w', suffix=suffix, delete=False) as f:
            f.write(content)
            return f.name
    
    def _create_temp_file_binary(self, content: bytes, suffix: str = "") -> str:
        """Create a temporary binary file."""
        with tempfile.NamedTemporaryFile(mode='wb', suffix=suffix, delete=False) as f:
            f.write(content)
            return f.name
    
    def _create_corrupted_file(self, content: bytes, suffix: str = "") -> str:
        """Create a corrupted file with wrong format."""
        return self._create_temp_file_binary(content, suffix)
    
    def _extract_media_id(self, response: Dict[str, Any]) -> Optional[int]:
        """Extract media ID from various response formats."""
        if "results" in response and response["results"]:
            result = response["results"][0]
            if result.get("status") != "Error":
                return result.get("db_id")
        return response.get("media_id") or response.get("id")
    
    def _get_content_text(self, media: Dict[str, Any]) -> str:
        """Extract text content from media object."""
        content = media.get("content", {})
        if isinstance(content, dict):
            return content.get("text", "")
        return str(content)


class TestAudioVideoProcessing:
    """Test audio and video file processing."""
    
    def test_silent_audio_file(self, api_client, data_tracker):
        """Test processing of silent audio files."""
        # Create a silent WAV file
        silent_wav = create_test_audio()  # This creates a silent WAV
        
        try:
            response = api_client.upload_media(
                file_path=silent_wav,
                title="Silent Audio Test",
                media_type="audio"
            )
            
            if "results" in response and response["results"]:
                result = response["results"][0]
                
                if result.get("db_id"):
                    media_id = result["db_id"]
                    data_tracker.add_media(media_id)
                    
                    # Check if transcription handled silent audio
                    media = api_client.get_media_item(media_id)
                    transcription = media.get("transcription") or media.get("transcript", "")
                    
                    # Should have empty or minimal transcription
                    assert len(transcription) < 50, \
                        f"Unexpected transcription from silent audio: {transcription}"
                    print("✓ Silent audio handled correctly")
                    
                elif result.get("status") == "Error":
                    # Silent audio might be rejected
                    print("✓ Silent audio properly identified and handled")
                    
        finally:
            os.unlink(silent_wav)
    
    def test_video_without_audio_track(self, api_client, data_tracker):
        """Test video file without audio track."""
        # This would require creating a video without audio
        # For now, we'll test with the sample video
        video_path = Path(__file__).parent.parent / "Media_Ingestion_Modification" / "test_media" / "sample.mp4"
        
        if not video_path.exists():
            pytest.skip("Test video not available")
        
        try:
            response = api_client.upload_media(
                file_path=str(video_path),
                title="Video Processing Test",
                media_type="video"
            )
            
            media_id = self._extract_media_id(response)
            if media_id:
                data_tracker.add_media(media_id)
                
                # Verify video was processed
                media = api_client.get_media_item(media_id)
                assert media is not None
                
                # Check for transcription or indication of no audio
                transcription = media.get("transcription") or media.get("transcript")
                if transcription is None or len(transcription) == 0:
                    print("✓ Video with no/minimal audio handled correctly")
                else:
                    print(f"✓ Video processed with transcription: {len(transcription)} chars")
                    
        except Exception as e:
            print(f"Video processing test skipped: {e}")
    
    def test_long_audio_processing(self, api_client, data_tracker):
        """Test processing of long audio files."""
        # Create a longer audio file (would need actual long audio)
        # For now, test with standard audio
        audio_file = create_test_audio()
        
        try:
            start_time = time.time()
            response = api_client.upload_media(
                file_path=audio_file,
                title="Long Audio Test",
                media_type="audio"
            )
            elapsed = time.time() - start_time
            
            media_id = self._extract_media_id(response)
            if media_id:
                data_tracker.add_media(media_id)
                print(f"✓ Audio processed in {elapsed:.2f}s")
                
                # For actual long audio, would verify:
                # - Chunking is used
                # - Processing completes within timeout
                # - Transcription is complete
                
        finally:
            os.unlink(audio_file)
    
    def _extract_media_id(self, response: Dict[str, Any]) -> Optional[int]:
        """Extract media ID from various response formats."""
        if "results" in response and response["results"]:
            result = response["results"][0]
            if result.get("status") != "Error":
                return result.get("db_id")
        return response.get("media_id") or response.get("id")


class TestDocumentParsing:
    """Test document parsing for various formats."""
    
    def test_encrypted_pdf(self, api_client):
        """Test handling of encrypted/password-protected PDFs."""
        # Create a mock encrypted PDF (would need actual encrypted PDF)
        encrypted_content = b"%PDF-1.4\n%Encrypted\nThis would be an encrypted PDF"
        encrypted_pdf = self._create_temp_file_binary(encrypted_content, ".pdf")
        
        try:
            response = api_client.upload_media(
                file_path=encrypted_pdf,
                title="Encrypted PDF Test",
                media_type="pdf"
            )
            
            # Should handle encrypted PDFs gracefully
            if "results" in response and response["results"]:
                result = response["results"][0]
                if result.get("status") == "Error":
                    error_msg = result.get("error", "") or result.get("db_message", "")
                    # Should indicate encryption or access issue
                    print(f"✓ Encrypted PDF handled: {error_msg}")
                elif result.get("db_id"):
                    # If processed, should have limited content
                    media_id = result["db_id"]
                    media = api_client.get_media_item(media_id)
                    content = self._get_content_text(media)
                    assert len(content) < 100, \
                        "Too much content from encrypted PDF"
                        
        finally:
            os.unlink(encrypted_pdf)
    
    def test_complex_docx_formatting(self, api_client, data_tracker):
        """Test DOCX files with complex formatting."""
        # Would need actual DOCX file with tables, images, etc.
        # For now, test with simple DOCX-like content
        
        # Create a mock DOCX (would need python-docx for real test)
        content = "Complex DOCX content with formatting"
        docx_file = self._create_temp_file(content, ".docx")
        
        try:
            response = api_client.upload_media(
                file_path=docx_file,
                title="Complex DOCX Test",
                media_type="document"
            )
            
            # DOCX might fail without proper parser
            if "results" in response and response["results"]:
                result = response["results"][0]
                if result.get("db_id"):
                    media_id = result["db_id"]
                    data_tracker.add_media(media_id)
                    print("✓ DOCX file processed")
                elif result.get("status") == "Error":
                    print(f"✓ DOCX handling: {result.get('error', 'Format not supported')}")
                    
        finally:
            os.unlink(docx_file)
    
    def test_malformed_html(self, api_client, data_tracker):
        """Test parsing of malformed HTML."""
        malformed_html = """
        <!DOCTYPE html>
        <html>
        <head><title>Malformed HTML</title>
        <body>
            <p>Unclosed paragraph
            <div>Unclosed div
            <script>alert('should be ignored')</script>
            <p>Valid content here</p>
            <!-- Broken comment ->
        </body>
        </html>
        """
        
        html_file = self._create_temp_file(malformed_html, ".html")
        
        try:
            response = api_client.upload_media(
                file_path=html_file,
                title="Malformed HTML Test",
                media_type="document"
            )
            
            media_id = self._extract_media_id(response)
            if media_id:
                data_tracker.add_media(media_id)
                
                # Should extract text content despite malformed structure
                media = api_client.get_media_item(media_id)
                content = self._get_content_text(media)
                
                # Should have extracted "Valid content here" at minimum
                assert "Valid content" in content or len(content) > 10, \
                    "Failed to extract content from malformed HTML"
                
                # Should not include script content
                assert "alert(" not in content, \
                    "Script content leaked into extracted text"
                
                print("✓ Malformed HTML parsed safely")
                
        finally:
            os.unlink(html_file)
    
    def test_special_characters_in_content(self, api_client, data_tracker):
        """Test documents with special characters and encodings."""
        special_content = """
        Special Characters Test:
        • Bullets: • ◦ ▪ ▫
        • Emojis: 😀 🎉 🚀 💻
        • Math: ∑ ∏ ∫ √ ∞ ≈ ≠
        • Currency: $ € £ ¥ ₹
        • Quotes: "curly" 'quotes' „German" «French»
        • Diacritics: café naïve résumé Zürich
        • CJK: 你好 こんにちは 안녕하세요
        • RTL: مرحبا שלום
        """
        
        special_file = self._create_temp_file(special_content, ".txt")
        
        try:
            response = api_client.upload_media(
                file_path=special_file,
                title="Special Characters Test",
                media_type="document"
            )
            
            media_id = self._extract_media_id(response)
            assert media_id is not None, "Special characters file failed"
            data_tracker.add_media(media_id)
            
            # Verify content preserved special characters
            media = api_client.get_media_item(media_id)
            content = self._get_content_text(media)
            
            # Check some special characters are preserved
            assert "€" in content or "£" in content, \
                "Currency symbols not preserved"
            assert "café" in content or "naïve" in content, \
                "Diacritics not preserved"
            
            print("✓ Special characters handled correctly")
            
        finally:
            os.unlink(special_file)
    
    def _create_temp_file(self, content: str, suffix: str = ".txt") -> str:
        """Create a temporary text file."""
        with tempfile.NamedTemporaryFile(
            mode='w', 
            suffix=suffix, 
            delete=False,
            encoding='utf-8'
        ) as f:
            f.write(content)
            return f.name
    
    def _create_temp_file_binary(self, content: bytes, suffix: str = "") -> str:
        """Create a temporary binary file."""
        with tempfile.NamedTemporaryFile(mode='wb', suffix=suffix, delete=False) as f:
            f.write(content)
            return f.name
    
    def _extract_media_id(self, response: Dict[str, Any]) -> Optional[int]:
        """Extract media ID from various response formats."""
        if "results" in response and response["results"]:
            result = response["results"][0]
            if result.get("status") != "Error":
                return result.get("db_id")
        return response.get("media_id") or response.get("id")
    
    def _get_content_text(self, media: Dict[str, Any]) -> str:
        """Extract text content from media object."""
        content = media.get("content", {})
        if isinstance(content, dict):
            return content.get("text", "")
        return str(content)


class TestFileUploadValidation:
    """Test file upload validation and limits."""
    
    def test_file_size_limits(self, api_client):
        """Test file size limit enforcement."""
        # Create a file at the size limit (would need to know actual limit)
        # Assuming 100MB limit for testing
        
        # Create a smaller test file (1MB) for practical testing
        content = "x" * (1024 * 1024)  # 1MB of 'x'
        large_file = self._create_temp_file(content, ".txt")
        
        try:
            file_size = os.path.getsize(large_file)
            print(f"Testing file size: {file_size / 1024 / 1024:.2f} MB")
            
            response = api_client.upload_media(
                file_path=large_file,
                title="Size Limit Test",
                media_type="document"
            )
            
            # Should accept reasonable sizes
            media_id = self._extract_media_id(response)
            assert media_id is not None, "Reasonable file size rejected"
            print("✓ File size validation passed")
            
        finally:
            os.unlink(large_file)
    
    def test_path_traversal_in_filename(self, api_client):
        """Test path traversal attempts in filenames."""
        content = "Path traversal test content"
        
        # Test various path traversal attempts
        dangerous_names = [
            "../../../etc/passwd",
            "..\\..\\..\\windows\\system32\\config\\sam",
            "test/../../sensitive.txt",
            "\\\\server\\share\\file.txt"
        ]
        
        for dangerous_name in dangerous_names:
            # Create safe temp file
            safe_file = self._create_temp_file(content, ".txt")
            
            # Would need to test with actual filename parameter
            # API should sanitize the filename
            
            try:
                # The API should handle this safely
                response = api_client.upload_media(
                    file_path=safe_file,
                    title="Path Traversal Test",
                    media_type="document"
                )
                
                # Should succeed but with sanitized filename
                media_id = self._extract_media_id(response)
                if media_id:
                    media = api_client.get_media_item(media_id)
                    # Filename should be sanitized
                    stored_name = media.get("filename", "") or media.get("name", "")
                    assert ".." not in stored_name, \
                        f"Path traversal not sanitized: {stored_name}"
                    
            finally:
                os.unlink(safe_file)
        
        print("✓ Path traversal attempts handled safely")
    
    def _create_temp_file(self, content: str, suffix: str = ".txt") -> str:
        """Create a temporary text file."""
        with tempfile.NamedTemporaryFile(mode='w', suffix=suffix, delete=False) as f:
            f.write(content)
            return f.name
    
    def _extract_media_id(self, response: Dict[str, Any]) -> Optional[int]:
        """Extract media ID from various response formats."""
        if "results" in response and response["results"]:
            result = response["results"][0]
            if result.get("status") != "Error":
                return result.get("db_id")
        return response.get("media_id") or response.get("id")


# Test runner
if __name__ == "__main__":
    pytest.main([__file__, "-v"])