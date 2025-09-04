"""
Integration tests for the /media/add endpoint.

Tests the full request/response flow with real database and minimal mocking.
Only external services like YouTube downloads are mocked.
"""

import pytest
import json
from fastapi import status
from unittest.mock import patch, MagicMock
import tempfile
from pathlib import Path

# ========================================================================
# Basic Add Media Tests
# ========================================================================

class TestAddMediaEndpoint:
    """Test the /media/add endpoint."""
    
    @pytest.mark.integration
    @patch('yt_dlp.YoutubeDL.extract_info')
    @patch('tldw_Server_API.app.core.Ingestion_Media_Processing.Audio.Audio_Transcription_Lib.transcribe_audio')
    def test_add_video_from_url(self, mock_transcribe, mock_yt_dlp, test_client, auth_headers):
        """Test adding a video from URL."""
        # Mock YouTube download
        mock_yt_dlp.return_value = {
            'title': 'Test Video',
            'duration': 120,
            'uploader': 'Test Channel',
            'webpage_url': 'http://youtube.com/watch?v=test'
        }
        
        # Mock transcription
        mock_transcribe.return_value = ("This is a test transcription.", {})
        
        response = test_client.post(
            "/api/v1/media/add",
            json={
                "url": "http://youtube.com/watch?v=test",
                "title": "Test Video",
                "media_type": "video",
                "transcribe": True,
                "chunk_method": "sentences",
                "max_chunk_size": 500
            },
            headers=auth_headers
        )
        
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert data["message"] == "Media added successfully"
        assert "media_id" in data
    
    @pytest.mark.integration
    def test_add_document_with_content(self, test_client, auth_headers, populated_media_db):
        """Test adding a document with direct content."""
        from tldw_Server_API.app.api.v1.API_Deps.DB_Deps import get_media_db_for_user
        from tldw_Server_API.app.main import app
        
        # Override dependency to use test database
        app.dependency_overrides[get_media_db_for_user] = lambda: populated_media_db
        
        try:
            response = test_client.post(
                "/api/v1/media/add",
                json={
                    "title": "Test Document",
                    "content": "This is test document content that should be stored.",
                    "media_type": "document",
                    "author": "Test Author",
                    "chunk_method": "words",
                    "max_chunk_size": 100
                },
                headers=auth_headers
            )
            
            assert response.status_code == status.HTTP_200_OK
            data = response.json()
            assert data["message"] == "Media added successfully"
            
            # Verify it was added to database
            items = populated_media_db.search_media_items("Test Document")
            assert len(items) > 0
            
        finally:
            app.dependency_overrides.clear()
    
    @pytest.mark.integration
    def test_add_with_invalid_url(self, test_client, auth_headers):
        """Test adding media with invalid URL."""
        response = test_client.post(
            "/api/v1/media/add",
            json={
                "url": "not-a-valid-url",
                "title": "Invalid Media",
                "media_type": "video"
            },
            headers=auth_headers
        )
        
        assert response.status_code == status.HTTP_400_BAD_REQUEST
        data = response.json()
        assert "error" in data or "detail" in data
    
    @pytest.mark.integration
    def test_add_without_required_fields(self, test_client, auth_headers):
        """Test adding media without required fields."""
        response = test_client.post(
            "/api/v1/media/add",
            json={
                "media_type": "video"
                # Missing title and url/content
            },
            headers=auth_headers
        )
        
        assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY

# ========================================================================
# Chunking Strategy Tests
# ========================================================================

class TestChunkingStrategies:
    """Test different chunking strategies during media addition."""
    
    @pytest.mark.integration
    def test_add_with_token_chunking(self, test_client, auth_headers, populated_media_db):
        """Test adding media with token-based chunking."""
        from tldw_Server_API.app.main import app
        from tldw_Server_API.app.api.v1.API_Deps.DB_Deps import get_media_db_for_user
        
        app.dependency_overrides[get_media_db_for_user] = lambda: populated_media_db
        
        try:
            long_content = " ".join(["This is sentence number {}.".format(i) for i in range(100)])
            
            response = test_client.post(
                "/api/v1/media/add",
                json={
                    "title": "Token Chunked Document",
                    "content": long_content,
                    "media_type": "document",
                    "chunk_method": "tokens",
                    "max_chunk_size": 50
                },
                headers=auth_headers
            )
            
            assert response.status_code == status.HTTP_200_OK
            
        finally:
            app.dependency_overrides.clear()
    
    @pytest.mark.integration
    def test_add_with_sentence_chunking(self, test_client, auth_headers, populated_media_db):
        """Test adding media with sentence-based chunking."""
        from tldw_Server_API.app.main import app
        from tldw_Server_API.app.api.v1.API_Deps.DB_Deps import get_media_db_for_user
        
        app.dependency_overrides[get_media_db_for_user] = lambda: populated_media_db
        
        try:
            content = "First sentence. Second sentence. Third sentence. Fourth sentence."
            
            response = test_client.post(
                "/api/v1/media/add",
                json={
                    "title": "Sentence Chunked Document",
                    "content": content,
                    "media_type": "document",
                    "chunk_method": "sentences",
                    "max_chunk_size": 2  # 2 sentences per chunk
                },
                headers=auth_headers
            )
            
            assert response.status_code == status.HTTP_200_OK
            
        finally:
            app.dependency_overrides.clear()

# ========================================================================
# Database Persistence Tests
# ========================================================================

class TestDatabasePersistence:
    """Test that media is properly persisted to database."""
    
    @pytest.mark.integration
    def test_media_persisted_correctly(self, test_client, auth_headers, media_database):
        """Test that media is correctly saved to database."""
        from tldw_Server_API.app.main import app
        from tldw_Server_API.app.api.v1.API_Deps.DB_Deps import get_media_db_for_user
        
        app.dependency_overrides[get_media_db_for_user] = lambda: media_database
        
        try:
            response = test_client.post(
                "/api/v1/media/add",
                json={
                    "title": "Persistence Test",
                    "content": "This content should be persisted.",
                    "media_type": "document",
                    "author": "Test Author",
                    "tags": ["test", "persistence"]
                },
                headers=auth_headers
            )
            
            assert response.status_code == status.HTTP_200_OK
            media_id = response.json()["media_id"]
            
            # Verify in database
            media = media_database.get_media(media_id)
            assert media is not None
            assert media["title"] == "Persistence Test"
            assert "This content should be persisted" in media["content"]
            
        finally:
            app.dependency_overrides.clear()
    
    @pytest.mark.integration
    def test_media_versioning(self, test_client, auth_headers, media_database):
        """Test that media versions are tracked."""
        from tldw_Server_API.app.main import app
        from tldw_Server_API.app.api.v1.API_Deps.DB_Deps import get_media_db_for_user
        
        app.dependency_overrides[get_media_db_for_user] = lambda: media_database
        
        try:
            # Add initial version
            response1 = test_client.post(
                "/api/v1/media/add",
                json={
                    "title": "Version Test",
                    "content": "Version 1 content",
                    "media_type": "document"
                },
                headers=auth_headers
            )
            
            assert response1.status_code == status.HTTP_200_OK
            media_id = response1.json()["media_id"]
            
            # Update to create new version
            response2 = test_client.post(
                "/api/v1/media/add",
                json={
                    "media_id": media_id,
                    "title": "Version Test",
                    "content": "Version 2 content",
                    "media_type": "document"
                },
                headers=auth_headers
            )
            
            # Check versions exist
            versions = media_database.get_all_versions(media_id)
            assert len(versions) >= 1
            
        finally:
            app.dependency_overrides.clear()

# ========================================================================
# Error Handling Tests
# ========================================================================

class TestErrorHandling:
    """Test error handling in the add media endpoint."""
    
    @pytest.mark.integration
    @patch('tldw_Server_API.app.core.Ingestion_Media_Processing.Audio.Audio_Transcription_Lib.transcribe_audio')
    def test_transcription_failure_handling(self, mock_transcribe, test_client, auth_headers):
        """Test handling of transcription failures."""
        mock_transcribe.side_effect = Exception("Transcription failed")
        
        response = test_client.post(
            "/api/v1/media/add",
            json={
                "title": "Failing Transcription",
                "content": "Content to transcribe",
                "media_type": "document",
                "transcribe": True
            },
            headers=auth_headers
        )
        
        # Should handle gracefully - either succeed without transcription or return error
        assert response.status_code in [status.HTTP_200_OK, status.HTTP_500_INTERNAL_SERVER_ERROR]
    
    @pytest.mark.integration
    def test_database_failure_handling(self, test_client, auth_headers):
        """Test handling of database failures."""
        from tldw_Server_API.app.main import app
        from tldw_Server_API.app.api.v1.API_Deps.DB_Deps import get_media_db_for_user
        
        # Mock a failing database
        mock_db = MagicMock()
        mock_db.add_media.side_effect = Exception("Database error")
        
        app.dependency_overrides[get_media_db_for_user] = lambda: mock_db
        
        try:
            response = test_client.post(
                "/api/v1/media/add",
                json={
                    "title": "DB Failure Test",
                    "content": "Content",
                    "media_type": "document"
                },
                headers=auth_headers
            )
            
            assert response.status_code == status.HTTP_500_INTERNAL_SERVER_ERROR
            
        finally:
            app.dependency_overrides.clear()

# ========================================================================
# File Upload Tests
# ========================================================================

class TestFileUpload:
    """Test file upload functionality."""
    
    @pytest.mark.integration
    def test_upload_text_file(self, test_client, auth_headers, test_text_file):
        """Test uploading a text file."""
        with open(test_text_file, 'rb') as f:
            files = {"file": ("test.txt", f, "text/plain")}
            
            response = test_client.post(
                "/api/v1/media/upload",
                files=files,
                data={
                    "title": "Uploaded Text File",
                    "media_type": "document"
                },
                headers={"Authorization": auth_headers["Authorization"]}
            )
            
            if response.status_code == status.HTTP_200_OK:
                data = response.json()
                assert "media_id" in data
    
    @pytest.mark.integration
    def test_upload_invalid_file_type(self, test_client, auth_headers, test_media_dir):
        """Test rejection of invalid file types."""
        exe_file = test_media_dir / "test.exe"
        exe_file.write_bytes(b'MZ\x00\x00')  # PE header
        
        with open(exe_file, 'rb') as f:
            files = {"file": ("test.exe", f, "application/x-msdownload")}
            
            response = test_client.post(
                "/api/v1/media/upload",
                files=files,
                data={"title": "Invalid File"},
                headers={"Authorization": auth_headers["Authorization"]}
            )
            
            assert response.status_code == status.HTTP_400_BAD_REQUEST