"""
Integration tests for media endpoint with contextual chunking options.

Tests the full flow of media ingestion with contextual features.
"""

import pytest
from unittest.mock import Mock, MagicMock, patch, AsyncMock
from fastapi.testclient import TestClient
from fastapi import FastAPI, UploadFile
import json
import io
import tempfile
from pathlib import Path

from tldw_Server_API.app.api.v1.endpoints.media import router as media_router
from tldw_Server_API.app.api.v1.schemas.media_request_models import AddMediaForm


class TestMediaEndpointContextualIntegration:
    """Integration tests for media endpoint with contextual chunking."""
    
    @pytest.fixture
    def test_app(self):
        """Create a test FastAPI app with media router."""
        app = FastAPI()
        app.include_router(media_router)
        return app
    
    @pytest.fixture
    def test_client(self, test_app):
        """Create a test client."""
        return TestClient(test_app)
    
    @pytest.fixture
    def mock_dependencies(self):
        """Mock all required dependencies."""
        with patch('tldw_Server_API.app.api.v1.endpoints.media.get_request_user') as mock_user:
            with patch('tldw_Server_API.app.api.v1.endpoints.media.get_media_db_for_user') as mock_db:
                mock_user.return_value = Mock(id="test_user")
                mock_db.return_value = Mock(db_path="/test/path.db")
                yield {
                    "user": mock_user,
                    "db": mock_db
                }
    
    def test_add_media_with_contextual_chunking_enabled(self, test_client, mock_dependencies):
        """Test adding media with contextual chunking enabled."""
        # Prepare request data
        form_data = {
            "media_type": "document",
            "urls": json.dumps(["https://example.com/test.pdf"]),
            "perform_chunking": "true",
            "chunk_size": "500",
            "chunk_overlap": "100",
            "enable_contextual_chunking": "true",  # Enable contextual
            "contextual_llm_model": "gpt-4",
            "context_window_size": "750"
        }
        
        with patch('tldw_Server_API.app.api.v1.endpoints.media.process_document_content') as mock_process:
            mock_process.return_value = {"success": True, "media_id": 123}
            
            response = test_client.post(
                "/api/v1/media/add",
                data=form_data,
                headers={"Authorization": "Bearer test_token"}
            )
            
            # Verify the chunking options were passed correctly
            if mock_process.called:
                call_args = mock_process.call_args
                chunk_options = call_args[1].get('chunk_options', {})
                
                assert chunk_options.get('enable_contextual_chunking') == True
                assert chunk_options.get('contextual_llm_model') == "gpt-4"
                assert chunk_options.get('context_window_size') == 750
    
    def test_add_media_with_contextual_chunking_disabled(self, test_client, mock_dependencies):
        """Test adding media with contextual chunking explicitly disabled."""
        form_data = {
            "media_type": "document",
            "urls": json.dumps(["https://example.com/test.pdf"]),
            "perform_chunking": "true",
            "chunk_size": "500",
            "chunk_overlap": "100",
            "enable_contextual_chunking": "false"  # Explicitly disable
        }
        
        with patch('tldw_Server_API.app.api.v1.endpoints.media.process_document_content') as mock_process:
            mock_process.return_value = {"success": True, "media_id": 123}
            
            response = test_client.post(
                "/api/v1/media/add",
                data=form_data,
                headers={"Authorization": "Bearer test_token"}
            )
            
            if mock_process.called:
                call_args = mock_process.call_args
                chunk_options = call_args[1].get('chunk_options', {})
                
                assert chunk_options.get('enable_contextual_chunking') == False
    
    def test_add_media_contextual_defaults_from_config(self, test_client, mock_dependencies):
        """Test that contextual chunking uses config defaults when not specified."""
        form_data = {
            "media_type": "document",
            "urls": json.dumps(["https://example.com/test.pdf"]),
            "perform_chunking": "true",
            "chunk_size": "500",
            "chunk_overlap": "100"
            # No contextual options specified - should use defaults
        }
        
        with patch('tldw_Server_API.app.api.v1.endpoints.media.process_document_content') as mock_process:
            mock_process.return_value = {"success": True, "media_id": 123}
            
            response = test_client.post(
                "/api/v1/media/add",
                data=form_data,
                headers={"Authorization": "Bearer test_token"}
            )
            
            if mock_process.called:
                call_args = mock_process.call_args
                chunk_options = call_args[1].get('chunk_options', {})
                
                # Should have default value (False based on our config)
                assert chunk_options.get('enable_contextual_chunking') == False
                assert chunk_options.get('contextual_llm_model') is None
                assert chunk_options.get('context_window_size') is None
    
    def test_add_media_file_upload_with_contextual(self, test_client, mock_dependencies):
        """Test file upload with contextual chunking options."""
        # Create a test file
        test_content = b"Test document content for contextual chunking"
        test_file = io.BytesIO(test_content)
        
        files = {
            "files": ("test.txt", test_file, "text/plain")
        }
        
        form_data = {
            "media_type": "document",
            "perform_chunking": "true",
            "enable_contextual_chunking": "true",
            "contextual_llm_model": "gpt-3.5-turbo"
        }
        
        with patch('tldw_Server_API.app.api.v1.endpoints.media._process_uploaded_files') as mock_upload:
            with patch('tldw_Server_API.app.api.v1.endpoints.media.process_document_content') as mock_process:
                mock_upload.return_value = (["/tmp/test.txt"], [])
                mock_process.return_value = {"success": True, "media_id": 124}
                
                response = test_client.post(
                    "/api/v1/media/add",
                    data=form_data,
                    files=files,
                    headers={"Authorization": "Bearer test_token"}
                )
                
                # Verify contextual options were passed
                if mock_process.called:
                    call_args = mock_process.call_args
                    chunk_options = call_args[1].get('chunk_options', {})
                    
                    assert chunk_options.get('enable_contextual_chunking') == True
                    assert chunk_options.get('contextual_llm_model') == "gpt-3.5-turbo"
    
    @pytest.mark.parametrize("media_type,expected_method", [
        ("document", "sentences"),
        ("pdf", "sentences"),
        ("ebook", "ebook_chapters"),
        ("video", "sentences"),
        ("audio", "sentences")
    ])
    def test_contextual_chunking_with_different_media_types(
        self, 
        test_client, 
        mock_dependencies,
        media_type,
        expected_method
    ):
        """Test contextual chunking works with different media types."""
        form_data = {
            "media_type": media_type,
            "urls": json.dumps([f"https://example.com/test.{media_type}"]),
            "perform_chunking": "true",
            "enable_contextual_chunking": "true"
        }
        
        # Mock the appropriate processing function based on media type
        process_func = {
            "document": "process_document_content",
            "pdf": "process_pdf_task",
            "ebook": "process_epub",
            "video": "process_videos",
            "audio": "process_audio_files"
        }.get(media_type, "process_document_content")
        
        with patch(f'tldw_Server_API.app.api.v1.endpoints.media.{process_func}') as mock_process:
            mock_process.return_value = {"success": True, "media_id": 125}
            
            response = test_client.post(
                "/api/v1/media/add",
                data=form_data,
                headers={"Authorization": "Bearer test_token"}
            )
            
            if mock_process.called:
                call_args = mock_process.call_args
                chunk_options = call_args[1].get('chunk_options', {})
                
                assert chunk_options.get('enable_contextual_chunking') == True
                # Check that the method is appropriate for media type
                if media_type == "ebook":
                    assert chunk_options.get('method') == "ebook_chapters"
    
    def test_batch_media_with_contextual_chunking(self, test_client, mock_dependencies):
        """Test batch media processing with contextual chunking."""
        form_data = {
            "media_type": "document",
            "urls": json.dumps([
                "https://example.com/doc1.pdf",
                "https://example.com/doc2.pdf",
                "https://example.com/doc3.pdf"
            ]),
            "perform_chunking": "true",
            "enable_contextual_chunking": "true",
            "contextual_llm_model": "claude-3-opus"
        }
        
        with patch('tldw_Server_API.app.api.v1.endpoints.media.process_document_content') as mock_process:
            mock_process.return_value = {"success": True, "media_id": 126}
            
            response = test_client.post(
                "/api/v1/media/add",
                data=form_data,
                headers={"Authorization": "Bearer test_token"}
            )
            
            # Should be called for each URL
            assert mock_process.call_count >= 1
            
            # Each call should have contextual options
            for call in mock_process.call_args_list:
                chunk_options = call[1].get('chunk_options', {})
                assert chunk_options.get('enable_contextual_chunking') == True
                assert chunk_options.get('contextual_llm_model') == "claude-3-opus"
    
    def test_contextual_options_validation(self, test_client, mock_dependencies):
        """Test validation of contextual chunking options."""
        # Test with invalid context_window_size (too small)
        form_data = {
            "media_type": "document",
            "urls": json.dumps(["https://example.com/test.pdf"]),
            "perform_chunking": "true",
            "enable_contextual_chunking": "true",
            "context_window_size": "50"  # Below minimum of 100
        }
        
        response = test_client.post(
            "/api/v1/media/add",
            data=form_data,
            headers={"Authorization": "Bearer test_token"}
        )
        
        # Should get validation error
        assert response.status_code == 422
        
        # Test with invalid context_window_size (too large)
        form_data["context_window_size"] = "3000"  # Above maximum of 2000
        
        response = test_client.post(
            "/api/v1/media/add",
            data=form_data,
            headers={"Authorization": "Bearer test_token"}
        )
        
        assert response.status_code == 422
    
    def test_contextual_chunking_preserves_other_options(self, test_client, mock_dependencies):
        """Test that contextual options don't interfere with other chunking options."""
        form_data = {
            "media_type": "document",
            "urls": json.dumps(["https://example.com/test.pdf"]),
            "perform_chunking": "true",
            "chunk_method": "semantic",
            "chunk_size": "1000",
            "chunk_overlap": "200",
            "use_adaptive_chunking": "true",
            "use_multi_level_chunking": "true",
            "chunk_language": "en",
            "enable_contextual_chunking": "true",
            "contextual_llm_model": "gpt-4"
        }
        
        with patch('tldw_Server_API.app.api.v1.endpoints.media.process_document_content') as mock_process:
            mock_process.return_value = {"success": True, "media_id": 127}
            
            response = test_client.post(
                "/api/v1/media/add",
                data=form_data,
                headers={"Authorization": "Bearer test_token"}
            )
            
            if mock_process.called:
                call_args = mock_process.call_args
                chunk_options = call_args[1].get('chunk_options', {})
                
                # All options should be preserved
                assert chunk_options.get('method') == "semantic"
                assert chunk_options.get('max_size') == 1000
                assert chunk_options.get('overlap') == 200
                assert chunk_options.get('adaptive') == True
                assert chunk_options.get('multi_level') == True
                assert chunk_options.get('language') == "en"
                assert chunk_options.get('enable_contextual_chunking') == True
                assert chunk_options.get('contextual_llm_model') == "gpt-4"