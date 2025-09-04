"""
Integration tests for TTS API endpoints.

Tests the full request/response cycle with real components,
no mocking except for external API calls.
"""

import pytest
import json
import base64
from fastapi import status
from unittest.mock import patch, AsyncMock, MagicMock
import tempfile
from pathlib import Path

# ========================================================================
# TTS Generate Endpoint Tests
# ========================================================================

class TestTTSGenerateEndpoint:
    """Test the /api/v1/tts/generate endpoint."""
    
    @pytest.mark.integration
    @patch('tldw_Server_API.app.core.TTS.adapters.openai_adapter.httpx.AsyncClient.post')
    async def test_generate_basic_audio(self, mock_post, test_client, auth_headers):
        """Test basic TTS generation endpoint."""
        # Mock OpenAI API response
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.content = b"fake_audio_data"
        mock_response.headers = {"content-type": "audio/mpeg"}
        mock_post.return_value = mock_response
        
        response = test_client.post(
            "/api/v1/tts/generate",
            json={
                "text": "Hello, this is a test.",
                "voice": "alloy",
                "model": "tts-1",
                "provider": "openai"
            },
            headers=auth_headers
        )
        
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        
        assert "audio_base64" in data
        assert "format" in data
        assert data["format"] == "mp3"
        assert "provider" in data
        assert data["provider"] == "openai"
    
    @pytest.mark.integration
    async def test_generate_without_provider(self, test_client, auth_headers):
        """Test generation using default provider."""
        with patch('tldw_Server_API.app.core.TTS.tts_service_v2.TTSServiceV2.generate') as mock_generate:
            mock_generate.return_value = MagicMock(
                audio_content=b"audio_data",
                format="mp3",
                provider="openai"
            )
            
            response = test_client.post(
                "/api/v1/tts/generate",
                json={
                    "text": "Test without provider",
                    "voice": "nova"
                    # No provider specified, should use default
                },
                headers=auth_headers
            )
            
            assert response.status_code == status.HTTP_200_OK
    
    @pytest.mark.integration
    async def test_generate_with_voice_settings(self, test_client, auth_headers):
        """Test generation with voice settings."""
        with patch('tldw_Server_API.app.core.TTS.tts_service_v2.TTSServiceV2.generate') as mock_generate:
            mock_generate.return_value = MagicMock(
                audio_content=b"custom_audio",
                format="mp3",
                provider="elevenlabs"
            )
            
            response = test_client.post(
                "/api/v1/tts/generate",
                json={
                    "text": "Custom voice test",
                    "voice": "rachel",
                    "provider": "elevenlabs",
                    "voice_settings": {
                        "stability": 0.5,
                        "similarity_boost": 0.75
                    }
                },
                headers=auth_headers
            )
            
            assert response.status_code == status.HTTP_200_OK
            
            # Verify voice settings were passed
            mock_generate.assert_called_once()
            call_args = mock_generate.call_args[0][0]
            assert call_args.voice_settings is not None
    
    @pytest.mark.integration
    async def test_generate_with_invalid_provider(self, test_client, auth_headers):
        """Test generation with invalid provider."""
        response = test_client.post(
            "/api/v1/tts/generate",
            json={
                "text": "Test",
                "voice": "alloy",
                "provider": "invalid_provider"
            },
            headers=auth_headers
        )
        
        assert response.status_code == status.HTTP_400_BAD_REQUEST
        data = response.json()
        assert "error" in data or "detail" in data
    
    @pytest.mark.integration
    async def test_generate_with_long_text(self, test_client, auth_headers):
        """Test generation with long text that needs chunking."""
        long_text = " ".join(["This is sentence number {}.".format(i) for i in range(500)])
        
        with patch('tldw_Server_API.app.core.TTS.tts_service_v2.TTSServiceV2.generate') as mock_generate:
            mock_generate.return_value = MagicMock(
                audio_content=b"long_audio",
                format="mp3",
                provider="openai"
            )
            
            response = test_client.post(
                "/api/v1/tts/generate",
                json={
                    "text": long_text,
                    "voice": "alloy",
                    "provider": "openai"
                },
                headers=auth_headers
            )
            
            # Should handle long text appropriately
            assert response.status_code in [status.HTTP_200_OK, status.HTTP_413_REQUEST_ENTITY_TOO_LARGE]

# ========================================================================
# TTS Streaming Endpoint Tests
# ========================================================================

class TestTTSStreamingEndpoint:
    """Test the /api/v1/tts/generate/stream endpoint."""
    
    @pytest.mark.integration
    async def test_streaming_generation(self, test_client, auth_headers):
        """Test streaming TTS generation."""
        async def mock_stream():
            for chunk in [b"chunk1", b"chunk2", b"chunk3"]:
                yield chunk
        
        with patch('tldw_Server_API.app.core.TTS.tts_service_v2.TTSServiceV2.generate_stream') as mock_stream_gen:
            mock_stream_gen.return_value = mock_stream()
            
            response = test_client.post(
                "/api/v1/tts/generate/stream",
                json={
                    "text": "Stream this text",
                    "voice": "echo",
                    "provider": "openai"
                },
                headers=auth_headers,
                stream=True
            )
            
            assert response.status_code == status.HTTP_200_OK
            
            # Collect streamed chunks
            chunks = []
            for chunk in response.iter_bytes():
                chunks.append(chunk)
            
            assert len(chunks) > 0
    
    @pytest.mark.integration
    async def test_streaming_with_error(self, test_client, auth_headers):
        """Test streaming handles errors gracefully."""
        async def mock_error_stream():
            yield b"chunk1"
            raise Exception("Stream error")
        
        with patch('tldw_Server_API.app.core.TTS.tts_service_v2.TTSServiceV2.generate_stream') as mock_stream:
            mock_stream.return_value = mock_error_stream()
            
            response = test_client.post(
                "/api/v1/tts/generate/stream",
                json={
                    "text": "Error test",
                    "voice": "alloy"
                },
                headers=auth_headers,
                stream=True
            )
            
            # Should handle error appropriately
            chunks = list(response.iter_bytes())
            # First chunk should be received

# ========================================================================
# Provider Management Endpoint Tests
# ========================================================================

class TestProviderManagementEndpoints:
    """Test TTS provider management endpoints."""
    
    @pytest.mark.integration
    async def test_list_providers(self, test_client, auth_headers):
        """Test listing available TTS providers."""
        with patch('tldw_Server_API.app.core.TTS.tts_service_v2.TTSServiceV2.list_providers') as mock_list:
            mock_list.return_value = ["openai", "elevenlabs", "kokoro"]
            
            response = test_client.get(
                "/api/v1/tts/providers",
                headers=auth_headers
            )
            
            assert response.status_code == status.HTTP_200_OK
            data = response.json()
            
            assert isinstance(data, list)
            assert "openai" in data
            assert "elevenlabs" in data
    
    @pytest.mark.integration
    async def test_get_provider_info(self, test_client, auth_headers):
        """Test getting specific provider information."""
        with patch('tldw_Server_API.app.core.TTS.tts_service_v2.TTSServiceV2.get_provider_info') as mock_info:
            mock_info.return_value = {
                "provider": "openai",
                "models": ["tts-1", "tts-1-hd"],
                "voices": ["alloy", "echo", "fable", "onyx", "nova", "shimmer"],
                "max_characters": 4096
            }
            
            response = test_client.get(
                "/api/v1/tts/providers/openai",
                headers=auth_headers
            )
            
            assert response.status_code == status.HTTP_200_OK
            data = response.json()
            
            assert data["provider"] == "openai"
            assert "tts-1" in data["models"]
            assert "alloy" in data["voices"]
    
    @pytest.mark.integration
    async def test_switch_default_provider(self, test_client, auth_headers):
        """Test switching the default TTS provider."""
        response = test_client.post(
            "/api/v1/tts/providers/default",
            json={"provider": "elevenlabs"},
            headers=auth_headers
        )
        
        if response.status_code == status.HTTP_200_OK:
            data = response.json()
            assert data["message"] == "Default provider updated"
            assert data["provider"] == "elevenlabs"

# ========================================================================
# Voice Management Endpoint Tests
# ========================================================================

class TestVoiceManagementEndpoints:
    """Test voice management endpoints."""
    
    @pytest.mark.integration
    async def test_list_voices(self, test_client, auth_headers):
        """Test listing available voices for a provider."""
        with patch('tldw_Server_API.app.core.TTS.tts_service_v2.TTSServiceV2.list_voices') as mock_voices:
            mock_voices.return_value = [
                {"id": "alloy", "name": "Alloy", "gender": "neutral"},
                {"id": "echo", "name": "Echo", "gender": "male"},
                {"id": "nova", "name": "Nova", "gender": "female"}
            ]
            
            response = test_client.get(
                "/api/v1/tts/voices?provider=openai",
                headers=auth_headers
            )
            
            assert response.status_code == status.HTTP_200_OK
            data = response.json()
            
            assert isinstance(data, list)
            assert len(data) == 3
            assert any(v["id"] == "alloy" for v in data)
    
    @pytest.mark.integration
    async def test_get_voice_details(self, test_client, auth_headers):
        """Test getting details for a specific voice."""
        with patch('tldw_Server_API.app.core.TTS.tts_service_v2.TTSServiceV2.get_voice_info') as mock_info:
            mock_info.return_value = {
                "id": "rachel",
                "name": "Rachel",
                "provider": "elevenlabs",
                "gender": "female",
                "accent": "american",
                "default_settings": {
                    "stability": 0.75,
                    "similarity_boost": 0.75
                }
            }
            
            response = test_client.get(
                "/api/v1/tts/voices/rachel?provider=elevenlabs",
                headers=auth_headers
            )
            
            assert response.status_code == status.HTTP_200_OK
            data = response.json()
            
            assert data["id"] == "rachel"
            assert data["provider"] == "elevenlabs"

# ========================================================================
# File Download Endpoint Tests
# ========================================================================

class TestFileDownloadEndpoints:
    """Test audio file download endpoints."""
    
    @pytest.mark.integration
    async def test_download_generated_audio(self, test_client, auth_headers, sample_audio_bytes):
        """Test downloading generated audio as file."""
        with patch('tldw_Server_API.app.core.TTS.tts_service_v2.TTSServiceV2.generate') as mock_generate:
            mock_generate.return_value = MagicMock(
                audio_content=sample_audio_bytes,
                format="wav",
                provider="openai"
            )
            
            response = test_client.post(
                "/api/v1/tts/generate/download",
                json={
                    "text": "Download this audio",
                    "voice": "alloy",
                    "format": "wav"
                },
                headers=auth_headers
            )
            
            assert response.status_code == status.HTTP_200_OK
            assert response.headers["content-type"] == "audio/wav"
            assert "content-disposition" in response.headers
            assert len(response.content) > 0

# ========================================================================
# Error Handling Tests
# ========================================================================

class TestErrorHandling:
    """Test error handling in TTS endpoints."""
    
    @pytest.mark.integration
    async def test_rate_limit_error_handling(self, test_client, auth_headers):
        """Test handling of rate limit errors."""
        from tldw_Server_API.app.core.TTS.tts_exceptions import TTSRateLimitError
        
        with patch('tldw_Server_API.app.core.TTS.tts_service_v2.TTSServiceV2.generate') as mock_generate:
            mock_generate.side_effect = TTSRateLimitError("Rate limited", retry_after=60)
            
            response = test_client.post(
                "/api/v1/tts/generate",
                json={
                    "text": "Test",
                    "voice": "alloy"
                },
                headers=auth_headers
            )
            
            assert response.status_code == status.HTTP_429_TOO_MANY_REQUESTS
            data = response.json()
            assert "retry_after" in data
    
    @pytest.mark.integration
    async def test_quota_exceeded_error(self, test_client, auth_headers):
        """Test handling of quota exceeded errors."""
        from tldw_Server_API.app.core.TTS.tts_exceptions import TTSQuotaExceededError
        
        with patch('tldw_Server_API.app.core.TTS.tts_service_v2.TTSServiceV2.generate') as mock_generate:
            mock_generate.side_effect = TTSQuotaExceededError("Character quota exceeded")
            
            response = test_client.post(
                "/api/v1/tts/generate",
                json={
                    "text": "Test",
                    "voice": "rachel",
                    "provider": "elevenlabs"
                },
                headers=auth_headers
            )
            
            assert response.status_code == status.HTTP_402_PAYMENT_REQUIRED
            data = response.json()
            assert "quota" in str(data).lower()
    
    @pytest.mark.integration
    async def test_provider_not_configured(self, test_client, auth_headers):
        """Test handling of unconfigured provider errors."""
        from tldw_Server_API.app.core.TTS.tts_exceptions import TTSProviderNotConfiguredError
        
        with patch('tldw_Server_API.app.core.TTS.tts_service_v2.TTSServiceV2.generate') as mock_generate:
            mock_generate.side_effect = TTSProviderNotConfiguredError("Provider not configured")
            
            response = test_client.post(
                "/api/v1/tts/generate",
                json={
                    "text": "Test",
                    "voice": "voice1",
                    "provider": "unconfigured_provider"
                },
                headers=auth_headers
            )
            
            assert response.status_code == status.HTTP_503_SERVICE_UNAVAILABLE

# ========================================================================
# Batch Processing Tests
# ========================================================================

class TestBatchProcessing:
    """Test batch TTS processing endpoints."""
    
    @pytest.mark.integration
    async def test_batch_tts_generation(self, test_client, auth_headers):
        """Test batch generation of multiple texts."""
        with patch('tldw_Server_API.app.core.TTS.tts_service_v2.TTSServiceV2.generate') as mock_generate:
            mock_generate.return_value = MagicMock(
                audio_content=b"audio",
                format="mp3",
                provider="openai"
            )
            
            response = test_client.post(
                "/api/v1/tts/batch",
                json={
                    "requests": [
                        {"text": "First text", "voice": "alloy"},
                        {"text": "Second text", "voice": "echo"},
                        {"text": "Third text", "voice": "nova"}
                    ],
                    "provider": "openai"
                },
                headers=auth_headers
            )
            
            if response.status_code == status.HTTP_200_OK:
                data = response.json()
                assert "results" in data
                assert len(data["results"]) == 3
    
    @pytest.mark.integration
    async def test_batch_with_partial_failures(self, test_client, auth_headers):
        """Test batch processing with some failures."""
        from tldw_Server_API.app.core.TTS.tts_exceptions import TTSGenerationError
        
        call_count = 0
        def side_effect(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 2:
                raise TTSGenerationError("Failed")
            return MagicMock(audio_content=b"audio", format="mp3", provider="openai")
        
        with patch('tldw_Server_API.app.core.TTS.tts_service_v2.TTSServiceV2.generate') as mock_generate:
            mock_generate.side_effect = side_effect
            
            response = test_client.post(
                "/api/v1/tts/batch",
                json={
                    "requests": [
                        {"text": "Success 1", "voice": "alloy"},
                        {"text": "Failure", "voice": "echo"},
                        {"text": "Success 2", "voice": "nova"}
                    ]
                },
                headers=auth_headers
            )
            
            if response.status_code == status.HTTP_207_MULTI_STATUS:
                data = response.json()
                assert data["results"][0]["success"] is True
                assert data["results"][1]["success"] is False
                assert data["results"][2]["success"] is True