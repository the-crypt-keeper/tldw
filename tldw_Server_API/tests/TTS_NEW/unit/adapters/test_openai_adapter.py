"""
Unit tests for OpenAI TTS adapter.

Tests the OpenAI TTS adapter with minimal mocking - only mocking
the actual OpenAI API calls.
"""

import pytest
from unittest.mock import Mock, AsyncMock, patch, MagicMock
import httpx
from io import BytesIO

from tldw_Server_API.app.core.TTS.adapters.openai_adapter import OpenAITTSAdapter
from tldw_Server_API.app.core.TTS.adapters.base import (
    TTSRequest,
    TTSResponse,
    AudioFormat,
    VoiceSettings
)
from tldw_Server_API.app.core.TTS.tts_exceptions import (
    TTSProviderNotConfiguredError,
    TTSGenerationError,
    TTSRateLimitError,
    TTSValidationError
)

# ========================================================================
# Adapter Initialization Tests
# ========================================================================

class TestOpenAIAdapterInitialization:
    """Test OpenAI adapter initialization and configuration."""
    
    @pytest.mark.unit
    def test_adapter_initialization_with_config(self):
        """Test adapter initialization with configuration."""
        config = {
            "api_key": "test-key-123",
            "base_url": "https://api.openai.com/v1",
            "timeout": 30
        }
        
        adapter = OpenAITTSAdapter(config)
        
        assert adapter.provider == "openai"
        assert adapter.is_available
        assert adapter.api_key == "test-key-123"
    
    @pytest.mark.unit
    def test_adapter_initialization_without_api_key(self):
        """Test adapter initialization without API key."""
        config = {
            "base_url": "https://api.openai.com/v1"
        }
        
        with pytest.raises(TTSProviderNotConfiguredError):
            OpenAITTSAdapter(config)
    
    @pytest.mark.unit
    def test_adapter_supported_models(self):
        """Test adapter reports supported models."""
        config = {"api_key": "test-key"}
        adapter = OpenAITTSAdapter(config)
        
        assert "tts-1" in adapter.supported_models
        assert "tts-1-hd" in adapter.supported_models
    
    @pytest.mark.unit
    def test_adapter_supported_voices(self):
        """Test adapter reports supported voices."""
        config = {"api_key": "test-key"}
        adapter = OpenAITTSAdapter(config)
        
        expected_voices = ["alloy", "echo", "fable", "onyx", "nova", "shimmer"]
        for voice in expected_voices:
            assert voice in adapter.supported_voices

# ========================================================================
# Request Validation Tests
# ========================================================================

class TestRequestValidation:
    """Test request validation in OpenAI adapter."""
    
    @pytest.mark.unit
    async def test_validate_valid_request(self):
        """Test validation of valid request."""
        adapter = OpenAITTSAdapter({"api_key": "test-key"})
        
        request = TTSRequest(
            text="Hello world",
            voice="alloy",
            model="tts-1"
        )
        
        # Should not raise
        await adapter.validate_request(request)
    
    @pytest.mark.unit
    async def test_validate_invalid_voice(self):
        """Test validation rejects invalid voice."""
        adapter = OpenAITTSAdapter({"api_key": "test-key"})
        
        request = TTSRequest(
            text="Hello",
            voice="invalid_voice",
            model="tts-1"
        )
        
        with pytest.raises(TTSValidationError):
            await adapter.validate_request(request)
    
    @pytest.mark.unit
    async def test_validate_invalid_model(self):
        """Test validation rejects invalid model."""
        adapter = OpenAITTSAdapter({"api_key": "test-key"})
        
        request = TTSRequest(
            text="Hello",
            voice="alloy",
            model="invalid-model"
        )
        
        with pytest.raises(TTSValidationError):
            await adapter.validate_request(request)
    
    @pytest.mark.unit
    async def test_validate_text_too_long(self):
        """Test validation rejects text exceeding limit."""
        adapter = OpenAITTSAdapter({"api_key": "test-key"})
        
        # OpenAI has 4096 char limit
        long_text = "a" * 5000
        request = TTSRequest(
            text=long_text,
            voice="alloy",
            model="tts-1"
        )
        
        with pytest.raises(TTSValidationError) as exc_info:
            await adapter.validate_request(request)
        
        assert "exceeds maximum" in str(exc_info.value)
    
    @pytest.mark.unit
    async def test_validate_speed_out_of_range(self):
        """Test validation rejects speed out of range."""
        adapter = OpenAITTSAdapter({"api_key": "test-key"})
        
        request = TTSRequest(
            text="Hello",
            voice="alloy",
            model="tts-1",
            speed=5.0  # OpenAI supports 0.25-4.0
        )
        
        with pytest.raises(TTSValidationError):
            await adapter.validate_request(request)

# ========================================================================
# Audio Generation Tests
# ========================================================================

class TestAudioGeneration:
    """Test audio generation functionality."""
    
    @pytest.mark.unit
    @patch('httpx.AsyncClient.post')
    async def test_generate_basic_audio(self, mock_post):
        """Test basic audio generation."""
        # Mock response
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.content = b"fake_audio_data"
        mock_response.headers = {"content-type": "audio/mpeg"}
        mock_post.return_value = mock_response
        
        adapter = OpenAITTSAdapter({"api_key": "test-key"})
        request = TTSRequest(
            text="Hello world",
            voice="alloy",
            model="tts-1"
        )
        
        response = await adapter.generate(request)
        
        assert response.audio_content == b"fake_audio_data"
        assert response.provider == "openai"
        assert response.model == "tts-1"
        
        # Verify API call
        mock_post.assert_called_once()
        call_args = mock_post.call_args
        assert "audio/speech" in str(call_args[0][0])
    
    @pytest.mark.unit
    @patch('httpx.AsyncClient.post')
    async def test_generate_with_hd_model(self, mock_post):
        """Test generation with HD model."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.content = b"hd_audio_data"
        mock_post.return_value = mock_response
        
        adapter = OpenAITTSAdapter({"api_key": "test-key"})
        request = TTSRequest(
            text="High quality audio",
            voice="nova",
            model="tts-1-hd"
        )
        
        response = await adapter.generate(request)
        
        assert response.model == "tts-1-hd"
        assert response.audio_content == b"hd_audio_data"
    
    @pytest.mark.unit
    @patch('httpx.AsyncClient.post')
    async def test_generate_with_speed_adjustment(self, mock_post):
        """Test generation with speed adjustment."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.content = b"audio_data"
        mock_post.return_value = mock_response
        
        adapter = OpenAITTSAdapter({"api_key": "test-key"})
        request = TTSRequest(
            text="Slow speech",
            voice="echo",
            model="tts-1",
            speed=0.75
        )
        
        response = await adapter.generate(request)
        
        # Verify speed was included in request
        call_kwargs = mock_post.call_args[1]
        assert call_kwargs['json']['speed'] == 0.75
    
    @pytest.mark.unit
    @patch('httpx.AsyncClient.post')
    async def test_generate_different_formats(self, mock_post):
        """Test generation with different audio formats."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.content = b"audio_data"
        mock_post.return_value = mock_response
        
        adapter = OpenAITTSAdapter({"api_key": "test-key"})
        
        # Test different formats
        formats = [AudioFormat.MP3, AudioFormat.OPUS, AudioFormat.AAC, AudioFormat.FLAC]
        
        for audio_format in formats:
            request = TTSRequest(
                text="Test",
                voice="alloy",
                model="tts-1",
                format=audio_format
            )
            
            response = await adapter.generate(request)
            assert response.format == audio_format

# ========================================================================
# Streaming Generation Tests
# ========================================================================

class TestStreamingGeneration:
    """Test streaming audio generation."""
    
    @pytest.mark.unit
    @patch('httpx.AsyncClient.stream')
    async def test_streaming_generation(self, mock_stream):
        """Test streaming audio generation."""
        # Mock streaming response
        async def mock_iter():
            for chunk in [b"chunk1", b"chunk2", b"chunk3"]:
                yield chunk
        
        mock_response = AsyncMock()
        mock_response.aiter_bytes = mock_iter
        mock_response.__aenter__ = AsyncMock(return_value=mock_response)
        mock_response.__aexit__ = AsyncMock()
        
        mock_stream.return_value = mock_response
        
        adapter = OpenAITTSAdapter({"api_key": "test-key"})
        request = TTSRequest(
            text="Stream this",
            voice="fable",
            model="tts-1"
        )
        
        chunks = []
        async for chunk in adapter.generate_stream(request):
            chunks.append(chunk)
        
        assert chunks == [b"chunk1", b"chunk2", b"chunk3"]
    
    @pytest.mark.unit
    @patch('httpx.AsyncClient.stream')
    async def test_streaming_with_error(self, mock_stream):
        """Test streaming handles errors gracefully."""
        async def mock_error_iter():
            yield b"chunk1"
            raise httpx.HTTPError("Connection lost")
        
        mock_response = AsyncMock()
        mock_response.aiter_bytes = mock_error_iter
        mock_response.__aenter__ = AsyncMock(return_value=mock_response)
        mock_response.__aexit__ = AsyncMock()
        
        mock_stream.return_value = mock_response
        
        adapter = OpenAITTSAdapter({"api_key": "test-key"})
        request = TTSRequest(text="Test", voice="alloy", model="tts-1")
        
        chunks = []
        with pytest.raises(TTSGenerationError):
            async for chunk in adapter.generate_stream(request):
                chunks.append(chunk)
        
        # Should have received first chunk before error
        assert len(chunks) == 1

# ========================================================================
# Error Handling Tests
# ========================================================================

class TestErrorHandling:
    """Test error handling in OpenAI adapter."""
    
    @pytest.mark.unit
    @patch('httpx.AsyncClient.post')
    async def test_handle_rate_limit_error(self, mock_post):
        """Test handling of rate limit errors."""
        mock_response = MagicMock()
        mock_response.status_code = 429
        mock_response.headers = {"retry-after": "60"}
        mock_response.json.return_value = {"error": {"message": "Rate limit exceeded"}}
        mock_response.raise_for_status.side_effect = httpx.HTTPStatusError(
            "429", request=Mock(), response=mock_response
        )
        mock_post.return_value = mock_response
        
        adapter = OpenAITTSAdapter({"api_key": "test-key"})
        request = TTSRequest(text="Test", voice="alloy", model="tts-1")
        
        with pytest.raises(TTSRateLimitError) as exc_info:
            await adapter.generate(request)
        
        assert exc_info.value.retry_after == 60
    
    @pytest.mark.unit
    @patch('httpx.AsyncClient.post')
    async def test_handle_api_error(self, mock_post):
        """Test handling of API errors."""
        mock_response = MagicMock()
        mock_response.status_code = 500
        mock_response.json.return_value = {"error": {"message": "Internal server error"}}
        mock_response.raise_for_status.side_effect = httpx.HTTPStatusError(
            "500", request=Mock(), response=mock_response
        )
        mock_post.return_value = mock_response
        
        adapter = OpenAITTSAdapter({"api_key": "test-key"})
        request = TTSRequest(text="Test", voice="alloy", model="tts-1")
        
        with pytest.raises(TTSGenerationError):
            await adapter.generate(request)
    
    @pytest.mark.unit
    @patch('httpx.AsyncClient.post')
    async def test_handle_network_error(self, mock_post):
        """Test handling of network errors."""
        mock_post.side_effect = httpx.ConnectError("Connection failed")
        
        adapter = OpenAITTSAdapter({"api_key": "test-key"})
        request = TTSRequest(text="Test", voice="alloy", model="tts-1")
        
        with pytest.raises(TTSGenerationError) as exc_info:
            await adapter.generate(request)
        
        assert "Connection" in str(exc_info.value)

# ========================================================================
# Metadata and Info Tests
# ========================================================================

class TestMetadataAndInfo:
    """Test metadata and information methods."""
    
    @pytest.mark.unit
    def test_get_adapter_info(self):
        """Test getting adapter information."""
        adapter = OpenAITTSAdapter({"api_key": "test-key"})
        
        info = adapter.get_info()
        
        assert info["provider"] == "openai"
        assert "tts-1" in info["models"]
        assert "alloy" in info["voices"]
        assert info["max_characters"] == 4096
    
    @pytest.mark.unit
    @patch('httpx.AsyncClient.post')
    async def test_response_includes_metadata(self, mock_post):
        """Test that response includes proper metadata."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.content = b"audio_data"
        mock_post.return_value = mock_response
        
        adapter = OpenAITTSAdapter({"api_key": "test-key"})
        request = TTSRequest(
            text="Test metadata",
            voice="shimmer",
            model="tts-1-hd"
        )
        
        response = await adapter.generate(request)
        
        assert response.provider == "openai"
        assert response.model == "tts-1-hd"
        assert "characters" in response.metadata
        assert response.metadata["characters"] == len("Test metadata")