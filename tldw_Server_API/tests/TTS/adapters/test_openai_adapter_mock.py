# test_openai_adapter_mock.py
# Description: Mock/Unit tests for OpenAI TTS adapter
#
# Imports
import pytest
import asyncio
from unittest.mock import AsyncMock, MagicMock, patch
import httpx
#
# Local Imports
from tldw_Server_API.app.core.TTS.adapters.openai_adapter import OpenAIAdapter
from tldw_Server_API.app.core.TTS.adapters.base import (
    TTSRequest,
    TTSResponse,
    AudioFormat,
    ProviderStatus
)
from tldw_Server_API.app.core.TTS.tts_exceptions import (
    TTSProviderNotConfiguredError,
    TTSAuthenticationError,
    TTSRateLimitError,
    TTSProviderError
)
#
#######################################################################################################################
#
# Mock Tests for OpenAI Adapter

@pytest.mark.asyncio
class TestOpenAIAdapterMock:
    """Mock/Unit tests for OpenAI adapter"""
    
    async def test_initialization_without_api_key(self):
        """Test initialization without API key"""
        adapter = OpenAIAdapter({})
        assert adapter.api_key is None
        
        # Should not initialize without key
        success = await adapter.initialize()
        assert not success
        assert adapter._status == ProviderStatus.NOT_CONFIGURED
    
    async def test_initialization_with_api_key(self):
        """Test initialization with API key"""
        adapter = OpenAIAdapter({"openai_api_key": "test-key-123"})
        assert adapter.api_key == "test-key-123"
        
        success = await adapter.initialize()
        assert success
        assert adapter._status == ProviderStatus.AVAILABLE
    
    async def test_capabilities_reporting(self):
        """Test capabilities are correctly reported"""
        adapter = OpenAIAdapter({"openai_api_key": "test-key"})
        await adapter.initialize()
        
        caps = await adapter.get_capabilities()
        
        assert caps.provider_name == "OpenAI"
        assert caps.supports_streaming is True
        assert caps.supports_voice_cloning is False
        assert caps.max_text_length == 4096
        assert AudioFormat.MP3 in caps.supported_formats
        assert AudioFormat.OPUS in caps.supported_formats
        assert AudioFormat.AAC in caps.supported_formats
        assert AudioFormat.FLAC in caps.supported_formats
        assert AudioFormat.WAV in caps.supported_formats
        assert AudioFormat.PCM in caps.supported_formats
    
    async def test_voice_mapping(self):
        """Test voice mapping functionality"""
        adapter = OpenAIAdapter({})
        
        # Test standard voices
        assert adapter.map_voice("alloy") == "alloy"
        assert adapter.map_voice("echo") == "echo"
        assert adapter.map_voice("fable") == "fable"
        assert adapter.map_voice("onyx") == "onyx"
        assert adapter.map_voice("nova") == "nova"
        assert adapter.map_voice("shimmer") == "shimmer"
        
        # Test generic mappings
        assert adapter.map_voice("female") == "nova"
        assert adapter.map_voice("male") == "onyx"
        assert adapter.map_voice("narrator") == "fable"
        
        # Test unknown voice defaults
        assert adapter.map_voice("unknown") == "alloy"
    
    @patch('httpx.AsyncClient.post')
    async def test_successful_generation(self, mock_post):
        """Test successful audio generation"""
        # Mock successful response
        mock_response = AsyncMock()
        mock_response.status_code = 200
        mock_response.content = b"FAKE_AUDIO_DATA"
        mock_response.headers = {"content-type": "audio/mpeg"}
        mock_post.return_value = mock_response
        
        adapter = OpenAIAdapter({"openai_api_key": "test-key"})
        await adapter.initialize()
        
        request = TTSRequest(
            text="Hello world",
            voice="nova",
            format=AudioFormat.MP3,
            stream=False
        )
        
        response = await adapter.generate(request)
        
        assert response.audio_data == b"FAKE_AUDIO_DATA"
        assert response.format == AudioFormat.MP3
        assert response.voice_used == "nova"
        assert response.provider == "OpenAI"
        
        # Verify API call
        mock_post.assert_called_once()
        call_args = mock_post.call_args
        assert "speech" in call_args[0][0]  # URL contains 'speech'
    
    @patch('httpx.AsyncClient.post')
    async def test_authentication_error(self, mock_post):
        """Test authentication error handling"""
        # Mock 401 response
        mock_response = MagicMock()
        mock_response.status_code = 401
        mock_response.headers = {}
        mock_response.text = '{"error": {"message": "Invalid API key"}}'
        mock_response.json.return_value = {"error": {"message": "Invalid API key"}}
        mock_response.aread = AsyncMock(return_value=b'{"error": {"message": "Invalid API key"}}')
        
        error = httpx.HTTPStatusError(
            "401 Unauthorized",
            request=MagicMock(),
            response=mock_response
        )
        mock_post.side_effect = error
        
        adapter = OpenAIAdapter({"openai_api_key": "invalid-key"})
        await adapter.initialize()
        
        request = TTSRequest(text="Test", stream=False)
        
        with pytest.raises(TTSAuthenticationError) as exc_info:
            await adapter.generate(request)
        
        assert exc_info.value.provider == "OpenAI"
        assert "Invalid API key" in str(exc_info.value)
    
    @patch('httpx.AsyncClient.post')
    async def test_rate_limit_error(self, mock_post):
        """Test rate limit error handling"""
        # Mock 429 response
        mock_response = MagicMock()
        mock_response.status_code = 429
        mock_response.headers = {"retry-after": "60"}
        mock_response.text = '{"error": {"message": "Rate limit exceeded"}}'
        mock_response.json.return_value = {"error": {"message": "Rate limit exceeded"}}
        mock_response.aread = AsyncMock(return_value=b'{"error": {"message": "Rate limit exceeded"}}')
        
        error = httpx.HTTPStatusError(
            "429 Too Many Requests",
            request=MagicMock(),
            response=mock_response
        )
        mock_post.side_effect = error
        
        adapter = OpenAIAdapter({"openai_api_key": "test-key"})
        await adapter.initialize()
        
        request = TTSRequest(text="Test", stream=False)
        
        with pytest.raises(TTSRateLimitError) as exc_info:
            await adapter.generate(request)
        
        assert exc_info.value.provider == "OpenAI"
    
    @patch('httpx.AsyncClient.post')
    async def test_streaming_generation(self, mock_post):
        """Test streaming audio generation"""
        # Mock streaming response
        async def mock_aiter_bytes(chunk_size=1024):
            yield b"chunk1"
            yield b"chunk2"
            yield b"chunk3"
        
        mock_response = AsyncMock()
        mock_response.status_code = 200
        mock_response.headers = {"content-type": "audio/mpeg"}
        mock_response.aiter_bytes = mock_aiter_bytes
        mock_post.return_value = mock_response
        
        adapter = OpenAIAdapter({"openai_api_key": "test-key"})
        await adapter.initialize()
        
        request = TTSRequest(
            text="Stream test",
            voice="nova",
            format=AudioFormat.MP3,
            stream=True
        )
        
        response = await adapter.generate(request)
        
        assert response.audio_stream is not None
        assert response.audio_data is None
        
        # Collect streamed chunks
        chunks = []
        async for chunk in response.audio_stream:
            chunks.append(chunk)
        
        assert chunks == [b"chunk1", b"chunk2", b"chunk3"]
    
    async def test_request_validation(self):
        """Test request validation"""
        adapter = OpenAIAdapter({"openai_api_key": "test-key"})
        await adapter.initialize()
        
        # Test with empty text
        with pytest.raises(Exception):  # Should raise validation error
            request = TTSRequest(text="", voice="nova")
            await adapter.generate(request)
        
        # Test with text exceeding limit
        long_text = "a" * 5000  # Exceeds 4096 character limit
        request = TTSRequest(text=long_text, voice="nova")
        
        with pytest.raises(Exception):  # Should raise validation error
            await adapter.generate(request)
    
    async def test_model_selection(self):
        """Test model selection"""
        adapter = OpenAIAdapter({
            "openai_api_key": "test-key",
            "openai_tts_model": "tts-1-hd"
        })
        
        assert adapter.model == "tts-1-hd"
        
        # Test with default model
        adapter2 = OpenAIAdapter({"openai_api_key": "test-key"})
        assert adapter2.model == "tts-1"
    
    async def test_cleanup_on_close(self):
        """Test resource cleanup on close"""
        adapter = OpenAIAdapter({"openai_api_key": "test-key"})
        await adapter.initialize()
        
        assert adapter._initialized is True
        assert adapter._status == ProviderStatus.AVAILABLE
        
        await adapter.close()
        
        assert adapter._initialized is False
        assert adapter._status == ProviderStatus.DISABLED

#######################################################################################################################
#
# End of test_openai_adapter_mock.py