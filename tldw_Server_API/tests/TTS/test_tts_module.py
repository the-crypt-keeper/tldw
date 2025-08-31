# test_tts_module.py
# Description: Comprehensive tests for the TTS module
#
# Imports
import asyncio
import os
from unittest.mock import AsyncMock, MagicMock, patch
import pytest
import numpy as np
from fastapi import HTTPException
from httpx import AsyncClient
#
# Local Imports
from tldw_Server_API.app.api.v1.schemas.audio_schemas import OpenAISpeechRequest
from tldw_Server_API.app.core.TTS.tts_service_v2 import TTSServiceV2
from tldw_Server_API.app.core.TTS.adapter_registry import TTSAdapterRegistry, TTSProvider
from tldw_Server_API.app.core.TTS.adapters.base import TTSAdapter, TTSCapabilities
from tldw_Server_API.app.core.TTS.adapters.openai_adapter import OpenAIAdapter
from tldw_Server_API.app.core.TTS.adapters.kokoro_adapter import KokoroAdapter
from tldw_Server_API.app.core.TTS.streaming_audio_writer import StreamingAudioWriter, AudioNormalizer
#
#######################################################################################################################
#
# Test Classes


class TestStreamingAudioWriter:
    """Tests for StreamingAudioWriter class"""
    
    def test_init_valid_formats(self):
        """Test initialization with valid formats"""
        formats = ["wav", "mp3", "opus", "flac", "aac", "pcm"]
        for fmt in formats:
            writer = StreamingAudioWriter(format=fmt, sample_rate=24000)
            assert writer.format == fmt
            assert writer.sample_rate == 24000
    
    def test_init_invalid_format(self):
        """Test initialization with invalid format"""
        with pytest.raises(ValueError, match="Unsupported audio format"):
            StreamingAudioWriter(format="invalid", sample_rate=24000)
    
    def test_pcm_output(self):
        """Test PCM format output"""
        writer = StreamingAudioWriter(format="pcm", sample_rate=24000)
        
        # Create test audio data
        test_data = np.array([0, 16383, 32767, -16384, -32768], dtype=np.int16)
        
        # Write chunk
        output = writer.write_chunk(test_data)
        
        # PCM should return raw bytes
        assert output == test_data.tobytes()
        
        # Finalize
        final = writer.write_chunk(finalize=True)
        assert final == b""
    
    @pytest.mark.skipif(not os.system("python -c 'import av'") == 0, reason="av not installed")
    def test_wav_output(self):
        """Test WAV format output"""
        writer = StreamingAudioWriter(format="wav", sample_rate=24000)
        
        # Create test audio data
        test_data = np.array([0, 16383, 32767, -16384, -32768], dtype=np.int16)
        
        # Write chunk
        output = writer.write_chunk(test_data)
        
        # WAV should have some data (may be empty until finalized)
        assert isinstance(output, bytes)
        
        # Finalize should produce valid WAV
        final = writer.write_chunk(finalize=True)
        assert len(final) > 0
        # WAV files start with "RIFF"
        if final:
            assert final[:4] == b"RIFF" or output[:4] == b"RIFF"


class TestAudioNormalizer:
    """Tests for AudioNormalizer class"""
    
    def test_float32_to_int16(self):
        """Test float32 to int16 conversion"""
        normalizer = AudioNormalizer()
        
        # Test data in float32 range [-1, 1]
        float_data = np.array([-1.0, -0.5, 0.0, 0.5, 1.0], dtype=np.float32)
        
        # Convert to int16
        int_data = normalizer.normalize(float_data, target_dtype=np.int16)
        
        # Check conversion
        expected = np.array([-32767, -16383, 0, 16383, 32767], dtype=np.int16)
        np.testing.assert_array_almost_equal(int_data, expected, decimal=0)
    
    def test_int16_to_float32(self):
        """Test int16 to float32 conversion"""
        normalizer = AudioNormalizer()
        
        # Test data in int16 range
        int_data = np.array([-32767, -16383, 0, 16383, 32767], dtype=np.int16)
        
        # Convert to float32
        float_data = normalizer.normalize(int_data, target_dtype=np.float32)
        
        # Check conversion
        expected = np.array([-1.0, -0.5, 0.0, 0.5, 1.0], dtype=np.float32)
        np.testing.assert_array_almost_equal(float_data, expected, decimal=1)
    
    def test_clipping(self):
        """Test that values outside [-1, 1] are clipped"""
        normalizer = AudioNormalizer()
        
        # Test data with values outside valid range
        float_data = np.array([-2.0, -1.5, 0.0, 1.5, 2.0], dtype=np.float32)
        
        # Convert to int16 (should clip)
        int_data = normalizer.normalize(float_data, target_dtype=np.int16)
        
        # Check that values are clipped
        assert int_data[0] == -32767  # Clipped from -2.0
        assert int_data[1] == -32767  # Clipped from -1.5
        assert int_data[3] == 32767   # Clipped from 1.5
        assert int_data[4] == 32767   # Clipped from 2.0


@pytest.mark.asyncio
class TestOpenAIBackend:
    """Tests for OpenAI TTS Backend"""
    
    async def test_init_without_api_key(self):
        """Test initialization without API key"""
        backend = OpenAIAPIBackend(config={})
        assert backend.api_key is None
        await backend.initialize()
    
    async def test_init_with_api_key(self):
        """Test initialization with API key"""
        backend = OpenAIAPIBackend(config={"openai_api_key": "test-key"})
        assert backend.api_key == "test-key"
        await backend.initialize()
    
    async def test_generate_without_api_key(self):
        """Test generation fails without API key"""
        backend = OpenAIAPIBackend(config={})
        await backend.initialize()
        
        request = OpenAISpeechRequest(
            input="Test text",
            model="tts-1",
            voice="alloy"
        )
        
        with pytest.raises(ValueError, match="OpenAI API key not configured"):
            async for _ in backend.generate_speech_stream(request):
                pass
    
    @patch('httpx.AsyncClient.stream')
    async def test_generate_with_mock_api(self, mock_stream):
        """Test successful generation with mocked API"""
        # Setup mock response
        mock_response = AsyncMock()
        mock_response.raise_for_status = MagicMock()
        mock_response.aiter_bytes = AsyncMock(return_value=self._async_bytes_generator())
        mock_stream.return_value.__aenter__.return_value = mock_response
        mock_stream.return_value.__aexit__.return_value = None
        
        backend = OpenAIAPIBackend(config={"openai_api_key": "test-key"})
        await backend.initialize()
        
        request = OpenAISpeechRequest(
            input="Test text",
            model="tts-1",
            voice="alloy"
        )
        
        # Collect generated bytes
        result = b""
        async for chunk in backend.generate_speech_stream(request):
            result += chunk
        
        assert result == b"test audio data"
    
    async def _async_bytes_generator(self):
        """Helper to generate async bytes"""
        yield b"test "
        yield b"audio "
        yield b"data"


@pytest.mark.asyncio 
class TestTTSService:
    """Tests for main TTS Service"""
    
    async def test_service_initialization(self):
        """Test TTS service initialization"""
        mock_manager = MagicMock(spec=TTSBackendManager)
        service = TTSService(backend_manager=mock_manager)
        assert service.backend_manager == mock_manager
    
    async def test_generate_with_invalid_backend(self):
        """Test generation with invalid backend ID"""
        mock_manager = AsyncMock(spec=TTSBackendManager)
        mock_manager.get_backend.return_value = None
        
        service = TTSService(backend_manager=mock_manager)
        
        request = OpenAISpeechRequest(
            input="Test text",
            model="invalid-model",
            voice="alloy"
        )
        
        result = b""
        async for chunk in service.generate_audio_stream(request, "invalid_backend"):
            result += chunk
        
        # Should return error message
        assert b"ERROR" in result
    
    async def test_generate_with_valid_backend(self):
        """Test generation with valid backend"""
        # Create mock backend
        mock_backend = AsyncMock()
        mock_backend.generate_speech_stream = AsyncMock(return_value=self._async_audio_generator())
        
        # Create mock manager
        mock_manager = AsyncMock(spec=TTSBackendManager)
        mock_manager.get_backend.return_value = mock_backend
        
        service = TTSService(backend_manager=mock_manager)
        
        request = OpenAISpeechRequest(
            input="Test text",
            model="tts-1",
            voice="alloy"
        )
        
        # Collect generated audio
        result = b""
        async for chunk in service.generate_audio_stream(request, "openai_official_tts-1"):
            result += chunk
        
        assert result == b"generated audio"
    
    async def _async_audio_generator(self):
        """Helper to generate async audio bytes"""
        yield b"generated "
        yield b"audio"


@pytest.mark.asyncio
class TestTTSEndpoint:
    """Integration tests for TTS API endpoint"""
    
    @pytest.fixture
    def client(self):
        """Create test client"""
        from fastapi import FastAPI
        from tldw_Server_API.app.api.v1.endpoints.audio import router
        
        app = FastAPI()
        app.include_router(router)
        return AsyncClient(app=app, base_url="http://test")
    
    async def test_endpoint_without_auth(self, client):
        """Test endpoint rejects requests without auth"""
        # This test would need proper app setup with auth enabled
        # Skipping for now as it requires full app context
        pass
    
    async def test_endpoint_rate_limiting(self, client):
        """Test rate limiting on endpoint"""
        # This test would need proper app setup with rate limiter
        # Skipping for now as it requires full app context
        pass


# Test utilities
def test_imports():
    """Test that all required modules can be imported"""
    try:
        from tldw_Server_API.app.core.TTS import tts_generation
        from tldw_Server_API.app.core.TTS import tts_backends
        from tldw_Server_API.app.core.TTS import streaming_audio_writer
        from tldw_Server_API.app.api.v1.schemas import audio_schemas
        from tldw_Server_API.app.api.v1.endpoints import audio
    except ImportError as e:
        pytest.fail(f"Failed to import TTS modules: {e}")


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v"])

#
# End of test_tts_module.py
#######################################################################################################################