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
from fastapi.testclient import TestClient
#
# Local Imports
from tldw_Server_API.app.api.v1.schemas.audio_schemas import OpenAISpeechRequest
from tldw_Server_API.app.core.TTS.tts_service_v2 import TTSServiceV2
from tldw_Server_API.app.core.TTS.adapter_registry import TTSAdapterRegistry, TTSProvider
from tldw_Server_API.app.core.TTS.adapters.base import TTSAdapter, TTSCapabilities, AudioFormat
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
class TestOpenAIAdapter:
    """Tests for OpenAI TTS Adapter"""
    
    @pytest.mark.asyncio
    async def test_init_without_api_key(self):
        """Test initialization without API key"""
        adapter = OpenAIAdapter(config={})
        assert adapter.config == {}
        assert not adapter._initialized
        
        # Should fail to initialize without API key
        result = await adapter.initialize()
        assert not result
    
    @pytest.mark.asyncio
    async def test_init_with_api_key(self):
        """Test initialization with API key"""
        adapter = OpenAIAdapter(config={"api_key": "test-key"})
        assert adapter.config["api_key"] == "test-key"
        assert not adapter._initialized
    
    @pytest.mark.asyncio
    async def test_adapter_capabilities(self):
        """Test adapter capabilities"""
        adapter = OpenAIAdapter(config={"api_key": "test-key"})
        
        # Mock initialization
        adapter._initialized = True
        from tldw_Server_API.app.core.TTS.adapters.base import VoiceInfo
        adapter._capabilities = TTSCapabilities(
            provider_name="openai",
            supports_streaming=True,
            supports_voice_cloning=False,
            supported_languages={"en"},
            supported_formats={AudioFormat.MP3, AudioFormat.OPUS, AudioFormat.AAC, AudioFormat.FLAC},
            max_text_length=4096,
            supported_voices=[
                VoiceInfo(id="alloy", name="Alloy", language="en"),
                VoiceInfo(id="echo", name="Echo", language="en"),
                VoiceInfo(id="fable", name="Fable", language="en")
            ]
        )
        
        capabilities = await adapter.get_capabilities()
        assert capabilities.supports_streaming
        assert not capabilities.supports_voice_cloning
        assert "en" in capabilities.supported_languages


# TestTTSService removed - replaced by tests in test_tts_service_v2.py
    
    
    


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
        return TestClient(app)
    
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
        from tldw_Server_API.app.core.TTS import tts_service_v2
        from tldw_Server_API.app.core.TTS import adapter_registry
        from tldw_Server_API.app.core.TTS import streaming_audio_writer
        from tldw_Server_API.app.core.TTS import tts_exceptions
        from tldw_Server_API.app.core.TTS import tts_validation
        from tldw_Server_API.app.core.TTS import tts_resource_manager
        from tldw_Server_API.app.api.v1.schemas import audio_schemas
    except ImportError as e:
        pytest.fail(f"Failed to import TTS modules: {e}")


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v"])

#
# End of test_tts_module.py
#######################################################################################################################