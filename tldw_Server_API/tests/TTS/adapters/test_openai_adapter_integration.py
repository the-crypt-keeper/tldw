# test_openai_adapter_integration.py
# Description: Integration tests for OpenAI TTS adapter
#
# Imports
import pytest
import os
import asyncio
import platform
#
# Local Imports
from tldw_Server_API.app.core.TTS.adapters.openai_adapter import OpenAIAdapter
from tldw_Server_API.app.core.TTS.adapters.base import (
    TTSRequest,
    AudioFormat,
    ProviderStatus
)
from tldw_Server_API.app.core.TTS.tts_exceptions import (
    TTSAuthenticationError,
    TTSRateLimitError
)
#
#######################################################################################################################
#
# Integration Tests for OpenAI Adapter

@pytest.mark.integration
@pytest.mark.asyncio
class TestOpenAIAdapterIntegration:
    """Integration tests for OpenAI adapter - requires real API key"""
    
    @pytest.mark.skipif(
        not os.getenv("OPENAI_API_KEY"),
        reason="Requires OPENAI_API_KEY environment variable"
    )
    async def test_real_api_initialization(self):
        """Test initialization with real API key"""
        adapter = OpenAIAdapter({
            "openai_api_key": os.getenv("OPENAI_API_KEY")
        })
        
        success = await adapter.initialize()
        assert success
        assert adapter._status == ProviderStatus.AVAILABLE
        
        # Cleanup
        await adapter.close()
    
    @pytest.mark.skipif(
        not os.getenv("OPENAI_API_KEY"),
        reason="Requires OPENAI_API_KEY environment variable"
    )
    async def test_real_audio_generation(self):
        """Test actual audio generation with OpenAI API"""
        adapter = OpenAIAdapter({
            "openai_api_key": os.getenv("OPENAI_API_KEY")
        })
        
        await adapter.initialize()
        
        request = TTSRequest(
            text="Hello, this is a test of the OpenAI text-to-speech system.",
            voice="nova",
            format=AudioFormat.MP3,
            speed=1.0,
            stream=False
        )
        
        response = await adapter.generate(request)
        
        # Verify response
        assert response.audio_data is not None
        assert len(response.audio_data) > 1000  # Should have substantial audio data
        assert response.format == AudioFormat.MP3
        assert response.voice_used == "nova"
        assert response.provider == "OpenAI"
        
        # Cleanup
        await adapter.close()
    
    @pytest.mark.skipif(
        not os.getenv("OPENAI_API_KEY"),
        reason="Requires OPENAI_API_KEY environment variable"
    )
    async def test_real_streaming_generation(self):
        """Test streaming audio generation with real API"""
        adapter = OpenAIAdapter({
            "openai_api_key": os.getenv("OPENAI_API_KEY")
        })
        
        await adapter.initialize()
        
        request = TTSRequest(
            text="This is a streaming test.",
            voice="echo",
            format=AudioFormat.MP3,
            stream=True
        )
        
        response = await adapter.generate(request)
        
        assert response.audio_stream is not None
        assert response.audio_data is None
        
        # Collect streamed data
        chunks = []
        async for chunk in response.audio_stream:
            chunks.append(chunk)
        
        assert len(chunks) > 0
        total_size = sum(len(chunk) for chunk in chunks)
        assert total_size > 1000  # Should have substantial audio data
        
        # Cleanup
        await adapter.close()
    
    @pytest.mark.skipif(
        not os.getenv("OPENAI_API_KEY"),
        reason="Requires OPENAI_API_KEY environment variable"
    )
    async def test_different_voices(self):
        """Test generation with different voices"""
        adapter = OpenAIAdapter({
            "openai_api_key": os.getenv("OPENAI_API_KEY")
        })
        
        await adapter.initialize()
        
        voices = ["alloy", "echo", "fable", "onyx", "nova", "shimmer"]
        
        for voice in voices:
            request = TTSRequest(
                text=f"Testing voice: {voice}",
                voice=voice,
                format=AudioFormat.MP3,
                stream=False
            )
            
            response = await adapter.generate(request)
            
            assert response.audio_data is not None
            assert response.voice_used == voice
        
        # Cleanup
        await adapter.close()
    
    @pytest.mark.skipif(
        not os.getenv("OPENAI_API_KEY"),
        reason="Requires OPENAI_API_KEY environment variable"
    )
    async def test_different_formats(self):
        """Test generation with different audio formats"""
        adapter = OpenAIAdapter({
            "openai_api_key": os.getenv("OPENAI_API_KEY")
        })
        
        await adapter.initialize()
        
        formats = [AudioFormat.MP3, AudioFormat.OPUS, AudioFormat.AAC, AudioFormat.FLAC]
        
        for audio_format in formats:
            request = TTSRequest(
                text="Format test",
                voice="nova",
                format=audio_format,
                stream=False
            )
            
            response = await adapter.generate(request)
            
            assert response.audio_data is not None
            assert response.format == audio_format
        
        # Cleanup
        await adapter.close()
    
    @pytest.mark.skipif(
        not os.getenv("OPENAI_API_KEY"),
        reason="Requires OPENAI_API_KEY environment variable"
    )
    async def test_speed_variations(self):
        """Test generation with different speech speeds"""
        adapter = OpenAIAdapter({
            "openai_api_key": os.getenv("OPENAI_API_KEY")
        })
        
        await adapter.initialize()
        
        speeds = [0.25, 1.0, 2.0, 4.0]
        
        for speed in speeds:
            request = TTSRequest(
                text="Speed test",
                voice="nova",
                format=AudioFormat.MP3,
                speed=speed,
                stream=False
            )
            
            response = await adapter.generate(request)
            
            assert response.audio_data is not None
        
        # Cleanup
        await adapter.close()
    
    @pytest.mark.skipif(
        not os.getenv("OPENAI_API_KEY"),
        reason="Requires OPENAI_API_KEY environment variable"
    )
    async def test_concurrent_requests(self):
        """Test handling multiple concurrent requests"""
        adapter = OpenAIAdapter({
            "openai_api_key": os.getenv("OPENAI_API_KEY")
        })
        
        await adapter.initialize()
        
        # Create multiple concurrent requests
        requests = [
            TTSRequest(
                text=f"Concurrent request {i}",
                voice="nova",
                format=AudioFormat.MP3,
                stream=False
            )
            for i in range(3)
        ]
        
        # Execute concurrently
        tasks = [adapter.generate(req) for req in requests]
        responses = await asyncio.gather(*tasks)
        
        # Verify all succeeded
        assert len(responses) == 3
        for i, response in enumerate(responses):
            assert response.audio_data is not None
            assert response.provider == "OpenAI"
        
        # Cleanup
        await adapter.close()
    
    async def test_invalid_api_key(self):
        """Test with invalid API key"""
        adapter = OpenAIAdapter({
            "openai_api_key": "invalid-key-12345"
        })
        
        await adapter.initialize()
        
        request = TTSRequest(
            text="Test",
            voice="nova",
            format=AudioFormat.MP3,
            stream=False
        )
        
        with pytest.raises(TTSAuthenticationError):
            await adapter.generate(request)
        
        # Cleanup
        await adapter.close()
    
    @pytest.mark.skipif(
        not os.getenv("OPENAI_API_KEY"),
        reason="Requires OPENAI_API_KEY environment variable"
    )
    async def test_long_text_generation(self):
        """Test generation with long text"""
        adapter = OpenAIAdapter({
            "openai_api_key": os.getenv("OPENAI_API_KEY")
        })
        
        await adapter.initialize()
        
        # Create text near the limit
        long_text = "This is a test sentence. " * 100  # ~2500 characters
        
        request = TTSRequest(
            text=long_text,
            voice="nova",
            format=AudioFormat.MP3,
            stream=False
        )
        
        response = await adapter.generate(request)
        
        assert response.audio_data is not None
        assert len(response.audio_data) > 10000  # Should be substantial
        
        # Cleanup
        await adapter.close()
    
    @pytest.mark.skipif(
        not os.getenv("OPENAI_API_KEY"),
        reason="Requires OPENAI_API_KEY environment variable"
    )
    async def test_hd_model(self):
        """Test with HD model if available"""
        adapter = OpenAIAdapter({
            "openai_api_key": os.getenv("OPENAI_API_KEY"),
            "openai_tts_model": "tts-1-hd"
        })
        
        await adapter.initialize()
        
        request = TTSRequest(
            text="Testing HD model quality",
            voice="nova",
            format=AudioFormat.MP3,
            stream=False
        )
        
        response = await adapter.generate(request)
        
        assert response.audio_data is not None
        assert response.provider == "OpenAI"
        
        # Cleanup
        await adapter.close()

#######################################################################################################################
#
# End of test_openai_adapter_integration.py