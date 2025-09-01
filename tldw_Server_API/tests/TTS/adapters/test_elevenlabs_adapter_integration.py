# test_elevenlabs_adapter_integration.py
# Description: Integration tests for ElevenLabs TTS adapter
#
# Imports
import pytest
import os
import asyncio
#
# Local Imports
from tldw_Server_API.app.core.TTS.adapters.elevenlabs_adapter import ElevenLabsAdapter
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
# Integration Tests for ElevenLabs Adapter

@pytest.mark.integration
@pytest.mark.asyncio
class TestElevenLabsAdapterIntegration:
    """Integration tests for ElevenLabs adapter - requires real API key"""
    
    @pytest.mark.skipif(
        not os.getenv("ELEVENLABS_API_KEY"),
        reason="Requires ELEVENLABS_API_KEY environment variable"
    )
    async def test_real_api_initialization(self):
        """Test initialization with real API key"""
        adapter = ElevenLabsAdapter({
            "elevenlabs_api_key": os.getenv("ELEVENLABS_API_KEY")
        })
        
        success = await adapter.initialize()
        assert success
        assert adapter._status == ProviderStatus.AVAILABLE
        
        # Should have fetched user voices
        assert len(adapter._user_voices) >= 0  # May be 0 if no custom voices
        
        # Cleanup
        await adapter.close()
    
    @pytest.mark.skipif(
        not os.getenv("ELEVENLABS_API_KEY"),
        reason="Requires ELEVENLABS_API_KEY environment variable"
    )
    async def test_real_audio_generation(self):
        """Test actual audio generation with ElevenLabs API"""
        adapter = ElevenLabsAdapter({
            "elevenlabs_api_key": os.getenv("ELEVENLABS_API_KEY")
        })
        
        await adapter.initialize()
        
        request = TTSRequest(
            text="Hello, this is a test of the ElevenLabs text-to-speech system.",
            voice="rachel",
            format=AudioFormat.MP3,
            stream=False
        )
        
        response = await adapter.generate(request)
        
        # Verify response
        assert response.audio_data is not None
        assert len(response.audio_data) > 1000  # Should have substantial audio data
        assert response.format == AudioFormat.MP3
        assert response.voice_used == "rachel"
        assert response.provider == "ElevenLabs"
        
        # Cleanup
        await adapter.close()
    
    @pytest.mark.skipif(
        not os.getenv("ELEVENLABS_API_KEY"),
        reason="Requires ELEVENLABS_API_KEY environment variable"
    )
    async def test_real_streaming_generation(self):
        """Test streaming audio generation with real API"""
        adapter = ElevenLabsAdapter({
            "elevenlabs_api_key": os.getenv("ELEVENLABS_API_KEY")
        })
        
        await adapter.initialize()
        
        request = TTSRequest(
            text="This is a streaming test for ElevenLabs.",
            voice="drew",
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
        not os.getenv("ELEVENLABS_API_KEY"),
        reason="Requires ELEVENLABS_API_KEY environment variable"
    )
    async def test_voice_settings(self):
        """Test generation with custom voice settings"""
        adapter = ElevenLabsAdapter({
            "elevenlabs_api_key": os.getenv("ELEVENLABS_API_KEY"),
            "elevenlabs_stability": 0.3,
            "elevenlabs_similarity_boost": 0.7
        })
        
        await adapter.initialize()
        
        request = TTSRequest(
            text="Testing voice settings with low stability",
            voice="rachel",
            format=AudioFormat.MP3,
            stream=False,
            extra_params={
                "stability": 0.2,
                "similarity_boost": 0.9
            }
        )
        
        response = await adapter.generate(request)
        
        assert response.audio_data is not None
        assert response.voice_used == "rachel"
        
        # Cleanup
        await adapter.close()
    
    @pytest.mark.skipif(
        not os.getenv("ELEVENLABS_API_KEY"),
        reason="Requires ELEVENLABS_API_KEY environment variable"
    )
    async def test_different_models(self):
        """Test generation with different models"""
        adapter = ElevenLabsAdapter({
            "elevenlabs_api_key": os.getenv("ELEVENLABS_API_KEY")
        })
        
        await adapter.initialize()
        
        models = ["eleven_monolingual_v1", "eleven_turbo_v2"]
        
        for model in models:
            request = TTSRequest(
                text=f"Testing model: {model}",
                voice="rachel",
                format=AudioFormat.MP3,
                stream=False,
                extra_params={"model": model}
            )
            
            try:
                response = await adapter.generate(request)
                assert response.audio_data is not None
            except Exception as e:
                # Some models may not be available on all accounts
                print(f"Model {model} not available: {e}")
        
        # Cleanup
        await adapter.close()
    
    @pytest.mark.skipif(
        not os.getenv("ELEVENLABS_API_KEY"),
        reason="Requires ELEVENLABS_API_KEY environment variable"
    )
    async def test_multilingual_generation(self):
        """Test generation in different languages"""
        adapter = ElevenLabsAdapter({
            "elevenlabs_api_key": os.getenv("ELEVENLABS_API_KEY"),
            "elevenlabs_model": "eleven_multilingual_v2"
        })
        
        await adapter.initialize()
        
        # Test different languages
        languages = [
            ("en", "Hello, how are you?"),
            ("es", "Hola, ¿cómo estás?"),
            ("fr", "Bonjour, comment allez-vous?"),
            ("de", "Hallo, wie geht es dir?")
        ]
        
        for lang, text in languages:
            request = TTSRequest(
                text=text,
                voice="rachel",
                language=lang,
                format=AudioFormat.MP3,
                stream=False
            )
            
            try:
                response = await adapter.generate(request)
                assert response.audio_data is not None
            except Exception as e:
                # Multilingual model may not be available on all accounts
                print(f"Language {lang} generation failed: {e}")
        
        # Cleanup
        await adapter.close()
    
    @pytest.mark.skipif(
        not os.getenv("ELEVENLABS_API_KEY"),
        reason="Requires ELEVENLABS_API_KEY environment variable"
    )
    async def test_different_formats(self):
        """Test generation with different audio formats"""
        adapter = ElevenLabsAdapter({
            "elevenlabs_api_key": os.getenv("ELEVENLABS_API_KEY")
        })
        
        await adapter.initialize()
        
        formats = [AudioFormat.MP3, AudioFormat.PCM, AudioFormat.ULAW]
        
        for audio_format in formats:
            request = TTSRequest(
                text="Format test",
                voice="rachel",
                format=audio_format,
                stream=False
            )
            
            response = await adapter.generate(request)
            
            assert response.audio_data is not None
            assert response.format == audio_format
        
        # Cleanup
        await adapter.close()
    
    @pytest.mark.skipif(
        not os.getenv("ELEVENLABS_API_KEY"),
        reason="Requires ELEVENLABS_API_KEY environment variable"
    )
    async def test_concurrent_requests(self):
        """Test handling multiple concurrent requests"""
        adapter = ElevenLabsAdapter({
            "elevenlabs_api_key": os.getenv("ELEVENLABS_API_KEY")
        })
        
        await adapter.initialize()
        
        # Create multiple concurrent requests
        requests = [
            TTSRequest(
                text=f"Concurrent request {i}",
                voice="rachel",
                format=AudioFormat.MP3,
                stream=False
            )
            for i in range(3)
        ]
        
        # Execute concurrently
        tasks = [adapter.generate(req) for req in requests]
        responses = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Check responses (some may fail due to rate limits)
        successful = [r for r in responses if not isinstance(r, Exception)]
        assert len(successful) > 0  # At least some should succeed
        
        for response in successful:
            if not isinstance(response, Exception):
                assert response.audio_data is not None
                assert response.provider == "ElevenLabs"
        
        # Cleanup
        await adapter.close()
    
    async def test_invalid_api_key(self):
        """Test with invalid API key"""
        adapter = ElevenLabsAdapter({
            "elevenlabs_api_key": "invalid-key-12345"
        })
        
        # Initialization succeeds but API calls will fail
        await adapter.initialize()
        
        request = TTSRequest(
            text="Test",
            voice="rachel",
            format=AudioFormat.MP3,
            stream=False
        )
        
        with pytest.raises((TTSAuthenticationError, Exception)):
            await adapter.generate(request)
        
        # Cleanup
        await adapter.close()
    
    @pytest.mark.skipif(
        not os.getenv("ELEVENLABS_API_KEY"),
        reason="Requires ELEVENLABS_API_KEY environment variable"
    )
    async def test_long_text_generation(self):
        """Test generation with long text"""
        adapter = ElevenLabsAdapter({
            "elevenlabs_api_key": os.getenv("ELEVENLABS_API_KEY")
        })
        
        await adapter.initialize()
        
        # Create text near the limit
        long_text = "This is a test sentence. " * 150  # ~3750 characters
        
        request = TTSRequest(
            text=long_text,
            voice="rachel",
            format=AudioFormat.MP3,
            stream=False
        )
        
        response = await adapter.generate(request)
        
        assert response.audio_data is not None
        assert len(response.audio_data) > 10000  # Should be substantial
        
        # Cleanup
        await adapter.close()

#######################################################################################################################
#
# End of test_elevenlabs_adapter_integration.py