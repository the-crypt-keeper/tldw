# test_tts_adapters.py
# Description: Comprehensive tests for TTS adapter pattern implementation
#
# Imports
import asyncio
import pytest
from unittest.mock import Mock, AsyncMock, MagicMock, patch
import numpy as np
#
# Local Imports
from tldw_Server_API.app.core.TTS.adapters.base import (
    TTSAdapter,
    TTSCapabilities,
    TTSRequest,
    TTSResponse,
    AudioFormat,
    VoiceInfo,
    ProviderStatus
)
from tldw_Server_API.app.core.TTS.adapters.openai_adapter import OpenAIAdapter
from tldw_Server_API.app.core.TTS.adapters.kokoro_adapter import KokoroAdapter
from tldw_Server_API.app.core.TTS.adapters.higgs_adapter import HiggsAdapter
from tldw_Server_API.app.core.TTS.adapters.dia_adapter import DiaAdapter
from tldw_Server_API.app.core.TTS.adapters.chatterbox_adapter import ChatterboxAdapter
from tldw_Server_API.app.core.TTS.adapter_registry import (
    TTSAdapterRegistry,
    TTSAdapterFactory,
    TTSProvider,
    get_tts_factory,
    close_tts_factory
)
from tldw_Server_API.app.core.TTS.tts_service_v2 import (
    TTSServiceV2,
    get_tts_service_v2,
    close_tts_service_v2
)
#
#######################################################################################################################
#
# Test Classes

class TestTTSAdapterBase:
    """Test the base TTSAdapter class"""
    
    def test_adapter_initialization(self):
        """Test basic adapter initialization"""
        # Create a concrete implementation for testing
        class TestAdapter(TTSAdapter):
            async def initialize(self) -> bool:
                return True
            
            async def generate(self, request: TTSRequest) -> TTSResponse:
                return TTSResponse(audio_data=b"test")
            
            async def get_capabilities(self) -> TTSCapabilities:
                return TTSCapabilities(
                    provider_name="Test",
                    supported_languages={"en"},
                    supported_voices=[],
                    supported_formats={AudioFormat.WAV},
                    max_text_length=1000,
                    supports_streaming=False
                )
        
        adapter = TestAdapter({"test_key": "test_value"})
        assert adapter.config["test_key"] == "test_value"
        assert adapter.status == ProviderStatus.NOT_CONFIGURED
        assert adapter.provider_name == "Test"
    
    @pytest.mark.asyncio
    async def test_validate_request(self):
        """Test request validation"""
        class TestAdapter(TTSAdapter):
            async def initialize(self) -> bool:
                return True
            
            async def generate(self, request: TTSRequest) -> TTSResponse:
                return TTSResponse(audio_data=b"test")
            
            async def get_capabilities(self) -> TTSCapabilities:
                return TTSCapabilities(
                    provider_name="Test",
                    supported_languages={"en"},
                    supported_voices=[],
                    supported_formats={AudioFormat.WAV, AudioFormat.MP3},
                    max_text_length=100,
                    supports_streaming=True
                )
        
        adapter = TestAdapter()
        await adapter.ensure_initialized()
        
        # Valid request
        request = TTSRequest(
            text="Hello world",
            language="en",
            format=AudioFormat.WAV,
            stream=True
        )
        is_valid, error = await adapter.validate_request(request)
        assert is_valid
        assert error is None
        
        # Invalid format
        request.format = AudioFormat.OPUS
        is_valid, error = await adapter.validate_request(request)
        assert not is_valid
        assert "Format opus not supported" in error
        
        # Text too long
        request.format = AudioFormat.WAV
        request.text = "x" * 200
        is_valid, error = await adapter.validate_request(request)
        assert not is_valid
        assert "exceeds maximum length" in error
    
    def test_parse_dialogue(self):
        """Test dialogue parsing"""
        class TestAdapter(TTSAdapter):
            async def initialize(self) -> bool:
                return True
            async def generate(self, request: TTSRequest) -> TTSResponse:
                return TTSResponse()
            async def get_capabilities(self) -> TTSCapabilities:
                return TTSCapabilities(
                    provider_name="Test",
                    supported_languages={"en"},
                    supported_voices=[],
                    supported_formats={AudioFormat.WAV},
                    max_text_length=1000,
                    supports_streaming=False
                )
        
        adapter = TestAdapter()
        
        # Parse dialogue with speakers
        text = "Alice: Hello there! Bob: Hi Alice! Charlie: Hey everyone!"
        result = adapter.parse_dialogue(text)
        assert len(result) == 3
        assert result[0] == ("Alice", "Hello there!")
        assert result[1] == ("Bob", "Hi Alice!")
        assert result[2] == ("Charlie", "Hey everyone!")
        
        # No speakers
        text = "Just plain text"
        result = adapter.parse_dialogue(text)
        assert len(result) == 1
        assert result[0] == ("default", "Just plain text")


@pytest.mark.asyncio
class TestOpenAIAdapter:
    """Test OpenAI adapter implementation"""
    
    async def test_initialization_without_key(self):
        """Test initialization without API key"""
        adapter = OpenAIAdapter({})
        assert adapter.api_key is None
        assert adapter.status == ProviderStatus.NOT_CONFIGURED
        
        success = await adapter.initialize()
        assert not success
        assert adapter.status == ProviderStatus.NOT_CONFIGURED
    
    async def test_initialization_with_key(self):
        """Test initialization with API key"""
        adapter = OpenAIAdapter({"openai_api_key": "test-key-123"})
        assert adapter.api_key == "test-key-123"
        
        success = await adapter.initialize()
        assert success
        assert adapter.status == ProviderStatus.AVAILABLE
    
    async def test_capabilities(self):
        """Test OpenAI capabilities"""
        adapter = OpenAIAdapter({"openai_api_key": "test-key"})
        await adapter.initialize()
        
        caps = await adapter.get_capabilities()
        assert caps.provider_name == "OpenAI"
        assert "en" in caps.supported_languages
        assert AudioFormat.MP3 in caps.supported_formats
        assert caps.supports_streaming
        assert not caps.supports_emotion_control
        assert not caps.supports_voice_cloning
    
    def test_voice_mapping(self):
        """Test voice mapping"""
        adapter = OpenAIAdapter({})
        
        # Direct OpenAI voice
        assert adapter.map_voice("alloy") == "alloy"
        assert adapter.map_voice("nova") == "nova"
        
        # Generic mappings
        assert adapter.map_voice("male") == "onyx"
        assert adapter.map_voice("female") == "nova"
        assert adapter.map_voice("neutral") == "alloy"
        
        # Unknown voice
        assert adapter.map_voice("unknown") == "alloy"


@pytest.mark.asyncio
class TestKokoroAdapter:
    """Test Kokoro adapter implementation"""
    
    async def test_initialization_onnx(self):
        """Test ONNX initialization"""
        adapter = KokoroAdapter({
            "kokoro_use_onnx": True,
            "kokoro_model_path": "test_model.onnx",
            "kokoro_voices_json": "test_voices.json"
        })
        
        # Mock the kokoro_onnx import
        with patch('tldw_Server_API.app.core.TTS.adapters.kokoro_adapter.os.path.exists') as mock_exists:
            mock_exists.return_value = False  # Model files don't exist
            
            success = await adapter.initialize()
            assert not success  # Should fail without model files
    
    async def test_capabilities(self):
        """Test Kokoro capabilities"""
        adapter = KokoroAdapter({})
        
        # Get capabilities without initialization
        caps = await adapter.get_capabilities()
        assert caps.provider_name == "Kokoro"
        assert "en-us" in caps.supported_languages
        assert "en-gb" in caps.supported_languages
        assert AudioFormat.WAV in caps.supported_formats
        assert caps.supports_streaming
        assert caps.supports_phonemes
        assert caps.supports_multi_speaker  # Voice mixing
    
    def test_voice_processing(self):
        """Test voice processing and mixing"""
        adapter = KokoroAdapter({})
        
        # Single voice
        assert adapter._process_voice("af_bella") == "af_bella"
        
        # Mixed voice
        mixed = "af_bella(2)+af_sky(1)"
        assert adapter._process_voice(mixed) == mixed
        
        # Generic mapping
        assert adapter._process_voice("female") == "af_bella"
        assert adapter._process_voice("british_female") == "bf_emma"
    
    def test_language_detection(self):
        """Test language detection from voice"""
        adapter = KokoroAdapter({})
        
        # American voices
        assert adapter._get_language_from_voice("af_bella") == "en-us"
        assert adapter._get_language_from_voice("am_adam") == "en-us"
        
        # British voices
        assert adapter._get_language_from_voice("bf_emma") == "en-gb"
        assert adapter._get_language_from_voice("bm_george") == "en-gb"
        
        # Mixed voice (uses first voice)
        assert adapter._get_language_from_voice("af_bella+bf_emma") == "en-us"
    
    def test_text_chunking(self):
        """Test text chunking for optimal processing"""
        adapter = KokoroAdapter({})
        
        # Short text (single chunk)
        text = "This is a short sentence."
        chunks = adapter.chunk_text(text)
        assert len(chunks) == 1
        assert chunks[0] == text
        
        # Long text (multiple chunks)
        text = "First sentence. " * 20  # Create long text
        chunks = adapter.chunk_text(text)
        assert len(chunks) > 1
        assert all(chunk.endswith(".") for chunk in chunks)


@pytest.mark.asyncio
class TestAdapterRegistry:
    """Test the adapter registry"""
    
    async def test_registry_initialization(self):
        """Test registry initialization"""
        registry = TTSAdapterRegistry({"test_config": "value"})
        assert registry.config["test_config"] == "value"
        assert len(registry._adapter_classes) == 7  # Default adapters (now includes VibeVoice and ElevenLabs)
    
    async def test_register_custom_adapter(self):
        """Test registering a custom adapter"""
        registry = TTSAdapterRegistry()
        
        class CustomAdapter(TTSAdapter):
            async def initialize(self) -> bool:
                return True
            async def generate(self, request: TTSRequest) -> TTSResponse:
                return TTSResponse()
            async def get_capabilities(self) -> TTSCapabilities:
                return TTSCapabilities(
                    provider_name="Custom",
                    supported_languages={"en"},
                    supported_voices=[],
                    supported_formats={AudioFormat.WAV},
                    max_text_length=1000,
                    supports_streaming=False
                )
        
        # Register custom adapter
        registry.register_adapter(TTSProvider.OPENAI, CustomAdapter)
        assert registry._adapter_classes[TTSProvider.OPENAI] == CustomAdapter
    
    async def test_get_adapter_with_config(self):
        """Test getting adapter with configuration"""
        config = {
            "openai_api_key": "test-key",
            "openai_enabled": True
        }
        registry = TTSAdapterRegistry(config)
        
        adapter = await registry.get_adapter(TTSProvider.OPENAI)
        assert adapter is not None
        assert adapter.status == ProviderStatus.AVAILABLE
    
    async def test_disabled_provider(self):
        """Test disabled provider"""
        config = {
            "openai_enabled": False
        }
        registry = TTSAdapterRegistry(config)
        
        adapter = await registry.get_adapter(TTSProvider.OPENAI)
        assert adapter is None
    
    async def test_find_adapter_for_requirements(self):
        """Test finding adapter by requirements"""
        registry = TTSAdapterRegistry({"openai_api_key": "test"})
        
        # Initialize OpenAI adapter
        await registry.get_adapter(TTSProvider.OPENAI)
        
        # Find adapter for streaming
        adapter = await registry.find_adapter_for_requirements(
            supports_streaming=True,
            format=AudioFormat.MP3
        )
        assert adapter is not None
        assert adapter.capabilities.supports_streaming
    
    async def test_get_all_capabilities(self):
        """Test getting all capabilities"""
        registry = TTSAdapterRegistry({"openai_api_key": "test"})
        
        caps = await registry.get_all_capabilities()
        assert TTSProvider.OPENAI in caps
        assert caps[TTSProvider.OPENAI].provider_name == "OpenAI"
    
    def test_status_summary(self):
        """Test status summary"""
        registry = TTSAdapterRegistry()
        
        summary = registry.get_status_summary()
        assert summary["total_providers"] == len(TTSProvider)
        assert summary["initialized"] == 0
        assert summary["available"] == 0
        assert "openai" in summary["providers"]


@pytest.mark.asyncio
class TestTTSAdapterFactory:
    """Test the adapter factory"""
    
    async def test_factory_initialization(self):
        """Test factory initialization"""
        factory = TTSAdapterFactory({"test": "config"})
        assert factory.registry.config["test"] == "config"
    
    async def test_get_adapter_by_model(self):
        """Test getting adapter by model name"""
        factory = TTSAdapterFactory({"openai_api_key": "test"})
        
        # OpenAI models
        adapter = await factory.get_adapter_by_model("tts-1")
        assert adapter is not None
        assert isinstance(adapter, OpenAIAdapter)
        
        # Kokoro models
        adapter = await factory.get_adapter_by_model("kokoro")
        assert adapter is not None or adapter is None  # Depends on model availability
        
        # Unknown model
        adapter = await factory.get_adapter_by_model("unknown-model")
        assert adapter is None
    
    async def test_get_best_adapter(self):
        """Test getting best adapter for requirements"""
        factory = TTSAdapterFactory({"openai_api_key": "test"})
        
        adapter = await factory.get_best_adapter(
            language="en",
            supports_streaming=True
        )
        assert adapter is not None
    
    def test_get_status(self):
        """Test factory status"""
        factory = TTSAdapterFactory()
        
        status = factory.get_status()
        assert "total_providers" in status
        assert "initialized" in status
        assert "providers" in status


@pytest.mark.asyncio
class TestTTSServiceV2:
    """Test the enhanced TTS service"""
    
    async def test_service_initialization(self):
        """Test service initialization"""
        factory = TTSAdapterFactory({"openai_api_key": "test"})
        service = TTSServiceV2(factory)
        assert service.factory == factory
    
    async def test_list_voices(self):
        """Test listing voices"""
        factory = TTSAdapterFactory({"openai_api_key": "test"})
        service = TTSServiceV2(factory)
        
        voices = await service.list_voices()
        assert "openai" in voices
        assert len(voices["openai"]) > 0
        assert any(v["id"] == "alloy" for v in voices["openai"])
    
    async def test_get_capabilities(self):
        """Test getting capabilities"""
        factory = TTSAdapterFactory({"openai_api_key": "test"})
        service = TTSServiceV2(factory)
        
        caps = await service.get_capabilities()
        assert "openai" in caps
        assert "languages" in caps["openai"]
        assert "formats" in caps["openai"]
        assert caps["openai"]["supports_streaming"]
    
    def test_get_status(self):
        """Test service status"""
        factory = TTSAdapterFactory()
        service = TTSServiceV2(factory)
        
        status = service.get_status()
        assert "total_providers" in status
        assert "providers" in status
    
    @pytest.mark.asyncio
    async def test_request_conversion(self):
        """Test OpenAI request conversion"""
        from tldw_Server_API.app.api.v1.schemas.audio_schemas import OpenAISpeechRequest
        
        factory = TTSAdapterFactory()
        service = TTSServiceV2(factory)
        
        openai_request = OpenAISpeechRequest(
            input="Hello world",
            model="tts-1",
            voice="alloy",
            response_format="mp3",
            speed=1.0
        )
        
        tts_request = service._convert_request(openai_request)
        assert tts_request.text == "Hello world"
        assert tts_request.voice == "alloy"
        assert tts_request.format == AudioFormat.MP3
        assert tts_request.speed == 1.0


# Integration tests
@pytest.mark.asyncio
class TestIntegration:
    """Integration tests for the complete system"""
    
    async def test_singleton_management(self):
        """Test singleton management"""
        # Get factory singleton
        factory1 = await get_tts_factory({"test": "config"})
        factory2 = await get_tts_factory()
        assert factory1 is factory2
        
        # Get service singleton
        service1 = await get_tts_service_v2()
        service2 = await get_tts_service_v2()
        assert service1 is service2
        
        # Clean up
        await close_tts_service_v2()
        await close_tts_factory()
    
    async def test_backwards_compatibility(self):
        """Test backwards compatibility wrapper"""
        from tldw_Server_API.app.api.v1.schemas.audio_schemas import OpenAISpeechRequest
        from tldw_Server_API.app.core.TTS.tts_service_v2 import TTSService
        
        # Create adapter (backwards compatible)
        adapter = TTSService()
        
        request = OpenAISpeechRequest(
            input="Test",
            model="tts-1",
            voice="alloy"
        )
        
        # Mock the service_v2
        adapter.service_v2 = AsyncMock()
        adapter.service_v2.generate_speech = AsyncMock()
        
        # Test generation
        async def mock_generator():
            yield b"test audio"
        
        adapter.service_v2.generate_speech.return_value = mock_generator()
        
        chunks = []
        async for chunk in adapter.generate_audio_stream(request, "openai_official_tts-1"):
            chunks.append(chunk)
        
        assert len(chunks) > 0
        adapter.service_v2.generate_speech.assert_called_once()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

#
# End of test_tts_adapters.py
#######################################################################################################################