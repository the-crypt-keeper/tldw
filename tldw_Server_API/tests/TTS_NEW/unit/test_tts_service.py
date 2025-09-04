"""
Unit tests for the TTSServiceV2 core functionality.

Tests the main service logic, adapter selection, and request processing
with mocked dependencies.
"""

import pytest
from unittest.mock import Mock, AsyncMock, patch, MagicMock
import asyncio

from tldw_Server_API.app.core.TTS.tts_service_v2 import TTSServiceV2
from tldw_Server_API.app.core.TTS.adapters.base import (
    TTSRequest,
    TTSResponse,
    AudioFormat
)
from tldw_Server_API.app.core.TTS.tts_exceptions import (
    TTSProviderNotConfiguredError,
    TTSGenerationError,
    TTSRateLimitError
)

# ========================================================================
# Service Initialization Tests
# ========================================================================

class TestServiceInitialization:
    """Test TTS service initialization and setup."""
    
    @pytest.mark.unit
    async def test_service_initialization(self):
        """Test basic service initialization."""
        service = TTSServiceV2()
        
        assert service is not None
        assert hasattr(service, 'generate')
        assert hasattr(service, 'generate_stream')
        
        await service.shutdown()
    
    @pytest.mark.unit
    @patch('tldw_Server_API.app.core.TTS.tts_service_v2.get_tts_factory')
    async def test_service_with_mock_factory(self, mock_factory_getter, mock_adapter_factory):
        """Test service initialization with mocked factory."""
        mock_factory_getter.return_value = mock_adapter_factory
        
        service = TTSServiceV2()
        
        assert service._factory == mock_adapter_factory
        
        await service.shutdown()
    
    @pytest.mark.unit
    async def test_service_shutdown(self, tts_service):
        """Test service shutdown process."""
        # Service from fixture
        await tts_service.shutdown()
        
        # Should handle multiple shutdowns gracefully
        await tts_service.shutdown()

# ========================================================================
# Text Generation Tests
# ========================================================================

class TestTextGeneration:
    """Test text-to-speech generation."""
    
    @pytest.mark.unit
    async def test_basic_generation(self, tts_service, basic_tts_request):
        """Test basic TTS generation."""
        result = await tts_service.generate(basic_tts_request)
        
        assert result is not None
        assert hasattr(result, 'audio_content')
        assert result.audio_content == b"mock_audio"
    
    @pytest.mark.unit
    async def test_generation_with_provider_selection(self, tts_service, basic_tts_request):
        """Test generation with specific provider."""
        basic_tts_request.provider = "elevenlabs"
        
        result = await tts_service.generate(basic_tts_request)
        
        assert result is not None
        tts_service._factory.get_adapter.assert_called_with("elevenlabs")
    
    @pytest.mark.unit
    async def test_generation_with_invalid_provider(self, tts_service, basic_tts_request):
        """Test generation with invalid provider."""
        basic_tts_request.provider = "invalid_provider"
        tts_service._factory.get_adapter.side_effect = TTSProviderNotConfiguredError("Provider not found")
        
        with pytest.raises(TTSProviderNotConfiguredError):
            await tts_service.generate(basic_tts_request)
    
    @pytest.mark.unit
    async def test_generation_with_long_text(self, tts_service, long_text_request):
        """Test generation with long text that needs chunking."""
        result = await tts_service.generate(long_text_request)
        
        assert result is not None
        # Should handle long text appropriately
    
    @pytest.mark.unit
    @patch('tldw_Server_API.app.core.TTS.tts_service_v2.validate_tts_request')
    async def test_request_validation_called(self, mock_validate, tts_service, basic_tts_request):
        """Test that request validation is called."""
        mock_validate.return_value = None  # Validation passes
        
        await tts_service.generate(basic_tts_request)
        
        mock_validate.assert_called_once()

# ========================================================================
# Streaming Generation Tests
# ========================================================================

class TestStreamingGeneration:
    """Test streaming TTS generation."""
    
    @pytest.mark.unit
    async def test_streaming_generation(self, tts_service, basic_tts_request):
        """Test streaming TTS generation."""
        # Mock adapter to return streaming response
        adapter = tts_service._factory.get_adapter("openai")
        
        async def mock_stream():
            for chunk in [b"chunk1", b"chunk2", b"chunk3"]:
                yield chunk
        
        adapter.generate_stream = AsyncMock(return_value=mock_stream())
        
        chunks = []
        async for chunk in tts_service.generate_stream(basic_tts_request):
            chunks.append(chunk)
        
        assert len(chunks) == 3
        assert chunks == [b"chunk1", b"chunk2", b"chunk3"]
    
    @pytest.mark.unit
    async def test_streaming_with_error(self, tts_service, basic_tts_request):
        """Test streaming generation error handling."""
        adapter = tts_service._factory.get_adapter("openai")
        
        async def mock_error_stream():
            yield b"chunk1"
            raise TTSGenerationError("Stream interrupted")
        
        adapter.generate_stream = AsyncMock(return_value=mock_error_stream())
        
        chunks = []
        with pytest.raises(TTSGenerationError):
            async for chunk in tts_service.generate_stream(basic_tts_request):
                chunks.append(chunk)
        
        # Should have received first chunk before error
        assert len(chunks) == 1

# ========================================================================
# Provider Management Tests
# ========================================================================

class TestProviderManagement:
    """Test provider management functionality."""
    
    @pytest.mark.unit
    async def test_list_providers(self, tts_service):
        """Test listing available providers."""
        providers = await tts_service.list_providers()
        
        assert isinstance(providers, list)
        assert "openai" in providers
        assert "elevenlabs" in providers
    
    @pytest.mark.unit
    async def test_get_provider_info(self, tts_service):
        """Test getting provider information."""
        # Mock the adapter
        adapter = tts_service._factory.get_adapter("openai")
        adapter.get_info = Mock(return_value={
            "name": "openai",
            "models": ["tts-1", "tts-1-hd"],
            "voices": ["alloy", "echo"]
        })
        
        info = await tts_service.get_provider_info("openai")
        
        assert info["name"] == "openai"
        assert "tts-1" in info["models"]
        assert "alloy" in info["voices"]
    
    @pytest.mark.unit
    async def test_switch_default_provider(self, tts_service):
        """Test switching the default provider."""
        await tts_service.set_default_provider("elevenlabs")
        
        # Generate without specifying provider should use new default
        basic_request = TTSRequest(text="Test", voice="rachel")
        await tts_service.generate(basic_request)
        
        tts_service._factory.get_adapter.assert_called_with("elevenlabs")

# ========================================================================
# Error Handling Tests
# ========================================================================

class TestErrorHandling:
    """Test error handling in the service."""
    
    @pytest.mark.unit
    async def test_handle_rate_limit_error(self, tts_service, basic_tts_request):
        """Test handling of rate limit errors."""
        adapter = tts_service._factory.get_adapter("openai")
        adapter.generate = AsyncMock(side_effect=TTSRateLimitError("Rate limited", retry_after=60))
        
        with pytest.raises(TTSRateLimitError) as exc_info:
            await tts_service.generate(basic_tts_request)
        
        assert exc_info.value.retry_after == 60
    
    @pytest.mark.unit
    async def test_handle_generation_error(self, tts_service, basic_tts_request):
        """Test handling of generation errors."""
        adapter = tts_service._factory.get_adapter("openai")
        adapter.generate = AsyncMock(side_effect=TTSGenerationError("Generation failed"))
        
        with pytest.raises(TTSGenerationError):
            await tts_service.generate(basic_tts_request)
    
    @pytest.mark.unit
    async def test_fallback_provider_on_error(self, tts_service, basic_tts_request):
        """Test fallback to another provider on error."""
        # First adapter fails
        openai_adapter = tts_service._factory.get_adapter("openai")
        openai_adapter.generate = AsyncMock(side_effect=TTSGenerationError("Failed"))
        
        # Second adapter succeeds
        elevenlabs_adapter = tts_service._factory.get_adapter("elevenlabs")
        elevenlabs_adapter.generate = AsyncMock(return_value=MagicMock(audio_content=b"fallback_audio"))
        
        # Enable fallback
        tts_service.enable_fallback = True
        
        result = await tts_service.generate_with_fallback(basic_tts_request, fallback_providers=["elevenlabs"])
        
        assert result.audio_content == b"fallback_audio"

# ========================================================================
# Resource Management Tests
# ========================================================================

class TestResourceManagement:
    """Test resource management in the service."""
    
    @pytest.mark.unit
    @patch('tldw_Server_API.app.core.TTS.tts_service_v2.get_resource_manager')
    async def test_resource_check_before_generation(self, mock_resource_manager, tts_service, basic_tts_request):
        """Test that resources are checked before generation."""
        manager = MagicMock()
        manager.check_resources = AsyncMock(return_value=True)
        mock_resource_manager.return_value = manager
        
        await tts_service.generate(basic_tts_request)
        
        manager.check_resources.assert_called()
    
    @pytest.mark.unit
    @patch('tldw_Server_API.app.core.TTS.tts_service_v2.get_resource_manager')
    async def test_insufficient_resources(self, mock_resource_manager, tts_service, basic_tts_request):
        """Test handling of insufficient resources."""
        manager = MagicMock()
        manager.check_resources = AsyncMock(return_value=False)
        mock_resource_manager.return_value = manager
        
        from tldw_Server_API.app.core.TTS.tts_exceptions import TTSResourceError
        
        with pytest.raises(TTSResourceError):
            await tts_service.generate(basic_tts_request)

# ========================================================================
# Metrics Collection Tests
# ========================================================================

class TestMetricsCollection:
    """Test metrics collection in the service."""
    
    @pytest.mark.unit
    @patch('tldw_Server_API.app.core.TTS.tts_service_v2.get_metrics_registry')
    async def test_metrics_recorded_on_success(self, mock_metrics, tts_service, basic_tts_request):
        """Test that metrics are recorded on successful generation."""
        metrics_registry = MagicMock()
        mock_metrics.return_value = metrics_registry
        
        await tts_service.generate(basic_tts_request)
        
        # Should record success metrics
        metrics_registry.record.assert_called()
    
    @pytest.mark.unit
    @patch('tldw_Server_API.app.core.TTS.tts_service_v2.get_metrics_registry')
    async def test_metrics_recorded_on_failure(self, mock_metrics, tts_service, basic_tts_request):
        """Test that metrics are recorded on failed generation."""
        metrics_registry = MagicMock()
        mock_metrics.return_value = metrics_registry
        
        adapter = tts_service._factory.get_adapter("openai")
        adapter.generate = AsyncMock(side_effect=TTSGenerationError("Failed"))
        
        with pytest.raises(TTSGenerationError):
            await tts_service.generate(basic_tts_request)
        
        # Should record failure metrics
        metrics_registry.record.assert_called()

# ========================================================================
# Caching Tests
# ========================================================================

class TestCaching:
    """Test caching functionality in the service."""
    
    @pytest.mark.unit
    async def test_cache_hit(self, tts_service, basic_tts_request):
        """Test cache hit for repeated requests."""
        # First request
        result1 = await tts_service.generate(basic_tts_request)
        
        # Second identical request should hit cache
        result2 = await tts_service.generate(basic_tts_request)
        
        # Adapter should only be called once if caching works
        adapter = tts_service._factory.get_adapter("openai")
        assert adapter.generate.call_count == 1
    
    @pytest.mark.unit
    async def test_cache_miss_different_params(self, tts_service):
        """Test cache miss with different parameters."""
        request1 = TTSRequest(text="Test 1", voice="alloy")
        request2 = TTSRequest(text="Test 2", voice="alloy")
        
        await tts_service.generate(request1)
        await tts_service.generate(request2)
        
        # Should call adapter twice for different texts
        adapter = tts_service._factory.get_adapter("openai")
        assert adapter.generate.call_count == 2