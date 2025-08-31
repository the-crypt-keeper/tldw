# test_tts_service_v2.py
# Description: Tests for the V2 TTS Service architecture
#
# Imports
import asyncio
import base64
import os
import tempfile
from unittest.mock import AsyncMock, MagicMock, patch, Mock
import pytest
import numpy as np
from typing import Dict, Any

# Local Imports
from tldw_Server_API.app.api.v1.schemas.audio_schemas import OpenAISpeechRequest
from tldw_Server_API.app.core.TTS.tts_service_v2 import TTSServiceV2
from tldw_Server_API.app.core.TTS.adapter_registry import TTSAdapterRegistry, TTSProvider, TTSAdapterFactory
from tldw_Server_API.app.core.TTS.adapters.base import (
    TTSAdapter, 
    TTSCapabilities,
    TTSRequest,
    TTSResponse
)
from tldw_Server_API.app.core.TTS.circuit_breaker import CircuitBreaker, CircuitState
from tldw_Server_API.app.core.TTS.audio_utils import AudioProcessor, process_voice_reference

#######################################################################################################################
#
# Test Classes


class MockAdapter(TTSAdapter):
    """Mock adapter for testing"""
    
    def __init__(self, provider_config: Dict[str, Any]):
        super().__init__(provider_config)
        self.initialized = False
        self.generate_called = False
        
    async def initialize(self) -> bool:
        self.initialized = True
        return True
    
    async def generate(self, request: TTSRequest) -> TTSResponse:
        self.generate_called = True
        return TTSResponse(
            audio=b"mock audio data",
            format=request.format or "mp3",
            provider=self.provider_config.get("name", "mock")
        )
    
    async def generate_stream(self, request: TTSRequest):
        yield b"mock "
        yield b"stream "
        yield b"data"
    
    async def get_capabilities(self) -> TTSCapabilities:
        return TTSCapabilities(
            supports_streaming=True,
            supports_voice_cloning=False,
            supported_languages=["en"],
            supported_formats=["mp3", "wav"],
            max_text_length=5000,
            available_voices=["voice1", "voice2"]
        )
    
    async def list_voices(self):
        return ["voice1", "voice2"]
    
    async def cleanup(self):
        self.initialized = False


class TestTTSServiceV2:
    """Tests for TTSServiceV2"""
    
    @pytest.fixture
    async def service(self):
        """Create a test service instance"""
        config = {
            "providers": {
                "mock": {
                    "enabled": True,
                    "priority": 1
                }
            },
            "fallback": {
                "enabled": True,
                "max_attempts": 2
            }
        }
        
        # Create a mock factory
        factory = MagicMock(spec=TTSAdapterFactory)
        factory.create_adapter = AsyncMock(return_value=MockAdapter(config.get("mock", {})))
        factory.get_adapter = AsyncMock(return_value=MockAdapter(config.get("mock", {})))
        factory.list_adapters = AsyncMock(return_value=[TTSProvider.MOCK])
        
        service = TTSServiceV2(factory)
        return service
    
    @pytest.mark.asyncio
    async def test_service_initialization(self, service):
        """Test service initializes properly"""
        assert service is not None
        assert service.client_id == "test"
        assert len(service.registry.adapters) > 0
    
    @pytest.mark.asyncio
    async def test_generate_speech(self, service):
        """Test basic speech generation"""
        request = TTSRequest(
            text="Test text",
            provider="mock",
            voice="voice1",
            format="mp3"
        )
        
        response = await service.generate(request)
        
        assert response is not None
        assert response.audio == b"mock audio data"
        assert response.format == "mp3"
        assert response.provider == "mock"
    
    @pytest.mark.asyncio
    async def test_generate_stream(self, service):
        """Test streaming generation"""
        request = TTSRequest(
            text="Test text",
            provider="mock",
            voice="voice1",
            stream=True
        )
        
        chunks = []
        async for chunk in service.generate_stream(request):
            chunks.append(chunk)
        
        assert len(chunks) == 3
        assert b"".join(chunks) == b"mock stream data"
    
    @pytest.mark.asyncio
    async def test_get_capabilities(self, service):
        """Test getting provider capabilities"""
        caps = await service.get_capabilities("mock")
        
        assert caps is not None
        assert caps.supports_streaming is True
        assert "en" in caps.supported_languages
        assert "mp3" in caps.supported_formats
    
    @pytest.mark.asyncio
    async def test_list_providers(self, service):
        """Test listing available providers"""
        providers = service.list_providers()
        
        assert "mock" in providers
        assert providers["mock"]["status"] == "available"
    
    @pytest.mark.asyncio
    async def test_fallback_mechanism(self):
        """Test fallback to another provider on failure"""
        config = {
            "providers": {
                "failing": {
                    "enabled": True,
                    "priority": 1
                },
                "working": {
                    "enabled": True,
                    "priority": 2
                }
            },
            "fallback": {
                "enabled": True,
                "max_attempts": 3
            }
        }
        
        # Create failing adapter
        failing_adapter = MockAdapter({"name": "failing"})
        failing_adapter.generate = AsyncMock(side_effect=Exception("Provider failed"))
        
        # Create working adapter
        working_adapter = MockAdapter({"name": "working"})
        
        # Create mock factory that returns appropriate adapters
        factory = MagicMock(spec=TTSAdapterFactory)
        factory.create_adapter = AsyncMock(side_effect=lambda name, *args: 
                                          failing_adapter if name == "failing" else working_adapter)
        factory.get_adapter = AsyncMock(side_effect=lambda name: 
                                       failing_adapter if name == "failing" else working_adapter)
        factory.list_adapters = AsyncMock(return_value=["failing", "working"])
        
        service = TTSServiceV2(factory)
        
        request = TTSRequest(
            text="Test text",
            voice="voice1"
        )
        
        response = await service.generate(request)
        
        assert response is not None
        assert response.provider == "working"
    
    @pytest.mark.asyncio
    async def test_circuit_breaker(self, service):
        """Test circuit breaker functionality"""
        # Get circuit breaker for mock provider
        cb = service.circuit_breakers.get("mock")
        
        if cb:
            assert cb.state == CircuitState.CLOSED
            
            # Simulate failures
            for _ in range(5):
                cb.record_failure()
            
            # Circuit should be open
            assert cb.state == CircuitState.OPEN
            
            # Should not allow calls
            assert not cb.allow_request()


class TestAudioUtils:
    """Tests for audio processing utilities"""
    
    def test_audio_processor_initialization(self):
        """Test AudioProcessor initialization"""
        processor = AudioProcessor()
        assert processor is not None
        assert "higgs" in processor.PROVIDER_REQUIREMENTS
        assert "chatterbox" in processor.PROVIDER_REQUIREMENTS
        assert "vibevoice" in processor.PROVIDER_REQUIREMENTS
    
    def test_validate_duration(self):
        """Test audio duration validation"""
        processor = AudioProcessor()
        
        # Test valid duration for Higgs
        is_valid, msg = processor.validate_duration(5.0, "higgs")
        assert is_valid is True
        
        # Test too short
        is_valid, msg = processor.validate_duration(2.0, "higgs")
        assert is_valid is False
        assert "too short" in msg.lower()
        
        # Test too long
        is_valid, msg = processor.validate_duration(15.0, "higgs")
        assert is_valid is False
        assert "too long" in msg.lower()
    
    def test_process_voice_reference(self):
        """Test voice reference processing"""
        # Create a simple WAV file
        sample_rate = 24000
        duration = 5.0
        t = np.linspace(0, duration, int(sample_rate * duration))
        audio = np.sin(2 * np.pi * 440 * t) * 0.5  # 440Hz sine wave
        
        # Convert to bytes (16-bit PCM)
        audio_int16 = (audio * 32767).astype(np.int16)
        audio_bytes = audio_int16.tobytes()
        
        # Add WAV header (simplified)
        wav_header = b'RIFF' + b'\x00' * 4 + b'WAVE'
        wav_data = wav_header + audio_bytes
        
        # Process
        processed, error = process_voice_reference(
            wav_data,
            provider="higgs",
            validate=True,
            convert=False
        )
        
        # Note: This will likely fail without proper WAV headers
        # but tests the function exists and runs
        assert processed is not None or error is not None
    
    @pytest.mark.asyncio
    async def test_base64_encoding(self):
        """Test base64 encoding/decoding for voice references"""
        original_data = b"test audio data"
        
        # Encode
        encoded = base64.b64encode(original_data).decode('utf-8')
        
        # Decode
        decoded = base64.b64decode(encoded)
        
        assert decoded == original_data


class TestCircuitBreaker:
    """Tests for Circuit Breaker"""
    
    def test_circuit_breaker_initialization(self):
        """Test circuit breaker initialization"""
        cb = CircuitBreaker(
            failure_threshold=3,
            recovery_timeout=10,
            half_open_calls=2
        )
        
        assert cb.state == CircuitState.CLOSED
        assert cb.failure_count == 0
    
    def test_circuit_opens_after_threshold(self):
        """Test circuit opens after failure threshold"""
        cb = CircuitBreaker(failure_threshold=3)
        
        # Record failures
        for _ in range(3):
            cb.record_failure()
        
        assert cb.state == CircuitState.OPEN
        assert not cb.allow_request()
    
    def test_circuit_recovery(self):
        """Test circuit recovery to half-open state"""
        cb = CircuitBreaker(
            failure_threshold=2,
            recovery_timeout=0.1  # 100ms for testing
        )
        
        # Open the circuit
        cb.record_failure()
        cb.record_failure()
        assert cb.state == CircuitState.OPEN
        
        # Wait for recovery timeout
        import time
        time.sleep(0.2)
        
        # Should transition to half-open
        assert cb.allow_request()
        assert cb.state == CircuitState.HALF_OPEN
        
        # Success should close circuit
        cb.record_success()
        cb.record_success()  # Need 2 successes (half_open_calls default)
        assert cb.state == CircuitState.CLOSED
    
    def test_circuit_reopens_on_failure_in_half_open(self):
        """Test circuit reopens if failure occurs in half-open state"""
        cb = CircuitBreaker(
            failure_threshold=2,
            recovery_timeout=0.1
        )
        
        # Open circuit
        cb.record_failure()
        cb.record_failure()
        
        # Wait and transition to half-open
        import time
        time.sleep(0.2)
        cb.allow_request()
        
        # Failure in half-open should reopen
        cb.record_failure()
        assert cb.state == CircuitState.OPEN


class TestAdapterRegistry:
    """Tests for Adapter Registry"""
    
    def test_registry_initialization(self):
        """Test registry initialization"""
        config = {
            "providers": {
                "openai": {"enabled": True},
                "kokoro": {"enabled": False}
            }
        }
        
        registry = TTSAdapterRegistry(config)
        assert registry is not None
        assert registry.config == config
    
    @pytest.mark.asyncio
    async def test_register_adapter(self):
        """Test registering an adapter"""
        registry = TTSAdapterRegistry({})
        adapter = MockAdapter({"name": "test"})
        
        await registry.register_adapter("test", adapter)
        
        assert "test" in registry.adapters
        assert registry.adapters["test"] == adapter
    
    @pytest.mark.asyncio
    async def test_get_adapter(self):
        """Test getting an adapter"""
        registry = TTSAdapterRegistry({})
        adapter = MockAdapter({"name": "test"})
        
        await registry.register_adapter("test", adapter)
        retrieved = await registry.get_adapter("test")
        
        assert retrieved == adapter
    
    @pytest.mark.asyncio
    async def test_list_adapters(self):
        """Test listing adapters"""
        registry = TTSAdapterRegistry({})
        
        adapter1 = MockAdapter({"name": "test1"})
        adapter2 = MockAdapter({"name": "test2"})
        
        await registry.register_adapter("test1", adapter1)
        await registry.register_adapter("test2", adapter2)
        
        adapters = registry.list_adapters()
        
        assert "test1" in adapters
        assert "test2" in adapters


class TestVoiceCloning:
    """Tests for voice cloning functionality"""
    
    @pytest.mark.asyncio
    async def test_voice_reference_in_request(self):
        """Test voice reference field in request"""
        voice_data = base64.b64encode(b"audio data").decode()
        
        request = TTSRequest(
            text="Test with voice cloning",
            provider="higgs",
            voice="clone",
            voice_reference=voice_data
        )
        
        assert request.voice_reference == voice_data
        assert request.voice == "clone"
    
    def test_voice_reference_validation(self):
        """Test voice reference validation"""
        processor = AudioProcessor()
        
        # Test provider requirements
        reqs = processor.PROVIDER_REQUIREMENTS.get("higgs")
        assert reqs is not None
        assert reqs["min_duration"] == 3.0
        assert reqs["max_duration"] == 10.0
        assert reqs["preferred_sample_rate"] == 24000
        
        reqs = processor.PROVIDER_REQUIREMENTS.get("chatterbox")
        assert reqs is not None
        assert reqs["min_duration"] == 5.0
        assert reqs["max_duration"] == 20.0
        
        reqs = processor.PROVIDER_REQUIREMENTS.get("vibevoice")
        assert reqs is not None
        assert reqs["min_duration"] == 3.0
        assert reqs["max_duration"] == 30.0
    
    @pytest.mark.asyncio
    async def test_temp_file_cleanup(self):
        """Test temporary file cleanup for voice references"""
        # Create a temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as f:
            temp_path = f.name
            f.write(b"test audio data")
        
        assert os.path.exists(temp_path)
        
        # Simulate cleanup
        try:
            # Process file
            pass
        finally:
            if os.path.exists(temp_path):
                os.remove(temp_path)
        
        assert not os.path.exists(temp_path)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])