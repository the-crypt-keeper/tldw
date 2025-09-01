# test_dia_adapter_mock.py
# Description: Mock/Unit tests for Dia TTS adapter
#
# Imports
import pytest
import asyncio
from unittest.mock import AsyncMock, MagicMock, patch
import torch
#
# Local Imports
from tldw_Server_API.app.core.TTS.adapters.dia_adapter import DiaAdapter
from tldw_Server_API.app.core.TTS.adapters.base import (
    TTSRequest,
    TTSResponse,
    AudioFormat,
    ProviderStatus
)
from tldw_Server_API.app.core.TTS.tts_exceptions import (
    TTSModelNotFoundError,
    TTSModelLoadError,
    TTSProviderInitializationError
)
#
#######################################################################################################################
#
# Mock Tests for Dia Adapter

@pytest.mark.asyncio
class TestDiaAdapterMock:
    """Mock/Unit tests for Dia adapter"""
    
    async def test_initialization_configuration(self):
        """Test initialization with configuration"""
        adapter = DiaAdapter({
            "dia_api_key": "test-key",
            "dia_api_endpoint": "https://api.dia.ai/v1",
            "dia_model_path": "nari-labs/dia"
        })
        
        assert adapter.config.get("dia_api_key") == "test-key"
        assert adapter.config.get("dia_api_endpoint") == "https://api.dia.ai/v1"
        assert adapter.config.get("dia_model_path") == "nari-labs/dia"
    
    async def test_capabilities_reporting(self):
        """Test capabilities are correctly reported"""
        adapter = DiaAdapter({})
        caps = await adapter.get_capabilities()
        
        assert caps.provider_name == "Dia"
        assert caps.supports_streaming is True
        assert caps.supports_voice_cloning is True
        assert caps.supports_multi_speaker is True
        assert caps.max_text_length == 30000
        assert caps.sample_rate == 24000
        assert AudioFormat.WAV in caps.supported_formats
        assert AudioFormat.MP3 in caps.supported_formats
        assert AudioFormat.FLAC in caps.supported_formats
        assert AudioFormat.PCM in caps.supported_formats
        assert AudioFormat.OPUS in caps.supported_formats
    
    async def test_multi_speaker_support(self):
        """Test multi-speaker dialogue support"""
        adapter = DiaAdapter({})
        
        # Check voice presets
        assert "speaker1" in adapter.VOICE_PRESETS
        assert "speaker2" in adapter.VOICE_PRESETS
        assert "speaker3" in adapter.VOICE_PRESETS
        assert "narrator" in adapter.VOICE_PRESETS
    
    async def test_voice_mapping(self):
        """Test voice mapping functionality"""
        adapter = DiaAdapter({})
        
        # Test speaker voice mapping
        assert adapter.map_voice("speaker1") == "speaker1"
        assert adapter.map_voice("speaker2") == "speaker2"
        assert adapter.map_voice("narrator") == "narrator"
        
        # Test generic mappings
        assert adapter.map_voice("default") == "speaker1"
        assert adapter.map_voice("assistant") == "speaker1"
        assert adapter.map_voice("unknown") == "speaker1"
    
    async def test_dialogue_parsing(self):
        """Test dialogue parsing functionality"""
        adapter = DiaAdapter({})
        
        text = "Speaker1: Hello! Speaker2: Hi there! How are you?"
        dialogues = adapter.parse_dialogue(text)
        
        assert len(dialogues) == 2
        assert dialogues[0] == ("Speaker1", "Hello!")
        assert dialogues[1] == ("Speaker2", "Hi there! How are you?")
        
        # Test with no speaker markers
        plain_text = "Just plain text"
        dialogues = adapter.parse_dialogue(plain_text)
        assert len(dialogues) == 1
        assert dialogues[0] == ("default", "Just plain text")
    
    async def test_device_selection(self):
        """Test device selection for inference"""
        # Test CUDA selection
        with patch('torch.cuda.is_available', return_value=True):
            adapter = DiaAdapter({"dia_device": "cuda"})
            assert adapter.device == "cuda"
        
        # Test CPU fallback
        with patch('torch.cuda.is_available', return_value=False):
            adapter = DiaAdapter({"dia_device": "cuda"})
            assert adapter.device == "cpu"
    
    @patch('tldw_Server_API.app.core.TTS.adapters.dia_adapter.get_resource_manager')
    async def test_resource_manager_integration(self, mock_get_manager):
        """Test integration with resource manager"""
        mock_manager = AsyncMock()
        mock_manager.memory_monitor.is_memory_critical.return_value = False
        mock_get_manager.return_value = mock_manager
        
        adapter = DiaAdapter({})
        
        # Mock model loading
        with patch('tldw_Server_API.app.core.TTS.adapters.dia_adapter.DiaAdapter._load_dia_model', return_value=True):
            success = await adapter.initialize()
            # Will fail without actual model, but resource manager should be called
            mock_get_manager.assert_called()
    
    async def test_generation_without_initialization(self):
        """Test generation fails without initialization"""
        adapter = DiaAdapter({})
        
        request = TTSRequest(
            text="Test",
            voice="speaker1",
            format=AudioFormat.WAV
        )
        
        with pytest.raises(Exception):  # Should raise provider not configured
            await adapter.generate(request)
    
    async def test_cleanup_on_close(self):
        """Test resource cleanup on close"""
        adapter = DiaAdapter({"dia_device": "cuda"})
        
        # Mock resources
        adapter.model = MagicMock()
        adapter.processor = MagicMock()
        adapter._initialized = True
        adapter._status = ProviderStatus.AVAILABLE
        
        with patch('torch.cuda.is_available', return_value=True):
            with patch('torch.cuda.empty_cache') as mock_empty_cache:
                await adapter.close()
                
                assert adapter.model is None
                assert adapter.processor is None
                assert adapter._initialized is False
                assert adapter._status == ProviderStatus.DISABLED
                mock_empty_cache.assert_called_once()
    
    async def test_dialogue_request_preparation(self):
        """Test preparation of dialogue requests"""
        adapter = DiaAdapter({})
        
        request = TTSRequest(
            text="Speaker1: Hello! Speaker2: Hi there!",
            speakers={"Speaker1": "speaker1", "Speaker2": "speaker2"},
            format=AudioFormat.WAV
        )
        
        # Verify dialogue detection
        dialogues = adapter.parse_dialogue(request.text)
        assert len(dialogues) == 2
        
        # Verify speaker mapping
        if request.speakers:
            for speaker, voice in request.speakers.items():
                assert voice in adapter.VOICE_PRESETS
    
    async def test_model_path_configuration(self):
        """Test model path configuration"""
        # Test with HuggingFace path
        adapter = DiaAdapter({"dia_model_path": "nari-labs/dia"})
        assert adapter.model_path == "nari-labs/dia"
        
        # Test with local path
        adapter = DiaAdapter({"dia_model_path": "./models/dia"})
        assert adapter.model_path == "./models/dia"
        
        # Test default
        adapter = DiaAdapter({})
        assert adapter.model_path == "nari-labs/dia"

#######################################################################################################################
#
# End of test_dia_adapter_mock.py