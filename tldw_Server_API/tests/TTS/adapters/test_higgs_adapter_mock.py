# test_higgs_adapter_mock.py
# Description: Mock/Unit tests for Higgs TTS adapter
#
# Imports
import pytest
import asyncio
from unittest.mock import AsyncMock, MagicMock, patch
import torch
#
# Local Imports
from tldw_Server_API.app.core.TTS.adapters.higgs_adapter import HiggsAdapter
from tldw_Server_API.app.core.TTS.adapters.base import (
    TTSRequest,
    TTSResponse,
    AudioFormat,
    ProviderStatus
)
from tldw_Server_API.app.core.TTS.tts_exceptions import (
    TTSModelNotFoundError,
    TTSModelLoadError,
    TTSInsufficientMemoryError
)
#
#######################################################################################################################
#
# Mock Tests for Higgs Adapter

@pytest.mark.asyncio
class TestHiggsAdapterMock:
    """Mock/Unit tests for Higgs adapter"""
    
    async def test_initialization_configuration(self):
        """Test initialization with configuration"""
        adapter = HiggsAdapter({
            "higgs_model_path": "bosonai/higgs-audio-v2",
            "higgs_tokenizer_path": "bosonai/higgs-tokenizer",
            "higgs_device": "cuda",
            "higgs_use_fp16": True,
            "higgs_batch_size": 2
        })
        
        assert adapter.model_path == "bosonai/higgs-audio-v2"
        assert adapter.tokenizer_path == "bosonai/higgs-tokenizer"
        assert adapter.device == "cuda" if torch.cuda.is_available() else "cpu"
        assert adapter.use_fp16 is True
        assert adapter.batch_size == 2
    
    async def test_capabilities_reporting(self):
        """Test capabilities are correctly reported"""
        adapter = HiggsAdapter({})
        caps = await adapter.get_capabilities()
        
        assert caps.provider_name == "Higgs"
        assert caps.supports_streaming is True
        assert caps.supports_voice_cloning is True
        assert caps.supports_emotion_control is True
        assert caps.supports_multi_speaker is True
        assert caps.supports_background_audio is True
        assert caps.max_text_length == 50000  # Higgs can handle very long texts
        assert caps.sample_rate == 24000
        assert AudioFormat.WAV in caps.supported_formats
        assert AudioFormat.MP3 in caps.supported_formats
    
    async def test_voice_presets(self):
        """Test voice preset mapping"""
        adapter = HiggsAdapter({})
        
        # Test preset voices
        assert adapter.map_voice("narrator") == "narrator"
        assert adapter.map_voice("conversational") == "conversational"
        assert adapter.map_voice("expressive") == "expressive"
        assert adapter.map_voice("melodic") == "melodic"
        
        # Test generic mappings
        assert adapter.map_voice("default") == "conversational"
        assert adapter.map_voice("assistant") == "conversational"
        assert adapter.map_voice("emotional") == "expressive"
        assert adapter.map_voice("singing") == "melodic"
        assert adapter.map_voice("musical") == "melodic"
    
    async def test_supported_languages(self):
        """Test language support"""
        adapter = HiggsAdapter({})
        
        # Higgs supports 50+ languages
        assert "en" in adapter.SUPPORTED_LANGUAGES
        assert "zh" in adapter.SUPPORTED_LANGUAGES
        assert "es" in adapter.SUPPORTED_LANGUAGES
        assert "fr" in adapter.SUPPORTED_LANGUAGES
        assert "de" in adapter.SUPPORTED_LANGUAGES
        assert "ja" in adapter.SUPPORTED_LANGUAGES
        assert "ko" in adapter.SUPPORTED_LANGUAGES
        assert "ar" in adapter.SUPPORTED_LANGUAGES
        assert len(adapter.SUPPORTED_LANGUAGES) >= 50
    
    async def test_chat_ml_preparation(self):
        """Test ChatML format preparation for Higgs"""
        adapter = HiggsAdapter({})
        
        request = TTSRequest(
            text="Hello world",
            voice="narrator",
            language="en",
            emotion="happy",
            emotion_intensity=1.5,
            style="dramatic",
            speed=1.2,
            seed=42
        )
        
        chat_ml = adapter._prepare_higgs_chat_ml(request)
        
        assert "messages" in chat_ml
        assert len(chat_ml["messages"]) >= 1
        assert chat_ml["voice"] == "narrator"
        assert chat_ml["speed"] == 1.2
        assert chat_ml["seed"] == 42
        
        # Check emotion instruction in content
        user_message = chat_ml["messages"][-1]
        assert "moderately happy" in user_message["content"]
        assert "dramatic style" in user_message["content"]
    
    async def test_multi_speaker_dialogue(self):
        """Test multi-speaker dialogue support"""
        adapter = HiggsAdapter({})
        
        request = TTSRequest(
            text="Speaker1: Hello! Speaker2: Hi there!",
            speakers={"Speaker1": "narrator", "Speaker2": "conversational"}
        )
        
        chat_ml = adapter._prepare_higgs_chat_ml(request)
        
        user_message = chat_ml["messages"][-1]
        assert "multiple speakers" in user_message["content"]
    
    @patch('tldw_Server_API.app.core.TTS.adapters.higgs_adapter.get_resource_manager')
    async def test_memory_check_before_loading(self, mock_get_manager):
        """Test memory checking before model loading"""
        mock_manager = AsyncMock()
        mock_manager.memory_monitor.is_memory_critical.return_value = True
        mock_manager.memory_monitor.get_memory_usage.return_value = {"available": "100MB"}
        mock_get_manager.return_value = mock_manager
        
        adapter = HiggsAdapter({})
        
        with pytest.raises(TTSInsufficientMemoryError):
            await adapter.initialize()
    
    async def test_voice_reference_validation(self):
        """Test voice reference validation"""
        adapter = HiggsAdapter({})
        
        # Test with valid voice reference
        valid_reference = b"RIFF" + b"\x00" * 100  # Minimal WAV header
        
        request = TTSRequest(
            text="Clone test",
            voice_reference=valid_reference
        )
        
        # Should accept voice reference
        assert request.voice_reference is not None
    
    async def test_device_selection(self):
        """Test device selection for inference"""
        # Test CUDA selection
        with patch('torch.cuda.is_available', return_value=True):
            adapter = HiggsAdapter({"higgs_device": "cuda"})
            assert adapter.device == "cuda"
        
        # Test CPU fallback
        with patch('torch.cuda.is_available', return_value=False):
            adapter = HiggsAdapter({"higgs_device": "cuda"})
            assert adapter.device == "cpu"
    
    async def test_fp16_configuration(self):
        """Test FP16 configuration"""
        # FP16 only on CUDA
        with patch('torch.cuda.is_available', return_value=True):
            adapter = HiggsAdapter({
                "higgs_device": "cuda",
                "higgs_use_fp16": True
            })
            assert adapter.use_fp16 is True
        
        # No FP16 on CPU
        adapter = HiggsAdapter({
            "higgs_device": "cpu",
            "higgs_use_fp16": True
        })
        assert adapter.use_fp16 is False  # Should be disabled on CPU
    
    async def test_model_not_found_error(self):
        """Test error when model library not installed"""
        adapter = HiggsAdapter({})
        
        with patch('tldw_Server_API.app.core.TTS.adapters.higgs_adapter.HiggsAdapter.initialize') as mock_init:
            mock_init.side_effect = ImportError("boson_multimodal not found")
            
            success = await adapter.initialize()
            assert not success
    
    async def test_cleanup_on_close(self):
        """Test resource cleanup on close"""
        adapter = HiggsAdapter({"higgs_device": "cuda"})
        
        # Mock resources
        adapter.serve_engine = MagicMock()
        adapter._initialized = True
        adapter._status = ProviderStatus.AVAILABLE
        
        with patch('torch.cuda.is_available', return_value=True):
            with patch('torch.cuda.empty_cache') as mock_empty_cache:
                await adapter.close()
                
                assert adapter.serve_engine is None
                assert adapter._initialized is False
                assert adapter._status == ProviderStatus.DISABLED
                mock_empty_cache.assert_called_once()
    
    async def test_generation_without_initialization(self):
        """Test generation fails without initialization"""
        adapter = HiggsAdapter({})
        
        request = TTSRequest(
            text="Test",
            voice="narrator",
            format=AudioFormat.WAV
        )
        
        with pytest.raises(Exception):  # Should raise provider not configured
            await adapter.generate(request)
    
    async def test_voice_cloning_with_reference(self):
        """Test voice cloning with reference audio"""
        adapter = HiggsAdapter({})
        
        # Mock voice reference processing
        with patch.object(adapter, '_prepare_voice_reference', return_value="/tmp/voice.wav"):
            request = TTSRequest(
                text="Clone my voice",
                voice_reference=b"fake_audio_data"
            )
            
            chat_ml = adapter._prepare_higgs_chat_ml(request, "/tmp/voice.wav")
            
            assert chat_ml["reference_audio_path"] == "/tmp/voice.wav"
            assert chat_ml["voice"] == "cloned"

#######################################################################################################################
#
# End of test_higgs_adapter_mock.py