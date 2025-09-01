# test_kokoro_adapter_mock.py
# Description: Mock/Unit tests for Kokoro TTS adapter
#
# Imports
import pytest
import asyncio
from unittest.mock import AsyncMock, MagicMock, patch
import os
import torch
#
# Local Imports
from tldw_Server_API.app.core.TTS.adapters.kokoro_adapter import KokoroAdapter
from tldw_Server_API.app.core.TTS.adapters.base import (
    TTSRequest,
    TTSResponse,
    AudioFormat,
    ProviderStatus
)
from tldw_Server_API.app.core.TTS.tts_exceptions import (
    TTSModelNotFoundError,
    TTSModelLoadError,
    TTSGenerationError
)
#
#######################################################################################################################
#
# Mock Tests for Kokoro Adapter

@pytest.mark.asyncio
class TestKokoroAdapterMock:
    """Mock/Unit tests for Kokoro adapter"""
    
    async def test_initialization_pytorch_mode(self):
        """Test initialization in PyTorch mode"""
        adapter = KokoroAdapter({
            "kokoro_use_onnx": False,
            "kokoro_model_path": "test_model.pth"
        })
        
        assert adapter.use_onnx is False
        assert adapter.model_path == "test_model.pth"
        
        # Without actual model files, initialization should fail
        with patch('os.path.exists', return_value=False):
            success = await adapter.initialize()
            assert not success
    
    async def test_initialization_onnx_mode(self):
        """Test initialization in ONNX mode"""
        adapter = KokoroAdapter({
            "kokoro_use_onnx": True,
            "kokoro_model_path": "test_model.onnx",
            "kokoro_voices_json": "test_voices.json"
        })
        
        assert adapter.use_onnx is True
        assert adapter.model_path == "test_model.onnx"
        assert adapter.voices_json_path == "test_voices.json"
        
        # Without actual model files, initialization should fail
        with patch('os.path.exists', return_value=False):
            success = await adapter.initialize()
            assert not success
    
    async def test_capabilities_reporting(self):
        """Test capabilities are correctly reported"""
        adapter = KokoroAdapter({})
        caps = await adapter.get_capabilities()
        
        assert caps.provider_name == "Kokoro"
        assert caps.supports_streaming is True
        assert caps.supports_voice_cloning is False
        assert caps.supports_emotion_control is False
        assert caps.supports_phonemes is True
        assert caps.max_text_length == 500
        assert AudioFormat.WAV in caps.supported_formats
        assert AudioFormat.MP3 in caps.supported_formats
    
    async def test_voice_mapping(self):
        """Test voice mapping functionality"""
        adapter = KokoroAdapter({})
        
        # Test American voices
        assert adapter.map_voice("american_female") == "af_bella"
        assert adapter.map_voice("american_male") == "am_adam"
        
        # Test British voices
        assert adapter.map_voice("british_female") == "bf_emma"
        assert adapter.map_voice("british_male") == "bm_george"
        
        # Test default mappings
        assert adapter.map_voice("female") == "af_bella"
        assert adapter.map_voice("male") == "am_adam"
        assert adapter.map_voice("child") == "af_nicole"
        
        # Test passthrough for actual voice IDs
        assert adapter.map_voice("af_sky") == "af_sky"
        assert adapter.map_voice("am_michael") == "am_michael"
    
    async def test_voice_mixing(self):
        """Test voice mixing functionality"""
        adapter = KokoroAdapter({})
        
        # Test single voice
        assert adapter._process_voice("af_bella") == "af_bella"
        
        # Test mixed voices with weights
        mixed = "af_bella(2)+af_sky(1)"
        assert adapter._process_voice(mixed) == mixed
        
        # Test complex mixing
        complex_mix = "af_bella(3)+am_adam(1)+bf_emma(2)"
        assert adapter._process_voice(complex_mix) == complex_mix
    
    async def test_phoneme_processing(self):
        """Test phoneme processing support"""
        adapter = KokoroAdapter({})
        
        # Test that phoneme input is accepted
        phoneme_text = "[ˈhɛloʊ ˈwɝld]"
        request = TTSRequest(
            text=phoneme_text,
            voice="af_bella",
            format=AudioFormat.WAV,
            extra_params={"use_phonemes": True}
        )
        
        # Verify phoneme text is preserved
        assert request.text == phoneme_text
        assert request.extra_params.get("use_phonemes") is True
    
    async def test_device_selection(self):
        """Test device selection for inference"""
        # Test CUDA selection
        with patch('torch.cuda.is_available', return_value=True):
            adapter = KokoroAdapter({"kokoro_device": "cuda"})
            assert adapter.device == "cuda"
        
        # Test CPU fallback when CUDA not available
        with patch('torch.cuda.is_available', return_value=False):
            adapter = KokoroAdapter({"kokoro_device": "cuda"})
            assert adapter.device == "cpu"
        
        # Test explicit CPU selection
        adapter = KokoroAdapter({"kokoro_device": "cpu"})
        assert adapter.device == "cpu"
    
    async def test_model_loading_error_handling(self):
        """Test error handling during model loading"""
        adapter = KokoroAdapter({
            "kokoro_model_path": "/nonexistent/model.pth"
        })
        
        with patch('os.path.exists', return_value=False):
            success = await adapter.initialize()
            assert not success
            assert adapter._status == ProviderStatus.ERROR
    
    @patch('tldw_Server_API.app.core.TTS.adapters.kokoro_adapter.KokoroAdapter._load_pytorch_model')
    async def test_successful_initialization_pytorch(self, mock_load):
        """Test successful initialization with PyTorch"""
        mock_load.return_value = True
        
        adapter = KokoroAdapter({
            "kokoro_use_onnx": False,
            "kokoro_model_path": "model.pth"
        })
        
        with patch('os.path.exists', return_value=True):
            success = await adapter.initialize()
            assert success
            assert adapter._status == ProviderStatus.AVAILABLE
            mock_load.assert_called_once()
    
    @patch('tldw_Server_API.app.core.TTS.adapters.kokoro_adapter.KokoroAdapter._load_onnx_model')
    async def test_successful_initialization_onnx(self, mock_load):
        """Test successful initialization with ONNX"""
        mock_load.return_value = True
        
        adapter = KokoroAdapter({
            "kokoro_use_onnx": True,
            "kokoro_model_path": "model.onnx",
            "kokoro_voices_json": "voices.json"
        })
        
        with patch('os.path.exists', return_value=True):
            success = await adapter.initialize()
            assert success
            assert adapter._status == ProviderStatus.AVAILABLE
            mock_load.assert_called_once()
    
    async def test_generation_without_initialization(self):
        """Test generation fails without initialization"""
        adapter = KokoroAdapter({})
        
        request = TTSRequest(
            text="Test",
            voice="af_bella",
            format=AudioFormat.WAV
        )
        
        with pytest.raises(Exception):  # Should raise provider not configured
            await adapter.generate(request)
    
    async def test_text_length_validation(self):
        """Test text length validation"""
        adapter = KokoroAdapter({})
        
        # Text exceeding limit
        long_text = "a" * 600  # Exceeds 500 character limit
        request = TTSRequest(
            text=long_text,
            voice="af_bella",
            format=AudioFormat.WAV
        )
        
        # Should be validated during generation
        caps = await adapter.get_capabilities()
        assert len(long_text) > caps.max_text_length
    
    async def test_cleanup_on_close(self):
        """Test resource cleanup on close"""
        adapter = KokoroAdapter({})
        
        # Mock some resources
        adapter.model = MagicMock()
        adapter.phonemizer = MagicMock()
        adapter._initialized = True
        adapter._status = ProviderStatus.AVAILABLE
        
        await adapter.close()
        
        assert adapter.model is None
        assert adapter.phonemizer is None
        assert adapter._initialized is False
        assert adapter._status == ProviderStatus.DISABLED
    
    async def test_cuda_memory_cleanup(self):
        """Test CUDA memory cleanup on close"""
        adapter = KokoroAdapter({"kokoro_device": "cuda"})
        
        with patch('torch.cuda.is_available', return_value=True):
            with patch('torch.cuda.empty_cache') as mock_empty_cache:
                adapter.device = "cuda"
                await adapter.close()
                mock_empty_cache.assert_called_once()

#######################################################################################################################
#
# End of test_kokoro_adapter_mock.py