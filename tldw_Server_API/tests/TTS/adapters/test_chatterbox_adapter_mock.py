# test_chatterbox_adapter_mock.py
# Description: Mock/Unit tests for Chatterbox TTS adapter
#
# Imports
import pytest
import asyncio
from unittest.mock import AsyncMock, MagicMock, patch
import torch
#
# Local Imports
from tldw_Server_API.app.core.TTS.adapters.chatterbox_adapter import ChatterboxAdapter
from tldw_Server_API.app.core.TTS.adapters.base import (
    TTSRequest,
    TTSResponse,
    AudioFormat,
    ProviderStatus
)
from tldw_Server_API.app.core.TTS.tts_exceptions import (
    TTSModelNotFoundError,
    TTSModelLoadError
)
#
#######################################################################################################################
#
# Mock Tests for Chatterbox Adapter

@pytest.mark.asyncio
class TestChatterboxAdapterMock:
    """Mock/Unit tests for Chatterbox adapter"""
    
    async def test_initialization_configuration(self):
        """Test initialization with configuration"""
        adapter = ChatterboxAdapter({
            "chatterbox_model": "large-v2",
            "chatterbox_api_key": "test-key",
            "chatterbox_model_path": "./models/chatterbox",
            "chatterbox_device": "cuda"
        })
        
        assert adapter.config.get("chatterbox_model") == "large-v2"
        assert adapter.config.get("chatterbox_api_key") == "test-key"
        assert adapter.config.get("chatterbox_model_path") == "./models/chatterbox"
        assert adapter.device == "cuda" if torch.cuda.is_available() else "cpu"
    
    async def test_capabilities_reporting(self):
        """Test capabilities are correctly reported"""
        adapter = ChatterboxAdapter({})
        caps = await adapter.get_capabilities()
        
        assert caps.provider_name == "Chatterbox"
        assert caps.supports_streaming is True
        assert caps.supports_voice_cloning is False
        assert caps.supports_emotion_control is True
        assert caps.supports_speech_rate is True
        assert caps.max_text_length == 10000
        assert caps.sample_rate == 24000
        assert AudioFormat.WAV in caps.supported_formats
        assert AudioFormat.MP3 in caps.supported_formats
        assert AudioFormat.OPUS in caps.supported_formats
    
    async def test_character_voice_presets(self):
        """Test character voice presets"""
        adapter = ChatterboxAdapter({})
        
        # Check character voices exist
        assert "narrator" in adapter.CHARACTER_VOICES
        assert "hero" in adapter.CHARACTER_VOICES
        assert "villain" in adapter.CHARACTER_VOICES
        assert "sidekick" in adapter.CHARACTER_VOICES
        assert "sage" in adapter.CHARACTER_VOICES
        assert "comic_relief" in adapter.CHARACTER_VOICES
    
    async def test_voice_mapping(self):
        """Test voice mapping functionality"""
        adapter = ChatterboxAdapter({})
        
        # Test character voice mapping
        assert adapter.map_voice("narrator") == "narrator"
        assert adapter.map_voice("hero") == "hero"
        assert adapter.map_voice("villain") == "villain"
        
        # Test generic mappings
        assert adapter.map_voice("default") == "narrator"
        assert adapter.map_voice("assistant") == "sidekick"
        assert adapter.map_voice("evil") == "villain"
        assert adapter.map_voice("wise") == "sage"
        assert adapter.map_voice("funny") == "comic_relief"
    
    async def test_style_parameters(self):
        """Test speech style parameters"""
        adapter = ChatterboxAdapter({})
        
        request = TTSRequest(
            text="Dramatic speech",
            voice="narrator",
            style="dramatic",
            extra_params={
                "emphasis_level": 0.8,
                "tone": "serious",
                "pacing": "slow"
            }
        )
        
        assert request.style == "dramatic"
        assert request.extra_params.get("emphasis_level") == 0.8
        assert request.extra_params.get("tone") == "serious"
        assert request.extra_params.get("pacing") == "slow"
    
    async def test_device_selection(self):
        """Test device selection for inference"""
        # Test CUDA selection
        with patch('torch.cuda.is_available', return_value=True):
            adapter = ChatterboxAdapter({"chatterbox_device": "cuda"})
            assert adapter.device == "cuda"
        
        # Test CPU fallback
        with patch('torch.cuda.is_available', return_value=False):
            adapter = ChatterboxAdapter({"chatterbox_device": "cuda"})
            assert adapter.device == "cpu"
    
    async def test_model_not_installed_error(self):
        """Test error when Chatterbox library not installed"""
        adapter = ChatterboxAdapter({})
        
        # Mock import error for chatterbox-tts
        with patch('builtins.__import__', side_effect=ImportError("chatterbox-tts not found")):
            success = await adapter.initialize()
            assert not success
            assert adapter._status == ProviderStatus.ERROR
    
    async def test_generation_without_initialization(self):
        """Test generation fails without initialization"""
        adapter = ChatterboxAdapter({})
        
        request = TTSRequest(
            text="Test",
            voice="narrator",
            format=AudioFormat.WAV
        )
        
        with pytest.raises(Exception):  # Should raise provider not configured
            await adapter.generate(request)
    
    async def test_character_dialogue_preparation(self):
        """Test preparation of character dialogue"""
        adapter = ChatterboxAdapter({})
        
        # Test dialogue with character voices
        text = "Narrator: Once upon a time... Hero: I must save the day! Villain: You'll never stop me!"
        dialogues = adapter.parse_dialogue(text)
        
        assert len(dialogues) == 3
        assert dialogues[0] == ("Narrator", "Once upon a time...")
        assert dialogues[1] == ("Hero", "I must save the day!")
        assert dialogues[2] == ("Villain", "You'll never stop me!")
    
    async def test_emotion_intensity_mapping(self):
        """Test emotion intensity mapping"""
        adapter = ChatterboxAdapter({})
        
        # Test different emotion intensities
        emotions = [
            ("happy", 0.5, "slightly happy"),
            ("sad", 1.0, "moderately sad"),
            ("angry", 2.0, "very angry"),
            ("excited", 0.3, "mildly excited")
        ]
        
        for emotion, intensity, expected in emotions:
            request = TTSRequest(
                text="Test",
                emotion=emotion,
                emotion_intensity=intensity
            )
            
            # Verify emotion parameters
            assert request.emotion == emotion
            assert request.emotion_intensity == intensity
    
    async def test_cleanup_on_close(self):
        """Test resource cleanup on close"""
        adapter = ChatterboxAdapter({"chatterbox_device": "cuda"})
        
        # Mock resources
        adapter.model = MagicMock()
        adapter.vocoder = MagicMock()
        adapter._initialized = True
        adapter._status = ProviderStatus.AVAILABLE
        
        with patch('torch.cuda.is_available', return_value=True):
            with patch('torch.cuda.empty_cache') as mock_empty_cache:
                await adapter.close()
                
                assert adapter.model is None
                assert adapter.vocoder is None
                assert adapter._initialized is False
                assert adapter._status == ProviderStatus.DISABLED
                mock_empty_cache.assert_called_once()
    
    async def test_model_variant_selection(self):
        """Test model variant selection"""
        # Test small model
        adapter_small = ChatterboxAdapter({"chatterbox_model": "small"})
        assert adapter_small.config.get("chatterbox_model") == "small"
        
        # Test medium model
        adapter_medium = ChatterboxAdapter({"chatterbox_model": "medium"})
        assert adapter_medium.config.get("chatterbox_model") == "medium"
        
        # Test large model
        adapter_large = ChatterboxAdapter({"chatterbox_model": "large-v2"})
        assert adapter_large.config.get("chatterbox_model") == "large-v2"
    
    async def test_speech_rate_control(self):
        """Test speech rate control"""
        adapter = ChatterboxAdapter({})
        
        speeds = [0.5, 1.0, 1.5, 2.0]
        
        for speed in speeds:
            request = TTSRequest(
                text="Speed test",
                voice="narrator",
                speed=speed
            )
            
            assert request.speed == speed

#######################################################################################################################
#
# End of test_chatterbox_adapter_mock.py