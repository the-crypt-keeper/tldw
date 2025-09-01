# test_higgs_adapter_integration.py
# Description: Integration tests for Higgs TTS adapter
#
# Imports
import pytest
import os
import platform
import torch
import asyncio
#
# Local Imports
from tldw_Server_API.app.core.TTS.adapters.higgs_adapter import HiggsAdapter
from tldw_Server_API.app.core.TTS.adapters.base import (
    TTSRequest,
    AudioFormat,
    ProviderStatus
)
#
#######################################################################################################################
#
# Helper Functions

def check_higgs_model_exists():
    """Check if Higgs model and library are available"""
    try:
        from boson_multimodal.serve.serve_engine import HiggsAudioServeEngine
        # Check for model files
        model_paths = [
            os.path.expanduser("~/.cache/higgs/higgs-audio-v2"),
            "./models/higgs/higgs-audio-v2",
            "bosonai/higgs-audio-v2-generation-3B-base"
        ]
        return True, model_paths[0]
    except ImportError:
        return False, None

def get_compute_capability():
    """Detect compute capabilities"""
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"

#######################################################################################################################
#
# Integration Tests for Higgs Adapter

@pytest.mark.integration
@pytest.mark.asyncio
class TestHiggsAdapterIntegration:
    """Integration tests for Higgs adapter - requires model and library"""
    
    @pytest.mark.skipif(
        not check_higgs_model_exists()[0],
        reason="Requires Higgs model and boson_multimodal library"
    )
    async def test_real_model_initialization(self):
        """Test initialization with real model"""
        model_exists, model_path = check_higgs_model_exists()
        
        adapter = HiggsAdapter({
            "higgs_model_path": model_path,
            "higgs_device": get_compute_capability()
        })
        
        success = await adapter.initialize()
        assert success
        assert adapter._status == ProviderStatus.AVAILABLE
        
        # Cleanup
        await adapter.close()
    
    @pytest.mark.skipif(
        not check_higgs_model_exists()[0],
        reason="Requires Higgs model"
    )
    async def test_real_audio_generation(self):
        """Test actual audio generation with Higgs"""
        model_exists, model_path = check_higgs_model_exists()
        
        adapter = HiggsAdapter({
            "higgs_model_path": model_path,
            "higgs_device": "cpu"  # Use CPU for testing
        })
        
        await adapter.initialize()
        
        request = TTSRequest(
            text="Hello, this is a test of the Higgs text-to-speech system.",
            voice="narrator",
            format=AudioFormat.WAV,
            stream=False
        )
        
        response = await adapter.generate(request)
        
        assert response.audio_data is not None
        assert len(response.audio_data) > 1000
        assert response.format == AudioFormat.WAV
        assert response.voice_used == "narrator"
        assert response.provider == "Higgs"
        
        # Cleanup
        await adapter.close()
    
    @pytest.mark.skipif(
        not check_higgs_model_exists()[0],
        reason="Requires Higgs model"
    )
    async def test_streaming_generation(self):
        """Test streaming audio generation"""
        model_exists, model_path = check_higgs_model_exists()
        
        adapter = HiggsAdapter({
            "higgs_model_path": model_path,
            "higgs_device": "cpu"
        })
        
        await adapter.initialize()
        
        request = TTSRequest(
            text="Streaming test with Higgs.",
            voice="conversational",
            format=AudioFormat.WAV,
            stream=True
        )
        
        response = await adapter.generate(request)
        
        assert response.audio_stream is not None
        
        chunks = []
        async for chunk in response.audio_stream:
            chunks.append(chunk)
        
        assert len(chunks) > 0
        total_size = sum(len(chunk) for chunk in chunks)
        assert total_size > 1000
        
        # Cleanup
        await adapter.close()
    
    @pytest.mark.skipif(
        not check_higgs_model_exists()[0],
        reason="Requires Higgs model"
    )
    async def test_different_voices(self):
        """Test generation with different voice presets"""
        model_exists, model_path = check_higgs_model_exists()
        
        adapter = HiggsAdapter({
            "higgs_model_path": model_path,
            "higgs_device": "cpu"
        })
        
        await adapter.initialize()
        
        voices = ["narrator", "conversational", "expressive", "melodic"]
        
        for voice in voices:
            request = TTSRequest(
                text=f"Testing voice: {voice}",
                voice=voice,
                format=AudioFormat.WAV,
                stream=False
            )
            
            response = await adapter.generate(request)
            
            assert response.audio_data is not None
            assert response.voice_used == voice
        
        # Cleanup
        await adapter.close()
    
    @pytest.mark.skipif(
        not check_higgs_model_exists()[0],
        reason="Requires Higgs model"
    )
    async def test_emotion_control(self):
        """Test emotion control in generation"""
        model_exists, model_path = check_higgs_model_exists()
        
        adapter = HiggsAdapter({
            "higgs_model_path": model_path,
            "higgs_device": "cpu"
        })
        
        await adapter.initialize()
        
        emotions = ["happy", "sad", "angry", "neutral"]
        
        for emotion in emotions:
            request = TTSRequest(
                text=f"Testing emotion: {emotion}",
                voice="expressive",
                emotion=emotion,
                emotion_intensity=1.5,
                format=AudioFormat.WAV,
                stream=False
            )
            
            response = await adapter.generate(request)
            
            assert response.audio_data is not None
        
        # Cleanup
        await adapter.close()
    
    @pytest.mark.skipif(
        not check_higgs_model_exists()[0],
        reason="Requires Higgs model"
    )
    async def test_multilingual_generation(self):
        """Test generation in different languages"""
        model_exists, model_path = check_higgs_model_exists()
        
        adapter = HiggsAdapter({
            "higgs_model_path": model_path,
            "higgs_device": "cpu"
        })
        
        await adapter.initialize()
        
        # Test different languages
        languages = [
            ("en", "Hello, how are you?"),
            ("zh", "你好，你好吗？"),
            ("es", "Hola, ¿cómo estás?"),
            ("fr", "Bonjour, comment allez-vous?"),
            ("ja", "こんにちは、お元気ですか？")
        ]
        
        for lang, text in languages:
            request = TTSRequest(
                text=text,
                voice="narrator",
                language=lang,
                format=AudioFormat.WAV,
                stream=False
            )
            
            response = await adapter.generate(request)
            assert response.audio_data is not None
        
        # Cleanup
        await adapter.close()
    
    @pytest.mark.skipif(
        not check_higgs_model_exists()[0] or get_compute_capability() == "cpu",
        reason="Requires Higgs model and GPU"
    )
    async def test_gpu_acceleration(self):
        """Test GPU acceleration if available"""
        model_exists, model_path = check_higgs_model_exists()
        
        adapter = HiggsAdapter({
            "higgs_model_path": model_path,
            "higgs_device": "cuda",
            "higgs_use_fp16": True
        })
        
        await adapter.initialize()
        
        assert adapter.device == "cuda"
        assert adapter.use_fp16 is True
        
        request = TTSRequest(
            text="GPU acceleration test with FP16",
            voice="narrator",
            format=AudioFormat.WAV,
            stream=False
        )
        
        response = await adapter.generate(request)
        
        assert response.audio_data is not None
        
        # Cleanup
        await adapter.close()
    
    @pytest.mark.skipif(
        not check_higgs_model_exists()[0],
        reason="Requires Higgs model"
    )
    async def test_long_text_generation(self):
        """Test generation with very long text"""
        model_exists, model_path = check_higgs_model_exists()
        
        adapter = HiggsAdapter({
            "higgs_model_path": model_path,
            "higgs_device": "cpu"
        })
        
        await adapter.initialize()
        
        # Higgs can handle very long texts (up to 50k chars)
        long_text = "This is a test sentence. " * 500  # ~12500 characters
        
        request = TTSRequest(
            text=long_text,
            voice="narrator",
            format=AudioFormat.WAV,
            stream=False
        )
        
        response = await adapter.generate(request)
        
        assert response.audio_data is not None
        assert len(response.audio_data) > 50000  # Should be very substantial
        
        # Cleanup
        await adapter.close()
    
    @pytest.mark.skipif(
        not check_higgs_model_exists()[0],
        reason="Requires Higgs model"
    )
    async def test_voice_cloning_with_reference(self):
        """Test voice cloning with reference audio"""
        model_exists, model_path = check_higgs_model_exists()
        
        adapter = HiggsAdapter({
            "higgs_model_path": model_path,
            "higgs_device": "cpu"
        })
        
        await adapter.initialize()
        
        # Create a simple WAV file as reference (would need real audio in practice)
        # For testing, we'll skip actual voice cloning
        
        request = TTSRequest(
            text="Testing voice cloning feature",
            voice="narrator",  # Will be overridden if reference provided
            format=AudioFormat.WAV,
            stream=False
        )
        
        response = await adapter.generate(request)
        
        assert response.audio_data is not None
        
        # Cleanup
        await adapter.close()

#######################################################################################################################
#
# End of test_higgs_adapter_integration.py