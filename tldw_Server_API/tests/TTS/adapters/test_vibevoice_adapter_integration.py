# test_vibevoice_adapter_integration.py
# Description: Integration tests for VibeVoice TTS adapter
#
# Imports
import pytest
import os
import platform
import torch
import asyncio
#
# Local Imports
from tldw_Server_API.app.core.TTS.adapters.vibevoice_adapter import VibeVoiceAdapter
from tldw_Server_API.app.core.TTS.adapters.base import (
    TTSRequest,
    AudioFormat,
    ProviderStatus
)
#
#######################################################################################################################
#
# Helper Functions

def check_vibevoice_model_exists():
    """Check if VibeVoice model files exist"""
    # Check for model files in common locations
    model_paths = [
        os.path.expanduser("~/.cache/vibevoice"),
        "./models/vibevoice/775M",
        "./models/vibevoice/1.5B",
        os.path.expanduser("~/models/vibevoice")
    ]
    
    for path in model_paths:
        if os.path.exists(path):
            # Check which variant
            if "1.5B" in path or os.path.exists(os.path.join(path, "1.5B")):
                return True, path, "1.5B"
            elif "775M" in path or os.path.exists(os.path.join(path, "775M")):
                return True, path, "775M"
            else:
                return True, path, "775M"  # Default variant
    
    return False, None, None

def get_compute_capability():
    """Detect compute capabilities"""
    if platform.system() == "Darwin":
        if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            return "mps"
    elif torch.cuda.is_available():
        return "cuda"
    return "cpu"

#######################################################################################################################
#
# Integration Tests for VibeVoice Adapter

@pytest.mark.integration
@pytest.mark.asyncio
class TestVibeVoiceAdapterIntegration:
    """Integration tests for VibeVoice adapter - requires model files"""
    
    @pytest.mark.skipif(
        not check_vibevoice_model_exists()[0],
        reason="Requires VibeVoice model files to be downloaded"
    )
    async def test_real_model_initialization(self):
        """Test initialization with real model"""
        model_exists, model_path, variant = check_vibevoice_model_exists()
        
        adapter = VibeVoiceAdapter({
            "vibevoice_model_path": model_path,
            "vibevoice_variant": variant,
            "vibevoice_device": get_compute_capability()
        })
        
        success = await adapter.initialize()
        assert success
        assert adapter._status == ProviderStatus.AVAILABLE
        assert adapter.variant == variant
        
        # Cleanup
        await adapter.close()
    
    @pytest.mark.skipif(
        not check_vibevoice_model_exists()[0],
        reason="Requires VibeVoice model files"
    )
    async def test_real_audio_generation(self):
        """Test actual audio generation with VibeVoice"""
        model_exists, model_path, variant = check_vibevoice_model_exists()
        
        adapter = VibeVoiceAdapter({
            "vibevoice_model_path": model_path,
            "vibevoice_variant": variant,
            "vibevoice_device": "cpu"  # Use CPU for testing
        })
        
        await adapter.initialize()
        
        request = TTSRequest(
            text="Hello, this is a test of the VibeVoice text-to-speech system.",
            voice="professional",
            format=AudioFormat.WAV,
            stream=False
        )
        
        response = await adapter.generate(request)
        
        assert response.audio_data is not None
        assert len(response.audio_data) > 1000
        assert response.format == AudioFormat.WAV
        assert response.voice_used == "professional"
        assert response.provider == f"VibeVoice-{variant}"
        
        # Cleanup
        await adapter.close()
    
    @pytest.mark.skipif(
        not check_vibevoice_model_exists()[0],
        reason="Requires VibeVoice model files"
    )
    async def test_streaming_generation(self):
        """Test streaming audio generation"""
        model_exists, model_path, variant = check_vibevoice_model_exists()
        
        adapter = VibeVoiceAdapter({
            "vibevoice_model_path": model_path,
            "vibevoice_variant": variant,
            "vibevoice_device": "cpu"
        })
        
        await adapter.initialize()
        
        request = TTSRequest(
            text="Streaming test with VibeVoice.",
            voice="casual",
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
        not check_vibevoice_model_exists()[0],
        reason="Requires VibeVoice model files"
    )
    async def test_different_voices(self):
        """Test generation with different voice presets"""
        model_exists, model_path, variant = check_vibevoice_model_exists()
        
        adapter = VibeVoiceAdapter({
            "vibevoice_model_path": model_path,
            "vibevoice_variant": variant,
            "vibevoice_device": "cpu"
        })
        
        await adapter.initialize()
        
        voices = ["professional", "casual", "narrator", "energetic", "calm"]
        
        for voice in voices:
            request = TTSRequest(
                text=f"Testing {voice} voice preset",
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
        not check_vibevoice_model_exists()[0],
        reason="Requires VibeVoice model files"
    )
    async def test_multilingual_generation(self):
        """Test generation in multiple languages"""
        model_exists, model_path, variant = check_vibevoice_model_exists()
        
        adapter = VibeVoiceAdapter({
            "vibevoice_model_path": model_path,
            "vibevoice_variant": variant,
            "vibevoice_device": "cpu"
        })
        
        await adapter.initialize()
        
        # Test different languages
        languages = [
            ("en", "Hello, how are you?"),
            ("es", "Hola, ¿cómo estás?"),
            ("fr", "Bonjour, comment allez-vous?"),
            ("de", "Hallo, wie geht es dir?"),
            ("ja", "こんにちは、お元気ですか？"),
            ("zh", "你好，你好吗？")
        ]
        
        for lang, text in languages:
            request = TTSRequest(
                text=text,
                voice="professional",
                language=lang,
                format=AudioFormat.WAV,
                stream=False
            )
            
            response = await adapter.generate(request)
            assert response.audio_data is not None
        
        # Cleanup
        await adapter.close()
    
    @pytest.mark.skipif(
        not check_vibevoice_model_exists()[0],
        reason="Requires VibeVoice model files"
    )
    async def test_different_formats(self):
        """Test generation with different audio formats"""
        model_exists, model_path, variant = check_vibevoice_model_exists()
        
        adapter = VibeVoiceAdapter({
            "vibevoice_model_path": model_path,
            "vibevoice_variant": variant,
            "vibevoice_device": "cpu"
        })
        
        await adapter.initialize()
        
        formats = [AudioFormat.WAV, AudioFormat.MP3, AudioFormat.OPUS, AudioFormat.FLAC]
        
        for audio_format in formats:
            request = TTSRequest(
                text="Format test",
                voice="professional",
                format=audio_format,
                stream=False
            )
            
            response = await adapter.generate(request)
            
            assert response.audio_data is not None
            assert response.format == audio_format
        
        # Cleanup
        await adapter.close()
    
    @pytest.mark.skipif(
        not check_vibevoice_model_exists()[0] or get_compute_capability() == "cpu",
        reason="Requires VibeVoice model and GPU"
    )
    async def test_gpu_acceleration(self):
        """Test GPU acceleration if available"""
        model_exists, model_path, variant = check_vibevoice_model_exists()
        compute = get_compute_capability()
        
        adapter = VibeVoiceAdapter({
            "vibevoice_model_path": model_path,
            "vibevoice_variant": variant,
            "vibevoice_device": compute  # cuda or mps
        })
        
        await adapter.initialize()
        
        assert adapter.device == compute
        
        request = TTSRequest(
            text="GPU acceleration test",
            voice="professional",
            format=AudioFormat.WAV,
            stream=False
        )
        
        response = await adapter.generate(request)
        
        assert response.audio_data is not None
        
        # Cleanup
        await adapter.close()
    
    @pytest.mark.skipif(
        not check_vibevoice_model_exists()[0],
        reason="Requires VibeVoice model files"
    )
    async def test_long_text_generation(self):
        """Test generation with long text"""
        model_exists, model_path, variant = check_vibevoice_model_exists()
        
        adapter = VibeVoiceAdapter({
            "vibevoice_model_path": model_path,
            "vibevoice_variant": variant,
            "vibevoice_device": "cpu"
        })
        
        await adapter.initialize()
        
        # Create text based on variant's context length
        if variant == "1.5B":
            # 1.5B supports up to 6144 chars
            long_text = "This is a test sentence. " * 200  # ~5000 characters
        else:
            # 775M supports up to 4096 chars
            long_text = "This is a test sentence. " * 150  # ~3750 characters
        
        request = TTSRequest(
            text=long_text,
            voice="narrator",
            format=AudioFormat.WAV,
            stream=False
        )
        
        response = await adapter.generate(request)
        
        assert response.audio_data is not None
        assert len(response.audio_data) > 50000  # Should be substantial
        
        # Cleanup
        await adapter.close()
    
    @pytest.mark.skipif(
        not check_vibevoice_model_exists()[0],
        reason="Requires VibeVoice model files"
    )
    async def test_speed_variations(self):
        """Test generation with different speech speeds"""
        model_exists, model_path, variant = check_vibevoice_model_exists()
        
        adapter = VibeVoiceAdapter({
            "vibevoice_model_path": model_path,
            "vibevoice_variant": variant,
            "vibevoice_device": "cpu"
        })
        
        await adapter.initialize()
        
        speeds = [0.75, 1.0, 1.25, 1.5]
        
        for speed in speeds:
            request = TTSRequest(
                text=f"Testing speed {speed}",
                voice="professional",
                speed=speed,
                format=AudioFormat.WAV,
                stream=False
            )
            
            response = await adapter.generate(request)
            
            assert response.audio_data is not None
        
        # Cleanup
        await adapter.close()
    
    @pytest.mark.skipif(
        not check_vibevoice_model_exists()[0],
        reason="Requires VibeVoice model files"
    )
    async def test_concurrent_requests(self):
        """Test handling multiple concurrent requests"""
        model_exists, model_path, variant = check_vibevoice_model_exists()
        
        adapter = VibeVoiceAdapter({
            "vibevoice_model_path": model_path,
            "vibevoice_variant": variant,
            "vibevoice_device": "cpu"
        })
        
        await adapter.initialize()
        
        # Create concurrent requests with different voices
        requests = [
            TTSRequest(
                text=f"Concurrent request {i}",
                voice=["professional", "casual", "narrator"][i % 3],
                format=AudioFormat.WAV,
                stream=False
            )
            for i in range(3)
        ]
        
        # Execute concurrently
        tasks = [adapter.generate(req) for req in requests]
        responses = await asyncio.gather(*tasks)
        
        assert len(responses) == 3
        for response in responses:
            assert response.audio_data is not None
        
        # Cleanup
        await adapter.close()

#######################################################################################################################
#
# End of test_vibevoice_adapter_integration.py