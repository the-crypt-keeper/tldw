# test_chatterbox_adapter_integration.py
# Description: Integration tests for Chatterbox TTS adapter
#
# Imports
import pytest
import os
import platform
import torch
import asyncio
#
# Local Imports
from tldw_Server_API.app.core.TTS.adapters.chatterbox_adapter import ChatterboxAdapter
from tldw_Server_API.app.core.TTS.adapters.base import (
    TTSRequest,
    AudioFormat,
    ProviderStatus
)
#
#######################################################################################################################
#
# Helper Functions

def check_chatterbox_installed():
    """Check if Chatterbox library and model are available"""
    try:
        # Try to import chatterbox-tts
        import chatterbox_tts
        
        # Check for model files
        model_paths = [
            os.path.expanduser("~/.cache/chatterbox"),
            "./models/chatterbox",
            os.path.expanduser("~/models/chatterbox")
        ]
        
        for path in model_paths:
            if os.path.exists(path):
                return True, path
        
        # Library exists but no model
        return False, None
    except ImportError:
        return False, None

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
# Integration Tests for Chatterbox Adapter

@pytest.mark.integration
@pytest.mark.asyncio
class TestChatterboxAdapterIntegration:
    """Integration tests for Chatterbox adapter - requires model and library"""
    
    @pytest.mark.skipif(
        not check_chatterbox_installed()[0],
        reason="Requires Chatterbox library and model files"
    )
    async def test_real_model_initialization(self):
        """Test initialization with real model"""
        installed, model_path = check_chatterbox_installed()
        
        adapter = ChatterboxAdapter({
            "chatterbox_model_path": model_path,
            "chatterbox_device": get_compute_capability()
        })
        
        success = await adapter.initialize()
        assert success
        assert adapter._status == ProviderStatus.AVAILABLE
        
        # Cleanup
        await adapter.close()
    
    @pytest.mark.skipif(
        not check_chatterbox_installed()[0],
        reason="Requires Chatterbox model"
    )
    async def test_real_audio_generation(self):
        """Test actual audio generation with Chatterbox"""
        installed, model_path = check_chatterbox_installed()
        
        adapter = ChatterboxAdapter({
            "chatterbox_model_path": model_path,
            "chatterbox_device": "cpu"  # Use CPU for testing
        })
        
        await adapter.initialize()
        
        request = TTSRequest(
            text="Hello, this is a test of the Chatterbox text-to-speech system.",
            voice="narrator",
            format=AudioFormat.WAV,
            stream=False
        )
        
        response = await adapter.generate(request)
        
        assert response.audio_data is not None
        assert len(response.audio_data) > 1000
        assert response.format == AudioFormat.WAV
        assert response.voice_used == "narrator"
        assert response.provider == "Chatterbox"
        
        # Cleanup
        await adapter.close()
    
    @pytest.mark.skipif(
        not check_chatterbox_installed()[0],
        reason="Requires Chatterbox model"
    )
    async def test_character_voice_generation(self):
        """Test generation with different character voices"""
        installed, model_path = check_chatterbox_installed()
        
        adapter = ChatterboxAdapter({
            "chatterbox_model_path": model_path,
            "chatterbox_device": "cpu"
        })
        
        await adapter.initialize()
        
        voices = ["narrator", "hero", "villain", "sidekick", "sage"]
        
        for voice in voices:
            request = TTSRequest(
                text=f"Testing the {voice} character voice",
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
        not check_chatterbox_installed()[0],
        reason="Requires Chatterbox model"
    )
    async def test_streaming_generation(self):
        """Test streaming audio generation"""
        installed, model_path = check_chatterbox_installed()
        
        adapter = ChatterboxAdapter({
            "chatterbox_model_path": model_path,
            "chatterbox_device": "cpu"
        })
        
        await adapter.initialize()
        
        request = TTSRequest(
            text="Streaming test with Chatterbox.",
            voice="narrator",
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
        not check_chatterbox_installed()[0],
        reason="Requires Chatterbox model"
    )
    async def test_emotion_control(self):
        """Test emotion control in generation"""
        installed, model_path = check_chatterbox_installed()
        
        adapter = ChatterboxAdapter({
            "chatterbox_model_path": model_path,
            "chatterbox_device": "cpu"
        })
        
        await adapter.initialize()
        
        emotions = ["happy", "sad", "angry", "excited", "calm"]
        
        for emotion in emotions:
            request = TTSRequest(
                text=f"Speaking with {emotion} emotion",
                voice="hero",
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
        not check_chatterbox_installed()[0],
        reason="Requires Chatterbox model"
    )
    async def test_style_variations(self):
        """Test different speech styles"""
        installed, model_path = check_chatterbox_installed()
        
        adapter = ChatterboxAdapter({
            "chatterbox_model_path": model_path,
            "chatterbox_device": "cpu"
        })
        
        await adapter.initialize()
        
        styles = ["dramatic", "casual", "formal", "whisper", "shout"]
        
        for style in styles:
            request = TTSRequest(
                text=f"Speaking in {style} style",
                voice="narrator",
                style=style,
                format=AudioFormat.WAV,
                stream=False
            )
            
            response = await adapter.generate(request)
            
            assert response.audio_data is not None
        
        # Cleanup
        await adapter.close()
    
    @pytest.mark.skipif(
        not check_chatterbox_installed()[0],
        reason="Requires Chatterbox model"
    )
    async def test_speed_variations(self):
        """Test different speech speeds"""
        installed, model_path = check_chatterbox_installed()
        
        adapter = ChatterboxAdapter({
            "chatterbox_model_path": model_path,
            "chatterbox_device": "cpu"
        })
        
        await adapter.initialize()
        
        speeds = [0.5, 1.0, 1.5, 2.0]
        
        for speed in speeds:
            request = TTSRequest(
                text=f"Testing speed {speed}",
                voice="narrator",
                speed=speed,
                format=AudioFormat.WAV,
                stream=False
            )
            
            response = await adapter.generate(request)
            
            assert response.audio_data is not None
        
        # Cleanup
        await adapter.close()
    
    @pytest.mark.skipif(
        not check_chatterbox_installed()[0] or get_compute_capability() == "cpu",
        reason="Requires Chatterbox model and GPU"
    )
    async def test_gpu_acceleration(self):
        """Test GPU acceleration if available"""
        installed, model_path = check_chatterbox_installed()
        compute = get_compute_capability()
        
        adapter = ChatterboxAdapter({
            "chatterbox_model_path": model_path,
            "chatterbox_device": compute  # cuda or mps
        })
        
        await adapter.initialize()
        
        assert adapter.device == compute
        
        request = TTSRequest(
            text="GPU acceleration test",
            voice="narrator",
            format=AudioFormat.WAV,
            stream=False
        )
        
        response = await adapter.generate(request)
        
        assert response.audio_data is not None
        
        # Cleanup
        await adapter.close()
    
    @pytest.mark.skipif(
        not check_chatterbox_installed()[0],
        reason="Requires Chatterbox model"
    )
    async def test_character_dialogue(self):
        """Test character dialogue generation"""
        installed, model_path = check_chatterbox_installed()
        
        adapter = ChatterboxAdapter({
            "chatterbox_model_path": model_path,
            "chatterbox_device": "cpu"
        })
        
        await adapter.initialize()
        
        dialogue = """
        Narrator: Once upon a time in a distant land...
        Hero: I must find the ancient artifact!
        Villain: You'll never succeed, foolish hero!
        Hero: We'll see about that!
        Narrator: And so the battle began...
        """
        
        request = TTSRequest(
            text=dialogue,
            format=AudioFormat.WAV,
            stream=False
        )
        
        response = await adapter.generate(request)
        
        assert response.audio_data is not None
        assert len(response.audio_data) > 10000  # Dialogue should be substantial
        
        # Cleanup
        await adapter.close()
    
    @pytest.mark.skipif(
        not check_chatterbox_installed()[0],
        reason="Requires Chatterbox model"
    )
    async def test_concurrent_requests(self):
        """Test handling multiple concurrent requests"""
        installed, model_path = check_chatterbox_installed()
        
        adapter = ChatterboxAdapter({
            "chatterbox_model_path": model_path,
            "chatterbox_device": "cpu"
        })
        
        await adapter.initialize()
        
        # Create concurrent requests with different voices
        requests = [
            TTSRequest(
                text=f"Request {i} with voice",
                voice=["narrator", "hero", "villain"][i % 3],
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
# End of test_chatterbox_adapter_integration.py