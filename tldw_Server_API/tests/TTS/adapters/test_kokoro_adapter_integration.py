# test_kokoro_adapter_integration.py
# Description: Integration tests for Kokoro TTS adapter
#
# Imports
import pytest
import os
import platform
import torch
import asyncio
#
# Local Imports
from tldw_Server_API.app.core.TTS.adapters.kokoro_adapter import KokoroAdapter
from tldw_Server_API.app.core.TTS.adapters.base import (
    TTSRequest,
    AudioFormat,
    ProviderStatus
)
#
#######################################################################################################################
#
# Helper Functions

def check_kokoro_model_exists():
    """Check if Kokoro model files exist"""
    # Check for common Kokoro model locations
    model_paths = [
        os.path.expanduser("~/.cache/kokoro/kokoro-v0_19.pth"),
        os.path.expanduser("~/.cache/kokoro/kokoro-v0_19.onnx"),
        "./models/kokoro/kokoro-v0_19.pth",
        "./models/kokoro/kokoro-v0_19.onnx"
    ]
    
    for path in model_paths:
        if os.path.exists(path):
            return True, path
    return False, None

def get_compute_capability():
    """Detect compute capabilities"""
    if platform.system() == "Darwin":
        # macOS with Apple Silicon
        if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            return "mps"
    elif torch.cuda.is_available():
        return "cuda"
    return "cpu"

#######################################################################################################################
#
# Integration Tests for Kokoro Adapter

@pytest.mark.integration
@pytest.mark.asyncio
class TestKokoroAdapterIntegration:
    """Integration tests for Kokoro adapter - requires model files"""
    
    @pytest.mark.skipif(
        not check_kokoro_model_exists()[0],
        reason="Requires Kokoro model files to be downloaded"
    )
    async def test_real_model_initialization(self):
        """Test initialization with real model files"""
        model_exists, model_path = check_kokoro_model_exists()
        
        if model_path.endswith('.onnx'):
            adapter = KokoroAdapter({
                "kokoro_use_onnx": True,
                "kokoro_model_path": model_path,
                "kokoro_voices_json": model_path.replace('.onnx', '_voices.json')
            })
        else:
            adapter = KokoroAdapter({
                "kokoro_use_onnx": False,
                "kokoro_model_path": model_path
            })
        
        success = await adapter.initialize()
        assert success
        assert adapter._status == ProviderStatus.AVAILABLE
        
        # Cleanup
        await adapter.close()
    
    @pytest.mark.skipif(
        not check_kokoro_model_exists()[0],
        reason="Requires Kokoro model files"
    )
    async def test_real_audio_generation(self):
        """Test actual audio generation with Kokoro"""
        model_exists, model_path = check_kokoro_model_exists()
        
        if model_path.endswith('.onnx'):
            adapter = KokoroAdapter({
                "kokoro_use_onnx": True,
                "kokoro_model_path": model_path,
                "kokoro_voices_json": model_path.replace('.onnx', '_voices.json'),
                "kokoro_device": "cpu"  # Use CPU for testing
            })
        else:
            adapter = KokoroAdapter({
                "kokoro_use_onnx": False,
                "kokoro_model_path": model_path,
                "kokoro_device": "cpu"
            })
        
        await adapter.initialize()
        
        request = TTSRequest(
            text="Hello, this is a test of the Kokoro text-to-speech system.",
            voice="af_bella",
            format=AudioFormat.WAV,
            stream=False
        )
        
        response = await adapter.generate(request)
        
        assert response.audio_data is not None
        assert len(response.audio_data) > 1000
        assert response.format == AudioFormat.WAV
        assert response.voice_used == "af_bella"
        assert response.provider == "Kokoro"
        
        # Cleanup
        await adapter.close()
    
    @pytest.mark.skipif(
        not check_kokoro_model_exists()[0],
        reason="Requires Kokoro model files"
    )
    async def test_streaming_generation(self):
        """Test streaming audio generation"""
        model_exists, model_path = check_kokoro_model_exists()
        
        adapter = KokoroAdapter({
            "kokoro_use_onnx": model_path.endswith('.onnx'),
            "kokoro_model_path": model_path,
            "kokoro_device": "cpu"
        })
        
        if model_path.endswith('.onnx'):
            adapter.config["kokoro_voices_json"] = model_path.replace('.onnx', '_voices.json')
        
        await adapter.initialize()
        
        request = TTSRequest(
            text="Streaming test.",
            voice="am_adam",
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
        not check_kokoro_model_exists()[0],
        reason="Requires Kokoro model files"
    )
    async def test_different_voices(self):
        """Test generation with different voices"""
        model_exists, model_path = check_kokoro_model_exists()
        
        adapter = KokoroAdapter({
            "kokoro_use_onnx": model_path.endswith('.onnx'),
            "kokoro_model_path": model_path,
            "kokoro_device": "cpu"
        })
        
        if model_path.endswith('.onnx'):
            adapter.config["kokoro_voices_json"] = model_path.replace('.onnx', '_voices.json')
        
        await adapter.initialize()
        
        voices = ["af_bella", "am_adam", "bf_emma", "bm_george"]
        
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
        not check_kokoro_model_exists()[0],
        reason="Requires Kokoro model files"
    )
    async def test_voice_mixing(self):
        """Test voice mixing feature"""
        model_exists, model_path = check_kokoro_model_exists()
        
        adapter = KokoroAdapter({
            "kokoro_use_onnx": model_path.endswith('.onnx'),
            "kokoro_model_path": model_path,
            "kokoro_device": "cpu"
        })
        
        if model_path.endswith('.onnx'):
            adapter.config["kokoro_voices_json"] = model_path.replace('.onnx', '_voices.json')
        
        await adapter.initialize()
        
        # Test mixed voice
        request = TTSRequest(
            text="Testing mixed voices",
            voice="af_bella(2)+am_adam(1)",  # Mix 2 parts Bella with 1 part Adam
            format=AudioFormat.WAV,
            stream=False
        )
        
        response = await adapter.generate(request)
        
        assert response.audio_data is not None
        assert "af_bella" in response.voice_used or "mix" in response.voice_used.lower()
        
        # Cleanup
        await adapter.close()
    
    @pytest.mark.skipif(
        not check_kokoro_model_exists()[0] or get_compute_capability() == "cpu",
        reason="Requires Kokoro model and GPU"
    )
    async def test_gpu_acceleration(self):
        """Test GPU acceleration if available"""
        model_exists, model_path = check_kokoro_model_exists()
        compute = get_compute_capability()
        
        adapter = KokoroAdapter({
            "kokoro_use_onnx": False,  # PyTorch for GPU
            "kokoro_model_path": model_path if model_path.endswith('.pth') else model_path.replace('.onnx', '.pth'),
            "kokoro_device": compute  # cuda or mps
        })
        
        await adapter.initialize()
        
        assert adapter.device == compute
        
        request = TTSRequest(
            text="GPU acceleration test",
            voice="af_bella",
            format=AudioFormat.WAV,
            stream=False
        )
        
        response = await adapter.generate(request)
        
        assert response.audio_data is not None
        
        # Cleanup
        await adapter.close()
    
    @pytest.mark.skipif(
        not check_kokoro_model_exists()[0],
        reason="Requires Kokoro model files"
    )
    async def test_phoneme_generation(self):
        """Test generation with phoneme input"""
        model_exists, model_path = check_kokoro_model_exists()
        
        adapter = KokoroAdapter({
            "kokoro_use_onnx": model_path.endswith('.onnx'),
            "kokoro_model_path": model_path,
            "kokoro_device": "cpu"
        })
        
        if model_path.endswith('.onnx'):
            adapter.config["kokoro_voices_json"] = model_path.replace('.onnx', '_voices.json')
        
        await adapter.initialize()
        
        # Use phonetic text
        request = TTSRequest(
            text="[ˈhɛloʊ ˈwɝld]",
            voice="af_bella",
            format=AudioFormat.WAV,
            stream=False,
            extra_params={"use_phonemes": True}
        )
        
        response = await adapter.generate(request)
        
        assert response.audio_data is not None
        
        # Cleanup
        await adapter.close()
    
    @pytest.mark.skipif(
        not check_kokoro_model_exists()[0],
        reason="Requires Kokoro model files"
    )
    async def test_concurrent_requests(self):
        """Test handling multiple concurrent requests"""
        model_exists, model_path = check_kokoro_model_exists()
        
        adapter = KokoroAdapter({
            "kokoro_use_onnx": model_path.endswith('.onnx'),
            "kokoro_model_path": model_path,
            "kokoro_device": "cpu"
        })
        
        if model_path.endswith('.onnx'):
            adapter.config["kokoro_voices_json"] = model_path.replace('.onnx', '_voices.json')
        
        await adapter.initialize()
        
        # Create concurrent requests
        requests = [
            TTSRequest(
                text=f"Concurrent request {i}",
                voice="af_bella",
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
    
    @pytest.mark.skipif(
        not check_kokoro_model_exists()[0],
        reason="Requires Kokoro model files"
    )
    async def test_format_conversion(self):
        """Test audio format conversion"""
        model_exists, model_path = check_kokoro_model_exists()
        
        adapter = KokoroAdapter({
            "kokoro_use_onnx": model_path.endswith('.onnx'),
            "kokoro_model_path": model_path,
            "kokoro_device": "cpu"
        })
        
        if model_path.endswith('.onnx'):
            adapter.config["kokoro_voices_json"] = model_path.replace('.onnx', '_voices.json')
        
        await adapter.initialize()
        
        # Test MP3 output (requires conversion from WAV)
        request = TTSRequest(
            text="Format conversion test",
            voice="af_bella",
            format=AudioFormat.MP3,
            stream=False
        )
        
        response = await adapter.generate(request)
        
        assert response.audio_data is not None
        assert response.format == AudioFormat.MP3
        
        # Cleanup
        await adapter.close()

#######################################################################################################################
#
# End of test_kokoro_adapter_integration.py