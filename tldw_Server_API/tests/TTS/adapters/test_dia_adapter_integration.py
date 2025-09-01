# test_dia_adapter_integration.py
# Description: Integration tests for Dia TTS adapter
#
# Imports
import pytest
import os
import platform
import torch
import asyncio
#
# Local Imports
from tldw_Server_API.app.core.TTS.adapters.dia_adapter import DiaAdapter
from tldw_Server_API.app.core.TTS.adapters.base import (
    TTSRequest,
    AudioFormat,
    ProviderStatus
)
#
#######################################################################################################################
#
# Helper Functions

def check_dia_model_exists():
    """Check if Dia model is available"""
    # Check for model files or API access
    model_paths = [
        os.path.expanduser("~/.cache/huggingface/hub/models--nari-labs--dia"),
        "./models/dia",
        os.path.expanduser("~/models/dia")
    ]
    
    for path in model_paths:
        if os.path.exists(path):
            return True, path
    
    # Check if HuggingFace token is available for private model
    if os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_TOKEN"):
        return True, "nari-labs/dia"
    
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
# Integration Tests for Dia Adapter

@pytest.mark.integration
@pytest.mark.asyncio
class TestDiaAdapterIntegration:
    """Integration tests for Dia adapter - requires model"""
    
    @pytest.mark.skipif(
        not check_dia_model_exists()[0],
        reason="Requires Dia model from HuggingFace (may need authentication)"
    )
    async def test_real_model_initialization(self):
        """Test initialization with real model"""
        model_exists, model_path = check_dia_model_exists()
        
        adapter = DiaAdapter({
            "dia_model_path": model_path,
            "dia_device": get_compute_capability()
        })
        
        success = await adapter.initialize()
        assert success
        assert adapter._status == ProviderStatus.AVAILABLE
        
        # Cleanup
        await adapter.close()
    
    @pytest.mark.skipif(
        not check_dia_model_exists()[0],
        reason="Requires Dia model"
    )
    async def test_real_audio_generation(self):
        """Test actual audio generation with Dia"""
        model_exists, model_path = check_dia_model_exists()
        
        adapter = DiaAdapter({
            "dia_model_path": model_path,
            "dia_device": "cpu"  # Use CPU for testing
        })
        
        await adapter.initialize()
        
        request = TTSRequest(
            text="Hello, this is a test of the Dia text-to-speech system.",
            voice="speaker1",
            format=AudioFormat.WAV,
            stream=False
        )
        
        response = await adapter.generate(request)
        
        assert response.audio_data is not None
        assert len(response.audio_data) > 1000
        assert response.format == AudioFormat.WAV
        assert response.voice_used == "speaker1"
        assert response.provider == "Dia"
        
        # Cleanup
        await adapter.close()
    
    @pytest.mark.skipif(
        not check_dia_model_exists()[0],
        reason="Requires Dia model"
    )
    async def test_dialogue_generation(self):
        """Test multi-speaker dialogue generation"""
        model_exists, model_path = check_dia_model_exists()
        
        adapter = DiaAdapter({
            "dia_model_path": model_path,
            "dia_device": "cpu"
        })
        
        await adapter.initialize()
        
        request = TTSRequest(
            text="Speaker1: Hello there! Speaker2: Hi! How are you today? Speaker1: I'm doing great, thanks!",
            format=AudioFormat.WAV,
            stream=False
        )
        
        response = await adapter.generate(request)
        
        assert response.audio_data is not None
        assert len(response.audio_data) > 5000  # Dialogue should be longer
        
        # Cleanup
        await adapter.close()
    
    @pytest.mark.skipif(
        not check_dia_model_exists()[0],
        reason="Requires Dia model"
    )
    async def test_streaming_generation(self):
        """Test streaming audio generation"""
        model_exists, model_path = check_dia_model_exists()
        
        adapter = DiaAdapter({
            "dia_model_path": model_path,
            "dia_device": "cpu"
        })
        
        await adapter.initialize()
        
        request = TTSRequest(
            text="Streaming test with Dia.",
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
        not check_dia_model_exists()[0],
        reason="Requires Dia model"
    )
    async def test_different_speakers(self):
        """Test generation with different speakers"""
        model_exists, model_path = check_dia_model_exists()
        
        adapter = DiaAdapter({
            "dia_model_path": model_path,
            "dia_device": "cpu"
        })
        
        await adapter.initialize()
        
        speakers = ["speaker1", "speaker2", "speaker3", "narrator"]
        
        for speaker in speakers:
            request = TTSRequest(
                text=f"Testing {speaker} voice",
                voice=speaker,
                format=AudioFormat.WAV,
                stream=False
            )
            
            response = await adapter.generate(request)
            
            assert response.audio_data is not None
            assert response.voice_used == speaker
        
        # Cleanup
        await adapter.close()
    
    @pytest.mark.skipif(
        not check_dia_model_exists()[0],
        reason="Requires Dia model"
    )
    async def test_different_formats(self):
        """Test generation with different audio formats"""
        model_exists, model_path = check_dia_model_exists()
        
        adapter = DiaAdapter({
            "dia_model_path": model_path,
            "dia_device": "cpu"
        })
        
        await adapter.initialize()
        
        formats = [AudioFormat.WAV, AudioFormat.MP3, AudioFormat.FLAC]
        
        for audio_format in formats:
            request = TTSRequest(
                text="Format test",
                voice="speaker1",
                format=audio_format,
                stream=False
            )
            
            response = await adapter.generate(request)
            
            assert response.audio_data is not None
            assert response.format == audio_format
        
        # Cleanup
        await adapter.close()
    
    @pytest.mark.skipif(
        not check_dia_model_exists()[0] or get_compute_capability() == "cpu",
        reason="Requires Dia model and GPU"
    )
    async def test_gpu_acceleration(self):
        """Test GPU acceleration if available"""
        model_exists, model_path = check_dia_model_exists()
        compute = get_compute_capability()
        
        adapter = DiaAdapter({
            "dia_model_path": model_path,
            "dia_device": compute  # cuda or mps
        })
        
        await adapter.initialize()
        
        assert adapter.device == compute
        
        request = TTSRequest(
            text="GPU acceleration test",
            voice="speaker1",
            format=AudioFormat.WAV,
            stream=False
        )
        
        response = await adapter.generate(request)
        
        assert response.audio_data is not None
        
        # Cleanup
        await adapter.close()
    
    @pytest.mark.skipif(
        not check_dia_model_exists()[0],
        reason="Requires Dia model"
    )
    async def test_long_text_generation(self):
        """Test generation with long text"""
        model_exists, model_path = check_dia_model_exists()
        
        adapter = DiaAdapter({
            "dia_model_path": model_path,
            "dia_device": "cpu"
        })
        
        await adapter.initialize()
        
        # Create long dialogue text
        long_text = "Speaker1: " + ("This is a test sentence. " * 100)  # ~2500 characters
        long_text += " Speaker2: " + ("I agree with that statement. " * 50)
        
        request = TTSRequest(
            text=long_text,
            format=AudioFormat.WAV,
            stream=False
        )
        
        response = await adapter.generate(request)
        
        assert response.audio_data is not None
        assert len(response.audio_data) > 50000  # Should be substantial
        
        # Cleanup
        await adapter.close()
    
    @pytest.mark.skipif(
        not check_dia_model_exists()[0],
        reason="Requires Dia model"
    )
    async def test_concurrent_requests(self):
        """Test handling multiple concurrent requests"""
        model_exists, model_path = check_dia_model_exists()
        
        adapter = DiaAdapter({
            "dia_model_path": model_path,
            "dia_device": "cpu"
        })
        
        await adapter.initialize()
        
        # Create concurrent requests
        requests = [
            TTSRequest(
                text=f"Speaker{i%3+1}: Concurrent request {i}",
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
# End of test_dia_adapter_integration.py