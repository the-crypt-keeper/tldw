# test_tts_integration.py
# Description: Integration tests for TTS providers with real inference
#
# This module contains integration tests that perform actual TTS operations.
# Tests are skipped based on platform, hardware capabilities, and API key availability.
#
# Imports
import asyncio
import os
import platform
import shutil
import sys
from typing import Optional
import pytest
import tempfile
from pathlib import Path

# Check for optional dependencies
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

# Local Imports
from tldw_Server_API.app.core.TTS.adapters.base import TTSRequest, AudioFormat
from tldw_Server_API.app.core.TTS.adapters.openai_adapter import OpenAIAdapter
from tldw_Server_API.app.core.TTS.adapters.elevenlabs_adapter import ElevenLabsAdapter
from tldw_Server_API.app.core.TTS.adapters.kokoro_adapter import KokoroAdapter
from tldw_Server_API.app.core.TTS.adapters.higgs_adapter import HiggsAdapter
from tldw_Server_API.app.core.TTS.adapters.dia_adapter import DiaAdapter
from tldw_Server_API.app.core.TTS.adapters.chatterbox_adapter import ChatterboxAdapter
from tldw_Server_API.app.core.TTS.adapters.vibevoice_adapter import VibeVoiceAdapter

#######################################################################################################################
#
# Platform Detection Utilities

def get_compute_capability() -> str:
    """
    Detect compute capabilities: cuda, mps, or cpu.
    
    Returns:
        String indicating compute capability
    """
    if not TORCH_AVAILABLE:
        return "cpu"
    
    if platform.system() == "Darwin":
        # Check for Apple Silicon MPS
        try:
            if torch.backends.mps.is_available():
                return "mps"
        except:
            pass
        return "cpu"
    elif platform.system() == "Linux":
        # Check for CUDA
        try:
            if torch.cuda.is_available():
                return "cuda"
        except:
            pass
        # Check if nvidia-smi exists
        if shutil.which("nvidia-smi"):
            return "cuda"
    elif platform.system() == "Windows":
        # Check for CUDA on Windows
        try:
            if torch.cuda.is_available():
                return "cuda"
        except:
            pass
    
    return "cpu"


def can_run_local_model(model_name: str) -> bool:
    """
    Check if a local model can run on this system.
    
    Args:
        model_name: Name of the TTS model
        
    Returns:
        Boolean indicating if the model can run
    """
    compute = get_compute_capability()
    
    # Model requirements map: model -> compute -> can_run
    model_requirements = {
        "kokoro": {
            "cpu": True,   # Can run on CPU but slow
            "mps": True,   # Works on Apple Silicon
            "cuda": True   # Best on CUDA
        },
        "higgs": {
            "cpu": False,  # Too slow on CPU
            "mps": True,   # Works on Apple Silicon
            "cuda": True   # Best on CUDA
        },
        "dia": {
            "cpu": False,  # Requires GPU
            "mps": False,  # Not optimized for MPS
            "cuda": True   # CUDA only
        },
        "chatterbox": {
            "cpu": True,   # Can run on CPU
            "mps": True,   # Works on Apple Silicon
            "cuda": True   # Best on CUDA
        },
        "vibevoice": {
            "cpu": False,  # Too slow on CPU
            "mps": True,   # Works on Apple Silicon
            "cuda": True   # Best on CUDA
        }
    }
    
    return model_requirements.get(model_name, {}).get(compute, False)


def has_sufficient_memory(min_gb: float = 4.0) -> bool:
    """
    Check if system has sufficient memory for model loading.
    
    Args:
        min_gb: Minimum GB of RAM required
        
    Returns:
        Boolean indicating sufficient memory
    """
    try:
        import psutil
        mem = psutil.virtual_memory()
        available_gb = mem.available / (1024 ** 3)
        return available_gb >= min_gb
    except ImportError:
        # If psutil not available, assume sufficient memory
        return True


#######################################################################################################################
#
# API-based Integration Tests

@pytest.mark.integration
class TestAPITTSIntegration:
    """Integration tests for API-based TTS providers"""
    
    @pytest.mark.skipif(
        not os.getenv("OPENAI_API_KEY"),
        reason="Requires OPENAI_API_KEY environment variable"
    )
    @pytest.mark.asyncio
    async def test_openai_real_generation(self):
        """Test real OpenAI TTS generation"""
        adapter = OpenAIAdapter({"openai_api_key": os.getenv("OPENAI_API_KEY")})
        await adapter.initialize()
        
        request = TTSRequest(
            text="Hello, this is a test of OpenAI text to speech.",
            voice="nova",
            format=AudioFormat.MP3,
            speed=1.0
        )
        
        response = await adapter.generate(request)
        
        assert response is not None
        assert response.audio is not None
        assert len(response.audio) > 0
        assert response.format == "mp3"
        assert response.provider == "OpenAI"
        
        # Save to temp file and verify it's valid
        with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as f:
            f.write(response.audio)
            temp_path = f.name
        
        # Check file was created and has content
        assert os.path.exists(temp_path)
        assert os.path.getsize(temp_path) > 0
        
        # Cleanup
        os.unlink(temp_path)
        await adapter.close()
    
    @pytest.mark.skipif(
        not os.getenv("ELEVENLABS_API_KEY"),
        reason="Requires ELEVENLABS_API_KEY environment variable"
    )
    @pytest.mark.asyncio
    async def test_elevenlabs_streaming(self):
        """Test real ElevenLabs streaming generation"""
        adapter = ElevenLabsAdapter({"elevenlabs_api_key": os.getenv("ELEVENLABS_API_KEY")})
        await adapter.initialize()
        
        request = TTSRequest(
            text="Testing ElevenLabs streaming capability.",
            voice="rachel",
            format=AudioFormat.MP3,
            stream=True
        )
        
        chunks = []
        async for chunk in adapter.generate_stream(request):
            chunks.append(chunk)
        
        assert len(chunks) > 0
        
        # Combine chunks and verify
        audio_data = b"".join(chunks)
        assert len(audio_data) > 0
        
        await adapter.close()
    
    @pytest.mark.skipif(
        not (os.getenv("OPENAI_API_KEY") and os.getenv("ELEVENLABS_API_KEY")),
        reason="Requires both OPENAI_API_KEY and ELEVENLABS_API_KEY"
    )
    @pytest.mark.asyncio
    async def test_adapter_fallback_real(self):
        """Test real fallback between API providers"""
        # Primary adapter with invalid key to force failure
        primary = OpenAIAdapter({"openai_api_key": "invalid-key"})
        await primary.initialize()
        
        # Fallback adapter with real key
        fallback = ElevenLabsAdapter({"elevenlabs_api_key": os.getenv("ELEVENLABS_API_KEY")})
        await fallback.initialize()
        
        request = TTSRequest(
            text="Testing fallback mechanism.",
            voice="default"
        )
        
        # Try primary (should fail)
        try:
            await primary.generate(request)
            assert False, "Primary should have failed"
        except Exception:
            # Expected to fail
            pass
        
        # Try fallback (should succeed)
        response = await fallback.generate(request)
        assert response is not None
        assert response.provider == "ElevenLabs"
        
        await primary.close()
        await fallback.close()


#######################################################################################################################
#
# Local Model Integration Tests

@pytest.mark.integration
class TestLocalTTSIntegration:
    """Integration tests for local TTS models"""
    
    @pytest.mark.skipif(
        not can_run_local_model("kokoro") or not has_sufficient_memory(2.0),
        reason="Requires appropriate hardware and 2GB+ RAM for Kokoro"
    )
    @pytest.mark.asyncio
    async def test_kokoro_cpu_inference(self):
        """Test Kokoro inference on CPU"""
        compute = get_compute_capability()
        
        adapter = KokoroAdapter({
            "model_type": "onnx",  # ONNX is fastest for CPU
            "device": compute if compute != "mps" else "cpu"  # MPS not supported by all models
        })
        
        # Note: Initialization may download model on first run
        success = await adapter.initialize()
        if not success:
            pytest.skip("Kokoro model initialization failed - may need to download model")
        
        request = TTSRequest(
            text="Testing Kokoro text to speech on local hardware.",
            voice="af_bella",
            format=AudioFormat.WAV
        )
        
        response = await adapter.generate(request)
        
        assert response is not None
        assert response.audio is not None
        assert len(response.audio) > 0
        assert response.format == "wav"
        
        await adapter.close()
    
    @pytest.mark.skipif(
        platform.system() != "Linux" or get_compute_capability() != "cuda",
        reason="Requires Linux with CUDA for optimal Kokoro performance"
    )
    @pytest.mark.asyncio
    async def test_kokoro_gpu_inference(self):
        """Test Kokoro inference with GPU acceleration"""
        adapter = KokoroAdapter({
            "model_type": "phonbert",  # PhonBERT version
            "device": "cuda",
            "use_fp16": True  # Use half precision for faster inference
        })
        
        success = await adapter.initialize()
        if not success:
            pytest.skip("Kokoro model initialization failed")
        
        request = TTSRequest(
            text="Testing GPU accelerated Kokoro inference.",
            voice="af_sarah",
            format=AudioFormat.WAV,
            speed=1.2
        )
        
        # Measure inference time
        import time
        start = time.time()
        response = await adapter.generate(request)
        duration = time.time() - start
        
        assert response is not None
        assert response.audio is not None
        print(f"GPU inference took {duration:.2f} seconds")
        
        await adapter.close()
    
    @pytest.mark.skipif(
        not can_run_local_model("higgs") or not shutil.which("ffmpeg"),
        reason="Requires GPU/MPS and ffmpeg for Higgs voice cloning"
    )
    @pytest.mark.asyncio
    async def test_higgs_voice_cloning(self):
        """Test Higgs voice cloning with real audio"""
        compute = get_compute_capability()
        
        adapter = HiggsAdapter({
            "device": compute if compute != "mps" else "cpu"
        })
        
        success = await adapter.initialize()
        if not success:
            pytest.skip("Higgs model initialization failed")
        
        # Create a simple test audio file for voice reference
        # In real use, this would be a user's voice recording
        import numpy as np
        import wave
        
        sample_rate = 24000
        duration = 5.0
        t = np.linspace(0, duration, int(sample_rate * duration))
        # Generate a 440Hz sine wave (A note)
        audio = np.sin(2 * np.pi * 440 * t) * 0.3
        audio_int16 = (audio * 32767).astype(np.int16)
        
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            with wave.open(f.name, 'wb') as wav_file:
                wav_file.setnchannels(1)
                wav_file.setsampwidth(2)
                wav_file.setframerate(sample_rate)
                wav_file.writeframes(audio_int16.tobytes())
            voice_ref_path = f.name
        
        # Read the voice reference
        with open(voice_ref_path, 'rb') as f:
            voice_ref_data = f.read()
        
        request = TTSRequest(
            text="Testing Higgs voice cloning capability.",
            voice="clone",
            voice_reference=voice_ref_data,
            format=AudioFormat.WAV
        )
        
        response = await adapter.generate(request)
        
        assert response is not None
        assert response.audio is not None
        
        # Cleanup
        os.unlink(voice_ref_path)
        await adapter.close()
    
    @pytest.mark.skipif(
        not can_run_local_model("chatterbox"),
        reason="Requires appropriate hardware for Chatterbox"
    )
    @pytest.mark.asyncio
    async def test_chatterbox_character_voices(self):
        """Test Chatterbox character voice generation"""
        adapter = ChatterboxAdapter({
            "model": "base",  # Use base model for testing
            "device": get_compute_capability()
        })
        
        success = await adapter.initialize()
        if not success:
            pytest.skip("Chatterbox model initialization failed")
        
        # Test different character voices
        voices = ["character_1", "character_2", "narrator"]
        
        for voice in voices:
            request = TTSRequest(
                text=f"Testing {voice} voice style.",
                voice=voice,
                format=AudioFormat.WAV
            )
            
            response = await adapter.generate(request)
            assert response is not None
            assert response.audio is not None
        
        await adapter.close()


#######################################################################################################################
#
# Performance Benchmarking

@pytest.mark.benchmark
@pytest.mark.integration
class TestTTSPerformance:
    """Performance benchmarking for TTS providers"""
    
    @pytest.mark.skipif(
        not can_run_local_model("kokoro"),
        reason="Requires Kokoro model"
    )
    @pytest.mark.asyncio
    async def test_kokoro_performance_benchmark(self, benchmark):
        """Benchmark Kokoro generation speed"""
        adapter = KokoroAdapter({
            "model_type": "onnx",
            "device": get_compute_capability()
        })
        
        success = await adapter.initialize()
        if not success:
            pytest.skip("Kokoro initialization failed")
        
        request = TTSRequest(
            text="The quick brown fox jumps over the lazy dog.",
            voice="af_bella",
            format=AudioFormat.WAV
        )
        
        # Warm up
        await adapter.generate(request)
        
        # Benchmark
        async def generate():
            return await adapter.generate(request)
        
        result = await benchmark(generate)
        assert result is not None
        
        await adapter.close()
    
    @pytest.mark.skipif(
        not os.getenv("OPENAI_API_KEY"),
        reason="Requires OPENAI_API_KEY"
    )
    @pytest.mark.asyncio
    async def test_api_latency_comparison(self):
        """Compare latency between different API providers"""
        import time
        
        results = {}
        
        # Test OpenAI
        if os.getenv("OPENAI_API_KEY"):
            adapter = OpenAIAdapter({"openai_api_key": os.getenv("OPENAI_API_KEY")})
            await adapter.initialize()
            
            request = TTSRequest(
                text="Testing latency.",
                voice="nova",
                format=AudioFormat.MP3
            )
            
            start = time.time()
            await adapter.generate(request)
            results["OpenAI"] = time.time() - start
            await adapter.close()
        
        # Test ElevenLabs
        if os.getenv("ELEVENLABS_API_KEY"):
            adapter = ElevenLabsAdapter({"elevenlabs_api_key": os.getenv("ELEVENLABS_API_KEY")})
            await adapter.initialize()
            
            request = TTSRequest(
                text="Testing latency.",
                voice="rachel",
                format=AudioFormat.MP3
            )
            
            start = time.time()
            await adapter.generate(request)
            results["ElevenLabs"] = time.time() - start
            await adapter.close()
        
        # Print results
        for provider, latency in results.items():
            print(f"{provider}: {latency:.2f}s")
        
        assert len(results) > 0


#######################################################################################################################
#
# Platform-Specific Tests

@pytest.mark.integration
class TestPlatformSpecific:
    """Platform-specific integration tests"""
    
    @pytest.mark.skipif(
        platform.system() != "Darwin",
        reason="macOS specific test"
    )
    @pytest.mark.asyncio
    async def test_macos_mps_acceleration(self):
        """Test MPS acceleration on Apple Silicon"""
        if get_compute_capability() != "mps":
            pytest.skip("MPS not available on this Mac")
        
        # Test with a model that supports MPS
        adapter = KokoroAdapter({
            "device": "mps"
        })
        
        success = await adapter.initialize()
        if not success:
            pytest.skip("Model initialization failed on MPS")
        
        request = TTSRequest(
            text="Testing Metal Performance Shaders acceleration.",
            voice="af_bella"
        )
        
        response = await adapter.generate(request)
        assert response is not None
        
        await adapter.close()
    
    @pytest.mark.skipif(
        platform.system() != "Windows",
        reason="Windows specific test"
    )
    @pytest.mark.asyncio
    async def test_windows_compatibility(self):
        """Test TTS on Windows platform"""
        # Use a simple model for Windows testing
        compute = get_compute_capability()
        
        if can_run_local_model("kokoro"):
            adapter = KokoroAdapter({
                "model_type": "onnx",
                "device": compute
            })
            
            success = await adapter.initialize()
            if success:
                request = TTSRequest(
                    text="Testing on Windows platform.",
                    voice="af_bella"
                )
                
                response = await adapter.generate(request)
                assert response is not None
                
                await adapter.close()
        else:
            pytest.skip("No suitable local model for Windows")
    
    @pytest.mark.skipif(
        platform.system() != "Linux",
        reason="Linux specific test"
    )
    @pytest.mark.asyncio
    async def test_linux_docker_compatibility(self):
        """Test TTS in Docker-like environment (Linux)"""
        # Check if running in container
        is_container = os.path.exists("/.dockerenv") or os.path.exists("/run/.containerenv")
        
        if is_container:
            # Use CPU-only model in container
            adapter = KokoroAdapter({
                "model_type": "onnx",
                "device": "cpu"
            })
        else:
            # Use best available on host
            compute = get_compute_capability()
            adapter = KokoroAdapter({
                "device": compute
            })
        
        success = await adapter.initialize()
        if not success:
            pytest.skip("Model initialization failed in Linux environment")
        
        request = TTSRequest(
            text="Testing in Linux environment.",
            voice="af_bella"
        )
        
        response = await adapter.generate(request)
        assert response is not None
        
        await adapter.close()


if __name__ == "__main__":
    # Run integration tests
    pytest.main([__file__, "-v", "-m", "integration"])