"""
Test script for OpenAI-compatible audio transcription API endpoints.
"""

import asyncio
import io
import os
import tempfile
import numpy as np
import soundfile as sf
from httpx import AsyncClient
import pytest
from fastapi.testclient import TestClient


# Mock audio data for testing
def create_test_audio(duration=1.0, sample_rate=16000):
    """Create a simple test audio file (sine wave)."""
    t = np.linspace(0, duration, int(sample_rate * duration), False)
    frequency = 440  # A4 note
    audio_data = np.sin(frequency * 2 * np.pi * t).astype(np.float32)
    return audio_data, sample_rate


@pytest.mark.asyncio
async def test_transcription_endpoint():
    """Test the /v1/audio/transcriptions endpoint."""
    from tldw_Server_API.app.main import app
    
    # Create test audio
    audio_data, sample_rate = create_test_audio()
    
    # Save to temporary file
    with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp_file:
        sf.write(tmp_file.name, audio_data, sample_rate)
        tmp_path = tmp_file.name
    
    try:
        async with AsyncClient(app=app, base_url="http://test") as client:
            # Read file for upload
            with open(tmp_path, 'rb') as f:
                files = {'file': ('test.wav', f, 'audio/wav')}
                data = {
                    'model': 'whisper-1',
                    'response_format': 'json'
                }
                
                response = await client.post(
                    "/api/v1/audio/transcriptions",
                    files=files,
                    data=data
                )
            
            assert response.status_code == 200
            result = response.json()
            assert 'text' in result
            print(f"Transcription result: {result}")
    
    finally:
        # Clean up
        if os.path.exists(tmp_path):
            os.remove(tmp_path)


@pytest.mark.asyncio
async def test_transcription_with_parakeet():
    """Test transcription using Parakeet model."""
    from tldw_Server_API.app.main import app
    
    # Create test audio
    audio_data, sample_rate = create_test_audio(duration=2.0)
    
    # Save to temporary file
    with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp_file:
        sf.write(tmp_file.name, audio_data, sample_rate)
        tmp_path = tmp_file.name
    
    try:
        async with AsyncClient(app=app, base_url="http://test") as client:
            with open(tmp_path, 'rb') as f:
                files = {'file': ('test.wav', f, 'audio/wav')}
                data = {
                    'model': 'parakeet',
                    'response_format': 'json',
                    'language': 'en'
                }
                
                response = await client.post(
                    "/api/v1/audio/transcriptions",
                    files=files,
                    data=data
                )
            
            # Parakeet might not be available in test environment
            if response.status_code == 200:
                result = response.json()
                assert 'text' in result
                print(f"Parakeet transcription: {result}")
            else:
                print(f"Parakeet not available: {response.status_code}")
    
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)


@pytest.mark.asyncio
async def test_transcription_formats():
    """Test different response formats."""
    from tldw_Server_API.app.main import app
    
    # Create test audio
    audio_data, sample_rate = create_test_audio()
    
    # Save to temporary file
    with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp_file:
        sf.write(tmp_file.name, audio_data, sample_rate)
        tmp_path = tmp_file.name
    
    try:
        async with AsyncClient(app=app, base_url="http://test") as client:
            # Test different formats
            formats = ['json', 'text', 'srt', 'vtt', 'verbose_json']
            
            for fmt in formats:
                with open(tmp_path, 'rb') as f:
                    files = {'file': ('test.wav', f, 'audio/wav')}
                    data = {
                        'model': 'whisper-1',
                        'response_format': fmt
                    }
                    
                    response = await client.post(
                        "/api/v1/audio/transcriptions",
                        files=files,
                        data=data
                    )
                
                assert response.status_code == 200
                
                if fmt == 'json' or fmt == 'verbose_json':
                    result = response.json()
                    assert 'text' in result
                    if fmt == 'verbose_json':
                        assert 'task' in result
                        assert 'duration' in result
                elif fmt == 'text':
                    assert isinstance(response.text, str)
                elif fmt == 'srt':
                    assert '00:00:00,000' in response.text
                elif fmt == 'vtt':
                    assert 'WEBVTT' in response.text
                
                print(f"Format {fmt} test passed")
    
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)


@pytest.mark.asyncio
async def test_translation_endpoint():
    """Test the /v1/audio/translations endpoint."""
    from tldw_Server_API.app.main import app
    
    # Create test audio
    audio_data, sample_rate = create_test_audio()
    
    # Save to temporary file
    with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp_file:
        sf.write(tmp_file.name, audio_data, sample_rate)
        tmp_path = tmp_file.name
    
    try:
        async with AsyncClient(app=app, base_url="http://test") as client:
            with open(tmp_path, 'rb') as f:
                files = {'file': ('test.wav', f, 'audio/wav')}
                data = {
                    'model': 'whisper-1',
                    'response_format': 'json'
                }
                
                response = await client.post(
                    "/api/v1/audio/translations",
                    files=files,
                    data=data
                )
            
            assert response.status_code == 200
            result = response.json()
            assert 'text' in result
            print(f"Translation result: {result}")
    
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)


def test_sync_transcription():
    """Test synchronous transcription using TestClient."""
    from tldw_Server_API.app.main import app
    
    client = TestClient(app)
    
    # Create test audio
    audio_data, sample_rate = create_test_audio()
    
    # Create in-memory file
    audio_buffer = io.BytesIO()
    sf.write(audio_buffer, audio_data, sample_rate, format='WAV')
    audio_buffer.seek(0)
    
    # Upload file
    files = {'file': ('test.wav', audio_buffer, 'audio/wav')}
    data = {
        'model': 'whisper-1',
        'response_format': 'json'
    }
    
    response = client.post(
        "/api/v1/audio/transcriptions",
        files=files,
        data=data
    )
    
    # Check response (might fail if authentication is required)
    if response.status_code == 401:
        print("Authentication required for transcription endpoint")
    else:
        assert response.status_code == 200
        result = response.json()
        assert 'text' in result
        print(f"Sync transcription result: {result}")


# Example usage with curl commands
def print_curl_examples():
    """Print example curl commands for testing the API."""
    print("""
    # Example curl commands for testing the transcription API:
    
    # 1. Basic transcription with Whisper
    curl -X POST "http://localhost:8000/api/v1/audio/transcriptions" \\
      -H "Authorization: Bearer YOUR_API_TOKEN" \\
      -F "file=@audio.wav" \\
      -F "model=whisper-1" \\
      -F "response_format=json"
    
    # 2. Transcription with Parakeet
    curl -X POST "http://localhost:8000/api/v1/audio/transcriptions" \\
      -H "Authorization: Bearer YOUR_API_TOKEN" \\
      -F "file=@audio.wav" \\
      -F "model=parakeet" \\
      -F "response_format=json" \\
      -F "language=en"
    
    # 3. Transcription with Canary (multilingual)
    curl -X POST "http://localhost:8000/api/v1/audio/transcriptions" \\
      -H "Authorization: Bearer YOUR_API_TOKEN" \\
      -F "file=@audio.wav" \\
      -F "model=canary" \\
      -F "response_format=json" \\
      -F "language=es"  # Spanish
    
    # 4. Get transcription in SRT format
    curl -X POST "http://localhost:8000/api/v1/audio/transcriptions" \\
      -H "Authorization: Bearer YOUR_API_TOKEN" \\
      -F "file=@audio.wav" \\
      -F "model=whisper-1" \\
      -F "response_format=srt"
    
    # 5. Translation to English
    curl -X POST "http://localhost:8000/api/v1/audio/translations" \\
      -H "Authorization: Bearer YOUR_API_TOKEN" \\
      -F "file=@foreign_audio.wav" \\
      -F "model=whisper-1" \\
      -F "response_format=json"
    
    # Using with OpenAI Python client:
    from openai import OpenAI
    
    client = OpenAI(
        base_url="http://localhost:8000/api/v1",
        api_key="YOUR_API_TOKEN"
    )
    
    # Transcription
    with open("audio.wav", "rb") as audio_file:
        transcript = client.audio.transcriptions.create(
            model="whisper-1",  # or "parakeet", "canary"
            file=audio_file,
            response_format="json"
        )
        print(transcript.text)
    
    # Translation
    with open("foreign_audio.wav", "rb") as audio_file:
        translation = client.audio.translations.create(
            model="whisper-1",
            file=audio_file
        )
        print(translation.text)
    """)


if __name__ == "__main__":
    # Print examples
    print_curl_examples()
    
    # Run basic test
    test_sync_transcription()
    
    # Run async tests
    asyncio.run(test_transcription_endpoint())
    asyncio.run(test_transcription_formats())