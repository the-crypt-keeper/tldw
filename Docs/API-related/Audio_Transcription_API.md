# Audio Transcription API Documentation

## Overview

The tldw_server provides a comprehensive audio transcription API that is fully compatible with OpenAI's Audio API while offering additional transcription engines including NVIDIA Nemo models (Canary and Parakeet) for improved performance and flexibility.

## Table of Contents
- [Features](#features)
- [Supported Models](#supported-models)
- [API Endpoints](#api-endpoints)
- [Configuration](#configuration)
- [Live Transcription](#live-transcription)
- [Usage Examples](#usage-examples)
- [Performance Comparison](#performance-comparison)

## Features

### Core Capabilities
- **OpenAI API Compatible**: Drop-in replacement for OpenAI's audio transcription endpoints
- **Multiple Transcription Engines**: Choose from faster-whisper, NVIDIA Nemo models, or Qwen2Audio
- **Live Transcription**: Real-time audio streaming with VAD and silence detection
- **Model Optimization**: Support for ONNX and MLX variants for better performance
- **Multi-format Support**: Handle various audio formats (WAV, MP3, M4A, etc.)
- **Response Formats**: JSON, text, SRT, VTT, verbose JSON

### Advanced Features
- **Voice Activity Detection (VAD)**: Intelligent speech segmentation
- **Streaming Support**: Process long audio files efficiently
- **Language Detection**: Automatic language identification
- **Partial Transcriptions**: Get interim results during live transcription
- **Model Caching**: Efficient model management for repeated use

## Supported Models

### 1. Whisper (faster-whisper)
- **Model**: `whisper-1` (OpenAI compatible name)
- **Variants**: tiny, base, small, medium, large-v3
- **Languages**: 99+ languages
- **Best For**: General-purpose transcription, multi-language support

### 2. NVIDIA Canary-1b
- **Model**: `canary`
- **Size**: 1 billion parameters
- **Languages**: English, Spanish, German, French
- **Best For**: Multi-lingual transcription with high accuracy
- **Special Features**: Built-in punctuation and capitalization

### 3. NVIDIA Parakeet TDT
- **Model**: `parakeet`
- **Size**: 0.6 billion parameters
- **Variants**: 
  - Standard (PyTorch)
  - ONNX (optimized for CPU/GPU)
  - MLX (optimized for Apple Silicon)
- **Languages**: English (primarily)
- **Best For**: Fast, efficient transcription with good accuracy

### 4. Qwen2Audio
- **Model**: `qwen2audio`
- **Size**: 7 billion parameters
- **Languages**: Multiple languages
- **Best For**: Complex audio understanding tasks

## API Endpoints

### POST /v1/audio/transcriptions

Transcribe audio into text.

**Request Parameters:**

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| file | file | Yes | The audio file to transcribe (max 25MB) |
| model | string | No | Model to use: `whisper-1`, `parakeet`, `canary`, `qwen2audio` (default: `whisper-1`) |
| language | string | No | Language code in ISO-639-1 format (e.g., 'en', 'es') |
| prompt | string | No | Optional text to guide the model's style |
| response_format | string | No | Output format: `json`, `text`, `srt`, `vtt`, `verbose_json` (default: `json`) |
| temperature | float | No | Sampling temperature 0-1 (default: 0) |
| timestamp_granularities | array | No | Timestamp levels: `segment`, `word` |

**Response (JSON format):**
```json
{
  "text": "Transcribed text here",
  "language": "en",
  "duration": 10.5,
  "segments": [
    {
      "id": 0,
      "start": 0.0,
      "end": 10.5,
      "text": "Transcribed text here"
    }
  ]
}
```

### POST /v1/audio/translations

Translate audio into English.

**Request Parameters:**

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| file | file | Yes | The audio file to translate |
| model | string | No | Model to use (default: `whisper-1`) |
| prompt | string | No | Optional text to guide the model's style |
| response_format | string | No | Output format (default: `json`) |
| temperature | float | No | Sampling temperature 0-1 |

## Configuration

### config.txt Settings

Add the following section to your `config.txt`:

```ini
[STT-Settings]
# Default transcription provider
default_transcriber = faster-whisper
# Options: faster-whisper, parakeet, canary, qwen2audio

# Nemo model variant (for Parakeet)
nemo_model_variant = standard
# Options: standard, onnx, mlx

# Device for Nemo models
nemo_device = cuda
# Options: cpu, cuda

# Cache directory for downloaded models
nemo_cache_dir = ./models/nemo
```

### Environment Variables

You can override config settings with environment variables:
- `TLDW_DEFAULT_TRANSCRIBER`: Set default transcription provider
- `TLDW_NEMO_DEVICE`: Set device for Nemo models
- `TLDW_NEMO_CACHE_DIR`: Set model cache directory

## Live Transcription

### Basic Live Transcription

```python
from tldw_Server_API.app.core.Ingestion_Media_Processing.Audio.Audio_Live_Transcription_Nemo import (
    create_live_transcriber
)

# Create transcriber with callbacks
def on_transcription(text):
    print(f"Final: {text}")

def on_partial(text):
    print(f"Partial: {text}")

transcriber = create_live_transcriber(
    model='parakeet',
    mode='silence_based',
    on_transcription=on_transcription,
    on_partial=on_partial
)

# Start transcription
transcriber.start()
# ... speak into microphone ...
transcriber.stop()
```

### Streaming File Transcription

```python
from tldw_Server_API.app.core.Ingestion_Media_Processing.Audio.Audio_Live_Transcription_Nemo import (
    NemoStreamingTranscriber
)

# Create streaming transcriber
transcriber = NemoStreamingTranscriber(
    model='parakeet',
    variant='onnx',
    chunk_duration=5.0
)

# Initialize with sample rate
transcriber.initialize(sample_rate=16000)

# Process audio chunks
for chunk in audio_chunks:
    text = transcriber.process_chunk(chunk)
    if text:
        print(f"Transcribed: {text}")

# Get complete transcription
full_text = transcriber.get_full_transcription()
```

### Transcription Modes

1. **Continuous Mode**: Process audio continuously without pause detection
2. **VAD-Based Mode**: Use Voice Activity Detection for intelligent segmentation
3. **Silence-Based Mode**: Simple amplitude-based silence detection (default)

## Usage Examples

### Using curl

```bash
# Basic transcription with Whisper
curl -X POST "http://localhost:8000/api/v1/audio/transcriptions" \
  -H "Authorization: Bearer YOUR_API_TOKEN" \
  -F "file=@audio.wav" \
  -F "model=whisper-1" \
  -F "response_format=json"

# Fast transcription with Parakeet
curl -X POST "http://localhost:8000/api/v1/audio/transcriptions" \
  -H "Authorization: Bearer YOUR_API_TOKEN" \
  -F "file=@audio.wav" \
  -F "model=parakeet" \
  -F "response_format=json"

# Multi-lingual with Canary (Spanish)
curl -X POST "http://localhost:8000/api/v1/audio/transcriptions" \
  -H "Authorization: Bearer YOUR_API_TOKEN" \
  -F "file=@spanish_audio.wav" \
  -F "model=canary" \
  -F "language=es"

# Get SRT subtitles
curl -X POST "http://localhost:8000/api/v1/audio/transcriptions" \
  -H "Authorization: Bearer YOUR_API_TOKEN" \
  -F "file=@video_audio.wav" \
  -F "model=whisper-1" \
  -F "response_format=srt"
```

### Using Python (OpenAI Client)

```python
from openai import OpenAI

# Configure client to use tldw_server
client = OpenAI(
    base_url="http://localhost:8000/api/v1",
    api_key="YOUR_API_TOKEN"
)

# Basic transcription
with open("audio.wav", "rb") as audio_file:
    transcript = client.audio.transcriptions.create(
        model="whisper-1",
        file=audio_file,
        response_format="json"
    )
    print(transcript.text)

# Using Parakeet for faster transcription
with open("audio.wav", "rb") as audio_file:
    transcript = client.audio.transcriptions.create(
        model="parakeet",
        file=audio_file,
        response_format="json"
    )
    print(transcript.text)

# Multi-lingual transcription with Canary
with open("spanish_audio.wav", "rb") as audio_file:
    transcript = client.audio.transcriptions.create(
        model="canary",
        file=audio_file,
        language="es",
        response_format="verbose_json"
    )
    print(f"Language: {transcript.language}")
    print(f"Text: {transcript.text}")
    print(f"Duration: {transcript.duration}")

# Translation to English
with open("foreign_audio.wav", "rb") as audio_file:
    translation = client.audio.translations.create(
        model="whisper-1",
        file=audio_file
    )
    print(translation.text)
```

### Using Python (Direct API)

```python
import requests

# Transcribe with Parakeet
url = "http://localhost:8000/api/v1/audio/transcriptions"
headers = {"Authorization": "Bearer YOUR_API_TOKEN"}

with open("audio.wav", "rb") as f:
    files = {"file": ("audio.wav", f, "audio/wav")}
    data = {
        "model": "parakeet",
        "response_format": "json"
    }
    
    response = requests.post(url, headers=headers, files=files, data=data)
    result = response.json()
    print(result["text"])
```

### Live Transcription Example

```python
from tldw_Server_API.app.core.Ingestion_Media_Processing.Audio.Audio_Transcription_Lib import (
    LiveAudioStreamer
)

# Configure for Parakeet with ONNX
streamer = LiveAudioStreamer(
    transcription_provider='parakeet',
    nemo_variant='onnx',
    silence_threshold=0.01,
    silence_duration=1.5
)

# Custom handler for transcribed text
def handle_text(text):
    print(f"Transcribed: {text}")
    # Process text (save, send to chat, etc.)

streamer.handle_transcribed_text = handle_text

# Start live transcription
streamer.start()
print("Listening... Press Ctrl+C to stop")

try:
    import time
    while True:
        time.sleep(0.1)
except KeyboardInterrupt:
    streamer.stop()
    print("Stopped")
```

## Performance Comparison

### Speed Comparison (Relative to Real-Time)

| Model | Speed | Accuracy | Memory Usage | Best Use Case |
|-------|-------|----------|--------------|---------------|
| Whisper (tiny) | 10-15x | Good | 1GB | Quick drafts |
| Whisper (base) | 8-12x | Better | 1.5GB | General use |
| Whisper (large-v3) | 2-4x | Best | 10GB | High accuracy |
| Parakeet (standard) | 15-20x | Very Good | 2GB | Fast transcription |
| Parakeet (ONNX) | 20-30x | Very Good | 1.5GB | CPU optimization |
| Parakeet (MLX) | 25-35x | Very Good | 1.5GB | Apple Silicon |
| Canary-1b | 8-12x | Excellent | 4GB | Multi-lingual |
| Qwen2Audio | 1-2x | Excellent | 14GB | Complex audio |

### Recommendations

1. **For Speed**: Use Parakeet with ONNX or MLX variant
2. **For Accuracy**: Use Whisper large-v3 or Canary
3. **For Multi-lingual**: Use Canary (4 languages) or Whisper (99+ languages)
4. **For Live Transcription**: Use Parakeet with VAD mode
5. **For Resource-Constrained**: Use Parakeet ONNX or Whisper tiny

## Troubleshooting

### Common Issues

1. **Model Download Fails**
   - Check internet connection
   - Ensure sufficient disk space in cache directory
   - Try manual download from Hugging Face

2. **CUDA Out of Memory**
   - Use smaller model variant
   - Set `nemo_device = cpu` in config
   - Use ONNX variant for better memory efficiency

3. **Slow Transcription**
   - Use Parakeet instead of Whisper
   - Enable GPU acceleration (`nemo_device = cuda`)
   - Use ONNX or MLX variants

4. **Poor Accuracy**
   - Use larger model (Whisper large-v3 or Canary)
   - Specify correct language parameter
   - Provide prompt for context

### Debug Logging

Enable debug logging for troubleshooting:

```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

## API Rate Limits

- Transcription endpoint: 20 requests/minute per IP
- Translation endpoint: 20 requests/minute per IP
- File size limit: 25MB per request

## Security Considerations

1. **Authentication**: Always use Bearer token authentication in production
2. **File Validation**: The API validates file types and sizes
3. **Rate Limiting**: Built-in protection against abuse
4. **Input Sanitization**: All inputs are validated and sanitized

## Future Enhancements

- [ ] Batch transcription API
- [ ] WebSocket support for live transcription
- [ ] Speaker diarization with Nemo models
- [ ] Custom vocabulary support
- [ ] Fine-tuning support for domain-specific transcription
- [ ] Multi-GPU support for parallel processing

## Related Documentation

- [API Overview](./API_Overview.md)
- [Configuration Guide](../User_Guides/Configuration.md)
- [Live Transcription Guide](../User_Guides/Live_Transcription.md)
- [Model Selection Guide](../User_Guides/Model_Selection.md)