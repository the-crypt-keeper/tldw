# TTS Module Deployment Guide

## Overview
This guide provides instructions for deploying and configuring the Text-to-Speech (TTS) module in the tldw_server API.

## Prerequisites

### System Requirements
- Python 3.8+
- FFmpeg (for audio processing)
- 2GB+ RAM for API backends
- 4GB+ RAM if using local Kokoro models
- (Optional) NVIDIA GPU for faster local inference

### Required Dependencies
```bash
# Install Python dependencies
pip install -r tldw_Server_API/requirements.txt

# Key dependencies for TTS:
# - av (pyav) - Audio encoding/streaming
# - httpx - Async HTTP client for API calls
# - numpy - Audio data processing
# - kokoro_onnx - Local TTS model support
# - scipy - Audio file I/O
```

### Environment Setup
```bash
# For Kokoro ONNX (if using local TTS)
export PHONEMIZER_ESPEAK_LIBRARY=/path/to/libespeak-ng.so  # Linux
export PHONEMIZER_ESPEAK_LIBRARY=/usr/local/lib/libespeak-ng.dylib  # macOS

# For OpenAI (if not in config file)
export OPENAI_API_KEY=your-api-key-here

# For ElevenLabs (if not in config file)
export ELEVENLABS_API_KEY=your-api-key-here
```

## Configuration

### 1. API Keys Configuration

Edit `config.txt` or set environment variables:

```ini
[API]
# OpenAI TTS
openai_api_key = sk-...your-key-here...
openai_model = tts-1  # or tts-1-hd for higher quality

# ElevenLabs (optional)
elevenlabs_api_key = your-elevenlabs-key-here

# Custom OpenAI-compatible endpoint (optional)
custom_openai_api_key = your-custom-key
custom_openai_api_ip = https://your-api-endpoint.com
```

### 2. TTS-Specific Settings

Add to `config.txt`:

```ini
[TTS]
# Default TTS provider (openai, kokoro, elevenlabs)
default_provider = openai

# Kokoro local model settings
kokoro_model_path = models/kokoro-v0_19.onnx
kokoro_voices_json = models/voices.json
kokoro_device = cpu  # or cuda for GPU

# Rate limiting (requests per minute)
tts_rate_limit = 10

# Maximum input text length
max_input_length = 4096

# Supported audio formats
supported_formats = mp3,wav,opus,flac,pcm

# Default audio format
default_format = mp3
```

### 3. Authentication Setup

The TTS module uses the existing authentication system:

```ini
[Authentication]
# Set to true to require authentication
require_auth = true

# API token for service accounts
api_token = your-secure-token-here
```

## Deployment Steps

### 1. Basic Deployment

```bash
# Clone repository
git clone https://github.com/your-repo/tldw_server.git
cd tldw_server

# Install dependencies
pip install -r tldw_Server_API/requirements.txt

# Configure API keys
cp Config_Files/Backup_Config.txt config.txt
# Edit config.txt with your API keys

# Start the server
python -m uvicorn tldw_Server_API.app.main:app --host 0.0.0.0 --port 8000
```

### 2. Docker Deployment

```dockerfile
# Dockerfile
FROM python:3.10-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    ffmpeg \
    espeak-ng \
    libespeak-ng1 \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements
COPY tldw_Server_API/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application
COPY tldw_Server_API/ ./tldw_Server_API/
COPY config.txt .

# Set environment variables
ENV PHONEMIZER_ESPEAK_LIBRARY=/usr/lib/x86_64-linux-gnu/libespeak-ng.so.1

# Expose port
EXPOSE 8000

# Run application
CMD ["uvicorn", "tldw_Server_API.app.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

Build and run:
```bash
docker build -t tldw-tts .
docker run -p 8000:8000 -v $(pwd)/config.txt:/app/config.txt tldw-tts
```

### 3. Production Deployment with Gunicorn

```bash
# Install gunicorn
pip install gunicorn

# Run with multiple workers
gunicorn tldw_Server_API.app.main:app \
    --workers 4 \
    --worker-class uvicorn.workers.UvicornWorker \
    --bind 0.0.0.0:8000 \
    --timeout 300 \
    --access-logfile - \
    --error-logfile -
```

### 4. Nginx Reverse Proxy Configuration

```nginx
server {
    listen 80;
    server_name your-domain.com;

    location /api/v1/audio/ {
        proxy_pass http://localhost:8000;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection 'upgrade';
        proxy_set_header Host $host;
        proxy_cache_bypass $http_upgrade;
        
        # Important for streaming
        proxy_buffering off;
        proxy_set_header X-Accel-Buffering no;
        
        # Timeout for long TTS generation
        proxy_read_timeout 300s;
        proxy_connect_timeout 75s;
    }
}
```

## Testing the Deployment

### 1. Health Check
```bash
# Check if service is running
curl http://localhost:8000/health
```

### 2. Test TTS Generation
```bash
# With authentication
curl -X POST http://localhost:8000/api/v1/audio/speech \
  -H "Authorization: Bearer your-api-token" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "tts-1",
    "input": "Hello, this is a test of the TTS system.",
    "voice": "alloy",
    "response_format": "mp3"
  }' \
  --output test.mp3
```

### 3. Test Streaming
```python
import httpx
import asyncio

async def test_streaming():
    async with httpx.AsyncClient() as client:
        response = await client.post(
            "http://localhost:8000/api/v1/audio/speech",
            headers={"Authorization": "Bearer your-token"},
            json={
                "model": "tts-1",
                "input": "This is a streaming test.",
                "voice": "nova",
                "stream": True
            },
            timeout=30.0
        )
        
        with open("stream_test.mp3", "wb") as f:
            async for chunk in response.aiter_bytes():
                f.write(chunk)

asyncio.run(test_streaming())
```

## Using Local Kokoro Models

### 1. Download Models
```bash
# Create models directory
mkdir -p models

# Download Kokoro ONNX model
wget https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files/kokoro-v0_19.onnx \
     -O models/kokoro-v0_19.onnx

# Download voices configuration
wget https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files/voices.json \
     -O models/voices.json
```

### 2. Configure for Local Model
```ini
[TTS]
default_provider = kokoro
kokoro_model_path = models/kokoro-v0_19.onnx
kokoro_voices_json = models/voices.json
```

### 3. Test Local Generation
```bash
curl -X POST http://localhost:8000/api/v1/audio/speech \
  -H "Authorization: Bearer your-token" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "kokoro_local",
    "input": "Testing local Kokoro model.",
    "voice": "af_bella",
    "response_format": "wav"
  }' \
  --output kokoro_test.wav
```

## Monitoring

### 1. Logs
The application uses loguru for comprehensive logging:
```bash
# View logs
tail -f logs/tldw_server.log

# Filter TTS-specific logs
grep "TTS" logs/tldw_server.log
```

### 2. Metrics
Monitor these key metrics:
- Request rate (should stay under configured limit)
- Response time (typically 1-5 seconds for short text)
- Error rate (authentication failures, API errors)
- Memory usage (increases with local models)

### 3. Common Issues and Solutions

| Issue | Solution |
|-------|----------|
| "OpenAI API key not configured" | Add `openai_api_key` to config.txt |
| "Rate limit exceeded" | Increase `tts_rate_limit` in config |
| "Kokoro ONNX backend not initialized" | Check model files exist and espeak is installed |
| "Unsupported audio format" | Use one of: mp3, wav, opus, flac, pcm |
| "Authentication required" | Include Authorization header with Bearer token |
| High memory usage | Reduce worker count or disable local models |
| Slow generation | Use GPU for local models or switch to API backend |

## Security Considerations

1. **API Keys**: Never commit API keys to version control
2. **Authentication**: Always enable in production (`require_auth = true`)
3. **Rate Limiting**: Adjust based on your infrastructure capacity
4. **Input Validation**: Maximum input length is enforced (4096 chars)
5. **HTTPS**: Always use HTTPS in production (configure in Nginx)

## Performance Tuning

### For API Backends (OpenAI, ElevenLabs)
- Increase worker count for parallel processing
- Use connection pooling (already implemented)
- Cache frequently requested phrases (future enhancement)

### For Local Models (Kokoro)
- Use GPU acceleration when available
- Reduce model precision for faster inference
- Pre-load models at startup (already implemented)

## Troubleshooting

### Enable Debug Logging
```python
# In config.txt
[Logging]
log_level = DEBUG
```

### Test Individual Components
```bash
# Test OpenAI backend
python -c "from tldw_Server_API.app.core.TTS.tts_backends import OpenAIAPIBackend; print('OK')"

# Test audio writer
python -c "from tldw_Server_API.app.core.TTS.streaming_audio_writer import StreamingAudioWriter; print('OK')"

# Test Kokoro
python -c "import kokoro_onnx; print('Kokoro OK')"
```

### Check Dependencies
```bash
# Verify all TTS dependencies
pip list | grep -E "av|httpx|numpy|kokoro|scipy"

# Check FFmpeg
ffmpeg -version

# Check espeak (for Kokoro)
espeak-ng --version
```

## Support

For issues or questions:
1. Check the logs for detailed error messages
2. Verify all dependencies are installed
3. Ensure API keys are valid and have sufficient credits
4. Test with curl commands before integrating
5. Report issues with full error messages and configuration

---

Last Updated: 2025-08-24
Version: 1.0.0