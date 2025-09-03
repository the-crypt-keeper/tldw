# TTS Module - Text-to-Speech Service

## Overview

The TTS module provides a production-ready, extensible Text-to-Speech service with support for multiple providers, voice cloning, and OpenAI-compatible API endpoints. Built with an adapter pattern architecture, it offers seamless fallback between providers, comprehensive error handling, and enterprise-grade features.

## Features

### Core Capabilities
- **Multi-Provider Support**: OpenAI, ElevenLabs, and 4 open-source models
- **Voice Cloning**: Support for voice reference audio with Higgs, Chatterbox, and VibeVoice
- **Streaming Audio**: Real-time audio streaming for all providers
- **Format Support**: MP3, WAV, OPUS, FLAC, PCM output formats
- **OpenAI Compatibility**: Drop-in replacement for OpenAI TTS API
- **Fault Tolerance**: Circuit breaker pattern with automatic failover
- **Performance Metrics**: Built-in monitoring and health checks
- **Transcription/Translation**: OpenAI-compatible speech-to-text endpoints

### Supported Providers

| Provider | Type | Languages | Voice Cloning | Key Features |
|----------|------|-----------|---------------|--------------|
| **OpenAI** | Commercial API | 50+ | ❌ | Industry standard, HD quality |
| **ElevenLabs** | Commercial API | 29 | ✅ (Pro) | Premium quality, emotion control |
| **Kokoro** | Local ONNX | EN | ❌ | Lightweight, CPU-friendly, offline |
| **Higgs** | Local PyTorch | 50+ | ✅ (3-10s) | Music generation, multi-lingual |
| **Chatterbox** | Local PyTorch | EN | ✅ (5-20s) | Emotion exaggeration control |
| **Dia** | Local PyTorch | EN | ❌ | Multi-speaker dialogue specialist |
| **VibeVoice** | Local PyTorch | 12 | ✅ (Any) | Long-form (90min), spontaneous music |

## Architecture

### V2 Adapter Pattern

```
┌─────────────┐     ┌──────────────┐     ┌─────────────┐
│   API       │────▶│  TTS Service │────▶│  Adapter    │
│  Endpoint   │     │      V2      │     │  Registry   │
└─────────────┘     └──────────────┘     └─────────────┘
                            │                     │
                    ┌───────▼───────┐    ┌───────▼───────┐
                    │Circuit Breaker│    │   Provider    │
                    │   Manager     │    │   Adapters   │
                    └───────────────┘    └───────────────┘
                                                 │
                    ┌────────────────────────────┼────────────────────────────┐
                    │                            │                            │
              ┌─────▼─────┐            ┌────────▼────────┐           ┌───────▼───────┐
              │  OpenAI   │            │   Local Models  │           │  Commercial   │
              │  Adapter  │            │   (Kokoro,etc)  │           │   (ElevenLabs)│
              └───────────┘            └─────────────────┘           └───────────────┘
```

### Key Components

1. **TTSServiceV2** (`tts_service_v2.py`)
   - Main service orchestrator
   - Handles provider selection and fallback
   - Integrates metrics and circuit breaker

2. **Adapter Registry** (`adapter_registry.py`)
   - Provider registration and management
   - Capability discovery
   - Dynamic adapter loading

3. **Base Adapter** (`adapters/base.py`)
   - Abstract interface for all providers
   - Standard request/response formats
   - Capability reporting

4. **Provider Adapters** (`adapters/*.py`)
   - Provider-specific implementations
   - Handle authentication and API calls
   - Audio generation and streaming

5. **Circuit Breaker** (`circuit_breaker.py`)
   - Fault tolerance and recovery
   - Automatic provider failover
   - Configurable thresholds

6. **Audio Utils** (`audio_utils.py`)
   - Voice reference processing
   - Format conversion
   - Audio validation

## Installation

### Prerequisites

```bash
# System dependencies
apt-get install ffmpeg espeak-ng  # Ubuntu/Debian
brew install ffmpeg espeak         # macOS

# Python dependencies
pip install -r requirements.txt
```

### Quick Start

1. **Configure API Keys**
```bash
# In config.txt
[API]
openai_api_key = sk-...
elevenlabs_api_key = xi-...
```

2. **Configure TTS Settings**
```yaml
# In tts_providers_config.yaml
provider_priority:
  - openai      # Primary provider
  - kokoro      # Fallback to local
  
providers:
  openai:
    enabled: true
    model: tts-1-hd
  
  kokoro:
    enabled: true
    model_path: ./models/kokoro-v0_19.onnx
```

3. **Start the Server**
```bash
python -m uvicorn tldw_Server_API.app.main:app --host 0.0.0.0 --port 8000
```

## API Usage

### Basic Text-to-Speech

```python
import requests

response = requests.post(
    "http://localhost:8000/api/v1/audio/speech",
    headers={"Authorization": "Bearer your-token"},
    json={
        "model": "tts-1",
        "input": "Hello, world!",
        "voice": "alloy",
        "response_format": "mp3"
    }
)

with open("output.mp3", "wb") as f:
    f.write(response.content)
```

### Voice Cloning

```python
import base64

# Prepare voice reference
with open("voice_sample.wav", "rb") as f:
    voice_ref = base64.b64encode(f.read()).decode()

response = requests.post(
    "http://localhost:8000/api/v1/audio/speech",
    json={
        "model": "higgs",  # or chatterbox, vibevoice
        "input": "This will sound like the reference voice.",
        "voice": "default",
        "voice_reference": voice_ref,  # Base64 encoded audio
        "response_format": "mp3"
    }
)
```

### Streaming Audio

```python
import httpx

async with httpx.AsyncClient() as client:
    response = await client.post(
        "http://localhost:8000/api/v1/audio/speech",
        json={
            "model": "kokoro",
            "input": "Streaming audio test.",
            "voice": "af_bella",
            "stream": True
        },
        timeout=30.0
    )
    
    with open("stream.mp3", "wb") as f:
        async for chunk in response.aiter_bytes():
            f.write(chunk)
```

### Transcription (Speech-to-Text)

```python
# Transcribe audio file
with open("audio.mp3", "rb") as f:
    response = requests.post(
        "http://localhost:8000/api/v1/audio/transcriptions",
        headers={"Authorization": "Bearer your-token"},
        files={"file": f},
        data={
            "model": "whisper-1",
            "language": "en",
            "response_format": "json"
        }
    )
    
print(response.json()["text"])
```

## Configuration

### Provider Configuration (tts_providers_config.yaml)

```yaml
# Provider priority (fallback order)
provider_priority:
  - openai          # Try first
  - elevenlabs      # Try second
  - kokoro          # Local fallback

# Individual provider settings
providers:
  openai:
    enabled: true
    api_key: ${OPENAI_API_KEY}  # From environment
    model: tts-1-hd
    timeout: 30
    
  kokoro:
    enabled: true
    model_path: ./models/kokoro-v0_19.onnx
    device: cpu  # or cuda
    
  higgs:
    enabled: true
    model_path: bosonai/higgs-audio-v2-generation-3B-base
    device: cuda
    use_fp16: true
    
  chatterbox:
    enabled: true
    model_path: resemble-ai/chatterbox
    enable_watermark: true
    
  vibevoice:
    enabled: true
    variant: 1.5B  # or 7B
    device: cuda

# Fallback configuration
fallback:
  enabled: true
  max_attempts: 3
  retry_delay_ms: 1000

# Circuit breaker settings
circuit_breaker:
  failure_threshold: 5
  recovery_timeout: 60
  half_open_calls: 3

# Performance settings
performance:
  max_concurrent_generations: 4
  cache_enabled: false
```

### Voice Cloning Requirements

| Provider | Min Duration | Max Duration | Format | Sample Rate |
|----------|-------------|--------------|--------|-------------|
| Higgs | 3s | 10s | WAV/MP3/FLAC | 24kHz |
| Chatterbox | 5s | 20s | WAV/MP3 | 24kHz+ |
| VibeVoice | 3s | 30s | WAV/MP3 | 22.05kHz |

## Monitoring

### Health Check
```bash
curl http://localhost:8000/api/v1/audio/health
```

Response:
```json
{
  "status": "healthy",
  "providers": {
    "total": 7,
    "available": 3,
    "details": {
      "openai": "available",
      "kokoro": "available",
      "higgs": "not_initialized"
    }
  },
  "circuit_breakers": {
    "openai": "closed",
    "kokoro": "closed"
  }
}
```

### List Providers
```bash
curl http://localhost:8000/api/v1/audio/providers
```

### Metrics
The service integrates with the application's metrics system:
- Request counts per provider
- Response times (p50, p95, p99)
- Error rates by category
- Active request gauges
- Audio generation sizes

## Troubleshooting

### Common Issues

| Issue | Solution |
|-------|----------|
| "Provider not available" | Check API keys and model files |
| "Voice reference validation failed" | Ensure audio meets duration/format requirements |
| "Circuit breaker open" | Provider temporarily disabled due to failures, will auto-recover |
| "Model not found" | Download required model files (see setup guide) |
| "Out of memory" | Reduce batch size or use smaller model variant |

### Debug Mode

Enable detailed logging:
```yaml
# In tts_providers_config.yaml
logging:
  level: DEBUG
  include_metrics: true
```

### Voice Cloning Issues

1. **Audio too short/long**: Check provider-specific duration requirements
2. **Poor quality clone**: Ensure clean audio, single speaker, no background noise
3. **Format not supported**: Convert to WAV 24kHz mono
4. **Memory error**: Voice cloning requires more VRAM, try CPU mode

## Development

### Adding a New Provider

1. Create adapter class in `adapters/`
```python
class MyProviderAdapter(TTSAdapter):
    async def initialize(self) -> bool:
        # Load models/setup API
        
    async def generate(self, request: TTSRequest) -> TTSResponse:
        # Generate audio
        
    async def get_capabilities(self) -> TTSCapabilities:
        # Return provider capabilities
```

2. Register in `adapter_registry.py`
```python
TTSProvider.MY_PROVIDER = "my_provider"
DEFAULT_ADAPTERS[TTSProvider.MY_PROVIDER] = MyProviderAdapter
```

3. Add configuration to YAML
```yaml
providers:
  my_provider:
    enabled: true
    # provider-specific settings
```

### Testing

```bash
# Run TTS tests
pytest tests/TTS/ -v

# Test specific provider
pytest tests/TTS/test_adapters.py::test_openai_adapter -v

# Test with coverage
pytest tests/TTS/ --cov=tldw_Server_API.app.core.TTS
```

## Security Considerations

### Voice Cloning Ethics
- Only clone voices with explicit consent
- Add watermarking when available (Chatterbox)
- Implement usage logging for audit trails
- Consider rate limiting voice cloning requests

### API Security
- Always use authentication in production
- Validate and sanitize all inputs
- Limit file upload sizes
- Use HTTPS for API endpoints
- Rotate API keys regularly

## Performance Optimization

### For API Providers
- Use connection pooling
- Implement response caching
- Batch requests when possible

### For Local Models
- Use GPU acceleration (CUDA)
- Enable mixed precision (FP16/BF16)
- Pre-load models at startup
- Use ONNX runtime for CPU inference

### Voice Cloning
- Cache processed voice references
- Limit reference audio duration
- Use efficient audio formats (WAV)
- Consider CPU/GPU memory limits

## License

The TTS module follows the main project's dual license:
- AGPL-3.0 for open source use
- Commercial license available

Individual model licenses:
- Kokoro: Apache 2.0
- Higgs: Custom research license
- Chatterbox: MIT
- VibeVoice: Microsoft research license

## Support

For issues or questions:
1. Check the [troubleshooting guide](#troubleshooting)
2. Review [API documentation](http://localhost:8000/docs)
3. Check logs in `logs/tldw_server.log`
4. Report issues with full error messages

---

*Last Updated: 2025-08-31*
*Version: 2.0.0*