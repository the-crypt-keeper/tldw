# TTS (Text-to-Speech) Module

## Overview

The TTS module provides a unified, production-ready interface for multiple text-to-speech providers, supporting both cloud-based APIs and local models. It features automatic fallback, circuit breaker patterns, comprehensive error handling, and resource management.

## Architecture

### Core Components

1. **Adapter Pattern** - Each TTS provider has its own adapter implementing a common interface
2. **Registry & Factory** - Manages adapter lifecycle and provider selection
3. **Service Layer** - High-level API with fallback and error recovery
4. **Resource Management** - Memory monitoring, connection pooling, and session management
5. **Configuration System** - Unified configuration from multiple sources
6. **Circuit Breaker** - Automatic failure detection and recovery

### Directory Structure

```
TTS/
├── adapters/                  # Provider adapters
│   ├── base.py               # Base adapter interface
│   ├── openai_adapter.py     # OpenAI TTS
│   ├── kokoro_adapter.py     # Kokoro (local)
│   ├── higgs_adapter.py      # Higgs Audio V2
│   ├── dia_adapter.py        # Dia (dialogue)
│   ├── chatterbox_adapter.py # Chatterbox (emotion)
│   ├── elevenlabs_adapter.py # ElevenLabs
│   └── vibevoice_adapter.py  # VibeVoice (Microsoft)
├── tts_service_v2.py         # Main service layer
├── adapter_registry.py       # Adapter management
├── circuit_breaker.py        # Circuit breaker implementation
├── tts_config.py            # Unified configuration
├── tts_exceptions.py        # Exception hierarchy
├── tts_validation.py        # Input validation & sanitization
├── tts_resource_manager.py  # Resource management
├── audio_utils.py           # Audio processing utilities
└── streaming_audio_writer.py # Streaming audio handling
```

## Supported Providers

### Cloud Providers

| Provider | Models | Languages | Key Features |
|----------|--------|-----------|--------------|
| **OpenAI** | tts-1, tts-1-hd | 50+ | High quality, reliable |
| **ElevenLabs** | Multiple | 20+ | Voice cloning, emotions |

### Local Models

| Provider | Model Size | Languages | Key Features |
|----------|------------|-----------|--------------|
| **Kokoro** | ~50MB | English | Fast, lightweight, ONNX |
| **Higgs** | 3B params | 50+ | Multilingual, singing |
| **Dia** | 1.6B params | English | Multi-speaker dialogue |
| **Chatterbox** | ~1B params | English | Emotion control |
| **VibeVoice** | 1.5B/7B | 12+ | 90min generation, music |

## Configuration

### Configuration Sources (Priority Order)

1. **Environment Variables** - Highest priority
2. **config.txt** - Main application config
3. **tts_providers_config.yaml** - TTS-specific config
4. **Defaults** - Built-in defaults

### Example Configuration

#### tts_providers_config.yaml
```yaml
provider_priority:
  - openai      # Try first
  - kokoro      # Fallback to local
  
providers:
  openai:
    enabled: true
    api_key: ${OPENAI_API_KEY}
    model: "tts-1"
    
  kokoro:
    enabled: true
    use_onnx: true
    device: "cpu"
    
performance:
  max_concurrent_generations: 4
  memory_warning_threshold: 80
  memory_critical_threshold: 90
  
fallback:
  enabled: true
  max_attempts: 3
```

#### config.txt
```ini
[TTS-Settings]
default_tts_provider = kokoro
default_tts_voice = af_bella
default_tts_speed = 1.0
local_tts_device = cpu
```

#### Environment Variables
```bash
export OPENAI_API_KEY="sk-..."
export ELEVENLABS_API_KEY="..."
export TTS_DEFAULT_PROVIDER="openai"
export TTS_DEVICE="cuda"
```

## Usage

### Basic Usage

```python
from tldw_Server_API.app.core.TTS.tts_service_v2 import get_tts_service_v2
from tldw_Server_API.app.api.v1.schemas.audio_schemas import OpenAISpeechRequest

# Get service instance
tts_service = await get_tts_service_v2()

# Create request
request = OpenAISpeechRequest(
    input="Hello, this is a test.",
    model="tts-1",
    voice="alloy",
    response_format="mp3",
    speed=1.0
)

# Generate speech (with automatic fallback)
async for audio_chunk in tts_service.generate_speech(request):
    # Process audio chunks
    pass
```

### Direct Adapter Usage

```python
from tldw_Server_API.app.core.TTS.adapter_registry import get_tts_factory, TTSProvider
from tldw_Server_API.app.core.TTS.adapters.base import TTSRequest, AudioFormat

# Get factory
factory = await get_tts_factory()

# Get specific adapter
adapter = await factory.registry.get_adapter(TTSProvider.KOKORO)

# Create request
request = TTSRequest(
    text="Hello world",
    voice="af_bella",
    format=AudioFormat.WAV,
    speed=1.0
)

# Generate
response = await adapter.generate(request)
```

## Error Handling

### Exception Hierarchy

```
TTSError (base)
├── TTSConfigurationError
│   ├── TTSProviderNotConfiguredError
│   ├── TTSProviderInitializationError
│   └── TTSModelNotFoundError
├── TTSValidationError
│   ├── TTSTextTooLongError
│   ├── TTSInvalidVoiceReferenceError
│   └── TTSUnsupportedFormatError
├── TTSProviderError
│   ├── TTSAuthenticationError
│   ├── TTSRateLimitError
│   ├── TTSNetworkError
│   └── TTSTimeoutError
├── TTSGenerationError
│   └── TTSStreamingError
└── TTSResourceError
    ├── TTSInsufficientMemoryError
    └── TTSGPUError
```

### Error Recovery

The service automatically handles errors with:
- **Retryable errors** - Network, rate limits, timeouts
- **Fallback providers** - Automatic failover
- **Circuit breaker** - Prevents cascading failures
- **Exponential backoff** - Smart retry timing

## Security Features

### Input Validation
- Text sanitization (XSS, SQL injection prevention)
- Parameter bounds checking
- Voice reference file validation
- Provider-specific limits enforcement

### Resource Protection
- Memory monitoring with thresholds
- Connection pooling with limits
- Session management with cleanup
- GPU memory tracking

## Performance Optimization

### Caching (Future)
- Response caching for repeated requests
- Voice embedding caching
- Model caching for local providers

### Streaming
- Chunk-based audio streaming
- Configurable chunk sizes
- Format conversion on-the-fly

### Resource Management
- Automatic cleanup of idle resources
- Connection reuse for API providers
- Memory-aware model loading

## Testing

### Test Coverage
- `test_tts_exceptions.py` - Exception hierarchy tests
- `test_tts_validation.py` - Validation and security tests
- `test_tts_resource_manager.py` - Resource management tests
- `test_tts_adapters.py` - Adapter functionality tests
- `test_tts_service_v2.py` - Service layer tests

### Running Tests
```bash
# Run all TTS tests
python -m pytest tldw_Server_API/tests/TTS/ -v

# Run specific test file
python -m pytest tldw_Server_API/tests/TTS/test_tts_validation.py -v

# Run with coverage
python -m pytest tldw_Server_API/tests/TTS/ --cov=tldw_Server_API.app.core.TTS
```

## Adding New Providers

### 1. Create Adapter

```python
# adapters/myprovider_adapter.py
from .base import TTSAdapter, TTSRequest, TTSResponse

class MyProviderAdapter(TTSAdapter):
    async def initialize(self) -> bool:
        # Initialize connection/model
        pass
    
    async def generate(self, request: TTSRequest) -> TTSResponse:
        # Generate speech
        pass
    
    async def get_capabilities(self) -> TTSCapabilities:
        # Return provider capabilities
        pass
```

### 2. Register Provider

```python
# adapter_registry.py
class TTSProvider(Enum):
    MYPROVIDER = "myprovider"

# Add to DEFAULT_ADAPTERS
DEFAULT_ADAPTERS = {
    TTSProvider.MYPROVIDER: MyProviderAdapter,
    ...
}
```

### 3. Add Configuration

```yaml
# tts_providers_config.yaml
providers:
  myprovider:
    enabled: true
    api_key: ${MYPROVIDER_API_KEY}
    # ... other settings
```

## Troubleshooting

### Common Issues

1. **Provider Not Available**
   - Check provider is enabled in config
   - Verify API keys are set
   - Check model files for local providers

2. **Memory Errors**
   - Adjust memory thresholds in config
   - Use smaller models or batch sizes
   - Enable model offloading

3. **Network Timeouts**
   - Increase timeout in provider config
   - Check network connectivity
   - Enable fallback providers

4. **Audio Quality Issues**
   - Try different models (e.g., tts-1-hd)
   - Adjust voice settings
   - Check audio format compatibility

### Debug Logging

```python
# Enable debug logging
import logging
logging.getLogger("tldw_Server_API.app.core.TTS").setLevel(logging.DEBUG)
```

## API Reference

### OpenAI-Compatible Endpoint

```
POST /v1/audio/speech
```

Request:
```json
{
  "input": "Text to synthesize",
  "model": "tts-1",
  "voice": "alloy",
  "response_format": "mp3",
  "speed": 1.0
}
```

Response: Audio stream in requested format

### Supported Voices

Each provider has different voice options:
- **OpenAI**: alloy, echo, fable, onyx, nova, shimmer
- **Kokoro**: af_bella, af_nicole, am_adam, am_michael, etc.
- **ElevenLabs**: rachel, drew, clyde, paul, domi, etc.

### Audio Formats

Supported formats vary by provider:
- **MP3** - Widely supported
- **WAV** - Uncompressed
- **OPUS** - Efficient compression
- **FLAC** - Lossless
- **PCM** - Raw audio

## License

This module is part of the tldw_server project and follows the same dual licensing:
- AGPL-3.0 for open source use
- Commercial license available

## Contributing

When contributing to the TTS module:
1. Follow the adapter pattern for new providers
2. Include comprehensive error handling
3. Add validation for all inputs
4. Write tests for new functionality
5. Update this documentation