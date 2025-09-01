# TTS Adapter Tests

This directory contains comprehensive test suites for all TTS adapters, organized into mock (unit) tests and integration tests.

## Test Organization

Each adapter has two test files:
- `test_{adapter}_adapter_mock.py` - Unit tests with mocked dependencies
- `test_{adapter}_adapter_integration.py` - Integration tests with real services/models

## Test Files

### OpenAI Adapter
- **Mock Tests**: `test_openai_adapter_mock.py`
  - Initialization with/without API key
  - Capabilities reporting
  - Voice mapping
  - Successful generation (mocked)
  - Error handling (auth, rate limit)
  - Streaming support
  - Request validation
  
- **Integration Tests**: `test_openai_adapter_integration.py`
  - Real API initialization
  - Actual audio generation
  - Streaming generation
  - Different voices and formats
  - Speed variations
  - Concurrent requests
  - Long text handling

### ElevenLabs Adapter
- **Mock Tests**: `test_elevenlabs_adapter_mock.py`
  - Initialization with/without API key
  - Capabilities reporting
  - Voice mapping and ID detection
  - Model selection logic
  - Voice settings configuration
  - Streaming context manager
  - Error handling
  
- **Integration Tests**: `test_elevenlabs_adapter_integration.py`
  - Real API initialization
  - Actual audio generation
  - Voice cloning (if supported)
  - Emotion control
  - Multi-language support
  - Concurrent requests

### Kokoro Adapter
- **Mock Tests**: `test_kokoro_adapter_mock.py`
  - PyTorch vs ONNX mode initialization
  - Voice mapping and mixing
  - Phoneme processing
  - Device selection (CPU/CUDA/MPS)
  - Model loading error handling
  
- **Integration Tests**: `test_kokoro_adapter_integration.py`
  - Real model initialization
  - Audio generation with local model
  - Voice mixing feature
  - GPU acceleration
  - Phoneme input
  - Format conversion

### Higgs Adapter
- **Mock Tests**: `test_higgs_adapter_mock.py`
  - Local model configuration
  - Voice cloning support
  - GPU configuration
  - Multi-language support
  
- **Integration Tests**: `test_higgs_adapter_integration.py`
  - Model loading and initialization
  - Audio generation
  - Voice cloning with reference audio
  - Emotion and style control

### Dia Adapter
- **Mock Tests**: `test_dia_adapter_mock.py`
  - API endpoint configuration
  - Multi-speaker dialogue support
  - Language support
  
- **Integration Tests**: `test_dia_adapter_integration.py`
  - Model initialization
  - Dialogue generation
  - Speaker management

### Chatterbox Adapter
- **Mock Tests**: `test_chatterbox_adapter_mock.py`
  - Model selection
  - Character voice support
  - Speech style parameters
  
- **Integration Tests**: `test_chatterbox_adapter_integration.py`
  - Model initialization
  - Character-based generation
  - Style variations

### VibeVoice Adapter
- **Mock Tests**: `test_vibevoice_adapter_mock.py`
  - Workspace configuration
  - Voice creation support
  - Batch processing
  
- **Integration Tests**: `test_vibevoice_adapter_integration.py`
  - Model initialization
  - Audio generation
  - Custom voice features

## Running Tests

### Run All Mock Tests
```bash
pytest tests/TTS/adapters/test_*_mock.py -v
```

### Run All Integration Tests
```bash
pytest tests/TTS/adapters/test_*_integration.py -v -m integration
```

### Run Tests for Specific Adapter
```bash
# OpenAI adapter tests
pytest tests/TTS/adapters/test_openai_adapter_*.py -v

# Kokoro adapter tests  
pytest tests/TTS/adapters/test_kokoro_adapter_*.py -v
```

### Run Only Tests That Don't Require API Keys
```bash
pytest tests/TTS/adapters/test_*_mock.py -v
```

### Run Integration Tests with API Keys
```bash
# Set environment variables first
export OPENAI_API_KEY="your-key"
export ELEVENLABS_API_KEY="your-key"

# Run integration tests
pytest tests/TTS/adapters/test_*_integration.py -v -m integration
```

## Test Markers

- `@pytest.mark.asyncio` - Async test functions
- `@pytest.mark.integration` - Integration tests requiring real services
- `@pytest.mark.skipif` - Conditional test execution

## Requirements

### For Mock Tests
- No external dependencies required
- All services are mocked

### For Integration Tests
- **OpenAI**: Requires `OPENAI_API_KEY` environment variable
- **ElevenLabs**: Requires `ELEVENLABS_API_KEY` environment variable
- **Kokoro**: Requires downloaded model files in `~/.cache/kokoro/` or `./models/kokoro/`
- **Higgs**: Requires Higgs Audio model and boson_multimodal library
- **Dia**: Requires Dia model from HuggingFace
- **Chatterbox**: Requires Chatterbox model and library
- **VibeVoice**: Requires VibeVoice model files

## Platform-Specific Notes

### GPU Acceleration
- **CUDA**: Tests will use CUDA if available on Linux/Windows
- **MPS**: Tests will use Metal Performance Shaders on macOS with Apple Silicon
- **CPU**: Falls back to CPU if no GPU available

### Model Downloads
For local models (Kokoro, Higgs, etc.), download models first:
```bash
# Kokoro
huggingface-cli download hexgrad/kokoro-v0_19 --local-dir ~/.cache/kokoro/

# Higgs
huggingface-cli download bosonai/higgs-audio-v2 --local-dir ./models/higgs/
```

## Coverage

To generate coverage reports:
```bash
pytest tests/TTS/adapters/ --cov=tldw_Server_API.app.core.TTS.adapters --cov-report=html
```

## Debugging

For verbose output during test failures:
```bash
pytest tests/TTS/adapters/test_openai_adapter_mock.py -xvs
```

## CI/CD Integration

Mock tests should run on every commit, while integration tests can be run:
- On merge to main branch
- Nightly builds
- When TTS-related code changes

Example GitHub Actions workflow:
```yaml
- name: Run TTS Mock Tests
  run: pytest tests/TTS/adapters/test_*_mock.py -v

- name: Run TTS Integration Tests
  if: github.event_name == 'push' && github.ref == 'refs/heads/main'
  env:
    OPENAI_API_KEY: ${{ secrets.OPENAI_API_KEY }}
    ELEVENLABS_API_KEY: ${{ secrets.ELEVENLABS_API_KEY }}
  run: pytest tests/TTS/adapters/test_*_integration.py -v -m integration
```