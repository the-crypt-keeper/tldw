# TTS Module Rewrite Plan - Phase 1

## Overall Status
**Start Date**: 2025-08-24  
**Target Completion**: 5 weeks  
**Current Phase**: 1 - Security & Integration

## Phase 1: Security & Integration (Week 1)

### 1. Add Authentication & Authorization ✅ COMPLETE
- [x] Added authentication check using existing auth_utils
- [x] Added Bearer token validation
- [x] Added WWW-Authenticate headers for 401 responses
- [x] Uses existing auth system configuration
- [x] Service accounts supported via API tokens

### 2. Add Rate Limiting ✅ COMPLETE
- [x] Added slowapi rate limiter (10 requests/minute per IP)
- [x] Added 429 status code documentation
- [x] Rate limiting enforced at endpoint level
- [x] Can be adjusted via code configuration

### 3. Input Validation ✅ COMPLETE
- [x] Added empty input validation
- [x] Added maximum input length limit (4096 chars)
- [x] Added input trimming check

### 4. Fix Application Lifecycle ✅ COMPLETE
- [x] Added TTS service initialization to app startup
- [x] Added proper shutdown handlers
- [x] Fixed resource management and connection pooling
- [x] Integrated with main app lifespan management

## Phase 2: Core Functionality (Week 2-3)

### 5. Complete Backend Implementations ✅ MOSTLY COMPLETE
- [x] Removed placeholder/dummy code
- [x] Completed OpenAI backend with proper streaming
- [x] Fixed LocalKokoro ONNX backend with streaming support
- [ ] PyTorch Kokoro backend (marked as not implemented)
- [ ] Add ElevenLabs backend (structure in place)
- [ ] Add AllTalk backend (structure in place)

### 6. Fix Audio Processing ✅ COMPLETE
- [x] Added av (pyav) dependency to requirements.txt
- [x] Implemented StreamingAudioWriter class
- [x] Added AudioNormalizer for format conversion
- [x] Supports WAV, MP3, Opus, FLAC, AAC, PCM formats
- [x] Full streaming support implemented

### 7. Error Handling & Recovery ✅ PARTIALLY COMPLETE
- [x] Added proper error responses with detailed messages
- [x] Improved error handling in all backends
- [x] Added specific error types (auth, rate limit, etc.)
- [ ] Add retry logic for external API calls
- [ ] Implement circuit breaker pattern
- [ ] Add fallback mechanisms

## Phase 3: Testing & Documentation (Week 4)

### 8. Add Comprehensive Tests 🔴 NOT STARTED
- [ ] Unit tests for each backend
- [ ] Integration tests for API endpoints
- [ ] Security tests for auth/rate limiting
- [ ] Performance tests for streaming
- [ ] Load tests for concurrent requests

### 9. Complete Documentation 🔴 NOT STARTED
- [ ] API documentation with examples
- [ ] Configuration guide
- [ ] Deployment instructions
- [ ] Migration guide from old TTS system
- [ ] Troubleshooting guide

## Phase 4: Production Preparation (Week 5)

### 10. Add Monitoring & Observability 🔴 NOT STARTED
- [ ] Integrate with existing telemetry system
- [ ] Add metrics for TTS generation (latency, errors, usage)
- [ ] Add structured logging throughout
- [ ] Add health check endpoint

### 11. Configuration Management 🔴 NOT STARTED
- [ ] Move hardcoded values to config files
- [ ] Add environment-specific configurations
- [ ] Document all configuration options
- [ ] Add configuration validation

## Critical Issues Found During Assessment

### Security Issues (FIXED/IN PROGRESS)
- ✅ No authentication - FIXED
- ✅ No rate limiting - FIXED  
- ✅ No input validation - FIXED
- ⏳ API keys in source code - IN PROGRESS
- 🔴 No user permissions - NOT STARTED

### Architectural Issues
- 🔴 Service not integrated into app lifecycle
- 🔴 Memory leaks - resources never cleaned up
- 🔴 No connection pooling
- 🔴 Incorrect singleton pattern with race conditions
- 🔴 Missing error recovery

### Code Quality Issues
- 🔴 Zero test coverage
- 🔴 30+ FIXME comments
- 🔴 Placeholder implementations throughout
- 🔴 Missing dependencies (pyav)
- 🔴 Dead code from old implementation

## Files Modified/Created

### Phase 1 Changes
1. `/app/api/v1/endpoints/audio.py` - Added auth, rate limiting, validation, connected real service
2. `/app/main.py` - Added TTS service lifecycle integration
3. `/app/core/TTS/tts_generation.py` - Cleaned up, removed placeholders
4. `/app/core/TTS/tts_backends.py` - Implemented OpenAI and Kokoro backends
5. `/app/core/TTS/streaming_audio_writer.py` - NEW - Audio format conversion
6. `/requirements.txt` - Added av dependency

## Next Immediate Tasks
1. ✅ DONE - Write comprehensive tests for the TTS module
2. Add ElevenLabs backend implementation
3. Add configuration for API keys in config.txt template
4. Test with actual API keys and audio generation
5. Add health check endpoint for TTS service
6. Document deployment requirements

## Notes
- Using existing auth system from chat endpoint as reference
- Rate limiting set conservatively at 10/min (can be adjusted)
- Input length limit set at 4096 chars (OpenAI uses 4096)
- Need to decide on keeping or removing LocalKokoro backend