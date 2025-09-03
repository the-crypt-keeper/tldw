# TTS Voice Cloning Guide

## Overview

Voice cloning, also known as voice synthesis or voice mimicry, allows the TTS system to generate speech that sounds like a specific person based on a reference audio sample. This guide covers the implementation, usage, and best practices for voice cloning in the tldw_server TTS module.

## Table of Contents

- [Supported Providers](#supported-providers)
- [How Voice Cloning Works](#how-voice-cloning-works)
- [API Reference](#api-reference)
- [Implementation Details](#implementation-details)
- [Audio Preparation Guide](#audio-preparation-guide)
- [Usage Examples](#usage-examples)
- [Performance Considerations](#performance-considerations)
- [Security & Ethics](#security--ethics)
- [Troubleshooting](#troubleshooting)
- [Future Enhancements](#future-enhancements)

## Supported Providers

### Provider Comparison

| Provider | Min Duration | Max Duration | Languages | Quality | Speed | VRAM Usage |
|----------|-------------|--------------|-----------|---------|-------|------------|
| **Higgs** | 3s | 10s | 50+ | Excellent | Medium | 6GB+ |
| **Chatterbox** | 5s | 20s | English | Very Good | Fast | 4GB+ |
| **VibeVoice** | 3s | 30s | 12 | Good | Medium | 3-16GB |
| **ElevenLabs** | 1 min | 5 min | 29 | Excellent | Fast | N/A (Cloud) |

### Higgs Audio V2

**Strengths:**
- Multilingual support (50+ languages)
- Music generation capabilities
- High-quality voice reproduction
- Cross-lingual voice transfer

**Limitations:**
- Requires 3-10 second samples
- Higher VRAM requirements
- Slower than Chatterbox

**Best Use Cases:**
- International applications
- High-quality audiobooks
- Multilingual content

### Chatterbox

**Strengths:**
- Low latency (sub-200ms)
- Emotion control
- Perth watermarking for security
- Optimized for English

**Limitations:**
- English only
- Requires 5-20 second samples
- Limited to single speaker

**Best Use Cases:**
- Real-time applications
- Emotional narratives
- English-only content

### VibeVoice

**Strengths:**
- Long-form generation (up to 90 min)
- Spontaneous background music
- Multi-speaker support
- Natural conversational tone

**Limitations:**
- Larger model size
- Higher latency
- Limited language support

**Best Use Cases:**
- Podcasts
- Long-form content
- Multi-speaker dialogues

## How Voice Cloning Works

### Technical Process

1. **Audio Analysis**
   - Extract acoustic features (pitch, timbre, prosody)
   - Generate speaker embeddings
   - Analyze speaking style and patterns

2. **Embedding Generation**
   - Convert audio to mel-spectrograms
   - Generate speaker embeddings using neural networks
   - Store embeddings for synthesis

3. **Voice Synthesis**
   - Apply speaker embeddings to text encoder
   - Generate audio with target voice characteristics
   - Post-process for quality enhancement

### Architecture Flow

```
Input Audio → Audio Processor → Feature Extraction → Speaker Encoder
                                                            ↓
Text Input → Text Encoder → Synthesis Model ← Speaker Embeddings
                                    ↓
                            Generated Audio → Post-Processing → Output
```

## API Reference

### Request Schema

```python
class TTSRequest(BaseModel):
    model: str  # "higgs", "chatterbox", or "vibevoice"
    input: str  # Text to synthesize
    voice: str = "clone"  # Use "clone" for voice cloning
    voice_reference: Optional[str] = None  # Base64-encoded audio
    response_format: str = "mp3"  # Output format
    speed: float = 1.0  # Speech speed (0.25 to 4.0)
    extra_params: Optional[Dict[str, Any]] = None  # Provider-specific
```

### Voice Reference Processing

```python
class VoiceReferenceRequest(BaseModel):
    audio_data: str  # Base64-encoded audio
    provider: str  # Target provider
    validate: bool = True  # Validate audio requirements
    convert: bool = True  # Auto-convert format
```

### Response Format

```python
class TTSResponse(BaseModel):
    audio: bytes  # Generated audio data
    format: str  # Audio format
    duration: float  # Duration in seconds
    provider: str  # Provider used
    voice_id: Optional[str]  # Generated voice ID for reuse
```

## Implementation Details

### Audio Processing Pipeline

The `audio_utils.py` module handles all voice reference processing:

```python
from tldw_Server_API.app.core.TTS.audio_utils import (
    process_voice_reference,
    validate_audio_requirements,
    convert_audio_format
)

# Process voice reference
processed_audio, error = process_voice_reference(
    voice_reference_bytes,
    provider='higgs',
    validate=True,
    convert=True
)
```

### Provider Integration

Each provider adapter implements voice cloning through the `_prepare_voice_reference` method:

```python
class HiggsAdapter(TTSAdapter):
    async def _prepare_voice_reference(
        self, 
        voice_reference: bytes
    ) -> Optional[str]:
        """Process and prepare voice reference for Higgs."""
        # Validate audio requirements
        processed_audio, error = process_voice_reference(
            voice_reference,
            provider='higgs',
            validate=True,
            convert=True
        )
        
        if error:
            logger.error(f"Voice reference processing failed: {error}")
            return None
            
        # Save to temporary file
        temp_path = f"/tmp/voice_ref_{uuid.uuid4()}.wav"
        with open(temp_path, 'wb') as f:
            f.write(processed_audio)
            
        return temp_path
```

### Temporary File Management

Voice references are stored as temporary files during processing:

```python
# Cleanup pattern used in all adapters
try:
    # Process with voice reference
    audio_data = await self._generate_with_reference(
        text, 
        voice_ref_path
    )
finally:
    # Always cleanup temporary files
    if voice_ref_path and os.path.exists(voice_ref_path):
        os.remove(voice_ref_path)
```

## Audio Preparation Guide

### Recording Best Practices

1. **Environment Setup**
   - Use a quiet room with minimal echo
   - Position microphone 6-12 inches from mouth
   - Use pop filter to reduce plosives
   - Maintain consistent distance

2. **Recording Settings**
   - Sample rate: 24kHz or 48kHz
   - Bit depth: 16-bit or 24-bit
   - Format: WAV or FLAC (lossless)
   - Mono recording preferred

3. **Speaking Guidelines**
   - Speak naturally at normal pace
   - Maintain consistent volume
   - Include variety in intonation
   - Avoid extreme emotions

### Audio Processing Tools

#### Using FFmpeg

```bash
# Basic conversion to required format
ffmpeg -i input.mp3 -ar 24000 -ac 1 -c:a pcm_s16le output.wav

# Normalize audio levels
ffmpeg -i input.wav -af loudnorm=I=-16:TP=-1.5:LRA=11 normalized.wav

# Remove silence from beginning/end
ffmpeg -i input.wav -af silenceremove=1:0:-50dB output.wav

# Extract segment from longer audio
ffmpeg -i input.wav -ss 00:00:05 -t 00:00:10 segment.wav
```

#### Using Python (librosa)

```python
import librosa
import soundfile as sf
import numpy as np

def prepare_voice_sample(input_path, output_path, target_sr=24000):
    """Prepare voice sample for cloning."""
    # Load audio
    audio, sr = librosa.load(input_path, sr=None)
    
    # Resample if needed
    if sr != target_sr:
        audio = librosa.resample(audio, orig_sr=sr, target_sr=target_sr)
    
    # Normalize amplitude
    audio = audio / np.max(np.abs(audio))
    audio = audio * 0.95  # Leave headroom
    
    # Trim silence
    audio, _ = librosa.effects.trim(audio, top_db=20)
    
    # Apply gentle compression
    audio = np.tanh(audio * 0.7) / 0.7
    
    # Save processed audio
    sf.write(output_path, audio, target_sr, subtype='PCM_16')
    
    return len(audio) / target_sr  # Return duration
```

### Quality Validation

```python
def validate_voice_sample(audio_path):
    """Validate voice sample quality."""
    audio, sr = librosa.load(audio_path, sr=None)
    duration = len(audio) / sr
    
    # Check duration
    if duration < 3:
        return False, "Audio too short (min 3 seconds)"
    if duration > 30:
        return False, "Audio too long (max 30 seconds)"
    
    # Check energy level
    rms = np.sqrt(np.mean(audio**2))
    if rms < 0.01:
        return False, "Audio too quiet"
    
    # Check for clipping
    if np.max(np.abs(audio)) >= 0.99:
        return False, "Audio is clipping"
    
    # Check silence ratio
    silence = np.sum(np.abs(audio) < 0.01) / len(audio)
    if silence > 0.3:
        return False, "Too much silence in audio"
    
    return True, "Audio validated successfully"
```

## Usage Examples

### Basic Voice Cloning

```python
import base64
import requests
from pathlib import Path

def clone_voice(text, voice_file, provider="higgs"):
    """Clone voice from audio file."""
    # Read and encode voice sample
    voice_data = Path(voice_file).read_bytes()
    voice_b64 = base64.b64encode(voice_data).decode()
    
    # Make API request
    response = requests.post(
        "http://localhost:8000/api/v1/audio/speech",
        json={
            "model": provider,
            "input": text,
            "voice": "clone",
            "voice_reference": voice_b64,
            "response_format": "mp3"
        }
    )
    
    if response.status_code == 200:
        return response.content
    else:
        raise Exception(f"Error: {response.text}")

# Example usage
audio = clone_voice(
    "Hello, this is my cloned voice speaking.",
    "my_voice_sample.wav",
    provider="chatterbox"
)
Path("output.mp3").write_bytes(audio)
```

### Advanced Cloning with Parameters

```python
class VoiceCloner:
    def __init__(self, base_url="http://localhost:8000"):
        self.base_url = base_url
        self.session = requests.Session()
    
    def clone_with_emotion(self, text, voice_file, emotion="neutral"):
        """Clone voice with emotion control (Chatterbox)."""
        voice_b64 = self._encode_file(voice_file)
        
        response = self.session.post(
            f"{self.base_url}/api/v1/audio/speech",
            json={
                "model": "chatterbox",
                "input": text,
                "voice": "clone",
                "voice_reference": voice_b64,
                "extra_params": {
                    "emotion": emotion,
                    "emotion_intensity": 1.2,
                    "enable_watermark": True
                }
            }
        )
        return response.content
    
    def clone_with_music(self, text, voice_file, enable_music=True):
        """Clone voice with background music (VibeVoice)."""
        voice_b64 = self._encode_file(voice_file)
        
        response = self.session.post(
            f"{self.base_url}/api/v1/audio/speech",
            json={
                "model": "vibevoice",
                "input": text,
                "voice": "clone",
                "voice_reference": voice_b64,
                "extra_params": {
                    "enable_music": enable_music,
                    "vibe": "casual",
                    "vibe_intensity": 1.0
                }
            }
        )
        return response.content
    
    def _encode_file(self, file_path):
        """Encode file to base64."""
        with open(file_path, 'rb') as f:
            return base64.b64encode(f.read()).decode()
```

### Batch Voice Cloning

```python
import asyncio
import aiohttp
from typing import List, Tuple

async def batch_clone_voices(
    texts: List[str],
    voice_file: str,
    provider: str = "higgs"
) -> List[bytes]:
    """Clone multiple texts with same voice concurrently."""
    
    # Prepare voice reference once
    with open(voice_file, 'rb') as f:
        voice_b64 = base64.b64encode(f.read()).decode()
    
    async def clone_single(session, text):
        async with session.post(
            "http://localhost:8000/api/v1/audio/speech",
            json={
                "model": provider,
                "input": text,
                "voice": "clone",
                "voice_reference": voice_b64
            }
        ) as response:
            return await response.read()
    
    # Process concurrently
    async with aiohttp.ClientSession() as session:
        tasks = [clone_single(session, text) for text in texts]
        return await asyncio.gather(*tasks)

# Usage
texts = [
    "First paragraph to clone.",
    "Second paragraph to clone.",
    "Third paragraph to clone."
]

audio_files = asyncio.run(
    batch_clone_voices(texts, "voice_sample.wav")
)
```

## Performance Considerations

### Optimization Strategies

1. **Voice Reference Caching**
   ```python
   class VoiceReferenceCache:
       def __init__(self, ttl_hours=24):
           self.cache = {}
           self.ttl = ttl_hours * 3600
       
       def get_or_process(self, audio_data, provider):
           cache_key = hashlib.md5(audio_data).hexdigest()
           
           if cache_key in self.cache:
               cached, timestamp = self.cache[cache_key]
               if time.time() - timestamp < self.ttl:
                   return cached
           
           # Process and cache
           processed = process_voice_reference(audio_data, provider)
           self.cache[cache_key] = (processed, time.time())
           return processed
   ```

2. **Batch Processing**
   - Group multiple requests with same voice
   - Process in parallel when possible
   - Reuse speaker embeddings

3. **Resource Management**
   - Use FP16/BF16 for reduced memory
   - Implement request queuing
   - Monitor VRAM usage

### Benchmarks

| Provider | Processing Time | VRAM Usage | Quality Score |
|----------|----------------|------------|---------------|
| Higgs | 2-3s per sentence | 6GB | 4.5/5 |
| Chatterbox | 0.2-0.5s per sentence | 4GB | 4.2/5 |
| VibeVoice | 1-2s per sentence | 3-16GB | 4.0/5 |

## Security & Ethics

### Consent Management

```python
class VoiceConsentManager:
    """Manage voice cloning consent."""
    
    def __init__(self, db_path="consent.db"):
        self.db = sqlite3.connect(db_path)
        self._init_db()
    
    def record_consent(
        self,
        voice_id: str,
        owner_name: str,
        consent_text: str,
        expiry_date: Optional[datetime] = None
    ):
        """Record consent for voice cloning."""
        self.db.execute("""
            INSERT INTO voice_consent 
            (voice_id, owner_name, consent_text, expiry_date, created_at)
            VALUES (?, ?, ?, ?, ?)
        """, (voice_id, owner_name, consent_text, expiry_date, datetime.now()))
        self.db.commit()
    
    def verify_consent(self, voice_id: str) -> bool:
        """Verify active consent exists."""
        result = self.db.execute("""
            SELECT expiry_date FROM voice_consent
            WHERE voice_id = ? 
            AND (expiry_date IS NULL OR expiry_date > ?)
        """, (voice_id, datetime.now())).fetchone()
        return result is not None
```

### Usage Logging

```python
class VoiceCloningAudit:
    """Audit voice cloning usage."""
    
    def log_usage(
        self,
        user_id: str,
        voice_id: str,
        text_hash: str,
        provider: str,
        duration: float
    ):
        """Log voice cloning usage."""
        audit_entry = {
            "timestamp": datetime.utcnow().isoformat(),
            "user_id": user_id,
            "voice_id": voice_id,
            "text_hash": hashlib.sha256(text_hash.encode()).hexdigest(),
            "provider": provider,
            "duration": duration,
            "ip_address": request.remote_addr
        }
        
        # Log to file or database
        logger.info(f"Voice cloning audit: {json.dumps(audit_entry)}")
```

### Watermarking

Chatterbox supports Perth watermarking for traceability:

```python
def generate_with_watermark(text, voice_ref, user_id):
    """Generate audio with embedded watermark."""
    return requests.post(
        "http://localhost:8000/api/v1/audio/speech",
        json={
            "model": "chatterbox",
            "input": text,
            "voice": "clone",
            "voice_reference": voice_ref,
            "extra_params": {
                "enable_watermark": True,
                "watermark_data": {
                    "user_id": user_id,
                    "timestamp": datetime.utcnow().isoformat(),
                    "purpose": "authorized_clone"
                }
            }
        }
    )
```

### Best Practices

1. **Always obtain explicit consent** before cloning voices
2. **Implement usage logging** for audit trails
3. **Use watermarking** when available
4. **Set expiration dates** for voice models
5. **Restrict access** to voice cloning APIs
6. **Monitor for abuse** patterns
7. **Provide opt-out** mechanisms
8. **Document voice sources** clearly

## Troubleshooting

### Common Issues

#### 1. Voice Quality Issues

**Problem:** Generated voice doesn't match reference

**Solutions:**
- Ensure reference audio is clear and noise-free
- Use longer reference samples (up to provider maximum)
- Try different providers for different voice types
- Check sample rate matches provider requirements

```python
def diagnose_voice_quality(voice_file):
    """Diagnose potential voice quality issues."""
    audio, sr = librosa.load(voice_file, sr=None)
    
    issues = []
    
    # Check SNR
    signal_power = np.mean(audio**2)
    noise_floor = np.percentile(np.abs(audio), 5)**2
    snr = 10 * np.log10(signal_power / noise_floor)
    if snr < 15:
        issues.append(f"Low SNR: {snr:.1f}dB (recommend >15dB)")
    
    # Check frequency content
    D = np.abs(librosa.stft(audio))
    spectral_rolloff = librosa.feature.spectral_rolloff(S=D, sr=sr)[0]
    if np.mean(spectral_rolloff) < 2000:
        issues.append("Limited frequency range")
    
    return issues
```

#### 2. Memory Errors

**Problem:** Out of memory during voice cloning

**Solutions:**
```yaml
# Adjust provider configuration
higgs:
  use_fp16: true  # Reduce precision
  batch_size: 1   # Reduce batch size
  offload_to_cpu: true  # CPU offloading
```

#### 3. Slow Generation

**Problem:** Voice cloning takes too long

**Solutions:**
- Use Chatterbox for low-latency needs
- Enable GPU acceleration
- Implement caching for repeated voices
- Consider preprocessing voice references

#### 4. API Errors

**Problem:** 400 Bad Request errors

**Debugging Steps:**
```python
def debug_voice_request(voice_file):
    """Debug voice cloning request."""
    # Check file size
    size_mb = os.path.getsize(voice_file) / 1024 / 1024
    print(f"File size: {size_mb:.2f}MB")
    
    # Check audio properties
    audio, sr = librosa.load(voice_file, sr=None)
    duration = len(audio) / sr
    print(f"Duration: {duration:.2f}s")
    print(f"Sample rate: {sr}Hz")
    
    # Check encoding
    try:
        with open(voice_file, 'rb') as f:
            b64 = base64.b64encode(f.read()).decode()
        print(f"Base64 length: {len(b64)}")
    except Exception as e:
        print(f"Encoding error: {e}")
```

## Future Enhancements

### Planned Features

1. **Voice Bank Management**
   - Store and manage multiple voice profiles
   - Voice mixing and blending
   - Voice style transfer

2. **Real-time Voice Cloning**
   - Stream-based voice cloning
   - Live voice conversion
   - WebRTC integration

3. **Advanced Controls**
   - Fine-grained prosody control
   - Accent modification
   - Age/gender transformation

4. **Multi-speaker Synthesis**
   - Conversation generation
   - Automatic speaker assignment
   - Voice consistency across sessions

5. **Quality Improvements**
   - Neural vocoder integration
   - Super-resolution for low-quality references
   - Noise-robust voice cloning

### Experimental Features

```python
# Voice style transfer (future)
def transfer_voice_style(
    content_voice: str,
    style_voice: str,
    text: str
) -> bytes:
    """Apply style from one voice to another."""
    pass

# Voice morphing (future)
def morph_voices(
    voice_a: str,
    voice_b: str,
    blend_ratio: float,
    text: str
) -> bytes:
    """Blend two voices together."""
    pass

# Emotional voice transformation (future)
def transform_emotion(
    voice: str,
    source_emotion: str,
    target_emotion: str,
    text: str
) -> bytes:
    """Transform emotional tone of voice."""
    pass
```

## References

### Papers & Research
- [YourTTS: Towards Zero-shot Multi-speaker TTS](https://arxiv.org/abs/2112.02418)
- [Neural Voice Cloning with Few Samples](https://arxiv.org/abs/1802.06006)
- [Transfer Learning from Speaker Verification](https://arxiv.org/abs/1806.04558)

### Provider Documentation
- [Higgs Audio V2](https://github.com/bosonai/higgs-audio)
- [Chatterbox](https://github.com/resemble-ai/chatterbox)
- [VibeVoice](https://github.com/microsoft/VibeVoice)

### Tools & Libraries
- [librosa](https://librosa.org/) - Audio analysis
- [ffmpeg](https://ffmpeg.org/) - Audio processing
- [pyannote](https://github.com/pyannote/pyannote-audio) - Speaker diarization

---

*Last Updated: 2025-08-31*
*Version: 1.0.0*