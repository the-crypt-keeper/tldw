# Speech-To-Text Documentation


## Overview
- **Backend**
  - faster_whisper
  - Model: `WhisperModel` (User-selectable)

- **Whisper Models**
  - https://huggingface.co/distil-whisper/distil-large-v3

### Speech-to-Text
- **Flow**
    1. Convert an input file (like .m4a or .mp4) to .wav via `convert_to_wav(...)`.
    2. Transcribe with `speech_to_text(...)`, which uses Faster Whisper to generate time-stamped text segments.
    3. (Optional)Diarize the same .wav with `audio_diarization(...)`, which uses pyannote to determine speaker boundaries.
    4. (Optional)Combine them in `combine_transcription_and_diarization(...)` to match the transcription segments to the speakers based on time.
- **Key Libraries:**
    - [Faster Whisper](https://github.com/SYSTRAN/faster-whisper) (faster_whisper.WhisperModel) for transcription.
    - [Pyannote](https://github.com/pyannote/pyannote-audio) (pyannote.audio.pipelines.speaker_diarization) for speaker diarization.
    - [FFmpeg](https://www.ffmpeg.org/) (via subprocess or os.system) to convert audio to the desired WAV format.
    - [PyAudio](https://people.csail.mit.edu/hubert/pyaudio/) (for optional live recording).


### Benchmarks
- https://github.com/Picovoice/speech-to-text-benchmark


### Link Dump:
STT
    https://github.com/KoljaB/RealtimeSTT
    https://github.com/southbridgeai/offmute
    https://github.com/flatmax/speech-to-text
    https://github.com/collabora/WhisperLive
    https://github.com/fedirz/faster-whisper-server
    https://github.com/ufal/whisper_streaming
    MoonShine
        https://github.com/usefulsensors/moonshine
        https://github.com/huggingface/transformers.js-examples/tree/main/moonshine-web
        https://huggingface.co/onnx-community/moonshine-base-ONNX
