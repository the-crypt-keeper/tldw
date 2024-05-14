# Too Long, Didnt Watch

YouTube contains an incredible amount of knowledge, much of which is locked inside multi-hour videos.  Let's extract and summarize with AI!

- `diarize.py` - download, transrcibe and diarize audio
  - [yt-dlp](https://github.com/yt-dlp/yt-dlp) - download audio tracks of youtube videos
  - [ffmpeg](https://github.com/FFmpeg/FFmpeg) - decompress audio
  - [faster_whisper](https://github.com/SYSTRAN/faster-whisper) - speech to text
  - [pyannote](https://github.com/pyannote/pyannote-audio) - diarization

- `chunker.py` - break text into parts and prepare each part for LLM summarization

- `roller-*.py` - rolling summarization
  - [can-ai-code](https://github.com/the-crypt-keeper/can-ai-code) - interview executors to run LLM inference

- `compare.py` - prepare LLM outputs for webapp
- `compare-app.py` - summary viewer webapp

This project is under active development and is not ready for production use.
