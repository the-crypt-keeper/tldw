# Too Long, Didnt Watch

YouTube contains an incredible amount of knowledge, much of which is locked inside multi-hour videos.  Let's extract and summarize with AI!

- [yt-dlp](https://github.com/yt-dlp/yt-dlp) - download audio tracks of youtube videos
- [ffmpeg](https://github.com/FFmpeg/FFmpeg) - decompress audio
- [whisper.cpp](https://github.com/ggerganov/whisper.cpp) - transcribe audio to text
- `chunk.py` - break text into parts and prepare each part for LLM summarization
- [can-ai-code](https://github.com/the-crypt-keeper/can-ai-code) - leverage `interview_cuda` or `interview-llamacpp`` executor to run LLM inference
- `compare.py` - prepare LLM outputs for webapp
- `compare-app.py` - summary viewer webapp

This project is under active development and is not ready for production use.

# [DEMO @ HF Space](https://huggingface.co/spaces/mike-ravkine/too-long-didnt-watch)

## Video Transcript Datasets

|Filename|Title|Whisper Model|URL|
|--------|-----|-------------|---|
|ufo.txt| Subcommittee on National Security, the Border, and Foreign Affairs Hearing | small.en | https://www.youtube.com/watch?v=KQ7Dw-739VY |
|aoe-grand-finale.txt| GRAND FINAL $10,000 AoE2 Event (The Resurgence) | medium.en | https://www.youtube.com/watch?v=jnoxjLJind4 |

## Creating a Dataset

### Download with yt-dlp

Download the audio track:

```
pip install yt-dlp
yt-dlp -f "bestaudio[ext=m4a]" --extract-audio  'https://www.youtube.com/watch?v=<video>'
```

### Transcode with ffmpeg

Convert the audio track to wav:

```
ffmpeg -i *.m4a -hide_banner -vn -loglevel error -ar 16000 -ac 1 -c:a pcm_s16le -y resampled.wav
```

### Transcribe with whisper.cpp

Transcribe the wav to txt:

```
main -m ../models/ggml-medium.en.bin -f resampled.wav -t 32 -otxt
```
