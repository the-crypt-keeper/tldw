# TL/DW: Too Long, Didnt Watch

YouTube contains an incredible amount of knowledge, much of which is locked inside multi-hour videos.  Let's extract and summarize with AI!

### Pieces
- `diarize.py` - download, transcribe and diarize audio
  1. First uses [yt-dlp](https://github.com/yt-dlp/yt-dlp) to download audio(optionally video) from supplied URL
  2. Next, it uses [ffmpeg](https://github.com/FFmpeg/FFmpeg) to convert the resulting `.m4a` file to `.wav`
  3. Then it uses [faster_whisper](https://github.com/SYSTRAN/faster-whisper) to transcribe the `.wav` file to `.txt`
  4. After that, it uses [pyannote](https://github.com/pyannote/pyannote-audio) to perform 'diarorization'
  5. Finally, it'll send the resulting txt to an LLM endpoint of your choice for summarization of the text.
    * Goal is to support OpenAI/Claude/Cohere/Groq/local OpenAI endpoint (oobabooga/llama.cpp/exllama2) so you can either do a batch query to X endpoint, or just feed them one at a time. Your choice.
- `chunker.py` - break text into parts and prepare each part for LLM summarization
- `roller-*.py` - rolling summarization
  - [can-ai-code](https://github.com/the-crypt-keeper/can-ai-code) - interview executors to run LLM inference
- `compare.py` - prepare LLM outputs for webapp
- `compare-app.py` - summary viewer webapp


### Setup
- **Linux**
    1. X
    2. Create a virtual env: `python -m venv`
    3. Launch/activate your virtual env: `. .\scripts\activate.sh`
    4. `pip install -r requirements.txt`
    5. 
- **Windows**
    1. X
    2. Create a virtual env: `python -m venv`
    3. Launch/activate your virtual env: `. .\scripts\activate.ps1`
    4. `pip install -r requirements.txt`
    5. 


### Credits
- [original](https://github.com/the-crypt-keeper/tldw)
- [yt-dlp](https://github.com/yt-dlp/yt-dlp)
- [ffmpeg](https://github.com/FFmpeg/FFmpeg)
- [faster_whisper](https://github.com/SYSTRAN/faster-whisper)
- [pyannote](https://github.com/pyannote/pyannote-audio)