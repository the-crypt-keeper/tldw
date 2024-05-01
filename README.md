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
    1. Download necessary packages (Python3, ffmpeg[sudo apt install ffmpeg / dnf install ffmpeg], ?)
    2. Create a virtual env: `python -m venv ./`
    3. Launch/activate your virtual env: `. .\scripts\activate.sh`
    4. See `Linux && Windows`
- **Windows**
    1. Download necessary packages (Python3, [ffmpeg](https://www.gyan.dev/ffmpeg/builds/), ?)
    2. Create a virtual env: `python -m venv .\`
    3. Launch/activate your virtual env: `. .\scripts\activate.ps1`
    4. See `Linux && Windows`
- **Linux && Windows**
    1. `pip install -r requirements.txt` - may take a bit of time...
    2. Run `python ./diarize.py <video_url>` - The video URL does _not_ have to be a youtube URL. It can be any site that ytdl supports.
    3. You'll then be asked if you'd like to run the transcription through GPU(1) or CPU(2).
    4. Next, the video will be downloaded to the local directory by ytdl.
    5. Then the video will be transcribed by faster_whisper. (You can see this in the console output)
      * The resulting transcription output will be stored as both a json file with timestamps, as well as a txt file with no timestamps.
    6. Finally, you can have the transcription summarized through feeding it into an LLM of your choice.
    7. For running it locally, here's the commands to do so:
      * FIXME
    8. For feeding the transcriptions to the API of your choice, simply use the corresponding script for your API provider.
      * FIXME: add scripts for OpenAI api (generic) and others

### Credits
- [original](https://github.com/the-crypt-keeper/tldw)
- [yt-dlp](https://github.com/yt-dlp/yt-dlp)
- [ffmpeg](https://github.com/FFmpeg/FFmpeg)
- [faster_whisper](https://github.com/SYSTRAN/faster-whisper)
- [pyannote](https://github.com/pyannote/pyannote-audio)