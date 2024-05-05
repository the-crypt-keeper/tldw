# TL/DW: Too Long, Didnt Watch

Take a URL, single video, list of URLs, or list of local videos + URLs and feed it into the script and have each video transcribed (and downloaded if not local) using faster-whisper. Transcriptions can then be shuffled off to an LLM API endpoint of your choice, whether that be local or remote. Any site supported by yt-dl is supported, so you can use this with sites besides just youtube.

Original: `YouTube contains an incredible amount of knowledge, much of which is locked inside multi-hour videos.  Let's extract and summarize it with AI!`

### tl/dr:
- Use the script to transcribe a local file or remote url. Any url youtube-dl supports _should_ work. If you pass an OpenAPI endpoint as a second argument, and add your API key to the config file, you can have your resulting transcriptions summarized as well.
  * The current approach to summarization is currently 'dumb'/naive, and will likely be replaced or additional functionality added to reflect actual practices and not just 'dump txt in and get an answer' approach.

Save time and use the `config.txt` file, it allows you to set these settings and have them used when ran.
```
usage: diarize.py [-h] [--api_url API_URL] [--num_speakers NUM_SPEAKERS] [--whisper_model WHISPER_MODEL]
                  [--offset OFFSET] [--vad_filter]
                  [input_path]

positional arguments:
  input_path            Path or URL of the video

options:
  -h, --help            show this help message and exit
  --api_url API_URL     API URL for summarization (optional)
  --num_speakers NUM_SPEAKERS
                        Number of speakers (default: 2)
  --whisper_model WHISPER_MODEL
                        Whisper model (default: small.en)
  --offset OFFSET       Offset in seconds (default: 0)
  --vad_filter          Enable VAD filter
```


### Pieces
- **Workflow**
  1. Setup python + packages
  2. Setup ffmpeg
  3. Run `python diarize.py <video_url>` or `python diarize.py <List_of_videos.txt>`
  4. If you want summarization, add your API keys (if needed[is needed for now]) to the `config.txt` file, and then re-run the script, passing in the name of the API [or URL endpoint - to be added] to the script.
    * `python diarize.py https://www.youtube.com/watch?v=4nd1CDZP21s --api_name anthropic` - This will attempt to download the video, then upload the resulting json file to the anthropic API endpoint, referring to values set in the config file (API key and model) to request summarization.
    - OpenAI: 
    - Anthropic:
      * Opus: `claude-3-opus-20240229`
      * Sonnet: `claude-3-sonnet-20240229`
      * Haiku: `claude-3-haiku-20240307`
    - Cohere: 

### What's in the repo?
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

### Usage
- Single file (remote URL) transcription
  * Single URL: `python diarize.py https://example.com/video.mp4`
- Single file (local) transcription)
  * Transcribe a local file: `python diarize.py /path/to/your/localfile.mp4`
- Multiple files (local & remote)
  * List of Files(can be URLs and local files mixed): `python diarize.py ./path/to/your/text_file.txt"`


### Credits
- [original](https://github.com/the-crypt-keeper/tldw)
- [yt-dlp](https://github.com/yt-dlp/yt-dlp)
- [ffmpeg](https://github.com/FFmpeg/FFmpeg)
- [faster_whisper](https://github.com/SYSTRAN/faster-whisper)
- [pyannote](https://github.com/pyannote/pyannote-audio)