# Too Long, Didnt Watch

WIP experiments in summarizing long youtube videos.

[HF Space](https://huggingface.co/spaces/mike-ravkine/too-long-didnt-watch)

## Whats going on here?

1. Downloaded https://www.youtube.com/watch?v=KQ7Dw-739VY
2. Used whisper.cpp to transcribe audio to text (see `ufo-clean.txt`)
3. Trim the relevant section of text (see `ufo-clean-parts.txt`)
4. Break text into chunks
5. Summarize each chunk
6. Profit?

## Step-by-step

### Download audio (m4a)

`pip install `

### Transcode audio (wav)

`ffmpeg ..`
`
### Transcribe with whisper.cpp

`main -m models/.. -t <threads>`

### Chunk

### Process
