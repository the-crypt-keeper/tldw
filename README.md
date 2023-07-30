# Too Long, Didnt Watch

WIP experiments in summarizing long youtube videos.

[HF Space](https://huggingface.co/spaces/mike-ravkine/too-long-didnt-watch)

## Whats going on here?

1. Use whisper.cpp to transcribe audio to text (see `ufo-clean.txt`)
2. Trim the relevant section of text (see `ufo-clean-parts.txt`)
3. Break text into chunks
4. Summarize each chunk
5. Profit?