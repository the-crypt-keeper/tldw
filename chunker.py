#!/usr/bin/env python3
from jinja2 import Template
from pathlib import Path
import string
import json

def segment_merger(filename, max_text_len = 1000):
    segments = json.load(open(filename))

    text = ''
    last_segment = { 'speaker': None }
    start_time = None
    stop_chars = string.punctuation.replace(',','')

    for segment in segments:
        early_break = (max_text_len > 0) and (len(text) > max_text_len) and (text[-1] in stop_chars)
        if last_segment['speaker'] != segment['speaker'] or early_break:
            if text != '':
                yield { 'speaker': last_segment['speaker'], 'text': text, 'start': start_time, 'end': last_segment['end'] }
            text = segment['text'].lstrip()
            start_time = segment['start']
        else:
            text += segment['text']
        last_segment = segment

    if text != '':
        yield { 'speaker': last_segment['speaker'], 'text': text, 'start': start_time, 'end': last_segment['end'] }

def time_splitter(merged_segments, chunk_size = 300):
    start_time = None
    text = ''

    for segment in merged_segments:
        if start_time is None:
            start_time = segment['start']
        text += f"{segment['speaker']}: {segment['text']}\n"
        if segment['end'] - start_time >= chunk_size:
            yield { 'text': text, 'start': start_time, 'end': segment['end'] }
            start_time = None
            text = ''

def main(prefix: str, chunk_size: int = 300, max_text_len: int = 1000):
    merged_segments = list(segment_merger(prefix+'.diarize.json', max_text_len))
    split_segments = list(time_splitter(merged_segments, chunk_size))
    with open(prefix+'.chunk.json', 'w') as f:
        json.dump(split_segments, f)
    print(f"Wrote {len(split_segments)} chunks to {prefix}.chunk.json")

if __name__ == "__main__":
    import fire
    fire.Fire(main)