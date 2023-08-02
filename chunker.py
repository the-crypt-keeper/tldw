#!/usr/bin/env python3
from jinja2 import Template
from pathlib import Path
import json

def whisper_chunker(filename, chunk_size = 300):
    lines = json.load(open(filename))
    texts = []

    text = ''
    start_time = None

    if 'transcription' in lines:
        sections = [ { 'text': x['text'], 'from': x['timestamps']['from']/1000.0, 'to': x['timestamps']['to']/1000.0  } for x in lines['transcription'] ]
    elif 'segments' in lines:
        sections = [ { 'text': x['text'], 'from': x['start'], 'to': x['end']  } for x in lines['segments'] ]
    else:
        raise Exception('No transcription or segments found')
    
    for section in sections:
        text += section['text']
        if start_time is None: start_time = section['from']
        if section['to'] - start_time > chunk_size:
            texts.append(text)
            text = ''
            start_time = section['from']
    texts.append(text)

    return texts

def main(text: str, template: str, chunk_size: int = 300):
    the_template = Template(open(template).read())
    texts = whisper_chunker(text, chunk_size=chunk_size)

    prepare = []
    for idx, chunk in enumerate(texts):
        prompt = the_template.render(chunk=chunk)
        item = { 'language': 'english', 'name': f'chunk-{idx}', 'prompt': prompt }
        prepare.append(item)

    text_clean = Path(template).stem.replace('.','-').replace('_','-')
    templateout_name = Path(template).stem 
    outfn = f'results/prepare_{text_clean}-{chunk_size}_english_{templateout_name}.ndjson'
    open(outfn,'w').write('\n'.join([json.dumps(x) for x in prepare]))
    print('Wrote',len(prepare),'items to',outfn)

if __name__ == "__main__":
    import fire
    fire.Fire(main)