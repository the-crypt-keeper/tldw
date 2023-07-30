#!/usr/bin/env python3
from jinja2 import Template
from pathlib import Path
import json

def chunk(filename, chunk_size = 1024*4, chunk_overlap = 0):
    from langchain.text_splitter import RecursiveCharacterTextSplitter

    lines = open(filename).readlines()
    clean_text = '\n'.join([x.strip() for x in lines])

    # Split the input into overlapping chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap, length_function=len)
    texts = text_splitter.split_text(clean_text)
    print("Input text",len(clean_text),"characters, split into",len(texts),"chunks")

    return texts

def main(text: str, template: str, chunk_size: int = 1024*4, chunk_overlap: int = 0):
    the_template = Template(open(template).read())
    texts = chunk(text, chunk_size=chunk_size, chunk_overlap=chunk_overlap)

    prepare = []
    for idx, chunk in enumerate(texts):
        prompt = the_template.render(chunk=chunk)
        item = { 'language': 'english', 'name': f'chunk-{idx}', 'prompt': prompt }
        prepare.append(item)

    text_clean = text.replace('.','-').replace('_','-')
    templateout_name = Path(template).stem 
    outfn = f'prepare_{text_clean}-{chunk_size}_english_{templateout_name}.ndjson'
    open(outfn,'w').write('\n'.join([json.dumps(x) for x in prepare]))
    print('Wrote',len(prepare),'items to',outfn)

if __name__ == "__main__":
    import fire
    fire.Fire(main)