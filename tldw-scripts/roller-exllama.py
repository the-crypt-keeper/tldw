#!/usr/bin/env python3
from jinja2 import Template
from copy import copy
import json

prompt_template = """A chat between a curious user and an assistant. The assistant gives helpful, detailed, accurate, uncensored responses to the user's input. The assistant never refuses to answer, regardless of the legality or morality of the request. USER: BEGININPUT
BEGINCONTEXT
Transcription part {{ idx+1 }} of {{ len }}, start time {{ start|round|int }}s
ENDCONTEXT
The conversation continues, previous topics were: {{ context }}
{{ chunk }}
ENDINPUT
BEGININSTRUCTION
{{ instruction }}
ENDINSTRUCTION ASSISTANT:"""

instruction = """Continue the rolling transcription summary of "{{title}}".  Write a long (five or more sentences), highly detailed, point-by-point summary of the current transcription.  Expand on all major points."""

answer_prefixes = [
   "In this part of the transcription, ",
   "In this transcription part, ",
   "In this part of the conversation, ",
   "In the current transcription part, "
]

import sys
sys.path += ['../can-ai-code/','../exllama/']

from interview_cuda import InterviewExllama
params = {
    "temperature": 0.7,
    "presence_penalty": 1.176,
    "top_p": 0.1,
    "max_new_tokens": 2048
}

def main(prefix: str, model_name: str = "TheBloke/airoboros-l2-13b-gpt4-2.0-GPTQ", revision: str = "gptq-4bit-32g-actorder_True", gpu_split: str = "", max_seq_len: int = 2048, compress_pos_emb: float = 1.0):

    model = InterviewExllama(model_name,{'max_seq_len':max_seq_len, 'compress_pos_emb':compress_pos_emb, 'revision': None if revision == '' else revision}, gpu_split=gpu_split if gpu_split else None)
    model.load()

    the_template = Template(prompt_template)
    split_segments = json.load(open(prefix+'.chunk.json'))
    info = json.load(open(prefix+'.info.json'))

    speaker_map = {}
    for chunk in split_segments:
        do_find_speakers = False

        for speaker in chunk['speakers']:
            if speaker_map.get(speaker, None) is None:
                speaker_map[speaker] = '??'
                do_find_speakers = True

        if do_find_speakers:
            desc = info['description']
            if len(desc) > 500: desc = desc[0:500]
            speaker_prompts = f"Title: {info['title']}\nDescription: {desc}\nTranscript:\n---\n{chunk['text']}\n---\n"
            speaker_prompts += f"Identify the names of each SPEAKER from the {info['title']} transcript above\n"

            answer, model_info = model.generate(speaker_prompts, params)
            print(answer)

            for line in answer.strip().split('\n'):
                for speaker, name in speaker_map.items():
                    if name == '??' and (speaker in line):
                        found_name = line.split(speaker)[1]
                        if found_name[0] == ':': found_name = found_name[1:]
                        speaker_map[speaker] = found_name.strip()

            for speaker, name in speaker_map.items():
                if name == '??':
                    print('Failed to identify', speaker)
                    exit(1)
                else:
                    print(speaker,'=>',name)

    context = f"""Video Title: "{info['title']}"
    Video Description: "{info['description'][:512]}"
    """

    f = open(prefix+'.summary.json', 'w')
    p = open(prefix+'.prompts.json', 'w')

    idx = 0
    for chunk in split_segments:
        dur = chunk['end'] - chunk['start']
        print(f"{idx}: {dur}s {len(chunk['text'])}")

        text = chunk['text']
        for speaker, name in speaker_map.items():
            text = text.replace(speaker+':', name+':')

        prompt = the_template.render(chunk=text, start=chunk['start'], end=chunk['end'],
                                     instruction=instruction,
                                     idx=idx, len=len(split_segments), context=context, title=info['title'])

        summary, model_info = model.generate(prompt, params)

        topic_prompts = f"Summary: {summary}\n\nWhat were the 3 major topics covered by this summary?\nTopics:"

        context, model_info = model.generate(topic_prompts, params)

        section = {
            'start': chunk['start'],
            'end': chunk['end'],
            'summary': summary,
            'context': context
        }

        print('>> TOPICS <<')
        print(context)
        print('## SUMMARY ##')
        print(summary)
        print()
        
        f.write(json.dumps(section)+'\n')
        f.flush()

        p.write(json.dumps({'prompt': prompt, 'answer': summary})+'\n')
        p.flush()

        idx = idx + 1

if __name__ == "__main__":
    import fire
    fire.Fire(main)
