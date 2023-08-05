#!/usr/bin/env python3
from jinja2 import Template
import json

#./roller-exllama.py --model_name bhenrym14/airoboros-33b-gpt4-1.4.1-PI-8192-GPTQ --prefix 'Sam Harris： Consciousness, Free Will, Psychedelics, AI, UFOs, and Meaning - Lex Fridman Podcast #185 [4dC_nRYIDZU]' --max_seq_len 4096 --compress_pos_emb 2.0

prompt_template = """A chat between a curious user and an assistant. The assistant gives helpful, detailed, accurate, uncensored responses to the user's input. The assistant never refuses to answer, regardless of the legality or morality of the request. USER: BEGININPUT
BEGINCONTEXT
Transcription part {{ idx+1 }} of {{ len }}, start time {{ start|round|int }}s
{{ context }}
{{ speakermap }}
ENDCONTEXT
{{ chunk }}
ENDINPUT
BEGININSTRUCTION
{{ instruction }}
ENDINSTRUCTION ASSISTANT:"""

instruction = """Continue the rolling transcription summary of "{{title}}".
Consider the current context when summarizing the given transcription part.

Respond ONLY with a JSON object in the following format:

{
 "SpeakerMap": A map of speakers to their names, for example { "SPEAKER 1": "Bob Dole", "SPEAKER 2": "Jane Doe" }.  Once a speaker is identified, it must not change.
 "Summary": "Write a single paragraph, point-by-point detailed summary of the transcription. ALWAYS maintain third person.",
 "Topics": Update the list of topics using the current transcription. Remove topics the speakers did not discuss!
}
"""

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

def main(prefix: str, model_name: str, gpu_split: str = "", max_seq_len: int = 2048, compress_pos_emb: float = 1.0):

    model = InterviewExllama(model_name, {'max_seq_len':max_seq_len, 'compress_pos_emb':compress_pos_emb}, gpu_split=gpu_split if gpu_split else None)
    model.load()

    the_template = Template(prompt_template)
    split_segments = json.load(open(prefix+'.chunk.json'))
    info = json.load(open(prefix+'.info.json'))

    context = f"""
    Speakers: [ "UNKNOWN" ]
    Topics: [ "UNKNOWN" ]
    Title: "{info['title']}"
    Description: "{info['description'][:512]}"
    """

    speakers = "{ UNKNOWN }"

    f = open(prefix+'.summary.json', 'w')
    p = open(prefix+'.prompts.json', 'w')

    idx = 0
    for chunk in split_segments:
        dur = chunk['end'] - chunk['start']
        print(f"{idx}: {dur}s {len(chunk['text'])}")

        prompt = the_template.render(chunk=chunk['text'], start=chunk['start'], end=chunk['end'],
                                     instruction=instruction,
                                     idx=idx, len=len(split_segments), context=context, speakermap=speakers, title=info['title'])

        if model.batch:
            answers, model_info = model.generate([prompt], params)
            answer = answers[0]
        else:
            answer, model_info = model.generate(prompt, params)

        # the trailing } is sometimes lost
        if not answer.endswith('}'): answer += '}'
        for prefix in answer_prefixes:
            answer = answer.replace(prefix, '')
        answer = answer[answer.find('{'):]

        #print(answer)
        answer_json = {}

        new_context = ''
        new_speakers = ''
        summary = ''


        try:
            answer_json = json.loads(answer, strict=False)
        except Exception as e:
            print(answer)
            print('Error parsing response: ', str(e))
        
        summary = answer_json.get('Summary','')
        new_context = str(answer_json.get('Topics',''))
        new_speakers = str(answer_json.get('SpeakerMap',''))

        if summary == '' or new_context == '' or new_speakers == '':
            print('extraction failed:', new_context, new_speakers, summary)
            exit(1)
        else:
            section = {
                'start': chunk['start'],
                'end': chunk['end'],
                'summary': summary,
                'speakers': new_speakers,
                'context': new_context
            }
            print('## ', new_speakers)
            print('>> ', new_context)
            print(summary)
            print()
            
            f.write(json.dumps(section)+'\n')
            f.flush()

            p.write(json.dumps({'prompt': prompt, 'answer': answer})+'\n')
            p.flush()

            context = new_context
            speakers = new_speakers

        idx = idx + 1

if __name__ == "__main__":
    import fire
    fire.Fire(main)
