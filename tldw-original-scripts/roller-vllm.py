#!/usr/bin/env python3
from jinja2 import Template
import json

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

instruction_v1 = """Continue the rolling transcription summary of "{{title}}".
Consider the current context when summarizing the given transcription part.
Respond ONLY with a JSON object with 3 keys in the following format:
{
 Speaker-Map: A map of speakers to their names, for example { "SPEAKER 1": "Bob Dole", "SPEAKER 2": "Jane Doe" }.  Once a speaker is identified, it must not change.
 Next-Context: "An updated context for the next part of the transcription. Always include the speakers and the current topics of discussion.",
 Summary: "A detailed, point-by-point summary of the current transcription."
}
"""

# this gives longer replies but has a much higher chance of stopping in the middle
# re-investigate when splitting the prompts
# Summary: "A detailed, point-by-point summary of the current transcription.  Include details of major points.  Write at least 3 sentences but no more then 6 sentences.",

instruction = """Continue the rolling transcription summary of "{{title}}".
Consider the current context when summarizing the given transcription part.
Respond ONLY with a JSON object with 3 keys in the following format:
{
 Speaker-Map: A map of speakers to their names, for example { "SPEAKER 1": "Bob Dole", "SPEAKER 2": "Jane Doe" }.  Once a speaker is identified, it must not change.
 Summary: "A detailed, point-by-point summary of the current transcription.  Include details of major points.  Write at least three sentences and no more then six sentences. ALWAYS maintain third person.",
 Next-Context: "List of topics from the transcription Summary above."
}
"""

answer_prefixes = [
   "In this part of the transcription, ",
   "In this transcription part, ",
   "In this part of the conversation, ",
   "In the current transcription part, "
]

import sys
sys.path.append('../can-ai-code/')
from interview_cuda import InterviewVLLM
params = {
    "temperature": 0.7,
    "presence_penalty": 1.176,
    "top_p": 0.1,
    "max_tokens": 2048
}

def main(prefix: str, model_name: str, gpu_split: str = "", init_speakers: str = "", max_seq_len: int = 2048, ):

    model = InterviewVLLM(model_name, json.loads(info), gpu_split=gpu_split if gpu_split else None)
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
        new_context = str(answer_json.get('Next-Context',''))
        new_speakers = str(answer_json.get('Speaker-Map',''))

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
