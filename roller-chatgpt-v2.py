#!/usr/bin/env python3
from jinja2 import Template
import json

prompt_template = """
Continue the rolling transcription summary of "{{title}}".  Consider the current context when summarizing the given transcription part.

### Context: {{ context }}
Speaker-Map: {{ speakermap }}

### Transcription part {{ idx }} of {{ len }}, start time {{ start }}:
{{ chunk }}

### Instruction: Using the Context above, analyze the Trasncription and respond with a JSON object in this form:

{
    "Speaker-Map": { "SPEAKER 1": "Bob Dole", "SPEAKER 2": "Jane Doe" } // A map of speakers to their names, make sure to remember all previous speakers.
    "Next-Context": "..." // An updated context for the next part of the transcription. Always include the speakers and the current topics of discussion.
    "Summary": "..." // A detailed, point-by-point summary of the current transcription.
}
"""

from openai import OpenAI

client = OpenAI()

def main(prefix: str, init_speakers: str = ""):
    the_template = Template(prompt_template)

    split_segments = json.load(open(prefix+'.chunk.json'))
    info = json.load(open(prefix+'.info.json'))

    context = f"""
    Video Title: {info['title']}
    Video Description: {info['description'][:1024]}
    """
    
    speakers = "{ UNKNOWN }"

    f = open(prefix+'.summary.json', 'w')
    idx = 0
    for chunk in split_segments:
        dur = chunk['end'] - chunk['start']
        print(f"{idx}: {dur}s {len(chunk)}")

        prompt = the_template.render(chunk=chunk['text'], start=chunk['start'], end=chunk['end'],
                                     idx=idx, len=len(split_segments), context=context, speakermap=speakers, title=info['title'])
        
        messages = [{'role': 'user', 'content': prompt }]
        response = client.chat.completions.create(messages=messages,model='gpt-3.5-turbo-1106',temperature=0.1,max_tokens=1024, response_format={ "type": "json_object" })

        answer = response.choices[0].message.content
        
        parsed = json.loads(answer)
        
        summary = parsed.get('Summary','')
        new_speakers = parsed.get('Speaker-Map','')
        new_context = parsed.get('Next-Context','')
        
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

            context = new_context
            speakers = new_speakers

        idx = idx + 1

if __name__ == "__main__":
    import fire
    fire.Fire(main)