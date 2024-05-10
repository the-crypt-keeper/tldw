import json
import sys

in_file = sys.argv[1]
with open(in_file) as infile:
    chunks = [json.loads(line) for line in infile.readlines()]

def part_to_time(part):
    mins = part*5
    oh = mins // 60
    om = mins % 60
    return f'{oh:02}:{om:02}'

text = ''
for idx, chunk in enumerate(chunks):
    #text += f'\n\n[{part_to_time(idx)} - {part_to_time(idx+1)}] '
    text += f'\nSection {idx+1}: {chunk["answer"]}\n'

out_file = in_file.replace('ndjson','txt')
with open(out_file,'w') as outfile:
    outfile.write(text)

from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained('hf-internal-testing/llama-tokenizer', use_fast = True)
logits = tokenizer.encode(text)

print('chunks:', len(chunks))
print('summary bytes:', len(text))
print('summary tokens:', len(logits))