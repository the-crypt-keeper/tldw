import os
from typing import List, Tuple, Optional
from openai import OpenAI
import tiktoken
from tqdm import tqdm



# Open dataset
with open(".\\tldw-original-scripts\\Samples\\ai_wikipedia.txt", "r") as file:
    artificial_intelligence = file.read()

# load encoding and check length of dataset
encoding = tiktoken.encoding_for_model('gpt-4-turbo')
print(len(encoding.encode(artificial_intelligence)))


# Call wrapper to OpenAI
client = OpenAI(api_key="")

def get_chat_completion(messages, model='gpt-4-turbo'):
    response = client.chat.completions.create(
        model = model,
        messages = messages,
        temperature = 0,
    )
    return response.choices[0].message.content


# Message Chunking <----- THE JUICY STUFF
def tokenize(text: str) -> List[str]:
    encoding = tiktoken.encoding_for_model('gpt-4-turbo')
    return encoding.encode(text)

# This function chunks a text into smaller pieces based on a maximum token count and a delimiter
def chunk_on_delimiter(input_string: str,
                       max_tokens: int,
                       delimiter: str) -> List[str]:
    chunks = input_string.split(delimiter)
    combined_chunks, _, dropped_chunk_count = combine_chunks_with_no_minimum(
        chunks, max_tokens, chunk_delimiter=delimiter, add_ellipsis_for_overflow=True
    )
    if dropped_chunk_count > 0:
        print(f"Warning: {dropped_chunk_count} chunks were dropped due to exceeding the token limit.")
    combined_chunks = [f"{chunk}{delimiter}" for chunk in combined_chunks]
    return combined_chunks

# This function combines text chunks into larger blocks without exceeding a specified token count.
#   It returns the combined chunks, their original indices, and the number of dropped chunks due to overflow.
def combine_chunks_with_no_minimum(
        chunks: List[str],
        max_tokens: int,
        chunk_delimiter: str = "\n\n",
        header: Optional[str] = None,
        add_ellipsis_for_overflow: bool = False,


