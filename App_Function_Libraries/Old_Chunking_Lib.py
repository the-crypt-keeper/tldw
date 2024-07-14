# Old_Chunking_Lib.py
#########################################
# Old Chunking Library
# This library is used to handle chunking of text for summarization.
#
####
import logging
####################
# Function List
#
# 1. chunk_transcript(transcript: str, chunk_duration: int, words_per_second) -> List[str]
# 2. summarize_chunks(api_name: str, api_key: str, transcript: List[dict], chunk_duration: int, words_per_second: int) -> str
# 3. get_chat_completion(messages, model='gpt-4-turbo')
# 4. chunk_on_delimiter(input_string: str, max_tokens: int, delimiter: str) -> List[str]
# 5. combine_chunks_with_no_minimum(chunks: List[str], max_tokens: int, chunk_delimiter="\n\n", header: Optional[str] = None, add_ellipsis_for_overflow=False) -> Tuple[List[str], List[int]]
# 6. rolling_summarize(text: str, detail: float = 0, model: str = 'gpt-4-turbo', additional_instructions: Optional[str] = None, minimum_chunk_size: Optional[int] = 500, chunk_delimiter: str = ".", summarize_recursively=False, verbose=False)
# 7. chunk_transcript(transcript: str, chunk_duration: int, words_per_second) -> List[str]
# 8. summarize_chunks(api_name: str, api_key: str, transcript: List[dict], chunk_duration: int, words_per_second: int) -> str
#
####################

# Import necessary libraries
import os
from typing import Optional, List, Tuple
#
# Import 3rd party
from openai import OpenAI
from App_Function_Libraries.Tokenization_Methods_Lib import openai_tokenize
#
# Import Local
#
#######################################################################################################################
# Function Definitions
#

######### Words-per-second Chunking #########
def chunk_transcript(transcript: str, chunk_duration: int, words_per_second) -> List[str]:
    words = transcript.split()
    words_per_chunk = chunk_duration * words_per_second
    chunks = [' '.join(words[i:i + words_per_chunk]) for i in range(0, len(words), words_per_chunk)]
    return chunks


# def summarize_chunks(api_name: str, api_key: str, transcript: List[dict], chunk_duration: int,
#                      words_per_second: int) -> str:
#     if api_name not in summarizers:  # See 'summarizers' dict in the main script
#         return f"Unsupported API: {api_name}"
#
#     summarizer = summarizers[api_name]
#     text = extract_text_from_segments(transcript)
#     chunks = chunk_transcript(text, chunk_duration, words_per_second)
#
#     summaries = []
#     for chunk in chunks:
#         if api_name == 'openai':
#             # Ensure the correct model and prompt are passed
#             summaries.append(summarizer(api_key, chunk, custom_prompt))
#         else:
#             summaries.append(summarizer(api_key, chunk))
#
#     return "\n\n".join(summaries)


################## ####################


######### Token-size Chunking ######### FIXME - OpenAI only currently
# This is dirty and shameful and terrible. It should be replaced with a proper implementation.
# anyways lets get to it....
openai_api_key = "Fake_key" # FIXME
client = OpenAI(api_key=openai_api_key)





# This function chunks a text into smaller pieces based on a maximum token count and a delimiter
def chunk_on_delimiter(input_string: str,
                       max_tokens: int,
                       delimiter: str) -> List[str]:
    chunks = input_string.split(delimiter)
    combined_chunks, _, dropped_chunk_count = combine_chunks_with_no_minimum(
        chunks, max_tokens, chunk_delimiter=delimiter, add_ellipsis_for_overflow=True)
    if dropped_chunk_count > 0:
        print(f"Warning: {dropped_chunk_count} chunks were dropped due to exceeding the token limit.")
    combined_chunks = [f"{chunk}{delimiter}" for chunk in combined_chunks]
    return combined_chunks





#######################################


######### Words-per-second Chunking #########
# FIXME - WHole section needs to be re-written
def chunk_transcript(transcript: str, chunk_duration: int, words_per_second) -> List[str]:
    words = transcript.split()
    words_per_chunk = chunk_duration * words_per_second
    chunks = [' '.join(words[i:i + words_per_chunk]) for i in range(0, len(words), words_per_chunk)]
    return chunks


# def summarize_chunks(api_name: str, api_key: str, transcript: List[dict], chunk_duration: int,
#                      words_per_second: int) -> str:
    # if api_name not in summarizers:  # See 'summarizers' dict in the main script
    #     return f"Unsupported API: {api_name}"
    #
    # if not transcript:
    #     logging.error("Empty or None transcript provided to summarize_chunks")
    #     return "Error: Empty or None transcript provided"
    #
    # text = extract_text_from_segments(transcript)
    # chunks = chunk_transcript(text, chunk_duration, words_per_second)
    #
    # #FIXME
    # custom_prompt = args.custom_prompt
    #
    # summaries = []
    # for chunk in chunks:
    #     if api_name == 'openai':
    #         # Ensure the correct model and prompt are passed
    #         summaries.append(summarize_with_openai(api_key, chunk, custom_prompt))
    #     elif api_name == 'anthropic':
    #         summaries.append(summarize_with_cohere(api_key, chunk, anthropic_model, custom_prompt))
    #     elif api_name == 'cohere':
    #         summaries.append(summarize_with_anthropic(api_key, chunk, cohere_model, custom_prompt))
    #     elif api_name == 'groq':
    #         summaries.append(summarize_with_groq(api_key, chunk, groq_model, custom_prompt))
    #     elif api_name == 'llama':
    #         summaries.append(summarize_with_llama(llama_api_IP, chunk, api_key, custom_prompt))
    #     elif api_name == 'kobold':
    #         summaries.append(summarize_with_kobold(kobold_api_IP, chunk, api_key, custom_prompt))
    #     elif api_name == 'ooba':
    #         summaries.append(summarize_with_oobabooga(ooba_api_IP, chunk, api_key, custom_prompt))
    #     elif api_name == 'tabbyapi':
    #         summaries.append(summarize_with_vllm(api_key, tabby_api_IP, chunk, summarize.llm_model, custom_prompt))
    #     elif api_name == 'local-llm':
    #         summaries.append(summarize_with_local_llm(chunk, custom_prompt))
    #     else:
    #         return f"Unsupported API: {api_name}"
    #
    # return "\n\n".join(summaries)

# FIXME - WHole section needs to be re-written
def summarize_with_detail_openai(text, detail, verbose=False):
    summary_with_detail_variable = rolling_summarize(text, detail=detail, verbose=True)
    print(len(openai_tokenize(summary_with_detail_variable)))
    return summary_with_detail_variable


def summarize_with_detail_recursive_openai(text, detail, verbose=False):
    summary_with_recursive_summarization = rolling_summarize(text, detail=detail, summarize_recursively=True)
    print(summary_with_recursive_summarization)

#
#
#################################################################################
