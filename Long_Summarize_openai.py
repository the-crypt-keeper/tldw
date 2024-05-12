from typing import List, Tuple, Optional
from openai import OpenAI
import tiktoken
from tqdm import tqdm


# script from: https://github.com/openai/openai-cookbook/blob/main/examples/Summarizing_long_documents.ipynb


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
        model=model,
        messages=messages,
        temperature=0,
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
        chunk_delimiter="\n\n",
        header: Optional[str] = None,
        add_ellipsis_for_overflow=False,
) -> Tuple[List[str], List[int]]:
    dropped_chunk_count = 0
    output = []  # list to hold the final combined chunks
    output_indices = []  # list to hold the indices of the final combined chunks
    candidate = (
        [] if header is None else [header]
    )  # list to hold the current combined chunk candidate
    candidate_indices = []
    for chunk_i, chunk in enumerate(chunks):
        chunk_with_header = [chunk] if header is None else [header, chunk]
        if len(tokenize(chunk_delimiter.join(chunk_with_header))) > max_tokens:
            print(f"warning: chunk overflow")
            if (
                    add_ellipsis_for_overflow
                    and len(tokenize(chunk_delimiter.join(candidate + ["..."]))) <= max_tokens
            ):
                candidate.append("...")
                dropped_chunk_count += 1
            continue  # this case would break downstream assumptions
        # estimate token count with the current chunk added
        extended_candidate_token_count = len(tokenize(chunk_delimiter.join(candidate + [chunk])))
        # If the token count exceeds max_tokens, add the current candidate to output and start a new candidate
        if extended_candidate_token_count > max_tokens:
            output.append(chunk_delimiter.join(candidate))
            output_indices.append(candidate_indices)
            candidate = chunk_with_header  # re-initialize candidate
            candidate_indices = [chunk_i]
        # otherwise keep extending the candidate
        else:
            candidate.append(chunk)
            candidate_indices.append(chunk_i)
    # add the remaining candidate to output if it's not empty
    if (header is not None and len(candidate) > 1) or (header is None and len(candidate) > 0):
        output.append(chunk_delimiter.join(candidate))
        output_indices.append(candidate_indices)
    return output, output_indices, dropped_chunk_count


def summarize(text: str,
              detail: float = 0,
              model: str = 'gpt-4-turbo',
              additional_instructions: Optional[str] = None,
              minimum_chunk_size: Optional[int] = 500,
              chunk_delimiter: str = ".",
              summarize_recursively=False,
              verbose=False):
    """
    Summarizes a given text by splitting it into chunks, each of which is summarized individually.
    The level of detail in the summary can be adjusted, and the process can optionally be made recursive.

    Parameters: - text (str): The text to be summarized. - detail (float, optional): A value between 0 and 1
    indicating the desired level of detail in the summary. 0 leads to a higher level summary, and 1 results in a more
    detailed summary. Defaults to 0. - model (str, optional): The model to use for generating summaries. Defaults to
    'gpt-3.5-turbo'. - additional_instructions (Optional[str], optional): Additional instructions to provide to the
    model for customizing summaries. - minimum_chunk_size (Optional[int], optional): The minimum size for text
    chunks. Defaults to 500. - chunk_delimiter (str, optional): The delimiter used to split the text into chunks.
    Defaults to ".". - summarize_recursively (bool, optional): If True, summaries are generated recursively,
    using previous summaries for context. - verbose (bool, optional): If True, prints detailed information about the
    chunking process.

    Returns:
    - str: The final compiled summary of the text.

    The function first determines the number of chunks by interpolating between a minimum and a maximum chunk count
    based on the `detail` parameter. It then splits the text into chunks and summarizes each chunk. If
    `summarize_recursively` is True, each summary is based on the previous summaries, adding more context to the
    summarization process. The function returns a compiled summary of all chunks.
    """

    # check detail is set correctly
    assert 0 <= detail <= 1

    # interpolate the number of chunks based to get specified level of detail
    max_chunks = len(chunk_on_delimiter(text, minimum_chunk_size, chunk_delimiter))
    min_chunks = 1
    num_chunks = int(min_chunks + detail * (max_chunks - min_chunks))

    # adjust chunk_size based on interpolated number of chunks
    document_length = len(tokenize(text))
    chunk_size = max(minimum_chunk_size, document_length // num_chunks)
    text_chunks = chunk_on_delimiter(text, chunk_size, chunk_delimiter)
    if verbose:
        print(f"Splitting the text into {len(text_chunks)} chunks to be summarized.")
        print(f"Chunk lengths are {[len(tokenize(x)) for x in text_chunks]}")

    # set system message
    system_message_content = "Rewrite this text in summarized form."
    if additional_instructions is not None:
        system_message_content += f"\n\n{additional_instructions}"

    accumulated_summaries = []
    for chunk in tqdm(text_chunks):
        if summarize_recursively and accumulated_summaries:
            # Creating a structured prompt for recursive summarization
            accumulated_summaries_string = '\n\n'.join(accumulated_summaries)
            user_message_content = f"Previous summaries:\n\n{accumulated_summaries_string}\n\nText to summarize next:\n\n{chunk}"
        else:
            # Directly passing the chunk for summarization without recursive context
            user_message_content = chunk

        # Constructing messages based on whether recursive summarization is applied
        messages = [
            {"role": "system", "content": system_message_content},
            {"role": "user", "content": user_message_content}
        ]

        # Assuming this function gets the completion and works as expected
        response = get_chat_completion(messages, model=model)
        accumulated_summaries.append(response)

    # Compile final summary from partial summaries
    final_summary = '\n\n'.join(accumulated_summaries)

    return final_summary

# Summary at 0 detail
summary_with_detail_0 = summarize(artificial_intelligence, detail=0, verbose=True)


# Summary at 0.25 detail
summary_with_detail_pt25 = summarize(artificial_intelligence, detail=0.25, verbose=True)


# Summary at 0.5 detail
summary_with_detail_pt5 = summarize(artificial_intelligence, detail=0.5, verbose=True)


# Summary at 0.75 detail
summary_with_detail_pt75 = summarize(artificial_intelligence, detail=0.75, verbose=True)


# Summart at 1 detail
summary_with_detail_1 = summarize(artificial_intelligence, detail=1, verbose=True)


# Lengths of summaries:
[len(tokenize(x)) for x in
 [summary_with_detail_0, summary_with_detail_pt25, summary_with_detail_pt5, summary_with_detail_pt75, summary_with_detail_1]]

# print 0 detail summary
print(summary_with_detail_0)


# print 0.25 detail summary
print(summary_with_detail_pt25)


# print 0.5 detail summary
print(summary_with_detail_pt5)


# print 0.75 detail summary
print(summary_with_detail_pt75)


# print 1.0 detail summary
print(summary_with_detail_1)


# Print summary using additional instructions:
summary_with_additional_instructions = summarize(artificial_intelligence, detail=0.1,
                                                 additional_instructions="Write in point form and focus on numerical data.")
print(summary_with_additional_instructions)


# Print summary using recursive summarization:
recursive_summary = summarize(artificial_intelligence, detail=0.1, summarize_recursively=True)
print(recursive_summary)

