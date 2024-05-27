# Old-Chunking-Lib.py
#########################################
# Old Chunking Library
# This library is used to handle chunking of text for summarization.
#
####



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




#######################################################################################################################
# Function Definitions
#

######### Words-per-second Chunking #########
def chunk_transcript(transcript: str, chunk_duration: int, words_per_second) -> List[str]:
    words = transcript.split()
    words_per_chunk = chunk_duration * words_per_second
    chunks = [' '.join(words[i:i + words_per_chunk]) for i in range(0, len(words), words_per_chunk)]
    return chunks


def summarize_chunks(api_name: str, api_key: str, transcript: List[dict], chunk_duration: int,
                     words_per_second: int) -> str:
    if api_name not in summarizers:  # See 'summarizers' dict in the main script
        return f"Unsupported API: {api_name}"

    summarizer = summarizers[api_name]
    text = extract_text_from_segments(transcript)
    chunks = chunk_transcript(text, chunk_duration, words_per_second)

    summaries = []
    for chunk in chunks:
        if api_name == 'openai':
            # Ensure the correct model and prompt are passed
            summaries.append(summarizer(api_key, chunk, custom_prompt))
        else:
            summaries.append(summarizer(api_key, chunk))

    return "\n\n".join(summaries)


################## ####################


######### Token-size Chunking ######### FIXME - OpenAI only currently
# This is dirty and shameful and terrible. It should be replaced with a proper implementation.
# anyways lets get to it....
client = OpenAI(api_key=openai_api_key)


def get_chat_completion(messages, model='gpt-4-turbo'):
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=0,
    )
    return response.choices[0].message.content


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
        # FIXME MAKE NOT OPENAI SPECIFIC
        if len(openai_tokenize(chunk_delimiter.join(chunk_with_header))) > max_tokens:
            print(f"warning: chunk overflow")
            if (
                    add_ellipsis_for_overflow
                    # FIXME MAKE NOT OPENAI SPECIFIC
                    and len(openai_tokenize(chunk_delimiter.join(candidate + ["..."]))) <= max_tokens
            ):
                candidate.append("...")
                dropped_chunk_count += 1
            continue  # this case would break downstream assumptions
        # estimate token count with the current chunk added
        # FIXME MAKE NOT OPENAI SPECIFIC
        extended_candidate_token_count = len(openai_tokenize(chunk_delimiter.join(candidate + [chunk])))
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


def rolling_summarize(text: str,
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
    # FIXME MAKE NOT OPENAI SPECIFIC
    document_length = len(openai_tokenize(text))
    chunk_size = max(minimum_chunk_size, document_length // num_chunks)
    text_chunks = chunk_on_delimiter(text, chunk_size, chunk_delimiter)
    if verbose:
        print(f"Splitting the text into {len(text_chunks)} chunks to be summarized.")
        # FIXME MAKE NOT OPENAI SPECIFIC
        print(f"Chunk lengths are {[len(openai_tokenize(x)) for x in text_chunks]}")

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
    global final_summary
    final_summary = '\n\n'.join(accumulated_summaries)

    return final_summary


#######################################


######### Words-per-second Chunking #########
# FIXME - WHole section needs to be re-written
def chunk_transcript(transcript: str, chunk_duration: int, words_per_second) -> List[str]:
    words = transcript.split()
    words_per_chunk = chunk_duration * words_per_second
    chunks = [' '.join(words[i:i + words_per_chunk]) for i in range(0, len(words), words_per_chunk)]
    return chunks


def summarize_chunks(api_name: str, api_key: str, transcript: List[dict], chunk_duration: int,
                     words_per_second: int) -> str:
    if api_name not in summarizers:  # See 'summarizers' dict in the main script
        return f"Unsupported API: {api_name}"

    if not transcript:
        logging.error("Empty or None transcript provided to summarize_chunks")
        return "Error: Empty or None transcript provided"

    text = extract_text_from_segments(transcript)
    chunks = chunk_transcript(text, chunk_duration, words_per_second)

    custom_prompt = args.custom_prompt

    summaries = []
    for chunk in chunks:
        if api_name == 'openai':
            # Ensure the correct model and prompt are passed
            summaries.append(summarize_with_openai(api_key, chunk, custom_prompt))
        elif api_name == 'anthropic':
            summaries.append(summarize_with_cohere(api_key, chunk, anthropic_model, custom_prompt))
        elif api_name == 'cohere':
            summaries.append(summarize_with_claude(api_key, chunk, cohere_model, custom_prompt))
        elif api_name == 'groq':
            summaries.append(summarize_with_groq(api_key, chunk, groq_model, custom_prompt))
        elif api_name == 'llama':
            summaries.append(summarize_with_llama(llama_api_IP, chunk, api_key, custom_prompt))
        elif api_name == 'kobold':
            summaries.append(summarize_with_kobold(kobold_api_IP, chunk, api_key, custom_prompt))
        elif api_name == 'ooba':
            summaries.append(summarize_with_oobabooga(ooba_api_IP, chunk, api_key, custom_prompt))
        elif api_name == 'tabbyapi':
            summaries.append(summarize_with_vllm(api_key, tabby_api_IP, chunk, llm_model, custom_prompt))
        elif api_name == 'local-llm':
            summaries.append(summarize_with_local_llm(chunk, custom_prompt))
        else:
            return f"Unsupported API: {api_name}"

    return "\n\n".join(summaries)

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
