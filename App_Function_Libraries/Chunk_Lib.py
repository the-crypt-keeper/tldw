# Chunk_Lib.py
#########################################
# Chunking Library
# This library is used to perform chunking of input files.
# Currently, uses naive approaches. Nothing fancy.
#
####
# Import necessary libraries
import logging
import re

from typing import List, Optional, Tuple

from openai import OpenAI
from tqdm import tqdm
#
# Import 3rd party
from transformers import GPT2Tokenizer
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
#
# Import Local
from App_Function_Libraries.Summarization_General_Lib import openai_api_key
from App_Function_Libraries.Tokenization_Methods_Lib import openai_tokenize
#
#######################################################################################################################
# Function Definitions
#

# FIXME - Make sure it only downloads if it already exists, and does a check first.
# Ensure NLTK data is downloaded
def ntlk_prep():
    nltk.download('punkt')

# Load GPT2 tokenizer
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")


def load_document(file_path):
    with open(file_path, 'r') as file:
        text = file.read()
    return re.sub('\\s+', ' ', text).strip()


# Chunk based on maximum number of words, using ' ' (space) as a delimiter
def chunk_text_by_words(text, max_words=300):
    words = text.split()
    chunks = [' '.join(words[i:i + max_words]) for i in range(0, len(words), max_words)]
    return chunks


# Chunk based on sentences, not exceeding a max amount, using nltk
def chunk_text_by_sentences(text, max_sentences=10):
    sentences = nltk.tokenize.sent_tokenize(text)
    chunks = [' '.join(sentences[i:i + max_sentences]) for i in range(0, len(sentences), max_sentences)]
    return chunks


# Chunk text by paragraph, marking paragraphs by (delimiter) '\n\n'
def chunk_text_by_paragraphs(text, max_paragraphs=5):
    paragraphs = text.split('\n\n')
    chunks = ['\n\n'.join(paragraphs[i:i + max_paragraphs]) for i in range(0, len(paragraphs), max_paragraphs)]
    return chunks


# Naive chunking based on token count
def chunk_text_by_tokens(text, max_tokens=1000):
    tokens = tokenizer.encode(text)
    chunks = [tokenizer.decode(tokens[i:i + max_tokens]) for i in range(0, len(tokens), max_tokens)]
    return chunks


# Hybrid approach, chunk each sentence while ensuring total token size does not exceed a maximum number
def chunk_text_hybrid(text, max_tokens=1000):
    sentences = nltk.tokenize.sent_tokenize(text)
    chunks = []
    current_chunk = []
    current_length = 0

    for sentence in sentences:
        tokens = tokenizer.encode(sentence)
        if current_length + len(tokens) <= max_tokens:
            current_chunk.append(sentence)
            current_length += len(tokens)
        else:
            chunks.append(' '.join(current_chunk))
            current_chunk = [sentence]
            current_length = len(tokens)

    if current_chunk:
        chunks.append(' '.join(current_chunk))

    return chunks

# Thanks openai
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


# def rolling_summarize_function(text: str,
#                                 detail: float = 0,
#                                 api_name: str = None,
#                                 api_key: str = None,
#                                 model: str = None,
#                                 custom_prompt: str = None,
#                                 chunk_by_words: bool = False,
#                                 max_words: int = 300,
#                                 chunk_by_sentences: bool = False,
#                                 max_sentences: int = 10,
#                                 chunk_by_paragraphs: bool = False,
#                                 max_paragraphs: int = 5,
#                                 chunk_by_tokens: bool = False,
#                                 max_tokens: int = 1000,
#                                 summarize_recursively=False,
#                                 verbose=False):
#     """
#     Summarizes a given text by splitting it into chunks, each of which is summarized individually.
#     Allows selecting the method for chunking (words, sentences, paragraphs, tokens).
#
#     Parameters:
#         - text (str): The text to be summarized.
#         - detail (float, optional): A value between 0 and 1 indicating the desired level of detail in the summary.
#         - api_name (str, optional): Name of the API to use for summarization.
#         - api_key (str, optional): API key for the specified API.
#         - model (str, optional): Model identifier for the summarization engine.
#         - custom_prompt (str, optional): Custom prompt for the summarization.
#         - chunk_by_words (bool, optional): If True, chunks the text by words.
#         - max_words (int, optional): Maximum number of words per chunk.
#         - chunk_by_sentences (bool, optional): If True, chunks the text by sentences.
#         - max_sentences (int, optional): Maximum number of sentences per chunk.
#         - chunk_by_paragraphs (bool, optional): If True, chunks the text by paragraphs.
#         - max_paragraphs (int, optional): Maximum number of paragraphs per chunk.
#         - chunk_by_tokens (bool, optional): If True, chunks the text by tokens.
#         - max_tokens (int, optional): Maximum number of tokens per chunk.
#         - summarize_recursively (bool, optional): If True, summaries are generated recursively.
#         - verbose (bool, optional): If verbose, prints additional output.
#
#     Returns:
#         - str: The final compiled summary of the text.
#     """
#
#     def extract_text_from_segments(segments):
#         text = ' '.join([segment['Text'] for segment in segments if 'Text' in segment])
#         return text
#     # Validate input
#     if not text:
#         raise ValueError("Input text cannot be empty.")
#     if any([max_words <= 0, max_sentences <= 0, max_paragraphs <= 0, max_tokens <= 0]):
#         raise ValueError("All maximum chunk size parameters must be positive integers.")
#     global segments
#
#     if isinstance(text, dict) and 'transcription' in text:
#         text = extract_text_from_segments(text['transcription'])
#     elif isinstance(text, list):
#         text = extract_text_from_segments(text)
#
#     # Select the chunking function based on the method specified
#     if chunk_by_words:
#         chunks = chunk_text_by_words(text, max_words)
#     elif chunk_by_sentences:
#         chunks = chunk_text_by_sentences(text, max_sentences)
#     elif chunk_by_paragraphs:
#         chunks = chunk_text_by_paragraphs(text, max_paragraphs)
#     elif chunk_by_tokens:
#         chunks = chunk_text_by_tokens(text, max_tokens)
#     else:
#         chunks = [text]
#
#     # Process each chunk for summarization
#     accumulated_summaries = []
#     for chunk in chunks:
#         if summarize_recursively and accumulated_summaries:
#             # Creating a structured prompt for recursive summarization
#             previous_summaries = '\n\n'.join(accumulated_summaries)
#             user_message_content = f"Previous summaries:\n\n{previous_summaries}\n\nText to summarize next:\n\n{chunk}"
#         else:
#             # Directly passing the chunk for summarization without recursive context
#             user_message_content = chunk
#
#         # Extracting the completion from the response
#         try:
#             if api_name.lower() == 'openai':
#                 # def summarize_with_openai(api_key, input_data, custom_prompt_arg)
#                 summary = summarize_with_openai(user_message_content, text, custom_prompt)
#
#             elif api_name.lower() == "anthropic":
#                 # def summarize_with_anthropic(api_key, input_data, model, custom_prompt_arg, max_retries=3, retry_delay=5):
#                 summary = summarize_with_anthropic(user_message_content, text, custom_prompt)
#             elif api_name.lower() == "cohere":
#                 # def summarize_with_cohere(api_key, input_data, model, custom_prompt_arg)
#                 summary = summarize_with_cohere(user_message_content, text, custom_prompt)
#
#             elif api_name.lower() == "groq":
#                 logging.debug(f"MAIN: Trying to summarize with groq")
#                 # def summarize_with_groq(api_key, input_data, model, custom_prompt_arg):
#                 summary = summarize_with_groq(user_message_content, text, custom_prompt)
#
#             elif api_name.lower() == "openrouter":
#                 logging.debug(f"MAIN: Trying to summarize with OpenRouter")
#                 # def summarize_with_openrouter(api_key, input_data, custom_prompt_arg):
#                 summary = summarize_with_openrouter(user_message_content, text, custom_prompt)
#
#             elif api_name.lower() == "deepseek":
#                 logging.debug(f"MAIN: Trying to summarize with DeepSeek")
#                 # def summarize_with_deepseek(api_key, input_data, custom_prompt_arg):
#                 summary = summarize_with_deepseek(api_key, user_message_content,custom_prompt)
#
#             elif api_name.lower() == "llama.cpp":
#                 logging.debug(f"MAIN: Trying to summarize with Llama.cpp")
#                 # def summarize_with_llama(api_url, file_path, token, custom_prompt)
#                 summary = summarize_with_llama(user_message_content, custom_prompt)
#
#             elif api_name.lower() == "kobold":
#                 logging.debug(f"MAIN: Trying to summarize with Kobold.cpp")
#                 # def summarize_with_kobold(input_data, kobold_api_token, custom_prompt_input, api_url):
#                 summary = summarize_with_kobold(user_message_content, api_key, custom_prompt)
#
#             elif api_name.lower() == "ooba":
#                 # def summarize_with_oobabooga(input_data, api_key, custom_prompt, api_url):
#                 summary = summarize_with_oobabooga(user_message_content, api_key, custom_prompt)
#
#             elif api_name.lower() == "tabbyapi":
#                 # def summarize_with_tabbyapi(input_data, tabby_model, custom_prompt_input, api_key=None, api_IP):
#                 summary = summarize_with_tabbyapi(user_message_content, custom_prompt)
#
#             elif api_name.lower() == "vllm":
#                 logging.debug(f"MAIN: Trying to summarize with VLLM")
#                 # def summarize_with_vllm(api_key, input_data, custom_prompt_input):
#                 summary = summarize_with_vllm(user_message_content, custom_prompt)
#
#             elif api_name.lower() == "local-llm":
#                 logging.debug(f"MAIN: Trying to summarize with Local LLM")
#                 summary = summarize_with_local_llm(user_message_content, custom_prompt)
#
#             elif api_name.lower() == "huggingface":
#                 logging.debug(f"MAIN: Trying to summarize with huggingface")
#                 # def summarize_with_huggingface(api_key, input_data, custom_prompt_arg):
#                 summarize_with_huggingface(api_key, user_message_content, custom_prompt)
#             # Add additional API handlers here...
#             else:
#                 logging.warning(f"Unsupported API: {api_name}")
#                 summary = None
#         except requests.exceptions.ConnectionError:
#             logging.error("Connection error while summarizing")
#             summary = None
#         except Exception as e:
#             logging.error(f"Error summarizing with {api_name}: {str(e)}")
#             summary = None
#
#         if summary:
#             logging.info(f"Summary generated using {api_name} API")
#             accumulated_summaries.append(summary)
#         else:
#             logging.warning(f"Failed to generate summary using {api_name} API")
#
#     # Compile final summary from partial summaries
#     final_summary = '\n\n'.join(accumulated_summaries)
#     return final_summary



# Sample text for testing
sample_text = """
Natural language processing (NLP) is a subfield of linguistics, computer science, and artificial intelligence 
concerned with the interactions between computers and human language, in particular how to program computers 
to process and analyze large amounts of natural language data. The result is a computer capable of "understanding" 
the contents of documents, including the contextual nuances of the language within them. The technology can then 
accurately extract information and insights contained in the documents as well as categorize and organize the documents themselves.

Challenges in natural language processing frequently involve speech recognition, natural language understanding, 
and natural language generation.

Natural language processing has its roots in the 1950s. Already in 1950, Alan Turing published an article titled 
"Computing Machinery and Intelligence" which proposed what is now called the Turing test as a criterion of intelligence.
"""

# Example usage of different chunking methods
# print("Chunking by words:")
# print(chunk_text_by_words(sample_text, max_words=50))
#
# print("\nChunking by sentences:")
# print(chunk_text_by_sentences(sample_text, max_sentences=2))
#
# print("\nChunking by paragraphs:")
# print(chunk_text_by_paragraphs(sample_text, max_paragraphs=1))
#
# print("\nChunking by tokens:")
# print(chunk_text_by_tokens(sample_text, max_tokens=50))
#
# print("\nHybrid chunking:")
# print(chunk_text_hybrid(sample_text, max_tokens=50))



#######################################################################################################################
#
# Experimental Semantic Chunking
#

# Chunk text into segments based on semantic similarity
def count_units(text, unit='tokens'):
    if unit == 'words':
        return len(text.split())
    elif unit == 'tokens':
        return len(word_tokenize(text))
    elif unit == 'characters':
        return len(text)
    else:
        raise ValueError("Invalid unit. Choose 'words', 'tokens', or 'characters'.")


def semantic_chunking(text, max_chunk_size=2000, unit='words'):
    nltk.download('punkt', quiet=True)
    sentences = sent_tokenize(text)
    vectorizer = TfidfVectorizer()
    sentence_vectors = vectorizer.fit_transform(sentences)

    chunks = []
    current_chunk = []
    current_size = 0

    for i, sentence in enumerate(sentences):
        sentence_size = count_units(sentence, unit)
        if current_size + sentence_size > max_chunk_size and current_chunk:
            chunks.append(' '.join(current_chunk))
            overlap_size = count_units(' '.join(current_chunk[-3:]), unit)  # Use last 3 sentences for overlap
            current_chunk = current_chunk[-3:]  # Keep last 3 sentences for overlap
            current_size = overlap_size

        current_chunk.append(sentence)
        current_size += sentence_size

        if i + 1 < len(sentences):
            current_vector = sentence_vectors[i]
            next_vector = sentence_vectors[i + 1]
            similarity = cosine_similarity(current_vector, next_vector)[0][0]
            if similarity < 0.5 and current_size >= max_chunk_size // 2:
                chunks.append(' '.join(current_chunk))
                overlap_size = count_units(' '.join(current_chunk[-3:]), unit)
                current_chunk = current_chunk[-3:]
                current_size = overlap_size

    if current_chunk:
        chunks.append(' '.join(current_chunk))

    return chunks


def semantic_chunk_long_file(file_path, max_chunk_size=1000, overlap=100):
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            content = file.read()

        chunks = semantic_chunking(content, max_chunk_size, overlap)
        return chunks
    except Exception as e:
        logging.error(f"Error chunking text file: {str(e)}")
        return None
#######################################################################################################################






#######################################################################################################################
#
# OpenAI Rolling Summarization
#

client = OpenAI(api_key=openai_api_key)
def get_chat_completion(messages, model='gpt-4-turbo'):
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=0,
    )
    return response.choices[0].message.content


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

    Parameters:
        - text (str): The text to be summarized.
        - detail (float, optional): A value between 0 and 1
            indicating the desired level of detail in the summary. 0 leads to a higher level summary, and 1 results in a more
            detailed summary. Defaults to 0.
        - additional_instructions (Optional[str], optional): Additional instructions to provide to the
            model for customizing summaries. - minimum_chunk_size (Optional[int], optional): The minimum size for text
            chunks. Defaults to 500.
        - chunk_delimiter (str, optional): The delimiter used to split the text into chunks. Defaults to ".".
        - summarize_recursively (bool, optional): If True, summaries are generated recursively, using previous summaries for context.
        - verbose (bool, optional): If True, prints detailed information about the chunking process.
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