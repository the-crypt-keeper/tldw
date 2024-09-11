# Chunk_Lib.py
#########################################
# Chunking Library
# This library is used to perform chunking of input files.
# Currently, uses naive approaches. Nothing fancy.
#
####
# Import necessary libraries
import hashlib
import logging
import re
from typing import Any, Dict, List, Optional, Tuple
#
# Import 3rd party
from openai import OpenAI
from tqdm import tqdm
from langdetect import detect
from transformers import GPT2Tokenizer
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import textstat
#
# Import Local
from App_Function_Libraries.Tokenization_Methods_Lib import openai_tokenize
from App_Function_Libraries.Utils.Utils import load_comprehensive_config


#
#######################################################################################################################
# Config Settings
#
#
# FIXME - Make sure it only downloads if it already exists, and does a check first.
# Ensure NLTK data is downloaded
def ntlk_prep():
    nltk.download('punkt')
#
# Load GPT2 tokenizer
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
#
# Load configuration
config = load_comprehensive_config()
#chunk_options = config.get('Chunking', {})
chunk_options = config.get('Chunking', {}) if isinstance(config, dict) else {}
#
openai_api_key = config.get('API', 'openai_api_key')
#
# End of settings
#######################################################################################################################
#
# Functions:

def detect_language(text: str) -> str:
    try:
        return detect(text)
    except:
        # Default to English if detection fails
        return 'en'


def load_document(file_path):
    with open(file_path, 'r') as file:
        text = file.read()
    return re.sub('\\s+', ' ', text).strip()


def improved_chunking_process(text: str, custom_chunk_options: Dict[str, Any] = None) -> List[Dict[str, Any]]:
    options = chunk_options.copy()
    if custom_chunk_options:
        options.update(custom_chunk_options)

    chunk_method = options.get('method', 'words')
    base_size = options.get('base_size', 1000)
    min_size = options.get('min_size', 100)
    max_size = options.get('max_size', 2000)
    overlap = options.get('overlap', 0)
    language = options.get('language', None)
    adaptive = options.get('adaptive', False)
    multi_level = options.get('multi_level', False)

    if language is None:
        language = detect_language(text)

    if adaptive:
        max_chunk_size = adaptive_chunk_size(text, base_size, min_size, max_size)
    else:
        max_chunk_size = base_size

    if multi_level:
        chunks = multi_level_chunking(text, chunk_method, max_chunk_size, overlap, language)
    else:
        chunks = chunk_text(text, chunk_method, max_chunk_size, overlap, language)

    chunks_with_metadata = []
    for i, chunk in enumerate(chunks):
        metadata = get_chunk_metadata(
            chunk,
            text,
            chunk_type=chunk_method,
            language=language
        )
        metadata['chunk_index'] = i
        metadata['total_chunks'] = len(chunks)

        chunks_with_metadata.append({
            'text': chunk,
            'metadata': metadata
        })

    return chunks_with_metadata


def multi_level_chunking(text: str, method: str, max_size: int, overlap: int, language: str) -> List[str]:
    # First level: chunk by paragraphs
    paragraphs = chunk_text_by_paragraphs(text, max_size * 2, overlap)

    # Second level: chunk each paragraph further
    chunks = []
    for para in paragraphs:
        if method == 'words':
            chunks.extend(chunk_text_by_words(para, max_size, overlap, language))
        elif method == 'sentences':
            chunks.extend(chunk_text_by_sentences(para, max_size, overlap, language))
        else:
            chunks.append(para)

    return chunks


def chunk_text(text: str, method: str, max_size: int, overlap: int, language: str) -> List[str]:
    if method == 'words':
        return chunk_text_by_words(text, max_size, overlap, language)
    elif method == 'sentences':
        return chunk_text_by_sentences(text, max_size, overlap, language)
    elif method == 'paragraphs':
        return chunk_text_by_paragraphs(text, max_size, overlap)
    elif method == 'tokens':
        return chunk_text_by_tokens(text, max_size, overlap)
    elif method == 'semantic':
        return semantic_chunking(text, max_size)
    else:
        return [text]

def determine_chunk_position(relative_position: float) -> str:
    if relative_position < 0.33:
        return "This chunk is from the beginning of the document"
    elif relative_position < 0.66:
        return "This chunk is from the middle of the document"
    else:
        return "This chunk is from the end of the document"


def chunk_text_by_words(text: str, max_words: int = 300, overlap: int = 0, language: str = None) -> List[str]:
    if language is None:
        language = detect_language(text)

    if language.startswith('zh'):  # Chinese
        import jieba
        words = list(jieba.cut(text))
    elif language == 'ja':  # Japanese
        import fugashi
        tagger = fugashi.Tagger()
        words = [word.surface for word in tagger(text)]
    else:  # Default to simple splitting for other languages
        words = text.split()

    chunks = []
    for i in range(0, len(words), max_words - overlap):
        chunk = ' '.join(words[i:i + max_words])
        chunks.append(chunk)
    return post_process_chunks(chunks)


def chunk_text_by_sentences(text: str, max_sentences: int = 10, overlap: int = 0, language: str = None) -> List[str]:
    if language is None:
        language = detect_language(text)

    nltk.download('punkt', quiet=True)

    if language.startswith('zh'):  # Chinese
        import jieba
        sentences = list(jieba.cut(text, cut_all=False))
    elif language == 'ja':  # Japanese
        import fugashi
        tagger = fugashi.Tagger()
        sentences = [word.surface for word in tagger(text) if word.feature.pos1 in ['記号', '補助記号'] and word.surface.strip()]
    else:  # Default to NLTK for other languages
        sentences = sent_tokenize(text, language=language)

    chunks = []
    for i in range(0, len(sentences), max_sentences - overlap):
        chunk = ' '.join(sentences[i:i + max_sentences])
        chunks.append(chunk)
    return post_process_chunks(chunks)


def chunk_text_by_paragraphs(text: str, max_paragraphs: int = 5, overlap: int = 0) -> List[str]:
    paragraphs = re.split(r'\n\s*\n', text)
    chunks = []
    for i in range(0, len(paragraphs), max_paragraphs - overlap):
        chunk = '\n\n'.join(paragraphs[i:i + max_paragraphs])
        chunks.append(chunk)
    return post_process_chunks(chunks)


def chunk_text_by_tokens(text: str, max_tokens: int = 1000, overlap: int = 0) -> List[str]:
    # This is a simplified token-based chunking. For more accurate tokenization,
    # consider using a proper tokenizer like GPT-2 TokenizerFast
    words = text.split()
    chunks = []
    current_chunk = []
    current_token_count = 0

    for word in words:
        word_token_count = len(word) // 4 + 1  # Rough estimate of token count
        if current_token_count + word_token_count > max_tokens and current_chunk:
            chunks.append(' '.join(current_chunk))
            current_chunk = current_chunk[-overlap:] if overlap > 0 else []
            current_token_count = sum(len(w) // 4 + 1 for w in current_chunk)

        current_chunk.append(word)
        current_token_count += word_token_count

    if current_chunk:
        chunks.append(' '.join(current_chunk))

    return post_process_chunks(chunks)


def post_process_chunks(chunks: List[str]) -> List[str]:
    return [chunk.strip() for chunk in chunks if chunk.strip()]


def get_chunk_metadata(chunk: str, full_text: str, chunk_type: str = "generic",
                       chapter_number: Optional[int] = None,
                       chapter_pattern: Optional[str] = None,
                       language: str = None) -> Dict[str, Any]:
    try:
        start_index = full_text.index(chunk)
        end_index = start_index + len(chunk)
        # Calculate a hash for the chunk
        chunk_hash = hashlib.md5(chunk.encode()).hexdigest()

        metadata = {
            'start_index': start_index,
            'end_index': end_index,
            'word_count': len(chunk.split()),
            'char_count': len(chunk),
            'chunk_type': chunk_type,
            'language': language,
            'chunk_hash': chunk_hash,
            'relative_position': start_index / len(full_text),
            'readability_score': textstat.flesch_reading_ease(chunk)
        }

        if chunk_type == "chapter":
            metadata['chapter_number'] = chapter_number
            metadata['chapter_pattern'] = chapter_pattern

        # Add readability score (you might need to install 'textstat')
        metadata['readability_score'] = textstat.flesch_reading_ease(chunk)

        return metadata
    except ValueError as e:
        logging.error(f"Chunk not found in full_text: {chunk[:50]}... Full text length: {len(full_text)}")
        raise


def process_document_with_metadata(text: str, chunk_options: Dict[str, Any],
                                   document_metadata: Dict[str, Any]) -> Dict[str, Any]:
    chunks = improved_chunking_process(text, chunk_options)

    return {
        'document_metadata': document_metadata,
        'chunks': chunks
    }


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




# ????FIXME
def recursive_summarize_chunks(chunks, summarize_func, custom_prompt, temp=None, system_prompt=None):
    summarized_chunks = []
    current_summary = ""

    logging.debug(f"recursive_summarize_chunks: Summarizing {len(chunks)} chunks recursively...")
    logging.debug(f"recursive_summarize_chunks:  temperature is @ {temp}")
    for i, chunk in enumerate(chunks):
        if i == 0:
            current_summary = summarize_func(chunk, custom_prompt, temp, system_prompt)
        else:
            combined_text = current_summary + "\n\n" + chunk
            current_summary = summarize_func(combined_text, custom_prompt, temp, system_prompt)

        summarized_chunks.append(current_summary)

    return summarized_chunks


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
def count_units(text, unit='words'):
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


def semantic_chunk_long_file(file_path, max_chunk_size=1000, overlap=100, unit='words'):
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            content = file.read()

        chunks = semantic_chunking(content, max_chunk_size, unit)
        return chunks
    except Exception as e:
        logging.error(f"Error chunking text file: {str(e)}")
        return None

#
#
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
) -> Tuple[List[str], List[List[int]], int]:
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

    # set system message - FIXME
    system_message_content = "Rewrite this text in summarized form."
    if additional_instructions is not None:
        system_message_content += f"\n\n{additional_instructions}"

    accumulated_summaries = []
    for i, chunk in enumerate(tqdm(text_chunks)):
        if summarize_recursively and accumulated_summaries:
            # Combine previous summary with current chunk for recursive summarization
            combined_text = accumulated_summaries[-1] + "\n\n" + chunk
            user_message_content = f"Previous summary and new content to summarize:\n\n{combined_text}"
        else:
            user_message_content = chunk

        messages = [
            {"role": "system", "content": system_message_content},
            {"role": "user", "content": user_message_content}
        ]

        response = get_chat_completion(messages, model=model)
        accumulated_summaries.append(response)

    final_summary = '\n\n'.join(accumulated_summaries)
    return final_summary

#
#
#######################################################################################################################
#
# Ebook Chapter Chunking


def chunk_ebook_by_chapters(text: str, chunk_options: Dict[str, Any]) -> List[Dict[str, Any]]:
    max_chunk_size = chunk_options.get('max_size', 300)
    overlap = chunk_options.get('overlap', 0)
    custom_pattern = chunk_options.get('custom_chapter_pattern', None)

    # List of chapter heading patterns to try, in order
    chapter_patterns = [
        custom_pattern,
        r'^#{1,2}\s+',  # Markdown style: '# ' or '## '
        r'^Chapter\s+\d+',  # 'Chapter ' followed by numbers
        r'^\d+\.\s+',  # Numbered chapters: '1. ', '2. ', etc.
        r'^[A-Z\s]+$'  # All caps headings
    ]

    chapter_positions = []
    used_pattern = None

    for pattern in chapter_patterns:
        if pattern is None:
            continue
        chapter_regex = re.compile(pattern, re.MULTILINE | re.IGNORECASE)
        chapter_positions = [match.start() for match in chapter_regex.finditer(text)]
        if chapter_positions:
            used_pattern = pattern
            break

    # If no chapters found, return the entire content as one chunk
    if not chapter_positions:
        return [{'text': text, 'metadata': get_chunk_metadata(text, text, chunk_type="whole_document")}]

    # Split content into chapters
    chunks = []
    for i in range(len(chapter_positions)):
        start = chapter_positions[i]
        end = chapter_positions[i + 1] if i + 1 < len(chapter_positions) else None
        chapter = text[start:end]

        # Apply overlap if specified
        if overlap > 0 and i > 0:
            overlap_start = max(0, start - overlap)
            chapter = text[overlap_start:end]

        chunks.append(chapter)

    # Post-process chunks
    processed_chunks = post_process_chunks(chunks)

    # Add metadata to chunks
    return [{'text': chunk, 'metadata': get_chunk_metadata(chunk, text, chunk_type="chapter", chapter_number=i + 1,
                                                           chapter_pattern=used_pattern)}
            for i, chunk in enumerate(processed_chunks)]


# # Example usage
# if __name__ == "__main__":
#     sample_ebook_content = """
# # Chapter 1: Introduction
#
# This is the introduction.
#
# ## Section 1.1
#
# Some content here.
#
# # Chapter 2: Main Content
#
# This is the main content.
#
# ## Section 2.1
#
# More content here.
#
# CHAPTER THREE
#
# This is the third chapter.
#
# 4. Fourth Chapter
#
# This is the fourth chapter.
# """
#
#     chunk_options = {
#         'method': 'chapters',
#         'max_size': 500,
#         'overlap': 50,
#         'custom_chapter_pattern': r'^CHAPTER\s+[A-Z]+'  # Custom pattern for 'CHAPTER THREE' style
#     }
#
#     chunked_chapters = improved_chunking_process(sample_ebook_content, chunk_options)
#
#     for i, chunk in enumerate(chunked_chapters, 1):
#         print(f"Chunk {i}:")
#         print(chunk['text'])
#         print(f"Metadata: {chunk['metadata']}\n")

#
# End of ebook chapter chunking
#######################################################################################################################

#######################################################################################################################
#
# Functions for adapative chunking:

# FIXME - punkt
def adaptive_chunk_size(text: str, base_size: int = 1000, min_size: int = 500, max_size: int = 2000) -> int:
    # Ensure NLTK data is downloaded
    nltk.download('punkt', quiet=True)

    # Tokenize the text into sentences
    sentences = sent_tokenize(text)

    # Calculate average sentence length
    avg_sentence_length = sum(len(s.split()) for s in sentences) / len(sentences)

    # Adjust chunk size based on average sentence length
    if avg_sentence_length < 10:
        size_factor = 1.2  # Increase chunk size for short sentences
    elif avg_sentence_length > 20:
        size_factor = 0.8  # Decrease chunk size for long sentences
    else:
        size_factor = 1.0

    # Calculate adaptive chunk size
    adaptive_size = int(base_size * size_factor)

    # Ensure chunk size is within bounds
    return max(min_size, min(adaptive_size, max_size))

# Non-Punkt version
# def adaptive_chunk_size(text: str, base_size: int, min_size: int = 100, max_size: int = 2000) -> int:
#     # Adaptive logic: adjust chunk size based on text complexity
#     words = text.split()
#     if not words:
#         return base_size  # Return base_size if text is empty
#
#     avg_word_length = sum(len(word) for word in words) / len(words)
#
#     if avg_word_length > 6:  # Threshold for "complex" text
#         adjusted_size = int(base_size * 0.8)  # Reduce chunk size for complex text
#     elif avg_word_length < 4:  # Threshold for "simple" text
#         adjusted_size = int(base_size * 1.2)  # Increase chunk size for simple text
#     else:
#         adjusted_size = base_size
#
#     # Ensure the chunk size is within the specified range
#     return max(min_size, min(adjusted_size, max_size))


def adaptive_chunking(text: str, base_size: int = 1000, min_size: int = 500, max_size: int = 2000) -> List[str]:
    chunk_size = adaptive_chunk_size(text, base_size, min_size, max_size)
    words = text.split()
    chunks = []
    current_chunk = []
    current_length = 0

    for word in words:
        if current_length + len(word) > chunk_size and current_chunk:
            chunks.append(' '.join(current_chunk))
            current_chunk = []
            current_length = 0
        current_chunk.append(word)
        current_length += len(word) + 1  # +1 for space

    if current_chunk:
        chunks.append(' '.join(current_chunk))

    return chunks

# FIXME - usage example
# chunk_options = {
#     'method': 'words',  # or any other method
#     'base_size': 1000,
#     'min_size': 100,
#     'max_size': 2000,
#     'adaptive': True,
#     'language': 'en'
# }
#chunks = improved_chunking_process(your_text, chunk_options)


# Example of chunking a document with metadata
# document_metadata = {
#     'title': 'Example Document',
#     'author': 'John Doe',
#     'creation_date': '2023-06-14',
#     'source': 'https://example.com/document',
#     'document_type': 'article'
# }
#
# chunk_options = {
#     'method': 'sentences',
#     'base_size': 1000,
#     'adaptive': True,
#     'language': 'en'
# }
#
# processed_document = process_document_with_metadata(your_text, chunk_options, document_metadata)


#
# End of Chunking Library
#######################################################################################################################