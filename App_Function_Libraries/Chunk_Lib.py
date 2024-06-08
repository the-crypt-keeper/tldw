from transformers import GPT2Tokenizer
import nltk
import re








# Import Local
import summarize
from Article_Summarization_Lib import *
from Article_Extractor_Lib import *
from Audio_Transcription_Lib import *
#from Chunk_Lib import *
from Diarization_Lib import *
from Video_DL_Ingestion_Lib import *
from Local_File_Processing_Lib import *
from Local_LLM_Inference_Engine_Lib import *
from Local_Summarization_Lib import *
from Old_Chunking_Lib import *
from SQLite_DB import *
from App_Function_Libraries.Local_Summarization_Lib import *
from App_Function_Libraries.Summarization_General_Lib import *
from App_Function_Libraries.Summarization_General_Lib import *
from System_Checks_Lib import *
from Tokenization_Methods_Lib import *
from Video_DL_Ingestion_Lib import *
from Web_UI_Lib import *


# FIXME - Make sure it only downloads if it already exists, and does a check first.
# Ensure NLTK data is downloaded
def ntlk_prep():
    nltk.download('punkt')

# Load GPT2 tokenizer
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")


def load_document(file_path):
    with open(file_path, 'r') as file:
        text = file.read()
    return re.sub('\s+', ' ', text).strip()


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


def rolling_summarize_function(text: str,
                                detail: float = 0,
                                api_name: str = None,
                                api_key: str = None,
                                model: str = None,
                                custom_prompt: str = None,
                                chunk_by_words: bool = False,
                                max_words: int = 300,
                                chunk_by_sentences: bool = False,
                                max_sentences: int = 10,
                                chunk_by_paragraphs: bool = False,
                                max_paragraphs: int = 5,
                                chunk_by_tokens: bool = False,
                                max_tokens: int = 1000,
                                summarize_recursively=False,
                                verbose=False):
    """
    Summarizes a given text by splitting it into chunks, each of which is summarized individually.
    Allows selecting the method for chunking (words, sentences, paragraphs, tokens).

    Parameters:
        - text (str): The text to be summarized.
        - detail (float, optional): A value between 0 and 1 indicating the desired level of detail in the summary.
        - api_name (str, optional): Name of the API to use for summarization.
        - api_key (str, optional): API key for the specified API.
        - model (str, optional): Model identifier for the summarization engine.
        - custom_prompt (str, optional): Custom prompt for the summarization.
        - chunk_by_words (bool, optional): If True, chunks the text by words.
        - max_words (int, optional): Maximum number of words per chunk.
        - chunk_by_sentences (bool, optional): If True, chunks the text by sentences.
        - max_sentences (int, optional): Maximum number of sentences per chunk.
        - chunk_by_paragraphs (bool, optional): If True, chunks the text by paragraphs.
        - max_paragraphs (int, optional): Maximum number of paragraphs per chunk.
        - chunk_by_tokens (bool, optional): If True, chunks the text by tokens.
        - max_tokens (int, optional): Maximum number of tokens per chunk.
        - summarize_recursively (bool, optional): If True, summaries are generated recursively.
        - verbose (bool, optional): If verbose, prints additional output.

    Returns:
        - str: The final compiled summary of the text.
    """

    def extract_text_from_segments(segments):
        text = ' '.join([segment['Text'] for segment in segments if 'Text' in segment])
        return text
    # Validate input
    if not text:
        raise ValueError("Input text cannot be empty.")
    if any([max_words <= 0, max_sentences <= 0, max_paragraphs <= 0, max_tokens <= 0]):
        raise ValueError("All maximum chunk size parameters must be positive integers.")
    global segments

    if isinstance(text, dict) and 'transcription' in text:
        text = extract_text_from_segments(text['transcription'])
    elif isinstance(text, list):
        text = extract_text_from_segments(text)

    # Select the chunking function based on the method specified
    if chunk_by_words:
        chunks = chunk_text_by_words(text, max_words)
    elif chunk_by_sentences:
        chunks = chunk_text_by_sentences(text, max_sentences)
    elif chunk_by_paragraphs:
        chunks = chunk_text_by_paragraphs(text, max_paragraphs)
    elif chunk_by_tokens:
        chunks = chunk_text_by_tokens(text, max_tokens)
    else:
        chunks = [text]

    # Process each chunk for summarization
    accumulated_summaries = []
    for chunk in chunks:
        if summarize_recursively and accumulated_summaries:
            # Creating a structured prompt for recursive summarization
            previous_summaries = '\n\n'.join(accumulated_summaries)
            user_message_content = f"Previous summaries:\n\n{previous_summaries}\n\nText to summarize next:\n\n{chunk}"
        else:
            # Directly passing the chunk for summarization without recursive context
            user_message_content = chunk

        # Extracting the completion from the response
        try:
            if api_name.lower() == 'openai':
                summary = summarize_with_openai(api_key, user_message_content, custom_prompt)
            elif api_name.lower() == "cohere":
                cohere_api_key = api_key if api_key else config.get('API', 'cohere_api_key', fallback=None)
                if not cohere_api_key:
                    logging.error("MAIN: Cohere API key not found.")
                    return None
                logging.debug(f"MAIN: Trying to summarize with cohere")
                summary = summarize_with_cohere(cohere_api_key, user_message_content,
                                                config.get('API', 'cohere_model', fallback='command-r-plus'),
                                                custom_prompt)
            elif api_name.lower() == "groq":
                groq_api_key = api_key if api_key else config.get('API', 'groq_api_key', fallback=None)
                if not groq_api_key:
                    logging.error("MAIN: Groq API key not found.")
                    return None
                logging.debug(f"MAIN: Trying to summarize with groq")
                summary = summarize_with_groq(groq_api_key, user_message_content, custom_prompt)
            elif api_name.lower() == "openrouter":
                openrouter_api_key = api_key if api_key else config.get('API', 'openrouter_api_key', fallback=None)
                if not openrouter_api_key:
                    logging.error("MAIN: OpenRouter API key not found.")
                    return None
                logging.debug(f"MAIN: Trying to summarize with OpenRouter")
                summary = summarize_with_openrouter(openrouter_api_key, user_message_content, custom_prompt)
            elif api_name.lower() == "llama":
                logging.debug(f"MAIN: Trying to summarize with Llama.cpp")
                llama_token = api_key if api_key else config.get('Local-API', 'llama_token', fallback=None)
                summary = summarize_with_llama(user_message_content, custom_prompt)
            elif api_name.lower() == "kobold":
                kobold_api_key = api_key if api_key else config.get('Local-API', 'kobold_api_key', fallback=None)
                if not kobold_api_key:
                    logging.error("MAIN: Kobold API key not found.")
                    return None
                logging.debug(f"MAIN: Trying to summarize with Kobold.cpp")
                summary = summarize_with_kobold(kobold_api_key, user_message_content, custom_prompt)
            elif api_name.lower() == "ooba":
                ooba_token = api_key if api_key else config.get('Local-API', 'ooba_api_key', fallback=None)
                ooba_ip = config.get('API', 'ooba_ip', fallback=None)
                summary = summarize_with_oobabooga(ooba_ip, user_message_content, ooba_token, custom_prompt)
            elif api_name.lower() == "tabbyapi":
                tabbyapi_key = api_key if api_key else config.get('Local-API', 'tabbyapi_token', fallback=None)
                tabby_model = config.get('Local-API', 'tabby_model', fallback=None)
                summary = summarize_with_tabbyapi(tabby_api_key, config.get('Local-API', 'tabby_api_IP',
                                                                            fallback='http://127.0.0.1:5000/api/v1/generate'),
                                                  user_message_content, tabby_model, custom_prompt)
            elif api_name.lower() == "vllm":
                logging.debug(f"MAIN: Trying to summarize with VLLM")
                vllm_api_key = api_key if api_key else config.get('Local-API', 'vllm_api_key', fallback=None)
                summary = summarize_with_vllm(
                    config.get('Local-API', 'vllm_api_IP', fallback='http://127.0.0.1:500/api/v1/chat/completions'),
                    vllm_api_key, config.get('API', 'vllm_model', fallback=''), user_message_content, custom_prompt)
            elif api_name.lower() == "local-llm":
                logging.debug(f"MAIN: Trying to summarize with Local LLM")
                summary = summarize_with_local_llm(user_message_content, custom_prompt)
            elif api_name.lower() == "huggingface":
                logging.debug(f"MAIN: Trying to summarize with huggingface")
                huggingface_api_key = api_key if api_key else config.get('API', 'huggingface_api_key', fallback=None)
                summary = summarize_with_huggingface(huggingface_api_key, user_message_content, custom_prompt)
            # Add additional API handlers here...
            else:
                logging.warning(f"Unsupported API: {api_name}")
                summary = None
        except requests.exceptions.ConnectionError:
            logging.error("Connection error while summarizing")
            summary = None
        except Exception as e:
            logging.error(f"Error summarizing with {api_name}: {str(e)}")
            summary = None

        if summary:
            logging.info(f"Summary generated using {api_name} API")
            accumulated_summaries.append(summary)
        else:
            logging.warning(f"Failed to generate summary using {api_name} API")

    # Compile final summary from partial summaries
    final_summary = '\n\n'.join(accumulated_summaries)
    return final_summary



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
