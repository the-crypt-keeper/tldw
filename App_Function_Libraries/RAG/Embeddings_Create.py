# Embeddings_Create.py
# Description: Functions for Creating and managing Embeddings in ChromaDB with LLama.cpp/OpenAI/Transformers
#
# Imports:
import logging
import time
from functools import wraps
from threading import Lock, Timer
from typing import List, Dict, Any
#
# 3rd-Party Imports:
import requests
from transformers import AutoTokenizer, AutoModel
import torch
#
# Local Imports:
from App_Function_Libraries.LLM_API_Calls import get_openai_embeddings
from App_Function_Libraries.Summarization_General_Lib import summarize
from App_Function_Libraries.Utils.Utils import load_comprehensive_config
from App_Function_Libraries.Chunk_Lib import chunk_options, improved_chunking_process, determine_chunk_position
#
#######################################################################################################################
#
# Functions:

# FIXME - Add all globals to summarize.py
loaded_config = load_comprehensive_config()
embedding_provider = loaded_config['Embeddings']['embedding_provider']
embedding_model = loaded_config['Embeddings']['embedding_model']
embedding_api_url = loaded_config['Embeddings']['embedding_api_url']
embedding_api_key = loaded_config['Embeddings']['embedding_api_key']

# Embedding Chunking Settings
chunk_size = loaded_config['Embeddings']['chunk_size']
overlap = loaded_config['Embeddings']['overlap']


# FIXME - Add logging


class HuggingFaceEmbedder:
    def __init__(self, model_name, timeout_seconds=120):  # Default timeout of 2 minutes
        self.model_name = model_name
        self.tokenizer = None
        self.model = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.timeout_seconds = timeout_seconds
        self.last_used_time = 0
        self.unload_timer = None

    def load_model(self):
        if self.model is None:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModel.from_pretrained(self.model_name)
            self.model.to(self.device)
        self.last_used_time = time.time()
        self.reset_timer()

    def unload_model(self):
        if self.model is not None:
            del self.model
            del self.tokenizer
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            self.model = None
            self.tokenizer = None
        if self.unload_timer:
            self.unload_timer.cancel()

    def reset_timer(self):
        if self.unload_timer:
            self.unload_timer.cancel()
        self.unload_timer = Timer(self.timeout_seconds, self.unload_model)
        self.unload_timer.start()

    def create_embeddings(self, texts):
        self.load_model()
        inputs = self.tokenizer(texts, return_tensors="pt", padding=True, truncation=True, max_length=512)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = self.model(**inputs)
        embeddings = outputs.last_hidden_state.mean(dim=1)
        return embeddings.cpu().numpy()

# Global variable to hold the embedder
huggingface_embedder = None


class RateLimiter:
    def __init__(self, max_calls, period):
        self.max_calls = max_calls
        self.period = period
        self.calls = []
        self.lock = Lock()

    def __call__(self, func):
        def wrapper(*args, **kwargs):
            with self.lock:
                now = time.time()
                self.calls = [call for call in self.calls if call > now - self.period]
                if len(self.calls) >= self.max_calls:
                    sleep_time = self.calls[0] - (now - self.period)
                    time.sleep(sleep_time)
                self.calls.append(time.time())
            return func(*args, **kwargs)
        return wrapper


def exponential_backoff(max_retries=5, base_delay=1):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            for attempt in range(max_retries):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    if attempt == max_retries - 1:
                        raise
                    delay = base_delay * (2 ** attempt)
                    logging.warning(f"Attempt {attempt + 1} failed. Retrying in {delay} seconds. Error: {str(e)}")
                    time.sleep(delay)
        return wrapper
    return decorator


# FIXME - refactor/setup to use config file & perform chunking
@exponential_backoff()
@RateLimiter(max_calls=50, period=60)  # Adjust these values based on API limits
def create_embeddings_batch(texts: List[str], provider: str, model: str, api_url: str, timeout_seconds: int = 300) -> \
List[List[float]]:
    global huggingface_embedder

    if provider.lower() == 'huggingface':
        if huggingface_embedder is None or huggingface_embedder.model_name != model:
            if huggingface_embedder is not None:
                huggingface_embedder.unload_model()
            huggingface_embedder = HuggingFaceEmbedder(model, timeout_seconds)

        embeddings = huggingface_embedder.create_embeddings(texts).tolist()
        return embeddings

    elif provider.lower() == 'openai':
        logging.debug(f"Creating embeddings for {len(texts)} texts using OpenAI API")
        return [create_openai_embedding(text, model) for text in texts]

    elif provider.lower() == 'local':
        response = requests.post(
            api_url,
            json={"texts": texts, "model": model},
            headers={"Authorization": f"Bearer {embedding_api_key}"}
        )
        if response.status_code == 200:
            return response.json()['embeddings']
        else:
            raise Exception(f"Error from local API: {response.text}")
    else:
        raise ValueError(f"Unsupported embedding provider: {provider}")

def create_embedding(text: str, provider: str, model: str, api_url: str) -> List[float]:
    return create_embeddings_batch([text], provider, model, api_url)[0]

# FIXME
def create_stella_embeddings(text: str) -> List[float]:
    if embedding_provider == 'local':
        # Load the model and tokenizer
        tokenizer = AutoTokenizer.from_pretrained("dunzhang/stella_en_400M_v5")
        model = AutoModel.from_pretrained("dunzhang/stella_en_400M_v5")

        # Tokenize and encode the text
        inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)

        # Generate embeddings
        with torch.no_grad():
            outputs = model(**inputs)

        # Use the mean of the last hidden state as the sentence embedding
        embeddings = outputs.last_hidden_state.mean(dim=1)

        return embeddings[0].tolist()  # Convert to list for consistency
    elif embedding_provider == 'openai':
        return get_openai_embeddings(text, embedding_model)
    else:
        raise ValueError(f"Unsupported embedding provider: {embedding_provider}")


def chunk_for_embedding(text: str, file_name: str, api_name, custom_chunk_options: Dict[str, Any] = None) -> List[Dict[str, Any]]:
    options = chunk_options.copy()
    if custom_chunk_options:
        options.update(custom_chunk_options)

    if api_name is not None:
        full_summary = summarize(text, None, api_name, None, None, None)
    else:
        full_summary = "Full document summary not available."

    chunks = improved_chunking_process(text, options)
    total_chunks = len(chunks)

    chunked_text_with_headers = []
    for i, chunk in enumerate(chunks, 1):
        chunk_text = chunk['text']
        chunk_position = determine_chunk_position(chunk['metadata']['relative_position'])

        chunk_header = f"""
        Original Document: {file_name}
        Full Document Summary: {full_summary}
        Chunk: {i} of {total_chunks}
        Position: {chunk_position}

        --- Chunk Content ---
        """

        full_chunk_text = chunk_header + chunk_text
        chunk['text'] = full_chunk_text
        chunk['metadata']['file_name'] = file_name
        chunked_text_with_headers.append(chunk)

    return chunked_text_with_headers

def create_openai_embedding(text: str, model: str) -> List[float]:
    embedding = get_openai_embeddings(text, model)
    return embedding

#Dead
# def create_local_embedding(text: str, model: str, api_url: str, api_key: str) -> List[float]:
#     response = requests.post(
#         api_url,
#         json={"text": text, "model": model},
#         headers={"Authorization": f"Bearer {api_key}"}
#     )
#     response.raise_for_status()
#     return response.json().get('embedding', None)

# Dead
# def create_llamacpp_embedding(text: str, api_url: str) -> List[float]:
#     response = requests.post(
#         api_url,
#         json={"input": text}
#     )
#     response.raise_for_status()
#     return response.json()['embedding']

# dead
# def create_huggingface_embedding(text: str, model: str) -> List[float]:
#     tokenizer = AutoTokenizer.from_pretrained(model)
#     model = AutoModel.from_pretrained(model)
#
#     inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
#     with torch.no_grad():
#         outputs = model(**inputs)
#
#     embeddings = outputs.last_hidden_state.mean(dim=1)
#     return embeddings[0].tolist()

#
# End of File.
#######################################################################################################################
