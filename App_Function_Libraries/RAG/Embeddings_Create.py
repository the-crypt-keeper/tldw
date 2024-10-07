# Embeddings_Create.py
# Description: Functions for Creating and managing Embeddings in ChromaDB with LLama.cpp/OpenAI/Transformers
#
# Imports:
import logging
import time
from functools import wraps
from threading import Lock, Timer
from typing import List
#
# 3rd-Party Imports:
import numpy as np
import onnxruntime as ort
import requests
from transformers import AutoTokenizer, AutoModel
import torch

from App_Function_Libraries.DB.Character_Chat_DB import fetch_all_chats
#
# Local Imports:
from App_Function_Libraries.LLM_API_Calls import get_openai_embeddings
from App_Function_Libraries.RAG.ChromaDB_Library import process_and_store_content
from App_Function_Libraries.Utils.Utils import load_comprehensive_config
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


def create_onnx_embeddings(text: str) -> List[float]:
    if embedding_provider == 'local':
        # Load the tokenizer (same as before)
        tokenizer = AutoTokenizer.from_pretrained("dunzhang/stella_en_400M_v5")

        # Load the ONNX model
        onnx_model_path = "path_to_your_stella_model.onnx"
        session = ort.InferenceSession(onnx_model_path)

        # Tokenize and encode the text
        inputs = tokenizer(text, return_tensors="np", padding=True, truncation=True, max_length=512)

        # Prepare the ONNX model input in int64 format
        input_ids = inputs["input_ids"].astype(np.int64)  # Ensure int64 data type for ONNX
        attention_mask = inputs["attention_mask"].astype(np.int64)

        # Create the input dictionary for ONNX Runtime
        ort_inputs = {
            "input_ids": input_ids,
            "attention_mask": attention_mask
        }

        # Perform inference with ONNX Runtime
        ort_outputs = session.run(None, ort_inputs)

        # Extract the last hidden state (typically the first output of the model)
        last_hidden_state = ort_outputs[0]  # Confirm this matches your model structure

        # Use the mean of the last hidden state along the sequence dimension (axis=1)
        embeddings = np.mean(last_hidden_state, axis=1)

        return embeddings[0].tolist()  # Convert to list for consistency
    elif embedding_provider == 'openai':
        return get_openai_embeddings(text, embedding_model)
    else:
        raise ValueError(f"Unsupported embedding provider: {embedding_provider}")

def create_openai_embedding(text: str, model: str) -> List[float]:
    embedding = get_openai_embeddings(text, model)
    return embedding


def embed_and_store_chats():
    """
    Fetch all chat messages, create embeddings, and store them in ChromaDB.
    """
    chats = fetch_all_chats()
    total_chats = len(chats)

    for index, chat in enumerate(chats, 1):
        media_id = chat['id']  # Assuming 'id' is the primary key
        content = chat['chat_history']  # Assuming 'chat_history' contains the message content
        file_name = f"chat_{media_id}"

        collection_name = "all_chat_embeddings"

        logging.info(f"Processing chat {index} of {total_chats}: ID {media_id}")

        try:
            # Process and store content
            process_and_store_content(
                database=None,  # Assuming no additional database interaction needed
                content=content,
                collection_name=collection_name,
                media_id=media_id,
                file_name=file_name,
                create_embeddings=True,
                create_contextualized=False,  # Contextualization may not be needed for chats
                api_name="gpt-3.5-turbo"  # Or any other relevant model
            )

            logging.info(f"Successfully processed chat ID {media_id}")
        except Exception as e:
            logging.error(f"Error processing chat ID {media_id}: {str(e)}")


#
# End of File.
#######################################################################################################################
