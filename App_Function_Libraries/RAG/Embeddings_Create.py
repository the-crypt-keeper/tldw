# Embeddings_Create.py
# Description: Functions for Creating and managing Embeddings in ChromaDB with LLama.cpp/OpenAI/Transformers
#
# Imports:
import logging
import os
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
#
# Local Imports:
from App_Function_Libraries.LLM_API_Calls import get_openai_embeddings
from App_Function_Libraries.Utils.Utils import load_comprehensive_config
#
#######################################################################################################################
#
# Functions:

# FIXME - Version 2

# Load configuration
loaded_config = load_comprehensive_config()
embedding_provider = loaded_config['Embeddings']['embedding_provider']
embedding_model = loaded_config['Embeddings']['embedding_model']
embedding_api_url = loaded_config['Embeddings']['embedding_api_url']
embedding_api_key = loaded_config['Embeddings']['embedding_api_key']
model_dir = loaded_config['Embeddings'].get('model_dir', '/tldw/App_Function_Libraries/models/embedding_models/')

# Embedding Chunking Settings
chunk_size = loaded_config['Embeddings']['chunk_size']
overlap = loaded_config['Embeddings']['overlap']

# Global cache for embedding models
embedding_models = {}

class HuggingFaceEmbedder:
    def __init__(self, model_name, timeout_seconds=30):
        self.model_name = model_name
        self.tokenizer = None
        self.model = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.timeout_seconds = timeout_seconds
        self.last_used_time = 0
        self.unload_timer = None

    def load_model(self):
        if self.model is None:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, trust_remote_code=True)
            # Dirty hack? FIXME
            self.model = AutoModel.from_pretrained(self.model_name, trust_remote_code=True)
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
        try:
            with torch.no_grad():
                outputs = self.model(**inputs)
            embeddings = outputs.last_hidden_state.mean(dim=1)
            return embeddings.cpu().float().numpy()  # Convert to float32 before returning
        except RuntimeError as e:
            if "Got unsupported ScalarType BFloat16" in str(e):
                logging.warning("BFloat16 not supported. Falling back to float32.")
                # Convert model to float32
                self.model = self.model.float()
                with torch.no_grad():
                    outputs = self.model(**inputs)
                embeddings = outputs.last_hidden_state.mean(dim=1)
                return embeddings.cpu().float().numpy()
            else:
                raise

class ONNXEmbedder:
    def __init__(self, model_name, onnx_model_dir, timeout_seconds=30):
        self.model_name = model_name
        self.model_path = os.path.join(onnx_model_dir, f"{model_name}.onnx")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        self.session = None
        self.timeout_seconds = timeout_seconds
        self.last_used_time = 0
        self.unload_timer = None
        self.device = "cpu"  # ONNX Runtime will default to CPU unless GPU is configured

    def load_model(self):
        if self.session is None:
            if not os.path.exists(self.model_path):
                raise FileNotFoundError(f"ONNX model not found at {self.model_path}")
            logging.info(f"Loading ONNX model from {self.model_path}")
            self.session = ort.InferenceSession(self.model_path)
        self.last_used_time = time.time()
        self.reset_timer()

    def unload_model(self):
        if self.session is not None:
            logging.info("Unloading ONNX model to free resources.")
            self.session = None
        if self.unload_timer:
            self.unload_timer.cancel()

    def reset_timer(self):
        if self.unload_timer:
            self.unload_timer.cancel()
        self.unload_timer = Timer(self.timeout_seconds, self.unload_model)
        self.unload_timer.start()

    def create_embeddings(self, texts: List[str]) -> List[List[float]]:
        self.load_model()

        try:
            inputs = self.tokenizer(texts, return_tensors="np", padding=True, truncation=True, max_length=512)
            input_ids = inputs["input_ids"].astype(np.int64)
            attention_mask = inputs["attention_mask"].astype(np.int64)

            ort_inputs = {
                "input_ids": input_ids,
                "attention_mask": attention_mask
            }

            ort_outputs = self.session.run(None, ort_inputs)

            last_hidden_state = ort_outputs[0]
            embeddings = np.mean(last_hidden_state, axis=1)

            return embeddings.tolist()
        except Exception as e:
            logging.error(f"Error creating embeddings with ONNX model: {str(e)}")
            raise

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

@exponential_backoff()
@RateLimiter(max_calls=50, period=60)
def create_embeddings_batch(texts: List[str],
                            provider: str,
                            model: str,
                            api_url: str,
                            timeout_seconds: int = 300
                            ) -> List[List[float]]:
    global embedding_models

    try:
        if provider.lower() == 'huggingface':
            if model not in embedding_models:
                if model == "dunzhang/stella_en_400M_v5":
                    embedding_models[model] = ONNXEmbedder(model, model_dir, timeout_seconds)
                else:
                    embedding_models[model] = HuggingFaceEmbedder(model, timeout_seconds)
            embedder = embedding_models[model]
            return embedder.create_embeddings(texts)

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
    except Exception as e:
        logging.error(f"Error in create_embeddings_batch: {str(e)}")
        raise

def create_embedding(text: str, provider: str, model: str, api_url: str) -> List[float]:
    embedding = create_embeddings_batch([text], provider, model, api_url)[0]
    if isinstance(embedding, np.ndarray):
        embedding = embedding.tolist()
    return embedding

def create_openai_embedding(text: str, model: str) -> List[float]:
    embedding = get_openai_embeddings(text, model)
    return embedding


# FIXME - Version 1
# # FIXME - Add all globals to summarize.py
# loaded_config = load_comprehensive_config()
# embedding_provider = loaded_config['Embeddings']['embedding_provider']
# embedding_model = loaded_config['Embeddings']['embedding_model']
# embedding_api_url = loaded_config['Embeddings']['embedding_api_url']
# embedding_api_key = loaded_config['Embeddings']['embedding_api_key']
#
# # Embedding Chunking Settings
# chunk_size = loaded_config['Embeddings']['chunk_size']
# overlap = loaded_config['Embeddings']['overlap']
#
#
# # FIXME - Add logging
#
# class HuggingFaceEmbedder:
#     def __init__(self, model_name, timeout_seconds=120):  # Default timeout of 2 minutes
#         self.model_name = model_name
#         self.tokenizer = None
#         self.model = None
#         self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#         self.timeout_seconds = timeout_seconds
#         self.last_used_time = 0
#         self.unload_timer = None
#
#     def load_model(self):
#         if self.model is None:
#             self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
#             self.model = AutoModel.from_pretrained(self.model_name)
#             self.model.to(self.device)
#         self.last_used_time = time.time()
#         self.reset_timer()
#
#     def unload_model(self):
#         if self.model is not None:
#             del self.model
#             del self.tokenizer
#             if torch.cuda.is_available():
#                 torch.cuda.empty_cache()
#             self.model = None
#             self.tokenizer = None
#         if self.unload_timer:
#             self.unload_timer.cancel()
#
#     def reset_timer(self):
#         if self.unload_timer:
#             self.unload_timer.cancel()
#         self.unload_timer = Timer(self.timeout_seconds, self.unload_model)
#         self.unload_timer.start()
#
#     def create_embeddings(self, texts):
#         self.load_model()
#         inputs = self.tokenizer(texts, return_tensors="pt", padding=True, truncation=True, max_length=512)
#         inputs = {k: v.to(self.device) for k, v in inputs.items()}
#         with torch.no_grad():
#             outputs = self.model(**inputs)
#         embeddings = outputs.last_hidden_state.mean(dim=1)
#         return embeddings.cpu().numpy()
#
# # Global variable to hold the embedder
# huggingface_embedder = None
#
#
# class RateLimiter:
#     def __init__(self, max_calls, period):
#         self.max_calls = max_calls
#         self.period = period
#         self.calls = []
#         self.lock = Lock()
#
#     def __call__(self, func):
#         def wrapper(*args, **kwargs):
#             with self.lock:
#                 now = time.time()
#                 self.calls = [call for call in self.calls if call > now - self.period]
#                 if len(self.calls) >= self.max_calls:
#                     sleep_time = self.calls[0] - (now - self.period)
#                     time.sleep(sleep_time)
#                 self.calls.append(time.time())
#             return func(*args, **kwargs)
#         return wrapper
#
#
# def exponential_backoff(max_retries=5, base_delay=1):
#     def decorator(func):
#         @wraps(func)
#         def wrapper(*args, **kwargs):
#             for attempt in range(max_retries):
#                 try:
#                     return func(*args, **kwargs)
#                 except Exception as e:
#                     if attempt == max_retries - 1:
#                         raise
#                     delay = base_delay * (2 ** attempt)
#                     logging.warning(f"Attempt {attempt + 1} failed. Retrying in {delay} seconds. Error: {str(e)}")
#                     time.sleep(delay)
#         return wrapper
#     return decorator
#
#
# # FIXME - refactor/setup to use config file & perform chunking
# @exponential_backoff()
# @RateLimiter(max_calls=50, period=60)
# def create_embeddings_batch(texts: List[str], provider: str, model: str, api_url: str, timeout_seconds: int = 300) -> List[List[float]]:
#     global embedding_models
#
#     try:
#         if provider.lower() == 'huggingface':
#             if model not in embedding_models:
#                 if model == "dunzhang/stella_en_400M_v5":
#                     embedding_models[model] = ONNXEmbedder(model, model_dir, timeout_seconds)
#                 else:
#                     embedding_models[model] = HuggingFaceEmbedder(model, timeout_seconds)
#             embedder = embedding_models[model]
#             return embedder.create_embeddings(texts)
#
#         elif provider.lower() == 'openai':
#             logging.debug(f"Creating embeddings for {len(texts)} texts using OpenAI API")
#             return [create_openai_embedding(text, model) for text in texts]
#
#         elif provider.lower() == 'local':
#             response = requests.post(
#                 api_url,
#                 json={"texts": texts, "model": model},
#                 headers={"Authorization": f"Bearer {embedding_api_key}"}
#             )
#             if response.status_code == 200:
#                 return response.json()['embeddings']
#             else:
#                 raise Exception(f"Error from local API: {response.text}")
#         else:
#             raise ValueError(f"Unsupported embedding provider: {provider}")
#     except Exception as e:
#         logging.error(f"Error in create_embeddings_batch: {str(e)}")
#         raise
#
# def create_embedding(text: str, provider: str, model: str, api_url: str) -> List[float]:
#     return create_embeddings_batch([text], provider, model, api_url)[0]
#
#
# def create_openai_embedding(text: str, model: str) -> List[float]:
#     embedding = get_openai_embeddings(text, model)
#     return embedding
#
#
# # FIXME - refactor to use onnx embeddings callout
# def create_stella_embeddings(text: str) -> List[float]:
#     if embedding_provider == 'local':
#         # Load the model and tokenizer
#         tokenizer = AutoTokenizer.from_pretrained("dunzhang/stella_en_400M_v5")
#         model = AutoModel.from_pretrained("dunzhang/stella_en_400M_v5")
#
#         # Tokenize and encode the text
#         inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
#
#         # Generate embeddings
#         with torch.no_grad():
#             outputs = model(**inputs)
#
#         # Use the mean of the last hidden state as the sentence embedding
#         embeddings = outputs.last_hidden_state.mean(dim=1)
#
#         return embeddings[0].tolist()  # Convert to list for consistency
#     elif embedding_provider == 'openai':
#         return get_openai_embeddings(text, embedding_model)
#     else:
#         raise ValueError(f"Unsupported embedding provider: {embedding_provider}")
# #
# # End of F
# ##############################################################
#
#
# ##############################################################
# #
# # ONNX Embeddings Functions
#
# # FIXME - UPDATE
# # Define the model path
# model_dir = "/tldw/App_Function_Libraries/models/embedding_models/"
# model_name = "your-huggingface-model-name"
# onnx_model_path = os.path.join(model_dir, model_name, "model.onnx")
#
# # Tokenizer download (if applicable)
# #tokenizer = AutoTokenizer.from_pretrained(model_name)
#
# # Ensure the model directory exists
# #if not os.path.exists(onnx_model_path):
#     # You can add logic to download the ONNX model from a remote source
#     # if it's not already available in the folder.
#     # Example: huggingface_hub.download (if model is hosted on Hugging Face Hub)
# #    raise Exception(f"ONNX model not found at {onnx_model_path}")
#
# class ONNXEmbedder:
#     def __init__(self, model_name, model_dir, timeout_seconds=120):
#         self.model_name = model_name
#         self.model_path = os.path.join(model_dir, f"{model_name}.onnx")
#         self.tokenizer = AutoTokenizer.from_pretrained(model_name)
#         self.session = None
#         self.timeout_seconds = timeout_seconds
#         self.last_used_time = 0
#         self.unload_timer = None
#         self.device = "cpu"  # ONNX Runtime will default to CPU unless GPU is configured
#
#     def load_model(self):
#         if self.session is None:
#             if not os.path.exists(self.model_path):
#                 raise FileNotFoundError(f"ONNX model not found at {self.model_path}")
#             logging.info(f"Loading ONNX model from {self.model_path}")
#             self.session = ort.InferenceSession(self.model_path)
#         self.last_used_time = time.time()
#         self.reset_timer()
#
#     def unload_model(self):
#         if self.session is not None:
#             logging.info("Unloading ONNX model to free resources.")
#             self.session = None
#         if self.unload_timer:
#             self.unload_timer.cancel()
#
#     def reset_timer(self):
#         if self.unload_timer:
#             self.unload_timer.cancel()
#         self.unload_timer = Timer(self.timeout_seconds, self.unload_model)
#         self.unload_timer.start()
#
#     def create_embeddings(self, texts: List[str]) -> List[List[float]]:
#         self.load_model()
#
#         try:
#             inputs = self.tokenizer(texts, return_tensors="np", padding=True, truncation=True, max_length=512)
#             input_ids = inputs["input_ids"].astype(np.int64)
#             attention_mask = inputs["attention_mask"].astype(np.int64)
#
#             ort_inputs = {
#                 "input_ids": input_ids,
#                 "attention_mask": attention_mask
#             }
#
#             ort_outputs = self.session.run(None, ort_inputs)
#
#             last_hidden_state = ort_outputs[0]
#             embeddings = np.mean(last_hidden_state, axis=1)
#
#             return embeddings.tolist()
#         except Exception as e:
#             logging.error(f"Error creating embeddings with ONNX model: {str(e)}")
#             raise
#
# # Global cache for the ONNX embedder instance
# onnx_embedder = None
#
# # Global cache for embedding models
# embedding_models = {}
#
# def create_onnx_embeddings(texts: List[str]) -> List[List[float]]:
#     global onnx_embedder
#     model_dir = "/tldw/App_Function_Libraries/models/embedding_models/"
#     model_name = "your-huggingface-model-name"  # This can be pulled from config
#
#     if onnx_embedder is None:
#         onnx_embedder = ONNXEmbedder(model_name=model_name, model_dir=model_dir)
#
#     # Generate embeddings
#     embeddings = onnx_embedder.create_embeddings(texts)
#     return embeddings
#
# #
# # End of ONNX Embeddings Functions
# ##############################################################

#
# End of File.
#######################################################################################################################
