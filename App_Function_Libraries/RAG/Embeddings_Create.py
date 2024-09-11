# Embeddings_Create.py
# Description: Functions for Creating and managing Embeddings in ChromaDB with LLama.cpp/OpenAI/Transformers
#
# Imports:
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

# FIXME - refactor/setup to use config file & perform chunking
def create_embedding(text: str) -> List[float]:

    if embedding_provider == 'openai':
        return get_openai_embeddings(text, embedding_model)
    elif embedding_provider == 'local':
        response = requests.post(
            embedding_api_url,
            json={"text": text, "model": embedding_model},
            headers={"Authorization": f"Bearer {embedding_api_key}"}
        )
        return response.json()['embedding']
    elif embedding_provider == 'huggingface':
        tokenizer = AutoTokenizer.from_pretrained(embedding_model)
        model = AutoModel.from_pretrained(embedding_model)

        inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
        with torch.no_grad():
            outputs = model(**inputs)

        # Use the mean of the last hidden state as the sentence embedding
        embeddings = outputs.last_hidden_state.mean(dim=1)
        # Convert to list for consistency
        return embeddings[0].tolist()
    else:
        raise ValueError(f"Unsupported embedding provider: {embedding_provider}")


def chunk_for_embedding(text: str, file_name: str, api_name, custom_chunk_options: Dict[str, Any] = None) -> List[Dict[str, Any]]:
    options = chunk_options.copy()
    if custom_chunk_options:
        options.update(custom_chunk_options)


    # FIXME
    if api_name is not None:
        # Generate summary of the full document
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


def create_llamacpp_embedding(text: str, api_url: str) -> List[float]:
    response = requests.post(
        api_url,
        json={"input": text}
    )
    if response.status_code == 200:
        return response.json()['embedding']
    else:
        raise Exception(f"Error from Llama.cpp API: {response.text}")


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

#
# End of File.
#######################################################################################################################
