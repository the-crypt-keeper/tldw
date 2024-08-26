import configparser
import logging
from typing import List

import chromadb
import requests

from App_Function_Libraries.Chunk_Lib import improved_chunking_process

#######################################################################################################################
#
# Functions for ChromaDB

# Get ChromaDB settings
# Load configuration
config = configparser.ConfigParser()
config.read('config.txt')
chroma_db_path = config.get('Database', 'chroma_db_path', fallback='chroma_db')
chroma_client = chromadb.PersistentClient(path=chroma_db_path)

# Get embedding settings
embedding_provider = config.get('Embeddings', 'provider', fallback='openai')
embedding_model = config.get('Embeddings', 'model', fallback='text-embedding-3-small')
embedding_api_key = config.get('Embeddings', 'api_key', fallback='')
embedding_api_url = config.get('Embeddings', 'api_url', fallback='')

# Get chunking options
chunk_options = {
    'method': config.get('Chunking', 'method', fallback='words'),
    'max_size': config.getint('Chunking', 'max_size', fallback=400),
    'overlap': config.getint('Chunking', 'overlap', fallback=200),
    'adaptive': config.getboolean('Chunking', 'adaptive', fallback=False),
    'multi_level': config.getboolean('Chunking', 'multi_level', fallback=False),
    'language': config.get('Chunking', 'language', fallback='english')
}

def auto_update_chroma_embeddings(media_id: int, content: str):
    """
    Automatically update ChromaDB embeddings when a new item is ingested into the SQLite database.

    :param media_id: The ID of the newly ingested media item
    :param content: The content of the newly ingested media item
    """
    collection_name = f"media_{media_id}"
    process_and_store_content(content, collection_name)
    logging.info(f"Updated ChromaDB embeddings for media ID: {media_id}")

# Function to process content, create chunks, embeddings, and store in ChromaDB and SQLite
def process_and_store_content(content: str, collection_name: str):
    chunks = improved_chunking_process(content, chunk_options)
    texts = [chunk['text'] for chunk in chunks]
    embeddings = [create_embedding(text) for text in texts]
    store_in_chroma(collection_name, texts, embeddings)

    # Store in SQLite FTS as well
    from App_Function_Libraries.DB_Manager import db
    with db.get_connection() as conn:
        cursor = conn.cursor()
        for text in texts:
            cursor.execute("INSERT INTO media_fts (content) VALUES (?)", (text,))
        conn.commit()


# Function to store documents and their embeddings in ChromaDB
def store_in_chroma(collection_name: str, texts: List[str], embeddings: List[List[float]]):
    collection = chroma_client.get_or_create_collection(name=collection_name)
    for i, (text, embedding) in enumerate(zip(texts, embeddings)):
        collection.add(
            documents=[text],
            embeddings=[embedding],
            ids=[f"doc_{i}"]
        )


# Function to perform vector search using ChromaDB
def vector_search(collection_name: str, query: str, k: int = 10) -> List[str]:
    query_embedding = create_embedding(query)
    collection = chroma_client.get_collection(name=collection_name)
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=k
    )
    return results['documents'][0]


def create_embedding(text: str) -> List[float]:
    if embedding_provider == 'openai':
        import openai
        openai.api_key = embedding_api_key
        response = openai.Embedding.create(input=text, model=embedding_model)
        return response['data'][0]['embedding']
    elif embedding_provider == 'local':
        # Assuming a local API that accepts POST requests with JSON payload
        response = requests.post(
            embedding_api_url,
            json={"text": text, "model": embedding_model},
            headers={"Authorization": f"Bearer {embedding_api_key}"}
        )
        return response.json()['embedding']
    elif embedding_provider == 'huggingface':
        from transformers import AutoTokenizer, AutoModel
        import torch

        tokenizer = AutoTokenizer.from_pretrained(embedding_model)
        model = AutoModel.from_pretrained(embedding_model)

        inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
        with torch.no_grad():
            outputs = model(**inputs)

        # Use the mean of the last hidden state as the sentence embedding
        embeddings = outputs.last_hidden_state.mean(dim=1)
        return embeddings[0].tolist()  # Convert to list for consistency
    else:
        raise ValueError(f"Unsupported embedding provider: {embedding_provider}")



#
# End of Functions for ChromaDB
#######################################################################################################################