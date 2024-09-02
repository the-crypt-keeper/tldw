import configparser
import logging
import sqlite3
from typing import List, Dict, Any

import chromadb
import requests
from chromadb import Settings

from App_Function_Libraries.Chunk_Lib import improved_chunking_process

#######################################################################################################################
#
# Functions for ChromaDB

# Get ChromaDB settings
# Load configuration
config = configparser.ConfigParser()
config.read('config.txt')
chroma_db_path = config.get('Database', 'chroma_db_path', fallback='chroma_db')
chroma_client = chromadb.PersistentClient(path=chroma_db_path, settings=Settings(anonymized_telemetry=False))

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

    # Initialize or get the ChromaDB collection
    collection = chroma_client.get_or_create_collection(name=collection_name)

    # Check if embeddings already exist for this media_id
    existing_embeddings = collection.get(ids=[f"{media_id}_chunk_{i}" for i in range(len(content))])

    if existing_embeddings and len(existing_embeddings) > 0:
        logging.info(f"Embeddings already exist for media ID {media_id}, skipping...")
    else:
        # Process and store content if embeddings do not already exist
        process_and_store_content(content, collection_name, media_id)
        logging.info(f"Updated ChromaDB embeddings for media ID: {media_id}")


# Function to process content, create chunks, embeddings, and store in ChromaDB and SQLite
def process_and_store_content(content: str, collection_name: str, media_id: int):
    # Process the content into chunks
    chunks = improved_chunking_process(content, chunk_options)
    texts = [chunk['text'] for chunk in chunks]

    # Generate embeddings for each chunk
    embeddings = [create_embedding(text) for text in texts]

    # Create unique IDs for each chunk using the media_id and chunk index
    ids = [f"{media_id}_chunk_{i}" for i in range(len(texts))]

    # Store the texts, embeddings, and IDs in ChromaDB
    store_in_chroma(collection_name, texts, embeddings, ids)

    # Store the chunks in SQLite FTS as well
    from App_Function_Libraries.DB.DB_Manager import db
    with db.get_connection() as conn:
        cursor = conn.cursor()
        for text in texts:
            cursor.execute("INSERT INTO media_fts (content) VALUES (?)", (text,))
        conn.commit()


# Function to store documents and their embeddings in ChromaDB
def store_in_chroma(collection_name: str, texts: List[str], embeddings: List[List[float]], ids: List[str]):
    collection = chroma_client.get_or_create_collection(name=collection_name)
    collection.add(
        documents=texts,
        embeddings=embeddings,
        ids=ids
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
        # FIXME - This is a placeholder for API calls to a local embedding model
        response = requests.post(
            embedding_api_url,
            json={"text": text, "model": embedding_model},
            headers={"Authorization": f"Bearer {embedding_api_key}"}
        )
        return response.json()['embedding']
    # FIXME - this seems correct, but idk....
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


def create_all_embeddings(api_choice: str) -> str:
    try:
        global embedding_provider
        embedding_provider = api_choice

        all_content = get_all_content_from_database()

        if not all_content:
            return "No content found in the database."

        texts_to_embed = []
        embeddings_to_store = []
        ids_to_store = []
        collection_name = "all_content_embeddings"

        # Initialize or get the ChromaDB collection
        collection = chroma_client.get_or_create_collection(name=collection_name)

        for content_item in all_content:
            media_id = content_item['id']
            text = content_item['content']

            # Check if the embedding already exists in ChromaDB
            embedding_exists = collection.get(ids=[f"doc_{media_id}"])

            if embedding_exists:
                logging.info(f"Embedding already exists for media ID {media_id}, skipping...")
                continue  # Skip if embedding already exists

            # Create the embedding
            embedding = create_embedding(text)

            # Collect the text, embedding, and ID for batch storage
            texts_to_embed.append(text)
            embeddings_to_store.append(embedding)
            ids_to_store.append(f"doc_{media_id}")

        # Store all new embeddings in ChromaDB
        if texts_to_embed and embeddings_to_store:
            store_in_chroma(collection_name, texts_to_embed, embeddings_to_store, ids_to_store)

        return "Embeddings created and stored successfully for all new content."
    except Exception as e:
        logging.error(f"Error during embedding creation: {str(e)}")
        return f"Error: {str(e)}"


def get_all_content_from_database() -> List[Dict[str, Any]]:
    """
    Retrieve all media content from the database that requires embedding.

    Returns:
        List[Dict[str, Any]]: A list of dictionaries, each containing the media ID, content, title, and other relevant fields.
    """
    try:
        from App_Function_Libraries.DB.DB_Manager import db
        with db.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT id, content, title, author, type
                FROM Media
                WHERE is_trash = 0  -- Exclude items marked as trash
            """)
            media_items = cursor.fetchall()

            # Convert the results into a list of dictionaries
            all_content = [
                {
                    'id': item[0],
                    'content': item[1],
                    'title': item[2],
                    'author': item[3],
                    'type': item[4]
                }
                for item in media_items
            ]

        return all_content

    except sqlite3.Error as e:
        logging.error(f"Error retrieving all content from database: {e}")
        from App_Function_Libraries.DB.SQLite_DB import DatabaseError
        raise DatabaseError(f"Error retrieving all content from database: {e}")


def store_in_chroma_with_citation(collection_name: str, texts: List[str], embeddings: List[List[float]], ids: List[str], sources: List[str]):
    collection = chroma_client.get_or_create_collection(name=collection_name)
    collection.add(
        documents=texts,
        embeddings=embeddings,
        ids=ids,
        metadatas=[{'source': source} for source in sources]
    )


#
# End of Functions for ChromaDB
#######################################################################################################################