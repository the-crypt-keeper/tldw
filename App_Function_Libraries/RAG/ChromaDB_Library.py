# ChromaDB_Library.py
# Description: Functions for managing embeddings in ChromaDB
#
# Imports:
import logging
from typing import List, Dict, Any
# 3rd-Party Imports:
import chromadb
from chromadb import Settings
#
# Local Imports:
from App_Function_Libraries.RAG.Embeddings_Create import chunk_for_embedding
from App_Function_Libraries.DB.DB_Manager import add_media_chunk, update_fts_for_media, \
    get_all_content_from_database
from App_Function_Libraries.RAG.Embeddings_Create import create_embedding, create_llamacpp_embedding, \
    create_openai_embedding
from App_Function_Libraries.Utils.Utils import get_database_path, ensure_directory_exists, \
    load_comprehensive_config
#
#######################################################################################################################
#
# Config Settings for ChromaDB Functions
#
# FIXME - Refactor so that all globals are set in summarize.py
# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
#
# Load config
config = load_comprehensive_config()
#
# ChromaDB settings
chroma_db_path = config.get('Database', 'chroma_db_path', fallback=get_database_path('chroma_db'))
ensure_directory_exists(chroma_db_path)
chroma_client = chromadb.PersistentClient(path=chroma_db_path, settings=Settings(anonymized_telemetry=False))
#
# Embedding settings
embedding_provider = config.get('Embeddings', 'embedding_provider', fallback='openai')
embedding_model = config.get('Embeddings', 'embedding_model', fallback='text-embedding-3-small')
embedding_api_key = config.get('Embeddings', 'api_key', fallback='')
embedding_api_url = config.get('Embeddings', 'api_url', fallback='')
#
# Chunking options
chunk_options = {
    'method': config.get('Chunking', 'method', fallback='words'),
    'max_size': config.getint('Chunking', 'max_size', fallback=400),
    'overlap': config.getint('Chunking', 'overlap', fallback=200),
    'adaptive': config.getboolean('Chunking', 'adaptive', fallback=False),
    'multi_level': config.getboolean('Chunking', 'multi_level', fallback=False),
    'language': config.get('Chunking', 'language', fallback='english')
}
#
# End of Config Settings
#######################################################################################################################
#
# Functions:

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
        process_and_store_content(content, collection_name, media_id, f"media_{media_id}")
        logging.info(f"Updated ChromaDB embeddings for media ID: {media_id}")


# Function to process content, create chunks, embeddings, and store in ChromaDB and SQLite
# def process_and_store_content(content: str, collection_name: str, media_id: int):
#     # Process the content into chunks
#     chunks = improved_chunking_process(content, chunk_options)
#     texts = [chunk['text'] for chunk in chunks]
#
#     # Generate embeddings for each chunk
#     embeddings = [create_embedding(text) for text in texts]
#
#     # Create unique IDs for each chunk using the media_id and chunk index
#     ids = [f"{media_id}_chunk_{i}" for i in range(len(texts))]
#
#     # Store the texts, embeddings, and IDs in ChromaDB
#     store_in_chroma(collection_name, texts, embeddings, ids)
#
#     # Store the chunk metadata in SQLite
#     for i, chunk in enumerate(chunks):
#         add_media_chunk(media_id, chunk['text'], chunk['start'], chunk['end'], ids[i])
#
#     # Update the FTS table
#     update_fts_for_media(media_id)

# Function to process content, create chunks, embeddings, and store in ChromaDB and SQLite with metadata
# def process_and_store_content(content: str, collection_name: str, media_id: int):
#     # Process the content into chunks
#     chunks = improved_chunking_process(content, chunk_options)
#     texts = [chunk['text'] for chunk in chunks]
#
#     # Generate embeddings for each chunk
#     embeddings = [create_embedding(text) for text in texts]
#     # Create unique IDs for each chunk using the media_id and chunk index
#     ids = [f"{media_id}_chunk_{i}" for i in range(len(texts))]
#     # Create metadata for each chunk tagged to the media_id
#     metadatas = [{"media_id": media_id} for _ in range(len(texts))]
#
#     # Store the texts, embeddings, and IDs in ChromaDB
#     store_in_chroma(collection_name, texts, embeddings, ids, metadatas)
#
#     # Store the chunk metadata in SQLite
#     for i, chunk in enumerate(chunks):
#         add_media_chunk(media_id, chunk['text'], chunk['start'], chunk['end'], ids[i])
#
#     # Update FTS table
#     update_fts_for_media(media_id)

# Function to store media according to ID and their embeddings in ChromaDB
# def process_and_store_content(content: str, collection_name: str, media_id: int):
#     # Process the content into chunks
#     chunks = improved_chunking_process(content, chunk_options)
#
#     texts = []
#     embeddings = []
#     ids = []
#     metadatas = []
#
#     for i, chunk in enumerate(chunks):
#         chunk_text = chunk['text']
#         chunk_embedding = create_embedding(chunk_text)
#         chunk_id = f"{media_id}_chunk_{i}"
#
#         texts.append(chunk_text)
#         embeddings.append(chunk_embedding)
#         ids.append(chunk_id)
#         metadatas.append({
#             "media_id": media_id,
#             "chunk_index": i,
#             "start_index": chunk['metadata']['start_index'],
#             "end_index": chunk['metadata']['end_index']
#         })
#
#         # Store the chunk metadata in SQLite
#         add_media_chunk(media_id, chunk_text, chunk['metadata']['start_index'], chunk['metadata']['end_index'], chunk_id)
#
#     # Get or create the collection
#     collection = chroma_client.get_or_create_collection(name=collection_name)
#
#     # Check for existing IDs and update or add as necessary
#     existing_ids = collection.get(ids=ids)['ids']
#     new_ids = [id for id in ids if id not in existing_ids]
#     update_ids = [id for id in ids if id in existing_ids]
#
#     if new_ids:
#         collection.add(
#             documents=[text for i, text in enumerate(texts) if ids[i] in new_ids],
#             embeddings=[emb for i, emb in enumerate(embeddings) if ids[i] in new_ids],
#             ids=new_ids,
#             metadatas=[meta for i, meta in enumerate(metadatas) if ids[i] in new_ids]
#         )
#
#     if update_ids:
#         collection.update(
#             documents=[text for i, text in enumerate(texts) if ids[i] in update_ids],
#             embeddings=[emb for i, emb in enumerate(embeddings) if ids[i] in update_ids],
#             ids=update_ids,
#             metadatas=[meta for i, meta in enumerate(metadatas) if ids[i] in update_ids]
#         )
#
#     # Update FTS table
#     update_fts_for_media(media_id)

# Function to store media according to ID and their embeddings in ChromaDB using chunking + Metadata
def process_and_store_content(content: str, collection_name: str, media_id: int, file_name: str):
    try:
        # Use the new chunk_for_embedding function
        chunks = chunk_for_embedding(content, file_name, chunk_options)

        texts = []
        embeddings = []
        ids = []
        metadatas = []

        for i, chunk in enumerate(chunks, 1):
            try:
                chunk_text = chunk['text']
                chunk_embedding = create_embedding(chunk_text)
                chunk_id = f"{media_id}_chunk_{i}"

                texts.append(chunk_text)
                embeddings.append(chunk_embedding)
                ids.append(chunk_id)
                metadatas.append({
                    "media_id": media_id,
                    "chunk_index": i,
                    "total_chunks": len(chunks),
                    "start_index": chunk['metadata']['start_index'],
                    "end_index": chunk['metadata']['end_index'],
                    "file_name": file_name,
                    "relative_position": chunk['metadata']['relative_position']
                })

                # Store the chunk metadata in SQLite
                add_media_chunk(media_id, chunk_text, chunk['metadata']['start_index'],
                                chunk['metadata']['end_index'], chunk_id)

            except Exception as e:
                logger.error(f"Error processing chunk {i} for media_id {media_id}: {str(e)}")

        # Get or create the collection
        collection = chroma_client.get_or_create_collection(name=collection_name)

        # Check for existing IDs and update or add as necessary
        existing_ids = collection.get(ids=ids)['ids']
        new_ids = [id for id in ids if id not in existing_ids]
        update_ids = [id for id in ids if id in existing_ids]

        if new_ids:
            collection.add(
                documents=[text for i, text in enumerate(texts) if ids[i] in new_ids],
                embeddings=[emb for i, emb in enumerate(embeddings) if ids[i] in new_ids],
                ids=new_ids,
                metadatas=[meta for i, meta in enumerate(metadatas) if ids[i] in new_ids]
            )

        if update_ids:
            collection.update(
                documents=[text for i, text in enumerate(texts) if ids[i] in update_ids],
                embeddings=[emb for i, emb in enumerate(embeddings) if ids[i] in update_ids],
                ids=update_ids,
                metadatas=[meta for i, meta in enumerate(metadatas) if ids[i] in update_ids]
            )

        # Update FTS table
        update_fts_for_media(media_id)

    except Exception as e:
        logger.error(f"Error in process_and_store_content for media_id {media_id}: {str(e)}")
        raise

def create_all_embeddings(api_choice: str, model_or_url: str) -> str:
    try:
        all_content = get_all_content_from_database()

        if not all_content:
            return "No content found in the database."

        texts_to_embed = []
        embeddings_to_store = []
        ids_to_store = []
        metadatas_to_store = []
        collection_name = "all_content_embeddings"

        collection = chroma_client.get_or_create_collection(name=collection_name)

        for content_item in all_content:
            media_id = content_item['id']
            text = content_item['content']

            embedding_exists = collection.get(ids=[f"doc_{media_id}"])

            if embedding_exists['ids']:
                logging.info(f"Embedding already exists for media ID {media_id}, skipping...")
                continue

            # FIXME - Refactor to use the create_embedding function
            if api_choice == "openai":
                embedding = create_openai_embedding(text, model_or_url)
            else:  # Llama.cpp
                embedding = create_llamacpp_embedding(text, model_or_url)

            texts_to_embed.append(text)
            embeddings_to_store.append(embedding)
            ids_to_store.append(f"doc_{media_id}")
            metadatas_to_store.append({"media_id": media_id})

        if texts_to_embed and embeddings_to_store:
            store_in_chroma(collection_name, texts_to_embed, embeddings_to_store, ids_to_store, metadatas_to_store)

        return "Embeddings created and stored successfully for all new content."
    except Exception as e:
        logging.error(f"Error during embedding creation: {str(e)}")
        return f"Error: {str(e)}"


def check_embedding_status(selected_item):
    if not selected_item:
        return "Please select an item", ""
    item_id = selected_item.split('(')[0].strip()
    collection = chroma_client.get_or_create_collection(name="all_content_embeddings")
    result = collection.get(ids=[f"doc_{item_id}"])
    if result['ids']:
        embedding = result['embeddings'][0]
        metadata = result['metadatas'][0]
        embedding_preview = str(embedding[:50])  # Convert first 50 elements to string
        return f"Embedding exists for item: {item_id}", f"Embedding preview: {embedding_preview}...\nMetadata: {metadata}"
    else:
        return f"No embedding found for item: {item_id}", ""


def create_new_embedding(selected_item, api_choice, openai_model, llamacpp_url):
    if not selected_item:
        return "Please select an item"
    item_id = selected_item.split('(')[0].strip()
    items = get_all_content_from_database()
    item = next((item for item in items if item['title'] == item_id), None)
    if not item:
        return f"Item not found: {item_id}"

    try:
        if api_choice == "OpenAI":
            embedding = create_embedding(item['content'])
        else:  # Llama.cpp
            embedding = create_embedding(item['content'])

        collection_name = "all_content_embeddings"
        store_in_chroma(collection_name, [item['content']], [embedding], [f"doc_{item['id']}"], [{"media_id": item['id']}])
        return f"New embedding created and stored for item: {item_id}"
    except Exception as e:
        return f"Error creating embedding: {str(e)}"


# Function to store documents and their embeddings in ChromaDB
def store_in_chroma(collection_name: str, texts: List[str], embeddings: List[List[float]], ids: List[str], metadatas: List[Dict[str, Any]]):
    collection = chroma_client.get_or_create_collection(name=collection_name)
    collection.add(
        documents=texts,
        embeddings=embeddings,
        ids=ids,
        metadatas=metadatas
    )


# Function to perform vector search using ChromaDB + Keywords from the media_db
def vector_search(collection_name: str, query: str, k: int = 10) -> List[Dict[str, Any]]:
    try:
        logger.info(f"Converting query to vectors: {query}")
        query_embedding = create_embedding(query)
        collection = chroma_client.get_collection(name=collection_name)
        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=k,
            include=["documents", "metadatas"]
        )
        return [{"content": doc, "metadata": meta} for doc, meta in zip(results['documents'][0], results['metadatas'][0])]
    except Exception as e:
        logger.error(f"Error in vector_search: {str(e)}")
        raise


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