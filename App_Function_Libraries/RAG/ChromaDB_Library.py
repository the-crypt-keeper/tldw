# ChromaDB_Library.py
# Description: Functions for managing embeddings in ChromaDB
#
# Imports:
import logging
from typing import List, Dict, Any
# 3rd-Party Imports:
import chromadb
from chromadb import Settings
from itertools import islice
#
# Local Imports:
from App_Function_Libraries.RAG.Embeddings_Create import chunk_for_embedding, create_embeddings_batch
from App_Function_Libraries.DB.DB_Manager import update_fts_for_media, db, batch_insert_chunks
from App_Function_Libraries.RAG.Embeddings_Create import create_embedding
from App_Function_Libraries.Summarization_General_Lib import summarize
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

def batched(iterable, n):
    "Batch data into lists of length n. The last batch may be shorter."
    it = iter(iterable)
    while True:
        batch = list(islice(it, n))
        if not batch:
            return
        yield batch


# FIXME - Fix summarization of entire document/storign in chunk issue
# FIXME - update all uses to reflect 'api_name' parameter
def process_and_store_content(content: str, collection_name: str, media_id: int, file_name: str,
                              create_embeddings: bool = False, create_summary: bool = False, api_name: str = None):
    try:
        logging.debug(f"Processing content for media_id {media_id} in collection {collection_name}")

        full_summary = None
        if create_summary and api_name:
            full_summary = summarize(content, None, api_name, None, None, None)

        chunks = chunk_for_embedding(content, file_name, full_summary, chunk_options)

        with db.get_connection() as conn:
            conn.execute("BEGIN TRANSACTION")
            try:
                batch_insert_chunks(conn, chunks, media_id)
                conn.commit()
            except Exception as e:
                conn.rollback()
                logging.error(f"Error in batch insert: {str(e)}")
                raise

        if create_embeddings:
            texts = [chunk['text'] for chunk in chunks]
            embeddings = create_embeddings_batch(texts, embedding_provider, embedding_model, embedding_api_url)
            ids = [f"{media_id}_chunk_{i}" for i in range(1, len(chunks) + 1)]
            metadatas = [{
                "media_id": str(media_id),
                "chunk_index": i,
                "total_chunks": len(chunks),
                "start_index": int(chunk['metadata']['start_index']),
                "end_index": int(chunk['metadata']['end_index']),
                "file_name": str(file_name),
                "relative_position": float(chunk['metadata']['relative_position'])
            } for i, chunk in enumerate(chunks, 1)]

            store_in_chroma(collection_name, texts, embeddings, ids, metadatas)

        update_fts_for_media(media_id)

    except Exception as e:
        logging.error(f"Error in process_and_store_content for media_id {media_id}: {str(e)}")
        raise


def check_embedding_status(selected_item, item_mapping):
    if not selected_item:
        return "Please select an item", ""

    try:
        item_id = item_mapping.get(selected_item)
        if item_id is None:
            return f"Invalid item selected: {selected_item}", ""

        item_title = selected_item.rsplit(' (', 1)[0]
        collection = chroma_client.get_or_create_collection(name="all_content_embeddings")

        result = collection.get(ids=[f"doc_{item_id}"], include=["embeddings", "metadatas"])
        logging.info(f"ChromaDB result for item '{item_title}' (ID: {item_id}): {result}")

        if not result['ids']:
            return f"No embedding found for item '{item_title}' (ID: {item_id})", ""

        if not result['embeddings'] or not result['embeddings'][0]:
            return f"Embedding data missing for item '{item_title}' (ID: {item_id})", ""

        embedding = result['embeddings'][0]
        metadata = result['metadatas'][0] if result['metadatas'] else {}
        embedding_preview = str(embedding[:50])
        status = f"Embedding exists for item '{item_title}' (ID: {item_id})"
        return status, f"First 50 elements of embedding:\n{embedding_preview}\n\nMetadata: {metadata}"

    except Exception as e:
        logging.error(f"Error in check_embedding_status: {str(e)}")
        return f"Error processing item: {selected_item}. Details: {str(e)}", ""

def reset_chroma_collection(collection_name: str):
    try:
        chroma_client.delete_collection(collection_name)
        chroma_client.create_collection(collection_name)
        logging.info(f"Reset ChromaDB collection: {collection_name}")
    except Exception as e:
        logging.error(f"Error resetting ChromaDB collection: {str(e)}")


def store_in_chroma(collection_name: str, texts: List[str], embeddings: List[List[float]], ids: List[str], metadatas: List[Dict[str, Any]]):
    try:
        collection = chroma_client.get_or_create_collection(name=collection_name)

        # Log the inputs for debugging
        logging.debug(f"Storing in ChromaDB - Collection: {collection_name}")
        logging.debug(f"Texts (first 100 chars): {texts[0][:100]}...")
        logging.debug(f"Embeddings (first 5 values): {embeddings[0][:5]}")
        logging.debug(f"IDs: {ids}")
        logging.debug(f"Metadatas: {metadatas}")

        # Use upsert instead of add/update
        collection.upsert(
            documents=texts,
            embeddings=embeddings,
            ids=ids,
            metadatas=metadatas
        )

        # Verify storage
        for doc_id in ids:
            result = collection.get(ids=[doc_id], include=["embeddings"])
            if not result['embeddings'] or result['embeddings'][0] is None:
                logging.error(f"Failed to store embedding for {doc_id}")
            else:
                logging.info(f"Embedding stored successfully for {doc_id}")

    except Exception as e:
        logging.error(f"Error storing embeddings in ChromaDB: {str(e)}")
        raise


# Function to perform vector search using ChromaDB + Keywords from the media_db
def vector_search(collection_name: str, query: str, k: int = 10) -> List[Dict[str, Any]]:
    try:
        query_embedding = create_embedding(query, embedding_provider, embedding_model, embedding_api_url)
        collection = chroma_client.get_collection(name=collection_name)
        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=k,
            include=["documents", "metadatas"]
        )
        return [{"content": doc, "metadata": meta} for doc, meta in zip(results['documents'][0], results['metadatas'][0])]
    except Exception as e:
        logging.error(f"Error in vector_search: {str(e)}")
        raise


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


#
# End of Functions for ChromaDB
#######################################################################################################################