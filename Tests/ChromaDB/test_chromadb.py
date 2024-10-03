# test_chromadb.py
# Description: This file contains the test cases for the ChromaDB_Library.py file in the App_Function_Libraries directory.
#
# Imports
import os
import sys

from unittest.mock import patch, MagicMock
#
# Third-party library imports
import pytest
#
####################################################################################################
#
# Status: FIXME

# Add the project root (parent directory of App_Function_Libraries) to sys.path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, project_root)

print(f"Project root added to sys.path: {project_root}")

# Local Imports
from App_Function_Libraries.RAG.ChromaDB_Library import (
    preprocess_all_content, process_and_store_content, check_embedding_status,
    reset_chroma_collection, vector_search, store_in_chroma, batched, situate_context, schedule_embedding
)
#
############################################
# Fixtures for Reusable Mocking and Setup
############################################

@pytest.fixture
def mock_chroma_client(mocker):
    """Fixture to mock the ChromaDB client."""
    mock_client = mocker.patch('App_Function_Libraries.RAG.ChromaDB_Library.chroma_client')
    return mock_client

@pytest.fixture
def mock_collection(mock_chroma_client):
    """Fixture to mock ChromaDB collection."""
    mock_collection = MagicMock()
    mock_chroma_client.get_or_create_collection.return_value = mock_collection
    return mock_collection


##############################
# Test: preprocess_all_content
##############################

@pytest.fixture
def mock_unprocessed_media(mocker):
    """Fixture to mock unprocessed media data."""
    return [(1, "Test Content", "video", "test_file.mp4")]

@pytest.fixture
def mock_process_and_store(mocker):
    """Fixture to mock process_and_store_content."""
    return mocker.patch("App_Function_Libraries.RAG.ChromaDB_Library.process_and_store_content")

@pytest.fixture
def mock_mark_media_processed(mocker):
    """Fixture to mock mark_media_as_processed."""
    return mocker.patch("App_Function_Libraries.RAG.ChromaDB_Library.mark_media_as_processed")

def test_preprocess_all_content(mock_unprocessed_media, mock_process_and_store, mock_mark_media_processed, mocker):
    # Mock get_unprocessed_media to return unprocessed media
    mocker.patch('App_Function_Libraries.RAG.ChromaDB_Library.get_unprocessed_media', return_value=mock_unprocessed_media)

    preprocess_all_content(database="mock_db", create_contextualized=False)

    mock_process_and_store.assert_called_once_with(
        database="mock_db",
        content="Test Content",
        collection_name="video_1",
        media_id=1,
        file_name="test_file.mp4",
        create_embeddings=True,
        create_contextualized=False,
        api_name="gpt-3.5-turbo"
    )
    mock_mark_media_processed.assert_called_once_with("mock_db", 1)


##############################
# Test: process_and_store_content
##############################

@pytest.fixture
def mock_chunk_for_embedding(mocker):
    """Fixture to mock chunk_for_embedding."""
    return mocker.patch("App_Function_Libraries.RAG.ChromaDB_Library.chunk_for_embedding", return_value=[
        {"text": "Chunk 1 text", "metadata": {"start_index": 0, "end_index": 100, "file_name": "test.mp4", "relative_position": 0.1}},
        {"text": "Chunk 2 text", "metadata": {"start_index": 101, "end_index": 200, "file_name": "test.mp4", "relative_position": 0.2}}
    ])

@pytest.fixture
def mock_process_chunks(mocker):
    """Fixture to mock process_chunks."""
    return mocker.patch("App_Function_Libraries.RAG.ChromaDB_Library.process_chunks")

@pytest.fixture
def mock_create_embeddings_batch(mocker):
    """Fixture to mock create_embeddings_batch."""
    return mocker.patch("App_Function_Libraries.RAG.ChromaDB_Library.create_embeddings_batch", return_value=[[0.1, 0.2], [0.3, 0.4]])

@pytest.fixture
def mock_situate_context(mocker):
    """Fixture to mock situate_context."""
    return mocker.patch("App_Function_Libraries.RAG.ChromaDB_Library.situate_context", return_value="Context for chunk")

def test_process_and_store_content(mock_chunk_for_embedding, mock_process_chunks, mock_situate_context, mock_create_embeddings_batch, mock_collection):
    process_and_store_content(
        database="mock_db", content="Test Content", collection_name="test_collection", media_id=1,
        file_name="test.mp4", create_embeddings=True, create_contextualized=True
    )

    mock_process_chunks.assert_called_once_with("mock_db", mock_chunk_for_embedding.return_value, 1)
    mock_collection.upsert.assert_called_once()


##############################
# Test: check_embedding_status
##############################

def test_check_embedding_status(mock_collection):
    mock_collection.get.return_value = {
        'ids': ['doc_1'],
        'embeddings': [[0.1, 0.2, 0.3, 0.4]],
        'metadatas': [{'key': 'value'}]
    }

    status, details = check_embedding_status("Test Item", {"Test Item": 1})

    assert "Embedding exists" in status
    assert "Metadata: {'key': 'value'}" in details


##############################
# Test: reset_chroma_collection
##############################

def test_reset_chroma_collection(mock_chroma_client):
    reset_chroma_collection("test_collection")

    mock_chroma_client.delete_collection.assert_called_once_with("test_collection")
    mock_chroma_client.create_collection.assert_called_once_with("test_collection")


##############################
# Test: store_in_chroma
##############################

def test_store_in_chroma(mock_collection):
    store_in_chroma(
        collection_name="test_collection",
        texts=["Text 1", "Text 2"],
        embeddings=[[0.1, 0.2], [0.3, 0.4]],
        ids=["id1", "id2"],
        metadatas=[{"key1": "value1"}, {"key2": "value2"}]
    )

    mock_collection.upsert.assert_called_once_with(
        documents=["Text 1", "Text 2"],
        embeddings=[[0.1, 0.2], [0.3, 0.4]],
        ids=["id1", "id2"],
        metadatas=[{"key1": "value1"}, {"key2": "value2"}]
    )


##############################
# Test: vector_search
##############################

@pytest.fixture
def mock_create_embedding(mocker):
    """Fixture to mock create_embedding."""
    return mocker.patch("App_Function_Libraries.RAG.ChromaDB_Library.create_embedding", return_value=[0.1, 0.2])

def test_vector_search(mock_create_embedding, mock_collection):
    mock_collection.query.return_value = {
        'documents': [["Document 1", "Document 2"]],
        'metadatas': [["Metadata 1", "Metadata 2"]]
    }

    results = vector_search("test_collection", "query text")

    assert len(results) == 2
    assert results[0]["content"] == "Document 1"
    assert results[0]["metadata"] == "Metadata 1"


##############################
# Parametrized Test: batched
##############################

@pytest.mark.parametrize("iterable, batch_size, expected_batches", [
    ([1, 2, 3, 4, 5], 2, [[1, 2], [3, 4], [5]]),
    ([1, 2], 3, [[1, 2]]),
    ([], 2, [])
])
def test_batched(iterable, batch_size, expected_batches):
    batches = list(batched(iterable, batch_size))
    assert batches == expected_batches


##############################
# Test: situate_context
##############################

# def test_situate_context(mock_situate_context):
#     result = situate_context(api_name="gpt-3.5-turbo", doc_content="Document", chunk_content="Chunk")
#     assert result == "Context for chunk"


##############################
# Test: schedule_embedding
##############################

# def test_schedule_embedding(mock_chunk_for_embedding, mock_create_embeddings_batch, mock_collection):
#     schedule_embedding(media_id=1, content="Test Content", media_name="test.mp4")
#
#     mock_collection.upsert.assert_called_once()

#
# End of File
####################################################################################################
# Force git to recognize this file.
