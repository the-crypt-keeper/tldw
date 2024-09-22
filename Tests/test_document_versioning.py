import pytest
import os
from App_Function_Libraries.DB.DB_Manager import create_document_version, get_document_version
from App_Function_Libraries.DB.SQLite_DB import Database

@pytest.fixture
def sample_media(empty_db):
    """Create a media entry for each test in a fresh database."""
    db = empty_db

    # Insert a single media entry for each test
    with db.get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute('''
            INSERT INTO Media (url, title, type, content, author, ingestion_date, transcription_model)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        ''', ("http://example.com", "Sample Title", "document", "Sample content", "Author", "2024-01-01", "whisper_model"))
        media_id = cursor.lastrowid

    return media_id

@pytest.fixture
def sample_document(sample_media):
    """Create the first version of a document for a given media."""
    return create_document_version(sample_media, "Initial content")

def test_create_document_version(sample_media):
    """Test creating a new version of a document."""
    version = create_document_version(sample_media, "Test content")
    assert version == 1  # Should return version 1 if this is the first version

def test_get_document_version(sample_document):
    """Test fetching the latest version of a document."""
    version_data = get_document_version(sample_document)
    assert version_data['version_number'] == 1
    assert version_data['content'] == "Initial content"

def test_create_multiple_versions(sample_media):
    """Test creating multiple versions for the same document."""
    v1 = create_document_version(sample_media, "Version 1 content")
    v2 = create_document_version(sample_media, "Version 2 content")

    assert v1 == 1  # First version should be 1
    assert v2 == 2  # Second version should be 2

    v2_data = get_document_version(sample_media, 2)
    assert v2_data['content'] == "Version 2 content"