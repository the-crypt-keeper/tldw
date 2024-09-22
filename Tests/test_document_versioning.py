# test_document_versioning.py
# Tests document versioning in SQLite database.

import pytest
import os
from App_Function_Libraries.DB.SQLite_DB import Database, create_document_version, get_document_version, DatabaseError
import sqlite3
from datetime import datetime, timedelta
import time

@pytest.fixture(scope="function")
def db(tmp_path):
    """Create a temporary database file for testing."""
    db_file = tmp_path / "test_db.sqlite"
    database = Database(str(db_file))
    with database.get_connection() as conn:
        cursor = conn.cursor()
        cursor.executescript('''
            CREATE TABLE IF NOT EXISTS Media (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                url TEXT,
                title TEXT NOT NULL,
                type TEXT NOT NULL,
                content TEXT,
                author TEXT,
                ingestion_date TEXT,
                transcription_model TEXT
            );
            CREATE TABLE IF NOT EXISTS DocumentVersions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                media_id INTEGER NOT NULL,
                version_number INTEGER NOT NULL,
                content TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (media_id) REFERENCES Media(id)
            );
        ''')
    yield database
    database.close_connection()
    time.sleep(0.1)
    try:
        os.remove(db_file)
    except PermissionError:
        print(f"Warning: Unable to remove temporary database file: {db_file}")

@pytest.fixture
def sample_media(db):
    """Create a sample media entry for testing."""
    with db.get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute('''
            INSERT INTO Media (url, title, type, content, author, ingestion_date, transcription_model)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        ''', ("http://example.com", "Sample Title", "document", "Sample content", "Author", "2024-01-01", "whisper_model"))
        return cursor.lastrowid

def test_create_initial_document_version(db, sample_media):
    """Test creating the first version of a document."""
    initial_version = create_document_version(sample_media, "Initial content")
    assert isinstance(initial_version, int), f"Expected an integer version number, got {type(initial_version)}"

    stored_version = get_document_version(sample_media)
    assert stored_version['version_number'] == initial_version
    assert stored_version['content'] == "Initial content"

def test_create_multiple_versions(db, sample_media):
    """Test creating multiple versions of a document."""
    versions = [
        create_document_version(sample_media, f"Version {i} content")
        for i in range(1, 4)
    ]
    assert len(versions) == 3, f"Expected 3 versions, got {len(versions)}"
    assert versions[2] > versions[1] > versions[0], "Version numbers should be increasing"

def test_get_specific_document_version(db, sample_media):
    """Test retrieving a specific version of a document."""
    created_versions = []
    for i in range(1, 4):
        version = create_document_version(sample_media, f"Version {i} content")
        created_versions.append(version)

    for i, version_number in enumerate(created_versions, start=1):
        version = get_document_version(sample_media, version_number)
        assert version['content'] == f"Version {i} content"

def test_get_nonexistent_version(db, sample_media):
    """Test getting a version that doesn't exist."""
    create_document_version(sample_media, "Version 1 content")
    result = get_document_version(sample_media, 999999)
    assert 'error' in result

def test_create_version_for_nonexistent_media(db):
    """Test creating a version for a media that doesn't exist."""
    with pytest.raises(ValueError):
        create_document_version(999999, "Content")

def test_get_version_for_nonexistent_media(db):
    """Test getting a version for a media that doesn't exist."""
    result = get_document_version(999999)
    assert 'error' in result

def test_create_large_number_of_versions(db, sample_media):
    """Test creating a large number of versions."""
    num_versions = 100
    versions = []
    for i in range(1, num_versions + 1):
        version = create_document_version(sample_media, f"Version {i} content")
        versions.append(version)
    assert len(versions) == num_versions, f"Expected {num_versions} versions, got {len(versions)}"
    assert versions[-1] > versions[0], "Last version should be greater than first version"

def test_get_latest_version_after_multiple_creations(db, sample_media):
    """Test getting the latest version after creating multiple versions."""
    num_versions = 5
    latest_version = None
    for i in range(num_versions):
        latest_version = create_document_version(sample_media, f"Version {i + 1}")

    retrieved_version = get_document_version(sample_media)
    assert retrieved_version['version_number'] == latest_version
    assert retrieved_version['content'] == f"Version {num_versions}"

def test_create_version_with_empty_content(db, sample_media):
    """Test creating a version with empty content."""
    version = create_document_version(sample_media, "")
    assert isinstance(version, int), f"Expected an integer version number, got {type(version)}"

    stored_version = get_document_version(sample_media)
    assert stored_version['content'] == ""

def test_create_version_with_large_content(db, sample_media):
    """Test creating a version with large content."""
    large_content = "A" * 1000000  # 1 MB of content
    version = create_document_version(sample_media, large_content)
    assert isinstance(version, int), f"Expected an integer version number, got {type(version)}"

    stored_version = get_document_version(sample_media)
    assert len(stored_version['content']) == 1000000

def test_concurrent_version_creation(db, sample_media):
    """Test creating versions concurrently."""
    import threading

    def create_version(content):
        create_document_version(sample_media, content)

    threads = []
    for i in range(10):
        t = threading.Thread(target=create_version, args=(f"Concurrent Version {i}",))
        threads.append(t)
        t.start()

    for t in threads:
        t.join()

    latest_version = get_document_version(sample_media)
    assert latest_version['version_number'] == 10, "Expected 10 versions after concurrent creation"

def test_version_creation_performance(db, sample_media):
    """Test the performance of version creation."""
    import time

    start_time = time.time()
    num_versions = 1000
    for i in range(num_versions):
        create_document_version(sample_media, f"Performance test version {i}")
    end_time = time.time()

    total_time = end_time - start_time
    assert total_time < 10, f"Creating {num_versions} versions took more than 10 seconds"

# The following tests are commented out as they require functions that are not yet implemented

# def test_get_all_versions(db, sample_media):
#     """Test retrieving all versions of a document."""
#     num_versions = 5
#     for i in range(num_versions):
#         create_document_version(sample_media, f"Version {i + 1}")
#
#     # This test requires the implementation of get_all_document_versions
#     # all_versions = get_all_document_versions(sample_media)
#     # assert len(all_versions) == num_versions, f"Expected {num_versions} versions, got {len(all_versions)}"
#     # for i, version in enumerate(all_versions, start=1):
#     #     assert version['content'] == f"Version {i}"

# def test_delete_specific_version(db, sample_media):
#     """Test deleting a specific version of a document."""
#     for i in range(3):
#         create_document_version(sample_media, f"Version {i + 1}")
#
#     # This test requires the implementation of delete_document_version and get_all_document_versions
#     # delete_document_version(sample_media, 2)
#     #
#     # versions = get_all_document_versions(sample_media)
#     # assert len(versions) == 2, "Expected 2 versions after deletion"
#     # assert versions[0]['content'] == "Version 1"
#     # assert versions[1]['content'] == "Version 3"

# def test_rollback_to_previous_version(db, sample_media):
#     """Test rolling back to a previous version."""
#     for i in range(3):
#         create_document_version(sample_media, f"Version {i + 1}")
#
#     # This test requires the implementation of rollback_to_version
#     # rollback_to_version(sample_media, 2)
#     #
#     # latest_version = get_document_version(sample_media)
#     # assert latest_version['content'] == "Version 2"
#     # assert latest_version['version_number'] == 4  # New version created during rollback
