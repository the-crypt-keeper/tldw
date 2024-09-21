# tests/conftest.py
import pytest
import os
import tempfile
from App_Function_Libraries.DB.SQLite_DB import Database

@pytest.fixture(scope="session")
def test_db():
    # Create a temporary database for testing
    _, db_path = tempfile.mkstemp(suffix='.db')
    db = Database(db_path)
    yield db
    # Clean up the temporary database after tests
    os.unlink(db_path)

@pytest.fixture(scope="function")
def empty_db(test_db):
    # Ensure the database is empty before each test
    with test_db.get_connection() as conn:
        cursor = conn.cursor()
        cursor.executescript('''
            DROP TABLE IF EXISTS Media;
            DROP TABLE IF EXISTS Keywords;
            DROP TABLE IF EXISTS MediaKeywords;
            DROP TABLE IF EXISTS MediaVersion;
            DROP TABLE IF EXISTS MediaModifications;
            DROP TABLE IF EXISTS ChatConversations;
            DROP TABLE IF EXISTS ChatMessages;
            DROP TABLE IF EXISTS Transcripts;
            DROP TABLE IF EXISTS MediaChunks;
            DROP TABLE IF EXISTS media_fts;
            DROP TABLE IF EXISTS keyword_fts;
        ''')
    return test_db