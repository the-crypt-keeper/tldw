# tests/test_error_handling.py
import pytest
from App_Function_Libraries.DB.SQLite_DB import Database, DatabaseError, InputError

@pytest.fixture
def test_db():
    return Database(':memory:')

def test_add_media_with_invalid_data(test_db):
    with pytest.raises(InputError):
        test_db.add_media_to_database(None, None, None, None, None, None, None)

def test_get_nonexistent_media_content(test_db):
    with pytest.raises(DatabaseError):
        test_db.get_media_content(9999)

def test_add_duplicate_media(test_db):
    info_dict = {'title': 'Test Video', 'uploader': 'Test Uploader'}
    url = 'https://example.com/test'
    content = "Test content"
    summary = 'Test summary'
    keywords = ['test']
    prompt = 'Test prompt'
    whisper_model = 'whisper-1'

    # Add media for the first time
    test_db.add_media_to_database(url, info_dict, content, summary, keywords, prompt, whisper_model)

    # Try to add the same media again
    with pytest.raises(DatabaseError):
        test_db.add_media_to_database(url, info_dict, content, summary, keywords, prompt, whisper_model)

def test_invalid_keyword_operation(test_db):
    with pytest.raises(DatabaseError):
        test_db.add_keyword(None)

def test_invalid_search_query(test_db):
    with pytest.raises(InputError):
        test_db.search_db(None, ['title'], '', 1, 10)

def test_invalid_document_version(test_db):
    with pytest.raises(ValueError):
        test_db.create_document_version(9999, "Test content")

def test_invalid_chat_operation(test_db):
    with pytest.raises(DatabaseError):
        test_db.create_chat_conversation(None, None)

def test_invalid_transcript_operation(test_db):
    with pytest.raises(DatabaseError):
        test_db.get_media_transcripts(None)

def test_invalid_media_update(test_db):
    with pytest.raises(DatabaseError):
        test_db.update_media_content(None, None, None, None, None)

def test_invalid_prompt_operation(test_db):
    with pytest.raises(DatabaseError):
        test_db.add_prompt(None, None, None)

def test_invalid_media_chunk_operation(test_db):
    with pytest.raises(DatabaseError):
        test_db.process_chunks(None, None, None)

def test_invalid_trash_operation(test_db):
    with pytest.raises(DatabaseError):
        test_db.mark_as_trash(None)

def test_invalid_workflow_operation(test_db):
    with pytest.raises(DatabaseError):
        test_db.save_workflow_chat_to_db(None, None)
