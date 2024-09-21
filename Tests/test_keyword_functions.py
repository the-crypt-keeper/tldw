# tests/test_keyword_functions.py
import pytest
from App_Function_Libraries.DB.DB_Manager import add_keyword, delete_keyword, fetch_keywords_for_media, update_keywords_for_media

@pytest.fixture
def sample_keywords(empty_db):
    add_keyword('test1')
    add_keyword('test2')
    return ['test1', 'test2']

def test_add_keyword(empty_db):
    keyword_id = add_keyword('testkeyword')
    assert keyword_id is not None

def test_delete_keyword(empty_db, sample_keywords):
    result = delete_keyword('test1')
    assert "deleted successfully" in result
    result = delete_keyword('nonexistent')
    assert "not found" in result

def test_fetch_keywords_for_media(empty_db, sample_media):
    keywords = fetch_keywords_for_media(sample_media)
    assert 'test' in keywords

def test_update_keywords_for_media(empty_db, sample_media):
    new_keywords = ['updated1', 'updated2']
    update_keywords_for_media(sample_media, new_keywords)
    updated_keywords = fetch_keywords_for_media(sample_media)
    assert set(new_keywords) == set(updated_keywords)