# tests/test_search_functions.py
import pytest
from App_Function_Libraries.DB.DB_Manager import search_db, search_media_database, add_media_to_database
#
####################################################################################################
# Test Status:
# FIXME

@pytest.fixture
def sample_data(empty_db):
    info_dict = {'title': 'Test Video', 'uploader': 'Test Uploader'}
    add_media_to_database('https://example.com/test1', info_dict, [], 'Test summary 1', ['keyword1'], 'Test prompt 1', 'whisper-1')
    add_media_to_database('https://example.com/test2', info_dict, [], 'Test summary 2', ['keyword2'], 'Test prompt 2', 'whisper-1')

def test_search_db(empty_db, sample_data):
    results = search_db('Test', ['title'], 'keyword1')
    assert len(results) == 1
    assert results[0]['title'] == 'Test Video'

def test_search_media_database(empty_db, sample_data):
    results = search_media_database('Test')
    assert len(results) == 2
    assert all('Test Video' in result[1] for result in results)