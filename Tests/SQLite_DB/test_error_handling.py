# tests/test_error_handling.py
# Tests error handling in SQLite database.
#
# Imports:
import pytest
from App_Function_Libraries.DB.DB_Manager import add_media_to_database, get_media_content
from App_Function_Libraries.DB.SQLite_DB import DatabaseError
#
####################################################################################################
# Test Status:
# FIXME


def test_add_media_with_invalid_data(empty_db):
    with pytest.raises(ValueError):
        add_media_to_database(None, None, None, None, None, None, None)


def test_get_nonexistent_media_content(empty_db):
    with pytest.raises(DatabaseError):
        get_media_content(9999)  # Assuming 9999 is an invalid media_id


def test_add_duplicate_media(empty_db):
    info_dict = {'title': 'Test Video', 'uploader': 'Test Uploader'}
    url = 'https://example.com/test'

    # Add media for the first time
    add_media_to_database(url, info_dict, [], 'Test summary', ['test'], 'Test prompt', 'whisper-1')

    # Try to add the same media again
    with pytest.raises(DatabaseError):
        add_media_to_database(url, info_dict, [], 'Test summary', ['test'], 'Test prompt', 'whisper-1')