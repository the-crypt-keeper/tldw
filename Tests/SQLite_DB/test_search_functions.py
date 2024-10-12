# tests/test_search_functions.py
import pytest
import sqlite3
from unittest.mock import patch, MagicMock
from typing import List, Tuple
#
# Updated import statement
from App_Function_Libraries.DB.DB_Manager import sqlite_search_db, search_media_database, db
#
#
####################################################################################################
# Test Status:
# Working as of 2024-10-01

import pytest
from unittest.mock import patch, MagicMock
import sqlite3
from contextlib import contextmanager

from App_Function_Libraries.DB.DB_Manager import sqlite_search_db, search_media_database, Database

# Modify the functions to accept a connection parameter for testing
def sqlite_search_db_testable(search_query: str, search_fields: List[str], keywords: str, page: int = 1, results_per_page: int = 10, connection=None):
    if connection is None:
        with db.get_connection() as conn:
            return sqlite_search_db(search_query, search_fields, keywords, page, results_per_page)
    else:
        # Use the provided connection for testing
        return sqlite_search_db(search_query, search_fields, keywords, page, results_per_page, connection=connection)

def search_media_database_testable(query: str, connection=None):
    if connection is None:
        with db.get_connection() as conn:
            return search_media_database(query)
    else:
        # Use the provided connection for testing
        return search_media_database(query, connection=connection)

@pytest.fixture
def mock_connection():
    mock_conn = MagicMock()
    mock_cursor = MagicMock()
    mock_conn.cursor.return_value = mock_cursor
    return mock_conn, mock_cursor

def test_sqlite_search_db(mock_connection):
    mock_conn, mock_cursor = mock_connection
    mock_cursor.fetchall.return_value = [
        (1, 'http://example.com', 'Test Title', 'video', 'content', 'author', '2023-01-01', 'prompt', 'summary')
    ]

    results = sqlite_search_db_testable('Test', ['title'], '', page=1, results_per_page=10, connection=mock_conn)

    assert len(results) == 1
    assert results[0][2] == 'Test Title'
    mock_cursor.execute.assert_called()
    call_args = mock_cursor.execute.call_args[0]
    assert 'SELECT DISTINCT Media.id, Media.url, Media.title' in call_args[0]
    assert 'WHERE Media.title LIKE ?' in call_args[0]
    assert '%Test%' in call_args[1]

def test_sqlite_search_db_with_keywords(mock_connection):
    mock_conn, mock_cursor = mock_connection
    mock_cursor.fetchall.return_value = [
        (1, 'http://example.com', 'Test Title', 'video', 'content', 'author', '2023-01-01', 'prompt', 'summary')
    ]

    results = sqlite_search_db_testable('Test', ['title'], 'keyword1,keyword2', page=1, results_per_page=10, connection=mock_conn)

    assert len(results) == 1
    mock_cursor.execute.assert_called()
    call_args = mock_cursor.execute.call_args[0]
    assert 'EXISTS (SELECT 1 FROM MediaKeywords mk JOIN Keywords k ON mk.keyword_id = k.id WHERE mk.media_id = Media.id AND k.keyword LIKE ?)' in call_args[0]
    assert '%keyword1%' in call_args[1]
    assert '%keyword2%' in call_args[1]

def test_sqlite_search_db_pagination(mock_connection):
    mock_conn, mock_cursor = mock_connection
    mock_cursor.fetchall.return_value = [
        (2, 'http://example2.com', 'Second Title', 'article', 'content2', 'author2', '2023-01-02', 'prompt2', 'summary2')
    ]

    results = sqlite_search_db_testable('', ['title'], '', page=2, results_per_page=1, connection=mock_conn)

    assert len(results) == 1
    assert results[0][2] == 'Second Title'
    mock_cursor.execute.assert_called()
    call_args = mock_cursor.execute.call_args[0]
    assert 'LIMIT ? OFFSET ?' in call_args[0]
    assert call_args[1][-2:] == [1, 1]  # LIMIT 1 OFFSET 1

def test_sqlite_search_db_invalid_page():
    with pytest.raises(ValueError, match="Page number must be 1 or greater."):
        sqlite_search_db_testable('Test', ['title'], '', page=0, results_per_page=10)


def test_search_media_database(mock_connection):
    mock_conn, mock_cursor = mock_connection
    mock_cursor.fetchall.return_value = [
        (1, 'Test Title', 'http://example.com')
    ]

    results = search_media_database('Test', mock_conn)

    assert len(results) == 1
    assert results[0] == (1, 'Test Title', 'http://example.com')
    mock_cursor.execute.assert_called_with(
        "SELECT id, title, url FROM Media WHERE title LIKE ?",
        ('%Test%',)
    )


def test_search_media_database_error(mock_connection):
    mock_conn, mock_cursor = mock_connection
    mock_cursor.execute.side_effect = sqlite3.Error("Test database error")

    with pytest.raises(Exception) as exc_info:
        search_media_database('Test', connection=mock_conn)

    assert str(exc_info.value) == "Error searching media database: Test database error"

#
# End of File
####################################################################################################