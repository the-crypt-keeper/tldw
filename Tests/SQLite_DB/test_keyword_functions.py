# tests/test_keyword_functions.py
import logging

import pytest
from App_Function_Libraries.DB.DB_Manager import add_keyword, delete_keyword, fetch_keywords_for_media, update_keywords_for_media
#
####################################################################################################
# Test Status:
# FIXME

import pytest
from App_Function_Libraries.DB.SQLite_DB import Database, add_keyword, delete_keyword, fetch_keywords_for_media, \
    update_keywords_for_media


logging.basicConfig(level=logging.DEBUG)


@pytest.fixture
def test_db(tmp_path):
    db_file = tmp_path / "test.db"
    db = Database(str(db_file))
    with db.get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS Keywords (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                keyword TEXT UNIQUE NOT NULL
            )
        ''')
        cursor.execute('''
            CREATE VIRTUAL TABLE IF NOT EXISTS keyword_fts USING fts5(keyword)
        ''')
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS Media (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                title TEXT NOT NULL
            )
        ''')
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS MediaKeywords (
                media_id INTEGER,
                keyword_id INTEGER,
                FOREIGN KEY (media_id) REFERENCES Media(id),
                FOREIGN KEY (keyword_id) REFERENCES Keywords(id),
                PRIMARY KEY (media_id, keyword_id)
            )
        ''')
    yield db
    db.close_connection()


def test_add_keyword(test_db):
    try:
        keyword = 'testkeyword'
        keyword_id = add_keyword(keyword)
        print(f"Returned keyword_id: {keyword_id}")

        with test_db.get_connection() as conn:
            cursor = conn.cursor()

            # Check Keywords table
            cursor.execute("SELECT * FROM Keywords WHERE keyword = ?", (keyword,))
            keyword_result = cursor.fetchone()
            print(f"Keyword in Keywords table: {keyword_result}")

            # Check keyword_fts table
            cursor.execute("SELECT * FROM keyword_fts WHERE keyword = ?", (keyword,))
            fts_result = cursor.fetchone()
            print(f"Keyword in keyword_fts table: {fts_result}")

        assert keyword_id is not None
        assert keyword_result is not None
        assert fts_result is not None
        assert keyword_result[1] == keyword
        assert fts_result[0] == keyword

    except Exception as e:
        print(f"Exception details: {str(e)}")
        raise


def test_delete_keyword(test_db):
    add_keyword('test1')
    result = delete_keyword('test1')
    assert "successfully" in result
    with test_db.get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT keyword FROM Keywords WHERE keyword = ?", ('test1',))
        result = cursor.fetchone()
    assert result is None


def test_fetch_keywords_for_media(test_db):
    media_id = 1
    keywords = ['test1', 'test2', 'test3']

    # Add media
    with test_db.get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("INSERT INTO Media (id, title) VALUES (?, ?)", (media_id, "Test Media"))

    # Add keywords and associate with media
    for keyword in keywords:
        keyword_id = add_keyword(keyword)
        with test_db.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("INSERT INTO MediaKeywords (media_id, keyword_id) VALUES (?, ?)", (media_id, keyword_id))

    fetched_keywords = fetch_keywords_for_media(media_id)
    assert set(fetched_keywords) == set(keywords)


def test_update_keywords_for_media(test_db):
    media_id = 2
    initial_keywords = ['initial1', 'initial2']
    new_keywords = ['new1', 'new2', 'new3']

    # Add media
    with test_db.get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("INSERT INTO Media (id, title) VALUES (?, ?)", (media_id, "Test Media 2"))

    # Add initial keywords
    for keyword in initial_keywords:
        keyword_id = add_keyword(keyword)
        with test_db.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("INSERT INTO MediaKeywords (media_id, keyword_id) VALUES (?, ?)", (media_id, keyword_id))

    # Update keywords
    update_keywords_for_media(media_id, new_keywords)

    fetched_keywords = fetch_keywords_for_media(media_id)
    assert set(fetched_keywords) == set(new_keywords)