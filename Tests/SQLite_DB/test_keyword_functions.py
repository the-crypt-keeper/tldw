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


@pytest.fixture
def sample_keywords(test_db):
    keywords = ['test1', 'test2', 'test3']
    for keyword in keywords:
        keyword_id = add_keyword(keyword)
        logging.debug(f"Added keyword '{keyword}' with ID: {keyword_id}")
    return keywords


@pytest.fixture
def sample_media(test_db):
    with test_db.get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("INSERT INTO Media (title) VALUES (?)", ("Test Media",))
        media_id = cursor.lastrowid
    logging.debug(f"Added sample media with ID: {media_id}")
    return media_id


def test_add_keyword(test_db):
    keyword_id = add_keyword('testkeyword')
    logging.debug(f"Added keyword 'testkeyword' with ID: {keyword_id}")
    assert keyword_id is not None
    with test_db.get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT keyword FROM Keywords WHERE id = ?", (keyword_id,))
        result = cursor.fetchone()
    logging.debug(f"Retrieved keyword: {result}")
    assert result is not None
    assert result[0] == 'testkeyword'


def test_delete_keyword(test_db, sample_keywords):
    result = delete_keyword('test1')
    logging.debug(f"Delete keyword result: {result}")
    assert "successfully" in result
    with test_db.get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT keyword FROM Keywords WHERE keyword = ?", ('test1',))
        result = cursor.fetchone()
    logging.debug(f"Retrieved keyword after deletion: {result}")
    assert result is None


def test_fetch_keywords_for_media(test_db, sample_media, sample_keywords):
    # Associate keywords with media
    with test_db.get_connection() as conn:
        cursor = conn.cursor()
        for keyword in sample_keywords:
            cursor.execute("SELECT id FROM Keywords WHERE keyword = ?", (keyword,))
            result = cursor.fetchone()
            logging.debug(f"Fetched keyword ID for '{keyword}': {result}")
            if result:
                keyword_id = result[0]
                cursor.execute("INSERT INTO MediaKeywords (media_id, keyword_id) VALUES (?, ?)",
                               (sample_media, keyword_id))
                logging.debug(f"Associated keyword '{keyword}' with media ID {sample_media}")

    fetched_keywords = fetch_keywords_for_media(sample_media)
    logging.debug(f"Fetched keywords for media: {fetched_keywords}")
    assert set(fetched_keywords) == set(sample_keywords)


def test_update_keywords_for_media(test_db, sample_media):
    new_keywords = ['new1', 'new2', 'new3']
    update_keywords_for_media(sample_media, new_keywords)
    fetched_keywords = fetch_keywords_for_media(sample_media)
    logging.debug(f"Updated keywords for media: {fetched_keywords}")
    assert set(fetched_keywords) == set(new_keywords)
