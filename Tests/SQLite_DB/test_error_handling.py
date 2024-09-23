# tests/test_error_handling.py
import pytest
import sqlite3
from App_Function_Libraries.DB.SQLite_DB import Database, DatabaseError, InputError, add_keyword, delete_keyword, add_media_with_keywords, sqlite_search_db


@pytest.fixture
def test_db():
    return Database(':memory:')


def test_execute_query_with_invalid_sql(test_db):
    with pytest.raises(sqlite3.Error):
        test_db.execute_query("INSERT INTO nonexistent_table (column) VALUES (?)", (1,))


def test_execute_many_with_invalid_data(test_db):
    with pytest.raises(sqlite3.Error):
        test_db.execute_many("INSERT INTO nonexistent_table (column) VALUES (?)", [(1,), (2,)])


def test_table_exists_nonexistent(test_db):
    assert not test_db.table_exists("nonexistent_table")

def test_transaction_rollback(test_db):
    test_db.execute_query("CREATE TABLE test (id INTEGER PRIMARY KEY)")
    with pytest.raises(sqlite3.OperationalError):
        with test_db.transaction() as conn:
            conn.execute("INSERT INTO test (id) VALUES (?)", (1,))
            conn.execute("INSERT INTO nonexistent_table (id) VALUES (?)", (1,))


def test_add_keyword_with_invalid_data():
    with pytest.raises(DatabaseError):
        add_keyword(None)

def test_delete_nonexistent_keyword():
    result = delete_keyword("nonexistent_keyword")
    assert "not found" in result

def test_add_media_with_invalid_data():
    with pytest.raises(InputError):
        add_media_with_keywords(None, None, None, None, None, None, None, None, None, None)

def test_invalid_search_query():
    with pytest.raises(InputError):
        sqlite_search_db(None, ['title'], '', 1, 10)
