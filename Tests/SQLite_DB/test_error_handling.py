# tests/test_error_handling.py
import pytest
import os
import sqlite3
from App_Function_Libraries.DB.SQLite_DB import Database, DatabaseError, InputError, add_keyword, delete_keyword, add_media_with_keywords, sqlite_search_db


@pytest.fixture
def test_db(tmp_path):
    db_file = tmp_path / "test.db"
    db = Database(str(db_file))
    yield db
    db.close_connection()  # Ensure the connection is closed
    try:
        os.remove(db_file)
    except PermissionError:
        print(f"Warning: Unable to delete {db_file}. It may still be in use.")


def test_execute_query_with_invalid_sql(test_db):
    with pytest.raises(sqlite3.Error):
        test_db.execute_query("INSERT INTO nonexistent_table (column) VALUES (?)", (1,))


def test_execute_many_with_invalid_data(test_db):
    with pytest.raises(sqlite3.Error):
        test_db.execute_many("INSERT INTO nonexistent_table (column) VALUES (?)", [(1,), (2,)])


def test_table_exists_nonexistent(test_db):
    assert not test_db.table_exists("nonexistent_table")


def test_transaction_rollback(test_db):
    with test_db.get_connection() as conn:
        conn.execute("CREATE TABLE test (id INTEGER PRIMARY KEY)")

    with pytest.raises(sqlite3.OperationalError):
        with test_db.transaction() as conn:
            conn.execute("INSERT INTO test (id) VALUES (?)", (1,))
            conn.execute("INSERT INTO nonexistent_table (id) VALUES (?)", (1,))

    # Verify the transaction was rolled back
    result = test_db.execute_query("SELECT COUNT(*) FROM test")
    assert result[0][0] == 0


def test_add_keyword_with_invalid_data():
    with pytest.raises(AttributeError):
        add_keyword(None)

def test_delete_nonexistent_keyword():
    result = delete_keyword("nonexistent_keyword")
    assert "not found" in result

def test_add_media_with_invalid_data():
    with pytest.raises(InputError):
        add_media_with_keywords(None, None, None, None, None, None, None, None, None, None)

def test_sqlite_search_db_with_invalid_data():
    # Test with invalid page number
    with pytest.raises(ValueError):
        sqlite_search_db("query", ['title'], "", 0, 10)

    # Test with None values (should not raise an exception)
    result = sqlite_search_db(None, ['title'], "", 1, 10)
    assert isinstance(result, list)

    # Test with empty search fields
    result = sqlite_search_db("query", [], "", 1, 10)
    assert isinstance(result, list)

    # Test with invalid search fields
    with pytest.raises(sqlite3.OperationalError):
        sqlite_search_db("query", ['nonexistent_field'], "", 1, 10)
