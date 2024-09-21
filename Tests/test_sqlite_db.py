# tests/test_sqlite_db.py
import pytest
from App_Function_Libraries.DB.SQLite_DB import DatabaseError


def test_database_connection(empty_db):
    with empty_db.get_connection() as conn:
        assert conn is not None


def test_execute_query(empty_db):
    empty_db.execute_query("CREATE TABLE test (id INTEGER PRIMARY KEY, name TEXT)")
    empty_db.execute_query("INSERT INTO test (name) VALUES (?)", ('Test Name',))
    result = empty_db.execute_query("SELECT name FROM test WHERE id = 1")
    assert result[0][0] == 'Test Name'


def test_database_error(empty_db):
    with pytest.raises(DatabaseError):
        empty_db.execute_query("SELECT * FROM non_existent_table")


def test_transaction(empty_db):
    with empty_db.transaction() as conn:
        conn.execute("CREATE TABLE test (id INTEGER PRIMARY KEY, name TEXT)")
        conn.execute("INSERT INTO test (name) VALUES (?)", ('Test Name',))

    result = empty_db.execute_query("SELECT name FROM test WHERE id = 1")
    assert result[0][0] == 'Test Name'


def test_transaction_rollback(empty_db):
    try:
        with empty_db.transaction() as conn:
            conn.execute("CREATE TABLE test (id INTEGER PRIMARY KEY, name TEXT)")
            conn.execute("INSERT INTO test (name) VALUES (?)", ('Test Name',))
            raise Exception("Simulated error")
    except Exception:
        pass

    with pytest.raises(DatabaseError):
        empty_db.execute_query("SELECT * FROM test")