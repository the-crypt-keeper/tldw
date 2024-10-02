# tests/test_sqlite_db.py
import os
import sqlite3
import tempfile

import pytest
from App_Function_Libraries.DB.SQLite_DB import DatabaseError, create_tables, Database
from App_Function_Libraries.Utils import Utils
#
####################################################################################################
# Test Status:
# FIXME

@pytest.fixture
def db_factory():
    temp_files = []

    def create_db():
        # Create a temporary file
        temp_db = tempfile.NamedTemporaryFile(delete=False, suffix='.db')
        temp_db.close()
        temp_files.append(temp_db.name)

        # Mock the get_database_path function
        def mock_get_database_path(db_name):
            return temp_db.name

        # Create and return the database
        with pytest.MonkeyPatch.context() as m:
            m.setattr(Utils, 'get_database_path', mock_get_database_path)
            return Database('test.db')

    yield create_db

    # Cleanup: close connections and delete temporary files
    for file in temp_files:
        try:
            os.unlink(file)
        except Exception as e:
            print(f"Error deleting {file}: {e}")


def test_database_connection(db_factory):
    db = db_factory()
    with db.get_connection() as conn:
        assert conn is not None


def test_execute_query(db_factory):
    db = db_factory()
    db.execute_query("CREATE TABLE test (id INTEGER PRIMARY KEY, name TEXT)")
    db.execute_query("INSERT INTO test (name) VALUES (?)", ('Test Name',))
    result = db.execute_query("SELECT name FROM test WHERE id = 1")
    assert result[0][0] == 'Test Name'


def test_database_error(db_factory):
    db = db_factory()
    with pytest.raises(sqlite3.OperationalError):
        db.execute_query("SELECT * FROM non_existent_table")


def test_transaction(db_factory):
    db = db_factory()
    with db.transaction() as conn:
        conn.execute("CREATE TABLE test (id INTEGER PRIMARY KEY, name TEXT)")
        conn.execute("INSERT INTO test (name) VALUES (?)", ('Test Name',))

    result = db.execute_query("SELECT name FROM test WHERE id = 1")
    assert result[0][0] == 'Test Name'


def test_transaction_rollback(db_factory):
    db = db_factory()
    try:
        with db.transaction() as conn:
            conn.execute("CREATE TABLE test (id INTEGER PRIMARY KEY, name TEXT)")
            conn.execute("INSERT INTO test (name) VALUES (?)", ('Test Name',))
            raise Exception("Simulated error")
    except Exception:
        pass

    # The table should not exist after rollback
    with pytest.raises(sqlite3.OperationalError):
        db.execute_query("SELECT * FROM test")


def test_execute_many(db_factory):
    db = db_factory()
    db.execute_query("CREATE TABLE test (id INTEGER PRIMARY KEY, name TEXT)")
    data = [('Name1',), ('Name2',), ('Name3',)]
    db.execute_many("INSERT INTO test (name) VALUES (?)", data)
    result = db.execute_query("SELECT COUNT(*) FROM test")
    assert result[0][0] == 3


def test_table_exists(db_factory):
    db = db_factory()
    db.execute_query("CREATE TABLE test (id INTEGER PRIMARY KEY, name TEXT)")
    assert db.table_exists('test') == True
    assert db.table_exists('non_existent_table') == False


def test_close_connection(db_factory):
    db = db_factory()
    with db.get_connection() as conn:
        assert conn is not None
    db.close_connection()
    assert not hasattr(db._local, 'connection') or db._local.connection is None


def test_create_tables(db_factory):
    db = db_factory()
    create_tables(db)

    # Check if some of the tables were created
    assert db.table_exists('Media')
    assert db.table_exists('Keywords')
    assert db.table_exists('MediaKeywords')
    assert db.table_exists('ChatConversations')


def test_multiple_connections(db_factory):
    db = db_factory()

    def worker():
        with db.get_connection() as conn:
            conn.execute("SELECT 1")

    import threading
    threads = [threading.Thread(target=worker) for _ in range(5)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    # If this test completes without errors, it means multiple threads could use the database simultaneously


