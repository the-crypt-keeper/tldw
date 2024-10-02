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

@pytest.fixture(scope="module")
def db():
    # This fixture creates a single database for all tests
    database = Database('test.db')
    yield database
    database.close_connection()

    # Clean up the database file after all tests
    db_path = Utils.get_database_path('test.db')
    if os.path.exists(db_path):
        os.remove(db_path)


@pytest.fixture(autouse=True)
def reset_db(db):
    # This fixture runs automatically before each test
    yield
    # After each test, drop all tables to reset the database state
    with db.get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
        tables = cursor.fetchall()
        for table in tables:
            if table[0] != 'sqlite_sequence':
                cursor.execute(f"DROP TABLE IF EXISTS {table[0]}")
        conn.commit()


def test_database_connection(db):
    with db.get_connection() as conn:
        assert conn is not None


def test_execute_query(db):
    db.execute_query("CREATE TABLE test (id INTEGER PRIMARY KEY, name TEXT)")
    db.execute_query("INSERT INTO test (name) VALUES (?)", ('Test Name',))
    result = db.execute_query("SELECT name FROM test WHERE id = 1")
    assert result[0][0] == 'Test Name'


def test_database_error(db):
    with pytest.raises(sqlite3.OperationalError):
        db.execute_query("SELECT * FROM non_existent_table")


def test_transaction(db):
    with db.transaction() as conn:
        conn.execute("CREATE TABLE test (id INTEGER PRIMARY KEY, name TEXT)")
        conn.execute("INSERT INTO test (name) VALUES (?)", ('Test Name',))

    result = db.execute_query("SELECT name FROM test WHERE id = 1")
    assert result[0][0] == 'Test Name'


def test_transaction_rollback(db):
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


def test_execute_many(db):
    db.execute_query("CREATE TABLE test (id INTEGER PRIMARY KEY, name TEXT)")
    data = [('Name1',), ('Name2',), ('Name3',)]
    db.execute_many("INSERT INTO test (name) VALUES (?)", data)
    result = db.execute_query("SELECT COUNT(*) FROM test")
    assert result[0][0] == 3


def test_table_exists(db):
    db.execute_query("CREATE TABLE test (id INTEGER PRIMARY KEY, name TEXT)")
    assert db.table_exists('test') == True
    assert db.table_exists('non_existent_table') == False


def test_close_connection(db):
    with db.get_connection() as conn:
        assert conn is not None
    db.close_connection()
    assert not hasattr(db._local, 'connection') or db._local.connection is None


def test_create_tables(db):
    create_tables(db)

    # Check if some of the tables were created
    assert db.table_exists('Media')
    assert db.table_exists('Keywords')
    assert db.table_exists('MediaKeywords')
    assert db.table_exists('ChatConversations')


def test_multiple_connections(db):
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
