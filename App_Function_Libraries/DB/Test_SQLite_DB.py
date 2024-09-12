# Test_SQLite_DB.py
# Description:  Test file for SQLite_DB.py
#
# Usage: python -m unittest test_sqlite_db.py
#
# Imports
import unittest
import sqlite3
import threading
import time
from unittest.mock import patch
#
# Local Imports
from App_Function_Libraries.DB.SQLite_DB import Database, add_media_with_keywords, add_media_version, DatabaseError
#
#######################################################################################################################
#
# Functions:

class TestDatabase(unittest.TestCase):
    def setUp(self):
        self.db = Database(':memory:')  # Use in-memory database for testing

    def test_connection_management(self):
        with self.db.get_connection() as conn:
            self.assertIsInstance(conn, sqlite3.Connection)
        self.assertEqual(len(self.db.pool), 1)

    def test_execute_query(self):
        self.db.execute_query("CREATE TABLE test (id INTEGER PRIMARY KEY, name TEXT)")
        self.db.execute_query("INSERT INTO test (name) VALUES (?)", ("test_name",))
        with self.db.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT name FROM test")
            result = cursor.fetchone()
        self.assertEqual(result[0], "test_name")

    def test_execute_many(self):
        self.db.execute_query("CREATE TABLE test (id INTEGER PRIMARY KEY, name TEXT)")
        data = [("name1",), ("name2",), ("name3",)]
        self.db.execute_many("INSERT INTO test (name) VALUES (?)", data)
        with self.db.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM test")
            count = cursor.fetchone()[0]
        self.assertEqual(count, 3)

    def test_connection_retry(self):
        def lock_database():
            with self.db.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("BEGIN EXCLUSIVE TRANSACTION")
                time.sleep(2)  # Hold the lock for 2 seconds

        thread = threading.Thread(target=lock_database)
        thread.start()
        time.sleep(0.1)  # Give the thread time to acquire the lock

        with self.assertRaises(DatabaseError):
            self.db.execute_query("SELECT 1")  # This should retry and eventually fail

        thread.join()

class TestAddMediaWithKeywords(unittest.TestCase):
    def setUp(self):
        self.db = Database(':memory:')
        self.db.execute_query("""
            CREATE TABLE Media (
                id INTEGER PRIMARY KEY,
                url TEXT,
                title TEXT NOT NULL,
                type TEXT NOT NULL,
                content TEXT,
                author TEXT,
                ingestion_date TEXT,
                transcription_model TEXT
            )
        """)
        self.db.execute_query("CREATE TABLE Keywords (id INTEGER PRIMARY KEY, keyword TEXT NOT NULL UNIQUE)")
        self.db.execute_query("""
            CREATE TABLE MediaKeywords (
                id INTEGER PRIMARY KEY,
                media_id INTEGER NOT NULL,
                keyword_id INTEGER NOT NULL,
                FOREIGN KEY (media_id) REFERENCES Media(id),
                FOREIGN KEY (keyword_id) REFERENCES Keywords(id)
            )
        """)
        self.db.execute_query("""
            CREATE TABLE MediaModifications (
                id INTEGER PRIMARY KEY,
                media_id INTEGER NOT NULL,
                prompt TEXT,
                summary TEXT,
                modification_date TEXT,
                FOREIGN KEY (media_id) REFERENCES Media(id)
            )
        """)
        self.db.execute_query("""
            CREATE TABLE MediaVersion (
                id INTEGER PRIMARY KEY,
                media_id INTEGER NOT NULL,
                version INTEGER NOT NULL,
                prompt TEXT,
                summary TEXT,
                created_at TEXT NOT NULL,
                FOREIGN KEY (media_id) REFERENCES Media(id)
            )
        """)
        self.db.execute_query("CREATE VIRTUAL TABLE media_fts USING fts5(title, content)")

    @patch('App_Function_Libraries.DB.SQLite_DB.db', new_callable=lambda: Database(':memory:'))
    def test_add_new_media(self, mock_db):
        mock_db.get_connection = self.db.get_connection
        result = add_media_with_keywords(
            url="http://example.com",
            title="Test Title",
            media_type="article",
            content="Test content",
            keywords="test,keyword",
            prompt="Test prompt",
            summary="Test summary",
            transcription_model="Test model",
            author="Test Author",
            ingestion_date="2023-01-01"
        )
        self.assertIn("added/updated successfully", result)

        with self.db.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM Media")
            self.assertEqual(cursor.fetchone()[0], 1)

            cursor.execute("SELECT COUNT(*) FROM Keywords")
            self.assertEqual(cursor.fetchone()[0], 2)

            cursor.execute("SELECT COUNT(*) FROM MediaKeywords")
            self.assertEqual(cursor.fetchone()[0], 2)

            cursor.execute("SELECT COUNT(*) FROM MediaModifications")
            self.assertEqual(cursor.fetchone()[0], 1)

            cursor.execute("SELECT COUNT(*) FROM MediaVersion")
            self.assertEqual(cursor.fetchone()[0], 1)

    @patch('App_Function_Libraries.DB.SQLite_DB.db', new_callable=lambda: Database(':memory:'))
    def test_update_existing_media(self, mock_db):
        mock_db.get_connection = self.db.get_connection
        add_media_with_keywords(
            url="http://example.com",
            title="Test Title",
            media_type="article",
            content="Test content",
            keywords="test,keyword",
            prompt="Test prompt",
            summary="Test summary",
            transcription_model="Test model",
            author="Test Author",
            ingestion_date="2023-01-01"
        )

        result = add_media_with_keywords(
            url="http://example.com",
            title="Updated Title",
            media_type="article",
            content="Updated content",
            keywords="test,new",
            prompt="Updated prompt",
            summary="Updated summary",
            transcription_model="Updated model",
            author="Updated Author",
            ingestion_date="2023-01-02"
        )

        self.assertIn("added/updated successfully", result)

        with self.db.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM Media")
            self.assertEqual(cursor.fetchone()[0], 1)

            cursor.execute("SELECT title FROM Media")
            self.assertEqual(cursor.fetchone()[0], "Updated Title")

            cursor.execute("SELECT COUNT(*) FROM Keywords")
            self.assertEqual(cursor.fetchone()[0], 3)

            cursor.execute("SELECT COUNT(*) FROM MediaKeywords")
            self.assertEqual(cursor.fetchone()[0], 3)

            cursor.execute("SELECT COUNT(*) FROM MediaModifications")
            self.assertEqual(cursor.fetchone()[0], 2)

            cursor.execute("SELECT COUNT(*) FROM MediaVersion")
            self.assertEqual(cursor.fetchone()[0], 2)

if __name__ == '__main__':
    unittest.main()

#
# End of File
#######################################################################################################################
