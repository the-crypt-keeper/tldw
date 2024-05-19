import sqlite3
import gradio as gr
import pandas as pd
import logging
import os
import re
import time
from typing import List, Tuple, Union
from contextlib import contextmanager
from urllib.parse import urlparse
from datetime import datetime

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# Custom exceptions
class DatabaseError(Exception):
    pass


class InputError(Exception):
    pass


# Database connection function with connection pooling
class Database:
    def __init__(self, db_name=None):
        self.db_name = db_name or os.getenv('DB_NAME', 'media_summary.db')
        self.pool = []
        self.pool_size = 10

    @contextmanager
    def get_connection(self):
        retry_count = 5
        retry_delay = 1
        conn = None
        while retry_count > 0:
            try:
                conn = self.pool.pop() if self.pool else sqlite3.connect(self.db_name, check_same_thread=False)
                yield conn
                self.pool.append(conn)
                return
            except sqlite3.OperationalError as e:
                if 'database is locked' in str(e):
                    logger.warning(f"Database is locked, retrying in {retry_delay} seconds...")
                    retry_count -= 1
                    time.sleep(retry_delay)
                else:
                    raise DatabaseError(f"Database error: {e}")
            except Exception as e:
                raise DatabaseError(f"Unexpected error: {e}")
            finally:
                # Ensure the connection is returned to the pool even on failure
                if conn:
                    self.pool.append(conn)
        raise DatabaseError("Database is locked and retries have been exhausted")

    def execute_query(self, query: str, params: Tuple = ()) -> None:
        with self.get_connection() as conn:
            try:
                cursor = conn.cursor()
                cursor.execute(query, params)
                conn.commit()
            except sqlite3.Error as e:
                raise DatabaseError(f"Database error: {e}, Query: {query}")

db = Database()


# Function to create tables with the new media schema
def create_tables() -> None:
    table_queries = [
        '''
        CREATE TABLE IF NOT EXISTS Media (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            url TEXT,
            title TEXT NOT NULL,
            type TEXT NOT NULL,
            content TEXT,
            author TEXT,
            ingestion_date TEXT,
            prompt TEXT,
            summary TEXT,
            transcription_model TEXT
        )
        ''',
        '''
        CREATE TABLE IF NOT EXISTS Keywords (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            keyword TEXT NOT NULL UNIQUE
        )
        ''',
        '''
        CREATE TABLE IF NOT EXISTS MediaKeywords (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            media_id INTEGER NOT NULL,
            keyword_id INTEGER NOT NULL,
            FOREIGN KEY (media_id) REFERENCES Media(id),
            FOREIGN KEY (keyword_id) REFERENCES Keywords(id)
        )
        ''',
        '''
        CREATE TABLE IF NOT EXISTS MediaVersion (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            media_id INTEGER NOT NULL,
            version INTEGER NOT NULL,
            prompt TEXT,
            summary TEXT,
            created_at TEXT NOT NULL,
            FOREIGN KEY (media_id) REFERENCES Media(id)
        )
        ''',
        '''
        CREATE TABLE IF NOT EXISTS MediaModifications (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            media_id INTEGER NOT NULL,
            prompt TEXT,
            summary TEXT,
            modification_date TEXT,
            FOREIGN KEY (media_id) REFERENCES Media(id)
        )
        ''',
        '''
        CREATE VIRTUAL TABLE IF NOT EXISTS media_fts USING fts5(title, content);
        ''',
        '''
        CREATE VIRTUAL TABLE IF NOT EXISTS keyword_fts USING fts5(keyword);
        ''',
        '''
        CREATE INDEX IF NOT EXISTS idx_media_title ON Media(title);
        ''',
        '''
        CREATE INDEX IF NOT EXISTS idx_media_type ON Media(type);
        ''',
        '''
        CREATE INDEX IF NOT EXISTS idx_media_author ON Media(author);
        ''',
        '''
        CREATE INDEX IF NOT EXISTS idx_media_ingestion_date ON Media(ingestion_date);
        ''',
        '''
        CREATE INDEX IF NOT EXISTS idx_keywords_keyword ON Keywords(keyword);
        ''',
        '''
        CREATE INDEX IF NOT EXISTS idx_mediakeywords_media_id ON MediaKeywords(media_id);
        ''',
        '''
        CREATE INDEX IF NOT EXISTS idx_mediakeywords_keyword_id ON MediaKeywords(keyword_id);
        ''',
        '''
        CREATE INDEX IF NOT EXISTS idx_media_version_media_id ON MediaVersion(media_id);
        '''
    ]
    for query in table_queries:
        db.execute_query(query)

create_tables()


# Function to add a keyword
def add_keyword(keyword: str) -> str:
    keyword = keyword.strip().lower()
    with db.get_connection() as conn:
        cursor = conn.cursor()
        try:
            cursor.execute('INSERT OR IGNORE INTO Keywords (keyword) VALUES (?)', (keyword,))
            cursor.execute('SELECT id FROM Keywords WHERE keyword = ?', (keyword,))
            keyword_id = cursor.fetchone()[0]
            cursor.execute('INSERT OR IGNORE INTO keyword_fts (rowid, keyword) VALUES (?, ?)', (keyword_id, keyword))
            conn.commit()
            return f"Keyword '{keyword}' added successfully."
        except sqlite3.Error as e:
            logging.error(f"Error adding keyword: {e}")
            raise DatabaseError(f"Error adding keyword: {e}")


# Function to delete a keyword
def delete_keyword(keyword: str) -> str:
    keyword = keyword.strip().lower()
    with db.get_connection() as conn:
        cursor = conn.cursor()
        try:
            cursor.execute('SELECT id FROM Keywords WHERE keyword = ?', (keyword,))
            keyword_id = cursor.fetchone()
            if keyword_id:
                cursor.execute('DELETE FROM Keywords WHERE keyword = ?', (keyword,))
                cursor.execute('DELETE FROM keyword_fts WHERE rowid = ?', (keyword_id[0],))
                conn.commit()
                return f"Keyword '{keyword}' deleted successfully."
            else:
                return f"Keyword '{keyword}' not found."
        except sqlite3.Error as e:
            raise DatabaseError(f"Error deleting keyword: {e}")



# Function to add media with keywords
# Function to add media with keywords
def add_media_with_keywords(url: str, title: str, media_type: str, content: str, keywords: str, prompt: str, summary: str, transcription_model: str, author: str = None, ingestion_date: str = None) -> str:
    # Validate input
    if not title or not media_type or not content or not keywords or not prompt or not summary or not transcription_model:
        raise InputError("Please provide all required fields.")

    # Use 'localhost' as the URL if no valid URL is provided
    if not url or not is_valid_url(url):
        url = 'localhost'

    if media_type not in ['document', 'video', 'article']:
        raise InputError("Invalid media type. Allowed types: document, video, article.")

    if ingestion_date and not is_valid_date(ingestion_date):
        raise InputError("Invalid ingestion date format. Use YYYY-MM-DD.")

    if not ingestion_date:
        ingestion_date = datetime.now().strftime('%Y-%m-%d')

    logging.info(f"URL: {url}")
    logging.info(f"Title: {title}")
    logging.info(f"Media Type: {media_type}")
    logging.info(f"Content: {content}")
    logging.info(f"Keywords: {keywords}")
    logging.info(f"Prompt: {prompt}")
    logging.info(f"Summary: {summary}")
    logging.info(f"Author: {author}")
    logging.info(f"Ingestion Date: {ingestion_date}")
    logging.info(f"Transcription Model: {transcription_model}")

    try:
        with db.get_connection() as conn:
            cursor = conn.cursor()

            # Initialize keyword_list
            keyword_list = keywords.split(',')

            # Check if media already exists
            cursor.execute('SELECT id FROM Media WHERE url = ?', (url,))
            existing_media = cursor.fetchone()

            if existing_media:
                media_id = existing_media[0]
                logger.info(f"Existing media found with ID: {media_id}")

                # Insert new prompt and summary into MediaModifications
                cursor.execute('''
                INSERT INTO MediaModifications (media_id, prompt, summary, modification_date)
                VALUES (?, ?, ?, ?)
                ''', (media_id, prompt, summary, ingestion_date))
                logger.info("New summary and prompt added to MediaModifications")
            else:
                logger.info("New media entry being created")

                # Insert new media item
                cursor.execute('''
                INSERT INTO Media (url, title, type, content, author, ingestion_date, transcription_model) 
                VALUES (?, ?, ?, ?, ?, ?, ?)
                ''', (url, title, media_type, content, author, ingestion_date, transcription_model))
                media_id = cursor.lastrowid

                # Insert keywords and associate with media item
                keyword_list = keywords.split(',')
                for keyword in keyword_list:
                    keyword = keyword.strip().lower()
                    keyword_id = add_keyword(keyword)
                    cursor.execute('INSERT OR IGNORE INTO MediaKeywords (media_id, keyword_id) VALUES (?, ?)', (media_id, keyword_id))
                cursor.execute('INSERT INTO media_fts (rowid, title, content) VALUES (?, ?, ?)', (media_id, title, content))

                # Also insert the initial prompt and summary into MediaModifications
                cursor.execute('''
                INSERT INTO MediaModifications (media_id, prompt, summary, modification_date)
                VALUES (?, ?, ?, ?)
                ''', (media_id, prompt, summary, ingestion_date))

            conn.commit()

            # Insert initial version of the prompt and summary
            add_media_version(media_id, prompt, summary)

            return f"Media '{title}' added successfully with keywords: {', '.join(keyword_list)}"
    except sqlite3.Error as e:
        logger.error(f"SQL Error: {e}")
        raise DatabaseError(f"Error adding media with keywords: {e}")
    except Exception as e:
        logger.error(f"Unexpected Error: {e}")
        raise DatabaseError(f"Unexpected error: {e}")


# Function to add a version of a prompt and summary
def add_media_version(media_id: int, prompt: str, summary: str) -> None:
    try:
        with db.get_connection() as conn:
            cursor = conn.cursor()

            # Get the current version number
            cursor.execute('SELECT MAX(version) FROM MediaVersion WHERE media_id = ?', (media_id,))
            current_version = cursor.fetchone()[0] or 0

            # Insert the new version
            cursor.execute('''
            INSERT INTO MediaVersion (media_id, version, prompt, summary, created_at)
            VALUES (?, ?, ?, ?, ?)
            ''', (media_id, current_version + 1, prompt, summary, datetime.now().strftime('%Y-%m-%d %H:%M:%S')))
            conn.commit()
    except sqlite3.Error as e:
        raise DatabaseError(f"Error adding media version: {e}")


# Function to search the database with advanced options, including keyword search and full-text search
def search_db(search_query: str, search_fields: List[str], keyword: str, page: int = 1, results_per_page: int = 10) -> Union[List[Tuple], str]:
    # Validate input
    if page < 1:
        raise InputError("Page number must be 1 or greater.")

    keyword = keyword.strip().lower()
    with db.get_connection() as conn:
        cursor = conn.cursor()
        offset = (page - 1) * results_per_page

        search_conditions = []
        if search_fields:
            search_conditions.append(" OR ".join([f"media_fts.{field} MATCH ?" for field in search_fields]))
        if keyword:
            search_conditions.append("keyword_fts.keyword MATCH ?")

        where_clause = " AND ".join(search_conditions)

        query = f'''
        SELECT Media.url, Media.title, Media.type, Media.content, Media.author, Media.ingestion_date, Media.prompt, Media.summary
        FROM Media
        JOIN media_fts ON Media.id = media_fts.rowid
        JOIN MediaKeywords ON Media.id = MediaKeywords.media_id
        JOIN Keywords ON MediaKeywords.keyword_id = Keywords.id
        JOIN keyword_fts ON Keywords.id = keyword_fts.rowid
        WHERE {where_clause}
        LIMIT ? OFFSET ?
        '''

        try:
            params = tuple([search_query] * len(search_fields) + [keyword] if keyword else [])
            cursor.execute(query, params + (results_per_page, offset))
            results = cursor.fetchall()
            if not results:
                return "No results found."
            return results
        except sqlite3.Error as e:
            raise DatabaseError(f"Error executing query: {e}")


# Function to format results for display
def format_results(results: Union[List[Tuple], str]) -> pd.DataFrame:
    if isinstance(results, str):
        return pd.DataFrame()  # Return an empty DataFrame if results is a string

    df = pd.DataFrame(results,
                      columns=['URL', 'Title', 'Type', 'Content', 'Author', 'Ingestion Date', 'Prompt', 'Summary'])
    return df


# Gradio function to handle user input and display results with pagination, with better feedback
def search_and_display(search_query: str, search_fields: List[str], keyword: str, page: int, submit: bool):
    if not submit:
        return [], gr.update(visible=False)

    try:
        if not search_query.strip():
            raise InputError("Please enter a valid search query.")

        results = search_db(search_query, search_fields, keyword, page)
        df = format_results(results)

        if df.empty:
            return df, gr.update(value="No results found.", visible=True)
        else:
            return df, gr.update(visible=False)
    except (DatabaseError, InputError) as e:
        return pd.DataFrame(), gr.update(value=str(e), visible=True)


# Function to export search results to CSV with pagination
def export_to_csv(search_query: str, search_fields: List[str], keyword: str, page: int = 1, results_per_file: int = 1000):
    try:
        results = search_db(search_query, search_fields, keyword, page, results_per_file)
        df = format_results(results)
        filename = f'search_results_page_{page}.csv'
        df.to_csv(filename, index=False)
        return f"Results exported to {filename}"
    except (DatabaseError, InputError) as e:
        return str(e)


# Helper function to validate URL format
def is_valid_url(url: str) -> bool:
    regex = re.compile(
        r'^(?:http|ftp)s?://'  # http:// or https://
        r'(?:(?:[A-Z0-9](?:[A-Z0-9-]{0,61}[A-Z0-9])?\.)+(?:[A-Z]{2,6}\.?|[A-Z0-9-]{2,}\.?)|'  # domain...
        r'localhost|'  # localhost...
        r'\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}|'  # ...or ipv4
        r'\[?[A-F0-9]*:[A-F0-9:]+\]?)'  # ...or ipv6
        r'(?::\d+)?'  # optional port
        r'(?:/?|[/?]\S+)$', re.IGNORECASE)
    return re.match(regex, url) is not None


# Helper function to validate date format
def is_valid_date(date_string: str) -> bool:
    try:
        datetime.strptime(date_string, '%Y-%m-%d')
        return True
    except ValueError:
        return False

