import sqlite3
import gradio as gr
import pandas as pd
import logging
import os
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
        conn = self.pool.pop() if self.pool else sqlite3.connect(self.db_name, check_same_thread=False)
        try:
            yield conn
        except sqlite3.Error as e:
            raise DatabaseError(f"Database error: {e}")
        finally:
            self.pool.append(conn)


db = Database()


# Function to create tables with the new media schema
def create_tables():
    with db.get_connection() as conn:
        cursor = conn.cursor()
        try:
            cursor.execute('''
            CREATE TABLE IF NOT EXISTS Media (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                url TEXT,
                title TEXT NOT NULL,
                type TEXT NOT NULL,
                content TEXT,
                author TEXT,
                publication_date TEXT
            )
            ''')
            cursor.execute('''
            CREATE TABLE IF NOT EXISTS Keywords (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                keyword TEXT NOT NULL UNIQUE
            )
            ''')
            cursor.execute('''
            CREATE TABLE IF NOT EXISTS MediaKeywords (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                media_id INTEGER NOT NULL,
                keyword_id INTEGER NOT NULL,
                FOREIGN KEY (media_id) REFERENCES Media(id),
                FOREIGN KEY (keyword_id) REFERENCES Keywords(id)
            )
            ''')
            cursor.execute('''
            CREATE VIRTUAL TABLE IF NOT EXISTS media_fts USING fts5(title, content);
            ''')
            cursor.execute('''
            CREATE VIRTUAL TABLE IF NOT EXISTS keyword_fts USING fts5(keyword);
            ''')
            # Ensure indexing for efficient query performance
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_media_title ON Media(title)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_media_type ON Media(type)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_media_author ON Media(author)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_media_publication_date ON Media(publication_date)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_keywords_keyword ON Keywords(keyword)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_mediakeywords_media_id ON MediaKeywords(media_id)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_mediakeywords_keyword_id ON MediaKeywords(keyword_id)')
            conn.commit()
        except sqlite3.Error as e:
            raise DatabaseError(f"Error creating tables: {e}")


# Function to add a keyword
def add_keyword(keyword: str) -> int:
    keyword = keyword.strip().lower()
    with db.get_connection() as conn:
        cursor = conn.cursor()
        try:
            cursor.execute('INSERT OR IGNORE INTO Keywords (keyword) VALUES (?)', (keyword,))
            cursor.execute('SELECT id FROM Keywords WHERE keyword = ?', (keyword,))
            keyword_id = cursor.fetchone()[0]
            cursor.execute('INSERT INTO keyword_fts (rowid, keyword) VALUES (?, ?)', (keyword_id, keyword))
            conn.commit()
            return keyword_id
        except sqlite3.Error as e:
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


# Function to add multiple keywords
def add_keywords(keywords: str) -> str:
    if not keywords.strip():
        return "No keywords provided."

    keyword_list = [kw.strip().lower() for kw in keywords.split(',')]
    added_keywords = []
    for keyword in keyword_list:
        try:
            keyword_id = add_keyword(keyword)
            added_keywords.append(keyword)
        except DatabaseError as e:
            logger.error(f"Error adding keyword '{keyword}': {e}")
    return f"Keywords added: {', '.join(added_keywords)}"


# Function to add media with keywords
def add_media_with_keywords(url: str, title: str, media_type: str, content: str, keywords: str, author: str = None, publication_date: str = None) -> str:
    # Validate input
    if not url or not title or not media_type or not content or not keywords:
        raise InputError("Please provide all required fields.")

    if not is_valid_url(url):
        raise InputError("Invalid URL format.")

    if media_type not in ['document', 'video', 'article']:
        raise InputError("Invalid media type. Allowed types: document, video, article.")

    if publication_date and not is_valid_date(publication_date):
        raise InputError("Invalid publication date format. Use YYYY-MM-DD.")

    keyword_list = [kw.strip().lower() for kw in keywords.split(',')]
    with db.get_connection() as conn:
        cursor = conn.cursor()
        try:
            cursor.execute('''
            INSERT INTO Media (url, title, type, content, author, publication_date) 
            VALUES (?, ?, ?, ?, ?, ?)
            ''', (url, title, media_type, content, author, publication_date))
            media_id = cursor.lastrowid
            for keyword in keyword_list:
                keyword_id = add_keyword(keyword)
                cursor.execute('INSERT INTO MediaKeywords (media_id, keyword_id) VALUES (?, ?)', (media_id, keyword_id))
            cursor.execute('INSERT INTO media_fts (rowid, title, content) VALUES (?, ?, ?)', (media_id, title, content))
            conn.commit()
            return f"Media '{title}' added successfully with keywords: {', '.join(keyword_list)}"
        except sqlite3.Error as e:
            raise DatabaseError(f"Error adding media with keywords: {e}")


# Function to search the database with advanced options, including keyword search and full-text search
def search_db(search_query: str, search_fields: List[str], keyword: str, page: int = 1, results_per_page: int = 10) -> Union[List[Tuple], str]:
    # Validate input
    if page < 1:
        raise InputError("Page number must be 1 or greater.")

    keyword = keyword.strip().lower()
    with db.get_connection() as conn:
        cursor = conn.cursor()
        offset = (page - 1) * results_per_page
        search_columns = " OR ".join([f"media_fts.{field} MATCH ?" for field in search_fields])

        query = f'''
        SELECT Media.url, Media.title, Media.type, Media.content, Media.author, Media.publication_date
        FROM Media
        JOIN media_fts ON Media.id = media_fts.rowid
        JOIN MediaKeywords ON Media.id = MediaKeywords.media_id
        JOIN Keywords ON MediaKeywords.keyword_id = Keywords.id
        JOIN keyword_fts ON Keywords.id = keyword_fts.rowid
        WHERE ({search_columns}) AND keyword_fts.keyword MATCH ?
        LIMIT ? OFFSET ?
        '''
        try:
            cursor.execute(query, tuple([search_query] * len(search_fields) + [keyword, results_per_page, offset]))
            results = cursor.fetchall()
            if not results:
                return "No results found."
            return results
        except sqlite3.Error as e:
            raise DatabaseError(f"Error executing query: {e}")


# Function to format results for display
def format_results(results: Union[List[Tuple], str]) -> Union[pd.DataFrame, str]:
    if isinstance(results, str):
        return results  # Return error message directly

    df = pd.DataFrame(results, columns=['URL', 'Title', 'Type', 'Content', 'Author', 'Publication Date'])
    return df


# Gradio function to handle user input and display results with pagination, with better feedback
def search_and_display(search_query: str, search_fields: List[str], keyword: str, page: int):
    try:
        if not search_query.strip():
            raise InputError("Please enter a valid search query.")

        results = search_db(search_query, search_fields, keyword, page)
        return format_results(results)
    except (DatabaseError, InputError) as e:
        return str(e)


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
    try:
        result = urlparse(url)
        return all([result.scheme, result.netloc])
    except ValueError:
        return False


# Helper function to validate date format
def is_valid_date(date_string: str) -> bool:
    try:
        datetime.strptime(date_string, '%Y-%m-%d')
        return True
    except ValueError:
        return False


# Create tables
# create_tables()
#
# # Gradio interface setup with tabs
# search_tab = gr.Interface(
#     fn=search_and_display,
#     inputs=[
#         gr.Textbox(label="Search Query", placeholder="Enter your search query here..."),
#         gr.CheckboxGroup(label="Search Fields", choices=["Title", "Content"], default=["Title"]),
#         gr.Textbox(label="Keyword", placeholder="Enter keywords here..."),
#         gr.Number(label="Page", default=1, precision=0)
#     ],
#     outputs=gr.Dataframe(label="Search Results"),
#     title="Search Media Summaries",
#     description="Search for media (documents, videos, articles) and their summaries in the database. Use keywords for better filtering.",
#     live=True
# )
#
# export_tab = gr.Interface(
#     fn=export_to_csv,
#     inputs=[
#         gr.Textbox(label="Search Query", placeholder="Enter your search query here..."),
#         gr.CheckboxGroup(label="Search Fields", choices=["Title", "Content"], default=["Title"]),
#         gr.Textbox(label="Keyword", placeholder="Enter keywords here..."),
#         gr.Number(label="Page", default=1, precision=0),
#         gr.Number(label="Results per File", default=1000, precision=0)
#     ],
#     outputs="text",
#     title="Export Search Results to CSV",
#     description="Export the search results to a CSV file."
# )
#
# keyword_tab = gr.Interface(
#     fn=add_keywords,
#     inputs=gr.Textbox(label="Add Keywords (comma-separated)", placeholder="Enter keywords here..."),
#     outputs="text",
#     title="Add Keywords",
#     description="Add multiple keywords to the database."
# )
#
# delete_keyword_tab = gr.Interface(
#     fn=delete_keyword,
#     inputs=gr.Textbox(label="Delete Keyword", placeholder="Enter keyword to delete here..."),
#     outputs="text",
#     title="Delete Keyword",
#     description="Delete a keyword from the database."
# )
#
# # Combine interfaces into a tabbed interface
# tabbed_interface = gr.TabbedInterface([search_tab, export_tab, keyword_tab, delete_keyword_tab], ["Search", "Export", "Add Keywords", "Delete Keywords"])
#
# # Launch the interface
# tabbed_interface.launch()