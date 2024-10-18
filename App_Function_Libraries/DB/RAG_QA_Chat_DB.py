# RAG_QA_Chat_DB.py
# Description: This file contains the database operations for the RAG QA Chat + Notes system.
#
# Imports
import configparser
import logging
import re
import sqlite3
import uuid
from contextlib import contextmanager
from datetime import datetime

from App_Function_Libraries.Utils.Utils import get_project_relative_path, get_database_path

#
# External Imports
#
# Local Imports
#
########################################################################################################################
#
# Functions:
# Construct the path to the config file
config_path = get_project_relative_path('Config_Files/config.txt')

# Read the config file
config = configparser.ConfigParser()
config.read(config_path)

# Get the SQLite path from the config, or use the default if not specified
if config.has_section('Database') and config.has_option('Database', 'rag_qa_db_path'):
    rag_qa_db_path = config.get('Database', 'rag_qa_db_path')
else:
    rag_qa_db_path = get_database_path('RAG_QA_Chat.db')

print(f"RAG QA Chat Database path: {rag_qa_db_path}")

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Database schema
SCHEMA_SQL = '''
-- Table for storing chat messages
CREATE TABLE IF NOT EXISTS rag_qa_chats (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    conversation_id TEXT NOT NULL,
    timestamp DATETIME NOT NULL,
    role TEXT NOT NULL,
    content TEXT NOT NULL
);

-- Table for storing conversation metadata
CREATE TABLE IF NOT EXISTS conversation_metadata (
    conversation_id TEXT PRIMARY KEY,
    created_at DATETIME NOT NULL,
    last_updated DATETIME NOT NULL,
    title TEXT NOT NULL
);

-- Table for storing keywords
CREATE TABLE IF NOT EXISTS rag_qa_keywords (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    keyword TEXT NOT NULL UNIQUE
);

-- Table for linking keywords to conversations
CREATE TABLE IF NOT EXISTS rag_qa_conversation_keywords (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    conversation_id TEXT NOT NULL,
    keyword_id INTEGER NOT NULL,
    FOREIGN KEY (conversation_id) REFERENCES conversation_metadata(conversation_id),
    FOREIGN KEY (keyword_id) REFERENCES rag_qa_keywords(id)
);

-- Table for storing keyword collections
CREATE TABLE IF NOT EXISTS rag_qa_keyword_collections (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT NOT NULL UNIQUE,
    parent_id INTEGER,
    FOREIGN KEY (parent_id) REFERENCES rag_qa_keyword_collections(id)
);

-- Table for linking keywords to collections
CREATE TABLE IF NOT EXISTS rag_qa_collection_keywords (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    collection_id INTEGER NOT NULL,
    keyword_id INTEGER NOT NULL,
    FOREIGN KEY (collection_id) REFERENCES rag_qa_keyword_collections(id),
    FOREIGN KEY (keyword_id) REFERENCES rag_qa_keywords(id)
);

-- Table for storing notes
CREATE TABLE IF NOT EXISTS rag_qa_notes (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    conversation_id TEXT NOT NULL,
    content TEXT NOT NULL,
    timestamp DATETIME NOT NULL,
    FOREIGN KEY (conversation_id) REFERENCES conversation_metadata(conversation_id)
);

-- Table for linking notes to keywords
CREATE TABLE IF NOT EXISTS rag_qa_note_keywords (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    note_id INTEGER NOT NULL,
    keyword_id INTEGER NOT NULL,
    FOREIGN KEY (note_id) REFERENCES rag_qa_notes(id),
    FOREIGN KEY (keyword_id) REFERENCES rag_qa_keywords(id)
);

-- Indexes for improved query performance
CREATE INDEX IF NOT EXISTS idx_rag_qa_chats_conversation_id ON rag_qa_chats(conversation_id);
CREATE INDEX IF NOT EXISTS idx_rag_qa_chats_timestamp ON rag_qa_chats(timestamp);
CREATE INDEX IF NOT EXISTS idx_rag_qa_keywords_keyword ON rag_qa_keywords(keyword);
CREATE INDEX IF NOT EXISTS idx_rag_qa_conversation_keywords_conversation_id ON rag_qa_conversation_keywords(conversation_id);
CREATE INDEX IF NOT EXISTS idx_rag_qa_conversation_keywords_keyword_id ON rag_qa_conversation_keywords(keyword_id);
CREATE INDEX IF NOT EXISTS idx_rag_qa_keyword_collections_parent_id ON rag_qa_keyword_collections(parent_id);
CREATE INDEX IF NOT EXISTS idx_rag_qa_collection_keywords_collection_id ON rag_qa_collection_keywords(collection_id);
CREATE INDEX IF NOT EXISTS idx_rag_qa_collection_keywords_keyword_id ON rag_qa_collection_keywords(keyword_id);

-- Full-text search virtual table for chat content
CREATE VIRTUAL TABLE IF NOT EXISTS rag_qa_chats_fts USING fts5(conversation_id, timestamp, role, content);

-- Trigger to keep the FTS table up to date
CREATE TRIGGER IF NOT EXISTS rag_qa_chats_ai AFTER INSERT ON rag_qa_chats BEGIN
  INSERT INTO rag_qa_chats_fts(conversation_id, timestamp, role, content) VALUES (new.conversation_id, new.timestamp, new.role, new.content);
END;
'''

# Database connection management
@contextmanager
def get_db_connection():
    conn = sqlite3.connect(rag_qa_db_path)
    try:
        yield conn
    finally:
        conn.close()

@contextmanager
def transaction():
    with get_db_connection() as conn:
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise

def execute_query(query, params=None, conn=None):
    if conn:
        cursor = conn.cursor()
        if params:
            cursor.execute(query, params)
        else:
            cursor.execute(query)
        return cursor.fetchall()
    else:
        with get_db_connection() as conn:
            cursor = conn.cursor()
            if params:
                cursor.execute(query, params)
            else:
                cursor.execute(query)
            conn.commit()
            return cursor.fetchall()

def create_tables():
    with get_db_connection() as conn:
        conn.executescript(SCHEMA_SQL)
    logger.info("All RAG QA Chat tables created successfully")

# Initialize the database
create_tables()

# Input validation
def validate_keyword(keyword):
    if not isinstance(keyword, str):
        raise ValueError("Keyword must be a string")
    if not keyword.strip():
        raise ValueError("Keyword cannot be empty or just whitespace")
    if len(keyword) > 100:
        raise ValueError("Keyword is too long (max 100 characters)")
    if not re.match(r'^[a-zA-Z0-9\s\-_]+$', keyword):
        raise ValueError("Keyword contains invalid characters")
    return keyword.strip()

def validate_collection_name(name):
    if not isinstance(name, str):
        raise ValueError("Collection name must be a string")
    if not name.strip():
        raise ValueError("Collection name cannot be empty or just whitespace")
    if len(name) > 100:
        raise ValueError("Collection name is too long (max 100 characters)")
    if not re.match(r'^[a-zA-Z0-9\s\-_]+$', name):
        raise ValueError("Collection name contains invalid characters")
    return name.strip()

# Core functions
def add_keyword(keyword, conn=None):
    try:
        validated_keyword = validate_keyword(keyword)
        query = "INSERT OR IGNORE INTO rag_qa_keywords (keyword) VALUES (?)"
        execute_query(query, (validated_keyword,), conn)
        logger.info(f"Keyword '{validated_keyword}' added successfully")
    except ValueError as e:
        logger.error(f"Invalid keyword: {e}")
        raise
    except Exception as e:
        logger.error(f"Error adding keyword '{keyword}': {e}")
        raise

def create_keyword_collection(name, parent_id=None):
    try:
        validated_name = validate_collection_name(name)
        query = "INSERT INTO rag_qa_keyword_collections (name, parent_id) VALUES (?, ?)"
        execute_query(query, (validated_name, parent_id))
        logger.info(f"Keyword collection '{validated_name}' created successfully")
    except ValueError as e:
        logger.error(f"Invalid collection name: {e}")
        raise
    except Exception as e:
        logger.error(f"Error creating keyword collection '{name}': {e}")
        raise

def add_keyword_to_collection(collection_name, keyword):
    try:
        validated_collection_name = validate_collection_name(collection_name)
        validated_keyword = validate_keyword(keyword)

        with transaction() as conn:
            add_keyword(validated_keyword, conn)

            query = '''
            INSERT INTO rag_qa_collection_keywords (collection_id, keyword_id)
            SELECT c.id, k.id
            FROM rag_qa_keyword_collections c, rag_qa_keywords k
            WHERE c.name = ? AND k.keyword = ?
            '''
            execute_query(query, (validated_collection_name, validated_keyword), conn)

        logger.info(f"Keyword '{validated_keyword}' added to collection '{validated_collection_name}' successfully")
    except ValueError as e:
        logger.error(f"Invalid input: {e}")
        raise
    except Exception as e:
        logger.error(f"Error adding keyword '{keyword}' to collection '{collection_name}': {e}")
        raise

def add_keywords_to_conversation(conversation_id, keywords):
    if not isinstance(keywords, (list, tuple)):
        raise ValueError("Keywords must be a list or tuple")
    try:
        with transaction() as conn:
            for keyword in keywords:
                validated_keyword = validate_keyword(keyword)
                add_keyword(validated_keyword, conn)

                query = '''
                INSERT INTO rag_qa_conversation_keywords (conversation_id, keyword_id)
                SELECT ?, id FROM rag_qa_keywords WHERE keyword = ?
                '''
                execute_query(query, (conversation_id, validated_keyword), conn)

        logger.info(f"Keywords added to conversation '{conversation_id}' successfully")
    except ValueError as e:
        logger.error(f"Invalid keyword: {e}")
        raise
    except Exception as e:
        logger.error(f"Error adding keywords to conversation '{conversation_id}': {e}")
        raise

def get_keywords_for_conversation(conversation_id):
    try:
        query = '''
        SELECT k.keyword
        FROM rag_qa_keywords k
        JOIN rag_qa_conversation_keywords ck ON k.id = ck.keyword_id
        WHERE ck.conversation_id = ?
        '''
        result = execute_query(query, (conversation_id,))
        keywords = [row[0] for row in result]
        logger.info(f"Retrieved {len(keywords)} keywords for conversation '{conversation_id}'")
        return keywords
    except Exception as e:
        logger.error(f"Error getting keywords for conversation '{conversation_id}': {e}")
        raise

def get_keywords_for_collection(collection_name):
    try:
        query = '''
        SELECT k.keyword
        FROM rag_qa_keywords k
        JOIN rag_qa_collection_keywords ck ON k.id = ck.keyword_id
        JOIN rag_qa_keyword_collections c ON ck.collection_id = c.id
        WHERE c.name = ?
        '''
        result = execute_query(query, (collection_name,))
        keywords = [row[0] for row in result]
        logger.info(f"Retrieved {len(keywords)} keywords for collection '{collection_name}'")
        return keywords
    except Exception as e:
        logger.error(f"Error getting keywords for collection '{collection_name}': {e}")
        raise

def save_notes(conversation_id, content):
    """Save notes to the database."""
    try:
        query = "INSERT INTO rag_qa_notes (conversation_id, content, timestamp) VALUES (?, ?, ?)"
        timestamp = datetime.now().isoformat()
        execute_query(query, (conversation_id, content, timestamp))
        logger.info(f"Notes saved for conversation '{conversation_id}'")
    except Exception as e:
        logger.error(f"Error saving notes for conversation '{conversation_id}': {e}")
        raise

def get_notes(conversation_id):
    """Retrieve notes for a given conversation."""
    try:
        query = "SELECT content FROM rag_qa_notes WHERE conversation_id = ?"
        result = execute_query(query, (conversation_id,))
        notes = [row[0] for row in result]
        logger.info(f"Retrieved {len(notes)} notes for conversation '{conversation_id}'")
        return notes
    except Exception as e:
        logger.error(f"Error getting notes for conversation '{conversation_id}': {e}")
        raise

def clear_notes(conversation_id):
    """Clear all notes for a given conversation."""
    try:
        query = "DELETE FROM rag_qa_notes WHERE conversation_id = ?"
        execute_query(query, (conversation_id,))
        logger.info(f"Cleared notes for conversation '{conversation_id}'")
    except Exception as e:
        logger.error(f"Error clearing notes for conversation '{conversation_id}': {e}")
        raise

def add_keywords_to_note(note_id, keywords):
    """Associate keywords with a note."""
    try:
        with transaction() as conn:
            for keyword in keywords:
                validated_keyword = validate_keyword(keyword)
                add_keyword(validated_keyword, conn)

                # Retrieve the keyword ID
                query = "SELECT id FROM rag_qa_keywords WHERE keyword = ?"
                result = execute_query(query, (validated_keyword,), conn)
                if result:
                    keyword_id = result[0][0]
                else:
                    raise Exception(f"Keyword '{validated_keyword}' not found after insertion")

                # Link the note and keyword
                query = "INSERT INTO rag_qa_note_keywords (note_id, keyword_id) VALUES (?, ?)"
                execute_query(query, (note_id, keyword_id), conn)

        logger.info(f"Keywords added to note ID '{note_id}' successfully")
    except Exception as e:
        logger.error(f"Error adding keywords to note ID '{note_id}': {e}")
        raise

def get_keywords_for_note(note_id):
    """Retrieve keywords associated with a given note."""
    try:
        query = '''
        SELECT k.keyword
        FROM rag_qa_keywords k
        JOIN rag_qa_note_keywords nk ON k.id = nk.keyword_id
        WHERE nk.note_id = ?
        '''
        result = execute_query(query, (note_id,))
        keywords = [row[0] for row in result]
        logger.info(f"Retrieved {len(keywords)} keywords for note ID '{note_id}'")
        return keywords
    except Exception as e:
        logger.error(f"Error getting keywords for note ID '{note_id}': {e}")
        raise

def clear_keywords_from_note(note_id):
    """Clear all keywords from a given note."""
    try:
        query = "DELETE FROM rag_qa_note_keywords WHERE note_id = ?"
        execute_query(query, (note_id,))
        logger.info(f"Cleared keywords for note ID '{note_id}'")
    except Exception as e:
        logger.error(f"Error clearing keywords for note ID '{note_id}': {e}")
        raise

def save_message(conversation_id, role, content):
    try:
        timestamp = datetime.now().isoformat()
        query = "INSERT INTO rag_qa_chats (conversation_id, timestamp, role, content) VALUES (?, ?, ?, ?)"
        execute_query(query, (conversation_id, timestamp, role, content))

        # Update last_updated in conversation_metadata
        update_query = "UPDATE conversation_metadata SET last_updated = ? WHERE conversation_id = ?"
        execute_query(update_query, (timestamp, conversation_id))

        logger.info(f"Message saved for conversation '{conversation_id}'")
    except Exception as e:
        logger.error(f"Error saving message for conversation '{conversation_id}': {e}")
        raise

def start_new_conversation(title="Untitled Conversation"):
    try:
        conversation_id = str(uuid.uuid4())
        query = "INSERT INTO conversation_metadata (conversation_id, created_at, last_updated, title) VALUES (?, ?, ?, ?)"
        now = datetime.now().isoformat()
        execute_query(query, (conversation_id, now, now, title))
        logger.info(f"New conversation '{conversation_id}' started with title '{title}'")
        return conversation_id
    except Exception as e:
        logger.error(f"Error starting new conversation: {e}")
        raise

# Pagination helper function
def get_paginated_results(query, params=None, page=1, page_size=20):
    try:
        offset = (page - 1) * page_size
        paginated_query = f"{query} LIMIT ? OFFSET ?"
        if params:
            paginated_params = params + (page_size, offset)
        else:
            paginated_params = (page_size, offset)

        result = execute_query(paginated_query, paginated_params)

        count_query = f"SELECT COUNT(*) FROM ({query}) AS total"
        count_params = params if params else ()

        total_count = execute_query(count_query, count_params)[0][0]

        total_pages = (total_count + page_size - 1) // page_size

        logger.info(f"Retrieved page {page} of {total_pages} (total items: {total_count})")
        return result, total_pages, total_count
    except Exception as e:
        logger.error(f"Error retrieving paginated results: {e}")
        raise

def get_all_collections(page=1, page_size=20):
    try:
        query = "SELECT name FROM rag_qa_keyword_collections"
        results, total_pages, total_count = get_paginated_results(query, page=page, page_size=page_size)
        collections = [row[0] for row in results]
        logger.info(f"Retrieved {len(collections)} keyword collections (page {page} of {total_pages})")
        return collections, total_pages, total_count
    except Exception as e:
        logger.error(f"Error getting collections: {e}")
        raise

def search_conversations_by_keywords(keywords, page=1, page_size=20):
    try:
        placeholders = ','.join(['?' for _ in keywords])
        query = f'''
        SELECT DISTINCT cm.conversation_id, cm.title
        FROM conversation_metadata cm
        JOIN rag_qa_conversation_keywords ck ON cm.conversation_id = ck.conversation_id
        JOIN rag_qa_keywords k ON ck.keyword_id = k.id
        WHERE k.keyword IN ({placeholders})
        '''
        results, total_pages, total_count = get_paginated_results(query, tuple(keywords), page, page_size)
        logger.info(
            f"Found {total_count} conversations matching keywords: {', '.join(keywords)} (page {page} of {total_pages})")
        return results, total_pages, total_count
    except Exception as e:
        logger.error(f"Error searching conversations by keywords {keywords}: {e}")
        raise

def load_chat_history(conversation_id, page=1, page_size=50):
    try:
        query = "SELECT role, content FROM rag_qa_chats WHERE conversation_id = ? ORDER BY timestamp"
        results, total_pages, total_count = get_paginated_results(query, (conversation_id,), page, page_size)
        logger.info(
            f"Loaded {len(results)} messages for conversation '{conversation_id}' (page {page} of {total_pages})")
        return results, total_pages, total_count
    except Exception as e:
        logger.error(f"Error loading chat history for conversation '{conversation_id}': {e}")
        raise

#
# End of RAG_QA_Chat_DB.py
####################################################################################################
