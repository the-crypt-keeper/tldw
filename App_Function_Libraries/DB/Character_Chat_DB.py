# character_chat_db.py
# Database functions for managing character cards and chat histories.
# #
# Imports
import configparser
import sqlite3
import json
import os
from typing import List, Dict, Optional, Tuple

from App_Function_Libraries.Utils.Utils import get_database_dir, get_project_relative_path, get_database_path
from Tests.Chat_APIs.Chat_APIs_Integration_test import logging

#
#######################################################################################################################
#
#

def ensure_database_directory():
    os.makedirs(get_database_dir(), exist_ok=True)

ensure_database_directory()

# FIXME - Setup properly and test/add documentation for its existence...
# Construct the path to the config file
config_path = get_project_relative_path('Config_Files/config.txt')

# Read the config file
config = configparser.ConfigParser()
config.read(config_path)

# Get the chat db path from the config, or use the default if not specified
chat_DB_PATH = config.get('Database', 'chatDB_path', fallback=get_database_path('chatDB.db'))
print(f"Chat Database path: {chat_DB_PATH}")

# Functions

def initialize_database():
    """Initialize the SQLite database with required tables."""
    conn = sqlite3.connect(chat_DB_PATH)
    cursor = conn.cursor()

    # Create CharacterCards table with image as BLOB
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS CharacterCards (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        name TEXT UNIQUE NOT NULL,
        description TEXT,
        personality TEXT,
        scenario TEXT,
        image BLOB,
        post_history_instructions TEXT,
        first_message TEXT,
        created_at DATETIME DEFAULT CURRENT_TIMESTAMP
    );
    """)

    # Create CharacterChats table
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS CharacterChats (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        character_id INTEGER NOT NULL,
        conversation_name TEXT,
        chat_history TEXT,
        is_snapshot BOOLEAN DEFAULT FALSE,
        created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY (character_id) REFERENCES CharacterCards(id)
    );
    """)

    conn.commit()
    conn.close()

initialize_database()

def add_character_card(card_data: Dict) -> Optional[int]:
    """Add or update a character card in the database.

    Returns the ID of the inserted character or None if failed.
    """
    conn = sqlite3.connect(chat_DB_PATH)
    cursor = conn.cursor()
    try:
        # Ensure all required fields are present
        required_fields = ['name', 'description', 'personality', 'scenario', 'image', 'post_history_instructions', 'first_message']
        for field in required_fields:
            if field not in card_data:
                card_data[field] = ''  # Assign empty string if field is missing

        # Check if character already exists
        cursor.execute("SELECT id FROM CharacterCards WHERE name = ?", (card_data['name'],))
        row = cursor.fetchone()

        if row:
            # Update existing character
            character_id = row[0]
            cursor.execute("""
                UPDATE CharacterCards
                SET description = ?, personality = ?, scenario = ?, image = ?, post_history_instructions = ?, first_message = ?
                WHERE id = ?
            """, (
                card_data['description'],
                card_data['personality'],
                card_data['scenario'],
                card_data['image'],
                card_data['post_history_instructions'],
                card_data['first_message'],
                character_id
            ))
        else:
            # Insert new character
            cursor.execute("""
                INSERT INTO CharacterCards (name, description, personality, scenario, image, post_history_instructions, first_message)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (
                card_data['name'],
                card_data['description'],
                card_data['personality'],
                card_data['scenario'],
                card_data['image'],
                card_data['post_history_instructions'],
                card_data['first_message']
            ))
            character_id = cursor.lastrowid

        conn.commit()
        return cursor.lastrowid
    except sqlite3.IntegrityError as e:
        logging.error(f"Error adding character card: {e}")
        return None
    except Exception as e:
        logging.error(f"Unexpected error adding character card: {e}")
        return None
    finally:
        conn.close()


def get_character_cards() -> List[Dict]:
    """Retrieve all character cards from the database."""
    conn = sqlite3.connect(chat_DB_PATH)
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM CharacterCards")
    rows = cursor.fetchall()
    columns = [description[0] for description in cursor.description]
    conn.close()
    characters = [dict(zip(columns, row)) for row in rows]
    logging.debug(f"Characters fetched from DB: {characters}")
    return characters


def get_character_card_by_id(character_id: int) -> Optional[Dict]:
    """Retrieve a single character card by its ID."""
    conn = sqlite3.connect(chat_DB_PATH)
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM CharacterCards WHERE id = ?", (character_id,))
    row = cursor.fetchone()
    conn.close()
    if row:
        columns = [description[0] for description in cursor.description]
        return dict(zip(columns, row))
    return None


def update_character_card(character_id: int, card_data: Dict) -> bool:
    """Update an existing character card."""
    conn = sqlite3.connect(chat_DB_PATH)
    cursor = conn.cursor()
    try:
        cursor.execute("""
            UPDATE CharacterCards
            SET name = ?, description = ?, personality = ?, scenario = ?, image = ?, post_history_instructions = ?, first_message = ?
            WHERE id = ?
        """, (
            card_data.get('name'),
            card_data.get('description'),
            card_data.get('personality'),
            card_data.get('scenario'),
            card_data.get('image'),
            card_data.get('post_history_instructions', ''),
            card_data.get('first_message', "Hello! I'm ready to chat."),
            character_id
        ))
        conn.commit()
        return cursor.rowcount > 0
    except sqlite3.IntegrityError as e:
        logging.error(f"Error updating character card: {e}")
        return False
    finally:
        conn.close()


def delete_character_card(character_id: int) -> bool:
    """Delete a character card and its associated chats."""
    conn = sqlite3.connect(chat_DB_PATH)
    cursor = conn.cursor()
    try:
        # Delete associated chats first due to foreign key constraint
        cursor.execute("DELETE FROM CharacterChats WHERE character_id = ?", (character_id,))
        cursor.execute("DELETE FROM CharacterCards WHERE id = ?", (character_id,))
        conn.commit()
        return cursor.rowcount > 0
    except sqlite3.Error as e:
        logging.error(f"Error deleting character card: {e}")
        return False
    finally:
        conn.close()


def add_character_chat(character_id: int, conversation_name: str, chat_history: List[Tuple[str, str]],
                       is_snapshot: bool = False) -> Optional[int]:
    """Add a new chat history for a character.

    Returns the ID of the inserted chat or None if failed.
    """
    conn = sqlite3.connect(chat_DB_PATH)
    cursor = conn.cursor()
    try:
        chat_history_json = json.dumps(chat_history)
        cursor.execute("""
            INSERT INTO CharacterChats (character_id, conversation_name, chat_history, is_snapshot)
            VALUES (?, ?, ?, ?)
        """, (
            character_id,
            conversation_name,
            chat_history_json,
            is_snapshot
        ))
        conn.commit()
        return cursor.lastrowid
    except sqlite3.Error as e:
        logging.error(f"Error adding character chat: {e}")
        return None
    finally:
        conn.close()


def get_character_chats(character_id: Optional[int] = None) -> List[Dict]:
    """Retrieve all chats, or chats for a specific character if character_id is provided."""
    conn = sqlite3.connect(chat_DB_PATH)
    cursor = conn.cursor()
    if character_id is not None:
        cursor.execute("SELECT * FROM CharacterChats WHERE character_id = ?", (character_id,))
    else:
        cursor.execute("SELECT * FROM CharacterChats")
    rows = cursor.fetchall()
    columns = [description[0] for description in cursor.description]
    conn.close()
    return [dict(zip(columns, row)) for row in rows]


def get_character_chat_by_id(chat_id: int) -> Optional[Dict]:
    """Retrieve a single chat by its ID."""
    conn = sqlite3.connect(chat_DB_PATH)
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM CharacterChats WHERE id = ?", (chat_id,))
    row = cursor.fetchone()
    conn.close()
    if row:
        columns = [description[0] for description in cursor.description]
        chat = dict(zip(columns, row))
        chat['chat_history'] = json.loads(chat['chat_history'])
        return chat
    return None


def update_character_chat(chat_id: int, chat_history: List[Tuple[str, str]]) -> bool:
    """Update an existing chat history."""
    conn = sqlite3.connect(chat_DB_PATH)
    cursor = conn.cursor()
    try:
        chat_history_json = json.dumps(chat_history)
        cursor.execute("""
            UPDATE CharacterChats
            SET chat_history = ?
            WHERE id = ?
        """, (
            chat_history_json,
            chat_id
        ))
        conn.commit()
        return cursor.rowcount > 0
    except sqlite3.Error as e:
        logging.error(f"Error updating character chat: {e}")
        return False
    finally:
        conn.close()


def delete_character_chat(chat_id: int) -> bool:
    """Delete a specific chat."""
    conn = sqlite3.connect(chat_DB_PATH)
    cursor = conn.cursor()
    try:
        cursor.execute("DELETE FROM CharacterChats WHERE id = ?", (chat_id,))
        conn.commit()
        return cursor.rowcount > 0
    except sqlite3.Error as e:
        logging.error(f"Error deleting character chat: {e}")
        return False
    finally:
        conn.close()

def save_chat_history_to_character_db(character_id: int, conversation_name: str, chat_history: List[Tuple[str, str]]) -> Optional[int]:
    """Save chat history to the CharacterChats table.

    Returns the ID of the inserted chat or None if failed.
    """
    return add_character_chat(character_id, conversation_name, chat_history)

def migrate_chat_to_media_db():
    pass

#
# End of Character_Chat_DB.py
#######################################################################################################################
