# character_chat_db.py
# Database functions for managing character cards and chat histories.
# #
# Imports
import sqlite3
import json
import os
from typing import List, Dict, Optional, Tuple

from Tests.Chat_APIs.Chat_APIs_Integration_test import logging

#
#######################################################################################################################
#
# Constants
DB_PATH = "character_chat.db"
#
# Functions

def initialize_database():
    """Initialize the SQLite database with required tables."""
    conn = sqlite3.connect(DB_PATH)
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

# Migrate to main/make a flaggable feature for character chat usage(outside of an overarching persona) - FIXME
initialize_database()

def add_character_card(card_data: Dict) -> Optional[int]:
    """Add a new character card to the database.

    Returns the ID of the inserted character or None if failed.
    """
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    try:
        # Ensure all required fields are present
        required_fields = ['name', 'description', 'personality', 'scenario', 'image', 'post_history_instructions', 'first_message']
        for field in required_fields:
            if field not in card_data:
                card_data[field] = ''  # Assign empty string if field is missing

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
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM CharacterCards")
    rows = cursor.fetchall()
    conn.close()
    columns = [description[0] for description in cursor.description]
    return [dict(zip(columns, row)) for row in rows]


def get_character_card_by_id(character_id: int) -> Optional[Dict]:
    """Retrieve a single character card by its ID."""
    conn = sqlite3.connect(DB_PATH)
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
    conn = sqlite3.connect(DB_PATH)
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
        print(f"Error updating character card: {e}")
        return False
    finally:
        conn.close()


def delete_character_card(character_id: int) -> bool:
    """Delete a character card and its associated chats."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    try:
        # Delete associated chats first due to foreign key constraint
        cursor.execute("DELETE FROM CharacterChats WHERE character_id = ?", (character_id,))
        cursor.execute("DELETE FROM CharacterCards WHERE id = ?", (character_id,))
        conn.commit()
        return cursor.rowcount > 0
    except sqlite3.Error as e:
        print(f"Error deleting character card: {e}")
        return False
    finally:
        conn.close()


def add_character_chat(character_id: int, conversation_name: str, chat_history: List[Tuple[str, str]],
                       is_snapshot: bool = False) -> Optional[int]:
    """Add a new chat history for a character.

    Returns the ID of the inserted chat or None if failed.
    """
    conn = sqlite3.connect(DB_PATH)
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
        print(f"Error adding character chat: {e}")
        return None
    finally:
        conn.close()


def get_character_chats(character_id: Optional[int] = None) -> List[Dict]:
    """Retrieve all chats, or chats for a specific character if character_id is provided."""
    conn = sqlite3.connect(DB_PATH)
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
    conn = sqlite3.connect(DB_PATH)
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
    conn = sqlite3.connect(DB_PATH)
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
        print(f"Error updating character chat: {e}")
        return False
    finally:
        conn.close()


def delete_character_chat(chat_id: int) -> bool:
    """Delete a specific chat."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    try:
        cursor.execute("DELETE FROM CharacterChats WHERE id = ?", (chat_id,))
        conn.commit()
        return cursor.rowcount > 0
    except sqlite3.Error as e:
        print(f"Error deleting character chat: {e}")
        return False
    finally:
        conn.close()


def migrate_chat_to_media_db(chat_id: int, media_db_path: str = "media_db.db") -> bool:
    """Migrate a chat from CharacterChats to Media DB."""
    # This function assumes that the Media DB has a similar schema for storing chats.
    # You'll need to adjust the schema and fields accordingly.
    character_chat = get_character_chat_by_id(chat_id)
    if not character_chat:
        print(f"No chat found with ID {chat_id}")
        return False

    try:
        conn_media = sqlite3.connect(media_db_path)
        cursor_media = conn_media.cursor()

        # Example: Assuming MediaChats table exists with similar fields
        cursor_media.execute("""
            INSERT INTO MediaChats (character_id, conversation_name, chat_history, is_snapshot, created_at)
            VALUES (?, ?, ?, ?, ?)
        """, (
            character_chat['character_id'],
            character_chat['conversation_name'],
            character_chat['chat_history'],
            character_chat['is_snapshot'],
            character_chat['created_at']
        ))
        conn_media.commit()
        conn_media.close()

        # Optionally, delete the chat from CharacterChats after migration
        delete_character_chat(chat_id)
        return True
    except sqlite3.Error as e:
        print(f"Error migrating chat to Media DB: {e}")
        return False


def save_chat_history_to_character_db(character_name, chat_history):
    try:
        with sqlite3.connect('character_chat.db') as conn:
            cursor = conn.cursor()
            cursor.execute('''
            INSERT INTO CharacterChats (character_name, chat_history)
            VALUES (?, ?)
            ''', (character_name, json.dumps(chat_history)))
            conn.commit()
        return "Chat saved successfully to Character Chat DB."
    except Exception as e:
        logging.error(f"Error saving chat to Character Chat DB: {e}")
        return f"Error saving chat: {e}"


# Update existing chat
def update_chat(chat_id, updated_history):
    try:
        with sqlite3.connect('character_chat.db') as conn:
            cursor = conn.cursor()
            cursor.execute('''
            UPDATE CharacterChats
            SET chat_history = ?
            WHERE id = ?
            ''', (json.dumps(updated_history), chat_id))
            conn.commit()
        return "Chat updated successfully."
    except Exception as e:
        logging.error(f"Error updating chat: {e}")
        return f"Error: {e}"

# Save Character Card to DB
def save_character_card_to_db(card_data):
    try:
        with sqlite3.connect('character_chat.db') as conn:
            cursor = conn.cursor()
            cursor.execute('''
            INSERT INTO CharacterCards (character_name, card_data)
            VALUES (?, ?)
            ''', (card_data['name'], json.dumps(card_data)))
            conn.commit()
    except Exception as e:
        logging.error(f"Error saving character card: {e}")

# Load Character Cards for Management
def load_character_cards():
    try:
        with sqlite3.connect('character_chat.db') as conn:
            cursor = conn.cursor()
            cursor.execute('SELECT character_name FROM CharacterCards')
            return [row[0] for row in cursor.fetchall()]
    except Exception as e:
        logging.error(f"Error loading character cards: {e}")
        return []

#
# End of Character_Chat_DB.py
#######################################################################################################################
