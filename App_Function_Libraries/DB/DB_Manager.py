# DB_Manager.py
# Description: This file contains the DatabaseManager class, which is responsible for managing the database connection, i.e. either SQLite or Elasticsearch.
#
# Imports
import configparser
import os
import logging
import time
from typing import Tuple, List, Union, Dict
#
# 3rd-Party Libraries
from elasticsearch import Elasticsearch
#
# Import your existing SQLite functions
from App_Function_Libraries.DB.SQLite_DB import DatabaseError
from App_Function_Libraries.DB.SQLite_DB import (
    update_media_content as sqlite_update_media_content,
    list_prompts as sqlite_list_prompts,
    search_and_display as sqlite_search_and_display,
    fetch_prompt_details as sqlite_fetch_prompt_details,
    keywords_browser_interface as sqlite_keywords_browser_interface,
    add_keyword as sqlite_add_keyword,
    delete_keyword as sqlite_delete_keyword,
    export_keywords_to_csv as sqlite_export_keywords_to_csv,
    ingest_article_to_db as sqlite_ingest_article_to_db,
    add_media_to_database as sqlite_add_media_to_database,
    import_obsidian_note_to_db as sqlite_import_obsidian_note_to_db,
    add_prompt as sqlite_add_prompt,
    delete_chat_message as sqlite_delete_chat_message,
    update_chat_message as sqlite_update_chat_message,
    add_chat_message as sqlite_add_chat_message,
    get_chat_messages as sqlite_get_chat_messages,
    search_chat_conversations as sqlite_search_chat_conversations,
    create_chat_conversation as sqlite_create_chat_conversation,
    save_chat_history_to_database as sqlite_save_chat_history_to_database,
    view_database as sqlite_view_database,
    get_transcripts as sqlite_get_transcripts,
    get_trashed_items as sqlite_get_trashed_items,
    user_delete_item as sqlite_user_delete_item,
    empty_trash as sqlite_empty_trash,
    create_automated_backup as sqlite_create_automated_backup,
    add_or_update_prompt as sqlite_add_or_update_prompt,
    load_prompt_details as sqlite_load_prompt_details,
    load_preset_prompts as sqlite_load_preset_prompts,
    insert_prompt_to_db as sqlite_insert_prompt_to_db,
    delete_prompt as sqlite_delete_prompt,
    search_and_display_items as sqlite_search_and_display_items,
    get_conversation_name as sqlite_get_conversation_name,
    add_media_with_keywords as sqlite_add_media_with_keywords,
    check_media_and_whisper_model as sqlite_check_media_and_whisper_model, \
    create_document_version as sqlite_create_document_version,
    get_document_version as sqlite_get_document_version, sqlite_search_db, add_media_chunk as sqlite_add_media_chunk,
    sqlite_update_fts_for_media, sqlite_get_unprocessed_media, fetch_item_details as sqlite_fetch_item_details, \
    search_media_database as sqlite_search_media_database, mark_as_trash as sqlite_mark_as_trash, \
    get_media_transcripts as sqlite_get_media_transcripts, get_specific_transcript as sqlite_get_specific_transcript, \
    get_media_summaries as sqlite_get_media_summaries, get_specific_summary as sqlite_get_specific_summary, \
    get_media_prompts as sqlite_get_media_prompts, get_specific_prompt as sqlite_get_specific_prompt, \
    delete_specific_transcript as sqlite_delete_specific_transcript,
    delete_specific_summary as sqlite_delete_specific_summary, \
    delete_specific_prompt as sqlite_delete_specific_prompt,
    fetch_keywords_for_media as sqlite_fetch_keywords_for_media, \
    update_keywords_for_media as sqlite_update_keywords_for_media, check_media_exists as sqlite_check_media_exists, \
    search_prompts as sqlite_search_prompts, get_media_content as sqlite_get_media_content, \
    get_paginated_files as sqlite_get_paginated_files, get_media_title as sqlite_get_media_title, \
    get_all_content_from_database as sqlite_get_all_content_from_database,
    get_next_media_id as sqlite_get_next_media_id, \
    batch_insert_chunks as sqlite_batch_insert_chunks, Database, save_workflow_chat_to_db as sqlite_save_workflow_chat_to_db, \
    get_workflow_chat as sqlite_get_workflow_chat, update_media_content_with_version as sqlite_update_media_content_with_version, \
    check_existing_media as sqlite_check_existing_media,
)
#
# Local Imports
from App_Function_Libraries.Utils.Utils import load_comprehensive_config, get_database_path, get_project_relative_path
#
# End of imports
############################################################################################################


############################################################################################################
#
# Database Config loading

logger = logging.getLogger(__name__)

config_path = get_project_relative_path('Config_Files/config.txt')
config = configparser.ConfigParser()
config.read(config_path)

db_path: str = config.get('Database', 'sqlite_path', fallback='./Databases/media_summary.db')
backup_path: str = config.get('Database', 'backup_path', fallback='database_backups')
backup_dir: Union[str, bytes] = os.environ.get('DB_BACKUP_DIR', backup_path)

def get_db_config():
    try:
        config = load_comprehensive_config()

        if 'Database' not in config:
            print("Warning: 'Database' section not found in config. Using default values.")
            return default_db_config()

        return {
            'type': config.get('Database', 'type', fallback='sqlite'),
            'sqlite_path': config.get('Database', 'sqlite_path', fallback='Databases/media_summary.db'),
            'elasticsearch_host': config.get('Database', 'elasticsearch_host', fallback='localhost'),
            'elasticsearch_port': config.getint('Database', 'elasticsearch_port', fallback=9200)
        }
    except FileNotFoundError:
        print("Warning: Config file not found. Using default database configuration.")
        return default_db_config()
    except Exception as e:
        print(f"Error reading config: {str(e)}. Using default database configuration.")
        return default_db_config()

def default_db_config():
    return {
        'type': 'sqlite',
        'sqlite_path': get_database_path('media_summary.db'),
        'elasticsearch_host': 'localhost',
        'elasticsearch_port': 9200
    }

def ensure_directory_exists(file_path):
    directory = os.path.dirname(file_path)
    if not os.path.exists(directory):
        os.makedirs(directory)
        print(f"Created directory: {directory}")

db_config = get_db_config()
db_type = db_config['type']

if db_type == 'sqlite':
    db = Database(os.path.basename(db_config['sqlite_path']))
elif db_type == 'elasticsearch':
    raise NotImplementedError("Elasticsearch support not yet implemented")
else:
    raise ValueError(f"Unsupported database type: {db_type}")

print(f"Database path: {db.db_path}")

def get_db_config():
    try:
        config = load_comprehensive_config()

        if 'Database' not in config:
            print("Warning: 'Database' section not found in config. Using default values.")
            return default_db_config()

        return {
            'type': config.get('Database', 'type', fallback='sqlite'),
            'sqlite_path': config.get('Database', 'sqlite_path', fallback='Databases/media_summary.db'),
            'elasticsearch_host': config.get('Database', 'elasticsearch_host', fallback='localhost'),
            'elasticsearch_port': config.getint('Database', 'elasticsearch_port', fallback=9200)
        }
    except FileNotFoundError:
        print("Warning: Config file not found. Using default database configuration.")
        return default_db_config()
    except Exception as e:
        print(f"Error reading config: {str(e)}. Using default database configuration.")
        return default_db_config()


def default_db_config():
    """Return the default database configuration with project-relative paths."""
    return {
        'type': 'sqlite',
        'sqlite_path': get_database_path('media_summary.db'),
        'elasticsearch_host': 'localhost',
        'elasticsearch_port': 9200
    }


def ensure_directory_exists(file_path):
    directory = os.path.dirname(file_path)
    if not os.path.exists(directory):
        os.makedirs(directory)
        print(f"Created directory: {directory}")

# Use the config to set up the database
db_config = get_db_config()
db_type = db_config['type']

if db_type == 'sqlite':
    db = Database(os.path.basename(db_config['sqlite_path']))
elif db_type == 'elasticsearch':
    # Implement Elasticsearch setup here if needed
    raise NotImplementedError("Elasticsearch support not yet implemented")
else:
    raise ValueError(f"Unsupported database type: {db_type}")

# Print database path for debugging
print(f"Database path: {db.db_path}")

# Sanity Check for SQLite DB
# FIXME - Remove this after testing / Writing Unit tests
# try:
#     db.execute_query("CREATE TABLE IF NOT EXISTS test_table (id INTEGER PRIMARY KEY)")
#     logger.info("Successfully created test table")
# except DatabaseError as e:
#     logger.error(f"Failed to create test table: {e}")

#
# End of Database Config loading
############################################################################################################
#
# DB Search functions

def search_db(search_query: str, search_fields: List[str], keywords: str, page: int = 1, results_per_page: int = 10):
    if db_type == 'sqlite':
        return sqlite_search_db(search_query, search_fields, keywords, page, results_per_page)
    elif db_type == 'elasticsearch':
        # Implement Elasticsearch version when available
        raise NotImplementedError("Elasticsearch version of search_db not yet implemented")
    else:
        raise ValueError(f"Unsupported database type: {db_type}")

def view_database(*args, **kwargs):
    if db_type == 'sqlite':
        return sqlite_view_database(*args, **kwargs)
    elif db_type == 'elasticsearch':
        # Implement Elasticsearch version
        raise NotImplementedError("Elasticsearch version of add_media_with_keywords not yet implemented")

def search_and_display_items(*args, **kwargs):
    if db_type == 'sqlite':
        return sqlite_search_and_display_items(*args, **kwargs)
    elif db_type == 'elasticsearch':
        # Implement Elasticsearch version
        raise NotImplementedError("Elasticsearch version of add_media_with_keywords not yet implemented")

def get_all_content_from_database():
    if db_type == 'sqlite':
        return sqlite_get_all_content_from_database()
    elif db_type == 'elasticsearch':
        # Implement Elasticsearch version
        raise NotImplementedError("Elasticsearch version of add_media_with_keywords not yet implemented")

def search_and_display(*args, **kwargs):
    if db_type == 'sqlite':
        return sqlite_search_and_display(*args, **kwargs)
    elif db_type == 'elasticsearch':
        # Implement Elasticsearch version
        raise NotImplementedError("Elasticsearch version of add_media_with_keywords not yet implemented")

def check_media_exists(*args, **kwargs):
    if db_type == 'sqlite':
        return sqlite_check_media_exists(*args, **kwargs)
    elif db_type == 'elasticsearch':
        # Implement Elasticsearch version
        raise NotImplementedError("Elasticsearch version of add_media_with_keywords not yet implemented")

def get_paginated_files(*args, **kwargs):
    if db_type == 'sqlite':
        return sqlite_get_paginated_files(*args, **kwargs)
    elif db_type == 'elasticsearch':
        # Implement Elasticsearch version
        raise NotImplementedError("Elasticsearch version of add_media_with_keywords not yet implemented")

def get_media_title(*args, **kwargs):
    if db_type == 'sqlite':
        return sqlite_get_media_title(*args, **kwargs)
    elif db_type == 'elasticsearch':
        # Implement Elasticsearch version
        raise NotImplementedError("Elasticsearch version of add_media_with_keywords not yet implemented")

def get_next_media_id():
    if db_type == 'sqlite':
        return sqlite_get_next_media_id()
    elif db_type == 'elasticsearch':
        # Implement Elasticsearch version
        raise NotImplementedError("Elasticsearch version of add_media_with_keywords not yet implemented")

#
# End of DB-Searching functions
############################################################################################################


############################################################################################################
#
# Transcript-related Functions

def get_transcripts(*args, **kwargs):
    if db_type == 'sqlite':
        return sqlite_get_transcripts(*args, **kwargs)
    elif db_type == 'elasticsearch':
        # Implement Elasticsearch version
        raise NotImplementedError("Elasticsearch version of add_media_with_keywords not yet implemented")

#
# End of Transcript-related Functions
############################################################################################################


############################################################################################################
#
# DB-Ingestion functions

def add_media_to_database(*args, **kwargs):
    if db_type == 'sqlite':
        result = sqlite_add_media_to_database(*args, **kwargs)

        # Extract content
        segments = args[2]
        if isinstance(segments, list):
            content = ' '.join([segment.get('Text', '') for segment in segments if 'Text' in segment])
        elif isinstance(segments, dict):
            content = segments.get('text', '') or segments.get('content', '')
        else:
            content = str(segments)

        # Extract media_id from the result
        # Assuming the result is in the format "Media 'Title' added/updated successfully with ID: {media_id}"
        import re
        match = re.search(r"with ID: (\d+)", result)
        if match:
            media_id = int(match.group(1))

            # Create initial document version
            sqlite_create_document_version(media_id, content)

        return result
    elif db_type == 'elasticsearch':
        # Implement Elasticsearch version
        raise NotImplementedError("Elasticsearch version of add_media_to_database not yet implemented")

def check_existing_media(*args, **kwargs):
    if db_type == 'sqlite':
        return sqlite_check_existing_media(*args, **kwargs)
    elif db_type == 'elasticsearch':
        # Implement Elasticsearch version
        raise NotImplementedError("Elasticsearch version of check_existing_media not yet implemented")

def update_media_content_with_version(*args, **kwargs):
    if db_type == 'sqlite':
        return sqlite_update_media_content_with_version(*args, **kwargs)
    elif db_type == 'elasticsearch':
        # Implement Elasticsearch version
        raise NotImplementedError("Elasticsearch version of update_media_content not yet implemented")

def import_obsidian_note_to_db(*args, **kwargs):
    if db_type == 'sqlite':
        return sqlite_import_obsidian_note_to_db(*args, **kwargs)
    elif db_type == 'elasticsearch':
        # Implement Elasticsearch version
        raise NotImplementedError("Elasticsearch version of add_media_with_keywords not yet implemented")


def update_media_content(*args, **kwargs):
    if db_type == 'sqlite':
        result = sqlite_update_media_content(*args, **kwargs)

        # Extract media_id and content
        selected_item = args[0]
        item_mapping = args[1]
        content_input = args[2]

        if selected_item and item_mapping and selected_item in item_mapping:
            media_id = item_mapping[selected_item]

            # Create new document version
            sqlite_create_document_version(media_id, content_input)

        return result
    elif db_type == 'elasticsearch':
        # Implement Elasticsearch version
        raise NotImplementedError("Elasticsearch version of update_media_content not yet implemented")


def add_media_with_keywords(*args, **kwargs):
    if db_type == 'sqlite':
        return sqlite_add_media_with_keywords(*args, **kwargs)
    elif db_type == 'elasticsearch':
        raise NotImplementedError("Elasticsearch version of add_media_with_keywords not yet implemented")

def check_media_and_whisper_model(*args, **kwargs):
    if db_type == 'sqlite':
        return sqlite_check_media_and_whisper_model(*args, **kwargs)
    elif db_type == 'elasticsearch':
        raise NotImplementedError("Elasticsearch version of check_media_and_whisper_model not yet implemented")

def ingest_article_to_db(url, title, author, content, keywords, summary, ingestion_date, custom_prompt):
    if db_type == 'sqlite':
        return sqlite_ingest_article_to_db(url, title, author, content, keywords, summary, ingestion_date, custom_prompt)
    elif db_type == 'elasticsearch':
        # Implement Elasticsearch version
        raise NotImplementedError("Elasticsearch version of ingest_article_to_db not yet implemented")
    else:
        raise ValueError(f"Unsupported database type: {db_type}")


def add_media_chunk(*args, **kwargs):
    if db_type == 'sqlite':
        sqlite_add_media_chunk(*args, **kwargs)
    elif db_type == 'elasticsearch':
        # Implement Elasticsearch version
        raise NotImplementedError("Elasticsearch version not yet implemented")
    else:
        raise ValueError(f"Unsupported database type: {db_type}")

def batch_insert_chunks(*args, **kwargs):
    if db_type == 'sqlite':
        sqlite_batch_insert_chunks(*args, **kwargs)
    elif db_type == 'elasticsearch':
        # Implement Elasticsearch version
        raise NotImplementedError("Elasticsearch version not yet implemented")
    else:
        raise ValueError(f"Unsupported database type: {db_type}")

def update_fts_for_media(media_id: int):
    if db_type == 'sqlite':
        sqlite_update_fts_for_media(db, media_id)
    elif db_type == 'elasticsearch':
        # Implement Elasticsearch version
        raise NotImplementedError("Elasticsearch version not yet implemented")
    else:
        raise ValueError(f"Unsupported database type: {db_type}")


def get_unprocessed_media():
    if db_type == 'sqlite':
        return sqlite_get_unprocessed_media(db)
    elif db_type == 'elasticsearch':
        # Implement Elasticsearch version
        raise NotImplementedError("Elasticsearch version of get_unprocessed_media not yet implemented")
    else:
        raise ValueError(f"Unsupported database type: {db_type}")


#
# End of DB-Ingestion functions
############################################################################################################


############################################################################################################
#
# Prompt-related functions #FIXME rename /resort

def list_prompts(*args, **kwargs):
    if db_type == 'sqlite':
        return sqlite_list_prompts(*args, **kwargs)
    elif db_type == 'elasticsearch':
        # Implement Elasticsearch version
        raise NotImplementedError("Elasticsearch version of add_media_with_keywords not yet implemented")

def search_prompts(query):
    if db_type == 'sqlite':
        return sqlite_search_prompts(query)
    elif db_type == 'elasticsearch':
        # Implement Elasticsearch version
        raise NotImplementedError("Elasticsearch version of add_media_with_keywords not yet implemented")

def fetch_prompt_details(*args, **kwargs):
    if db_type == 'sqlite':
        return sqlite_fetch_prompt_details(*args, **kwargs)
    elif db_type == 'elasticsearch':
        # Implement Elasticsearch version
        raise NotImplementedError("Elasticsearch version of add_media_with_keywords not yet implemented")

def add_prompt(*args, **kwargs):
    if db_type == 'sqlite':
        return sqlite_add_prompt(*args, **kwargs)
    elif db_type == 'elasticsearch':
        # Implement Elasticsearch version
        raise NotImplementedError("Elasticsearch version of add_media_with_keywords not yet implemented")


def add_or_update_prompt(*args, **kwargs):
    if db_type == 'sqlite':
        return sqlite_add_or_update_prompt(*args, **kwargs)
    elif db_type == 'elasticsearch':
        # Implement Elasticsearch version
        raise NotImplementedError("Elasticsearch version of add_media_with_keywords not yet implemented")

def load_prompt_details(*args, **kwargs):
    if db_type == 'sqlite':
        return sqlite_load_prompt_details(*args, **kwargs)
    elif db_type == 'elasticsearch':
        # Implement Elasticsearch version
        raise NotImplementedError("Elasticsearch version of add_media_with_keywords not yet implemented")

def load_preset_prompts(*args, **kwargs):
    if db_type == 'sqlite':
        return sqlite_load_preset_prompts()
    elif db_type == 'elasticsearch':
        # Implement Elasticsearch version
        raise NotImplementedError("Elasticsearch version of add_media_with_keywords not yet implemented")

def insert_prompt_to_db(*args, **kwargs):
    if db_type == 'sqlite':
        return sqlite_insert_prompt_to_db(*args, **kwargs)
    elif db_type == 'elasticsearch':
        # Implement Elasticsearch version
        raise NotImplementedError("Elasticsearch version of add_media_with_keywords not yet implemented")

def delete_prompt(*args, **kwargs):
    if db_type == 'sqlite':
        return sqlite_delete_prompt(*args, **kwargs)
    elif db_type == 'elasticsearch':
        # Implement Elasticsearch version
        raise NotImplementedError("Elasticsearch version of add_media_with_keywords not yet implemented")

def search_media_database(query: str) -> List[Tuple[int, str, str]]:
    if db_type == 'sqlite':
        return sqlite_search_media_database(query)
    elif db_type == 'elasticsearch':
        # Implement Elasticsearch version when available
        raise NotImplementedError("Elasticsearch version of search_media_database not yet implemented")
    else:
        raise ValueError(f"Unsupported database type: {db_type}")

def mark_as_trash(media_id: int) -> None:
    if db_type == 'sqlite':
        return sqlite_mark_as_trash(media_id)
    elif db_type == 'elasticsearch':
        # Implement Elasticsearch version when available
        raise NotImplementedError("Elasticsearch version of mark_as_trash not yet implemented")
    else:
        raise ValueError(f"Unsupported database type: {db_type}")

def get_media_content(media_id: int) -> str:
    if db_type == 'sqlite':
        return sqlite_get_media_content(media_id)
    elif db_type == 'elasticsearch':
        raise NotImplementedError("Elasticsearch version of get_media_content not yet implemented")
    else:
        raise ValueError(f"Unsupported database type: {db_type}")

def get_media_transcripts(media_id: int) -> List[Dict]:
    if db_type == 'sqlite':
        return sqlite_get_media_transcripts(media_id)
    elif db_type == 'elasticsearch':
        raise NotImplementedError("Elasticsearch version of get_media_transcripts not yet implemented")
    else:
        raise ValueError(f"Unsupported database type: {db_type}")

def get_specific_transcript(transcript_id: int) -> Dict:
    if db_type == 'sqlite':
        return sqlite_get_specific_transcript(transcript_id)
    elif db_type == 'elasticsearch':
        raise NotImplementedError("Elasticsearch version of get_specific_transcript not yet implemented")
    else:
        raise ValueError(f"Unsupported database type: {db_type}")

def get_media_summaries(media_id: int) -> List[Dict]:
    if db_type == 'sqlite':
        return sqlite_get_media_summaries(media_id)
    elif db_type == 'elasticsearch':
        raise NotImplementedError("Elasticsearch version of get_media_summaries not yet implemented")
    else:
        raise ValueError(f"Unsupported database type: {db_type}")

def get_specific_summary(summary_id: int) -> Dict:
    if db_type == 'sqlite':
        return sqlite_get_specific_summary(summary_id)
    elif db_type == 'elasticsearch':
        raise NotImplementedError("Elasticsearch version of get_specific_summary not yet implemented")
    else:
        raise ValueError(f"Unsupported database type: {db_type}")

def get_media_prompts(media_id: int) -> List[Dict]:
    if db_type == 'sqlite':
        return sqlite_get_media_prompts(media_id)
    elif db_type == 'elasticsearch':
        raise NotImplementedError("Elasticsearch version of get_media_prompts not yet implemented")
    else:
        raise ValueError(f"Unsupported database type: {db_type}")

def get_specific_prompt(prompt_id: int) -> Dict:
    if db_type == 'sqlite':
        return sqlite_get_specific_prompt(prompt_id)
    elif db_type == 'elasticsearch':
        raise NotImplementedError("Elasticsearch version of get_specific_prompt not yet implemented")
    else:
        return {'error': f"Unsupported database type: {db_type}"}

def delete_specific_transcript(transcript_id: int) -> str:
    if db_type == 'sqlite':
        return sqlite_delete_specific_transcript(transcript_id)
    elif db_type == 'elasticsearch':
        raise NotImplementedError("Elasticsearch version of delete_specific_transcript not yet implemented")
    else:
        raise ValueError(f"Unsupported database type: {db_type}")

def delete_specific_summary(summary_id: int) -> str:
    if db_type == 'sqlite':
        return sqlite_delete_specific_summary(summary_id)
    elif db_type == 'elasticsearch':
        raise NotImplementedError("Elasticsearch version of delete_specific_summary not yet implemented")
    else:
        raise ValueError(f"Unsupported database type: {db_type}")

def delete_specific_prompt(prompt_id: int) -> str:
    if db_type == 'sqlite':
        return sqlite_delete_specific_prompt(prompt_id)
    elif db_type == 'elasticsearch':
        raise NotImplementedError("Elasticsearch version of delete_specific_prompt not yet implemented")
    else:
        raise ValueError(f"Unsupported database type: {db_type}")


#
# End of Prompt-related functions
############################################################################################################

############################################################################################################
#
# Keywords-related Functions

def keywords_browser_interface(*args, **kwargs):
    if db_type == 'sqlite':
        return sqlite_keywords_browser_interface()
    elif db_type == 'elasticsearch':
        # Implement Elasticsearch version
        raise NotImplementedError("Elasticsearch version of add_media_with_keywords not yet implemented")

def add_keyword(*args, **kwargs):
    if db_type == 'sqlite':
        with db.get_connection() as conn:
            cursor = conn.cursor()
        return sqlite_add_keyword(*args, **kwargs)
    elif db_type == 'elasticsearch':
        # Implement Elasticsearch version
        raise NotImplementedError("Elasticsearch version of add_media_with_keywords not yet implemented")

def delete_keyword(*args, **kwargs):
    if db_type == 'sqlite':
        return sqlite_delete_keyword(*args, **kwargs)
    elif db_type == 'elasticsearch':
        # Implement Elasticsearch version
        raise NotImplementedError("Elasticsearch version of add_media_with_keywords not yet implemented")

def export_keywords_to_csv(*args, **kwargs):
    if db_type == 'sqlite':
        return sqlite_export_keywords_to_csv()
    elif db_type == 'elasticsearch':
        # Implement Elasticsearch version
        raise NotImplementedError("Elasticsearch version of add_media_with_keywords not yet implemented")

def update_keywords_for_media(*args, **kwargs):
    if db_type == 'sqlite':
        return sqlite_update_keywords_for_media(*args, **kwargs)
    elif db_type == 'elasticsearch':
        # Implement Elasticsearch version
        raise NotImplementedError("Elasticsearch version of add_media_with_keywords not yet implemented")

def fetch_keywords_for_media(*args, **kwargs):
    if db_type == 'sqlite':
        return sqlite_fetch_keywords_for_media(*args, **kwargs)
    elif db_type == 'elasticsearch':
        # Implement Elasticsearch version
        raise NotImplementedError("Elasticsearch version of add_media_with_keywords not yet implemented")

#
# End of Keywords-related Functions
############################################################################################################

############################################################################################################
#
# Chat-related Functions

def delete_chat_message(*args, **kwargs):
    if db_type == 'sqlite':
        return sqlite_delete_chat_message(*args, **kwargs)
    elif db_type == 'elasticsearch':
        # Implement Elasticsearch version
        raise NotImplementedError("Elasticsearch version of add_media_with_keywords not yet implemented")

def update_chat_message(*args, **kwargs):
    if db_type == 'sqlite':
        return sqlite_update_chat_message(*args, **kwargs)
    elif db_type == 'elasticsearch':
        # Implement Elasticsearch version
        raise NotImplementedError("Elasticsearch version of add_media_with_keywords not yet implemented")

def add_chat_message(*args, **kwargs):
    if db_type == 'sqlite':
        return sqlite_add_chat_message(*args, **kwargs)
    elif db_type == 'elasticsearch':
        # Implement Elasticsearch version
        raise NotImplementedError("Elasticsearch version of add_media_with_keywords not yet implemented")

def get_chat_messages(*args, **kwargs):
    if db_type == 'sqlite':
        return sqlite_get_chat_messages(*args, **kwargs)
    elif db_type == 'elasticsearch':
        # Implement Elasticsearch version
        raise NotImplementedError("Elasticsearch version of add_media_with_keywords not yet implemented")

def search_chat_conversations(*args, **kwargs):
    if db_type == 'sqlite':
        return sqlite_search_chat_conversations(*args, **kwargs)
    elif db_type == 'elasticsearch':
        # Implement Elasticsearch version
        raise NotImplementedError("Elasticsearch version of add_media_with_keywords not yet implemented")

def create_chat_conversation(*args, **kwargs):
    if db_type == 'sqlite':
        return sqlite_create_chat_conversation(*args, **kwargs)
    elif db_type == 'elasticsearch':
        # Implement Elasticsearch version
        raise NotImplementedError("Elasticsearch version of add_media_with_keywords not yet implemented")

def save_chat_history_to_database(*args, **kwargs):
    if db_type == 'sqlite':
        return sqlite_save_chat_history_to_database(*args, **kwargs)
    elif db_type == 'elasticsearch':
        # Implement Elasticsearch version
        raise NotImplementedError("Elasticsearch version of add_media_with_keywords not yet implemented")

def get_conversation_name(*args, **kwargs):
    if db_type == 'sqlite':
        return sqlite_get_conversation_name(*args, **kwargs)
    elif db_type == 'elasticsearch':
        # Implement Elasticsearch version
        raise NotImplementedError("Elasticsearch version of add_media_with_keywords not yet implemented")

#
# End of Chat-related Functions
############################################################################################################

############################################################################################################
#
# Trash-related Functions

def get_trashed_items(*args, **kwargs):
    if db_type == 'sqlite':
        return sqlite_get_trashed_items()
    elif db_type == 'elasticsearch':
        # Implement Elasticsearch version
        raise NotImplementedError("Elasticsearch version of add_media_with_keywords not yet implemented")

def user_delete_item(*args, **kwargs):
    if db_type == 'sqlite':
        return sqlite_user_delete_item(*args, **kwargs)
    elif db_type == 'elasticsearch':
        # Implement Elasticsearch version
        raise NotImplementedError("Elasticsearch version of add_media_with_keywords not yet implemented")

def empty_trash(*args, **kwargs):
    if db_type == 'sqlite':
        return sqlite_empty_trash(*args, **kwargs)
    elif db_type == 'elasticsearch':
        # Implement Elasticsearch version
        raise NotImplementedError("Elasticsearch version of add_media_with_keywords not yet implemented")


def fetch_item_details(media_id: int) -> Tuple[str, str, str]:
    """
    Fetch the details of a media item including content, prompt, and summary.

    Args:
        media_id (int): The ID of the media item.

    Returns:
        Tuple[str, str, str]: A tuple containing (content, prompt, summary).
        If an error occurs, it returns empty strings for each field.
    """
    if db_type == 'sqlite':
        return sqlite_fetch_item_details(media_id)
    elif db_type == 'elasticsearch':
        # Implement Elasticsearch version when available
        raise NotImplementedError("Elasticsearch version of fetch_item_details not yet implemented")
    else:
        raise ValueError(f"Unsupported database type: {db_type}")

#
# End of Trash-related Functions
############################################################################################################


############################################################################################################
#
# DB-Backup Functions

def create_automated_backup(*args, **kwargs):
    if db_type == 'sqlite':
        return sqlite_create_automated_backup(*args, **kwargs)
    elif db_type == 'elasticsearch':
        # Implement Elasticsearch version
        raise NotImplementedError("Elasticsearch version of add_media_with_keywords not yet implemented")

#
# End of DB-Backup Functions
############################################################################################################


############################################################################################################
#
# Document Versioning Functions

def create_document_version(*args, **kwargs):
    if db_type == 'sqlite':
        return sqlite_create_document_version(*args, **kwargs)
    elif db_type == 'elasticsearch':
        # Implement Elasticsearch version
        raise NotImplementedError("Elasticsearch version of create_document_version not yet implemented")

def get_document_version(*args, **kwargs):
    if db_type == 'sqlite':
        return sqlite_get_document_version(*args, **kwargs)
    elif db_type == 'elasticsearch':
        # Implement Elasticsearch version
        raise NotImplementedError("Elasticsearch version of get_document_version not yet implemented")

#
# End of Document Versioning Functions
############################################################################################################


############################################################################################################
#
# Workflow Functions

def get_workflow_chat(*args, **kwargs):
    if db_type == 'sqlite':
        return sqlite_get_workflow_chat(*args, **kwargs)
    elif db_type == 'elasticsearch':
        # Implement Elasticsearch version
        raise NotImplementedError("Elasticsearch version of get_workflow_chat not yet implemented")


def save_workflow_chat_to_db(*args, **kwargs):
    if db_type == 'sqlite':
        # FIXME
        return sqlite_save_workflow_chat_to_db(*args, **kwargs)
    elif db_type == 'elasticsearch':
        # Implement Elasticsearch version
        raise NotImplementedError("Elasticsearch version of save_workflow_chat_to_db not yet implemented")

#
# End of Workflow Functions
############################################################################################################

# Dead code FIXME
# def close_connection():
#     if db_type == 'sqlite':
#         db.get_connection().close()

#
# End of file
############################################################################################################
