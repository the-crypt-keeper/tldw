# Character_Chat_Lib_facade.py
"""
Facade for Character_Chat_Lib to maintain backward compatibility.

This file imports from the refactored modules to provide the same interface
as the original monolithic Character_Chat_Lib.py file.
"""

# Import all functions from the refactored modules
from .modules import *

# Future structure (when refactoring is complete):
"""
# Utility functions (character_utils.py)
from .modules.character_utils import (
    replace_placeholders,
    replace_user_placeholder,
    extract_character_id_from_ui_choice,
    get_character_list_for_ui
)

# I/O operations (character_io.py)
from .modules.character_io import (
    extract_json_from_image_file,
    import_character_card_from_json_string,
    load_character_card_from_string_content,
    import_and_save_character_from_file,
    load_chat_history_from_file_and_save_to_db
)

# Validation and parsing (character_validation.py)
from .modules.character_validation import (
    parse_v1_card,
    parse_v2_card,
    parse_pygmalion_card,
    parse_textgen_card,
    parse_alpaca_card,
    parse_character_book,
    validate_character_book,
    validate_character_book_entry,
    validate_v2_card
)

# Database operations (character_db.py)
from .modules.character_db import (
    _prepare_character_data_for_db_storage,
    create_new_character_from_data,
    get_character_details,
    update_existing_character_details,
    delete_character_from_db,
    search_characters_by_query_text,
    load_character_and_image,
    load_character_wrapper
)

# Chat operations (character_chat.py)
from .modules.character_chat import (
    process_db_messages_to_ui_history,
    load_chat_and_character,
    start_new_chat_session,
    list_character_conversations,
    get_conversation_metadata,
    update_conversation_metadata,
    delete_conversation_by_id,
    search_conversations_by_title_query,
    post_message_to_conversation,
    retrieve_message_details,
    retrieve_conversation_messages_for_ui,
    edit_message_content,
    set_message_ranking,
    remove_message_from_conversation,
    find_messages_in_conversation
)

# Template operations (character_templates.py)
from .modules.character_templates import (
    get_character_template,
    list_character_templates,
    create_character_from_template
)
"""

# Module structure documentation
MODULE_STRUCTURE = {
    "character_utils.py": {
        "description": "Utility functions for text processing and UI helpers",
        "lines": "~150",
        "functions": [
            "replace_placeholders",
            "replace_user_placeholder", 
            "extract_character_id_from_ui_choice",
            "get_character_list_for_ui"
        ]
    },
    "character_io.py": {
        "description": "Import/export operations for character cards",
        "lines": "~600",
        "functions": [
            "extract_json_from_image_file",
            "import_character_card_from_json_string",
            "load_character_card_from_string_content",
            "import_and_save_character_from_file",
            "load_chat_history_from_file_and_save_to_db"
        ]
    },
    "character_validation.py": {
        "description": "Validation and parsing for different card formats",
        "lines": "~500",
        "functions": [
            "parse_v1_card",
            "parse_v2_card",
            "parse_pygmalion_card",
            "parse_textgen_card",
            "parse_alpaca_card",
            "parse_character_book",
            "validate_character_book",
            "validate_character_book_entry",
            "validate_v2_card"
        ]
    },
    "character_db.py": {
        "description": "Database CRUD operations for characters",
        "lines": "~700",
        "functions": [
            "_prepare_character_data_for_db_storage",
            "create_new_character_from_data",
            "get_character_details",
            "update_existing_character_details",
            "delete_character_from_db",
            "search_characters_by_query_text",
            "load_character_and_image",
            "load_character_wrapper"
        ]
    },
    "character_chat.py": {
        "description": "Chat session and message management",
        "lines": "~700",
        "functions": [
            "process_db_messages_to_ui_history",
            "load_chat_and_character",
            "start_new_chat_session",
            "list_character_conversations",
            "get_conversation_metadata",
            "update_conversation_metadata",
            "delete_conversation_by_id",
            "search_conversations_by_title_query",
            "post_message_to_conversation",
            "retrieve_message_details",
            "retrieve_conversation_messages_for_ui",
            "edit_message_content",
            "set_message_ranking",
            "remove_message_from_conversation",
            "find_messages_in_conversation"
        ]
    },
    "character_templates.py": {
        "description": "Character template management",
        "lines": "~200",
        "functions": [
            "get_character_template",
            "list_character_templates",
            "create_character_from_template"
        ]
    }
}

def get_module_info():
    """Return information about the refactored module structure."""
    total_lines = sum(int(m["lines"].replace("~", "")) for m in MODULE_STRUCTURE.values())
    return {
        "total_modules": len(MODULE_STRUCTURE),
        "total_lines": total_lines,
        "average_lines_per_module": total_lines // len(MODULE_STRUCTURE),
        "modules": MODULE_STRUCTURE
    }

# Maintain the same __all__ export for backward compatibility
__all__ = [
    # Utils
    'replace_placeholders',
    'replace_user_placeholder',
    'extract_character_id_from_ui_choice',
    'get_character_list_for_ui',
    
    # I/O
    'extract_json_from_image_file',
    'import_character_card_from_json_string',
    'load_character_card_from_string_content',
    'import_and_save_character_from_file',
    'load_chat_history_from_file_and_save_to_db',
    
    # Validation
    'parse_v1_card',
    'parse_v2_card',
    'parse_pygmalion_card',
    'parse_textgen_card',
    'parse_alpaca_card',
    'parse_character_book',
    'validate_character_book',
    'validate_character_book_entry',
    'validate_v2_card',
    
    # Database
    '_prepare_character_data_for_db_storage',
    'create_new_character_from_data',
    'get_character_details',
    'update_existing_character_details',
    'delete_character_from_db',
    'search_characters_by_query_text',
    'load_character_and_image',
    'load_character_wrapper',
    
    # Chat
    'process_db_messages_to_ui_history',
    'load_chat_and_character',
    'start_new_chat_session',
    'list_character_conversations',
    'get_conversation_metadata',
    'update_conversation_metadata',
    'delete_conversation_by_id',
    'search_conversations_by_title_query',
    'post_message_to_conversation',
    'retrieve_message_details',
    'retrieve_conversation_messages_for_ui',
    'edit_message_content',
    'set_message_ranking',
    'remove_message_from_conversation',
    'find_messages_in_conversation',
    
    # Templates
    'get_character_template',
    'list_character_templates',
    'create_character_from_template'
]