"""
Character validation and parsing module.

This module contains functions for validating and parsing different character card formats.
"""

import json
from typing import Dict, List, Optional, Tuple, Any, Set, Union

from loguru import logger


def parse_character_book(book_data: Dict[str, Any]) -> Dict[str, Any]:
    """Parses character book data from a V2 character card structure.

    This function processes the 'character_book' section of a character card,
    extracting its properties and entries according to the V2 specification.

    Args:
        book_data (Dict[str, Any]): A dictionary representing the
            'character_book' field from a character card's data node.

    Returns:
        Dict[str, Any]: A dictionary containing the parsed character book data,
        including 'name', 'description', 'entries', and other book properties.
        Entries are parsed into a structured list.
    """
    parsed_book = {
        'name': book_data.get('name', ''),
        'description': book_data.get('description', ''),
        'scan_depth': book_data.get('scan_depth'),
        'token_budget': book_data.get('token_budget'),
        'recursive_scanning': book_data.get('recursive_scanning', False),
        'extensions': book_data.get('extensions', {}),
        'entries': []
    }

    for entry_raw in book_data.get('entries', []):
        if not isinstance(entry_raw, dict):
            logger.warning(f"Skipping non-dict entry in character_book: {entry_raw}")
            continue

        # Ensure required fields for an entry are present
        if not entry_raw.get('keys') or not isinstance(entry_raw['keys'], list) or \
                'content' not in entry_raw or \
                'enabled' not in entry_raw or \
                'insertion_order' not in entry_raw:
            logger.warning(
                f"Skipping invalid character_book entry due to missing core fields: {entry_raw.get('name', 'N/A')}")
            continue

        parsed_entry = {
            'keys': entry_raw['keys'],
            'content': entry_raw['content'],
            'extensions': entry_raw.get('extensions', {}),
            'enabled': entry_raw['enabled'],
            'insertion_order': entry_raw['insertion_order'],
            'case_sensitive': entry_raw.get('case_sensitive', False),
            'name': entry_raw.get('name', ''),
            'priority': entry_raw.get('priority'),
            'id': entry_raw.get('id'),  # Can be None
            'comment': entry_raw.get('comment', ''),
            'selective': entry_raw.get('selective', False),
            'secondary_keys': entry_raw.get('secondary_keys', []),
            'constant': entry_raw.get('constant', False),
            'position': entry_raw.get('position', 'before_char')  # Default if not specified
        }
        parsed_book['entries'].append(parsed_entry)
    return parsed_book


def parse_v2_card(card_data_json: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Parses a V2 character card (spec_version '2.0') JSON data.

    This function takes a dictionary representing a V2 character card,
    extracts relevant fields from its 'data' node (or root if 'data' is
    absent but structure is V2-like), and maps them to a new dictionary
    with keys corresponding to the application's database schema.
    It assumes basic structural validity (e.g., presence of key fields)
    may have been checked by a prior validation step.

    Args:
        card_data_json (Dict[str, Any]): The dictionary parsed from a V2
            character card JSON. This should be the entire card object.

    Returns:
        Optional[Dict[str, Any]]: A dictionary containing the parsed and
        mapped character data (e.g., 'first_mes' becomes 'first_message').
        Returns None if essential V2 fields are missing or if an unexpected
        error occurs during parsing.
    """
    try:
        # data_node can be 'data' or root for some V2 variants (parsing flexibility)
        data_node = card_data_json.get('data', card_data_json)
        if not isinstance(data_node, dict):
            logger.error("V2 card 'data' node is missing or not a dictionary during parsing.")
            return None

        # Required fields in the source V2 card (using original spec names for parsing)
        # This parsing function relies on these fields existing as per V2 spec.
        required_spec_fields = ['name', 'description', 'personality', 'scenario', 'first_mes', 'mes_example']
        for field in required_spec_fields:
            if field not in data_node or data_node[field] is None:
                logger.error(f"Missing required field '{field}' in V2 card data node during parsing.")
                return None

        # Map to DB schema names
        parsed_data = {
            'name': data_node['name'],
            'description': data_node['description'],
            'personality': data_node['personality'],
            'scenario': data_node['scenario'],
            'first_message': data_node['first_mes'],
            'message_example': data_node['mes_example'],

            'creator_notes': data_node.get('creator_notes', ''),
            'system_prompt': data_node.get('system_prompt', ''),
            'post_history_instructions': data_node.get('post_history_instructions', ''),
            'alternate_greetings': data_node.get('alternate_greetings', []),
            'tags': data_node.get('tags', []),
            'creator': data_node.get('creator', ''),
            'character_version': data_node.get('character_version', ''),
            'extensions': data_node.get('extensions', {}),
            'image_base64': data_node.get('char_image') or data_node.get('image')
        }

        if 'character_book' in data_node and isinstance(data_node['character_book'], dict):
            if not isinstance(parsed_data['extensions'], dict):
                parsed_data['extensions'] = {}
            parsed_data['extensions']['character_book'] = parse_character_book(data_node['character_book'])

        # Log spec/version from top level if present, for info, but parsing proceeds based on data_node content.
        spec = card_data_json.get('spec')
        spec_version = card_data_json.get('spec_version')
        if spec and spec != 'chara_card_v2':
            logger.warning(f"Parsing V2-like card with unexpected 'spec': {spec}.")
        if spec_version and spec_version != '2.0':
            logger.warning(f"Parsing V2-like card with 'spec_version': {spec_version} (expected '2.0').")

        return parsed_data
    except KeyError as e:
        logger.error(f"Missing key during V2 card parsing: {e}")
    except Exception as e:
        logger.error(f"Error parsing V2 card data: {e}", exc_info=True)
    return None


def parse_v1_card(card_data_json: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Parses a V1 character card (flat JSON) into a V2-like structure.

    This function converts a V1 character card, which has a flat JSON structure,
    into a dictionary format that aligns with the V2 card structure and the
    application's database schema. Fields are renamed (e.g., 'first_mes' to
    'first_message'), and any non-standard V1 fields are collected into an
    'extensions' dictionary.

    Args:
        card_data_json (Dict[str, Any]): The dictionary parsed from a V1
            character card JSON.

    Returns:
        Optional[Dict[str, Any]]: A dictionary containing the parsed and
        mapped character data. Returns None if an unexpected error occurs.

    Raises:
        ValueError: If any of the required V1 fields ('name', 'description',
            'personality', 'scenario', 'first_mes', 'mes_example') are missing
            from `card_data_json`.
    """
    try:
        # Required fields in the source V1 card (using original spec names)
        required_spec_fields = ['name', 'description', 'personality', 'scenario', 'first_mes', 'mes_example']
        for field in required_spec_fields:
            if field not in card_data_json or card_data_json[field] is None:
                raise ValueError(f"Missing required field '{field}' in V1 card.")

        # Map to DB schema names
        parsed_data = {
            'name': card_data_json['name'],
            'description': card_data_json['description'],
            'personality': card_data_json['personality'],
            'scenario': card_data_json['scenario'],
            'first_message': card_data_json['first_mes'],
            'message_example': card_data_json['mes_example'],

            'creator_notes': card_data_json.get('creator_notes', ''),
            'system_prompt': card_data_json.get('system_prompt', ''),
            'post_history_instructions': card_data_json.get('post_history_instructions', ''),
            'alternate_greetings': card_data_json.get('alternate_greetings', []),
            'tags': card_data_json.get('tags', []),
            'creator': card_data_json.get('creator', ''),
            'character_version': card_data_json.get('character_version', ''),
            'extensions': {},
            'image_base64': card_data_json.get('char_image') or card_data_json.get('image')
        }

        # Collect any non-standard fields into 'extensions'
        standard_fields = set(required_spec_fields + ['creator_notes', 'system_prompt', 'post_history_instructions',
                                                       'alternate_greetings', 'tags', 'creator', 'character_version',
                                                       'char_image', 'image'])
        for key, value in card_data_json.items():
            if key not in standard_fields:
                parsed_data['extensions'][key] = value

        return parsed_data
    except ValueError as e:
        logger.error(f"Validation error in V1 card: {e}")
        raise
    except Exception as e:
        logger.error(f"Error parsing V1 card data: {e}", exc_info=True)
    return None


def parse_pygmalion_card(card_data_json: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Parse Pygmalion format character card."""
    try:
        # Pygmalion cards have a similar structure to V1 cards
        return parse_v1_card(card_data_json)
    except Exception as e:
        logger.error(f"Error parsing Pygmalion card: {e}", exc_info=True)
        return None


def parse_textgen_card(card_data_json: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Parse TextGen format character card."""
    try:
        # TextGen cards typically have a 'char_name' field instead of 'name'
        if 'char_name' in card_data_json and 'name' not in card_data_json:
            card_data_json['name'] = card_data_json['char_name']
        
        # Map TextGen-specific fields
        if 'char_persona' in card_data_json and 'personality' not in card_data_json:
            card_data_json['personality'] = card_data_json['char_persona']
        
        if 'char_greeting' in card_data_json and 'first_mes' not in card_data_json:
            card_data_json['first_mes'] = card_data_json['char_greeting']
        
        if 'example_dialogue' in card_data_json and 'mes_example' not in card_data_json:
            card_data_json['mes_example'] = card_data_json['example_dialogue']
        
        return parse_v1_card(card_data_json)
    except Exception as e:
        logger.error(f"Error parsing TextGen card: {e}", exc_info=True)
        return None


def parse_alpaca_card(card_data_json: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Parse Alpaca format character card."""
    try:
        # Alpaca cards may have different field names
        if 'char_name' in card_data_json and 'name' not in card_data_json:
            card_data_json['name'] = card_data_json['char_name']
        
        if 'char_description' in card_data_json and 'description' not in card_data_json:
            card_data_json['description'] = card_data_json['char_description']
        
        if 'char_personality' in card_data_json and 'personality' not in card_data_json:
            card_data_json['personality'] = card_data_json['char_personality']
        
        if 'world_scenario' in card_data_json and 'scenario' not in card_data_json:
            card_data_json['scenario'] = card_data_json['world_scenario']
        
        if 'char_greeting' in card_data_json and 'first_mes' not in card_data_json:
            card_data_json['first_mes'] = card_data_json['char_greeting']
        
        if 'example_dialogue' in card_data_json and 'mes_example' not in card_data_json:
            card_data_json['mes_example'] = card_data_json['example_dialogue']
        
        return parse_v1_card(card_data_json)
    except Exception as e:
        logger.error(f"Error parsing Alpaca card: {e}", exc_info=True)
        return None


def validate_character_book(book_data: Dict[str, Any]) -> Tuple[bool, List[str]]:
    """Validates the structure and content of a 'character_book' dictionary.

    Checks for required fields, correct data types, and valid values within
    the character book data, including its entries. This is typically part
    of validating a V2 character card.

    Args:
        book_data (Dict[str, Any]): The character book dictionary to validate.
            This usually comes from the 'character_book' field of a character
            card's 'data' node.

    Returns:
        Tuple[bool, List[str]]: A tuple where:
            - The first element (bool) is `True` if the book data is valid,
              `False` otherwise.
            - The second element (List[str]) is a list of error messages
              describing validation failures. Empty if valid.
    """
    validation_messages = []

    # Optional fields with expected types
    optional_fields = {
        'name': str,
        'description': str,
        'scan_depth': (int, float),
        'token_budget': (int, float),
        'recursive_scanning': bool,
        'extensions': dict,
        # 'entries' is technically required if 'character_book' exists
    }

    for field, expected_type in optional_fields.items():
        if field in book_data:
            if not isinstance(book_data[field], expected_type):
                validation_messages.append(
                    f"Field 'character_book.{field}' must be of type '{expected_type.__name__ if isinstance(expected_type, type) else expected_type}'.")

    # 'entries' is required if character_book itself is present
    if 'entries' not in book_data or not isinstance(book_data['entries'], list):
        validation_messages.append(
            "Field 'character_book.entries' is required and must be a list if 'character_book' is defined.")
        return False, validation_messages  # Cannot proceed without entries

    # Validate each entry in 'entries'
    entries = book_data.get('entries', [])
    entry_ids: Set[Union[int, float]] = set()  # Store IDs to check for uniqueness
    for idx, entry in enumerate(entries):
        if not isinstance(entry, dict):
            validation_messages.append(f"Entry {idx} in 'character_book.entries' is not a dictionary.")
            continue
        is_valid_entry, entry_messages = validate_character_book_entry(entry, idx, entry_ids)
        if not is_valid_entry:
            validation_messages.extend(entry_messages)

    is_valid = len(validation_messages) == 0
    return is_valid, validation_messages


def validate_character_book_entry(entry: Dict[str, Any], idx: int, entry_ids: Set[Union[int, float]]) -> Tuple[
    bool, List[str]]:
    """Validates a single entry within a character book.

    Checks for required fields, correct data types, and valid values for
    a character book entry.

    Args:
        entry (Dict[str, Any]): The character book entry dictionary to validate.
        idx (int): The index of this entry in the entries list (for error messages).
        entry_ids (Set[Union[int, float]]): A set of already seen entry IDs to
            check for uniqueness.

    Returns:
        Tuple[bool, List[str]]: A tuple where:
            - The first element (bool) is `True` if the entry is valid,
              `False` otherwise.
            - The second element (List[str]) is a list of error messages
              describing validation failures. Empty if valid.
    """
    validation_messages = []
    entry_name = entry.get('name', f'Entry {idx}')

    # Required fields with expected types
    required_fields = {
        'keys': list,
        'content': str,
        'enabled': bool,
        'insertion_order': (int, float)
    }

    for field, expected_type in required_fields.items():
        if field not in entry:
            validation_messages.append(f"Entry '{entry_name}' (index {idx}): Missing required field '{field}'.")
        elif not isinstance(entry[field], expected_type):
            validation_messages.append(
                f"Entry '{entry_name}' (index {idx}): Field '{field}' must be of type '{expected_type.__name__ if isinstance(expected_type, type) else expected_type}'.")

    # Optional fields with expected types
    optional_fields = {
        'extensions': dict,
        'case_sensitive': bool,
        'name': str,
        'priority': (int, float),
        'id': (int, float),
        'comment': str,
        'selective': bool,
        'secondary_keys': list,
        'constant': bool,
        'position': str
    }

    for field, expected_type in optional_fields.items():
        if field in entry and not isinstance(entry[field], expected_type):
            validation_messages.append(
                f"Entry '{entry_name}' (index {idx}): Field '{field}' must be of type '{expected_type.__name__ if isinstance(expected_type, type) else expected_type}'.")

    # Additional validation: check keys is non-empty list of strings
    if 'keys' in entry and isinstance(entry['keys'], list):
        if not entry['keys']:
            validation_messages.append(f"Entry '{entry_name}' (index {idx}): 'keys' must be a non-empty list.")
        else:
            for key_idx, key in enumerate(entry['keys']):
                if not isinstance(key, str):
                    validation_messages.append(
                        f"Entry '{entry_name}' (index {idx}): 'keys[{key_idx}]' must be a string.")

    # Check for unique ID if present
    if 'id' in entry and entry['id'] is not None:
        if entry['id'] in entry_ids:
            validation_messages.append(f"Entry '{entry_name}' (index {idx}): Duplicate entry ID '{entry['id']}'.")
        else:
            entry_ids.add(entry['id'])

    # Validate 'position' field if present
    if 'position' in entry and isinstance(entry['position'], str):
        valid_positions = ['before_char', 'after_char']
        if entry['position'] not in valid_positions:
            validation_messages.append(
                f"Entry '{entry_name}' (index {idx}): 'position' must be one of {valid_positions}.")

    is_valid = len(validation_messages) == 0
    return is_valid, validation_messages


def validate_v2_card(card_data: Dict[str, Any]) -> Tuple[bool, List[str]]:
    """Validates a V2 character card structure and content.

    Performs comprehensive validation of a V2 character card, checking for
    required fields, correct data types, and valid values according to the
    V2 specification.

    Args:
        card_data (Dict[str, Any]): The complete V2 character card dictionary
            to validate, including the 'spec', 'spec_version', and 'data' nodes.

    Returns:
        Tuple[bool, List[str]]: A tuple where:
            - The first element (bool) is `True` if the card is valid,
              `False` otherwise.
            - The second element (List[str]) is a list of error messages
              describing validation failures. Empty if valid.
    """
    validation_messages = []

    # Check top-level structure
    if 'spec' in card_data and card_data['spec'] != 'chara_card_v2':
        validation_messages.append(f"Invalid 'spec' value: '{card_data['spec']}'. Expected 'chara_card_v2'.")

    if 'spec_version' in card_data and card_data['spec_version'] != '2.0':
        validation_messages.append(
            f"Invalid 'spec_version' value: '{card_data['spec_version']}'. Expected '2.0'.")

    # Check for 'data' node
    if 'data' not in card_data:
        # Some V2 cards might have fields at root level (fallback)
        data_node = card_data
    else:
        data_node = card_data['data']
        if not isinstance(data_node, dict):
            validation_messages.append("'data' node must be a dictionary.")
            return False, validation_messages

    # Required fields in 'data' node
    required_fields = {
        'name': str,
        'description': str,
        'personality': str,
        'scenario': str,
        'first_mes': str,
        'mes_example': str
    }

    for field, expected_type in required_fields.items():
        if field not in data_node:
            validation_messages.append(f"Missing required field '{field}' in data node.")
        elif not isinstance(data_node[field], expected_type):
            validation_messages.append(
                f"Field '{field}' must be of type '{expected_type.__name__}'.")

    # Optional fields with expected types
    optional_fields = {
        'creator_notes': str,
        'system_prompt': str,
        'post_history_instructions': str,
        'alternate_greetings': list,
        'tags': list,
        'creator': str,
        'character_version': str,
        'extensions': dict
    }

    for field, expected_type in optional_fields.items():
        if field in data_node and not isinstance(data_node[field], expected_type):
            validation_messages.append(
                f"Field '{field}' must be of type '{expected_type.__name__}'.")

    # Validate character_book if present
    if 'character_book' in data_node:
        if not isinstance(data_node['character_book'], dict):
            validation_messages.append("Field 'character_book' must be a dictionary.")
        else:
            is_valid_book, book_messages = validate_character_book(data_node['character_book'])
            if not is_valid_book:
                validation_messages.extend(book_messages)

    is_valid = len(validation_messages) == 0
    return is_valid, validation_messages