"""
Character I/O operations module.

This module contains functions for importing and exporting character cards.
"""

import base64
import binascii
import io
import json
import os
import yaml
from typing import Dict, List, Optional, Tuple, Any, Union

from PIL import Image
from loguru import logger

from tldw_Server_API.app.core.DB_Management.ChaChaNotes_DB import CharactersRAGDB, CharactersRAGDBError, ConflictError, InputError

# Import validation and parsing functions from character_validation module
from .character_validation import (
    parse_v1_card, parse_v2_card, parse_pygmalion_card,
    parse_textgen_card, parse_alpaca_card, validate_v2_card
)

# Import database functions from character_db module
from .character_db import create_new_character_from_data


def extract_json_from_image_file(image_file_input: Union[str, bytes, io.BytesIO]) -> Optional[str]:
    """Extracts 'chara' metadata (Base64 encoded JSON) from an image file.

    Typically used for PNG character cards (e.g., TavernAI format) that embed
    character data in a 'chara' metadata chunk.

    Args:
        image_file_input (Union[str, bytes, io.BytesIO]): The image data,
            which can be a file path (str), raw image bytes, or a BytesIO stream.

    Returns:
        Optional[str]: The decoded JSON string from the 'chara' metadata if
        found, valid, and successfully decoded. Returns None if the 'chara'
        metadata is not found, the image cannot be processed, or if there's
        an error during decoding or JSON validation.
    """
    img_obj: Optional[Image.Image] = None
    file_name_for_log = "image_stream"
    image_source_to_use: Optional[io.BytesIO] = None

    try:
        if isinstance(image_file_input, str) and os.path.exists(image_file_input):
            file_name_for_log = image_file_input
            with open(image_file_input, 'rb') as f_bytes:
                image_source_to_use = io.BytesIO(f_bytes.read())
        elif isinstance(image_file_input, bytes):
            image_source_to_use = io.BytesIO(image_file_input)
        elif hasattr(image_file_input, 'read'):  # File-like object
            if hasattr(image_file_input, 'name') and image_file_input.name:
                file_name_for_log = image_file_input.name
            image_file_input.seek(0)
            image_source_to_use = io.BytesIO(image_file_input.read())
            image_file_input.seek(0)  # Reset original stream pointer
        else:
            logger.error("extract_json_from_image_file: Invalid input type. Must be file path, bytes, or BytesIO.")
            return None

        if not image_source_to_use: return None

        logger.debug(f"Attempting to extract JSON from image: {file_name_for_log}")

        img_obj = Image.open(image_source_to_use)

        # Support for PNG and WEBP cards (TavernAI, SillyTavern convention)
        if img_obj.format not in ['PNG', 'WEBP']:
            logger.warning(
                f"Image '{file_name_for_log}' is not in PNG or WEBP format (format: {img_obj.format}). 'chara' metadata extraction may fail or not be applicable.")

        # Enhanced metadata extraction - check multiple possible fields
        # 'info' attribute in Pillow Image objects holds metadata chunks.
        # For PNGs, these are tEXt, zTXt, or iTXt chunks.
        # Also check for 'character', 'tEXt' and other common metadata fields
        metadata_found = False
        chara_base64_str = None
        
        if hasattr(img_obj, 'info') and isinstance(img_obj.info, dict):
            # Check multiple possible metadata keys
            for metadata_key in ['chara', 'character', 'tEXt']:
                if metadata_key in img_obj.info:
                    chara_base64_str = img_obj.info[metadata_key]
                    metadata_found = True
                    logger.debug(f"Found character data in '{metadata_key}' field")
                    break
        
        if metadata_found and chara_base64_str:
            try:
                decoded_chara_json_str = base64.b64decode(chara_base64_str).decode('utf-8')
                json.loads(decoded_chara_json_str)  # Validate it's JSON
                logger.info(f"Successfully extracted and decoded 'chara' JSON from '{file_name_for_log}'.")
                return decoded_chara_json_str
            except (binascii.Error, UnicodeDecodeError, json.JSONDecodeError) as decode_err:
                logger.error(
                    f"Error decoding 'chara' metadata from '{file_name_for_log}': {decode_err}. Content (start): {str(chara_base64_str)[:100]}...")
                return None  # Explicitly return None on decode error
            except Exception as e:  # Catch any other unexpected error during decode/load
                logger.error(f"Unexpected error during 'chara' processing from '{file_name_for_log}': {e}",
                             exc_info=True)
                return None
        else:
            logger.debug(
                f"'chara' key not found in image metadata for '{file_name_for_log}'. Available metadata keys: {list(img_obj.info.keys()) if isinstance(img_obj.info, dict) else 'N/A'}")
            return None

    except FileNotFoundError:
        logger.error(f"Image file not found for JSON extraction: {file_name_for_log}")
    except IOError as e:  # Catches PIL.UnidentifiedImageError and other file I/O issues
        logger.error(f"Cannot open or read image file (or not a valid image): {file_name_for_log}. Error: {e}",
                     exc_info=True)
    except Exception as e:
        logger.error(f"Unexpected error extracting JSON from image '{file_name_for_log}': {e}", exc_info=True)
    finally:
        if img_obj:
            img_obj.close()
        if image_source_to_use:
            image_source_to_use.close()
    return None


def import_character_card_from_json_string(json_content_str: str) -> Optional[Dict[str, Any]]:
    """Imports and parses a character card from a JSON string.

    This function attempts to parse a character card from the provided JSON
    string. It automatically detects whether the card is V1 or V2 format.
    For V2 cards (identified by 'spec' or 'spec_version' fields, or
    heuristically by a 'data' node), it performs V2 structural validation.
    If V2 processing fails or is not applicable, it attempts V1 parsing.
    The parsed data is then mapped to a dictionary with keys corresponding
    to the application's database schema.

    Args:
        json_content_str (str): A string containing the character card data
            in JSON format.

    Returns:
        Optional[Dict[str, Any]]: A dictionary with character data mapped to
        DB schema field names (e.g., 'first_message', 'message_example') if
        parsing and validation are successful. Returns None if the JSON is
        invalid, the card structure is unrecognized, critical fields are
        missing (like 'name' after parsing), or any other parsing error occurs.
    """
    if not json_content_str or not json_content_str.strip():
        logger.error("JSON content string is empty or whitespace.")
        return None
    try:
        card_data_dict = json.loads(json_content_str.strip())

        parsed_card: Optional[Dict[str, Any]] = None

        # Enhanced format detection supporting more character card formats
        # Check for various format indicators
        is_explicit_v2_spec = card_data_dict.get('spec') == 'chara_card_v2'
        is_explicit_v2_version_str = str(card_data_dict.get('spec_version', ''))
        is_explicit_v2_version = is_explicit_v2_version_str.startswith("2.")
        
        # Check for Tavern/SillyTavern format
        has_tavern_fields = all(field in card_data_dict for field in ['name', 'description', 'first_mes'])
        
        # Check for Pygmalion format
        has_pygmalion_fields = 'char_name' in card_data_dict and 'char_persona' in card_data_dict
        
        # Check for Text Generation WebUI format
        has_textgen_fields = 'context' in card_data_dict and 'greeting' in card_data_dict
        
        # Check for Alpaca/instruction format
        has_alpaca_fields = 'instruction' in card_data_dict or 'input' in card_data_dict

        has_data_node_heuristic = isinstance(card_data_dict.get('data'), dict) and \
                                  'name' in card_data_dict['data']  # Heuristic for implicit V2

        attempt_v2_processing = is_explicit_v2_spec or is_explicit_v2_version or \
                                (has_data_node_heuristic and not is_explicit_v2_spec and not is_explicit_v2_version)

        if attempt_v2_processing:
            logger.debug("Attempting V2 validation based on card structure/spec.")
            is_valid_v2_struct, v2_errors = validate_v2_card(card_data_dict)

            if not is_valid_v2_struct:
                logger.error(f"V2 Card structural validation failed: {'; '.join(v2_errors)}.")
                if is_explicit_v2_spec or is_explicit_v2_version:
                    logger.error("Card explicitly declared as V2 but failed V2 structural validation. Import aborted.")
                    return None
                else:  # Implicit V2 guess failed validation
                    logger.warning(
                        "Heuristically identified V2 card failed V2 structural validation. Will attempt V1 parsing as fallback.")
                    # No 'return None' here, proceed to V1 attempt below
            else:  # V2 structural validation passed
                logger.info("V2 Card structural validation passed. Attempting to parse as V2 character card.")
                parsed_card = parse_v2_card(card_data_dict)
                if not parsed_card:
                    logger.warning(
                        "V2 parsing failed despite passing V2 structural validation. This might indicate an issue with the parser or an edge case. Attempting V1 parsing as fallback.")
                    # `parsed_card` is None, will fall through to V1 attempt

        # Try other known formats before falling back to V1
        if parsed_card is None:
            # Try Pygmalion format
            if has_pygmalion_fields:
                logger.info("Attempting to parse as Pygmalion format character card.")
                parsed_card = parse_pygmalion_card(card_data_dict)
            
            # Try Text Generation WebUI format
            elif has_textgen_fields:
                logger.info("Attempting to parse as Text Generation WebUI format character card.")
                parsed_card = parse_textgen_card(card_data_dict)
            
            # Try Alpaca/instruction format
            elif has_alpaca_fields:
                logger.info("Attempting to parse as Alpaca/instruction format.")
                parsed_card = parse_alpaca_card(card_data_dict)
        
        # Fallback to V1 if other formats didn't work
        if parsed_card is None:
            logger.info("Attempting to parse as V1 character card.")
            try:
                # parse_v1_card raises ValueError if required fields are missing, or returns None on other errors
                parsed_card = parse_v1_card(card_data_dict)
            except ValueError as ve_v1:
                logger.error(f"V1 card parsing error (likely missing required V1 fields): {ve_v1}")
                parsed_card = None  # Ensure parsed_card is None on this error

        # Final check and return
        if parsed_card and parsed_card.get('name'):  # Name is fundamental
            logger.info(f"Successfully parsed card: '{parsed_card.get('name')}'")
            return parsed_card
        else:
            if parsed_card and not parsed_card.get('name'):
                logger.error("Parsed card is missing 'name'. Import failed.")
            else:  # parsed_card is None
                logger.error("All parsing attempts (V2 and V1) failed to produce a valid card.")
            return None

    except json.JSONDecodeError as e:
        logger.error(f"JSON decode error from string: {e}. Content (start): {json_content_str[:150]}...")
    except Exception as e:  # Catch any other unexpected errors during the process
        logger.error(f"Unexpected error parsing card from JSON string: {e}", exc_info=True)
    return None


def load_character_card_from_string_content(content_str: str) -> Optional[Dict[str, Any]]:
    """Load a character card from a string (JSON or YAML format).
    
    Args:
        content_str: The string content containing character data
        
    Returns:
        Parsed character data dictionary, or None on error
    """
    if not content_str or not content_str.strip():
        logger.error("Content string is empty or whitespace.")
        return None
    
    # Try JSON first
    try:
        return import_character_card_from_json_string(content_str)
    except:
        pass
    
    # Try YAML
    try:
        yaml_data = yaml.safe_load(content_str)
        if isinstance(yaml_data, dict):
            # Convert YAML to JSON string then parse
            json_str = json.dumps(yaml_data)
            return import_character_card_from_json_string(json_str)
    except yaml.YAMLError as e:
        logger.error(f"Failed to parse as YAML: {e}")
    except Exception as e:
        logger.error(f"Unexpected error parsing content: {e}", exc_info=True)
    
    return None


def import_and_save_character_from_file(
    db: CharactersRAGDB,
    file_path: Optional[str] = None,
    file_content: Optional[Union[str, bytes]] = None,
    file_type: Optional[str] = None
) -> Tuple[bool, str, Optional[int]]:
    """Import and save a character from a file.
    
    Args:
        db: Database instance
        file_path: Path to the file (optional if file_content provided)
        file_content: File content (optional if file_path provided)
        file_type: File type hint ('json', 'png', 'yaml', etc.)
        
    Returns:
        Tuple of (success, message, character_id)
    """
    try:
        parsed_card = None
        
        # Determine file type
        if file_path:
            file_ext = os.path.splitext(file_path)[1].lower()
            if file_ext in ['.png', '.webp', '.jpg', '.jpeg']:
                file_type = 'image'
            elif file_ext in ['.json']:
                file_type = 'json'
            elif file_ext in ['.yaml', '.yml']:
                file_type = 'yaml'
        
        # Handle different file types
        if file_type == 'image' or (file_path and file_path.lower().endswith(('.png', '.webp'))):
            # Try to extract JSON from image metadata
            if file_path:
                json_str = extract_json_from_image_file(file_path)
            elif file_content and isinstance(file_content, bytes):
                json_str = extract_json_from_image_file(file_content)
            else:
                return False, "Image file requires bytes content or file path", None
            
            if json_str:
                parsed_card = import_character_card_from_json_string(json_str)
            else:
                return False, "No character data found in image metadata", None
        
        elif file_type in ['json', 'yaml', 'text'] or file_content:
            # Handle text-based formats
            if file_content:
                if isinstance(file_content, bytes):
                    content_str = file_content.decode('utf-8')
                else:
                    content_str = file_content
            elif file_path:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content_str = f.read()
            else:
                return False, "No file content or path provided", None
            
            parsed_card = load_character_card_from_string_content(content_str)
        
        else:
            return False, f"Unsupported file type: {file_type}", None
        
        # Save to database if parsing successful
        if parsed_card:
            character_id = create_new_character_from_data(db, parsed_card)
            if character_id:
                character_name = parsed_card.get('name', 'Unknown')
                return True, f"Successfully imported character '{character_name}'", character_id
            else:
                return False, "Failed to save character to database", None
        else:
            return False, "Failed to parse character data from file", None
    
    except FileNotFoundError:
        return False, f"File not found: {file_path}", None
    except UnicodeDecodeError as e:
        return False, f"Failed to decode file content: {e}", None
    except Exception as e:
        logger.error(f"Unexpected error importing character: {e}", exc_info=True)
        return False, f"Unexpected error: {str(e)}", None


def load_chat_history_from_file_and_save_to_db(
    db: CharactersRAGDB,
    character_id: int,
    file_path: Optional[str] = None,
    file_content: Optional[str] = None,
    title: Optional[str] = None
) -> Tuple[bool, str, Optional[str]]:
    """Load chat history from a file and save to database.
    
    Args:
        db: Database instance
        character_id: The character this chat belongs to
        file_path: Path to the chat history file
        file_content: Chat history content (if not using file_path)
        title: Optional title for the conversation
        
    Returns:
        Tuple of (success, message, conversation_id)
    """
    try:
        # Load content
        if file_content:
            content = file_content
        elif file_path:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
        else:
            return False, "No file path or content provided", None
        
        # Try to parse as JSON
        try:
            chat_data = json.loads(content)
        except json.JSONDecodeError:
            # Try other formats (YAML, plain text, etc.)
            try:
                chat_data = yaml.safe_load(content)
            except:
                # Treat as plain text conversation
                chat_data = {"messages": [{"role": "user", "content": content}]}
        
        # Create conversation
        if not title:
            title = f"Imported Chat - {time.strftime('%Y-%m-%d %H:%M:%S')}"
        
        from .character_chat import start_new_chat_session, post_message_to_conversation
        
        conversation_id = start_new_chat_session(db, character_id, title)
        if not conversation_id:
            return False, "Failed to create conversation", None
        
        # Add messages
        messages = chat_data.get('messages', [])
        if isinstance(messages, list):
            for msg in messages:
                if isinstance(msg, dict):
                    role = msg.get('role', 'user')
                    content = msg.get('content', '')
                    if content:
                        post_message_to_conversation(db, conversation_id, role, content)
        
        return True, f"Successfully imported chat history", conversation_id
    
    except Exception as e:
        logger.error(f"Error loading chat history: {e}", exc_info=True)
        return False, f"Error: {str(e)}", None