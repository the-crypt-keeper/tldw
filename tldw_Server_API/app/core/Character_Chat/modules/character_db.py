"""
Character database operations module.

This module contains functions for CRUD operations on character data.
"""

import base64
import binascii
import io
import json
from typing import Dict, List, Optional, Tuple, Any

from PIL import Image
from loguru import logger

from tldw_Server_API.app.core.DB_Management.ChaChaNotes_DB import CharactersRAGDB, CharactersRAGDBError, ConflictError, InputError


def _prepare_character_data_for_db_storage(
    input_data: Dict[str, Any],
    is_update: bool = False
) -> Dict[str, Any]:
    """
    Prepares character data (from a dictionary, often derived from Pydantic model)
    for DB insertion/update.
    Handles 'image_base64' to 'image' (bytes) conversion.
    Ensures JSON-like fields are Python objects.
    """
    db_data = input_data.copy() # Work on a copy

    # Handle image_base64: convert to bytes for 'image' field, with optimization
    if 'image_base64' in db_data:
        base64_str = db_data.pop('image_base64')
        if base64_str and isinstance(base64_str, str):
            try:
                if ',' in base64_str and base64_str.startswith("data:image"):
                    base64_str = base64_str.split(',', 1)[1]
                image_bytes = base64.b64decode(base64_str, validate=True)
                
                # Image optimization - convert to WEBP and resize if needed
                try:
                    img = Image.open(io.BytesIO(image_bytes))
                    
                    # Convert RGBA to RGB if needed
                    if img.mode == 'RGBA':
                        background = Image.new('RGB', img.size, (255, 255, 255))
                        background.paste(img, mask=img.split()[3])
                        img = background
                    elif img.mode not in ['RGB', 'L']:
                        img = img.convert('RGB')
                    
                    # Resize if too large (max 512x768)
                    max_size = (512, 768)
                    if img.size[0] > max_size[0] or img.size[1] > max_size[1]:
                        img.thumbnail(max_size, Image.Resampling.LANCZOS)
                        logger.info(f"Resized character image from {img.size} to fit within {max_size}")
                    
                    # Save as optimized WEBP
                    output = io.BytesIO()
                    img.save(output, format='WEBP', quality=85, method=6, optimize=True)
                    db_data['image'] = output.getvalue()
                    logger.info(f"Optimized character image: {len(image_bytes)} -> {len(db_data['image'])} bytes")
                except Exception as img_err:
                    logger.warning(f"Could not optimize image, using original: {img_err}")
                    db_data['image'] = image_bytes  # Fall back to original
                    
            except (binascii.Error, ValueError) as e: # ValueError for invalid padding etc.
                logger.error(f"Invalid image_base64 data for character: {e}")
                # Raise an InputError that API can catch and convert to 400
                raise InputError(f"Invalid image_base64 data: {e}")
        else: # image_base64 was None or empty string
            db_data['image'] = None # Explicitly set image to None to remove it if updating
    elif not is_update and 'image' not in db_data : # For create, if image_base64 and image not provided
        db_data['image'] = None

    # Ensure JSON fields are Python objects (lists/dicts).
    # The DB layer (add_character_card, update_character_card) expects Python objects
    # and will serialize them using _ensure_json_string_from_mixed.
    # If input_data comes from a Pydantic model with proper validators, this might be redundant,
    # but it's a good safeguard if raw dicts are passed.
    for field_name in ["alternate_greetings", "tags"]:
        if field_name in db_data and isinstance(db_data[field_name], str):
            try:
                db_data[field_name] = json.loads(db_data[field_name])
                if not isinstance(db_data[field_name], list): # Should be a list
                    logger.warning(f"Field '{field_name}' was a JSON string but not a list. Resetting to empty list.")
                    db_data[field_name] = []
            except json.JSONDecodeError:
                logger.warning(f"Field '{field_name}' is not valid JSON string. Treating as simple tag list if appropriate or ignoring.")
                # This logic depends on how you want to handle malformed JSON strings for tags/greetings
                # For now, if it's not a list after trying to parse, make it an empty list.
                if not isinstance(db_data[field_name], list):
                     db_data[field_name] = [] # Default to empty list for safety
        elif field_name in db_data and db_data[field_name] is None:
             db_data[field_name] = [] # Store None as empty list

    if "extensions" in db_data and isinstance(db_data["extensions"], str):
        try:
            db_data["extensions"] = json.loads(db_data["extensions"])
            if not isinstance(db_data["extensions"], dict): # Should be a dict
                logger.warning("Field 'extensions' was a JSON string but not a dict. Resetting to empty dict.")
                db_data["extensions"] = {}
        except json.JSONDecodeError:
            logger.warning("Field 'extensions' is not valid JSON string. Resetting to empty dict.")
            db_data["extensions"] = {}
    elif "extensions" in db_data and db_data["extensions"] is None:
        db_data["extensions"] = {} # Store None as empty dict

    return db_data


def create_new_character_from_data(db: CharactersRAGDB, character_payload: Dict[str, Any]) -> Optional[int]:
    """Create a new character with the provided data."""
    try:
        prepared_data = _prepare_character_data_for_db_storage(character_payload)
        char_id = db.add_character_card(prepared_data)
        logger.info(f"Successfully created character with ID: {char_id}")
        return char_id
    except ConflictError as e:
        logger.error(f"Conflict error creating character: {e}")
        raise
    except InputError as e:
        logger.error(f"Input error creating character: {e}")
        raise
    except CharactersRAGDBError as e:
        logger.error(f"Database error creating character: {e}")
        return None
    except Exception as e:
        logger.error(f"Unexpected error creating character: {e}", exc_info=True)
        return None


def get_character_details(db: CharactersRAGDB, character_id: int) -> Optional[Dict[str, Any]]:
    """Get character details by ID."""
    try:
        character = db.get_character_card_by_id(character_id)
        return character
    except CharactersRAGDBError as e:
        logger.error(f"Database error fetching character {character_id}: {e}")
        return None
    except Exception as e:
        logger.error(f"Unexpected error fetching character {character_id}: {e}", exc_info=True)
        return None


def update_existing_character_details(
    db: CharactersRAGDB,
    character_id: int,
    update_payload: Dict[str, Any],
    expected_version: int
) -> bool:
    """Update an existing character with version checking."""
    try:
        prepared_data = _prepare_character_data_for_db_storage(update_payload, is_update=True)
        success = db.update_character_card(character_id, prepared_data, expected_version)
        if success:
            logger.info(f"Successfully updated character {character_id}")
        else:
            logger.warning(f"Failed to update character {character_id} - version mismatch or not found")
        return success
    except ConflictError as e:
        logger.error(f"Conflict error updating character {character_id}: {e}")
        raise
    except InputError as e:
        logger.error(f"Input error updating character {character_id}: {e}")
        raise
    except CharactersRAGDBError as e:
        logger.error(f"Database error updating character {character_id}: {e}")
        return False
    except Exception as e:
        logger.error(f"Unexpected error updating character {character_id}: {e}", exc_info=True)
        return False


def delete_character_from_db(db: CharactersRAGDB, character_id: int, expected_version: int) -> bool:
    """Delete a character from the database."""
    try:
        success = db.delete_character_card(character_id, expected_version)
        if success:
            logger.info(f"Successfully deleted character {character_id}")
        else:
            logger.warning(f"Failed to delete character {character_id} - version mismatch or not found")
        return success
    except CharactersRAGDBError as e:
        logger.error(f"Database error deleting character {character_id}: {e}")
        return False
    except Exception as e:
        logger.error(f"Unexpected error deleting character {character_id}: {e}", exc_info=True)
        return False


def search_characters_by_query_text(
    db: CharactersRAGDB,
    query: str,
    limit: int = 10,
    offset: int = 0
) -> List[Dict[str, Any]]:
    """Search for characters by text query."""
    try:
        results = db.search_character_cards(query, limit, offset)
        logger.info(f"Character search for '{query}' returned {len(results)} results")
        return results
    except CharactersRAGDBError as e:
        logger.error(f"Database error searching characters: {e}")
        return []
    except Exception as e:
        logger.error(f"Unexpected error searching characters: {e}", exc_info=True)
        return []


def load_character_and_image(
    db: CharactersRAGDB,
    character_id: int
) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    """Load character data and image."""
    try:
        character = db.get_character_card_by_id(character_id)
        if not character:
            logger.warning(f"Character {character_id} not found")
            return None, None
        
        # Convert image bytes to base64 if present
        image_base64 = None
        if character.get('image'):
            try:
                image_base64 = base64.b64encode(character['image']).decode('utf-8')
                image_base64 = f"data:image/webp;base64,{image_base64}"
            except Exception as e:
                logger.error(f"Error encoding character image: {e}")
        
        return character, image_base64
    except CharactersRAGDBError as e:
        logger.error(f"Database error loading character {character_id}: {e}")
        return None, None
    except Exception as e:
        logger.error(f"Unexpected error loading character {character_id}: {e}", exc_info=True)
        return None, None


def load_character_wrapper(
    db: CharactersRAGDB,
    character_name: Optional[str] = None,
    character_id: Optional[int] = None
) -> Optional[Dict[str, Any]]:
    """Wrapper to load character by name or ID."""
    try:
        if character_id:
            return db.get_character_card_by_id(character_id)
        elif character_name:
            # Search by name and return first match
            results = db.search_character_cards(character_name, limit=1)
            if results:
                return results[0]
        
        logger.warning("No character_name or character_id provided")
        return None
    except CharactersRAGDBError as e:
        logger.error(f"Database error in load_character_wrapper: {e}")
        return None
    except Exception as e:
        logger.error(f"Unexpected error in load_character_wrapper: {e}", exc_info=True)
        return None