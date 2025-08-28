# characters.py
# Description:
#
# Imports
import base64
import json
import pathlib
from datetime import datetime
from typing import List, Union, Any, Dict, Optional
#
# Third-party Libraries
from fastapi import HTTPException, Depends, Query, UploadFile, File, APIRouter, Path as FastAPIPath
from loguru import logger
from pydantic import BaseModel, Field, field_validator
from pydantic import ValidationInfo
from pydantic_core.core_schema import FieldValidationInfo
from starlette import status
#
# Local Imports
from tldw_Server_API.app.api.v1.API_Deps.ChaCha_Notes_DB_Deps import get_chacha_db_for_user
from tldw_Server_API.app.core.AuthNZ.User_DB_Handling import get_request_user, User
from tldw_Server_API.app.core.Character_Chat.character_rate_limiter import get_character_rate_limiter
from tldw_Server_API.app.api.v1.schemas.character_schemas import CharacterResponse, CharacterImportResponse, \
    CharacterCreate, CharacterUpdate, DeletionResponse
from tldw_Server_API.app.api.v1.schemas.world_book_schemas import (
    WorldBookCreate, WorldBookUpdate, WorldBookResponse, WorldBookWithEntries,
    WorldBookListResponse, WorldBookEntryCreate, WorldBookEntryUpdate,
    WorldBookEntryResponse, EntryListResponse, CharacterWorldBookAttachment,
    CharacterWorldBookResponse, ProcessContextRequest, ProcessContextResponse,
    WorldBookImportRequest, WorldBookImportResponse, WorldBookExport,
    WorldBookStatistics, BulkEntryOperation, BulkOperationResponse
)
from tldw_Server_API.app.core.Character_Chat.Character_Chat_Lib import import_and_save_character_from_file, \
    search_characters_by_query_text, delete_character_from_db, get_character_details, update_existing_character_details, \
    create_new_character_from_data
from tldw_Server_API.app.core.Character_Chat.world_book_manager import WorldBookService
from tldw_Server_API.app.core.DB_Management.ChaChaNotes_DB import CharactersRAGDB, CharactersRAGDBError
from tldw_Server_API.app.core.DB_Management.ChaChaNotes_DB import ConflictError, InputError, CharactersRAGDBError
#
#######################################################################################################################
#
# Functions:



# --- Router ---
router = APIRouter()


# --- Helper Functions (Keep _convert_db_char_to_response_model as is) ---
def _convert_db_char_to_response_model(char_dict_from_db: Dict[str, Any]) -> CharacterResponse:
    response_data = char_dict_from_db.copy()
    if response_data.get('image') and isinstance(response_data['image'], bytes):
        try:
            response_data['image_base64'] = base64.b64encode(response_data['image']).decode('utf-8')
            response_data['image_present'] = True
        except Exception as e:
            logger.error(f"Error encoding image for char {response_data.get('id')}: {e}")
            response_data['image_base64'] = None;
            response_data['image_present'] = False
    else:
        response_data['image_base64'] = None
        response_data['image_present'] = bool(
            response_data.get('image') and isinstance(response_data.get('image'), bytes))
    for field_name in ["alternate_greetings", "tags", "extensions"]:
        value = response_data.get(field_name)
        if isinstance(value, str):  # Already deserialized by DB layer if stored as JSON text
            pass  # Should be Python objects now from DB layer
    response_data.pop('image', None)
    return CharacterResponse.model_validate(response_data)


# --- API Endpoints ---

@router.post("/import", response_model=CharacterImportResponse, summary="Import character card file",
             tags=["Characters"], status_code=status.HTTP_201_CREATED)
async def import_character_from_file_endpoint(
        character_file: UploadFile = File(..., description="Character card file (PNG, WEBP, JSON, MD)."),
        db: CharactersRAGDB = Depends(get_chacha_db_for_user),
        current_user: User = Depends(get_request_user)
):
    try:
        file_content_bytes = await character_file.read()
        if not file_content_bytes:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Uploaded file is empty.")

        # Check rate limits
        rate_limiter = get_character_rate_limiter()
        await rate_limiter.check_rate_limit(current_user.id, "character_import")
        rate_limiter.check_import_size(len(file_content_bytes))
        
        # Check character count limit
        existing_chars = db.list_character_cards(limit=10000)  # Get count
        await rate_limiter.check_character_limit(current_user.id, len(existing_chars))

        logger.info(f"API: Attempting to import character from file: {character_file.filename}")

        # import_and_save_character_from_file now raises ConflictError if name exists
        # and InputError/CharactersRAGDBError for other issues.
        char_id = import_and_save_character_from_file(db, file_content_bytes)

        # If no exception, char_id should be valid
        imported_char_db = db.get_character_card_by_id(char_id)  # type: ignore
        if not imported_char_db:
            raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                                detail="Character imported but could not be retrieved.")

        response_char_model = _convert_db_char_to_response_model(imported_char_db)
        return CharacterImportResponse(
            message=f"Character '{response_char_model.name}' processed successfully (ID: {char_id}).",
            character=response_char_model
        )
    except ConflictError as e:  # Character with same name already exists
        # The library function might return the ID of the existing char if we want that behavior.
        # For now, let's assume ConflictError means it tried to add but failed.
        # If import_and_save_character_from_file returns existing ID on conflict, API status code could be 200.
        # The current lib function returns the existing ID on conflict.
        logger.warning(f"Conflict during import: {e}")
        # Try to retrieve the conflicting character if the error message provides enough info or if the lib returned an ID
        # This part needs careful alignment with how `import_and_save_character_from_file` signals "already exists"
        # If it returns the existing ID, then the initial `char_id` would be that.
        existing_char_id_from_conflict = None
        if hasattr(e, 'entity_id') and isinstance(e.entity_id, int):  # If ConflictError has the ID
            existing_char_id_from_conflict = e.entity_id
        elif isinstance(e.entity_id, str):  # If entity_id is the name
            existing_char_obj = db.get_character_card_by_name(e.entity_id)
            if existing_char_obj: existing_char_id_from_conflict = existing_char_obj['id']

        if existing_char_id_from_conflict:
            existing_char_db = db.get_character_card_by_id(existing_char_id_from_conflict)
            if existing_char_db:
                return CharacterImportResponse(
                    message=f"Character '{existing_char_db['name']}' already exists (ID: {existing_char_id_from_conflict}). Details provided.",
                    character=_convert_db_char_to_response_model(existing_char_db)
                )  # Consider HTTP 200 OK for this case

        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=str(e))

    except (InputError, CharactersRAGDBError) as e:
        logger.error(f"Error during character import: {e}")
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Unexpected error during character import: {e}", exc_info=True)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                            detail="An unexpected error occurred during file import.")
    finally:
        await character_file.close()


@router.get("/", response_model=List[CharacterResponse], summary="List characters", tags=["Characters"])
async def list_all_characters(  # Renamed from list_characters to avoid conflict with Python's list
        db: CharactersRAGDB = Depends(get_chacha_db_for_user),
        limit: int = Query(100, ge=1, le=1000),
        offset: int = Query(0, ge=0)
):
    try:
        # Using the interop library function that calls db.list_character_cards
        # but get_character_list_for_ui returns simplified data. We need full data here.
        raw_cards = db.list_character_cards(limit=limit, offset=offset)  # Direct DB call for full data
        return [_convert_db_char_to_response_model(card) for card in raw_cards]
    except CharactersRAGDBError as e:
        logger.error(f"DB error listing characters: {e}", exc_info=True)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))
    except Exception as e:
        logger.error(f"Unexpected error listing characters: {e}", exc_info=True)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="An unexpected error occurred.")


@router.get("/rate-limit-status", summary="Get rate limit status", tags=["Characters"])
async def get_rate_limit_status(
    current_user: User = Depends(get_request_user)
):
    """Get current rate limit usage statistics for the authenticated user."""
    rate_limiter = get_character_rate_limiter()
    stats = await rate_limiter.get_usage_stats(current_user.id)
    return stats


@router.post("/", response_model=CharacterResponse, status_code=status.HTTP_201_CREATED, summary="Create character",
             tags=["Characters"])
async def create_new_character_endpoint(
        character_data: CharacterCreate,
        db: CharactersRAGDB = Depends(get_chacha_db_for_user),
        current_user: User = Depends(get_request_user)
):
    try:
        # Check rate limits
        rate_limiter = get_character_rate_limiter()
        await rate_limiter.check_rate_limit(current_user.id, "character_create")
        
        # Check character count limit
        existing_chars = db.list_character_cards(limit=10000)
        await rate_limiter.check_character_limit(current_user.id, len(existing_chars))
        
        # The Pydantic model CharacterCreate ensures 'name' is present.
        # The interop function create_new_character_from_data handles image_base64 etc.
        # and calls db.add_character_card.
        # It will raise ConflictError if name exists, InputError for bad data.

        # Convert Pydantic model to dict for the interop library function
        payload_dict = character_data.model_dump(exclude_unset=False)  # include all fields

        char_id = create_new_character_from_data(db, payload_dict)

        if not char_id:  # Should be caught by exceptions in lib layer
            raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                                detail="Failed to create character (no ID returned).")

        created_char_db = get_character_details(db, char_id)  # Use interop get
        if not created_char_db:
            raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                                detail="Failed to retrieve character after creation.")
        return _convert_db_char_to_response_model(created_char_db)
    except (InputError, ConflictError) as e:  # Propagated from lib
        status_code = status.HTTP_400_BAD_REQUEST if isinstance(e, InputError) else status.HTTP_409_CONFLICT
        logger.warning(f"Error creating character: {e} (Status: {status_code})")
        raise HTTPException(status_code=status_code, detail=str(e))
    except CharactersRAGDBError as e:
        logger.error(f"DB error creating character: {e}", exc_info=True)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Unexpected error creating character: {e}", exc_info=True)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="An unexpected error occurred.")


@router.get("/{character_id}", response_model=CharacterResponse, summary="Get character by ID", tags=["Characters"])
async def get_character_by_id_endpoint(  # Renamed from get_character
        character_id: int = FastAPIPath(..., description="ID of the character.", gt=0),
        db: CharactersRAGDB = Depends(get_chacha_db_for_user)
):
    try:
        char_db = get_character_details(db, character_id)  # Use interop get
        if not char_db:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND,
                                detail=f"Character with ID {character_id} not found.")
        return _convert_db_char_to_response_model(char_db)
    except CharactersRAGDBError as e:
        logger.error(f"DB error getting character {character_id}: {e}", exc_info=True)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Unexpected error getting character {character_id}: {e}", exc_info=True)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="An unexpected error occurred.")


@router.put("/{character_id}", response_model=CharacterResponse, summary="Update character", tags=["Characters"])
async def update_character_endpoint(  # Renamed from update_character
        update_data: CharacterUpdate,
        character_id: int = FastAPIPath(..., description="ID of the character to update.", gt=0),
        expected_version: int = Query(...,
                                      description="Expected current version of the character for optimistic locking."),
        db: CharactersRAGDB = Depends(get_chacha_db_for_user)
):
    try:
        # Check if character exists before attempting update, to provide 404 early
        # The lib function update_existing_character_details might also do this or rely on DB layer
        current_char_for_check = get_character_details(db, character_id)
        if not current_char_for_check:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND,
                                detail=f"Character with ID {character_id} not found for update.")
        # Validate expected_version against actual current version
        if current_char_for_check['version'] != expected_version:
            raise HTTPException(status_code=status.HTTP_409_CONFLICT,
                                detail=f"Version mismatch. Expected {expected_version}, found {current_char_for_check['version']}. Please refresh and try again.")

        payload_dict = update_data.model_dump(exclude_unset=True)  # Only include fields that were set

        success = update_existing_character_details(db, character_id, payload_dict, expected_version)

        if not success:  # Should be caught by specific exceptions from lib
            raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                                detail="Failed to update character (unexpected boolean failure).")

        updated_char_db = get_character_details(db, character_id)
        if not updated_char_db:  # Should not happen if update was successful
            raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                                detail="Failed to retrieve character after update.")
        return _convert_db_char_to_response_model(updated_char_db)

    except (InputError, ConflictError) as e:
        status_code = status.HTTP_400_BAD_REQUEST if isinstance(e, InputError) else status.HTTP_409_CONFLICT
        logger.warning(f"Error updating character {character_id}: {e} (Status: {status_code})")
        raise HTTPException(status_code=status_code, detail=str(e))
    except CharactersRAGDBError as e:
        logger.error(f"DB error updating character {character_id}: {e}", exc_info=True)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Unexpected error updating character {character_id}: {e}", exc_info=True)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="An unexpected error occurred.")


@router.delete("/{character_id}", response_model=DeletionResponse, summary="Delete character", tags=["Characters"])
async def delete_character_endpoint(  # Renamed from delete_character
        character_id: int = FastAPIPath(..., description="ID of the character to delete.", gt=0),
        expected_version: int = Query(...,
                                      description="Expected current version of the character for optimistic locking."),
        db: CharactersRAGDB = Depends(get_chacha_db_for_user)
):
    try:
        # Check existence and get name for response message before delete attempt
        char_to_delete = get_character_details(db, character_id)
        if not char_to_delete:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND,
                                detail=f"Character with ID {character_id} not found for deletion.")

        # Validate expected_version here before calling lib, for clearer HTTP 409
        if char_to_delete['version'] != expected_version:
            raise HTTPException(status_code=status.HTTP_409_CONFLICT,
                                detail=f"Version mismatch for deletion. Expected {expected_version}, found {char_to_delete['version']}. Please refresh.")

        char_name = char_to_delete.get('name', 'N/A')
        success = delete_character_from_db(db, character_id, expected_version)

        if not success:  # Should be caught by specific exceptions from lib
            raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                                detail="Failed to delete character (unexpected boolean failure).")

        return DeletionResponse(
            message=f"Character '{char_name}' (ID: {character_id}) soft-deleted.",
            character_id=character_id
        )
    except ConflictError as e:  # From lib (e.g. if somehow version changed between API check and lib call, or FK issue)
        logger.warning(f"Conflict error deleting character {character_id}: {e}")
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=str(e))
    except CharactersRAGDBError as e:
        logger.error(f"DB error deleting character {character_id}: {e}", exc_info=True)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Unexpected error deleting character {character_id}: {e}", exc_info=True)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="An unexpected error occurred.")


@router.get("/search/", response_model=List[CharacterResponse], summary="Search characters", tags=["Characters"])
async def search_characters_endpoint(
        query: str = Query(..., description="Search term for character name, description, etc."),
        limit: int = Query(10, ge=1, le=100),
        db: CharactersRAGDB = Depends(get_chacha_db_for_user)
):
    """
    Searches for characters based on a query string.
    The search is performed against FTS-indexed fields in the database.
    """
    if not query.strip():
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Search query cannot be empty.")
    try:
        results_db = search_characters_by_query_text(db, query, limit=limit)
        return [_convert_db_char_to_response_model(card) for card in results_db]
    except CharactersRAGDBError as e:
        logger.error(f"DB error searching characters for '{query}': {e}", exc_info=True)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))
    except Exception as e:
        logger.error(f"Unexpected error searching characters for '{query}': {e}", exc_info=True)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="An unexpected error occurred.")

# --- World Book Endpoints ---

@router.post("/world-books", response_model=WorldBookResponse, status_code=status.HTTP_201_CREATED,
             summary="Create a world book", tags=["World Books"])
async def create_world_book(
        world_book: WorldBookCreate,
        db: CharactersRAGDB = Depends(get_chacha_db_for_user)
):
    """Create a new world book for the user."""
    try:
        service = WorldBookService(db)
        world_book_id = service.create_world_book(
            name=world_book.name,
            description=world_book.description,
            scan_depth=world_book.scan_depth,
            token_budget=world_book.token_budget,
            recursive_scanning=world_book.recursive_scanning,
            enabled=world_book.enabled
        )
        
        created_book = service.get_world_book(world_book_id)
        if not created_book:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to retrieve world book after creation"
            )
        
        # Add entry count
        entries = service.get_entries(world_book_id, enabled_only=False)
        created_book['entry_count'] = len(entries)
        
        return WorldBookResponse(**created_book)
        
    except InputError as e:
        logger.warning(f"Input error creating world book: {e}")
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))
    except ConflictError as e:
        logger.warning(f"Conflict creating world book: {e}")
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=str(e))
    except CharactersRAGDBError as e:
        logger.error(f"DB error creating world book: {e}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))
    except Exception as e:
        logger.error(f"Unexpected error creating world book: {e}", exc_info=True)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="An unexpected error occurred")


@router.get("/world-books", response_model=WorldBookListResponse, summary="List world books", tags=["World Books"])
async def list_world_books(
        include_disabled: bool = Query(False, description="Include disabled world books"),
        db: CharactersRAGDB = Depends(get_chacha_db_for_user)
):
    """List all world books for the user."""
    try:
        service = WorldBookService(db)
        books = service.list_world_books(include_disabled=include_disabled)
        
        # Add entry counts
        for book in books:
            entries = service.get_entries(book['id'], enabled_only=False)
            book['entry_count'] = len(entries)
        
        # Convert to response models
        book_responses = [WorldBookResponse(**book) for book in books]
        
        # Calculate statistics
        enabled_count = sum(1 for b in book_responses if b.enabled)
        disabled_count = len(book_responses) - enabled_count
        
        return WorldBookListResponse(
            world_books=book_responses,
            total=len(book_responses),
            enabled_count=enabled_count,
            disabled_count=disabled_count
        )
        
    except CharactersRAGDBError as e:
        logger.error(f"DB error listing world books: {e}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))
    except Exception as e:
        logger.error(f"Unexpected error listing world books: {e}", exc_info=True)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="An unexpected error occurred")


@router.get("/world-books/{world_book_id}", response_model=WorldBookWithEntries,
            summary="Get world book with entries", tags=["World Books"])
async def get_world_book(
        world_book_id: int = FastAPIPath(..., description="World book ID", gt=0),
        db: CharactersRAGDB = Depends(get_chacha_db_for_user)
):
    """Get a world book with all its entries."""
    try:
        service = WorldBookService(db)
        book = service.get_world_book(world_book_id)
        
        if not book:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"World book with ID {world_book_id} not found"
            )
        
        # Get entries
        entries = service.get_entries(world_book_id, enabled_only=False)
        book['entry_count'] = len(entries)
        
        # Convert entries to response models
        entry_responses = []
        for entry in entries:
            entry_dict = entry.to_dict()
            entry_dict['created_at'] = book['created_at']  # Use book's timestamp as fallback
            entry_dict['last_modified'] = book['last_modified']
            entry_responses.append(WorldBookEntryResponse(**entry_dict))
        
        return WorldBookWithEntries(
            **book,
            entries=entry_responses
        )
        
    except HTTPException:
        raise
    except CharactersRAGDBError as e:
        logger.error(f"DB error getting world book {world_book_id}: {e}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))
    except Exception as e:
        logger.error(f"Unexpected error getting world book {world_book_id}: {e}", exc_info=True)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="An unexpected error occurred")


@router.put("/world-books/{world_book_id}", response_model=WorldBookResponse,
            summary="Update world book", tags=["World Books"])
async def update_world_book(
        world_book_id: int,
        update_data: WorldBookUpdate,
        db: CharactersRAGDB = Depends(get_chacha_db_for_user)
):
    """Update a world book."""
    try:
        service = WorldBookService(db)
        
        # Check if exists
        existing = service.get_world_book(world_book_id)
        if not existing:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"World book with ID {world_book_id} not found"
            )
        
        # Update
        success = service.update_world_book(
            world_book_id=world_book_id,
            name=update_data.name,
            description=update_data.description,
            scan_depth=update_data.scan_depth,
            token_budget=update_data.token_budget,
            recursive_scanning=update_data.recursive_scanning,
            enabled=update_data.enabled
        )
        
        if not success:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to update world book"
            )
        
        # Get updated book
        updated_book = service.get_world_book(world_book_id)
        entries = service.get_entries(world_book_id, enabled_only=False)
        updated_book['entry_count'] = len(entries)
        
        return WorldBookResponse(**updated_book)
        
    except HTTPException:
        raise
    except ConflictError as e:
        logger.warning(f"Conflict updating world book {world_book_id}: {e}")
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=str(e))
    except CharactersRAGDBError as e:
        logger.error(f"DB error updating world book {world_book_id}: {e}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))
    except Exception as e:
        logger.error(f"Unexpected error updating world book {world_book_id}: {e}", exc_info=True)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="An unexpected error occurred")


@router.delete("/world-books/{world_book_id}", response_model=DeletionResponse,
               summary="Delete world book", tags=["World Books"])
async def delete_world_book(
        world_book_id: int = FastAPIPath(..., description="World book ID", gt=0),
        hard_delete: bool = Query(False, description="Permanently delete (default is soft delete)"),
        db: CharactersRAGDB = Depends(get_chacha_db_for_user)
):
    """Delete a world book."""
    try:
        service = WorldBookService(db)
        
        # Check if exists and get name
        book = service.get_world_book(world_book_id)
        if not book:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"World book with ID {world_book_id} not found"
            )
        
        book_name = book['name']
        success = service.delete_world_book(world_book_id, hard_delete=hard_delete)
        
        if not success:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to delete world book"
            )
        
        delete_type = "permanently deleted" if hard_delete else "soft-deleted"
        return DeletionResponse(
            message=f"World book '{book_name}' (ID: {world_book_id}) {delete_type}",
            character_id=world_book_id  # Reusing field name from character deletion
        )
        
    except HTTPException:
        raise
    except CharactersRAGDBError as e:
        logger.error(f"DB error deleting world book {world_book_id}: {e}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))
    except Exception as e:
        logger.error(f"Unexpected error deleting world book {world_book_id}: {e}", exc_info=True)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="An unexpected error occurred")


# --- World Book Entry Endpoints ---

@router.post("/world-books/{world_book_id}/entries", response_model=WorldBookEntryResponse,
             status_code=status.HTTP_201_CREATED, summary="Add entry to world book", tags=["World Books"])
async def add_world_book_entry(
        world_book_id: int,
        entry: WorldBookEntryCreate,
        db: CharactersRAGDB = Depends(get_chacha_db_for_user)
):
    """Add an entry to a world book."""
    try:
        service = WorldBookService(db)
        
        # Check if world book exists
        book = service.get_world_book(world_book_id)
        if not book:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"World book with ID {world_book_id} not found"
            )
        
        entry_id = service.add_entry(
            world_book_id=world_book_id,
            keywords=entry.keywords,
            content=entry.content,
            priority=entry.priority,
            enabled=entry.enabled,
            case_sensitive=entry.case_sensitive,
            regex_match=entry.regex_match,
            whole_word_match=entry.whole_word_match,
            metadata=entry.metadata
        )
        
        # Get the created entry
        entries = service.get_entries(world_book_id, enabled_only=False)
        created_entry = next((e for e in entries if e.entry_id == entry_id), None)
        
        if not created_entry:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to retrieve entry after creation"
            )
        
        entry_dict = created_entry.to_dict()
        entry_dict['created_at'] = book['created_at']
        entry_dict['last_modified'] = book['last_modified']
        
        return WorldBookEntryResponse(**entry_dict)
        
    except HTTPException:
        raise
    except InputError as e:
        logger.warning(f"Input error adding entry to world book {world_book_id}: {e}")
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))
    except CharactersRAGDBError as e:
        logger.error(f"DB error adding entry to world book {world_book_id}: {e}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))
    except Exception as e:
        logger.error(f"Unexpected error adding entry to world book {world_book_id}: {e}", exc_info=True)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="An unexpected error occurred")


@router.get("/world-books/{world_book_id}/entries", response_model=EntryListResponse,
            summary="List world book entries", tags=["World Books"])
async def list_world_book_entries(
        world_book_id: int = FastAPIPath(..., description="World book ID", gt=0),
        enabled_only: bool = Query(False, description="Only show enabled entries"),
        db: CharactersRAGDB = Depends(get_chacha_db_for_user)
):
    """List all entries in a world book."""
    try:
        service = WorldBookService(db)
        
        # Check if world book exists
        book = service.get_world_book(world_book_id)
        if not book:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"World book with ID {world_book_id} not found"
            )
        
        entries = service.get_entries(world_book_id, enabled_only=enabled_only)
        
        # Convert to response models
        entry_responses = []
        for entry in entries:
            entry_dict = entry.to_dict()
            entry_dict['created_at'] = book['created_at']
            entry_dict['last_modified'] = book['last_modified']
            entry_responses.append(WorldBookEntryResponse(**entry_dict))
        
        return EntryListResponse(
            entries=entry_responses,
            total=len(entry_responses),
            world_book_id=world_book_id
        )
        
    except HTTPException:
        raise
    except CharactersRAGDBError as e:
        logger.error(f"DB error listing entries for world book {world_book_id}: {e}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))
    except Exception as e:
        logger.error(f"Unexpected error listing entries for world book {world_book_id}: {e}", exc_info=True)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="An unexpected error occurred")


@router.put("/world-books/entries/{entry_id}", response_model=WorldBookEntryResponse,
            summary="Update world book entry", tags=["World Books"])
async def update_world_book_entry(
        entry_id: int,
        update_data: WorldBookEntryUpdate,
        db: CharactersRAGDB = Depends(get_chacha_db_for_user)
):
    """Update a world book entry."""
    try:
        service = WorldBookService(db)
        
        success = service.update_entry(
            entry_id=entry_id,
            keywords=update_data.keywords,
            content=update_data.content,
            priority=update_data.priority,
            enabled=update_data.enabled,
            case_sensitive=update_data.case_sensitive,
            regex_match=update_data.regex_match,
            whole_word_match=update_data.whole_word_match,
            metadata=update_data.metadata
        )
        
        if not success:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Entry with ID {entry_id} not found"
            )
        
        # Find the updated entry
        # We need to search through all world books to find this entry
        all_entries = service.get_entries(enabled_only=False)
        updated_entry = next((e for e in all_entries if e.entry_id == entry_id), None)
        
        if not updated_entry:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to retrieve entry after update"
            )
        
        # Get the world book for timestamps
        book = service.get_world_book(updated_entry.world_book_id)
        entry_dict = updated_entry.to_dict()
        entry_dict['created_at'] = book['created_at'] if book else datetime.utcnow()
        entry_dict['last_modified'] = book['last_modified'] if book else datetime.utcnow()
        
        return WorldBookEntryResponse(**entry_dict)
        
    except HTTPException:
        raise
    except InputError as e:
        logger.warning(f"Input error updating entry {entry_id}: {e}")
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))
    except CharactersRAGDBError as e:
        logger.error(f"DB error updating entry {entry_id}: {e}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))
    except Exception as e:
        logger.error(f"Unexpected error updating entry {entry_id}: {e}", exc_info=True)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="An unexpected error occurred")


@router.delete("/world-books/entries/{entry_id}", response_model=DeletionResponse,
               summary="Delete world book entry", tags=["World Books"])
async def delete_world_book_entry(
        entry_id: int = FastAPIPath(..., description="Entry ID", gt=0),
        db: CharactersRAGDB = Depends(get_chacha_db_for_user)
):
    """Delete a world book entry."""
    try:
        service = WorldBookService(db)
        
        success = service.delete_entry(entry_id)
        
        if not success:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Entry with ID {entry_id} not found"
            )
        
        return DeletionResponse(
            message=f"World book entry (ID: {entry_id}) deleted",
            character_id=entry_id  # Reusing field name
        )
        
    except HTTPException:
        raise
    except CharactersRAGDBError as e:
        logger.error(f"DB error deleting entry {entry_id}: {e}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))
    except Exception as e:
        logger.error(f"Unexpected error deleting entry {entry_id}: {e}", exc_info=True)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="An unexpected error occurred")


# --- Character-World Book Association Endpoints ---

@router.post("/{character_id}/world-books", response_model=CharacterWorldBookResponse,
             status_code=status.HTTP_201_CREATED, summary="Attach world book to character", tags=["World Books"])
async def attach_world_book_to_character(
        character_id: int,
        attachment: CharacterWorldBookAttachment,
        db: CharactersRAGDB = Depends(get_chacha_db_for_user)
):
    """Attach a world book to a character."""
    try:
        # Check if character exists
        char = get_character_details(db, character_id)
        if not char:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Character with ID {character_id} not found"
            )
        
        service = WorldBookService(db)
        
        # Check if world book exists
        book = service.get_world_book(attachment.world_book_id)
        if not book:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"World book with ID {attachment.world_book_id} not found"
            )
        
        success = service.attach_to_character(
            character_id=character_id,
            world_book_id=attachment.world_book_id,
            enabled=attachment.enabled,
            priority=attachment.priority
        )
        
        if not success:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to attach world book to character"
            )
        
        # Get entry count
        entries = service.get_entries(attachment.world_book_id, enabled_only=False)
        book['entry_count'] = len(entries)
        
        return CharacterWorldBookResponse(
            **book,
            attachment_enabled=attachment.enabled,
            attachment_priority=attachment.priority
        )
        
    except HTTPException:
        raise
    except CharactersRAGDBError as e:
        logger.error(f"DB error attaching world book {attachment.world_book_id} to character {character_id}: {e}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))
    except Exception as e:
        logger.error(f"Unexpected error attaching world book to character: {e}", exc_info=True)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="An unexpected error occurred")


@router.delete("/{character_id}/world-books/{world_book_id}", response_model=DeletionResponse,
               summary="Detach world book from character", tags=["World Books"])
async def detach_world_book_from_character(
        character_id: int = FastAPIPath(..., description="Character ID", gt=0),
        world_book_id: int = FastAPIPath(..., description="World book ID", gt=0),
        db: CharactersRAGDB = Depends(get_chacha_db_for_user)
):
    """Detach a world book from a character."""
    try:
        service = WorldBookService(db)
        
        success = service.detach_from_character(character_id, world_book_id)
        
        if not success:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Attachment between character {character_id} and world book {world_book_id} not found"
            )
        
        return DeletionResponse(
            message=f"World book {world_book_id} detached from character {character_id}",
            character_id=world_book_id  # Reusing field name
        )
        
    except HTTPException:
        raise
    except CharactersRAGDBError as e:
        logger.error(f"DB error detaching world book {world_book_id} from character {character_id}: {e}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))
    except Exception as e:
        logger.error(f"Unexpected error detaching world book from character: {e}", exc_info=True)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="An unexpected error occurred")


@router.get("/{character_id}/world-books", response_model=List[CharacterWorldBookResponse],
            summary="List character's world books", tags=["World Books"])
async def get_character_world_books(
        character_id: int = FastAPIPath(..., description="Character ID", gt=0),
        enabled_only: bool = Query(True, description="Only show enabled attachments"),
        db: CharactersRAGDB = Depends(get_chacha_db_for_user)
):
    """Get all world books attached to a character."""
    try:
        # Check if character exists
        char = get_character_details(db, character_id)
        if not char:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Character with ID {character_id} not found"
            )
        
        service = WorldBookService(db)
        books = service.get_character_world_books(character_id, enabled_only=enabled_only)
        
        # Add entry counts and convert to response models
        response_books = []
        for book in books:
            entries = service.get_entries(book['id'], enabled_only=False)
            book['entry_count'] = len(entries)
            response_books.append(CharacterWorldBookResponse(**book))
        
        return response_books
        
    except HTTPException:
        raise
    except CharactersRAGDBError as e:
        logger.error(f"DB error getting world books for character {character_id}: {e}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))
    except Exception as e:
        logger.error(f"Unexpected error getting character's world books: {e}", exc_info=True)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="An unexpected error occurred")


# --- Processing Endpoints ---

@router.post("/world-books/process", response_model=ProcessContextResponse,
             summary="Process text with world info", tags=["World Books"])
async def process_context_with_world_info(
        request: ProcessContextRequest,
        db: CharactersRAGDB = Depends(get_chacha_db_for_user)
):
    """Process text to find and inject relevant world info."""
    try:
        service = WorldBookService(db)
        
        injected_content, stats = service.process_context(
            text=request.text,
            world_book_ids=request.world_book_ids,
            character_id=request.character_id,
            scan_depth=request.scan_depth,
            token_budget=request.token_budget,
            recursive_scanning=request.recursive_scanning
        )
        
        return ProcessContextResponse(
            injected_content=injected_content,
            entries_matched=stats['entries_matched'],
            tokens_used=stats['tokens_used'],
            books_used=stats['books_used'],
            entry_ids=stats['entry_ids']
        )
        
    except CharactersRAGDBError as e:
        logger.error(f"DB error processing context: {e}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))
    except Exception as e:
        logger.error(f"Unexpected error processing context: {e}", exc_info=True)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="An unexpected error occurred")


# --- Import/Export Endpoints ---

@router.post("/world-books/import", response_model=WorldBookImportResponse,
             status_code=status.HTTP_201_CREATED, summary="Import world book", tags=["World Books"])
async def import_world_book(
        import_data: WorldBookImportRequest,
        db: CharactersRAGDB = Depends(get_chacha_db_for_user)
):
    """Import a world book from external format."""
    try:
        service = WorldBookService(db)
        
        world_book_id = service.import_world_book(
            data={
                "world_book": import_data.world_book,
                "entries": import_data.entries
            },
            merge_on_conflict=import_data.merge_on_conflict
        )
        
        # Get the imported/merged book
        book = service.get_world_book(world_book_id)
        if not book:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to retrieve world book after import"
            )
        
        return WorldBookImportResponse(
            world_book_id=world_book_id,
            name=book['name'],
            entries_imported=len(import_data.entries),
            merged=import_data.merge_on_conflict and book['version'] > 1
        )
        
    except InputError as e:
        logger.warning(f"Input error importing world book: {e}")
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))
    except ConflictError as e:
        logger.warning(f"Conflict importing world book: {e}")
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=str(e))
    except CharactersRAGDBError as e:
        logger.error(f"DB error importing world book: {e}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))
    except Exception as e:
        logger.error(f"Unexpected error importing world book: {e}", exc_info=True)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="An unexpected error occurred")


@router.get("/world-books/{world_book_id}/export", response_model=WorldBookExport,
            summary="Export world book", tags=["World Books"])
async def export_world_book(
        world_book_id: int = FastAPIPath(..., description="World book ID", gt=0),
        db: CharactersRAGDB = Depends(get_chacha_db_for_user)
):
    """Export a world book to external format."""
    try:
        service = WorldBookService(db)
        
        export_data = service.export_world_book(world_book_id)
        
        return WorldBookExport(
            world_book=export_data['world_book'],
            entries=export_data['entries'],
            export_date=datetime.utcnow(),
            format_version="1.0"
        )
        
    except InputError as e:
        logger.warning(f"World book {world_book_id} not found for export: {e}")
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(e))
    except CharactersRAGDBError as e:
        logger.error(f"DB error exporting world book {world_book_id}: {e}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))
    except Exception as e:
        logger.error(f"Unexpected error exporting world book {world_book_id}: {e}", exc_info=True)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="An unexpected error occurred")


# --- Statistics Endpoint ---

@router.get("/world-books/{world_book_id}/statistics", response_model=WorldBookStatistics,
            summary="Get world book statistics", tags=["World Books"])
async def get_world_book_statistics(
        world_book_id: int = FastAPIPath(..., description="World book ID", gt=0),
        db: CharactersRAGDB = Depends(get_chacha_db_for_user)
):
    """Get statistics for a world book."""
    try:
        service = WorldBookService(db)
        
        book = service.get_world_book(world_book_id)
        if not book:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"World book with ID {world_book_id} not found"
            )
        
        entries = service.get_entries(world_book_id, enabled_only=False)
        
        # Calculate statistics
        enabled_entries = sum(1 for e in entries if e.enabled)
        disabled_entries = len(entries) - enabled_entries
        total_keywords = sum(len(e.keywords) for e in entries)
        regex_entries = sum(1 for e in entries if e.regex_match)
        case_sensitive_entries = sum(1 for e in entries if e.case_sensitive)
        
        priorities = [e.priority for e in entries]
        average_priority = sum(priorities) / len(priorities) if priorities else 0.0
        
        total_content_length = sum(len(e.content) for e in entries)
        estimated_tokens = service._estimate_tokens(" ".join(e.content for e in entries))
        
        return WorldBookStatistics(
            world_book_id=world_book_id,
            name=book['name'],
            total_entries=len(entries),
            enabled_entries=enabled_entries,
            disabled_entries=disabled_entries,
            total_keywords=total_keywords,
            regex_entries=regex_entries,
            case_sensitive_entries=case_sensitive_entries,
            average_priority=average_priority,
            total_content_length=total_content_length,
            estimated_tokens=estimated_tokens
        )
        
    except HTTPException:
        raise
    except CharactersRAGDBError as e:
        logger.error(f"DB error getting statistics for world book {world_book_id}: {e}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))
    except Exception as e:
        logger.error(f"Unexpected error getting world book statistics: {e}", exc_info=True)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="An unexpected error occurred")


# --- Bulk Operations Endpoint ---

@router.post("/world-books/entries/bulk", response_model=BulkOperationResponse,
             summary="Bulk operations on entries", tags=["World Books"])
async def bulk_entry_operations(
        operation: BulkEntryOperation,
        db: CharactersRAGDB = Depends(get_chacha_db_for_user)
):
    """Perform bulk operations on world book entries."""
    try:
        service = WorldBookService(db)
        
        affected_count = 0
        failed_ids = []
        
        for entry_id in operation.entry_ids:
            try:
                if operation.operation == "delete":
                    success = service.delete_entry(entry_id)
                elif operation.operation == "enable":
                    success = service.update_entry(entry_id, enabled=True)
                elif operation.operation == "disable":
                    success = service.update_entry(entry_id, enabled=False)
                elif operation.operation == "set_priority" and operation.priority is not None:
                    success = service.update_entry(entry_id, priority=operation.priority)
                else:
                    success = False
                
                if success:
                    affected_count += 1
                else:
                    failed_ids.append(entry_id)
                    
            except Exception as e:
                logger.warning(f"Failed to perform {operation.operation} on entry {entry_id}: {e}")
                failed_ids.append(entry_id)
        
        message = f"Operation '{operation.operation}' completed: {affected_count} entries affected"
        if failed_ids:
            message += f", {len(failed_ids)} failed"
        
        return BulkOperationResponse(
            success=len(failed_ids) == 0,
            affected_count=affected_count,
            failed_ids=failed_ids,
            message=message
        )
        
    except CharactersRAGDBError as e:
        logger.error(f"DB error performing bulk operation: {e}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))
    except Exception as e:
        logger.error(f"Unexpected error performing bulk operation: {e}", exc_info=True)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="An unexpected error occurred")

#
# End of characters.py
#######################################################################################################################
