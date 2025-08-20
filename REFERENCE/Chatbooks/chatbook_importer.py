# chatbook_importer.py
# Description: Service for importing chatbooks/knowledge packs
#
"""
Chatbook Importer
-----------------

Handles the import and validation of chatbooks into the application.
"""

import json
import shutil
import zipfile
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple, Set
from enum import Enum
from loguru import logger

from .chatbook_models import (
    Chatbook, ChatbookManifest, ChatbookContent,
    ContentItem, ContentType, ChatbookVersion
)
from .conflict_resolver import ConflictResolver, ConflictResolution
from .error_handler import ChatbookErrorHandler, ChatbookErrorType, safe_chatbook_operation
from ..DB.ChaChaNotes_DB import CharactersRAGDB
from ..DB.Client_Media_DB_v2 import MediaDatabase
from ..DB.Prompts_DB import PromptsDatabase
from ..DB.RAG_Indexing_DB import RAGIndexingDB
from ..DB.Evals_DB import EvalsDB
from ..Character_Chat.character_card_formats import detect_and_parse_character_card


class ImportStatus:
    """Track import progress and results."""
    def __init__(self):
        self.total_items = 0
        self.processed_items = 0
        self.successful_items = 0
        self.failed_items = 0
        self.skipped_items = 0
        self.errors: List[str] = []
        self.warnings: List[str] = []
        
    def add_error(self, error: str):
        """Add an error message."""
        self.errors.append(error)
        
    def add_warning(self, warning: str):
        """Add a warning message."""
        self.warnings.append(warning)
        
    def to_dict(self) -> dict:
        """Convert status to dictionary."""
        return {
            "total_items": self.total_items,
            "processed_items": self.processed_items,
            "successful_items": self.successful_items,
            "failed_items": self.failed_items,
            "skipped_items": self.skipped_items,
            "errors": self.errors,
            "warnings": self.warnings
        }


class ChatbookImporter:
    """Service for importing chatbooks into the application."""
    
    def __init__(self, db_paths: Dict[str, str]):
        """
        Initialize the chatbook importer.
        
        Args:
            db_paths: Dictionary mapping database names to their paths
        """
        self.db_paths = db_paths
        self.temp_dir = Path.home() / ".local" / "share" / "tldw_cli" / "temp" / "imports"
        self.temp_dir.mkdir(parents=True, exist_ok=True)
        self.conflict_resolver = ConflictResolver()
        
    def preview_chatbook(self, chatbook_path: Path) -> Tuple[Optional[ChatbookManifest], Optional[str]]:
        """
        Preview a chatbook without importing it.
        
        Args:
            chatbook_path: Path to the chatbook file
            
        Returns:
            Tuple of (manifest, error_message)
        """
        try:
            # Extract to temporary directory
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            extract_dir = self.temp_dir / f"preview_{timestamp}"
            
            # Extract archive
            if chatbook_path.suffix == '.zip':
                with zipfile.ZipFile(chatbook_path, 'r') as zf:
                    zf.extractall(extract_dir)
            else:
                return None, "Unsupported chatbook format. Only ZIP files are supported."
            
            # Load manifest
            manifest_path = extract_dir / "manifest.json"
            if not manifest_path.exists():
                shutil.rmtree(extract_dir)
                return None, "Invalid chatbook: manifest.json not found"
            
            with open(manifest_path, 'r', encoding='utf-8') as f:
                manifest_data = json.load(f)
            
            manifest = ChatbookManifest.from_dict(manifest_data)
            
            # Cleanup
            shutil.rmtree(extract_dir)
            
            return manifest, None
            
        except Exception as e:
            logger.error(f"Error previewing chatbook: {e}")
            return None, f"Error previewing chatbook: {str(e)}"
    
    def import_chatbook(
        self,
        chatbook_path: Path,
        content_selections: Optional[Dict[ContentType, List[str]]] = None,
        conflict_resolution: ConflictResolution = ConflictResolution.ASK,
        prefix_imported: bool = False,
        import_media: bool = True,
        import_embeddings: bool = False,
        import_status: Optional[ImportStatus] = None
    ) -> Tuple[bool, str]:
        """
        Import a chatbook into the application.
        
        Args:
            chatbook_path: Path to the chatbook file
            content_selections: Optional dict of content types to specific IDs to import
            conflict_resolution: How to handle conflicts
            prefix_imported: Whether to prefix imported content titles
            import_media: Whether to import media files
            import_embeddings: Whether to import embeddings
            
        Returns:
            Tuple of (success, message)
        """
        logger.info(f"ChatbookImporter.import_chatbook: Starting import of {chatbook_path}")
        logger.info(f"ChatbookImporter.import_chatbook: Options - conflict_resolution={conflict_resolution}, prefix_imported={prefix_imported}, import_media={import_media}, import_embeddings={import_embeddings}")
        status = import_status if import_status else ImportStatus()
        
        try:
            # Extract chatbook
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            extract_dir = self.temp_dir / f"import_{timestamp}"
            
            logger.info(f"Importing chatbook from {chatbook_path}")
            
            # Extract archive
            if chatbook_path.suffix == '.zip':
                with zipfile.ZipFile(chatbook_path, 'r') as zf:
                    zf.extractall(extract_dir)
            else:
                status.add_error("Unsupported chatbook format. Only ZIP files are supported.")
                return False, status
            
            # Load manifest
            manifest_path = extract_dir / "manifest.json"
            logger.info(f"ChatbookImporter.import_chatbook: Looking for manifest at {manifest_path}")
            if not manifest_path.exists():
                logger.error(f"ChatbookImporter.import_chatbook: manifest.json not found")
                status.add_error("Invalid chatbook: manifest.json not found")
                shutil.rmtree(extract_dir)
                return False, status
            
            with open(manifest_path, 'r', encoding='utf-8') as f:
                manifest_data = json.load(f)
            logger.info(f"ChatbookImporter.import_chatbook: Loaded manifest with {len(manifest_data.get('content', {}))} content types")
            
            manifest = ChatbookManifest.from_dict(manifest_data)
            logger.info(f"ChatbookImporter.import_chatbook: Manifest - version {manifest.version}, {manifest.total_conversations} conversations, {manifest.total_notes} notes, {manifest.total_characters} characters, {manifest.total_media_items} media")
            
            # Check version compatibility
            if manifest.version != ChatbookVersion.V1:
                status.add_warning(f"Chatbook version {manifest.version.value} may not be fully compatible")
            
            # Set up content selections
            if content_selections is None:
                # Import everything by default
                content_selections = {}
                for item in manifest.content_items:
                    if item.type not in content_selections:
                        content_selections[item.type] = []
                    content_selections[item.type].append(item.id)
            
            # Count total items to import
            status.total_items = sum(len(ids) for ids in content_selections.values())
            
            # Import each content type
            if ContentType.CHARACTER in content_selections:
                # Import characters first as they may be dependencies
                self._import_characters(
                    extract_dir,
                    manifest,
                    content_selections[ContentType.CHARACTER],
                    conflict_resolution,
                    prefix_imported,
                    status
                )
            
            if ContentType.CONVERSATION in content_selections:
                self._import_conversations(
                    extract_dir,
                    manifest,
                    content_selections[ContentType.CONVERSATION],
                    conflict_resolution,
                    prefix_imported,
                    status
                )
            
            if ContentType.NOTE in content_selections:
                self._import_notes(
                    extract_dir,
                    manifest,
                    content_selections[ContentType.NOTE],
                    conflict_resolution,
                    prefix_imported,
                    status
                )
            
            if ContentType.PROMPT in content_selections:
                self._import_prompts(
                    extract_dir,
                    manifest,
                    content_selections[ContentType.PROMPT],
                    conflict_resolution,
                    prefix_imported,
                    status
                )
            
            if import_media and ContentType.MEDIA in content_selections:
                self._import_media(
                    extract_dir,
                    manifest,
                    content_selections[ContentType.MEDIA],
                    conflict_resolution,
                    status
                )
            
            # Cleanup
            shutil.rmtree(extract_dir)
            
            # Success if we processed items without fatal errors
            # This includes both imported and skipped items
            success = (status.successful_items + status.skipped_items) > 0 or status.total_items == 0
            
            if success:
                if status.successful_items > 0:
                    details = []
                    if status.skipped_items > 0:
                        details.append(f"{status.skipped_items} skipped")
                    if status.failed_items > 0:
                        details.append(f"{status.failed_items} failed")
                    
                    message = f"Successfully imported {status.successful_items}/{status.total_items} items"
                    if details:
                        message += f" ({', '.join(details)})"
                elif status.skipped_items > 0:
                    message = f"Skipped {status.skipped_items}/{status.total_items} items due to conflicts"
                    if status.failed_items > 0:
                        message += f" ({status.failed_items} failed)"
                else:
                    message = "No items to import"
                logger.info(message)
            else:
                message = f"Failed to import any items from chatbook"
                logger.error(message)
            
            return success, message
            
        except Exception as e:
            error_msg = f"Fatal error: {str(e)}"
            logger.error(f"Error importing chatbook: {e}")
            status.add_error(error_msg)
            return False, error_msg
    
    def _import_conversations(
        self,
        extract_dir: Path,
        manifest: ChatbookManifest,
        conversation_ids: List[str],
        conflict_resolution: ConflictResolution,
        prefix_imported: bool,
        status: ImportStatus
    ) -> None:
        """Import conversations."""
        logger.info(f"ChatbookImporter._import_conversations: Starting import of {len(conversation_ids)} conversations")
        db_path = self.db_paths.get("ChaChaNotes")
        if not db_path:
            logger.error("ChatbookImporter._import_conversations: ChaChaNotes database path not configured")
            status.add_error("ChaChaNotes database path not configured")
            return
            
        db = CharactersRAGDB(db_path, "chatbook_importer")
        conv_dir = extract_dir / "content" / "conversations"
        logger.info(f"ChatbookImporter._import_conversations: Looking for conversations in {conv_dir}")
        
        for conv_id in conversation_ids:
            status.processed_items += 1
            logger.info(f"ChatbookImporter._import_conversations: Processing conversation {conv_id} ({status.processed_items}/{len(conversation_ids)})")
            
            try:
                # Find conversation file
                conv_file = conv_dir / f"conversation_{conv_id}.json"
                if not conv_file.exists():
                    logger.warning(f"ChatbookImporter._import_conversations: Conversation file not found: {conv_file.name}")
                    status.add_warning(f"Conversation file not found: {conv_file.name}")
                    status.failed_items += 1
                    continue
                
                # Load conversation data
                with open(conv_file, 'r', encoding='utf-8') as f:
                    conv_data = json.load(f)
                
                # Check for existing conversation with same name
                conv_name = conv_data['name']
                if prefix_imported:
                    conv_name = f"[Imported] {conv_name}"
                
                # Check for existing conversations with same name
                existing_conversations = db.get_conversation_by_name(conv_name)
                logger.info(f"ChatbookImporter._import_conversations: Found {len(existing_conversations) if existing_conversations else 0} existing conversations with name '{conv_name}'")
                
                if existing_conversations:
                    # Handle conflict - use the first (most recent) conversation
                    existing = existing_conversations[0]
                    resolution = self.conflict_resolver.resolve_conversation_conflict(
                        existing,
                        conv_data,
                        conflict_resolution
                    )
                    
                    if resolution == ConflictResolution.SKIP:
                        logger.info(f"ChatbookImporter._import_conversations: Skipping conversation due to conflict resolution")
                        status.skipped_items += 1
                        continue
                    elif resolution == ConflictResolution.RENAME:
                        old_name = conv_name
                        conv_name = self._generate_unique_name(conv_name, db)
                        logger.info(f"ChatbookImporter._import_conversations: Renamed conversation from '{old_name}' to '{conv_name}'")
                
                # Create conversation
                character_id = conv_data.get('character_id')
                conv_dict = {
                    'title': conv_name,
                    'created_at': conv_data.get('created_at', datetime.now().isoformat()),
                    'updated_at': conv_data.get('updated_at', datetime.now().isoformat()),
                    'character_id': character_id,
                    'root_id': f"imported_{conv_data.get('id', 'unknown')}"
                }
                new_conv_id = db.add_conversation(conv_dict)
                logger.info(f"ChatbookImporter._import_conversations: Created conversation with ID {new_conv_id}")
                
                if new_conv_id:
                    # Import messages
                    logger.info(f"ChatbookImporter._import_conversations: Importing {len(conv_data.get('messages', []))} messages")
                    for msg in conv_data.get('messages', []):
                        msg_dict = {
                            'conversation_id': new_conv_id,
                            'sender': msg['role'],
                            'content': msg['content'],
                            'timestamp': msg.get('timestamp', datetime.now().isoformat())
                        }
                        db.add_message(msg_dict)
                    
                    status.successful_items += 1
                    logger.info(f"ChatbookImporter._import_conversations: Successfully imported conversation: {conv_name}")
                else:
                    status.failed_items += 1
                    status.add_error(f"Failed to create conversation: {conv_name}")
                    logger.error(f"ChatbookImporter._import_conversations: Failed to create conversation: {conv_name}")
                    
            except Exception as e:
                status.failed_items += 1
                status.add_error(f"Error importing conversation {conv_id}: {str(e)}")
                logger.error(f"ChatbookImporter._import_conversations: Error importing conversation {conv_id}: {e}", exc_info=True)
    
    def _import_notes(
        self,
        extract_dir: Path,
        manifest: ChatbookManifest,
        note_ids: List[str],
        conflict_resolution: ConflictResolution,
        prefix_imported: bool,
        status: ImportStatus
    ) -> None:
        """Import notes."""
        logger.info(f"ChatbookImporter._import_notes: Starting import of {len(note_ids)} notes")
        db_path = self.db_paths.get("ChaChaNotes")
        if not db_path:
            logger.error("ChatbookImporter._import_notes: ChaChaNotes database path not configured")
            status.add_error("ChaChaNotes database path not configured")
            return
            
        db = CharactersRAGDB(db_path, "chatbook_importer")
        notes_dir = extract_dir / "content" / "notes"
        logger.info(f"ChatbookImporter._import_notes: Looking for notes in {notes_dir}")
        
        for note_id in note_ids:
            status.processed_items += 1
            logger.info(f"ChatbookImporter._import_notes: Processing note {note_id} ({status.processed_items}/{len(note_ids)})")
            
            try:
                # Find note item in manifest
                note_item = None
                for item in manifest.content_items:
                    if item.id == note_id and item.type == ContentType.NOTE:
                        note_item = item
                        break
                
                if not note_item or not note_item.file_path:
                    logger.warning(f"ChatbookImporter._import_notes: Note metadata not found for ID: {note_id}")
                    status.add_warning(f"Note metadata not found for ID: {note_id}")
                    status.failed_items += 1
                    continue
                
                # Load note file
                note_file = extract_dir / note_item.file_path
                logger.info(f"ChatbookImporter._import_notes: Loading note file from {note_file}")
                if not note_file.exists():
                    logger.warning(f"ChatbookImporter._import_notes: Note file not found: {note_file}")
                    status.add_warning(f"Note file not found: {note_file}")
                    status.failed_items += 1
                    continue
                
                # Parse markdown with frontmatter
                with open(note_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                # Extract frontmatter if present
                if content.startswith('---'):
                    parts = content.split('---', 2)
                    if len(parts) >= 3:
                        # Parse frontmatter
                        frontmatter = parts[1].strip()
                        note_content = parts[2].strip()
                    else:
                        note_content = content
                else:
                    note_content = content
                
                # Check for existing note with same title
                note_title = note_item.title
                if prefix_imported:
                    note_title = f"[Imported] {note_title}"
                
                # Check for existing note
                existing = db.get_note_by_title(note_title)
                
                if existing:
                    # Handle conflict
                    resolution = self.conflict_resolver.resolve_note_conflict(
                        existing,
                        {"title": note_title, "content": note_content},
                        conflict_resolution
                    )
                    
                    if resolution == ConflictResolution.SKIP:
                        status.skipped_items += 1
                        continue
                    elif resolution == ConflictResolution.RENAME:
                        note_title = self._generate_unique_note_title(note_title, db)
                
                # Create note
                # Note: keywords/tags are not stored in the notes table
                new_note_id = db.add_note(
                    title=note_title,
                    content=note_content
                )
                
                if new_note_id:
                    status.successful_items += 1
                    logger.info(f"Imported note: {note_title}")
                else:
                    status.failed_items += 1
                    status.add_error(f"Failed to create note: {note_title}")
                    
            except Exception as e:
                status.failed_items += 1
                status.add_error(f"Error importing note {note_id}: {str(e)}")
                logger.error(f"ChatbookImporter._import_notes: Error importing note {note_id}: {e}", exc_info=True)
    
    def _import_characters(
        self,
        extract_dir: Path,
        manifest: ChatbookManifest,
        character_ids: List[str],
        conflict_resolution: ConflictResolution,
        prefix_imported: bool,
        status: ImportStatus
    ) -> None:
        """Import characters."""
        logger.info(f"ChatbookImporter._import_characters: Starting import of {len(character_ids)} characters")
        db_path = self.db_paths.get("ChaChaNotes")
        if not db_path:
            logger.error("ChatbookImporter._import_characters: ChaChaNotes database path not configured")
            status.add_error("ChaChaNotes database path not configured")
            return
            
        db = CharactersRAGDB(db_path, "chatbook_importer")
        chars_dir = extract_dir / "content" / "characters"
        logger.info(f"ChatbookImporter._import_characters: Looking for characters in {chars_dir}")
        
        for char_id in character_ids:
            status.processed_items += 1
            logger.info(f"ChatbookImporter._import_characters: Processing character {char_id} ({status.processed_items}/{len(character_ids)})")
            
            try:
                # Find character file
                char_file = chars_dir / f"character_{char_id}.json"
                if not char_file.exists():
                    logger.warning(f"ChatbookImporter._import_characters: Character file not found: {char_file.name}")
                    status.add_warning(f"Character file not found: {char_file.name}")
                    status.failed_items += 1
                    continue
                
                # Load character data
                with open(char_file, 'r', encoding='utf-8') as f:
                    raw_char_data = json.load(f)
                
                # Detect and parse character card format
                parsed_card, format_name = detect_and_parse_character_card(raw_char_data)
                logger.info(f"ChatbookImporter._import_characters: Detected format '{format_name}' for character {char_id}")
                if not parsed_card:
                    logger.error(f"ChatbookImporter._import_characters: Failed to parse character card for {char_id} (format: {format_name})")
                    status.add_error(f"Failed to parse character card for {char_id} (format: {format_name})")
                    status.failed_items += 1
                    continue
                
                # Log the detected format
                logger.info(f"ChatbookImporter._import_characters: Successfully parsed character {char_id} from {format_name} format")
                
                # Extract character data from parsed V2 format
                char_data = parsed_card.get('data', parsed_card)
                
                # Check for existing character with same name
                char_name = char_data.get('name', 'Unknown')
                if prefix_imported:
                    char_name = f"[Imported] {char_name}"
                
                # Check for existing character
                existing = db.get_character_card_by_name(char_name)
                logger.info(f"ChatbookImporter._import_characters: Found existing character: {True if existing else False}")
                
                if existing:
                    # Handle conflict
                    resolution = self.conflict_resolver.resolve_character_conflict(
                        existing,
                        char_data,
                        conflict_resolution
                    )
                    
                    if resolution == ConflictResolution.SKIP:
                        logger.info(f"ChatbookImporter._import_characters: Skipping character due to conflict resolution")
                        status.skipped_items += 1
                        continue
                    elif resolution == ConflictResolution.RENAME:
                        old_name = char_name
                        char_name = self._generate_unique_character_name(char_name, db)
                        logger.info(f"ChatbookImporter._import_characters: Renamed character from '{old_name}' to '{char_name}'")
                
                # Create character with V2 formatted data
                # Map V2 fields to database fields
                card_data = {
                    'name': char_name,
                    'description': char_data.get('description', ''),
                    'personality': char_data.get('personality', ''),
                    'scenario': char_data.get('scenario', ''),
                    'first_message': char_data.get('first_mes', ''),
                    'example_messages': char_data.get('mes_example', ''),
                    'creator_notes': char_data.get('creator_notes', ''),
                    'system_prompt': char_data.get('system_prompt', ''),
                    'post_history_instructions': char_data.get('post_history_instructions', ''),
                    'alternate_greetings': char_data.get('alternate_greetings', []),
                    'tags': char_data.get('tags', []),
                    'creator': char_data.get('creator', ''),
                    'character_version': char_data.get('character_version', ''),
                    'extensions': char_data.get('extensions', {}),
                    'character_book': char_data.get('character_book'),
                    'version': 1,  # DB schema version
                    'format': format_name  # Store original format for reference
                }
                
                # If the raw data had a 'card' field with additional data, preserve it
                if 'card' in raw_char_data and isinstance(raw_char_data['card'], dict):
                    # Merge any additional fields from original card
                    for key, value in raw_char_data['card'].items():
                        if key not in card_data and value is not None:
                            card_data[key] = value
                
                new_char_id = db.add_character_card(card_data)
                logger.info(f"ChatbookImporter._import_characters: Created character with ID {new_char_id}")
                
                if new_char_id:
                    status.successful_items += 1
                    logger.info(f"ChatbookImporter._import_characters: Successfully imported character: {char_name}")
                else:
                    status.failed_items += 1
                    status.add_error(f"Failed to create character: {char_name}")
                    logger.error(f"ChatbookImporter._import_characters: Failed to create character: {char_name}")
                    
            except Exception as e:
                status.failed_items += 1
                status.add_error(f"Error importing character {char_id}: {str(e)}")
                logger.error(f"ChatbookImporter._import_characters: Error importing character {char_id}: {e}", exc_info=True)
    
    def _import_prompts(
        self,
        extract_dir: Path,
        manifest: ChatbookManifest,
        prompt_ids: List[str],
        conflict_resolution: ConflictResolution,
        prefix_imported: bool,
        status: ImportStatus
    ) -> None:
        """Import prompts."""
        db_path = self.db_paths.get("Prompts")
        if not db_path:
            status.add_error("Prompts database path not configured")
            return
            
        db = PromptsDatabase(db_path, "chatbook_importer")
        prompts_dir = extract_dir / "content" / "prompts"
        
        for prompt_id in prompt_ids:
            status.processed_items += 1
            
            try:
                # Find prompt file
                prompt_file = prompts_dir / f"prompt_{prompt_id}.json"
                if not prompt_file.exists():
                    status.add_warning(f"Prompt file not found: {prompt_file.name}")
                    status.failed_items += 1
                    continue
                
                # Load prompt data
                with open(prompt_file, 'r', encoding='utf-8') as f:
                    prompt_data = json.load(f)
                
                # Check for existing prompt with same name
                prompt_name = prompt_data['name']
                if prefix_imported:
                    prompt_name = f"[Imported] {prompt_name}"
                
                # Create prompt
                # add_prompt returns tuple: (prompt_id, action, message)
                result = db.add_prompt(
                    name=prompt_name,
                    author=None,
                    details=prompt_data.get('description', ''),
                    system_prompt=prompt_data.get('content', ''),
                    user_prompt=None,
                    keywords=None,
                    overwrite=False
                )
                new_prompt_id = result[0] if result else None
                
                if new_prompt_id:
                    status.successful_items += 1
                    logger.info(f"Imported prompt: {prompt_name}")
                else:
                    status.failed_items += 1
                    status.add_error(f"Failed to create prompt: {prompt_name}")
                    
            except Exception as e:
                status.failed_items += 1
                status.add_error(f"Error importing prompt {prompt_id}: {str(e)}")
                logger.error(f"Error importing prompt {prompt_id}: {e}")
    
    def _import_media(
        self,
        extract_dir: Path,
        manifest: ChatbookManifest,
        media_ids: List[str],
        conflict_resolution: ConflictResolution,
        status: ImportStatus
    ) -> None:
        """Import media items."""
        db_path = self.db_paths.get("Media")
        if not db_path:
            status.add_error("Media database path not configured")
            return
            
        db = MediaDatabase(db_path, "chatbook_importer")
        media_dir = extract_dir / "content" / "media"
        metadata_dir = media_dir / "metadata"
        
        for media_id in media_ids:
            status.processed_items += 1
            
            try:
                # Find media metadata file
                metadata_file = metadata_dir / f"media_{media_id}.json"
                if not metadata_file.exists():
                    status.add_warning(f"Media metadata file not found: {metadata_file.name}")
                    status.failed_items += 1
                    continue
                
                # Load media metadata
                with open(metadata_file, 'r', encoding='utf-8') as f:
                    media_data = json.load(f)
                
                # Check for existing media with same title and URL
                title = media_data.get('title', 'Untitled')
                url = media_data.get('url')
                
                # Check if media already exists by URL
                existing = None
                if url:
                    existing = db.get_media_by_url(url)
                
                if existing:
                    # Handle conflict
                    if conflict_resolution == ConflictResolution.SKIP:
                        status.skipped_items += 1
                        logger.info(f"Skipped existing media: {title}")
                        continue
                    elif conflict_resolution == ConflictResolution.RENAME:
                        title = self._generate_unique_media_title(title, db)
                
                # Load content if available
                content = ""
                content_file = media_dir / f"media_{media_id}.txt"
                if content_file.exists():
                    with open(content_file, 'r', encoding='utf-8') as f:
                        content = f.read()
                
                # Prepare media data for import
                keywords = media_data.get('metadata', {}).get('media_keywords', '')
                if isinstance(keywords, list):
                    keywords = ', '.join(keywords)
                
                # Add media to database
                try:
                    new_media_id = db.add_media_with_keywords(
                        url=url,
                        title=title,
                        media_type=media_data.get('media_type'),
                        content=content or media_data.get('content', ''),
                        media_keywords=keywords,
                        prompt=media_data.get('metadata', {}).get('prompt'),
                        summary=media_data.get('metadata', {}).get('summary'),
                        transcription_model=media_data.get('metadata', {}).get('transcription_model'),
                        author=media_data.get('author'),
                        ingestion_date=media_data.get('metadata', {}).get('ingestion_date')
                    )
                    
                    if new_media_id:
                        status.successful_items += 1
                        logger.info(f"Imported media: {title}")
                    else:
                        status.failed_items += 1
                        status.add_error(f"Failed to create media: {title}")
                        
                except Exception as e:
                    status.failed_items += 1
                    status.add_error(f"Database error importing media '{title}': {str(e)}")
                    
            except Exception as e:
                status.failed_items += 1
                status.add_error(f"Error importing media {media_id}: {str(e)}")
                logger.error(f"Error importing media {media_id}: {e}")
    
    def _generate_unique_media_title(self, base_title: str, db: MediaDatabase) -> str:
        """Generate a unique media title."""
        counter = 1
        new_title = f"{base_title} ({counter})"
        # MediaDatabase doesn't have a get_by_title method, so we'll just append a counter
        # This is fine since media is primarily identified by URL
        return new_title
    
    def _generate_unique_name(self, base_name: str, db: CharactersRAGDB) -> str:
        """Generate a unique conversation name."""
        counter = 1
        while True:
            new_name = f"{base_name} ({counter})"
            # Check if any conversations exist with this name
            if not db.get_conversation_by_name(new_name):  # Empty list is falsy
                return new_name
            counter += 1
    
    def _generate_unique_note_title(self, base_title: str, db: CharactersRAGDB) -> str:
        """Generate a unique note title."""
        counter = 1
        while True:
            new_title = f"{base_title} ({counter})"
            if not db.get_note_by_title(new_title):
                return new_title
            counter += 1
    
    def _generate_unique_character_name(self, base_name: str, db: CharactersRAGDB) -> str:
        """Generate a unique character name."""
        counter = 1
        while True:
            new_name = f"{base_name} ({counter})"
            if not db.get_character_card_by_name(new_name):
                return new_name
            counter += 1