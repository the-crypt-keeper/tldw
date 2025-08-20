# chatbook_creator.py
# Description: Service for creating chatbooks/knowledge packs
#
"""
Chatbook Creator
----------------

Handles the creation and packaging of chatbooks from database content.
"""

import json
import shutil
import zipfile
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional, Set, Tuple
from loguru import logger

from .chatbook_models import (
    Chatbook, ChatbookManifest, ChatbookContent, 
    ContentItem, ContentType, ChatbookVersion, Relationship
)
from .error_handler import ChatbookErrorHandler, ChatbookErrorType, safe_chatbook_operation
from ..DB.ChaChaNotes_DB import CharactersRAGDB
from ..DB.Client_Media_DB_v2 import MediaDatabase
from ..DB.Prompts_DB import PromptsDatabase
from ..DB.RAG_Indexing_DB import RAGIndexingDB
from ..DB.Evals_DB import EvalsDB
from ..DB.Subscriptions_DB import SubscriptionsDB
from ..Utils.text import sanitize_filename
from ..Utils.paths import get_user_data_dir


class ChatbookCreator:
    """Service for creating chatbooks from database content."""
    
    def __init__(self, db_paths: Dict[str, str]):
        """
        Initialize the chatbook creator.
        
        Args:
            db_paths: Dictionary mapping database names to their paths
        """
        logger.info(f"ChatbookCreator.__init__: Initializing with db_paths={db_paths}")
        self.db_paths = db_paths
        # Use cross-platform user data directory
        user_data_dir = get_user_data_dir()
        self.temp_dir = user_data_dir / "temp" / "chatbooks"
        logger.info(f"ChatbookCreator.__init__: Creating temp directory at {self.temp_dir}")
        self.temp_dir.mkdir(parents=True, exist_ok=True)
        self.missing_dependencies: Set[int] = set()
        self.auto_included_characters: Set[int] = set()
        self._selected_characters: Set[str] = set()  # Track explicitly selected characters
        logger.info("ChatbookCreator.__init__: Initialization complete")
        
    def create_chatbook(
        self,
        name: str,
        description: str,
        content_selections: Dict[ContentType, List[str]],
        output_path: Path,
        author: Optional[str] = None,
        include_media: bool = False,
        media_quality: str = "thumbnail",
        include_embeddings: bool = False,
        tags: List[str] = None,
        categories: List[str] = None,
        auto_include_dependencies: bool = True
    ) -> Tuple[bool, str, Dict[str, Any]]:
        """
        Create a chatbook from selected content.
        
        Args:
            name: Name of the chatbook
            description: Description of the chatbook
            content_selections: Dictionary mapping content types to lists of IDs
            output_path: Path where the chatbook should be saved
            author: Author name (optional)
            include_media: Whether to include media files
            media_quality: Quality of media to include (thumbnail/compressed/original)
            include_embeddings: Whether to include embeddings
            tags: List of tags for the chatbook
            categories: List of categories for the chatbook
            auto_include_dependencies: Whether to automatically include missing character dependencies
            
        Returns:
            Tuple of (success, message, dependency_info)
            dependency_info contains:
                - missing_dependencies: List of character IDs that were referenced but not included
                - auto_included: List of character IDs that were automatically included
        """
        logger.info(f"ChatbookCreator.create_chatbook: Starting creation of '{name}'")
        logger.info(f"ChatbookCreator.create_chatbook: Options - include_media={include_media}, media_quality={media_quality}, include_embeddings={include_embeddings}, auto_include_dependencies={auto_include_dependencies}")
        logger.info(f"ChatbookCreator.create_chatbook: Content selections - {[(t.value, len(ids)) for t, ids in content_selections.items()]}")
        
        try:
            # Reset dependency tracking
            self.missing_dependencies.clear()
            self.auto_included_characters.clear()
            self._selected_characters = set(content_selections.get(ContentType.CHARACTER, []))
            logger.info(f"ChatbookCreator.create_chatbook: Tracking {len(self._selected_characters)} explicitly selected characters")
            
            # Create temporary working directory
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            work_dir = self.temp_dir / f"chatbook_{timestamp}"
            work_dir.mkdir(parents=True, exist_ok=True)
            logger.info(f"ChatbookCreator.create_chatbook: Working directory created at {work_dir}")
            
            # Initialize manifest
            logger.info("ChatbookCreator.create_chatbook: Initializing manifest")
            manifest = ChatbookManifest(
                version=ChatbookVersion.V1,
                name=name,
                description=description,
                author=author,
                include_media=include_media,
                include_embeddings=include_embeddings,
                media_quality=media_quality,
                tags=tags or [],
                categories=categories or []
            )
            logger.info(f"ChatbookCreator.create_chatbook: Manifest created - version {manifest.version}, media={include_media}, embeddings={include_embeddings}")
            
            # Initialize content container
            content = ChatbookContent()
            
            # Process each content type
            logger.info(f"ChatbookCreator.create_chatbook: Processing content types")
            
            # Collect conversations
            if ContentType.CONVERSATION in content_selections:
                logger.info(f"ChatbookCreator.create_chatbook: Collecting {len(content_selections[ContentType.CONVERSATION])} conversations")
                self._collect_conversations(
                    content_selections[ContentType.CONVERSATION],
                    work_dir,
                    manifest,
                    content,
                    auto_include_dependencies
                )
            
            # Collect notes
            if ContentType.NOTE in content_selections:
                logger.info(f"ChatbookCreator.create_chatbook: Collecting {len(content_selections[ContentType.NOTE])} notes")
                self._collect_notes(
                    content_selections[ContentType.NOTE],
                    work_dir,
                    manifest,
                    content
                )
            
            # Collect characters
            if ContentType.CHARACTER in content_selections:
                logger.info(f"ChatbookCreator.create_chatbook: Collecting {len(content_selections[ContentType.CHARACTER])} characters")
                self._collect_characters(
                    content_selections[ContentType.CHARACTER],
                    work_dir,
                    manifest,
                    content
                )
            
            # Collect media (if enabled)
            if include_media and ContentType.MEDIA in content_selections:
                logger.info(f"ChatbookCreator.create_chatbook: Collecting {len(content_selections[ContentType.MEDIA])} media items (quality={media_quality})")
                self._collect_media(
                    content_selections[ContentType.MEDIA],
                    work_dir,
                    manifest,
                    content,
                    media_quality
                )
            
            # Collect prompts
            if ContentType.PROMPT in content_selections:
                logger.info(f"ChatbookCreator.create_chatbook: Collecting {len(content_selections[ContentType.PROMPT])} prompts")
                self._collect_prompts(
                    content_selections[ContentType.PROMPT],
                    work_dir,
                    manifest,
                    content
                )
            
            # Auto-discover relationships
            logger.info("ChatbookCreator.create_chatbook: Discovering relationships")
            self._discover_relationships(manifest, content)
            
            # Update statistics
            manifest.total_conversations = len(content.conversations)
            manifest.total_notes = len(content.notes)
            manifest.total_characters = len(content.characters)
            manifest.total_media_items = len(content.media_items)
            logger.info(f"ChatbookCreator.create_chatbook: Final stats - conversations={manifest.total_conversations}, notes={manifest.total_notes}, characters={manifest.total_characters}, media={manifest.total_media_items}")
            
            # Write manifest
            manifest_path = work_dir / "manifest.json"
            logger.info(f"ChatbookCreator.create_chatbook: Writing manifest to {manifest_path}")
            with open(manifest_path, 'w', encoding='utf-8') as f:
                json.dump(manifest.to_dict(), f, indent=2, ensure_ascii=False)
            
            # Create README
            logger.info("ChatbookCreator.create_chatbook: Creating README")
            self._create_readme(work_dir, manifest)
            
            # Package into archive
            if output_path.suffix == '.zip':
                logger.info(f"ChatbookCreator.create_chatbook: Creating ZIP archive at {output_path}")
                self._create_zip_archive(work_dir, output_path)
            else:
                # Default to ZIP if no extension specified
                output_path = output_path.with_suffix('.zip')
                logger.info(f"ChatbookCreator.create_chatbook: Creating ZIP archive at {output_path} (defaulted to .zip)")
                self._create_zip_archive(work_dir, output_path)
            
            # Calculate final size
            manifest.total_size_bytes = output_path.stat().st_size
            logger.info(f"ChatbookCreator.create_chatbook: Archive size: {manifest.total_size_bytes} bytes")
            
            # Cleanup temp directory
            logger.info(f"ChatbookCreator.create_chatbook: Cleaning up temporary directory {work_dir}")
            shutil.rmtree(work_dir)
            
            # Prepare dependency info
            dependency_info = {
                "missing_dependencies": list(self.missing_dependencies),
                "auto_included": list(self.auto_included_characters)
            }
            
            # Build success message
            message = f"Chatbook created successfully at {output_path}"
            if self.auto_included_characters:
                message += f". Auto-included {len(self.auto_included_characters)} character dependencies"
            if self.missing_dependencies:
                message += f". Warning: {len(self.missing_dependencies)} character dependencies are missing"
            
            logger.info(f"ChatbookCreator.create_chatbook: Success - {message}")
            return True, message, dependency_info
            
        except Exception as e:
            logger.error(f"ChatbookCreator.create_chatbook: Error - {e}", exc_info=True)
            dependency_info = {
                "missing_dependencies": list(self.missing_dependencies),
                "auto_included": list(self.auto_included_characters)
            }
            return False, f"Error creating chatbook: {str(e)}", dependency_info
    
    def _collect_conversations(
        self,
        conversation_ids: List[str],
        work_dir: Path,
        manifest: ChatbookManifest,
        content: ChatbookContent,
        auto_include_dependencies: bool
    ) -> None:
        """Collect conversations and their messages."""
        logger.info(f"ChatbookCreator._collect_conversations: Collecting {len(conversation_ids)} conversations")
        db_path = self.db_paths.get("ChaChaNotes")
        if not db_path:
            logger.warning("ChatbookCreator._collect_conversations: ChaChaNotes database path not configured")
            return
            
        db = CharactersRAGDB(db_path, "chatbook_creator")
        conv_dir = work_dir / "content" / "conversations"
        conv_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"ChatbookCreator._collect_conversations: Created conversations directory at {conv_dir}")
        
        for conv_id in conversation_ids:
            logger.debug(f"ChatbookCreator._collect_conversations: Processing conversation {conv_id}")
            try:
                # Get conversation details
                conv = db.get_conversation_by_id(conv_id)
                if not conv:
                    logger.warning(f"ChatbookCreator._collect_conversations: Conversation {conv_id} not found")
                    continue
                
                # Get all messages
                messages = db.get_messages_for_conversation(conv_id)
                logger.debug(f"ChatbookCreator._collect_conversations: Found {len(messages) if messages else 0} messages for conversation {conv_id}")
                
                # Create conversation data
                # Convert datetime to string if needed
                created_at = conv['created_at']
                if hasattr(created_at, 'isoformat'):
                    created_at = created_at.isoformat()
                    
                last_modified = conv.get('last_modified', conv['created_at'])
                if hasattr(last_modified, 'isoformat'):
                    last_modified = last_modified.isoformat()
                    
                conv_data = {
                    "id": conv['id'],
                    "name": conv.get('title', conv.get('conversation_name', 'Untitled')),
                    "created_at": created_at,
                    "updated_at": last_modified,
                    "character_id": conv.get('character_id'),
                    "messages": [
                        {
                            "id": msg['id'],
                            "role": msg['sender'],
                            "content": msg['message'] if 'message' in msg else msg.get('content', ''),
                            "timestamp": msg['timestamp'] if isinstance(msg['timestamp'], str) else msg['timestamp'].isoformat()
                        }
                        for msg in messages
                    ]
                }
                
                # Write conversation file
                conv_file = conv_dir / f"conversation_{conv_id}.json"
                with open(conv_file, 'w', encoding='utf-8') as f:
                    json.dump(conv_data, f, indent=2, ensure_ascii=False)
                
                # Add to content
                content.conversations.append(conv_data)
                
                # Add to manifest
                manifest.content_items.append(ContentItem(
                    id=conv_id,
                    type=ContentType.CONVERSATION,
                    title=conv.get('title', conv.get('conversation_name', 'Untitled')),
                    created_at=datetime.fromisoformat(conv_data['created_at']),
                    updated_at=datetime.fromisoformat(conv_data['updated_at']),
                    file_path=f"content/conversations/conversation_{conv_id}.json"
                ))
                
                # Track character dependency if present
                if conv.get('character_id'):
                    self._add_character_dependency(conv['character_id'], manifest, content, work_dir, auto_include_dependencies)
                    
            except Exception as e:
                logger.error(f"Error collecting conversation {conv_id}: {e}")
    
    def _collect_notes(
        self,
        note_ids: List[str],
        work_dir: Path,
        manifest: ChatbookManifest,
        content: ChatbookContent
    ) -> None:
        """Collect notes and export as markdown."""
        db_path = self.db_paths.get("ChaChaNotes")
        if not db_path:
            return
            
        db = CharactersRAGDB(db_path, "chatbook_creator")
        notes_dir = work_dir / "content" / "notes"
        notes_dir.mkdir(parents=True, exist_ok=True)
        
        for note_id in note_ids:
            try:
                # Get note details
                note = db.get_note_by_id(note_id)
                if not note:
                    continue
                
                # Create note metadata
                # Convert datetime to string if needed
                created_at = note['created_at']
                if hasattr(created_at, 'isoformat'):
                    created_at = created_at.isoformat()
                    
                last_modified = note.get('last_modified', note['created_at'])
                if hasattr(last_modified, 'isoformat'):
                    last_modified = last_modified.isoformat()
                    
                note_data = {
                    "id": note['id'],
                    "title": note['title'],
                    "content": note['content'],
                    "created_at": created_at,
                    "updated_at": last_modified,
                    "tags": note.get('keywords', '').split(',') if note.get('keywords') else []
                }
                
                # Write markdown file
                note_file = notes_dir / f"{sanitize_filename(note['title'])}.md"
                with open(note_file, 'w', encoding='utf-8') as f:
                    # Write frontmatter
                    f.write("---\n")
                    f.write(f"id: {note['id']}\n")
                    f.write(f"title: {note['title']}\n")
                    f.write(f"created_at: {note_data['created_at']}\n")
                    f.write(f"updated_at: {note_data['updated_at']}\n")
                    if note_data['tags']:
                        f.write(f"tags: {', '.join(note_data['tags'])}\n")
                    f.write("---\n\n")
                    
                    # Write content
                    f.write(note['content'])
                
                # Add to content
                content.notes.append(note_data)
                
                # Add to manifest
                manifest.content_items.append(ContentItem(
                    id=note_id,
                    type=ContentType.NOTE,
                    title=note['title'],
                    created_at=datetime.fromisoformat(note_data['created_at']),
                    updated_at=datetime.fromisoformat(note_data['updated_at']),
                    tags=note_data['tags'],
                    file_path=f"content/notes/{note_file.name}"
                ))
                
            except Exception as e:
                logger.error(f"Error collecting note {note_id}: {e}")
    
    def _collect_characters(
        self,
        character_ids: List[str],
        work_dir: Path,
        manifest: ChatbookManifest,
        content: ChatbookContent
    ) -> None:
        """Collect character cards."""
        db_path = self.db_paths.get("ChaChaNotes")
        if not db_path:
            return
            
        db = CharactersRAGDB(db_path, "chatbook_creator")
        chars_dir = work_dir / "content" / "characters"
        chars_dir.mkdir(parents=True, exist_ok=True)
        
        for char_id in character_ids:
            try:
                # Get character card (which includes all details)
                char = db.get_character_card_by_id(int(char_id))
                if not char:
                    continue
                
                # Create character data
                char_data = {
                    "id": char['id'],
                    "name": char['name'],
                    "description": char.get('description', ''),
                    "personality": char.get('personality', ''),
                    "created_at": datetime.now().isoformat(),  # Characters don't have timestamps in DB
                    "updated_at": datetime.now().isoformat(),
                    "avatar_path": char.get('avatar_path'),
                    "card": char  # The full character card data
                }
                
                # Write character file (remove 'card' field which may have non-serializable objects)
                char_file = chars_dir / f"character_{char_id}.json"
                char_data_for_json = {k: v for k, v in char_data.items() if k != 'card'}
                with open(char_file, 'w', encoding='utf-8') as f:
                    json.dump(char_data_for_json, f, indent=2, ensure_ascii=False)
                
                # Add to content
                content.characters.append(char_data)
                
                # Add to manifest
                manifest.content_items.append(ContentItem(
                    id=char_id,
                    type=ContentType.CHARACTER,
                    title=char['name'],
                    description=char.get('description'),
                    created_at=datetime.fromisoformat(char_data['created_at']),
                    updated_at=datetime.fromisoformat(char_data['updated_at']),
                    file_path=f"content/characters/character_{char_id}.json"
                ))
                
            except Exception as e:
                logger.error(f"Error collecting character {char_id}: {e}")
    
    def _collect_media(
        self,
        media_ids: List[str],
        work_dir: Path,
        manifest: ChatbookManifest,
        content: ChatbookContent,
        quality: str
    ) -> None:
        """Collect media items and their files."""
        db_path = self.db_paths.get("Media")
        if not db_path:
            logger.warning("Media database path not configured")
            return
            
        db = MediaDatabase(db_path, "chatbook_creator")
        media_dir = work_dir / "content" / "media"
        media_dir.mkdir(parents=True, exist_ok=True)
        
        # Create metadata directory
        metadata_dir = media_dir / "metadata"
        metadata_dir.mkdir(exist_ok=True)
        
        for media_id in media_ids:
            try:
                # Get media details from database
                media_item = db.get_media_by_id(int(media_id))
                if not media_item:
                    logger.warning(f"Media item not found: {media_id}")
                    continue
                
                # Create media data structure
                media_data = {
                    "id": media_item['id'],
                    "title": media_item.get('title', 'Untitled'),
                    "media_type": media_item.get('media_type'),
                    "url": media_item.get('url'),
                    "author": media_item.get('author'),
                    "content": media_item.get('content', ''),
                    "created_at": media_item.get('created_at'),
                    "updated_at": media_item.get('updated_at'),
                    "metadata": {
                        "ingestion_date": media_item.get('ingestion_date'),
                        "media_keywords": media_item.get('media_keywords'),
                        "prompt": media_item.get('prompt'),
                        "summary": media_item.get('summary'),
                        "transcription_model": media_item.get('transcription_model')
                    }
                }
                
                # Handle media file if it exists
                # Note: The current MediaDatabase doesn't store actual file paths,
                # so we'll store the textual content and metadata
                media_filename = f"media_{media_id}"
                
                # Save media metadata
                metadata_file = metadata_dir / f"{media_filename}.json"
                with open(metadata_file, 'w', encoding='utf-8') as f:
                    json.dump(media_data, f, indent=2, ensure_ascii=False)
                
                # If media has content (transcription, text, etc.), save it
                if media_item.get('content'):
                    content_file = media_dir / f"{media_filename}.txt"
                    with open(content_file, 'w', encoding='utf-8') as f:
                        f.write(media_item['content'])
                
                # Add to content
                content.media_items.append(media_data)
                
                # Add to manifest
                manifest.content_items.append(ContentItem(
                    id=media_id,
                    type=ContentType.MEDIA,
                    title=media_item.get('title', 'Untitled'),
                    description=media_item.get('summary'),
                    created_at=datetime.fromisoformat(media_item['created_at']) if media_item.get('created_at') else datetime.now(),
                    updated_at=datetime.fromisoformat(media_item['updated_at']) if media_item.get('updated_at') else datetime.now(),
                    metadata={
                        "media_type": media_item.get('media_type'),
                        "quality": quality,
                        "has_content": bool(media_item.get('content'))
                    },
                    file_path=f"content/media/metadata/{media_filename}.json"
                ))
                
                logger.info(f"Collected media item: {media_item.get('title', 'Untitled')}")
                
            except Exception as e:
                logger.error(f"Error collecting media {media_id}: {e}")
    
    def _collect_prompts(
        self,
        prompt_ids: List[str],
        work_dir: Path,
        manifest: ChatbookManifest,
        content: ChatbookContent
    ) -> None:
        """Collect prompts."""
        db_path = self.db_paths.get("Prompts")
        if not db_path:
            return
            
        db = PromptsDatabase(db_path, "chatbook_creator")
        prompts_dir = work_dir / "content" / "prompts"
        prompts_dir.mkdir(parents=True, exist_ok=True)
        
        for prompt_id in prompt_ids:
            try:
                # Get prompt details
                prompt = db.get_prompt_by_id(int(prompt_id))
                if not prompt:
                    continue
                
                # Create prompt data
                prompt_data = {
                    "id": prompt['id'],
                    "name": prompt['name'],
                    "description": prompt.get('details', ''),
                    "content": prompt.get('system_prompt', '') or prompt.get('user_prompt', ''),
                    "created_at": prompt.get('created_at', datetime.now().isoformat()),
                    "updated_at": prompt.get('updated_at', datetime.now().isoformat())
                }
                
                # Write prompt file
                prompt_file = prompts_dir / f"prompt_{prompt_id}.json"
                with open(prompt_file, 'w', encoding='utf-8') as f:
                    json.dump(prompt_data, f, indent=2, ensure_ascii=False)
                
                # Add to content
                content.prompts.append(prompt_data)
                
                # Add to manifest
                manifest.content_items.append(ContentItem(
                    id=prompt_id,
                    type=ContentType.PROMPT,
                    title=prompt['name'],
                    description=prompt.get('description'),
                    created_at=datetime.fromisoformat(prompt['created_at']),
                    updated_at=datetime.fromisoformat(prompt['updated_at']),
                    file_path=f"content/prompts/prompt_{prompt_id}.json"
                ))
                
            except Exception as e:
                logger.error(f"Error collecting prompt {prompt_id}: {e}")
    
    def _add_character_dependency(self, character_id: int, manifest: ChatbookManifest, 
                                  content: ChatbookContent, work_dir: Path, 
                                  auto_include: bool) -> None:
        """Add a character as a dependency if not already included."""
        # Check if character already in manifest
        for item in manifest.content_items:
            if item.type == ContentType.CHARACTER and item.id == str(character_id):
                return
        
        # Check if character is explicitly selected
        if str(character_id) in self._selected_characters:
            return  # Will be collected later in _collect_characters
        
        # Track missing dependency
        self.missing_dependencies.add(character_id)
        
        if auto_include:
            # Auto-include the character
            db_path = self.db_paths.get("ChaChaNotes")
            if not db_path:
                logger.error(f"Cannot auto-include character {character_id}: Database path not configured")
                return
                
            db = CharactersRAGDB(db_path, "chatbook_creator")
            
            try:
                # Get character card (which includes all details)
                char = db.get_character_card_by_id(character_id)
                if not char:
                    logger.error(f"Character {character_id} not found in database")
                    return
                
                # Create character data
                char_data = {
                    "id": char['id'],
                    "name": char['name'],
                    "description": char.get('description', ''),
                    "personality": char.get('personality', ''),
                    "created_at": datetime.now().isoformat(),  # Characters don't have timestamps in DB
                    "updated_at": datetime.now().isoformat(),
                    "avatar_path": char.get('avatar_path'),
                    "card": char  # The full character card data
                }
                
                # Create characters directory if it doesn't exist
                chars_dir = work_dir / "content" / "characters"
                chars_dir.mkdir(parents=True, exist_ok=True)
                
                # Write character file (remove 'card' field which may have non-serializable objects)
                char_file = chars_dir / f"character_{character_id}.json"
                char_data_for_json = {k: v for k, v in char_data.items() if k != 'card'}
                with open(char_file, 'w', encoding='utf-8') as f:
                    json.dump(char_data_for_json, f, indent=2, ensure_ascii=False)
                
                # Add to content
                content.characters.append(char_data)
                
                # Add to manifest
                manifest.content_items.append(ContentItem(
                    id=str(character_id),
                    type=ContentType.CHARACTER,
                    title=char['name'],
                    description=char.get('description'),
                    created_at=datetime.fromisoformat(char_data['created_at']),
                    updated_at=datetime.fromisoformat(char_data['updated_at']),
                    file_path=f"content/characters/character_{character_id}.json",
                    metadata={"auto_included": True}
                ))
                
                # Track auto-included character
                self.auto_included_characters.add(character_id)
                self.missing_dependencies.remove(character_id)
                
                logger.info(f"Auto-included character dependency: {char['name']} (ID: {character_id})")
                
            except Exception as e:
                logger.error(f"Error auto-including character {character_id}: {e}")
        else:
            logger.warning(f"Character {character_id} is referenced but not included in chatbook")
    
    def _discover_relationships(self, manifest: ChatbookManifest, content: ChatbookContent) -> None:
        """Discover relationships between content items."""
        # Find conversation-character relationships
        for conv in content.conversations:
            if conv.get('character_id'):
                # Find if character is in manifest
                char_id = str(conv['character_id'])
                for item in manifest.content_items:
                    if item.type == ContentType.CHARACTER and item.id == char_id:
                        manifest.relationships.append(Relationship(
                            source_id=str(conv['id']),
                            target_id=char_id,
                            relationship_type="uses_character"
                        ))
                        break
        
        # TODO: Add more relationship discovery logic
        # - Notes mentioning conversations
        # - Media used in conversations
        # - etc.
    
    def _create_readme(self, work_dir: Path, manifest: ChatbookManifest) -> None:
        """Create a README file for the chatbook."""
        readme_path = work_dir / "README.md"
        
        with open(readme_path, 'w', encoding='utf-8') as f:
            f.write(f"# {manifest.name}\n\n")
            f.write(f"{manifest.description}\n\n")
            
            if manifest.author:
                f.write(f"**Author:** {manifest.author}\n\n")
            
            f.write(f"**Created:** {manifest.created_at.strftime('%Y-%m-%d %H:%M')}\n\n")
            
            f.write("## Contents\n\n")
            
            if manifest.total_conversations > 0:
                f.write(f"- **Conversations:** {manifest.total_conversations}\n")
            if manifest.total_notes > 0:
                f.write(f"- **Notes:** {manifest.total_notes}\n")
            if manifest.total_characters > 0:
                f.write(f"- **Characters:** {manifest.total_characters}\n")
            if manifest.total_media_items > 0:
                f.write(f"- **Media Items:** {manifest.total_media_items}\n")
            
            if manifest.tags:
                f.write(f"\n## Tags\n\n")
                f.write(", ".join(manifest.tags))
                f.write("\n")
            
            f.write("\n## Structure\n\n")
            f.write("```\n")
            f.write("chatbook/\n")
            f.write("├── manifest.json     # Chatbook metadata\n")
            f.write("├── README.md        # This file\n")
            f.write("└── content/         # Content files\n")
            if manifest.total_conversations > 0:
                f.write("    ├── conversations/   # Chat conversations\n")
            if manifest.total_notes > 0:
                f.write("    ├── notes/          # Markdown notes\n")
            if manifest.total_characters > 0:
                f.write("    ├── characters/     # Character cards\n")
            if manifest.total_media_items > 0:
                f.write("    ├── media/          # Media files and content\n")
                f.write("    │   └── metadata/   # Media metadata JSON files\n")
            f.write("```\n")
            
            f.write("\n## License\n\n")
            if manifest.license:
                f.write(manifest.license)
            else:
                f.write("See individual content files for licensing information.")
    
    def _create_zip_archive(self, work_dir: Path, output_path: Path) -> None:
        """Create a ZIP archive of the chatbook."""
        with zipfile.ZipFile(output_path, 'w', zipfile.ZIP_DEFLATED) as zf:
            for file_path in work_dir.rglob('*'):
                if file_path.is_file():
                    arcname = file_path.relative_to(work_dir)
                    zf.write(file_path, arcname)