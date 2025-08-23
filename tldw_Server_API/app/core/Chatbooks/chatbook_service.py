# chatbook_service.py
# Description: Service for creating and importing chatbooks with multi-user support
# Adapted from single-user to multi-user architecture
#
"""
Chatbook Service for Multi-User Environment
--------------------------------------------

Handles the creation, import, and export of chatbooks with user isolation.

Key Adaptations from Single-User:
- User-specific exports with access control
- Job-based operations for async processing
- Temporary storage with automatic cleanup
- Per-user database isolation
- No global state or singletons
"""

import json
import shutil
import zipfile
import hashlib
import asyncio
import aiofiles
import aiofiles.os
import os
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple, Set
from uuid import uuid4
from loguru import logger
from contextlib import asynccontextmanager

from .chatbook_models import (
    ChatbookManifest, ChatbookContent, ContentItem, ContentType,
    ChatbookVersion, Relationship, ExportJob, ImportJob,
    ExportStatus, ImportStatus, ConflictResolution, ImportConflict
)
from ..DB_Management.ChaChaNotes_DB import CharactersRAGDB


class ChatbookService:
    """Service for creating and importing chatbooks with user isolation."""
    
    def __init__(self, user_id: str, db: CharactersRAGDB):
        """
        Initialize the chatbook service for a specific user.
        
        Args:
            user_id: User identifier
            db: User's ChaChaNotes database instance
        """
        self.user_id = user_id
        self.db = db
        
        # Secure user-specific directory using application data path
        # Get base path from environment or use secure default
        import os
        base_data_dir = Path(os.environ.get('TLDW_USER_DATA_PATH', '/var/lib/tldw/user_data'))
        
        # Create secure user-specific directory with restricted permissions
        self.user_data_dir = base_data_dir / 'users' / str(user_id) / 'chatbooks'
        self.user_data_dir.mkdir(parents=True, exist_ok=True, mode=0o700)
        
        # Separate directories for exports and imports
        self.export_dir = self.user_data_dir / 'exports'
        self.import_dir = self.user_data_dir / 'imports'
        self.temp_dir = self.user_data_dir / 'temp'
        
        for directory in [self.export_dir, self.import_dir, self.temp_dir]:
            directory.mkdir(parents=True, exist_ok=True, mode=0o700)
        
        # Initialize job tracking tables in database
        self._init_job_tables()
    
    def _init_job_tables(self):
        """Initialize database tables for job tracking."""
        try:
            # Export jobs table
            self.db.execute_query("""
                CREATE TABLE IF NOT EXISTS export_jobs (
                    job_id TEXT PRIMARY KEY,
                    user_id TEXT NOT NULL,
                    status TEXT NOT NULL,
                    chatbook_name TEXT NOT NULL,
                    output_path TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    started_at TIMESTAMP,
                    completed_at TIMESTAMP,
                    error_message TEXT,
                    progress_percentage INTEGER DEFAULT 0,
                    total_items INTEGER DEFAULT 0,
                    processed_items INTEGER DEFAULT 0,
                    file_size_bytes INTEGER,
                    download_url TEXT,
                    expires_at TIMESTAMP
                )
            """)
            
            # Import jobs table
            self.db.execute_query("""
                CREATE TABLE IF NOT EXISTS import_jobs (
                    job_id TEXT PRIMARY KEY,
                    user_id TEXT NOT NULL,
                    status TEXT NOT NULL,
                    chatbook_path TEXT NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    started_at TIMESTAMP,
                    completed_at TIMESTAMP,
                    error_message TEXT,
                    progress_percentage INTEGER DEFAULT 0,
                    total_items INTEGER DEFAULT 0,
                    processed_items INTEGER DEFAULT 0,
                    successful_items INTEGER DEFAULT 0,
                    failed_items INTEGER DEFAULT 0,
                    skipped_items INTEGER DEFAULT 0,
                    conflicts TEXT,  -- JSON array
                    warnings TEXT    -- JSON array
                )
            """)
        except Exception as e:
            logger.error(f"Error initializing job tables: {e}")
    
    async def create_chatbook(
        self,
        name: str,
        description: str,
        content_selections: Dict[ContentType, List[str]],
        author: Optional[str] = None,
        include_media: bool = False,
        media_quality: str = "compressed",
        include_embeddings: bool = False,
        include_generated_content: bool = True,
        tags: List[str] = None,
        categories: List[str] = None,
        async_mode: bool = False
    ) -> Tuple[bool, str, Optional[str]]:
        """
        Create a chatbook from selected content.
        
        Args:
            name: Chatbook name
            description: Chatbook description
            content_selections: Content to include by type and IDs
            author: Author name
            include_media: Include media files
            media_quality: Media quality level
            include_embeddings: Include embeddings
            include_generated_content: Include generated documents
            tags: Chatbook tags
            categories: Chatbook categories
            async_mode: Run as background job
            
        Returns:
            Tuple of (success, message, job_id or file_path)
        """
        if async_mode:
            # Create job and run asynchronously
            job_id = str(uuid4())
            job = ExportJob(
                job_id=job_id,
                user_id=self.user_id,
                status=ExportStatus.PENDING,
                chatbook_name=name
            )
            
            # Store job in database
            self._save_export_job(job)
            
            # Start async task
            asyncio.create_task(self._create_chatbook_async(
                job_id, name, description, content_selections,
                author, include_media, media_quality, include_embeddings,
                include_generated_content, tags, categories
            ))
            
            return True, f"Export job started: {job_id}", job_id
        else:
            # Run asynchronously
            return await self._create_chatbook_async(
                name, description, content_selections,
                author, include_media, media_quality, include_embeddings,
                include_generated_content, tags, categories
            )
    
    @asynccontextmanager
    async def _transaction_context(self):
        """Context manager for database transactions."""
        # Start transaction
        conn = self.db.get_connection()
        try:
            conn.execute("BEGIN TRANSACTION")
            yield conn
            conn.execute("COMMIT")
        except Exception as e:
            conn.execute("ROLLBACK")
            logger.error(f"Transaction rolled back: {e}")
            raise
        finally:
            conn.close()
    
    async def _create_chatbook_async(
        self,
        name: str,
        description: str,
        content_selections: Dict[ContentType, List[str]],
        author: Optional[str] = None,
        include_media: bool = False,
        media_quality: str = "compressed",
        include_embeddings: bool = False,
        include_generated_content: bool = True,
        tags: List[str] = None,
        categories: List[str] = None
    ) -> Tuple[bool, str, Optional[str]]:
        """
        Synchronously create a chatbook.
        
        Returns:
            Tuple of (success, message, file_path)
        """
        try:
            # Create working directory in secure temp location
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            work_dir = self.temp_dir / f"export_{timestamp}_{uuid4().hex[:8]}"
            work_dir.mkdir(parents=True, exist_ok=True, mode=0o700)
            
            # Initialize manifest
            manifest = ChatbookManifest(
                version=ChatbookVersion.V1,
                name=name,
                description=description,
                author=author,
                user_id=hashlib.sha256(self.user_id.encode()).hexdigest()[:16],  # Anonymized
                include_media=include_media,
                include_embeddings=include_embeddings,
                include_generated_content=include_generated_content,
                media_quality=media_quality,
                tags=tags or [],
                categories=categories or [],
                export_id=str(uuid4())
            )
            
            # Collect content
            content = ChatbookContent()
            
            # Process each content type
            if ContentType.CONVERSATION in content_selections:
                self._collect_conversations(
                    content_selections[ContentType.CONVERSATION],
                    work_dir, manifest, content
                )
            
            if ContentType.NOTE in content_selections:
                self._collect_notes(
                    content_selections[ContentType.NOTE],
                    work_dir, manifest, content
                )
            
            if ContentType.CHARACTER in content_selections:
                self._collect_characters(
                    content_selections[ContentType.CHARACTER],
                    work_dir, manifest, content
                )
            
            if ContentType.WORLD_BOOK in content_selections:
                self._collect_world_books(
                    content_selections[ContentType.WORLD_BOOK],
                    work_dir, manifest, content
                )
            
            if ContentType.DICTIONARY in content_selections:
                self._collect_dictionaries(
                    content_selections[ContentType.DICTIONARY],
                    work_dir, manifest, content
                )
            
            if include_generated_content and ContentType.GENERATED_DOCUMENT in content_selections:
                self._collect_generated_documents(
                    content_selections[ContentType.GENERATED_DOCUMENT],
                    work_dir, manifest, content
                )
            
            # Update statistics
            manifest.total_conversations = len(content.conversations)
            manifest.total_notes = len(content.notes)
            manifest.total_characters = len(content.characters)
            manifest.total_world_books = len(content.world_books)
            manifest.total_dictionaries = len(content.dictionaries)
            manifest.total_documents = len(content.generated_documents)
            
            # Write manifest asynchronously
            manifest_path = work_dir / "manifest.json"
            async with aiofiles.open(manifest_path, 'w', encoding='utf-8') as f:
                await f.write(json.dumps(manifest.to_dict(), indent=2, ensure_ascii=False))
            
            # Create README asynchronously
            await self._create_readme_async(work_dir, manifest)
            
            # Create archive in secure export directory
            safe_name = "".join(c if c.isalnum() or c in "_-" else "_" for c in name)
            output_filename = f"{safe_name}_{timestamp}_{uuid4().hex[:8]}.zip"
            output_path = self.export_dir / output_filename
            await self._create_zip_archive_async(work_dir, output_path)
            
            # Update manifest with file size
            manifest.total_size_bytes = output_path.stat().st_size
            
            # Cleanup working directory asynchronously
            await asyncio.to_thread(shutil.rmtree, work_dir)
            
            # Store file path in job record (will be retrieved by job_id)
            # No direct filename access for security
            download_url = None  # Will be generated from job_id
            
            return True, f"Chatbook created successfully", str(output_path)
            
        except Exception as e:
            logger.error(f"Error creating chatbook: {e}")
            return False, f"Error creating chatbook: {str(e)}", None
    
    async def _create_chatbook_async(
        self,
        job_id: str,
        name: str,
        description: str,
        content_selections: Dict[ContentType, List[str]],
        author: Optional[str],
        include_media: bool,
        media_quality: str,
        include_embeddings: bool,
        include_generated_content: bool,
        tags: List[str],
        categories: List[str]
    ):
        """
        Asynchronously create a chatbook.
        """
        # Get job from database
        job = self._get_export_job(job_id)
        if not job:
            return
        
        try:
            # Update job status
            job.status = ExportStatus.IN_PROGRESS
            job.started_at = datetime.utcnow()
            self._save_export_job(job)
            
            # Create chatbook synchronously (could be made truly async)
            success, message, file_path = self._create_chatbook_sync(
                name, description, content_selections,
                author, include_media, media_quality, include_embeddings,
                include_generated_content, tags, categories
            )
            
            if success:
                # Update job with success
                job.status = ExportStatus.COMPLETED
                job.completed_at = datetime.utcnow()
                job.output_path = file_path
                job.file_size_bytes = Path(file_path).stat().st_size if file_path else None
                job.download_url = f"/api/v1/chatbooks/download/{Path(file_path).name}" if file_path else None
                job.expires_at = datetime.utcnow() + timedelta(hours=24)
            else:
                # Update job with failure
                job.status = ExportStatus.FAILED
                job.completed_at = datetime.utcnow()
                job.error_message = message
            
            self._save_export_job(job)
            
        except Exception as e:
            # Update job with error
            job.status = ExportStatus.FAILED
            job.completed_at = datetime.utcnow()
            job.error_message = str(e)
            self._save_export_job(job)
    
    async def import_chatbook(
        self,
        file_path: str,
        content_selections: Optional[Dict[ContentType, List[str]]] = None,
        conflict_resolution: ConflictResolution = ConflictResolution.SKIP,
        prefix_imported: bool = False,
        import_media: bool = True,
        import_embeddings: bool = False,
        async_mode: bool = False
    ) -> Tuple[bool, str, Optional[str]]:
        """
        Import a chatbook.
        
        Args:
            file_path: Path to chatbook file
            content_selections: Specific content to import
            conflict_resolution: How to handle conflicts
            prefix_imported: Add prefix to imported items
            import_media: Import media files
            import_embeddings: Import embeddings
            async_mode: Run as background job
            
        Returns:
            Tuple of (success, message, job_id or None)
        """
        if async_mode:
            # Create job and run asynchronously
            job_id = str(uuid4())
            job = ImportJob(
                job_id=job_id,
                user_id=self.user_id,
                status=ImportStatus.PENDING,
                chatbook_path=file_path
            )
            
            # Store job in database
            self._save_import_job(job)
            
            # Start async task
            asyncio.create_task(self._import_chatbook_async(
                job_id, file_path, content_selections,
                conflict_resolution, prefix_imported,
                import_media, import_embeddings
            ))
            
            return True, f"Import job started: {job_id}", job_id
        else:
            # Run synchronously
            return self._import_chatbook_sync(
                file_path, content_selections,
                conflict_resolution, prefix_imported,
                import_media, import_embeddings
            )
    
    def _import_chatbook_sync(
        self,
        file_path: str,
        content_selections: Optional[Dict[ContentType, List[str]]],
        conflict_resolution: ConflictResolution,
        prefix_imported: bool,
        import_media: bool,
        import_embeddings: bool
    ) -> Tuple[bool, str, None]:
        """
        Synchronously import a chatbook.
        """
        try:
            # Validate file first
            if not self._validate_zip_file(file_path):
                return False, "Invalid or potentially malicious archive file", None
            
            # Extract chatbook to secure temp location
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            extract_dir = self.temp_dir / f"import_{timestamp}_{uuid4().hex[:8]}"
            extract_dir.mkdir(parents=True, exist_ok=True, mode=0o700)
            
            # Extract archive with size limits
            with zipfile.ZipFile(file_path, 'r') as zf:
                # Check total uncompressed size
                total_size = sum(zinfo.file_size for zinfo in zf.filelist)
                if total_size > 500 * 1024 * 1024:  # 500MB limit
                    return False, "Archive too large (>500MB uncompressed)", None
                
                # Extract with path validation
                for member in zf.namelist():
                    # Prevent path traversal
                    if os.path.isabs(member) or ".." in member:
                        return False, f"Unsafe path in archive: {member}", None
                    
                    # Extract individual file
                    zf.extract(member, extract_dir)
            
            # Load manifest
            manifest_path = extract_dir / "manifest.json"
            if not manifest_path.exists():
                shutil.rmtree(extract_dir)
                return False, "Invalid chatbook: manifest.json not found", None
            
            with open(manifest_path, 'r', encoding='utf-8') as f:
                manifest_data = json.load(f)
            
            manifest = ChatbookManifest.from_dict(manifest_data)
            
            # Check version compatibility
            if manifest.version != ChatbookVersion.V1:
                logger.warning(f"Chatbook version {manifest.version.value} may not be fully compatible")
            
            # Set up content selections if not provided
            if content_selections is None:
                content_selections = {}
                for item in manifest.content_items:
                    if item.type not in content_selections:
                        content_selections[item.type] = []
                    content_selections[item.type].append(item.id)
            
            # Import each content type
            import_status = ImportJob(
                job_id="temp",
                user_id=self.user_id,
                status=ImportStatus.IN_PROGRESS,
                chatbook_path=file_path
            )
            
            import_status.total_items = sum(len(ids) for ids in content_selections.values())
            
            # Import characters first (they may be dependencies)
            if ContentType.CHARACTER in content_selections:
                self._import_characters(
                    extract_dir, manifest,
                    content_selections[ContentType.CHARACTER],
                    conflict_resolution, prefix_imported,
                    import_status
                )
            
            # Import world books
            if ContentType.WORLD_BOOK in content_selections:
                self._import_world_books(
                    extract_dir, manifest,
                    content_selections[ContentType.WORLD_BOOK],
                    conflict_resolution, prefix_imported,
                    import_status
                )
            
            # Import dictionaries
            if ContentType.DICTIONARY in content_selections:
                self._import_dictionaries(
                    extract_dir, manifest,
                    content_selections[ContentType.DICTIONARY],
                    conflict_resolution, prefix_imported,
                    import_status
                )
            
            # Import conversations
            if ContentType.CONVERSATION in content_selections:
                self._import_conversations(
                    extract_dir, manifest,
                    content_selections[ContentType.CONVERSATION],
                    conflict_resolution, prefix_imported,
                    import_status
                )
            
            # Import notes
            if ContentType.NOTE in content_selections:
                self._import_notes(
                    extract_dir, manifest,
                    content_selections[ContentType.NOTE],
                    conflict_resolution, prefix_imported,
                    import_status
                )
            
            # Cleanup
            shutil.rmtree(extract_dir)
            
            # Build result message
            if import_status.successful_items > 0:
                message = f"Successfully imported {import_status.successful_items}/{import_status.total_items} items"
                if import_status.skipped_items > 0:
                    message += f" ({import_status.skipped_items} skipped)"
                return True, message, None
            else:
                return False, "No items were imported", None
            
        except Exception as e:
            logger.error(f"Error importing chatbook: {e}")
            return False, f"Error importing chatbook: {str(e)}", None
    
    async def _import_chatbook_async(
        self,
        job_id: str,
        file_path: str,
        content_selections: Optional[Dict[ContentType, List[str]]],
        conflict_resolution: ConflictResolution,
        prefix_imported: bool,
        import_media: bool,
        import_embeddings: bool
    ):
        """
        Asynchronously import a chatbook.
        """
        # Get job from database
        job = self._get_import_job(job_id)
        if not job:
            return
        
        try:
            # Update job status
            job.status = ImportStatus.IN_PROGRESS
            job.started_at = datetime.utcnow()
            self._save_import_job(job)
            
            # Import chatbook synchronously (could be made truly async)
            success, message, _ = self._import_chatbook_sync(
                file_path, content_selections,
                conflict_resolution, prefix_imported,
                import_media, import_embeddings
            )
            
            if success:
                job.status = ImportStatus.COMPLETED
            else:
                job.status = ImportStatus.FAILED
                job.error_message = message
            
            job.completed_at = datetime.utcnow()
            self._save_import_job(job)
            
        except Exception as e:
            job.status = ImportStatus.FAILED
            job.completed_at = datetime.utcnow()
            job.error_message = str(e)
            self._save_import_job(job)
    
    def preview_chatbook(self, file_path: str) -> Tuple[Optional[ChatbookManifest], Optional[str]]:
        """
        Preview a chatbook without importing it.
        
        Args:
            file_path: Path to chatbook file
            
        Returns:
            Tuple of (manifest, error_message)
        """
        try:
            # Extract to temporary directory
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            extract_dir = self.temp_dir / f"preview_{timestamp}"
            
            # Extract archive
            with zipfile.ZipFile(file_path, 'r') as zf:
                zf.extractall(extract_dir)
            
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
    
    def get_export_job(self, job_id: str) -> Optional[ExportJob]:
        """Get export job status."""
        return self._get_export_job(job_id)
    
    def get_import_job(self, job_id: str) -> Optional[ImportJob]:
        """Get import job status."""
        return self._get_import_job(job_id)
    
    def list_export_jobs(self) -> List[ExportJob]:
        """List all export jobs for this user."""
        try:
            results = self.db.execute_query(
                "SELECT * FROM export_jobs WHERE user_id = ? ORDER BY created_at DESC",
                (self.user_id,)
            )
            
            jobs = []
            for row in results:
                job = ExportJob(
                    job_id=row['job_id'],
                    user_id=row['user_id'],
                    status=ExportStatus(row['status']),
                    chatbook_name=row['chatbook_name'],
                    output_path=row['output_path'],
                    created_at=datetime.fromisoformat(row['created_at']) if row['created_at'] else None,
                    started_at=datetime.fromisoformat(row['started_at']) if row['started_at'] else None,
                    completed_at=datetime.fromisoformat(row['completed_at']) if row['completed_at'] else None,
                    error_message=row['error_message'],
                    progress_percentage=row['progress_percentage'],
                    total_items=row['total_items'],
                    processed_items=row['processed_items'],
                    file_size_bytes=row['file_size_bytes'],
                    download_url=row['download_url'],
                    expires_at=datetime.fromisoformat(row['expires_at']) if row['expires_at'] else None
                )
                jobs.append(job)
            
            return jobs
        except Exception as e:
            logger.error(f"Error listing export jobs: {e}")
            return []    
    def list_import_jobs(self) -> List[ImportJob]:
        """List all import jobs for this user."""
        try:
            results = self.db.execute_query(
                "SELECT * FROM import_jobs WHERE user_id = ? ORDER BY created_at DESC",
                (self.user_id,)
            )
            
            jobs = []
            for row in results:
                job = ImportJob(
                    job_id=row['job_id'],
                    user_id=row['user_id'],
                    status=ImportStatus(row['status']),
                    chatbook_path=row['chatbook_path'],
                    created_at=datetime.fromisoformat(row['created_at']) if row['created_at'] else None,
                    started_at=datetime.fromisoformat(row['started_at']) if row['started_at'] else None,
                    completed_at=datetime.fromisoformat(row['completed_at']) if row['completed_at'] else None,
                    error_message=row['error_message'],
                    progress_percentage=row['progress_percentage'],
                    total_items=row['total_items'],
                    processed_items=row['processed_items'],
                    successful_items=row['successful_items'],
                    failed_items=row['failed_items'],
                    skipped_items=row['skipped_items'],
                    conflicts=json.loads(row['conflicts']) if row['conflicts'] else [],
                    warnings=json.loads(row['warnings']) if row['warnings'] else []
                )
                jobs.append(job)
            
            return jobs
        except Exception as e:
            logger.error(f"Error listing import jobs: {e}")
            return []
    
    def cleanup_expired_exports(self) -> int:
        """Clean up expired export files. Returns number of files deleted."""
        try:
            # Get expired jobs
            now = datetime.utcnow()
            results = self.db.execute_query(
                "SELECT * FROM export_jobs WHERE user_id = ? AND expires_at < ? AND status = ?",
                (self.user_id, now.isoformat(), ExportStatus.COMPLETED.value)
            )
            
            deleted_count = 0
            for row in results:
                if row['output_path'] and Path(row['output_path']).exists():
                    try:
                        Path(row['output_path']).unlink()
                        deleted_count += 1
                    except Exception as e:
                        logger.error(f"Error deleting expired export: {e}")
                
                # Update job status
                self.db.execute_query(
                    "UPDATE export_jobs SET status = ? WHERE job_id = ?",
                    ('expired', row['job_id'])
                )
            
            return deleted_count
        except Exception as e:
            logger.error(f"Error cleaning up expired exports: {e}")
            return 0
    
    # Helper methods for collecting content
    
    def _collect_conversations(
        self,
        conversation_ids: List[str],
        work_dir: Path,
        manifest: ChatbookManifest,
        content: ChatbookContent
    ):
        """Collect conversations for export."""
        conv_dir = work_dir / "content" / "conversations"
        conv_dir.mkdir(parents=True, exist_ok=True)
        
        for conv_id in conversation_ids:
            try:
                # Get conversation
                conv = self.db.get_conversation_by_id(conv_id)
                if not conv:
                    continue
                
                # Get messages
                messages = self.db.get_messages_for_conversation(conv_id)
                
                # Create conversation data
                conv_data = {
                    "id": conv['id'],
                    "name": conv.get('title', 'Untitled'),
                    "created_at": conv['created_at'].isoformat() if hasattr(conv['created_at'], 'isoformat') else conv['created_at'],
                    "character_id": conv.get('character_id'),
                    "messages": [
                        {
                            "id": msg['id'],
                            "role": msg['sender'],
                            "content": msg.get('message', msg.get('content', '')),
                            "timestamp": msg['timestamp'].isoformat() if hasattr(msg['timestamp'], 'isoformat') else msg['timestamp']
                        }
                        for msg in (messages or [])
                    ]
                }
                
                # Write to file
                conv_file = conv_dir / f"conversation_{conv_id}.json"
                with open(conv_file, 'w', encoding='utf-8') as f:
                    json.dump(conv_data, f, indent=2, ensure_ascii=False)
                
                # Add to content
                content.conversations[conv_id] = conv_data
                
                # Add to manifest
                manifest.content_items.append(ContentItem(
                    id=conv_id,
                    type=ContentType.CONVERSATION,
                    title=conv_data['name'],
                    file_path=f"content/conversations/conversation_{conv_id}.json"
                ))
                
            except Exception as e:
                logger.error(f"Error collecting conversation {conv_id}: {e}")
    
    def _collect_notes(
        self,
        note_ids: List[str],
        work_dir: Path,
        manifest: ChatbookManifest,
        content: ChatbookContent
    ):
        """Collect notes for export."""
        notes_dir = work_dir / "content" / "notes"
        notes_dir.mkdir(parents=True, exist_ok=True)
        
        for note_id in note_ids:
            try:
                # Get note
                note = self.db.get_note_by_id(note_id)
                if not note:
                    continue
                
                # Create note data
                note_data = {
                    "id": note['id'],
                    "title": note['title'],
                    "content": note['content'],
                    "created_at": note['created_at'].isoformat() if hasattr(note['created_at'], 'isoformat') else note['created_at']
                }
                
                # Write markdown file
                note_file = notes_dir / f"note_{note_id}.md"
                with open(note_file, 'w', encoding='utf-8') as f:
                    # Write frontmatter
                    f.write("---\n")
                    f.write(f"id: {note['id']}\n")
                    f.write(f"title: {note['title']}\n")
                    f.write(f"created_at: {note_data['created_at']}\n")
                    f.write("---\n\n")
                    f.write(note['content'])
                
                # Add to content
                content.notes[note_id] = note_data
                
                # Add to manifest
                manifest.content_items.append(ContentItem(
                    id=note_id,
                    type=ContentType.NOTE,
                    title=note['title'],
                    file_path=f"content/notes/note_{note_id}.md"
                ))
                
            except Exception as e:
                logger.error(f"Error collecting note {note_id}: {e}")
    
    def _collect_characters(
        self,
        character_ids: List[str],
        work_dir: Path,
        manifest: ChatbookManifest,
        content: ChatbookContent
    ):
        """Collect character cards for export."""
        chars_dir = work_dir / "content" / "characters"
        chars_dir.mkdir(parents=True, exist_ok=True)
        
        for char_id in character_ids:
            try:
                # Get character
                char = self.db.get_character_card_by_id(int(char_id))
                if not char:
                    continue
                
                # Write character file
                char_file = chars_dir / f"character_{char_id}.json"
                with open(char_file, 'w', encoding='utf-8') as f:
                    json.dump(char, f, indent=2, ensure_ascii=False)
                
                # Add to content
                content.characters[char_id] = char
                
                # Add to manifest
                manifest.content_items.append(ContentItem(
                    id=char_id,
                    type=ContentType.CHARACTER,
                    title=char.get('name', 'Unnamed'),
                    file_path=f"content/characters/character_{char_id}.json"
                ))
                
            except Exception as e:
                logger.error(f"Error collecting character {char_id}: {e}")
    
    def _collect_world_books(
        self,
        world_book_ids: List[str],
        work_dir: Path,
        manifest: ChatbookManifest,
        content: ChatbookContent
    ):
        """Collect world books for export."""
        wb_dir = work_dir / "content" / "world_books"
        wb_dir.mkdir(parents=True, exist_ok=True)
        
        # Import the world book service
        from ..Character_Chat.world_book_manager import WorldBookService
        
        wb_service = WorldBookService(self.db)
        
        for wb_id in world_book_ids:
            try:
                # Get world book with entries
                wb_data = wb_service.get_world_book(int(wb_id))
                if not wb_data:
                    continue
                
                # Write world book file
                wb_file = wb_dir / f"world_book_{wb_id}.json"
                with open(wb_file, 'w', encoding='utf-8') as f:
                    json.dump(wb_data, f, indent=2, ensure_ascii=False)
                
                # Add to content
                content.world_books[wb_id] = wb_data
                
                # Add to manifest
                manifest.content_items.append(ContentItem(
                    id=wb_id,
                    type=ContentType.WORLD_BOOK,
                    title=wb_data.get('name', 'Unnamed'),
                    file_path=f"content/world_books/world_book_{wb_id}.json"
                ))
                
            except Exception as e:
                logger.error(f"Error collecting world book {wb_id}: {e}")
    
    def _collect_dictionaries(
        self,
        dictionary_ids: List[str],
        work_dir: Path,
        manifest: ChatbookManifest,
        content: ChatbookContent
    ):
        """Collect chat dictionaries for export."""
        dict_dir = work_dir / "content" / "dictionaries"
        dict_dir.mkdir(parents=True, exist_ok=True)
        
        # Import the dictionary service
        from ..Character_Chat.chat_dictionary import ChatDictionaryService
        
        dict_service = ChatDictionaryService(self.db)
        
        for dict_id in dictionary_ids:
            try:
                # Get dictionary with entries
                dict_data = dict_service.get_dictionary(int(dict_id))
                if not dict_data:
                    continue
                
                # Write dictionary file
                dict_file = dict_dir / f"dictionary_{dict_id}.json"
                with open(dict_file, 'w', encoding='utf-8') as f:
                    json.dump(dict_data, f, indent=2, ensure_ascii=False)
                
                # Add to content
                content.dictionaries[dict_id] = dict_data
                
                # Add to manifest
                manifest.content_items.append(ContentItem(
                    id=dict_id,
                    type=ContentType.DICTIONARY,
                    title=dict_data.get('name', 'Unnamed'),
                    file_path=f"content/dictionaries/dictionary_{dict_id}.json"
                ))
                
            except Exception as e:
                logger.error(f"Error collecting dictionary {dict_id}: {e}")
    
    def _collect_generated_documents(
        self,
        document_ids: List[str],
        work_dir: Path,
        manifest: ChatbookManifest,
        content: ChatbookContent
    ):
        """Collect generated documents for export."""
        docs_dir = work_dir / "content" / "generated_documents"
        docs_dir.mkdir(parents=True, exist_ok=True)
        
        # Import the document generator service
        from ..Chat.document_generator import DocumentGeneratorService
        
        doc_service = DocumentGeneratorService(self.db, self.user_id)
        
        for doc_id in document_ids:
            try:
                # Get document
                doc = doc_service.get_document(doc_id)
                if not doc:
                    continue
                
                # Write document file
                doc_file = docs_dir / f"document_{doc_id}.json"
                with open(doc_file, 'w', encoding='utf-8') as f:
                    json.dump(doc, f, indent=2, ensure_ascii=False)
                
                # Add to content
                content.generated_documents[doc_id] = doc
                
                # Add to manifest
                manifest.content_items.append(ContentItem(
                    id=doc_id,
                    type=ContentType.GENERATED_DOCUMENT,
                    title=doc.get('title', 'Untitled'),
                    file_path=f"content/generated_documents/document_{doc_id}.json"
                ))
                
            except Exception as e:
                logger.error(f"Error collecting document {doc_id}: {e}")
    
    # Helper methods for importing content
    
    def _import_conversations(
        self,
        extract_dir: Path,
        manifest: ChatbookManifest,
        conversation_ids: List[str],
        conflict_resolution: ConflictResolution,
        prefix_imported: bool,
        status: ImportJob
    ):
        """Import conversations from chatbook."""
        conv_dir = extract_dir / "content" / "conversations"
        
        for conv_id in conversation_ids:
            status.processed_items += 1
            
            try:
                # Load conversation file
                conv_file = conv_dir / f"conversation_{conv_id}.json"
                if not conv_file.exists():
                    status.failed_items += 1
                    status.warnings.append(f"Conversation file not found: {conv_file.name}")
                    continue
                
                with open(conv_file, 'r', encoding='utf-8') as f:
                    conv_data = json.load(f)
                
                # Check for existing conversation
                conv_name = conv_data['name']
                if prefix_imported:
                    conv_name = f"[Imported] {conv_name}"
                
                existing = self.db.get_conversation_by_name(conv_name)
                if existing and conflict_resolution == ConflictResolution.SKIP:
                    status.skipped_items += 1
                    continue
                elif existing and conflict_resolution == ConflictResolution.RENAME:
                    conv_name = self._generate_unique_name(conv_name, "conversation")
                
                # Create conversation
                conv_dict = {
                    'title': conv_name,
                    'created_at': conv_data.get('created_at'),
                    'character_id': conv_data.get('character_id')
                }
                new_conv_id = self.db.add_conversation(conv_dict)
                
                if new_conv_id:
                    # Import messages
                    for msg in conv_data.get('messages', []):
                        msg_dict = {
                            'conversation_id': new_conv_id,
                            'sender': msg['role'],
                            'content': msg['content'],
                            'timestamp': msg.get('timestamp')
                        }
                        self.db.add_message(msg_dict)
                    
                    status.successful_items += 1
                else:
                    status.failed_items += 1
                    
            except Exception as e:
                status.failed_items += 1
                status.warnings.append(f"Error importing conversation {conv_id}: {str(e)}")
    
    def _import_notes(
        self,
        extract_dir: Path,
        manifest: ChatbookManifest,
        note_ids: List[str],
        conflict_resolution: ConflictResolution,
        prefix_imported: bool,
        status: ImportJob
    ):
        """Import notes from chatbook."""
        notes_dir = extract_dir / "content" / "notes"
        
        for note_id in note_ids:
            status.processed_items += 1
            
            try:
                # Find note file
                note_file = notes_dir / f"note_{note_id}.md"
                if not note_file.exists():
                    status.failed_items += 1
                    status.warnings.append(f"Note file not found: {note_file.name}")
                    continue
                
                # Parse markdown with frontmatter
                with open(note_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                # Extract frontmatter
                note_content = content
                note_title = f"Imported Note {note_id}"
                
                if content.startswith('---'):
                    parts = content.split('---', 2)
                    if len(parts) >= 3:
                        # Parse frontmatter for title
                        frontmatter = parts[1].strip()
                        for line in frontmatter.split('\n'):
                            if line.startswith('title:'):
                                note_title = line.replace('title:', '').strip()
                        note_content = parts[2].strip()
                
                if prefix_imported:
                    note_title = f"[Imported] {note_title}"
                
                # Check for existing note
                existing = self.db.get_note_by_title(note_title)
                if existing and conflict_resolution == ConflictResolution.SKIP:
                    status.skipped_items += 1
                    continue
                elif existing and conflict_resolution == ConflictResolution.RENAME:
                    note_title = self._generate_unique_name(note_title, "note")
                
                # Create note
                new_note_id = self.db.add_note(title=note_title, content=note_content)
                
                if new_note_id:
                    status.successful_items += 1
                else:
                    status.failed_items += 1
                    
            except Exception as e:
                status.failed_items += 1
                status.warnings.append(f"Error importing note {note_id}: {str(e)}")
    
    def _import_characters(
        self,
        extract_dir: Path,
        manifest: ChatbookManifest,
        character_ids: List[str],
        conflict_resolution: ConflictResolution,
        prefix_imported: bool,
        status: ImportJob
    ):
        """Import character cards from chatbook."""
        chars_dir = extract_dir / "content" / "characters"
        
        for char_id in character_ids:
            status.processed_items += 1
            
            try:
                # Load character file
                char_file = chars_dir / f"character_{char_id}.json"
                if not char_file.exists():
                    status.failed_items += 1
                    status.warnings.append(f"Character file not found: {char_file.name}")
                    continue
                
                with open(char_file, 'r', encoding='utf-8') as f:
                    char_data = json.load(f)
                
                # Check for existing character
                char_name = char_data.get('name', 'Unnamed')
                if prefix_imported:
                    char_name = f"[Imported] {char_name}"
                    char_data['name'] = char_name
                
                existing = self.db.get_character_card_by_name(char_name)
                if existing and conflict_resolution == ConflictResolution.SKIP:
                    status.skipped_items += 1
                    continue
                elif existing and conflict_resolution == ConflictResolution.RENAME:
                    char_name = self._generate_unique_name(char_name, "character")
                    char_data['name'] = char_name
                
                # Create character
                new_char_id = self.db.add_character_card(char_data)
                
                if new_char_id:
                    status.successful_items += 1
                else:
                    status.failed_items += 1
                    
            except Exception as e:
                status.failed_items += 1
                status.warnings.append(f"Error importing character {char_id}: {str(e)}")
    
    def _import_world_books(
        self,
        extract_dir: Path,
        manifest: ChatbookManifest,
        world_book_ids: List[str],
        conflict_resolution: ConflictResolution,
        prefix_imported: bool,
        status: ImportJob
    ):
        """Import world books from chatbook."""
        wb_dir = extract_dir / "content" / "world_books"
        
        # Import the world book service
        from ..Character_Chat.world_book_manager import WorldBookService
        wb_service = WorldBookService(self.db)
        
        for wb_id in world_book_ids:
            status.processed_items += 1
            
            try:
                # Load world book file
                wb_file = wb_dir / f"world_book_{wb_id}.json"
                if not wb_file.exists():
                    status.failed_items += 1
                    status.warnings.append(f"World book file not found: {wb_file.name}")
                    continue
                
                with open(wb_file, 'r', encoding='utf-8') as f:
                    wb_data = json.load(f)
                
                # Handle import with conflict resolution
                wb_name = wb_data.get('name', 'Unnamed')
                if prefix_imported:
                    wb_name = f"[Imported] {wb_name}"
                    wb_data['name'] = wb_name
                
                # Check for existing world book
                existing = wb_service.get_world_book_by_name(wb_name)
                if existing and conflict_resolution == ConflictResolution.SKIP:
                    status.skipped_items += 1
                    continue
                elif existing and conflict_resolution == ConflictResolution.RENAME:
                    wb_name = self._generate_unique_name(wb_name, "world_book")
                    wb_data['name'] = wb_name
                
                # Import world book
                success = wb_service.import_world_book(wb_data)
                
                if success:
                    status.successful_items += 1
                else:
                    status.failed_items += 1
                    
            except Exception as e:
                status.failed_items += 1
                status.warnings.append(f"Error importing world book {wb_id}: {str(e)}")
    
    def _import_dictionaries(
        self,
        extract_dir: Path,
        manifest: ChatbookManifest,
        dictionary_ids: List[str],
        conflict_resolution: ConflictResolution,
        prefix_imported: bool,
        status: ImportJob
    ):
        """Import chat dictionaries from chatbook."""
        dict_dir = extract_dir / "content" / "dictionaries"
        
        # Import the dictionary service
        from ..Character_Chat.chat_dictionary import ChatDictionaryService
        dict_service = ChatDictionaryService(self.db)
        
        for dict_id in dictionary_ids:
            status.processed_items += 1
            
            try:
                # Load dictionary file
                dict_file = dict_dir / f"dictionary_{dict_id}.json"
                if not dict_file.exists():
                    status.failed_items += 1
                    status.warnings.append(f"Dictionary file not found: {dict_file.name}")
                    continue
                
                with open(dict_file, 'r', encoding='utf-8') as f:
                    dict_data = json.load(f)
                
                # Handle import with conflict resolution
                dict_name = dict_data.get('name', 'Unnamed')
                if prefix_imported:
                    dict_name = f"[Imported] {dict_name}"
                    dict_data['name'] = dict_name
                
                # Check for existing dictionary
                existing = dict_service.get_dictionary_by_name(dict_name)
                if existing and conflict_resolution == ConflictResolution.SKIP:
                    status.skipped_items += 1
                    continue
                elif existing and conflict_resolution == ConflictResolution.RENAME:
                    dict_name = self._generate_unique_name(dict_name, "dictionary")
                    dict_data['name'] = dict_name
                
                # Create dictionary
                new_dict_id = dict_service.create_dictionary(
                    dict_name,
                    dict_data.get('description', ''),
                    dict_data.get('is_active', True)
                )
                
                if new_dict_id:
                    # Import entries
                    for entry in dict_data.get('entries', []):
                        dict_service.add_entry(
                            new_dict_id,
                            entry['key_pattern'],
                            entry['replacement'],
                            entry.get('is_regex', False),
                            entry.get('probability', 100),
                            entry.get('max_replacements', 1)
                        )
                    status.successful_items += 1
                else:
                    status.failed_items += 1
                    
            except Exception as e:
                status.failed_items += 1
                status.warnings.append(f"Error importing dictionary {dict_id}: {str(e)}")
    
    # Database helper methods
    
    def _save_export_job(self, job: ExportJob):
        """Save export job to database."""
        try:
            self.db.execute_query("""
                INSERT OR REPLACE INTO export_jobs (
                    job_id, user_id, status, chatbook_name, output_path,
                    created_at, started_at, completed_at, error_message,
                    progress_percentage, total_items, processed_items,
                    file_size_bytes, download_url, expires_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                job.job_id, job.user_id, job.status.value, job.chatbook_name,
                job.output_path, job.created_at.isoformat() if job.created_at else None,
                job.started_at.isoformat() if job.started_at else None,
                job.completed_at.isoformat() if job.completed_at else None,
                job.error_message, job.progress_percentage, job.total_items,
                job.processed_items, job.file_size_bytes, job.download_url,
                job.expires_at.isoformat() if job.expires_at else None
            ))
        except Exception as e:
            logger.error(f"Error saving export job: {e}")
    
    def _get_export_job(self, job_id: str) -> Optional[ExportJob]:
        """Get export job from database."""
        try:
            results = self.db.execute_query(
                "SELECT * FROM export_jobs WHERE job_id = ? AND user_id = ?",
                (job_id, self.user_id)
            )
            
            if not results:
                return None
            
            row = results[0]
            return ExportJob(
                job_id=row['job_id'],
                user_id=row['user_id'],
                status=ExportStatus(row['status']),
                chatbook_name=row['chatbook_name'],
                output_path=row['output_path'],
                created_at=datetime.fromisoformat(row['created_at']) if row['created_at'] else None,
                started_at=datetime.fromisoformat(row['started_at']) if row['started_at'] else None,
                completed_at=datetime.fromisoformat(row['completed_at']) if row['completed_at'] else None,
                error_message=row['error_message'],
                progress_percentage=row['progress_percentage'],
                total_items=row['total_items'],
                processed_items=row['processed_items'],
                file_size_bytes=row['file_size_bytes'],
                download_url=row['download_url'],
                expires_at=datetime.fromisoformat(row['expires_at']) if row['expires_at'] else None
            )
        except Exception as e:
            logger.error(f"Error getting export job: {e}")
            return None
    
    def _save_import_job(self, job: ImportJob):
        """Save import job to database."""
        try:
            self.db.execute_query("""
                INSERT OR REPLACE INTO import_jobs (
                    job_id, user_id, status, chatbook_path,
                    created_at, started_at, completed_at, error_message,
                    progress_percentage, total_items, processed_items,
                    successful_items, failed_items, skipped_items,
                    conflicts, warnings
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                job.job_id, job.user_id, job.status.value, job.chatbook_path,
                job.created_at.isoformat() if job.created_at else None,
                job.started_at.isoformat() if job.started_at else None,
                job.completed_at.isoformat() if job.completed_at else None,
                job.error_message, job.progress_percentage, job.total_items,
                job.processed_items, job.successful_items, job.failed_items,
                job.skipped_items, json.dumps(job.conflicts), json.dumps(job.warnings)
            ))
        except Exception as e:
            logger.error(f"Error saving import job: {e}")
    
    def _get_import_job(self, job_id: str) -> Optional[ImportJob]:
        """Get import job from database."""
        try:
            results = self.db.execute_query(
                "SELECT * FROM import_jobs WHERE job_id = ? AND user_id = ?",
                (job_id, self.user_id)
            )
            
            if not results:
                return None
            
            row = results[0]
            return ImportJob(
                job_id=row['job_id'],
                user_id=row['user_id'],
                status=ImportStatus(row['status']),
                chatbook_path=row['chatbook_path'],
                created_at=datetime.fromisoformat(row['created_at']) if row['created_at'] else None,
                started_at=datetime.fromisoformat(row['started_at']) if row['started_at'] else None,
                completed_at=datetime.fromisoformat(row['completed_at']) if row['completed_at'] else None,
                error_message=row['error_message'],
                progress_percentage=row['progress_percentage'],
                total_items=row['total_items'],
                processed_items=row['processed_items'],
                successful_items=row['successful_items'],
                failed_items=row['failed_items'],
                skipped_items=row['skipped_items'],
                conflicts=json.loads(row['conflicts']) if row['conflicts'] else [],
                warnings=json.loads(row['warnings']) if row['warnings'] else []
            )
        except Exception as e:
            logger.error(f"Error getting import job: {e}")
            return None
    
    def _generate_unique_name(self, base_name: str, item_type: str) -> str:
        """Generate a unique name for an item."""
        counter = 1
        while True:
            new_name = f"{base_name} ({counter})"
            
            # Check if name exists based on item type
            if item_type == "conversation":
                if not self.db.get_conversation_by_name(new_name):
                    return new_name
            elif item_type == "note":
                if not self.db.get_note_by_title(new_name):
                    return new_name
            elif item_type == "character":
                if not self.db.get_character_card_by_name(new_name):
                    return new_name
            elif item_type == "world_book":
                # Check in world books table
                result = self.db.execute_query(
                    "SELECT id FROM world_books WHERE name = ?",
                    (new_name,)
                )
                if not result:
                    return new_name
            elif item_type == "dictionary":
                # Check in dictionaries table
                result = self.db.execute_query(
                    "SELECT id FROM chat_dictionaries WHERE name = ?",
                    (new_name,)
                )
                if not result:
                    return new_name
            
            counter += 1
    
    async def _create_readme_async(self, work_dir: Path, manifest: ChatbookManifest):
        """Create README file for the chatbook asynchronously."""
        readme_path = work_dir / "README.md"
        
        content = []
        content.append(f"# {manifest.name}\n\n")
        content.append(f"{manifest.description}\n\n")
        
        if manifest.author:
            content.append(f"**Author:** {manifest.author}\n\n")
        
        content.append(f"**Created:** {manifest.created_at.strftime('%Y-%m-%d %H:%M')}\n\n")
        content.append("## Contents\n\n")
        
        if manifest.total_conversations > 0:
            content.append(f"- **Conversations:** {manifest.total_conversations}\n")
        if manifest.total_notes > 0:
            content.append(f"- **Notes:** {manifest.total_notes}\n")
        if manifest.total_characters > 0:
            content.append(f"- **Characters:** {manifest.total_characters}\n")
        if manifest.total_world_books > 0:
            content.append(f"- **World Books:** {manifest.total_world_books}\n")
        if manifest.total_dictionaries > 0:
            content.append(f"- **Dictionaries:** {manifest.total_dictionaries}\n")
        if manifest.total_documents > 0:
            content.append(f"- **Generated Documents:** {manifest.total_documents}\n")
        
        if manifest.tags:
            content.append(f"\n## Tags\n\n{', '.join(manifest.tags)}\n")
        
        content.append("\n## License\n\n")
        content.append(manifest.license or "See individual content files for licensing information.")
        
        async with aiofiles.open(readme_path, 'w', encoding='utf-8') as f:
            await f.write(''.join(content))
    
    def _create_readme(self, work_dir: Path, manifest: ChatbookManifest):
        """Create README file for the chatbook (sync version for backwards compatibility)."""
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
            if manifest.total_world_books > 0:
                f.write(f"- **World Books:** {manifest.total_world_books}\n")
            if manifest.total_dictionaries > 0:
                f.write(f"- **Dictionaries:** {manifest.total_dictionaries}\n")
            if manifest.total_documents > 0:
                f.write(f"- **Generated Documents:** {manifest.total_documents}\n")
            
            if manifest.tags:
                f.write(f"\n## Tags\n\n{', '.join(manifest.tags)}\n")
            
            f.write("\n## License\n\n")
            f.write(manifest.license or "See individual content files for licensing information.")
    
    def _validate_zip_file(self, file_path: str) -> bool:
        """
        Validate a ZIP file for safety and integrity.
        
        Args:
            file_path: Path to the ZIP file
            
        Returns:
            True if valid and safe, False otherwise
        """
        try:
            # Check file exists and has reasonable size
            file_path_obj = Path(file_path)
            if not file_path_obj.exists():
                return False
            
            file_size = file_path_obj.stat().st_size
            if file_size > 100 * 1024 * 1024:  # 100MB compressed limit
                logger.warning(f"ZIP file too large: {file_size} bytes")
                return False
            
            # Verify it's actually a ZIP file (check magic bytes)
            with open(file_path, 'rb') as f:
                magic = f.read(4)
                if magic[:2] != b'PK':  # ZIP magic number
                    logger.warning("File is not a valid ZIP archive")
                    return False
            
            # Test ZIP integrity
            with zipfile.ZipFile(file_path, 'r') as zf:
                # Check for path traversal attempts
                for name in zf.namelist():
                    if os.path.isabs(name) or ".." in name or name.startswith("/"):
                        logger.warning(f"Unsafe path in ZIP: {name}")
                        return False
                
                # Test CRC integrity
                result = zf.testzip()
                if result is not None:
                    logger.warning(f"ZIP file corrupt: {result}")
                    return False
            
            return True
            
        except (zipfile.BadZipFile, OSError) as e:
            logger.error(f"Invalid ZIP file: {e}")
            return False
        except Exception as e:
            logger.error(f"Error validating ZIP file: {e}")
            return False
    
    async def _create_zip_archive_async(self, work_dir: Path, output_path: Path):
        """Create ZIP archive of the chatbook asynchronously with compression limits."""
        def _create_archive():
            with zipfile.ZipFile(output_path, 'w', zipfile.ZIP_DEFLATED, compresslevel=6) as zf:
                total_size = 0
                for file_path in work_dir.rglob('*'):
                    if file_path.is_file():
                        # Check individual file size
                        file_size = file_path.stat().st_size
                        if file_size > 50 * 1024 * 1024:  # 50MB per file limit
                            logger.warning(f"Skipping large file: {file_path} ({file_size} bytes)")
                            continue
                        
                        total_size += file_size
                        if total_size > 500 * 1024 * 1024:  # 500MB total limit
                            raise ValueError("Archive size exceeds 500MB limit")
                        
                        arcname = file_path.relative_to(work_dir)
                        zf.write(file_path, arcname)
        
        # Run in thread pool to avoid blocking
        await asyncio.to_thread(_create_archive)
    
    def _create_zip_archive(self, work_dir: Path, output_path: Path):
        """Create ZIP archive of the chatbook with compression limits (sync version)."""
        with zipfile.ZipFile(output_path, 'w', zipfile.ZIP_DEFLATED, compresslevel=6) as zf:
            total_size = 0
            for file_path in work_dir.rglob('*'):
                if file_path.is_file():
                    # Check individual file size
                    file_size = file_path.stat().st_size
                    if file_size > 50 * 1024 * 1024:  # 50MB per file limit
                        logger.warning(f"Skipping large file: {file_path} ({file_size} bytes)")
                        continue
                    
                    total_size += file_size
                    if total_size > 500 * 1024 * 1024:  # 500MB total limit
                        raise ValueError("Archive size exceeds 500MB limit")
                    
                    arcname = file_path.relative_to(work_dir)
                    zf.write(file_path, arcname)