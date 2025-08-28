# chatbook_models.py
# Description: Data models for chatbook/knowledge pack structures with multi-user support
# Adapted from single-user to multi-user architecture
#
"""
Chatbook Models for Multi-User Environment
------------------------------------------

Defines the data structures for chatbooks including manifest,
content organization, and metadata with user isolation.

Key Adaptations from Single-User:
- User-specific exports with access control
- Sanitized content to prevent cross-user data leakage
- Job-based export/import for large operations
- Temporary storage with automatic cleanup
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Dict, Optional, Any, Set
from enum import Enum
import json
from typing import Dict, Any


class ChatbookVersion(Enum):
    """Chatbook format versions."""
    V1 = "1.0"
    V2 = "2.0"  # Future version with enhanced features


class ContentType(Enum):
    """Types of content that can be included in a chatbook."""
    CONVERSATION = "conversation"
    NOTE = "note"
    CHARACTER = "character"
    MEDIA = "media"
    EMBEDDING = "embedding"
    PROMPT = "prompt"
    EVALUATION = "evaluation"
    WORLD_BOOK = "world_book"
    DICTIONARY = "dictionary"
    GENERATED_DOCUMENT = "generated_document"


class ExportStatus(Enum):
    """Status of export job."""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class ImportStatus(Enum):
    """Status of import job."""
    PENDING = "pending"
    VALIDATING = "validating"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class ConflictResolution(Enum):
    """How to handle conflicts during import."""
    SKIP = "skip"          # Skip conflicting items
    OVERWRITE = "overwrite"  # Overwrite existing items
    RENAME = "rename"      # Rename imported items
    MERGE = "merge"        # Merge with existing (where applicable)
    ASK = "ask"           # Ask user for each conflict (not for API)


@dataclass
class ImportStatusData:
    """Import status tracking data."""
    total_items: int = 0
    successful_items: int = 0
    failed_items: int = 0
    skipped_items: int = 0
    conflicts: List[Dict[str, Any]] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "total_items": self.total_items,
            "successful_items": self.successful_items,
            "failed_items": self.failed_items,
            "skipped_items": self.skipped_items,
            "conflicts": self.conflicts,
            "warnings": self.warnings
        }


@dataclass
class ContentItem:
    """Individual content item in a chatbook."""
    id: str
    type: ContentType
    title: str
    description: Optional[str] = None
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None
    tags: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    file_path: Optional[str] = None  # Relative path within chatbook
    checksum: Optional[str] = None  # For integrity verification
    
    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "id": self.id,
            "type": self.type.value if hasattr(self.type, 'value') else str(self.type),
            "title": self.title,
            "description": self.description,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
            "tags": self.tags,
            "metadata": self.metadata,
            "file_path": self.file_path,
            "checksum": self.checksum
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> 'ContentItem':
        """Create ContentItem from dictionary."""
        return cls(
            id=data["id"],
            type=ContentType(data["type"]),
            title=data["title"],
            description=data.get("description"),
            created_at=datetime.fromisoformat(data["created_at"]) if data.get("created_at") else None,
            updated_at=datetime.fromisoformat(data["updated_at"]) if data.get("updated_at") else None,
            tags=data.get("tags", []),
            metadata=data.get("metadata", {}),
            file_path=data.get("file_path"),
            checksum=data.get("checksum")
        )


@dataclass
class Relationship:
    """Relationship between content items."""
    source_id: str
    target_id: str
    relationship_type: str  # e.g., "references", "parent_of", "requires", "uses_character"
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "source_id": self.source_id,
            "target_id": self.target_id,
            "relationship_type": self.relationship_type,
            "metadata": self.metadata
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> 'Relationship':
        """Create Relationship from dictionary."""
        return cls(
            source_id=data["source_id"],
            target_id=data["target_id"],
            relationship_type=data["relationship_type"],
            metadata=data.get("metadata", {})
        )


@dataclass
class ChatbookManifest:
    """Manifest file containing chatbook metadata and contents listing."""
    version: ChatbookVersion
    name: str
    description: str
    author: Optional[str] = None
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)
    export_id: Optional[str] = None  # Unique export identifier
    exported_at: Optional[str] = None  # For compatibility with tests
    user_id: Optional[str] = None  # User who created the export
    
    # Content summary
    content_items: List[ContentItem] = field(default_factory=list)
    relationships: List[Relationship] = field(default_factory=list)
    content_summary: Optional[Dict[str, int]] = field(default_factory=dict)  # For compatibility
    metadata: Optional[Dict[str, Any]] = field(default_factory=dict)  # For compatibility
    
    # Configuration
    include_media: bool = False
    include_embeddings: bool = False
    include_generated_content: bool = True
    media_quality: str = "compressed"  # thumbnail, compressed, original
    max_file_size_mb: int = 100  # Maximum chatbook size
    
    # Statistics
    total_conversations: int = 0
    total_notes: int = 0
    total_characters: int = 0
    total_media_items: int = 0
    total_world_books: int = 0
    total_dictionaries: int = 0
    total_documents: int = 0
    total_size_bytes: int = 0
    
    # Metadata
    tags: List[str] = field(default_factory=list)
    categories: List[str] = field(default_factory=list)
    language: str = "en"
    license: Optional[str] = None
    
    # User info (sanitized for export)
    user_id: Optional[str] = None  # Anonymized user identifier
    
    def to_dict(self) -> dict:
        """Convert manifest to dictionary for JSON serialization."""
        return {
            "version": self.version.value if hasattr(self.version, 'value') else str(self.version),
            "name": self.name,
            "description": self.description,
            "author": self.author,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "export_id": self.export_id,
            "content_items": [item.to_dict() for item in self.content_items],
            "relationships": [rel.to_dict() for rel in self.relationships],
            "configuration": {
                "include_media": self.include_media,
                "include_embeddings": self.include_embeddings,
                "include_generated_content": self.include_generated_content,
                "media_quality": self.media_quality,
                "max_file_size_mb": self.max_file_size_mb
            },
            "statistics": {
                "total_conversations": self.total_conversations,
                "total_notes": self.total_notes,
                "total_characters": self.total_characters,
                "total_media_items": self.total_media_items,
                "total_world_books": self.total_world_books,
                "total_dictionaries": self.total_dictionaries,
                "total_documents": self.total_documents,
                "total_size_bytes": self.total_size_bytes
            },
            "metadata": {
                "tags": self.tags,
                "categories": self.categories,
                "language": self.language,
                "license": self.license
            },
            "user_info": {
                "user_id": self.user_id
            }
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> 'ChatbookManifest':
        """Create ChatbookManifest from dictionary."""
        config = data.get("configuration", {})
        stats = data.get("statistics", {})
        meta = data.get("metadata", {})
        user = data.get("user_info", {})
        
        return cls(
            version=ChatbookVersion(data["version"]),
            name=data["name"],
            description=data["description"],
            author=data.get("author"),
            created_at=datetime.fromisoformat(data["created_at"]) if data.get("created_at") else datetime.utcnow(),
            updated_at=datetime.fromisoformat(data["updated_at"]) if data.get("updated_at") else datetime.utcnow(),
            export_id=data.get("export_id"),
            content_items=[ContentItem.from_dict(item) for item in data.get("content_items", [])],
            relationships=[Relationship.from_dict(rel) for rel in data.get("relationships", [])],
            include_media=config.get("include_media", False),
            include_embeddings=config.get("include_embeddings", False),
            include_generated_content=config.get("include_generated_content", True),
            media_quality=config.get("media_quality", "compressed"),
            max_file_size_mb=config.get("max_file_size_mb", 100),
            total_conversations=stats.get("total_conversations", 0),
            total_notes=stats.get("total_notes", 0),
            total_characters=stats.get("total_characters", 0),
            total_media_items=stats.get("total_media_items", 0),
            total_world_books=stats.get("total_world_books", 0),
            total_dictionaries=stats.get("total_dictionaries", 0),
            total_documents=stats.get("total_documents", 0),
            total_size_bytes=stats.get("total_size_bytes", 0),
            tags=meta.get("tags", []),
            categories=meta.get("categories", []),
            language=meta.get("language", "en"),
            license=meta.get("license"),
            user_id=user.get("user_id")
        )


@dataclass
class ChatbookContent:
    """Container for all content in a chatbook."""
    conversations: Dict[str, Any] = field(default_factory=dict)
    notes: Dict[str, Any] = field(default_factory=dict)
    characters: Dict[str, Any] = field(default_factory=dict)
    media: Dict[str, Any] = field(default_factory=dict)
    embeddings: Dict[str, Any] = field(default_factory=dict)
    prompts: Dict[str, Any] = field(default_factory=dict)
    evaluations: Dict[str, Any] = field(default_factory=dict)
    world_books: Dict[str, Any] = field(default_factory=dict)
    dictionaries: Dict[str, Any] = field(default_factory=dict)
    generated_documents: Dict[str, Any] = field(default_factory=dict)
    
    def get_all_ids(self) -> Set[str]:
        """Get all content IDs."""
        all_ids = set()
        for content_dict in [
            self.conversations, self.notes, self.characters,
            self.media, self.embeddings, self.prompts,
            self.evaluations, self.world_books, self.dictionaries,
            self.generated_documents
        ]:
            all_ids.update(content_dict.keys())
        return all_ids


@dataclass
class ExportJob:
    """Track export job status."""
    job_id: str
    user_id: str
    status: ExportStatus
    chatbook_name: str
    output_path: Optional[str] = None
    created_at: datetime = field(default_factory=datetime.utcnow)
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    error_message: Optional[str] = None
    progress_percentage: int = 0
    total_items: int = 0
    processed_items: int = 0
    file_size_bytes: Optional[int] = None
    download_url: Optional[str] = None
    expires_at: Optional[datetime] = None
    metadata: Optional[Dict[str, Any]] = field(default_factory=dict)  # For storing additional data
    
    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "job_id": self.job_id,
            "user_id": self.user_id,
            "status": self.status.value,
            "chatbook_name": self.chatbook_name,
            "output_path": self.output_path,
            "created_at": self.created_at.isoformat(),
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "error_message": self.error_message,
            "progress_percentage": self.progress_percentage,
            "total_items": self.total_items,
            "processed_items": self.processed_items,
            "file_size_bytes": self.file_size_bytes,
            "download_url": self.download_url,
            "expires_at": self.expires_at.isoformat() if self.expires_at else None,
            "metadata": self.metadata
        }


@dataclass
class ImportJob:
    """Track import job status."""
    job_id: str
    user_id: str
    status: ImportStatus
    chatbook_path: str
    created_at: datetime = field(default_factory=datetime.utcnow)
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    error_message: Optional[str] = None
    progress_percentage: int = 0
    total_items: int = 0
    processed_items: int = 0
    successful_items: int = 0
    failed_items: int = 0
    skipped_items: int = 0
    conflicts: List[Dict[str, Any]] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    
    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "job_id": self.job_id,
            "user_id": self.user_id,
            "status": self.status.value,
            "chatbook_path": self.chatbook_path,
            "created_at": self.created_at.isoformat(),
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "error_message": self.error_message,
            "progress_percentage": self.progress_percentage,
            "total_items": self.total_items,
            "processed_items": self.processed_items,
            "successful_items": self.successful_items,
            "failed_items": self.failed_items,
            "skipped_items": self.skipped_items,
            "conflicts": self.conflicts,
            "warnings": self.warnings
        }


@dataclass
class ImportConflict:
    """Represents a conflict during import."""
    item_id: str
    item_type: ContentType
    item_title: str
    existing_id: str
    existing_title: str
    suggested_resolution: ConflictResolution
    user_resolution: Optional[ConflictResolution] = None
    new_title: Optional[str] = None  # For rename resolution
    
    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "item_id": self.item_id,
            "item_type": self.item_type.value,
            "item_title": self.item_title,
            "existing_id": self.existing_id,
            "existing_title": self.existing_title,
            "suggested_resolution": self.suggested_resolution.value,
            "user_resolution": self.user_resolution.value if self.user_resolution else None,
            "new_title": self.new_title
        }


# Export main classes
__all__ = [
    'ChatbookVersion',
    'ContentType',
    'ExportStatus',
    'ImportStatus',
    'ConflictResolution',
    'ContentItem',
    'Relationship',
    'ChatbookManifest',
    'ChatbookContent',
    'ExportJob',
    'ImportJob',
    'ImportConflict'
]