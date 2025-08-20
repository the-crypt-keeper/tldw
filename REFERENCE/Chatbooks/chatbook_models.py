# chatbook_models.py
# Description: Data models for chatbook/knowledge pack structures
#
"""
Chatbook Models
---------------

Defines the data structures for chatbooks including manifest,
content organization, and metadata.
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Dict, Optional, Any
from pathlib import Path
from enum import Enum


class ChatbookVersion(Enum):
    """Chatbook format versions."""
    V1 = "1.0"
    V2 = "2.0"  # Future version


class ContentType(Enum):
    """Types of content that can be included in a chatbook."""
    CONVERSATION = "conversation"
    NOTE = "note"
    CHARACTER = "character"
    MEDIA = "media"
    EMBEDDING = "embedding"
    PROMPT = "prompt"
    EVALUATION = "evaluation"


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
    
    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "id": self.id,
            "type": self.type.value,
            "title": self.title,
            "description": self.description,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
            "tags": self.tags,
            "metadata": self.metadata,
            "file_path": self.file_path
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
            file_path=data.get("file_path")
        )


@dataclass
class Relationship:
    """Relationship between content items."""
    source_id: str
    target_id: str
    relationship_type: str  # e.g., "references", "parent_of", "requires"
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "source_id": self.source_id,
            "target_id": self.target_id,
            "relationship_type": self.relationship_type,
            "metadata": self.metadata
        }


@dataclass
class ChatbookManifest:
    """Manifest file containing chatbook metadata and contents listing."""
    version: ChatbookVersion
    name: str
    description: str
    author: Optional[str] = None
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    
    # Content summary
    content_items: List[ContentItem] = field(default_factory=list)
    relationships: List[Relationship] = field(default_factory=list)
    
    # Configuration
    include_media: bool = False
    include_embeddings: bool = False
    media_quality: str = "thumbnail"  # thumbnail, compressed, original
    
    # Statistics
    total_conversations: int = 0
    total_notes: int = 0
    total_characters: int = 0
    total_media_items: int = 0
    total_size_bytes: int = 0
    
    # Metadata
    tags: List[str] = field(default_factory=list)
    categories: List[str] = field(default_factory=list)
    language: str = "en"
    license: Optional[str] = None
    
    def to_dict(self) -> dict:
        """Convert manifest to dictionary for JSON serialization."""
        return {
            "version": self.version.value,
            "name": self.name,
            "description": self.description,
            "author": self.author,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "content_items": [item.to_dict() for item in self.content_items],
            "relationships": [rel.to_dict() for rel in self.relationships],
            "include_media": self.include_media,
            "include_embeddings": self.include_embeddings,
            "media_quality": self.media_quality,
            "statistics": {
                "total_conversations": self.total_conversations,
                "total_notes": self.total_notes,
                "total_characters": self.total_characters,
                "total_media_items": self.total_media_items,
                "total_size_bytes": self.total_size_bytes
            },
            "tags": self.tags,
            "categories": self.categories,
            "language": self.language,
            "license": self.license
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> 'ChatbookManifest':
        """Create ChatbookManifest from dictionary."""
        manifest = cls(
            version=ChatbookVersion(data["version"]),
            name=data["name"],
            description=data["description"],
            author=data.get("author"),
            created_at=datetime.fromisoformat(data["created_at"]),
            updated_at=datetime.fromisoformat(data["updated_at"])
        )
        
        # Load content items
        manifest.content_items = [
            ContentItem.from_dict(item) for item in data.get("content_items", [])
        ]
        
        # Load relationships
        manifest.relationships = [
            Relationship(**rel) for rel in data.get("relationships", [])
        ]
        
        # Load configuration
        manifest.include_media = data.get("include_media", False)
        manifest.include_embeddings = data.get("include_embeddings", False)
        manifest.media_quality = data.get("media_quality", "thumbnail")
        
        # Load statistics
        stats = data.get("statistics", {})
        manifest.total_conversations = stats.get("total_conversations", 0)
        manifest.total_notes = stats.get("total_notes", 0)
        manifest.total_characters = stats.get("total_characters", 0)
        manifest.total_media_items = stats.get("total_media_items", 0)
        manifest.total_size_bytes = stats.get("total_size_bytes", 0)
        
        # Load metadata
        manifest.tags = data.get("tags", [])
        manifest.categories = data.get("categories", [])
        manifest.language = data.get("language", "en")
        manifest.license = data.get("license")
        
        return manifest


@dataclass
class ChatbookContent:
    """Container for all chatbook content."""
    conversations: List[Dict[str, Any]] = field(default_factory=list)
    notes: List[Dict[str, Any]] = field(default_factory=list)
    characters: List[Dict[str, Any]] = field(default_factory=list)
    media_items: List[Dict[str, Any]] = field(default_factory=list)
    embeddings: List[Dict[str, Any]] = field(default_factory=list)
    prompts: List[Dict[str, Any]] = field(default_factory=list)
    evaluations: List[Dict[str, Any]] = field(default_factory=list)


@dataclass
class Chatbook:
    """Complete chatbook structure."""
    manifest: ChatbookManifest
    content: ChatbookContent
    base_path: Optional[Path] = None
    
    def get_content_by_type(self, content_type: ContentType) -> List[ContentItem]:
        """Get all content items of a specific type."""
        return [
            item for item in self.manifest.content_items
            if item.type == content_type
        ]
    
    def get_content_by_id(self, content_id: str) -> Optional[ContentItem]:
        """Get a specific content item by ID."""
        for item in self.manifest.content_items:
            if item.id == content_id:
                return item
        return None
    
    def get_related_content(self, content_id: str) -> List[ContentItem]:
        """Get all content items related to a specific item."""
        related_ids = set()
        
        # Find relationships where this item is source or target
        for rel in self.manifest.relationships:
            if rel.source_id == content_id:
                related_ids.add(rel.target_id)
            elif rel.target_id == content_id:
                related_ids.add(rel.source_id)
        
        # Return the actual content items
        return [
            item for item in self.manifest.content_items
            if item.id in related_ids
        ]