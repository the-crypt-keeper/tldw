# chatbook_schemas.py
# Description: Pydantic schemas for chatbook import/export operations
#
"""
Chatbook Schemas
----------------

Pydantic models for chatbook creation, import, export, and preview operations.
"""

from datetime import datetime
from typing import List, Dict, Optional, Any
from pydantic import BaseModel, Field
from enum import Enum


# Enums

class ChatbookVersion(str, Enum):
    """Chatbook format versions."""
    V1 = "1.0"
    V2 = "2.0"  # Future version


class ContentType(str, Enum):
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


class ExportStatus(str, Enum):
    """Status of export job."""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    EXPIRED = "expired"


class ImportStatus(str, Enum):
    """Status of import job."""
    PENDING = "pending"
    VALIDATING = "validating"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class ConflictResolution(str, Enum):
    """How to handle conflicts during import."""
    SKIP = "skip"          # Skip conflicting items
    OVERWRITE = "overwrite"  # Overwrite existing items
    RENAME = "rename"      # Rename imported items
    MERGE = "merge"        # Merge with existing (where applicable)


# Request Schemas

class CreateChatbookRequest(BaseModel):
    """Request for creating a chatbook."""
    name: str = Field(..., description="Name of the chatbook")
    description: str = Field(..., description="Description of the chatbook")
    content_selections: Dict[ContentType, List[str]] = Field(
        ..., 
        description="Content to include by type and IDs"
    )
    author: Optional[str] = Field(None, description="Author name")
    include_media: bool = Field(False, description="Include media files")
    media_quality: str = Field("compressed", description="Media quality level (thumbnail/compressed/original)")
    include_embeddings: bool = Field(False, description="Include embeddings")
    include_generated_content: bool = Field(True, description="Include generated documents")
    tags: List[str] = Field(default_factory=list, description="Chatbook tags")
    categories: List[str] = Field(default_factory=list, description="Chatbook categories")
    async_mode: bool = Field(False, description="Run as background job")
    
    class Config:
        schema_extra = {
            "example": {
                "name": "My Research Chatbook",
                "description": "Collection of research conversations and notes",
                "content_selections": {
                    "conversation": ["conv123", "conv456"],
                    "note": ["note789"],
                    "character": ["char001"]
                },
                "author": "Jane Doe",
                "include_media": False,
                "media_quality": "compressed",
                "include_embeddings": False,
                "include_generated_content": True,
                "tags": ["research", "AI"],
                "categories": ["Work"],
                "async_mode": False
            }
        }


class ImportChatbookRequest(BaseModel):
    """Request for importing a chatbook."""
    content_selections: Optional[Dict[ContentType, List[str]]] = Field(
        None,
        description="Specific content to import, or None for all"
    )
    conflict_resolution: ConflictResolution = Field(
        ConflictResolution.SKIP,
        description="How to handle conflicts"
    )
    prefix_imported: bool = Field(
        False,
        description="Add [Imported] prefix to items"
    )
    import_media: bool = Field(True, description="Import media files")
    import_embeddings: bool = Field(False, description="Import embeddings")
    async_mode: bool = Field(False, description="Run as background job")
    
    class Config:
        schema_extra = {
            "example": {
                "conflict_resolution": "skip",
                "prefix_imported": True,
                "import_media": False,
                "import_embeddings": False,
                "async_mode": False
            }
        }


# Response Schemas

class ContentItemResponse(BaseModel):
    """Individual content item in a chatbook."""
    id: str
    type: ContentType
    title: str
    description: Optional[str] = None
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None
    tags: List[str] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)
    file_path: Optional[str] = None
    checksum: Optional[str] = None


class ChatbookManifestResponse(BaseModel):
    """Chatbook manifest information."""
    version: ChatbookVersion
    name: str
    description: str
    author: Optional[str] = None
    created_at: datetime
    updated_at: datetime
    export_id: Optional[str] = None
    
    # Content summary
    content_items: List[ContentItemResponse] = Field(default_factory=list)
    
    # Configuration
    include_media: bool = False
    include_embeddings: bool = False
    include_generated_content: bool = True
    media_quality: str = "compressed"
    max_file_size_mb: int = 100
    
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
    tags: List[str] = Field(default_factory=list)
    categories: List[str] = Field(default_factory=list)
    language: str = "en"
    license: Optional[str] = None


class ExportJobResponse(BaseModel):
    """Export job status."""
    job_id: str
    status: ExportStatus
    chatbook_name: str
    output_path: Optional[str] = None
    created_at: datetime
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    error_message: Optional[str] = None
    progress_percentage: int = 0
    total_items: int = 0
    processed_items: int = 0
    file_size_bytes: Optional[int] = None
    download_url: Optional[str] = None
    expires_at: Optional[datetime] = None


class ImportJobResponse(BaseModel):
    """Import job status."""
    job_id: str
    status: ImportStatus
    chatbook_path: str
    created_at: datetime
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    error_message: Optional[str] = None
    progress_percentage: int = 0
    total_items: int = 0
    processed_items: int = 0
    successful_items: int = 0
    failed_items: int = 0
    skipped_items: int = 0
    conflicts: List[Dict[str, Any]] = Field(default_factory=list)
    warnings: List[str] = Field(default_factory=list)


class ImportConflictResponse(BaseModel):
    """Details about an import conflict."""
    item_id: str
    item_type: ContentType
    item_title: str
    existing_id: str
    existing_title: str
    suggested_resolution: ConflictResolution
    user_resolution: Optional[ConflictResolution] = None
    new_title: Optional[str] = None


class CreateChatbookResponse(BaseModel):
    """Response for chatbook creation."""
    success: bool
    message: str
    job_id: Optional[str] = Field(None, description="Job ID if async mode")
    file_path: Optional[str] = Field(None, description="File path if sync mode")
    download_url: Optional[str] = Field(None, description="Download URL if sync mode")


class ImportChatbookResponse(BaseModel):
    """Response for chatbook import."""
    success: bool
    message: str
    job_id: Optional[str] = Field(None, description="Job ID if async mode")
    imported_items: Optional[Dict[str, int]] = Field(
        None,
        description="Count of imported items by type"
    )


class PreviewChatbookResponse(BaseModel):
    """Response for chatbook preview."""
    manifest: Optional[ChatbookManifestResponse] = None
    error: Optional[str] = None


class ListExportJobsResponse(BaseModel):
    """Response for listing export jobs."""
    jobs: List[ExportJobResponse]
    total: int


class ListImportJobsResponse(BaseModel):
    """Response for listing import jobs."""
    jobs: List[ImportJobResponse]
    total: int


class CleanupExpiredExportsResponse(BaseModel):
    """Response for cleanup operation."""
    deleted_count: int
    message: str


class DownloadChatbookResponse(BaseModel):
    """Response for download request."""
    file_path: str
    file_name: str
    content_type: str = "application/zip"
    file_size: int


# Query Parameters

class ListJobsQuery(BaseModel):
    """Query parameters for listing jobs."""
    status: Optional[str] = Field(None, description="Filter by status")
    limit: int = Field(100, ge=1, le=1000, description="Maximum results")
    offset: int = Field(0, ge=0, description="Offset for pagination")
    order_by: str = Field("created_at", description="Sort field")
    order_desc: bool = Field(True, description="Sort descending")


# Error Responses

class ChatbookErrorResponse(BaseModel):
    """Error response for chatbook operations."""
    detail: str
    error_type: str
    job_id: Optional[str] = None
    suggestions: List[str] = Field(default_factory=list)