# chunking_templates_schemas.py
"""
Pydantic schemas for chunking template API endpoints.
"""

from typing import Optional, List, Dict, Any
from datetime import datetime
from pydantic import BaseModel, Field, validator
import json


class ChunkingTemplateBase(BaseModel):
    """Base schema for chunking template data."""
    name: str = Field(..., description="Template name (must be unique)")
    description: Optional[str] = Field(None, description="Template description")
    tags: Optional[List[str]] = Field(default_factory=list, description="Tags for categorization")


class TemplateConfig(BaseModel):
    """Template configuration structure."""
    preprocessing: Optional[List[Dict[str, Any]]] = Field(
        default_factory=list,
        description="List of preprocessing operations"
    )
    chunking: Dict[str, Any] = Field(
        ...,
        description="Chunking method configuration"
    )
    postprocessing: Optional[List[Dict[str, Any]]] = Field(
        default_factory=list,
        description="List of postprocessing operations"
    )
    
    @validator('chunking')
    def validate_chunking(cls, v):
        """Validate chunking configuration has required fields."""
        if 'method' not in v:
            raise ValueError("Chunking configuration must include 'method'")
        if 'config' not in v:
            v['config'] = {}
        return v


class ChunkingTemplateCreate(ChunkingTemplateBase):
    """Schema for creating a new chunking template."""
    template: TemplateConfig = Field(..., description="Template configuration")
    user_id: Optional[str] = Field(None, description="User ID for ownership tracking")


class ChunkingTemplateUpdate(BaseModel):
    """Schema for updating an existing chunking template."""
    description: Optional[str] = Field(None, description="New template description")
    tags: Optional[List[str]] = Field(None, description="New tags list")
    template: Optional[TemplateConfig] = Field(None, description="New template configuration")


class ChunkingTemplateResponse(ChunkingTemplateBase):
    """Schema for chunking template response."""
    id: int = Field(..., description="Template database ID")
    uuid: str = Field(..., description="Template UUID")
    template_json: str = Field(..., description="Template configuration as JSON string")
    is_builtin: bool = Field(False, description="Whether this is a built-in template")
    created_at: datetime = Field(..., description="Creation timestamp")
    updated_at: datetime = Field(..., description="Last update timestamp")
    version: int = Field(1, description="Template version number")
    user_id: Optional[str] = Field(None, description="User ID of template owner")
    
    class Config:
        orm_mode = True
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }
    
    @validator('template_json', pre=False)
    def ensure_json_string(cls, v):
        """Ensure template_json is a string."""
        if isinstance(v, dict):
            return json.dumps(v)
        return v


class ChunkingTemplateListResponse(BaseModel):
    """Response schema for listing chunking templates."""
    templates: List[ChunkingTemplateResponse] = Field(
        ...,
        description="List of chunking templates"
    )
    total: int = Field(..., description="Total number of templates")
    
    class Config:
        orm_mode = True


class ChunkingTemplateFilter(BaseModel):
    """Schema for filtering chunking templates."""
    include_builtin: bool = Field(True, description="Include built-in templates")
    include_custom: bool = Field(True, description="Include custom templates")
    tags: Optional[List[str]] = Field(None, description="Filter by tags")
    user_id: Optional[str] = Field(None, description="Filter by user ID")


class ApplyTemplateRequest(BaseModel):
    """Request schema for applying a template to text."""
    template_name: str = Field(..., description="Name of the template to apply")
    text: str = Field(..., description="Text to chunk using the template")
    override_options: Optional[Dict[str, Any]] = Field(
        None,
        description="Options to override template defaults"
    )


class ApplyTemplateResponse(BaseModel):
    """Response schema for template application."""
    template_name: str = Field(..., description="Applied template name")
    chunks: List[str] = Field(..., description="Resulting text chunks")
    metadata: Optional[Dict[str, Any]] = Field(
        None,
        description="Additional metadata from processing"
    )


class TemplateValidationError(BaseModel):
    """Schema for template validation errors."""
    field: str = Field(..., description="Field with validation error")
    message: str = Field(..., description="Error message")
    
    
class TemplateValidationResponse(BaseModel):
    """Response schema for template validation."""
    valid: bool = Field(..., description="Whether the template is valid")
    errors: Optional[List[TemplateValidationError]] = Field(
        None,
        description="List of validation errors if invalid"
    )
    warnings: Optional[List[str]] = Field(
        None,
        description="Non-critical warnings about the template"
    )


class BulkTemplateOperation(BaseModel):
    """Schema for bulk template operations."""
    operation: str = Field(
        ...,
        description="Operation to perform",
        pattern="^(delete|export|tag)$"
    )
    template_ids: List[int] = Field(
        ...,
        description="List of template IDs to operate on"
    )
    options: Optional[Dict[str, Any]] = Field(
        None,
        description="Operation-specific options"
    )


class BulkTemplateOperationResponse(BaseModel):
    """Response schema for bulk template operations."""
    operation: str = Field(..., description="Operation performed")
    success_count: int = Field(..., description="Number of successful operations")
    failed_count: int = Field(..., description="Number of failed operations")
    errors: Optional[List[Dict[str, Any]]] = Field(
        None,
        description="Details of failed operations"
    )