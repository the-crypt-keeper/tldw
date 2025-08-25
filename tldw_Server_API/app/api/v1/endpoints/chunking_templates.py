# chunking_templates.py
"""
API endpoints for managing chunking templates.
"""

import json
from typing import List, Optional, Dict, Any
from fastapi import APIRouter, Depends, HTTPException, Query, Body
from loguru import logger

from tldw_Server_API.app.api.v1.schemas.chunking_templates_schemas import (
    ChunkingTemplateCreate,
    ChunkingTemplateUpdate,
    ChunkingTemplateResponse,
    ChunkingTemplateListResponse,
    ChunkingTemplateFilter,
    ApplyTemplateRequest,
    ApplyTemplateResponse,
    TemplateValidationResponse,
    TemplateValidationError
)
from tldw_Server_API.app.core.DB_Management.Media_DB_v2 import MediaDatabase
from tldw_Server_API.app.core.Chunking.templates import TemplateProcessor, ChunkingTemplate, TemplateStage
from tldw_Server_API.app.core.Chunking.chunker import Chunker

router = APIRouter(prefix="/chunking/templates", tags=["Chunking Templates"])


def get_database() -> MediaDatabase:
    """Get database instance."""
    # TODO: Update this to use proper dependency injection with config
    import os
    db_path = os.environ.get('TLDW_DB_PATH', 'Databases/Media_DB_v2.db')
    return MediaDatabase(
        db_path=db_path,
        client_id='api_client'
    )


@router.get("", response_model=ChunkingTemplateListResponse)
async def list_templates(
    include_builtin: bool = Query(True, description="Include built-in templates"),
    include_custom: bool = Query(True, description="Include custom templates"),
    tags: Optional[List[str]] = Query(None, description="Filter by tags"),
    user_id: Optional[str] = Query(None, description="Filter by user ID"),
    db: MediaDatabase = Depends(get_database)
) -> ChunkingTemplateListResponse:
    """
    List all available chunking templates with optional filtering.
    
    Returns:
        List of chunking templates matching the filter criteria
    """
    try:
        templates = db.list_chunking_templates(
            include_builtin=include_builtin,
            include_custom=include_custom,
            tags=tags,
            user_id=user_id,
            include_deleted=False
        )
        
        # Convert to response format
        template_responses = []
        for template in templates:
            template_responses.append(ChunkingTemplateResponse(
                id=template['id'],
                uuid=template['uuid'],
                name=template['name'],
                description=template['description'],
                template_json=template['template_json'],
                is_builtin=template['is_builtin'],
                tags=template['tags'],
                created_at=template['created_at'],
                updated_at=template['updated_at'],
                version=template['version'],
                user_id=template['user_id']
            ))
        
        return ChunkingTemplateListResponse(
            templates=template_responses,
            total=len(template_responses)
        )
        
    except Exception as e:
        logger.error(f"Error listing templates: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/{template_name}", response_model=ChunkingTemplateResponse)
async def get_template(
    template_name: str,
    db: MediaDatabase = Depends(get_database)
) -> ChunkingTemplateResponse:
    """
    Get a specific chunking template by name.
    
    Args:
        template_name: Name of the template to retrieve
        
    Returns:
        The requested chunking template
        
    Raises:
        404: Template not found
    """
    try:
        template = db.get_chunking_template(name=template_name)
        
        if not template:
            raise HTTPException(status_code=404, detail=f"Template '{template_name}' not found")
        
        return ChunkingTemplateResponse(
            id=template['id'],
            uuid=template['uuid'],
            name=template['name'],
            description=template['description'],
            template_json=template['template_json'],
            is_builtin=template['is_builtin'],
            tags=template['tags'],
            created_at=template['created_at'],
            updated_at=template['updated_at'],
            version=template['version'],
            user_id=template['user_id']
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting template: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("", response_model=ChunkingTemplateResponse, status_code=201)
async def create_template(
    template_data: ChunkingTemplateCreate,
    db: MediaDatabase = Depends(get_database)
) -> ChunkingTemplateResponse:
    """
    Create a new chunking template.
    
    Args:
        template_data: Template configuration and metadata
        
    Returns:
        The created chunking template
        
    Raises:
        400: Invalid template configuration
        409: Template with same name already exists
    """
    try:
        # Convert template config to JSON string
        template_json = json.dumps(template_data.template.dict())
        
        # Create template in database
        created = db.create_chunking_template(
            name=template_data.name,
            template_json=template_json,
            description=template_data.description,
            is_builtin=False,
            tags=template_data.tags,
            user_id=template_data.user_id
        )
        
        return ChunkingTemplateResponse(
            id=created['id'],
            uuid=created['uuid'],
            name=created['name'],
            description=created['description'],
            template_json=created['template_json'],
            is_builtin=created['is_builtin'],
            tags=created['tags'],
            created_at=created.get('created_at', ''),
            updated_at=created.get('updated_at', ''),
            version=created['version'],
            user_id=created['user_id']
        )
        
    except Exception as e:
        if "already exists" in str(e):
            raise HTTPException(status_code=409, detail=str(e))
        elif "Invalid template JSON" in str(e):
            raise HTTPException(status_code=400, detail=str(e))
        else:
            logger.error(f"Error creating template: {e}")
            raise HTTPException(status_code=500, detail=str(e))


@router.put("/{template_name}", response_model=ChunkingTemplateResponse)
async def update_template(
    template_name: str,
    template_update: ChunkingTemplateUpdate,
    db: MediaDatabase = Depends(get_database)
) -> ChunkingTemplateResponse:
    """
    Update an existing chunking template.
    
    Args:
        template_name: Name of the template to update
        template_update: Fields to update
        
    Returns:
        The updated chunking template
        
    Raises:
        400: Cannot modify built-in templates
        404: Template not found
    """
    try:
        # Get existing template
        existing = db.get_chunking_template(name=template_name)
        if not existing:
            raise HTTPException(status_code=404, detail=f"Template '{template_name}' not found")
        
        # Check if built-in
        if existing['is_builtin']:
            raise HTTPException(status_code=400, detail="Cannot modify built-in templates")
        
        # Prepare update data
        template_json = None
        if template_update.template:
            template_json = json.dumps(template_update.template.dict())
        
        # Update template
        success = db.update_chunking_template(
            name=template_name,
            template_json=template_json,
            description=template_update.description,
            tags=template_update.tags
        )
        
        if not success:
            raise HTTPException(status_code=500, detail="Failed to update template")
        
        # Get updated template
        updated = db.get_chunking_template(name=template_name)
        
        return ChunkingTemplateResponse(
            id=updated['id'],
            uuid=updated['uuid'],
            name=updated['name'],
            description=updated['description'],
            template_json=updated['template_json'],
            is_builtin=updated['is_builtin'],
            tags=updated['tags'],
            created_at=updated['created_at'],
            updated_at=updated['updated_at'],
            version=updated['version'],
            user_id=updated['user_id']
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error updating template: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/{template_name}", status_code=204)
async def delete_template(
    template_name: str,
    hard_delete: bool = Query(False, description="Permanently delete template"),
    db: MediaDatabase = Depends(get_database)
) -> None:
    """
    Delete a chunking template.
    
    Args:
        template_name: Name of the template to delete
        hard_delete: If true, permanently delete; otherwise soft delete
        
    Raises:
        400: Cannot delete built-in templates
        404: Template not found
    """
    try:
        # Get existing template
        existing = db.get_chunking_template(name=template_name)
        if not existing:
            raise HTTPException(status_code=404, detail=f"Template '{template_name}' not found")
        
        # Check if built-in
        if existing['is_builtin']:
            raise HTTPException(status_code=400, detail="Cannot delete built-in templates")
        
        # Delete template
        success = db.delete_chunking_template(
            name=template_name,
            hard_delete=hard_delete
        )
        
        if not success:
            raise HTTPException(status_code=500, detail="Failed to delete template")
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting template: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/apply", response_model=ApplyTemplateResponse)
async def apply_template(
    request: ApplyTemplateRequest,
    db: MediaDatabase = Depends(get_database)
) -> ApplyTemplateResponse:
    """
    Apply a chunking template to text.
    
    Args:
        request: Template name and text to chunk
        
    Returns:
        The chunked text results
        
    Raises:
        404: Template not found
        400: Template application error
    """
    try:
        # Get template from database
        template_data = db.get_chunking_template(name=request.template_name)
        if not template_data:
            raise HTTPException(status_code=404, detail=f"Template '{request.template_name}' not found")
        
        # Parse template JSON
        template_config = json.loads(template_data['template_json'])
        
        # Create ChunkingTemplate object
        stages = []
        
        # Add preprocessing stage if exists
        if 'preprocessing' in template_config:
            stages.append(TemplateStage(
                name='preprocess',
                operations=template_config['preprocessing'],
                enabled=True
            ))
        
        # Add chunking stage
        stages.append(TemplateStage(
            name='chunk',
            operations=[template_config['chunking']],
            enabled=True
        ))
        
        # Add postprocessing stage if exists
        if 'postprocessing' in template_config:
            stages.append(TemplateStage(
                name='postprocess',
                operations=template_config['postprocessing'],
                enabled=True
            ))
        
        template = ChunkingTemplate(
            name=template_data['name'],
            description=template_data['description'] or "",
            base_method=template_config['chunking']['method'],
            stages=stages,
            default_options=template_config['chunking'].get('config', {}),
            metadata={'tags': template_data['tags']}
        )
        
        # Apply template using TemplateProcessor
        processor = TemplateProcessor()
        
        # Override options if provided
        options = {}
        if request.override_options:
            options.update(request.override_options)
        
        chunks = processor.process_template(
            text=request.text,
            template=template,
            **options
        )
        
        return ApplyTemplateResponse(
            template_name=request.template_name,
            chunks=chunks,
            metadata={
                'chunk_count': len(chunks),
                'template_version': template_data['version']
            }
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error applying template: {e}")
        raise HTTPException(status_code=400, detail=f"Template application error: {str(e)}")


@router.post("/validate", response_model=TemplateValidationResponse)
async def validate_template(
    template_config: Dict[str, Any] = Body(..., description="Template configuration to validate")
) -> TemplateValidationResponse:
    """
    Validate a template configuration without saving it.
    
    Args:
        template_config: Template configuration to validate
        
    Returns:
        Validation results with any errors or warnings
    """
    errors = []
    warnings = []
    
    try:
        # Check required fields
        if 'chunking' not in template_config:
            errors.append(TemplateValidationError(
                field='chunking',
                message='Chunking configuration is required'
            ))
        else:
            chunking = template_config['chunking']
            if 'method' not in chunking:
                errors.append(TemplateValidationError(
                    field='chunking.method',
                    message='Chunking method is required'
                ))
            else:
                # Validate chunking method exists
                valid_methods = [
                    'words', 'sentences', 'paragraphs', 'tokens',
                    'semantic', 'regex', 'markdown', 'code', 'custom'
                ]
                if chunking['method'] not in valid_methods:
                    warnings.append(
                        f"Unknown chunking method '{chunking['method']}'. Valid methods: {', '.join(valid_methods)}"
                    )
        
        # Validate preprocessing operations
        if 'preprocessing' in template_config:
            if not isinstance(template_config['preprocessing'], list):
                errors.append(TemplateValidationError(
                    field='preprocessing',
                    message='Preprocessing must be a list of operations'
                ))
            else:
                for i, op in enumerate(template_config['preprocessing']):
                    if not isinstance(op, dict) or 'operation' not in op:
                        errors.append(TemplateValidationError(
                            field=f'preprocessing[{i}]',
                            message='Each preprocessing operation must have an "operation" field'
                        ))
        
        # Validate postprocessing operations
        if 'postprocessing' in template_config:
            if not isinstance(template_config['postprocessing'], list):
                errors.append(TemplateValidationError(
                    field='postprocessing',
                    message='Postprocessing must be a list of operations'
                ))
            else:
                for i, op in enumerate(template_config['postprocessing']):
                    if not isinstance(op, dict) or 'operation' not in op:
                        errors.append(TemplateValidationError(
                            field=f'postprocessing[{i}]',
                            message='Each postprocessing operation must have an "operation" field'
                        ))
        
        # Try to serialize as JSON to catch any serialization issues
        try:
            json.dumps(template_config)
        except Exception as e:
            errors.append(TemplateValidationError(
                field='template_config',
                message=f'Template configuration is not JSON serializable: {str(e)}'
            ))
        
        return TemplateValidationResponse(
            valid=len(errors) == 0,
            errors=errors if errors else None,
            warnings=warnings if warnings else None
        )
        
    except Exception as e:
        logger.error(f"Error validating template: {e}")
        return TemplateValidationResponse(
            valid=False,
            errors=[TemplateValidationError(
                field='template_config',
                message=f'Validation error: {str(e)}'
            )]
        )