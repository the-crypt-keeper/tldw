# prompt_studio_prompts.py
# API endpoints for Prompt Studio prompt management with versioning

from typing import List, Optional, Dict, Any
from fastapi import APIRouter, Depends, HTTPException, status, Query, Path
from loguru import logger

# Local imports
from tldw_Server_API.app.api.v1.schemas.prompt_studio_base import (
    StandardResponse, ListResponse
)
from tldw_Server_API.app.api.v1.schemas.prompt_studio_project import (
    PromptCreate, PromptUpdate, PromptResponse, PromptVersion,
    SignatureCreate, SignatureUpdate, SignatureResponse,
    PromptGenerateRequest, PromptImproveRequest, ExampleGenerateRequest
)
from tldw_Server_API.app.api.v1.API_Deps.prompt_studio_deps import (
    get_prompt_studio_db, get_prompt_studio_user, require_project_access,
    require_project_write_access, check_rate_limit, get_security_config,
    PromptStudioDatabase, SecurityConfig
)
from tldw_Server_API.app.core.DB_Management.PromptStudioDatabase import (
    DatabaseError, InputError, ConflictError
)

########################################################################################################################
# Router Setup

router = APIRouter(
    prefix="/api/v1/prompt_studio/prompts",
    tags=["Prompt Studio - Prompts"],
    responses={
        401: {"description": "Unauthorized"},
        403: {"description": "Forbidden"},
        404: {"description": "Not found"},
        429: {"description": "Rate limit exceeded"}
    }
)

########################################################################################################################
# Prompt CRUD Endpoints

@router.post("/create", response_model=StandardResponse, status_code=status.HTTP_201_CREATED)
async def create_prompt(
    prompt_data: PromptCreate,
    _: bool = Depends(lambda: require_project_write_access(prompt_data.project_id)),
    db: PromptStudioDatabase = Depends(get_prompt_studio_db),
    security_config: SecurityConfig = Depends(get_security_config),
    user_context: Dict = Depends(get_prompt_studio_user)
) -> StandardResponse:
    """
    Create a new prompt in a project.
    
    Args:
        prompt_data: Prompt creation data
        db: Database instance
        security_config: Security configuration
        user_context: Current user context
        
    Returns:
        Created prompt details
    """
    try:
        # Validate prompt length
        if prompt_data.system_prompt and len(prompt_data.system_prompt) > security_config.max_prompt_length:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"System prompt exceeds maximum length of {security_config.max_prompt_length}"
            )
        
        if prompt_data.user_prompt and len(prompt_data.user_prompt) > security_config.max_prompt_length:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"User prompt exceeds maximum length of {security_config.max_prompt_length}"
            )
        
        # Create prompt in database
        conn = db.get_connection()
        cursor = conn.cursor()
        
        import uuid
        import json
        
        prompt_uuid = str(uuid.uuid4())
        
        # Convert few_shot_examples and modules_config to JSON
        few_shot_json = None
        if prompt_data.few_shot_examples:
            few_shot_json = json.dumps([ex.dict() for ex in prompt_data.few_shot_examples])
        
        modules_json = None
        if prompt_data.modules_config:
            modules_json = json.dumps([mod.dict() for mod in prompt_data.modules_config])
        
        cursor.execute("""
            INSERT INTO prompt_studio_prompts (
                uuid, project_id, signature_id, version_number, name,
                system_prompt, user_prompt, few_shot_examples, modules_config,
                parent_version_id, change_description, client_id
            ) VALUES (?, ?, ?, 1, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            prompt_uuid, prompt_data.project_id, prompt_data.signature_id,
            prompt_data.name, prompt_data.system_prompt, prompt_data.user_prompt,
            few_shot_json, modules_json, prompt_data.parent_version_id,
            prompt_data.change_description, user_context["client_id"]
        ))
        
        prompt_id = cursor.lastrowid
        conn.commit()
        
        # Get created prompt
        prompt = db.get_prompt(prompt_id)
        
        logger.info(f"User {user_context['user_id']} created prompt: {prompt_data.name}")
        
        return StandardResponse(
            success=True,
            data=PromptResponse(**prompt)
        )
        
    except sqlite3.IntegrityError as e:
        if "UNIQUE" in str(e):
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail="Prompt with this name already exists in the project"
            )
        raise DatabaseError(f"Failed to create prompt: {e}")
    except DatabaseError as e:
        logger.error(f"Database error creating prompt: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to create prompt"
        )

@router.get("/list/{project_id}", response_model=ListResponse)
async def list_prompts(
    project_id: int = Path(..., description="Project ID"),
    page: int = Query(1, ge=1, description="Page number"),
    per_page: int = Query(20, ge=1, le=100, description="Items per page"),
    include_deleted: bool = Query(False, description="Include deleted prompts"),
    _: bool = Depends(lambda: require_project_access(project_id)),
    db: PromptStudioDatabase = Depends(get_prompt_studio_db)
) -> ListResponse:
    """
    List prompts in a project.
    
    Args:
        project_id: Project ID
        page: Page number
        per_page: Items per page
        include_deleted: Include soft-deleted prompts
        db: Database instance
        
    Returns:
        Paginated list of prompts
    """
    try:
        conn = db.get_connection()
        cursor = conn.cursor()
        
        # Build query
        base_query = """
            FROM prompt_studio_prompts 
            WHERE project_id = ?
        """
        
        params = [project_id]
        
        if not include_deleted:
            base_query += " AND deleted = 0"
        
        # Count total
        count_query = f"SELECT COUNT(*) {base_query}"
        cursor.execute(count_query, params)
        total = cursor.fetchone()[0]
        
        # Get prompts with pagination
        offset = (page - 1) * per_page
        query = f"""
            SELECT * {base_query}
            ORDER BY updated_at DESC, version_number DESC
            LIMIT ? OFFSET ?
        """
        params.extend([per_page, offset])
        
        cursor.execute(query, params)
        prompts = [db._row_to_dict(cursor, row) for row in cursor.fetchall()]
        
        return ListResponse(
            success=True,
            data=[PromptResponse(**p) for p in prompts],
            metadata={
                "page": page,
                "per_page": per_page,
                "total": total,
                "total_pages": (total + per_page - 1) // per_page
            }
        )
        
    except DatabaseError as e:
        logger.error(f"Database error listing prompts: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to list prompts"
        )

@router.get("/get/{prompt_id}", response_model=StandardResponse)
async def get_prompt(
    prompt_id: int = Path(..., description="Prompt ID"),
    db: PromptStudioDatabase = Depends(get_prompt_studio_db)
) -> StandardResponse:
    """
    Get a specific prompt by ID.
    
    Args:
        prompt_id: Prompt ID
        db: Database instance
        
    Returns:
        Prompt details
    """
    try:
        conn = db.get_connection()
        cursor = conn.cursor()
        
        # Get prompt with project check
        cursor.execute("""
            SELECT p.*, proj.user_id as project_user_id
            FROM prompt_studio_prompts p
            JOIN prompt_studio_projects proj ON p.project_id = proj.id
            WHERE p.id = ? AND p.deleted = 0
        """, (prompt_id,))
        
        row = cursor.fetchone()
        if not row:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Prompt {prompt_id} not found"
            )
        
        prompt = db._row_to_dict(cursor, row)
        
        # Check access
        user_context = await get_prompt_studio_user(None, None, None)
        if prompt["project_user_id"] != user_context["user_id"] and not user_context["is_admin"]:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Access denied to this prompt"
            )
        
        # Remove the extra field
        prompt.pop("project_user_id", None)
        
        return StandardResponse(
            success=True,
            data=PromptResponse(**prompt)
        )
        
    except DatabaseError as e:
        logger.error(f"Database error getting prompt: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to get prompt"
        )

@router.put("/update/{prompt_id}", response_model=StandardResponse)
async def update_prompt(
    prompt_id: int = Path(..., description="Prompt ID"),
    updates: PromptUpdate = ...,
    db: PromptStudioDatabase = Depends(get_prompt_studio_db),
    security_config: SecurityConfig = Depends(get_security_config),
    user_context: Dict = Depends(get_prompt_studio_user)
) -> StandardResponse:
    """
    Update a prompt (creates a new version).
    
    Args:
        prompt_id: Prompt ID
        updates: Fields to update
        db: Database instance
        security_config: Security configuration
        user_context: Current user context
        
    Returns:
        New prompt version details
    """
    try:
        # Validate prompt lengths if provided
        if updates.system_prompt and len(updates.system_prompt) > security_config.max_prompt_length:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"System prompt exceeds maximum length of {security_config.max_prompt_length}"
            )
        
        if updates.user_prompt and len(updates.user_prompt) > security_config.max_prompt_length:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"User prompt exceeds maximum length of {security_config.max_prompt_length}"
            )
        
        conn = db.get_connection()
        cursor = conn.cursor()
        
        # Get current prompt
        cursor.execute("""
            SELECT p.*, proj.user_id as project_user_id
            FROM prompt_studio_prompts p
            JOIN prompt_studio_projects proj ON p.project_id = proj.id
            WHERE p.id = ? AND p.deleted = 0
        """, (prompt_id,))
        
        current_prompt = db._row_to_dict(cursor, cursor.fetchone())
        
        if not current_prompt:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Prompt {prompt_id} not found"
            )
        
        # Check access
        if current_prompt["project_user_id"] != user_context["user_id"] and not user_context["is_admin"]:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Access denied to this prompt"
            )
        
        # Create new version
        import uuid
        import json
        
        new_uuid = str(uuid.uuid4())
        new_version = current_prompt["version_number"] + 1
        
        # Merge updates with current values
        new_name = updates.name or current_prompt["name"]
        new_system = updates.system_prompt if updates.system_prompt is not None else current_prompt["system_prompt"]
        new_user = updates.user_prompt if updates.user_prompt is not None else current_prompt["user_prompt"]
        
        new_examples = None
        if updates.few_shot_examples is not None:
            new_examples = json.dumps([ex.dict() for ex in updates.few_shot_examples])
        else:
            new_examples = json.dumps(current_prompt["few_shot_examples"]) if current_prompt["few_shot_examples"] else None
        
        new_modules = None
        if updates.modules_config is not None:
            new_modules = json.dumps([mod.dict() for mod in updates.modules_config])
        else:
            new_modules = json.dumps(current_prompt["modules_config"]) if current_prompt["modules_config"] else None
        
        # Insert new version
        cursor.execute("""
            INSERT INTO prompt_studio_prompts (
                uuid, project_id, signature_id, version_number, name,
                system_prompt, user_prompt, few_shot_examples, modules_config,
                parent_version_id, change_description, client_id
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            new_uuid, current_prompt["project_id"], current_prompt["signature_id"],
            new_version, new_name, new_system, new_user, new_examples, new_modules,
            prompt_id, updates.change_description, user_context["client_id"]
        ))
        
        new_prompt_id = cursor.lastrowid
        conn.commit()
        
        # Get new version
        cursor.execute("SELECT * FROM prompt_studio_prompts WHERE id = ?", (new_prompt_id,))
        new_prompt = db._row_to_dict(cursor, cursor.fetchone())
        
        logger.info(f"User {user_context['user_id']} created version {new_version} of prompt {prompt_id}")
        
        return StandardResponse(
            success=True,
            data=PromptResponse(**new_prompt)
        )
        
    except DatabaseError as e:
        logger.error(f"Database error updating prompt: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to update prompt"
        )

@router.get("/history/{prompt_id}", response_model=StandardResponse)
async def get_prompt_history(
    prompt_id: int = Path(..., description="Prompt ID"),
    db: PromptStudioDatabase = Depends(get_prompt_studio_db)
) -> StandardResponse:
    """
    Get version history for a prompt.
    
    Args:
        prompt_id: Prompt ID
        db: Database instance
        
    Returns:
        List of prompt versions
    """
    try:
        conn = db.get_connection()
        cursor = conn.cursor()
        
        # Get prompt to find its name and project
        cursor.execute("""
            SELECT name, project_id FROM prompt_studio_prompts 
            WHERE id = ? AND deleted = 0
        """, (prompt_id,))
        
        prompt_info = cursor.fetchone()
        if not prompt_info:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Prompt {prompt_id} not found"
            )
        
        name, project_id = prompt_info
        
        # Check access
        await require_project_access(project_id)
        
        # Get all versions of this prompt
        cursor.execute("""
            SELECT id, uuid, version_number, name, change_description, 
                   created_at, parent_version_id
            FROM prompt_studio_prompts
            WHERE project_id = ? AND name = ? AND deleted = 0
            ORDER BY version_number DESC
        """, (project_id, name))
        
        versions = [
            PromptVersion(**db._row_to_dict(cursor, row))
            for row in cursor.fetchall()
        ]
        
        return StandardResponse(
            success=True,
            data=versions
        )
        
    except DatabaseError as e:
        logger.error(f"Database error getting prompt history: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to get prompt history"
        )

@router.post("/revert/{prompt_id}/{version}", response_model=StandardResponse)
async def revert_prompt(
    prompt_id: int = Path(..., description="Current prompt ID"),
    version: int = Path(..., description="Version number to revert to"),
    db: PromptStudioDatabase = Depends(get_prompt_studio_db),
    user_context: Dict = Depends(get_prompt_studio_user)
) -> StandardResponse:
    """
    Revert a prompt to a previous version (creates a new version).
    
    Args:
        prompt_id: Current prompt ID
        version: Version number to revert to
        db: Database instance
        user_context: Current user context
        
    Returns:
        New prompt version details
    """
    try:
        conn = db.get_connection()
        cursor = conn.cursor()
        
        # Get current prompt info
        cursor.execute("""
            SELECT name, project_id FROM prompt_studio_prompts 
            WHERE id = ? AND deleted = 0
        """, (prompt_id,))
        
        current_info = cursor.fetchone()
        if not current_info:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Prompt {prompt_id} not found"
            )
        
        name, project_id = current_info
        
        # Check access
        await require_project_write_access(project_id)
        
        # Find the version to revert to
        cursor.execute("""
            SELECT * FROM prompt_studio_prompts
            WHERE project_id = ? AND name = ? AND version_number = ? AND deleted = 0
        """, (project_id, name, version))
        
        target_version = db._row_to_dict(cursor, cursor.fetchone())
        
        if not target_version:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Version {version} not found for this prompt"
            )
        
        # Get max version number
        cursor.execute("""
            SELECT MAX(version_number) FROM prompt_studio_prompts
            WHERE project_id = ? AND name = ?
        """, (project_id, name))
        
        max_version = cursor.fetchone()[0] or 0
        new_version = max_version + 1
        
        # Create new version from target
        import uuid
        import json
        
        new_uuid = str(uuid.uuid4())
        
        cursor.execute("""
            INSERT INTO prompt_studio_prompts (
                uuid, project_id, signature_id, version_number, name,
                system_prompt, user_prompt, few_shot_examples, modules_config,
                parent_version_id, change_description, client_id
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            new_uuid, project_id, target_version["signature_id"], new_version, name,
            target_version["system_prompt"], target_version["user_prompt"],
            json.dumps(target_version["few_shot_examples"]) if target_version["few_shot_examples"] else None,
            json.dumps(target_version["modules_config"]) if target_version["modules_config"] else None,
            prompt_id, f"Reverted to version {version}", user_context["client_id"]
        ))
        
        new_prompt_id = cursor.lastrowid
        conn.commit()
        
        # Get new version
        cursor.execute("SELECT * FROM prompt_studio_prompts WHERE id = ?", (new_prompt_id,))
        new_prompt = db._row_to_dict(cursor, cursor.fetchone())
        
        logger.info(f"User {user_context['user_id']} reverted prompt {prompt_id} to version {version}")
        
        return StandardResponse(
            success=True,
            data=PromptResponse(**new_prompt)
        )
        
    except DatabaseError as e:
        logger.error(f"Database error reverting prompt: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to revert prompt"
        )