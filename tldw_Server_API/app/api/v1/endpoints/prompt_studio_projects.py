# prompt_studio_projects.py
# API endpoints for Prompt Studio project management

from typing import List, Optional, Dict, Any
from fastapi import APIRouter, Depends, HTTPException, status, Query, Path
from loguru import logger

# Local imports
from tldw_Server_API.app.api.v1.schemas.prompt_studio_base import (
    StandardResponse, ListResponse, ListQueryParams
)
from tldw_Server_API.app.api.v1.schemas.prompt_studio_project import (
    ProjectCreate, ProjectUpdate, ProjectResponse, ProjectListItem
)
from tldw_Server_API.app.api.v1.API_Deps.prompt_studio_deps import (
    get_prompt_studio_db, get_prompt_studio_user, require_project_access,
    require_project_write_access, check_rate_limit, get_security_config,
    PromptStudioDatabase
)
from tldw_Server_API.app.core.DB_Management.PromptStudioDatabase import (
    DatabaseError, InputError, ConflictError
)

########################################################################################################################
# Router Setup

router = APIRouter(
    prefix="/api/v1/prompt_studio/projects",
    tags=["Prompt Studio - Projects"],
    responses={
        401: {"description": "Unauthorized"},
        403: {"description": "Forbidden"},
        404: {"description": "Not found"},
        429: {"description": "Rate limit exceeded"}
    }
)

########################################################################################################################
# Project CRUD Endpoints

@router.post("/create", response_model=StandardResponse, status_code=status.HTTP_201_CREATED)
async def create_project(
    project_data: ProjectCreate,
    user_context: Dict = Depends(get_prompt_studio_user),
    db: PromptStudioDatabase = Depends(get_prompt_studio_db),
    _: bool = Depends(lambda: check_rate_limit("create_project"))
) -> StandardResponse:
    """
    Create a new Prompt Studio project.
    
    Args:
        project_data: Project creation data
        user_context: Current user context
        db: Database instance
        
    Returns:
        Created project details
    """
    try:
        # Create project
        project = db.create_project(
            name=project_data.name,
            description=project_data.description,
            status=project_data.status.value,
            metadata=project_data.metadata
        )
        
        logger.info(f"User {user_context['user_id']} created project: {project['name']}")
        
        return StandardResponse(
            success=True,
            data=ProjectResponse(**project)
        )
        
    except ConflictError as e:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=str(e)
        )
    except DatabaseError as e:
        logger.error(f"Database error creating project: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to create project"
        )

@router.get("/list", response_model=ListResponse)
async def list_projects(
    page: int = Query(1, ge=1, description="Page number"),
    per_page: int = Query(20, ge=1, le=100, description="Items per page"),
    status: Optional[str] = Query(None, description="Filter by status"),
    include_deleted: bool = Query(False, description="Include deleted projects"),
    search: Optional[str] = Query(None, description="Search in name and description"),
    user_context: Dict = Depends(get_prompt_studio_user),
    db: PromptStudioDatabase = Depends(get_prompt_studio_db)
) -> ListResponse:
    """
    List projects for the current user.
    
    Args:
        page: Page number
        per_page: Items per page
        status: Filter by project status
        include_deleted: Include soft-deleted projects
        search: Search query
        user_context: Current user context
        db: Database instance
        
    Returns:
        Paginated list of projects
    """
    try:
        # Get projects for user
        result = db.list_projects(
            user_id=user_context["user_id"] if not user_context["is_admin"] else None,
            status=status,
            include_deleted=include_deleted,
            page=page,
            per_page=per_page
        )
        
        # Convert to response models
        projects = [ProjectListItem(**p) for p in result["projects"]]
        
        return ListResponse(
            success=True,
            data=projects,
            metadata=result["pagination"]
        )
        
    except DatabaseError as e:
        logger.error(f"Database error listing projects: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to list projects"
        )

@router.get("/get/{project_id}", response_model=StandardResponse)
async def get_project(
    project_id: int = Path(..., description="Project ID"),
    _: bool = Depends(require_project_access),
    db: PromptStudioDatabase = Depends(get_prompt_studio_db)
) -> StandardResponse:
    """
    Get a specific project by ID.
    
    Args:
        project_id: Project ID
        db: Database instance
        
    Returns:
        Project details
    """
    try:
        project = db.get_project(project_id)
        
        if not project:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Project {project_id} not found"
            )
        
        return StandardResponse(
            success=True,
            data=ProjectResponse(**project)
        )
        
    except DatabaseError as e:
        logger.error(f"Database error getting project: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to get project"
        )

@router.put("/update/{project_id}", response_model=StandardResponse)
async def update_project(
    project_id: int = Path(..., description="Project ID"),
    updates: ProjectUpdate = ...,
    _: bool = Depends(require_project_write_access),
    db: PromptStudioDatabase = Depends(get_prompt_studio_db),
    user_context: Dict = Depends(get_prompt_studio_user)
) -> StandardResponse:
    """
    Update a project.
    
    Args:
        project_id: Project ID
        updates: Fields to update
        db: Database instance
        user_context: Current user context
        
    Returns:
        Updated project details
    """
    try:
        # Filter out None values
        update_data = {k: v for k, v in updates.dict().items() if v is not None}
        
        if not update_data:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="No fields to update"
            )
        
        # Convert enum to string if present
        if "status" in update_data and hasattr(update_data["status"], "value"):
            update_data["status"] = update_data["status"].value
        
        # Update project
        project = db.update_project(project_id, update_data)
        
        logger.info(f"User {user_context['user_id']} updated project {project_id}")
        
        return StandardResponse(
            success=True,
            data=ProjectResponse(**project)
        )
        
    except InputError as e:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(e)
        )
    except DatabaseError as e:
        logger.error(f"Database error updating project: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to update project"
        )

@router.delete("/delete/{project_id}", response_model=StandardResponse)
async def delete_project(
    project_id: int = Path(..., description="Project ID"),
    permanent: bool = Query(False, description="Permanently delete (cannot be undone)"),
    _: bool = Depends(require_project_write_access),
    db: PromptStudioDatabase = Depends(get_prompt_studio_db),
    user_context: Dict = Depends(get_prompt_studio_user)
) -> StandardResponse:
    """
    Delete a project (soft delete by default).
    
    Args:
        project_id: Project ID
        permanent: If True, permanently delete the project
        db: Database instance
        user_context: Current user context
        
    Returns:
        Success response
    """
    try:
        success = db.delete_project(project_id, hard_delete=permanent)
        
        if not success:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Project {project_id} not found or already deleted"
            )
        
        logger.info(
            f"User {user_context['user_id']} {'permanently' if permanent else 'soft'} "
            f"deleted project {project_id}"
        )
        
        return StandardResponse(
            success=True,
            data={"message": f"Project {'permanently' if permanent else 'soft'} deleted"}
        )
        
    except DatabaseError as e:
        logger.error(f"Database error deleting project: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to delete project"
        )

@router.post("/archive/{project_id}", response_model=StandardResponse)
async def archive_project(
    project_id: int = Path(..., description="Project ID"),
    _: bool = Depends(require_project_write_access),
    db: PromptStudioDatabase = Depends(get_prompt_studio_db),
    user_context: Dict = Depends(get_prompt_studio_user)
) -> StandardResponse:
    """
    Archive a project (set status to archived).
    
    Args:
        project_id: Project ID
        db: Database instance
        user_context: Current user context
        
    Returns:
        Updated project details
    """
    try:
        # Update status to archived
        project = db.update_project(project_id, {"status": "archived"})
        
        logger.info(f"User {user_context['user_id']} archived project {project_id}")
        
        return StandardResponse(
            success=True,
            data=ProjectResponse(**project)
        )
        
    except InputError as e:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(e)
        )
    except DatabaseError as e:
        logger.error(f"Database error archiving project: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to archive project"
        )

@router.post("/unarchive/{project_id}", response_model=StandardResponse)
async def unarchive_project(
    project_id: int = Path(..., description="Project ID"),
    _: bool = Depends(require_project_write_access),
    db: PromptStudioDatabase = Depends(get_prompt_studio_db),
    user_context: Dict = Depends(get_prompt_studio_user)
) -> StandardResponse:
    """
    Unarchive a project (set status to active).
    
    Args:
        project_id: Project ID
        db: Database instance
        user_context: Current user context
        
    Returns:
        Updated project details
    """
    try:
        # Update status to active
        project = db.update_project(project_id, {"status": "active"})
        
        logger.info(f"User {user_context['user_id']} unarchived project {project_id}")
        
        return StandardResponse(
            success=True,
            data=ProjectResponse(**project)
        )
        
    except InputError as e:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(e)
        )
    except DatabaseError as e:
        logger.error(f"Database error unarchiving project: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to unarchive project"
        )

########################################################################################################################
# Project Statistics Endpoint

@router.get("/stats/{project_id}", response_model=StandardResponse)
async def get_project_stats(
    project_id: int = Path(..., description="Project ID"),
    _: bool = Depends(require_project_access),
    db: PromptStudioDatabase = Depends(get_prompt_studio_db)
) -> StandardResponse:
    """
    Get statistics for a project.
    
    Args:
        project_id: Project ID
        db: Database instance
        
    Returns:
        Project statistics
    """
    try:
        conn = db.get_connection()
        cursor = conn.cursor()
        
        # Get various counts
        stats_queries = {
            "prompt_count": "SELECT COUNT(*) FROM prompt_studio_prompts WHERE project_id = ? AND deleted = 0",
            "signature_count": "SELECT COUNT(*) FROM prompt_studio_signatures WHERE project_id = ? AND deleted = 0",
            "test_case_count": "SELECT COUNT(*) FROM prompt_studio_test_cases WHERE project_id = ? AND deleted = 0",
            "golden_test_count": "SELECT COUNT(*) FROM prompt_studio_test_cases WHERE project_id = ? AND deleted = 0 AND is_golden = 1",
            "evaluation_count": "SELECT COUNT(*) FROM prompt_studio_evaluations WHERE project_id = ?",
            "optimization_count": "SELECT COUNT(*) FROM prompt_studio_optimizations WHERE project_id = ?",
            "total_test_runs": "SELECT COUNT(*) FROM prompt_studio_test_runs WHERE project_id = ?",
            "total_tokens_used": "SELECT COALESCE(SUM(tokens_used), 0) FROM prompt_studio_test_runs WHERE project_id = ?",
            "total_cost": "SELECT COALESCE(SUM(cost_estimate), 0) FROM prompt_studio_test_runs WHERE project_id = ?"
        }
        
        stats = {}
        for key, query in stats_queries.items():
            cursor.execute(query, (project_id,))
            stats[key] = cursor.fetchone()[0]
        
        return StandardResponse(
            success=True,
            data=stats
        )
        
    except DatabaseError as e:
        logger.error(f"Database error getting project stats: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to get project statistics"
        )