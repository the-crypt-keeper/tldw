# prompt_studio_optimization.py
# API endpoints for prompt optimization in Prompt Studio

from typing import List, Optional, Dict, Any
from fastapi import APIRouter, Depends, HTTPException, status, Query, Path, Body, BackgroundTasks
from loguru import logger

# Local imports
from tldw_Server_API.app.api.v1.schemas.prompt_studio_base import StandardResponse, ListResponse
from tldw_Server_API.app.api.v1.schemas.prompt_studio_optimization import (
    OptimizationCreate, OptimizationResponse,
    OptimizationConfig
)
from tldw_Server_API.app.api.v1.schemas.prompt_studio_optimization_requests import (
    CompareStrategiesRequest
)
from tldw_Server_API.app.api.v1.API_Deps.prompt_studio_deps import (
    get_prompt_studio_db, get_prompt_studio_user, require_project_write_access,
    check_rate_limit, get_security_config, PromptStudioDatabase, SecurityConfig
)
from tldw_Server_API.app.core.Prompt_Management.prompt_studio.optimization_engine import OptimizationEngine
from tldw_Server_API.app.core.Prompt_Management.prompt_studio.job_manager import JobManager, JobType
from tldw_Server_API.app.core.DB_Management.PromptStudioDatabase import DatabaseError

########################################################################################################################
# Router Setup

router = APIRouter(
    prefix="/api/v1/prompt-studio/optimizations",
    tags=["Prompt Studio - Optimizations"],
    responses={
        401: {"description": "Unauthorized"},
        403: {"description": "Forbidden"},
        404: {"description": "Not found"},
        429: {"description": "Rate limit exceeded"}
    }
)

########################################################################################################################
# Optimization CRUD Endpoints

@router.post("/create", response_model=StandardResponse, status_code=status.HTTP_201_CREATED)
async def create_optimization(
    optimization_data: OptimizationCreate,
    background_tasks: BackgroundTasks,
    _: bool = Depends(lambda: check_rate_limit("optimization")),
    db: PromptStudioDatabase = Depends(get_prompt_studio_db),
    security_config: SecurityConfig = Depends(get_security_config),
    user_context: Dict = Depends(get_prompt_studio_user)
) -> StandardResponse:
    """
    Create and start a new optimization.
    
    Args:
        optimization_data: Optimization configuration
        background_tasks: Background task manager
        db: Database instance
        security_config: Security configuration
        user_context: Current user context
        
    Returns:
        Created optimization details
    """
    try:
        # Validate project access
        conn = db.get_connection()
        cursor = conn.cursor()
        
        # Get project from prompt
        cursor.execute(
            "SELECT project_id FROM prompt_studio_prompts WHERE id = ?",
            (optimization_data.initial_prompt_id,)
        )
        row = cursor.fetchone()
        if not row:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Prompt {optimization_data.initial_prompt_id} not found"
            )
        
        project_id = row[0]
        await require_project_write_access(project_id)
        
        # Create optimization record
        cursor.execute("""
            INSERT INTO prompt_studio_optimizations (
                uuid, project_id, name, initial_prompt_id,
                test_case_ids, model_config, optimizer_type,
                optimizer_config, max_iterations, status, client_id
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            f"opt-{user_context['user_id']}-{datetime.utcnow().timestamp()}",
            project_id,
            optimization_data.name,
            optimization_data.initial_prompt_id,
            json.dumps(optimization_data.test_case_ids),
            json.dumps(optimization_data.model_config.dict()),
            optimization_data.strategy.value,
            json.dumps(optimization_data.optimizer_config.dict()),
            optimization_data.max_iterations,
            "pending",
            db.client_id
        ))
        
        optimization_id = cursor.lastrowid
        conn.commit()
        
        # Create job for optimization
        job_manager = JobManager(db)
        job = job_manager.create_job(
            job_type=JobType.OPTIMIZATION,
            entity_id=optimization_id,
            payload={
                "optimization_id": optimization_id,
                "strategy": optimization_data.strategy.value
            },
            priority=optimization_data.priority
        )
        
        logger.info(f"User {user_context['user_id']} created optimization {optimization_id}")
        
        # Start optimization in background (if auto_start is true)
        if optimization_data.auto_start:
            background_tasks.add_task(
                run_optimization_async,
                optimization_id,
                db
            )
        
        return StandardResponse(
            success=True,
            data=OptimizationResponse(
                id=optimization_id,
                uuid=cursor.lastrowid,
                project_id=project_id,
                name=optimization_data.name,
                status="pending",
                job_id=job["id"]
            )
        )
        
    except DatabaseError as e:
        logger.error(f"Database error creating optimization: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to create optimization"
        )

@router.get("/list/{project_id}", response_model=ListResponse)
async def list_optimizations(
    project_id: int = Path(..., description="Project ID"),
    page: int = Query(1, ge=1, description="Page number"),
    per_page: int = Query(20, ge=1, le=100, description="Items per page"),
    status: Optional[str] = Query(None, description="Filter by status"),
    _: bool = Depends(lambda: require_project_access(project_id)),
    db: PromptStudioDatabase = Depends(get_prompt_studio_db)
) -> ListResponse:
    """
    List optimizations for a project.
    
    Args:
        project_id: Project ID
        page: Page number
        per_page: Items per page
        status: Optional status filter
        db: Database instance
        
    Returns:
        Paginated list of optimizations
    """
    try:
        conn = db.get_connection()
        cursor = conn.cursor()
        
        # Build query
        query = """
            SELECT * FROM prompt_studio_optimizations
            WHERE project_id = ? AND deleted = 0
        """
        params = [project_id]
        
        if status:
            query += " AND status = ?"
            params.append(status)
        
        query += " ORDER BY created_at DESC LIMIT ? OFFSET ?"
        params.extend([per_page, (page - 1) * per_page])
        
        cursor.execute(query, params)
        
        optimizations = []
        for row in cursor.fetchall():
            opt = db._row_to_dict(cursor, row)
            optimizations.append(OptimizationResponse(**opt))
        
        # Get total count
        count_query = """
            SELECT COUNT(*) FROM prompt_studio_optimizations
            WHERE project_id = ? AND deleted = 0
        """
        count_params = [project_id]
        
        if status:
            count_query += " AND status = ?"
            count_params.append(status)
        
        cursor.execute(count_query, count_params)
        total_count = cursor.fetchone()[0]
        
        return ListResponse(
            success=True,
            data=optimizations,
            metadata={
                "page": page,
                "per_page": per_page,
                "total": total_count,
                "total_pages": (total_count + per_page - 1) // per_page
            }
        )
        
    except Exception as e:
        logger.error(f"Error listing optimizations: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to list optimizations"
        )

@router.get("/get/{optimization_id}", response_model=StandardResponse)
async def get_optimization(
    optimization_id: int = Path(..., description="Optimization ID"),
    db: PromptStudioDatabase = Depends(get_prompt_studio_db)
) -> StandardResponse:
    """
    Get optimization details.
    
    Args:
        optimization_id: Optimization ID
        db: Database instance
        
    Returns:
        Optimization details
    """
    try:
        conn = db.get_connection()
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT * FROM prompt_studio_optimizations
            WHERE id = ? AND deleted = 0
        """, (optimization_id,))
        
        row = cursor.fetchone()
        if not row:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Optimization {optimization_id} not found"
            )
        
        optimization = db._row_to_dict(cursor, row)
        
        # Check project access
        await require_project_access(optimization["project_id"])
        
        # Parse JSON fields
        optimization["test_case_ids"] = json.loads(optimization.get("test_case_ids", "[]"))
        optimization["model_config"] = json.loads(optimization.get("model_config", "{}"))
        optimization["optimizer_config"] = json.loads(optimization.get("optimizer_config", "{}"))
        optimization["initial_metrics"] = json.loads(optimization.get("initial_metrics", "{}"))
        optimization["final_metrics"] = json.loads(optimization.get("final_metrics", "{}"))
        
        return StandardResponse(
            success=True,
            data=optimization
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting optimization: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to get optimization"
        )

@router.post("/cancel/{optimization_id}", response_model=StandardResponse)
async def cancel_optimization(
    optimization_id: int = Path(..., description="Optimization ID"),
    reason: str = Body(None, description="Cancellation reason"),
    db: PromptStudioDatabase = Depends(get_prompt_studio_db),
    user_context: Dict = Depends(get_prompt_studio_user)
) -> StandardResponse:
    """
    Cancel a running optimization.
    
    Args:
        optimization_id: Optimization ID
        reason: Optional cancellation reason
        db: Database instance
        user_context: Current user context
        
    Returns:
        Success response
    """
    try:
        # Get optimization
        conn = db.get_connection()
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT project_id, status FROM prompt_studio_optimizations
            WHERE id = ? AND deleted = 0
        """, (optimization_id,))
        
        row = cursor.fetchone()
        if not row:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Optimization {optimization_id} not found"
            )
        
        project_id, status = row
        
        # Check access
        await require_project_write_access(project_id)
        
        # Check if can cancel
        if status in ["completed", "failed", "cancelled"]:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Cannot cancel optimization with status: {status}"
            )
        
        # Cancel associated job
        job_manager = JobManager(db)
        cursor.execute("""
            SELECT id FROM prompt_studio_job_queue
            WHERE job_type = 'optimization' AND entity_id = ?
            ORDER BY created_at DESC LIMIT 1
        """, (optimization_id,))
        
        job_row = cursor.fetchone()
        if job_row:
            job_manager.cancel_job(job_row[0], reason or "User cancelled")
        
        # Update optimization status
        cursor.execute("""
            UPDATE prompt_studio_optimizations
            SET status = 'cancelled',
                error_message = ?,
                completed_at = CURRENT_TIMESTAMP
            WHERE id = ?
        """, (reason or "Cancelled by user", optimization_id))
        
        conn.commit()
        
        logger.info(f"User {user_context['user_id']} cancelled optimization {optimization_id}")
        
        return StandardResponse(
            success=True,
            data={"message": "Optimization cancelled"}
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error cancelling optimization: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to cancel optimization"
        )

########################################################################################################################
# Optimization Strategy Endpoints

@router.get("/strategies", response_model=StandardResponse)
async def get_optimization_strategies() -> StandardResponse:
    """
    Get available optimization strategies.
    
    Returns:
        List of available strategies with descriptions
    """
    strategies = [
        {
            "name": "mipro",
            "display_name": "MIPRO",
            "description": "Multi-Instruction Prompt Optimization - iteratively refines instructions",
            "parameters": {
                "target_metric": "Metric to optimize (accuracy, f1_score, etc.)",
                "min_improvement": "Minimum improvement to continue (0.01-0.1)"
            }
        },
        {
            "name": "bootstrap",
            "display_name": "Bootstrap Few-Shot",
            "description": "Automatically selects best examples for few-shot learning",
            "parameters": {
                "num_examples": "Number of examples to include (1-10)",
                "selection_strategy": "How to select examples (best, diverse, random)"
            }
        },
        {
            "name": "iterative",
            "display_name": "Iterative Refinement",
            "description": "Analyzes errors and iteratively refines the prompt",
            "parameters": {}
        },
        {
            "name": "hyperparameter",
            "display_name": "Hyperparameter Tuning",
            "description": "Optimizes model parameters like temperature and max_tokens",
            "parameters": {
                "params_to_optimize": "List of parameters to tune",
                "search_method": "Search method (bayesian, grid, random)"
            }
        },
        {
            "name": "genetic",
            "display_name": "Genetic Algorithm",
            "description": "Evolves prompts using genetic algorithm techniques",
            "parameters": {
                "population_size": "Population size (5-20)",
                "mutation_rate": "Mutation probability (0.05-0.2)"
            }
        }
    ]
    
    return StandardResponse(
        success=True,
        data=strategies
    )

@router.post("/compare", response_model=StandardResponse)
async def compare_strategies(
    request: CompareStrategiesRequest,
    background_tasks: BackgroundTasks = None,
    _: bool = Depends(lambda: check_rate_limit("optimization")),
    db: PromptStudioDatabase = Depends(get_prompt_studio_db),
    user_context: Dict = Depends(get_prompt_studio_user)
) -> StandardResponse:
    """
    Compare multiple optimization strategies.
    
    Args:
        prompt_id: Prompt to optimize
        test_case_ids: Test cases
        strategies: List of strategies to compare
        model_config: Model configuration
        background_tasks: Background task manager
        db: Database instance
        user_context: Current user context
        
    Returns:
        Comparison job details
    """
    try:
        # Validate prompt
        conn = db.get_connection()
        cursor = conn.cursor()
        
        cursor.execute(
            "SELECT project_id FROM prompt_studio_prompts WHERE id = ?",
            (prompt_id,)
        )
        row = cursor.fetchone()
        if not row:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Prompt {prompt_id} not found"
            )
        
        project_id = row[0]
        await require_project_write_access(project_id)
        
        # Create optimizations for each strategy
        optimization_ids = []
        
        for strategy in strategies:
            cursor.execute("""
                INSERT INTO prompt_studio_optimizations (
                    uuid, project_id, name, initial_prompt_id,
                    test_case_ids, model_config, optimizer_type,
                    optimizer_config, max_iterations, status, client_id
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                f"compare-{strategy}-{datetime.utcnow().timestamp()}",
                project_id,
                f"Compare: {strategy}",
                request.prompt_id,
                json.dumps(request.test_case_ids),
                json.dumps(request.model_configuration),
                strategy,
                json.dumps({"strategy": strategy}),
                10,  # Reduced iterations for comparison
                "pending",
                db.client_id
            ))
            
            optimization_ids.append(cursor.lastrowid)
        
        conn.commit()
        
        # Create jobs for each optimization
        job_manager = JobManager(db)
        jobs = []
        
        for opt_id in optimization_ids:
            job = job_manager.create_job(
                job_type=JobType.OPTIMIZATION,
                entity_id=opt_id,
                payload={"optimization_id": opt_id},
                priority=5
            )
            jobs.append(job)
        
        logger.info(f"User {user_context['user_id']} created strategy comparison")
        
        return StandardResponse(
            success=True,
            data={
                "optimization_ids": optimization_ids,
                "job_ids": [j["id"] for j in jobs],
                "strategies": strategies,
                "message": f"Comparing {len(strategies)} optimization strategies"
            }
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error comparing strategies: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to compare strategies"
        )

########################################################################################################################
# Helper Functions

import json
from datetime import datetime

async def run_optimization_async(optimization_id: int, db: PromptStudioDatabase):
    """
    Run optimization asynchronously.
    
    Args:
        optimization_id: Optimization ID
        db: Database instance
    """
    try:
        engine = OptimizationEngine(db)
        await engine.optimize(optimization_id)
    except Exception as e:
        logger.error(f"Async optimization failed: {e}")
        
        # Update status to failed
        conn = db.get_connection()
        cursor = conn.cursor()
        cursor.execute("""
            UPDATE prompt_studio_optimizations
            SET status = 'failed',
                error_message = ?,
                completed_at = CURRENT_TIMESTAMP
            WHERE id = ?
        """, (str(e), optimization_id))
        conn.commit()

async def require_project_access(project_id: int) -> bool:
    """Check if user has access to project."""
    # Placeholder - implement actual access control
    return True