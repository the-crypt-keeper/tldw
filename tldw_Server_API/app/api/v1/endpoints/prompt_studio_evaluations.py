# prompt_studio_evaluations.py
# Evaluation endpoints for Prompt Studio

from fastapi import APIRouter, Depends, HTTPException, Query, BackgroundTasks
from typing import List, Optional, Dict, Any
import uuid
import json
from datetime import datetime
from loguru import logger

from ....core.DB_Management.PromptStudioDatabase import PromptStudioDatabase
from ....core.Prompt_Management.prompt_studio.test_runner import TestRunner
from ....core.Prompt_Management.prompt_studio.evaluation_metrics import EvaluationMetrics
from ....core.Prompt_Management.prompt_studio.evaluation_manager import EvaluationManager
from ..API_Deps.prompt_studio_deps import get_prompt_studio_db, get_prompt_studio_user
from ..schemas.prompt_studio_schemas import (
    EvaluationCreate,
    EvaluationResponse,
    EvaluationList,
    EvaluationUpdate,
    EvaluationMetrics
)

router = APIRouter(prefix="/api/v1/prompt-studio")

@router.post("/evaluations", response_model=EvaluationResponse)
async def create_evaluation(
    evaluation: EvaluationCreate,
    background_tasks: BackgroundTasks,
    db: PromptStudioDatabase = Depends(get_prompt_studio_db),
    user_context: Dict = Depends(get_prompt_studio_user)
) -> EvaluationResponse:
    """
    Create a new evaluation for a prompt.
    
    Args:
        evaluation: Evaluation configuration
        background_tasks: FastAPI background tasks
        db: Database instance
        user_context: Current user context
        
    Returns:
        Created evaluation response
    """
    try:
        # Use EvaluationManager to handle the evaluation
        eval_manager = EvaluationManager(db)
        
        # Get model and parameters from config
        model = evaluation.config.model if evaluation.config else "gpt-3.5-turbo"
        temperature = evaluation.config.temperature if evaluation.config else 0.7
        max_tokens = evaluation.config.max_tokens if evaluation.config else 1000
        
        # Run evaluation (can be made async with background tasks)
        if getattr(evaluation, 'run_async', False):
            # Queue for background processing
            background_tasks.add_task(
                eval_manager.run_evaluation,
                prompt_id=evaluation.prompt_id,
                test_case_ids=evaluation.test_case_ids or [],
                model=model,
                temperature=temperature,
                max_tokens=max_tokens
            )
            
            # Create pending evaluation record
            eval_uuid = str(uuid.uuid4())
            conn = db.get_connection()
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO prompt_studio_evaluations (
                    uuid, prompt_id, model, status, test_case_ids, client_id
                ) VALUES (?, ?, ?, 'pending', ?, ?)
            """, (
                eval_uuid,
                evaluation.prompt_id,
                model,
                json.dumps(evaluation.test_case_ids or []),
                user_context.get("client_id", "api")
            ))
            eval_id = cursor.lastrowid
            conn.commit()
            
            return EvaluationResponse(
                id=eval_id,
                uuid=eval_uuid,
                project_id=evaluation.project_id,
                prompt_id=evaluation.prompt_id,
                name=evaluation.name or "Evaluation",
                description=evaluation.description or "",
                status="pending",
                created_at=datetime.now().isoformat()
            )
        else:
            # Run synchronously and return results
            result = eval_manager.run_evaluation(
                prompt_id=evaluation.prompt_id,
                test_case_ids=evaluation.test_case_ids or [],
                model=model,
                temperature=temperature,
                max_tokens=max_tokens
            )
            
            return EvaluationResponse(
                id=result["id"],
                uuid=result["uuid"],
                project_id=evaluation.project_id,
                prompt_id=result["prompt_id"],
                name=evaluation.name or "Evaluation",
                description=evaluation.description or "",
                status=result["status"],
                created_at=datetime.now().isoformat(),
                metrics=result.get("metrics")
            )
        
    except Exception as e:
        logger.error(f"Failed to create evaluation: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/evaluations", response_model=List[EvaluationResponse])
async def list_evaluations(
    project_id: int = Query(..., description="Project ID"),
    prompt_id: Optional[int] = Query(None, description="Filter by prompt ID"),
    limit: int = Query(100, ge=1, le=1000),
    offset: int = Query(0, ge=0),
    db: PromptStudioDatabase = Depends(get_prompt_studio_db),
    user_context: Dict = Depends(get_prompt_studio_user)
) -> List[EvaluationResponse]:
    """
    List evaluations for a project.
    
    Args:
        project_id: Project ID
        prompt_id: Optional prompt ID filter
        limit: Maximum results
        offset: Pagination offset
        db: Database instance
        user_context: Current user context
        
    Returns:
        List of evaluations
    """
    try:
        # Use EvaluationManager to list evaluations
        eval_manager = EvaluationManager(db)
        
        # Calculate page from offset
        page = (offset // limit) + 1 if limit > 0 else 1
        
        result = eval_manager.list_evaluations(
            project_id=project_id,
            prompt_id=prompt_id,
            page=page,
            per_page=limit
        )
        
        # Convert to response format
        evaluations = []
        for eval_data in result.get("evaluations", []):
            evaluations.append(EvaluationResponse(
                id=eval_data["id"],
                uuid=eval_data.get("uuid", ""),
                project_id=project_id,  # Add project_id since it might not be in the result
                prompt_id=eval_data["prompt_id"],
                name=eval_data.get("prompt_name", "Evaluation"),
                description="",  # Not returned by manager
                status=eval_data.get("status", "pending"),
                created_at=eval_data.get("created_at", ""),
                completed_at=eval_data.get("completed_at"),
                metrics=eval_data.get("aggregate_metrics")
            ))
        
        return evaluations
        
    except Exception as e:
        logger.error(f"Failed to list evaluations: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/evaluations/{evaluation_id}", response_model=EvaluationResponse)
async def get_evaluation(
    evaluation_id: int,
    db: PromptStudioDatabase = Depends(get_prompt_studio_db),
    user_context: Dict = Depends(get_prompt_studio_user)
) -> EvaluationResponse:
    """
    Get a specific evaluation.
    
    Args:
        evaluation_id: Evaluation ID
        db: Database instance
        user_context: Current user context
        
    Returns:
        Evaluation details
    """
    try:
        conn = db.get_connection()
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT id, uuid, project_id, prompt_id, name, description,
                   status, created_at, completed_at, aggregate_metrics
            FROM prompt_studio_evaluations
            WHERE id = ?
        """, (evaluation_id,))
        
        row = cursor.fetchone()
        if not row:
            raise HTTPException(status_code=404, detail="Evaluation not found")
        
        eval_dict = db.row_to_dict(row, cursor)
        
        return EvaluationResponse(
            id=eval_dict["id"],
            uuid=eval_dict["uuid"],
            project_id=eval_dict["project_id"],
            prompt_id=eval_dict["prompt_id"],
            name=eval_dict.get("name", ""),
            description=eval_dict.get("description", ""),
            status=eval_dict.get("status", "pending"),
            created_at=eval_dict.get("created_at", ""),
            completed_at=eval_dict.get("completed_at"),
            metrics=json.loads(eval_dict["aggregate_metrics"]) if eval_dict.get("aggregate_metrics") else {}
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get evaluation: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.delete("/evaluations/{evaluation_id}")
async def delete_evaluation(
    evaluation_id: int,
    db: PromptStudioDatabase = Depends(get_prompt_studio_db),
    user_context: Dict = Depends(get_prompt_studio_user)
) -> Dict[str, str]:
    """
    Delete an evaluation (soft delete).
    
    Args:
        evaluation_id: Evaluation ID
        db: Database instance
        user_context: Current user context
        
    Returns:
        Success message
    """
    try:
        conn = db.get_connection()
        cursor = conn.cursor()
        
        # Soft delete
        cursor.execute("""
            UPDATE prompt_studio_evaluations
            SET deleted = 1, deleted_at = ?
            WHERE id = ?
        """, (datetime.now().isoformat(), evaluation_id))
        
        if cursor.rowcount == 0:
            raise HTTPException(status_code=404, detail="Evaluation not found")
        
        conn.commit()
        
        return {"message": f"Evaluation {evaluation_id} deleted successfully"}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to delete evaluation: {e}")
        raise HTTPException(status_code=500, detail=str(e))

async def run_evaluation_async(evaluation_id: int, db: PromptStudioDatabase):
    """
    Run evaluation asynchronously.
    
    Args:
        evaluation_id: Evaluation ID
        db: Database instance
    """
    try:
        # This is a placeholder for async evaluation logic
        # In a real implementation, this would run tests and calculate metrics
        logger.info(f"Running async evaluation {evaluation_id}")
        
        # Update status to completed
        conn = db.get_connection()
        cursor = conn.cursor()
        cursor.execute("""
            UPDATE prompt_studio_evaluations
            SET status = 'completed', completed_at = ?
            WHERE id = ?
        """, (datetime.now().isoformat(), evaluation_id))
        conn.commit()
        
    except Exception as e:
        logger.error(f"Failed to run async evaluation: {e}")
        # Update status to failed
        try:
            conn = db.get_connection()
            cursor = conn.cursor()
            cursor.execute("""
                UPDATE prompt_studio_evaluations
                SET status = 'failed', error_message = ?
                WHERE id = ?
            """, (str(e), evaluation_id))
            conn.commit()
        except:
            pass