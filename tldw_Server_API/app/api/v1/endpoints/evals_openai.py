# evals_openai.py - OpenAI-compatible evaluation API endpoints
"""
OpenAI-compatible evaluation API implementation.

Provides endpoints for:
- Evaluation CRUD operations
- Evaluation runs with async processing
- Dataset management
- Progress tracking and results retrieval
"""

import os
from datetime import datetime
from typing import List, Optional, Dict, Any, AsyncGenerator
from fastapi import APIRouter, HTTPException, Depends, BackgroundTasks, status, Query, Security, Request
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.responses import StreamingResponse
from slowapi import Limiter
from slowapi.util import get_remote_address
from loguru import logger
import asyncio
import json

from tldw_Server_API.app.api.v1.schemas.openai_eval_schemas import (
    CreateEvaluationRequest, UpdateEvaluationRequest, EvaluationResponse,
    CreateRunRequest, RunResponse, RunResultsResponse,
    CreateDatasetRequest, DatasetResponse,
    EvaluationListResponse, RunListResponse, DatasetListResponse,
    ListQueryParams, RunListQueryParams,
    ErrorResponse, ErrorDetail
)
from tldw_Server_API.app.core.DB_Management.Evaluations_DB import EvaluationsDatabase
from tldw_Server_API.app.core.Evaluations.eval_runner import EvaluationRunner
from tldw_Server_API.app.core.config import load_comprehensive_config

# Create router
router = APIRouter(tags=["Evaluations"])

# Security
security = HTTPBearer(auto_error=False)

# Rate limiting
limiter = Limiter(key_func=get_remote_address)

# Rate limit decorators for different operations
create_limit = limiter.limit("10/minute")  # Create operations
read_limit = limiter.limit("100/minute")   # Read operations
run_limit = limiter.limit("5/minute")      # Run operations (resource intensive)

# Initialize database and runner
# Get database path from config or use default
config = load_comprehensive_config()
# Config is a ConfigParser object, not a dict
if config and config.has_section("Database"):
    db_path = config.get("Database", "evaluations_db_path", fallback="Databases/evaluations.db")
else:
    db_path = "Databases/evaluations.db"
    
if not os.path.isabs(db_path):
    db_path = os.path.join(os.path.dirname(__file__), "..", "..", "..", "..", db_path)

os.makedirs(os.path.dirname(db_path), exist_ok=True)

eval_db = EvaluationsDatabase(db_path)
eval_runner = EvaluationRunner(db_path)


# ============= Authentication =============

async def verify_api_key(credentials: Optional[HTTPAuthorizationCredentials] = Security(security)) -> str:
    """Verify API key in OpenAI format"""
    # Default API key for testing/development
    DEFAULT_API_KEY = "default-secret-key-for-single-user"
    
    if not credentials:
        # Check if we're in development mode with default key
        api_bearer = os.getenv("API_BEARER", DEFAULT_API_KEY)
        if api_bearer == DEFAULT_API_KEY:
            logger.warning("No credentials provided, using default API key for development")
            return DEFAULT_API_KEY
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail={"error": {
                "message": "Missing API key",
                "type": "authentication_error",
                "code": "missing_api_key"
            }}
        )
    
    token = credentials.credentials
    
    # Accept both Bearer tokens and sk- prefixed keys for OpenAI compatibility
    if token.startswith("Bearer "):
        token = token[7:]  # Remove "Bearer " prefix
    
    # Check against configured API key or default
    expected_token = os.getenv("API_BEARER", DEFAULT_API_KEY)
    
    # In development, accept the default key
    if token == DEFAULT_API_KEY or token == expected_token:
        return token
    
    # For OpenAI compatibility, also accept sk- prefixed keys
    if token.startswith("sk-") and len(token) > 3:
        # In production, validate against stored keys
        # For now, accept any sk- key in development
        logger.info(f"Accepted OpenAI-style key: sk-{'*' * 8}")
        return token
    
    raise HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail={"error": {
            "message": "Invalid API key",
            "type": "authentication_error",
            "code": "invalid_api_key"
        }}
    )
    
    return token


# ============= Error Handling =============

def create_error_response(
    message: str,
    error_type: str = "invalid_request_error",
    param: Optional[str] = None,
    code: Optional[str] = None,
    status_code: int = status.HTTP_400_BAD_REQUEST
) -> HTTPException:
    """Create standardized error response"""
    return HTTPException(
        status_code=status_code,
        detail={
            "error": {
                "message": message,
                "type": error_type,
                "param": param,
                "code": code
            }
        }
    )


# ============= Evaluation Endpoints =============

@router.post("/v1/evals", response_model=EvaluationResponse, status_code=status.HTTP_201_CREATED)
@create_limit
async def create_evaluation(
    eval_request: CreateEvaluationRequest,
    request: Request,
    api_key: str = Depends(verify_api_key)
):
    """
    Create a new evaluation definition.
    
    This endpoint creates an evaluation that can be run multiple times with different models.
    
    **Rate Limit:** 10 requests per minute
    
    **Request Body:**
    - `name`: Unique name for the evaluation
    - `description`: Optional description
    - `eval_type`: Type of evaluation (model_graded, exact_match, includes, etc.)
    - `eval_spec`: Evaluation specification including metrics and thresholds
    - `dataset`: Optional inline dataset (will create a new dataset)
    - `dataset_id`: Optional reference to existing dataset
    - `metadata`: Optional metadata dictionary
    
    **Returns:** Created evaluation object with generated ID
    """
    try:
        # If inline dataset provided, create it first
        dataset_id = eval_request.dataset_id
        if eval_request.dataset and not dataset_id:
            dataset_id = eval_db.create_dataset(
                name=f"{eval_request.name}_dataset",
                samples=eval_request.dataset,
                description=f"Dataset for {eval_request.name}",
                created_by=api_key
            )
        
        # Create evaluation
        eval_id = eval_db.create_evaluation(
            name=eval_request.name,
            description=eval_request.description,
            eval_type=eval_request.eval_type,
            eval_spec=eval_request.eval_spec.dict(),
            dataset_id=dataset_id,
            created_by=api_key,
            metadata=eval_request.metadata.dict() if eval_request.metadata else None
        )
        
        # Get created evaluation
        evaluation = eval_db.get_evaluation(eval_id)
        if not evaluation:
            raise ValueError("Failed to retrieve created evaluation")
        
        return EvaluationResponse(**evaluation)
        
    except Exception as e:
        logger.error(f"Failed to create evaluation: {e}")
        raise create_error_response(
            message=f"Failed to create evaluation: {str(e)}",
            error_type="server_error",
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR
        )


@router.get("/v1/evals", response_model=EvaluationListResponse)
@read_limit
async def list_evaluations(
    request: Request,
    limit: int = Query(20, ge=1, le=100),
    after: Optional[str] = Query(None),
    eval_type: Optional[str] = Query(None),
    api_key: str = Depends(verify_api_key)
):
    """List evaluations with pagination"""
    try:
        evaluations, has_more = eval_db.list_evaluations(
            limit=limit,
            after=after,
            eval_type=eval_type
        )
        
        first_id = evaluations[0]["id"] if evaluations else None
        last_id = evaluations[-1]["id"] if evaluations else None
        
        return EvaluationListResponse(
            object="list",
            data=[EvaluationResponse(**eval) for eval in evaluations],
            has_more=has_more,
            first_id=first_id,
            last_id=last_id
        )
        
    except Exception as e:
        logger.error(f"Failed to list evaluations: {e}")
        raise create_error_response(
            message=f"Failed to list evaluations: {str(e)}",
            error_type="server_error",
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR
        )


@router.get("/v1/evals/{eval_id}", response_model=EvaluationResponse)
async def get_evaluation(
    eval_id: str,
    api_key: str = Depends(verify_api_key)
):
    """Get evaluation by ID"""
    try:
        evaluation = eval_db.get_evaluation(eval_id)
        if not evaluation:
            raise create_error_response(
                message=f"Evaluation {eval_id} not found",
                error_type="not_found_error",
                param="eval_id",
                status_code=status.HTTP_404_NOT_FOUND
            )
        
        return EvaluationResponse(**evaluation)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get evaluation {eval_id}: {e}")
        raise create_error_response(
            message=f"Failed to get evaluation: {str(e)}",
            error_type="server_error",
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR
        )


@router.patch("/v1/evals/{eval_id}", response_model=EvaluationResponse)
async def update_evaluation(
    eval_id: str,
    update_request: UpdateEvaluationRequest,
    api_key: str = Depends(verify_api_key)
):
    """Update evaluation definition"""
    try:
        updates = update_request.dict(exclude_unset=True)
        if not updates:
            raise create_error_response(
                message="No updates provided",
                error_type="invalid_request_error"
            )
        
        success = eval_db.update_evaluation(eval_id, updates)
        if not success:
            raise create_error_response(
                message=f"Evaluation {eval_id} not found",
                error_type="not_found_error",
                param="eval_id",
                status_code=status.HTTP_404_NOT_FOUND
            )
        
        evaluation = eval_db.get_evaluation(eval_id)
        return EvaluationResponse(**evaluation)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to update evaluation {eval_id}: {e}")
        raise create_error_response(
            message=f"Failed to update evaluation: {str(e)}",
            error_type="server_error",
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR
        )


@router.delete("/v1/evals/{eval_id}")
async def delete_evaluation(
    eval_id: str,
    api_key: str = Depends(verify_api_key)
):
    """Delete (soft) an evaluation"""
    try:
        success = eval_db.delete_evaluation(eval_id)
        if not success:
            raise create_error_response(
                message=f"Evaluation {eval_id} not found",
                error_type="not_found_error",
                param="eval_id",
                status_code=status.HTTP_404_NOT_FOUND
            )
        
        return {"deleted": True, "id": eval_id}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to delete evaluation {eval_id}: {e}")
        raise create_error_response(
            message=f"Failed to delete evaluation: {str(e)}",
            error_type="server_error",
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR
        )


# ============= Run Endpoints =============

@router.post("/v1/evals/{eval_id}/runs", response_model=RunResponse, status_code=status.HTTP_202_ACCEPTED)
@run_limit
async def create_run(
    eval_id: str,
    run_request: CreateRunRequest,
    background_tasks: BackgroundTasks,
    request: Request,
    api_key: str = Depends(verify_api_key)
):
    """
    Create and start an evaluation run.
    
    This endpoint starts an asynchronous evaluation run for the specified evaluation.
    
    **Rate Limit:** 5 requests per minute (resource intensive)
    
    **Path Parameters:**
    - `eval_id`: ID of the evaluation to run
    
    **Request Body:**
    - `target_model`: Model to evaluate (e.g., "gpt-3.5-turbo")
    - `dataset_override`: Optional dataset to use instead of evaluation's default
    - `config`: Run configuration (temperature, max_workers, timeout_seconds)
    - `webhook_url`: Optional webhook URL for completion notification
    
    **Returns:** Run object with status "pending" and run ID for tracking
    
    **Note:** Use `/v1/runs/{run_id}` to check status or `/v1/runs/{run_id}/stream` for real-time updates
    """
    try:
        # Verify evaluation exists
        evaluation = eval_db.get_evaluation(eval_id)
        if not evaluation:
            raise create_error_response(
                message=f"Evaluation {eval_id} not found",
                error_type="not_found_error",
                param="eval_id",
                status_code=status.HTTP_404_NOT_FOUND
            )
        
        # Create run
        run_id = eval_db.create_run(
            eval_id=eval_id,
            target_model=run_request.target_model,
            config=run_request.config.dict(),
            webhook_url=run_request.webhook_url
        )
        
        # Prepare evaluation config
        eval_config = {
            "eval_type": evaluation["eval_type"],
            "eval_spec": evaluation["eval_spec"],
            "dataset_id": evaluation.get("dataset_id"),
            "dataset_override": run_request.dataset_override.dict() if run_request.dataset_override else None,
            "config": run_request.config.dict(),
            "webhook_url": run_request.webhook_url
        }
        
        # Start evaluation in background
        background_tasks.add_task(
            eval_runner.run_evaluation,
            run_id=run_id,
            eval_id=eval_id,
            eval_config=eval_config,
            background=False  # Already in background task
        )
        
        # Return run info
        run = eval_db.get_run(run_id)
        return RunResponse(**run)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to create run for evaluation {eval_id}: {e}")
        raise create_error_response(
            message=f"Failed to create run: {str(e)}",
            error_type="server_error",
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR
        )


@router.get("/v1/evals/{eval_id}/runs", response_model=RunListResponse)
async def list_runs(
    eval_id: str,
    limit: int = Query(20, ge=1, le=100),
    after: Optional[str] = Query(None),
    status: Optional[str] = Query(None),
    api_key: str = Depends(verify_api_key)
):
    """List runs for an evaluation"""
    try:
        # Verify evaluation exists
        evaluation = eval_db.get_evaluation(eval_id)
        if not evaluation:
            raise create_error_response(
                message=f"Evaluation {eval_id} not found",
                error_type="not_found_error",
                param="eval_id",
                status_code=status.HTTP_404_NOT_FOUND
            )
        
        runs, has_more = eval_db.list_runs(
            eval_id=eval_id,
            status=status,
            limit=limit,
            after=after
        )
        
        first_id = runs[0]["id"] if runs else None
        last_id = runs[-1]["id"] if runs else None
        
        return RunListResponse(
            object="list",
            data=[RunResponse(**run) for run in runs],
            has_more=has_more,
            first_id=first_id,
            last_id=last_id
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to list runs for evaluation {eval_id}: {e}")
        raise create_error_response(
            message=f"Failed to list runs: {str(e)}",
            error_type="server_error",
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR
        )


@router.get("/v1/runs/{run_id}", response_model=RunResponse)
async def get_run(
    run_id: str,
    api_key: str = Depends(verify_api_key)
):
    """Get run status and details"""
    try:
        run = eval_db.get_run(run_id)
        if not run:
            raise create_error_response(
                message=f"Run {run_id} not found",
                error_type="not_found_error",
                param="run_id",
                status_code=status.HTTP_404_NOT_FOUND
            )
        
        # Add estimated completion for running tasks
        if run["status"] == "running" and run.get("progress"):
            progress = run["progress"]
            if progress.get("total_samples", 0) > 0:
                percent_complete = (progress.get("completed_samples", 0) / progress["total_samples"]) * 100
                # Simple estimation (could be improved)
                if percent_complete > 0:
                    elapsed = int((datetime.utcnow() - datetime.fromisoformat(run["started_at"])).total_seconds())
                    estimated_total = elapsed / (percent_complete / 100)
                    run["estimated_completion"] = int(datetime.utcnow().timestamp() + (estimated_total - elapsed))
        
        return RunResponse(**run)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get run {run_id}: {e}")
        raise create_error_response(
            message=f"Failed to get run: {str(e)}",
            error_type="server_error",
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR
        )


@router.get("/v1/runs/{run_id}/results", response_model=RunResultsResponse)
async def get_run_results(
    run_id: str,
    api_key: str = Depends(verify_api_key)
):
    """Get run results (when complete)"""
    try:
        run = eval_db.get_run(run_id)
        if not run:
            raise create_error_response(
                message=f"Run {run_id} not found",
                error_type="not_found_error",
                param="run_id",
                status_code=status.HTTP_404_NOT_FOUND
            )
        
        if run["status"] not in ["completed", "failed"]:
            raise create_error_response(
                message=f"Run {run_id} is not complete (status: {run['status']})",
                error_type="invalid_request_error",
                param="run_id"
            )
        
        if not run.get("results"):
            raise create_error_response(
                message=f"No results available for run {run_id}",
                error_type="not_found_error",
                param="run_id"
            )
        
        # Calculate duration
        started_at = datetime.fromisoformat(run["started_at"]) if run["started_at"] else None
        completed_at = datetime.fromisoformat(run["completed_at"]) if run["completed_at"] else None
        duration_seconds = (completed_at - started_at).total_seconds() if started_at and completed_at else 0
        
        return RunResultsResponse(
            id=run["id"],
            eval_id=run["eval_id"],
            status=run["status"],
            started_at=int(started_at.timestamp()) if started_at else 0,
            completed_at=int(completed_at.timestamp()) if completed_at else 0,
            results=run["results"],
            usage=run.get("usage"),
            duration_seconds=duration_seconds
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get results for run {run_id}: {e}")
        raise create_error_response(
            message=f"Failed to get run results: {str(e)}",
            error_type="server_error",
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR
        )


@router.post("/v1/runs/{run_id}/cancel")
async def cancel_run(
    run_id: str,
    api_key: str = Depends(verify_api_key)
):
    """Cancel a running evaluation"""
    try:
        success = eval_runner.cancel_run(run_id)
        if not success:
            # Check if run exists
            run = eval_db.get_run(run_id)
            if not run:
                raise create_error_response(
                    message=f"Run {run_id} not found",
                    error_type="not_found_error",
                    param="run_id",
                    status_code=status.HTTP_404_NOT_FOUND
                )
            else:
                raise create_error_response(
                    message=f"Run {run_id} is not currently running (status: {run['status']})",
                    error_type="invalid_request_error",
                    param="run_id"
                )
        
        return {"cancelled": True, "id": run_id}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to cancel run {run_id}: {e}")
        raise create_error_response(
            message=f"Failed to cancel run: {str(e)}",
            error_type="server_error",
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR
        )


# ============= SSE Progress Tracking =============

@router.get("/v1/runs/{run_id}/stream")
async def stream_run_progress(
    run_id: str,
    api_key: str = Depends(verify_api_key)
):
    """
    Stream run progress updates via Server-Sent Events (SSE).
    
    This endpoint provides real-time progress updates for an evaluation run.
    
    **Event Types:**
    - `progress`: Progress update with completed/total samples
    - `completed`: Run completed successfully with results
    - `failed`: Run failed with error message
    - `cancelled`: Run was cancelled
    - `heartbeat`: Keep-alive signal
    - `error`: Stream error
    
    **Example Usage:**
    ```javascript
    const eventSource = new EventSource('/v1/runs/run_abc123/stream');
    eventSource.addEventListener('progress', (e) => {
        const progress = JSON.parse(e.data);
        console.log(`Progress: ${progress.completed_samples}/${progress.total_samples}`);
    });
    ```
    """
    
    async def generate_events() -> AsyncGenerator[str, None]:
        """Generate SSE events for run progress"""
        try:
            last_progress = None
            completed = False
            
            while not completed:
                # Get current run status
                run = eval_db.get_run(run_id)
                if not run:
                    yield f"event: error\ndata: {json.dumps({'message': 'Run not found'})}\n\n"
                    break
                
                # Check if status changed
                if run["status"] in ["completed", "failed", "cancelled"]:
                    completed = True
                    
                    # Send final status
                    if run["status"] == "completed":
                        results = eval_db.get_run_results(run_id)
                        yield f"event: completed\ndata: {json.dumps(results or {})}\n\n"
                    elif run["status"] == "failed":
                        yield f"event: failed\ndata: {json.dumps({'error': run.get('error_message', 'Unknown error')})}\n\n"
                    else:  # cancelled
                        yield f"event: cancelled\ndata: {json.dumps({'message': 'Run was cancelled'})}\n\n"
                    break
                
                # Send progress update if changed
                current_progress = run.get("progress", {})
                if current_progress != last_progress:
                    yield f"event: progress\ndata: {json.dumps(current_progress)}\n\n"
                    last_progress = current_progress
                
                # Send heartbeat to keep connection alive
                yield f"event: heartbeat\ndata: {json.dumps({'timestamp': datetime.utcnow().isoformat()})}\n\n"
                
                # Wait before next check
                await asyncio.sleep(1)
                
        except Exception as e:
            logger.error(f"Error streaming progress for run {run_id}: {e}")
            yield f"event: error\ndata: {json.dumps({'message': str(e)})}\n\n"
    
    return StreamingResponse(
        generate_events(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no"  # Disable Nginx buffering
        }
    )


# ============= Dataset Endpoints =============

@router.post("/v1/datasets", response_model=DatasetResponse)
async def create_dataset(
    dataset_request: CreateDatasetRequest,
    api_key: str = Depends(verify_api_key)
):
    """Create a new dataset"""
    try:
        dataset_id = eval_db.create_dataset(
            name=dataset_request.name,
            description=dataset_request.description,
            samples=dataset_request.samples,
            created_by=api_key,
            metadata=dataset_request.metadata
        )
        
        dataset = eval_db.get_dataset(dataset_id)
        if not dataset:
            raise ValueError("Failed to retrieve created dataset")
        
        return DatasetResponse(**dataset)
        
    except Exception as e:
        logger.error(f"Failed to create dataset: {e}")
        raise create_error_response(
            message=f"Failed to create dataset: {str(e)}",
            error_type="server_error",
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR
        )


@router.get("/v1/datasets", response_model=DatasetListResponse)
async def list_datasets(
    limit: int = Query(20, ge=1, le=100),
    after: Optional[str] = Query(None),
    api_key: str = Depends(verify_api_key)
):
    """List datasets with pagination"""
    try:
        datasets, has_more = eval_db.list_datasets(
            limit=limit,
            after=after
        )
        
        first_id = datasets[0]["id"] if datasets else None
        last_id = datasets[-1]["id"] if datasets else None
        
        return DatasetListResponse(
            object="list",
            data=[DatasetResponse(**ds) for ds in datasets],
            has_more=has_more,
            first_id=first_id,
            last_id=last_id
        )
        
    except Exception as e:
        logger.error(f"Failed to list datasets: {e}")
        raise create_error_response(
            message=f"Failed to list datasets: {str(e)}",
            error_type="server_error",
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR
        )


@router.get("/v1/datasets/{dataset_id}", response_model=DatasetResponse)
async def get_dataset(
    dataset_id: str,
    api_key: str = Depends(verify_api_key)
):
    """Get dataset by ID"""
    try:
        dataset = eval_db.get_dataset(dataset_id)
        if not dataset:
            raise create_error_response(
                message=f"Dataset {dataset_id} not found",
                error_type="not_found_error",
                param="dataset_id",
                status_code=status.HTTP_404_NOT_FOUND
            )
        
        return DatasetResponse(**dataset)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get dataset {dataset_id}: {e}")
        raise create_error_response(
            message=f"Failed to get dataset: {str(e)}",
            error_type="server_error",
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR
        )


@router.delete("/v1/datasets/{dataset_id}")
async def delete_dataset(
    dataset_id: str,
    api_key: str = Depends(verify_api_key)
):
    """Delete a dataset"""
    try:
        success = eval_db.delete_dataset(dataset_id)
        if not success:
            raise create_error_response(
                message=f"Dataset {dataset_id} not found",
                error_type="not_found_error",
                param="dataset_id",
                status_code=status.HTTP_404_NOT_FOUND
            )
        
        return {"deleted": True, "id": dataset_id}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to delete dataset {dataset_id}: {e}")
        raise create_error_response(
            message=f"Failed to delete dataset: {str(e)}",
            error_type="server_error",
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR
        )