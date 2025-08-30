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
from fastapi import APIRouter, HTTPException, Depends, BackgroundTasks, status, Query, Security, Request, Header
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
from tldw_Server_API.app.core.AuthNZ.settings import get_settings
from tldw_Server_API.app.core.AuthNZ.jwt_service import JWTService
from tldw_Server_API.app.core.AuthNZ.exceptions import InvalidTokenError, TokenExpiredError

# Create router
router = APIRouter(tags=["Evaluations"])

# Security
security = HTTPBearer(auto_error=False)

# Rate limiting configuration from settings
settings = get_settings()

# Configure rate limiter with user-specific or IP-based keys
def get_rate_limit_key(request: Request) -> str:
    """Get rate limit key based on authentication mode."""
    # Try to get authenticated user from request state
    if hasattr(request.state, "user_id"):
        return f"user_{request.state.user_id}"
    # Fall back to IP address
    return get_remote_address(request)

limiter = Limiter(key_func=get_rate_limit_key)

# Rate limit decorators based on configuration
# Use settings if available, otherwise use sensible defaults
rate_limit_per_minute = getattr(settings, 'RATE_LIMIT_PER_MINUTE', 60)
burst_limit = getattr(settings, 'RATE_LIMIT_BURST', 10)

# Different limits for different operation types
create_limit = limiter.limit(f"{rate_limit_per_minute}/minute")  # Create operations
read_limit = limiter.limit(f"{rate_limit_per_minute * 2}/minute")  # Read operations (allow more)
run_limit = limiter.limit(f"{max(10, rate_limit_per_minute // 2)}/minute")  # Run operations (more restrictive)
burst_limit_decorator = limiter.limit(f"{burst_limit}/second")  # Burst protection

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

async def verify_api_key(
    credentials: Optional[HTTPAuthorizationCredentials] = Security(security),
    x_api_key: Optional[str] = Header(None, alias="X-API-KEY")
) -> str:
    """
    Verify API key or JWT token based on authentication mode.
    
    Supports:
    - Single-user mode: API key from environment or config (X-API-KEY header or Bearer token)
    - Multi-user mode: JWT tokens (Bearer token)
    - OpenAI compatibility: sk- prefixed keys
    """
    settings = get_settings()
    
    # Check for X-API-KEY header first (for single-user mode)
    if settings.AUTH_MODE == "single_user" and x_api_key:
        token = x_api_key
    elif credentials:
        token = credentials.credentials
    else:
        # No credentials provided
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail={"error": {
                "message": "Missing API key or token",
                "type": "authentication_error",
                "code": "missing_credentials"
            }}
        )
    
    # Remove Bearer prefix if present
    if token.startswith("Bearer "):
        token = token[7:]
    
    # Handle based on authentication mode
    if settings.AUTH_MODE == "single_user":
        # Single-user mode: Check against configured API key
        expected_token = os.getenv("API_BEARER") or os.getenv("SINGLE_USER_API_KEY") or settings.SINGLE_USER_API_KEY
        
        if not expected_token:
            logger.error("No API key configured for single-user mode")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail={"error": {
                    "message": "Server authentication not configured",
                    "type": "configuration_error",
                    "code": "auth_not_configured"
                }}
            )
        
        # Check token matches expected (including sk- prefixed keys)
        if token == expected_token:
            return token
        
        # For OpenAI compatibility, accept sk- prefixed keys that match expected token
        if token.startswith("sk-") and token == expected_token:
            return token
            
    elif settings.AUTH_MODE == "multi_user":
        # Multi-user mode: Verify JWT token
        try:
            jwt_service = JWTService(settings)
            payload = jwt_service.decode_access_token(token)
            # Return user ID as the authenticated identifier
            return f"user_{payload['sub']}"
        except TokenExpiredError:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail={"error": {
                    "message": "Token has expired",
                    "type": "authentication_error",
                    "code": "token_expired"
                }}
            )
        except InvalidTokenError as e:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail={"error": {
                    "message": sanitize_error_message(e, "authentication"),
                    "type": "authentication_error",
                    "code": "invalid_token"
                }}
            )
    
    # If we get here, authentication failed
    raise HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail={"error": {
            "message": "Invalid API key or token",
            "type": "authentication_error",
            "code": "invalid_credentials"
        }}
    )


# ============= Error Handling =============

def sanitize_error_message(error: Exception, context: str = "") -> str:
    """Sanitize error messages to prevent information exposure.
    
    Args:
        error: The exception to sanitize
        context: Optional context about where the error occurred
        
    Returns:
        A safe error message that doesn't expose sensitive information
    """
    # Log the full error details for debugging
    logger.error(f"Error in {context}: {type(error).__name__}: {str(error)}")
    
    # Map specific exception types to safe messages
    error_type = type(error).__name__
    
    # Common safe error messages
    safe_messages = {
        "FileNotFoundError": "The requested resource was not found",
        "PermissionError": "Permission denied for this operation",
        "ValueError": "Invalid input provided",
        "KeyError": "Required data is missing",
        "ConnectionError": "Connection failed. Please try again later",
        "TimeoutError": "Operation timed out. Please try again",
        "DatabaseError": "Database operation failed",
        "IntegrityError": "Data integrity error occurred",
    }
    
    # Return safe message based on error type
    if error_type in safe_messages:
        return safe_messages[error_type]
    
    # For unknown errors, return a generic message
    if context:
        return f"An error occurred during {context}"
    return "An internal error occurred. Please try again later"


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

@router.post("/evals", response_model=EvaluationResponse, status_code=status.HTTP_201_CREATED)
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
            message=f"Failed to create evaluation: {sanitize_error_message(e, 'evaluation creation')}",
            error_type="server_error",
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR
        )


@router.get("/evals", response_model=EvaluationListResponse)
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
            message=f"Failed to list evaluations: {sanitize_error_message(e, 'listing evaluations')}",
            error_type="server_error",
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR
        )


@router.get("/evals/{eval_id}", response_model=EvaluationResponse)
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
            message=f"Failed to get evaluation: {sanitize_error_message(e, 'retrieving evaluation')}",
            error_type="server_error",
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR
        )


@router.patch("/evals/{eval_id}", response_model=EvaluationResponse)
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
        
        # Handle metadata - if it's a dict from EvaluationMetadata model, extract it
        if "metadata" in updates and isinstance(updates["metadata"], dict):
            # Already a dict, use as-is
            pass
        
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
            message=f"Failed to update evaluation: {sanitize_error_message(e, 'updating evaluation')}",
            error_type="server_error",
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR
        )


@router.delete("/evals/{eval_id}", status_code=status.HTTP_204_NO_CONTENT)
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
        
        # Return nothing for 204 No Content
        return
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to delete evaluation {eval_id}: {e}")
        raise create_error_response(
            message=f"Failed to delete evaluation: {sanitize_error_message(e, 'deleting evaluation')}",
            error_type="server_error",
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR
        )


# ============= Run Endpoints =============

@router.post("/evals/{eval_id}/runs", response_model=RunResponse, status_code=status.HTTP_202_ACCEPTED)
@run_limit
@burst_limit_decorator
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
    
    **Note:** Use `/api/v1/runs/{run_id}` to check status or `/api/v1/runs/{run_id}/stream` for real-time updates
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
            message=f"Failed to create run: {sanitize_error_message(e, 'creating run')}",
            error_type="server_error",
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR
        )


@router.get("/evals/{eval_id}/runs", response_model=RunListResponse)
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
            message=f"Failed to list runs: {sanitize_error_message(e, 'listing runs')}",
            error_type="server_error",
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR
        )


@router.get("/runs/{run_id}", response_model=RunResponse)
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
            message=f"Failed to get run: {sanitize_error_message(e, 'retrieving run')}",
            error_type="server_error",
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR
        )


@router.get("/runs/{run_id}/results", response_model=RunResultsResponse)
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
            message=f"Failed to get run results: {sanitize_error_message(e, 'retrieving run results')}",
            error_type="server_error",
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR
        )


@router.post("/runs/{run_id}/cancel")
async def cancel_run(
    run_id: str,
    api_key: str = Depends(verify_api_key)
):
    """Cancel a running evaluation"""
    try:
        # Check if run exists
        run = eval_db.get_run(run_id)
        if not run:
            raise create_error_response(
                message=f"Run {run_id} not found",
                error_type="not_found_error",
                param="run_id",
                status_code=status.HTTP_404_NOT_FOUND
            )
        
        # If already completed or cancelled, return success
        if run['status'] in ['completed', 'failed', 'cancelled']:
            return {"status": run['status'], "id": run_id}
        
        # Try to cancel the run
        success = eval_runner.cancel_run(run_id)
        if success:
            return {"status": "cancelled", "id": run_id}
        else:
            # If not in running tasks, update status directly
            eval_db.update_run_status(run_id, "cancelled")
            return {"status": "cancelled", "id": run_id}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to cancel run {run_id}: {e}")
        raise create_error_response(
            message=f"Failed to cancel run: {sanitize_error_message(e, 'cancelling run')}",
            error_type="server_error",
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR
        )


# ============= SSE Progress Tracking =============

@router.get("/runs/{run_id}/stream")
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
    const eventSource = new EventSource('/api/v1/runs/run_abc123/stream');
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
            safe_error_msg = sanitize_error_message(e, f"streaming progress for run {run_id}")
            yield f"event: error\ndata: {json.dumps({'message': safe_error_msg})}\n\n"
    
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

@router.post("/datasets", response_model=DatasetResponse, status_code=status.HTTP_201_CREATED)
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
            message=f"Failed to create dataset: {sanitize_error_message(e, 'creating dataset')}",
            error_type="server_error",
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR
        )


@router.get("/datasets", response_model=DatasetListResponse)
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
            message=f"Failed to list datasets: {sanitize_error_message(e, 'listing datasets')}",
            error_type="server_error",
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR
        )


@router.get("/datasets/{dataset_id}", response_model=DatasetResponse)
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
            message=f"Failed to get dataset: {sanitize_error_message(e, 'retrieving dataset')}",
            error_type="server_error",
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR
        )


@router.delete("/datasets/{dataset_id}", status_code=status.HTTP_204_NO_CONTENT)
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
        
        # Return nothing for 204 No Content
        return
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to delete dataset {dataset_id}: {e}")
        raise create_error_response(
            message=f"Failed to delete dataset: {sanitize_error_message(e, 'deleting dataset')}",
            error_type="server_error",
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR
        )