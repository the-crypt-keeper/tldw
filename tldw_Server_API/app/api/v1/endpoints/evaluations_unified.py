# evaluations_unified.py - Unified evaluation API endpoints
"""
Unified evaluation API combining OpenAI-compatible and tldw-specific endpoints.

This module provides a single, cohesive API for all evaluation functionality.
"""

import os
import json
import asyncio
from datetime import datetime
from typing import List, Optional, Dict, Any
from fastapi import APIRouter, HTTPException, Depends, BackgroundTasks, status, Query, Request, Response, Header
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.responses import StreamingResponse
from loguru import logger

# Import unified schemas
from tldw_Server_API.app.api.v1.schemas.evaluation_schemas_unified import (
    # OpenAI-compatible schemas
    CreateEvaluationRequest, UpdateEvaluationRequest, EvaluationResponse,
    CreateRunRequest, RunResponse, RunResultsResponse,
    CreateDatasetRequest, DatasetResponse,
    EvaluationListResponse, RunListResponse, DatasetListResponse,
    
    # tldw-specific schemas
    GEvalRequest, GEvalResponse,
    RAGEvaluationRequest, RAGEvaluationResponse,
    ResponseQualityRequest, ResponseQualityResponse,
    BatchEvaluationRequest, BatchEvaluationResponse,
    CustomMetricRequest, CustomMetricResponse,
    EvaluationComparisonRequest, EvaluationComparisonResponse,
    EvaluationHistoryRequest, EvaluationHistoryResponse,
    
    # Webhook schemas
    WebhookRegistrationRequest, WebhookRegistrationResponse,
    WebhookUpdateRequest, WebhookStatusResponse,
    WebhookTestRequest, WebhookTestResponse,
    RateLimitStatusResponse,
    
    # Common schemas
    ErrorResponse, ErrorDetail, HealthCheckResponse
)

# Import unified service
from tldw_Server_API.app.core.Evaluations.unified_evaluation_service import (
    get_unified_evaluation_service,
    UnifiedEvaluationService
)

# Import auth and rate limiting
from tldw_Server_API.app.core.AuthNZ.settings import get_settings
from tldw_Server_API.app.core.AuthNZ.jwt_service import JWTService
from tldw_Server_API.app.core.AuthNZ.exceptions import InvalidTokenError, TokenExpiredError
from tldw_Server_API.app.core.AuthNZ.rate_limiter import RateLimiter, get_rate_limiter
from tldw_Server_API.app.api.v1.API_Deps.auth_deps import get_rate_limiter_dep

# Import additional services
from tldw_Server_API.app.core.Evaluations.webhook_manager import webhook_manager, WebhookEvent
from tldw_Server_API.app.core.Evaluations.user_rate_limiter import user_rate_limiter
from tldw_Server_API.app.core.Evaluations.metrics_advanced import advanced_metrics

# Create router
router = APIRouter(prefix="/evaluations", tags=["Evaluations"])

# Security
security = HTTPBearer(auto_error=False)

# Lazy evaluation service initialization 
_evaluation_service = None

def get_evaluation_service():
    """Get evaluation service with lazy initialization"""
    global _evaluation_service
    if _evaluation_service is None:
        _evaluation_service = get_unified_evaluation_service()
    return _evaluation_service


# ============= Authentication =============

async def verify_api_key(
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(security),
    x_api_key: Optional[str] = Header(None, alias="X-API-KEY")
) -> str:
    """
    Verify API key or JWT token based on authentication mode.
    
    Supports both single-user and multi-user modes with OpenAI compatibility.
    """
    settings = get_settings()
    
    # Determine token source
    if settings.AUTH_MODE == "single_user" and x_api_key:
        token = x_api_key
    elif credentials:
        token = credentials.credentials
    else:
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
        
        if token == expected_token or (token.startswith("sk-") and token == expected_token):
            return "single_user"
            
    elif settings.AUTH_MODE == "multi_user":
        try:
            jwt_service = JWTService(settings)
            payload = jwt_service.decode_access_token(token)
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
    
    raise HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail={"error": {
            "message": "Invalid API key or token",
            "type": "authentication_error",
            "code": "invalid_credentials"
        }}
    )


# ============= Rate Limiting =============

async def check_evaluation_rate_limit(
    request: Request,
    rate_limiter: RateLimiter = Depends(get_rate_limiter_dep)
):
    """Check rate limit for evaluation endpoints"""
    client_ip = request.client.host if request.client else "unknown"
    path = request.url.path
    
    # Determine rate limit based on endpoint type
    if "batch" in path:
        limit = 5
        endpoint_type = "eval_batch"
    elif "/runs" in path:
        limit = 10
        endpoint_type = "eval_run"
    else:
        limit = 60
        endpoint_type = "eval_standard"
    
    allowed, metadata = await rate_limiter.check_rate_limit(
        client_ip,
        endpoint_type,
        limit=limit,
        window_minutes=1
    )
    
    if not allowed:
        retry_after = metadata.get("retry_after", 60)
        logger.warning(f"Rate limit exceeded for {client_ip} on {endpoint_type}")
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail=f"Rate limit exceeded. Retry after {retry_after} seconds",
            headers={"Retry-After": str(retry_after)}
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
        "NotFoundError": "The requested resource was not found",
        "ValidationError": "Validation failed for the provided data",
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


# ============= OpenAI-Compatible Evaluation Endpoints =============

@router.post("", response_model=EvaluationResponse, status_code=status.HTTP_201_CREATED)
async def create_evaluation(
    eval_request: CreateEvaluationRequest,
    user_id: str = Depends(verify_api_key)
):
    """
    Create a new evaluation definition (OpenAI-compatible).
    
    This endpoint creates an evaluation that can be run multiple times with different models.
    """
    try:
        evaluation = await get_evaluation_service().create_evaluation(
            name=eval_request.name,
            description=eval_request.description,
            eval_type=eval_request.eval_type,
            eval_spec=eval_request.eval_spec.dict(),
            dataset_id=eval_request.dataset_id,
            dataset=[sample.dict() for sample in eval_request.dataset] if eval_request.dataset else None,
            metadata=eval_request.metadata.dict() if eval_request.metadata else None,
            created_by=user_id
        )
        
        return EvaluationResponse(**evaluation)
        
    except Exception as e:
        logger.error(f"Failed to create evaluation: {e}")
        raise create_error_response(
            message=f"Failed to create evaluation: {sanitize_error_message(e, 'evaluation creation')}",
            error_type="server_error",
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR
        )


@router.get("", response_model=EvaluationListResponse)
async def list_evaluations(
    limit: int = Query(20, ge=1, le=100),
    after: Optional[str] = Query(None),
    eval_type: Optional[str] = Query(None),
    user_id: str = Depends(verify_api_key)
):
    """List evaluations with pagination"""
    try:
        evaluations, has_more = await get_evaluation_service().list_evaluations(
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


@router.get("/{eval_id}", response_model=EvaluationResponse)
async def get_evaluation(
    eval_id: str,
    user_id: str = Depends(verify_api_key)
):
    """Get evaluation by ID"""
    try:
        evaluation = await get_evaluation_service().get_evaluation(eval_id)
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


@router.patch("/{eval_id}", response_model=EvaluationResponse)
async def update_evaluation(
    eval_id: str,
    update_request: UpdateEvaluationRequest,
    user_id: str = Depends(verify_api_key)
):
    """Update evaluation definition"""
    try:
        updates = update_request.dict(exclude_unset=True)
        if not updates:
            raise create_error_response(
                message="No updates provided",
                error_type="invalid_request_error"
            )
        
        success = await get_evaluation_service().update_evaluation(
            eval_id, updates, updated_by=user_id
        )
        
        if not success:
            raise create_error_response(
                message=f"Evaluation {eval_id} not found",
                error_type="not_found_error",
                param="eval_id",
                status_code=status.HTTP_404_NOT_FOUND
            )
        
        evaluation = await get_evaluation_service().get_evaluation(eval_id)
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


@router.delete("/{eval_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_evaluation(
    eval_id: str,
    user_id: str = Depends(verify_api_key)
):
    """Delete an evaluation"""
    try:
        success = await get_evaluation_service().delete_evaluation(eval_id, deleted_by=user_id)
        if not success:
            raise create_error_response(
                message=f"Evaluation {eval_id} not found",
                error_type="not_found_error",
                param="eval_id",
                status_code=status.HTTP_404_NOT_FOUND
            )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to delete evaluation {eval_id}: {e}")
        raise create_error_response(
            message=f"Failed to delete evaluation: {sanitize_error_message(e, 'deleting evaluation')}",
            error_type="server_error",
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR
        )


# ============= Run Management Endpoints =============

@router.post("/{eval_id}/runs", response_model=RunResponse, status_code=status.HTTP_202_ACCEPTED)
async def create_run(
    eval_id: str,
    run_request: CreateRunRequest,
    background_tasks: BackgroundTasks,
    user_id: str = Depends(verify_api_key),
    _: None = Depends(check_evaluation_rate_limit)
):
    """Create and start an evaluation run"""
    try:
        run = await get_evaluation_service().create_run(
            eval_id=eval_id,
            target_model=run_request.target_model,
            config=run_request.config.dict() if run_request.config else None,
            dataset_override=run_request.dataset_override.dict() if run_request.dataset_override else None,
            webhook_url=str(run_request.webhook_url) if run_request.webhook_url else None,
            created_by=user_id
        )
        
        return RunResponse(**run)
        
    except ValueError as e:
        raise create_error_response(
            message=sanitize_error_message(e, "creating run"),
            error_type="not_found_error",
            param="eval_id",
            status_code=status.HTTP_404_NOT_FOUND
        )
    except Exception as e:
        logger.error(f"Failed to create run: {e}")
        raise create_error_response(
            message=f"Failed to create run: {sanitize_error_message(e, 'creating run')}",
            error_type="server_error",
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR
        )


@router.get("/{eval_id}/runs", response_model=RunListResponse)
async def list_runs(
    eval_id: str,
    limit: int = Query(20, ge=1, le=100),
    after: Optional[str] = Query(None),
    status: Optional[str] = Query(None),
    user_id: str = Depends(verify_api_key)
):
    """List runs for an evaluation"""
    try:
        runs, has_more = await get_evaluation_service().list_runs(
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
        
    except Exception as e:
        logger.error(f"Failed to list runs: {e}")
        raise create_error_response(
            message=f"Failed to list runs: {sanitize_error_message(e, 'listing runs')}",
            error_type="server_error",
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR
        )


# ============= tldw-Specific Evaluation Endpoints =============

@router.post("/geval", response_model=GEvalResponse, dependencies=[Depends(check_evaluation_rate_limit)])
async def evaluate_geval(
    request: GEvalRequest,
    user_id: str = Depends(verify_api_key)
):
    """
    Evaluate a summary using G-Eval metrics.
    
    G-Eval evaluates summaries on fluency, consistency, relevance, and coherence.
    """
    try:
        result = await get_evaluation_service().evaluate_geval(
            source_text=request.source_text,
            summary=request.summary,
            metrics=request.metrics,
            api_name=request.api_name,
            api_key=request.api_key,
            user_id=user_id
        )
        
        # Format response
        return GEvalResponse(
            metrics=result["results"].get("metrics", {}),
            average_score=result["results"].get("average_score", 0.0),
            summary_assessment=result["results"].get("assessment", "Evaluation complete"),
            evaluation_time=result["evaluation_time"],
            metadata={"evaluation_id": result["evaluation_id"]}
        )
        
    except Exception as e:
        logger.error(f"G-Eval evaluation failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Evaluation failed: {sanitize_error_message(e, 'G-Eval evaluation')}"
        )


@router.post("/rag", response_model=RAGEvaluationResponse, dependencies=[Depends(check_evaluation_rate_limit)])
async def evaluate_rag(
    request: RAGEvaluationRequest,
    user_id: str = Depends(verify_api_key)
):
    """
    Evaluate RAG system performance.
    
    Evaluates relevance, faithfulness, answer similarity, and context precision.
    """
    try:
        result = await get_evaluation_service().evaluate_rag(
            query=request.query,
            contexts=request.retrieved_contexts,
            response=request.generated_response,
            ground_truth=request.ground_truth,
            metrics=request.metrics,
            api_name=request.api_name,
            user_id=user_id
        )
        
        # Extract metrics from results
        metrics = result["results"].get("metrics", {})
        
        return RAGEvaluationResponse(
            metrics=metrics,
            overall_score=result["results"].get("overall_score", 0.0),
            retrieval_quality=result["results"].get("retrieval_quality", 0.0),
            generation_quality=result["results"].get("generation_quality", 0.0),
            suggestions=result["results"].get("suggestions", []),
            metadata={"evaluation_id": result["evaluation_id"]}
        )
        
    except Exception as e:
        logger.error(f"RAG evaluation failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"RAG evaluation failed: {sanitize_error_message(e, 'RAG evaluation')}"
        )


@router.post("/response-quality", response_model=ResponseQualityResponse, dependencies=[Depends(check_evaluation_rate_limit)])
async def evaluate_response_quality(
    request: ResponseQualityRequest,
    user_id: str = Depends(verify_api_key)
):
    """
    Evaluate the quality of a generated response.
    
    Checks relevance, completeness, accuracy, and format compliance.
    """
    try:
        result = await get_evaluation_service().evaluate_response_quality(
            prompt=request.prompt,
            response=request.response,
            expected_format=request.expected_format,
            custom_criteria=request.evaluation_criteria,
            api_name=request.api_name,
            user_id=user_id
        )
        
        return ResponseQualityResponse(
            metrics=result["results"].get("metrics", {}),
            overall_quality=result["results"].get("overall_quality", 0.0),
            format_compliance=result["results"].get("format_compliance", {}),
            issues=result["results"].get("issues", []),
            improvements=result["results"].get("improvements", [])
        )
        
    except Exception as e:
        logger.error(f"Response quality evaluation failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Quality evaluation failed: {sanitize_error_message(e, 'quality evaluation')}"
        )


# ============= Dataset Management Endpoints =============

@router.post("/datasets", response_model=DatasetResponse, status_code=status.HTTP_201_CREATED)
async def create_dataset(
    dataset_request: CreateDatasetRequest,
    user_id: str = Depends(verify_api_key)
):
    """Create a new dataset"""
    try:
        dataset_id = await get_evaluation_service().create_dataset(
            name=dataset_request.name,
            description=dataset_request.description,
            samples=[s.dict() for s in dataset_request.samples],
            metadata=dataset_request.metadata,
            created_by=user_id
        )
        
        dataset = await get_evaluation_service().get_dataset(dataset_id)
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
    user_id: str = Depends(verify_api_key)
):
    """List datasets with pagination"""
    try:
        datasets, has_more = await get_evaluation_service().list_datasets(
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
    user_id: str = Depends(verify_api_key)
):
    """Get dataset by ID"""
    try:
        dataset = await get_evaluation_service().get_dataset(dataset_id)
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
    user_id: str = Depends(verify_api_key)
):
    """Delete a dataset"""
    try:
        success = await get_evaluation_service().delete_dataset(dataset_id, deleted_by=user_id)
        if not success:
            raise create_error_response(
                message=f"Dataset {dataset_id} not found",
                error_type="not_found_error",
                param="dataset_id",
                status_code=status.HTTP_404_NOT_FOUND
            )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to delete dataset {dataset_id}: {e}")
        raise create_error_response(
            message=f"Failed to delete dataset: {sanitize_error_message(e, 'deleting dataset')}",
            error_type="server_error",
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR
        )


# ============= Health & Metrics Endpoints =============

@router.get("/health", response_model=HealthCheckResponse)
async def health_check():
    """Check evaluation service health"""
    try:
        health = await get_evaluation_service().health_check()
        return HealthCheckResponse(**health)
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return HealthCheckResponse(
            status="unhealthy",
            version="1.0.0",
            uptime=0,
            database="disconnected"
        )


@router.get("/metrics")
async def get_metrics(request: Request):
    """Get Prometheus metrics"""
    try:
        metrics_summary = await get_evaluation_service().get_metrics_summary()
        
        # Handle failure from service so error message is never exposed
        if "error" in metrics_summary:
            logger.error(f"Metrics endpoint service error: {metrics_summary['error']}")
            # Return a generic error message in both text/plain and JSON responses
            if "text/plain" in request.headers.get("accept", ""):
                # Prometheus format error response
                output = "# HELP evaluation_metrics_failed Metric collection failure\n"
                output += "# TYPE evaluation_metrics_failed counter\n"
                output += "evaluation_metrics_failed{} 1\n"
                return Response(
                    content=output,
                    media_type="text/plain; version=0.0.4; charset=utf-8"
                )
            # Return JSON error response
            return {"error": "Metrics are currently unavailable"}
        
        # Format as Prometheus text format if requested
        if "text/plain" in request.headers.get("accept", ""):
            # Convert to Prometheus format (simplified)
            output = "# HELP evaluation_requests_total Total evaluation requests\n"
            output += "# TYPE evaluation_requests_total counter\n"
            output += f"evaluation_requests_total {{}} {metrics_summary.get('total_requests', 0)}\n"
            
            return Response(
                content=output,
                media_type="text/plain; version=0.0.4; charset=utf-8"
            )
        
        # Return JSON format
        return metrics_summary
        
    except Exception as e:
        logger.error(f"Failed to get metrics: {e}")
        # Do not expose internal error, return generic error
        return {"error": "Metrics are currently unavailable"}


# ============= Webhook Management Endpoints =============

@router.post("/webhooks", response_model=WebhookRegistrationResponse)
async def register_webhook(
    request: WebhookRegistrationRequest,
    user_id: str = Depends(verify_api_key)
):
    """Register a webhook for evaluation notifications"""
    try:
        # Convert enum values to strings for webhook manager
        events = [e.value for e in request.events]
        
        result = await webhook_manager.register_webhook(
            user_id=user_id,
            url=str(request.url),
            events=events,
            secret=request.secret
        )
        
        return WebhookRegistrationResponse(**result)
        
    except Exception as e:
        logger.error(f"Failed to register webhook: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to register webhook: {sanitize_error_message(e, 'webhook registration')}"
        )


@router.get("/webhooks", response_model=List[WebhookStatusResponse])
async def list_webhooks(
    user_id: str = Depends(verify_api_key)
):
    """List all registered webhooks for the authenticated user"""
    try:
        webhooks = await webhook_manager.get_webhook_status(user_id)
        return [WebhookStatusResponse(**w) for w in webhooks]
        
    except Exception as e:
        logger.error(f"Failed to list webhooks: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to list webhooks: {sanitize_error_message(e, 'listing webhooks')}"
        )


@router.delete("/webhooks")
async def unregister_webhook(
    url: str = Query(..., description="Webhook URL to unregister"),
    user_id: str = Depends(verify_api_key)
):
    """Unregister a webhook"""
    try:
        success = await webhook_manager.unregister_webhook(user_id, url)
        
        if not success:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Webhook not found"
            )
        
        return {"message": "Webhook unregistered successfully"}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to unregister webhook: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to unregister webhook: {sanitize_error_message(e, 'webhook removal')}"
        )


@router.post("/webhooks/test", response_model=WebhookTestResponse)
async def test_webhook(
    request: WebhookTestRequest,
    user_id: str = Depends(verify_api_key)
):
    """Send a test webhook to verify endpoint configuration"""
    try:
        result = await webhook_manager.test_webhook(user_id, str(request.url))
        return WebhookTestResponse(**result)
        
    except Exception as e:
        logger.error(f"Failed to test webhook: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to test webhook: {sanitize_error_message(e, 'webhook testing')}"
        )


# ============= Rate Limit Management =============

@router.get("/rate-limits", response_model=RateLimitStatusResponse)
async def get_rate_limit_status(
    user_id: str = Depends(verify_api_key)
):
    """Get current rate limit status for the authenticated user"""
    try:
        summary = await user_rate_limiter.get_usage_summary(user_id)
        return RateLimitStatusResponse(**summary)
        
    except Exception as e:
        logger.error(f"Failed to get rate limit status: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get rate limit status: {sanitize_error_message(e, 'rate limit check')}"
        )


# ============= Additional Run Endpoints =============

@router.get("/runs/{run_id}", response_model=RunResponse)
async def get_run(
    run_id: str,
    user_id: str = Depends(verify_api_key)
):
    """Get run status and details"""
    try:
        run = await get_evaluation_service().get_run(run_id)
        if not run:
            raise create_error_response(
                message=f"Run {run_id} not found",
                error_type="not_found_error",
                param="run_id",
                status_code=status.HTTP_404_NOT_FOUND
            )
        
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


@router.post("/runs/{run_id}/cancel")
async def cancel_run(
    run_id: str,
    user_id: str = Depends(verify_api_key)
):
    """Cancel a running evaluation"""
    try:
        success = await get_evaluation_service().cancel_run(run_id, cancelled_by=user_id)
        
        if success:
            return {"status": "cancelled", "id": run_id}
        else:
            raise create_error_response(
                message=f"Failed to cancel run {run_id}",
                error_type="server_error",
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR
            )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to cancel run {run_id}: {e}")
        raise create_error_response(
            message=f"Failed to cancel run: {sanitize_error_message(e, 'cancelling run')}",
            error_type="server_error",
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR
        )