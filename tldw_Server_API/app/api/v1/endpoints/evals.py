# evals.py - Evaluation API endpoints
"""
API endpoints for evaluation functionality.

Provides endpoints for:
- G-Eval summarization evaluation
- RAG system evaluation
- Response quality assessment
- Batch evaluations
- Evaluation history and comparison
"""

import asyncio
import json
import time
from typing import List, Dict, Any, Optional
from fastapi import APIRouter, HTTPException, Depends, status, Query, Request, Response
from loguru import logger

# Import schemas
from tldw_Server_API.app.api.v1.schemas.evaluation_schema import (
    GEvalRequest, GEvalResponse,
    RAGEvaluationRequest, RAGEvaluationResponse,
    ResponseQualityRequest, ResponseQualityResponse,
    BatchEvaluationRequest, BatchEvaluationResponse,
    EvaluationHistoryRequest, EvaluationHistoryResponse,
    CustomMetricRequest, CustomMetricResponse,
    EvaluationComparisonRequest, EvaluationComparisonResponse,
    EvaluationMetric
)

# Import evaluation modules
from tldw_Server_API.app.core.Evaluations.ms_g_eval import run_geval
from tldw_Server_API.app.core.Evaluations.rag_evaluator import RAGEvaluator
from tldw_Server_API.app.core.Evaluations.response_quality_evaluator import ResponseQualityEvaluator
from tldw_Server_API.app.core.Evaluations.evaluation_manager import EvaluationManager
from tldw_Server_API.app.core.Evaluations.metrics import (
    get_metrics, track_request_metrics, track_evaluation_metrics
)
from tldw_Server_API.app.core.Evaluations.config_validator import validate_configuration

# Import new production features
from tldw_Server_API.app.core.Evaluations.user_rate_limiter import user_rate_limiter, UserTier
from tldw_Server_API.app.core.Evaluations.webhook_manager import webhook_manager, WebhookEvent
from tldw_Server_API.app.core.Evaluations.metrics_advanced import advanced_metrics

# Import webhook schemas
from tldw_Server_API.app.api.v1.schemas.webhook_schemas import (
    WebhookRegistrationRequest, WebhookRegistrationResponse,
    WebhookUpdateRequest, WebhookStatusResponse,
    WebhookTestRequest, WebhookTestResponse,
    RateLimitStatusResponse
)

# Import rate limiting and auth dependencies
from tldw_Server_API.app.core.AuthNZ.rate_limiter import RateLimiter, get_rate_limiter
from tldw_Server_API.app.api.v1.API_Deps.auth_deps import get_rate_limiter_dep

# Create router
router = APIRouter(prefix="/evaluations", tags=["Evaluations"])

# Initialize evaluation components
evaluation_manager = EvaluationManager()
# Lazy initialization - create evaluators only when needed to avoid startup embedding checks
_rag_evaluator = None
_quality_evaluator = None

def get_rag_evaluator() -> RAGEvaluator:
    """Get or create RAG evaluator instance (lazy initialization)."""
    global _rag_evaluator
    if _rag_evaluator is None:
        _rag_evaluator = RAGEvaluator()
    return _rag_evaluator

def get_quality_evaluator() -> ResponseQualityEvaluator:
    """Get or create quality evaluator instance (lazy initialization)."""
    global _quality_evaluator
    if _quality_evaluator is None:
        _quality_evaluator = ResponseQualityEvaluator()
    return _quality_evaluator


# Rate limiting dependency for evaluation endpoints
async def check_evaluation_rate_limit(
    request: Request,
    rate_limiter: RateLimiter = Depends(get_rate_limiter_dep)
):
    """
    Check rate limit for evaluation endpoints.
    
    Evaluation operations are expensive, so we apply stricter limits:
    - Standard evaluations: 10 requests per minute
    - Batch evaluations: 5 requests per minute
    
    Args:
        request: FastAPI request object
        rate_limiter: Rate limiter instance
        
    Raises:
        HTTPException: If rate limit exceeded
    """
    # Get client IP
    client_ip = request.client.host if request.client else "unknown"
    
    # Determine endpoint type from path
    path = request.url.path
    if "batch" in path:
        # Stricter limit for batch operations
        limit = 5
        endpoint_type = "eval_batch"
    else:
        # Standard limit for single evaluations
        limit = 10
        endpoint_type = "eval_standard"
    
    # Check rate limit
    allowed, metadata = await rate_limiter.check_rate_limit(
        client_ip,
        endpoint_type,
        limit=limit,
        window_minutes=1
    )
    retry_after = metadata.get("retry_after", 60)
    
    if not allowed:
        logger.warning(f"Rate limit exceeded for {client_ip} on {endpoint_type}")
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail=f"Rate limit exceeded. Please retry after {retry_after} seconds",
            headers={"Retry-After": str(retry_after)}
        )


@router.post("/geval", response_model=GEvalResponse, dependencies=[Depends(check_evaluation_rate_limit)])
@track_request_metrics("/evaluations/geval")
@track_evaluation_metrics("geval")
async def evaluate_summary_geval(
    request: GEvalRequest,
    user_id: str = Depends(get_rate_limiter_dep)  # Get user ID from auth
):
    """
    Evaluate a summary using G-Eval metrics.
    
    G-Eval evaluates summaries on:
    - Fluency: Grammar, spelling, punctuation, word choice
    - Consistency: Factual alignment with source
    - Relevance: Selection of important content
    - Coherence: Overall structure and organization
    """
    try:
        start_time = time.time()
        evaluation_id = f"geval_{int(time.time())}_{user_id[:8]}"
        
        # Check per-user rate limits
        allowed, rate_metadata = await user_rate_limiter.check_rate_limit(
            user_id=user_id,
            endpoint="/api/v1/evaluations/geval",
            tokens_requested=len(request.source_text) + len(request.summary),
            estimated_cost=0.01  # Estimate based on model
        )
        
        if not allowed:
            if advanced_metrics.enabled:
                advanced_metrics.track_rate_limit_hit("free", "minute")
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail=rate_metadata,
                headers=rate_metadata.get("headers", {})
            )
        
        # Send webhook for evaluation started
        asyncio.create_task(webhook_manager.send_webhook(
            user_id=user_id,
            event=WebhookEvent.EVALUATION_STARTED,
            evaluation_id=evaluation_id,
            data={"evaluation_type": "geval", "api_name": request.api_name}
        ))
        
        # Run G-Eval with SLI tracking
        if advanced_metrics.enabled:
            with advanced_metrics.track_sli_request("/api/v1/evaluations/geval"):
                result = await asyncio.to_thread(
                    run_geval,
                    transcript=request.source_text,
                    summary=request.summary,
                    api_key=request.api_key,
                    api_name=request.api_name,
                    save=request.save_results
                )
        else:
            result = await asyncio.to_thread(
                run_geval,
                transcript=request.source_text,
                summary=request.summary,
                api_key=request.api_key,
                api_name=request.api_name,
                save=request.save_results
            )
        
        # Parse the formatted result back into structured data
        # The run_geval function returns a formatted string, so we need to extract the scores
        scores = {}
        avg_scores = {}
        
        # Extract scores from the formatted result string
        import re
        
        # Extract individual scores
        for metric in ["coherence", "consistency", "fluency", "relevance"]:
            pattern = f"{metric.capitalize()}: (\\d+\\.\\d+)"
            match = re.search(pattern, result, re.IGNORECASE)
            if match:
                score_val = float(match.group(1))
                # Normalize scores (fluency is 1-3, others are 1-5)
                if metric == "fluency":
                    normalized = (score_val - 1) / 2
                else:
                    normalized = (score_val - 1) / 4
                    
                scores[metric] = EvaluationMetric(
                    name=metric,
                    score=normalized,
                    raw_score=score_val,
                    explanation=f"{metric.capitalize()} score based on G-Eval criteria"
                )
        
        # Extract average scores
        for metric in ["fluency", "consistency", "relevance", "coherence"]:
            pattern = f"{metric.capitalize()}: (\\d+\\.\\d+)"
            match = re.search(pattern, result.split("average scores are:")[1] if "average scores are:" in result else result, re.IGNORECASE)
            if match:
                avg_scores[f"average_{metric}"] = float(match.group(1))
        
        # Calculate overall average
        if scores:
            average_score = sum(m.score for m in scores.values()) / len(scores)
        else:
            average_score = 0.0
        
        evaluation_time = time.time() - start_time
        
        # Store evaluation result
        eval_id = await evaluation_manager.store_evaluation(
            evaluation_type="geval",
            input_data={
                "source_text": request.source_text,
                "summary": request.summary
            },
            results={
                "metrics": {k: v.dict() for k, v in scores.items()},
                "average_score": average_score
            },
            metadata={
                "api_name": request.api_name,
                "evaluation_time": evaluation_time,
                "user_id": user_id
            }
        )
        
        # Track advanced metrics
        if advanced_metrics.enabled:
            # Track evaluation quality
            advanced_metrics.track_evaluation_quality(
                evaluation_type="geval",
                model=request.api_name,
                accuracy=average_score,
                confidence=0.85  # Could be calculated from consistency
            )
            
            # Track cost
            advanced_metrics.track_evaluation_cost(
                user_tier="free",  # Would get from user profile
                provider=request.api_name.split("/")[0] if "/" in request.api_name else request.api_name,
                model=request.api_name,
                evaluation_type="geval",
                cost=0.01  # Actual cost calculation
            )
        
        # Send webhook for evaluation completed
        asyncio.create_task(webhook_manager.send_webhook(
            user_id=user_id,
            event=WebhookEvent.EVALUATION_COMPLETED,
            evaluation_id=evaluation_id,
            data={
                "evaluation_type": "geval",
                "scores": {k: v.dict() for k, v in scores.items()},
                "average_score": average_score,
                "processing_time": evaluation_time
            }
        ))
        
        return GEvalResponse(
            metrics=scores,
            average_score=average_score,
            summary_assessment=result.split("Overall Assessment:")[1].strip() if "Overall Assessment:" in result else "Evaluation complete",
            evaluation_time=evaluation_time,
            metadata={"evaluation_id": eval_id}
        )
        
    except Exception as e:
        logger.error(f"G-Eval evaluation failed: {e}")
        
        # Send webhook for evaluation failed
        if 'evaluation_id' in locals():
            asyncio.create_task(webhook_manager.send_webhook(
                user_id=user_id,
                event=WebhookEvent.EVALUATION_FAILED,
                evaluation_id=evaluation_id,
                data={
                    "evaluation_type": "geval",
                    "error": str(e)
                }
            ))
        
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Evaluation failed: {str(e)}"
        )


@router.post("/rag", response_model=RAGEvaluationResponse, dependencies=[Depends(check_evaluation_rate_limit)])
@track_request_metrics("/evaluations/rag")
@track_evaluation_metrics("rag")
async def evaluate_rag_system(
    request: RAGEvaluationRequest,
    user_id: str = Depends(get_rate_limiter_dep)  # Get user ID from auth
):
    """
    Evaluate RAG system performance.
    
    Evaluates:
    - Relevance: How relevant is the response to the query
    - Faithfulness: Is the response grounded in retrieved contexts
    - Answer Similarity: How similar is the response to ground truth (if provided)
    - Context Precision: Are retrieved contexts precise and relevant
    - Context Recall: Do contexts contain necessary information
    """
    try:
        start_time = time.time()
        evaluation_id = f"rag_{int(time.time())}_{user_id[:8]}"
        
        # Check per-user rate limits
        context_size = sum(len(c) for c in request.retrieved_contexts)
        allowed, rate_metadata = await user_rate_limiter.check_rate_limit(
            user_id=user_id,
            endpoint="/api/v1/evaluations/rag",
            tokens_requested=len(request.query) + context_size + len(request.generated_response),
            estimated_cost=0.02  # Estimate based on model
        )
        
        if not allowed:
            if advanced_metrics.enabled:
                advanced_metrics.track_rate_limit_hit("free", "minute")
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail=rate_metadata,
                headers=rate_metadata.get("headers", {})
            )
        
        # Send webhook for evaluation started
        asyncio.create_task(webhook_manager.send_webhook(
            user_id=user_id,
            event=WebhookEvent.EVALUATION_STARTED,
            evaluation_id=evaluation_id,
            data={"evaluation_type": "rag", "api_name": request.api_name}
        ))
        
        # Run RAG evaluation with SLI tracking
        if advanced_metrics.enabled:
            with advanced_metrics.track_sli_request("/api/v1/evaluations/rag"):
                results = await get_rag_evaluator().evaluate(
                    query=request.query,
                    contexts=request.retrieved_contexts,
                    response=request.generated_response,
                    ground_truth=request.ground_truth,
                    metrics=request.metrics,
                    api_name=request.api_name
                )
        else:
            results = await get_rag_evaluator().evaluate(
                query=request.query,
                contexts=request.retrieved_contexts,
                response=request.generated_response,
                ground_truth=request.ground_truth,
                metrics=request.metrics,
                api_name=request.api_name
            )
        
        # Calculate scores
        retrieval_metrics = ["context_precision", "context_recall"]
        generation_metrics = ["relevance", "faithfulness", "answer_similarity"]
        
        retrieval_scores = [
            results["metrics"][m].score 
            for m in retrieval_metrics 
            if m in results["metrics"]
        ]
        retrieval_quality = sum(retrieval_scores) / len(retrieval_scores) if retrieval_scores else 0.0
        
        generation_scores = [
            results["metrics"][m].score 
            for m in generation_metrics 
            if m in results["metrics"]
        ]
        generation_quality = sum(generation_scores) / len(generation_scores) if generation_scores else 0.0
        
        overall_score = (retrieval_quality + generation_quality) / 2
        
        evaluation_time = time.time() - start_time
        
        # Store evaluation with user ID
        eval_id = await evaluation_manager.store_evaluation(
            evaluation_type="rag",
            input_data=request.dict(),
            results=results,
            metadata={
                "evaluation_time": evaluation_time,
                "overall_score": overall_score,
                "user_id": user_id
            }
        )
        
        # Track advanced metrics
        if advanced_metrics.enabled:
            # Track evaluation quality
            advanced_metrics.track_evaluation_quality(
                evaluation_type="rag",
                model=request.api_name,
                accuracy=overall_score,
                confidence=0.85  # Could be calculated from metrics
            )
            
            # Track cost
            advanced_metrics.track_evaluation_cost(
                user_tier="free",  # Would get from user profile
                provider=request.api_name.split("/")[0] if "/" in request.api_name else request.api_name,
                model=request.api_name,
                evaluation_type="rag",
                cost=0.02  # Actual cost calculation
            )
        
        # Send webhook for evaluation completed
        asyncio.create_task(webhook_manager.send_webhook(
            user_id=user_id,
            event=WebhookEvent.EVALUATION_COMPLETED,
            evaluation_id=evaluation_id,
            data={
                "evaluation_type": "rag",
                "overall_score": overall_score,
                "retrieval_quality": retrieval_quality,
                "generation_quality": generation_quality,
                "processing_time": evaluation_time
            }
        ))
        
        return RAGEvaluationResponse(
            metrics=results["metrics"],
            overall_score=overall_score,
            retrieval_quality=retrieval_quality,
            generation_quality=generation_quality,
            suggestions=results.get("suggestions", []),
            metadata={"evaluation_id": eval_id}
        )
        
    except Exception as e:
        logger.error(f"RAG evaluation failed: {e}")
        
        # Send webhook for evaluation failed
        if 'evaluation_id' in locals():
            asyncio.create_task(webhook_manager.send_webhook(
                user_id=user_id,
                event=WebhookEvent.EVALUATION_FAILED,
                evaluation_id=evaluation_id,
                data={
                    "evaluation_type": "rag",
                    "error": str(e)
                }
            ))
        
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"RAG evaluation failed: {str(e)}"
        )


@router.post("/response-quality", response_model=ResponseQualityResponse, dependencies=[Depends(check_evaluation_rate_limit)])
async def evaluate_response_quality(
    request: ResponseQualityRequest,
    user_id: str = Depends(get_rate_limiter_dep)  # Get user ID from auth
):
    """
    Evaluate the quality of a generated response.
    
    Checks:
    - Relevance to prompt
    - Completeness
    - Accuracy
    - Format compliance
    - Custom criteria
    """
    try:
        start_time = time.time()
        evaluation_id = f"quality_{int(time.time())}_{user_id[:8]}"
        
        # Check per-user rate limits
        allowed, rate_metadata = await user_rate_limiter.check_rate_limit(
            user_id=user_id,
            endpoint="/api/v1/evaluations/response-quality",
            tokens_requested=len(request.prompt) + len(request.response),
            estimated_cost=0.015  # Estimate based on model
        )
        
        if not allowed:
            if advanced_metrics.enabled:
                advanced_metrics.track_rate_limit_hit("free", "minute")
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail=rate_metadata,
                headers=rate_metadata.get("headers", {})
            )
        
        # Send webhook for evaluation started
        asyncio.create_task(webhook_manager.send_webhook(
            user_id=user_id,
            event=WebhookEvent.EVALUATION_STARTED,
            evaluation_id=evaluation_id,
            data={"evaluation_type": "response_quality", "api_name": request.api_name}
        ))
        
        # Run quality evaluation with SLI tracking
        if advanced_metrics.enabled:
            with advanced_metrics.track_sli_request("/api/v1/evaluations/response-quality"):
                results = await get_quality_evaluator().evaluate(
                    prompt=request.prompt,
                    response=request.response,
                    expected_format=request.expected_format,
                    custom_criteria=request.evaluation_criteria,
                    api_name=request.api_name
                )
        else:
            results = await get_quality_evaluator().evaluate(
                prompt=request.prompt,
                response=request.response,
                expected_format=request.expected_format,
                custom_criteria=request.evaluation_criteria,
                api_name=request.api_name
            )
        
        evaluation_time = time.time() - start_time
        
        # Store evaluation with user ID
        eval_id = await evaluation_manager.store_evaluation(
            evaluation_type="response_quality",
            input_data=request.dict(),
            results=results,
            metadata={
                "evaluation_time": evaluation_time,
                "user_id": user_id
            }
        )
        
        # Track advanced metrics
        if advanced_metrics.enabled:
            # Track evaluation quality
            advanced_metrics.track_evaluation_quality(
                evaluation_type="response_quality",
                model=request.api_name,
                accuracy=results.get("overall_quality", 0.0),
                confidence=0.85
            )
            
            # Track cost
            advanced_metrics.track_evaluation_cost(
                user_tier="free",  # Would get from user profile
                provider=request.api_name.split("/")[0] if "/" in request.api_name else request.api_name,
                model=request.api_name,
                evaluation_type="response_quality",
                cost=0.015  # Actual cost calculation
            )
        
        # Send webhook for evaluation completed
        asyncio.create_task(webhook_manager.send_webhook(
            user_id=user_id,
            event=WebhookEvent.EVALUATION_COMPLETED,
            evaluation_id=evaluation_id,
            data={
                "evaluation_type": "response_quality",
                "overall_quality": results.get("overall_quality", 0.0),
                "format_compliance": results.get("format_compliance", {}),
                "processing_time": evaluation_time
            }
        ))
        
        return ResponseQualityResponse(
            metrics=results["metrics"],
            overall_quality=results["overall_quality"],
            format_compliance=results["format_compliance"],
            issues=results.get("issues", []),
            improvements=results.get("improvements", [])
        )
        
    except Exception as e:
        logger.error(f"Response quality evaluation failed: {e}")
        
        # Send webhook for evaluation failed
        if 'evaluation_id' in locals():
            asyncio.create_task(webhook_manager.send_webhook(
                user_id=user_id,
                event=WebhookEvent.EVALUATION_FAILED,
                evaluation_id=evaluation_id,
                data={
                    "evaluation_type": "response_quality",
                    "error": str(e)
                }
            ))
        
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Quality evaluation failed: {str(e)}"
        )


@router.post("/batch", response_model=BatchEvaluationResponse, dependencies=[Depends(check_evaluation_rate_limit)])
@track_request_metrics("/evaluations/batch")
@track_evaluation_metrics("batch")
async def evaluate_batch(
    request: BatchEvaluationRequest,
    user_id: str = Depends(get_rate_limiter_dep)  # Get user ID from auth
):
    """
    Perform batch evaluation of multiple items.
    
    Supports parallel processing for efficiency.
    """
    try:
        start_time = time.time()
        batch_id = f"batch_{int(time.time())}_{user_id[:8]}"
        
        # Check per-user rate limits for batch operations
        # Batch operations have stricter limits
        allowed, rate_metadata = await user_rate_limiter.check_rate_limit(
            user_id=user_id,
            endpoint="/api/v1/evaluations/batch",
            tokens_requested=len(request.items) * 1000,  # Estimate tokens per item
            estimated_cost=len(request.items) * 0.02  # Estimate cost per item
        )
        
        if not allowed:
            if advanced_metrics.enabled:
                advanced_metrics.track_rate_limit_hit("free", "batch")
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail=rate_metadata,
                headers=rate_metadata.get("headers", {})
            )
        
        # Send webhook for batch started
        asyncio.create_task(webhook_manager.send_webhook(
            user_id=user_id,
            event=WebhookEvent.BATCH_STARTED,
            evaluation_id=batch_id,
            data={
                "evaluation_type": request.evaluation_type,
                "item_count": len(request.items),
                "parallel_workers": request.parallel_workers
            }
        ))
        
        # Create evaluation tasks with user_id
        tasks = []
        for item in request.items:
            if request.evaluation_type == "geval":
                task = evaluate_summary_geval(GEvalRequest(**item), user_id)
            elif request.evaluation_type == "rag":
                task = evaluate_rag_system(RAGEvaluationRequest(**item), user_id)
            elif request.evaluation_type == "response_quality":
                task = evaluate_response_quality(ResponseQualityRequest(**item), user_id)
            else:
                raise ValueError(f"Unknown evaluation type: {request.evaluation_type}")
            
            tasks.append(task)
        
        # Run evaluations in parallel with worker limit
        results = []
        failed = 0
        
        # Process in batches based on parallel_workers
        for i in range(0, len(tasks), request.parallel_workers):
            batch = tasks[i:i + request.parallel_workers]
            batch_results = await asyncio.gather(*batch, return_exceptions=True)
            
            for result in batch_results:
                if isinstance(result, Exception):
                    failed += 1
                    results.append({"error": str(result)})
                else:
                    results.append(result.dict())
        
        # Calculate aggregate metrics
        aggregate_metrics = {}
        if request.evaluation_type == "geval":
            all_scores = [r["average_score"] for r in results if "average_score" in r]
            if all_scores:
                aggregate_metrics["average_score"] = sum(all_scores) / len(all_scores)
        
        processing_time = time.time() - start_time
        
        # Track advanced metrics for batch
        if advanced_metrics.enabled:
            # Track batch cost
            total_cost = len(request.items) * 0.02  # Actual cost calculation
            advanced_metrics.track_evaluation_cost(
                user_tier="free",  # Would get from user profile
                provider="batch",
                model=request.evaluation_type,
                evaluation_type="batch",
                cost=total_cost
            )
            
            # Track queue depth if applicable
            advanced_metrics.track_queue_depth("batch", 0)
        
        # Send webhook for batch completed
        asyncio.create_task(webhook_manager.send_webhook(
            user_id=user_id,
            event=WebhookEvent.BATCH_COMPLETED,
            evaluation_id=batch_id,
            data={
                "evaluation_type": request.evaluation_type,
                "total_items": len(request.items),
                "successful": len(request.items) - failed,
                "failed": failed,
                "aggregate_metrics": aggregate_metrics,
                "processing_time": processing_time
            }
        ))
        
        return BatchEvaluationResponse(
            total_items=len(request.items),
            successful=len(request.items) - failed,
            failed=failed,
            results=results,
            aggregate_metrics=aggregate_metrics,
            processing_time=processing_time
        )
        
    except Exception as e:
        logger.error(f"Batch evaluation failed: {e}")
        
        # Send webhook for batch failed
        if 'batch_id' in locals():
            asyncio.create_task(webhook_manager.send_webhook(
                user_id=user_id,
                event=WebhookEvent.BATCH_FAILED,
                evaluation_id=batch_id,
                data={
                    "evaluation_type": request.evaluation_type if 'request' in locals() else "unknown",
                    "error": str(e)
                }
            ))
        
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Batch evaluation failed: {str(e)}"
        )


@router.post("/history", response_model=EvaluationHistoryResponse, dependencies=[Depends(check_evaluation_rate_limit)])
async def get_evaluation_history(request: EvaluationHistoryRequest):
    """
    Retrieve evaluation history with filtering and aggregation.
    """
    try:
        history = await evaluation_manager.get_history(
            evaluation_type=request.evaluation_type,
            start_date=request.start_date,
            end_date=request.end_date,
            limit=request.limit,
            offset=request.offset
        )
        
        return EvaluationHistoryResponse(**history)
        
    except Exception as e:
        logger.error(f"Failed to retrieve evaluation history: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to retrieve history: {str(e)}"
        )


@router.post("/custom-metric", response_model=CustomMetricResponse, dependencies=[Depends(check_evaluation_rate_limit)])
async def evaluate_custom_metric(request: CustomMetricRequest):
    """
    Evaluate using a custom metric definition.
    
    Allows users to define their own evaluation criteria.
    """
    try:
        result = await evaluation_manager.evaluate_custom_metric(
            metric_name=request.name,
            description=request.description,
            evaluation_prompt=request.evaluation_prompt,
            input_data=request.input_data,
            scoring_criteria=request.scoring_criteria,
            api_name=request.api_name
        )
        
        return CustomMetricResponse(**result)
        
    except Exception as e:
        logger.error(f"Custom metric evaluation failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Custom evaluation failed: {str(e)}"
        )


@router.post("/compare", response_model=EvaluationComparisonResponse, dependencies=[Depends(check_evaluation_rate_limit)])
async def compare_evaluations(request: EvaluationComparisonRequest):
    """
    Compare multiple evaluations to identify improvements or regressions.
    """
    try:
        comparison = await evaluation_manager.compare_evaluations(
            evaluation_ids=request.evaluation_ids,
            metrics_to_compare=request.metrics_to_compare
        )
        
        return EvaluationComparisonResponse(**comparison)
        
    except Exception as e:
        logger.error(f"Evaluation comparison failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Comparison failed: {str(e)}"
        )


@router.get("/metrics", include_in_schema=False)
async def get_evaluation_metrics(request: Request):
    """
    Prometheus metrics endpoint for evaluation service.
    
    Returns metrics in Prometheus text format.
    This endpoint is not rate-limited to allow monitoring systems to scrape it.
    """
    metrics = get_metrics()
    metrics_output = metrics.get_metrics()
    
    return Response(
        content=metrics_output,
        media_type="text/plain; version=0.0.4; charset=utf-8"
    )


@router.get("/health/config")
async def check_configuration_health():
    """
    Check configuration health and production readiness.
    
    Returns detailed configuration validation results.
    """
    try:
        validation_results = validate_configuration()
        
        # Determine HTTP status based on validation
        if validation_results["status"] == "error":
            status_code = status.HTTP_500_INTERNAL_SERVER_ERROR
        elif validation_results["status"] == "warning":
            status_code = status.HTTP_200_OK
        else:
            status_code = status.HTTP_200_OK
        
        return Response(
            content=json.dumps(validation_results, indent=2),
            status_code=status_code,
            media_type="application/json"
        )
        
    except Exception as e:
        logger.error(f"Configuration health check failed: {e}")
        return Response(
            content=json.dumps({
                "status": "error",
                "message": f"Health check failed: {str(e)}"
            }),
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            media_type="application/json"
        )


# ============= Webhook Management Endpoints =============

@router.post("/webhooks", response_model=WebhookRegistrationResponse)
async def register_webhook(
    request: WebhookRegistrationRequest,
    user_id: str = Depends(get_rate_limiter_dep)  # Gets user_id from auth
):
    """
    Register a webhook for evaluation notifications.
    
    Webhooks will receive POST requests with evaluation events.
    The payload is signed with HMAC-SHA256 using the provided secret.
    """
    try:
        # Convert string events to enum
        events = [WebhookEvent(e.value) for e in request.events]
        
        result = await webhook_manager.register_webhook(
            user_id=user_id,
            url=str(request.url),
            events=events,
            secret=request.secret
        )
        
        # Track in metrics
        if advanced_metrics.enabled:
            advanced_metrics.feature_adoption.labels(
                feature_name="webhooks",
                user_tier="unknown"  # Would get from user profile
            ).set(1.0)
        
        return WebhookRegistrationResponse(**result)
        
    except Exception as e:
        logger.error(f"Failed to register webhook: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to register webhook: {str(e)}"
        )


@router.get("/webhooks", response_model=List[WebhookStatusResponse])
async def list_webhooks(
    user_id: str = Depends(get_rate_limiter_dep)
):
    """
    List all registered webhooks for the authenticated user.
    """
    try:
        webhooks = await webhook_manager.get_webhook_status(user_id)
        return [WebhookStatusResponse(**w) for w in webhooks]
        
    except Exception as e:
        logger.error(f"Failed to list webhooks: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to list webhooks: {str(e)}"
        )


@router.delete("/webhooks")
async def unregister_webhook(
    url: str = Query(..., description="Webhook URL to unregister"),
    user_id: str = Depends(get_rate_limiter_dep)
):
    """
    Unregister a webhook.
    """
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
            detail=f"Failed to unregister webhook: {str(e)}"
        )


@router.post("/webhooks/test", response_model=WebhookTestResponse)
async def test_webhook(
    request: WebhookTestRequest,
    user_id: str = Depends(get_rate_limiter_dep)
):
    """
    Send a test webhook to verify endpoint configuration.
    
    This will send a test payload to the webhook URL with a sample event.
    """
    try:
        result = await webhook_manager.test_webhook(user_id, str(request.url))
        return WebhookTestResponse(**result)
        
    except Exception as e:
        logger.error(f"Failed to test webhook: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to test webhook: {str(e)}"
        )


# ============= Rate Limit Management Endpoints =============

@router.get("/rate-limits", response_model=RateLimitStatusResponse)
async def get_rate_limit_status(
    user_id: str = Depends(get_rate_limiter_dep)
):
    """
    Get current rate limit status for the authenticated user.
    
    Shows current tier, limits, usage, and remaining allowance.
    """
    try:
        summary = await user_rate_limiter.get_usage_summary(user_id)
        return RateLimitStatusResponse(**summary)
        
    except Exception as e:
        logger.error(f"Failed to get rate limit status: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get rate limit status: {str(e)}"
        )


@router.post("/rate-limits/upgrade")
async def upgrade_user_tier(
    tier: str = Query(..., description="Target tier: basic, premium, or enterprise"),
    user_id: str = Depends(get_rate_limiter_dep)
):
    """
    Upgrade user to a higher tier (admin only in production).
    
    This endpoint is for testing/demo purposes. In production,
    tier upgrades would be handled through billing/subscription system.
    """
    try:
        # In production, verify admin permissions here
        # For now, allow for testing
        
        tier_enum = UserTier(tier.lower())
        success = await user_rate_limiter.upgrade_user_tier(user_id, tier_enum)
        
        if not success:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to upgrade tier"
            )
        
        return {
            "message": f"Successfully upgraded to {tier} tier",
            "tier": tier,
            "user_id": user_id
        }
        
    except ValueError:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid tier. Choose from: basic, premium, enterprise"
        )
    except Exception as e:
        logger.error(f"Failed to upgrade tier: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to upgrade tier: {str(e)}"
        )