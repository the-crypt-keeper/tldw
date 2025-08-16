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

# Import rate limiting and auth dependencies
from tldw_Server_API.app.core.AuthNZ.rate_limiter import RateLimiter, get_rate_limiter
from tldw_Server_API.app.api.v1.API_Deps.auth_deps import get_rate_limiter_dep

# Create router
router = APIRouter(prefix="/evaluations", tags=["Evaluations"])

# Initialize evaluation components
evaluation_manager = EvaluationManager()
rag_evaluator = RAGEvaluator()
quality_evaluator = ResponseQualityEvaluator()


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
async def evaluate_summary_geval(request: GEvalRequest):
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
        
        # Run G-Eval
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
                "evaluation_time": evaluation_time
            }
        )
        
        return GEvalResponse(
            metrics=scores,
            average_score=average_score,
            summary_assessment=result.split("Overall Assessment:")[1].strip() if "Overall Assessment:" in result else "Evaluation complete",
            evaluation_time=evaluation_time,
            metadata={"evaluation_id": eval_id}
        )
        
    except Exception as e:
        logger.error(f"G-Eval evaluation failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Evaluation failed: {str(e)}"
        )


@router.post("/rag", response_model=RAGEvaluationResponse, dependencies=[Depends(check_evaluation_rate_limit)])
@track_request_metrics("/evaluations/rag")
@track_evaluation_metrics("rag")
async def evaluate_rag_system(request: RAGEvaluationRequest):
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
        
        # Run RAG evaluation
        results = await rag_evaluator.evaluate(
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
        
        # Store evaluation
        eval_id = await evaluation_manager.store_evaluation(
            evaluation_type="rag",
            input_data=request.dict(),
            results=results,
            metadata={
                "evaluation_time": evaluation_time,
                "overall_score": overall_score
            }
        )
        
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
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"RAG evaluation failed: {str(e)}"
        )


@router.post("/response-quality", response_model=ResponseQualityResponse, dependencies=[Depends(check_evaluation_rate_limit)])
async def evaluate_response_quality(request: ResponseQualityRequest):
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
        
        # Run quality evaluation
        results = await quality_evaluator.evaluate(
            prompt=request.prompt,
            response=request.response,
            expected_format=request.expected_format,
            custom_criteria=request.evaluation_criteria,
            api_name=request.api_name
        )
        
        evaluation_time = time.time() - start_time
        
        # Store evaluation
        eval_id = await evaluation_manager.store_evaluation(
            evaluation_type="response_quality",
            input_data=request.dict(),
            results=results,
            metadata={"evaluation_time": evaluation_time}
        )
        
        return ResponseQualityResponse(
            metrics=results["metrics"],
            overall_quality=results["overall_quality"],
            format_compliance=results["format_compliance"],
            issues=results.get("issues", []),
            improvements=results.get("improvements", [])
        )
        
    except Exception as e:
        logger.error(f"Response quality evaluation failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Quality evaluation failed: {str(e)}"
        )


@router.post("/batch", response_model=BatchEvaluationResponse, dependencies=[Depends(check_evaluation_rate_limit)])
@track_request_metrics("/evaluations/batch")
@track_evaluation_metrics("batch")
async def evaluate_batch(request: BatchEvaluationRequest):
    """
    Perform batch evaluation of multiple items.
    
    Supports parallel processing for efficiency.
    """
    try:
        start_time = time.time()
        
        # Create evaluation tasks
        tasks = []
        for item in request.items:
            if request.evaluation_type == "geval":
                task = evaluate_summary_geval(GEvalRequest(**item))
            elif request.evaluation_type == "rag":
                task = evaluate_rag_system(RAGEvaluationRequest(**item))
            elif request.evaluation_type == "response_quality":
                task = evaluate_response_quality(ResponseQualityRequest(**item))
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