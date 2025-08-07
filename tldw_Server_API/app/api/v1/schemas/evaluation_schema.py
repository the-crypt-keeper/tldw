# evaluation_schema.py - Pydantic schemas for evaluation endpoints
"""
Evaluation system schemas for API requests and responses.

Supports multiple evaluation types:
- G-Eval for summarization quality
- RAG evaluation for retrieval quality
- Response quality metrics
- Custom evaluation metrics
"""

from typing import Dict, List, Optional, Any, Literal
from pydantic import BaseModel, Field, field_validator
from datetime import datetime


class EvaluationMetric(BaseModel):
    """Base evaluation metric"""
    name: str = Field(..., description="Metric name")
    score: float = Field(..., ge=0.0, le=1.0, description="Normalized score between 0 and 1")
    raw_score: Optional[float] = Field(None, description="Raw score before normalization")
    explanation: Optional[str] = Field(None, description="Explanation of the score")
    metadata: Optional[Dict[str, Any]] = Field(default_factory=dict)


class GEvalRequest(BaseModel):
    """Request for G-Eval summarization evaluation"""
    source_text: str = Field(..., description="Original source text")
    summary: str = Field(..., description="Summary to evaluate")
    metrics: Optional[List[Literal["fluency", "consistency", "relevance", "coherence"]]] = Field(
        default=["fluency", "consistency", "relevance", "coherence"],
        description="Metrics to evaluate"
    )
    api_name: Optional[str] = Field("openai", description="LLM API to use for evaluation")
    api_key: Optional[str] = Field(None, description="API key (if not in config)")
    save_results: bool = Field(False, description="Save results to file")


class GEvalResponse(BaseModel):
    """Response from G-Eval evaluation"""
    metrics: Dict[str, EvaluationMetric] = Field(..., description="Individual metric scores")
    average_score: float = Field(..., description="Average of all metrics")
    summary_assessment: str = Field(..., description="Overall assessment of the summary")
    evaluation_time: float = Field(..., description="Time taken for evaluation in seconds")
    metadata: Optional[Dict[str, Any]] = Field(default_factory=dict)


class RAGEvaluationRequest(BaseModel):
    """Request for RAG system evaluation"""
    query: str = Field(..., description="User query")
    retrieved_contexts: List[str] = Field(..., description="Retrieved context chunks")
    generated_response: str = Field(..., description="Generated response")
    ground_truth: Optional[str] = Field(None, description="Ground truth answer if available")
    metrics: Optional[List[Literal["relevance", "faithfulness", "answer_similarity", "context_precision", "context_recall"]]] = Field(
        default=["relevance", "faithfulness", "answer_similarity"],
        description="Metrics to evaluate"
    )
    api_name: Optional[str] = Field("openai", description="LLM API to use for evaluation")


class RAGEvaluationResponse(BaseModel):
    """Response from RAG evaluation"""
    metrics: Dict[str, EvaluationMetric] = Field(..., description="Individual metric scores")
    overall_score: float = Field(..., description="Overall RAG quality score")
    retrieval_quality: float = Field(..., description="Quality of retrieved contexts")
    generation_quality: float = Field(..., description="Quality of generated response")
    suggestions: List[str] = Field(default_factory=list, description="Improvement suggestions")
    metadata: Optional[Dict[str, Any]] = Field(default_factory=dict)


class ResponseQualityRequest(BaseModel):
    """Request for response quality evaluation"""
    prompt: str = Field(..., description="Original prompt")
    response: str = Field(..., description="Generated response")
    expected_format: Optional[str] = Field(None, description="Expected response format")
    evaluation_criteria: Optional[Dict[str, str]] = Field(
        default_factory=dict,
        description="Custom evaluation criteria"
    )
    api_name: Optional[str] = Field("openai", description="LLM API to use for evaluation")


class ResponseQualityResponse(BaseModel):
    """Response from quality evaluation"""
    metrics: Dict[str, EvaluationMetric] = Field(..., description="Quality metrics")
    overall_quality: float = Field(..., description="Overall quality score")
    format_compliance: bool = Field(..., description="Whether response matches expected format")
    issues: List[str] = Field(default_factory=list, description="Identified issues")
    improvements: List[str] = Field(default_factory=list, description="Suggested improvements")


class BatchEvaluationRequest(BaseModel):
    """Request for batch evaluation"""
    evaluation_type: Literal["geval", "rag", "response_quality"] = Field(..., description="Type of evaluation")
    items: List[Dict[str, Any]] = Field(..., description="Items to evaluate")
    metrics: Optional[List[str]] = Field(None, description="Metrics to compute")
    api_name: Optional[str] = Field("openai", description="LLM API to use")
    parallel_workers: int = Field(4, ge=1, le=16, description="Number of parallel workers")


class BatchEvaluationResponse(BaseModel):
    """Response from batch evaluation"""
    total_items: int = Field(..., description="Total items evaluated")
    successful: int = Field(..., description="Successfully evaluated items")
    failed: int = Field(..., description="Failed evaluations")
    results: List[Dict[str, Any]] = Field(..., description="Individual evaluation results")
    aggregate_metrics: Dict[str, float] = Field(..., description="Aggregated metrics")
    processing_time: float = Field(..., description="Total processing time in seconds")


class EvaluationHistoryRequest(BaseModel):
    """Request for evaluation history"""
    evaluation_type: Optional[Literal["geval", "rag", "response_quality", "all"]] = Field(
        "all", description="Filter by evaluation type"
    )
    start_date: Optional[datetime] = Field(None, description="Start date filter")
    end_date: Optional[datetime] = Field(None, description="End date filter")
    limit: int = Field(50, ge=1, le=1000, description="Maximum results to return")
    offset: int = Field(0, ge=0, description="Offset for pagination")


class EvaluationHistoryResponse(BaseModel):
    """Response with evaluation history"""
    total_count: int = Field(..., description="Total evaluations matching criteria")
    items: List[Dict[str, Any]] = Field(..., description="Evaluation records")
    average_scores: Dict[str, float] = Field(..., description="Average scores by metric")
    trends: Optional[Dict[str, Any]] = Field(None, description="Score trends over time")


class CustomMetricRequest(BaseModel):
    """Request for custom metric evaluation"""
    name: str = Field(..., description="Metric name")
    description: str = Field(..., description="Metric description")
    evaluation_prompt: str = Field(..., description="Prompt template for evaluation")
    input_data: Dict[str, Any] = Field(..., description="Data to evaluate")
    scoring_criteria: Dict[str, Any] = Field(..., description="Scoring criteria")
    api_name: Optional[str] = Field("openai", description="LLM API to use")


class CustomMetricResponse(BaseModel):
    """Response from custom metric evaluation"""
    metric_name: str = Field(..., description="Custom metric name")
    score: float = Field(..., description="Computed score")
    explanation: str = Field(..., description="Score explanation")
    raw_output: Optional[str] = Field(None, description="Raw LLM output")
    metadata: Dict[str, Any] = Field(default_factory=dict)


class EvaluationComparisonRequest(BaseModel):
    """Request to compare multiple evaluations"""
    evaluation_ids: List[str] = Field(..., min_items=2, description="Evaluation IDs to compare")
    metrics_to_compare: Optional[List[str]] = Field(None, description="Specific metrics to compare")


class EvaluationComparisonResponse(BaseModel):
    """Response comparing evaluations"""
    comparison_summary: str = Field(..., description="Summary of comparison")
    metric_comparisons: Dict[str, List[float]] = Field(..., description="Metric values for each evaluation")
    best_performing: Dict[str, str] = Field(..., description="Best performing evaluation for each metric")
    statistical_analysis: Optional[Dict[str, Any]] = Field(None, description="Statistical analysis if applicable")