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
from pydantic import BaseModel, Field, field_validator, model_validator
from datetime import datetime
import re
import html

try:
    import bleach  # Robust HTML sanitizer
except Exception:  # pragma: no cover - optional dependency fallback
    bleach = None


def sanitize_html_text(value: Optional[str]) -> Optional[str]:
    """Sanitize user-provided text using a robust HTML sanitizer.
    - Uses bleach to strip all tags, attributes, protocols, and comments when available.
    - Falls back to conservative regex + html.escape if bleach is unavailable.
    - Also removes null bytes/control characters and normalizes whitespace.
    """
    if value is None:
        return None

    v = value

    # Normalize line endings first
    v = v.replace("\r", "\n")

    if bleach is not None:
        # Strip all HTML tags and comments, keep only text content
        v = bleach.clean(
            v,
            tags=[],
            attributes={},
            protocols=[],
            strip=True,
            strip_comments=True,
        )
    else:
        # Fallback: remove dangerous tags and all HTML, then escape
        # Strategy: Remove script/style content blocks, then remove all HTML tags
        
        # Remove script tags with their content - handle multiple cases:
        # 1. Well-formed script tags with closing tag
        v = re.sub(r'<\s*script(?:\s+[^>]*)?>.*?<\s*/\s*script\s*>', '', v, 
                   flags=re.IGNORECASE | re.DOTALL)
        
        # 2. Style tags with their content
        v = re.sub(r'<\s*style(?:\s+[^>]*)?>.*?<\s*/\s*style\s*>', '', v, 
                   flags=re.IGNORECASE | re.DOTALL)
        
        # 3. For orphaned opening script/style tags (no closing), we can't just remove
        # everything after them, so we'll remove the tag itself and let content remain
        # This is safer than potentially removing legitimate content
        
        # 4. Remove all HTML tags (including any remaining script/style tags)
        # This catches malformed tags, orphaned tags, and all other HTML
        v = re.sub(r'<[^>]*>', '', v)
        
        # 5. Escape the result for safe output
        v = html.escape(v)

    # Remove null bytes and most control characters except \n and \t
    v = v.replace('\x00', '')
    v = ''.join(ch for ch in v if ord(ch) >= 32 or ch in ('\n', '\t'))

    # Collapse excessive whitespace
    v = re.sub(r'\n{3,}', '\n\n', v)
    v = re.sub(r' {2,}', ' ', v)

    return v.strip() if v is not None else None


class EvaluationMetric(BaseModel):
    """Base evaluation metric"""
    name: str = Field(..., description="Metric name")
    score: float = Field(..., ge=0.0, le=1.0, description="Normalized score between 0 and 1")
    raw_score: Optional[float] = Field(None, description="Raw score before normalization")
    explanation: Optional[str] = Field(None, description="Explanation of the score")
    metadata: Optional[Dict[str, Any]] = Field(default_factory=dict)


class GEvalRequest(BaseModel):
    """Request for G-Eval summarization evaluation"""
    source_text: str = Field(
        ..., 
        description="Original source text",
        min_length=10,
        max_length=100000  # ~100KB limit
    )
    summary: str = Field(
        ..., 
        description="Summary to evaluate",
        min_length=10,
        max_length=50000  # ~50KB limit
    )
    metrics: Optional[List[Literal["fluency", "consistency", "relevance", "coherence"]]] = Field(
        default=["fluency", "consistency", "relevance", "coherence"],
        description="Metrics to evaluate"
    )
    api_name: Optional[str] = Field("openai", description="LLM API to use for evaluation")
    api_key: Optional[str] = Field(None, description="API key (if not in config)")
    save_results: bool = Field(False, description="Save results to file")
    
    @field_validator('source_text', 'summary')
    @classmethod
    def sanitize_text(cls, v: str) -> str:
        """Sanitize input text to prevent injection attacks"""
        if not v:
            raise ValueError("Text cannot be empty")
        sanitized = sanitize_html_text(v)
        if not sanitized:
            raise ValueError("Text cannot be empty after sanitization")
        return sanitized
    
    @field_validator('api_name')
    @classmethod
    def validate_api_name(cls, v: str) -> str:
        """Validate API name against allowed providers"""
        allowed_providers = [
            "openai", "anthropic", "google", "cohere", "mistral", 
            "groq", "openrouter", "deepseek", "local-llm"
        ]
        if v and v.lower() not in allowed_providers:
            raise ValueError(f"API provider '{v}' not supported. Must be one of: {', '.join(allowed_providers)}")
        return v.lower() if v else "openai"


class GEvalResponse(BaseModel):
    """Response from G-Eval evaluation"""
    metrics: Dict[str, EvaluationMetric] = Field(..., description="Individual metric scores")
    average_score: float = Field(..., description="Average of all metrics")
    summary_assessment: str = Field(..., description="Overall assessment of the summary")
    evaluation_time: float = Field(..., description="Time taken for evaluation in seconds")
    metadata: Optional[Dict[str, Any]] = Field(default_factory=dict)


class RAGEvaluationRequest(BaseModel):
    """Request for RAG system evaluation"""
    query: str = Field(
        ...,
        description="User query",
        min_length=3,
        max_length=5000
    )
    retrieved_contexts: List[str] = Field(
        ...,
        description="Retrieved context chunks",
        min_items=1,
        max_items=20  # Limit number of contexts
    )
    generated_response: str = Field(
        ...,
        description="Generated response",
        min_length=1,
        max_length=50000
    )
    ground_truth: Optional[str] = Field(
        None,
        description="Ground truth answer if available",
        max_length=50000
    )
    metrics: Optional[List[Literal["relevance", "faithfulness", "answer_similarity", "context_precision", "context_recall"]]] = Field(
        default=["relevance", "faithfulness", "answer_similarity"],
        description="Metrics to evaluate"
    )
    api_name: Optional[str] = Field("openai", description="LLM API to use for evaluation")
    
    @field_validator('query', 'generated_response', 'ground_truth')
    @classmethod
    def sanitize_text(cls, v: Optional[str]) -> Optional[str]:
        """Sanitize input text"""
        if v is None:
            return None
        return sanitize_html_text(v)
    
    @field_validator('retrieved_contexts')
    @classmethod
    def validate_contexts(cls, v: List[str]) -> List[str]:
        """Validate and sanitize context chunks"""
        if not v:
            raise ValueError("At least one context is required")
        
        sanitized = []
        for context in v:
            if len(context) > 20000:  # 20KB per context limit
                raise ValueError(f"Context too large: {len(context)} characters (max 20000)")
            cleaned = sanitize_html_text(context)
            if cleaned:
                sanitized.append(cleaned)
        
        if not sanitized:
            raise ValueError("No valid contexts after sanitization")
            
        return sanitized


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
    prompt: str = Field(
        ...,
        description="Original prompt",
        min_length=1,
        max_length=10000
    )
    response: str = Field(
        ...,
        description="Generated response",
        min_length=1,
        max_length=50000
    )
    expected_format: Optional[str] = Field(
        None,
        description="Expected response format",
        max_length=1000
    )
    evaluation_criteria: Optional[Dict[str, str]] = Field(
        default_factory=dict,
        description="Custom evaluation criteria"
    )
    api_name: Optional[str] = Field("openai", description="LLM API to use for evaluation")
    
    @field_validator('prompt', 'response', 'expected_format')
    @classmethod
    def sanitize_text(cls, v: Optional[str]) -> Optional[str]:
        """Sanitize input text"""
        if v is None:
            return None
        return sanitize_html_text(v)
    
    @field_validator('evaluation_criteria')
    @classmethod
    def validate_criteria(cls, v: Dict[str, str]) -> Dict[str, str]:
        """Validate evaluation criteria"""
        if len(v) > 20:
            raise ValueError("Too many evaluation criteria (max 20)")
            
        sanitized = {}
        for key, value in v.items():
            # Sanitize both keys and values
            key = re.sub(r'[^a-zA-Z0-9_-]', '', key)[:50]  # Limit key length
            value_str = sanitize_html_text(str(value)) or ''
            value_str = value_str[:500]  # Limit value length
            
            if key and value_str:
                sanitized[key] = value_str
                
        return sanitized


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
    items: List[Dict[str, Any]] = Field(
        ...,
        description="Items to evaluate",
        min_items=1,
        max_items=100  # Limit batch size
    )
    metrics: Optional[List[str]] = Field(None, description="Metrics to compute")
    api_name: Optional[str] = Field("openai", description="LLM API to use")
    parallel_workers: int = Field(4, ge=1, le=16, description="Number of parallel workers")
    
    @model_validator(mode='after')
    def validate_batch_size(self) -> 'BatchEvaluationRequest':
        """Validate total batch size doesn't exceed limits"""
        # Calculate approximate total size
        total_size = 0
        for item in self.items:
            # Estimate size of each item
            item_str = str(item)
            total_size += len(item_str)
            
        # 10MB total limit for batch
        if total_size > 10 * 1024 * 1024:
            raise ValueError(f"Batch too large: ~{total_size / (1024*1024):.1f}MB (max 10MB)")
            
        return self


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