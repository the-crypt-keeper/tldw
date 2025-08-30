# evaluation_schemas_unified.py - Unified Pydantic schemas for evaluation endpoints
"""
Unified evaluation schemas combining OpenAI-compatible and tldw-specific schemas.

This module provides all request/response models for the unified evaluation API.
"""

from typing import Dict, List, Optional, Any, Literal, Union
from pydantic import BaseModel, Field, field_validator, model_validator, HttpUrl
from datetime import datetime
from enum import Enum
import re
import html

try:
    import bleach
except Exception:
    bleach = None


# ============= Utility Functions =============

def sanitize_html_text(value: Optional[str]) -> Optional[str]:
    """Sanitize user-provided text to prevent injection attacks"""
    if value is None:
        return None

    v = value
    v = v.replace("\r", "\n")

    if bleach is not None:
        # Use bleach for proper HTML sanitization
        v = bleach.clean(
            v,
            tags=[],
            attributes={},
            protocols=[],
            strip=True,
            strip_comments=True,
        )
    else:
        # Fallback: More robust sanitization without bleach
        # First, decode any HTML entities to catch encoded attacks
        try:
            import html as html_module
            # Decode HTML entities multiple times to handle nested encoding
            for _ in range(3):
                decoded = html_module.unescape(v)
                if decoded == v:
                    break
                v = decoded
        except Exception:
            pass
        
        # Remove all HTML tags and dangerous patterns more thoroughly
        # Remove script tags and their content (case insensitive, handles broken tags)
        v = re.sub(r'<\s*script[^>]*>.*?<\s*/\s*script\s*>', '', v, flags=re.IGNORECASE | re.DOTALL)
        v = re.sub(r'<\s*script[^>]*>', '', v, flags=re.IGNORECASE)
        
        # Remove style tags and their content
        v = re.sub(r'<\s*style[^>]*>.*?<\s*/\s*style\s*>', '', v, flags=re.IGNORECASE | re.DOTALL)
        v = re.sub(r'<\s*style[^>]*>', '', v, flags=re.IGNORECASE)
        
        # Remove all event handlers (onclick, onload, etc.)
        v = re.sub(r'\s*on\w+\s*=\s*["\']?[^"\'>\s]*["\']?', '', v, flags=re.IGNORECASE)
        
        # Remove javascript: protocol
        v = re.sub(r'javascript\s*:', '', v, flags=re.IGNORECASE)
        v = re.sub(r'data\s*:', '', v, flags=re.IGNORECASE)
        v = re.sub(r'vbscript\s*:', '', v, flags=re.IGNORECASE)
        
        # Remove all remaining HTML tags
        v = re.sub(r'<[^>]+>', '', v)
        
        # Finally, escape any remaining HTML special characters
        v = html.escape(v)

    # Remove null bytes and control characters
    v = v.replace('\x00', '')
    v = ''.join(ch for ch in v if ord(ch) >= 32 or ch in ('\n', '\t'))
    
    # Normalize whitespace
    v = re.sub(r'\n{3,}', '\n\n', v)
    v = re.sub(r' {2,}', ' ', v)

    return v.strip() if v is not None else None


# ============= Enums =============

class EvaluationType(str, Enum):
    """Supported evaluation types"""
    MODEL_GRADED = "model_graded"
    EXACT_MATCH = "exact_match"
    INCLUDES = "includes"
    FUZZY_MATCH = "fuzzy_match"
    GEVAL = "geval"
    RAG = "rag"
    RESPONSE_QUALITY = "response_quality"
    CUSTOM = "custom"


class RunStatus(str, Enum):
    """Run status values"""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class WebhookEventType(str, Enum):
    """Webhook event types"""
    EVALUATION_STARTED = "evaluation.started"
    EVALUATION_PROGRESS = "evaluation.progress"
    EVALUATION_COMPLETED = "evaluation.completed"
    EVALUATION_FAILED = "evaluation.failed"
    EVALUATION_CANCELLED = "evaluation.cancelled"
    BATCH_STARTED = "batch.started"
    BATCH_COMPLETED = "batch.completed"
    BATCH_FAILED = "batch.failed"


# ============= Base Models =============

class EvaluationMetric(BaseModel):
    """Base evaluation metric"""
    name: str = Field(..., description="Metric name")
    score: float = Field(..., ge=0.0, le=1.0, description="Normalized score (0-1)")
    raw_score: Optional[float] = Field(None, description="Raw score before normalization")
    explanation: Optional[str] = Field(None, description="Explanation of the score")
    metadata: Optional[Dict[str, Any]] = Field(default_factory=dict)


class EvaluationSpec(BaseModel):
    """Evaluation specification"""
    metrics: Optional[List[str]] = Field(
        default=None,
        description="Metrics to compute"
    )
    thresholds: Optional[Dict[str, float]] = Field(
        default=None,
        description="Pass/fail thresholds"
    )
    model: Optional[str] = Field(
        default="gpt-3.5-turbo",
        description="Model to use for evaluation"
    )
    temperature: Optional[float] = Field(
        default=0.3,
        ge=0.0,
        le=2.0,
        description="Temperature for model-based evaluation"
    )
    custom_prompts: Optional[Dict[str, str]] = Field(
        default=None,
        description="Custom evaluation prompts"
    )


class RunConfig(BaseModel):
    """Run configuration"""
    temperature: Optional[float] = Field(0.7, ge=0.0, le=2.0)
    max_workers: Optional[int] = Field(4, ge=1, le=100)
    timeout_seconds: Optional[int] = Field(300, ge=1, le=3600)
    retry_attempts: Optional[int] = Field(3, ge=0, le=10)
    batch_size: Optional[int] = Field(10, ge=1, le=100)


class DatasetSample(BaseModel):
    """Dataset sample"""
    input: Any = Field(..., description="Input data")
    expected: Optional[Any] = Field(None, description="Expected output")
    metadata: Optional[Dict[str, Any]] = Field(default_factory=dict)


class EvaluationMetadata(BaseModel):
    """Evaluation metadata"""
    project: Optional[str] = Field(None, description="Project name")
    version: Optional[str] = Field(None, description="Version")
    tags: Optional[List[str]] = Field(default_factory=list)
    custom: Optional[Dict[str, Any]] = Field(default_factory=dict)


# ============= OpenAI-Compatible Schemas =============

class CreateEvaluationRequest(BaseModel):
    """Create evaluation request (OpenAI-compatible)"""
    name: str = Field(..., description="Unique evaluation name", min_length=1, max_length=255)
    description: Optional[str] = Field(None, max_length=1000)
    eval_type: EvaluationType = Field(..., description="Type of evaluation")
    eval_spec: EvaluationSpec = Field(..., description="Evaluation specification")
    dataset_id: Optional[str] = Field(None, description="Reference to existing dataset")
    dataset: Optional[List[DatasetSample]] = Field(None, description="Inline dataset")
    metadata: Optional[EvaluationMetadata] = Field(None)

    @field_validator('name')
    @classmethod
    def validate_name(cls, v: str) -> str:
        if not re.match(r'^[a-zA-Z0-9_-]+$', v):
            raise ValueError("Name must contain only alphanumeric characters, hyphens, and underscores")
        return v


class UpdateEvaluationRequest(BaseModel):
    """Update evaluation request"""
    description: Optional[str] = Field(None, max_length=1000)
    eval_spec: Optional[EvaluationSpec] = Field(None)
    metadata: Optional[EvaluationMetadata] = Field(None)


class EvaluationResponse(BaseModel):
    """Evaluation response (OpenAI-compatible)"""
    id: str = Field(..., description="Evaluation ID")
    object: str = Field(default="evaluation")
    name: str
    description: Optional[str] = None
    eval_type: str
    eval_spec: Dict[str, Any]
    dataset_id: Optional[str] = None
    created_at: int
    created_by: str
    metadata: Optional[Dict[str, Any]] = None


class CreateRunRequest(BaseModel):
    """Create run request"""
    target_model: str = Field(..., description="Model to evaluate")
    dataset_override: Optional[DatasetSample] = Field(None)
    config: Optional[RunConfig] = Field(default_factory=RunConfig)
    webhook_url: Optional[HttpUrl] = Field(None)


class RunProgress(BaseModel):
    """Run progress information"""
    completed_samples: int = Field(0, ge=0)
    total_samples: int = Field(0, ge=0)
    percent_complete: float = Field(0.0, ge=0.0, le=100.0)
    current_sample: Optional[int] = None
    estimated_completion: Optional[int] = None


class RunResponse(BaseModel):
    """Run response"""
    id: str = Field(..., description="Run ID")
    object: str = Field(default="run")
    eval_id: str
    status: RunStatus
    target_model: str
    created_at: int
    started_at: Optional[int] = None
    completed_at: Optional[int] = None
    progress: Optional[RunProgress] = None
    error_message: Optional[str] = None
    results: Optional[Dict[str, Any]] = None
    usage: Optional[Dict[str, int]] = None


class RunResultsResponse(BaseModel):
    """Run results response"""
    id: str
    eval_id: str
    status: str
    started_at: int
    completed_at: int
    results: Dict[str, Any]
    usage: Optional[Dict[str, int]] = None
    duration_seconds: float


class CreateDatasetRequest(BaseModel):
    """Create dataset request"""
    name: str = Field(..., min_length=1, max_length=255)
    description: Optional[str] = Field(None, max_length=1000)
    samples: List[DatasetSample] = Field(..., min_items=1)
    metadata: Optional[Dict[str, Any]] = None


class DatasetResponse(BaseModel):
    """Dataset response"""
    id: str
    object: str = Field(default="dataset")
    name: str
    description: Optional[str] = None
    sample_count: int
    created_at: int
    created_by: str
    metadata: Optional[Dict[str, Any]] = None


# ============= List Responses =============

class ListResponse(BaseModel):
    """Base list response"""
    object: str = Field(default="list")
    has_more: bool = Field(False)
    first_id: Optional[str] = None
    last_id: Optional[str] = None


class EvaluationListResponse(ListResponse):
    """Evaluation list response"""
    data: List[EvaluationResponse]


class RunListResponse(ListResponse):
    """Run list response"""
    data: List[RunResponse]


class DatasetListResponse(ListResponse):
    """Dataset list response"""
    data: List[DatasetResponse]


# ============= tldw-Specific Evaluation Schemas =============

class GEvalRequest(BaseModel):
    """G-Eval summarization request"""
    source_text: str = Field(
        ...,
        description="Original source text",
        min_length=10,
        max_length=100000
    )
    summary: str = Field(
        ...,
        description="Summary to evaluate",
        min_length=10,
        max_length=50000
    )
    metrics: Optional[List[Literal["fluency", "consistency", "relevance", "coherence"]]] = Field(
        default=["fluency", "consistency", "relevance", "coherence"]
    )
    api_name: Optional[str] = Field("openai", description="LLM API to use")
    api_key: Optional[str] = Field(None, description="API key if not in config")
    save_results: bool = Field(False, description="Save results to database")

    @field_validator('source_text', 'summary')
    @classmethod
    def sanitize_text(cls, v: str) -> str:
        sanitized = sanitize_html_text(v)
        if not sanitized or len(sanitized.strip()) < 10:
            raise ValueError("Text too short after sanitization")
        return sanitized


class GEvalResponse(BaseModel):
    """G-Eval response"""
    metrics: Dict[str, EvaluationMetric]
    average_score: float = Field(..., ge=0.0, le=1.0)
    summary_assessment: str
    evaluation_time: float
    metadata: Optional[Dict[str, Any]] = None


class RAGEvaluationRequest(BaseModel):
    """RAG evaluation request"""
    query: str = Field(..., min_length=1, max_length=10000)
    retrieved_contexts: List[str] = Field(..., min_items=1, max_items=20)
    generated_response: str = Field(..., min_length=1, max_length=50000)
    ground_truth: Optional[str] = Field(None, max_length=50000)
    metrics: Optional[List[str]] = Field(
        default=["relevance", "faithfulness", "answer_similarity", "context_precision"]
    )
    api_name: Optional[str] = Field("openai")

    @field_validator('query', 'generated_response', 'ground_truth')
    @classmethod
    def sanitize_text_fields(cls, v: Optional[str]) -> Optional[str]:
        if v is None:
            return None
        return sanitize_html_text(v)

    @field_validator('retrieved_contexts')
    @classmethod
    def sanitize_contexts(cls, v: List[str]) -> List[str]:
        return [sanitize_html_text(ctx) for ctx in v if ctx]


class RAGEvaluationResponse(BaseModel):
    """RAG evaluation response"""
    metrics: Dict[str, EvaluationMetric]
    overall_score: float = Field(..., ge=0.0, le=1.0)
    retrieval_quality: float = Field(..., ge=0.0, le=1.0)
    generation_quality: float = Field(..., ge=0.0, le=1.0)
    suggestions: Optional[List[str]] = None
    metadata: Optional[Dict[str, Any]] = None


class ResponseQualityRequest(BaseModel):
    """Response quality request"""
    prompt: str = Field(..., min_length=1, max_length=50000)
    response: str = Field(..., min_length=1, max_length=50000)
    expected_format: Optional[str] = Field(None, max_length=1000)
    evaluation_criteria: Optional[Dict[str, str]] = None
    api_name: Optional[str] = Field("openai")

    @field_validator('prompt', 'response', 'expected_format')
    @classmethod
    def sanitize_text_fields(cls, v: Optional[str]) -> Optional[str]:
        if v is None:
            return None
        return sanitize_html_text(v)


class ResponseQualityResponse(BaseModel):
    """Response quality response"""
    metrics: Dict[str, EvaluationMetric]
    overall_quality: float = Field(..., ge=0.0, le=1.0)
    format_compliance: Optional[Dict[str, bool]] = None
    issues: Optional[List[str]] = None
    improvements: Optional[List[str]] = None


class BatchEvaluationRequest(BaseModel):
    """Batch evaluation request"""
    evaluation_type: Literal["geval", "rag", "response_quality"]
    items: List[Dict[str, Any]] = Field(..., min_items=1, max_items=1000)
    parallel_workers: int = Field(4, ge=1, le=20)
    continue_on_error: bool = Field(True)


class BatchEvaluationResponse(BaseModel):
    """Batch evaluation response"""
    total_items: int
    successful: int
    failed: int
    results: List[Dict[str, Any]]
    aggregate_metrics: Optional[Dict[str, float]] = None
    processing_time: float


class CustomMetricRequest(BaseModel):
    """Custom metric request"""
    name: str = Field(..., min_length=1, max_length=100)
    description: str = Field(..., max_length=1000)
    evaluation_prompt: str = Field(..., max_length=10000)
    input_data: Dict[str, Any]
    scoring_criteria: Dict[str, Any]
    api_name: Optional[str] = Field("openai")


class CustomMetricResponse(BaseModel):
    """Custom metric response"""
    metric_name: str
    score: float = Field(..., ge=0.0, le=1.0)
    explanation: str
    details: Optional[Dict[str, Any]] = None


class EvaluationComparisonRequest(BaseModel):
    """Evaluation comparison request"""
    evaluation_ids: List[str] = Field(..., min_items=2, max_items=10)
    metrics_to_compare: Optional[List[str]] = None


class EvaluationComparisonResponse(BaseModel):
    """Evaluation comparison response"""
    evaluations: List[Dict[str, Any]]
    metric_comparisons: Dict[str, List[float]]
    improvements: Optional[Dict[str, float]] = None
    regressions: Optional[Dict[str, float]] = None
    summary: str


class EvaluationHistoryRequest(BaseModel):
    """Evaluation history request"""
    evaluation_type: Optional[str] = None
    start_date: Optional[datetime] = None
    end_date: Optional[datetime] = None
    limit: int = Field(20, ge=1, le=100)
    offset: int = Field(0, ge=0)


class EvaluationHistoryResponse(BaseModel):
    """Evaluation history response"""
    total_count: int
    items: List[Dict[str, Any]]
    aggregations: Optional[Dict[str, Any]] = None


# ============= Webhook Schemas =============

class WebhookRegistrationRequest(BaseModel):
    """Webhook registration request"""
    url: HttpUrl = Field(..., description="Webhook endpoint URL")
    events: List[WebhookEventType] = Field(..., min_items=1)
    secret: Optional[str] = Field(None, min_length=32)


class WebhookRegistrationResponse(BaseModel):
    """Webhook registration response"""
    webhook_id: int
    url: str
    events: List[str]
    secret: str
    created_at: datetime
    status: str = "active"


class WebhookUpdateRequest(BaseModel):
    """Webhook update request"""
    url: Optional[HttpUrl] = None
    events: Optional[List[WebhookEventType]] = None
    status: Optional[Literal["active", "paused"]] = None


class WebhookStatusResponse(BaseModel):
    """Webhook status response"""
    webhook_id: int
    url: str
    events: List[str]
    status: str
    created_at: datetime
    last_triggered: Optional[datetime] = None
    failure_count: int = 0


class WebhookTestRequest(BaseModel):
    """Webhook test request"""
    url: HttpUrl


class WebhookTestResponse(BaseModel):
    """Webhook test response"""
    success: bool
    status_code: Optional[int] = None
    response_time_ms: Optional[float] = None
    error: Optional[str] = None


# ============= Rate Limiting Schemas =============

class RateLimitStatusResponse(BaseModel):
    """Rate limit status response"""
    tier: str
    limits: Dict[str, int]
    usage: Dict[str, int]
    remaining: Dict[str, int]
    reset_at: datetime


# ============= Error Response =============

class ErrorDetail(BaseModel):
    """Error detail"""
    message: str
    type: str = "error"
    param: Optional[str] = None
    code: Optional[str] = None


class ErrorResponse(BaseModel):
    """Error response"""
    error: ErrorDetail


# ============= Health Check =============

class HealthCheckResponse(BaseModel):
    """Health check response"""
    status: Literal["healthy", "degraded", "unhealthy"]
    version: str
    uptime: float
    database: str
    circuit_breaker: Optional[str] = None
    rate_limit: Optional[Dict[str, Any]] = None
    checks: Optional[Dict[str, bool]] = None