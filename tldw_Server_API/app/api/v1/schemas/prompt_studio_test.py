# prompt_studio_test.py
# Test case and evaluation schemas for Prompt Studio

from typing import List, Optional, Dict, Any
from pydantic import BaseModel, ConfigDict, Field, field_validator
from datetime import datetime

from .prompt_studio_base import (
    TimestampMixin, SoftDeleteMixin, UUIDMixin,
    EvaluationStatus, JobStatus
)

########################################################################################################################
# Test Case Schemas

class TestCaseBase(BaseModel):
    """Base test case model"""
    name: Optional[str] = Field(None, max_length=255, description="Test case name")
    description: Optional[str] = Field(None, max_length=1000, description="Test case description")
    inputs: Dict[str, Any] = Field(..., description="Input data for test")
    expected_outputs: Optional[Dict[str, Any]] = Field(None, description="Expected output data")
    tags: Optional[List[str]] = Field(None, description="Tags for categorization")
    is_golden: bool = Field(default=False, description="Golden test case flag")

class TestCaseCreate(TestCaseBase):
    """Test case creation request"""
    project_id: int = Field(..., description="Parent project ID")
    signature_id: Optional[int] = Field(None, description="Associated signature ID")

class TestCaseUpdate(BaseModel):
    """Test case update request"""
    name: Optional[str] = Field(None, max_length=255)
    description: Optional[str] = Field(None, max_length=1000)
    inputs: Optional[Dict[str, Any]] = None
    expected_outputs: Optional[Dict[str, Any]] = None
    tags: Optional[List[str]] = None
    is_golden: Optional[bool] = None

class TestCaseResponse(TestCaseBase, TimestampMixin, UUIDMixin):
    """Test case response model"""
    id: int
    project_id: int
    signature_id: Optional[int]
    actual_outputs: Optional[Dict[str, Any]] = None
    is_generated: bool = False
    
    model_config = ConfigDict(from_attributes=True)

class TestCaseBulkCreate(BaseModel):
    """Bulk test case creation request"""
    project_id: int
    signature_id: Optional[int] = None
    test_cases: List[TestCaseBase]

class TestCaseImportRequest(BaseModel):
    """Test case import request"""
    project_id: int
    format: str = Field(..., pattern="^(csv|json)$", description="Import format")
    data: str = Field(..., description="Base64 encoded file data or JSON string")
    signature_id: Optional[int] = None
    auto_generate_names: bool = Field(default=True, description="Auto-generate names if missing")

class TestCaseExportRequest(BaseModel):
    """Test case export request"""
    project_id: int
    format: str = Field(default="json", pattern="^(csv|json)$", description="Export format")
    include_golden_only: bool = Field(default=False, description="Export only golden test cases")
    tag_filter: Optional[List[str]] = Field(None, description="Filter by tags")

class TestCaseGenerateRequest(BaseModel):
    """Request to auto-generate test cases"""
    project_id: int
    prompt_id: Optional[int] = None
    signature_id: Optional[int] = None
    num_cases: int = Field(default=5, ge=1, le=50, description="Number of test cases to generate")
    generation_strategy: str = Field(default="diverse", description="Generation strategy")
    base_on_description: Optional[str] = Field(None, max_length=2000, description="Description to base generation on")

########################################################################################################################
# Test Run Schemas

class TestRunBase(BaseModel):
    """Base test run model"""
    model_name: str = Field(..., description="Model used for test")
    model_params: Optional[Dict[str, Any]] = Field(None, description="Model parameters")
    inputs: Dict[str, Any]
    outputs: Dict[str, Any]
    expected_outputs: Optional[Dict[str, Any]] = None
    scores: Optional[Dict[str, float]] = None
    execution_time_ms: Optional[int] = None
    tokens_used: Optional[int] = None
    cost_estimate: Optional[float] = None
    error_message: Optional[str] = None

class TestRunCreate(BaseModel):
    """Test run execution request"""
    project_id: int
    prompt_id: int
    test_case_id: int
    model_name: str
    model_params: Optional[Dict[str, Any]] = None

class TestRunResponse(TestRunBase, UUIDMixin):
    """Test run response model"""
    id: int
    project_id: int
    prompt_id: int
    test_case_id: int
    created_at: datetime
    
    model_config = ConfigDict(from_attributes=True)

class BatchTestRequest(BaseModel):
    """Batch test execution request"""
    project_id: int
    prompt_id: int
    test_case_ids: List[int] = Field(..., min_length=1, description="Test cases to run")
    model_name: str
    model_params: Optional[Dict[str, Any]] = None
    parallel_execution: bool = Field(default=True, description="Execute tests in parallel")
    stop_on_error: bool = Field(default=False, description="Stop batch on first error")

class BatchTestResponse(BaseModel):
    """Batch test execution response"""
    total: int
    successful: int
    failed: int
    test_runs: List[TestRunResponse]
    aggregate_metrics: Dict[str, Any]

########################################################################################################################
# Evaluation Schemas

class EvaluationMetric(BaseModel):
    """Evaluation metric configuration"""
    name: str = Field(..., description="Metric name")
    type: str = Field(..., description="Metric type (accuracy, f1, custom, etc.)")
    config: Optional[Dict[str, Any]] = Field(None, description="Metric-specific configuration")
    weight: float = Field(default=1.0, ge=0.0, le=1.0, description="Metric weight")

class EvaluationCreate(BaseModel):
    """Evaluation creation request"""
    project_id: int
    prompt_id: int
    name: Optional[str] = Field(None, max_length=255, description="Evaluation name")
    description: Optional[str] = Field(None, max_length=1000, description="Evaluation description")
    test_case_ids: List[int] = Field(..., min_length=1, description="Test cases to evaluate")
    model_configs: List[Dict[str, Any]] = Field(..., min_length=1, description="Model configurations")
    metrics: Optional[List[EvaluationMetric]] = Field(None, description="Metrics to calculate")

class EvaluationResponse(TimestampMixin, UUIDMixin):
    """Evaluation response model"""
    id: int
    project_id: int
    prompt_id: int
    name: Optional[str]
    description: Optional[str]
    test_case_ids: List[int]
    test_run_ids: Optional[List[int]]
    aggregate_metrics: Optional[Dict[str, Any]]
    model_configs: List[Dict[str, Any]]
    total_tokens: Optional[int]
    total_cost: Optional[float]
    status: EvaluationStatus
    error_message: Optional[str]
    started_at: Optional[datetime]
    completed_at: Optional[datetime]
    
    model_config = ConfigDict(from_attributes=True)

class EvaluationStatusResponse(BaseModel):
    """Evaluation status response"""
    evaluation_id: int
    status: EvaluationStatus
    progress: Optional[float] = Field(None, ge=0.0, le=1.0, description="Progress percentage")
    current_step: Optional[str] = None
    error_message: Optional[str] = None

class EvaluationCompareRequest(BaseModel):
    """Request to compare multiple evaluations"""
    evaluation_ids: List[int] = Field(..., min_length=2, max_length=10, description="Evaluations to compare")
    metrics_focus: Optional[List[str]] = Field(None, description="Specific metrics to focus on")

class EvaluationCompareResponse(BaseModel):
    """Evaluation comparison response"""
    evaluations: List[EvaluationResponse]
    comparison_matrix: Dict[str, Any]
    best_performing: Dict[str, int]  # metric -> evaluation_id
    statistical_significance: Optional[Dict[str, Any]] = None

########################################################################################################################
# Score Schemas

class ScoreConfig(BaseModel):
    """Configuration for scoring"""
    scoring_method: str = Field(default="exact_match", description="Scoring method")
    threshold: Optional[float] = Field(None, ge=0.0, le=1.0, description="Score threshold")
    custom_scorer: Optional[str] = Field(None, description="Custom scorer function name")

class ScoreResult(BaseModel):
    """Score result for a test run"""
    metric_name: str
    score: float
    passed: bool
    details: Optional[Dict[str, Any]] = None