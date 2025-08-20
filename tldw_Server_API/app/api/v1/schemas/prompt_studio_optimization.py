# prompt_studio_optimization.py
# Optimization and job queue schemas for Prompt Studio

from typing import List, Optional, Dict, Any
from pydantic import BaseModel, ConfigDict, Field, field_validator
from datetime import datetime

from .prompt_studio_base import (
    TimestampMixin, UUIDMixin,
    JobType, JobStatus, EvaluationStatus
)

########################################################################################################################
# Optimization Schemas

class OptimizationTechnique(str):
    """Available optimization techniques"""
    MIPRO = "mipro"
    BOOTSTRAP = "bootstrap"
    HILL_CLIMBING = "hill_climbing"
    RANDOM_SEARCH = "random_search"
    GRID_SEARCH = "grid_search"
    BAYESIAN = "bayesian"

class OptimizationConfig(BaseModel):
    """Configuration for optimization run"""
    optimizer_type: str = Field(..., description="Type of optimizer to use")
    max_iterations: int = Field(default=50, ge=1, le=500, description="Maximum iterations")
    target_metric: str = Field(..., description="Metric to optimize")
    target_value: Optional[float] = Field(None, description="Target metric value to achieve")
    early_stopping: bool = Field(default=True, description="Enable early stopping")
    early_stopping_patience: int = Field(default=5, ge=1, description="Iterations without improvement before stopping")
    temperature_range: List[float] = Field(default=[0.0, 1.0], description="Temperature range to explore")
    techniques_to_try: List[str] = Field(default=["cot", "few_shot"], description="Prompt techniques to try")
    models_to_test: Optional[List[str]] = Field(None, description="Models to test during optimization")
    budget_limit: Optional[float] = Field(None, ge=0.0, description="Maximum budget in dollars")
    
    @field_validator('temperature_range')
    @classmethod
    def validate_temperature_range(cls, v):
        if len(v) != 2 or v[0] >= v[1]:
            raise ValueError("temperature_range must be [min, max] where min < max")
        if v[0] < 0.0 or v[1] > 2.0:
            raise ValueError("temperature values must be between 0.0 and 2.0")
        return v

class BootstrapConfig(BaseModel):
    """Configuration for bootstrapping examples"""
    num_samples: int = Field(default=50, ge=10, le=1000, description="Number of samples to bootstrap")
    selection_method: str = Field(default="diverse", description="Selection method for examples")
    quality_threshold: float = Field(default=0.7, ge=0.0, le=1.0, description="Quality threshold for examples")
    max_examples_per_prompt: int = Field(default=5, ge=1, le=20, description="Max examples to include")

class OptimizationCreate(BaseModel):
    """Optimization creation request"""
    project_id: int
    initial_prompt_id: int
    optimization_config: OptimizationConfig
    bootstrap_config: Optional[BootstrapConfig] = None
    test_case_ids: Optional[List[int]] = Field(None, description="Specific test cases to optimize against")
    name: Optional[str] = Field(None, max_length=255, description="Optimization run name")
    description: Optional[str] = Field(None, max_length=1000, description="Optimization run description")

class OptimizationResponse(TimestampMixin, UUIDMixin):
    """Optimization response model"""
    id: int
    project_id: int
    initial_prompt_id: int
    optimized_prompt_id: Optional[int]
    optimizer_type: str
    optimization_config: Dict[str, Any]
    initial_metrics: Optional[Dict[str, Any]]
    final_metrics: Optional[Dict[str, Any]]
    improvement_percentage: Optional[float]
    iterations_completed: Optional[int]
    max_iterations: int
    bootstrap_samples: Optional[int]
    status: str
    error_message: Optional[str]
    total_tokens: Optional[int]
    total_cost: Optional[float]
    started_at: Optional[datetime]
    completed_at: Optional[datetime]
    
    model_config = ConfigDict(from_attributes=True)

class OptimizationStatusResponse(BaseModel):
    """Optimization status response"""
    optimization_id: int
    status: str
    progress: float = Field(ge=0.0, le=1.0, description="Progress percentage")
    current_iteration: int
    max_iterations: int
    current_best_metric: Optional[float] = None
    estimated_time_remaining: Optional[int] = Field(None, description="Estimated seconds remaining")
    current_step: Optional[str] = None

class OptimizationIteration(BaseModel):
    """Single iteration of optimization"""
    iteration_number: int
    prompt_variant: Dict[str, Any]
    metrics: Dict[str, float]
    tokens_used: int
    cost: float
    timestamp: datetime

class OptimizationHistory(BaseModel):
    """Optimization history"""
    optimization_id: int
    iterations: List[OptimizationIteration]
    best_iteration: int
    convergence_data: Dict[str, Any]

########################################################################################################################
# Job Queue Schemas

class JobCreate(BaseModel):
    """Job creation request"""
    job_type: JobType
    entity_id: int = Field(..., description="ID of entity (evaluation, optimization, etc.)")
    priority: int = Field(default=5, ge=1, le=10, description="Job priority (1=lowest, 10=highest)")
    payload: Dict[str, Any] = Field(..., description="Job-specific payload")
    max_retries: int = Field(default=3, ge=0, le=10, description="Maximum retry attempts")

class JobResponse(TimestampMixin, UUIDMixin):
    """Job response model"""
    id: int
    job_type: JobType
    entity_id: int
    priority: int
    status: JobStatus
    payload: Dict[str, Any]
    result: Optional[Dict[str, Any]]
    error_message: Optional[str]
    retry_count: int
    max_retries: int
    started_at: Optional[datetime]
    completed_at: Optional[datetime]
    
    model_config = ConfigDict(from_attributes=True)

class JobStatusResponse(BaseModel):
    """Job status response"""
    job_id: int
    status: JobStatus
    progress: Optional[float] = Field(None, ge=0.0, le=1.0)
    current_step: Optional[str] = None
    error_message: Optional[str] = None
    retry_count: int

class JobListResponse(BaseModel):
    """Job list response"""
    jobs: List[JobResponse]
    queued_count: int
    processing_count: int
    completed_count: int
    failed_count: int

class JobCancelRequest(BaseModel):
    """Job cancellation request"""
    reason: Optional[str] = Field(None, max_length=500, description="Cancellation reason")

########################################################################################################################
# Module Configuration Schemas

from enum import Enum

class ModuleType(str, Enum):
    """Available prompt modules"""
    CHAIN_OF_THOUGHT = "chain_of_thought"
    REACT = "react"
    PROGRAM_OF_THOUGHT = "program_of_thought"
    MULTI_CHAIN = "multi_chain"
    ROLE_PLAY = "role_play"
    STRUCTURED_OUTPUT = "structured_output"

class ModuleConfig(BaseModel):
    """Module configuration"""
    module_type: ModuleType
    enabled: bool = True
    parameters: Dict[str, Any] = Field(default_factory=dict)
    priority: int = Field(default=5, ge=1, le=10)

class ModuleLibrary(BaseModel):
    """Available modules library"""
    modules: List[ModuleConfig]
    presets: Dict[str, List[ModuleConfig]]

########################################################################################################################
# Cost Analysis Schemas

class CostEstimate(BaseModel):
    """Cost estimate for an operation"""
    estimated_tokens: int
    estimated_cost: float
    cost_breakdown: Dict[str, float]
    model_pricing: Dict[str, float]

class CostAnalysisRequest(BaseModel):
    """Request for cost analysis"""
    project_id: int
    prompt_id: Optional[int] = None
    test_case_ids: Optional[List[int]] = None
    model_name: str
    include_optimization: bool = Field(default=False)
    optimization_iterations: int = Field(default=50)

class CostAnalysisResponse(BaseModel):
    """Cost analysis response"""
    total_estimated_cost: float
    cost_per_test_case: float
    cost_per_optimization_iteration: Optional[float] = None
    recommendations: List[str]
    alternative_models: Dict[str, float]