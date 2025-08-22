"""
Core types and protocols for functional RAG pipelines.

This module defines all the core types, protocols, and data structures used
throughout the functional pipeline system. It provides type safety and clear
interfaces for pipeline components.
"""

from typing import (
    Protocol, TypedDict, List, Dict, Any, Tuple, Optional, Union, 
    Callable, Awaitable, Generic, TypeVar, runtime_checkable
)
from dataclasses import dataclass, field
from enum import Enum
from pydantic import BaseModel, Field
import time
from abc import ABC, abstractmethod


# ==============================================================================
# Result Type for Error Handling
# ==============================================================================

T = TypeVar('T')
E = TypeVar('E')


@dataclass(frozen=True)
class Success(Generic[T]):
    """Represents a successful computation result."""
    value: T
    
    def is_success(self) -> bool:
        return True
    
    def is_failure(self) -> bool:
        return False
    
    def map(self, func: Callable[[T], 'U']) -> 'Result[U, E]':
        """Apply a function to the success value."""
        return Success(func(self.value))
    
    def flat_map(self, func: Callable[[T], 'Result[U, E]']) -> 'Result[U, E]':
        """Apply a function that returns a Result."""
        return func(self.value)
    
    def or_else(self, default: T) -> T:
        """Return the value or a default."""
        return self.value


@dataclass(frozen=True)
class Failure(Generic[E]):
    """Represents a failed computation with error information."""
    error: E
    
    def is_success(self) -> bool:
        return False
    
    def is_failure(self) -> bool:
        return True
    
    def map(self, func: Callable[[Any], Any]) -> 'Result[Any, E]':
        """Map does nothing for failures."""
        return self
    
    def flat_map(self, func: Callable[[Any], Any]) -> 'Result[Any, E]':
        """FlatMap does nothing for failures."""
        return self
    
    def or_else(self, default: Any) -> Any:
        """Return the default for failures."""
        return default


# Result is a union of Success and Failure
Result = Union[Success[T], Failure[E]]


# ==============================================================================
# Pipeline Error Types
# ==============================================================================

class PipelineErrorType(Enum):
    """Types of errors that can occur in pipelines."""
    RETRIEVAL_ERROR = "retrieval_error"
    PROCESSING_ERROR = "processing_error"
    FORMATTING_ERROR = "formatting_error"
    RESOURCE_ERROR = "resource_error"
    VALIDATION_ERROR = "validation_error"
    TIMEOUT_ERROR = "timeout_error"
    UNKNOWN_ERROR = "unknown_error"


@dataclass
class PipelineError:
    """Detailed error information for pipeline failures."""
    error_type: PipelineErrorType
    message: str
    step_name: Optional[str] = None
    cause: Optional[Exception] = None
    timestamp: float = field(default_factory=time.time)
    context: Dict[str, Any] = field(default_factory=dict)


# ==============================================================================
# Search Result Types
# ==============================================================================

class SearchResult(BaseModel):
    """Type-safe search result with all necessary fields."""
    source: str = Field(..., description="Source type: media, conversation, notes")
    id: str = Field(..., description="Unique identifier")
    title: str = Field(..., description="Result title")
    content: str = Field(..., description="Result content")
    score: float = Field(0.0, ge=0.0, le=1.0, description="Relevance score")
    metadata: Dict[str, Any] = Field(default_factory=dict)
    
    # Optional fields for enhanced results
    citations: Optional[List[Dict[str, Any]]] = None
    embedding_vector: Optional[List[float]] = None
    highlight_spans: Optional[List[Tuple[int, int]]] = None
    
    class Config:
        """Pydantic configuration."""
        json_encoders = {
            float: lambda v: round(v, 4)  # Round scores for cleaner output
        }


# ==============================================================================
# Pipeline Context Types
# ==============================================================================

class PipelineContext(TypedDict, total=False):
    """Type-safe pipeline execution context."""
    # Required fields
    query: str
    sources: Dict[str, bool]
    
    # Resource references (set by executor)
    resources: 'PipelineResources'
    
    # Configuration
    config: Dict[str, Any]
    params: Dict[str, Any]
    
    # Execution state
    step_index: int
    total_steps: int
    start_time: float
    
    # Intermediate results
    results: List[SearchResult]
    parallel_results: List[List[SearchResult]]
    formatted_output: str
    
    # Metadata
    trace_id: str
    user_id: Optional[str]
    session_id: Optional[str]


# ==============================================================================
# Resource Types
# ==============================================================================

@dataclass(frozen=True)
class PipelineResources:
    """Immutable container for pipeline resources."""
    app: Any  # App instance (avoid circular imports)
    media_db: Optional[Any] = None
    conversations_db: Optional[Any] = None
    notes_service: Optional[Any] = None
    vector_store: Optional[Any] = None
    embeddings_service: Optional[Any] = None
    connection_pool: Optional[Any] = None
    thread_pool: Optional[Any] = None
    cache: Optional[Any] = None
    
    def has_media_db(self) -> bool:
        """Check if media database is available."""
        return self.media_db is not None
    
    def has_conversations_db(self) -> bool:
        """Check if conversations database is available."""
        return self.conversations_db is not None
    
    def has_notes_service(self) -> bool:
        """Check if notes service is available."""
        return self.notes_service is not None
    
    def has_vector_store(self) -> bool:
        """Check if vector store is available."""
        return self.vector_store is not None


# ==============================================================================
# Pipeline Function Protocols
# ==============================================================================

@runtime_checkable
class RetrievalFunction(Protocol):
    """Protocol for retrieval functions."""
    async def __call__(
        self,
        context: PipelineContext,
        config: Dict[str, Any]
    ) -> Result[List[SearchResult], PipelineError]:
        """Execute retrieval and return results or error."""
        ...


@runtime_checkable
class ProcessingFunction(Protocol):
    """Protocol for processing functions (synchronous)."""
    def __call__(
        self,
        results: List[SearchResult],
        context: PipelineContext,
        config: Dict[str, Any]
    ) -> Result[List[SearchResult], PipelineError]:
        """Process results and return modified results or error."""
        ...


@runtime_checkable
class FormattingFunction(Protocol):
    """Protocol for formatting functions."""
    def __call__(
        self,
        results: List[SearchResult],
        context: PipelineContext,
        config: Dict[str, Any]
    ) -> Result[str, PipelineError]:
        """Format results and return string or error."""
        ...


# ==============================================================================
# Pipeline Configuration Types
# ==============================================================================

class StepType(Enum):
    """Types of pipeline steps."""
    RETRIEVE = "retrieve"
    PROCESS = "process"
    FORMAT = "format"
    PARALLEL = "parallel"
    MERGE = "merge"
    CONDITIONAL = "conditional"


@dataclass
class PipelineStep:
    """Configuration for a single pipeline step."""
    step_type: StepType
    function_name: Optional[str] = None
    config: Dict[str, Any] = field(default_factory=dict)
    
    # For parallel steps
    parallel_functions: Optional[List['PipelineStep']] = None
    
    # For conditional steps
    condition: Optional[str] = None
    if_true: Optional['PipelineStep'] = None
    if_false: Optional['PipelineStep'] = None
    
    # Metadata
    name: Optional[str] = None
    description: Optional[str] = None
    timeout_seconds: Optional[float] = None
    retry_count: int = 0


@dataclass
class PipelineConfig:
    """Complete pipeline configuration."""
    id: str
    name: str
    description: str
    steps: List[PipelineStep]
    
    # Pipeline metadata
    version: str = "1.0"
    tags: List[str] = field(default_factory=list)
    enabled: bool = True
    
    # Performance settings
    timeout_seconds: float = 30.0
    cache_results: bool = True
    cache_ttl_seconds: float = 3600.0
    
    # Error handling
    on_error: str = "fail"  # "fail", "continue", "fallback"
    fallback_pipeline: Optional[str] = None


# ==============================================================================
# Effect System Types
# ==============================================================================

@dataclass
class Effect:
    """Represents a side effect to be executed."""
    name: str
    action: Callable[[], Awaitable[None]]
    metadata: Dict[str, Any] = field(default_factory=dict)


class EffectType(Enum):
    """Types of effects."""
    LOG = "log"
    METRIC = "metric"
    CACHE_WRITE = "cache_write"
    NOTIFICATION = "notification"
    DATABASE_WRITE = "database_write"


@dataclass
class TypedEffect(Effect):
    """Effect with type information."""
    effect_type: EffectType
    
    @classmethod
    def log(cls, level: str, message: str, **kwargs) -> 'TypedEffect':
        """Create a logging effect."""
        async def action():
            from loguru import logger
            getattr(logger, level)(message, **kwargs)
        
        return cls(
            name=f"log_{level}",
            effect_type=EffectType.LOG,
            action=action,
            metadata={"level": level, "message": message, **kwargs}
        )
    
    @classmethod
    def metric(cls, name: str, value: float, metric_type: str = "counter") -> 'TypedEffect':
        """Create a metrics effect."""
        async def action():
            from ...Metrics.metrics_logger import log_counter, log_histogram, log_gauge
            if metric_type == "counter":
                log_counter(name, value)
            elif metric_type == "histogram":
                log_histogram(name, value)
            elif metric_type == "gauge":
                log_gauge(name, value)
        
        return cls(
            name=f"metric_{name}",
            effect_type=EffectType.METRIC,
            action=action,
            metadata={"metric_name": name, "value": value, "type": metric_type}
        )


# ==============================================================================
# Pipeline Types
# ==============================================================================

# Type alias for a complete pipeline function
Pipeline = Callable[
    [PipelineContext],
    Awaitable[Tuple[Result[List[SearchResult], PipelineError], List[Effect]]]
]


# ==============================================================================
# Utility Types
# ==============================================================================

@dataclass
class PipelineMetrics:
    """Metrics collected during pipeline execution."""
    total_duration_ms: float
    step_durations_ms: Dict[str, float] = field(default_factory=dict)
    retrieval_count: int = 0
    processing_count: int = 0
    final_result_count: int = 0
    cache_hits: int = 0
    cache_misses: int = 0
    errors: List[PipelineError] = field(default_factory=list)


@dataclass
class PipelineTrace:
    """Detailed trace of pipeline execution for debugging."""
    trace_id: str
    pipeline_id: str
    start_time: float
    end_time: Optional[float] = None
    steps: List[Dict[str, Any]] = field(default_factory=list)
    metrics: Optional[PipelineMetrics] = None
    context_snapshot: Optional[Dict[str, Any]] = None
    

# ==============================================================================
# Validation Types
# ==============================================================================

class ValidationError(Exception):
    """Raised when pipeline validation fails."""
    def __init__(self, errors: List[str]):
        self.errors = errors
        super().__init__(f"Pipeline validation failed with {len(errors)} errors")


@dataclass
class ValidationResult:
    """Result of pipeline validation."""
    is_valid: bool
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    
    def raise_if_invalid(self):
        """Raise ValidationError if validation failed."""
        if not self.is_valid:
            raise ValidationError(self.errors)