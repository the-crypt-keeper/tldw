"""
Unified RAG Schema - Single comprehensive schema for the unified pipeline

This schema maps directly to the unified_rag_pipeline parameters,
providing a clean API interface with all features accessible.
"""

from typing import List, Optional, Literal, Dict, Any
from pydantic import BaseModel, Field, validator


class UnifiedRAGRequest(BaseModel):
    """
    Unified RAG request with ALL features as optional parameters.
    Every feature in the RAG system is accessible through this single schema.
    """
    
    # ========== REQUIRED ==========
    query: str = Field(
        ...,
        description="The search query",
        min_length=1,
        max_length=2000,
        example="What is machine learning?"
    )
    
    # ========== DATA SOURCES ==========
    sources: Optional[List[str]] = Field(
        default=["media_db"],
        description="Databases to search: media_db, notes, characters, chats",
        example=["media_db", "notes"]
    )
    
    # ========== SEARCH CONFIGURATION ==========
    search_mode: Literal["fts", "vector", "hybrid"] = Field(
        default="hybrid",
        description="Search mode: fts (full-text), vector (semantic), or hybrid",
        example="hybrid"
    )
    
    hybrid_alpha: float = Field(
        default=0.7,
        ge=0.0,
        le=1.0,
        description="Hybrid search weight (0=FTS only, 1=Vector only)",
        example=0.7
    )
    
    top_k: int = Field(
        default=10,
        ge=1,
        le=100,
        description="Number of results to return",
        example=10
    )
    
    min_score: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Minimum relevance score",
        example=0.0
    )
    
    # ========== QUERY EXPANSION ==========
    expand_query: bool = Field(
        default=False,
        description="Enable query expansion",
        example=True
    )
    
    expansion_strategies: Optional[List[str]] = Field(
        default=None,
        description="Expansion strategies: acronym, synonym, domain, entity",
        example=["acronym", "synonym"]
    )
    
    spell_check: bool = Field(
        default=False,
        description="Enable spell checking",
        example=False
    )
    
    # ========== CACHING ==========
    enable_cache: bool = Field(
        default=True,
        description="Enable semantic caching",
        example=True
    )
    
    cache_threshold: float = Field(
        default=0.85,
        ge=0.0,
        le=1.0,
        description="Cache similarity threshold",
        example=0.85
    )
    
    adaptive_cache: bool = Field(
        default=True,
        description="Use adaptive cache thresholds",
        example=True
    )
    
    # ========== FILTERING ==========
    keyword_filter: Optional[List[str]] = Field(
        default=None,
        description="Filter results by keywords",
        example=["python", "api"]
    )
    
    # ========== SECURITY & PRIVACY ==========
    enable_security_filter: bool = Field(
        default=False,
        description="Enable security filtering",
        example=False
    )
    
    detect_pii: bool = Field(
        default=False,
        description="Detect personally identifiable information",
        example=False
    )
    
    redact_pii: bool = Field(
        default=False,
        description="Redact detected PII",
        example=False
    )
    
    sensitivity_level: Literal["public", "internal", "confidential", "restricted"] = Field(
        default="public",
        description="Maximum sensitivity level for results",
        example="internal"
    )
    
    content_filter: bool = Field(
        default=False,
        description="Enable content filtering",
        example=False
    )
    
    # ========== DOCUMENT PROCESSING ==========
    enable_table_processing: bool = Field(
        default=False,
        description="Enable table extraction and processing",
        example=False
    )
    
    table_method: Literal["markdown", "html", "hybrid"] = Field(
        default="markdown",
        description="Table serialization method",
        example="markdown"
    )
    
    # ========== CHUNKING & CONTEXT ==========
    enable_enhanced_chunking: bool = Field(
        default=False,
        description="Enable enhanced document chunking",
        example=False
    )
    
    chunk_type_filter: Optional[List[str]] = Field(
        default=None,
        description="Filter chunks by type: text, code, table, list",
        example=["text", "code"]
    )
    
    enable_parent_expansion: bool = Field(
        default=False,
        description="Expand chunks with parent document context",
        example=False
    )
    
    parent_context_size: int = Field(
        default=500,
        ge=100,
        le=2000,
        description="Size of parent context in characters",
        example=500
    )
    
    include_sibling_chunks: bool = Field(
        default=False,
        description="Include adjacent chunks from same document",
        example=False
    )
    
    # ========== RERANKING ==========
    enable_reranking: bool = Field(
        default=True,
        description="Enable document reranking",
        example=True
    )
    
    reranking_strategy: Literal["flashrank", "cross_encoder", "hybrid", "none"] = Field(
        default="flashrank",
        description="Reranking strategy",
        example="hybrid"
    )
    
    rerank_top_k: Optional[int] = Field(
        default=None,
        ge=1,
        le=100,
        description="Number of documents to rerank (defaults to top_k)",
        example=20
    )
    
    # ========== CITATIONS ==========
    enable_citations: bool = Field(
        default=False,
        description="Generate citations from results",
        example=True
    )
    
    citation_style: Literal["apa", "mla", "chicago", "harvard"] = Field(
        default="apa",
        description="Citation format style",
        example="apa"
    )
    
    include_page_numbers: bool = Field(
        default=False,
        description="Include page numbers in citations",
        example=False
    )
    
    # ========== ANSWER GENERATION ==========
    enable_generation: bool = Field(
        default=False,
        description="Generate an answer from retrieved context",
        example=True
    )
    
    generation_model: Optional[str] = Field(
        default=None,
        description="LLM model for answer generation",
        example="gpt-3.5-turbo"
    )
    
    generation_prompt: Optional[str] = Field(
        default=None,
        description="Custom prompt template for generation",
        example=None
    )
    
    max_generation_tokens: int = Field(
        default=500,
        ge=50,
        le=2000,
        description="Maximum tokens for generated answer",
        example=500
    )
    
    # ========== FEEDBACK ==========
    collect_feedback: bool = Field(
        default=False,
        description="Enable feedback collection",
        example=False
    )
    
    feedback_user_id: Optional[str] = Field(
        default=None,
        description="User ID for feedback tracking",
        example="user123"
    )
    
    apply_feedback_boost: bool = Field(
        default=False,
        description="Apply feedback-based result boosting",
        example=False
    )
    
    # ========== MONITORING ==========
    enable_monitoring: bool = Field(
        default=False,
        description="Enable performance monitoring",
        example=True
    )
    
    enable_observability: bool = Field(
        default=False,
        description="Enable observability tracing",
        example=False
    )
    
    trace_id: Optional[str] = Field(
        default=None,
        description="Trace ID for observability",
        example=None
    )
    
    # ========== PERFORMANCE ==========
    enable_performance_analysis: bool = Field(
        default=False,
        description="Enable detailed performance analysis",
        example=False
    )
    
    timeout_seconds: Optional[float] = Field(
        default=None,
        ge=1.0,
        le=60.0,
        description="Request timeout in seconds",
        example=10.0
    )
    
    # ========== QUICK WINS ==========
    highlight_results: bool = Field(
        default=False,
        description="Highlight matching terms in results",
        example=False
    )
    
    highlight_query_terms: bool = Field(
        default=False,
        description="Highlight query terms specifically",
        example=False
    )
    
    track_cost: bool = Field(
        default=False,
        description="Track estimated API costs",
        example=False
    )
    
    debug_mode: bool = Field(
        default=False,
        description="Enable debug logging",
        example=False
    )
    
    # ========== BATCH PROCESSING ==========
    enable_batch: bool = Field(
        default=False,
        description="Enable batch query processing",
        example=False
    )
    
    batch_queries: Optional[List[str]] = Field(
        default=None,
        description="Additional queries for batch processing",
        example=["query1", "query2"]
    )
    
    batch_concurrent: int = Field(
        default=5,
        ge=1,
        le=20,
        description="Maximum concurrent batch queries",
        example=5
    )
    
    # ========== RESILIENCE ==========
    enable_resilience: bool = Field(
        default=False,
        description="Enable resilience features",
        example=False
    )
    
    retry_attempts: int = Field(
        default=3,
        ge=1,
        le=5,
        description="Number of retry attempts",
        example=3
    )
    
    circuit_breaker: bool = Field(
        default=False,
        description="Enable circuit breaker",
        example=False
    )
    
    # ========== USER CONTEXT ==========
    user_id: Optional[str] = Field(
        default=None,
        description="User ID for personalization",
        example="user123"
    )
    
    session_id: Optional[str] = Field(
        default=None,
        description="Session ID for tracking",
        example="session456"
    )
    
    class Config:
        schema_extra = {
            "example": {
                "query": "What is machine learning?",
                "sources": ["media_db", "notes"],
                "expand_query": True,
                "expansion_strategies": ["synonym", "acronym"],
                "enable_citations": True,
                "enable_generation": True,
                "enable_reranking": True,
                "reranking_strategy": "hybrid"
            }
        }
    
    @validator('sources')
    def validate_sources(cls, v):
        valid_sources = {"media_db", "media", "notes", "characters", "chats"}
        if v:
            invalid = set(v) - valid_sources
            if invalid:
                raise ValueError(f"Invalid sources: {invalid}. Valid options: {valid_sources}")
        return v
    
    @validator('expansion_strategies')
    def validate_expansion_strategies(cls, v):
        if v:
            valid_strategies = {"acronym", "synonym", "semantic", "domain", "entity"}
            invalid = set(v) - valid_strategies
            if invalid:
                raise ValueError(f"Invalid strategies: {invalid}. Valid options: {valid_strategies}")
        return v
    
    @validator('chunk_type_filter')
    def validate_chunk_types(cls, v):
        if v:
            valid_types = {"text", "code", "table", "list"}
            invalid = set(v) - valid_types
            if invalid:
                raise ValueError(f"Invalid chunk types: {invalid}. Valid options: {valid_types}")
        return v


class UnifiedRAGResponse(BaseModel):
    """Unified response structure for RAG queries."""
    
    documents: List[Dict[str, Any]] = Field(
        description="Retrieved documents"
    )
    
    query: str = Field(
        description="Original query"
    )
    
    expanded_queries: List[str] = Field(
        default_factory=list,
        description="Expanded query variations"
    )
    
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional metadata"
    )
    
    timings: Dict[str, float] = Field(
        default_factory=dict,
        description="Performance timings"
    )
    
    citations: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Generated citations"
    )
    
    feedback_id: Optional[str] = Field(
        default=None,
        description="Feedback tracking ID"
    )
    
    generated_answer: Optional[str] = Field(
        default=None,
        description="Generated answer from context"
    )
    
    cache_hit: bool = Field(
        default=False,
        description="Whether result was from cache"
    )
    
    errors: List[str] = Field(
        default_factory=list,
        description="Any errors encountered"
    )
    
    security_report: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Security analysis report"
    )
    
    total_time: float = Field(
        default=0.0,
        description="Total execution time"
    )
    
    class Config:
        schema_extra = {
            "example": {
                "documents": [
                    {
                        "id": "doc1",
                        "content": "Machine learning is a subset of artificial intelligence...",
                        "score": 0.95,
                        "metadata": {"source": "media_db", "title": "ML Introduction"}
                    }
                ],
                "query": "What is machine learning?",
                "expanded_queries": ["machine learning definition", "ML explanation"],
                "metadata": {
                    "sources_searched": ["media_db", "notes"],
                    "documents_retrieved": 10,
                    "cache_hit": False
                },
                "timings": {
                    "query_expansion": 0.05,
                    "retrieval": 0.2,
                    "reranking": 0.1,
                    "total": 0.35
                },
                "citations": [
                    {
                        "text": "Machine learning is a subset of artificial intelligence",
                        "source": "ML Introduction",
                        "confidence": 0.95,
                        "type": "exact"
                    }
                ],
                "generated_answer": "Machine learning is a branch of AI that enables systems to learn from data...",
                "cache_hit": False,
                "errors": [],
                "total_time": 0.35
            }
        }


class UnifiedBatchRequest(BaseModel):
    """Request for batch processing multiple queries."""
    
    queries: List[str] = Field(
        ...,
        min_items=1,
        max_items=100,
        description="List of queries to process",
        example=["What is AI?", "Explain neural networks", "Define machine learning"]
    )
    
    max_concurrent: int = Field(
        default=5,
        ge=1,
        le=20,
        description="Maximum concurrent processing",
        example=5
    )
    
    # Include all parameters from UnifiedRAGRequest except query
    # These will be applied to all queries in the batch
    **{k: v for k, v in UnifiedRAGRequest.__fields__.items() if k != 'query'}
    
    class Config:
        schema_extra = {
            "example": {
                "queries": ["What is AI?", "Explain neural networks"],
                "max_concurrent": 5,
                "expand_query": True,
                "enable_citations": True,
                "enable_reranking": True
            }
        }


class UnifiedBatchResponse(BaseModel):
    """Response for batch processing."""
    
    results: List[UnifiedRAGResponse] = Field(
        description="Results for each query"
    )
    
    total_queries: int = Field(
        description="Total number of queries processed"
    )
    
    successful: int = Field(
        description="Number of successful queries"
    )
    
    failed: int = Field(
        description="Number of failed queries"
    )
    
    total_time: float = Field(
        description="Total batch processing time"
    )
    
    class Config:
        schema_extra = {
            "example": {
                "results": [],  # List of UnifiedRAGResponse objects
                "total_queries": 2,
                "successful": 2,
                "failed": 0,
                "total_time": 0.75
            }
        }