# rag_schemas_simple.py - Simplified RAG API Schemas
"""
Simplified schemas for the refactored RAG API endpoints.
These schemas provide a cleaner, more developer-friendly interface
for search and agent functionality.
"""

from typing import List, Optional, Dict, Any, Literal
from pydantic import BaseModel, Field, field_validator
from uuid import UUID, uuid4
from enum import Enum


# ============= Enums =============

class SearchType(str, Enum):
    """Available search types"""
    HYBRID = "hybrid"
    SEMANTIC = "semantic"
    FULLTEXT = "fulltext"


class DatabaseType(str, Enum):
    """Available databases to search"""
    MEDIA_DB = "media_db"
    MEDIA = "media"  # Alias for media_db
    NOTES = "notes"
    CHARACTERS = "characters"
    CHAT_HISTORY = "chat_history"
    CHATS = "chats"  # Alias for chat_history


class AgentMode(str, Enum):
    """Agent operation modes"""
    RAG = "rag"  # Simple Q&A with retrieval
    RESEARCH = "research"  # Multi-step research with tools


class ResearchTool(str, Enum):
    """Available tools for research mode"""
    WEB_SEARCH = "web_search"
    WEB_SCRAPE = "web_scrape"
    REASONING = "reasoning"
    PYTHON_EXECUTOR = "python_executor"


class SearchStrategy(str, Enum):
    """Advanced search strategies"""
    VANILLA = "vanilla"
    QUERY_FUSION = "query_fusion"
    HYDE = "hyde"


# ============= Simple Search Schemas =============

class SimpleSearchRequest(BaseModel):
    """
    Simple search request with essential parameters only.
    Covers 90% of search use cases with minimal complexity.
    """
    query: str = Field(
        ...,
        description="The search query string",
        min_length=1,
        max_length=1000,
        examples=["machine learning basics", "how to use RAG"]
    )
    
    search_type: SearchType = Field(
        default=SearchType.HYBRID,
        description="Type of search to perform"
    )
    
    limit: int = Field(
        default=10,
        ge=1,
        le=100,
        description="Maximum number of results to return"
    )
    
    databases: List[str] = Field(
        default=["media_db"],
        description="List of databases to search. Options: media_db, notes, characters, chat_history",
        examples=[["media_db", "notes"], ["media_db"]]
    )
    
    keywords: Optional[List[str]] = Field(
        default=None,
        description="Optional keywords to filter results",
        examples=[["AI", "ML"], ["python", "tutorial"]]
    )
    
    @field_validator('databases')
    def validate_databases(cls, v):
        """Validate and normalize database names"""
        valid_dbs = {"media_db", "media", "notes", "characters", "chat_history", "chats"}
        normalized = []
        for db in v:
            db_lower = db.lower()
            if db_lower not in valid_dbs:
                raise ValueError(f"Invalid database: {db}. Valid options: {valid_dbs}")
            # Normalize aliases
            if db_lower == "media":
                normalized.append("media_db")
            elif db_lower == "chats":
                normalized.append("chat_history")
            else:
                normalized.append(db_lower)
        return normalized


class SearchResult(BaseModel):
    """Individual search result"""
    id: str = Field(..., description="Unique identifier of the result")
    title: str = Field(..., description="Title of the content")
    content: str = Field(..., description="Content snippet relevant to the query")
    score: float = Field(..., description="Relevance score (higher is better)")
    source: str = Field(..., description="Source database")
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional metadata about the result"
    )


class SimpleSearchResponse(BaseModel):
    """Simple search response"""
    results: List[SearchResult] = Field(
        ...,
        description="List of search results"
    )
    total_results: int = Field(
        ...,
        description="Total number of results found"
    )
    query_id: str = Field(
        default_factory=lambda: str(uuid4()),
        description="Unique identifier for this search query"
    )
    search_type_used: SearchType = Field(
        ...,
        description="The search type that was used"
    )
    
    class Config:
        json_schema_extra = {
            "example": {
                "results": [
                    {
                        "id": "doc_123",
                        "title": "Introduction to RAG",
                        "content": "Retrieval-Augmented Generation combines...",
                        "score": 0.95,
                        "source": "media_db",
                        "metadata": {"author": "AI Expert", "date": "2024-01-15"}
                    }
                ],
                "total_results": 1,
                "query_id": "550e8400-e29b-41d4-a716-446655440000",
                "search_type_used": "hybrid"
            }
        }


# ============= Advanced Search Schemas =============

class HybridSearchConfig(BaseModel):
    """Configuration for hybrid search"""
    semantic_weight: float = Field(
        default=5.0,
        ge=0.0,
        le=10.0,
        description="Weight for semantic search results"
    )
    fulltext_weight: float = Field(
        default=1.0,
        ge=0.0,
        le=10.0,
        description="Weight for full-text search results"
    )
    rrf_k: int = Field(
        default=50,
        ge=1,
        description="K parameter for Reciprocal Rank Fusion"
    )


class SemanticSearchConfig(BaseModel):
    """Configuration for semantic search"""
    similarity_threshold: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Minimum similarity score threshold"
    )
    rerank: bool = Field(
        default=False,
        description="Whether to use reranking model"
    )


class SearchConfig(BaseModel):
    """Advanced search configuration"""
    search_type: SearchType = Field(default=SearchType.HYBRID)
    limit: int = Field(default=10, ge=1, le=1000)
    offset: int = Field(default=0, ge=0)
    databases: List[str] = Field(default=["media_db"])
    keywords: Optional[List[str]] = None
    date_range: Optional[Dict[str, str]] = Field(
        default=None,
        description="Date range filter with 'start' and 'end' keys (ISO format)",
        examples=[{"start": "2024-01-01", "end": "2024-12-31"}]
    )
    metadata_filters: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Complex metadata filters",
        examples=[{"author": "John Doe", "type": "article"}]
    )
    include_scores: bool = Field(default=True)
    include_full_content: bool = Field(default=False)


class AdvancedSearchRequest(BaseModel):
    """Advanced search request with full control"""
    query: str = Field(..., min_length=1, max_length=1000)
    search_config: SearchConfig = Field(default_factory=SearchConfig)
    hybrid_config: Optional[HybridSearchConfig] = None
    semantic_config: Optional[SemanticSearchConfig] = None
    strategy: SearchStrategy = Field(default=SearchStrategy.VANILLA)


class AdvancedSearchResponse(BaseModel):
    """Advanced search response with additional details"""
    results: List[SearchResult]
    total_results: int
    query_id: str = Field(default_factory=lambda: str(uuid4()))
    search_type_used: SearchType
    strategy_used: SearchStrategy
    search_config: Dict[str, Any] = Field(
        description="The actual search configuration used"
    )
    debug_info: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Debug information (if requested)"
    )


# ============= Simple Agent Schemas =============

class SimpleAgentRequest(BaseModel):
    """
    Simple agent request for basic Q&A with retrieval.
    """
    message: str = Field(
        ...,
        description="The user's message or question",
        min_length=1,
        max_length=4000,
        examples=["What is machine learning?", "Explain the concept of RAG"]
    )
    
    conversation_id: Optional[str] = Field(
        default=None,
        description="Optional conversation ID to maintain context"
    )
    
    search_databases: List[str] = Field(
        default=["media_db"],
        description="Which databases to search for context",
        examples=[["media_db", "notes"]]
    )
    
    model: Optional[str] = Field(
        default=None,
        description="Optional model to use (uses default if not specified)",
        examples=["gpt-4", "claude-3", "llama-3"]
    )
    
    @field_validator('search_databases')
    def validate_search_databases(cls, v):
        """Validate and normalize database names"""
        valid_dbs = {"media_db", "media", "notes", "characters", "chat_history", "chats"}
        normalized = []
        for db in v:
            db_lower = db.lower()
            if db_lower not in valid_dbs:
                raise ValueError(f"Invalid database: {db}. Valid options: {valid_dbs}")
            # Normalize aliases
            if db_lower == "media":
                normalized.append("media_db")
            elif db_lower == "chats":
                normalized.append("chat_history")
            else:
                normalized.append(db_lower)
        return normalized


class Source(BaseModel):
    """Source information for agent response"""
    title: str = Field(..., description="Title of the source")
    content: str = Field(..., description="Relevant content from the source")
    database: str = Field(..., description="Source database")
    relevance_score: float = Field(..., description="Relevance score")


class SimpleAgentResponse(BaseModel):
    """Simple agent response"""
    response: str = Field(
        ...,
        description="The agent's response to the user's message"
    )
    conversation_id: str = Field(
        ...,
        description="Conversation ID for continuing the conversation"
    )
    sources: List[Source] = Field(
        default_factory=list,
        description="Sources used to generate the response"
    )
    
    class Config:
        json_schema_extra = {
            "example": {
                "response": "Machine learning is a subset of artificial intelligence...",
                "conversation_id": "conv_123456",
                "sources": [
                    {
                        "title": "ML Basics",
                        "content": "Machine learning enables computers to learn...",
                        "database": "media_db",
                        "relevance_score": 0.92
                    }
                ]
            }
        }


# ============= Advanced Agent Schemas =============

class GenerationConfig(BaseModel):
    """Configuration for response generation"""
    model: Optional[str] = Field(
        default=None,
        description="Model to use for generation"
    )
    temperature: float = Field(
        default=0.7,
        ge=0.0,
        le=2.0,
        description="Sampling temperature"
    )
    max_tokens: int = Field(
        default=1024,
        ge=1,
        le=32000,
        description="Maximum tokens to generate"
    )
    stream: bool = Field(
        default=False,
        description="Whether to stream the response"
    )


class AgentSearchConfig(BaseModel):
    """Search configuration for agent context retrieval"""
    search_type: SearchType = Field(default=SearchType.HYBRID)
    databases: List[str] = Field(default=["media_db"])
    keywords: Optional[List[str]] = None
    limit: int = Field(default=10, ge=1, le=50)


class AdvancedAgentRequest(BaseModel):
    """Advanced agent request with full control"""
    message: str = Field(..., min_length=1, max_length=4000)
    conversation_id: Optional[str] = None
    mode: AgentMode = Field(
        default=AgentMode.RAG,
        description="RAG for simple Q&A, research for multi-step reasoning"
    )
    generation_config: Optional[GenerationConfig] = None
    search_config: Optional[AgentSearchConfig] = None
    tools: Optional[List[ResearchTool]] = Field(
        default=None,
        description="Tools available for research mode"
    )
    system_prompt: Optional[str] = Field(
        default=None,
        description="Custom system prompt"
    )


class AdvancedAgentResponse(BaseModel):
    """Advanced agent response with additional details"""
    response: str
    conversation_id: str
    sources: List[Source]
    mode_used: AgentMode
    tools_used: Optional[List[str]] = None
    search_stats: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Statistics about the search process"
    )
    generation_stats: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Statistics about the generation process"
    )


# ============= Error Response =============

class ErrorResponse(BaseModel):
    """Standard error response"""
    error: str = Field(..., description="Error message")
    detail: Optional[str] = Field(None, description="Detailed error information")
    code: Optional[str] = Field(None, description="Error code")
    
    class Config:
        json_schema_extra = {
            "example": {
                "error": "Invalid search type",
                "detail": "Search type 'invalid' is not supported. Use: hybrid, semantic, or fulltext",
                "code": "INVALID_SEARCH_TYPE"
            }
        }