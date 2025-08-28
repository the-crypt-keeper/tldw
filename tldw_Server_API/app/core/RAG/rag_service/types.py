"""
Type definitions and base interfaces for the RAG service.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import List, Dict, Any, Optional, Protocol, TypeVar, Generic
import numpy as np


class DataSource(Enum):
    """Supported data sources for RAG."""
    MEDIA_DB = "media_db"
    CHAT_HISTORY = "chat_history"
    NOTES = "notes"
    CHARACTER_CARDS = "character_cards"
    WEB_CONTENT = "web_content"
    PROMPTS = "prompts"  # Add missing PROMPTS source


class CitationType(Enum):
    """Type of citation match."""
    EXACT = "exact"          # Exact phrase match
    SEMANTIC = "semantic"    # Semantic similarity
    FUZZY = "fuzzy"         # Fuzzy/partial match
    KEYWORD = "keyword"      # Keyword/FTS5 match


@dataclass
class Citation:
    """
    Represents a citation to a source document.
    
    Attributes:
        document_id: Unique identifier of the source document
        document_title: Human-readable title of the document
        chunk_id: ID of the specific chunk within the document
        text: The actual text snippet being cited
        start_char: Character offset in the original document
        end_char: End character offset in the original document
        confidence: Confidence score (0-1) for this citation
        match_type: Type of match that produced this citation
        metadata: Additional metadata (author, date, URL, etc.)
    """
    document_id: str
    document_title: str
    chunk_id: str
    text: str
    start_char: int
    end_char: int
    confidence: float
    match_type: CitationType
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """Validate citation data."""
        if self.confidence < 0 or self.confidence > 1:
            raise ValueError(f"Confidence must be between 0 and 1, got {self.confidence}")
        if self.start_char < 0:
            raise ValueError(f"start_char must be non-negative, got {self.start_char}")
        if self.end_char < self.start_char:
            raise ValueError(f"end_char must be >= start_char, got start={self.start_char}, end={self.end_char}")
    
    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            "document_id": self.document_id,
            "document_title": self.document_title,
            "chunk_id": self.chunk_id,
            "text": self.text,
            "start_char": self.start_char,
            "end_char": self.end_char,
            "confidence": self.confidence,
            "match_type": self.match_type.value,
            "metadata": self.metadata
        }


@dataclass
class Document:
    """
    Represents a document in the RAG system.
    
    This is a unified representation for all types of content
    (media transcripts, chat messages, notes, etc.)
    """
    id: str  # Unique identifier
    content: str  # The actual text content
    metadata: Dict[str, Any]  # Source-specific metadata
    source: DataSource  # Where this document came from
    score: float = 0.0  # Relevance score (set during retrieval)
    embedding: Optional[np.ndarray] = None  # Vector embedding if available
    
    # Citation support
    citations: List[Citation] = field(default_factory=list)
    
    # Parent document support
    parent_id: Optional[str] = None  # ID of parent document if this is a chunk
    children_ids: List[str] = field(default_factory=list)  # IDs of child chunks
    chunk_index: Optional[int] = None  # Position in parent document
    
    # Character positions for citation tracking
    start_char: Optional[int] = None  # Start position in original document
    end_char: Optional[int] = None  # End position in original document
    
    def __hash__(self):
        return hash(self.id)
    
    def __eq__(self, other):
        if not isinstance(other, Document):
            return False
        return self.id == other.id
    
    def add_citation(self, citation: Citation) -> None:
        """Add a citation to this document."""
        self.citations.append(citation)
    
    def get_citations_by_type(self, citation_type: CitationType) -> List[Citation]:
        """Get citations of a specific type."""
        return [c for c in self.citations if c.match_type == citation_type]


@dataclass
class SearchResult:
    """Result from a search operation."""
    documents: List[Document]
    query: str
    search_type: str  # "vector", "fts", "hybrid"
    metadata: Dict[str, Any] = None  # Additional search metadata
    
    # Enhanced features
    citations: List[Citation] = field(default_factory=list)
    expanded_context: Optional[str] = None  # Context expanded with parent documents
    query_variations: List[str] = field(default_factory=list)  # Query expansion results


@dataclass
class EnhancedSearchResult(SearchResult):
    """Enhanced search result with additional features."""
    parent_documents: List[Document] = field(default_factory=list)
    reranked: bool = False
    diversity_score: float = 0.0


@dataclass
class RAGContext:
    """Context prepared for generation."""
    documents: List[Document]
    combined_text: str
    total_tokens: int
    metadata: Dict[str, Any]
    
    # Enhanced features
    citations: List[Citation] = field(default_factory=list)
    parent_context: Optional[str] = None  # Expanded context from parent documents
    structured_sections: Dict[str, str] = field(default_factory=dict)  # Structured content


@dataclass
class RAGResponse:
    """Response from the RAG pipeline."""
    answer: str
    context: RAGContext
    sources: List[Document]
    metadata: Dict[str, Any]  # Timing, model used, etc.
    
    # Enhanced features
    citations: List[Citation] = field(default_factory=list)
    confidence_score: float = 0.0  # Overall confidence in the response


# Protocol definitions for better type checking

class Embedder(Protocol):
    """Protocol for embedding models."""
    def embed(self, text: str) -> np.ndarray:
        """Generate embedding for text."""
        ...
    
    def embed_batch(self, texts: List[str]) -> List[np.ndarray]:
        """Generate embeddings for multiple texts."""
        ...


class Reranker(Protocol):
    """Protocol for reranking models."""
    def rerank(self, query: str, documents: List[Document]) -> List[Document]:
        """Rerank documents based on relevance to query."""
        ...


T = TypeVar('T')


class Cache(Protocol, Generic[T]):
    """Protocol for cache implementations."""
    def get(self, key: str) -> Optional[T]:
        """Get item from cache."""
        ...
    
    def set(self, key: str, value: T, ttl: Optional[int] = None) -> None:
        """Set item in cache with optional TTL."""
        ...
    
    def delete(self, key: str) -> None:
        """Delete item from cache."""
        ...
    
    def clear(self) -> None:
        """Clear all items from cache."""
        ...


# Abstract base classes for strategy pattern

class RetrieverStrategy(ABC):
    """Base class for retrieval strategies."""
    
    @abstractmethod
    async def retrieve(
        self, 
        query: str, 
        filters: Optional[Dict[str, Any]] = None,
        top_k: int = 10
    ) -> SearchResult:
        """
        Retrieve relevant documents for the query.
        
        Args:
            query: The search query
            filters: Optional filters to apply
            top_k: Number of results to return
            
        Returns:
            SearchResult containing relevant documents
        """
        pass
    
    @property
    @abstractmethod
    def source_type(self) -> DataSource:
        """The data source this retriever handles."""
        pass


class ProcessingStrategy(ABC):
    """Base class for document processing strategies."""
    
    @abstractmethod
    def process(
        self,
        search_results: List[SearchResult],
        query: str,
        max_context_length: int = 4096
    ) -> RAGContext:
        """
        Process search results into a context for generation.
        
        Args:
            search_results: Results from various retrievers
            query: The original query
            max_context_length: Maximum context length in tokens
            
        Returns:
            Processed context ready for generation
        """
        pass


class GenerationStrategy(ABC):
    """Base class for generation strategies."""
    
    @abstractmethod
    async def generate(
        self,
        context: RAGContext,
        query: str,
        **kwargs
    ) -> str:
        """
        Generate response using the context.
        
        Args:
            context: The prepared context
            query: The original query
            **kwargs: Additional generation parameters
            
        Returns:
            Generated response
        """
        pass


# Exceptions

class RAGError(Exception):
    """Base exception for RAG service."""
    pass


class RetrievalError(RAGError):
    """Error during document retrieval."""
    pass


class ProcessingError(RAGError):
    """Error during document processing."""
    pass


class GenerationError(RAGError):
    """Error during response generation."""
    pass


class ConfigurationError(RAGError):
    """Error in configuration."""
    pass