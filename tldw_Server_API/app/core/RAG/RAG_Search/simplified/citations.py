"""
Citations support for RAG search results.

This module provides data models and utilities for managing citations in RAG search results,
allowing precise source attribution with document references, text snippets, and confidence scores.
"""

from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any
from enum import Enum
import json


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
    
    @classmethod
    def from_dict(cls, data: dict) -> 'Citation':
        """Create Citation from dictionary."""
        return cls(
            document_id=data["document_id"],
            document_title=data["document_title"],
            chunk_id=data["chunk_id"],
            text=data["text"],
            start_char=data["start_char"],
            end_char=data["end_char"],
            confidence=data["confidence"],
            match_type=CitationType(data["match_type"]),
            metadata=data.get("metadata", {})
        )
    
    def format_citation(self, style: str = "inline") -> str:
        """
        Format citation for display.
        
        Args:
            style: Citation style - one of:
                - "inline": Brief inline citation
                - "footnote": Footnote style
                - "academic": Academic style (author, date)
                - "full": Full citation with all details
            
        Returns:
            Formatted citation string
        """
        if style == "inline":
            return f"[{self.document_title}, chars {self.start_char}-{self.end_char}]"
        elif style == "footnote":
            return f"{self.document_title} ({self.match_type.value} match, {self.confidence:.0%} confidence)"
        elif style == "academic":
            author = self.metadata.get("author", "Unknown")
            date = self.metadata.get("date", "n.d.")
            return f"({author}, {date})"
        elif style == "full":
            return (f"{self.document_title} [{self.document_id}:{self.chunk_id}] "
                   f"(chars {self.start_char}-{self.end_char}, "
                   f"{self.match_type.value} match, {self.confidence:.0%} confidence)")
        else:
            return f"[{self.document_id}:{self.chunk_id}]"
    
    def __str__(self) -> str:
        """String representation."""
        return self.format_citation("inline")


@dataclass
class SearchResultWithCitations:
    """Enhanced search result that includes citations."""
    id: str
    score: float
    document: str
    metadata: dict = field(default_factory=dict)
    citations: List[Citation] = field(default_factory=list)
    
    def get_unique_sources(self) -> List[str]:
        """Get list of unique source documents cited."""
        return list(set(c.document_id for c in self.citations))
    
    def get_citations_by_type(self, citation_type: CitationType) -> List[Citation]:
        """Get citations filtered by type."""
        return [c for c in self.citations if c.match_type == citation_type]
    
    def get_highest_confidence_citation(self) -> Optional[Citation]:
        """Get the citation with highest confidence score."""
        if not self.citations:
            return None
        return max(self.citations, key=lambda c: c.confidence)
    
    def format_with_citations(self, 
                            style: str = "inline", 
                            max_citations: Optional[int] = None) -> str:
        """
        Format the result with inline citations.
        
        Args:
            style: Citation formatting style
            max_citations: Maximum number of citations to include (None for all)
        
        Returns:
            Document text with citations inserted at appropriate points
        """
        # Get citations to include
        citations_to_use = self.citations
        if max_citations is not None and len(citations_to_use) > max_citations:
            # Sort by confidence and take top N
            citations_to_use = sorted(citations_to_use, key=lambda c: c.confidence, reverse=True)[:max_citations]
        
        # Format citations
        if not citations_to_use:
            return self.document
        
        # Group citations by type for better formatting
        exact_citations = [c for c in citations_to_use if c.match_type == CitationType.EXACT]
        other_citations = [c for c in citations_to_use if c.match_type != CitationType.EXACT]
        
        # Build citation text
        citation_parts = []
        
        # Add exact matches first (if any)
        if exact_citations:
            exact_refs = [c.format_citation(style) for c in exact_citations]
            citation_parts.append(f"Exact matches: {', '.join(exact_refs)}")
        
        # Add other citations
        if other_citations:
            other_refs = [c.format_citation(style) for c in other_citations]
            citation_parts.append(f"Related: {', '.join(other_refs)}")
        
        # Combine document with citations
        citation_text = " | ".join(citation_parts) if citation_parts else ""
        return f"{self.document}\n\nSources: {citation_text}" if citation_text else self.document
    
    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            "id": self.id,
            "score": self.score,
            "document": self.document,
            "metadata": self.metadata,
            "citations": [c.to_dict() for c in self.citations]
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> 'SearchResultWithCitations':
        """Create SearchResultWithCitations from dictionary."""
        return cls(
            id=data["id"],
            score=data["score"],
            document=data["document"],
            metadata=data.get("metadata", {}),
            citations=[Citation.from_dict(c) for c in data.get("citations", [])]
        )
    
    def to_json(self) -> str:
        """Convert to JSON string."""
        return json.dumps(self.to_dict(), indent=2)
    
    @classmethod
    def from_json(cls, json_str: str) -> 'SearchResultWithCitations':
        """Create from JSON string."""
        return cls.from_dict(json.loads(json_str))


# Utility functions for working with citations

def merge_citations(citations_list: List[List[Citation]]) -> List[Citation]:
    """
    Merge multiple lists of citations, removing duplicates.
    
    Duplicates are identified by (document_id, chunk_id, start_char, end_char).
    When duplicates are found, the one with highest confidence is kept.
    """
    citation_map = {}
    
    for citations in citations_list:
        for citation in citations:
            key = (citation.document_id, citation.chunk_id, citation.start_char, citation.end_char)
            
            if key not in citation_map or citation.confidence > citation_map[key].confidence:
                citation_map[key] = citation
    
    return list(citation_map.values())


def group_citations_by_document(citations: List[Citation]) -> Dict[str, List[Citation]]:
    """Group citations by document ID."""
    groups = {}
    for citation in citations:
        if citation.document_id not in groups:
            groups[citation.document_id] = []
        groups[citation.document_id].append(citation)
    return groups


def filter_overlapping_citations(citations: List[Citation], 
                               prefer_exact: bool = True) -> List[Citation]:
    """
    Filter out overlapping citations from the same document.
    
    Args:
        citations: List of citations to filter
        prefer_exact: If True, prefer EXACT match citations over others
    
    Returns:
        Filtered list with no overlapping citations
    """
    # Group by document
    doc_groups = group_citations_by_document(citations)
    
    filtered = []
    for doc_id, doc_citations in doc_groups.items():
        # Sort by start position and confidence
        sorted_citations = sorted(
            doc_citations, 
            key=lambda c: (c.start_char, -c.confidence, 
                          0 if c.match_type == CitationType.EXACT else 1)
        )
        
        # Keep non-overlapping citations
        kept_citations = []
        for citation in sorted_citations:
            # Check if overlaps with any kept citation
            overlaps = False
            for kept in kept_citations:
                if (citation.start_char < kept.end_char and 
                    citation.end_char > kept.start_char):
                    overlaps = True
                    break
            
            if not overlaps:
                kept_citations.append(citation)
        
        filtered.extend(kept_citations)
    
    return filtered