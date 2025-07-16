"""
Data models for RAG system.

This module contains shared data structures used across the RAG system
to avoid circular imports.
"""

from dataclasses import dataclass
from typing import Optional


@dataclass
class IndexingResult:
    """Result of indexing operation."""
    doc_id: str
    chunks_created: int
    time_taken: float
    success: bool = True
    error: Optional[str] = None
    
    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "doc_id": self.doc_id,
            "chunks_created": self.chunks_created,
            "time_taken": self.time_taken,
            "success": self.success,
            "error": self.error
        }