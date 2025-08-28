# parent_retrieval.py
"""
Parent document retrieval system for the RAG service.

This module provides hierarchical document retrieval with parent-child relationships,
sibling chunk retrieval, and context expansion for improved retrieval quality.
"""

import asyncio
from typing import List, Dict, Any, Optional, Set, Tuple
from dataclasses import dataclass, field
from enum import Enum
import hashlib
from collections import defaultdict

from loguru import logger
import numpy as np

from .types import Document


class ExpansionStrategy(Enum):
    """Strategies for expanding document context."""
    PARENT_ONLY = "parent_only"           # Only retrieve parent document
    SIBLINGS = "siblings"                  # Retrieve sibling chunks
    WINDOW = "window"                      # Retrieve surrounding chunks
    HIERARCHICAL = "hierarchical"          # Multi-level parent retrieval
    SEMANTIC_FAMILY = "semantic_family"    # Retrieve semantically similar chunks from same parent


@dataclass
class ParentDocument:
    """A parent document containing multiple chunks."""
    id: str
    title: str
    source: str
    chunks: List[str]           # Chunk IDs
    chunk_positions: Dict[str, int]  # Chunk ID to position mapping
    metadata: Dict[str, Any] = field(default_factory=dict)
    embedding: Optional[np.ndarray] = None
    
    def get_chunk_window(self, chunk_id: str, window_size: int = 2) -> List[str]:
        """Get surrounding chunks within window."""
        if chunk_id not in self.chunk_positions:
            return [chunk_id]
        
        pos = self.chunk_positions[chunk_id]
        start = max(0, pos - window_size)
        end = min(len(self.chunks), pos + window_size + 1)
        
        return self.chunks[start:end]
    
    def get_siblings(self, chunk_id: str) -> List[str]:
        """Get all sibling chunks (excluding the given chunk)."""
        return [cid for cid in self.chunks if cid != chunk_id]


@dataclass
class RetrievalContext:
    """Context for parent document retrieval."""
    query: str
    initial_chunks: List[Document]
    expanded_chunks: List[Document] = field(default_factory=list)
    parent_documents: List[ParentDocument] = field(default_factory=list)
    expansion_strategy: ExpansionStrategy = ExpansionStrategy.SIBLINGS
    metadata: Dict[str, Any] = field(default_factory=dict)


class ParentDocumentIndex:
    """Index for managing parent-child document relationships."""
    
    def __init__(self):
        """Initialize the parent document index."""
        self.parent_docs: Dict[str, ParentDocument] = {}
        self.chunk_to_parent: Dict[str, str] = {}
        self.parent_embeddings: Dict[str, np.ndarray] = {}
        
    def add_parent_document(
        self,
        parent_id: str,
        title: str,
        source: str,
        chunks: List[Document],
        parent_embedding: Optional[np.ndarray] = None
    ) -> ParentDocument:
        """Add a parent document with its chunks to the index."""
        chunk_ids = [chunk.id for chunk in chunks]
        chunk_positions = {chunk.id: i for i, chunk in enumerate(chunks)}
        
        parent_doc = ParentDocument(
            id=parent_id,
            title=title,
            source=source,
            chunks=chunk_ids,
            chunk_positions=chunk_positions,
            embedding=parent_embedding
        )
        
        # Store parent document
        self.parent_docs[parent_id] = parent_doc
        
        # Map chunks to parent
        for chunk_id in chunk_ids:
            self.chunk_to_parent[chunk_id] = parent_id
        
        # Store embedding if provided
        if parent_embedding is not None:
            self.parent_embeddings[parent_id] = parent_embedding
        
        logger.debug(f"Added parent document {parent_id} with {len(chunks)} chunks")
        
        return parent_doc
    
    def get_parent(self, chunk_id: str) -> Optional[ParentDocument]:
        """Get parent document for a chunk."""
        parent_id = self.chunk_to_parent.get(chunk_id)
        if parent_id:
            return self.parent_docs.get(parent_id)
        return None
    
    def get_parent_by_id(self, parent_id: str) -> Optional[ParentDocument]:
        """Get parent document by ID."""
        return self.parent_docs.get(parent_id)
    
    def build_from_documents(
        self,
        documents: List[Document],
        group_by: str = "source"
    ) -> None:
        """Build index from a list of documents."""
        # Group documents by parent
        parent_groups = defaultdict(list)
        
        for doc in documents:
            # Determine parent ID based on grouping strategy
            if group_by == "source":
                parent_key = doc.metadata.get("source", "unknown")
            elif group_by == "title":
                parent_key = doc.metadata.get("title", "unknown")
            elif group_by == "parent_id":
                parent_key = doc.metadata.get("parent_id", doc.id)
            else:
                parent_key = doc.id
            
            parent_groups[parent_key].append(doc)
        
        # Create parent documents
        for parent_key, chunks in parent_groups.items():
            parent_id = self._generate_parent_id(parent_key)
            
            # Extract common metadata
            title = chunks[0].metadata.get("title", parent_key)
            source = chunks[0].metadata.get("source", parent_key)
            
            self.add_parent_document(
                parent_id=parent_id,
                title=title,
                source=source,
                chunks=chunks
            )
        
        logger.info(f"Built index with {len(self.parent_docs)} parent documents")
    
    def _generate_parent_id(self, key: str) -> str:
        """Generate a unique parent document ID."""
        return hashlib.md5(key.encode()).hexdigest()[:12]


class ParentDocumentRetriever:
    """Retrieves and expands documents using parent-child relationships."""
    
    def __init__(
        self,
        index: Optional[ParentDocumentIndex] = None,
        default_strategy: ExpansionStrategy = ExpansionStrategy.SIBLINGS,
        max_expansion_factor: int = 3,
        diversity_threshold: float = 0.3
    ):
        """
        Initialize parent document retriever.
        
        Args:
            index: Parent document index
            default_strategy: Default expansion strategy
            max_expansion_factor: Maximum expansion factor for chunks
            diversity_threshold: Threshold for diversity scoring
        """
        self.index = index or ParentDocumentIndex()
        self.default_strategy = default_strategy
        self.max_expansion_factor = max_expansion_factor
        self.diversity_threshold = diversity_threshold
    
    async def retrieve_with_parents(
        self,
        query: str,
        initial_documents: List[Document],
        strategy: Optional[ExpansionStrategy] = None,
        window_size: int = 2,
        max_siblings: int = 5
    ) -> RetrievalContext:
        """
        Retrieve documents with parent context expansion.
        
        Args:
            query: Search query
            initial_documents: Initial retrieved documents
            strategy: Expansion strategy to use
            window_size: Window size for WINDOW strategy
            max_siblings: Maximum siblings for SIBLINGS strategy
            
        Returns:
            RetrievalContext with expanded documents
        """
        strategy = strategy or self.default_strategy
        
        context = RetrievalContext(
            query=query,
            initial_chunks=initial_documents,
            expansion_strategy=strategy
        )
        
        # Build index if not already built
        if not self.index.parent_docs:
            self.index.build_from_documents(initial_documents)
        
        # Expand based on strategy
        if strategy == ExpansionStrategy.PARENT_ONLY:
            await self._expand_parent_only(context)
        elif strategy == ExpansionStrategy.SIBLINGS:
            await self._expand_siblings(context, max_siblings)
        elif strategy == ExpansionStrategy.WINDOW:
            await self._expand_window(context, window_size)
        elif strategy == ExpansionStrategy.HIERARCHICAL:
            await self._expand_hierarchical(context)
        elif strategy == ExpansionStrategy.SEMANTIC_FAMILY:
            await self._expand_semantic_family(context)
        
        # Apply diversity scoring
        await self._apply_diversity_scoring(context)
        
        # Log expansion results
        logger.info(
            f"Expanded {len(context.initial_chunks)} chunks to "
            f"{len(context.expanded_chunks)} using {strategy.value}"
        )
        
        return context
    
    async def _expand_parent_only(self, context: RetrievalContext) -> None:
        """Expand by retrieving full parent documents."""
        expanded_chunks = []
        seen_parents = set()
        
        for chunk in context.initial_chunks:
            parent = self.index.get_parent(chunk.id)
            
            if parent and parent.id not in seen_parents:
                seen_parents.add(parent.id)
                context.parent_documents.append(parent)
                
                # Create a single document from parent
                parent_doc = Document(
                    id=parent.id,
                    content=f"[Parent Document: {parent.title}]\n",
                    metadata={
                        "type": "parent",
                        "title": parent.title,
                        "source": parent.source,
                        "num_chunks": len(parent.chunks)
                    },
                    score=chunk.score * 0.9  # Slightly lower score for parent
                )
                expanded_chunks.append(parent_doc)
        
        # Add original chunks
        expanded_chunks.extend(context.initial_chunks)
        context.expanded_chunks = expanded_chunks
    
    async def _expand_siblings(
        self,
        context: RetrievalContext,
        max_siblings: int = 5
    ) -> None:
        """Expand by retrieving sibling chunks."""
        expanded_chunks = list(context.initial_chunks)
        seen_chunks = {chunk.id for chunk in context.initial_chunks}
        
        for chunk in context.initial_chunks:
            parent = self.index.get_parent(chunk.id)
            
            if parent:
                siblings = parent.get_siblings(chunk.id)[:max_siblings]
                
                for sibling_id in siblings:
                    if sibling_id not in seen_chunks:
                        # Create sibling document
                        sibling_doc = Document(
                            id=sibling_id,
                            content="",  # Content would be retrieved from storage
                            metadata={
                                "type": "sibling",
                                "parent_id": parent.id,
                                "original_chunk": chunk.id
                            },
                            score=chunk.score * 0.7  # Lower score for siblings
                        )
                        expanded_chunks.append(sibling_doc)
                        seen_chunks.add(sibling_id)
        
        context.expanded_chunks = expanded_chunks
    
    async def _expand_window(
        self,
        context: RetrievalContext,
        window_size: int = 2
    ) -> None:
        """Expand by retrieving surrounding chunks within a window."""
        expanded_chunks = list(context.initial_chunks)
        seen_chunks = {chunk.id for chunk in context.initial_chunks}
        
        for chunk in context.initial_chunks:
            parent = self.index.get_parent(chunk.id)
            
            if parent:
                window_chunks = parent.get_chunk_window(chunk.id, window_size)
                
                for window_chunk_id in window_chunks:
                    if window_chunk_id not in seen_chunks:
                        # Calculate distance-based score
                        chunk_pos = parent.chunk_positions[chunk.id]
                        window_pos = parent.chunk_positions[window_chunk_id]
                        distance = abs(chunk_pos - window_pos)
                        distance_factor = 1.0 / (1.0 + distance * 0.2)
                        
                        window_doc = Document(
                            id=window_chunk_id,
                            content="",  # Content would be retrieved from storage
                            metadata={
                                "type": "window",
                                "parent_id": parent.id,
                                "distance": distance,
                                "original_chunk": chunk.id
                            },
                            score=chunk.score * distance_factor * 0.8
                        )
                        expanded_chunks.append(window_doc)
                        seen_chunks.add(window_chunk_id)
        
        context.expanded_chunks = expanded_chunks
    
    async def _expand_hierarchical(self, context: RetrievalContext) -> None:
        """Expand using hierarchical parent relationships."""
        expanded_chunks = list(context.initial_chunks)
        
        # Track parent levels
        parent_levels = defaultdict(list)
        
        for chunk in context.initial_chunks:
            parent = self.index.get_parent(chunk.id)
            
            if parent:
                # Level 1 parent
                parent_levels[1].append(parent)
                
                # Check for grandparent (if parent has a parent)
                grandparent_id = parent.metadata.get("parent_id")
                if grandparent_id:
                    grandparent = self.index.get_parent_by_id(grandparent_id)
                    if grandparent:
                        parent_levels[2].append(grandparent)
        
        # Add hierarchical documents
        for level, parents in parent_levels.items():
            for parent in parents:
                hierarchical_doc = Document(
                    id=f"{parent.id}_level{level}",
                    content=f"[Level {level} Parent: {parent.title}]\n",
                    metadata={
                        "type": "hierarchical",
                        "level": level,
                        "parent_id": parent.id,
                        "title": parent.title
                    },
                    score=context.initial_chunks[0].score * (0.9 ** level)
                )
                expanded_chunks.append(hierarchical_doc)
        
        context.expanded_chunks = expanded_chunks
    
    async def _expand_semantic_family(self, context: RetrievalContext) -> None:
        """Expand by retrieving semantically similar chunks from same parent."""
        expanded_chunks = list(context.initial_chunks)
        
        # Group chunks by parent
        parent_chunk_groups = defaultdict(list)
        for chunk in context.initial_chunks:
            parent = self.index.get_parent(chunk.id)
            if parent:
                parent_chunk_groups[parent.id].append(chunk)
        
        # For each parent, find semantically similar chunks
        for parent_id, chunks in parent_chunk_groups.items():
            parent = self.index.get_parent_by_id(parent_id)
            if not parent:
                continue
            
            # Calculate semantic similarity within family
            # (This would use actual embeddings in production)
            family_chunks = []
            for chunk_id in parent.chunks:
                if chunk_id not in {c.id for c in chunks}:
                    # Simulate semantic similarity score
                    similarity = np.random.random() * 0.5 + 0.5
                    
                    if similarity > 0.7:  # Threshold for semantic family
                        family_doc = Document(
                            id=chunk_id,
                            content="",  # Content would be retrieved
                            metadata={
                                "type": "semantic_family",
                                "parent_id": parent_id,
                                "similarity": similarity
                            },
                            score=chunks[0].score * similarity * 0.8
                        )
                        family_chunks.append(family_doc)
            
            # Add top semantic family members
            family_chunks.sort(key=lambda x: x.score, reverse=True)
            expanded_chunks.extend(family_chunks[:3])
        
        context.expanded_chunks = expanded_chunks
    
    async def _apply_diversity_scoring(self, context: RetrievalContext) -> None:
        """Apply diversity scoring to reduce redundancy."""
        if not context.expanded_chunks:
            return
        
        # Group chunks by parent
        parent_groups = defaultdict(list)
        for chunk in context.expanded_chunks:
            parent_id = chunk.metadata.get("parent_id", chunk.id)
            parent_groups[parent_id].append(chunk)
        
        # Apply diversity penalty within groups
        for parent_id, chunks in parent_groups.items():
            if len(chunks) > 1:
                # Sort by score
                chunks.sort(key=lambda x: x.score, reverse=True)
                
                # Apply diminishing returns penalty
                for i, chunk in enumerate(chunks[1:], 1):
                    diversity_penalty = 1.0 - (i * self.diversity_threshold)
                    diversity_penalty = max(0.5, diversity_penalty)  # Minimum 50% score
                    chunk.score *= diversity_penalty
        
        # Sort final results by adjusted score
        context.expanded_chunks.sort(key=lambda x: x.score, reverse=True)
        
        # Limit expansion based on max_expansion_factor
        max_chunks = len(context.initial_chunks) * self.max_expansion_factor
        context.expanded_chunks = context.expanded_chunks[:max_chunks]
        
        # Update metadata
        context.metadata["diversity_applied"] = True
        context.metadata["final_chunk_count"] = len(context.expanded_chunks)


class SmartParentRetriever(ParentDocumentRetriever):
    """Advanced parent retriever with intelligent strategy selection."""
    
    async def retrieve_adaptively(
        self,
        query: str,
        initial_documents: List[Document],
        query_type: Optional[str] = None
    ) -> RetrievalContext:
        """
        Adaptively select retrieval strategy based on query characteristics.
        
        Args:
            query: Search query
            initial_documents: Initial retrieved documents
            query_type: Optional query type hint
            
        Returns:
            RetrievalContext with expanded documents
        """
        # Analyze query to determine best strategy
        strategy = self._select_strategy(query, initial_documents, query_type)
        
        logger.info(f"Selected strategy: {strategy.value} for query: {query[:50]}...")
        
        # Apply selected strategy
        return await self.retrieve_with_parents(
            query=query,
            initial_documents=initial_documents,
            strategy=strategy
        )
    
    def _select_strategy(
        self,
        query: str,
        documents: List[Document],
        query_type: Optional[str] = None
    ) -> ExpansionStrategy:
        """Select best expansion strategy based on query and documents."""
        # Use hints if provided
        if query_type:
            if query_type == "detailed":
                return ExpansionStrategy.HIERARCHICAL
            elif query_type == "contextual":
                return ExpansionStrategy.WINDOW
            elif query_type == "exploratory":
                return ExpansionStrategy.SEMANTIC_FAMILY
        
        # Analyze query length
        query_words = len(query.split())
        if query_words > 10:  # Long, detailed query
            return ExpansionStrategy.HIERARCHICAL
        
        # Analyze document scores
        if documents:
            avg_score = np.mean([doc.score for doc in documents])
            if avg_score < 0.5:  # Low confidence results
                return ExpansionStrategy.SEMANTIC_FAMILY
            elif avg_score > 0.8:  # High confidence results
                return ExpansionStrategy.WINDOW
        
        # Default to siblings for general queries
        return ExpansionStrategy.SIBLINGS


# Pipeline integration function
async def expand_with_parent_context(context: Any, **kwargs) -> Any:
    """Expand documents with parent context for pipeline."""
    config = context.config.get("parent_retrieval", {})
    
    # Check if parent retrieval is enabled
    if not config.get("enabled", True):
        return context
    
    # Create retriever
    retriever = SmartParentRetriever(
        default_strategy=ExpansionStrategy[config.get("strategy", "SIBLINGS").upper()],
        max_expansion_factor=config.get("max_expansion", 3),
        diversity_threshold=config.get("diversity_threshold", 0.3)
    )
    
    # Retrieve with parent expansion
    retrieval_context = await retriever.retrieve_adaptively(
        query=context.query,
        initial_documents=context.documents,
        query_type=config.get("query_type")
    )
    
    # Update context
    context.documents = retrieval_context.expanded_chunks
    context.metadata["parent_retrieval"] = {
        "strategy": retrieval_context.expansion_strategy.value,
        "initial_docs": len(retrieval_context.initial_chunks),
        "expanded_docs": len(retrieval_context.expanded_chunks),
        "parent_docs": len(retrieval_context.parent_documents)
    }
    
    logger.info(
        f"Expanded {len(retrieval_context.initial_chunks)} to "
        f"{len(retrieval_context.expanded_chunks)} documents"
    )
    
    return context