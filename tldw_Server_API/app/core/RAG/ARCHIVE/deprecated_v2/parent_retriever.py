"""
Parent document retriever wrapper for context expansion.

This module provides a wrapper that adds parent document retrieval capabilities
to existing retrieval strategies, allowing for expanded context windows.
"""

from typing import List, Dict, Any, Optional, Set
from dataclasses import dataclass
import asyncio

from loguru import logger

from .types import (
    RetrieverStrategy, SearchResult, Document, DataSource,
    EnhancedSearchResult
)


class ParentDocumentRetriever(RetrieverStrategy):
    """
    Wrapper that adds parent document retrieval for context expansion.
    
    This wrapper:
    - Delegates initial retrieval to the base retriever
    - Fetches parent documents for retrieved chunks
    - Expands context with sibling chunks
    - Maintains relationships between chunks and parents
    """
    
    def __init__(
        self,
        base_retriever: RetrieverStrategy,
        parent_store: Dict[str, Document],  # In production, this would be a database
        parent_size_multiplier: int = 3,
        expand_to_siblings: bool = True,
        max_parent_docs: int = 5
    ):
        """
        Initialize parent document retriever.
        
        Args:
            base_retriever: The underlying retriever to wrap
            parent_store: Storage for parent documents (dict for now, DB in production)
            parent_size_multiplier: How much larger parent docs are than chunks
            expand_to_siblings: Whether to include sibling chunks
            max_parent_docs: Maximum number of parent documents to retrieve
        """
        self.base_retriever = base_retriever
        self.parent_store = parent_store
        self.parent_size_multiplier = parent_size_multiplier
        self.expand_to_siblings = expand_to_siblings
        self.max_parent_docs = max_parent_docs
        
        logger.info(
            f"Initialized ParentDocumentRetriever with multiplier={parent_size_multiplier}, "
            f"siblings={expand_to_siblings}"
        )
    
    @property
    def source_type(self) -> DataSource:
        """Delegate to base retriever."""
        return self.base_retriever.source_type
    
    async def retrieve(
        self,
        query: str,
        filters: Optional[Dict[str, Any]] = None,
        top_k: int = 10
    ) -> SearchResult:
        """
        Retrieve documents with parent document expansion.
        
        Args:
            query: The search query
            filters: Optional filters to apply
            top_k: Number of results to return
            
        Returns:
            EnhancedSearchResult with parent documents included
        """
        # Get results from base retriever
        result = await self.base_retriever.retrieve(query, filters, top_k)
        
        # Expand with parent documents
        enhanced_result = await self._expand_with_parents(result)
        
        return enhanced_result
    
    async def _expand_with_parents(self, result: SearchResult) -> EnhancedSearchResult:
        """
        Expand search results with parent documents.
        
        Args:
            result: Original search result
            
        Returns:
            Enhanced search result with parent documents
        """
        parent_documents = []
        parent_ids_seen = set()
        expanded_context_parts = []
        
        # Process each retrieved document
        for doc in result.documents[:self.max_parent_docs]:
            if doc.parent_id and doc.parent_id not in parent_ids_seen:
                parent_doc = await self._fetch_parent_document(doc.parent_id)
                if parent_doc:
                    parent_documents.append(parent_doc)
                    parent_ids_seen.add(doc.parent_id)
                    
                    # Add parent content to expanded context
                    expanded_context_parts.append(
                        f"[Parent Document: {parent_doc.metadata.get('title', 'Untitled')}]\n"
                        f"{parent_doc.content}\n"
                    )
                    
                    # Optionally fetch sibling chunks
                    if self.expand_to_siblings:
                        siblings = await self._fetch_sibling_chunks(doc, parent_doc)
                        for sibling in siblings:
                            if sibling.id not in {d.id for d in result.documents}:
                                result.documents.append(sibling)
        
        # Create enhanced result
        enhanced_result = EnhancedSearchResult(
            documents=result.documents,
            query=result.query,
            search_type=result.search_type,
            metadata=result.metadata or {},
            citations=getattr(result, 'citations', []),
            expanded_context="\n\n".join(expanded_context_parts) if expanded_context_parts else None,
            query_variations=getattr(result, 'query_variations', []),
            parent_documents=parent_documents,
            reranked=False,
            diversity_score=self._calculate_diversity_score(result.documents)
        )
        
        logger.debug(
            f"Expanded with {len(parent_documents)} parent documents, "
            f"total documents: {len(enhanced_result.documents)}"
        )
        
        return enhanced_result
    
    async def _fetch_parent_document(self, parent_id: str) -> Optional[Document]:
        """
        Fetch a parent document by ID.
        
        Args:
            parent_id: ID of the parent document
            
        Returns:
            Parent document or None if not found
        """
        # In production, this would be a database query
        # For now, using the in-memory store
        return self.parent_store.get(parent_id)
    
    async def _fetch_sibling_chunks(
        self,
        chunk: Document,
        parent: Document
    ) -> List[Document]:
        """
        Fetch sibling chunks of a given chunk.
        
        Args:
            chunk: The chunk to find siblings for
            parent: The parent document
            
        Returns:
            List of sibling chunks
        """
        siblings = []
        
        if not parent.children_ids:
            return siblings
        
        # Get chunks before and after the current chunk
        if chunk.chunk_index is not None:
            target_indices = [
                chunk.chunk_index - 1,  # Previous chunk
                chunk.chunk_index + 1,  # Next chunk
            ]
            
            for child_id in parent.children_ids:
                if child_id != chunk.id and child_id in self.parent_store:
                    sibling = self.parent_store[child_id]
                    if sibling.chunk_index in target_indices:
                        siblings.append(sibling)
        
        return siblings
    
    def _calculate_diversity_score(self, documents: List[Document]) -> float:
        """
        Calculate diversity score based on unique sources and parent documents.
        
        Args:
            documents: List of documents
            
        Returns:
            Diversity score between 0 and 1
        """
        if not documents:
            return 0.0
        
        unique_sources = len(set(doc.source for doc in documents))
        unique_parents = len(set(doc.parent_id for doc in documents if doc.parent_id))
        unique_docs = len(set(doc.id for doc in documents))
        
        # Calculate diversity as ratio of unique elements
        diversity = (unique_sources + unique_parents) / (len(documents) * 2)
        
        return min(1.0, diversity)


class HierarchicalRetriever(RetrieverStrategy):
    """
    Advanced retriever that maintains document hierarchy.
    
    This retriever understands document structure and can retrieve
    at different levels of granularity (document, section, paragraph).
    """
    
    def __init__(
        self,
        base_retriever: RetrieverStrategy,
        hierarchy_store: Dict[str, Dict[str, Any]],
        retrieval_depth: int = 2
    ):
        """
        Initialize hierarchical retriever.
        
        Args:
            base_retriever: The underlying retriever
            hierarchy_store: Store for document hierarchy information
            retrieval_depth: How many levels up/down to retrieve
        """
        self.base_retriever = base_retriever
        self.hierarchy_store = hierarchy_store
        self.retrieval_depth = retrieval_depth
    
    @property
    def source_type(self) -> DataSource:
        """Delegate to base retriever."""
        return self.base_retriever.source_type
    
    async def retrieve(
        self,
        query: str,
        filters: Optional[Dict[str, Any]] = None,
        top_k: int = 10
    ) -> SearchResult:
        """
        Retrieve documents with hierarchical context.
        
        Args:
            query: The search query
            filters: Optional filters to apply
            top_k: Number of results to return
            
        Returns:
            SearchResult with hierarchical context
        """
        # Get base results
        result = await self.base_retriever.retrieve(query, filters, top_k)
        
        # Expand based on hierarchy
        expanded_docs = []
        seen_ids = set()
        
        for doc in result.documents:
            # Add the document itself
            if doc.id not in seen_ids:
                expanded_docs.append(doc)
                seen_ids.add(doc.id)
            
            # Get hierarchical context
            hierarchy_info = self.hierarchy_store.get(doc.id, {})
            
            # Add ancestors (sections, chapters)
            ancestors = hierarchy_info.get('ancestors', [])
            for ancestor_id in ancestors[:self.retrieval_depth]:
                if ancestor_id not in seen_ids and ancestor_id in self.hierarchy_store:
                    ancestor_doc = await self._fetch_document(ancestor_id)
                    if ancestor_doc:
                        expanded_docs.append(ancestor_doc)
                        seen_ids.add(ancestor_id)
            
            # Add descendants (sub-sections, paragraphs)
            descendants = hierarchy_info.get('descendants', [])
            for descendant_id in descendants[:self.retrieval_depth]:
                if descendant_id not in seen_ids and descendant_id in self.hierarchy_store:
                    descendant_doc = await self._fetch_document(descendant_id)
                    if descendant_doc:
                        expanded_docs.append(descendant_doc)
                        seen_ids.add(descendant_id)
        
        # Update result with expanded documents
        result.documents = expanded_docs
        
        logger.debug(f"Hierarchical expansion: {len(result.documents)} documents total")
        
        return result
    
    async def _fetch_document(self, doc_id: str) -> Optional[Document]:
        """Fetch a document by ID from the hierarchy store."""
        # In production, this would be a database query
        doc_data = self.hierarchy_store.get(doc_id)
        if doc_data:
            return Document(
                id=doc_id,
                content=doc_data.get('content', ''),
                metadata=doc_data.get('metadata', {}),
                source=DataSource.MEDIA_DB,  # Default, should be from doc_data
                parent_id=doc_data.get('parent_id'),
                children_ids=doc_data.get('children_ids', []),
                chunk_index=doc_data.get('chunk_index')
            )
        return None