"""
Citation-aware retriever wrapper for adding citation support to any retriever.

This module provides a wrapper that adds citation generation capabilities
to existing retrieval strategies.
"""

import re
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
import difflib

from loguru import logger

from .types import (
    RetrieverStrategy, SearchResult, Document, Citation, 
    CitationType, DataSource
)


class CitationAwareRetriever(RetrieverStrategy):
    """
    Wrapper that adds citation generation to any retriever.
    
    This wrapper:
    - Delegates retrieval to the base retriever
    - Generates citations for retrieved documents
    - Tracks character positions for precise references
    - Supports multiple citation types (exact, semantic, fuzzy, keyword)
    """
    
    def __init__(
        self,
        base_retriever: RetrieverStrategy,
        max_citation_length: int = 200,
        citation_context_chars: int = 50,
        enable_fuzzy_matching: bool = True,
        fuzzy_threshold: float = 0.8
    ):
        """
        Initialize citation-aware retriever.
        
        Args:
            base_retriever: The underlying retriever to wrap
            max_citation_length: Maximum length of citation text
            citation_context_chars: Characters of context around citation
            enable_fuzzy_matching: Whether to use fuzzy matching for citations
            fuzzy_threshold: Threshold for fuzzy matching (0-1)
        """
        self.base_retriever = base_retriever
        self.max_citation_length = max_citation_length
        self.citation_context_chars = citation_context_chars
        self.enable_fuzzy_matching = enable_fuzzy_matching
        self.fuzzy_threshold = fuzzy_threshold
        
        logger.info(f"Initialized CitationAwareRetriever wrapping {type(base_retriever).__name__}")
    
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
        Retrieve documents with citation generation.
        
        Args:
            query: The search query
            filters: Optional filters to apply
            top_k: Number of results to return
            
        Returns:
            SearchResult with citations added to documents
        """
        # Get results from base retriever
        result = await self.base_retriever.retrieve(query, filters, top_k)
        
        # Generate citations for each document
        for doc in result.documents:
            citations = self._generate_citations(query, doc)
            doc.citations.extend(citations)
        
        # Add citations to the result
        all_citations = []
        for doc in result.documents:
            all_citations.extend(doc.citations)
        result.citations = all_citations
        
        logger.debug(f"Generated {len(all_citations)} citations for {len(result.documents)} documents")
        
        return result
    
    def _generate_citations(self, query: str, document: Document) -> List[Citation]:
        """
        Generate citations for a document based on the query.
        
        Args:
            query: The search query
            document: The document to generate citations for
            
        Returns:
            List of citations found in the document
        """
        citations = []
        
        # Extract query terms for keyword matching
        query_terms = self._extract_query_terms(query)
        
        # 1. Check for exact matches
        exact_citations = self._find_exact_matches(query, document)
        citations.extend(exact_citations)
        
        # 2. Check for keyword matches
        keyword_citations = self._find_keyword_matches(query_terms, document)
        citations.extend(keyword_citations)
        
        # 3. Check for fuzzy matches if enabled
        if self.enable_fuzzy_matching and not exact_citations:
            fuzzy_citations = self._find_fuzzy_matches(query, document)
            citations.extend(fuzzy_citations)
        
        # 4. Add semantic citation if document has high score but no other citations
        if not citations and document.score > 0.7:
            semantic_citation = self._create_semantic_citation(document)
            citations.append(semantic_citation)
        
        return citations
    
    def _extract_query_terms(self, query: str) -> List[str]:
        """Extract significant terms from query."""
        # Remove common stop words and split
        stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'is', 'was', 'are', 'were'}
        terms = query.lower().split()
        return [term for term in terms if term not in stop_words and len(term) > 2]
    
    def _find_exact_matches(self, query: str, document: Document) -> List[Citation]:
        """Find exact query matches in document."""
        citations = []
        content_lower = document.content.lower()
        query_lower = query.lower()
        
        # Find all occurrences of the exact query
        start = 0
        while True:
            pos = content_lower.find(query_lower, start)
            if pos == -1:
                break
            
            # Extract citation text with context
            citation_start = max(0, pos - self.citation_context_chars)
            citation_end = min(len(document.content), pos + len(query) + self.citation_context_chars)
            citation_text = document.content[citation_start:citation_end]
            
            # Adjust character positions
            actual_start = pos
            actual_end = pos + len(query)
            
            citation = Citation(
                document_id=document.id,
                document_title=document.metadata.get('title', 'Untitled'),
                chunk_id=document.id,
                text=citation_text[:self.max_citation_length],
                start_char=actual_start,
                end_char=actual_end,
                confidence=1.0,
                match_type=CitationType.EXACT,
                metadata={'query': query}
            )
            citations.append(citation)
            
            start = pos + 1
            
            # Limit to 3 exact matches per document
            if len(citations) >= 3:
                break
        
        return citations
    
    def _find_keyword_matches(self, query_terms: List[str], document: Document) -> List[Citation]:
        """Find keyword matches in document."""
        citations = []
        content_lower = document.content.lower()
        
        for term in query_terms:
            # Find first occurrence of each term
            pos = content_lower.find(term.lower())
            if pos != -1:
                # Extract citation text with context
                citation_start = max(0, pos - self.citation_context_chars)
                citation_end = min(len(document.content), pos + len(term) + self.citation_context_chars)
                citation_text = document.content[citation_start:citation_end]
                
                citation = Citation(
                    document_id=document.id,
                    document_title=document.metadata.get('title', 'Untitled'),
                    chunk_id=document.id,
                    text=citation_text[:self.max_citation_length],
                    start_char=pos,
                    end_char=pos + len(term),
                    confidence=0.7,
                    match_type=CitationType.KEYWORD,
                    metadata={'matched_term': term}
                )
                citations.append(citation)
        
        return citations[:5]  # Limit keyword citations
    
    def _find_fuzzy_matches(self, query: str, document: Document) -> List[Citation]:
        """Find fuzzy matches using sequence matching."""
        citations = []
        
        # Split document into sentences for fuzzy matching
        sentences = re.split(r'[.!?]\s+', document.content)
        
        for i, sentence in enumerate(sentences):
            # Calculate similarity ratio
            ratio = difflib.SequenceMatcher(None, query.lower(), sentence.lower()).ratio()
            
            if ratio >= self.fuzzy_threshold:
                # Find sentence position in document
                sentence_pos = document.content.find(sentence)
                if sentence_pos != -1:
                    citation = Citation(
                        document_id=document.id,
                        document_title=document.metadata.get('title', 'Untitled'),
                        chunk_id=document.id,
                        text=sentence[:self.max_citation_length],
                        start_char=sentence_pos,
                        end_char=sentence_pos + len(sentence),
                        confidence=ratio,
                        match_type=CitationType.FUZZY,
                        metadata={'similarity_ratio': ratio}
                    )
                    citations.append(citation)
        
        # Sort by confidence and return top matches
        citations.sort(key=lambda c: c.confidence, reverse=True)
        return citations[:3]
    
    def _create_semantic_citation(self, document: Document) -> Citation:
        """Create a semantic citation for high-scoring documents without exact matches."""
        # Take the first substantial portion of the document
        content_preview = document.content[:self.max_citation_length * 2]
        
        # Find a good break point (end of sentence if possible)
        for punct in ['. ', '? ', '! ']:
            pos = content_preview.find(punct)
            if pos != -1 and pos < self.max_citation_length:
                content_preview = content_preview[:pos + 1]
                break
        else:
            content_preview = content_preview[:self.max_citation_length]
        
        return Citation(
            document_id=document.id,
            document_title=document.metadata.get('title', 'Untitled'),
            chunk_id=document.id,
            text=content_preview,
            start_char=0,
            end_char=len(content_preview),
            confidence=document.score,
            match_type=CitationType.SEMANTIC,
            metadata={'relevance_score': document.score}
        )


def merge_citations(citations: List[Citation], max_citations: int = 10) -> List[Citation]:
    """
    Merge and deduplicate citations from multiple sources.
    
    Args:
        citations: List of citations to merge
        max_citations: Maximum number of citations to return
        
    Returns:
        Merged and deduplicated list of citations
    """
    # Group citations by document and position
    citation_map = {}
    
    for citation in citations:
        key = (citation.document_id, citation.start_char, citation.end_char)
        
        if key not in citation_map:
            citation_map[key] = citation
        else:
            # Keep the citation with higher confidence
            if citation.confidence > citation_map[key].confidence:
                citation_map[key] = citation
    
    # Sort by confidence and return top citations
    merged = list(citation_map.values())
    merged.sort(key=lambda c: c.confidence, reverse=True)
    
    return merged[:max_citations]