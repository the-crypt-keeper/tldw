# citations.py
"""
Citation generation system for the RAG service.

This module provides citation generation with multiple matching strategies,
character-level precision, and confidence scoring.
"""

import re
import difflib
from typing import List, Dict, Any, Optional, Tuple, Set
from dataclasses import dataclass, field
from enum import Enum
import hashlib
from collections import defaultdict

import numpy as np
from loguru import logger

from .types import Document


class CitationType(Enum):
    """Types of citations based on matching strategy."""
    EXACT = "exact"          # Exact string match
    KEYWORD = "keyword"      # Keyword/phrase match
    FUZZY = "fuzzy"         # Fuzzy string match
    SEMANTIC = "semantic"    # Semantic similarity
    REFERENCE = "reference"  # Document reference


@dataclass
class Citation:
    """A citation with position and confidence information."""
    id: str
    document_id: str
    document_title: str
    chunk_id: Optional[str]
    text: str                # The cited text
    start_char: int         # Start position in document
    end_char: int           # End position in document
    confidence: float       # Confidence score (0-1)
    match_type: CitationType
    query_overlap: Optional[str] = None  # Part of query that matched
    context: Optional[str] = None        # Surrounding context
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __hash__(self):
        """Make Citation hashable for deduplication."""
        return hash((self.document_id, self.start_char, self.end_char))
    
    def __eq__(self, other):
        """Check equality based on document and position."""
        if not isinstance(other, Citation):
            return False
        return (
            self.document_id == other.document_id and
            self.start_char == other.start_char and
            self.end_char == other.end_char
        )


class CitationGenerator:
    """Generate citations for documents based on queries."""
    
    def __init__(
        self,
        max_citation_length: int = 200,
        context_window: int = 50,
        enable_fuzzy: bool = True,
        fuzzy_threshold: float = 0.8,
        enable_semantic: bool = True,
        semantic_threshold: float = 0.7
    ):
        """
        Initialize citation generator.
        
        Args:
            max_citation_length: Maximum length of citation text
            context_window: Characters of context around citation
            enable_fuzzy: Enable fuzzy matching
            fuzzy_threshold: Minimum similarity for fuzzy matches
            enable_semantic: Enable semantic matching
            semantic_threshold: Minimum similarity for semantic matches
        """
        self.max_citation_length = max_citation_length
        self.context_window = context_window
        self.enable_fuzzy = enable_fuzzy
        self.fuzzy_threshold = fuzzy_threshold
        self.enable_semantic = enable_semantic
        self.semantic_threshold = semantic_threshold
        
        # Compile regex patterns
        self._compile_patterns()
    
    def _compile_patterns(self):
        """Pre-compile regex patterns for performance."""
        # Sentence boundary pattern
        self.sentence_pattern = re.compile(r'[.!?]\s+')
        
        # Word boundary pattern
        self.word_boundary = re.compile(r'\b')
        
        # Whitespace normalization
        self.whitespace_pattern = re.compile(r'\s+')
    
    def generate_citations(
        self,
        query: str,
        documents: List[Document],
        max_citations: int = 10
    ) -> List[Citation]:
        """
        Generate citations for documents based on query.
        
        Args:
            query: The search query
            documents: Documents to generate citations from
            max_citations: Maximum number of citations to return
            
        Returns:
            List of citations sorted by confidence
        """
        all_citations = []
        
        for doc in documents:
            # Generate citations for this document
            doc_citations = self._generate_document_citations(query, doc)
            all_citations.extend(doc_citations)
        
        # Deduplicate citations
        unique_citations = self._deduplicate_citations(all_citations)
        
        # Sort by confidence and return top N
        sorted_citations = sorted(
            unique_citations,
            key=lambda c: c.confidence,
            reverse=True
        )
        
        return sorted_citations[:max_citations]
    
    def _generate_document_citations(
        self,
        query: str,
        document: Document
    ) -> List[Citation]:
        """Generate all types of citations for a single document."""
        citations = []
        
        # Extract query terms
        query_terms = self._extract_query_terms(query)
        
        # 1. Exact match citations
        exact_citations = self._find_exact_matches(query, document)
        citations.extend(exact_citations)
        
        # 2. Keyword match citations
        keyword_citations = self._find_keyword_matches(query_terms, document)
        citations.extend(keyword_citations)
        
        # 3. Fuzzy match citations (if enabled and no exact matches)
        if self.enable_fuzzy and not exact_citations:
            fuzzy_citations = self._find_fuzzy_matches(query, document)
            citations.extend(fuzzy_citations)
        
        # 4. Semantic citations (if enabled and document has high score)
        if self.enable_semantic and document.score > self.semantic_threshold:
            semantic_citation = self._create_semantic_citation(document)
            if semantic_citation:
                citations.append(semantic_citation)
        
        return citations
    
    def _extract_query_terms(self, query: str) -> List[str]:
        """Extract significant terms from query."""
        # Common stop words to exclude
        stop_words = {
            'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 
            'for', 'of', 'with', 'by', 'from', 'as', 'is', 'was', 'are', 
            'were', 'been', 'be', 'have', 'has', 'had', 'do', 'does', 'did',
            'will', 'would', 'could', 'should', 'may', 'might', 'can', 'must',
            'shall', 'what', 'which', 'who', 'when', 'where', 'why', 'how'
        }
        
        # Tokenize and filter
        words = query.lower().split()
        terms = [
            word.strip('.,!?;:"')
            for word in words
            if word.lower() not in stop_words and len(word) > 2
        ]
        
        # Also extract multi-word phrases (bigrams)
        phrases = []
        for i in range(len(words) - 1):
            if words[i].lower() not in stop_words or words[i+1].lower() not in stop_words:
                phrase = f"{words[i]} {words[i+1]}"
                phrases.append(phrase.strip('.,!?;:"'))
        
        return terms + phrases
    
    def _find_exact_matches(
        self,
        query: str,
        document: Document
    ) -> List[Citation]:
        """Find exact query matches in document."""
        citations = []
        content = document.content
        content_lower = content.lower()
        query_lower = query.lower()
        
        # Find all occurrences
        start = 0
        while True:
            pos = content_lower.find(query_lower, start)
            if pos == -1:
                break
            
            # Extract citation with context
            citation_start = max(0, pos - self.context_window)
            citation_end = min(
                len(content),
                pos + len(query) + self.context_window
            )
            
            citation_text = content[citation_start:citation_end]
            
            # Truncate if too long
            if len(citation_text) > self.max_citation_length:
                citation_text = citation_text[:self.max_citation_length] + "..."
            
            # Create citation
            citation = Citation(
                id=self._generate_citation_id(document.id, pos),
                document_id=document.id,
                document_title=document.metadata.get("title", "Untitled"),
                chunk_id=document.metadata.get("chunk_id"),
                text=citation_text,
                start_char=pos,
                end_char=pos + len(query),
                confidence=1.0,  # Exact match has highest confidence
                match_type=CitationType.EXACT,
                query_overlap=query,
                context=content[citation_start:pos] + "[MATCH]" + content[pos + len(query):citation_end],
                metadata={"match_text": content[pos:pos + len(query)]}
            )
            
            citations.append(citation)
            start = pos + 1
            
            # Limit exact matches per document
            if len(citations) >= 3:
                break
        
        return citations
    
    def _find_keyword_matches(
        self,
        query_terms: List[str],
        document: Document
    ) -> List[Citation]:
        """Find keyword matches in document."""
        citations = []
        content = document.content
        content_lower = content.lower()
        
        # Track which terms we've found
        found_terms = defaultdict(list)
        
        for term in query_terms:
            term_lower = term.lower()
            
            # Find first occurrence of each term
            pos = content_lower.find(term_lower)
            if pos != -1:
                found_terms[term].append(pos)
        
        # Create citations for found terms
        for term, positions in found_terms.items():
            for pos in positions[:2]:  # Limit to 2 citations per term
                # Find sentence boundaries
                sentence_start = self._find_sentence_start(content, pos)
                sentence_end = self._find_sentence_end(content, pos + len(term))
                
                # Extract citation text
                citation_text = content[sentence_start:sentence_end]
                
                # Truncate if needed
                if len(citation_text) > self.max_citation_length:
                    # Try to keep the matched term visible
                    term_offset = pos - sentence_start
                    if term_offset > self.max_citation_length // 2:
                        # Term is far from start, truncate beginning
                        new_start = pos - self.max_citation_length // 2
                        citation_text = "..." + content[new_start:sentence_end]
                    else:
                        # Truncate end
                        citation_text = citation_text[:self.max_citation_length] + "..."
                
                # Calculate confidence based on term importance
                confidence = 0.7 * (len(term) / max(len(qt) for qt in query_terms))
                
                citation = Citation(
                    id=self._generate_citation_id(document.id, pos),
                    document_id=document.id,
                    document_title=document.metadata.get("title", "Untitled"),
                    chunk_id=document.metadata.get("chunk_id"),
                    text=citation_text,
                    start_char=sentence_start,
                    end_char=sentence_end,
                    confidence=confidence,
                    match_type=CitationType.KEYWORD,
                    query_overlap=term,
                    metadata={"matched_term": term, "term_position": pos}
                )
                
                citations.append(citation)
        
        return citations
    
    def _find_fuzzy_matches(
        self,
        query: str,
        document: Document
    ) -> List[Citation]:
        """Find fuzzy matches using sequence matching."""
        citations = []
        content = document.content
        
        # Split into sentences for fuzzy matching
        sentences = self.sentence_pattern.split(content)
        
        for i, sentence in enumerate(sentences):
            if len(sentence) < 20:  # Skip very short sentences
                continue
            
            # Calculate similarity
            similarity = difflib.SequenceMatcher(
                None,
                query.lower(),
                sentence.lower()
            ).ratio()
            
            if similarity >= self.fuzzy_threshold:
                # Find sentence position in original content
                sentence_pos = content.find(sentence)
                if sentence_pos == -1:
                    continue
                
                # Extract with context
                citation_start = max(0, sentence_pos - self.context_window)
                citation_end = min(
                    len(content),
                    sentence_pos + len(sentence) + self.context_window
                )
                
                citation_text = content[citation_start:citation_end]
                
                # Truncate if needed
                if len(citation_text) > self.max_citation_length:
                    citation_text = citation_text[:self.max_citation_length] + "..."
                
                citation = Citation(
                    id=self._generate_citation_id(document.id, sentence_pos),
                    document_id=document.id,
                    document_title=document.metadata.get("title", "Untitled"),
                    chunk_id=document.metadata.get("chunk_id"),
                    text=citation_text,
                    start_char=sentence_pos,
                    end_char=sentence_pos + len(sentence),
                    confidence=similarity,
                    match_type=CitationType.FUZZY,
                    metadata={
                        "similarity_ratio": similarity,
                        "sentence_index": i
                    }
                )
                
                citations.append(citation)
        
        # Return top fuzzy matches
        citations.sort(key=lambda c: c.confidence, reverse=True)
        return citations[:3]
    
    def _create_semantic_citation(self, document: Document) -> Optional[Citation]:
        """Create semantic citation for high-scoring document."""
        content = document.content
        
        # Find the most relevant section (usually the beginning for summaries)
        # Or look for the section with highest keyword density
        
        # For now, take the first substantial paragraph
        paragraphs = content.split('\n\n')
        
        for para in paragraphs:
            if len(para) > 50:  # Substantial paragraph
                citation_text = para[:self.max_citation_length]
                if len(para) > self.max_citation_length:
                    citation_text += "..."
                
                return Citation(
                    id=self._generate_citation_id(document.id, 0),
                    document_id=document.id,
                    document_title=document.metadata.get("title", "Untitled"),
                    chunk_id=document.metadata.get("chunk_id"),
                    text=citation_text,
                    start_char=0,
                    end_char=len(citation_text),
                    confidence=document.score,
                    match_type=CitationType.SEMANTIC,
                    metadata={
                        "relevance_score": document.score,
                        "is_summary": True
                    }
                )
        
        # Fallback to document beginning
        citation_text = content[:self.max_citation_length]
        if len(content) > self.max_citation_length:
            citation_text += "..."
        
        return Citation(
            id=self._generate_citation_id(document.id, 0),
            document_id=document.id,
            document_title=document.metadata.get("title", "Untitled"),
            chunk_id=document.metadata.get("chunk_id"),
            text=citation_text,
            start_char=0,
            end_char=len(citation_text),
            confidence=document.score,
            match_type=CitationType.SEMANTIC,
            metadata={"relevance_score": document.score}
        )
    
    def _find_sentence_start(self, text: str, pos: int) -> int:
        """Find the start of the sentence containing position."""
        # Look backward for sentence boundary
        for i in range(pos, -1, -1):
            if i == 0:
                return 0
            if text[i] in '.!?' and i < pos - 1:
                # Found sentence boundary, return position after it
                return i + 1 if i + 1 < len(text) else i
        return 0
    
    def _find_sentence_end(self, text: str, pos: int) -> int:
        """Find the end of the sentence containing position."""
        # Look forward for sentence boundary
        for i in range(pos, len(text)):
            if text[i] in '.!?':
                return i + 1 if i + 1 < len(text) else i
        return len(text)
    
    def _deduplicate_citations(self, citations: List[Citation]) -> List[Citation]:
        """Remove duplicate citations, keeping highest confidence."""
        # Group by document and overlapping positions
        citation_groups = defaultdict(list)
        
        for citation in citations:
            key = citation.document_id
            citation_groups[key].append(citation)
        
        unique_citations = []
        
        for doc_id, doc_citations in citation_groups.items():
            # Sort by position
            doc_citations.sort(key=lambda c: c.start_char)
            
            # Merge overlapping citations
            merged = []
            for citation in doc_citations:
                # Check if overlaps with any merged citation
                overlap_found = False
                for i, existing in enumerate(merged):
                    if self._citations_overlap(citation, existing):
                        # Keep the one with higher confidence
                        if citation.confidence > existing.confidence:
                            merged[i] = citation
                        overlap_found = True
                        break
                
                if not overlap_found:
                    merged.append(citation)
            
            unique_citations.extend(merged)
        
        return unique_citations
    
    def _citations_overlap(self, c1: Citation, c2: Citation) -> bool:
        """Check if two citations overlap in position."""
        if c1.document_id != c2.document_id:
            return False
        
        # Check if ranges overlap
        return not (c1.end_char < c2.start_char or c2.end_char < c1.start_char)
    
    def _generate_citation_id(self, doc_id: str, position: int) -> str:
        """Generate unique citation ID."""
        content = f"{doc_id}_{position}"
        return hashlib.md5(content.encode()).hexdigest()[:12]
    
    def format_citations_for_display(
        self,
        citations: List[Citation],
        format_type: str = "numbered"
    ) -> str:
        """
        Format citations for display in response.
        
        Args:
            citations: List of citations
            format_type: Format style ("numbered", "inline", "footnote")
            
        Returns:
            Formatted citation text
        """
        if format_type == "numbered":
            formatted = []
            for i, citation in enumerate(citations, 1):
                formatted.append(
                    f"[{i}] {citation.document_title}: \"{citation.text}\" "
                    f"(confidence: {citation.confidence:.2f})"
                )
            return "\n".join(formatted)
        
        elif format_type == "inline":
            formatted = []
            for i, citation in enumerate(citations, 1):
                formatted.append(f"[{i}]")
            return " ".join(formatted)
        
        elif format_type == "footnote":
            formatted = []
            for i, citation in enumerate(citations, 1):
                formatted.append(
                    f"[^{i}]: {citation.document_title} - {citation.text[:50]}..."
                )
            return "\n".join(formatted)
        
        else:
            return str(citations)


# Pipeline integration function
async def generate_citations(context: Any, **kwargs) -> Any:
    """Generate citations for pipeline context."""
    config = context.config.get("citations", {})
    
    # Check if citations are enabled
    if not config.get("enabled", True):
        return context
    
    # Create citation generator
    generator = CitationGenerator(
        max_citation_length=config.get("max_length", 200),
        context_window=config.get("context_window", 50),
        enable_fuzzy=config.get("enable_fuzzy", True),
        fuzzy_threshold=config.get("fuzzy_threshold", 0.8),
        enable_semantic=config.get("enable_semantic", True),
        semantic_threshold=config.get("semantic_threshold", 0.7)
    )
    
    # Generate citations
    citations = generator.generate_citations(
        query=context.query,
        documents=context.documents,
        max_citations=config.get("max_citations", 10)
    )
    
    # Add to context
    context.citations = citations
    context.metadata["citations"] = {
        "count": len(citations),
        "types": defaultdict(int)
    }
    
    # Count citation types
    for citation in citations:
        context.metadata["citations"]["types"][citation.match_type.value] += 1
    
    logger.info(f"Generated {len(citations)} citations")
    
    return context