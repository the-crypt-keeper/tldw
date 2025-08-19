"""
Query expansion strategies for improved retrieval.

This module provides various query expansion techniques to improve
search recall and precision.
"""

import re
import asyncio
from typing import List, Dict, Any, Optional, Set, Tuple
from dataclasses import dataclass
from abc import ABC, abstractmethod
import json

from loguru import logger

from .types import RetrieverStrategy, SearchResult, Document, DataSource


@dataclass
class ExpandedQuery:
    """Represents an expanded query with variations."""
    original_query: str
    variations: List[str]
    synonyms: Dict[str, List[str]]
    keywords: List[str]
    entities: List[str]
    metadata: Dict[str, Any]


class QueryExpansionStrategy(ABC):
    """Base class for query expansion strategies."""
    
    @abstractmethod
    async def expand(self, query: str) -> ExpandedQuery:
        """
        Expand a query into variations.
        
        Args:
            query: Original query string
            
        Returns:
            ExpandedQuery with variations
        """
        pass


class SynonymExpansion(QueryExpansionStrategy):
    """
    Expands queries using synonyms and related terms.
    
    This implementation uses a simple dictionary for demonstration.
    In production, you'd use WordNet, ConceptNet, or a custom thesaurus.
    """
    
    def __init__(self, synonym_dict: Optional[Dict[str, List[str]]] = None):
        """
        Initialize synonym expansion.
        
        Args:
            synonym_dict: Dictionary mapping words to synonyms
        """
        self.synonym_dict = synonym_dict or self._get_default_synonyms()
    
    def _get_default_synonyms(self) -> Dict[str, List[str]]:
        """Get default synonym dictionary for common terms."""
        return {
            # Technical terms
            "search": ["find", "query", "lookup", "retrieve"],
            "document": ["file", "text", "content", "record"],
            "database": ["db", "storage", "repository", "datastore"],
            "create": ["make", "generate", "build", "construct"],
            "delete": ["remove", "erase", "clear", "purge"],
            "update": ["modify", "change", "edit", "revise"],
            "error": ["bug", "issue", "problem", "fault"],
            "fix": ["repair", "resolve", "patch", "correct"],
            
            # Machine learning terms
            "model": ["algorithm", "network", "system"],
            "train": ["fit", "learn", "optimize"],
            "predict": ["forecast", "estimate", "infer"],
            "accuracy": ["precision", "performance", "score"],
            
            # General terms
            "fast": ["quick", "rapid", "speedy", "swift"],
            "slow": ["sluggish", "delayed", "lagging"],
            "large": ["big", "huge", "massive", "extensive"],
            "small": ["tiny", "minor", "compact", "minimal"],
            "good": ["excellent", "great", "positive", "beneficial"],
            "bad": ["poor", "negative", "problematic", "inferior"],
        }
    
    async def expand(self, query: str) -> ExpandedQuery:
        """
        Expand query using synonyms.
        
        Args:
            query: Original query string
            
        Returns:
            ExpandedQuery with synonym variations
        """
        words = query.lower().split()
        found_synonyms = {}
        variations = set()
        
        # Find synonyms for each word
        for word in words:
            if word in self.synonym_dict:
                found_synonyms[word] = self.synonym_dict[word]
                
                # Create variations by replacing the word with each synonym
                for synonym in self.synonym_dict[word]:
                    variation = query.lower().replace(word, synonym)
                    if variation != query.lower():
                        variations.add(variation)
        
        # Also create a variation with all synonyms combined
        if found_synonyms:
            expanded_words = []
            for word in words:
                if word in found_synonyms:
                    # Add original and first synonym
                    expanded_words.append(f"({word} OR {found_synonyms[word][0]})")
                else:
                    expanded_words.append(word)
            variations.add(" ".join(expanded_words))
        
        return ExpandedQuery(
            original_query=query,
            variations=list(variations)[:5],  # Limit variations
            synonyms=found_synonyms,
            keywords=words,
            entities=[],
            metadata={"strategy": "synonym_expansion"}
        )


class MultiQueryGeneration(QueryExpansionStrategy):
    """
    Generates multiple query variations using different perspectives.
    
    This simulates what an LLM might do to rephrase queries.
    """
    
    def __init__(self, generation_templates: Optional[List[str]] = None):
        """
        Initialize multi-query generation.
        
        Args:
            generation_templates: Templates for query generation
        """
        self.templates = generation_templates or self._get_default_templates()
    
    def _get_default_templates(self) -> List[str]:
        """Get default query generation templates."""
        return [
            "What is {query}",
            "Explain {query}",
            "How to {query}",
            "Definition of {query}",
            "Examples of {query}",
            "{query} tutorial",
            "{query} documentation",
            "Best practices for {query}",
            "Common issues with {query}",
            "Troubleshooting {query}"
        ]
    
    async def expand(self, query: str) -> ExpandedQuery:
        """
        Generate multiple query variations.
        
        Args:
            query: Original query string
            
        Returns:
            ExpandedQuery with generated variations
        """
        variations = []
        
        # Apply templates that make sense for the query
        query_lower = query.lower()
        
        # Determine query type
        is_question = any(query_lower.startswith(q) for q in ["what", "how", "why", "when", "where", "who"])
        is_action = any(word in query_lower for word in ["create", "make", "build", "fix", "solve", "implement"])
        
        if is_question:
            # For questions, rephrase differently
            variations.append(query.replace("?", ""))
            if "what is" in query_lower:
                variations.append(query_lower.replace("what is", "define"))
            if "how to" in query_lower:
                variations.append(query_lower.replace("how to", "steps to"))
        elif is_action:
            # For action queries, add context
            variations.append(f"tutorial {query}")
            variations.append(f"guide to {query}")
            variations.append(f"best way to {query}")
        else:
            # For general queries, use templates
            for template in self.templates[:3]:
                if "{query}" in template:
                    variations.append(template.format(query=query))
        
        # Extract key phrases
        keywords = self._extract_key_phrases(query)
        
        return ExpandedQuery(
            original_query=query,
            variations=variations[:5],  # Limit variations
            synonyms={},
            keywords=keywords,
            entities=self._extract_entities(query),
            metadata={"strategy": "multi_query_generation"}
        )
    
    def _extract_key_phrases(self, query: str) -> List[str]:
        """Extract key phrases from query."""
        # Simple implementation - in production, use NLP
        phrases = []
        words = query.split()
        
        # Extract 2-grams and 3-grams
        for i in range(len(words) - 1):
            phrases.append(f"{words[i]} {words[i+1]}")
        for i in range(len(words) - 2):
            phrases.append(f"{words[i]} {words[i+1]} {words[i+2]}")
        
        return phrases[:5]
    
    def _extract_entities(self, query: str) -> List[str]:
        """Extract named entities from query."""
        # Simple pattern matching - in production, use NER
        entities = []
        
        # Look for capitalized words (potential entities)
        words = query.split()
        for word in words:
            if word[0].isupper() and word.lower() not in ["what", "how", "why", "when", "where", "who"]:
                entities.append(word)
        
        return entities


class HybridQueryExpansion(QueryExpansionStrategy):
    """
    Combines multiple expansion strategies for comprehensive query expansion.
    """
    
    def __init__(self, strategies: Optional[List[QueryExpansionStrategy]] = None):
        """
        Initialize hybrid expansion.
        
        Args:
            strategies: List of expansion strategies to combine
        """
        self.strategies = strategies or [
            SynonymExpansion(),
            MultiQueryGeneration()
        ]
    
    async def expand(self, query: str) -> ExpandedQuery:
        """
        Expand query using multiple strategies.
        
        Args:
            query: Original query string
            
        Returns:
            Combined ExpandedQuery from all strategies
        """
        # Run all strategies in parallel
        expansion_tasks = [strategy.expand(query) for strategy in self.strategies]
        expansions = await asyncio.gather(*expansion_tasks)
        
        # Combine results
        all_variations = []
        all_synonyms = {}
        all_keywords = []
        all_entities = []
        
        for expansion in expansions:
            all_variations.extend(expansion.variations)
            all_synonyms.update(expansion.synonyms)
            all_keywords.extend(expansion.keywords)
            all_entities.extend(expansion.entities)
        
        # Deduplicate while preserving order
        seen = set()
        unique_variations = []
        for v in all_variations:
            if v not in seen:
                seen.add(v)
                unique_variations.append(v)
        
        return ExpandedQuery(
            original_query=query,
            variations=unique_variations[:10],  # Limit total variations
            synonyms=all_synonyms,
            keywords=list(set(all_keywords))[:20],
            entities=list(set(all_entities)),
            metadata={
                "strategy": "hybrid",
                "strategies_used": [type(s).__name__ for s in self.strategies]
            }
        )


class QueryExpansionRetriever(RetrieverStrategy):
    """
    Retriever wrapper that applies query expansion before search.
    
    This wrapper:
    - Expands the query using configured strategies
    - Performs retrieval with original and expanded queries
    - Merges and deduplicates results
    - Preserves query variations in metadata
    """
    
    def __init__(
        self,
        base_retriever: RetrieverStrategy,
        expansion_strategy: Optional[QueryExpansionStrategy] = None,
        max_variations: int = 3,
        merge_strategy: str = "union"  # "union" or "intersection"
    ):
        """
        Initialize query expansion retriever.
        
        Args:
            base_retriever: The underlying retriever
            expansion_strategy: Strategy for query expansion
            max_variations: Maximum query variations to use
            merge_strategy: How to merge results from variations
        """
        self.base_retriever = base_retriever
        self.expansion_strategy = expansion_strategy or HybridQueryExpansion()
        self.max_variations = max_variations
        self.merge_strategy = merge_strategy
        
        logger.info(f"Initialized QueryExpansionRetriever with {type(expansion_strategy).__name__}")
    
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
        Retrieve with query expansion.
        
        Args:
            query: The search query
            filters: Optional filters to apply
            top_k: Number of results to return
            
        Returns:
            SearchResult with expanded query results
        """
        # Expand the query
        expanded = await self.expansion_strategy.expand(query)
        
        # Select variations to use
        queries_to_search = [query] + expanded.variations[:self.max_variations]
        
        # Perform parallel retrieval for all query variations
        retrieval_tasks = [
            self.base_retriever.retrieve(q, filters, top_k)
            for q in queries_to_search
        ]
        results = await asyncio.gather(*retrieval_tasks)
        
        # Merge results
        merged_result = self._merge_results(results, query, expanded)
        
        # Add expansion metadata
        merged_result.query_variations = queries_to_search
        if not merged_result.metadata:
            merged_result.metadata = {}
        merged_result.metadata["expansion"] = {
            "variations_used": len(queries_to_search),
            "synonyms_found": len(expanded.synonyms),
            "keywords_extracted": len(expanded.keywords)
        }
        
        logger.debug(f"Query expansion: {len(queries_to_search)} variations, {len(merged_result.documents)} results")
        
        return merged_result
    
    def _merge_results(
        self,
        results: List[SearchResult],
        original_query: str,
        expanded: ExpandedQuery
    ) -> SearchResult:
        """
        Merge results from multiple query variations.
        
        Args:
            results: List of search results from variations
            original_query: The original query
            expanded: The expanded query information
            
        Returns:
            Merged SearchResult
        """
        if self.merge_strategy == "intersection":
            # Only keep documents that appear in multiple results
            doc_counts = {}
            for result in results:
                for doc in result.documents:
                    doc_counts[doc.id] = doc_counts.get(doc.id, 0) + 1
            
            # Filter documents that appear in at least 2 results
            merged_docs = []
            seen_ids = set()
            for result in results:
                for doc in result.documents:
                    if doc.id not in seen_ids and doc_counts[doc.id] >= 2:
                        merged_docs.append(doc)
                        seen_ids.add(doc.id)
        else:  # union
            # Keep all unique documents
            merged_docs = []
            seen_ids = set()
            
            for result in results:
                for doc in result.documents:
                    if doc.id not in seen_ids:
                        merged_docs.append(doc)
                        seen_ids.add(doc.id)
        
        # Re-score based on appearance frequency
        for doc in merged_docs:
            appearances = sum(1 for r in results if any(d.id == doc.id for d in r.documents))
            doc.score = doc.score * (1 + 0.1 * (appearances - 1))  # Boost score for multiple appearances
        
        # Sort by score
        merged_docs.sort(key=lambda d: d.score, reverse=True)
        
        return SearchResult(
            documents=merged_docs,
            query=original_query,
            search_type="expanded_" + results[0].search_type if results else "expanded"
        )