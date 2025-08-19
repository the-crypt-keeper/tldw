# advanced_query_expansion.py - Enhanced Query Expansion Module
"""
Advanced query expansion module with improved techniques for RAG search.

This module extends the basic query expansion with:
- Semantic similarity-based expansion
- Entity recognition and expansion
- Acronym and abbreviation handling
- Domain-specific term expansion
- Multi-language support
"""

import re
import json
import logging
from typing import List, Dict, Any, Optional, Set, Tuple
from dataclasses import dataclass, field
from enum import Enum
import asyncio
from functools import lru_cache

logger = logging.getLogger(__name__)


class ExpansionStrategy(Enum):
    """Query expansion strategies"""
    SEMANTIC = "semantic"          # Use embeddings for semantic similarity
    LINGUISTIC = "linguistic"      # Synonyms, hyponyms, hypernyms
    ENTITY = "entity"              # Named entity recognition
    ACRONYM = "acronym"            # Expand/contract acronyms
    MULTILINGUAL = "multilingual"  # Translate to other languages
    DOMAIN = "domain"              # Domain-specific expansions
    HYBRID = "hybrid"              # Combine multiple strategies


@dataclass
class ExpansionConfig:
    """Configuration for advanced query expansion"""
    strategies: List[ExpansionStrategy] = field(
        default_factory=lambda: [ExpansionStrategy.SEMANTIC, ExpansionStrategy.LINGUISTIC]
    )
    max_expansions_per_strategy: int = 3
    total_max_expansions: int = 10
    semantic_similarity_threshold: float = 0.7
    enable_entity_recognition: bool = True
    enable_acronym_expansion: bool = True
    domain_vocabulary: Optional[Dict[str, List[str]]] = None
    target_languages: List[str] = field(default_factory=lambda: ["en"])
    use_cache: bool = True
    cache_ttl: int = 3600


class AdvancedQueryExpander:
    """Advanced query expansion with multiple strategies"""
    
    # Common acronyms and their expansions
    ACRONYM_DB = {
        "ai": ["artificial intelligence", "ai"],
        "ml": ["machine learning", "ml"],
        "nlp": ["natural language processing", "nlp"],
        "llm": ["large language model", "llm"],
        "rag": ["retrieval augmented generation", "rag"],
        "api": ["application programming interface", "api"],
        "db": ["database", "db"],
        "ui": ["user interface", "ui"],
        "ux": ["user experience", "ux"],
        "cli": ["command line interface", "cli"],
        "gpu": ["graphics processing unit", "gpu"],
        "cpu": ["central processing unit", "cpu"],
        "ram": ["random access memory", "ram"],
        "ssd": ["solid state drive", "ssd"],
        "hdd": ["hard disk drive", "hdd"],
        "os": ["operating system", "os"],
        "vm": ["virtual machine", "vm"],
        "ci": ["continuous integration", "ci"],
        "cd": ["continuous deployment", "continuous delivery", "cd"],
        "sdk": ["software development kit", "sdk"],
        "ide": ["integrated development environment", "ide"],
        "json": ["javascript object notation", "json"],
        "xml": ["extensible markup language", "xml"],
        "html": ["hypertext markup language", "html"],
        "css": ["cascading style sheets", "css"],
        "sql": ["structured query language", "sql"],
        "http": ["hypertext transfer protocol", "http"],
        "https": ["hypertext transfer protocol secure", "https"],
        "rest": ["representational state transfer", "rest"],
        "crud": ["create read update delete", "crud"],
        "mvc": ["model view controller", "mvc"],
        "mvp": ["model view presenter", "minimum viable product", "mvp"],
        "tdd": ["test driven development", "tdd"],
        "bdd": ["behavior driven development", "bdd"],
        "poc": ["proof of concept", "poc"],
        "roi": ["return on investment", "roi"],
        "kpi": ["key performance indicator", "kpi"],
        "saas": ["software as a service", "saas"],
        "paas": ["platform as a service", "paas"],
        "iaas": ["infrastructure as a service", "iaas"],
    }
    
    # Domain-specific term relationships
    DOMAIN_TERMS = {
        "transcription": ["speech to text", "audio to text", "voice recognition", "speech recognition"],
        "summarization": ["summary", "abstract", "synopsis", "digest", "condensation"],
        "embedding": ["vector", "representation", "encoding", "feature vector"],
        "chunking": ["segmentation", "splitting", "partitioning", "division"],
        "retrieval": ["search", "lookup", "query", "fetch", "find"],
        "generation": ["creation", "synthesis", "production", "composition"],
        "model": ["algorithm", "system", "network", "architecture"],
        "training": ["learning", "fitting", "optimization", "tuning"],
        "inference": ["prediction", "generation", "output", "result"],
        "context": ["background", "setting", "environment", "situation"],
        "prompt": ["input", "query", "instruction", "command"],
        "token": ["word", "subword", "piece", "unit"],
        "attention": ["focus", "weight", "importance", "relevance"],
        "transformer": ["model", "architecture", "network", "system"],
        "fine-tuning": ["adaptation", "customization", "specialization", "adjustment"],
    }
    
    def __init__(self, config: ExpansionConfig, embeddings_model=None):
        """
        Initialize the advanced query expander.
        
        Args:
            config: Expansion configuration
            embeddings_model: Optional embeddings model for semantic expansion
        """
        self.config = config
        self.embeddings_model = embeddings_model
        self._cache = {} if config.use_cache else None
        
        # Build reverse acronym lookup
        self._reverse_acronyms = {}
        for acronym, expansions in self.ACRONYM_DB.items():
            for expansion in expansions:
                if expansion != acronym:  # Don't map to itself
                    self._reverse_acronyms[expansion] = acronym
    
    async def expand_query(self, query: str) -> List[str]:
        """
        Expand a query using configured strategies.
        
        Args:
            query: The original query
            
        Returns:
            List of expanded queries
        """
        # Check cache
        if self._cache is not None and query in self._cache:
            return self._cache[query]
        
        # Collect expansions from each strategy
        all_expansions = set()
        
        for strategy in self.config.strategies:
            try:
                if strategy == ExpansionStrategy.SEMANTIC:
                    expansions = await self._semantic_expansion(query)
                elif strategy == ExpansionStrategy.LINGUISTIC:
                    expansions = self._linguistic_expansion(query)
                elif strategy == ExpansionStrategy.ENTITY:
                    expansions = self._entity_expansion(query)
                elif strategy == ExpansionStrategy.ACRONYM:
                    expansions = self._acronym_expansion(query)
                elif strategy == ExpansionStrategy.DOMAIN:
                    expansions = self._domain_expansion(query)
                elif strategy == ExpansionStrategy.MULTILINGUAL:
                    expansions = await self._multilingual_expansion(query)
                elif strategy == ExpansionStrategy.HYBRID:
                    expansions = await self._hybrid_expansion(query)
                
                # Add expansions up to per-strategy limit
                for exp in expansions[:self.config.max_expansions_per_strategy]:
                    if exp.lower() != query.lower():
                        all_expansions.add(exp)
                
            except Exception as e:
                logger.error(f"Error in {strategy.value} expansion: {e}")
                continue
        
        # Convert to list and limit total expansions
        expansions = list(all_expansions)[:self.config.total_max_expansions]
        
        # Cache result
        if self._cache is not None:
            self._cache[query] = expansions
        
        logger.info(f"Generated {len(expansions)} expansions for query: {query}")
        return expansions
    
    async def _semantic_expansion(self, query: str) -> List[str]:
        """Generate semantically similar queries using embeddings."""
        if not self.embeddings_model:
            return []
        
        expansions = []
        
        # Split query into meaningful segments
        segments = self._extract_key_phrases(query)
        
        for segment in segments:
            # Generate variations
            variations = [
                f"{segment} explanation",
                f"what is {segment}",
                f"how does {segment} work",
                f"{segment} tutorial",
                f"{segment} guide",
            ]
            expansions.extend(variations)
        
        # TODO: Use actual embeddings to find similar phrases
        # This would require access to a vocabulary or phrase bank
        
        return expansions
    
    def _linguistic_expansion(self, query: str) -> List[str]:
        """Expand using linguistic relationships (synonyms, related terms)."""
        expansions = []
        words = query.lower().split()
        
        # Simple synonym mapping (in production, use WordNet or similar)
        synonyms = {
            "find": ["search", "locate", "discover", "retrieve"],
            "create": ["make", "generate", "produce", "build"],
            "update": ["modify", "change", "edit", "alter"],
            "delete": ["remove", "erase", "eliminate", "destroy"],
            "show": ["display", "present", "exhibit", "reveal"],
            "list": ["enumerate", "itemize", "catalog", "index"],
            "get": ["retrieve", "fetch", "obtain", "acquire"],
            "set": ["configure", "establish", "define", "assign"],
            "run": ["execute", "perform", "operate", "launch"],
            "stop": ["halt", "terminate", "cease", "end"],
        }
        
        # Generate expansions with synonyms
        for word in words:
            if word in synonyms:
                for synonym in synonyms[word][:2]:  # Limit synonyms per word
                    expanded = query.lower().replace(word, synonym)
                    if expanded != query.lower():
                        expansions.append(expanded)
        
        return expansions
    
    def _entity_expansion(self, query: str) -> List[str]:
        """Expand based on recognized entities."""
        if not self.config.enable_entity_recognition:
            return []
        
        expansions = []
        
        # Simple entity patterns (in production, use NER model)
        patterns = {
            "date": r'\b\d{4}[-/]\d{1,2}[-/]\d{1,2}\b',
            "time": r'\b\d{1,2}:\d{2}(?::\d{2})?\s*(?:AM|PM|am|pm)?\b',
            "email": r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',
            "url": r'https?://[^\s]+',
            "number": r'\b\d+\.?\d*\b',
        }
        
        # Check for entities and create targeted expansions
        for entity_type, pattern in patterns.items():
            if re.search(pattern, query):
                if entity_type == "date":
                    expansions.extend([
                        query + " events",
                        query + " activities",
                        query + " timeline",
                    ])
                elif entity_type == "email":
                    expansions.extend([
                        query + " sender",
                        query + " recipient",
                        query + " conversation",
                    ])
        
        return expansions
    
    def _acronym_expansion(self, query: str) -> List[str]:
        """Expand or contract acronyms in the query."""
        if not self.config.enable_acronym_expansion:
            return []
        
        expansions = []
        words = query.lower().split()
        
        # Check each word for acronym expansion
        for i, word in enumerate(words):
            if word in self.ACRONYM_DB:
                # Expand acronym
                for expansion in self.ACRONYM_DB[word]:
                    if expansion != word:
                        new_words = words.copy()
                        new_words[i] = expansion
                        expansions.append(" ".join(new_words))
        
        # Check for contractable phrases
        query_lower = query.lower()
        for full_form, acronym in self._reverse_acronyms.items():
            if full_form in query_lower:
                expanded = query_lower.replace(full_form, acronym)
                if expanded != query_lower:
                    expansions.append(expanded)
        
        return expansions
    
    def _domain_expansion(self, query: str) -> List[str]:
        """Expand using domain-specific vocabulary."""
        expansions = []
        
        # Use built-in domain terms
        domain_terms = self.DOMAIN_TERMS
        if self.config.domain_vocabulary:
            domain_terms.update(self.config.domain_vocabulary)
        
        # Check for domain terms in query
        query_lower = query.lower()
        for term, related_terms in domain_terms.items():
            if term in query_lower:
                for related in related_terms[:2]:  # Limit related terms
                    expanded = query_lower.replace(term, related)
                    if expanded != query_lower:
                        expansions.append(expanded)
        
        return expansions
    
    async def _multilingual_expansion(self, query: str) -> List[str]:
        """Expand by translating to other languages."""
        # This would require translation API integration
        # For now, return empty list
        return []
    
    async def _hybrid_expansion(self, query: str) -> List[str]:
        """Combine multiple expansion strategies."""
        expansions = []
        
        # Get expansions from main strategies
        semantic = await self._semantic_expansion(query)
        linguistic = self._linguistic_expansion(query)
        domain = self._domain_expansion(query)
        
        # Combine and deduplicate
        all_exp = set(semantic + linguistic + domain)
        
        # Apply acronym expansion to combined results
        for exp in list(all_exp):
            acronym_exp = self._acronym_expansion(exp)
            all_exp.update(acronym_exp)
        
        return list(all_exp)
    
    def _extract_key_phrases(self, query: str) -> List[str]:
        """Extract key phrases from the query."""
        # Simple noun phrase extraction
        # In production, use spaCy or similar for better extraction
        
        # Remove common words
        stop_words = {
            'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 
            'to', 'for', 'of', 'with', 'by', 'from', 'about', 'as',
            'is', 'are', 'was', 'were', 'been', 'be', 'have', 'has',
            'had', 'do', 'does', 'did', 'will', 'would', 'could', 'should',
            'may', 'might', 'must', 'shall', 'can', 'need'
        }
        
        words = query.lower().split()
        phrases = []
        
        # Extract single important words
        important_words = [w for w in words if w not in stop_words and len(w) > 2]
        phrases.extend(important_words)
        
        # Extract bigrams
        for i in range(len(words) - 1):
            if words[i] not in stop_words or words[i+1] not in stop_words:
                phrases.append(f"{words[i]} {words[i+1]}")
        
        return phrases[:5]  # Limit number of phrases
    
    def get_expansion_stats(self) -> Dict[str, Any]:
        """Get statistics about query expansions."""
        stats = {
            "configured_strategies": [s.value for s in self.config.strategies],
            "max_expansions_per_strategy": self.config.max_expansions_per_strategy,
            "total_max_expansions": self.config.total_max_expansions,
            "cache_enabled": self.config.use_cache,
            "cache_size": len(self._cache) if self._cache else 0,
        }
        
        return stats


# Example usage
if __name__ == "__main__":
    async def test_expansion():
        config = ExpansionConfig(
            strategies=[
                ExpansionStrategy.LINGUISTIC,
                ExpansionStrategy.ACRONYM,
                ExpansionStrategy.DOMAIN
            ],
            max_expansions_per_strategy=3,
            total_max_expansions=10
        )
        
        expander = AdvancedQueryExpander(config)
        
        test_queries = [
            "find LLM training data",
            "how to create embeddings",
            "RAG pipeline optimization",
            "machine learning model deployment",
            "update database schema",
        ]
        
        for query in test_queries:
            expansions = await expander.expand_query(query)
            print(f"\nQuery: {query}")
            print(f"Expansions ({len(expansions)}):")
            for exp in expansions:
                print(f"  - {exp}")
    
    asyncio.run(test_expansion())