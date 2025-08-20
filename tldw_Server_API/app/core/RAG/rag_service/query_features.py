# query_features.py
"""
Advanced query features for the RAG service.

This module provides query understanding, intent detection, query rewriting,
and other advanced query processing capabilities.
"""

import re
import hashlib
from dataclasses import dataclass, field
from enum import Enum
from typing import List, Dict, Any, Optional, Tuple, Set
from collections import defaultdict
import json

from loguru import logger
import nltk
from nltk.corpus import wordnet, stopwords
from nltk.tokenize import word_tokenize

# Download NLTK data if needed
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', quiet=True)

try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('wordnet', quiet=True)

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords', quiet=True)


class QueryIntent(Enum):
    """Types of query intent."""
    FACTUAL = "factual"          # Looking for specific facts
    EXPLORATORY = "exploratory"   # Broad exploration of topic
    COMPARATIVE = "comparative"   # Comparing multiple things
    PROCEDURAL = "procedural"     # How-to questions
    DEFINITIONAL = "definitional" # What is X questions
    CAUSAL = "causal"            # Why/cause questions
    TEMPORAL = "temporal"         # When/timeline questions
    ANALYTICAL = "analytical"     # Analysis/evaluation questions


class QueryComplexity(Enum):
    """Query complexity levels."""
    SIMPLE = "simple"      # Single concept
    MODERATE = "moderate"  # 2-3 concepts
    COMPLEX = "complex"    # Multiple interrelated concepts


@dataclass
class QueryAnalysis:
    """Analysis results for a query."""
    original_query: str
    cleaned_query: str
    intent: QueryIntent
    complexity: QueryComplexity
    key_terms: List[str]
    entities: List[str]
    temporal_refs: List[str]
    question_type: Optional[str] = None
    domain: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class QueryRewrite:
    """A rewritten version of a query."""
    rewritten_query: str
    rewrite_type: str
    confidence: float
    explanation: str


class QueryAnalyzer:
    """Analyzes queries to understand intent and structure."""
    
    def __init__(self):
        """Initialize query analyzer."""
        self.stop_words = set(stopwords.words('english'))
        self.intent_patterns = self._compile_intent_patterns()
        self.domain_keywords = self._load_domain_keywords()
    
    def _compile_intent_patterns(self) -> Dict[QueryIntent, List[re.Pattern]]:
        """Compile regex patterns for intent detection."""
        patterns = {
            QueryIntent.FACTUAL: [
                re.compile(r'^(what|who|where|which)\s+', re.I),
                re.compile(r'\b(fact|detail|information|data)\b', re.I)
            ],
            QueryIntent.EXPLORATORY: [
                re.compile(r'^tell me about\s+', re.I),
                re.compile(r'\b(explore|overview|explain|describe)\b', re.I)
            ],
            QueryIntent.COMPARATIVE: [
                re.compile(r'\b(compare|versus|vs|difference|similar|better|worse)\b', re.I),
                re.compile(r'\b(than|between|among)\b', re.I)
            ],
            QueryIntent.PROCEDURAL: [
                re.compile(r'^how (to|do|can)\s+', re.I),
                re.compile(r'\b(step|procedure|method|process|guide)\b', re.I)
            ],
            QueryIntent.DEFINITIONAL: [
                re.compile(r'^what (is|are)\s+', re.I),
                re.compile(r'\b(define|definition|meaning|means)\b', re.I)
            ],
            QueryIntent.CAUSAL: [
                re.compile(r'^why\s+', re.I),
                re.compile(r'\b(cause|reason|because|result|effect|lead to)\b', re.I)
            ],
            QueryIntent.TEMPORAL: [
                re.compile(r'^when\s+', re.I),
                re.compile(r'\b(timeline|history|date|year|period|era)\b', re.I),
                re.compile(r'\b(before|after|during|since|until)\b', re.I)
            ],
            QueryIntent.ANALYTICAL: [
                re.compile(r'\b(analyze|evaluate|assess|critique|review)\b', re.I),
                re.compile(r'\b(impact|significance|implications|consequences)\b', re.I)
            ]
        }
        return patterns
    
    def _load_domain_keywords(self) -> Dict[str, Set[str]]:
        """Load domain-specific keywords."""
        return {
            "technology": {"software", "hardware", "code", "programming", "algorithm", 
                          "database", "network", "system", "application", "framework"},
            "science": {"research", "experiment", "hypothesis", "theory", "data",
                       "analysis", "study", "observation", "evidence", "methodology"},
            "business": {"market", "strategy", "revenue", "customer", "product",
                        "service", "growth", "investment", "profit", "company"},
            "medical": {"patient", "treatment", "diagnosis", "symptom", "disease",
                       "medication", "therapy", "clinical", "health", "medical"},
            "legal": {"law", "regulation", "contract", "legal", "court",
                     "rights", "obligation", "liability", "compliance", "jurisdiction"}
        }
    
    def analyze_query(self, query: str) -> QueryAnalysis:
        """
        Analyze a query to understand its intent and structure.
        
        Args:
            query: The query to analyze
            
        Returns:
            QueryAnalysis object with analysis results
        """
        # Clean query
        cleaned_query = self._clean_query(query)
        
        # Detect intent
        intent = self._detect_intent(query)
        
        # Assess complexity
        complexity = self._assess_complexity(cleaned_query)
        
        # Extract key terms
        key_terms = self._extract_key_terms(cleaned_query)
        
        # Extract entities
        entities = self._extract_entities(cleaned_query)
        
        # Extract temporal references
        temporal_refs = self._extract_temporal_references(cleaned_query)
        
        # Detect question type
        question_type = self._detect_question_type(query)
        
        # Detect domain
        domain = self._detect_domain(cleaned_query)
        
        return QueryAnalysis(
            original_query=query,
            cleaned_query=cleaned_query,
            intent=intent,
            complexity=complexity,
            key_terms=key_terms,
            entities=entities,
            temporal_refs=temporal_refs,
            question_type=question_type,
            domain=domain,
            metadata={
                "word_count": len(cleaned_query.split()),
                "has_question_mark": '?' in query,
                "has_quotes": '"' in query or "'" in query
            }
        )
    
    def _clean_query(self, query: str) -> str:
        """Clean and normalize query."""
        # Remove extra whitespace
        cleaned = ' '.join(query.split())
        
        # Normalize punctuation
        cleaned = re.sub(r'\s+([?.!,])', r'\1', cleaned)
        
        return cleaned
    
    def _detect_intent(self, query: str) -> QueryIntent:
        """Detect query intent."""
        query_lower = query.lower()
        
        # Check each intent pattern
        intent_scores = defaultdict(int)
        
        for intent, patterns in self.intent_patterns.items():
            for pattern in patterns:
                if pattern.search(query_lower):
                    intent_scores[intent] += 1
        
        # Return intent with highest score
        if intent_scores:
            return max(intent_scores.items(), key=lambda x: x[1])[0]
        
        # Default to factual
        return QueryIntent.FACTUAL
    
    def _assess_complexity(self, query: str) -> QueryComplexity:
        """Assess query complexity."""
        # Count concepts (approximated by noun phrases)
        words = word_tokenize(query.lower())
        
        # Remove stop words
        content_words = [w for w in words if w not in self.stop_words and w.isalnum()]
        
        # Count logical operators
        logical_ops = sum(1 for w in words if w in ['and', 'or', 'but', 'however', 'although'])
        
        # Assess based on content words and operators
        if len(content_words) <= 3 and logical_ops == 0:
            return QueryComplexity.SIMPLE
        elif len(content_words) <= 6 or logical_ops <= 1:
            return QueryComplexity.MODERATE
        else:
            return QueryComplexity.COMPLEX
    
    def _extract_key_terms(self, query: str) -> List[str]:
        """Extract key terms from query."""
        words = word_tokenize(query.lower())
        
        # Filter stop words and punctuation
        key_terms = [
            w for w in words
            if w not in self.stop_words and w.isalnum() and len(w) > 2
        ]
        
        # Remove duplicates while preserving order
        seen = set()
        unique_terms = []
        for term in key_terms:
            if term not in seen:
                seen.add(term)
                unique_terms.append(term)
        
        return unique_terms
    
    def _extract_entities(self, query: str) -> List[str]:
        """Extract named entities from query."""
        entities = []
        
        # Simple pattern-based entity extraction
        # Look for capitalized words (not at sentence start)
        words = query.split()
        for i, word in enumerate(words):
            if i > 0 and word[0].isupper() and word.isalpha():
                entities.append(word)
        
        # Look for quoted phrases
        quoted = re.findall(r'"([^"]*)"', query)
        entities.extend(quoted)
        
        return entities
    
    def _extract_temporal_references(self, query: str) -> List[str]:
        """Extract temporal references from query."""
        temporal_refs = []
        
        # Year patterns
        years = re.findall(r'\b(19|20)\d{2}\b', query)
        temporal_refs.extend(years)
        
        # Month patterns
        months = re.findall(
            r'\b(January|February|March|April|May|June|July|August|September|October|November|December|'
            r'Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\b',
            query, re.I
        )
        temporal_refs.extend(months)
        
        # Relative time patterns
        relative = re.findall(
            r'\b(yesterday|today|tomorrow|last\s+\w+|next\s+\w+|recent|recently)\b',
            query, re.I
        )
        temporal_refs.extend(relative)
        
        return temporal_refs
    
    def _detect_question_type(self, query: str) -> Optional[str]:
        """Detect the type of question."""
        query_lower = query.lower().strip()
        
        if query_lower.startswith('what'):
            return 'what'
        elif query_lower.startswith('why'):
            return 'why'
        elif query_lower.startswith('how'):
            return 'how'
        elif query_lower.startswith('when'):
            return 'when'
        elif query_lower.startswith('where'):
            return 'where'
        elif query_lower.startswith('who'):
            return 'who'
        elif query_lower.startswith('which'):
            return 'which'
        elif query_lower.startswith(('is', 'are', 'do', 'does', 'can', 'could', 'should', 'would')):
            return 'yes/no'
        
        return None
    
    def _detect_domain(self, query: str) -> Optional[str]:
        """Detect the domain of the query."""
        query_lower = query.lower()
        
        domain_scores = {}
        for domain, keywords in self.domain_keywords.items():
            score = sum(1 for keyword in keywords if keyword in query_lower)
            if score > 0:
                domain_scores[domain] = score
        
        if domain_scores:
            return max(domain_scores.items(), key=lambda x: x[1])[0]
        
        return None


class QueryRewriter:
    """Rewrites queries for better retrieval."""
    
    def __init__(self):
        """Initialize query rewriter."""
        self.analyzer = QueryAnalyzer()
    
    def rewrite_query(
        self,
        query: str,
        strategies: Optional[List[str]] = None
    ) -> List[QueryRewrite]:
        """
        Rewrite query using various strategies.
        
        Args:
            query: Original query
            strategies: List of strategies to use
            
        Returns:
            List of query rewrites
        """
        if strategies is None:
            strategies = ["synonym", "decompose", "generalize", "specify"]
        
        rewrites = []
        
        # Analyze original query
        analysis = self.analyzer.analyze_query(query)
        
        for strategy in strategies:
            if strategy == "synonym":
                rewrites.extend(self._rewrite_with_synonyms(query, analysis))
            elif strategy == "decompose":
                rewrites.extend(self._decompose_query(query, analysis))
            elif strategy == "generalize":
                rewrites.extend(self._generalize_query(query, analysis))
            elif strategy == "specify":
                rewrites.extend(self._specify_query(query, analysis))
            elif strategy == "clarify":
                rewrites.extend(self._clarify_query(query, analysis))
        
        return rewrites
    
    def _rewrite_with_synonyms(
        self,
        query: str,
        analysis: QueryAnalysis
    ) -> List[QueryRewrite]:
        """Rewrite query using synonyms."""
        rewrites = []
        
        for term in analysis.key_terms:
            synonyms = self._get_synonyms(term)
            
            for synonym in synonyms[:2]:  # Limit to 2 synonyms per term
                rewritten = query.replace(term, synonym)
                if rewritten != query:
                    rewrites.append(QueryRewrite(
                        rewritten_query=rewritten,
                        rewrite_type="synonym",
                        confidence=0.8,
                        explanation=f"Replaced '{term}' with synonym '{synonym}'"
                    ))
        
        return rewrites
    
    def _decompose_query(
        self,
        query: str,
        analysis: QueryAnalysis
    ) -> List[QueryRewrite]:
        """Decompose complex query into simpler parts."""
        rewrites = []
        
        if analysis.complexity != QueryComplexity.COMPLEX:
            return rewrites
        
        # Split on conjunctions
        parts = re.split(r'\b(?:and|or|but)\b', query, flags=re.I)
        
        if len(parts) > 1:
            for i, part in enumerate(parts):
                part = part.strip()
                if len(part) > 10:  # Meaningful part
                    rewrites.append(QueryRewrite(
                        rewritten_query=part,
                        rewrite_type="decompose",
                        confidence=0.7,
                        explanation=f"Part {i+1} of decomposed query"
                    ))
        
        return rewrites
    
    def _generalize_query(
        self,
        query: str,
        analysis: QueryAnalysis
    ) -> List[QueryRewrite]:
        """Generalize specific query."""
        rewrites = []
        
        # Remove specific entities
        generalized = query
        for entity in analysis.entities:
            generalized = generalized.replace(entity, "[entity]")
        
        # Remove temporal references
        for temporal in analysis.temporal_refs:
            generalized = generalized.replace(temporal, "[time]")
        
        if generalized != query:
            rewrites.append(QueryRewrite(
                rewritten_query=generalized,
                rewrite_type="generalize",
                confidence=0.6,
                explanation="Removed specific entities and temporal references"
            ))
        
        return rewrites
    
    def _specify_query(
        self,
        query: str,
        analysis: QueryAnalysis
    ) -> List[QueryRewrite]:
        """Add specificity to vague query."""
        rewrites = []
        
        # Add domain context if detected
        if analysis.domain:
            specified = f"{query} in {analysis.domain} context"
            rewrites.append(QueryRewrite(
                rewritten_query=specified,
                rewrite_type="specify",
                confidence=0.7,
                explanation=f"Added domain context: {analysis.domain}"
            ))
        
        # Add intent-based specifications
        if analysis.intent == QueryIntent.PROCEDURAL:
            specified = f"{query} step by step"
            rewrites.append(QueryRewrite(
                rewritten_query=specified,
                rewrite_type="specify",
                confidence=0.8,
                explanation="Added procedural specification"
            ))
        
        return rewrites
    
    def _clarify_query(
        self,
        query: str,
        analysis: QueryAnalysis
    ) -> List[QueryRewrite]:
        """Clarify ambiguous query."""
        rewrites = []
        
        # Add question words if missing
        if not analysis.question_type and not query.endswith('?'):
            if analysis.intent == QueryIntent.DEFINITIONAL:
                clarified = f"What is {query}?"
            elif analysis.intent == QueryIntent.PROCEDURAL:
                clarified = f"How to {query}?"
            elif analysis.intent == QueryIntent.CAUSAL:
                clarified = f"Why {query}?"
            else:
                clarified = f"What about {query}?"
            
            rewrites.append(QueryRewrite(
                rewritten_query=clarified,
                rewrite_type="clarify",
                confidence=0.6,
                explanation="Added question structure for clarity"
            ))
        
        return rewrites
    
    def _get_synonyms(self, word: str) -> List[str]:
        """Get synonyms for a word using WordNet."""
        synonyms = set()
        
        for synset in wordnet.synsets(word):
            for lemma in synset.lemmas():
                synonym = lemma.name().replace('_', ' ')
                if synonym.lower() != word.lower():
                    synonyms.add(synonym)
        
        return list(synonyms)


class QueryRouter:
    """Routes queries to appropriate retrieval strategies."""
    
    def __init__(self):
        """Initialize query router."""
        self.analyzer = QueryAnalyzer()
        self.routing_rules = self._define_routing_rules()
    
    def _define_routing_rules(self) -> Dict[QueryIntent, Dict[str, Any]]:
        """Define routing rules based on query intent."""
        return {
            QueryIntent.FACTUAL: {
                "retrieval_strategy": "precise",
                "reranking": True,
                "top_k": 5,
                "use_keywords": True
            },
            QueryIntent.EXPLORATORY: {
                "retrieval_strategy": "broad",
                "reranking": False,
                "top_k": 20,
                "use_semantic": True
            },
            QueryIntent.COMPARATIVE: {
                "retrieval_strategy": "multi_doc",
                "reranking": True,
                "top_k": 10,
                "group_by_source": True
            },
            QueryIntent.PROCEDURAL: {
                "retrieval_strategy": "sequential",
                "reranking": True,
                "top_k": 10,
                "preserve_order": True
            },
            QueryIntent.ANALYTICAL: {
                "retrieval_strategy": "comprehensive",
                "reranking": True,
                "top_k": 15,
                "include_context": True
            }
        }
    
    def route_query(self, query: str) -> Dict[str, Any]:
        """
        Route query to appropriate retrieval strategy.
        
        Args:
            query: The query to route
            
        Returns:
            Routing configuration
        """
        # Analyze query
        analysis = self.analyzer.analyze_query(query)
        
        # Get base routing from intent
        routing = self.routing_rules.get(
            analysis.intent,
            self.routing_rules[QueryIntent.FACTUAL]
        ).copy()
        
        # Adjust based on complexity
        if analysis.complexity == QueryComplexity.COMPLEX:
            routing["top_k"] = min(routing["top_k"] * 2, 30)
            routing["use_query_expansion"] = True
        elif analysis.complexity == QueryComplexity.SIMPLE:
            routing["top_k"] = max(routing["top_k"] // 2, 3)
        
        # Add metadata
        routing["query_analysis"] = {
            "intent": analysis.intent.value,
            "complexity": analysis.complexity.value,
            "domain": analysis.domain,
            "key_terms": analysis.key_terms
        }
        
        logger.debug(f"Routed query with intent {analysis.intent.value} to {routing['retrieval_strategy']}")
        
        return routing


# Pipeline integration functions

async def analyze_query(context: Any, **kwargs) -> Any:
    """Analyze query for pipeline context."""
    analyzer = QueryAnalyzer()
    
    analysis = analyzer.analyze_query(context.query)
    
    # Store analysis in context
    context.metadata["query_analysis"] = {
        "intent": analysis.intent.value,
        "complexity": analysis.complexity.value,
        "key_terms": analysis.key_terms,
        "entities": analysis.entities,
        "domain": analysis.domain,
        "question_type": analysis.question_type
    }
    
    # Adjust pipeline based on analysis
    if analysis.complexity == QueryComplexity.COMPLEX:
        context.config["top_k"] = min(context.config.get("top_k", 10) * 2, 30)
    
    if analysis.intent == QueryIntent.COMPARATIVE:
        context.config["group_by_source"] = True
    
    logger.info(f"Query analysis: intent={analysis.intent.value}, complexity={analysis.complexity.value}")
    
    return context


async def rewrite_query(context: Any, **kwargs) -> Any:
    """Rewrite query for better retrieval."""
    config = context.config.get("query_rewriting", {})
    
    if not config.get("enabled", True):
        return context
    
    rewriter = QueryRewriter()
    
    strategies = config.get("strategies", ["synonym", "decompose"])
    rewrites = rewriter.rewrite_query(context.query, strategies)
    
    # Store rewrites
    context.metadata["query_rewrites"] = [
        {
            "query": r.rewritten_query,
            "type": r.rewrite_type,
            "confidence": r.confidence
        }
        for r in rewrites
    ]
    
    # Use best rewrite if confidence is high
    if rewrites:
        best_rewrite = max(rewrites, key=lambda r: r.confidence)
        if best_rewrite.confidence > config.get("min_confidence", 0.7):
            context.metadata["original_query"] = context.query
            context.query = best_rewrite.rewritten_query
            logger.info(f"Rewrote query using {best_rewrite.rewrite_type}: {best_rewrite.rewritten_query}")
    
    return context


async def route_query(context: Any, **kwargs) -> Any:
    """Route query to appropriate retrieval strategy."""
    router = QueryRouter()
    
    routing = router.route_query(context.query)
    
    # Apply routing configuration
    context.config.update(routing)
    
    # Store routing decision
    context.metadata["query_routing"] = routing
    
    logger.info(f"Routed query to {routing['retrieval_strategy']} strategy")
    
    return context