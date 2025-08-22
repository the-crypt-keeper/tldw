# advanced_reranker.py - Advanced Reranking Module
"""
Advanced reranking module with multiple reranking strategies.

This module provides sophisticated reranking capabilities including:
- Cross-encoder based reranking
- LLM-based relevance scoring
- Diversity-aware reranking
- Multi-criteria reranking
- Learning-to-rank approaches
"""

import asyncio
import time
import re
from typing import List, Dict, Any, Optional, Tuple, Union
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
import numpy as np
import logging

logger = logging.getLogger(__name__)


class RerankingStrategy(Enum):
    """Available reranking strategies"""
    CROSS_ENCODER = "cross_encoder"
    LLM_SCORING = "llm_scoring"
    DIVERSITY = "diversity"
    MULTI_CRITERIA = "multi_criteria"
    LEARNED = "learned"
    HYBRID = "hybrid"


@dataclass
class RerankingConfig:
    """Configuration for reranking"""
    strategy: RerankingStrategy = RerankingStrategy.CROSS_ENCODER
    model_name: Optional[str] = None
    top_k: int = 10
    diversity_weight: float = 0.3
    relevance_weight: float = 0.7
    min_similarity_threshold: float = 0.5
    batch_size: int = 32
    temperature: float = 0.0
    criteria_weights: Dict[str, float] = field(default_factory=dict)
    enable_caching: bool = True
    cache_ttl: int = 3600


@dataclass
class ScoredDocument:
    """Document with reranking scores"""
    content: str
    metadata: Dict[str, Any]
    original_score: float
    rerank_score: float
    relevance_score: float = 0.0
    diversity_score: float = 0.0
    criteria_scores: Dict[str, float] = field(default_factory=dict)
    explanation: Optional[str] = None
    
    @property
    def final_score(self) -> float:
        """Calculate final score based on all factors"""
        return self.rerank_score


class BaseReranker(ABC):
    """Base class for all reranking strategies"""
    
    def __init__(self, config: RerankingConfig):
        self.config = config
        self._cache = {} if config.enable_caching else None
    
    @abstractmethod
    async def rerank(
        self, 
        query: str, 
        documents: List[Dict[str, Any]], 
        original_scores: Optional[List[float]] = None
    ) -> List[ScoredDocument]:
        """
        Rerank documents based on query.
        
        Args:
            query: The search query
            documents: List of documents to rerank
            original_scores: Original retrieval scores
            
        Returns:
            List of reranked documents with scores
        """
        pass
    
    def _normalize_scores(self, scores: List[float]) -> List[float]:
        """Normalize scores to [0, 1] range"""
        if not scores:
            return []
        
        min_score = min(scores)
        max_score = max(scores)
        
        if max_score == min_score:
            return [0.5] * len(scores)
        
        return [(s - min_score) / (max_score - min_score) for s in scores]
    
    def _calculate_diversity_scores(self, documents: List[str]) -> List[float]:
        """Calculate diversity scores for documents"""
        if len(documents) <= 1:
            return [1.0] * len(documents)
        
        diversity_scores = []
        
        for i, doc in enumerate(documents):
            # Simple diversity: inverse of average similarity to other docs
            similarities = []
            for j, other_doc in enumerate(documents):
                if i != j:
                    # Simple character overlap as similarity proxy
                    # In production, use proper text similarity
                    similarity = self._text_similarity(doc, other_doc)
                    similarities.append(similarity)
            
            avg_similarity = sum(similarities) / len(similarities) if similarities else 0
            diversity_score = 1.0 - avg_similarity
            diversity_scores.append(diversity_score)
        
        return diversity_scores
    
    def _text_similarity(self, text1: str, text2: str) -> float:
        """Simple text similarity calculation"""
        # Jaccard similarity on words
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())
        
        if not words1 or not words2:
            return 0.0
        
        intersection = words1.intersection(words2)
        union = words1.union(words2)
        
        return len(intersection) / len(union) if union else 0.0


class CrossEncoderReranker(BaseReranker):
    """Cross-encoder based reranking"""
    
    def __init__(self, config: RerankingConfig):
        super().__init__(config)
        self.model = None  # Would load actual cross-encoder model
        logger.info(f"Initialized CrossEncoderReranker with model: {config.model_name}")
    
    async def rerank(
        self, 
        query: str, 
        documents: List[Dict[str, Any]], 
        original_scores: Optional[List[float]] = None
    ) -> List[ScoredDocument]:
        """Rerank using cross-encoder model"""
        
        if not documents:
            return []
        
        # Extract text content
        texts = [doc.get('content', '') or doc.get('text', '') for doc in documents]
        
        # Calculate relevance scores
        # In production, this would use actual cross-encoder model
        relevance_scores = await self._calculate_relevance_scores(query, texts)
        
        # Calculate diversity scores if needed
        diversity_scores = [0.0] * len(documents)
        if self.config.diversity_weight > 0:
            diversity_scores = self._calculate_diversity_scores(texts)
        
        # Combine scores
        scored_docs = []
        for i, (doc, text) in enumerate(zip(documents, texts)):
            relevance = relevance_scores[i]
            diversity = diversity_scores[i]
            
            # Weighted combination
            final_score = (
                self.config.relevance_weight * relevance +
                self.config.diversity_weight * diversity
            )
            
            scored_doc = ScoredDocument(
                content=text,
                metadata=doc.get('metadata', {}),
                original_score=original_scores[i] if original_scores else 0.0,
                rerank_score=final_score,
                relevance_score=relevance,
                diversity_score=diversity
            )
            scored_docs.append(scored_doc)
        
        # Sort by final score
        scored_docs.sort(key=lambda x: x.final_score, reverse=True)
        
        # Return top-k
        return scored_docs[:self.config.top_k]
    
    async def _calculate_relevance_scores(self, query: str, texts: List[str]) -> List[float]:
        """Calculate relevance scores using cross-encoder"""
        # Simulate cross-encoder scoring
        # In production, use actual model inference
        
        scores = []
        for text in texts:
            # Simple relevance based on query term overlap
            query_terms = set(query.lower().split())
            text_terms = set(text.lower().split())
            
            if not query_terms:
                scores.append(0.0)
                continue
            
            overlap = len(query_terms.intersection(text_terms))
            score = overlap / len(query_terms)
            
            # Boost for exact phrase match
            if query.lower() in text.lower():
                score = min(1.0, score + 0.3)
            
            scores.append(score)
        
        return self._normalize_scores(scores)


class LLMScoringReranker(BaseReranker):
    """LLM-based relevance scoring reranker"""
    
    def __init__(self, config: RerankingConfig):
        super().__init__(config)
        self.llm_provider = config.model_name or "openai"
        logger.info(f"Initialized LLMScoringReranker with provider: {self.llm_provider}")
    
    async def rerank(
        self, 
        query: str, 
        documents: List[Dict[str, Any]], 
        original_scores: Optional[List[float]] = None
    ) -> List[ScoredDocument]:
        """Rerank by having LLM score relevance"""
        
        if not documents:
            return []
        
        # Extract text content
        texts = [doc.get('content', '') or doc.get('text', '') for doc in documents]
        
        # Score documents in batches
        relevance_scores = []
        explanations = []
        
        for i in range(0, len(texts), self.config.batch_size):
            batch_texts = texts[i:i + self.config.batch_size]
            batch_scores, batch_explanations = await self._score_batch(query, batch_texts)
            relevance_scores.extend(batch_scores)
            explanations.extend(batch_explanations)
        
        # Create scored documents
        scored_docs = []
        for i, (doc, text) in enumerate(zip(documents, texts)):
            scored_doc = ScoredDocument(
                content=text,
                metadata=doc.get('metadata', {}),
                original_score=original_scores[i] if original_scores else 0.0,
                rerank_score=relevance_scores[i],
                relevance_score=relevance_scores[i],
                explanation=explanations[i]
            )
            scored_docs.append(scored_doc)
        
        # Sort by score
        scored_docs.sort(key=lambda x: x.rerank_score, reverse=True)
        
        return scored_docs[:self.config.top_k]
    
    async def _score_batch(
        self, 
        query: str, 
        texts: List[str]
    ) -> Tuple[List[float], List[str]]:
        """Score a batch of texts using LLM"""
        # In production, this would call actual LLM API
        # For now, simulate with simple scoring
        
        scores = []
        explanations = []
        
        for text in texts:
            # Simulate LLM scoring
            score, explanation = self._simulate_llm_scoring(query, text)
            scores.append(score)
            explanations.append(explanation)
        
        return scores, explanations
    
    def _simulate_llm_scoring(self, query: str, text: str) -> Tuple[float, str]:
        """Simulate LLM relevance scoring"""
        query_terms = set(query.lower().split())
        text_terms = set(text.lower().split())
        
        overlap = len(query_terms.intersection(text_terms))
        score = min(1.0, overlap / len(query_terms)) if query_terms else 0.0
        
        if score > 0.8:
            explanation = "Highly relevant - contains all key query terms"
        elif score > 0.5:
            explanation = "Moderately relevant - contains some query terms"
        elif score > 0.2:
            explanation = "Somewhat relevant - limited term overlap"
        else:
            explanation = "Low relevance - minimal connection to query"
        
        return score, explanation


class DiversityAwareReranker(BaseReranker):
    """Reranker that optimizes for diversity while maintaining relevance"""
    
    async def rerank(
        self, 
        query: str, 
        documents: List[Dict[str, Any]], 
        original_scores: Optional[List[float]] = None
    ) -> List[ScoredDocument]:
        """Rerank with diversity optimization using MMR"""
        
        if not documents:
            return []
        
        texts = [doc.get('content', '') or doc.get('text', '') for doc in documents]
        
        # Start with relevance scores (use original if provided)
        if original_scores:
            relevance_scores = self._normalize_scores(original_scores)
        else:
            relevance_scores = await self._calculate_simple_relevance(query, texts)
        
        # MMR (Maximal Marginal Relevance) reranking
        selected_indices = []
        remaining_indices = list(range(len(documents)))
        
        while len(selected_indices) < min(self.config.top_k, len(documents)) and remaining_indices:
            mmr_scores = []
            
            for idx in remaining_indices:
                relevance = relevance_scores[idx]
                
                # Calculate max similarity to already selected docs
                max_sim = 0.0
                for selected_idx in selected_indices:
                    sim = self._text_similarity(texts[idx], texts[selected_idx])
                    max_sim = max(max_sim, sim)
                
                # MMR score
                mmr = (self.config.relevance_weight * relevance - 
                       self.config.diversity_weight * max_sim)
                mmr_scores.append((idx, mmr))
            
            # Select document with highest MMR score
            best_idx = max(mmr_scores, key=lambda x: x[1])[0]
            selected_indices.append(best_idx)
            remaining_indices.remove(best_idx)
        
        # Create scored documents in selected order
        scored_docs = []
        for rank, idx in enumerate(selected_indices):
            doc = documents[idx]
            scored_doc = ScoredDocument(
                content=texts[idx],
                metadata=doc.get('metadata', {}),
                original_score=original_scores[idx] if original_scores else 0.0,
                rerank_score=1.0 - (rank / len(selected_indices)),  # Rank-based score
                relevance_score=relevance_scores[idx]
            )
            scored_docs.append(scored_doc)
        
        return scored_docs
    
    async def _calculate_simple_relevance(self, query: str, texts: List[str]) -> List[float]:
        """Calculate simple relevance scores"""
        scores = []
        query_terms = set(query.lower().split())
        
        for text in texts:
            text_terms = set(text.lower().split())
            if not query_terms:
                scores.append(0.0)
                continue
            
            overlap = len(query_terms.intersection(text_terms))
            score = overlap / len(query_terms)
            scores.append(score)
        
        return self._normalize_scores(scores)


class MultiCriteriaReranker(BaseReranker):
    """Reranker that considers multiple criteria"""
    
    def __init__(self, config: RerankingConfig):
        super().__init__(config)
        
        # Default criteria weights if not provided
        if not config.criteria_weights:
            config.criteria_weights = {
                "relevance": 0.4,
                "recency": 0.2,
                "authority": 0.2,
                "completeness": 0.2
            }
    
    async def rerank(
        self, 
        query: str, 
        documents: List[Dict[str, Any]], 
        original_scores: Optional[List[float]] = None
    ) -> List[ScoredDocument]:
        """Rerank based on multiple criteria"""
        
        if not documents:
            return []
        
        texts = [doc.get('content', '') or doc.get('text', '') for doc in documents]
        
        # Calculate scores for each criterion
        criteria_scores = {}
        
        # Relevance
        if "relevance" in self.config.criteria_weights:
            criteria_scores["relevance"] = await self._calculate_simple_relevance(
                query, texts
            )
        
        # Recency (if timestamps available)
        if "recency" in self.config.criteria_weights:
            criteria_scores["recency"] = self._calculate_recency_scores(documents)
        
        # Authority (based on source)
        if "authority" in self.config.criteria_weights:
            criteria_scores["authority"] = self._calculate_authority_scores(documents)
        
        # Completeness (length/detail)
        if "completeness" in self.config.criteria_weights:
            criteria_scores["completeness"] = self._calculate_completeness_scores(texts)
        
        # Combine scores
        scored_docs = []
        for i, (doc, text) in enumerate(zip(documents, texts)):
            # Calculate weighted sum
            final_score = 0.0
            doc_criteria_scores = {}
            
            for criterion, weight in self.config.criteria_weights.items():
                if criterion in criteria_scores:
                    score = criteria_scores[criterion][i]
                    doc_criteria_scores[criterion] = score
                    final_score += weight * score
            
            scored_doc = ScoredDocument(
                content=text,
                metadata=doc.get('metadata', {}),
                original_score=original_scores[i] if original_scores else 0.0,
                rerank_score=final_score,
                criteria_scores=doc_criteria_scores
            )
            scored_docs.append(scored_doc)
        
        # Sort by final score
        scored_docs.sort(key=lambda x: x.rerank_score, reverse=True)
        
        return scored_docs[:self.config.top_k]
    
    async def _calculate_simple_relevance(self, query: str, texts: List[str]) -> List[float]:
        """Calculate relevance scores"""
        scores = []
        query_terms = set(query.lower().split())
        
        for text in texts:
            if not query_terms:
                scores.append(0.0)
                continue
            
            text_terms = set(text.lower().split())
            overlap = len(query_terms.intersection(text_terms))
            score = overlap / len(query_terms)
            scores.append(score)
        
        return self._normalize_scores(scores)
    
    def _calculate_recency_scores(self, documents: List[Dict[str, Any]]) -> List[float]:
        """Calculate recency scores based on timestamps"""
        scores = []
        current_time = time.time()
        
        for doc in documents:
            # Look for timestamp in metadata
            timestamp = doc.get('metadata', {}).get('timestamp', 0)
            if timestamp:
                # Score based on age (newer = higher score)
                age_days = (current_time - timestamp) / (24 * 3600)
                # Exponential decay
                score = np.exp(-age_days / 30)  # 30-day half-life
            else:
                score = 0.5  # Default for unknown age
            
            scores.append(score)
        
        return scores
    
    def _calculate_authority_scores(self, documents: List[Dict[str, Any]]) -> List[float]:
        """Calculate authority scores based on source"""
        # Simple authority scoring based on source type
        authority_map = {
            "official": 1.0,
            "academic": 0.9,
            "verified": 0.8,
            "community": 0.6,
            "unknown": 0.5
        }
        
        scores = []
        for doc in documents:
            source_type = doc.get('metadata', {}).get('source_type', 'unknown')
            score = authority_map.get(source_type, 0.5)
            scores.append(score)
        
        return scores
    
    def _calculate_completeness_scores(self, texts: List[str]) -> List[float]:
        """Calculate completeness scores based on content length and structure"""
        scores = []
        
        for text in texts:
            # Simple completeness based on length and structure
            word_count = len(text.split())
            
            # Score based on word count (sigmoid function)
            length_score = 1 / (1 + np.exp(-(word_count - 100) / 50))
            
            # Check for structural elements
            has_sections = bool(re.search(r'\n\n', text))
            has_list = bool(re.search(r'^\s*[-*]\s+', text, re.MULTILINE))
            has_numbers = bool(re.search(r'\d+', text))
            
            structure_score = sum([has_sections, has_list, has_numbers]) / 3
            
            # Combine scores
            completeness = 0.7 * length_score + 0.3 * structure_score
            scores.append(completeness)
        
        return scores


class HybridReranker(BaseReranker):
    """Combines multiple reranking strategies"""
    
    def __init__(self, config: RerankingConfig):
        super().__init__(config)
        
        # Initialize sub-rerankers
        self.rerankers = {
            "cross_encoder": CrossEncoderReranker(config),
            "diversity": DiversityAwareReranker(config),
            "multi_criteria": MultiCriteriaReranker(config)
        }
    
    async def rerank(
        self, 
        query: str, 
        documents: List[Dict[str, Any]], 
        original_scores: Optional[List[float]] = None
    ) -> List[ScoredDocument]:
        """Rerank using ensemble of strategies"""
        
        if not documents:
            return []
        
        # Get results from each reranker
        all_results = {}
        for name, reranker in self.rerankers.items():
            results = await reranker.rerank(query, documents, original_scores)
            all_results[name] = results
        
        # Combine results using rank aggregation
        # Create document ID to scores mapping
        doc_scores = {}
        
        for name, results in all_results.items():
            for rank, scored_doc in enumerate(results):
                # Use content hash as ID
                doc_id = hash(scored_doc.content)
                
                if doc_id not in doc_scores:
                    doc_scores[doc_id] = {
                        "doc": scored_doc,
                        "ranks": {},
                        "scores": {}
                    }
                
                doc_scores[doc_id]["ranks"][name] = rank
                doc_scores[doc_id]["scores"][name] = scored_doc.rerank_score
        
        # Calculate final scores using Borda count
        final_docs = []
        for doc_id, info in doc_scores.items():
            # Average rank (lower is better)
            avg_rank = sum(info["ranks"].values()) / len(info["ranks"])
            
            # Average score
            avg_score = sum(info["scores"].values()) / len(info["scores"])
            
            # Combine (convert rank to score)
            max_rank = len(documents)
            rank_score = 1.0 - (avg_rank / max_rank)
            
            # Final score
            final_score = 0.5 * rank_score + 0.5 * avg_score
            
            # Update document
            doc = info["doc"]
            doc.rerank_score = final_score
            final_docs.append(doc)
        
        # Sort by final score
        final_docs.sort(key=lambda x: x.rerank_score, reverse=True)
        
        return final_docs[:self.config.top_k]


def create_reranker(
    strategy: Union[str, RerankingStrategy], 
    **kwargs
) -> BaseReranker:
    """
    Factory function to create reranker.
    
    Args:
        strategy: Reranking strategy to use
        **kwargs: Additional configuration parameters
        
    Returns:
        Configured reranker instance
    """
    if isinstance(strategy, str):
        strategy = RerankingStrategy(strategy)
    
    config = RerankingConfig(strategy=strategy, **kwargs)
    
    if strategy == RerankingStrategy.CROSS_ENCODER:
        return CrossEncoderReranker(config)
    elif strategy == RerankingStrategy.LLM_SCORING:
        return LLMScoringReranker(config)
    elif strategy == RerankingStrategy.DIVERSITY:
        return DiversityAwareReranker(config)
    elif strategy == RerankingStrategy.MULTI_CRITERIA:
        return MultiCriteriaReranker(config)
    elif strategy == RerankingStrategy.HYBRID:
        return HybridReranker(config)
    else:
        raise ValueError(f"Unknown reranking strategy: {strategy}")


# Example usage
if __name__ == "__main__":
    async def test_reranking():
        # Test documents
        documents = [
            {"content": "Machine learning is a subset of artificial intelligence", "metadata": {"source": "wiki"}},
            {"content": "Deep learning uses neural networks with multiple layers", "metadata": {"source": "tutorial"}},
            {"content": "Python is a popular programming language for ML", "metadata": {"source": "blog"}},
            {"content": "TensorFlow and PyTorch are popular ML frameworks", "metadata": {"source": "docs"}},
            {"content": "Supervised learning uses labeled training data", "metadata": {"source": "textbook"}},
        ]
        
        query = "machine learning frameworks"
        
        # Test different rerankers
        strategies = [
            RerankingStrategy.CROSS_ENCODER,
            RerankingStrategy.DIVERSITY,
            RerankingStrategy.MULTI_CRITERIA,
            RerankingStrategy.HYBRID
        ]
        
        for strategy in strategies:
            print(f"\n{'='*50}")
            print(f"Testing {strategy.value} reranker")
            print('='*50)
            
            reranker = create_reranker(strategy, top_k=3)
            results = await reranker.rerank(query, documents)
            
            for i, doc in enumerate(results):
                print(f"\n{i+1}. Score: {doc.rerank_score:.3f}")
                print(f"   Content: {doc.content[:100]}...")
                if doc.explanation:
                    print(f"   Explanation: {doc.explanation}")
    
    asyncio.run(test_reranking())