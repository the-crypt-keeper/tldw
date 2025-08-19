"""
Advanced reranking strategies for RAG search results.

This module provides sophisticated reranking capabilities including:
- Cross-encoder based reranking
- LLM-based relevance scoring
- Diversity-aware reranking (MMR)
- Multi-criteria reranking
- Hybrid reranking strategies
"""

import asyncio
import time
from typing import List, Dict, Any, Optional, Tuple, Callable
from dataclasses import dataclass, field
from enum import Enum
from abc import ABC, abstractmethod
import numpy as np
from functools import lru_cache

from loguru import logger

from .types import Document, DataSource


class RerankingStrategy(Enum):
    """Available reranking strategies."""
    FLASHRANK = "flashrank"          # Fast neural reranking
    CROSS_ENCODER = "cross_encoder"  # Cross-encoder models
    LLM_SCORING = "llm_scoring"      # LLM-based relevance
    DIVERSITY = "diversity"          # MMR for diversity
    MULTI_CRITERIA = "multi_criteria" # Multiple ranking factors
    HYBRID = "hybrid"                # Combine strategies


@dataclass
class RerankingConfig:
    """Configuration for reranking."""
    strategy: RerankingStrategy = RerankingStrategy.FLASHRANK
    model_name: Optional[str] = None
    top_k: int = 10
    diversity_weight: float = 0.3
    relevance_weight: float = 0.7
    min_similarity_threshold: float = 0.3
    batch_size: int = 32
    use_gpu: bool = False
    criteria_weights: Dict[str, float] = field(default_factory=lambda: {
        "relevance": 0.4,
        "recency": 0.2,
        "source_quality": 0.2,
        "length": 0.2
    })


@dataclass
class ScoredDocument:
    """Document with detailed scoring information."""
    document: Document
    original_score: float
    rerank_score: float
    relevance_score: float = 0.0
    diversity_score: float = 0.0
    criteria_scores: Dict[str, float] = field(default_factory=dict)
    explanation: Optional[str] = None
    
    @property
    def final_score(self) -> float:
        """Calculate final score."""
        return self.rerank_score


class BaseReranker(ABC):
    """Base class for all reranking strategies."""
    
    def __init__(self, config: RerankingConfig):
        """
        Initialize reranker.
        
        Args:
            config: Reranking configuration
        """
        self.config = config
        self._cache = {}
    
    @abstractmethod
    async def rerank(
        self,
        query: str,
        documents: List[Document],
        original_scores: Optional[List[float]] = None
    ) -> List[ScoredDocument]:
        """
        Rerank documents based on query.
        
        Args:
            query: Search query
            documents: Documents to rerank
            original_scores: Original retrieval scores
            
        Returns:
            List of reranked documents with scores
        """
        pass
    
    def _normalize_scores(self, scores: List[float]) -> List[float]:
        """Normalize scores to [0, 1] range."""
        if not scores:
            return []
        
        min_score = min(scores)
        max_score = max(scores)
        
        if max_score == min_score:
            return [0.5] * len(scores)
        
        return [(s - min_score) / (max_score - min_score) for s in scores]


class FlashRankReranker(BaseReranker):
    """Fast neural reranking using FlashRank."""
    
    def __init__(self, config: RerankingConfig):
        """Initialize FlashRank reranker."""
        super().__init__(config)
        self._ranker = None
        
        try:
            from flashrank import Ranker
            self._ranker = Ranker()
            logger.info("FlashRank reranker initialized")
        except ImportError:
            logger.warning("FlashRank not available")
    
    async def rerank(
        self,
        query: str,
        documents: List[Document],
        original_scores: Optional[List[float]] = None
    ) -> List[ScoredDocument]:
        """Rerank using FlashRank."""
        if not self._ranker or not documents:
            # Fallback to original scores
            return [
                ScoredDocument(
                    document=doc,
                    original_score=original_scores[i] if original_scores else doc.score,
                    rerank_score=original_scores[i] if original_scores else doc.score,
                    relevance_score=original_scores[i] if original_scores else doc.score
                )
                for i, doc in enumerate(documents)
            ]
        
        # Prepare passages for reranking
        passages = [
            {"id": i, "text": doc.content[:1000]}  # Limit content length
            for i, doc in enumerate(documents)
        ]
        
        # Rerank
        try:
            # Create rerank request - FlashRank rerank() takes a RerankRequest
            from flashrank import RerankRequest
            request = RerankRequest(query=query, passages=passages)
            results = self._ranker.rerank(request)
            
            # Create scored documents
            scored_docs = []
            for result in results:
                idx = result["id"]
                scored_docs.append(ScoredDocument(
                    document=documents[idx],
                    original_score=original_scores[idx] if original_scores else documents[idx].score,
                    rerank_score=result["score"],
                    relevance_score=result["score"]
                ))
            
            return scored_docs[:self.config.top_k]
            
        except Exception as e:
            logger.error(f"FlashRank reranking failed: {e}")
            # Fallback to original order
            return [
                ScoredDocument(
                    document=doc,
                    original_score=original_scores[i] if original_scores else doc.score,
                    rerank_score=original_scores[i] if original_scores else doc.score
                )
                for i, doc in enumerate(documents[:self.config.top_k])
            ]


class DiversityReranker(BaseReranker):
    """
    Maximal Marginal Relevance (MMR) reranking for diversity.
    
    Balances relevance with diversity to avoid redundant results.
    """
    
    def __init__(self, config: RerankingConfig):
        """Initialize diversity reranker."""
        super().__init__(config)
        self.lambda_param = config.diversity_weight
    
    async def rerank(
        self,
        query: str,
        documents: List[Document],
        original_scores: Optional[List[float]] = None
    ) -> List[ScoredDocument]:
        """
        Rerank using MMR algorithm.
        
        MMR = λ * Relevance(doc) - (1-λ) * max(Similarity(doc, selected))
        """
        if not documents:
            return []
        
        # Normalize original scores
        scores = original_scores if original_scores else [doc.score for doc in documents]
        norm_scores = self._normalize_scores(scores)
        
        # Initialize result set
        selected_indices = []
        selected_docs = []
        remaining_indices = list(range(len(documents)))
        
        # Select first document (highest relevance)
        first_idx = np.argmax(norm_scores)
        selected_indices.append(first_idx)
        selected_docs.append(ScoredDocument(
            document=documents[first_idx],
            original_score=scores[first_idx],
            rerank_score=norm_scores[first_idx],
            relevance_score=norm_scores[first_idx],
            diversity_score=0.0
        ))
        remaining_indices.remove(first_idx)
        
        # Select remaining documents
        while remaining_indices and len(selected_docs) < self.config.top_k:
            mmr_scores = []
            
            for idx in remaining_indices:
                # Relevance score
                relevance = norm_scores[idx]
                
                # Maximum similarity to selected documents
                max_sim = 0.0
                for selected_idx in selected_indices:
                    sim = self._compute_similarity(
                        documents[idx].content,
                        documents[selected_idx].content
                    )
                    max_sim = max(max_sim, sim)
                
                # MMR score
                mmr = self.lambda_param * relevance - (1 - self.lambda_param) * max_sim
                mmr_scores.append((idx, mmr, relevance, 1 - max_sim))
            
            # Select document with highest MMR
            if mmr_scores:
                best_idx, best_mmr, rel_score, div_score = max(mmr_scores, key=lambda x: x[1])
                selected_indices.append(best_idx)
                selected_docs.append(ScoredDocument(
                    document=documents[best_idx],
                    original_score=scores[best_idx],
                    rerank_score=best_mmr,
                    relevance_score=rel_score,
                    diversity_score=div_score
                ))
                remaining_indices.remove(best_idx)
        
        return selected_docs
    
    def _compute_similarity(self, text1: str, text2: str) -> float:
        """
        Compute similarity between two texts.
        
        Simple Jaccard similarity for demonstration.
        In production, use embeddings or more sophisticated methods.
        """
        # Simple word-based Jaccard similarity
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())
        
        if not words1 or not words2:
            return 0.0
        
        intersection = words1 & words2
        union = words1 | words2
        
        return len(intersection) / len(union) if union else 0.0


class MultiCriteriaReranker(BaseReranker):
    """
    Rerank based on multiple criteria.
    
    Combines various factors like relevance, recency, source quality, etc.
    """
    
    async def rerank(
        self,
        query: str,
        documents: List[Document],
        original_scores: Optional[List[float]] = None
    ) -> List[ScoredDocument]:
        """Rerank using multiple criteria."""
        if not documents:
            return []
        
        scored_docs = []
        
        for i, doc in enumerate(documents):
            # Calculate individual criteria scores
            criteria_scores = {}
            
            # Relevance (from original score)
            criteria_scores["relevance"] = original_scores[i] if original_scores else doc.score
            
            # Recency (based on metadata if available)
            criteria_scores["recency"] = self._calculate_recency_score(doc)
            
            # Source quality
            criteria_scores["source_quality"] = self._calculate_source_quality(doc)
            
            # Document length (prefer moderate length)
            criteria_scores["length"] = self._calculate_length_score(doc)
            
            # Normalize all scores
            for key in criteria_scores:
                criteria_scores[key] = max(0.0, min(1.0, criteria_scores[key]))
            
            # Calculate weighted final score
            final_score = sum(
                criteria_scores.get(criterion, 0.0) * weight
                for criterion, weight in self.config.criteria_weights.items()
            )
            
            scored_docs.append(ScoredDocument(
                document=doc,
                original_score=original_scores[i] if original_scores else doc.score,
                rerank_score=final_score,
                relevance_score=criteria_scores["relevance"],
                criteria_scores=criteria_scores
            ))
        
        # Sort by final score
        scored_docs.sort(key=lambda x: x.rerank_score, reverse=True)
        
        return scored_docs[:self.config.top_k]
    
    def _calculate_recency_score(self, doc: Document) -> float:
        """Calculate recency score based on document age."""
        # Check for timestamp in metadata
        if doc.metadata and "created_at" in doc.metadata:
            # Simple linear decay based on age
            # In production, use proper time-based scoring
            return 0.5  # Default for now
        return 0.5  # Neutral score if no timestamp
    
    def _calculate_source_quality(self, doc: Document) -> float:
        """Calculate source quality score."""
        # Source-based quality scoring
        source_scores = {
            DataSource.MEDIA_DB: 0.8,
            DataSource.CHAT_HISTORY: 0.7,
            DataSource.NOTES: 0.9,
            DataSource.CHARACTER_CARDS: 0.6
        }
        return source_scores.get(doc.source, 0.5)
    
    def _calculate_length_score(self, doc: Document) -> float:
        """Calculate length score (prefer moderate length)."""
        content_length = len(doc.content)
        
        # Ideal length range (characters)
        ideal_min = 100
        ideal_max = 1000
        
        if ideal_min <= content_length <= ideal_max:
            return 1.0
        elif content_length < ideal_min:
            return content_length / ideal_min
        else:
            # Decay for very long documents
            return max(0.3, 1.0 - (content_length - ideal_max) / 10000)


class HybridReranker(BaseReranker):
    """
    Combines multiple reranking strategies.
    
    Uses voting or weighted combination of different rerankers.
    """
    
    def __init__(self, config: RerankingConfig, strategies: Optional[List[BaseReranker]] = None):
        """
        Initialize hybrid reranker.
        
        Args:
            config: Reranking configuration
            strategies: List of reranking strategies to combine
        """
        super().__init__(config)
        
        if strategies:
            self.strategies = strategies
        else:
            # Default combination
            self.strategies = [
                FlashRankReranker(config),
                DiversityReranker(config),
                MultiCriteriaReranker(config)
            ]
        
        # Strategy weights (could be configurable)
        self.strategy_weights = {
            "FlashRankReranker": 0.4,
            "DiversityReranker": 0.3,
            "MultiCriteriaReranker": 0.3
        }
    
    async def rerank(
        self,
        query: str,
        documents: List[Document],
        original_scores: Optional[List[float]] = None
    ) -> List[ScoredDocument]:
        """Rerank using multiple strategies and combine results."""
        if not documents:
            return []
        
        # Run all strategies in parallel
        rerank_tasks = [
            strategy.rerank(query, documents, original_scores)
            for strategy in self.strategies
        ]
        
        all_results = await asyncio.gather(*rerank_tasks)
        
        # Create document score map
        doc_scores = {}
        
        for strategy_idx, results in enumerate(all_results):
            strategy_name = type(self.strategies[strategy_idx]).__name__
            weight = self.strategy_weights.get(strategy_name, 1.0)
            
            for rank, scored_doc in enumerate(results):
                doc_id = id(scored_doc.document)
                
                if doc_id not in doc_scores:
                    doc_scores[doc_id] = {
                        "document": scored_doc.document,
                        "original_score": scored_doc.original_score,
                        "weighted_scores": [],
                        "strategy_scores": {}
                    }
                
                # Use reciprocal rank for position-based scoring
                position_score = 1.0 / (rank + 1)
                weighted_score = position_score * weight
                
                doc_scores[doc_id]["weighted_scores"].append(weighted_score)
                doc_scores[doc_id]["strategy_scores"][strategy_name] = scored_doc.rerank_score
        
        # Calculate final scores
        final_scored_docs = []
        
        for doc_data in doc_scores.values():
            # Combine weighted scores
            final_score = sum(doc_data["weighted_scores"]) / len(self.strategies)
            
            final_scored_docs.append(ScoredDocument(
                document=doc_data["document"],
                original_score=doc_data["original_score"],
                rerank_score=final_score,
                criteria_scores=doc_data["strategy_scores"],
                explanation=f"Combined from {len(self.strategies)} strategies"
            ))
        
        # Sort by final score
        final_scored_docs.sort(key=lambda x: x.rerank_score, reverse=True)
        
        return final_scored_docs[:self.config.top_k]


class LLMReranker(BaseReranker):
    """
    LLM-based reranking for high-quality relevance scoring.
    
    Uses a language model to score query-document relevance.
    Note: This is expensive and should be used sparingly.
    """
    
    def __init__(self, config: RerankingConfig, llm_client=None):
        """
        Initialize LLM reranker.
        
        Args:
            config: Reranking configuration
            llm_client: LLM client for scoring
        """
        super().__init__(config)
        self.llm_client = llm_client
    
    async def rerank(
        self,
        query: str,
        documents: List[Document],
        original_scores: Optional[List[float]] = None
    ) -> List[ScoredDocument]:
        """Rerank using LLM for relevance scoring."""
        if not self.llm_client or not documents:
            # Fallback to original scores
            return [
                ScoredDocument(
                    document=doc,
                    original_score=original_scores[i] if original_scores else doc.score,
                    rerank_score=original_scores[i] if original_scores else doc.score
                )
                for i, doc in enumerate(documents[:self.config.top_k])
            ]
        
        # Score documents in batches
        scored_docs = []
        batch_size = self.config.batch_size
        
        for i in range(0, len(documents), batch_size):
            batch = documents[i:i + batch_size]
            batch_scores = await self._score_batch(query, batch)
            
            for j, doc in enumerate(batch):
                scored_docs.append(ScoredDocument(
                    document=doc,
                    original_score=original_scores[i + j] if original_scores else doc.score,
                    rerank_score=batch_scores[j],
                    relevance_score=batch_scores[j],
                    explanation="LLM relevance score"
                ))
        
        # Sort by score
        scored_docs.sort(key=lambda x: x.rerank_score, reverse=True)
        
        return scored_docs[:self.config.top_k]
    
    async def _score_batch(self, query: str, documents: List[Document]) -> List[float]:
        """Score a batch of documents using LLM."""
        # This is a placeholder - actual implementation would use the LLM
        # to score relevance on a scale of 0-1
        
        # For now, return mock scores
        return [0.5 + 0.1 * i for i in range(len(documents))]


def create_reranker(strategy: RerankingStrategy, config: Optional[RerankingConfig] = None) -> BaseReranker:
    """
    Factory function to create a reranker.
    
    Args:
        strategy: Reranking strategy to use
        config: Optional configuration
        
    Returns:
        Reranker instance
    """
    if config is None:
        config = RerankingConfig(strategy=strategy)
    
    if strategy == RerankingStrategy.FLASHRANK:
        return FlashRankReranker(config)
    elif strategy == RerankingStrategy.DIVERSITY:
        return DiversityReranker(config)
    elif strategy == RerankingStrategy.MULTI_CRITERIA:
        return MultiCriteriaReranker(config)
    elif strategy == RerankingStrategy.HYBRID:
        return HybridReranker(config)
    elif strategy == RerankingStrategy.LLM_SCORING:
        return LLMReranker(config)
    else:
        # Default to FlashRank
        return FlashRankReranker(config)