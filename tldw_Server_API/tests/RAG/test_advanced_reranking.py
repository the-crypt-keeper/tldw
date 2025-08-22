"""
Tests for advanced reranking functionality.
"""

import pytest
import asyncio
from typing import List
import numpy as np

from tldw_Server_API.app.core.RAG.rag_service.advanced_reranking import (
    RerankingStrategy, RerankingConfig, ScoredDocument,
    FlashRankReranker, DiversityReranker, MultiCriteriaReranker,
    HybridReranker, LLMReranker, create_reranker
)
from tldw_Server_API.app.core.RAG.rag_service.types import Document, DataSource


def create_test_documents(n: int = 5) -> List[Document]:
    """Create test documents with varying content."""
    docs = []
    for i in range(n):
        docs.append(Document(
            id=f"doc_{i}",
            content=f"Document {i} content. " + ("test " * (i + 1)),
            metadata={"index": i, "created_at": f"2024-01-{i+1:02d}"},
            score=0.9 - (i * 0.1),  # Decreasing scores
            source=DataSource.MEDIA_DB if i % 2 == 0 else DataSource.NOTES
        ))
    return docs


class TestFlashRankReranker:
    """Test FlashRank reranking."""
    
    @pytest.mark.asyncio
    async def test_flashrank_fallback(self):
        """Test FlashRank fallback when not available."""
        config = RerankingConfig(strategy=RerankingStrategy.FLASHRANK, top_k=3)
        reranker = FlashRankReranker(config)
        
        docs = create_test_documents()
        query = "test query"
        
        results = await reranker.rerank(query, docs)
        
        # Should return documents (may be reordered by FlashRank)
        assert len(results) <= config.top_k
        assert all(isinstance(r, ScoredDocument) for r in results)
        # Check that original scores are preserved even if order changes
        original_scores = {doc.score for doc in docs[:config.top_k]}
        result_original_scores = {r.original_score for r in results}
        assert len(result_original_scores & original_scores) > 0  # Some overlap
    
    @pytest.mark.asyncio
    async def test_empty_documents(self):
        """Test handling of empty document list."""
        config = RerankingConfig()
        reranker = FlashRankReranker(config)
        
        results = await reranker.rerank("query", [])
        assert results == []


class TestDiversityReranker:
    """Test diversity-aware reranking."""
    
    @pytest.mark.asyncio
    async def test_mmr_reranking(self):
        """Test MMR algorithm for diversity."""
        config = RerankingConfig(
            strategy=RerankingStrategy.DIVERSITY,
            diversity_weight=0.5,
            top_k=3
        )
        reranker = DiversityReranker(config)
        
        # Create documents with similar content
        docs = [
            Document(id="1", content="machine learning algorithms", metadata={}, score=0.9, source=DataSource.MEDIA_DB),
            Document(id="2", content="machine learning models", metadata={}, score=0.85, source=DataSource.MEDIA_DB),
            Document(id="3", content="deep learning networks", metadata={}, score=0.8, source=DataSource.MEDIA_DB),
            Document(id="4", content="database systems", metadata={}, score=0.75, source=DataSource.MEDIA_DB),
        ]
        
        results = await reranker.rerank("machine learning", docs)
        
        assert len(results) == 3
        # First document should be highest relevance
        assert results[0].document.id == "1"
        # Should promote diversity - database doc might appear despite lower score
        doc_ids = [r.document.id for r in results]
        assert len(set(doc_ids)) == 3  # All unique
    
    @pytest.mark.asyncio
    async def test_diversity_scores(self):
        """Test that diversity scores are calculated."""
        config = RerankingConfig(
            strategy=RerankingStrategy.DIVERSITY,
            diversity_weight=0.3,
            top_k=5
        )
        reranker = DiversityReranker(config)
        
        docs = create_test_documents()
        results = await reranker.rerank("test", docs)
        
        # Check diversity scores are set
        for i, result in enumerate(results):
            if i > 0:  # First doc has no diversity score
                assert hasattr(result, 'diversity_score')
                assert 0 <= result.diversity_score <= 1
    
    @pytest.mark.asyncio 
    async def test_similarity_computation(self):
        """Test document similarity computation."""
        config = RerankingConfig()
        reranker = DiversityReranker(config)
        
        # Test identical documents
        sim1 = reranker._compute_similarity("test content", "test content")
        assert sim1 == 1.0
        
        # Test completely different documents
        sim2 = reranker._compute_similarity("alpha beta", "gamma delta")
        assert sim2 == 0.0
        
        # Test partial overlap
        sim3 = reranker._compute_similarity("test content here", "test different here")
        assert 0 < sim3 < 1


class TestMultiCriteriaReranker:
    """Test multi-criteria reranking."""
    
    @pytest.mark.asyncio
    async def test_criteria_scoring(self):
        """Test multiple criteria scoring."""
        config = RerankingConfig(
            strategy=RerankingStrategy.MULTI_CRITERIA,
            criteria_weights={
                "relevance": 0.4,
                "recency": 0.2,
                "source_quality": 0.2,
                "length": 0.2
            },
            top_k=3
        )
        reranker = MultiCriteriaReranker(config)
        
        docs = create_test_documents()
        results = await reranker.rerank("test", docs)
        
        assert len(results) == 3
        # Check all criteria are scored
        for result in results:
            assert "relevance" in result.criteria_scores
            assert "source_quality" in result.criteria_scores
            assert "length" in result.criteria_scores
            assert all(0 <= score <= 1 for score in result.criteria_scores.values())
    
    @pytest.mark.asyncio
    async def test_source_quality_scoring(self):
        """Test source quality affects ranking."""
        config = RerankingConfig(
            strategy=RerankingStrategy.MULTI_CRITERIA,
            criteria_weights={
                "relevance": 0.3,
                "source_quality": 0.7,  # Heavy weight on source
                "recency": 0.0,
                "length": 0.0
            }
        )
        reranker = MultiCriteriaReranker(config)
        
        # Create docs with different sources
        docs = [
            Document(id="1", content="test", metadata={}, score=0.5, source=DataSource.MEDIA_DB),
            Document(id="2", content="test", metadata={}, score=0.9, source=DataSource.NOTES),  # Better source
            Document(id="3", content="test", metadata={}, score=0.7, source=DataSource.CHARACTER_CARDS),
        ]
        
        results = await reranker.rerank("test", docs)
        
        # Notes source should rank higher due to source quality weight
        assert results[0].document.source == DataSource.NOTES
    
    @pytest.mark.asyncio
    async def test_length_scoring(self):
        """Test document length scoring."""
        config = RerankingConfig(
            strategy=RerankingStrategy.MULTI_CRITERIA,
            criteria_weights={
                "length": 1.0,  # Only consider length
                "relevance": 0.0,
                "source_quality": 0.0,
                "recency": 0.0
            }
        )
        reranker = MultiCriteriaReranker(config)
        
        # Create docs with different lengths
        docs = [
            Document(id="1", content="x" * 50, metadata={}, score=0.5, source=DataSource.MEDIA_DB),   # Too short
            Document(id="2", content="x" * 500, metadata={}, score=0.5, source=DataSource.MEDIA_DB),  # Ideal
            Document(id="3", content="x" * 5000, metadata={}, score=0.5, source=DataSource.MEDIA_DB), # Too long
        ]
        
        results = await reranker.rerank("test", docs)
        
        # Medium length document should rank highest
        assert results[0].document.id == "2"


class TestHybridReranker:
    """Test hybrid reranking strategy."""
    
    @pytest.mark.asyncio
    async def test_strategy_combination(self):
        """Test combining multiple strategies."""
        config = RerankingConfig(
            strategy=RerankingStrategy.HYBRID,
            top_k=3
        )
        
        # Create with specific strategies
        strategies = [
            DiversityReranker(config),
            MultiCriteriaReranker(config)
        ]
        
        reranker = HybridReranker(config, strategies)
        
        docs = create_test_documents()
        results = await reranker.rerank("test", docs)
        
        assert len(results) == 3
        # Check combined scoring
        for result in results:
            assert result.rerank_score > 0
            assert "DiversityReranker" in result.criteria_scores
            assert "MultiCriteriaReranker" in result.criteria_scores
    
    @pytest.mark.asyncio
    async def test_default_strategies(self):
        """Test hybrid with default strategy combination."""
        config = RerankingConfig(
            strategy=RerankingStrategy.HYBRID,
            top_k=5
        )
        reranker = HybridReranker(config)
        
        docs = create_test_documents()
        results = await reranker.rerank("test", docs)
        
        assert len(results) == 5
        # Should have explanation
        assert all(r.explanation for r in results)
        assert "strategies" in results[0].explanation
    
    @pytest.mark.asyncio
    async def test_weighted_voting(self):
        """Test weighted voting in hybrid reranking."""
        config = RerankingConfig(top_k=3)
        reranker = HybridReranker(config)
        
        # Adjust weights
        reranker.strategy_weights = {
            "FlashRankReranker": 0.8,  # Heavy weight on one strategy
            "DiversityReranker": 0.1,
            "MultiCriteriaReranker": 0.1
        }
        
        docs = create_test_documents()
        results = await reranker.rerank("test", docs)
        
        assert len(results) == 3
        # Results should be influenced by weighted voting
        assert all(r.rerank_score >= 0 for r in results)


class TestLLMReranker:
    """Test LLM-based reranking."""
    
    @pytest.mark.asyncio
    async def test_llm_fallback(self):
        """Test LLM reranker fallback when no client."""
        config = RerankingConfig(
            strategy=RerankingStrategy.LLM_SCORING,
            top_k=3
        )
        reranker = LLMReranker(config, llm_client=None)
        
        docs = create_test_documents()
        results = await reranker.rerank("test", docs)
        
        # Should fallback to original scores
        assert len(results) == 3
        assert results[0].original_score == docs[0].score
    
    @pytest.mark.asyncio
    async def test_batch_scoring(self):
        """Test batch scoring with mock LLM."""
        config = RerankingConfig(
            strategy=RerankingStrategy.LLM_SCORING,
            batch_size=2,
            top_k=5
        )
        
        # Mock LLM client
        class MockLLM:
            async def score(self, query, docs):
                return [0.5] * len(docs)
        
        reranker = LLMReranker(config, llm_client=MockLLM())
        
        docs = create_test_documents()
        results = await reranker.rerank("test", docs)
        
        assert len(results) == 5
        # All should have LLM explanation
        assert all("LLM" in r.explanation for r in results)


class TestRerankingFactory:
    """Test reranker factory function."""
    
    def test_create_flashrank_reranker(self):
        """Test creating FlashRank reranker."""
        reranker = create_reranker(RerankingStrategy.FLASHRANK)
        assert isinstance(reranker, FlashRankReranker)
    
    def test_create_diversity_reranker(self):
        """Test creating diversity reranker."""
        config = RerankingConfig(diversity_weight=0.7)
        reranker = create_reranker(RerankingStrategy.DIVERSITY, config)
        assert isinstance(reranker, DiversityReranker)
        assert reranker.lambda_param == 0.7
    
    def test_create_hybrid_reranker(self):
        """Test creating hybrid reranker."""
        reranker = create_reranker(RerankingStrategy.HYBRID)
        assert isinstance(reranker, HybridReranker)
        assert len(reranker.strategies) > 0
    
    def test_default_strategy(self):
        """Test default strategy creation."""
        # Invalid strategy should default to FlashRank
        reranker = create_reranker(None, RerankingConfig())
        assert isinstance(reranker, FlashRankReranker)


class TestIntegration:
    """Integration tests for reranking."""
    
    @pytest.mark.asyncio
    async def test_pipeline_integration(self):
        """Test reranking in a pipeline."""
        # Create documents with various characteristics
        docs = [
            Document(id="1", content="machine learning tutorial for beginners", 
                    metadata={}, score=0.8, source=DataSource.NOTES),
            Document(id="2", content="advanced machine learning techniques", 
                    metadata={}, score=0.9, source=DataSource.MEDIA_DB),
            Document(id="3", content="database optimization guide", 
                    metadata={}, score=0.7, source=DataSource.MEDIA_DB),
            Document(id="4", content="machine learning machine learning repeated", 
                    metadata={}, score=0.85, source=DataSource.CHARACTER_CARDS),
            Document(id="5", content="introduction to deep learning", 
                    metadata={}, score=0.75, source=DataSource.NOTES),
        ]
        
        # Test different strategies
        strategies = [
            RerankingStrategy.FLASHRANK,
            RerankingStrategy.DIVERSITY,
            RerankingStrategy.MULTI_CRITERIA,
            RerankingStrategy.HYBRID
        ]
        
        for strategy in strategies:
            config = RerankingConfig(strategy=strategy, top_k=3)
            reranker = create_reranker(strategy, config)
            
            results = await reranker.rerank("machine learning", docs)
            
            assert len(results) <= 3
            assert all(isinstance(r, ScoredDocument) for r in results)
            # Results should be sorted by rerank score
            scores = [r.rerank_score for r in results]
            assert scores == sorted(scores, reverse=True)
    
    @pytest.mark.asyncio
    async def test_performance(self):
        """Test reranking performance with many documents."""
        import time
        
        # Create many documents
        docs = create_test_documents(100)
        
        config = RerankingConfig(top_k=10)
        reranker = create_reranker(RerankingStrategy.DIVERSITY, config)
        
        start = time.time()
        results = await reranker.rerank("test query", docs)
        duration = time.time() - start
        
        assert len(results) == 10
        # Should complete reasonably quickly (under 1 second)
        assert duration < 1.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])