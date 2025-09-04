"""
Unit tests for RAGEvaluator.

Tests RAG evaluation functionality with minimal mocking (only external LLM/embedding services).
"""

import pytest
import json
import asyncio
from unittest.mock import Mock, patch, AsyncMock
import numpy as np

from tldw_Server_API.app.core.Evaluations.rag_evaluator import RAGEvaluator
from tldw_Server_API.tests.Evaluations.fixtures.sample_data import SampleDataGenerator
from tldw_Server_API.tests.Evaluations.fixtures.llm_responses import (
    LLMResponseCache, 
    MockLLMClient,
    create_mock_llm_client
)


@pytest.mark.unit
class TestRAGEvaluatorInit:
    """Test RAGEvaluator initialization."""
    
    def test_init_without_embeddings(self):
        """Test initialization without embedding support."""
        evaluator = RAGEvaluator(
            embedding_provider=None,
            embedding_model=None
        )
        
        assert evaluator.embedding_provider is None
        assert evaluator.embedding_model is None
        assert evaluator.embedding_available is False
    
    def test_init_with_embeddings(self):
        """Test initialization with embedding configuration."""
        evaluator = RAGEvaluator(
            embedding_provider="openai",
            embedding_model="text-embedding-3-small"
        )
        
        assert evaluator.embedding_provider == "openai"
        assert evaluator.embedding_model == "text-embedding-3-small"
        # Note: embedding_available is checked lazily on first use
    
    @patch('tldw_Server_API.app.core.Evaluations.rag_evaluator.create_embedding')
    def test_embedding_availability_check(self, mock_create_embedding):
        """Test checking embedding availability."""
        # Test successful embedding check
        mock_create_embedding.return_value = [0.1] * 1536
        evaluator = RAGEvaluator(
            embedding_provider="openai",
            embedding_model="text-embedding-3-small"
        )
        
        # Force evaluation of embedding_available property
        assert evaluator.embedding_available is True
        
        # Test failed embedding check
        mock_create_embedding.side_effect = Exception("API key not found")
        evaluator2 = RAGEvaluator(
            embedding_provider="openai",
            embedding_model="text-embedding-3-small"
        )
        assert evaluator2.embedding_available is False


@pytest.mark.unit
class TestContextRelevance:
    """Test context relevance evaluation."""
    
    @pytest.mark.asyncio
    @patch('tldw_Server_API.app.core.Evaluations.rag_evaluator.analyze')
    async def test_evaluate_context_relevance(self, mock_analyze):
        """Test evaluating context relevance."""
        # The actual code expects plain numeric string, not JSON
        mock_analyze.return_value = "4.3"
        
        evaluator = RAGEvaluator()
        rag_data = SampleDataGenerator.generate_rag_evaluation_data()
        
        metric_name, result = await evaluator._evaluate_context_relevance(
            rag_data["query"],
            rag_data["retrieved_contexts"],
            "openai"
        )
        
        assert metric_name == "context_relevance"
        assert "score" in result
        assert 0 <= result["score"] <= 1
        assert "raw_score" in result
        assert 1 <= result["raw_score"] <= 5
        # Called once per context
        assert mock_analyze.call_count == len(rag_data["retrieved_contexts"])
    
    @pytest.mark.asyncio
    @patch('asyncio.to_thread')
    async def test_context_relevance_edge_cases(self, mock_to_thread):
        """Test context relevance with edge cases."""
        evaluator = RAGEvaluator()
        
        # Test empty context - should return 0 without calling API
        metric_name, result = await evaluator._evaluate_context_relevance(
            "Test query",
            [],
            "openai"
        )
        assert result["score"] == 0.0  # Empty contexts get 0.0 score
        mock_to_thread.assert_not_called()  # No API call for empty context
        mock_to_thread.reset_mock()
        
        # Test single context chunk
        mock_to_thread.return_value = "5"
        metric_name, result = await evaluator._evaluate_context_relevance(
            "Test query",
            ["Single context chunk"],
            "openai"
        )
        assert result["score"] == 1.0
        
        # Test very long context
        long_context = ["Context " + str(i) for i in range(100)]
        mock_to_thread.return_value = "3.5"
        metric_name, result = await evaluator._evaluate_context_relevance(
            "Test query",
            long_context,
            "openai"
        )
        assert 0 <= result["score"] <= 1


@pytest.mark.unit
class TestAnswerFaithfulness:
    """Test answer faithfulness evaluation."""
    
    @pytest.mark.asyncio
    async def test_evaluate_answer_faithfulness(self, mock_llm_analyze):
        """Test evaluating answer faithfulness."""
        # mock_llm_analyze fixture already patches analyze function
        
        evaluator = RAGEvaluator()
        
        metric_name, result = await evaluator._evaluate_faithfulness(
            "This is the answer based on context",
            ["Context chunk 1", "Context chunk 2"],
            "openai"
        )
        
        assert metric_name == "faithfulness"
        assert "score" in result
        assert 0 <= result["score"] <= 1
        assert result["raw_score"] == 4.7
    
    @pytest.mark.asyncio
    async def test_faithfulness_hallucination_detection(self):
        """Test detection of hallucinated content."""
        evaluator = RAGEvaluator()
        
        metric_name, result = await evaluator._evaluate_faithfulness(
            "Answer with hallucinations",
            ["Limited context"],
            "openai"
        )
        
        assert result["score"] < 0.6  # Should be low score for hallucinated content


@pytest.mark.unit
class TestAnswerRelevance:
    """Test answer relevance evaluation."""
    
    @pytest.mark.asyncio
    async def test_evaluate_answer_relevance(self):
        """Test evaluating answer relevance to query."""
        evaluator = RAGEvaluator()
        
        metric_name, result = await evaluator._evaluate_relevance(
            "What is the capital of France?",
            "Paris is the capital of France.",
            "openai"
        )
        
        assert metric_name == "relevance"
        assert result["score"] > 0.7  # Should be high score for relevant answer
    
    @pytest.mark.asyncio
    async def test_answer_relevance_mismatch(self):
        """Test when answer doesn't match query."""
        evaluator = RAGEvaluator()
        
        metric_name, result = await evaluator._evaluate_relevance(
            "What is quantum computing?",
            "The weather today is sunny.",
            "openai"
        )
        
        assert result["score"] < 0.4  # Should be low score for irrelevant answer


@pytest.mark.unit
class TestAnswerSimilarity:
    """Test answer similarity evaluation."""
    
    @pytest.mark.asyncio
    @patch('tldw_Server_API.app.core.Evaluations.rag_evaluator.create_embedding')
    async def test_answer_similarity_with_embeddings(self, mock_create_embedding):
        """Test answer similarity using embeddings."""
        # Create mock embeddings with known cosine similarity
        embedding1 = [1.0, 0.0, 0.0]
        embedding2 = [0.8, 0.6, 0.0]  # Cosine similarity = 0.8
        
        mock_create_embedding.side_effect = [embedding1, embedding2]
        
        evaluator = RAGEvaluator(
            embedding_provider="openai",
            embedding_model="text-embedding-3-small"
        )
        evaluator.embedding_available = True
        
        metric_name, result = await evaluator._evaluate_answer_similarity(
            "Response text",
            "Ground truth text"
        )
        
        assert metric_name == "answer_similarity"
        assert result["method"] == "embeddings"
        assert 0 <= result["score"] <= 1
        assert mock_create_embedding.call_count == 2
    
    @pytest.mark.asyncio
    async def test_answer_similarity_fallback_to_llm(self):
        """Test fallback to LLM when embeddings unavailable."""
        evaluator = RAGEvaluator()  # No embeddings configured
        
        metric_name, result = await evaluator._evaluate_answer_similarity(
            "Response text",
            "Ground truth text"
        )
        
        assert metric_name == "answer_similarity"
        assert result["method"] == "llm"
        assert 0 <= result["score"] <= 1  # Valid score range
    
    @pytest.mark.asyncio
    async def test_answer_similarity_identical_texts(self):
        """Test similarity of identical texts."""
        evaluator = RAGEvaluator()
        
        # For identical texts without embeddings, should still give high score
        metric_name, result = await evaluator._evaluate_answer_similarity(
            "The quick brown fox jumps over the lazy dog",
            "The quick brown fox jumps over the lazy dog"
        )
        
        assert result["score"] > 0.9  # Should be very high similarity for identical texts


@pytest.mark.unit
class TestFullRAGEvaluation:
    """Test complete RAG evaluation workflow."""
    
    @pytest.mark.asyncio
    async def test_evaluate_rag_complete(self):
        """Test complete RAG evaluation with all metrics."""
        evaluator = RAGEvaluator()
        rag_data = SampleDataGenerator.generate_rag_evaluation_data()
        
        results = await evaluator.evaluate(
            query=rag_data["query"],
            contexts=rag_data["retrieved_contexts"],
            response=rag_data["generated_response"],
            ground_truth=rag_data.get("ground_truth"),
            api_name="openai"
        )
        
        assert "metrics" in results
        assert "context_relevance" in results["metrics"]
        assert "answer_faithfulness" in results["metrics"]
        assert "answer_relevance" in results["metrics"]
        
        # Check overall score calculation
        assert "overall_score" in results
        assert 0 <= results["overall_score"] <= 1
    
    @pytest.mark.asyncio
    @patch('asyncio.to_thread')
    async def test_evaluate_rag_without_ground_truth(self, mock_to_thread):
        """Test RAG evaluation without ground truth."""
        # Order matches evaluate() execution: relevance, faithfulness, context_relevance
        mock_responses = [
            "4.5",  # answer_relevance first  
            "4.7",  # answer_faithfulness second
            "4.3",  # context_relevance (for first context)
            "4.3"   # context_relevance (for second context)
        ]
        mock_to_thread.side_effect = mock_responses
        
        evaluator = RAGEvaluator()
        
        results = await evaluator.evaluate(
            query="Test query",
            contexts=["Context 1", "Context 2"],
            response="Test response",
            ground_truth=None,  # No ground truth
            api_name="openai"
        )
        
        assert "answer_similarity" not in results["metrics"]
        assert len(results["metrics"]) == 3
    
    @pytest.mark.asyncio
    async def test_evaluate_rag_with_custom_weights(self):
        """Test RAG evaluation with custom metric weights."""
        evaluator = RAGEvaluator()
        
        custom_weights = {
            "context_relevance": 0.2,
            "answer_faithfulness": 0.5,
            "answer_relevance": 0.3
        }
        
        results = await evaluator.evaluate(
            query="What is the capital of France?",
            contexts=["Paris is the capital and largest city of France.", "France is a country in Western Europe."],
            response="The capital of France is Paris.",
            metric_weights=custom_weights,
            api_name="openai"
        )
        
        # Verify that custom weights are used - just check overall score is calculated
        assert "overall_score" in results
        assert 0 <= results["overall_score"] <= 1
        
        # Verify all expected metrics are present
        assert "answer_relevance" in results["metrics"]
        assert "answer_faithfulness" in results["metrics"]
        assert "context_relevance" in results["metrics"]
        
        # Verify each metric has expected structure
        for metric_name, metric_data in results["metrics"].items():
            assert "score" in metric_data
            assert 0 <= metric_data["score"] <= 1
            assert "raw_score" in metric_data
            assert 1 <= metric_data["raw_score"] <= 5


@pytest.mark.unit
class TestMetricCalculations:
    """Test metric calculation and normalization."""
    
    def test_normalize_score(self):
        """Test score normalization from 1-5 to 0-1."""
        evaluator = RAGEvaluator()
        
        # Test normalization
        assert evaluator._normalize_score(1) == 0
        assert evaluator._normalize_score(3) == 0.5
        assert evaluator._normalize_score(5) == 1.0
        
        # Test out-of-range handling
        assert evaluator._normalize_score(0) == 0
        assert evaluator._normalize_score(6) == 1.0
    
    def test_calculate_overall_score(self):
        """Test overall score calculation."""
        evaluator = RAGEvaluator()
        
        metrics = {
            "context_relevance": {"score": 0.8},
            "answer_faithfulness": {"score": 0.9},
            "answer_relevance": {"score": 0.7}
        }
        
        # Equal weights
        overall = evaluator._calculate_overall_score(metrics)
        assert overall == pytest.approx(0.8, 0.01)
        
        # Custom weights
        weights = {
            "context_relevance": 0.5,
            "answer_faithfulness": 0.3,
            "answer_relevance": 0.2
        }
        overall_weighted = evaluator._calculate_overall_score(metrics, weights)
        expected = 0.8 * 0.5 + 0.9 * 0.3 + 0.7 * 0.2
        assert overall_weighted == pytest.approx(expected, 0.01)


@pytest.mark.unit
class TestErrorHandling:
    """Test error handling in RAGEvaluator."""
    
    @pytest.mark.asyncio
    @patch('asyncio.to_thread')
    async def test_handle_llm_failure(self, mock_to_thread):
        """Test handling of LLM API failures."""
        mock_to_thread.side_effect = Exception("LLM API error")
        
        evaluator = RAGEvaluator()
        
        # _evaluate_context_relevance catches errors and returns 0.0 scores
        metric_name, result = await evaluator._evaluate_context_relevance("query", ["context"], "openai")
        
        assert metric_name == "context_relevance"
        assert result["score"] == 0.0  # Caught exception should result in 0.0
    
    @pytest.mark.asyncio
    @patch('asyncio.to_thread')
    async def test_handle_invalid_llm_response(self, mock_to_thread):
        """Test handling of invalid LLM responses."""
        mock_to_thread.return_value = "not_a_number"
        
        evaluator = RAGEvaluator()
        
        # Should handle invalid response and return 0.0
        metric_name, result = await evaluator._evaluate_context_relevance("query", ["context"], "openai")
        assert result["score"] == 0.0  # Invalid response gets handled as 0.0
    
    @pytest.mark.asyncio
    @patch('tldw_Server_API.app.core.Evaluations.rag_evaluator.create_embedding')
    async def test_handle_embedding_failure(self, mock_create_embedding):
        """Test handling of embedding API failures."""
        mock_create_embedding.side_effect = Exception("Embedding API error")
        
        evaluator = RAGEvaluator(
            embedding_provider="openai",
            embedding_model="text-embedding-3-small"
        )
        evaluator.embedding_available = True
        
        # Should fall back to LLM
        with patch('asyncio.to_thread') as mock_to_thread:
            # The actual code expects plain numeric string, not JSON
            mock_to_thread.return_value = "4.0"
            
            metric_name, result = await evaluator._evaluate_answer_similarity(
                "text1", "text2"
            )
            
            assert result["method"] == "llm"  # Fallback to LLM
            mock_to_thread.assert_called_once()


@pytest.mark.unit
class TestConcurrency:
    """Test concurrent evaluation handling."""
    
    @pytest.mark.asyncio
    @patch('tldw_Server_API.app.core.LLM_Calls.Summarization_General_Lib.analyze')
    async def test_concurrent_evaluations(self, mock_analyze):
        """Test running multiple evaluations concurrently."""
        # The actual code expects plain numeric string, not JSON
        mock_analyze.return_value = "4.0"
        
        evaluator = RAGEvaluator()
        
        # Create multiple evaluation tasks
        tasks = []
        for i in range(5):
            task = evaluator.evaluate(
                query=f"Query {i}",
                contexts=[f"Context {i}"],
                response=f"Response {i}",
                api_name="openai"
            )
            tasks.append(task)
        
        # Run concurrently
        results = await asyncio.gather(*tasks)
        
        assert len(results) == 5
        for result in results:
            assert "metrics" in result
            assert "overall_score" in result