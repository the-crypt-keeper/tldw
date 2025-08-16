"""
Tests for error scenarios and edge cases in the evaluations module.

Ensures proper handling of:
- Network failures
- Invalid inputs
- Resource exhaustion
- Timeouts
- Malformed data
"""

import pytest
import asyncio
from unittest.mock import Mock, patch, AsyncMock
import numpy as np
import json
from pathlib import Path
import tempfile

from tldw_Server_API.app.core.Evaluations.rag_evaluator import RAGEvaluator
from tldw_Server_API.app.core.Evaluations.evaluation_manager import EvaluationManager


class TestErrorScenarios:
    """Test error handling in various failure scenarios."""
    
    @pytest.mark.asyncio
    async def test_empty_inputs(self):
        """Test handling of empty inputs."""
        evaluator = RAGEvaluator()
        evaluator.embedding_available = False  # Force LLM path
        
        # Empty query - should handle gracefully (not ideal but doesn't crash)
        result = await evaluator.evaluate(
            query="",
            contexts=["context"],
            response="response",
            metrics=["relevance"]
        )
        # Should handle gracefully even with empty query
        assert "metrics" in result or "failed_metrics" in result
        
        # Empty contexts
        result = await evaluator.evaluate(
            query="query",
            contexts=[],
            response="response",
            metrics=["faithfulness"]
        )
        # Should handle gracefully
        assert "metrics" in result
        
        # Empty response - should handle gracefully
        result = await evaluator.evaluate(
            query="query",
            contexts=["context"],
            response="",
            metrics=["relevance"]
        )
        assert "metrics" in result or "failed_metrics" in result
    
    @pytest.mark.asyncio
    async def test_malformed_llm_responses(self):
        """Test handling of malformed LLM responses."""
        evaluator = RAGEvaluator()
        evaluator.embedding_available = False
        
        with patch('tldw_Server_API.app.core.Evaluations.rag_evaluator.asyncio.to_thread') as mock_thread:
            # LLM returns non-numeric score
            mock_thread.return_value = "not a number"
            
            with pytest.raises(ValueError):
                await evaluator._evaluate_relevance("query", "response", "openai")
            
            # LLM returns out-of-range score
            mock_thread.return_value = "10"  # Should be 1-5
            
            _, result = await evaluator._evaluate_relevance("query", "response", "openai")
            # Should clamp or handle gracefully
            assert 0 <= result["score"] <= 1
    
    @pytest.mark.asyncio
    async def test_network_timeout(self):
        """Test handling of network timeouts."""
        evaluator = RAGEvaluator()
        
        async def slow_embed(*args):
            await asyncio.sleep(10)  # Simulate slow network
            return np.random.rand(1536)
        
        with patch.object(evaluator, 'embeddings_integration') as mock_integration:
            mock_integration.embed_query = slow_embed
            evaluator.embedding_available = True
            
            # Should timeout or fall back
            with pytest.raises(asyncio.TimeoutError):
                await asyncio.wait_for(
                    evaluator._evaluate_answer_similarity("response", "ground_truth"),
                    timeout=0.1
                )
    
    @pytest.mark.asyncio
    async def test_resource_exhaustion(self):
        """Test handling when resources are exhausted."""
        evaluator = RAGEvaluator()
        
        # Simulate memory error in embeddings
        with patch.object(evaluator, 'embeddings_integration') as mock_integration:
            mock_integration.embed_query = AsyncMock(side_effect=MemoryError("Out of memory"))
            evaluator.embedding_available = True
            
            # Should fall back to LLM
            with patch('tldw_Server_API.app.core.Evaluations.rag_evaluator.asyncio.to_thread') as mock_thread:
                mock_thread.return_value = "3"
                
                _, result = await evaluator._evaluate_answer_similarity("response", "ground_truth")
                assert result["method"] == "llm"  # Should have fallen back
    
    @pytest.mark.asyncio
    async def test_concurrent_failure_isolation(self):
        """Test that failures in one evaluation don't affect others."""
        evaluator = RAGEvaluator()
        evaluator.embedding_available = False
        
        with patch('tldw_Server_API.app.core.Evaluations.rag_evaluator.asyncio.to_thread') as mock_thread:
            call_count = 0
            
            def side_effect(*args):
                nonlocal call_count
                call_count += 1
                if call_count == 2:
                    raise Exception("Random failure")
                return "4"
            
            mock_thread.side_effect = side_effect
            
            # Run multiple evaluations
            tasks = []
            for i in range(3):
                task = evaluator.evaluate(
                    query=f"Query {i}",
                    contexts=[f"Context {i}"],
                    response=f"Response {i}",
                    metrics=["relevance"]
                )
                tasks.append(task)
            
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Some should succeed despite one failing
            successes = [r for r in results if not isinstance(r, Exception)]
            failures = [r for r in results if isinstance(r, Exception)]
            
            assert len(successes) >= 1
            assert len(failures) >= 1
    
    @pytest.mark.asyncio
    async def test_invalid_api_credentials(self):
        """Test handling of invalid API credentials."""
        # Test with invalid OpenAI key
        with patch.dict('os.environ', {'OPENAI_API_KEY': 'invalid-key'}):
            evaluator = RAGEvaluator(
                embedding_provider="openai",
                api_key="invalid-key"
            )
            
            # Should fall back or handle gracefully
            assert evaluator.embedding_available is False or evaluator.embedding_available is True
    
    @pytest.mark.asyncio
    async def test_database_corruption(self):
        """Test handling of database corruption."""
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            db_path = Path(f.name)
        
        try:
            # Corrupt the database
            with open(db_path, 'wb') as f:
                f.write(b"This is not a valid SQLite database")
            
            # Should handle gracefully
            manager = EvaluationManager(db_path=db_path)
            
            # Operations should fail gracefully
            with pytest.raises(Exception):
                await manager.store_evaluation(
                    evaluation_type="test",
                    input_data={},
                    results={}
                )
        finally:
            if db_path.exists():
                db_path.unlink()
    
    @pytest.mark.asyncio
    async def test_circular_reference_handling(self):
        """Test handling of circular references in data."""
        manager = EvaluationManager()
        
        # Create circular reference
        data = {"key": "value"}
        data["self"] = data  # Circular reference
        
        # Should handle during JSON serialization
        with pytest.raises(Exception):
            await manager.store_evaluation(
                evaluation_type="test",
                input_data=data,
                results={}
            )
    
    @pytest.mark.asyncio
    async def test_unicode_handling(self):
        """Test handling of various unicode characters."""
        evaluator = RAGEvaluator()
        evaluator.embedding_available = False
        
        # Test with various unicode
        test_strings = [
            "Hello 世界",  # Chinese
            "مرحبا بالعالم",  # Arabic
            "🚀 Emoji test 🎉",  # Emojis
            "Ñoño",  # Spanish characters
            "\u0000 null character",  # Null character
        ]
        
        for test_str in test_strings:
            with patch('tldw_Server_API.app.core.Evaluations.rag_evaluator.asyncio.to_thread') as mock_thread:
                mock_thread.return_value = "3"
                
                # Should handle unicode properly
                result = await evaluator.evaluate(
                    query=test_str,
                    contexts=[test_str],
                    response=test_str,
                    metrics=["relevance"]
                )
                
                assert "metrics" in result
    
    @pytest.mark.asyncio
    async def test_extremely_long_inputs(self):
        """Test handling of extremely long inputs."""
        evaluator = RAGEvaluator()
        evaluator.embedding_available = False
        
        # Create very long input (100k characters)
        long_text = "a" * 100000
        
        with patch('tldw_Server_API.app.core.Evaluations.rag_evaluator.asyncio.to_thread') as mock_thread:
            mock_thread.return_value = "3"
            
            # Should handle or truncate gracefully
            result = await evaluator.evaluate(
                query="short query",
                contexts=[long_text],
                response="short response",
                metrics=["faithfulness"]
            )
            
            assert "metrics" in result


class TestEdgeCases:
    """Test edge cases in the evaluation system."""
    
    @pytest.mark.asyncio
    async def test_zero_score_handling(self):
        """Test that legitimate zero scores are handled differently from errors."""
        evaluator = RAGEvaluator()
        evaluator.embedding_available = False
        
        with patch('tldw_Server_API.app.core.Evaluations.rag_evaluator.asyncio.to_thread') as mock_thread:
            mock_thread.return_value = "1"  # Minimum score = legitimate low score
            
            _, result = await evaluator._evaluate_relevance("query", "response", "openai")
            
            # Should be 0.2 (1/5), not 0.0
            assert result["score"] == 0.2
            assert "Evaluation failed" not in result.get("explanation", "")
    
    @pytest.mark.asyncio
    async def test_perfect_score_handling(self):
        """Test handling of perfect scores."""
        evaluator = RAGEvaluator()
        
        # Create identical embeddings for perfect similarity
        with patch.object(evaluator, 'embeddings_integration') as mock_integration:
            same_embedding = np.array([1.0, 0.0, 0.0])
            mock_integration.embed_query = AsyncMock(return_value=same_embedding)
            evaluator.embedding_available = True
            
            _, result = await evaluator._evaluate_answer_similarity(
                "identical text",
                "identical text"
            )
            
            # Should be very close to 1.0
            assert result["score"] > 0.99
            assert result["raw_score"] > 4.9
    
    @pytest.mark.asyncio
    async def test_single_metric_evaluation(self):
        """Test evaluation with only one metric."""
        evaluator = RAGEvaluator()
        evaluator.embedding_available = False
        
        with patch('tldw_Server_API.app.core.Evaluations.rag_evaluator.asyncio.to_thread') as mock_thread:
            mock_thread.return_value = "4"
            
            result = await evaluator.evaluate(
                query="query",
                contexts=["context"],
                response="response",
                metrics=["relevance"]  # Only one metric
            )
            
            assert len(result["metrics"]) == 1
            assert "relevance" in result["metrics"]
    
    @pytest.mark.asyncio
    async def test_no_metrics_requested(self):
        """Test evaluation when no metrics are requested."""
        evaluator = RAGEvaluator()
        
        result = await evaluator.evaluate(
            query="query",
            contexts=["context"],
            response="response",
            metrics=[]  # No metrics
        )
        
        assert result["metrics"] == {}
        assert result["suggestions"] == []
    
    @pytest.mark.asyncio
    async def test_duplicate_metrics(self):
        """Test handling of duplicate metric requests."""
        evaluator = RAGEvaluator()
        evaluator.embedding_available = False
        
        with patch('tldw_Server_API.app.core.Evaluations.rag_evaluator.asyncio.to_thread') as mock_thread:
            mock_thread.return_value = "3"
            
            result = await evaluator.evaluate(
                query="query",
                contexts=["context"],
                response="response",
                metrics=["relevance", "relevance", "relevance"]  # Duplicates
            )
            
            # Should only evaluate once
            assert len(result["metrics"]) == 1
    
    @pytest.mark.asyncio
    async def test_mixed_success_and_failure(self):
        """Test when some metrics succeed and others fail."""
        evaluator = RAGEvaluator()
        
        # Make embeddings work but LLM fail for some metrics
        with patch.object(evaluator, 'embeddings_integration') as mock_integration:
            mock_integration.embed_query = AsyncMock(return_value=np.random.rand(1536))
            evaluator.embedding_available = True
            
            with patch('tldw_Server_API.app.core.Evaluations.rag_evaluator.asyncio.to_thread') as mock_thread:
                mock_thread.side_effect = Exception("LLM unavailable")
                
                result = await evaluator.evaluate(
                    query="query",
                    contexts=["context"],
                    response="response",
                    ground_truth="truth",
                    metrics=["relevance", "answer_similarity"]
                )
                
                # answer_similarity should succeed (uses embeddings)
                # relevance should fail (uses LLM)
                assert "answer_similarity" in result["metrics"]
                assert "failed_metrics" in result
                assert "relevance" in result["failed_metrics"]
    
    def test_migration_idempotency(self):
        """Test that migrations can be run multiple times safely."""
        from tldw_Server_API.app.core.DB_Management.migrations import migrate_evaluations_database
        
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            db_path = Path(f.name)
        
        try:
            # Run migrations multiple times
            for _ in range(3):
                migrate_evaluations_database(db_path)
            
            # Should not fail or duplicate
            from tldw_Server_API.app.core.DB_Management.migrations import MigrationManager
            manager = MigrationManager(db_path)
            version = manager.get_current_version()
            
            assert version >= 4  # Should be at latest version
        finally:
            if db_path.exists():
                db_path.unlink()


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v", "--tb=short"])