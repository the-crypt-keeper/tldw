"""
Comprehensive integration tests for the evaluations module.

Tests the complete evaluation pipeline including:
- Embeddings integration
- Error handling
- Database operations
- Authentication
- Rate limiting
"""

import pytest
import asyncio
import tempfile
import os
from pathlib import Path
from unittest.mock import Mock, patch, AsyncMock, MagicMock
import numpy as np
import sqlite3
import json
from datetime import datetime

from tldw_Server_API.app.core.Evaluations.rag_evaluator import RAGEvaluator
from tldw_Server_API.app.core.Evaluations.evaluation_manager import EvaluationManager
from tldw_Server_API.app.core.DB_Management.migrations import migrate_evaluations_database, MigrationManager


class TestEvaluationIntegration:
    """Integration tests for the complete evaluation pipeline."""
    
    @pytest.fixture
    def temp_db_path(self):
        """Create a temporary database for testing."""
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            db_path = Path(f.name)
        yield db_path
        # Cleanup
        if db_path.exists():
            db_path.unlink()
    
    @pytest.fixture
    def mock_embeddings_integration(self):
        """Create a mock embeddings integration."""
        mock = Mock()
        mock.embed_query = AsyncMock(return_value=np.random.rand(1536))
        mock.embed_documents = AsyncMock(return_value=np.random.rand(3, 1536))
        mock.close = Mock()
        return mock
    
    @pytest.fixture
    def evaluation_manager(self, temp_db_path):
        """Create an evaluation manager with temporary database."""
        # Patch the db_path after initialization since EvaluationManager doesn't take db_path arg
        manager = EvaluationManager()
        manager.db_path = temp_db_path
        manager._init_database()  # Re-initialize with new path
        return manager
    
    @pytest.mark.asyncio
    async def test_full_evaluation_pipeline(self, evaluation_manager, mock_embeddings_integration):
        """Test the complete evaluation pipeline from input to storage."""
        with patch('tldw_Server_API.app.core.Evaluations.rag_evaluator.create_rag_embeddings_integration') as mock_create:
            mock_create.return_value = mock_embeddings_integration
            
            # Create evaluator
            evaluator = RAGEvaluator(embedding_provider="openai")
            
            # Run evaluation
            results = await evaluator.evaluate(
                query="What is machine learning?",
                contexts=["ML is a subset of AI", "It learns from data"],
                response="Machine learning is AI that learns from data",
                ground_truth="ML is artificial intelligence that learns patterns from data",
                metrics=["answer_similarity"]
            )
            
            # Verify results structure
            assert "metrics" in results
            assert "answer_similarity" in results["metrics"]
            assert results["metrics"]["answer_similarity"]["method"] == "embeddings"
            
            # Store evaluation
            eval_id = await evaluation_manager.store_evaluation(
                evaluation_type="rag_evaluation",
                input_data={"query": "What is machine learning?"},
                results=results
            )
            
            assert eval_id is not None
            
            # Retrieve and verify
            history = await evaluation_manager.get_history(
                evaluation_type="rag_evaluation",
                limit=1
            )
            
            # get_history returns a dict with 'items' key
            assert "items" in history
            assert len(history["items"]) == 1
            assert history["items"][0]["evaluation_id"] == eval_id
            
            # Cleanup
            evaluator.close()
    
    @pytest.mark.asyncio
    async def test_error_propagation(self, mock_embeddings_integration):
        """Test that errors are properly propagated instead of returning 0.0."""
        with patch('tldw_Server_API.app.core.Evaluations.rag_evaluator.create_rag_embeddings_integration') as mock_create:
            # Make embeddings fail
            mock_embeddings_integration.embed_query.side_effect = Exception("Service unavailable")
            mock_create.return_value = mock_embeddings_integration
            
            evaluator = RAGEvaluator()
            
            # Mock LLM to also fail
            with patch('tldw_Server_API.app.core.Evaluations.rag_evaluator.asyncio.to_thread') as mock_thread:
                mock_thread.side_effect = Exception("LLM error")
                
                # Should raise error instead of returning 0.0
                with pytest.raises(ValueError) as exc_info:
                    await evaluator._evaluate_answer_similarity(
                        "response", "ground_truth"
                    )
                
                assert "Answer similarity evaluation failed" in str(exc_info.value)
    
    @pytest.mark.asyncio
    async def test_partial_failure_handling(self, mock_embeddings_integration):
        """Test that partial failures are handled gracefully."""
        with patch('tldw_Server_API.app.core.Evaluations.rag_evaluator.create_rag_embeddings_integration') as mock_create:
            mock_create.return_value = mock_embeddings_integration
            
            evaluator = RAGEvaluator()
            
            # Mock one metric to fail
            with patch.object(evaluator, '_evaluate_relevance') as mock_relevance:
                with patch.object(evaluator, '_evaluate_faithfulness') as mock_faithfulness:
                    mock_relevance.side_effect = ValueError("Relevance failed")
                    mock_faithfulness.return_value = ("faithfulness", {
                        "score": 0.8,
                        "name": "faithfulness"
                    })
                    
                    results = await evaluator.evaluate(
                        query="test",
                        contexts=["context"],
                        response="response",
                        metrics=["relevance", "faithfulness"]
                    )
                    
                    # Should have partial results
                    assert "metrics" in results
                    assert "faithfulness" in results["metrics"]
                    assert "failed_metrics" in results
                    assert "relevance" in results["failed_metrics"]
                    assert results.get("partial_results") is True
    
    def test_database_migration(self, temp_db_path):
        """Test that database migrations work correctly."""
        # Apply migrations
        migrate_evaluations_database(temp_db_path)
        
        # Check that tables exist with correct schema
        with sqlite3.connect(temp_db_path) as conn:
            cursor = conn.cursor()
            
            # Check evaluations table
            cursor.execute("PRAGMA table_info(evaluations)")
            columns = {col[1] for col in cursor.fetchall()}
            
            # Should have all columns from migrations
            expected_columns = {
                'id', 'evaluation_id', 'evaluation_type', 'created_at',
                'input_data', 'results', 'metadata', 'user_id',
                'status', 'error_message', 'completed_at',
                'embedding_provider', 'embedding_model'
            }
            
            assert expected_columns.issubset(columns)
            
            # Check migration tracking
            cursor.execute("SELECT version FROM schema_migrations ORDER BY version")
            versions = [row[0] for row in cursor.fetchall()]
            assert len(versions) >= 4  # Should have at least 4 migrations
    
    @pytest.mark.asyncio
    async def test_embedding_fallback(self):
        """Test that evaluation falls back to LLM when embeddings unavailable."""
        # Create evaluator without embeddings
        with patch('tldw_Server_API.app.core.Evaluations.rag_evaluator.create_rag_embeddings_integration') as mock_create:
            mock_create.side_effect = Exception("No API key")
            
            evaluator = RAGEvaluator()
            assert evaluator.embedding_available is False
            
            # Mock LLM response
            with patch('tldw_Server_API.app.core.Evaluations.rag_evaluator.asyncio.to_thread') as mock_thread:
                mock_thread.return_value = "4"
                
                _, result = await evaluator._evaluate_answer_similarity(
                    "response", "ground_truth"
                )
                
                assert result["method"] == "llm"
                assert result["score"] == 0.8  # 4/5
    
    @pytest.mark.asyncio
    async def test_concurrent_evaluations(self, evaluation_manager, mock_embeddings_integration):
        """Test that multiple evaluations can run concurrently."""
        with patch('tldw_Server_API.app.core.Evaluations.rag_evaluator.create_rag_embeddings_integration') as mock_create:
            mock_create.return_value = mock_embeddings_integration
            
            evaluator = RAGEvaluator()
            
            # Create multiple evaluation tasks
            tasks = []
            for i in range(5):
                task = evaluator.evaluate(
                    query=f"Query {i}",
                    contexts=[f"Context {i}"],
                    response=f"Response {i}",
                    metrics=["answer_similarity"]
                )
                tasks.append(task)
            
            # Run concurrently
            results = await asyncio.gather(*tasks)
            
            # All should complete
            assert len(results) == 5
            for result in results:
                assert "metrics" in result
            
            evaluator.close()
    
    @pytest.mark.asyncio
    async def test_evaluation_history_tracking(self, evaluation_manager):
        """Test that evaluation history is properly tracked."""
        # Store multiple evaluations
        eval_ids = []
        for i in range(3):
            eval_id = await evaluation_manager.store_evaluation(
                evaluation_type="test_eval",
                input_data={"index": i},
                results={"score": i * 0.1},
                metadata={"test": True}
            )
            eval_ids.append(eval_id)
            await asyncio.sleep(0.01)  # Small delay to ensure different timestamps
        
        # Get history
        history = await evaluation_manager.get_history(
            evaluation_type="test_eval",
            limit=10
        )
        
        # Should have all evaluations
        assert "items" in history
        assert len(history["items"]) == 3
        
        # Should be in reverse chronological order
        retrieved_ids = [h["evaluation_id"] for h in history["items"]]
        assert retrieved_ids == list(reversed(eval_ids))
    
    @pytest.mark.asyncio
    async def test_evaluation_comparison(self, evaluation_manager):
        """Test comparing multiple evaluations."""
        # Store evaluations with metrics
        eval_ids = []
        for i in range(2):
            results = {
                "metrics": {
                    "relevance": {"score": 0.5 + i * 0.2},
                    "faithfulness": {"score": 0.6 + i * 0.1}
                }
            }
            eval_id = await evaluation_manager.store_evaluation(
                evaluation_type="comparison_test",
                input_data={"version": i},
                results=results
            )
            eval_ids.append(eval_id)
        
        # Compare evaluations
        comparison = await evaluation_manager.compare_evaluations(eval_ids)
        
        assert "evaluations" in comparison
        assert len(comparison["evaluations"]) == 2
        assert "summary" in comparison
        assert "relevance" in comparison["summary"]


class TestAuthentication:
    """Test authentication improvements."""
    
    @pytest.fixture
    def mock_settings(self):
        """Create mock settings."""
        settings = Mock()
        settings.AUTH_MODE = "single_user"
        settings.RATE_LIMIT_PER_MINUTE = 60
        settings.RATE_LIMIT_BURST = 10
        return settings
    
    @pytest.mark.asyncio
    async def test_single_user_auth(self, mock_settings):
        """Test single-user authentication without hardcoded keys."""
        from tldw_Server_API.app.api.v1.endpoints.evals_openai import verify_api_key
        from fastapi import HTTPException
        from fastapi.security import HTTPAuthorizationCredentials
        
        with patch('tldw_Server_API.app.api.v1.endpoints.evals_openai.get_settings') as mock_get_settings:
            mock_get_settings.return_value = mock_settings
            
            # Test with valid API key
            with patch.dict(os.environ, {'API_BEARER': 'test-api-key'}):
                creds = HTTPAuthorizationCredentials(scheme="Bearer", credentials="test-api-key")
                result = await verify_api_key(creds)
                assert result == "test-api-key"
            
            # Test with invalid API key
            with patch.dict(os.environ, {'API_BEARER': 'correct-key'}):
                creds = HTTPAuthorizationCredentials(scheme="Bearer", credentials="wrong-key")
                with pytest.raises(HTTPException) as exc_info:
                    await verify_api_key(creds)
                assert exc_info.value.status_code == 401
    
    @pytest.mark.asyncio
    async def test_multi_user_jwt_auth(self, mock_settings):
        """Test multi-user JWT authentication."""
        from tldw_Server_API.app.api.v1.endpoints.evals_openai import verify_api_key
        from fastapi.security import HTTPAuthorizationCredentials
        
        mock_settings.AUTH_MODE = "multi_user"
        
        with patch('tldw_Server_API.app.api.v1.endpoints.evals_openai.get_settings') as mock_get_settings:
            mock_get_settings.return_value = mock_settings
            
            with patch('tldw_Server_API.app.api.v1.endpoints.evals_openai.JWTService') as MockJWTService:
                mock_jwt = Mock()
                mock_jwt.verify_access_token.return_value = {
                    'sub': '123',
                    'username': 'testuser',
                    'role': 'user'
                }
                MockJWTService.return_value = mock_jwt
                
                creds = HTTPAuthorizationCredentials(scheme="Bearer", credentials="valid-jwt-token")
                result = await verify_api_key(creds)
                assert result == "user_123"


class TestRateLimiting:
    """Test rate limiting configuration."""
    
    def test_rate_limit_configuration(self):
        """Test that rate limits are properly configured from settings."""
        from tldw_Server_API.app.api.v1.endpoints.evals_openai import (
            rate_limit_per_minute, burst_limit
        )
        
        # Should have sensible defaults even without settings
        assert rate_limit_per_minute > 0
        assert burst_limit > 0
    
    def test_rate_limit_key_function(self):
        """Test rate limit key generation."""
        from tldw_Server_API.app.api.v1.endpoints.evals_openai import get_rate_limit_key
        from fastapi import Request
        
        # Test with user ID
        request = Mock(spec=Request)
        request.state = Mock()
        request.state.user_id = "123"
        request.client = Mock()
        request.client.host = "127.0.0.1"
        
        key = get_rate_limit_key(request)
        assert key == "user_123"
        
        # Test without user ID (falls back to IP)
        request = Mock(spec=Request)
        request.state = Mock(spec=[])  # No user_id attribute
        request.client = Mock()
        request.client.host = "192.168.1.1"
        
        key = get_rate_limit_key(request)
        assert "192.168.1.1" in key


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v", "--tb=short"])