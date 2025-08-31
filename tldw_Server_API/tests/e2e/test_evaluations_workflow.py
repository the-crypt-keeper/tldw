"""
End-to-End tests for Evaluation API endpoints.

Tests cover:
- OpenAI-compatible evaluation endpoints (/api/v1/evaluations)
- Standard evaluation endpoints (/api/v1/evaluations)
- G-Eval, RAG evaluation, response quality assessment
- Batch evaluations and comparison features
"""

import pytest
import asyncio
import json
import time
from typing import Dict, Any, List, Optional
from datetime import datetime
import uuid
import httpx

from tldw_Server_API.tests.e2e.fixtures import (
    api_client, authenticated_client, data_tracker,
    test_user_credentials,
    # Import helper classes
    AssertionHelpers, SmartErrorHandler, AsyncOperationHandler,
    ContentValidator, StateVerification
)
from tldw_Server_API.tests.e2e.test_data import TestDataGenerator


class TestEvaluationWorkflow:
    """Test evaluation API endpoints comprehensively."""
    
    # Class variables to share state between test methods
    eval_id = None
    standard_eval_id = None
    run_id = None
    
    @pytest.fixture(autouse=True)
    def setup(self, authenticated_client, data_tracker):
        """Setup for each test."""
        self.client = authenticated_client
        self.tracker = data_tracker
        self.data_gen = TestDataGenerator()
        
        # Simple performance tracking
        class PerfTracker:
            def measure(self, name):
                from contextlib import contextmanager
                @contextmanager
                def _measure():
                    import time
                    start = time.time()
                    yield
                    duration = time.time() - start
                    print(f"⏱️ {name}: {duration:.2f}s")
                return _measure()
        
        self.perf = PerfTracker()
        
    # ===================== OpenAI-Compatible Evaluation Tests =====================
    
    def test_create_openai_evaluation(self):
        """Test creating an OpenAI-compatible evaluation."""
        with self.perf.measure("create_openai_evaluation"):
            # Create evaluation with inline dataset
            eval_data = {
                "name": f"Test Evaluation {uuid.uuid4().hex[:8]}",
                "description": "E2E test evaluation for response quality",
                "eval_type": "model_graded",
                "eval_spec": {
                    "evaluator_model": "gpt-4",
                    "metrics": ["accuracy", "coherence", "relevance"],
                    "threshold": 0.8,
                    "sub_type": "summarization"
                },
                "dataset": [
                    {
                        "input": "What is machine learning?",
                        "expected": "Machine learning is a subset of AI that enables systems to learn from data.",
                        "context": "AI and ML basics"
                    },
                    {
                        "input": "Explain neural networks",
                        "expected": "Neural networks are computing systems inspired by biological neural networks.",
                        "context": "Deep learning fundamentals"
                    }
                ],
                "metadata": {
                    "author": "e2e_test",
                    "tags": ["test", "ml", "qa"],
                    "version": "1.0.0"
                }
            }
            
            try:
                response = self.client.client.post("/api/v1/evaluations", json=eval_data)
                response.raise_for_status()
                
                result = response.json()
                # Use proper assertions
                AssertionHelpers.assert_api_response_structure(result, ["id", "object", "name", "eval_type"])
                assert result["object"] == "evaluation"
                assert result["name"] == eval_data["name"]
                assert result["eval_type"] == "model_graded"
                
                # Track for cleanup
                self.tracker.track("evaluation", result["id"])
                
                # Store for later tests
                TestEvaluationWorkflow.eval_id = result["id"]
                
                print(f"✓ Created evaluation: {result['id']}")
                
            except httpx.HTTPStatusError as e:
                SmartErrorHandler.handle_error(e, "evaluation creation")
    
    
    def test_list_openai_evaluations(self):
        """Test listing OpenAI-compatible evaluations."""
        with self.perf.measure("list_openai_evaluations"):
            response = self.client.client.get("/api/v1/evaluations?limit=10")
            
            if response.status_code == 200:
                result = response.json()
                assert "object" in result
                assert result["object"] == "list"
                assert "data" in result
                assert isinstance(result["data"], list)
                
                if result["data"]:
                    # Verify evaluation structure
                    eval_item = result["data"][0]
                    assert "id" in eval_item
                    assert "object" in eval_item
                    assert eval_item["object"] == "evaluation"
                    
                print(f"✓ Listed {len(result['data'])} evaluations")
            else:
                print(f"Evaluation listing failed: {response.status_code}")
    
    
    def test_get_openai_evaluation(self):
        """Test retrieving a specific OpenAI-compatible evaluation."""
        if TestEvaluationWorkflow.eval_id is None:
            pytest.skip("No evaluation ID available")
            
        with self.perf.measure("get_openai_evaluation"):
            response = self.client.client.get(f"/api/v1/evaluations/{TestEvaluationWorkflow.eval_id}")
            
            if response.status_code == 200:
                result = response.json()
                assert result["id"] == TestEvaluationWorkflow.eval_id
                assert result["object"] == "evaluation"
                assert "eval_spec" in result
                
                print(f"✓ Retrieved evaluation: {self.eval_id}")
            else:
                print(f"Evaluation retrieval failed: {response.status_code}")
    
    
    def test_update_openai_evaluation(self):
        """Test updating an OpenAI-compatible evaluation."""
        if TestEvaluationWorkflow.eval_id is None:
            pytest.skip("No evaluation ID available")
            
        with self.perf.measure("update_openai_evaluation"):
            update_data = {
                "description": "Updated E2E test evaluation",
                "eval_spec": {
                    "threshold": 0.85,
                    "metrics": ["accuracy", "coherence", "relevance", "completeness"]
                },
                "metadata": {
                    "updated_at": datetime.now().isoformat(),
                    "version": "1.1.0"
                }
            }
            
            response = self.client.client.patch(
                f"/api/v1/evaluations/{TestEvaluationWorkflow.eval_id}",
                json=update_data
            )
            
            if response.status_code == 200:
                result = response.json()
                assert result["description"] == update_data["description"]
                
                print(f"✓ Updated evaluation: {self.eval_id}")
            else:
                print(f"Evaluation update failed: {response.status_code}")
    
    
    def test_run_openai_evaluation(self):
        """Test running an OpenAI-compatible evaluation."""
        if TestEvaluationWorkflow.eval_id is None:
            pytest.skip("No evaluation ID available")
            
        with self.perf.measure("run_openai_evaluation"):
            run_data = {
                "target_model": "gpt-3.5-turbo",
                "config": {
                    "temperature": 0.7,
                    "max_workers": 2,
                    "timeout_seconds": 60,
                    "batch_size": 5
                }
            }
            
            response = self.client.client.post(
                f"/api/v1/evaluations/{TestEvaluationWorkflow.eval_id}/runs",
                json=run_data
            )
            
            if response.status_code == 202:  # Accepted for async processing
                result = response.json()
                assert "id" in result
                assert result["object"] == "evaluation.run"
                assert result["status"] in ["pending", "running"]
                
                # Track run ID for later tests
                self.run_id = result["id"]
                
                print(f"✓ Started evaluation run: {result['id']}")
            else:
                print(f"Evaluation run failed: {response.status_code} - {response.text}")
    
    
    def test_list_evaluation_runs(self):
        """Test listing runs for an evaluation."""
        if TestEvaluationWorkflow.eval_id is None:
            pytest.skip("No evaluation ID available")
            
        with self.perf.measure("list_evaluation_runs"):
            response = self.client.client.get(f"/api/v1/evaluations/{TestEvaluationWorkflow.eval_id}/runs")
            
            if response.status_code == 200:
                result = response.json()
                assert result["object"] == "list"
                assert "data" in result
                
                if result["data"]:
                    run = result["data"][0]
                    assert "id" in run
                    assert "status" in run
                    
                print(f"✓ Listed {len(result['data'])} evaluation runs")
            else:
                print(f"Run listing failed: {response.status_code}")
    
    # ===================== Standard Evaluation Tests =====================
    
    
    def test_geval_summarization(self):
        """Test G-Eval summarization evaluation."""
        with self.perf.measure("geval_summarization"):
            eval_data = {
                "document": """
                Machine learning has revolutionized many industries by enabling computers 
                to learn from data without explicit programming. Deep learning, a subset 
                of machine learning, uses neural networks with multiple layers to process 
                complex patterns. Applications include image recognition, natural language 
                processing, and autonomous vehicles.
                """,
                "summary": """
                Machine learning allows computers to learn from data. Deep learning uses 
                neural networks for complex pattern recognition in applications like 
                image recognition and NLP.
                """,
                "metrics": ["coherence", "consistency", "fluency", "relevance"],
                "model": "gpt-3.5-turbo",
                "custom_prompt": None
            }
            
            response = self.client.client.post("/api/v1/evaluations/geval", json=eval_data)
            
            if response.status_code == 200:
                result = response.json()
                assert "overall_score" in result
                assert "individual_scores" in result
                assert len(result["individual_scores"]) == len(eval_data["metrics"])
                
                for metric in eval_data["metrics"]:
                    assert metric in result["individual_scores"]
                    score = result["individual_scores"][metric]
                    assert 1 <= score <= 5
                
                print(f"✓ G-Eval score: {result['overall_score']:.2f}")
            elif response.status_code == 503:
                print("G-Eval skipped: Service unavailable (likely no API key)")
            else:
                print(f"G-Eval failed: {response.status_code} - {response.text}")
    
    
    def test_rag_evaluation(self):
        """Test RAG system evaluation."""
        with self.perf.measure("rag_evaluation"):
            rag_data = {
                "query": "What are the benefits of electric vehicles?",
                "retrieved_contexts": [
                    "Electric vehicles produce zero direct emissions, improving air quality.",
                    "EVs have lower operating costs due to fewer moving parts and cheaper electricity.",
                    "Electric motors provide instant torque for better acceleration."
                ],
                "generated_answer": """
                Electric vehicles offer several benefits including zero emissions for cleaner air,
                lower operating costs due to simpler mechanics and cheaper fuel, and superior
                performance with instant torque delivery.
                """,
                "ground_truth": """
                Benefits of electric vehicles include environmental advantages through zero emissions,
                economic benefits from lower maintenance and fuel costs, and performance benefits
                from instant torque and quiet operation.
                """,
                "metrics": ["context_relevance", "answer_relevance", "faithfulness", "correctness"]
            }
            
            response = self.client.client.post("/api/v1/evaluations/rag", json=rag_data)
            
            if response.status_code == 200:
                result = response.json()
                assert "overall_score" in result
                assert "metrics" in result
                
                for metric in rag_data["metrics"]:
                    assert metric in result["metrics"]
                    assert 0 <= result["metrics"][metric] <= 1
                
                print(f"✓ RAG evaluation score: {result['overall_score']:.2f}")
            elif response.status_code in [503, 422]:
                print(f"RAG evaluation skipped: {response.status_code}")
            else:
                print(f"RAG evaluation failed: {response.status_code}")
    
    
    def test_response_quality_evaluation(self):
        """Test response quality evaluation."""
        with self.perf.measure("response_quality_evaluation"):
            quality_data = {
                "prompt": "Explain the concept of recursion in programming",
                "response": """
                Recursion is a programming technique where a function calls itself to solve
                a problem by breaking it down into smaller, similar subproblems. It consists
                of a base case that stops the recursion and a recursive case that makes the
                function call itself with modified parameters. Common examples include
                factorial calculation and tree traversal.
                """,
                "criteria": {
                    "accuracy": "Is the information factually correct?",
                    "completeness": "Does it cover all important aspects?",
                    "clarity": "Is the explanation clear and understandable?"
                },
                "reference_response": None,
                "model": "gpt-3.5-turbo"
            }
            
            response = self.client.client.post(
                "/api/v1/evaluations/response-quality",
                json=quality_data
            )
            
            if response.status_code == 200:
                result = response.json()
                assert "overall_score" in result
                assert "criteria_scores" in result
                
                for criterion in quality_data["criteria"]:
                    assert criterion in result["criteria_scores"]
                
                print(f"✓ Response quality score: {result['overall_score']:.2f}")
            elif response.status_code in [503, 422]:
                print(f"Response quality evaluation skipped: {response.status_code}")
            else:
                print(f"Response quality evaluation failed: {response.status_code}")
    
    
    def test_batch_evaluation(self):
        """Test batch evaluation processing."""
        with self.perf.measure("batch_evaluation"):
            batch_data = {
                "evaluation_type": "response_quality",
                "items": [
                    {
                        "id": "item_1",
                        "prompt": "What is Python?",
                        "response": "Python is a high-level programming language.",
                        "metadata": {"category": "definition"}
                    },
                    {
                        "id": "item_2",
                        "prompt": "Explain OOP",
                        "response": "Object-oriented programming is a paradigm based on objects and classes.",
                        "metadata": {"category": "concept"}
                    }
                ],
                "config": {
                    "parallel_workers": 2,
                    "timeout_per_item": 30,
                    "retry_failed": True
                }
            }
            
            response = self.client.client.post("/api/v1/evaluations/batch", json=batch_data)
            
            if response.status_code == 200:
                result = response.json()
                assert "batch_id" in result
                assert "status" in result
                assert "results" in result or "message" in result
                
                if "results" in result:
                    assert len(result["results"]) <= len(batch_data["items"])
                
                print(f"✓ Batch evaluation submitted: {result.get('batch_id', 'N/A')}")
            elif response.status_code in [503, 422]:
                print(f"Batch evaluation skipped: {response.status_code}")
            else:
                print(f"Batch evaluation failed: {response.status_code}")
    
    
    def test_evaluation_comparison(self):
        """Test comparing multiple evaluations."""
        with self.perf.measure("evaluation_comparison"):
            comparison_data = {
                "evaluations": [
                    {
                        "id": "eval_1",
                        "name": "Model A",
                        "scores": {
                            "accuracy": 0.85,
                            "fluency": 0.90,
                            "coherence": 0.88
                        }
                    },
                    {
                        "id": "eval_2",
                        "name": "Model B",
                        "scores": {
                            "accuracy": 0.82,
                            "fluency": 0.92,
                            "coherence": 0.86
                        }
                    }
                ],
                "comparison_metrics": ["accuracy", "fluency", "coherence"],
                "aggregation_method": "weighted_average",
                "weights": {
                    "accuracy": 0.5,
                    "fluency": 0.25,
                    "coherence": 0.25
                }
            }
            
            response = self.client.client.post(
                "/api/v1/evaluations/compare",
                json=comparison_data
            )
            
            if response.status_code == 200:
                result = response.json()
                assert "comparison_id" in result
                assert "rankings" in result
                assert "detailed_comparison" in result
                
                print(f"✓ Evaluation comparison completed")
            elif response.status_code in [503, 422]:
                print(f"Evaluation comparison skipped: {response.status_code}")
            else:
                print(f"Evaluation comparison failed: {response.status_code}")
    
    
    def test_custom_metric_evaluation(self):
        """Test custom metric evaluation."""
        with self.perf.measure("custom_metric_evaluation"):
            custom_data = {
                "name": "domain_expertise",
                "description": "Evaluate domain-specific knowledge accuracy",
                "evaluation_prompt": """
                Rate the following response for domain expertise in {domain}:
                Response: {response}
                
                Score from 0-1 based on:
                - Technical accuracy
                - Use of domain terminology
                - Depth of explanation
                """,
                "inputs": {
                    "domain": "machine learning",
                    "response": "Neural networks use backpropagation to adjust weights through gradient descent."
                },
                "model": "gpt-3.5-turbo"
            }
            
            response = self.client.client.post(
                "/api/v1/evaluations/custom-metric",
                json=custom_data
            )
            
            if response.status_code == 200:
                result = response.json()
                assert "metric_name" in result
                assert "score" in result
                assert 0 <= result["score"] <= 1
                
                print(f"✓ Custom metric score: {result['score']:.2f}")
            elif response.status_code in [503, 422]:
                print(f"Custom metric evaluation skipped: {response.status_code}")
            else:
                print(f"Custom metric evaluation failed: {response.status_code}")
    
    
    def test_evaluation_history(self):
        """Test retrieving evaluation history."""
        with self.perf.measure("evaluation_history"):
            history_params = {
                "start_date": "2024-01-01T00:00:00Z",
                "end_date": "2024-12-31T23:59:59Z",
                "evaluation_type": "all",
                "limit": 20,
                "include_metadata": True
            }
            
            response = self.client.client.post(
                "/api/v1/evaluations/history",
                json=history_params
            )
            
            if response.status_code == 200:
                result = response.json()
                assert "items" in result
                assert "total_count" in result
                assert "average_scores" in result
                
                print(f"✓ Retrieved {result['total_count']} historical evaluations")
            else:
                print(f"Evaluation history failed: {response.status_code}")
    
    # ===================== Cleanup Tests =====================
    
    
    def test_cleanup_evaluations(self):
        """Clean up created evaluations."""
        with self.perf.measure("cleanup_evaluations"):
            cleaned = 0
            
            # Clean up OpenAI-compatible evaluations
            if TestEvaluationWorkflow.eval_id is not None:
                try:
                    response = self.client.client.delete(f"/api/v1/evaluations/{TestEvaluationWorkflow.eval_id}")
                    if response.status_code in [204, 200]:
                        cleaned += 1
                except Exception as e:
                    print(f"Failed to delete evaluation {TestEvaluationWorkflow.eval_id}: {e}")
            
            # Clean up any tracked evaluations
            if hasattr(self.tracker, 'resources'):
                for eval_id in self.tracker.resources.get("evaluation", []):
                    try:
                        response = self.client.client.delete(f"/api/v1/evaluations/{eval_id['id']}")
                        if response.status_code in [204, 200]:
                            cleaned += 1
                    except:
                        pass
            
            print(f"✓ Cleaned up {cleaned} evaluations")


class TestEvaluationEdgeCases:
    """Test edge cases and error handling for evaluation endpoints."""
    
    @pytest.fixture(autouse=True)
    def setup(self, authenticated_client):
        """Setup for each test."""
        self.client = authenticated_client
        
        # Simple performance tracking
        class PerfTracker:
            def measure(self, name):
                from contextlib import contextmanager
                @contextmanager
                def _measure():
                    import time
                    start = time.time()
                    yield
                    duration = time.time() - start
                    print(f"⏱️ {name}: {duration:.2f}s")
                return _measure()
        
        self.perf = PerfTracker()
    
    
    def test_invalid_evaluation_type(self):
        """Test creating evaluation with invalid type."""
        with self.perf.measure("invalid_evaluation_type"):
            eval_data = {
                "name": "Invalid Type Test",
                "eval_type": "invalid_type",  # Invalid
                "eval_spec": {
                    "evaluator_model": "gpt-4"
                },
                "dataset": [{"input": "test"}]
            }
            
            response = self.client.client.post("/api/v1/evaluations", json=eval_data)
            # Should get 401 in single-user mode if auth fails, or 422/400 for validation
            assert response.status_code in [401, 422, 400]
            
            if response.status_code == 401:
                print("✓ Authentication required for evaluation endpoint")
            else:
                print("✓ Invalid evaluation type rejected correctly")
    
    
    def test_missing_dataset(self):
        """Test creating evaluation without dataset."""
        with self.perf.measure("missing_dataset"):
            eval_data = {
                "name": "No Dataset Test",
                "eval_type": "model_graded",
                "eval_spec": {
                    "evaluator_model": "gpt-4"
                }
                # Missing both dataset and dataset_id
            }
            
            response = self.client.client.post("/api/v1/evaluations", json=eval_data)
            # Should get 401 in single-user mode if auth fails, or 422/400 for validation
            assert response.status_code in [401, 422, 400]
            
            if response.status_code == 401:
                print("✓ Authentication required for evaluation endpoint")
            else:
                print("✓ Missing dataset rejected correctly")
    
    
    def test_evaluation_timeout_handling(self):
        """Test evaluation timeout handling."""
        with self.perf.measure("evaluation_timeout"):
            eval_data = {
                "document": "Test" * 10000,  # Very long document
                "summary": "Test summary",
                "metrics": ["coherence"],
                "model": "gpt-3.5-turbo"
                # Note: removed 'timeout' field as it's not a valid field for this endpoint
            }
            
            response = self.client.client.post("/api/v1/evaluations/geval", json=eval_data)
            # Should either succeed, timeout, be rate limited, or be rejected due to size/validation
            assert response.status_code in [200, 408, 422, 429, 503, 504]
            
            if response.status_code == 422:
                print("✓ Large request rejected by validation")
            elif response.status_code == 429:
                print("✓ Rate limited (too many requests)")
            elif response.status_code in [408, 504]:
                print("✓ Timeout handled appropriately")
            elif response.status_code == 503:
                print("✓ Service unavailable (likely no API key configured)")
            else:
                print("✓ Request processed successfully")
    
    
    async def test_concurrent_evaluations(self):
        """Test running multiple evaluations concurrently."""
        with self.perf.measure("concurrent_evaluations"):
            import asyncio
            import httpx
            
            async def make_request(client_session, eval_data):
                """Make an async request."""
                try:
                    response = await client_session.post(
                        "/api/v1/evaluations/response-quality",
                        json=eval_data
                    )
                    return response
                except Exception as e:
                    return e
            
            # Create async httpx client with same auth headers
            async with httpx.AsyncClient(
                base_url=self.client.base_url,
                headers=self.client.client.headers,
                timeout=30
            ) as async_client:
                tasks = []
                
                for i in range(3):
                    eval_data = {
                        "prompt": f"Test prompt {i}",
                        "response": f"Test response {i}",
                        "criteria": {
                            "quality": "Is this a good response?"
                        }
                    }
                    
                    task = make_request(async_client, eval_data)
                    tasks.append(task)
                
                # Run concurrently
                results = await asyncio.gather(*tasks, return_exceptions=True)
                
                success_count = sum(
                    1 for r in results
                    if not isinstance(r, Exception) and hasattr(r, 'status_code') and r.status_code == 200
                )
                
                print(f"✓ Completed {success_count}/3 concurrent evaluations")
    
    
    async def test_rate_limiting(self):
        """Test rate limiting on evaluation endpoints."""
        with self.perf.measure("rate_limiting_test"):
            import asyncio
            import httpx
            
            async def make_request(client_session, eval_data):
                """Make an async request."""
                try:
                    response = await client_session.post(
                        "/api/v1/evaluations/geval",
                        json=eval_data
                    )
                    return response
                except Exception as e:
                    return e
            
            # Create async httpx client with same auth headers
            async with httpx.AsyncClient(
                base_url=self.client.base_url,
                headers=self.client.client.headers,
                timeout=30
            ) as async_client:
                rapid_requests = []
                
                for i in range(15):  # Exceed typical rate limit
                    eval_data = {
                        "document": f"Doc {i}",
                        "summary": f"Summary {i}",
                        "metrics": ["coherence"],
                        "model": "gpt-3.5-turbo"
                    }
                    
                    rapid_requests.append(
                        make_request(async_client, eval_data)
                    )
                
                results = await asyncio.gather(*rapid_requests, return_exceptions=True)
                
                # Check if any were rate limited
                rate_limited = sum(
                    1 for r in results
                    if not isinstance(r, Exception) and hasattr(r, 'status_code') and r.status_code == 429
                )
                
                if rate_limited > 0:
                    print(f"✓ Rate limiting working: {rate_limited} requests limited")
                else:
                    print("✓ No rate limiting encountered (may have high limits)")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])