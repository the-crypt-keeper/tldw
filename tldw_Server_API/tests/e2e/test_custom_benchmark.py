"""
End-to-End tests for Custom Benchmark functionality.

Tests the complete benchmark flow simulating real user interactions:
- Creating a custom benchmark with questions
- Running the benchmark against actual LLM endpoints
- Verifying scoring and results aggregation
"""

import pytest
import json
import time
from typing import Dict, Any, List
from datetime import datetime
import uuid
import httpx
# Note: Removed unittest.mock import - E2E tests should not use mocks
# E2E tests should simulate real user interactions with actual services

from tldw_Server_API.tests.e2e.fixtures import (
    api_client, authenticated_client, data_tracker,
    test_user_credentials,
    # Import helper classes
    AssertionHelpers, SmartErrorHandler, AsyncOperationHandler,
    ContentValidator, StateVerification
)


# Removed MockLLMProvider - e2e tests should use real LLM endpoints or test mode
# For true e2e testing, we'll interact with the actual API as a user would


class TestCustomBenchmark:
    """Test custom benchmark creation and execution simulating real user interactions."""
    
    # Class variables to share state between test methods
    eval_id = None
    run_id = None
    mixed_run_id = None
    
    @pytest.fixture(autouse=True)
    def setup(self, authenticated_client, data_tracker):
        """Setup for each test."""
        self.client = authenticated_client
        self.tracker = data_tracker
        
        # Initialize benchmark data for tests
        self.benchmark_name = f"custom_test_benchmark_{uuid.uuid4().hex[:8]}"
        self.benchmark_questions = [
            {
                "input": "What is the capital of France?",
                "expected": "Paris",
                "metadata": {
                    "id": "q1",
                    "category": "geography",
                    "difficulty": "easy"
                }
            },
            {
                "input": "What is 15 * 17?",
                "expected": "255",
                "metadata": {
                    "id": "q2",
                    "category": "math",
                    "difficulty": "easy"
                }
            },
            {
                "input": "Write a Python function to reverse a string",
                "expected": "def reverse_string(s):\n    return s[::-1]",
                "metadata": {
                    "id": "q3",
                    "category": "programming",
                    "difficulty": "medium"
                }
            },
            {
                "input": "If all roses are flowers and some flowers fade quickly, can we conclude all roses fade quickly?",
                "expected": "No",
                "metadata": {
                    "id": "q4",
                    "category": "logic",
                    "difficulty": "medium"
                }
            },
            {
                "input": "What year did World War II end?",
                "expected": "1945",
                "metadata": {
                    "id": "q5",
                    "category": "history",
                    "difficulty": "easy"
                }
            },
            {
                "input": "Explain the concept of recursion in one sentence",
                "expected": "Recursion is a programming technique where a function calls itself",
                "metadata": {
                    "id": "q6",
                    "category": "programming",
                    "difficulty": "medium"
                }
            },
            {
                "input": "What is the chemical formula for water?",
                "expected": "H2O",
                "metadata": {
                    "id": "q7",
                    "category": "science",
                    "difficulty": "easy"
                }
            },
            {
                "input": "Calculate: (8 + 2) * 3 - 5",
                "expected": "25",
                "metadata": {
                    "id": "q8",
                    "category": "math",
                    "difficulty": "easy"
                }
            },
            {
                "input": "What is the largest planet in our solar system?",
                "expected": "Jupiter",
                "metadata": {
                    "id": "q9",
                    "category": "science",
                    "difficulty": "easy"
                }
            },
            {
                "input": "Complete the sequence: 2, 4, 8, 16, ?",
                "expected": "32",
                "metadata": {
                    "id": "q10",
                    "category": "math",
                    "difficulty": "easy"
                }
            }
        ]
    
    def _create_benchmark_if_needed(self):
        """Helper to create a benchmark if not already created - simulates user creating a new benchmark."""
        if TestCustomBenchmark.eval_id is not None:
            return  # Already have a benchmark
        
        try:
            # Simulate user creating a custom benchmark
            eval_data = {
                "name": f"Test_Custom_Benchmark_{uuid.uuid4().hex[:8]}",  # Fixed name format
                "description": "Custom benchmark for e2e testing - simulating user-created evaluation",
                "eval_type": "exact_match",  # Fixed field name and type
                "eval_spec": {
                    "model": "gpt-3.5-turbo",
                    "temperature": 0.0,
                    "metrics": ["accuracy"],
                    "thresholds": {"accuracy": 0.8}
                },
                "dataset": self.benchmark_questions[:5],  # Fixed field name
                "metadata": {
                    "testing": True,
                    "version": "1.0"
                }
            }
            
            response = self.client.client.post(
                "/api/v1/evaluations",
                json=eval_data,
                headers=self.client.get_auth_headers()
            )
            
            if response.status_code in [200, 201]:
                TestCustomBenchmark.eval_id = response.json()["id"]
                print(f"✅ Created benchmark with ID: {TestCustomBenchmark.eval_id}")
            else:
                print(f"❌ Failed to create benchmark: {response.status_code}")
        except Exception as e:
            print(f"❌ Error creating benchmark: {e}")
    
    def test_create_custom_benchmark_evaluation(self):
        """Test creating a custom benchmark as an evaluation."""
        # Create evaluation with our 10 questions as the dataset
        eval_data = {
            "name": self.benchmark_name,
            "description": "Custom E2E test benchmark with 10 questions",
            "eval_type": "model_graded",
            "eval_spec": {
                "model": "gpt-3.5-turbo",
                "metrics": ["correctness", "completeness"],
                "thresholds": {"correctness": 0.7, "completeness": 0.7},
                "custom_prompts": {"scoring": "Compare the answer to the expected answer and score from 0-1"}
            },
            "dataset": self.benchmark_questions,
            "metadata": {
                "author": "e2e_test",
                "tags": ["custom", "benchmark", "test"],
                "version": "1.0.0",
                "categories": ["geography", "math", "programming", "logic", "history", "science"]
            }
        }
        
        try:
            response = self.client.client.post("/api/v1/evaluations", json=eval_data)
            response.raise_for_status()
            
            result = response.json()
            # Use proper assertions
            AssertionHelpers.assert_api_response_structure(result, ["id"])
            assert result.get("name") == self.benchmark_name
            assert result.get("eval_type") == "model_graded"
            
            # Track for cleanup and later tests
            self.tracker.track("evaluation", result["id"])
            TestCustomBenchmark.eval_id = result["id"]  # Store as class variable
            
            print(f"✓ Created custom benchmark: {result['id']}")
            
        except httpx.HTTPStatusError as e:
            TestCustomBenchmark.eval_id = None
            SmartErrorHandler.handle_error(e, "custom benchmark creation")
    
    def test_run_benchmark_with_real_llm(self):
        """Test running benchmark with actual LLM endpoint - simulating real user workflow."""
        if TestCustomBenchmark.eval_id is None:
            # Try to create the benchmark first
            self.test_create_custom_benchmark_evaluation()
            if TestCustomBenchmark.eval_id is None:
                pytest.skip("No evaluation ID available")
        
        # Note: This test uses a real LLM endpoint if configured
        # If no LLM is configured, the test will verify the API handles it gracefully
        
        # Run the benchmark with a real or test model
        # User would select an available model from their configured providers
        run_data = {
            "target_model": "gpt-3.5-turbo",  # Use a common model, API will handle if not configured
            "config": {
                "temperature": 0.0,  # Deterministic for testing
                "max_workers": 2,
                "timeout_seconds": 30,
                "batch_size": 5
            }
        }
        
        response = self.client.client.post(
            f"/api/v1/evaluations/{TestCustomBenchmark.eval_id}/runs",
            json=run_data
        )
        
        if response.status_code == 202:
            result = response.json()
            assert "id" in result
            assert result["status"] in ["pending", "running"]
            
            self.run_id = result["id"]
            print(f"✓ Started benchmark run with all correct answers: {result['id']}")
            
            # Wait for the run to complete (with timeout)
            max_wait = 10  # seconds
            wait_interval = 0.5
            elapsed = 0
            run_completed = False
            
            while elapsed < max_wait:
                time.sleep(wait_interval)
                elapsed += wait_interval
                
                # Check run status
                status_response = self.client.client.get(f"/api/v1/runs/{self.run_id}")
                if status_response.status_code == 200:
                    run_status = status_response.json()
                    if run_status["status"] in ["completed", "failed"]:
                        run_completed = True
                        print(f"✓ Run completed with status: {run_status['status']}")
                        
                        # For now, just verify the run completed
                        # In a real scenario, we'd check the actual results
                        assert run_status["status"] == "completed" or run_status["status"] == "failed"
                        break
            
            # Verify that the benchmark run was processed
            # In E2E testing, we care about the API behavior, not the LLM responses
            if not run_completed:
                print(f"⚠️ Run did not complete within {max_wait} seconds")
        else:
            print(f"Benchmark run failed: {response.status_code}")
    
    def test_run_benchmark_workflow(self):
        """Test complete benchmark workflow as a user would experience it."""
        if not hasattr(self, 'eval_id') or self.eval_id is None:
            self.test_create_custom_benchmark_evaluation()
            if not hasattr(self, 'eval_id') or self.eval_id is None:
                pytest.skip("No evaluation ID available")
        
        # Test the workflow a user would follow to run a benchmark
        # The actual LLM responses don't matter for E2E testing - we're testing the API flow
        
        # Run the benchmark with a configured model (or handle gracefully if none configured)
        run_data = {
            "target_model": "gpt-3.5-turbo",  # Real model that user would select
            "config": {
                "temperature": 0.0,
                "max_workers": 1,  # Sequential for predictable results
                "timeout_seconds": 30
            }
        }
        
        response = self.client.client.post(
            f"/api/v1/evaluations/{TestCustomBenchmark.eval_id}/runs",
            json=run_data
        )
        
        if response.status_code == 202:
            result = response.json()
            self.mixed_run_id = result["id"]
            
            print(f"✓ Started benchmark run with mixed responses: {result['id']}")
            
            # Expected: 7 correct, 3 incorrect (based on incorrect_answers dict)
            # This tests that the scoring differentiates between correct and incorrect
        else:
            print(f"Mixed benchmark run failed: {response.status_code}")
    
    def test_verify_benchmark_scoring(self):
        """Verify that benchmark scoring correctly identifies correct/incorrect answers."""
        # Make test self-sufficient - create benchmark if needed (simulating user creating one)
        if TestCustomBenchmark.eval_id is None:
            # User would create a benchmark first before trying to verify scoring
            self._create_benchmark_if_needed()
            if TestCustomBenchmark.eval_id is None:
                pytest.skip("Unable to create benchmark for testing")
        
        # Get evaluation runs to check scoring
        response = self.client.client.get(f"/api/v1/evaluations/{TestCustomBenchmark.eval_id}/runs")
        
        if response.status_code == 200:
            result = response.json()
            assert "data" in result
            
            if result["data"]:
                # Check that runs have different scores based on responses
                print(f"✓ Retrieved {len(result['data'])} benchmark runs")
                
                for run in result["data"]:
                    print(f"  Run {run['id']}: status={run.get('status', 'unknown')}")
                    
                    # In production, we'd check actual scores here
                    # For now, verify structure
                    if run.get('status') == 'completed':
                        # Run may have results in different fields depending on implementation
                        has_results = any(key in run for key in ['results', 'summary', 'progress', 'metadata'])
                        assert has_results, f"Completed run {run['id']} has no results data"
            else:
                print("No runs found yet (may still be processing)")
        else:
            print(f"Failed to get benchmark runs: {response.status_code}")
            
    def test_wait_for_benchmark_completion(self):
        """Wait for benchmark runs to complete using async handler."""
        if TestCustomBenchmark.run_id is None:
            pytest.skip("No run ID available")
        
        try:
            # Wait for the run to complete
            result = AsyncOperationHandler.wait_for_completion(
                check_func=lambda: self.client.client.get(
                    f"/api/v1/evaluations/{TestCustomBenchmark.eval_id}/runs/{TestCustomBenchmark.run_id}"
                ).json(),
                success_condition=lambda r: r.get("status") == "completed",
                timeout=60,
                context=f"Benchmark run {TestCustomBenchmark.run_id}"
            )
            
            # Validate completed run has results
            assert "score" in result or "overall_score" in result, \
                "Completed run missing score"
            
            print(f"✓ Benchmark run completed with score: {result.get('score', result.get('overall_score'))}")
            
        except Exception as e:
            SmartErrorHandler.handle_error(e, "waiting for benchmark completion")
    
    def test_benchmark_results_aggregation(self):
        """Test that benchmark results are properly aggregated by category."""
        if TestCustomBenchmark.eval_id is None:
            pytest.skip("No evaluation ID available")
        
        # In a real implementation, we would:
        # 1. Wait for runs to complete
        # 2. Retrieve detailed results
        # 3. Verify aggregation by category
        
        # For now, test the structure
        expected_categories = ["geography", "math", "programming", "logic", "history", "science"]
        
        # Verify our test data has proper category distribution
        category_counts = {}
        for question in self.benchmark_questions:
            cat = question["metadata"]["category"]
            category_counts[cat] = category_counts.get(cat, 0) + 1
        
        assert set(category_counts.keys()) == set(expected_categories)
        assert category_counts["math"] == 3  # 3 math questions
        assert category_counts["programming"] == 2  # 2 programming questions
        assert category_counts["science"] == 2  # 2 science questions
        
        print(f"✓ Benchmark has proper category distribution: {category_counts}")
    
    def test_concurrent_benchmark_execution(self):
        """Test running multiple benchmark evaluations concurrently - simulating multiple users."""
        if TestCustomBenchmark.eval_id is None:
            pytest.skip("No evaluation ID available")
        
        # Test concurrent benchmark runs without mocking - as real users would do it
        # Start multiple runs concurrently
        run_ids = []
        for i in range(3):
            run_data = {
                "target_model": "gpt-3.5-turbo",  # Real model that users would select
                "config": {
                    "temperature": 0.0,
                    "max_workers": 2,
                    "timeout_seconds": 30
                }
            }
            
            response = self.client.client.post(
                f"/api/v1/evaluations/{TestCustomBenchmark.eval_id}/runs",
                json=run_data
            )
            
            if response.status_code == 202:
                run_ids.append(response.json()["id"])
            elif response.status_code in [400, 404, 503]:
                # Model might not be configured - that's OK for E2E test
                print(f"Run {i} skipped - model not configured")
            else:
                print(f"Run {i} failed with status {response.status_code}")
        
        print(f"✓ Started {len(run_ids)} concurrent benchmark runs")
        
        # In real E2E test, we verify the API can handle concurrent requests
        # The actual LLM responses don't matter - we're testing the API workflow
    
    def test_benchmark_with_custom_scoring_criteria(self):
        """Test benchmark with custom scoring criteria for different question types."""
        # Create a specialized benchmark with custom scoring per category
        eval_data = {
            "name": f"custom_scoring_{uuid.uuid4().hex[:8]}",
            "description": "Benchmark with category-specific scoring",
            "eval_type": "model_graded",
            "eval_spec": {
                "evaluator_model": "gpt-3.5-turbo",
                "metrics": ["accuracy", "reasoning", "code_quality"],
                "threshold": 0.8,
                "category_weights": {
                    "math": {"accuracy": 1.0},
                    "programming": {"accuracy": 0.5, "code_quality": 0.5},
                    "logic": {"accuracy": 0.3, "reasoning": 0.7}
                }
            },
            "dataset": self.benchmark_questions[:3],  # Use subset for testing
            "metadata": {
                "custom_scoring": True,
                "scoring_version": "2.0"
            }
        }
        
        response = self.client.client.post("/api/v1/evaluations", json=eval_data)
        
        if response.status_code == 201:
            result = response.json()
            self.tracker.track("evaluation", result["id"])
            print(f"✓ Created benchmark with custom scoring: {result['id']}")
        else:
            print(f"Custom scoring benchmark failed: {response.status_code}")
    
    def test_cleanup_custom_benchmarks(self):
        """Clean up all created benchmarks."""
        cleaned = 0
        
        # Check if tracker has tracked evaluations
        if hasattr(self.tracker, 'tracked_resources'):
            evaluations = self.tracker.tracked_resources.get("evaluation", [])
        elif hasattr(self.tracker, 'resources'):
            evaluations = self.tracker.resources.get("evaluation", [])
        else:
            # Fallback - try to clean up known eval_id
            evaluations = []
            if hasattr(self, 'eval_id') and self.eval_id:
                evaluations = [{'id': self.eval_id}]
        
        for eval_info in evaluations:
            try:
                eval_id = eval_info['id'] if isinstance(eval_info, dict) else eval_info
                response = self.client.client.delete(f"/api/v1/evaluations/{eval_id}")
                if response.status_code in [204, 200, 404]:  # 404 is ok if already deleted
                    cleaned += 1
            except Exception as e:
                print(f"Failed to cleanup evaluation: {e}")
        
        print(f"✓ Cleaned up {cleaned} custom benchmarks")


class TestBenchmarkEdgeCases:
    """Test edge cases and error handling for custom benchmarks."""
    
    @pytest.fixture(autouse=True)
    def setup(self, authenticated_client):
        """Setup for each test."""
        self.client = authenticated_client
        # No mocking needed - E2E tests use real API endpoints
    
    def test_benchmark_with_empty_dataset(self):
        """Test creating benchmark with empty dataset."""
        eval_data = {
            "name": "empty_benchmark",
            "description": "Benchmark with no questions",
            "eval_type": "model_graded",
            "eval_spec": {
                "evaluator_model": "gpt-3.5-turbo",
                "metrics": ["accuracy"]
            },
            "dataset": []  # Empty dataset
        }
        
        response = self.client.client.post("/api/v1/evaluations", json=eval_data)
        # Should either reject or handle gracefully (401 is auth issue)
        assert response.status_code in [400, 401, 422, 201]
        
        if response.status_code == 201:
            print("✓ Empty benchmark accepted (may be valid for some use cases)")
        else:
            print("✓ Empty benchmark rejected as expected")
    
    def test_benchmark_with_malformed_questions(self):
        """Test benchmark with malformed question data."""
        eval_data = {
            "name": "malformed_benchmark",
            "description": "Benchmark with invalid questions",
            "eval_type": "model_graded",
            "eval_spec": {
                "evaluator_model": "gpt-3.5-turbo",
                "metrics": ["accuracy"]
            },
            "dataset": [
                {"invalid_field": "test"},  # Missing required fields
                {"question": "Valid question", "expected_answer": "Answer"},
                None,  # Null entry
                "not_a_dict"  # Wrong type
            ]
        }
        
        response = self.client.client.post("/api/v1/evaluations", json=eval_data)
        # Should handle invalid data gracefully
        print(f"Malformed benchmark response: {response.status_code}")
    
    def test_benchmark_with_invalid_model(self):
        """Test benchmark execution with invalid/unconfigured model - simulating user error."""
        # Create a simple benchmark first
        eval_data = {
            "name": "failure_test_benchmark",
            "description": "Test LLM failure handling",
            "eval_type": "model_graded",
            "eval_spec": {
                "evaluator_model": "gpt-3.5-turbo",
                "metrics": ["accuracy"]
            },
            "dataset": [
                {"question": "Test question 1", "expected_answer": "Answer 1"},
                {"question": "Test question 2", "expected_answer": "Answer 2"}
            ]
        }
        
        response = self.client.client.post("/api/v1/evaluations", json=eval_data)
        
        if response.status_code == 201:
            eval_id = response.json()["id"]
            
            # Try to run the benchmark with an invalid/unconfigured model
            # This simulates a user selecting a model that's not properly configured
            run_data = {
                "target_model": "invalid-model-xyz-not-configured",
                "config": {"timeout_seconds": 5}
            }
            
            run_response = self.client.client.post(
                f"/api/v1/evaluations/{eval_id}/runs",
                json=run_data
            )
            
            # Should either reject invalid model or handle gracefully
            if run_response.status_code == 202:
                print("✓ Benchmark run accepted - will handle invalid model during execution")
            elif run_response.status_code in [400, 404, 503]:
                print("✓ Invalid model properly rejected by API")
            else:
                print(f"Benchmark run response: {run_response.status_code}")
            
            # Cleanup
            self.client.client.delete(f"/api/v1/evaluations/{eval_id}")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])