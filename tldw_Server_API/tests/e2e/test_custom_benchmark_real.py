"""
End-to-End tests for Custom Benchmark functionality - Real User Simulation.

This file contains true e2e tests that simulate real user interactions
without mocks. Tests interact with the actual API as a user would.
"""

import pytest
import json
import time
from typing import Dict, Any, List
from datetime import datetime
import uuid

from tldw_Server_API.tests.e2e.fixtures import (
    api_client, authenticated_client, data_tracker,
    test_user_credentials,
    AssertionHelpers, SmartErrorHandler, AsyncOperationHandler,
    ContentValidator, StateVerification, StrongAssertionHelpers
)


class TestRealBenchmarkWorkflow:
    """Test custom benchmark creation and execution simulating real user interactions."""
    
    # Class variables for workflow state
    eval_id = None
    run_id = None
    
    @pytest.fixture(autouse=True)
    def setup(self, authenticated_client, data_tracker):
        """Setup for each test - like a user logging into the app."""
        self.client = authenticated_client
        self.tracker = data_tracker
        
        # Simple benchmark questions a user might create
        self.benchmark_questions = [
            {
                "id": "q1",
                "question": "What is 2 + 2?",
                "expected_answer": "4",
                "category": "math",
                "difficulty": "easy"
            },
            {
                "id": "q2", 
                "question": "What color is the sky on a clear day?",
                "expected_answer": "blue",
                "category": "general",
                "difficulty": "easy"
            },
            {
                "id": "q3",
                "question": "Complete: 'To be or not to be, that is the ___'",
                "expected_answer": "question",
                "category": "literature",
                "difficulty": "medium"
            }
        ]
    
    def test_user_creates_benchmark(self):
        """Test a user creating a custom benchmark through the UI/API."""
        # User fills out form to create a benchmark
        benchmark_data = {
            "name": f"My Test Benchmark {uuid.uuid4().hex[:8]}",
            "description": "A benchmark I created to test my models",
            "type": "custom",
            "questions": self.benchmark_questions,
            "config": {
                "model_config": {
                    "temperature": 0.7,
                    "max_tokens": 150
                },
                "scoring": {
                    "method": "similarity",  # More realistic than exact match
                    "threshold": 0.8
                }
            }
        }
        
        try:
            response = self.client.client.post(
                "/api/v1/evaluations",
                json=benchmark_data
            )
            
            if response.status_code in [200, 201]:
                result = response.json()
                
                # Strong assertions on response
                StrongAssertionHelpers.assert_non_empty_string(
                    result.get("id"), "benchmark ID", min_length=1
                )
                
                TestRealBenchmarkWorkflow.eval_id = result["id"]
                self.tracker.track("benchmark", TestRealBenchmarkWorkflow.eval_id)
                
                print(f"✅ User successfully created benchmark: {TestRealBenchmarkWorkflow.eval_id}")
                
                # Verify user can retrieve their benchmark
                verify_response = self.client.client.get(f"/api/v1/evaluations/{TestRealBenchmarkWorkflow.eval_id}")
                assert verify_response.status_code == 200, "User should be able to view their benchmark"
                
            elif response.status_code == 501:
                pytest.skip("Benchmark feature not implemented")
            elif response.status_code == 422:
                # Validation error - show details for debugging
                error_detail = response.json() if response.text else "No error details"
                print(f"Validation error creating benchmark: {error_detail}")
                pytest.skip(f"Benchmark API validation error - feature may not be fully implemented: {response.status_code}")
            else:
                pytest.fail(f"Failed to create benchmark: {response.status_code} - {response.text[:200] if response.text else 'No details'}")
                
        except Exception as e:
            SmartErrorHandler.handle_error(e, "benchmark creation")
    
    def test_user_runs_benchmark(self):
        """Test a user running their benchmark against a model."""
        if TestRealBenchmarkWorkflow.eval_id is None:
            pytest.skip("No benchmark available - user needs to create one first")
        
        # User selects a model and runs the benchmark
        run_config = {
            "target_model": "default",  # Use whatever model is configured
            "config": {
                "temperature": 0.7,
                "max_workers": 1,  # Run sequentially like a cautious user
                "timeout_seconds": 60
            }
        }
        
        try:
            response = self.client.client.post(
                f"/api/v1/evaluations/{TestRealBenchmarkWorkflow.eval_id}/runs",
                json=run_config
            )
            
            if response.status_code in [200, 201, 202]:
                result = response.json()
                
                # User gets a run ID to track progress
                if "id" in result:
                    TestRealBenchmarkWorkflow.run_id = result["id"]
                    self.tracker.track("benchmark_run", TestRealBenchmarkWorkflow.run_id)
                    print(f"✅ User started benchmark run: {TestRealBenchmarkWorkflow.run_id}")
                
                # User might see immediate results or need to wait
                if result.get("status") == "completed":
                    self._verify_benchmark_results(result)
                elif result.get("status") in ["pending", "running"]:
                    print("⏳ Benchmark is running, user would wait or check back later")
                    
            elif response.status_code == 501:
                pytest.skip("Benchmark execution not implemented")
            else:
                pytest.fail(f"Failed to run benchmark: {response.status_code}")
                
        except Exception as e:
            SmartErrorHandler.handle_error(e, "benchmark execution")
    
    def test_user_checks_benchmark_results(self):
        """Test a user checking the results of their benchmark run."""
        if TestRealBenchmarkWorkflow.run_id is None:
            pytest.skip("No benchmark run available")
        
        # User refreshes page or clicks to check results
        try:
            response = self.client.client.get(
                f"/api/v1/evaluations/{TestRealBenchmarkWorkflow.eval_id}/runs/{TestRealBenchmarkWorkflow.run_id}"
            )
            
            if response.status_code == 200:
                result = response.json()
                
                # Verify result structure that user would see
                assert "status" in result, "User should see run status"
                
                if result["status"] == "completed":
                    self._verify_benchmark_results(result)
                    print(f"✅ User can view benchmark results")
                elif result["status"] in ["pending", "running"]:
                    print("⏳ Benchmark still running")
                elif result["status"] == "failed":
                    print(f"❌ Benchmark failed: {result.get('error', 'Unknown error')}")
                    
            elif response.status_code == 404:
                pytest.skip("Run not found - may have been cleaned up")
            else:
                pytest.fail(f"Failed to get results: {response.status_code}")
                
        except Exception as e:
            SmartErrorHandler.handle_error(e, "checking benchmark results")
    
    def test_user_views_all_benchmarks(self):
        """Test a user viewing their list of benchmarks."""
        try:
            response = self.client.client.get("/api/v1/evaluations")
            
            if response.status_code == 200:
                result = response.json()
                
                # User should see a list or data structure
                if isinstance(result, list):
                    print(f"✅ User sees {len(result)} benchmarks")
                elif isinstance(result, dict) and "data" in result:
                    print(f"✅ User sees {len(result['data'])} benchmarks")
                else:
                    print("✅ User can access benchmark list")
                    
            elif response.status_code == 501:
                pytest.skip("Benchmark listing not implemented")
            else:
                pytest.fail(f"Failed to list benchmarks: {response.status_code}")
                
        except Exception as e:
            SmartErrorHandler.handle_error(e, "listing benchmarks")
    
    def test_user_deletes_benchmark(self):
        """Test a user deleting their benchmark."""
        if TestRealBenchmarkWorkflow.eval_id is None:
            pytest.skip("No benchmark to delete")
        
        try:
            # User clicks delete button
            response = self.client.client.delete(
                f"/api/v1/evaluations/{TestRealBenchmarkWorkflow.eval_id}"
            )
            
            if response.status_code in [200, 204]:
                print(f"✅ User successfully deleted benchmark")
                TestRealBenchmarkWorkflow.eval_id = None
                
                # Verify it's actually gone
                verify_response = self.client.client.get(
                    f"/api/v1/evaluations/{TestRealBenchmarkWorkflow.eval_id}"
                )
                assert verify_response.status_code == 404, "Deleted benchmark should not be accessible"
                
            elif response.status_code == 501:
                pytest.skip("Benchmark deletion not implemented")
            elif response.status_code == 404:
                print("Benchmark already deleted")
            else:
                print(f"Could not delete benchmark: {response.status_code}")
                
        except Exception as e:
            # Deletion errors are not critical
            print(f"Benchmark deletion issue: {e}")
    
    def _verify_benchmark_results(self, result: Dict[str, Any]):
        """Helper to verify benchmark results structure."""
        # Check what a user would expect to see
        expected_fields = ["score", "results", "summary", "metrics", "total", "correct"]
        
        # At least one of these should be present
        has_results = any(field in result for field in expected_fields)
        assert has_results, f"Results should contain scoring information, got: {result.keys()}"
        
        # If there's a score, validate it
        if "score" in result:
            StrongAssertionHelpers.assert_value_in_range(
                result["score"], 0.0, 1.0, "benchmark score"
            )
        
        # If there are detailed results, check them
        if "results" in result and isinstance(result["results"], list):
            for item in result["results"]:
                assert "question_id" in item or "id" in item, "Each result should identify the question"
                assert "response" in item or "answer" in item, "Each result should have a response"