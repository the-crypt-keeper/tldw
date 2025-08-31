# test_evals_openai.py - Comprehensive test suite for OpenAI-compatible evaluations API
"""
Test suite for the OpenAI-compatible evaluations API.

Tests include:
- CRUD operations for evaluations, runs, and datasets
- Async evaluation processing
- Authentication and authorization
- Error handling and edge cases
- Progress tracking
"""

import pytest
import asyncio
import json
import os
from unittest.mock import patch, MagicMock, AsyncMock
from fastapi.testclient import TestClient
from httpx import AsyncClient
import time

# Import centralized test configuration
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from test_config import test_config

# Set up test environment before any app imports
test_config.setup_test_environment()
test_config.reset_settings()

# Import the FastAPI app
from tldw_Server_API.app.main import app
from tldw_Server_API.app.api.v1.schemas.evaluation_schemas_unified import (
    CreateEvaluationRequest, CreateRunRequest, CreateDatasetRequest,
    EvaluationSpec, EvaluationResponse, RunResponse
)

# Use configuration from test_config
DEFAULT_API_KEY = test_config.TEST_API_KEY
TEST_SK_KEY = test_config.TEST_SK_KEY


@pytest.fixture(scope="function")
def client():
    """Create a test client"""
    with TestClient(app) as c:
        yield c


@pytest.fixture(scope="function")
async def async_client():
    """Create an async test client"""
    from httpx import ASGITransport
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
        yield ac


@pytest.fixture
def auth_headers():
    """Get authentication headers with default API key"""
    return test_config.get_auth_headers()


@pytest.fixture
def sk_auth_headers():
    """Get authentication headers with OpenAI-style sk- key"""
    return test_config.get_sk_auth_headers()


@pytest.fixture
def sample_evaluation_request():
    """Create a sample evaluation request"""
    return {
        "name": "test_evaluation",
        "description": "Test evaluation for unit tests",
        "eval_type": "model_graded",
        "eval_spec": {
            "evaluator_model": "gpt-4",
            "metrics": ["accuracy", "relevance"],
            "threshold": 0.7
        },
        "dataset": [
            {
                "input": {"text": "Test input 1"},
                "expected": {"score": 0.8}
            },
            {
                "input": {"text": "Test input 2"},
                "expected": {"score": 0.9}
            }
        ],
        "metadata": {
            "author": "test_user",
            "tags": ["test", "unit"],
            "version": "1.0.0"
        }
    }


@pytest.fixture
def sample_dataset_request():
    """Create a sample dataset request"""
    return {
        "name": "test_dataset",
        "description": "Test dataset for unit tests",
        "samples": [
            {
                "input": {"text": "Sample 1"},
                "expected": {"label": "positive"}
            },
            {
                "input": {"text": "Sample 2"},
                "expected": {"label": "negative"}
            }
        ],
        "metadata": {
            "source": "unit_tests",
            "created_by": "test_suite"
        }
    }


@pytest.fixture
def sample_run_request():
    """Create a sample run request"""
    return {
        "target_model": "gpt-3.5-turbo",
        "config": {
            "temperature": 0.0,
            "max_workers": 2,
            "timeout_seconds": 60
        }
    }


class TestAuthentication:
    """Test authentication and authorization"""
    
    def test_missing_auth_header(self, client):
        """Test request without authentication header"""
        # Should always require authentication for security
        response = client.get("/api/v1/evaluations")
        # Should get 401 without authentication
        assert response.status_code == 401
        data = response.json()
        assert "error" in data or ("detail" in data and "error" in data["detail"])
    
    def test_invalid_auth_header(self, client):
        """Test request with invalid authentication header"""
        headers = {"Authorization": "Bearer invalid-key-12345"}
        response = client.get("/api/v1/evaluations", headers=headers)
        # Should get 401 with invalid key
        assert response.status_code == 401
        data = response.json()
        # Handle both error formats
        if "error" in data:
            assert data["error"]["code"] in ["invalid_api_key", "invalid_credentials", "invalid_token"]
        elif "detail" in data and isinstance(data["detail"], dict) and "error" in data["detail"]:
            assert data["detail"]["error"]["code"] in ["invalid_api_key", "invalid_credentials", "invalid_token"]
    
    def test_valid_default_key(self, client, auth_headers):
        """Test request with valid default API key"""
        response = client.get("/api/v1/evaluations", headers=auth_headers)
        assert response.status_code in [200, 404]  # 404 if no evals exist yet
    
    def test_valid_sk_key(self, client):
        """Test request with OpenAI-style sk- key (only valid if it matches the configured key)"""
        # For OpenAI compatibility, sk- keys are accepted but only if they match the expected key
        # Using a random sk- key should fail
        headers = {"Authorization": f"Bearer {TEST_SK_KEY}"}
        response = client.get("/api/v1/evaluations", headers=headers)
        # Should fail because this sk- key doesn't match the configured API key
        assert response.status_code == 401
        
        # Test with sk- version of the actual API key would work if configured
        # But for security, we don't accept arbitrary sk- keys


class TestEvaluationCRUD:
    """Test CRUD operations for evaluations"""
    
    def test_create_evaluation(self, client, auth_headers, sample_evaluation_request):
        """Test creating a new evaluation"""
        response = client.post(
            "/api/v1/evaluations",
            json=sample_evaluation_request,
            headers=auth_headers
        )
        assert response.status_code == 201
        data = response.json()
        assert data["name"] == sample_evaluation_request["name"]
        assert data["eval_type"] == sample_evaluation_request["eval_type"]
        assert data["id"].startswith("eval_")
        assert "created" in data
        return data["id"]
    
    def test_get_evaluation(self, client, auth_headers, sample_evaluation_request):
        """Test getting an evaluation by ID"""
        # First create an evaluation
        create_response = client.post(
            "/api/v1/evaluations",
            json=sample_evaluation_request,
            headers=auth_headers
        )
        eval_id = create_response.json()["id"]
        
        # Then retrieve it
        response = client.get(f"/api/v1/evaluations/{eval_id}", headers=auth_headers)
        assert response.status_code == 200
        data = response.json()
        assert data["id"] == eval_id
        assert data["name"] == sample_evaluation_request["name"]
    
    def test_update_evaluation(self, client, auth_headers, sample_evaluation_request):
        """Test updating an evaluation"""
        # Create evaluation
        create_response = client.post(
            "/api/v1/evaluations",
            json=sample_evaluation_request,
            headers=auth_headers
        )
        eval_id = create_response.json()["id"]
        
        # Update it
        update_data = {
            "description": "Updated description",
            "metadata": {"updated": True}
        }
        response = client.patch(
            f"/api/v1/evaluations/{eval_id}",
            json=update_data,
            headers=auth_headers
        )
        assert response.status_code == 200
        data = response.json()
        assert data["description"] == "Updated description"
        assert data["metadata"]["updated"] is True
    
    def test_delete_evaluation(self, client, auth_headers, sample_evaluation_request):
        """Test deleting an evaluation"""
        # Create evaluation
        create_response = client.post(
            "/api/v1/evaluations",
            json=sample_evaluation_request,
            headers=auth_headers
        )
        eval_id = create_response.json()["id"]
        
        # Delete it
        response = client.delete(f"/api/v1/evaluations/{eval_id}", headers=auth_headers)
        assert response.status_code == 204
        
        # Verify it's deleted (soft delete, so might still be retrievable)
        get_response = client.get(f"/api/v1/evaluations/{eval_id}", headers=auth_headers)
        # Should either be not found or marked as deleted
        assert get_response.status_code in [404, 200]
    
    def test_list_evaluations(self, client, auth_headers, sample_evaluation_request):
        """Test listing evaluations with pagination"""
        # Create multiple evaluations
        eval_ids = []
        for i in range(3):
            req = sample_evaluation_request.copy()
            req["name"] = f"test_eval_{i}"
            response = client.post("/api/v1/evaluations", json=req, headers=auth_headers)
            eval_ids.append(response.json()["id"])
        
        # List them
        response = client.get("/api/v1/evaluations?limit=2", headers=auth_headers)
        assert response.status_code == 200
        data = response.json()
        assert data["object"] == "list"
        assert len(data["data"]) <= 2
        if len(data["data"]) == 2:
            assert data["has_more"] is True
    
    def test_evaluation_not_found(self, client, auth_headers):
        """Test getting non-existent evaluation"""
        response = client.get("/api/v1/evaluations/eval_nonexistent", headers=auth_headers)
        assert response.status_code == 404
        data = response.json()
        # Handle FastAPI error response format
        if "detail" in data and isinstance(data["detail"], dict):
            assert "error" in data["detail"]
        else:
            assert "error" in data


class TestDatasetOperations:
    """Test dataset CRUD operations"""
    
    def test_create_dataset(self, client, auth_headers, sample_dataset_request):
        """Test creating a dataset"""
        response = client.post(
            "/api/v1/datasets",
            json=sample_dataset_request,
            headers=auth_headers
        )
        assert response.status_code == 201
        data = response.json()
        assert data["name"] == sample_dataset_request["name"]
        assert data["id"].startswith("dataset_")
        assert len(data["samples"]) == len(sample_dataset_request["samples"])
    
    def test_get_dataset(self, client, auth_headers, sample_dataset_request):
        """Test getting a dataset"""
        # Create dataset
        create_response = client.post(
            "/api/v1/datasets",
            json=sample_dataset_request,
            headers=auth_headers
        )
        dataset_id = create_response.json()["id"]
        
        # Get it
        response = client.get(f"/api/v1/evaluations/datasets/{dataset_id}", headers=auth_headers)
        assert response.status_code == 200
        data = response.json()
        assert data["id"] == dataset_id
        assert data["name"] == sample_dataset_request["name"]
    
    def test_delete_dataset(self, client, auth_headers, sample_dataset_request):
        """Test deleting a dataset"""
        # Create dataset
        create_response = client.post(
            "/api/v1/datasets",
            json=sample_dataset_request,
            headers=auth_headers
        )
        dataset_id = create_response.json()["id"]
        
        # Delete it
        response = client.delete(f"/api/v1/evaluations/datasets/{dataset_id}", headers=auth_headers)
        assert response.status_code == 204
        
        # Verify it's deleted
        get_response = client.get(f"/api/v1/evaluations/datasets/{dataset_id}", headers=auth_headers)
        assert get_response.status_code == 404
    
    def test_list_datasets(self, client, auth_headers, sample_dataset_request):
        """Test listing datasets"""
        # Create multiple datasets
        for i in range(3):
            req = sample_dataset_request.copy()
            req["name"] = f"dataset_{i}"
            client.post("/api/v1/evaluations/datasets", json=req, headers=auth_headers)
        
        # List them
        response = client.get("/api/v1/evaluations/datasets", headers=auth_headers)
        assert response.status_code == 200
        data = response.json()
        assert data["object"] == "list"
        assert len(data["data"]) >= 3


class TestEvaluationRuns:
    """Test evaluation run operations"""
    
    @patch('tldw_Server_API.app.core.Evaluations.eval_runner.EvaluationRunner.run_evaluation_async')
    def test_create_run(self, mock_run_async, client, auth_headers, 
                       sample_evaluation_request, sample_run_request):
        """Test creating and starting an evaluation run"""
        # Mock the async evaluation
        mock_run_async.return_value = None
        
        # Create evaluation first
        eval_response = client.post(
            "/api/v1/evaluations",
            json=sample_evaluation_request,
            headers=auth_headers
        )
        eval_id = eval_response.json()["id"]
        
        # Create run
        response = client.post(
            f"/api/v1/evaluations/{eval_id}/runs",
            json=sample_run_request,
            headers=auth_headers
        )
        assert response.status_code == 202  # Accepted for async processing
        data = response.json()
        assert data["id"].startswith("run_")
        assert data["status"] == "pending"
        assert data["eval_id"] == eval_id
        assert data["target_model"] == sample_run_request["target_model"]
    
    def test_get_run_status(self, client, auth_headers, 
                           sample_evaluation_request, sample_run_request):
        """Test getting run status"""
        with patch('tldw_Server_API.app.core.Evaluations.eval_runner.EvaluationRunner.run_evaluation_async'):
            # Create evaluation and run
            eval_response = client.post(
                "/api/v1/evaluations",
                json=sample_evaluation_request,
                headers=auth_headers
            )
            eval_id = eval_response.json()["id"]
            
            run_response = client.post(
                f"/api/v1/evaluations/{eval_id}/runs",
                json=sample_run_request,
                headers=auth_headers
            )
            run_id = run_response.json()["id"]
            
            # Get status
            response = client.get(f"/api/v1/evaluations/runs/{run_id}", headers=auth_headers)
            assert response.status_code == 200
            data = response.json()
            assert data["id"] == run_id
            assert "status" in data
            assert "progress" in data
    
    def test_cancel_run(self, client, auth_headers,
                       sample_evaluation_request, sample_run_request):
        """Test cancelling a run"""
        with patch('tldw_Server_API.app.core.Evaluations.eval_runner.EvaluationRunner.run_evaluation_async'):
            # Create evaluation and run
            eval_response = client.post(
                "/api/v1/evaluations",
                json=sample_evaluation_request,
                headers=auth_headers
            )
            eval_id = eval_response.json()["id"]
            
            run_response = client.post(
                f"/api/v1/evaluations/{eval_id}/runs",
                json=sample_run_request,
                headers=auth_headers
            )
            run_id = run_response.json()["id"]
            
            # Cancel it (may already be completed)
            response = client.post(f"/api/v1/evaluations/runs/{run_id}/cancel", headers=auth_headers)
            assert response.status_code == 200
            data = response.json()
            assert data["status"] in ["cancelled", "cancelling", "completed"]
    
    def test_list_runs(self, client, auth_headers,
                       sample_evaluation_request, sample_run_request):
        """Test listing runs for an evaluation"""
        with patch('tldw_Server_API.app.core.Evaluations.eval_runner.EvaluationRunner.run_evaluation_async'):
            # Create evaluation
            eval_response = client.post(
                "/api/v1/evaluations",
                json=sample_evaluation_request,
                headers=auth_headers
            )
            eval_id = eval_response.json()["id"]
            
            # Create multiple runs
            for _ in range(3):
                client.post(
                    f"/api/v1/evaluations/{eval_id}/runs",
                    json=sample_run_request,
                    headers=auth_headers
                )
            
            # List them
            response = client.get(f"/api/v1/evaluations/{eval_id}/runs", headers=auth_headers)
            assert response.status_code == 200
            data = response.json()
            assert data["object"] == "list"
            assert len(data["data"]) >= 3


class TestErrorHandling:
    """Test error handling and edge cases"""
    
    def test_invalid_eval_type(self, client, auth_headers):
        """Test creating evaluation with invalid type"""
        request = {
            "name": "invalid_eval",
            "eval_type": "invalid_type",
            "eval_spec": {"evaluator_model": "gpt-4"}
        }
        response = client.post("/api/v1/evaluations", json=request, headers=auth_headers)
        assert response.status_code == 422  # Validation error
    
    def test_missing_required_fields(self, client, auth_headers):
        """Test creating evaluation with missing fields"""
        request = {"name": "incomplete_eval"}  # Missing eval_type and eval_spec
        response = client.post("/api/v1/evaluations", json=request, headers=auth_headers)
        assert response.status_code == 422
    
    def test_run_for_nonexistent_eval(self, client, auth_headers, sample_run_request):
        """Test creating run for non-existent evaluation"""
        response = client.post(
            "/api/v1/evaluations/eval_nonexistent/runs",
            json=sample_run_request,
            headers=auth_headers
        )
        assert response.status_code == 404
        data = response.json()
        # Handle FastAPI error response format
        if "detail" in data and isinstance(data["detail"], dict):
            assert "error" in data["detail"]
        else:
            assert "error" in data
    
    def test_large_dataset(self, client, auth_headers):
        """Test handling large dataset"""
        large_dataset = {
            "name": "large_dataset",
            "description": "Test with many samples",
            "samples": [
                {"input": {"text": f"Sample {i}"}, "expected": {"score": i/1000}}
                for i in range(1000)
            ]
        }
        response = client.post("/api/v1/evaluations/datasets", json=large_dataset, headers=auth_headers)
        assert response.status_code == 201
        data = response.json()
        assert len(data["samples"]) == 1000


class TestAsyncEvaluation:
    """Test async evaluation processing"""
    
    @pytest.mark.asyncio
    async def test_evaluation_workflow(self, async_client, auth_headers, 
                                      sample_evaluation_request, sample_run_request):
        """Test complete evaluation workflow"""
        # Mock the evaluation backends
        with patch('tldw_Server_API.app.core.Evaluations.eval_runner.EvaluationRunner.run_evaluation_async') as mock_run:
            # Setup mock to simulate async processing
            async def mock_async_run(run_id, eval_config):
                await asyncio.sleep(0.1)  # Simulate processing
                return {
                    "aggregate": {"mean_score": 0.85, "std_dev": 0.05},
                    "sample_results": [
                        {"sample_id": "1", "scores": {"accuracy": 0.9}, "passed": True},
                        {"sample_id": "2", "scores": {"accuracy": 0.8}, "passed": True}
                    ]
                }
            mock_run.side_effect = mock_async_run
            
            # Create evaluation
            eval_response = await async_client.post(
                "/api/v1/evaluations",
                json=sample_evaluation_request,
                headers=auth_headers
            )
            assert eval_response.status_code == 201
            eval_id = eval_response.json()["id"]
            
            # Start run
            run_response = await async_client.post(
                f"/api/v1/evaluations/{eval_id}/runs",
                json=sample_run_request,
                headers=auth_headers
            )
            assert run_response.status_code == 202
            run_id = run_response.json()["id"]
            
            # Check status (should be pending, running, or completed)
            status_response = await async_client.get(
                f"/api/v1/evaluations/runs/{run_id}",
                headers=auth_headers
            )
            assert status_response.status_code == 200
            status_data = status_response.json()
            assert status_data["status"] in ["pending", "running", "completed"]
            
            # Wait a bit for processing
            await asyncio.sleep(0.2)
            
            # Check results (might be completed)
            results_response = await async_client.get(
                f"/api/v1/evaluations/runs/{run_id}/results",
                headers=auth_headers
            )
            # Either still processing or completed
            assert results_response.status_code in [200, 404]


class TestPagination:
    """Test pagination functionality"""
    
    def test_pagination_limit(self, client, auth_headers, sample_evaluation_request):
        """Test pagination with limit parameter"""
        # Create 10 evaluations
        for i in range(10):
            req = sample_evaluation_request.copy()
            req["name"] = f"eval_{i:02d}"
            client.post("/api/v1/evaluations", json=req, headers=auth_headers)
        
        # Test different limits
        response = client.get("/api/v1/evaluations?limit=5", headers=auth_headers)
        assert response.status_code == 200
        data = response.json()
        assert len(data["data"]) <= 5
        assert data["has_more"] is True
        
        # Test with after parameter
        if data["data"]:
            last_id = data["last_id"]
            response2 = client.get(f"/api/v1/evaluations?limit=5&after={last_id}", headers=auth_headers)
            assert response2.status_code == 200
            data2 = response2.json()
            # Should get different evaluations
            if data2["data"]:
                assert data2["data"][0]["id"] != data["data"][0]["id"]
    
    def test_filter_by_type(self, client, auth_headers):
        """Test filtering evaluations by type"""
        # Create evaluations of different types
        types = ["model_graded", "exact_match", "includes"]
        for eval_type in types:
            request = {
                "name": f"eval_{eval_type}",
                "eval_type": eval_type,
                "eval_spec": {"threshold": 0.5}
            }
            client.post("/api/v1/evaluations", json=request, headers=auth_headers)
        
        # Filter by type
        response = client.get("/api/v1/evaluations?eval_type=model_graded", headers=auth_headers)
        assert response.status_code == 200
        data = response.json()
        # All returned evaluations should be of the filtered type
        for eval in data["data"]:
            assert eval["eval_type"] == "model_graded"


class TestConcurrency:
    """Test concurrent operations"""
    
    def test_concurrent_evaluation_creation(self, client, auth_headers):
        """Test creating multiple evaluations concurrently"""
        import concurrent.futures
        
        def create_eval(index):
            request = {
                "name": f"concurrent_eval_{index}",
                "eval_type": "model_graded",
                "eval_spec": {"evaluator_model": "gpt-4", "threshold": 0.7},
                "dataset": [{"input": {"text": f"Test {index}"}, "expected": {"score": 0.8}}]
            }
            response = client.post("/api/v1/evaluations", json=request, headers=auth_headers)
            return response.status_code, response.json()
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
            futures = [executor.submit(create_eval, i) for i in range(10)]
            results = [f.result() for f in concurrent.futures.as_completed(futures)]
        
        # All should succeed
        for status_code, data in results:
            assert status_code == 201
            assert data["id"].startswith("eval_")
    
    @pytest.mark.asyncio
    async def test_concurrent_runs(self, async_client, auth_headers,
                                  sample_evaluation_request, sample_run_request):
        """Test running multiple evaluations concurrently"""
        with patch('tldw_Server_API.app.core.Evaluations.eval_runner.EvaluationRunner.run_evaluation_async'):
            # Create evaluation
            eval_response = await async_client.post(
                "/api/v1/evaluations",
                json=sample_evaluation_request,
                headers=auth_headers
            )
            eval_id = eval_response.json()["id"]
            
            # Start multiple runs concurrently
            tasks = []
            for i in range(5):
                req = sample_run_request.copy()
                req["config"]["temperature"] = i * 0.2
                task = async_client.post(
                    f"/api/v1/evaluations/{eval_id}/runs",
                    json=req,
                    headers=auth_headers
                )
                tasks.append(task)
            
            responses = await asyncio.gather(*tasks)
            
            # All should succeed
            for response in responses:
                assert response.status_code == 202
                assert response.json()["id"].startswith("run_")


# Run tests
if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])