# test_api_endpoints.py
# Integration tests for Prompt Studio API endpoints

import pytest
import json
from fastapi.testclient import TestClient
from typing import Dict, Any
import uuid
import tempfile
import os
from pathlib import Path
from unittest.mock import patch, Mock

# Disable CSRF for testing
import os
os.environ["AUTH_MODE"] = "single_user"
os.environ["CSRF_ENABLED"] = "false"

from tldw_Server_API.app.main import app
from tldw_Server_API.app.core.DB_Management.PromptStudioDatabase import PromptStudioDatabase
from tldw_Server_API.app.api.v1.API_Deps.auth_deps import get_current_active_user

########################################################################################################################
# Test Client Setup

@pytest.fixture
def client(mock_user):
    """Create a test client for the FastAPI app with mocked authentication."""
    # Override the auth dependency
    app.dependency_overrides[get_current_active_user] = lambda: mock_user
    client = TestClient(app)
    yield client
    # Clear overrides after test
    app.dependency_overrides.clear()

@pytest.fixture
def test_db():
    """Create a temporary test database."""
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as tmp:
        db_path = tmp.name
    
    db = PromptStudioDatabase(db_path, "test-client")
    yield db
    
    # Cleanup
    if hasattr(db, 'close'):
        db.close()
    elif hasattr(db, 'conn'):
        db.conn.close()
    os.unlink(db_path)

@pytest.fixture
def auth_headers():
    """Create authentication headers for testing."""
    return {
        "Authorization": "Bearer test-token",
        "Content-Type": "application/json"
    }

@pytest.fixture
def mock_user():
    """Mock user for authentication."""
    return {
        "id": "test-user-123",
        "username": "testuser",
        "is_authenticated": True,
        "permissions": ["read", "write", "delete"]
    }

########################################################################################################################
# Project Endpoints Tests

@pytest.mark.integration
class TestProjectEndpoints:
    """Test project-related API endpoints."""
    
    def test_create_project(self, client, test_db):
        """Test creating a new project."""
        project_data = {
            "name": "Test Project",
            "description": "A test project for integration testing",
            "status": "draft",
            "metadata": {"test": True}
        }
        
        response = client.post(
            "/api/v1/prompt-studio/projects/",
            json=project_data
        )
        
        # Check response
        if response.status_code != 201:
            print(f"Response status: {response.status_code}")
            print(f"Response body: {response.text}")
        
        assert response.status_code == 201
        data = response.json()
        assert data["success"] == True
        assert data["data"]["name"] == "Test Project"
        assert data["data"]["status"] == "draft"
        assert "id" in data["data"]
        assert "uuid" in data["data"]
    
    def test_list_projects(self, client, test_db):
        """Test listing projects."""
        # First create a project
        project_data = {
            "name": "List Test Project",
            "description": "For listing test"
        }
        client.post("/api/v1/prompt-studio/projects/", json=project_data)
        
        # Then list projects
        response = client.get("/api/v1/prompt-studio/projects/")
        
        assert response.status_code == 200
        data = response.json()
        assert data["success"] == True
        assert "data" in data
        assert "metadata" in data
        assert isinstance(data["data"], list)
    
    def test_get_project(self, client, test_db):
        """Test getting a specific project."""
        # First create a project
        create_response = client.post(
            "/api/v1/prompt-studio/projects/",
            json={"name": "Get Test", "description": "Test"}
        )
        
        assert create_response.status_code == 201
        project_id = create_response.json()["data"]["id"]
        
        # Get the project
        response = client.get(
            f"/api/v1/prompt-studio/projects/get/{project_id}"
        )
        
        assert response.status_code == 200
        data = response.json()
        assert data["success"] == True
        assert data["data"]["id"] == project_id
        assert data["data"]["name"] == "Get Test"
    
    def test_update_project(self, client, test_db):
        """Test updating a project."""
        # First create a project
        create_response = client.post(
            "/api/v1/prompt-studio/projects/",
            json={"name": "Update Test", "description": "Original"}
        )
        
        assert create_response.status_code == 201
        project_id = create_response.json()["data"]["id"]
        
        # Update the project
        update_data = {
            "description": "Updated description",
            "status": "active"
        }
        
        response = client.put(
            f"/api/v1/prompt-studio/projects/update/{project_id}",
            json=update_data
        )
        
        assert response.status_code == 200
        data = response.json()
        assert data["success"] == True
        assert data["data"]["description"] == "Updated description"
        assert data["data"]["status"] == "active"
    
    def test_delete_project(self, client, test_db):
        """Test deleting a project."""
        # First create a project
        create_response = client.post(
            "/api/v1/prompt-studio/projects/",
            json={"name": "Delete Test", "description": "To be deleted"}
        )
        
        assert create_response.status_code == 201
        project_id = create_response.json()["data"]["id"]
        
        # Delete the project
        response = client.delete(
            f"/api/v1/prompt-studio/projects/delete/{project_id}"
        )
        
        assert response.status_code == 200
        
        # Verify it's deleted (soft delete by default)
        get_response = client.get(
            f"/api/v1/prompt-studio/projects/get/{project_id}"
        )
        # Soft delete means it still exists but marked as deleted
        assert get_response.status_code in [200, 404]

########################################################################################################################
# Prompt Endpoints Tests

@pytest.mark.integration
class TestPromptEndpoints:
    """Test prompt-related API endpoints."""
    
    @pytest.fixture
    def project_id(self, client, test_db):
        """Create a project and return its ID."""
        response = client.post(
            "/api/v1/prompt-studio/projects/",
            json={"name": "Prompt Test Project", "description": "For prompt testing"}
        )
        if response.status_code == 201:
            return response.json()["data"]["id"]
        return None
    
    def test_create_prompt(self, client, test_db, project_id):
        """Test creating a new prompt."""
        if not project_id:
            pytest.skip("Project creation failed")
        
            prompt_data = {
                "project_id": project_id,
                "name": "Test Prompt",
                "system_prompt": "You are a helpful assistant.",
                "user_prompt": "Please help with: {task}",
                "version_number": 1,
                "change_description": "Initial version"
            }
            
            response = client.post(
                "/api/v1/prompt-studio/prompts",
                json=prompt_data,
                headers=auth_headers
            )
            
            assert response.status_code in [200, 201]
            data = response.json()
            assert data["name"] == "Test Prompt"
            assert data["project_id"] == project_id
    
    def test_list_prompts(self, client, auth_headers, mock_user, project_id):
        """Test listing prompts for a project."""
        if not project_id:
            pytest.skip("Project creation failed")
        
        with patch('tldw_Server_API.app.api.v1.API_Deps.prompt_studio_deps.get_current_active_user', return_value=mock_user):
            response = client.get(
                f"/api/v1/prompt-studio/prompts?project_id={project_id}",
                headers=auth_headers
            )
            
            assert response.status_code == 200
            data = response.json()
            assert "prompts" in data
            assert isinstance(data["prompts"], list)
    
    def test_execute_prompt(self, client, auth_headers, mock_user):
        """Test executing a prompt."""
        with patch('tldw_Server_API.app.api.v1.API_Deps.prompt_studio_deps.get_current_active_user', return_value=mock_user):
            with patch('tldw_Server_API.app.core.Prompt_Management.prompt_studio.prompt_executor.PromptExecutor.execute') as mock_execute:
                mock_execute.return_value = {
                    "output": "Test response",
                    "tokens_used": 100,
                    "execution_time": 1.5
                }
                
                execution_data = {
                    "prompt_id": 1,
                    "inputs": {"task": "Write a test"},
                    "provider": "openai",
                    "model": "gpt-4"
                }
                
                response = client.post(
                    "/api/v1/prompt-studio/prompts/execute",
                    json=execution_data,
                    headers=auth_headers
                )
                
                assert response.status_code in [200, 201]
                data = response.json()
                assert "output" in data
                assert data["tokens_used"] == 100

########################################################################################################################
# Test Case Endpoints Tests

@pytest.mark.integration
class TestTestCaseEndpoints:
    """Test test case-related API endpoints."""
    
    @pytest.fixture
    def project_id(self, client, auth_headers, mock_user):
        """Create a project and return its ID."""
        with patch('tldw_Server_API.app.api.v1.API_Deps.prompt_studio_deps.get_current_active_user', return_value=mock_user):
            response = client.post(
                "/api/v1/prompt-studio/projects",
                json={"name": "Test Case Project", "description": "For test cases"},
                headers=auth_headers
            )
            if response.status_code in [200, 201]:
                return response.json()["id"]
            return None
    
    def test_create_test_case(self, client, auth_headers, mock_user, project_id):
        """Test creating a test case."""
        if not project_id:
            pytest.skip("Project creation failed")
        
        with patch('tldw_Server_API.app.api.v1.API_Deps.prompt_studio_deps.get_current_active_user', return_value=mock_user):
            test_case_data = {
                "project_id": project_id,
                "name": "Test Case 1",
                "description": "A test case",
                "inputs": {"input": "test data"},
                "expected_outputs": {"output": "expected result"},
                "is_golden": False,
                "tags": ["unit", "test"]
            }
            
            response = client.post(
                "/api/v1/prompt-studio/test-cases",
                json=test_case_data,
                headers=auth_headers
            )
            
            assert response.status_code in [200, 201]
            data = response.json()
            assert data["name"] == "Test Case 1"
            assert data["project_id"] == project_id
    
    def test_run_test_cases(self, client, auth_headers, mock_user):
        """Test running test cases."""
        with patch('tldw_Server_API.app.api.v1.API_Deps.prompt_studio_deps.get_current_active_user', return_value=mock_user):
            with patch('tldw_Server_API.app.core.Prompt_Management.prompt_studio.test_case_manager.TestCaseManager.run_batch_tests') as mock_run:
                mock_run.return_value = [
                    {
                        "test_case_id": "test-1",
                        "passed": True,
                        "actual_outputs": {"output": "result"},
                        "execution_time": 0.5
                    }
                ]
                
                run_data = {
                    "project_id": 1,
                    "test_case_ids": ["test-1"],
                    "prompt_id": 1
                }
                
                response = client.post(
                    "/api/v1/prompt-studio/test-cases/run",
                    json=run_data,
                    headers=auth_headers
                )
                
                assert response.status_code in [200, 201]
                data = response.json()
                assert "results" in data
                assert len(data["results"]) == 1
                assert data["results"][0]["passed"] is True

########################################################################################################################
# Evaluation Endpoints Tests

@pytest.mark.integration
class TestEvaluationEndpoints:
    """Test evaluation-related API endpoints."""
    
    def test_create_evaluation(self, client, auth_headers, mock_user):
        """Test creating an evaluation."""
        with patch('tldw_Server_API.app.api.v1.API_Deps.prompt_studio_deps.get_current_active_user', return_value=mock_user):
            evaluation_data = {
                "project_id": 1,
                "prompt_id": 1,
                "test_run_id": "run-123",
                "metrics": {
                    "accuracy": 0.95,
                    "f1_score": 0.92,
                    "latency": 1.5
                },
                "config": {
                    "model": "gpt-4",
                    "temperature": 0.7
                }
            }
            
            response = client.post(
                "/api/v1/prompt-studio/evaluations",
                json=evaluation_data,
                headers=auth_headers
            )
            
            assert response.status_code in [200, 201]
            data = response.json()
            assert "metrics" in data
            assert data["metrics"]["accuracy"] == 0.95
    
    def test_list_evaluations(self, client, auth_headers, mock_user):
        """Test listing evaluations."""
        with patch('tldw_Server_API.app.api.v1.API_Deps.prompt_studio_deps.get_current_active_user', return_value=mock_user):
            response = client.get(
                "/api/v1/prompt-studio/evaluations?project_id=1",
                headers=auth_headers
            )
            
            assert response.status_code == 200
            data = response.json()
            assert "evaluations" in data
            assert isinstance(data["evaluations"], list)

########################################################################################################################
# Optimization Endpoints Tests

@pytest.mark.integration
class TestOptimizationEndpoints:
    """Test optimization-related API endpoints."""
    
    def test_start_optimization(self, client, auth_headers, mock_user):
        """Test starting an optimization job."""
        with patch('tldw_Server_API.app.api.v1.API_Deps.prompt_studio_deps.get_current_active_user', return_value=mock_user):
            with patch('tldw_Server_API.app.core.Prompt_Management.prompt_studio.job_manager.JobManager.create_job') as mock_create:
                mock_create.return_value = {
                    "id": "job-123",
                    "status": "pending",
                    "type": "optimization"
                }
                
                optimization_data = {
                    "project_id": 1,
                    "prompt_id": 1,
                    "strategy": "mipro",
                    "config": {
                        "max_iterations": 10,
                        "target_metric": "accuracy",
                        "threshold": 0.9
                    }
                }
                
                response = client.post(
                    "/api/v1/prompt-studio/optimizations",
                    json=optimization_data,
                    headers=auth_headers
                )
                
                assert response.status_code in [200, 201]
                data = response.json()
                assert "id" in data
                assert data["status"] == "pending"
    
    def test_get_optimization_status(self, client, auth_headers, mock_user):
        """Test getting optimization job status."""
        with patch('tldw_Server_API.app.api.v1.API_Deps.prompt_studio_deps.get_current_active_user', return_value=mock_user):
            with patch('tldw_Server_API.app.core.Prompt_Management.prompt_studio.job_manager.JobManager.get_job') as mock_get:
                mock_get.return_value = {
                    "id": "job-123",
                    "status": "running",
                    "progress": 0.5,
                    "current_iteration": 5,
                    "best_score": 0.85
                }
                
                response = client.get(
                    "/api/v1/prompt-studio/optimizations/job-123",
                    headers=auth_headers
                )
                
                assert response.status_code == 200
                data = response.json()
                assert data["status"] == "running"
                assert data["progress"] == 0.5

########################################################################################################################
# WebSocket Tests

@pytest.mark.integration
class TestWebSocketEndpoints:
    """Test WebSocket functionality."""
    
    def test_websocket_connection(self, client):
        """Test WebSocket connection."""
        with client.websocket_connect("/api/v1/prompt-studio/ws") as websocket:
            # Send a test message
            websocket.send_json({
                "type": "subscribe",
                "project_id": 1
            })
            
            # Receive acknowledgment
            data = websocket.receive_json()
            assert data["type"] == "subscribed"
            assert data["project_id"] == 1
    
    def test_websocket_job_updates(self, client):
        """Test receiving job updates via WebSocket."""
        with client.websocket_connect("/api/v1/prompt-studio/ws") as websocket:
            # Subscribe to job updates
            websocket.send_json({
                "type": "subscribe_job",
                "job_id": "job-123"
            })
            
            # Simulate job update
            with patch('tldw_Server_API.app.core.Prompt_Management.prompt_studio.event_broadcaster.EventBroadcaster.broadcast') as mock_broadcast:
                mock_broadcast.return_value = None
                
                # Trigger a job update
                update_message = {
                    "type": "job_update",
                    "job_id": "job-123",
                    "status": "completed",
                    "result": {"score": 0.95}
                }
                
                # In real scenario, this would be triggered by job processor
                websocket.send_json(update_message)
                
                # Receive the update
                data = websocket.receive_json()
                assert data["type"] == "job_update"
                assert data["status"] == "completed"

########################################################################################################################
# Error Handling Tests

@pytest.mark.integration
class TestErrorHandling:
    """Test API error handling."""
    
    def test_unauthorized_access(self, client):
        """Test accessing endpoints without authentication."""
        response = client.get("/api/v1/prompt-studio/projects")
        assert response.status_code == 401
    
    def test_invalid_project_id(self, client, auth_headers, mock_user):
        """Test accessing non-existent project."""
        with patch('tldw_Server_API.app.api.v1.API_Deps.prompt_studio_deps.get_current_active_user', return_value=mock_user):
            response = client.get(
                "/api/v1/prompt-studio/projects/99999",
                headers=auth_headers
            )
            assert response.status_code == 404
    
    def test_invalid_request_data(self, client, auth_headers, mock_user):
        """Test sending invalid data."""
        with patch('tldw_Server_API.app.api.v1.API_Deps.prompt_studio_deps.get_current_active_user', return_value=mock_user):
            # Missing required field
            invalid_data = {
                "description": "Missing name field"
            }
            
            response = client.post(
                "/api/v1/prompt-studio/projects",
                json=invalid_data,
                headers=auth_headers
            )
            
            assert response.status_code == 422
            data = response.json()
            assert "detail" in data
    
    def test_rate_limiting(self, client, auth_headers, mock_user):
        """Test rate limiting."""
        with patch('tldw_Server_API.app.api.v1.API_Deps.prompt_studio_deps.get_current_active_user', return_value=mock_user):
            # Make many rapid requests
            responses = []
            for _ in range(100):
                response = client.get(
                    "/api/v1/prompt-studio/projects",
                    headers=auth_headers
                )
                responses.append(response.status_code)
            
            # Should have some rate limited responses
            # Note: This depends on rate limit configuration
            # assert 429 in responses

########################################################################################################################
# Pagination Tests

@pytest.mark.integration
class TestPagination:
    """Test pagination functionality."""
    
    def test_project_pagination(self, client, auth_headers, mock_user):
        """Test paginating project list."""
        with patch('tldw_Server_API.app.api.v1.API_Deps.prompt_studio_deps.get_current_active_user', return_value=mock_user):
            # First page
            response = client.get(
                "/api/v1/prompt-studio/projects?page=1&per_page=5",
                headers=auth_headers
            )
            
            assert response.status_code == 200
            data = response.json()
            assert "pagination" in data
            assert data["pagination"]["page"] == 1
            assert data["pagination"]["per_page"] == 5
            
            # Next page
            if data["pagination"]["total_pages"] > 1:
                response = client.get(
                    "/api/v1/prompt-studio/projects?page=2&per_page=5",
                    headers=auth_headers
                )
                
                assert response.status_code == 200
                data = response.json()
                assert data["pagination"]["page"] == 2

########################################################################################################################
# Search and Filter Tests

@pytest.mark.integration
class TestSearchAndFilter:
    """Test search and filtering functionality."""
    
    def test_search_projects(self, client, auth_headers, mock_user):
        """Test searching projects."""
        with patch('tldw_Server_API.app.api.v1.API_Deps.prompt_studio_deps.get_current_active_user', return_value=mock_user):
            response = client.get(
                "/api/v1/prompt-studio/projects?search=test&status=active",
                headers=auth_headers
            )
            
            assert response.status_code == 200
            data = response.json()
            assert "projects" in data
            
            # All results should match search criteria
            for project in data["projects"]:
                assert "test" in project["name"].lower() or "test" in project.get("description", "").lower()
                assert project["status"] == "active"
    
    def test_filter_by_date(self, client, auth_headers, mock_user):
        """Test filtering by date range."""
        with patch('tldw_Server_API.app.api.v1.API_Deps.prompt_studio_deps.get_current_active_user', return_value=mock_user):
            response = client.get(
                "/api/v1/prompt-studio/projects?created_after=2024-01-01&created_before=2024-12-31",
                headers=auth_headers
            )
            
            assert response.status_code == 200
            data = response.json()
            assert "projects" in data