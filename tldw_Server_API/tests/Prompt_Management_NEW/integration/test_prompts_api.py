"""
Integration tests for Prompt Management API endpoints.

Tests the complete API flow with real components, no mocking.
"""

import pytest
import json
from datetime import datetime
from fastapi.testclient import TestClient

# ========================================================================
# Prompt CRUD Endpoint Tests
# ========================================================================

class TestPromptCRUDEndpoints:
    """Test prompt CRUD operations through API."""
    
    @pytest.mark.integration
    def test_create_prompt_endpoint(self, test_client, auth_headers, sample_prompt):
        """Test creating a prompt via API."""
        response = test_client.post(
            "/api/v1/prompts/create",
            json=sample_prompt,
            headers=auth_headers
        )
        
        assert response.status_code == 200
        data = response.json()
        assert 'prompt_id' in data
        assert data['prompt_id'] > 0
    
    @pytest.mark.integration
    def test_get_prompt_endpoint(self, test_client, auth_headers, populated_prompts_db):
        """Test getting a prompt via API."""
        # First create a prompt
        create_response = test_client.post(
            "/api/v1/prompts/create",
            json={
                'name': 'API Test Prompt',
                'content': 'Test content {{variable}}',
                'author': 'api_test',
                'keywords': ['test', 'api']
            },
            headers=auth_headers
        )
        prompt_id = create_response.json()['prompt_id']
        
        # Then get it
        response = test_client.get(
            f"/api/v1/prompts/get/{prompt_id}",
            headers=auth_headers
        )
        
        assert response.status_code == 200
        data = response.json()
        assert data['name'] == 'API Test Prompt'
        assert data['content'] == 'Test content {{variable}}'
        assert 'test' in data['keywords']
    
    @pytest.mark.integration
    def test_list_prompts_endpoint(self, test_client, auth_headers, populated_prompts_db):
        """Test listing prompts via API."""
        response = test_client.get(
            "/api/v1/prompts/list",
            headers=auth_headers
        )
        
        assert response.status_code == 200
        data = response.json()
        assert 'prompts' in data
        assert len(data['prompts']) >= 3  # From populated_prompts_db
    
    @pytest.mark.integration
    def test_update_prompt_endpoint(self, test_client, auth_headers):
        """Test updating a prompt via API."""
        # Create prompt
        create_response = test_client.post(
            "/api/v1/prompts/create",
            json={
                'name': 'Update Test',
                'content': 'Original content',
                'author': 'test'
            },
            headers=auth_headers
        )
        prompt_id = create_response.json()['prompt_id']
        
        # Update it
        update_response = test_client.put(
            f"/api/v1/prompts/update/{prompt_id}",
            json={
                'content': 'Updated content {{new_var}}',
                'version_comment': 'Added variable'
            },
            headers=auth_headers
        )
        
        assert update_response.status_code == 200
        data = update_response.json()
        assert data['success'] is True
        assert 'new_version' in data
    
    @pytest.mark.integration
    def test_delete_prompt_endpoint(self, test_client, auth_headers):
        """Test deleting a prompt via API."""
        # Create prompt
        create_response = test_client.post(
            "/api/v1/prompts/create",
            json={
                'name': 'Delete Test',
                'content': 'To be deleted',
                'author': 'test'
            },
            headers=auth_headers
        )
        prompt_id = create_response.json()['prompt_id']
        
        # Delete it
        delete_response = test_client.delete(
            f"/api/v1/prompts/delete/{prompt_id}",
            headers=auth_headers
        )
        
        assert delete_response.status_code == 200
        data = delete_response.json()
        assert data['success'] is True
        
        # Verify it's deleted (soft delete)
        get_response = test_client.get(
            f"/api/v1/prompts/get/{prompt_id}",
            headers=auth_headers
        )
        assert get_response.status_code == 404

# ========================================================================
# Version Management Endpoint Tests
# ========================================================================

class TestVersionEndpoints:
    """Test version management endpoints."""
    
    @pytest.mark.integration
    def test_get_versions_endpoint(self, test_client, auth_headers):
        """Test getting prompt versions via API."""
        # Create prompt with multiple versions
        create_response = test_client.post(
            "/api/v1/prompts/create",
            json={
                'name': 'Version Test',
                'content': 'Version 1',
                'author': 'test'
            },
            headers=auth_headers
        )
        prompt_id = create_response.json()['prompt_id']
        
        # Create version 2
        test_client.put(
            f"/api/v1/prompts/update/{prompt_id}",
            json={'content': 'Version 2'},
            headers=auth_headers
        )
        
        # Create version 3
        test_client.put(
            f"/api/v1/prompts/update/{prompt_id}",
            json={'content': 'Version 3'},
            headers=auth_headers
        )
        
        # Get versions
        response = test_client.get(
            f"/api/v1/prompts/{prompt_id}/versions",
            headers=auth_headers
        )
        
        assert response.status_code == 200
        data = response.json()
        assert 'versions' in data
        assert len(data['versions']) >= 3
    
    @pytest.mark.integration
    def test_restore_version_endpoint(self, test_client, auth_headers):
        """Test restoring a specific version via API."""
        # Create prompt with versions
        create_response = test_client.post(
            "/api/v1/prompts/create",
            json={
                'name': 'Restore Test',
                'content': 'Original',
                'author': 'test'
            },
            headers=auth_headers
        )
        prompt_id = create_response.json()['prompt_id']
        
        # Update to v2
        test_client.put(
            f"/api/v1/prompts/update/{prompt_id}",
            json={'content': 'Version 2'},
            headers=auth_headers
        )
        
        # Update to v3
        test_client.put(
            f"/api/v1/prompts/update/{prompt_id}",
            json={'content': 'Version 3'},
            headers=auth_headers
        )
        
        # Restore to v1
        restore_response = test_client.post(
            f"/api/v1/prompts/{prompt_id}/restore",
            json={'version': 1},
            headers=auth_headers
        )
        
        assert restore_response.status_code == 200
        
        # Verify content is restored
        get_response = test_client.get(
            f"/api/v1/prompts/get/{prompt_id}",
            headers=auth_headers
        )
        assert 'Original' in get_response.json()['content']

# ========================================================================
# Search and Filter Endpoint Tests
# ========================================================================

class TestSearchEndpoints:
    """Test search and filter endpoints."""
    
    @pytest.mark.integration
    def test_search_prompts_endpoint(self, test_client, auth_headers, populated_prompts_db):
        """Test searching prompts via API."""
        response = test_client.get(
            "/api/v1/prompts/search",
            params={'query': 'assistant'},
            headers=auth_headers
        )
        
        assert response.status_code == 200
        data = response.json()
        assert 'results' in data
        assert len(data['results']) > 0
    
    @pytest.mark.integration
    def test_filter_by_keywords_endpoint(self, test_client, auth_headers, populated_prompts_db):
        """Test filtering by keywords via API."""
        response = test_client.post(
            "/api/v1/prompts/filter",
            json={'keywords': ['test', 'assistant']},
            headers=auth_headers
        )
        
        assert response.status_code == 200
        data = response.json()
        assert 'prompts' in data
        for prompt in data['prompts']:
            assert any(k in ['test', 'assistant'] for k in prompt.get('keywords', []))
    
    @pytest.mark.integration
    def test_filter_by_author_endpoint(self, test_client, auth_headers, populated_prompts_db):
        """Test filtering by author via API."""
        response = test_client.post(
            "/api/v1/prompts/filter",
            json={'author': 'test_user'},
            headers=auth_headers
        )
        
        assert response.status_code == 200
        data = response.json()
        assert 'prompts' in data
        for prompt in data['prompts']:
            assert prompt['author'] == 'test_user'

# ========================================================================
# Import/Export Endpoint Tests
# ========================================================================

class TestImportExportEndpoints:
    """Test import/export endpoints."""
    
    @pytest.mark.integration
    def test_export_prompts_endpoint(self, test_client, auth_headers, populated_prompts_db):
        """Test exporting prompts via API."""
        response = test_client.post(
            "/api/v1/prompts/export",
            json={'prompt_ids': [1, 2]},
            headers=auth_headers
        )
        
        assert response.status_code == 200
        data = response.json()
        assert 'version' in data
        assert 'prompts' in data
        assert 'exported_at' in data
        assert len(data['prompts']) >= 2
    
    @pytest.mark.integration
    def test_import_prompts_endpoint(self, test_client, auth_headers, export_data):
        """Test importing prompts via API."""
        response = test_client.post(
            "/api/v1/prompts/import",
            json=export_data,
            headers=auth_headers
        )
        
        assert response.status_code == 200
        data = response.json()
        assert 'imported' in data
        assert 'failed' in data
        assert data['imported'] > 0
    
    @pytest.mark.integration
    def test_import_with_duplicates_endpoint(self, test_client, auth_headers, export_data):
        """Test importing with duplicate handling via API."""
        # Import once
        first_response = test_client.post(
            "/api/v1/prompts/import",
            json=export_data,
            headers=auth_headers
        )
        assert first_response.status_code == 200
        
        # Import again with skip_duplicates
        second_response = test_client.post(
            "/api/v1/prompts/import",
            json={**export_data, 'skip_duplicates': True},
            headers=auth_headers
        )
        
        assert second_response.status_code == 200
        data = second_response.json()
        assert 'skipped' in data
        assert data['skipped'] > 0

# ========================================================================
# Template Processing Endpoint Tests
# ========================================================================

class TestTemplateEndpoints:
    """Test template processing endpoints."""
    
    @pytest.mark.integration
    def test_render_template_endpoint(self, test_client, auth_headers):
        """Test rendering a template via API."""
        # Create template prompt
        create_response = test_client.post(
            "/api/v1/prompts/create",
            json={
                'name': 'Template Test',
                'content': 'You are a {{role}}. {{instruction}}',
                'author': 'test'
            },
            headers=auth_headers
        )
        prompt_id = create_response.json()['prompt_id']
        
        # Render template
        render_response = test_client.post(
            f"/api/v1/prompts/{prompt_id}/render",
            json={
                'variables': {
                    'role': 'helpful assistant',
                    'instruction': 'Please help the user'
                }
            },
            headers=auth_headers
        )
        
        assert render_response.status_code == 200
        data = render_response.json()
        assert 'rendered' in data
        assert 'helpful assistant' in data['rendered']
        assert '{{' not in data['rendered']
    
    @pytest.mark.integration
    def test_extract_variables_endpoint(self, test_client, auth_headers):
        """Test extracting template variables via API."""
        # Create template
        create_response = test_client.post(
            "/api/v1/prompts/create",
            json={
                'name': 'Variable Test',
                'content': 'Process {{input}} and return {{output_format}}',
                'author': 'test'
            },
            headers=auth_headers
        )
        prompt_id = create_response.json()['prompt_id']
        
        # Extract variables
        response = test_client.get(
            f"/api/v1/prompts/{prompt_id}/variables",
            headers=auth_headers
        )
        
        assert response.status_code == 200
        data = response.json()
        assert 'variables' in data
        assert 'input' in data['variables']
        assert 'output_format' in data['variables']

# ========================================================================
# Bulk Operations Endpoint Tests
# ========================================================================

class TestBulkOperationsEndpoints:
    """Test bulk operation endpoints."""
    
    @pytest.mark.integration
    def test_bulk_delete_endpoint(self, test_client, auth_headers):
        """Test bulk delete via API."""
        # Create multiple prompts
        prompt_ids = []
        for i in range(3):
            response = test_client.post(
                "/api/v1/prompts/create",
                json={
                    'name': f'Bulk Delete {i}',
                    'content': f'Content {i}',
                    'author': 'test'
                },
                headers=auth_headers
            )
            prompt_ids.append(response.json()['prompt_id'])
        
        # Bulk delete
        delete_response = test_client.post(
            "/api/v1/prompts/bulk/delete",
            json={'prompt_ids': prompt_ids},
            headers=auth_headers
        )
        
        assert delete_response.status_code == 200
        data = delete_response.json()
        assert data['deleted'] == 3
        assert data['failed'] == 0
    
    @pytest.mark.integration
    def test_bulk_update_keywords_endpoint(self, test_client, auth_headers):
        """Test bulk keyword update via API."""
        # Create prompts
        prompt_ids = []
        for i in range(2):
            response = test_client.post(
                "/api/v1/prompts/create",
                json={
                    'name': f'Keyword Test {i}',
                    'content': f'Content {i}',
                    'author': 'test',
                    'keywords': ['old-tag']
                },
                headers=auth_headers
            )
            prompt_ids.append(response.json()['prompt_id'])
        
        # Bulk update keywords
        update_response = test_client.post(
            "/api/v1/prompts/bulk/keywords",
            json={
                'prompt_ids': prompt_ids,
                'add_keywords': ['new-tag', 'bulk'],
                'remove_keywords': ['old-tag']
            },
            headers=auth_headers
        )
        
        assert update_response.status_code == 200
        data = update_response.json()
        assert data['updated'] == 2

# ========================================================================
# Collection Management Endpoint Tests
# ========================================================================

class TestCollectionEndpoints:
    """Test collection management endpoints."""
    
    @pytest.mark.integration
    def test_create_collection_endpoint(self, test_client, auth_headers):
        """Test creating a collection via API."""
        # Create prompts for collection
        prompt_ids = []
        for i in range(3):
            response = test_client.post(
                "/api/v1/prompts/create",
                json={
                    'name': f'Collection Item {i}',
                    'content': f'Content {i}',
                    'author': 'test'
                },
                headers=auth_headers
            )
            prompt_ids.append(response.json()['prompt_id'])
        
        # Create collection
        collection_response = test_client.post(
            "/api/v1/prompts/collections/create",
            json={
                'name': 'Test Collection',
                'description': 'A test collection',
                'prompt_ids': prompt_ids
            },
            headers=auth_headers
        )
        
        assert collection_response.status_code == 200
        data = collection_response.json()
        assert 'collection_id' in data
        assert data['collection_id'] > 0
    
    @pytest.mark.integration
    def test_get_collection_endpoint(self, test_client, auth_headers):
        """Test getting a collection via API."""
        # Create collection
        create_response = test_client.post(
            "/api/v1/prompts/collections/create",
            json={
                'name': 'Get Test',
                'description': 'Test',
                'prompt_ids': []
            },
            headers=auth_headers
        )
        collection_id = create_response.json()['collection_id']
        
        # Get collection
        get_response = test_client.get(
            f"/api/v1/prompts/collections/{collection_id}",
            headers=auth_headers
        )
        
        assert get_response.status_code == 200
        data = get_response.json()
        assert data['name'] == 'Get Test'
        assert data['description'] == 'Test'

# ========================================================================
# Error Handling Endpoint Tests
# ========================================================================

class TestErrorHandling:
    """Test API error handling."""
    
    @pytest.mark.integration
    def test_not_found_error(self, test_client, auth_headers):
        """Test 404 for non-existent prompt."""
        response = test_client.get(
            "/api/v1/prompts/get/99999",
            headers=auth_headers
        )
        
        assert response.status_code == 404
        data = response.json()
        assert 'detail' in data
    
    @pytest.mark.integration
    def test_validation_error(self, test_client, auth_headers):
        """Test 422 for invalid data."""
        response = test_client.post(
            "/api/v1/prompts/create",
            json={
                'name': '',  # Empty name
                'content': 'Content'
            },
            headers=auth_headers
        )
        
        assert response.status_code == 422
        data = response.json()
        assert 'detail' in data
    
    @pytest.mark.integration
    def test_unauthorized_error(self, test_client):
        """Test 401 for missing auth."""
        response = test_client.get("/api/v1/prompts/list")
        
        assert response.status_code in [401, 403]
    
    @pytest.mark.integration
    def test_duplicate_prompt_error(self, test_client, auth_headers):
        """Test handling duplicate prompt names."""
        # Create first prompt
        first_response = test_client.post(
            "/api/v1/prompts/create",
            json={
                'name': 'Unique Name',
                'content': 'Content 1',
                'author': 'test'
            },
            headers=auth_headers
        )
        assert first_response.status_code == 200
        
        # Try to create duplicate
        duplicate_response = test_client.post(
            "/api/v1/prompts/create",
            json={
                'name': 'Unique Name',
                'content': 'Content 2',
                'author': 'test'
            },
            headers=auth_headers
        )
        
        assert duplicate_response.status_code in [400, 409]

# ========================================================================
# Pagination and Filtering Tests
# ========================================================================

class TestPaginationAndFiltering:
    """Test pagination and filtering in endpoints."""
    
    @pytest.mark.integration
    def test_pagination_endpoint(self, test_client, auth_headers, populated_prompts_db):
        """Test pagination in list endpoint."""
        # Get first page
        page1_response = test_client.get(
            "/api/v1/prompts/list",
            params={'limit': 2, 'offset': 0},
            headers=auth_headers
        )
        
        assert page1_response.status_code == 200
        page1_data = page1_response.json()
        assert len(page1_data['prompts']) <= 2
        
        # Get second page
        page2_response = test_client.get(
            "/api/v1/prompts/list",
            params={'limit': 2, 'offset': 2},
            headers=auth_headers
        )
        
        assert page2_response.status_code == 200
        page2_data = page2_response.json()
        
        # Ensure different results
        if page1_data['prompts'] and page2_data['prompts']:
            assert page1_data['prompts'][0]['id'] != page2_data['prompts'][0]['id']
    
    @pytest.mark.integration
    def test_sorting_endpoint(self, test_client, auth_headers, populated_prompts_db):
        """Test sorting in list endpoint."""
        # Sort by name ascending
        asc_response = test_client.get(
            "/api/v1/prompts/list",
            params={'sort_by': 'name', 'sort_order': 'asc'},
            headers=auth_headers
        )
        
        assert asc_response.status_code == 200
        asc_data = asc_response.json()
        
        # Sort by name descending
        desc_response = test_client.get(
            "/api/v1/prompts/list",
            params={'sort_by': 'name', 'sort_order': 'desc'},
            headers=auth_headers
        )
        
        assert desc_response.status_code == 200
        desc_data = desc_response.json()
        
        # Verify different order
        if len(asc_data['prompts']) > 1 and len(desc_data['prompts']) > 1:
            assert asc_data['prompts'][0]['id'] != desc_data['prompts'][0]['id']